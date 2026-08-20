# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainable actor-critic core around VLA's Qwen3-VL/Psi0 model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.nn as nn

from alpagym_g1_vla.critic import VlaValueModel
from alpagym_g1_vla.flow import (
    VLA_FLOW_IGNORE_LAST,
    VLA_FLOW_NOISE_LEVEL,
    VlaFlowSchedule,
    VlaFlowTrace,
    replay_flow_transition_logprob,
    sample_flow_sde,
)
from alpagym_g1_vla.normalization import (
    VlaQ99Normalizer,
    VlaWireActions,
)


class _PsiModelProtocol(Protocol):
    """Structural surface consumed from the external Psi0 model."""

    vlm_model: nn.Module
    action_header: nn.Module

    def _vlm_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None,
        visual_pool_factors: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return final VLM condition tokens."""
        ...

    def _select_vlm_planner_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the hidden tokens consumed by the action planner."""
        ...


class _PsiActionOutput(Protocol):
    """Action-head result field used by flow replay."""

    action: torch.Tensor


@dataclass(frozen=True)
class VlaPolicySample:
    """In-memory rollout sample with density, wire, replay, and value views.

    Replay serialization stores these fields as plain tensors and the schedule
    string; it does not place this policy-package dataclass on the transport.
    """

    density_latent: torch.Tensor
    clipped_normalized: torch.Tensor
    denormalized_wire: torch.Tensor
    trace: VlaFlowTrace
    values: torch.Tensor

    def __post_init__(self) -> None:
        """Require every returned view to describe the exact sampled chunk."""
        wire = VlaWireActions(
            density_latent=self.density_latent,
            clipped_normalized=self.clipped_normalized,
            denormalized=self.denormalized_wire,
        )
        del wire
        batch = int(self.density_latent.shape[0])
        if not torch.equal(self.trace.chain[:, -1], self.density_latent):
            raise ValueError("VLA sample density latent differs from trace final")
        if (
            tuple(self.values.shape) != (batch,)
            or not torch.isfinite(self.values).all()
        ):
            raise ValueError("VLA sample values must be finite [B]")

    @property
    def old_element_logprobs(self) -> torch.Tensor:
        """Return flattened density factors retained for exact-sum audit."""
        return self.trace.old_element_logprobs.flatten(start_dim=1)

    @property
    def old_log_probs(self) -> torch.Tensor:
        """Return one full-chunk joint behavior log-probability per sample."""
        return self.old_element_logprobs.sum(dim=-1)


class VlaPsiActorCritic(nn.Module):
    """Score native 30x38 Psi action chunks and estimate boundary values.

    The injected model is the real ``Psi0Model`` from ``alpa_policy_eval`` or a
    test double with the same three members: ``_vlm_hidden_states``,
    ``_select_vlm_planner_hidden``, and ``action_header``. This package does not
    import or modify ``alpa_policy_eval``.

    Psi always remains in evaluation mode so dropout cannot change replay
    densities. Evaluation mode does not disable autograd: trainable action-head
    parameters still receive PPO gradients. The v1 ownership contract freezes
    the VLM and trains only ``action_header`` plus the critic.
    """

    psi_model: nn.Module
    normalizer: VlaQ99Normalizer
    critic: VlaValueModel

    def __init__(
        self,
        *,
        psi_model: nn.Module,
        schedule: VlaFlowSchedule,
        noise_level: float,
        normalizer: VlaQ99Normalizer,
        vlm_hidden_dim: int,
        rtc_max_delay_exclusive: int,
        critic_hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        detach_vlm_from_critic: bool = True,
    ) -> None:
        """Initialize the Psi actor and its observation-only value head.

        Args:
            psi_model: Loaded VLA ``Psi0Model`` with its VLM parameters
                already frozen. Its condition path and trainable action head
                are called directly.
            schedule: Exact VLA model-timestep and sigma grid used during
                rollout.
            noise_level: Flow-SDE diffusion scale used during rollout.
            normalizer: Bundle-owned q01/q99 state and action transform.
            vlm_hidden_dim: Last-layer Qwen hidden width consumed by the critic.
            rtc_max_delay_exclusive: Exclusive hard-prefix row bound read from
                the attested checkpoint run config.
            critic_hidden_sizes: Widths of the critic MLP.
            detach_vlm_from_critic: Whether value loss is prevented from
                updating the VLM. Policy log-probability gradients are unaffected.
        """
        super().__init__()
        if float(noise_level) != VLA_FLOW_NOISE_LEVEL:
            raise ValueError(
                f"VLA v1 requires Flow-SDE noise_level={VLA_FLOW_NOISE_LEVEL}"
            )
        if (
            isinstance(rtc_max_delay_exclusive, bool)
            or not isinstance(rtc_max_delay_exclusive, int)
            or not 2 <= rtc_max_delay_exclusive <= 30
        ):
            raise ValueError(
                "VLA RTC max_delay must be an integer in [2, 30] with an "
                "exclusive upper bound"
            )
        psi = cast(_PsiModelProtocol, psi_model)
        trainable_vlm = [
            name
            for name, parameter in psi.vlm_model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable_vlm:
            raise ValueError(
                "VLA v1 requires a frozen VLM before adapter construction; "
                f"trainable VLM parameters: {trainable_vlm[:3]}"
            )
        if not any(
            parameter.requires_grad for parameter in psi.action_header.parameters()
        ):
            raise ValueError("VLA v1 action_header must have trainable parameters")

        self.psi_model = psi_model
        self.schedule = schedule
        self.noise_level = float(noise_level)
        self.ignore_last = VLA_FLOW_IGNORE_LAST
        self.rtc_max_delay_exclusive = rtc_max_delay_exclusive
        self.normalizer = normalizer
        self.critic = VlaValueModel(
            vlm_hidden_dim=vlm_hidden_dim,
            hidden_sizes=critic_hidden_sizes,
            detach_vlm=detach_vlm_from_critic,
        )
        self.psi_model.eval()

    def train(self, mode: bool = True) -> VlaPsiActorCritic:
        """Set critic mode while keeping the stochastic Psi actor in eval mode."""
        super().train(mode)
        self.psi_model.eval()
        return self

    def materialize_action_views(
        self, latent_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create clipped-normalized and denormalized wire actions.

        Args:
            latent_action: Final pre-clip flow latent ending in ``[30, 38]``.

        Returns:
            A pair ``(clipped_normalized, denormalized_wire)``. The first item
            lies in the checkpoint support ``[-1, 1]``; the second uses the
            bundle's q01/q99 statistics and is the only view sent to SONIC.
        """
        if latent_action.ndim < 2 or tuple(latent_action.shape[-2:]) != (30, 38):
            raise ValueError("VLA latent action must end in [30, 38]")
        if (
            not torch.is_floating_point(latent_action)
            or not torch.isfinite(latent_action).all()
        ):
            raise ValueError("VLA latent action must be finite floating point")
        wire = self.normalizer.to_wire(latent_action)
        return wire.clipped_normalized, wire.denormalized

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sequence_lengths: torch.Tensor,
        image_counts: torch.Tensor,
        image_offsets: torch.Tensor,
        image_patch_counts: torch.Tensor,
        patch_counts: torch.Tensor,
        patch_offsets: torch.Tensor,
        physical_states: torch.Tensor,
        rtc_prefix_normalized_actions: torch.Tensor,
        rtc_prefix_mask: torch.Tensor,
        generator: torch.Generator,
        effective_image_grid_thw: torch.Tensor | None = None,
        visual_pool_factors: torch.Tensor | None = None,
        traj2ds: torch.Tensor | None = None,
    ) -> VlaPolicySample:
        """Sample one native action chunk and retain exact PPO replay state.

        The method runs rollout under ``no_grad`` and forces Psi evaluation
        mode. It samples initial noise plus one RLinf-compatible denoise index
        shared by the whole sampling call from the caller-owned generator,
        then uses the same native condition/action-head path as
        :meth:`forward`.

        Args:
            input_ids: Right-padded Qwen prompt tokens with shape ``[B, S]``.
            attention_mask: Prefix mask matching ``sequence_lengths``.
            pixel_values: Ragged-packed Qwen image patches.
            image_grid_thw: Ragged-packed per-image grid metadata.
            sequence_lengths: Unpadded token count for each sample.
            image_counts: Number of selected BATS images per sample.
            image_offsets: Exclusive offsets into ``image_grid_thw``.
            image_patch_counts: Patch rows contributed by each image.
            patch_counts: Patch rows contributed by each sample.
            patch_offsets: Exclusive offsets into ``pixel_values``.
            physical_states: Physical 29-D joint positions; normalization is
                applied internally from the attested bundle statistics.
            rtc_prefix_normalized_actions: Deterministic normalized RTC rows.
            rtc_prefix_mask: Rows fixed by RTC, shape ``[B, 30]``.
            generator: Device-compatible generator owning all rollout noise.
            effective_image_grid_thw: Optional pooled history image grids.
            visual_pool_factors: Optional per-image history pooling factors.
            traj2ds: Optional Psi trajectory-condition tensor.

        Returns:
            Final pre-clip latent, both executable action views, the exact
            Flow-SDE trace, and one boundary value per batch item.
        """
        self.psi_model.eval()
        self._validate_rtc_inputs(
            rtc_prefix_normalized_actions, rtc_prefix_mask, batch=input_ids.shape[0]
        )
        self._validate_packed_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sequence_lengths=sequence_lengths,
            image_counts=image_counts,
            image_offsets=image_offsets,
            image_patch_counts=image_patch_counts,
            patch_counts=patch_counts,
            patch_offsets=patch_offsets,
        )
        batch = int(input_ids.shape[0])
        if physical_states.ndim < 1 or int(physical_states.shape[0]) != batch:
            raise ValueError("VLA physical_states batch differs from Qwen inputs")
        vlm_hidden, vlm_attention_mask = self._encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            effective_image_grid_thw=effective_image_grid_thw,
            visual_pool_factors=visual_pool_factors,
        )
        normalized_states = self.normalizer.normalize_state(physical_states)

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._action_velocity(
                vlm_hidden=vlm_hidden,
                vlm_attention_mask=vlm_attention_mask,
                normalized_states=normalized_states,
                latent=latent,
                timestep=timestep,
                traj2ds=traj2ds,
            )

        initial_noise = torch.randn(
            (batch, 30, 38),
            dtype=torch.float32,
            device=normalized_states.device,
            generator=generator,
        )
        density_latent, trace = sample_flow_sde(
            velocity_fn,
            initial_noise,
            self.schedule,
            noise_level=self.noise_level,
            prefix_actions=rtc_prefix_normalized_actions,
            prefix_mask=rtc_prefix_mask,
            generator=generator,
        )
        wire = self.normalizer.to_wire(density_latent)
        values = self.critic(
            vlm_hidden,
            vlm_attention_mask,
            normalized_states,
            rtc_prefix_normalized_actions,
            rtc_prefix_mask,
        )
        return VlaPolicySample(
            density_latent=density_latent,
            clipped_normalized=wire.clipped_normalized,
            denormalized_wire=wire.denormalized,
            trace=trace,
            values=values,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sequence_lengths: torch.Tensor,
        image_counts: torch.Tensor,
        image_offsets: torch.Tensor,
        image_patch_counts: torch.Tensor,
        patch_counts: torch.Tensor,
        patch_offsets: torch.Tensor,
        physical_states: torch.Tensor,
        latent_chain: torch.Tensor,
        denoise_indices: torch.Tensor,
        flow_schedule_sha256: str,
        clipped_normalized_actions: torch.Tensor,
        denormalized_wire_actions: torch.Tensor,
        rtc_prefix_normalized_actions: torch.Tensor,
        rtc_prefix_mask: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None = None,
        visual_pool_factors: torch.Tensor | None = None,
        traj2ds: torch.Tensor | None = None,
        teacher_model: Any = None,
        return_log_prob: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Replay one sampled Flow-SDE transition and score the joint chunk.

        The density is evaluated only on ``latent_chain``, before support
        clipping or q01/q99 denormalization. The two executable action views are
        required to prove that replay and wire artifacts belong to that exact
        latent trajectory. PPO receives one ratio for the complete ``30x38``
        action. Element densities leave the model only so the trainer can
        audit their exact sum; PPO still forms one ratio for the joint chunk.
        """
        del return_log_prob
        if teacher_model is not None:
            raise NotImplementedError(
                "VLA reference-model KL is not implemented for Flow-SDE replay"
            )
        if flow_schedule_sha256 != self.schedule.sha256:
            raise ValueError("VLA replay flow schedule identity changed")
        self.psi_model.eval()
        self._validate_rtc_inputs(
            rtc_prefix_normalized_actions, rtc_prefix_mask, batch=input_ids.shape[0]
        )
        self._validate_packed_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sequence_lengths=sequence_lengths,
            image_counts=image_counts,
            image_offsets=image_offsets,
            image_patch_counts=image_patch_counts,
            patch_counts=patch_counts,
            patch_offsets=patch_offsets,
        )
        self._validate_action_views(
            latent_chain=latent_chain,
            clipped_normalized_actions=clipped_normalized_actions,
            denormalized_wire_actions=denormalized_wire_actions,
        )
        vlm_hidden, vlm_attention_mask = self._encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            effective_image_grid_thw=effective_image_grid_thw,
            visual_pool_factors=visual_pool_factors,
        )
        normalized_states = self.normalizer.normalize_state(physical_states)

        def velocity_fn(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._action_velocity(
                vlm_hidden=vlm_hidden,
                vlm_attention_mask=vlm_attention_mask,
                normalized_states=normalized_states,
                latent=latent,
                timestep=timestep,
                traj2ds=traj2ds,
            )

        element_log_probs = replay_flow_transition_logprob(
            velocity_fn,
            latent_chain,
            denoise_indices,
            self.schedule,
            noise_level=self.noise_level,
            prefix_actions=rtc_prefix_normalized_actions,
            prefix_mask=rtc_prefix_mask,
        )
        expected_element_shape = (int(input_ids.shape[0]), 30, 38)
        if tuple(element_log_probs.shape) != expected_element_shape:
            raise ValueError("VLA Flow-SDE element log-probabilities must be [B,30,38]")
        element_log_probs = element_log_probs.flatten(start_dim=1)
        log_probs = element_log_probs.sum(dim=-1)
        values = self.critic(
            vlm_hidden,
            vlm_attention_mask,
            normalized_states,
            rtc_prefix_normalized_actions,
            rtc_prefix_mask,
        )
        return {
            "log_probs": log_probs,
            "element_log_probs": element_log_probs,
            "values": values,
            "kl_div": None,
        }

    def forward_values(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sequence_lengths: torch.Tensor,
        image_counts: torch.Tensor,
        image_offsets: torch.Tensor,
        image_patch_counts: torch.Tensor,
        patch_counts: torch.Tensor,
        patch_offsets: torch.Tensor,
        physical_states: torch.Tensor,
        rtc_prefix_normalized_actions: torch.Tensor,
        rtc_prefix_mask: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None = None,
        visual_pool_factors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate boundary values without invoking the Psi action head."""
        self.psi_model.eval()
        self._validate_rtc_inputs(
            rtc_prefix_normalized_actions, rtc_prefix_mask, batch=input_ids.shape[0]
        )
        self._validate_packed_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            sequence_lengths=sequence_lengths,
            image_counts=image_counts,
            image_offsets=image_offsets,
            image_patch_counts=image_patch_counts,
            patch_counts=patch_counts,
            patch_offsets=patch_offsets,
        )
        vlm_hidden, vlm_attention_mask = self._encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            effective_image_grid_thw=effective_image_grid_thw,
            visual_pool_factors=visual_pool_factors,
        )
        normalized_states = self.normalizer.normalize_state(physical_states)
        return self.critic(
            vlm_hidden,
            vlm_attention_mask,
            normalized_states,
            rtc_prefix_normalized_actions,
            rtc_prefix_mask,
        )

    def _validate_rtc_inputs(
        self,
        rtc_prefix_normalized_actions: torch.Tensor,
        rtc_prefix_mask: torch.Tensor,
        *,
        batch: int,
    ) -> None:
        """Validate a complete raw chunk plus an optional deterministic RTC prefix."""
        if tuple(rtc_prefix_normalized_actions.shape) != (batch, 30, 38):
            raise ValueError("VLA RTC action buffer must have shape [B,30,38]")
        if tuple(rtc_prefix_mask.shape) != (batch, 30):
            raise ValueError("VLA RTC mask must have shape [B,30]")
        if rtc_prefix_mask.dtype is not torch.bool:
            raise TypeError("VLA RTC mask must use bool dtype")
        if not torch.isfinite(rtc_prefix_normalized_actions).all():
            raise ValueError("VLA RTC action buffer contains non-finite values")
        prefix_lengths = rtc_prefix_mask.sum(dim=1)
        if bool((prefix_lengths >= self.rtc_max_delay_exclusive).any()):
            raise ValueError(
                "VLA RTC prefix length exceeds the attested checkpoint "
                f"exclusive max_delay={self.rtc_max_delay_exclusive}"
            )
        expected_mask = (
            torch.arange(30, device=rtc_prefix_mask.device)[None]
            < prefix_lengths[:, None]
        )
        if not torch.equal(rtc_prefix_mask, expected_mask):
            raise ValueError("VLA RTC mask must be one contiguous leading prefix")

    def clone_for_inference_lease(self) -> VlaPsiActorCritic:
        """Snapshot trainable heads while sharing the frozen 2B VLM backbone.

        Returns:
            An independent, frozen evaluation module. Its Psi action head and
            critic are deep copies, while ``psi_model.vlm_model`` is the exact
            same frozen module as the training model.

        Raises:
            RuntimeError: If any Psi parameter outside ``action_header`` is
                trainable, because sharing it would violate lease isolation.
        """
        unexpected_trainable = [
            name
            for name, parameter in self.psi_model.named_parameters()
            if parameter.requires_grad and not name.startswith("action_header.")
        ]
        if unexpected_trainable:
            raise RuntimeError(
                "VLA inference leases can share only a frozen VLM; trainable "
                f"non-action-head parameters: {unexpected_trainable[:3]}"
            )

        leased_psi = copy.copy(self.psi_model)
        leased_psi._parameters = self.psi_model._parameters.copy()
        leased_psi._buffers = self.psi_model._buffers.copy()
        leased_psi._modules = self.psi_model._modules.copy()
        psi = cast(_PsiModelProtocol, self.psi_model)
        leased_psi_view = cast(_PsiModelProtocol, leased_psi)
        leased_psi_view.action_header = copy.deepcopy(psi.action_header)
        leased_psi_view.vlm_model = psi.vlm_model

        lease = copy.copy(self)
        lease._parameters = self._parameters.copy()
        lease._buffers = {
            name: buffer.detach().clone() if buffer is not None else None
            for name, buffer in self._buffers.items()
        }
        lease._modules = self._modules.copy()
        lease.psi_model = leased_psi
        lease.critic = copy.deepcopy(self.critic)
        lease.normalizer = copy.deepcopy(self.normalizer)
        lease.requires_grad_(False)
        lease.eval()
        return lease

    def _validate_action_views(
        self,
        *,
        latent_chain: torch.Tensor,
        clipped_normalized_actions: torch.Tensor,
        denormalized_wire_actions: torch.Tensor,
    ) -> None:
        """Fail closed when replay, clipped, and wire action artifacts diverge."""
        if latent_chain.ndim != 4 or tuple(latent_chain.shape[-2:]) != (30, 38):
            raise ValueError("VLA latent_chain must have shape [B, N+1, 30, 38]")
        batch = int(latent_chain.shape[0])
        expected_shape = (batch, 30, 38)
        if tuple(clipped_normalized_actions.shape) != expected_shape:
            raise ValueError(
                "VLA clipped_normalized_actions must have shape [B, 30, 38]"
            )
        if tuple(denormalized_wire_actions.shape) != expected_shape:
            raise ValueError(
                "VLA denormalized_wire_actions must have shape [B, 30, 38]"
            )
        expected_clipped, expected_wire = self.materialize_action_views(
            latent_chain[:, -1]
        )
        clipped = clipped_normalized_actions.to(
            device=expected_clipped.device, dtype=expected_clipped.dtype
        )
        wire = denormalized_wire_actions.to(
            device=expected_wire.device, dtype=expected_wire.dtype
        )
        if not torch.isfinite(clipped).all() or not torch.isfinite(wire).all():
            raise ValueError("VLA executable action views must be finite")
        if not torch.allclose(clipped, expected_clipped, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError(
                "VLA clipped normalized action does not match final latent"
            )
        if not torch.allclose(wire, expected_wire, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError(
                "VLA denormalized wire action does not match q01/q99 statistics"
            )

    def _validate_packed_condition(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        sequence_lengths: torch.Tensor,
        image_counts: torch.Tensor,
        image_offsets: torch.Tensor,
        image_patch_counts: torch.Tensor,
        patch_counts: torch.Tensor,
        patch_offsets: torch.Tensor,
    ) -> None:
        """Validate policy-owned ragged history boundaries before calling Psi."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("VLA token inputs must have matching [B, S] shapes")
        batch, padded_length = input_ids.shape
        if pixel_values.ndim != 2:
            raise ValueError("VLA packed pixel_values must have shape [P, F]")
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError("VLA packed image_grid_thw must have shape [I, 3]")
        integer_tensors = {
            "sequence_lengths": sequence_lengths,
            "image_counts": image_counts,
            "image_offsets": image_offsets,
            "image_patch_counts": image_patch_counts,
            "patch_counts": patch_counts,
            "patch_offsets": patch_offsets,
            "image_grid_thw": image_grid_thw,
        }
        for name, tensor in integer_tensors.items():
            if tensor.dtype is torch.bool or torch.is_floating_point(tensor):
                raise TypeError(f"VLA packed {name} must use an integer dtype")
        if tuple(sequence_lengths.shape) != (batch,) or not torch.all(
            (sequence_lengths > 0) & (sequence_lengths <= padded_length)
        ):
            raise ValueError("VLA sequence_lengths must be valid [B] lengths")
        expected_attention = (
            torch.arange(padded_length, device=attention_mask.device)[None, :]
            < sequence_lengths.to(device=attention_mask.device)[:, None]
        )
        if not torch.equal(attention_mask.to(dtype=torch.bool), expected_attention):
            raise ValueError(
                "VLA attention_mask must be the prefix selected by sequence_lengths"
            )
        if tuple(image_counts.shape) != (batch,) or not torch.all(image_counts > 0):
            raise ValueError("VLA image_counts must be positive [B] counts")
        if tuple(patch_counts.shape) != (batch,) or not torch.all(patch_counts > 0):
            raise ValueError("VLA patch_counts must be positive [B] counts")
        if tuple(image_offsets.shape) != (batch + 1,) or not torch.equal(
            image_offsets[1:] - image_offsets[:-1], image_counts
        ):
            raise ValueError("VLA image_offsets do not match image_counts")
        if int(image_offsets[0].item()) != 0 or int(image_offsets[-1].item()) != int(
            image_grid_thw.shape[0]
        ):
            raise ValueError("VLA image_offsets do not cover image_grid_thw")
        if tuple(patch_offsets.shape) != (batch + 1,) or not torch.equal(
            patch_offsets[1:] - patch_offsets[:-1], patch_counts
        ):
            raise ValueError("VLA patch_offsets do not match patch_counts")
        if int(patch_offsets[0].item()) != 0 or int(patch_offsets[-1].item()) != int(
            pixel_values.shape[0]
        ):
            raise ValueError("VLA patch_offsets do not cover pixel_values")
        if tuple(image_patch_counts.shape) != (image_grid_thw.shape[0],):
            raise ValueError("VLA image_patch_counts must have one row per image")
        expected_image_patches = image_grid_thw.to(dtype=torch.int64).prod(dim=1)
        if not torch.equal(
            image_patch_counts.to(device=expected_image_patches.device),
            expected_image_patches,
        ):
            raise ValueError("VLA image_patch_counts do not match image_grid_thw")
        sample_ids = torch.repeat_interleave(
            torch.arange(batch, device=image_counts.device), image_counts
        )
        expected_patch_counts = torch.zeros_like(patch_counts).scatter_add(
            0,
            sample_ids.to(device=patch_counts.device),
            image_patch_counts.to(device=patch_counts.device, dtype=patch_counts.dtype),
        )
        if not torch.equal(expected_patch_counts, patch_counts):
            raise ValueError("VLA per-sample image and patch boundaries disagree")

    def _encode_condition(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None,
        visual_pool_factors: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Psi's native VLM condition path exactly once per forward."""
        psi = cast(_PsiModelProtocol, self.psi_model)
        hidden = psi._vlm_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            effective_image_grid_thw=effective_image_grid_thw,
            visual_pool_factors=visual_pool_factors,
        )
        hidden, planner_attention_mask = psi._select_vlm_planner_hidden(
            hidden, attention_mask
        )
        if hidden.ndim != 3 or tuple(planner_attention_mask.shape) != tuple(
            hidden.shape[:2]
        ):
            raise ValueError("Psi condition output must be [B, S, D] plus [B, S]")
        return hidden, planner_attention_mask

    def _action_velocity(
        self,
        *,
        vlm_hidden: torch.Tensor,
        vlm_attention_mask: torch.Tensor,
        normalized_states: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        traj2ds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Call the native Psi action head for one selected denoise step."""
        psi = cast(_PsiModelProtocol, self.psi_model)
        output = cast(
            _PsiActionOutput,
            psi.action_header(
                hidden_states=None,
                timestep=timestep,
                joint_attention_kwargs={
                    "action_hidden_embeds": latent,
                    "views": vlm_hidden.unsqueeze(1),
                    "obs": normalized_states,
                    "traj2ds": traj2ds,
                },
                vlm_attn_mask=vlm_attention_mask,
                return_dict=True,
            ),
        )
        velocity = output.action
        if tuple(velocity.shape) != tuple(latent.shape):
            raise ValueError("Psi action head velocity shape differs from flow latent")
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("Psi action head returned non-finite velocity")
        return velocity
