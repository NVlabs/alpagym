# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replayable Flow-SDE sampling for Wenhao's non-uniform Psi0 schedule.

The transition distribution follows RLinf's pinned piRL OpenPI sampler, while
the sigma/model-timestep grid remains the exact grid materialized by Wenhao's
``FlowMatchEulerDiscreteScheduler``. The actor callback owns Psi0 conditioning;
this module owns only stochastic integration and replay density.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import torch

from alpagym_runtime.third_party.rlinf.flow_sampler import gaussian_logprob

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

WENHAO_FLOW_NOISE_LEVEL = 0.4
WENHAO_FLOW_IGNORE_LAST = True


@dataclass(frozen=True)
class WenhaoFlowSchedule:
    """Immutable model timesteps plus their matching latent sigma boundaries."""

    model_timesteps: torch.Tensor
    sigmas: torch.Tensor

    def __post_init__(self) -> None:
        model_timesteps = torch.as_tensor(
            self.model_timesteps, dtype=torch.float32
        ).detach()
        sigmas = torch.as_tensor(self.sigmas, dtype=torch.float32).detach()
        if model_timesteps.ndim != 1 or sigmas.ndim != 1:
            raise ValueError("Wenhao flow schedule tensors must be one-dimensional")
        if sigmas.numel() != model_timesteps.numel() + 1:
            raise ValueError("Wenhao flow schedule requires N timesteps and N+1 sigmas")
        if model_timesteps.numel() < 2:
            raise ValueError("Wenhao flow schedule requires at least two denoise steps")
        if (
            not torch.isfinite(model_timesteps).all()
            or not torch.isfinite(sigmas).all()
        ):
            raise ValueError("Wenhao flow schedule contains non-finite values")
        if not torch.all(sigmas[:-1] > sigmas[1:]):
            raise ValueError("Wenhao flow sigmas must be strictly decreasing")
        if not torch.isclose(sigmas[0], torch.tensor(1.0), atol=1.0e-6):
            raise ValueError("Wenhao flow schedule must start at sigma=1")
        if not torch.isclose(sigmas[-1], torch.tensor(0.0), atol=1.0e-7):
            raise ValueError("Wenhao flow schedule must end at sigma=0")
        object.__setattr__(self, "model_timesteps", model_timesteps.contiguous())
        object.__setattr__(self, "sigmas", sigmas.contiguous())

    @property
    def num_steps(self) -> int:
        """Return the number of velocity evaluations in this schedule."""
        return int(self.model_timesteps.numel())

    @property
    def sha256(self) -> str:
        """Return a stable digest of the exact float32 schedule."""
        digest = hashlib.sha256()
        for tensor in (self.model_timesteps, self.sigmas):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def to(self, device: torch.device | str) -> WenhaoFlowSchedule:
        """Copy the immutable schedule to ``device``."""
        return WenhaoFlowSchedule(
            model_timesteps=self.model_timesteps.to(device=device),
            sigmas=self.sigmas.to(device=device),
        )


@dataclass(frozen=True)
class WenhaoFlowTrace:
    """Exact latent chain and factorized full-chunk behavior density."""

    chain: torch.Tensor
    denoise_indices: torch.Tensor
    old_element_logprobs: torch.Tensor
    schedule_sha256: str

    def __post_init__(self) -> None:
        if self.chain.ndim != 4:
            raise ValueError("Wenhao flow chain must have shape [B, N+1, H, A]")
        batch, chain_steps, horizon, action_dim = self.chain.shape
        if tuple(self.denoise_indices.shape) != (batch,):
            raise ValueError("Wenhao denoise_indices must have shape [B]")
        if tuple(self.old_element_logprobs.shape) != (batch, horizon, action_dim):
            raise ValueError(
                "Wenhao old element logprobs must match the sampled [B,H,A] chunk"
            )
        if (
            not torch.isfinite(self.chain).all()
            or not torch.isfinite(self.old_element_logprobs).all()
        ):
            raise ValueError("Wenhao flow trace contains non-finite values")
        if not torch.all(
            (self.denoise_indices >= 0) & (self.denoise_indices < chain_steps - 2)
        ):
            raise ValueError(
                "Wenhao Flow-PPO excludes the final low-variance denoise transition"
            )
        if len(self.schedule_sha256) != 64:
            raise ValueError("Wenhao flow trace schedule digest must be SHA-256")


def _schedule_mean_std(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    noise_level: float,
    stochastic: torch.Tensor,
    first_non_unit_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalize RLinf's Flow-SDE transition to an explicit sigma grid."""
    if x_t.shape != velocity.shape:
        raise ValueError("Wenhao flow latent and velocity shapes differ")
    if noise_level <= 0.0:
        raise ValueError("Wenhao Flow-SDE noise_level must be positive")
    sigma = sigma.to(device=x_t.device, dtype=x_t.dtype).reshape(-1, 1, 1)
    sigma_next = sigma_next.to(device=x_t.device, dtype=x_t.dtype).reshape(-1, 1, 1)
    delta = sigma - sigma_next
    if not torch.all(delta > 0):
        raise ValueError("Wenhao Flow-SDE transition requires decreasing sigmas")
    x0_pred = x_t - velocity * sigma
    x1_pred = x_t + velocity * (1.0 - sigma)
    ode_mean = x0_pred * (1.0 - sigma_next) + x1_pred * sigma_next

    denom_sigma = torch.where(
        sigma == 1,
        first_non_unit_sigma.to(device=x_t.device, dtype=x_t.dtype),
        sigma,
    )
    diffusion = noise_level * torch.sqrt(sigma / (1.0 - denom_sigma))
    sde_mean = x0_pred * (1.0 - sigma_next) + x1_pred * (
        sigma_next - diffusion.square() * delta / (2.0 * sigma)
    )
    sde_std = torch.sqrt(delta) * diffusion
    stochastic = stochastic.to(device=x_t.device, dtype=torch.bool).reshape(-1, 1, 1)
    mean = torch.where(stochastic, sde_mean, ode_mean)
    std = torch.where(stochastic, sde_std.expand_as(x_t), torch.zeros_like(x_t))
    return mean, std


def _validated_prefix(
    initial_noise: torch.Tensor,
    prefix_actions: torch.Tensor | None,
    prefix_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    batch, horizon, action_dim = initial_noise.shape
    if prefix_actions is None and prefix_mask is None:
        return None, torch.zeros(
            (batch, horizon), dtype=torch.bool, device=initial_noise.device
        )
    if prefix_actions is None or prefix_mask is None:
        raise ValueError("Wenhao RTC prefix actions and mask must be provided together")
    actions = torch.as_tensor(
        prefix_actions, device=initial_noise.device, dtype=initial_noise.dtype
    )
    mask = torch.as_tensor(prefix_mask, device=initial_noise.device, dtype=torch.bool)
    if tuple(actions.shape) != (batch, horizon, action_dim):
        raise ValueError("Wenhao RTC prefix actions must match the flow latent")
    if tuple(mask.shape) != (batch, horizon):
        raise ValueError("Wenhao RTC prefix mask must have shape [B, H]")
    if not torch.isfinite(actions).all():
        raise ValueError("Wenhao RTC prefix actions contain non-finite values")
    return actions, mask


def _actor_timestep(
    model_timestep: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Use scalar timesteps for plain flow and row timesteps for RTC."""
    if not bool(prefix_mask.any()):
        return model_timestep
    return torch.where(
        prefix_mask,
        torch.zeros_like(prefix_mask, dtype=model_timestep.dtype),
        model_timestep[:, None].expand_as(prefix_mask),
    )


def sample_flow_sde(
    velocity_fn: VelocityFn,
    initial_noise: torch.Tensor,
    schedule: WenhaoFlowSchedule,
    *,
    noise_level: float,
    denoise_indices: torch.Tensor | None = None,
    prefix_actions: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, WenhaoFlowTrace]:
    """Sample one SDE transition per item and retain the exact replay chain."""
    if initial_noise.ndim != 3 or not torch.isfinite(initial_noise).all():
        raise ValueError("Wenhao initial_noise must be finite [B, H, A]")
    batch = int(initial_noise.shape[0])
    schedule = schedule.to(initial_noise.device)
    if denoise_indices is None:
        chosen_index = torch.randint(
            0,
            schedule.num_steps - 1,
            (1,),
            device=initial_noise.device,
            generator=generator,
        )
        denoise_indices = chosen_index.expand(batch).clone()
    else:
        denoise_indices = torch.as_tensor(
            denoise_indices, device=initial_noise.device, dtype=torch.int64
        )
    if tuple(denoise_indices.shape) != (batch,) or not torch.all(
        (denoise_indices >= 0) & (denoise_indices < schedule.num_steps - 1)
    ):
        raise ValueError(
            "Wenhao Flow-PPO denoise_indices must be [B] and exclude the final "
            "low-variance transition"
        )
    if batch > 1 and not bool((denoise_indices == denoise_indices[0]).all()):
        raise ValueError(
            "Wenhao RLinf sampling requires one shared denoise index per call"
        )
    prefix_actions, prefix_mask = _validated_prefix(
        initial_noise, prefix_actions, prefix_mask
    )

    x_t = initial_noise.to(dtype=torch.float32)
    if prefix_actions is not None:
        x_t = torch.where(prefix_mask[..., None], prefix_actions, x_t)
    chains = [x_t]
    selected_element_logprob = torch.zeros_like(x_t)
    first_non_unit = schedule.sigmas[1].reshape(1, 1, 1)
    for step_index in range(schedule.num_steps):
        if prefix_actions is not None:
            x_t = torch.where(prefix_mask[..., None], prefix_actions, x_t)
        model_timestep = schedule.model_timesteps[step_index].expand(batch)
        actor_timestep = _actor_timestep(model_timestep, prefix_mask)
        velocity = velocity_fn(x_t, actor_timestep).to(dtype=torch.float32)
        stochastic = denoise_indices == step_index
        mean, std = _schedule_mean_std(
            x_t,
            velocity,
            schedule.sigmas[step_index].expand(batch),
            schedule.sigmas[step_index + 1].expand(batch),
            noise_level=noise_level,
            stochastic=stochastic,
            first_non_unit_sigma=first_non_unit,
        )
        step_noise = torch.randn(
            x_t.shape,
            dtype=x_t.dtype,
            device=x_t.device,
            generator=generator,
        )
        x_next = mean + step_noise * std
        if prefix_actions is not None:
            mean = torch.where(prefix_mask[..., None], prefix_actions, mean)
            std = torch.where(prefix_mask[..., None], torch.zeros_like(std), std)
            x_next = torch.where(prefix_mask[..., None], prefix_actions, x_next)
        # RLinf's Flow-PPO action is the complete sampled trajectory chunk.
        # Gaussian independence is used only to audit the complete joint
        # density. The model sums all 30x38 factors before PPO forms its one
        # chunk ratio; these elements are never independently clipped.
        element_logprob = gaussian_logprob(x_next, mean, std)
        selected_element_logprob = selected_element_logprob + torch.where(
            stochastic[:, None, None],
            element_logprob,
            torch.zeros_like(element_logprob),
        )
        x_t = x_next
        chains.append(x_t)

    trace = WenhaoFlowTrace(
        chain=torch.stack(chains, dim=1).contiguous(),
        denoise_indices=denoise_indices.contiguous(),
        old_element_logprobs=selected_element_logprob.contiguous(),
        schedule_sha256=schedule.sha256,
    )
    return x_t, trace


def sample_flow_ode(
    velocity_fn: VelocityFn,
    initial_noise: torch.Tensor,
    schedule: WenhaoFlowSchedule,
    *,
    prefix_actions: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Wenhao's deterministic Euler path on the exact diffusers grid."""
    if initial_noise.ndim != 3 or not torch.isfinite(initial_noise).all():
        raise ValueError("Wenhao initial_noise must be finite [B, H, A]")
    batch = int(initial_noise.shape[0])
    schedule = schedule.to(initial_noise.device)
    prefix_actions, prefix_mask = _validated_prefix(
        initial_noise, prefix_actions, prefix_mask
    )
    x_t = initial_noise.to(dtype=torch.float32)
    for step_index in range(schedule.num_steps):
        if prefix_actions is not None:
            x_t = torch.where(prefix_mask[..., None], prefix_actions, x_t)
        model_timestep = schedule.model_timesteps[step_index].expand(batch)
        actor_timestep = _actor_timestep(model_timestep, prefix_mask)
        velocity = velocity_fn(x_t, actor_timestep).to(dtype=torch.float32)
        delta = schedule.sigmas[step_index] - schedule.sigmas[step_index + 1]
        x_t = x_t - delta.to(device=x_t.device, dtype=x_t.dtype) * velocity
        if prefix_actions is not None:
            x_t = torch.where(prefix_mask[..., None], prefix_actions, x_t)
    return x_t


def replay_flow_logprob(
    velocity_fn: VelocityFn,
    trace: WenhaoFlowTrace,
    schedule: WenhaoFlowSchedule,
    *,
    noise_level: float,
    prefix_actions: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute the selected Flow-SDE transition density with gradients."""
    if trace.schedule_sha256 != schedule.sha256:
        raise ValueError("Wenhao replay schedule identity changed")
    return replay_flow_transition_logprob(
        velocity_fn,
        trace.chain,
        trace.denoise_indices,
        schedule,
        noise_level=noise_level,
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
    )


def replay_flow_transition_logprob(
    velocity_fn: VelocityFn,
    latent_chain: torch.Tensor,
    denoise_indices: torch.Tensor,
    schedule: WenhaoFlowSchedule,
    *,
    noise_level: float,
    prefix_actions: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score one recorded stochastic transition from each latent chain.

    Args:
        velocity_fn: Psi action-head callback. It receives the selected
            pre-clip latent and the matching model timestep.
        latent_chain: Exact pre-clip normalized chain with shape
            ``[B, N+1, H, A]``.
        denoise_indices: The stochastic denoise step selected for each batch
            item, with shape ``[B]``.
        schedule: The exact sigma and model-timestep grid used for rollout.
        noise_level: Flow-SDE diffusion scale used for rollout.
        prefix_actions: Optional deterministic RTC prefix with shape
            ``[B, H, A]``.
        prefix_mask: Optional RTC row mask with shape ``[B, H]``.

    Returns:
        The selected transition's factorized conditional log-probabilities
        with shape ``[B,30,38]``. Their complete sum is the joint chunk
        log-probability used by PPO.
    """
    if latent_chain.ndim != 4 or not torch.isfinite(latent_chain).all():
        raise ValueError("Wenhao replay latent_chain must be finite [B, N+1, H, A]")
    chain = latent_chain
    batch, chain_steps, _horizon, _action_dim = chain.shape
    schedule = schedule.to(chain.device)
    if chain_steps != schedule.num_steps + 1:
        raise ValueError("Wenhao replay chain length does not match schedule")
    denoise_indices = torch.as_tensor(
        denoise_indices, device=chain.device, dtype=torch.int64
    )
    if tuple(denoise_indices.shape) != (batch,) or not torch.all(
        (denoise_indices >= 0) & (denoise_indices < schedule.num_steps - 1)
    ):
        raise ValueError(
            "Wenhao Flow-PPO replay excludes the final low-variance denoise transition"
        )
    prefix_actions, prefix_mask = _validated_prefix(
        chain[:, 0], prefix_actions, prefix_mask
    )
    if prefix_actions is not None and bool(prefix_mask.any()):
        expected_prefix = prefix_actions[:, None].expand(-1, chain_steps, -1, -1)
        expanded_mask = prefix_mask[:, None, :, None].expand_as(chain)
        if not torch.equal(chain[expanded_mask], expected_prefix[expanded_mask]):
            raise ValueError(
                "Wenhao replay chain does not preserve its deterministic RTC prefix"
            )
    row_index = torch.arange(batch, device=chain.device)
    x_t = chain[row_index, denoise_indices].to(dtype=torch.float32)
    x_next = chain[row_index, denoise_indices + 1].to(dtype=torch.float32)
    if prefix_actions is not None:
        x_t = torch.where(prefix_mask[..., None], prefix_actions, x_t)
    model_timestep = schedule.model_timesteps[denoise_indices]
    actor_timestep = _actor_timestep(model_timestep, prefix_mask)
    velocity = velocity_fn(x_t, actor_timestep).to(dtype=torch.float32)
    mean, std = _schedule_mean_std(
        x_t,
        velocity,
        schedule.sigmas[denoise_indices],
        schedule.sigmas[denoise_indices + 1],
        noise_level=noise_level,
        stochastic=torch.ones(batch, dtype=torch.bool, device=chain.device),
        first_non_unit_sigma=schedule.sigmas[1].reshape(1, 1, 1),
    )
    if prefix_actions is not None:
        mean = torch.where(prefix_mask[..., None], prefix_actions, mean)
        std = torch.where(prefix_mask[..., None], torch.zeros_like(std), std)
        x_next = torch.where(prefix_mask[..., None], prefix_actions, x_next)
    return gaussian_logprob(x_next, mean, std)
