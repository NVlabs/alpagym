# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence actor-critic used to optimize VideoMimic shadow rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from alpagym_g1_mjlab.model import (
    DEFAULT_ACTION_DIM,
    OBS_DIMS,
    OBS_KEYS,
    G1MjlabActorCriticModel,
    G1MjlabConfig,
)
from cosmos_rl.policy.model.base import IdentityWeightMapper, ModelRegistry
from transformers import AutoConfig

MODEL_TYPE = "g1_videomimic_planner_actor_critic"
REPLAY_SCHEMA = "alpagym_humanoid.videomimic_planner.v1"
OBSERVATION_SCHEMA = "videomimic_motion_planner_state.v1"
ACTION_SCHEMA_H70 = "g1_motion_reference_29d_50hz_h70.v1"
REFERENCE_FRAME_COUNT = 70
SHADOW_ACTION_STEPS = REFERENCE_FRAME_COUNT - 1
REFERENCE_PERIOD_US = 20_000
MACRO_PERIOD_US = 500_000
CONTROLLER_TICKS_PER_MACRO = MACRO_PERIOD_US // REFERENCE_PERIOD_US
PLANNER_MODE_SHADOW_ROLLOUT = "shadow_rollout"
PLANNER_MODES = frozenset((PLANNER_MODE_SHADOW_ROLLOUT,))
# GRAIL samples ten future frames at offsets [0, 5, ..., 45]. H70 supplies the
# complete future window while the controller executes the first 25 ticks.
GRAIL_FUTURE_REFERENCE_OFFSET = 45
REFERENCE_JOINT_DIM = 29


@dataclass(frozen=True)
class PlannerModeContract:
    """Actor, reference, and execution timing owned by one planner mode."""

    actor_steps: int
    controller_ticks: int
    macro_period_us: int
    reference_frames: int
    action_schema: str


def planner_mode_contract(
    mode: str,
    *,
    macro_period_us: int = MACRO_PERIOD_US,
) -> PlannerModeContract:
    """Return the complete wire and replay contract for one planner mode.

    The current VideoMimic actor autoregressively generates all 69 actions in
    the H70 reference. The outer policy remains a fixed 2 Hz/K25 contract so
    GRAIL receives its 25 realized frames and 45-frame future window.
    """

    if (
        isinstance(macro_period_us, bool)
        or int(macro_period_us) != macro_period_us
        or int(macro_period_us) <= 0
    ):
        raise ValueError("planner macro_period_us must be a positive integer")
    macro_period_us = int(macro_period_us)
    controller_ticks, remainder_us = divmod(macro_period_us, REFERENCE_PERIOD_US)
    if remainder_us:
        raise ValueError(
            "planner macro_period_us must lie on the 20000us controller grid"
        )

    if mode == PLANNER_MODE_SHADOW_ROLLOUT:
        if controller_ticks != CONTROLLER_TICKS_PER_MACRO:
            raise ValueError("shadow_rollout requires the 500000us/K25 contract")
        return PlannerModeContract(
            actor_steps=SHADOW_ACTION_STEPS,
            controller_ticks=controller_ticks,
            macro_period_us=macro_period_us,
            reference_frames=REFERENCE_FRAME_COUNT,
            action_schema=ACTION_SCHEMA_H70,
        )
    raise ValueError(
        f"planner_mode must be one of {sorted(PLANNER_MODES)}, got {mode!r}"
    )


class G1VideoMimicPlannerConfig(G1MjlabConfig):
    """HF-shaped config for the fixed-horizon VideoMimic planner."""

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        shadow_action_steps: int = SHADOW_ACTION_STEPS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.shadow_action_steps = int(shadow_action_steps)
        if self.shadow_action_steps <= 0:
            raise ValueError("shadow_action_steps must be positive")
        if int(self.action_dim) != DEFAULT_ACTION_DIM:
            raise ValueError(
                f"VideoMimic planner action_dim must be {DEFAULT_ACTION_DIM}, "
                f"got {self.action_dim}"
            )


class G1VideoMimicPlannerActorCriticModel(G1MjlabActorCriticModel):
    """Score the trainable VideoMimic decisions in one planner step.

    Inputs are ``[B, H, ...]`` where ``H=69`` is fixed by the checkpoint. Every
    action is sampled from the current actor while the private MuJoCo state rolls
    forward. ``log_probs`` is the exact sum of those trainable-token
    log-probabilities. The critic estimates the macro-boundary state and consumes
    only H=0.
    """

    config: G1VideoMimicPlannerConfig

    def __init__(self, hf_config: G1VideoMimicPlannerConfig) -> None:
        super().__init__(hf_config)
        self.config = hf_config

    @staticmethod
    def supported_model_types() -> list[str]:
        return [MODEL_TYPE]

    def forward(
        self,
        *,
        actions: torch.Tensor | None = None,
        teacher_model: Any = None,
        return_log_prob: bool = True,
        **obs: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        del return_log_prob
        self._validate_shadow_observations(obs)
        mean = self._head(obs, actor=True)
        first_obs = {key: value[:, 0] for key, value in obs.items()}
        value = self._head(first_obs, actor=False).squeeze(-1)
        scored_actions = (
            mean
            if actions is None
            else actions.to(
                device=mean.device,
                dtype=mean.dtype,
            )
        )
        expected_action_shape = (*mean.shape[:-1], int(self.config.action_dim))
        if tuple(scored_actions.shape) != expected_action_shape:
            raise ValueError(
                f"planner actions must have shape {expected_action_shape}, got "
                f"{tuple(scored_actions.shape)}"
            )
        dist = self.distribution(mean)
        token_log_probs = dist.log_prob(scored_actions).sum(dim=-1)

        token_kl_div = None
        kl_div = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_mean = teacher_model._head(obs, actor=True)
                teacher_std = teacher_model.std.clamp(min=1.0e-6).expand_as(
                    teacher_mean
                )
                teacher_dist = torch.distributions.Normal(teacher_mean, teacher_std)
            token_kl_div = torch.distributions.kl_divergence(dist, teacher_dist).sum(
                dim=-1
            )
            # Mean-over-token keeps the regularizer scale invariant to H.
            kl_div = token_kl_div.mean(dim=-1)
        return {
            "token_log_probs": token_log_probs,
            "log_probs": token_log_probs.sum(dim=-1),
            "values": value,
            "token_kl_div": token_kl_div,
            "kl_div": kl_div,
        }

    def act_step(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one VideoMimic action while an external shadow sim advances."""
        return super().act(
            obs,
            deterministic=deterministic,
            generator=generator,
        )

    def forward_values(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate macro-boundary values without generating an H70 reference."""

        self._validate_shadow_observations(obs)
        # The replay packer also carries model-control kwargs such as
        # ``return_log_prob``.  Only the explicit VideoMimic observation ABI is
        # temporal and belongs in the critic input.
        first_obs = {key: obs[key][:, 0] for key in OBS_KEYS}
        return self._head(first_obs, actor=False).squeeze(-1)

    def reset_planner_critic_(self) -> None:
        """Reinitialize the macro critic and make its initial value exactly zero."""
        self.critic_terrain.reset_parameters()
        nn.init.ones_(self.critic_attention)
        if self.critic_context is not None:
            self.critic_context.reset_parameters()
        linear_layers = [layer for layer in self.critic if isinstance(layer, nn.Linear)]
        for layer in linear_layers:
            layer.reset_parameters()
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def actor_state_sha256(self) -> str:
        """Return the canonical identity of the exact PPO actor parameter group."""

        from alpagym_g1_videomimic_planner.provenance import (
            canonical_actor_state_sha256,
        )

        return canonical_actor_state_sha256(self)

    def model_weights_sha256(self) -> str:
        """Return the policy-native safetensors identity of the full model."""

        from alpagym_g1_videomimic_planner.provenance import model_weights_sha256

        return model_weights_sha256(self)

    def _validate_shadow_observations(self, obs: Mapping[str, torch.Tensor]) -> None:
        horizon = int(self.config.shadow_action_steps)
        batch_size: int | None = None
        for key in OBS_KEYS:
            if key not in obs:
                raise KeyError(f"planner observation is missing {key!r}")
            tensor = obs[key]
            if tensor.ndim != 3:
                raise ValueError(
                    f"planner observation {key!r} must be [B, H, D], got "
                    f"{tuple(tensor.shape)}"
                )
            expected_tail = (horizon, OBS_DIMS[key])
            if tuple(tensor.shape[1:]) != expected_tail:
                raise ValueError(
                    f"planner observation {key!r} must end in {expected_tail}, "
                    f"got {tuple(tensor.shape[1:])}"
                )
            if batch_size is None:
                batch_size = int(tensor.shape[0])
            elif int(tensor.shape[0]) != batch_size:
                raise ValueError("planner observations disagree on batch size")


def register_planner_model() -> None:
    """Register the planner config and model with Transformers/Cosmos."""
    try:
        AutoConfig.register(MODEL_TYPE, G1VideoMimicPlannerConfig)
    except ValueError as exc:
        if "is already used" not in str(exc) and "already exists" not in str(exc):
            raise
    if MODEL_TYPE not in ModelRegistry._MODEL_REGISTRY:
        ModelRegistry.register_model(
            G1VideoMimicPlannerActorCriticModel,
            IdentityWeightMapper,
        )
