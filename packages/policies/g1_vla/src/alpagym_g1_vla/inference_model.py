# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout handle for VLA HumanoidPolicyService callbacks."""

from __future__ import annotations

from typing import Any

import torch

from alpagym_g1_vla.model import VlaPsiActorCritic


def unwrap_actor_critic(model: torch.nn.Module) -> VlaPsiActorCritic:
    """Return the exact VLA actor-critic from one supported model owner.

    Cosmos owns a ``VlaPsiPPOModel`` wrapper while lightweight rollout tests
    may own the core directly. No structural/attribute fallback is accepted,
    because silently scoring with a different module would break behavior-policy
    versioning.
    """
    if isinstance(model, VlaPsiActorCritic):
        return model
    from alpagym_g1_vla.cosmos_model import VlaPsiPPOModel

    if isinstance(model, VlaPsiPPOModel):
        return model.actor_critic
    raise TypeError(
        "expected VLA actor-critic owner "
        f"(VlaPsiActorCritic or VlaPsiPPOModel), got {type(model).__name__}"
    )


class VlaNativeInferenceModel:
    """Expose a strict mutable model handle to humanoid callbacks."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Store one validated Cosmos wrapper or actor-critic core."""
        unwrap_actor_critic(model)
        self.model = model

    def sample_trajectories_from_data(
        self,
        model_input: Any,
        sampling: Any,
        return_trace_for_rl: bool = False,
    ) -> Any:
        """Reject the unrelated AV ``ModelInput`` inference path."""
        del model_input, sampling, return_trace_for_rl
        raise NotImplementedError(
            "VLA rollouts call the leased model from HumanoidPolicyService; "
            "they do not use the AV trajectory dialect"
        )

    def build_policy_replay_data(
        self,
        model_input: Any,
        model_output: Any,
        action_selection: Any,
    ) -> Any:
        """Reject replay construction outside the native policy callback."""
        del model_input, model_output, action_selection
        raise NotImplementedError(
            "VLA replay is emitted atomically with its model action chunk"
        )

    def get_model(self) -> torch.nn.Module:
        """Return the live model owner used for leases and weight sync."""
        return self.model

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the live model only with a supported exact owner type."""
        unwrap_actor_critic(model)
        self.model = model
