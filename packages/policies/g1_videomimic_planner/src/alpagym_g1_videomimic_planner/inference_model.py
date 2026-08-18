# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout-side inference handle for the VideoMimic planner."""

from __future__ import annotations

from typing import Any

import torch

from alpagym_g1_videomimic_planner.model import (
    G1VideoMimicPlannerActorCriticModel,
)


class G1VideoMimicPlannerInferenceModel:
    """Expose the mutable model handle consumed by HumanoidPolicyService."""

    def __init__(self, model: G1VideoMimicPlannerActorCriticModel) -> None:
        self.model = model

    def sample_trajectories_from_data(
        self,
        model_input: Any,
        sampling: Any,
        return_trace_for_rl: bool = False,
    ) -> Any:
        del model_input, sampling, return_trace_for_rl
        raise NotImplementedError(
            "VideoMimic planner rollouts use HumanoidPolicyService and a shadow "
            "simulator, not the AV trajectory input dialect"
        )

    def build_policy_replay_data(
        self,
        model_input: Any,
        model_output: Any,
        action_selection: Any,
    ) -> Any:
        del model_input, model_output, action_selection
        raise NotImplementedError(
            "VideoMimic planner replay is emitted by its humanoid policy adapter"
        )

    def get_model(self) -> torch.nn.Module:
        return self.model

    def set_model(self, model: torch.nn.Module) -> None:
        if not isinstance(model, G1VideoMimicPlannerActorCriticModel):
            raise TypeError(
                "expected G1VideoMimicPlannerActorCriticModel, got "
                f"{type(model).__name__}"
            )
        self.model = model
