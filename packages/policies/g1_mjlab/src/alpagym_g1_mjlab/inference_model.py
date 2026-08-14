# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout-side inference adapter for the G1 mjlab policy bundle."""

from __future__ import annotations

from typing import Any

import torch

from alpagym_g1_mjlab.model import G1MjlabActorCriticModel


class G1MjlabInferenceModel:
    """Small adapter exposing the AlpaGym ``InferenceModel`` protocol."""

    def __init__(self, model: G1MjlabActorCriticModel) -> None:
        self.model = model

    def sample_trajectories_from_data(
        self,
        model_input: Any,
        sampling: Any,
        return_trace_for_rl: bool = False,
    ) -> Any:
        del model_input, sampling, return_trace_for_rl
        raise NotImplementedError(
            "G1 mjlab rollouts use HumanoidPolicyService callbacks, not "
            "the AV trajectory InferenceEngine input dialect."
        )

    def build_policy_replay_data(
        self,
        model_input: Any,
        model_output: Any,
        action_selection: Any,
    ) -> Any:
        del model_input, model_output, action_selection
        raise NotImplementedError(
            "G1 mjlab replay data is produced directly by the humanoid policy adapter."
        )

    def get_model(self) -> torch.nn.Module:
        return self.model

    def set_model(self, model: torch.nn.Module) -> None:
        if not isinstance(model, G1MjlabActorCriticModel):
            raise TypeError(f"expected G1MjlabActorCriticModel, got {type(model).__name__}")
        self.model = model
