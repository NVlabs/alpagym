# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanoidPolicyService adapter for the G1 mjlab actor-critic."""

from __future__ import annotations

from typing import Any

import torch

from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1_MJLAB_REPLAY_SCHEMA,
    OBS_KEYS,
    split_flat_observation,
    stack_observations,
)
from alpagym_runtime.alpasim.humanoid_policy_server import (
    HumanoidPolicyInput,
    HumanoidPolicyStepOutput,
)
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.replay import ActionSelection, PolicyReplayData


class G1MjlabHumanoidPolicy:
    """Per-session G1 policy used by AlpaSim humanoid rollouts."""

    def __init__(
        self,
        model: G1MjlabActorCriticModel,
        *,
        device: torch.device,
        deterministic: bool = False,
    ) -> None:
        self._model = model
        self._device = device
        self._deterministic = bool(deterministic)

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        observations = [split_flat_observation(item.observation) for item in policy_inputs]
        obs_batch = stack_observations(observations, self._device)
        self._model.eval()
        with torch.no_grad():
            actions, log_probs, values, _means = self._model.act(
                obs_batch,
                deterministic=self._deterministic,
            )
        outputs: list[HumanoidPolicyStepOutput] = []
        for row, policy_input in enumerate(policy_inputs):
            action = actions[row].detach().cpu().to(dtype=torch.float32)
            log_prob = log_probs[row].detach().cpu().reshape(())
            value = values[row].detach().cpu().reshape(())
            replay_data = _build_replay_data(
                observation=observations[row],
                action=action,
                old_logprob=log_prob,
            )
            outputs.append(
                HumanoidPolicyStepOutput(
                    env_id=policy_input.env_id,
                    action=action,
                    logprob=log_prob,
                    value=value,
                    replay_data=replay_data,
                    model_extra={"humanoid_value": float(value.item())},
                )
            )
        return tuple(outputs)

    def close(self) -> None:
        return None


def build_humanoid_policy_factory(
    run_config: Any,
    inference_engine: InferenceEngine,
):
    """Build the per-session factory consumed by ``HumanoidPolicyServer``."""
    model = inference_engine.get_model()
    if not isinstance(model, G1MjlabActorCriticModel):
        raise TypeError(f"expected G1MjlabActorCriticModel, got {type(model).__name__}")
    device = torch.device(run_config.policy.model.device)
    deterministic = bool(run_config.policy.model.bundle_config.get("deterministic", False))

    def _factory(session_uuid: str, request: Any) -> G1MjlabHumanoidPolicy:
        del session_uuid, request
        return G1MjlabHumanoidPolicy(
            model,
            device=device,
            deterministic=deterministic,
        )

    return _factory


def _build_replay_data(
    *,
    observation: dict[str, torch.Tensor],
    action: torch.Tensor,
    old_logprob: torch.Tensor,
) -> PolicyReplayData:
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema=G1_MJLAB_REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_mjlab",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=old_logprob.detach().cpu().reshape(()),
        payload={
            "observation": {
                key: observation[key].detach().cpu().to(dtype=torch.float32)
                for key in OBS_KEYS
            },
            "action": action.detach().cpu().to(dtype=torch.float32),
        },
    )
