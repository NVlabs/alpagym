# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanoidPolicyService adapter for the G1 mjlab actor-critic."""

from __future__ import annotations

from typing import Any

import torch

from alpagym_g1_mjlab.model import (
    ACTION_SCHEMA,
    G1MjlabActorCriticModel,
    G1_MJLAB_REPLAY_SCHEMA,
    JOINT_NAMES,
    OBSERVATION_SCHEMA,
    OBS_DIMS,
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
        inference_engine: InferenceEngine,
        *,
        device: torch.device,
        deterministic: bool = False,
        random_seed: int = 0,
    ) -> None:
        self._inference_engine = inference_engine
        self._device = device
        self._deterministic = bool(deterministic)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(random_seed))

    def step(
        self,
        policy_inputs: tuple[HumanoidPolicyInput, ...],
        *,
        sample_actions: bool = True,
    ) -> tuple[HumanoidPolicyStepOutput, ...]:
        observations = [split_flat_observation(item.observation) for item in policy_inputs]
        obs_batch = stack_observations(observations, self._device)
        model = self._inference_engine.get_model()
        if not isinstance(model, G1MjlabActorCriticModel):
            raise TypeError(f"expected G1MjlabActorCriticModel, got {type(model).__name__}")
        model.eval()
        with torch.no_grad():
            actions, log_probs, values, _means = model.act(
                obs_batch,
                deterministic=self._deterministic or not sample_actions,
                generator=self._generator,
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
    device = torch.device(run_config.policy.model.device)
    deterministic = bool(run_config.policy.model.bundle_config.get("deterministic", False))

    def _factory(session_uuid: str, request: Any) -> G1MjlabHumanoidPolicy:
        del session_uuid
        _validate_session_request(request)
        return G1MjlabHumanoidPolicy(
            inference_engine,
            device=device,
            deterministic=deterministic,
            random_seed=int(request.random_seed),
        )

    return _factory


def _validate_session_request(request: Any) -> None:
    """Require the simulator session ABI to match this checkpoint exactly."""
    scalar_fields = {
        "action_size": (request.action_size, len(JOINT_NAMES)),
        "observation_schema": (request.observation_schema, OBSERVATION_SCHEMA),
        "action_schema": (request.action_schema, ACTION_SCHEMA),
    }
    for field, (actual, expected) in scalar_fields.items():
        if actual != expected:
            raise ValueError(f"Humanoid session {field} must be {expected!r}, got {actual!r}")
    observation_terms = tuple(
        (str(term.name), int(term.size)) for term in request.observation_terms
    )
    expected_terms = tuple((key, OBS_DIMS[key]) for key in OBS_KEYS)
    if observation_terms != expected_terms:
        raise ValueError(
            "Humanoid session observation terms do not match the checkpoint ABI: "
            f"expected={expected_terms}, actual={observation_terms}"
        )
    joint_names = tuple(str(name) for name in request.joint_names)
    if joint_names != JOINT_NAMES:
        raise ValueError(
            "Humanoid session joint_names do not match the VideoMimic v9 wire order: "
            f"expected={JOINT_NAMES}, actual={joint_names}"
        )
    for field in ("attempt_id", "scene_id", "scenario_id"):
        if not str(getattr(request, field)):
            raise ValueError(f"Humanoid session request requires non-empty {field}")


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
