# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict lane-local humanoid transition joins."""

import pytest
import torch

from alpagym_runtime.alpasim.humanoid_replay import attach_humanoid_transitions
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.types import PolicyOutput


def _output(step: int, value: float) -> PolicyOutput:
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="humanoid.test.v1",
        payload_schema_version=1,
        model_family="test",
        action_selection=ActionSelection(0, 0),
        old_logprob=torch.tensor(0.0),
        payload={"observation": {}, "action": torch.zeros(1)},
    )
    return PolicyOutput(
        chosen_xyz=torch.zeros((1, 1)),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        chosen_dt_us=torch.zeros(1, dtype=torch.int64),
        replay_data=replay,
        model_extra={
            "humanoid_env_id": 0,
            "humanoid_episode_id": 4,
            "humanoid_step_index": step,
            "humanoid_timestamp_us": step * 20_000,
            "humanoid_value": value,
        },
    )


def _metric(name: str, values: list[float]) -> dict[str, object]:
    return {
        "name": name,
        "timestamps_us": [(index + 1) * 20_000 for index in range(len(values))],
        "values": values,
        "valid": [True] * len(values),
    }


def test_truncated_episode_uses_next_values_and_real_final_bootstrap() -> None:
    outputs = (_output(0, 1.0), _output(1, 2.0))
    dense = {
        "humanoid_reward_env0": _metric("reward", [0.5, 1.5]),
        "humanoid_terminated_env0": _metric("terminated", [0.0, 0.0]),
        "humanoid_truncated_env0": _metric("truncated", [0.0, 1.0]),
        "humanoid_final_bootstrap_value_env0": {
            "name": "humanoid_final_bootstrap_value_env0",
            "timestamps_us": [40_000],
            "values": [3.0],
            "valid": [True],
        },
    }
    patched = attach_humanoid_transitions(
        outputs,
        dense,
        {
            "humanoid_episode_length_env0": 2.0,
            "humanoid_final_bootstrap_value_env0": 3.0,
        },
        behavior_policy_version=7,
        final_bootstrap_values={0: 3.0},
        control_timestep_us=20_000,
        expected_num_envs=1,
        max_transition_rows=750,
    )
    transitions = [output.replay_data.payload["transition"] for output in patched]
    assert transitions[0]["bootstrap_value"] == pytest.approx(2.0)
    assert transitions[1]["bootstrap_value"] == pytest.approx(3.0)
    assert {transition["behavior_policy_version"] for transition in transitions} == {7}


def test_truncated_episode_requires_runtime_final_bootstrap_echo() -> None:
    with pytest.raises(ValueError, match="missing final bootstrap metric"):
        attach_humanoid_transitions(
            (_output(0, 1.0),),
            {
                "humanoid_reward_env0": _metric("reward", [1.0]),
                "humanoid_terminated_env0": _metric("terminated", [0.0]),
                "humanoid_truncated_env0": _metric("truncated", [1.0]),
            },
            {"humanoid_episode_length_env0": 1.0},
            behavior_policy_version=1,
            final_bootstrap_values={0: 2.0},
            control_timestep_us=20_000,
            expected_num_envs=1,
            max_transition_rows=750,
        )


def test_truncated_episode_rejects_bootstrap_value_mismatch() -> None:
    dense = {
        "humanoid_reward_env0": _metric("reward", [1.0]),
        "humanoid_terminated_env0": _metric("terminated", [0.0]),
        "humanoid_truncated_env0": _metric("truncated", [1.0]),
        "humanoid_final_bootstrap_value_env0": _metric("final", [3.0]),
    }
    with pytest.raises(ValueError, match="does not match the policy response"):
        attach_humanoid_transitions(
            (_output(0, 1.0),),
            dense,
            {
                "humanoid_episode_length_env0": 1.0,
                "humanoid_final_bootstrap_value_env0": 3.0,
            },
            behavior_policy_version=1,
            final_bootstrap_values={0: 2.0},
            control_timestep_us=20_000,
            expected_num_envs=1,
            max_transition_rows=750,
        )


@pytest.mark.parametrize(
    ("metric", "aggregate"),
    [
        (
            {
                "timestamps_us": [20_000],
                "values": [float("nan")],
                "valid": [True],
            },
            2.0,
        ),
        (
            {"timestamps_us": [20_000], "values": [2.0], "valid": [False]},
            2.0,
        ),
        (
            {"timestamps_us": [19_999], "values": [2.0], "valid": [True]},
            2.0,
        ),
        (
            {"timestamps_us": [20_000], "values": [2.0], "valid": [True]},
            2.5,
        ),
    ],
)
def test_truncated_episode_rejects_invalid_runtime_bootstrap_echo(
    metric: dict[str, object],
    aggregate: float,
) -> None:
    dense = {
        "humanoid_reward_env0": _metric("reward", [1.0]),
        "humanoid_terminated_env0": _metric("terminated", [0.0]),
        "humanoid_truncated_env0": _metric("truncated", [1.0]),
        "humanoid_final_bootstrap_value_env0": metric,
    }
    with pytest.raises(ValueError, match="final bootstrap"):
        attach_humanoid_transitions(
            (_output(0, 1.0),),
            dense,
            {
                "humanoid_episode_length_env0": 1.0,
                "humanoid_final_bootstrap_value_env0": aggregate,
            },
            behavior_policy_version=1,
            final_bootstrap_values={0: 2.0},
            control_timestep_us=20_000,
            expected_num_envs=1,
            max_transition_rows=750,
        )


def test_missing_reward_metric_fails_instead_of_zero_filling() -> None:
    with pytest.raises(ValueError, match="missing transition metrics"):
        attach_humanoid_transitions(
            (_output(0, 1.0),),
            {
                "humanoid_terminated_env0": _metric("terminated", [1.0]),
                "humanoid_truncated_env0": _metric("truncated", [0.0]),
            },
            {"humanoid_episode_length_env0": 1.0},
            behavior_policy_version=0,
            final_bootstrap_values={},
            control_timestep_us=20_000,
            expected_num_envs=1,
            max_transition_rows=750,
        )
