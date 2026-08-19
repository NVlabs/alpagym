# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict lane-local humanoid transition joins."""

import pytest
import torch

from alpagym_runtime.alpasim.humanoid_replay import (
    attach_humanoid_transitions,
)
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


def _motion_output(
    *,
    duration: int,
    terminated: bool = True,
    outer_truncated: bool = False,
    active_digest: str | None = None,
    applied_digest: str | None = None,
    root_offset: float = 0.125,
    frame_count: int = 70,
    step: int = 0,
    reward_total: float = 1.0,
) -> PolicyOutput:
    digest = "a" * 64
    rewards = [reward_total / duration] * duration
    controller_step_start = step * 25
    ticks = [
        {
            "control_tick_offset": index + 1,
            "reference_action_index": index,
            "timestamp_us": (controller_step_start + index + 1) * 20_000,
            "active_reference_id": 7,
            "active_reference_sha256": active_digest or digest,
            "applied_reference_sha256": applied_digest or digest,
            "root_z_alignment_offset_m": root_offset,
            "reward": rewards[index],
            "terminated": terminated and index == duration - 1,
            "truncated": False,
            "control_episode_step": controller_step_start + index + 1,
        }
        for index in range(duration)
    ]
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="humanoid.motion.test.v1",
        payload_schema_version=1,
        model_family="test",
        action_selection=ActionSelection(0, 0),
        old_logprob=torch.tensor(0.0),
        payload={
            "reference_id": 7,
            "source_decision_id": step,
            "reference_sha256": digest,
            "root_z_alignment_offset_m": 0.125,
            "feedback_trace": {
                "env_id": 0,
                "source_decision_id": step,
                "ticks": ticks,
            },
            **({"outer_truncated": True} if outer_truncated else {}),
        },
    )
    return PolicyOutput(
        chosen_xyz=torch.zeros((frame_count, 3)),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * frame_count),
        chosen_dt_us=torch.arange(frame_count, dtype=torch.int64) * 20_000,
        replay_data=replay,
        model_extra={
            "humanoid_env_id": 0,
            "humanoid_episode_id": 4,
            "humanoid_step_index": step,
            "humanoid_decision_id": step,
            "humanoid_timestamp_us": controller_step_start * 20_000,
            "humanoid_value": 0.5,
        },
    )


def _motion_metrics(
    *,
    timestamp_us: int,
    terminated: bool,
    truncated: bool,
) -> dict[str, dict[str, object]]:
    return {
        "humanoid_reward_env0": {
            "timestamps_us": [timestamp_us],
            "values": [1.0],
            "valid": [True],
        },
        "humanoid_terminated_env0": {
            "timestamps_us": [timestamp_us],
            "values": [float(terminated)],
            "valid": [True],
        },
        "humanoid_truncated_env0": {
            "timestamps_us": [timestamp_us],
            "values": [float(truncated)],
            "valid": [True],
        },
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


@pytest.mark.parametrize("duration", (1, 5, 25))
def test_motion_reference_receipt_emits_exact_smdp_prefix(duration: int) -> None:
    patched = attach_humanoid_transitions(
        (_motion_output(duration=duration),),
        _motion_metrics(
            timestamp_us=duration * 20_000,
            terminated=True,
            truncated=False,
        ),
        {"humanoid_episode_length_env0": 1.0},
        behavior_policy_version=9,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[0].replay_data.payload["transition"]
    assert transition["duration_ticks"] == duration
    assert transition["primitive_reward_mask"] == [
        index < duration for index in range(25)
    ]
    assert sum(transition["primitive_rewards"]) == pytest.approx(1.0)
    assert transition["terminated"] is True
    assert transition["bootstrap_value"] == 0.0


def test_motion_reference_replay_rejects_non_k25_contract() -> None:
    with pytest.raises(ValueError, match="must select K=25"):
        attach_humanoid_transitions(
            (_motion_output(duration=1),),
            _motion_metrics(
                timestamp_us=20_000,
                terminated=True,
                truncated=False,
            ),
            {"humanoid_episode_length_env0": 1.0},
            behavior_policy_version=9,
            final_bootstrap_values={},
            control_timestep_us=20_000,
            expected_num_envs=1,
            max_transition_rows=30,
        )


def test_motion_reference_outer_horizon_truncates_after_full_controller_prefix() -> (
    None
):
    dense = _motion_metrics(
        timestamp_us=500_000,
        terminated=False,
        truncated=True,
    )
    dense["humanoid_final_bootstrap_value_env0"] = {
        "timestamps_us": [500_000],
        "values": [2.0],
        "valid": [True],
    }
    patched = attach_humanoid_transitions(
        (
            _motion_output(
                duration=25,
                terminated=False,
                outer_truncated=True,
            ),
        ),
        dense,
        {
            "humanoid_episode_length_env0": 1.0,
            "humanoid_final_bootstrap_value_env0": 2.0,
        },
        behavior_policy_version=9,
        final_bootstrap_values={0: 2.0},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[0].replay_data.payload["transition"]
    assert transition["terminated"] is False
    assert transition["truncated"] is True
    assert transition["duration_ticks"] == 25
    assert transition["bootstrap_value"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("output", "match"),
    (
        (
            _motion_output(duration=1, active_digest="b" * 64),
            "active reference digest",
        ),
        (
            _motion_output(duration=1, root_offset=0.5),
            "root Z alignment",
        ),
    ),
)
def test_motion_reference_receipt_rejects_mixed_identity(
    output: PolicyOutput,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        attach_humanoid_transitions(
            (output,),
            _motion_metrics(
                timestamp_us=20_000,
                terminated=True,
                truncated=False,
            ),
            {"humanoid_episode_length_env0": 1.0},
            behavior_policy_version=9,
            final_bootstrap_values={},
            control_timestep_us=500_000,
            expected_num_envs=1,
            max_transition_rows=30,
        )


def test_h70_replay_rejects_rewritten_applied_reference_hash() -> None:
    with pytest.raises(ValueError, match="H70 feedback applied reference digest"):
        attach_humanoid_transitions(
            (
                _motion_output(
                    duration=1,
                    applied_digest="b" * 64,
                ),
            ),
            _motion_metrics(
                timestamp_us=20_000,
                terminated=True,
                truncated=False,
            ),
            {"humanoid_episode_length_env0": 1.0},
            behavior_policy_version=9,
            final_bootstrap_values={},
            control_timestep_us=500_000,
            expected_num_envs=1,
            max_transition_rows=30,
        )
