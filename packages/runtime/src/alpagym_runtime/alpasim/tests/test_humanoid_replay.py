# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict lane-local humanoid transition joins."""

from dataclasses import replace

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
    frame_count: int = 50,
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
    reward_value: float = 1.0,
) -> dict[str, dict[str, object]]:
    return {
        "humanoid_reward_env0": {
            "timestamps_us": [timestamp_us],
            "values": [reward_value],
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
    assert transition["primitive_reward_mask"] == [True] * duration
    assert transition["actor_primitive_reward_mask"] == [True] * duration
    assert len(transition["primitive_rewards"]) == duration
    assert transition["actor_primitive_rewards"] == transition["primitive_rewards"]
    assert sum(transition["primitive_rewards"]) == pytest.approx(1.0)
    assert transition["terminated"] is True
    assert transition["bootstrap_value"] == 0.0


def test_motion_reference_replay_rejects_wrong_replan_period() -> None:
    with pytest.raises(ValueError, match="25-tick native replan trigger"):
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
            "third or unknown active reference",
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


def _async_motion_outputs(
    *, predecessor_ticks: int = 12, source_ticks: int = 25
) -> tuple[PolicyOutput, PolicyOutput]:
    """Build an initial sample followed by one old-to-new async interval."""
    duration = predecessor_ticks + source_ticks
    assert duration > 0
    first = _motion_output(duration=25, terminated=False, step=0)
    second = _motion_output(duration=duration, terminated=True, step=1)
    assert second.replay_data is not None
    payload = dict(second.replay_data.payload)
    payload.update(
        reference_id=8,
        reference_sha256="b" * 64,
    )
    ticks = []
    for tick_index in range(duration):
        predecessor = tick_index < predecessor_ticks
        ticks.append(
            {
                "control_tick_offset": tick_index + 1,
                "reference_action_index": 25 + tick_index
                if predecessor
                else tick_index,
                "timestamp_us": 520_000 + tick_index * 20_000,
                "active_reference_id": 7 if predecessor else 8,
                "active_reference_sha256": "a" * 64 if predecessor else "b" * 64,
                "applied_reference_sha256": "c" * 64 if predecessor else "d" * 64,
                "root_z_alignment_offset_m": 0.125,
                "reward": 1.0 / duration,
                "terminated": tick_index == duration - 1,
                "truncated": False,
                "control_episode_step": 26 + tick_index,
            }
        )
    payload["feedback_trace"] = {
        "env_id": 0,
        "source_decision_id": 1,
        "ticks": ticks,
    }
    second = replace(
        second,
        replay_data=replace(second.replay_data, payload=payload),
    )
    return first, second


def _async_motion_metrics(*, duration: int = 37) -> dict[str, dict[str, object]]:
    """Return two row-level ScenarioEval metrics for the async fixture."""
    timestamps = [500_000, 500_000 + duration * 20_000]
    return {
        "humanoid_reward_env0": {
            "timestamps_us": timestamps,
            "values": [1.0, 1.0],
            "valid": [True, True],
        },
        "humanoid_terminated_env0": {
            "timestamps_us": timestamps,
            "values": [0.0, 1.0],
            "valid": [True, True],
        },
        "humanoid_truncated_env0": {
            "timestamps_us": timestamps,
            "values": [0.0, 0.0],
            "valid": [True, True],
        },
    }


def test_async_motion_receipt_attributes_predecessor_prefix_to_owning_sample() -> None:
    outputs = _async_motion_outputs()
    patched = attach_humanoid_transitions(
        outputs,
        _async_motion_metrics(),
        {"humanoid_episode_length_env0": 2.0},
        behavior_policy_version=9,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[1].replay_data.payload["transition"]
    assert transition["duration_ticks"] == 37
    assert transition["owning_reference_executed_ticks"] == 25
    assert transition["actor_valid"] is True
    assert len(transition["primitive_rewards"]) == 37
    assert transition["actor_primitive_rewards"] == transition["primitive_rewards"]
    assert transition["primitive_reward_mask"] == [True] * 37
    assert transition["actor_primitive_reward_mask"] == [False] * 12 + [True] * 25
    assert transition["applied_reference_sha256"] == ["c" * 64] * 12 + ["d" * 64] * 25


@pytest.mark.parametrize(
    ("predecessor_ticks", "source_ticks"),
    ((0, 25), (12, 25), (25, 0)),
)
def test_async_motion_receipt_preserves_critic_chronology_and_actor_ownership(
    predecessor_ticks: int,
    source_ticks: int,
) -> None:
    """0/12/25-tick install delays retain time but exclude old-plan actor credit."""
    duration = predecessor_ticks + source_ticks
    patched = attach_humanoid_transitions(
        _async_motion_outputs(
            predecessor_ticks=predecessor_ticks,
            source_ticks=source_ticks,
        ),
        _async_motion_metrics(duration=duration),
        {"humanoid_episode_length_env0": 2.0},
        behavior_policy_version=9,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[1].replay_data.payload["transition"]
    assert transition["duration_ticks"] == duration
    assert transition["primitive_reward_mask"] == [True] * duration
    assert transition["actor_primitive_reward_mask"] == (
        [False] * predecessor_ticks + [True] * source_ticks
    )
    assert transition["actor_primitive_rewards"] == transition["primitive_rewards"]
    assert transition["owning_reference_executed_ticks"] == source_ticks
    assert transition["actor_valid"] is (source_ticks > 0)
    assert sum(transition["primitive_rewards"]) == pytest.approx(1.0)


def test_async_motion_receipt_removes_predecessor_owned_dense_gain_from_actor() -> None:
    """Critic conservation survives a support event owned by the prior plan."""
    first, second = _async_motion_outputs(predecessor_ticks=3, source_ticks=2)
    assert second.replay_data is not None
    payload = dict(second.replay_data.payload)
    trace = dict(payload["feedback_trace"])
    ticks = [dict(tick) for tick in trace["ticks"]]
    source_tick = ticks[3]
    source_tick["metrics"] = {
        "stable_support_reward_progress": 0.3,
        "stable_support_reward_height": 0.1,
        "stable_support_reward_dense_predecessor_owned": 0.4,
    }
    trace["ticks"] = ticks
    payload["feedback_trace"] = trace
    second = replace(
        second,
        replay_data=replace(second.replay_data, payload=payload),
    )

    patched = attach_humanoid_transitions(
        (first, second),
        _async_motion_metrics(duration=5),
        {"humanoid_episode_length_env0": 2.0},
        behavior_policy_version=9,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[1].replay_data.payload["transition"]
    critic_rewards = transition["primitive_rewards"]
    actor_rewards = transition["actor_primitive_rewards"]
    assert sum(critic_rewards) == pytest.approx(1.0)
    assert actor_rewards[:3] == critic_rewards[:3]
    assert actor_rewards[3] == pytest.approx(critic_rewards[3] - 0.4)
    assert actor_rewards[4] == critic_rewards[4]


@pytest.mark.parametrize("invalid_sequence", ("reversal", "third"))
def test_async_motion_receipt_rejects_illegal_reference_sequence(
    invalid_sequence: str,
) -> None:
    first, second = _async_motion_outputs()
    assert second.replay_data is not None
    payload = dict(second.replay_data.payload)
    trace = dict(payload["feedback_trace"])
    ticks = [dict(tick) for tick in trace["ticks"]]
    if invalid_sequence == "reversal":
        ticks[-1].update(
            active_reference_id=7,
            active_reference_sha256="a" * 64,
            applied_reference_sha256="c" * 64,
            reference_action_index=49,
        )
        match = "reversed"
    else:
        ticks[0].update(
            active_reference_id=9,
            active_reference_sha256="e" * 64,
            applied_reference_sha256="f" * 64,
        )
        match = "third or unknown"
    trace["ticks"] = ticks
    payload["feedback_trace"] = trace
    second = replace(
        second,
        replay_data=replace(second.replay_data, payload=payload),
    )

    with pytest.raises(ValueError, match=match):
        attach_humanoid_transitions(
            (first, second),
            _async_motion_metrics(),
            {"humanoid_episode_length_env0": 2.0},
            behavior_policy_version=9,
            final_bootstrap_values={},
            control_timestep_us=500_000,
            expected_num_envs=1,
            max_transition_rows=30,
        )


@pytest.mark.parametrize("terminal", (True, False))
def test_async_motion_receipt_allows_predecessor_only_only_at_terminal(
    terminal: bool,
) -> None:
    first, second = _async_motion_outputs()
    assert second.replay_data is not None
    payload = dict(second.replay_data.payload)
    trace = dict(payload["feedback_trace"])
    ticks = [dict(tick) for tick in trace["ticks"]]
    for tick_index, tick in enumerate(ticks):
        tick.update(
            active_reference_id=7,
            active_reference_sha256="a" * 64,
            applied_reference_sha256="c" * 64,
            reference_action_index=min(25 + tick_index, 49),
            terminated=terminal and tick_index == len(ticks) - 1,
        )
    trace["ticks"] = ticks
    payload["feedback_trace"] = trace
    second = replace(
        second,
        replay_data=replace(second.replay_data, payload=payload),
    )

    if not terminal:
        with pytest.raises(ValueError, match="predecessor-only"):
            attach_humanoid_transitions(
                (first, second),
                _async_motion_metrics(),
                {"humanoid_episode_length_env0": 2.0},
                behavior_policy_version=9,
                final_bootstrap_values={},
                control_timestep_us=500_000,
                expected_num_envs=1,
                max_transition_rows=30,
            )
        return

    patched = attach_humanoid_transitions(
        (first, second),
        _async_motion_metrics(),
        {"humanoid_episode_length_env0": 2.0},
        behavior_policy_version=9,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )
    transition = patched[1].replay_data.payload["transition"]
    assert transition["owning_reference_executed_ticks"] == 0
    assert transition["actor_valid"] is False
    assert not any(transition["actor_primitive_reward_mask"])


def test_async_motion_receipt_allows_predecessor_only_outer_truncation() -> None:
    first, second = _async_motion_outputs()
    assert second.replay_data is not None
    payload = dict(second.replay_data.payload)
    trace = dict(payload["feedback_trace"])
    ticks = [dict(tick) for tick in trace["ticks"]]
    for tick_index, tick in enumerate(ticks):
        tick.update(
            active_reference_id=7,
            active_reference_sha256="a" * 64,
            applied_reference_sha256="c" * 64,
            reference_action_index=min(25 + tick_index, 49),
            terminated=False,
        )
    trace["ticks"] = ticks
    payload["feedback_trace"] = trace
    payload["outer_truncated"] = True
    second = replace(
        second,
        replay_data=replace(second.replay_data, payload=payload),
    )
    dense = _async_motion_metrics()
    dense["humanoid_terminated_env0"]["values"] = [0.0, 0.0]
    dense["humanoid_truncated_env0"]["values"] = [0.0, 1.0]
    dense["humanoid_final_bootstrap_value_env0"] = {
        "timestamps_us": [1_240_000],
        "values": [2.0],
        "valid": [True],
    }

    patched = attach_humanoid_transitions(
        (first, second),
        dense,
        {
            "humanoid_episode_length_env0": 2.0,
            "humanoid_final_bootstrap_value_env0": 2.0,
        },
        behavior_policy_version=9,
        final_bootstrap_values={0: 2.0},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )
    transition = patched[1].replay_data.payload["transition"]
    assert transition["owning_reference_executed_ticks"] == 0
    assert transition["actor_valid"] is False
    assert not any(transition["actor_primitive_reward_mask"])
    assert transition["truncated"] is True
    assert transition["bootstrap_value"] == pytest.approx(2.0)


def test_h50_replay_retains_separate_applied_reference_hash() -> None:
    patched = attach_humanoid_transitions(
        (_motion_output(duration=1, applied_digest="b" * 64),),
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
    assert patched[0].replay_data.payload["transition"]["applied_reference_sha256"] == [
        "b" * 64
    ]


def test_invalid_plan_safe_hold_penalty_updates_the_rejected_source_actor() -> None:
    patched = attach_humanoid_transitions(
        (
            _motion_output(
                duration=1,
                terminated=True,
                applied_digest="b" * 64,
                reward_total=-10.0,
            ),
        ),
        _motion_metrics(
            timestamp_us=20_000,
            terminated=True,
            truncated=False,
            reward_value=-10.0,
        ),
        {"humanoid_episode_length_env0": 1.0},
        behavior_policy_version=1,
        final_bootstrap_values={},
        control_timestep_us=500_000,
        expected_num_envs=1,
        max_transition_rows=30,
    )

    transition = patched[0].replay_data.payload["transition"]
    assert transition["reward"] == pytest.approx(-10.0)
    assert transition["terminated"] is True
    assert transition["truncated"] is False
    assert transition["duration_ticks"] == 1
    assert transition["owning_reference_executed_ticks"] == 1
    assert transition["actor_valid"] is True
    assert transition["actor_primitive_reward_mask"] == [True]
    assert transition["actor_primitive_rewards"] == pytest.approx([-10.0])
    assert transition["reference_sha256"] == "a" * 64
    assert transition["applied_reference_sha256"] == ["b" * 64]
