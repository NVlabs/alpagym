# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strictly join humanoid policy rows to AlpaSim transition metrics."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, cast

from alpagym_runtime.types import PolicyOutput

_MOTION_REFERENCE_TICK_US = 20_000
_MOTION_CONTROLLER_TICKS = 25


def attach_humanoid_transitions(
    outputs: tuple[PolicyOutput, ...],
    dense_metrics: Mapping[str, Mapping[str, Any]],
    aggregate_metrics: Mapping[str, float],
    *,
    behavior_policy_version: int,
    final_bootstrap_values: Mapping[int, float],
    control_timestep_us: int,
    expected_num_envs: int,
    max_transition_rows: int,
) -> tuple[PolicyOutput, ...]:
    """Attach complete lane-local PPO facts without zero/default fallbacks."""
    if not outputs or len(outputs) > max_transition_rows:
        raise ValueError(
            "humanoid rollout row count must be in "
            f"[1, {max_transition_rows}], got {len(outputs)}"
        )
    if (
        behavior_policy_version < 0
        or control_timestep_us <= 0
        or expected_num_envs <= 0
    ):
        raise ValueError("invalid humanoid replay version, timestep, or lane count")

    indexed: dict[int, list[tuple[int, PolicyOutput]]] = {}
    episode_ids: set[int] = set()
    for output_index, output in enumerate(outputs):
        if output.replay_data is None or output.model_extra is None:
            raise ValueError("humanoid PPO output is missing replay data or identity")
        required = (
            "humanoid_env_id",
            "humanoid_episode_id",
            "humanoid_step_index",
            "humanoid_timestamp_us",
            "humanoid_value",
        )
        missing = [name for name in required if name not in output.model_extra]
        if missing:
            raise ValueError(f"humanoid PPO identity is missing fields {missing}")
        env_id = int(output.model_extra["humanoid_env_id"])
        episode_ids.add(int(output.model_extra["humanoid_episode_id"]))
        indexed.setdefault(env_id, []).append((output_index, output))

    expected_env_ids = set(range(expected_num_envs))
    if set(indexed) != expected_env_ids:
        raise ValueError(
            "humanoid replay lanes differ from num_envs: "
            f"expected={sorted(expected_env_ids)}, actual={sorted(indexed)}"
        )
    if len(episode_ids) != 1:
        raise ValueError(f"humanoid replay crossed reset IDs: {sorted(episode_ids)}")
    episode_id = next(iter(episode_ids))
    lane_lengths = {env_id: len(rows) for env_id, rows in indexed.items()}
    if len(set(lane_lengths.values())) != 1:
        raise ValueError(f"humanoid vector lanes have unequal lengths: {lane_lengths}")

    patched = list(outputs)
    truncated_lanes: set[int] = set()
    for env_id, rows in indexed.items():
        rows.sort(key=lambda item: int(item[1].model_extra["humanoid_step_index"]))
        step_indices: list[int] = []
        for _, row in rows:
            if row.model_extra is None:
                raise AssertionError("indexed humanoid row lost its model identity")
            step_indices.append(int(row.model_extra["humanoid_step_index"]))
        if step_indices != list(range(len(rows))):
            raise ValueError(f"humanoid env_id={env_id} has non-contiguous steps")
        episode_length_name = f"humanoid_episode_length_env{env_id}"
        episode_length = float(aggregate_metrics.get(episode_length_name, math.nan))
        if not episode_length.is_integer() or int(episode_length) != len(rows):
            raise ValueError(
                f"humanoid env_id={env_id} episode length metric does not match policy rows"
            )

        metric_names = (
            f"humanoid_reward_env{env_id}",
            f"humanoid_terminated_env{env_id}",
            f"humanoid_truncated_env{env_id}",
        )
        if any(name not in dense_metrics for name in metric_names):
            raise ValueError(f"humanoid env_id={env_id} is missing transition metrics")
        metrics = tuple(dense_metrics[name] for name in metric_names)
        timestamps = tuple(int(value) for value in metrics[0]["timestamps_us"])
        for metric in metrics:
            if tuple(int(value) for value in metric["timestamps_us"]) != timestamps:
                raise ValueError("humanoid transition metric timestamps differ")
            valid = tuple(bool(value) for value in metric.get("valid", ()))
            if valid and (len(valid) != len(rows) or not all(valid)):
                raise ValueError("humanoid transition metric contains invalid samples")
            if len(metric["values"]) != len(rows):
                raise ValueError(
                    "humanoid transition metric length differs from policy rows"
                )

        expected_control_episode_step = 1
        for position, (output_index, output) in enumerate(rows):
            extra = output.model_extra
            assert extra is not None
            metric_reward = float(metrics[0]["values"][position])
            metric_terminated = bool(metrics[1]["values"][position])
            metric_truncated = bool(metrics[2]["values"][position])
            replay_data = output.replay_data
            assert replay_data is not None
            payload = dict(replay_data.payload)
            motion_reference = "reference_sha256" in payload
            receipt_fields: dict[str, Any] = {}
            if motion_reference:
                reference_frame_count = int(output.chosen_xyz.shape[0])
                if reference_frame_count != 70:
                    raise ValueError("motion-reference replay must carry H70")
                controller_ticks, remainder_us = divmod(
                    control_timestep_us, _MOTION_REFERENCE_TICK_US
                )
                if remainder_us or controller_ticks != _MOTION_CONTROLLER_TICKS:
                    raise ValueError(
                        "motion-reference replay control_timestep_us must select "
                        "K=25 on the 20000us controller grid"
                    )
                (
                    reward,
                    terminated,
                    truncated,
                    expected_timestamp,
                    expected_control_episode_step,
                    receipt_fields,
                ) = _motion_reference_receipt(
                    payload,
                    env_id=env_id,
                    expected_source_decision_id=int(extra["humanoid_decision_id"]),
                    expected_control_episode_step=expected_control_episode_step,
                    metric_reward=metric_reward,
                    metric_terminated=metric_terminated,
                    metric_truncated=metric_truncated,
                    controller_ticks=controller_ticks,
                )
            else:
                expected_timestamp = (
                    int(extra["humanoid_timestamp_us"]) + control_timestep_us
                )
                reward = metric_reward
                terminated = metric_terminated
                truncated = metric_truncated
            if timestamps[position] != expected_timestamp:
                raise ValueError(
                    f"humanoid env_id={env_id} transition timestamp is misaligned"
                )
            if not math.isfinite(reward) or (terminated and truncated):
                raise ValueError(
                    "humanoid transition has invalid reward or terminal flags"
                )
            is_last = position == len(rows) - 1
            if is_last != (terminated or truncated):
                raise ValueError(
                    "humanoid terminal boundary does not match the final policy row"
                )
            old_value = float(extra["humanoid_value"])
            if terminated:
                bootstrap_value = 0.0
            elif truncated:
                truncated_lanes.add(env_id)
                if env_id not in final_bootstrap_values:
                    raise ValueError(
                        f"humanoid env_id={env_id} truncated without a final-state value"
                    )
                bootstrap_value = _final_bootstrap_from_metrics(
                    dense_metrics,
                    aggregate_metrics,
                    env_id=env_id,
                    expected_timestamp_us=timestamps[position],
                )
                recorded_bootstrap = float(final_bootstrap_values[env_id])
                if not math.isclose(
                    bootstrap_value,
                    recorded_bootstrap,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                ):
                    raise ValueError(
                        f"humanoid env_id={env_id} final bootstrap metric does not "
                        "match the policy response"
                    )
            else:
                next_extra = rows[position + 1][1].model_extra
                assert next_extra is not None
                bootstrap_value = float(next_extra["humanoid_value"])
            if not math.isfinite(old_value) or not math.isfinite(bootstrap_value):
                raise ValueError("humanoid transition contains non-finite values")

            if "transition" in payload:
                raise ValueError("humanoid replay already contains a transition block")
            payload["transition"] = {
                "env_id": env_id,
                "episode_id": episode_id,
                "step_index": position,
                "timestamp_us": timestamps[position],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "old_value": old_value,
                "bootstrap_value": bootstrap_value,
                "behavior_policy_version": behavior_policy_version,
                **receipt_fields,
            }
            patched[output_index] = replace(
                output,
                replay_data=replace(replay_data, payload=payload),
            )

    if set(final_bootstrap_values) != truncated_lanes:
        raise ValueError("final-state values do not match truncated humanoid lanes")
    return tuple(patched)


def _motion_reference_receipt(
    payload: Mapping[str, Any],
    *,
    env_id: int,
    expected_source_decision_id: int,
    expected_control_episode_step: int,
    metric_reward: float,
    metric_terminated: bool,
    metric_truncated: bool,
    controller_ticks: int,
) -> tuple[float, bool, bool, int, int, dict[str, Any]]:
    """Validate one exact controller trace and derive its SMDP transition."""
    trace = payload.get("feedback_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("motion-reference replay row is missing feedback_trace")
    if int(trace.get("env_id", -1)) != env_id:
        raise ValueError("motion-reference feedback env_id does not match replay lane")
    source_decision_id = int(trace.get("source_decision_id", -1))
    if source_decision_id != expected_source_decision_id or source_decision_id != int(
        payload.get("source_decision_id", -2)
    ):
        raise ValueError(
            "motion-reference feedback source decision does not match plan"
        )
    reference_id = int(payload.get("reference_id", -1))
    reference_sha256 = _require_sha256(
        "motion-reference replay reference_sha256",
        payload.get("reference_sha256"),
    )
    root_z_offset = float(payload.get("root_z_alignment_offset_m", math.nan))
    if reference_id <= 0 or not math.isfinite(root_z_offset):
        raise ValueError("motion-reference replay has invalid reference identity")
    ticks = trace.get("ticks")
    if not isinstance(ticks, list) or not 1 <= len(ticks) <= controller_ticks:
        raise ValueError("motion-reference feedback ticks must be a non-empty K-prefix")

    primitive_rewards = [0.0] * controller_ticks
    primitive_mask = [False] * controller_ticks
    previous_timestamp: int | None = None
    applied_hashes: list[str] = []
    for action_index, tick in enumerate(ticks):
        if not isinstance(tick, Mapping):
            raise TypeError("motion-reference feedback tick must be a mapping")
        tick = cast(Mapping[str, Any], tick)
        expected_offset = action_index + 1
        if int(tick.get("control_tick_offset", -1)) != expected_offset:
            raise ValueError("feedback control_tick_offset must be contiguous from one")
        if int(tick.get("reference_action_index", -1)) != action_index:
            raise ValueError(
                "feedback reference_action_index must be contiguous from zero"
            )
        if int(tick.get("active_reference_id", -1)) != reference_id:
            raise ValueError("feedback active_reference_id does not match plan")
        if (
            _require_sha256(
                "feedback active_reference_sha256",
                tick.get("active_reference_sha256"),
            )
            != reference_sha256
        ):
            raise ValueError("feedback active reference digest does not match plan")
        applied_sha256 = _require_sha256(
            "feedback applied_reference_sha256",
            tick.get("applied_reference_sha256"),
        )
        if applied_sha256 != reference_sha256:
            raise ValueError(
                "H70 feedback applied reference digest does not match source plan"
            )
        applied_hashes.append(applied_sha256)
        if not math.isclose(
            float(tick.get("root_z_alignment_offset_m", math.nan)),
            root_z_offset,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("feedback root Z alignment does not match plan")
        timestamp_us = int(tick.get("timestamp_us", -1))
        if (
            previous_timestamp is not None
            and timestamp_us != previous_timestamp + 20_000
        ):
            raise ValueError("feedback tick timestamps are not a contiguous 20 ms grid")
        previous_timestamp = timestamp_us
        control_episode_step = int(tick.get("control_episode_step", -1))
        if control_episode_step != expected_control_episode_step:
            raise ValueError("feedback control_episode_step is not contiguous")
        expected_control_episode_step += 1
        reward = float(tick.get("reward", math.nan))
        if not math.isfinite(reward):
            raise ValueError("feedback tick reward is non-finite")
        primitive_rewards[action_index] = reward
        primitive_mask[action_index] = True
        tick_terminated = bool(tick.get("terminated", False))
        tick_truncated = bool(tick.get("truncated", False))
        if tick_terminated and tick_truncated:
            raise ValueError("feedback tick cannot terminate and truncate together")
        if action_index < len(ticks) - 1 and (tick_terminated or tick_truncated):
            raise ValueError("only the final feedback tick may end a macro transition")
    assert previous_timestamp is not None
    final_tick = ticks[-1]
    terminated = bool(final_tick.get("terminated", False))
    truncated = bool(final_tick.get("truncated", False)) or bool(
        payload.get("outer_truncated", False)
    )
    if terminated and truncated:
        raise ValueError("terminated motion lane cannot be outer-truncated")
    if len(ticks) < controller_ticks and not (terminated or truncated):
        raise ValueError("short motion-reference trace must end the transition")
    reward = float(sum(primitive_rewards))
    if not math.isclose(reward, metric_reward, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError("controller tick rewards do not sum to ScenarioEval reward")
    if (terminated, truncated) != (metric_terminated, metric_truncated):
        raise ValueError("controller receipt terminal flags disagree with ScenarioEval")
    return (
        reward,
        terminated,
        truncated,
        previous_timestamp,
        expected_control_episode_step,
        {
            "primitive_rewards": primitive_rewards,
            "primitive_reward_mask": primitive_mask,
            "duration_ticks": len(ticks),
            "reference_id": reference_id,
            "source_decision_id": source_decision_id,
            "reference_sha256": reference_sha256,
            "applied_reference_sha256": applied_hashes,
            "root_z_alignment_offset_m": root_z_offset,
        },
    )


def _require_sha256(name: str, value: Any) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return digest


def _final_bootstrap_from_metrics(
    dense_metrics: Mapping[str, Mapping[str, Any]],
    aggregate_metrics: Mapping[str, float],
    *,
    env_id: int,
    expected_timestamp_us: int,
) -> float:
    """Cross-check the final-state value echoed by both AlpaSim metric paths."""
    name = f"humanoid_final_bootstrap_value_env{env_id}"
    if name not in dense_metrics:
        raise ValueError(
            f"humanoid env_id={env_id} is missing final bootstrap metric {name!r}"
        )
    metric = dense_metrics[name]
    timestamps = tuple(int(value) for value in metric.get("timestamps_us", ()))
    values = tuple(float(value) for value in metric.get("values", ()))
    valid = tuple(bool(value) for value in metric.get("valid", ()))
    if (
        len(timestamps) != 1
        or timestamps[0] != expected_timestamp_us
        or len(values) != 1
        or valid != (True,)
        or not math.isfinite(values[0])
    ):
        raise ValueError(
            f"humanoid env_id={env_id} final bootstrap metric must contain one "
            "valid finite value at the final transition timestamp"
        )
    if name not in aggregate_metrics:
        raise ValueError(
            f"humanoid env_id={env_id} is missing aggregate final bootstrap metric {name!r}"
        )
    aggregate_value = float(aggregate_metrics[name])
    if not math.isfinite(aggregate_value) or not math.isclose(
        aggregate_value,
        values[0],
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"humanoid env_id={env_id} dense and aggregate final bootstrap metrics differ"
        )
    return values[0]
