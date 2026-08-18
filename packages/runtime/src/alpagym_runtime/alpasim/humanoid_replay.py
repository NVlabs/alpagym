# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strictly join humanoid policy rows to AlpaSim transition metrics."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from alpagym_runtime.types import PolicyOutput


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
    if behavior_policy_version < 0 or control_timestep_us <= 0 or expected_num_envs <= 0:
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
        step_indices = [int(row.model_extra["humanoid_step_index"]) for _, row in rows]
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
                raise ValueError("humanoid transition metric length differs from policy rows")

        for position, (output_index, output) in enumerate(rows):
            extra = output.model_extra
            assert extra is not None
            expected_timestamp = int(extra["humanoid_timestamp_us"]) + control_timestep_us
            if timestamps[position] != expected_timestamp:
                raise ValueError(
                    f"humanoid env_id={env_id} transition timestamp is misaligned"
                )
            reward = float(metrics[0]["values"][position])
            terminated = bool(metrics[1]["values"][position])
            truncated = bool(metrics[2]["values"][position])
            if not math.isfinite(reward) or (terminated and truncated):
                raise ValueError("humanoid transition has invalid reward or terminal flags")
            is_last = position == len(rows) - 1
            if is_last != (terminated or truncated):
                raise ValueError("humanoid terminal boundary does not match the final policy row")
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

            replay_data = output.replay_data
            assert replay_data is not None
            payload = dict(replay_data.payload)
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
            }
            patched[output_index] = replace(
                output,
                replay_data=replace(replay_data, payload=payload),
            )

    if set(final_bootstrap_values) != truncated_lanes:
        raise ValueError("final-state values do not match truncated humanoid lanes")
    return tuple(patched)


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
        raise ValueError(f"humanoid env_id={env_id} is missing final bootstrap metric {name!r}")
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
