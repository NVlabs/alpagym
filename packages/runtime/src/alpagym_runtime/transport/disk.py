# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import redis
import torch

from alpagym_runtime.replay import parse_policy_replay_data
from alpagym_runtime.types import (
    EgoPose,
    EpisodeMetrics,
    EpisodeOutput,
    PolicyOutput,
    Pose,
    Quaternion,
    RewardResult,
    RouteWaypoint,
    Trajectory,
    Vec3,
)


def _tensor_to_list(tensor: torch.Tensor) -> list:
    """Detach a tensor and return its contents as nested Python lists."""
    return tensor.detach().cpu().tolist()


def _artifact_default(value: Any) -> Any:
    """`json.dumps` `default=` hook for tensor, ndarray, and dataclass leaves.

    Tensor and ndarray leaves serialize to plain Python lists, so the disk transport
    is lossy for dtype: leaves in schemaless ``dict[str, Any]`` slots (``model_extra``,
    ``metrics.dense``, the replay ``payload``) read back as lists, not tensors. The NCCL
    transport preserves torch-tensor dtype for the same slots. Trainer code that needs
    a typed tensor (e.g. ``ModelInput.from_payload``) re-applies the dtype on read.
    """
    if isinstance(value, torch.Tensor):
        return _tensor_to_list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ego_pose_to_dict(ego_pose: EgoPose) -> dict[str, Any]:
    """Serialize one `EgoPose` into a JSON-friendly dictionary."""
    return {
        "timestamp_us": int(ego_pose.timestamp_us),
        "pose": {
            "vec": {
                "x": float(ego_pose.pose.vec.x),
                "y": float(ego_pose.pose.vec.y),
                "z": float(ego_pose.pose.vec.z),
            },
            "quat": {
                "w": float(ego_pose.pose.quat.w),
                "x": float(ego_pose.pose.quat.x),
                "y": float(ego_pose.pose.quat.y),
                "z": float(ego_pose.pose.quat.z),
            },
        },
    }


def _ego_pose_from_dict(payload: Mapping[str, Any]) -> EgoPose:
    """Build an `EgoPose` from one dictionary produced by serialization."""
    pose_payload = payload["pose"]
    vec_payload = pose_payload["vec"]
    quat_payload = pose_payload["quat"]
    return EgoPose(
        timestamp_us=int(payload["timestamp_us"]),
        pose=Pose(
            vec=Vec3(
                x=float(vec_payload["x"]),
                y=float(vec_payload["y"]),
                z=float(vec_payload["z"]),
            ),
            quat=Quaternion(
                w=float(quat_payload["w"]),
                x=float(quat_payload["x"]),
                y=float(quat_payload["y"]),
                z=float(quat_payload["z"]),
            ),
        ),
    )


def _policy_output_to_dict(output: PolicyOutput) -> dict[str, Any]:
    """Serialize a `PolicyOutput` into a JSON-friendly dictionary."""
    return {
        "chosen_xyz": _tensor_to_list(output.chosen_xyz),
        "chosen_quat": _tensor_to_list(output.chosen_quat),
        "chosen_dt_us": _tensor_to_list(output.chosen_dt_us),
        "chosen_logprob": (
            _tensor_to_list(output.chosen_logprob)
            if output.chosen_logprob is not None
            else None
        ),
        "replay_data": output.replay_data.to_dict()
        if output.replay_data is not None
        else None,
        "all_pred_xyz": (
            _tensor_to_list(output.all_pred_xyz)
            if output.all_pred_xyz is not None
            else None
        ),
        "all_pred_quat": (
            _tensor_to_list(output.all_pred_quat)
            if output.all_pred_quat is not None
            else None
        ),
        "model_extra": dict(output.model_extra)
        if output.model_extra is not None
        else None,
    }


def _policy_output_from_dict(payload: Mapping[str, Any]) -> PolicyOutput:
    """Build a `PolicyOutput` from serialized data."""
    chosen_logprob = payload["chosen_logprob"]
    all_pred_xyz = payload["all_pred_xyz"]
    all_pred_quat = payload["all_pred_quat"]
    replay_data = payload["replay_data"]
    model_extra = payload["model_extra"]
    return PolicyOutput(
        chosen_xyz=torch.tensor(payload["chosen_xyz"], dtype=torch.float32),
        chosen_quat=torch.tensor(payload["chosen_quat"], dtype=torch.float32),
        chosen_dt_us=torch.tensor(payload["chosen_dt_us"], dtype=torch.int64),
        chosen_logprob=(
            torch.tensor(chosen_logprob, dtype=torch.float32)
            if chosen_logprob is not None
            else None
        ),
        replay_data=parse_policy_replay_data(replay_data)
        if replay_data is not None
        else None,
        all_pred_xyz=(
            torch.tensor(all_pred_xyz, dtype=torch.float32)
            if all_pred_xyz is not None
            else None
        ),
        all_pred_quat=(
            torch.tensor(all_pred_quat, dtype=torch.float32)
            if all_pred_quat is not None
            else None
        ),
        model_extra=dict(model_extra) if model_extra is not None else None,
    )


def _episode_to_artifact_dict(episode: EpisodeOutput) -> dict[str, Any]:
    """Return the JSON-serializable artifact payload for one episode."""
    policy_outputs = [
        _policy_output_to_dict(output) for output in episode.policy_outputs
    ]
    executed_ego_trajectory = [
        _ego_pose_to_dict(pose) for pose in episode.executed_ego_trajectory.poses
    ]
    route_waypoints = [
        {"x": waypoint.x, "y": waypoint.y, "z": waypoint.z}
        for waypoint in episode.route_waypoints
    ]

    metrics = None
    if episode.metrics is not None:
        metrics = {
            "aggregated": dict(episode.metrics.aggregated),
            "dense": dict(episode.metrics.dense),
        }

    reward = None
    if episode.reward is not None:
        reward = {
            "total": episode.reward.total,
            "report_metrics": dict(episode.reward.report_metrics),
        }

    return {
        "scene_id": episode.scene_id,
        "session_uuid": episode.session_uuid,
        "num_steps": episode.num_steps,
        "rollout_seed": episode.rollout_seed,
        "policy_outputs": policy_outputs,
        "executed_ego_trajectory": executed_ego_trajectory,
        "route_waypoints": route_waypoints,
        "metrics": metrics,
        "reward": reward,
        "is_valid": episode.is_valid,
    }


def _episode_from_artifact_dict(artifact: Mapping[str, Any]) -> EpisodeOutput:
    """Build an episode output from its JSON artifact payload."""
    policy_outputs = tuple(
        _policy_output_from_dict(output) for output in artifact["policy_outputs"]
    )
    executed_ego_trajectory = Trajectory(
        poses=tuple(
            _ego_pose_from_dict(pose) for pose in artifact["executed_ego_trajectory"]
        )
    )

    metrics = None
    if artifact["metrics"] is not None:
        metrics = EpisodeMetrics(
            aggregated=dict(artifact["metrics"]["aggregated"]),
            dense=dict(artifact["metrics"]["dense"]),
        )

    reward = None
    if artifact["reward"] is not None:
        reward = RewardResult(
            total=artifact["reward"]["total"],
            report_metrics=dict(artifact["reward"]["report_metrics"]),
        )

    return EpisodeOutput(
        scene_id=artifact["scene_id"],
        session_uuid=artifact["session_uuid"],
        num_steps=artifact["num_steps"],
        policy_outputs=policy_outputs,
        rollout_seed=artifact.get("rollout_seed"),
        executed_ego_trajectory=executed_ego_trajectory,
        route_waypoints=tuple(
            RouteWaypoint(
                x=waypoint["x"],
                y=waypoint["y"],
                z=waypoint["z"],
            )
            for waypoint in artifact["route_waypoints"]
        ),
        metrics=metrics,
        reward=reward,
        is_valid=artifact["is_valid"],
    )


def write_episode_json(path: Path, episode: EpisodeOutput) -> None:
    """Write ``episode`` as a JSON artifact at ``path``, creating parent dirs.

    Uses an atomic tmp-then-rename write so a preemption mid-write never leaves
    a partial file at the final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(
            _episode_to_artifact_dict(episode),
            indent=2,
            default=_artifact_default,
        ),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def read_episode_json(handle: str | Path) -> EpisodeOutput:
    """Read a rollout episode result from a disk artifact handle."""
    artifact_data: dict[str, Any] = json.loads(Path(handle).read_text(encoding="utf-8"))
    return _episode_from_artifact_dict(artifact_data)


class DiskEpisodeWriter:
    """Rollout-side disk egress: writes each episode as a JSON artifact."""

    def __init__(self, artifacts_dir: Path):
        """Create a writer that writes artifacts under ``artifacts_dir``."""
        self._artifacts_dir = Path(artifacts_dir).resolve()

    def write(self, episode: EpisodeOutput) -> str:
        """Persist ``episode`` as JSON and return its file path as the handle.

        The handle carries a fresh ``uuid4`` suffix so two episodes that share a
        ``(scene_id, session_uuid)`` cannot overwrite each other's artifact.
        """
        filename = f"{episode.scene_id}_{episode.session_uuid}_{uuid.uuid4().hex}.json"
        path = self._artifacts_dir / filename
        write_episode_json(path, episode)
        return str(path)

    def release(self, handle: str, reason: str) -> None:
        """Discard a JSON artifact that will not be read."""
        del reason
        Path(handle).unlink(missing_ok=True)

    def start_cleanup(self, redis_client: redis.Redis) -> None:
        """No-op: the disk writer has no out-of-band discard channel."""
        del redis_client

    def flush_pending_sends(self) -> None:
        """No-op: disk writes are synchronous, so nothing is ever pending."""

    def close(self) -> None:
        """No-op: disk writer holds no live resources."""
