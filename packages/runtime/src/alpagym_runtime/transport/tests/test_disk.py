# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from alpagym_runtime.inference.types import NUM_ROUTE_WAYPOINTS, ModelInput
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.transport.disk import read_episode_json, write_episode_json
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


def _make_replay_data(payload: dict[str, object] | None = None) -> PolicyReplayData:
    """Build a typed replay envelope for disk-boundary tests."""
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=1),
        old_logprob=torch.tensor(-0.25, dtype=torch.float32),
        payload=payload
        if payload is not None
        else {"scene_id": "scene_001", "step": 7},
    )


def _route_xy() -> torch.Tensor:
    """Build the fixed route tensor carried by replay model inputs."""
    return torch.full((NUM_ROUTE_WAYPOINTS, 2), float("nan"), dtype=torch.float32)


def _make_full_episode_output() -> EpisodeOutput:
    """Build a representative EpisodeOutput exercising every artifact field."""
    policy_output = PolicyOutput(
        chosen_xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        ),
        chosen_quat=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        chosen_dt_us=torch.tensor([0, 100_000, 200_000], dtype=torch.int64),
        chosen_logprob=torch.tensor([-0.5], dtype=torch.float32),
        replay_data=_make_replay_data(),
        all_pred_xyz=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32
        ),
        all_pred_quat=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
        ),
        model_extra={"score": 0.75},
    )
    return EpisodeOutput(
        scene_id="scene_001",
        session_uuid="session_001",
        num_steps=1,
        policy_outputs=(policy_output,),
        rollout_seed=(1 << 64) - 1,
        executed_ego_trajectory=Trajectory(
            poses=(
                EgoPose(
                    timestamp_us=100,
                    pose=Pose(
                        vec=Vec3(x=1.0, y=2.0, z=3.0),
                        quat=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
                    ),
                ),
                EgoPose(
                    timestamp_us=200,
                    pose=Pose(
                        vec=Vec3(x=4.0, y=5.0, z=6.0),
                        quat=Quaternion(w=0.5, x=0.5, y=0.5, z=0.5),
                    ),
                ),
            )
        ),
        route_waypoints=(
            RouteWaypoint(x=1.0, y=2.0, z=3.0),
            RouteWaypoint(x=4.0, y=5.0),
        ),
        metrics=EpisodeMetrics(
            aggregated={"route_progress": 0.5},
            dense={"speed": [1.0, 2.0]},
        ),
        reward=RewardResult(total=-1.25, report_metrics={"distance": 1.25}),
        is_valid=False,
    )


def _read_v2_artifact(path: Path) -> tuple[dict[str, Any], Path]:
    """Return a v2 manifest envelope and its referenced tensor sidecar."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["artifact_schema"] == "alpagym.disk_episode.v2"
    descriptor = artifact["tensor_sidecar"]
    sidecar_path = path.parent / descriptor["filename"]
    return artifact, sidecar_path


def _assert_policy_output_equal(actual: PolicyOutput, expected: PolicyOutput) -> None:
    """Assert a single PolicyOutput round-trips identically across fields."""
    assert actual.chosen_xyz.dtype == expected.chosen_xyz.dtype
    assert torch.equal(actual.chosen_xyz, expected.chosen_xyz)

    assert actual.chosen_quat.dtype == expected.chosen_quat.dtype
    assert torch.equal(actual.chosen_quat, expected.chosen_quat)

    assert actual.chosen_dt_us.dtype == expected.chosen_dt_us.dtype
    assert torch.equal(actual.chosen_dt_us, expected.chosen_dt_us)

    if expected.chosen_logprob is None:
        assert actual.chosen_logprob is None
    else:
        assert actual.chosen_logprob is not None
        assert torch.equal(actual.chosen_logprob, expected.chosen_logprob)

    if expected.all_pred_xyz is None:
        assert actual.all_pred_xyz is None
    else:
        assert actual.all_pred_xyz is not None
        assert torch.equal(actual.all_pred_xyz, expected.all_pred_xyz)

    if expected.all_pred_quat is None:
        assert actual.all_pred_quat is None
    else:
        assert actual.all_pred_quat is not None
        assert torch.equal(actual.all_pred_quat, expected.all_pred_quat)

    if expected.replay_data is None:
        assert actual.replay_data is None
    else:
        assert actual.replay_data is not None
        assert actual.replay_data.model_family == expected.replay_data.model_family
        assert actual.replay_data.payload_schema == expected.replay_data.payload_schema
        assert (
            actual.replay_data.action_selection == expected.replay_data.action_selection
        )
        assert actual.replay_data.payload == expected.replay_data.payload
        assert actual.replay_data.old_logprob is not None
        assert expected.replay_data.old_logprob is not None
        assert torch.equal(
            actual.replay_data.old_logprob, expected.replay_data.old_logprob
        )
    assert actual.model_extra == expected.model_extra


def test_write_episode_json_creates_parent_dirs(
    tmp_path: Path,
) -> None:
    """`write_episode_json` writes a manifest and sidecar under the target dir."""
    episode = _make_full_episode_output()
    target_path = tmp_path / "nested" / "scene_001.json"

    write_episode_json(target_path, episode)

    assert target_path.is_file()
    assert target_path.read_text(encoding="utf-8").startswith("{")
    artifact, sidecar_path = _read_v2_artifact(target_path)
    assert artifact["manifest_sha256"]
    assert sidecar_path.is_file()


def test_rollout_artifact_round_trips_full_episode_output(tmp_path: Path) -> None:
    """A fully populated EpisodeOutput round-trips through the disk API."""
    episode = _make_full_episode_output()
    target_path = tmp_path / "scene_001.json"

    write_episode_json(target_path, episode)
    loaded = read_episode_json(target_path)

    assert loaded.scene_id == episode.scene_id
    assert loaded.session_uuid == episode.session_uuid
    assert loaded.num_steps == episode.num_steps
    assert loaded.rollout_seed == (1 << 64) - 1
    artifact, _ = _read_v2_artifact(target_path)
    assert artifact["episode_manifest"]["rollout_seed"] == (1 << 64) - 1
    assert loaded.is_valid == episode.is_valid
    assert loaded.route_waypoints == episode.route_waypoints
    assert loaded.executed_ego_trajectory == episode.executed_ego_trajectory
    assert loaded.metrics == episode.metrics
    assert loaded.reward == episode.reward
    assert len(loaded.policy_outputs) == len(episode.policy_outputs)
    for actual_output, expected_output in zip(
        loaded.policy_outputs, episode.policy_outputs
    ):
        _assert_policy_output_equal(actual_output, expected_output)


def test_write_episode_json_extracts_tensor_replay_payload(tmp_path: Path) -> None:
    """Trace tensors become manifest references backed by the tensor sidecar."""
    source_logprobs = torch.tensor(
        [[-0.1, -0.2]], dtype=torch.float32, requires_grad=True
    )
    non_leaf_logprobs = source_logprobs * 1.0
    model_input = ModelInput(
        ego_history_xyz=torch.zeros((1, 2, 3), dtype=torch.float32),
        ego_history_rot=torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(1, 2, 3, 3),
        camera_frames=torch.zeros((0, 3, 1, 1), dtype=torch.uint8),
        camera_indices=torch.zeros((0,), dtype=torch.int64),
        relative_timestamps=torch.zeros((0,), dtype=torch.int64),
        route_xy=_route_xy(),
    )
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        chosen_logprob=torch.tensor([-0.25], dtype=torch.float32),
        replay_data=_make_replay_data(
            {
                "model_input": model_input,
                "per_traj_logprob": non_leaf_logprobs,
                "set_ix": 0,
                "sample_ix": 1,
            }
        ),
        model_extra={"cot_tensor": torch.tensor([1.0, 2.0])},
    )
    episode = EpisodeOutput(
        scene_id="trace_scene",
        session_uuid="trace_session",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    target_path = tmp_path / "trace.json"

    write_episode_json(target_path, episode)

    artifact, sidecar_path = _read_v2_artifact(target_path)
    manifest = artifact["episode_manifest"]
    raw_replay = manifest["policy_outputs"][0]["replay_data"]
    assert isinstance(raw_replay, dict)
    assert isinstance(raw_replay["payload"], dict)
    assert isinstance(raw_replay["payload"]["model_input"], dict)
    assert "__tensor_key__" in raw_replay["payload"]["per_traj_logprob"]
    assert (
        "__tensor_key__" in manifest["policy_outputs"][0]["model_extra"]["cot_tensor"]
    )
    assert sidecar_path.is_file()


def test_disk_round_trip_preserves_typed_model_input_and_uint8(
    tmp_path: Path,
) -> None:
    """Any-slot dataclasses and non-float tensor dtypes survive disk round-trip."""
    model_input = ModelInput(
        ego_history_xyz=torch.zeros((1, 2, 3), dtype=torch.float32),
        ego_history_rot=torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(1, 2, 3, 3),
        camera_frames=torch.ones((2, 3, 4, 5), dtype=torch.uint8),
        camera_indices=torch.tensor([0, 1], dtype=torch.int64),
        relative_timestamps=torch.tensor([0, 1], dtype=torch.int64),
        route_xy=_route_xy(),
    )
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        chosen_logprob=torch.tensor([-0.25], dtype=torch.float32),
        replay_data=_make_replay_data({"model_input": model_input}),
    )
    episode = EpisodeOutput(
        scene_id="uint8_scene",
        session_uuid="uint8_session",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    target_path = tmp_path / "uint8.json"

    write_episode_json(target_path, episode)
    loaded = read_episode_json(target_path)

    assert loaded.policy_outputs[0].replay_data is not None
    payload = loaded.policy_outputs[0].replay_data.payload
    restored = payload["model_input"]
    assert isinstance(restored, ModelInput)
    assert restored.camera_frames.dtype == torch.uint8
    assert restored.camera_frames.shape == (2, 3, 4, 5)
    assert restored.camera_indices.dtype == torch.int64


def test_replay_data_round_trips_as_typed_envelope_with_json_payload(
    tmp_path: Path,
) -> None:
    """Replay dataclasses and tensor leaves read back with their original types."""
    model_input = ModelInput(
        ego_history_xyz=torch.zeros((1, 2, 3), dtype=torch.float32),
        ego_history_rot=torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(1, 2, 3, 3),
        camera_frames=torch.zeros((0, 3, 1, 1), dtype=torch.uint8),
        camera_indices=torch.zeros((0,), dtype=torch.int64),
        relative_timestamps=torch.zeros((0,), dtype=torch.int64),
        route_xy=_route_xy(),
    )
    per_traj_logprob = torch.tensor([[-0.1, -0.2]], dtype=torch.float32)
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        chosen_logprob=torch.tensor([-0.25], dtype=torch.float32),
        replay_data=_make_replay_data(
            {
                "model_input": model_input,
                "per_traj_logprob": per_traj_logprob,
                "set_ix": 0,
                "sample_ix": 1,
            }
        ),
        model_extra={
            "cot_tensor": torch.tensor([1.0, 2.0]),
            "valid_mask": torch.tensor([True, False], dtype=torch.bool),
            "cot_ndarray": np.array([3.0, 4.0], dtype=np.float32),
        },
    )
    episode = EpisodeOutput(
        scene_id="trace_scene",
        session_uuid="trace_session",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    target_path = tmp_path / "trace.json"

    write_episode_json(target_path, episode)
    loaded = read_episode_json(target_path)

    actual_policy = loaded.policy_outputs[0]

    assert isinstance(actual_policy.chosen_xyz, torch.Tensor)
    assert actual_policy.chosen_logprob is not None
    assert isinstance(actual_policy.chosen_logprob, torch.Tensor)

    assert actual_policy.replay_data is not None
    assert isinstance(actual_policy.replay_data, PolicyReplayData)

    rehydrated_model_input = actual_policy.replay_data.payload["model_input"]
    assert isinstance(rehydrated_model_input, ModelInput)
    assert rehydrated_model_input.camera_frames.dtype == torch.uint8
    assert rehydrated_model_input.camera_frames.shape == (0, 3, 1, 1)
    assert torch.equal(
        rehydrated_model_input.ego_history_xyz,
        model_input.ego_history_xyz,
    )

    rehydrated_logprob = actual_policy.replay_data.payload["per_traj_logprob"]
    assert isinstance(rehydrated_logprob, torch.Tensor)
    assert torch.equal(rehydrated_logprob, per_traj_logprob)

    assert actual_policy.replay_data.payload["set_ix"] == 0
    assert actual_policy.replay_data.payload["sample_ix"] == 1

    assert actual_policy.model_extra is not None
    assert torch.equal(
        actual_policy.model_extra["cot_tensor"],
        torch.tensor([1.0, 2.0]),
    )
    assert actual_policy.model_extra["valid_mask"].dtype == torch.bool
    assert torch.equal(
        actual_policy.model_extra["valid_mask"],
        torch.tensor([True, False]),
    )
    assert isinstance(actual_policy.model_extra["cot_ndarray"], list)
    assert actual_policy.model_extra["cot_ndarray"] == [3.0, 4.0]


def test_rollout_artifact_round_trips_minimal_episode_output(tmp_path: Path) -> None:
    """Optional EpisodeOutput / PolicyOutput fields stay None across round-trip."""
    minimal_policy = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
    )
    episode = EpisodeOutput(
        scene_id="scene_minimal",
        session_uuid="session_minimal",
        num_steps=1,
        policy_outputs=(minimal_policy,),
    )
    target_path = tmp_path / "minimal.json"

    write_episode_json(target_path, episode)
    loaded = read_episode_json(target_path)

    assert loaded.scene_id == "scene_minimal"
    assert loaded.session_uuid == "session_minimal"
    assert loaded.num_steps == 1
    assert loaded.rollout_seed is None
    assert loaded.is_valid is True
    assert loaded.route_waypoints == ()
    assert loaded.executed_ego_trajectory == Trajectory(poses=())
    assert loaded.metrics is None
    assert loaded.reward is None

    assert len(loaded.policy_outputs) == 1
    actual_policy = loaded.policy_outputs[0]
    _assert_policy_output_equal(actual_policy, minimal_policy)


def test_read_episode_json_accepts_legacy_plain_json(tmp_path: Path) -> None:
    """Artifacts written by the original JSON-only transport remain readable."""
    legacy_artifact = {
        "scene_id": "legacy_scene",
        "session_uuid": "legacy_session",
        "num_steps": 1,
        "policy_outputs": [
            {
                "chosen_xyz": [[0.0, 0.0, 0.0]],
                "chosen_quat": [[1.0, 0.0, 0.0, 0.0]],
                "chosen_dt_us": [0],
                "chosen_logprob": None,
                "replay_data": None,
                "all_pred_xyz": None,
                "all_pred_quat": None,
                "model_extra": None,
            }
        ],
        "executed_ego_trajectory": [],
        "route_waypoints": [],
        "metrics": None,
        "reward": None,
        "is_valid": True,
    }
    target_path = tmp_path / "legacy.json"
    target_path.write_text(json.dumps(legacy_artifact), encoding="utf-8")

    loaded = read_episode_json(target_path)

    assert loaded.scene_id == "legacy_scene"
    assert loaded.session_uuid == "legacy_session"
    assert loaded.rollout_seed is None
    assert loaded.policy_outputs[0].chosen_xyz.dtype == torch.float32
    assert loaded.policy_outputs[0].chosen_dt_us.dtype == torch.int64


def test_read_episode_json_rejects_corrupted_tensor_sidecar(tmp_path: Path) -> None:
    """A byte-level tensor-file mutation is rejected before deserialization."""
    target_path = tmp_path / "corrupt_tensor.json"
    write_episode_json(target_path, _make_full_episode_output())
    _, sidecar_path = _read_v2_artifact(target_path)
    sidecar_bytes = bytearray(sidecar_path.read_bytes())
    sidecar_bytes[len(sidecar_bytes) // 2] ^= 0x01
    sidecar_path.write_bytes(sidecar_bytes)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_episode_json(target_path)


def test_read_episode_json_rejects_corrupted_manifest(tmp_path: Path) -> None:
    """A manifest mutation fails its digest instead of producing plausible data."""
    target_path = tmp_path / "corrupt_manifest.json"
    write_episode_json(target_path, _make_full_episode_output())
    artifact, _ = _read_v2_artifact(target_path)
    artifact["episode_manifest"]["num_steps"] = 99
    target_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        read_episode_json(target_path)


def test_read_episode_json_rejects_sidecar_path_traversal(tmp_path: Path) -> None:
    """A sidecar descriptor cannot escape the artifact directory."""
    target_path = tmp_path / "unsafe_path.json"
    write_episode_json(target_path, _make_full_episode_output())
    artifact, _ = _read_v2_artifact(target_path)
    artifact["tensor_sidecar"]["filename"] = "../outside.tensors.pt"
    target_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a basename"):
        read_episode_json(target_path)


def test_rewriting_artifact_releases_previous_sidecar(tmp_path: Path) -> None:
    """Atomic replacement does not leak the superseded tensor sidecar."""
    target_path = tmp_path / "replace.json"
    episode = _make_full_episode_output()
    write_episode_json(target_path, episode)
    _, previous_sidecar = _read_v2_artifact(target_path)

    write_episode_json(target_path, episode)
    _, current_sidecar = _read_v2_artifact(target_path)

    assert current_sidecar != previous_sidecar
    assert current_sidecar.is_file()
    assert not previous_sidecar.exists()
