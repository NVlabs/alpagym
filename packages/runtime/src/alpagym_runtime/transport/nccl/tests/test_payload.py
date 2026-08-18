# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EpisodeOutput packing across the NCCL split."""

import numpy as np
import pytest
import torch
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.transport.nccl.payload import (
    WirePayload,
    _pack,
    _unpack,
    pack,
    unpack,
)
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


def test_bool_tensor_packs_as_uint8_and_restores() -> None:
    """Bool tensors ship as uint8 (pynccl cannot send torch.bool) and unpack restores bool."""
    mask = torch.tensor([[True, False], [True, True]])
    tensors: dict[str, torch.Tensor] = {}
    leaf = _pack(mask, tensors)
    assert leaf["dtype"] == "torch.uint8"
    assert leaf["__bool_tensor__"] is True
    assert tensors[leaf["__tensor_key__"]].dtype == torch.uint8
    restored = _unpack(leaf, torch.Tensor, tensors)
    assert restored.dtype == torch.bool
    assert torch.equal(restored, mask)


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_zero_element_tensor_rejected_at_pack(shape: tuple[int, ...]) -> None:
    """Pynccl rejects empty buffers, so _pack fails before a manifest/rendezvous exists."""
    with pytest.raises(ValueError, match="zero-element tensor"):
        _pack(torch.zeros(shape), {})


def _minimal_policy_output() -> PolicyOutput:
    """Build a PolicyOutput with only the three required tensor fields populated."""
    return PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
    )


def _minimal_episode() -> EpisodeOutput:
    """Build a minimal EpisodeOutput with one PolicyOutput and no optional fields."""
    return EpisodeOutput(
        scene_id="scene_alpha",
        session_uuid="session_zero",
        num_steps=1,
        policy_outputs=(_minimal_policy_output(),),
    )


def _replay_data(
    payload: dict[str, object],
    old_logprob: torch.Tensor | None = None,
) -> PolicyReplayData:
    """Build a typed replay envelope for NCCL payload tests."""
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema="test.replay.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=1),
        old_logprob=old_logprob,
        payload=payload,
    )


def test_pack_minimal_episode_extracts_required_tensors() -> None:
    """Required tensor fields of PolicyOutput land in the flat tensor map."""
    payload = pack(_minimal_episode())
    assert set(payload.tensors.keys()) == {"tensor_0", "tensor_1", "tensor_2"}
    assert isinstance(payload, WirePayload)


def test_minimal_episode_round_trip() -> None:
    """Pack then unpack reproduces the minimal EpisodeOutput byte-for-byte."""
    episode = _minimal_episode()
    reconstructed = unpack(pack(episode))
    assert reconstructed.scene_id == episode.scene_id
    assert reconstructed.session_uuid == episode.session_uuid
    assert reconstructed.num_steps == episode.num_steps
    assert reconstructed.is_valid is True
    original_po = episode.policy_outputs[0]
    new_po = reconstructed.policy_outputs[0]
    assert torch.equal(new_po.chosen_xyz, original_po.chosen_xyz)
    assert torch.equal(new_po.chosen_quat, original_po.chosen_quat)
    assert torch.equal(new_po.chosen_dt_us, original_po.chosen_dt_us)
    assert new_po.chosen_logprob is None
    assert new_po.all_pred_xyz is None
    assert new_po.replay_data is None
    assert new_po.model_extra is None


def test_unpack_rejects_manifest_missing_field() -> None:
    """A manifest missing a dataclass field fails at the transport boundary.

    The NCCL split is a same-version in-process round-trip and ``pack`` emits every
    field, so a missing field is corruption. Unpack must raise rather than silently
    substitute the dataclass default (which would turn dropped metadata into a
    plausible value like ``model_extra=None``).
    """
    payload = pack(_minimal_episode())
    policy_output_entry = payload.manifest["policy_outputs"][0]
    assert "model_extra" in policy_output_entry
    del policy_output_entry["model_extra"]

    with pytest.raises(KeyError, match="model_extra"):
        unpack(payload)


def test_full_episode_round_trip() -> None:
    """All optional tensor fields, replay_data tensors, metrics, reward survive."""
    replay_tensor = torch.tensor([0.5, 0.25], dtype=torch.float32)
    nested_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        chosen_logprob=torch.tensor([-0.5], dtype=torch.float32),
        all_pred_xyz=torch.zeros((1, 4, 3), dtype=torch.float32),
        all_pred_quat=torch.zeros((1, 4, 4), dtype=torch.float32),
        replay_data=_replay_data(
            {
                "per_traj_logprob": replay_tensor,
                "nested": {"inner_tensor": nested_tensor},
            }
        ),
        model_extra={"score": 0.75},
    )
    episode = EpisodeOutput(
        scene_id="scene_full",
        session_uuid="session_full",
        num_steps=2,
        policy_outputs=(policy_output,),
        rollout_seed=(1 << 64) - 1,
        executed_ego_trajectory=Trajectory(
            poses=(
                EgoPose(
                    timestamp_us=100,
                    pose=Pose(vec=Vec3(x=1.0), quat=Quaternion()),
                ),
            )
        ),
        route_waypoints=(RouteWaypoint(x=2.0, y=3.0),),
        metrics=EpisodeMetrics(aggregated={"score": 0.5}, dense={}),
        reward=RewardResult(total=-1.0, report_metrics={"distance": 1.0}),
        is_valid=False,
    )
    reconstructed = unpack(pack(episode))
    new_po = reconstructed.policy_outputs[0]
    assert torch.equal(new_po.chosen_logprob, policy_output.chosen_logprob)
    assert torch.equal(new_po.all_pred_xyz, policy_output.all_pred_xyz)
    assert torch.equal(new_po.all_pred_quat, policy_output.all_pred_quat)
    assert new_po.replay_data is not None
    assert new_po.replay_data.action_selection.set_ix == 0
    assert new_po.replay_data.action_selection.sample_ix == 1
    assert torch.equal(new_po.replay_data.payload["per_traj_logprob"], replay_tensor)
    assert torch.equal(
        new_po.replay_data.payload["nested"]["inner_tensor"], nested_tensor
    )
    assert new_po.model_extra == {"score": 0.75}
    assert reconstructed.metrics == episode.metrics
    assert reconstructed.reward == episode.reward
    assert reconstructed.executed_ego_trajectory == episode.executed_ego_trajectory
    assert reconstructed.route_waypoints == episode.route_waypoints
    assert reconstructed.rollout_seed == (1 << 64) - 1
    assert reconstructed.is_valid is False


def test_replay_data_tensors_extracted_to_flat_map() -> None:
    """Tensors nested under replay_data flow over NCCL, not via the manifest."""
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        replay_data=_replay_data({"inner": torch.tensor([7.0, 8.0])}),
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    payload = pack(episode)
    assert len(payload.tensors) == 4


def test_model_extra_tensors_round_trip() -> None:
    """Tensors nested under model_extra extract to the flat map and rebuild on unpack."""
    flat_tensor = torch.tensor([1.5, 2.5], dtype=torch.float32)
    nested_tensor = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        model_extra={
            "scalar": 0.75,
            "flat_tensor": flat_tensor,
            "nested": {"inner_tensor": nested_tensor},
        },
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    payload = pack(episode)
    assert len(payload.tensors) == 5
    reconstructed = unpack(payload)
    new_extra = reconstructed.policy_outputs[0].model_extra
    assert new_extra is not None
    assert new_extra["scalar"] == 0.75
    assert torch.equal(new_extra["flat_tensor"], flat_tensor)
    assert torch.equal(new_extra["nested"]["inner_tensor"], nested_tensor)


def test_dataclass_in_any_slot_round_trips_with_type() -> None:
    """Dataclasses inside replay payload / model_extra rebuild as their original type.

    Without the ``__dataclass_type__`` marker the unpacker would return a
    plain dict (the surrounding slot is ``dict[str, Any]`` so no type info
    is available at the unpack call site). Consumers that store typed
    payloads (e.g. AlpamayoPolicy stamps a ``ModelInput`` into replay
    payload) rely on that type surviving the
    transport.
    """
    pose = Pose(
        vec=Vec3(x=1.0, y=2.0, z=3.0), quat=Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
    )
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        replay_data=_replay_data({"pose": pose, "scalar": 7}),
        model_extra={"vec": Vec3(x=9.0, y=10.0, z=11.0)},
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    reconstructed = unpack(pack(episode))
    new_replay = reconstructed.policy_outputs[0].replay_data
    new_extra = reconstructed.policy_outputs[0].model_extra
    assert new_replay is not None and new_extra is not None
    assert isinstance(new_replay.payload["pose"], Pose)
    assert new_replay.payload["pose"] == pose
    assert new_replay.payload["scalar"] == 7
    assert isinstance(new_extra["vec"], Vec3)
    assert new_extra["vec"] == Vec3(x=9.0, y=10.0, z=11.0)


@pytest.mark.parametrize("reserved_key", ["__tensor_key__", "__dataclass_type__"])
def test_pack_rejects_user_dicts_with_reserved_manifest_keys(reserved_key: str) -> None:
    """User payload dicts cannot masquerade as NCCL manifest marker dicts."""
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        replay_data=_replay_data({reserved_key: "user value"}),
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    with pytest.raises(ValueError, match="reserved manifest keys"):
        pack(episode)


def test_unpack_rejects_dataclass_type_outside_allowed_prefix() -> None:
    """Reconstructing a non-alpagym type from the marker is refused at unpack."""
    from alpagym_runtime.transport.nccl.payload import WirePayload

    forged = {
        "scene_id": "s",
        "session_uuid": "u",
        "num_steps": 1,
        "policy_outputs": [
            {
                "chosen_xyz": {
                    "__tensor_key__": "x",
                    "shape": [1, 3],
                    "dtype": "torch.float32",
                },
                "chosen_quat": {
                    "__tensor_key__": "q",
                    "shape": [1, 4],
                    "dtype": "torch.float32",
                },
                "chosen_dt_us": {
                    "__tensor_key__": "t",
                    "shape": [1],
                    "dtype": "torch.int64",
                },
                "chosen_logprob": None,
                "replay_data": {
                    "replay_schema_version": 1,
                    "payload_schema": "test.replay.v1",
                    "payload_schema_version": 1,
                    "model_family": "alpamayo_r1",
                    "action_selection": {
                        "set_ix": 0,
                        "sample_ix": 1,
                    },
                    "old_logprob": None,
                    "payload": {
                        "evil": {
                            "__dataclass_type__": "subprocess.Popen",
                        },
                    },
                },
                "all_pred_xyz": None,
                "all_pred_quat": None,
                "model_extra": None,
            },
        ],
        "executed_ego_trajectory": {"poses": []},
        "route_waypoints": [],
        "metrics": None,
        "reward": None,
        "is_valid": True,
    }
    tensors = {
        "x": torch.zeros((1, 3), dtype=torch.float32),
        "q": torch.zeros((1, 4), dtype=torch.float32),
        "t": torch.zeros((1,), dtype=torch.int64),
    }
    payload = WirePayload(tensors=tensors, manifest=forged)
    with pytest.raises(ValueError, match="Refusing to reconstruct dataclass outside"):
        unpack(payload)


def test_dict_keys_with_dots_do_not_collide() -> None:
    """Tensor keys are generated independently from raw dict paths."""
    tensor_a = torch.tensor([1.0], dtype=torch.float32)
    tensor_b = torch.tensor([2.0], dtype=torch.float32)
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        replay_data=_replay_data({"a.b": tensor_a, "a": {"b": tensor_b}}),
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    reconstructed = unpack(pack(episode))
    replay_data = reconstructed.policy_outputs[0].replay_data
    assert replay_data is not None
    assert torch.equal(replay_data.payload["a.b"], tensor_a)
    assert torch.equal(replay_data.payload["a"]["b"], tensor_b)


def test_numpy_leaves_are_json_safe() -> None:
    """Numpy arrays and scalars in Any slots become JSON-compatible values."""
    policy_output = PolicyOutput(
        chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        chosen_dt_us=torch.tensor([0], dtype=torch.int64),
        model_extra={
            "array": np.array([1.0, 2.0], dtype=np.float32),
            "scalar": np.float32(3.5),
        },
    )
    episode = EpisodeOutput(
        scene_id="s",
        session_uuid="u",
        num_steps=1,
        policy_outputs=(policy_output,),
    )
    reconstructed = unpack(pack(episode))
    model_extra = reconstructed.policy_outputs[0].model_extra
    assert model_extra == {"array": [1.0, 2.0], "scalar": pytest.approx(3.5)}


def test_unsupported_union_unpack_error_names_union() -> None:
    """Non-Optional unions fail with explicit context instead of tuple-unpack noise."""
    with pytest.raises(ValueError, match=r"Only Optional\[T\] unions are supported"):
        _unpack(1, int | float | None, {})
