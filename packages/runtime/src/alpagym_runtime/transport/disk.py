# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import redis
import torch

from alpagym_runtime.replay import parse_policy_replay_data
from alpagym_runtime.transport.nccl.payload import (
    TENSOR_KEY_MARKER,
    WirePayload,
    _pack,
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

_DISK_ARTIFACT_SCHEMA = "alpagym.disk_episode.v2"
_DISK_TENSOR_FORMAT = "torch.save.weights_only.v1"
_DISK_ARTIFACT_KEYS = {
    "artifact_schema",
    "episode_manifest",
    "manifest_sha256",
    "tensor_sidecar",
}
_SIDECAR_KEYS = {"filename", "format", "sha256", "size_bytes"}
_TENSOR_REF_KEYS = {TENSOR_KEY_MARKER, "shape", "dtype"}
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of a JSON-compatible manifest."""
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _new_tensor_sidecar_path(path: Path) -> Path:
    """Return a fresh sidecar path that can be atomically published before JSON."""
    return path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tensors.pt")


def _is_valid_sha256(value: object) -> bool:
    """Return whether ``value`` is one lowercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX_CHARS for character in value)
    )


def _validate_sidecar_filename(path: Path, filename: object) -> str:
    """Validate and return the artifact-owned sidecar basename."""
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Disk tensor sidecar filename must be a basename")
    prefix = f"{path.stem}."
    suffix = ".tensors.pt"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(
            f"Disk tensor sidecar filename {filename!r} does not belong to {path.name!r}"
        )
    token = filename[len(prefix) : -len(suffix)]
    if len(token) != 32 or any(
        character not in _SHA256_HEX_CHARS for character in token
    ):
        raise ValueError("Disk tensor sidecar filename has an invalid UUID token")
    return filename


def _collect_tensor_specs(
    value: Any,
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> None:
    """Collect and strictly validate tensor references in ``value``."""
    if isinstance(value, dict):
        if TENSOR_KEY_MARKER in value:
            if set(value) != _TENSOR_REF_KEYS:
                raise ValueError("Disk manifest tensor reference has invalid fields")
            key = value[TENSOR_KEY_MARKER]
            shape = value["shape"]
            dtype_name = value["dtype"]
            if not isinstance(key, str) or not key:
                raise ValueError("Disk manifest tensor reference has an invalid key")
            if key in specs:
                raise ValueError(f"Disk manifest tensor key {key!r} is duplicated")
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                raise ValueError(f"Disk manifest tensor {key!r} has an invalid shape")
            if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
                raise ValueError(f"Disk manifest tensor {key!r} has an invalid dtype")
            dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"Disk manifest tensor {key!r} has an unsupported dtype"
                )
            specs[key] = (tuple(shape), dtype)
            return
        for child in value.values():
            _collect_tensor_specs(child, specs)
        return
    if isinstance(value, list):
        for child in value:
            _collect_tensor_specs(child, specs)


def _load_tensor_sidecar(
    path: Path,
    descriptor: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Load a sidecar only after validating its descriptor, digest, and tensor ABI."""
    if set(descriptor) != _SIDECAR_KEYS:
        raise ValueError("Disk tensor sidecar descriptor has invalid fields")
    if descriptor["format"] != _DISK_TENSOR_FORMAT:
        raise ValueError(
            f"Unsupported disk tensor sidecar format: {descriptor['format']!r}"
        )
    filename = _validate_sidecar_filename(path, descriptor["filename"])
    expected_sha256 = descriptor["sha256"]
    if not _is_valid_sha256(expected_sha256):
        raise ValueError("Disk tensor sidecar has an invalid SHA-256 digest")
    expected_size = descriptor["size_bytes"]
    if type(expected_size) is not int or expected_size < 0:
        raise ValueError("Disk tensor sidecar has an invalid byte size")

    sidecar_path = path.parent / filename
    if sidecar_path.is_symlink():
        raise ValueError("Disk tensor sidecar must not be a symbolic link")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Disk tensor sidecar is missing: {sidecar_path}")

    digest = hashlib.sha256()
    actual_size = 0
    with sidecar_path.open("rb") as sidecar_file:
        while chunk := sidecar_file.read(1024 * 1024):
            digest.update(chunk)
            actual_size += len(chunk)
        if actual_size != expected_size:
            raise ValueError(
                f"Disk tensor sidecar size mismatch: {actual_size} != {expected_size}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Disk tensor sidecar SHA-256 mismatch")
        sidecar_file.seek(0)
        tensor_payload = torch.load(
            sidecar_file,
            map_location="cpu",
            weights_only=True,
        )

    if type(tensor_payload) is not dict:
        raise ValueError("Disk tensor sidecar payload must be a tensor dictionary")
    if any(
        not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
        for key, tensor in tensor_payload.items()
    ):
        raise ValueError("Disk tensor sidecar payload contains a non-tensor entry")

    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    _collect_tensor_specs(manifest, specs)
    if set(tensor_payload) != set(specs):
        raise ValueError("Disk tensor sidecar keys do not match the manifest")
    for key, tensor in tensor_payload.items():
        expected_shape, expected_dtype = specs[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Disk tensor {key!r} shape mismatch: "
                f"{tuple(tensor.shape)} != {expected_shape}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"Disk tensor {key!r} dtype mismatch: "
                f"{tensor.dtype} != {expected_dtype}"
            )
    return tensor_payload


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


def _owned_sidecar_path(path: Path) -> Path | None:
    """Return the sidecar owned by an existing valid v2 artifact, if any."""
    if not path.is_file():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            return None
        if artifact.get("artifact_schema") != _DISK_ARTIFACT_SCHEMA:
            return None
        descriptor = artifact.get("tensor_sidecar")
        if not isinstance(descriptor, dict):
            return None
        filename = _validate_sidecar_filename(path, descriptor.get("filename"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    return path.parent / filename


def _file_sha256(path: Path) -> tuple[int, str]:
    """Return ``(size_bytes, sha256)`` for one file."""
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()


def write_episode_json(path: Path, episode: EpisodeOutput) -> None:
    """Atomically publish an episode manifest plus a lossless tensor sidecar.

    The sidecar is published first under a fresh name and the JSON manifest is
    renamed last. Readers therefore observe either the previous complete
    artifact or the new complete artifact, never a partially written pair.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_sidecar = _owned_sidecar_path(path)

    tensors: dict[str, torch.Tensor] = {}
    manifest = _pack(
        episode,
        tensors,
        reject_empty_tensors=False,
        encode_bool_as_uint8=False,
    )
    if not isinstance(manifest, dict):
        raise TypeError("Disk episode manifest must be a dictionary")
    stored_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if tensor.layout != torch.strided:
            raise ValueError(
                f"Disk tensor sidecar only supports dense tensors; {key!r} "
                f"uses {tensor.layout}"
            )
        # Clone after the device copy so a small contiguous view cannot make
        # ``torch.save`` persist the unrelated remainder of its base storage.
        stored_tensors[key] = (
            tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
        )

    sidecar_path = _new_tensor_sidecar_path(path)
    tmp_token = uuid.uuid4().hex
    sidecar_tmp = sidecar_path.with_name(f".{sidecar_path.name}.{tmp_token}.tmp")
    manifest_tmp = path.with_name(f".{path.name}.{tmp_token}.tmp")
    sidecar_published = False
    try:
        torch.save(stored_tensors, sidecar_tmp)
        sidecar_size, sidecar_sha256 = _file_sha256(sidecar_tmp)
        artifact = {
            "artifact_schema": _DISK_ARTIFACT_SCHEMA,
            "episode_manifest": manifest,
            "manifest_sha256": _manifest_sha256(manifest),
            "tensor_sidecar": {
                "filename": sidecar_path.name,
                "format": _DISK_TENSOR_FORMAT,
                "sha256": sidecar_sha256,
                "size_bytes": sidecar_size,
            },
        }
        manifest_tmp.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(sidecar_tmp, sidecar_path)
        sidecar_published = True
        os.replace(manifest_tmp, path)
    except BaseException:
        sidecar_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        if sidecar_published:
            sidecar_path.unlink(missing_ok=True)
        raise

    if previous_sidecar is not None and previous_sidecar != sidecar_path:
        previous_sidecar.unlink(missing_ok=True)


def read_episode_json(handle: str | Path) -> EpisodeOutput:
    """Read a v2 sidecar artifact or a backward-compatible legacy JSON artifact."""
    path = Path(handle)
    artifact_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact_data, dict):
        raise ValueError("Disk episode artifact must be a JSON object")
    if "artifact_schema" not in artifact_data:
        return _episode_from_artifact_dict(artifact_data)
    if artifact_data["artifact_schema"] != _DISK_ARTIFACT_SCHEMA:
        raise ValueError(
            f"Unsupported disk episode schema: {artifact_data['artifact_schema']!r}"
        )
    if set(artifact_data) != _DISK_ARTIFACT_KEYS:
        raise ValueError("Disk episode artifact has invalid fields")

    manifest = artifact_data["episode_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("Disk episode manifest must be a JSON object")
    expected_manifest_sha256 = artifact_data["manifest_sha256"]
    if not _is_valid_sha256(expected_manifest_sha256):
        raise ValueError("Disk episode manifest has an invalid SHA-256 digest")
    if _manifest_sha256(manifest) != expected_manifest_sha256:
        raise ValueError("Disk episode manifest SHA-256 mismatch")

    descriptor = artifact_data["tensor_sidecar"]
    if not isinstance(descriptor, dict):
        raise ValueError("Disk tensor sidecar descriptor must be a JSON object")
    tensors = _load_tensor_sidecar(path, descriptor, manifest)
    episode = unpack(WirePayload(tensors=tensors, manifest=manifest))
    if not isinstance(episode, EpisodeOutput):
        raise TypeError("Disk tensor manifest did not reconstruct EpisodeOutput")
    return episode


class DiskEpisodeWriter:
    """Rollout-side disk egress: writes a JSON manifest plus tensor sidecar."""

    def __init__(self, artifacts_dir: Path):
        """Create a writer that writes artifacts under ``artifacts_dir``."""
        self._artifacts_dir = Path(artifacts_dir).resolve()

    def write(self, episode: EpisodeOutput) -> str:
        """Persist ``episode`` and return its JSON manifest path as the handle.

        The handle carries a fresh ``uuid4`` suffix so two episodes that share a
        ``(scene_id, session_uuid)`` cannot overwrite each other's artifact.
        """
        filename = f"{episode.scene_id}_{episode.session_uuid}_{uuid.uuid4().hex}.json"
        path = self._artifacts_dir / filename
        write_episode_json(path, episode)
        return str(path)

    def release(self, handle: str, reason: str) -> None:
        """Discard an artifact manifest and its owned tensor sidecar."""
        del reason
        path = Path(handle)
        sidecar_path = _owned_sidecar_path(path)
        path.unlink(missing_ok=True)
        if sidecar_path is not None:
            sidecar_path.unlink(missing_ok=True)

    def start_cleanup(self, redis_client: redis.Redis) -> None:
        """No-op: the disk writer has no out-of-band discard channel."""
        del redis_client

    def flush_pending_sends(self) -> None:
        """No-op: disk writes are synchronous, so nothing is ever pending."""

    def close(self) -> None:
        """No-op: disk writer holds no live resources."""
