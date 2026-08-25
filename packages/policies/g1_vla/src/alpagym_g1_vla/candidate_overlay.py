# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, content-addressed rollout overlays for trained G1 VLA candidates.

The original VLA delivery remains the only architecture/source/processor
authority.  A candidate contains only parameters that Cosmos is allowed to
train, plus a manifest binding those tensors to the exact attested base bundle
and the Cosmos optimizer step that produced them.  A Cosmos resume directory is
therefore provenance for an overlay, never a model root accepted by rollout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
import errno
from contextlib import ExitStack, contextmanager
from ctypes import CDLL, c_char_p, c_int, get_errno
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from alpagym_host.checkpoint_resume import (
    canonical_json_sha256,
    checkpoint_tree_snapshot,
    read_stable_regular_file,
)
from alpagym_g1_vla.provenance import (
    VlaBundleProfile,
    file_sha256,
    vla_bundle_profile_for_model_root,
)
from alpagym_runtime.policies.registry import PolicyCheckpointExportContext

CANDIDATE_OVERLAY_SCHEMA = "alpagym.g1_vla_candidate_overlay.v1"
CANDIDATE_MANIFEST_FILENAME = "candidate_manifest.json"
CANDIDATE_WEIGHTS_FILENAME = "trainable_overlay.safetensors"
BASE_ATTESTED_SOURCE = "base_attested"
CANDIDATE_OVERLAY_SOURCE = "candidate_overlay"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
_COSMOS_TIMESTAMP = re.compile(r"^[0-9]{14}$")
_STEP_NAME = re.compile(r"^step_([1-9][0-9]*)$")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_ACTION_HEADER_PREFIX = "actor_critic.psi_model.action_header."
_CRITIC_PREFIX = "actor_critic.critic."
_REQUIRED_TRAINABLE_PREFIXES = (
    _ACTION_HEADER_PREFIX,
    _CRITIC_PREFIX,
)
_OVERLAY_PREFIXES = (
    *_REQUIRED_TRAINABLE_PREFIXES,
    "actor_critic.normalizer.",
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_schema",
        "base_bundle",
        "candidate_sha256",
        "source_training_state",
        "tensors",
        "weights",
    }
)


@dataclass(frozen=True)
class CandidateOverlay:
    """One fully verified candidate directory and its immutable identity."""

    root: Path
    candidate_sha256: str
    source_training_state: Mapping[str, Any]
    weights_path: Path
    weights_sha256: str
    weights_size_bytes: int
    tensor_metadata: tuple[Mapping[str, Any], ...]
    base_profile: VlaBundleProfile
    source_formal_run_id: str
    postrun_receipt_sha256: str
    native_checkpoint_path: Path
    native_checkpoint_tree_sha256: str

    def artifact_identity(self) -> dict[str, Any]:
        """Return the JSON identity stamped into qualification artifacts."""
        return {
            "source_kind": CANDIDATE_OVERLAY_SOURCE,
            "base_bundle": _base_bundle_identity(self.base_profile),
            "candidate_overlay": {
                "path": str(self.root),
                "candidate_sha256": self.candidate_sha256,
                "weights_sha256": self.weights_sha256,
                "source_training_state": dict(self.source_training_state),
                "source_formal_run_id": self.source_formal_run_id,
                "postrun_receipt_sha256": self.postrun_receipt_sha256,
                "formal_run_valid": True,
                "native_checkpoint_path": str(self.native_checkpoint_path),
                "native_checkpoint_tree_sha256": (self.native_checkpoint_tree_sha256),
            },
        }


def export_model_checkpoint(
    model: torch.nn.Module,
    destination: Path,
    context: PolicyCheckpointExportContext,
) -> None:
    """Atomically export all and only trainable G1 VLA parameters.

    ``destination`` must be Cosmos's ``safetensors/step_N`` path. The manifest
    binds the live training step and resolved run config. The separately saved
    Cosmos resume bundle is only for trainer continuation; it is not represented
    as the byte source of this independently exported overlay.
    """
    from alpagym_g1_vla.cosmos_model import VlaPsiPPOModel

    if not isinstance(model, VlaPsiPPOModel):
        raise TypeError(
            "G1 VLA candidate export requires VlaPsiPPOModel, got "
            f"{type(model).__name__}"
        )
    destination = Path(destination).expanduser()
    if not destination.is_absolute():
        raise ValueError("G1 VLA candidate destination must be absolute")
    step = _step_from_candidate_root(destination)
    if not isinstance(context, PolicyCheckpointExportContext):
        raise TypeError("G1 VLA candidate export context has an unexpected type")
    if context.training_step != step:
        raise ValueError(
            "G1 VLA candidate path step does not match export training_step"
        )
    if destination.parent.name != "safetensors":
        raise ValueError("G1 VLA candidate destination must use .../safetensors/step_N")
    cosmos_output_dir = destination.parent.parent
    cosmos_root = cosmos_output_dir.parent
    formal_run_dir = cosmos_root.parent
    if (
        cosmos_root.name != "cosmos"
        or not cosmos_output_dir.name
        or formal_run_dir.name != context.cosmos_run_id
    ):
        raise ValueError(
            "G1 VLA candidate destination does not match the formal cosmos_run_id"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable G1 VLA candidate: {destination}"
        )

    source_bundle = getattr(model, "source_bundle", None)
    profile = getattr(source_bundle, "profile", None)
    if not isinstance(profile, VlaBundleProfile):
        raise TypeError("G1 VLA candidate model has no attested source-bundle profile")
    config = getattr(model, "config", None)
    if getattr(config, "bundle_model_id", None) != profile.model_id:
        raise ValueError("G1 VLA candidate model/base profile identity changed")

    overlay_tensors = _overlay_tensor_map(model)
    cpu_state = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in overlay_tensors.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("G1 VLA candidate parent must not be a symlink")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        weights_path = temporary / CANDIDATE_WEIGHTS_FILENAME
        save_file(cpu_state, weights_path)
        weights_sha256 = file_sha256(weights_path)
        tensor_metadata = _read_safetensors_metadata(weights_path)
        unsigned_manifest: dict[str, Any] = {
            "artifact_schema": CANDIDATE_OVERLAY_SCHEMA,
            "base_bundle": _base_bundle_identity(profile),
            "source_training_state": {
                "kind": "cosmos_live_optimizer_state",
                "training_step": context.training_step,
                "total_training_steps": context.total_training_steps,
                "optimizer_steps_applied": context.optimizer_steps_applied,
                "resolved_config_sha256": context.resolved_config_sha256,
                "cosmos_run_id": context.cosmos_run_id,
            },
            "tensors": list(tensor_metadata),
            "weights": {
                "filename": CANDIDATE_WEIGHTS_FILENAME,
                "sha256": weights_sha256,
                "size_bytes": weights_path.stat().st_size,
            },
        }
        candidate_sha256 = _canonical_manifest_sha256(unsigned_manifest)
        manifest = {**unsigned_manifest, "candidate_sha256": candidate_sha256}
        manifest_path = temporary / CANDIDATE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(weights_path, 0o444)
        os.chmod(manifest_path, 0o444)
        os.chmod(temporary, 0o555)
        _rename_directory_noreplace(temporary, destination)
    except BaseException:
        if temporary.exists():
            os.chmod(temporary, 0o700)
            for child in temporary.iterdir():
                if child.is_file() and not child.is_symlink():
                    os.chmod(child, 0o600)
            shutil.rmtree(temporary)
        raise


def resolve_inference_source(
    run_config: Any,
) -> tuple[CandidateOverlay | None, dict[str, Any]]:
    """Resolve the explicit base/candidate qualification source, fail closed."""
    model_root = (
        Path(str(run_config.policy.model.path)).expanduser().resolve(strict=False)
    )
    profile = vla_bundle_profile_for_model_root(model_root)
    bundle_config = run_config.policy.model.bundle_config
    if not isinstance(bundle_config, Mapping):
        raise TypeError("VLA policy.model.bundle_config must be a mapping")
    source_kind = bundle_config.get("qualification_model_source")
    candidate_value = bundle_config.get("candidate_overlay_path")
    expected_candidate_sha256 = bundle_config.get("expected_candidate_sha256")
    if source_kind == BASE_ATTESTED_SOURCE:
        if candidate_value not in (None, "") or expected_candidate_sha256 not in (
            None,
            "",
        ):
            raise ValueError(
                "qualification_model_source=base_attested forbids "
                "candidate_overlay_path and expected_candidate_sha256"
            )
        return None, {
            "source_kind": BASE_ATTESTED_SOURCE,
            "base_bundle": _base_bundle_identity(profile),
        }
    if source_kind != CANDIDATE_OVERLAY_SOURCE:
        raise ValueError(
            "VLA qualification_model_source must be explicitly "
            f"{BASE_ATTESTED_SOURCE!r} or {CANDIDATE_OVERLAY_SOURCE!r}"
        )
    if not isinstance(candidate_value, str) or not candidate_value:
        raise ValueError(
            "qualification_model_source=candidate_overlay requires "
            "candidate_overlay_path"
        )
    candidate_path = Path(candidate_value).expanduser()
    if not candidate_path.is_absolute():
        raise ValueError("VLA candidate_overlay_path must be absolute")
    expected_candidate_sha256 = _require_sha256(
        "expected_candidate_sha256", expected_candidate_sha256
    )
    if candidate_path.resolve(strict=False) == model_root:
        raise ValueError("VLA candidate overlay cannot replace policy.model.path")
    candidate = verify_candidate_overlay(candidate_path, expected_profile=profile)
    if candidate.candidate_sha256 != expected_candidate_sha256:
        raise ValueError(
            "G1 VLA candidate SHA256 differs from expected_candidate_sha256"
        )
    return candidate, candidate.artifact_identity()


def verify_candidate_overlay(
    candidate_root: str | Path,
    *,
    expected_profile: VlaBundleProfile,
) -> CandidateOverlay:
    """Verify one candidate manifest, weight file, and base binding."""
    raw_root = Path(candidate_root).expanduser()
    if raw_root.is_symlink():
        raise ValueError("G1 VLA candidate root must not be a symlink")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    step = _step_from_candidate_root(root)
    children = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    if any(path.is_symlink() for path in children):
        raise ValueError("G1 VLA candidate must not contain symlinks")
    expected_names = {CANDIDATE_MANIFEST_FILENAME, CANDIDATE_WEIGHTS_FILENAME}
    actual_names = {path.name for path in children}
    if actual_names != expected_names or any(not path.is_file() for path in children):
        raise ValueError(
            "G1 VLA candidate must contain exactly candidate_manifest.json and "
            "trainable_overlay.safetensors"
        )

    manifest_path = root / CANDIDATE_MANIFEST_FILENAME
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("G1 VLA candidate manifest is not valid JSON") from exc
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != _TOP_LEVEL_KEYS:
        raise ValueError("G1 VLA candidate manifest fields changed")
    if raw_manifest["artifact_schema"] != CANDIDATE_OVERLAY_SCHEMA:
        raise ValueError("G1 VLA candidate artifact_schema changed")
    candidate_sha256 = _require_sha256(
        "candidate_sha256", raw_manifest["candidate_sha256"]
    )
    unsigned = dict(raw_manifest)
    del unsigned["candidate_sha256"]
    actual_candidate_sha256 = _canonical_manifest_sha256(unsigned)
    if candidate_sha256 != actual_candidate_sha256:
        raise ValueError(
            "G1 VLA candidate manifest SHA256 mismatch: expected "
            f"{candidate_sha256}, got {actual_candidate_sha256}"
        )
    if raw_manifest["base_bundle"] != _base_bundle_identity(expected_profile):
        raise ValueError("G1 VLA candidate does not bind the selected attested base")

    source_training_state = _validate_source_training_state(
        raw_manifest["source_training_state"], expected_step=step
    )
    weights = raw_manifest["weights"]
    if not isinstance(weights, dict) or set(weights) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("G1 VLA candidate weights metadata changed")
    if weights["filename"] != CANDIDATE_WEIGHTS_FILENAME:
        raise ValueError("G1 VLA candidate weights filename changed")
    if isinstance(weights["size_bytes"], bool) or not isinstance(
        weights["size_bytes"], int
    ):
        raise TypeError("G1 VLA candidate weights size must be an integer")
    weights_path = root / CANDIDATE_WEIGHTS_FILENAME
    weights_sha256 = _require_sha256("weights.sha256", weights["sha256"])
    tensor_metadata = _validate_tensor_metadata(raw_manifest["tensors"])
    with _open_verified_weights(
        root,
        expected_size=weights["size_bytes"],
        expected_sha256=weights_sha256,
        expected_metadata=tensor_metadata,
    ) as candidate_weights:
        formal_binding = _verify_formal_run_binding(
            candidate_root=root,
            source_training_state=source_training_state,
            candidate_weights=candidate_weights,
            candidate_tensor_metadata=tensor_metadata,
        )
    return CandidateOverlay(
        root=root,
        candidate_sha256=candidate_sha256,
        source_training_state=source_training_state,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        weights_size_bytes=weights["size_bytes"],
        tensor_metadata=tensor_metadata,
        base_profile=expected_profile,
        source_formal_run_id=formal_binding["source_formal_run_id"],
        postrun_receipt_sha256=formal_binding["postrun_receipt_sha256"],
        native_checkpoint_path=formal_binding["native_checkpoint_path"],
        native_checkpoint_tree_sha256=formal_binding["native_checkpoint_tree_sha256"],
    )


def apply_candidate_overlay(
    model: torch.nn.Module,
    candidate: CandidateOverlay,
) -> CandidateOverlay:
    """Reverify and copy one candidate into the exact mutable model surface."""
    from alpagym_g1_vla.cosmos_model import VlaPsiPPOModel

    if not isinstance(model, VlaPsiPPOModel):
        raise TypeError(
            f"G1 VLA candidate load requires VlaPsiPPOModel, got {type(model).__name__}"
        )
    verified = verify_candidate_overlay(
        candidate.root,
        expected_profile=candidate.base_profile,
    )
    if verified.candidate_sha256 != candidate.candidate_sha256:
        raise ValueError("G1 VLA candidate changed between resolution and load")
    targets = _overlay_tensor_map(model)
    candidate_names = tuple(item["name"] for item in verified.tensor_metadata)
    if tuple(sorted(targets)) != candidate_names:
        raise ValueError(
            "G1 VLA candidate parameter set does not exactly match the mutable model"
        )
    with _open_verified_weights(
        verified.root,
        expected_size=verified.weights_size_bytes,
        expected_sha256=verified.weights_sha256,
        expected_metadata=verified.tensor_metadata,
    ) as weights:
        if tuple(sorted(weights.keys())) != candidate_names:
            raise ValueError("G1 VLA candidate safetensors keys changed")
        with torch.no_grad():
            for name in candidate_names:
                target = targets[name]
                source = weights.get_tensor(name)
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(f"G1 VLA candidate shape mismatch for {name}")
                _validate_inference_load_dtype(
                    name=name,
                    candidate_dtype=source.dtype,
                    model_dtype=target.dtype,
                )
                if not torch.isfinite(source).all():
                    raise ValueError(f"G1 VLA candidate tensor {name} is non-finite")
                converted = source.to(device=target.device, dtype=target.dtype)
                if not torch.isfinite(converted).all():
                    raise ValueError(
                        f"G1 VLA candidate tensor {name} became non-finite "
                        "during inference dtype conversion"
                    )
                target.copy_(converted)
    return verified


def _validate_inference_load_dtype(
    *,
    name: str,
    candidate_dtype: torch.dtype,
    model_dtype: torch.dtype,
) -> None:
    """Validate the one intentional master-to-rollout precision conversion.

    Cosmos persists the optimizer's FP32 master action-head weights in both the
    native checkpoint and the immutable candidate overlay.  The attested rollout
    architecture intentionally materializes that same action head in BF16 while
    retaining the numerical-stability critic in FP32.  Keep exact FP32 equality
    as the checkpoint trust boundary, then permit only this named FP32-to-BF16
    load conversion.  Any other namespace or dtype transition remains an error.
    """
    if candidate_dtype == model_dtype:
        return
    if (
        name.startswith(_ACTION_HEADER_PREFIX)
        and candidate_dtype == torch.float32
        and model_dtype == torch.bfloat16
    ):
        return
    raise ValueError(
        f"G1 VLA candidate dtype mismatch for {name}: "
        f"candidate={candidate_dtype}, model={model_dtype}"
    )


def _verify_formal_run_binding(
    *,
    candidate_root: Path,
    source_training_state: Mapping[str, Any],
    candidate_weights: Any,
    candidate_tensor_metadata: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    """Verify the finalized formal run and its matching native checkpoint."""
    step = _step_from_candidate_root(candidate_root)
    safetensors_dir = candidate_root.parent
    cosmos_output_dir = safetensors_dir.parent
    cosmos_dir = cosmos_output_dir.parent
    formal_run_dir = cosmos_dir.parent
    source_formal_run_id = source_training_state["cosmos_run_id"]
    if (
        safetensors_dir.name != "safetensors"
        or cosmos_dir.name != "cosmos"
        or _COSMOS_TIMESTAMP.fullmatch(cosmos_output_dir.name) is None
        or _FORMAL_RUN_ID.fullmatch(formal_run_dir.name) is None
        or formal_run_dir.name != source_formal_run_id
    ):
        raise ValueError(
            "G1 VLA candidate must use "
            "<formal_run>/cosmos/<timestamp>/safetensors/step_N"
        )

    postrun_path = formal_run_dir / "provenance" / "postrun.json"
    postrun_data, _postrun_metadata = read_stable_regular_file(postrun_path)
    try:
        postrun = json.loads(postrun_data.decode("utf-8", errors="strict"))
    except json.JSONDecodeError as exc:
        raise ValueError("G1 VLA formal postrun receipt is not valid JSON") from exc
    if not isinstance(postrun, dict):
        raise TypeError("G1 VLA formal postrun receipt must be a JSON object")
    if postrun.get("schema_id") != "alpagym.formal_run_postrun.v2":
        raise ValueError("G1 VLA candidate requires a v2 formal postrun receipt")
    postrun_receipt_sha256 = _require_sha256(
        "postrun receipt_sha256", postrun.get("receipt_sha256")
    )
    unsigned_postrun = dict(postrun)
    del unsigned_postrun["receipt_sha256"]
    if canonical_json_sha256(unsigned_postrun) != postrun_receipt_sha256:
        raise ValueError("G1 VLA formal postrun receipt self hash is invalid")
    required_true = (
        "run_completed",
        "cleanup_succeeded",
        "runtime_ready_captured",
        "source_watch_clean",
        "sources_unchanged",
        "configs_unchanged",
        "formal_run_valid",
    )
    failed = [name for name in required_true if postrun.get(name) is not True]
    if failed or postrun.get("cleanup_failure") is not None:
        raise ValueError(
            "G1 VLA candidate source formal run is invalid: "
            + ", ".join(failed or ["cleanup_failure"])
        )

    native_checkpoints = postrun.get("native_checkpoints")
    if not isinstance(native_checkpoints, list):
        raise TypeError("G1 VLA formal postrun native_checkpoints must be a list")
    expected_cosmos_output = cosmos_output_dir.relative_to(formal_run_dir).as_posix()
    checkpoint_relative_to_output = f"checkpoints/step_{step}/policy"
    expected_policy_relative = (
        (cosmos_output_dir / checkpoint_relative_to_output)
        .relative_to(formal_run_dir)
        .as_posix()
    )
    matching = [
        entry
        for entry in native_checkpoints
        if isinstance(entry, dict)
        and entry.get("cosmos_output_relative_path") == expected_cosmos_output
        and entry.get("policy_relative_path") == expected_policy_relative
        and entry.get("step") == step
    ]
    if len(matching) != 1:
        raise ValueError(
            "G1 VLA formal postrun receipt has no unique matching native checkpoint"
        )
    native_receipt = matching[0]
    expected_native_fields = {
        "cosmos_output_relative_path",
        "policy_relative_path",
        "step",
        "tree_sha256",
        "file_count",
        "total_size_bytes",
        "files",
        "ranks",
    }
    if set(native_receipt) != expected_native_fields:
        raise ValueError("G1 VLA formal native-checkpoint receipt fields changed")
    native_checkpoint_tree_sha256 = _require_sha256(
        "native checkpoint tree_sha256", native_receipt["tree_sha256"]
    )
    ranks = native_receipt["ranks"]
    if ranks != [0]:
        raise ValueError(
            "G1 VLA candidate verification requires one unsharded native rank"
        )

    native_checkpoint_path = cosmos_output_dir / checkpoint_relative_to_output
    snapshot = checkpoint_tree_snapshot(native_checkpoint_path)
    expected_files = [identity.to_dict() for identity in snapshot.files]
    if (
        native_receipt["tree_sha256"] != snapshot.tree_sha256
        or native_receipt["file_count"] != snapshot.file_count
        or native_receipt["total_size_bytes"] != snapshot.total_size_bytes
        or native_receipt["files"] != expected_files
    ):
        raise ValueError(
            "G1 VLA native checkpoint differs from its formal postrun receipt"
        )
    expected_native_names = {
        ".rank_0_complete",
        "cosmos_config",
        "extra_info_rank_0.pth",
        "model_rank_0.pth",
        "optimizer_rank_0.pth",
        "scheduler_rank_0.pth",
    }
    if {identity.name for identity in snapshot.files} != expected_native_names:
        raise ValueError("G1 VLA native checkpoint file set is incomplete")
    native_model_identity = snapshot.file("model_rank_0.pth")
    with _open_verified_native_model_state(
        native_checkpoint_path,
        expected_size=native_model_identity.size_bytes,
        expected_sha256=native_model_identity.sha256,
    ) as native_state:
        _verify_native_overlay_tensors(
            candidate_weights=candidate_weights,
            candidate_tensor_metadata=candidate_tensor_metadata,
            native_state=native_state,
        )
    return {
        "source_formal_run_id": source_formal_run_id,
        "postrun_receipt_sha256": postrun_receipt_sha256,
        "native_checkpoint_path": native_checkpoint_path,
        "native_checkpoint_tree_sha256": native_checkpoint_tree_sha256,
    }


@contextmanager
def _open_verified_native_model_state(
    checkpoint_root: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> Iterator[Mapping[str, Any]]:
    """Load the native model from the same verified inode used for hashing."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = os.open(checkpoint_root, directory_flags)
    try:
        model_fd = os.open("model_rank_0.pth", file_flags, dir_fd=root_fd)
        try:
            before = os.fstat(model_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_size
            ):
                raise ValueError("G1 VLA native model checkpoint metadata changed")
            if _fd_sha256(model_fd) != expected_sha256:
                raise ValueError("G1 VLA native model checkpoint SHA256 changed")
            descriptor_path = f"/proc/self/fd/{model_fd}"
            try:
                native_state = torch.load(
                    descriptor_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except Exception as exc:
                raise ValueError(
                    "G1 VLA native model checkpoint is not loadable"
                ) from exc
            if not isinstance(native_state, Mapping):
                raise TypeError("G1 VLA native model checkpoint must be a mapping")
            yield native_state
            after = os.fstat(model_fd)
            stable_before = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            stable_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if stable_after != stable_before:
                raise ValueError(
                    "G1 VLA native model checkpoint changed while being compared"
                )
        finally:
            os.close(model_fd)
    finally:
        os.close(root_fd)


def _verify_native_overlay_tensors(
    *,
    candidate_weights: Any,
    candidate_tensor_metadata: tuple[Mapping[str, Any], ...],
    native_state: Mapping[str, Any],
) -> None:
    """Require candidate actor, critic, and persisted normalizer state to match."""
    candidate_names = tuple(item["name"] for item in candidate_tensor_metadata)
    native_overlay = {
        name: tensor
        for name, tensor in native_state.items()
        if isinstance(name, str)
        and any(name.startswith(prefix) for prefix in _OVERLAY_PREFIXES)
    }
    if tuple(sorted(native_overlay)) != candidate_names:
        raise ValueError(
            "G1 VLA candidate tensor set differs from the native checkpoint"
        )
    for name in candidate_names:
        native = native_overlay[name]
        if not isinstance(native, torch.Tensor):
            raise TypeError(f"G1 VLA native checkpoint value {name} is not a tensor")
        candidate = candidate_weights.get_tensor(name)
        if (
            native.device.type == "meta"
            or tuple(native.shape) != tuple(candidate.shape)
            or native.dtype != candidate.dtype
            or not torch.isfinite(native).all()
            or not torch.isfinite(candidate).all()
            or not torch.equal(native, candidate)
        ):
            raise ValueError(
                f"G1 VLA candidate tensor {name} differs from native checkpoint"
            )


@contextmanager
def _open_verified_weights(
    root: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_metadata: tuple[Mapping[str, Any], ...],
) -> Iterator[Any]:
    """Open, verify, and consume candidate weights through one pinned inode.

    A pathname hash followed by a second pathname open has a substitution
    window.  Qualification instead opens the candidate directory and weights
    with ``O_NOFOLLOW``, hashes that descriptor, and points safetensors at the
    same live descriptor through Linux ``/proc/self/fd``.  The descriptor stays
    open until every tensor has been copied, then is hashed again so an in-place
    mutation also fails the qualification process.
    """
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        raise RuntimeError("G1 VLA candidate verification requires Linux /proc/self/fd")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = os.open(root, directory_flags)
    try:
        weights_fd = os.open(
            CANDIDATE_WEIGHTS_FILENAME,
            file_flags,
            dir_fd=root_fd,
        )
        try:
            before = os.fstat(weights_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("G1 VLA candidate weights must be a regular file")
            if before.st_size != expected_size:
                raise ValueError("G1 VLA candidate weights size changed")
            actual_sha256 = _fd_sha256(weights_fd)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "G1 VLA candidate weights SHA256 mismatch: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )

            descriptor_path = f"/proc/self/fd/{weights_fd}"
            descriptor_stat = os.stat(descriptor_path)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise RuntimeError(
                    "G1 VLA candidate descriptor no longer identifies its opened inode"
                )
            with ExitStack() as stack:
                try:
                    weights = stack.enter_context(
                        safe_open(
                            descriptor_path,
                            framework="pt",
                            device="cpu",
                        )
                    )
                    actual_metadata = _read_open_safetensors_metadata(weights)
                    if actual_metadata != expected_metadata:
                        raise ValueError("G1 VLA candidate tensor metadata changed")
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(
                        "G1 VLA candidate weights are not valid safetensors"
                    ) from exc
                yield weights

            after = os.fstat(weights_fd)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ) or _fd_sha256(weights_fd) != expected_sha256:
                raise ValueError(
                    "G1 VLA candidate weights changed while they were being loaded"
                )
        finally:
            os.close(weights_fd)
    finally:
        os.close(root_fd)


def _fd_sha256(fd: int) -> str:
    """Hash one already-open regular file without reopening its pathname."""
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while block := os.read(fd, 8 * 1024 * 1024):
        digest.update(block)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish an immutable directory without ever replacing a race winner.

    Linux filesystems with ``renameat2(RENAME_NOREPLACE)`` get an atomic
    directory publish. Filesystems such as Lustre that reject that flag use a
    manifest-last hardlink fallback so readers fail closed until publication
    is complete.
    """
    if sys.platform != "linux":
        raise RuntimeError("immutable G1 VLA candidate publish requires Linux")
    libc = CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("libc.renameat2 is required for no-replace publish")
    renameat2.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]
    renameat2.restype = c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite immutable G1 VLA candidate: {destination}",
            str(destination),
        )
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        try:
            destination.mkdir(mode=0o700)
        except FileExistsError as error:
            raise FileExistsError(
                error.errno,
                f"refusing to overwrite immutable G1 VLA candidate: {destination}",
                str(destination),
            ) from error
        try:
            children = sorted(
                source.iterdir(),
                key=lambda child: child.name == CANDIDATE_MANIFEST_FILENAME,
            )
            for child in children:
                if not child.is_file() or child.is_symlink():
                    raise ValueError(
                        "G1 VLA candidate staging directory contains a non-regular file"
                    )
                os.link(child, destination / child.name, follow_symlinks=False)
            os.chmod(destination, 0o555)
            os.chmod(source, 0o700)
            for child in children:
                child.unlink()
            source.rmdir()
            return
        except BaseException:
            os.chmod(destination, 0o700)
            for child in destination.iterdir():
                child.unlink()
            destination.rmdir()
            raise
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _overlay_tensor_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return complete state for the exact trainable actor+critic modules."""
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise ValueError("G1 VLA candidate model has no trainable parameters")
    unsupported = sorted(
        name
        for name in trainable
        if not any(name.startswith(prefix) for prefix in _REQUIRED_TRAINABLE_PREFIXES)
    )
    if unsupported:
        raise ValueError(
            "G1 VLA candidate model exposes unsupported trainable parameters: "
            f"{unsupported[:5]}"
        )
    for prefix in _REQUIRED_TRAINABLE_PREFIXES:
        if not any(name.startswith(prefix) for name in trainable):
            raise ValueError(
                f"G1 VLA candidate model is missing trainable namespace {prefix!r}"
            )
    state = model.state_dict(keep_vars=True)
    overlay = {
        name: tensor
        for name, tensor in state.items()
        if any(name.startswith(prefix) for prefix in _OVERLAY_PREFIXES)
    }
    if not set(trainable).issubset(overlay):
        raise ValueError("G1 VLA trainable parameters are absent from model state")
    if any(not isinstance(tensor, torch.Tensor) for tensor in overlay.values()):
        raise TypeError("G1 VLA candidate state must contain only tensors")
    if any(tensor.device.type == "meta" for tensor in overlay.values()):
        raise ValueError("G1 VLA candidate state cannot contain meta tensors")
    return dict(sorted(overlay.items()))


def _base_bundle_identity(profile: VlaBundleProfile) -> dict[str, Any]:
    """Return all content pins needed to disambiguate the immutable base."""
    return {
        "model_id": profile.model_id,
        "checkpoint_step": profile.checkpoint_step,
        "model_sha256": profile.model_sha256,
        "run_config_sha256": profile.run_config_sha256,
        "argv_sha256": profile.argv_sha256,
        "stats_sha256": profile.stats_sha256,
        "base_vlm_tree_sha256": profile.base_vlm_tree_sha256,
        "psi_source_tree_sha256": profile.psi_source_tree_sha256,
        "camera_profile": profile.camera_profile,
        "camera_logical_id": profile.camera_logical_id,
        "camera_image_format": profile.camera_image_format,
        "camera_contract_sha256": profile.camera_contract_sha256,
        "camera_source_resolution": list(profile.camera_source_resolution),
        "camera_preprocess_profile": profile.camera_preprocess_profile,
        "language_instruction": profile.language_instruction,
        "native_qualification_inference_steps": (
            profile.native_qualification_inference_steps
        ),
        "native_qualification_schedule_sha256": (
            profile.native_qualification_schedule_sha256
        ),
        "native_qualification_clip_normalized_actions": (
            profile.native_qualification_clip_normalized_actions
        ),
        "native_qualification_continuity_prefix_rows": (
            profile.native_qualification_continuity_prefix_rows
        ),
        "stats_id": profile.stats_id,
    }


def _validate_source_training_state(
    value: Any,
    *,
    expected_step: int,
) -> Mapping[str, Any]:
    expected_keys = {
        "kind",
        "training_step",
        "total_training_steps",
        "optimizer_steps_applied",
        "resolved_config_sha256",
        "cosmos_run_id",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("G1 VLA candidate source training-state fields changed")
    if value["kind"] != "cosmos_live_optimizer_state":
        raise ValueError("G1 VLA candidate source training-state kind changed")
    try:
        PolicyCheckpointExportContext(
            training_step=value["training_step"],
            total_training_steps=value["total_training_steps"],
            optimizer_steps_applied=value["optimizer_steps_applied"],
            resolved_config_sha256=value["resolved_config_sha256"],
            cosmos_run_id=value["cosmos_run_id"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("G1 VLA candidate source training state is invalid") from exc
    if value["training_step"] != expected_step:
        raise ValueError(
            "G1 VLA candidate source training step differs from directory step"
        )
    return dict(value)


def _step_from_candidate_root(root: Path) -> int:
    match = _STEP_NAME.fullmatch(root.name)
    if match is None:
        raise ValueError("G1 VLA candidate directory must be named step_N for N >= 1")
    return int(match.group(1))


def _canonical_manifest_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_safetensors_metadata(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            return _read_open_safetensors_metadata(weights)
    except Exception as exc:
        raise ValueError("G1 VLA candidate weights are not valid safetensors") from exc


def _read_open_safetensors_metadata(weights: Any) -> tuple[Mapping[str, Any], ...]:
    """Read tensor metadata from one already-open safetensors handle."""
    metadata: list[Mapping[str, Any]] = []
    for name in sorted(weights.keys()):
        view = weights.get_slice(name)
        metadata.append(
            {
                "name": name,
                "shape": [int(value) for value in view.get_shape()],
                "dtype": str(view.get_dtype()),
            }
        )
    if not metadata:
        raise ValueError("G1 VLA candidate weights contain no tensors")
    return tuple(metadata)


def _validate_tensor_metadata(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("G1 VLA candidate tensors metadata must be a non-empty list")
    normalized: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "shape", "dtype"}:
            raise ValueError("G1 VLA candidate tensor metadata fields changed")
        name = item["name"]
        shape = item["shape"]
        dtype = item["dtype"]
        if not isinstance(name, str) or not name:
            raise ValueError("G1 VLA candidate tensor name must be non-empty")
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise ValueError("G1 VLA candidate tensor shape is invalid")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("G1 VLA candidate tensor dtype is invalid")
        normalized.append({"name": name, "shape": shape, "dtype": dtype})
    names = [item["name"] for item in normalized]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("G1 VLA candidate tensor names must be unique and sorted")
    return tuple(normalized)


def _require_sha256(label: str, value: Any) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"G1 VLA candidate {label} must be a lowercase SHA-256")
    return digest
