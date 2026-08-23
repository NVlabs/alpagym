# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed identity checks for native Cosmos checkpoint continuation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alpagym_host.config import CosmosRLCheckpointResumeConfig


FORMAL_RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
COSMOS_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{14}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STEP_DIRECTORY_PATTERN = re.compile(r"^step_([1-9][0-9]*)$")
_RANK_FILE_PATTERNS = {
    "complete": re.compile(r"^\.rank_([0-9]+)_complete$"),
    "model": re.compile(r"^model_rank_([0-9]+)\.pth$"),
    "optimizer": re.compile(r"^optimizer_rank_([0-9]+)\.pth$"),
    "scheduler": re.compile(r"^scheduler_rank_([0-9]+)\.pth$"),
    "extra_info": re.compile(r"^extra_info_rank_([0-9]+)\.pth$"),
}


@dataclass(frozen=True)
class CheckpointFileIdentity:
    """One stable regular file included in a checkpoint-tree digest."""

    name: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CheckpointTreeSnapshot:
    """Canonical identity of one flat Cosmos ``step_N/policy`` directory."""

    tree_sha256: str
    file_count: int
    total_size_bytes: int
    files: tuple[CheckpointFileIdentity, ...]

    def file(self, name: str) -> CheckpointFileIdentity:
        matches = [identity for identity in self.files if identity.name == name]
        if len(matches) != 1:
            raise ValueError(f"checkpoint tree has no unique file {name!r}")
        return matches[0]


@dataclass(frozen=True)
class CheckpointCursor:
    """Rank-consistent logical training cursor stored by Cosmos."""

    step: int
    total_steps: int
    remain_samples_num: int
    is_final: bool


@dataclass(frozen=True)
class ValidatedCheckpointResumeSource:
    """Resolved prior-run source admitted by the typed resume contract."""

    checkpoint_path: Path
    prior_formal_run_dir: Path
    prior_postrun_path: Path
    prior_postrun_receipt_sha256: str
    ranks: tuple[int, ...]
    snapshot: CheckpointTreeSnapshot
    cursor: CheckpointCursor


def canonical_json_sha256(value: Any) -> str:
    """Hash the strict canonical JSON encoding used by formal-run receipts."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_tree_snapshot(path: str | Path) -> CheckpointTreeSnapshot:
    """Hash an exact native Cosmos policy directory without following links.

    Cosmos's native checkpoint layout is flat. Rejecting nested directories,
    symlinks, special files, hard-linked files, and unstable reads makes the
    configured digest an identity for the bytes the native loader will consume.
    """

    root = _canonical_existing_directory(path, label="checkpoint_path")
    root_before = os.lstat(root)
    identities: list[CheckpointFileIdentity] = []
    with os.scandir(root) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    if not entries:
        raise ValueError("checkpoint policy directory is empty")
    for entry in entries:
        entry_path = root / entry.name
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"checkpoint tree contains a symlink: {entry_path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"checkpoint tree must be flat regular files: {entry_path}"
            )
        data, stable = read_stable_regular_file(entry_path)
        identities.append(
            CheckpointFileIdentity(
                name=entry.name,
                size_bytes=stable.st_size,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    root_after = os.lstat(root)
    if _stable_metadata(root_before) != _stable_metadata(root_after):
        raise RuntimeError("checkpoint directory changed while it was hashed")
    payload = {
        "schema_id": "alpagym.cosmos_checkpoint_tree.v1",
        "files": [identity.to_dict() for identity in identities],
    }
    return CheckpointTreeSnapshot(
        tree_sha256=canonical_json_sha256(payload),
        file_count=len(identities),
        total_size_bytes=sum(identity.size_bytes for identity in identities),
        files=tuple(identities),
    )


def read_stable_regular_file(path: str | Path) -> tuple[bytes, os.stat_result]:
    """Read one single-link regular file through ``O_NOFOLLOW``."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"checkpoint artifact is not regular: {source}")
        if before.st_nlink != 1:
            raise ValueError(f"checkpoint artifact must have one hard link: {source}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        if after.st_nlink != 1:
            raise ValueError(f"checkpoint artifact hard-link count changed: {source}")
        if _stable_metadata(before) != _stable_metadata(after):
            raise RuntimeError(f"checkpoint artifact changed while reading: {source}")
        if len(data) != before.st_size:
            raise RuntimeError(f"checkpoint artifact was read short: {source}")
        path_after = os.lstat(source)
        if _stable_metadata(after) != _stable_metadata(path_after):
            raise RuntimeError(
                f"checkpoint artifact path changed while reading: {source}"
            )
        return data, after
    finally:
        os.close(descriptor)


def validate_checkpoint_resume_source(
    contract: CosmosRLCheckpointResumeConfig,
) -> ValidatedCheckpointResumeSource:
    """Validate every prior-run identity and return the exact native source."""

    _validate_enabled_contract_fields(contract)
    assert contract.checkpoint_path is not None
    assert contract.checkpoint_step is not None
    assert contract.prior_formal_run_id is not None
    assert contract.checkpoint_tree_sha256 is not None
    assert contract.prior_postrun_receipt_sha256 is not None
    checkpoint_path = _canonical_existing_directory(
        contract.checkpoint_path,
        label="cosmos.train.resume.checkpoint_path",
    )
    step_directory = checkpoint_path.parent
    match = _STEP_DIRECTORY_PATTERN.fullmatch(step_directory.name)
    if checkpoint_path.name != "policy" or match is None:
        raise ValueError("checkpoint_path must end in checkpoints/step_N/policy")
    path_step = int(match.group(1))
    if path_step != contract.checkpoint_step:
        raise ValueError(
            "checkpoint_path step does not match cosmos.train.resume.checkpoint_step"
        )
    checkpoints_directory = step_directory.parent
    timestamp_directory = checkpoints_directory.parent
    cosmos_directory = timestamp_directory.parent
    prior_run_dir = cosmos_directory.parent
    if (
        checkpoints_directory.name != "checkpoints"
        or COSMOS_TIMESTAMP_PATTERN.fullmatch(timestamp_directory.name) is None
        or cosmos_directory.name != "cosmos"
        or prior_run_dir.name != contract.prior_formal_run_id
        or FORMAL_RUN_ID_PATTERN.fullmatch(prior_run_dir.name) is None
    ):
        raise ValueError(
            "checkpoint_path is not inside the declared formal-run Cosmos layout"
        )

    snapshot = checkpoint_tree_snapshot(checkpoint_path)
    ranks = validate_native_checkpoint_files(snapshot)
    if snapshot.tree_sha256 != contract.checkpoint_tree_sha256:
        raise ValueError(
            "checkpoint tree SHA256 does not match cosmos.train.resume contract"
        )
    cursor = _validate_checkpoint_cursors(
        checkpoint_path=checkpoint_path,
        snapshot=snapshot,
        ranks=ranks,
        expected_step=contract.checkpoint_step,
    )

    postrun_path = prior_run_dir / "provenance" / "postrun.json"
    postrun_receipt_sha256 = _validate_prior_postrun_receipt(
        postrun_path,
        expected_receipt_sha256=contract.prior_postrun_receipt_sha256,
        prior_formal_run_dir=prior_run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_step=contract.checkpoint_step,
        checkpoint_snapshot=snapshot,
        checkpoint_ranks=ranks,
    )
    return ValidatedCheckpointResumeSource(
        checkpoint_path=checkpoint_path,
        prior_formal_run_dir=prior_run_dir,
        prior_postrun_path=postrun_path,
        prior_postrun_receipt_sha256=postrun_receipt_sha256,
        ranks=ranks,
        snapshot=snapshot,
        cursor=cursor,
    )


def cosmos_resume_value(contract: CosmosRLCheckpointResumeConfig) -> bool | str:
    """Translate the typed contract to Cosmos's bool/string resume field."""

    if not contract.enabled:
        _validate_disabled_contract_fields(contract)
        return False
    return str(validate_checkpoint_resume_source(contract).checkpoint_path)


def validate_disabled_checkpoint_resume_contract(
    contract: CosmosRLCheckpointResumeConfig,
) -> None:
    """Reject identity fields on an explicitly disabled contract."""

    _validate_disabled_contract_fields(contract)


def _validate_enabled_contract_fields(
    contract: CosmosRLCheckpointResumeConfig,
) -> None:
    if contract.enabled is not True:
        raise ValueError("checkpoint resume source validation requires enabled=true")
    required_strings = {
        "prior_formal_run_id": contract.prior_formal_run_id,
        "checkpoint_path": contract.checkpoint_path,
        "checkpoint_tree_sha256": contract.checkpoint_tree_sha256,
        "prior_postrun_receipt_sha256": contract.prior_postrun_receipt_sha256,
    }
    for name, value in required_strings.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"cosmos.train.resume.{name} must be a non-empty string")
    if FORMAL_RUN_ID_PATTERN.fullmatch(contract.prior_formal_run_id or "") is None:
        raise ValueError("cosmos.train.resume.prior_formal_run_id is invalid")
    for name, value in (
        ("checkpoint_tree_sha256", contract.checkpoint_tree_sha256),
        ("prior_postrun_receipt_sha256", contract.prior_postrun_receipt_sha256),
    ):
        if SHA256_PATTERN.fullmatch(value or "") is None:
            raise ValueError(f"cosmos.train.resume.{name} must be lowercase SHA256")
    for name, value in (
        ("checkpoint_step", contract.checkpoint_step),
        ("expected_next_training_step", contract.expected_next_training_step),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"cosmos.train.resume.{name} must be a positive integer")
    assert contract.checkpoint_step is not None
    if contract.expected_next_training_step != contract.checkpoint_step + 1:
        raise ValueError(
            "cosmos.train.resume.expected_next_training_step must equal "
            "checkpoint_step + 1"
        )


def _validate_disabled_contract_fields(
    contract: CosmosRLCheckpointResumeConfig,
) -> None:
    if contract.enabled is not False:
        raise ValueError("cosmos.train.resume.enabled must be a boolean")
    populated = [
        name
        for name in (
            "prior_formal_run_id",
            "checkpoint_step",
            "checkpoint_path",
            "checkpoint_tree_sha256",
            "prior_postrun_receipt_sha256",
            "expected_next_training_step",
        )
        if getattr(contract, name) is not None
    ]
    if populated:
        raise ValueError(
            "disabled cosmos.train.resume cannot retain identity fields: "
            + ", ".join(populated)
        )


def validate_native_checkpoint_files(
    snapshot: CheckpointTreeSnapshot,
) -> tuple[int, ...]:
    names = {identity.name for identity in snapshot.files}
    if "cosmos_config" not in names:
        raise ValueError("native Cosmos checkpoint is missing cosmos_config")
    unmatched = names - {"cosmos_config"}
    rank_sets: dict[str, set[int]] = {kind: set() for kind in _RANK_FILE_PATTERNS}
    for name in sorted(unmatched):
        matched = False
        for kind, pattern in _RANK_FILE_PATTERNS.items():
            rank_match = pattern.fullmatch(name)
            if rank_match is not None:
                rank_sets[kind].add(int(rank_match.group(1)))
                matched = True
                break
        if not matched:
            raise ValueError(f"native Cosmos checkpoint has an unexpected file: {name}")
    expected = rank_sets["complete"]
    if not expected or any(ranks != expected for ranks in rank_sets.values()):
        raise ValueError(
            "native Cosmos checkpoint rank files and completion markers disagree"
        )
    contiguous = set(range(max(expected) + 1))
    if expected != contiguous:
        raise ValueError("native Cosmos checkpoint ranks must be contiguous from zero")
    return tuple(sorted(expected))


def _validate_checkpoint_cursors(
    *,
    checkpoint_path: Path,
    snapshot: CheckpointTreeSnapshot,
    ranks: tuple[int, ...],
    expected_step: int,
) -> CheckpointCursor:
    """Load every sealed rank cursor and require one coherent logical state."""

    cursors: list[CheckpointCursor] = []
    for rank in ranks:
        filename = f"extra_info_rank_{rank}.pth"
        expected_file = snapshot.file(filename)
        data, metadata = read_stable_regular_file(checkpoint_path / filename)
        if (
            metadata.st_size != expected_file.size_bytes
            or hashlib.sha256(data).hexdigest() != expected_file.sha256
        ):
            raise RuntimeError("checkpoint extra-info changed after tree validation")
        payload = torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("native Cosmos checkpoint extra-info must be a mapping")
        required = {
            "rng_state",
            "step",
            "total_steps",
            "remain_samples_num",
            "is_final",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f"native Cosmos checkpoint extra-info is missing: {sorted(missing)}"
            )
        integer_values: dict[str, int] = {}
        for field_name in ("step", "total_steps", "remain_samples_num"):
            value = payload[field_name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"checkpoint extra-info {field_name} must be an integer"
                )
            integer_values[field_name] = value
        if integer_values["step"] != expected_step:
            raise ValueError("checkpoint extra-info step differs from resume contract")
        if integer_values["step"] <= 0:
            raise ValueError("checkpoint extra-info step must be positive")
        if integer_values["total_steps"] < integer_values["step"]:
            raise ValueError("checkpoint extra-info total_steps precedes its step")
        if integer_values["remain_samples_num"] < 0:
            raise ValueError(
                "checkpoint extra-info remaining samples must be nonnegative"
            )
        is_final = payload["is_final"]
        if not isinstance(is_final, bool):
            raise TypeError("checkpoint extra-info is_final must be a boolean")
        rng_state = payload["rng_state"]
        if not isinstance(rng_state, dict) or not {
            "torch",
            "numpy",
            "python",
        }.issubset(rng_state):
            raise ValueError(
                "checkpoint extra-info rng_state must contain torch, numpy, and python"
            )
        cursor = CheckpointCursor(
            step=integer_values["step"],
            total_steps=integer_values["total_steps"],
            remain_samples_num=integer_values["remain_samples_num"],
            is_final=is_final,
        )
        expected_final = (
            cursor.remain_samples_num == 0 and cursor.step == cursor.total_steps
        )
        if cursor.is_final is not expected_final:
            raise ValueError(
                "checkpoint extra-info is_final disagrees with its logical cursor"
            )
        cursors.append(cursor)

    first = cursors[0]
    if any(cursor != first for cursor in cursors[1:]):
        raise ValueError("native Cosmos checkpoint rank cursors disagree")
    return first


def _validate_prior_postrun_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
    prior_formal_run_dir: Path,
    checkpoint_path: Path,
    checkpoint_step: int,
    checkpoint_snapshot: CheckpointTreeSnapshot,
    checkpoint_ranks: tuple[int, ...],
) -> str:
    canonical = _canonical_existing_file(path, label="prior formal postrun receipt")
    data, _metadata = read_stable_regular_file(canonical)
    payload = json.loads(data.decode("utf-8", errors="strict"))
    if not isinstance(payload, dict):
        raise TypeError("prior formal postrun receipt must be a JSON object")
    if payload.get("schema_id") != "alpagym.formal_run_postrun.v2":
        raise ValueError("prior formal postrun receipt has an unsupported schema")
    stored = payload.get("receipt_sha256")
    if not isinstance(stored, str) or SHA256_PATTERN.fullmatch(stored) is None:
        raise ValueError("prior formal postrun receipt has an invalid self hash")
    body = dict(payload)
    del body["receipt_sha256"]
    if canonical_json_sha256(body) != stored:
        raise ValueError("prior formal postrun receipt self hash is invalid")
    if stored != expected_receipt_sha256:
        raise ValueError("prior formal postrun receipt SHA256 does not match contract")
    required_true = (
        "run_completed",
        "cleanup_succeeded",
        "runtime_ready_captured",
        "source_watch_clean",
        "sources_unchanged",
        "configs_unchanged",
        "formal_run_valid",
    )
    failed = [name for name in required_true if payload.get(name) is not True]
    if failed or payload.get("cleanup_failure") is not None:
        raise ValueError(
            "prior formal run is not valid for checkpoint continuation: "
            + ", ".join(failed or ["cleanup_failure"])
        )
    native_checkpoints = payload.get("native_checkpoints")
    if not isinstance(native_checkpoints, list):
        raise ValueError("prior formal postrun receipt has no native checkpoint seal")
    relative_policy_path = checkpoint_path.relative_to(prior_formal_run_dir).as_posix()
    matches = [
        item
        for item in native_checkpoints
        if isinstance(item, dict)
        and item.get("policy_relative_path") == relative_policy_path
        and item.get("step") == checkpoint_step
    ]
    if len(matches) != 1:
        raise ValueError(
            "prior formal postrun receipt has no unique matching checkpoint seal"
        )
    sealed = matches[0]
    expected_fields = {
        "cosmos_output_relative_path",
        "policy_relative_path",
        "step",
        "tree_sha256",
        "file_count",
        "total_size_bytes",
        "files",
        "ranks",
    }
    if set(sealed) != expected_fields:
        raise ValueError("prior formal checkpoint seal fields changed")
    expected_files = [identity.to_dict() for identity in checkpoint_snapshot.files]
    expected_identity = {
        "policy_relative_path": relative_policy_path,
        "step": checkpoint_step,
        "tree_sha256": checkpoint_snapshot.tree_sha256,
        "file_count": checkpoint_snapshot.file_count,
        "total_size_bytes": checkpoint_snapshot.total_size_bytes,
        "files": expected_files,
        "ranks": list(checkpoint_ranks),
    }
    for key, expected in expected_identity.items():
        if sealed.get(key) != expected:
            raise ValueError(f"prior formal checkpoint seal does not match live {key}")
    output_relative_path = sealed["cosmos_output_relative_path"]
    if not isinstance(output_relative_path, str) or not output_relative_path:
        raise ValueError("prior formal checkpoint output path seal is invalid")
    expected_output = (
        checkpoint_path.parents[2].relative_to(prior_formal_run_dir).as_posix()
    )
    if output_relative_path != expected_output:
        raise ValueError("prior formal checkpoint output path seal does not match")
    return stored


def _canonical_existing_directory(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = value.resolve(strict=True)
    if value != resolved:
        raise ValueError(f"{label} must be canonical and contain no symlink components")
    metadata = os.lstat(resolved)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    return resolved


def _canonical_existing_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    resolved = value.resolve(strict=True)
    if value != resolved:
        raise ValueError(f"{label} must be canonical and contain no symlink components")
    metadata = os.lstat(resolved)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = [
    "CheckpointFileIdentity",
    "CheckpointCursor",
    "CheckpointTreeSnapshot",
    "ValidatedCheckpointResumeSource",
    "canonical_json_sha256",
    "checkpoint_tree_snapshot",
    "cosmos_resume_value",
    "read_stable_regular_file",
    "validate_native_checkpoint_files",
    "validate_checkpoint_resume_source",
    "validate_disabled_checkpoint_resume_contract",
]
