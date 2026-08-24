# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed source and runtime identity for formal humanoid runs."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from alpagym_host.alpasim_wizard import wizard_compose_project
from alpagym_host.checkpoint_resume import (
    checkpoint_tree_snapshot,
    read_stable_regular_file,
    validate_native_checkpoint_files,
)


_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ISDIR = 0x40000000
_IN_NONBLOCK = getattr(os, "O_NONBLOCK", 0x800)
_IN_CLOEXEC = getattr(os, "O_CLOEXEC", 0x80000)
_INOTIFY_EVENT = struct.Struct("iIII")
_SOURCE_WATCH_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_Q_OVERFLOW
)
_SOURCE_WATCH_EVENT_NAMES = {
    _IN_MODIFY: "MODIFY",
    _IN_ATTRIB: "ATTRIB",
    _IN_CLOSE_WRITE: "CLOSE_WRITE",
    _IN_MOVED_FROM: "MOVED_FROM",
    _IN_MOVED_TO: "MOVED_TO",
    _IN_CREATE: "CREATE",
    _IN_DELETE: "DELETE",
    _IN_DELETE_SELF: "DELETE_SELF",
    _IN_MOVE_SELF: "MOVE_SELF",
    _IN_UNMOUNT: "UNMOUNT",
    _IN_Q_OVERFLOW: "Q_OVERFLOW",
    _IN_IGNORED: "IGNORED",
    _IN_ISDIR: "ISDIR",
}
_IGNORED_WATCH_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}
_REQUIRED_CRITICAL_ENVIRONMENT = {
    "ALPASIM_GRPC_ROOT",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONDONTWRITEBYTECODE",
}
_REQUIRED_HUMANOID_DESCRIPTOR_FIELDS = {
    "HumanoidSessionAbortRequest": {"session_uuid": 1},
    "HumanoidCameraImage": {
        "scene_fingerprint": 13,
        "model_signature_sha256": 14,
        "camera_to_world_sha256": 15,
        "renderer_binding_sha256": 16,
    },
}
_REQUIRED_HUMANOID_DESCRIPTOR_SERVICE_METHOD_INPUTS = {
    "HumanoidPolicyService": {
        "abort_session": "humanoid.HumanoidSessionAbortRequest",
    },
    "HumanoidDynamicsService": {
        "abort_session": "humanoid.HumanoidSessionAbortRequest",
    },
}
_REQUIRED_HUMANOID_GRPC_BINDINGS = {
    "policy_stub_abort_session",
    "dynamics_stub_abort_session",
    "policy_servicer_abort_session",
    "dynamics_servicer_abort_session",
}
_COSMOS_OUTPUT_DIRECTORY = re.compile(r"^[0-9]{14}$")
_COSMOS_CHECKPOINT_STEP = re.compile(r"^step_([1-9][0-9]*)$")
_FORMAL_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
_PPO_UPDATE_DIAGNOSTIC = re.compile(r"^step_([0-9]+)_rank_([0-9]+)\.json$")
_PPO_UPDATE_DIAGNOSTIC_STATES = frozenset({"pre_rejected", "post_rejected", "accepted"})
_PPO_CAPTURED_AT_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?\+00:00$"
)


def build_import_probe_receipt(
    *,
    command: list[str],
    environment: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict receipt for one no-sync formal import probe."""
    module_origins = payload.get("module_origins")
    descriptor_fields = payload.get("descriptor_fields")
    descriptor_services = payload.get("descriptor_services")
    grpc_bindings = payload.get("grpc_bindings")
    receipt = {
        "schema_id": "alpagym.formal_import_probe.v2",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "command": list(command),
        "command_sha256": _canonical_sha256(list(command)),
        "environment": dict(environment),
        "environment_sha256": _canonical_sha256(environment),
        "module_origins": module_origins,
        "descriptor_fields": descriptor_fields,
        "descriptor_services": descriptor_services,
        "grpc_bindings": grpc_bindings,
    }
    receipt["identity_sha256"] = _canonical_sha256(receipt)
    _validate_import_probe_identity(receipt)
    return receipt


@dataclass
class _WatchRule:
    """Filter inotify events for one watched directory or regular file."""

    path: Path
    kind: str = "directory"
    watch_all: bool = False
    exact_names: set[str] | None = None


class _EffectiveSourceWatch:
    """Kernel-backed, permanent evidence of effective-source mutations."""

    def __init__(self, *, file_descriptor: int, rules: dict[int, _WatchRule]) -> None:
        self._file_descriptor = file_descriptor
        self._rules = rules
        self._events: list[dict[str, Any]] = []
        self._health_failures: list[str] = []
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        repositories: tuple[RepositorySource, ...],
        exact_config_paths: tuple[Path, ...],
    ) -> _EffectiveSourceWatch:
        """Install directory and inode watches before source snapshotting."""
        repository_paths = {
            repository.name: _repository_effective_paths(repository)
            for repository in repositories
        }
        path_rules: dict[Path, _WatchRule] = {}
        for repository in repositories:
            for relative_path in repository_paths[repository.name]:
                parent = repository.root.joinpath(*relative_path.parts).parent
                while parent == repository.root or parent.is_relative_to(
                    repository.root
                ):
                    if parent.is_dir():
                        rule = path_rules.setdefault(parent, _WatchRule(path=parent))
                        rule.watch_all = True
                    if parent == repository.root:
                        break
                    parent = parent.parent

        normalized_config_paths = tuple(
            _normalized_absolute_path(config_path) for config_path in exact_config_paths
        )
        for resolved in normalized_config_paths:
            rule = path_rules.setdefault(
                resolved.parent,
                _WatchRule(path=resolved.parent),
            )
            if rule.exact_names is None:
                rule.exact_names = set()
            rule.exact_names.add(resolved.name)

        if not path_rules:
            raise ValueError("formal source watch resolved no effective directories")

        source_paths = {
            _normalized_absolute_path(repository.root.joinpath(*relative_path.parts))
            for repository in repositories
            for relative_path in repository_paths[repository.name]
        }
        config_paths = set(normalized_config_paths)
        file_descriptor = -1
        rules: dict[int, _WatchRule] = {}
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            inotify_init1 = libc.inotify_init1
            inotify_init1.argtypes = [ctypes.c_int]
            inotify_init1.restype = ctypes.c_int
            file_descriptor = int(inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC))
            if file_descriptor < 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), "inotify_init1")
            inotify_add_watch = libc.inotify_add_watch
            inotify_add_watch.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            inotify_add_watch.restype = ctypes.c_int
            for path, rule in sorted(path_rules.items(), key=lambda item: str(item[0])):
                watch_descriptor = int(
                    inotify_add_watch(
                        file_descriptor,
                        os.fsencode(path),
                        _SOURCE_WATCH_MASK,
                    )
                )
                if watch_descriptor < 0:
                    error_number = ctypes.get_errno()
                    raise OSError(error_number, os.strerror(error_number), str(path))
                rules[watch_descriptor] = rule

            regular_file_baselines: list[tuple[Path, tuple[int, ...]]] = []
            for path in sorted(source_paths | config_paths, key=str):
                try:
                    source_file_descriptor = _open_path_without_symlinks(
                        path,
                        final_flags=getattr(os, "O_PATH", os.O_RDONLY),
                    )
                except FileNotFoundError:
                    if path in config_paths:
                        raise
                    continue
                try:
                    metadata = os.fstat(source_file_descriptor)
                    if not stat.S_ISREG(metadata.st_mode):
                        if path in config_paths:
                            raise ValueError(
                                "formal provenance config must be a regular file: "
                                f"{path}"
                            )
                        continue
                    _require_single_link(metadata=metadata, path=path)
                    regular_file_baselines.append(
                        (path, _stable_file_metadata(metadata))
                    )
                finally:
                    os.close(source_file_descriptor)

            for path, baseline in regular_file_baselines:
                source_file_descriptor = _open_path_without_symlinks(
                    path,
                    final_flags=getattr(os, "O_PATH", os.O_RDONLY),
                )
                try:
                    before = os.fstat(source_file_descriptor)
                    if not stat.S_ISREG(before.st_mode):
                        raise RuntimeError(
                            "formal regular source changed type before its inode "
                            f"watch was installed: {path}"
                        )
                    _require_single_link(metadata=before, path=path)
                    if _stable_file_metadata(before) != baseline:
                        raise RuntimeError(
                            "formal source changed before its inode watch was "
                            f"installed: {path}"
                        )

                    watch_descriptor = int(
                        inotify_add_watch(
                            file_descriptor,
                            os.fsencode(f"/proc/self/fd/{source_file_descriptor}"),
                            _SOURCE_WATCH_MASK,
                        )
                    )
                    if watch_descriptor < 0:
                        error_number = ctypes.get_errno()
                        raise OSError(
                            error_number, os.strerror(error_number), str(path)
                        )
                    if watch_descriptor in rules:
                        raise RuntimeError(
                            f"formal source watch descriptor collision for {path}"
                        )
                    rules[watch_descriptor] = _WatchRule(path=path, kind="regular_file")

                    after = os.fstat(source_file_descriptor)
                    _require_single_link(metadata=after, path=path)
                    if _stable_file_metadata(after) != baseline:
                        raise RuntimeError(
                            "formal source changed while its inode watch was "
                            f"installed: {path}"
                        )
                    verification_descriptor = _open_path_without_symlinks(
                        path,
                        final_flags=getattr(os, "O_PATH", os.O_RDONLY),
                    )
                    try:
                        verification = os.fstat(verification_descriptor)
                        if not stat.S_ISREG(verification.st_mode):
                            raise RuntimeError(
                                "formal regular source changed type while its inode "
                                f"watch was installed: {path}"
                            )
                        _require_single_link(metadata=verification, path=path)
                        if _stable_file_metadata(verification) != baseline:
                            raise RuntimeError(
                                "formal source path changed while its inode watch was "
                                f"installed: {path}"
                            )
                    finally:
                        os.close(verification_descriptor)
                finally:
                    os.close(source_file_descriptor)
        except BaseException:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            raise
        return cls(file_descriptor=file_descriptor, rules=rules)

    def checkpoint(self, *, stage: str) -> dict[str, Any]:
        """Drain all queued events and return an immutable checkpoint body."""
        self._drain()
        if self._closed:
            self._health_failures.append("inotify file descriptor was closed")
        else:
            try:
                os.fstat(self._file_descriptor)
            except OSError as exc:
                self._health_failures.append(f"inotify fstat failed: {exc}")
        watched_directories = sorted(
            str(rule.path) for rule in self._rules.values() if rule.kind == "directory"
        )
        watched_regular_files = sorted(
            str(rule.path)
            for rule in self._rules.values()
            if rule.kind == "regular_file"
        )
        receipt = {
            "schema_id": "alpagym.formal_source_watch.v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "backend": "linux_inotify",
            "watch_mask": _SOURCE_WATCH_MASK,
            "watched_directory_count": len(watched_directories),
            "watched_directories_sha256": _canonical_sha256(watched_directories),
            "watched_regular_file_count": len(watched_regular_files),
            "watched_regular_files_sha256": _canonical_sha256(watched_regular_files),
            "healthy": not self._health_failures,
            "health_failures": list(self._health_failures),
            "relevant_event_count": len(self._events),
            "relevant_events": list(self._events),
            "clean": not self._health_failures and not self._events,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return receipt

    def close(self) -> None:
        """Close the inotify descriptor exactly once."""
        if self._closed:
            return
        self._closed = True
        os.close(self._file_descriptor)

    def _drain(self) -> None:
        """Drain the nonblocking inotify queue without losing overflow evidence."""
        if self._closed:
            return
        while True:
            try:
                data = os.read(self._file_descriptor, 1024 * 1024)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                    return
                self._health_failures.append(f"inotify read failed: {exc}")
                return
            if not data:
                self._health_failures.append("inotify returned unexpected EOF")
                return
            offset = 0
            while offset < len(data):
                if len(data) - offset < _INOTIFY_EVENT.size:
                    self._health_failures.append("truncated inotify event header")
                    return
                watch_descriptor, mask, cookie, name_length = (
                    _INOTIFY_EVENT.unpack_from(data, offset)
                )
                offset += _INOTIFY_EVENT.size
                name_bytes = data[offset : offset + name_length]
                if len(name_bytes) != name_length:
                    self._health_failures.append("truncated inotify event name")
                    return
                offset += name_length
                name = os.fsdecode(name_bytes.rstrip(b"\0")) if name_bytes else ""
                self._record_event(
                    watch_descriptor=watch_descriptor,
                    mask=mask,
                    cookie=cookie,
                    name=name,
                )

    def _record_event(
        self, *, watch_descriptor: int, mask: int, cookie: int, name: str
    ) -> None:
        """Permanently record one relevant event after strict filtering."""
        if mask & _IN_Q_OVERFLOW:
            path = None
        else:
            rule = self._rules.get(watch_descriptor)
            if rule is None:
                self._health_failures.append(
                    f"inotify event referenced unknown watch descriptor {watch_descriptor}"
                )
                return
            if name and _is_ignored_watch_name(name):
                return
            if (
                name
                and not rule.watch_all
                and (rule.exact_names is None or name not in rule.exact_names)
            ):
                return
            path = str(rule.path / name) if name else str(rule.path)
        self._events.append(
            {
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "path_annotation": path,
                "mask": mask,
                "mask_names": [
                    event_name
                    for event_mask, event_name in _SOURCE_WATCH_EVENT_NAMES.items()
                    if mask & event_mask
                ],
                "cookie": cookie,
            }
        )


@dataclass(frozen=True)
class RepositorySource:
    """One worktree whose exact effective source participates in a run."""

    name: str
    root: Path
    capture_ignored_generated_pb2: bool = False


class FormalRunProvenance:
    """Own immutable prelaunch, runtime-ready, and postrun provenance receipts."""

    def __init__(
        self,
        *,
        provenance_dir: Path,
        repositories: tuple[RepositorySource, ...],
        prelaunch_manifest: dict[str, Any],
        source_watch: _EffectiveSourceWatch,
    ) -> None:
        """Retain the frozen identities needed for runtime and postrun checks."""
        self.provenance_dir = provenance_dir
        self.repositories = repositories
        self.prelaunch_manifest = prelaunch_manifest
        self._source_watch = source_watch
        # The runtime receipt's self-hash detects accidental corruption, but it is
        # not a trust root: a writer could change the receipt and recompute that
        # hash. Retain the hash observed at creation in this lifecycle object so
        # finalization can reject even a consistently re-signed replacement.
        self._runtime_ready_receipt_sha256: str | None = None

    @classmethod
    def capture_prelaunch(
        cls,
        *,
        provenance_dir: Path,
        repositories: tuple[RepositorySource, ...],
        resolved_config_path: Path,
        cosmos_config_path: Path,
        critical_environment: dict[str, str],
    ) -> FormalRunProvenance:
        """Freeze effective dirty worktrees and host invocation before startup.

        The snapshot contains a binary tracked patch plus byte-for-byte copies of
        every non-ignored untracked file. AlpaSim additionally captures ignored
        generated protobuf modules imported by the runtime. A partially existing
        provenance directory is rejected rather than reused.
        """
        provenance_dir = provenance_dir.resolve()
        if provenance_dir.exists():
            raise FileExistsError(
                f"formal provenance directory already exists: {provenance_dir}"
            )
        provenance_dir.mkdir(parents=True)
        prelaunch_dir = provenance_dir / "prelaunch"
        prelaunch_dir.mkdir()

        resolved_repositories = tuple(
            RepositorySource(
                name=repository.name,
                root=_require_git_worktree_root(repository.root),
                capture_ignored_generated_pb2=(
                    repository.capture_ignored_generated_pb2
                ),
            )
            for repository in repositories
        )
        if len({repository.name for repository in resolved_repositories}) != len(
            resolved_repositories
        ):
            raise ValueError("formal provenance repository names must be unique")
        for repository in resolved_repositories:
            if provenance_dir == repository.root or provenance_dir.is_relative_to(
                repository.root
            ):
                raise ValueError(
                    "formal provenance directory must be outside every captured "
                    f"worktree: {repository.root}"
                )

        normalized_environment = _validate_critical_environment(critical_environment)
        source_watch = _EffectiveSourceWatch.start(
            repositories=resolved_repositories,
            exact_config_paths=(resolved_config_path, cosmos_config_path),
        )
        try:
            repository_manifests: dict[str, Any] = {}
            for repository in resolved_repositories:
                repository_manifests[repository.name] = _capture_repository(
                    repository=repository,
                    destination=prelaunch_dir / "repositories" / repository.name,
                )

            invocation = {
                "argv": list(sys.argv),
                "cwd": str(Path.cwd().resolve()),
                "python_executable": str(Path(sys.executable).resolve()),
            }
            invocation["identity_sha256"] = _canonical_sha256(invocation)
            config_artifacts: dict[str, Any] = {
                "resolved_config": _file_identity(resolved_config_path),
                "cosmos_config": _file_identity(cosmos_config_path),
            }
            config_artifacts["identity_sha256"] = _canonical_sha256(
                {
                    name: {
                        "filename": identity["filename"],
                        "sha256": identity["sha256"],
                        "size_bytes": identity["size_bytes"],
                    }
                    for name, identity in config_artifacts.items()
                    if name != "identity_sha256"
                }
            )
            portable_repositories = {
                name: {
                    "head": manifest["head"],
                    "tree": manifest["tree"],
                    "tracked_patch_sha256": manifest["tracked_patch"]["sha256"],
                    "untracked_manifest_sha256": manifest["untracked"][
                        "manifest_sha256"
                    ],
                    "ignored_generated_pb2_manifest_sha256": manifest[
                        "ignored_generated_pb2"
                    ]["manifest_sha256"],
                    "source_identity_sha256": manifest["source_identity_sha256"],
                }
                for name, manifest in repository_manifests.items()
            }
            watch_state = source_watch.checkpoint(stage="prelaunch")
            if not watch_state["clean"]:
                _write_json_exclusive(
                    provenance_dir / "source_watch_prelaunch_failure.json",
                    watch_state,
                )
                raise RuntimeError(
                    "effective source changed while formal prelaunch was captured"
                )
            manifest = {
                "schema_id": "alpagym.formal_run_prelaunch.v1",
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "repositories": repository_manifests,
                "invocation": invocation,
                "critical_environment": normalized_environment,
                "critical_environment_sha256": _canonical_sha256(
                    normalized_environment
                ),
                "source_watch": watch_state,
                "config_artifacts": config_artifacts,
                "portable_source_set_sha256": _canonical_sha256(portable_repositories),
            }
            _write_json_exclusive(prelaunch_dir / "manifest.json", manifest)
            _make_tree_read_only(prelaunch_dir)
            return cls(
                provenance_dir=provenance_dir,
                repositories=resolved_repositories,
                prelaunch_manifest=manifest,
                source_watch=source_watch,
            )
        except BaseException:
            source_watch.close()
            raise

    def capture_runtime_ready(
        self,
        *,
        wizard_log_dirs: tuple[Path, ...],
        workload_command: list[str],
        workload_kind: str = "cosmos_training",
        scene_store_root: Path,
        scene_cache_root: Path | None,
        runtime_cache_root: Path | None,
        import_probe: dict[str, Any],
        controller_release_root: Path | None = None,
    ) -> Path:
        """Freeze running Compose containers, images, mounts, and workload identity.

        Runtime and humanoid-dynamics containers must be running and must bind
        the snapshotted AlpaSim source/plugin directories and Humanoid worktree.
        The receipt is written only after every runtime passes these checks.
        """
        runtime_ready_path = self.provenance_dir / "runtime_ready.json"
        if runtime_ready_path.exists():
            raise FileExistsError(
                f"formal runtime receipt already exists: {runtime_ready_path}"
            )
        if not wizard_log_dirs:
            raise ValueError("formal runtime receipt requires at least one Wizard")
        if workload_kind not in {"cosmos_training", "qualification_rollout"}:
            raise ValueError("formal runtime receipt has an unsupported workload kind")
        if not workload_command or not all(
            isinstance(argument, str) and argument for argument in workload_command
        ):
            raise TypeError("formal workload command must be a non-empty string list")
        if workload_kind == "qualification_rollout":
            expected_command = [
                self.prelaunch_manifest["invocation"]["python_executable"],
                *self.prelaunch_manifest["invocation"]["argv"],
            ]
            if workload_command != expected_command:
                raise RuntimeError(
                    "formal qualification command differs from the captured host "
                    "invocation"
                )
        self._verify_critical_environment()
        entry_watch_state = self._record_source_watch_checkpoint(
            filename="source_watch_runtime_ready_entry.json",
            stage="runtime_ready_entry",
        )
        if not entry_watch_state["clean"]:
            raise RuntimeError(
                "formal source watch observed a mutation before runtime inspection"
            )
        repository_checks, config_checks = self._scan_live_identity()
        if not all(
            check["matches_prelaunch"] for check in repository_checks.values()
        ) or not all(check["matches_prelaunch"] for check in config_checks.values()):
            raise RuntimeError(
                "formal source or config changed between prelaunch snapshot and "
                "runtime-ready inspection"
            )
        repository_roots = {
            repository.name: repository.root for repository in self.repositories
        }
        alpasim_root = repository_roots["alpasim"]
        humanoid_root = repository_roots["humanoid"]
        validated_import_probe = _validate_import_probe(
            import_probe,
            alpagym_root=repository_roots["alpagym"],
            alpasim_root=alpasim_root,
            critical_environment=self.prelaunch_manifest["critical_environment"],
        )
        scene_cache_identity = (
            _directory_tree_identity(scene_cache_root)
            if scene_cache_root is not None
            else None
        )
        controller_release_identity = (
            _directory_tree_identity(
                controller_release_root,
                schema_id="alpagym.controller_release_tree.v1",
                label="controller release",
            )
            if controller_release_root is not None
            else None
        )
        runtime_receipts = [
            _capture_compose_runtime(
                wizard_log_dir=wizard_log_dir,
                alpasim_root=alpasim_root,
                humanoid_root=humanoid_root,
                scene_store_root=scene_store_root,
                scene_cache_root=scene_cache_root,
                runtime_cache_root=runtime_cache_root,
                controller_release_root=controller_release_root,
            )
            for wizard_log_dir in wizard_log_dirs
        ]
        if scene_cache_root is not None:
            assert scene_cache_identity is not None
            observed_scene_cache_identity = _directory_tree_identity(scene_cache_root)
            if (
                observed_scene_cache_identity["tree_sha256"]
                != scene_cache_identity["tree_sha256"]
            ):
                raise RuntimeError(
                    "formal scene cache changed during runtime-ready inspection"
                )
        if controller_release_root is not None:
            assert controller_release_identity is not None
            observed_controller_release_identity = _directory_tree_identity(
                controller_release_root,
                schema_id="alpagym.controller_release_tree.v1",
                label="controller release",
            )
            if (
                observed_controller_release_identity["tree_sha256"]
                != controller_release_identity["tree_sha256"]
            ):
                raise RuntimeError(
                    "formal controller release changed during runtime-ready inspection"
                )
        admission_watch_state = self._record_source_watch_checkpoint(
            filename="source_watch_runtime_ready.json",
            stage="runtime_ready_admission",
        )
        if not admission_watch_state["clean"]:
            raise RuntimeError(
                "formal source watch observed a mutation during runtime inspection"
            )
        receipt = {
            "schema_id": "alpagym.formal_run_runtime_ready.v4",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "workload_kind": workload_kind,
            "workload_command": list(workload_command),
            "workload_command_sha256": _canonical_sha256(list(workload_command)),
            "prelaunch_portable_source_set_sha256": self.prelaunch_manifest[
                "portable_source_set_sha256"
            ],
            "prelaunch_config_artifacts_identity_sha256": self.prelaunch_manifest[
                "config_artifacts"
            ]["identity_sha256"],
            "critical_environment_sha256": self.prelaunch_manifest[
                "critical_environment_sha256"
            ],
            "import_probe": validated_import_probe,
            "source_watch_receipt_sha256": admission_watch_state["receipt_sha256"],
            "scene_cache_identity": scene_cache_identity,
            "controller_release_identity": controller_release_identity,
            "runtimes": runtime_receipts,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        _write_json_exclusive(runtime_ready_path, receipt)
        self._runtime_ready_receipt_sha256 = receipt["receipt_sha256"]
        return runtime_ready_path

    def cleanup_compose_identity(self, compose_path: Path) -> tuple[str, str]:
        """Return the runtime-bound Compose SHA and project for exact cleanup."""
        if self._runtime_ready_receipt_sha256 is None:
            raise RuntimeError(
                "formal runtime-ready identity is unavailable for cleanup"
            )
        receipt = _read_runtime_ready_receipt(
            self.provenance_dir / "runtime_ready.json",
            expected_source_set_sha256=self.prelaunch_manifest[
                "portable_source_set_sha256"
            ],
            expected_config_identity_sha256=self.prelaunch_manifest["config_artifacts"][
                "identity_sha256"
            ],
            expected_critical_environment_sha256=self.prelaunch_manifest[
                "critical_environment_sha256"
            ],
            expected_receipt_sha256=self._runtime_ready_receipt_sha256,
        )
        resolved_path = compose_path.resolve(strict=True)
        matches = [
            runtime
            for runtime in receipt["runtimes"]
            if Path(runtime["compose_path_annotation"]).resolve(strict=True)
            == resolved_path
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"exact cleanup Compose path has no unique runtime binding: {compose_path}"
            )
        runtime = matches[0]
        if _file_identity(compose_path)["sha256"] != runtime["compose_sha256"]:
            raise RuntimeError(
                f"exact cleanup Compose bytes differ from runtime-ready: {compose_path}"
            )
        return runtime["compose_sha256"], runtime["compose_project"]

    def finalize(
        self,
        *,
        run_completed: bool,
        cleanup_error: BaseException | None = None,
    ) -> Path:
        """Rehash live sources and invalidate a successful run on any drift.

        The postrun receipt is always retained. If Cosmos completed but source or
        config bytes changed, this method raises only after recording the invalid
        status, leaving checkpoints and logs untouched for diagnosis.
        """
        postrun_path = self.provenance_dir / "postrun.json"
        try:
            try:
                if os.path.lexists(postrun_path):
                    raise FileExistsError(
                        f"formal postrun receipt already exists: {postrun_path}"
                    )
                self._verify_critical_environment()
                repository_checks, config_checks = self._scan_live_identity()
                runtime_ready = _read_runtime_ready_receipt(
                    self.provenance_dir / "runtime_ready.json",
                    expected_source_set_sha256=self.prelaunch_manifest[
                        "portable_source_set_sha256"
                    ],
                    expected_config_identity_sha256=self.prelaunch_manifest[
                        "config_artifacts"
                    ]["identity_sha256"],
                    expected_critical_environment_sha256=self.prelaunch_manifest[
                        "critical_environment_sha256"
                    ],
                    expected_receipt_sha256=self._runtime_ready_receipt_sha256,
                )
                expected_scene_cache = runtime_ready["scene_cache_identity"]
                if expected_scene_cache is not None:
                    observed_scene_cache = _directory_tree_identity(
                        Path(expected_scene_cache["path_annotation"])
                    )
                    if (
                        observed_scene_cache["tree_sha256"]
                        != expected_scene_cache["tree_sha256"]
                    ):
                        raise RuntimeError(
                            "formal scene cache changed after runtime admission"
                        )
                expected_controller_release = runtime_ready[
                    "controller_release_identity"
                ]
                if expected_controller_release is not None:
                    observed_controller_release = _directory_tree_identity(
                        Path(expected_controller_release["path_annotation"]),
                        schema_id="alpagym.controller_release_tree.v1",
                        label="controller release",
                    )
                    if (
                        observed_controller_release["tree_sha256"]
                        != expected_controller_release["tree_sha256"]
                    ):
                        raise RuntimeError(
                            "formal controller release changed after runtime admission"
                        )
                watch_state = self._record_source_watch_checkpoint(
                    filename="source_watch_postrun.json",
                    stage="postrun",
                )
            except BaseException as exc:
                self._write_finalize_failure_receipt(
                    run_completed=run_completed,
                    cleanup_error=cleanup_error,
                    failure=exc,
                )
                raise

            sources_unchanged = all(
                check["matches_prelaunch"] for check in repository_checks.values()
            )
            configs_unchanged = all(
                check["matches_prelaunch"] for check in config_checks.values()
            )
            try:
                native_checkpoints = _capture_native_checkpoint_seals(
                    self.provenance_dir.parent
                )
                ppo_update_diagnostic_receipts = _capture_ppo_update_diagnostic_seals(
                    self.provenance_dir.parent,
                    expected_resolved_config_sha256=self.prelaunch_manifest[
                        "config_artifacts"
                    ]["resolved_config"]["sha256"],
                    expected_resolved_config_path=Path(
                        self.prelaunch_manifest["config_artifacts"]["resolved_config"][
                            "path"
                        ]
                    ),
                    expected_resolved_config_size_bytes=self.prelaunch_manifest[
                        "config_artifacts"
                    ]["resolved_config"]["size_bytes"],
                )
            except BaseException as exc:
                self._write_finalize_failure_receipt(
                    run_completed=run_completed,
                    cleanup_error=cleanup_error,
                    failure=exc,
                )
                raise
            cleanup_succeeded = cleanup_error is None
            valid = (
                run_completed
                and cleanup_succeeded
                and watch_state["clean"]
                and sources_unchanged
                and configs_unchanged
            )
            receipt = {
                "schema_id": "alpagym.formal_run_postrun.v2",
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "run_completed": run_completed,
                "cleanup_succeeded": cleanup_succeeded,
                "cleanup_failure": _exception_identity(cleanup_error),
                "runtime_ready_captured": True,
                "runtime_ready_receipt_sha256": runtime_ready["receipt_sha256"],
                "source_watch_clean": watch_state["clean"],
                "source_watch_receipt_sha256": watch_state["receipt_sha256"],
                "sources_unchanged": sources_unchanged,
                "configs_unchanged": configs_unchanged,
                "formal_run_valid": valid,
                "native_checkpoints": native_checkpoints,
                "ppo_update_diagnostic_receipts": ppo_update_diagnostic_receipts,
                "repositories": repository_checks,
                "config_artifacts": config_checks,
            }
            receipt["receipt_sha256"] = _canonical_sha256(receipt)
            _write_json_exclusive(postrun_path, receipt)
            if run_completed and not valid:
                raise RuntimeError(
                    "formal run completed with provenance, source-watch, or cleanup "
                    "failure; checkpoints and logs were retained but the run is "
                    f"invalid: {postrun_path}"
                )
            return postrun_path
        finally:
            self._source_watch.close()

    def _verify_critical_environment(self) -> None:
        """Reject drift in environment inherited by Wizard and Cosmos."""
        observed = {
            name: os.environ.get(name)
            for name in sorted(_REQUIRED_CRITICAL_ENVIRONMENT)
        }
        if observed != self.prelaunch_manifest["critical_environment"]:
            raise RuntimeError(
                "formal critical subprocess environment differs from prelaunch"
            )

    def _record_source_watch_checkpoint(
        self, *, filename: str, stage: str
    ) -> dict[str, Any]:
        """Persist one exclusive source-watch checkpoint before using its state."""
        state = self._source_watch.checkpoint(stage=stage)
        _write_json_exclusive(self.provenance_dir / filename, state)
        return state

    def _scan_live_identity(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compare every live source and config artifact with prelaunch."""
        repository_checks: dict[str, Any] = {}
        for repository in self.repositories:
            current = _repository_identity(repository)
            expected = self.prelaunch_manifest["repositories"][repository.name]
            repository_checks[repository.name] = {
                "expected_source_identity_sha256": expected["source_identity_sha256"],
                "observed_source_identity_sha256": current["source_identity_sha256"],
                "matches_prelaunch": current["source_identity_sha256"]
                == expected["source_identity_sha256"],
                "observed": current,
            }

        config_checks: dict[str, Any] = {}
        for name, expected in self.prelaunch_manifest["config_artifacts"].items():
            if name == "identity_sha256":
                continue
            observed = _file_identity(Path(expected["path"]))
            config_checks[name] = {
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed["sha256"],
                "matches_prelaunch": observed["sha256"] == expected["sha256"],
            }

        return repository_checks, config_checks

    def _write_finalize_failure_receipt(
        self,
        *,
        run_completed: bool,
        cleanup_error: BaseException | None,
        failure: BaseException,
    ) -> None:
        """Persist an exclusive minimal invalid receipt before propagating failure."""
        postrun_path = self.provenance_dir / "postrun.json"
        failure_path = (
            self.provenance_dir / "postrun_failure.json"
            if os.path.lexists(postrun_path)
            else postrun_path
        )
        receipt = {
            "schema_id": "alpagym.formal_run_postrun_failure.v1",
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "run_completed": run_completed,
            "cleanup_succeeded": cleanup_error is None,
            "cleanup_failure": _exception_identity(cleanup_error),
            "formal_run_valid": False,
            "failure_type": type(failure).__name__,
            "failure_message": str(failure),
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        try:
            _write_json_exclusive(failure_path, receipt)
        except BaseException as write_failure:
            raise BaseExceptionGroup(
                "formal finalization and failure-receipt persistence both failed",
                [failure, write_failure],
            ) from None


def _capture_native_checkpoint_seals(run_dir: Path) -> list[dict[str, Any]]:
    """Hash every complete native Cosmos checkpoint retained by this run.

    The seal is part of the immutable postrun receipt. A later continuation or
    candidate qualification must match these exact bytes; merely supplying a
    freshly computed hash for a replaced checkpoint is insufficient.
    """

    canonical_run_dir = run_dir.resolve(strict=True)
    cosmos_root = canonical_run_dir / "cosmos"
    if not os.path.lexists(cosmos_root):
        return []
    if cosmos_root.is_symlink() or not cosmos_root.is_dir():
        raise ValueError("formal Cosmos output root must be a non-symlink directory")

    seals: list[dict[str, Any]] = []
    for output_dir in sorted(cosmos_root.iterdir(), key=lambda path: path.name):
        if _COSMOS_OUTPUT_DIRECTORY.fullmatch(output_dir.name) is None:
            continue
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("formal Cosmos output must be a non-symlink directory")
        checkpoints_dir = output_dir / "checkpoints"
        if not os.path.lexists(checkpoints_dir):
            continue
        if checkpoints_dir.is_symlink() or not checkpoints_dir.is_dir():
            raise ValueError("native checkpoint root must be a non-symlink directory")
        for step_dir in sorted(checkpoints_dir.iterdir(), key=lambda path: path.name):
            match = _COSMOS_CHECKPOINT_STEP.fullmatch(step_dir.name)
            if match is None:
                continue
            if step_dir.is_symlink() or not step_dir.is_dir():
                raise ValueError("native checkpoint step must be a directory")
            policy_dir = step_dir / "policy"
            if policy_dir.is_symlink() or not policy_dir.is_dir():
                raise ValueError(
                    f"native checkpoint has no complete policy directory: {step_dir}"
                )
            if policy_dir.resolve(strict=True) != policy_dir:
                raise ValueError("native checkpoint path must be canonical")
            snapshot = checkpoint_tree_snapshot(policy_dir)
            ranks = validate_native_checkpoint_files(snapshot)
            seals.append(
                {
                    "cosmos_output_relative_path": output_dir.relative_to(
                        canonical_run_dir
                    ).as_posix(),
                    "policy_relative_path": policy_dir.relative_to(
                        canonical_run_dir
                    ).as_posix(),
                    "step": int(match.group(1)),
                    "tree_sha256": snapshot.tree_sha256,
                    "file_count": snapshot.file_count,
                    "total_size_bytes": snapshot.total_size_bytes,
                    "files": [identity.to_dict() for identity in snapshot.files],
                    "ranks": list(ranks),
                }
            )
    return seals


def _capture_ppo_update_diagnostic_seals(
    run_dir: Path,
    *,
    expected_resolved_config_sha256: str,
    expected_resolved_config_path: Path,
    expected_resolved_config_size_bytes: int,
) -> list[dict[str, Any]]:
    """Validate and seal immutable PPO update-boundary diagnostics.

    Receipts are useful specifically when the trainer aborts, so this scan is
    independent of ``run_completed`` and always contributes to postrun.
    """

    canonical_run_dir = run_dir.resolve(strict=True)
    receipt_dir = canonical_run_dir / "artifacts" / "ppo_update_diagnostics"
    if not os.path.lexists(receipt_dir):
        return []
    if (
        _FORMAL_RUN_ID.fullmatch(canonical_run_dir.name) is None
        or canonical_run_dir != run_dir
    ):
        raise ValueError("PPO diagnostic seal requires a canonical formal run")
    canonical_config = expected_resolved_config_path.resolve(strict=True)
    if (
        canonical_config != expected_resolved_config_path
        or canonical_config.parent != canonical_run_dir
    ):
        raise ValueError("PPO diagnostic seal resolved-config path is invalid")
    expected_config_relative_path = canonical_config.relative_to(
        canonical_run_dir
    ).as_posix()
    config_bytes, config_metadata = read_stable_regular_file(canonical_config)
    if (
        config_metadata.st_size != expected_resolved_config_size_bytes
        or hashlib.sha256(config_bytes).hexdigest() != expected_resolved_config_sha256
    ):
        raise ValueError("PPO diagnostic seal resolved-config identity differs")
    try:
        resolved_config = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise ValueError("PPO diagnostic seal resolved config is invalid YAML") from exc
    if not isinstance(resolved_config, dict):
        raise TypeError("PPO diagnostic seal resolved config must be an object")
    cosmos = resolved_config.get("cosmos")
    train = cosmos.get("train") if isinstance(cosmos, dict) else None
    train_policy = train.get("train_policy") if isinstance(train, dict) else None
    on_policy = (
        train_policy.get("on_policy") if isinstance(train_policy, dict) else None
    )
    if on_policy is not None and not isinstance(on_policy, bool):
        raise TypeError("PPO diagnostic seal on_policy config must be boolean")
    expected_scene_ids: frozenset[str] | None = None
    expected_packed_rows: int | None = None
    expected_compact_optimizer_padding: bool | None = None
    rollout_seed_base: int | None = None
    train_batch_per_replica: int | None = None
    policy_replicas: int | None = None
    if on_policy is True:
        dataset = resolved_config.get("dataset")
        raw_scene_ids = dataset.get("scene_ids") if isinstance(dataset, dict) else None
        if (
            not isinstance(raw_scene_ids, list)
            or not raw_scene_ids
            or any(
                not isinstance(scene_id, str) or not scene_id
                for scene_id in raw_scene_ids
            )
            or len(set(raw_scene_ids)) != len(raw_scene_ids)
        ):
            raise ValueError("on-policy PPO seal dataset scene_ids are invalid")
        expected_scene_ids = frozenset(cast(list[str], raw_scene_ids))

        raw_expected_rows = resolved_config.get("expected_valid_steps")
        if (
            isinstance(raw_expected_rows, bool)
            or not isinstance(raw_expected_rows, int)
            or raw_expected_rows <= 0
        ):
            raise ValueError("on-policy PPO seal expected_valid_steps is invalid")
        expected_packed_rows = raw_expected_rows

        raw_train_batch = (
            train.get("train_batch_per_replica") if isinstance(train, dict) else None
        )
        launch = cosmos.get("launch") if isinstance(cosmos, dict) else None
        raw_policy_replicas = (
            launch.get("policy_replicas") if isinstance(launch, dict) else None
        )
        alpasim = resolved_config.get("alpasim")
        humanoid = alpasim.get("humanoid") if isinstance(alpasim, dict) else None
        raw_seed_base = (
            humanoid.get("rollout_seed_base") if isinstance(humanoid, dict) else None
        )
        for name, value in (
            ("train_batch_per_replica", raw_train_batch),
            ("policy_replicas", raw_policy_replicas),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"on-policy PPO seal {name} is invalid")
        if (
            isinstance(raw_seed_base, bool)
            or not isinstance(raw_seed_base, int)
            or not 0 <= raw_seed_base < 2**64
        ):
            raise ValueError("on-policy PPO seal rollout_seed_base is invalid")
        train_batch_per_replica = raw_train_batch
        policy_replicas = raw_policy_replicas
        rollout_seed_base = raw_seed_base
        expected_compact_optimizer_padding = (
            cosmos.get("mode") == "colocated" and policy_replicas == 1
        )

    if (
        receipt_dir.is_symlink()
        or not receipt_dir.is_dir()
        or receipt_dir.resolve(strict=True) != receipt_dir
    ):
        raise ValueError("PPO update diagnostic root must be a canonical directory")

    seals: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[int, int]] = set()
    seen_on_policy_completion_paths: set[str] = set()
    seen_on_policy_sessions: set[str] = set()
    seen_on_policy_seeds: set[int] = set()
    with os.scandir(receipt_dir) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    if entries and on_policy is not True:
        raise ValueError(
            "PPO v2 diagnostic sealing requires explicit on-policy training"
        )
    for entry in entries:
        match = _PPO_UPDATE_DIAGNOSTIC.fullmatch(entry.name)
        path = receipt_dir / entry.name
        if match is None:
            raise ValueError(f"unexpected PPO update diagnostic artifact: {path}")
        entry_metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(entry_metadata.st_mode) or entry.is_symlink():
            raise ValueError(f"PPO update diagnostic is not regular: {path}")
        data, metadata = read_stable_regular_file(path)
        if metadata.st_mode & 0o222:
            raise ValueError(f"PPO update diagnostic is writable: {path}")
        try:
            receipt = json.loads(data.decode("utf-8", errors="strict"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"PPO update diagnostic is invalid JSON: {path}") from exc
        _validate_ppo_update_diagnostic_receipt(
            receipt,
            expected_formal_run_root=canonical_run_dir,
            expected_behavior_weight_version=(
                int(match.group(1)) - 1 if on_policy is True else None
            ),
            expected_scene_ids=expected_scene_ids,
            expected_packed_rows_per_rollout=expected_packed_rows,
            expected_compact_optimizer_padding=expected_compact_optimizer_padding,
            expected_rollout_seeds=(
                frozenset(
                    rollout_seed_base
                    + (int(match.group(1)) - 1) * train_batch_per_replica
                    + offset
                    for offset in range(train_batch_per_replica)
                )
                if on_policy is True
                and policy_replicas == 1
                and rollout_seed_base is not None
                and train_batch_per_replica is not None
                else None
            ),
            expected_formal_run_id=canonical_run_dir.name,
            expected_resolved_config_sha256=expected_resolved_config_sha256,
            expected_resolved_config_relative_path=expected_config_relative_path,
            expected_resolved_config_size_bytes=(expected_resolved_config_size_bytes),
            expected_step=int(match.group(1)),
            expected_rank=int(match.group(2)),
        )
        bound_output = canonical_run_dir / receipt["cosmos_output_relative_path"]
        if (
            not bound_output.is_dir()
            or bound_output.is_symlink()
            or bound_output.resolve(strict=True) != bound_output
        ):
            raise ValueError("PPO update diagnostic bound Cosmos output is invalid")
        coordinate = (receipt["current_step"], receipt["rank"])
        if coordinate in seen_coordinates:
            raise ValueError("duplicate PPO update diagnostic step/rank coordinate")
        seen_coordinates.add(coordinate)
        if on_policy is True:
            for record in receipt["consumed_rollout_artifacts"]:
                completion = record["completion_relative_path"]
                session_uuid = record["session_uuid"]
                rollout_seed = record["rollout_seed"]
                if (
                    completion in seen_on_policy_completion_paths
                    or session_uuid in seen_on_policy_sessions
                    or rollout_seed in seen_on_policy_seeds
                ):
                    raise ValueError(
                        "on-policy PPO update reused rollout ownership across steps"
                    )
                seen_on_policy_completion_paths.add(completion)
                seen_on_policy_sessions.add(session_uuid)
                seen_on_policy_seeds.add(rollout_seed)
        seals.append(
            {
                "receipt_relative_path": path.relative_to(canonical_run_dir).as_posix(),
                "state": receipt["state"],
                "current_step": receipt["current_step"],
                "rank": receipt["rank"],
                "receipt_sha256": receipt["receipt_sha256"],
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    return seals


def _validate_ppo_update_diagnostic_receipt(
    receipt: Any,
    *,
    expected_formal_run_root: Path | None = None,
    expected_behavior_weight_version: int | None = None,
    expected_scene_ids: frozenset[str] | None = None,
    expected_packed_rows_per_rollout: int | None = None,
    expected_compact_optimizer_padding: bool | None = None,
    expected_rollout_seeds: frozenset[int] | None = None,
    expected_formal_run_id: str,
    expected_resolved_config_sha256: str,
    expected_resolved_config_relative_path: str,
    expected_resolved_config_size_bytes: int,
    expected_step: int,
    expected_rank: int,
) -> None:
    """Validate the strict schema, state invariants, bindings, and self hash."""

    if not isinstance(receipt, dict):
        raise TypeError("PPO update diagnostic receipt must be an object")
    expected_keys = {
        "schema_id",
        "captured_at_utc",
        "formal_run_id",
        "resolved_config_relative_path",
        "resolved_config_sha256",
        "resolved_config_size_bytes",
        "cosmos_output_relative_path",
        "rank",
        "current_step",
        "total_steps",
        "state",
        "received_rollouts",
        "trainable_rollouts",
        "sample_rows",
        "actor_sample_rows",
        "behavior_weight_versions",
        "consumed_rollout_artifacts",
        "consumed_rollout_batch_sha256",
        "is_master_replica",
        "checkpoint_requested",
        "boundary",
        "pre_update_metrics",
        "optimizer_metrics",
        "post_update_metrics",
        "rejection",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("PPO update diagnostic receipt has an unexpected schema")
    if receipt["schema_id"] != "alpagym.ppo_update_diagnostic.v2":
        raise ValueError("PPO update diagnostic schema_id is unsupported")
    if receipt["formal_run_id"] != expected_formal_run_id:
        raise ValueError("PPO update diagnostic formal-run binding differs")
    _validate_ppo_captured_at_utc(
        receipt["captured_at_utc"],
        formal_run_id=expected_formal_run_id,
    )
    if (
        receipt["resolved_config_sha256"] != expected_resolved_config_sha256
        or receipt["resolved_config_relative_path"]
        != expected_resolved_config_relative_path
    ):
        raise ValueError("PPO update diagnostic resolved-config binding differs")
    if (
        not isinstance(receipt["resolved_config_size_bytes"], int)
        or isinstance(receipt["resolved_config_size_bytes"], bool)
        or receipt["resolved_config_size_bytes"] != expected_resolved_config_size_bytes
    ):
        raise ValueError("PPO update diagnostic config size is invalid")
    output_parts = PurePosixPath(receipt["cosmos_output_relative_path"]).parts
    if (
        len(output_parts) != 2
        or output_parts[0] != "cosmos"
        or _COSMOS_OUTPUT_DIRECTORY.fullmatch(output_parts[1]) is None
    ):
        raise ValueError("PPO update diagnostic Cosmos output binding is invalid")

    for name, expected in (("current_step", expected_step), ("rank", expected_rank)):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"PPO update diagnostic {name} differs from filename")
    total_steps = receipt["total_steps"]
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < expected_step
    ):
        raise ValueError("PPO update diagnostic total_steps is invalid")
    for name in (
        "received_rollouts",
        "trainable_rollouts",
        "sample_rows",
        "actor_sample_rows",
    ):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"PPO update diagnostic {name} is invalid")
    if receipt["trainable_rollouts"] > receipt["received_rollouts"]:
        raise ValueError("PPO update diagnostic rollout counts are inconsistent")
    if receipt["trainable_rollouts"] == 0 or receipt["sample_rows"] == 0:
        raise ValueError("PPO update diagnostic optimizer batch is empty")
    if receipt["actor_sample_rows"] > receipt["sample_rows"]:
        raise ValueError("PPO update diagnostic actor row count is inconsistent")
    versions = receipt["behavior_weight_versions"]
    if not isinstance(versions, list) or any(
        isinstance(version, bool) or not isinstance(version, int) or version < 0
        for version in versions
    ):
        raise TypeError("PPO update diagnostic behavior versions are invalid")
    if versions != sorted(versions):
        raise ValueError("PPO update diagnostic behavior versions are not sorted")
    _validate_consumed_rollout_artifacts(
        receipt,
        expected_formal_run_root=expected_formal_run_root,
        expected_scene_ids=expected_scene_ids,
        expected_packed_rows_per_rollout=expected_packed_rows_per_rollout,
        expected_compact_optimizer_padding=expected_compact_optimizer_padding,
        expected_rollout_seeds=expected_rollout_seeds,
    )
    if (
        expected_behavior_weight_version is not None
        and versions
        != [expected_behavior_weight_version] * receipt["trainable_rollouts"]
    ):
        raise ValueError(
            "PPO update diagnostic behavior version differs from on-policy step"
        )
    if not isinstance(receipt["is_master_replica"], bool) or not isinstance(
        receipt["checkpoint_requested"], bool
    ):
        raise TypeError("PPO update diagnostic boolean fields are invalid")

    state = receipt["state"]
    if state not in _PPO_UPDATE_DIAGNOSTIC_STATES:
        raise ValueError("PPO update diagnostic state is unsupported")
    boundary = receipt["boundary"]
    if not isinstance(boundary, dict) or set(boundary) != {
        "optimizer_steps_applied",
        "scheduler_advanced",
        "checkpoint_started",
        "weight_sync_started",
    }:
        raise ValueError("PPO update diagnostic boundary is invalid")
    optimizer_steps = boundary["optimizer_steps_applied"]
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 0
        or any(
            boundary[name] is not False
            for name in (
                "scheduler_advanced",
                "checkpoint_started",
                "weight_sync_started",
            )
        )
    ):
        raise ValueError("PPO update diagnostic boundary values are invalid")
    pre_metrics = receipt["pre_update_metrics"]
    optimizer_metrics = receipt["optimizer_metrics"]
    post_metrics = receipt["post_update_metrics"]
    rejection = receipt["rejection"]
    if not isinstance(pre_metrics, dict):
        raise TypeError("PPO update diagnostic pre-update metrics must be an object")
    if state == "pre_rejected":
        if optimizer_metrics is not None or post_metrics is not None or optimizer_steps:
            raise ValueError("pre-rejected PPO diagnostic contains optimizer state")
    elif state == "accepted":
        if not isinstance(optimizer_metrics, dict) or not isinstance(
            post_metrics, dict
        ):
            raise TypeError(
                "accepted PPO diagnostic requires optimizer and post-update metrics"
            )
        if (
            optimizer_steps != 1
            or optimizer_metrics.get("train/optimizer_steps_applied") != 1
        ):
            raise ValueError(
                "accepted PPO diagnostic must record exactly one optimizer step"
            )
    else:
        if not isinstance(optimizer_metrics, dict):
            raise TypeError("post-rejected PPO diagnostic requires optimizer metrics")
        metric_optimizer_steps = optimizer_metrics.get("train/optimizer_steps_applied")
        if (
            isinstance(metric_optimizer_steps, bool)
            or not isinstance(metric_optimizer_steps, int)
            or metric_optimizer_steps not in {0, 1}
            or optimizer_steps != metric_optimizer_steps
        ):
            raise ValueError(
                "post-rejected PPO diagnostic optimizer boundary differs from metrics"
            )
        # A post-update replay can itself fail after the optimizer mutates the
        # policy.  The runtime restores the actor and writes ``null`` rather
        # than attaching stale candidate metrics to the restored live state.
        if post_metrics is not None and not isinstance(post_metrics, dict):
            raise TypeError(
                "post-rejected PPO diagnostic post-update metrics must be an "
                "object or null"
            )
    if state == "accepted":
        if rejection is not None:
            raise ValueError("accepted PPO diagnostic cannot contain a rejection")
    elif (
        not isinstance(rejection, dict)
        or set(rejection) != {"type", "message"}
        or not all(isinstance(value, str) for value in rejection.values())
    ):
        raise TypeError("rejected PPO diagnostic has no valid exception identity")

    stored_sha256 = receipt["receipt_sha256"]
    if not isinstance(stored_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", stored_sha256
    ):
        raise ValueError("PPO update diagnostic self hash is invalid")
    body = dict(receipt)
    del body["receipt_sha256"]
    if stored_sha256 != _canonical_sha256(body):
        raise ValueError("PPO update diagnostic self hash differs from its contents")


def _validate_ppo_captured_at_utc(value: Any, *, formal_run_id: str) -> None:
    """Require the exact UTC timestamp shape emitted by the trainer.

    This is a consistency check, not an authentication primitive.  Receipt
    authentication comes from the captured clean producer source plus
    re-reading the current formal-run artifacts below; the receipt's self hash
    alone is deliberately not treated as trusted evidence.
    """

    if not isinstance(value, str) or _PPO_CAPTURED_AT_UTC.fullmatch(value) is None:
        raise ValueError("PPO update diagnostic captured_at_utc is invalid")
    try:
        captured_at = datetime.fromisoformat(value)
        run_started_at = datetime.strptime(
            formal_run_id[:16],
            "%Y%m%dT%H%M%SZ",
        ).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("PPO update diagnostic captured_at_utc is invalid") from exc
    if captured_at.utcoffset() != timedelta(0) or captured_at < run_started_at:
        raise ValueError("PPO update diagnostic captured_at_utc is inconsistent")


def _validate_consumed_rollout_artifacts(
    receipt: dict[str, Any],
    *,
    expected_formal_run_root: Path | None,
    expected_scene_ids: frozenset[str] | None,
    expected_packed_rows_per_rollout: int | None,
    expected_compact_optimizer_padding: bool | None,
    expected_rollout_seeds: frozenset[int] | None,
) -> None:
    """Validate the exact ordered disk episodes consumed by one optimizer step."""

    records = receipt["consumed_rollout_artifacts"]
    trainable_rollouts = receipt["trainable_rollouts"]
    if not isinstance(records, list):
        raise TypeError("PPO consumed rollout artifacts must be a list")
    if len(records) != trainable_rollouts:
        raise ValueError(
            "PPO consumed rollout artifact count differs from trainable rollouts"
        )

    expected_record_keys = {
        "rollout_index",
        "transport_kind",
        "completion_relative_path",
        "episode_file_sha256",
        "episode_file_size_bytes",
        "episode_manifest_sha256",
        "tensor_sidecar",
        "session_uuid",
        "rollout_seed",
        "scene_id",
        "num_steps",
        "behavior_weight_version",
        "optimizer_sample_rows",
        "optimizer_actor_valid_rows",
    }
    expected_sidecar_keys = {"filename", "sha256", "size_bytes"}
    seen_sessions: set[str] = set()
    seen_seeds: set[int] = set()
    seen_completion_paths: set[PurePosixPath] = set()
    sample_rows = 0
    actor_rows = 0
    record_versions: list[int] = []
    for expected_index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict) or set(raw_record) != expected_record_keys:
            raise ValueError("PPO consumed rollout artifact has an unexpected schema")
        record = cast(dict[str, Any], raw_record)
        if record["rollout_index"] != expected_index or isinstance(
            record["rollout_index"], bool
        ):
            raise ValueError("PPO consumed rollout artifact order is invalid")
        if record["transport_kind"] != "disk_episode_v2":
            raise ValueError("PPO consumed rollout transport kind is invalid")

        completion = record["completion_relative_path"]
        if not isinstance(completion, str) or "\\" in completion:
            raise TypeError("PPO consumed rollout completion path is invalid")
        completion_path = PurePosixPath(completion)
        if (
            completion_path.is_absolute()
            or completion_path.parts in ((), (".",))
            or ".." in completion_path.parts
            or len(completion_path.parts) != 2
            or completion_path.parts[0] != "artifacts"
            or completion_path.suffix != ".json"
            or completion_path.as_posix() != completion
        ):
            raise ValueError("PPO consumed rollout completion path is unsafe")
        if completion_path in seen_completion_paths:
            raise ValueError("PPO consumed rollout completion path is duplicated")
        seen_completion_paths.add(completion_path)

        for name in ("episode_file_sha256", "episode_manifest_sha256"):
            if (
                not isinstance(record[name], str)
                or re.fullmatch(r"[0-9a-f]{64}", record[name]) is None
            ):
                raise ValueError(f"PPO consumed rollout {name} is invalid")
        episode_size = record["episode_file_size_bytes"]
        if (
            isinstance(episode_size, bool)
            or not isinstance(episode_size, int)
            or episode_size <= 0
        ):
            raise ValueError("PPO consumed rollout episode file size is invalid")

        raw_sidecar = record["tensor_sidecar"]
        if (
            not isinstance(raw_sidecar, dict)
            or set(raw_sidecar) != expected_sidecar_keys
        ):
            raise ValueError("PPO consumed rollout sidecar has an unexpected schema")
        sidecar = cast(dict[str, Any], raw_sidecar)
        sidecar_filename = sidecar["filename"]
        completion_stem = completion_path.stem
        if (
            not isinstance(sidecar_filename, str)
            or "\\" in sidecar_filename
            or PurePosixPath(sidecar_filename).name != sidecar_filename
            or re.fullmatch(
                rf"{re.escape(completion_stem)}\.[0-9a-f]{{32}}\.tensors\.pt",
                sidecar_filename,
            )
            is None
        ):
            raise ValueError("PPO consumed rollout sidecar filename is invalid")
        if (
            not isinstance(sidecar["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", sidecar["sha256"]) is None
        ):
            raise ValueError("PPO consumed rollout sidecar SHA-256 is invalid")
        sidecar_size = sidecar["size_bytes"]
        if (
            isinstance(sidecar_size, bool)
            or not isinstance(sidecar_size, int)
            or sidecar_size <= 0
        ):
            raise ValueError("PPO consumed rollout sidecar size is invalid")

        session_uuid = record["session_uuid"]
        if (
            not isinstance(session_uuid, str)
            or not session_uuid
            or session_uuid in seen_sessions
        ):
            raise ValueError("PPO consumed rollout session ownership is invalid")
        seen_sessions.add(session_uuid)
        rollout_seed = record["rollout_seed"]
        if (
            isinstance(rollout_seed, bool)
            or not isinstance(rollout_seed, int)
            or not 0 <= rollout_seed < 2**64
            or rollout_seed in seen_seeds
        ):
            raise ValueError("PPO consumed rollout seed ownership is invalid")
        seen_seeds.add(rollout_seed)
        if not isinstance(record["scene_id"], str) or not record["scene_id"]:
            raise ValueError("PPO consumed rollout scene is invalid")

        for name in (
            "num_steps",
            "optimizer_sample_rows",
            "optimizer_actor_valid_rows",
        ):
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"PPO consumed rollout {name} is invalid")
        if record["num_steps"] == 0 or record["optimizer_sample_rows"] == 0:
            raise ValueError("PPO consumed rollout has no optimizer samples")
        if record["optimizer_actor_valid_rows"] > record["optimizer_sample_rows"]:
            raise ValueError("PPO consumed rollout actor row count is inconsistent")
        behavior_version = record["behavior_weight_version"]
        if (
            isinstance(behavior_version, bool)
            or not isinstance(behavior_version, int)
            or behavior_version < 0
        ):
            raise TypeError("PPO consumed rollout behavior version is invalid")

        sample_rows += record["optimizer_sample_rows"]
        actor_rows += record["optimizer_actor_valid_rows"]
        record_versions.append(behavior_version)

        if expected_formal_run_root is not None:
            _validate_consumed_rollout_files(
                formal_run_root=expected_formal_run_root,
                completion_path=completion_path,
                record=record,
            )

    if sample_rows != receipt["sample_rows"]:
        raise ValueError("PPO consumed rollout sample rows do not conserve")
    if actor_rows != receipt["actor_sample_rows"]:
        raise ValueError("PPO consumed rollout actor rows do not conserve")
    if sorted(record_versions) != receipt["behavior_weight_versions"]:
        raise ValueError("PPO consumed rollout behavior versions differ")
    if expected_scene_ids is not None and any(
        record["scene_id"] not in expected_scene_ids for record in records
    ):
        raise ValueError("PPO consumed rollout scene differs from resolved config")
    if expected_packed_rows_per_rollout is not None:
        for record in records:
            num_steps = record["num_steps"]
            optimizer_rows = record["optimizer_sample_rows"]
            if num_steps > expected_packed_rows_per_rollout:
                raise ValueError(
                    "PPO consumed rollout real row count exceeds resolved T_pack"
                )
            if expected_compact_optimizer_padding is True:
                if optimizer_rows != num_steps:
                    raise ValueError(
                        "colocated single-policy PPO must compact padding before "
                        "the optimizer"
                    )
            elif not (num_steps <= optimizer_rows <= expected_packed_rows_per_rollout):
                raise ValueError(
                    "PPO consumed rollout optimizer rows fall outside real-row/T_pack "
                    "bounds"
                )
    if (
        expected_rollout_seeds is not None
        and {record["rollout_seed"] for record in records} != expected_rollout_seeds
    ):
        raise ValueError("PPO consumed rollout seeds differ from resolved schedule")

    batch_sha256 = receipt["consumed_rollout_batch_sha256"]
    if (
        not isinstance(batch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", batch_sha256) is None
    ):
        raise ValueError("PPO consumed rollout batch SHA-256 is invalid")
    if batch_sha256 != _canonical_sha256(records):
        raise ValueError("PPO consumed rollout batch SHA-256 differs")


def _validate_consumed_rollout_files(
    *,
    formal_run_root: Path,
    completion_path: PurePosixPath,
    record: dict[str, Any],
) -> None:
    """Re-authenticate current bytes and independently check episode semantics.

    The host intentionally has no Torch/runtime dependency.  It verifies the
    current single-link sidecar's exact filename, size, and SHA-256, but the
    captured clean ``alpagym_runtime`` disk reader remains the trust boundary
    for ``weights_only`` loading and exact tensor key/shape/dtype ABI
    validation.  Consequently this function is not, by itself, proof against a
    privileged actor that replaces both artifacts and the receipt.
    """

    canonical_root = formal_run_root.resolve(strict=True)
    if (
        canonical_root != formal_run_root
        or _FORMAL_RUN_ID.fullmatch(canonical_root.name) is None
    ):
        raise ValueError("PPO consumed rollout formal root is invalid")
    episode_path = canonical_root.joinpath(*completion_path.parts)
    episode_bytes, episode_metadata = read_stable_regular_file(episode_path)
    if (
        episode_metadata.st_size != record["episode_file_size_bytes"]
        or hashlib.sha256(episode_bytes).hexdigest() != record["episode_file_sha256"]
    ):
        raise ValueError("PPO consumed rollout episode file identity differs")
    try:
        artifact = json.loads(episode_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("PPO consumed rollout episode is invalid JSON") from exc
    if not isinstance(artifact, dict) or set(artifact) != {
        "artifact_schema",
        "episode_manifest",
        "manifest_sha256",
        "tensor_sidecar",
    }:
        raise ValueError("PPO consumed rollout disk artifact schema differs")
    manifest = artifact["episode_manifest"]
    sidecar = artifact["tensor_sidecar"]
    if (
        artifact["artifact_schema"] != "alpagym.disk_episode.v2"
        or not isinstance(manifest, dict)
        or artifact["manifest_sha256"] != _canonical_sha256(manifest)
        or artifact["manifest_sha256"] != record["episode_manifest_sha256"]
    ):
        raise ValueError("PPO consumed rollout episode manifest identity differs")
    expected_manifest_identity = {
        "session_uuid": record["session_uuid"],
        "rollout_seed": record["rollout_seed"],
        "scene_id": record["scene_id"],
        "num_steps": record["num_steps"],
    }
    if any(
        type(manifest.get(name)) is not type(value) or manifest.get(name) != value
        for name, value in expected_manifest_identity.items()
    ):
        raise ValueError("PPO consumed rollout episode ownership differs")
    policy_outputs = manifest.get("policy_outputs")
    if (
        not isinstance(policy_outputs, list)
        or len(policy_outputs) != record["num_steps"]
    ):
        raise ValueError("PPO consumed rollout episode row count differs")
    behavior_versions: set[int] = set()
    actor_valid_rows = 0
    for output in policy_outputs:
        if not isinstance(output, dict):
            raise ValueError("PPO consumed rollout policy output is invalid")
        replay_data = output.get("replay_data")
        if not isinstance(replay_data, dict):
            raise ValueError("PPO consumed rollout replay data is invalid")
        payload = replay_data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("PPO consumed rollout replay payload is invalid")
        transition = payload.get("transition")
        if transition is None:
            behavior_version = 0
            actor_valid = True
        else:
            if not isinstance(transition, dict):
                raise ValueError("PPO consumed rollout transition is invalid")
            behavior_version = transition.get("behavior_policy_version")
            actor_valid = transition.get("actor_valid", True)
            if (
                isinstance(behavior_version, bool)
                or not isinstance(behavior_version, int)
                or behavior_version < 0
            ):
                raise ValueError(
                    "PPO consumed rollout transition behavior version is invalid"
                )
            if not isinstance(actor_valid, bool):
                raise ValueError(
                    "PPO consumed rollout transition actor validity is invalid"
                )
        behavior_versions.add(behavior_version)
        actor_valid_rows += int(actor_valid)
    if behavior_versions != {record["behavior_weight_version"]}:
        raise ValueError("PPO consumed rollout behavior version differs from episode")
    if record["optimizer_sample_rows"] < len(policy_outputs):
        raise ValueError(
            "PPO consumed rollout optimizer row count differs from episode"
        )
    if record["optimizer_actor_valid_rows"] != actor_valid_rows:
        raise ValueError("PPO consumed rollout actor row count differs from episode")
    if (
        not isinstance(sidecar, dict)
        or set(sidecar) != {"filename", "format", "sha256", "size_bytes"}
        or sidecar["format"] != "torch.save.weights_only.v1"
        or {
            "filename": sidecar["filename"],
            "sha256": sidecar["sha256"],
            "size_bytes": sidecar["size_bytes"],
        }
        != record["tensor_sidecar"]
    ):
        raise ValueError("PPO consumed rollout sidecar descriptor differs")
    sidecar_path = episode_path.parent / sidecar["filename"]
    sidecar_identity = _hash_regular_file(
        source=sidecar_path,
        relative_path=PurePosixPath("artifacts") / sidecar["filename"],
    )
    if (
        sidecar_identity["sha256"] != sidecar["sha256"]
        or sidecar_identity["size_bytes"] != sidecar["size_bytes"]
    ):
        raise ValueError("PPO consumed rollout sidecar file identity differs")


def _capture_repository(
    *, repository: RepositorySource, destination: Path
) -> dict[str, Any]:
    """Capture one worktree's patch and untracked byte contents."""
    destination.mkdir(parents=True)
    tracked_patch = _git_bytes(repository.root, "diff", "--binary", "HEAD", "--")
    patch_path = destination / "tracked.patch"
    _write_bytes_exclusive(patch_path, tracked_patch)

    untracked_paths = _git_paths(
        repository.root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked = _capture_file_set(
        root=repository.root,
        paths=untracked_paths,
        destination=destination / "untracked",
    )
    ignored_pb2_paths: tuple[PurePosixPath, ...] = ()
    if repository.capture_ignored_generated_pb2:
        ignored_paths = _git_paths(
            repository.root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
        ignored_pb2_paths = tuple(
            path for path in ignored_paths if _is_executed_generated_pb2(path)
        )
    ignored_generated_pb2 = _capture_file_set(
        root=repository.root,
        paths=ignored_pb2_paths,
        destination=destination / "ignored_generated_pb2",
    )

    identity = _repository_identity_components(
        root=repository.root,
        tracked_patch=tracked_patch,
        untracked=untracked,
        ignored_generated_pb2=ignored_generated_pb2,
    )
    live_identity = _repository_identity(repository)
    if live_identity["source_identity_sha256"] != identity["source_identity_sha256"]:
        raise RuntimeError(
            f"worktree changed during formal prelaunch snapshot: {repository.root}"
        )
    return {
        "root_annotation": str(repository.root),
        **identity,
    }


def _repository_identity(repository: RepositorySource) -> dict[str, Any]:
    """Compute a live worktree identity without creating another copy."""
    _validate_repository_regular_files(repository)
    tracked_patch = _git_bytes(repository.root, "diff", "--binary", "HEAD", "--")
    untracked = _hash_file_set(
        root=repository.root,
        paths=_git_paths(
            repository.root, "ls-files", "--others", "--exclude-standard", "-z"
        ),
    )
    ignored_generated_pb2: dict[str, Any] = _empty_file_set_identity()
    if repository.capture_ignored_generated_pb2:
        ignored_paths = _git_paths(
            repository.root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
        ignored_generated_pb2 = _hash_file_set(
            root=repository.root,
            paths=tuple(
                path for path in ignored_paths if _is_executed_generated_pb2(path)
            ),
        )
    return _repository_identity_components(
        root=repository.root,
        tracked_patch=tracked_patch,
        untracked=untracked,
        ignored_generated_pb2=ignored_generated_pb2,
    )


def _repository_identity_components(
    *,
    root: Path,
    tracked_patch: bytes,
    untracked: dict[str, Any],
    ignored_generated_pb2: dict[str, Any],
) -> dict[str, Any]:
    """Build the portable identity shared by prelaunch and postrun scans."""
    head = _git_text(root, "rev-parse", "HEAD")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}")
    tracked_patch_identity = {
        "sha256": hashlib.sha256(tracked_patch).hexdigest(),
        "size_bytes": len(tracked_patch),
    }
    portable_identity = {
        "head": head,
        "tree": tree,
        "tracked_patch_sha256": tracked_patch_identity["sha256"],
        "untracked_manifest_sha256": untracked["manifest_sha256"],
        "ignored_generated_pb2_manifest_sha256": ignored_generated_pb2[
            "manifest_sha256"
        ],
    }
    return {
        "head": head,
        "tree": tree,
        "tracked_patch": tracked_patch_identity,
        "untracked": untracked,
        "ignored_generated_pb2": ignored_generated_pb2,
        "source_identity_sha256": _canonical_sha256(portable_identity),
    }


def _capture_file_set(
    *, root: Path, paths: tuple[PurePosixPath, ...], destination: Path
) -> dict[str, Any]:
    """Copy regular files into a fresh tree and return their strict manifest."""
    destination.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for relative_path in sorted(paths, key=str):
        source = _safe_worktree_file(root, relative_path)
        target = destination.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        entries.append(
            _copy_regular_file(
                source=source,
                target=target,
                relative_path=relative_path,
            )
        )
    return {
        "files": entries,
        "manifest_sha256": _canonical_sha256(entries),
    }


def _hash_file_set(*, root: Path, paths: tuple[PurePosixPath, ...]) -> dict[str, Any]:
    """Hash a live set of regular worktree files without copying it."""
    entries = [
        _hash_regular_file(
            source=_safe_worktree_file(root, relative_path),
            relative_path=relative_path,
        )
        for relative_path in sorted(paths, key=str)
    ]
    return {
        "files": entries,
        "manifest_sha256": _canonical_sha256(entries),
    }


def _empty_file_set_identity() -> dict[str, Any]:
    """Return the canonical identity of an empty file set."""
    return {"files": [], "manifest_sha256": _canonical_sha256([])}


def _copy_regular_file(
    *, source: Path, target: Path, relative_path: PurePosixPath
) -> dict[str, Any]:
    """Copy one file without following its final symlink and detect live races."""
    file_descriptor = _open_path_without_symlinks(source, final_flags=os.O_RDONLY)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"formal provenance only admits regular files: {source}")
        _require_single_link(metadata=before, path=source)
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(os.dup(file_descriptor), "rb") as source_file,
            target.open("xb") as target_file,
        ):
            while chunk := source_file.read(1024 * 1024):
                target_file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(file_descriptor)
        _require_single_link(metadata=after, path=source)
        if _stable_file_metadata(before) != _stable_file_metadata(after):
            raise RuntimeError(f"worktree file changed while snapshotting: {source}")
        if size != before.st_size:
            raise RuntimeError(f"short read while snapshotting worktree file: {source}")
        mode = stat.S_IMODE(before.st_mode)
        target.chmod(mode)
        return {
            "path": relative_path.as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "mode": f"{mode:04o}",
        }
    finally:
        os.close(file_descriptor)


def _hash_regular_file(*, source: Path, relative_path: PurePosixPath) -> dict[str, Any]:
    """Hash one regular file and reject concurrent mutation."""
    file_descriptor = _open_path_without_symlinks(source, final_flags=os.O_RDONLY)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"formal provenance only admits regular files: {source}")
        _require_single_link(metadata=before, path=source)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(os.dup(file_descriptor), "rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(file_descriptor)
        _require_single_link(metadata=after, path=source)
        if _stable_file_metadata(before) != _stable_file_metadata(after):
            raise RuntimeError(f"worktree file changed while hashing: {source}")
        if size != before.st_size:
            raise RuntimeError(f"short read while hashing worktree file: {source}")
        mode = stat.S_IMODE(before.st_mode)
        return {
            "path": relative_path.as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "mode": f"{mode:04o}",
        }
    finally:
        os.close(file_descriptor)


def _safe_worktree_file(root: Path, relative_path: PurePosixPath) -> Path:
    """Resolve one Git-returned path without permitting symlink traversal."""
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe Git worktree path: {relative_path}")
    source = root.joinpath(*relative_path.parts)
    current = root
    for part in relative_path.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"worktree path traverses a symlink: {relative_path}")
    return source


def _validate_repository_regular_files(repository: RepositorySource) -> None:
    """Require one link for every regular file that can affect execution."""
    for relative_path in _repository_effective_paths(repository):
        source = _safe_worktree_file(repository.root, relative_path)
        try:
            file_descriptor = _open_path_without_symlinks(
                source,
                final_flags=getattr(os, "O_PATH", os.O_RDONLY),
            )
        except FileNotFoundError:
            continue
        try:
            metadata = os.fstat(file_descriptor)
            if stat.S_ISREG(metadata.st_mode):
                _require_single_link(metadata=metadata, path=source)
        finally:
            os.close(file_descriptor)


def _normalized_absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without following any symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _open_path_without_symlinks(path: Path, *, final_flags: int) -> int:
    """Open one absolute path while refusing symlinks in every component."""
    absolute_path = _normalized_absolute_path(path)
    if absolute_path == Path(absolute_path.anchor):
        raise ValueError(f"formal provenance expected a file path: {absolute_path}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | _IN_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    current_descriptor = os.open(absolute_path.anchor, directory_flags)
    try:
        for component in absolute_path.parts[1:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=current_descriptor,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        flags = final_flags | _IN_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(
            absolute_path.name,
            flags,
            dir_fd=current_descriptor,
        )
    finally:
        os.close(current_descriptor)


def _require_single_link(*, metadata: os.stat_result, path: Path) -> None:
    """Reject regular files writable through an unobserved external hard link."""
    if metadata.st_nlink != 1:
        raise ValueError(
            "formal provenance regular files must have exactly one hard link: "
            f"{path} has st_nlink={metadata.st_nlink}"
        )


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    """Return metadata that exposes content, inode, and hard-link races."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_import_probe(
    probe: dict[str, Any],
    *,
    alpagym_root: Path,
    alpasim_root: Path,
    critical_environment: dict[str, str],
) -> dict[str, Any]:
    """Require actual formal subprocess imports to resolve to captured sources."""
    _validate_import_probe_identity(probe)
    if probe["environment"] != critical_environment:
        raise RuntimeError("formal import probe environment differs from prelaunch")
    expected_roots = {
        "alpagym_host": alpagym_root,
        "alpagym_runtime": alpagym_root,
        "alpagym_g1_vla": alpagym_root,
        "alpasim_grpc.v0.humanoid_pb2": alpasim_root / "src" / "grpc",
        "alpasim_grpc.v0.humanoid_pb2_grpc": alpasim_root / "src" / "grpc",
    }
    if set(probe["module_origins"]) != set(expected_roots):
        raise ValueError("formal import probe has an unexpected module set")
    for module_name, expected_root in expected_roots.items():
        origin = Path(probe["module_origins"][module_name]).resolve(strict=True)
        if not origin.is_file() or not origin.is_relative_to(
            expected_root.resolve(strict=True)
        ):
            raise RuntimeError(
                f"formal import {module_name} resolved outside captured source: {origin}"
            )
    for message_name, required_fields in _REQUIRED_HUMANOID_DESCRIPTOR_FIELDS.items():
        observed_fields = probe["descriptor_fields"].get(message_name, {})
        if any(
            observed_fields.get(field_name) != field_number
            for field_name, field_number in required_fields.items()
        ):
            raise RuntimeError(
                f"formal humanoid protobuf descriptor {message_name} has wrong "
                "field names or numbers"
            )
    for (
        service_name,
        required_methods,
    ) in _REQUIRED_HUMANOID_DESCRIPTOR_SERVICE_METHOD_INPUTS.items():
        observed_methods = probe["descriptor_services"].get(service_name, {})
        if any(
            observed_methods.get(method_name) != request_type
            for method_name, request_type in required_methods.items()
        ):
            raise RuntimeError(
                f"formal humanoid protobuf service {service_name} has wrong "
                "method names or request types"
            )
    if set(probe["grpc_bindings"]) != _REQUIRED_HUMANOID_GRPC_BINDINGS or not all(
        probe["grpc_bindings"].values()
    ):
        raise RuntimeError(
            "formal humanoid generated gRPC bindings are missing AbortSession"
        )
    return probe


def _validate_import_probe_identity(probe: Any) -> None:
    """Validate the strict schema and nested self-hash of an import probe."""
    if not isinstance(probe, dict):
        raise TypeError("formal import probe must be an object")
    expected_keys = {
        "schema_id",
        "captured_at_utc",
        "command",
        "command_sha256",
        "environment",
        "environment_sha256",
        "module_origins",
        "descriptor_fields",
        "descriptor_services",
        "grpc_bindings",
        "identity_sha256",
    }
    if set(probe) != expected_keys:
        raise ValueError("formal import probe has an unexpected schema")
    if probe["schema_id"] != "alpagym.formal_import_probe.v2":
        raise ValueError("formal import probe schema_id is unsupported")
    if not isinstance(probe["command"], list) or not all(
        isinstance(argument, str) for argument in probe["command"]
    ):
        raise TypeError("formal import probe command must be a string list")
    if probe["command_sha256"] != _canonical_sha256(probe["command"]):
        raise ValueError("formal import probe command SHA256 is invalid")
    if not isinstance(probe["environment"], dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in probe["environment"].items()
    ):
        raise TypeError("formal import probe environment must be a string map")
    if probe["environment_sha256"] != _canonical_sha256(probe["environment"]):
        raise ValueError("formal import probe environment SHA256 is invalid")
    if not isinstance(probe["module_origins"], dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in probe["module_origins"].items()
    ):
        raise TypeError("formal import probe origins must be a string map")
    if not isinstance(probe["descriptor_fields"], dict) or not all(
        isinstance(name, str)
        and isinstance(fields, dict)
        and all(
            isinstance(field_name, str)
            and isinstance(field_number, int)
            and not isinstance(field_number, bool)
            for field_name, field_number in fields.items()
        )
        for name, fields in probe["descriptor_fields"].items()
    ):
        raise TypeError("formal import probe descriptor fields are invalid")
    if not isinstance(probe["descriptor_services"], dict) or not all(
        isinstance(name, str)
        and isinstance(methods, dict)
        and all(
            isinstance(method_name, str) and isinstance(request_type, str)
            for method_name, request_type in methods.items()
        )
        for name, methods in probe["descriptor_services"].items()
    ):
        raise TypeError("formal import probe descriptor services are invalid")
    if not isinstance(probe["grpc_bindings"], dict) or not all(
        isinstance(name, str) and isinstance(present, bool)
        for name, present in probe["grpc_bindings"].items()
    ):
        raise TypeError("formal import probe generated gRPC bindings are invalid")
    body = dict(probe)
    identity_sha256 = body.pop("identity_sha256")
    if identity_sha256 != _canonical_sha256(body):
        raise ValueError("formal import probe identity SHA256 is invalid")


def _validate_source_watch_receipt(receipt: Any, *, expected_stage: str) -> None:
    """Validate one persisted inotify checkpoint before trusting its binding."""
    if not isinstance(receipt, dict):
        raise TypeError("formal source-watch receipt must be an object")
    expected_keys = {
        "schema_id",
        "captured_at_utc",
        "stage",
        "backend",
        "watch_mask",
        "watched_directory_count",
        "watched_directories_sha256",
        "watched_regular_file_count",
        "watched_regular_files_sha256",
        "healthy",
        "health_failures",
        "relevant_event_count",
        "relevant_events",
        "clean",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("formal source-watch receipt has an unexpected schema")
    if receipt["schema_id"] != "alpagym.formal_source_watch.v1":
        raise ValueError("formal source-watch receipt schema_id is unsupported")
    if receipt["stage"] != expected_stage:
        raise ValueError("formal source-watch receipt stage is invalid")
    if receipt["backend"] != "linux_inotify" or receipt["watch_mask"] != (
        _SOURCE_WATCH_MASK
    ):
        raise ValueError("formal source-watch backend or mask is invalid")
    if receipt["relevant_event_count"] != len(receipt["relevant_events"]):
        raise ValueError("formal source-watch event count is invalid")
    body = dict(receipt)
    stored_sha256 = body.pop("receipt_sha256")
    if stored_sha256 != _canonical_sha256(body):
        raise ValueError("formal source-watch receipt SHA256 is invalid")


def _directory_tree_identity(
    root: Path,
    *,
    schema_id: str = "alpagym.scene_cache_tree.v1",
    label: str = "scene cache",
) -> dict[str, Any]:
    """Hash an immutable directory tree without following links."""

    root = _normalized_absolute_path(root)
    resolved_root = root.resolve(strict=True)
    if root != resolved_root or not root.is_dir():
        raise ValueError(
            f"formal {label} must be a canonical non-symlink directory: {root}"
        )
    files: list[dict[str, Any]] = []
    directories: list[str] = []

    def scan(directory: Path, relative_directory: PurePosixPath) -> None:
        """Collect one stable directory level and recurse into real children."""

        before = os.lstat(directory)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"formal {label} entry is not a directory: {directory}")
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            entry_path = directory / entry.name
            relative_path = relative_directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"formal {label} must not contain symlinks: {entry_path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(relative_path.as_posix())
                scan(entry_path, relative_path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"formal {label} contains a special file: {entry_path}"
                )
            identity = _file_identity(entry_path)
            files.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "sha256": identity["sha256"],
                    "size_bytes": identity["size_bytes"],
                }
            )
        after = os.lstat(directory)
        if _stable_file_metadata(before) != _stable_file_metadata(after):
            raise RuntimeError(
                f"formal {label} directory changed while hashing: {directory}"
            )

    scan(root, PurePosixPath())
    payload = {
        "schema_id": schema_id,
        "directories": directories,
        "files": files,
    }
    return {
        "path_annotation": str(root),
        "tree_sha256": _canonical_sha256(payload),
        "file_count": len(files),
        "total_size_bytes": sum(entry["size_bytes"] for entry in files),
        "directories": directories,
        "files": files,
    }


def _validate_directory_tree_identity(
    value: Any,
    *,
    schema_id: str = "alpagym.scene_cache_tree.v1",
    label: str = "scene cache",
) -> None:
    """Validate one optional immutable-tree identity from a runtime receipt."""

    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {
        "path_annotation",
        "tree_sha256",
        "file_count",
        "total_size_bytes",
        "directories",
        "files",
    }:
        raise ValueError(f"formal {label} identity has an unexpected schema")
    if not Path(value["path_annotation"]).is_absolute():
        raise ValueError(f"formal {label} identity path must be absolute")
    if (
        not isinstance(value["tree_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["tree_sha256"]) is None
    ):
        raise ValueError(f"formal {label} identity SHA256 is invalid")
    if not isinstance(value["directories"], list) or not all(
        isinstance(path, str) and path for path in value["directories"]
    ):
        if value["directories"] != []:
            raise TypeError(f"formal {label} directories must be strings")
    if not isinstance(value["files"], list):
        raise TypeError(f"formal {label} files must be a list")
    if value["file_count"] != len(value["files"]):
        raise ValueError(f"formal {label} file count is invalid")
    total_size_bytes = 0
    for entry in value["files"]:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"formal {label} file identity is invalid")
        relative_path = PurePosixPath(entry["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"formal {label} file path is unsafe")
        if (
            not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise ValueError(f"formal {label} file SHA256 is invalid")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
            raise ValueError(f"formal {label} file size is invalid")
        total_size_bytes += entry["size_bytes"]
    if value["total_size_bytes"] != total_size_bytes:
        raise ValueError(f"formal {label} total size is invalid")
    payload = {
        "schema_id": schema_id,
        "directories": value["directories"],
        "files": value["files"],
    }
    if value["tree_sha256"] != _canonical_sha256(payload):
        raise ValueError(f"formal {label} tree SHA256 is invalid")


def _read_runtime_ready_receipt(
    path: Path,
    *,
    expected_source_set_sha256: str,
    expected_config_identity_sha256: str,
    expected_critical_environment_sha256: str,
    expected_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Parse and verify a runtime receipt and its prelaunch bindings."""
    if expected_receipt_sha256 is None:
        raise RuntimeError(
            "formal runtime receipt was not captured by this lifecycle object"
        )
    receipt = _read_json_regular(path)
    expected_keys = {
        "schema_id",
        "captured_at_utc",
        "workload_kind",
        "workload_command",
        "workload_command_sha256",
        "prelaunch_portable_source_set_sha256",
        "prelaunch_config_artifacts_identity_sha256",
        "critical_environment_sha256",
        "import_probe",
        "source_watch_receipt_sha256",
        "scene_cache_identity",
        "controller_release_identity",
        "runtimes",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("formal runtime receipt has an unexpected schema")
    if receipt["schema_id"] != "alpagym.formal_run_runtime_ready.v4":
        raise ValueError("formal runtime receipt schema_id is not supported")
    stored_sha256 = receipt["receipt_sha256"]
    if not isinstance(stored_sha256, str) or len(stored_sha256) != 64:
        raise ValueError("formal runtime receipt has no valid SHA256")
    if stored_sha256 != expected_receipt_sha256:
        raise ValueError(
            "formal runtime receipt SHA256 differs from its in-memory lifecycle binding"
        )
    receipt_body = dict(receipt)
    del receipt_body["receipt_sha256"]
    if _canonical_sha256(receipt_body) != stored_sha256:
        raise ValueError("formal runtime receipt SHA256 does not match its contents")
    if receipt["prelaunch_portable_source_set_sha256"] != expected_source_set_sha256:
        raise ValueError("formal runtime receipt source binding differs from prelaunch")
    if (
        receipt["prelaunch_config_artifacts_identity_sha256"]
        != expected_config_identity_sha256
    ):
        raise ValueError("formal runtime receipt config binding differs from prelaunch")
    if receipt["critical_environment_sha256"] != expected_critical_environment_sha256:
        raise ValueError(
            "formal runtime receipt critical environment differs from prelaunch"
        )
    _validate_import_probe_identity(receipt["import_probe"])
    _validate_directory_tree_identity(receipt["scene_cache_identity"])
    _validate_directory_tree_identity(
        receipt["controller_release_identity"],
        schema_id="alpagym.controller_release_tree.v1",
        label="controller release",
    )
    source_watch = _read_json_regular(path.parent / "source_watch_runtime_ready.json")
    _validate_source_watch_receipt(
        source_watch, expected_stage="runtime_ready_admission"
    )
    if receipt["source_watch_receipt_sha256"] != source_watch["receipt_sha256"]:
        raise ValueError("formal runtime receipt source-watch binding is invalid")
    if source_watch["clean"] is not True:
        raise ValueError("formal runtime receipt bound a dirty source watch")
    workload_kind = receipt["workload_kind"]
    if workload_kind not in {"cosmos_training", "qualification_rollout"}:
        raise ValueError("formal runtime receipt workload kind is unsupported")
    command = receipt["workload_command"]
    if not isinstance(command, list) or not all(
        isinstance(argument, str) and argument for argument in command
    ):
        raise TypeError("formal runtime receipt workload command must be a string list")
    if receipt["workload_command_sha256"] != _canonical_sha256(command):
        raise ValueError("formal runtime receipt workload command SHA256 is invalid")
    runtimes = receipt["runtimes"]
    if not isinstance(runtimes, list) or not runtimes:
        raise ValueError("formal runtime receipt must contain at least one runtime")
    for runtime in runtimes:
        _validate_runtime_entry(runtime)
    return receipt


def _validate_runtime_entry(runtime: Any) -> None:
    """Validate one Compose runtime entry and its nested identity digest."""
    if not isinstance(runtime, dict):
        raise TypeError("formal runtime entry must be an object")
    expected_keys = {
        "wizard_log_dir_annotation",
        "compose_path_annotation",
        "compose_sha256",
        "compose_size_bytes",
        "compose_project",
        "services",
        "images",
        "runtime_identity_sha256",
    }
    if set(runtime) != expected_keys:
        raise ValueError("formal Compose runtime entry has an unexpected schema")
    services = runtime["services"]
    images = runtime["images"]
    if not isinstance(services, list) or not services:
        raise ValueError("formal Compose runtime entry has no services")
    if not isinstance(images, list) or not images:
        raise ValueError("formal Compose runtime entry has no images")
    service_names = []
    for service in services:
        if not isinstance(service, dict) or not isinstance(service.get("service"), str):
            raise TypeError("formal Compose service entry is invalid")
        service_names.append(service["service"])
    if not any(name.startswith("runtime-") for name in service_names):
        raise ValueError("formal Compose receipt has no runtime service")
    if not any(name.startswith("humanoid_dynamics-") for name in service_names):
        raise ValueError("formal Compose receipt has no humanoid-dynamics service")
    expected_identity = _canonical_sha256(
        {
            "compose_sha256": runtime["compose_sha256"],
            "compose_project": runtime["compose_project"],
            "services": services,
            "images": images,
        }
    )
    if runtime["runtime_identity_sha256"] != expected_identity:
        raise ValueError("formal Compose runtime identity SHA256 is invalid")


def _read_json_regular(path: Path) -> dict[str, Any]:
    """Read strict JSON from one regular file without following a final symlink."""
    path = _normalized_absolute_path(path)
    file_descriptor = _open_path_without_symlinks(path, final_flags=os.O_RDONLY)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"formal provenance receipt is not regular: {path}")
        with os.fdopen(os.dup(file_descriptor), "rb") as file:
            data = file.read()
        after = os.fstat(file_descriptor)
        if _stable_file_metadata(before) != _stable_file_metadata(after):
            raise RuntimeError(
                f"formal provenance receipt changed while reading: {path}"
            )
    finally:
        os.close(file_descriptor)
    value = json.loads(data.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise TypeError("formal provenance receipt must be a JSON object")
    return value


def _capture_compose_runtime(
    *,
    wizard_log_dir: Path,
    alpasim_root: Path,
    humanoid_root: Path,
    scene_store_root: Path,
    scene_cache_root: Path | None,
    runtime_cache_root: Path | None,
    controller_release_root: Path | None,
) -> dict[str, Any]:
    """Capture and validate one Wizard-owned local Compose runtime."""
    wizard_log_dir = wizard_log_dir.resolve(strict=True)
    expected_compose_project = wizard_compose_project(wizard_log_dir)
    compose_path = wizard_log_dir / "docker-compose.yaml"
    compose_bytes = compose_path.read_bytes()
    compose_data = yaml.safe_load(compose_bytes)
    if not isinstance(compose_data, dict) or not isinstance(
        compose_data.get("services"), dict
    ):
        raise ValueError(f"invalid Wizard Compose file: {compose_path}")
    expected_services = set(compose_data["services"])
    if not expected_services:
        raise ValueError(f"Wizard Compose file has no services: {compose_path}")

    container_ids = tuple(
        line
        for line in _run_command(
            [
                "docker",
                "compose",
                "--project-name",
                expected_compose_project,
                "--file",
                str(compose_path),
                "ps",
                "--all",
                "--quiet",
            ]
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    if not container_ids:
        raise RuntimeError(f"Wizard Compose runtime has no containers: {compose_path}")
    inspections = json.loads(
        _run_command(["docker", "inspect", *container_ids]).decode("utf-8")
    )
    if not isinstance(inspections, list) or len(inspections) != len(container_ids):
        raise RuntimeError("docker inspect did not return every Compose container")

    service_inspections: dict[str, dict[str, Any]] = {}
    for inspection in inspections:
        if not isinstance(inspection, dict):
            raise TypeError("docker inspect container entry must be an object")
        config = inspection["Config"]
        state = inspection["State"]
        labels = config["Labels"] or {}
        service = labels.get("com.docker.compose.service")
        if service not in expected_services:
            raise RuntimeError(
                f"unexpected or unlabeled Compose container service: {service!r}"
            )
        if service in service_inspections:
            raise RuntimeError(f"duplicate Compose container for service {service!r}")
        if state.get("Running") is not True:
            raise RuntimeError(f"Compose service is not running: {service}")
        project = labels.get("com.docker.compose.project")
        config_hash = labels.get("com.docker.compose.config-hash")
        config_files = labels.get("com.docker.compose.project.config_files")
        if project != expected_compose_project:
            raise RuntimeError(
                f"Compose service {service!r} has project label {project!r}; "
                "expected the pinned Wizard launch project "
                f"{expected_compose_project!r}"
            )
        if not isinstance(config_hash, str) or not config_hash:
            raise RuntimeError(f"Compose service {service!r} has no config-hash label")
        if not isinstance(config_files, str) or not config_files:
            raise RuntimeError(f"Compose service {service!r} has no config-files label")
        labeled_config_paths = {
            Path(value).expanduser().resolve(strict=True)
            for value in config_files.split(",")
            if value
        }
        if compose_path not in labeled_config_paths:
            raise RuntimeError(
                f"Compose service {service!r} is not bound to {compose_path}"
            )
        service_inspections[service] = inspection
    if set(service_inspections) != expected_services:
        raise RuntimeError(
            "Compose services differ from running containers: "
            f"missing={sorted(expected_services - set(service_inspections))}"
        )
    runtime_services = {
        name for name in expected_services if name.startswith("runtime-")
    }
    dynamics_services = {
        name for name in expected_services if name.startswith("humanoid_dynamics-")
    }
    if not runtime_services:
        raise RuntimeError("formal humanoid Compose runtime has no runtime service")
    if not dynamics_services:
        raise RuntimeError(
            "formal humanoid Compose runtime has no humanoid-dynamics service"
        )
    executable_services = runtime_services | dynamics_services
    alpasim_src = (alpasim_root / "src").resolve(strict=True)
    alpasim_plugins = (alpasim_root / "plugins").resolve(strict=True)
    humanoid_source = humanoid_root.resolve(strict=True)
    scene_store = scene_store_root.expanduser().resolve(strict=True)
    controller_release = (
        controller_release_root.expanduser().resolve(strict=True)
        if controller_release_root is not None
        else None
    )
    if (scene_cache_root is None) != (runtime_cache_root is None):
        raise ValueError(
            "formal scene and runtime cache roots must either both be set or both be absent"
        )
    common_mounts = {
        str(alpasim_src): ("/repo/src", False),
        str(alpasim_plugins): ("/repo/plugins", False),
        str(humanoid_source): ("/repo/humanoid-rl-joint-sim", False),
        str(scene_store): ("/mnt/humanoid-scene-store", False),
    }
    controller_mounts: dict[str, tuple[str, bool]] = {}
    if controller_release is not None:
        _require_isolated_root(
            label="controller release",
            root=controller_release,
            protected_sources={
                "AlpaSim source": alpasim_src,
                "AlpaSim plugins": alpasim_plugins,
                "Humanoid source": humanoid_source,
                "SceneStore": scene_store,
            },
        )
        controller_mounts[str(controller_release)] = (
            "/mnt/sonic-visual-release",
            False,
        )
    cache_mounts: dict[str, tuple[str, bool]] = {}
    if scene_cache_root is not None and runtime_cache_root is not None:
        scene_cache = scene_cache_root.expanduser().resolve(strict=True)
        runtime_cache = runtime_cache_root.expanduser().resolve(strict=True)
        _require_cache_source_isolation(
            scene_cache=scene_cache,
            runtime_cache=runtime_cache,
            protected_sources={
                "AlpaSim source": alpasim_src,
                "AlpaSim plugins": alpasim_plugins,
                "Humanoid source": humanoid_source,
                "SceneStore": scene_store,
                **(
                    {"controller release": controller_release}
                    if controller_release is not None
                    else {}
                ),
            },
        )
        cache_mounts[str(scene_cache)] = (
            "/mnt/humanoid-scene-cache",
            False,
        )
        cache_mounts[str(runtime_cache)] = ("/root/.cache", True)
    canonical_mounts = {**common_mounts, **cache_mounts, **controller_mounts}
    cache_service_count = 0
    for service in executable_services:
        inspection = service_inspections[service]
        actual_bind_mounts = sorted(
            [
                {
                    "source": str(Path(mount["Source"]).resolve()),
                    "destination": str(mount["Destination"]),
                    "rw": bool(mount.get("RW", False)),
                }
                for mount in inspection["Mounts"]
                if mount.get("Type") == "bind"
            ],
            key=lambda mount: (mount["source"], mount["destination"]),
        )
        expected_bind_mounts = _configured_bind_mounts(
            service_config=compose_data["services"][service],
            compose_directory=compose_path.parent,
        )
        if actual_bind_mounts != expected_bind_mounts:
            raise RuntimeError(
                f"Compose service {service!r} live bind mounts differ from its exact "
                "generated Compose config"
            )
        _audit_canonical_mount_isolation(
            service=service,
            mounts=inspection["Mounts"],
            canonical_mounts=canonical_mounts,
        )
        for source, (destination, expected_rw) in common_mounts.items():
            source_mounts = [
                mount for mount in actual_bind_mounts if mount["source"] == source
            ]
            if source_mounts != [
                {"source": source, "destination": destination, "rw": expected_rw}
            ]:
                access = "read-write" if expected_rw else "read-only"
                raise RuntimeError(
                    f"Compose service {service!r} violates canonical {access} "
                    f"mount mapping {source} -> {destination}"
                )
        if service in dynamics_services and controller_mounts:
            for source, (destination, expected_rw) in controller_mounts.items():
                observed = [
                    mount for mount in actual_bind_mounts if mount["source"] == source
                ]
                expected = {
                    "source": source,
                    "destination": destination,
                    "rw": expected_rw,
                }
                if observed != [expected]:
                    raise RuntimeError(
                        f"Compose service {service!r} violates canonical read-only "
                        f"controller release mapping {source} -> {destination}"
                    )
        if cache_mounts:
            observed_cache_mounts = {
                source: [
                    mount for mount in actual_bind_mounts if mount["source"] == source
                ]
                for source in cache_mounts
            }
            if any(observed_cache_mounts.values()):
                for source, (destination, expected_rw) in cache_mounts.items():
                    expected = {
                        "source": source,
                        "destination": destination,
                        "rw": expected_rw,
                    }
                    if observed_cache_mounts[source] != [expected]:
                        raise RuntimeError(
                            f"Compose service {service!r} must mount the complete "
                            "scene/runtime cache pair with canonical access modes"
                        )
                cache_service_count += 1
    if cache_mounts and cache_service_count == 0:
        raise RuntimeError(
            "formal visual runtime has no service with the canonical cache pair"
        )

    image_ids = sorted({str(inspection["Image"]) for inspection in inspections})
    image_inspections = json.loads(
        _run_command(["docker", "image", "inspect", *image_ids]).decode("utf-8")
    )
    if not isinstance(image_inspections, list):
        raise TypeError("docker image inspect output must be a list")
    images_by_id = {
        str(image["Id"]): {
            "id": str(image["Id"]),
            "repo_digests": sorted(
                str(value) for value in image.get("RepoDigests") or []
            ),
            "repo_tags": sorted(str(value) for value in image.get("RepoTags") or []),
        }
        for image in image_inspections
    }
    if set(images_by_id) != set(image_ids):
        raise RuntimeError(
            "docker image inspect did not resolve every running image ID"
        )

    containers = []
    for service in sorted(service_inspections):
        inspection = service_inspections[service]
        config = inspection["Config"]
        labels = config["Labels"] or {}
        mounts = [
            {
                "type": mount["Type"],
                "source": mount["Source"],
                "destination": mount["Destination"],
                "mode": mount.get("Mode", ""),
                "rw": bool(mount.get("RW", False)),
                "propagation": mount.get("Propagation", ""),
            }
            for mount in inspection["Mounts"]
        ]
        containers.append(
            {
                "service": service,
                "container_id": inspection["Id"],
                "name": str(inspection.get("Name", "")).lstrip("/"),
                "running": True,
                "configured_image": config["Image"],
                "image_config_id": inspection["Image"],
                "oci_platform_manifest_digest": _image_manifest_digest(inspection),
                "image_manifest_descriptor": inspection.get("ImageManifestDescriptor"),
                "compose_project": labels.get("com.docker.compose.project"),
                "compose_config_hash": labels.get("com.docker.compose.config-hash"),
                "entrypoint": config.get("Entrypoint"),
                "command": config.get("Cmd"),
                "mounts": sorted(
                    mounts,
                    key=lambda mount: (
                        str(mount["destination"]),
                        str(mount["source"]),
                    ),
                ),
            }
        )
    receipt = {
        "wizard_log_dir_annotation": str(wizard_log_dir),
        "compose_path_annotation": str(compose_path),
        "compose_sha256": hashlib.sha256(compose_bytes).hexdigest(),
        "compose_size_bytes": len(compose_bytes),
        "compose_project": expected_compose_project,
        "services": containers,
        "images": [images_by_id[image_id] for image_id in sorted(images_by_id)],
    }
    if compose_path.read_bytes() != compose_bytes:
        raise RuntimeError(
            f"Wizard Compose file changed during inspection: {compose_path}"
        )
    receipt["runtime_identity_sha256"] = _canonical_sha256(
        {
            "compose_sha256": receipt["compose_sha256"],
            "compose_project": receipt["compose_project"],
            "services": containers,
            "images": receipt["images"],
        }
    )
    return receipt


def _audit_canonical_mount_isolation(
    *,
    service: str,
    mounts: Any,
    canonical_mounts: dict[str, tuple[str, bool]],
) -> None:
    """Reject bind/volume/tmpfs shadows around provenance-owned source roots."""
    if not isinstance(mounts, list):
        raise TypeError(f"Compose service {service!r} Mounts must be a list")
    canonical_sources = {
        Path(source).resolve(strict=True): (PurePosixPath(destination), expected_rw)
        for source, (destination, expected_rw) in canonical_mounts.items()
    }
    for mount in mounts:
        if not isinstance(mount, dict):
            raise TypeError(f"Compose service {service!r} mount must be an object")
        mount_type = mount.get("Type")
        destination_value = mount.get("Destination")
        if not isinstance(mount_type, str) or not isinstance(destination_value, str):
            raise ValueError(
                f"Compose service {service!r} mount lacks type or destination"
            )
        destination = PurePosixPath(destination_value)
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError(
                f"Compose service {service!r} has unsafe mount destination "
                f"{destination_value!r}"
            )
        source_path: Path | None = None
        if mount_type == "bind":
            source_value = mount.get("Source")
            if not isinstance(source_value, str):
                raise ValueError(f"Compose service {service!r} bind has no source")
            source_path = Path(source_value).resolve(strict=True)

        allowed_canonical = any(
            mount_type == "bind"
            and source_path == canonical_source
            and destination == canonical_destination
            and mount.get("RW") is expected_rw
            for canonical_source, (
                canonical_destination,
                expected_rw,
            ) in canonical_sources.items()
        )
        destination_overlaps = any(
            destination == canonical_destination
            or destination.is_relative_to(canonical_destination)
            or canonical_destination.is_relative_to(destination)
            for canonical_destination, _expected_rw in canonical_sources.values()
        )
        if destination_overlaps and not allowed_canonical:
            raise RuntimeError(
                f"Compose service {service!r} has a mount shadowing canonical "
                f"destination: {destination}"
            )
        if source_path is not None:
            source_overlaps = any(
                source_path == canonical_source
                or source_path.is_relative_to(canonical_source)
                or canonical_source.is_relative_to(source_path)
                for canonical_source in canonical_sources
            )
            if source_overlaps and not allowed_canonical:
                raise RuntimeError(
                    f"Compose service {service!r} remounts canonical source at an "
                    f"alternate location: {source_path} -> {destination}"
                )


def _require_cache_source_isolation(
    *,
    scene_cache: Path,
    runtime_cache: Path,
    protected_sources: dict[str, Path],
) -> None:
    """Reject cache roots that alias or contain provenance-owned source roots."""

    pairs = [
        ("runtime cache", runtime_cache, "scene cache", scene_cache),
        *[
            ("scene cache", scene_cache, label, source)
            for label, source in protected_sources.items()
        ],
        *[
            ("runtime cache", runtime_cache, label, source)
            for label, source in protected_sources.items()
        ],
    ]
    for first_label, first, second_label, second in pairs:
        if (
            first == second
            or first.is_relative_to(second)
            or second.is_relative_to(first)
        ):
            raise RuntimeError(
                f"formal {first_label} must be disjoint from {second_label}: "
                f"{first} versus {second}"
            )


def _require_isolated_root(
    *,
    label: str,
    root: Path,
    protected_sources: dict[str, Path],
) -> None:
    """Reject an immutable input that aliases another provenance-owned root."""

    for protected_label, protected in protected_sources.items():
        if (
            root == protected
            or root.is_relative_to(protected)
            or protected.is_relative_to(root)
        ):
            raise RuntimeError(
                f"formal {label} must be disjoint from {protected_label}: "
                f"{root} versus {protected}"
            )


def _configured_bind_mounts(
    *, service_config: Any, compose_directory: Path
) -> list[dict[str, Any]]:
    """Resolve the exact host bind mounts declared for one Compose service."""
    if not isinstance(service_config, dict):
        raise TypeError("Compose service config must be an object")
    volumes = service_config.get("volumes", [])
    if not isinstance(volumes, list):
        raise TypeError("Compose service volumes must be a list")
    mounts: list[dict[str, Any]] = []
    for volume in volumes:
        source: str
        destination: str
        read_only: bool
        if isinstance(volume, str):
            parts = volume.split(":", maxsplit=2)
            if len(parts) < 2:
                continue
            source, destination = parts[:2]
            options = parts[2].split(",") if len(parts) == 3 else []
            read_only = "ro" in options
        elif isinstance(volume, dict):
            if volume.get("type") != "bind":
                continue
            source_value = volume.get("source")
            destination_value = volume.get("target")
            if not isinstance(source_value, str) or not isinstance(
                destination_value, str
            ):
                raise ValueError("Compose bind mount requires source and target")
            source = source_value
            destination = destination_value
            read_only = bool(volume.get("read_only", False))
        else:
            raise TypeError("Compose volume entry must be a string or object")
        if not source.startswith(("/", ".")):
            continue
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = compose_directory / source_path
        resolved_source = str(source_path.resolve(strict=True))
        mounts.append(
            {
                "source": resolved_source,
                "destination": destination,
                "rw": not read_only,
            }
        )
    return sorted(mounts, key=lambda mount: (mount["source"], mount["destination"]))


def _image_manifest_digest(inspection: dict[str, Any]) -> str | None:
    """Return the OCI platform-manifest digest without confusing it with config ID."""
    descriptor = inspection.get("ImageManifestDescriptor")
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise TypeError("container ImageManifestDescriptor must be an object")
    digest = descriptor.get("digest", descriptor.get("Digest"))
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("container image manifest descriptor has no SHA256 digest")
    return digest


def _require_git_worktree_root(path: Path) -> Path:
    """Resolve a configured repo and require it to be the Git toplevel."""
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(
            f"formal provenance repository is not a directory: {root}"
        )
    reported_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if reported_root != root:
        raise ValueError(
            f"formal provenance path must be the Git worktree root: {root} != {reported_root}"
        )
    return root


def _git_paths(root: Path, *arguments: str) -> tuple[PurePosixPath, ...]:
    """Run a NUL-delimited Git path query and validate its encoding."""
    output = _git_bytes(root, *arguments)
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise RuntimeError(f"Git path output was not NUL terminated in {root}")
    return tuple(
        PurePosixPath(value.decode("utf-8", errors="strict"))
        for value in output[:-1].split(b"\0")
    )


def _repository_effective_paths(
    repository: RepositorySource,
) -> tuple[PurePosixPath, ...]:
    """Return every Git-effective path whose parent must be watched."""
    _reject_hidden_index_flags(repository)
    paths = set(_git_paths(repository.root, "ls-files", "-z"))
    paths.update(
        _git_paths(
            repository.root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
    )
    if repository.capture_ignored_generated_pb2:
        paths.update(
            path
            for path in _git_paths(
                repository.root,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            )
            if _is_executed_generated_pb2(path)
        )
    return tuple(sorted(paths, key=str))


def _reject_hidden_index_flags(repository: RepositorySource) -> None:
    """Reject index flags that can hide tracked working-tree byte changes."""
    output = _git_bytes(repository.root, "ls-files", "-v", "-z")
    if output and not output.endswith(b"\0"):
        raise RuntimeError(
            f"Git index flag output was not NUL terminated in {repository.root}"
        )
    hidden_paths: list[str] = []
    for record in output[:-1].split(b"\0") if output else ():
        if len(record) < 3 or record[1:2] != b" ":
            raise RuntimeError(
                f"Git index flag output was malformed in {repository.root}"
            )
        tag = record[:1].decode("ascii", errors="strict")
        path = record[2:].decode("utf-8", errors="strict")
        if tag == "S" or tag.islower():
            hidden_paths.append(path)
    if hidden_paths:
        raise ValueError(
            "formal provenance rejects assume-unchanged or skip-worktree index "
            f"flags in {repository.root}: {hidden_paths}"
        )


def _git_text(root: Path, *arguments: str) -> str:
    """Run Git and return one stripped UTF-8 value."""
    return _git_bytes(root, *arguments).decode("utf-8", errors="strict").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    """Run one fail-closed Git command against an explicit worktree."""
    return _run_command(["git", "-C", str(root), *arguments])


def _run_command(command: list[str]) -> bytes:
    """Run a provenance command without a shell and return exact stdout bytes."""
    return subprocess.run(command, check=True, capture_output=True).stdout


def _is_executed_generated_pb2(path: PurePosixPath) -> bool:
    """Select generated protobuf Python modules while excluding environment copies."""
    excluded_parts = {".venv", "venv", "site-packages", "site_packages", "__pycache__"}
    return (
        not excluded_parts.intersection(path.parts)
        and path.suffix == ".py"
        and "_pb2" in path.name
    )


def _is_ignored_watch_name(name: str) -> bool:
    """Ignore only environment/cache writes that cannot become executed source."""
    return name in _IGNORED_WATCH_NAMES or name.endswith((".pyc", ".pyo"))


def _file_identity(path: Path) -> dict[str, Any]:
    """Hash one required regular artifact and retain its path only as annotation."""
    path = _normalized_absolute_path(path)
    file_descriptor = _open_path_without_symlinks(path, final_flags=os.O_RDONLY)
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"formal provenance artifact must be a regular file: {path}"
            )
        _require_single_link(metadata=before, path=path)
        with os.fdopen(os.dup(file_descriptor), "rb") as file:
            data = file.read()
        after = os.fstat(file_descriptor)
        _require_single_link(metadata=after, path=path)
        if _stable_file_metadata(before) != _stable_file_metadata(after):
            raise RuntimeError(f"formal artifact changed while hashing: {path}")
        if len(data) != before.st_size:
            raise RuntimeError(f"short read while hashing formal artifact: {path}")
    finally:
        os.close(file_descriptor)
    return {
        "path": str(path),
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _canonical_sha256(value: Any) -> str:
    """Hash a strict canonical JSON encoding."""
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _validate_critical_environment(environment: dict[str, str]) -> dict[str, str]:
    """Require and bind the exact environment inherited by formal subprocesses."""
    if set(environment) != _REQUIRED_CRITICAL_ENVIRONMENT:
        raise ValueError(
            "formal critical environment must contain exactly "
            f"{sorted(_REQUIRED_CRITICAL_ENVIRONMENT)}"
        )
    normalized = {name: str(environment[name]) for name in sorted(environment)}
    if normalized["PYTHONDONTWRITEBYTECODE"] != "1":
        raise ValueError("formal PYTHONDONTWRITEBYTECODE must be 1")
    if normalized["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("formal CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    for name, expected in normalized.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                f"formal critical environment {name} differs from os.environ"
            )
    python_path_entries = normalized["PYTHONPATH"].split(os.pathsep)
    if not python_path_entries or any(not entry for entry in python_path_entries):
        raise ValueError("formal PYTHONPATH must contain non-empty absolute entries")
    if any(not Path(entry).is_absolute() for entry in python_path_entries):
        raise ValueError("formal PYTHONPATH entries must be absolute")
    grpc_root = Path(normalized["ALPASIM_GRPC_ROOT"])
    if not grpc_root.is_absolute() or not grpc_root.is_dir():
        raise ValueError("formal ALPASIM_GRPC_ROOT must be an existing absolute dir")
    if normalized["PYTHONPATH"].split(os.pathsep)[0] != str(grpc_root):
        raise ValueError("formal PYTHONPATH must start with ALPASIM_GRPC_ROOT")
    pycache_prefix = Path(normalized["PYTHONPYCACHEPREFIX"])
    if not pycache_prefix.is_absolute() or not pycache_prefix.is_dir():
        raise ValueError("formal PYTHONPYCACHEPREFIX must be an existing absolute dir")
    return normalized


def _exception_identity(failure: BaseException | None) -> dict[str, str] | None:
    """Return a stable, non-lossy-enough receipt view of a lifecycle failure."""
    if failure is None:
        return None
    return {"type": type(failure).__name__, "message": str(failure)}


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write bytes exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(data)


def _write_json_exclusive(path: Path, value: Any) -> None:
    """Write a strict, human-readable JSON artifact exactly once."""
    data = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _write_bytes_exclusive(path, data + b"\n")
    path.chmod(0o444)


def _make_tree_read_only(root: Path) -> None:
    """Remove write bits from a completed immutable snapshot tree."""
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError(f"formal provenance snapshot contains a symlink: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
