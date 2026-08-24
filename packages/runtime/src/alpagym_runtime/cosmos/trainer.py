# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPO, actor-critic PPO, and Flow-PPO adapters for closed-loop training.

The trainer is policy-agnostic: per-policy tokenizer resolution and data
packer construction are looked up via the
``alpagym_runtime.policies.registry`` bundle for the configured
policy kind string.  Cosmos currently exposes their shared scheduling fields
through ``GrpoConfig``; the selected ``trainer_type`` determines the objective.
"""

import copy
import io
import hashlib
import json
import logging
import math
import os
import re
import secrets
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from alpagym_host.checkpoint_resume import (
    ValidatedCheckpointResumeSource,
    canonical_json_sha256,
    read_stable_regular_file,
    validate_checkpoint_resume_source,
    validate_disabled_checkpoint_resume_contract,
)
from alpagym_host.config import RunConfig, load_run_config
from cosmos_rl.dispatcher.data import schema as _rollout_schema
from cosmos_rl.policy import config as _cosmos_config
from cosmos_rl.policy.trainer import base as _trainer_base
from cosmos_rl.policy.trainer.llm_trainer import grpo_trainer as _grpo_trainer
from cosmos_rl.utils import distributed as dist_util, parallelism as _parallelism

from alpagym_runtime.cosmos.replay_objective import (
    assert_replay_shapes,
    compute_flow_ppo_surrogate,
    compute_kl_penalty,
    compute_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.cosmos.rollout_filter import filter_trainable_rollouts
from alpagym_runtime.perf.instrument.lifecycle import initialize_perf
from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.scope import measure_perf
from alpagym_runtime.policies.registry import (
    PolicyCheckpointExportContext,
    get_policy_bundle,
)
from alpagym_runtime.replay import RolloutArtifactIdentity, TrainingSignal
from alpagym_runtime.tensor_utils import to_device_recursive

logger = logging.getLogger(__name__)

_FORMAL_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
_COSMOS_TIMESTAMP = re.compile(r"^[0-9]{14}$")
_PPO_UPDATE_RECEIPT_STATES = frozenset({"pre_rejected", "post_rejected", "accepted"})


def _enforce_strict_training_determinism(config: Any) -> None:
    """Make a requested deterministic trainer fail closed.

    Cosmos enables deterministic algorithms with ``warn_only=True``.  That
    still permits PyTorch's memory-efficient scaled-dot-product-attention
    backward kernel to run nondeterministically, which makes controlled PPO
    calibrations incomparable.  Formal AlpaGym trainers instead require the
    deterministic variant of every selected kernel and reject any operation
    without one.
    """

    if not bool(getattr(config.train, "deterministic", False)):
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic training algorithms were not enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("deterministic training unexpectedly remained warn-only")


def _assert_strict_training_determinism(config: Any) -> None:
    """Fail if another runtime component relaxed the formal trainer contract."""

    if not bool(getattr(config.train, "deterministic", False)):
        return
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic training algorithms became disabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("deterministic training became warn-only")


def _optimizer_learning_rates_before_scheduler(optimizers: Any) -> list[float]:
    """Read every leaf optimizer LR from a Cosmos optimizer container.

    Cosmos's ``OptimizersContainer`` is itself a ``torch.optim.Optimizer`` so
    it exposes a synthetic top-level ``param_groups`` entry.  For multi-part
    models that entry contains only ``optimizers_args``; the actor/critic LRs
    live in the nested leaf optimizers.  Reading the synthetic group therefore
    both loses the per-part rates and raises ``KeyError('lr')`` in production.

    Walk the nested container explicitly and fail closed if a leaf optimizer
    cannot provide a finite, non-negative LR.  A plain PyTorch optimizer is a
    leaf and follows the same validation path.
    """

    learning_rates: list[float] = []
    active_container_ids: set[int] = set()

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return

        nested = getattr(node, "optimizers", None)
        if nested is not None:
            node_id = id(node)
            if node_id in active_container_ids:
                raise ValueError("optimizer container contains a cycle")
            active_container_ids.add(node_id)
            try:
                visit(nested)
            finally:
                active_container_ids.remove(node_id)
            return

        param_groups = getattr(node, "param_groups", None)
        if not isinstance(param_groups, (list, tuple)) or not param_groups:
            raise TypeError("leaf optimizer must expose non-empty param_groups")
        for group in param_groups:
            if not isinstance(group, dict):
                raise TypeError("optimizer param group must be a mapping")
            if "lr" not in group:
                raise KeyError("leaf optimizer param group is missing lr")
            raw_learning_rate = group["lr"]
            if isinstance(raw_learning_rate, torch.Tensor):
                if raw_learning_rate.numel() != 1:
                    raise ValueError("optimizer learning-rate tensor must be scalar")
                raw_learning_rate = raw_learning_rate.detach().cpu().item()
            if isinstance(raw_learning_rate, bool):
                raise TypeError("optimizer learning rate cannot be a boolean")
            try:
                learning_rate = float(raw_learning_rate)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError("optimizer learning rate must be numeric") from error
            if not math.isfinite(learning_rate) or learning_rate < 0.0:
                raise ValueError(
                    "optimizer learning rate must be finite and non-negative"
                )
            learning_rates.append(learning_rate)

    visit(optimizers)
    if not learning_rates:
        raise ValueError("optimizer container has no leaf learning rates")
    return learning_rates


def _optimizer_parameter_id_sets(optimizers: Any) -> list[set[int]]:
    """Return each leaf optimizer's exact owned Parameter identities."""

    parameter_sets: list[set[int]] = []
    active_container_ids: set[int] = set()

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
            return
        nested = getattr(node, "optimizers", None)
        if nested is not None:
            node_id = id(node)
            if node_id in active_container_ids:
                raise ValueError("optimizer container contains a cycle")
            active_container_ids.add(node_id)
            try:
                visit(nested)
            finally:
                active_container_ids.remove(node_id)
            return

        param_groups = getattr(node, "param_groups", None)
        if not isinstance(param_groups, (list, tuple)) or not param_groups:
            raise TypeError("leaf optimizer must expose non-empty param_groups")
        parameter_ids: set[int] = set()
        for group in param_groups:
            if not isinstance(group, dict):
                raise TypeError("optimizer param group must be a mapping")
            raw_parameters = group.get("params")
            if not isinstance(raw_parameters, (list, tuple)):
                raise TypeError(
                    "optimizer param group params must be a concrete sequence"
                )
            for parameter in raw_parameters:
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError("optimizer params must contain torch Parameters")
                parameter_id = id(parameter)
                if parameter_id in parameter_ids:
                    raise ValueError("leaf optimizer owns one Parameter more than once")
                parameter_ids.add(parameter_id)
        if not parameter_ids:
            raise ValueError("leaf optimizer owns no Parameters")
        parameter_sets.append(parameter_ids)

    visit(optimizers)
    if not parameter_sets:
        raise ValueError("optimizer container has no leaf parameter sets")
    return parameter_sets


def _deterministic_optimizer_permutation(
    *,
    num_steps: int,
    base_seed: int,
    current_step: int,
    optimization_iteration: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build and fingerprint a restart-stable optimizer-row permutation.

    The global PyTorch RNG is shared with model initialization and arbitrary
    runtime bookkeeping.  Using it for ``randperm`` made two otherwise equal
    PPO calibration runs consume an unobserved training order.  Derive a
    private CPU generator seed from the immutable training coordinates so the
    exact order is independent of unrelated RNG consumers and reproducible
    after checkpoint resume.
    """

    coordinates = {
        "schema_id": "alpagym.optimizer_permutation.v1",
        "base_seed": base_seed,
        "current_step": current_step,
        "optimization_iteration": optimization_iteration,
        "num_steps": num_steps,
    }
    for name, value in coordinates.items():
        if name == "schema_id":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"optimizer permutation {name} must be an integer")
    if num_steps <= 0:
        raise ValueError("optimizer permutation num_steps must be positive")
    if not 0 <= base_seed <= 2**64 - 1:
        raise ValueError("optimizer permutation base_seed must be uint64")
    if current_step < 0 or optimization_iteration < 0:
        raise ValueError("optimizer permutation step coordinates must be non-negative")

    coordinate_bytes = json.dumps(
        coordinates,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    coordinate_sha256 = hashlib.sha256(coordinate_bytes).hexdigest()
    derived_seed = int.from_bytes(
        bytes.fromhex(coordinate_sha256)[:8], byteorder="big", signed=False
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    indices = torch.randperm(num_steps, generator=generator, device="cpu")
    index_bytes = indices.to(dtype=torch.int64).numpy().tobytes(order="C")
    record: dict[str, Any] = {
        **coordinates,
        "coordinate_sha256": coordinate_sha256,
        "derived_seed": derived_seed,
        "indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "indices_head": [int(index) for index in indices[:16]],
        "indices_tail": [int(index) for index in indices[-16:]],
        "torch_version": torch.__version__,
    }
    return indices, record


def _fixed_reference_reset_interval(value: int | None) -> int:
    """Normalize Cosmos's fixed-anchor spelling and reject moving KL anchors."""
    interval = 0 if value is None else int(value)
    if interval != 0:
        raise ValueError(
            "AlpaGym replay trainers require reference_reset_interval=0 so "
            "the KL anchor remains restart-stable"
        )
    return interval


def _load_run_config(config: _cosmos_config.Config) -> RunConfig:
    """Return the resolved AlpaGym run config referenced by ``config``.

    The cosmos ``Config`` carries the resolved AlpaGym config path under
    ``custom.resolved_config_path``. Raises when the path is absent - every
    production cosmos invocation passes it via ``--config``.
    """
    custom = getattr(config, "custom", None) or {}
    resolved_config_path = (
        custom.get("resolved_config_path") if hasattr(custom, "get") else None
    )
    if not resolved_config_path:
        raise ValueError(
            "Cosmos config missing custom.resolved_config_path; cannot load "
            "AlpaGym run config. Production cosmos invocations set this via --config."
        )
    return load_run_config(resolved_config_path)


def _policy_checkpoint_export_context(
    *,
    config: Any,
    current_step: int,
    total_steps: int,
    optimizer_steps_applied: int,
) -> PolicyCheckpointExportContext:
    """Bind a policy-native export to the formal host run, not a leaf directory.

    Cosmos appends its own timestamp directory to ``train.output_dir`` at load
    time.  Neither that timestamp nor the authored ``cosmos`` directory is the
    AlpaGym run identity.  The canonical identity is the generated host run
    directory recorded in ``resolved_config.yaml``.
    """
    custom = getattr(config, "custom", None) or {}
    resolved_config_value = (
        custom.get("resolved_config_path") if hasattr(custom, "get") else None
    )
    if not isinstance(resolved_config_value, str) or not resolved_config_value:
        raise ValueError(
            "policy-native checkpoint export requires custom.resolved_config_path"
        )
    resolved_config_path = Path(resolved_config_value).expanduser()
    if resolved_config_path.is_symlink() or not resolved_config_path.is_file():
        raise ValueError(
            "policy-native checkpoint export resolved config must be a regular file"
        )
    digest = hashlib.sha256()
    with resolved_config_path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    run_config = _load_run_config(config)
    formal_run_dir = Path(run_config.artifact_paths.run_dir).expanduser()
    canonical_config_path = Path(
        run_config.artifact_paths.resolved_config_path
    ).expanduser()
    if not formal_run_dir.is_absolute() or not canonical_config_path.is_absolute():
        raise ValueError("policy-native checkpoint formal run paths must be absolute")
    if not formal_run_dir.is_dir() or formal_run_dir.is_symlink():
        raise ValueError(
            "policy-native checkpoint formal run directory must be a regular directory"
        )
    if not canonical_config_path.is_file() or canonical_config_path.is_symlink():
        raise ValueError(
            "policy-native checkpoint canonical resolved config must be a regular file"
        )
    if not os.path.samefile(resolved_config_path, canonical_config_path):
        raise ValueError(
            "policy-native checkpoint resolved config differs from the canonical "
            "formal-run config"
        )
    formal_run_id = formal_run_dir.name
    if _FORMAL_RUN_ID.fullmatch(formal_run_id) is None:
        raise ValueError(
            "policy-native checkpoint canonical run_dir has no formal run ID"
        )

    output_dir = Path(str(config.train.output_dir)).expanduser()
    if (
        not output_dir.is_absolute()
        or not output_dir.is_dir()
        or output_dir.is_symlink()
    ):
        raise ValueError(
            "policy-native checkpoint Cosmos output_dir must be an absolute regular "
            "directory"
        )
    cosmos_timestamp = getattr(config.train, "timestamp", None)
    if (
        not isinstance(cosmos_timestamp, str)
        or _COSMOS_TIMESTAMP.fullmatch(cosmos_timestamp) is None
        or output_dir.name != cosmos_timestamp
    ):
        raise ValueError(
            "policy-native checkpoint Cosmos output_dir must end in its runtime "
            "timestamp"
        )
    expected_cosmos_root = formal_run_dir / "cosmos"
    if output_dir.parent != expected_cosmos_root:
        raise ValueError(
            "policy-native checkpoint Cosmos output_dir is not owned by the "
            "canonical formal run"
        )
    return PolicyCheckpointExportContext(
        training_step=current_step,
        total_training_steps=total_steps,
        optimizer_steps_applied=optimizer_steps_applied,
        resolved_config_sha256=digest.hexdigest(),
        cosmos_run_id=formal_run_id,
    )


def _validated_runtime_resume_source(
    *,
    config: Any,
    run_config: RunConfig,
) -> ValidatedCheckpointResumeSource | None:
    """Bind Cosmos's public string to the host-authored typed contract."""

    contract = run_config.cosmos.train.resume
    runtime_resume = getattr(config.train, "resume", False)
    if not contract.enabled:
        validate_disabled_checkpoint_resume_contract(contract)
        if runtime_resume not in (False, None):
            raise ValueError(
                "Cosmos train.resume is set but the AlpaGym typed resume contract "
                "is disabled"
            )
        return None
    source = validate_checkpoint_resume_source(contract)
    if not isinstance(runtime_resume, str):
        raise ValueError(
            "enabled AlpaGym checkpoint resume requires Cosmos train.resume to be "
            "the exact policy-directory string"
        )
    if runtime_resume != str(source.checkpoint_path):
        raise ValueError(
            "Cosmos train.resume differs from the exact typed checkpoint path"
        )
    return source


def _load_checkpoint_extra_info(
    source: ValidatedCheckpointResumeSource,
    *,
    rank: int,
    expected_step: int,
    expected_next_step: int,
) -> dict[str, Any]:
    """Read and validate the rank-local native extra-info before mutation."""

    if rank not in source.ranks:
        raise ValueError(f"checkpoint tree has no complete state for rank {rank}")
    filename = f"extra_info_rank_{rank}.pth"
    expected_file = source.snapshot.file(filename)
    data, metadata = read_stable_regular_file(source.checkpoint_path / filename)
    if (
        metadata.st_size != expected_file.size_bytes
        or hashlib.sha256(data).hexdigest() != expected_file.sha256
    ):
        raise RuntimeError("checkpoint extra-info changed after tree validation")
    payload = torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("native Cosmos checkpoint extra-info must be a mapping")
    required = {"rng_state", "step", "total_steps", "remain_samples_num", "is_final"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"native Cosmos checkpoint extra-info is missing: {sorted(missing)}"
        )
    for field_name in ("step", "total_steps", "remain_samples_num"):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"checkpoint extra-info {field_name} must be an integer")
    if payload["step"] != expected_step:
        raise ValueError("checkpoint extra-info step differs from resume contract")
    if expected_next_step != payload["step"] + 1:
        raise ValueError("checkpoint extra-info does not precede expected next step")
    if payload["total_steps"] < payload["step"]:
        raise ValueError("checkpoint extra-info total_steps precedes its step")
    if not isinstance(payload["is_final"], bool):
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
    return payload


def _nested_state_equal(first: Any, second: Any) -> bool:
    """Compare nested RNG states without ambiguous tensor/array truth values."""

    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return first.shape == second.shape and torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return first.dtype == second.dtype and np.array_equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            _nested_state_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _nested_state_equal(left, right)
            for left, right in zip(first, second, strict=True)
        )
    try:
        result = first == second
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _validate_native_resume_result(
    *,
    extra_info: dict[str, Any],
    restored: Any,
    current_rng_state: Any,
) -> None:
    """Prove that native load returned the exact step and restored RNG."""

    if not isinstance(restored, dict):
        raise TypeError("Cosmos native resume result must be a mapping")
    for name in ("step", "total_steps", "remain_samples_num", "is_final"):
        if restored.get(name) != extra_info[name]:
            raise ValueError(f"Cosmos native resume result changed {name}")
    if not _nested_state_equal(current_rng_state, extra_info["rng_state"]):
        raise RuntimeError(
            "Cosmos native resume did not restore the checkpoint RNG state"
        )


def _checkpoint_restore_receipt_path(
    *,
    config: Any,
    run_config: RunConfig,
    rank: int,
) -> tuple[Path, str, str]:
    """Resolve the current formal-run receipt path and immutable config identity."""

    run_dir = Path(run_config.artifact_paths.run_dir)
    resolved_config = Path(run_config.artifact_paths.resolved_config_path)
    custom = getattr(config, "custom", None) or {}
    configured_resolved = (
        custom.get("resolved_config_path") if hasattr(custom, "get") else None
    )
    if not run_dir.is_absolute() or _FORMAL_RUN_ID.fullmatch(run_dir.name) is None:
        raise ValueError("checkpoint restore receipt requires a formal current run")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("checkpoint restore receipt current run_dir is invalid")
    if (
        not isinstance(configured_resolved, str)
        or Path(configured_resolved) != resolved_config
        or not resolved_config.is_file()
        or resolved_config.is_symlink()
    ):
        raise ValueError(
            "checkpoint restore receipt resolved config binding is invalid"
        )
    output_dir = Path(str(config.train.output_dir))
    timestamp = getattr(config.train, "timestamp", None)
    if (
        not isinstance(timestamp, str)
        or _COSMOS_TIMESTAMP.fullmatch(timestamp) is None
        or output_dir != run_dir / "cosmos" / timestamp
        or not output_dir.is_dir()
        or output_dir.is_symlink()
    ):
        raise ValueError("checkpoint restore receipt Cosmos output binding is invalid")
    config_data, _metadata = read_stable_regular_file(resolved_config)
    receipt_dir = Path(run_config.artifact_paths.artifacts_dir) / (
        "checkpoint_resume_receipts"
    )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if receipt_dir.is_symlink() or receipt_dir.resolve(strict=True) != receipt_dir:
        raise ValueError("checkpoint restore receipt directory is not canonical")
    return (
        receipt_dir / f"rank_{rank}.json",
        run_dir.name,
        hashlib.sha256(config_data).hexdigest(),
    )


def _write_checkpoint_restore_receipt(
    *,
    config: Any,
    run_config: RunConfig,
    source: ValidatedCheckpointResumeSource,
    rank: int,
    restored: dict[str, Any],
) -> Path:
    """Persist one exclusive per-rank receipt after every restore check passes."""

    path, current_run_id, resolved_config_sha256 = _checkpoint_restore_receipt_path(
        config=config,
        run_config=run_config,
        rank=rank,
    )
    contract = run_config.cosmos.train.resume
    receipt = {
        "schema_id": "alpagym.cosmos_checkpoint_restore.v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "current_formal_run_id": current_run_id,
        "current_resolved_config_sha256": resolved_config_sha256,
        "rank": rank,
        "prior_formal_run_id": contract.prior_formal_run_id,
        "prior_postrun_receipt_sha256": source.prior_postrun_receipt_sha256,
        "checkpoint_path": str(source.checkpoint_path),
        "checkpoint_step": contract.checkpoint_step,
        "checkpoint_tree_sha256": source.snapshot.tree_sha256,
        "checkpoint_file_count": source.snapshot.file_count,
        "checkpoint_total_size_bytes": source.snapshot.total_size_bytes,
        "restored_training_step": restored["step"],
        "expected_next_training_step": contract.expected_next_training_step,
        "restored_prior_total_steps": restored["total_steps"],
        "restored_remain_samples_num": restored["remain_samples_num"],
        "restored_prior_is_final": restored["is_final"],
        "native_restore": {
            "loader": "cosmos_rl.CheckpointMananger.load_checkpoint",
            "loader_completed": True,
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored_and_verified": True,
            "training_step_restored_and_verified": True,
            "checkpoint_tree_revalidated_after_load": True,
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return path


def _ppo_update_receipt_context(
    *,
    config: Any,
    run_config: RunConfig,
    rank: int,
    current_step: int,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one formal-run-bound immutable PPO diagnostic destination."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("PPO update diagnostic receipt requires a valid global rank")
    if (
        isinstance(current_step, bool)
        or not isinstance(current_step, int)
        or current_step < 0
    ):
        raise ValueError("PPO update diagnostic receipt requires a valid trainer step")

    run_dir = Path(run_config.artifact_paths.run_dir).expanduser()
    artifacts_dir = Path(run_config.artifact_paths.artifacts_dir).expanduser()
    resolved_config = Path(run_config.artifact_paths.resolved_config_path).expanduser()
    if (
        not run_dir.is_absolute()
        or _FORMAL_RUN_ID.fullmatch(run_dir.name) is None
        or not run_dir.is_dir()
        or run_dir.is_symlink()
        or run_dir.resolve(strict=True) != run_dir
    ):
        raise ValueError(
            "PPO update diagnostic receipt requires a canonical formal run"
        )
    if (
        artifacts_dir != run_dir / "artifacts"
        or not artifacts_dir.is_dir()
        or artifacts_dir.is_symlink()
        or artifacts_dir.resolve(strict=True) != artifacts_dir
    ):
        raise ValueError("PPO update diagnostic artifact directory is invalid")

    custom = getattr(config, "custom", None) or {}
    configured_resolved = (
        custom.get("resolved_config_path") if hasattr(custom, "get") else None
    )
    if (
        not isinstance(configured_resolved, str)
        or Path(configured_resolved).expanduser() != resolved_config
        or not resolved_config.is_absolute()
        or not resolved_config.is_file()
        or resolved_config.is_symlink()
        or resolved_config.resolve(strict=True) != resolved_config
    ):
        raise ValueError("PPO update diagnostic resolved-config binding is invalid")
    resolved_config_data, resolved_config_metadata = read_stable_regular_file(
        resolved_config
    )

    output_dir = Path(str(config.train.output_dir)).expanduser()
    timestamp = getattr(config.train, "timestamp", None)
    if (
        not isinstance(timestamp, str)
        or _COSMOS_TIMESTAMP.fullmatch(timestamp) is None
        or output_dir != run_dir / "cosmos" / timestamp
        or not output_dir.is_dir()
        or output_dir.is_symlink()
        or output_dir.resolve(strict=True) != output_dir
    ):
        raise ValueError("PPO update diagnostic Cosmos output binding is invalid")

    receipt_dir = artifacts_dir / "ppo_update_diagnostics"
    receipt_dir.mkdir(mode=0o755, parents=False, exist_ok=True)
    if (
        receipt_dir.is_symlink()
        or not receipt_dir.is_dir()
        or receipt_dir.resolve(strict=True) != receipt_dir
    ):
        raise ValueError("PPO update diagnostic receipt directory is invalid")
    return (
        receipt_dir / f"step_{current_step}_rank_{rank}.json",
        {
            "formal_run_id": run_dir.name,
            "resolved_config_relative_path": resolved_config.relative_to(
                run_dir
            ).as_posix(),
            "resolved_config_sha256": hashlib.sha256(resolved_config_data).hexdigest(),
            "resolved_config_size_bytes": resolved_config_metadata.st_size,
            "cosmos_output_relative_path": output_dir.relative_to(run_dir).as_posix(),
        },
    )


def _write_bytes_atomic_immutable(path: Path, payload: bytes) -> None:
    """Publish bytes once through an atomic no-replace hard-link operation.

    The temporary inode is fully written, fsynced, and made read-only before it
    becomes visible at ``path``. ``os.link`` is the atomic no-clobber publish
    primitive: an existing final receipt makes the trainer fail closed.
    """

    temp_path = path.parent / (f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temp_path, flags, 0o600)
    published = False
    try:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        os.link(temp_path, path, follow_symlinks=False)
        published = True
        temp_path.unlink()
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temp_path):
            temp_path.unlink()
        if published:
            metadata = os.lstat(path)
            if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
                raise RuntimeError(
                    "published PPO update diagnostic receipt is not immutable"
                )


def _consumed_rollout_artifact_records(
    rollouts: list[Any],
    samples: list[Any],
    *,
    flow_chunk_density: bool,
    formal_run_root: Path,
) -> tuple[list[dict[str, Any]], str, int]:
    """Bind the exact ordered disk artifacts that produced an optimizer batch."""

    samples_by_rollout: dict[str, list[Any]] = {}
    for sample in samples:
        rollout_id = getattr(sample, "rollout_id", None)
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("PPO receipt sample has no rollout_id")
        samples_by_rollout.setdefault(rollout_id, []).append(sample)

    raw_formal_root = Path(formal_run_root).expanduser()
    canonical_formal_root = raw_formal_root.resolve(strict=True)
    if (
        not raw_formal_root.is_absolute()
        or raw_formal_root != canonical_formal_root
        or _FORMAL_RUN_ID.fullmatch(canonical_formal_root.name) is None
        or not canonical_formal_root.is_dir()
    ):
        raise ValueError("PPO receipt formal run root is invalid")

    records: list[dict[str, Any]] = []
    seen_rollout_ids: set[str] = set()
    seen_rollout_seeds: set[int] = set()
    for rollout_index, rollout in enumerate(rollouts):
        completion = getattr(rollout, "completion", None)
        if not isinstance(completion, (str, os.PathLike)):
            raise TypeError(
                "formal PPO receipt requires disk-episode rollout completions"
            )
        completion_source = Path(completion).expanduser()
        if not completion_source.is_absolute() or ".." in completion_source.parts:
            raise ValueError(
                "PPO receipt completion path must be absolute and normalized"
            )
        if (
            completion_source.parent != canonical_formal_root / "artifacts"
            or completion_source.is_symlink()
            or not completion_source.is_file()
        ):
            raise ValueError("PPO receipt completion is outside formal artifacts")
        completion_path = completion_source.resolve(strict=True)

        matching_samples = [
            sample
            for rollout_samples in samples_by_rollout.values()
            for sample in rollout_samples
            if getattr(
                getattr(sample, "artifact_identity", None),
                "completion_path",
                None,
            )
            == str(completion_path)
        ]
        identities = {
            getattr(sample, "artifact_identity", None) for sample in matching_samples
        }
        if len(identities) != 1:
            raise ValueError(
                "PPO receipt samples do not share one loaded artifact identity"
            )
        identity = next(iter(identities))
        if not isinstance(identity, RolloutArtifactIdentity):
            raise TypeError("PPO receipt sample has no loaded disk-artifact identity")
        session_uuid = identity.session_uuid
        rollout_seed = identity.rollout_seed
        scene_id = identity.scene_id
        num_steps = identity.num_steps
        if (
            not isinstance(session_uuid, str)
            or not session_uuid
            or session_uuid in seen_rollout_ids
            or isinstance(rollout_seed, bool)
            or not isinstance(rollout_seed, int)
            or not 0 <= rollout_seed < 2**64
            or rollout_seed in seen_rollout_seeds
            or not isinstance(scene_id, str)
            or not scene_id
            or isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("PPO receipt disk episode ownership is invalid")
        filename = identity.tensor_sidecar_filename
        sidecar_sha256 = identity.tensor_sidecar_sha256
        sidecar_size = identity.tensor_sidecar_size_bytes
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(sidecar_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sidecar_sha256) is None
            or isinstance(sidecar_size, bool)
            or not isinstance(sidecar_size, int)
            or sidecar_size <= 0
        ):
            raise ValueError("PPO receipt tensor-sidecar identity is invalid")
        if (
            not isinstance(identity.episode_file_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", identity.episode_file_sha256) is None
            or isinstance(identity.episode_file_size_bytes, bool)
            or not isinstance(identity.episode_file_size_bytes, int)
            or identity.episode_file_size_bytes <= 0
            or not isinstance(identity.episode_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", identity.episode_manifest_sha256) is None
        ):
            raise ValueError("PPO receipt loaded episode identity is invalid")

        rollout_samples = samples_by_rollout.pop(session_uuid, None)
        if not rollout_samples:
            raise ValueError("PPO receipt rollout has no optimizer samples")
        if matching_samples != rollout_samples:
            raise ValueError(
                "PPO receipt optimizer rows differ from the loaded disk episode"
            )
        actor_rows = 0
        valid_sample_rows = 0
        for sample in rollout_samples:
            if getattr(sample, "artifact_identity", None) != identity:
                raise ValueError("PPO receipt rollout mixes artifact identities")
            signal = getattr(sample, "training_signal", None)
            if not isinstance(signal, TrainingSignal):
                raise TypeError("PPO receipt sample training signal is invalid")
            is_padding = getattr(signal, "is_padding", None)
            if (
                not isinstance(is_padding, torch.Tensor)
                or is_padding.numel() != 1
                or is_padding.dtype != torch.bool
            ):
                raise ValueError("PPO receipt sample validity signal is invalid")
            padding_row = bool(is_padding.item())
            actor_mask = _ppo_actor_valid_mask(
                signal,
                required=flow_chunk_density,
            )
            if actor_mask.numel() != 1 or actor_mask.dtype != torch.bool:
                raise ValueError("PPO receipt actor-valid mask is not scalar")
            actor_valid = bool(actor_mask.item())
            if padding_row and actor_valid:
                raise ValueError("PPO receipt padding row is actor-valid")
            if not padding_row:
                valid_sample_rows += 1
                actor_rows += int(actor_valid)
        if valid_sample_rows != num_steps:
            raise ValueError(
                "PPO receipt valid optimizer rows differ from the disk episode"
            )
        weight_version = getattr(rollout, "weight_version", None)
        if isinstance(weight_version, bool) or not isinstance(weight_version, int):
            raise TypeError("PPO receipt rollout weight version is invalid")
        seen_rollout_ids.add(session_uuid)
        seen_rollout_seeds.add(rollout_seed)
        records.append(
            {
                "rollout_index": rollout_index,
                "transport_kind": "disk_episode_v2",
                "completion_relative_path": completion_path.relative_to(
                    canonical_formal_root
                ).as_posix(),
                "episode_file_sha256": identity.episode_file_sha256,
                "episode_file_size_bytes": identity.episode_file_size_bytes,
                "episode_manifest_sha256": identity.episode_manifest_sha256,
                "tensor_sidecar": {
                    "filename": filename,
                    "sha256": sidecar_sha256,
                    "size_bytes": sidecar_size,
                },
                "session_uuid": session_uuid,
                "rollout_seed": rollout_seed,
                "scene_id": scene_id,
                "num_steps": num_steps,
                "behavior_weight_version": weight_version,
                "optimizer_sample_rows": len(rollout_samples),
                "optimizer_actor_valid_rows": actor_rows,
            }
        )
    if samples_by_rollout:
        raise ValueError("PPO receipt has optimizer samples from an undeclared rollout")
    actor_sample_rows = sum(
        int(record["optimizer_actor_valid_rows"]) for record in records
    )
    return records, canonical_json_sha256(records), actor_sample_rows


def _write_ppo_update_diagnostic_receipt(
    *,
    config: Any,
    run_config: RunConfig,
    rank: int,
    current_step: int,
    total_steps: int,
    state: str,
    received_rollouts: int,
    trainable_rollouts: int,
    sample_rows: int,
    actor_sample_rows: int,
    behavior_weight_versions: list[int],
    consumed_rollout_artifacts: list[dict[str, Any]],
    consumed_rollout_batch_sha256: str,
    is_master_replica: bool,
    do_save_checkpoint: bool,
    pre_update_metrics: dict[str, float | int],
    optimizer_metrics: dict[str, Any] | None,
    post_update_metrics: dict[str, float | int] | None,
    rejection: BaseException | None,
) -> Path:
    """Persist the terminal pre-reject, post-reject, or accepted update state."""

    if state not in _PPO_UPDATE_RECEIPT_STATES:
        raise ValueError(f"unsupported PPO update receipt state: {state!r}")
    if state == "pre_rejected":
        if optimizer_metrics is not None or post_update_metrics is not None:
            raise ValueError("pre-rejected PPO receipt cannot contain post-update data")
        if rejection is None:
            raise ValueError("pre-rejected PPO receipt requires a rejection")
    elif state == "post_rejected":
        if optimizer_metrics is None:
            raise ValueError("post-rejected PPO receipt requires optimizer data")
        # A post-update diagnostic can itself fail after the optimizer has
        # mutated parameters.  In that case there is intentionally no metric
        # payload to pretend describes the live (restored) actor.
        if rejection is None:
            raise ValueError("post-rejected PPO receipt requires a rejection")
    else:
        if optimizer_metrics is None or post_update_metrics is None:
            raise ValueError("accepted PPO receipt requires complete post-update data")
        if rejection is not None:
            raise ValueError("accepted PPO receipt cannot contain a rejection")

    path, binding = _ppo_update_receipt_context(
        config=config,
        run_config=run_config,
        rank=rank,
        current_step=current_step,
    )
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        **binding,
        "rank": rank,
        "current_step": current_step,
        "total_steps": total_steps,
        "state": state,
        "received_rollouts": received_rollouts,
        "trainable_rollouts": trainable_rollouts,
        "sample_rows": sample_rows,
        "actor_sample_rows": actor_sample_rows,
        "behavior_weight_versions": sorted(behavior_weight_versions),
        "consumed_rollout_artifacts": consumed_rollout_artifacts,
        "consumed_rollout_batch_sha256": consumed_rollout_batch_sha256,
        "is_master_replica": is_master_replica,
        "checkpoint_requested": do_save_checkpoint,
        "boundary": {
            "optimizer_steps_applied": 0
            if optimizer_metrics is None
            else int(optimizer_metrics["train/optimizer_steps_applied"]),
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": pre_update_metrics,
        "optimizer_metrics": optimizer_metrics,
        "post_update_metrics": post_update_metrics,
        "rejection": None
        if rejection is None
        else {"type": type(rejection).__name__, "message": str(rejection)},
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    payload = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic_immutable(path, payload)
    return path


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_grpo")
class AlpagymGRPOTrainer(_grpo_trainer.GRPOTrainer):
    """GRPO trainer that replaces Cosmos's text-token loss with AlpaGym replay training.

    Consumes AlpaGym rollout artifact completions, recomputes logprobs for each
    recorded selected action, and applies the PPO/GRPO replay objective. The
    step is the minibatching unit: all rollouts are flattened into one pool of
    per-step replay samples, shuffled, then split into minibatches.

    Forward contract for ``self.model``:

        model(
            **model_inputs,                # whatever the policy's data packer collated
            teacher_model: nn.Module | None,
        ) -> {
            "log_probs": Tensor[R],        # current logprob for recorded action
            "kl_div":    Tensor[R] | None,
        }

    ``kl_div`` (when present) is one scalar per replay row; the trainer excludes
    padded rows before reducing KL. Additional return keys are allowed but
    ignored by the trainer.

    ``PolicyOutput.replay_data`` must carry a ``PolicyReplayData`` envelope.
    The packer raises before collation if the payload family, old rollout
    logprob, selected action data, or required trace fields are missing.
    """

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Cosmos-RL config; reads `train.train_policy.*` hyperparams
                and `policy.model_name_or_path` for the policy bundle.
            parallel_dims: Cosmos-RL parallelism description.
            **kwargs: Forwarded to `GRPOTrainer.__init__`
                (`train_stream`, `data_packer`, `val_data_packer`, ...).
        """
        # Loaded once and reused for perf init and the policy-bundle lookup below.
        # Production cosmos invocations always set `custom.resolved_config_path` via
        # `--config`.
        run_config = _load_run_config(config)
        initialize_perf(run_config)
        # Cosmos's super-init resolves a tokenizer from
        # ``config.policy.model_name_or_path`` and calls ``ModelRegistry.build_model``.
        # Policy bundles whose path lacks tokenizer files need that resolution
        # against a per-bundle location, and need their model
        # registered with cosmos beforehand; the registered bundle's
        # ``setup_tokenizer`` handles both.
        bundle = get_policy_bundle(run_config.policy.model.kind)
        self._policy_bundle = bundle
        if run_config.alpasim.simulation_domain == "humanoid":
            scene_ids = run_config.dataset.scene_ids
            if scene_ids is None or not scene_ids:
                raise ValueError(
                    "humanoid training requires explicit dataset.scene_ids"
                )
            self._global_train_batch_size = (
                run_config.cosmos.train.train_batch_per_replica
                * run_config.cosmos.launch.policy_replicas
            )
            logical_rollout_samples = (
                len(scene_ids)
                * run_config.cosmos.rollout.n_generation
                * run_config.cosmos.train.num_epochs
            )
            if logical_rollout_samples % self._global_train_batch_size != 0:
                raise ValueError(
                    "humanoid training horizon is not divisible by the global batch"
                )
            self._logical_total_training_steps = (
                logical_rollout_samples // self._global_train_batch_size
            )
        bundle_tokenizer = bundle.setup_tokenizer(config)
        if bundle_tokenizer is not None:
            self.tokenizer = bundle_tokenizer

        # Apply before Cosmos constructs the model so attention backend
        # selection is deterministic during any initialization forwards.
        _enforce_strict_training_determinism(config)
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        # Cosmos's LLMTrainer currently resets this to warn_only=True.  Restore
        # the formal fail-closed contract before the first rollout replay and
        # backward pass.
        _enforce_strict_training_determinism(config)
        if bool(getattr(config.train, "deterministic", False)):
            logger.info(
                "Strict deterministic training enabled: algorithms=true warn_only=false"
            )

        # A final real trainer step and Cosmos's synthetic
        # TrainingCompleteCommand can both save the same step.  Candidate
        # overlays are immutable, so remember exports completed by this exact
        # trainer instance and never invoke an exporter twice for that step.
        self._completed_policy_native_exports: dict[
            int, tuple[Path, PolicyCheckpointExportContext]
        ] = {}

        grpo_config = config.train.train_policy
        if not isinstance(grpo_config, _cosmos_config.GrpoConfig):
            raise TypeError("config.train.train_policy must be GrpoConfig.")
        self._grpo_ratio_clip_low: float = float(grpo_config.epsilon_low)
        self._grpo_ratio_clip_high: float = float(grpo_config.epsilon_high)
        self._grpo_optimization_iterations: int = int(grpo_config.mu_iterations)
        self._mini_batch: int = int(grpo_config.mini_batch)
        self._kl_beta: float = float(grpo_config.kl_beta)
        self._allowed_outdated_steps: int = int(grpo_config.allowed_outdated_steps)
        self._on_policy: bool = bool(grpo_config.on_policy)
        # Cosmos optionally allows reference_reset_interval=None to mean "never";
        # normalize that to the restart-stable fixed-anchor value.
        self._reference_reset_interval = _fixed_reference_reset_interval(
            grpo_config.reference_reset_interval
        )
        # GRPO groups one prompt's rollouts together; that count lives on the
        # rollout config in Cosmos.
        self._group_size: int = int(config.rollout.n_generation)

        self._reference_model: Any = None
        record_perf_marker("trainer/ready", cpu_snapshot=True, gpu_snapshot=True)

    @measure_perf(
        "trainer/step",
        category="compute_gpu_wall",
        cpu_snapshot=True,
        gpu_snapshot=True,
    )
    def step_training(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        rollouts: list[_rollout_schema.Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one GRPO step over the rollouts Cosmos provides.

        Filters stale rollouts, builds per-step samples via the data packer,
        runs ``grpo_optimization_iterations × num_mini_batches`` PPO updates, and returns
        the metrics dict Cosmos reports.

        Args:
            rollouts: Cosmos-RL rollouts; each carries ``prompt``,
                ``completion`` (artifact path), ``advantage``, and
                ``weight_version``.
            current_step: Current training step index.
            total_steps: Total configured training steps.
            remain_samples_num: Remaining samples reported by Cosmos-RL; forwarded
                to the checkpoint manager so resume restores the same data pointer.
            inter_policy_nccl: NCCL communicator across DP replicas.
            is_master_replica: Whether this is the master policy replica; only
                the master writes checkpoints.
            do_save_checkpoint: Whether Cosmos-RL requested checkpointing this step.

        Returns:
            Dict of training metrics for Cosmos to log.
        """
        del kwargs
        _assert_strict_training_determinism(self.config)
        received_rollout_count = len(rollouts)
        logger.info(
            "AlpaGym trainer step start current_step=%d total_steps=%d received_rollouts=%d "
            "group_size=%d mini_batch=%d grpo_optimization_iterations=%d "
            "is_master_replica=%s do_save_checkpoint=%s",
            current_step,
            total_steps,
            len(rollouts),
            self._group_size,
            self._mini_batch,
            self._grpo_optimization_iterations,
            is_master_replica,
            do_save_checkpoint,
        )
        rollouts = filter_trainable_rollouts(
            rollouts,
            current_step=current_step,
            train_batch_per_replica=int(self.config.train.train_batch_per_replica),
            allowed_outdated_steps=self._allowed_outdated_steps,
        )
        if self._on_policy:
            expected_behavior_version = max(current_step - 1, 0)
            kept_versions = [int(rollout.weight_version) for rollout in rollouts]
            if any(version != expected_behavior_version for version in kept_versions):
                raise ValueError(
                    "On-policy AlpaGym training requires the exact behavior "
                    f"version {expected_behavior_version} for optimizer update "
                    f"{current_step}, got {kept_versions}"
                )
        samples, advantages = self._prepare_training_data(rollouts)
        if not samples:
            raise ValueError(
                "AlpaGym trainer step has no trainable samples after filtering: "
                f"current_step={current_step}"
            )

        write_ppo_receipt = bool(
            getattr(self, "_write_ppo_update_diagnostic_receipts", False)
        )
        receipt_run_config: RunConfig | None = None
        receipt_rank: int | None = None
        behavior_weight_versions: list[int] = []
        consumed_rollout_artifacts: list[dict[str, Any]] = []
        consumed_rollout_batch_sha256 = ""
        actor_sample_rows = 0
        optimizer_learning_rates_before_scheduler: list[float] | None = None
        behavior_kl_backtrack = bool(getattr(self, "_behavior_kl_backtrack", False))
        if write_ppo_receipt:
            behavior_weight_versions = [
                int(rollout.weight_version) for rollout in rollouts
            ]
            receipt_run_config = _load_run_config(self.config)
            (
                consumed_rollout_artifacts,
                consumed_rollout_batch_sha256,
                actor_sample_rows,
            ) = _consumed_rollout_artifact_records(
                rollouts,
                samples,
                flow_chunk_density=bool(getattr(self, "_flow_chunk_density", False)),
                formal_run_root=receipt_run_config.artifact_paths.run_dir,
            )
            candidate_rank = getattr(self.ckpt_manager, "global_rank", None)
            if (
                isinstance(candidate_rank, bool)
                or not isinstance(candidate_rank, int)
                or candidate_rank < 0
            ):
                raise ValueError(
                    "PPO update diagnostic receipt requires a valid global rank"
                )
            receipt_rank = candidate_rank
            receipt_path, _receipt_binding = _ppo_update_receipt_context(
                config=self.config,
                run_config=receipt_run_config,
                rank=receipt_rank,
                current_step=current_step,
            )
            if os.path.lexists(receipt_path):
                raise FileExistsError(
                    "PPO update diagnostic receipt already exists before optimizer "
                    f"entry: {receipt_path}"
                )
        if write_ppo_receipt or behavior_kl_backtrack:
            # Audit the real leaf optimizer topology before any parameter can
            # mutate.  A new or malformed container must fail at the entry
            # boundary, never after an otherwise successful optimizer step.
            optimizer_learning_rates_before_scheduler = (
                _optimizer_learning_rates_before_scheduler(self.optimizers)
            )

        pre_update_metrics = self._pre_update_diagnostics(samples)
        actor_snapshot: tuple[list[torch.nn.Parameter], list[torch.Tensor]] | None = (
            None
        )
        try:
            self._validate_update_diagnostics(
                pre_update_metrics,
                phase="pre_update",
            )
            if behavior_kl_backtrack:
                actor_snapshot = self._snapshot_actor_for_kl_backtracking()
        except Exception as guard_failure:
            if write_ppo_receipt:
                assert receipt_run_config is not None
                assert receipt_rank is not None
                try:
                    receipt_path = _write_ppo_update_diagnostic_receipt(
                        config=self.config,
                        run_config=receipt_run_config,
                        rank=receipt_rank,
                        current_step=current_step,
                        total_steps=total_steps,
                        state="pre_rejected",
                        received_rollouts=received_rollout_count,
                        trainable_rollouts=len(rollouts),
                        sample_rows=len(samples),
                        actor_sample_rows=actor_sample_rows,
                        behavior_weight_versions=behavior_weight_versions,
                        consumed_rollout_artifacts=consumed_rollout_artifacts,
                        consumed_rollout_batch_sha256=(consumed_rollout_batch_sha256),
                        is_master_replica=is_master_replica,
                        do_save_checkpoint=do_save_checkpoint,
                        pre_update_metrics=pre_update_metrics,
                        optimizer_metrics=None,
                        post_update_metrics=None,
                        rejection=guard_failure,
                    )
                    logger.error(
                        "PPO pre-update guard rejection receipt: %s", receipt_path
                    )
                except Exception as receipt_failure:
                    raise ExceptionGroup(
                        "PPO pre-update guard and diagnostic receipt both failed",
                        [guard_failure, receipt_failure],
                    ) from None
            raise
        self._active_optimizer_step = current_step
        try:
            (
                total_loss,
                total_kl,
                num_batches,
                ratio_max,
                ratio_min,
                clip_fraction_sum,
                grad_norm_sum,
            ) = self._run_training_loop(samples, advantages, inter_policy_nccl)
        finally:
            del self._active_optimizer_step
        optimizer_steps_applied = self._optimizer_steps_applied_in_training_step
        optimizer_metrics: dict[str, Any] | None = None
        if write_ppo_receipt:
            assert optimizer_learning_rates_before_scheduler is not None
            optimizer_metrics = {
                "train/loss_sum": total_loss,
                "train/loss_avg_local": total_loss / num_batches
                if num_batches
                else 0.0,
                "train/kl_sum": total_kl,
                "train/kl_avg_local": total_kl / num_batches if num_batches else 0.0,
                "train/num_batches": num_batches,
                "train/num_micro_batches": int(
                    getattr(self, "_last_micro_batches", num_batches)
                ),
                "train/optimizer_steps_applied": optimizer_steps_applied,
                "train/ratio_max": ratio_max,
                "train/ratio_min": ratio_min,
                "train/clip_fraction": clip_fraction_sum / num_batches
                if num_batches
                else 0.0,
                "train/grad_norm": grad_norm_sum / num_batches if num_batches else 0.0,
                "train/optimizer_learning_rates_before_scheduler": (
                    optimizer_learning_rates_before_scheduler
                ),
                "train/target_behavior_kl": getattr(self, "_target_behavior_kl", None),
                **getattr(self, "_last_optimizer_permutation_metrics", {}),
                **getattr(self, "_last_ppo_advantage_metrics", {}),
            }
        self._last_behavior_kl_backtrack_metrics = {}
        post_update_metrics: dict[str, float | int] | None = None
        try:
            # This no-grad replay describes the final candidate policy.  Keep
            # it inside the mutation guard because the diagnostic itself may
            # fail after Adam has already changed parameters.
            post_update_metrics = self._post_update_diagnostics(samples)
            if behavior_kl_backtrack and optimizer_steps_applied:
                if actor_snapshot is None:
                    raise RuntimeError(
                        "PPO behavior-KL backtracking lost its pre-step actor snapshot"
                    )
                if optimizer_learning_rates_before_scheduler is None:
                    raise RuntimeError(
                        "PPO behavior-KL backtracking has no audited actor LR"
                    )
                post_update_metrics, backtrack_metrics = self._backtrack_behavior_kl(
                    samples,
                    post_update_metrics,
                    actor_snapshot,
                    actor_learning_rate=(optimizer_learning_rates_before_scheduler[0]),
                )
                self._last_behavior_kl_backtrack_metrics = {
                    key: value
                    for key, value in backtrack_metrics.items()
                    if key != "train/actor_backtrack_history"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                if optimizer_metrics is not None:
                    optimizer_metrics.update(backtrack_metrics)
                if int(backtrack_metrics["train/actor_backtrack_failed"]):
                    raise FloatingPointError(
                        "PPO actor behavior-KL backtracking produced a non-finite or "
                        "catastrophic candidate; restored the pre-step actor and "
                        "refusing scheduler, checkpoint, or weight sync"
                    )
            self._validate_update_diagnostics(
                post_update_metrics,
                phase="post_update",
            )
        except Exception as guard_failure:
            rejection: BaseException = guard_failure
            if behavior_kl_backtrack and optimizer_steps_applied:
                restore_metadata: dict[str, float | int] = {
                    "train/actor_restore_attempted": 0,
                    "train/actor_restore_succeeded": 0,
                    # Adam moments and the critic update are process-local and
                    # intentionally not rewound.  Fail-stop below prevents
                    # scheduler, checkpoint, and weight-sync publication.
                    "train/actor_optimizer_state_rolled_back": 0,
                }
                if post_update_metrics is not None:
                    candidate_kl = float(
                        post_update_metrics.get(
                            "train/post_update_approx_kl", float("nan")
                        )
                    )
                    target_behavior_kl = getattr(self, "_target_behavior_kl", None)
                    if (
                        math.isfinite(candidate_kl)
                        and target_behavior_kl is not None
                        and candidate_kl > float(target_behavior_kl)
                    ):
                        restore_metadata["train/behavior_kl_last_unsafe_candidate"] = (
                            candidate_kl
                        )
                if actor_snapshot is None:
                    restore_failure: BaseException = RuntimeError(
                        "PPO post-update rejection cannot restore the actor because "
                        "its immutable pre-step snapshot is missing"
                    )
                    rejection = ExceptionGroup(
                        "PPO post-update rejection and actor restoration failed",
                        [guard_failure, restore_failure],
                    )
                    post_update_metrics = None
                else:
                    parameters, snapshots = actor_snapshot
                    restore_metadata["train/actor_restore_attempted"] = 1
                    try:
                        self._scale_actor_step_toward_snapshot_(
                            parameters,
                            snapshots,
                            relative_scale=0.0,
                        )
                        restore_metadata["train/actor_restore_succeeded"] = 1
                    except Exception as restore_failure:
                        rejection = ExceptionGroup(
                            "PPO post-update rejection and actor restoration failed",
                            [guard_failure, restore_failure],
                        )
                        post_update_metrics = None
                    else:
                        try:
                            restored_metrics = self._post_update_diagnostics(samples)
                        except Exception as restored_diagnostic_failure:
                            rejection = ExceptionGroup(
                                "PPO post-update rejection and restored-state "
                                "diagnostic failed",
                                [guard_failure, restored_diagnostic_failure],
                            )
                            post_update_metrics = None
                        else:
                            post_update_metrics = restored_metrics
                            restored_kl = float(
                                restored_metrics.get(
                                    "train/post_update_approx_kl", float("nan")
                                )
                            )
                            if math.isfinite(restored_kl):
                                restore_metadata["train/behavior_kl_after_restore"] = (
                                    restored_kl
                                )
                if optimizer_metrics is not None:
                    optimizer_metrics.update(restore_metadata)
            if write_ppo_receipt:
                assert receipt_run_config is not None
                assert receipt_rank is not None
                try:
                    receipt_path = _write_ppo_update_diagnostic_receipt(
                        config=self.config,
                        run_config=receipt_run_config,
                        rank=receipt_rank,
                        current_step=current_step,
                        total_steps=total_steps,
                        state="post_rejected",
                        received_rollouts=received_rollout_count,
                        trainable_rollouts=len(rollouts),
                        sample_rows=len(samples),
                        actor_sample_rows=actor_sample_rows,
                        behavior_weight_versions=behavior_weight_versions,
                        consumed_rollout_artifacts=consumed_rollout_artifacts,
                        consumed_rollout_batch_sha256=(consumed_rollout_batch_sha256),
                        is_master_replica=is_master_replica,
                        do_save_checkpoint=do_save_checkpoint,
                        pre_update_metrics=pre_update_metrics,
                        optimizer_metrics=optimizer_metrics,
                        post_update_metrics=post_update_metrics,
                        rejection=rejection,
                    )
                    logger.error(
                        "PPO post-update guard rejection receipt: %s", receipt_path
                    )
                except Exception as receipt_failure:
                    raise ExceptionGroup(
                        "PPO post-update guard and diagnostic receipt both failed",
                        [rejection, receipt_failure],
                    ) from None
            if rejection is guard_failure:
                raise
            raise rejection from None
        finally:
            # Release the roughly 2 GB VLA action-head CPU snapshot before a
            # checkpoint starts serializing model and optimizer state.
            actor_snapshot = None
        if post_update_metrics is None:
            raise RuntimeError(
                "PPO accepted-update path has no final policy diagnostics"
            )

        lr_scheduler = self.lr_schedulers
        if lr_scheduler is None:
            raise RuntimeError("Cosmos trainer did not initialize its LR scheduler")
        if write_ppo_receipt:
            assert receipt_run_config is not None
            assert receipt_rank is not None
            receipt_path = _write_ppo_update_diagnostic_receipt(
                config=self.config,
                run_config=receipt_run_config,
                rank=receipt_rank,
                current_step=current_step,
                total_steps=total_steps,
                state="accepted",
                received_rollouts=received_rollout_count,
                trainable_rollouts=len(rollouts),
                sample_rows=len(samples),
                actor_sample_rows=actor_sample_rows,
                behavior_weight_versions=behavior_weight_versions,
                consumed_rollout_artifacts=consumed_rollout_artifacts,
                consumed_rollout_batch_sha256=consumed_rollout_batch_sha256,
                is_master_replica=is_master_replica,
                do_save_checkpoint=do_save_checkpoint,
                pre_update_metrics=pre_update_metrics,
                optimizer_metrics=optimizer_metrics,
                post_update_metrics=post_update_metrics,
                rejection=None,
            )
            logger.info(
                "PPO accepted-update diagnostic receipt before scheduler/checkpoint: %s",
                receipt_path,
            )
        if optimizer_steps_applied:
            lr_scheduler.step()
        else:
            logger.warning(
                "AlpaGym trainer applied no optimizer step at current_step=%d; "
                "leaving the LR scheduler unchanged",
                current_step,
            )
        checkpoint_config = getattr(self.config.train, "ckpt", None)
        checkpoint_enabled = bool(
            getattr(checkpoint_config, "enable_checkpoint", False)
        )
        final_checkpoint_fallback = checkpoint_enabled and current_step == total_steps
        if is_master_replica and (do_save_checkpoint or final_checkpoint_fallback):
            if final_checkpoint_fallback and not do_save_checkpoint:
                # Some Cosmos colocated controller paths send the final real
                # DataFetchCommand with do_save=False and then stop the policy
                # worker before its synthetic TrainingCompleteCommand can run.
                # The trainer owns the last successfully applied optimizer
                # state, so persist it here when checkpointing is enabled.
                logger.info(
                    "Cosmos did not request a checkpoint on final trainer step %d; "
                    "applying AlpaGym final-checkpoint fallback",
                    current_step,
                )
            self._save_checkpoint(current_step, total_steps, remain_samples_num)

        avg_loss = total_loss / num_batches if num_batches else 0.0
        avg_kl = total_kl / num_batches if num_batches else 0.0
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            global_avg_loss = float(
                dist_util.dist_mean(loss_tensor, self.parallel_dims.mesh["dp_cp"])
            )
            global_max_loss = float(
                dist_util.dist_max(loss_tensor, self.parallel_dims.mesh["dp_cp"])
            )
        else:
            global_avg_loss = global_max_loss = avg_loss

        metrics = {
            "train_step": current_step,
            "train/loss_avg": global_avg_loss,
            "train/loss_max": global_max_loss,
            "train/kl_avg": avg_kl,
            "train/learning_rate": float(lr_scheduler.get_last_lr()[0]),
            "train/num_batches": num_batches,
            "train/num_micro_batches": int(
                getattr(self, "_last_micro_batches", num_batches)
            ),
            "train/optimizer_steps_applied": optimizer_steps_applied,
            "train/ratio_max": ratio_max,
            "train/ratio_min": ratio_min,
            "train/clip_fraction": clip_fraction_sum / num_batches
            if num_batches
            else 0.0,
            "train/grad_norm": grad_norm_sum / num_batches if num_batches else 0.0,
            "train/iteration_time": 0.0,
            **pre_update_metrics,
            **post_update_metrics,
            **getattr(self, "_last_ppo_advantage_metrics", {}),
            **getattr(self, "_last_behavior_kl_backtrack_metrics", {}),
        }
        logger.info(
            "AlpaGym trainer step end current_step=%d steps=%d batches=%d "
            "loss_avg=%.6f kl_avg=%.6f ratio_min=%.6f ratio_max=%.6f "
            "clip_fraction=%.6f grad_norm=%.6f lr=%.8g",
            current_step,
            len(samples),
            num_batches,
            float(metrics["train/loss_avg"]),
            avg_kl,
            ratio_min,
            ratio_max,
            float(metrics["train/clip_fraction"]),
            float(metrics["train/grad_norm"]),
            float(metrics["train/learning_rate"]),
        )
        return metrics

    def _pre_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Return optional metrics before any optimizer update is applied."""
        del samples
        return {}

    def _post_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Return optional metrics evaluated after all optimizer updates.

        Generic GRPO policies retain their existing metric path. Actor-critic
        PPO overrides this hook because its replay includes the exact behavior
        actions and masks needed to evaluate final-policy drift.
        """
        del samples
        return {}

    def _validate_update_diagnostics(
        self,
        metrics: dict[str, float | int],
        *,
        phase: str,
    ) -> None:
        """Optionally reject an update from pre/post replay diagnostics."""
        del metrics, phase

    # ------------------------------------------------------------------
    # GRPO orchestration
    # ------------------------------------------------------------------

    def _prepare_training_data(
        self,
        rollouts: list[_rollout_schema.Rollout],
    ) -> tuple[list[Any], torch.Tensor]:
        """Flatten rollouts into a single pool of per-step samples plus advantages.

        Each rollout's artifact unpacks into a fixed-length list of single-step
        replay samples (valid steps plus ``is_padding`` rows); the trainer
        extends one flat pool across all rollouts and records the matching
        per-step advantage. Cosmos supplies one advantage per rollout, replayed
        across that rollout's valid steps; padding steps carry zero so they
        contribute no policy gradient.

        Args:
            rollouts: Cosmos-RL rollouts surviving the staleness filter.

        Returns:
            Tuple ``(samples, advantages)`` aligned row-for-row: ``samples`` is
            the flat per-step pool, ``advantages`` the per-step advantage.
        """
        samples: list[Any] = []
        advantages: list[float] = []
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        for rollout in rollouts:
            step_samples = data_packer.get_policy_input(
                rollout.prompt,
                rollout.completion,
                n_ignore_prefix_tokens=rollout.n_ignore_prefix_tokens,
            )
            for step in step_samples:
                is_padding = bool(step.training_signal.is_padding.item())
                advantages.append(0.0 if is_padding else float(rollout.advantage))
            samples.extend(step_samples)
        return samples, torch.tensor(advantages, dtype=torch.float32)

    def _run_training_loop(
        self,
        samples: list[Any],
        advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, int, float, float, float, float]:
        """Run ``grpo_optimization_iterations × num_mini_batches`` PPO updates.

        Minibatching is at the step level: ``samples`` is the flattened pool of
        per-step replay samples across all rollouts, shuffled fresh each
        optimization iteration and split into minibatches. The collate step
        stacks one minibatch of single-step samples into the ``[B, ...]`` inputs
        the model forward sees.

        Returns:
            Aggregates: ``(total_loss, total_kl, num_batches, ratio_max,
            ratio_min, clip_fraction_sum, grad_norm_sum)``.
        """
        self._ensure_reference_model()
        self._optimizer_steps_applied_in_training_step = 0

        num_steps = len(samples)
        mini_batch_size = min(self._mini_batch, num_steps)
        num_mini_batches = (num_steps + mini_batch_size - 1) // mini_batch_size

        total_loss = 0.0
        total_kl = 0.0
        num_batches = 0
        ratio_max = float("-inf")
        ratio_min = float("inf")
        clip_fraction_sum = 0.0
        grad_norm_sum = 0.0

        for _ in range(self._grpo_optimization_iterations):
            indices = torch.randperm(num_steps)
            for minibatch_index in range(num_mini_batches):
                start = minibatch_index * mini_batch_size
                end = min(start + mini_batch_size, num_steps)
                minibatch_indices = indices[start:end]
                minibatch_samples = [samples[int(index)] for index in minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]

                (
                    loss_value,
                    kl_value,
                    batch_ratio_max,
                    batch_ratio_min,
                    batch_clip_fraction,
                    batch_grad_norm,
                ) = self._train_minibatch(
                    minibatch_samples,
                    minibatch_advantages,
                    inter_policy_nccl,
                )
                total_loss += loss_value
                total_kl += kl_value
                num_batches += 1
                ratio_max = max(ratio_max, batch_ratio_max)
                ratio_min = min(ratio_min, batch_ratio_min)
                clip_fraction_sum += batch_clip_fraction
                grad_norm_sum += batch_grad_norm

        if num_batches == 0:
            ratio_max = 0.0
            ratio_min = 0.0
        return (
            total_loss,
            total_kl,
            num_batches,
            ratio_max,
            ratio_min,
            clip_fraction_sum,
            grad_norm_sum,
        )

    def _train_minibatch(
        self,
        minibatch_samples: list[Any],
        minibatch_advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, float, float, float, float]:
        """Train on a single step-level minibatch and apply gradient.

        Orchestrates: collate the per-step samples, forward the full minibatch,
        compute PPO surrogate + KL penalty, backward + optimizer step, emit
        metrics. Padding rows are forwarded like any other (so every DP worker
        runs the identical model forward in lockstep) but are neutralized: their
        advantage is zero (no policy-loss gradient) and they are masked out of
        the KL term and the diagnostics.

        Args:
            minibatch_samples: one single-step replay sample per row.
            minibatch_advantages: ``[B]`` per-step advantages (zero on padding
                rows), aligned row-for-row with ``minibatch_samples``.
            inter_policy_nccl: NCCL communicator across DP replicas.

        Returns:
            Tuple ``(loss, kl_value, ratio_max, ratio_min, clip_fraction)``.
        """
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        minibatch = data_packer.policy_collate_fn(minibatch_samples)
        is_padding = minibatch.training_signal.is_padding.to(self.device)
        old_logprobs = minibatch.training_signal.old_logprobs.to(self.device)
        advantages = minibatch_advantages.to(device=self.device, dtype=torch.float32)
        new_logprobs, kl_div = self._forward_with_reference(minibatch.model_inputs)
        assert_replay_shapes(new_logprobs, old_logprobs, advantages, kl_div)
        policy_loss, ratio = compute_ppo_surrogate(
            new_logprobs,
            old_logprobs,
            advantages,
            ratio_clip_low=self._grpo_ratio_clip_low,
            ratio_clip_high=self._grpo_ratio_clip_high,
            is_padding=is_padding,
        )
        kl_loss = compute_kl_penalty(
            kl_div,
            is_padding,
            kl_beta=self._kl_beta,
            device=self.device,
        )
        loss = policy_loss + kl_loss

        self.optimizers.zero_grad()
        loss.backward()
        grad_norm = self.all_reduce_states(inter_policy_nccl)

        return self._minibatch_metrics(
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            ratio=ratio,
            is_padding=is_padding,
            advantages=advantages,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            grad_norm=grad_norm,
        )

    def _forward_with_reference(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model forward with the reference model attached for KL."""
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        return result["log_probs"], result.get("kl_div")

    def _minibatch_metrics(
        self,
        policy_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        ratio: torch.Tensor,
        is_padding: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        grad_norm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute ratio/clip diagnostics over the valid (non-padding) rows.

        ``_train_minibatch`` forwards the full minibatch, so the ratio/clip
        diagnostics mask out padding rows here; an all-padding minibatch has no
        valid rows and reports neutral defaults.
        """
        valid_mask = ~is_padding
        loss_value = float((policy_loss + kl_loss).item())
        with torch.no_grad():
            valid_ratio = ratio[valid_mask]
            if valid_ratio.numel() == 0:
                clip_fraction = 0.0
                batch_ratio_max = 1.0
                batch_ratio_min = 1.0
                advantage_mean = 0.0
            else:
                clipped = (valid_ratio < 1.0 - self._grpo_ratio_clip_low) | (
                    valid_ratio > 1.0 + self._grpo_ratio_clip_high
                )
                clip_fraction = float(clipped.float().mean().item())
                batch_ratio_max = float(valid_ratio.max().item())
                batch_ratio_min = float(valid_ratio.min().item())
                advantage_mean = float(advantages[valid_mask].mean().item())
        logger.info(
            "AlpaGym trainer minibatch rows=%d valid_rows=%d loss=%.6f "
            "policy_loss=%.6f kl_loss=%.6f ratio_min=%.6f ratio_max=%.6f "
            "clip_fraction=%.6f advantage_mean=%.6f old_logprob_mean=%.6f "
            "new_logprob_mean=%.6f grad_norm=%.6f",
            int(old_logprobs.numel()),
            int(valid_mask.sum().item()),
            loss_value,
            float(policy_loss.item()),
            float(kl_loss.item()),
            batch_ratio_min,
            batch_ratio_max,
            clip_fraction,
            advantage_mean,
            float(old_logprobs.mean().item()),
            float(new_logprobs.mean().item()),
            grad_norm,
        )
        return (
            loss_value,
            float(kl_loss.item()),
            batch_ratio_max,
            batch_ratio_min,
            clip_fraction,
            grad_norm,
        )

    def all_reduce_states(
        self, inter_policy_nccl: dist_util.HighAvailabilitylNccl
    ) -> float:
        """Reduce gradients across DP replicas, clip norm, and step optimizer.

        Override of `GRPOTrainer.all_reduce_states`. Three differences:
        - Iterates `self.model.parameters()` directly rather than
          `self.model_parts`. Some policy wrappers keep trainable parameters
          under nested child modules, so `model_parts` may not carry the right
          param refs.
        - Captures `current_stream` BEFORE entering the `train_stream`
          context so the all-reduce on `train_stream` waits for FSDP's
          reduce-scatter on the default stream.
        - Raises on a non-finite reduced gradient and skips the step on a zero
          one. Both checks run after the cross-replica reduce, so every DP
          worker reaches the same verdict in lockstep.
        """
        train_stream = self.train_stream
        if train_stream is None:
            raise RuntimeError("Cosmos trainer did not initialize its CUDA stream")
        backward_stream = torch.cuda.current_stream()
        with torch.cuda.stream(train_stream):
            train_stream.wait_stream(backward_stream)
            params = [param for param in self.model.parameters() if param.requires_grad]
            if params:
                dist_util.gradient_reduce_across_dp_replicas_(params, inter_policy_nccl)
            grads = [param.grad for param in params if param.grad is not None]
            if grads and not torch.stack(torch._foreach_norm(grads)).sum().isfinite():
                raise FloatingPointError(
                    "[GRPO:grad-guard] non-finite reduced gradient after backward"
                )
            grad_norm = dist_util.gradient_norm_clipping(
                params,
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=(
                    self.parallel_dims.mesh["pp"]
                    if self.parallel_dims.pp_enabled
                    else None
                ),
                return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
            )
            grad_norm_value = float(grad_norm) if grad_norm is not None else 0.0
            # Skip the optimizer step on a zero gradient (all-padding or
            # zero-advantage minibatch) so weight decay / Adam state do not
            # advance on no signal.
            if grad_norm_value != 0.0:
                self.optimizers.step()
                self._optimizer_steps_applied_in_training_step += 1
            self.optimizers.zero_grad()
        return grad_norm_value

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def weight_resume(self) -> dict[str, Any]:
        """Populate weights through base load or one fail-closed native restore.

        Upstream Cosmos catches every native checkpoint error and then falls
        back to Hugging Face. That behavior is unsafe for formal continuation:
        a run can claim to continue step N while actually training base weights.
        This override retains Cosmos's native loader but never catches a restore
        failure. The restore receipt is written only after the exact tree is
        revalidated and model/optimizer/scheduler/RNG/step restoration succeeds.
        """

        run_config = _load_run_config(self.config)
        source = _validated_runtime_resume_source(
            config=self.config,
            run_config=run_config,
        )
        kl_enabled = float(self.config.train.train_policy.kl_beta) != 0.0
        if kl_enabled:
            # Preserve Cosmos's restart-stable fixed reference: capture the
            # configured base policy before native resume replaces live weights.
            self.model_load_from_hf()
            self.reference_state_dict = {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            }

        if source is None:
            if not kl_enabled:
                self.model_load_from_hf()
            if self.map_w_from_policy_to_rollout is None:
                raise RuntimeError("no policy-to-rollout parameter mapping exists")
            self.set_model_train()
            return {}

        contract = run_config.cosmos.train.resume
        assert contract.checkpoint_step is not None
        assert contract.expected_next_training_step is not None
        rank = getattr(self.ckpt_manager, "global_rank", None)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("Cosmos checkpoint manager has no valid global rank")
        expected_extra_info = _load_checkpoint_extra_info(
            source,
            rank=rank,
            expected_step=contract.checkpoint_step,
            expected_next_step=contract.expected_next_training_step,
        )

        logger.info(
            "[Policy] Fail-closed native resume from %s at step %d",
            source.checkpoint_path,
            contract.checkpoint_step,
        )
        # Deliberately do not catch: both Cosmos's translated FileNotFoundError
        # and any direct loader failure abort this run instead of loading HF/base.
        restored = self.model_resume_from_checkpoint()
        if self.lr_schedulers is None:
            raise RuntimeError("Cosmos native resume did not restore a scheduler")
        rng_reader = getattr(self.ckpt_manager, "get_rng_state", None)
        if not callable(rng_reader):
            raise RuntimeError("Cosmos checkpoint manager cannot verify restored RNG")
        _validate_native_resume_result(
            extra_info=expected_extra_info,
            restored=restored,
            current_rng_state=rng_reader(),
        )
        # Rehash after native torch.load calls so a concurrent or accidental
        # checkpoint mutation cannot produce a successful receipt.
        source_after = validate_checkpoint_resume_source(contract)
        if source_after.snapshot != source.snapshot:
            raise RuntimeError("checkpoint tree changed during native restore")
        if self.map_w_from_policy_to_rollout is None:
            raise RuntimeError("no policy-to-rollout parameter mapping exists")
        self.set_model_train()
        receipt_path = _write_checkpoint_restore_receipt(
            config=self.config,
            run_config=run_config,
            source=source_after,
            rank=rank,
            restored=restored,
        )
        logger.info("[Policy] Native checkpoint restore receipt: %s", receipt_path)
        return restored

    def save_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        *,
        is_final: bool,
    ) -> None:
        """Handle Cosmos's public synthetic-EOS checkpoint entry point.

        Cosmos's policy worker calls this public method for
        ``TrainingCompleteCommand``.  The inherited GRPO implementation uses a
        generic Hugging Face exporter and therefore bypasses policy-native
        candidate invariants.  Route that call through the same exporter as a
        normal AlpaGym trainer step.  Public non-terminal calls are not part of
        the AlpaGym scheduling contract and fail closed.
        """
        if not is_final:
            raise ValueError(
                "AlpaGym public save_checkpoint is reserved for synthetic "
                "terminal completion"
            )
        if (
            isinstance(current_step, bool)
            or not isinstance(current_step, int)
            or current_step < 1
            or isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps < current_step
        ):
            raise ValueError(
                "AlpaGym synthetic terminal checkpoint coordinates are invalid"
            )
        self._save_checkpoint(
            current_step,
            total_steps,
            remain_samples_num,
            allow_completed_policy_native_export=True,
        )

    def _save_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        *,
        allow_completed_policy_native_export: bool = False,
    ) -> None:
        """Save policy weights at ``current_step``.

        First writes and completes the Cosmos resume checkpoint (model +
        optimizer + scheduler + ``remain_samples_num``), then exports deployable
        weights when ``config.train.ckpt.export_safetensors`` is set.  This order
        prevents a failed resume save from leaving a candidate that claims a
        durable training step, while an export failure still leaves a resumable
        optimizer state.

        The inherited ``ckpt_manager`` and ``export_safetensors`` come from
        ``LLMTrainer``; ``output_dir`` / ``ckpt`` / ``param_dtype`` come from
        ``cosmos_config.toml``.
        """
        logical_total_steps = int(
            getattr(self, "_logical_total_training_steps", total_steps)
        )
        global_train_batch_size = getattr(self, "_global_train_batch_size", None)
        if not current_step <= total_steps <= logical_total_steps:
            raise ValueError(
                "checkpoint process-stage coordinates exceed the logical horizon"
            )
        if remain_samples_num < 0:
            raise ValueError("checkpoint remaining samples must be nonnegative")
        if global_train_batch_size is not None:
            expected_remaining_samples = (
                logical_total_steps - current_step
            ) * global_train_batch_size
            if remain_samples_num != expected_remaining_samples:
                raise ValueError(
                    "checkpoint remaining samples disagree with the logical horizon"
                )
        is_last_step = current_step == logical_total_steps and remain_samples_num == 0
        export_hook = getattr(
            getattr(self, "_policy_bundle", None),
            "export_model_checkpoint",
            None,
        )
        completed_exports = getattr(
            self,
            "_completed_policy_native_exports",
            None,
        )
        if completed_exports is None:
            completed_exports = {}
            self._completed_policy_native_exports = completed_exports
        prior_export = completed_exports.get(current_step)
        if prior_export is not None:
            if not allow_completed_policy_native_export:
                raise RuntimeError(
                    "policy-native checkpoint step was already exported by this "
                    "trainer instance"
                )
            if not self.config.train.ckpt.export_safetensors or export_hook is None:
                raise RuntimeError(
                    "completed policy-native export no longer matches checkpoint config"
                )
            expected_path = (
                Path(self.config.train.output_dir)
                / "safetensors"
                / f"step_{current_step}"
            )
            expected_context = _policy_checkpoint_export_context(
                config=self.config,
                current_step=current_step,
                total_steps=logical_total_steps,
                optimizer_steps_applied=int(
                    getattr(
                        self,
                        "_optimizer_steps_applied_in_training_step",
                        0,
                    )
                ),
            )
            if prior_export != (expected_path, expected_context):
                raise RuntimeError(
                    "synthetic terminal checkpoint differs from the completed "
                    "policy-native export"
                )
            if not expected_path.is_dir() or expected_path.is_symlink():
                raise RuntimeError(
                    "completed policy-native candidate disappeared before synthetic EOS"
                )
        scheduler = self.lr_schedulers
        if scheduler is None:
            raise RuntimeError("Cosmos trainer did not initialize its LR scheduler")
        logger.info("[Policy] Saving cosmos checkpoint at step %d", current_step)
        self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=scheduler,
            step=current_step,
            total_steps=logical_total_steps,
            remain_samples_num=remain_samples_num,
            is_final=is_last_step,
        )
        self.ckpt_manager.save_check(step=current_step)

        if self.config.train.ckpt.export_safetensors:
            export_rel_path = os.path.join("safetensors", f"step_{current_step}")
            if export_hook is not None:
                export_path = Path(self.config.train.output_dir) / export_rel_path
                if prior_export is not None:
                    logger.info(
                        "[Policy] Preserving immutable policy-native checkpoint at "
                        "step %d during synthetic terminal save",
                        current_step,
                    )
                    return
                export_context = _policy_checkpoint_export_context(
                    config=self.config,
                    current_step=current_step,
                    total_steps=logical_total_steps,
                    optimizer_steps_applied=int(
                        getattr(
                            self,
                            "_optimizer_steps_applied_in_training_step",
                            0,
                        )
                    ),
                )
                logger.info(
                    "[Policy] Saving policy-native checkpoint at step %d to %s",
                    current_step,
                    export_path,
                )
                export_hook(self.model, export_path, export_context)
                completed_exports[current_step] = (export_path, export_context)
            else:
                logger.info(
                    "[Policy] Saving huggingface checkpoint at step %d to %s",
                    current_step,
                    self.config.train.output_dir,
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=export_rel_path,
                    trainable_only=False,
                    is_final=is_last_step,
                    # cosmos's `param_dtype` is one of "bfloat16" / "float16" /
                    # "float32"; all map to `torch.<name>` directly.
                    dtype=getattr(torch, str(self.config.train.param_dtype).lower()),
                )

    # ------------------------------------------------------------------
    # Reference model lifecycle
    # ------------------------------------------------------------------

    def _ensure_reference_model(self) -> None:
        """Create the frozen initial-policy reference on first use.

        Stores the reference on a private attribute (not registered as a
        submodule, since `Trainer` is an ABC, not an `nn.Module`). AlpaGym's
        ``weight_resume`` records the configured initial policy before a
        Cosmos checkpoint can restore different live weights. Building the
        teacher from that state keeps both KL and actor-mean anchoring fixed
        across process restarts. We drive the reference via ``teacher_model=``
        on the model forward rather than Cosmos's state-dict swapping path.
        """
        if self._kl_beta <= 0.0 or self._reference_model is not None:
            return
        reference_state_dict = getattr(self, "reference_state_dict", None)
        if not reference_state_dict:
            raise RuntimeError(
                "KL reference weights are unavailable; Cosmos weight_resume() "
                "must run before the first AlpaGym training step"
            )
        self._reference_model = copy.deepcopy(self.model)
        self._reference_model.load_state_dict(reference_state_dict, strict=True)
        self._reference_model.eval()
        for param in self._reference_model.parameters():
            param.requires_grad_(False)


def _custom_config_section(config: _cosmos_config.Config, key: str) -> dict[str, Any]:
    """Return one mapping from Cosmos ``config.custom`` without assuming its concrete type."""
    custom = getattr(config, "custom", None) or {}
    section = custom.get(key, {}) if hasattr(custom, "get") else {}
    return dict(section) if isinstance(section, dict) else {}


def _require_ppo_signal(tensor: torch.Tensor | None, field_name: str) -> torch.Tensor:
    """Return a required PPO training-signal tensor or raise a targeted error."""
    if tensor is None:
        raise ValueError(f"PPO trainer requires TrainingSignal.{field_name}")
    return tensor


def _ppo_actor_valid_mask(
    signal: TrainingSignal,
    *,
    required: bool = False,
) -> torch.Tensor:
    """Return rows whose sampled policy action reached the controller."""
    if signal.actor_valid is None:
        if required:
            raise ValueError(
                "Flow-PPO replay requires TrainingSignal.actor_valid so "
                "unexecuted sampled references cannot enter the actor loss"
            )
        return torch.ones_like(signal.is_padding, dtype=torch.bool)
    return signal.actor_valid


def _time_scaled_discount(
    nominal_discount: float,
    *,
    duration_ticks: int,
    nominal_duration_ticks: int,
) -> float:
    """Convert a policy-clock discount to one realized tick interval.

    Flow-PPO configures gamma and lambda on the nominal 2 Hz policy clock,
    while motion-reference feedback is recorded on the 50 Hz controller
    clock.  Raising the nominal factor by ``duration / nominal_duration``
    preserves the same physical-time horizon for ordinary, delayed, and empty
    intervals.  Duration zero is useful at boundaries and has unit discount;
    replay transitions themselves still require at least one realized tick.
    """
    if not math.isfinite(nominal_discount) or not 0.0 <= nominal_discount <= 1.0:
        raise ValueError("nominal discount must be finite and within [0, 1]")
    if duration_ticks < 0:
        raise ValueError("duration_ticks cannot be negative")
    if nominal_duration_ticks <= 0:
        raise ValueError("nominal_duration_ticks must be positive")
    if duration_ticks == 0:
        return 1.0
    return nominal_discount ** (duration_ticks / nominal_duration_ticks)


def _discounted_transition_reward(
    signal: TrainingSignal,
    *,
    gamma_tick: float,
    reward_values: torch.Tensor | None = None,
    credit_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Return one macro reward and its controller-tick duration.

    Direct policies carry one scalar reward and therefore have duration one.
    Motion-reference policies carry the exact committed controller-tick reward
    prefix. Keeping the primitive rewards in replay lets the trainer change
    neither their time order nor the semi-Markov discount by accident.  An
    optional actor reward view removes predecessor-owned certification gain,
    while a credit mask excludes pre-install ticks. Neither operation rebases
    later rewards to tick zero, so inference latency remains in the discount.
    """
    if signal.primitive_rewards is None:
        if credit_mask is not None or reward_values is not None:
            raise ValueError("PPO actor reward views require primitive rewards")
        reward = _require_ppo_signal(signal.rewards, "rewards")
        return float(reward.item()), 1

    primitive_rewards = (
        signal.primitive_rewards if reward_values is None else reward_values
    ).reshape(-1)
    if primitive_rewards.shape != signal.primitive_rewards.reshape(-1).shape:
        raise ValueError("PPO reward view shape must match primitive rewards")
    primitive_mask = _require_ppo_signal(
        signal.primitive_reward_mask,
        "primitive_reward_mask",
    ).reshape(-1)
    duration_tensor = _require_ppo_signal(signal.duration_ticks, "duration_ticks")
    duration = int(duration_tensor.item())
    if duration <= 0 or duration > primitive_rewards.numel():
        raise ValueError(
            "PPO duration_ticks must select a non-empty primitive reward prefix"
        )
    expected_mask = torch.arange(primitive_rewards.numel()) < duration
    if not torch.equal(primitive_mask.cpu(), expected_mask):
        raise ValueError(
            "PPO primitive_reward_mask must be a contiguous prefix matching duration_ticks"
        )
    selected = primitive_rewards[:duration]
    if not torch.isfinite(selected).all():
        raise ValueError("PPO primitive rewards must be finite")
    if credit_mask is None:
        selected_credit = torch.ones(duration, dtype=torch.bool)
    else:
        credit_mask = credit_mask.reshape(-1)
        if credit_mask.shape != primitive_mask.shape:
            raise ValueError("PPO credit mask shape must match primitive rewards")
        if credit_mask.dtype != torch.bool:
            raise ValueError("PPO credit mask dtype must be bool")
        credit_mask_cpu = credit_mask.cpu()
        if torch.any(credit_mask_cpu & ~primitive_mask.cpu()):
            raise ValueError("PPO credit mask cannot select unrealized reward ticks")
        selected_credit = credit_mask_cpu[:duration]
    powers = torch.pow(
        torch.tensor(float(gamma_tick), dtype=torch.float64),
        torch.arange(duration, dtype=torch.float64),
    )
    reward = torch.sum(
        selected.to(dtype=torch.float64).cpu()
        * powers
        * selected_credit.to(dtype=torch.float64)
    )
    return float(reward.item()), duration


def _with_ppo_targets(sample: Any, *, advantage: float, ret: float) -> Any:
    """Attach trainer-computed PPO targets to one replay sample."""
    signal = sample.training_signal
    return replace(
        sample,
        training_signal=replace(
            signal,
            advantages=torch.tensor([advantage], dtype=torch.float32),
            returns=torch.tensor([ret], dtype=torch.float32),
        ),
    )


def _replace_advantages(
    samples: list[Any],
    advantages: torch.Tensor,
    per_rollout_ranges: list[tuple[int, int]],
) -> list[Any]:
    """Mirror normalized advantages into each sample's internal training signal."""
    del per_rollout_ranges
    return [
        _with_ppo_targets(
            sample,
            advantage=float(advantages[index].item()),
            ret=float(
                _require_ppo_signal(sample.training_signal.returns, "returns").item()
            ),
        )
        for index, sample in enumerate(samples)
    ]


def _ppo_advantage_metrics(
    advantages: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Summarize the exact actor advantages consumed by Flow-PPO.

    Cosmos also reports rollout-level GRPO advantages.  Those values are zero
    when ``n_generation=1`` and are intentionally ignored by PPO, so exposing
    the trainer-derived GAE here prevents that transport statistic from being
    mistaken for the policy-gradient signal.
    """
    values = advantages.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
        }
    if not torch.isfinite(values).all():
        raise FloatingPointError("PPO actor advantages contain non-finite values")
    return {
        f"{prefix}_rows": int(values.numel()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
    }


def _summarize_behavior_log_ratios(
    log_ratios: torch.Tensor,
    *,
    phase: str,
    ratio_clip_low: float,
    ratio_clip_high: float,
) -> dict[str, float | int]:
    """Summarize one flat, already-masked behavior-ratio pool."""
    if phase not in {"pre_update", "post_update"}:
        raise ValueError(f"Unsupported PPO behavior-diagnostic phase: {phase!r}")
    prefix = f"train/{phase}"
    flattened = log_ratios.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if flattened.numel() == 0:
        return {
            f"{prefix}_valid_rows": 0,
            f"{prefix}_ratio_p01": 1.0,
            f"{prefix}_ratio_p50": 1.0,
            f"{prefix}_ratio_p99": 1.0,
            f"{prefix}_clip_fraction": 0.0,
            f"{prefix}_approx_kl": 0.0,
            f"{prefix}_max_abs_log_ratio": 0.0,
            f"{prefix}_max_abs_ratio_error": 0.0,
        }
    if not torch.isfinite(flattened).all():
        raise FloatingPointError(
            f"PPO {phase.replace('_', '-')} log-ratios contain non-finite values"
        )

    # Ratio diagnostics match the bounded exponent used by the PPO objective,
    # but approximate KL intentionally retains the raw log-ratio so the clamp
    # cannot hide policy divergence. Float64 keeps large finite joint Flow
    # deltas observable without overflowing at float32's exponent boundary.
    bounded_log_ratios = flattened.clamp(min=-5.0, max=5.0)
    ratios = bounded_log_ratios.exp()
    quantiles = torch.quantile(
        ratios,
        torch.tensor([0.01, 0.5, 0.99], dtype=ratios.dtype),
    )
    clipped = (ratios < 1.0 - ratio_clip_low) | (ratios > 1.0 + ratio_clip_high)
    raw_log_ratios = flattened.to(dtype=torch.float64)
    approx_kl = torch.expm1(raw_log_ratios) - raw_log_ratios
    if not torch.isfinite(approx_kl).all():
        raise FloatingPointError(
            f"PPO {phase.replace('_', '-')} approximate KL is non-finite "
            "for raw log-ratios"
        )
    ratio_errors = torch.expm1(raw_log_ratios).abs()
    if not torch.isfinite(ratio_errors).all():
        raise FloatingPointError(
            f"PPO {phase.replace('_', '-')} ratio errors are non-finite"
        )
    return {
        f"{prefix}_valid_rows": int(ratios.numel()),
        f"{prefix}_ratio_p01": float(quantiles[0].item()),
        f"{prefix}_ratio_p50": float(quantiles[1].item()),
        f"{prefix}_ratio_p99": float(quantiles[2].item()),
        f"{prefix}_clip_fraction": float(clipped.float().mean().item()),
        f"{prefix}_approx_kl": float(approx_kl.mean().item()),
        f"{prefix}_max_abs_log_ratio": float(raw_log_ratios.abs().max().item()),
        f"{prefix}_max_abs_ratio_error": float(ratio_errors.max().item()),
    }


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_ppo")
class AlpagymPPOTrainer(AlpagymGRPOTrainer):
    """Actor-critic PPO trainer over AlpaGym replay payloads.

    This trainer keeps the AlpaGym rollout transport exactly the same as GRPO:
    completed rollouts still arrive as ``EpisodeOutput`` artifacts and each
    ``PolicyOutput.replay_data`` still owns the per-step model replay payload.
    The difference is the training signal source: PPO computes per-step
    ``advantages`` and ``returns`` inside the trainer from replayed transition
    rewards, terminal flags, and rollout-time values, then trains a value head
    from model forward key ``values``.
    """

    _flow_chunk_density = False
    _write_ppo_update_diagnostic_receipts = True
    _dual_clip_ratio: float | None = None
    _value_huber_delta: float | None = None

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO-specific hyperparameters after shared Cosmos setup."""
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        ppo_config = _custom_config_section(config, "ppo")
        self._value_loss_coef = float(ppo_config.get("value_loss_coef", 0.5))
        if self._value_loss_coef < 0.0:
            raise ValueError(
                f"PPO value_loss_coef must be non-negative, got {self._value_loss_coef}"
            )
        value_clip_range = ppo_config.get("value_clip_range")
        self._value_clip_range = (
            None if value_clip_range is None else float(value_clip_range)
        )
        if self._value_clip_range is not None and self._value_clip_range <= 0.0:
            raise ValueError(
                f"PPO value_clip_range must be positive when set, got {self._value_clip_range}"
            )
        dual_clip_ratio = ppo_config.get("dual_clip_ratio")
        self._dual_clip_ratio = (
            None if dual_clip_ratio is None else float(dual_clip_ratio)
        )
        if self._dual_clip_ratio is not None and self._dual_clip_ratio <= 1.0:
            raise ValueError("PPO dual_clip_ratio must be greater than 1")
        value_huber_delta = ppo_config.get("value_huber_delta")
        self._value_huber_delta = (
            None if value_huber_delta is None else float(value_huber_delta)
        )
        if self._value_huber_delta is not None and self._value_huber_delta <= 0.0:
            raise ValueError("PPO value_huber_delta must be positive")
        self._normalize_advantages = bool(ppo_config.get("normalize_advantages", True))
        self._gamma = float(ppo_config.get("gamma", 0.99))
        self._gae_lambda = float(ppo_config.get("gae_lambda", 0.95))
        self._min_action_std = float(ppo_config.get("min_action_std", 0.02))
        self._max_action_std = float(ppo_config.get("max_action_std", 2.0))
        if not 0.0 < self._min_action_std <= self._max_action_std:
            raise ValueError(
                "PPO action std bounds must satisfy 0 < min_action_std <= "
                f"max_action_std; got {self._min_action_std}, {self._max_action_std}"
            )
        target_behavior_kl = ppo_config.get("target_behavior_kl")
        self._target_behavior_kl = (
            None if target_behavior_kl is None else float(target_behavior_kl)
        )
        if self._target_behavior_kl is not None and (
            not math.isfinite(self._target_behavior_kl)
            or self._target_behavior_kl <= 0.0
        ):
            raise ValueError(
                "PPO target_behavior_kl must be finite and positive when set, "
                f"got {target_behavior_kl!r}"
            )
        behavior_kl_target_mode = ppo_config.get("behavior_kl_target_mode", "hard")
        if behavior_kl_target_mode not in {"hard", "soft"}:
            raise ValueError(
                "PPO behavior_kl_target_mode must be either 'hard' or 'soft'"
            )
        self._behavior_kl_target_mode = behavior_kl_target_mode
        behavior_kl_hard_limit = ppo_config.get("behavior_kl_hard_limit")
        self._behavior_kl_hard_limit = (
            None if behavior_kl_hard_limit is None else float(behavior_kl_hard_limit)
        )
        if self._behavior_kl_target_mode == "soft":
            if self._target_behavior_kl is None or self._behavior_kl_hard_limit is None:
                raise ValueError(
                    "PPO soft behavior-KL target mode requires both "
                    "target_behavior_kl and behavior_kl_hard_limit"
                )
            if (
                not math.isfinite(self._behavior_kl_hard_limit)
                or self._behavior_kl_hard_limit <= self._target_behavior_kl
            ):
                raise ValueError(
                    "PPO behavior_kl_hard_limit must be finite and greater than "
                    "the soft target_behavior_kl"
                )
        elif self._behavior_kl_hard_limit is not None:
            raise ValueError(
                "PPO behavior_kl_hard_limit is only valid in soft target mode"
            )
        behavior_kl_backtrack = ppo_config.get("behavior_kl_backtrack", False)
        if type(behavior_kl_backtrack) is not bool:
            raise TypeError("PPO behavior-KL backtracking flag must be a boolean")
        self._behavior_kl_backtrack = behavior_kl_backtrack
        behavior_kl_backtrack_margin = ppo_config.get(
            "behavior_kl_backtrack_margin", 0.9
        )
        if isinstance(behavior_kl_backtrack_margin, bool):
            raise TypeError("PPO behavior-KL backtracking margin cannot be boolean")
        self._behavior_kl_backtrack_margin = float(behavior_kl_backtrack_margin)
        behavior_kl_backtrack_max_attempts = ppo_config.get(
            "behavior_kl_backtrack_max_attempts", 4
        )
        if type(behavior_kl_backtrack_max_attempts) is not int:
            raise TypeError(
                "PPO behavior-KL backtracking max attempts must be an integer"
            )
        self._behavior_kl_backtrack_max_attempts = behavior_kl_backtrack_max_attempts
        if self._behavior_kl_backtrack:
            if self._target_behavior_kl is None:
                raise ValueError(
                    "PPO behavior-KL backtracking requires target_behavior_kl"
                )
            if self._grpo_optimization_iterations != 1:
                raise ValueError(
                    "PPO behavior-KL backtracking requires exactly one optimizer "
                    "iteration so actor-delta interpolation remains equivalent to "
                    "lowering the actor learning rate"
                )
            if not 0.0 < self._behavior_kl_backtrack_margin < 1.0:
                raise ValueError(
                    "PPO behavior-KL backtracking margin must be within (0, 1)"
                )
            if self._behavior_kl_backtrack_max_attempts <= 0:
                raise ValueError(
                    "PPO behavior-KL backtracking max attempts must be positive"
                )
            if int(getattr(parallel_dims, "world_size", 1)) != 1:
                raise NotImplementedError(
                    "PPO behavior-KL backtracking currently requires one policy rank"
                )
        step_mini_batch = ppo_config.get("step_mini_batch", self._mini_batch)
        if (
            isinstance(step_mini_batch, bool)
            or not isinstance(step_mini_batch, int)
            or step_mini_batch <= 0
        ):
            raise ValueError(
                "PPO step_mini_batch must be a positive integer, "
                f"got {step_mini_batch!r}"
            )
        # Cosmos's train_policy.mini_batch remains rollout/shard geometry;
        # this loop batches flattened actor-critic transitions independently.
        self._mini_batch = step_mini_batch
        if not 0.0 <= self._gamma <= 1.0:
            raise ValueError(f"PPO gamma must be in [0, 1], got {self._gamma}")
        if not 0.0 <= self._gae_lambda <= 1.0:
            raise ValueError(
                f"PPO gae_lambda must be in [0, 1], got {self._gae_lambda}"
            )

    def all_reduce_states(
        self,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> float:
        """Apply the optimizer step and clamp Gaussian std on its CUDA stream."""
        caller_stream = torch.cuda.current_stream()
        grad_norm = super().all_reduce_states(inter_policy_nccl)
        clamp_std = getattr(self.model, "clamp_std_", None)
        if callable(clamp_std):
            # The shared optimizer step is queued on ``train_stream``.  Queue
            # the in-place clamp on the same stream so it cannot race Adam,
            # then make the caller/default stream wait before the next model
            # forward observes the parameters.
            with torch.cuda.stream(self.train_stream):
                clamp_std(
                    min_std=self._min_action_std,
                    max_std=self._max_action_std,
                )
        caller_stream.wait_stream(self.train_stream)
        return grad_norm

    def _actor_parameters_for_kl_backtracking(self) -> list[torch.nn.Parameter]:
        """Return the action-head parameters owned by the first optimizer part.

        The VLA model publishes ``[action_header, critic]`` through
        ``separate_model_parts``.  Keeping this boundary explicit prevents a
        behavior-KL correction from weakening the independently supervised
        critic update.
        """

        separate_model_parts = getattr(self.model, "separate_model_parts", None)
        if not callable(separate_model_parts):
            raise TypeError(
                "PPO behavior-KL backtracking requires model.separate_model_parts()"
            )
        parts = separate_model_parts()
        if not isinstance(parts, (list, tuple)) or len(parts) != 2:
            raise ValueError(
                "PPO behavior-KL backtracking requires exactly [actor, critic] parts"
            )
        actor = parts[0]
        critic = parts[1]

        def trainable_parameters(part: Any, *, label: str) -> list[torch.nn.Parameter]:
            parameters_method = getattr(part, "parameters", None)
            if not callable(parameters_method):
                raise TypeError(f"PPO {label} part must expose parameters()")
            parameters: list[torch.nn.Parameter] = []
            seen: set[int] = set()
            for parameter in parameters_method():
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(f"PPO {label} parameters must be torch Parameters")
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                parameters.append(parameter)
            return parameters

        parameters = trainable_parameters(actor, label="actor")
        critic_parameters = trainable_parameters(critic, label="critic")
        if not parameters:
            raise ValueError("PPO actor part has no trainable parameters")
        if not critic_parameters:
            raise ValueError("PPO critic part has no trainable parameters")
        actor_parameter_ids = {id(parameter) for parameter in parameters}
        critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
        if actor_parameter_ids & critic_parameter_ids:
            raise ValueError("PPO actor and critic trainable parameters overlap")
        optimizer_parameter_sets = _optimizer_parameter_id_sets(self.optimizers)
        if len(optimizer_parameter_sets) != 2:
            raise ValueError(
                "PPO behavior-KL backtracking requires actor and critic optimizer leaves"
            )
        if optimizer_parameter_sets[0] != actor_parameter_ids:
            raise ValueError(
                "PPO first optimizer leaf does not exactly own the actor parameters"
            )
        if optimizer_parameter_sets[1] != critic_parameter_ids:
            raise ValueError(
                "PPO second optimizer leaf does not exactly own the critic parameters"
            )
        return parameters

    def _snapshot_actor_for_kl_backtracking(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.Tensor]]:
        """Copy the pre-step actor to CPU before the sole Adam update."""

        parameters = self._actor_parameters_for_kl_backtracking()
        snapshots = [
            parameter.detach().to(device="cpu", copy=True) for parameter in parameters
        ]
        return parameters, snapshots

    @staticmethod
    def _scale_actor_step_toward_snapshot_(
        parameters: list[torch.nn.Parameter],
        snapshots: list[torch.Tensor],
        *,
        relative_scale: float,
    ) -> None:
        """Scale the current actor delta toward its immutable pre-step state."""

        if len(parameters) != len(snapshots) or not parameters:
            raise ValueError("PPO actor snapshot does not match live parameters")
        if not math.isfinite(relative_scale) or not 0.0 <= relative_scale <= 1.0:
            raise ValueError("PPO actor relative step scale must be within [0, 1]")
        # Validate the complete structure before mutating the first tensor.  A
        # malformed later entry must not leave an earlier parameter partially
        # interpolated.
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            if tuple(parameter.shape) != tuple(snapshot.shape):
                raise ValueError("PPO actor snapshot shape changed")
            if not parameter.is_floating_point() or not snapshot.is_floating_point():
                raise TypeError("PPO actor backtracking requires floating parameters")
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots, strict=True):
                if relative_scale == 0.0:
                    # Exact rejection recovery must overwrite NaN/Inf.  The
                    # algebraically equivalent ``parameter * 0 + old`` is not
                    # safe because IEEE NaN/Inf multiplied by zero remains
                    # non-finite.
                    parameter.copy_(snapshot)
                    continue
                if relative_scale == 1.0:
                    continue
                old_parameter = snapshot.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                parameter.mul_(relative_scale).add_(
                    old_parameter,
                    alpha=1.0 - relative_scale,
                )

    def _backtrack_behavior_kl(
        self,
        samples: list[Any],
        initial_metrics: dict[str, float | int],
        actor_snapshot: tuple[list[torch.nn.Parameter], list[torch.Tensor]],
        *,
        actor_learning_rate: float,
    ) -> tuple[dict[str, float | int], dict[str, Any]]:
        """Shrink one actor update toward its measured behavior-KL target.

        Adam and the critic are updated exactly once.  Only the actor parameter
        delta is interpolated toward the pre-step actor, which is equivalent to
        lowering the actor LR for this update while preserving Adam's moment
        accumulation and the complete critic update.
        """

        target = self._target_behavior_kl
        if target is None:
            raise RuntimeError("PPO behavior-KL backtracking has no target")
        parameters, snapshots = actor_snapshot
        initial_kl = float(initial_metrics["train/post_update_approx_kl"])
        metrics = initial_metrics
        current_scale = 1.0
        attempts = 0
        history: list[dict[str, float | int]] = [
            {"attempt": 0, "actor_step_scale": 1.0, "behavior_kl": initial_kl}
        ]
        try:
            if math.isfinite(initial_kl) and initial_kl > target:
                for attempt in range(1, self._behavior_kl_backtrack_max_attempts + 1):
                    # ``margin`` is a conservative scale factor, not a second
                    # threshold.  The target guides bounded correction; a
                    # finite residual overshoot is reported, not made fatal.
                    proposed_scale = current_scale * min(
                        0.75,
                        self._behavior_kl_backtrack_margin
                        * math.sqrt(
                            target / float(metrics["train/post_update_approx_kl"])
                        ),
                    )
                    relative_scale = proposed_scale / current_scale
                    self._scale_actor_step_toward_snapshot_(
                        parameters,
                        snapshots,
                        relative_scale=relative_scale,
                    )
                    current_scale = proposed_scale
                    attempts = attempt
                    metrics = self._post_update_diagnostics(samples)
                    observed_kl = float(metrics["train/post_update_approx_kl"])
                    history.append(
                        {
                            "attempt": attempt,
                            "actor_step_scale": current_scale,
                            "behavior_kl": observed_kl,
                        }
                    )
                    if math.isfinite(observed_kl) and observed_kl <= target:
                        break
                    if not math.isfinite(observed_kl):
                        break
        except Exception as candidate_failure:
            # A diagnostic can fail after a partial interpolation.  Restore in
            # the helper itself so no caller can accidentally observe that
            # partially scaled actor, then preserve the original exception.
            try:
                self._scale_actor_step_toward_snapshot_(
                    parameters,
                    snapshots,
                    relative_scale=0.0,
                )
            except Exception as restore_failure:
                raise ExceptionGroup(
                    "PPO actor backtracking and exception-safety restore failed",
                    [candidate_failure, restore_failure],
                ) from None
            raise

        candidate_kl = float(metrics["train/post_update_approx_kl"])
        target_met = math.isfinite(candidate_kl) and candidate_kl <= target
        target_mode = getattr(self, "_behavior_kl_target_mode", "hard")
        hard_limit = getattr(self, "_behavior_kl_hard_limit", None)
        if target_mode == "soft":
            if hard_limit is None:
                raise RuntimeError(
                    "PPO soft behavior-KL target mode has no catastrophic hard limit"
                )
            candidate_accepted = math.isfinite(candidate_kl) and candidate_kl <= float(
                hard_limit
            )
        else:
            candidate_accepted = target_met
        # In soft mode, a finite target miss below the independent catastrophe
        # limit stays live and is explicit in the receipt. Hard mode preserves
        # the legacy target-as-acceptance behavior for other profiles.
        backtrack_failed = not candidate_accepted
        restored_kl: float | None = None
        if backtrack_failed:
            # Non-finite or above-limit diagnostics restore the exact old
            # actor for the rejection receipt. The critic and optimizer moments remain
            # process-local and can never cross scheduler/checkpoint/sync.
            try:
                self._scale_actor_step_toward_snapshot_(
                    parameters,
                    snapshots,
                    relative_scale=0.0,
                )
                current_scale = 0.0
                metrics = self._post_update_diagnostics(samples)
                restored_kl = float(metrics["train/post_update_approx_kl"])
            except Exception as rejection_failure:
                try:
                    self._scale_actor_step_toward_snapshot_(
                        parameters,
                        snapshots,
                        relative_scale=0.0,
                    )
                except Exception as restore_failure:
                    raise ExceptionGroup(
                        "PPO rejected candidate and actor restoration failed",
                        [rejection_failure, restore_failure],
                    ) from None
                raise
            history.append(
                {
                    "attempt": attempts + 1,
                    "actor_step_scale": 0.0,
                    "behavior_kl": restored_kl,
                }
            )
        backtrack_metrics: dict[str, Any] = {
            "train/behavior_kl_before_backtrack": initial_kl,
            "train/behavior_kl_after_backtrack": candidate_kl,
            "train/actor_step_scale": current_scale,
            "train/actor_backtrack_attempts": attempts,
            "train/actor_configured_learning_rate": actor_learning_rate,
            "train/actor_effective_learning_rate": (
                actor_learning_rate * current_scale
            ),
            "train/actor_backtrack_failed": int(backtrack_failed),
            "train/behavior_kl_target_mode": target_mode,
            "train/behavior_kl_hard_limit": hard_limit,
            "train/behavior_kl_hard_gate_passed": int(candidate_accepted),
            "train/actor_backtrack_target_met": int(target_met),
            "train/actor_backtrack_target_missed": int(
                math.isfinite(candidate_kl) and not target_met
            ),
            "train/actor_backtrack_exhausted": int(
                math.isfinite(candidate_kl)
                and not target_met
                and attempts >= self._behavior_kl_backtrack_max_attempts
            ),
            "train/actor_backtrack_history": history,
        }
        if math.isfinite(candidate_kl):
            backtrack_metrics["train/behavior_kl_target_overshoot"] = max(
                candidate_kl - target, 0.0
            )
        if backtrack_failed:
            backtrack_metrics.update(
                {
                    "train/actor_restore_attempted": 1,
                    "train/actor_restore_succeeded": 1,
                    "train/actor_optimizer_state_rolled_back": 0,
                }
            )
            if math.isfinite(candidate_kl):
                backtrack_metrics["train/behavior_kl_last_unsafe_candidate"] = (
                    candidate_kl
                )
            if restored_kl is not None and math.isfinite(restored_kl):
                backtrack_metrics["train/behavior_kl_after_restore"] = restored_kl
        logger.info(
            "AlpaGym PPO behavior-KL actor backtracking initial_kl=%.6f "
            "candidate_kl=%.6f target=%.6f target_met=%s attempts=%d "
            "actor_step_scale=%.6f effective_actor_lr=%.8g",
            initial_kl,
            candidate_kl,
            target,
            target_met,
            attempts,
            current_scale,
            actor_learning_rate * current_scale,
        )
        return metrics, backtrack_metrics

    def _prepare_training_data(
        self,
        rollouts: list[_rollout_schema.Rollout],
    ) -> tuple[list[Any], torch.Tensor]:
        """Compute full-episode GAE, then flatten only rows needed for training.

        On one GPU, synthetic padding rows are removed after GAE so their cloned
        camera tensors never enter the expensive visual model forward. Real
        ``actor_valid=false`` rows remain because they still supervise the value
        head. Multi-rank jobs retain padding until cross-rank microbatch
        scheduling is made collective-safe.
        """
        samples: list[Any] = []
        advantages: list[float] = []
        actor_valid_rows: list[bool] = []
        per_rollout_ranges: list[tuple[int, int]] = []
        compact_padding = (
            int(getattr(getattr(self, "parallel_dims", None), "world_size", 1)) == 1
        )
        padding_rows_dropped = 0
        rollout_versions: set[int] = set()
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        for rollout in rollouts:
            start = len(samples)
            step_samples = data_packer.get_policy_input(
                rollout.prompt,
                rollout.completion,
                n_ignore_prefix_tokens=rollout.n_ignore_prefix_tokens,
            )
            rollout_weight_version = int(rollout.weight_version)
            rollout_versions.add(rollout_weight_version)
            for step in step_samples:
                replay_weight_version = int(step.weight_version.item())
                if replay_weight_version != rollout_weight_version:
                    raise ValueError(
                        "PPO replay behavior version does not match Cosmos rollout "
                        f"version: replay={replay_weight_version}, "
                        f"rollout={rollout_weight_version}"
                    )
            rollout_advantages, rollout_returns = self._compute_gae(step_samples)
            for step, advantage, ret in zip(
                step_samples, rollout_advantages, rollout_returns
            ):
                is_padding = bool(step.training_signal.is_padding.item())
                if is_padding and compact_padding:
                    padding_rows_dropped += 1
                    continue
                actor_valid = (
                    bool(
                        _ppo_actor_valid_mask(
                            step.training_signal,
                            required=self._flow_chunk_density,
                        ).item()
                    )
                    and not is_padding
                )
                actor_advantage = advantage if actor_valid else 0.0
                samples.append(
                    _with_ppo_targets(
                        step,
                        advantage=actor_advantage,
                        ret=ret,
                    )
                )
                advantages.append(actor_advantage)
                actor_valid_rows.append(actor_valid)
            per_rollout_ranges.append((start, len(samples)))

        if getattr(self, "_on_policy", False) and len(rollout_versions) > 1:
            raise ValueError(
                "On-policy PPO requires one frozen behavior weight version per "
                f"optimizer batch, got {sorted(rollout_versions)}"
            )
        if padding_rows_dropped:
            logger.info(
                "AlpaGym PPO removed %d synthetic padding rows before visual forward; "
                "%d real rows remain",
                padding_rows_dropped,
                len(samples),
            )

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        valid_mask = torch.tensor(actor_valid_rows, dtype=torch.bool)
        raw_valid_advantages = advantage_tensor[valid_mask]
        self._last_ppo_advantage_metrics = _ppo_advantage_metrics(
            raw_valid_advantages,
            prefix="train/ppo_advantage_raw",
        )
        if self._normalize_advantages and advantage_tensor.numel() > 0:
            if int(valid_mask.sum().item()) > 1:
                valid_advantages = advantage_tensor[valid_mask]
                std = valid_advantages.std(unbiased=False)
                if float(std.item()) > 0.0:
                    advantage_tensor[valid_mask] = (
                        valid_advantages - valid_advantages.mean()
                    ) / (std + 1.0e-8)
            advantage_tensor[~valid_mask] = 0.0
            samples = _replace_advantages(samples, advantage_tensor, per_rollout_ranges)
        self._last_ppo_advantage_metrics.update(
            _ppo_advantage_metrics(
                advantage_tensor[valid_mask],
                prefix="train/ppo_advantage_effective",
            )
        )
        logger.info(
            "AlpaGym PPO actor advantages rows=%d raw_min=%.6f raw_mean=%.6f "
            "raw_max=%.6f raw_std=%.6f effective_min=%.6f "
            "effective_mean=%.6f effective_max=%.6f effective_std=%.6f",
            int(self._last_ppo_advantage_metrics["train/ppo_advantage_raw_rows"]),
            float(self._last_ppo_advantage_metrics["train/ppo_advantage_raw_min"]),
            float(self._last_ppo_advantage_metrics["train/ppo_advantage_raw_mean"]),
            float(self._last_ppo_advantage_metrics["train/ppo_advantage_raw_max"]),
            float(self._last_ppo_advantage_metrics["train/ppo_advantage_raw_std"]),
            float(
                self._last_ppo_advantage_metrics["train/ppo_advantage_effective_min"]
            ),
            float(
                self._last_ppo_advantage_metrics["train/ppo_advantage_effective_mean"]
            ),
            float(
                self._last_ppo_advantage_metrics["train/ppo_advantage_effective_max"]
            ),
            float(
                self._last_ppo_advantage_metrics["train/ppo_advantage_effective_std"]
            ),
        )
        return samples, advantage_tensor

    def _run_training_loop(
        self,
        samples: list[Any],
        advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, int, float, float, float, float]:
        """Accumulate transition microbatches into one Adam step per PPO pass.

        ``step_mini_batch`` limits one visual forward/backward. It does not
        define the optimizer batch. Actor and value losses use separate exact
        row-count weights, so splitting a batch does not change either masked
        mean objective.
        """
        self._ensure_reference_model()
        self._optimizer_steps_applied_in_training_step = 0
        if not samples:
            return (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        if len(advantages) != len(samples):
            raise ValueError(
                "PPO advantages must align one-for-one with replay samples: "
                f"{len(advantages)} != {len(samples)}"
            )

        def row_counts(rows: list[Any]) -> tuple[int, int]:
            actor_rows = 0
            value_rows = 0
            for sample in rows:
                signal = sample.training_signal
                is_padding = bool(signal.is_padding.item())
                if is_padding:
                    continue
                value_rows += 1
                if bool(
                    _ppo_actor_valid_mask(
                        signal,
                        required=self._flow_chunk_density,
                    ).item()
                ):
                    actor_rows += 1
            return actor_rows, value_rows

        total_actor_rows, total_value_rows = row_counts(samples)
        if total_value_rows == 0:
            raise ValueError("PPO optimizer batch has no non-padding value rows")

        num_steps = len(samples)
        micro_batch_size = min(self._mini_batch, num_steps)
        num_micro_batches = (num_steps + micro_batch_size - 1) // micro_batch_size
        self._last_micro_batches = 0

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        ratio_max = float("-inf")
        ratio_min = float("inf")
        clip_fraction_sum = 0.0
        grad_norm_sum = 0.0

        config_train = getattr(getattr(self, "config", None), "train", None)
        configured_seed = getattr(config_train, "seed", None)
        write_receipts = bool(
            getattr(self, "_write_ppo_update_diagnostic_receipts", False)
        )
        if configured_seed is None:
            if write_receipts:
                raise ValueError(
                    "formal PPO optimizer permutation requires config.train.seed"
                )
            configured_seed = int(torch.initial_seed())
        if isinstance(configured_seed, bool) or not isinstance(configured_seed, int):
            raise TypeError("PPO optimizer permutation seed must be an integer")
        current_step = getattr(self, "_active_optimizer_step", None)
        if current_step is None:
            if write_receipts:
                raise RuntimeError(
                    "formal PPO optimizer permutation requires the active trainer step"
                )
            current_step = 0
        permutation_records: list[dict[str, Any]] = []

        for optimization_iteration in range(self._grpo_optimization_iterations):
            indices, permutation_record = _deterministic_optimizer_permutation(
                num_steps=num_steps,
                base_seed=configured_seed,
                current_step=current_step,
                optimization_iteration=optimization_iteration,
            )
            permutation_records.append(permutation_record)
            self.optimizers.zero_grad()
            update_loss = 0.0
            update_kl = 0.0
            update_clip_fraction = 0.0
            update_ratio_max = float("-inf")
            update_ratio_min = float("inf")

            for microbatch_index in range(num_micro_batches):
                start = microbatch_index * micro_batch_size
                end = min(start + micro_batch_size, num_steps)
                minibatch_indices = indices[start:end]
                minibatch_samples = [samples[int(index)] for index in minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                actor_rows, value_rows = row_counts(minibatch_samples)
                actor_loss_scale = (
                    actor_rows / total_actor_rows if total_actor_rows else 0.0
                )
                value_loss_scale = value_rows / total_value_rows

                (
                    loss_value,
                    kl_value,
                    batch_ratio_max,
                    batch_ratio_min,
                    batch_clip_fraction,
                    _batch_grad_norm,
                ) = self._train_minibatch(
                    minibatch_samples,
                    minibatch_advantages,
                    inter_policy_nccl,
                    actor_loss_scale=actor_loss_scale,
                    value_loss_scale=value_loss_scale,
                    apply_optimizer=False,
                )
                update_loss += loss_value
                update_kl += actor_loss_scale * kl_value
                if actor_rows:
                    update_ratio_max = max(update_ratio_max, batch_ratio_max)
                    update_ratio_min = min(update_ratio_min, batch_ratio_min)
                    update_clip_fraction += actor_loss_scale * batch_clip_fraction
                self._last_micro_batches += 1

            optimizer_steps_before = self._optimizer_steps_applied_in_training_step
            grad_norm = self.all_reduce_states(inter_policy_nccl)
            optimizer_step_applied = (
                self._optimizer_steps_applied_in_training_step > optimizer_steps_before
            )
            if total_actor_rows:
                ratio_max = max(ratio_max, update_ratio_max)
                ratio_min = min(ratio_min, update_ratio_min)
            total_loss += update_loss
            total_kl += update_kl
            clip_fraction_sum += update_clip_fraction
            grad_norm_sum += grad_norm
            if optimizer_step_applied:
                num_updates += 1
            logger.info(
                "AlpaGym PPO effective optimizer update rows=%d actor_rows=%d "
                "micro_batches=%d applied=%s loss=%.6f kl=%.6f grad_norm=%.6f",
                total_value_rows,
                total_actor_rows,
                num_micro_batches,
                optimizer_step_applied,
                update_loss,
                update_kl,
                grad_norm,
            )

        self._last_optimizer_permutation_metrics = {
            "train/optimizer_permutation_records": permutation_records,
        }

        if not total_actor_rows:
            ratio_max = 1.0
            ratio_min = 1.0

        return (
            total_loss,
            total_kl,
            num_updates,
            ratio_max,
            ratio_min,
            clip_fraction_sum,
            grad_norm_sum,
        )

    def _pre_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Measure behavior/replay alignment before any optimizer mutation."""
        return self._behavior_diagnostics(samples, phase="pre_update")

    def _post_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Measure behavior-policy drift after the effective optimizer step."""
        return self._behavior_diagnostics(samples, phase="post_update")

    def _behavior_diagnostics(
        self,
        samples: list[Any],
        *,
        phase: str,
    ) -> dict[str, float | int]:
        """Re-score frozen behavior actions under one coherent policy state.

        This is a pure no-grad forward pass: it never calls backward, gradient
        reduction, an optimizer, or the scheduler. Each actor-valid replay row
        contributes one scalar policy-density ratio.
        """
        if phase not in {"pre_update", "post_update"}:
            raise ValueError(f"Unsupported PPO behavior-diagnostic phase: {phase!r}")
        if not samples:
            raise ValueError(f"PPO {phase} diagnostics require replay samples")
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError(f"PPO {phase} diagnostics require a data packer")

        valid_log_ratios: list[torch.Tensor] = []
        valid_value_deltas: list[torch.Tensor] = []
        diagnostic_batch_size = min(self._mini_batch, len(samples))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(samples), diagnostic_batch_size):
                    minibatch = data_packer.policy_collate_fn(
                        samples[start : start + diagnostic_batch_size]
                    )
                    signal = minibatch.training_signal
                    is_padding = signal.is_padding.to(self.device)
                    actor_valid = _ppo_actor_valid_mask(
                        signal,
                        required=self._flow_chunk_density,
                    ).to(self.device)
                    old_logprobs = signal.old_logprobs.to(self.device)
                    old_values = _require_ppo_signal(
                        signal.old_values,
                        "old_values",
                    ).to(self.device)
                    (
                        new_logprobs,
                        _kl_div,
                        values,
                        new_element_logprobs,
                        old_element_logprobs,
                    ) = self._forward_with_reference_and_value(minibatch.model_inputs)

                    if tuple(new_logprobs.shape) != tuple(old_logprobs.shape):
                        raise ValueError(
                            f"PPO {phase} new/old scalar log-probability shapes differ: "
                            f"{tuple(new_logprobs.shape)} != "
                            f"{tuple(old_logprobs.shape)}"
                        )
                    if not torch.isfinite(new_logprobs).all():
                        raise FloatingPointError(
                            f"PPO {phase} model returned non-finite log-probabilities"
                        )
                    if not torch.isfinite(old_logprobs).all():
                        raise FloatingPointError(
                            f"PPO {phase} replay has non-finite old log-probabilities"
                        )
                    if tuple(values.shape) != tuple(old_values.shape):
                        raise ValueError(
                            f"PPO {phase} current/old value shapes differ: "
                            f"{tuple(values.shape)} != {tuple(old_values.shape)}"
                        )
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            f"PPO {phase} model returned non-finite values"
                        )
                    if not torch.isfinite(old_values).all():
                        raise FloatingPointError(
                            f"PPO {phase} replay has non-finite old values"
                        )

                    if self._flow_chunk_density:
                        if new_element_logprobs is None or old_element_logprobs is None:
                            raise ValueError(
                                "Flow-PPO requires full selected-transition element "
                                "log-probabilities"
                            )
                        if not torch.allclose(
                            old_element_logprobs.sum(dim=-1),
                            old_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ) or not torch.allclose(
                            new_element_logprobs.sum(dim=-1),
                            new_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ):
                            raise ValueError(
                                "Flow-PPO joint log-probability differs from its "
                                "selected-transition elements"
                            )
                    elif (
                        new_element_logprobs is not None
                        or old_element_logprobs is not None
                    ):
                        raise ValueError(
                            "alpagym_ppo accepts only scalar policy densities; "
                            "use alpagym_flow_ppo for Flow-SDE elements"
                        )

                    log_ratios = new_logprobs - old_logprobs
                    valid_mask = (~is_padding) & actor_valid

                    selected = log_ratios[valid_mask]
                    if selected.numel() > 0:
                        valid_log_ratios.append(
                            selected.detach().to(device="cpu", dtype=torch.float32)
                        )
                    selected_value_deltas = (values - old_values).abs()[~is_padding]
                    if selected_value_deltas.numel() > 0:
                        valid_value_deltas.append(
                            selected_value_deltas.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            )
                        )
        finally:
            self.model.train(was_training)

        metrics = _summarize_behavior_log_ratios(
            torch.cat(valid_log_ratios) if valid_log_ratios else torch.empty(0),
            phase=phase,
            ratio_clip_low=self._grpo_ratio_clip_low,
            ratio_clip_high=self._grpo_ratio_clip_high,
        )
        prefix = f"train/{phase}"
        value_deltas = (
            torch.cat(valid_value_deltas)
            if valid_value_deltas
            else torch.empty(0, dtype=torch.float32)
        )
        metrics[f"{prefix}_value_valid_rows"] = int(value_deltas.numel())
        metrics[f"{prefix}_value_max_abs_delta"] = (
            float(value_deltas.max().item()) if value_deltas.numel() else 0.0
        )
        logger.info(
            "AlpaGym PPO %s diagnostics valid_rows=%d "
            "ratio_p01=%.6f ratio_p50=%.6f ratio_p99=%.6f "
            "clip_fraction=%.6f approx_kl=%.6f max_abs_log_ratio=%.6f "
            "max_abs_ratio_error=%.6f "
            "value_rows=%d value_max_abs_delta=%.6f",
            phase,
            int(metrics[f"{prefix}_valid_rows"]),
            float(metrics[f"{prefix}_ratio_p01"]),
            float(metrics[f"{prefix}_ratio_p50"]),
            float(metrics[f"{prefix}_ratio_p99"]),
            float(metrics[f"{prefix}_clip_fraction"]),
            float(metrics[f"{prefix}_approx_kl"]),
            float(metrics[f"{prefix}_max_abs_log_ratio"]),
            float(metrics[f"{prefix}_max_abs_ratio_error"]),
            int(metrics[f"{prefix}_value_valid_rows"]),
            float(metrics[f"{prefix}_value_max_abs_delta"]),
        )
        return metrics

    def _validate_update_diagnostics(
        self,
        metrics: dict[str, float | int],
        *,
        phase: str,
    ) -> None:
        """Validate replay identity and finite behavior-policy diagnostics.

        ``target_behavior_kl`` is an optimizer calibration target.  Bounded
        actor backtracking records whether it was reached, but a finite target
        miss is not an infrastructure failure and must not terminate training.
        """
        if phase == "pre_update" and self._on_policy:
            valid_rows = int(metrics["train/pre_update_valid_rows"])
            max_abs_ratio_error = float(metrics["train/pre_update_max_abs_ratio_error"])
            if valid_rows > 0 and (
                not math.isfinite(max_abs_ratio_error) or max_abs_ratio_error > 1.0e-5
            ):
                raise FloatingPointError(
                    "On-policy PPO pre-update replay differs from its behavior "
                    "policy: max_abs_ratio_error="
                    f"{max_abs_ratio_error:.6g} exceeds 1e-05"
                )
            if self._flow_chunk_density:
                value_rows = int(metrics["train/pre_update_value_valid_rows"])
                max_abs_value_delta = float(
                    metrics["train/pre_update_value_max_abs_delta"]
                )
                if value_rows <= 0:
                    raise FloatingPointError(
                        "On-policy Flow-PPO pre-update replay has no value rows"
                    )
                if (
                    not math.isfinite(max_abs_value_delta)
                    or max_abs_value_delta > 1.0e-5
                ):
                    raise FloatingPointError(
                        "On-policy Flow-PPO pre-update replay differs from its "
                        "behavior critic: max_abs_value_delta="
                        f"{max_abs_value_delta:.6g} exceeds 1e-05"
                    )
        target = getattr(self, "_target_behavior_kl", None)
        if target is None:
            return
        if phase not in {"pre_update", "post_update"}:
            raise ValueError(f"Unsupported PPO behavior-KL phase: {phase!r}")
        phase_label = phase.replace("_", "-")
        prefix = f"train/{phase}"
        valid_rows = int(metrics[f"{prefix}_valid_rows"])
        approx_kl = float(metrics[f"{prefix}_approx_kl"])
        if valid_rows <= 0:
            if self._flow_chunk_density:
                # Flow actor and critic parameters are disjoint, so a
                # critic-only batch cannot move the behavior density.
                return
            raise FloatingPointError(
                f"PPO {phase_label} behavior-KL guard has no actor-valid replay rows"
            )
        if not math.isfinite(approx_kl):
            raise FloatingPointError(
                f"PPO {phase_label} behavior KL is non-finite: {approx_kl}"
            )
        target_mode = getattr(self, "_behavior_kl_target_mode", "hard")
        hard_limit = getattr(self, "_behavior_kl_hard_limit", None)
        acceptance_limit = target if target_mode == "hard" else hard_limit
        if acceptance_limit is None:
            raise RuntimeError(
                "PPO soft behavior-KL target mode has no catastrophic hard limit"
            )
        if approx_kl > float(acceptance_limit):
            raise FloatingPointError(
                f"PPO {phase_label} behavior KL {approx_kl:.6g} exceeds "
                f"{target_mode} acceptance limit {float(acceptance_limit):.6g}; "
                "refusing to advance scheduler, checkpoint, or weight sync"
            )
        if phase == "post_update" and approx_kl > target:
            logger.warning(
                "PPO post-update behavior KL %.6g remains above soft target %.6g; "
                "continuing with the finite audited candidate",
                approx_kl,
                target,
            )

    def _compute_gae(self, step_samples: list[Any]) -> tuple[list[float], list[float]]:
        """Compute direct or semi-Markov GAE targets for one rollout."""
        transitions: list[tuple[float, int, bool, bool, float, float | None]] = []
        valid_indices: list[int] = []
        for index, step in enumerate(step_samples):
            signal = step.training_signal
            if bool(signal.is_padding.item()):
                continue
            terminated = _require_ppo_signal(signal.terminateds, "terminateds")
            truncated = (
                False if signal.truncateds is None else bool(signal.truncateds.item())
            )
            if bool(terminated.item()) and truncated:
                raise ValueError(
                    "PPO transition cannot be both terminated and truncated"
                )
            value = _require_ppo_signal(signal.old_values, "old_values")
            reward, duration = _discounted_transition_reward(
                signal,
                gamma_tick=self._gamma,
            )
            bootstrap = (
                None
                if signal.bootstrap_values is None
                else float(signal.bootstrap_values.item())
            )
            transitions.append(
                (
                    reward,
                    duration,
                    bool(terminated.item()),
                    truncated,
                    float(value.item()),
                    bootstrap,
                )
            )
            valid_indices.append(index)

        advantages = [0.0 for _ in step_samples]
        returns = [0.0 for _ in step_samples]
        last_gae = 0.0
        for valid_pos in reversed(range(len(valid_indices))):
            sample_index = valid_indices[valid_pos]
            reward, duration, terminated, truncated, value, bootstrap = transitions[
                valid_pos
            ]
            if bootstrap is None:
                bootstrap = (
                    transitions[valid_pos + 1][4]
                    if valid_pos + 1 < len(transitions)
                    else 0.0
                )
            bootstrap_discount = 0.0 if terminated else self._gamma**duration
            delta = reward + bootstrap_discount * bootstrap - value
            continues = (
                not terminated and not truncated and valid_pos + 1 < len(transitions)
            )
            trace_discount = (
                (self._gamma * self._gae_lambda) ** duration if continues else 0.0
            )
            last_gae = delta + trace_discount * last_gae
            advantages[sample_index] = float(last_gae)
            returns[sample_index] = float(last_gae + value)
        return advantages, returns

    def _train_minibatch(
        self,
        minibatch_samples: list[Any],
        minibatch_advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
        *,
        actor_loss_scale: float = 1.0,
        value_loss_scale: float = 1.0,
        apply_optimizer: bool = True,
    ) -> tuple[float, float, float, float, float, float]:
        """Backprop one PPO transition microbatch.

        Direct callers retain the historical one-minibatch/one-step behavior.
        The PPO loop passes exact actor/value row fractions and defers the Adam
        step so several visual microbatches form one effective optimizer batch.
        """
        if actor_loss_scale < 0.0 or value_loss_scale < 0.0:
            raise ValueError("PPO microbatch loss scales must be non-negative")
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        minibatch = data_packer.policy_collate_fn(minibatch_samples)
        signal = minibatch.training_signal
        is_padding = signal.is_padding.to(self.device)
        actor_valid = _ppo_actor_valid_mask(
            signal,
            required=self._flow_chunk_density,
        ).to(self.device)
        actor_is_padding = is_padding | ~actor_valid
        old_logprobs = signal.old_logprobs.to(self.device)
        advantages = minibatch_advantages.to(device=self.device, dtype=torch.float32)
        returns = _require_ppo_signal(signal.returns, "returns").to(self.device)
        old_values_tensor = signal.old_values
        old_values = (
            None if old_values_tensor is None else old_values_tensor.to(self.device)
        )
        if self._value_clip_range is not None and old_values is None:
            raise ValueError("PPO value clipping requires TrainingSignal.old_values")

        (
            new_logprobs,
            kl_div,
            values,
            new_element_logprobs,
            old_element_logprobs,
        ) = self._forward_with_reference_and_value(minibatch.model_inputs)
        assert_replay_shapes(
            new_logprobs,
            old_logprobs,
            advantages,
            kl_div,
            values=values,
            returns=returns,
            old_values=old_values,
        )
        if self._flow_chunk_density:
            if new_element_logprobs is None or old_element_logprobs is None:
                raise ValueError(
                    "Flow-PPO requires full selected-transition element "
                    "log-probabilities"
                )
            dual_clip_ratio = self._dual_clip_ratio
            if dual_clip_ratio is None:
                raise ValueError("Flow-PPO requires a configured dual_clip_ratio")
            policy_loss, ratio = compute_flow_ppo_surrogate(
                new_element_logprobs,
                old_element_logprobs,
                new_logprobs,
                old_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                dual_clip_ratio=dual_clip_ratio,
                is_padding=actor_is_padding,
            )
        else:
            if new_element_logprobs is not None or old_element_logprobs is not None:
                raise ValueError(
                    "alpagym_ppo accepts only scalar policy densities; "
                    "use alpagym_flow_ppo for Flow-SDE elements"
                )
            policy_loss, ratio = compute_ppo_surrogate(
                new_logprobs,
                old_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                is_padding=actor_is_padding,
                dual_clip_ratio=self._dual_clip_ratio,
            )
        kl_loss = compute_kl_penalty(
            kl_div,
            actor_is_padding,
            kl_beta=self._kl_beta,
            device=self.device,
        )
        value_loss = compute_value_loss(
            values,
            returns,
            is_padding,
            old_values=old_values,
            value_clip_range=self._value_clip_range,
            huber_delta=self._value_huber_delta,
        )
        loss = self._value_loss_coef * value_loss_scale * value_loss
        has_actor_rows = bool(((~is_padding) & actor_valid).any().item())
        if actor_loss_scale > 0.0 and has_actor_rows:
            # Do not attach an all-zero actor objective to critic-only batches.
            # A zero-but-non-None AdamW gradient would still apply weight decay
            # and advance actor optimizer state when the critic steps.
            loss = loss + actor_loss_scale * (policy_loss + kl_loss)

        if apply_optimizer:
            self.optimizers.zero_grad()
        loss.backward()
        grad_norm = (
            self.all_reduce_states(inter_policy_nccl) if apply_optimizer else 0.0
        )

        return self._ppo_minibatch_metrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_loss=kl_loss,
            ratio=ratio,
            is_padding=is_padding,
            actor_valid=actor_valid,
            advantages=advantages,
            returns=returns,
            values=values,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            grad_norm=grad_norm,
        )

    def _forward_with_reference_and_value(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Run actor-critic forward with optional Flow-density elements."""
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        old_element_logprobs = forward_kwargs.pop("old_element_logprobs", None)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        new_element_logprobs = result.get("element_log_probs")
        if (new_element_logprobs is None) != (old_element_logprobs is None):
            raise ValueError(
                "Flow-PPO requires both model element_log_probs and replay "
                "old_element_logprobs"
            )
        return (
            result["log_probs"],
            result.get("kl_div"),
            result["values"].reshape(-1),
            new_element_logprobs,
            old_element_logprobs,
        )

    def _ppo_minibatch_metrics(
        self,
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        ratio: torch.Tensor,
        is_padding: torch.Tensor,
        actor_valid: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        grad_norm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute PPO diagnostics over valid rows and return trainer-loop metrics."""
        actor_mask = (~is_padding) & actor_valid
        value_mask = ~is_padding
        loss_value = float(loss.item())
        with torch.no_grad():
            if bool(value_mask.any()):
                return_mean = float(returns[value_mask].mean().item())
                value_mean = float(values[value_mask].mean().item())
            else:
                return_mean = 0.0
                value_mean = 0.0
            if bool(actor_mask.any()):
                old_logprob_mean = float(old_logprobs[actor_mask].mean().item())
                new_logprob_mean = float(new_logprobs[actor_mask].mean().item())
            else:
                old_logprob_mean = 0.0
                new_logprob_mean = 0.0
            if ratio.shape != actor_mask.shape:
                raise ValueError(
                    "PPO joint ratio shape must match replay rows: "
                    f"{tuple(ratio.shape)} != {tuple(actor_mask.shape)}"
                )
            valid_ratio = ratio[actor_mask]
            if valid_ratio.numel() == 0:
                clip_fraction = 0.0
                batch_ratio_max = 1.0
                batch_ratio_min = 1.0
                advantage_mean = 0.0
            else:
                clipped = (valid_ratio < 1.0 - self._grpo_ratio_clip_low) | (
                    valid_ratio > 1.0 + self._grpo_ratio_clip_high
                )
                clip_fraction = float(clipped.float().mean().item())
                batch_ratio_max = float(valid_ratio.max().item())
                batch_ratio_min = float(valid_ratio.min().item())
                advantage_mean = float(advantages[actor_mask].mean().item())
        logger.info(
            "AlpaGym PPO minibatch rows=%d valid_rows=%d loss=%.6f "
            "policy_loss=%.6f value_loss=%.6f kl_loss=%.6f ratio_min=%.6f "
            "ratio_max=%.6f clip_fraction=%.6f advantage_mean=%.6f "
            "return_mean=%.6f value_mean=%.6f old_logprob_mean=%.6f "
            "new_logprob_mean=%.6f grad_norm=%.6f",
            int(old_logprobs.numel()),
            int(actor_mask.sum().item()),
            loss_value,
            float(policy_loss.item()),
            float(value_loss.item()),
            float(kl_loss.item()),
            batch_ratio_min,
            batch_ratio_max,
            clip_fraction,
            advantage_mean,
            return_mean,
            value_mean,
            old_logprob_mean,
            new_logprob_mean,
            grad_norm,
        )
        return (
            loss_value,
            float(kl_loss.item()),
            batch_ratio_max,
            batch_ratio_min,
            clip_fraction,
            grad_norm,
        )


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_flow_ppo")
class AlpagymFlowPPOTrainer(AlpagymPPOTrainer):
    """RLinf Flow-PPO over one selected Flow-SDE transition per decision.

    Policy packages retain the selected transition's elementwise Gaussian
    log-probabilities. The trainer sums the complete transition density into
    one joint chunk log-probability before applying the PPO ratio and clip.
    """

    _flow_chunk_density = True

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize and verify the source-pinned Flow-PPO loss contract."""
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        expected = {
            "ratio_clip_low": (self._grpo_ratio_clip_low, 0.2),
            "ratio_clip_high": (self._grpo_ratio_clip_high, 0.28),
            "dual_clip_ratio": (self._dual_clip_ratio, 3.0),
            "value_loss_coef": (self._value_loss_coef, 1.0),
            "value_clip_range": (self._value_clip_range, 0.2),
            "value_huber_delta": (self._value_huber_delta, 10.0),
            "gamma": (self._gamma, 0.99),
            "gae_lambda": (self._gae_lambda, 0.95),
        }
        mismatches = [
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "alpagym_flow_ppo differs from the pinned RLinf contract: "
                + ", ".join(mismatches)
            )
        if not self._normalize_advantages:
            raise ValueError("alpagym_flow_ppo requires normalized advantages")
        if self._kl_beta != 0.0 or self._reference_model is not None:
            raise ValueError("alpagym_flow_ppo does not use reference-model KL")

    def _compute_gae(self, step_samples: list[Any]) -> tuple[list[float], list[float]]:
        """Compute causal actor GAE and chronological critic returns at 50 Hz.

        RLinf configures ``gamma`` and ``lambda`` per nominal 2 Hz policy
        decision.  The realized feedback interval is instead a variable number
        of 50 Hz controller ticks, so both factors are converted to physical
        time before reward aggregation, bootstrap, and GAE tracing.

        Async inference can leave a predecessor reference active at the start
        of a sampled plan's interval.  The critic consumes that full
        chronological interval. The actor excludes the predecessor prefix
        through ``actor_primitive_reward_mask`` and consumes the causal
        ``actor_primitive_rewards`` view, which can also remove a support gain
        majority-owned by that predecessor. Source rewards retain their
        absolute tick offsets and the full interval still controls bootstrap
        and trace discount. Thus policy latency cannot be erased by rebasing the
        causal suffix to tick zero. The actor trace then follows the next row's
        full chronological GAE: its predecessor prefix is often the current plan
        still executing and therefore remains valid downstream credit.
        """
        nominal_duration_ticks = 25
        gamma_tick = _time_scaled_discount(
            self._gamma,
            duration_ticks=1,
            nominal_duration_ticks=nominal_duration_ticks,
        )
        transitions: list[
            tuple[float, float, int, bool, bool, float, float | None]
        ] = []
        valid_indices: list[int] = []
        for index, step in enumerate(step_samples):
            signal = step.training_signal
            if bool(signal.is_padding.item()):
                continue
            terminated = bool(
                _require_ppo_signal(signal.terminateds, "terminateds").item()
            )
            truncated = (
                False if signal.truncateds is None else bool(signal.truncateds.item())
            )
            if terminated and truncated:
                raise ValueError(
                    "Flow-PPO transition cannot be both terminated and truncated"
                )
            primitive_rewards = _require_ppo_signal(
                signal.primitive_rewards, "primitive_rewards"
            ).reshape(-1)
            full_reward, duration = _discounted_transition_reward(
                signal,
                gamma_tick=gamma_tick,
            )
            actor_reward_mask = _require_ppo_signal(
                signal.actor_primitive_reward_mask,
                "actor_primitive_reward_mask",
            )
            actor_primitive_rewards = _require_ppo_signal(
                signal.actor_primitive_rewards,
                "actor_primitive_rewards",
            )
            actor_reward, actor_duration = _discounted_transition_reward(
                signal,
                gamma_tick=gamma_tick,
                reward_values=actor_primitive_rewards,
                credit_mask=actor_reward_mask,
            )
            if actor_duration != duration:
                raise AssertionError("actor and critic reward durations diverged")
            raw_transition_reward = (
                primitive_rewards[:duration].to(dtype=torch.float64).cpu().sum()
            )
            transported_reward = _require_ppo_signal(signal.rewards, "rewards").reshape(
                -1
            )
            if (
                transported_reward.numel() != 1
                or not torch.isfinite(transported_reward).all()
            ):
                raise ValueError("Flow-PPO requires one finite transition reward")
            if not torch.isclose(
                raw_transition_reward,
                transported_reward[0].to(dtype=torch.float64),
                rtol=1.0e-6,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "Flow-PPO transition reward differs from its realized tick rewards"
                )
            value = float(_require_ppo_signal(signal.old_values, "old_values").item())
            bootstrap = (
                None
                if signal.bootstrap_values is None
                else float(signal.bootstrap_values.item())
            )
            transitions.append(
                (
                    full_reward,
                    actor_reward,
                    duration,
                    terminated,
                    truncated,
                    value,
                    bootstrap,
                )
            )
            valid_indices.append(index)

        advantages = [0.0 for _ in step_samples]
        returns = [0.0 for _ in step_samples]
        last_critic_gae = 0.0
        for valid_pos in reversed(range(len(valid_indices))):
            (
                full_reward,
                actor_reward,
                duration,
                terminated,
                truncated,
                value,
                bootstrap,
            ) = transitions[valid_pos]
            if bootstrap is None:
                bootstrap = (
                    transitions[valid_pos + 1][5]
                    if valid_pos + 1 < len(transitions)
                    else 0.0
                )
            gamma_duration = _time_scaled_discount(
                self._gamma,
                duration_ticks=duration,
                nominal_duration_ticks=nominal_duration_ticks,
            )
            bootstrap_term = 0.0 if terminated else gamma_duration * bootstrap
            actor_delta = actor_reward + bootstrap_term - value
            critic_delta = full_reward + bootstrap_term - value
            continues = (
                not terminated and not truncated and valid_pos + 1 < len(transitions)
            )
            trace_discount = (
                _time_scaled_discount(
                    self._gamma * self._gae_lambda,
                    duration_ticks=duration,
                    nominal_duration_ticks=nominal_duration_ticks,
                )
                if continues
                else 0.0
            )
            next_critic_gae = last_critic_gae
            last_critic_gae = critic_delta + trace_discount * next_critic_gae
            actor_gae = actor_delta + trace_discount * next_critic_gae
            sample_index = valid_indices[valid_pos]
            advantages[sample_index] = float(actor_gae)
            returns[sample_index] = float(last_critic_gae + value)
        return advantages, returns

    def _forward_with_reference_and_value(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Replay Flow density under the same CUDA bf16 path as rollout."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            return super()._forward_with_reference_and_value(model_inputs)
