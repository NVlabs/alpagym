# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tarfile
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import tomli_w
import yaml
from omegaconf import DictConfig, OmegaConf

from alpagym_host.config import ArtifactPaths, RunConfig, merge_run_config_schema


def build_artifact_paths(config: DictConfig) -> ArtifactPaths:
    """Build paths for one host-owned run directory."""
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex}"
    run_dir = Path(config.run_root) / run_id
    artifacts_dir = run_dir / "artifacts"
    return ArtifactPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        policy_model_bundle_dir=artifacts_dir / "policy_model_bundle",
        resolved_config_path=run_dir / "resolved_config.yaml",
        cosmos_config_path=run_dir / "cosmos_config.toml",
        submit_script_path=run_dir / "submit.sbatch",
        log_dir=run_dir / "logs",
        topology_registry_dir=run_dir / "topology",
        alpasim_log_dir=run_dir / "alpasim",
        alpasim_scene_ids_path=run_dir / "alpasim_scene_ids.yaml",
        perf_dir=run_dir / "perf",
    )


def build_run_config(
    config: DictConfig,
    artifact_paths: ArtifactPaths,
) -> RunConfig:
    """Build the concrete run config from authored config and generated paths."""
    config_dict = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    config_dict["artifact_paths"] = {
        field.name: str(getattr(artifact_paths, field.name))
        for field in fields(ArtifactPaths)
    }

    return merge_run_config_schema(RunConfig, config_dict)


def write_run_artifacts(config: RunConfig) -> None:
    """Create the run directory and write generated config artifacts."""
    artifact_paths = config.artifact_paths
    artifact_paths.run_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths.artifacts_dir.mkdir(exist_ok=True)
    artifact_paths.log_dir.mkdir(exist_ok=True)
    artifact_paths.alpasim_log_dir.mkdir(exist_ok=True)
    artifact_paths.perf_dir.mkdir(exist_ok=True)

    artifact_paths.cosmos_config_path.write_text(
        tomli_w.dumps(_build_cosmos_config(config)),
        encoding="utf-8",
    )

    config_dict = cast(dict[str, Any], _to_plain_data(asdict(config)))
    artifact_paths.resolved_config_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=False),
        encoding="utf-8",
    )


def normalize_generated_policy_model_path(config: RunConfig) -> RunConfig:
    """Extract tarball model paths and return a config pointing at the HF bundle directory."""
    model_path = Path(config.policy.model.path)
    if not (model_path.is_file() and tarfile.is_tarfile(model_path)):
        return config

    bundle_dir = config.artifact_paths.policy_model_bundle_dir
    _extract_policy_model_tarball(model_path, bundle_dir)
    return replace(
        config,
        policy=replace(
            config.policy,
            model=replace(config.policy.model, path=str(bundle_dir)),
        ),
    )


def _build_cosmos_config(config: RunConfig) -> dict[str, Any]:
    """Build the generated Cosmos-RL config as nested Python data.

    Adds fixed Cosmos-RL fields not exposed in Hydra.
    """
    artifact_paths = config.artifact_paths
    cosmos = cast(dict[str, Any], _to_plain_data(asdict(config.cosmos)))
    train = dict(cosmos["train"])
    train["epoch"] = train.pop("num_epochs")
    logging = dict(cosmos["logging"])
    logging["log_interval"] = logging.pop("log_training_metrics_every_n_steps")
    train_policy = dict(train["train_policy"])
    trainer_type = str(train_policy.pop("trainer_type", "alpagym_grpo"))
    ppo_config = {
        "value_loss_coef": train_policy.pop("ppo_value_loss_coef", 0.5),
        "value_clip_range": train_policy.pop("ppo_value_clip_range", None),
        "normalize_advantages": train_policy.pop("ppo_normalize_advantages", True),
        "gamma": train_policy.pop("ppo_gamma", 0.99),
        "gae_lambda": train_policy.pop("ppo_gae_lambda", 0.95),
        "min_action_std": train_policy.pop("ppo_min_action_std", 0.02),
        "max_action_std": train_policy.pop("ppo_max_action_std", 2.0),
    }
    step_mini_batch = train_policy.pop("step_mini_batch", None)
    if step_mini_batch is not None:
        ppo_config["step_mini_batch"] = step_mini_batch
    train_policy["epsilon_low"] = train_policy.pop("grpo_ratio_clip_low")
    train_policy["epsilon_high"] = train_policy.pop("grpo_ratio_clip_high")
    train_policy["mu_iterations"] = train_policy.pop("grpo_optimization_iterations")
    train["output_dir"] = str(artifact_paths.run_dir / "cosmos")
    # `non_text=True` avoids pickling large payloads by using threads not processes.
    train["non_text"] = True
    # ``BaseCosmosWrapper.parallelize_fn`` asserts compile is off.
    train["compile"] = False
    train["train_policy"] = {
        # Cosmos uses GRPO to select the RL policy and rollout worker path.
        "type": "grpo",
        "trainer_type": trainer_type,
        # Need to set this value, otherwise CosmosRL overwrites `type` with `sft`
        "use_remote_reward": False,
        **train_policy,
    }
    custom_config: dict[str, Any] = {
        "resolved_config_path": str(artifact_paths.resolved_config_path),
    }
    if trainer_type == "alpagym_ppo":
        custom_config["ppo"] = ppo_config

    cosmos_config = {
        "mode": cosmos["mode"],
        "train": train,
        "policy": {
            "model_name_or_path": config.policy.model.path,
            **dict(cosmos["policy"]),
        },
        "rollout": {
            "backend": "alpagym_rollout",
            **dict(cosmos["rollout"]),
        },
        "logging": logging,
        "custom": custom_config,
    }
    return cast(dict[str, Any], _drop_none_mapping_values(cosmos_config))


def _extract_policy_model_tarball(tarball_path: Path, bundle_dir: Path) -> None:
    """Extract a model tarball into the generated HF bundle directory."""
    with tarfile.open(tarball_path, "r:*") as tar:
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        if bundle_dir.exists():
            if not bundle_dir.is_dir():
                raise ValueError(
                    f"model bundle extraction target is not a directory: {bundle_dir}"
                )
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir()
        tar.extractall(bundle_dir, filter="data")


def is_supported_hf_bundle_dir(bundle_dir: Path) -> bool:
    """Return whether a directory has the HF files the HF bundle loader can read."""
    if not (bundle_dir / "config.json").is_file():
        return False
    if any(
        path.is_file() and _is_supported_hf_weight_filename(path.name)
        for path in bundle_dir.iterdir()
    ):
        return True
    return any(
        _shard_index_references_existing_files(bundle_dir / filename, bundle_dir)
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    )


def _is_supported_hf_weight_filename(filename: str) -> bool:
    """Return whether a filename matches the runtime HF bundle loader's weight globs."""
    if filename in ("model.safetensors", "pytorch_model.bin"):
        return True
    return any(
        fnmatch(filename, pattern)
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
    )


def _shard_index_references_existing_files(index_path: Path, bundle_dir: Path) -> bool:
    """Return whether a shard index references files present in `bundle_dir`."""
    if not index_path.is_file():
        return False
    index_data: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    shard_files = _shard_files_from_index(index_data)
    return bool(shard_files) and all(
        (bundle_dir / filename).is_file() for filename in shard_files
    )


def _shard_files_from_index(index_data: Any) -> set[str]:
    """Extract root-level shard filenames from a standard HF shard index."""
    if not isinstance(index_data, dict):
        return set()
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        return set()
    shard_files = {str(filename) for filename in weight_map.values() if filename}
    if any("/" in filename or filename.startswith(".") for filename in shard_files):
        return set()
    return shard_files


def _to_plain_data(value: Any) -> Any:
    """Convert config data to YAML/TOML-serializable Python values.

    Needed because `RunConfig` includes values that serializers do not
    handle directly.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _drop_none_mapping_values(value: Any) -> Any:
    """Recursively remove null mapping fields before writing TOML.

    TOML has no null value. Omit optional dict fields instead of maintaining
    one-off deletion logic for each Cosmos config key.
    """
    if isinstance(value, dict):
        return {
            key: _drop_none_mapping_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_mapping_values(item) for item in value]
    return value
