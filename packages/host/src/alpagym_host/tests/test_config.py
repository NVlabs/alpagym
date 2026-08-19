# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tarfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import alpagym_host.cli as host_cli
import pytest
import torch
import yaml
from alpagym_host.cli import load_or_create_run_config
from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ArtifactPaths,
    CosmosRLMode,
    ExecutionBackend,
    HumanoidAlpaSimConfig,
    HumanoidExecutionProfile,
    RunConfig,
    SeparateNodesSlurmTopologyConfig,
    TransportKind,
    load_run_config,
    register_config_schema,
)
from alpagym_host.config_validation import validate_run_config
from alpagym_host.humanoid_scene_identity import freeze_humanoid_scene_fingerprints
from alpagym_host.run_artifacts import (
    build_artifact_paths,
    build_run_config,
    write_run_artifacts,
)
from hydra import compose, initialize_config_module
from safetensors.torch import save_file


def test_host_writes_and_loads_handoff_artifacts(
    tmp_path: Path,
) -> None:
    """Writes generated handoff artifacts and loads the resolved host config."""
    register_config_schema()
    model_path = tmp_path / "model_bundle"
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "dataset.scene_ids=[scene_a,scene_b]",
                "cosmos.train.num_epochs=5",
                "cosmos.train.train_batch_per_replica=3",
                "cosmos.train.train_policy.mini_batch=3",
                "cosmos.train.train_policy.grpo_ratio_clip_low=0.1",
                "cosmos.train.train_policy.grpo_ratio_clip_high=0.3",
                "cosmos.train.train_policy.grpo_optimization_iterations=2",
                "cosmos.logging.log_training_metrics_every_n_steps=7",
                "logging_level=DEBUG",
                "cosmos.policy.parallelism.tp_size=2",
                "cosmos.rollout.parallelism.pp_size=3",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)
    write_run_artifacts(run_config)
    config_dict = yaml.safe_load(artifact_paths.resolved_config_path.read_text())
    loaded_config = load_run_config(artifact_paths.resolved_config_path)
    cosmos_config = tomllib.loads(artifact_paths.cosmos_config_path.read_text())

    assert config_dict["artifact_paths"] == {
        field: str(getattr(artifact_paths, field))
        for field in ArtifactPaths.__dataclass_fields__
    }
    assert isinstance(loaded_config, RunConfig)
    assert loaded_config.artifact_paths.topology_registry_dir == (
        artifact_paths.topology_registry_dir
    )
    assert (
        loaded_config.artifact_paths.alpasim_log_dir == artifact_paths.alpasim_log_dir
    )
    assert (
        loaded_config.artifact_paths.submit_script_path
        == artifact_paths.submit_script_path
    )
    assert artifact_paths.submit_script_path == artifact_paths.run_dir / "submit.sbatch"
    assert isinstance(
        loaded_config.execution.slurm.topology, AllInOneSlurmTopologyConfig
    )
    assert loaded_config.execution.slurm.topology.alpasim_gpus == 1
    assert loaded_config.policy.kind == "alpamayo"
    assert loaded_config.policy.model.kind == "alpamayo_r1"
    assert loaded_config.cosmos.train.train_policy.mini_batch == 3
    assert loaded_config.cosmos.train.train_policy.grpo_ratio_clip_low == 0.1
    assert loaded_config.cosmos.train.train_policy.grpo_ratio_clip_high == 0.3
    assert loaded_config.cosmos.train.train_policy.grpo_optimization_iterations == 2
    assert loaded_config.cosmos.train.num_epochs == 5
    assert loaded_config.cosmos.logging.log_training_metrics_every_n_steps == 7
    assert loaded_config.logging_level == "DEBUG"
    assert config_dict["logging_level"] == "DEBUG"
    assert loaded_config.dataset.scene_ids == ["scene_a", "scene_b"]
    assert cosmos_config["mode"] == "colocated"
    assert cosmos_config["rollout"]["backend"] == "alpagym_rollout"
    assert cosmos_config["train"]["train_policy"]["trainer_type"] == "alpagym_grpo"
    assert cosmos_config["train"]["train_policy"]["mini_batch"] == 3
    assert cosmos_config["train"]["train_policy"]["epsilon_low"] == 0.1
    assert cosmos_config["train"]["train_policy"]["epsilon_high"] == 0.3
    assert cosmos_config["train"]["train_policy"]["mu_iterations"] == 2
    assert "grpo_ratio_clip_low" not in cosmos_config["train"]["train_policy"]
    assert "grpo_ratio_clip_high" not in cosmos_config["train"]["train_policy"]
    assert "grpo_optimization_iterations" not in cosmos_config["train"]["train_policy"]
    assert cosmos_config["train"]["train_batch_per_replica"] == 3
    assert cosmos_config["train"]["epoch"] == 5
    assert "seed" not in cosmos_config["train"]
    assert cosmos_config["logging"]["log_interval"] == 7
    assert cosmos_config["policy"]["model_name_or_path"] == model_path.as_posix()
    assert cosmos_config["policy"]["parallelism"]["tp_size"] == 2
    assert cosmos_config["rollout"]["parallelism"]["pp_size"] == 3
    assert "n_init_replicas" not in cosmos_config["policy"]["parallelism"]
    assert "n_init_replicas" not in cosmos_config["rollout"]["parallelism"]
    assert cosmos_config["custom"] == {
        "resolved_config_path": str(artifact_paths.resolved_config_path),
    }


def test_host_writes_alpagym_ppo_trainer_config(
    tmp_path: Path,
) -> None:
    """Host artifacts can select the actor-critic PPO trainer and custom value knobs."""
    register_config_schema()
    model_path = tmp_path / "model_bundle"
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "cosmos.train.train_policy.trainer_type=alpagym_ppo",
                "cosmos.train.train_policy.ppo_value_loss_coef=0.75",
                "cosmos.train.train_policy.ppo_value_clip_range=0.2",
                "cosmos.train.train_policy.ppo_normalize_advantages=false",
                "cosmos.train.train_policy.ppo_gamma=0.97",
                "cosmos.train.train_policy.ppo_gae_lambda=0.9",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)
    write_run_artifacts(run_config)
    cosmos_config = tomllib.loads(artifact_paths.cosmos_config_path.read_text())

    train_policy = cosmos_config["train"]["train_policy"]
    assert train_policy["trainer_type"] == "alpagym_ppo"
    assert "ppo_value_loss_coef" not in train_policy
    assert "ppo_value_clip_range" not in train_policy
    assert "ppo_normalize_advantages" not in train_policy
    assert "ppo_gamma" not in train_policy
    assert "ppo_gae_lambda" not in train_policy
    assert cosmos_config["custom"]["ppo"] == {
        "value_loss_coef": 0.75,
        "value_clip_range": 0.2,
        "normalize_advantages": False,
        "gamma": 0.97,
        "gae_lambda": 0.9,
        "min_action_std": 0.02,
        "max_action_std": 2.0,
    }


def test_model_config_accepts_arbitrary_kind_and_round_trips_bundle_config(
    tmp_path: Path,
) -> None:
    """The public schema carries any policy kind plus opaque bundle_config knobs."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=some_future_policy",
                f"policy.model.path={(tmp_path / 'bundle').as_posix()}",
                "+policy.model.bundle_config.tokenizer=fast",
                "+policy.model.bundle_config.max_tokens=512",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)
    write_run_artifacts(run_config)
    loaded_config = load_run_config(artifact_paths.resolved_config_path)

    assert loaded_config.policy.model.kind == "some_future_policy"
    assert loaded_config.policy.model.bundle_config == {
        "tokenizer": "fast",
        "max_tokens": 512,
    }


def test_dataset_selector_rejects_scene_ids_and_test_suite(tmp_path: Path) -> None:
    """A run has one AlpaSim scene selector, not competing selectors."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                *_model_overrides(tmp_path),
                "dataset.scene_ids=[scene_a]",
                "dataset.test_suite_id=alpagym_smoke",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)

    with pytest.raises(ValueError, match="exactly one"):
        validate_run_config(run_config, requested_command="run")


def test_topology_preset_selects_separate_nodes_schema(tmp_path: Path) -> None:
    """The distributed topology preset composes the separate-node Slurm schema."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=slurm",
                "topology=slurm_distributed_1_1_1",
                f"cache_root_dir={(tmp_path / 'cache').as_posix()}",
                "execution.slurm.partition=batch",
                "execution.slurm.account=research",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    resolved_config = build_run_config(cfg, artifact_paths)

    assert "layout" not in cfg.execution.slurm
    assert "cosmos_nodes" not in cfg.execution.slurm
    assert "alpasim_nodes" not in cfg.execution.slurm
    assert "alpasim_gpus" not in cfg.execution.slurm
    assert isinstance(
        resolved_config.execution.slurm.topology, SeparateNodesSlurmTopologyConfig
    )
    assert resolved_config.execution.backend is ExecutionBackend.slurm
    assert resolved_config.execution.slurm.topology.cosmos_nodes == 2
    assert resolved_config.execution.slurm.topology.alpasim_nodes == 1


@pytest.mark.parametrize(
    "topology",
    [
        "local_colocated_1gpu",
        "local_disaggregated_2gpu",
    ],
)
def test_partial_node_topology_presets_are_nonexclusive(
    tmp_path: Path,
    topology: str,
) -> None:
    """Partial-node topology presets do not request exclusive Slurm allocation."""
    run_config = _make_run_config(
        tmp_path,
        f"topology={topology}",
    )

    assert run_config.execution.slurm.exclusive is False


def test_host_writes_disaggregated_cosmos_config_for_slurm(
    tmp_path: Path,
) -> None:
    """Slurm preserves the authored disaggregated Cosmos mode."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=slurm",
                "topology=slurm_full_node_1_3_4",
                f"cache_root_dir={(tmp_path / 'cache').as_posix()}",
                "execution.slurm.partition=batch",
                "execution.slurm.account=research",
                *_model_overrides(tmp_path),
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)
    write_run_artifacts(run_config)
    cosmos_config = tomllib.loads(artifact_paths.cosmos_config_path.read_text())

    assert cosmos_config["mode"] == "disaggregated"


def test_cosmos_config_extracts_real_model_tarball(tmp_path: Path) -> None:
    """Generated runs normalize model tarballs into the Cosmos model path."""
    register_config_schema()
    model_path = _write_hf_bundle_tarball(tmp_path, member_prefix="./")
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    run_config = load_or_create_run_config(cfg)
    artifact_paths = run_config.artifact_paths
    run_config = load_run_config(artifact_paths.resolved_config_path)
    cosmos_config = tomllib.loads(artifact_paths.cosmos_config_path.read_text())

    extracted_bundle_dir = artifact_paths.policy_model_bundle_dir
    assert cosmos_config["policy"]["model_name_or_path"] == str(extracted_bundle_dir)
    assert run_config.policy.model.path == str(extracted_bundle_dir)
    assert (
        json.loads((extracted_bundle_dir / "config.json").read_text())["model_type"]
        == "alpamayo_r1"
    )


def test_load_or_create_run_config_validates_before_writing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated host runs validate the full config before writing run artifacts."""
    register_config_schema()
    calls: list[str] = []

    def fail_validation(run_config: RunConfig, command: str) -> None:
        del run_config, command
        calls.append("validate")
        raise ValueError("validation sentinel")

    def record_write(run_config: RunConfig) -> None:
        del run_config
        calls.append("write")

    monkeypatch.setattr(host_cli, "validate_run_config", fail_validation)
    monkeypatch.setattr(host_cli, "write_run_artifacts", record_write)
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}",
            ],
        )

    with pytest.raises(ValueError, match="validation sentinel"):
        load_or_create_run_config(cfg)

    assert calls == ["validate"]
    assert not any(tmp_path.iterdir())


def test_load_prepared_run_config_accepts_extracted_hf_bundle_dir(
    tmp_path: Path,
) -> None:
    """Prepared resolved configs accept normalized HF bundle directories."""
    register_config_schema()
    model_path = _write_hf_bundle_tarball(tmp_path)
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    generated_config = load_or_create_run_config(cfg)
    artifact_paths = generated_config.artifact_paths
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        prepared_cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                f"execution.resolved_config_path={artifact_paths.resolved_config_path}",
            ],
        )

    loaded_config = load_or_create_run_config(prepared_cfg)

    assert loaded_config.policy.model.path == str(
        artifact_paths.policy_model_bundle_dir
    )


def test_load_prepared_run_config_rejects_model_tarball_path(tmp_path: Path) -> None:
    """Prepared resolved configs must not retain pre-extraction tarball paths."""
    register_config_schema()
    model_path = _write_hf_bundle_dir(tmp_path)
    tarball_path = _write_hf_bundle_tarball(tmp_path)
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    write_run_artifacts(build_run_config(cfg, artifact_paths))
    config_dict = yaml.safe_load(artifact_paths.resolved_config_path.read_text())
    config_dict["policy"]["model"]["path"] = tarball_path.as_posix()
    artifact_paths.resolved_config_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=False),
        encoding="utf-8",
    )
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        prepared_cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                f"execution.resolved_config_path={artifact_paths.resolved_config_path}",
            ],
        )

    with pytest.raises(ValueError, match="Regenerate run artifacts"):
        load_or_create_run_config(prepared_cfg)


@pytest.mark.parametrize(
    ("package_kwargs", "match"),
    [
        ({"include_config_json": False}, "config.json"),
        ({"include_weight_file": False}, "checkpoint weight"),
        (
            {
                "include_weight_file": False,
                "include_shard_index": True,
                "weight_filename": "model-00001-of-00002.safetensors",
            },
            "checkpoint weight",
        ),
    ],
)
def test_load_or_create_run_config_rejects_invalid_hf_bundle_tarball(
    tmp_path: Path,
    package_kwargs: dict[str, object],
    match: str,
) -> None:
    """Generated runs reject tarballs that do not unpack to valid HF bundles."""
    register_config_schema()
    model_path = _write_hf_bundle_tarball(tmp_path, **package_kwargs)
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    with pytest.raises(ValueError, match=match):
        load_or_create_run_config(cfg)


def test_load_or_create_run_config_accepts_hf_bundle_tarball_with_sharded_weights(
    tmp_path: Path,
) -> None:
    """Generated runs accept tarballs with standard HF shard-index checkpoint bundles."""
    register_config_schema()
    model_path = _write_hf_bundle_tarball(
        tmp_path,
        weight_filename="model-00001-of-00002.safetensors",
        include_shard_index=True,
    )
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    run_config = load_or_create_run_config(cfg)

    assert run_config.policy.model.path == str(
        run_config.artifact_paths.policy_model_bundle_dir
    )


def test_load_or_create_run_config_accepts_numbered_single_file_hf_bundle_tarball(
    tmp_path: Path,
) -> None:
    """Generated runs match the runtime loader's numbered HF weight filenames."""
    register_config_schema()
    model_path = _write_hf_bundle_tarball(
        tmp_path,
        weight_filename="model-00001-of-00001.safetensors",
    )
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
            ],
        )

    run_config = load_or_create_run_config(cfg)

    assert run_config.policy.model.path == str(
        run_config.artifact_paths.policy_model_bundle_dir
    )


@pytest.mark.parametrize(
    ("include_shard_index", "include_shard_file", "match"),
    [
        (False, False, "checkpoint weight"),
        (True, False, "checkpoint weight"),
        (True, True, None),
    ],
)
def test_run_config_validates_hf_bundle_directory_weights(
    tmp_path: Path,
    include_shard_index: bool,
    include_shard_file: bool,
    match: str | None,
) -> None:
    """Directory-backed HF bundles must expose supported complete weights."""
    model_path = _write_hf_bundle_dir(
        tmp_path,
        include_weight_file=False,
        include_shard_index=include_shard_index,
        include_shard_file=include_shard_file,
    )
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
    )

    if match is None:
        validate_run_config(run_config, "run")
    else:
        with pytest.raises(ValueError, match=match):
            validate_run_config(run_config, "run")


def test_run_config_accepts_numbered_single_file_hf_bundle_dir(tmp_path: Path) -> None:
    """Directory validation accepts numbered single-file HF checkpoints."""
    model_path = _write_hf_bundle_dir(
        tmp_path,
        weight_filename="pytorch_model-00001-of-00001.bin",
    )
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
    )

    validate_run_config(run_config, "run")


def test_training_policy_config_rejects_disabled_replay_trace(tmp_path: Path) -> None:
    """Current Cosmos trainer launches require model replay traces."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
        "policy.inference.return_trace_for_rl=false",
    )

    with pytest.raises(ValueError, match="return_trace_for_rl"):
        validate_run_config(run_config, "run")


def test_run_config_rejects_zero_force_gt_for_av_domain(tmp_path: Path) -> None:
    """AV runs still require a positive force-GT warmup."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
        "alpasim.wizard_args.force_gt_duration_us=0",
    )

    with pytest.raises(ValueError, match="force_gt_duration_us must be positive"):
        validate_run_config(run_config, "run")


def test_run_config_allows_zero_force_gt_for_humanoid_domain(tmp_path: Path) -> None:
    """Humanoid dynamics-only rollouts do not use AV force-GT warmup."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
        "alpasim.simulation_domain=humanoid",
        "dataset.scene_ids=[stairs]",
        "alpasim.wizard_args.force_gt_duration_us=0",
        "cosmos.rollout.prefetch_rollout=false",
    )
    run_config.alpasim.humanoid = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
    )

    validate_run_config(run_config, "run")


def test_humanoid_config_rejects_vector_env_until_lane_local_gae_exists() -> None:
    with pytest.raises(ValueError, match="num_envs must be 1"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            num_envs=2,
        )


@pytest.mark.parametrize("rollout_seed_base", [-1, 1 << 64, True, 1.5])
def test_humanoid_config_rejects_non_uint64_rollout_seed_base(
    rollout_seed_base: object,
) -> None:
    """The optional deterministic panel base matches the uint64 proto ABI."""
    with pytest.raises(ValueError, match="rollout_seed_base must be a uint64"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            rollout_seed_base=rollout_seed_base,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("rollout_seed_base", [0, (1 << 64) - 1])
def test_humanoid_config_accepts_uint64_rollout_seed_boundaries(
    rollout_seed_base: int,
) -> None:
    config = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
        rollout_seed_base=rollout_seed_base,
    )

    assert config.rollout_seed_base == rollout_seed_base


@pytest.mark.parametrize(
    ("execution_profile", "reward_profile_id", "grail_root_path", "match"),
    [
        (
            HumanoidExecutionProfile.direct_action,
            "reference_route_centered.v2",
            None,
            "direct_action requires",
        ),
        (
            HumanoidExecutionProfile.direct_action,
            "videomimic_v9_formula_grail_target_equivalent.v1",
            None,
            "direct_action requires",
        ),
        (
            HumanoidExecutionProfile.motion_reference,
            "reference_route_centered.v4",
            "/tmp/GRAIL",
            "reward_profile_id must be one of",
        ),
    ],
)
def test_humanoid_config_rejects_reward_profiles_outside_execution_abi(
    execution_profile: HumanoidExecutionProfile,
    reward_profile_id: str,
    grail_root_path: str | None,
    match: str,
) -> None:
    """Each humanoid execution ABI accepts only its known reward profiles."""
    with pytest.raises(ValueError, match=match):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=execution_profile,
            grail_root_path=grail_root_path,
            reward_profile_id=reward_profile_id,
        )


def test_motion_reference_freeze_injects_single_source_paths_and_fingerprint(
    tmp_path: Path,
) -> None:
    model_path = _write_h70_planner_bundle_dir(tmp_path)
    scene_store = tmp_path / "scene_store"
    scene_root = scene_store / "scenes" / "stairs"
    scene_root.mkdir(parents=True)
    digest = "a" * 64
    (scene_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": "stairs",
                "identity": {"scene_content_sha256": digest},
            }
        ),
        encoding="utf-8",
    )
    run_config = _make_run_config(
        tmp_path,
        "alpasim.simulation_domain=humanoid",
        "dataset.scene_ids=[stairs]",
        "alpasim.wizard_args.force_gt_duration_us=0",
        "alpasim.wizard_args.control_timestep_us=500000",
        "cosmos.rollout.prefetch_rollout=false",
    )
    run_config.alpasim.humanoid = HumanoidAlpaSimConfig(
        repo_path="/tmp/humanoid-support",
        scene_store_path=str(scene_store),
        scenario_ids_by_scene={"stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/tmp/GRAIL",
        reward_profile_id="reference_route_centered.v1",
    )
    run_config = replace(
        run_config,
        policy=replace(
            run_config.policy,
            model=replace(
                run_config.policy.model,
                kind="g1_videomimic_planner",
                path=str(model_path),
                bundle_config={"planner_mode": "shadow_rollout"},
                step_dt_us=500_000,
            ),
        ),
    )

    frozen = freeze_humanoid_scene_fingerprints(run_config)

    assert frozen.alpasim.humanoid is not None
    assert frozen.alpasim.humanoid.expected_scene_fingerprints == {"stairs": digest}
    assert frozen.policy.model.bundle_config["humanoid_repo_path"] == (
        "/tmp/humanoid-support"
    )
    assert frozen.policy.model.bundle_config["scene_store_path"] == str(
        scene_store.resolve()
    )
    assert frozen.policy.model.bundle_config["expected_scene_fingerprints_json"] == (
        '{"stairs":"' + digest + '"}'
    )


@pytest.mark.parametrize(
    ("reward_profile_override", "expected_reward_profile_id"),
    [
        (None, "reference_route_centered.v3"),
        (
            "alpasim.humanoid.reward_profile_id=reference_route_centered.v2",
            "reference_route_centered.v2",
        ),
        (
            "alpasim.humanoid.reward_profile_id=reference_route_centered.v3",
            "reference_route_centered.v3",
        ),
    ],
)
def test_motion_reference_experiment_resolves_seed_panel_and_horizon(
    tmp_path: Path,
    reward_profile_override: str | None,
    expected_reward_profile_id: str,
) -> None:
    """The preset resolves its horizon, panel seed, and opt-in reward profile."""
    model_path = _write_h70_planner_bundle_dir(tmp_path)
    scene_root = tmp_path / "scene_store" / "scenes" / "hq_stairs"
    scene_root.mkdir(parents=True)
    (scene_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": "hq_stairs",
                "identity": {"scene_content_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                "experiment=g1_videomimic_planner_hq_stairs_current_policy",
                f"run_root={tmp_path.as_posix()}",
                f"policy.model.path={model_path.as_posix()}",
                f"alpasim.repo_path={(tmp_path / 'alpasim').as_posix()}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
                f"alpasim.humanoid.repo_path={(tmp_path / 'humanoid').as_posix()}",
                "alpasim.humanoid.scene_store_path="
                f"{(tmp_path / 'scene_store').as_posix()}",
                f"alpasim.humanoid.grail_root_path={(tmp_path / 'GRAIL').as_posix()}",
                "alpasim.humanoid.rollout_seed_base=9000",
                *([reward_profile_override] if reward_profile_override else []),
            ],
        )
    run_config = freeze_humanoid_scene_fingerprints(
        build_run_config(cfg, build_artifact_paths(cfg))
    )

    validate_run_config(run_config, "run")
    write_run_artifacts(run_config)
    resolved_config = yaml.safe_load(
        run_config.artifact_paths.resolved_config_path.read_text(encoding="utf-8")
    )

    assert (
        run_config.alpasim.wizard_args.n_sim_steps
        == run_config.expected_valid_steps
        == 60
    )
    assert run_config.alpasim.wizard_args.control_timestep_us == 500_000
    assert run_config.cosmos.train.optm_lr == pytest.approx(1.0e-6)
    assert run_config.cosmos.train.train_policy.step_mini_batch == 60
    assert run_config.cosmos.train.train_policy.grpo_optimization_iterations == 1
    assert run_config.cosmos.train.train_policy.kl_beta == pytest.approx(0.1)
    assert run_config.cosmos.train.train_policy.reference_reset_interval == 0
    assert run_config.cosmos.train.train_policy.ppo_gamma == pytest.approx(0.99)
    assert run_config.cosmos.train.train_policy.ppo_gae_lambda == pytest.approx(0.95)
    assert run_config.cosmos.train.train_policy.ppo_min_action_std == pytest.approx(
        0.05
    )
    assert run_config.cosmos.train.train_policy.ppo_max_action_std == pytest.approx(
        0.15
    )
    assert run_config.alpasim.humanoid is not None
    assert run_config.alpasim.humanoid.rollout_seed_base == 9000
    assert resolved_config["alpasim"]["humanoid"]["rollout_seed_base"] == 9000
    assert run_config.alpasim.humanoid.reward_profile_id == expected_reward_profile_id
    assert (
        resolved_config["alpasim"]["humanoid"]["reward_profile_id"]
        == expected_reward_profile_id
    )


def test_humanoid_config_rejects_unqualified_distributed_async_mode(
    tmp_path: Path,
) -> None:
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
        "alpasim.simulation_domain=humanoid",
        "dataset.scene_ids=[stairs]",
        "alpasim.wizard_args.force_gt_duration_us=0",
        "cosmos.rollout.prefetch_rollout=false",
    )
    run_config.alpasim.humanoid = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
    )
    run_config.cosmos.mode = CosmosRLMode.disaggregated

    with pytest.raises(ValueError, match="colocated only"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("expected_valid_steps=0", "expected_valid_steps"),
        ("policy.model.num_context_frames=0", "num_context_frames"),
    ],
)
def test_training_policy_config_rejects_nonpositive_replay_shape(
    tmp_path: Path,
    override: str,
    match: str,
) -> None:
    """Replay-shape config validation happens before Cosmos workers start."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={model_path.as_posix()}",
        override,
    )

    with pytest.raises(ValueError, match=match):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            [
                "cosmos.train.train_batch_per_replica=6",
                "cosmos.train.train_policy.mini_batch=4",
            ],
            "mini_batch",
        ),
        (
            [
                "cosmos.train.train_batch_per_replica=6",
                "cosmos.train.train_policy.mini_batch=2",
                "cosmos.policy.parallelism.dp_shard_size=4",
            ],
            "dp_shard_size",
        ),
    ],
)
def test_cosmos_config_rejects_invalid_grpo_batch_geometry(
    tmp_path: Path,
    overrides: list[str],
    match: str,
) -> None:
    """GRPO launches reject incompatible batch geometry."""
    run_config = _make_run_config(tmp_path, *overrides)

    with pytest.raises(ValueError, match=match):
        validate_run_config(run_config, "run")


def test_cosmos_config_accepts_non_group_aligned_train_batch(tmp_path: Path) -> None:
    """GRPO launches accept train batches that do not align to rollout groups."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        "cosmos.train.train_batch_per_replica=3",
        "cosmos.train.train_policy.mini_batch=1",
        "cosmos.policy.parallelism.dp_shard_size=1",
        "cosmos.rollout.n_generation=2",
    )

    validate_run_config(run_config, "run")


def test_cosmos_config_accepts_grpo_batch_geometry(tmp_path: Path) -> None:
    """GRPO launches accept compatible batch geometry."""
    register_config_schema()
    model_path = _write_hf_bundle_dir(tmp_path)
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                "cosmos.train.train_batch_per_replica=8",
                "cosmos.train.train_policy.mini_batch=2",
                "cosmos.policy.parallelism.dp_shard_size=2",
                "cosmos.rollout.n_generation=2",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    run_config = build_run_config(cfg, artifact_paths)

    validate_run_config(run_config, "run")

    assert run_config.cosmos.train.train_batch_per_replica == 8


def test_slurm_cosmos_capacity_rejects_replicas_that_do_not_fit(
    tmp_path: Path,
) -> None:
    """Slurm configs reject Cosmos replica plans that exceed visible Cosmos GPUs."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        "topology=slurm_full_node_1_3_4",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
        "cosmos.launch.policy_replicas=4",
        "cosmos.launch.rollout_replicas=1",
    )

    with pytest.raises(ValueError, match="Cosmos Slurm GPU capacity"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize(
    "topology",
    [
        "slurm_distributed_1_1_1",
        "slurm_distributed_1_2_2",
        "slurm_distributed_shared_cosmos_2_3",
    ],
)
def test_slurm_distributed_presets_fit_cosmos_capacity(
    tmp_path: Path,
    topology: str,
) -> None:
    """Distributed Slurm presets keep Cosmos replicas within the Cosmos GPU pool."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        f"topology={topology}",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
    )

    validate_run_config(run_config, "run")


def test_slurm_distributed_shared_cosmos_2_3_preset_keeps_coupled_run_shape(
    tmp_path: Path,
) -> None:
    """The 2-Cosmos / 3-AlpaSim topology keeps enough rollout depth per AlpaSim node."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        "topology=slurm_distributed_shared_cosmos_2_3",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
    )

    validate_run_config(run_config, "run")

    assert isinstance(
        run_config.execution.slurm.topology, SeparateNodesSlurmTopologyConfig
    )
    topology = run_config.execution.slurm.topology
    assert (
        run_config.execution.slurm.nodes
        == topology.cosmos_nodes + topology.alpasim_nodes
    )
    assert topology.cosmos_nodes == 2
    assert topology.alpasim_nodes == 3
    assert run_config.transport.kind is TransportKind.disk
    assert (
        "runtime.simulation_config.n_sim_steps=30"
        in run_config.alpasim.wizard_args.extra_overrides
    )
    assert (
        "runtime.simulation_config.force_gt_duration_us=1600000"
        in run_config.alpasim.wizard_args.extra_overrides
    )
    rollout_depth = (
        run_config.cosmos.launch.rollout_replicas
        * run_config.cosmos.rollout.batch_size
        * run_config.cosmos.rollout.n_generation
    )
    assert rollout_depth == 48
    assert rollout_depth // topology.alpasim_nodes == 16
    assert run_config.cosmos.train.train_batch_per_replica == (
        run_config.cosmos.launch.rollout_replicas
    )
    assert run_config.cosmos.train.train_policy.mini_batch == 1
    assert run_config.expected_valid_steps == 22


def test_slurm_distributed_shared_cosmos_2_3_rejects_later_run_shape_resets(
    tmp_path: Path,
) -> None:
    """Validation catches demand presets that override the topology-owned geometry."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        "topology=slurm_distributed_shared_cosmos_2_3",
        "run=public_2507_1epoch",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
    )

    with pytest.raises(ValueError, match="slurm_distributed_shared_cosmos_2_3"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize(
    "topology",
    [
        "slurm_distributed_1_1_1",
        "slurm_distributed_1_2_2",
    ],
)
def test_validate_run_config_accepts_nccl_on_multi_cosmos_host_topology(
    tmp_path: Path,
    topology: str,
) -> None:
    """The routable rendezvous lets NCCL span Cosmos hosts, so distributed presets validate."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        f"topology={topology}",
        "transport=nccl",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
    )

    validate_run_config(run_config, "run")


def test_run_config_validates_slurm_topology_shape_before_slurm_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validation owns authored Slurm topology constraints."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
        "topology=slurm_full_node_1_3_4",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
        "execution.slurm.nodes=2",
    )
    monkeypatch.setattr(
        "alpagym_host.config_validation.validate_slurm_config",
        lambda execution: None,
    )
    monkeypatch.setattr(
        "alpagym_host.config_validation.build_slurm_topology",
        lambda **kwargs: SimpleNamespace(
            cosmos_host_plans=(SimpleNamespace(cosmos_gpu_count=8),),
        ),
    )

    with pytest.raises(ValueError, match="all_in_one requires nodes=1"):
        validate_run_config(run_config, "run")


def test_run_config_rejects_relative_slurm_cache_before_creating_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null cache_root_dir resolves uv_cache_dir to a relative 'None/uv'; validation must
    reject it before any Slurm helper mkdirs a stray directory under the cwd.
    """
    monkeypatch.chdir(tmp_path)
    _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        "topology=slurm_full_node_1_3_4",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
        "cache_root_dir=null",
    )
    assert run_config.execution.slurm.uv_cache_dir == "None/uv"

    with pytest.raises(ValueError, match="must be an absolute path"):
        validate_run_config(run_config, "run")
    assert not (tmp_path / "None").exists()


def _make_run_config(tmp_path: Path, *overrides: str) -> RunConfig:
    """Build a resolved config for validation-only host tests.

    Slurm topology overrides pair with deploy=slurm and an
    absolute cache_root_dir so the derived uv_cache_dir/checkout_cache_dir pass
    the Slurm-branch absolute-path and writability checks. Local topologies pair
    with deploy=local.
    """
    register_config_schema()
    requests_slurm = any(
        override.startswith("topology=slurm") for override in overrides
    )
    if requests_slurm:
        base_overrides = [
            f"run_root={tmp_path.as_posix()}",
            "deploy=slurm",
            f"cache_root_dir={(tmp_path / 'cache').as_posix()}",
            "execution.slurm.partition=batch",
            "execution.slurm.account=research",
        ]
        if not any(
            override.startswith("execution.slurm.container_mounts")
            for override in overrides
        ):
            # Identity-mount the tmp run dir so the Slurm host-path mount check accepts the
            # tmp_path run_root and model bundle, alongside the required uv-cache mount.
            base_overrides.append(
                "execution.slurm.container_mounts=["
                '"${execution.slurm.uv_cache_dir}:${execution.slurm.uv_cache_dir}",'
                f'"{tmp_path.as_posix()}:{tmp_path.as_posix()}"]'
            )
    else:
        base_overrides = [f"run_root={tmp_path.as_posix()}", "deploy=local"]
        if not any(override.startswith("topology=") for override in overrides):
            base_overrides.append("topology=local_colocated_1gpu")
    if not any(override.startswith("policy.model.kind=") for override in overrides):
        base_overrides.append("policy.model.kind=alpamayo_r1")
    if not any(override.startswith("policy.model.path=") for override in overrides):
        base_overrides.append(
            f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}"
        )
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[*base_overrides, *overrides],
        )
    artifact_paths = build_artifact_paths(cfg)
    return build_run_config(cfg, artifact_paths)


def _model_overrides(tmp_path: Path) -> list[str]:
    """Return policy overrides for tests that do not validate model files."""
    return [
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}",
    ]


def _write_h70_planner_bundle_dir(tmp_path: Path) -> Path:
    """Create a minimal provenance-valid current-policy H70 actor bundle."""

    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir()
    weights_path = bundle_dir / "model.safetensors"
    save_file(
        {
            "actor.0.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "std": torch.tensor([0.15], dtype=torch.float32),
            "critic.0.weight": torch.tensor([[2.0]], dtype=torch.float32),
        },
        str(weights_path),
    )
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    initialization = {
        "schema": "videomimic_v9_actor_initialization.v1",
        "source_v9_checkpoint_sha256": "b" * 64,
        "source_v9_actor_parity_samples": 16,
        "source_v9_actor_parity_max_abs": 0.0,
        "planner_critic_seed": 0,
    }
    initialization_sha256 = hashlib.sha256(
        json.dumps(initialization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lineage = {
        "schema": "videomimic_planner_actor_update_lineage.v1",
        "current_actor_status": "v9_initialized_unmodified",
        "initialization_attestation_sha256": initialization_sha256,
        "current_model_weights_sha256": weights_sha256,
        "parent_model_weights_sha256": None,
        "parent_update_lineage_sha256": None,
        "training_export_step": 0,
    }
    config = {
        "model_type": "g1_videomimic_planner_actor_critic",
        "shadow_action_steps": 69,
        "checkpoint_path": "model.safetensors",
        **{
            key: initialization[key]
            for key in (
                "source_v9_checkpoint_sha256",
                "source_v9_actor_parity_samples",
                "source_v9_actor_parity_max_abs",
                "planner_critic_seed",
            )
        },
        "actor_initialization_attestation": initialization,
        "actor_update_lineage": lineage,
    }
    (bundle_dir / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return bundle_dir


def _write_hf_bundle_dir(
    tmp_path: Path,
    include_weight_file: bool = True,
    include_shard_index: bool = False,
    include_shard_file: bool = True,
    weight_filename: str = "model.safetensors",
) -> Path:
    """Create a minimal local HF bundle directory."""
    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        json.dumps({"model_type": "alpamayo_r1"}), encoding="utf-8"
    )
    if include_weight_file:
        (bundle_dir / weight_filename).write_text("weights", encoding="utf-8")
    if include_shard_index:
        shard_filename = "model-00001-of-00002.safetensors"
        (bundle_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.weight": shard_filename}}),
            encoding="utf-8",
        )
        if include_shard_file:
            (bundle_dir / shard_filename).write_text("weights", encoding="utf-8")
    return bundle_dir


def _write_hf_bundle_tarball(
    tmp_path: Path,
    include_config_json: bool = True,
    include_weight_file: bool = True,
    include_shard_index: bool = False,
    weight_filename: str = "model.safetensors",
    member_prefix: str = "",
    weights_text: str = "weights",
) -> Path:
    """Create a tarball containing a minimal HF bundle."""
    source_dir = tmp_path / "hf_bundle_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    if include_config_json:
        (source_dir / "config.json").write_text(
            json.dumps({"model_type": "alpamayo_r1"}),
            encoding="utf-8",
        )
        (source_dir / "weights.bin").write_text(weights_text, encoding="utf-8")
        if include_weight_file:
            (source_dir / weight_filename).write_text(weights_text, encoding="utf-8")
        if include_shard_index:
            (source_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": weight_filename}}),
                encoding="utf-8",
            )
    tarball_path = tmp_path / "model.tar"
    with tarfile.open(tarball_path, "w") as tar:
        if include_config_json:
            tar.add(source_dir / "config.json", arcname=f"{member_prefix}config.json")
            tar.add(source_dir / "weights.bin", arcname=f"{member_prefix}weights.bin")
            if include_weight_file:
                tar.add(
                    source_dir / weight_filename,
                    arcname=f"{member_prefix}{weight_filename}",
                )
            if include_shard_index:
                tar.add(
                    source_dir / "model.safetensors.index.json",
                    arcname=f"{member_prefix}model.safetensors.index.json",
                )
    return tarball_path
