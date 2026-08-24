# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import tarfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import alpagym_host.cli as host_cli
import pytest
import yaml
from alpagym_host.cli import load_or_create_run_config
from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ArtifactPaths,
    CosmosRLMode,
    ExecutionBackend,
    HumanoidAlpaSimConfig,
    HumanoidExecutionProfile,
    HumanoidPolicyCameraProfile,
    HumanoidReferenceControllerProfile,
    ProvenanceMode,
    RunConfig,
    SeparateNodesSlurmTopologyConfig,
    TransportKind,
    load_run_config,
    register_config_schema,
)
from alpagym_host.config_validation import (
    _validate_humanoid_config,
    _validate_provenance_config,
    _validate_vla_slurm_worker_mounts,
    _validate_wizard_startup_config,
    validate_run_config,
)
from alpagym_host.humanoid_scene_identity import freeze_humanoid_scene_fingerprints
from alpagym_host.run_artifacts import (
    build_artifact_paths,
    build_run_config,
    write_run_artifacts,
)
from hydra import compose, initialize_config_module


def test_required_provenance_accepts_only_resolvable_local_humanoid_repos(
    tmp_path: Path,
) -> None:
    """Formal provenance must own concrete local AlpaSim and Humanoid worktrees."""
    alpasim_root = tmp_path / "alpasim"
    humanoid_root = tmp_path / "humanoid"
    alpasim_root.mkdir()
    humanoid_root.mkdir()
    config = _make_run_config(tmp_path)
    config.execution.provenance_mode = ProvenanceMode.required
    config.alpasim.repo_path = str(alpasim_root.resolve())
    config.alpasim.humanoid = HumanoidAlpaSimConfig(
        repo_path=str(humanoid_root.resolve()),
        scene_store_path=str(tmp_path / "scenes"),
        scenario_ids_by_scene={"stairs": "ascend"},
    )

    _validate_provenance_config(config=config, requested_command="run")

    _validate_provenance_config(config=config, requested_command="rollout")

    config.cosmos.train.ckpt.save_mode = "async"
    with pytest.raises(ValueError, match="synchronous native checkpoint"):
        _validate_provenance_config(config=config, requested_command="run")
    _validate_provenance_config(config=config, requested_command="rollout")
    config.cosmos.train.ckpt.save_mode = "sync"

    config.execution.backend = ExecutionBackend.slurm
    with pytest.raises(ValueError, match="only supports local_process"):
        _validate_provenance_config(config=config, requested_command="run")

    config.execution.backend = ExecutionBackend.local_process
    with pytest.raises(ValueError, match="command=run or command=rollout only"):
        _validate_provenance_config(config=config, requested_command="submit")


def test_hq_stairs_experiment_requires_formal_provenance() -> None:
    """The formal staircase training profile cannot silently disable evidence."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        config = compose(
            config_name="default",
            overrides=[
                "experiment=g1_vla_hq_stairs_local_1gpu",
                "policy.model.path=/tmp/model",
                "alpasim.humanoid.repo_path=/tmp/humanoid",
                "alpasim.humanoid.scene_store_path=/tmp/scene_store",
                "alpasim.humanoid.grail_root_path=/tmp/grail",
                "alpasim.humanoid.visual_controller_release_path=/tmp/controller",
                "alpasim.humanoid.scene_cache_path=/tmp/cache",
                "alpasim.humanoid.runtime_cache_path=/tmp/runtime-cache",
            ],
        )

    assert config.execution.provenance_mode is ProvenanceMode.required


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
                "cosmos.train.seed=12345",
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
    assert cosmos_config["train"]["sync_weight_interval"] == 1
    assert cosmos_config["train"]["ckpt"]["save_mode"] == "sync"
    assert cosmos_config["train"]["epoch"] == 5
    assert cosmos_config["train"]["seed"] == 12345
    assert cosmos_config["train"]["deterministic"] is False
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
        "behavior_kl_backtrack": False,
        "behavior_kl_backtrack_margin": 0.9,
        "behavior_kl_backtrack_max_attempts": 4,
    }


def test_host_writes_vla_flow_ppo_config(tmp_path: Path) -> None:
    """Host serializes the exact Flow-PPO optimizer and loss configuration."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}",
                "cosmos.train.optm_part_lrs=[5e-6,1e-4]",
                "cosmos.train.epsilon=1e-8",
                "cosmos.train.optm_weight_decay=0.01",
                "cosmos.train.optm_betas=[0.9,0.999]",
                "cosmos.train.optm_grad_norm_clip=1.0",
                "cosmos.train.optm_warmup_steps=0",
                "cosmos.train.train_policy.trainer_type=alpagym_flow_ppo",
                "cosmos.train.train_policy.grpo_ratio_clip_low=0.2",
                "cosmos.train.train_policy.grpo_ratio_clip_high=0.28",
                "cosmos.train.train_policy.ppo_value_loss_coef=1.0",
                "cosmos.train.train_policy.ppo_value_clip_range=0.2",
                "cosmos.train.train_policy.ppo_dual_clip_ratio=3.0",
                "cosmos.train.train_policy.ppo_value_huber_delta=10.0",
                "cosmos.train.train_policy.ppo_normalize_advantages=true",
                "cosmos.train.train_policy.ppo_gamma=0.99",
                "cosmos.train.train_policy.ppo_gae_lambda=0.95",
                "cosmos.train.train_policy.ppo_target_behavior_kl=0.05",
                "cosmos.train.train_policy.ppo_behavior_kl_backtrack=true",
                "cosmos.train.train_policy.ppo_behavior_kl_backtrack_margin=0.8",
                "cosmos.train.train_policy.ppo_behavior_kl_backtrack_max_attempts=3",
                "cosmos.train.train_policy.kl_beta=0.0",
            ],
        )

    artifact_paths = build_artifact_paths(cfg)
    write_run_artifacts(build_run_config(cfg, artifact_paths))
    cosmos_config = tomllib.loads(artifact_paths.cosmos_config_path.read_text())

    train_policy = cosmos_config["train"]["train_policy"]
    assert train_policy["trainer_type"] == "alpagym_flow_ppo"
    assert train_policy["epsilon_low"] == 0.2
    assert train_policy["epsilon_high"] == 0.28
    assert train_policy["kl_beta"] == 0.0
    train = cosmos_config["train"]
    assert train["optm_lr"] == [5.0e-6, 1.0e-4]
    assert train["epsilon"] == 1.0e-8
    assert train["optm_weight_decay"] == 0.01
    assert train["optm_betas"] == [0.9, 0.999]
    assert train["optm_grad_norm_clip"] == 1.0
    assert train["optm_warmup_steps"] == 0
    assert cosmos_config["custom"]["ppo"] == {
        "value_loss_coef": 1.0,
        "value_clip_range": 0.2,
        "dual_clip_ratio": 3.0,
        "value_huber_delta": 10.0,
        "normalize_advantages": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "min_action_std": 0.02,
        "max_action_std": 2.0,
        "target_behavior_kl": 0.05,
        "behavior_kl_backtrack": True,
        "behavior_kl_backtrack_margin": 0.8,
        "behavior_kl_backtrack_max_attempts": 3,
    }


@pytest.mark.parametrize(
    "target_behavior_kl",
    (0.0, -0.1, float("nan"), float("inf")),
)
def test_training_policy_config_rejects_invalid_target_behavior_kl(
    tmp_path: Path,
    target_behavior_kl: float,
) -> None:
    """Behavior-policy KL guards must use a finite positive threshold."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.ppo_target_behavior_kl = target_behavior_kl

    with pytest.raises(ValueError, match="ppo_target_behavior_kl"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize("interval", (0, -1, True, 1.5))
def test_training_rejects_invalid_sync_weight_interval(
    tmp_path: Path,
    interval: object,
) -> None:
    """The generated Cosmos weight-sync interval must be an explicit integer."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.sync_weight_interval = interval

    with pytest.raises(ValueError, match="sync_weight_interval"):
        validate_run_config(run_config, "run")


def test_on_policy_training_requires_every_step_weight_sync(tmp_path: Path) -> None:
    """A fresh on-policy rollout must consume the latest published lease."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.on_policy = True
    run_config.cosmos.train.sync_weight_interval = 2

    with pytest.raises(ValueError, match="requires.*sync_weight_interval == 1"):
        validate_run_config(run_config, "run")


def test_training_policy_backtrack_requires_target_behavior_kl(tmp_path: Path) -> None:
    """Actor-step retries require an explicit behavior-policy KL target."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.ppo_behavior_kl_backtrack = True

    with pytest.raises(ValueError, match="requires ppo_target_behavior_kl"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize("iterations", (0, 2, True, 1.0))
def test_training_policy_backtrack_requires_single_optimizer_iteration(
    tmp_path: Path,
    iterations: object,
) -> None:
    """Whole-step actor interpolation is valid only for one optimizer update."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    train_policy = run_config.cosmos.train.train_policy
    train_policy.ppo_target_behavior_kl = 0.003
    train_policy.ppo_behavior_kl_backtrack = True
    train_policy.grpo_optimization_iterations = iterations

    with pytest.raises(
        ValueError,
        match="requires grpo_optimization_iterations == 1",
    ):
        validate_run_config(run_config, "run")


def test_training_policy_backtrack_flag_must_be_boolean(tmp_path: Path) -> None:
    """Backtracking cannot be activated by a truthy non-boolean value."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.ppo_behavior_kl_backtrack = "true"

    with pytest.raises(ValueError, match="ppo_behavior_kl_backtrack must be a boolean"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize(
    "margin",
    (0.0, 1.0, -0.1, float("nan"), float("inf"), True, "0.9"),
)
def test_training_policy_rejects_invalid_behavior_kl_backtrack_margin(
    tmp_path: Path,
    margin: object,
) -> None:
    """The retry acceptance margin must be finite and strictly inside (0, 1)."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.ppo_behavior_kl_backtrack_margin = margin

    with pytest.raises(ValueError, match="ppo_behavior_kl_backtrack_margin"):
        validate_run_config(run_config, "run")


@pytest.mark.parametrize("attempts", (0, -1, 1.5, True))
def test_training_policy_rejects_invalid_behavior_kl_backtrack_attempts(
    tmp_path: Path,
    attempts: object,
) -> None:
    """Retry count is a strict positive integer, not a coercible scalar."""
    model_path = _write_hf_bundle_dir(tmp_path)
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_path.as_posix()}",
    )
    run_config.cosmos.train.train_policy.ppo_behavior_kl_backtrack_max_attempts = (
        attempts
    )

    with pytest.raises(ValueError, match="ppo_behavior_kl_backtrack_max_attempts"):
        validate_run_config(run_config, "run")


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


@pytest.mark.parametrize(
    ("x", "y", "yaw", "match"),
    (
        (1.0, None, 0.0, "requires root x, y, and yaw"),
        (True, 2.0, 0.0, "must be finite"),
        (10_001.0, 2.0, 0.0, "within 10 km"),
        (1.0, 2.0, 3.2, "yaw must be"),
    ),
)
def test_humanoid_runtime_spawn_override_is_atomic_and_bounded(
    x: object,
    y: object,
    yaw: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
            reward_profile_id="direct_v9_shaped.v1",
            runtime_spawn_root_x_m=x,  # type: ignore[arg-type]
            runtime_spawn_root_y_m=y,  # type: ignore[arg-type]
            runtime_spawn_root_yaw_rad=yaw,  # type: ignore[arg-type]
        )


def test_humanoid_runtime_spawn_override_rejects_direct_action() -> None:
    with pytest.raises(ValueError, match="requires motion_reference"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            runtime_spawn_root_x_m=1.0,
            runtime_spawn_root_y_m=2.0,
            runtime_spawn_root_yaw_rad=0.0,
        )


def test_humanoid_policy_camera_requires_scene_cache() -> None:
    with pytest.raises(ValueError, match="scene_cache_path is required"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
            policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
            service_image="alpasim-humanoid-nurec:local",
            reward_profile_id="reference_route_centered.v3",
        )


def test_humanoid_policy_camera_requires_disjoint_runtime_cache() -> None:
    with pytest.raises(ValueError, match="runtime_cache_path is required"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scene_cache_path="/tmp/humanoid-cache",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
            policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
            service_image="alpasim-humanoid-nurec:local",
            reward_profile_id="reference_route_centered.v3",
        )


@pytest.mark.parametrize(
    ("scene_cache_path", "runtime_cache_path", "match"),
    (
        ("/tmp/humanoid-cache", "relative-cache", "must be absolute"),
        ("/tmp/humanoid-cache", "/tmp/humanoid-cache", "scene_cache_path"),
        (
            "/tmp/humanoid-cache",
            "/tmp/alpasim-humanoid/native-cache",
            "repo_path",
        ),
        (
            "/tmp/humanoid-scenes/render-cache",
            "/tmp/humanoid-runtime-cache",
            "scene_store_path",
        ),
        (
            "/tmp/humanoid-cache",
            "/tmp/visual-sonic-release/native-cache",
            "visual_controller_release_path",
        ),
    ),
)
def test_managed_visual_cache_roots_are_disjoint(
    scene_cache_path: str,
    runtime_cache_path: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _visual_sonic_humanoid_config(
            scene_cache_path=scene_cache_path,
            runtime_cache_path=runtime_cache_path,
        )


def _visual_sonic_humanoid_config(**overrides: object) -> HumanoidAlpaSimConfig:
    values: dict[str, object] = {
        "repo_path": "/tmp/alpasim-humanoid",
        "scene_store_path": "/tmp/humanoid-scenes",
        "scene_cache_path": "/tmp/humanoid-cache",
        "runtime_cache_path": "/tmp/humanoid-runtime-cache",
        "scenario_ids_by_scene": {"stairs": "ascend"},
        "execution_profile": HumanoidExecutionProfile.motion_reference,
        "reference_controller_profile": (
            HumanoidReferenceControllerProfile.sonic_visual
        ),
        "visual_controller_release_path": "/tmp/visual-sonic-release",
        "robot_physics_profile": (
            "sonic.isaac_training.g1_cylinder_model_12.mujoco_port.v1"
        ),
        "service_image": "alpasim-humanoid-nurec:local",
        "reward_profile_id": "direct_v9_shaped.v1",
    }
    values.update(overrides)
    return HumanoidAlpaSimConfig(**values)  # type: ignore[arg-type]


def test_visual_sonic_requires_explicit_robot_physics_profile() -> None:
    with pytest.raises(ValueError, match="requires an explicit robot_physics_profile"):
        _visual_sonic_humanoid_config(robot_physics_profile=None)


def test_visual_sonic_is_self_contained_without_grail_checkout() -> None:
    config = _visual_sonic_humanoid_config()
    assert config.grail_root_path is None


def test_heightmap_sonic_still_requires_grail_checkout() -> None:
    with pytest.raises(ValueError, match="grail_heightmap motion_reference"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            reference_controller_profile=(
                HumanoidReferenceControllerProfile.grail_heightmap
            ),
            reward_profile_id="reference_route_centered.v3",
        )


@pytest.mark.parametrize(
    "profile",
    [
        "g1_cylinder_model_12",
        "sonic_visual.mujoco_release.unknown.v1",
        "sonic_visual.mujoco_release.g1_29dof_rev_1_0.capsule_raft.v1",
    ],
)
def test_visual_sonic_rejects_unqualified_robot_physics_profile(
    profile: str,
) -> None:
    with pytest.raises(ValueError, match="robot_physics_profile must be one of"):
        _visual_sonic_humanoid_config(robot_physics_profile=profile)


def test_nonvisual_controller_rejects_unused_robot_physics_profile() -> None:
    with pytest.raises(
        ValueError,
        match="robot_physics_profile requires reference_controller_profile=sonic_visual",
    ):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
            robot_physics_profile=(
                "sonic.isaac_training.g1_cylinder_model_12.mujoco_port.v1"
            ),
            reward_profile_id="direct_v9_shaped.v1",
        )


def test_humanoid_policy_camera_profile_round_trips_resolved_config(
    tmp_path: Path,
) -> None:
    """External Hydra-group values must not leak into the typed YAML enum field."""
    run_config = _make_run_config(tmp_path)
    humanoid = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scene_cache_path="/tmp/humanoid-cache",
        runtime_cache_path="/tmp/humanoid-runtime-cache",
        scenario_ids_by_scene={"stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/tmp/GRAIL",
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d435_native,
        service_image="alpasim-humanoid-nurec:local",
        reward_profile_id="reference_route_centered.v3",
    )
    run_config = replace(
        run_config,
        alpasim=replace(
            run_config.alpasim,
            simulation_domain="humanoid",
            humanoid=humanoid,
        ),
    )

    write_run_artifacts(run_config)

    raw_config = yaml.safe_load(
        run_config.artifact_paths.resolved_config_path.read_text(encoding="utf-8")
    )
    loaded_config = load_run_config(run_config.artifact_paths.resolved_config_path)
    assert (
        raw_config["alpasim"]["humanoid"]["policy_camera_profile"] == "vla_d435_native"
    )
    assert HumanoidPolicyCameraProfile.vla_d435_native.value == "vla_d435_native"
    assert (
        HumanoidPolicyCameraProfile.vla_d435_native.wizard_config_group
        == "humanoid_vla_d435_native"
    )
    assert loaded_config.alpasim.humanoid is not None
    assert (
        loaded_config.alpasim.humanoid.policy_camera_profile
        is HumanoidPolicyCameraProfile.vla_d435_native
    )


def test_humanoid_motion_reference_without_camera_remains_supported() -> None:
    config = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/tmp/GRAIL",
        reward_profile_id="reference_route_centered.v3",
    )

    assert config.policy_camera_profile is None
    assert config.scene_cache_path is None
    assert config.runtime_cache_path is None


def test_vla_policy_rejects_av_runtime_route() -> None:
    """The VLA must fail before an AV runtime can dispatch the wrong factory."""
    config = _make_valid_vla_validation_config()
    config.alpasim.simulation_domain = "av"

    with pytest.raises(ValueError, match="simulation_domain=humanoid"):
        _validate_humanoid_config(config)


def test_vla_policy_rejects_av_policy_dispatch() -> None:
    """The VLA model cannot enter the AV policy factory through an override."""
    config = _make_valid_vla_validation_config()
    config.policy.kind = "alpamayo"

    with pytest.raises(ValueError, match="policy.kind=humanoid"):
        _validate_humanoid_config(config)


def test_vla_runtime_cache_must_be_disjoint_from_policy_model() -> None:
    """A renderer cache cannot write through the frozen VLA model tree."""
    config = _make_valid_vla_validation_config()
    assert config.alpasim.humanoid is not None
    config.alpasim.humanoid.runtime_cache_path = f"{config.policy.model.path}/native"

    with pytest.raises(ValueError, match="policy.model.path"):
        _validate_humanoid_config(config)


def test_vla_policy_rejects_direct_action_runtime_route() -> None:
    """The one-second VLA output cannot enter the direct-action execution ABI."""
    config = _make_valid_vla_validation_config()
    assert config.alpasim.humanoid is not None
    config.alpasim.humanoid.execution_profile = HumanoidExecutionProfile.direct_action

    with pytest.raises(ValueError, match="execution_profile=motion_reference"):
        _validate_humanoid_config(config)


def test_vla_policy_accepts_bounded_action_lr_calibration() -> None:
    """Actor LR may decrease while optimizer-group order and critic LR stay pinned."""
    config = _make_valid_vla_validation_config()
    config.cosmos.train.optm_part_lrs = [2.5e-7, 1.0e-4]

    _validate_humanoid_config(config)


@pytest.mark.parametrize("seed", [None, 0, -1, 2**32, True])
def test_vla_policy_rejects_non_reproducible_train_seed(seed: object) -> None:
    """Formal VLA training must initialize its critic from one valid seed."""
    config = _make_valid_vla_validation_config()
    config.cosmos.train.seed = seed

    with pytest.raises(ValueError, match="cosmos.train.seed"):
        _validate_humanoid_config(config)


def test_vla_policy_requires_deterministic_training() -> None:
    """Formal VLA comparisons cannot silently use nondeterministic kernels."""
    config = _make_valid_vla_validation_config()
    config.cosmos.train.deterministic = False

    with pytest.raises(ValueError, match="deterministic=true"):
        _validate_humanoid_config(config)


@pytest.mark.parametrize(
    ("part_lrs", "message"),
    [
        ([2.5e-7], "exactly two optm_part_lrs"),
        ([1.1e-6, 1.0e-4], "action_header optm_part_lrs\\[0\\]"),
        ([2.5e-7, 5.0e-5], "critic optm_part_lrs\\[1\\]"),
    ],
)
def test_vla_policy_rejects_unsafe_optimizer_group_lrs(
    part_lrs: list[float],
    message: str,
) -> None:
    """VLA calibration cannot increase actor LR or retune the critic group."""
    config = _make_valid_vla_validation_config()
    config.cosmos.train.optm_part_lrs = part_lrs

    with pytest.raises(ValueError, match=message):
        _validate_humanoid_config(config)


@pytest.mark.parametrize("factory_spec", [None, "zero"])
def test_vla_policy_requires_native_runtime_factory(
    factory_spec: str | None,
) -> None:
    """Missing or smoke-test factories cannot silently replace the VLA."""
    config = _make_valid_vla_validation_config()
    if factory_spec is None:
        config.policy.model.bundle_config.pop("humanoid_policy_factory")
    else:
        config.policy.model.bundle_config["humanoid_policy_factory"] = factory_spec

    with pytest.raises(ValueError, match="native humanoid_policy_factory"):
        _validate_humanoid_config(config)


@pytest.mark.parametrize("camera_requirement", [None, False])
def test_vla_policy_requires_strict_camera_runtime(
    camera_requirement: bool | None,
) -> None:
    """A visual VLA cannot run after its strict D455 camera ABI is disabled."""
    config = _make_valid_vla_validation_config()
    if camera_requirement is None:
        config.policy.model.bundle_config.pop("require_policy_camera")
    else:
        config.policy.model.bundle_config["require_policy_camera"] = camera_requirement

    with pytest.raises(ValueError, match="require_policy_camera=true"):
        _validate_humanoid_config(config)


def test_humanoid_policy_camera_rejects_dynamics_only_image() -> None:
    with pytest.raises(ValueError, match="combined image"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scene_cache_path="/tmp/humanoid-cache",
            runtime_cache_path="/tmp/humanoid-runtime-cache",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
            policy_camera_profile=HumanoidPolicyCameraProfile.vla_d455,
            reward_profile_id="reference_route_centered.v3",
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


def test_motion_reference_accepts_direct_v9_reward_profile() -> None:
    """The V9 scalar contract is valid at the motion-reference boundary."""
    config = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path="/tmp/GRAIL",
        reward_profile_id="direct_v9_shaped.v1",
    )

    assert config.reward_profile_id == "direct_v9_shaped.v1"


def test_motion_reference_requires_explicit_reward_profile() -> None:
    with pytest.raises(ValueError, match="explicit reward_profile_id"):
        HumanoidAlpaSimConfig(
            repo_path="/tmp/alpasim-humanoid",
            scene_store_path="/tmp/humanoid-scenes",
            scenario_ids_by_scene={"stairs": "ascend"},
            execution_profile=HumanoidExecutionProfile.motion_reference,
            grail_root_path="/tmp/GRAIL",
        )


def test_direct_action_resolves_its_historical_reward_default() -> None:
    config = HumanoidAlpaSimConfig(
        repo_path="/tmp/alpasim-humanoid",
        scene_store_path="/tmp/humanoid-scenes",
        scenario_ids_by_scene={"stairs": "ascend"},
    )

    assert config.reward_profile_id == "direct_v9_shaped.v1"


def test_vla_direct_v9_reward_requires_source_horizon() -> None:
    config = _make_valid_vla_validation_config()
    config.alpasim.wizard_args.n_sim_steps = 29
    config.expected_valid_steps = 29

    with pytest.raises(ValueError, match="750-tick / 15-second horizon"):
        _validate_humanoid_config(config)


def test_vla_stable_support_reward_requires_thirty_second_horizon() -> None:
    config = _make_valid_vla_validation_config()
    config.alpasim.humanoid.reward_profile_id = "stable_support_route.v2"

    with pytest.raises(ValueError, match="1500-tick / 30-second horizon"):
        _validate_humanoid_config(config)


def test_motion_reference_freeze_injects_single_source_paths_and_fingerprint(
    tmp_path: Path,
) -> None:
    model_path = _write_hf_bundle_dir(tmp_path)
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
                kind="g1_mjlab",
                path=str(model_path),
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


@pytest.mark.parametrize(
    "missing_label",
    [
        "VLA policy_eval_root",
        "alpasim.humanoid.repo_path",
        "alpasim.humanoid.scene_store_path",
        "alpasim.humanoid.grail_root_path",
        "alpasim.humanoid.scene_cache_path",
        "alpasim.humanoid.runtime_cache_path",
        "alpasim.repo_path",
    ],
)
def test_vla_slurm_requires_every_worker_path_identity_mounted(
    tmp_path: Path,
    missing_label: str,
) -> None:
    """A model-leaf mount cannot hide missing VLA source/service mounts."""
    policy_eval_root = tmp_path / "policy_eval"
    model_root = policy_eval_root / "models" / "vla-model"
    required_paths = {
        "VLA policy_eval_root": policy_eval_root,
        "alpasim.humanoid.repo_path": tmp_path / "humanoid_repo",
        "alpasim.humanoid.scene_store_path": tmp_path / "scene_store",
        "alpasim.humanoid.grail_root_path": tmp_path / "grail",
        "alpasim.humanoid.scene_cache_path": tmp_path / "scene_cache",
        "alpasim.humanoid.runtime_cache_path": tmp_path / "runtime_cache",
        "alpasim.repo_path": tmp_path / "alpasim_repo",
    }
    mounts = [f"{model_root}:{model_root}"]
    mounts.extend(
        f"{path}:{path}"
        for label, path in required_paths.items()
        if label != missing_label
    )
    config = SimpleNamespace(
        execution=SimpleNamespace(
            backend=ExecutionBackend.slurm,
            slurm=SimpleNamespace(container_mounts=mounts),
        ),
        policy=SimpleNamespace(
            model=SimpleNamespace(kind="g1_vla", path=str(model_root))
        ),
        alpasim=SimpleNamespace(
            repo_path=str(required_paths["alpasim.repo_path"]),
            checkout_cache_dir=None,
            humanoid=SimpleNamespace(
                execution_profile=HumanoidExecutionProfile.motion_reference,
                repo_path=str(required_paths["alpasim.humanoid.repo_path"]),
                scene_store_path=str(
                    required_paths["alpasim.humanoid.scene_store_path"]
                ),
                grail_root_path=str(required_paths["alpasim.humanoid.grail_root_path"]),
                scene_cache_path=str(
                    required_paths["alpasim.humanoid.scene_cache_path"]
                ),
                runtime_cache_path=str(
                    required_paths["alpasim.humanoid.runtime_cache_path"]
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match=re.escape(missing_label)):
        _validate_vla_slurm_worker_mounts(cast(RunConfig, config))


def test_vla_slurm_accepts_all_worker_mounts_and_local_needs_none(
    tmp_path: Path,
) -> None:
    """Complete Slurm visibility passes while local execution remains unchanged."""
    policy_eval_root = tmp_path / "policy_eval"
    model_root = policy_eval_root / "models" / "vla-model"
    worker_paths = [
        policy_eval_root,
        tmp_path / "humanoid_repo",
        tmp_path / "scene_store",
        tmp_path / "grail",
        tmp_path / "scene_cache",
        tmp_path / "runtime_cache",
        tmp_path / "alpasim_checkout_cache",
    ]
    config = SimpleNamespace(
        execution=SimpleNamespace(
            backend=ExecutionBackend.slurm,
            slurm=SimpleNamespace(
                container_mounts=[f"{path}:{path}" for path in worker_paths]
            ),
        ),
        policy=SimpleNamespace(
            model=SimpleNamespace(kind="g1_vla", path=str(model_root))
        ),
        alpasim=SimpleNamespace(
            repo_path=None,
            checkout_cache_dir=str(worker_paths[-1]),
            humanoid=SimpleNamespace(
                execution_profile=HumanoidExecutionProfile.motion_reference,
                repo_path=str(worker_paths[1]),
                scene_store_path=str(worker_paths[2]),
                grail_root_path=str(worker_paths[3]),
                scene_cache_path=str(worker_paths[4]),
                runtime_cache_path=str(worker_paths[5]),
            ),
        ),
    )

    _validate_vla_slurm_worker_mounts(cast(RunConfig, config))
    config.policy.model.path = "models/vla-model"
    with pytest.raises(ValueError, match="policy.model.path must be an absolute path"):
        _validate_vla_slurm_worker_mounts(cast(RunConfig, config))
    config.policy.model.path = str(model_root)
    config.alpasim.checkout_cache_dir = None
    with pytest.raises(ValueError, match="alpasim.repo_path or"):
        _validate_vla_slurm_worker_mounts(cast(RunConfig, config))
    config.execution.backend = ExecutionBackend.local_process
    config.execution.slurm.container_mounts = []
    _validate_vla_slurm_worker_mounts(cast(RunConfig, config))


def test_vla_visual_sonic_slurm_needs_no_grail_mount(tmp_path: Path) -> None:
    policy_eval_root = tmp_path / "policy_eval"
    model_root = policy_eval_root / "models" / "vla-model"
    humanoid_repo = tmp_path / "humanoid_repo"
    scene_store = tmp_path / "scene_store"
    scene_cache = tmp_path / "scene_cache"
    runtime_cache = tmp_path / "runtime_cache"
    visual_release = tmp_path / "visual_release"
    alpasim_repo = tmp_path / "alpasim_repo"
    worker_paths = (
        policy_eval_root,
        humanoid_repo,
        scene_store,
        scene_cache,
        runtime_cache,
        visual_release,
        alpasim_repo,
    )
    config = SimpleNamespace(
        execution=SimpleNamespace(
            backend=ExecutionBackend.slurm,
            slurm=SimpleNamespace(
                container_mounts=[f"{path}:{path}" for path in worker_paths]
            ),
        ),
        policy=SimpleNamespace(
            model=SimpleNamespace(kind="g1_vla", path=str(model_root))
        ),
        alpasim=SimpleNamespace(
            repo_path=str(alpasim_repo),
            checkout_cache_dir=None,
            humanoid=SimpleNamespace(
                execution_profile=HumanoidExecutionProfile.motion_reference,
                reference_controller_profile=(
                    HumanoidReferenceControllerProfile.sonic_visual
                ),
                repo_path=str(humanoid_repo),
                scene_store_path=str(scene_store),
                grail_root_path=None,
                scene_cache_path=str(scene_cache),
                runtime_cache_path=str(runtime_cache),
                visual_controller_release_path=str(visual_release),
            ),
        ),
    )

    _validate_vla_slurm_worker_mounts(cast(RunConfig, config))


def test_vla_mount_preflight_precedes_unqualified_slurm_mode_rejection(
    tmp_path: Path,
) -> None:
    """A candidate Slurm run reports its latent model-root mount first."""
    policy_eval_root = tmp_path / "policy_eval"
    model_root = policy_eval_root / "models" / "vla-model"
    run_config = _make_run_config(
        tmp_path,
        f"policy.model.path={model_root}",
        "topology=slurm_partial_node_1_2_1",
        "execution.slurm.container_image=/containers/alpagym.sqsh",
    )
    humanoid_paths = {
        "repo": tmp_path / "humanoid_repo",
        "scene_store": tmp_path / "scene_store",
        "grail": tmp_path / "grail",
        "scene_cache": tmp_path / "scene_cache",
        "runtime_cache": tmp_path / "runtime_cache",
    }
    run_config.alpasim.simulation_domain = "humanoid"
    run_config.alpasim.humanoid = HumanoidAlpaSimConfig(
        repo_path=str(humanoid_paths["repo"]),
        scene_store_path=str(humanoid_paths["scene_store"]),
        scenario_ids_by_scene={"stairs": "ascend"},
        execution_profile=HumanoidExecutionProfile.motion_reference,
        grail_root_path=str(humanoid_paths["grail"]),
        policy_camera_profile=HumanoidPolicyCameraProfile.vla_d435_native,
        scene_cache_path=str(humanoid_paths["scene_cache"]),
        runtime_cache_path=str(humanoid_paths["runtime_cache"]),
        service_image="combined-humanoid:latest",
        reward_profile_id="reference_route_centered.v3",
    )
    run_config.policy.model.kind = "g1_vla"
    run_config.policy.model.path = str(model_root)
    run_config.execution.slurm.container_mounts = [
        f"{model_root}:{model_root}",
        *[f"{path}:{path}" for path in humanoid_paths.values()],
        (
            f"{run_config.alpasim.checkout_cache_dir}:"
            f"{run_config.alpasim.checkout_cache_dir}"
        ),
    ]

    with pytest.raises(ValueError, match="VLA policy_eval_root"):
        validate_run_config(run_config, "run")


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


def _make_valid_vla_validation_config() -> RunConfig:
    """Build the complete config slice consumed by humanoid preflight."""
    fingerprints = {"stairs": "a" * 64}
    humanoid_repo_path = "/tmp/humanoid-support"
    scene_store_path = "/tmp/humanoid-scenes"
    return cast(
        RunConfig,
        SimpleNamespace(
            policy=SimpleNamespace(
                kind="humanoid",
                model=SimpleNamespace(
                    kind="g1_vla",
                    path="/tmp/vla-model",
                    step_dt_us=500_000,
                    use_cameras=["vla_d435_policy_rgb"],
                    bundle_config={
                        "humanoid_policy_factory": (
                            "alpagym_g1_vla.humanoid_policy:"
                            "build_humanoid_policy_factory"
                        ),
                        "require_policy_camera": True,
                        "expected_scene_fingerprints_json": json.dumps(
                            fingerprints,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "humanoid_repo_path": humanoid_repo_path,
                        "scene_store_path": scene_store_path,
                    },
                ),
            ),
            alpasim=SimpleNamespace(
                simulation_domain="humanoid",
                repo_path="/tmp/alpasim",
                humanoid=SimpleNamespace(
                    repo_path=humanoid_repo_path,
                    scene_store_path=scene_store_path,
                    scenario_ids_by_scene={"stairs": "ascend"},
                    expected_scene_fingerprints=fingerprints,
                    execution_profile=HumanoidExecutionProfile.motion_reference,
                    reward_profile_id="direct_v9_shaped.v1",
                    reference_controller_profile=(
                        HumanoidReferenceControllerProfile.sonic_visual
                    ),
                    reference_frame_count=50,
                    policy_camera_profile=HumanoidPolicyCameraProfile.vla_d435_native,
                    runtime_cache_path="/tmp/native-runtime-cache",
                ),
                wizard_args=SimpleNamespace(
                    control_timestep_us=500_000,
                    force_gt_duration_us=0,
                    n_sim_steps=30,
                ),
            ),
            dataset=SimpleNamespace(scene_ids=["stairs"]),
            cosmos=SimpleNamespace(
                mode=CosmosRLMode.colocated,
                rollout=SimpleNamespace(prefetch_rollout=False),
                train=SimpleNamespace(
                    seed=20260822,
                    deterministic=True,
                    train_policy=SimpleNamespace(
                        trainer_type="alpagym_flow_ppo",
                        grpo_ratio_clip_low=0.2,
                        grpo_ratio_clip_high=0.28,
                        ppo_value_loss_coef=1.0,
                        ppo_value_clip_range=0.2,
                        ppo_gamma=0.99,
                        ppo_gae_lambda=0.95,
                        ppo_dual_clip_ratio=3.0,
                        ppo_value_huber_delta=10.0,
                        ppo_normalize_advantages=True,
                        kl_beta=0.0,
                    ),
                    optm_part_lrs=[1.0e-6, 1.0e-4],
                    epsilon=1.0e-8,
                    optm_weight_decay=0.01,
                    optm_betas=[0.9, 0.999],
                    optm_grad_norm_clip=1.0,
                    optm_warmup_steps=0,
                ),
            ),
            expected_valid_steps=30,
        ),
    )


def _model_overrides(tmp_path: Path) -> list[str]:
    """Return policy overrides for tests that do not validate model files."""
    return [
        "policy.model.kind=alpamayo_r1",
        f"policy.model.path={(tmp_path / 'model_bundle').as_posix()}",
    ]


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


def test_alpasim_rejects_inner_outer_deadline_race(tmp_path: Path) -> None:
    config = _make_run_config(tmp_path)
    config.alpasim.runtime_rollout_timeout_s = 600.0
    config.alpasim.simulation_cleanup_margin_s = 60.0
    config.alpasim.simulation_timeout_s = 600.0

    with pytest.raises(ValueError, match="structured timeout after bounded teardown"):
        _validate_wizard_startup_config(config.alpasim, config.dataset)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True])
def test_alpasim_deadlines_must_be_finite_positive(
    tmp_path: Path, value: float
) -> None:
    config = _make_run_config(tmp_path)
    config.alpasim.simulation_cleanup_margin_s = value

    with pytest.raises(ValueError, match="simulation_cleanup_margin_s"):
        _validate_wizard_startup_config(config.alpasim, config.dataset)
