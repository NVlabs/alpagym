# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import alpagym_host.config_validation as config_validation
from alpagym_host.config import RunConfig, register_config_schema
from alpagym_host.run_artifacts import build_artifact_paths, build_run_config
from hydra import compose, initialize_config_module


def test_train2_staged_profile_is_independent_from_four_rollout_smoke(
    tmp_path: Path,
) -> None:
    """The staged profile changes training geometry without changing the smoke."""

    smoke = _experiment_config(
        tmp_path=tmp_path / "smoke",
        experiment="g1_vla_hq_stairs_local_1gpu",
    )
    staged = _experiment_config(
        tmp_path=tmp_path / "staged",
        experiment="g1_vla_hq_stairs_local_1gpu_train2_staged",
    )

    assert (
        smoke.cosmos.train.train_batch_per_replica,
        smoke.cosmos.rollout.n_generation,
        smoke.cosmos.train.num_epochs,
        smoke.cosmos.train.max_num_steps,
    ) == (4, 4, 1, 1)
    assert (
        staged.cosmos.train.train_batch_per_replica,
        staged.cosmos.rollout.n_generation,
        staged.cosmos.train.num_epochs,
        staged.cosmos.train.max_num_steps,
        staged.cosmos.train.ckpt.save_freq,
    ) == (2, 2, 50, 5, 5)
    logical_samples = (
        len(staged.dataset.scene_ids or [])
        * staged.cosmos.rollout.n_generation
        * staged.cosmos.train.num_epochs
    )
    global_batch = (
        staged.cosmos.train.train_batch_per_replica
        * staged.cosmos.launch.policy_replicas
    )
    assert logical_samples == 100
    assert global_batch == 2
    assert logical_samples % global_batch == 0
    assert logical_samples // global_batch == 50
    assert staged.cosmos.train.sync_weight_interval == 1
    assert staged.cosmos.train.train_policy.on_policy is True
    assert staged.cosmos.train.train_policy.allowed_outdated_steps == 0
    assert smoke.cosmos.train.train_policy.ppo_behavior_kl_target_mode == "soft"
    assert staged.cosmos.train.train_policy.ppo_behavior_kl_target_mode == "soft"
    assert staged.cosmos.train.train_policy.ppo_target_behavior_kl == 0.003
    assert staged.cosmos.train.train_policy.ppo_behavior_kl_hard_limit == 0.01
    assert staged.cosmos.train.ckpt.enable_checkpoint is True
    assert staged.cosmos.train.ckpt.save_mode == "sync"
    assert smoke.alpasim.humanoid.rollout_seed_base == 292285
    assert staged.alpasim.humanoid.rollout_seed_base == 202608240100000


def test_staged_resume_accepts_step_five_to_cumulative_step_ten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact step-five checkpoint may continue the profile through step ten."""

    config, source = _resume_config(tmp_path)
    monkeypatch.setattr(
        config_validation,
        "validate_checkpoint_resume_source",
        lambda contract: source,
    )

    config_validation._validate_checkpoint_resume_config(
        config=config,
        requested_command="run",
    )


@pytest.mark.parametrize(
    ("cursor_field", "value", "message"),
    [
        ("total_steps", 10, "logical total_steps"),
        ("remain_samples_num", 10, "remaining sample cursor"),
        ("is_final", True, "final logical checkpoint"),
    ],
)
def test_staged_resume_rejects_stage_local_checkpoint_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cursor_field: str,
    value: Any,
    message: str,
) -> None:
    """A checkpoint must describe the 50-step run, not one process stage."""

    config, source = _resume_config(tmp_path)
    setattr(source.cursor, cursor_field, value)
    monkeypatch.setattr(
        config_validation,
        "validate_checkpoint_resume_source",
        lambda contract: source,
    )

    with pytest.raises(ValueError, match=message):
        config_validation._validate_checkpoint_resume_config(
            config=config,
            requested_command="run",
        )


def test_staged_resume_rejects_nonconstant_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restarting a stage cannot reinterpret a step-relative LR schedule."""

    config, source = _resume_config(tmp_path)
    config.cosmos.train.optm_decay_type = "cosine"
    config.cosmos.train.optm_decay_ratio = 1.0
    monkeypatch.setattr(
        config_validation,
        "validate_checkpoint_resume_source",
        lambda contract: source,
    )

    with pytest.raises(ValueError, match="constant LR schedule"):
        config_validation._validate_checkpoint_resume_config(
            config=config,
            requested_command="run",
        )


def test_staged_resume_rejects_seed_cursor_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first resumed rollout seed must remain an exact uint64."""

    config, source = _resume_config(tmp_path)
    config.alpasim.humanoid.rollout_seed_base = (1 << 64) - 1
    monkeypatch.setattr(
        config_validation,
        "validate_checkpoint_resume_source",
        lambda contract: source,
    )

    with pytest.raises(ValueError, match="seed panel exceeds uint64"):
        config_validation._validate_checkpoint_resume_config(
            config=config,
            requested_command="run",
        )


def test_staged_resume_rejects_nondivisible_logical_sample_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every authored rollout sample must belong to one complete global batch."""

    config, source = _resume_config(tmp_path)
    config.cosmos.train.train_batch_per_replica = 3
    monkeypatch.setattr(
        config_validation,
        "validate_checkpoint_resume_source",
        lambda contract: source,
    )

    with pytest.raises(ValueError, match="divisible by the global training batch"):
        config_validation._validate_checkpoint_resume_config(
            config=config,
            requested_command="run",
        )


def _resume_config(tmp_path: Path) -> tuple[RunConfig, SimpleNamespace]:
    """Compose the staged profile at its second cumulative process boundary."""

    prior_run_id = "20260823T120000Z-11111111111111111111111111111111"
    prior_run = tmp_path / prior_run_id
    config = _experiment_config(
        tmp_path=tmp_path,
        experiment="g1_vla_hq_stairs_local_1gpu_train2_staged",
        overrides=[
            "cosmos.train.max_num_steps=10",
            "cosmos.train.resume.enabled=true",
            f"cosmos.train.resume.prior_formal_run_id={prior_run_id}",
            "cosmos.train.resume.checkpoint_step=5",
            "cosmos.train.resume.checkpoint_path="
            f"{prior_run}/cosmos/20260823120000/checkpoints/step_5/policy",
            f"cosmos.train.resume.checkpoint_tree_sha256={'a' * 64}",
            f"cosmos.train.resume.prior_postrun_receipt_sha256={'b' * 64}",
            "cosmos.train.resume.expected_next_training_step=6",
        ],
    )
    source = SimpleNamespace(
        prior_formal_run_dir=prior_run,
        cursor=SimpleNamespace(
            step=5,
            total_steps=50,
            remain_samples_num=90,
            is_final=False,
        ),
    )
    return config, source


def _experiment_config(
    *,
    tmp_path: Path,
    experiment: str,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Compose one G1 VLA experiment into the typed host run contract."""

    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        authored = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path}",
                f"experiment={experiment}",
                f"policy.model.path={tmp_path / 'model'}",
                f"alpasim.repo_path={tmp_path / 'alpasim'}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
                f"alpasim.humanoid.repo_path={tmp_path / 'humanoid'}",
                f"alpasim.humanoid.scene_store_path={tmp_path / 'scenes'}",
                "alpasim.humanoid.visual_controller_release_path="
                f"{tmp_path / 'controller'}",
                f"alpasim.humanoid.scene_cache_path={tmp_path / 'scene-cache'}",
                f"alpasim.humanoid.runtime_cache_path={tmp_path / 'runtime-cache'}",
                *(overrides or []),
            ],
        )
    artifact_paths = build_artifact_paths(authored)
    return build_run_config(authored, artifact_paths)
