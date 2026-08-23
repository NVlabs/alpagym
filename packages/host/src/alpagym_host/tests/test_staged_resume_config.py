# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import alpagym_host.config_validation as config_validation
from alpagym_host.config import ProvenanceMode


def test_staged_resume_accepts_one_logical_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage stop may advance while checkpoint metadata stays at the full horizon."""

    config, source = _resume_config()
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
        ("total_steps", 5, "logical total_steps"),
        ("remain_samples_num", 12, "remaining sample cursor"),
        ("is_final", True, "final logical checkpoint"),
    ],
)
def test_staged_resume_rejects_stage_local_checkpoint_cursor(
    monkeypatch: pytest.MonkeyPatch,
    cursor_field: str,
    value: Any,
    message: str,
) -> None:
    """A checkpoint must describe the 50-step run, not one process stage."""

    config, source = _resume_config()
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restarting a stage cannot reinterpret a step-relative LR schedule."""

    config, source = _resume_config()
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first resumed rollout seed must remain an exact uint64."""

    config, source = _resume_config()
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


def _resume_config() -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build the minimal typed shape consumed by continuation preflight."""

    current_run = Path("/tmp/20260823T123456Z-22222222222222222222222222222222")
    prior_run = Path("/tmp/20260823T120000Z-11111111111111111111111111111111")
    contract = SimpleNamespace(
        enabled=True,
        checkpoint_step=2,
        expected_next_training_step=3,
    )
    source = SimpleNamespace(
        prior_formal_run_dir=prior_run,
        cursor=SimpleNamespace(
            step=2,
            total_steps=50,
            remain_samples_num=192,
            is_final=False,
        ),
    )
    config = SimpleNamespace(
        artifact_paths=SimpleNamespace(run_dir=current_run),
        execution=SimpleNamespace(provenance_mode=ProvenanceMode.required),
        dataset=SimpleNamespace(scene_ids=["hq_stairs"]),
        alpasim=SimpleNamespace(
            simulation_domain="humanoid",
            humanoid=SimpleNamespace(rollout_seed_base=292_285),
        ),
        cosmos=SimpleNamespace(
            rollout=SimpleNamespace(n_generation=4),
            launch=SimpleNamespace(policy_replicas=1),
            train=SimpleNamespace(
                resume=contract,
                max_num_steps=5,
                num_epochs=50,
                train_batch_per_replica=4,
                optm_warmup_steps=0,
                optm_decay_type="none",
                optm_decay_ratio=0.0,
                ckpt=SimpleNamespace(enable_checkpoint=True),
            ),
        ),
    )
    return config, source
