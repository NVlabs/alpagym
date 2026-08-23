# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch

from alpagym_host.checkpoint_resume import (
    canonical_json_sha256,
    checkpoint_tree_snapshot,
    validate_checkpoint_resume_source,
    validate_native_checkpoint_files,
)
from alpagym_host.config import CosmosRLCheckpointResumeConfig


def test_resume_requires_checkpoint_bytes_sealed_by_valid_postrun(
    tmp_path: Path,
) -> None:
    prior_run = tmp_path / "20260823T123456Z-0123456789abcdef0123456789abcdef"
    checkpoint = (
        prior_run / "cosmos" / "20260823123456" / "checkpoints" / "step_2" / "policy"
    )
    _write_native_checkpoint(
        checkpoint,
        step=2,
        total_steps=5,
        remain_samples_num=12,
        is_final=False,
    )
    snapshot = checkpoint_tree_snapshot(checkpoint)
    receipt_sha256 = _write_valid_postrun(
        prior_run=prior_run,
        checkpoint=checkpoint,
        step=2,
    )
    contract = CosmosRLCheckpointResumeConfig(
        enabled=True,
        prior_formal_run_id=prior_run.name,
        checkpoint_step=2,
        checkpoint_path=str(checkpoint),
        checkpoint_tree_sha256=snapshot.tree_sha256,
        prior_postrun_receipt_sha256=receipt_sha256,
        expected_next_training_step=3,
    )

    source = validate_checkpoint_resume_source(contract)

    assert source.snapshot.tree_sha256 == snapshot.tree_sha256
    assert source.ranks == (0,)
    assert source.cursor.step == 2
    assert source.cursor.total_steps == 5
    assert source.cursor.remain_samples_num == 12
    assert source.cursor.is_final is False

    # Replacing a checkpoint and updating only the caller-provided hash must not
    # make the replacement resumable: the immutable prior postrun sealed bytes.
    (checkpoint / "model_rank_0.pth").write_bytes(b"replaced model")
    replaced = checkpoint_tree_snapshot(checkpoint)
    contract.checkpoint_tree_sha256 = replaced.tree_sha256
    with pytest.raises(ValueError, match="checkpoint seal does not match live"):
        validate_checkpoint_resume_source(contract)


def test_resume_rejects_invalid_formal_run_even_with_matching_checkpoint(
    tmp_path: Path,
) -> None:
    prior_run = tmp_path / "20260823T123456Z-fedcba9876543210fedcba9876543210"
    checkpoint = (
        prior_run / "cosmos" / "20260823123456" / "checkpoints" / "step_1" / "policy"
    )
    _write_native_checkpoint(
        checkpoint,
        step=1,
        total_steps=1,
        remain_samples_num=0,
        is_final=True,
    )
    snapshot = checkpoint_tree_snapshot(checkpoint)
    receipt_sha256 = _write_valid_postrun(
        prior_run=prior_run,
        checkpoint=checkpoint,
        step=1,
        formal_run_valid=False,
    )
    contract = CosmosRLCheckpointResumeConfig(
        enabled=True,
        prior_formal_run_id=prior_run.name,
        checkpoint_step=1,
        checkpoint_path=str(checkpoint),
        checkpoint_tree_sha256=snapshot.tree_sha256,
        prior_postrun_receipt_sha256=receipt_sha256,
        expected_next_training_step=2,
    )

    with pytest.raises(ValueError, match="not valid for checkpoint continuation"):
        validate_checkpoint_resume_source(contract)


def test_resume_rejects_incoherent_checkpoint_cursor(tmp_path: Path) -> None:
    prior_run = tmp_path / "20260823T123456Z-11111111111111111111111111111111"
    checkpoint = (
        prior_run / "cosmos" / "20260823123456" / "checkpoints" / "step_2" / "policy"
    )
    _write_native_checkpoint(
        checkpoint,
        step=2,
        total_steps=5,
        remain_samples_num=12,
        is_final=True,
    )
    snapshot = checkpoint_tree_snapshot(checkpoint)
    receipt_sha256 = _write_valid_postrun(
        prior_run=prior_run,
        checkpoint=checkpoint,
        step=2,
    )
    contract = CosmosRLCheckpointResumeConfig(
        enabled=True,
        prior_formal_run_id=prior_run.name,
        checkpoint_step=2,
        checkpoint_path=str(checkpoint),
        checkpoint_tree_sha256=snapshot.tree_sha256,
        prior_postrun_receipt_sha256=receipt_sha256,
        expected_next_training_step=3,
    )

    with pytest.raises(ValueError, match="is_final disagrees"):
        validate_checkpoint_resume_source(contract)


def _write_native_checkpoint(
    path: Path,
    *,
    step: int,
    total_steps: int,
    remain_samples_num: int,
    is_final: bool,
) -> None:
    path.mkdir(parents=True)
    files = {
        ".rank_0_complete": b"",
        "cosmos_config": b"{}",
        "model_rank_0.pth": b"model",
        "optimizer_rank_0.pth": b"optimizer",
        "scheduler_rank_0.pth": b"scheduler",
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    torch.save(
        {
            "rng_state": {"torch": b"torch", "numpy": (), "python": ()},
            "step": step,
            "total_steps": total_steps,
            "remain_samples_num": remain_samples_num,
            "is_final": is_final,
        },
        path / "extra_info_rank_0.pth",
    )


def _write_valid_postrun(
    *,
    prior_run: Path,
    checkpoint: Path,
    step: int,
    formal_run_valid: bool = True,
) -> str:
    snapshot = checkpoint_tree_snapshot(checkpoint)
    ranks = validate_native_checkpoint_files(snapshot)
    body = {
        "schema_id": "alpagym.formal_run_postrun.v2",
        "run_completed": True,
        "cleanup_succeeded": True,
        "cleanup_failure": None,
        "runtime_ready_captured": True,
        "source_watch_clean": True,
        "sources_unchanged": True,
        "configs_unchanged": True,
        "formal_run_valid": formal_run_valid,
        "native_checkpoints": [
            {
                "cosmos_output_relative_path": checkpoint.parents[2]
                .relative_to(prior_run)
                .as_posix(),
                "policy_relative_path": checkpoint.relative_to(prior_run).as_posix(),
                "step": step,
                "tree_sha256": snapshot.tree_sha256,
                "file_count": snapshot.file_count,
                "total_size_bytes": snapshot.total_size_bytes,
                "files": [identity.to_dict() for identity in snapshot.files],
                "ranks": list(ranks),
            }
        ],
    }
    receipt_sha256 = canonical_json_sha256(body)
    postrun = {**body, "receipt_sha256": receipt_sha256}
    provenance = prior_run / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "postrun.json").write_text(
        json.dumps(postrun, sort_keys=True),
        encoding="utf-8",
    )
    return receipt_sha256
