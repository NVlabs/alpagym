# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict export/load tests for immutable G1 VLA rollout candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from alpagym_host.checkpoint_resume import (
    canonical_json_sha256,
    checkpoint_tree_snapshot,
)
import alpagym_g1_vla.candidate_overlay as candidate_overlay_module
from alpagym_g1_vla.candidate_overlay import (
    CANDIDATE_MANIFEST_FILENAME,
    CANDIDATE_OVERLAY_SOURCE,
    CANDIDATE_WEIGHTS_FILENAME,
    apply_candidate_overlay,
    export_model_checkpoint,
    resolve_inference_source,
    verify_candidate_overlay,
)
from alpagym_g1_vla.cosmos_model import VlaPsiPPOModel
from alpagym_g1_vla.provenance import (
    LEGACY_VLA_BUNDLE_PROFILE,
    STAIRSBLOCKS_VLA_BUNDLE_PROFILE,
)
from alpagym_runtime.policies.registry import PolicyCheckpointExportContext


_FORMAL_RUN_ID = "20260822T123456Z-0123456789abcdef0123456789abcdef"
_COSMOS_TIMESTAMP = "20260822123456"


class _TinyPsi(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_header = torch.nn.Linear(3, 2)
        self.frozen_backbone = torch.nn.Linear(3, 3)
        self.frozen_backbone.requires_grad_(False)


class _TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.psi_model = _TinyPsi()
        self.critic = torch.nn.Linear(3, 1)
        self.normalizer = torch.nn.Module()
        self.normalizer.register_buffer(
            "action_low", torch.full((2,), -1.0), persistent=True
        )
        self.normalizer.register_buffer(
            "action_high", torch.full((2,), 1.0), persistent=True
        )


def _tiny_model(
    *,
    fill: float,
    action_dtype: torch.dtype = torch.float32,
    critic_dtype: torch.dtype = torch.float32,
) -> VlaPsiPPOModel:
    model = object.__new__(VlaPsiPPOModel)
    torch.nn.Module.__init__(model)
    model.actor_critic = _TinyActorCritic()
    model.actor_critic.psi_model.action_header.to(dtype=action_dtype)
    model.actor_critic.critic.to(dtype=critic_dtype)
    model.source_bundle = SimpleNamespace(profile=LEGACY_VLA_BUNDLE_PROFILE)
    model.config = SimpleNamespace(bundle_model_id=LEGACY_VLA_BUNDLE_PROFILE.model_id)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
    return model


def _candidate_path(tmp_path: Path, step: int = 7) -> Path:
    return (
        tmp_path
        / _FORMAL_RUN_ID
        / "cosmos"
        / _COSMOS_TIMESTAMP
        / "safetensors"
        / f"step_{step}"
    )


def _export_context(step: int = 7) -> PolicyCheckpointExportContext:
    return PolicyCheckpointExportContext(
        training_step=step,
        total_training_steps=50,
        optimizer_steps_applied=1,
        resolved_config_sha256="d" * 64,
        cosmos_run_id=_FORMAL_RUN_ID,
    )


def _seal_formal_run(
    candidate_path: Path,
    native_model: VlaPsiPPOModel,
    *,
    formal_run_valid: bool = True,
) -> dict[str, object]:
    """Write a tiny native checkpoint and finalized v2 formal receipt."""
    step = int(candidate_path.name.removeprefix("step_"))
    cosmos_output_dir = candidate_path.parent.parent
    formal_run_dir = cosmos_output_dir.parent.parent
    checkpoint_path = cosmos_output_dir / "checkpoints" / f"step_{step}" / "policy"
    checkpoint_path.mkdir(parents=True)
    torch.save(dict(native_model.state_dict()), checkpoint_path / "model_rank_0.pth")
    (checkpoint_path / ".rank_0_complete").write_bytes(b"")
    (checkpoint_path / "cosmos_config").write_text("config\n", encoding="utf-8")
    torch.save({"step": step}, checkpoint_path / "extra_info_rank_0.pth")
    torch.save({"optimizer": True}, checkpoint_path / "optimizer_rank_0.pth")
    torch.save({"scheduler": True}, checkpoint_path / "scheduler_rank_0.pth")
    snapshot = checkpoint_tree_snapshot(checkpoint_path)
    receipt: dict[str, object] = {
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
                "cosmos_output_relative_path": (
                    cosmos_output_dir.relative_to(formal_run_dir).as_posix()
                ),
                "policy_relative_path": checkpoint_path.relative_to(
                    formal_run_dir
                ).as_posix(),
                "step": step,
                "tree_sha256": snapshot.tree_sha256,
                "file_count": snapshot.file_count,
                "total_size_bytes": snapshot.total_size_bytes,
                "files": [identity.to_dict() for identity in snapshot.files],
                "ranks": [0],
            }
        ],
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    postrun_path = formal_run_dir / "provenance" / "postrun.json"
    postrun_path.parent.mkdir()
    postrun_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _run_config(
    model_root: Path,
    candidate: Path | None,
    *,
    expected_candidate_sha256: str | None = None,
) -> SimpleNamespace:
    bundle_config: dict[str, object] = {
        "qualification_model_source": (
            CANDIDATE_OVERLAY_SOURCE if candidate is not None else "base_attested"
        )
    }
    if candidate is not None:
        bundle_config["candidate_overlay_path"] = str(candidate)
        if expected_candidate_sha256 is None:
            manifest = json.loads(
                (candidate / CANDIDATE_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            expected_candidate_sha256 = manifest["candidate_sha256"]
        bundle_config["expected_candidate_sha256"] = expected_candidate_sha256
    return SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(model_root),
                bundle_config=bundle_config,
            )
        )
    )


def test_candidate_round_trip_updates_only_mutable_overlay(tmp_path: Path) -> None:
    source = _tiny_model(fill=4.0)
    target = _tiny_model(fill=-2.0)
    candidate_path = _candidate_path(tmp_path)

    export_model_checkpoint(source, candidate_path, _export_context())
    receipt = _seal_formal_run(candidate_path, source)
    candidate = verify_candidate_overlay(
        candidate_path,
        expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
    )
    frozen_before = (
        target.actor_critic.psi_model.frozen_backbone.weight.detach().clone()
    )
    loaded = apply_candidate_overlay(target, candidate)

    assert loaded.candidate_sha256 == candidate.candidate_sha256
    assert loaded.source_training_state["training_step"] == 7
    assert loaded.source_training_state["resolved_config_sha256"] == "d" * 64
    assert loaded.source_formal_run_id == _FORMAL_RUN_ID
    assert loaded.postrun_receipt_sha256 == receipt["receipt_sha256"]
    assert (
        loaded.native_checkpoint_tree_sha256
        == receipt["native_checkpoints"][0]["tree_sha256"]
    )
    torch.testing.assert_close(
        target.actor_critic.psi_model.action_header.weight,
        source.actor_critic.psi_model.action_header.weight,
    )
    torch.testing.assert_close(
        target.actor_critic.critic.weight,
        source.actor_critic.critic.weight,
    )
    torch.testing.assert_close(
        target.actor_critic.normalizer.action_low,
        source.actor_critic.normalizer.action_low,
    )
    torch.testing.assert_close(
        target.actor_critic.psi_model.frozen_backbone.weight,
        frozen_before,
    )
    manifest = json.loads(
        (candidate_path / CANDIDATE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["candidate_sha256"] == candidate.candidate_sha256
    assert manifest["source_training_state"] == {
        "kind": "cosmos_live_optimizer_state",
        "training_step": 7,
        "total_training_steps": 50,
        "optimizer_steps_applied": 1,
        "resolved_config_sha256": "d" * 64,
        "cosmos_run_id": _FORMAL_RUN_ID,
    }


def test_candidate_load_casts_fp32_action_master_to_bf16_rollout(
    tmp_path: Path,
) -> None:
    source = _tiny_model(fill=1.234567)
    target = _tiny_model(fill=-2.0, action_dtype=torch.bfloat16)
    candidate_path = _candidate_path(tmp_path)

    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)
    candidate = verify_candidate_overlay(
        candidate_path,
        expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
    )
    apply_candidate_overlay(target, candidate)

    assert target.actor_critic.psi_model.action_header.weight.dtype == torch.bfloat16
    assert target.actor_critic.critic.weight.dtype == torch.float32
    torch.testing.assert_close(
        target.actor_critic.psi_model.action_header.weight,
        source.actor_critic.psi_model.action_header.weight.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target.actor_critic.critic.weight,
        source.actor_critic.critic.weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("action_dtype", "critic_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_candidate_load_rejects_unapproved_dtype_conversions(
    tmp_path: Path,
    action_dtype: torch.dtype,
    critic_dtype: torch.dtype,
) -> None:
    source = _tiny_model(fill=1.0)
    target = _tiny_model(
        fill=-2.0,
        action_dtype=action_dtype,
        critic_dtype=critic_dtype,
    )
    candidate_path = _candidate_path(tmp_path)

    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)
    candidate = verify_candidate_overlay(
        candidate_path,
        expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
    )

    with pytest.raises(ValueError, match="candidate dtype mismatch"):
        apply_candidate_overlay(target, candidate)


def test_candidate_export_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    model = _tiny_model(fill=1.0)
    candidate_path = _candidate_path(tmp_path)
    export_model_checkpoint(model, candidate_path, _export_context())

    with pytest.raises(FileExistsError, match="immutable"):
        export_model_checkpoint(model, candidate_path, _export_context())

    assert {path.name for path in candidate_path.iterdir()} == {
        CANDIDATE_MANIFEST_FILENAME,
        CANDIDATE_WEIGHTS_FILENAME,
    }
    assert candidate_path.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in candidate_path.iterdir())


def test_candidate_export_requires_formal_run_directory_layout(
    tmp_path: Path,
) -> None:
    model = _tiny_model(fill=1.0)
    wrong_formal_run = (
        tmp_path
        / "20260822T123456Z-ffffffffffffffffffffffffffffffff"
        / "cosmos"
        / _COSMOS_TIMESTAMP
        / "safetensors"
        / "step_7"
    )

    with pytest.raises(ValueError, match="formal cosmos_run_id"):
        export_model_checkpoint(model, wrong_formal_run, _export_context())

    assert not wrong_formal_run.exists()


def test_candidate_publish_never_replaces_concurrent_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A race winner at the final name survives the atomic publish."""
    candidate_path = _candidate_path(tmp_path)
    real_save_file = candidate_overlay_module.save_file

    def _racing_save_file(state: object, path: Path) -> None:
        candidate_path.mkdir()
        real_save_file(state, path)

    monkeypatch.setattr(candidate_overlay_module, "save_file", _racing_save_file)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_model_checkpoint(
            _tiny_model(fill=1.0), candidate_path, _export_context()
        )

    assert candidate_path.is_dir()
    assert list(candidate_path.iterdir()) == []
    assert not any(
        path.name.startswith(f".{candidate_path.name}.tmp-")
        for path in candidate_path.parent.iterdir()
    )


def test_candidate_tamper_and_wrong_base_fail_closed(tmp_path: Path) -> None:
    candidate_path = _candidate_path(tmp_path)
    source = _tiny_model(fill=1.0)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)

    with pytest.raises(ValueError, match="selected attested base"):
        verify_candidate_overlay(
            candidate_path,
            expected_profile=STAIRSBLOCKS_VLA_BUNDLE_PROFILE,
        )

    weights = candidate_path / CANDIDATE_WEIGHTS_FILENAME
    weights.chmod(0o644)
    with weights.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="size changed|SHA256 mismatch"):
        verify_candidate_overlay(
            candidate_path,
            expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
        )


def test_source_selection_is_explicit_and_raw_resume_is_not_a_model_root(
    tmp_path: Path,
) -> None:
    model_root = (
        tmp_path / "policy-eval" / "models" / LEGACY_VLA_BUNDLE_PROFILE.model_id
    )
    candidate_path = _candidate_path(tmp_path)
    source = _tiny_model(fill=3.0)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)

    candidate, identity = resolve_inference_source(
        _run_config(model_root, candidate_path)
    )
    assert candidate is not None
    assert identity["source_kind"] == CANDIDATE_OVERLAY_SOURCE
    assert identity["candidate_overlay"]["candidate_sha256"] == (
        candidate.candidate_sha256
    )
    assert identity["candidate_overlay"]["source_formal_run_id"] == _FORMAL_RUN_ID
    assert identity["candidate_overlay"]["formal_run_valid"] is True
    assert identity["candidate_overlay"]["postrun_receipt_sha256"] == (
        candidate.postrun_receipt_sha256
    )
    assert identity["candidate_overlay"]["native_checkpoint_tree_sha256"] == (
        candidate.native_checkpoint_tree_sha256
    )

    config = _run_config(model_root, None)
    config.policy.model.bundle_config.clear()
    with pytest.raises(ValueError, match="must be explicitly"):
        resolve_inference_source(config)

    raw_resume = tmp_path / "cosmos" / "run-stamp" / "checkpoints" / "step_7"
    config = _run_config(raw_resume, None)
    with pytest.raises(ValueError, match=r"models/<model_id>"):
        resolve_inference_source(config)


def test_candidate_source_requires_the_exact_configured_digest(tmp_path: Path) -> None:
    model_root = (
        tmp_path / "policy-eval" / "models" / LEGACY_VLA_BUNDLE_PROFILE.model_id
    )
    candidate_path = _candidate_path(tmp_path)
    source = _tiny_model(fill=3.0)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)

    with pytest.raises(ValueError, match="expected_candidate_sha256"):
        resolve_inference_source(
            _run_config(
                model_root,
                candidate_path,
                expected_candidate_sha256="0" * 64,
            )
        )


def test_candidate_load_is_bound_to_the_verified_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the pathname after hashing cannot replace loaded tensors."""
    source = _tiny_model(fill=4.0)
    candidate_path = _candidate_path(tmp_path)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source)
    candidate = verify_candidate_overlay(
        candidate_path,
        expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
    )
    evil_path = tmp_path / "evil.safetensors"
    save_file(
        {
            name: torch.full_like(tensor, 9.0)
            for name, tensor in candidate_overlay_module._overlay_tensor_map(
                source
            ).items()
        },
        evil_path,
    )
    assert evil_path.stat().st_size == candidate.weights_size_bytes
    candidate_path.chmod(0o755)

    real_fd_sha256 = candidate_overlay_module._fd_sha256
    call_count = 0

    def _replace_after_apply_hash(fd: int) -> str:
        nonlocal call_count
        digest = real_fd_sha256(fd)
        if CANDIDATE_WEIGHTS_FILENAME not in os.readlink(f"/proc/self/fd/{fd}"):
            return digest
        call_count += 1
        # apply_candidate_overlay first reverifies (calls 1 and 2), then opens
        # the load descriptor. Replace its pathname only after call 3 hashed it.
        if call_count == 3:
            os.replace(evil_path, candidate.weights_path)
        return digest

    monkeypatch.setattr(
        candidate_overlay_module,
        "_fd_sha256",
        _replace_after_apply_hash,
    )
    target = _tiny_model(fill=-2.0)

    apply_candidate_overlay(target, candidate)

    torch.testing.assert_close(
        target.actor_critic.psi_model.action_header.weight,
        source.actor_critic.psi_model.action_header.weight,
    )
    assert call_count == 4


def test_candidate_rejects_invalid_formal_run(tmp_path: Path) -> None:
    source = _tiny_model(fill=1.0)
    candidate_path = _candidate_path(tmp_path)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, source, formal_run_valid=False)

    with pytest.raises(ValueError, match="formal run is invalid"):
        verify_candidate_overlay(
            candidate_path,
            expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
        )


def test_candidate_rejects_weights_not_saved_in_native_checkpoint(
    tmp_path: Path,
) -> None:
    source = _tiny_model(fill=1.0)
    candidate_path = _candidate_path(tmp_path)
    export_model_checkpoint(source, candidate_path, _export_context())
    _seal_formal_run(candidate_path, _tiny_model(fill=8.0))

    with pytest.raises(ValueError, match="differs from native checkpoint"):
        verify_candidate_overlay(
            candidate_path,
            expected_profile=LEGACY_VLA_BUNDLE_PROFILE,
        )
