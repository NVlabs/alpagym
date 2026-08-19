# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alpagym_host.config import HumanoidExecutionProfile
from alpagym_host.humanoid_scene_identity import (
    _canonical_json_sha256,
    _current_actor_attestation,
    _ensure_run_owned_actor_state_identity,
    _snapshot_policy_model_bundle,
    snapshot_scene_fingerprints,
    validate_frozen_policy_model_bundle,
)
from alpagym_host.safetensors_identity import planner_actor_state_sha256
from safetensors.torch import save_file


def _write_scene(store: Path, scene_id: str, digest: str) -> None:
    scene_root = store / "scenes" / scene_id
    scene_root.mkdir(parents=True)
    (scene_root / "manifest.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "identity": {"scene_content_sha256": digest},
            }
        ),
        encoding="utf-8",
    )


def test_snapshot_scene_fingerprints_is_closed_over_selected_scenes(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path, "scene_b", "b" * 64)
    _write_scene(tmp_path, "scene_a", "a" * 64)

    snapshot = snapshot_scene_fingerprints(tmp_path, ("scene_b", "scene_a"))

    assert snapshot == {"scene_b": "b" * 64, "scene_a": "a" * 64}


def test_snapshot_scene_fingerprints_rejects_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path, "scene", "A" * 64)

    with pytest.raises(ValueError, match="lowercase SHA256"):
        snapshot_scene_fingerprints(tmp_path, ("scene",))


def _write_current_actor_bundle(
    bundle_dir: Path,
    *,
    std_clamped: bool = False,
) -> Path:
    """Write a minimal H70 actor bundle with valid initialization lineage."""
    bundle_dir.mkdir(parents=True)
    weights_path = bundle_dir / "model.safetensors"
    save_file(
        {
            "actor.0.weight": torch.tensor([[1.0]], dtype=torch.float32),
            "std": torch.tensor([0.15], dtype=torch.float32),
            "critic.0.weight": torch.tensor([[2.0]], dtype=torch.float32),
        },
        str(weights_path),
    )
    initialization = {
        "schema": "videomimic_v9_actor_initialization.v1",
        "source_v9_checkpoint_sha256": "a" * 64,
        "source_v9_actor_parity_samples": 16,
        "source_v9_actor_parity_max_abs": 0.0,
        "planner_critic_seed": 7,
    }
    if std_clamped:
        initialization.update(
            {
                "source_v9_action_std_min": 0.01,
                "source_v9_action_std_max": 0.25,
                "applied_action_std_clamp_min": 0.05,
                "applied_action_std_clamp_max": 0.15,
                "exported_action_std_min": 0.05,
                "exported_action_std_max": 0.15,
                "source_v9_actor_parity_scope": "deterministic_action_mean_only",
            }
        )
    initialization_sha256 = hashlib.sha256(
        json.dumps(initialization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    config = {
        "model_type": "g1_videomimic_planner_actor_critic",
        "shadow_action_steps": 69,
        "checkpoint_path": weights_path.name,
        **{key: value for key, value in initialization.items() if key != "schema"},
        "actor_initialization_attestation": initialization,
        "actor_update_lineage": {
            "schema": "videomimic_planner_actor_update_lineage.v1",
            "current_actor_status": (
                "v9_actor_mean_initialized_std_clamped"
                if std_clamped
                else "v9_initialized_unmodified"
            ),
            "initialization_attestation_sha256": initialization_sha256,
            "current_model_weights_sha256": weights_sha256,
            "parent_model_weights_sha256": None,
            "parent_update_lineage_sha256": None,
            "training_export_step": 0,
        },
    }
    (bundle_dir / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return bundle_dir


def test_current_actor_snapshot_is_closed_over_original_bundle_bytes(
    tmp_path: Path,
) -> None:
    """A frozen run reads its actor snapshot after the authored bundle changes."""
    source = _write_current_actor_bundle(tmp_path / "source")
    destination = tmp_path / "run" / "artifacts" / "policy_model_bundle"
    original_weights = (source / "model.safetensors").read_bytes()

    snapshot, manifest_json, manifest_sha256 = _snapshot_policy_model_bundle(
        source,
        destination=destination,
    )
    (source / "model.safetensors").write_bytes(b"externally-mutated")

    assert snapshot == destination.resolve()
    assert (snapshot / "model.safetensors").read_bytes() == original_weights
    assert (snapshot / "alpagym_bundle_manifest.json").read_text(encoding="utf-8") == (
        manifest_json + "\n"
    )
    assert hashlib.sha256(manifest_json.encode()).hexdigest() == manifest_sha256
    assert (
        _current_actor_attestation(snapshot)["actor_update_lineage"][
            "current_actor_status"
        ]
        == "v9_initialized_unmodified"
    )


def test_current_actor_accepts_complete_std_clamp_attestation(tmp_path: Path) -> None:
    bundle = _write_current_actor_bundle(tmp_path / "actor", std_clamped=True)

    config = _current_actor_attestation(bundle)

    initialization = config["actor_initialization_attestation"]
    assert initialization["source_v9_action_std_min"] == pytest.approx(0.01)
    assert initialization["source_v9_action_std_max"] == pytest.approx(0.25)
    assert initialization["applied_action_std_clamp_min"] == pytest.approx(0.05)
    assert initialization["applied_action_std_clamp_max"] == pytest.approx(0.15)
    assert config["actor_update_lineage"]["current_actor_status"] == (
        "v9_actor_mean_initialized_std_clamped"
    )


def test_current_actor_accepts_actor_identical_critic_only_lineage(
    tmp_path: Path,
) -> None:
    bundle = _write_current_actor_bundle(tmp_path / "actor")
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lineage = config["actor_update_lineage"]
    actor_state_sha256 = planner_actor_state_sha256(bundle / "model.safetensors")
    lineage.update(
        {
            "current_actor_status": "actor_bit_identical_to_parent",
            "parent_model_weights_sha256": "b" * 64,
            "parent_update_lineage_sha256": "c" * 64,
            "training_export_step": 1,
            "actor_state_hash_schema": (
                "videomimic_planner_actor_state.safetensors.v1"
            ),
            "current_actor_state_sha256": actor_state_sha256,
            "parent_actor_state_sha256": actor_state_sha256,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    validated = _current_actor_attestation(bundle)
    assert validated["actor_update_lineage"]["current_actor_status"] == (
        "actor_bit_identical_to_parent"
    )

    lineage["current_actor_status"] = "trained_descendant"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with actor-state hashes"):
        _current_actor_attestation(bundle)


def test_run_owned_legacy_step_zero_is_upgraded_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = _write_current_actor_bundle(tmp_path / "source")
    destination = tmp_path / "run" / "artifacts" / "policy_model_bundle"
    snapshot, _legacy_manifest, _legacy_manifest_sha = _snapshot_policy_model_bundle(
        source,
        destination=destination,
    )

    config, manifest_json, manifest_sha256 = _ensure_run_owned_actor_state_identity(
        snapshot
    )

    source_lineage = json.loads((source / "config.json").read_text(encoding="utf-8"))[
        "actor_update_lineage"
    ]
    assert "current_actor_state_sha256" not in source_lineage
    lineage = config["actor_update_lineage"]
    assert lineage["current_actor_state_sha256"] == planner_actor_state_sha256(
        snapshot / "model.safetensors"
    )
    assert lineage["parent_actor_state_sha256"] is None
    assert hashlib.sha256(manifest_json.encode()).hexdigest() == manifest_sha256
    assert (snapshot / "alpagym_bundle_manifest.json").read_text(encoding="utf-8") == (
        manifest_json + "\n"
    )


def test_current_actor_rejects_frozen_completion_identity(tmp_path: Path) -> None:
    """The only accepted fake-plan actor is the current H70 policy."""
    bundle = _write_current_actor_bundle(tmp_path / "actor")
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["completion_policy_sha256"] = "b" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must not identify a frozen completion"):
        _current_actor_attestation(bundle)


def test_resolved_run_revalidates_frozen_actor_snapshot(tmp_path: Path) -> None:
    """A resumed run rejects mutations to its run-owned current actor."""

    source = _write_current_actor_bundle(tmp_path / "source")
    destination = tmp_path / "run" / "artifacts" / "policy_model_bundle"
    snapshot, manifest_json, manifest_sha256 = _snapshot_policy_model_bundle(
        source,
        destination=destination,
    )
    actor_config = _current_actor_attestation(snapshot)
    initialization = actor_config["actor_initialization_attestation"]
    lineage = actor_config["actor_update_lineage"]
    config = SimpleNamespace(
        alpasim=SimpleNamespace(
            humanoid=SimpleNamespace(
                execution_profile=HumanoidExecutionProfile.motion_reference
            )
        ),
        artifact_paths=SimpleNamespace(policy_model_bundle_dir=snapshot),
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(snapshot),
                bundle_config={
                    "current_actor_bundle_manifest_json": manifest_json,
                    "current_actor_bundle_manifest_sha256": manifest_sha256,
                    "actor_initialization_attestation_sha256": (
                        _canonical_json_sha256(initialization)
                    ),
                    "actor_update_lineage_sha256": _canonical_json_sha256(lineage),
                    "current_actor_weights_sha256": lineage[
                        "current_model_weights_sha256"
                    ],
                },
            )
        ),
    )

    validate_frozen_policy_model_bundle(config)
    (snapshot / "model.safetensors").write_bytes(b"mutated-after-freeze")

    with pytest.raises(ValueError, match="differ from the frozen manifest"):
        validate_frozen_policy_model_bundle(config)
