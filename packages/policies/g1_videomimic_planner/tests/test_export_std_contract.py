# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the native-V9 exploration std export contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from alpagym_host.safetensors_identity import planner_actor_state_sha256
from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    register_g1_mjlab_model,
)
from alpagym_g1_videomimic_planner.export import export_planner_checkpoint
from alpagym_g1_videomimic_planner.bundle import export_model_checkpoint
from alpagym_g1_videomimic_planner.model import (
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
)
from alpagym_g1_videomimic_planner.provenance import (
    ACTION_STD_ATTESTATION_FIELDS,
    ACTOR_STATE_HASH_SCHEMA,
    ACTOR_STATUS_INITIAL,
    ACTOR_STATUS_INITIAL_STD_CLAMPED,
    ACTOR_STATUS_TRAINED,
    ACTOR_STATUS_UNCHANGED,
    canonical_actor_state_sha256,
    initial_actor_lineage,
    initialization_attestation,
    trained_actor_lineage,
)
from safetensors.torch import load_file


def _source_checkpoint(path: Path) -> tuple[Path, torch.Tensor]:
    torch.manual_seed(17)
    register_g1_mjlab_model()
    model = G1MjlabActorCriticModel(G1MjlabConfig(hidden_dims=[16, 8], init_std=0.25))
    source_std = torch.linspace(0.01, 0.25, steps=model.std.numel())
    with torch.no_grad():
        model.std.copy_(source_std)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path, source_std


def test_export_clamps_std_before_step_zero_and_attests_source_and_result(
    tmp_path: Path,
) -> None:
    source_path, source_std = _source_checkpoint(tmp_path / "v9.pt")
    output_dir = tmp_path / "planner"

    export_planner_checkpoint(
        source_checkpoint=source_path,
        output_dir=output_dir,
        config=G1VideoMimicPlannerConfig(hidden_dims=[16, 8], init_std=0.25),
        critic_seed=3,
    )

    exported_std = load_file(str(output_dir / "model.safetensors"))["std"]
    torch.testing.assert_close(
        exported_std,
        source_std.clamp(min=0.05, max=0.15),
        rtol=0.0,
        atol=0.0,
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    attestation = config["actor_initialization_attestation"]
    for name in ACTION_STD_ATTESTATION_FIELDS:
        assert config[name] == attestation[name]
    assert attestation["source_v9_action_std_min"] == pytest.approx(0.01)
    assert attestation["source_v9_action_std_max"] == pytest.approx(0.25)
    assert attestation["applied_action_std_clamp_min"] == pytest.approx(0.05)
    assert attestation["applied_action_std_clamp_max"] == pytest.approx(0.15)
    assert attestation["exported_action_std_min"] == pytest.approx(0.05)
    assert attestation["exported_action_std_max"] == pytest.approx(0.15)
    assert attestation["source_v9_actor_parity_scope"] == (
        "deterministic_action_mean_only"
    )
    assert config["actor_update_lineage"]["current_actor_status"] == (
        ACTOR_STATUS_INITIAL_STD_CLAMPED
    )
    lineage = config["actor_update_lineage"]
    assert lineage["actor_state_hash_schema"] == ACTOR_STATE_HASH_SCHEMA
    assert lineage["parent_actor_state_sha256"] is None
    state_dict = load_file(str(output_dir / "model.safetensors"))
    planner = G1VideoMimicPlannerActorCriticModel(
        G1VideoMimicPlannerConfig(hidden_dims=[16, 8], init_std=0.25)
    )
    assert lineage["current_actor_state_sha256"] == canonical_actor_state_sha256(
        planner,
        state_dict=state_dict,
    )
    assert lineage["current_actor_state_sha256"] == planner_actor_state_sha256(
        output_dir / "model.safetensors"
    )


def test_legacy_v1_attestation_keeps_legacy_hash_shape_and_status() -> None:
    attestation = initialization_attestation(
        source_v9_checkpoint_sha256="a" * 64,
        source_v9_actor_parity_samples=16,
        source_v9_actor_parity_max_abs=0.0,
        planner_critic_seed=0,
    )

    assert set(attestation) == {
        "schema",
        "source_v9_checkpoint_sha256",
        "source_v9_actor_parity_samples",
        "source_v9_actor_parity_max_abs",
        "planner_critic_seed",
    }
    lineage = initial_actor_lineage(
        attestation,
        current_model_weights_sha256="b" * 64,
    )
    assert lineage["current_actor_status"] == ACTOR_STATUS_INITIAL
    assert set(lineage) == {
        "schema",
        "current_actor_status",
        "initialization_attestation_sha256",
        "current_model_weights_sha256",
        "parent_model_weights_sha256",
        "parent_update_lineage_sha256",
        "training_export_step",
    }


def test_actor_state_hash_uses_exact_optimizer_actor_group() -> None:
    torch.manual_seed(23)
    planner = G1VideoMimicPlannerActorCriticModel(
        G1VideoMimicPlannerConfig(hidden_dims=[16, 8])
    )
    baseline = canonical_actor_state_sha256(planner)

    with torch.no_grad():
        planner.critic_attention.add_(1.0)
    assert canonical_actor_state_sha256(planner) == baseline

    with torch.no_grad():
        planner.actor_attention.add_(1.0)
    actor_changed = canonical_actor_state_sha256(planner)
    assert actor_changed != baseline

    with torch.no_grad():
        planner.std.add_(0.01)
    assert canonical_actor_state_sha256(planner) != actor_changed


def test_actor_lineage_classifies_hash_equality_and_keeps_legacy_parent() -> None:
    attestation = initialization_attestation(
        source_v9_checkpoint_sha256="a" * 64,
        source_v9_actor_parity_samples=16,
        source_v9_actor_parity_max_abs=0.0,
        planner_critic_seed=0,
    )
    initial = initial_actor_lineage(
        attestation,
        current_model_weights_sha256="b" * 64,
        current_actor_state_sha256="c" * 64,
    )
    critic_only = trained_actor_lineage(
        attestation,
        initial,
        current_model_weights_sha256="d" * 64,
        current_actor_state_sha256="c" * 64,
        training_export_step=1,
    )
    assert critic_only["current_actor_status"] == ACTOR_STATUS_UNCHANGED
    assert critic_only["current_actor_state_sha256"] == "c" * 64
    assert critic_only["parent_actor_state_sha256"] == "c" * 64

    actor_update = trained_actor_lineage(
        attestation,
        critic_only,
        current_model_weights_sha256="e" * 64,
        current_actor_state_sha256="f" * 64,
        training_export_step=2,
    )
    assert actor_update["current_actor_status"] == ACTOR_STATUS_TRAINED
    assert actor_update["parent_actor_state_sha256"] == "c" * 64

    legacy_initial = initial_actor_lineage(
        attestation,
        current_model_weights_sha256="b" * 64,
    )
    legacy_step = trained_actor_lineage(
        attestation,
        legacy_initial,
        current_model_weights_sha256="d" * 64,
        current_actor_state_sha256="c" * 64,
        training_export_step=1,
    )
    assert legacy_step["current_actor_status"] == ACTOR_STATUS_TRAINED
    assert "current_actor_state_sha256" not in legacy_step


def test_policy_checkpoint_marks_critic_only_step_actor_unchanged(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    planner = G1VideoMimicPlannerActorCriticModel(
        G1VideoMimicPlannerConfig(hidden_dims=[16, 8])
    )
    attestation = initialization_attestation(
        source_v9_checkpoint_sha256="a" * 64,
        source_v9_actor_parity_samples=16,
        source_v9_actor_parity_max_abs=0.0,
        planner_critic_seed=0,
    )
    initial_actor_hash = canonical_actor_state_sha256(planner)
    planner.config.actor_initialization_attestation = attestation
    planner.config.actor_update_lineage = initial_actor_lineage(
        attestation,
        current_model_weights_sha256="b" * 64,
        current_actor_state_sha256=initial_actor_hash,
    )
    with torch.no_grad():
        planner.critic_attention.add_(0.25)

    output_dir = tmp_path / "step_1"
    export_model_checkpoint(planner, output_dir)

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    lineage = config["actor_update_lineage"]
    assert lineage["current_actor_status"] == ACTOR_STATUS_UNCHANGED
    assert lineage["current_actor_state_sha256"] == initial_actor_hash
    assert lineage["parent_actor_state_sha256"] == initial_actor_hash


def test_export_rejects_invalid_std_clamp_before_writing(tmp_path: Path) -> None:
    source_path, _ = _source_checkpoint(tmp_path / "v9.pt")
    output_dir = tmp_path / "planner"

    with pytest.raises(ValueError, match="0 < min_action_std <= max_action_std"):
        export_planner_checkpoint(
            source_checkpoint=source_path,
            output_dir=output_dir,
            config=G1VideoMimicPlannerConfig(hidden_dims=[16, 8]),
            min_action_std=0.2,
            max_action_std=0.1,
        )

    assert not output_dir.exists()
