# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit provenance for initialized and subsequently trained H70 actors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from alpagym_host.safetensors_identity import (
    ACTOR_STATE_HASH_SCHEMA,
    canonical_safetensors_subset_sha256,
)
from safetensors.torch import save

ACTOR_INITIALIZATION_ATTESTATION_SCHEMA = "videomimic_v9_actor_initialization.v1"
ACTOR_UPDATE_LINEAGE_SCHEMA = "videomimic_planner_actor_update_lineage.v1"
ACTOR_STATUS_INITIAL = "v9_initialized_unmodified"
ACTOR_STATUS_INITIAL_STD_CLAMPED = "v9_actor_mean_initialized_std_clamped"
ACTOR_STATUS_TRAINED = "trained_descendant"
ACTOR_STATUS_UNCHANGED = "actor_bit_identical_to_parent"
ACTOR_STATE_LINEAGE_FIELDS = (
    "actor_state_hash_schema",
    "current_actor_state_sha256",
    "parent_actor_state_sha256",
)
ACTION_STD_PARITY_SCOPE = "deterministic_action_mean_only"
ACTION_STD_ATTESTATION_FIELDS = (
    "source_v9_action_std_min",
    "source_v9_action_std_max",
    "applied_action_std_clamp_min",
    "applied_action_std_clamp_max",
    "exported_action_std_min",
    "exported_action_std_max",
    "source_v9_actor_parity_scope",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRAINING_STEP_PATTERN = re.compile(r"step_([1-9][0-9]*)")


def file_sha256(path: Path) -> str:
    """Hash one immutable bundle file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash one JSON mapping using the repository's canonical encoding."""

    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_actor_state_sha256(
    model: torch.nn.Module,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> str:
    """Hash exactly the model-owned PPO actor parameter group.

    The canonical identity is derived from each tensor's name, dtype, shape,
    and exact bytes, independently of safetensors header/layout ordering.
    Parameter ownership comes from
    ``ppo_parameter_groups()['actor']`` rather than name prefixes, so ``std``
    and optional actor context parameters are included while every critic
    parameter is excluded.
    """

    group_builder = getattr(model, "ppo_parameter_groups", None)
    if not callable(group_builder):
        raise TypeError("actor state hashing requires model.ppo_parameter_groups()")
    groups = group_builder()
    if not isinstance(groups, Mapping) or "actor" not in groups:
        raise TypeError("model PPO parameter groups do not identify an actor group")
    actor_parameters = tuple(groups["actor"])
    actor_parameter_ids = {id(parameter) for parameter in actor_parameters}
    if not actor_parameters or len(actor_parameter_ids) != len(actor_parameters):
        raise ValueError("model PPO actor parameter group is empty or contains aliases")
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    missing_names = actor_parameter_ids - set(names_by_id)
    if missing_names:
        raise ValueError("model PPO actor group contains unnamed parameters")
    actor_names = tuple(
        sorted(names_by_id[parameter_id] for parameter_id in actor_parameter_ids)
    )

    source_state = model.state_dict() if state_dict is None else state_dict
    missing_state = tuple(name for name in actor_names if name not in source_state)
    if missing_state:
        raise ValueError(
            "model state dict is missing PPO actor parameters: "
            + ", ".join(missing_state)
        )
    actor_state: dict[str, torch.Tensor] = {}
    for name in actor_names:
        tensor = source_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"actor state {name!r} is not a tensor")
        actor_state[name] = tensor.detach().to(device="cpu").contiguous()
    payload = save(actor_state)
    return canonical_safetensors_subset_sha256(payload, tensor_names=actor_names)


def model_weights_sha256(
    model: torch.nn.Module,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> str:
    """Hash the exact policy-native safetensors payload for the full model."""

    source_state = model.state_dict() if state_dict is None else state_dict
    weights: dict[str, torch.Tensor] = {}
    for name, tensor in source_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {name!r} is not a tensor")
        weights[name] = tensor.detach().to(device="cpu").contiguous()
    if not weights:
        raise ValueError("model state dict is empty")
    return hashlib.sha256(save(weights)).hexdigest()


def initialization_attestation(
    *,
    source_v9_checkpoint_sha256: str,
    source_v9_actor_parity_samples: int,
    source_v9_actor_parity_max_abs: float,
    planner_critic_seed: int,
    source_v9_action_std_min: float | None = None,
    source_v9_action_std_max: float | None = None,
    applied_action_std_clamp_min: float | None = None,
    applied_action_std_clamp_max: float | None = None,
    exported_action_std_min: float | None = None,
    exported_action_std_max: float | None = None,
) -> dict[str, Any]:
    """Build the immutable attestation for the actor's initialization event.

    The std fields are an optional extension of the v1 payload so existing
    bundles retain their canonical hash and remain loadable. New exports pass
    the complete extension and state explicitly that V9 parity covers the
    deterministic action mean while exploration std is clamped separately.
    """

    result = {
        "schema": ACTOR_INITIALIZATION_ATTESTATION_SCHEMA,
        "source_v9_checkpoint_sha256": source_v9_checkpoint_sha256,
        "source_v9_actor_parity_samples": int(source_v9_actor_parity_samples),
        "source_v9_actor_parity_max_abs": float(source_v9_actor_parity_max_abs),
        "planner_critic_seed": int(planner_critic_seed),
    }
    result.update(
        _action_std_attestation_extension(
            source_v9_action_std_min=source_v9_action_std_min,
            source_v9_action_std_max=source_v9_action_std_max,
            applied_action_std_clamp_min=applied_action_std_clamp_min,
            applied_action_std_clamp_max=applied_action_std_clamp_max,
            exported_action_std_min=exported_action_std_min,
            exported_action_std_max=exported_action_std_max,
        )
    )
    return result


def initial_actor_lineage(
    attestation: Mapping[str, Any],
    *,
    current_model_weights_sha256: str,
    current_actor_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Describe a step-0 actor, including optional actor-only identity.

    Omitting ``current_actor_state_sha256`` deliberately preserves the exact
    legacy-v1 mapping so already published bundles retain their canonical
    lineage hash.
    """

    if attestation.get("schema") != ACTOR_INITIALIZATION_ATTESTATION_SCHEMA:
        raise ValueError("actor initialization attestation schema is invalid")
    _require_sha256(current_model_weights_sha256, "current_model_weights_sha256")
    std_was_clamped = _has_action_std_attestation(attestation)
    lineage = {
        "schema": ACTOR_UPDATE_LINEAGE_SCHEMA,
        "current_actor_status": (
            ACTOR_STATUS_INITIAL_STD_CLAMPED
            if std_was_clamped
            else ACTOR_STATUS_INITIAL
        ),
        "initialization_attestation_sha256": canonical_json_sha256(attestation),
        "current_model_weights_sha256": current_model_weights_sha256,
        "parent_model_weights_sha256": None,
        "parent_update_lineage_sha256": None,
        "training_export_step": 0,
    }
    if current_actor_state_sha256 is not None:
        lineage.update(
            {
                "actor_state_hash_schema": ACTOR_STATE_HASH_SCHEMA,
                "current_actor_state_sha256": _require_sha256(
                    current_actor_state_sha256,
                    "current_actor_state_sha256",
                ),
                "parent_actor_state_sha256": None,
            }
        )
    return lineage


def trained_actor_lineage(
    attestation: Mapping[str, Any],
    previous_lineage: Mapping[str, Any],
    *,
    current_model_weights_sha256: str,
    training_export_step: int,
    current_actor_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Describe a training checkpoint and classify actual actor mutation.

    Legacy parents have no actor-only identity, so their descendants preserve
    the legacy ``trained_descendant`` mapping.  A parent with the complete
    optional extension is classified from bit-exact actor hashes.
    """

    previous = dict(previous_lineage)
    if attestation.get("schema") != ACTOR_INITIALIZATION_ATTESTATION_SCHEMA:
        raise ValueError("actor initialization attestation schema is invalid")
    if previous.get("schema") != ACTOR_UPDATE_LINEAGE_SCHEMA:
        raise ValueError("previous actor update lineage schema is invalid")
    attestation_sha256 = canonical_json_sha256(attestation)
    if previous.get("initialization_attestation_sha256") != attestation_sha256:
        raise ValueError(
            "previous actor lineage identifies another initialization attestation"
        )
    parent_weights = _require_sha256(
        previous.get("current_model_weights_sha256"),
        "parent current_model_weights_sha256",
    )
    if (
        not isinstance(training_export_step, int)
        or isinstance(training_export_step, bool)
        or training_export_step < 1
    ):
        raise ValueError("training_export_step must be a positive integer")
    _require_sha256(current_model_weights_sha256, "current_model_weights_sha256")
    extension_presence = tuple(name in previous for name in ACTOR_STATE_LINEAGE_FIELDS)
    if any(extension_presence) and not all(extension_presence):
        raise ValueError("previous actor state lineage extension is incomplete")
    actor_extension: dict[str, Any] = {}
    actor_status = ACTOR_STATUS_TRAINED
    if all(extension_presence):
        if previous["actor_state_hash_schema"] != ACTOR_STATE_HASH_SCHEMA:
            raise ValueError("previous actor state hash schema is invalid")
        parent_actor_state_sha256 = _require_sha256(
            previous["current_actor_state_sha256"],
            "parent current_actor_state_sha256",
        )
        if current_actor_state_sha256 is None:
            raise ValueError(
                "current_actor_state_sha256 is required by the parent lineage"
            )
        current_actor_state_sha256 = _require_sha256(
            current_actor_state_sha256,
            "current_actor_state_sha256",
        )
        actor_status = (
            ACTOR_STATUS_UNCHANGED
            if current_actor_state_sha256 == parent_actor_state_sha256
            else ACTOR_STATUS_TRAINED
        )
        actor_extension = {
            "actor_state_hash_schema": ACTOR_STATE_HASH_SCHEMA,
            "current_actor_state_sha256": current_actor_state_sha256,
            "parent_actor_state_sha256": parent_actor_state_sha256,
        }
    lineage = {
        "schema": ACTOR_UPDATE_LINEAGE_SCHEMA,
        "current_actor_status": actor_status,
        "initialization_attestation_sha256": attestation_sha256,
        "current_model_weights_sha256": current_model_weights_sha256,
        "parent_model_weights_sha256": parent_weights,
        "parent_update_lineage_sha256": canonical_json_sha256(previous),
        "training_export_step": training_export_step,
    }
    lineage.update(actor_extension)
    return lineage


def training_export_step(output_dir: Path) -> int:
    """Read the trainer step from the policy-native ``step_N`` directory."""

    match = _TRAINING_STEP_PATTERN.fullmatch(output_dir.name)
    if match is None:
        raise ValueError(
            "provenance-aware planner checkpoints require a step_N output directory"
        )
    return int(match.group(1))


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return digest


def _action_std_attestation_extension(
    *,
    source_v9_action_std_min: float | None,
    source_v9_action_std_max: float | None,
    applied_action_std_clamp_min: float | None,
    applied_action_std_clamp_max: float | None,
    exported_action_std_min: float | None,
    exported_action_std_max: float | None,
) -> dict[str, Any]:
    values = {
        "source_v9_action_std_min": source_v9_action_std_min,
        "source_v9_action_std_max": source_v9_action_std_max,
        "applied_action_std_clamp_min": applied_action_std_clamp_min,
        "applied_action_std_clamp_max": applied_action_std_clamp_max,
        "exported_action_std_min": exported_action_std_min,
        "exported_action_std_max": exported_action_std_max,
    }
    present = tuple(value is not None for value in values.values())
    if not any(present):
        return {}
    if not all(present):
        raise ValueError("action std initialization attestation must be complete")
    parsed = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) and value > 0.0 for value in parsed.values()):
        raise ValueError(
            "action std initialization attestation must be finite and positive"
        )
    source_min = parsed["source_v9_action_std_min"]
    source_max = parsed["source_v9_action_std_max"]
    clamp_min = parsed["applied_action_std_clamp_min"]
    clamp_max = parsed["applied_action_std_clamp_max"]
    exported_min = parsed["exported_action_std_min"]
    exported_max = parsed["exported_action_std_max"]
    if source_min > source_max or clamp_min > clamp_max or exported_min > exported_max:
        raise ValueError("action std initialization attestation has inverted bounds")
    expected_exported_min = min(max(source_min, clamp_min), clamp_max)
    expected_exported_max = min(max(source_max, clamp_min), clamp_max)
    if not math.isclose(
        exported_min, expected_exported_min, rel_tol=0.0, abs_tol=1.0e-6
    ) or not math.isclose(
        exported_max, expected_exported_max, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError("exported action std does not match the recorded clamp")
    return {
        **parsed,
        "source_v9_actor_parity_scope": ACTION_STD_PARITY_SCOPE,
    }


def _has_action_std_attestation(attestation: Mapping[str, Any]) -> bool:
    present = tuple(name in attestation for name in ACTION_STD_ATTESTATION_FIELDS)
    if not any(present):
        return False
    if not all(present):
        raise ValueError("action std initialization attestation must be complete")
    expected = _action_std_attestation_extension(
        source_v9_action_std_min=attestation["source_v9_action_std_min"],
        source_v9_action_std_max=attestation["source_v9_action_std_max"],
        applied_action_std_clamp_min=attestation["applied_action_std_clamp_min"],
        applied_action_std_clamp_max=attestation["applied_action_std_clamp_max"],
        exported_action_std_min=attestation["exported_action_std_min"],
        exported_action_std_max=attestation["exported_action_std_max"],
    )
    if (
        attestation["source_v9_actor_parity_scope"]
        != expected["source_v9_actor_parity_scope"]
    ):
        raise ValueError("source V9 actor parity scope is invalid")
    return True


__all__ = (
    "ACTOR_INITIALIZATION_ATTESTATION_SCHEMA",
    "ACTOR_STATE_HASH_SCHEMA",
    "ACTOR_STATE_LINEAGE_FIELDS",
    "ACTOR_STATUS_INITIAL",
    "ACTOR_STATUS_INITIAL_STD_CLAMPED",
    "ACTOR_STATUS_TRAINED",
    "ACTOR_STATUS_UNCHANGED",
    "ACTOR_UPDATE_LINEAGE_SCHEMA",
    "ACTION_STD_ATTESTATION_FIELDS",
    "ACTION_STD_PARITY_SCOPE",
    "canonical_actor_state_sha256",
    "canonical_json_sha256",
    "file_sha256",
    "initial_actor_lineage",
    "initialization_attestation",
    "model_weights_sha256",
    "trained_actor_lineage",
    "training_export_step",
)
