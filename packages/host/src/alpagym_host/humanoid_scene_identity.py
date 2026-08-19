# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze SceneStore identity into one immutable humanoid run config."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

from alpagym_host.config import HumanoidExecutionProfile, RunConfig
from alpagym_host.safetensors_identity import (
    ACTOR_STATE_HASH_SCHEMA,
    planner_actor_state_sha256,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_MANIFEST_FILENAME = "alpagym_bundle_manifest.json"
_BUNDLE_MANIFEST_FORMAT = "alpagym.policy_bundle_manifest.v1"
_SOURCE_V9_SHA256_KEY = "source_v9_checkpoint_sha256"
_INITIALIZATION_ATTESTATION_SCHEMA = "videomimic_v9_actor_initialization.v1"
_UPDATE_LINEAGE_SCHEMA = "videomimic_planner_actor_update_lineage.v1"
_ACTOR_STATUS_INITIAL = "v9_initialized_unmodified"
_ACTOR_STATUS_INITIAL_STD_CLAMPED = "v9_actor_mean_initialized_std_clamped"
_ACTOR_STATUS_TRAINED = "trained_descendant"
_ACTOR_STATUS_UNCHANGED = "actor_bit_identical_to_parent"
_ACTOR_STATE_LINEAGE_FIELDS = (
    "actor_state_hash_schema",
    "current_actor_state_sha256",
    "parent_actor_state_sha256",
)
_ACTION_STD_ATTESTATION_FIELDS = (
    "source_v9_action_std_min",
    "source_v9_action_std_max",
    "applied_action_std_clamp_min",
    "applied_action_std_clamp_max",
    "exported_action_std_min",
    "exported_action_std_max",
    "source_v9_actor_parity_scope",
)


def freeze_humanoid_scene_fingerprints(config: RunConfig) -> RunConfig:
    """Freeze scene and planner-side environment identities into both peers.

    The returned config is the artifact written for the run.  Policy workers
    receive the same canonical JSON map through ``bundle_config`` that Wizard
    forwards to AlpaSim dynamics/controller registration. Motion-reference
    runs also receive an atomic, run-owned snapshot of the attested current
    H70 actor, so later edits to the authored checkpoint cannot change a run.

    Args:
        config: Fully resolved host run configuration.

    Returns:
        A copy containing frozen scene identities and, for motion-reference
        execution, the run-owned current-actor path and provenance manifest.
    """
    humanoid = config.alpasim.humanoid
    if humanoid is None:
        return config
    scene_ids = config.dataset.scene_ids
    if not scene_ids:
        raise ValueError("humanoid scene fingerprint freeze requires dataset.scene_ids")
    fingerprints = snapshot_scene_fingerprints(
        Path(humanoid.scene_store_path),
        tuple(scene_ids),
    )
    fingerprint_json = json.dumps(
        fingerprints,
        sort_keys=True,
        separators=(",", ":"),
    )
    bundle_config = dict(config.policy.model.bundle_config)
    bundle_config["humanoid_repo_path"] = str(
        Path(humanoid.repo_path).expanduser().resolve()
    )
    bundle_config["scene_store_path"] = str(
        Path(humanoid.scene_store_path).expanduser().resolve()
    )
    bundle_config["expected_scene_fingerprints_json"] = fingerprint_json
    policy_model_path = config.policy.model.path
    reference_frame_count = humanoid.reference_frame_count
    if humanoid.execution_profile is HumanoidExecutionProfile.motion_reference:
        if config.policy.model.kind != "g1_videomimic_planner":
            raise ValueError(
                "motion_reference requires policy.model.kind='g1_videomimic_planner'"
            )
        planner_mode = str(bundle_config.get("planner_mode", "shadow_rollout"))
        if planner_mode != "shadow_rollout":
            raise ValueError(
                "VideoMimic fake plans require planner_mode='shadow_rollout' so all "
                "H70 actions come from the current policy"
            )
        if any("completion" in str(key) for key in bundle_config):
            raise ValueError(
                "VideoMimic fake plans must not configure a frozen completion policy"
            )
        if config.alpasim.wizard_args.control_timestep_us != 500_000:
            raise ValueError(
                "current-policy H70 planning requires a 500000us/K25 outer step"
            )
        if config.policy.model.step_dt_us != 500_000:
            raise ValueError(
                "current-policy H70 planning requires policy.model.step_dt_us=500000"
            )

        bundle_config["planner_mode"] = "shadow_rollout"
        source_bundle = Path(policy_model_path).expanduser()
        actor_config = _current_actor_attestation(source_bundle)
        (
            snapshot_path,
            manifest_json,
            manifest_sha256,
        ) = _snapshot_policy_model_bundle(
            source_bundle,
            destination=config.artifact_paths.policy_model_bundle_dir,
        )
        snapshot_actor_config = _current_actor_attestation(snapshot_path)
        if snapshot_actor_config != actor_config:
            raise ValueError("current actor config changed while being snapshotted")
        if config.cosmos.train.train_policy.grpo_optimization_iterations > 0:
            (
                actor_config,
                manifest_json,
                manifest_sha256,
            ) = _ensure_run_owned_actor_state_identity(snapshot_path)
        else:
            actor_config = snapshot_actor_config
        initialization = actor_config["actor_initialization_attestation"]
        lineage = actor_config["actor_update_lineage"]
        bundle_config["current_actor_bundle_manifest_json"] = manifest_json
        bundle_config["current_actor_bundle_manifest_sha256"] = manifest_sha256
        bundle_config["actor_initialization_attestation_sha256"] = (
            _canonical_json_sha256(initialization)
        )
        bundle_config["actor_update_lineage_sha256"] = _canonical_json_sha256(lineage)
        bundle_config["current_actor_weights_sha256"] = str(
            lineage["current_model_weights_sha256"]
        )
        if "current_actor_state_sha256" in lineage:
            bundle_config["current_actor_state_sha256"] = str(
                lineage["current_actor_state_sha256"]
            )
        policy_model_path = str(snapshot_path)
        reference_frame_count = 70
    return replace(
        config,
        alpasim=replace(
            config.alpasim,
            humanoid=replace(
                humanoid,
                expected_scene_fingerprints=fingerprints,
                reference_frame_count=reference_frame_count,
            ),
        ),
        policy=replace(
            config.policy,
            model=replace(
                config.policy.model,
                path=policy_model_path,
                bundle_config=bundle_config,
            ),
        ),
    )


def validate_frozen_policy_model_bundle(config: RunConfig) -> None:
    """Revalidate the run-owned H70 actor snapshot before execution or resume.

    Initial run preparation copies the actor into the run directory. Resolved
    configs bypass that preparation step, so this check recomputes every
    load-bearing identity instead of trusting the serialized manifest fields.
    """

    humanoid = config.alpasim.humanoid
    if (
        humanoid is None
        or humanoid.execution_profile is not HumanoidExecutionProfile.motion_reference
    ):
        return
    bundle_dir = Path(config.policy.model.path).expanduser().resolve(strict=True)
    expected_dir = config.artifact_paths.policy_model_bundle_dir.expanduser().resolve()
    if bundle_dir != expected_dir:
        raise ValueError(
            "motion_reference policy.model.path must identify the run-owned "
            "policy_model_bundle snapshot"
        )
    bundle_config = config.policy.model.bundle_config
    manifest_json = bundle_config.get("current_actor_bundle_manifest_json")
    manifest_sha256 = bundle_config.get("current_actor_bundle_manifest_sha256")
    if not isinstance(manifest_json, str) or not manifest_json:
        raise ValueError("current actor snapshot has no canonical bundle manifest")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256.fullmatch(manifest_sha256) is None
        or hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != manifest_sha256
    ):
        raise ValueError("current actor snapshot manifest digest is invalid")
    source_files = _policy_model_files(bundle_dir)
    actual_manifest_json, actual_manifest_sha256 = _policy_bundle_manifest(
        bundle_dir,
        source_files,
    )
    if (
        actual_manifest_json != manifest_json
        or actual_manifest_sha256 != manifest_sha256
    ):
        raise ValueError("current actor snapshot files differ from the frozen manifest")
    sidecar = bundle_dir / _BUNDLE_MANIFEST_FILENAME
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar.read_text(encoding="utf-8") != manifest_json + "\n"
    ):
        raise ValueError(
            "current actor snapshot manifest sidecar is missing or changed"
        )

    actor_config = _current_actor_attestation(bundle_dir)
    initialization = actor_config["actor_initialization_attestation"]
    lineage = actor_config["actor_update_lineage"]
    expected_identities = {
        "actor_initialization_attestation_sha256": _canonical_json_sha256(
            initialization
        ),
        "actor_update_lineage_sha256": _canonical_json_sha256(lineage),
        "current_actor_weights_sha256": str(lineage["current_model_weights_sha256"]),
    }
    if "current_actor_state_sha256" in lineage:
        expected_identities["current_actor_state_sha256"] = str(
            lineage["current_actor_state_sha256"]
        )
    for key, expected in expected_identities.items():
        if bundle_config.get(key) != expected:
            raise ValueError(f"current actor snapshot {key} differs from its bytes")


def _file_sha256(path: Path) -> str:
    """Hash one identity-owned file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_actor_attestation(bundle_dir: Path) -> dict[str, Any]:
    """Validate one current H70 actor's V9 initialization and update lineage."""
    if not bundle_dir.is_dir():
        raise ValueError(f"current actor bundle must be a directory: {bundle_dir}")
    config_path = bundle_dir / "config.json"
    try:
        raw: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read current actor config: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("current actor config.json must contain an object")
    if raw.get("model_type") != "g1_videomimic_planner_actor_critic":
        raise ValueError("current actor config has the wrong model_type")
    if raw.get("shadow_action_steps") != 69:
        raise ValueError("current actor config must encode the H70/H69 contract")
    initialization = raw.get("actor_initialization_attestation")
    lineage = raw.get("actor_update_lineage")
    if not isinstance(initialization, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError(
            "current actor requires explicit initialization attestation "
            "and current update lineage"
        )
    if any(
        "completion" in str(key)
        for provenance in (raw, initialization, lineage)
        for key in provenance
    ):
        raise ValueError("current actor config must not identify a frozen completion")
    if initialization.get("schema") != _INITIALIZATION_ATTESTATION_SCHEMA:
        raise ValueError("current actor initialization schema is invalid")

    initialization_fields = (
        _SOURCE_V9_SHA256_KEY,
        "source_v9_actor_parity_samples",
        "source_v9_actor_parity_max_abs",
        "planner_critic_seed",
    )
    std_field_presence = tuple(
        key in initialization or key in raw for key in _ACTION_STD_ATTESTATION_FIELDS
    )
    if any(std_field_presence):
        if not all(
            key in initialization and key in raw
            for key in _ACTION_STD_ATTESTATION_FIELDS
        ):
            raise ValueError("current actor action std attestation is incomplete")
        initialization_fields = (
            *initialization_fields,
            *_ACTION_STD_ATTESTATION_FIELDS,
        )
    for key in initialization_fields:
        if raw.get(key) != initialization.get(key):
            raise ValueError(
                f"current actor field {key!r} differs from its "
                "initialization attestation"
            )
    source_digest = initialization.get(_SOURCE_V9_SHA256_KEY)
    if not isinstance(source_digest, str) or _SHA256.fullmatch(source_digest) is None:
        raise ValueError(
            "current actor initialization must contain lowercase "
            f"{_SOURCE_V9_SHA256_KEY}"
        )
    source_samples = initialization.get("source_v9_actor_parity_samples")
    source_max_abs = initialization.get("source_v9_actor_parity_max_abs")
    critic_seed = initialization.get("planner_critic_seed")
    if (
        not isinstance(source_samples, int)
        or isinstance(source_samples, bool)
        or source_samples != 16
        or not isinstance(source_max_abs, (int, float))
        or isinstance(source_max_abs, bool)
        or not math.isfinite(float(source_max_abs))
        or float(source_max_abs) != 0.0
    ):
        raise ValueError("current actor initialization differs from its raw V9 source")
    if (
        not isinstance(critic_seed, int)
        or isinstance(critic_seed, bool)
        or not 0 <= critic_seed < 2**63
    ):
        raise ValueError("current actor must record a valid planner_critic_seed")
    std_was_clamped = bool(any(std_field_presence))
    if std_was_clamped:
        _validate_action_std_attestation(initialization)

    initialization_sha256 = _canonical_json_sha256(initialization)
    if lineage.get("schema") != _UPDATE_LINEAGE_SCHEMA or (
        lineage.get("initialization_attestation_sha256") != initialization_sha256
    ):
        raise ValueError("current actor update lineage is invalid")
    checkpoint_name = raw.get("checkpoint_path")
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise ValueError("current actor checkpoint_path must be a root file")
    weights_path = bundle_dir / checkpoint_name
    current_weights_sha256 = lineage.get("current_model_weights_sha256")
    if (
        not weights_path.is_file()
        or weights_path.is_symlink()
        or not isinstance(current_weights_sha256, str)
        or _SHA256.fullmatch(current_weights_sha256) is None
        or _file_sha256(weights_path) != current_weights_sha256
    ):
        raise ValueError("current actor weight lineage does not match bytes")

    actor_extension_presence = tuple(
        name in lineage for name in _ACTOR_STATE_LINEAGE_FIELDS
    )
    if any(actor_extension_presence) and not all(actor_extension_presence):
        raise ValueError("current actor state lineage extension is incomplete")
    has_actor_state_identity = all(actor_extension_presence)
    current_actor_state: str | None = None
    if has_actor_state_identity:
        if lineage.get("actor_state_hash_schema") != ACTOR_STATE_HASH_SCHEMA:
            raise ValueError("current actor state hash schema is invalid")
        candidate = lineage.get("current_actor_state_sha256")
        if not isinstance(candidate, str) or _SHA256.fullmatch(candidate) is None:
            raise ValueError("current actor state identity is invalid")
        current_actor_state = candidate
        if planner_actor_state_sha256(weights_path) != current_actor_state:
            raise ValueError("current actor state identity does not match bytes")

    status = lineage.get("current_actor_status")
    training_step = lineage.get("training_export_step")
    parent_weights = lineage.get("parent_model_weights_sha256")
    parent_lineage = lineage.get("parent_update_lineage_sha256")
    if status in {_ACTOR_STATUS_INITIAL, _ACTOR_STATUS_INITIAL_STD_CLAMPED}:
        if (status == _ACTOR_STATUS_INITIAL_STD_CLAMPED) != std_was_clamped:
            raise ValueError(
                "initial actor status disagrees with its action std attestation"
            )
        if (
            training_step != 0
            or parent_weights is not None
            or parent_lineage is not None
            or (
                has_actor_state_identity
                and lineage.get("parent_actor_state_sha256") is not None
            )
        ):
            raise ValueError("initial V9 actor has an inconsistent update lineage")
    elif status in {_ACTOR_STATUS_TRAINED, _ACTOR_STATUS_UNCHANGED}:
        if (
            not isinstance(training_step, int)
            or isinstance(training_step, bool)
            or training_step < 1
            or not isinstance(parent_weights, str)
            or _SHA256.fullmatch(parent_weights) is None
            or not isinstance(parent_lineage, str)
            or _SHA256.fullmatch(parent_lineage) is None
        ):
            raise ValueError("trained actor has an inconsistent update lineage")
        if has_actor_state_identity:
            parent_actor_state = lineage.get("parent_actor_state_sha256")
            if (
                not isinstance(parent_actor_state, str)
                or _SHA256.fullmatch(parent_actor_state) is None
            ):
                raise ValueError("trained actor has no valid parent actor identity")
            expected_status = (
                _ACTOR_STATUS_UNCHANGED
                if current_actor_state == parent_actor_state
                else _ACTOR_STATUS_TRAINED
            )
            if status != expected_status:
                raise ValueError(
                    "current actor status disagrees with actor-state hashes"
                )
        elif status != _ACTOR_STATUS_TRAINED:
            raise ValueError(
                "legacy actor lineage cannot claim bit-identical actor state"
            )
    else:
        raise ValueError("current actor status is unknown")
    return dict(raw)


def _ensure_run_owned_actor_state_identity(
    bundle_dir: Path,
) -> tuple[dict[str, Any], str, str]:
    """Upgrade one run-owned legacy step-0 snapshot to actor-only identity.

    Published legacy bundles remain byte-for-byte untouched. The run-owned
    copy is the durable step-0 parent used by future policy-native exports, so
    enriching that copy lets a critic-only first update prove that the actor is
    unchanged instead of inferring actor training from a full-model digest.
    Legacy trained checkpoints are accepted as-is because their historical
    parent actor bytes are unavailable and must not be invented.
    """

    bundle_dir = bundle_dir.resolve(strict=True)
    config = _current_actor_attestation(bundle_dir)
    lineage = dict(config["actor_update_lineage"])
    actor_extension_presence = tuple(
        name in lineage for name in _ACTOR_STATE_LINEAGE_FIELDS
    )
    if not any(actor_extension_presence) and lineage.get("training_export_step") == 0:
        weights_path = bundle_dir / str(config["checkpoint_path"])
        lineage.update(
            {
                "actor_state_hash_schema": ACTOR_STATE_HASH_SCHEMA,
                "current_actor_state_sha256": planner_actor_state_sha256(weights_path),
                "parent_actor_state_sha256": None,
            }
        )
        config["actor_update_lineage"] = lineage
        config_path = bundle_dir / "config.json"
        temporary_config_path = bundle_dir / ".config.json.actor-state.tmp"
        temporary_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_config_path, config_path)
        config = _current_actor_attestation(bundle_dir)

    source_files = _policy_model_files(bundle_dir)
    manifest_json, manifest_sha256 = _policy_bundle_manifest(bundle_dir, source_files)
    _write_bundle_manifest(bundle_dir, manifest_json)
    return config, manifest_json, manifest_sha256


def _validate_action_std_attestation(initialization: Mapping[str, Any]) -> None:
    """Validate the optional v1 std-clamp extension used by new H70 exports."""

    numeric_fields = _ACTION_STD_ATTESTATION_FIELDS[:-1]
    try:
        values = {name: float(initialization[name]) for name in numeric_fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("current actor action std attestation is invalid") from exc
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("current actor action std attestation is invalid")
    source_min = values["source_v9_action_std_min"]
    source_max = values["source_v9_action_std_max"]
    clamp_min = values["applied_action_std_clamp_min"]
    clamp_max = values["applied_action_std_clamp_max"]
    exported_min = values["exported_action_std_min"]
    exported_max = values["exported_action_std_max"]
    if source_min > source_max or clamp_min > clamp_max or exported_min > exported_max:
        raise ValueError("current actor action std attestation has inverted bounds")
    expected_min = min(max(source_min, clamp_min), clamp_max)
    expected_max = min(max(source_max, clamp_min), clamp_max)
    if (
        not math.isclose(exported_min, expected_min, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(exported_max, expected_max, rel_tol=0.0, abs_tol=1.0e-6)
        or initialization["source_v9_actor_parity_scope"]
        != "deterministic_action_mean_only"
    ):
        raise ValueError(
            "current actor action std attestation disagrees with its clamp"
        )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash one provenance mapping using the policy export's canonical encoding."""
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_policy_model_bundle(
    source: Path,
    *,
    destination: Path,
) -> tuple[Path, str, str]:
    """Atomically snapshot the load-bearing actor files and their content manifest."""
    source = source.resolve(strict=True)
    destination = destination.resolve()
    source_files = _policy_model_files(source)
    manifest_json, manifest_sha256 = _policy_bundle_manifest(source, source_files)

    if source == destination:
        _write_bundle_manifest(destination, manifest_json)
        return destination, manifest_json, manifest_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"policy model snapshot target already exists: {destination}")

    temporary = Path(
        tempfile.mkdtemp(prefix=".policy_model_bundle_", dir=destination.parent)
    )
    try:
        for relative_path in source_files:
            shutil.copyfile(source / relative_path, temporary / relative_path)
        copied_manifest_json, copied_manifest_sha256 = _policy_bundle_manifest(
            temporary,
            source_files,
        )
        if (
            copied_manifest_json != manifest_json
            or copied_manifest_sha256 != manifest_sha256
        ):
            raise ValueError("policy model bundle changed while being snapshotted")
        _write_bundle_manifest(temporary, manifest_json)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination, manifest_json, manifest_sha256


def _policy_model_files(bundle_dir: Path) -> tuple[Path, ...]:
    """Select the root-level config and weight files consumed by HF loading."""
    config_path = bundle_dir / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError(f"policy bundle requires a regular config.json: {bundle_dir}")
    weights: list[Path] = []
    for path in bundle_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in {
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        } or any(
            fnmatch(path.name, pattern)
            for pattern in ("model*.safetensors", "pytorch_model*.bin")
        ):
            weights.append(Path(path.name))
    if not weights:
        raise ValueError(
            "current actor bundle has no supported safetensors/bin weights: "
            f"{bundle_dir}"
        )
    return (Path("config.json"), *tuple(sorted(weights, key=lambda item: item.name)))


def _policy_bundle_manifest(
    bundle_dir: Path,
    relative_paths: tuple[Path, ...],
) -> tuple[str, str]:
    """Return canonical per-file identity and its aggregate SHA256."""
    files = []
    for relative_path in relative_paths:
        path = bundle_dir / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"policy bundle file is missing or not regular: {path}")
        files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
        )
    manifest_json = json.dumps(
        {"format": _BUNDLE_MANIFEST_FORMAT, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    )
    return manifest_json, hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def _write_bundle_manifest(bundle_dir: Path, manifest_json: str) -> None:
    """Write the host-owned manifest beside its immutable actor snapshot."""
    (bundle_dir / _BUNDLE_MANIFEST_FILENAME).write_text(
        manifest_json + "\n",
        encoding="utf-8",
    )


def snapshot_scene_fingerprints(
    scene_store_path: Path,
    scene_ids: tuple[str, ...],
) -> dict[str, str]:
    """Read the canonical content digest from each selected scene manifest."""
    root = scene_store_path.expanduser().resolve(strict=True)
    scenes_root = (root / "scenes").resolve(strict=True)
    fingerprints: dict[str, str] = {}
    for scene_id in scene_ids:
        if not scene_id or scene_id in fingerprints:
            raise ValueError(
                "dataset.scene_ids must contain unique non-empty scene IDs"
            )
        scene_root = (scenes_root / scene_id).resolve(strict=True)
        if scenes_root not in scene_root.parents:
            raise ValueError(f"scene_id escapes SceneStore root: {scene_id!r}")
        manifest_path = scene_root / "manifest.json"
        try:
            raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read SceneStore manifest for scene {scene_id!r}: {manifest_path}"
            ) from exc
        if not isinstance(raw, Mapping) or str(raw.get("scene_id", "")) != scene_id:
            raise ValueError(
                f"SceneStore manifest scene_id does not match directory {scene_id!r}"
            )
        identity = raw.get("identity")
        digest = (
            identity.get("scene_content_sha256")
            if isinstance(identity, Mapping)
            else None
        )
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"scene {scene_id!r} identity.scene_content_sha256 is not lowercase SHA256"
            )
        fingerprints[scene_id] = digest
    return fingerprints
