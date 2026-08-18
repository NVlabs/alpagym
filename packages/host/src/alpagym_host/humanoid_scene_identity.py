# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze SceneStore identity into one immutable humanoid run config."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from alpagym_host.config import RunConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def freeze_humanoid_scene_fingerprints(config: RunConfig) -> RunConfig:
    """Snapshot every selected scene fingerprint and inject it into both peers.

    The returned config is the artifact written for the run.  Policy workers
    receive the same canonical JSON map through ``bundle_config`` that Wizard
    forwards to AlpaSim dynamics/controller registration.
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
    return replace(
        config,
        alpasim=replace(
            config.alpasim,
            humanoid=replace(
                humanoid,
                expected_scene_fingerprints=fingerprints,
            ),
        ),
        policy=replace(
            config.policy,
            model=replace(config.policy.model, bundle_config=bundle_config),
        ),
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
