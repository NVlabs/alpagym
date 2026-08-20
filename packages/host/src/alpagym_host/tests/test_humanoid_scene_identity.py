# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from alpagym_host.humanoid_scene_identity import snapshot_scene_fingerprints


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
