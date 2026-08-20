# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the disk rollout-transport writer.

Asserts the disk writer satisfies its role protocol and that an egressed artifact
reads back through ``read_episode_json`` (the trainer-side disk read). Encoding
and integrity edge cases live in test_disk.py.
"""

import json
from pathlib import Path

import torch
from alpagym_runtime.transport import DiskEpisodeWriter
from alpagym_runtime.transport.base import EpisodeWriter
from alpagym_runtime.transport.disk import read_episode_json
from alpagym_runtime.types import EpisodeOutput, PolicyOutput


def _make_episode() -> EpisodeOutput:
    """Build a minimal EpisodeOutput with one PolicyOutput tensor leaf."""
    return EpisodeOutput(
        scene_id="scene_alpha",
        session_uuid="session_zero",
        num_steps=1,
        policy_outputs=(
            PolicyOutput(
                chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                chosen_dt_us=torch.tensor([0], dtype=torch.int64),
            ),
        ),
    )


def test_disk_round_trips_episode_output(tmp_path: Path) -> None:
    """An episode written via the writer reads back through ``read_episode_json``."""
    episode = _make_episode()
    handle = DiskEpisodeWriter(tmp_path).write(episode)
    loaded = read_episode_json(handle)
    assert loaded.scene_id == episode.scene_id
    assert loaded.session_uuid == episode.session_uuid
    assert loaded.num_steps == episode.num_steps
    assert torch.equal(
        loaded.policy_outputs[0].chosen_xyz,
        episode.policy_outputs[0].chosen_xyz,
    )


def test_disk_write_places_artifact_under_configured_dir(tmp_path: Path) -> None:
    """The disk writer lands each artifact as a file under its configured dir."""
    handle = DiskEpisodeWriter(tmp_path).write(_make_episode())
    artifact = Path(handle)
    assert artifact.is_file()
    assert tmp_path in artifact.parents


def test_disk_release_unlinks_artifact_and_is_idempotent(tmp_path: Path) -> None:
    """Disk release removes manifest and sidecar and tolerates duplicate cleanup."""
    writer = DiskEpisodeWriter(tmp_path)
    handle = writer.write(_make_episode())
    artifact = Path(handle)
    descriptor = json.loads(artifact.read_text(encoding="utf-8"))["tensor_sidecar"]
    sidecar = artifact.parent / descriptor["filename"]
    assert sidecar.is_file()

    writer.release(handle, "discarded")
    writer.release(handle, "discarded")

    assert not artifact.exists()
    assert not sidecar.exists()


def test_disk_writer_satisfies_role_protocol(tmp_path: Path) -> None:
    """The disk writer structurally matches the rollout-writer role protocol."""
    assert isinstance(DiskEpisodeWriter(tmp_path), EpisodeWriter)
