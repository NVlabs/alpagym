# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import Dataset


def repeat_scene_ids(scene_ids: list[str], repetitions: int) -> list[str]:
    """Repeat each runtime scene as an independent Cosmos prompt work item."""
    return [scene_id for scene_id in scene_ids for _ in range(repetitions)]


class AlpagymSceneDataset(Dataset):
    """Dataset exposing scene IDs as Cosmos prompt work items."""

    def __init__(self, scene_ids: list[str]) -> None:
        """Store the scene IDs Cosmos should enumerate."""
        self._scene_ids = scene_ids

    def __len__(self) -> int:
        """Return the number of scene prompts."""
        return len(self._scene_ids)

    def __getitem__(self, index: int) -> str:
        """Return the scene ID for a prompt index."""
        return self._scene_ids[index]
