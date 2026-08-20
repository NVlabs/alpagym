"""Tests for Wenhao's policy-owned ragged visual replay collation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from alpagym_g1_wenhao_vla.replay_collator import (
    collate_wenhao_replay_samples,
)
from alpagym_runtime.replay import TrainerReplayData, TrainingSignal


def _sample(
    *,
    rollout_id: str,
    input_ids: list[int],
    grids: list[list[int]],
    pixel_markers: list[float],
    pool_factors: list[int] | None = None,
) -> TrainerReplayData:
    grid = torch.tensor(grids, dtype=torch.int64)
    patch_rows = int(grid.prod(dim=1).sum().item())
    assert len(pixel_markers) == patch_rows
    model_inputs: dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.int64),
        "pixel_values": torch.tensor(pixel_markers, dtype=torch.float32)[
            :, None
        ].repeat(1, 2),
        "image_grid_thw": grid,
        "flow_chain": torch.full((2, 3, 4), float(len(input_ids))),
    }
    if pool_factors is not None:
        model_inputs["effective_image_grid_thw"] = torch.ones_like(grid)
        model_inputs["visual_pool_factors"] = torch.tensor(
            pool_factors, dtype=torch.int64
        )
    return TrainerReplayData(
        model_inputs=model_inputs,
        training_signal=TrainingSignal(
            old_logprobs=torch.tensor([0.25], dtype=torch.float32),
            is_padding=torch.zeros(1, dtype=torch.bool),
            rewards=torch.tensor([1.0], dtype=torch.float32),
        ),
        rollout_id=rollout_id,
        weight_version=torch.tensor(3, dtype=torch.int64),
    )


def test_collator_preserves_ragged_history_order_and_boundaries() -> None:
    first = _sample(
        rollout_id="rollout-a",
        input_ids=[101, 102, 103],
        grids=[[1, 1, 2], [1, 1, 1]],
        pixel_markers=[10.0, 11.0, 20.0],
        pool_factors=[2, 1],
    )
    second = _sample(
        rollout_id="rollout-b",
        input_ids=[201, 202, 203, 204, 205],
        grids=[[1, 2, 2]],
        pixel_markers=[30.0, 31.0, 32.0, 33.0],
        pool_factors=[1],
    )

    batch = collate_wenhao_replay_samples([first, second], pad_token_id=999)
    inputs = batch.model_inputs

    torch.testing.assert_close(
        inputs["input_ids"],
        torch.tensor(
            [[101, 102, 103, 999, 999], [201, 202, 203, 204, 205]],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.int64),
    )
    torch.testing.assert_close(inputs["sequence_lengths"], torch.tensor([3, 5]))
    torch.testing.assert_close(inputs["image_counts"], torch.tensor([2, 1]))
    torch.testing.assert_close(inputs["image_offsets"], torch.tensor([0, 2, 3]))
    torch.testing.assert_close(inputs["image_patch_counts"], torch.tensor([2, 1, 4]))
    torch.testing.assert_close(inputs["patch_counts"], torch.tensor([3, 4]))
    torch.testing.assert_close(inputs["patch_offsets"], torch.tensor([0, 3, 7]))
    torch.testing.assert_close(
        inputs["pixel_values"][:, 0],
        torch.tensor([10.0, 11.0, 20.0, 30.0, 31.0, 32.0, 33.0]),
    )
    torch.testing.assert_close(
        inputs["image_grid_thw"],
        torch.tensor([[1, 1, 2], [1, 1, 1], [1, 2, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(inputs["visual_pool_factors"], torch.tensor([2, 1, 1]))
    assert inputs["flow_chain"].shape == (2, 2, 3, 4)
    assert inputs["return_log_prob"] is True
    assert batch.rollout_ids == ("rollout-a", "rollout-b")
    torch.testing.assert_close(batch.training_signal.rewards, torch.tensor([1.0, 1.0]))

    # Packed tensors are a replay snapshot, not aliases of mutable rollout rows.
    first.model_inputs["pixel_values"].zero_()
    first.model_inputs["image_grid_thw"].fill_(9)
    torch.testing.assert_close(
        inputs["pixel_values"][:, 0],
        torch.tensor([10.0, 11.0, 20.0, 30.0, 31.0, 32.0, 33.0]),
    )
    torch.testing.assert_close(
        inputs["image_grid_thw"],
        torch.tensor([[1, 1, 2], [1, 1, 1], [1, 2, 2]], dtype=torch.int64),
    )


def test_collator_right_pads_mixed_primitive_reward_durations() -> None:
    first = _sample(
        rollout_id="rollout-a",
        input_ids=[101],
        grids=[[1, 1, 1]],
        pixel_markers=[10.0],
    )
    second = _sample(
        rollout_id="rollout-b",
        input_ids=[201],
        grids=[[1, 1, 1]],
        pixel_markers=[20.0],
    )
    first = replace(
        first,
        training_signal=replace(
            first.training_signal,
            primitive_rewards=torch.arange(25, dtype=torch.float32).reshape(1, 25),
            primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
            duration_ticks=torch.tensor([25], dtype=torch.int64),
            actor_valid=torch.tensor([True], dtype=torch.bool),
        ),
    )
    second = replace(
        second,
        training_signal=replace(
            second.training_signal,
            primitive_rewards=torch.arange(37, dtype=torch.float32).reshape(1, 37),
            primitive_reward_mask=torch.ones((1, 37), dtype=torch.bool),
            duration_ticks=torch.tensor([37], dtype=torch.int64),
            actor_valid=torch.tensor([False], dtype=torch.bool),
        ),
    )

    signal = collate_wenhao_replay_samples(
        [first, second], pad_token_id=999
    ).training_signal
    assert signal.primitive_rewards.shape == (2, 37)
    torch.testing.assert_close(signal.primitive_rewards[0, 25:], torch.zeros(12))
    torch.testing.assert_close(
        signal.primitive_reward_mask[0],
        torch.tensor([True] * 25 + [False] * 12),
    )
    assert bool(signal.primitive_reward_mask[1].all())
    torch.testing.assert_close(signal.duration_ticks, torch.tensor([25, 37]))
    torch.testing.assert_close(signal.actor_valid, torch.tensor([True, False]))


def test_collator_rejects_pixel_rows_that_do_not_match_original_grid() -> None:
    sample = _sample(
        rollout_id="rollout-a",
        input_ids=[101],
        grids=[[1, 1, 2]],
        pixel_markers=[10.0, 11.0],
    )
    sample.model_inputs["pixel_values"] = torch.ones((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="do not match image_grid_thw patch rows"):
        collate_wenhao_replay_samples([sample], pad_token_id=999)


def test_collator_requires_complete_pooling_metadata() -> None:
    sample = _sample(
        rollout_id="rollout-a",
        input_ids=[101],
        grids=[[1, 1, 1]],
        pixel_markers=[10.0],
    )
    sample.model_inputs["effective_image_grid_thw"] = torch.ones(
        (1, 3), dtype=torch.int64
    )

    with pytest.raises(ValueError, match="must be present together"):
        collate_wenhao_replay_samples([sample], pad_token_id=999)
