# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy-owned replay collation for VLA's ragged Qwen3-VL inputs."""

from __future__ import annotations

from dataclasses import replace

import torch

from alpagym_runtime.replay import TrainerReplayData, TrainerReplayDataBatch


def collate_vla_replay_samples(
    samples: list[TrainerReplayData],
    *,
    pad_token_id: int,
) -> TrainerReplayDataBatch:
    """Collate frozen rollout observations without resampling visual history.

    Rollout owns BATS selection and stores the selected images in chronological
    order inside the Qwen processor tensors. This function preserves that order:
    token rows are right-padded, while image metadata and pixel-patch rows are
    packed sample-by-sample. Explicit offsets recover every sample and image
    boundary without assuming equal history lengths.

    Args:
        samples: Single-step replay samples. Each ``model_inputs`` mapping must
            contain ``input_ids[S]``, ``attention_mask[S]``,
            ``pixel_values[P,F]``, and ``image_grid_thw[I,3]``. Compressed-history
            inputs additionally contain both ``effective_image_grid_thw[I,3]``
            and ``visual_pool_factors[I]``.
        pad_token_id: Tokenizer pad id written only to the right-padding region.

    Returns:
        A replay minibatch whose rectangular fields have leading dimension
        ``B``. Visual rows remain packed: ``pixel_values`` has ``sum(P)`` rows
        and image metadata has ``sum(I)`` rows. ``sequence_lengths``,
        ``image_offsets``, ``patch_offsets``, and ``image_patch_counts`` describe
        the packed boundaries.
    """
    if not samples:
        raise ValueError("VLA replay collation requires at least one sample")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("VLA pad_token_id must be an integer")
    if pad_token_id < 0:
        raise ValueError("VLA pad_token_id must be non-negative")

    required_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
    pooling_keys = {"effective_image_grid_thw", "visual_pool_factors"}
    packed_metadata_keys = {
        "sequence_lengths",
        "image_counts",
        "image_offsets",
        "image_patch_counts",
        "patch_counts",
        "patch_offsets",
    }
    first_keys = set(samples[0].model_inputs)
    if any(set(sample.model_inputs) != first_keys for sample in samples):
        raise ValueError("VLA replay model input keys differ across samples")
    missing = sorted(required_keys - first_keys)
    if missing:
        raise ValueError(
            "VLA replay model inputs are missing required fields: " + ", ".join(missing)
        )
    pooling_present = pooling_keys & first_keys
    if pooling_present and pooling_present != pooling_keys:
        raise ValueError(
            "VLA effective_image_grid_thw and visual_pool_factors must be "
            "present together"
        )
    collisions = sorted(packed_metadata_keys & first_keys)
    if collisions:
        raise ValueError(
            "VLA replay inputs contain collator-owned fields: " + ", ".join(collisions)
        )

    input_ids: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    pixel_values: list[torch.Tensor] = []
    image_grids: list[torch.Tensor] = []
    effective_image_grids: list[torch.Tensor] = []
    visual_pool_factors: list[torch.Tensor] = []
    per_image_patch_counts: list[torch.Tensor] = []

    first_input_dtype: torch.dtype | None = None
    first_attention_dtype: torch.dtype | None = None
    first_pixel_dtype: torch.dtype | None = None
    first_pixel_width: int | None = None
    first_grid_dtype: torch.dtype | None = None
    first_effective_grid_dtype: torch.dtype | None = None
    first_pool_dtype: torch.dtype | None = None
    token_device: torch.device | None = None
    pixel_device: torch.device | None = None
    grid_device: torch.device | None = None

    for sample_index, sample in enumerate(samples):
        ids = sample.model_inputs["input_ids"]
        attention = sample.model_inputs["attention_mask"]
        pixels = sample.model_inputs["pixel_values"]
        grid = sample.model_inputs["image_grid_thw"]
        if not all(
            isinstance(value, torch.Tensor) for value in (ids, attention, pixels, grid)
        ):
            raise TypeError("VLA ragged replay fields must be tensors")
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError(
                f"VLA input_ids for sample {sample_index} must be non-empty [S]"
            )
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("VLA input_ids must use an integer dtype")
        if bool((ids < 0).any()):
            raise ValueError("VLA input_ids must be non-negative")
        if attention.ndim != 1 or attention.shape != ids.shape:
            raise ValueError(
                f"VLA attention_mask for sample {sample_index} must match input_ids"
            )
        if attention.dtype not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("VLA attention_mask must use a bool or integer dtype")
        if not bool(((attention == 0) | (attention == 1)).all()):
            raise ValueError("VLA attention_mask values must be zero or one")
        if pixels.ndim != 2 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
            raise ValueError(
                f"VLA pixel_values for sample {sample_index} must be non-empty [P,F]"
            )
        if not torch.is_floating_point(pixels):
            raise TypeError("VLA pixel_values must use a floating-point dtype")
        if not bool(torch.isfinite(pixels).all()):
            raise ValueError("VLA pixel_values contain non-finite values")
        if grid.ndim != 2 or grid.shape[0] == 0 or grid.shape[1] != 3:
            raise ValueError(
                f"VLA image_grid_thw for sample {sample_index} must be non-empty [I,3]"
            )
        if grid.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("VLA image_grid_thw must use an integer dtype")
        if not bool((grid > 0).all()):
            raise ValueError("VLA image_grid_thw entries must be positive")

        image_patch_counts = grid.to(dtype=torch.int64).prod(dim=1)
        expected_pixel_rows = int(image_patch_counts.sum().item())
        if pixels.shape[0] != expected_pixel_rows:
            raise ValueError(
                f"VLA pixel_values rows for sample {sample_index} "
                f"({pixels.shape[0]}) do not match image_grid_thw patch rows "
                f"({expected_pixel_rows})"
            )

        if first_input_dtype is None:
            first_input_dtype = ids.dtype
            first_attention_dtype = attention.dtype
            first_pixel_dtype = pixels.dtype
            first_pixel_width = int(pixels.shape[1])
            first_grid_dtype = grid.dtype
            token_device = ids.device
            pixel_device = pixels.device
            grid_device = grid.device
        elif (
            ids.dtype != first_input_dtype
            or attention.dtype != first_attention_dtype
            or pixels.dtype != first_pixel_dtype
            or pixels.shape[1] != first_pixel_width
            or grid.dtype != first_grid_dtype
            or ids.device != token_device
            or attention.device != token_device
            or pixels.device != pixel_device
            or grid.device != grid_device
        ):
            raise ValueError(
                "VLA ragged replay tensor dtype, device, or pixel width "
                "differs across samples"
            )
        if attention.device != ids.device:
            raise ValueError("VLA input_ids and attention_mask must share a device")

        if pooling_keys <= first_keys:
            effective_grid = sample.model_inputs["effective_image_grid_thw"]
            pool_factors = sample.model_inputs["visual_pool_factors"]
            if not isinstance(effective_grid, torch.Tensor) or not isinstance(
                pool_factors, torch.Tensor
            ):
                raise TypeError("VLA visual pooling metadata must be tensors")
            if effective_grid.shape != grid.shape:
                raise ValueError(
                    "VLA effective_image_grid_thw must match image_grid_thw rows"
                )
            if effective_grid.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(
                    "VLA effective_image_grid_thw must use an integer dtype"
                )
            if not bool((effective_grid > 0).all()):
                raise ValueError(
                    "VLA effective_image_grid_thw entries must be positive"
                )
            if pool_factors.ndim != 1 or pool_factors.shape[0] != grid.shape[0]:
                raise ValueError(
                    "VLA visual_pool_factors must have one entry per image"
                )
            if pool_factors.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("VLA visual_pool_factors must use an integer dtype")
            if not bool((pool_factors > 0).all()):
                raise ValueError("VLA visual_pool_factors must be positive")
            if sample_index == 0:
                first_effective_grid_dtype = effective_grid.dtype
                first_pool_dtype = pool_factors.dtype
            elif (
                effective_grid.dtype != first_effective_grid_dtype
                or pool_factors.dtype != first_pool_dtype
                or effective_grid.device != grid_device
                or pool_factors.device != grid_device
            ):
                raise ValueError(
                    "VLA visual pooling metadata dtype or device differs across samples"
                )
            if (
                effective_grid.device != grid.device
                or pool_factors.device != grid.device
            ):
                raise ValueError(
                    "VLA image grids and visual_pool_factors must share a device"
                )
            effective_image_grids.append(effective_grid)
            visual_pool_factors.append(pool_factors)

        input_ids.append(ids)
        attention_masks.append(attention)
        pixel_values.append(pixels)
        image_grids.append(grid)
        per_image_patch_counts.append(image_patch_counts.to(device=grid.device))

    assert first_input_dtype is not None
    assert first_attention_dtype is not None
    assert token_device is not None
    assert grid_device is not None
    max_sequence_length = max(int(ids.numel()) for ids in input_ids)
    padded_input_ids = torch.full(
        (len(samples), max_sequence_length),
        pad_token_id,
        dtype=first_input_dtype,
        device=token_device,
    )
    padded_attention = torch.zeros(
        (len(samples), max_sequence_length),
        dtype=first_attention_dtype,
        device=token_device,
    )
    for row, (ids, attention) in enumerate(zip(input_ids, attention_masks)):
        length = int(ids.numel())
        padded_input_ids[row, :length] = ids
        padded_attention[row, :length] = attention

    sequence_lengths = torch.tensor(
        [ids.numel() for ids in input_ids], dtype=torch.int64, device=token_device
    )
    image_counts = torch.tensor(
        [grid.shape[0] for grid in image_grids],
        dtype=torch.int64,
        device=grid_device,
    )
    patch_counts = torch.tensor(
        [pixels.shape[0] for pixels in pixel_values],
        dtype=torch.int64,
        device=grid_device,
    )
    image_offsets = _exclusive_offsets(image_counts)
    patch_offsets = _exclusive_offsets(patch_counts)

    ragged_batch: dict[str, torch.Tensor] = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention,
        "sequence_lengths": sequence_lengths,
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_thw": torch.cat(image_grids, dim=0),
        "image_counts": image_counts,
        "image_offsets": image_offsets,
        "image_patch_counts": torch.cat(per_image_patch_counts, dim=0),
        "patch_counts": patch_counts,
        "patch_offsets": patch_offsets,
    }
    if pooling_keys <= first_keys:
        ragged_batch["effective_image_grid_thw"] = torch.cat(
            effective_image_grids, dim=0
        )
        ragged_batch["visual_pool_factors"] = torch.cat(visual_pool_factors, dim=0)

    ragged_input_keys = required_keys | pooling_keys
    rectangular_samples = [
        replace(
            sample,
            model_inputs={
                key: value
                for key, value in sample.model_inputs.items()
                if key not in ragged_input_keys
            },
        )
        for sample in samples
    ]
    primitive_widths = [
        int(sample.training_signal.primitive_rewards.shape[1])
        for sample in rectangular_samples
        if sample.training_signal.primitive_rewards is not None
    ]
    if primitive_widths:
        if len(primitive_widths) != len(rectangular_samples):
            raise ValueError(
                "VLA primitive reward signals mix present and missing rows"
            )
        max_primitive_width = max(primitive_widths)
        padded_samples: list[TrainerReplayData] = []
        for sample in rectangular_samples:
            primitive_rewards = sample.training_signal.primitive_rewards
            primitive_reward_mask = sample.training_signal.primitive_reward_mask
            assert primitive_rewards is not None
            assert primitive_reward_mask is not None
            pad_width = max_primitive_width - int(primitive_rewards.shape[1])
            if pad_width:
                primitive_rewards = torch.cat(
                    (
                        primitive_rewards,
                        torch.zeros(
                            (primitive_rewards.shape[0], pad_width),
                            dtype=primitive_rewards.dtype,
                            device=primitive_rewards.device,
                        ),
                    ),
                    dim=1,
                )
                primitive_reward_mask = torch.cat(
                    (
                        primitive_reward_mask,
                        torch.zeros(
                            (primitive_reward_mask.shape[0], pad_width),
                            dtype=torch.bool,
                            device=primitive_reward_mask.device,
                        ),
                    ),
                    dim=1,
                )
            padded_samples.append(
                replace(
                    sample,
                    training_signal=replace(
                        sample.training_signal,
                        primitive_rewards=primitive_rewards,
                        primitive_reward_mask=primitive_reward_mask,
                    ),
                )
            )
        rectangular_samples = padded_samples
    rectangular_batch = TrainerReplayDataBatch.stack(rectangular_samples)
    return replace(
        rectangular_batch,
        model_inputs={**rectangular_batch.model_inputs, **ragged_batch},
    )


def _exclusive_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Return ``[0, cumsum(counts)]`` on the input tensor's device."""
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=counts.device),
            counts.to(dtype=torch.int64).cumsum(dim=0),
        ],
        dim=0,
    )
