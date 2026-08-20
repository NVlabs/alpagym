# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from alpagym_g1_wenhao_vla.normalization import (
    WENHAO_ACTION_DIM,
    WENHAO_ACTION_ROWS,
    WENHAO_STATE_DIM,
    WenhaoQ99Normalizer,
)


def _normalizer() -> WenhaoQ99Normalizer:
    return WenhaoQ99Normalizer(
        state_q01=torch.zeros(WENHAO_STATE_DIM),
        state_q99=torch.full((WENHAO_STATE_DIM,), 2.0),
        action_q01=torch.full((WENHAO_ACTION_DIM,), -2.0),
        action_q99=torch.full((WENHAO_ACTION_DIM,), 2.0),
    )


def test_q99_state_and_action_normalization_match_training_formula() -> None:
    normalizer = _normalizer()
    state = torch.stack(
        (
            torch.zeros(WENHAO_STATE_DIM),
            torch.ones(WENHAO_STATE_DIM),
            torch.full((WENHAO_STATE_DIM,), 3.0),
        )
    )
    action = torch.tensor([-2.0, 0.0, 2.0, 4.0]).repeat(WENHAO_ACTION_DIM // 4 + 1)[
        :WENHAO_ACTION_DIM
    ]

    normalized_state = normalizer.normalize_state(state)
    normalized_action = normalizer.normalize_action(action)

    torch.testing.assert_close(normalized_state[0], torch.full_like(state[0], -1.0))
    torch.testing.assert_close(normalized_state[1], torch.zeros_like(state[1]))
    torch.testing.assert_close(normalized_state[2], torch.ones_like(state[2]))
    torch.testing.assert_close(
        normalized_action[:4], torch.tensor([-1.0, 0.0, 1.0, 1.0])
    )


def test_wire_clips_after_preserving_density_latent() -> None:
    normalizer = _normalizer()
    latent = torch.zeros((1, WENHAO_ACTION_ROWS, WENHAO_ACTION_DIM))
    latent[0, 0, :3] = torch.tensor([-1.5, 0.25, 2.0])

    wire = normalizer.to_wire(latent)

    assert wire.density_latent is latent
    torch.testing.assert_close(
        wire.clipped_normalized[0, 0, :3], torch.tensor([-1.0, 0.25, 1.0])
    )
    torch.testing.assert_close(
        wire.denormalized[0, 0, :3], torch.tensor([-2.0, 0.5, 2.0])
    )


def test_q99_normalizer_rejects_near_degenerate_stats() -> None:
    try:
        WenhaoQ99Normalizer(
            state_q01=torch.zeros(WENHAO_STATE_DIM),
            state_q99=torch.zeros(WENHAO_STATE_DIM),
            action_q01=torch.zeros(WENHAO_ACTION_DIM),
            action_q99=torch.ones(WENHAO_ACTION_DIM),
        )
    except ValueError as error:
        assert "near-degenerate" in str(error)
    else:
        raise AssertionError("degenerate q01/q99 bounds were accepted")
