# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared replay payload contracts."""

import pytest
import torch
from alpagym_runtime.replay import (
    ActionSelection,
    PolicyReplayData,
    TrainerReplayData,
    TrainerReplayDataBatch,
    TrainingSignal,
    parse_policy_replay_data,
)


def test_action_selection_selects_candidate_dims() -> None:
    """Candidate selection indexes repo ``[N, K]`` dims."""
    value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    selection = ActionSelection(set_ix=1, sample_ix=2)

    selected = value[selection.set_ix, selection.sample_ix]

    assert torch.equal(selected, value[1, 2])


def test_action_selection_rejects_negative_indices() -> None:
    """Candidate selection indices must not wrap from the end."""
    with pytest.raises(ValueError, match="indices must be non-negative"):
        ActionSelection(set_ix=-1, sample_ix=0)


def test_policy_replay_data_round_trips_opaque_payload_and_common_old_logprob() -> None:
    """Shared replay preserves opaque payload data without family-specific parsing."""
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(-2.5, dtype=torch.float32),
        payload={
            "model_input": {
                "camera_frames": torch.zeros((2, 3, 4, 5), dtype=torch.uint8),
                "camera_indices": torch.tensor([1, 1], dtype=torch.int64),
                "relative_timestamps": torch.tensor([-1, 0], dtype=torch.int64),
                "ego_history_xyz": torch.zeros((1, 2, 3), dtype=torch.float32),
                "ego_history_rot": torch.eye(3).expand(1, 2, 3, 3).clone(),
                "route_xy": None,
            },
            "tokenized_data": {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                "attention_mask": torch.ones(3, dtype=torch.bool),
                "position_ids": None,
                "attention_mask_4d": None,
                "labels_mask": torch.tensor([False, True, True]),
            },
            "ego_future_xyz": torch.zeros((1, 4, 3), dtype=torch.float32),
            "ego_future_rot": torch.eye(3).expand(1, 4, 3, 3).clone(),
            "token_logprob_count": torch.tensor(2, dtype=torch.int64),
        },
    )

    parsed = parse_policy_replay_data(replay.to_dict())

    assert parsed.model_family == "alpamayo_r1"
    assert parsed.payload_schema == "alpamayo_r1.trajectory.v1"
    assert isinstance(parsed.payload, dict)
    assert "old_logprob" not in parsed.payload
    assert torch.equal(torch.as_tensor(parsed.old_logprob), torch.tensor(-2.5))
    torch.testing.assert_close(
        parsed.payload["model_input"]["camera_frames"],
        replay.payload["model_input"]["camera_frames"],
    )
    torch.testing.assert_close(
        parsed.payload["tokenized_data"]["input_ids"],
        torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    assert torch.equal(
        parsed.payload["token_logprob_count"], torch.tensor(2, dtype=torch.int64)
    )


def test_policy_replay_data_to_dict_preserves_non_leaf_tensors() -> None:
    """Live rollout replay tensors can carry autograd history until artifact serialization."""
    source = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    non_leaf = source * 2.0
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=non_leaf[0],
        payload={"selected_future": non_leaf},
    )

    serialized = replay.to_dict()

    assert serialized["old_logprob"] is replay.old_logprob
    assert serialized["payload"]["selected_future"] is non_leaf


def test_training_signal_requires_flat_aligned_rows() -> None:
    """Trainer signals are flattened ``[BT]`` tensors with aligned lengths."""
    with pytest.raises(ValueError, match="is_padding length"):
        TrainingSignal(
            old_logprobs=torch.zeros(2),
            is_padding=torch.zeros(3, dtype=torch.bool),
        )


def test_training_signal_requires_bool_padding_mask() -> None:
    """Padding masks stay boolean so valid-row masking is unambiguous."""
    with pytest.raises(ValueError, match="is_padding dtype must be bool"):
        TrainingSignal(
            old_logprobs=torch.zeros(2),
            is_padding=torch.zeros(2),
        )


def test_training_signal_requires_bool_actor_valid_mask() -> None:
    """Actor eligibility cannot be represented by ambiguous numeric weights."""
    with pytest.raises(ValueError, match="actor_valid dtype must be bool"):
        TrainingSignal(
            old_logprobs=torch.zeros(2),
            is_padding=torch.zeros(2, dtype=torch.bool),
            actor_valid=torch.ones(2),
        )


def test_training_signal_requires_actor_reward_mask_to_be_bool_subset() -> None:
    """The actor view cannot claim ticks absent from the critic chronology."""
    base = {
        "old_logprobs": torch.zeros(1),
        "is_padding": torch.zeros(1, dtype=torch.bool),
        "primitive_rewards": torch.zeros((1, 2)),
        "primitive_reward_mask": torch.tensor([[True, False]]),
        "duration_ticks": torch.tensor([1], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="dtype must be bool"):
        TrainingSignal(
            **base,
            actor_primitive_reward_mask=torch.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="must be a subset"):
        TrainingSignal(
            **base,
            actor_primitive_reward_mask=torch.tensor([[False, True]]),
        )


def test_shared_replay_schema_does_not_validate_family_payload_fields() -> None:
    """Family-required fields are checked by the family packer, not replay.py."""
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(-1.0),
        payload={"sampled_action": torch.zeros((2, 3))},
    ).to_dict()

    parsed = parse_policy_replay_data(replay)

    torch.testing.assert_close(
        parsed.payload["sampled_action"],
        torch.zeros((2, 3), dtype=torch.float32),
    )


def test_replay_envelope_accepts_arbitrary_model_family() -> None:
    """The replay envelope carries any policy's family string; bundles own family meaning."""
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="custom.trajectory.v1",
        payload_schema_version=1,
        model_family="some_future_policy",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(0.0),
        payload={},
    ).to_dict()

    parsed = parse_policy_replay_data(replay)

    assert parsed.model_family == "some_future_policy"


def test_replay_schema_rejects_unknown_versions() -> None:
    """Schema version mismatches fail at the replay envelope boundary."""
    replay = PolicyReplayData(
        replay_schema_version=2,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(0.0),
        payload={},
    )

    with pytest.raises(ValueError, match="Unsupported replay_schema_version"):
        parse_policy_replay_data(replay.to_dict())


def test_replay_schema_rejects_non_string_payload_schema() -> None:
    """Corrupt envelopes must not coerce missing schemas into string values."""
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(0.0),
        payload={},
    ).to_dict()
    replay["payload_schema"] = None

    with pytest.raises(ValueError, match="payload_schema must be a non-empty string"):
        parse_policy_replay_data(replay)


def test_replay_payload_round_trips_scalar_old_logprob() -> None:
    """Replay envelopes preserve scalar old-logprob values."""
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema="alpamayo_r1.trajectory.v1",
        payload_schema_version=1,
        model_family="alpamayo_r1",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(0.0),
        payload={},
    )

    parsed = parse_policy_replay_data(replay.to_dict())

    assert parsed.payload == {}
    assert torch.as_tensor(parsed.old_logprob).shape == ()


def test_trainer_replay_batch_stacks_optional_ppo_signals() -> None:
    """Optional PPO signals stay row-aligned through minibatch stacking."""
    samples = [
        TrainerReplayData(
            model_inputs={"x": torch.tensor(float(index))},
            training_signal=TrainingSignal(
                old_logprobs=torch.tensor([float(index)]),
                is_padding=torch.tensor([False]),
                advantages=torch.tensor([float(index + 1)]),
                returns=torch.tensor([float(index + 2)]),
                old_values=torch.tensor([float(index + 3)]),
                actor_valid=torch.tensor([index == 0]),
            ),
            rollout_id=f"rollout-{index}",
            weight_version=torch.tensor(index, dtype=torch.int64),
        )
        for index in range(2)
    ]

    batch = TrainerReplayDataBatch.stack(samples)

    torch.testing.assert_close(
        batch.training_signal.advantages, torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(batch.training_signal.returns, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(
        batch.training_signal.old_values, torch.tensor([3.0, 4.0])
    )
    torch.testing.assert_close(
        batch.training_signal.actor_valid,
        torch.tensor([True, False]),
    )


def test_trainer_replay_batch_rejects_mixed_optional_ppo_signals() -> None:
    """A minibatch cannot silently mix PPO and non-PPO replay rows."""
    samples = [
        TrainerReplayData(
            model_inputs={"x": torch.tensor(0.0)},
            training_signal=TrainingSignal(
                old_logprobs=torch.tensor([0.0]),
                is_padding=torch.tensor([False]),
                advantages=torch.tensor([1.0]),
            ),
            rollout_id="rollout-a",
            weight_version=torch.tensor(0, dtype=torch.int64),
        ),
        TrainerReplayData(
            model_inputs={"x": torch.tensor(1.0)},
            training_signal=TrainingSignal(
                old_logprobs=torch.tensor([0.0]),
                is_padding=torch.tensor([False]),
            ),
            rollout_id="rollout-b",
            weight_version=torch.tensor(0, dtype=torch.int64),
        ),
    ]

    with pytest.raises(ValueError, match="advantages mixes present and missing"):
        TrainerReplayDataBatch.stack(samples)
