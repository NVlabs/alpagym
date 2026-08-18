# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-side replay signal tests."""

import importlib
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
from alpagym_runtime.cosmos.replay_objective import (
    assert_replay_shapes,
    compute_kl_penalty,
    compute_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.cosmos.rollout_filter import filter_trainable_rollouts
from alpagym_runtime.replay import TrainerReplayData, TrainerReplayDataBatch, TrainingSignal


def test_trainer_applies_supplied_per_step_advantages(cosmos_stubs: None) -> None:
    """All-valid minibatch: the trainer applies the per-step advantages it is handed.

    The default packer marks every row valid, so this covers the no-padding path
    (including a zero-advantage valid row); padding masking is covered by
    ``test_trainer_minibatch_matches_clipped_grpo_oracle``.
    """
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_ScalarLogProbModel())

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = trainer._train_minibatch(
        minibatch_samples=[object(), object(), object(), object()],
        minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
        inter_policy_nccl=object(),
    )
    expected_loss = _clipped_grpo_policy_loss(
        new_logprobs=torch.zeros(4, dtype=torch.float32),
        old_logprobs=torch.zeros(4, dtype=torch.float32),
        advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
        grpo_ratio_clip_low=0.2,
        grpo_ratio_clip_high=0.2,
    )

    assert loss == pytest.approx(float(expected_loss.item()))
    assert kl == 0.0
    assert ratio_max == 1.0
    assert ratio_min == 1.0
    assert clip_fraction == 0.0
    assert trainer.model.rows_seen == 4
    assert torch.equal(
        trainer.model.ego_history_seen,
        torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
    )
    assert torch.isfinite(trainer.model.bias)
    assert trainer.model.bias.item() < 0.0


def test_trainer_minibatch_matches_clipped_grpo_oracle(cosmos_stubs: None) -> None:
    """Ratio clipping follows the configured PPO/GRPO objective."""
    del cosmos_stubs
    new_logprobs = torch.log(torch.tensor([1.3, 0.8, 1.1, 0.8], dtype=torch.float32))
    trainer = _trainer_for_replay_test(_VectorLogProbModel(new_logprobs))
    trainer.data_packer = _PaddingCapturingPacker()  # row 1 is padding
    trainer._grpo_ratio_clip_low = 0.05
    trainer._grpo_ratio_clip_high = 0.28
    before = trainer.model.log_probs.detach().clone()

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = trainer._train_minibatch(
        minibatch_samples=[object(), object(), object(), object()],
        minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
        inter_policy_nccl=object(),
    )
    # All 4 rows are forwarded, but the loss and the diagnostics (ratio min/max,
    # clip fraction) normalize over the 3 valid rows (0, 2, 3); the padded row 1
    # is excluded from both numerator and denominator, so the per-sample gradient
    # scale is independent of the padding count.
    expected_loss = _clipped_grpo_policy_loss(
        new_logprobs=before[[0, 2, 3]],
        old_logprobs=torch.zeros(3, dtype=torch.float32),
        advantages=torch.tensor([1.0, -2.0, -2.0], dtype=torch.float32),
        grpo_ratio_clip_low=0.05,
        grpo_ratio_clip_high=0.28,
    )

    assert loss == pytest.approx(float(expected_loss.item()))
    assert kl == 0.0
    assert ratio_max == pytest.approx(1.3)
    assert ratio_min == pytest.approx(0.8)
    assert clip_fraction == pytest.approx(2.0 / 3.0)
    assert torch.isfinite(trainer.model.log_probs).all()
    assert trainer.model.log_probs.detach()[2] < before[2]
    # Padding row 1 carries advantage 0, so it is forwarded but receives no
    # gradient and is left untouched by the update.
    torch.testing.assert_close(trainer.model.log_probs.detach()[1], before[1])


def test_trainer_requires_log_probs_output(cosmos_stubs: None) -> None:
    """Model wrappers must expose trajectory-level ``log_probs`` for GRPO."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_MissingLogProbModel())

    with pytest.raises(KeyError, match="log_probs"):
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
            inter_policy_nccl=object(),
        )


def test_prepare_training_data_flattens_steps_and_zeros_padding_advantage(
    cosmos_stubs: None,
) -> None:
    """Rollouts flatten into a per-step pool; padding steps get zero advantage."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.data_packer = _PerStepPacker({"a": [False, False], "b": [False, True]})
    rollouts = [
        SimpleNamespace(prompt="a", completion="a", n_ignore_prefix_tokens=0, advantage=1.5),
        SimpleNamespace(prompt="b", completion="b", n_ignore_prefix_tokens=0, advantage=-3.0),
    ]

    samples, advantages = trainer._prepare_training_data(rollouts)

    assert len(samples) == 4
    torch.testing.assert_close(
        advantages,
        torch.tensor([1.5, 1.5, -3.0, 0.0], dtype=torch.float32),
    )


def test_train_minibatch_forwards_all_rows_including_padding(cosmos_stubs: None) -> None:
    """Padding rows are forwarded, not dropped, so every DP worker runs the
    identical model forward in lockstep.

    The fake packer marks row 1 as padding, but the model still sees all 4 rows;
    the padded row is neutralized by its zero advantage, not by exclusion from
    the batch.
    """
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_ScalarLogProbModel())
    trainer.data_packer = _PaddingCapturingPacker()

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = trainer._train_minibatch(
        minibatch_samples=[object(), object(), object(), object()],
        minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
        inter_policy_nccl=object(),
    )

    assert trainer.model.rows_seen == 4
    assert torch.equal(
        trainer.model.ego_history_seen,
        torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
    )
    assert torch.isfinite(torch.tensor(loss))


def test_ppo_prepare_training_data_computes_gae_inside_trainer(
    cosmos_stubs: None,
) -> None:
    """PPO ignores Cosmos rollout advantages and derives GAE from transition signals."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoPerStepPacker({"a": [(1.0, 0.0, False, False), (1.0, 0.0, True, False)]})
    trainer._normalize_advantages = False
    trainer._gamma = 1.0
    trainer._gae_lambda = 1.0
    rollouts = [
        SimpleNamespace(
            prompt="a",
            completion="a",
            n_ignore_prefix_tokens=0,
            advantage=99.0,
            weight_version=0,
        )
    ]

    samples, advantages = trainer._prepare_training_data(rollouts)

    assert len(samples) == 2
    torch.testing.assert_close(advantages, torch.tensor([2.0, 1.0], dtype=torch.float32))
    torch.testing.assert_close(
        samples[0].training_signal.returns,
        torch.tensor([2.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        samples[1].training_signal.returns,
        torch.tensor([1.0], dtype=torch.float32),
    )


def test_ppo_rejects_behavior_version_mismatch(cosmos_stubs: None) -> None:
    """The Cosmos envelope and every real/padded replay row must agree."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoPerStepPacker(
        {"a": [(1.0, 0.0, True, False)]}
    )
    trainer._normalize_advantages = False
    trainer._gamma = 1.0
    trainer._gae_lambda = 1.0
    with pytest.raises(ValueError, match="behavior version"):
        trainer._prepare_training_data(
            [
                SimpleNamespace(
                    prompt="a",
                    completion="a",
                    n_ignore_prefix_tokens=0,
                    weight_version=1,
                )
            ]
        )
@pytest.mark.parametrize(
    ("configured", "expected"),
    ((None, 3), (64, 64)),
)
def test_ppo_transition_minibatch_is_independent_from_cosmos_rollout_batch(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    configured: int | None,
    expected: int,
) -> None:
    """A 750-row replay can use bounded updates without changing shard geometry."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    def initialize_base(trainer: Any, **kwargs: Any) -> None:
        del kwargs
        trainer._mini_batch = 3

    monkeypatch.setattr(trainer_module.AlpagymGRPOTrainer, "__init__", initialize_base)
    ppo: dict[str, object] = {}
    if configured is not None:
        ppo["step_mini_batch"] = configured
    trainer = trainer_module.AlpagymPPOTrainer(
        config=SimpleNamespace(custom={"ppo": ppo}),
        parallel_dims=object(),
    )
    assert trainer._mini_batch == expected


@pytest.mark.parametrize("configured", (0, -1, True, 1.5, "64"))
def test_ppo_transition_minibatch_must_be_a_positive_integer(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    configured: object,
) -> None:
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    def initialize_base(trainer: Any, **kwargs: Any) -> None:
        del kwargs
        trainer._mini_batch = 1

    monkeypatch.setattr(trainer_module.AlpagymGRPOTrainer, "__init__", initialize_base)
    with pytest.raises(ValueError, match="step_mini_batch must be a positive integer"):
        trainer_module.AlpagymPPOTrainer(
            config=SimpleNamespace(custom={"ppo": {"step_mini_batch": configured}}),
            parallel_dims=object(),
        )


def test_ppo_smoke_rl_training_step_with_mlp_actor_and_value_network(
    cosmos_stubs: None,
) -> None:
    """Smoke one trainer-owned PPO update through MLP actor and value networks."""
    del cosmos_stubs
    torch.manual_seed(7)
    model = _GaussianMlpActorCritic(obs_dim=3, action_dim=2, hidden_dim=16)
    trainer = _trainer_for_ppo_replay_test(model)
    trainer.data_packer = _PpoSmokePacker()
    trainer._gamma = 1.0
    trainer._gae_lambda = 1.0
    trainer._normalize_advantages = False
    trainer._value_loss_coef = 0.5
    trainer.optimizers = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
    rollouts = [
        SimpleNamespace(
            prompt="smoke",
            completion="smoke",
            n_ignore_prefix_tokens=0,
            advantage=0.0,
            weight_version=0,
        )
    ]

    samples, advantages = trainer._prepare_training_data(rollouts)
    actor_before = _clone_parameters(model.actor)
    value_before = _clone_parameters(model.value_net)
    log_std_before = model.log_std.detach().clone()

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = trainer._train_minibatch(
        minibatch_samples=samples,
        minibatch_advantages=advantages,
        inter_policy_nccl=object(),
    )

    assert len(samples) == 3
    torch.testing.assert_close(advantages, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32))
    torch.testing.assert_close(
        torch.cat([sample.training_signal.returns for sample in samples]),
        torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert kl == 0.0
    assert ratio_max > 0.0
    assert ratio_min > 0.0
    assert clip_fraction >= 0.0
    assert _parameters_changed(model.actor, actor_before) or not torch.equal(
        model.log_std.detach(),
        log_std_before,
    )
    assert _parameters_changed(model.value_net, value_before)


def test_ppo_minibatch_trains_value_head(cosmos_stubs: None) -> None:
    """The PPO trainer backprops value loss through model forward key ``values``."""
    del cosmos_stubs
    trainer = _trainer_for_ppo_replay_test(_ActorCriticValueModel())

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = trainer._train_minibatch(
        minibatch_samples=[object(), object(), object(), object()],
        minibatch_advantages=torch.zeros(4, dtype=torch.float32),
        inter_policy_nccl=object(),
    )

    assert loss == pytest.approx(2.0)
    assert kl == 0.0
    assert ratio_max == 1.0
    assert ratio_min == 1.0
    assert clip_fraction == 0.0
    assert trainer.model.value_bias.item() > 0.0
    assert trainer.model.logprob_bias.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Direct math-method tests: no fake model, no fake packer.
# ---------------------------------------------------------------------------


def test_compute_ppo_surrogate_matches_oracle(cosmos_stubs: None) -> None:
    """PPO surrogate matches the clipped objective for asymmetric bounds."""
    del cosmos_stubs
    new_logprobs = torch.log(torch.tensor([1.3, 0.8, 1.1, 0.8], dtype=torch.float32))
    old_logprobs = torch.zeros(4, dtype=torch.float32)
    advantages = torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32)

    loss, ratio = compute_ppo_surrogate(
        new_logprobs,
        old_logprobs,
        advantages,
        ratio_clip_low=0.05,
        ratio_clip_high=0.28,
        is_padding=torch.zeros(4, dtype=torch.bool),
    )

    expected = _clipped_grpo_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        grpo_ratio_clip_low=0.05,
        grpo_ratio_clip_high=0.28,
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(ratio, torch.tensor([1.3, 0.8, 1.1, 0.8], dtype=torch.float32))


def test_compute_ppo_surrogate_normalizes_over_valid_rows(cosmos_stubs: None) -> None:
    """Padding rows must not dilute the policy loss.

    The loss normalizes over valid rows (like the KL penalty), so adding padding
    rows to a minibatch with identical valid rows leaves the loss unchanged — the
    per-sample gradient scale is independent of the padding count.
    """
    del cosmos_stubs
    new = torch.log(torch.tensor([1.3, 1.1], dtype=torch.float32))
    old = torch.zeros(2, dtype=torch.float32)
    adv = torch.tensor([1.0, -2.0], dtype=torch.float32)
    loss_no_pad, _ = compute_ppo_surrogate(
        new,
        old,
        adv,
        ratio_clip_low=0.05,
        ratio_clip_high=0.28,
        is_padding=torch.zeros(2, dtype=torch.bool),
    )
    loss_padded, _ = compute_ppo_surrogate(
        torch.cat([new, torch.zeros(2)]),
        torch.zeros(4, dtype=torch.float32),
        torch.tensor([1.0, -2.0, 0.0, 0.0], dtype=torch.float32),
        ratio_clip_low=0.05,
        ratio_clip_high=0.28,
        is_padding=torch.tensor([False, False, True, True]),
    )
    torch.testing.assert_close(loss_no_pad, loss_padded)


def test_compute_value_loss_masks_padding(cosmos_stubs: None) -> None:
    """Value loss normalizes over valid rows only."""
    del cosmos_stubs
    loss = compute_value_loss(
        values=torch.tensor([1.0, 10.0, 3.0], dtype=torch.float32),
        returns=torch.tensor([2.0, 0.0, 1.0], dtype=torch.float32),
        is_padding=torch.tensor([False, True, False]),
    )

    assert float(loss.item()) == pytest.approx(1.25)


def test_compute_value_loss_supports_ppo_clipping(cosmos_stubs: None) -> None:
    """Clipped value loss uses the larger clipped/unclipped squared error."""
    del cosmos_stubs
    loss = compute_value_loss(
        values=torch.tensor([2.0], dtype=torch.float32),
        returns=torch.tensor([0.0], dtype=torch.float32),
        is_padding=torch.tensor([False]),
        old_values=torch.tensor([0.0], dtype=torch.float32),
        value_clip_range=0.2,
    )

    assert float(loss.item()) == pytest.approx(2.0)


def test_compute_kl_penalty_zero_when_disabled(cosmos_stubs: None) -> None:
    """kl_beta=0 short-circuits regardless of kl_div content."""
    del cosmos_stubs
    kl_div = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    is_padding = torch.tensor([False, False, False, False])

    kl_loss = compute_kl_penalty(kl_div, is_padding, kl_beta=0.0, device=torch.device("cpu"))

    assert float(kl_loss.item()) == 0.0


def test_compute_kl_penalty_zero_when_kl_div_missing(cosmos_stubs: None) -> None:
    """None kl_div (model didn't return one) is a no-op."""
    del cosmos_stubs
    is_padding = torch.tensor([False, False])

    kl_loss = compute_kl_penalty(None, is_padding, kl_beta=0.5, device=torch.device("cpu"))

    assert float(kl_loss.item()) == 0.0


def test_compute_kl_penalty_masks_padding(cosmos_stubs: None) -> None:
    """Padding rows are excluded from the KL mean."""
    del cosmos_stubs
    kl_div = torch.tensor([1.0, 100.0, 3.0, 100.0], dtype=torch.float32)
    is_padding = torch.tensor([False, True, False, True])

    kl_loss = compute_kl_penalty(kl_div, is_padding, kl_beta=2.0, device=torch.device("cpu"))

    # mean(1.0, 3.0) * kl_beta = 2.0 * 2.0 = 4.0
    assert float(kl_loss.item()) == pytest.approx(4.0)


def test_compute_kl_penalty_zero_when_all_padding(cosmos_stubs: None) -> None:
    """All-padding minibatch can't contribute KL — return zero."""
    del cosmos_stubs
    kl_div = torch.tensor([1.0, 2.0], dtype=torch.float32)
    is_padding = torch.tensor([True, True])

    kl_loss = compute_kl_penalty(kl_div, is_padding, kl_beta=1.0, device=torch.device("cpu"))

    assert float(kl_loss.item()) == 0.0


def test_assert_shape_contract_passes_on_matching_shapes(cosmos_stubs: None) -> None:
    """Conformant tensors do not raise."""
    del cosmos_stubs
    t = torch.zeros(4, dtype=torch.float32)

    assert_replay_shapes(t, t, t, t)
    assert_replay_shapes(t, t, t, None)


def test_assert_shape_contract_rejects_non_finite(cosmos_stubs: None) -> None:
    """Non-finite logprobs are a real bug: every forwarded row, padding included, must be finite."""
    del cosmos_stubs
    finite = torch.zeros(4, dtype=torch.float32)
    nan_logprobs = torch.tensor([0.0, float("nan"), 0.0, 0.0], dtype=torch.float32)

    with pytest.raises(FloatingPointError, match="non-finite log_probs"):
        assert_replay_shapes(nan_logprobs, finite, finite, None)

    nan_kl = torch.tensor([0.0, 0.0, float("inf"), 0.0], dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="non-finite kl_div"):
        assert_replay_shapes(finite, finite, finite, nan_kl)


def test_step_training_no_samples_fails_before_scheduler_step(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty trainer steps fail fast instead of reporting fake zero metrics."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    scheduler = _ListLRScheduler(0.125)
    trainer.lr_schedulers = scheduler
    trainer._group_size = 1
    trainer._mini_batch = 1
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 100
    trainer.config = SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=1))
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._prepare_training_data = lambda rollouts: (
        [],
        torch.empty(0, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="no trainable samples"):
        trainer.step_training(
            rollouts=[],
            current_step=7,
            total_steps=10,
            remain_samples_num=0,
            inter_policy_nccl=object(),
            is_master_replica=True,
        )

    assert scheduler.steps == 0


def test_step_training_success_reports_scalar_scheduler_lr(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training step metrics path also unwraps scheduler LR lists."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    scheduler = _ListLRScheduler(0.05)
    trainer.lr_schedulers = scheduler
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False,
        dp_shard_enabled=False,
        cp_enabled=False,
    )
    trainer._group_size = 1
    trainer._mini_batch = 1
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 100
    trainer.config = SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=1))
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._prepare_training_data = lambda rollouts: (
        [object()],
        torch.tensor([0.5], dtype=torch.float32),
    )
    trainer._run_training_loop = lambda samples, advantages, nccl: (
        2.0,
        0.25,
        2,
        1.1,
        0.9,
        0.5,
        0.0,
    )
    trainer._reference_reset = lambda current_step: None

    metrics = trainer.step_training(
        rollouts=[object()],
        current_step=8,
        total_steps=10,
        remain_samples_num=0,
        inter_policy_nccl=object(),
        is_master_replica=True,
    )

    assert metrics["train/learning_rate"] == 0.05
    assert metrics["train/loss_avg"] == 1.0
    assert metrics["train/kl_avg"] == 0.125
    assert metrics["train/clip_fraction"] == 0.25
    assert scheduler.steps == 1


def test_filter_rollouts_allows_empty_noop(cosmos_stubs: None) -> None:
    """No rollout completions can flow through the no-trainable-samples path."""
    del cosmos_stubs
    assert (
        filter_trainable_rollouts(
            [],
            current_step=11,
            train_batch_per_replica=2,
            allowed_outdated_steps=100,
        )
        == []
    )


def test_filter_rollouts_accepts_round_robin_split_groups(cosmos_stubs: None) -> None:
    """DP-rank slices may contain one rollout per prompt after Cosmos round-robin."""
    del cosmos_stubs
    kept = filter_trainable_rollouts(
        [
            _rollout(prompt="scene-a", completion="a-0", weight_version=10),
            _rollout(prompt="scene-b", completion="b-0", weight_version=10),
            _rollout(prompt="scene-c", completion="c-0", weight_version=10),
            _rollout(prompt="scene-d", completion="d-0", weight_version=10),
        ],
        current_step=11,
        train_batch_per_replica=4,
        allowed_outdated_steps=100,
    )

    expected_prompts = ["scene-a", "scene-b", "scene-c", "scene-d"]
    assert [rollout.prompt for rollout in kept] == expected_prompts


def test_filter_rollouts_rejects_duplicate_completion_paths(cosmos_stubs: None) -> None:
    """Rollout artifacts cannot share a path because filtering unlinks dropped files."""
    del cosmos_stubs
    with pytest.raises(ValueError, match="duplicate completion paths"):
        filter_trainable_rollouts(
            [
                _rollout(prompt="scene-a", completion="same-path", weight_version=10),
                _rollout(prompt="scene-b", completion="same-path", weight_version=10),
            ],
            current_step=11,
            train_batch_per_replica=2,
            allowed_outdated_steps=100,
        )


def test_filter_rollouts_keeps_fresh_rollouts_and_unlinks_dropped_artifacts(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """Freshness keeps the highest-version rollouts regardless of arrival order."""
    del cosmos_stubs
    newer_paths = [tmp_path / "newer-0.pt", tmp_path / "newer-1.pt"]
    older_paths = [tmp_path / "older-0.pt", tmp_path / "older-1.pt"]
    for path in newer_paths + older_paths:
        path.write_text("{}", encoding="utf-8")

    # Interleave high/low weight_version across arrival order: a positional
    # whole-chunk drop (keep first-N or last-N) would keep the wrong rollouts,
    # so passing this proves per-rollout version selection, not chunk slicing.
    kept = filter_trainable_rollouts(
        [
            _rollout(prompt="scene-newer", completion=newer_paths[0], weight_version=10),
            _rollout(prompt="scene-older", completion=older_paths[0], weight_version=1),
            _rollout(prompt="scene-other-newer", completion=newer_paths[1], weight_version=9),
            _rollout(prompt="scene-other-older", completion=older_paths[1], weight_version=2),
        ],
        current_step=11,
        train_batch_per_replica=2,
        allowed_outdated_steps=100,
    )

    assert [rollout.prompt for rollout in kept] == ["scene-newer", "scene-other-newer"]
    assert all(path.exists() for path in newer_paths)
    assert not any(path.exists() for path in older_paths)


def test_filter_rollouts_keeps_stale_rollouts_with_warning(
    cosmos_stubs: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Kept rollouts older than the staleness window are warned about, not dropped.

    Cosmos computes GRPO advantages before the trainer sees the data, so dropping
    a stale-but-kept rollout would discard valid training terms; the filter warns
    and trains on them instead.
    """
    del cosmos_stubs
    with caplog.at_level(logging.WARNING):
        kept = filter_trainable_rollouts(
            [
                _rollout(prompt="scene-old", completion="stale-0", weight_version=4),
                _rollout(prompt="scene-older", completion="stale-1", weight_version=4),
            ],
            current_step=8,
            train_batch_per_replica=2,
            allowed_outdated_steps=3,
        )

    assert len(kept) == 2
    assert "stale rollouts" in caplog.text


def test_trainer_rejects_logprob_shape_mismatch(cosmos_stubs: None) -> None:
    """Current-policy logprobs must align with transported rollout logprobs."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_BadShapeLogProbModel())

    with pytest.raises(ValueError, match="new log_probs shape"):
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
            inter_policy_nccl=object(),
        )


def test_trainer_rejects_kl_shape_mismatch(cosmos_stubs: None) -> None:
    """KL diagnostics must align with the same valid replay rows as logprobs."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_BadShapeKLModel())
    trainer._kl_beta = 0.1

    with pytest.raises(ValueError, match="kl_div shape"):
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor([1.0, 0.0, -2.0, -2.0], dtype=torch.float32),
            inter_policy_nccl=object(),
        )


def test_reference_model_copy_is_frozen_and_reset(cosmos_stubs: None) -> None:
    """KL reference models are frozen deep copies and reset from live weights."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_WrapperShapedModel())
    trainer._kl_beta = 0.1
    trainer._reference_reset_interval = 1

    trainer._ensure_reference_model()

    first_reference = trainer._reference_model
    assert first_reference is not trainer.model
    assert first_reference.policy is not trainer.model.policy
    assert all(not param.requires_grad for param in first_reference.parameters())

    with torch.no_grad():
        trainer.model.policy.weight.fill_(3.0)
    trainer._reference_reset(current_step=1)

    second_reference = trainer._reference_model
    assert second_reference is not first_reference
    assert second_reference.policy is not trainer.model.policy
    assert all(not param.requires_grad for param in second_reference.parameters())
    torch.testing.assert_close(
        second_reference.policy.weight,
        torch.full_like(second_reference.policy.weight, 3.0),
    )


def _rollout(
    prompt: str,
    completion: object,
    weight_version: int,
) -> Any:
    """Build a tiny Cosmos rollout-shaped object."""
    return SimpleNamespace(
        prompt=prompt,
        completion=completion,
        weight_version=weight_version,
        advantage=0.0,
        n_ignore_prefix_tokens=0,
    )


def _trainer_for_replay_test(model: torch.nn.Module) -> Any:
    """Build a minimal trainer instance around a fake model and packer."""
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.device = torch.device("cpu")
    trainer._reference_model = None
    trainer._grpo_ratio_clip_low = 0.2
    trainer._grpo_ratio_clip_high = 0.2
    trainer._kl_beta = 0.0
    trainer.model = model
    trainer.optimizers = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.data_packer = _SignalCapturingPacker()
    trainer.all_reduce_states = MethodType(_step_without_distributed, trainer)
    return trainer


def _step_without_distributed(self: Any, inter_policy_nccl: Any) -> float:
    """Apply optimizer updates without distributed collectives."""
    del inter_policy_nccl
    self.optimizers.step()
    self.optimizers.zero_grad()
    return 0.0


def _clipped_grpo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    grpo_ratio_clip_low: float,
    grpo_ratio_clip_high: float,
) -> torch.Tensor:
    """Compute the configured clipped PPO/GRPO objective."""
    ratio = torch.exp((new_logprobs - old_logprobs).clamp(min=-5.0, max=5.0))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - grpo_ratio_clip_low, 1.0 + grpo_ratio_clip_high) * advantages
    return -torch.min(surr1, surr2).mean()


class _ScalarLogProbModel(torch.nn.Module):
    """Model returning scalar trajectory-level logprobs."""

    def __init__(self) -> None:
        """Create one trainable scalar used as every row's logprob."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.rows_seen = 0
        self.ego_history_seen: torch.Tensor | None = None

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return ``[BT]`` logprobs for the generic trainer contract."""
        del teacher_model
        self.rows_seen = int(ego_history_xyz.shape[0])
        self.ego_history_seen = ego_history_xyz.detach().clone()
        return {
            "log_probs": self.bias.expand(ego_history_xyz.shape[0]),
            "kl_div": None,
        }


class _VectorLogProbModel(torch.nn.Module):
    """Model returning one controlled logprob per valid replay row."""

    def __init__(self, log_probs: torch.Tensor) -> None:
        """Store trainable row logprobs."""
        super().__init__()
        self.log_probs = torch.nn.Parameter(log_probs.clone())

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return controlled ``[BT]`` logprobs."""
        del teacher_model
        return {
            "log_probs": self.log_probs[: ego_history_xyz.shape[0]],
            "kl_div": None,
        }


class _MissingLogProbModel(torch.nn.Module):
    """Model violating the trainer-facing replay scoring contract."""

    def __init__(self) -> None:
        """Create a dummy parameter so optimizers can be constructed."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return no ``log_probs`` key."""
        del ego_history_xyz, teacher_model
        return {"kl_div": None}


class _BadShapeLogProbModel(torch.nn.Module):
    """Model returning the wrong number of trainer rows."""

    def __init__(self) -> None:
        """Create one trainable scalar used in a bad-shaped output."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return ``[BT - 1]`` logprobs."""
        del teacher_model
        return {
            "log_probs": self.bias.expand(ego_history_xyz.shape[0] - 1),
            "kl_div": None,
        }


class _BadShapeKLModel(torch.nn.Module):
    """Model returning KL for the wrong number of trainer rows."""

    def __init__(self) -> None:
        """Create one trainable scalar used as every row's logprob."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return valid logprobs and bad-shaped KL."""
        del teacher_model
        return {
            "log_probs": self.bias.expand(ego_history_xyz.shape[0]),
            "kl_div": self.bias.expand(ego_history_xyz.shape[0] + 1),
        }


class _WrapperShapedModel(torch.nn.Module):
    """Small model with a ``policy`` child like a Cosmos model wrapper."""

    def __init__(self) -> None:
        """Create one nested trainable parameter."""
        super().__init__()
        self.policy = torch.nn.Linear(1, 1, bias=False)

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return a scalar logprob per row."""
        del teacher_model
        return {
            "log_probs": self.policy(ego_history_xyz).reshape(-1),
            "kl_div": None,
        }


class _ListLRScheduler:
    """Scheduler stub matching PyTorch's ``get_last_lr`` shape."""

    def __init__(self, lr: float) -> None:
        """Store the LR and step count."""
        self.lr = lr
        self.steps = 0

    def get_last_lr(self) -> list[float]:
        """Return the usual PyTorch scheduler list form."""
        return [self.lr]

    def step(self) -> None:
        """Record that the scheduler advanced."""
        self.steps += 1


class _PerStepPacker:
    """Packer returning a per-rollout list of single-step samples for flattening."""

    def __init__(self, padding_by_prompt: dict[str, list[bool]]) -> None:
        """Map each prompt to the is_padding flag of each of its steps."""
        self._padding_by_prompt = padding_by_prompt

    def get_policy_input(
        self,
        prompt: str,
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> list[TrainerReplayData]:
        """Return one single-step sample per configured step for ``prompt``."""
        del completion, n_ignore_prefix_tokens
        return [
            TrainerReplayData(
                model_inputs={"x": torch.zeros(1, dtype=torch.float32)},
                training_signal=TrainingSignal(
                    old_logprobs=torch.zeros(1, dtype=torch.float32),
                    is_padding=torch.tensor([is_padding], dtype=torch.bool),
                ),
                rollout_id=prompt,
                weight_version=torch.zeros((), dtype=torch.int64),
            )
            for is_padding in self._padding_by_prompt[prompt]
        ]


class _SignalCapturingPacker:
    """Tiny packer exposing the trainer's expected methods (4 all-valid rows)."""

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Return a fixed 4-row, no-padding replay batch."""
        del samples
        return TrainerReplayDataBatch(
            model_inputs={"ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)},
            training_signal=TrainingSignal(
                old_logprobs=torch.zeros(4, dtype=torch.float32),
                is_padding=torch.zeros(4, dtype=torch.bool),
            ),
            rollout_ids=("rollout-a", "rollout-b", "rollout-c", "rollout-d"),
            weight_versions=torch.zeros(4, dtype=torch.int64),
        )


class _PaddingCapturingPacker:
    """Packer marking row 1 as padding: all rows forward, but loss/KL/metrics mask it out."""

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Return a 4-row batch where row 1 is padding."""
        del samples
        return TrainerReplayDataBatch(
            model_inputs={"ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)},
            training_signal=TrainingSignal(
                old_logprobs=torch.zeros(4, dtype=torch.float32),
                is_padding=torch.tensor([False, True, False, False]),
            ),
            rollout_ids=("rollout-a", "rollout-b", "rollout-c", "rollout-d"),
            weight_versions=torch.zeros(4, dtype=torch.int64),
        )


def _trainer_for_ppo_replay_test(model: torch.nn.Module) -> Any:
    """Build a minimal PPO trainer instance around a fake actor-critic model."""
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.device = torch.device("cpu")
    trainer._reference_model = None
    trainer._grpo_ratio_clip_low = 0.2
    trainer._grpo_ratio_clip_high = 0.2
    trainer._kl_beta = 0.0
    trainer._value_loss_coef = 1.0
    trainer._value_clip_range = None
    trainer.model = model
    trainer.optimizers = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.data_packer = _PpoSignalCapturingPacker()
    trainer.all_reduce_states = MethodType(_step_without_distributed, trainer)
    return trainer


class _ActorCriticValueModel(torch.nn.Module):
    """Tiny actor-critic surface returning log probabilities and values."""

    def __init__(self) -> None:
        """Create independent actor and critic scalars."""
        super().__init__()
        self.logprob_bias = torch.nn.Parameter(torch.tensor(0.0))
        self.value_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return one actor logprob and one critic value per row."""
        del teacher_model
        rows = ego_history_xyz.shape[0]
        return {
            "log_probs": self.logprob_bias.expand(rows),
            "values": self.value_bias.expand(rows),
            "kl_div": None,
        }


class _PpoPerStepPacker:
    """Packer returning configured raw-transition samples for flattening tests."""

    def __init__(self, rows_by_prompt: dict[str, list[tuple[float, float, bool, bool]]]) -> None:
        """Map prompt to ``(reward, old_value, terminated, is_padding)`` rows."""
        self._rows_by_prompt = rows_by_prompt

    def get_policy_input(
        self,
        prompt: str,
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> list[TrainerReplayData]:
        """Return one raw-transition sample per configured step for ``prompt``."""
        del completion, n_ignore_prefix_tokens
        return [
            TrainerReplayData(
                model_inputs={"x": torch.zeros(1, dtype=torch.float32)},
                training_signal=TrainingSignal(
                    old_logprobs=torch.zeros(1, dtype=torch.float32),
                    is_padding=torch.tensor([is_padding], dtype=torch.bool),
                    rewards=torch.tensor([reward], dtype=torch.float32),
                    terminateds=torch.tensor([terminated], dtype=torch.bool),
                    old_values=torch.tensor([old_value], dtype=torch.float32),
                ),
                rollout_id=prompt,
                weight_version=torch.zeros((), dtype=torch.int64),
            )
            for reward, old_value, terminated, is_padding in self._rows_by_prompt[prompt]
        ]


class _PpoSignalCapturingPacker:
    """Tiny packer exposing the PPO trainer's expected collate method."""

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Return a 4-row PPO batch with one padding row."""
        del samples
        return TrainerReplayDataBatch(
            model_inputs={"ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)},
            training_signal=TrainingSignal(
                old_logprobs=torch.zeros(4, dtype=torch.float32),
                is_padding=torch.tensor([False, True, False, False]),
                advantages=torch.zeros(4, dtype=torch.float32),
                returns=torch.tensor([2.0, 0.0, 2.0, 2.0], dtype=torch.float32),
                old_values=torch.zeros(4, dtype=torch.float32),
            ),
            rollout_ids=("rollout-a", "rollout-b", "rollout-c", "rollout-d"),
            weight_versions=torch.zeros(4, dtype=torch.int64),
        )


class _GaussianMlpActorCritic(torch.nn.Module):
    """Small continuous-control actor plus a parallel MLP value network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        """Build actor and value MLPs with matching hidden width."""
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        return_log_prob: bool = True,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Score replay actions and predict values for PPO."""
        del return_log_prob, teacher_model
        mean = self.actor(features.float())
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions.float()).sum(dim=-1)
        values = self.value_net(features.float()).squeeze(-1)
        return {"log_probs": log_probs, "values": values, "kl_div": None}


class _PpoSmokePacker:
    """Packer with one tiny rollout of raw transition replay rows."""

    def get_policy_input(
        self,
        prompt: str,
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> list[TrainerReplayData]:
        """Return one three-step terminal rollout for PPO smoke training."""
        del prompt, completion, n_ignore_prefix_tokens
        features = torch.tensor(
            [[1.0, 0.0, 0.5], [0.5, 1.0, -0.25], [-0.5, 0.25, 1.0]],
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [[0.5, -0.25], [0.25, 0.75], [-0.75, 0.5]],
            dtype=torch.float32,
        )
        rewards = torch.ones(3, dtype=torch.float32)
        terminateds = torch.tensor([False, False, True], dtype=torch.bool)
        return [
            TrainerReplayData(
                model_inputs={"features": features[index], "actions": actions[index]},
                training_signal=TrainingSignal(
                    old_logprobs=torch.zeros(1, dtype=torch.float32),
                    is_padding=torch.zeros(1, dtype=torch.bool),
                    rewards=rewards[index].reshape(1),
                    terminateds=terminateds[index].reshape(1),
                    old_values=torch.zeros(1, dtype=torch.float32),
                ),
                rollout_id="smoke",
                weight_version=torch.zeros((), dtype=torch.int64),
            )
            for index in range(3)
        ]

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Use the production replay-batch stacker."""
        return TrainerReplayDataBatch.stack(samples)


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    """Clone detached parameters for change detection."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(module: torch.nn.Module, before: list[torch.Tensor]) -> bool:
    """Return whether any parameter differs from its saved value."""
    return any(
        not torch.equal(parameter.detach(), old)
        for parameter, old in zip(module.parameters(), before, strict=True)
    )
