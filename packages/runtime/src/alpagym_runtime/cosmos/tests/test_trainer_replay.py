# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-side replay signal tests."""

import importlib
import json
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
from alpagym_runtime.replay import (
    TrainerReplayData,
    TrainerReplayDataBatch,
    TrainingSignal,
)


def test_trainer_applies_supplied_per_step_advantages(cosmos_stubs: None) -> None:
    """All-valid minibatch: the trainer applies the per-step advantages it is handed.

    The default packer marks every row valid, so this covers the no-padding path
    (including a zero-advantage valid row); padding masking is covered by
    ``test_trainer_minibatch_matches_clipped_grpo_oracle``.
    """
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_ScalarLogProbModel())

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
            inter_policy_nccl=object(),
        )
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

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
            inter_policy_nccl=object(),
        )
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
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
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
        SimpleNamespace(
            prompt="a", completion="a", n_ignore_prefix_tokens=0, advantage=1.5
        ),
        SimpleNamespace(
            prompt="b", completion="b", n_ignore_prefix_tokens=0, advantage=-3.0
        ),
    ]

    samples, advantages = trainer._prepare_training_data(rollouts)

    assert len(samples) == 4
    torch.testing.assert_close(
        advantages,
        torch.tensor([1.5, 1.5, -3.0, 0.0], dtype=torch.float32),
    )


def test_train_minibatch_forwards_all_rows_including_padding(
    cosmos_stubs: None,
) -> None:
    """Padding rows are forwarded, not dropped, so every DP worker runs the
    identical model forward in lockstep.

    The fake packer marks row 1 as padding, but the model still sees all 4 rows;
    the padded row is neutralized by its zero advantage, not by exclusion from
    the batch.
    """
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_ScalarLogProbModel())
    trainer.data_packer = _PaddingCapturingPacker()

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
            inter_policy_nccl=object(),
        )
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
    trainer.data_packer = _PpoPerStepPacker(
        {"a": [(1.0, 0.0, False, False), (1.0, 0.0, True, False)]}
    )
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
    torch.testing.assert_close(
        advantages, torch.tensor([2.0, 1.0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        samples[0].training_signal.returns,
        torch.tensor([2.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        samples[1].training_signal.returns,
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert trainer._last_ppo_advantage_metrics == pytest.approx(
        {
            "train/ppo_advantage_raw_rows": 2,
            "train/ppo_advantage_raw_min": 1.0,
            "train/ppo_advantage_raw_mean": 1.5,
            "train/ppo_advantage_raw_max": 2.0,
            "train/ppo_advantage_raw_std": 0.5,
            "train/ppo_advantage_effective_rows": 2,
            "train/ppo_advantage_effective_min": 1.0,
            "train/ppo_advantage_effective_mean": 1.5,
            "train/ppo_advantage_effective_max": 2.0,
            "train/ppo_advantage_effective_std": 0.5,
        }
    )


def test_ppo_reports_normalized_effective_advantages_separately_from_raw_gae(
    cosmos_stubs: None,
) -> None:
    """Diagnostics expose the real PPO signal, not Cosmos's GRPO placeholder."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoPerStepPacker(
        {"a": [(1.0, 0.0, False, False), (1.0, 0.0, True, False)]}
    )
    trainer._normalize_advantages = True
    trainer._gamma = 1.0
    trainer._gae_lambda = 1.0

    _samples, advantages = trainer._prepare_training_data(
        [
            SimpleNamespace(
                prompt="a",
                completion="a",
                n_ignore_prefix_tokens=0,
                advantage=0.0,
                weight_version=0,
            )
        ]
    )

    torch.testing.assert_close(advantages, torch.tensor([1.0, -1.0]))
    metrics = trainer._last_ppo_advantage_metrics
    assert metrics["train/ppo_advantage_raw_mean"] == pytest.approx(1.5)
    assert metrics["train/ppo_advantage_raw_std"] == pytest.approx(0.5)
    assert metrics["train/ppo_advantage_effective_mean"] == pytest.approx(0.0)
    assert metrics["train/ppo_advantage_effective_std"] == pytest.approx(1.0)


def test_ppo_prepare_keeps_actor_invalid_transition_in_gae_and_value_targets(
    cosmos_stubs: None,
) -> None:
    """An unexecuted terminal action closes GAE but gets zero actor advantage."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoActorValidityPacker()
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

    torch.testing.assert_close(advantages, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(
        torch.cat([sample.training_signal.returns for sample in samples]),
        torch.tensor([2.0, 1.0]),
    )
    torch.testing.assert_close(
        samples[1].training_signal.advantages,
        torch.tensor([0.0]),
    )


def test_ppo_prepare_compacts_padding_after_gae_on_single_gpu(
    cosmos_stubs: None,
) -> None:
    """Single-GPU replay drops fake visual rows but keeps critic-only rows.

    The padding row is present while GAE is computed and only then removed.
    The terminal non-padding row has ``actor_valid=False``: it must remain in
    the replay pool with a return target even though its actor advantage is
    zero.
    """
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoActorValidityAndPaddingPacker()
    trainer.parallel_dims = SimpleNamespace(world_size=1)
    trainer._normalize_advantages = False
    trainer._gamma = 1.0
    trainer._gae_lambda = 1.0
    gae_padding_masks: list[list[bool]] = []
    compute_gae = trainer._compute_gae

    def _record_gae_input(step_samples: list[Any]) -> tuple[list[float], list[float]]:
        gae_padding_masks.append(
            [bool(step.training_signal.is_padding.item()) for step in step_samples]
        )
        return compute_gae(step_samples)

    trainer._compute_gae = _record_gae_input
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

    assert gae_padding_masks == [[False, False, True]]
    assert len(samples) == 2
    assert all(not bool(sample.training_signal.is_padding.item()) for sample in samples)
    assert [bool(sample.training_signal.actor_valid.item()) for sample in samples] == [
        True,
        False,
    ]
    torch.testing.assert_close(advantages, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(
        torch.cat([sample.training_signal.returns for sample in samples]),
        torch.tensor([2.0, 1.0]),
    )


def test_ppo_smdp_gae_discounts_variable_controller_tick_blocks(
    cosmos_stubs: None,
) -> None:
    """Macro transitions preserve primitive reward order and elapsed time."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._gamma = 0.9
    trainer._gae_lambda = 0.8
    samples = [
        _smdp_sample(
            rewards=(1.0, 2.0, 3.0),
            old_value=0.5,
            bootstrap_value=0.25,
            terminated=False,
        ),
        _smdp_sample(
            rewards=(4.0, 5.0),
            old_value=0.25,
            bootstrap_value=0.0,
            terminated=True,
        ),
    ]

    advantages, returns = trainer._compute_gae(samples)

    final_advantage = (4.0 + 0.9 * 5.0) - 0.25
    first_delta = (1.0 + 0.9 * 2.0 + 0.9**2 * 3.0) + 0.9**3 * 0.25 - 0.5
    first_advantage = first_delta + (0.9 * 0.8) ** 3 * final_advantage
    assert advantages == pytest.approx([first_advantage, final_advantage])
    assert returns == pytest.approx([first_advantage + 0.5, final_advantage + 0.25])


def test_ppo_smdp_k1_matches_direct_transition_gae(cosmos_stubs: None) -> None:
    """The semi-Markov representation is an exact K=1 direct-policy regression."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._gamma = 0.95
    trainer._gae_lambda = 0.8
    direct = [
        _direct_ppo_sample(1.0, old_value=0.3, terminated=False),
        _direct_ppo_sample(2.0, old_value=0.4, terminated=True),
    ]
    smdp = [
        _smdp_sample(
            rewards=(1.0,),
            old_value=0.3,
            bootstrap_value=0.4,
            terminated=False,
            width=1,
        ),
        _smdp_sample(
            rewards=(2.0,),
            old_value=0.4,
            bootstrap_value=0.0,
            terminated=True,
            width=1,
        ),
    ]

    direct_advantages, direct_returns = trainer._compute_gae(direct)
    smdp_advantages, smdp_returns = trainer._compute_gae(smdp)

    assert smdp_advantages == pytest.approx(direct_advantages)
    assert smdp_returns == pytest.approx(direct_returns)


def test_ppo_rejects_behavior_version_mismatch(cosmos_stubs: None) -> None:
    """The Cosmos envelope and every real/padded replay row must agree."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.data_packer = _PpoPerStepPacker({"a": [(1.0, 0.0, True, False)]})
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


def test_ppo_std_clamp_is_ordered_on_optimizer_cuda_stream(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-step clamp cannot race Adam and is visible to the next forward."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    events: list[object] = []
    train_stream = object()

    class _CallerStream:
        def wait_stream(self, stream: object) -> None:
            events.append(("caller_wait", stream))

    class _StreamContext:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> None:
            events.append(("enter", self._stream))

        def __exit__(self, *exc: object) -> None:
            del exc
            events.append(("exit", self._stream))

    def _base_step(self: object, nccl: object) -> float:
        del self, nccl
        events.append("optimizer")
        return 3.0

    def _clamp_std(*, min_std: float, max_std: float) -> None:
        events.append(("clamp", min_std, max_std))

    monkeypatch.setattr(
        trainer_module.AlpagymGRPOTrainer,
        "all_reduce_states",
        _base_step,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _CallerStream())
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.train_stream = train_stream
    trainer.model = SimpleNamespace(clamp_std_=_clamp_std)
    trainer._min_action_std = 0.02
    trainer._max_action_std = 2.0

    grad_norm = trainer.all_reduce_states(object())

    assert grad_norm == 3.0
    assert events == [
        "optimizer",
        ("enter", train_stream),
        ("clamp", 0.02, 2.0),
        ("exit", train_stream),
        ("caller_wait", train_stream),
    ]


def test_zero_reduced_gradient_skips_adam_state_and_weight_decay(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-signal batch cannot mutate parameters or initialize Adam moments."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    class _FakeStream:
        def wait_stream(self, other: object) -> None:
            del other

    class _StreamContext:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> object:
            return self._stream

        def __exit__(self, *exc: object) -> None:
            del exc

    monkeypatch.setattr(torch.cuda, "current_stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)
    monkeypatch.setattr(
        trainer_module.dist_util,
        "gradient_reduce_across_dp_replicas_",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        trainer_module.dist_util,
        "gradient_norm_clipping",
        lambda *args, **kwargs: torch.tensor(0.0),
        raising=False,
    )

    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.train_stream = _FakeStream()
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight.data.fill_(2.0)
    trainer.model.weight.grad = torch.zeros_like(trainer.model.weight)
    trainer.optimizers = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.config = SimpleNamespace(train=SimpleNamespace(optm_grad_norm_clip=1.0))
    trainer._optimizer_steps_applied_in_training_step = 0
    before = trainer.model.weight.detach().clone()

    grad_norm = trainer.all_reduce_states(object())

    assert grad_norm == 0.0
    torch.testing.assert_close(trainer.model.weight, before)
    assert trainer.optimizers.state == {}
    assert trainer._optimizer_steps_applied_in_training_step == 0


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

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=samples,
            minibatch_advantages=advantages,
            inter_policy_nccl=object(),
        )
    )

    assert len(samples) == 3
    torch.testing.assert_close(
        advantages, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    )
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


def test_ppo_step_microbatches_accumulate_one_exact_optimizer_update(
    cosmos_stubs: None,
) -> None:
    """GPU microbatches equal one full-batch PPO update with mixed masks.

    Actor and critic reductions have different denominators because two real
    rows are critic-only. Splitting the four rows into two forwards must scale
    those losses independently, accumulate gradients, and call ``step`` once.
    """
    del cosmos_stubs
    samples, advantages = _ppo_accumulation_samples()
    full_batch = _trainer_for_ppo_accumulation_test(step_mini_batch=4)
    micro_batch = _trainer_for_ppo_accumulation_test(step_mini_batch=2)

    torch.manual_seed(31)
    full_metrics = full_batch._run_training_loop(samples, advantages, object())
    torch.manual_seed(31)
    micro_metrics = micro_batch._run_training_loop(samples, advantages, object())

    assert full_batch.optimizers.step_calls == 1
    assert micro_batch.optimizers.step_calls == 1
    assert full_metrics[2] == 1
    assert micro_metrics[2] == 1
    assert full_batch._last_micro_batches == 1
    assert micro_batch._last_micro_batches == 2
    assert full_batch.model.actor_weight.item() != 0.0
    assert full_batch.model.value_weight.item() != 0.0
    torch.testing.assert_close(
        micro_batch.model.actor_weight,
        full_batch.model.actor_weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        micro_batch.model.value_weight,
        full_batch.model.value_weight,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert (
        micro_batch._last_optimizer_permutation_metrics
        == full_batch._last_optimizer_permutation_metrics
    )


def test_ppo_optimizer_permutation_is_private_restart_stable_and_auditable(
    cosmos_stubs: None,
) -> None:
    """PPO row order must not depend on unrelated global RNG consumption."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()

    first, first_record = trainer_module._deterministic_optimizer_permutation(
        num_steps=98,
        base_seed=20260822,
        current_step=1,
        optimization_iteration=0,
    )
    rng_after = torch.get_rng_state().clone()
    _unrelated_random_values = torch.rand(1000)
    second, second_record = trainer_module._deterministic_optimizer_permutation(
        num_steps=98,
        base_seed=20260822,
        current_step=1,
        optimization_iteration=0,
    )
    next_step, next_record = trainer_module._deterministic_optimizer_permutation(
        num_steps=98,
        base_seed=20260822,
        current_step=2,
        optimization_iteration=0,
    )

    assert torch.equal(rng_after, rng_before)
    assert torch.equal(first, second)
    assert first_record == second_record
    assert first_record["schema_id"] == "alpagym.optimizer_permutation.v1"
    assert first_record["num_steps"] == 98
    assert len(first_record["indices_sha256"]) == 64
    assert first_record["indices_head"] == [int(index) for index in first[:16]]
    assert first_record["indices_tail"] == [int(index) for index in first[-16:]]
    assert not torch.equal(first, next_step)
    assert first_record["indices_sha256"] != next_record["indices_sha256"]


def test_ppo_minibatch_trains_value_head(cosmos_stubs: None) -> None:
    """The PPO trainer backprops value loss through model forward key ``values``."""
    del cosmos_stubs
    trainer = _trainer_for_ppo_replay_test(_ActorCriticValueModel())

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=[object(), object(), object(), object()],
            minibatch_advantages=torch.zeros(4, dtype=torch.float32),
            inter_policy_nccl=object(),
        )
    )

    assert loss == pytest.approx(2.0)
    assert kl == 0.0
    assert ratio_max == 1.0
    assert ratio_min == 1.0
    assert clip_fraction == 0.0
    assert trainer.model.value_bias.item() > 0.0
    assert trainer.model.logprob_bias.item() == pytest.approx(0.0)


def test_ppo_actor_invalid_rows_train_value_but_not_actor_or_kl(
    cosmos_stubs: None,
) -> None:
    """A never-executed sampled action remains a critic target only."""
    del cosmos_stubs
    model = _ActorCriticKlValueModel()
    trainer = _trainer_for_ppo_replay_test(model)
    trainer.data_packer = _ActorInvalidPpoPacker()
    trainer._kl_beta = 1.0

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            minibatch_samples=[object(), object()],
            minibatch_advantages=torch.tensor([5.0, -3.0]),
            inter_policy_nccl=object(),
        )
    )

    assert loss == pytest.approx(2.0)
    assert kl == 0.0
    assert ratio_max == 1.0
    assert ratio_min == 1.0
    assert clip_fraction == 0.0
    assert model.logprob_bias.item() == pytest.approx(0.0)
    assert model.kl_bias.item() == pytest.approx(1.0)
    assert model.value_bias.item() > 0.0

    trainer._mini_batch = 2
    trainer._flow_chunk_density = False
    metrics = trainer._post_update_diagnostics([object(), object()])
    assert metrics["train/post_update_valid_rows"] == 0
    assert metrics["train/post_update_ratio_p50"] == 1.0
    assert metrics["train/post_update_value_valid_rows"] == 2
    assert metrics["train/post_update_value_max_abs_delta"] > 0.0


def test_ppo_critic_only_adamw_does_not_decay_or_advance_actor(
    cosmos_stubs: None,
) -> None:
    """Critic-only replay leaves actor parameters and AdamW state untouched."""
    del cosmos_stubs
    model = _ActorCriticValueModel()
    model.logprob_bias.data.fill_(2.0)
    trainer = _trainer_for_ppo_replay_test(model)
    trainer.data_packer = _ActorInvalidPpoPacker()
    trainer.optimizers = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    actor_before = model.logprob_bias.detach().clone()
    critic_before = model.value_bias.detach().clone()

    trainer._train_minibatch(
        minibatch_samples=[object(), object()],
        minibatch_advantages=torch.tensor([5.0, -3.0]),
        inter_policy_nccl=object(),
    )

    torch.testing.assert_close(model.logprob_bias, actor_before)
    assert not torch.equal(model.value_bias.detach(), critic_before)
    assert model.logprob_bias not in trainer.optimizers.state
    assert model.value_bias in trainer.optimizers.state


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
    torch.testing.assert_close(
        ratio, torch.tensor([1.3, 0.8, 1.1, 0.8], dtype=torch.float32)
    )


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

    kl_loss = compute_kl_penalty(
        kl_div, is_padding, kl_beta=0.0, device=torch.device("cpu")
    )

    assert float(kl_loss.item()) == 0.0


def test_compute_kl_penalty_zero_when_kl_div_missing(cosmos_stubs: None) -> None:
    """None kl_div (model didn't return one) is a no-op."""
    del cosmos_stubs
    is_padding = torch.tensor([False, False])

    kl_loss = compute_kl_penalty(
        None, is_padding, kl_beta=0.5, device=torch.device("cpu")
    )

    assert float(kl_loss.item()) == 0.0


def test_compute_kl_penalty_masks_padding(cosmos_stubs: None) -> None:
    """Padding rows are excluded from the KL mean."""
    del cosmos_stubs
    kl_div = torch.tensor([1.0, 100.0, 3.0, 100.0], dtype=torch.float32)
    is_padding = torch.tensor([False, True, False, True])

    kl_loss = compute_kl_penalty(
        kl_div, is_padding, kl_beta=2.0, device=torch.device("cpu")
    )

    # mean(1.0, 3.0) * kl_beta = 2.0 * 2.0 = 4.0
    assert float(kl_loss.item()) == pytest.approx(4.0)


def test_compute_kl_penalty_zero_when_all_padding(cosmos_stubs: None) -> None:
    """All-padding minibatch can't contribute KL — return zero."""
    del cosmos_stubs
    kl_div = torch.tensor([1.0, 2.0], dtype=torch.float32)
    is_padding = torch.tensor([True, True])

    kl_loss = compute_kl_penalty(
        kl_div, is_padding, kl_beta=1.0, device=torch.device("cpu")
    )

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
    trainer._on_policy = True
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
    trainer._on_policy = True
    trainer._last_ppo_advantage_metrics = {"train/ppo_advantage_effective_mean": 0.25}
    trainer.config = SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=1))
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._prepare_training_data = lambda rollouts: (
        [object()],
        torch.tensor([0.5], dtype=torch.float32),
    )

    def _successful_training_loop(
        samples: list[Any],
        advantages: torch.Tensor,
        nccl: object,
    ) -> tuple[float, float, int, float, float, float, float]:
        del samples, advantages, nccl
        trainer._optimizer_steps_applied_in_training_step = 2
        return (2.0, 0.25, 2, 1.1, 0.9, 0.5, 0.0)

    trainer._run_training_loop = _successful_training_loop
    trainer._reference_reset = lambda current_step: None

    metrics = trainer.step_training(
        rollouts=[SimpleNamespace(weight_version=7)],
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
    assert metrics["train/optimizer_steps_applied"] == 2
    assert metrics["train/ppo_advantage_effective_mean"] == 0.25
    assert scheduler.steps == 1


def test_step_training_rejects_uniformly_stale_on_policy_batch_before_unpack(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-stale batch cannot bypass the mixed-version replay guard."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer._group_size = 1
    trainer._mini_batch = 1
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 0
    trainer._on_policy = True
    trainer.config = SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=2))
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    unpacked = False

    def _unexpected_unpack(rollouts: list[Any]) -> Any:
        del rollouts
        nonlocal unpacked
        unpacked = True
        raise AssertionError("stale replay reached artifact unpack")

    trainer._prepare_training_data = _unexpected_unpack

    with pytest.raises(ValueError, match="exact behavior version 2"):
        trainer.step_training(
            rollouts=[
                SimpleNamespace(weight_version=1),
                SimpleNamespace(weight_version=1),
            ],
            current_step=3,
            total_steps=3,
            remain_samples_num=0,
            inter_policy_nccl=object(),
            is_master_replica=True,
        )

    assert not unpacked


def test_step_training_does_not_advance_scheduler_without_optimizer_step(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-gradient PPO pass leaves both Adam and LR clocks unchanged."""
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
    trainer._allowed_outdated_steps = 0
    trainer._on_policy = True
    trainer.config = SimpleNamespace(train=SimpleNamespace(train_batch_per_replica=1))
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._prepare_training_data = lambda rollouts: (
        [object()],
        torch.tensor([0.0], dtype=torch.float32),
    )

    def _zero_gradient_pass(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        trainer._optimizer_steps_applied_in_training_step = 0
        return (0.0, 0.0, 0, 1.0, 1.0, 0.0, 0.0)

    trainer._run_training_loop = _zero_gradient_pass

    metrics = trainer.step_training(
        rollouts=[SimpleNamespace(weight_version=0)],
        current_step=1,
        total_steps=2,
        remain_samples_num=0,
        inter_policy_nccl=object(),
        is_master_replica=True,
    )

    assert metrics["train/optimizer_steps_applied"] == 0
    assert scheduler.steps == 0


def test_ppo_pre_update_behavior_kl_guard_rejects_before_training(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-diverged replay batch must not apply another PPO update."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    events: list[str] = []
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_approx_kl": 0.25,
        "train/pre_update_valid_rows": len(samples),
    }
    trainer._post_update_diagnostics = lambda samples: {
        "train/post_update_approx_kl": 0.0,
        "train/post_update_valid_rows": len(samples),
    }

    def _unexpected_training(*args: Any) -> Any:
        del args
        events.append("train")
        raise AssertionError("training ran after the pre-update KL guard failed")

    trainer._run_training_loop = _unexpected_training

    with pytest.raises(FloatingPointError, match="pre-update.*behavior KL"):
        trainer.step_training(
            rollouts=[object()],
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
            inter_policy_nccl=object(),
            is_master_replica=True,
        )

    assert events == []
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []


def test_on_policy_ppo_rejects_same_version_with_wrong_behavior_density(
    cosmos_stubs: None,
) -> None:
    """Version labels cannot hide a trace scored under different weights."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    with pytest.raises(FloatingPointError, match="differs from its behavior policy"):
        trainer._validate_update_diagnostics(
            {
                "train/pre_update_valid_rows": 30,
                "train/pre_update_max_abs_log_ratio": 1.0e-3,
                "train/pre_update_max_abs_ratio_error": 1.0e-3,
            },
            phase="pre_update",
        )


def test_on_policy_ppo_accepts_exact_pre_update_behavior_density(
    cosmos_stubs: None,
) -> None:
    """Exact same-weight replay passes independently of the optional KL target."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 30,
            "train/pre_update_max_abs_log_ratio": 0.0,
            "train/pre_update_max_abs_ratio_error": 0.0,
        },
        phase="pre_update",
    )


def test_on_policy_ppo_rejects_same_weight_density_above_formal_tolerance(
    cosmos_stubs: None,
) -> None:
    """The formal same-weight replay contract is 1e-5, not the old 1e-4."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    with pytest.raises(FloatingPointError, match="exceeds 1e-05"):
        trainer._validate_update_diagnostics(
            {
                "train/pre_update_valid_rows": 30,
                "train/pre_update_max_abs_log_ratio": 5.0e-5,
                "train/pre_update_max_abs_ratio_error": 5.0e-5,
            },
            phase="pre_update",
        )


def test_on_policy_ppo_accepts_formal_tolerance_boundary(
    cosmos_stubs: None,
) -> None:
    """The documented 1e-5 same-weight boundary is inclusive."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 30,
            "train/pre_update_max_abs_log_ratio": 1.0e-5,
            "train/pre_update_max_abs_ratio_error": 1.0e-5,
        },
        phase="pre_update",
    )


def test_on_policy_ppo_accepts_critic_only_batch_without_density_rows(
    cosmos_stubs: None,
) -> None:
    """An all-unexecuted batch may still train the detached value head."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 0,
            "train/pre_update_max_abs_log_ratio": 0.0,
            "train/pre_update_max_abs_ratio_error": 0.0,
        },
        phase="pre_update",
    )


def test_on_policy_flow_ppo_rejects_wrong_behavior_critic_value(
    cosmos_stubs: None,
) -> None:
    """Actor equality cannot hide a stale or partially synced behavior critic."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymFlowPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    with pytest.raises(FloatingPointError, match="behavior critic"):
        trainer._validate_update_diagnostics(
            {
                "train/pre_update_valid_rows": 30,
                "train/pre_update_max_abs_log_ratio": 0.0,
                "train/pre_update_max_abs_ratio_error": 0.0,
                "train/pre_update_value_valid_rows": 30,
                "train/pre_update_value_max_abs_delta": 1.0e-3,
            },
            phase="pre_update",
        )


def test_on_policy_flow_ppo_accepts_exact_behavior_critic_value(
    cosmos_stubs: None,
) -> None:
    """Exact actor and critic replay passes the complete on-policy gate."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymFlowPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 30,
            "train/pre_update_max_abs_log_ratio": 0.0,
            "train/pre_update_max_abs_ratio_error": 0.0,
            "train/pre_update_value_valid_rows": 30,
            "train/pre_update_value_max_abs_delta": 0.0,
        },
        phase="pre_update",
    )


def test_on_policy_flow_ppo_rejects_critic_delta_above_formal_tolerance(
    cosmos_stubs: None,
) -> None:
    """The critic must satisfy the same 1e-5 pre-update replay contract."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymFlowPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    with pytest.raises(FloatingPointError, match="exceeds 1e-05"):
        trainer._validate_update_diagnostics(
            {
                "train/pre_update_valid_rows": 30,
                "train/pre_update_max_abs_log_ratio": 0.0,
                "train/pre_update_max_abs_ratio_error": 0.0,
                "train/pre_update_value_valid_rows": 30,
                "train/pre_update_value_max_abs_delta": 5.0e-5,
            },
            phase="pre_update",
        )


def test_on_policy_flow_ppo_accepts_critic_formal_tolerance_boundary(
    cosmos_stubs: None,
) -> None:
    """The documented 1e-5 critic replay boundary is inclusive."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymFlowPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = None

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 30,
            "train/pre_update_max_abs_log_ratio": 0.0,
            "train/pre_update_max_abs_ratio_error": 0.0,
            "train/pre_update_value_valid_rows": 30,
            "train/pre_update_value_max_abs_delta": 1.0e-5,
        },
        phase="pre_update",
    )


def test_flow_ppo_optional_kl_guard_allows_critic_only_batch(
    cosmos_stubs: None,
) -> None:
    """No actor rows means no Flow policy density moved for the KL guard."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymFlowPPOTrainer)
    trainer._on_policy = True
    trainer._target_behavior_kl = 0.1

    trainer._validate_update_diagnostics(
        {
            "train/pre_update_valid_rows": 0,
            "train/pre_update_max_abs_log_ratio": 0.0,
            "train/pre_update_max_abs_ratio_error": 0.0,
            "train/pre_update_value_valid_rows": 2,
            "train/pre_update_value_max_abs_delta": 0.0,
            "train/pre_update_approx_kl": 0.0,
        },
        phase="pre_update",
    )
    trainer._validate_update_diagnostics(
        {
            "train/post_update_valid_rows": 0,
            "train/post_update_approx_kl": 0.0,
        },
        phase="post_update",
    )


def test_ppo_post_update_behavior_kl_guard_precedes_scheduler_and_checkpoint(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad candidate update cannot advance LR state or become a checkpoint."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    events: list[str] = []
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_approx_kl": 0.01,
        "train/pre_update_valid_rows": len(samples),
    }
    trainer._post_update_diagnostics = lambda samples: {
        "train/post_update_approx_kl": 0.25,
        "train/post_update_valid_rows": len(samples),
    }

    def _record_training(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        events.append("train")
        trainer._optimizer_steps_applied_in_training_step = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 0.0)

    trainer._run_training_loop = _record_training

    with pytest.raises(FloatingPointError, match="post-update.*behavior KL"):
        trainer.step_training(
            rollouts=[object()],
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
            inter_policy_nccl=object(),
            is_master_replica=True,
        )

    assert events == ["train"]
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []


def test_ppo_behavior_kl_backtracking_scales_actor_only_before_accept(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One oversized Adam step is shrunk without weakening the critic update."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    trainer._write_ppo_update_diagnostic_receipts = True
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    receipt_path = tmp_path / "step_1_rank_0.json"
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda config: object())
    monkeypatch.setattr(
        trainer_module,
        "_ppo_update_receipt_context",
        lambda **kwargs: (receipt_path, {}),
    )
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.0, critic_weight=0.0)
    trainer.optimizers = _actor_critic_optimizer_container(trainer.model)
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 4
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }

    def post_update(samples: list[object]) -> dict[str, float | int]:
        actor_scale = float(trainer.model.actor.weight.item())
        return {
            "train/post_update_valid_rows": len(samples),
            "train/post_update_approx_kl": 0.004 * actor_scale**2,
        }

    trainer._post_update_diagnostics = post_update

    def oversized_update(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        with torch.no_grad():
            trainer.model.actor.weight.fill_(1.0)
            trainer.model.critic.weight.fill_(2.0)
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 1.0)

    trainer._run_training_loop = oversized_update

    metrics = trainer.step_training(
        rollouts=[SimpleNamespace(weight_version=0)],
        current_step=1,
        total_steps=2,
        remain_samples_num=1,
        inter_policy_nccl=object(),
        is_master_replica=True,
        do_save_checkpoint=True,
    )

    assert trainer.model.actor.weight.item() == pytest.approx(0.75)
    assert trainer.model.critic.weight.item() == pytest.approx(2.0)
    assert metrics["train/post_update_approx_kl"] == pytest.approx(0.00225)
    assert metrics["train/behavior_kl_before_backtrack"] == pytest.approx(0.004)
    assert metrics["train/behavior_kl_after_backtrack"] == pytest.approx(0.00225)
    assert metrics["train/actor_step_scale"] == pytest.approx(0.75)
    assert metrics["train/actor_effective_learning_rate"] == pytest.approx(0.075)
    assert metrics["train/actor_backtrack_attempts"] == 1
    assert metrics["train/actor_backtrack_failed"] == 0
    assert trainer._optimizer_steps_applied_in_training_step == 1
    assert trainer.lr_schedulers.steps == 1
    assert trainer.saved_checkpoints == [(1, 2, 1)]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "accepted"
    assert receipt["optimizer_metrics"]["train/actor_step_scale"] == pytest.approx(0.75)
    history = receipt["optimizer_metrics"]["train/actor_backtrack_history"]
    assert [entry["attempt"] for entry in history] == [0, 1]
    assert [entry["actor_step_scale"] for entry in history] == pytest.approx(
        [1.0, 0.75]
    )
    assert [entry["behavior_kl"] for entry in history] == pytest.approx(
        [0.004, 0.00225]
    )


def test_ppo_behavior_kl_backtracking_restores_actor_and_rejects_bad_diagnostic(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-responsive KL diagnostic restores the actor and publishes nothing."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.25, critic_weight=0.0)
    trainer.optimizers = _actor_critic_optimizer_container(trainer.model)
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 2
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }
    trainer._post_update_diagnostics = lambda samples: {
        "train/post_update_valid_rows": len(samples),
        "train/post_update_approx_kl": 0.01,
    }

    def oversized_update(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        with torch.no_grad():
            trainer.model.actor.weight.fill_(1.0)
            trainer.model.critic.weight.fill_(2.0)
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 1.0)

    trainer._run_training_loop = oversized_update

    with pytest.raises(FloatingPointError, match="backtracking exhausted"):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=2,
            remain_samples_num=1,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=True,
        )

    assert trainer.model.actor.weight.item() == pytest.approx(0.25)
    assert trainer.model.critic.weight.item() == pytest.approx(2.0)
    assert trainer._optimizer_steps_applied_in_training_step == 1
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []


def test_ppo_behavior_kl_backtracking_restores_after_initial_post_diagnostic_failure(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A diagnostic crash after Adam restores actor and emits a null-metric rejection."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    trainer._write_ppo_update_diagnostic_receipts = True
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    receipt_path = tmp_path / "step_1_rank_0.json"
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda config: object())
    monkeypatch.setattr(
        trainer_module,
        "_ppo_update_receipt_context",
        lambda **kwargs: (receipt_path, {}),
    )
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.25, critic_weight=-0.5)
    trainer.optimizers = _actor_critic_optimizer_container(trainer.model)
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 4
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }
    diagnostic_calls = 0

    def failed_post_diagnostic(samples: list[object]) -> dict[str, float | int]:
        del samples
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        raise RuntimeError("post-update replay unavailable")

    trainer._post_update_diagnostics = failed_post_diagnostic

    def mutate_once(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        with torch.no_grad():
            trainer.model.actor.weight.fill_(1.0)
            trainer.model.critic.weight.fill_(2.0)
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 1.0)

    trainer._run_training_loop = mutate_once

    with pytest.raises(ExceptionGroup, match="restored-state diagnostic failed"):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=2,
            remain_samples_num=1,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=True,
        )

    assert diagnostic_calls == 2
    assert trainer.model.actor.weight.item() == pytest.approx(0.25)
    assert trainer.model.critic.weight.item() == pytest.approx(2.0)
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "post_rejected"
    assert receipt["post_update_metrics"] is None
    assert receipt["boundary"] == {
        "optimizer_steps_applied": 1,
        "scheduler_advanced": False,
        "checkpoint_started": False,
        "weight_sync_started": False,
    }
    optimizer_metrics = receipt["optimizer_metrics"]
    assert optimizer_metrics["train/actor_restore_attempted"] == 1
    assert optimizer_metrics["train/actor_restore_succeeded"] == 1
    assert optimizer_metrics["train/actor_optimizer_state_rolled_back"] == 0
    assert "train/behavior_kl_after_restore" not in optimizer_metrics


def test_ppo_behavior_kl_backtracking_exactly_restores_nan_actor(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exact restore overwrites a non-finite Adam result instead of computing NaN*0."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    trainer._write_ppo_update_diagnostic_receipts = True
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    receipt_path = tmp_path / "step_1_rank_0.json"
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda config: object())
    monkeypatch.setattr(
        trainer_module,
        "_ppo_update_receipt_context",
        lambda **kwargs: (receipt_path, {}),
    )
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.25, critic_weight=-0.5)
    actor_before = trainer.model.actor.weight.detach().clone()
    trainer.optimizers = _actor_critic_optimizer_container(trainer.model)
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 4
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }
    diagnostic_calls = 0

    def reject_nonfinite_actor(samples: list[object]) -> dict[str, float | int]:
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        if not torch.isfinite(trainer.model.actor.weight).all():
            raise FloatingPointError("post-update actor is non-finite")
        return {
            "train/post_update_valid_rows": len(samples),
            "train/post_update_approx_kl": 0.0,
        }

    trainer._post_update_diagnostics = reject_nonfinite_actor

    def write_nan_actor(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        with torch.no_grad():
            trainer.model.actor.weight.fill_(float("nan"))
            trainer.model.critic.weight.fill_(2.0)
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 1.0)

    trainer._run_training_loop = write_nan_actor

    with pytest.raises(FloatingPointError, match="post-update actor is non-finite"):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=2,
            remain_samples_num=1,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=True,
        )

    assert diagnostic_calls == 2
    assert torch.equal(trainer.model.actor.weight.detach(), actor_before)
    assert torch.isfinite(trainer.model.actor.weight).all()
    assert trainer.model.critic.weight.item() == pytest.approx(2.0)
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "post_rejected"
    assert receipt["post_update_metrics"]["train/post_update_approx_kl"] == 0.0
    optimizer_metrics = receipt["optimizer_metrics"]
    assert optimizer_metrics["train/actor_restore_attempted"] == 1
    assert optimizer_metrics["train/actor_restore_succeeded"] == 1
    assert optimizer_metrics["train/actor_optimizer_state_rolled_back"] == 0
    assert optimizer_metrics["train/behavior_kl_after_restore"] == pytest.approx(0.0)
    assert receipt["boundary"] == {
        "optimizer_steps_applied": 1,
        "scheduler_advanced": False,
        "checkpoint_started": False,
        "weight_sync_started": False,
    }


def test_ppo_behavior_kl_backtracking_validates_all_snapshots_before_mutation(
    cosmos_stubs: None,
) -> None:
    """A malformed later snapshot cannot partially scale an earlier parameter."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    first = torch.nn.Parameter(torch.tensor([2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first_before = first.detach().clone()
    second_before = second.detach().clone()

    with pytest.raises(ValueError, match="snapshot shape changed"):
        trainer_module.AlpagymPPOTrainer._scale_actor_step_toward_snapshot_(
            [first, second],
            [torch.tensor([1.0]), torch.tensor([1.0, 1.0])],
            relative_scale=0.5,
        )

    assert torch.equal(first.detach(), first_before)
    assert torch.equal(second.detach(), second_before)


def test_ppo_behavior_kl_backtracking_restores_after_partial_scale_diagnostic_failure(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A crash after partial interpolation cannot leave the scaled actor live."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    trainer._write_ppo_update_diagnostic_receipts = True
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    receipt_path = tmp_path / "step_1_rank_0.json"
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda config: object())
    monkeypatch.setattr(
        trainer_module,
        "_ppo_update_receipt_context",
        lambda **kwargs: (receipt_path, {}),
    )
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.0, critic_weight=-0.5)
    trainer.optimizers = _actor_critic_optimizer_container(trainer.model)
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 4
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }
    diagnostic_calls = 0

    def injected_post_diagnostic(
        samples: list[object],
    ) -> dict[str, float | int]:
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        if diagnostic_calls == 1:
            assert trainer.model.actor.weight.item() == pytest.approx(1.0)
            return {
                "train/post_update_valid_rows": len(samples),
                "train/post_update_approx_kl": 0.004,
            }
        if diagnostic_calls == 2:
            assert trainer.model.actor.weight.item() == pytest.approx(0.75)
            raise RuntimeError("partial-backtrack replay unavailable")
        assert trainer.model.actor.weight.item() == pytest.approx(0.0)
        return {
            "train/post_update_valid_rows": len(samples),
            "train/post_update_approx_kl": 0.0,
        }

    trainer._post_update_diagnostics = injected_post_diagnostic

    def mutate_once(
        *args: Any,
    ) -> tuple[float, float, int, float, float, float, float]:
        del args
        with torch.no_grad():
            trainer.model.actor.weight.fill_(1.0)
            trainer.model.critic.weight.fill_(2.0)
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 1.0)

    trainer._run_training_loop = mutate_once

    with pytest.raises(RuntimeError, match="partial-backtrack replay unavailable"):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=2,
            remain_samples_num=1,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=True,
        )

    assert diagnostic_calls == 3
    assert trainer.model.actor.weight.item() == pytest.approx(0.0)
    assert trainer.model.critic.weight.item() == pytest.approx(2.0)
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "post_rejected"
    assert receipt["post_update_metrics"]["train/post_update_approx_kl"] == 0.0
    optimizer_metrics = receipt["optimizer_metrics"]
    assert optimizer_metrics["train/actor_restore_attempted"] == 1
    assert optimizer_metrics["train/actor_restore_succeeded"] == 1
    assert optimizer_metrics["train/actor_optimizer_state_rolled_back"] == 0
    assert optimizer_metrics[
        "train/behavior_kl_last_unsafe_candidate"
    ] == pytest.approx(0.004)
    assert optimizer_metrics["train/behavior_kl_after_restore"] == pytest.approx(0.0)


def test_ppo_behavior_kl_backtracking_rejects_swapped_optimizer_ownership_before_update(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor/critic optimizer ownership is checked before any weight mutation."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )

    trainer.model = _BacktrackActorCritic(actor_weight=0.25, critic_weight=-0.5)
    trainer.optimizers = _actor_critic_optimizer_container(
        trainer.model,
        swap_ownership=True,
    )
    trainer._behavior_kl_backtrack = True
    trainer._behavior_kl_backtrack_margin = 0.9
    trainer._behavior_kl_backtrack_max_attempts = 4
    trainer._target_behavior_kl = 0.003
    trainer._pre_update_diagnostics = lambda samples: {
        "train/pre_update_valid_rows": len(samples),
        "train/pre_update_approx_kl": 0.0,
    }

    optimizer_entered = False

    def unexpected_update(*args: Any) -> Any:
        del args
        nonlocal optimizer_entered
        optimizer_entered = True
        with torch.no_grad():
            trainer.model.actor.weight.fill_(99.0)
            trainer.model.critic.weight.fill_(99.0)
        raise AssertionError("optimizer ran before ownership validation")

    trainer._run_training_loop = unexpected_update

    with pytest.raises(
        ValueError,
        match="first optimizer leaf does not exactly own the actor parameters",
    ):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=2,
            remain_samples_num=1,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=True,
        )

    assert optimizer_entered is False
    assert trainer.model.actor.weight.item() == pytest.approx(0.25)
    assert trainer.model.critic.weight.item() == pytest.approx(-0.5)
    assert trainer.lr_schedulers.steps == 0
    assert trainer.saved_checkpoints == []


def test_ppo_behavior_kl_backtracking_requires_one_optimizer_iteration(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-delta interpolation is invalid after multiple optimizer steps."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    def shared_init(
        trainer: Any,
        *,
        config: object,
        parallel_dims: object,
        **kwargs: object,
    ) -> None:
        del config, parallel_dims, kwargs
        trainer._grpo_optimization_iterations = 2
        trainer._mini_batch = 1

    monkeypatch.setattr(
        trainer_module.AlpagymGRPOTrainer,
        "__init__",
        shared_init,
    )
    config = SimpleNamespace(
        custom={
            "ppo": {
                "target_behavior_kl": 0.003,
                "behavior_kl_backtrack": True,
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="requires exactly one optimizer iteration",
    ):
        trainer_module.AlpagymPPOTrainer(
            config=config,
            parallel_dims=SimpleNamespace(world_size=1),
        )


def test_final_checkpoint_respects_disabled_safetensors_export(
    cosmos_stubs: None,
) -> None:
    """Remaining samples keep a process-boundary checkpoint nonterminal."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            output_dir="/tmp/alpagym-checkpoint-test",
            param_dtype="float32",
            ckpt=SimpleNamespace(export_safetensors=False),
        )
    )
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    exports: list[dict[str, object]] = []
    manager_calls: list[tuple[str, dict[str, object]]] = []
    trainer.export_safetensors = lambda **kwargs: exports.append(kwargs)

    class _CheckpointManager:
        def save_checkpoint(self, **kwargs: object) -> None:
            manager_calls.append(("save_checkpoint", kwargs))

        def save_check(self, **kwargs: object) -> None:
            manager_calls.append(("save_check", kwargs))

    trainer.ckpt_manager = _CheckpointManager()

    trainer._save_checkpoint(current_step=1, total_steps=1, remain_samples_num=17)

    assert exports == []
    assert manager_calls[0] == (
        "save_checkpoint",
        {
            "model": trainer.model,
            "optimizer": trainer.optimizers,
            "scheduler": trainer.lr_schedulers,
            "step": 1,
            "total_steps": 1,
            "remain_samples_num": 17,
            "is_final": False,
        },
    )
    assert manager_calls[1] == ("save_check", {"step": 1})


def test_staged_checkpoint_persists_the_logical_training_horizon(
    cosmos_stubs: None,
) -> None:
    """A five-step process stage must remain resumable within a 50-step run."""

    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            output_dir="/tmp/alpagym-staged-checkpoint-test",
            param_dtype="float32",
            ckpt=SimpleNamespace(export_safetensors=False),
        )
    )
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._logical_total_training_steps = 50
    trainer._global_train_batch_size = 4
    manager_calls: list[dict[str, object]] = []
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: manager_calls.append(kwargs),
        save_check=lambda **kwargs: None,
    )

    trainer._save_checkpoint(
        current_step=5,
        total_steps=5,
        remain_samples_num=180,
    )

    assert manager_calls[0]["step"] == 5
    assert manager_calls[0]["total_steps"] == 50
    assert manager_calls[0]["remain_samples_num"] == 180
    assert manager_calls[0]["is_final"] is False


def _canonical_checkpoint_test_config(
    tmp_path: Path,
    *,
    cosmos_timestamp: str = "20260822123456",
) -> tuple[SimpleNamespace, str]:
    """Write a real host config and model Cosmos's timestamped output layout."""
    from alpagym_host.config import register_config_schema
    from alpagym_host.run_artifacts import (
        build_artifact_paths,
        build_run_config,
        write_run_artifacts,
    )
    from hydra import compose, initialize_config_module

    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        authored = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={(tmp_path / 'model').as_posix()}",
            ],
        )
    artifact_paths = build_artifact_paths(authored)
    run_config = build_run_config(authored, artifact_paths)
    write_run_artifacts(run_config)
    cosmos_output_dir = artifact_paths.run_dir / "cosmos" / cosmos_timestamp
    cosmos_output_dir.mkdir(parents=True)
    config = SimpleNamespace(
        custom={"resolved_config_path": str(artifact_paths.resolved_config_path)},
        train=SimpleNamespace(
            output_dir=str(cosmos_output_dir),
            timestamp=cosmos_timestamp,
            param_dtype="float32",
            ckpt=SimpleNamespace(export_safetensors=True),
        ),
    )
    return config, artifact_paths.run_dir.name


def test_final_checkpoint_uses_policy_native_export_hook(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """Non-generative policies bypass Cosmos generation-config discovery."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._optimizer_steps_applied_in_training_step = 1
    exported: list[tuple[object, Path, object]] = []
    events: list[str] = []

    def _record_export(model: object, path: Path, context: object) -> None:
        events.append("export")
        exported.append((model, path, context))

    trainer._policy_bundle = SimpleNamespace(export_model_checkpoint=_record_export)

    def reject_cosmos_export(**kwargs: object) -> None:
        del kwargs
        raise AssertionError("policy-native export must bypass Cosmos LLM exporter")

    trainer.export_safetensors = reject_cosmos_export
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append("resume"),
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    trainer._save_checkpoint(current_step=1, total_steps=1, remain_samples_num=0)

    assert len(exported) == 1
    assert exported[0][:2] == (
        trainer.model,
        Path(trainer.config.train.output_dir) / "safetensors" / "step_1",
    )
    export_context = exported[0][2]
    assert export_context.training_step == 1
    assert export_context.total_training_steps == 1
    assert export_context.optimizer_steps_applied == 1
    assert export_context.cosmos_run_id == formal_run_id
    assert export_context.cosmos_run_id != Path(trainer.config.train.output_dir).name
    assert export_context.cosmos_run_id != "cosmos"
    assert events == ["resume", "resume_complete", "export"]


def test_policy_native_export_failure_leaves_completed_resume_state(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """A deployable-export failure occurs only after the resume commit."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, _formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._optimizer_steps_applied_in_training_step = 1
    events: list[str] = []

    def _fail_export(*args: object) -> None:
        del args
        events.append("export")
        raise RuntimeError("candidate export failed")

    trainer._policy_bundle = SimpleNamespace(
        export_model_checkpoint=_fail_export,
    )
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append("resume"),
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    with pytest.raises(RuntimeError, match="candidate export failed"):
        trainer._save_checkpoint(
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
        )

    assert events == ["resume", "resume_complete", "export"]


def test_resume_failure_never_publishes_policy_native_candidate(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """An uncommitted Cosmos training state cannot produce a candidate."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            output_dir=str(tmp_path),
            param_dtype="float32",
            ckpt=SimpleNamespace(export_safetensors=True),
        ),
    )
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    events: list[str] = []
    trainer._policy_bundle = SimpleNamespace(
        export_model_checkpoint=lambda *args: events.append("export"),
    )

    def _fail_resume(**kwargs: object) -> None:
        del kwargs
        events.append("resume")
        raise RuntimeError("resume save failed")

    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=_fail_resume,
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    with pytest.raises(RuntimeError, match="resume save failed"):
        trainer._save_checkpoint(
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
        )

    assert events == ["resume"]


def test_checkpoint_context_rejects_authored_cosmos_root_as_run_identity(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """The literal ``cosmos`` leaf cannot become candidate provenance."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    config, _formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    formal_run_dir = Path(config.train.output_dir).parents[1]
    config.train.output_dir = str(formal_run_dir / "cosmos")
    config.train.timestamp = "cosmos"

    with pytest.raises(ValueError, match="runtime timestamp"):
        trainer_module._policy_checkpoint_export_context(
            config=config,
            current_step=1,
            total_steps=1,
            optimizer_steps_applied=1,
        )


def test_public_terminal_checkpoint_uses_native_two_file_export(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """A normal synthetic EOS cannot reach Cosmos's inherited HF exporter."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._optimizer_steps_applied_in_training_step = 1
    trainer._completed_policy_native_exports = {}
    events: list[str] = []
    exported_contexts: list[object] = []

    def _native_export(model: object, path: Path, context: object) -> None:
        assert model is trainer.model
        path.mkdir(parents=True)
        (path / "candidate_manifest.json").write_text("{}\n", encoding="utf-8")
        (path / "trainable_overlay.safetensors").write_bytes(b"weights")
        events.append("native_export")
        exported_contexts.append(context)

    trainer._policy_bundle = SimpleNamespace(export_model_checkpoint=_native_export)
    trainer.export_safetensors = lambda **kwargs: pytest.fail(
        f"generic HF exporter was called: {kwargs}"
    )
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append("resume"),
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    trainer.save_checkpoint(
        current_step=1,
        total_steps=1,
        remain_samples_num=0,
        is_final=True,
    )

    candidate = Path(trainer.config.train.output_dir) / "safetensors" / "step_1"
    assert {entry.name for entry in candidate.iterdir()} == {
        "candidate_manifest.json",
        "trainable_overlay.safetensors",
    }
    assert exported_contexts[0].cosmos_run_id == formal_run_id
    assert events == ["resume", "resume_complete", "native_export"]
    assert (
        trainer_module.AlpagymGRPOTrainer.__dict__["save_checkpoint"]
        is trainer_module.AlpagymGRPOTrainer.save_checkpoint
    )


def test_synthetic_terminal_resave_preserves_immutable_candidate(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """Real-step fallback followed by synthetic EOS exports exactly once."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, _formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._optimizer_steps_applied_in_training_step = 1
    trainer._completed_policy_native_exports = {}
    exports = 0
    resume_saves = 0

    def _native_export(model: object, path: Path, context: object) -> None:
        nonlocal exports
        del model, context
        exports += 1
        path.mkdir(parents=True)
        (path / "candidate_manifest.json").write_text("manifest\n", encoding="utf-8")
        (path / "trainable_overlay.safetensors").write_bytes(b"weights")

    def _resume_save(**kwargs: object) -> None:
        nonlocal resume_saves
        del kwargs
        resume_saves += 1

    trainer._policy_bundle = SimpleNamespace(export_model_checkpoint=_native_export)
    trainer.export_safetensors = lambda **kwargs: pytest.fail(
        f"generic HF exporter was called: {kwargs}"
    )
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=_resume_save,
        save_check=lambda **kwargs: None,
    )

    trainer._save_checkpoint(current_step=2, total_steps=2, remain_samples_num=0)
    candidate = Path(trainer.config.train.output_dir) / "safetensors" / "step_2"
    before = {entry.name: entry.read_bytes() for entry in candidate.iterdir()}
    trainer.save_checkpoint(
        current_step=2,
        total_steps=2,
        remain_samples_num=0,
        is_final=True,
    )
    after = {entry.name: entry.read_bytes() for entry in candidate.iterdir()}

    assert exports == 1
    assert resume_saves == 2
    assert before == after
    assert set(after) == {
        "candidate_manifest.json",
        "trainable_overlay.safetensors",
    }


def test_public_nonterminal_checkpoint_fails_before_writing(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """An abnormal public/synthetic call cannot publish or mark a checkpoint."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, _formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    events: list[str] = []
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append("resume"),
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    with pytest.raises(ValueError, match="synthetic terminal"):
        trainer.save_checkpoint(
            current_step=1,
            total_steps=2,
            remain_samples_num=0,
            is_final=False,
        )

    assert events == []


def test_synthetic_terminal_rejects_untracked_preexisting_candidate(
    cosmos_stubs: None,
    tmp_path: Path,
) -> None:
    """A stale/foreign step directory is never adopted or overwritten at EOS."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config, _formal_run_id = _canonical_checkpoint_test_config(tmp_path)
    trainer.model = object()
    trainer.optimizers = object()
    trainer.lr_schedulers = object()
    trainer._optimizer_steps_applied_in_training_step = 1
    trainer._completed_policy_native_exports = {}
    candidate = Path(trainer.config.train.output_dir) / "safetensors" / "step_1"
    candidate.mkdir(parents=True)
    contaminant = candidate / "foreign.bin"
    contaminant.write_bytes(b"do-not-touch")
    events: list[str] = []

    def _fail_closed_export(model: object, path: Path, context: object) -> None:
        del model, context
        events.append("native_export_attempt")
        if path.exists():
            raise FileExistsError("immutable candidate already exists")
        raise AssertionError("preexisting candidate unexpectedly disappeared")

    trainer._policy_bundle = SimpleNamespace(
        export_model_checkpoint=_fail_closed_export
    )
    trainer.export_safetensors = lambda **kwargs: pytest.fail(
        f"generic HF exporter was called: {kwargs}"
    )
    trainer.ckpt_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append("resume"),
        save_check=lambda **kwargs: events.append("resume_complete"),
    )

    with pytest.raises(FileExistsError, match="immutable"):
        trainer.save_checkpoint(
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
            is_final=True,
        )

    assert contaminant.read_bytes() == b"do-not-touch"
    assert {entry.name for entry in candidate.iterdir()} == {"foreign.bin"}
    assert events == ["resume", "resume_complete", "native_export_attempt"]


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
            _rollout(
                prompt="scene-newer", completion=newer_paths[0], weight_version=10
            ),
            _rollout(prompt="scene-older", completion=older_paths[0], weight_version=1),
            _rollout(
                prompt="scene-other-newer", completion=newer_paths[1], weight_version=9
            ),
            _rollout(
                prompt="scene-other-older", completion=older_paths[1], weight_version=2
            ),
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
            current_step=9,
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
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
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
            minibatch_advantages=torch.tensor(
                [1.0, 0.0, -2.0, -2.0], dtype=torch.float32
            ),
            inter_policy_nccl=object(),
        )


def test_reference_model_copy_is_frozen_pre_resume_policy(cosmos_stubs: None) -> None:
    """KL stays anchored to pre-resume weights instead of the restored live policy."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_WrapperShapedModel())
    trainer._kl_beta = 0.1
    with torch.no_grad():
        trainer.model.policy.weight.fill_(2.0)
    trainer.reference_state_dict = {
        key: torch.ones_like(value) for key, value in trainer.model.state_dict().items()
    }

    trainer._ensure_reference_model()

    first_reference = trainer._reference_model
    assert first_reference is not trainer.model
    assert first_reference.policy is not trainer.model.policy
    assert all(not param.requires_grad for param in first_reference.parameters())
    torch.testing.assert_close(
        first_reference.policy.weight,
        torch.ones_like(first_reference.policy.weight),
    )
    torch.testing.assert_close(
        trainer.model.policy.weight,
        torch.full_like(trainer.model.policy.weight, 2.0),
    )


def test_reference_model_requires_cosmos_initial_weights(cosmos_stubs: None) -> None:
    """KL fails closed if training starts before Cosmos establishes its anchor."""
    del cosmos_stubs
    trainer = _trainer_for_replay_test(_WrapperShapedModel())
    trainer._kl_beta = 0.1
    trainer.reference_state_dict = {}

    with pytest.raises(RuntimeError, match="weight_resume"):
        trainer._ensure_reference_model()


@pytest.mark.parametrize("value", (1, 10))
def test_replay_trainer_rejects_moving_kl_anchor(
    cosmos_stubs: None,
    value: int,
) -> None:
    """A moving reference cannot silently change meaning across Cosmos resume."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    with pytest.raises(ValueError, match="reference_reset_interval=0"):
        trainer_module._fixed_reference_reset_interval(value)

    assert trainer_module._fixed_reference_reset_interval(None) == 0
    assert trainer_module._fixed_reference_reset_interval(0) == 0


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
    self._optimizer_steps_applied_in_training_step = (
        int(getattr(self, "_optimizer_steps_applied_in_training_step", 0)) + 1
    )
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
    surr2 = (
        torch.clamp(ratio, 1.0 - grpo_ratio_clip_low, 1.0 + grpo_ratio_clip_high)
        * advantages
    )
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


def test_optimizer_learning_rates_use_nested_cosmos_leaf_groups(
    cosmos_stubs: None,
) -> None:
    """A multi-part Cosmos container must ignore its synthetic top-level group."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    actor = torch.nn.Parameter(torch.tensor(0.0))
    critic = torch.nn.Parameter(torch.tensor(0.0))
    actor_optimizer = torch.optim.AdamW([actor], lr=5.0e-8)
    critic_optimizer = torch.optim.AdamW([critic], lr=1.0e-4)
    cosmos_container = SimpleNamespace(
        # This matches the real multi-part OptimizersContainer group that
        # triggered v22: it deliberately has no direct ``lr`` key.
        param_groups=[
            {
                "params": [actor, critic],
                "optimizers_args": [{"lr": 5.0e-8}, {"lr": 1.0e-4}],
            }
        ],
        optimizers=[[actor_optimizer], [critic_optimizer]],
    )

    assert trainer_module._optimizer_learning_rates_before_scheduler(
        cosmos_container
    ) == pytest.approx([5.0e-8, 1.0e-4])


def test_optimizer_learning_rates_reject_missing_leaf_lr(
    cosmos_stubs: None,
) -> None:
    """Diagnostics fail closed when no real leaf learning rate is auditable."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    malformed_container = SimpleNamespace(
        param_groups=[{"optimizers_args": [{}]}],
        optimizers=[[SimpleNamespace(param_groups=[{"params": []}])]],
    )

    with pytest.raises(KeyError, match="leaf optimizer param group is missing lr"):
        trainer_module._optimizer_learning_rates_before_scheduler(malformed_container)


def test_receipt_optimizer_topology_fails_before_training_mutates(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unauditable optimizer topology is rejected before the update loop."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = _ppo_step_guard_trainer(trainer_module)
    trainer._write_ppo_update_diagnostic_receipts = True
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    trainer.optimizers = SimpleNamespace(
        param_groups=[{"optimizers_args": [{}]}],
        optimizers=[[SimpleNamespace(param_groups=[{"params": []}])]],
    )
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **kwargs: rollouts,
    )
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda config: object())
    monkeypatch.setattr(
        trainer_module,
        "_ppo_update_receipt_context",
        lambda **kwargs: (tmp_path / "step_1_rank_0.json", {}),
    )
    mutated = False

    def _unexpected_training(*args: Any) -> Any:
        del args
        nonlocal mutated
        mutated = True
        raise AssertionError("optimizer loop ran before topology validation")

    trainer._run_training_loop = _unexpected_training

    with pytest.raises(KeyError, match="leaf optimizer param group is missing lr"):
        trainer.step_training(
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=1,
            remain_samples_num=0,
            inter_policy_nccl=object(),
            is_master_replica=True,
        )

    assert mutated is False


class _BacktrackActorCritic(torch.nn.Module):
    """Minimal two-part model exposing the production actor/critic boundary."""

    def __init__(self, *, actor_weight: float, critic_weight: float) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(1, 1, bias=False)
        self.critic = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.actor.weight.fill_(actor_weight)
            self.critic.weight.fill_(critic_weight)

    def separate_model_parts(self) -> list[torch.nn.Module]:
        """Match the production ``[actor, critic]`` optimizer-part contract."""
        return [self.actor, self.critic]


def _actor_critic_optimizer_container(
    model: _BacktrackActorCritic,
    *,
    swap_ownership: bool = False,
) -> SimpleNamespace:
    """Build two Cosmos-style leaves with exact, auditable parameter ownership."""
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic.parameters())
    if swap_ownership:
        actor_parameters, critic_parameters = critic_parameters, actor_parameters
    actor_optimizer = SimpleNamespace(
        param_groups=[{"lr": 0.1, "params": actor_parameters}]
    )
    critic_optimizer = SimpleNamespace(
        param_groups=[{"lr": 0.2, "params": critic_parameters}]
    )
    return SimpleNamespace(optimizers=[[actor_optimizer], [critic_optimizer]])


def _ppo_step_guard_trainer(trainer_module: Any) -> Any:
    """Build a stubbed PPO trainer whose KL-guard ordering is observable."""
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.lr_schedulers = _ListLRScheduler(0.05)
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False,
        dp_shard_enabled=False,
        cp_enabled=False,
    )
    trainer._group_size = 1
    trainer._mini_batch = 1
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 100
    trainer._on_policy = False
    trainer._target_behavior_kl = 0.1
    trainer._write_ppo_update_diagnostic_receipts = False
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=1,
            ckpt=SimpleNamespace(enable_checkpoint=True),
        )
    )
    trainer._prepare_training_data = lambda rollouts: (
        [object()],
        torch.tensor([0.5], dtype=torch.float32),
    )
    trainer.saved_checkpoints = []

    def _record_save(step: int, steps: int, remaining: int) -> None:
        trainer.saved_checkpoints.append((step, steps, remaining))

    trainer._save_checkpoint = _record_save
    return trainer


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
            model_inputs={
                "ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)
            },
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
            model_inputs={
                "ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)
            },
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


def _trainer_for_ppo_accumulation_test(*, step_mini_batch: int) -> Any:
    """Build a scalar actor-critic trainer for exact accumulation checks."""
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.device = torch.device("cpu")
    trainer._reference_model = None
    trainer._grpo_ratio_clip_low = 0.2
    trainer._grpo_ratio_clip_high = 0.2
    trainer._grpo_optimization_iterations = 1
    trainer._mini_batch = step_mini_batch
    trainer._kl_beta = 0.0
    trainer._value_loss_coef = 1.0
    trainer._value_clip_range = None
    trainer._value_huber_delta = None
    trainer._flow_chunk_density = False
    trainer._dual_clip_ratio = None
    trainer._write_ppo_update_diagnostic_receipts = False
    trainer.config = SimpleNamespace(train=SimpleNamespace(seed=31))
    trainer.model = _ScalarActorCriticModel()
    trainer.optimizers = _CountingSGD(trainer.model.parameters(), lr=0.05)
    trainer.data_packer = _StackingPpoPacker()
    trainer._ensure_reference_model = MethodType(_noop_reference_model, trainer)
    trainer.all_reduce_states = MethodType(_step_without_distributed, trainer)
    return trainer


def _noop_reference_model(self: Any) -> None:
    """Leave the optional fixed-reference policy disabled in unit tests."""
    del self


def _ppo_accumulation_samples() -> tuple[list[TrainerReplayData], torch.Tensor]:
    """Return four real PPO rows with two actor and four critic targets."""
    actor_valid = (True, False, True, False)
    advantage_values = (1.0, 0.0, -2.0, 0.0)
    returns = (1.0, 2.0, -1.0, 4.0)
    samples = [
        TrainerReplayData(
            model_inputs={"features": torch.tensor([float(index + 1)])},
            training_signal=TrainingSignal(
                old_logprobs=torch.zeros(1, dtype=torch.float32),
                is_padding=torch.zeros(1, dtype=torch.bool),
                actor_valid=torch.tensor([actor_valid[index]], dtype=torch.bool),
                advantages=torch.tensor([advantage_values[index]], dtype=torch.float32),
                returns=torch.tensor([returns[index]], dtype=torch.float32),
                old_values=torch.zeros(1, dtype=torch.float32),
            ),
            rollout_id=f"accum-{index}",
            weight_version=torch.zeros((), dtype=torch.int64),
        )
        for index in range(4)
    ]
    return samples, torch.tensor(advantage_values, dtype=torch.float32)


class _CountingSGD(torch.optim.SGD):
    """SGD optimizer exposing how many parameter updates were applied."""

    def __init__(self, params: Any, *, lr: float) -> None:
        """Initialize SGD and its observable step counter."""
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        """Count and delegate one optimizer update."""
        self.step_calls += 1
        return super().step(closure)


class _StackingPpoPacker:
    """Collate the exact subset selected by the trainer microbatch loop."""

    def policy_collate_fn(
        self, samples: list[TrainerReplayData]
    ) -> TrainerReplayDataBatch:
        """Stack single-row replay objects without synthesizing extra rows."""
        return TrainerReplayDataBatch.stack(samples)


class _ScalarActorCriticModel(torch.nn.Module):
    """Independent scalar actor and critic used for accumulation equivalence."""

    def __init__(self) -> None:
        """Initialize both linear coefficients at the behavior policy."""
        super().__init__()
        self.actor_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.value_weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        features: torch.Tensor,
        return_log_prob: bool = True,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Score each row with independent actor and critic coefficients."""
        del return_log_prob, teacher_model
        flattened = features.float().reshape(-1)
        return {
            "log_probs": flattened * self.actor_weight,
            "values": flattened * self.value_weight,
            "kl_div": None,
        }


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


class _ActorCriticKlValueModel(_ActorCriticValueModel):
    """Independent actor, critic, and KL parameters for masking regressions."""

    def __init__(self) -> None:
        """Create three independently observable scalar parameters."""
        super().__init__()
        self.kl_bias = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        ego_history_xyz: torch.Tensor,
        teacher_model: Any = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return row-aligned policy, value, and differentiable KL terms."""
        result = super().forward(ego_history_xyz, teacher_model)
        result["kl_div"] = self.kl_bias.expand(ego_history_xyz.shape[0])
        return result


class _PpoPerStepPacker:
    """Packer returning configured raw-transition samples for flattening tests."""

    def __init__(
        self, rows_by_prompt: dict[str, list[tuple[float, float, bool, bool]]]
    ) -> None:
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
            for reward, old_value, terminated, is_padding in self._rows_by_prompt[
                prompt
            ]
        ]


class _PpoActorValidityPacker:
    """Two-transition rollout whose terminal sampled action never executes."""

    def get_policy_input(
        self,
        prompt: str,
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> list[TrainerReplayData]:
        """Return one actor-valid transition and one critic-only terminal row."""
        del prompt, completion, n_ignore_prefix_tokens
        return [
            TrainerReplayData(
                model_inputs={"x": torch.zeros(1, dtype=torch.float32)},
                training_signal=TrainingSignal(
                    old_logprobs=torch.zeros(1, dtype=torch.float32),
                    is_padding=torch.zeros(1, dtype=torch.bool),
                    rewards=torch.ones(1, dtype=torch.float32),
                    terminateds=torch.tensor([index == 1], dtype=torch.bool),
                    old_values=torch.zeros(1, dtype=torch.float32),
                    actor_valid=torch.tensor([index == 0], dtype=torch.bool),
                ),
                rollout_id="actor-validity",
                weight_version=torch.zeros((), dtype=torch.int64),
            )
            for index in range(2)
        ]


class _PpoActorValidityAndPaddingPacker:
    """Two real transitions followed by one packer-added visual padding row."""

    def get_policy_input(
        self,
        prompt: str,
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> list[TrainerReplayData]:
        """Return actor-valid, critic-only, then padding rows in one rollout."""
        del prompt, completion, n_ignore_prefix_tokens
        rows: list[TrainerReplayData] = []
        for index in range(3):
            is_padding = index == 2
            rows.append(
                TrainerReplayData(
                    model_inputs={"x": torch.tensor([float(index)])},
                    training_signal=TrainingSignal(
                        old_logprobs=torch.zeros(1, dtype=torch.float32),
                        is_padding=torch.tensor([is_padding], dtype=torch.bool),
                        rewards=torch.tensor(
                            [100.0 if is_padding else 1.0], dtype=torch.float32
                        ),
                        terminateds=torch.tensor([index == 1], dtype=torch.bool),
                        old_values=torch.zeros(1, dtype=torch.float32),
                        actor_valid=torch.tensor([index == 0], dtype=torch.bool),
                    ),
                    rollout_id="actor-validity-padding",
                    weight_version=torch.zeros((), dtype=torch.int64),
                )
            )
        return rows


def _direct_ppo_sample(
    reward: float,
    *,
    old_value: float,
    terminated: bool,
) -> TrainerReplayData:
    """Build one direct-policy transition for K=1 GAE comparisons."""
    return TrainerReplayData(
        model_inputs={"x": torch.zeros(1, dtype=torch.float32)},
        training_signal=TrainingSignal(
            old_logprobs=torch.zeros(1, dtype=torch.float32),
            is_padding=torch.zeros(1, dtype=torch.bool),
            rewards=torch.tensor([reward], dtype=torch.float32),
            terminateds=torch.tensor([terminated], dtype=torch.bool),
            truncateds=torch.zeros(1, dtype=torch.bool),
            old_values=torch.tensor([old_value], dtype=torch.float32),
        ),
        rollout_id="direct",
        weight_version=torch.zeros((), dtype=torch.int64),
    )


def _smdp_sample(
    *,
    rewards: tuple[float, ...],
    old_value: float,
    bootstrap_value: float,
    terminated: bool,
    width: int = 5,
) -> TrainerReplayData:
    """Build one fixed-width macro transition with a valid reward prefix."""
    primitive_rewards = torch.zeros((1, width), dtype=torch.float32)
    primitive_reward_mask = torch.zeros((1, width), dtype=torch.bool)
    primitive_rewards[0, : len(rewards)] = torch.tensor(rewards, dtype=torch.float32)
    primitive_reward_mask[0, : len(rewards)] = True
    return TrainerReplayData(
        model_inputs={"x": torch.zeros(1, dtype=torch.float32)},
        training_signal=TrainingSignal(
            old_logprobs=torch.zeros(1, dtype=torch.float32),
            is_padding=torch.zeros(1, dtype=torch.bool),
            terminateds=torch.tensor([terminated], dtype=torch.bool),
            truncateds=torch.zeros(1, dtype=torch.bool),
            old_values=torch.tensor([old_value], dtype=torch.float32),
            bootstrap_values=torch.tensor([bootstrap_value], dtype=torch.float32),
            primitive_rewards=primitive_rewards,
            primitive_reward_mask=primitive_reward_mask,
            duration_ticks=torch.tensor([len(rewards)], dtype=torch.int64),
        ),
        rollout_id="smdp",
        weight_version=torch.zeros((), dtype=torch.int64),
    )


class _PpoSignalCapturingPacker:
    """Tiny packer exposing the PPO trainer's expected collate method."""

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Return a 4-row PPO batch with one padding row."""
        del samples
        return TrainerReplayDataBatch(
            model_inputs={
                "ego_history_xyz": torch.arange(4, dtype=torch.float32).reshape(4, 1)
            },
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


class _ActorInvalidPpoPacker:
    """PPO minibatch in which every sampled action is critic-only."""

    def policy_collate_fn(self, samples: list[Any]) -> TrainerReplayDataBatch:
        """Return two forwarded rows excluded only from actor terms."""
        del samples
        return TrainerReplayDataBatch(
            model_inputs={
                "ego_history_xyz": torch.arange(2, dtype=torch.float32).reshape(2, 1)
            },
            training_signal=TrainingSignal(
                old_logprobs=torch.zeros(2, dtype=torch.float32),
                is_padding=torch.zeros(2, dtype=torch.bool),
                actor_valid=torch.zeros(2, dtype=torch.bool),
                advantages=torch.tensor([5.0, -3.0], dtype=torch.float32),
                returns=torch.tensor([2.0, 2.0], dtype=torch.float32),
                old_values=torch.zeros(2, dtype=torch.float32),
            ),
            rollout_ids=("rollout-a", "rollout-b"),
            weight_versions=torch.zeros(2, dtype=torch.int64),
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


def test_filter_rollouts_treats_previous_version_as_fresh_for_next_update(
    cosmos_stubs: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Optimizer update N consumes freshly generated behavior version N-1."""
    del cosmos_stubs
    with caplog.at_level(logging.WARNING):
        kept = filter_trainable_rollouts(
            [
                _rollout(prompt="scene-a", completion="fresh-0", weight_version=2),
                _rollout(prompt="scene-b", completion="fresh-1", weight_version=2),
            ],
            current_step=3,
            train_batch_per_replica=2,
            allowed_outdated_steps=0,
        )

    assert len(kept) == 2
    assert "stale rollouts" not in caplog.text


@pytest.mark.parametrize(
    (
        "checkpoint_enabled",
        "current_step",
        "total_steps",
        "requested",
        "master",
        "saved",
    ),
    (
        # Regression: Cosmos colocated can omit do_save on its only/final
        # DataFetchCommand. AlpaGym must still persist the applied update.
        (True, 1, 1, False, True, True),
        (False, 1, 1, False, True, False),
        (True, 1, 2, False, True, False),
        (True, 1, 2, True, True, True),
        (True, 1, 1, False, False, False),
    ),
)
def test_step_training_honors_requested_and_final_checkpoint_fallbacks(
    cosmos_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_enabled: bool,
    current_step: int,
    total_steps: int,
    requested: bool,
    master: bool,
    saved: bool,
) -> None:
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.lr_schedulers = _ListLRScheduler(0.05)
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False,
        dp_shard_enabled=False,
        cp_enabled=False,
    )
    trainer._group_size = 1
    trainer._mini_batch = 1
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 100
    trainer._on_policy = False
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            train_batch_per_replica=1,
            ckpt=SimpleNamespace(enable_checkpoint=checkpoint_enabled),
        )
    )
    monkeypatch.setattr(
        trainer_module, "filter_trainable_rollouts", lambda rollouts, **kwargs: rollouts
    )
    trainer._prepare_training_data = lambda rollouts: (
        [object()],
        torch.tensor([0.5], dtype=torch.float32),
    )

    def _successful_training_loop(
        samples: list[Any],
        advantages: torch.Tensor,
        nccl: object,
    ) -> tuple[float, float, int, float, float, float, float]:
        del samples, advantages, nccl
        trainer._optimizer_steps_applied_in_training_step = 1
        return (1.0, 0.0, 1, 1.0, 1.0, 0.0, 0.0)

    trainer._run_training_loop = _successful_training_loop
    saves: list[tuple[int, int, int]] = []

    def _record_save(step: int, steps: int, remaining: int) -> None:
        saves.append((step, steps, remaining))

    trainer._save_checkpoint = _record_save

    trainer.step_training(
        rollouts=[object()],
        current_step=current_step,
        total_steps=total_steps,
        remain_samples_num=17,
        inter_policy_nccl=object(),
        is_master_replica=master,
        do_save_checkpoint=requested,
    )

    assert saves == ([(current_step, total_steps, 17)] if saved else [])


def test_ppo_smdp_k25_keeps_literal_50hz_gamma_and_lambda(
    cosmos_stubs: None,
) -> None:
    """A 2 Hz macro raises the direct 50 Hz factors to its realized duration."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95
    samples = [
        _smdp_sample(
            rewards=(0.0,) * 25,
            old_value=0.2,
            bootstrap_value=0.3,
            terminated=False,
            width=25,
        ),
        _smdp_sample(
            rewards=(1.0,),
            old_value=0.3,
            bootstrap_value=0.0,
            terminated=True,
            width=25,
        ),
    ]

    advantages, returns = trainer._compute_gae(samples)

    final_advantage = 1.0 - 0.3
    first_delta = 0.99**25 * 0.3 - 0.2
    first_advantage = first_delta + (0.99 * 0.95) ** 25 * final_advantage
    assert advantages == pytest.approx([first_advantage, final_advantage])
    assert returns == pytest.approx([first_advantage + 0.2, 1.0])
