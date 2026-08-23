"""Source-parity tests for the RLinf Flow-PPO trainer contract."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alpagym_runtime.cosmos.replay_objective import (
    compute_flow_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.replay import TrainingSignal


@pytest.mark.parametrize(
    ("state", "optimizer_metrics", "post_metrics", "rejection"),
    (
        (
            "pre_rejected",
            None,
            None,
            FloatingPointError("pre guard"),
        ),
        (
            "post_rejected",
            {"train/optimizer_steps_applied": 1, "train/loss_avg_local": 0.25},
            {"train/post_update_approx_kl": 0.02},
            FloatingPointError("post guard"),
        ),
        (
            "accepted",
            {"train/optimizer_steps_applied": 1, "train/loss_avg_local": 0.25},
            {"train/post_update_approx_kl": 0.002},
            None,
        ),
    ),
)
def test_ppo_update_diagnostic_receipt_is_three_state_atomic_and_immutable(
    cosmos_stubs: None,
    tmp_path: Path,
    state: str,
    optimizer_metrics: dict[str, float | int] | None,
    post_metrics: dict[str, float | int] | None,
    rejection: BaseException | None,
) -> None:
    """Every guard outcome publishes one self-bound receipt that cannot clobber."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    artifacts_dir = run_dir / "artifacts"
    output_dir = run_dir / "cosmos" / "20260823120001"
    artifacts_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text("formal: true\n", encoding="utf-8")
    run_config = SimpleNamespace(
        artifact_paths=SimpleNamespace(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            resolved_config_path=resolved_config,
        )
    )
    config = SimpleNamespace(
        custom={"resolved_config_path": str(resolved_config)},
        train=SimpleNamespace(
            output_dir=str(output_dir),
            timestamp="20260823120001",
        ),
    )
    arguments = {
        "config": config,
        "run_config": run_config,
        "rank": 0,
        "current_step": 1,
        "total_steps": 4,
        "state": state,
        "received_rollouts": 4,
        "trainable_rollouts": 4,
        "sample_rows": 56,
        "behavior_weight_versions": [0, 0, 0, 0],
        "is_master_replica": True,
        "do_save_checkpoint": True,
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": optimizer_metrics,
        "post_update_metrics": post_metrics,
        "rejection": rejection,
    }

    path = trainer_module._write_ppo_update_diagnostic_receipt(**arguments)

    original = path.read_bytes()
    receipt = json.loads(original)
    stored_sha256 = receipt.pop("receipt_sha256")
    assert stored_sha256 == trainer_module.canonical_json_sha256(receipt)
    assert receipt["formal_run_id"] == run_dir.name
    assert receipt["state"] == state
    assert os.stat(path).st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        trainer_module._write_ppo_update_diagnostic_receipt(**arguments)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("rejected_phase", "expected_state"),
    (
        ("pre_update", "pre_rejected"),
        ("post_update", "post_rejected"),
        (None, "accepted"),
    ),
)
def test_ppo_step_receipt_brackets_optimizer_scheduler_and_guard(
    cosmos_stubs: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rejected_phase: str | None,
    expected_state: str,
) -> None:
    """The terminal receipt is written at the exact update boundary it describes."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    run_dir = tmp_path / "20260823T120000Z-fedcba9876543210fedcba9876543210"
    artifacts_dir = run_dir / "artifacts"
    output_dir = run_dir / "cosmos" / "20260823120002"
    artifacts_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text("formal: true\n", encoding="utf-8")
    run_config = SimpleNamespace(
        artifact_paths=SimpleNamespace(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            resolved_config_path=resolved_config,
        )
    )
    config = SimpleNamespace(
        custom={"resolved_config_path": str(resolved_config)},
        train=SimpleNamespace(
            deterministic=False,
            train_batch_per_replica=4,
            output_dir=str(output_dir),
            timestamp="20260823120002",
            ckpt=SimpleNamespace(enable_checkpoint=False),
        ),
    )
    monkeypatch.setattr(trainer_module, "_load_run_config", lambda _config: run_config)
    monkeypatch.setattr(
        trainer_module,
        "filter_trainable_rollouts",
        lambda rollouts, **_kwargs: rollouts,
    )

    trainer = object.__new__(trainer_module.AlpagymPPOTrainer)
    trainer.config = config
    trainer._group_size = 4
    trainer._mini_batch = 56
    trainer._grpo_optimization_iterations = 1
    trainer._allowed_outdated_steps = 0
    trainer._on_policy = False
    trainer._target_behavior_kl = 0.003
    trainer.ckpt_manager = SimpleNamespace(global_rank=0)
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_enabled=False,
        dp_shard_enabled=False,
        cp_enabled=False,
    )
    trainer.optimizers = SimpleNamespace(param_groups=[{"lr": 1.0e-7}])
    trainer._prepare_training_data = lambda _rollouts: ([object()], torch.tensor([1.0]))
    trainer._pre_update_diagnostics = lambda _samples: {
        "train/pre_update_approx_kl": 0.0
    }
    trainer._post_update_diagnostics = lambda _samples: {
        "train/post_update_approx_kl": 0.005
    }
    optimizer_called = False

    def validate(_metrics: dict[str, float | int], *, phase: str) -> None:
        if phase == rejected_phase:
            raise FloatingPointError(f"{phase} guard")

    def run_training_loop(
        _samples: list[object],
        _advantages: torch.Tensor,
        _inter_policy_nccl: object,
    ) -> tuple[float, float, int, float, float, float, float]:
        nonlocal optimizer_called
        optimizer_called = True
        trainer._optimizer_steps_applied_in_training_step = 1
        trainer._last_micro_batches = 2
        trainer._last_ppo_advantage_metrics = {"train/ppo_advantage_raw_mean": 0.4}
        return (0.25, 0.001, 1, 1.01, 0.99, 0.1, 0.2)

    trainer._validate_update_diagnostics = validate
    trainer._run_training_loop = run_training_loop
    receipt_path = artifacts_dir / "ppo_update_diagnostics" / "step_1_rank_0.json"

    class Scheduler:
        """Assert accepted evidence exists before any scheduler mutation."""

        def __init__(self) -> None:
            self.called = False

        def step(self) -> None:
            assert json.loads(receipt_path.read_text())["state"] == "accepted"
            self.called = True

        def get_last_lr(self) -> list[float]:
            return [1.0e-7]

    scheduler = Scheduler()
    trainer.lr_schedulers = scheduler
    step_training = inspect.unwrap(trainer_module.AlpagymGRPOTrainer.step_training)

    def call() -> dict[str, object]:
        return step_training(
            trainer,
            rollouts=[SimpleNamespace(weight_version=0)],
            current_step=1,
            total_steps=4,
            remain_samples_num=3,
            inter_policy_nccl=object(),
            is_master_replica=True,
            do_save_checkpoint=False,
        )

    if rejected_phase is None:
        metrics = call()
        assert metrics["train/optimizer_steps_applied"] == 1
    else:
        with pytest.raises(FloatingPointError, match=f"{rejected_phase} guard"):
            call()

    receipt = json.loads(receipt_path.read_text())
    assert receipt["state"] == expected_state
    assert optimizer_called is (rejected_phase != "pre_update")
    assert scheduler.called is (rejected_phase is None)
    if expected_state == "post_rejected":
        assert receipt["post_update_metrics"]["train/post_update_approx_kl"] == 0.005
        assert receipt["optimizer_metrics"]["train/grad_norm"] == 0.2


def test_formal_trainer_determinism_is_fail_closed(
    cosmos_stubs: None,
) -> None:
    """Formal PPO must reject nondeterminism instead of merely warning."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        trainer_module._enforce_strict_training_determinism(
            SimpleNamespace(train=SimpleNamespace(deterministic=True))
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        trainer_module._assert_strict_training_determinism(
            SimpleNamespace(train=SimpleNamespace(deterministic=True))
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )


def test_optional_trainer_determinism_does_not_mutate_torch(
    cosmos_stubs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-formal configurations retain upstream Cosmos behavior."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda mode, *, warn_only=False: calls.append((mode, warn_only)),
    )
    trainer_module._enforce_strict_training_determinism(
        SimpleNamespace(train=SimpleNamespace(deterministic=False))
    )
    assert calls == []


def test_formal_trainer_rejects_warn_only_determinism(
    cosmos_stubs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The update boundary detects an upstream relaxation to warn-only mode."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: True,
    )
    with pytest.raises(RuntimeError, match="became warn-only"):
        trainer_module._assert_strict_training_determinism(
            SimpleNamespace(train=SimpleNamespace(deterministic=True))
        )


def test_flow_ppo_uses_one_joint_chunk_ratio(cosmos_stubs: None) -> None:
    """Element densities are summed before one asymmetric PPO clip."""
    del cosmos_stubs
    element_ratios = torch.tensor([[1.4, 1.0], [0.8, 1.0]], dtype=torch.float32)
    old_elements = torch.zeros_like(element_ratios)
    new_elements = element_ratios.log()
    old_joint = old_elements.sum(dim=-1)
    new_joint = new_elements.sum(dim=-1)

    loss, ratios = compute_flow_ppo_surrogate(
        new_elements,
        old_elements,
        new_joint,
        old_joint,
        torch.tensor([1.0, -2.0], dtype=torch.float32),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=torch.zeros(2, dtype=torch.bool),
    )

    torch.testing.assert_close(ratios, torch.tensor([1.4, 0.8]))
    torch.testing.assert_close(loss, torch.tensor((-1.28 + 1.6) / 2.0))


def test_flow_ppo_cancelling_element_ratios_form_one_unit_ratio(
    cosmos_stubs: None,
) -> None:
    """Opposite element log-ratios cancel before the single PPO ratio."""
    del cosmos_stubs
    old_elements = torch.zeros((1, 2), dtype=torch.float32)
    new_elements = torch.tensor([[0.7, -0.7]], dtype=torch.float32)

    loss, ratios = compute_flow_ppo_surrogate(
        new_elements,
        old_elements,
        new_elements.sum(dim=-1),
        old_elements.sum(dim=-1),
        torch.tensor([2.0], dtype=torch.float32),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=torch.zeros(1, dtype=torch.bool),
    )

    torch.testing.assert_close(ratios, torch.ones(1))
    torch.testing.assert_close(loss, torch.tensor(-2.0))


def test_flow_ppo_dual_clip_matches_rlinf(cosmos_stubs: None) -> None:
    """RLinf's dual clip bounds a large adverse joint importance ratio."""
    del cosmos_stubs
    new_elements = torch.log(torch.tensor([[10.0, 1.0]], dtype=torch.float32))
    old_elements = torch.zeros_like(new_elements)
    loss, _ratios = compute_flow_ppo_surrogate(
        new_elements,
        old_elements,
        new_elements.sum(dim=-1),
        old_elements.sum(dim=-1),
        torch.tensor([-2.0]),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=torch.zeros(1, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, torch.tensor(6.0))


def test_flow_ppo_large_joint_delta_has_finite_loss_and_gradient(
    cosmos_stubs: None,
) -> None:
    """A 1,140-factor joint density cannot overflow the PPO exponent."""
    del cosmos_stubs
    old_elements = torch.zeros((1, 1140), dtype=torch.float32)
    new_elements = torch.full(
        (1, 1140),
        0.1,
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, ratios = compute_flow_ppo_surrogate(
        new_elements,
        old_elements,
        new_elements.sum(dim=-1),
        old_elements.sum(dim=-1),
        torch.tensor([1.0], dtype=torch.float32),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=torch.zeros(1, dtype=torch.bool),
    )
    loss.backward()

    assert torch.isfinite(loss)
    torch.testing.assert_close(ratios, torch.tensor([5.0]).exp())
    assert new_elements.grad is not None
    assert torch.isfinite(new_elements.grad).all()


def test_flow_ppo_diagnostics_keep_raw_joint_approx_kl(
    cosmos_stubs: None,
) -> None:
    """The objective clamp cannot hide a large finite post-update divergence."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    metrics = trainer_module._summarize_behavior_log_ratios(
        torch.tensor([114.0], dtype=torch.float32),
        phase="post_update",
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
    )

    assert metrics["train/post_update_ratio_p50"] == pytest.approx(math.exp(5.0))
    assert metrics["train/post_update_approx_kl"] == pytest.approx(
        math.expm1(114.0) - 114.0
    )


def test_flow_ppo_diagnostics_reject_nonfinite_raw_joint_approx_kl(
    cosmos_stubs: None,
) -> None:
    """A finite log-ratio whose raw KL overflows fails closed as divergence."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")

    with pytest.raises(FloatingPointError, match="approximate KL is non-finite"):
        trainer_module._summarize_behavior_log_ratios(
            torch.tensor([1000.0], dtype=torch.float32),
            phase="post_update",
            ratio_clip_low=0.2,
            ratio_clip_high=0.28,
        )


def test_flow_ppo_actor_mask_excludes_unexecuted_chunk(cosmos_stubs: None) -> None:
    """The Flow surrogate ignores a sampled chunk that never reached control."""
    del cosmos_stubs
    new_elements = torch.log(
        torch.tensor([[10.0, 1.0], [1.1, 1.0]], dtype=torch.float32)
    )
    old_elements = torch.zeros_like(new_elements)
    loss, ratios = compute_flow_ppo_surrogate(
        new_elements,
        old_elements,
        new_elements.sum(dim=-1),
        old_elements.sum(dim=-1),
        torch.tensor([100.0, 2.0]),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=torch.tensor([True, False]),
    )

    torch.testing.assert_close(ratios, torch.tensor([0.0, 1.1]))
    torch.testing.assert_close(loss, torch.tensor(-2.2))


def test_flow_ppo_requires_explicit_actor_valid(cosmos_stubs: None) -> None:
    """A missing ownership bit cannot default a Flow chunk into actor training."""
    del cosmos_stubs
    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    signal = TrainingSignal(
        old_logprobs=torch.zeros(1, dtype=torch.float32),
        actor_valid=None,
        is_padding=torch.zeros(1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="requires TrainingSignal.actor_valid"):
        trainer_module._ppo_actor_valid_mask(signal, required=True)


def test_flow_ppo_value_loss_uses_clipped_huber(cosmos_stubs: None) -> None:
    """The critic follows RLinf's max(unclipped, clipped) Huber objective."""
    del cosmos_stubs
    loss = compute_value_loss(
        values=torch.tensor([2.0]),
        returns=torch.tensor([0.0]),
        is_padding=torch.zeros(1, dtype=torch.bool),
        old_values=torch.tensor([0.0]),
        value_clip_range=0.2,
        huber_delta=10.0,
    )
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_flow_ppo_gae_uses_policy_chunk_clock(cosmos_stubs: None) -> None:
    """A nominal 25-tick interval is one 2 Hz decision, not 25 GAE steps."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95

    def sample(*, terminated: bool) -> SimpleNamespace:
        """Build one nominal 25-tick training signal for the GAE test."""
        return SimpleNamespace(
            training_signal=SimpleNamespace(
                is_padding=torch.tensor([False]),
                rewards=torch.tensor([25.0]),
                terminateds=torch.tensor([terminated]),
                truncateds=torch.tensor([False]),
                primitive_rewards=torch.ones((1, 25), dtype=torch.float32),
                primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
                actor_primitive_rewards=torch.ones((1, 25), dtype=torch.float32),
                actor_primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
                duration_ticks=torch.tensor([25], dtype=torch.int64),
                old_values=torch.tensor([0.0]),
                bootstrap_values=torch.tensor([0.0]),
            )
        )

    advantages, returns = trainer._compute_gae(
        [sample(terminated=False), sample(terminated=True)]
    )

    gamma_tick = 0.99 ** (1.0 / 25.0)
    transition_reward = sum(gamma_tick**index for index in range(25))
    assert advantages[1] == pytest.approx(transition_reward)
    assert advantages[0] == pytest.approx(
        transition_reward + 0.99 * 0.95 * transition_reward
    )
    assert returns == pytest.approx(advantages)


def test_flow_ppo_gae_scales_gamma_for_late_plan_duration(
    cosmos_stubs: None,
) -> None:
    """A 50-tick late-plan interval uses two nominal decision discounts."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95
    sample = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=torch.tensor([0.0]),
            terminateds=torch.tensor([False]),
            truncateds=torch.tensor([True]),
            primitive_rewards=torch.zeros((1, 50), dtype=torch.float32),
            primitive_reward_mask=torch.ones((1, 50), dtype=torch.bool),
            actor_primitive_rewards=torch.zeros((1, 50), dtype=torch.float32),
            actor_primitive_reward_mask=torch.ones((1, 50), dtype=torch.bool),
            duration_ticks=torch.tensor([50], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([10.0]),
        )
    )

    advantages, returns = trainer._compute_gae([sample])

    expected = 0.99**2 * 10.0
    assert advantages == pytest.approx([expected])
    assert returns == pytest.approx([expected])


def test_flow_ppo_truncation_bootstraps_but_breaks_trace(cosmos_stubs: None) -> None:
    """A time-limit chunk bootstraps its delta without tracing into another episode."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95

    def sample(*, truncated: bool, bootstrap: float) -> SimpleNamespace:
        """Build one nominal interval with an explicit boundary bootstrap."""
        return SimpleNamespace(
            training_signal=SimpleNamespace(
                is_padding=torch.tensor([False]),
                rewards=torch.tensor([25.0]),
                terminateds=torch.tensor([False]),
                truncateds=torch.tensor([truncated]),
                primitive_rewards=torch.ones((1, 25), dtype=torch.float32),
                primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
                actor_primitive_rewards=torch.ones((1, 25), dtype=torch.float32),
                actor_primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
                duration_ticks=torch.tensor([25], dtype=torch.int64),
                old_values=torch.tensor([3.0]),
                bootstrap_values=torch.tensor([bootstrap]),
            )
        )

    advantages, returns = trainer._compute_gae(
        [sample(truncated=True, bootstrap=5.0), sample(truncated=False, bootstrap=0.0)]
    )

    gamma_tick = 0.99 ** (1.0 / 25.0)
    transition_reward = sum(gamma_tick**index for index in range(25))
    expected_first = transition_reward + 0.99 * 5.0 - 3.0
    assert advantages[0] == pytest.approx(expected_first)
    assert returns[0] == pytest.approx(expected_first + 3.0)


@pytest.mark.parametrize("duration_ticks", (0, 12, 25))
def test_flow_ppo_discount_preserves_physical_time(
    cosmos_stubs: None,
    duration_ticks: int,
) -> None:
    """A policy-clock factor has a well-defined 50 Hz duration, including D=0."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import _time_scaled_discount

    factor = _time_scaled_discount(
        0.99,
        duration_ticks=duration_ticks,
        nominal_duration_ticks=25,
    )

    assert factor == pytest.approx(0.99 ** (duration_ticks / 25.0))


def test_flow_ppo_actor_excludes_preinstall_reward_without_rebasing(
    cosmos_stubs: None,
) -> None:
    """Actor sees only its causal suffix while critic retains chronological return."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95
    rewards = torch.tensor([10.0] * 12 + [1.0] * 25, dtype=torch.float32)
    actor_mask = torch.tensor([False] * 12 + [True] * 25, dtype=torch.bool)
    sample = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=rewards.sum().reshape(1),
            terminateds=torch.tensor([True]),
            truncateds=torch.tensor([False]),
            primitive_rewards=rewards.reshape(1, -1),
            primitive_reward_mask=torch.ones((1, 37), dtype=torch.bool),
            actor_primitive_rewards=rewards.reshape(1, -1),
            actor_primitive_reward_mask=actor_mask.reshape(1, -1),
            duration_ticks=torch.tensor([37], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            # A true terminal must ignore even a nonzero transported bootstrap.
            bootstrap_values=torch.tensor([1000.0]),
        )
    )

    advantages, returns = trainer._compute_gae([sample])

    gamma_tick = 0.99 ** (1.0 / 25.0)
    expected_actor = sum(gamma_tick**index for index in range(12, 37))
    expected_critic = sum(
        (10.0 if index < 12 else 1.0) * gamma_tick**index for index in range(37)
    )
    assert advantages == pytest.approx([expected_actor])
    assert returns == pytest.approx([expected_critic])


def test_flow_ppo_next_predecessor_reward_credits_previous_plan_only(
    cosmos_stubs: None,
) -> None:
    """A row-two predecessor prefix stays downstream credit for row-one action."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95

    first = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=torch.tensor([0.0]),
            terminateds=torch.tensor([False]),
            truncateds=torch.tensor([False]),
            primitive_rewards=torch.zeros((1, 25), dtype=torch.float32),
            primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
            actor_primitive_rewards=torch.zeros((1, 25), dtype=torch.float32),
            actor_primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
            duration_ticks=torch.tensor([25], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([0.0]),
        )
    )
    second_rewards = torch.tensor([10.0] * 12 + [0.0] * 25, dtype=torch.float32)
    second = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=second_rewards.sum().reshape(1),
            terminateds=torch.tensor([True]),
            truncateds=torch.tensor([False]),
            primitive_rewards=second_rewards.reshape(1, -1),
            primitive_reward_mask=torch.ones((1, 37), dtype=torch.bool),
            actor_primitive_rewards=second_rewards.reshape(1, -1),
            actor_primitive_reward_mask=torch.tensor(
                [[False] * 12 + [True] * 25], dtype=torch.bool
            ),
            duration_ticks=torch.tensor([37], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([0.0]),
        )
    )

    advantages, returns = trainer._compute_gae([first, second])

    gamma_tick = 0.99 ** (1.0 / 25.0)
    second_prefix_return = sum(10.0 * gamma_tick**index for index in range(12))
    expected_first_actor = 0.99 * 0.95 * second_prefix_return
    # Row two did not control its predecessor prefix, but row one's reference did.
    assert advantages == pytest.approx([expected_first_actor, 0.0])
    assert returns == pytest.approx([expected_first_actor, second_prefix_return])


def test_flow_ppo_support_majority_gain_credits_previous_plan_only(
    cosmos_stubs: None,
) -> None:
    """A boundary certification stays in critic chronology but not the new actor."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95
    first_rewards = torch.zeros((1, 25), dtype=torch.float32)
    first = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=torch.tensor([0.0]),
            terminateds=torch.tensor([False]),
            truncateds=torch.tensor([False]),
            primitive_rewards=first_rewards,
            primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
            actor_primitive_rewards=first_rewards,
            actor_primitive_reward_mask=torch.ones((1, 25), dtype=torch.bool),
            duration_ticks=torch.tensor([25], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([0.0]),
        )
    )
    critic_rewards = torch.zeros((1, 5), dtype=torch.float32)
    critic_rewards[0, 3] = 1.0
    second = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=torch.tensor([1.0]),
            terminateds=torch.tensor([True]),
            truncateds=torch.tensor([False]),
            primitive_rewards=critic_rewards,
            primitive_reward_mask=torch.ones((1, 5), dtype=torch.bool),
            actor_primitive_rewards=torch.zeros((1, 5), dtype=torch.float32),
            actor_primitive_reward_mask=torch.tensor(
                [[False, False, False, True, True]], dtype=torch.bool
            ),
            duration_ticks=torch.tensor([5], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([0.0]),
        )
    )

    advantages, returns = trainer._compute_gae([first, second])

    gamma_tick = 0.99 ** (1.0 / 25.0)
    event_return = gamma_tick**3
    assert advantages == pytest.approx([0.99 * 0.95 * event_return, 0.0])
    assert returns == pytest.approx([0.99 * 0.95 * event_return, event_return])


def test_flow_ppo_twelve_tick_truncation_bootstraps_in_physical_time(
    cosmos_stubs: None,
) -> None:
    """A partial interval bootstraps at gamma_plan**(12/25) and ends its trace."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95
    primitive_mask = torch.tensor([True] * 12 + [False] * 13, dtype=torch.bool)
    sample = SimpleNamespace(
        training_signal=SimpleNamespace(
            is_padding=torch.tensor([False]),
            rewards=torch.tensor([0.0]),
            terminateds=torch.tensor([False]),
            truncateds=torch.tensor([True]),
            primitive_rewards=torch.zeros((1, 25), dtype=torch.float32),
            primitive_reward_mask=primitive_mask.reshape(1, -1),
            actor_primitive_rewards=torch.zeros((1, 25), dtype=torch.float32),
            actor_primitive_reward_mask=primitive_mask.reshape(1, -1),
            duration_ticks=torch.tensor([12], dtype=torch.int64),
            old_values=torch.tensor([0.0]),
            bootstrap_values=torch.tensor([10.0]),
        )
    )

    advantages, returns = trainer._compute_gae([sample])

    expected = 0.99 ** (12.0 / 25.0) * 10.0
    assert advantages == pytest.approx([expected])
    assert returns == pytest.approx([expected])


def test_flow_ppo_lambda_scales_with_variable_physical_duration(
    cosmos_stubs: None,
) -> None:
    """GAE lambda spans 12 controller ticks, not one arbitrary replay row."""
    del cosmos_stubs
    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    trainer = object.__new__(AlpagymFlowPPOTrainer)
    trainer._gamma = 0.99
    trainer._gae_lambda = 0.95

    def sample(*, duration: int, reward: float, terminated: bool) -> SimpleNamespace:
        primitive_mask = torch.tensor(
            [True] * duration + [False] * (25 - duration), dtype=torch.bool
        )
        primitive_rewards = torch.zeros((1, 25), dtype=torch.float32)
        primitive_rewards[0, :duration] = reward
        return SimpleNamespace(
            training_signal=SimpleNamespace(
                is_padding=torch.tensor([False]),
                rewards=torch.tensor([reward * duration]),
                terminateds=torch.tensor([terminated]),
                truncateds=torch.tensor([False]),
                primitive_rewards=primitive_rewards,
                primitive_reward_mask=primitive_mask.reshape(1, -1),
                actor_primitive_rewards=primitive_rewards,
                actor_primitive_reward_mask=primitive_mask.reshape(1, -1),
                duration_ticks=torch.tensor([duration], dtype=torch.int64),
                old_values=torch.tensor([0.0]),
                bootstrap_values=torch.tensor([0.0]),
            )
        )

    advantages, returns = trainer._compute_gae(
        [
            sample(duration=12, reward=0.0, terminated=False),
            sample(duration=25, reward=1.0, terminated=True),
        ]
    )

    gamma_tick = 0.99 ** (1.0 / 25.0)
    final_advantage = sum(gamma_tick**index for index in range(25))
    expected_first = (0.99 * 0.95) ** (12.0 / 25.0) * final_advantage
    assert advantages == pytest.approx([expected_first, final_advantage])
    assert returns == pytest.approx(advantages)


def test_flow_ppo_has_a_dedicated_cosmos_trainer(cosmos_stubs: None) -> None:
    """The Flow policy cannot silently select generic Gaussian PPO."""
    del cosmos_stubs
    from cosmos_rl.policy.trainer.base import TrainerRegistry

    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    assert TrainerRegistry.get_trainer_cls("alpagym_flow_ppo") is (
        AlpagymFlowPPOTrainer
    )
