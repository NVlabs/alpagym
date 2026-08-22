"""Source-parity tests for the RLinf Flow-PPO trainer contract."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from alpagym_runtime.cosmos.replay_objective import (
    compute_flow_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.replay import TrainingSignal


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

    metrics = trainer_module._summarize_post_update_log_ratios(
        torch.tensor([114.0], dtype=torch.float32),
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
        trainer_module._summarize_post_update_log_ratios(
            torch.tensor([1000.0], dtype=torch.float32),
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
                duration_ticks=torch.tensor([25], dtype=torch.int64),
                old_values=torch.tensor([0.0]),
                bootstrap_values=torch.tensor([0.0]),
            )
        )

    advantages, returns = trainer._compute_gae(
        [sample(terminated=False), sample(terminated=True)]
    )

    assert advantages[1] == pytest.approx(25.0)
    assert advantages[0] == pytest.approx(25.0 + 0.99 * 0.95 * 25.0)
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
                duration_ticks=torch.tensor([25], dtype=torch.int64),
                old_values=torch.tensor([3.0]),
                bootstrap_values=torch.tensor([bootstrap]),
            )
        )

    advantages, returns = trainer._compute_gae(
        [sample(truncated=True, bootstrap=5.0), sample(truncated=False, bootstrap=0.0)]
    )

    expected_first = 25.0 + 0.99 * 5.0 - 3.0
    assert advantages[0] == pytest.approx(expected_first)
    assert returns[0] == pytest.approx(expected_first + 3.0)


def test_flow_ppo_has_a_dedicated_cosmos_trainer(cosmos_stubs: None) -> None:
    """The Flow policy cannot silently select generic Gaussian PPO."""
    del cosmos_stubs
    from cosmos_rl.policy.trainer.base import TrainerRegistry

    from alpagym_runtime.cosmos.trainer import AlpagymFlowPPOTrainer

    assert TrainerRegistry.get_trainer_cls("alpagym_flow_ppo") is (
        AlpagymFlowPPOTrainer
    )
