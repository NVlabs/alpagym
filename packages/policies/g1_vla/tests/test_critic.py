"""Tests for the RLinf-derived VLA critic overlay."""

import torch

from alpagym_g1_vla.critic import VlaValueModel
from alpagym_runtime.third_party.rlinf.value_head import ValueHead


def test_rlinf_value_head_supports_its_default_gelu_activation() -> None:
    torch.manual_seed(3)
    value_head = ValueHead(input_dim=4, hidden_sizes=(8,), output_dim=1)
    inputs = torch.randn(2, 4, requires_grad=True)

    values = value_head(inputs)

    assert values.shape == (2, 1)
    assert torch.isfinite(values).all()
    values.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_critic_uses_observation_state_and_deterministic_prefix() -> None:
    torch.manual_seed(5)
    critic = VlaValueModel(
        vlm_hidden_dim=16,
        prefix_embed_dim=8,
        hidden_sizes=(32, 16),
    )
    vlm_hidden = torch.randn(2, 7, 16)
    attention = torch.tensor([[True, True, True, True, True, False, False], [True] * 7])
    state = torch.randn(2, 1, 29)
    prefix = torch.randn(2, 30, 38)
    prefix_mask = torch.zeros(2, 30, dtype=torch.bool)
    prefix_mask[1, :7] = True

    values = critic(vlm_hidden, attention, state, prefix, prefix_mask)
    assert values.shape == (2,)
    loss = values.square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in critic.parameters() if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    for gradient in gradients:
        assert isinstance(gradient, torch.Tensor)
        assert torch.isfinite(gradient).all()
