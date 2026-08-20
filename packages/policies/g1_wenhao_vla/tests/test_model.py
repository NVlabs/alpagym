"""Behavior tests for the native Wenhao Psi actor-critic core."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from alpagym_g1_wenhao_vla.flow import WenhaoFlowSchedule
from alpagym_g1_wenhao_vla.model import WenhaoPsiActorCritic
from alpagym_g1_wenhao_vla.normalization import WenhaoQ99Normalizer


class _FakeVlm(torch.nn.Module):
    """Tiny frozen condition model with dropout to expose mode mistakes."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.dropout = torch.nn.Dropout(p=0.8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token conditions for integer input ids."""
        return self.dropout(self.embedding(input_ids))


class _FakeActionHead(torch.nn.Module):
    """Tiny trainable velocity head with dropout and the Psi call signature."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.dropout = torch.nn.Dropout(p=0.7)
        self.calls = 0
        self.last_states: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: None,
        timestep: torch.Tensor,
        joint_attention_kwargs: dict[str, torch.Tensor | None],
        vlm_attn_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Return a Psi-shaped flow velocity output."""
        del hidden_states, vlm_attn_mask, return_dict
        self.calls += 1
        latent = joint_attention_kwargs["action_hidden_embeds"]
        views = joint_attention_kwargs["views"]
        states = joint_attention_kwargs["obs"]
        assert latent is not None and views is not None and states is not None
        self.last_states = states.detach().clone()
        if timestep.ndim == 1:
            timestep_feature = timestep[:, None, None]
        else:
            timestep_feature = timestep[..., None]
        condition = views.mean(dim=(1, 2, 3))[:, None, None]
        proprio = states.mean(dim=(1, 2))[:, None, None]
        velocity = (
            self.dropout(latent) * self.scale
            + condition * 0.01
            + proprio * 0.02
            + timestep_feature.to(latent.dtype) / 10_000.0
        )
        return SimpleNamespace(action=velocity)


class _FakePsi(torch.nn.Module):
    """Fake exposing only the native Psi condition/action-head surface."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm_model = _FakeVlm()
        self.action_header = _FakeActionHead()
        self.condition_calls = 0

    def _vlm_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        effective_image_grid_thw: torch.Tensor | None,
        visual_pool_factors: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mirror Psi's condition entry point for adapter tests."""
        del (
            attention_mask,
            pixel_values,
            image_grid_thw,
            effective_image_grid_thw,
            visual_pool_factors,
        )
        self.condition_calls += 1
        return self.vlm_model(input_ids)

    def _select_vlm_planner_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror Wenhao's current identity planner-token selection."""
        return hidden, attention_mask


def _schedule() -> WenhaoFlowSchedule:
    return WenhaoFlowSchedule(
        model_timesteps=torch.tensor(
            [1000.0, 889.0, 778.0, 667.0, 556.0, 445.0, 334.0, 223.0, 112.0, 1.0]
        ),
        sigmas=torch.tensor(
            [1.0, 0.889, 0.778, 0.667, 0.556, 0.445, 0.334, 0.223, 0.112, 0.001, 0.0]
        ),
    )


def _normalizer() -> WenhaoQ99Normalizer:
    return WenhaoQ99Normalizer(
        state_q01=torch.zeros(29),
        state_q99=torch.full((29,), 2.0),
        action_q01=torch.linspace(-2.0, -1.0, 38),
        action_q99=torch.linspace(1.0, 2.0, 38),
    )


def _model() -> WenhaoPsiActorCritic:
    psi = _FakePsi()
    psi.vlm_model.requires_grad_(False)
    return WenhaoPsiActorCritic(
        psi_model=psi,
        schedule=_schedule(),
        noise_level=0.4,
        normalizer=_normalizer(),
        vlm_hidden_dim=8,
        rtc_max_delay_exclusive=8,
        critic_hidden_sizes=(32, 16),
    )


def _inputs(model: WenhaoPsiActorCritic) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(23)
    latent_chain = torch.randn((2, 11, 30, 38), generator=generator) * 0.3
    latent_chain[0, -1, 0, 0] = 1.25
    latent_chain[1, -1, 0, 1] = -1.4
    clipped, wire = model.materialize_action_views(latent_chain[:, -1])
    return {
        "input_ids": torch.randint(0, 32, (2, 6), generator=generator),
        "attention_mask": torch.tensor(
            [[True, True, True, True, False, False], [True] * 6]
        ),
        "pixel_values": torch.randn((4, 4), generator=generator),
        "image_grid_thw": torch.ones((4, 3), dtype=torch.int64),
        "sequence_lengths": torch.tensor([4, 6]),
        "image_counts": torch.tensor([2, 2]),
        "image_offsets": torch.tensor([0, 2, 4]),
        "image_patch_counts": torch.ones(4, dtype=torch.int64),
        "patch_counts": torch.tensor([2, 2]),
        "patch_offsets": torch.tensor([0, 2, 4]),
        "physical_states": torch.randn((2, 1, 29), generator=generator),
        "latent_chain": latent_chain,
        "denoise_indices": torch.tensor([2, 7]),
        "flow_schedule_sha256": model.schedule.sha256,
        "clipped_normalized_actions": clipped,
        "denormalized_wire_actions": wire,
        "rtc_prefix_normalized_actions": torch.zeros((2, 30, 38)),
        "rtc_prefix_mask": torch.zeros((2, 30), dtype=torch.bool),
    }


def test_constructor_requires_frozen_vlm_and_trainable_action_head() -> None:
    psi = _FakePsi()
    with pytest.raises(ValueError, match="requires a frozen VLM"):
        WenhaoPsiActorCritic(
            psi_model=psi,
            schedule=_schedule(),
            noise_level=0.4,
            normalizer=_normalizer(),
            vlm_hidden_dim=8,
            rtc_max_delay_exclusive=8,
            critic_hidden_sizes=(32, 16),
        )

    psi.vlm_model.requires_grad_(False)
    psi.action_header.requires_grad_(False)
    with pytest.raises(ValueError, match="action_header must have trainable"):
        WenhaoPsiActorCritic(
            psi_model=psi,
            schedule=_schedule(),
            noise_level=0.4,
            normalizer=_normalizer(),
            vlm_hidden_dim=8,
            rtc_max_delay_exclusive=8,
            critic_hidden_sizes=(32, 16),
        )


def test_actor_stays_eval_while_ppo_gradients_flow() -> None:
    model = _model()
    inputs = _inputs(model)
    model.train()
    psi = model.psi_model
    assert isinstance(psi, _FakePsi)

    assert model.training
    assert model.critic.training
    assert not psi.training
    assert not psi.action_header.training
    first = model(**inputs)
    second = model(**inputs)
    assert first["log_probs"] is not None
    assert first["log_probs"].shape == (2,)
    assert first["values"] is not None
    assert first["element_log_probs"] is not None
    assert first["element_log_probs"].shape == (2, 30 * 38)
    torch.testing.assert_close(
        first["element_log_probs"].sum(dim=-1), first["log_probs"]
    )
    torch.testing.assert_close(first["log_probs"], second["log_probs"])
    torch.testing.assert_close(first["element_log_probs"], second["element_log_probs"])
    assert psi.condition_calls == 2
    assert psi.action_header.calls == 2
    assert psi.action_header.last_states is not None
    torch.testing.assert_close(
        psi.action_header.last_states,
        model.normalizer.normalize_state(inputs["physical_states"]),
    )

    loss = -first["log_probs"].mean() + first["values"].square().mean()
    loss.backward()
    action_gradient = psi.action_header.scale.grad
    assert isinstance(action_gradient, torch.Tensor)
    assert torch.isfinite(action_gradient)
    assert float(action_gradient.abs()) > 0.0
    critic_gradients = [
        parameter.grad
        for parameter in model.critic.parameters()
        if parameter.requires_grad
    ]
    assert critic_gradients and all(
        gradient is not None for gradient in critic_gradients
    )


def test_rollout_sample_replays_exactly_under_the_same_weights() -> None:
    model = _model().train()
    inputs = _inputs(model)
    prefix_actions = torch.zeros((2, 30, 38))
    prefix_mask = torch.zeros((2, 30), dtype=torch.bool)
    sample = model.sample_actions(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        sequence_lengths=inputs["sequence_lengths"],
        image_counts=inputs["image_counts"],
        image_offsets=inputs["image_offsets"],
        image_patch_counts=inputs["image_patch_counts"],
        patch_counts=inputs["patch_counts"],
        patch_offsets=inputs["patch_offsets"],
        physical_states=inputs["physical_states"],
        rtc_prefix_normalized_actions=prefix_actions,
        rtc_prefix_mask=prefix_mask,
        generator=torch.Generator().manual_seed(101),
    )

    assert sample.density_latent.shape == (2, 30, 38)
    assert sample.trace.chain.shape == (2, 11, 30, 38)
    assert sample.trace.denoise_indices.shape == (2,)
    assert sample.old_element_logprobs.shape == (2, 30 * 38)
    assert sample.old_log_probs.shape == (2,)
    torch.testing.assert_close(
        sample.old_element_logprobs.sum(dim=-1), sample.old_log_probs
    )
    assert sample.values.shape == (2,)
    assert not sample.density_latent.requires_grad
    assert not sample.old_log_probs.requires_grad
    assert not sample.values.requires_grad
    assert not model.psi_model.training
    replayed = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        sequence_lengths=inputs["sequence_lengths"],
        image_counts=inputs["image_counts"],
        image_offsets=inputs["image_offsets"],
        image_patch_counts=inputs["image_patch_counts"],
        patch_counts=inputs["patch_counts"],
        patch_offsets=inputs["patch_offsets"],
        physical_states=inputs["physical_states"],
        latent_chain=sample.trace.chain,
        denoise_indices=sample.trace.denoise_indices,
        flow_schedule_sha256=sample.trace.schedule_sha256,
        clipped_normalized_actions=sample.clipped_normalized,
        denormalized_wire_actions=sample.denormalized_wire,
        rtc_prefix_normalized_actions=prefix_actions,
        rtc_prefix_mask=prefix_mask,
    )
    assert replayed["log_probs"] is not None
    assert replayed["values"] is not None
    assert replayed["element_log_probs"] is not None
    torch.testing.assert_close(
        replayed["element_log_probs"], sample.old_element_logprobs, rtol=0, atol=0
    )
    torch.testing.assert_close(
        replayed["log_probs"], sample.old_log_probs, rtol=0, atol=0
    )
    torch.testing.assert_close(replayed["values"], sample.values, rtol=0, atol=0)


def test_density_uses_preclip_chain_and_wire_views_fail_closed() -> None:
    model = _model().eval()
    inputs = _inputs(model)
    baseline = model(**inputs)["log_probs"]
    assert baseline is not None
    assert inputs["clipped_normalized_actions"][0, 0, 0] == 1.0
    assert inputs["clipped_normalized_actions"][1, 0, 1] == -1.0

    changed_inputs = dict(inputs)
    changed_chain = inputs["latent_chain"].clone()
    changed_chain[0, 2] += 0.15
    changed_chain[1, 7] -= 0.1
    changed_inputs["latent_chain"] = changed_chain
    changed = model(**changed_inputs)["log_probs"]
    assert changed is not None
    assert not torch.allclose(baseline, changed)

    wrong_clipped = dict(inputs)
    wrong_clipped["clipped_normalized_actions"] = inputs[
        "clipped_normalized_actions"
    ].clone()
    wrong_clipped["clipped_normalized_actions"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="clipped normalized action"):
        model(**wrong_clipped)

    wrong_wire = dict(inputs)
    wrong_wire["denormalized_wire_actions"] = inputs[
        "denormalized_wire_actions"
    ].clone()
    wrong_wire["denormalized_wire_actions"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="denormalized wire action"):
        model(**wrong_wire)

    wrong_schedule = dict(inputs)
    wrong_schedule["flow_schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="flow schedule identity"):
        model(**wrong_schedule)


def test_rtc_replay_requires_frozen_chain_but_critic_accepts_prefix() -> None:
    model = _model().eval()
    inputs = _inputs(model)
    inputs["rtc_prefix_normalized_actions"][:, :7] = 0.25
    inputs["rtc_prefix_mask"][:, :7] = True

    with pytest.raises(ValueError, match="does not preserve.*RTC prefix"):
        model(**inputs)
    value_inputs = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            "latent_chain",
            "denoise_indices",
            "flow_schedule_sha256",
            "clipped_normalized_actions",
            "denormalized_wire_actions",
        }
    }
    values = model.forward_values(**value_inputs)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()

    over_limit = dict(value_inputs)
    over_limit["rtc_prefix_mask"] = torch.zeros((2, 30), dtype=torch.bool)
    over_limit["rtc_prefix_mask"][:, :8] = True
    with pytest.raises(ValueError, match="attested checkpoint exclusive max_delay=8"):
        model.forward_values(**over_limit)


def test_inference_lease_copies_heads_and_shares_only_frozen_vlm() -> None:
    model = _model().eval()
    inputs = _inputs(model)
    expected = model(**inputs)
    lease = model.clone_for_inference_lease()
    actual = lease(**inputs)
    live_psi = model.psi_model
    leased_psi = lease.psi_model
    assert isinstance(live_psi, _FakePsi)
    assert isinstance(leased_psi, _FakePsi)

    assert lease is not model
    assert leased_psi is not live_psi
    assert leased_psi.vlm_model is live_psi.vlm_model
    assert leased_psi.action_header is not live_psi.action_header
    assert lease.critic is not model.critic
    assert not lease.training
    assert not lease.psi_model.training
    assert all(not parameter.requires_grad for parameter in lease.parameters())
    torch.testing.assert_close(actual["log_probs"], expected["log_probs"])
    torch.testing.assert_close(
        actual["element_log_probs"], expected["element_log_probs"]
    )
    torch.testing.assert_close(actual["values"], expected["values"])

    leased_scale = leased_psi.action_header.scale.detach().clone()
    with torch.no_grad():
        live_psi.action_header.scale.add_(1.0)
    torch.testing.assert_close(leased_psi.action_header.scale, leased_scale)

    next(live_psi.vlm_model.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="share only a frozen VLM"):
        model.clone_for_inference_lease()


def test_critic_stays_fp32_inside_outer_autocast() -> None:
    model = _model().eval()
    inputs = _inputs(model)
    projection_dtypes: list[torch.dtype] = []
    handle = model.critic.prefix_projection.register_forward_hook(
        lambda _module, _args, output: projection_dtypes.append(output.dtype)
    )
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            values = model.forward_values(
                **{
                    key: value
                    for key, value in inputs.items()
                    if key
                    not in {
                        "latent_chain",
                        "denoise_indices",
                        "flow_schedule_sha256",
                        "clipped_normalized_actions",
                        "denormalized_wire_actions",
                    }
                }
            )
    finally:
        handle.remove()

    assert values.dtype == torch.float32
    assert projection_dtypes == [torch.float32]
