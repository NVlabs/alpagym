"""Numerical and replay tests for VLA's Flow-SDE adapter."""

from __future__ import annotations

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from alpagym_g1_vla.flow import (
    VlaFlowSchedule,
    replay_flow_logprob,
    sample_flow_ode,
    sample_flow_sde,
)


def _schedule() -> VlaFlowSchedule:
    return VlaFlowSchedule(
        model_timesteps=torch.tensor(
            [1000.0, 889.0, 778.0, 667.0, 556.0, 445.0, 334.0, 223.0, 112.0, 1.0]
        ),
        sigmas=torch.tensor(
            [1.0, 0.889, 0.778, 0.667, 0.556, 0.445, 0.334, 0.223, 0.112, 0.001, 0.0]
        ),
    )


class _Velocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.125))

    def forward(self, latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 1:
            timestep = timestep[:, None, None]
        else:
            timestep = timestep[..., None]
        return latent * self.scale + timestep.to(latent.dtype) / 10_000.0


def test_flow_replay_identity_and_gradient() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(7)
    initial = torch.randn((2, 30, 38), generator=generator)
    final, trace = sample_flow_sde(
        model,
        initial,
        _schedule(),
        noise_level=0.4,
        denoise_indices=torch.tensor([2, 2]),
        generator=generator,
    )

    assert final.shape == (2, 30, 38)
    assert trace.chain.shape == (2, 11, 30, 38)
    replayed = replay_flow_logprob(
        model,
        trace,
        _schedule(),
        noise_level=0.4,
    )
    assert replayed.shape == (2, 30, 38)
    assert trace.old_element_logprobs.shape == (2, 30, 38)
    torch.testing.assert_close(replayed, trace.old_element_logprobs)
    joint_logprobs = replayed.flatten(start_dim=1).sum(dim=-1)
    assert joint_logprobs.shape == (2,)
    loss = -joint_logprobs.mean()
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    assert float(model.scale.grad.abs()) > 0.0


def test_deterministic_flow_uses_vla_nonuniform_euler_grid() -> None:
    schedule = _schedule()
    initial = torch.linspace(-1.0, 1.0, 30 * 38).reshape(1, 30, 38)

    def constant_velocity(latent: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.full_like(latent, 0.25)

    actual = sample_flow_ode(constant_velocity, initial, schedule)
    # Sum_i (sigma_i - sigma_{i+1}) == 1 even on VLA's .889/.001 grid.
    torch.testing.assert_close(actual, initial - 0.25, rtol=1.0e-6, atol=1.0e-6)


def test_native_ode_is_byte_exact_with_diffusers_scheduler() -> None:
    """Qualification Euler integration must remain checkpoint-scheduler exact."""
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(10, device="cpu")
    schedule = VlaFlowSchedule(
        model_timesteps=scheduler.timesteps,
        sigmas=scheduler.sigmas,
    )
    assert schedule.sha256 == (
        "d01fcb068a81712b78f9b72ff95877e332ae03245a2d1aba518c83354cf7ad12"
    )
    initial = torch.randn((2, 30, 38), generator=torch.Generator().manual_seed(37))

    def nonlinear_velocity(
        latent: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return latent.square() * 0.017 + timestep[:, None, None] / 10_000.0

    expected = initial.clone()
    for timestep in scheduler.timesteps:
        velocity = nonlinear_velocity(expected, timestep.expand(2))
        expected = scheduler.step(velocity, timestep, expected).prev_sample

    actual = sample_flow_ode(nonlinear_velocity, initial, schedule)
    assert torch.equal(actual, expected)


def test_zero_rtc_mask_replays_exactly() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(11)
    initial = torch.randn((1, 30, 38), generator=generator)
    prefix_actions = torch.randn((1, 30, 38), generator=generator)
    prefix_mask = torch.zeros((1, 30), dtype=torch.bool)
    final, trace = sample_flow_sde(
        model,
        initial,
        _schedule(),
        noise_level=0.4,
        denoise_indices=torch.tensor([4]),
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
        generator=generator,
    )

    assert final.shape == (1, 30, 38)
    replayed = replay_flow_logprob(
        model,
        trace,
        _schedule(),
        noise_level=0.4,
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
    )
    torch.testing.assert_close(replayed, trace.old_element_logprobs)


def test_nonempty_rtc_prefix_is_deterministic_and_zero_density() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(13)
    initial = torch.randn((1, 30, 38), generator=generator)
    prefix_actions = torch.randn((1, 30, 38), generator=generator)
    prefix_mask = torch.zeros((1, 30), dtype=torch.bool)
    prefix_mask[:, 0] = True

    final, trace = sample_flow_sde(
        model,
        initial,
        _schedule(),
        noise_level=0.4,
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
        generator=generator,
    )
    torch.testing.assert_close(final[:, 0], prefix_actions[:, 0])
    torch.testing.assert_close(
        trace.chain[:, :, 0],
        prefix_actions[:, None, 0].expand_as(trace.chain[:, :, 0]),
    )
    assert not bool(trace.old_element_logprobs[:, 0].any())
    replayed = replay_flow_logprob(
        model,
        trace,
        _schedule(),
        noise_level=0.4,
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
    )
    torch.testing.assert_close(replayed, trace.old_element_logprobs)
    ode = sample_flow_ode(
        model,
        initial,
        _schedule(),
        prefix_actions=prefix_actions,
        prefix_mask=prefix_mask,
    )
    torch.testing.assert_close(ode[:, 0], prefix_actions[:, 0])


def test_schedule_identity_fails_closed() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(19)
    _final, trace = sample_flow_sde(
        model,
        torch.randn((1, 30, 38), generator=generator),
        _schedule(),
        noise_level=0.4,
        denoise_indices=torch.tensor([1]),
        generator=generator,
    )
    changed = VlaFlowSchedule(
        model_timesteps=_schedule().model_timesteps,
        sigmas=torch.tensor(
            [1.0, 0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0.001, 0.0]
        ),
    )
    with pytest.raises(ValueError, match="schedule identity changed"):
        replay_flow_logprob(model, trace, changed, noise_level=0.4)


def test_flow_ppo_excludes_final_low_variance_transition() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(23)
    with pytest.raises(ValueError, match="exclude.*final"):
        sample_flow_sde(
            model,
            torch.randn((1, 30, 38), generator=generator),
            _schedule(),
            noise_level=0.4,
            denoise_indices=torch.tensor([9]),
            generator=generator,
        )


def test_flow_ppo_random_selection_honors_ignore_last() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(29)
    _final, trace = sample_flow_sde(
        model,
        torch.randn((128, 30, 38), generator=generator),
        _schedule(),
        noise_level=0.4,
        generator=generator,
    )

    assert int(trace.denoise_indices.min()) >= 0
    assert int(trace.denoise_indices.max()) < _schedule().num_steps - 1
    assert not bool((trace.denoise_indices == _schedule().num_steps - 1).any())
    assert trace.denoise_indices.unique().numel() == 1


def test_flow_sampling_rejects_per_item_denoise_indices() -> None:
    model = _Velocity()
    generator = torch.Generator().manual_seed(31)
    with pytest.raises(ValueError, match="one shared denoise index"):
        sample_flow_sde(
            model,
            torch.randn((2, 30, 38), generator=generator),
            _schedule(),
            noise_level=0.4,
            denoise_indices=torch.tensor([1, 2]),
            generator=generator,
        )
