# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the alpagym inference-engine factory."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from alpagym_host.config import (
    AlpamayoPolicyConfig,
    DiffusionSamplingConfig,
    InferenceConfig,
    ModelConfig,
    SamplingParamsConfig,
    TrajectorySelectorKind,
)
from alpagym_plugins.plugins import PluginNotFoundError
from alpagym_runtime.inference.types import BatchedModelInput, BatchedModelOutput
from alpagym_runtime.policies import factory
from alpagym_runtime.policies.registry import PolicyBundle


def _make_resolved_config(max_batch_size: int = 1) -> SimpleNamespace:
    """Build the minimum ``RunConfig`` slice ``build_inference_engine`` reads."""
    model_cfg = ModelConfig(
        kind="alpamayo_r1",
        path="/nonexistent/path/loader/is/stubbed",
        device="cpu",
        dtype="bfloat16",
        use_cameras=["camera_front_wide_120fov"],
        num_context_frames=4,
        num_historical_waypoints=16,
        num_future_waypoints=64,
        step_dt_us=100000,
        input_size=[320, 512],
    )
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=1,
        num_traj_sets=1,
        diffusion_kwargs=DiffusionSamplingConfig(int_method="sde", noise_level=0.2),
    )
    inference_cfg = InferenceConfig(
        max_batch_size=max_batch_size,
        sampling=sampling,
        return_trace_for_rl=True,
    )
    policy = AlpamayoPolicyConfig(
        kind="alpamayo",
        model=model_cfg,
        inference=inference_cfg,
        trajectory_selector=TrajectorySelectorKind.identity,
    )
    return SimpleNamespace(policy=policy)


class _FakeInferenceModel:
    """Stand-in returned by the stubbed bundle."""

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Match the InferenceModel protocol; never invoked by factory."""
        raise NotImplementedError


def _make_fake_bundle(
    *,
    load_inference_model: Any | None = None,
    wrap_inference_model: Any | None = None,
) -> PolicyBundle:
    """Build a real ``PolicyBundle`` with no-op hooks plus optional overrides."""
    return PolicyBundle(
        setup_tokenizer=lambda config: None,
        build_data_packer=lambda run_config, cosmos_role: None,
        install_runtime_bridge=lambda: None,
        load_inference_model=load_inference_model
        or (lambda run_config, device, dtype: None),
        build_model_inputs=lambda run_config: None,
        wrap_inference_model=wrap_inference_model,
    )


def test_build_inference_engine_wires_bundle_model_to_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the bundle's loaded inference model is wired into the engine."""
    captured: dict[str, Any] = {}
    fake_model = _FakeInferenceModel()

    def fake_load(run_config: Any, device: Any, dtype: Any) -> Any:
        captured["run_config"] = run_config
        captured["device"] = device
        captured["dtype"] = dtype
        return fake_model

    config = _make_resolved_config()
    monkeypatch.setattr(
        factory,
        "get_policy_bundle",
        lambda kind: _make_fake_bundle(load_inference_model=fake_load),
    )

    engine = factory.build_inference_engine(config)

    assert engine._inference_model is fake_model
    assert captured["run_config"] is config
    assert captured["dtype"] is torch.bfloat16
    assert captured["device"] == torch.device("cpu")


def test_build_inference_engine_wraps_existing_colocated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A colocated policy model is wrapped without loading a second checkpoint."""
    existing_model = torch.nn.Linear(2, 2)
    fake_model = _FakeInferenceModel()
    loaded = False

    def fake_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        loaded = True
        return fake_model

    def fake_wrap(model: torch.nn.Module) -> Any:
        assert model is existing_model
        return fake_model

    monkeypatch.setattr(
        factory,
        "get_policy_bundle",
        lambda kind: _make_fake_bundle(
            load_inference_model=fake_load,
            wrap_inference_model=fake_wrap,
        ),
    )

    engine = factory.build_inference_engine(
        _make_resolved_config(), existing_model=existing_model
    )

    assert engine._inference_model is fake_model
    assert not loaded


def test_build_inference_engine_accepts_batch_size_greater_than_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_batch_size`` passes through to the engine with no guard anywhere."""
    fake_model = _FakeInferenceModel()
    monkeypatch.setattr(
        factory,
        "get_policy_bundle",
        lambda kind: _make_fake_bundle(load_inference_model=lambda *a, **k: fake_model),
    )

    config = _make_resolved_config(max_batch_size=2)
    engine = factory.build_inference_engine(config)

    assert engine._inference_model is fake_model
    assert engine._max_batch_size == 2


def test_disaggregated_humanoid_engine_requires_session_model_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only separate humanoid rollout processes enable immutable snapshots."""

    fake_model = _FakeInferenceModel()
    monkeypatch.setattr(
        factory,
        "get_policy_bundle",
        lambda kind: _make_fake_bundle(
            load_inference_model=lambda *args, **kwargs: fake_model
        ),
    )
    config = _make_resolved_config()
    config.alpasim = SimpleNamespace(simulation_domain="humanoid")
    config.cosmos = SimpleNamespace(mode="disaggregated")

    engine = factory.build_inference_engine(config)

    assert engine.requires_session_model_leases


def test_build_inference_engine_unknown_kind_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninstalled policy kind surfaces the registry's ``PluginNotFoundError``."""

    def raise_not_found(kind: str) -> Any:
        raise PluginNotFoundError(f"no bundle for {kind!r}")

    monkeypatch.setattr(factory, "get_policy_bundle", raise_not_found)

    config = _make_resolved_config()
    with pytest.raises(PluginNotFoundError):
        factory.build_inference_engine(config)


def test_visual_humanoid_policy_explicitly_requires_strict_camera_abi() -> None:
    config = _make_resolved_config()
    config.policy.model.bundle_config["require_policy_camera"] = True

    assert factory.humanoid_policy_camera_required(config)


def test_humanoid_policy_camera_requirement_rejects_string_boolean() -> None:
    config = _make_resolved_config()
    config.policy.model.bundle_config["require_policy_camera"] = "true"

    with pytest.raises(TypeError, match="must be a boolean"):
        factory.humanoid_policy_camera_required(config)


def test_humanoid_policy_factory_requires_explicit_builder() -> None:
    """A missing factory must not silently dispatch a zero-action policy."""
    config = _make_resolved_config()

    with pytest.raises(KeyError, match="humanoid_policy_factory"):
        factory.build_humanoid_policy_factory(config, object())


def test_humanoid_policy_factory_rejects_removed_zero_alias() -> None:
    """The former smoke alias cannot enter the production config route."""
    config = _make_resolved_config()
    config.policy.model.bundle_config["humanoid_policy_factory"] = "zero"

    with pytest.raises(ValueError, match="module:attribute"):
        factory.build_humanoid_policy_factory(config, object())
