# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for `InferenceEngine`."""

import threading

import pytest
import torch
from alpagym_host.config import SamplingParamsConfig
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.inference.types import (
    NUM_ROUTE_WAYPOINTS,
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
)


class _FakeInferenceModel:
    """Minimal in-test inference model echoing per-input tags."""

    def __init__(
        self,
        output_batch_size: int | None = None,
    ) -> None:
        """Configure optional output batch size and zero-init the call log."""
        self.calls: list[BatchedModelInput] = []
        self.output_batch_size = output_batch_size
        self.model = torch.nn.Linear(1, 1)

    def get_model(self) -> torch.nn.Module:
        """Return the current rollout-serving module."""
        return self.model

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the rollout-serving module."""
        self.model = model

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Record the dispatch and emit a deterministic batched output."""
        self.calls.append(model_input)
        tags = model_input.ego_history_xyz[:, 0, 0, 0]
        batch_size = (
            self.output_batch_size
            if self.output_batch_size is not None
            else tags.shape[0]
        )
        pred_xyz = torch.zeros(
            (batch_size, sampling.num_traj_sets, sampling.num_traj_samples, 1, 3),
            dtype=tags.dtype,
        )
        pred_xyz[..., 0, 0] = tags[:batch_size, None, None]
        pred_rot = (
            torch.eye(3, dtype=tags.dtype)
            .expand(
                batch_size, sampling.num_traj_sets, sampling.num_traj_samples, 1, 3, 3
            )
            .clone()
        )
        return BatchedModelOutput(pred_xyz=pred_xyz, pred_rot=pred_rot)


def _sampling() -> SamplingParamsConfig:
    """Sampling params used by every test; values are arbitrary but fixed."""
    return SamplingParamsConfig(
        top_p=0.9,
        top_k=None,
        temperature=0.7,
        num_traj_samples=1,
        num_traj_sets=1,
        max_generation_length=None,
    )


def _route_xy(value: float) -> torch.Tensor:
    """Build one fixed-shape route tensor."""
    return torch.full((NUM_ROUTE_WAYPOINTS, 2), value, dtype=torch.float32)


def _model_input(tag: int, seed: int | None = None) -> ModelInput:
    """Build a one-element `ModelInput` whose tag rides on `ego_history_xyz`."""
    return _model_input_with_route(tag, _route_xy(float(tag)), seed=seed)


def _model_input_with_route(
    tag: int,
    route_xy: torch.Tensor,
    seed: int | None = None,
) -> ModelInput:
    """Build a one-element `ModelInput` whose tag rides on `ego_history_xyz`."""
    ego_history_xyz = torch.tensor([[[float(tag), 0.0, 0.0]]], dtype=torch.float32)
    ego_history_rot = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
    seed_tensor = torch.tensor(seed, dtype=torch.int64) if seed is not None else None
    return ModelInput(
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        camera_frames=torch.zeros((0, 3, 1, 1), dtype=torch.uint8),
        camera_indices=torch.zeros((0,), dtype=torch.int64),
        relative_timestamps=torch.zeros((0,), dtype=torch.int64),
        route_xy=route_xy,
        seed=seed_tensor,
    )


def test_engine_resolves_outputs_in_queue_order() -> None:
    """A mixed-size batch of three inputs resolves to per-input outputs in order."""
    inference_model = _FakeInferenceModel()
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=_sampling(),
        return_trace_for_rl=False,
        max_batch_size=8,
    )
    futures = [inference_engine.infer(_model_input(i)) for i in range(3)]
    thread = threading.Thread(target=inference_engine.run_loop, daemon=True)
    thread.start()
    try:
        results = [
            int(f.result(timeout=2.0).pred_xyz[0, 0, 0, 0].item()) for f in futures
        ]
    finally:
        inference_engine.shutdown()
        thread.join(timeout=2.0)
    assert results == [0, 1, 2]
    assert len(inference_model.calls) == 1
    assert inference_model.calls[0].ego_history_xyz.shape[0] == 3


def test_engine_stacks_seeds_in_queue_order() -> None:
    """Queued requests carry per-session seeds through typed batching."""
    inference_model = _FakeInferenceModel()
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=_sampling(),
        return_trace_for_rl=False,
        max_batch_size=8,
    )
    futures = [_model_input(i, seed=100 + i) for i in range(3)]
    futures = [inference_engine.infer(model_input) for model_input in futures]
    thread = threading.Thread(target=inference_engine.run_loop, daemon=True)
    thread.start()
    try:
        [future.result(timeout=2.0) for future in futures]
    finally:
        inference_engine.shutdown()
        thread.join(timeout=2.0)
    assert len(inference_model.calls) == 1
    assert inference_model.calls[0].seed is not None
    assert inference_model.calls[0].seed.tolist() == [100, 101, 102]


def test_batched_model_input_stack_rejects_partial_seeds() -> None:
    """Seeded inference batches must be all seeded or all unseeded."""
    with pytest.raises(TypeError):
        BatchedModelInput.stack([_model_input(0, seed=100), _model_input(1)])


def test_engine_forwards_model_hooks() -> None:
    """Inference engine forwards Cosmos weight-sync hooks to the model adapter."""
    inference_model = _FakeInferenceModel()
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=_sampling(),
        return_trace_for_rl=False,
        max_batch_size=8,
    )
    synced_model = torch.nn.Linear(1, 1)

    assert inference_engine.get_model() is inference_model.model

    inference_engine.set_model(synced_model)

    assert inference_engine.get_model() is synced_model
    assert inference_model.model is synced_model
    assert inference_engine.get_model_for_session("colocated-session") is synced_model


def test_session_model_lease_survives_inplace_and_replacement_weight_sync() -> None:
    """A disaggregated episode never observes a mid-session live-model update."""

    inference_model = _FakeInferenceModel()
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=_sampling(),
        return_trace_for_rl=False,
        max_batch_size=8,
        require_session_model_leases=True,
    )
    with torch.no_grad():
        inference_model.model.weight.fill_(1.0)
        inference_model.model.bias.fill_(1.0)

    lease = inference_engine.create_model_lease(behavior_policy_version=7)
    inference_engine.register_session_model_lease("session-7", lease)
    leased_model = inference_engine.get_model_for_session("session-7")

    assert leased_model is lease.model
    assert not leased_model.training
    assert all(not parameter.requires_grad for parameter in leased_model.parameters())

    # Distributed R2R updates the live parameter tensors in place.
    with torch.no_grad():
        inference_engine.get_model().weight.fill_(2.0)
        inference_engine.get_model().bias.fill_(2.0)
    torch.testing.assert_close(
        leased_model.weight, torch.ones_like(leased_model.weight)
    )

    # Colocated P2R may replace the live model object instead.
    replacement = torch.nn.Linear(1, 1)
    with torch.no_grad():
        replacement.weight.fill_(3.0)
        replacement.bias.fill_(3.0)
    inference_engine.set_model(replacement)

    assert inference_engine.get_model() is replacement
    assert inference_engine.get_model_for_session("session-7") is leased_model
    torch.testing.assert_close(
        leased_model.weight, torch.ones_like(leased_model.weight)
    )

    inference_engine.release_session_model_lease("session-7")
    with pytest.raises(RuntimeError, match="no immutable model lease"):
        inference_engine.get_model_for_session("session-7")


def test_engine_batches_fixed_route_inputs() -> None:
    """Fixed-shape route tensors batch without route-specific padding."""
    inference_model = _FakeInferenceModel()
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=_sampling(),
        return_trace_for_rl=False,
        max_batch_size=8,
    )
    futures = [
        inference_engine.infer(_model_input_with_route(0, _route_xy(1.0))),
        inference_engine.infer(_model_input_with_route(1, _route_xy(2.0))),
        inference_engine.infer(_model_input_with_route(2, _route_xy(3.0))),
    ]
    thread = threading.Thread(target=inference_engine.run_loop, daemon=True)
    thread.start()
    try:
        results = [
            int(f.result(timeout=2.0).pred_xyz[0, 0, 0, 0].item()) for f in futures
        ]
    finally:
        inference_engine.shutdown()
        thread.join(timeout=2.0)

    assert results == [0, 1, 2]
    assert len(inference_model.calls) == 1
    route_xy = inference_model.calls[0].route_xy
    assert route_xy.shape == (3, NUM_ROUTE_WAYPOINTS, 2)
    torch.testing.assert_close(route_xy[0], _route_xy(1.0))
    torch.testing.assert_close(route_xy[1], _route_xy(2.0))
    torch.testing.assert_close(route_xy[2], _route_xy(3.0))
    assert inference_model.calls[0].ego_history_xyz[:, 0, 0, 0].tolist() == [
        0.0,
        1.0,
        2.0,
    ]
