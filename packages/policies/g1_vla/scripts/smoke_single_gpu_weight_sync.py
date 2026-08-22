#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an opt-in real-checkpoint VLA PPO and weight-sync smoke on one GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from alpagym_host.config import SamplingParamsConfig

from alpagym_g1_vla.cosmos_model import load_vla_rollout_model
from alpagym_g1_vla.inference_model import VlaNativeInferenceModel
from alpagym_g1_vla.model import VlaPolicySample, VlaPsiActorCritic
from alpagym_runtime.cosmos.replay_objective import (
    compute_flow_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.inference.inference_engine import InferenceEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two attested VLA models, take one real PPO optimizer step, "
            "sync the trainable overlay, and verify versioned lease isolation."
        )
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Pinned VLA model directory under alpa_policy_eval/models.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default="walk ahead.")
    return parser.parse_args()


def _build_model_inputs(
    core: VlaPsiActorCritic,
    *,
    instruction: str,
    device: torch.device,
) -> dict[str, Any]:
    """Build one deterministic real-processor observation for the smoke."""
    image = Image.new("RGB", (224, 224), color=(32, 48, 64))
    try:
        psi = cast(Any, core.psi_model)
        built = psi._build_vlm_batch([[image]], [instruction])
    finally:
        image.close()
    if not isinstance(built, tuple) or len(built) != 6:
        raise TypeError("Psi _build_vlm_batch returned an unexpected contract")
    converted = tuple(
        torch.as_tensor(item) if item is not None else None for item in built
    )
    input_ids, attention_mask, pixel_values, grid, effective_grid, pool_factors = (
        converted
    )
    if any(item is None for item in (input_ids, attention_mask, pixel_values, grid)):
        raise TypeError("Psi processor omitted a required tensor")
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert isinstance(pixel_values, torch.Tensor)
    assert isinstance(grid, torch.Tensor)
    image_patch_counts = grid.to(dtype=torch.int64).prod(dim=1)
    image_count = int(grid.shape[0])
    patch_count = int(image_patch_counts.sum().item())
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
        "sequence_lengths": attention_mask.to(dtype=torch.int64).sum(dim=1),
        "image_counts": torch.tensor(
            [image_count], device=grid.device, dtype=torch.int64
        ),
        "image_offsets": torch.tensor(
            [0, image_count], device=grid.device, dtype=torch.int64
        ),
        "image_patch_counts": image_patch_counts,
        "patch_counts": torch.tensor(
            [patch_count], device=grid.device, dtype=torch.int64
        ),
        "patch_offsets": torch.tensor(
            [0, patch_count], device=grid.device, dtype=torch.int64
        ),
        "physical_states": torch.zeros((1, 1, 29), device=device, dtype=torch.float32),
        "rtc_prefix_normalized_actions": torch.zeros(
            (1, 30, 38), device=device, dtype=torch.float32
        ),
        "rtc_prefix_mask": torch.zeros((1, 30), device=device, dtype=torch.bool),
        "effective_image_grid_thw": effective_grid,
        "visual_pool_factors": pool_factors,
    }


def _sample(
    core: VlaPsiActorCritic,
    inputs: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> VlaPolicySample:
    """Sample one trace using the real bf16 action path."""
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return core.sample_actions(**inputs, generator=generator)


def _replay(
    core: VlaPsiActorCritic,
    inputs: dict[str, Any],
    sample: VlaPolicySample,
) -> dict[str, torch.Tensor | None]:
    """Replay one frozen Flow-SDE trace through the supplied model version."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return core(
            **inputs,
            latent_chain=sample.trace.chain,
            denoise_indices=sample.trace.denoise_indices,
            flow_schedule_sha256=sample.trace.schedule_sha256,
            clipped_normalized_actions=sample.clipped_normalized,
            denormalized_wire_actions=sample.denormalized_wire,
        )


def _max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def main() -> None:
    """Execute the real-model single-GPU synchronization assertions."""
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("the VLA weight-sync smoke requires one CUDA GPU")
    model_root = args.model_root.expanduser().resolve(strict=True)

    torch.manual_seed(20260819)
    torch.cuda.manual_seed_all(20260819)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    trainer = load_vla_rollout_model(model_root, device, torch.bfloat16)
    rollout = load_vla_rollout_model(model_root, device, torch.bfloat16)
    trainer_params = dict(trainer.named_parameters())
    rollout_params = dict(rollout.named_parameters())
    if trainer_params.keys() != rollout_params.keys():
        raise RuntimeError("trainer and rollout parameter maps differ")
    trainable_names = tuple(
        name for name, parameter in trainer_params.items() if parameter.requires_grad
    )
    if not trainable_names:
        raise RuntimeError("VLA model exposes no trainable parameters")
    with torch.no_grad():
        for name in trainable_names:
            destination = rollout_params[name]
            source = trainer_params[name]
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise RuntimeError(f"weight-sync tensor contract differs for {name}")
            destination.copy_(source)

    engine = InferenceEngine(
        VlaNativeInferenceModel(rollout),
        # Native humanoid callbacks do not consume this AV-only configuration.
        sampling=SamplingParamsConfig(
            top_p=1.0,
            top_k=None,
            temperature=1.0,
            num_traj_samples=1,
            num_traj_sets=1,
        ),
        return_trace_for_rl=True,
        max_batch_size=1,
        require_session_model_leases=True,
    )
    lease0 = engine.create_model_lease(behavior_policy_version=0)
    engine.register_session_model_lease("weight-sync-smoke-v0", lease0)
    core0 = lease0.model
    if not isinstance(core0, VlaPsiActorCritic):
        raise TypeError("VLA lease did not return its actor-critic core")
    inputs = _build_model_inputs(core0, instruction=args.instruction, device=device)
    sample0 = _sample(core0, inputs, seed=314159, device=device)
    trainer_before = _replay(trainer.actor_critic, inputs, sample0)
    old_before = _replay(core0, inputs, sample0)
    trainer_elements_before = trainer_before["element_log_probs"]
    trainer_joint_before = trainer_before["log_probs"]
    trainer_values_before = trainer_before["values"]
    old_elements_before = old_before["element_log_probs"]
    old_joint_before = old_before["log_probs"]
    if any(
        item is None
        for item in (
            trainer_elements_before,
            trainer_joint_before,
            trainer_values_before,
            old_elements_before,
            old_joint_before,
        )
    ):
        raise RuntimeError("VLA replay omitted PPO outputs")
    assert isinstance(trainer_elements_before, torch.Tensor)
    assert isinstance(trainer_joint_before, torch.Tensor)
    assert isinstance(trainer_values_before, torch.Tensor)
    assert isinstance(old_elements_before, torch.Tensor)
    assert isinstance(old_joint_before, torch.Tensor)
    if _max_delta(trainer_elements_before, sample0.old_element_logprobs) != 0.0:
        raise RuntimeError("pre-update trainer replay differs from behavior policy")
    if _max_delta(old_elements_before, sample0.old_element_logprobs) != 0.0:
        raise RuntimeError("version-0 lease does not replay its own trace exactly")
    if _max_delta(trainer_joint_before, sample0.old_log_probs) != 0.0:
        raise RuntimeError("pre-update trainer joint density differs from behavior")
    if _max_delta(old_joint_before, sample0.old_log_probs) != 0.0:
        raise RuntimeError("version-0 lease joint density differs from behavior")

    action_parameters = tuple(trainer.actor_critic.psi_model.action_header.parameters())
    critic_parameters = tuple(trainer.actor_critic.critic.parameters())
    action_parameter_ids = {id(parameter) for parameter in action_parameters}
    critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
    trainable_parameter_ids = {id(trainer_params[name]) for name in trainable_names}
    if action_parameter_ids & critic_parameter_ids:
        raise RuntimeError("VLA optimizer parameter groups overlap")
    if action_parameter_ids | critic_parameter_ids != trainable_parameter_ids:
        raise RuntimeError("VLA optimizer groups do not cover trainable parameters")
    action_optimizer = torch.optim.AdamW(
        action_parameters,
        lr=5.0e-6,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    critic_optimizer = torch.optim.AdamW(
        critic_parameters,
        lr=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    action_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    padding = torch.zeros((1,), device=device, dtype=torch.bool)
    policy_loss, ratio_before = compute_flow_ppo_surrogate(
        new_element_logprobs=trainer_elements_before,
        old_element_logprobs=sample0.old_element_logprobs,
        new_joint_logprobs=trainer_joint_before,
        old_joint_logprobs=sample0.old_log_probs,
        advantages=torch.ones((1,), device=device, dtype=torch.float32),
        ratio_clip_low=0.2,
        ratio_clip_high=0.28,
        dual_clip_ratio=3.0,
        is_padding=padding,
    )
    value_loss = compute_value_loss(
        trainer_values_before,
        sample0.values + 1.0,
        padding,
        old_values=sample0.values,
        value_clip_range=0.2,
        huber_delta=10.0,
    )
    loss = policy_loss + value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [trainer_params[name] for name in trainable_names], max_norm=1.0
    )
    finite_nonzero_gradients = 0
    for name in trainable_names:
        gradient = trainer_params[name].grad
        if (
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and float(gradient.detach().abs().max().cpu()) > 0.0
        ):
            finite_nonzero_gradients += 1
    if finite_nonzero_gradients == 0:
        raise RuntimeError("PPO update produced no finite nonzero trainable gradient")
    action_optimizer.step()
    critic_optimizer.step()

    trainer_after = _replay(trainer.actor_critic, inputs, sample0)
    trainer_elements_after = trainer_after["element_log_probs"]
    trainer_values_after = trainer_after["values"]
    if trainer_elements_after is None or trainer_values_after is None:
        raise RuntimeError("post-update VLA replay omitted PPO outputs")
    element_update_delta = _max_delta(
        trainer_elements_after, sample0.old_element_logprobs
    )
    value_update_delta = _max_delta(trainer_values_after, sample0.values)
    if element_update_delta == 0.0 or value_update_delta == 0.0:
        raise RuntimeError(
            "optimizer step did not change both policy and value outputs"
        )

    with torch.no_grad():
        for name in trainable_names:
            rollout_params[name].copy_(trainer_params[name])
    lease1 = engine.create_model_lease(behavior_policy_version=1)
    core1 = lease1.model
    if not isinstance(core1, VlaPsiActorCritic):
        raise TypeError("VLA lease did not return its actor-critic core")
    if engine.get_model_for_session("weight-sync-smoke-v0") is not core0:
        raise RuntimeError("version-0 session lease changed during weight sync")

    old_after = _replay(core0, inputs, sample0)
    new_after = _replay(core1, inputs, sample0)
    old_elements_after = old_after["element_log_probs"]
    new_elements_after = new_after["element_log_probs"]
    new_values_after = new_after["values"]
    if any(
        item is None
        for item in (old_elements_after, new_elements_after, new_values_after)
    ):
        raise RuntimeError("lease replay omitted PPO outputs")
    assert isinstance(old_elements_after, torch.Tensor)
    assert isinstance(new_elements_after, torch.Tensor)
    assert isinstance(new_values_after, torch.Tensor)
    old_immutability_delta = _max_delta(
        old_elements_after, sample0.old_element_logprobs
    )
    new_element_sync_delta = _max_delta(new_elements_after, trainer_elements_after)
    new_value_sync_delta = _max_delta(new_values_after, trainer_values_after)
    if old_immutability_delta != 0.0:
        raise RuntimeError("open version-0 lease changed during weight sync")
    if new_element_sync_delta != 0.0 or new_value_sync_delta != 0.0:
        raise RuntimeError("version-1 lease differs from the updated trainer")
    psi0 = cast(Any, core0.psi_model)
    psi1 = cast(Any, core1.psi_model)
    live_psi = cast(Any, rollout.actor_critic.psi_model)
    if psi0.vlm_model is not live_psi.vlm_model:
        raise RuntimeError("version-0 lease failed to share the frozen VLM")
    if psi1.vlm_model is not live_psi.vlm_model:
        raise RuntimeError("version-1 lease failed to share the frozen VLM")
    if psi0.action_header is live_psi.action_header:
        raise RuntimeError("version-0 lease shares its mutable action head")
    if psi1.action_header is live_psi.action_header:
        raise RuntimeError("version-1 lease shares its mutable action head")

    sample1 = _sample(core1, inputs, seed=271828, device=device)
    replay1 = _replay(core1, inputs, sample1)
    replay1_elements = replay1["element_log_probs"]
    if replay1_elements is None:
        raise RuntimeError("version-1 replay omitted element log-probabilities")
    new_sample_replay_delta = _max_delta(replay1_elements, sample1.old_element_logprobs)
    if new_sample_replay_delta != 0.0:
        raise RuntimeError("version-1 lease does not replay its own sample exactly")

    torch.cuda.synchronize(device)
    valid_ratios = ratio_before[~padding]
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(device),
        "model_root": str(model_root),
        "versions": [lease0.behavior_policy_version, lease1.behavior_policy_version],
        "trainable_parameter_tensors": len(trainable_names),
        "finite_nonzero_gradient_tensors": finite_nonzero_gradients,
        "action_learning_rate": 5.0e-6,
        "critic_learning_rate": 1.0e-4,
        "valid_ratio_before_min": float(valid_ratios.detach().min().cpu()),
        "valid_ratio_before_max": float(valid_ratios.detach().max().cpu()),
        "post_update_element_logprob_max_delta": element_update_delta,
        "post_update_value_max_delta": value_update_delta,
        "old_lease_immutability_max_delta": old_immutability_delta,
        "new_lease_vs_trainer_element_max_delta": new_element_sync_delta,
        "new_lease_vs_trainer_value_max_delta": new_value_sync_delta,
        "new_lease_sample_replay_max_delta": new_sample_replay_delta,
        "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    engine.release_session_model_lease("weight-sync-smoke-v0")


if __name__ == "__main__":
    main()
