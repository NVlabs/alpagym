# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay objective helpers for the AlpaGym Cosmos trainer."""

from __future__ import annotations

import torch

from alpagym_runtime.third_party.rlinf.ppo import (
    compute_ppo_actor_loss,
    compute_ppo_value_loss,
)


def assert_replay_shapes(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    kl_div: torch.Tensor | None,
    values: torch.Tensor | None = None,
    returns: torch.Tensor | None = None,
    old_values: torch.Tensor | None = None,
) -> None:
    """Raise if model outputs and trainer signals disagree on row count."""
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"new log_probs shape {tuple(new_logprobs.shape)} != old_logprobs "
            f"shape {tuple(old_logprobs.shape)}"
        )
    if advantages.shape != old_logprobs.shape:
        raise ValueError(
            f"advantages shape {tuple(advantages.shape)} != old_logprobs "
            f"shape {tuple(old_logprobs.shape)}"
        )
    if kl_div is not None and kl_div.shape != old_logprobs.shape:
        raise ValueError(
            f"kl_div shape {tuple(kl_div.shape)} != old_logprobs shape {tuple(old_logprobs.shape)}"
        )
    for name, tensor in (
        ("values", values),
        ("returns", returns),
        ("old_values", old_values),
    ):
        if tensor is not None and tensor.shape != old_logprobs.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} != old_logprobs "
                f"shape {tuple(old_logprobs.shape)}"
            )
    # Every forwarded row (padding included) must score finite; padding rows clone
    # a valid step's inputs, so non-finite values here signal a real bug.
    if not torch.isfinite(new_logprobs).all():
        raise FloatingPointError("model returned non-finite log_probs")
    if not torch.isfinite(old_logprobs).all():
        raise FloatingPointError("rollout payload contains non-finite old_logprobs")
    if kl_div is not None and not torch.isfinite(kl_div).all():
        raise FloatingPointError("model returned non-finite kl_div")
    for name, tensor in (
        ("values", values),
        ("returns", returns),
        ("old_values", old_values),
    ):
        if tensor is not None and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"PPO replay contains non-finite {name}")


def compute_ppo_surrogate(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio_clip_low: float,
    ratio_clip_high: float,
    is_padding: torch.Tensor,
    dual_clip_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped surrogate loss and importance ratio.

    The loss is averaged over valid (non-padding) rows, matching how
    ``compute_kl_penalty`` reduces, so the per-sample gradient scale and the
    policy-vs-KL balance do not depend on how many padding rows a shuffled
    minibatch happens to contain. An all-padding minibatch yields a
    graph-connected zero so every DP worker still backprops in lockstep.
    """
    return compute_ppo_actor_loss(
        new_logprobs,
        old_logprobs,
        ratio_clip_low,
        ratio_clip_high,
        advantages,
        loss_mask=~is_padding,
        clip_log_ratio_min=-5.0,
        clip_log_ratio_max=5.0,
        dual_clip_ratio=dual_clip_ratio,
    )


def compute_flow_ppo_surrogate(
    new_element_logprobs: torch.Tensor,
    old_element_logprobs: torch.Tensor,
    new_joint_logprobs: torch.Tensor,
    old_joint_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ratio_clip_low: float,
    ratio_clip_high: float,
    dual_clip_ratio: float,
    is_padding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PPO to the full selected Flow-SDE transition density."""
    if new_element_logprobs.ndim != 2 or new_element_logprobs.shape[1] == 0:
        raise ValueError("Flow-PPO element log-probabilities must have shape [B, D]")
    if old_element_logprobs.shape != new_element_logprobs.shape:
        raise ValueError("Flow-PPO new and old element log-probabilities must match")
    if not torch.allclose(
        new_element_logprobs.sum(dim=-1),
        new_joint_logprobs,
        rtol=1.0e-5,
        atol=1.0e-5,
    ):
        raise ValueError("Flow-PPO new joint log-probability differs from its elements")
    if not torch.allclose(
        old_element_logprobs.sum(dim=-1),
        old_joint_logprobs,
        rtol=1.0e-5,
        atol=1.0e-5,
    ):
        raise ValueError("Flow-PPO old joint log-probability differs from its elements")
    return compute_ppo_actor_loss(
        new_joint_logprobs,
        old_joint_logprobs,
        ratio_clip_low,
        ratio_clip_high,
        advantages,
        loss_mask=~is_padding,
        clip_log_ratio_min=-5.0,
        clip_log_ratio_max=5.0,
        dual_clip_ratio=dual_clip_ratio,
    )


def compute_kl_penalty(
    kl_div: torch.Tensor | None,
    is_padding: torch.Tensor,
    kl_beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute KL penalty over valid rows, returning zero when KL is disabled."""
    if kl_div is None or kl_beta <= 0.0:
        return torch.tensor(0.0, device=device)
    valid_kl = kl_div[~is_padding]
    if valid_kl.numel() == 0:
        return torch.tensor(0.0, device=device)
    return valid_kl.mean() * kl_beta


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    is_padding: torch.Tensor,
    old_values: torch.Tensor | None = None,
    value_clip_range: float | None = None,
    huber_delta: float | None = None,
) -> torch.Tensor:
    """Compute PPO value-function loss over valid rows.

    The unclipped path is ``0.5 * (V(s) - R)^2``. When ``old_values`` and a
    positive clip range are supplied, this uses the standard PPO clipped value
    loss and takes the max of clipped vs. unclipped squared error per row.
    Padding rows are masked and an all-padding minibatch returns a graph-connected
    zero so distributed workers stay in lockstep.
    """
    if value_clip_range is not None and value_clip_range <= 0.0:
        raise ValueError(
            f"value_clip_range must be positive when set, got {value_clip_range}"
        )

    if huber_delta is not None:
        if old_values is None or value_clip_range is None:
            raise ValueError(
                "PPO Huber value loss requires old_values and value_clip_range"
            )
        return compute_ppo_value_loss(
            values,
            returns,
            old_values,
            value_clip=value_clip_range,
            huber_delta=huber_delta,
            loss_mask=~is_padding,
        )

    value_error = values - returns
    value_losses = value_error.square()
    if old_values is not None and value_clip_range is not None:
        clipped_values = old_values + (values - old_values).clamp(
            min=-value_clip_range,
            max=value_clip_range,
        )
        clipped_losses = (clipped_values - returns).square()
        value_losses = torch.maximum(value_losses, clipped_losses)

    valid = (~is_padding).to(values.dtype)
    return 0.5 * (value_losses * valid).sum() / valid.sum().clamp_min(1.0)
