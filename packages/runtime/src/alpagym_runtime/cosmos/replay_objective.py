# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay objective helpers for the AlpaGym Cosmos trainer."""

from __future__ import annotations

import torch


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped surrogate loss and importance ratio.

    The loss is averaged over valid (non-padding) rows, matching how
    ``compute_kl_penalty`` reduces, so the per-sample gradient scale and the
    policy-vs-KL balance do not depend on how many padding rows a shuffled
    minibatch happens to contain. An all-padding minibatch yields a
    graph-connected zero so every DP worker still backprops in lockstep.
    """
    # Bound the exponent input for numeric stability; PPO ratio clipping is below.
    log_ratio = (new_logprobs - old_logprobs).clamp(min=-5.0, max=5.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ratio_clip_low, 1.0 + ratio_clip_high) * advantages
    per_row = -torch.min(surr1, surr2)
    valid = (~is_padding).to(per_row.dtype)
    return (per_row * valid).sum() / valid.sum().clamp_min(1.0), ratio


def compute_token_ppo_surrogate(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio_clip_low: float,
    ratio_clip_high: float,
    is_padding: torch.Tensor,
    token_causality_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a clipped PPO objective independently for each action token.

    A planner macro-decision may contain a fixed sequence of stochastic action
    tokens.  Treating the sum of their log-probabilities as one importance ratio
    makes the ratio variance grow with sequence length and clips the whole block
    when only one token moved.  This objective broadcasts the macro advantage to
    every token, clips each token ratio independently, and averages over valid
    tokens. Row padding removes a whole macro-decision. An optional token mask
    removes a shadow tail generated after an early controller terminal, which
    cannot have affected the realized reward.
    """
    if new_token_logprobs.ndim != 2:
        raise ValueError(
            "token log-probabilities must have shape [B, H], got "
            f"{tuple(new_token_logprobs.shape)}"
        )
    if old_token_logprobs.shape != new_token_logprobs.shape:
        raise ValueError(
            "new token log-probabilities shape "
            f"{tuple(new_token_logprobs.shape)} != old token log-probabilities "
            f"shape {tuple(old_token_logprobs.shape)}"
        )
    batch_size = int(new_token_logprobs.shape[0])
    if tuple(advantages.shape) != (batch_size,):
        raise ValueError(
            f"advantages must have shape ({batch_size},), got {tuple(advantages.shape)}"
        )
    if tuple(is_padding.shape) != (batch_size,):
        raise ValueError(
            f"is_padding must have shape ({batch_size},), got {tuple(is_padding.shape)}"
        )
    if token_causality_mask is not None:
        if tuple(token_causality_mask.shape) != tuple(new_token_logprobs.shape):
            raise ValueError(
                "token_causality_mask must match token log-probabilities shape "
                f"{tuple(new_token_logprobs.shape)}, got "
                f"{tuple(token_causality_mask.shape)}"
            )
        if token_causality_mask.dtype is not torch.bool:
            raise TypeError("token_causality_mask must have dtype bool")
    if not torch.isfinite(new_token_logprobs).all():
        raise FloatingPointError("model returned non-finite token log-probabilities")
    if not torch.isfinite(old_token_logprobs).all():
        raise FloatingPointError(
            "replay contains non-finite old token log-probabilities"
        )

    log_ratio = (new_token_logprobs - old_token_logprobs).clamp(min=-5.0, max=5.0)
    ratio = torch.exp(log_ratio)
    token_advantages = advantages.unsqueeze(-1)
    surr1 = ratio * token_advantages
    surr2 = (
        torch.clamp(ratio, 1.0 - ratio_clip_low, 1.0 + ratio_clip_high)
        * token_advantages
    )
    per_token = -torch.min(surr1, surr2)
    valid_mask = (~is_padding).unsqueeze(-1).expand_as(per_token)
    if token_causality_mask is not None:
        valid_mask = valid_mask & token_causality_mask
    valid = valid_mask.to(per_token.dtype)
    return (per_token * valid).sum() / valid.sum().clamp_min(1.0), ratio


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
