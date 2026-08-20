# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal RLinf PPO actor loss without RLinf runtime/registry dependencies.

Modified by AlpaGym to remove registry and metrics imports while preserving the
clipped-surrogate numerical core from ``rlinf/algorithms/losses.py``.
"""

from __future__ import annotations

import torch


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    clip_log_ratio_min: float | None = None,
    clip_log_ratio_max: float | None = None,
    dual_clip_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RLinf's clipped PPO loss and elementwise behavior ratio."""
    for name, tensor in (
        ("logprobs", logprobs),
        ("old_logprobs", old_logprobs),
        ("advantages", advantages),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32")
    if logprobs.shape != old_logprobs.shape or advantages.shape != logprobs.shape:
        raise ValueError("PPO logprobs, old_logprobs, and advantages must match")
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs, dtype=torch.bool)
    if loss_mask.shape != logprobs.shape or loss_mask.dtype is not torch.bool:
        raise TypeError("PPO loss_mask must be bool and match logprobs")

    log_ratio = logprobs - old_logprobs
    if clip_log_ratio_min is not None:
        log_ratio = torch.clamp(log_ratio, min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = torch.clamp(log_ratio, max=clip_log_ratio_max)
    ratio = torch.where(loss_mask, torch.exp(log_ratio), 0.0)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
    if dual_clip_ratio is not None:
        if dual_clip_ratio <= 1.0:
            raise ValueError("dual_clip_ratio must be greater than 1")
        dual_clip_loss = torch.sign(advantages) * dual_clip_ratio * advantages
        policy_loss = torch.minimum(policy_loss, dual_clip_loss)
    valid = loss_mask.to(policy_loss.dtype)
    return (policy_loss * valid).sum() / valid.sum().clamp_min(1.0), ratio


def compute_ppo_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    *,
    value_clip: float,
    huber_delta: float,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Return RLinf's clipped Huber critic loss over valid rows."""
    if values.dtype != torch.float32:
        raise TypeError("values must be float32")
    if returns.dtype != torch.float32:
        raise TypeError("returns must be float32")
    if old_values.dtype != torch.float32:
        raise TypeError("old_values must be float32")
    if values.shape != returns.shape or values.shape != old_values.shape:
        raise ValueError("values, returns, and old_values must match")
    if loss_mask.shape != values.shape or loss_mask.dtype is not torch.bool:
        raise TypeError("loss_mask must be bool and match values")
    if value_clip <= 0.0:
        raise ValueError("value_clip must be positive")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")

    clipped_values = old_values + (values - old_values).clamp(
        min=-value_clip,
        max=value_clip,
    )

    def huber(error: torch.Tensor) -> torch.Tensor:
        """Compute the elementwise Huber loss used by RLinf."""
        absolute = error.abs()
        quadratic = torch.minimum(
            absolute,
            torch.tensor(huber_delta, dtype=absolute.dtype, device=absolute.device),
        )
        linear = absolute - quadratic
        return 0.5 * quadratic.square() + huber_delta * linear

    loss = torch.maximum(
        huber(returns - values),
        huber(returns - clipped_values),
    )
    valid = loss_mask.to(loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)
