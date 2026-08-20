# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact q01/q99 normalization used by the pinned Wenhao navigation model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

WENHAO_STATE_DIM = 29
WENHAO_ACTION_DIM = 38
WENHAO_ACTION_ROWS = 30


@dataclass(frozen=True)
class WenhaoWireActions:
    """Keep PPO density-space latents separate from dispatched physical rows."""

    density_latent: torch.Tensor
    clipped_normalized: torch.Tensor
    denormalized: torch.Tensor

    def __post_init__(self) -> None:
        expected = self.density_latent.shape
        if self.density_latent.ndim != 3 or expected[-2:] != (
            WENHAO_ACTION_ROWS,
            WENHAO_ACTION_DIM,
        ):
            raise ValueError("Wenhao actions must have shape [B, 30, 38]")
        for name, tensor in (
            ("clipped_normalized", self.clipped_normalized),
            ("denormalized", self.denormalized),
        ):
            if tensor.shape != expected:
                raise ValueError(f"Wenhao {name} shape differs from density latent")
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (
                self.density_latent,
                self.clipped_normalized,
                self.denormalized,
            )
        ):
            raise ValueError("Wenhao wire actions contain non-finite values")
        if not torch.equal(self.clipped_normalized, self.density_latent.clamp(-1, 1)):
            raise ValueError(
                "Wenhao clipped_normalized must exactly equal clamp(density_latent)"
            )


class WenhaoQ99Normalizer(nn.Module):
    """Torch implementation of Psi ``ActionStateTransform(bounds_q99)``."""

    state_low: torch.Tensor
    state_high: torch.Tensor
    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(
        self,
        *,
        state_q01: torch.Tensor,
        state_q99: torch.Tensor,
        action_q01: torch.Tensor,
        action_q99: torch.Tensor,
    ) -> None:
        super().__init__()
        state_low, state_high = _bounds(
            state_q01, state_q99, width=WENHAO_STATE_DIM, label="state"
        )
        action_low, action_high = _bounds(
            action_q01, action_q99, width=WENHAO_ACTION_DIM, label="action"
        )
        self.register_buffer("state_low", state_low, persistent=False)
        self.register_buffer("state_high", state_high, persistent=False)
        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer("action_high", action_high, persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path) -> "WenhaoQ99Normalizer":
        """Load only the attested q01/q99 fields from one Psi stats JSON."""
        stats_path = Path(path)
        raw: Any = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("Wenhao stats root must be a mapping")
        state = raw.get("states")
        action = raw.get("action")
        if not isinstance(state, Mapping) or not isinstance(action, Mapping):
            raise ValueError("Wenhao stats require 'states' and 'action' mappings")
        try:
            return cls(
                state_q01=torch.as_tensor(state["q01"], dtype=torch.float32),
                state_q99=torch.as_tensor(state["q99"], dtype=torch.float32),
                action_q01=torch.as_tensor(action["q01"], dtype=torch.float32),
                action_q99=torch.as_tensor(action["q99"], dtype=torch.float32),
            )
        except KeyError as exc:
            raise ValueError(f"Wenhao stats are missing {exc.args[0]!r}") from exc

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize and clamp finite 29-D proprio exactly like Psi training."""
        return _normalize(
            state,
            self.state_low,
            self.state_high,
            width=WENHAO_STATE_DIM,
            label="state",
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize physical 38-D rows for RTC prefix conditioning."""
        return _normalize(
            action,
            self.action_low,
            self.action_high,
            width=WENHAO_ACTION_DIM,
            label="action",
        )

    def denormalize_action(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized rows to physical targets without implicit clipping."""
        normalized = _finite_last_dim(
            normalized, width=WENHAO_ACTION_DIM, label="normalized action"
        )
        low = self.action_low.to(device=normalized.device, dtype=normalized.dtype)
        high = self.action_high.to(device=normalized.device, dtype=normalized.dtype)
        return 0.5 * (normalized + 1.0) * (high - low) + low

    def to_wire(self, density_latent: torch.Tensor) -> WenhaoWireActions:
        """Clamp only after density evaluation, then denormalize for dispatch."""
        density_latent = _finite_last_dim(
            density_latent, width=WENHAO_ACTION_DIM, label="density latent"
        )
        if density_latent.ndim != 3 or density_latent.shape[-2] != WENHAO_ACTION_ROWS:
            raise ValueError("Wenhao density latent must have shape [B, 30, 38]")
        clipped = density_latent.clamp(-1.0, 1.0)
        return WenhaoWireActions(
            density_latent=density_latent,
            clipped_normalized=clipped,
            denormalized=self.denormalize_action(clipped),
        )


def _bounds(
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    width: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.as_tensor(low, dtype=torch.float32).detach().reshape(-1)
    high = torch.as_tensor(high, dtype=torch.float32).detach().reshape(-1)
    if tuple(low.shape) != (width,) or tuple(high.shape) != (width,):
        raise ValueError(f"Wenhao {label} q01/q99 must have shape ({width},)")
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError(f"Wenhao {label} q01/q99 contain non-finite values")
    tolerance = 1.0e-4 * (low.abs() + high.abs() + 1.0e-8)
    if torch.any((high - low).abs() < tolerance):
        raise ValueError(f"Wenhao {label} q01/q99 contain a near-degenerate dimension")
    if not torch.all(high > low):
        raise ValueError(f"Wenhao {label} q99 must exceed q01")
    return low.contiguous(), high.contiguous()


def _finite_last_dim(
    value: torch.Tensor,
    *,
    width: int,
    label: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim < 1 or tensor.shape[-1] != width:
        raise ValueError(f"Wenhao {label} must end in dimension {width}")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Wenhao {label} contains non-finite values")
    return tensor


def _normalize(
    value: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    width: int,
    label: str,
) -> torch.Tensor:
    value = _finite_last_dim(value, width=width, label=label)
    low = low.to(device=value.device, dtype=value.dtype)
    high = high.to(device=value.device, dtype=value.dtype)
    return ((value - low) / (high - low) * 2.0 - 1.0).clamp(-1.0, 1.0)
