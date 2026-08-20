# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation-only critic overlay for the Wenhao Psi0 actor."""

from __future__ import annotations

import torch
import torch.nn as nn

from alpagym_runtime.third_party.rlinf.value_head import ValueHead


class WenhaoValueModel(nn.Module):
    """Value head over VLM context, proprio state, and deterministic RTC prefix."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        state_dim: int = 29,
        action_dim: int = 38,
        prefix_embed_dim: int = 64,
        hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        detach_vlm: bool = True,
    ) -> None:
        super().__init__()
        if min(vlm_hidden_dim, state_dim, action_dim, prefix_embed_dim) <= 0:
            raise ValueError("Wenhao critic dimensions must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.detach_vlm = bool(detach_vlm)
        self.prefix_projection = nn.Linear(action_dim, prefix_embed_dim)
        self.value_head = ValueHead(
            input_dim=vlm_hidden_dim + state_dim + prefix_embed_dim + 1,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

    def forward(
        self,
        vlm_hidden: torch.Tensor,
        vlm_attention_mask: torch.Tensor,
        state: torch.Tensor,
        rtc_prefix_actions: torch.Tensor,
        rtc_prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one finite state value per batch item."""
        if vlm_hidden.ndim != 3:
            raise ValueError("Wenhao critic VLM hidden must be [B, S, D]")
        batch, sequence_length, _hidden = vlm_hidden.shape
        attention_mask = torch.as_tensor(
            vlm_attention_mask, device=vlm_hidden.device, dtype=torch.bool
        )
        if tuple(attention_mask.shape) != (batch, sequence_length):
            raise ValueError("Wenhao critic attention mask must be [B, S]")
        state = torch.as_tensor(state, device=vlm_hidden.device, dtype=torch.float32)
        if tuple(state.shape) == (batch, 1, self.state_dim):
            state = state[:, 0]
        if tuple(state.shape) != (batch, self.state_dim):
            raise ValueError("Wenhao critic state must be [B, 29] or [B, 1, 29]")
        prefix_actions = torch.as_tensor(
            rtc_prefix_actions, device=vlm_hidden.device, dtype=torch.float32
        )
        prefix_mask = torch.as_tensor(
            rtc_prefix_mask, device=vlm_hidden.device, dtype=torch.bool
        )
        if (
            prefix_actions.ndim != 3
            or tuple(prefix_actions.shape[:2]) != tuple(prefix_mask.shape)
            or prefix_actions.shape[0] != batch
            or prefix_actions.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "Wenhao critic RTC prefix tensors have incompatible shapes"
            )
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (vlm_hidden, state, prefix_actions)
        ):
            raise ValueError("Wenhao critic input contains non-finite values")

        # Rollout runs the 2B actor under bf16 CUDA autocast. Keep the small
        # critic genuinely fp32 so rollout, replay, and bootstrap values share
        # one numerical contract instead of inheriting the caller's autocast.
        with torch.autocast(device_type=vlm_hidden.device.type, enabled=False):
            critic_hidden = vlm_hidden.detach() if self.detach_vlm else vlm_hidden
            critic_hidden = critic_hidden.to(torch.float32)
            attention = attention_mask.to(torch.float32).unsqueeze(-1)
            vlm_pooled = (critic_hidden * attention).sum(dim=1) / attention.sum(
                dim=1
            ).clamp_min(1.0)
            prefix_embeddings = torch.relu(
                self.prefix_projection(prefix_actions.to(torch.float32))
            )
            prefix_weight = prefix_mask.to(torch.float32).unsqueeze(-1)
            prefix_pooled = (prefix_embeddings * prefix_weight).sum(
                dim=1
            ) / prefix_weight.sum(dim=1).clamp_min(1.0)
            prefix_fraction = prefix_mask.to(torch.float32).mean(dim=1, keepdim=True)
            features = torch.cat(
                [vlm_pooled, state, prefix_pooled, prefix_fraction], dim=-1
            ).to(torch.float32)
            value = self.value_head(features).squeeze(-1).to(torch.float32)
        if tuple(value.shape) != (batch,) or not torch.isfinite(value).all():
            raise FloatingPointError("Wenhao critic returned an invalid value")
        return value
