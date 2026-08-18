# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint conversion from a direct VideoMimic V9 policy to the planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    register_g1_mjlab_model,
)

from alpagym_g1_videomimic_planner.model import (
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    register_planner_model,
)


def initialize_actor_from_v9_checkpoint(
    planner: G1VideoMimicPlannerActorCriticModel,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Load only V9 actor/terrain/std and explicitly reset the planner critic.

    A temporary direct model decodes both native and legacy VideoMimic key
    layouts.  Only actor-owned parameters are copied into the planner. The
    source critic never enters the planner state, avoiding an accidental value
    scale inherited from 50 Hz direct-control training.
    """
    raw_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("V9 checkpoint must contain a state dict")
    register_g1_mjlab_model()
    direct_config = G1MjlabConfig(
        action_dim=int(planner.config.action_dim),
        hidden_dims=list(planner.config.hidden_dims),
        init_std=float(planner.config.init_std),
        activation=str(planner.config.activation),
        task_context_dim=int(planner.config.task_context_dim),
        checkpoint_path=None,
    )
    direct = G1MjlabActorCriticModel(direct_config).to(device=device)
    direct._load_state_dict(dict(raw_state), device=device)
    planner.actor.load_state_dict(direct.actor.state_dict())
    planner.actor_terrain.load_state_dict(direct.actor_terrain.state_dict())
    planner.actor_attention.data.copy_(direct.actor_attention.data)
    if planner.actor_context is not None:
        if direct.actor_context is None:
            raise ValueError("source V9 checkpoint has no actor context layer")
        planner.actor_context.load_state_dict(direct.actor_context.state_dict())
    planner.std.data.copy_(direct.std.data)
    planner.reset_planner_critic_()


def export_planner_checkpoint(
    *,
    source_checkpoint: Path,
    output_dir: Path,
    config: G1VideoMimicPlannerConfig,
) -> None:
    """Write one self-contained planner config/checkpoint directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"planner output directory is not empty: {output_dir}")
    register_planner_model()
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V9 checkpoint root must be a mapping")
    planner = G1VideoMimicPlannerActorCriticModel(config)
    initialize_actor_from_v9_checkpoint(planner, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": planner.state_dict()},
        output_dir / "pytorch_model.bin",
    )
