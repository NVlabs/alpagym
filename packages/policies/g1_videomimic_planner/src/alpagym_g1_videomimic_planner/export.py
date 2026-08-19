# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint conversion from a direct VideoMimic V9 policy to the planner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    register_g1_mjlab_model,
)
from safetensors.torch import save_file

from alpagym_g1_videomimic_planner.model import (
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    OBS_DIMS,
    OBS_KEYS,
    register_planner_model,
)
from alpagym_g1_videomimic_planner.provenance import (
    ACTION_STD_ATTESTATION_FIELDS,
    canonical_actor_state_sha256,
    file_sha256,
    initial_actor_lineage,
    initialization_attestation,
)

_PARITY_OBSERVATION_COUNT = 16
DEFAULT_EXPORT_ACTION_STD_MIN = 0.05
DEFAULT_EXPORT_ACTION_STD_MAX = 0.15


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
    direct = _load_direct_v9_model(planner.config, checkpoint, device=device)
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
    critic_seed: int = 0,
    min_action_std: float = DEFAULT_EXPORT_ACTION_STD_MIN,
    max_action_std: float = DEFAULT_EXPORT_ACTION_STD_MAX,
) -> None:
    """Write an H70 planner bundle initialized from the raw V9 actor.

    Native V9 clamps exploration std to ``[0.05, 0.15]`` before collecting a
    rollout. Apply that contract during export so step 0 and critic-only warmup
    never sample from the wider std stored in some source checkpoints.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"planner output directory is not empty: {output_dir}")
    if not isinstance(critic_seed, int) or not 0 <= critic_seed < 2**63:
        raise ValueError("critic_seed must be an integer in [0, 2**63)")
    min_action_std, max_action_std = _validated_action_std_clamp(
        min_action_std,
        max_action_std,
    )
    register_planner_model()
    source_sha256 = file_sha256(source_checkpoint)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V9 checkpoint root must be a mapping")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(critic_seed)
        planner = G1VideoMimicPlannerActorCriticModel(config)
        initialize_actor_from_v9_checkpoint(planner, checkpoint)
    direct = _load_direct_v9_model(config, checkpoint, device=torch.device("cpu"))
    source_std_min, source_std_max = _action_std_range(direct)
    planner.clamp_std_(min_std=min_action_std, max_std=max_action_std)
    exported_std_min, exported_std_max = _action_std_range(planner)
    parity_max_abs = _actor_parity_max_abs(planner, direct)
    if parity_max_abs != 0.0:
        raise ValueError(
            "exported planner actor differs from source V9 actor: "
            f"max_abs={parity_max_abs}"
        )
    if file_sha256(source_checkpoint) != source_sha256:
        raise ValueError("source V9 checkpoint changed while being exported")
    config.source_v9_checkpoint_sha256 = source_sha256
    config.planner_critic_seed = critic_seed
    config.source_v9_actor_parity_samples = _PARITY_OBSERVATION_COUNT
    config.source_v9_actor_parity_max_abs = parity_max_abs
    config.checkpoint_path = "model.safetensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    state_dict = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in planner.state_dict().items()
    }
    save_file(state_dict, str(weights_path))
    actor_initialization_attestation = initialization_attestation(
        source_v9_checkpoint_sha256=source_sha256,
        source_v9_actor_parity_samples=_PARITY_OBSERVATION_COUNT,
        source_v9_actor_parity_max_abs=parity_max_abs,
        planner_critic_seed=critic_seed,
        source_v9_action_std_min=source_std_min,
        source_v9_action_std_max=source_std_max,
        applied_action_std_clamp_min=min_action_std,
        applied_action_std_clamp_max=max_action_std,
        exported_action_std_min=exported_std_min,
        exported_action_std_max=exported_std_max,
    )
    for name in ACTION_STD_ATTESTATION_FIELDS:
        setattr(config, name, actor_initialization_attestation[name])
    config.actor_initialization_attestation = actor_initialization_attestation
    config.actor_update_lineage = initial_actor_lineage(
        actor_initialization_attestation,
        current_model_weights_sha256=file_sha256(weights_path),
        current_actor_state_sha256=canonical_actor_state_sha256(
            planner,
            state_dict=state_dict,
        ),
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_direct_v9_model(
    planner_config: G1VideoMimicPlannerConfig,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> G1MjlabActorCriticModel:
    """Decode one native or legacy V9 state into the direct actor architecture."""
    raw_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("V9 checkpoint must contain a state dict")
    register_g1_mjlab_model()
    direct_config = G1MjlabConfig(
        action_dim=int(planner_config.action_dim),
        hidden_dims=list(planner_config.hidden_dims),
        init_std=float(planner_config.init_std),
        activation=str(planner_config.activation),
        task_context_dim=int(planner_config.task_context_dim),
        checkpoint_path=None,
    )
    direct = G1MjlabActorCriticModel(direct_config).to(device=device)
    direct._load_state_dict(dict(raw_state), device=device)
    direct.eval()
    return direct


def _validated_action_std_clamp(
    min_action_std: float,
    max_action_std: float,
) -> tuple[float, float]:
    parsed_min = float(min_action_std)
    parsed_max = float(max_action_std)
    if (
        not math.isfinite(parsed_min)
        or not math.isfinite(parsed_max)
        or parsed_min <= 0.0
        or parsed_min > parsed_max
    ):
        raise ValueError(
            "action std clamp must satisfy 0 < min_action_std <= max_action_std"
        )
    return parsed_min, parsed_max


def _action_std_range(
    model: G1MjlabActorCriticModel,
) -> tuple[float, float]:
    std = model.std.detach()
    if std.ndim != 1 or std.numel() != int(model.config.action_dim):
        raise ValueError("V9 action std has an invalid shape")
    if not torch.isfinite(std).all() or not bool((std > 0.0).all()):
        raise ValueError("V9 action std must be finite and positive")
    return float(std.min().item()), float(std.max().item())


def _actor_parity_max_abs(
    planner: G1VideoMimicPlannerActorCriticModel,
    direct: G1MjlabActorCriticModel,
) -> float:
    """Verify the exported actor on deterministic real-shape observations."""
    observations = _parity_observations()
    planner.eval()
    direct.eval()
    with torch.no_grad():
        planner_actions = planner._head(observations, actor=True)
        direct_actions = direct._head(observations, actor=True)
    return float((planner_actions - direct_actions).abs().max().item())


def _parity_observations() -> dict[str, torch.Tensor]:
    """Build a deterministic real-shape actor probe panel."""
    return {
        key: torch.linspace(
            -0.75,
            0.75,
            steps=_PARITY_OBSERVATION_COUNT * OBS_DIMS[key],
            dtype=torch.float32,
        ).reshape(_PARITY_OBSERVATION_COUNT, OBS_DIMS[key])
        for key in OBS_KEYS
    }
