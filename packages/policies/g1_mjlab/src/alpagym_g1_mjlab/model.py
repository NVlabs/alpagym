# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-RL model registration for the G1 mjlab actor-critic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from cosmos_rl.policy.model.base import BaseModel, IdentityWeightMapper, ModelRegistry
from transformers import AutoConfig, PretrainedConfig

G1_MJLAB_MODEL_TYPE = "g1_mjlab_actor_critic"
G1_MJLAB_REPLAY_SCHEMA = "alpagym_humanoid.g1_mjlab.v1"
OBSERVATION_SCHEMA = "videomimic_v9_direct.v1"
ACTION_SCHEMA = "g1_joint_offset.v1"

OBS_KEYS = (
    "history_torso_real",
    "torso_xy_rel",
    "torso_yaw_rel",
    "terrain_height_noisy",
)
BASE_OBS_KEYS = ("history_torso_real", "torso_xy_rel", "torso_yaw_rel")
TERRAIN_KEY = "terrain_height_noisy"
TASK_CONTEXT_KEY = "task_context"
OBS_DIMS = {
    "history_torso_real": 375,
    "torso_xy_rel": 2,
    "torso_yaw_rel": 1,
    "terrain_height_noisy": 121,
}
BASE_OBS_DIM = sum(OBS_DIMS[key] for key in BASE_OBS_KEYS)
DEFAULT_ACTION_DIM = 23
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


class G1MjlabConfig(PretrainedConfig):
    """HF-shaped config for a small continuous-action G1 actor-critic."""

    model_type = G1_MJLAB_MODEL_TYPE

    def __init__(
        self,
        *,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 512, 256, 128),
        init_std: float = 0.25,
        activation: str = "elu",
        checkpoint_path: str | None = "pytorch_model.bin",
        task_context_dim: int = 0,
        motion_reference_residual: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.action_dim = int(action_dim)
        self.hidden_dims = [int(width) for width in hidden_dims]
        self.init_std = float(init_std)
        self.activation = str(activation)
        self.checkpoint_path = checkpoint_path
        self.task_context_dim = int(task_context_dim)
        self.motion_reference_residual = bool(motion_reference_residual)


class G1MjlabActorCriticModel(BaseModel):
    """Checkpoint-compatible Gaussian actor-critic for G1 stair/VideoMimic PPO."""

    def __init__(self, hf_config: G1MjlabConfig) -> None:
        super().__init__(hf_config)
        self.config = hf_config
        hidden_dims = tuple(int(width) for width in hf_config.hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        first = hidden_dims[0]
        action_dim = int(hf_config.action_dim)
        self.actor_terrain = nn.Linear(OBS_DIMS[TERRAIN_KEY], first)
        self.critic_terrain = nn.Linear(OBS_DIMS[TERRAIN_KEY], first)
        self.actor_attention = nn.Parameter(torch.ones(first))
        self.critic_attention = nn.Parameter(torch.ones(first))
        self.actor_context = (
            nn.Linear(int(hf_config.task_context_dim), first, bias=False)
            if int(hf_config.task_context_dim) > 0
            else None
        )
        self.critic_context = (
            nn.Linear(int(hf_config.task_context_dim), first, bias=False)
            if int(hf_config.task_context_dim) > 0
            else None
        )
        if self.actor_context is not None:
            nn.init.zeros_(self.actor_context.weight)
        if self.critic_context is not None:
            nn.init.zeros_(self.critic_context.weight)
        self.actor = self._mlp(
            BASE_OBS_DIM, hidden_dims, action_dim, hf_config.activation
        )
        self.critic = self._mlp(BASE_OBS_DIM, hidden_dims, 1, hf_config.activation)
        self.std = nn.Parameter(torch.full((action_dim,), float(hf_config.init_std)))
        self._model_dir: Path | None = None
        self._weights_loaded = False

    @staticmethod
    def supported_model_types() -> list[str]:
        return [G1_MJLAB_MODEL_TYPE]

    @property
    def parallelize_fn(self):
        return _identity_parallelize, None

    def apply_pipeline_split(self, pp_rank: int, pp_size: int) -> None:
        del pp_rank, pp_size
        return None

    def post_to_empty_hook(self, cosmos_config: Any) -> None:
        self.load_hf_weights(
            cosmos_config.policy.model_name_or_path,
            cosmos_config.parallelism
            if hasattr(cosmos_config, "parallelism")
            else None,
            self.current_device(),
        )

    def get_position_ids(self, **kwargs: Any):
        del kwargs
        device = self.current_device()
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, 0

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: Any,
        device: torch.device,
        revision: str | None = None,
    ) -> None:
        del parallel_dims, revision
        if self._weights_loaded:
            return
        checkpoint_path = _resolve_checkpoint_path(
            model_name_or_path=model_name_or_path,
            configured_path=self.config.checkpoint_path,
        )
        if checkpoint_path is None:
            self._weights_loaded = True
            return
        if checkpoint_path.name == "model.safetensors.index.json":
            state_dict = _load_safetensors_shards(checkpoint_path)
        elif checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path), device="cpu")
        else:
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"G1 checkpoint {checkpoint_path} must contain a state dict or "
                "a mapping with model_state_dict"
            )
        self._load_state_dict(dict(state_dict), device=device)
        self._weights_loaded = True

    def separate_model_parts(self) -> list[nn.Module]:
        return [self]

    @classmethod
    def from_pretrained(
        cls,
        hf_config: G1MjlabConfig,
        model_name_or_path: str,
        max_position_embeddings: int | None = None,
    ) -> G1MjlabActorCriticModel:
        del max_position_embeddings
        model = cls(hf_config)
        model._model_dir = Path(model_name_or_path)
        return model

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        del seq_len
        return 0, 0

    def forward(
        self,
        *,
        actions: torch.Tensor | None = None,
        teacher_model: Any = None,
        return_log_prob: bool = True,
        **obs: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        del return_log_prob
        mean, value = self._forward_heads(obs)
        scored_actions = (
            mean
            if actions is None
            else actions.to(device=mean.device, dtype=mean.dtype)
        )
        dist = self.distribution(mean)
        log_probs = dist.log_prob(scored_actions).sum(dim=-1)
        kl_div = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_mean, _teacher_value = teacher_model._forward_heads(obs)
                teacher_std = teacher_model.std.clamp(min=1.0e-6).expand_as(
                    teacher_mean
                )
                teacher_dist = torch.distributions.Normal(teacher_mean, teacher_std)
            kl_div = torch.distributions.kl_divergence(dist, teacher_dist).sum(dim=-1)
        return {"log_probs": log_probs, "values": value.squeeze(-1), "kl_div": kl_div}

    def forward_values(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate only the critic head for value-only PPO optimization."""

        return self._head(obs, actor=False).squeeze(-1)

    def ppo_parameter_groups(
        self,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return the complete actor/critic ownership partition.

        Planner checkpoint provenance hashes the actor group, including the
        standalone action standard deviation and optional actor context, while
        excluding every critic parameter.
        """
        actor_parameters = [
            *self.actor.parameters(),
            *self.actor_terrain.parameters(),
            self.actor_attention,
            self.std,
        ]
        critic_parameters = [
            *self.critic.parameters(),
            *self.critic_terrain.parameters(),
            self.critic_attention,
        ]
        if self.actor_context is not None:
            actor_parameters.extend(self.actor_context.parameters())
        if self.critic_context is not None:
            critic_parameters.extend(self.critic_context.parameters())
        groups = {
            "actor": tuple(actor_parameters),
            "critic": tuple(critic_parameters),
        }
        trainable_ids = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        actor_ids = {id(parameter) for parameter in groups["actor"]}
        critic_ids = {id(parameter) for parameter in groups["critic"]}
        if actor_ids & critic_ids:
            raise RuntimeError("actor and critic parameter groups overlap")
        if actor_ids | critic_ids != trainable_ids:
            raise RuntimeError(
                "actor/critic parameter groups do not cover every trainable "
                "model parameter"
            )
        return groups

    def act(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self._forward_heads(obs)
        dist = self.distribution(mean)
        if deterministic:
            action = mean
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            action = mean + dist.scale * noise
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1), mean

    def evaluate_actions(
        self,
        obs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self._forward_heads(obs)
        dist = self.distribution(mean)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy, mean

    def distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        std = self.std.clamp(min=1.0e-6).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def clamp_std_(self, *, min_std: float, max_std: float) -> None:
        self.std.data.clamp_(min=float(min_std), max=float(max_std))

    def _forward_heads(
        self,
        obs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._head(obs, actor=True), self._head(obs, actor=False)

    def _head(self, obs: Mapping[str, torch.Tensor], *, actor: bool) -> torch.Tensor:
        net = self.actor if actor else self.critic
        terrain = self.actor_terrain if actor else self.critic_terrain
        attention = self.actor_attention if actor else self.critic_attention
        base = torch.cat([obs[key] for key in BASE_OBS_KEYS], dim=-1)
        hidden = net[0](base) + terrain(obs[TERRAIN_KEY]) * attention
        context_layer = self.actor_context if actor else self.critic_context
        if context_layer is not None:
            context = obs.get(TASK_CONTEXT_KEY)
            if context is None:
                context = torch.zeros(
                    (*base.shape[:-1], int(self.config.task_context_dim)),
                    dtype=base.dtype,
                    device=base.device,
                )
            elif int(context.shape[-1]) < int(self.config.task_context_dim):
                raise ValueError(
                    f"task context has width {context.shape[-1]}, expected at least "
                    f"{self.config.task_context_dim}"
                )
            else:
                context = context[..., : int(self.config.task_context_dim)]
            hidden = hidden + context_layer(context)
        x = net[1](hidden)
        for layer in net[2:]:
            x = layer(x)
        return x

    def _mlp(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        activation: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(_activation(activation))
            prev = int(width)
        layers.append(nn.Linear(prev, int(out_dim)))
        return nn.Sequential(*layers)

    def _load_state_dict(
        self, state_dict: dict[str, Any], *, device: torch.device
    ) -> None:
        normalized = _strip_known_prefixes(state_dict)
        if "actor_terrain.weight" in normalized:
            self.load_state_dict(_to_device_state(normalized, device), strict=False)
            return
        self._load_legacy_state_dict(normalized, device=device)

    def _load_legacy_state_dict(
        self, state_dict: Mapping[str, Any], *, device: torch.device
    ) -> None:
        self.std.data.copy_(_state_tensor(state_dict, "std", self.std, device))
        self.actor_attention.data.copy_(
            _state_tensor(
                state_dict,
                "actor_input_net.extra_proj_heads.terrain_height_noisy.attention",
                self.actor_attention,
                device,
            )
        )
        self.actor_terrain.weight.data.copy_(
            _state_tensor(
                state_dict,
                "actor_input_net.extra_proj_heads.terrain_height_noisy.embed.embed.input_proc.weight",
                self.actor_terrain.weight,
                device,
            )
        )
        self.actor_terrain.bias.data.copy_(
            _state_tensor(
                state_dict,
                "actor_input_net.extra_proj_heads.terrain_height_noisy.embed.embed.input_proc.bias",
                self.actor_terrain.bias,
                device,
            )
        )
        self.critic_attention.data.copy_(
            _state_tensor(
                state_dict,
                "critic_input_net.extra_proj_heads.terrain_height_noisy.attention",
                self.critic_attention,
                device,
            )
        )
        self.critic_terrain.weight.data.copy_(
            _state_tensor(
                state_dict,
                "critic_input_net.extra_proj_heads.terrain_height_noisy.embed.embed.input_proc.weight",
                self.critic_terrain.weight,
                device,
            )
        )
        self.critic_terrain.bias.data.copy_(
            _state_tensor(
                state_dict,
                "critic_input_net.extra_proj_heads.terrain_height_noisy.embed.embed.input_proc.bias",
                self.critic_terrain.bias,
                device,
            )
        )
        self.actor.load_state_dict(
            {
                key.removeprefix("actor."): value.to(device=device)
                for key, value in state_dict.items()
                if key.startswith("actor.")
            }
        )
        self.critic.load_state_dict(
            {
                key.removeprefix("critic."): value.to(device=device)
                for key, value in state_dict.items()
                if key.startswith("critic.")
            }
        )


def register_g1_mjlab_model() -> None:
    """Register G1 model config, tokenizer path, and weight mapper with Cosmos."""

    try:
        AutoConfig.register(G1_MJLAB_MODEL_TYPE, G1MjlabConfig)
    except ValueError as exc:
        if "is already used" not in str(exc) and "already exists" not in str(exc):
            raise
    if G1_MJLAB_MODEL_TYPE not in ModelRegistry._MODEL_REGISTRY:
        ModelRegistry.register_model(G1MjlabActorCriticModel, IdentityWeightMapper)


def split_flat_observation(flat: Any) -> dict[str, torch.Tensor]:
    tensor = torch.as_tensor(flat, dtype=torch.float32).reshape(-1)
    expected = sum(OBS_DIMS[key] for key in OBS_KEYS)
    if tuple(tensor.shape) != (expected,):
        raise ValueError(
            f"flat observation must have shape ({expected},), got {tuple(tensor.shape)}"
        )
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for key in OBS_KEYS:
        dim = OBS_DIMS[key]
        out[key] = tensor[offset : offset + dim].clone()
        offset += dim
    return out


def stack_observations(
    observations: list[Mapping[str, torch.Tensor]], device: torch.device
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("cannot stack an empty observation sequence")
    return {
        key: torch.stack(
            [obs[key].to(dtype=torch.float32) for obs in observations], dim=0
        ).to(device)
        for key in OBS_KEYS
    }


def _identity_parallelize(
    model: G1MjlabActorCriticModel,
    parallel_dims: Any,
    config: Any,
    pp_loss_fn: Any = None,
) -> tuple[None, None]:
    del model, parallel_dims, config, pp_loss_fn
    return None, None


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


def _resolve_checkpoint_path(
    model_name_or_path: str, configured_path: str | None
) -> Path | None:
    if not configured_path:
        return None
    path = Path(configured_path)
    if not path.is_absolute():
        model_dir = Path(model_name_or_path)
        path = model_dir / path
        safetensors_index = model_dir / "model.safetensors.index.json"
        if path.name == "pytorch_model.bin" and safetensors_index.is_file():
            return safetensors_index
        safetensors_file = model_dir / "model.safetensors"
        if path.name == "pytorch_model.bin" and safetensors_file.is_file():
            return safetensors_file
    if not path.is_file():
        raise FileNotFoundError(f"G1 checkpoint not found: {path}")
    return path


def _load_safetensors_shards(index_path: Path) -> dict[str, torch.Tensor]:
    """Load the tensors named by a HuggingFace safetensors shard index."""
    from safetensors.torch import load_file

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise TypeError(f"invalid safetensors weight_map in {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for filename in dict.fromkeys(weight_map.values()):
        shard_path = index_path.parent / filename
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"safetensors shard listed by {index_path} not found: {shard_path}"
            )
        shard = load_file(str(shard_path), device="cpu")
        expected_keys = {
            key
            for key, mapped_filename in weight_map.items()
            if mapped_filename == filename
        }
        missing_keys = expected_keys.difference(shard)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"safetensors shard {shard_path} is missing: {missing}")
        state_dict.update({key: shard[key] for key in expected_keys})
    return state_dict


def _strip_known_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key.removeprefix(prefix)
        result[new_key] = value
    return result


def _to_device_state(
    state_dict: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value).to(device=device)
        for key, value in state_dict.items()
    }


def _state_tensor(
    state_dict: Mapping[str, Any],
    key: str,
    target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(state_dict[key], dtype=target.dtype).to(device=device)
