# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaGym policy bundle entry point for G1 mjlab PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from alpagym_runtime.policies.registry import PolicyBundle
from alpagym_runtime.replay import PolicyReplayData, require_payload_keys
from alpagym_g1_mjlab.inference_model import G1MjlabInferenceModel
from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    G1_MJLAB_REPLAY_SCHEMA,
    OBS_KEYS,
    register_g1_mjlab_model,
)


def setup_tokenizer(config: Any) -> Any:
    """Register the G1 model with Cosmos and return a tokenizer-shaped no-op."""
    del config
    register_g1_mjlab_model()
    return _no_op_tokenizer()


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the standard AlpaGym replay packer with G1 model-input parsing."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    register_g1_mjlab_model()
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
    )


def install_runtime_bridge() -> None:
    """Install runtime registrations needed before Cosmos constructs workers."""
    register_g1_mjlab_model()


def load_inference_model(
    run_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> G1MjlabInferenceModel:
    """Load a G1 actor-critic bundle for rollout-side humanoid policy serving."""
    if dtype != torch.float32:
        raise ValueError(f"G1 mjlab expects dtype=float32; got {dtype!r}")
    register_g1_mjlab_model()
    bundle_dir = Path(run_config.policy.model.path)
    config = G1MjlabConfig.from_pretrained(bundle_dir)
    model = G1MjlabActorCriticModel.from_pretrained(config, str(bundle_dir))
    model.to(device=device, dtype=torch.float32)
    model.load_hf_weights(str(bundle_dir), parallel_dims=None, device=device)
    model.eval()
    return G1MjlabInferenceModel(model)


def build_model_inputs(
    run_config: Any,
) -> Callable[[PolicyReplayData], tuple[dict[str, Any], torch.Tensor]]:
    """Return the trainer-side replay parser for G1 continuous actions."""
    del run_config

    def _build(replay_data: PolicyReplayData) -> tuple[dict[str, Any], torch.Tensor]:
        if replay_data.payload_schema != G1_MJLAB_REPLAY_SCHEMA:
            raise ValueError(
                f"payload_schema={replay_data.payload_schema!r}; expected "
                f"{G1_MJLAB_REPLAY_SCHEMA!r}"
            )
        require_payload_keys(
            "g1_mjlab",
            replay_data.payload,
            ("observation", "action"),
        )
        observation = replay_data.payload["observation"]
        if not isinstance(observation, dict):
            raise TypeError("G1 replay payload['observation'] must be a mapping")
        require_payload_keys("g1_mjlab.observation", observation, OBS_KEYS)
        model_inputs: dict[str, Any] = {
            key: torch.as_tensor(observation[key], dtype=torch.float32)
            for key in OBS_KEYS
        }
        if "task_context" in replay_data.payload:
            model_inputs["task_context"] = torch.as_tensor(
                replay_data.payload["task_context"],
                dtype=torch.float32,
            )
        model_inputs["actions"] = torch.as_tensor(
            replay_data.payload["action"],
            dtype=torch.float32,
        )
        if replay_data.old_logprob is None:
            raise ValueError("G1 replay payload is missing old_logprob")
        return model_inputs, replay_data.old_logprob.to(dtype=torch.float32).reshape(())

    return _build


def get_bundle() -> PolicyBundle:
    """Return the G1 mjlab policy hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )


def _no_op_tokenizer() -> Any:
    try:
        from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer

        return NoOpTokenizer()
    except Exception:
        class _NoOpTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                return {}

        return _NoOpTokenizer()
