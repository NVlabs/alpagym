# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaGym policy bundle entry point for G1 mjlab PPO."""

from __future__ import annotations

import json
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

        tokenizer: Any = NoOpTokenizer()
    except Exception:

        class _NoOpTokenizer:
            pad_token_id = 0
            eos_token_id = 0
            model_max_length = 512
            vocab_size = 4

            def encode(self, item: Any, **kwargs: Any) -> list[int]:
                del kwargs
                if isinstance(item, dict):
                    return [0] * max(1, int(item.get("episode_length", 1)))
                return [0]

            def decode(self, *args: Any, **kwargs: Any) -> str:
                del args, kwargs
                return ""

            def batch_decode(self, batches: Any, **kwargs: Any) -> list[str]:
                del kwargs
                try:
                    return [""] * len(batches)
                except TypeError:
                    return [""]

            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                del args, kwargs
                return {}

        tokenizer = _NoOpTokenizer()
    return _CheckpointableNoOpTokenizer(tokenizer)


class _CheckpointableNoOpTokenizer:
    """Add deterministic checkpoint persistence to Cosmos's non-text tokenizer.

    Cosmos calls ``data_packer.save_state()`` from its safetensors export
    thread, which in turn unconditionally calls ``tokenizer.save_pretrained``.
    The upstream ``NoOpTokenizer`` intentionally implements only inference
    methods, so the G1 bundle supplies the missing persistence boundary here.
    Reload does not consume this marker: ``setup_tokenizer`` always recreates
    the same non-text tokenizer after registering the G1 model family.
    """

    _MARKER_NAME = "g1_mjlab_no_op_tokenizer.json"

    def __init__(self, tokenizer: Any) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._tokenizer, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(*args, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs: Any,
    ) -> tuple[str]:
        """Persist a small, deterministic marker using the HF method shape."""
        del kwargs
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        marker_path = destination / self._MARKER_NAME
        payload = {
            "format_version": 1,
            "tokenizer_type": "alpagym_g1_mjlab_no_op",
            "pad_token_id": int(self.pad_token_id),
            "eos_token_id": int(self.eos_token_id),
            "model_max_length": int(getattr(self, "model_max_length", 1)),
            "vocab_size": int(getattr(self, "vocab_size", 1)),
        }
        marker_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return (str(marker_path),)

    def __repr__(self) -> str:
        return f"Checkpointable({self._tokenizer!r})"
