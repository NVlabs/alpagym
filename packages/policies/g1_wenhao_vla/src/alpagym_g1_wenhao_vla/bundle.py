# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy bundle and fail-closed trainer replay parser for Wenhao Psi0."""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch

from alpagym_g1_wenhao_vla.replay_collator import (
    collate_wenhao_replay_samples,
)
from alpagym_g1_wenhao_vla.cosmos_model import (
    load_wenhao_rollout_model,
    register_wenhao_psi_ppo_model,
)
from alpagym_g1_wenhao_vla.inference_model import WenhaoNativeInferenceModel
from alpagym_g1_wenhao_vla.flow import (
    WENHAO_FLOW_IGNORE_LAST,
    WENHAO_FLOW_NOISE_LEVEL,
)
from alpagym_g1_wenhao_vla.provenance import MODEL_ID, RUN_CONFIG_SHA256
from alpagym_runtime.policies.registry import PolicyBundle
from alpagym_runtime.replay import PolicyReplayData, require_payload_keys

REPLAY_SCHEMA = "g1_wenhao_vla.flow_sde.v1"
MODEL_FAMILY = "g1_wenhao_vla"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def setup_tokenizer(config: Any) -> Any:
    """Return Cosmos's non-text tokenizer for humanoid VLA rollouts."""
    del config
    register_wenhao_psi_ppo_model()
    from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer

    tokenizer = NoOpTokenizer()
    tokenizer.pad_token_id = 151643
    tokenizer.eos_token_id = 151645
    return _CheckpointableNoOpTokenizer(tokenizer)


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the generic replay packer with Wenhao's ragged collator."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    bundle_config = _bundle_config(run_config)
    if "pad_token_id" in bundle_config:
        raise ValueError(
            "Wenhao pad_token_id is bundle-owned and must not be configured"
        )
    if "policy_eval_root" in bundle_config:
        raise ValueError(
            "Wenhao policy_eval_root is obsolete; metadata is derived from "
            "policy.model.path"
        )
    pad_token_id = _attested_pad_token_id(run_config)
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
        collate_samples=functools.partial(
            collate_wenhao_replay_samples,
            pad_token_id=pad_token_id,
        ),
    )


def install_runtime_bridge() -> None:
    """Register the content-addressed Wenhao model with Cosmos and HF."""
    register_wenhao_psi_ppo_model()


def load_inference_model(
    run_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Load the attested Psi PPO model for humanoid callbacks."""
    register_wenhao_psi_ppo_model()
    model = load_wenhao_rollout_model(
        Path(run_config.policy.model.path),
        device=device,
        dtype=dtype,
    )
    return WenhaoNativeInferenceModel(model)


def build_model_inputs(
    run_config: Any,
) -> Callable[[PolicyReplayData], tuple[dict[str, Any], torch.Tensor]]:
    """Return the strict parser for one Wenhao Flow-SDE replay row."""
    expected_schedule_sha256 = _expected_schedule_sha256(run_config)
    expected_noise_level, expected_ignore_last = _expected_flow_contract(run_config)

    def _build(replay_data: PolicyReplayData) -> tuple[dict[str, Any], torch.Tensor]:
        if replay_data.replay_schema_version != 1:
            raise ValueError(
                "Wenhao replay_schema_version must be 1, got "
                f"{replay_data.replay_schema_version}"
            )
        if replay_data.payload_schema != REPLAY_SCHEMA:
            raise ValueError(
                f"payload_schema={replay_data.payload_schema!r}; expected "
                f"{REPLAY_SCHEMA!r}"
            )
        if replay_data.payload_schema_version != 1:
            raise ValueError(
                "Wenhao payload_schema_version must be 1, got "
                f"{replay_data.payload_schema_version}"
            )
        if replay_data.model_family != MODEL_FAMILY:
            raise ValueError(
                f"Wenhao replay model_family={replay_data.model_family!r}; "
                f"expected {MODEL_FAMILY!r}"
            )
        require_payload_keys(
            MODEL_FAMILY,
            replay_data.payload,
            (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "physical_states",
                "latent_chain",
                "denoise_index",
                "clipped_normalized_actions",
                "denormalized_wire_actions",
                "rtc_prefix_normalized_actions",
                "rtc_prefix_mask",
                "schedule_sha256",
                "flow_noise_level",
                "flow_ignore_last",
                "run_config_sha256",
                "old_element_logprobs",
                "humanoid",
            ),
        )
        payload_schedule = _require_sha256(
            "schedule_sha256", replay_data.payload["schedule_sha256"]
        )
        if payload_schedule != expected_schedule_sha256:
            raise ValueError(
                "Wenhao replay schedule_sha256 does not match "
                "policy.model.bundle_config.expected_schedule_sha256"
            )
        payload_noise_level = replay_data.payload["flow_noise_level"]
        if (
            isinstance(payload_noise_level, bool)
            or not isinstance(payload_noise_level, (int, float))
            or float(payload_noise_level) != expected_noise_level
        ):
            raise ValueError(
                "Wenhao replay flow_noise_level does not match the pinned policy"
            )
        if replay_data.payload["flow_ignore_last"] is not expected_ignore_last:
            raise ValueError(
                "Wenhao replay flow_ignore_last does not match the pinned policy"
            )
        replay_run_config_sha256 = _require_sha256(
            "run_config_sha256", replay_data.payload["run_config_sha256"]
        )
        if replay_run_config_sha256 != RUN_CONFIG_SHA256:
            raise ValueError(
                "Wenhao replay run_config_sha256 does not match the attested checkpoint"
            )

        input_ids = _integer_tensor(
            "input_ids", replay_data.payload["input_ids"], ndim=1
        )
        if input_ids.numel() == 0 or bool((input_ids < 0).any()):
            raise ValueError("Wenhao input_ids must be non-empty and non-negative")
        attention_mask = _integer_or_bool_tensor(
            "attention_mask", replay_data.payload["attention_mask"], ndim=1
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Wenhao attention_mask must match input_ids")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("Wenhao attention_mask values must be zero or one")
        attention_mask = attention_mask.to(dtype=torch.int64)

        image_grid_thw = _integer_tensor(
            "image_grid_thw", replay_data.payload["image_grid_thw"], ndim=2
        )
        if image_grid_thw.shape[0] == 0 or image_grid_thw.shape[1] != 3:
            raise ValueError("Wenhao image_grid_thw must be non-empty [I, 3]")
        if not bool((image_grid_thw > 0).all()):
            raise ValueError("Wenhao image_grid_thw entries must be positive")
        pixel_values = _floating_tensor(
            "pixel_values", replay_data.payload["pixel_values"], ndim=2
        )
        if pixel_values.shape[1] == 0:
            raise ValueError("Wenhao pixel_values feature width must be positive")
        expected_pixel_rows = int(
            image_grid_thw.to(dtype=torch.int64).prod(dim=1).sum().item()
        )
        if pixel_values.shape[0] != expected_pixel_rows:
            raise ValueError(
                "Wenhao pixel_values rows do not match image_grid_thw patch rows"
            )

        effective_present = "effective_image_grid_thw" in replay_data.payload
        pool_present = "visual_pool_factors" in replay_data.payload
        if effective_present != pool_present:
            raise ValueError(
                "Wenhao effective_image_grid_thw and visual_pool_factors must be "
                "present together"
            )

        physical_states = _finite_float32(
            "physical_states",
            replay_data.payload["physical_states"],
            expected_shape=(1, 29),
        )
        latent_chain = _floating_tensor(
            "latent_chain", replay_data.payload["latent_chain"], ndim=3
        ).to(dtype=torch.float32)
        if latent_chain.shape[0] < 2 or tuple(latent_chain.shape[1:]) != (30, 38):
            raise ValueError(
                "Wenhao latent_chain must have shape [N+1, 30, 38] with N >= 1"
            )
        denoise_index = _scalar_integer(
            "denoise_index", replay_data.payload["denoise_index"]
        )
        if not 0 <= denoise_index < latent_chain.shape[0] - 2:
            raise ValueError(
                "Wenhao Flow-PPO excludes the final low-variance denoise transition"
            )

        clipped_actions = _finite_float32(
            "clipped_normalized_actions",
            replay_data.payload["clipped_normalized_actions"],
            expected_shape=(30, 38),
        )
        if not torch.equal(clipped_actions, latent_chain[-1].clamp(-1.0, 1.0)):
            raise ValueError(
                "Wenhao clipped_normalized_actions must exactly equal "
                "clip(latent_chain[-1], -1, 1)"
            )
        wire_actions = _finite_float32(
            "denormalized_wire_actions",
            replay_data.payload["denormalized_wire_actions"],
            expected_shape=(30, 38),
        )
        rtc_prefix_actions = _finite_float32(
            "rtc_prefix_normalized_actions",
            replay_data.payload["rtc_prefix_normalized_actions"],
            expected_shape=(30, 38),
        )
        rtc_prefix_mask = _bool_tensor(
            "rtc_prefix_mask",
            replay_data.payload["rtc_prefix_mask"],
            expected_shape=(30,),
        )
        if torch.any((rtc_prefix_actions < -1.0) | (rtc_prefix_actions > 1.0)):
            raise ValueError(
                "Wenhao RTC prefix normalized actions must remain within [-1, 1]"
            )
        prefix_length = int(rtc_prefix_mask.sum().item())
        if prefix_length > 7:
            raise ValueError("Wenhao RTC prefix length must be in [0, 7]")
        expected_prefix_mask = (
            torch.arange(rtc_prefix_mask.numel(), device=rtc_prefix_mask.device)
            < prefix_length
        )
        if not torch.equal(rtc_prefix_mask, expected_prefix_mask):
            raise ValueError("Wenhao RTC mask must be one contiguous leading prefix")
        if prefix_length and not torch.equal(
            latent_chain[:, :prefix_length],
            rtc_prefix_actions[:prefix_length]
            .unsqueeze(0)
            .expand(latent_chain.shape[0], -1, -1),
        ):
            raise ValueError(
                "Wenhao latent chain changed the deterministic RTC fixed prefix"
            )
        old_element_logprobs = _finite_float32(
            "old_element_logprobs",
            replay_data.payload["old_element_logprobs"],
            expected_shape=(30 * 38,),
        )
        if replay_data.old_logprob is None:
            raise ValueError("Wenhao replay is missing joint chunk old_logprob")
        old_logprob_value = torch.as_tensor(
            replay_data.old_logprob, dtype=torch.float32
        )
        if old_logprob_value.ndim != 0:
            raise ValueError("Wenhao old_logprob must be scalar")
        old_logprob = old_logprob_value.detach().clone().reshape(())
        if not torch.isfinite(old_logprob):
            raise ValueError("Wenhao scalar old_logprob must be finite")
        if not torch.allclose(
            old_element_logprobs.sum(), old_logprob, rtol=1.0e-5, atol=1.0e-5
        ):
            raise ValueError("Wenhao old_logprob must equal sum(old_element_logprobs)")
        humanoid = replay_data.payload["humanoid"]
        if not isinstance(humanoid, Mapping):
            raise TypeError("Wenhao replay humanoid payload must be a mapping")
        require_payload_keys(
            MODEL_FAMILY + ".humanoid", humanoid, ("wenhao_raw_action_rows",)
        )
        recorded_raw_rows = _finite_float32(
            "humanoid.wenhao_raw_action_rows",
            humanoid["wenhao_raw_action_rows"],
            expected_shape=(30, 38),
        )
        if not torch.equal(recorded_raw_rows, wire_actions):
            raise ValueError(
                "Wenhao humanoid.wenhao_raw_action_rows must exactly equal "
                "denormalized_wire_actions"
            )

        if effective_present:
            effective_grid = _integer_tensor(
                "effective_image_grid_thw",
                replay_data.payload["effective_image_grid_thw"],
                ndim=2,
            )
            if effective_grid.shape != image_grid_thw.shape or not bool(
                (effective_grid > 0).all()
            ):
                raise ValueError(
                    "Wenhao effective_image_grid_thw must be positive [I, 3] "
                    "matching image_grid_thw"
                )
            pool_factors = _integer_tensor(
                "visual_pool_factors",
                replay_data.payload["visual_pool_factors"],
                ndim=1,
            )
            if pool_factors.shape[0] != image_grid_thw.shape[0] or not bool(
                (pool_factors > 0).all()
            ):
                raise ValueError(
                    "Wenhao visual_pool_factors must contain one positive entry "
                    "per image"
                )
        else:
            # Wenhao emits no pooling metadata for an uncompressed visual
            # history (notably the first single-image decision), then emits it
            # once BATS starts pooling older frames.  Normalize the optional
            # wire representation to the model's exact identity defaults so
            # one rollout has a stable trainer-input schema across all steps.
            effective_grid = image_grid_thw.clone()
            pool_factors = torch.ones(
                image_grid_thw.shape[0],
                dtype=torch.int64,
                device=image_grid_thw.device,
            )

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "physical_states": physical_states,
            "latent_chain": latent_chain,
            "denoise_indices": torch.tensor(
                denoise_index,
                dtype=torch.int64,
                device=latent_chain.device,
            ),
            # Keep the attested schedule identity in the trainer forward. The
            # model checks it again so a replay row cannot be scored under a
            # different discretization even after collation or transport.
            "flow_schedule_sha256": payload_schedule,
            "clipped_normalized_actions": clipped_actions,
            "denormalized_wire_actions": wire_actions,
            "rtc_prefix_normalized_actions": rtc_prefix_actions,
            "rtc_prefix_mask": rtc_prefix_mask,
            # The trainer retains factors only to audit the full joint sum.
            # Flow-PPO never clips or masks these elements independently.
            "old_element_logprobs": old_element_logprobs,
            "effective_image_grid_thw": effective_grid,
            "visual_pool_factors": pool_factors,
        }
        return model_inputs, old_logprob

    return _build


def get_bundle() -> PolicyBundle:
    """Return Wenhao's parser, model, rollout, and collation hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )


def _bundle_config(run_config: Any) -> Mapping[str, Any]:
    """Return the required Wenhao bundle configuration mapping."""
    bundle_config = run_config.policy.model.bundle_config
    if not isinstance(bundle_config, Mapping):
        raise TypeError("Wenhao policy.model.bundle_config must be a mapping")
    return bundle_config


def _expected_schedule_sha256(run_config: Any) -> str:
    """Read and validate the mandatory behavior schedule identity."""
    bundle_config = _bundle_config(run_config)
    if "expected_schedule_sha256" not in bundle_config:
        raise ValueError(
            "Wenhao policy.model.bundle_config.expected_schedule_sha256 is required"
        )
    return _require_sha256(
        "policy.model.bundle_config.expected_schedule_sha256",
        bundle_config["expected_schedule_sha256"],
    )


def _expected_flow_contract(run_config: Any) -> tuple[float, bool]:
    """Require the pinned RLinf Flow-SDE sampling controls."""
    bundle_config = _bundle_config(run_config)
    noise = bundle_config.get("expected_flow_noise_level")
    ignore_last = bundle_config.get("expected_flow_ignore_last")
    if (
        isinstance(noise, bool)
        or not isinstance(noise, (int, float))
        or float(noise) != WENHAO_FLOW_NOISE_LEVEL
    ):
        raise ValueError(
            "Wenhao policy.model.bundle_config.expected_flow_noise_level "
            f"must be {WENHAO_FLOW_NOISE_LEVEL}"
        )
    if ignore_last is not WENHAO_FLOW_IGNORE_LAST:
        raise ValueError(
            "Wenhao policy.model.bundle_config.expected_flow_ignore_last must be true"
        )
    return float(noise), ignore_last


def _attested_pad_token_id(run_config: Any) -> int:
    """Read Qwen's pad token below the single resolved model root.

    The model path is the only checkpoint-location authority.  Requiring the
    exact ``models/<MODEL_ID>`` layout prevents an independently configured
    metadata root from silently describing a different checkpoint.
    """
    raw_model_path = Path(str(run_config.policy.model.path)).expanduser()
    try:
        model_root = raw_model_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Wenhao policy.model.path does not exist: {raw_model_path}"
        ) from error
    if (
        not model_root.is_dir()
        or model_root.name != MODEL_ID
        or model_root.parent.name != "models"
    ):
        raise ValueError(
            f"Wenhao policy.model.path must resolve to .../models/{MODEL_ID}"
        )
    path = model_root / "base_vlm" / "generation_config.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Wenhao generation metadata is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("pad_token_id") != 151643:
        raise ValueError("Wenhao base-VLM pad token identity changed")
    return 151643


def _require_sha256(name: str, value: Any) -> str:
    """Require one lowercase SHA-256 digest."""
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"Wenhao {name} must be a lowercase SHA-256 digest")
    return digest


def _floating_tensor(name: str, value: Any, *, ndim: int) -> torch.Tensor:
    """Clone one finite floating tensor with the requested rank."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != ndim:
        raise ValueError(f"Wenhao {name} must have rank {ndim}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"Wenhao {name} must use a floating-point dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Wenhao {name} contains non-finite values")
    return tensor.detach().clone().contiguous()


def _finite_float32(
    name: str,
    value: Any,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Clone one finite tensor as float32 with an exact shape."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Wenhao {name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Wenhao {name} contains non-finite values")
    return tensor.detach().clone().contiguous()


def _integer_tensor(name: str, value: Any, *, ndim: int) -> torch.Tensor:
    """Clone one integer tensor as int64 with the requested rank."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != ndim:
        raise ValueError(f"Wenhao {name} must have rank {ndim}")
    if tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"Wenhao {name} must use an integer dtype")
    return tensor.to(dtype=torch.int64).detach().clone().contiguous()


def _integer_or_bool_tensor(name: str, value: Any, *, ndim: int) -> torch.Tensor:
    """Clone one integer or boolean tensor with the requested rank."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != ndim:
        raise ValueError(f"Wenhao {name} must have rank {ndim}")
    if tensor.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"Wenhao {name} must use a bool or integer dtype")
    return tensor.detach().clone().contiguous()


def _scalar_integer(name: str, value: Any) -> int:
    """Return one scalar integer without accepting bool or float coercion."""
    tensor = torch.as_tensor(value)
    if tensor.ndim != 0:
        raise ValueError(f"Wenhao {name} must be scalar")
    if tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"Wenhao {name} must use an integer dtype")
    return int(tensor.item())


def _bool_tensor(
    name: str,
    value: Any,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Clone one exact-shape boolean tensor."""
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Wenhao {name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bool:
        raise TypeError(f"Wenhao {name} must have dtype bool")
    return tensor.detach().clone().contiguous()


class _CheckpointableNoOpTokenizer:
    """Persist Cosmos's non-text tokenizer without inventing text semantics."""

    def __init__(self, tokenizer: Any) -> None:
        """Wrap one Cosmos no-op tokenizer."""
        object.__setattr__(self, "_tokenizer", tokenizer)

    def __getattr__(self, name: str) -> Any:
        """Delegate tokenizer-shaped reads to Cosmos's implementation."""
        return getattr(self._tokenizer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate tokenizer-shaped writes to Cosmos's implementation."""
        setattr(self._tokenizer, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate no-op tokenization calls."""
        return self._tokenizer(*args, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs: Any,
    ) -> tuple[str]:
        """Write a deterministic marker required by Cosmos checkpoint export."""
        del kwargs
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "g1_wenhao_vla_no_op_tokenizer.json"
        marker.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "tokenizer_type": "alpagym_g1_wenhao_vla_no_op",
                    "pad_token_id": int(self.pad_token_id),
                    "eos_token_id": int(self.eos_token_id),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return (str(marker),)
