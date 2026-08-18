# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaGym policy bundle and strict planner replay parser."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import torch
from alpagym_runtime.policies.registry import PolicyBundle
from alpagym_runtime.replay import PolicyReplayData, require_payload_keys
from safetensors.torch import save_file

from alpagym_g1_videomimic_planner.inference_model import (
    G1VideoMimicPlannerInferenceModel,
)
from alpagym_g1_videomimic_planner.model import (
    REPLAY_SCHEMA,
    SHADOW_ACTION_STEPS,
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    OBS_DIMS,
    OBS_KEYS,
    register_planner_model,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def setup_tokenizer(config: Any) -> Any:
    """Register the planner model and return Cosmos's no-op tokenizer."""
    del config
    register_planner_model()
    return _no_op_tokenizer()


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the generic replay packer with the planner's strict parser."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    register_planner_model()
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
    )


def install_runtime_bridge() -> None:
    """Install model registrations before Cosmos constructs its workers."""
    register_planner_model()


def load_inference_model(
    run_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> G1VideoMimicPlannerInferenceModel:
    """Load one exported VideoMimic planner checkpoint."""
    if dtype != torch.float32:
        raise ValueError(f"VideoMimic planner expects dtype=float32, got {dtype!r}")
    register_planner_model()
    bundle_dir = Path(run_config.policy.model.path)
    config = G1VideoMimicPlannerConfig.from_pretrained(bundle_dir)
    # The inherited factory is typed to its base class even though it constructs
    # ``cls``; preserve the concrete planner type at this boundary.
    model = cast(
        G1VideoMimicPlannerActorCriticModel,
        G1VideoMimicPlannerActorCriticModel.from_pretrained(config, str(bundle_dir)),
    )
    model.to(device=device, dtype=torch.float32)
    model.load_hf_weights(str(bundle_dir), parallel_dims=None, device=device)
    model.eval()
    return G1VideoMimicPlannerInferenceModel(model)


def build_model_inputs(
    run_config: Any,
) -> Callable[[PolicyReplayData], tuple[dict[str, Any], torch.Tensor]]:
    """Return a fail-closed parser for one macro-decision replay row."""
    del run_config

    def _build(replay_data: PolicyReplayData) -> tuple[dict[str, Any], torch.Tensor]:
        if replay_data.payload_schema != REPLAY_SCHEMA:
            raise ValueError(
                f"payload_schema={replay_data.payload_schema!r}; expected {REPLAY_SCHEMA!r}"
            )
        if replay_data.model_family != "g1_videomimic_planner":
            raise ValueError(
                "VideoMimic planner replay has unexpected model_family="
                f"{replay_data.model_family!r}"
            )
        require_payload_keys(
            "g1_videomimic_planner",
            replay_data.payload,
            (
                "shadow_observations",
                "raw_actions",
                "executed_actions",
                "old_token_logprobs",
                "reference_id",
                "source_decision_id",
                "reference_sha256",
                "root_z_alignment_offset_m",
                "feedback_trace",
                "transition",
            ),
        )
        observations = replay_data.payload["shadow_observations"]
        if not isinstance(observations, Mapping):
            raise TypeError("planner shadow_observations must be a mapping")
        require_payload_keys("planner.shadow_observations", observations, OBS_KEYS)
        model_inputs = {
            key: _finite_tensor(
                f"shadow_observations.{key}",
                observations[key],
                expected_shape=(SHADOW_ACTION_STEPS, OBS_DIMS[key]),
            )
            for key in OBS_KEYS
        }
        raw_actions = _finite_tensor(
            "raw_actions",
            replay_data.payload["raw_actions"],
            expected_shape=(SHADOW_ACTION_STEPS, 23),
        )
        executed_actions = _finite_tensor(
            "executed_actions",
            replay_data.payload["executed_actions"],
            expected_shape=(SHADOW_ACTION_STEPS, 23),
        )
        if not torch.equal(executed_actions, raw_actions.clamp(min=-8.0, max=8.0)):
            raise ValueError(
                "executed_actions must equal clip(raw_actions, -8, 8) exactly"
            )
        old_token_logprobs = _finite_tensor(
            "old_token_logprobs",
            replay_data.payload["old_token_logprobs"],
            expected_shape=(SHADOW_ACTION_STEPS,),
        )
        if replay_data.old_logprob is None:
            raise ValueError("planner replay is missing scalar old_logprob")
        old_logprob = torch.as_tensor(
            replay_data.old_logprob, dtype=torch.float32
        ).reshape(())
        if not torch.isfinite(old_logprob):
            raise ValueError("planner old_logprob must be finite")
        if not torch.allclose(
            old_token_logprobs.sum(),
            old_logprob,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(
                "planner scalar old_logprob must equal sum(old_token_logprobs)"
            )

        reference_sha256 = _require_sha256(
            "reference_sha256", replay_data.payload["reference_sha256"]
        )
        transition = replay_data.payload["transition"]
        if not isinstance(transition, Mapping):
            raise TypeError("planner replay transition must be a mapping")
        duration = _validate_feedback_receipt(
            replay_data.payload,
            transition=transition,
            expected_reference_sha256=reference_sha256,
        )

        model_inputs["actions"] = raw_actions
        model_inputs["old_token_logprobs"] = old_token_logprobs
        # Shadow reference frames consume actions at offsets [0, 5, ..., 45].
        # For a realized d-tick prefix only token indices < 44+d can have
        # affected that macro transition's reward.
        model_inputs["token_causality_mask"] = (
            torch.arange(SHADOW_ACTION_STEPS, dtype=torch.int64) < 44 + duration
        )
        return model_inputs, old_logprob

    return _build


def export_model_checkpoint(model: torch.nn.Module, output_dir: Path) -> None:
    """Write a directly loadable non-generative planner safetensors bundle."""
    if not isinstance(model, G1VideoMimicPlannerActorCriticModel):
        raise TypeError(
            "planner export expected G1VideoMimicPlannerActorCriticModel, got "
            f"{type(model).__name__}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.config.to_dict()
    config["checkpoint_path"] = "model.safetensors"
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_dict = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state_dict, str(output_dir / "model.safetensors"))


def get_bundle() -> PolicyBundle:
    """Return all VideoMimic planner policy hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
        export_model_checkpoint=export_model_checkpoint,
    )


def _finite_tensor(
    name: str, value: Any, *, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"planner {name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"planner {name} contains non-finite values")
    return tensor


def _require_sha256(name: str, value: Any) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"planner {name} must be a lowercase SHA-256 digest")
    return digest


def _validate_feedback_receipt(
    payload: Mapping[str, Any],
    *,
    transition: Mapping[str, Any],
    expected_reference_sha256: str,
) -> int:
    """Require every realized controller tick to credit the emitted reference."""
    raw_trace = payload["feedback_trace"]
    if not isinstance(raw_trace, Mapping):
        raise TypeError("planner feedback_trace must be a mapping")
    reference_id = int(payload["reference_id"])
    source_decision_id = int(payload["source_decision_id"])
    if (
        reference_id <= 0
        or int(raw_trace.get("source_decision_id", -1)) != source_decision_id
    ):
        raise ValueError("planner feedback trace identity does not match source plan")
    if int(raw_trace.get("env_id", -1)) != int(transition.get("env_id", -2)):
        raise ValueError("planner feedback env_id does not match transition")
    if int(transition.get("source_decision_id", -1)) != source_decision_id:
        raise ValueError("planner transition source_decision_id does not match plan")
    if int(transition.get("reference_id", -1)) != reference_id:
        raise ValueError("planner transition reference_id does not match plan")
    if str(transition.get("reference_sha256", "")) != expected_reference_sha256:
        raise ValueError("planner transition reference_sha256 does not match plan")
    root_z_offset = float(payload["root_z_alignment_offset_m"])
    if not torch.isfinite(torch.tensor(root_z_offset)):
        raise ValueError("planner root_z_alignment_offset_m must be finite")
    ticks = raw_trace.get("ticks")
    if not isinstance(ticks, (list, tuple)) or not ticks:
        raise ValueError("planner feedback_trace.ticks must be a non-empty sequence")
    duration = int(transition.get("duration_ticks", -1))
    if duration != len(ticks) or not 1 <= duration <= 5:
        raise ValueError(
            "planner duration_ticks must equal the feedback K-prefix length"
        )
    primitive_rewards = torch.as_tensor(
        transition.get("primitive_rewards"), dtype=torch.float32
    )
    primitive_mask = torch.as_tensor(
        transition.get("primitive_reward_mask"), dtype=torch.bool
    )
    if primitive_rewards.shape != (5,) or primitive_mask.shape != (5,):
        raise ValueError("planner primitive reward receipt must have shape (5,)")
    expected_mask = torch.arange(5) < duration
    if not torch.equal(primitive_mask, expected_mask):
        raise ValueError("planner primitive reward mask must be a contiguous K-prefix")
    previous_control_step: int | None = None
    for index, tick in enumerate(ticks):
        if not isinstance(tick, Mapping):
            raise TypeError(f"planner feedback tick {index} must be a mapping")
        active = _require_sha256(
            f"feedback_trace.ticks[{index}].active_reference_sha256",
            tick.get("active_reference_sha256"),
        )
        _require_sha256(
            f"feedback_trace.ticks[{index}].applied_reference_sha256",
            tick.get("applied_reference_sha256"),
        )
        if active != expected_reference_sha256:
            raise ValueError(
                "AlpaSim active reference SHA256 does not match the planner reference"
            )
        if int(tick.get("control_tick_offset", -1)) != index + 1:
            raise ValueError("planner feedback control_tick_offset is not 1..K")
        if int(tick.get("reference_action_index", -1)) != index:
            raise ValueError("planner feedback reference_action_index is not 0..K-1")
        if int(tick.get("active_reference_id", -1)) != reference_id:
            raise ValueError("planner feedback active_reference_id does not match plan")
        if not torch.isclose(
            torch.tensor(float(tick.get("root_z_alignment_offset_m", float("nan")))),
            torch.tensor(root_z_offset),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("planner feedback root Z alignment does not match plan")
        reward = float(tick.get("reward", float("nan")))
        if not torch.isfinite(torch.tensor(reward)) or reward != float(
            primitive_rewards[index]
        ):
            raise ValueError(
                "planner feedback reward does not match primitive reward receipt"
            )
        control_step = int(tick.get("control_episode_step", -1))
        if (
            previous_control_step is not None
            and control_step != previous_control_step + 1
        ):
            raise ValueError("planner feedback control_episode_step is not contiguous")
        previous_control_step = control_step
    return duration


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
