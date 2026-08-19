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
    GRAIL_FUTURE_REFERENCE_OFFSET,
    PLANNER_MODE_SHADOW_ROLLOUT,
    REPLAY_SCHEMA,
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    OBS_DIMS,
    OBS_KEYS,
    planner_mode_contract,
    register_planner_model,
)
from alpagym_g1_videomimic_planner.provenance import (
    ACTOR_STATE_HASH_SCHEMA,
    ACTOR_STATE_LINEAGE_FIELDS,
    canonical_actor_state_sha256,
    file_sha256,
    trained_actor_lineage,
    training_export_step,
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
    planner_mode = _planner_mode(run_config)
    contract = planner_mode_contract(planner_mode)
    if int(config.shadow_action_steps) != contract.actor_steps:
        raise ValueError(
            "planner checkpoint shadow_action_steps does not match planner_mode: "
            f"expected {contract.actor_steps}, got {config.shadow_action_steps}"
        )
    # The inherited factory is typed to its base class even though it constructs
    # ``cls``; preserve the concrete planner type at this boundary.
    model = cast(
        G1VideoMimicPlannerActorCriticModel,
        G1VideoMimicPlannerActorCriticModel.from_pretrained(config, str(bundle_dir)),
    )
    model.to(device=device, dtype=torch.float32)
    model.load_hf_weights(str(bundle_dir), parallel_dims=None, device=device)
    _validate_loaded_actor_state(model)
    model.eval()
    return G1VideoMimicPlannerInferenceModel(model)


def build_model_inputs(
    run_config: Any,
) -> Callable[[PolicyReplayData], tuple[dict[str, Any], torch.Tensor]]:
    """Return a fail-closed parser for one macro-decision replay row."""
    planner_mode = _planner_mode(run_config)
    contract = planner_mode_contract(
        planner_mode,
        macro_period_us=_policy_step_dt_us(run_config),
    )

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
        if any(
            key in replay_data.payload
            for key in (
                "completion_policy_sha256",
                "actor_bundle_sha256",
                "actor_source_v9_checkpoint_sha256",
            )
        ):
            raise ValueError(
                "current-policy planner replay must not identify a frozen completion"
            )
        observations = replay_data.payload["shadow_observations"]
        if not isinstance(observations, Mapping):
            raise TypeError("planner shadow_observations must be a mapping")
        require_payload_keys("planner.shadow_observations", observations, OBS_KEYS)
        model_inputs = {
            key: _finite_tensor(
                f"shadow_observations.{key}",
                observations[key],
                expected_shape=(contract.actor_steps, OBS_DIMS[key]),
            )
            for key in OBS_KEYS
        }
        raw_actions = _finite_tensor(
            "raw_actions",
            replay_data.payload["raw_actions"],
            expected_shape=(contract.actor_steps, 23),
        )
        executed_actions = _finite_tensor(
            "executed_actions",
            replay_data.payload["executed_actions"],
            expected_shape=(contract.actor_steps, 23),
        )
        if not torch.equal(executed_actions, raw_actions.clamp(min=-8.0, max=8.0)):
            raise ValueError(
                "executed_actions must equal clip(raw_actions, -8, 8) exactly"
            )
        old_token_logprobs = _finite_tensor(
            "old_token_logprobs",
            replay_data.payload["old_token_logprobs"],
            expected_shape=(contract.actor_steps,),
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
            controller_ticks=contract.controller_ticks,
            require_unmodified_applied_reference=True,
        )

        model_inputs["actions"] = raw_actions
        model_inputs["old_token_logprobs"] = old_token_logprobs
        # At controller tick j, GRAIL's farthest future reference is j+45.
        # Therefore a realized d-tick prefix can depend on the first 44+d
        # current-policy actions. A full K25 transition credits all 69 H70
        # actions; an early terminal excludes only the causally unseen tail.
        model_inputs["token_causality_mask"] = (
            torch.arange(contract.actor_steps, dtype=torch.int64)
            < GRAIL_FUTURE_REFERENCE_OFFSET - 1 + duration
        )
        return model_inputs, old_logprob

    return _build


def export_model_checkpoint(model: torch.nn.Module, output_dir: Path) -> None:
    """Write a loadable checkpoint with explicit current-actor provenance."""
    if not isinstance(model, G1VideoMimicPlannerActorCriticModel):
        raise TypeError(
            "planner export expected G1VideoMimicPlannerActorCriticModel, got "
            f"{type(model).__name__}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.config.to_dict()
    config["checkpoint_path"] = "model.safetensors"
    state_dict = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    current_actor_state_sha256 = canonical_actor_state_sha256(
        model,
        state_dict=state_dict,
    )
    weights_path = output_dir / "model.safetensors"
    temporary_weights_path = output_dir / "model.safetensors.tmp"
    temporary_weights_path.unlink(missing_ok=True)
    save_file(state_dict, str(temporary_weights_path))

    initialization = config.get("actor_initialization_attestation")
    previous_lineage = config.get("actor_update_lineage")
    if (initialization is None) != (previous_lineage is None):
        raise ValueError(
            "planner actor provenance must contain both initialization and lineage"
        )
    if initialization is not None:
        if not isinstance(initialization, Mapping) or not isinstance(
            previous_lineage, Mapping
        ):
            raise TypeError("planner actor provenance fields must be mappings")
        export_step = training_export_step(output_dir)
        previous_step = previous_lineage.get("training_export_step")
        current_weights_sha256 = file_sha256(temporary_weights_path)
        if previous_step == export_step:
            if (
                previous_lineage.get("current_model_weights_sha256")
                != current_weights_sha256
            ):
                temporary_weights_path.unlink(missing_ok=True)
                raise ValueError(
                    "re-exporting one training step with different policy weights "
                    "would corrupt actor lineage"
                )
            actor_extension_presence = tuple(
                name in previous_lineage for name in ACTOR_STATE_LINEAGE_FIELDS
            )
            if any(actor_extension_presence) and not all(actor_extension_presence):
                temporary_weights_path.unlink(missing_ok=True)
                raise ValueError("actor state lineage extension is incomplete")
            if all(actor_extension_presence) and (
                previous_lineage["current_actor_state_sha256"]
                != current_actor_state_sha256
            ):
                temporary_weights_path.unlink(missing_ok=True)
                raise ValueError(
                    "re-exporting one training step with different actor state "
                    "would corrupt actor lineage"
                )
            # A repeated final-save hook is idempotent: keep the existing parent
            # instead of manufacturing a self-parent edge.
            lineage = dict(previous_lineage)
        else:
            if (
                not isinstance(previous_step, int)
                or isinstance(previous_step, bool)
                or previous_step != export_step - 1
            ):
                temporary_weights_path.unlink(missing_ok=True)
                raise ValueError(
                    "policy-native checkpoints must be exported in contiguous "
                    f"steps: parent={previous_step!r}, export={export_step}"
                )
            lineage = trained_actor_lineage(
                initialization,
                previous_lineage,
                current_model_weights_sha256=current_weights_sha256,
                training_export_step=export_step,
                current_actor_state_sha256=current_actor_state_sha256,
            )
        config["actor_update_lineage"] = lineage
        # Preserve the chain if this process later writes another checkpoint.
        model.config.actor_update_lineage = lineage

    temporary_weights_path.replace(weights_path)

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _validate_loaded_actor_state(
    model: G1VideoMimicPlannerActorCriticModel,
) -> None:
    """Verify the optional actor-only provenance against loaded parameters."""

    lineage = getattr(model.config, "actor_update_lineage", None)
    if not isinstance(lineage, Mapping):
        return
    presence = tuple(name in lineage for name in ACTOR_STATE_LINEAGE_FIELDS)
    if not any(presence):
        return
    if not all(presence):
        raise ValueError("planner actor state lineage extension is incomplete")
    if lineage["actor_state_hash_schema"] != ACTOR_STATE_HASH_SCHEMA:
        raise ValueError("planner actor state hash schema is invalid")
    expected = _require_sha256(
        "current_actor_state_sha256",
        lineage["current_actor_state_sha256"],
    )
    actual = canonical_actor_state_sha256(model)
    if actual != expected:
        raise ValueError("planner actor state differs from actor-only provenance")


def _validate_feedback_receipt(
    payload: Mapping[str, Any],
    *,
    transition: Mapping[str, Any],
    expected_reference_sha256: str,
    controller_ticks: int,
    require_unmodified_applied_reference: bool,
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
    if duration != len(ticks) or not 1 <= duration <= controller_ticks:
        raise ValueError(
            "planner duration_ticks must equal the feedback K-prefix length"
        )
    primitive_rewards = torch.as_tensor(
        transition.get("primitive_rewards"), dtype=torch.float32
    )
    primitive_mask = torch.as_tensor(
        transition.get("primitive_reward_mask"), dtype=torch.bool
    )
    expected_shape = (controller_ticks,)
    if (
        primitive_rewards.shape != expected_shape
        or primitive_mask.shape != expected_shape
    ):
        raise ValueError(
            f"planner primitive reward receipt must have shape {expected_shape}"
        )
    expected_mask = torch.arange(controller_ticks) < duration
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
        applied = _require_sha256(
            f"feedback_trace.ticks[{index}].applied_reference_sha256",
            tick.get("applied_reference_sha256"),
        )
        if active != expected_reference_sha256:
            raise ValueError(
                "AlpaSim active reference SHA256 does not match the planner reference"
            )
        if require_unmodified_applied_reference and applied != active:
            raise ValueError(
                "current-policy H70 reference was modified before controller application"
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


def _planner_mode(run_config: Any) -> str:
    """Read the fixed H69/K25 current-policy rollout mode."""

    try:
        bundle_config = run_config.policy.model.bundle_config
    except AttributeError:
        return PLANNER_MODE_SHADOW_ROLLOUT
    if bundle_config is None:
        return PLANNER_MODE_SHADOW_ROLLOUT
    return str(dict(bundle_config).get("planner_mode", PLANNER_MODE_SHADOW_ROLLOUT))


def _policy_step_dt_us(run_config: Any) -> int:
    """Read the outer execution period, retaining K25 for unit fixtures."""

    try:
        return int(run_config.policy.model.step_dt_us)
    except AttributeError:
        return 500_000


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
