from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from alpagym_g1_mjlab.model import (
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    register_g1_mjlab_model,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData

from alpagym_g1_videomimic_planner._tensor_conversion import as_float32_tensor
from alpagym_g1_videomimic_planner.bundle import (
    build_model_inputs,
    export_model_checkpoint,
    load_inference_model,
)
from alpagym_g1_videomimic_planner.export import initialize_actor_from_v9_checkpoint
from alpagym_g1_videomimic_planner.model import (
    REPLAY_SCHEMA,
    SHADOW_ACTION_STEPS,
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    OBS_DIMS,
    OBS_KEYS,
    register_planner_model,
)


def _observations(*, batch_size: int | None = None) -> dict[str, torch.Tensor]:
    prefix = (
        (SHADOW_ACTION_STEPS,)
        if batch_size is None
        else (batch_size, SHADOW_ACTION_STEPS)
    )
    return {
        key: torch.randn(*prefix, OBS_DIMS[key], dtype=torch.float32)
        for key in OBS_KEYS
    }


def _planner() -> G1VideoMimicPlannerActorCriticModel:
    torch.manual_seed(11)
    register_planner_model()
    return G1VideoMimicPlannerActorCriticModel(
        G1VideoMimicPlannerConfig(hidden_dims=[16, 8], init_std=0.3)
    )


def _receipt(
    digest: str,
    *,
    active_digest: str | None = None,
    duration: int = 1,
) -> dict[str, object]:
    """Build the minimum complete one-tick realization receipt."""
    return {
        "reference_id": 1,
        "source_decision_id": 0,
        "root_z_alignment_offset_m": 0.125,
        "feedback_trace": {
            "env_id": 0,
            "source_decision_id": 0,
            "ticks": [
                {
                    "control_tick_offset": index + 1,
                    "reference_action_index": index,
                    "active_reference_id": 1,
                    "active_reference_sha256": active_digest or digest,
                    "applied_reference_sha256": digest,
                    "root_z_alignment_offset_m": 0.125,
                    "reward": 0.5,
                    "control_episode_step": index,
                }
                for index in range(duration)
            ],
        },
        "transition": {
            "env_id": 0,
            "source_decision_id": 0,
            "reference_id": 1,
            "reference_sha256": digest,
            "duration_ticks": duration,
            "primitive_rewards": torch.tensor(
                [0.5 if index < duration else 0.0 for index in range(5)]
            ),
            "primitive_reward_mask": torch.tensor(
                [index < duration for index in range(5)]
            ),
        },
    }


def test_read_only_numpy_conversion_is_warning_free_and_does_not_alias() -> None:
    source = np.arange(6, dtype=np.float32)
    source.flags.writeable = False

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        tensor = as_float32_tensor(source)

    assert tensor.data_ptr() != source.ctypes.data
    source.flags.writeable = True
    source[0] = 100.0
    assert tensor[0].item() == 0.0


def test_read_only_numpy_conversion_preserves_device_dtype_and_values() -> None:
    source = np.arange(4, dtype=np.float64)
    source.flags.writeable = False

    tensor = as_float32_tensor(source, device=torch.device("cpu"))

    assert tensor.dtype is torch.float32
    assert tensor.device.type == "cpu"
    torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32))


def test_planner_native_safetensors_export_round_trips_actor_and_critic(
    tmp_path: Path,
) -> None:
    """The non-generative export is directly loadable without generation files."""
    planner = _planner()
    with torch.no_grad():
        planner.std.add_(0.125)
        planner.actor[0].weight.add_(0.25)
        planner.critic[0].bias.sub_(0.5)

    export_model_checkpoint(planner, tmp_path)

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["checkpoint_path"] == "model.safetensors"
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "generation_config.json").exists()
    run_config = SimpleNamespace(
        policy=SimpleNamespace(model=SimpleNamespace(path=str(tmp_path)))
    )
    loaded = load_inference_model(
        run_config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).model

    expected = planner.state_dict()
    actual = loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_planner_forward_preserves_every_shadow_step_logprob() -> None:
    model = _planner()
    observations = _observations(batch_size=2)
    actions = torch.randn(2, SHADOW_ACTION_STEPS, 23)

    result = model(actions=actions, **observations)

    assert result["token_log_probs"].shape == (2, SHADOW_ACTION_STEPS)
    assert result["log_probs"].shape == (2,)
    assert result["values"].shape == (2,)
    torch.testing.assert_close(
        result["log_probs"],
        result["token_log_probs"].sum(dim=-1),
    )


def test_replay_teacher_forcing_reproduces_old_token_logprobs_exactly() -> None:
    model = _planner()
    observations = _observations(batch_size=1)
    actions = torch.randn(1, SHADOW_ACTION_STEPS, 23)
    with torch.no_grad():
        rollout_result = model(actions=actions, **observations)
    token_logprobs = rollout_result["token_log_probs"][0]
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_videomimic_planner",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=token_logprobs.sum(),
        payload={
            "shadow_observations": {
                key: value[0] for key, value in observations.items()
            },
            "raw_actions": actions[0],
            "executed_actions": actions[0].clamp(-8.0, 8.0),
            "old_token_logprobs": token_logprobs,
            "reference_sha256": "a" * 64,
            **_receipt("a" * 64),
        },
    )

    model_inputs, old_logprob = build_model_inputs(SimpleNamespace())(replay)
    old_tokens = model_inputs.pop("old_token_logprobs")
    batched_inputs = {key: value.unsqueeze(0) for key, value in model_inputs.items()}
    with torch.no_grad():
        replay_result = model(**batched_inputs)

    torch.testing.assert_close(replay_result["token_log_probs"][0], old_tokens)
    torch.testing.assert_close(replay_result["log_probs"][0], old_logprob)


@pytest.mark.parametrize(("duration", "causal_tokens"), ((1, 45), (5, 49)))
def test_replay_derives_early_terminal_token_causality_mask(
    duration: int,
    causal_tokens: int,
) -> None:
    digest = "a" * 64
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_videomimic_planner",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.zeros(()),
        payload={
            "shadow_observations": _observations(),
            "raw_actions": torch.zeros(SHADOW_ACTION_STEPS, 23),
            "executed_actions": torch.zeros(SHADOW_ACTION_STEPS, 23),
            "old_token_logprobs": torch.zeros(SHADOW_ACTION_STEPS),
            "reference_sha256": digest,
            **_receipt(digest, duration=duration),
        },
    )

    model_inputs, _ = build_model_inputs(SimpleNamespace())(replay)

    mask = model_inputs["token_causality_mask"]
    assert mask.dtype is torch.bool
    assert int(mask.sum().item()) == causal_tokens
    assert bool(mask[:causal_tokens].all())
    assert not bool(mask[causal_tokens:].any())


def test_replay_rejects_hash_mismatch_and_noncanonical_actions() -> None:
    observations = _observations()
    raw_actions = torch.full((SHADOW_ACTION_STEPS, 23), 9.0)
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_videomimic_planner",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.zeros(()),
        payload={
            "shadow_observations": observations,
            "raw_actions": raw_actions,
            "executed_actions": raw_actions,
            "old_token_logprobs": torch.zeros(SHADOW_ACTION_STEPS),
            "reference_sha256": "b" * 64,
            **_receipt("b" * 64, active_digest="c" * 64),
        },
    )
    with pytest.raises(ValueError, match="executed_actions"):
        build_model_inputs(SimpleNamespace())(replay)

    payload = dict(replay.payload)
    payload["executed_actions"] = raw_actions.clamp(-8.0, 8.0)
    replay = PolicyReplayData(
        replay_schema_version=replay.replay_schema_version,
        payload_schema=replay.payload_schema,
        payload_schema_version=replay.payload_schema_version,
        model_family=replay.model_family,
        action_selection=replay.action_selection,
        old_logprob=replay.old_logprob,
        payload=payload,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_model_inputs(SimpleNamespace())(replay)


def test_v9_export_copies_actor_but_zero_initializes_planner_critic() -> None:
    torch.manual_seed(7)
    register_g1_mjlab_model()
    direct = G1MjlabActorCriticModel(G1MjlabConfig(hidden_dims=[16, 8], init_std=0.4))
    planner = _planner()

    initialize_actor_from_v9_checkpoint(
        planner,
        {"model_state_dict": direct.state_dict()},
    )

    for actual, expected in zip(planner.actor.parameters(), direct.actor.parameters()):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        planner.actor_terrain.weight, direct.actor_terrain.weight
    )
    torch.testing.assert_close(planner.actor_attention, direct.actor_attention)
    torch.testing.assert_close(planner.std, direct.std)
    first_observation = {
        key: value[:, 0] for key, value in _observations(batch_size=3).items()
    }
    with torch.no_grad():
        values = planner._head(first_observation, actor=False).squeeze(-1)
    torch.testing.assert_close(values, torch.zeros_like(values))
