from __future__ import annotations

import json
import hashlib
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
    CONTROLLER_TICKS_PER_MACRO,
    PLANNER_MODE_SHADOW_ROLLOUT,
    REPLAY_SCHEMA,
    SHADOW_ACTION_STEPS,
    G1VideoMimicPlannerActorCriticModel,
    G1VideoMimicPlannerConfig,
    OBS_DIMS,
    OBS_KEYS,
    planner_mode_contract,
    register_planner_model,
)
from alpagym_g1_videomimic_planner.provenance import model_weights_sha256


def _humanoid_policy_modules():
    """Install test gRPC stubs before importing the humanoid policy facade."""
    from alpagym_runtime.alpasim.tests.test_proto_conversion import (
        install_alpasim_grpc_stubs,
    )

    install_alpasim_grpc_stubs()
    from alpagym_runtime.alpasim import humanoid_policy_server

    from alpagym_g1_videomimic_planner import humanoid_policy

    return humanoid_policy, humanoid_policy_server


def _observations(
    *,
    batch_size: int | None = None,
    actor_steps: int = SHADOW_ACTION_STEPS,
) -> dict[str, torch.Tensor]:
    prefix = (actor_steps,) if batch_size is None else (batch_size, actor_steps)
    return {
        key: torch.randn(*prefix, OBS_DIMS[key], dtype=torch.float32)
        for key in OBS_KEYS
    }


def _planner(
    *, actor_steps: int = SHADOW_ACTION_STEPS
) -> G1VideoMimicPlannerActorCriticModel:
    torch.manual_seed(11)
    register_planner_model()
    return G1VideoMimicPlannerActorCriticModel(
        G1VideoMimicPlannerConfig(
            hidden_dims=[16, 8],
            init_std=0.3,
            shadow_action_steps=actor_steps,
        )
    )


def _receipt(
    digest: str,
    *,
    active_digest: str | None = None,
    applied_digest: str | None = None,
    duration: int = 1,
    controller_ticks: int = CONTROLLER_TICKS_PER_MACRO,
) -> dict[str, object]:
    """Build one complete realized macro-transition receipt."""
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
                    "applied_reference_sha256": applied_digest or digest,
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
                [0.5 if index < duration else 0.0 for index in range(controller_ticks)]
            ),
            "primitive_reward_mask": torch.tensor(
                [index < duration for index in range(controller_ticks)]
            ),
        },
    }


def _run_config(*, planner_mode: str, step_dt_us: int = 500_000) -> SimpleNamespace:
    """Build the policy-owned mode selector consumed by the replay parser."""
    return SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                bundle_config={"planner_mode": planner_mode},
                step_dt_us=step_dt_us,
            )
        )
    )


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


@pytest.mark.parametrize("bootstrap_requested", (False, True))
def test_current_h70_finalize_uses_pure_critic_without_replanning(
    bootstrap_requested: bool,
) -> None:
    """Terminal prefixes stop; truncated prefixes evaluate V without a new H70."""
    humanoid_policy, _ = _humanoid_policy_modules()

    class _InferenceEngine:
        def __init__(self, model: G1VideoMimicPlannerActorCriticModel) -> None:
            self.model = model

        def get_model_for_session(
            self, session_uuid: str
        ) -> G1VideoMimicPlannerActorCriticModel:
            del session_uuid
            return self.model

    class _CurrentH70Facade:
        def __init__(self) -> None:
            self.feedback_count = 0
            self.critic_calls = 0

        def update_many(self, observations: tuple[object, ...]) -> None:
            self.feedback_count += len(observations)

        def critic_observation(self, observation: object) -> dict[str, torch.Tensor]:
            del observation
            self.critic_calls += 1
            return {
                key: torch.zeros(OBS_DIMS[key], dtype=torch.float32) for key in OBS_KEYS
            }

        def plan(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("finalize must not create another reference")

    model = _planner()
    facade = _CurrentH70Facade()
    policy = object.__new__(humanoid_policy.G1VideoMimicPlannerHumanoidPolicy)
    policy._inference_engine = _InferenceEngine(model)
    policy._session_uuid = "session"
    policy._request = SimpleNamespace(
        joint_names=tuple(f"joint_{i}" for i in range(29))
    )
    policy._device = torch.device("cpu")
    policy._planner_mode = PLANNER_MODE_SHADOW_ROLLOUT
    policy._actor_steps_per_plan = SHADOW_ACTION_STEPS
    policy._support = SimpleNamespace(
        PolicyObservation=lambda **kwargs: SimpleNamespace(**kwargs),
        RobotKinematicState=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    policy._lanes = {
        0: humanoid_policy._Lane(
            policy=facade,
            root_z_alignment_offset_m=0.0,
            generator=torch.Generator(),
        )
    }
    qpos = torch.zeros(36, dtype=torch.float32)
    qpos[3] = 1.0
    qvel = torch.zeros(35, dtype=torch.float32)
    ticks = tuple(
        SimpleNamespace(
            timestamp_us=(index + 1) * 20_000,
            qpos=qpos,
            qvel=qvel,
            observation=torch.zeros(2),
        )
        for index in range(7)
    )
    policy_input = SimpleNamespace(
        env_id=0,
        timestamp_us=140_000,
        qpos=qpos,
        qvel=qvel,
        observation=torch.zeros(2),
        scalars={},
        feedback_trace=SimpleNamespace(ticks=ticks),
        bootstrap_requested=bootstrap_requested,
    )

    (output,) = policy.step((policy_input,), sample_actions=False)

    assert facade.feedback_count == 7
    assert facade.critic_calls == int(bootstrap_requested)
    assert (output.value is not None) is bootstrap_requested


def test_current_h70_policy_session_requires_k25_execution_contract() -> None:
    humanoid_policy, humanoid_policy_server = _humanoid_policy_modules()
    joint_names = humanoid_policy_server.MOTION_REFERENCE_JOINT_NAMES
    request = SimpleNamespace(
        execution_mode=2,
        observation_schema="videomimic_motion_planner_state.v1",
        action_schema="g1_motion_reference_29d_50hz_h70.v1",
        action_size=0,
        joint_names=joint_names,
        observation_terms=[SimpleNamespace(name="navigation_position_xy", size=2)],
        reference_spec=SimpleNamespace(
            schema="g1_motion_reference_29d_50hz_h70.v1",
            joint_names=joint_names,
            frame_count=70,
            sample_period_us=20_000,
            control_ticks_per_policy_step=CONTROLLER_TICKS_PER_MACRO,
        ),
        attempt_id="attempt",
        scene_id="hq_stairs",
        scenario_id="ascend",
    )

    humanoid_policy._validate_session_request(
        request,
        planner_mode=PLANNER_MODE_SHADOW_ROLLOUT,
    )

    request.reference_spec.frame_count = 50
    with pytest.raises(ValueError, match="H=70"):
        humanoid_policy._validate_session_request(
            request,
            planner_mode=PLANNER_MODE_SHADOW_ROLLOUT,
        )

    request.reference_spec.frame_count = 70
    request.reference_spec.control_ticks_per_policy_step = 1
    with pytest.raises(ValueError, match="K25"):
        humanoid_policy._validate_session_request(
            request,
            planner_mode=PLANNER_MODE_SHADOW_ROLLOUT,
        )


def test_planner_mode_contract_is_only_h70_h69_k25() -> None:
    contract = planner_mode_contract(PLANNER_MODE_SHADOW_ROLLOUT)
    assert contract.actor_steps == 69
    assert contract.controller_ticks == 25
    assert contract.reference_frames == 70
    assert contract.action_schema == "g1_motion_reference_29d_50hz_h70.v1"

    with pytest.raises(ValueError, match="K25"):
        planner_mode_contract(PLANNER_MODE_SHADOW_ROLLOUT, macro_period_us=20_000)
    with pytest.raises(ValueError, match="K25"):
        planner_mode_contract(PLANNER_MODE_SHADOW_ROLLOUT, macro_period_us=100_000)


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
    assert (
        model_weights_sha256(planner)
        == hashlib.sha256((tmp_path / "model.safetensors").read_bytes()).hexdigest()
    )
    assert not (tmp_path / "generation_config.json").exists()
    run_config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(tmp_path),
                bundle_config={"planner_mode": PLANNER_MODE_SHADOW_ROLLOUT},
            )
        )
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


def test_planner_value_only_path_matches_h70_boundary_value_without_actor_gradients() -> (
    None
):
    model = _planner()
    observations = _observations(batch_size=5)

    full_values = model(**observations)["values"]
    value_only = model.forward_values(
        {**observations, "return_log_prob": True}  # type: ignore[dict-item]
    )

    assert full_values is not None
    torch.testing.assert_close(value_only, full_values, rtol=0.0, atol=0.0)
    value_only.sum().backward()
    groups = model.ppo_parameter_groups()
    assert all(parameter.grad is None for parameter in groups["actor"])
    assert any(parameter.grad is not None for parameter in groups["critic"])


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


def test_current_h70_replay_requires_unmodified_applied_reference() -> None:
    """The H70 controller must apply the current-policy reference directly."""
    digest = "d" * 64
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
            **_receipt(digest, applied_digest="e" * 64),
        },
    )

    with pytest.raises(ValueError, match="modified before controller"):
        build_model_inputs(_run_config(planner_mode=PLANNER_MODE_SHADOW_ROLLOUT))(
            replay
        )


@pytest.mark.parametrize(
    ("duration", "causal_tokens", "boundary_membership"),
    (
        (1, 45, (True, False, False)),
        (2, 46, (True, True, False)),
        (24, 68, (True, True, False)),
        (CONTROLLER_TICKS_PER_MACRO, 69, (True, True, True)),
    ),
    ids=("K1", "K2", "K24", "K25"),
)
def test_replay_derives_early_terminal_token_causality_mask(
    duration: int,
    causal_tokens: int,
    boundary_membership: tuple[bool, bool, bool],
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
    assert tuple(bool(mask[index]) for index in (44, 45, 68)) == boundary_membership


def test_replay_rejects_k25_only_actor_trace() -> None:
    """K25 is execution duration; PPO still owns all 69 H70 actor decisions."""
    actor_steps = CONTROLLER_TICKS_PER_MACRO
    digest = "a" * 64
    replay = PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_videomimic_planner",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.zeros(()),
        payload={
            "shadow_observations": _observations(actor_steps=actor_steps),
            "raw_actions": torch.zeros(actor_steps, 23),
            "executed_actions": torch.zeros(actor_steps, 23),
            "old_token_logprobs": torch.zeros(actor_steps),
            "reference_sha256": digest,
            **_receipt(digest, duration=CONTROLLER_TICKS_PER_MACRO),
        },
    )

    with pytest.raises(ValueError, match="must have shape"):
        build_model_inputs(_run_config(planner_mode=PLANNER_MODE_SHADOW_ROLLOUT))(
            replay
        )


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
