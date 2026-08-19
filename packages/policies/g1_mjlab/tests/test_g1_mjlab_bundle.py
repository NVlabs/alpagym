from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from alpagym_g1_mjlab.bundle import (
    build_model_inputs,
    get_bundle,
    load_inference_model,
    setup_tokenizer,
)
from alpagym_g1_mjlab.humanoid_policy import build_humanoid_policy_factory
from alpagym_g1_mjlab.model import (
    ACTION_SCHEMA,
    G1MjlabActorCriticModel,
    G1MjlabConfig,
    G1_MJLAB_REPLAY_SCHEMA,
    JOINT_NAMES,
    OBSERVATION_SCHEMA,
    OBS_DIMS,
    OBS_KEYS,
    register_g1_mjlab_model,
)
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.replay import (
    ActionSelection,
    PolicyReplayData,
    TrainerReplayData,
    TrainerReplayDataBatch,
    TrainingSignal,
)


def _write_bundle(path: Path) -> None:
    path.mkdir(exist_ok=True)
    register_g1_mjlab_model()
    config = G1MjlabConfig(hidden_dims=[8], init_std=0.2)
    (path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    model = G1MjlabActorCriticModel(config)
    torch.save({"model_state_dict": model.state_dict()}, path / "pytorch_model.bin")


def _replay() -> PolicyReplayData:
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema=G1_MJLAB_REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family="g1_mjlab",
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
        old_logprob=torch.tensor(-1.25),
        payload={
            "observation": {key: torch.zeros(OBS_DIMS[key]) for key in OBS_KEYS},
            "action": torch.zeros(23),
        },
    )


def test_bundle_entrypoint_is_policy_bundle() -> None:
    bundle = get_bundle()
    assert callable(bundle.build_data_packer)
    assert callable(bundle.load_inference_model)


def test_register_g1_model_is_idempotent() -> None:
    register_g1_mjlab_model()
    register_g1_mjlab_model()


def test_build_model_inputs_scores_continuous_action_payload() -> None:
    model_inputs, old_logprob = build_model_inputs(SimpleNamespace())(_replay())
    assert set(model_inputs) == {*OBS_KEYS, "actions"}
    assert model_inputs["actions"].shape == (23,)
    torch.testing.assert_close(old_logprob, torch.tensor(-1.25))


def _session_request() -> SimpleNamespace:
    return SimpleNamespace(
        action_size=23,
        random_seed=17,
        observation_schema=OBSERVATION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        observation_terms=[
            SimpleNamespace(name=key, size=OBS_DIMS[key]) for key in OBS_KEYS
        ],
        joint_names=JOINT_NAMES,
        attempt_id="attempt-0",
        scene_id="hq_stairs",
        scenario_id="ascend",
    )


def test_build_model_inputs_rejects_foreign_schema() -> None:
    replay = _replay()
    replay = PolicyReplayData(
        replay_schema_version=replay.replay_schema_version,
        payload_schema="other",
        payload_schema_version=replay.payload_schema_version,
        model_family=replay.model_family,
        action_selection=replay.action_selection,
        old_logprob=replay.old_logprob,
        payload=replay.payload,
    )
    with pytest.raises(ValueError, match="payload_schema"):
        build_model_inputs(SimpleNamespace())(replay)


def test_g1_model_trains_with_alpagym_ppo_minibatch() -> None:
    from alpagym_runtime.cosmos.trainer import AlpagymPPOTrainer

    register_g1_mjlab_model()
    model = G1MjlabActorCriticModel(G1MjlabConfig(hidden_dims=[64, 32], init_std=0.25))
    trainer = object.__new__(AlpagymPPOTrainer)
    trainer.device = torch.device("cpu")
    trainer._reference_model = None
    trainer._grpo_ratio_clip_low = 0.2
    trainer._grpo_ratio_clip_high = 0.2
    trainer._kl_beta = 0.0
    trainer._value_loss_coef = 0.5
    trainer._value_clip_range = None
    trainer.model = model
    trainer.optimizers = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer.data_packer = _G1SmokePacker()
    trainer.all_reduce_states = MethodType(_step_without_distributed, trainer)

    samples = _ppo_samples_for_model(model)
    advantages = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float32)
    actor_before = [param.detach().clone() for param in model.actor.parameters()]
    critic_before = [param.detach().clone() for param in model.critic.parameters()]
    std_before = model.std.detach().clone()

    loss, kl, ratio_max, ratio_min, clip_fraction, _grad_norm = (
        trainer._train_minibatch(
            samples,
            advantages,
            inter_policy_nccl=object(),
        )
    )

    assert torch.isfinite(torch.tensor(loss))
    assert kl == 0.0
    assert ratio_min == pytest.approx(1.0)
    assert ratio_max == pytest.approx(1.0)
    assert clip_fraction == 0.0
    assert _parameters_changed(
        model.actor.parameters(), actor_before
    ) or not torch.equal(
        model.std.detach(),
        std_before,
    )
    assert _parameters_changed(model.critic.parameters(), critic_before)


class _G1SmokePacker:
    def policy_collate_fn(
        self, samples: list[TrainerReplayData]
    ) -> TrainerReplayDataBatch:
        return TrainerReplayDataBatch.stack(samples)


def _step_without_distributed(self: object, inter_policy_nccl: object) -> float:
    del inter_policy_nccl
    self.optimizers.step()
    self.optimizers.zero_grad()
    return 0.0


def _ppo_samples_for_model(model: G1MjlabActorCriticModel) -> list[TrainerReplayData]:
    samples: list[TrainerReplayData] = []
    for step_index, reward in enumerate((1.0, 0.5, 0.25, 0.0)):
        obs = {key: torch.zeros(OBS_DIMS[key], dtype=torch.float32) for key in OBS_KEYS}
        obs["torso_xy_rel"][0] = 0.1 * step_index
        action = torch.zeros(23, dtype=torch.float32)
        with torch.no_grad():
            batched_obs = {key: value.unsqueeze(0) for key, value in obs.items()}
            old_logprob = model(actions=action.unsqueeze(0), **batched_obs)[
                "log_probs"
            ].reshape(1)
        samples.append(
            TrainerReplayData(
                model_inputs={**obs, "actions": action},
                training_signal=TrainingSignal(
                    old_logprobs=old_logprob.to(dtype=torch.float32),
                    is_padding=torch.zeros(1, dtype=torch.bool),
                    rewards=torch.tensor([reward], dtype=torch.float32),
                    terminateds=torch.tensor([False], dtype=torch.bool),
                    truncateds=torch.tensor([step_index == 3], dtype=torch.bool),
                    old_values=torch.zeros(1, dtype=torch.float32),
                    bootstrap_values=torch.zeros(1, dtype=torch.float32),
                    returns=torch.tensor([1.0 + reward], dtype=torch.float32),
                ),
                rollout_id="g1-smoke",
                weight_version=torch.zeros((), dtype=torch.int64),
            )
        )
    return samples


def _parameters_changed(
    parameters: object,
    before: list[torch.Tensor],
) -> bool:
    return any(
        not torch.equal(param.detach(), previous)
        for param, previous in zip(parameters, before)
    )


def _write_safetensors_bundle(path: Path, *, indexed: bool) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    path.mkdir(exist_ok=True)
    register_g1_mjlab_model()
    config = G1MjlabConfig(hidden_dims=[8], init_std=0.2)
    (path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    model = G1MjlabActorCriticModel(config)
    state_dict = {
        key: torch.full_like(value, 0.125).to(dtype=torch.bfloat16)
        for key, value in model.state_dict().items()
    }
    shard_name = "00000.safetensors" if indexed else "model.safetensors"
    save_file(state_dict, path / shard_name)
    if indexed:
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            value.numel() * value.element_size()
                            for value in state_dict.values()
                        )
                    },
                    "weight_map": {key: shard_name for key in state_dict},
                }
            ),
            encoding="utf-8",
        )
    return state_dict


def test_disaggregated_humanoid_policy_keeps_its_session_model_lease(
    tmp_path: Path,
) -> None:
    """A live-model sync cannot change actions inside an already-open episode."""

    _write_bundle(tmp_path)
    run_config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(tmp_path),
                device="cpu",
                bundle_config={"deterministic": True},
            )
        )
    )
    inference_model = load_inference_model(
        run_config,
        torch.device("cpu"),
        torch.float32,
    )
    engine = InferenceEngine(
        inference_model=inference_model,
        sampling=SimpleNamespace(),
        return_trace_for_rl=False,
        max_batch_size=1,
        require_session_model_leases=True,
    )
    lease = engine.create_model_lease(behavior_policy_version=4)
    engine.register_session_model_lease("session", lease)
    policy = build_humanoid_policy_factory(run_config, engine)(
        "session", _session_request()
    )
    flat_obs = torch.zeros(sum(OBS_DIMS.values()))
    policy_input = SimpleNamespace(env_id=0, observation=flat_obs)
    leased_action_before_sync = policy.step((policy_input,))[0].action

    replacement = G1MjlabActorCriticModel(G1MjlabConfig(hidden_dims=[8], init_std=0.2))
    with torch.no_grad():
        replacement.actor[-1].weight.zero_()
        replacement.actor[-1].bias.fill_(0.75)
    engine.set_model(replacement)
    leased_action_after_sync = policy.step((policy_input,))[0].action

    torch.testing.assert_close(
        leased_action_after_sync,
        leased_action_before_sync,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        leased_action_after_sync,
        torch.full((23,), 0.75),
    )


@pytest.mark.parametrize("indexed", [True, False], ids=["shard-index", "single-file"])
def test_load_inference_model_from_safetensors_export(
    tmp_path: Path, indexed: bool
) -> None:
    expected = _write_safetensors_bundle(tmp_path, indexed=indexed)
    run_config = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(tmp_path),
                device="cpu",
                bundle_config={"deterministic": True},
            )
        )
    )

    model = load_inference_model(
        run_config, torch.device("cpu"), torch.float32
    ).get_model()

    assert not (tmp_path / "pytorch_model.bin").exists()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key].to(dtype=torch.float32))


def test_no_op_tokenizer_can_be_saved_by_cosmos_checkpoint_export(
    tmp_path: Path,
) -> None:
    """Cosmos data-packer checkpointing requires ``save_pretrained``."""
    tokenizer = setup_tokenizer(SimpleNamespace())
    export_dir = tmp_path / "safetensors" / "step_1"

    saved_paths = tokenizer.save_pretrained(export_dir)

    marker_path = export_dir / "g1_mjlab_no_op_tokenizer.json"
    assert saved_paths == (str(marker_path),)
    assert json.loads(marker_path.read_text(encoding="utf-8")) == {
        "format_version": 1,
        "tokenizer_type": "alpagym_g1_mjlab_no_op",
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "model_max_length": tokenizer.model_max_length,
        "vocab_size": tokenizer.vocab_size,
    }
    assert tokenizer.encode({"episode_length": 3}) == [0, 0, 0]
