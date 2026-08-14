# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AlpaGym replay packing into trainer batches."""

import importlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from alpagym_runtime.replay import ActionSelection, DataPackerConfig, PolicyReplayData
from alpagym_runtime.transport.disk import write_episode_json
from alpagym_runtime.types import EpisodeOutput, PolicyOutput, RolloutArtifact


def _generic_build_model_inputs() -> Callable[
    [PolicyReplayData],
    tuple[dict[str, object], torch.Tensor],
]:
    """Return a policy-neutral trainer input builder."""
    return _build_generic_model_inputs


def _build_generic_model_inputs(
    replay: PolicyReplayData,
) -> tuple[dict[str, object], torch.Tensor]:
    """Convert one generic replay payload into trainer model inputs."""
    if replay.payload_schema != "generic.trajectory.v1":
        raise ValueError(f"payload_schema={replay.payload_schema!r}")
    payload = replay.payload
    model_inputs: dict[str, object] = {
        "step_index": torch.as_tensor(payload["step_index"], dtype=torch.int64),
        "token_logprob_count": torch.as_tensor(payload["token_logprob_count"], dtype=torch.int64),
    }
    for key in ("samples_list", "timesteps", "noise_level"):
        if payload.get(key) is not None:
            model_inputs[key] = torch.as_tensor(payload[key], dtype=torch.float32)
    if payload.get("cfm_method") is not None:
        model_inputs["cfm_method"] = payload["cfm_method"]
    return model_inputs, replay.old_logprob


def test_replay_packer_collates_padding_and_old_logprobs(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Replay payloads pack to fixed rows with old logprobs zero-padded."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=3),
        build_model_inputs=_generic_build_model_inputs(),
    )
    first = _write_episode(tmp_path / "first.json", "rollout-1", _generic_outputs([-1.0, -2.0]))
    second = _write_episode(tmp_path / "second.json", "rollout-2", _generic_outputs([-3.0]))

    steps_a = packer.get_policy_input(0, first.handle)
    steps_b = packer.get_policy_input(1, second.handle)
    batch = packer.policy_collate_fn([*steps_a, *steps_b])

    signal = batch.training_signal
    assert batch.rollout_ids == (
        "rollout-1",
        "rollout-1",
        "rollout-1",
        "rollout-2",
        "rollout-2",
        "rollout-2",
    )
    assert torch.equal(
        signal.old_logprobs,
        torch.tensor([-1.0, -2.0, 0.0, -3.0, 0.0, 0.0]),
    )
    assert torch.equal(
        signal.is_padding,
        torch.tensor([False, False, True, False, True, True]),
    )
    assert torch.equal(
        batch.model_inputs["step_index"],
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.int64),
    )
    assert torch.equal(
        batch.model_inputs["token_logprob_count"],
        torch.ones(6, dtype=torch.int64),
    )
    assert torch.equal(batch.weight_versions, torch.zeros(6, dtype=torch.int64))


def test_replay_packer_extracts_transition_signal_and_zero_pads_it(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Raw transition terms travel through the existing payload path and pad to zero."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=3),
        build_model_inputs=_generic_build_model_inputs(),
    )
    outputs = []
    for output, reward, terminated, old_value in zip(
        _generic_outputs([-1.0, -2.0]),
        [0.5, -1.5],
        [False, True],
        [0.25, 0.75],
    ):
        assert output.replay_data is not None
        payload = dict(output.replay_data.payload)
        payload["transition"] = {
            "reward": torch.tensor(reward),
            "terminated": terminated,
            "truncated": False,
            "old_value": torch.tensor(old_value),
        }
        replay_data = replace(output.replay_data, payload=payload)
        outputs.append(replace(output, replay_data=replay_data))
    artifact = _write_episode(tmp_path / "transition.json", "rollout-transition", outputs)

    batch = packer.policy_collate_fn(packer.get_policy_input(0, artifact.handle))

    assert batch.training_signal.advantages is None
    assert batch.training_signal.returns is None
    torch.testing.assert_close(
        batch.training_signal.rewards,
        torch.tensor([0.5, -1.5, 0.0], dtype=torch.float32),
    )
    assert torch.equal(
        batch.training_signal.terminateds,
        torch.tensor([False, True, False], dtype=torch.bool),
    )
    assert torch.equal(
        batch.training_signal.truncateds,
        torch.tensor([False, False, False], dtype=torch.bool),
    )
    torch.testing.assert_close(
        batch.training_signal.old_values,
        torch.tensor([0.25, 0.75, 0.0], dtype=torch.float32),
    )
    assert bool(batch.training_signal.is_padding[-1].item())


def test_replay_packer_exact_t_pack_has_no_padding(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """``T_valid == T_pack`` keeps every row valid."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=2),
        build_model_inputs=_generic_build_model_inputs(),
    )
    artifact = _write_episode(
        tmp_path / "exact.json",
        "rollout-exact",
        _generic_outputs([-1.0, -2.0]),
    )

    steps = packer.get_policy_input(0, artifact.handle)
    batch = packer.policy_collate_fn(steps)

    assert torch.equal(batch.training_signal.old_logprobs, torch.tensor([-1.0, -2.0]))
    assert torch.equal(batch.training_signal.is_padding, torch.tensor([False, False]))


def test_replay_packer_rejects_above_t_pack(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """``T_valid > T_pack`` fails instead of dropping trainer replay rows."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=2),
        build_model_inputs=_generic_build_model_inputs(),
    )
    artifact = _write_episode(
        tmp_path / "truncated.json",
        "rollout-truncated",
        _generic_outputs([-1.0, -2.0, -3.0]),
    )

    with pytest.raises(
        ValueError,
        match=r"rollout-truncated produced 3 policy outputs, exceeding expected_valid_steps=2",
    ):
        packer.get_policy_input(0, artifact.handle)


def test_replay_packer_rejects_missing_replay_data(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Valid rollout steps must carry replay data before trainer packing."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=_generic_build_model_inputs(),
    )
    artifact = _write_episode(
        tmp_path / "missing_replay.json",
        "rollout-missing-replay",
        [
            PolicyOutput(
                chosen_xyz=torch.zeros((2, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float32),
                chosen_dt_us=torch.tensor([0, 1], dtype=torch.int64),
                chosen_logprob=torch.zeros((1,), dtype=torch.float32),
                replay_data=None,
            )
        ],
    )

    with pytest.raises(ValueError, match="missing replay_data"):
        packer.get_policy_input(0, artifact.handle)


def test_replay_packer_rejects_any_policy_output_missing_replay_data(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Every policy output is part of the trainer artifact contract."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=_generic_build_model_inputs(),
    )
    valid_output, missing_replay = _generic_outputs([-1.0, -99.0])
    artifact = _write_episode(
        tmp_path / "missing_replay_not_skipped.json",
        "rollout-missing-replay-not-skipped",
        [valid_output, replace(missing_replay, replay_data=None)],
    )

    with pytest.raises(ValueError, match="missing replay_data"):
        packer.get_policy_input(0, artifact.handle)


def test_replay_packer_rejects_foreign_payload_schema(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """The model-input builder rejects payloads minted under another schema."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=_generic_build_model_inputs(),
    )
    output = _generic_outputs([-1.0])[0]
    assert output.replay_data is not None
    replay_data = replace(output.replay_data, payload_schema="rssm.trajectory.v1")
    artifact = _write_episode(
        tmp_path / "bad_schema.json",
        "rollout-bad-schema",
        [replace(output, replay_data=replay_data)],
    )

    with pytest.raises(ValueError, match="payload_schema"):
        packer.get_policy_input(0, artifact.handle)


def test_replay_packer_rejects_missing_token_logprob_count(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Replay payloads must carry fields required by the model-input builder."""
    del cosmos_stubs
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=_generic_build_model_inputs(),
    )
    output = _generic_outputs([-1.0])[0]
    assert output.replay_data is not None
    payload = dict(output.replay_data.payload)
    payload.pop("token_logprob_count")
    replay_data = replace(output.replay_data, payload=payload)
    artifact = _write_episode(
        tmp_path / "missing_token_count.json",
        "rollout-missing-token-count",
        [replace(output, replay_data=replay_data)],
    )

    with pytest.raises(KeyError, match="token_logprob_count"):
        packer.get_policy_input(0, artifact.handle)


def test_replay_packer_forwards_trace_fields(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Trace fields are preserved into trainer model inputs."""
    del cosmos_stubs
    samples_list = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
    timesteps = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")
    packer = packer_module.AlpagymDataPacker(
        config=DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=_generic_build_model_inputs(),
    )
    artifact = _write_episode(
        tmp_path / "cfm.json",
        "rollout-cfm",
        _generic_outputs(
            [-1.0],
            samples_list=samples_list,
            timesteps=timesteps,
            noise_level=torch.tensor([0.4], dtype=torch.float32),
            cfm_method="sde",
        ),
    )

    batch = packer.policy_collate_fn(packer.get_policy_input(0, artifact.handle))

    assert torch.equal(batch.model_inputs["samples_list"], samples_list.unsqueeze(0))
    assert torch.equal(batch.model_inputs["timesteps"], timesteps.unsqueeze(0))
    assert torch.equal(
        batch.model_inputs["noise_level"],
        torch.tensor([[0.4]], dtype=torch.float32),
    )
    assert batch.model_inputs["cfm_method"] == "sde"


def _generic_outputs(
    old_logprobs: list[float],
    samples_list: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    noise_level: torch.Tensor | None = None,
    cfm_method: str | None = None,
) -> list[PolicyOutput]:
    """Build valid policy outputs for packer tests."""
    outputs = []
    for step_index, old_logprob in enumerate(old_logprobs):
        replay = PolicyReplayData(
            replay_schema_version=1,
            payload_schema="generic.trajectory.v1",
            payload_schema_version=1,
            model_family="generic",
            action_selection=ActionSelection(set_ix=0, sample_ix=0),
            old_logprob=torch.tensor(old_logprob, dtype=torch.float32),
            payload={
                "step_index": torch.tensor(step_index, dtype=torch.int64),
                "token_logprob_count": torch.tensor(1, dtype=torch.int64),
                "samples_list": samples_list,
                "timesteps": timesteps,
                "noise_level": noise_level,
                "cfm_method": cfm_method,
            },
        )
        outputs.append(_policy_output(replay))
    return outputs


def _policy_output(replay_data: PolicyReplayData) -> PolicyOutput:
    """Build a valid policy output carrying replay data."""
    return PolicyOutput(
        chosen_xyz=torch.zeros((2, 3), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float32),
        chosen_dt_us=torch.tensor([0, 1], dtype=torch.int64),
        chosen_logprob=torch.zeros((1,), dtype=torch.float32),
        replay_data=replay_data,
    )


def _write_episode(
    path: Path,
    session_uuid: str,
    outputs: list[PolicyOutput],
) -> RolloutArtifact:
    """Persist one episode and return its artifact handle."""
    artifact = RolloutArtifact(
        handle=str(path),
        episode=EpisodeOutput(
            scene_id="scene",
            session_uuid=session_uuid,
            num_steps=len(outputs),
            policy_outputs=tuple(outputs),
        ),
    )
    write_episode_json(Path(artifact.handle), artifact.episode)
    return artifact
