"""Strict replay parsing and policy-owned collation for VLA Psi0."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alpagym_host.config import CosmosRLMode, TransportKind

from alpagym_g1_vla.bundle import (
    MODEL_FAMILY,
    REPLAY_SCHEMA,
    build_data_packer,
    build_model_inputs,
    get_bundle,
    load_inference_model,
)
from alpagym_g1_vla.model import VlaPsiActorCritic
from alpagym_g1_vla.provenance import MODEL_ID, RUN_CONFIG_SHA256
from alpagym_runtime.replay import (
    ActionSelection,
    PolicyReplayData,
    TrainerReplayData,
    TrainingSignal,
)
from alpagym_runtime.types import EpisodeOutput, PolicyOutput


def _run_config(
    *,
    expected_schedule_sha256: str = "a" * 64,
    model_path: Path | str = f"/test/models/{MODEL_ID}",
) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                bundle_config={
                    "expected_schedule_sha256": expected_schedule_sha256,
                    "expected_flow_noise_level": 0.4,
                    "expected_flow_ignore_last": True,
                },
                path=str(model_path),
            )
        ),
        expected_valid_steps=2,
        transport=SimpleNamespace(kind=TransportKind.disk),
        cosmos=SimpleNamespace(mode=CosmosRLMode.disaggregated),
    )


def _replay(
    *,
    input_ids: list[int] | None = None,
    image_grid_thw: torch.Tensor | None = None,
    include_pooling_metadata: bool = True,
) -> PolicyReplayData:
    if input_ids is None:
        input_ids = [101, 102, 103]
    if image_grid_thw is None:
        image_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]], dtype=torch.int64)
    pixel_rows = int(image_grid_thw.prod(dim=1).sum().item())
    latent_chain = torch.zeros((11, 30, 38), dtype=torch.float32)
    wire_actions = torch.full((30, 38), 0.25, dtype=torch.float32)
    old_element_logprobs = torch.linspace(-0.001, 0.001, 1140, dtype=torch.float32)
    rtc_prefix_mask = torch.zeros(30, dtype=torch.bool)
    payload = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.int64),
        "pixel_values": torch.arange(pixel_rows * 2, dtype=torch.float32).reshape(
            pixel_rows, 2
        ),
        "image_grid_thw": image_grid_thw,
        "physical_states": torch.zeros((1, 29), dtype=torch.float32),
        "latent_chain": latent_chain,
        "denoise_index": torch.tensor(3, dtype=torch.int64),
        "clipped_normalized_actions": latent_chain[-1].clone(),
        "denormalized_wire_actions": wire_actions,
        "rtc_prefix_normalized_actions": torch.zeros((30, 38), dtype=torch.float32),
        "rtc_prefix_mask": rtc_prefix_mask,
        "old_element_logprobs": old_element_logprobs,
        "schedule_sha256": "a" * 64,
        "flow_noise_level": 0.4,
        "flow_ignore_last": True,
        "run_config_sha256": RUN_CONFIG_SHA256,
        "humanoid": {"vla_raw_action_rows": wire_actions.clone()},
    }
    if include_pooling_metadata:
        payload["effective_image_grid_thw"] = torch.ones_like(image_grid_thw)
        payload["visual_pool_factors"] = torch.tensor(
            [2] * (image_grid_thw.shape[0] - 1) + [1], dtype=torch.int64
        )
    return PolicyReplayData(
        replay_schema_version=1,
        payload_schema=REPLAY_SCHEMA,
        payload_schema_version=1,
        model_family=MODEL_FAMILY,
        action_selection=ActionSelection(0, 0),
        old_logprob=old_element_logprobs.sum(),
        payload=payload,
    )


def _trainer_sample(replay: PolicyReplayData, *, rollout_id: str) -> TrainerReplayData:
    model_inputs, old_logprob = build_model_inputs(_run_config())(replay)
    return TrainerReplayData(
        model_inputs=model_inputs,
        training_signal=TrainingSignal(
            old_logprobs=old_logprob.reshape(1),
            is_padding=torch.zeros(1, dtype=torch.bool),
        ),
        rollout_id=rollout_id,
        weight_version=torch.tensor(4, dtype=torch.int64),
    )


def _local_model_root(tmp_path: Path) -> Path:
    """Create only the checkpoint metadata needed by unit-test collation."""
    model_root = tmp_path / "policy_eval" / "models" / MODEL_ID
    metadata = model_root / "base_vlm" / "generation_config.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"pad_token_id": 151643}', encoding="utf-8")
    return model_root


def test_bundle_uses_the_generic_vla_replay_identity() -> None:
    """Replay identity must match the public policy-bundle kind exactly."""
    assert MODEL_FAMILY == "g1_vla"
    assert REPLAY_SCHEMA == "g1_vla.flow_sde.v1"


def test_parser_builds_full_chunk_density_and_action_replay() -> None:
    replay = _replay()
    model_inputs, old_logprob = build_model_inputs(_run_config())(replay)

    torch.testing.assert_close(
        old_logprob, replay.payload["old_element_logprobs"].sum()
    )


def test_parser_normalizes_uncompressed_visual_history_to_identity_pooling() -> None:
    replay = _replay(include_pooling_metadata=False)
    model_inputs, _ = build_model_inputs(_run_config())(replay)

    torch.testing.assert_close(
        model_inputs["effective_image_grid_thw"],
        replay.payload["image_grid_thw"],
    )
    torch.testing.assert_close(
        model_inputs["visual_pool_factors"],
        torch.ones(replay.payload["image_grid_thw"].shape[0], dtype=torch.int64),
    )
    assert model_inputs["latent_chain"].shape == (11, 30, 38)
    assert model_inputs["denoise_indices"].shape == ()
    assert model_inputs["flow_schedule_sha256"] == "a" * 64
    assert model_inputs["old_element_logprobs"].shape == (30 * 38,)
    torch.testing.assert_close(
        model_inputs["denormalized_wire_actions"],
        replay.payload["humanoid"]["vla_raw_action_rows"],
        rtol=0,
        atol=0,
    )


def test_bundle_wires_ragged_collator_for_different_history_lengths(
    tmp_path: Path,
) -> None:
    first = _trainer_sample(_replay(), rollout_id="rollout-a")
    second = _trainer_sample(
        _replay(
            input_ids=[201, 202, 203, 204, 205],
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.int64),
        ),
        rollout_id="rollout-b",
    )
    packer = build_data_packer(
        _run_config(model_path=_local_model_root(tmp_path)), cosmos_role="Controller"
    )

    batch = packer.policy_collate_fn([first, second])

    assert batch.model_inputs["input_ids"].shape == (2, 5)
    torch.testing.assert_close(
        batch.model_inputs["input_ids"][0],
        torch.tensor([101, 102, 103, 151643, 151643]),
    )
    torch.testing.assert_close(
        batch.model_inputs["image_offsets"], torch.tensor([0, 2, 3])
    )
    torch.testing.assert_close(
        batch.model_inputs["patch_offsets"], torch.tensor([0, 3, 7])
    )
    assert batch.model_inputs["latent_chain"].shape == (2, 11, 30, 38)
    assert batch.model_inputs["old_element_logprobs"].shape == (2, 1140)
    assert batch.model_inputs["flow_schedule_sha256"] == "a" * 64

    # Mirror the trainer boundary: element factors are consumed for exact-sum
    # audit before model.forward. Every remaining field must bind to the core.
    forward_inputs = dict(batch.model_inputs)
    forward_inputs.pop("old_element_logprobs")
    inspect.signature(VlaPsiActorCritic.forward).bind(object(), **forward_inputs)


def test_packer_normalizes_mixed_pooling_metadata_within_one_episode(
    tmp_path: Path,
) -> None:
    first_replay = _replay(
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        include_pooling_metadata=False,
    )
    second_replay = _replay()

    def _output(replay_data: PolicyReplayData) -> PolicyOutput:
        return PolicyOutput(
            chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
            chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            chosen_dt_us=torch.zeros(1, dtype=torch.int64),
            replay_data=replay_data,
        )

    episode = EpisodeOutput(
        scene_id="scene",
        session_uuid="mixed-pooling",
        num_steps=2,
        policy_outputs=(_output(first_replay), _output(second_replay)),
    )
    packer = build_data_packer(
        _run_config(model_path=_local_model_root(tmp_path)), cosmos_role="Controller"
    )

    samples = packer.get_policy_input(None, episode)

    assert len({frozenset(sample.model_inputs) for sample in samples}) == 1
    torch.testing.assert_close(
        samples[0].model_inputs["effective_image_grid_thw"],
        first_replay.payload["image_grid_thw"],
    )
    torch.testing.assert_close(
        samples[0].model_inputs["visual_pool_factors"], torch.ones(1, dtype=torch.int64)
    )

    batch = packer.policy_collate_fn(samples)
    torch.testing.assert_close(
        batch.model_inputs["image_offsets"], torch.tensor([0, 1, 3])
    )
    torch.testing.assert_close(
        batch.model_inputs["visual_pool_factors"], torch.tensor([1, 2, 1])
    )


def test_parser_rejects_schedule_scalar_logprob_and_wire_drift() -> None:
    replay = _replay()
    parser = build_model_inputs(_run_config())

    payload = dict(replay.payload)
    payload["schedule_sha256"] = "b" * 64
    replay = replace(replay, payload=payload)
    with pytest.raises(ValueError, match="schedule_sha256 does not match"):
        parser(replay)

    replay = _replay()
    replay = replace(
        replay, old_logprob=replay.payload["old_element_logprobs"].sum()[None]
    )
    with pytest.raises(ValueError, match="old_logprob must be scalar"):
        parser(replay)

    replay = _replay()
    assert replay.old_logprob is not None
    replay = replace(replay, old_logprob=replay.old_logprob + 1.0)
    with pytest.raises(ValueError, match="equal sum"):
        parser(replay)

    replay = _replay()
    payload = dict(replay.payload)
    humanoid = dict(payload["humanoid"])
    raw_rows = humanoid["vla_raw_action_rows"].clone()
    raw_rows[0, 0] += 1.0
    humanoid["vla_raw_action_rows"] = raw_rows
    payload["humanoid"] = humanoid
    replay = replace(replay, payload=payload)
    with pytest.raises(ValueError, match="must exactly equal"):
        parser(replay)

    replay = _replay()
    payload = dict(replay.payload)
    payload["flow_noise_level"] = 0.5
    with pytest.raises(ValueError, match="flow_noise_level"):
        parser(replace(replay, payload=payload))

    replay = _replay()
    payload = dict(replay.payload)
    payload["flow_ignore_last"] = False
    with pytest.raises(ValueError, match="flow_ignore_last"):
        parser(replace(replay, payload=payload))

    replay = _replay()
    payload = dict(replay.payload)
    rtc_mask = payload["rtc_prefix_mask"].clone()
    rtc_mask[0] = True
    payload["rtc_prefix_mask"] = rtc_mask
    parsed, _ = parser(replace(replay, payload=payload))
    assert bool(parsed["rtc_prefix_mask"][0])

    replay = _replay()
    payload = dict(replay.payload)
    payload["run_config_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="attested checkpoint"):
        parser(replace(replay, payload=payload))


@pytest.mark.parametrize("failure", ["gap", "too_long", "range", "chain"])
def test_parser_rejects_invalid_rtc_prefix(failure: str) -> None:
    replay = _replay()
    payload = dict(replay.payload)
    mask = payload["rtc_prefix_mask"].clone()
    prefix = payload["rtc_prefix_normalized_actions"].clone()
    chain = payload["latent_chain"].clone()
    if failure == "gap":
        mask[[0, 2]] = True
        expected = "contiguous leading prefix"
    elif failure == "too_long":
        mask[:8] = True
        expected = r"\[0, 7\]"
    elif failure == "range":
        prefix[0, 0] = 1.01
        expected = r"within \[-1, 1\]"
    else:
        mask[:2] = True
        prefix[:2] = 0.25
        chain[:, :2] = 0.25
        chain[4, 1, 3] = 0.0
        expected = "changed the deterministic RTC fixed prefix"
    payload["rtc_prefix_mask"] = mask
    payload["rtc_prefix_normalized_actions"] = prefix
    payload["latent_chain"] = chain
    payload["clipped_normalized_actions"] = chain[-1].clamp(-1.0, 1.0)
    with pytest.raises(ValueError, match=expected):
        build_model_inputs(_run_config())(replace(replay, payload=payload))


def test_pad_metadata_uses_only_resolved_model_path(tmp_path: Path) -> None:
    model_root = _local_model_root(tmp_path)
    config = _run_config(model_path=model_root)
    config.policy.model.bundle_config["policy_eval_root"] = str(tmp_path / "wrong")
    with pytest.raises(ValueError, match="policy_eval_root is obsolete"):
        build_data_packer(config, cosmos_role="Controller")

    wrong_layout = tmp_path / "wrong-layout" / MODEL_ID
    metadata = wrong_layout / "base_vlm" / "generation_config.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"pad_token_id": 151643}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"models/qwen3vl"):
        build_data_packer(
            _run_config(model_path=wrong_layout), cosmos_role="Controller"
        )


def test_schedule_identity_is_required_and_loader_is_policy_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = _run_config()
    del run_config.policy.model.bundle_config["expected_schedule_sha256"]
    with pytest.raises(ValueError, match="expected_schedule_sha256 is required"):
        build_model_inputs(run_config)

    run_config = _run_config()
    del run_config.policy.model.bundle_config["expected_flow_noise_level"]
    with pytest.raises(ValueError, match="expected_flow_noise_level"):
        build_model_inputs(run_config)

    run_config = _run_config()
    del run_config.policy.model.bundle_config["expected_flow_ignore_last"]
    with pytest.raises(ValueError, match="expected_flow_ignore_last"):
        build_model_inputs(run_config)

    bundle = get_bundle()
    assert callable(bundle.load_inference_model)
    sentinel = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        "alpagym_g1_vla.bundle.load_vla_rollout_model",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        "alpagym_g1_vla.bundle.VlaNativeInferenceModel",
        lambda model: ("native", model),
    )
    assert load_inference_model(_run_config(), torch.device("cpu"), torch.float32) == (
        "native",
        sentinel,
    )
