# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch
from alpagym_host.config import CosmosRLMode, ExecutionBackend
from alpagym_host.config_validation import _validate_rollout_qualification_config
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.transport.disk import read_episode_json
from alpagym_runtime.types import PolicyOutput


def _policy_output() -> PolicyOutput:
    return PolicyOutput(
        chosen_xyz=torch.zeros((50, 29), dtype=torch.float32),
        chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(50, 1),
        chosen_dt_us=torch.arange(50, dtype=torch.int64) * 20_000,
        replay_data=PolicyReplayData(
            replay_schema_version=1,
            payload_schema="g1_vla.native_ode_qualification.v1",
            payload_schema_version=1,
            model_family="g1_vla",
            action_selection=ActionSelection(set_ix=0, sample_ix=0),
            old_logprob=None,
            payload={
                "sampling_mode": "native_ode_qualification",
                "feedback_trace": {
                    "env_id": 0,
                    "source_decision_id": 0,
                    "ticks": [
                        {
                            "timestamp_us": 20_000,
                            "qpos": torch.zeros(36),
                            "qvel": torch.zeros(35),
                        }
                    ],
                },
            },
        ),
    )


def _qualification_config(tmp_path: Path) -> SimpleNamespace:
    run_dir = tmp_path / "run"
    return SimpleNamespace(
        execution=SimpleNamespace(backend=ExecutionBackend.local_process),
        dataset=SimpleNamespace(scene_ids=["hq_stairs"]),
        policy=SimpleNamespace(
            kind="humanoid",
            model=SimpleNamespace(
                kind="g1_vla",
                bundle_config={"sampling_mode": "native_ode_qualification"},
            ),
            inference=SimpleNamespace(return_trace_for_rl=False),
        ),
        cosmos=SimpleNamespace(mode=CosmosRLMode.colocated),
        alpasim=SimpleNamespace(
            simulation_domain="humanoid",
            startup_timeout_s=10.0,
            simulation_timeout_s=20.0,
            humanoid=SimpleNamespace(
                scenario_ids_by_scene={"hq_stairs": "ascend"},
                rollout_seed_base=292285,
            ),
        ),
        artifact_paths=SimpleNamespace(
            run_dir=run_dir,
            artifacts_dir=run_dir / "artifacts",
            alpasim_log_dir=run_dir / "alpasim",
            alpasim_scene_ids_path=run_dir / "alpasim_scene_ids.yaml",
        ),
    )


def test_qualification_validation_is_native_ode_local_only(tmp_path: Path) -> None:
    config = _qualification_config(tmp_path)
    _validate_rollout_qualification_config(config)

    config.policy.inference.return_trace_for_rl = True
    try:
        _validate_rollout_qualification_config(config)
    except ValueError as exc:
        assert "return_trace_for_rl=false" in str(exc)
    else:
        raise AssertionError("qualification accepted PPO traces")


def test_cli_dispatches_rollout_without_entering_training(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from alpagym_host import cli
    from alpagym_host.qualification_lifecycle import QualificationArtifacts

    config = _qualification_config(tmp_path)
    episode = tmp_path / "episode.json"
    metrics = tmp_path / "metrics.json"
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_or_create_run_config", lambda cfg: config)
    monkeypatch.setattr(
        cli,
        "_execute_qualification_rollout",
        lambda candidate: (
            calls.append(candidate) or QualificationArtifacts(episode, metrics)
        ),
    )

    cli.main.__wrapped__(SimpleNamespace(command="rollout", logging_level="INFO"))

    assert calls == [config]
    stdout = capsys.readouterr().out
    assert f"Episode manifest: {episode}" in stdout
    assert f"Metrics manifest: {metrics}" in stdout


def test_execute_qualification_rollout_bypasses_trainer_and_persists_raw_episode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from alpagym_host import qualification_lifecycle
    from alpagym_host.qualification_lifecycle import execute_qualification_rollout
    from alpagym_runtime.alpasim.humanoid_policy_server import HumanoidSessionRecord

    config = _qualification_config(tmp_path)
    output = _policy_output()
    record = HumanoidSessionRecord(
        outputs=(output,),
        final_bootstrap_values={0: 1.25},
        behavior_policy_version=0,
    )
    rollout_return = SimpleNamespace(
        success=True,
        error="",
        behavior_policy_version="0",
        aggregated_metrics={
            "humanoid_total_return": 3.5,
            "humanoid_success": 1.0,
        },
        timestep_metrics=[
            SimpleNamespace(
                name="humanoid_reward",
                timestamps_us=[500_000],
                values=[3.5],
                valid=[True],
            )
        ],
    )

    calls: list[object] = []
    monkeypatch.setattr(
        qualification_lifecycle,
        "validate_local_qualification_prerequisites",
        lambda: calls.append("preflight"),
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "resolve_alpasim_checkout",
        lambda config: tmp_path / "alpasim",
    )

    class Process:
        pass

    process = Process()
    monkeypatch.setattr(
        qualification_lifecycle,
        "start_wizard",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "wait_for_runtime_ready",
        lambda **kwargs: ("localhost", 31000),
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "fetch_runtime_info",
        lambda host, port, timeout_s: (1, ["hq_stairs"]),
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "ensure_process_terminated",
        lambda candidate: calls.append(("terminated", candidate)),
    )

    class Engine:
        requires_session_model_leases = False

        def run_loop(self) -> None:
            calls.append("engine_loop")

        def shutdown(self) -> None:
            calls.append("engine_shutdown")

    engine = Engine()
    monkeypatch.setattr(
        qualification_lifecycle, "build_inference_engine", lambda config: engine
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "build_humanoid_policy_factory",
        lambda config, inference_engine: object(),
    )
    monkeypatch.setattr(
        qualification_lifecycle,
        "humanoid_policy_camera_required",
        lambda config: True,
    )

    class Servicer:
        def reserve_session(self, session_uuid, behavior_policy_version) -> None:
            calls.append(("reserve", session_uuid, behavior_policy_version))

        def pop_session_record(self, session_uuid):
            calls.append(("pop", session_uuid))
            return record

        def discard_session(self, session_uuid) -> None:
            calls.append(("discard", session_uuid))

    class PolicyServer:
        def __init__(self, **kwargs) -> None:
            calls.append(("policy_server", kwargs))
            self.servicer = Servicer()
            self.topology_endpoint = SimpleNamespace(host="localhost", port=32000)

        def start(self) -> None:
            calls.append("policy_start")

        def stop(self) -> None:
            calls.append("policy_stop")

    monkeypatch.setattr(qualification_lifecycle, "HumanoidPolicyServer", PolicyServer)

    class Channel:
        def close(self) -> None:
            calls.append("channel_close")

    channel = Channel()
    monkeypatch.setattr(
        qualification_lifecycle.grpc,
        "insecure_channel",
        lambda target, options: channel,
    )
    monkeypatch.setattr(
        qualification_lifecycle.grpc,
        "channel_ready_future",
        lambda candidate: SimpleNamespace(result=lambda timeout: None),
    )

    class RuntimeStub:
        def __init__(self, candidate) -> None:
            assert candidate is channel

        def simulate(self, request, timeout):
            calls.append(("simulate", request, timeout))
            return SimpleNamespace(rollout_returns=[rollout_return])

    monkeypatch.setattr(qualification_lifecycle, "RuntimeServiceStub", RuntimeStub)

    artifacts = execute_qualification_rollout(config)

    episode = read_episode_json(artifacts.episode_manifest)
    assert episode.num_steps == 1
    assert episode.reward is not None and episode.reward.total == 3.5
    assert episode.policy_outputs[0].replay_data is not None
    assert "transition" not in episode.policy_outputs[0].replay_data.payload
    assert artifacts.metrics_manifest.is_file()
    metrics = artifacts.metrics_manifest.read_text(encoding="utf-8")
    assert '"humanoid_total_return": 3.5' in metrics
    assert '"humanoid_reward"' in metrics
    assert not any(isinstance(call, tuple) and call[0] == "discard" for call in calls)
    assert ("terminated", process) in calls
