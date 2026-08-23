# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alpagym_host.config import CosmosRLMode, ExecutionBackend, ProvenanceMode
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
        execution=SimpleNamespace(
            backend=ExecutionBackend.local_process,
            provenance_mode=ProvenanceMode.disabled,
        ),
        dataset=SimpleNamespace(scene_ids=["hq_stairs"]),
        policy=SimpleNamespace(
            kind="humanoid",
            model=SimpleNamespace(
                kind="g1_vla",
                path=str(tmp_path / "models" / "attested-base"),
                bundle_config={
                    "sampling_mode": "native_ode_qualification",
                    "qualification_model_source": "base_attested",
                },
            ),
            inference=SimpleNamespace(return_trace_for_rl=False),
        ),
        cosmos=SimpleNamespace(mode=CosmosRLMode.colocated),
        alpasim=SimpleNamespace(
            simulation_domain="humanoid",
            startup_timeout_s=10.0,
            simulation_timeout_s=20.0,
            humanoid=SimpleNamespace(
                repo_path=str(tmp_path / "humanoid"),
                scene_store_path=str(tmp_path / "scene_store"),
                scene_cache_path=str(tmp_path / "scene_cache"),
                runtime_cache_path=str(tmp_path / "runtime_cache"),
                scenario_ids_by_scene={"hq_stairs": "ascend"},
                rollout_seed_base=292285,
            ),
        ),
        artifact_paths=SimpleNamespace(
            run_dir=run_dir,
            artifacts_dir=run_dir / "artifacts",
            alpasim_log_dir=run_dir / "alpasim",
            alpasim_scene_ids_path=run_dir / "alpasim_scene_ids.yaml",
            resolved_config_path=run_dir / "resolved_config.yaml",
            cosmos_config_path=run_dir / "cosmos_config.toml",
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


def test_qualification_validation_requires_explicit_nonfallback_model_source(
    tmp_path: Path,
) -> None:
    config = _qualification_config(tmp_path)
    del config.policy.model.bundle_config["qualification_model_source"]
    with pytest.raises(ValueError, match="explicit qualification_model_source"):
        _validate_rollout_qualification_config(config)

    config = _qualification_config(tmp_path)
    config.policy.model.bundle_config["candidate_overlay_path"] = str(
        tmp_path / "candidate"
    )
    with pytest.raises(ValueError, match="base_attested forbids"):
        _validate_rollout_qualification_config(config)

    config = _qualification_config(tmp_path)
    config.policy.model.bundle_config["qualification_model_source"] = (
        "candidate_overlay"
    )
    with pytest.raises(ValueError, match="requires candidate_overlay_path"):
        _validate_rollout_qualification_config(config)

    candidate = tmp_path / "candidate" / "step_3"
    candidate.mkdir(parents=True)
    (candidate / "candidate_manifest.json").write_text("{}", encoding="utf-8")
    config.policy.model.bundle_config["candidate_overlay_path"] = str(candidate)
    config.policy.model.bundle_config["expected_candidate_sha256"] = "a" * 64
    _validate_rollout_qualification_config(config)


def test_qualification_rejects_loaded_base_for_candidate_request(
    tmp_path: Path,
) -> None:
    from alpagym_host.qualification_lifecycle import (
        _require_loaded_source_matches_config,
    )

    candidate = tmp_path / "candidate" / "step_3"
    candidate.mkdir(parents=True)
    config = _qualification_config(tmp_path)
    config.policy.model.bundle_config.update(
        {
            "qualification_model_source": "candidate_overlay",
            "candidate_overlay_path": str(candidate),
            "expected_candidate_sha256": "a" * 64,
        }
    )
    with pytest.raises(ValueError, match="differs from the explicit config"):
        _require_loaded_source_matches_config(
            config=config,
            loaded_source={"source_kind": "base_attested"},
        )


def test_qualification_rejects_loaded_candidate_with_a_different_digest(
    tmp_path: Path,
) -> None:
    from alpagym_host.qualification_lifecycle import (
        _require_loaded_source_matches_config,
    )

    candidate = tmp_path / "candidate" / "step_3"
    candidate.mkdir(parents=True)
    config = _qualification_config(tmp_path)
    config.policy.model.bundle_config.update(
        {
            "qualification_model_source": "candidate_overlay",
            "candidate_overlay_path": str(candidate),
            "expected_candidate_sha256": "a" * 64,
        }
    )

    with pytest.raises(ValueError, match="candidate SHA256"):
        _require_loaded_source_matches_config(
            config=config,
            loaded_source={
                "source_kind": "candidate_overlay",
                "candidate_overlay": {
                    "path": str(candidate),
                    "candidate_sha256": "b" * 64,
                },
            },
        )


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


@pytest.mark.parametrize(
    "lifecycle_mode",
    ("disabled", "formal", "formal_cleanup_failure", "formal_finalize_failure"),
)
def test_execute_qualification_rollout_bypasses_trainer_and_persists_raw_episode(
    tmp_path: Path,
    monkeypatch,
    lifecycle_mode: str,
) -> None:
    from alpagym_host import qualification_lifecycle
    from alpagym_host.qualification_lifecycle import execute_qualification_rollout

    config = _qualification_config(tmp_path)
    if lifecycle_mode != "disabled":
        config.execution.provenance_mode = ProvenanceMode.required
    output = _policy_output()
    record = SimpleNamespace(
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
    provenance = None
    if lifecycle_mode != "disabled":
        monkeypatch.setattr(
            qualification_lifecycle,
            "_configure_formal_subprocess_environment",
            lambda **kwargs: {"formal": "environment"},
        )

        class Provenance:
            def capture_runtime_ready(self, **kwargs) -> None:
                calls.append(("runtime_ready", kwargs))

            def finalize(self, **kwargs) -> None:
                calls.append(("finalize", kwargs))
                if lifecycle_mode == "formal_finalize_failure":
                    raise RuntimeError("formal finalization failed")

        provenance = Provenance()

        class ProvenanceFactory:
            @staticmethod
            def capture_prelaunch(**kwargs):
                calls.append(("prelaunch", kwargs))
                return provenance

        monkeypatch.setattr(
            qualification_lifecycle,
            "FormalRunProvenance",
            ProvenanceFactory,
        )
        monkeypatch.setattr(
            qualification_lifecycle,
            "_run_formal_import_probe",
            lambda **kwargs: {"probe": "receipt"},
        )

        def cleanup_wizards(**kwargs) -> None:
            calls.append(("formal_cleanup", kwargs))
            if lifecycle_mode == "formal_cleanup_failure":
                raise RuntimeError("formal cleanup failed")

        monkeypatch.setattr(
            qualification_lifecycle,
            "_cleanup_wizard_processes",
            cleanup_wizards,
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

        def get_model(self):
            return SimpleNamespace(
                alpagym_inference_source_identity={
                    "source_kind": "base_attested",
                    "base_bundle": {"model_id": "test-base"},
                }
            )

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
    monkeypatch.setattr(
        qualification_lifecycle,
        "build_simulation_request_proto",
        lambda **kwargs: SimpleNamespace(**kwargs),
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

    if lifecycle_mode == "formal_cleanup_failure":
        with pytest.raises(RuntimeError, match="formal cleanup failed"):
            execute_qualification_rollout(config)
    elif lifecycle_mode == "formal_finalize_failure":
        with pytest.raises(RuntimeError, match="formal finalization failed"):
            execute_qualification_rollout(config)
    else:
        execute_qualification_rollout(config)

    episode_path = next(config.artifact_paths.artifacts_dir.glob("hq_stairs_*.json"))
    episode = read_episode_json(episode_path)
    assert episode.num_steps == 1
    assert episode.reward is not None and episode.reward.total == 3.5
    assert episode.policy_outputs[0].replay_data is not None
    assert "transition" not in episode.policy_outputs[0].replay_data.payload
    metrics_path = config.artifact_paths.artifacts_dir / "qualification_metrics.json"
    assert metrics_path.is_file()
    metrics = metrics_path.read_text(encoding="utf-8")
    assert '"humanoid_total_return": 3.5' in metrics
    assert '"humanoid_reward"' in metrics
    assert '"source_kind": "base_attested"' in metrics
    assert not any(isinstance(call, tuple) and call[0] == "discard" for call in calls)
    if lifecycle_mode == "disabled":
        assert ("terminated", process) in calls
    else:
        runtime_ready = next(call for call in calls if call[0] == "runtime_ready")
        assert runtime_ready[1]["workload_kind"] == "qualification_rollout"
        assert runtime_ready[1]["import_probe"] == {"probe": "receipt"}
        assert runtime_ready[1]["scene_cache_root"] == tmp_path / "scene_cache"
        assert runtime_ready[1]["runtime_cache_root"] == tmp_path / "runtime_cache"
        assert any(call[0] == "formal_cleanup" for call in calls)
        finalize = next(call for call in calls if call[0] == "finalize")
        assert finalize[1]["run_completed"] is True
        assert (finalize[1]["cleanup_error"] is not None) is (
            lifecycle_mode == "formal_cleanup_failure"
        )
