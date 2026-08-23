# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint
from PIL import Image


def _deep_merge_policy_overrides(
    base: dict[str, Any], overrides: dict[str, Any]
) -> None:
    """In-place deep merge `overrides` into `base`."""
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge_policy_overrides(existing, value)
        else:
            base[key] = value


def _write_resolved_config(
    tmp_path: Path,
    scene_ids: list[str],
    policy_overrides: dict[str, Any],
    max_concurrent_rollouts: int,
    simulation_timeout_s: float,
    experiment_name: str,
    execution_backend: str = "local_process",
    runtime_scene_ids: list[str] | None = None,
    runtime_capacity: int | None = None,
    rollout_replicas: int = 1,
    runtime_count: int = 1,
) -> Path:
    """Write a resolved alpagym config YAML and return its path."""
    base_policy: dict[str, Any] = {
        "kind": "alpamayo",
        "model": {
            "kind": "alpamayo_r1",
            "path": "unused-by-test",
            "device": "cpu",
            "dtype": "float32",
            "use_cameras": [],
            "input_size": [320, 512],
            "num_context_frames": 2,
            "num_historical_waypoints": 1,
            "num_future_waypoints": 2,
            "step_dt_us": 500_000,
        },
        "inference": {
            "max_batch_size": 4,
            "return_trace_for_rl": True,
            "sampling": {
                "top_p": 1.0,
                "top_k": None,
                "temperature": 1.0,
                "num_traj_samples": 1,
                "num_traj_sets": 1,
                "max_generation_length": None,
            },
        },
        "trajectory_selector": "identity",
    }
    _deep_merge_policy_overrides(base_policy, policy_overrides)
    resolved = {
        "command": "run",
        "run_root": str(tmp_path),
        "logging_level": "DEBUG",
        "cache_root_dir": str(tmp_path / "cache"),
        "execution": {
            "backend": execution_backend,
            "resolved_config_path": None,
            "slurm": {
                "job_name": "alpagym",
                "partition": None,
                "account": None,
                "time": "02:00:00",
                "nodes": 1,
                "gpus_per_node": 8,
                "topology": {
                    "kind": "all_in_one",
                    "alpasim_gpus": 4,
                },
                "exclusive": True,
                "cpus_per_task": None,
                "container_image": None,
                "container_cache_root": None,
                "container_workdir": "/alpamayo/projects/alpagym",
                "uv_cache_dir": str(tmp_path / "uv-cache"),
                "container_mounts": [],
                "export_env": [],
            },
        },
        "dataset": {"scene_ids": scene_ids},
        "policy": base_policy,
        "reward": {"terms": [{"kind": "distance_to_gt", "scale": -0.01}]},
        "alpasim": {
            "repo_url": "ssh://git@example/alpasim.git",
            "repo_ref": "abc123",
            "startup_timeout_s": 600.0,
            "simulation_timeout_s": simulation_timeout_s,
            "wizard_args": {
                "deploy": "local",
                "topology": "1gpu",
                "driver": None,
                "driver_source": "external_dynamic",
                "force_gt_duration_us": 3_000_000,
                "control_timestep_us": 100_000,
                "n_sim_steps": 38,
                "extra_overrides": "",
            },
        },
        "cosmos": {
            "mode": "colocated",
            "launch": {
                "policy_replicas": 1,
                "rollout_replicas": rollout_replicas,
                "controller_port": 29500,
            },
            "train": {
                "max_num_steps": 1,
                "num_epochs": 1,
                "train_batch_per_replica": 2,
                "seed": 42,
                "deterministic": False,
                "optm_lr": 1.0e-6,
                "optm_warmup_steps": 20,
                "train_policy": {
                    "allowed_outdated_steps": 4,
                    "on_policy": False,
                    "mini_batch": 1,
                    "grpo_ratio_clip_low": 0.2,
                    "grpo_ratio_clip_high": 0.2,
                    "grpo_optimization_iterations": 1,
                    "kl_beta": 0.0,
                    "reference_reset_interval": 0,
                },
            },
            "policy": {
                "parallelism": {
                    "tp_size": 1,
                    "cp_size": 1,
                    "ep_size": 1,
                    "dp_shard_size": 1,
                    "pp_size": 1,
                    "pp_micro_batch_size": 1,
                    "dp_replicate_size": 1,
                },
            },
            "rollout": {
                "n_generation": 2,
                "batch_size": 1,
                "parallelism": {
                    "tp_size": 1,
                    "pp_size": 1,
                },
            },
            "logging": {
                "logger": ["console"],
                "log_training_metrics_every_n_steps": 1,
                "project_name": "alpagym",
                "experiment_name": experiment_name,
            },
        },
        "expected_valid_steps": 8,
        "artifact_paths": {
            "run_dir": str(tmp_path),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "policy_model_bundle_dir": str(
                tmp_path / "artifacts" / "policy_model_bundle"
            ),
            "resolved_config_path": str(tmp_path / "resolved_config.yaml"),
            "cosmos_config_path": str(tmp_path / "cosmos_config.toml"),
            "submit_script_path": str(tmp_path / "submit.sbatch"),
            "log_dir": str(tmp_path / "logs"),
            "topology_registry_dir": str(tmp_path / "topology"),
            "alpasim_log_dir": str(tmp_path / "alpasim"),
            "alpasim_scene_ids_path": str(tmp_path / "alpasim_scene_ids.yaml"),
            "perf_dir": str(tmp_path / "perf"),
        },
        "perf": {
            "enabled": False,
            "sample_every_n": 1,
            "resource_sample_interval_s": 5.0,
            "max_samples_per_series": 1000,
            "flush_every_n_updates": 1000,
            "flush_interval_s": 60.0,
            "collect_cpu": True,
            "collect_gpu": True,
        },
    }
    resolved_config_path = tmp_path / "resolved_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(resolved))
    (tmp_path / "alpasim_scene_ids.yaml").write_text(
        yaml.safe_dump(
            {"scene_ids": runtime_scene_ids or scene_ids},
            sort_keys=False,
        )
    )
    registry = FileTopologyRegistry(tmp_path / "topology")
    for runtime_index in range(runtime_count):
        registry.publish_alpasim_runtime(
            TopologyEndpoint(
                id=f"alpasim-runtime-{runtime_index}",
                host="runtime.local",
                port=6101 + runtime_index,
                capacity=runtime_capacity or max_concurrent_rollouts,
            )
        )
    return resolved_config_path


_ALPAMAYO_TEST_CAMERA = "camera_front_wide_120fov"


def test_entrypoint_configures_logging_from_resolved_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Cosmos workers read AlpaGym log level from resolved YAML, not generated TOML."""
    del cosmos_stubs
    entrypoint_module = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["scene_000"],
        policy_overrides={},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="logging_from_yaml",
    )
    cosmos_config_path = tmp_path / "cosmos_config.toml"
    cosmos_config_path.write_text(
        f"[custom]\nresolved_config_path = {json.dumps(str(resolved_config_path))}\n",
        encoding="utf-8",
    )
    basic_config_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        entrypoint_module,
        "get_policy_bundle",
        lambda model_kind: SimpleNamespace(
            build_data_packer=lambda config, cosmos_role: SimpleNamespace(
                close=lambda: None
            )
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    monkeypatch.setattr(entrypoint_module, "launch_worker", lambda **kwargs: None)

    entrypoint_module.main(["--config", str(cosmos_config_path)])

    assert basic_config_calls[0]["level"] == logging.DEBUG


def test_entrypoint_installs_policy_bundle_tokenizer_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Policy bundles can replace Cosmos AutoTokenizer setup before controller startup."""
    del cosmos_stubs
    entrypoint_module = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["hq_stairs"],
        policy_overrides={"model": {"kind": "g1_mjlab"}},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="bundle_tokenizer",
    )
    cosmos_config_path = tmp_path / "cosmos_config.toml"
    cosmos_config_path.write_text(
        f"[custom]\nresolved_config_path = {json.dumps(str(resolved_config_path))}\n",
        encoding="utf-8",
    )
    from cosmos_rl.policy.trainer.llm_trainer import grpo_trainer
    from cosmos_rl.utils import util as cosmos_util

    fallback_tokenizer = object()
    bundle_tokenizer = object()
    calls: list[str] = []
    monkeypatch.setattr(cosmos_util, "setup_tokenizer", lambda path: fallback_tokenizer)
    monkeypatch.setattr(
        entrypoint_module,
        "get_policy_bundle",
        lambda model_kind: SimpleNamespace(
            install_runtime_bridge=lambda: calls.append("bridge"),
            setup_tokenizer=lambda config: bundle_tokenizer,
            build_data_packer=lambda config, cosmos_role: SimpleNamespace(
                close=lambda: None
            ),
        ),
    )
    monkeypatch.setattr(entrypoint_module, "launch_worker", lambda **kwargs: None)

    entrypoint_module.main(["--config", str(cosmos_config_path)])

    assert calls == ["bridge"]
    assert cosmos_util.setup_tokenizer("unused") is bundle_tokenizer
    assert grpo_trainer.setup_tokenizer("unused") is bundle_tokenizer


def test_entrypoint_dataset_uses_discovered_runtime_scenes(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Cosmos dataset uses AlpaSim-discovered scenes, not the authored selector list."""
    del cosmos_stubs
    entrypoint_module = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["authored_scene"],
        policy_overrides={},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="runtime_scenes",
        runtime_scene_ids=["runtime_scene_a", "runtime_scene_b"],
    )

    dataset = entrypoint_module._build_dataset(
        SimpleNamespace(custom={"resolved_config_path": str(resolved_config_path)})
    )

    assert len(dataset) == 2
    assert dataset[0] == "runtime_scene_a"
    assert dataset[1] == "runtime_scene_b"


def test_rollout_generation_declares_current_weight_version(
    cosmos_stubs: None,
) -> None:
    """`rollout_generation` declares cosmos-rl's `current_weight_version` kwarg.

    Cosmos-RL `_call_rollout_generation` forwards `current_weight_version`
    via kwargs. The signature declares it explicitly (Google-style) instead
    of absorbing it via `**kwargs`, so a future cosmos kwarg surfaces as a
    loud `TypeError` rather than being silently consumed.
    """
    del cosmos_stubs
    import inspect

    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    sig = inspect.signature(rollout_module.AlpagymRollout.rollout_generation)
    assert "current_weight_version" in sig.parameters, (
        "rollout_generation must declare `current_weight_version` so a "
        "new cosmos kwarg surfaces as TypeError instead of being absorbed."
    )


def test_resumed_rollout_seed_panel_skips_completed_global_batches(
    cosmos_stubs: None,
) -> None:
    """A checkpoint continuation must not replay rollout seeds already trained on."""

    del cosmos_stubs
    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    run_config = SimpleNamespace(
        alpasim=SimpleNamespace(humanoid=SimpleNamespace(rollout_seed_base=292_285)),
        cosmos=SimpleNamespace(
            train=SimpleNamespace(
                train_batch_per_replica=4,
                resume=SimpleNamespace(enabled=True, checkpoint_step=3),
            ),
            launch=SimpleNamespace(policy_replicas=2),
        ),
    )

    assert rollout_module._effective_humanoid_rollout_seed_base(run_config) == 292_309


def test_rollout_init_acquires_runtime_for_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Distributed rollout workers acquire the runtime assigned to their driver."""
    del cosmos_stubs
    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    from cosmos_rl.rollout.rollout_base import RolloutRegistry

    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["scene_000"],
        policy_overrides={"kind": "fake"},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="cosmos_runtime_binding",
        execution_backend="slurm",
        rollout_replicas=4,
        runtime_count=2,
    )

    class FakeDriverServer:
        """Driver server stand-in that records publish host wiring."""

        instances: list["FakeDriverServer"] = []

        def __init__(
            self,
            name: str,
            max_concurrent_rollouts: int,
            policy_factory: Any,
            publish_host: str = "localhost",
        ) -> None:
            """Capture constructor arguments used by rollout init."""
            del policy_factory
            self.name = name
            self.max_concurrent_rollouts = max_concurrent_rollouts
            self.publish_host = publish_host
            self.topology_endpoint = TopologyEndpoint(
                id=name,
                host=publish_host,
                port=43292,
                capacity=max_concurrent_rollouts,
            )
            self.started = False
            self.instances.append(self)

        def start(self) -> None:
            """Record driver startup."""
            self.started = True

        def stop(self) -> None:
            """Accept driver shutdown."""

    class FakeRegistry:
        """Topology registry stand-in with two runtime endpoints."""

        acquired_driver_ids: list[str] = []
        published_drivers: list[TopologyEndpoint] = []

        def __init__(self, registry_dir: str) -> None:
            """Accept the configured registry directory."""
            self.registry_dir = registry_dir

        def acquire_alpasim_runtime(self, driver_id: str) -> TopologyEndpoint:
            """Return the endpoint assigned to the driver."""
            self.acquired_driver_ids.append(driver_id)
            endpoints = [
                TopologyEndpoint(
                    id="runtime-0",
                    host="runtime-0.local",
                    port=6101,
                    capacity=1,
                ),
                TopologyEndpoint(
                    id="runtime-1",
                    host="runtime-1.local",
                    port=6102,
                    capacity=5,
                ),
            ]
            return endpoints[1]

        def list_alpasim_runtimes(self) -> list[TopologyEndpoint]:
            """Return published runtime endpoints."""
            return [
                TopologyEndpoint(
                    id="runtime-0",
                    host="runtime-0.local",
                    port=6101,
                    capacity=1,
                ),
                TopologyEndpoint(
                    id="runtime-1",
                    host="runtime-1.local",
                    port=6102,
                    capacity=5,
                ),
            ]

        def publish_driver(self, endpoint: TopologyEndpoint) -> None:
            """Record the rollout worker's driver endpoint."""
            self.published_drivers.append(endpoint)

    class FakeStreamingWorker:
        """Streaming worker stand-in that captures the selected runtime stub."""

        instances: list["FakeStreamingWorker"] = []

        def __init__(
            self,
            *,
            alpasim_runtime_stub: Any,
            driver_server: FakeDriverServer,
            simulation_timeout_s: float,
            reward_config: Any,
            max_concurrent_rollouts: int,
            rollouts_per_payload: int,
            scene_id_resolver: Any,
        ) -> None:
            """Capture constructor arguments used by rollout init."""
            del simulation_timeout_s, reward_config
            del rollouts_per_payload, scene_id_resolver
            self.alpasim_runtime_stub = alpasim_runtime_stub
            self.driver_server = driver_server
            self.max_concurrent_rollouts = max_concurrent_rollouts
            self.instances.append(self)

        def shutdown(self) -> None:
            """Accept rollout shutdown."""

    class FakeInferenceEngine:
        """Inference engine stand-in exposing Jef's model-sync interface."""

        def __init__(self) -> None:
            """Create the fake model."""
            self.model = torch.nn.Linear(1, 1)

        def get_model(self) -> torch.nn.Module:
            """Return a model object for rollout weight sync."""
            return self.model

        @property
        def requires_session_model_leases(self) -> bool:
            """The AV-only test never enables humanoid session snapshots."""
            return False

        def run_loop(self) -> None:
            """Return immediately for the test thread."""

        def shutdown(self) -> None:
            """Accept rollout shutdown."""

        def set_model(self, model: torch.nn.Module) -> None:
            """Update the fake model reference."""
            self.model = model

    class FakeChannel:
        """Stand-in gRPC channel."""

        def __init__(self, target: str, options: Any = None) -> None:
            """Capture the selected runtime target."""
            del options
            self.target = target

    def fake_channel_ready_future(channel: Any) -> Any:
        """Return an object whose `result(timeout=...)` is a no-op."""
        del channel
        return SimpleNamespace(result=lambda timeout: None)

    monkeypatch.setattr(
        rollout_module,
        "build_inference_engine",
        lambda config: FakeInferenceEngine(),
    )
    monkeypatch.setattr(
        rollout_module, "build_policy_factory", lambda config, engine: object()
    )
    monkeypatch.setattr(rollout_module.socket, "gethostname", lambda: "worker-host")
    monkeypatch.setattr(rollout_module, "EgodriverServer", FakeDriverServer)
    monkeypatch.setattr(rollout_module, "FileTopologyRegistry", FakeRegistry)
    monkeypatch.setattr(rollout_module, "StreamingRolloutWorker", FakeStreamingWorker)
    monkeypatch.setattr(rollout_module, "RuntimeServiceStub", lambda channel: channel)
    monkeypatch.setattr(rollout_module.grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(
        rollout_module.grpc, "channel_ready_future", fake_channel_ready_future
    )

    config = SimpleNamespace(custom={"resolved_config_path": str(resolved_config_path)})
    rollout = RolloutRegistry.get_rollout_cls("alpagym_rollout")(config=config)
    try:
        rollout.init_engine(quantization="none", seed=0, load_format="auto")

        assert FakeRegistry.acquired_driver_ids == [FakeDriverServer.instances[0].name]
        assert FakeDriverServer.instances[0].name.startswith("driver-worker-host-pid-")
        assert FakeDriverServer.instances[0].publish_host == "worker-host"
        assert FakeDriverServer.instances[0].max_concurrent_rollouts == 3
        assert FakeStreamingWorker.instances[0].max_concurrent_rollouts == 3
        assert FakeRegistry.published_drivers == [
            FakeDriverServer.instances[0].topology_endpoint
        ]
        assert (
            FakeStreamingWorker.instances[0].alpasim_runtime_stub.target
            == "runtime-1.local:6102"
        )
    finally:
        rollout.shutdown()


def _alpamayo_test_jpeg() -> bytes:
    """Encode a single-color RGB image as JPEG bytes for the Alpamayo rollout test."""
    image = Image.new("RGB", (8, 6), color=(123, 45, 67))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _alpamayo_test_calibration() -> Any:
    """Single-camera calibration matching the Alpamayo test policy config."""
    from alpagym_runtime.types import CameraCalibration, CameraIntrinsics, Pose

    return (
        CameraCalibration(
            name=_ALPAMAYO_TEST_CAMERA,
            logical_id=_ALPAMAYO_TEST_CAMERA,
            intrinsics=CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0),
            extrinsic_pose=Pose(),
        ),
    )


def _alpamayo_test_policy_input() -> Any:
    """Build a one-camera `PolicyInput` for the Alpamayo rollout test."""
    from alpagym_runtime.inference.types import NUM_ROUTE_WAYPOINTS
    from alpagym_runtime.types import (
        CameraImage,
        EgoPose,
        PolicyInput,
        Pose,
        RouteWaypoint,
        Trajectory,
    )

    nan = float("nan")
    live_waypoints = (RouteWaypoint(x=5.0, y=0.0), RouteWaypoint(x=10.0, y=1.0))
    route_waypoints = live_waypoints + tuple(
        RouteWaypoint(x=nan, y=nan, z=nan)
        for _ in range(NUM_ROUTE_WAYPOINTS - len(live_waypoints))
    )

    return PolicyInput(
        step_index=0,
        time_now_us=1_000_000,
        time_query_us=1_200_000,
        camera_images=(
            CameraImage(
                logical_id=_ALPAMAYO_TEST_CAMERA,
                image_bytes=_alpamayo_test_jpeg(),
                frame_end_us=1_000_000,
            ),
        ),
        ego_trajectory=Trajectory(
            poses=(
                EgoPose(timestamp_us=900_000, pose=Pose()),
                EgoPose(timestamp_us=1_000_000, pose=Pose()),
            )
        ),
        route_waypoints=route_waypoints,
        route_timestamp_us=1_000_000,
        calibration=_alpamayo_test_calibration(),
    )


def test_alpamayo_rollout_model_param_map_deduplicates_tied_parameters() -> None:
    """R2R sync should use the same unique parameter surface as P2R sync."""
    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    rollout = rollout_module.AlpagymRollout.__new__(rollout_module.AlpagymRollout)
    rollout._engine_initialized = True
    rollout._model_param_map = None

    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Linear(2, 2, bias=False)
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    model.lm_head.weight = model.embed_tokens.weight
    rollout._model = model

    class FakeWeightMapper:
        """Weight mapper stand-in for the Cosmos rollout parameter-map path."""

        @staticmethod
        def rollout_map_local_key_to_hf_key(name: str) -> str:
            """Leave fake parameter names unchanged."""
            return name

    assert set(model.state_dict()) == {"embed_tokens.weight", "lm_head.weight"}
    assert list(rollout.model_param_map(FakeWeightMapper())) == ["embed_tokens.weight"]
