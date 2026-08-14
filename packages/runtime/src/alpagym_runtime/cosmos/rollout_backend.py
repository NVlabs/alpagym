# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Cosmos-RL rollout backend driving AlpaSim simulator sessions."""

import atexit
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any

import grpc
import torch
import yaml
from alpagym_runtime.alpasim.grpc_import import ensure_alpasim_grpc_source

ensure_alpasim_grpc_source()
from alpagym_host.config import ExecutionBackend, load_run_config
from alpagym_host.endpoint_registry import (
    FileTopologyRegistry,
    TopologyEndpoint,
    rollout_worker_capacity,
)
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult

from alpagym_runtime.alpasim.driver_server import EgodriverServer
from alpagym_runtime.alpasim.humanoid_policy_server import HumanoidPolicyServer
from alpagym_runtime.episode_runner.streaming_worker import StreamingRolloutWorker
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.perf.instrument.lifecycle import initialize_perf
from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.scope import measure_perf
from alpagym_runtime.policies.factory import (
    build_humanoid_policy_factory,
    build_inference_engine,
    build_policy_factory,
)

logger = logging.getLogger(__name__)

_MAX_GRPC_MSG_SIZE = 256 * 1024 * 1024  # 256 MiB; matches AlpaSim runtime defaults.


@RolloutRegistry.register("alpagym_rollout", allow_override=True)
class AlpagymRollout(RolloutBase):
    """Cosmos-RL rollout backend that executes simulator sessions through AlpaSim."""

    # Cosmos may call `set_underlying_model` before `post_init_hook`, so the
    # init-state flag is declared at class scope and read directly.
    _model: torch.nn.Module | None = None
    _engine_initialized: bool = False

    def post_init_hook(self, **kwargs: Any) -> None:
        """Load the resolved alpagym config and topology registry."""
        del kwargs
        self._run_config = load_run_config(Path(self.config.custom["resolved_config_path"]))
        initialize_perf(self._run_config)
        self._topology_registry = FileTopologyRegistry(
            self._run_config.artifact_paths.topology_registry_dir
        )
        self._model_param_map = None
        # Components built lazily in `init_engine`. None sentinels so
        # `shutdown` no-ops cleanly on a partial init failure.
        self._inference_engine: InferenceEngine | None = None
        self._driver_server: EgodriverServer | None = None
        self._humanoid_policy_server: HumanoidPolicyServer | None = None
        self._alpasim_runtime_stub: RuntimeServiceStub | None = None
        self._worker: StreamingRolloutWorker | None = None
        self._engine_thread: threading.Thread | None = None
        self._engine_initialized = False
        self._shutdown_done = False

    @measure_perf("rollout/engine_init", category="orchestration", cpu_snapshot=True)
    def init_engine(
        self,
        quantization: str,
        seed: int,
        load_format: str,
        **kwargs: Any,
    ) -> None:
        """Build the inference engine, driver, streaming worker, and engine thread."""
        del quantization, seed, load_format, kwargs

        if self._engine_initialized:
            return

        rollouts_per_payload = int(self._run_config.cosmos.rollout.n_generation)
        scene_id_data: dict[str, list[str]] = yaml.safe_load(
            self._run_config.artifact_paths.alpasim_scene_ids_path.read_text(encoding="utf-8")
        )
        scene_ids = tuple(str(scene_id) for scene_id in scene_id_data["scene_ids"])

        self._inference_engine = build_inference_engine(self._run_config)
        self._model = self._inference_engine.get_model()
        record_perf_marker("rollout/model_ready", cpu_snapshot=True, gpu_snapshot=True)
        distributed = ExecutionBackend(self._run_config.execution.backend).is_slurm_run
        simulation_domain = str(getattr(self._run_config.alpasim, "simulation_domain", "av"))

        if simulation_domain == "av":
            policy_endpoint_id = f"driver-{socket.gethostname()}-pid-{os.getpid()}"
        else:
            policy_endpoint_id = f"{simulation_domain}-policy-{socket.gethostname()}-pid-{os.getpid()}"
        alpasim_runtime_endpoint: TopologyEndpoint = (
            self._topology_registry.acquire_alpasim_runtime(driver_id=policy_endpoint_id)
        )
        max_concurrent_rollouts = rollout_worker_capacity(
            runtime_capacity=int(alpasim_runtime_endpoint.capacity),
            rollout_replicas=int(self._run_config.cosmos.launch.rollout_replicas),
            alpasim_runtime_count=len(self._topology_registry.list_alpasim_runtimes()),
        )
        publish_host = socket.gethostname() if distributed else "localhost"
        if simulation_domain == "humanoid":
            humanoid_policy_factory = build_humanoid_policy_factory(
                self._run_config,
                self._inference_engine,
            )
            self._humanoid_policy_server = HumanoidPolicyServer(
                name=policy_endpoint_id,
                max_concurrent_rollouts=max_concurrent_rollouts,
                policy_factory=humanoid_policy_factory,
                publish_host=publish_host,
            )
            self._humanoid_policy_server.start()
        elif simulation_domain == "av":
            policy_factory = build_policy_factory(self._run_config, self._inference_engine)
            self._driver_server = EgodriverServer(
                name=policy_endpoint_id,
                max_concurrent_rollouts=max_concurrent_rollouts,
                policy_factory=policy_factory,
                publish_host=publish_host,
            )
            self._driver_server.start()
            self._topology_registry.publish_driver(self._driver_server.topology_endpoint)
        else:
            raise ValueError(f"unsupported alpasim.simulation_domain={simulation_domain!r}")

        channel = grpc.insecure_channel(
            alpasim_runtime_endpoint.to_grpc_target(),
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MSG_SIZE),
                ("grpc.max_receive_message_length", _MAX_GRPC_MSG_SIZE),
            ],
        )
        grpc.channel_ready_future(channel).result(timeout=5.0)
        self._alpasim_runtime_stub = RuntimeServiceStub(channel)

        # TODO(cosmos-rl): the right resolution path is
        # `data_packer.get_rollout_input(prompt_idx)`, but cosmos-rl's
        # `_prefetch_loop` does not forward `data_packer` to backends. Index
        # `scene_ids` by `prompt_idx` directly for now — correct for the
        # current `AlpagymDataPacker.get_rollout_input` mapping (identity).
        # A non-identity packer
        # needs cosmos-rl to thread `data_packer` into the prefetch path.
        def _scene_id_for(payload: RLPayload) -> str:
            """Return the dataset scene id for one cosmos-rl payload."""
            return scene_ids[int(payload.prompt_idx)]

        # The worker produces in-memory EpisodeOutputs; egress to the transport
        # happens later in the packer's get_rollout_output, after reward + DAPO.
        worker_kwargs: dict[str, Any] = {
            "alpasim_runtime_stub": self._alpasim_runtime_stub,
            "driver_server": self._driver_server,
            "simulation_timeout_s": float(self._run_config.alpasim.simulation_timeout_s),
            "reward_config": self._run_config.reward,
            "max_concurrent_rollouts": max_concurrent_rollouts,
            "rollouts_per_payload": rollouts_per_payload,
            "scene_id_resolver": _scene_id_for,
        }
        if simulation_domain == "humanoid":
            worker_kwargs.update(
                humanoid_policy_server=self._humanoid_policy_server,
                simulation_domain=simulation_domain,
            )
        self._worker = StreamingRolloutWorker(**worker_kwargs)

        # daemon=True so the engine thread does not block Python exit on a
        # clean cosmos shutdown. Cosmos-RL's colocated mode does not call
        # `RolloutBase.shutdown()`, so without the atexit hook below we would
        # have to rely on daemon-thread finalization to kill the engine,
        # which races against the ThreadPoolExecutor's own atexit cleanup.
        # Registering `shutdown` here drains the simulate pool, the gRPC
        # server, and the engine queue in the order required to avoid
        # closing the gRPC server before in-flight `drive()` handlers drain.
        self._engine_thread = threading.Thread(
            target=self._inference_engine.run_loop,
            name="alpagym-infer",
            daemon=True,
        )
        self._engine_thread.start()
        atexit.register(self.shutdown)

        self._engine_initialized = True
        record_perf_marker("rollout/backend_ready", cpu_snapshot=True, gpu_snapshot=True)
        logger.info(
            "[alpagym] Streaming rollout backend ready: runtime=%s domain=%s "
            "policy_endpoint=%s max_concurrent_rollouts=%d",
            alpasim_runtime_endpoint,
            simulation_domain,
            (
                self._humanoid_policy_server.topology_endpoint
                if self._humanoid_policy_server is not None
                else self._driver_server.topology_endpoint
            ),
            max_concurrent_rollouts,
        )

    def enqueue_prefetch_payloads(self, payloads: list[RLPayload]) -> None:
        """Cosmos-RL ``_prefetch_loop`` hook; opt-in by name via ``getattr``.

        Forwards each payload to the streaming worker as an early
        ``submit_payload``; the discarded return value is recovered by
        the later matching ``rollout_generation`` call.
        """
        if not self._engine_initialized or self._worker is None:
            logger.info("[Prefetch] skipped (engine not initialized)")
            return
        for payload in payloads:
            self._worker.submit_payload(payload)

    @measure_perf(
        "rollout/generate",
        category="orchestration",
        cpu_snapshot=True,
        gpu_snapshot=True,
    )
    def rollout_generation(
        self,
        payloads: list[RLPayload],
        stream: Any,
        data_packer: Any,
        data_fetcher: Any = None,
        is_validation: bool = False,
        current_weight_version: int | None = None,
    ) -> list[RolloutResult]:
        """Run AlpaSim sessions and return Cosmos-RL rollout results.

        The batch only exists at this boundary: each payload is submitted
        to the streaming worker individually via ``submit_payload``, then
        the per-payload futures are awaited in input order. Completions are
        the in-memory ``EpisodeOutput``s; the reward dispatcher reads
        ``reward.total`` off them and the packer's ``get_rollout_output``
        egresses them to the transport later. No write or handle happens here.
        """
        # `current_weight_version` is forwarded by Cosmos-RL's
        # `_call_rollout_generation` so async weight sync can tag in-flight
        # rollouts; AlpaGym does not consume it. `data_packer` is the egress
        # path used later in `get_rollout_output`, not here. Declaring and
        # deleting the unused kwargs is preferred over a `**kwargs` shim so a
        # new Cosmos kwarg surfaces as a loud TypeError instead of being
        # silently absorbed.
        del stream, data_fetcher, current_weight_version, data_packer
        if self._worker is None:
            raise RuntimeError("rollout_generation called before init_engine")
        logger.info(
            "rollout_generation: %s payloads, is_validation=%s",
            len(payloads),
            is_validation,
        )
        # Submit all before awaiting so simulate jobs run in parallel; the
        # in-order await just preserves cosmos's ordered-return contract.
        payload_states = [self._worker.submit_payload(p) for p in payloads]
        # The streaming worker resolves permanent-failure payloads with `[]`
        # (after `max_scene_retries` failed simulate attempts) instead of
        # raising. Surface that as a hard error here so cosmos sees the
        # failure at the rollout boundary rather than crashing downstream
        # with a shape mismatch when fewer than `n_target` completions reach
        # the advantage computation.
        results: list[RolloutResult] = []
        for payload_state in payload_states:
            episodes = payload_state.future.result()
            if len(episodes) < payload_state.n_target:
                raise RuntimeError(
                    "Rollout payload exhausted retries: "
                    f"prompt_idx={payload_state.payload.prompt_idx} "
                    f"collected={len(episodes)}/{payload_state.n_target}"
                )
            results.append(RolloutResult(completions=list(episodes)))
        return results

    def shutdown(self) -> None:
        """Stop accepting payloads and tear down worker, driver, and engine.

        Ordering: worker first so simulate-pool workers stop issuing new
        ``simulate()`` calls; driver next so AlpaSim's pending ``drive()``
        callbacks fail and let the runtime release in-flight sessions; engine
        sentinel last so ``drive()``-issued inference futures finish draining
        before the engine thread exits. Each component may be `None` if
        `init_engine` failed partway through. The transport endpoint is owned by
        the data packer and closed by the entrypoint's atexit hook, which runs
        after this (LIFO) so the worker stops producing before the writer closes.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._worker is not None:
            self._worker.shutdown()
        if self._driver_server is not None:
            self._driver_server.stop()
        if self._humanoid_policy_server is not None:
            self._humanoid_policy_server.stop()
        if self._inference_engine is not None:
            self._inference_engine.shutdown()
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=30.0)

    def get_underlying_model(self) -> torch.nn.Module:
        """Return the model object Cosmos should use for rollout weight sync."""
        if self._model is None:
            raise RuntimeError("AlpaGym rollout model is not initialized; call init_engine first")
        return self._model

    def model_param_map(self, weight_mapper: Any) -> dict[str, torch.Tensor]:
        """Build the rollout-side parameter map used by Cosmos R2R weight sync."""
        if self._model_param_map:
            return self._model_param_map

        param_map: dict[str, torch.Tensor] = {}
        for name, param in self.get_underlying_model().named_parameters():
            compatible_name = weight_mapper.rollout_map_local_key_to_hf_key(name)
            param_map[compatible_name] = param
        param_map.update(self.get_quantized_tensors(weight_mapper))
        self._model_param_map = param_map
        return self._model_param_map

    def set_underlying_model(self, model: torch.nn.Module) -> None:
        """Replace the model Cosmos weight sync should use for rollout inference."""
        self._model = model
        self._model_param_map = None
        # After init_engine(), keep the inference engine's reference in sync.
        if self._engine_initialized:
            if self._inference_engine is None:
                raise RuntimeError("Rollout inference engine is not initialized")
            self._inference_engine.set_model(model)
