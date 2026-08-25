# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TopologyEndpoint:
    """Published service endpoint in the AlpaSim topology."""

    id: str
    host: str
    port: int
    capacity: int

    def __str__(self) -> str:
        """Return the human-readable endpoint identity for logs."""
        return f"{self.id} at {self.to_grpc_target()} capacity={self.capacity}"

    def to_grpc_target(self) -> str:
        """Return the gRPC channel target for this endpoint."""
        return f"{self.host}:{self.port}"


def rollout_worker_capacity(
    runtime_capacity: int,
    rollout_replicas: int,
    alpasim_runtime_count: int,
) -> int:
    """Return one rollout worker's local concurrency for an assigned AlpaSim runtime.

    Computes:
      ceil(capacity / rollout_replicas_per_alpasim)
      where `rollout_replicas_per_alpasim` underestimates the count when
      division has a remainder.

    This overestimates capacity if either capacity cannot be cleanly divided
    among rollout workers, or AlpaSim runtimes cannot be cleanly divided among
    rollout replicas.
    """
    return math.ceil(
        runtime_capacity / max(1, rollout_replicas // alpasim_runtime_count)
    )


class FileTopologyRegistry:
    """File-backed registry of run-dir rendezvous endpoints.

    Brokers the endpoints a run's processes use to find each other on a local or
    shared filesystem: AlpaSim runtime services, egodrivers, and the NCCL
    ``TCPStore`` master.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        """Store the topology registry directory."""
        self._registry_dir = Path(registry_dir)

    def publish_alpasim_runtime(self, endpoint: TopologyEndpoint) -> None:
        """Publish one AlpaSim runtime endpoint."""
        path: Path = self._registry_dir / "alpasim_runtimes" / f"{endpoint.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(yaml.safe_dump(asdict(endpoint), sort_keys=False))

    def publish_driver(self, endpoint: TopologyEndpoint) -> None:
        """Publish one Egodriver endpoint."""
        path: Path = self._registry_dir / "drivers" / f"{endpoint.id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(yaml.safe_dump(asdict(endpoint), sort_keys=False))

    def publish_nccl_master(self, host: str, port: int) -> None:
        """Publish the singleton NCCL ``TCPStore`` master endpoint for peer workers.

        Written atomically (temporary file + rename): Policy and Rollout workers
        poll this file while the Controller writes it, so the rename guarantees a
        reader sees either no file or the complete endpoint, never a partial write.
        """
        path = self._registry_dir / "nccl_master.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".yaml.tmp")
        tmp_path.write_text(
            yaml.safe_dump({"host": host, "port": port}, sort_keys=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def read_nccl_master(self, timeout_s: float) -> tuple[str, int]:
        """Wait up to ``timeout_s`` for the Controller to publish the NCCL master endpoint.

        Unlike ``list_alpasim_runtimes`` (read immediately, since the host
        pre-publishes runtimes before the cosmos workers launch), the master is
        published live by a peer process, so discovery polls until the file
        appears or the timeout expires.

        Args:
            timeout_s: How long to wait for the endpoint, in seconds.

        Returns:
            The published ``(host, port)``.
        """
        path = self._registry_dir / "nccl_master.yaml"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                time.sleep(0.1)
                continue
            data = yaml.safe_load(raw)
            host = data["host"]
            port = data["port"]
            if not isinstance(host, str):
                raise TypeError(f"NCCL master host must be a string, got {host!r}")
            if isinstance(port, bool) or not isinstance(port, int):
                raise TypeError(f"NCCL master port must be an int, got {port!r}")
            return host, port
        raise RuntimeError(
            f"NCCL master endpoint was not published at {path} within {timeout_s}s"
        )

    def clear_nccl_master(self) -> None:
        """Remove any stale NCCL master endpoint before the Controller republishes.

        A Slurm requeue reuses the run directory, so a previous attempt's
        ``nccl_master.yaml`` can linger. Clearing it before the new Controller
        binds means a worker that polls during startup waits for the fresh
        publish instead of connecting to the previous attempt's dead ``TCPStore``.
        """
        (self._registry_dir / "nccl_master.yaml").unlink(missing_ok=True)

    def list_alpasim_runtimes(self) -> list[TopologyEndpoint]:
        """Return all published AlpaSim RuntimeService endpoints."""
        alpasim_runtime_paths = sorted(
            (self._registry_dir / "alpasim_runtimes").glob("*.yaml")
        )
        endpoints: list[TopologyEndpoint] = []
        for path in alpasim_runtime_paths:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            endpoints.append(
                TopologyEndpoint(
                    id=data["id"],
                    host=data["host"],
                    port=data["port"],
                    capacity=data["capacity"],
                )
            )
        return endpoints

    def acquire_alpasim_runtime(
        self,
        driver_id: str,
        preferred_runtime_id: str | None = None,
    ) -> TopologyEndpoint:
        """Return the assigned runtime, honoring explicit GPU-local affinity."""
        lock_path = self._registry_dir / "alpasim_assignments.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            return self._acquire_alpasim_runtime_locked(
                driver_id=driver_id,
                preferred_runtime_id=preferred_runtime_id,
            )

    def _acquire_alpasim_runtime_locked(
        self,
        driver_id: str,
        preferred_runtime_id: str | None,
    ) -> TopologyEndpoint:
        """Assign a driver under the requested affinity while holding the lock."""
        endpoints = self.list_alpasim_runtimes()
        if not endpoints:
            raise RuntimeError(f"No AlpaSim runtime endpoints in {self._registry_dir}")

        assignment_dir = self._registry_dir / "alpasim_assignments"
        assignment_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = assignment_dir / f"{driver_id}.yaml"
        endpoints_by_id = {endpoint.id: endpoint for endpoint in endpoints}

        if assignment_path.exists():
            data = yaml.safe_load(assignment_path.read_text(encoding="utf-8"))
            runtime_id = data["alpasim_runtime_id"]
            if runtime_id not in endpoints_by_id:
                raise RuntimeError(
                    f"Driver {driver_id!r} is assigned to unknown AlpaSim runtime {runtime_id!r}"
                )
            if preferred_runtime_id is not None and runtime_id != preferred_runtime_id:
                raise RuntimeError(
                    f"Driver {driver_id!r} is assigned to {runtime_id!r}, "
                    f"not required runtime {preferred_runtime_id!r}"
                )
            return endpoints_by_id[runtime_id]

        assignment_counts = {endpoint.id: 0 for endpoint in endpoints}
        for path in sorted(assignment_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            runtime_id = data["alpasim_runtime_id"]
            if runtime_id in assignment_counts:
                assignment_counts[runtime_id] += 1

        if preferred_runtime_id is not None:
            if preferred_runtime_id not in endpoints_by_id:
                raise RuntimeError(
                    f"Driver {driver_id!r} requires unknown AlpaSim runtime "
                    f"{preferred_runtime_id!r}"
                )
            endpoint = endpoints_by_id[preferred_runtime_id]
        else:
            endpoint = min(
                endpoints,
                key=lambda candidate: (assignment_counts[candidate.id], candidate.id),
            )
        assignment_path.write_text(
            yaml.safe_dump(
                {"driver_id": driver_id, "alpasim_runtime_id": endpoint.id},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return endpoint
