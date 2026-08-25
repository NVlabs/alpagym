# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from pathlib import Path

import pytest
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint


def test_file_topology_registry_lists_and_assigns_runtime_endpoints(
    tmp_path: Path,
) -> None:
    """Registry returns all runtimes and balances driver assignments."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_alpasim_runtime(
        TopologyEndpoint("alpasim-runtime-0", "node-a", 30051, 100)
    )
    registry.publish_alpasim_runtime(
        TopologyEndpoint("alpasim-runtime-1", "node-b", 30051, 1)
    )

    endpoints = registry.list_alpasim_runtimes()

    assert [endpoint.id for endpoint in endpoints] == [
        "alpasim-runtime-0",
        "alpasim-runtime-1",
    ]
    assert registry.acquire_alpasim_runtime(driver_id="driver-a").host == "node-a"
    assert registry.acquire_alpasim_runtime(driver_id="driver-b").host == "node-b"
    assert registry.acquire_alpasim_runtime(driver_id="driver-c").host == "node-a"
    assert registry.acquire_alpasim_runtime(driver_id="driver-a").host == "node-a"


def test_file_topology_registry_honors_gpu_local_runtime_affinity(
    tmp_path: Path,
) -> None:
    """A rollout cell is assigned to its explicitly pinned AlpaSim runtime."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_alpasim_runtime(
        TopologyEndpoint("alpasim-runtime-0", "node", 6000, 1)
    )
    registry.publish_alpasim_runtime(
        TopologyEndpoint("alpasim-runtime-1", "node", 6100, 1)
    )

    endpoint = registry.acquire_alpasim_runtime(
        driver_id="rollout-cell-1",
        preferred_runtime_id="alpasim-runtime-1",
    )

    assert endpoint.id == "alpasim-runtime-1"


def test_nccl_master_round_trips(tmp_path: Path) -> None:
    """The published NCCL master endpoint reads back as (host, port)."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_nccl_master(host="controller-node", port=29501)
    assert registry.read_nccl_master(timeout_s=5.0) == ("controller-node", 29501)


def test_read_nccl_master_times_out_when_unpublished(tmp_path: Path) -> None:
    """Reading the NCCL master raises once the timeout elapses with no endpoint."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    with pytest.raises(RuntimeError, match="not published"):
        registry.read_nccl_master(timeout_s=0.2)


def test_read_nccl_master_polls_until_published(tmp_path: Path) -> None:
    """The reader waits through an initial absence and returns once the master appears."""
    registry = FileTopologyRegistry(tmp_path / "topology")

    def _publish_after_delay() -> None:
        time.sleep(0.3)
        registry.publish_nccl_master(host="controller-node", port=29501)

    publisher = threading.Thread(target=_publish_after_delay)
    publisher.start()
    try:
        # The endpoint is absent when the read begins, so a non-polling reader would
        # fail; returning the published value proves the poll loop ran past the gap.
        assert registry.read_nccl_master(timeout_s=5.0) == ("controller-node", 29501)
    finally:
        publisher.join()


@pytest.mark.parametrize(
    "endpoint_yaml",
    [
        "host: null\nport: 29501\n",
        "host: controller-node\nport: true\n",
    ],
)
def test_read_nccl_master_rejects_malformed_endpoint(
    tmp_path: Path,
    endpoint_yaml: str,
) -> None:
    """A malformed endpoint fails at the registry boundary instead of coercing."""
    topology = tmp_path / "topology"
    topology.mkdir(parents=True)
    (topology / "nccl_master.yaml").write_text(endpoint_yaml, encoding="utf-8")
    registry = FileTopologyRegistry(topology)
    with pytest.raises(TypeError):
        registry.read_nccl_master(timeout_s=1.0)
