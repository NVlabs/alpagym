# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ExecutionBackend,
    SeparateNodesSlurmTopologyConfig,
    TrainerAndRolloutCellsSlurmTopologyConfig,
)
from alpagym_host.run_topology import (
    CosmosWorkerPlan,
    RunHostPlan,
    build_local_topology,
    build_slurm_topology,
)


def test_build_local_topology_creates_one_logical_host() -> None:
    """Local execution has one logical host with both workload roles."""
    topology = build_local_topology()
    local_host = topology.hosts[0]

    assert topology.hosts == (
        RunHostPlan(
            hostname="localhost",
            host_index=0,
            runs_cosmos=True,
            runs_alpasim=True,
            cosmos_gpus=0,
            alpasim_gpus=0,
        ),
    )
    assert topology.cosmos_host_plans == (local_host,)
    assert topology.alpasim_host_plans == (local_host,)


def test_build_slurm_topology_splits_one_node_between_cosmos_and_alpasim() -> None:
    """All-in-one Slurm runs reserve tail GPUs for AlpaSim on the same host."""
    topology = build_slurm_topology(
        backend=ExecutionBackend.slurm,
        hostnames=["single-0"],
        gpus_per_node=8,
        topology=AllInOneSlurmTopologyConfig(alpasim_gpus=4),
        policy_replicas=1,
        rollout_replicas=3,
    )

    assert topology.cosmos_hosts == ("single-0",)
    assert topology.alpasim_hosts == ("single-0",)
    assert topology.hosts[0].cosmos_gpu_count == 4
    assert topology.hosts[0].cosmos_gpu_ids == (0, 1, 2, 3)
    assert topology.hosts[0].alpasim_gpu_ids == (4, 5, 6, 7)


def test_build_slurm_topology_separates_cosmos_and_alpasim_nodes() -> None:
    """Separate-node Slurm runs keep Cosmos and AlpaSim hosts disjoint."""
    topology = build_slurm_topology(
        backend=ExecutionBackend.slurm,
        hostnames=["cosmos-0", "cosmos-1", "alpasim-0"],
        gpus_per_node=8,
        topology=SeparateNodesSlurmTopologyConfig(cosmos_nodes=2, alpasim_nodes=1),
        policy_replicas=1,
        rollout_replicas=2,
    )

    assert topology.cosmos_hosts == ("cosmos-0", "cosmos-1")
    assert topology.alpasim_hosts == ("alpasim-0",)
    assert topology.hosts[0].cosmos_gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert topology.hosts[0].alpasim_gpu_ids == ()
    assert topology.hosts[1].cosmos_gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert topology.hosts[1].alpasim_gpu_ids == ()
    assert topology.hosts[2].cosmos_gpu_ids == ()
    assert topology.hosts[2].alpasim_gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)


def test_build_slurm_topology_creates_gpu_local_rollout_cells() -> None:
    """Each rollout worker shares one physical GPU with its pinned AlpaSim runtime."""
    topology = build_slurm_topology(
        backend=ExecutionBackend.slurm,
        hostnames=["cell-0"],
        gpus_per_node=8,
        topology=TrainerAndRolloutCellsSlurmTopologyConfig(),
        policy_replicas=2,
        rollout_replicas=6,
    )

    assert topology.cosmos_host_plans[0].cosmos_workers == (
        CosmosWorkerPlan(gpu_ids=(0,), global_worker_index=0),
        CosmosWorkerPlan(gpu_ids=(1,), global_worker_index=1),
        *(
            CosmosWorkerPlan(
                gpu_ids=(gpu_id,),
                global_worker_index=gpu_id,
                alpasim_runtime_id=f"alpasim-runtime-{gpu_id - 2}",
            )
            for gpu_id in range(2, 8)
        ),
    )
    assert [host.alpasim_gpu_ids for host in topology.alpasim_host_plans] == [
        (gpu_id,) for gpu_id in range(2, 8)
    ]
