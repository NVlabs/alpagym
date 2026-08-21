# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ExecutionBackend,
    SeparateNodesSlurmTopologyConfig,
    SlurmLayout,
    SlurmTopologyConfig,
    TrainerAndRolloutCellsSlurmTopologyConfig,
)


@dataclass(frozen=True)
class CosmosWorkerPlan:
    """GPU placement and runtime affinity for one Cosmos worker."""

    gpu_ids: tuple[int, ...]
    global_worker_index: int
    alpasim_runtime_id: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid worker indices and GPU assignments."""
        if not self.gpu_ids:
            raise ValueError("Cosmos worker plan requires at least one GPU")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("Cosmos worker plan GPU ids must be unique")
        if self.global_worker_index < 0:
            raise ValueError("Cosmos global worker index must be non-negative")


@dataclass(frozen=True)
class RunHostPlan:
    """Workload role and GPU allocation for one logical run host."""

    hostname: str
    host_index: int
    runs_cosmos: bool
    runs_alpasim: bool
    cosmos_gpus: int
    alpasim_gpus: int
    cosmos_workers: tuple[CosmosWorkerPlan, ...] = ()
    alpasim_gpu_start: int | None = None

    def __post_init__(self) -> None:
        """Validate explicit worker and AlpaSim GPU placement."""
        if self.cosmos_workers and not self.runs_cosmos:
            raise ValueError("Cosmos worker plans require runs_cosmos=True")
        for worker in self.cosmos_workers:
            if not set(worker.gpu_ids) <= set(self.cosmos_gpu_ids):
                raise ValueError(
                    "Cosmos worker GPUs must be assigned to the host's Cosmos pool"
                )
        if self.alpasim_gpu_start is not None and not self.runs_alpasim:
            raise ValueError("alpasim_gpu_start requires runs_alpasim=True")

    @property
    def cosmos_gpu_count(self) -> int:
        """Return the number of GPUs assigned to Cosmos work."""
        return self.cosmos_gpus

    @property
    def cosmos_gpu_ids(self) -> tuple[int, ...]:
        """Return contiguous GPU ids assigned to Cosmos work."""
        return tuple(range(self.cosmos_gpu_count))

    @property
    def alpasim_gpu_ids(self) -> tuple[int, ...]:
        """Return contiguous GPU ids assigned to AlpaSim work."""
        alpasim_start = (
            self.alpasim_gpu_start
            if self.alpasim_gpu_start is not None
            else self.cosmos_gpu_count
            if self.runs_cosmos
            else 0
        )
        return tuple(range(alpasim_start, alpasim_start + self.alpasim_gpus))


@dataclass(frozen=True)
class RunTopologyPlan:
    """Expanded host topology for a run backend."""

    hosts: tuple[RunHostPlan, ...]

    def __post_init__(self) -> None:
        """Require consecutive global Cosmos worker indices."""
        worker_indices = [
            worker.global_worker_index
            for host in self.cosmos_host_plans
            for worker in host.cosmos_workers
        ]
        if worker_indices and sorted(worker_indices) != list(
            range(len(worker_indices))
        ):
            raise ValueError(
                "Cosmos global worker indices must form a zero-based sequence"
            )

    @property
    def cosmos_host_plans(self) -> tuple[RunHostPlan, ...]:
        """Return host plans that run Cosmos work."""
        return tuple(host for host in self.hosts if host.runs_cosmos)

    @property
    def alpasim_host_plans(self) -> tuple[RunHostPlan, ...]:
        """Return host plans that run AlpaSim work."""
        return tuple(host for host in self.hosts if host.runs_alpasim)

    @property
    def cosmos_hosts(self) -> tuple[str, ...]:
        """Return hostnames that run Cosmos work."""
        return tuple(host.hostname for host in self.cosmos_host_plans)

    @property
    def alpasim_hosts(self) -> tuple[str, ...]:
        """Return hostnames that run AlpaSim work."""
        return tuple(host.hostname for host in self.alpasim_host_plans)


def build_local_topology() -> RunTopologyPlan:
    """Return the single logical host used by local execution."""
    return RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname="localhost",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=True,
                # Local subprocesses do not bind GPUs through the topology plan.
                cosmos_gpus=0,
                alpasim_gpus=0,
            ),
        )
    )


def build_slurm_topology(
    backend: ExecutionBackend | str,
    hostnames: list[str],
    gpus_per_node: int,
    topology: SlurmTopologyConfig,
    policy_replicas: int,
    rollout_replicas: int,
) -> RunTopologyPlan:
    """Expand Slurm layout settings into per-host run topology.

    Args:
        backend: Slurm execution backend.
        hostnames: Slurm hostnames in allocation order.
        gpus_per_node: Number of GPUs available on each Slurm node.
        topology: Slurm host topology settings.
        policy_replicas: Number of trainer replicas.
        rollout_replicas: Number of rollout replicas.

    Returns:
        Per-host topology with Cosmos GPUs before AlpaSim GPUs on each host.

    Raises:
        ValueError: The backend, layout, host count, or GPU counts are invalid.
    """
    execution_backend = ExecutionBackend(backend)
    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be at least 1")
    if execution_backend is not ExecutionBackend.slurm:
        raise ValueError("backend must be a Slurm backend")

    match SlurmLayout(topology.kind):
        case SlurmLayout.all_in_one:
            if not isinstance(topology, AllInOneSlurmTopologyConfig):
                raise TypeError(type(topology))
            return _build_all_in_one_slurm_topology(
                hostnames=hostnames,
                gpus_per_node=gpus_per_node,
                alpasim_gpus=topology.alpasim_gpus,
            )
        case SlurmLayout.separate_nodes:
            if not isinstance(topology, SeparateNodesSlurmTopologyConfig):
                raise TypeError(type(topology))
            return _build_separate_nodes_slurm_topology(
                hostnames=hostnames,
                gpus_per_node=gpus_per_node,
                cosmos_nodes=topology.cosmos_nodes,
                alpasim_nodes=topology.alpasim_nodes,
            )
        case SlurmLayout.trainer_and_rollout_cells:
            if not isinstance(topology, TrainerAndRolloutCellsSlurmTopologyConfig):
                raise TypeError(type(topology))
            return _build_trainer_and_rollout_cells_slurm_topology(
                hostnames=hostnames,
                gpus_per_node=gpus_per_node,
                policy_replicas=policy_replicas,
                rollout_replicas=rollout_replicas,
            )


def _build_all_in_one_slurm_topology(
    hostnames: list[str],
    gpus_per_node: int,
    alpasim_gpus: int,
) -> RunTopologyPlan:
    """Build a one-node topology that colocates Cosmos and AlpaSim."""
    if len(hostnames) != 1:
        raise ValueError("all_in_one requires exactly one hostname")
    if alpasim_gpus < 1:
        raise ValueError("all_in_one requires at least one AlpaSim GPU")
    if alpasim_gpus >= gpus_per_node:
        raise ValueError(
            "all_in_one requires alpasim_gpus to leave at least one Cosmos GPU"
        )

    cosmos_gpus = gpus_per_node - alpasim_gpus
    return RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname=hostnames[0],
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=True,
                cosmos_gpus=cosmos_gpus,
                alpasim_gpus=alpasim_gpus,
                cosmos_workers=(
                    CosmosWorkerPlan(
                        gpu_ids=tuple(range(cosmos_gpus)),
                        global_worker_index=0,
                    ),
                ),
            ),
        )
    )


def _build_separate_nodes_slurm_topology(
    hostnames: list[str],
    gpus_per_node: int,
    cosmos_nodes: int,
    alpasim_nodes: int,
) -> RunTopologyPlan:
    """Build a topology with disjoint full-node Cosmos and AlpaSim hosts."""
    if cosmos_nodes < 1:
        raise ValueError("separate_nodes requires at least one Cosmos node")
    if alpasim_nodes < 1:
        raise ValueError("separate_nodes requires at least one AlpaSim node")
    if len(hostnames) != cosmos_nodes + alpasim_nodes:
        raise ValueError(
            "separate_nodes host count must match cosmos_nodes + alpasim_nodes"
        )

    hosts: list[RunHostPlan] = []
    for host_index, hostname in enumerate(hostnames[:cosmos_nodes]):
        hosts.append(
            RunHostPlan(
                hostname=hostname,
                host_index=host_index,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=gpus_per_node,
                alpasim_gpus=0,
                cosmos_workers=(
                    CosmosWorkerPlan(
                        gpu_ids=tuple(range(gpus_per_node)),
                        global_worker_index=host_index,
                    ),
                ),
            )
        )
    for host_index, hostname in enumerate(
        hostnames[cosmos_nodes:],
        start=cosmos_nodes,
    ):
        hosts.append(
            RunHostPlan(
                hostname=hostname,
                host_index=host_index,
                runs_cosmos=False,
                runs_alpasim=True,
                cosmos_gpus=0,
                alpasim_gpus=gpus_per_node,
            )
        )
    return RunTopologyPlan(hosts=tuple(hosts))


def _build_trainer_and_rollout_cells_slurm_topology(
    hostnames: list[str],
    gpus_per_node: int,
    policy_replicas: int,
    rollout_replicas: int,
) -> RunTopologyPlan:
    """Place trainers first, then one rollout worker and AlpaSim runtime per GPU."""
    if len(hostnames) != 1:
        raise ValueError("trainer_and_rollout_cells requires exactly one hostname")
    if policy_replicas < 1:
        raise ValueError(
            "trainer_and_rollout_cells requires at least one policy replica"
        )
    if policy_replicas + rollout_replicas != gpus_per_node:
        raise ValueError(
            "trainer_and_rollout_cells requires one Cosmos worker per GPU; "
            f"got {policy_replicas} policy and {rollout_replicas} rollout replicas "
            f"for {gpus_per_node} GPUs"
        )

    hostname = hostnames[0]
    trainer_workers = tuple(
        CosmosWorkerPlan(gpu_ids=(gpu_id,), global_worker_index=gpu_id)
        for gpu_id in range(policy_replicas)
    )
    rollout_workers = tuple(
        CosmosWorkerPlan(
            gpu_ids=(gpu_id,),
            global_worker_index=gpu_id,
            alpasim_runtime_id=f"alpasim-runtime-{runtime_index}",
        )
        for runtime_index, gpu_id in enumerate(range(policy_replicas, gpus_per_node))
    )
    cosmos_host = RunHostPlan(
        hostname=hostname,
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=False,
        cosmos_gpus=gpus_per_node,
        alpasim_gpus=0,
        cosmos_workers=trainer_workers + rollout_workers,
    )
    alpasim_hosts = tuple(
        RunHostPlan(
            hostname=hostname,
            host_index=0,
            runs_cosmos=False,
            runs_alpasim=True,
            cosmos_gpus=0,
            alpasim_gpus=1,
            alpasim_gpu_start=gpu_id,
        )
        for gpu_id in range(policy_replicas, gpus_per_node)
    )
    return RunTopologyPlan(hosts=(cosmos_host, *alpasim_hosts))
