# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from alpagym_host.config import SeparateNodesSlurmTopologyConfig, SlurmConfig
from alpagym_host.run_topology import CosmosWorkerPlan, RunHostPlan, RunTopologyPlan
from alpagym_host.slurm import (
    _gpu_mask,
    build_cosmos_srun_command,
    build_wizard_srun_command,
)


def test_build_wizard_srun_command_leaves_gpu_binding_to_nested_services() -> None:
    """The CPU-only Wizard must not narrow the allocation seen by its services."""
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=1,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=4,
        alpasim_gpus=4,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(),
        wizard_command=["python", "-m", "alpasim.wizard"],
        wizard_workdir=Path("/workspace/alpasim"),
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--nodelist=mixed-0" in command
    assert not any(value.startswith("--gpus-per-task=") for value in command)
    assert not any(value.startswith("--gpu-bind=") for value in command)
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(command)


def test_build_wizard_srun_command_scrubs_uv_project_env_before_exec() -> None:
    """The Wizard runs under `bash -lc`, which re-sources /etc/environment, so the script
    must unset the leaked venv vars before exec to keep them out of the service srun's.
    """
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=1,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=4,
        alpasim_gpus=4,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(),
        wizard_command=["python", "-m", "alpasim.wizard"],
        wizard_workdir=Path("/workspace/alpasim"),
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    script = command[-1]
    assert "unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV" in script
    assert script.index("unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV") < script.index(
        "exec "
    )


def test_build_wizard_srun_command_reapplies_container_home_and_path() -> None:
    """Pyxis/login initialization must not replace the authored runtime paths."""
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=1,
        alpasim_gpus=1,
    )
    slurm = _slurm_config()
    slurm.export_env = [
        "HOME=/lustre/run-home",
        "PATH=/cm/slurm/bin:/opt/venv/bin",
    ]

    command = build_wizard_srun_command(
        host=host,
        slurm=slurm,
        wizard_command=["python", "-m", "alpasim.wizard"],
        wizard_workdir=Path("/workspace/alpasim"),
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert command[-1].startswith(
        "export HOME=/lustre/run-home\n"
        "export PATH=/cm/slurm/bin:/opt/venv/bin\n"
    )


def test_build_wizard_srun_command_disables_cpu_binding_for_nonexclusive_step() -> None:
    """Partial-node Wizard srun steps must not inherit packed CPU binding."""
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=2,
        alpasim_gpus=1,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(exclusive=False),
        wizard_command=["python", "-m", "alpasim.wizard"],
        wizard_workdir=Path("/workspace/alpasim"),
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--overlap" in command
    assert "--cpu-bind=none" in command


def test_build_wizard_srun_command_keeps_default_cpu_binding_for_exclusive_step() -> (
    None
):
    host = RunHostPlan(
        hostname="alpasim-0",
        host_index=0,
        runs_cosmos=False,
        runs_alpasim=True,
        cosmos_gpus=0,
        alpasim_gpus=8,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(exclusive=True),
        wizard_command=["python", "-m", "alpasim.wizard"],
        wizard_workdir=Path("/workspace/alpasim"),
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--cpu-bind=none" not in command


def test_build_cosmos_srun_command_passes_worker_flags_to_cosmos_launcher() -> None:
    """Cosmos multi-worker flags appear before the entrypoint script argument."""
    topology = RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname="policy-0",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=4,
                alpasim_gpus=0,
                cosmos_workers=(CosmosWorkerPlan((0, 1, 2, 3), 0),),
            ),
            RunHostPlan(
                hostname="mixed-0",
                host_index=1,
                runs_cosmos=True,
                runs_alpasim=True,
                cosmos_gpus=4,
                alpasim_gpus=4,
                cosmos_workers=(CosmosWorkerPlan((0, 1, 2, 3), 1),),
            ),
        )
    )

    command = build_cosmos_srun_command(
        cosmos_hosts=topology.cosmos_host_plans,
        slurm=_slurm_config(),
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=[
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--all-packages",
            "--project",
            "/workspace/alpagym",
        ],
        worker_commands=(
            (
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "cosmos_rl.launcher.launch_all",
                    "--config",
                    "/tmp/cosmos.toml",
                    "--num-workers",
                    "2",
                    "--worker-idx",
                    "0",
                    "--port",
                    "29500",
                    "alpagym_runtime.cosmos.entrypoint",
                ],
            ),
            (
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "cosmos_rl.launcher.launch_all",
                    "--config",
                    "/tmp/cosmos.toml",
                    "--num-workers",
                    "2",
                    "--worker-idx",
                    "1",
                    "--url",
                    "policy-0:29500",
                    "alpagym_runtime.cosmos.entrypoint",
                ],
            ),
        ),
        log_dir=Path("/tmp/alpagym/logs"),
    )

    script = command[-1]
    assert script.startswith(
        "uv sync --frozen --inexact --all-packages --project /workspace/alpagym\n"
    )
    script = command[-1].split("  1)", maxsplit=1)[1].split("    ;;", maxsplit=1)[0]
    entrypoint_index = script.index("alpagym_runtime.cosmos.entrypoint")
    for expected_arg in (
        "--num-workers 2",
        "--worker-idx 1",
        "--url policy-0:29500",
    ):
        assert expected_arg in script
        assert script.index(expected_arg) < entrypoint_index
    assert "--gpu-bind=mask_gpu:0xf" in command
    assert "ALPAGYM_WORKER_INDEX" not in command[-1]


def test_build_cosmos_srun_command_reapplies_container_home_and_path() -> None:
    """Cosmos workers retain authored HOME/PATH inside the Pyxis shell."""
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=1,
        alpasim_gpus=1,
        cosmos_workers=(CosmosWorkerPlan((0,), 0),),
    )
    slurm = _slurm_config()
    slurm.export_env = [
        "HOME=/lustre/run-home",
        "PATH=/cm/slurm/bin:/opt/venv/bin",
    ]

    command = build_cosmos_srun_command(
        cosmos_hosts=(host,),
        slurm=slurm,
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=["true"],
        worker_commands=((["uv", "run", "python", "worker.py"],),),
        log_dir=Path("/tmp/alpagym/logs"),
    )

    assert command[-1].startswith(
        "export HOME=/lustre/run-home\n"
        "export PATH=/cm/slurm/bin:/opt/venv/bin\n"
    )


def test_build_cosmos_srun_command_pins_colocated_rollout_cell() -> None:
    """One host task launches GPU-scoped workers with explicit runtime affinity."""
    host = RunHostPlan(
        hostname="cell-0",
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=False,
        cosmos_gpus=2,
        alpasim_gpus=0,
        cosmos_workers=(
            CosmosWorkerPlan((0,), 0),
            CosmosWorkerPlan((1,), 1, "alpasim-runtime-0"),
        ),
    )

    command = build_cosmos_srun_command(
        cosmos_hosts=(host,),
        slurm=_slurm_config(),
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=["true"],
        worker_commands=((["trainer"], ["rollout"]),),
        log_dir=Path("/tmp/alpagym/logs"),
    )

    script = command[-1]
    assert "CUDA_VISIBLE_DEVICES=0 trainer &" in script
    assert (
        "CUDA_VISIBLE_DEVICES=1 ALPAGYM_ALPASIM_RUNTIME_ID=alpasim-runtime-0 rollout &"
    ) in script
    assert "wait -n" in script


def test_gpu_mask_preserves_non_contiguous_gpu_ids() -> None:
    """GPU binding masks preserve sparse GPU id selections."""
    assert _gpu_mask((0, 2, 4)) == "0x15"


def _slurm_config(*, exclusive: bool = True) -> SlurmConfig:
    """Build Slurm settings used by launch command tests."""
    return SlurmConfig(
        job_name="alpagym",
        partition="batch",
        account="av",
        time="02:00:00",
        nodes=2,
        gpus_per_node=8,
        exclusive=exclusive,
        cpus_per_task=16,
        container_image="/containers/alpagym.sqsh",
        container_cache_root=None,
        container_workdir="/workspace/alpagym",
        uv_cache_dir="/tmp/uv",
        container_mounts=["/host/data:/container/data", "/tmp/uv:/tmp/uv"],
        export_env=["UV_CACHE_DIR=/tmp/uv"],
        topology=SeparateNodesSlurmTopologyConfig(cosmos_nodes=2, alpasim_nodes=1),
    )
