# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pytest
from alpagym_host.config import register_config_schema
from alpagym_host.config_validation import validate_run_config
from alpagym_host.run_artifacts import build_artifact_paths, build_run_config
from hydra import compose, initialize_config_module
from hydra.errors import ConfigCompositionException


def test_topology_is_required(tmp_path: Path) -> None:
    """Host config composition requires an explicit topology preset."""
    with pytest.raises(ConfigCompositionException, match="topology"):
        _compose_test_config(tmp_path, topology=None)


def test_topology_selects_full_node_cosmos_and_alpasim_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlpaGym topology presets override the default host run settings."""
    monkeypatch.setenv("USER", "test-user")
    cfg = _compose_test_config(
        tmp_path,
        deploy="slurm",
        topology="slurm_full_node_1_3_4",
    )

    assert cfg.execution.backend == "slurm"
    assert cfg.alpasim.wizard_args.topology == "alpagym_4gpu"
    assert cfg.cosmos.mode == "disaggregated"


def test_validate_run_config_rejects_non_identity_slurm_artifact_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slurm artifact paths must be visible under the same absolute path."""
    resolved_config = _compose_slurm_test_config(tmp_path, monkeypatch)
    uv_cache_dir = resolved_config.execution.slurm.uv_cache_dir
    resolved_config.execution.slurm.container_mounts = [
        f"{uv_cache_dir}:{uv_cache_dir}",
        f"{tmp_path}:/workspace",
    ]
    with pytest.raises(ValueError, match="non-identity container mounts"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_ignores_none_slurm_mount_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``none:`` enroot tmpfs sentinel is excluded from host-path mount sources.

    Drop the identity run-dir mount so the run root is no longer covered. The mount
    check then rejects it, and the ``none:/dev/shm`` sentinel must be absent from the
    reported source list -- proving ``none`` is excluded rather than trivially covered
    by the identity mount.
    """
    resolved_config = _compose_slurm_test_config(tmp_path, monkeypatch)
    uv_cache_dir = resolved_config.execution.slurm.uv_cache_dir
    resolved_config.execution.slurm.container_mounts = [
        "none:/dev/shm:tmpfs,size=1g",
        f"{uv_cache_dir}:{uv_cache_dir}",
    ]
    with pytest.raises(
        ValueError, match="not under any execution.slurm.container_mounts source"
    ) as exc_info:
        validate_run_config(resolved_config, "run")
    message = str(exc_info.value)
    assert str(uv_cache_dir) in message
    assert "none" not in message


def test_validate_run_config_rejects_slurm_model_path_under_non_identity_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real model paths must be visible under the same absolute path in Slurm."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")
    resolved_config = _compose_slurm_test_config(
        tmp_path,
        monkeypatch,
        overrides=[f"policy.model.path={model_dir}"],
    )
    uv_cache_dir = resolved_config.execution.slurm.uv_cache_dir
    resolved_config.execution.slurm.container_mounts = [
        f"{uv_cache_dir}:{uv_cache_dir}",
        f"{tmp_path}:/workspace",
    ]
    with pytest.raises(ValueError, match="policy.model.path=.*non-identity"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_rejects_nccl_rollout_tensor_parallelism(
    tmp_path: Path,
) -> None:
    """NCCL transport currently supports one process per rollout replica."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            "cosmos.rollout.parallelism.tp_size=2",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="one process per rollout replica"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_rejects_nccl_policy_tensor_parallelism(
    tmp_path: Path,
) -> None:
    """NCCL policy rank sizing only supports cosmos.launch.policy_replicas * dp_shard_size."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            "cosmos.policy.parallelism.tp_size=2",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="unsupported policy parallelism axes"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_rejects_nccl_nonpositive_dp_shard_size(
    tmp_path: Path,
) -> None:
    """NCCL policy dp_shard_size must fail at host preflight."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            "cosmos.policy.parallelism.dp_shard_size=0",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="dp_shard_size"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_allows_nccl_multiple_policy_processes(
    tmp_path: Path,
) -> None:
    """NCCL supports concurrent policy receivers with transfer-id rendezvous."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            "cosmos.launch.policy_replicas=2",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    validate_run_config(resolved_config, "run")


def test_validate_run_config_rejects_nccl_colocated_mode(tmp_path: Path) -> None:
    """NCCL transfers tensors between separate processes, so it requires disaggregation."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=["transport=nccl"],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="requires cosmos.mode=disaggregated"):
        validate_run_config(resolved_config, "run")


def test_validate_run_config_rejects_nccl_without_timeout(tmp_path: Path) -> None:
    """NCCL transport requires NCCL_TIMEOUT so workers fail at preflight, not mid-run."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            "~transport.nccl_env.NCCL_TIMEOUT",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="NCCL_TIMEOUT"):
        validate_run_config(resolved_config, "run")


@pytest.mark.parametrize("bad_timeout", ["nan", "inf", "fast"])
def test_validate_run_config_rejects_nccl_nonfinite_timeout(
    tmp_path: Path, bad_timeout: str
) -> None:
    """A non-finite (nan/inf) or non-numeric ("fast") NCCL_TIMEOUT fails preflight by name."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "cosmos.mode=disaggregated",
            f"transport.nccl_env.NCCL_TIMEOUT={bad_timeout}",
        ],
    )
    resolved_config = build_run_config(cfg, build_artifact_paths(cfg))
    with pytest.raises(ValueError, match="NCCL_TIMEOUT"):
        validate_run_config(resolved_config, "run")


def test_config_rejects_invalid_nccl_read_device(tmp_path: Path) -> None:
    """Invalid NCCL read devices fail while Hydra materializes the config object."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "transport=nccl",
            "transport.nccl_read_device=cup",
        ],
    )
    with pytest.raises(ValueError, match="nccl_read_device"):
        build_run_config(cfg, build_artifact_paths(cfg))


def test_validate_run_config_rejects_submit_with_non_slurm_backend(tmp_path: Path) -> None:
    """Submitting requires a Slurm execution backend."""
    cfg = _compose_test_config(tmp_path, overrides=["command=submit"])
    config = build_run_config(cfg, build_artifact_paths(cfg))

    with pytest.raises(
        ValueError,
        match="command=submit requires execution.backend to be a Slurm backend",
    ):
        validate_run_config(config, requested_command="submit")


def test_validate_run_config_rejects_colocated_slurm_mode(tmp_path: Path) -> None:
    """Slurm runs must explicitly author the Cosmos disaggregated mode."""
    cfg = _compose_test_config(
        tmp_path,
        overrides=[
            "execution.backend=slurm",
            "execution.slurm.partition=batch",
            "execution.slurm.account=research",
            "execution.slurm.container_image=/containers/alpagym.sqsh",
        ],
    )
    config = build_run_config(cfg, build_artifact_paths(cfg))

    with pytest.raises(ValueError, match="cosmos.mode must be 'disaggregated' for slurm"):
        validate_run_config(config, requested_command="run")


def test_execute_run_delegates_to_unified_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CLI execution delegates backend orchestration to the unified lifecycle."""
    from alpagym_host import cli

    cfg = _compose_test_config(tmp_path, overrides=["command=run"])
    artifact_paths = build_artifact_paths(cfg)
    resolved_config = build_run_config(cfg, artifact_paths)
    calls: list[object] = []

    def record_run(config: object) -> None:
        calls.append(config)

    monkeypatch.setattr(cli, "load_or_create_run_config", lambda config: resolved_config)
    monkeypatch.setattr(cli, "validate_huggingface_access", lambda: None)
    monkeypatch.setattr(cli, "execute_run", record_run)
    cli.main.__wrapped__(cfg)

    assert calls == [resolved_config]


def test_main_validates_huggingface_access_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main validates HuggingFace access before dispatching run or submit.

    The check sits ahead of the command branch, so submit fails before queuing a
    Slurm allocation, not only once a local run reaches the AlpaSim checkout.
    """
    from types import SimpleNamespace

    from alpagym_host import cli

    call_order: list[str] = []
    run_config = object()

    monkeypatch.setattr(cli, "load_or_create_run_config", lambda cfg: run_config)
    monkeypatch.setattr(
        cli, "validate_huggingface_access", lambda: call_order.append("huggingface")
    )
    monkeypatch.setattr(cli, "execute_run", lambda cfg: call_order.append("execute_run"))

    cli.main.__wrapped__(SimpleNamespace(command="run", logging_level="INFO"))

    assert call_order == ["huggingface", "execute_run"]


def test_main_skips_huggingface_validation_for_humanoid_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Humanoid dynamics-only runs do not require AV/NuRec HuggingFace assets."""
    from types import SimpleNamespace

    from alpagym_host import cli

    call_order: list[str] = []
    run_config = SimpleNamespace(alpasim=SimpleNamespace(simulation_domain="humanoid"))

    monkeypatch.setattr(cli, "load_or_create_run_config", lambda cfg: run_config)
    monkeypatch.setattr(
        cli, "validate_huggingface_access", lambda: call_order.append("huggingface")
    )
    monkeypatch.setattr(cli, "execute_run", lambda cfg: call_order.append("execute_run"))

    cli.main.__wrapped__(SimpleNamespace(command="run", logging_level="INFO"))

    assert call_order == ["execute_run"]


def test_main_configures_logging_from_hydra_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host process uses the AlpaGym-authored logging level."""
    from alpagym_host import cli

    cfg = _compose_test_config(tmp_path, overrides=["logging_level=DEBUG"])
    artifact_paths = build_artifact_paths(cfg)
    resolved_config = build_run_config(cfg, artifact_paths)
    basic_config_calls: list[dict[str, object]] = []

    def record_basic_config(**kwargs: object) -> None:
        basic_config_calls.append(kwargs)

    monkeypatch.setattr(cli.logging, "basicConfig", record_basic_config)
    monkeypatch.setattr(cli, "load_or_create_run_config", lambda config: resolved_config)
    monkeypatch.setattr(cli, "validate_huggingface_access", lambda: None)
    monkeypatch.setattr(cli, "execute_run", lambda config: None)

    cli.main.__wrapped__(cfg)

    assert basic_config_calls[0]["level"] == logging.DEBUG


def _compose_test_config(
    tmp_path: Path,
    overrides: list[str] | None = None,
    deploy: str = "local",
    topology: str | None = "local_colocated_1gpu",
):
    """Compose the host Hydra config for CLI unit tests."""
    register_config_schema()
    model_path = _write_model_bundle_dir(tmp_path)
    topology_overrides = [] if topology is None else [f"topology={topology}"]
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        return compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                f"deploy={deploy}",
                "policy.model.kind=alpamayo_r1",
                f"policy.model.path={model_path.as_posix()}",
                *topology_overrides,
                *(overrides or []),
            ],
        )


def _compose_slurm_test_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: list[str] | None = None,
):
    """Resolve a Slurm host run config with its caches redirected under tmp_path.

    Uses deploy=slurm with fake site settings and redirects cache_root_dir into
    tmp_path so the config validates on a CI runner.
    """
    monkeypatch.setenv("USER", "test-user")
    cfg = _compose_test_config(
        tmp_path,
        deploy="slurm",
        topology="slurm_distributed_1_1_1",
        overrides=[
            f"cache_root_dir={(tmp_path / 'cache').as_posix()}",
            "execution.slurm.partition=batch",
            "execution.slurm.account=research",
            "execution.slurm.container_image=/local.sqsh",
            *(overrides or []),
        ],
    )
    return build_run_config(cfg, build_artifact_paths(cfg))


def _write_model_bundle_dir(tmp_path: Path) -> Path:
    """Write the minimal HF bundle shape needed by host config tests."""
    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    return bundle_dir
