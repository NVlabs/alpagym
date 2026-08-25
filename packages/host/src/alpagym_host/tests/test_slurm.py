# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ArtifactPaths,
    ExecutionBackend,
    ExecutionConfig,
    SeparateNodesSlurmTopologyConfig,
    SlurmConfig,
)
from alpagym_host.run_topology import CosmosWorkerPlan, RunHostPlan
from alpagym_host.slurm import (
    _resolve_container_digest,
    _split_registry_repository_tag,
    build_cosmos_srun_command,
    prepare_container_image,
    render_submit_script,
    submit_slurm_job,
    validate_docker_container,
    validate_slurm_config,
)


def test_prepare_container_image_imports_missing_docker_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing registry images are imported into the configured sqsh cache."""
    cache_root = tmp_path / "sqsh"
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    container_image = "registry.example.com/team/alpagym:abc123"
    image_digest = f"sha256:{'a' * 64}"
    expected_image = (
        cache_root / f"registry.example.com_team_alpagym_sha256_{'a' * 64}.sqsh"
    )
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    slurm = _slurm_config(
        partition="batch",
        account="research",
        container_image=container_image,
        container_cache_root=str(cache_root),
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path / "enroot-temp"))
    monkeypatch.setattr(
        "alpagym_host.slurm._resolve_container_digest", lambda image: image_digest
    )

    import_calls: list[dict[str, Any]] = []

    def fake_subprocess_run(*args, **kwargs):
        import_calls.append({"args": args, "kwargs": kwargs})
        assert Path(kwargs["env"]["ENROOT_TEMP_PATH"]).is_dir()
        command = args[0]
        Path(command[command.index("--output") + 1]).touch()
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr("alpagym_host.slurm.subprocess.run", fake_subprocess_run)

    assert prepare_container_image(
        container_image=container_image,
        container_cache_root=slurm.container_cache_root,
    ) == str(expected_image)
    assert expected_image.is_file()
    assert len(import_calls) == 1
    import_command = import_calls[0]["args"][0]
    assert import_command[:3] == ["enroot", "import", "--output"]
    # enroot imports to a private temp under the cache root, then the result is published
    # with an atomic rename, so a concurrent run never reads a half-written .sqsh.
    output_path = Path(import_command[3])
    assert output_path != expected_image
    assert output_path.name == expected_image.name
    assert output_path.parent.parent == cache_root
    # Import uses the tag URI; the resolved digest only names the cache file (enroot has
    # no digest-pinned import URI).
    assert import_command[4] == "docker://registry.example.com#team/alpagym:abc123"
    assert import_calls[0]["kwargs"]["env"]["ENROOT_CONFIG_PATH"] == str(
        enroot_config_path
    )


def test_prepare_container_image_refreshes_moving_tag_when_digest_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving tags ignore stale tag-keyed sqsh files and import the current digest."""
    cache_root = tmp_path / "sqsh"
    cache_root.mkdir()
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    container_image = "registry.example.com/team/alpagym:latest"
    image_digest = f"sha256:{'b' * 64}"
    stale_tag_cache = cache_root / "registry.example.com_team_alpagym_latest.sqsh"
    expected_image = (
        cache_root / f"registry.example.com_team_alpagym_sha256_{'b' * 64}.sqsh"
    )
    stale_tag_cache.write_text("old latest", encoding="utf-8")
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path / "enroot-temp"))
    monkeypatch.setattr(
        "alpagym_host.slurm._resolve_container_digest", lambda image: image_digest
    )

    import_commands: list[list[str]] = []

    def fake_subprocess_run(*args, **kwargs):
        del kwargs
        command = args[0]
        import_commands.append(command)
        Path(command[command.index("--output") + 1]).write_text(
            "new latest", encoding="utf-8"
        )
        return CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr("alpagym_host.slurm.subprocess.run", fake_subprocess_run)

    assert prepare_container_image(
        container_image=container_image,
        container_cache_root=str(cache_root),
    ) == str(expected_image)
    assert stale_tag_cache.read_text(encoding="utf-8") == "old latest"
    assert expected_image.read_text(encoding="utf-8") == "new latest"
    assert len(import_commands) == 1
    assert import_commands[0][:3] == ["enroot", "import", "--output"]
    assert import_commands[0][4] == "docker://registry.example.com#team/alpagym:latest"


def test_prepare_container_image_reuses_digest_cache_without_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching digest-keyed sqsh skips the expensive enroot import."""
    cache_root = tmp_path / "sqsh"
    cache_root.mkdir()
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    container_image = "registry.example.com/team/alpagym:latest"
    image_digest = f"sha256:{'c' * 64}"
    expected_image = (
        cache_root / f"registry.example.com_team_alpagym_sha256_{'c' * 64}.sqsh"
    )
    expected_image.write_text("cached latest", encoding="utf-8")
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    monkeypatch.setattr(
        "alpagym_host.slurm._resolve_container_digest", lambda image: image_digest
    )

    def fail_on_subprocess(*args, **kwargs):
        del kwargs
        raise AssertionError(f"unexpected subprocess call on cache hit: {args[0]}")

    monkeypatch.setattr("alpagym_host.slurm.subprocess.run", fail_on_subprocess)

    assert prepare_container_image(
        container_image=container_image,
        container_cache_root=str(cache_root),
    ) == str(expected_image)


@pytest.mark.parametrize(
    ("container_image", "expected"),
    [
        (
            "registry.example.com/org/alpagym:latest",
            ("registry.example.com", "org/alpagym", "latest"),
        ),
        (
            "docker://registry.example.com#team/alpagym:abc123",
            ("registry.example.com", "team/alpagym", "abc123"),
        ),
        (
            "docker://registry.example.com/team/alpagym:abc123",
            ("registry.example.com", "team/alpagym", "abc123"),
        ),
        (
            "docker://user@registry.example.com#team/alpagym:abc123",
            ("registry.example.com", "team/alpagym", "abc123"),
        ),
    ],
)
def test_split_registry_repository_tag(
    container_image: str, expected: tuple[str, str, str]
) -> None:
    """Parsing handles plain refs and both Enroot URI forms for the manifest lookup."""
    assert _split_registry_repository_tag(container_image) == expected


def test_split_registry_repository_tag_requires_tag() -> None:
    """A reference without a tag is rejected so the manifest lookup never guesses one."""
    with pytest.raises(ValueError, match="registry/repo:tag"):
        _split_registry_repository_tag("registry.example.com/team/alpagym")


def test_split_registry_repository_tag_rejects_digest_pinned_ref() -> None:
    """Digest-pinned refs are rejected instead of mixing pin and moving-tag semantics."""
    with pytest.raises(ValueError, match="tag-based"):
        _split_registry_repository_tag(
            f"registry.example.com/team/alpagym@sha256:{'a' * 64}"
        )


def test_validate_docker_container_checks_scheme_registry_and_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation uses only the scheme URI host for credentials and rejects malformed refs."""
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))

    validate_docker_container(
        container_image="docker://registry.example.com/team/alpagym:latest",
        container_cache_root=str(tmp_path / "sqsh"),
    )
    with pytest.raises(ValueError, match="registry/repo:tag"):
        validate_docker_container(
            container_image="docker://registry.example.com/team/alpagym",
            container_cache_root=str(tmp_path / "sqsh"),
        )


def test_resolve_container_digest_uses_timeouts_and_access_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry API calls are bounded and accept the OCI access_token response field."""
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    image_digest = f"sha256:{'e' * 64}"
    calls: list[tuple[str, int | None]] = []

    class Response:
        def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.body = body
            self.headers = headers or {}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request, *, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        calls.append((url, timeout))
        if url == "https://registry.example.com/v2/":
            raise urllib.error.HTTPError(
                url,
                401,
                "Unauthorized",
                {
                    "WWW-Authenticate": (
                        'Bearer realm="https://registry.example.com/jwt/auth",'
                        'service="container_registry"'
                    )
                },
                None,
            )
        if url.startswith("https://registry.example.com/jwt/auth?"):
            assert request.get_header("Authorization", "").startswith("Basic ")
            return Response(b'{"access_token": "bearer-token"}')
        if url == "https://registry.example.com/v2/team/alpagym/manifests/latest":
            assert request.get_header("Authorization") == "Bearer bearer-token"
            return Response(b"{}", {"Docker-Content-Digest": image_digest})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("alpagym_host.slurm.urllib.request.urlopen", fake_urlopen)

    assert (
        _resolve_container_digest("registry.example.com/team/alpagym:latest")
        == image_digest
    )
    assert calls == [
        ("https://registry.example.com/v2/", 30),
        (
            "https://registry.example.com/jwt/auth?service=container_registry"
            "&scope=repository:team/alpagym:pull",
            30,
        ),
        ("https://registry.example.com/v2/team/alpagym/manifests/latest", 30),
    ]


def test_concurrent_container_image_import_never_serves_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent imports publish a complete .sqsh atomically, never a partial file.

    Each thread's stubbed enroot writes a distinct *complete* sentinel into its own temp
    file; a barrier holds every thread at the `os.replace` publish so they race the rename.
    The final cached image must equal exactly one writer's full sentinel — never a prefix,
    empty, or interleaved write — proving the temp-then-atomic-rename keeps a concurrent
    reader from ever seeing a half-written image.
    """
    workers = 6
    cache_root = tmp_path / "sqsh"
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    container_image = "registry.example.com/team/alpagym:abc123"
    image_digest = f"sha256:{'d' * 64}"
    expected_image = (
        cache_root / f"registry.example.com_team_alpagym_sha256_{'d' * 64}.sqsh"
    )
    (enroot_config_path / ".credentials").write_text(
        "machine registry.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path / "enroot-temp"))
    monkeypatch.setattr(
        "alpagym_host.slurm._resolve_container_digest", lambda image: image_digest
    )

    barrier = threading.Barrier(workers)
    real_replace = os.replace
    lock = threading.Lock()
    counter = {"n": 0}
    output_targets: list[Path] = []

    def fake_run(*args, **kwargs):
        del kwargs
        command = args[0]
        with lock:
            index = counter["n"]
            counter["n"] += 1
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            f"SQSHFS-COMPLETE-{index}", encoding="utf-8"
        )  # a complete image
        with lock:
            output_targets.append(output)
        return CompletedProcess(args=command, returncode=0)

    def barrier_replace(src, dst):
        # Hold every import at the contended rename, then let them all race at once.
        barrier.wait(timeout=30)
        real_replace(src, dst)

    monkeypatch.setattr("alpagym_host.slurm.subprocess.run", fake_run)
    monkeypatch.setattr("alpagym_host.slurm.os.replace", barrier_replace)

    def prepare() -> str:
        return prepare_container_image(
            container_image=container_image,
            container_cache_root=str(cache_root),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [
            future.result() for future in [pool.submit(prepare) for _ in range(workers)]
        ]

    assert all(result == str(expected_image) for result in results)
    assert expected_image.read_text(encoding="utf-8") in {
        f"SQSHFS-COMPLETE-{i}" for i in range(workers)
    }  # one writer's full image, never a truncated or interleaved one
    # every import wrote to a private temp file, never the published cache path directly
    assert all(
        t != expected_image and t.parent.parent == cache_root for t in output_targets
    )
    assert [p for p in cache_root.iterdir() if p.name.startswith(".import-")] == []


def test_validate_slurm_config_requires_enroot_credentials_for_legacy_tag_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag-keyed sqsh cannot prove that a moving Docker tag is current."""
    monkeypatch.delenv("ENROOT_CONFIG_PATH", raising=False)
    container_image = "registry.example.com/team/alpagym:abc123"
    cache_root = tmp_path / "sqsh"
    cache_root.mkdir()
    (cache_root / "registry.example.com_team_alpagym_abc123.sqsh").touch()
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image=container_image,
            container_cache_root=str(cache_root),
        ),
    )

    with pytest.raises(ValueError, match="ENROOT_CONFIG_PATH"):
        validate_slurm_config(execution)


def test_validate_slurm_config_requires_enroot_credentials_for_missing_docker_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold sqsh cache requires registry credentials before the job starts."""
    enroot_config_path = tmp_path / "enroot"
    enroot_config_path.mkdir()
    (enroot_config_path / ".credentials").write_text(
        "machine other.example.com login user password token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENROOT_CONFIG_PATH", str(enroot_config_path))
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="registry.example.com/team/alpagym:abc123",
            container_cache_root=str(tmp_path / "sqsh"),
        ),
    )

    with pytest.raises(ValueError, match="registry.example.com"):
        validate_slurm_config(execution)


@pytest.mark.parametrize(
    ("container_mounts", "export_env", "expected_message"),
    [
        ([], None, "execution.slurm.container_mounts"),
        (None, [], "execution.slurm.export_env"),
    ],
)
def test_validate_slurm_config_requires_uv_cache_container_handoff(
    container_mounts: list[str] | None,
    export_env: list[str] | None,
    expected_message: str,
) -> None:
    """The uv cache directory must be mounted and exported into the container."""
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="/containers/alpagym.sqsh",
            container_mounts=container_mounts,
            export_env=export_env,
        ),
    )

    with pytest.raises(ValueError, match=expected_message):
        validate_slurm_config(execution)


def test_render_submit_script_accepts_multi_node_slurm_config(tmp_path: Path) -> None:
    """Slurm submit scripts preserve the multi-node allocation."""
    artifact_paths = _artifact_paths(tmp_path / "run")
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="/containers/alpagym.sqsh",
            nodes=3,
            topology=SeparateNodesSlurmTopologyConfig(cosmos_nodes=2, alpasim_nodes=1),
        ),
    )

    script = render_submit_script(
        execution_config=execution,
        artifact_paths=artifact_paths,
        project_root=Path("/repo/projects/alpagym"),
        deploy="cluster",
        topology="slurm_distributed_1_2_2",
    )

    assert "#SBATCH --nodes=3" in script
    assert "command=run" in script
    assert (
        "execution.resolved_config_path=" + str(artifact_paths.resolved_config_path)
        in script
    )
    # The compute-node re-entry must reselect both mandatory Hydra groups.
    assert "deploy=cluster" in script
    assert "topology=slurm_distributed_1_2_2" in script


def test_cosmos_srun_command_disables_cpu_binding_for_nonexclusive_steps(
    tmp_path: Path,
) -> None:
    """Non-exclusive partial-node Cosmos steps must not inherit packed CPU binding."""
    command = build_cosmos_srun_command(
        cosmos_hosts=(
            RunHostPlan(
                hostname="cosmos-0",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=2,
                alpasim_gpus=0,
                cosmos_workers=(CosmosWorkerPlan((0, 1), 0),),
            ),
        ),
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="/containers/alpagym.sqsh",
            exclusive=False,
        ),
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=["true"],
        worker_commands=((["python", "-m", "cosmos_rl.launcher.launch_all"],),),
        log_dir=tmp_path / "logs",
    )

    assert "--overlap" in command
    assert "--cpu-bind=none" in command
    assert "--cpus-per-task=16" not in command


def test_cosmos_srun_command_keeps_default_cpu_binding_for_exclusive_steps(
    tmp_path: Path,
) -> None:
    command = build_cosmos_srun_command(
        cosmos_hosts=(
            RunHostPlan(
                hostname="cosmos-0",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=8,
                alpasim_gpus=0,
                cosmos_workers=(CosmosWorkerPlan(tuple(range(8)), 0),),
            ),
        ),
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="/containers/alpagym.sqsh",
            exclusive=True,
        ),
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=["true"],
        worker_commands=((["python", "-m", "cosmos_rl.launcher.launch_all"],),),
        log_dir=tmp_path / "logs",
    )

    assert "--cpu-bind=none" not in command


def test_slurm_mem_is_requested_for_batch_and_container_step(tmp_path: Path) -> None:
    """Full-node jobs can request all node memory for the allocation and srun step."""
    slurm = _slurm_config(
        partition="batch",
        account="research",
        container_image="/containers/alpagym.sqsh",
        mem="0",
    )

    command = build_cosmos_srun_command(
        cosmos_hosts=(
            RunHostPlan(
                hostname="cosmos-0",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=8,
                alpasim_gpus=0,
                cosmos_workers=(CosmosWorkerPlan(tuple(range(8)), 0),),
            ),
        ),
        slurm=slurm,
        container_image="/containers/alpagym.sqsh",
        runtime_check_command=["true"],
        worker_commands=((["python", "-m", "cosmos_rl.launcher.launch_all"],),),
        log_dir=tmp_path / "logs",
    )
    script = render_submit_script(
        execution_config=ExecutionConfig(
            backend=ExecutionBackend.slurm,
            resolved_config_path=None,
            slurm=slurm,
        ),
        artifact_paths=_artifact_paths(tmp_path / "run"),
        project_root=Path("/repo/projects/alpagym"),
        deploy="cluster",
        topology="slurm_full_node_1_3_4",
    )

    assert "--mem=0" in command
    assert "#SBATCH --mem=0" in script


def test_submit_slurm_job_writes_script_and_calls_sbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submission writes submit.sbatch and invokes sbatch with that path."""
    artifact_paths = _artifact_paths(tmp_path / "run")
    artifact_paths.run_dir.mkdir()
    completed = CompletedProcess(
        args=["sbatch", str(artifact_paths.submit_script_path)],
        returncode=0,
        stdout="Submitted batch job 12345\n",
        stderr="",
    )
    subprocess_calls: list[dict[str, object]] = []

    def fake_subprocess_run(*args, **kwargs):
        subprocess_calls.append({"args": args, "kwargs": kwargs})
        return completed

    monkeypatch.setattr("alpagym_host.slurm.subprocess.run", fake_subprocess_run)
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch", account="research", container_image="/c.sqsh"
        ),
    )

    job_id = submit_slurm_job(
        execution=execution,
        artifact_paths=artifact_paths,
        project_root=Path("/repo/projects/alpagym"),
        deploy="cluster",
        topology="slurm_full_node_1_3_4",
    )

    script = artifact_paths.submit_script_path.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "#SBATCH --job-name=alpagym" in script
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --account=research" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "set -euo pipefail" in script
    assert "export UV_CACHE_DIR=/tmp/alpagym-uv-cache" in script
    assert "cd /repo/projects/alpagym" in script
    assert (
        "execution.resolved_config_path=" + str(artifact_paths.resolved_config_path)
        in script
    )
    assert "command=run" in script
    # Carry the deploy and topology choices so the compute-node re-entry
    # satisfies both mandatory Hydra groups when it composes default.yaml.
    assert "deploy=cluster" in script
    assert "topology=slurm_full_node_1_3_4" in script
    assert "alpasim_wizard" not in script
    assert "cosmos_rl.launcher.launch_all" not in script
    assert subprocess_calls == [
        {
            "args": (["sbatch", str(artifact_paths.submit_script_path)],),
            "kwargs": {
                "check": True,
                "text": True,
                "capture_output": True,
            },
        }
    ]
    assert job_id == "12345"


def test_render_submit_script_omits_cpu_and_exclusive_for_partial_node_jobs(
    tmp_path: Path,
) -> None:
    """Partial-node jobs can opt out of exclusive and explicit CPU requests."""
    artifact_paths = _artifact_paths(tmp_path / "run")
    execution = ExecutionConfig(
        backend=ExecutionBackend.slurm,
        resolved_config_path=None,
        slurm=_slurm_config(
            partition="batch",
            account="research",
            container_image="/c.sqsh",
            gpus_per_node=3,
            exclusive=False,
            cpus_per_task=None,
            topology=AllInOneSlurmTopologyConfig(alpasim_gpus=1),
        ),
    )

    script = render_submit_script(
        execution_config=execution,
        artifact_paths=artifact_paths,
        project_root=Path("/repo/projects/alpagym"),
        deploy="cluster",
        topology="slurm_full_node_1_3_4",
    )

    assert "#SBATCH --gpus-per-node=3" in script
    assert "#SBATCH --exclusive" not in script
    assert "#SBATCH --cpus-per-task" not in script


def _slurm_config(
    partition: str | None,
    account: str | None,
    container_image: str | None,
    gpus_per_node: int = 8,
    container_cache_root: str | None = None,
    container_workdir: str = "/tmp/alpagym",
    uv_cache_dir: str = "/tmp/alpagym-uv-cache",
    exclusive: bool = True,
    cpus_per_task: int | None = 16,
    container_mounts: list[str] | None = None,
    export_env: list[str] | None = None,
    nodes: int = 1,
    topology: AllInOneSlurmTopologyConfig
    | SeparateNodesSlurmTopologyConfig
    | None = None,
    mem: str | None = None,
) -> SlurmConfig:
    """Build a Slurm config for tests."""
    return SlurmConfig(
        job_name="alpagym",
        partition=partition,
        account=account,
        time="02:00:00",
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        topology=topology or AllInOneSlurmTopologyConfig(alpasim_gpus=4),
        exclusive=exclusive,
        cpus_per_task=cpus_per_task,
        container_image=container_image,
        container_cache_root=container_cache_root,
        container_workdir=container_workdir,
        uv_cache_dir=uv_cache_dir,
        container_mounts=container_mounts
        if container_mounts is not None
        else [f"{uv_cache_dir}:{uv_cache_dir}"],
        export_env=export_env
        if export_env is not None
        else [f"UV_CACHE_DIR={uv_cache_dir}"],
        mem=mem,
    )


def _artifact_paths(tmp_path: Path) -> ArtifactPaths:
    """Build artifact paths for tests."""
    return ArtifactPaths(
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        policy_model_bundle_dir=tmp_path / "artifacts" / "policy_model_bundle",
        resolved_config_path=tmp_path / "resolved_config.yaml",
        cosmos_config_path=tmp_path / "cosmos_config.toml",
        submit_script_path=tmp_path / "submit.sbatch",
        log_dir=tmp_path / "logs",
        topology_registry_dir=tmp_path / "topology",
        alpasim_log_dir=tmp_path / "alpasim",
        alpasim_scene_ids_path=tmp_path / "alpasim_scene_ids.yaml",
        perf_dir=tmp_path / "perf",
    )
