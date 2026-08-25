# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

from alpagym_host.config import (
    ALPASIM_RUNTIME_ID_ENV,
    ArtifactPaths,
    ExecutionBackend,
    ExecutionConfig,
    SlurmConfig,
)
from alpagym_host.run_topology import RunHostPlan


def allocated_hostnames() -> list[str]:
    """Return hostnames in the current Slurm allocation."""
    result = subprocess.run(
        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_wizard_srun_command(
    host: RunHostPlan,
    slurm: SlurmConfig,
    wizard_command: list[str],
    wizard_workdir: Path,
    log_path: Path,
) -> list[str]:
    """Build an srun command exposing the host GPUs to one AlpaSim Wizard."""
    if not host.runs_alpasim or host.alpasim_gpus < 1:
        raise ValueError("Wizard host must include AlpaSim GPUs")

    script = "\n".join(
        [
            *_container_shell_exports(slurm.export_env),
            "export SLURM_JOB_NODELIST=$(hostname)",
            "export SLURM_NODELIST=$(hostname)",
            # srun defaults to --export=ALL and this runs under `bash -lc`, a login shell
            # that re-sources /etc/environment (which sets UV_PROJECT_ENVIRONMENT/VIRTUAL_ENV
            # on the cluster's compute nodes). Unset them so the Wizard -- and the service
            # srun's it spawns -- select the venv from the interpreter in wizard_command,
            # not a leaked path that is wrong inside those service containers.
            "unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV",
            f"exec {shlex.join(wizard_command)}",
        ]
    )
    srun_command = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={host.hostname}",
    ]
    if slurm.container_image is not None:
        srun_command.append(f"--container-image={slurm.container_image}")
    if slurm.container_mounts:
        srun_command.append(
            f"--container-mounts={','.join(slurm.container_mounts)}"
        )
    srun_command.append(f"--container-workdir={wizard_workdir}")
    if not slurm.exclusive:
        srun_command.append("--cpu-bind=none")
    srun_command.extend(
        [
            f"--output={log_path}",
            f"--error={log_path}",
            f"--export={','.join(['ALL', *slurm.export_env])}",
            "bash",
            "-lc",
            script,
        ]
    )
    return srun_command


def build_cosmos_srun_command(
    cosmos_hosts: tuple[RunHostPlan, ...],
    slurm: SlurmConfig,
    container_image: str,
    runtime_check_command: list[str],
    worker_commands: tuple[tuple[list[str], ...], ...],
    log_dir: Path,
) -> list[str]:
    """Build an srun command for Cosmos tasks across Slurm hosts."""
    if len(worker_commands) != len(cosmos_hosts):
        raise ValueError("Cosmos host command count must match Cosmos host count")
    for host, host_commands in zip(cosmos_hosts, worker_commands, strict=True):
        if len(host_commands) != len(host.cosmos_workers):
            raise ValueError(
                "Cosmos worker command count must match the host worker plan"
            )

    hostnames = tuple(host.hostname for host in cosmos_hosts)
    cosmos_gpus_per_task = cosmos_hosts[0].cosmos_gpu_count
    srun_command = [
        "srun",
        "--overlap",
        f"--nodes={len(cosmos_hosts)}",
        f"--ntasks={len(cosmos_hosts)}",
        "--ntasks-per-node=1",
        "--distribution=block:block",
        f"--nodelist={','.join(hostnames)}",
        f"--gpus-per-task={cosmos_gpus_per_task}",
        f"--gpu-bind=mask_gpu:{_gpu_mask(tuple(range(cosmos_gpus_per_task)))}",
        f"--container-image={container_image}",
    ]
    if not slurm.exclusive:
        srun_command.append("--cpu-bind=none")
    if slurm.mem is not None:
        srun_command.append(f"--mem={slurm.mem}")
    if slurm.container_mounts:
        srun_command.append(f"--container-mounts={','.join(slurm.container_mounts)}")
    srun_command.extend(
        [
            f"--container-workdir={slurm.container_workdir}",
            f"--output={log_dir / 'cosmos_%t.log'}",
            f"--error={log_dir / 'cosmos_%t.log'}",
            f"--export={','.join(['ALL', *slurm.export_env])}",
            "bash",
            "-lc",
            _cosmos_launcher_script(
                runtime_check_command=runtime_check_command,
                cosmos_hosts=cosmos_hosts,
                worker_commands=worker_commands,
                export_env=slurm.export_env,
            ),
        ]
    )
    return srun_command


def _cosmos_launcher_script(
    runtime_check_command: list[str],
    cosmos_hosts: tuple[RunHostPlan, ...],
    worker_commands: tuple[tuple[list[str], ...], ...],
    export_env: list[str] | None = None,
) -> str:
    """Render the per-task dispatcher for one multi-task Cosmos Slurm step.

    `srun --ntasks=N` accepts one command template for the whole step, so all
    Cosmos tasks start the same shell script. Slurm assigns each task a distinct
    `SLURM_PROCID`; this wrapper uses that task id to exec the matching
    prebuilt Cosmos launcher command.
    """
    lines = [
        *_container_shell_exports(export_env or []),
        shlex.join(runtime_check_command),
        'case "$SLURM_PROCID" in',
    ]
    for host_index, (host, host_commands) in enumerate(
        zip(cosmos_hosts, worker_commands, strict=True)
    ):
        host_script = _cosmos_host_worker_script(
            host=host,
            worker_commands=host_commands,
        )
        lines.extend(
            [
                f"  {host_index})",
                *[f"    {line}" for line in host_script.splitlines()],
                "    ;;",
            ]
        )
    lines.extend(
        [
            "  *)",
            '    echo "Unexpected SLURM_PROCID: $SLURM_PROCID" >&2',
            "    exit 1",
            "    ;;",
            "esac",
        ]
    )
    return "\n".join(lines)


def _container_shell_exports(export_env: list[str]) -> list[str]:
    """Reapply shell-sensitive paths after Pyxis/login-shell initialization.

    Slurm's ``--export`` propagates ordinary variables, but Pyxis and
    ``bash -l`` may replace HOME and PATH from the container/user database.
    Those two values control where runtime state is written and whether the
    mounted Slurm clients are discoverable, so set them in the launched shell.
    """
    exports: list[str] = []
    for assignment in export_env:
        name, separator, value = assignment.partition("=")
        if separator and name in {"HOME", "PATH"}:
            exports.append(f"export {name}={shlex.quote(value)}")
    return exports


def _cosmos_host_worker_script(
    host: RunHostPlan,
    worker_commands: tuple[list[str], ...],
) -> str:
    """Render one Slurm task's GPU-scoped Cosmos worker commands."""
    if (
        len(host.cosmos_workers) == 1
        and host.cosmos_workers[0].gpu_ids == host.cosmos_gpu_ids
    ):
        return f"exec {shlex.join(worker_commands[0])}"

    command_lines: list[str] = []
    for worker, command in zip(host.cosmos_workers, worker_commands, strict=True):
        environment = [
            f"CUDA_VISIBLE_DEVICES={','.join(map(str, worker.gpu_ids))}",
        ]
        if worker.alpasim_runtime_id is not None:
            environment.append(f"{ALPASIM_RUNTIME_ID_ENV}={worker.alpasim_runtime_id}")
        command_lines.append(shlex.join(["env", *environment, *command]))

    if len(command_lines) == 1:
        return f"exec {command_lines[0]}"

    lines = [
        "pids=()",
        'cleanup() { kill "${pids[@]}" 2>/dev/null || true; }',
        "trap cleanup EXIT TERM INT",
    ]
    for command_line in command_lines:
        lines.extend([f"{command_line} &", 'pids+=("$!")'])
    lines.extend(
        [
            'remaining="${#pids[@]}"',
            "status=0",
            "while (( remaining > 0 )); do",
            "  if wait -n; then",
            "    ((remaining-=1))",
            "  else",
            '    status="$?"',
            "    cleanup",
            '    wait "${pids[@]}" 2>/dev/null || true',
            "    break",
            "  fi",
            "done",
            "trap - EXIT TERM INT",
            'exit "$status"',
        ]
    )
    return "\n".join(lines)


def _gpu_mask(gpu_ids: tuple[int, ...]) -> str:
    """Return a Slurm GPU binding mask for GPU ids."""
    if not gpu_ids:
        raise ValueError("GPU mask requires at least one GPU id")
    mask = 0
    for gpu_id in gpu_ids:
        mask |= 1 << gpu_id
    return hex(mask)


def validate_slurm_config(execution: ExecutionConfig) -> None:
    """Validate Slurm execution settings."""
    slurm = execution.slurm
    if slurm.nodes < 1:
        raise ValueError("execution.slurm.nodes must be at least 1 for slurm")
    if slurm.gpus_per_node < 1:
        raise ValueError("execution.slurm.gpus_per_node must be at least 1 for slurm")

    _validate_slurm_container_settings(slurm, backend_name="slurm")


def _validate_slurm_container_settings(slurm: SlurmConfig, backend_name: str) -> None:
    """Validate Slurm settings shared by all containerized Slurm backends."""
    missing_settings = [
        f"execution.slurm.{name}"
        for name, value in [
            ("partition", slurm.partition),
            ("account", slurm.account),
            ("container_image", slurm.container_image),
            ("uv_cache_dir", slurm.uv_cache_dir),
        ]
        if value is None or value == ""
    ]
    if missing_settings:
        raise ValueError(
            f"{', '.join(missing_settings)} must be set for {backend_name}"
        )
    if slurm.cpus_per_task is not None and slurm.cpus_per_task <= 0:
        raise ValueError("execution.slurm.cpus_per_task must be positive when set")

    uv_cache_mount = f"{slurm.uv_cache_dir}:{slurm.uv_cache_dir}"
    if uv_cache_mount not in slurm.container_mounts:
        raise ValueError(
            f"execution.slurm.container_mounts must include {uv_cache_mount!r} for {backend_name}"
        )
    uv_cache_export = f"UV_CACHE_DIR={slurm.uv_cache_dir}"
    if uv_cache_export not in slurm.export_env:
        raise ValueError(
            f"execution.slurm.export_env must include {uv_cache_export!r} for {backend_name}"
        )
    uv_cache_dir = Path(cast(str, slurm.uv_cache_dir))
    try:
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"execution.slurm.uv_cache_dir must be creatable: {uv_cache_dir}"
        ) from exc
    _validate_writable_directory(uv_cache_dir, "execution.slurm.uv_cache_dir")
    validate_docker_container(
        container_image=cast(str, slurm.container_image),
        container_cache_root=slurm.container_cache_root,
    )


def validate_docker_container(
    container_image: str,
    container_cache_root: str | None,
) -> None:
    """Validate Docker image caching settings before starting Slurm work."""
    if not _requires_container_cache(container_image):
        return
    if not container_cache_root:
        raise ValueError(
            "execution.slurm.container_cache_root must be set when "
            "execution.slurm.container_image is a Docker URI"
        )

    registry, _, _ = _split_registry_repository_tag(container_image)
    cache_root = Path(container_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _validate_writable_directory(cache_root, "execution.slurm.container_cache_root")
    # Resolving the credentials validates that an entry for this registry exists.
    _enroot_registry_credentials(registry)


def prepare_container_image(
    container_image: str,
    container_cache_root: str | None,
) -> str:
    """Return the image path to pass to Pyxis.

    Supported image forms:

    - `.sqsh` references and existing local image paths are returned unchanged.
    - Docker-style registry refs are converted to Enroot `docker://registry#image` URIs.
    - Registry refs are resolved to `sha256` digests before cache lookup, so moving tags
      such as `:latest` pick up new image contents automatically. The digest names the
      cache file; the tag URI drives the import (enroot has no digest-pinned import URI).
    - Missing registry-image cache entries are imported with `enroot import` and published
      with an atomic rename under `container_cache_root`.
    """
    if not _requires_container_cache(container_image):
        logging.info("Using Slurm container image without import: %s", container_image)
        return container_image

    validate_docker_container(
        container_image=container_image,
        container_cache_root=container_cache_root,
    )
    cache_root = Path(cast(str, container_cache_root))

    image_uri = _enroot_docker_uri(container_image)
    image_digest = _resolve_container_digest(container_image)
    cached_image = cache_root / _sqsh_filename(container_image, image_digest)
    if cached_image.is_file():
        logging.info(
            "Using cached Slurm container image: source=%s digest=%s path=%s",
            image_uri,
            image_digest,
            cached_image,
        )
        return str(cached_image)

    enroot_env = _enroot_import_env()
    logging.info(
        "Importing Slurm container image: source=%s digest=%s output=%s",
        image_uri,
        image_digest,
        cached_image,
    )
    # Import to a private temp file, then publish with an atomic rename. enroot
    # writes the squashfs in place with no atomic step, so importing straight to
    # `cached_image` would let a concurrent run read a half-written file (the
    # `is_file()` check above accepts any existing file). The rename stays within
    # `cache_root`, so it is atomic; concurrent imports of the same digest race the
    # rename and the last writer wins, which is safe because the file is complete.
    import_dir = Path(tempfile.mkdtemp(dir=cache_root, prefix=".import-"))
    tmp_image = import_dir / cached_image.name
    try:
        result = subprocess.run(
            ["enroot", "import", "--output", str(tmp_image), image_uri],
            check=False,
            env=enroot_env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"enroot import failed for {image_uri}: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        if not tmp_image.is_file():
            raise FileNotFoundError(f"enroot import did not produce {tmp_image}")
        os.replace(tmp_image, cached_image)
    finally:
        shutil.rmtree(import_dir, ignore_errors=True)
    logging.info("Imported Slurm container image: %s", cached_image)
    return str(cached_image)


def _requires_container_cache(container_image: str) -> bool:
    """Return whether the image needs on-demand sqsh caching."""
    return not (container_image.endswith(".sqsh") or Path(container_image).exists())


def _enroot_docker_uri(container_image: str) -> str:
    """Return an Enroot Docker URI for a registry image."""
    if "://" in container_image:
        return container_image
    registry, image = container_image.split("/", maxsplit=1)
    return f"docker://{registry}#{image}"


def _resolve_container_digest(container_image: str) -> str:
    """Return the sha256 manifest digest of a registry image tag.

    The cluster's enroot has no `digest` subcommand, so the digest that keys the sqsh cache
    is read straight from the Docker Registry v2 API, authenticating with the same
    `.credentials` entry enroot uses for `import`. A moving tag such as `:latest` therefore
    resolves to the digest of whatever image it currently points at.
    """
    registry, repository, tag = _split_registry_repository_tag(container_image)
    login, password = _enroot_registry_credentials(registry)

    # The registry answers an unauthenticated request with a 401 that names the token
    # endpoint to authenticate against.
    challenge = ""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"https://{registry}/v2/"), timeout=30
        ):
            pass
    except urllib.error.HTTPError as exc:
        challenge = exc.headers.get("WWW-Authenticate", "") or ""
    realm = re.search(r'realm="([^"]+)"', challenge)
    service = re.search(r'service="([^"]+)"', challenge)
    if realm is None or service is None:
        raise ValueError(
            f"{registry} did not issue a Bearer token challenge: {challenge!r}"
        )

    basic = base64.b64encode(f"{login}:{password}".encode()).decode()
    token_url = f"{realm.group(1)}?service={service.group(1)}&scope=repository:{repository}:pull"
    token_request = urllib.request.Request(
        token_url, headers={"Authorization": f"Basic {basic}"}
    )
    with urllib.request.urlopen(token_request, timeout=30) as response:
        token_response = json.load(response)
    token = token_response.get("token") or token_response.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError(f"{registry} token response did not include a token")

    manifest_request = urllib.request.Request(
        f"https://{registry}/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.oci.image.index.v1+json"
            ),
        },
    )
    with urllib.request.urlopen(manifest_request, timeout=30) as response:
        header_digest = response.headers.get("Docker-Content-Digest")
        manifest = response.read()
    if isinstance(header_digest, str) and header_digest:
        return header_digest
    return "sha256:" + hashlib.sha256(manifest).hexdigest()


def _split_registry_repository_tag(container_image: str) -> tuple[str, str, str]:
    """Return the (registry, repository, tag) of a Docker image reference.

    Accepts tag-based `registry/repo:tag` refs and Enroot `docker://registry#repo:tag`
    or `docker://registry/repo:tag` URIs. Digest-pinned refs should be passed as existing
    `.sqsh` paths after import; AlpaGym's moving-tag cache refresh path is tag based.
    """
    reference = container_image
    if "://" in reference:
        reference = reference.split("://", maxsplit=1)[1].replace("#", "/", 1)
    if "@sha256:" in reference:
        raise ValueError(
            "Docker image reference must be tag-based for AlpaGym cache refresh; "
            f"pass an existing .sqsh path for digest-pinned images: {container_image}"
        )
    registry, host_separator, remainder = reference.partition("/")
    if "@" in registry:
        registry = registry.rsplit("@", maxsplit=1)[1]
    repository, tag_separator, tag = remainder.rpartition(":")
    if (
        not registry
        or not host_separator
        or not repository
        or not tag_separator
        or "/" in tag
    ):
        raise ValueError(
            f"Docker image reference must be registry/repo:tag: {container_image}"
        )
    return registry, repository, tag


def _sqsh_filename(container_image: str, image_digest: str) -> str:
    """Return the digest-keyed sqsh filename for a registry image.

    The image reference contributes only the registry and repository name. Any mutable tag
    such as `:latest` or `:v2026.x` is removed before appending `image_digest`. This makes
    different tags that resolve to the same image share one `.sqsh`, while a moved tag
    resolves to a new cache file.
    """
    registry, repository, _ = _split_registry_repository_tag(container_image)
    digest_suffix = image_digest.replace(":", "_")
    filename = f"{registry}_{repository}_{digest_suffix}"
    return f"{re.sub(r'[^A-Za-z0-9._]+', '_', filename).strip('_')}.sqsh"


def _enroot_import_env() -> dict[str, str]:
    """Return the environment for importing Docker images with enroot."""
    enroot_config_path = os.environ.get("ENROOT_CONFIG_PATH")
    if not enroot_config_path or not Path(enroot_config_path).is_dir():
        raise ValueError("ENROOT_CONFIG_PATH must point to an enroot config directory")

    env = os.environ.copy()
    env["ENROOT_CONFIG_PATH"] = enroot_config_path
    env.setdefault("ENROOT_MAX_PROCESSORS", str(max(1, (os.cpu_count() or 2) // 2)))
    if "ENROOT_TEMP_PATH" not in env:
        env["ENROOT_TEMP_PATH"] = (
            f"/tmp/alpagym-enroot-import-{os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"
        )
    Path(env["ENROOT_TEMP_PATH"]).mkdir(parents=True, exist_ok=True)
    return env


def _validate_writable_directory(path: Path, config_key: str) -> None:
    """Validate that a directory accepts files from the submitting host."""
    probe_path = path / ".alpagym-write-test"
    try:
        probe_path.write_text("", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{config_key} must be writable: {path}") from exc
    finally:
        probe_path.unlink(missing_ok=True)


def _enroot_registry_credentials(registry: str) -> tuple[str, str]:
    """Return the (login, password) for a registry from enroot `.credentials`."""
    enroot_config_path = os.environ.get("ENROOT_CONFIG_PATH")
    if not enroot_config_path or not Path(enroot_config_path).is_dir():
        raise ValueError(
            "ENROOT_CONFIG_PATH must point to a directory containing enroot .credentials "
            "before AlpaGym can resolve or import a Docker image"
        )
    credentials_path = Path(enroot_config_path) / ".credentials"
    if not credentials_path.is_file():
        raise ValueError(
            f"{credentials_path} must exist before importing Docker images"
        )
    credentials = credentials_path.read_text(encoding="utf-8")
    match = re.search(
        rf"(?:^|\s)machine\s+{re.escape(registry)}\s+login\s+(\S+)\s+password\s+(\S+)",
        credentials,
    )
    if match is None:
        raise ValueError(
            f"{credentials_path} must contain a machine entry for {registry}"
        )
    return match.group(1), match.group(2)


def render_submit_script(
    execution_config: ExecutionConfig,
    artifact_paths: ArtifactPaths,
    project_root: Path,
    deploy: str,
    topology: str,
) -> str:
    """Render the sbatch script for an already prepared run.

    `deploy` and `topology` are the deploy- and topology-group choices of the
    submitting invocation. The compute-node re-entry loads the already-resolved
    config from disk, but Hydra still composes `default.yaml` first, where both
    groups are mandatory, so the re-entry must select them or compose fails
    before the CLI runs.
    """
    execution_backend = ExecutionBackend(execution_config.backend)
    if execution_backend is ExecutionBackend.slurm:
        validate_slurm_config(execution_config)
    else:
        raise ValueError("submit script rendering requires a Slurm backend")
    slurm = execution_config.slurm
    headers = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.job_name}",
        f"#SBATCH --time={slurm.time}",
        f"#SBATCH --nodes={slurm.nodes}",
        f"#SBATCH --gpus-per-node={slurm.gpus_per_node}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --output={artifact_paths.log_dir / 'slurm-%j.out'}",
        f"#SBATCH --error={artifact_paths.log_dir / 'slurm-%j.err'}",
    ]
    if slurm.cpus_per_task is not None:
        headers.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    if slurm.qos:
        headers.append(f"#SBATCH --qos={slurm.qos}")
    if slurm.mem is not None:
        headers.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.exclusive:
        headers.append("#SBATCH --exclusive")

    command = [
        "uv",
        "run",
        "--project",
        str(project_root),
        "--package",
        "alpagym-host",
        "python",
        "-m",
        "alpagym_host.cli",
        f"deploy={deploy}",
        f"topology={topology}",
        "command=run",
        f"execution.resolved_config_path={artifact_paths.resolved_config_path}",
    ]
    return "\n".join(
        [
            *headers,
            "",
            "set -euo pipefail",
            f"export UV_CACHE_DIR={shlex.quote(cast(str, slurm.uv_cache_dir))}",
            f"cd {shlex.quote(str(project_root))}",
            shlex.join(command),
            "",
        ]
    )


def submit_slurm_job(
    execution: ExecutionConfig,
    artifact_paths: ArtifactPaths,
    project_root: Path,
    deploy: str,
    topology: str,
) -> str:
    """Write submit.sbatch, submit it with sbatch, and return the job id."""
    artifact_paths.submit_script_path.write_text(
        render_submit_script(
            execution_config=execution,
            artifact_paths=artifact_paths,
            project_root=project_root,
            deploy=deploy,
            topology=topology,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["sbatch", str(artifact_paths.submit_script_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"Submitted batch job (\S+)", result.stdout)
    if match is None:
        raise RuntimeError(
            f"Could not parse sbatch job id from stdout: {result.stdout!r}"
        )
    return match.group(1)
