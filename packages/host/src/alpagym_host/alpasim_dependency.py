# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from alpagym_host.config import AlpaSimConfig


def resolve_alpasim_checkout(
    config: AlpaSimConfig, *, prepare_local_env: bool = True
) -> Path:
    """Resolve or prepare the AlpaSim checkout for Wizard startup.

    For a local ``repo_path``, optionally sync its environment in place. Slurm
    callers using a prebuilt container set ``prepare_local_env=False`` and use
    the checkout only as a source overlay. For a ``repo_url`` +
    ``repo_ref``, return a content-addressed cached checkout built once: concurrent
    runs that share the cache build into private temp dirs and publish the first one
    with an atomic rename, so a concurrent sweep never corrupts a shared checkout.
    """
    if config.repo_path is not None:
        checkout_root = Path(config.repo_path).expanduser().resolve()
        if not checkout_root.is_dir():
            raise NotADirectoryError(checkout_root)
        logging.info("Using local AlpaSim checkout %s", checkout_root)
        _validate_alpasim_layout(checkout_root)
        if prepare_local_env:
            _compile_protos(checkout_root)
            _sync_alpasim_env(checkout_root, relocatable=False)
        else:
            logging.info(
                "Using prebuilt container environment for local AlpaSim checkout %s",
                checkout_root,
            )
        return checkout_root

    if config.repo_url is None or config.repo_ref is None:
        raise ValueError(
            "AlpaSim config requires repo_url and repo_ref when repo_path is not set"
        )

    commit = _resolve_commit_sha(config.repo_url, config.repo_ref)
    checkout_cache_dir = _checkout_cache_dir(config)
    checkout_root = (
        checkout_cache_dir
        / hashlib.sha256(f"{config.repo_url}@{commit}".encode()).hexdigest()[:12]
    )
    if checkout_root.is_dir():
        # A prior run already built and published this commit; consume it read-only.
        logging.info("Reusing AlpaSim checkout %s (%s)", checkout_root, commit)
        return checkout_root

    checkout_cache_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(dir=checkout_cache_dir, prefix=".build-"))
    try:
        logging.info(
            "Building AlpaSim checkout %s@%s in %s", config.repo_url, commit, build_dir
        )
        subprocess.run(
            ["git", "clone", config.repo_url, str(build_dir)], check=True, text=True
        )
        subprocess.run(
            ["git", "checkout", commit], cwd=build_dir, check=True, text=True
        )
        _validate_alpasim_layout(build_dir)
        _compile_protos(build_dir)
        _sync_alpasim_env(build_dir, relocatable=True)
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    # Publish with an atomic rename. `os.replace` onto a non-empty directory raises
    # (ENOTEMPTY/EEXIST) when a concurrent run published first: discard our build and
    # reuse the winner. Any other error (e.g. EXDEV across filesystems) propagates.
    try:
        os.replace(build_dir, checkout_root)
        logging.info("Published AlpaSim checkout %s (%s)", checkout_root, commit)
    except OSError:
        shutil.rmtree(build_dir, ignore_errors=True)
        if not checkout_root.is_dir():
            raise
        logging.info("Concurrent run published %s first; reusing it", checkout_root)
    return checkout_root


def _resolve_commit_sha(repo_url: str, repo_ref: str) -> str:
    """Resolve an AlpaSim ref to an immutable commit SHA for content-addressing.

    A 40-character hex ref is already a commit. A branch or tag is resolved with
    `git ls-remote`, so the cache key tracks the exact commit rather than a moving
    name.
    """
    if re.fullmatch(r"[0-9a-f]{40}", repo_ref):
        return repo_ref
    refs = subprocess.run(
        ["git", "ls-remote", repo_url, repo_ref],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.split()
    if not refs:
        raise ValueError(f"git ls-remote found no ref {repo_ref!r} in {repo_url}")
    return refs[0]


def validate_alpasim_checkout_cache(config: AlpaSimConfig) -> None:
    """Validate that the configured checkout-cache directory is writable.

    A set repo_path uses that checkout directly, so the cache is unused and there is
    nothing to validate or create.
    """
    if config.repo_path is not None or config.checkout_cache_dir is None:
        return

    checkout_cache_dir = Path(config.checkout_cache_dir).expanduser()
    try:
        checkout_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"alpasim.checkout_cache_dir must be creatable: {checkout_cache_dir}"
        ) from exc

    probe_path = checkout_cache_dir / ".alpagym-write-test"
    try:
        probe_path.write_text("", encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"alpasim.checkout_cache_dir must be writable: {checkout_cache_dir}"
        ) from exc
    finally:
        probe_path.unlink(missing_ok=True)


def _sync_alpasim_env(checkout_root: Path, relocatable: bool) -> None:
    """Sync the AlpaSim project environment Wizard runs from.

    With ``relocatable=True`` (a cached checkout), build a relocatable venv and
    install members as wheels (``--no-editable``), so the venv survives the atomic
    rename to the content-addressed path. With ``False`` (a local checkout), build
    the usual editable venv in place.
    """
    venv = checkout_root / ".venv"
    if relocatable:
        subprocess.run(
            ["uv", "venv", "--relocatable", str(venv)],
            cwd=checkout_root,
            env=uv_env(venv),
            check=True,
            text=True,
        )
    # Sync every optional extra the checkout declares -- always the Wizard, plus any
    # site-specific extras a particular checkout adds -- so one sync path covers all
    # checkouts.
    command = ["uv", "sync", "--all-extras"]
    if relocatable:
        command.append("--no-editable")
    subprocess.run(command, cwd=checkout_root, env=uv_env(venv), check=True, text=True)
    if relocatable:
        # `--no-editable` wheels drop the plugins' Hydra config YAMLs, so restore them
        # here -- otherwise the Wizard cannot resolve `pkg://alpasim_<plugin>.configs`
        # for an installed plugin. An editable install (the local path below) does not
        # need this, since it imports those configs straight from the source tree.
        _install_plugin_configs(checkout_root, venv)


def _compile_protos(checkout_root: Path) -> None:
    """Generate checkout-local protobuf modules before any service imports them."""
    grpc_root = checkout_root / "src" / "grpc"
    subprocess.run(
        ["uv", "run", "compile-protos"],
        cwd=grpc_root,
        env=uv_env(grpc_root / ".venv"),
        check=True,
        text=True,
    )


def _install_plugin_configs(checkout_root: Path, venv: Path) -> None:
    """Copy each AlpaSim plugin's source ``configs`` tree into its installed package.

    ``uv sync --no-editable`` installs the AlpaSim plugins as wheels, but those wheels
    omit the plugins' Hydra config YAMLs (the plugin pyprojects do not declare them as
    package data). The Wizard adds ``pkg://alpasim_<plugin>.configs`` to its search path
    for each installed plugin, so without the YAMLs that provider is empty and the
    config groups it contributes are unresolvable. Copying the source configs over the installed
    package here -- in the build dir, before the atomic publish -- keeps the published
    checkout self-contained and avoids the editable ``.pth`` paths that the publish
    rename would invalidate.
    """
    site_packages = next(venv.glob("lib/python*/site-packages"), None)
    if site_packages is None:
        # Fail fast: a synced venv always has site-packages. Returning here would
        # publish a checkout whose plugin configs are missing and only surface much
        # later as a confusing Hydra resolution error.
        raise FileNotFoundError(
            f"relocatable venv has no site-packages directory: {venv}"
        )
    for plugin_dir in sorted((checkout_root / "plugins").glob("*")):
        source_configs = plugin_dir / "configs"
        plugin_pyproject = plugin_dir / "pyproject.toml"
        if not source_configs.is_dir() or not plugin_pyproject.is_file():
            continue
        # The wheel installs the plugin under its import name (pyproject name, ``-``->``_``).
        package = tomllib.loads(plugin_pyproject.read_text(encoding="utf-8"))[
            "project"
        ]["name"]
        installed_package = site_packages / package.replace("-", "_")
        # Skip plugins this sync did not install (e.g. an extra that was not requested).
        if not installed_package.is_dir():
            continue
        shutil.copytree(
            source_configs, installed_package / "configs", dirs_exist_ok=True
        )


def _validate_alpasim_layout(checkout_root: Path) -> None:
    """Validate that the checkout contains the Wizard files AlpaGym needs."""
    for relative_path in [
        "pyproject.toml",
        "src/wizard/configs/base_config.yaml",
        "src/wizard/pyproject.toml",
        "src/runtime/pyproject.toml",
        "src/grpc/pyproject.toml",
    ]:
        path = checkout_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)


def uv_env(environment: Path) -> dict[str, str]:
    """Return a subprocess env that points uv at a specific project environment.

    `UV_PROJECT_ENVIRONMENT` names the directory uv uses as the current project's
    environment for `uv run`/`uv sync`/`uv venv`; this helper is the single place
    that sets it. It matters because the host-side AlpaSim build runs `uv` against
    the checkout while the ambient `UV_PROJECT_ENVIRONMENT` points at the runtime's
    `/opt/venv`. `uv sync` rewrites whatever this points at, so a checkout uv call
    that did not set it would sync into, and prune, the runtime's `/opt/venv`.
    `VIRTUAL_ENV` is dropped only to silence uv's "does not match the project
    environment" warning; uv ignores it for project commands. `UV_NO_SYNC` is also
    dropped: dependency preparation explicitly owns the checkout sync and protobuf
    generation, so an ambient launcher optimization must not turn ``uv run
    compile-protos`` into a no-sync invocation against an empty fresh environment.
    See the README "uv environments" section for a diagram.

    Args:
        environment: Path to the project virtual environment uv should use,
            e.g. `<checkout>/.venv`.

    Returns:
        A copy of the current environment with `UV_PROJECT_ENVIRONMENT` set to
        `environment` and `VIRTUAL_ENV`/`UV_NO_SYNC` removed.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_NO_SYNC", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    return env


def _checkout_cache_dir(config: AlpaSimConfig) -> Path:
    """Return the directory containing cached AlpaSim checkouts."""
    if config.checkout_cache_dir is not None:
        return Path(config.checkout_cache_dir).expanduser()
    cache_base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_base / "alpagym" / "alpasim"
