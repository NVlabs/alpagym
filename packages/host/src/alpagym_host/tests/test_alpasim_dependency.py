# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from alpagym_host import alpasim_dependency
from alpagym_host.alpasim_dependency import resolve_alpasim_checkout
from alpagym_host.config import AlpaSimConfig, AlpaSimWizardArgs

REPO_URL = "ssh://git@example/alpasim.git"


def _content_key(repo_url: str, commit: str) -> str:
    """Return the content-addressed cache dir name for a repo and commit."""
    return hashlib.sha256(f"{repo_url}@{commit}".encode()).hexdigest()[:12]


def test_cached_checkout_builds_in_temp_then_publishes(tmp_path, monkeypatch) -> None:
    """Builds a content-addressed checkout in a private temp dir, then publishes it."""
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", _recording_run(commands))

    commit = "b" * 40
    dest = resolve_alpasim_checkout(_alpasim_config(repo_ref=commit))

    assert dest.is_dir()
    assert dest.parent == tmp_path / "cache" / "alpagym" / "alpasim"
    assert dest.name == _content_key(REPO_URL, commit)

    clone = next(c for c in commands if c[:2] == ["git", "clone"])
    build_dir = Path(clone[3])
    assert build_dir != dest  # built aside, not directly in the published path
    assert build_dir.parent == dest.parent

    assert ["git", "clone", REPO_URL, str(build_dir)] in commands
    assert ["git", "checkout", commit] in commands
    assert ["uv", "run", "compile-protos"] in commands
    assert ["uv", "venv", "--relocatable", str(build_dir / ".venv")] in commands
    assert ["uv", "sync", "--all-extras", "--no-editable"] in commands

    # The removed in-place mutations and the dropped editable configs install.
    assert not any(c[:2] == ["git", "fetch"] for c in commands)
    assert not any(c[:2] == ["git", "status"] for c in commands)
    assert not any(c[:3] == ["uv", "pip", "install"] for c in commands)


def test_cached_checkout_resolves_branch_ref_to_commit(tmp_path, monkeypatch) -> None:
    """A branch ref is resolved to its commit with `git ls-remote` before keying the cache."""
    commands: list[list[str]] = []
    resolved = "c" * 40
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        alpasim_dependency.subprocess,
        "run",
        _recording_run(commands, ls_remote_sha=resolved),
    )

    dest = resolve_alpasim_checkout(_alpasim_config(repo_ref="main"))

    assert ["git", "ls-remote", REPO_URL, "main"] in commands
    assert ["git", "checkout", resolved] in commands
    assert dest.name == _content_key(REPO_URL, resolved)


def test_cached_checkout_skips_build_when_already_published(
    tmp_path, monkeypatch
) -> None:
    """A second run on the same commit reuses the published checkout and builds nothing."""
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", _recording_run(commands))

    commit = "b" * 40
    first = resolve_alpasim_checkout(_alpasim_config(repo_ref=commit))
    commands.clear()
    second = resolve_alpasim_checkout(_alpasim_config(repo_ref=commit))

    assert second == first
    assert commands == []


def test_cached_checkout_reuses_winner_on_publish_race(tmp_path, monkeypatch) -> None:
    """When a concurrent run publishes first, discard our build and reuse the winner."""
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", _recording_run(commands))

    def racing_replace(src, dst) -> None:
        # Simulate a concurrent run that published `dst` first: it is now a
        # non-empty directory, so the rename fails as it would on a real FS.
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "winner").write_text("winner", encoding="utf-8")
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(alpasim_dependency.os, "replace", racing_replace)

    commit = "b" * 40
    dest = resolve_alpasim_checkout(_alpasim_config(repo_ref=commit))

    assert dest.name == _content_key(REPO_URL, commit)
    assert (dest / "winner").read_text(encoding="utf-8") == "winner"
    build_dir = Path(next(c for c in commands if c[:2] == ["git", "clone"])[3])
    assert not build_dir.exists()  # our losing build was cleaned up


def test_concurrent_checkout_publishes_once_without_corruption(
    tmp_path, monkeypatch
) -> None:
    """Real concurrent resolves of one commit publish exactly one intact checkout.

    Several threads build the same content-addressed checkout into one shared cache at
    once. A barrier holds every thread at the `os.replace` publish until all have finished
    building, so they truly race the atomic rename. The race is on the shared filesystem,
    which the kernel arbitrates the same way for threads and for separate deploy
    processes, so this exercises the multi-run path without a live Slurm sweep.
    """
    workers = 6
    checkout_cache_dir = tmp_path / "shared-cache"
    commit = "b" * 40
    barrier = threading.Barrier(workers)
    real_replace = alpasim_dependency.os.replace
    lock = threading.Lock()
    outcomes = {"won": 0, "lost": 0}
    lost_errnos: list[int] = []

    def run(command, cwd=None, env=None, check=True, text=True, capture_output=False):
        del cwd, env, check, text, capture_output
        if command[:2] == ["git", "clone"]:
            _write_alpasim_layout(Path(command[3]))
        if command[:3] == ["uv", "venv", "--relocatable"]:
            _write_relocatable_venv(Path(command[3]))
        return SimpleNamespace(stdout="")

    def barrier_replace(src, dst):
        # Hold every builder at the contended rename, then let them all race at once.
        barrier.wait(timeout=30)
        try:
            real_replace(src, dst)
        except OSError as exc:
            with lock:
                outcomes["lost"] += 1
                lost_errnos.append(exc.errno)
            raise
        with lock:
            outcomes["won"] += 1

    monkeypatch.setattr(alpasim_dependency.subprocess, "run", run)
    monkeypatch.setattr(alpasim_dependency.os, "replace", barrier_replace)

    def resolve() -> Path:
        return resolve_alpasim_checkout(
            _alpasim_config(repo_ref=commit, checkout_cache_dir=str(checkout_cache_dir))
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [
            future.result() for future in [pool.submit(resolve) for _ in range(workers)]
        ]

    published = checkout_cache_dir / _content_key(REPO_URL, commit)
    assert all(
        result == published for result in results
    )  # every run got the published checkout
    assert (
        published / "pyproject.toml"
    ).is_file()  # complete checkout, not a half-built temp dir
    assert (published / "src" / "grpc" / "pyproject.toml").is_file()
    assert outcomes["won"] == 1  # exactly one builder won the publish
    assert (
        outcomes["lost"] == workers - 1
    )  # the rest lost the rename race and reused the winner
    assert all(e in (errno.ENOTEMPTY, errno.EEXIST) for e in lost_errnos)
    assert [
        p for p in checkout_cache_dir.iterdir() if p.name.startswith(".build-")
    ] == []


def test_cached_checkout_isolates_by_commit(tmp_path, monkeypatch) -> None:
    """Different commits resolve to different published checkouts."""
    commands: list[list[str]] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", _recording_run(commands))

    first = resolve_alpasim_checkout(_alpasim_config(repo_ref="a" * 40))
    second = resolve_alpasim_checkout(_alpasim_config(repo_ref="d" * 40))

    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_cached_checkout_pins_uv_envs(tmp_path, monkeypatch) -> None:
    """All uv build steps target and fully prepare the checkout environment."""
    calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    def run(command, cwd=None, env=None, check=True, text=True, capture_output=False):
        del check, text, capture_output
        calls.append((command, cwd, env))
        if command[:2] == ["git", "clone"]:
            _write_alpasim_layout(Path(command[3]))
        if command[:3] == ["uv", "venv", "--relocatable"]:
            _write_relocatable_venv(Path(command[3]))
        return SimpleNamespace(stdout="")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", run)

    resolve_alpasim_checkout(_alpasim_config(repo_ref="b" * 40))

    build_dir = Path(next(c for c, _, _ in calls if c[:2] == ["git", "clone"])[3])
    uv_calls = [(c, env) for c, _, env in calls if c[0] == "uv"]
    assert uv_calls  # there is at least one uv build step
    for command, env in uv_calls:
        assert env is not None, command
        assert "VIRTUAL_ENV" not in env
        assert "UV_NO_SYNC" not in env
    compile_env = next(
        env for c, env in uv_calls if c == ["uv", "run", "compile-protos"]
    )
    assert compile_env["UV_PROJECT_ENVIRONMENT"] == str(
        build_dir / "src" / "grpc" / ".venv"
    )
    sync_env = next(env for c, env in uv_calls if c[:2] == ["uv", "sync"])
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == str(build_dir / ".venv")


def test_cached_checkout_restores_plugin_configs_dropped_by_no_editable(
    tmp_path, monkeypatch
) -> None:
    """The relocatable build copies each plugin's source configs into its installed wheel.

    `uv sync --no-editable` installs the plugins without their Hydra config YAMLs, so
    `pkg://alpasim_<plugin>.configs` would be empty and the Wizard could not resolve
    the groups they contribute. The build must restore the YAMLs into the published venv.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def run(command, cwd=None, env=None, check=True, text=True, capture_output=False):
        del env, check, text, capture_output
        if command[:2] == ["git", "clone"]:
            build = Path(command[3])
            _write_alpasim_layout(build)
            # A plugin ships Hydra config groups in its source tree.
            deploy = build / "plugins" / "example" / "configs" / "deploy"
            deploy.mkdir(parents=True)
            (deploy / "cluster.yaml").write_text("name: cluster\n", encoding="utf-8")
            (build / "plugins" / "example" / "pyproject.toml").write_text(
                '[project]\nname = "alpasim-example"\n', encoding="utf-8"
            )
        if command[:2] == ["uv", "sync"]:
            # `--no-editable` installs the plugin wheel without those YAMLs.
            pkg = (
                Path(str(cwd))
                / ".venv"
                / "lib"
                / "python3.12"
                / "site-packages"
                / "alpasim_example"
            )
            (pkg / "configs").mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "configs" / "__init__.py").write_text("", encoding="utf-8")
        return SimpleNamespace(
            stdout="b" * 40 if command[:2] == ["git", "ls-remote"] else ""
        )

    monkeypatch.setattr(alpasim_dependency.subprocess, "run", run)

    dest = resolve_alpasim_checkout(_alpasim_config(repo_ref="b" * 40))

    restored = (
        dest
        / ".venv/lib/python3.12/site-packages/alpasim_example/configs/deploy/cluster.yaml"
    )
    assert (
        restored.is_file()
    )  # the Wizard can now resolve pkg://alpasim_example.configs deploy=cluster


def test_local_checkout_is_validated_without_mutation(
    tmp_path, monkeypatch
) -> None:
    """An explicit local checkout is treated as a prepared immutable input."""
    local_checkout = tmp_path / "alpasim"
    _write_alpasim_layout(local_checkout)
    (local_checkout / ".venv" / "bin").mkdir(parents=True)
    (local_checkout / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    commands: list[tuple[list[str], Path | None]] = []

    def run(command, cwd=None, **kwargs):
        del kwargs
        commands.append((command, cwd))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(alpasim_dependency.subprocess, "run", run)

    checkout_root = resolve_alpasim_checkout(_local_config(local_checkout))

    assert checkout_root == local_checkout.resolve()
    assert commands == []


def test_cached_checkout_uses_configured_cache_dir(tmp_path, monkeypatch) -> None:
    """Configured checkout caches avoid node-local home directories."""
    commands: list[list[str]] = []
    checkout_cache_dir = tmp_path / "lustre-cache"
    monkeypatch.setattr(alpasim_dependency.subprocess, "run", _recording_run(commands))

    dest = resolve_alpasim_checkout(
        _alpasim_config(repo_ref="b" * 40, checkout_cache_dir=str(checkout_cache_dir))
    )

    assert dest.parent == checkout_cache_dir


def test_validate_alpasim_checkout_cache_creates_configured_dir(tmp_path: Path) -> None:
    """Preflight creates the configured cached-checkout directory."""
    checkout_cache_dir = tmp_path / "missing" / "alpasim"

    alpasim_dependency.validate_alpasim_checkout_cache(
        _alpasim_config(repo_ref="main", checkout_cache_dir=str(checkout_cache_dir))
    )

    assert checkout_cache_dir.is_dir()


def _recording_run(commands: list, *, ls_remote_sha: str = "a" * 40):
    """Return a `subprocess.run` replacement recording commands and faking git/uv.

    `git clone` writes the AlpaSim layout into the clone target so the build steps
    that follow find it; `git ls-remote` returns `ls_remote_sha`.
    """

    def run(command, cwd=None, env=None, check=True, text=True, capture_output=False):
        del cwd, env, check, text, capture_output
        commands.append(command)
        if command[:2] == ["git", "clone"]:
            _write_alpasim_layout(Path(command[3]))
        if command[:3] == ["uv", "venv", "--relocatable"]:
            _write_relocatable_venv(Path(command[3]))
        if command[:2] == ["git", "ls-remote"]:
            return SimpleNamespace(stdout=f"{ls_remote_sha}\trefs/heads/branch\n")
        return SimpleNamespace(stdout="")

    return run


def _alpasim_config(
    repo_ref: str, checkout_cache_dir: str | None = None
) -> AlpaSimConfig:
    """Build an AlpaSim config for the cached-checkout tests."""
    return AlpaSimConfig(
        repo_url=REPO_URL,
        repo_ref=repo_ref,
        checkout_cache_dir=checkout_cache_dir,
        startup_timeout_s=600.0,
        simulation_timeout_s=600.0,
        wizard_args=_wizard_args(),
    )


def _local_config(local_checkout: Path) -> AlpaSimConfig:
    """Build an AlpaSim config that points at a local checkout."""
    return AlpaSimConfig(
        repo_path=str(local_checkout),
        startup_timeout_s=600.0,
        simulation_timeout_s=600.0,
        wizard_args=_wizard_args(),
    )


def _write_relocatable_venv(venv: Path) -> None:
    """Model the site-packages directory `uv venv --relocatable` always creates.

    The build fakes record commands instead of running uv, so they must create this
    themselves; otherwise _install_plugin_configs sees no site-packages and fails fast.
    """
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True, exist_ok=True)


def _write_alpasim_layout(checkout_root: Path) -> None:
    """Write the AlpaSim files the dependency resolver requires."""
    pyproject = checkout_root / "pyproject.toml"
    pyproject.parent.mkdir(parents=True, exist_ok=True)
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "alpasim"',
                "",
                "[project.optional-dependencies]",
                "wizard = []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for relative_path in [
        "src/wizard/configs/base_config.yaml",
        "src/wizard/pyproject.toml",
        "src/runtime/pyproject.toml",
        "src/grpc/pyproject.toml",
    ]:
        path = checkout_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _wizard_args() -> AlpaSimWizardArgs:
    """Build Wizard args for dependency resolver tests."""
    return AlpaSimWizardArgs(
        deploy="local",
        topology="1gpu",
        driver_source="external_dynamic",
        force_gt_duration_us=3_000_000,
        control_timestep_us=100_000,
        n_sim_steps=38,
    )
