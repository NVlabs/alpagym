# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
import yaml
from alpagym_host.alpasim_wizard import wizard_compose_project
from alpagym_host.formal_run_provenance import (
    FormalRunProvenance,
    RepositorySource,
    _canonical_sha256,
    _capture_ppo_update_diagnostic_seals,
    _read_runtime_ready_receipt,
    _validate_ppo_update_diagnostic_receipt,
    build_import_probe_receipt,
)

_TEST_IMAGE_ID = "sha256:" + "a" * 64
_TEST_MANIFEST_DIGEST = "sha256:" + "b" * 64


def test_formal_provenance_copies_effective_dirty_sources_and_invalidates_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed run is invalid if any byte of its effective source later changes."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    owner = FormalRunProvenance.capture_prelaunch(
        provenance_dir=run_dir / "provenance",
        repositories=(
            RepositorySource(name="alpagym", root=alpagym),
            RepositorySource(
                name="alpasim",
                root=alpasim,
                capture_ignored_generated_pb2=True,
            ),
            RepositorySource(name="humanoid", root=humanoid),
        ),
        resolved_config_path=resolved_config,
        cosmos_config_path=cosmos_config,
        critical_environment=_test_critical_environment(
            tmp_path=tmp_path,
            alpasim=alpasim,
            monkeypatch=monkeypatch,
        ),
    )
    prelaunch = json.loads(
        (run_dir / "provenance" / "prelaunch" / "manifest.json").read_text()
    )
    assert prelaunch["repositories"]["alpagym"]["tracked_patch"]["size_bytes"] > 0
    assert (
        run_dir
        / "provenance"
        / "prelaunch"
        / "repositories"
        / "alpagym"
        / "untracked"
        / "notes"
        / "new.txt"
    ).read_text() == "untracked\n"
    ignored_entries = prelaunch["repositories"]["alpasim"]["ignored_generated_pb2"][
        "files"
    ]
    assert [entry["path"] for entry in ignored_entries] == [
        "src/grpc/alpasim_grpc/v0/humanoid_pb2.py"
    ]
    assert (
        run_dir
        / "provenance"
        / "prelaunch"
        / "repositories"
        / "alpasim"
        / "ignored_generated_pb2"
        / "src"
        / "grpc"
        / "alpasim_grpc"
        / "v0"
        / "humanoid_pb2.py"
    ).read_text() == "generated ABI\n"

    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )
    owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["uv", "run", "cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )
    (humanoid / "notes" / "new.txt").write_text("changed after launch\n")
    checkpoint = run_dir / "cosmos" / "checkpoint.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained")

    with pytest.raises(RuntimeError, match="provenance, source-watch, or cleanup"):
        owner.finalize(run_completed=True)

    postrun = json.loads((run_dir / "provenance" / "postrun.json").read_text())
    assert postrun["formal_run_valid"] is False
    assert postrun["repositories"]["humanoid"]["matches_prelaunch"] is False
    assert checkpoint.read_bytes() == b"retained"


@pytest.mark.parametrize("attack", ["descriptor_method", "generated_binding"])
def test_runtime_ready_rejects_incomplete_abort_abi_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    """A local proto origin is insufficient without both AbortSession bindings."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )
    probe = _test_import_probe(alpagym=alpagym, alpasim=alpasim)
    if attack == "descriptor_method":
        del probe["descriptor_services"]["HumanoidPolicyService"]["abort_session"]
    else:
        probe["grpc_bindings"]["dynamics_stub_abort_session"] = False
    body = dict(probe)
    body.pop("identity_sha256")
    probe["identity_sha256"] = _canonical_sha256(body)

    with pytest.raises(RuntimeError, match="AbortSession|method names"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["uv", "run", "cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=probe,
        )
    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_qualification_command_outside_host_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A qualification receipt cannot attest a command other than this process."""
    owner, alpagym, alpasim, _humanoid = _prepare_formal_owner(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="differs from the captured host invocation"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(tmp_path / "wizard",),
            workload_kind="qualification_rollout",
            workload_command=["python", "attacker.py"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_attests_the_exact_qualification_host_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen qualification uses the same source/runtime admission as training."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )
    expected_command = [
        owner.prelaunch_manifest["invocation"]["python_executable"],
        *owner.prelaunch_manifest["invocation"]["argv"],
    ]

    receipt_path = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_kind="qualification_rollout",
        workload_command=expected_command,
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_id"] == "alpagym.formal_run_runtime_ready.v5"
    assert receipt["workload_kind"] == "qualification_rollout"
    assert receipt["workload_command"] == expected_command
    assert receipt["scene_cache_identity"]["file_count"] == 0


def test_runtime_ready_receipt_records_actual_image_compose_and_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cosmos is admitted only after live containers match snapshotted source mounts."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    owner = FormalRunProvenance.capture_prelaunch(
        provenance_dir=tmp_path / "run" / "provenance",
        repositories=(
            RepositorySource(name="alpagym", root=alpagym),
            RepositorySource(
                name="alpasim",
                root=alpasim,
                capture_ignored_generated_pb2=True,
            ),
            RepositorySource(name="humanoid", root=humanoid),
        ),
        resolved_config_path=resolved_config,
        cosmos_config_path=cosmos_config,
        critical_environment=_test_critical_environment(
            tmp_path=tmp_path,
            alpasim=alpasim,
            monkeypatch=monkeypatch,
        ),
    )
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )

    receipt_path = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["uv", "run", "cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    receipt = json.loads(receipt_path.read_text())
    assert (
        receipt["prelaunch_portable_source_set_sha256"]
        == owner.prelaunch_manifest["portable_source_set_sha256"]
    )
    assert (
        receipt["prelaunch_config_artifacts_identity_sha256"]
        == (owner.prelaunch_manifest["config_artifacts"]["identity_sha256"])
    )
    runtime = receipt["runtimes"][0]
    assert runtime["compose_sha256"]
    assert runtime["images"] == [
        {
            "id": _TEST_IMAGE_ID,
            "labels": {"org.example.runtime-contract": "v1"},
            "repo_digests": ["runtime@example-sha256"],
            "repo_tags": ["runtime:local"],
        }
    ]
    assert [service["service"] for service in runtime["services"]] == [
        "humanoid_dynamics-0",
        "runtime-0",
    ]
    assert all(service["compose_config_hash"] for service in runtime["services"])
    assert all(
        service["image_config_id"] == _TEST_IMAGE_ID
        and service["oci_platform_manifest_digest"] == _TEST_MANIFEST_DIGEST
        for service in runtime["services"]
    )
    for service in runtime["services"]:
        mounts = {
            mount["destination"]: mount
            for mount in service["mounts"]
            if mount["type"] == "bind"
        }
        assert mounts["/mnt/humanoid-scene-cache"]["rw"] is False
        assert mounts["/root/.cache"]["rw"] is True


def test_runtime_capture_queries_exact_wizard_project_and_isolates_other_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime discovery cannot fall back to another Compose project."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    compose_path = wizard_log_dir / "docker-compose.yaml"
    expected_project = wizard_compose_project(wizard_log_dir)
    other_wizard_dir = tmp_path / "other_wizard"
    other_wizard_dir.mkdir()
    other_project = wizard_compose_project(other_wizard_dir)
    assert other_project != expected_project
    expected_query = [
        "docker",
        "compose",
        "--project-name",
        expected_project,
        "--file",
        str(compose_path),
        "ps",
        "--all",
        "--quiet",
    ]
    delegate = _mock_docker_command(
        alpasim=alpasim,
        humanoid=humanoid,
        compose_path=compose_path,
    )
    compose_queries: list[list[str]] = []
    inspected_ids: list[str] = []

    def run(command: list[str]) -> bytes:
        if command[:2] == ["docker", "compose"]:
            compose_queries.append(command)
            if command == expected_query:
                return b"runtime-0-id\nhumanoid_dynamics-0-id\n"
            # Model an unrelated Compose project that an unscoped query could see.
            return b"other-runtime-id\nother-humanoid-dynamics-id\n"
        if command[:2] == ["docker", "inspect"]:
            inspected_ids.extend(command[2:])
            assert command[2:] == ["runtime-0-id", "humanoid_dynamics-0-id"]
        return delegate(command)

    monkeypatch.setattr("alpagym_host.formal_run_provenance._run_command", run)

    receipt_path = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    assert compose_queries == [expected_query]
    assert inspected_ids == ["runtime-0-id", "humanoid_dynamics-0-id"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["runtimes"][0]["compose_project"] == expected_project


def test_runtime_capture_rejects_container_from_wrong_compose_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every inspected container must carry the pinned Wizard project label."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    compose_path = wizard_log_dir / "docker-compose.yaml"
    other_wizard_dir = tmp_path / "other_wizard"
    other_wizard_dir.mkdir()
    other_project = wizard_compose_project(other_wizard_dir)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=compose_path,
            compose_project=other_project,
        ),
    )

    with pytest.raises(RuntimeError, match="pinned Wizard launch project"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_container_without_snapshotted_source_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running container is insufficient when it executes a different source tree."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    owner = FormalRunProvenance.capture_prelaunch(
        provenance_dir=tmp_path / "run" / "provenance",
        repositories=(
            RepositorySource(name="alpagym", root=alpagym),
            RepositorySource(
                name="alpasim",
                root=alpasim,
                capture_ignored_generated_pb2=True,
            ),
            RepositorySource(name="humanoid", root=humanoid),
        ),
        resolved_config_path=resolved_config,
        cosmos_config_path=cosmos_config,
        critical_environment=_test_critical_environment(
            tmp_path=tmp_path,
            alpasim=alpasim,
            monkeypatch=monkeypatch,
        ),
    )
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            omit_humanoid_mount=True,
        ),
    )

    with pytest.raises(RuntimeError, match="live bind mounts differ"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )
    assert not (tmp_path / "run" / "provenance" / "runtime_ready.json").exists()


def test_runtime_ready_rehashes_sources_before_any_docker_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source drift after prelaunch refuses runtime admission before Docker runs."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    delegate = _mock_docker_command(
        alpasim=alpasim,
        humanoid=humanoid,
        compose_path=wizard_log_dir / "docker-compose.yaml",
    )
    docker_commands: list[list[str]] = []

    def run(command: list[str]) -> bytes:
        if command[0] == "docker":
            docker_commands.append(command)
        return delegate(command)

    monkeypatch.setattr("alpagym_host.formal_run_provenance._run_command", run)
    (humanoid / "tracked.py").write_text("drift before runtime inspection\n")

    with pytest.raises(RuntimeError, match="source watch observed a mutation"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert docker_commands == []
    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_finalize_rejects_tampered_runtime_receipt_and_writes_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed runtime receipt cannot be silently accepted at finalization."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    runtime_ready_path.chmod(0o644)
    receipt = json.loads(runtime_ready_path.read_text())
    receipt["receipt_sha256"] = "0" * 64
    runtime_ready_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="in-memory lifecycle binding"):
        owner.finalize(run_completed=True)

    _assert_durable_failure_receipt(owner.provenance_dir / "postrun.json")


def test_finalize_rejects_rewritten_runtime_receipt_with_recomputed_self_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-consistent replacement still differs from the in-memory trust root."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    runtime_ready_path.chmod(0o644)
    receipt = json.loads(runtime_ready_path.read_text())
    receipt["workload_command"] = ["attacker", "replacement"]
    receipt["workload_command_sha256"] = _canonical_sha256(receipt["workload_command"])
    del receipt["receipt_sha256"]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    runtime_ready_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="in-memory lifecycle binding"):
        owner.finalize(run_completed=True)

    _assert_durable_failure_receipt(owner.provenance_dir / "postrun.json")


def test_runtime_receipt_rejects_recomputed_wrong_prelaunch_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid self-hash cannot replace the frozen prelaunch source binding."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    runtime_ready_path.chmod(0o644)
    receipt = json.loads(runtime_ready_path.read_text())
    receipt["prelaunch_portable_source_set_sha256"] = "f" * 64
    del receipt["receipt_sha256"]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    runtime_ready_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="source binding differs"):
        _read_runtime_ready_receipt(
            runtime_ready_path,
            expected_source_set_sha256=owner.prelaunch_manifest[
                "portable_source_set_sha256"
            ],
            expected_config_identity_sha256=owner.prelaunch_manifest[
                "config_artifacts"
            ]["identity_sha256"],
            expected_critical_environment_sha256=owner.prelaunch_manifest[
                "critical_environment_sha256"
            ],
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["malformed_labels", "service_image_mismatch", "service_manifest_mismatch"],
)
def test_runtime_receipt_v5_rejects_invalid_exact_image_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    runtime_ready_path.chmod(0o644)
    receipt = json.loads(runtime_ready_path.read_text())
    runtime = receipt["runtimes"][0]
    if mutation == "malformed_labels":
        runtime["images"][0]["labels"] = ["not", "a", "mapping"]
        error = "image identity is invalid"
    elif mutation == "service_image_mismatch":
        runtime["services"][0]["image_config_id"] = "sha256:" + "c" * 64
        error = "service.*image identity is invalid"
    else:
        changed_digest = "sha256:" + "c" * 64
        runtime["services"][0]["oci_platform_manifest_digest"] = changed_digest
        runtime["services"][0]["image_manifest_descriptor"]["digest"] = changed_digest
        error = "services disagree on image manifest"
    runtime["runtime_identity_sha256"] = _canonical_sha256(
        {
            "compose_sha256": runtime["compose_sha256"],
            "compose_project": runtime["compose_project"],
            "services": runtime["services"],
            "images": runtime["images"],
        }
    )
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    runtime_ready_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        _read_runtime_ready_receipt(
            runtime_ready_path,
            expected_source_set_sha256=owner.prelaunch_manifest[
                "portable_source_set_sha256"
            ],
            expected_config_identity_sha256=owner.prelaunch_manifest[
                "config_artifacts"
            ]["identity_sha256"],
            expected_critical_environment_sha256=owner.prelaunch_manifest[
                "critical_environment_sha256"
            ],
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_runtime_receipt_v4_cannot_masquerade_as_label_bound_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    runtime_ready_path.chmod(0o644)
    receipt = json.loads(runtime_ready_path.read_text())
    receipt["schema_id"] = "alpagym.formal_run_runtime_ready.v4"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    runtime_ready_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_id is not supported"):
        _read_runtime_ready_receipt(
            runtime_ready_path,
            expected_source_set_sha256=owner.prelaunch_manifest[
                "portable_source_set_sha256"
            ],
            expected_config_identity_sha256=owner.prelaunch_manifest[
                "config_artifacts"
            ]["identity_sha256"],
            expected_critical_environment_sha256=owner.prelaunch_manifest[
                "critical_environment_sha256"
            ],
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_runtime_ready_rejects_compose_and_live_agreement_on_wrong_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compose/live agreement cannot override the canonical source destinations."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        alpasim_src_destination="/attacker/src",
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            alpasim_src_destination="/attacker/src",
        ),
    )

    with pytest.raises(RuntimeError, match="remounts canonical source"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_compose_and_live_agreement_on_read_write_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical source mounts remain read-only even when Compose and live agree."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        alpasim_src_read_only=False,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            alpasim_src_read_only=False,
        ),
    )

    with pytest.raises(RuntimeError, match="shadowing canonical destination"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_requires_runtime_cache_to_be_read_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutable native/JIT cache cannot be mounted read-only."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        runtime_cache_read_only=True,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            runtime_cache_read_only=True,
        ),
    )

    with pytest.raises(RuntimeError, match="shadowing canonical destination"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )


def test_runtime_ready_rejects_overlapping_scene_and_runtime_cache_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writable cache cannot alias any part of immutable scene provenance."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )

    with pytest.raises(RuntimeError, match="runtime cache must be disjoint"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "scene_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )


def test_runtime_ready_rejects_runtime_cache_overlapping_read_only_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writable cache cannot expose a child of any read-only bind."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    protected = tmp_path / "runtime_cache" / "controller-release"
    protected.mkdir(parents=True)
    extra_mounts = ((protected, "/mnt/controller-release", False),)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        extra_bind_mounts=extra_mounts,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            extra_bind_mounts=extra_mounts,
        ),
    )

    with pytest.raises(RuntimeError, match="remounts canonical source"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )


def test_runtime_ready_supports_non_visual_runtime_without_cache_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Formal non-visual humanoid runs do not require renderer caches."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        include_visual_caches=False,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            include_visual_caches=False,
        ),
    )

    receipt = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=None,
        runtime_cache_root=None,
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    assert receipt.is_file()


@pytest.mark.parametrize("cache_service", ("runtime-0", "humanoid_dynamics-0"))
def test_runtime_ready_supports_one_visual_cache_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_service: str,
) -> None:
    """Camera-only and controller-only profiles mount caches in one service."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    cache_services = (cache_service,)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        cache_services=cache_services,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            cache_services=cache_services,
        ),
    )

    receipt_path = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    services = {entry["service"]: entry for entry in receipt["runtimes"][0]["services"]}
    for service_name, service in services.items():
        destinations = {mount["destination"] for mount in service["mounts"]}
        assert ("/root/.cache" in destinations) is (service_name == cache_service)


def test_finalize_rejects_scene_cache_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coherent host-side cache replacement invalidates the formal run."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    (tmp_path / "scene_cache" / "replacement.bin").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="scene cache changed"):
        owner.finalize(run_completed=True)

    failure = json.loads(
        (owner.provenance_dir / "postrun.json").read_text(encoding="utf-8")
    )
    assert failure["run_completed"] is True


def test_runtime_ready_binds_visual_controller_release_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact Visual SONIC release used by dynamics remains auditable."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    controller_release = tmp_path / "controller_release"
    controller_release.mkdir()
    onnx_path = controller_release / "controller.onnx"
    onnx_path.write_bytes(b"locked-controller")
    controller_mount = ((controller_release, "/mnt/sonic-visual-release", False),)
    service_bind_mounts: dict[str, tuple[tuple[Path, str, bool], ...]] = {
        "humanoid_dynamics-0": controller_mount
    }
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        service_bind_mounts=service_bind_mounts,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            service_bind_mounts=service_bind_mounts,
        ),
    )

    receipt_path = owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        controller_release_root=controller_release,
        import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt["controller_release_identity"]
    assert identity["file_count"] == 1
    assert identity["files"][0]["relative_path"] == "controller.onnx"
    assert (
        identity["files"][0]["sha256"]
        == hashlib.sha256(b"locked-controller").hexdigest()
    )
    services = {
        service["service"]: service for service in receipt["runtimes"][0]["services"]
    }
    assert not any(
        mount["destination"] == "/mnt/sonic-visual-release"
        for mount in services["runtime-0"]["mounts"]
    )
    assert [
        mount
        for mount in services["humanoid_dynamics-0"]["mounts"]
        if mount["destination"] == "/mnt/sonic-visual-release"
    ] == [
        {
            "type": "bind",
            "source": str(controller_release),
            "destination": "/mnt/sonic-visual-release",
            "mode": "ro",
            "rw": False,
            "propagation": "rprivate",
        }
    ]

    onnx_path.write_bytes(b"different-controller")
    with pytest.raises(RuntimeError, match="controller release changed"):
        owner.finalize(run_completed=True)
    failure = json.loads(
        (owner.provenance_dir / "postrun.json").read_text(encoding="utf-8")
    )
    assert failure["formal_run_valid"] is False


def test_runtime_ready_rejects_unmounted_visual_controller_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host directory hash cannot attest a release dynamics did not mount."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    controller_release = tmp_path / "controller_release"
    controller_release.mkdir()
    (controller_release / "controller.onnx").write_bytes(b"locked-controller")
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )

    with pytest.raises(RuntimeError, match="controller release mapping"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            controller_release_root=controller_release,
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


@pytest.mark.parametrize(
    ("destination", "read_write", "error"),
    (
        ("/mnt/not-sonic-visual-release", False, "remounts canonical source"),
        ("/mnt/sonic-visual-release", True, "shadowing canonical destination"),
    ),
)
def test_runtime_ready_rejects_noncanonical_visual_controller_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination: str,
    read_write: bool,
    error: str,
) -> None:
    """Dynamics must consume the attested controller at its canonical RO mount."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    controller_release = tmp_path / "controller_release"
    controller_release.mkdir()
    (controller_release / "controller.onnx").write_bytes(b"locked-controller")
    service_bind_mounts: dict[str, tuple[tuple[Path, str, bool], ...]] = {
        "humanoid_dynamics-0": ((controller_release, destination, read_write),)
    }
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        service_bind_mounts=service_bind_mounts,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            service_bind_mounts=service_bind_mounts,
        ),
    )

    with pytest.raises(RuntimeError, match=error):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            controller_release_root=controller_release,
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_controller_release_overlapping_scene_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An immutable controller cannot be nested inside another mounted input."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    controller_release = tmp_path / "scene_store" / "controller_release"
    controller_release.mkdir(parents=True)
    (controller_release / "controller.onnx").write_bytes(b"locked-controller")
    service_bind_mounts: dict[str, tuple[tuple[Path, str, bool], ...]] = {
        "humanoid_dynamics-0": (
            (controller_release, "/mnt/sonic-visual-release", False),
        )
    }
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        service_bind_mounts=service_bind_mounts,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            service_bind_mounts=service_bind_mounts,
        ),
    )

    with pytest.raises(
        RuntimeError, match="controller release must be disjoint from SceneStore"
    ):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            controller_release_root=controller_release,
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_missing_humanoid_dynamics_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime-only Compose project is not a formal humanoid runtime."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    services = ("runtime-0",)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        services=services,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            services=services,
        ),
    )

    with pytest.raises(RuntimeError, match="no humanoid-dynamics service"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )

    assert not (owner.provenance_dir / "runtime_ready.json").exists()


def test_runtime_ready_rejects_read_write_shadow_under_canonical_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact RO root mount cannot be bypassed by a deeper RW bind shadow."""
    owner, alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    attacker_source = tmp_path / "attacker-source"
    attacker_source.mkdir()
    extra_bind_mounts = ((attacker_source, "/repo/src/alpasim_runtime", True),)
    wizard_log_dir = _write_compose_runtime(
        tmp_path,
        alpasim,
        humanoid,
        extra_bind_mounts=extra_bind_mounts,
    )
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
            extra_bind_mounts=extra_bind_mounts,
        ),
    )

    with pytest.raises(RuntimeError, match="shadowing canonical destination"):
        owner.capture_runtime_ready(
            wizard_log_dirs=(wizard_log_dir,),
            workload_command=["cosmos"],
            scene_store_root=tmp_path / "scene_store",
            scene_cache_root=tmp_path / "scene_cache",
            runtime_cache_root=tmp_path / "runtime_cache",
            import_probe=_test_import_probe(alpagym=alpagym, alpasim=alpasim),
        )


def test_finalize_scan_failure_writes_durable_invalid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postrun source-scan I/O failure leaves a machine-readable invalid receipt."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    humanoid.rename(tmp_path / "humanoid-unavailable")

    with pytest.raises(subprocess.CalledProcessError):
        owner.finalize(run_completed=True)

    _assert_durable_failure_receipt(owner.provenance_dir / "postrun.json")


def test_transient_source_edit_execute_restore_is_permanently_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inotify evidence closes the runtime-ready/postrun boundary-hash gap."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    source_path = humanoid / "tracked.py"
    original = source_path.read_bytes()
    source_path.write_bytes(b"attacker bytes executed\n")
    assert source_path.read_bytes() == b"attacker bytes executed\n"
    source_path.write_bytes(original)

    with pytest.raises(RuntimeError, match="source-watch"):
        owner.finalize(run_completed=True)

    postrun = json.loads((owner.provenance_dir / "postrun.json").read_text())
    assert postrun["sources_unchanged"] is True
    assert postrun["source_watch_clean"] is False
    assert postrun["formal_run_valid"] is False
    source_watch = json.loads(
        (owner.provenance_dir / "source_watch_postrun.json").read_text()
    )
    assert source_watch["relevant_event_count"] > 0
    assert any(
        event["path_annotation"] == str(source_path)
        for event in source_watch["relevant_events"]
    )


def test_external_hardlink_edit_execute_restore_is_permanently_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inode watch observes writes made through a repo-external hard link."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    source_path = humanoid / "tracked.py"
    external_link = tmp_path / "external-execution.py"
    original = source_path.read_bytes()
    os.link(source_path, external_link)
    external_link.write_bytes(b"executed = 'attacker bytes'\n")
    namespace: dict[str, str] = {}
    exec(source_path.read_text(encoding="utf-8"), namespace)
    assert namespace["executed"] == "attacker bytes"
    external_link.write_bytes(original)
    external_link.unlink()
    assert source_path.read_bytes() == original
    assert source_path.stat().st_nlink == 1

    with pytest.raises(RuntimeError, match="source-watch"):
        owner.finalize(run_completed=True)

    postrun = json.loads((owner.provenance_dir / "postrun.json").read_text())
    assert postrun["sources_unchanged"] is True
    assert postrun["source_watch_clean"] is False
    assert postrun["formal_run_valid"] is False
    source_watch = json.loads(
        (owner.provenance_dir / "source_watch_postrun.json").read_text()
    )
    assert any(
        event["path_annotation"] == str(source_path)
        for event in source_watch["relevant_events"]
    )


def test_three_repository_scale_survives_low_nofile_and_detects_external_hardlink(
    tmp_path: Path,
) -> None:
    """The real three-repo file scale needs constant FDs and keeps inode evidence."""
    source_counts = (280, 793, 396)
    repositories = tuple(
        _write_scaled_repo(tmp_path / name, source_count=source_count)
        for name, source_count in zip(
            ("alpagym", "alpasim", "humanoid"), source_counts, strict=True
        )
    )
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    external_link = tmp_path / "external-execution.py"
    script = textwrap.dedent(
        r"""
        import json
        import os
        import resource
        import sys
        from pathlib import Path

        from alpagym_host.formal_run_provenance import (
            RepositorySource,
            _EffectiveSourceWatch,
        )

        hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        soft_limit = 48
        if hard_limit != resource.RLIM_INFINITY:
            soft_limit = min(soft_limit, hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
        repository_paths = tuple(Path(value) for value in sys.argv[1:4])
        watch = _EffectiveSourceWatch.start(
            repositories=tuple(
                RepositorySource(name=name, root=path)
                for name, path in zip(
                    ("alpagym", "alpasim", "humanoid"),
                    repository_paths,
                    strict=True,
                )
            ),
            exact_config_paths=(Path(sys.argv[4]), Path(sys.argv[5])),
        )
        try:
            source_path = repository_paths[2] / "source_0000.py"
            external_link = Path(sys.argv[6])
            original = source_path.read_bytes()
            os.link(source_path, external_link)
            external_link.write_bytes(b"executed = 'low-nofile attack'\n")
            namespace = {}
            exec(source_path.read_text(encoding="utf-8"), namespace)
            external_link.write_bytes(original)
            external_link.unlink()
            checkpoint = watch.checkpoint(stage="low_nofile_attack")
            print(json.dumps({
                "soft_limit": soft_limit,
                "open_fd_count": len(tuple(Path("/proc/self/fd").iterdir())),
                "watched_regular_file_count": checkpoint[
                    "watched_regular_file_count"
                ],
                "event_count": checkpoint["relevant_event_count"],
                "clean": checkpoint["clean"],
                "executed": namespace["executed"],
            }, sort_keys=True))
        finally:
            watch.close()
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *(str(path) for path in repositories),
            str(resolved_config),
            str(cosmos_config),
            str(external_link),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    evidence = json.loads(result.stdout)

    assert evidence["soft_limit"] == 48
    assert evidence["open_fd_count"] < 16
    assert evidence["watched_regular_file_count"] == sum(source_counts) + 2
    assert evidence["executed"] == "low-nofile attack"
    assert evidence["event_count"] > 0
    assert evidence["clean"] is False


def test_prelaunch_rejects_repo_external_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing external link cannot cross the source-watch boundary."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    os.link(humanoid / "tracked.py", tmp_path / "external-source.py")

    with pytest.raises(ValueError, match="exactly one hard link"):
        FormalRunProvenance.capture_prelaunch(
            provenance_dir=tmp_path / "run" / "provenance",
            repositories=(
                RepositorySource(name="alpagym", root=alpagym),
                RepositorySource(
                    name="alpasim",
                    root=alpasim,
                    capture_ignored_generated_pb2=True,
                ),
                RepositorySource(name="humanoid", root=humanoid),
            ),
            resolved_config_path=resolved_config,
            cosmos_config_path=cosmos_config,
            critical_environment=_test_critical_environment(
                tmp_path=tmp_path,
                alpasim=alpasim,
                monkeypatch=monkeypatch,
            ),
        )


@pytest.mark.parametrize("config_name", ["resolved_config.yaml", "cosmos_config.toml"])
def test_prelaunch_rejects_config_external_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
) -> None:
    """Neither formal config artifact may be writable through another name."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    os.link(tmp_path / config_name, tmp_path / f"external-{config_name}")

    with pytest.raises(ValueError, match="exactly one hard link"):
        FormalRunProvenance.capture_prelaunch(
            provenance_dir=tmp_path / "run" / "provenance",
            repositories=(
                RepositorySource(name="alpagym", root=alpagym),
                RepositorySource(
                    name="alpasim",
                    root=alpasim,
                    capture_ignored_generated_pb2=True,
                ),
                RepositorySource(name="humanoid", root=humanoid),
            ),
            resolved_config_path=resolved_config,
            cosmos_config_path=cosmos_config,
            critical_environment=_test_critical_environment(
                tmp_path=tmp_path,
                alpasim=alpasim,
                monkeypatch=monkeypatch,
            ),
        )


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_prelaunch_rejects_git_index_flags_that_hide_tracked_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    """Index hints cannot suppress a pre-existing tracked source modification."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    _git(humanoid, "update-index", index_flag, "tracked.py")
    (humanoid / "tracked.py").write_text("hidden attacker bytes\n", encoding="utf-8")
    assert (
        "tracked.py"
        not in subprocess.run(
            ["git", "-C", str(humanoid), "diff", "--name-only", "HEAD", "--"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    with pytest.raises(ValueError, match="assume-unchanged or skip-worktree"):
        FormalRunProvenance.capture_prelaunch(
            provenance_dir=tmp_path / "run" / "provenance",
            repositories=(
                RepositorySource(name="alpagym", root=alpagym),
                RepositorySource(
                    name="alpasim",
                    root=alpasim,
                    capture_ignored_generated_pb2=True,
                ),
                RepositorySource(name="humanoid", root=humanoid),
            ),
            resolved_config_path=resolved_config,
            cosmos_config_path=cosmos_config,
            critical_environment=_test_critical_environment(
                tmp_path=tmp_path,
                alpasim=alpasim,
                monkeypatch=monkeypatch,
            ),
        )


def test_cleanup_failure_is_persisted_and_makes_completed_run_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed Cosmos process is not a valid formal run when cleanup failed."""
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    cleanup_error = FileNotFoundError("exact Compose file disappeared")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        owner.finalize(run_completed=True, cleanup_error=cleanup_error)

    postrun = json.loads((owner.provenance_dir / "postrun.json").read_text())
    assert postrun["run_completed"] is True
    assert postrun["cleanup_succeeded"] is False
    assert postrun["cleanup_failure"] == {
        "type": "FileNotFoundError",
        "message": "exact Compose file disappeared",
    }
    assert postrun["formal_run_valid"] is False


def test_preadmission_failure_still_downs_exact_compose_and_finalizes_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing runtime admission cannot suppress bounded project cleanup."""
    from types import SimpleNamespace

    from alpagym_host import run_lifecycle
    from alpagym_host.config import ExecutionBackend

    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    compose_path = wizard_log_dir / "docker-compose.yaml"
    compose_project = wizard_compose_project(wizard_log_dir)
    config = SimpleNamespace(artifact_paths=SimpleNamespace(alpasim_log_dir=tmp_path))
    monkeypatch.setattr(
        run_lifecycle, "ensure_process_terminated", lambda _process: None
    )
    real_subprocess_run = subprocess.run
    docker_commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] != "docker":
            return real_subprocess_run(command, **kwargs)
        docker_commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="")

    monkeypatch.setattr(run_lifecycle.subprocess, "run", run)

    with pytest.raises(
        RuntimeError, match="runtime-ready identity is unavailable"
    ) as caught:
        run_lifecycle._cleanup_wizard_processes(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            wizard_processes=[object()],
            provenance=owner,
        )

    assert docker_commands[0] == [
        "docker",
        "compose",
        "--project-name",
        compose_project,
        "--file",
        str(compose_path),
        "down",
    ]
    assert docker_commands[1] == [
        "docker",
        "ps",
        "--all",
        "--quiet",
        "--filter",
        f"label=com.docker.compose.project={compose_project}",
    ]

    with pytest.raises(RuntimeError, match="runtime receipt was not captured"):
        owner.finalize(run_completed=False, cleanup_error=caught.value)

    postrun = json.loads((owner.provenance_dir / "postrun.json").read_text())
    assert postrun["formal_run_valid"] is False
    assert postrun["cleanup_succeeded"] is False
    assert postrun["cleanup_failure"]["message"] == str(caught.value)


@pytest.mark.parametrize("compose_attack", ["missing", "bytes_drift"])
def test_compose_attack_uses_project_labels_for_forced_cleanup_and_invalidates_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compose_attack: str,
) -> None:
    """Compose path attacks cannot strand containers or produce a valid receipt."""
    from types import SimpleNamespace

    from alpagym_host import run_lifecycle
    from alpagym_host.config import ExecutionBackend

    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    runtime_ready_path = _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    assert runtime_ready_path.is_file()
    compose_path = tmp_path / "wizard_0" / "docker-compose.yaml"
    if compose_attack == "missing":
        compose_path.unlink()
    else:
        compose_path.write_text("services: {attacker: {}}\n", encoding="utf-8")
    config = SimpleNamespace(artifact_paths=SimpleNamespace(alpasim_log_dir=tmp_path))
    monkeypatch.setattr(
        run_lifecycle, "ensure_process_terminated", lambda _process: None
    )
    real_subprocess_run = subprocess.run
    docker_commands: list[list[str]] = []
    project_queries = 0

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal project_queries
        if command[0] != "docker":
            return real_subprocess_run(command, **kwargs)
        docker_commands.append(command)
        if command[:2] == ["docker", "compose"]:
            raise subprocess.CalledProcessError(1, command)
        if command[:2] == ["docker", "ps"]:
            project_queries += 1
            stdout = f"{'a' * 12}\n" if project_queries == 1 else ""
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=stdout
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="")
        raise AssertionError(f"unexpected Docker cleanup command: {command}")

    monkeypatch.setattr(run_lifecycle.subprocess, "run", run)

    with pytest.raises(BaseExceptionGroup) as caught:
        run_lifecycle._cleanup_wizard_processes(
            config=config,
            execution_backend=ExecutionBackend.local_process,
            wizard_processes=[object()],
            provenance=owner,
        )

    assert docker_commands[0][-1] == "down"
    assert ["docker", "rm", "--force", "a" * 12] in docker_commands
    assert project_queries == 2
    postrun_path = owner.finalize(
        run_completed=False,
        cleanup_error=caught.value,
    )
    postrun = json.loads(postrun_path.read_text())
    assert postrun["cleanup_succeeded"] is False
    assert postrun["formal_run_valid"] is False


def _prepare_formal_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FormalRunProvenance, Path, Path, Path]:
    """Create three dirty worktrees and their prelaunch provenance owner."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    resolved_config = tmp_path / "resolved_config.yaml"
    resolved_config.write_text("execution: {}\n", encoding="utf-8")
    cosmos_config = tmp_path / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    owner = FormalRunProvenance.capture_prelaunch(
        provenance_dir=tmp_path / "run" / "provenance",
        repositories=(
            RepositorySource(name="alpagym", root=alpagym),
            RepositorySource(
                name="alpasim",
                root=alpasim,
                capture_ignored_generated_pb2=True,
            ),
            RepositorySource(name="humanoid", root=humanoid),
        ),
        resolved_config_path=resolved_config,
        cosmos_config_path=cosmos_config,
        critical_environment=_test_critical_environment(
            tmp_path=tmp_path,
            alpasim=alpasim,
            monkeypatch=monkeypatch,
        ),
    )
    return owner, alpagym, alpasim, humanoid


def test_postrun_seals_complete_native_checkpoint_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    policy = (
        owner.provenance_dir.parent
        / "cosmos"
        / "20260823123456"
        / "checkpoints"
        / "step_2"
        / "policy"
    )
    policy.mkdir(parents=True)
    for name, data in {
        ".rank_0_complete": b"",
        "cosmos_config": b"{}",
        "model_rank_0.pth": b"model",
        "optimizer_rank_0.pth": b"optimizer",
        "scheduler_rank_0.pth": b"scheduler",
        "extra_info_rank_0.pth": b"extra",
    }.items():
        (policy / name).write_bytes(data)

    postrun_path = owner.finalize(run_completed=True)

    postrun = json.loads(postrun_path.read_text())
    assert postrun["schema_id"] == "alpagym.formal_run_postrun.v2"
    assert postrun["formal_run_valid"] is True
    assert len(postrun["native_checkpoints"]) == 1
    seal = postrun["native_checkpoints"][0]
    assert seal["step"] == 2
    assert seal["policy_relative_path"].endswith("checkpoints/step_2/policy")
    assert seal["tree_sha256"]
    assert seal["ranks"] == [0]


def test_incomplete_native_checkpoint_writes_invalid_postrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _alpagym, alpasim, humanoid = _prepare_formal_owner(tmp_path, monkeypatch)
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    policy = (
        owner.provenance_dir.parent
        / "cosmos"
        / "20260823123456"
        / "checkpoints"
        / "step_1"
        / "policy"
    )
    policy.mkdir(parents=True)
    (policy / "cosmos_config").write_text("{}", encoding="utf-8")
    (policy / ".rank_0_complete").write_bytes(b"")

    with pytest.raises(ValueError, match="rank files and completion markers"):
        owner.finalize(run_completed=True)

    _assert_durable_failure_receipt(owner.provenance_dir / "postrun.json")


def _test_consumed_rollout_records(
    *, count: int = 4, sample_rows: int = 14, actor_rows: int = 12
) -> list[dict[str, object]]:
    """Build one production-shaped ordered optimizer-artifact identity list."""

    return [
        {
            "rollout_index": index,
            "transport_kind": "disk_episode_v2",
            "completion_relative_path": f"artifacts/episode_{index}.json",
            "episode_file_sha256": f"{index + 1:064x}",
            "episode_file_size_bytes": 1024 + index,
            "episode_manifest_sha256": f"{index + 11:064x}",
            "tensor_sidecar": {
                "filename": (f"episode_{index}.{index + 21:032x}.tensors.pt"),
                "sha256": f"{index + 31:064x}",
                "size_bytes": 4096 + index,
            },
            "session_uuid": f"session-{index}",
            "rollout_seed": 1000 + index,
            "scene_id": "hq_stairs",
            "num_steps": sample_rows,
            "behavior_weight_version": 0,
            "optimizer_sample_rows": sample_rows,
            "optimizer_actor_valid_rows": actor_rows,
        }
        for index in range(count)
    ]


def _write_test_consumed_rollout_artifacts(
    *,
    run_dir: Path,
    records: list[dict[str, object]],
) -> None:
    """Materialize v2 episode/sidecar bytes and bind ``records`` to them."""

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for raw_record in records:
        raw_sidecar = raw_record["tensor_sidecar"]
        assert isinstance(raw_sidecar, dict)
        sidecar = cast(dict[str, object], raw_sidecar)
        sidecar_path = artifacts_dir / str(sidecar["filename"])
        sidecar_buffer = io.BytesIO()
        with zipfile.ZipFile(
            sidecar_buffer,
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            root = "archive"
            archive.writestr(f"{root}/data.pkl", b"\x80\x02}q\x00.")
            archive.writestr(f"{root}/byteorder", b"little")
            archive.writestr(f"{root}/data/0", b"test-storage")
            archive.writestr(f"{root}/version", b"3\n")
            archive.writestr(f"{root}/.data/serialization_id", b"test")
        sidecar_bytes = sidecar_buffer.getvalue()
        sidecar_path.write_bytes(sidecar_bytes)
        sidecar["sha256"] = hashlib.sha256(sidecar_bytes).hexdigest()
        sidecar["size_bytes"] = len(sidecar_bytes)

        num_steps = raw_record["num_steps"]
        actor_rows = raw_record["optimizer_actor_valid_rows"]
        behavior_version = raw_record["behavior_weight_version"]
        assert type(num_steps) is int
        assert type(actor_rows) is int
        assert type(behavior_version) is int
        manifest = {
            "scene_id": raw_record["scene_id"],
            "session_uuid": raw_record["session_uuid"],
            "num_steps": num_steps,
            "rollout_seed": raw_record["rollout_seed"],
            "policy_outputs": [
                {
                    "replay_data": {
                        "payload": {
                            "transition": {
                                "behavior_policy_version": behavior_version,
                                "actor_valid": index < actor_rows,
                            }
                        }
                    }
                }
                for index in range(num_steps)
            ],
        }
        manifest_sha256 = _canonical_sha256(manifest)
        raw_record["episode_manifest_sha256"] = manifest_sha256
        artifact = {
            "artifact_schema": "alpagym.disk_episode.v2",
            "episode_manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "tensor_sidecar": {
                "filename": sidecar["filename"],
                "format": "torch.save.weights_only.v1",
                "sha256": sidecar["sha256"],
                "size_bytes": sidecar["size_bytes"],
            },
        }
        episode_bytes = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        completion = PurePosixPath(str(raw_record["completion_relative_path"]))
        episode_path = run_dir.joinpath(*completion.parts)
        episode_path.write_bytes(episode_bytes)
        raw_record["episode_file_sha256"] = hashlib.sha256(episode_bytes).hexdigest()
        raw_record["episode_file_size_bytes"] = len(episode_bytes)


def _write_test_accepted_ppo_receipt(
    *,
    run_dir: Path,
    resolved_config: Path,
    records: list[dict[str, object]],
    step: int = 1,
) -> Path:
    """Write one immutable production-collector-shaped accepted v2 receipt."""

    cosmos_output = run_dir / "cosmos" / "20260823120001"
    cosmos_output.mkdir(parents=True, exist_ok=True)
    receipt_dir = run_dir / "artifacts" / "ppo_update_diagnostics"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = sum(int(record["optimizer_sample_rows"]) for record in records)
    actor_rows = sum(int(record["optimizer_actor_valid_rows"]) for record in records)
    receipt: dict[str, object] = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": "2026-08-23T12:00:02+00:00",
        "formal_run_id": run_dir.name,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        "resolved_config_size_bytes": resolved_config.stat().st_size,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": step,
        "total_steps": step,
        "state": "accepted",
        "received_rollouts": len(records),
        "trainable_rollouts": len(records),
        "sample_rows": sample_rows,
        "actor_sample_rows": actor_rows,
        "behavior_weight_versions": sorted(
            int(record["behavior_weight_version"]) for record in records
        ),
        "consumed_rollout_artifacts": records,
        "consumed_rollout_batch_sha256": _canonical_sha256(records),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": 1,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": {"train/optimizer_steps_applied": 1},
        "post_update_metrics": {"train/post_update_approx_kl": 0.001},
        "rejection": None,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    path = receipt_dir / f"step_{step}_rank_0.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path


def _write_test_on_policy_config(
    *,
    path: Path,
    batch_size: int,
    expected_rows: int,
) -> None:
    """Write the independent config authority used by the formal collector."""

    path.write_text(
        "dataset:\n"
        "  scene_ids: [hq_stairs]\n"
        f"expected_valid_steps: {expected_rows}\n"
        "cosmos:\n"
        "  mode: colocated\n"
        "  launch:\n"
        "    policy_replicas: 1\n"
        "  train:\n"
        f"    train_batch_per_replica: {batch_size}\n"
        "    train_policy:\n"
        "      on_policy: true\n"
        "alpasim:\n"
        "  humanoid:\n"
        "    rollout_seed_base: 1000\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("attack", "match"),
    (
        ("wrong_scene", "scene differs from resolved config"),
        ("wrong_seed", "seeds differ from resolved schedule"),
        ("claimed_num_steps", "episode ownership differs"),
        ("drop_artifact", "seeds differ from resolved schedule"),
    ),
)
def test_ppo_seal_rejects_fully_resigned_artifact_attack(
    tmp_path: Path,
    attack: str,
    match: str,
) -> None:
    """Reject resigned claims that contradict config or current artifact bytes."""

    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    expected_rows = 999 if attack == "claimed_num_steps" else 3
    records = _test_consumed_rollout_records(
        count=2,
        sample_rows=expected_rows,
        actor_rows=1 if attack == "claimed_num_steps" else 2,
    )
    if attack == "claimed_num_steps":
        for record in records:
            record["num_steps"] = 1
    if attack == "wrong_scene":
        for record in records:
            record["scene_id"] = "wrong_scene"
    elif attack == "wrong_seed":
        for index, record in enumerate(records):
            record["rollout_seed"] = 999_999 + index
    elif attack == "drop_artifact":
        records.pop()
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)

    if attack == "claimed_num_steps":
        # The current episode really contains one transition.  A resigned
        # receipt cannot relabel that as 999 real transitions; 999 packed rows
        # remain legal only when num_steps stays one (tested separately).
        for record in records:
            record["num_steps"] = 999
    resolved_config = run_dir / "resolved_config.yaml"
    _write_test_on_policy_config(
        path=resolved_config,
        batch_size=2,
        expected_rows=expected_rows,
    )
    _write_test_accepted_ppo_receipt(
        run_dir=run_dir,
        resolved_config=resolved_config,
        records=records,
    )

    with pytest.raises(ValueError, match=match):
        _capture_ppo_update_diagnostic_seals(
            run_dir,
            expected_resolved_config_sha256=hashlib.sha256(
                resolved_config.read_bytes()
            ).hexdigest(),
            expected_resolved_config_path=resolved_config,
            expected_resolved_config_size_bytes=resolved_config.stat().st_size,
        )


def test_ppo_seal_does_not_claim_host_sidecar_tensor_abi(tmp_path: Path) -> None:
    """Sidecar ABI belongs to the captured clean runtime, not this host seal."""

    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    records = _test_consumed_rollout_records(count=1, sample_rows=3, actor_rows=2)
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)
    record = records[0]
    sidecar = cast(dict[str, object], record["tensor_sidecar"])
    sidecar_bytes = b"host-only-seal-does-not-load-tensors"
    sidecar_path = run_dir / "artifacts" / str(sidecar["filename"])
    sidecar_path.write_bytes(sidecar_bytes)
    sidecar["sha256"] = hashlib.sha256(sidecar_bytes).hexdigest()
    sidecar["size_bytes"] = len(sidecar_bytes)
    episode_path = run_dir.joinpath(
        *PurePosixPath(str(record["completion_relative_path"])).parts
    )
    artifact = json.loads(episode_path.read_text(encoding="utf-8"))
    artifact["tensor_sidecar"]["sha256"] = sidecar["sha256"]
    artifact["tensor_sidecar"]["size_bytes"] = sidecar["size_bytes"]
    episode_bytes = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    episode_path.write_bytes(episode_bytes)
    record["episode_file_sha256"] = hashlib.sha256(episode_bytes).hexdigest()
    record["episode_file_size_bytes"] = len(episode_bytes)

    resolved_config = run_dir / "resolved_config.yaml"
    _write_test_on_policy_config(
        path=resolved_config,
        batch_size=1,
        expected_rows=3,
    )
    _write_test_accepted_ppo_receipt(
        run_dir=run_dir,
        resolved_config=resolved_config,
        records=records,
    )

    seals = _capture_ppo_update_diagnostic_seals(
        run_dir,
        expected_resolved_config_sha256=hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        expected_resolved_config_path=resolved_config,
        expected_resolved_config_size_bytes=resolved_config.stat().st_size,
    )

    assert len(seals) == 1


def test_ppo_seal_accepts_early_terminal_padding_without_inflating_real_rows(
    tmp_path: Path,
) -> None:
    """One real early-fall row may be padded to T_pack without becoming 999 rows."""

    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    records = _test_consumed_rollout_records(count=1, sample_rows=1, actor_rows=1)
    records[0]["num_steps"] = 1
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)
    resolved_config = run_dir / "resolved_config.yaml"
    _write_test_on_policy_config(
        path=resolved_config,
        batch_size=1,
        expected_rows=999,
    )
    _write_test_accepted_ppo_receipt(
        run_dir=run_dir,
        resolved_config=resolved_config,
        records=records,
    )

    seals = _capture_ppo_update_diagnostic_seals(
        run_dir,
        expected_resolved_config_sha256=hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        expected_resolved_config_path=resolved_config,
        expected_resolved_config_size_bytes=resolved_config.stat().st_size,
    )

    assert len(seals) == 1


def test_ppo_seal_rejects_uncompacted_padding_in_colocated_single_policy(
    tmp_path: Path,
) -> None:
    """The single-GPU optimizer receipt must name only its real input rows."""

    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    records = _test_consumed_rollout_records(count=1, sample_rows=60, actor_rows=1)
    records[0]["num_steps"] = 1
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)
    resolved_config = run_dir / "resolved_config.yaml"
    _write_test_on_policy_config(
        path=resolved_config,
        batch_size=1,
        expected_rows=60,
    )
    _write_test_accepted_ppo_receipt(
        run_dir=run_dir,
        resolved_config=resolved_config,
        records=records,
    )

    with pytest.raises(ValueError, match="compact padding"):
        _capture_ppo_update_diagnostic_seals(
            run_dir,
            expected_resolved_config_sha256=hashlib.sha256(
                resolved_config.read_bytes()
            ).hexdigest(),
            expected_resolved_config_path=resolved_config,
            expected_resolved_config_size_bytes=resolved_config.stat().st_size,
        )


def test_ppo_seal_requires_explicit_on_policy_config(tmp_path: Path) -> None:
    """A v2 receipt cannot opt itself into the stricter config-derived checks."""

    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    records = _test_consumed_rollout_records(count=1, sample_rows=3, actor_rows=2)
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(
        "dataset:\n"
        "  scene_ids: [hq_stairs]\n"
        "expected_valid_steps: 3\n"
        "alpasim:\n"
        "  humanoid:\n"
        "    rollout_seed_base: 1000\n",
        encoding="utf-8",
    )
    _write_test_accepted_ppo_receipt(
        run_dir=run_dir,
        resolved_config=resolved_config,
        records=records,
    )

    with pytest.raises(ValueError, match="requires explicit on-policy"):
        _capture_ppo_update_diagnostic_seals(
            run_dir,
            expected_resolved_config_sha256=hashlib.sha256(
                resolved_config.read_bytes()
            ).hexdigest(),
            expected_resolved_config_path=resolved_config,
            expected_resolved_config_size_bytes=resolved_config.stat().st_size,
        )


@pytest.mark.parametrize(
    "post_update_metrics",
    (
        {"train/post_update_approx_kl": 0.0051},
        None,
    ),
    ids=("restored-metrics", "diagnostic-failed-null-metrics"),
)
def test_failed_formal_run_seals_post_rejected_ppo_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_update_metrics: dict[str, float] | None,
) -> None:
    """A post-rejected run retains its immutable update evidence in postrun."""
    alpagym = _write_dirty_repo(tmp_path / "alpagym")
    alpasim = _write_dirty_repo(tmp_path / "alpasim", with_ignored_pb2=True)
    humanoid = _write_dirty_repo(tmp_path / "humanoid")
    run_dir = tmp_path / "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir.mkdir()
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(
        "dataset:\n"
        "  scene_ids: [hq_stairs]\n"
        "expected_valid_steps: 14\n"
        "cosmos:\n"
        "  launch:\n"
        "    policy_replicas: 1\n"
        "  train:\n"
        "    train_batch_per_replica: 4\n"
        "    train_policy:\n"
        "      on_policy: true\n"
        "alpasim:\n"
        "  humanoid:\n"
        "    rollout_seed_base: 1000\n",
        encoding="utf-8",
    )
    cosmos_config = run_dir / "cosmos_config.toml"
    cosmos_config.write_text("mode = 'colocated'\n", encoding="utf-8")
    owner = FormalRunProvenance.capture_prelaunch(
        provenance_dir=run_dir / "provenance",
        repositories=(
            RepositorySource(name="alpagym", root=alpagym),
            RepositorySource(
                name="alpasim",
                root=alpasim,
                capture_ignored_generated_pb2=True,
            ),
            RepositorySource(name="humanoid", root=humanoid),
        ),
        resolved_config_path=resolved_config,
        cosmos_config_path=cosmos_config,
        critical_environment=_test_critical_environment(
            tmp_path=tmp_path,
            alpasim=alpasim,
            monkeypatch=monkeypatch,
        ),
    )
    _capture_valid_runtime(
        owner=owner,
        tmp_path=tmp_path,
        alpasim=alpasim,
        humanoid=humanoid,
        monkeypatch=monkeypatch,
    )
    cosmos_output = run_dir / "cosmos" / "20260823120001"
    cosmos_output.mkdir(parents=True)
    receipt_dir = run_dir / "artifacts" / "ppo_update_diagnostics"
    receipt_dir.mkdir(parents=True)
    consumed_rollouts = _test_consumed_rollout_records()
    _write_test_consumed_rollout_artifacts(
        run_dir=run_dir,
        records=consumed_rollouts,
    )
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": "2026-08-23T12:00:02+00:00",
        "formal_run_id": run_dir.name,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        "resolved_config_size_bytes": resolved_config.stat().st_size,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": 1,
        "total_steps": 4,
        "state": "post_rejected",
        "received_rollouts": 4,
        "trainable_rollouts": 4,
        "sample_rows": 56,
        "actor_sample_rows": 48,
        "behavior_weight_versions": [0, 0, 0, 0],
        "consumed_rollout_artifacts": consumed_rollouts,
        "consumed_rollout_batch_sha256": _canonical_sha256(consumed_rollouts),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": 1,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": {
            "train/optimizer_steps_applied": 1,
            "train/loss_avg_local": 0.25,
        },
        "post_update_metrics": post_update_metrics,
        "rejection": {
            "type": "FloatingPointError",
            "message": "post-update behavior KL exceeds calibrated target",
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = receipt_dir / "step_1_rank_0.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    postrun_path = owner.finalize(run_completed=False)

    postrun = json.loads(postrun_path.read_text())
    assert postrun["formal_run_valid"] is False
    assert postrun["native_checkpoints"] == []
    assert postrun["ppo_update_diagnostic_receipts"] == [
        {
            "receipt_relative_path": (
                "artifacts/ppo_update_diagnostics/step_1_rank_0.json"
            ),
            "state": "post_rejected",
            "current_step": 1,
            "rank": 0,
            "receipt_sha256": receipt["receipt_sha256"],
            "file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "size_bytes": receipt_path.stat().st_size,
        }
    ]


@pytest.mark.parametrize(
    (
        "state",
        "optimizer_metrics",
        "boundary_optimizer_steps",
        "post_metrics",
        "rejection",
        "match",
    ),
    (
        (
            "pre_rejected",
            {"train/optimizer_steps_applied": 0},
            0,
            None,
            {"type": "RuntimeError", "message": "pre guard failed"},
            "pre-rejected PPO diagnostic contains optimizer state",
        ),
        (
            "accepted",
            {"train/optimizer_steps_applied": 1},
            1,
            None,
            None,
            "accepted PPO diagnostic requires optimizer and post-update metrics",
        ),
        (
            "post_rejected",
            None,
            0,
            None,
            {"type": "RuntimeError", "message": "post replay failed"},
            "post-rejected PPO diagnostic requires optimizer metrics",
        ),
        (
            "post_rejected",
            {"train/optimizer_steps_applied": 1},
            0,
            None,
            {"type": "RuntimeError", "message": "post replay failed"},
            "post-rejected PPO diagnostic optimizer boundary differs from metrics",
        ),
        (
            "accepted",
            {"train/optimizer_steps_applied": 1},
            1,
            {"train/post_update_approx_kl": 0.001},
            {"type": "RuntimeError", "message": "impossible accepted rejection"},
            "accepted PPO diagnostic cannot contain a rejection",
        ),
        (
            "post_rejected",
            {"train/optimizer_steps_applied": 1},
            1,
            None,
            None,
            "rejected PPO diagnostic has no valid exception identity",
        ),
    ),
)
def test_ppo_update_diagnostic_rejects_inconsistent_terminal_state(
    state: str,
    optimizer_metrics: dict[str, int] | None,
    boundary_optimizer_steps: int,
    post_metrics: dict[str, float] | None,
    rejection: dict[str, str] | None,
    match: str,
) -> None:
    """Host sealing rejects metric/rejection combinations the writer cannot emit."""
    formal_run_id = "20260823T120000Z-0123456789abcdef0123456789abcdef"
    resolved_config_sha256 = "1" * 64
    consumed_rollouts = _test_consumed_rollout_records()
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": "2026-08-23T12:00:02+00:00",
        "formal_run_id": formal_run_id,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": resolved_config_sha256,
        "resolved_config_size_bytes": 14,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": 1,
        "total_steps": 4,
        "state": state,
        "received_rollouts": 4,
        "trainable_rollouts": 4,
        "sample_rows": 56,
        "actor_sample_rows": 48,
        "behavior_weight_versions": [0, 0, 0, 0],
        "consumed_rollout_artifacts": consumed_rollouts,
        "consumed_rollout_batch_sha256": _canonical_sha256(consumed_rollouts),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": boundary_optimizer_steps,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": optimizer_metrics,
        "post_update_metrics": post_metrics,
        "rejection": rejection,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)

    with pytest.raises((TypeError, ValueError), match=match):
        _validate_ppo_update_diagnostic_receipt(
            receipt,
            expected_formal_run_id=formal_run_id,
            expected_resolved_config_sha256=resolved_config_sha256,
            expected_resolved_config_relative_path="resolved_config.yaml",
            expected_resolved_config_size_bytes=14,
            expected_step=1,
            expected_rank=0,
        )


@pytest.mark.parametrize(
    "captured_at_utc",
    (
        "not-a-timestamp",
        "2026-08-23T12:00:02",
        "2026-08-23T12:00:02-07:00",
        "2026-08-23T11:59:59+00:00",
    ),
)
def test_ppo_update_diagnostic_rejects_invalid_capture_time(
    captured_at_utc: str,
) -> None:
    """The formal root anchors a canonical UTC receipt time floor."""

    formal_run_id = "20260823T120000Z-0123456789abcdef0123456789abcdef"
    records = _test_consumed_rollout_records(count=1)
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": captured_at_utc,
        "formal_run_id": formal_run_id,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": "1" * 64,
        "resolved_config_size_bytes": 14,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": 1,
        "total_steps": 1,
        "state": "accepted",
        "received_rollouts": 1,
        "trainable_rollouts": 1,
        "sample_rows": 14,
        "actor_sample_rows": 12,
        "behavior_weight_versions": [0],
        "consumed_rollout_artifacts": records,
        "consumed_rollout_batch_sha256": _canonical_sha256(records),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": 1,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": {"train/optimizer_steps_applied": 1},
        "post_update_metrics": {"train/post_update_approx_kl": 0.001},
        "rejection": None,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)

    with pytest.raises(ValueError, match="captured_at_utc"):
        _validate_ppo_update_diagnostic_receipt(
            receipt,
            expected_formal_run_id=formal_run_id,
            expected_resolved_config_sha256="1" * 64,
            expected_resolved_config_relative_path="resolved_config.yaml",
            expected_resolved_config_size_bytes=14,
            expected_step=1,
            expected_rank=0,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("reorder", "artifact order is invalid"),
        ("tamper", "batch SHA-256 differs"),
        ("duplicate_session", "session ownership is invalid"),
        ("duplicate_seed", "seed ownership is invalid"),
        ("unsafe_path", "completion path is unsafe"),
        ("sample_count", "sample rows do not conserve"),
        ("actor_count", "actor rows do not conserve"),
        ("behavior_version", "behavior versions differ"),
        ("episode_sha", "episode_file_sha256 is invalid"),
        ("episode_size", "episode file size is invalid"),
        ("manifest_sha", "episode_manifest_sha256 is invalid"),
        ("sidecar_sha", "sidecar SHA-256 is invalid"),
        ("sidecar_size", "sidecar size is invalid"),
        ("record_count", "artifact count differs"),
    ),
)
def test_ppo_update_diagnostic_rejects_forged_consumed_rollout_batch(
    mutation: str,
    match: str,
) -> None:
    """Reject internally inconsistent, un-reindexed order/hash/ownership edits."""

    formal_run_id = "20260823T120000Z-0123456789abcdef0123456789abcdef"
    resolved_config_sha256 = "1" * 64
    consumed_rollouts = _test_consumed_rollout_records(count=2)
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": "2026-08-23T12:00:02+00:00",
        "formal_run_id": formal_run_id,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": resolved_config_sha256,
        "resolved_config_size_bytes": 14,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": 1,
        "total_steps": 4,
        "state": "accepted",
        "received_rollouts": 2,
        "trainable_rollouts": 2,
        "sample_rows": 28,
        "actor_sample_rows": 24,
        "behavior_weight_versions": [0, 0],
        "consumed_rollout_artifacts": consumed_rollouts,
        "consumed_rollout_batch_sha256": _canonical_sha256(consumed_rollouts),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": 1,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": {"train/optimizer_steps_applied": 1},
        "post_update_metrics": {"train/post_update_approx_kl": 0.001},
        "rejection": None,
    }
    records = receipt["consumed_rollout_artifacts"]
    assert isinstance(records, list)
    if mutation == "reorder":
        records[:] = [records[1], records[0]]
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "tamper":
        records[0]["episode_file_sha256"] = "f" * 64
    elif mutation == "duplicate_session":
        records[1]["session_uuid"] = records[0]["session_uuid"]
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "duplicate_seed":
        records[1]["rollout_seed"] = records[0]["rollout_seed"]
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "unsafe_path":
        records[0]["completion_relative_path"] = "artifacts/../episode_0.json"
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "sample_count":
        records[0]["optimizer_sample_rows"] = 13
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "actor_count":
        records[0]["optimizer_actor_valid_rows"] = 11
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "behavior_version":
        records[0]["behavior_weight_version"] = 1
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "episode_sha":
        records[0]["episode_file_sha256"] = "F" * 64
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "episode_size":
        records[0]["episode_file_size_bytes"] = True
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "manifest_sha":
        records[0]["episode_manifest_sha256"] = "bad"
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "sidecar_sha":
        sidecar = records[0]["tensor_sidecar"]
        assert isinstance(sidecar, dict)
        cast(dict[str, object], sidecar)["sha256"] = "0" * 63
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "sidecar_size":
        sidecar = records[0]["tensor_sidecar"]
        assert isinstance(sidecar, dict)
        cast(dict[str, object], sidecar)["size_bytes"] = 0
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    elif mutation == "record_count":
        records.pop()
        receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)
    receipt["receipt_sha256"] = _canonical_sha256(receipt)

    with pytest.raises((TypeError, ValueError), match=match):
        _validate_ppo_update_diagnostic_receipt(
            receipt,
            expected_formal_run_id=formal_run_id,
            expected_resolved_config_sha256=resolved_config_sha256,
            expected_resolved_config_relative_path="resolved_config.yaml",
            expected_resolved_config_size_bytes=14,
            expected_step=1,
            expected_rank=0,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "session_uuid",
        "rollout_seed",
        "scene_id",
        "num_steps",
        "behavior_weight_version",
        "optimizer_sample_rows",
        "optimizer_actor_valid_rows",
    ),
)
def test_ppo_update_diagnostic_reauthenticates_episode_semantics(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A fully resigned receipt cannot contradict the sealed episode bytes."""

    formal_run_id = "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir = tmp_path / formal_run_id
    run_dir.mkdir()
    records = _test_consumed_rollout_records(
        count=1,
        sample_rows=3,
        actor_rows=2,
    )
    _write_test_consumed_rollout_artifacts(run_dir=run_dir, records=records)
    receipt = {
        "schema_id": "alpagym.ppo_update_diagnostic.v2",
        "captured_at_utc": "2026-08-23T12:00:02+00:00",
        "formal_run_id": formal_run_id,
        "resolved_config_relative_path": "resolved_config.yaml",
        "resolved_config_sha256": "1" * 64,
        "resolved_config_size_bytes": 14,
        "cosmos_output_relative_path": "cosmos/20260823120001",
        "rank": 0,
        "current_step": 1,
        "total_steps": 4,
        "state": "accepted",
        "received_rollouts": 1,
        "trainable_rollouts": 1,
        "sample_rows": 3,
        "actor_sample_rows": 2,
        "behavior_weight_versions": [0],
        "consumed_rollout_artifacts": records,
        "consumed_rollout_batch_sha256": _canonical_sha256(records),
        "is_master_replica": True,
        "checkpoint_requested": True,
        "boundary": {
            "optimizer_steps_applied": 1,
            "scheduler_advanced": False,
            "checkpoint_started": False,
            "weight_sync_started": False,
        },
        "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
        "optimizer_metrics": {"train/optimizer_steps_applied": 1},
        "post_update_metrics": {"train/post_update_approx_kl": 0.001},
        "rejection": None,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _validate_ppo_update_diagnostic_receipt(
        receipt,
        expected_formal_run_root=run_dir,
        expected_formal_run_id=formal_run_id,
        expected_resolved_config_sha256="1" * 64,
        expected_resolved_config_relative_path="resolved_config.yaml",
        expected_resolved_config_size_bytes=14,
        expected_step=1,
        expected_rank=0,
    )

    record = records[0]
    if mutation == "session_uuid":
        record[mutation] = "forged-session"
    elif mutation == "rollout_seed":
        record[mutation] = 999999
    elif mutation == "scene_id":
        record[mutation] = "forged-scene"
    elif mutation == "num_steps":
        record[mutation] = 4
    elif mutation == "behavior_weight_version":
        record[mutation] = 9
        receipt["behavior_weight_versions"] = [9]
    elif mutation == "optimizer_sample_rows":
        record[mutation] = 2
        receipt["sample_rows"] = 2
    elif mutation == "optimizer_actor_valid_rows":
        record[mutation] = 1
        receipt["actor_sample_rows"] = 1
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(mutation)
    receipt["consumed_rollout_batch_sha256"] = _canonical_sha256(records)
    receipt["receipt_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValueError, match="differs"):
        _validate_ppo_update_diagnostic_receipt(
            receipt,
            expected_formal_run_root=run_dir,
            expected_formal_run_id=formal_run_id,
            expected_resolved_config_sha256="1" * 64,
            expected_resolved_config_relative_path="resolved_config.yaml",
            expected_resolved_config_size_bytes=14,
            expected_step=1,
            expected_rank=0,
        )


def test_ppo_seal_rejects_on_policy_rollout_reuse_across_steps(
    tmp_path: Path,
) -> None:
    """A later update cannot relabel a prior session/seed as fresh data."""

    formal_run_id = "20260823T120000Z-0123456789abcdef0123456789abcdef"
    run_dir = tmp_path / formal_run_id
    run_dir.mkdir()
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(
        "dataset:\n"
        "  scene_ids: [hq_stairs]\n"
        "expected_valid_steps: 3\n"
        "cosmos:\n"
        "  launch:\n"
        "    policy_replicas: 1\n"
        "  train:\n"
        "    train_batch_per_replica: 1\n"
        "    train_policy:\n"
        "      on_policy: true\n"
        "alpasim:\n"
        "  humanoid:\n"
        "    rollout_seed_base: 1000\n",
        encoding="utf-8",
    )
    (run_dir / "cosmos" / "20260823120001").mkdir(parents=True)
    receipt_dir = run_dir / "artifacts" / "ppo_update_diagnostics"
    receipt_dir.mkdir(parents=True)

    step_one_records = _test_consumed_rollout_records(
        count=1,
        sample_rows=3,
        actor_rows=2,
    )
    _write_test_consumed_rollout_artifacts(
        run_dir=run_dir,
        records=step_one_records,
    )
    step_two_records = _test_consumed_rollout_records(
        count=1,
        sample_rows=3,
        actor_rows=2,
    )
    step_two_record = step_two_records[0]
    step_two_record["completion_relative_path"] = "artifacts/episode_second.json"
    step_two_sidecar = step_two_record["tensor_sidecar"]
    assert isinstance(step_two_sidecar, dict)
    cast(dict[str, object], step_two_sidecar)["filename"] = (
        "episode_second.0123456789abcdef0123456789abcdef.tensors.pt"
    )
    step_two_record["rollout_seed"] = 1001
    step_two_record["behavior_weight_version"] = 1
    _write_test_consumed_rollout_artifacts(
        run_dir=run_dir,
        records=step_two_records,
    )

    def accepted_receipt(
        step: int,
        records: list[dict[str, object]],
    ) -> dict[str, object]:
        sample_rows = sum(int(record["optimizer_sample_rows"]) for record in records)
        actor_rows = sum(
            int(record["optimizer_actor_valid_rows"]) for record in records
        )
        receipt: dict[str, object] = {
            "schema_id": "alpagym.ppo_update_diagnostic.v2",
            "captured_at_utc": "2026-08-23T12:00:02+00:00",
            "formal_run_id": formal_run_id,
            "resolved_config_relative_path": "resolved_config.yaml",
            "resolved_config_sha256": hashlib.sha256(
                resolved_config.read_bytes()
            ).hexdigest(),
            "resolved_config_size_bytes": resolved_config.stat().st_size,
            "cosmos_output_relative_path": "cosmos/20260823120001",
            "rank": 0,
            "current_step": step,
            "total_steps": 2,
            "state": "accepted",
            "received_rollouts": len(records),
            "trainable_rollouts": len(records),
            "sample_rows": sample_rows,
            "actor_sample_rows": actor_rows,
            "behavior_weight_versions": [step - 1] * len(records),
            "consumed_rollout_artifacts": records,
            "consumed_rollout_batch_sha256": _canonical_sha256(records),
            "is_master_replica": True,
            "checkpoint_requested": True,
            "boundary": {
                "optimizer_steps_applied": 1,
                "scheduler_advanced": False,
                "checkpoint_started": False,
                "weight_sync_started": False,
            },
            "pre_update_metrics": {"train/pre_update_approx_kl": 0.0},
            "optimizer_metrics": {"train/optimizer_steps_applied": 1},
            "post_update_metrics": {"train/post_update_approx_kl": 0.001},
            "rejection": None,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return receipt

    for step, records in ((1, step_one_records), (2, step_two_records)):
        path = receipt_dir / f"step_{step}_rank_0.json"
        path.write_text(
            json.dumps(accepted_receipt(step, records), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)

    with pytest.raises(ValueError, match="reused rollout ownership"):
        _capture_ppo_update_diagnostic_seals(
            run_dir,
            expected_resolved_config_sha256=hashlib.sha256(
                resolved_config.read_bytes()
            ).hexdigest(),
            expected_resolved_config_path=resolved_config,
            expected_resolved_config_size_bytes=resolved_config.stat().st_size,
        )


def _capture_valid_runtime(
    *,
    owner: FormalRunProvenance,
    tmp_path: Path,
    alpasim: Path,
    humanoid: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Capture one canonical two-service runtime for a provenance test."""
    wizard_log_dir = _write_compose_runtime(tmp_path, alpasim, humanoid)
    monkeypatch.setattr(
        "alpagym_host.formal_run_provenance._run_command",
        _mock_docker_command(
            alpasim=alpasim,
            humanoid=humanoid,
            compose_path=wizard_log_dir / "docker-compose.yaml",
        ),
    )
    return owner.capture_runtime_ready(
        wizard_log_dirs=(wizard_log_dir,),
        workload_command=["uv", "run", "cosmos"],
        scene_store_root=tmp_path / "scene_store",
        scene_cache_root=tmp_path / "scene_cache",
        runtime_cache_root=tmp_path / "runtime_cache",
        import_probe=_test_import_probe(
            alpagym=next(
                repository.root
                for repository in owner.repositories
                if repository.name == "alpagym"
            ),
            alpasim=alpasim,
        ),
    )


def _assert_durable_failure_receipt(path: Path) -> None:
    """Assert the minimal postrun failure receipt is present and self-consistent."""
    receipt = json.loads(path.read_text())
    assert receipt["schema_id"] == "alpagym.formal_run_postrun_failure.v1"
    assert receipt["formal_run_valid"] is False
    stored_sha256 = receipt.pop("receipt_sha256")
    assert stored_sha256 == _canonical_sha256(receipt)


def _test_critical_environment(
    *, tmp_path: Path, alpasim: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """Install the exact critical subprocess environment used by formal tests."""
    grpc_root = (alpasim / "src" / "grpc").resolve()
    pycache_prefix = tmp_path / "formal-test-pycache"
    pycache_prefix.mkdir(exist_ok=True)
    environment = {
        "ALPASIM_GRPC_ROOT": str(grpc_root),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": str(grpc_root),
        "PYTHONPYCACHEPREFIX": str(pycache_prefix.resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return environment


def _test_import_probe(*, alpagym: Path, alpasim: Path) -> dict[str, object]:
    """Return a valid source-origin and protobuf-tag probe for temporary repos."""
    environment = {
        name: os.environ[name]
        for name in (
            "ALPASIM_GRPC_ROOT",
            "CUBLAS_WORKSPACE_CONFIG",
            "PYTHONPATH",
            "PYTHONPYCACHEPREFIX",
            "PYTHONDONTWRITEBYTECODE",
        )
    }
    return build_import_probe_receipt(
        command=["uv", "run", "--no-sync", "python", "-c", "probe"],
        environment=environment,
        payload={
            "module_origins": {
                "alpagym_host": str(alpagym / "tracked.py"),
                "alpagym_runtime": str(alpagym / "tracked.py"),
                "alpagym_g1_vla": str(alpagym / "tracked.py"),
                "alpasim_grpc.v0.humanoid_pb2": str(
                    alpasim / "src" / "grpc" / "alpasim_grpc" / "v0" / "humanoid_pb2.py"
                ),
                "alpasim_grpc.v0.humanoid_pb2_grpc": str(
                    alpasim
                    / "src"
                    / "grpc"
                    / "alpasim_grpc"
                    / "v0"
                    / "humanoid_pb2_grpc.py"
                ),
            },
            "descriptor_fields": {
                "HumanoidSessionAbortRequest": {"session_uuid": 1},
                "HumanoidCameraImage": {
                    "scene_fingerprint": 13,
                    "model_signature_sha256": 14,
                    "camera_to_world_sha256": 15,
                    "renderer_binding_sha256": 16,
                },
            },
            "descriptor_services": {
                "HumanoidPolicyService": {
                    "abort_session": "humanoid.HumanoidSessionAbortRequest"
                },
                "HumanoidDynamicsService": {
                    "abort_session": "humanoid.HumanoidSessionAbortRequest"
                },
            },
            "grpc_bindings": {
                "policy_stub_abort_session": True,
                "dynamics_stub_abort_session": True,
                "policy_servicer_abort_session": True,
                "dynamics_servicer_abort_session": True,
            },
        },
    )


def _write_dirty_repo(root: Path, *, with_ignored_pb2: bool = False) -> Path:
    """Create a committed Git root plus tracked and untracked dirty source."""
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "formal@example.com")
    _git(root, "config", "user.name", "Formal Test")
    (root / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    if with_ignored_pb2:
        (root / "src" / "grpc").mkdir(parents=True)
        (root / "src" / "grpc" / ".gitignore").write_text(
            "*_pb2.py\n", encoding="utf-8"
        )
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "base")
    (root / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (root / "notes").mkdir()
    (root / "notes" / "new.txt").write_text("untracked\n", encoding="utf-8")
    (root / "src").mkdir(exist_ok=True)
    (root / "plugins").mkdir()
    if with_ignored_pb2:
        generated = root / "src" / "grpc" / "alpasim_grpc" / "v0"
        generated.mkdir(parents=True)
        (generated / "humanoid_pb2.py").write_text("generated ABI\n", encoding="utf-8")
        (generated / "humanoid_pb2_grpc.py").write_text(
            "generated gRPC ABI\n", encoding="utf-8"
        )
        environment_copy = root / ".venv" / "lib"
        environment_copy.mkdir(parents=True)
        (environment_copy / "dependency_pb2.py").write_text(
            "not executed from source\n", encoding="utf-8"
        )
    return root.resolve()


def _write_scaled_repo(root: Path, *, source_count: int) -> Path:
    """Create one committed flat source tree with a production-scale file count."""
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "formal@example.com")
    _git(root, "config", "user.name", "Formal Test")
    for index in range(source_count):
        (root / f"source_{index:04d}.py").write_text(
            f"value = {index}\n", encoding="utf-8"
        )
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "scale")
    return root.resolve()


def _write_compose_runtime(
    tmp_path: Path,
    alpasim: Path,
    humanoid: Path,
    *,
    services: tuple[str, ...] = ("runtime-0", "humanoid_dynamics-0"),
    alpasim_src_destination: str = "/repo/src",
    alpasim_src_read_only: bool = True,
    include_visual_caches: bool = True,
    cache_services: tuple[str, ...] | None = None,
    runtime_cache_read_only: bool = False,
    extra_bind_mounts: tuple[tuple[Path, str, bool], ...] = (),
    service_bind_mounts: dict[str, tuple[tuple[Path, str, bool], ...]] | None = None,
) -> Path:
    """Write the generated Compose shape consumed by the receipt."""
    wizard_log_dir = tmp_path / "wizard_0"
    wizard_log_dir.mkdir()
    scene_store = tmp_path / "scene_store"
    scene_store.mkdir(exist_ok=True)
    scene_cache = tmp_path / "scene_cache"
    scene_cache.mkdir(exist_ok=True)
    runtime_cache = tmp_path / "runtime_cache"
    runtime_cache.mkdir(exist_ok=True)
    alpasim_src_mode = "ro" if alpasim_src_read_only else "rw"
    shared_volumes = [
        f"{alpasim / 'src'}:{alpasim_src_destination}:{alpasim_src_mode}",
        f"{alpasim / 'plugins'}:/repo/plugins:ro",
        f"{humanoid}:/repo/humanoid-rl-joint-sim:ro",
        f"{scene_store}:/mnt/humanoid-scene-store:ro",
        *[
            f"{source}:{destination}:{'rw' if read_write else 'ro'}"
            for source, destination, read_write in extra_bind_mounts
        ],
    ]
    visual_cache_volumes = [
        f"{scene_cache}:/mnt/humanoid-scene-cache:ro",
        f"{runtime_cache}:/root/.cache:{'ro' if runtime_cache_read_only else 'rw'}",
    ]
    resolved_cache_services = set(
        services if cache_services is None else cache_services
    )
    resolved_service_bind_mounts = service_bind_mounts or {}
    (wizard_log_dir / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    service: {
                        "image": "runtime:local",
                        "volumes": [
                            *shared_volumes,
                            *[
                                f"{source}:{destination}:{'rw' if read_write else 'ro'}"
                                for source, destination, read_write in (
                                    resolved_service_bind_mounts.get(service, ())
                                )
                            ],
                            *(
                                visual_cache_volumes
                                if include_visual_caches
                                and service in resolved_cache_services
                                else []
                            ),
                        ],
                    }
                    for service in services
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return wizard_log_dir


def _mock_docker_command(
    *,
    alpasim: Path,
    humanoid: Path,
    compose_path: Path,
    omit_humanoid_mount: bool = False,
    services: tuple[str, ...] = ("runtime-0", "humanoid_dynamics-0"),
    alpasim_src_destination: str = "/repo/src",
    alpasim_src_read_only: bool = True,
    include_visual_caches: bool = True,
    cache_services: tuple[str, ...] | None = None,
    runtime_cache_read_only: bool = False,
    extra_bind_mounts: tuple[tuple[Path, str, bool], ...] = (),
    service_bind_mounts: dict[str, tuple[tuple[Path, str, bool], ...]] | None = None,
    compose_project: str | None = None,
):
    """Return a deterministic Docker command fake for a two-service runtime."""

    resolved_cache_services = set(
        services if cache_services is None else cache_services
    )
    resolved_service_bind_mounts = service_bind_mounts or {}

    def run(command: list[str]) -> bytes:
        """Return Docker CLI JSON for the requested inspection."""
        if command[0] == "git":
            return subprocess.run(command, check=True, capture_output=True).stdout
        if command[:2] == ["docker", "compose"]:
            return "".join(f"{service}-id\n" for service in services).encode()
        if command[:2] == ["docker", "inspect"]:
            return json.dumps(
                [
                    _container_inspection(
                        container_id=f"{service}-id",
                        service=service,
                        alpasim=alpasim,
                        humanoid=humanoid,
                        compose_path=compose_path,
                        omit_humanoid_mount=omit_humanoid_mount,
                        alpasim_src_destination=alpasim_src_destination,
                        alpasim_src_read_only=alpasim_src_read_only,
                        include_visual_caches=(
                            include_visual_caches and service in resolved_cache_services
                        ),
                        runtime_cache_read_only=runtime_cache_read_only,
                        extra_bind_mounts=(
                            *extra_bind_mounts,
                            *resolved_service_bind_mounts.get(service, ()),
                        ),
                        compose_project=compose_project,
                    )
                    for service in services
                ]
            ).encode()
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": _TEST_IMAGE_ID,
                        "Config": {"Labels": {"org.example.runtime-contract": "v1"}},
                        "RepoDigests": ["runtime@example-sha256"],
                        "RepoTags": ["runtime:local"],
                    }
                ]
            ).encode()
        raise AssertionError(f"unexpected command: {command}")

    return run


def _container_inspection(
    *,
    container_id: str,
    service: str,
    alpasim: Path,
    humanoid: Path,
    compose_path: Path,
    omit_humanoid_mount: bool,
    alpasim_src_destination: str = "/repo/src",
    alpasim_src_read_only: bool = True,
    include_visual_caches: bool = True,
    runtime_cache_read_only: bool = False,
    extra_bind_mounts: tuple[tuple[Path, str, bool], ...] = (),
    compose_project: str | None = None,
) -> dict[str, object]:
    """Build one minimal `docker inspect` object."""
    sources = [
        (alpasim / "src", alpasim_src_destination),
        (alpasim / "plugins", "/repo/plugins"),
        (humanoid.parent / "scene_store", "/mnt/humanoid-scene-store"),
        *(
            [
                (humanoid.parent / "scene_cache", "/mnt/humanoid-scene-cache"),
                (humanoid.parent / "runtime_cache", "/root/.cache"),
            ]
            if include_visual_caches
            else []
        ),
    ]
    if not omit_humanoid_mount:
        sources.append((humanoid, "/repo/humanoid-rl-joint-sim"))
    sources.extend(
        (source, destination) for source, destination, _ in extra_bind_mounts
    )
    return {
        "Id": container_id,
        "Name": f"/{service}",
        "Image": _TEST_IMAGE_ID,
        "ImageManifestDescriptor": {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": _TEST_MANIFEST_DIGEST,
            "platform": {"architecture": "amd64", "os": "linux"},
        },
        "State": {"Running": True},
        "Config": {
            "Image": "runtime:local",
            "Entrypoint": ["bash"],
            "Cmd": ["run"],
            "Labels": {
                "com.docker.compose.service": service,
                "com.docker.compose.project": (
                    compose_project
                    if compose_project is not None
                    else wizard_compose_project(compose_path.parent)
                ),
                "com.docker.compose.config-hash": f"hash-{service}",
                "com.docker.compose.project.config_files": str(compose_path),
            },
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(source),
                "Destination": destination,
                "Mode": (
                    "rw"
                    if (
                        (source == alpasim / "src" and not alpasim_src_read_only)
                        or (
                            source == humanoid.parent / "runtime_cache"
                            and not runtime_cache_read_only
                        )
                        or any(
                            source == extra_source and extra_read_write
                            for extra_source, _, extra_read_write in extra_bind_mounts
                        )
                    )
                    else "ro"
                ),
                "RW": (
                    (source == alpasim / "src" and not alpasim_src_read_only)
                    or (
                        source == humanoid.parent / "runtime_cache"
                        and not runtime_cache_read_only
                    )
                    or any(
                        source == extra_source and extra_read_write
                        for extra_source, _, extra_read_write in extra_bind_mounts
                    )
                ),
                "Propagation": "rprivate",
            }
            for source, destination in sources
        ],
    }


def _git(root: Path, *arguments: str) -> None:
    """Run one Git setup command for a temporary worktree."""
    subprocess.run(["git", "-C", str(root), *arguments], check=True)
