# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-ownership behavior of ``AlpagymDataPacker``.

These cover the rollout-side egress (``get_rollout_output``) and the
trainer-side handle resolution. Replay collation is covered in
``test_replay_packer``.
"""

import sys
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alpagym_host.config import CosmosRLMode, TransportKind
from alpagym_runtime.cosmos.packer import AlpagymDataPacker, build_alpagym_data_packer
from alpagym_runtime.replay import DataPackerConfig
from alpagym_runtime.types import EpisodeOutput


class _StubWriter:
    """Rollout-side writer stub recording each egress."""

    def __init__(self) -> None:
        self.written: list[EpisodeOutput] = []

    def write(self, episode: EpisodeOutput) -> str:
        self.written.append(episode)
        return f"handle-{len(self.written)}"


def _episode(session_uuid: str) -> EpisodeOutput:
    """Build a minimal episode for packer tests."""
    return EpisodeOutput(
        scene_id="scene",
        session_uuid=session_uuid,
        num_steps=0,
        policy_outputs=(),
    )


def _rollout_packer(
    writer: _StubWriter | None, *, non_text: bool = True
) -> AlpagymDataPacker:
    """Build a packer with a stub writer and a minimal cosmos config."""
    packer = AlpagymDataPacker(
        DataPackerConfig(expected_valid_steps=1),
        build_model_inputs=lambda replay_data: ({}, None),
        writer=writer,
    )
    packer.config = SimpleNamespace(train=SimpleNamespace(non_text=non_text))
    return packer


def test_get_rollout_output_egresses_each_completion_to_writer() -> None:
    """Each in-memory episode is written to the writer, replaced by its handle."""
    writer = _StubWriter()
    packer = _rollout_packer(writer)
    episodes = [_episode("a"), _episode("b")]

    handles, conversations, logprobs, token_ids, extra = packer.get_rollout_output(
        episodes, ["conv"], ["lp"], ["tok"]
    )

    assert handles == ["handle-1", "handle-2"]
    assert writer.written == episodes
    # Non-completion fields pass through untouched.
    assert conversations == ["conv"]
    assert logprobs == ["lp"]
    assert token_ids == ["tok"]
    assert extra == {}


def test_get_rollout_output_requires_non_text() -> None:
    """Carrying live episodes to egress is only safe under train.non_text."""
    packer = _rollout_packer(_StubWriter(), non_text=False)
    with pytest.raises(RuntimeError, match="non_text"):
        packer.get_rollout_output([_episode("a")], [], [], [])


def test_get_rollout_output_rejects_non_episode_completion() -> None:
    """A non-episode completion (e.g. an already-egressed handle) fails loudly."""
    packer = _rollout_packer(_StubWriter())
    with pytest.raises(TypeError, match="EpisodeOutput"):
        packer.get_rollout_output(["nccl:already-a-handle"], [], [], [])


def test_get_rollout_output_without_writer_raises() -> None:
    """A trainer/controller packer has no writer and cannot egress."""
    packer = _rollout_packer(None)
    with pytest.raises(RuntimeError, match="rollout writer"):
        packer.get_rollout_output([_episode("a")], [], [], [])


def test_flush_and_close_no_op_without_writer() -> None:
    """A writerless packer's flush/close are safe no-ops."""
    packer = _rollout_packer(None)
    packer.flush_pending_sends()
    packer.close()


def _disk_run_config(tmp_path: Path, mode: CosmosRLMode) -> SimpleNamespace:
    """Minimal RunConfig stand-in for the disk-Policy ``build_alpagym_data_packer`` path."""
    return SimpleNamespace(
        expected_valid_steps=1,
        transport=SimpleNamespace(kind=TransportKind.disk),
        cosmos=SimpleNamespace(mode=mode),
        artifact_paths=SimpleNamespace(artifacts_dir=str(tmp_path)),
    )


def _nccl_run_config(tmp_path: Path) -> SimpleNamespace:
    """Minimal RunConfig stand-in for NCCL packer TCPStore wiring."""
    return SimpleNamespace(
        expected_valid_steps=1,
        transport=SimpleNamespace(
            kind=TransportKind.nccl,
            nccl_env={"NCCL_TIMEOUT": "7"},
            nccl_read_device="cpu",
        ),
        artifact_paths=SimpleNamespace(
            artifacts_dir=str(tmp_path / "artifacts"),
            topology_registry_dir=tmp_path / "topology",
        ),
        cosmos=SimpleNamespace(
            mode=CosmosRLMode.disaggregated,
            logging=SimpleNamespace(experiment_name="exp"),
            launch=SimpleNamespace(policy_replicas=1, rollout_replicas=1),
            policy=SimpleNamespace(parallelism=SimpleNamespace(dp_shard_size=1)),
        ),
    )


def _built_disk_policy_packer(tmp_path: Path, mode: CosmosRLMode) -> AlpagymDataPacker:
    """Build the disk ``Policy`` packer via the role/mode wiring under test."""
    packer = build_alpagym_data_packer(
        _disk_run_config(tmp_path, mode),
        cosmos_role="Policy",
        build_model_inputs=lambda replay_data: ({}, None),
    )
    packer.config = SimpleNamespace(train=SimpleNamespace(non_text=True))
    return packer


def test_colocated_disk_policy_packer_can_egress(tmp_path: Path) -> None:
    """Colocated Policy shares its packer with the in-process rollout worker, so it egresses.

    Regression: ``default.yaml`` ships ``mode=colocated``+``transport=disk``; without a
    writer on the Policy packer the colocated rollout worker raises on first egress.
    """
    packer = _built_disk_policy_packer(tmp_path, CosmosRLMode.colocated)
    handles, *_ = packer.get_rollout_output([_episode("a")], [], [], [])
    assert len(handles) == 1
    assert Path(handles[0]).exists()


def test_disaggregated_disk_policy_packer_is_read_only(tmp_path: Path) -> None:
    """The disaggregated disk trainer holds no writer and cannot egress."""
    packer = _built_disk_policy_packer(tmp_path, CosmosRLMode.disaggregated)
    with pytest.raises(RuntimeError, match="rollout writer"):
        packer.get_rollout_output([_episode("a")], [], [], [])


def test_nccl_policy_packer_preserves_policy_collator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NCCL wrapper must retain the same policy-owned collation hook."""
    from alpagym_runtime.cosmos import packer as packer_module
    from alpagym_runtime.transport.nccl.endpoints import NcclAlpagymDataPacker

    def _custom_collator(samples: list[Any]) -> Any:
        return samples

    monkeypatch.setattr(
        packer_module,
        "_build_nccl_receiver",
        lambda run_config: (object(), object(), "cpu"),
    )

    packer = build_alpagym_data_packer(
        _nccl_run_config(tmp_path),
        cosmos_role="Policy",
        build_model_inputs=lambda replay_data: ({}, None),
        collate_samples=_custom_collator,
    )

    assert isinstance(packer, NcclAlpagymDataPacker)
    assert packer._collate_samples is _custom_collator


def test_nccl_tcpstore_clients_use_configured_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both NCCL client roles give TCPStore the configured rendezvous timeout."""
    from alpagym_host.endpoint_registry import FileTopologyRegistry
    from alpagym_runtime.cosmos import packer as packer_module
    from alpagym_runtime.transport.nccl import (
        receiver as receiver_module,
        sender as sender_module,
    )

    run_config = _nccl_run_config(tmp_path)
    FileTopologyRegistry(
        run_config.artifact_paths.topology_registry_dir
    ).publish_nccl_master(
        host="controller-node",
        port=29501,
    )
    tcpstore_calls: list[dict[str, object]] = []

    class _FakeTCPStore:
        """Record TCPStore construction without opening a socket."""

        def __init__(self, **kwargs: object) -> None:
            """Store constructor arguments for assertions."""
            tcpstore_calls.append(kwargs)

    class _FakeSender:
        """Stand in for NcclSender after TCPStore construction."""

        rollout_idx = 0
        comm_idx = 0
        drain_timeout_seconds: float | None = None

        def __init__(self, **kwargs: object) -> None:
            """Accept sender construction kwargs."""
            del kwargs
            fake_senders.append(self)

        def setup(self) -> None:
            """Accept sender setup."""

        def wait_until_drained(self, timeout_seconds: float) -> bool:
            """Record the writer flush timeout and report no in-flight send."""
            self.drain_timeout_seconds = timeout_seconds
            return True

    fake_senders: list[_FakeSender] = []

    class _FakeReceiver:
        """Stand in for NcclReceiver after TCPStore construction."""

        rollout_comms = {0: 0}

        def __init__(self, **kwargs: object) -> None:
            """Accept receiver construction kwargs."""
            del kwargs

        def setup(self) -> None:
            """Accept receiver setup."""

    def _create_nccl_comm(*args: object, **kwargs: object) -> int:
        """Return a fake communicator id."""
        del args, kwargs
        return 0

    def _create_nccl_uid() -> list[int]:
        """Return a fake NCCL UID."""
        return [0]

    def _nccl_noop(*args: object, **kwargs: object) -> None:
        """Accept fake pynccl operations."""
        del args, kwargs

    pynccl_module = types.ModuleType("cosmos_rl.utils.pynccl")
    pynccl_module.create_nccl_comm = _create_nccl_comm
    pynccl_module.create_nccl_uid = _create_nccl_uid
    pynccl_module.nccl_abort = _nccl_noop
    pynccl_module.nccl_recv = _nccl_noop
    pynccl_module.nccl_send = _nccl_noop
    monkeypatch.setitem(sys.modules, "cosmos_rl.utils.pynccl", pynccl_module)
    monkeypatch.setattr(packer_module, "TCPStore", _FakeTCPStore)
    monkeypatch.setattr(sender_module, "assign_rollout_idx", lambda **kwargs: 0)
    monkeypatch.setattr(sender_module, "NcclSender", _FakeSender)
    monkeypatch.setattr(receiver_module, "NcclReceiver", _FakeReceiver)

    writer = packer_module._build_episode_writer(run_config, is_nccl=True)
    packer_module._build_nccl_receiver(run_config)
    writer.flush_pending_sends()

    assert [call["timeout"] for call in tcpstore_calls] == [
        timedelta(seconds=7),
        timedelta(seconds=7),
    ]
    assert [call["host_name"] for call in tcpstore_calls] == [
        "controller-node",
        "controller-node",
    ]
    assert [call["port"] for call in tcpstore_calls] == [29501, 29501]
    assert len(fake_senders) == 1
    assert fake_senders[0].drain_timeout_seconds == 7
