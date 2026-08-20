# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL transport: rollout-side egress writer and trainer-side packer mixin.

Backed by ``torch.distributed.TCPStore``. The rollout side holds an
:class:`NcclEpisodeWriter` (sender + discard-cleanup subscriber). The trainer side
composes :class:`NcclDataPackerMixin` over the AlpaGym packer. The mixin resolves
``nccl:`` handles inline on the policy loop, then delegates the resulting
``EpisodeOutput`` to the normal replay collation path. The small reconstruction
manifest travels through the store; bulk tensors flow over NCCL.
"""

import json
import logging
import threading
import uuid
from collections.abc import Mapping
from typing import Any

import redis
import torch
from torch.distributed import DistStoreError, TCPStore

from alpagym_runtime.cosmos.packer import AlpagymDataPacker
from alpagym_runtime.replay import DataPackerConfig
from alpagym_runtime.tensor_utils import to_device_recursive
from alpagym_runtime.transport.nccl.payload import (
    TENSOR_KEY_MARKER,
    WirePayload,
    pack,
    unpack,
)
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_COMPLETION_PREFIX,
    build_metadata_key,
    normalize_nccl_handle,
    to_external_nccl_handle,
)
from alpagym_runtime.transport.nccl.receiver import NcclReceiver
from alpagym_runtime.transport.nccl.sender import NcclSender
from alpagym_runtime.types import EpisodeOutput

logger = logging.getLogger(__name__)


class NcclEpisodeWriter:
    """Rollout-side NCCL egress.

    Registers each episode's tensors with the sender, publishes its
    reconstruction manifest to the store, and runs the controller-to-rollout
    discard-cleanup subscriber so stale sender buffers are freed.
    """

    def __init__(
        self,
        store: TCPStore,
        sender: NcclSender,
        experiment_name: str,
        job_id: str,
        flush_timeout_seconds: float = 300.0,
    ) -> None:
        """Wire the writer to its sender and the run identity used for the cleanup channel."""
        self._store = store
        self._sender = sender
        self._experiment_name = experiment_name
        self._job_id = job_id
        self._flush_timeout_seconds = flush_timeout_seconds
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_pubsub: Any | None = None
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def rollout_idx(self) -> int:
        """Return this sender's rollout index."""
        return self._sender.rollout_idx

    def write(self, episode: EpisodeOutput) -> str:
        """Register ``episode``'s tensors with the sender, publish the manifest, return the handle.

        ``start_cleanup`` runs first in the cosmos lifecycle (the packer's
        ``post_redis_injection``), so the discard-cleanup subscriber is active
        before any handle is emitted.
        """
        # Receivers route by the leading rollout_idx in the transfer id.
        transfer_id = f"{self._sender.rollout_idx}:{uuid.uuid4().hex}"
        payload = pack(episode)
        # Register tensors locally; the actual nccl_send fires from the sender's
        # background thread once the receiver issues its rendezvous request.
        self._sender.send(payload.tensors, transfer_id)
        # Publish the manifest only after local registration; roll the
        # registration back if the publish fails so nothing leaks.
        try:
            self._store.set(
                build_metadata_key(transfer_id),
                json.dumps({"manifest": payload.manifest}),
            )
        except Exception:
            self._sender.release(transfer_id, reason="metadata_publish_error")
            raise
        return to_external_nccl_handle(transfer_id)

    def release(self, handle: str, reason: str) -> None:
        """Drop one transfer: delete its manifest and free the sender's pending tensors."""
        transfer_id = normalize_nccl_handle(handle)
        try:
            self._store.delete_key(build_metadata_key(transfer_id))
        except DistStoreError:
            pass
        self._sender.release(transfer_id, reason=reason)

    def flush_pending_sends(self) -> None:
        """Wait for any in-flight send to finish before weight sync.

        Held payloads are left registered and ship after the broadcast; only a
        send already in flight would collide with the R2R communicator.
        """
        if not self._sender.wait_until_drained(self._flush_timeout_seconds):
            raise TimeoutError(
                f"NCCL sender still has an in-flight transfer after "
                f"{self._flush_timeout_seconds:.1f}s; cannot start weight sync while a data "
                "transfer is in flight on the same GPU."
            )

    def start_cleanup(self, redis_client: redis.Redis) -> None:
        """Subscribe to the controller's discard-cleanup channel for this rollout replica.

        Cosmos's controller publishes ``{"transfer_id": ...}`` for outdated rollouts;
        on each message the writer frees the matching sender buffer.
        """
        if redis_client is None:
            raise RuntimeError(
                "NcclEpisodeWriter.start_cleanup requires an injected Redis client"
            )
        if self._closed:
            raise RuntimeError("NcclEpisodeWriter.start_cleanup called after close")
        if self._cleanup_running():
            return
        # cosmos channel helpers are pure-Python; the rollout subscriber must use
        # them so it listens on the exact channel the controller publishes on. This
        # couples cleanup to the cosmos rev: these builders must exist in
        # cosmos_rl.utils.payload_transport.nccl at the pinned rev. A rename or removal
        # fails loudly here at import rather than silently routing cleanup to a dead channel.
        from cosmos_rl.utils.payload_transport.nccl import (
            build_cleanup_channel,
            build_nccl_prefix,
            build_rollout_prefix,
        )

        prefix = build_nccl_prefix(
            experiment_name=self._experiment_name, job_id=self._job_id
        )
        channel = build_cleanup_channel(
            build_rollout_prefix(prefix, self._sender.rollout_idx)
        )
        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._run_cleanup_listener,
            args=(channel, redis_client),
            name=f"alpagym-nccl-cleanup-{self._sender.rollout_idx}",
            daemon=True,
        )
        self._cleanup_thread.start()

    def close(self) -> None:
        """Stop the cleanup subscriber and tear down the sender."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_cleanup()
            self._sender.close()

    def _cleanup_running(self) -> bool:
        """Return whether the cleanup subscriber thread is alive."""
        return self._cleanup_thread is not None and self._cleanup_thread.is_alive()

    def _stop_cleanup(self) -> None:
        """Signal and join the cleanup subscriber thread."""
        self._cleanup_stop_event.set()
        thread = self._cleanup_thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    "NCCL cleanup subscriber did not stop within close() timeout"
                )
                return
        self._cleanup_thread = None

    def _run_cleanup_listener(self, channel: str, redis_client: redis.Redis) -> None:
        """Poll the Redis cleanup channel and free the sender buffer for each discard."""
        pubsub = redis_client.pubsub()
        self._cleanup_pubsub = pubsub
        try:
            pubsub.subscribe(channel)
            logger.info("Listening for NCCL cleanup messages on %s", channel)
            while not self._cleanup_stop_event.is_set():
                try:
                    message = pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=0.5
                    )
                except Exception as error:
                    if not self._cleanup_stop_event.is_set():
                        logger.warning(
                            "NCCL cleanup subscriber stopped after Redis error: %s",
                            error,
                        )
                    break
                if message is None:
                    continue
                self._process_cleanup_message(message)
        finally:
            try:
                pubsub.close()
            except Exception as error:
                logger.debug(
                    "Ignoring Redis pubsub close error during NCCL cleanup shutdown: %s",
                    error,
                )
            if self._cleanup_pubsub is pubsub:
                self._cleanup_pubsub = None

    def _process_cleanup_message(self, message: Any) -> None:
        """Free one discarded transfer; log at ERROR and continue on malformed payloads.

        A malformed message means the controller's cleanup payload drifted from
        the ``{"transfer_id": ...}`` contract, which would otherwise leak sender
        buffers with no visible signal. Log at ERROR so the drift surfaces, and
        let an unexpected (non-parse) error propagate to stop the listener.
        """
        try:
            data = message["data"] if isinstance(message, Mapping) else message
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            payload = json.loads(data)
            transfer_id = payload["transfer_id"]
            if not isinstance(transfer_id, str):
                raise ValueError("transfer_id must be a string")
        except (KeyError, TypeError, ValueError) as error:
            logger.error("Ignoring malformed NCCL cleanup message: %s", error)
            return
        self.release(transfer_id, "cosmos_cleanup")


class NcclDataPackerMixin:
    """Resolve ``nccl:`` rollout handles to episodes over the NCCL receiver.

    Composed in front of :class:`AlpagymDataPacker`; ``get_policy_input`` resolves
    ``nccl:`` handles synchronously on the policy loop, then delegates the
    resulting :class:`EpisodeOutput` to the underlying packer. The receiver state
    is set by :class:`NcclAlpagymDataPacker`.
    """

    _store: TCPStore
    _receiver: NcclReceiver
    _target_device: torch.device

    def get_policy_input(
        self,
        sample: Any,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Resolve ``nccl:`` handles inline before replay collation."""
        if isinstance(rollout_output, str) and rollout_output.startswith(
            NCCL_COMPLETION_PREFIX
        ):
            rollout_output = self._resolve_nccl_handle(rollout_output)
        return super().get_policy_input(
            sample, rollout_output, n_ignore_prefix_tokens, **kwargs
        )

    def _resolve_nccl_handle(self, handle: str) -> EpisodeOutput:
        """Resolve the manifest, rendezvous + receive the tensors, and unpack the episode.

        One-shot per handle: a successful read removes the metadata key, and terminal
        pre-receive failures remove it explicitly. cosmos-rl enforces single-consumer
        per handle at the policy-worker boundary.
        """
        transfer_id = normalize_nccl_handle(handle)
        # A missing manifest either means cleanup already discarded an outdated
        # rollout, or the handle broke the transport contract. Fail before rendezvous.
        metadata_key = build_metadata_key(transfer_id)
        if not self._store.check([metadata_key]):
            raise RuntimeError(f"No metadata for transfer {handle}")
        metadata_raw = self._store.get(metadata_key)
        metadata_str = (
            metadata_raw.decode("utf-8")
            if isinstance(metadata_raw, bytes)
            else metadata_raw
        )
        try:
            manifest = json.loads(metadata_str)["manifest"]
        except Exception:
            self._store.delete_key(metadata_key)
            raise
        # Read each tensor's (shape, dtype) so the receiver can allocate
        # destination buffers before calling nccl_recv.
        try:
            tensor_specs = _collect_tensor_specs(manifest)
        except Exception:
            self._store.delete_key(metadata_key)
            raise
        # Triggers the rendezvous handshake and the matching nccl_send; blocks
        # until every tensor arrives or the receiver's watchdog fires.
        try:
            tensors = self._receiver.recv(
                transfer_id=transfer_id, tensor_specs=tensor_specs
            )
        except Exception:
            self._store.delete_key(metadata_key)
            raise
        self._store.delete_key(metadata_key)
        tensors_on_device = {
            key: to_device_recursive(tensor, self._target_device)
            for key, tensor in tensors.items()
        }
        return unpack(WirePayload(tensors=tensors_on_device, manifest=manifest))


class NcclAlpagymDataPacker(NcclDataPackerMixin, AlpagymDataPacker):
    """Trainer-side AlpaGym packer that resolves ``nccl:`` handles over the receiver."""

    def __init__(
        self,
        config: DataPackerConfig,
        build_model_inputs: Any,
        store: TCPStore,
        receiver: NcclReceiver,
        target_device: torch.device,
        collate_samples: Any = None,
    ) -> None:
        """Wire the trainer packer to its NCCL receiver and TCPStore connection."""
        super().__init__(
            config,
            build_model_inputs,
            collate_samples=collate_samples,
        )
        self._store = store
        self._receiver = receiver
        self._target_device = target_device

    def close(self) -> None:
        """Tear down the receiver."""
        self._receiver.close()
        super().close()


def _collect_tensor_specs(manifest: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    """Walk the manifest and collect (tensor_key -> (shape, dtype)) entries."""
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    _walk(manifest, specs)
    return specs


def _walk(value: Any, specs: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """Recursively collect tensor-reference dicts into ``specs``."""
    if isinstance(value, dict):
        if TENSOR_KEY_MARKER in value:
            specs[value[TENSOR_KEY_MARKER]] = (
                _parse_tensor_shape(value["shape"]),
                _parse_tensor_dtype(value["dtype"]),
            )
            return
        for nested in value.values():
            _walk(nested, specs)
    elif isinstance(value, list):
        for item in value:
            _walk(item, specs)


def _parse_tensor_shape(shape: Any) -> tuple[int, ...]:
    """Validate a manifest tensor shape before the receiver allocates buffers."""
    if not isinstance(shape, list):
        raise ValueError(f"NCCL tensor metadata shape must be a list, got {shape!r}")
    dims = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in dims):
        raise ValueError(f"NCCL tensor metadata shape has a negative dim: {shape!r}")
    return dims


def _parse_tensor_dtype(dtype: Any) -> str:
    """Validate a manifest tensor dtype before it reaches the recv daemon."""
    if not isinstance(dtype, str) or not dtype.startswith("torch."):
        raise ValueError(
            f"NCCL tensor metadata dtype must be a torch dtype, got {dtype!r}"
        )
    dtype_name = dtype.removeprefix("torch.")
    torch_dtype = getattr(torch, dtype_name, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"NCCL tensor metadata dtype is unsupported: {dtype!r}")
    return dtype
