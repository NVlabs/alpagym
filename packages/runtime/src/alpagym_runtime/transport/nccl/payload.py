# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack EpisodeOutput across the NCCL split."""

import collections.abc
import dataclasses
import importlib
import types
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch

from alpagym_runtime.types import EpisodeOutput

TENSOR_KEY_MARKER = "__tensor_key__"
_DATACLASS_TYPE_MARKER = "__dataclass_type__"
_BOOL_TENSOR_MARKER = "__bool_tensor__"
_RESERVED_DICT_MARKERS = {
    TENSOR_KEY_MARKER,
    _DATACLASS_TYPE_MARKER,
    _BOOL_TENSOR_MARKER,
}
_ALLOWED_DATACLASS_TYPE_PREFIX = "alpagym_runtime."


@dataclass(frozen=True)
class WirePayload:
    """One transfer's wire-format: the bulk tensors and the structural manifest.

    ``tensors`` is a flat ``{key: Tensor}`` map shipped over NCCL.
    ``manifest`` mirrors the EpisodeOutput shape with tensor leaves replaced
    by reference dicts pointing back into ``tensors``; it ships through the
    TCPStore metadata channel.
    """

    tensors: dict[str, torch.Tensor]
    manifest: dict[str, Any]


def pack(episode: EpisodeOutput) -> WirePayload:
    """Split ``episode`` into the wire-format tensors + manifest."""
    tensors: dict[str, torch.Tensor] = {}
    manifest = _pack(episode, tensors)
    return WirePayload(tensors=tensors, manifest=manifest)


def unpack(payload: WirePayload) -> EpisodeOutput:
    """Reconstruct the original EpisodeOutput from a :class:`WirePayload`."""
    return _unpack(payload.manifest, EpisodeOutput, payload.tensors)


def _pack(
    value: Any,
    tensors: dict[str, torch.Tensor],
    *,
    reject_empty_tensors: bool = True,
    encode_bool_as_uint8: bool = True,
) -> Any:
    """Recursively pack a value; extract tensors to the flat ``tensors`` map.

    Dataclass instances are stamped with a ``__dataclass_type__`` marker
    alongside their field values so unpack can recover the original type
    even when the surrounding slot is ``dict[str, Any]`` and would
    otherwise dispatch through :func:`_resolve_tensor_refs`.

    The defaults implement NCCL's wire constraints. The colocated disk
    transport reuses the manifest ABI but disables the empty-tensor rejection
    and bool-to-uint8 conversion because ``torch.save`` supports both natively.
    """
    if isinstance(value, torch.Tensor):
        if reject_empty_tensors and value.numel() == 0:
            # pynccl rejects empty buffers; fail here at pack time rather than
            # after the manifest is published and the rendezvous is open, which
            # would raise mid-send and poison the communicator.
            raise ValueError(
                f"NCCL transport cannot ship a zero-element tensor (shape={tuple(value.shape)})"
            )
        key = f"tensor_{len(tensors)}"
        # pynccl cannot send torch.bool; ship bool tensors as uint8 and restore
        # the bool dtype on unpack (see _resolve_tensor_leaf).
        wire = (
            value.to(torch.uint8)
            if encode_bool_as_uint8 and value.dtype == torch.bool
            else value
        )
        tensors[key] = wire
        leaf: dict[str, Any] = {
            TENSOR_KEY_MARKER: key,
            "shape": list(value.shape),
            "dtype": str(wire.dtype),
        }
        if encode_bool_as_uint8 and value.dtype == torch.bool:
            leaf[_BOOL_TENSOR_MARKER] = True
        return leaf
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        packed = {
            field.name: _pack(
                getattr(value, field.name),
                tensors,
                reject_empty_tensors=reject_empty_tensors,
                encode_bool_as_uint8=encode_bool_as_uint8,
            )
            for field in dataclasses.fields(value)
        }
        value_type = type(value)
        packed[_DATACLASS_TYPE_MARKER] = (
            f"{value_type.__module__}.{value_type.__qualname__}"
        )
        return packed
    if isinstance(value, (list, tuple)):
        return [
            _pack(
                item,
                tensors,
                reject_empty_tensors=reject_empty_tensors,
                encode_bool_as_uint8=encode_bool_as_uint8,
            )
            for item in value
        ]
    if isinstance(value, dict):
        reserved_keys = sorted(set(value) & _RESERVED_DICT_MARKERS)
        if reserved_keys:
            raise ValueError(
                f"Transport payload dict uses reserved manifest keys: {reserved_keys}"
            )
        return {
            k: _pack(
                v,
                tensors,
                reject_empty_tensors=reject_empty_tensors,
                encode_bool_as_uint8=encode_bool_as_uint8,
            )
            for k, v in value.items()
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_tensor_leaf(
    leaf: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return the tensor for a manifest tensor-ref leaf, restoring bool dtype if marked."""
    tensor = tensors[leaf[TENSOR_KEY_MARKER]]
    if leaf.get(_BOOL_TENSOR_MARKER):
        return tensor.to(torch.bool)
    return tensor


def _unpack(value: Any, type_hint: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Recursively rebuild a typed value from its manifest entry.

    Dispatches on two axes:
        - value shape: a tensor-ref dict short-circuits to the flat ``tensors`` map.
        - type_hint kind: ``Optional[T]``, ``tuple``, ``dict``/``Mapping``, dataclass,
          ``Any``, or leaf (primitive).
    """
    if isinstance(value, dict) and TENSOR_KEY_MARKER in value:
        return _resolve_tensor_leaf(value, tensors)

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Union or origin is types.UnionType:
        if value is None:
            return None
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) != 1:
            raise ValueError(
                f"Only Optional[T] unions are supported in NCCL payload unpack; got {type_hint!r}"
            )
        non_none_type = non_none_types[0]
        return _unpack(value, non_none_type, tensors)

    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_unpack(item, args[0], tensors) for item in value)
        return tuple(_unpack(item, t, tensors) for item, t in zip(value, args))

    if origin is dict or origin is collections.abc.Mapping:
        _, value_type = args
        # `dict[..., Any]` (e.g. replay payload, model_extra) has no schema
        # for its leaves; walk them and swap tensor refs in place.
        if value_type is Any:
            return {k: _resolve_tensor_refs(v, tensors) for k, v in value.items()}
        return {k: _unpack(v, value_type, tensors) for k, v in value.items()}

    if dataclasses.is_dataclass(type_hint):
        field_types = get_type_hints(type_hint)
        # Index every field directly: the NCCL split is a same-version in-process
        # round-trip and ``_pack`` emits every field, so a missing manifest field is
        # corruption, not version skew. Fail fast at the transport boundary (KeyError)
        # rather than silently substituting a default (e.g. reward=None, is_valid=True).
        kwargs = {
            f.name: _unpack(value[f.name], field_types[f.name], tensors)
            for f in dataclasses.fields(type_hint)
        }
        return type_hint(**kwargs)

    if type_hint is Any:
        return _resolve_tensor_refs(value, tensors)

    return value


def _resolve_tensor_refs(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Walk an Any-typed value, swapping tensor refs for tensors and rebuilding dataclasses.

    Recognized markers (handled before generic dict recursion):

    - ``__tensor_key__`` — swap for the matching entry in ``tensors``.
    - ``__dataclass_type__`` — import the type and reconstruct via
      :func:`_unpack` with the resolved field annotations, so typed leaves
      survive a round-trip through ``dict[str, Any]`` slots.
    """
    if isinstance(value, dict):
        if TENSOR_KEY_MARKER in value:
            return _resolve_tensor_leaf(value, tensors)
        if _DATACLASS_TYPE_MARKER in value:
            return _reconstruct_dataclass(value, tensors)
        return {k: _resolve_tensor_refs(v, tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_tensor_refs(item, tensors) for item in value]
    return value


def _reconstruct_dataclass(
    value: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> Any:
    """Rebuild a dataclass instance from its packed dict using the embedded type path."""
    type_path = value[_DATACLASS_TYPE_MARKER]
    if not type_path.startswith(_ALLOWED_DATACLASS_TYPE_PREFIX):
        raise ValueError(
            f"Refusing to reconstruct dataclass outside "
            f"{_ALLOWED_DATACLASS_TYPE_PREFIX!r}: {type_path!r}"
        )
    module_name, _, class_name = type_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise ValueError(f"Resolved type {type_path!r} is not a dataclass class")
    field_types = get_type_hints(cls)
    kwargs = {
        f.name: _unpack(value[f.name], field_types[f.name], tensors)
        for f in dataclasses.fields(cls)
    }
    return cls(**kwargs)
