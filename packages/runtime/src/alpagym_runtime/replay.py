# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected-action replay contracts shared by rollout, packing, and training.

Rollout code records the selected action once, plus the model-family-specific
payload needed to score that exact action later. The trainer consumes
rollout-side training signals; it does not derive advantages from rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

import torch

_REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActionSelection:
    """Action selected by rollout.

    Shape notes:
        ``set_ix`` is a scalar index into ``N`` / repo name ``ns``.
        ``sample_ix`` is a scalar index into ``K`` / repo name ``nj``.
    """

    set_ix: int
    sample_ix: int

    def __post_init__(self) -> None:
        """Validate candidate selection semantics."""
        if self.set_ix < 0 or self.sample_ix < 0:
            raise ValueError("ActionSelection indices must be non-negative")


@dataclass(frozen=True)
class PolicyReplayData:
    """Selected-action replay envelope attached to ``PolicyOutput.replay_data``.

    The payload is not a generic training-data container. It is the data needed
    to replay and rescore the exact action selected during rollout.

    Shape notes:
        ``old_logprob`` is scalar trajectory-level rollout-policy logprob when
        the family can expose one. ``payload`` is a family-owned tensor tree.
    """

    replay_schema_version: int
    payload_schema: str
    payload_schema_version: int
    model_family: str
    action_selection: ActionSelection
    old_logprob: torch.Tensor | None
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping, leaving tensors for the disk hook."""
        return _dataclass_to_plain(self)


@dataclass(frozen=True)
class TrainingSignal:
    """Row-aligned trainer signal carrying rollout-time training terms.

    Shape notes:
        ``old_logprobs`` is ``[BT]`` trajectory-level rollout-policy logprob.
        ``is_padding`` is ``[BT]`` and masks packer-added rows.
        Transition fields are optional tensors used by actor-critic trainers.
        Scalar ``rewards`` and terminal/value fields are ``[BT]``. A
        semi-Markov action may additionally carry the controller-tick rewards
        it committed as ``primitive_rewards`` plus a same-shaped validity mask
        ``[BT, K]`` and an integer ``duration_ticks`` vector ``[BT]``.
        ``actor_primitive_rewards`` is a same-shaped causal actor view. It can
        remove a dense support-certification increment whose five-tick evidence
        is majority-owned by the predecessor while leaving the chronological
        critic rewards unchanged. ``actor_primitive_reward_mask`` excludes the
        predecessor-plan prefix that elapsed before the sampled plan reached
        the controller. Both actor tensors retain the original tick positions.
        ``old_values`` and ``bootstrap_values`` are rollout-policy value
        estimates at macro decision boundaries.
        ``actor_valid`` is ``[BT]`` and is false when a sampled action never
        reached the controller; those rows still train the value function and
        participate in GAE, but never enter actor-density losses or diagnostics.
        ``advantages`` and ``returns`` are trainer-derived and are never required
        in rollout replay payloads.
    """

    old_logprobs: torch.Tensor
    is_padding: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    terminateds: torch.Tensor | None = None
    truncateds: torch.Tensor | None = None
    old_values: torch.Tensor | None = None
    bootstrap_values: torch.Tensor | None = None
    primitive_rewards: torch.Tensor | None = None
    primitive_reward_mask: torch.Tensor | None = None
    actor_primitive_rewards: torch.Tensor | None = None
    actor_primitive_reward_mask: torch.Tensor | None = None
    duration_ticks: torch.Tensor | None = None
    actor_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Require aligned flattened trainer rows."""
        if self.old_logprobs.ndim != 1:
            raise ValueError(
                f"TrainingSignal old_logprobs must be [BT], got {tuple(self.old_logprobs.shape)}"
            )
        if self.is_padding.ndim != 1:
            raise ValueError(
                f"TrainingSignal is_padding must be [BT], got {tuple(self.is_padding.shape)}"
            )
        expected = self.old_logprobs.shape[0]
        if self.is_padding.shape[0] != expected:
            raise ValueError(
                f"TrainingSignal is_padding length {self.is_padding.shape[0]} "
                f"!= old_logprobs length {expected}"
            )
        if self.is_padding.dtype != torch.bool:
            raise ValueError(
                f"TrainingSignal is_padding dtype must be bool, got {self.is_padding.dtype}"
            )
        for field_name in (
            "advantages",
            "returns",
            "rewards",
            "old_values",
            "bootstrap_values",
        ):
            _validate_optional_signal_tensor(
                field_name, getattr(self, field_name), expected
            )
        for field_name in ("terminateds", "truncateds", "actor_valid"):
            tensor = getattr(self, field_name)
            _validate_optional_signal_tensor(field_name, tensor, expected)
            if tensor is not None and tensor.dtype != torch.bool:
                raise ValueError(
                    f"TrainingSignal {field_name} dtype must be bool, got {tensor.dtype}"
                )
        primitive_fields = (
            self.primitive_rewards,
            self.primitive_reward_mask,
            self.duration_ticks,
        )
        if any(value is not None for value in primitive_fields) and any(
            value is None for value in primitive_fields
        ):
            raise ValueError(
                "TrainingSignal primitive_rewards, primitive_reward_mask, and "
                "duration_ticks must be present together"
            )
        if self.primitive_rewards is not None:
            assert self.primitive_reward_mask is not None
            assert self.duration_ticks is not None
            if self.primitive_rewards.ndim != 2:
                raise ValueError(
                    "TrainingSignal primitive_rewards must be [BT, K], got "
                    f"{tuple(self.primitive_rewards.shape)}"
                )
            if self.primitive_reward_mask.shape != self.primitive_rewards.shape:
                raise ValueError(
                    "TrainingSignal primitive_reward_mask shape must match "
                    "primitive_rewards"
                )
            if self.primitive_rewards.shape[0] != expected:
                raise ValueError(
                    "TrainingSignal primitive_rewards leading dimension "
                    f"{self.primitive_rewards.shape[0]} != old_logprobs length {expected}"
                )
            if self.primitive_reward_mask.dtype != torch.bool:
                raise ValueError(
                    "TrainingSignal primitive_reward_mask dtype must be bool, got "
                    f"{self.primitive_reward_mask.dtype}"
                )
            if self.actor_primitive_rewards is not None:
                if self.actor_primitive_rewards.shape != self.primitive_rewards.shape:
                    raise ValueError(
                        "TrainingSignal actor_primitive_rewards shape must match "
                        "primitive_rewards"
                    )
                if not torch.isfinite(self.actor_primitive_rewards).all():
                    raise ValueError(
                        "TrainingSignal actor_primitive_rewards must be finite"
                    )
            if self.actor_primitive_reward_mask is not None:
                if (
                    self.actor_primitive_reward_mask.shape
                    != self.primitive_rewards.shape
                ):
                    raise ValueError(
                        "TrainingSignal actor_primitive_reward_mask shape must "
                        "match primitive_rewards"
                    )
                if self.actor_primitive_reward_mask.dtype != torch.bool:
                    raise ValueError(
                        "TrainingSignal actor_primitive_reward_mask dtype must be "
                        f"bool, got {self.actor_primitive_reward_mask.dtype}"
                    )
                if bool(
                    (
                        self.actor_primitive_reward_mask & ~self.primitive_reward_mask
                    ).any()
                ):
                    raise ValueError(
                        "TrainingSignal actor_primitive_reward_mask must be a "
                        "subset of primitive_reward_mask"
                    )
            _validate_optional_signal_tensor(
                "duration_ticks", self.duration_ticks, expected
            )
            if self.duration_ticks.dtype != torch.int64:
                raise ValueError(
                    "TrainingSignal duration_ticks dtype must be int64, got "
                    f"{self.duration_ticks.dtype}"
                )


@dataclass(frozen=True)
class RolloutArtifactIdentity:
    """Immutable identity of the exact disk bytes loaded into replay samples."""

    completion_path: str
    episode_file_sha256: str
    episode_file_size_bytes: int
    episode_manifest_sha256: str
    tensor_sidecar_filename: str
    tensor_sidecar_sha256: str
    tensor_sidecar_size_bytes: int
    session_uuid: str
    rollout_seed: int | None
    scene_id: str
    num_steps: int


@dataclass(frozen=True)
class TrainerReplayData:
    """One replay step extracted from a rollout artifact before collation.

    The packer flattens a rollout into single-step samples and pads each rollout
    to ``T_pack`` with ``is_padding`` rows; the trainer pools those steps across
    rollouts, shuffles, and minibatches over them.

    Shape notes:
        ``model_inputs`` leaves are ``[...]`` (a single step, no row dim).
        ``training_signal`` leaves are ``[1]``.
        ``weight_version`` is a scalar tensor.
    """

    model_inputs: dict[str, Any]
    training_signal: TrainingSignal
    rollout_id: str
    weight_version: torch.Tensor
    artifact_identity: RolloutArtifactIdentity | None = None


@dataclass(frozen=True)
class TrainerReplayDataBatch:
    """One minibatch of single-step replay samples stacked for the forward.

    Shape notes:
        ``model_inputs`` leaves are ``[B, ...]``.
        ``training_signal`` leaves are ``[B]``.
        ``rollout_ids`` has length ``B``.
        ``weight_versions`` is ``[B]``.
    """

    model_inputs: dict[str, Any]
    training_signal: TrainingSignal
    rollout_ids: tuple[str, ...]
    weight_versions: torch.Tensor

    @classmethod
    def stack(cls, samples: list[TrainerReplayData]) -> TrainerReplayDataBatch:
        """Stack single-step replay samples into a training minibatch.

        Each sample is one replay step (``[...]`` model-input leaves, ``[1]``
        signal leaves). They are stacked along a new row dim into ``[B, ...]``
        leaves; ``is_padding`` rows (padding steps the packer added to fill a
        rollout to ``T_pack``) are carried through unchanged.

        Args:
            samples: ``B`` single-step replay samples.
        """
        if not samples:
            raise ValueError(
                "TrainerReplayDataBatch.stack called with empty samples list"
            )

        old_logprobs = torch.cat(
            [sample.training_signal.old_logprobs for sample in samples], dim=0
        )
        is_padding = torch.cat(
            [sample.training_signal.is_padding for sample in samples], dim=0
        )
        model_inputs = {
            **stack_step_model_inputs([sample.model_inputs for sample in samples]),
            "return_log_prob": True,
        }
        return cls(
            model_inputs=model_inputs,
            training_signal=TrainingSignal(
                old_logprobs=old_logprobs,
                is_padding=is_padding,
                advantages=_cat_optional_signal(samples, "advantages"),
                returns=_cat_optional_signal(samples, "returns"),
                rewards=_cat_optional_signal(samples, "rewards"),
                terminateds=_cat_optional_signal(samples, "terminateds"),
                truncateds=_cat_optional_signal(samples, "truncateds"),
                old_values=_cat_optional_signal(samples, "old_values"),
                bootstrap_values=_cat_optional_signal(samples, "bootstrap_values"),
                primitive_rewards=_cat_optional_signal(samples, "primitive_rewards"),
                primitive_reward_mask=_cat_optional_signal(
                    samples, "primitive_reward_mask"
                ),
                actor_primitive_rewards=_cat_optional_signal(
                    samples, "actor_primitive_rewards"
                ),
                actor_primitive_reward_mask=_cat_optional_signal(
                    samples, "actor_primitive_reward_mask"
                ),
                duration_ticks=_cat_optional_signal(samples, "duration_ticks"),
                actor_valid=_cat_optional_signal(samples, "actor_valid"),
            ),
            rollout_ids=tuple(sample.rollout_id for sample in samples),
            weight_versions=torch.stack(
                [
                    sample.weight_version.to(dtype=torch.int64).reshape(())
                    for sample in samples
                ],
                dim=0,
            ),
        )


@dataclass(frozen=True)
class DataPackerConfig:
    """Run-config slice consumed by ``AlpagymDataPacker``.

    Bundles the run-level fields the packer needs so call sites pass one typed
    object instead of extracting scalars from ``RunConfig``.
    """

    expected_valid_steps: int

    def __post_init__(self) -> None:
        """Validate the trainer row budget."""
        if self.expected_valid_steps <= 0:
            raise ValueError(
                f"DataPackerConfig.expected_valid_steps must be positive, "
                f"got {self.expected_valid_steps}"
            )


def parse_policy_replay_data(raw: Mapping[str, Any]) -> PolicyReplayData:
    """Rehydrate one persisted replay envelope and tensor leaves."""
    replay_schema_version = int(raw["replay_schema_version"])
    payload_schema_version = int(raw["payload_schema_version"])
    if replay_schema_version != _REPLAY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported replay_schema_version {replay_schema_version}; "
            f"expected {_REPLAY_SCHEMA_VERSION}"
        )
    action_selection = ActionSelection(**dict(raw["action_selection"]))
    model_family = raw["model_family"]
    payload_schema = raw["payload_schema"]
    if not isinstance(payload_schema, str) or not payload_schema:
        raise ValueError("PolicyReplayData payload_schema must be a non-empty string")
    if payload_schema_version <= 0:
        raise ValueError("PolicyReplayData payload_schema_version must be positive")
    payload = raw["payload"]
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"PolicyReplayData payload must be a mapping, got {type(payload).__name__}"
        )
    old_logprob = raw["old_logprob"]
    old_logprob = (
        None
        if old_logprob is None
        else torch.as_tensor(old_logprob, dtype=torch.float32).reshape(())
    )
    return PolicyReplayData(
        replay_schema_version=replay_schema_version,
        payload_schema=payload_schema,
        payload_schema_version=payload_schema_version,
        model_family=model_family,
        action_selection=action_selection,
        old_logprob=old_logprob,
        payload=dict(payload),
    )


def _validate_optional_signal_tensor(
    field_name: str,
    tensor: torch.Tensor | None,
    expected: int,
) -> None:
    """Validate one optional flattened trainer signal."""
    if tensor is None:
        return
    if tensor.ndim != 1:
        raise ValueError(
            f"TrainingSignal {field_name} must be [BT], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != expected:
        raise ValueError(
            f"TrainingSignal {field_name} length {tensor.shape[0]} "
            f"!= old_logprobs length {expected}"
        )


def _cat_optional_signal(
    samples: list[TrainerReplayData],
    field_name: str,
) -> torch.Tensor | None:
    """Concatenate an optional signal field, rejecting mixed presence."""
    values = [getattr(sample.training_signal, field_name) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"TrainingSignal {field_name} mixes present and missing rows")
    return torch.cat(values, dim=0)


def require_payload_keys(
    model_family: str,
    payload: Mapping[str, Any],
    required_keys: tuple[str, ...],
) -> None:
    """Reject a replay payload missing family-required schema fields.

    Raises:
        ValueError: one or more ``required_keys`` are absent from ``payload``.
    """
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(
            f"{model_family} replay payload missing required fields: {', '.join(missing)}"
        )


def stack_step_model_inputs(step_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack per-step model-input dictionaries along a new leading row dim."""
    keys = set(step_inputs[0])
    if any(set(item) != keys for item in step_inputs):
        raise ValueError("Replay model input keys differ across steps")
    return {
        key: _stack_values([item[key] for item in step_inputs], key) for key in keys
    }


def clone_model_inputs(model_inputs: Any) -> Any:
    """Deep-copy a valid step's model-input tree for a padding step.

    A padding step reuses a real (valid) step's inputs rather than zero-filled
    ones so it forwards to a finite log-prob and stays in lockstep with the real
    rows; its zero advantage and the padding mask then neutralize it in the loss
    and diagnostics. Zero-filling instead would make the SDE log-prob divide by a
    zeroed timestep/noise and return non-finite values (which ``advantage * inf``
    turns into NaN gradients). Tensor leaves are cloned; non-tensor leaves (e.g.
    ``cfm_method``) pass through.
    """
    if isinstance(model_inputs, torch.Tensor):
        return model_inputs.clone()
    if isinstance(model_inputs, dict):
        return {key: clone_model_inputs(value) for key, value in model_inputs.items()}
    return model_inputs


def _dataclass_to_plain(value: Any) -> Any:
    """Convert dataclasses to dicts while preserving tensor leaves."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _dataclass_to_plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _dataclass_to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_dataclass_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_to_plain(item) for item in value]
    return value


def _stack_values(values: list[Any], key: str) -> Any:
    """Stack one nested field across steps."""
    if any(value is None for value in values):
        if all(value is None for value in values):
            return None
        raise ValueError(f"Optional model input {key} mixes None and tensor values")
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, dict):
        dict_keys = set(first)
        if any(
            not isinstance(value, dict) or set(value) != dict_keys for value in values
        ):
            raise ValueError(f"Nested model input keys differ for {key}")
        return {
            nested_key: _stack_values(
                [value[nested_key] for value in values], nested_key
            )
            for nested_key in dict_keys
        }
    if all(value == first for value in values):
        return first
    raise ValueError(f"Unsupported or varying non-tensor model input for {key}")
