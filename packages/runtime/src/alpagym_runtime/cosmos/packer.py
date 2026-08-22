# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert AlpaGym rollout artifacts into trainer replay batches, and own this
process's rollout-transport endpoint.

This packer is the single owner of the role's transport endpoint; cosmos-rl
calls the per-role hook, so the role is implied by which hook runs:

- Rollout: holds an :class:`EpisodeWriter`. ``get_rollout_output`` egresses each
  in-memory ``EpisodeOutput`` to the writer and returns the opaque handles. cosmos
  invokes it after the reward dispatcher and DAPO ``dynamic_sampling``, so reward
  reads off the in-memory completion and DAPO-dropped payloads never reach the sender.
- Policy: reads each handle back. NCCL resolves it inline through the
  :class:`NcclDataPackerMixin`; disk reads the JSON artifact.

``get_policy_input`` rejects artifacts that exceed the fixed ``T_pack`` row
budget, pads shorter rollouts, and emits the rollout-time old logprobs the
trainer needs to recompute the policy ratio. When a replay payload carries a
``payload["transition"]`` block, the packer also emits environment rewards,
terminal flags, and rollout-time value estimates for trainer-side GAE.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import redis
import torch
from alpagym_host.config import CosmosRLMode, RunConfig, TransportKind
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from torch.distributed import TCPStore

from alpagym_runtime.perf.instrument.scope import measure_perf, timed_scope
from alpagym_runtime.replay import (
    DataPackerConfig,
    PolicyReplayData,
    TrainerReplayData,
    TrainerReplayDataBatch,
    TrainingSignal,
    clone_model_inputs,
)
from alpagym_runtime.transport import EpisodeWriter
from alpagym_runtime.transport.disk import DiskEpisodeWriter, read_episode_json
from alpagym_runtime.types import EpisodeOutput

if TYPE_CHECKING:
    # Annotation-only: the NCCL receiver is imported lazily inside _build_nccl_receiver
    # so disk-only and colocated runs never import the cosmos NCCL payload-transport chain.
    from alpagym_runtime.transport.nccl.receiver import NcclReceiver

logger = logging.getLogger(__name__)

_TRANSITION_PAYLOAD_KEY = "transition"
_FLOAT_TRANSITION_SIGNAL_ALIASES = {
    "rewards": ("reward", "rewards"),
    "old_values": ("old_value", "value", "old_values"),
    "bootstrap_values": ("bootstrap_value", "bootstrap_values"),
}
_BOOL_TRANSITION_SIGNAL_ALIASES = {
    "terminateds": ("terminated", "terminateds"),
    "truncateds": ("truncated", "truncateds"),
    "actor_valid": ("actor_valid",),
}
_PRIMITIVE_REWARD_ALIASES = ("primitive_rewards", "controller_tick_rewards")
_PRIMITIVE_REWARD_MASK_ALIASES = (
    "primitive_reward_mask",
    "controller_tick_reward_mask",
)
_DURATION_TICK_ALIASES = ("duration_ticks", "controller_ticks")


class AlpagymDataPacker(DataPacker):
    """Own the rollout-transport endpoint and collate replay batches.

    The rollout worker holds a ``writer`` and egresses through
    ``get_rollout_output``; the disaggregated trainer holds no writer and reads
    each handle back in ``get_policy_input`` (disk reads JSON; the NCCL subclass
    resolves over the receiver). In colocated mode the single process does both.
    The controller holds neither.
    """

    def __init__(
        self,
        config: DataPackerConfig,
        build_model_inputs: Callable[
            [PolicyReplayData], tuple[dict[str, Any], torch.Tensor]
        ],
        writer: EpisodeWriter | None = None,
        collate_samples: Callable[[list[TrainerReplayData]], TrainerReplayDataBatch]
        | None = None,
    ) -> None:
        """Store replay collation settings and the optional rollout writer.

        Args:
            config: Run-config slice carrying ``expected_valid_steps`` (the
                fixed ``T_pack`` rows per rollout artifact).
            build_model_inputs: Per-family callable that converts one replay
                payload into model-forward kwargs plus the rollout-time old
                logprob; supplied by the selected ``PolicyBundle`` so this
                packer stays policy-agnostic.
            writer: Rollout-side egress, or ``None`` on the trainer/controller.
            collate_samples: Optional policy-owned minibatch collator. ``None``
                retains rectangular ``torch.stack`` collation.
        """
        super().__init__()
        self._expected_valid_steps = config.expected_valid_steps
        self._build_model_inputs = build_model_inputs
        self._writer = writer
        self._collate_samples = collate_samples
        # cosmos's CommMixin assigns this and then calls post_redis_injection();
        # the packer only forwards it to the writer's cleanup subscriber.
        self.redis_client: redis.Redis | None = None

    def get_rollout_input(self, item: int) -> int:
        """Return the prompt index unchanged for the rollout worker."""
        return item

    def get_rollout_output(
        self,
        completions: list[Any],
        completed_conversations: list[Any],
        logprobs: list[Any],
        token_ids: list[Any],
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any], dict[str, Any]]:
        """Egress each in-memory ``EpisodeOutput`` to the writer, returning handles.

        Exactly-once egress assumes cosmos calls this after the reward dispatcher and
        DAPO ``dynamic_sampling``, so ``completions`` are the live in-memory episodes
        that survived sampling and DAPO-dropped payloads never reach the writer. That
        ordering is a cosmos-side contract this packer depends on, not enforced here.
        ``train.non_text=True`` is required: it keeps cosmos from serializing the
        GPU-tensor completions.
        """
        if self._writer is None:
            raise RuntimeError(
                "get_rollout_output requires a rollout writer; this packer has none"
            )
        if not self.config.train.non_text:
            raise RuntimeError(
                "AlpaGym rollout egress carries in-memory EpisodeOutputs to "
                "get_rollout_output, which requires train.non_text=True"
            )
        handles: list[str] = []
        for completion in completions:
            if not isinstance(completion, EpisodeOutput):
                raise TypeError(
                    f"get_rollout_output expected EpisodeOutput completions, got {type(completion)}"
                )
            handles.append(self._writer.write(completion))
        return handles, completed_conversations, logprobs, token_ids, kwargs

    def post_redis_injection(self) -> None:
        """Start the rollout writer's discard-cleanup subscriber once Redis is injected.

        Only the rollout writer needs it: the NCCL writer subscribes to the
        controller's discard channel, the disk writer no-ops. The trainer and
        controller hold no writer.
        """
        if self._writer is not None:
            self._writer.start_cleanup(self.redis_client)

    def flush_pending_sends(self) -> None:
        """Block in-flight rollout egress before an R2R weight broadcast.

        cosmos opt-in hook called before each rollout-to-rollout (R2R) weight
        broadcast (not before policy-to-rollout sync). No-op when this packer
        holds no writer (trainer/controller) or the writer has no in-flight
        egress to wait on (disk).
        """
        if self._writer is not None:
            self._writer.flush_pending_sends()

    def close(self) -> None:
        """Release the rollout writer's resources (background threads, NCCL comms)."""
        if self._writer is not None:
            self._writer.close()

    @measure_perf("trainer/artifact_load", category="orchestration", cpu_snapshot=True)
    def get_policy_input(
        self,
        sample: Any,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        **kwargs: Any,
    ) -> list[TrainerReplayData]:
        """Extract one rollout result into a fixed-length list of per-step samples.

        The result is an in-memory ``EpisodeOutput`` on NCCL (already resolved)
        or a JSON artifact handle read here on disk.

        Each ``PolicyOutput`` must carry replay data, which the packer converts
        into one single-step ``TrainerReplayData`` (``[...]`` model-input leaves,
        ``[1]`` signal leaves). The list is padded to ``expected_valid_steps``
        with ``is_padding`` rows (cloning a valid step's inputs, zero advantage)
        so every rollout contributes the same step count to the trainer's
        flattened pool while the padding rows still forward to finite log-probs.
        """
        del sample, n_ignore_prefix_tokens, kwargs
        # nccl handles arrive already resolved inline to an EpisodeOutput (see
        # NcclDataPackerMixin); disk handles are JSON artifact paths read here.
        if isinstance(rollout_output, EpisodeOutput):
            episode = rollout_output
        else:
            with timed_scope("trainer/artifact_load/disk_read", category="io"):
                episode = read_episode_json(rollout_output)

        replay_rows: list[PolicyReplayData] = []
        for output in episode.policy_outputs:
            if output.replay_data is None:
                raise ValueError("Policy output is missing replay_data")
            replay_rows.append(output.replay_data)

        if len(replay_rows) > self._expected_valid_steps:
            raise ValueError(
                "AlpaGym replay session="
                f"{episode.session_uuid} produced {len(replay_rows)} policy outputs, "
                f"exceeding expected_valid_steps={self._expected_valid_steps}; "
                "refusing to drop replay rows"
            )
        if not replay_rows:
            raise ValueError(
                "Rollout produced no replay rows. Check rollout-side trainer artifact emission."
            )

        step_samples: list[TrainerReplayData] = []
        for replay_data in replay_rows:
            model_inputs, old_logprob = self._build_model_inputs(replay_data)
            weight_version = _transition_weight_version(replay_data)
            step_samples.append(
                TrainerReplayData(
                    model_inputs=model_inputs,
                    training_signal=TrainingSignal(
                        old_logprobs=old_logprob.reshape(1).to(dtype=torch.float32),
                        is_padding=torch.zeros(1, dtype=torch.bool),
                        **_extract_transition_training_signal(replay_data),
                    ),
                    rollout_id=episode.session_uuid,
                    weight_version=torch.tensor(weight_version, dtype=torch.int64),
                )
            )

        # Validate model-input key consistency at unpack so a malformed rollout
        # fails deterministically here, not later when a shuffled minibatch
        # happens to stack mismatched steps.
        keys = set(step_samples[0].model_inputs)
        if any(set(step.model_inputs) != keys for step in step_samples):
            raise ValueError(
                f"Replay step model-input keys differ within rollout session={episode.session_uuid}"
            )

        valid_steps = len(step_samples)
        template_model_inputs = step_samples[0].model_inputs
        while len(step_samples) < self._expected_valid_steps:
            step_samples.append(
                TrainerReplayData(
                    model_inputs=clone_model_inputs(template_model_inputs),
                    training_signal=TrainingSignal(
                        # Padding clones the first row's model inputs.  Some
                        # policies carry a token-level behavior trace in those
                        # inputs, so keep its scalar audit log-prob consistent
                        # as well.  The padding mask still makes this row a
                        # strict no-op for every loss and return calculation.
                        old_logprobs=(
                            step_samples[0].training_signal.old_logprobs.clone()
                        ),
                        is_padding=torch.ones(1, dtype=torch.bool),
                        **_zero_padding_transition_signal(
                            step_samples[0].training_signal
                        ),
                    ),
                    rollout_id=episode.session_uuid,
                    weight_version=step_samples[0].weight_version.clone(),
                )
            )

        logger.info(
            "Packed AlpaGym replay artifact session=%s total_steps=%d "
            "policy_outputs=%d valid_steps=%d padded_steps=%d reward=%.6f",
            episode.session_uuid,
            episode.num_steps,
            len(episode.policy_outputs),
            valid_steps,
            len(step_samples) - valid_steps,
            float(episode.reward.total) if episode.reward else 0.0,
        )
        return step_samples

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        """Return a placeholder sequence length unused by the AlpaGym trainer."""
        del processed_samples
        return 1

    def policy_collate_fn(
        self,
        processed_samples: list[TrainerReplayData],
        computed_max_len: int = 1,
    ) -> TrainerReplayDataBatch:
        """Stack one minibatch of single-step samples into a rectangular batch.

        The trainer has already flattened all rollouts into per-step samples and
        sliced a shuffled minibatch; this stacks those single-step samples along
        a new row dimension into the ``[B, ...]`` model inputs the forward
        consumes, where ``B`` is the minibatch step count.
        """
        del computed_max_len
        if self._collate_samples is None:
            batch = TrainerReplayDataBatch.stack(processed_samples)
        else:
            batch = self._collate_samples(processed_samples)
        logger.info(
            "Collated AlpaGym replay minibatch steps=%d padding_rows=%d",
            len(processed_samples),
            int(batch.training_signal.is_padding.sum().item()),
        )
        return batch


def _extract_transition_training_signal(
    replay_data: PolicyReplayData,
) -> dict[str, torch.Tensor]:
    """Extract optional transition facts from one replay payload.

    Policy bundles own the observation/action dialect in ``payload``. This
    runtime-level convention carries rollout facts needed by actor-critic
    trainers; algorithm-derived PPO advantages and returns are computed later by
    ``AlpagymPPOTrainer``.
    """
    transition_payload = replay_data.payload.get(_TRANSITION_PAYLOAD_KEY)
    if transition_payload is None:
        return {}
    if not isinstance(transition_payload, Mapping):
        raise TypeError("PolicyReplayData payload['transition'] must be a mapping")

    signals: dict[str, torch.Tensor] = {}
    actor_valid_raw: object | None = None
    duration_ticks_raw: object | None = None
    for field_name, aliases in _FLOAT_TRANSITION_SIGNAL_ALIASES.items():
        for alias in aliases:
            if alias in transition_payload:
                value = transition_payload[alias]
                if value is not None:
                    signals[field_name] = torch.as_tensor(
                        value, dtype=torch.float32
                    ).reshape(1)
                break
    for field_name, aliases in _BOOL_TRANSITION_SIGNAL_ALIASES.items():
        for alias in aliases:
            if alias in transition_payload:
                value = transition_payload[alias]
                if value is not None:
                    if field_name == "actor_valid":
                        actor_valid_raw = value
                    signals[field_name] = torch.as_tensor(
                        value, dtype=torch.bool
                    ).reshape(1)
                break
    for alias in _PRIMITIVE_REWARD_ALIASES:
        if alias in transition_payload:
            value = transition_payload[alias]
            if value is not None:
                signals["primitive_rewards"] = torch.as_tensor(
                    value, dtype=torch.float32
                ).reshape(1, -1)
            break
    for alias in _PRIMITIVE_REWARD_MASK_ALIASES:
        if alias in transition_payload:
            value = transition_payload[alias]
            if value is not None:
                signals["primitive_reward_mask"] = torch.as_tensor(
                    value, dtype=torch.bool
                ).reshape(1, -1)
            break
    for alias in _DURATION_TICK_ALIASES:
        if alias in transition_payload:
            value = transition_payload[alias]
            if value is not None:
                duration_ticks_raw = value
                signals["duration_ticks"] = torch.as_tensor(
                    value, dtype=torch.int64
                ).reshape(1)
            break
    if replay_data.model_family == "g1_vla" and "duration_ticks" in signals:
        if "actor_valid" not in signals:
            raise ValueError(
                "g1_vla motion-reference replay is missing transition.actor_valid"
            )
        if not isinstance(actor_valid_raw, bool):
            raise ValueError(
                "g1_vla motion-reference replay requires boolean transition.actor_valid"
            )
        if isinstance(duration_ticks_raw, bool) or not isinstance(
            duration_ticks_raw, int
        ):
            raise ValueError(
                "g1_vla motion-reference replay requires integer "
                "transition.duration_ticks"
            )
        owning_ticks_raw = transition_payload.get("owning_reference_executed_ticks")
        if isinstance(owning_ticks_raw, bool) or not isinstance(owning_ticks_raw, int):
            raise ValueError(
                "g1_vla motion-reference replay requires integer "
                "transition.owning_reference_executed_ticks"
            )
        duration = int(signals["duration_ticks"].item())
        if not 0 <= owning_ticks_raw <= duration:
            raise ValueError(
                "g1_vla owning_reference_executed_ticks must be within "
                f"[0, duration_ticks], got {owning_ticks_raw} and {duration}"
            )
        actor_valid = bool(signals["actor_valid"].item())
        if actor_valid != (owning_ticks_raw > 0):
            raise ValueError(
                "g1_vla transition.actor_valid must equal "
                "(owning_reference_executed_ticks > 0)"
            )
    return signals


def _transition_weight_version(replay_data: PolicyReplayData) -> int:
    """Return the exact behavior version carried by an actor-critic transition."""
    transition = replay_data.payload.get(_TRANSITION_PAYLOAD_KEY)
    if transition is None:
        return 0
    if not isinstance(transition, Mapping):
        raise TypeError("PolicyReplayData payload['transition'] must be a mapping")
    if "behavior_policy_version" not in transition:
        raise ValueError("actor-critic transition is missing behavior_policy_version")
    value = transition["behavior_policy_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("behavior_policy_version must be a non-negative integer")
    return value


def _zero_padding_transition_signal(
    template: TrainingSignal,
) -> dict[str, torch.Tensor]:
    """Return neutral transition signals for a padding row when transition data is active."""
    signals: dict[str, torch.Tensor] = {}
    for field_name in (
        *_FLOAT_TRANSITION_SIGNAL_ALIASES,
        *_BOOL_TRANSITION_SIGNAL_ALIASES,
    ):
        value = getattr(template, field_name)
        if value is not None:
            signals[field_name] = torch.zeros_like(value)
    for field_name in (
        "primitive_rewards",
        "primitive_reward_mask",
        "duration_ticks",
    ):
        value = getattr(template, field_name)
        if value is not None:
            signals[field_name] = torch.zeros_like(value)
    return signals


def build_alpagym_data_packer(
    run_config: RunConfig,
    cosmos_role: str | None,
    build_model_inputs: Callable[
        [PolicyReplayData], tuple[dict[str, Any], torch.Tensor]
    ],
    collate_samples: Callable[[list[TrainerReplayData]], TrainerReplayDataBatch]
    | None = None,
) -> AlpagymDataPacker:
    """Construct the role's packer with its transport endpoint wired in.

    cosmos builds the packer on every worker (and the controller, for its
    ``from_pretrained`` probes). The role selects the endpoint: a rollout writer
    on the rollout worker, an NCCL receiver on the NCCL trainer, neither on a
    disaggregated disk trainer or the controller. In colocated mode the single
    ``Policy`` process also runs the rollout worker (which shares this packer),
    so it gets a disk writer to egress through; NCCL colocated is rejected at
    host preflight, so the colocated writer is always disk.
    """
    config = DataPackerConfig(expected_valid_steps=run_config.expected_valid_steps)
    is_nccl = run_config.transport.kind == TransportKind.nccl
    if cosmos_role == "Rollout":
        return AlpagymDataPacker(
            config,
            build_model_inputs,
            writer=_build_episode_writer(run_config, is_nccl=is_nccl),
            collate_samples=collate_samples,
        )
    elif cosmos_role == "Policy":
        if is_nccl:
            from alpagym_runtime.transport.nccl.endpoints import NcclAlpagymDataPacker

            store, receiver, target_device = _build_nccl_receiver(run_config)
            return NcclAlpagymDataPacker(
                config,
                build_model_inputs,
                store=store,
                receiver=receiver,
                target_device=target_device,
                collate_samples=collate_samples,
            )
        # Disk Policy reads JSON artifacts back by handle. In colocated mode this
        # one process also runs the rollout worker, which shares this packer and
        # egresses through it, so it needs a disk writer too; the disaggregated
        # disk trainer is read-only.
        writer = (
            _build_episode_writer(run_config, is_nccl=False)
            if run_config.cosmos.mode == CosmosRLMode.colocated
            else None
        )
        return AlpagymDataPacker(
            config,
            build_model_inputs,
            writer=writer,
            collate_samples=collate_samples,
        )
    else:
        # The controller owns no data plane.
        return AlpagymDataPacker(
            config,
            build_model_inputs,
            collate_samples=collate_samples,
        )


def _build_episode_writer(run_config: RunConfig, is_nccl: bool) -> EpisodeWriter:
    """Build the rollout worker's egress writer (NCCL sender or JSON writer)."""
    if not is_nccl:
        return DiskEpisodeWriter(Path(run_config.artifact_paths.artifacts_dir))

    from alpagym_host.endpoint_registry import FileTopologyRegistry
    from cosmos_rl.utils.pynccl import (
        create_nccl_comm,
        create_nccl_uid,
        nccl_abort,
        nccl_send,
    )

    from alpagym_runtime.transport.nccl.comm_init import CommInitConfig
    from alpagym_runtime.transport.nccl.endpoints import NcclEpisodeWriter
    from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous
    from alpagym_runtime.transport.nccl.sender import (
        NcclSender,
        NcclSenderConfig,
        assign_rollout_idx,
    )

    nccl_timeout_seconds = float(run_config.transport.nccl_env["NCCL_TIMEOUT"])
    host, port = FileTopologyRegistry(
        run_config.artifact_paths.topology_registry_dir
    ).read_nccl_master(nccl_timeout_seconds)
    _set_cuda_device_from_local_rank()
    store = TCPStore(
        host_name=host,
        port=port,
        world_size=1,
        is_master=False,
        timeout=timedelta(seconds=nccl_timeout_seconds),
    )
    comm_init_config = CommInitConfig(
        barrier_wait_timeout_seconds=nccl_timeout_seconds,
        communicator_timeout_ms=int(nccl_timeout_seconds * 1000),
    )
    rendezvous = AckRendezvous(ack_timeout_seconds=nccl_timeout_seconds)
    experiment_name = run_config.cosmos.logging.experiment_name
    # Match cosmos's own SLURM_JOB_ID fallback ("test", fixed at the pin): the cosmos
    # controller publishes the cleanup channel under this job_id, so the sender's
    # subscriber must derive the same one or local (non-Slurm) cleanup goes to a dead channel.
    job_id = os.environ.get("SLURM_JOB_ID", "test")
    rollout_idx = assign_rollout_idx(
        experiment_name=experiment_name,
        job_id=job_id,
        num_rollout_replicas=run_config.cosmos.launch.rollout_replicas,
        store=store,
    )
    sender = NcclSender(
        experiment_name=experiment_name,
        job_id=job_id,
        rollout_idx=rollout_idx,
        num_policy_replicas=run_config.cosmos.launch.policy_replicas,
        dp_shard_size=run_config.cosmos.policy.parallelism.dp_shard_size,
        store=store,
        rendezvous=rendezvous,
        create_nccl_uid=create_nccl_uid,
        create_nccl_comm=create_nccl_comm,
        nccl_send=nccl_send,
        nccl_abort=nccl_abort,
        config=NcclSenderConfig(comm_init=comm_init_config),
    )
    sender.setup()
    logger.info(
        "NCCL writer wired (Rollout, rollout_idx=%s, comm_idx=%s)",
        rollout_idx,
        sender.comm_idx,
    )
    return NcclEpisodeWriter(
        store=store,
        sender=sender,
        experiment_name=experiment_name,
        job_id=job_id,
        flush_timeout_seconds=nccl_timeout_seconds,
    )


def _build_nccl_receiver(
    run_config: RunConfig,
) -> tuple[TCPStore, NcclReceiver, torch.device]:
    """Build the policy worker's NCCL receiver and its TCPStore connection."""
    from alpagym_host.endpoint_registry import FileTopologyRegistry
    from cosmos_rl.utils.pynccl import (
        create_nccl_comm,
        create_nccl_uid,
        nccl_abort,
        nccl_recv,
    )

    from alpagym_runtime.transport.nccl.comm_init import CommInitConfig
    from alpagym_runtime.transport.nccl.receiver import NcclReceiver, NcclReceiverConfig
    from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous

    nccl_timeout_seconds = float(run_config.transport.nccl_env["NCCL_TIMEOUT"])
    host, port = FileTopologyRegistry(
        run_config.artifact_paths.topology_registry_dir
    ).read_nccl_master(nccl_timeout_seconds)
    _set_cuda_device_from_local_rank()
    store = TCPStore(
        host_name=host,
        port=port,
        world_size=1,
        is_master=False,
        timeout=timedelta(seconds=nccl_timeout_seconds),
    )
    comm_init_config = CommInitConfig(
        barrier_wait_timeout_seconds=nccl_timeout_seconds,
        communicator_timeout_ms=int(nccl_timeout_seconds * 1000),
    )
    rendezvous = AckRendezvous(ack_timeout_seconds=nccl_timeout_seconds)
    receiver = NcclReceiver(
        experiment_name=run_config.cosmos.logging.experiment_name,
        # Match cosmos's SLURM_JOB_ID fallback ("test") so the receiver shares the
        # controller's job_id; see the writer-side note in _build_episode_writer.
        job_id=os.environ.get("SLURM_JOB_ID", "test"),
        num_policy_replicas=run_config.cosmos.launch.policy_replicas,
        dp_shard_size=run_config.cosmos.policy.parallelism.dp_shard_size,
        num_rollout_replicas=run_config.cosmos.launch.rollout_replicas,
        store=store,
        rendezvous=rendezvous,
        create_nccl_uid=create_nccl_uid,
        create_nccl_comm=create_nccl_comm,
        nccl_recv=nccl_recv,
        nccl_abort=nccl_abort,
        config=NcclReceiverConfig(
            recv_timeout_seconds=nccl_timeout_seconds,
            comm_init=comm_init_config,
        ),
    )
    receiver.setup()
    logger.info("NCCL reader wired (Policy, rollout_comms=%s)", receiver.rollout_comms)
    return store, receiver, torch.device(run_config.transport.nccl_read_device)


def _set_cuda_device_from_local_rank() -> None:
    """Bind pynccl setup to the local rank's CUDA device when CUDA is present."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
