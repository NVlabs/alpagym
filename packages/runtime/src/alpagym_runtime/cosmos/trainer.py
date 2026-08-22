# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPO, actor-critic PPO, and Flow-PPO adapters for closed-loop training.

The trainer is policy-agnostic: per-policy tokenizer resolution and data
packer construction are looked up via the
``alpagym_runtime.policies.registry`` bundle for the configured
policy kind string.  Cosmos currently exposes their shared scheduling fields
through ``GrpoConfig``; the selected ``trainer_type`` determines the objective.
"""

import copy
import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from alpagym_host.config import RunConfig, load_run_config
from cosmos_rl.dispatcher.data import schema as _rollout_schema
from cosmos_rl.policy import config as _cosmos_config
from cosmos_rl.policy.trainer import base as _trainer_base
from cosmos_rl.policy.trainer.llm_trainer import grpo_trainer as _grpo_trainer
from cosmos_rl.utils import distributed as dist_util, parallelism as _parallelism

from alpagym_runtime.cosmos.replay_objective import (
    assert_replay_shapes,
    compute_flow_ppo_surrogate,
    compute_kl_penalty,
    compute_ppo_surrogate,
    compute_value_loss,
)
from alpagym_runtime.cosmos.rollout_filter import filter_trainable_rollouts
from alpagym_runtime.perf.instrument.lifecycle import initialize_perf
from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.scope import measure_perf
from alpagym_runtime.policies.registry import get_policy_bundle
from alpagym_runtime.replay import TrainingSignal
from alpagym_runtime.tensor_utils import to_device_recursive

logger = logging.getLogger(__name__)


def _fixed_reference_reset_interval(value: int | None) -> int:
    """Normalize Cosmos's fixed-anchor spelling and reject moving KL anchors."""
    interval = 0 if value is None else int(value)
    if interval != 0:
        raise ValueError(
            "AlpaGym replay trainers require reference_reset_interval=0 so "
            "the KL anchor remains restart-stable"
        )
    return interval


def _load_run_config(config: _cosmos_config.Config) -> RunConfig:
    """Return the resolved AlpaGym run config referenced by ``config``.

    The cosmos ``Config`` carries the resolved AlpaGym config path under
    ``custom.resolved_config_path``. Raises when the path is absent - every
    production cosmos invocation passes it via ``--config``.
    """
    custom = getattr(config, "custom", None) or {}
    resolved_config_path = (
        custom.get("resolved_config_path") if hasattr(custom, "get") else None
    )
    if not resolved_config_path:
        raise ValueError(
            "Cosmos config missing custom.resolved_config_path; cannot load "
            "AlpaGym run config. Production cosmos invocations set this via --config."
        )
    return load_run_config(resolved_config_path)


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_grpo")
class AlpagymGRPOTrainer(_grpo_trainer.GRPOTrainer):
    """GRPO trainer that replaces Cosmos's text-token loss with AlpaGym replay training.

    Consumes AlpaGym rollout artifact completions, recomputes logprobs for each
    recorded selected action, and applies the PPO/GRPO replay objective. The
    step is the minibatching unit: all rollouts are flattened into one pool of
    per-step replay samples, shuffled, then split into minibatches.

    Forward contract for ``self.model``:

        model(
            **model_inputs,                # whatever the policy's data packer collated
            teacher_model: nn.Module | None,
        ) -> {
            "log_probs": Tensor[R],        # current logprob for recorded action
            "kl_div":    Tensor[R] | None,
        }

    ``kl_div`` (when present) is one scalar per replay row; the trainer excludes
    padded rows before reducing KL. Additional return keys are allowed but
    ignored by the trainer.

    ``PolicyOutput.replay_data`` must carry a ``PolicyReplayData`` envelope.
    The packer raises before collation if the payload family, old rollout
    logprob, selected action data, or required trace fields are missing.
    """

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Cosmos-RL config; reads `train.train_policy.*` hyperparams
                and `policy.model_name_or_path` for the policy bundle.
            parallel_dims: Cosmos-RL parallelism description.
            **kwargs: Forwarded to `GRPOTrainer.__init__`
                (`train_stream`, `data_packer`, `val_data_packer`, ...).
        """
        # Loaded once and reused for perf init and the policy-bundle lookup below.
        # Production cosmos invocations always set `custom.resolved_config_path` via
        # `--config`.
        run_config = _load_run_config(config)
        initialize_perf(run_config)
        # Cosmos's super-init resolves a tokenizer from
        # ``config.policy.model_name_or_path`` and calls ``ModelRegistry.build_model``.
        # Policy bundles whose path lacks tokenizer files need that resolution
        # against a per-bundle location, and need their model
        # registered with cosmos beforehand; the registered bundle's
        # ``setup_tokenizer`` handles both.
        bundle = get_policy_bundle(run_config.policy.model.kind)
        self._policy_bundle = bundle
        bundle_tokenizer = bundle.setup_tokenizer(config)
        if bundle_tokenizer is not None:
            self.tokenizer = bundle_tokenizer

        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)

        grpo_config = config.train.train_policy
        if not isinstance(grpo_config, _cosmos_config.GrpoConfig):
            raise TypeError("config.train.train_policy must be GrpoConfig.")
        self._grpo_ratio_clip_low: float = float(grpo_config.epsilon_low)
        self._grpo_ratio_clip_high: float = float(grpo_config.epsilon_high)
        self._grpo_optimization_iterations: int = int(grpo_config.mu_iterations)
        self._mini_batch: int = int(grpo_config.mini_batch)
        self._kl_beta: float = float(grpo_config.kl_beta)
        self._allowed_outdated_steps: int = int(grpo_config.allowed_outdated_steps)
        self._on_policy: bool = bool(grpo_config.on_policy)
        # Cosmos optionally allows reference_reset_interval=None to mean "never";
        # normalize that to the restart-stable fixed-anchor value.
        self._reference_reset_interval = _fixed_reference_reset_interval(
            grpo_config.reference_reset_interval
        )
        # GRPO groups one prompt's rollouts together; that count lives on the
        # rollout config in Cosmos.
        self._group_size: int = int(config.rollout.n_generation)

        self._reference_model: Any = None
        record_perf_marker("trainer/ready", cpu_snapshot=True, gpu_snapshot=True)

    @measure_perf(
        "trainer/step",
        category="compute_gpu_wall",
        cpu_snapshot=True,
        gpu_snapshot=True,
    )
    def step_training(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        rollouts: list[_rollout_schema.Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run one GRPO step over the rollouts Cosmos provides.

        Filters stale rollouts, builds per-step samples via the data packer,
        runs ``grpo_optimization_iterations × num_mini_batches`` PPO updates, and returns
        the metrics dict Cosmos reports.

        Args:
            rollouts: Cosmos-RL rollouts; each carries ``prompt``,
                ``completion`` (artifact path), ``advantage``, and
                ``weight_version``.
            current_step: Current training step index.
            total_steps: Total configured training steps.
            remain_samples_num: Remaining samples reported by Cosmos-RL; forwarded
                to the checkpoint manager so resume restores the same data pointer.
            inter_policy_nccl: NCCL communicator across DP replicas.
            is_master_replica: Whether this is the master policy replica; only
                the master writes checkpoints.
            do_save_checkpoint: Whether Cosmos-RL requested checkpointing this step.

        Returns:
            Dict of training metrics for Cosmos to log.
        """
        del kwargs
        logger.info(
            "AlpaGym trainer step start current_step=%d total_steps=%d received_rollouts=%d "
            "group_size=%d mini_batch=%d grpo_optimization_iterations=%d "
            "is_master_replica=%s do_save_checkpoint=%s",
            current_step,
            total_steps,
            len(rollouts),
            self._group_size,
            self._mini_batch,
            self._grpo_optimization_iterations,
            is_master_replica,
            do_save_checkpoint,
        )
        rollouts = filter_trainable_rollouts(
            rollouts,
            current_step=current_step,
            train_batch_per_replica=int(self.config.train.train_batch_per_replica),
            allowed_outdated_steps=self._allowed_outdated_steps,
        )
        if self._on_policy:
            expected_behavior_version = max(current_step - 1, 0)
            kept_versions = [int(rollout.weight_version) for rollout in rollouts]
            if any(version != expected_behavior_version for version in kept_versions):
                raise ValueError(
                    "On-policy AlpaGym training requires the exact behavior "
                    f"version {expected_behavior_version} for optimizer update "
                    f"{current_step}, got {kept_versions}"
                )
        samples, advantages = self._prepare_training_data(rollouts)
        if not samples:
            raise ValueError(
                "AlpaGym trainer step has no trainable samples after filtering: "
                f"current_step={current_step}"
            )

        pre_update_metrics = self._pre_update_diagnostics(samples)
        self._validate_update_diagnostics(
            pre_update_metrics,
            phase="pre_update",
        )
        (
            total_loss,
            total_kl,
            num_batches,
            ratio_max,
            ratio_min,
            clip_fraction_sum,
            grad_norm_sum,
        ) = self._run_training_loop(samples, advantages, inter_policy_nccl)
        # PPO overrides this hook with a no-grad replay pass against the final
        # post-update policy.  The minibatch diagnostics above are measured
        # before each individual optimizer step, so they cannot describe one
        # coherent final policy when a trainer step contains multiple updates.
        post_update_metrics = self._post_update_diagnostics(samples)
        self._validate_update_diagnostics(
            post_update_metrics,
            phase="post_update",
        )
        lr_scheduler = self.lr_schedulers
        if lr_scheduler is None:
            raise RuntimeError("Cosmos trainer did not initialize its LR scheduler")
        optimizer_steps_applied = self._optimizer_steps_applied_in_training_step
        if optimizer_steps_applied:
            lr_scheduler.step()
        else:
            logger.warning(
                "AlpaGym trainer applied no optimizer step at current_step=%d; "
                "leaving the LR scheduler unchanged",
                current_step,
            )
        checkpoint_config = getattr(self.config.train, "ckpt", None)
        checkpoint_enabled = bool(
            getattr(checkpoint_config, "enable_checkpoint", False)
        )
        final_checkpoint_fallback = checkpoint_enabled and current_step == total_steps
        if is_master_replica and (do_save_checkpoint or final_checkpoint_fallback):
            if final_checkpoint_fallback and not do_save_checkpoint:
                # Some Cosmos colocated controller paths send the final real
                # DataFetchCommand with do_save=False and then stop the policy
                # worker before its synthetic TrainingCompleteCommand can run.
                # The trainer owns the last successfully applied optimizer
                # state, so persist it here when checkpointing is enabled.
                logger.info(
                    "Cosmos did not request a checkpoint on final trainer step %d; "
                    "applying AlpaGym final-checkpoint fallback",
                    current_step,
                )
            self._save_checkpoint(current_step, total_steps, remain_samples_num)

        avg_loss = total_loss / num_batches if num_batches else 0.0
        avg_kl = total_kl / num_batches if num_batches else 0.0
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            global_avg_loss = float(
                dist_util.dist_mean(loss_tensor, self.parallel_dims.mesh["dp_cp"])
            )
            global_max_loss = float(
                dist_util.dist_max(loss_tensor, self.parallel_dims.mesh["dp_cp"])
            )
        else:
            global_avg_loss = global_max_loss = avg_loss

        metrics = {
            "train_step": current_step,
            "train/loss_avg": global_avg_loss,
            "train/loss_max": global_max_loss,
            "train/kl_avg": avg_kl,
            "train/learning_rate": float(lr_scheduler.get_last_lr()[0]),
            "train/num_batches": num_batches,
            "train/num_micro_batches": int(
                getattr(self, "_last_micro_batches", num_batches)
            ),
            "train/optimizer_steps_applied": optimizer_steps_applied,
            "train/ratio_max": ratio_max,
            "train/ratio_min": ratio_min,
            "train/clip_fraction": clip_fraction_sum / num_batches
            if num_batches
            else 0.0,
            "train/grad_norm": grad_norm_sum / num_batches if num_batches else 0.0,
            "train/iteration_time": 0.0,
            **pre_update_metrics,
            **post_update_metrics,
        }
        logger.info(
            "AlpaGym trainer step end current_step=%d steps=%d batches=%d "
            "loss_avg=%.6f kl_avg=%.6f ratio_min=%.6f ratio_max=%.6f "
            "clip_fraction=%.6f grad_norm=%.6f lr=%.8g",
            current_step,
            len(samples),
            num_batches,
            float(metrics["train/loss_avg"]),
            avg_kl,
            ratio_min,
            ratio_max,
            float(metrics["train/clip_fraction"]),
            float(metrics["train/grad_norm"]),
            float(metrics["train/learning_rate"]),
        )
        return metrics

    def _pre_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Return optional metrics before any optimizer update is applied."""
        del samples
        return {}

    def _post_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Return optional metrics evaluated after all optimizer updates.

        Generic GRPO policies retain their existing metric path. Actor-critic
        PPO overrides this hook because its replay includes the exact behavior
        actions and masks needed to evaluate final-policy drift.
        """
        del samples
        return {}

    def _validate_update_diagnostics(
        self,
        metrics: dict[str, float | int],
        *,
        phase: str,
    ) -> None:
        """Optionally reject an update from pre/post replay diagnostics."""
        del metrics, phase

    # ------------------------------------------------------------------
    # GRPO orchestration
    # ------------------------------------------------------------------

    def _prepare_training_data(
        self,
        rollouts: list[_rollout_schema.Rollout],
    ) -> tuple[list[Any], torch.Tensor]:
        """Flatten rollouts into a single pool of per-step samples plus advantages.

        Each rollout's artifact unpacks into a fixed-length list of single-step
        replay samples (valid steps plus ``is_padding`` rows); the trainer
        extends one flat pool across all rollouts and records the matching
        per-step advantage. Cosmos supplies one advantage per rollout, replayed
        across that rollout's valid steps; padding steps carry zero so they
        contribute no policy gradient.

        Args:
            rollouts: Cosmos-RL rollouts surviving the staleness filter.

        Returns:
            Tuple ``(samples, advantages)`` aligned row-for-row: ``samples`` is
            the flat per-step pool, ``advantages`` the per-step advantage.
        """
        samples: list[Any] = []
        advantages: list[float] = []
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        for rollout in rollouts:
            step_samples = data_packer.get_policy_input(
                rollout.prompt,
                rollout.completion,
                n_ignore_prefix_tokens=rollout.n_ignore_prefix_tokens,
            )
            for step in step_samples:
                is_padding = bool(step.training_signal.is_padding.item())
                advantages.append(0.0 if is_padding else float(rollout.advantage))
            samples.extend(step_samples)
        return samples, torch.tensor(advantages, dtype=torch.float32)

    def _run_training_loop(
        self,
        samples: list[Any],
        advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, int, float, float, float, float]:
        """Run ``grpo_optimization_iterations × num_mini_batches`` PPO updates.

        Minibatching is at the step level: ``samples`` is the flattened pool of
        per-step replay samples across all rollouts, shuffled fresh each
        optimization iteration and split into minibatches. The collate step
        stacks one minibatch of single-step samples into the ``[B, ...]`` inputs
        the model forward sees.

        Returns:
            Aggregates: ``(total_loss, total_kl, num_batches, ratio_max,
            ratio_min, clip_fraction_sum, grad_norm_sum)``.
        """
        self._ensure_reference_model()
        self._optimizer_steps_applied_in_training_step = 0

        num_steps = len(samples)
        mini_batch_size = min(self._mini_batch, num_steps)
        num_mini_batches = (num_steps + mini_batch_size - 1) // mini_batch_size

        total_loss = 0.0
        total_kl = 0.0
        num_batches = 0
        ratio_max = float("-inf")
        ratio_min = float("inf")
        clip_fraction_sum = 0.0
        grad_norm_sum = 0.0

        for _ in range(self._grpo_optimization_iterations):
            indices = torch.randperm(num_steps)
            for minibatch_index in range(num_mini_batches):
                start = minibatch_index * mini_batch_size
                end = min(start + mini_batch_size, num_steps)
                minibatch_indices = indices[start:end]
                minibatch_samples = [samples[int(index)] for index in minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]

                (
                    loss_value,
                    kl_value,
                    batch_ratio_max,
                    batch_ratio_min,
                    batch_clip_fraction,
                    batch_grad_norm,
                ) = self._train_minibatch(
                    minibatch_samples,
                    minibatch_advantages,
                    inter_policy_nccl,
                )
                total_loss += loss_value
                total_kl += kl_value
                num_batches += 1
                ratio_max = max(ratio_max, batch_ratio_max)
                ratio_min = min(ratio_min, batch_ratio_min)
                clip_fraction_sum += batch_clip_fraction
                grad_norm_sum += batch_grad_norm

        if num_batches == 0:
            ratio_max = 0.0
            ratio_min = 0.0
        return (
            total_loss,
            total_kl,
            num_batches,
            ratio_max,
            ratio_min,
            clip_fraction_sum,
            grad_norm_sum,
        )

    def _train_minibatch(
        self,
        minibatch_samples: list[Any],
        minibatch_advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, float, float, float, float]:
        """Train on a single step-level minibatch and apply gradient.

        Orchestrates: collate the per-step samples, forward the full minibatch,
        compute PPO surrogate + KL penalty, backward + optimizer step, emit
        metrics. Padding rows are forwarded like any other (so every DP worker
        runs the identical model forward in lockstep) but are neutralized: their
        advantage is zero (no policy-loss gradient) and they are masked out of
        the KL term and the diagnostics.

        Args:
            minibatch_samples: one single-step replay sample per row.
            minibatch_advantages: ``[B]`` per-step advantages (zero on padding
                rows), aligned row-for-row with ``minibatch_samples``.
            inter_policy_nccl: NCCL communicator across DP replicas.

        Returns:
            Tuple ``(loss, kl_value, ratio_max, ratio_min, clip_fraction)``.
        """
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        minibatch = data_packer.policy_collate_fn(minibatch_samples)
        is_padding = minibatch.training_signal.is_padding.to(self.device)
        old_logprobs = minibatch.training_signal.old_logprobs.to(self.device)
        advantages = minibatch_advantages.to(device=self.device, dtype=torch.float32)
        new_logprobs, kl_div = self._forward_with_reference(minibatch.model_inputs)
        assert_replay_shapes(new_logprobs, old_logprobs, advantages, kl_div)
        policy_loss, ratio = compute_ppo_surrogate(
            new_logprobs,
            old_logprobs,
            advantages,
            ratio_clip_low=self._grpo_ratio_clip_low,
            ratio_clip_high=self._grpo_ratio_clip_high,
            is_padding=is_padding,
        )
        kl_loss = compute_kl_penalty(
            kl_div,
            is_padding,
            kl_beta=self._kl_beta,
            device=self.device,
        )
        loss = policy_loss + kl_loss

        self.optimizers.zero_grad()
        loss.backward()
        grad_norm = self.all_reduce_states(inter_policy_nccl)

        return self._minibatch_metrics(
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            ratio=ratio,
            is_padding=is_padding,
            advantages=advantages,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            grad_norm=grad_norm,
        )

    def _forward_with_reference(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model forward with the reference model attached for KL."""
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        return result["log_probs"], result.get("kl_div")

    def _minibatch_metrics(
        self,
        policy_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        ratio: torch.Tensor,
        is_padding: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        grad_norm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute ratio/clip diagnostics over the valid (non-padding) rows.

        ``_train_minibatch`` forwards the full minibatch, so the ratio/clip
        diagnostics mask out padding rows here; an all-padding minibatch has no
        valid rows and reports neutral defaults.
        """
        valid_mask = ~is_padding
        loss_value = float((policy_loss + kl_loss).item())
        with torch.no_grad():
            valid_ratio = ratio[valid_mask]
            if valid_ratio.numel() == 0:
                clip_fraction = 0.0
                batch_ratio_max = 1.0
                batch_ratio_min = 1.0
                advantage_mean = 0.0
            else:
                clipped = (valid_ratio < 1.0 - self._grpo_ratio_clip_low) | (
                    valid_ratio > 1.0 + self._grpo_ratio_clip_high
                )
                clip_fraction = float(clipped.float().mean().item())
                batch_ratio_max = float(valid_ratio.max().item())
                batch_ratio_min = float(valid_ratio.min().item())
                advantage_mean = float(advantages[valid_mask].mean().item())
        logger.info(
            "AlpaGym trainer minibatch rows=%d valid_rows=%d loss=%.6f "
            "policy_loss=%.6f kl_loss=%.6f ratio_min=%.6f ratio_max=%.6f "
            "clip_fraction=%.6f advantage_mean=%.6f old_logprob_mean=%.6f "
            "new_logprob_mean=%.6f grad_norm=%.6f",
            int(old_logprobs.numel()),
            int(valid_mask.sum().item()),
            loss_value,
            float(policy_loss.item()),
            float(kl_loss.item()),
            batch_ratio_min,
            batch_ratio_max,
            clip_fraction,
            advantage_mean,
            float(old_logprobs.mean().item()),
            float(new_logprobs.mean().item()),
            grad_norm,
        )
        return (
            loss_value,
            float(kl_loss.item()),
            batch_ratio_max,
            batch_ratio_min,
            clip_fraction,
            grad_norm,
        )

    def all_reduce_states(
        self, inter_policy_nccl: dist_util.HighAvailabilitylNccl
    ) -> float:
        """Reduce gradients across DP replicas, clip norm, and step optimizer.

        Override of `GRPOTrainer.all_reduce_states`. Three differences:
        - Iterates `self.model.parameters()` directly rather than
          `self.model_parts`. Some policy wrappers keep trainable parameters
          under nested child modules, so `model_parts` may not carry the right
          param refs.
        - Captures `current_stream` BEFORE entering the `train_stream`
          context so the all-reduce on `train_stream` waits for FSDP's
          reduce-scatter on the default stream.
        - Raises on a non-finite reduced gradient and skips the step on a zero
          one. Both checks run after the cross-replica reduce, so every DP
          worker reaches the same verdict in lockstep.
        """
        train_stream = self.train_stream
        if train_stream is None:
            raise RuntimeError("Cosmos trainer did not initialize its CUDA stream")
        backward_stream = torch.cuda.current_stream()
        with torch.cuda.stream(train_stream):
            train_stream.wait_stream(backward_stream)
            params = [param for param in self.model.parameters() if param.requires_grad]
            if params:
                dist_util.gradient_reduce_across_dp_replicas_(params, inter_policy_nccl)
            grads = [param.grad for param in params if param.grad is not None]
            if grads and not torch.stack(torch._foreach_norm(grads)).sum().isfinite():
                raise FloatingPointError(
                    "[GRPO:grad-guard] non-finite reduced gradient after backward"
                )
            grad_norm = dist_util.gradient_norm_clipping(
                params,
                self.config.train.optm_grad_norm_clip,
                foreach=True,
                pp_mesh=(
                    self.parallel_dims.mesh["pp"]
                    if self.parallel_dims.pp_enabled
                    else None
                ),
                return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
            )
            grad_norm_value = float(grad_norm) if grad_norm is not None else 0.0
            # Skip the optimizer step on a zero gradient (all-padding or
            # zero-advantage minibatch) so weight decay / Adam state do not
            # advance on no signal.
            if grad_norm_value != 0.0:
                self.optimizers.step()
                self._optimizer_steps_applied_in_training_step += 1
            self.optimizers.zero_grad()
        return grad_norm_value

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
    ) -> None:
        """Save policy weights at ``current_step``.

        Exports deployable weights when
        ``config.train.ckpt.export_safetensors`` is set, then writes the cosmos
        resume checkpoint (model + optimizer + scheduler +
        ``remain_samples_num``) via ``self.ckpt_manager``. The resume checkpoint
        is always written when this method is called, including on the final
        step.

        The inherited ``ckpt_manager`` and ``export_safetensors`` come from
        ``LLMTrainer``; ``output_dir`` / ``ckpt`` / ``param_dtype`` come from
        ``cosmos_config.toml``.
        """
        is_last_step = current_step == total_steps
        if self.config.train.ckpt.export_safetensors:
            export_rel_path = os.path.join("safetensors", f"step_{current_step}")
            export_hook = getattr(
                getattr(self, "_policy_bundle", None),
                "export_model_checkpoint",
                None,
            )
            if export_hook is not None:
                export_path = Path(self.config.train.output_dir) / export_rel_path
                logger.info(
                    "[Policy] Saving policy-native checkpoint at step %d to %s",
                    current_step,
                    export_path,
                )
                export_hook(self.model, export_path)
            else:
                logger.info(
                    "[Policy] Saving huggingface checkpoint at step %d to %s",
                    current_step,
                    self.config.train.output_dir,
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=export_rel_path,
                    trainable_only=False,
                    is_final=is_last_step,
                    # cosmos's `param_dtype` is one of "bfloat16" / "float16" /
                    # "float32"; all map to `torch.<name>` directly.
                    dtype=getattr(torch, str(self.config.train.param_dtype).lower()),
                )

        scheduler = self.lr_schedulers
        if scheduler is None:
            raise RuntimeError("Cosmos trainer did not initialize its LR scheduler")
        logger.info("[Policy] Saving cosmos checkpoint at step %d", current_step)
        self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=scheduler,
            step=current_step,
            total_steps=total_steps,
            remain_samples_num=remain_samples_num,
            is_final=is_last_step,
        )
        self.ckpt_manager.save_check(step=current_step)

    # ------------------------------------------------------------------
    # Reference model lifecycle
    # ------------------------------------------------------------------

    def _ensure_reference_model(self) -> None:
        """Create the frozen initial-policy reference on first use.

        Stores the reference on a private attribute (not registered as a
        submodule, since `Trainer` is an ABC, not an `nn.Module`). AlpaGym's
        ``weight_resume`` records the configured initial policy before a
        Cosmos checkpoint can restore different live weights. Building the
        teacher from that state keeps both KL and actor-mean anchoring fixed
        across process restarts. We drive the reference via ``teacher_model=``
        on the model forward rather than Cosmos's state-dict swapping path.
        """
        if self._kl_beta <= 0.0 or self._reference_model is not None:
            return
        reference_state_dict = getattr(self, "reference_state_dict", None)
        if not reference_state_dict:
            raise RuntimeError(
                "KL reference weights are unavailable; Cosmos weight_resume() "
                "must run before the first AlpaGym training step"
            )
        self._reference_model = copy.deepcopy(self.model)
        self._reference_model.load_state_dict(reference_state_dict, strict=True)
        self._reference_model.eval()
        for param in self._reference_model.parameters():
            param.requires_grad_(False)


def _custom_config_section(config: _cosmos_config.Config, key: str) -> dict[str, Any]:
    """Return one mapping from Cosmos ``config.custom`` without assuming its concrete type."""
    custom = getattr(config, "custom", None) or {}
    section = custom.get(key, {}) if hasattr(custom, "get") else {}
    return dict(section) if isinstance(section, dict) else {}


def _require_ppo_signal(tensor: torch.Tensor | None, field_name: str) -> torch.Tensor:
    """Return a required PPO training-signal tensor or raise a targeted error."""
    if tensor is None:
        raise ValueError(f"PPO trainer requires TrainingSignal.{field_name}")
    return tensor


def _ppo_actor_valid_mask(
    signal: TrainingSignal,
    *,
    required: bool = False,
) -> torch.Tensor:
    """Return rows whose sampled policy action reached the controller."""
    if signal.actor_valid is None:
        if required:
            raise ValueError(
                "Flow-PPO replay requires TrainingSignal.actor_valid so "
                "unexecuted sampled references cannot enter the actor loss"
            )
        return torch.ones_like(signal.is_padding, dtype=torch.bool)
    return signal.actor_valid


def _discounted_transition_reward(
    signal: TrainingSignal,
    *,
    gamma: float,
) -> tuple[float, int]:
    """Return one macro reward and its controller-tick duration.

    Direct policies carry one scalar reward and therefore have duration one.
    Motion-reference policies carry the exact committed controller-tick reward
    prefix. Keeping the primitive rewards in replay lets the trainer change
    neither their time order nor the semi-Markov discount by accident.
    """
    if signal.primitive_rewards is None:
        reward = _require_ppo_signal(signal.rewards, "rewards")
        return float(reward.item()), 1

    primitive_rewards = signal.primitive_rewards.reshape(-1)
    primitive_mask = _require_ppo_signal(
        signal.primitive_reward_mask,
        "primitive_reward_mask",
    ).reshape(-1)
    duration_tensor = _require_ppo_signal(signal.duration_ticks, "duration_ticks")
    duration = int(duration_tensor.item())
    if duration <= 0 or duration > primitive_rewards.numel():
        raise ValueError(
            "PPO duration_ticks must select a non-empty primitive reward prefix"
        )
    expected_mask = torch.arange(primitive_rewards.numel()) < duration
    if not torch.equal(primitive_mask.cpu(), expected_mask):
        raise ValueError(
            "PPO primitive_reward_mask must be a contiguous prefix matching duration_ticks"
        )
    selected = primitive_rewards[:duration]
    if not torch.isfinite(selected).all():
        raise ValueError("PPO primitive rewards must be finite")
    powers = torch.pow(
        torch.tensor(float(gamma), dtype=torch.float64),
        torch.arange(duration, dtype=torch.float64),
    )
    reward = torch.sum(selected.to(dtype=torch.float64).cpu() * powers)
    return float(reward.item()), duration


def _with_ppo_targets(sample: Any, *, advantage: float, ret: float) -> Any:
    """Attach trainer-computed PPO targets to one replay sample."""
    signal = sample.training_signal
    return replace(
        sample,
        training_signal=replace(
            signal,
            advantages=torch.tensor([advantage], dtype=torch.float32),
            returns=torch.tensor([ret], dtype=torch.float32),
        ),
    )


def _replace_advantages(
    samples: list[Any],
    advantages: torch.Tensor,
    per_rollout_ranges: list[tuple[int, int]],
) -> list[Any]:
    """Mirror normalized advantages into each sample's internal training signal."""
    del per_rollout_ranges
    return [
        _with_ppo_targets(
            sample,
            advantage=float(advantages[index].item()),
            ret=float(
                _require_ppo_signal(sample.training_signal.returns, "returns").item()
            ),
        )
        for index, sample in enumerate(samples)
    ]


def _summarize_behavior_log_ratios(
    log_ratios: torch.Tensor,
    *,
    phase: str,
    ratio_clip_low: float,
    ratio_clip_high: float,
) -> dict[str, float | int]:
    """Summarize one flat, already-masked behavior-ratio pool."""
    if phase not in {"pre_update", "post_update"}:
        raise ValueError(f"Unsupported PPO behavior-diagnostic phase: {phase!r}")
    prefix = f"train/{phase}"
    flattened = log_ratios.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if flattened.numel() == 0:
        return {
            f"{prefix}_valid_rows": 0,
            f"{prefix}_ratio_p01": 1.0,
            f"{prefix}_ratio_p50": 1.0,
            f"{prefix}_ratio_p99": 1.0,
            f"{prefix}_clip_fraction": 0.0,
            f"{prefix}_approx_kl": 0.0,
            f"{prefix}_max_abs_log_ratio": 0.0,
        }
    if not torch.isfinite(flattened).all():
        raise FloatingPointError(
            f"PPO {phase.replace('_', '-')} log-ratios contain non-finite values"
        )

    # Ratio diagnostics match the bounded exponent used by the PPO objective,
    # but approximate KL intentionally retains the raw log-ratio so the clamp
    # cannot hide policy divergence. Float64 keeps large finite joint Flow
    # deltas observable without overflowing at float32's exponent boundary.
    bounded_log_ratios = flattened.clamp(min=-5.0, max=5.0)
    ratios = bounded_log_ratios.exp()
    quantiles = torch.quantile(
        ratios,
        torch.tensor([0.01, 0.5, 0.99], dtype=ratios.dtype),
    )
    clipped = (ratios < 1.0 - ratio_clip_low) | (ratios > 1.0 + ratio_clip_high)
    raw_log_ratios = flattened.to(dtype=torch.float64)
    approx_kl = torch.expm1(raw_log_ratios) - raw_log_ratios
    if not torch.isfinite(approx_kl).all():
        raise FloatingPointError(
            f"PPO {phase.replace('_', '-')} approximate KL is non-finite "
            "for raw log-ratios"
        )
    return {
        f"{prefix}_valid_rows": int(ratios.numel()),
        f"{prefix}_ratio_p01": float(quantiles[0].item()),
        f"{prefix}_ratio_p50": float(quantiles[1].item()),
        f"{prefix}_ratio_p99": float(quantiles[2].item()),
        f"{prefix}_clip_fraction": float(clipped.float().mean().item()),
        f"{prefix}_approx_kl": float(approx_kl.mean().item()),
        f"{prefix}_max_abs_log_ratio": float(raw_log_ratios.abs().max().item()),
    }


def _summarize_post_update_log_ratios(
    log_ratios: torch.Tensor,
    *,
    ratio_clip_low: float,
    ratio_clip_high: float,
) -> dict[str, float | int]:
    """Backward-compatible post-update behavior-ratio summary helper."""
    return _summarize_behavior_log_ratios(
        log_ratios,
        phase="post_update",
        ratio_clip_low=ratio_clip_low,
        ratio_clip_high=ratio_clip_high,
    )


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_ppo")
class AlpagymPPOTrainer(AlpagymGRPOTrainer):
    """Actor-critic PPO trainer over AlpaGym replay payloads.

    This trainer keeps the AlpaGym rollout transport exactly the same as GRPO:
    completed rollouts still arrive as ``EpisodeOutput`` artifacts and each
    ``PolicyOutput.replay_data`` still owns the per-step model replay payload.
    The difference is the training signal source: PPO computes per-step
    ``advantages`` and ``returns`` inside the trainer from replayed transition
    rewards, terminal flags, and rollout-time values, then trains a value head
    from model forward key ``values``.
    """

    _flow_chunk_density = False
    _dual_clip_ratio: float | None = None
    _value_huber_delta: float | None = None

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO-specific hyperparameters after shared Cosmos setup."""
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        ppo_config = _custom_config_section(config, "ppo")
        self._value_loss_coef = float(ppo_config.get("value_loss_coef", 0.5))
        if self._value_loss_coef < 0.0:
            raise ValueError(
                f"PPO value_loss_coef must be non-negative, got {self._value_loss_coef}"
            )
        value_clip_range = ppo_config.get("value_clip_range")
        self._value_clip_range = (
            None if value_clip_range is None else float(value_clip_range)
        )
        if self._value_clip_range is not None and self._value_clip_range <= 0.0:
            raise ValueError(
                f"PPO value_clip_range must be positive when set, got {self._value_clip_range}"
            )
        dual_clip_ratio = ppo_config.get("dual_clip_ratio")
        self._dual_clip_ratio = (
            None if dual_clip_ratio is None else float(dual_clip_ratio)
        )
        if self._dual_clip_ratio is not None and self._dual_clip_ratio <= 1.0:
            raise ValueError("PPO dual_clip_ratio must be greater than 1")
        value_huber_delta = ppo_config.get("value_huber_delta")
        self._value_huber_delta = (
            None if value_huber_delta is None else float(value_huber_delta)
        )
        if self._value_huber_delta is not None and self._value_huber_delta <= 0.0:
            raise ValueError("PPO value_huber_delta must be positive")
        self._normalize_advantages = bool(ppo_config.get("normalize_advantages", True))
        self._gamma = float(ppo_config.get("gamma", 0.99))
        self._gae_lambda = float(ppo_config.get("gae_lambda", 0.95))
        self._min_action_std = float(ppo_config.get("min_action_std", 0.02))
        self._max_action_std = float(ppo_config.get("max_action_std", 2.0))
        if not 0.0 < self._min_action_std <= self._max_action_std:
            raise ValueError(
                "PPO action std bounds must satisfy 0 < min_action_std <= "
                f"max_action_std; got {self._min_action_std}, {self._max_action_std}"
            )
        target_behavior_kl = ppo_config.get("target_behavior_kl")
        self._target_behavior_kl = (
            None if target_behavior_kl is None else float(target_behavior_kl)
        )
        if self._target_behavior_kl is not None and (
            not math.isfinite(self._target_behavior_kl)
            or self._target_behavior_kl <= 0.0
        ):
            raise ValueError(
                "PPO target_behavior_kl must be finite and positive when set, "
                f"got {target_behavior_kl!r}"
            )
        step_mini_batch = ppo_config.get("step_mini_batch", self._mini_batch)
        if (
            isinstance(step_mini_batch, bool)
            or not isinstance(step_mini_batch, int)
            or step_mini_batch <= 0
        ):
            raise ValueError(
                "PPO step_mini_batch must be a positive integer, "
                f"got {step_mini_batch!r}"
            )
        # Cosmos's train_policy.mini_batch remains rollout/shard geometry;
        # this loop batches flattened actor-critic transitions independently.
        self._mini_batch = step_mini_batch
        if not 0.0 <= self._gamma <= 1.0:
            raise ValueError(f"PPO gamma must be in [0, 1], got {self._gamma}")
        if not 0.0 <= self._gae_lambda <= 1.0:
            raise ValueError(
                f"PPO gae_lambda must be in [0, 1], got {self._gae_lambda}"
            )

    def all_reduce_states(
        self,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> float:
        """Apply the optimizer step and clamp Gaussian std on its CUDA stream."""
        caller_stream = torch.cuda.current_stream()
        grad_norm = super().all_reduce_states(inter_policy_nccl)
        clamp_std = getattr(self.model, "clamp_std_", None)
        if callable(clamp_std):
            # The shared optimizer step is queued on ``train_stream``.  Queue
            # the in-place clamp on the same stream so it cannot race Adam,
            # then make the caller/default stream wait before the next model
            # forward observes the parameters.
            with torch.cuda.stream(self.train_stream):
                clamp_std(
                    min_std=self._min_action_std,
                    max_std=self._max_action_std,
                )
        caller_stream.wait_stream(self.train_stream)
        return grad_norm

    def _prepare_training_data(
        self,
        rollouts: list[_rollout_schema.Rollout],
    ) -> tuple[list[Any], torch.Tensor]:
        """Compute full-episode GAE, then flatten only rows needed for training.

        On one GPU, synthetic padding rows are removed after GAE so their cloned
        camera tensors never enter the expensive visual model forward. Real
        ``actor_valid=false`` rows remain because they still supervise the value
        head. Multi-rank jobs retain padding until cross-rank microbatch
        scheduling is made collective-safe.
        """
        samples: list[Any] = []
        advantages: list[float] = []
        actor_valid_rows: list[bool] = []
        per_rollout_ranges: list[tuple[int, int]] = []
        compact_padding = (
            int(getattr(getattr(self, "parallel_dims", None), "world_size", 1)) == 1
        )
        padding_rows_dropped = 0
        rollout_versions: set[int] = set()
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        for rollout in rollouts:
            start = len(samples)
            step_samples = data_packer.get_policy_input(
                rollout.prompt,
                rollout.completion,
                n_ignore_prefix_tokens=rollout.n_ignore_prefix_tokens,
            )
            rollout_weight_version = int(rollout.weight_version)
            rollout_versions.add(rollout_weight_version)
            for step in step_samples:
                replay_weight_version = int(step.weight_version.item())
                if replay_weight_version != rollout_weight_version:
                    raise ValueError(
                        "PPO replay behavior version does not match Cosmos rollout "
                        f"version: replay={replay_weight_version}, "
                        f"rollout={rollout_weight_version}"
                    )
            rollout_advantages, rollout_returns = self._compute_gae(step_samples)
            for step, advantage, ret in zip(
                step_samples, rollout_advantages, rollout_returns
            ):
                is_padding = bool(step.training_signal.is_padding.item())
                if is_padding and compact_padding:
                    padding_rows_dropped += 1
                    continue
                actor_valid = (
                    bool(
                        _ppo_actor_valid_mask(
                            step.training_signal,
                            required=self._flow_chunk_density,
                        ).item()
                    )
                    and not is_padding
                )
                actor_advantage = advantage if actor_valid else 0.0
                samples.append(
                    _with_ppo_targets(
                        step,
                        advantage=actor_advantage,
                        ret=ret,
                    )
                )
                advantages.append(actor_advantage)
                actor_valid_rows.append(actor_valid)
            per_rollout_ranges.append((start, len(samples)))

        if getattr(self, "_on_policy", False) and len(rollout_versions) > 1:
            raise ValueError(
                "On-policy PPO requires one frozen behavior weight version per "
                f"optimizer batch, got {sorted(rollout_versions)}"
            )
        if padding_rows_dropped:
            logger.info(
                "AlpaGym PPO removed %d synthetic padding rows before visual forward; "
                "%d real rows remain",
                padding_rows_dropped,
                len(samples),
            )

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        if self._normalize_advantages and advantage_tensor.numel() > 0:
            valid_mask = torch.tensor(actor_valid_rows, dtype=torch.bool)
            if int(valid_mask.sum().item()) > 1:
                valid_advantages = advantage_tensor[valid_mask]
                std = valid_advantages.std(unbiased=False)
                if float(std.item()) > 0.0:
                    advantage_tensor[valid_mask] = (
                        valid_advantages - valid_advantages.mean()
                    ) / (std + 1.0e-8)
            advantage_tensor[~valid_mask] = 0.0
            samples = _replace_advantages(samples, advantage_tensor, per_rollout_ranges)
        return samples, advantage_tensor

    def _run_training_loop(
        self,
        samples: list[Any],
        advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
    ) -> tuple[float, float, int, float, float, float, float]:
        """Accumulate transition microbatches into one Adam step per PPO pass.

        ``step_mini_batch`` limits one visual forward/backward. It does not
        define the optimizer batch. Actor and value losses use separate exact
        row-count weights, so splitting a batch does not change either masked
        mean objective.
        """
        self._ensure_reference_model()
        self._optimizer_steps_applied_in_training_step = 0
        if not samples:
            return (0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        if len(advantages) != len(samples):
            raise ValueError(
                "PPO advantages must align one-for-one with replay samples: "
                f"{len(advantages)} != {len(samples)}"
            )

        def row_counts(rows: list[Any]) -> tuple[int, int]:
            actor_rows = 0
            value_rows = 0
            for sample in rows:
                signal = sample.training_signal
                is_padding = bool(signal.is_padding.item())
                if is_padding:
                    continue
                value_rows += 1
                if bool(
                    _ppo_actor_valid_mask(
                        signal,
                        required=self._flow_chunk_density,
                    ).item()
                ):
                    actor_rows += 1
            return actor_rows, value_rows

        total_actor_rows, total_value_rows = row_counts(samples)
        if total_value_rows == 0:
            raise ValueError("PPO optimizer batch has no non-padding value rows")

        num_steps = len(samples)
        micro_batch_size = min(self._mini_batch, num_steps)
        num_micro_batches = (num_steps + micro_batch_size - 1) // micro_batch_size
        self._last_micro_batches = 0

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0
        ratio_max = float("-inf")
        ratio_min = float("inf")
        clip_fraction_sum = 0.0
        grad_norm_sum = 0.0

        for _ in range(self._grpo_optimization_iterations):
            indices = torch.randperm(num_steps)
            self.optimizers.zero_grad()
            update_loss = 0.0
            update_kl = 0.0
            update_clip_fraction = 0.0
            update_ratio_max = float("-inf")
            update_ratio_min = float("inf")

            for microbatch_index in range(num_micro_batches):
                start = microbatch_index * micro_batch_size
                end = min(start + micro_batch_size, num_steps)
                minibatch_indices = indices[start:end]
                minibatch_samples = [samples[int(index)] for index in minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                actor_rows, value_rows = row_counts(minibatch_samples)
                actor_loss_scale = (
                    actor_rows / total_actor_rows if total_actor_rows else 0.0
                )
                value_loss_scale = value_rows / total_value_rows

                (
                    loss_value,
                    kl_value,
                    batch_ratio_max,
                    batch_ratio_min,
                    batch_clip_fraction,
                    _batch_grad_norm,
                ) = self._train_minibatch(
                    minibatch_samples,
                    minibatch_advantages,
                    inter_policy_nccl,
                    actor_loss_scale=actor_loss_scale,
                    value_loss_scale=value_loss_scale,
                    apply_optimizer=False,
                )
                update_loss += loss_value
                update_kl += actor_loss_scale * kl_value
                if actor_rows:
                    update_ratio_max = max(update_ratio_max, batch_ratio_max)
                    update_ratio_min = min(update_ratio_min, batch_ratio_min)
                    update_clip_fraction += actor_loss_scale * batch_clip_fraction
                self._last_micro_batches += 1

            optimizer_steps_before = self._optimizer_steps_applied_in_training_step
            grad_norm = self.all_reduce_states(inter_policy_nccl)
            optimizer_step_applied = (
                self._optimizer_steps_applied_in_training_step > optimizer_steps_before
            )
            if total_actor_rows:
                ratio_max = max(ratio_max, update_ratio_max)
                ratio_min = min(ratio_min, update_ratio_min)
            total_loss += update_loss
            total_kl += update_kl
            clip_fraction_sum += update_clip_fraction
            grad_norm_sum += grad_norm
            if optimizer_step_applied:
                num_updates += 1
            logger.info(
                "AlpaGym PPO effective optimizer update rows=%d actor_rows=%d "
                "micro_batches=%d applied=%s loss=%.6f kl=%.6f grad_norm=%.6f",
                total_value_rows,
                total_actor_rows,
                num_micro_batches,
                optimizer_step_applied,
                update_loss,
                update_kl,
                grad_norm,
            )

        if not total_actor_rows:
            ratio_max = 1.0
            ratio_min = 1.0

        return (
            total_loss,
            total_kl,
            num_updates,
            ratio_max,
            ratio_min,
            clip_fraction_sum,
            grad_norm_sum,
        )

    def _pre_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Measure behavior/replay alignment before any optimizer mutation."""
        return self._behavior_diagnostics(samples, phase="pre_update")

    def _post_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Measure behavior-policy drift after the effective optimizer step."""
        return self._behavior_diagnostics(samples, phase="post_update")

    def _behavior_diagnostics(
        self,
        samples: list[Any],
        *,
        phase: str,
    ) -> dict[str, float | int]:
        """Re-score frozen behavior actions under one coherent policy state.

        This is a pure no-grad forward pass: it never calls backward, gradient
        reduction, an optimizer, or the scheduler. Each actor-valid replay row
        contributes one scalar policy-density ratio.
        """
        if phase not in {"pre_update", "post_update"}:
            raise ValueError(f"Unsupported PPO behavior-diagnostic phase: {phase!r}")
        if not samples:
            raise ValueError(f"PPO {phase} diagnostics require replay samples")
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError(f"PPO {phase} diagnostics require a data packer")

        valid_log_ratios: list[torch.Tensor] = []
        valid_value_deltas: list[torch.Tensor] = []
        diagnostic_batch_size = min(self._mini_batch, len(samples))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(samples), diagnostic_batch_size):
                    minibatch = data_packer.policy_collate_fn(
                        samples[start : start + diagnostic_batch_size]
                    )
                    signal = minibatch.training_signal
                    is_padding = signal.is_padding.to(self.device)
                    actor_valid = _ppo_actor_valid_mask(
                        signal,
                        required=self._flow_chunk_density,
                    ).to(self.device)
                    old_logprobs = signal.old_logprobs.to(self.device)
                    old_values = _require_ppo_signal(
                        signal.old_values,
                        "old_values",
                    ).to(self.device)
                    (
                        new_logprobs,
                        _kl_div,
                        values,
                        new_element_logprobs,
                        old_element_logprobs,
                    ) = self._forward_with_reference_and_value(minibatch.model_inputs)

                    if tuple(new_logprobs.shape) != tuple(old_logprobs.shape):
                        raise ValueError(
                            f"PPO {phase} new/old scalar log-probability shapes differ: "
                            f"{tuple(new_logprobs.shape)} != "
                            f"{tuple(old_logprobs.shape)}"
                        )
                    if not torch.isfinite(new_logprobs).all():
                        raise FloatingPointError(
                            f"PPO {phase} model returned non-finite log-probabilities"
                        )
                    if not torch.isfinite(old_logprobs).all():
                        raise FloatingPointError(
                            f"PPO {phase} replay has non-finite old log-probabilities"
                        )
                    if tuple(values.shape) != tuple(old_values.shape):
                        raise ValueError(
                            f"PPO {phase} current/old value shapes differ: "
                            f"{tuple(values.shape)} != {tuple(old_values.shape)}"
                        )
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            f"PPO {phase} model returned non-finite values"
                        )
                    if not torch.isfinite(old_values).all():
                        raise FloatingPointError(
                            f"PPO {phase} replay has non-finite old values"
                        )

                    if self._flow_chunk_density:
                        if new_element_logprobs is None or old_element_logprobs is None:
                            raise ValueError(
                                "Flow-PPO requires full selected-transition element "
                                "log-probabilities"
                            )
                        if not torch.allclose(
                            old_element_logprobs.sum(dim=-1),
                            old_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ) or not torch.allclose(
                            new_element_logprobs.sum(dim=-1),
                            new_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ):
                            raise ValueError(
                                "Flow-PPO joint log-probability differs from its "
                                "selected-transition elements"
                            )
                    elif (
                        new_element_logprobs is not None
                        or old_element_logprobs is not None
                    ):
                        raise ValueError(
                            "alpagym_ppo accepts only scalar policy densities; "
                            "use alpagym_flow_ppo for Flow-SDE elements"
                        )

                    log_ratios = new_logprobs - old_logprobs
                    valid_mask = (~is_padding) & actor_valid

                    selected = log_ratios[valid_mask]
                    if selected.numel() > 0:
                        valid_log_ratios.append(
                            selected.detach().to(device="cpu", dtype=torch.float32)
                        )
                    selected_value_deltas = (values - old_values).abs()[~is_padding]
                    if selected_value_deltas.numel() > 0:
                        valid_value_deltas.append(
                            selected_value_deltas.detach().to(
                                device="cpu",
                                dtype=torch.float32,
                            )
                        )
        finally:
            self.model.train(was_training)

        metrics = _summarize_behavior_log_ratios(
            torch.cat(valid_log_ratios) if valid_log_ratios else torch.empty(0),
            phase=phase,
            ratio_clip_low=self._grpo_ratio_clip_low,
            ratio_clip_high=self._grpo_ratio_clip_high,
        )
        prefix = f"train/{phase}"
        value_deltas = (
            torch.cat(valid_value_deltas)
            if valid_value_deltas
            else torch.empty(0, dtype=torch.float32)
        )
        metrics[f"{prefix}_value_valid_rows"] = int(value_deltas.numel())
        metrics[f"{prefix}_value_max_abs_delta"] = (
            float(value_deltas.max().item()) if value_deltas.numel() else 0.0
        )
        logger.info(
            "AlpaGym PPO %s diagnostics valid_rows=%d "
            "ratio_p01=%.6f ratio_p50=%.6f ratio_p99=%.6f "
            "clip_fraction=%.6f approx_kl=%.6f max_abs_log_ratio=%.6f "
            "value_rows=%d value_max_abs_delta=%.6f",
            phase,
            int(metrics[f"{prefix}_valid_rows"]),
            float(metrics[f"{prefix}_ratio_p01"]),
            float(metrics[f"{prefix}_ratio_p50"]),
            float(metrics[f"{prefix}_ratio_p99"]),
            float(metrics[f"{prefix}_clip_fraction"]),
            float(metrics[f"{prefix}_approx_kl"]),
            float(metrics[f"{prefix}_max_abs_log_ratio"]),
            int(metrics[f"{prefix}_value_valid_rows"]),
            float(metrics[f"{prefix}_value_max_abs_delta"]),
        )
        return metrics

    def _validate_update_diagnostics(
        self,
        metrics: dict[str, float | int],
        *,
        phase: str,
    ) -> None:
        """Fail closed when calibrated behavior-policy KL exceeds its guard."""
        if phase == "pre_update" and self._on_policy:
            valid_rows = int(metrics["train/pre_update_valid_rows"])
            max_abs_log_ratio = float(metrics["train/pre_update_max_abs_log_ratio"])
            if valid_rows > 0 and (
                not math.isfinite(max_abs_log_ratio) or max_abs_log_ratio > 1.0e-4
            ):
                raise FloatingPointError(
                    "On-policy PPO pre-update replay differs from its behavior "
                    "policy: max_abs_log_ratio="
                    f"{max_abs_log_ratio:.6g} exceeds 0.0001"
                )
            if self._flow_chunk_density:
                value_rows = int(metrics["train/pre_update_value_valid_rows"])
                max_abs_value_delta = float(
                    metrics["train/pre_update_value_max_abs_delta"]
                )
                if value_rows <= 0:
                    raise FloatingPointError(
                        "On-policy Flow-PPO pre-update replay has no value rows"
                    )
                if (
                    not math.isfinite(max_abs_value_delta)
                    or max_abs_value_delta > 1.0e-4
                ):
                    raise FloatingPointError(
                        "On-policy Flow-PPO pre-update replay differs from its "
                        "behavior critic: max_abs_value_delta="
                        f"{max_abs_value_delta:.6g} exceeds 0.0001"
                    )
        target = getattr(self, "_target_behavior_kl", None)
        if target is None:
            return
        if phase not in {"pre_update", "post_update"}:
            raise ValueError(f"Unsupported PPO behavior-KL phase: {phase!r}")
        phase_label = phase.replace("_", "-")
        prefix = f"train/{phase}"
        valid_rows = int(metrics[f"{prefix}_valid_rows"])
        approx_kl = float(metrics[f"{prefix}_approx_kl"])
        if valid_rows <= 0:
            if self._flow_chunk_density:
                # Flow actor and critic parameters are disjoint, so a
                # critic-only batch cannot move the behavior density.
                return
            raise FloatingPointError(
                f"PPO {phase_label} behavior-KL guard has no actor-valid replay rows"
            )
        if not math.isfinite(approx_kl):
            raise FloatingPointError(
                f"PPO {phase_label} behavior KL is non-finite: {approx_kl}"
            )
        if approx_kl > target:
            raise FloatingPointError(
                f"PPO {phase_label} behavior KL {approx_kl:.6g} exceeds calibrated "
                f"target {target:.6g}; refusing to advance scheduler, checkpoint, "
                "or weight sync"
            )

    def _compute_gae(self, step_samples: list[Any]) -> tuple[list[float], list[float]]:
        """Compute direct or semi-Markov GAE targets for one rollout."""
        transitions: list[tuple[float, int, bool, bool, float, float | None]] = []
        valid_indices: list[int] = []
        for index, step in enumerate(step_samples):
            signal = step.training_signal
            if bool(signal.is_padding.item()):
                continue
            terminated = _require_ppo_signal(signal.terminateds, "terminateds")
            truncated = (
                False if signal.truncateds is None else bool(signal.truncateds.item())
            )
            if bool(terminated.item()) and truncated:
                raise ValueError(
                    "PPO transition cannot be both terminated and truncated"
                )
            value = _require_ppo_signal(signal.old_values, "old_values")
            reward, duration = _discounted_transition_reward(
                signal,
                gamma=self._gamma,
            )
            bootstrap = (
                None
                if signal.bootstrap_values is None
                else float(signal.bootstrap_values.item())
            )
            transitions.append(
                (
                    reward,
                    duration,
                    bool(terminated.item()),
                    truncated,
                    float(value.item()),
                    bootstrap,
                )
            )
            valid_indices.append(index)

        advantages = [0.0 for _ in step_samples]
        returns = [0.0 for _ in step_samples]
        last_gae = 0.0
        for valid_pos in reversed(range(len(valid_indices))):
            sample_index = valid_indices[valid_pos]
            reward, duration, terminated, truncated, value, bootstrap = transitions[
                valid_pos
            ]
            if bootstrap is None:
                bootstrap = (
                    transitions[valid_pos + 1][4]
                    if valid_pos + 1 < len(transitions)
                    else 0.0
                )
            bootstrap_discount = 0.0 if terminated else self._gamma**duration
            delta = reward + bootstrap_discount * bootstrap - value
            continues = (
                not terminated and not truncated and valid_pos + 1 < len(transitions)
            )
            trace_discount = (
                (self._gamma * self._gae_lambda) ** duration if continues else 0.0
            )
            last_gae = delta + trace_discount * last_gae
            advantages[sample_index] = float(last_gae)
            returns[sample_index] = float(last_gae + value)
        return advantages, returns

    def _train_minibatch(
        self,
        minibatch_samples: list[Any],
        minibatch_advantages: torch.Tensor,
        inter_policy_nccl: dist_util.HighAvailabilitylNccl,
        *,
        actor_loss_scale: float = 1.0,
        value_loss_scale: float = 1.0,
        apply_optimizer: bool = True,
    ) -> tuple[float, float, float, float, float, float]:
        """Backprop one PPO transition microbatch.

        Direct callers retain the historical one-minibatch/one-step behavior.
        The PPO loop passes exact actor/value row fractions and defers the Adam
        step so several visual microbatches form one effective optimizer batch.
        """
        if actor_loss_scale < 0.0 or value_loss_scale < 0.0:
            raise ValueError("PPO microbatch loss scales must be non-negative")
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("Cosmos trainer did not initialize its data packer")
        minibatch = data_packer.policy_collate_fn(minibatch_samples)
        signal = minibatch.training_signal
        is_padding = signal.is_padding.to(self.device)
        actor_valid = _ppo_actor_valid_mask(
            signal,
            required=self._flow_chunk_density,
        ).to(self.device)
        actor_is_padding = is_padding | ~actor_valid
        old_logprobs = signal.old_logprobs.to(self.device)
        advantages = minibatch_advantages.to(device=self.device, dtype=torch.float32)
        returns = _require_ppo_signal(signal.returns, "returns").to(self.device)
        old_values_tensor = signal.old_values
        old_values = (
            None if old_values_tensor is None else old_values_tensor.to(self.device)
        )
        if self._value_clip_range is not None and old_values is None:
            raise ValueError("PPO value clipping requires TrainingSignal.old_values")

        (
            new_logprobs,
            kl_div,
            values,
            new_element_logprobs,
            old_element_logprobs,
        ) = self._forward_with_reference_and_value(minibatch.model_inputs)
        assert_replay_shapes(
            new_logprobs,
            old_logprobs,
            advantages,
            kl_div,
            values=values,
            returns=returns,
            old_values=old_values,
        )
        if self._flow_chunk_density:
            if new_element_logprobs is None or old_element_logprobs is None:
                raise ValueError(
                    "Flow-PPO requires full selected-transition element "
                    "log-probabilities"
                )
            dual_clip_ratio = self._dual_clip_ratio
            if dual_clip_ratio is None:
                raise ValueError("Flow-PPO requires a configured dual_clip_ratio")
            policy_loss, ratio = compute_flow_ppo_surrogate(
                new_element_logprobs,
                old_element_logprobs,
                new_logprobs,
                old_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                dual_clip_ratio=dual_clip_ratio,
                is_padding=actor_is_padding,
            )
        else:
            if new_element_logprobs is not None or old_element_logprobs is not None:
                raise ValueError(
                    "alpagym_ppo accepts only scalar policy densities; "
                    "use alpagym_flow_ppo for Flow-SDE elements"
                )
            policy_loss, ratio = compute_ppo_surrogate(
                new_logprobs,
                old_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                is_padding=actor_is_padding,
                dual_clip_ratio=self._dual_clip_ratio,
            )
        kl_loss = compute_kl_penalty(
            kl_div,
            actor_is_padding,
            kl_beta=self._kl_beta,
            device=self.device,
        )
        value_loss = compute_value_loss(
            values,
            returns,
            is_padding,
            old_values=old_values,
            value_clip_range=self._value_clip_range,
            huber_delta=self._value_huber_delta,
        )
        loss = self._value_loss_coef * value_loss_scale * value_loss
        has_actor_rows = bool(((~is_padding) & actor_valid).any().item())
        if actor_loss_scale > 0.0 and has_actor_rows:
            # Do not attach an all-zero actor objective to critic-only batches.
            # A zero-but-non-None AdamW gradient would still apply weight decay
            # and advance actor optimizer state when the critic steps.
            loss = loss + actor_loss_scale * (policy_loss + kl_loss)

        if apply_optimizer:
            self.optimizers.zero_grad()
        loss.backward()
        grad_norm = (
            self.all_reduce_states(inter_policy_nccl) if apply_optimizer else 0.0
        )

        return self._ppo_minibatch_metrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_loss=kl_loss,
            ratio=ratio,
            is_padding=is_padding,
            actor_valid=actor_valid,
            advantages=advantages,
            returns=returns,
            values=values,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            grad_norm=grad_norm,
        )

    def _forward_with_reference_and_value(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Run actor-critic forward with optional Flow-density elements."""
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        old_element_logprobs = forward_kwargs.pop("old_element_logprobs", None)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        new_element_logprobs = result.get("element_log_probs")
        if (new_element_logprobs is None) != (old_element_logprobs is None):
            raise ValueError(
                "Flow-PPO requires both model element_log_probs and replay "
                "old_element_logprobs"
            )
        return (
            result["log_probs"],
            result.get("kl_div"),
            result["values"].reshape(-1),
            new_element_logprobs,
            old_element_logprobs,
        )

    def _ppo_minibatch_metrics(
        self,
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        ratio: torch.Tensor,
        is_padding: torch.Tensor,
        actor_valid: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        grad_norm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute PPO diagnostics over valid rows and return trainer-loop metrics."""
        actor_mask = (~is_padding) & actor_valid
        value_mask = ~is_padding
        loss_value = float(loss.item())
        with torch.no_grad():
            if bool(value_mask.any()):
                return_mean = float(returns[value_mask].mean().item())
                value_mean = float(values[value_mask].mean().item())
            else:
                return_mean = 0.0
                value_mean = 0.0
            if bool(actor_mask.any()):
                old_logprob_mean = float(old_logprobs[actor_mask].mean().item())
                new_logprob_mean = float(new_logprobs[actor_mask].mean().item())
            else:
                old_logprob_mean = 0.0
                new_logprob_mean = 0.0
            if ratio.shape != actor_mask.shape:
                raise ValueError(
                    "PPO joint ratio shape must match replay rows: "
                    f"{tuple(ratio.shape)} != {tuple(actor_mask.shape)}"
                )
            valid_ratio = ratio[actor_mask]
            if valid_ratio.numel() == 0:
                clip_fraction = 0.0
                batch_ratio_max = 1.0
                batch_ratio_min = 1.0
                advantage_mean = 0.0
            else:
                clipped = (valid_ratio < 1.0 - self._grpo_ratio_clip_low) | (
                    valid_ratio > 1.0 + self._grpo_ratio_clip_high
                )
                clip_fraction = float(clipped.float().mean().item())
                batch_ratio_max = float(valid_ratio.max().item())
                batch_ratio_min = float(valid_ratio.min().item())
                advantage_mean = float(advantages[actor_mask].mean().item())
        logger.info(
            "AlpaGym PPO minibatch rows=%d valid_rows=%d loss=%.6f "
            "policy_loss=%.6f value_loss=%.6f kl_loss=%.6f ratio_min=%.6f "
            "ratio_max=%.6f clip_fraction=%.6f advantage_mean=%.6f "
            "return_mean=%.6f value_mean=%.6f old_logprob_mean=%.6f "
            "new_logprob_mean=%.6f grad_norm=%.6f",
            int(old_logprobs.numel()),
            int(actor_mask.sum().item()),
            loss_value,
            float(policy_loss.item()),
            float(value_loss.item()),
            float(kl_loss.item()),
            batch_ratio_min,
            batch_ratio_max,
            clip_fraction,
            advantage_mean,
            return_mean,
            value_mean,
            old_logprob_mean,
            new_logprob_mean,
            grad_norm,
        )
        return (
            loss_value,
            float(kl_loss.item()),
            batch_ratio_max,
            batch_ratio_min,
            clip_fraction,
            grad_norm,
        )


@_trainer_base.TrainerRegistry.register(trainer_type="alpagym_flow_ppo")
class AlpagymFlowPPOTrainer(AlpagymPPOTrainer):
    """RLinf Flow-PPO over one selected Flow-SDE transition per decision.

    Policy packages retain the selected transition's elementwise Gaussian
    log-probabilities. The trainer sums the complete transition density into
    one joint chunk log-probability before applying the PPO ratio and clip.
    """

    _flow_chunk_density = True

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize and verify the source-pinned Flow-PPO loss contract."""
        super().__init__(config=config, parallel_dims=parallel_dims, **kwargs)
        expected = {
            "ratio_clip_low": (self._grpo_ratio_clip_low, 0.2),
            "ratio_clip_high": (self._grpo_ratio_clip_high, 0.28),
            "dual_clip_ratio": (self._dual_clip_ratio, 3.0),
            "value_loss_coef": (self._value_loss_coef, 1.0),
            "value_clip_range": (self._value_clip_range, 0.2),
            "value_huber_delta": (self._value_huber_delta, 10.0),
            "gamma": (self._gamma, 0.99),
            "gae_lambda": (self._gae_lambda, 0.95),
        }
        mismatches = [
            f"{name}={actual!r} (expected {required!r})"
            for name, (actual, required) in expected.items()
            if actual != required
        ]
        if mismatches:
            raise ValueError(
                "alpagym_flow_ppo differs from the pinned RLinf contract: "
                + ", ".join(mismatches)
            )
        if not self._normalize_advantages:
            raise ValueError("alpagym_flow_ppo requires normalized advantages")
        if self._kl_beta != 0.0 or self._reference_model is not None:
            raise ValueError("alpagym_flow_ppo does not use reference-model KL")

    def _compute_gae(self, step_samples: list[Any]) -> tuple[list[float], list[float]]:
        """Compute variable-duration GAE on the VLA's nominal 0.5 s clock.

        RLinf's configured ``gamma`` and ``lambda`` are per policy decision,
        not per 50 Hz controller tick.  A nominal transition is 25 controller
        ticks; delayed plan installation may make the realized interval longer.
        We therefore time-scale gamma by ``duration / 25`` while applying
        lambda once per sampled policy transition.
        """
        nominal_duration_ticks = 25.0
        transitions: list[tuple[float, int, bool, bool, float, float | None]] = []
        valid_indices: list[int] = []
        for index, step in enumerate(step_samples):
            signal = step.training_signal
            if bool(signal.is_padding.item()):
                continue
            terminated = bool(
                _require_ppo_signal(signal.terminateds, "terminateds").item()
            )
            truncated = (
                False if signal.truncateds is None else bool(signal.truncateds.item())
            )
            if terminated and truncated:
                raise ValueError(
                    "Flow-PPO transition cannot be both terminated and truncated"
                )
            primitive_rewards = _require_ppo_signal(
                signal.primitive_rewards, "primitive_rewards"
            ).reshape(-1)
            primitive_mask = _require_ppo_signal(
                signal.primitive_reward_mask, "primitive_reward_mask"
            ).reshape(-1)
            duration = int(
                _require_ppo_signal(signal.duration_ticks, "duration_ticks").item()
            )
            if not 1 <= duration <= primitive_rewards.numel():
                raise ValueError(
                    "Flow-PPO duration must select a non-empty realized reward prefix"
                )
            expected_mask = torch.arange(primitive_rewards.numel()) < duration
            if not torch.equal(primitive_mask.cpu(), expected_mask):
                raise ValueError(
                    "Flow-PPO primitive reward mask must match duration_ticks"
                )
            selected_rewards = primitive_rewards[:duration]
            if not torch.isfinite(selected_rewards).all():
                raise ValueError("Flow-PPO primitive rewards must be finite")
            transition_reward = selected_rewards.to(dtype=torch.float64).sum()
            transported_reward = _require_ppo_signal(signal.rewards, "rewards").reshape(
                -1
            )
            if (
                transported_reward.numel() != 1
                or not torch.isfinite(transported_reward).all()
            ):
                raise ValueError("Flow-PPO requires one finite transition reward")
            if not torch.isclose(
                transition_reward,
                transported_reward[0].to(dtype=torch.float64),
                rtol=1.0e-6,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "Flow-PPO transition reward differs from its realized tick rewards"
                )
            value = float(_require_ppo_signal(signal.old_values, "old_values").item())
            bootstrap = (
                None
                if signal.bootstrap_values is None
                else float(signal.bootstrap_values.item())
            )
            transitions.append(
                (
                    float(transition_reward.item()),
                    duration,
                    terminated,
                    truncated,
                    value,
                    bootstrap,
                )
            )
            valid_indices.append(index)

        advantages = [0.0 for _ in step_samples]
        returns = [0.0 for _ in step_samples]
        last_gae = 0.0
        for valid_pos in reversed(range(len(valid_indices))):
            reward, duration, terminated, truncated, value, bootstrap = transitions[
                valid_pos
            ]
            if bootstrap is None:
                bootstrap = (
                    transitions[valid_pos + 1][4]
                    if valid_pos + 1 < len(transitions)
                    else 0.0
                )
            gamma_duration = self._gamma ** (duration / nominal_duration_ticks)
            delta = reward + (0.0 if terminated else gamma_duration) * bootstrap - value
            continues = (
                not terminated and not truncated and valid_pos + 1 < len(transitions)
            )
            last_gae = delta + (
                gamma_duration * self._gae_lambda * last_gae if continues else 0.0
            )
            sample_index = valid_indices[valid_pos]
            advantages[sample_index] = float(last_gae)
            returns[sample_index] = float(last_gae + value)
        return advantages, returns

    def _forward_with_reference_and_value(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Replay Flow density under the same CUDA bf16 path as rollout."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            return super()._forward_with_reference_and_value(model_inputs)
