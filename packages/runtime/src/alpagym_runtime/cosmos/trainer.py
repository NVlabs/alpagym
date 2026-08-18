# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GRPO trainer for AlpaGym closed-loop training.

The trainer is policy-agnostic: per-policy tokenizer resolution and data
packer construction are looked up via the
``alpagym_runtime.policies.registry`` bundle for the configured
policy kind string. The cosmos entrypoint dispatches the data packer the same way.
"""

import copy
import logging
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
    compute_kl_penalty,
    compute_ppo_surrogate,
    compute_token_ppo_surrogate,
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
    ) -> dict[str, float | int]:
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
        samples, advantages = self._prepare_training_data(rollouts)
        if not samples:
            raise ValueError(
                "AlpaGym trainer step has no trainable samples after filtering: "
                f"current_step={current_step}"
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
        self.lr_schedulers.step()
        checkpoint_config = getattr(self.config.train, "ckpt", None)
        final_checkpoint_fallback = (
            bool(getattr(checkpoint_config, "enable_checkpoint", False))
            and current_step == total_steps
        )
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
            "train/learning_rate": float(self.lr_schedulers.get_last_lr()[0]),
            "train/num_batches": num_batches,
            "train/ratio_max": ratio_max,
            "train/ratio_min": ratio_min,
            "train/clip_fraction": clip_fraction_sum / num_batches
            if num_batches
            else 0.0,
            "train/grad_norm": grad_norm_sum / num_batches if num_batches else 0.0,
            "train/iteration_time": 0.0,
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
        for rollout in rollouts:
            step_samples = self.data_packer.get_policy_input(
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
        minibatch = self.data_packer.policy_collate_fn(minibatch_samples)
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
        backward_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.train_stream):
            self.train_stream.wait_stream(backward_stream)
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

        Mirrors the upstream ``GRPOTrainer.step_training`` checkpoint block:
        exports HuggingFace-compatible safetensors when
        ``config.train.ckpt.export_safetensors`` is set (always on the final
        step), then writes the cosmos resume checkpoint (model + optimizer +
        scheduler + ``remain_samples_num``) via ``self.ckpt_manager``.

        The inherited ``ckpt_manager`` and ``export_safetensors`` come from
        ``LLMTrainer``; ``output_dir`` / ``ckpt`` / ``param_dtype`` come from
        ``cosmos_config.toml``.
        """
        is_last_step = current_step == total_steps
        if is_last_step or self.config.train.ckpt.export_safetensors:
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

        logger.info("[Policy] Saving cosmos checkpoint at step %d", current_step)
        self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.lr_schedulers,
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
        """Create the frozen initial-policy KL reference on first use.

        Stores the reference on a private attribute (not registered as a
        submodule, since `Trainer` is an ABC, not an `nn.Module`). The
        inherited ``GRPOTrainer.weight_resume`` loads the configured policy
        before loading a Cosmos resume checkpoint and records those initial
        weights in ``self.reference_state_dict``. Building the teacher from
        that state, rather than from the possibly-resumed live model, keeps
        ``reference_reset_interval=0`` anchored to the same initial policy
        across process restarts. We drive KL via ``teacher_model=`` on the
        model forward rather than Cosmos's state-dict swapping path.
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


def _summarize_post_update_log_ratios(
    log_ratios: torch.Tensor,
    *,
    ratio_clip_low: float,
    ratio_clip_high: float,
) -> dict[str, float | int]:
    """Summarize one flat, already-masked post-update behavior-ratio pool."""
    flattened = log_ratios.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("PPO post-update diagnostics found no valid replay actions")
    if not torch.isfinite(flattened).all():
        raise FloatingPointError("PPO post-update log-ratios contain non-finite values")

    # Match the numerically bounded ratio used by the PPO objective. The
    # non-negative approximation below is (ratio - 1) - log(ratio), averaged
    # over individual valid causal actions rather than over minibatches.
    bounded_log_ratios = flattened.clamp(min=-5.0, max=5.0)
    ratios = bounded_log_ratios.exp()
    quantiles = torch.quantile(
        ratios,
        torch.tensor([0.01, 0.5, 0.99], dtype=ratios.dtype),
    )
    clipped = (ratios < 1.0 - ratio_clip_low) | (ratios > 1.0 + ratio_clip_high)
    approx_kl = torch.expm1(bounded_log_ratios) - bounded_log_ratios
    return {
        "train/post_update_valid_tokens": int(ratios.numel()),
        "train/post_update_ratio_p01": float(quantiles[0].item()),
        "train/post_update_ratio_p50": float(quantiles[1].item()),
        "train/post_update_ratio_p99": float(quantiles[2].item()),
        "train/post_update_clip_fraction": float(clipped.float().mean().item()),
        "train/post_update_approx_kl": float(approx_kl.mean().item()),
    }


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

    def __init__(
        self,
        config: _cosmos_config.Config,
        parallel_dims: _parallelism.ParallelDims,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO-specific hyperparameters after the shared GRPO setup."""
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
        """Flatten rollouts and compute PPO advantages/returns inside the trainer."""
        samples: list[Any] = []
        advantages: list[float] = []
        padding: list[bool] = []
        per_rollout_ranges: list[tuple[int, int]] = []
        for rollout in rollouts:
            start = len(samples)
            step_samples = self.data_packer.get_policy_input(
                rollout.prompt,
                rollout.completion,
                n_ignore_prefix_tokens=rollout.n_ignore_prefix_tokens,
            )
            rollout_weight_version = int(rollout.weight_version)
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
                samples.append(_with_ppo_targets(step, advantage=advantage, ret=ret))
                advantages.append(0.0 if is_padding else advantage)
                padding.append(is_padding)
            per_rollout_ranges.append((start, len(samples)))

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        if self._normalize_advantages and advantage_tensor.numel() > 0:
            valid_mask = ~torch.tensor(padding, dtype=torch.bool)
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

    def _post_update_diagnostics(
        self,
        samples: list[Any],
    ) -> dict[str, float | int]:
        """Re-score frozen behavior actions once under the final updated policy.

        This is a pure no-grad forward pass: it never calls backward, gradient
        reduction, an optimizer, or the scheduler. Token policies contribute
        one observation per valid causal token; row padding and non-causal
        early-terminal shadow tails are excluded before all reductions.
        """
        if not samples:
            raise ValueError("PPO post-update diagnostics require replay samples")
        data_packer = self.data_packer
        if data_packer is None:
            raise RuntimeError("PPO post-update diagnostics require a data packer")

        valid_log_ratios: list[torch.Tensor] = []
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
                    old_logprobs = signal.old_logprobs.to(self.device)
                    (
                        new_logprobs,
                        _kl_div,
                        _values,
                        new_token_logprobs,
                        old_token_logprobs,
                        token_causality_mask,
                    ) = self._forward_with_reference_and_value(minibatch.model_inputs)

                    if tuple(new_logprobs.shape) != tuple(old_logprobs.shape):
                        raise ValueError(
                            "PPO post-update new/old scalar log-probability shapes differ: "
                            f"{tuple(new_logprobs.shape)} != "
                            f"{tuple(old_logprobs.shape)}"
                        )
                    if not torch.isfinite(new_logprobs).all():
                        raise FloatingPointError(
                            "PPO post-update model returned non-finite log-probabilities"
                        )
                    if not torch.isfinite(old_logprobs).all():
                        raise FloatingPointError(
                            "PPO post-update replay has non-finite old log-probabilities"
                        )

                    if new_token_logprobs is None:
                        if token_causality_mask is not None:
                            raise ValueError(
                                "scalar PPO replay unexpectedly carries a token "
                                "causality mask"
                            )
                        log_ratios = new_logprobs - old_logprobs
                        valid_mask = ~is_padding
                    else:
                        if old_token_logprobs is None or token_causality_mask is None:
                            raise ValueError(
                                "token-level PPO post-update diagnostics require old "
                                "token log-probabilities and a causality mask"
                            )
                        if tuple(new_token_logprobs.shape) != tuple(
                            old_token_logprobs.shape
                        ):
                            raise ValueError(
                                "PPO post-update new/old token log-probability shapes "
                                f"differ: {tuple(new_token_logprobs.shape)} != "
                                f"{tuple(old_token_logprobs.shape)}"
                            )
                        if tuple(token_causality_mask.shape) != tuple(
                            new_token_logprobs.shape
                        ):
                            raise ValueError(
                                "PPO post-update token causality mask shape differs from "
                                "token log-probabilities: "
                                f"{tuple(token_causality_mask.shape)} != "
                                f"{tuple(new_token_logprobs.shape)}"
                            )
                        if token_causality_mask.dtype is not torch.bool:
                            raise TypeError(
                                "PPO post-update token causality mask must have dtype bool"
                            )
                        if not torch.allclose(
                            old_token_logprobs.sum(dim=-1),
                            old_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ):
                            raise ValueError(
                                "sum(old_token_logprobs) does not match replay "
                                "old_logprobs"
                            )
                        if not torch.allclose(
                            new_token_logprobs.sum(dim=-1),
                            new_logprobs,
                            rtol=1.0e-5,
                            atol=1.0e-5,
                        ):
                            raise ValueError(
                                "sum(new token log-probabilities) does not match model "
                                "log_probs"
                            )
                        log_ratios = new_token_logprobs - old_token_logprobs
                        valid_mask = (~is_padding).unsqueeze(-1).expand_as(log_ratios)
                        valid_mask = valid_mask & token_causality_mask

                    selected = log_ratios[valid_mask]
                    if selected.numel() > 0:
                        valid_log_ratios.append(
                            selected.detach().to(device="cpu", dtype=torch.float32)
                        )
        finally:
            self.model.train(was_training)

        metrics = _summarize_post_update_log_ratios(
            torch.cat(valid_log_ratios) if valid_log_ratios else torch.empty(0),
            ratio_clip_low=self._grpo_ratio_clip_low,
            ratio_clip_high=self._grpo_ratio_clip_high,
        )
        logger.info(
            "AlpaGym PPO post-update diagnostics valid_tokens=%d "
            "ratio_p01=%.6f ratio_p50=%.6f ratio_p99=%.6f "
            "clip_fraction=%.6f approx_kl=%.6f",
            int(metrics["train/post_update_valid_tokens"]),
            float(metrics["train/post_update_ratio_p01"]),
            float(metrics["train/post_update_ratio_p50"]),
            float(metrics["train/post_update_ratio_p99"]),
            float(metrics["train/post_update_clip_fraction"]),
            float(metrics["train/post_update_approx_kl"]),
        )
        return metrics

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
    ) -> tuple[float, float, float, float, float, float]:
        """Train actor and value head on one step-level PPO minibatch."""
        minibatch = self.data_packer.policy_collate_fn(minibatch_samples)
        signal = minibatch.training_signal
        is_padding = signal.is_padding.to(self.device)
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
            new_token_logprobs,
            old_token_logprobs,
            token_causality_mask,
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
        if new_token_logprobs is None:
            policy_loss, ratio = compute_ppo_surrogate(
                new_logprobs,
                old_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                is_padding=is_padding,
            )
        else:
            assert old_token_logprobs is not None
            if token_causality_mask is None:
                raise ValueError(
                    "token-level PPO replay is missing token_causality_mask"
                )
            if not torch.allclose(
                old_token_logprobs.sum(dim=-1),
                old_logprobs,
                rtol=1.0e-5,
                atol=1.0e-5,
            ):
                raise ValueError(
                    "sum(old_token_logprobs) does not match replay old_logprobs"
                )
            if not torch.allclose(
                new_token_logprobs.sum(dim=-1),
                new_logprobs,
                rtol=1.0e-5,
                atol=1.0e-5,
            ):
                raise ValueError(
                    "sum(new token log-probabilities) does not match model log_probs"
                )
            policy_loss, ratio = compute_token_ppo_surrogate(
                new_token_logprobs,
                old_token_logprobs,
                advantages,
                ratio_clip_low=self._grpo_ratio_clip_low,
                ratio_clip_high=self._grpo_ratio_clip_high,
                is_padding=is_padding,
                token_causality_mask=token_causality_mask,
            )
        kl_loss = compute_kl_penalty(
            kl_div,
            is_padding,
            kl_beta=self._kl_beta,
            device=self.device,
        )
        value_loss = compute_value_loss(
            values,
            returns,
            is_padding,
            old_values=old_values,
            value_clip_range=self._value_clip_range,
        )
        loss = policy_loss + kl_loss + self._value_loss_coef * value_loss

        self.optimizers.zero_grad()
        loss.backward()
        grad_norm = self.all_reduce_states(inter_policy_nccl)

        return self._ppo_minibatch_metrics(
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_loss=kl_loss,
            ratio=ratio,
            is_padding=is_padding,
            advantages=advantages,
            returns=returns,
            values=values,
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            token_causality_mask=token_causality_mask,
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
        torch.Tensor | None,
    ]:
        """Run actor-critic forward with optional reference model for KL."""
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        old_token_logprobs = forward_kwargs.pop("old_token_logprobs", None)
        token_causality_mask = forward_kwargs.pop("token_causality_mask", None)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        new_token_logprobs = result.get("token_log_probs")
        if (new_token_logprobs is None) != (old_token_logprobs is None):
            raise ValueError(
                "token-level PPO requires both model token_log_probs and replay "
                "old_token_logprobs"
            )
        return (
            result["log_probs"],
            result.get("kl_div"),
            result["values"].reshape(-1),
            new_token_logprobs,
            old_token_logprobs,
            token_causality_mask,
        )

    def _ppo_minibatch_metrics(
        self,
        loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        ratio: torch.Tensor,
        is_padding: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        token_causality_mask: torch.Tensor | None,
        grad_norm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute PPO diagnostics over valid rows and return trainer-loop metrics."""
        valid_mask = ~is_padding
        loss_value = float(loss.item())
        with torch.no_grad():
            ratio_valid_mask = valid_mask
            if ratio.ndim == 2:
                ratio_valid_mask = valid_mask.unsqueeze(-1).expand_as(ratio)
                if token_causality_mask is not None:
                    ratio_valid_mask = ratio_valid_mask & token_causality_mask
            valid_ratio = ratio[ratio_valid_mask]
            if valid_ratio.numel() == 0:
                clip_fraction = 0.0
                batch_ratio_max = 1.0
                batch_ratio_min = 1.0
                advantage_mean = 0.0
                return_mean = 0.0
                value_mean = 0.0
            else:
                clipped = (valid_ratio < 1.0 - self._grpo_ratio_clip_low) | (
                    valid_ratio > 1.0 + self._grpo_ratio_clip_high
                )
                clip_fraction = float(clipped.float().mean().item())
                batch_ratio_max = float(valid_ratio.max().item())
                batch_ratio_min = float(valid_ratio.min().item())
                advantage_mean = float(advantages[valid_mask].mean().item())
                return_mean = float(returns[valid_mask].mean().item())
                value_mean = float(values[valid_mask].mean().item())
        logger.info(
            "AlpaGym PPO minibatch rows=%d valid_rows=%d loss=%.6f "
            "policy_loss=%.6f value_loss=%.6f kl_loss=%.6f ratio_min=%.6f "
            "ratio_max=%.6f clip_fraction=%.6f advantage_mean=%.6f "
            "return_mean=%.6f value_mean=%.6f old_logprob_mean=%.6f "
            "new_logprob_mean=%.6f grad_norm=%.6f",
            int(old_logprobs.numel()),
            int(valid_mask.sum().item()),
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
