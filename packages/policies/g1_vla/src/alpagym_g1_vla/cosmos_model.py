# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos-RL lifecycle wrapper for the attested VLA Psi actor-critic."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import hashlib
import json
import re
import sys
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, cast

import torch
import torch.nn as nn
from cosmos_rl.policy.model.base import BaseModel, IdentityWeightMapper, ModelRegistry
from cosmos_rl.utils.model_config import register_local_model_config
from cosmos_rl.utils.util import cosmos_default_dtype
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from alpagym_g1_vla.flow import (
    VLA_FLOW_IGNORE_LAST,
    VLA_FLOW_NOISE_LEVEL,
    VlaFlowSchedule,
)
from alpagym_g1_vla.model import VlaPsiActorCritic
from alpagym_g1_vla.normalization import VlaQ99Normalizer
from alpagym_g1_vla.provenance import (
    ARGV_SHA256,
    BASE_VLM_TREE_SHA256,
    CHECKPOINT_STEP,
    MODEL_ID,
    MODEL_SHA256,
    PSI_SOURCE_TREE_SHA256,
    RUN_CONFIG_SHA256,
    STATS_SHA256,
    VlaSourceBundle,
    canonical_tree_snapshot,
)

VLA_PSI_PPO_MODEL_TYPE = "g1_vla_psi_ppo"
_NATIVE_RTC_MAX_DELAY_EXCLUSIVE = 8

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_CONFIG_REGISTERED = False
_PSI_IMPORT_LOCK = RLock()


class _AttestedPsiSourceLoader(importlib.abc.Loader):
    """Compile one verified Psi source file without consulting bytecode caches."""

    def __init__(
        self,
        *,
        source_root: Path,
        source_path: Path,
        expected_sha256: str,
    ) -> None:
        self.source_root = source_root
        self.source_path = source_path
        self.expected_sha256 = expected_sha256

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        """Use Python's default module allocation."""
        del spec
        return None

    def exec_module(self, module: Any) -> None:
        """Compile exactly the attested source bytes and execute them."""
        source_path = self.source_path.resolve(strict=True)
        if self.source_path.is_symlink() or not source_path.is_relative_to(
            self.source_root
        ):
            raise ImportError(
                f"Psi source escaped the attested tree: {self.source_path}"
            )
        source = source_path.read_bytes()
        actual_sha256 = hashlib.sha256(source).hexdigest()
        if actual_sha256 != self.expected_sha256:
            raise ImportError(
                f"Psi source changed after attestation: {source_path}; expected "
                f"{self.expected_sha256}, got {actual_sha256}"
            )
        module.__cached__ = None
        code = compile(source, str(source_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _AttestedPsiSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve every ``psi`` module only from an attested ``.py`` manifest."""

    def __init__(
        self,
        *,
        source_root: Path,
        expected_sha256_by_path: Mapping[str, str],
    ) -> None:
        self.source_root = source_root
        self.expected_sha256_by_path = dict(expected_sha256_by_path)

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Return a source-only spec, a closed namespace, or fail closed."""
        del path, target
        if fullname != "psi" and not fullname.startswith("psi."):
            return None
        relative_parts = fullname.split(".")[1:]
        relative_stem = "/".join(relative_parts)
        package_relative = (
            f"{relative_stem}/__init__.py" if relative_stem else "__init__.py"
        )
        package_sha256 = self.expected_sha256_by_path.get(package_relative)
        if package_sha256 is not None:
            package_path = self.source_root / package_relative
            return importlib.util.spec_from_file_location(
                fullname,
                package_path,
                loader=_AttestedPsiSourceLoader(
                    source_root=self.source_root,
                    source_path=package_path,
                    expected_sha256=package_sha256,
                ),
                submodule_search_locations=[str(package_path.parent)],
            )

        if relative_stem:
            module_relative = f"{relative_stem}.py"
            module_sha256 = self.expected_sha256_by_path.get(module_relative)
            if module_sha256 is not None:
                module_path = self.source_root / module_relative
                return importlib.util.spec_from_file_location(
                    fullname,
                    module_path,
                    loader=_AttestedPsiSourceLoader(
                        source_root=self.source_root,
                        source_path=module_path,
                        expected_sha256=module_sha256,
                    ),
                )

        namespace_prefix = f"{relative_stem}/" if relative_stem else ""
        if relative_stem and any(
            relative_path.startswith(namespace_prefix)
            for relative_path in self.expected_sha256_by_path
        ):
            namespace_path = self.source_root.joinpath(*relative_parts)
            spec = importlib.machinery.ModuleSpec(
                fullname, loader=None, is_package=True
            )
            spec.submodule_search_locations = [str(namespace_path)]
            return spec

        raise ModuleNotFoundError(
            f"Psi module {fullname!r} is absent from the attested source manifest"
        )


class _PsiRuntimeProtocol(Protocol):
    """Typed module ownership exposed by the external Psi runtime."""

    vlm_model: nn.Module
    action_header: nn.Module


class VlaPsiPPOConfig(PretrainedConfig):
    """HF-shaped, content-addressed configuration for VLA Psi PPO."""

    model_type = VLA_PSI_PPO_MODEL_TYPE
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        policy_eval_root: str,
        bundle_model_id: str,
        checkpoint_step: int,
        model_sha256: str,
        run_config_sha256: str,
        argv_sha256: str,
        stats_sha256: str,
        base_vlm_tree_sha256: str,
        psi_source_tree_sha256: str,
        normalization_type: str,
        flow_model_timesteps: list[float] | tuple[float, ...],
        flow_sigmas: list[float] | tuple[float, ...],
        flow_schedule_sha256: str,
        flow_noise_level: float,
        flow_ignore_last: bool,
        rtc_max_delay_exclusive: int,
        vlm_hidden_dim: int,
        critic_hidden_sizes: list[int] | tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        """Validate the exact source, weights, statistics, and PPO schedule."""
        super().__init__(**kwargs)
        expected_pins = {
            "bundle_model_id": (bundle_model_id, MODEL_ID),
            "checkpoint_step": (int(checkpoint_step), CHECKPOINT_STEP),
            "model_sha256": (model_sha256, MODEL_SHA256),
            "run_config_sha256": (run_config_sha256, RUN_CONFIG_SHA256),
            "argv_sha256": (argv_sha256, ARGV_SHA256),
            "stats_sha256": (stats_sha256, STATS_SHA256),
            "base_vlm_tree_sha256": (
                base_vlm_tree_sha256,
                BASE_VLM_TREE_SHA256,
            ),
            "psi_source_tree_sha256": (
                psi_source_tree_sha256,
                PSI_SOURCE_TREE_SHA256,
            ),
        }
        for label, (actual, expected) in expected_pins.items():
            if actual != expected:
                raise ValueError(
                    f"VLA {label} does not match the pinned bundle: "
                    f"expected {expected!r}, got {actual!r}"
                )
        for label in (
            "model_sha256",
            "run_config_sha256",
            "argv_sha256",
            "stats_sha256",
            "base_vlm_tree_sha256",
            "psi_source_tree_sha256",
            "flow_schedule_sha256",
        ):
            if _SHA256_PATTERN.fullmatch(str(locals()[label])) is None:
                raise ValueError(f"VLA {label} must be a lowercase SHA-256")
        if normalization_type != "bounds_q99":
            raise ValueError("VLA v1 requires normalization_type='bounds_q99'")
        if float(flow_noise_level) != VLA_FLOW_NOISE_LEVEL:
            raise ValueError(f"VLA v1 requires flow_noise_level={VLA_FLOW_NOISE_LEVEL}")
        if flow_ignore_last is not VLA_FLOW_IGNORE_LAST:
            raise ValueError("VLA v1 requires flow_ignore_last=true")
        if (
            isinstance(rtc_max_delay_exclusive, bool)
            or not isinstance(rtc_max_delay_exclusive, int)
            or rtc_max_delay_exclusive != _NATIVE_RTC_MAX_DELAY_EXCLUSIVE
        ):
            raise ValueError(
                "VLA pinned checkpoint requires exclusive RTC max_delay="
                f"{_NATIVE_RTC_MAX_DELAY_EXCLUSIVE}"
            )
        if vlm_hidden_dim <= 0:
            raise ValueError("VLA vlm_hidden_dim must be positive")
        hidden_sizes = [int(width) for width in critic_hidden_sizes]
        if not hidden_sizes or min(hidden_sizes) <= 0:
            raise ValueError("VLA critic_hidden_sizes must be positive")

        with torch.device("cpu"), cosmos_default_dtype(torch.float32):
            schedule = VlaFlowSchedule(
                model_timesteps=torch.tensor(flow_model_timesteps),
                sigmas=torch.tensor(flow_sigmas),
            )
        if schedule.sha256 != flow_schedule_sha256:
            raise ValueError(
                "VLA flow_schedule_sha256 does not identify the configured grid"
            )

        self.policy_eval_root = str(Path(policy_eval_root).expanduser())
        self.bundle_model_id = bundle_model_id
        self.checkpoint_step = int(checkpoint_step)
        self.model_sha256 = model_sha256
        self.run_config_sha256 = run_config_sha256
        self.argv_sha256 = argv_sha256
        self.stats_sha256 = stats_sha256
        self.base_vlm_tree_sha256 = base_vlm_tree_sha256
        self.psi_source_tree_sha256 = psi_source_tree_sha256
        self.normalization_type = normalization_type
        self.flow_model_timesteps = [float(value) for value in flow_model_timesteps]
        self.flow_sigmas = [float(value) for value in flow_sigmas]
        self.flow_schedule_sha256 = flow_schedule_sha256
        self.flow_noise_level = float(flow_noise_level)
        self.flow_ignore_last = flow_ignore_last
        self.rtc_max_delay_exclusive = rtc_max_delay_exclusive
        self.vlm_hidden_dim = int(vlm_hidden_dim)
        self.critic_hidden_sizes = hidden_sizes

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        """Skip text-generation defaults for this required-argument VLA config."""
        return {}


class VlaPsiPPOModel(BaseModel):
    """Single-GPU Cosmos wrapper around :class:`VlaPsiActorCritic`."""

    actor_critic: VlaPsiActorCritic

    def __init__(self, hf_config: VlaPsiPPOConfig) -> None:
        """Build the attested Psi architecture without allocating 2B weights."""
        super().__init__(hf_config)
        self.config = hf_config
        self.source_bundle = VlaSourceBundle.verify(hf_config.policy_eval_root)
        if self.source_bundle.model_root.name != hf_config.bundle_model_id:
            raise ValueError("Verified VLA model root does not match bundle_model_id")

        run_config = _read_attested_run_config(self.source_bundle)
        _validate_training_contract(run_config, hf_config)
        psi_model = _construct_psi_architecture(
            source_bundle=self.source_bundle,
            run_config=run_config,
            vlm_hidden_dim=hf_config.vlm_hidden_dim,
        )
        psi_runtime = cast(_PsiRuntimeProtocol, psi_model)
        for parameter in psi_runtime.vlm_model.parameters():
            parameter.requires_grad_(False)
        for parameter in psi_runtime.action_header.parameters():
            parameter.requires_grad_(True)
        if any(parameter.device.type != "meta" for parameter in psi_model.parameters()):
            raise RuntimeError(
                "VLA Psi must be constructed inside Cosmos's meta init context"
            )

        with torch.device("cpu"), cosmos_default_dtype(torch.float32):
            schedule = VlaFlowSchedule(
                model_timesteps=torch.tensor(hf_config.flow_model_timesteps),
                sigmas=torch.tensor(hf_config.flow_sigmas),
            )
            normalizer = VlaQ99Normalizer.from_stats_file(self.source_bundle.stats_path)
        self.actor_critic = VlaPsiActorCritic(
            psi_model=psi_model,
            schedule=schedule,
            noise_level=hf_config.flow_noise_level,
            normalizer=normalizer,
            vlm_hidden_dim=hf_config.vlm_hidden_dim,
            rtc_max_delay_exclusive=hf_config.rtc_max_delay_exclusive,
            critic_hidden_sizes=tuple(hf_config.critic_hidden_sizes),
        )
        self.actor_critic.critic.float()
        self._weights_loaded = False
        self._critic_initialized = False

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the model type registered with Transformers and Cosmos."""
        return [VLA_PSI_PPO_MODEL_TYPE]

    @property
    def parallelize_fn(self) -> tuple[Callable[..., tuple[None, None]], None]:
        """Return the fail-closed identity parallelizer used by v1."""
        return _identity_parallelize, None

    def apply_pipeline_split(self, pp_rank: int, pp_size: int) -> None:
        """Reject pipeline parallelism; v1 supports one complete GPU model."""
        del pp_rank
        if pp_size != 1:
            raise NotImplementedError("VLA Psi v1 does not support PP")

    def check_cp_compatible(self, cp_size: int, tp_size: int) -> None:
        """Reject context or tensor parallel layouts larger than one."""
        if cp_size != 1 or tp_size != 1:
            raise NotImplementedError("VLA Psi v1 does not support CP or TP")

    def check_tp_compatible(self, tp_size: int) -> None:
        """Reject tensor parallel layouts larger than one."""
        if tp_size != 1:
            raise NotImplementedError("VLA Psi v1 does not support TP")

    def post_to_empty_hook(self, cosmos_config: Any) -> None:
        """Initialize trainable state after Cosmos materializes the model.

        Cosmos passes its top-level ``Config`` here and performs the actual HF
        load later through :meth:`load_hf_weights`, with the resolved
        ``ParallelDims`` as a separate argument.  Keeping those lifecycle
        phases separate matches the native Cosmos model contract.
        """
        del cosmos_config
        if not self._critic_initialized:
            self.actor_critic.critic.prefix_projection.reset_parameters()
            self.actor_critic.critic.value_head._init_weights("relu")
            self._critic_initialized = True
        self.actor_critic.critic.float()
        psi_runtime = cast(_PsiRuntimeProtocol, self.actor_critic.psi_model)
        for parameter in psi_runtime.vlm_model.parameters():
            parameter.requires_grad_(False)
        for parameter in psi_runtime.action_header.parameters():
            parameter.requires_grad_(True)
        for parameter in self.actor_critic.critic.parameters():
            parameter.requires_grad_(True)
        self.actor_critic.psi_model.device = self.current_device()
        self.actor_critic.train()

    def get_position_ids(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return empty IDs because v1 does not expose Cosmos context parallelism."""
        del kwargs
        empty = torch.empty(0, dtype=torch.long, device=self.current_device())
        return empty, empty, 0

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: Any,
        device: torch.device,
        revision: str | None = None,
    ) -> None:
        """Stream the verified safetensors checkpoint into materialized Psi tensors."""
        del model_name_or_path, revision
        _require_single_gpu(parallel_dims)
        device = _canonical_torch_device(device)
        if self._weights_loaded:
            return
        targets = self.actor_critic.psi_model.state_dict(keep_vars=True)
        if any(tensor.device.type == "meta" for tensor in targets.values()):
            raise RuntimeError("VLA Psi weights must be materialized before loading")
        if any(tensor.device != device for tensor in targets.values()):
            raise RuntimeError("VLA Psi target tensors differ from the load device")

        tied_target = "vlm_model.lm_head.weight"
        tied_source = "vlm_model.model.language_model.embed_tokens.weight"
        with safe_open(
            self.source_bundle.checkpoint_path,
            framework="pt",
            device="cpu",
        ) as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            expected_keys = set(targets)
            missing = expected_keys - checkpoint_keys
            extra = checkpoint_keys - expected_keys
            if missing == {tied_target} and tied_source in checkpoint_keys:
                missing.clear()
            if missing or extra:
                raise RuntimeError(
                    "VLA checkpoint does not strictly match Psi architecture; "
                    f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
                )
            for name, target in targets.items():
                source_name = tied_source if name == tied_target else name
                source_shape = tuple(checkpoint.get_slice(source_name).get_shape())
                if source_shape != tuple(target.shape):
                    raise RuntimeError(
                        f"VLA tensor shape mismatch for {name}: "
                        f"checkpoint={source_shape}, model={tuple(target.shape)}"
                    )
            with torch.no_grad():
                for name, target in targets.items():
                    source_name = tied_source if name == tied_target else name
                    source = checkpoint.get_tensor(source_name)
                    target.copy_(source.to(device=device, dtype=target.dtype))
        self._weights_loaded = True

    def separate_model_parts(self) -> list[nn.Module]:
        """Expose action and critic as the two pinned optimizer groups."""
        psi = cast(_PsiRuntimeProtocol, self.actor_critic.psi_model)
        return [psi.action_header, self.actor_critic.critic]

    @classmethod
    def from_pretrained(
        cls,
        hf_config: Any,
        model_name_or_path: str,
        max_position_embeddings: int | None = None,
    ) -> VlaPsiPPOModel:
        """Construct architecture only; Cosmos materializes and loads it later."""
        del model_name_or_path, max_position_embeddings
        if not isinstance(hf_config, VlaPsiPPOConfig):
            raise TypeError("VLA Cosmos model requires VlaPsiPPOConfig")
        return cls(hf_config)

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """Return zero because this custom VLA has no Cosmos FLOP estimator."""
        del seq_len
        return 0, 0

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor | None]:
        """Delegate PPO replay scoring without changing the policy ABI."""
        return self.actor_critic(**kwargs)

    def forward_values(self, **kwargs: Any) -> torch.Tensor:
        """Delegate boundary-value evaluation without changing the policy ABI."""
        return self.actor_critic.forward_values(**kwargs)

    def clone_for_inference_lease(self) -> VlaPsiActorCritic:
        """Return the core's isolated inference snapshot."""
        return self.actor_critic.clone_for_inference_lease()


def register_vla_psi_ppo_model() -> None:
    """Idempotently register the config, local model root, and Cosmos model."""
    global _LOCAL_CONFIG_REGISTERED
    try:
        AutoConfig.register(VLA_PSI_PPO_MODEL_TYPE, VlaPsiPPOConfig)
    except ValueError as exc:
        if "is already used" not in str(exc) and "already exists" not in str(exc):
            raise
    if CONFIG_MAPPING[VLA_PSI_PPO_MODEL_TYPE] is not VlaPsiPPOConfig:
        raise ValueError(
            f"Transformers model type {VLA_PSI_PPO_MODEL_TYPE!r} is already registered"
        )
    registered = ModelRegistry._MODEL_REGISTRY.get(VLA_PSI_PPO_MODEL_TYPE)
    if registered is None:
        ModelRegistry.register_model(VlaPsiPPOModel, IdentityWeightMapper)
    elif registered is not VlaPsiPPOModel:
        raise ValueError(
            f"Cosmos model type {VLA_PSI_PPO_MODEL_TYPE!r} is already registered"
        )
    if not _LOCAL_CONFIG_REGISTERED:
        register_local_model_config(
            predicate=_is_attested_model_root,
            factory=_config_from_attested_model_root,
        )
        _LOCAL_CONFIG_REGISTERED = True


def load_vla_rollout_model(
    model_root: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> VlaPsiPPOModel:
    """Load the attested single-GPU wrapper for rollout inference.

    Args:
        model_root: Exact pinned VLA model directory under ``models/``.
        device: CUDA device that owns the complete model.
        dtype: Required backbone/action construction dtype. V1 accepts bf16;
            its numerical-stability critic remains float32.

    Returns:
        A fully materialized, strict-loaded evaluation wrapper.
    """
    if dtype is not torch.bfloat16:
        raise ValueError("VLA rollout v1 requires dtype=torch.bfloat16")
    if device.type != "cuda":
        raise ValueError("VLA rollout v1 requires one CUDA device")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise NotImplementedError("VLA rollout v1 is single-GPU only")

    register_vla_psi_ppo_model()
    config = _config_from_attested_model_root(str(model_root))
    with torch.device("meta"), cosmos_default_dtype(dtype):
        model = VlaPsiPPOModel(config)
    model._apply(
        lambda tensor: (
            torch.empty_like(tensor, device=device)
            if tensor.device.type == "meta"
            else tensor.to(device)
        )
    )
    parallel_dims = SimpleNamespace(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
    )
    model.post_to_empty_hook(
        SimpleNamespace(policy=SimpleNamespace(model_name_or_path=str(model_root)))
    )
    model.load_hf_weights(str(model_root), parallel_dims, device)
    model.eval()
    return model


def _is_attested_model_root(model_name_or_path: str) -> bool:
    """Match only paths shaped like the single pinned VLA model root."""
    path = Path(model_name_or_path).expanduser()
    return path.name == MODEL_ID and path.parent.name == "models"


def _config_from_attested_model_root(model_name_or_path: str) -> VlaPsiPPOConfig:
    """Create the v1 config after fully attesting a local VLA model root."""
    model_root = Path(model_name_or_path).expanduser().resolve(strict=True)
    policy_eval_root = model_root.parent.parent
    bundle = VlaSourceBundle.verify(policy_eval_root)
    if bundle.model_root != model_root:
        raise ValueError("VLA local config path differs from the verified model root")
    run_config = _read_attested_run_config(bundle)
    rtc_max_delay_exclusive = _rtc_max_delay_exclusive(run_config)
    with cosmos_default_dtype(torch.float32):
        schedule = VlaFlowSchedule(
            model_timesteps=torch.tensor(
                [1000, 889, 778, 667, 556, 445, 334, 223, 112, 1],
                dtype=torch.float32,
                device="cpu",
            ),
            sigmas=torch.tensor(
                [
                    1.0,
                    0.889,
                    0.778,
                    0.667,
                    0.556,
                    0.445,
                    0.334,
                    0.223,
                    0.112,
                    0.001,
                    0.0,
                ],
                dtype=torch.float32,
                device="cpu",
            ),
        )
    return VlaPsiPPOConfig(
        policy_eval_root=str(policy_eval_root),
        bundle_model_id=MODEL_ID,
        checkpoint_step=CHECKPOINT_STEP,
        model_sha256=MODEL_SHA256,
        run_config_sha256=RUN_CONFIG_SHA256,
        argv_sha256=ARGV_SHA256,
        stats_sha256=STATS_SHA256,
        base_vlm_tree_sha256=BASE_VLM_TREE_SHA256,
        psi_source_tree_sha256=PSI_SOURCE_TREE_SHA256,
        normalization_type="bounds_q99",
        flow_model_timesteps=schedule.model_timesteps.tolist(),
        flow_sigmas=schedule.sigmas.tolist(),
        flow_schedule_sha256=schedule.sha256,
        flow_noise_level=VLA_FLOW_NOISE_LEVEL,
        flow_ignore_last=VLA_FLOW_IGNORE_LAST,
        rtc_max_delay_exclusive=rtc_max_delay_exclusive,
        vlm_hidden_dim=2048,
        critic_hidden_sizes=[1024, 512, 256],
    )


def _read_attested_run_config(bundle: VlaSourceBundle) -> Mapping[str, Any]:
    """Read the run config whose digest was checked by bundle verification."""
    raw = json.loads(
        (bundle.model_root / "run_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, Mapping):
        raise TypeError("VLA run_config.json root must be a mapping")
    return raw


def _validate_training_contract(
    run_config: Mapping[str, Any], config: VlaPsiPPOConfig
) -> None:
    """Check architecture and normalization fields that affect checkpoint meaning."""
    model = run_config["model"]
    field = run_config["data"]["transform"]["action"]["field"]
    if not isinstance(model, Mapping) or not isinstance(field, Mapping):
        raise TypeError("VLA run config model/field entries must be mappings")
    expected = {
        "action_dim": 38,
        "action_chunk_size": 30,
        "odim": 29,
        "noise_scheduler": "flow",
        "train_diffusion_steps": 1000,
        "eval_diffusion_steps": 10,
        "view_feature_dim": config.vlm_hidden_dim,
        "use_dit": False,
    }
    for key, value in expected.items():
        if model[key] != value:
            raise ValueError(
                f"VLA run config {key!r} must be {value!r}, got {model[key]!r}"
            )
    if model.get("rtc") is not True:
        raise ValueError("VLA v1 requires an RTC-trained checkpoint")
    if _rtc_max_delay_exclusive(run_config) != config.rtc_max_delay_exclusive:
        raise ValueError(
            "VLA run config max_delay does not match the attested PPO config"
        )
    if field["action_norm_type"] != config.normalization_type:
        raise ValueError("VLA run config normalization does not match PPO config")
    if field["use_norm_mask"] is not False:
        raise ValueError("VLA v1 requires use_norm_mask=false")
    if run_config["train"]["lora"] is not False:
        raise NotImplementedError("VLA Psi v1 does not support LoRA checkpoints")


def _rtc_max_delay_exclusive(run_config: Mapping[str, Any]) -> int:
    """Read the native exclusive RTC row bound from an attested run config."""

    model = run_config.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("VLA run config model entry must be a mapping")
    if model.get("rtc") is not True:
        raise ValueError("VLA v1 requires an RTC-trained checkpoint")
    value = model.get("max_delay")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != _NATIVE_RTC_MAX_DELAY_EXCLUSIVE
    ):
        raise ValueError(
            "VLA attested run config must declare exclusive RTC max_delay="
            f"{_NATIVE_RTC_MAX_DELAY_EXCLUSIVE}"
        )
    return value


def _construct_psi_architecture(
    *,
    source_bundle: VlaSourceBundle,
    run_config: Mapping[str, Any],
    vlm_hidden_dim: int,
) -> nn.Module:
    """Import only attested Psi source and construct its full runtime on meta."""
    _import_attested_psi(
        source_bundle.psi_source_root,
        expected_tree_sha256=PSI_SOURCE_TREE_SHA256,
    )
    psi_module = importlib.import_module("psi.models.psi0")
    model_config_module = importlib.import_module("psi.config.model_psi0")
    transform_module = importlib.import_module("psi.config.transform")
    model_config = model_config_module.Psi0ModelConfig.model_validate(
        run_config["model"]
    )
    model_transform = transform_module.Psi0ModelTransform.model_validate(
        run_config["data"]["transform"]["action"]["model"]
    )

    base_vlm_root = source_bundle.model_root / "base_vlm"
    vlm_config = AutoConfig.from_pretrained(base_vlm_root, local_files_only=True)
    if vlm_config.model_type != "qwen3_vl":
        raise ValueError("Pinned VLA v1 bundle requires a qwen3_vl backbone")
    if int(vlm_config.text_config.hidden_size) != vlm_hidden_dim:
        raise ValueError("Pinned VLA VLM hidden width does not match PPO config")
    vlm_config._attn_implementation = "flash_attention_2"
    vlm_model = psi_module.Qwen3VLForConditionalGeneration(vlm_config).to(
        dtype=torch.bfloat16
    )
    psi_model = psi_module.Psi0Model(
        model_cfg=model_config,
        vlm_model=vlm_model,
    )

    with torch.device("cpu"):
        psi_model.vlm_processor = AutoProcessor.from_pretrained(
            base_vlm_root,
            use_fast=True,
            local_files_only=True,
        )
        psi_model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=model_config.train_diffusion_steps
        )
    psi_model.vlm_processor_use_fast = True
    psi_model.model_transform = model_transform
    psi_model.action_horizon = model_config.action_chunk_size
    psi_model.action_dim = model_config.action_dim
    psi_model.device = torch.device("meta")
    psi_model._vision_cache_storage = "cpu"
    psi_model._vision_cache_strict = True
    psi_model._vision_cache_capacity_bytes = 0
    psi_model._vision_cache = None
    return psi_model


def _import_attested_psi(
    psi_source_root: Path,
    *,
    expected_tree_sha256: str,
) -> None:
    """Expose one verified Psi checkout through a source-only import boundary."""
    with _PSI_IMPORT_LOCK:
        _import_attested_psi_locked(
            psi_source_root,
            expected_tree_sha256=expected_tree_sha256,
        )


def _import_attested_psi_locked(
    psi_source_root: Path,
    *,
    expected_tree_sha256: str,
) -> None:
    """Validate and install one Psi source tree while holding the import lock."""
    source_root = psi_source_root.resolve(strict=True)
    snapshot = canonical_tree_snapshot(
        source_root,
        format_name="wenhao-psi-source-tree.v1",
        suffix=".py",
    )
    if snapshot.sha256 != expected_tree_sha256:
        raise ValueError(
            "VLA Psi source tree SHA256 mismatch at import: expected "
            f"{expected_tree_sha256}, got {snapshot.sha256}"
        )
    for name, module in tuple(sys.modules.items()):
        if name != "psi" and not name.startswith("psi."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            resolved = Path(module_file).resolve(strict=True)
            if not resolved.is_relative_to(source_root):
                raise RuntimeError(
                    f"Loaded module {name!r} comes from unverified Psi source "
                    f"{resolved}"
                )
            loader = getattr(module, "__loader__", None)
            if not isinstance(loader, _AttestedPsiSourceLoader):
                raise RuntimeError(
                    f"Loaded module {name!r} did not use the attested source-only "
                    "Psi loader"
                )
            relative_path = resolved.relative_to(source_root).as_posix()
            if (
                loader.source_root != source_root
                or loader.expected_sha256
                != snapshot.file_sha256_by_path.get(relative_path)
            ):
                raise RuntimeError(
                    f"Loaded module {name!r} has stale Psi source attestation"
                )
            continue

        # ``psi.models`` and ``psi.config`` are PEP 420 namespace packages in
        # the attested delivery.  They legitimately have no ``__file__`` when a
        # second trainer/rollout model is built in one colocated process, but
        # every namespace search location must still remain inside the verified
        # Psi tree.
        namespace_paths = tuple(getattr(module, "__path__", ()))
        if not namespace_paths:
            raise RuntimeError(f"Loaded module {name!r} has no attested source path")
        resolved_paths = tuple(
            Path(path).resolve(strict=True) for path in namespace_paths
        )
        if any(not path.is_relative_to(source_root) for path in resolved_paths):
            raise RuntimeError(
                f"Loaded namespace module {name!r} comes from unverified Psi "
                f"search paths {resolved_paths}"
            )

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, _AttestedPsiSourceFinder)
    ]
    sys.meta_path.insert(
        0,
        _AttestedPsiSourceFinder(
            source_root=source_root,
            expected_sha256_by_path=snapshot.file_sha256_by_path,
        ),
    )
    importlib.invalidate_caches()
    module = importlib.import_module("psi")
    imported_file = module.__file__
    if imported_file is None:
        raise RuntimeError("Imported Psi package has no attested source path")
    module_file = Path(imported_file).resolve(strict=True)
    if not module_file.is_relative_to(source_root):
        raise RuntimeError(f"Imported Psi package comes from {module_file}")
    if not isinstance(module.__loader__, _AttestedPsiSourceLoader):
        raise RuntimeError("Imported Psi package bypassed the source-only loader")


def _require_single_gpu(parallel_dims: Any) -> None:
    """Fail closed unless every Cosmos parallel dimension is exactly one."""
    if parallel_dims is None:
        raise ValueError("VLA Psi v1 requires explicit Cosmos parallel dimensions")
    values = {
        name: int(getattr(parallel_dims, name))
        for name in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep", "world_size")
    }
    if any(value != 1 for value in values.values()):
        raise NotImplementedError(
            f"VLA Psi v1 is single-GPU only, got parallel dimensions {values}"
        )


def _canonical_torch_device(device: torch.device) -> torch.device:
    """Resolve an index-free CUDA device to this worker's concrete local device."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _identity_parallelize(
    model: nn.Module | None,
    parallel_dims: Any,
    config: Any,
    pp_loss_fn: Any = None,
) -> tuple[None, None]:
    """Validate the one-GPU contract and leave the model unsharded."""
    del model, config, pp_loss_fn
    _require_single_gpu(parallel_dims)
    return None, None
