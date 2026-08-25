# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Cosmos lifecycle around the attested VLA Psi model."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import marshal
import py_compile
import sys
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from cosmos_rl.policy.model.base import ModelRegistry
from cosmos_rl.utils.model_config import load_model_config
from safetensors.torch import save_file
from transformers import AutoConfig

import alpagym_g1_vla.cosmos_model as cosmos_model
from alpagym_g1_vla.flow import VlaFlowSchedule
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


class _TinyVisionRotaryEmbedding(nn.Module):
    """Small non-persistent vision RoPE buffer matching Qwen3-VL."""

    def __init__(self) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4.0))
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class _TinyTextRotaryEmbedding(nn.Module):
    """Small non-persistent text RoPE buffer matching Qwen3-VL's API."""

    def __init__(self) -> None:
        super().__init__()
        inv_freq, _attention_scaling = self.rope_init_fn(None, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.config = None

    @staticmethod
    def rope_init_fn(
        config: object,
        device: torch.device | None,
    ) -> tuple[torch.Tensor, float]:
        """Return deterministic inverse frequencies on the requested device."""
        del config
        return torch.tensor([1.0, 0.25, 0.0625], device=device), 1.0


class _TinyVisionModel(nn.Module):
    """Own the vision RoPE at Qwen3-VL's exact module path."""

    def __init__(self) -> None:
        super().__init__()
        self.rotary_pos_emb = _TinyVisionRotaryEmbedding()


class _TinyLanguageModel(nn.Module):
    """Own the text RoPE at Qwen3-VL's exact module path."""

    def __init__(self) -> None:
        super().__init__()
        self.rotary_emb = _TinyTextRotaryEmbedding()


class _TinyVlmCore(nn.Module):
    """Preserve Qwen3-VL's vision and language module layout."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = _TinyVisionModel()
        self.language_model = _TinyLanguageModel()


class _TinyVlm(nn.Module):
    """Small trainable VLM plus Qwen3-VL's nested model surface."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.model = _TinyVlmCore()


class _TinyPsi(nn.Module):
    """Small stand-in preserving Psi's VLM/action ownership surface."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm_model = _TinyVlm()
        self.action_header = nn.Linear(4, 4)
        self.action_header.register_parameter(
            "fixed_encoding",
            nn.Parameter(torch.ones(4), requires_grad=False),
        )


def _run_config() -> dict[str, object]:
    """Return the architecture fields checked by the wrapper."""
    return {
        "train": {"lora": False},
        "model": {
            "rtc": True,
            "max_delay": 8,
            "action_dim": 38,
            "action_chunk_size": 30,
            "odim": 29,
            "noise_scheduler": "flow",
            "train_diffusion_steps": 1000,
            "eval_diffusion_steps": 10,
            "view_feature_dim": 4,
            "use_dit": False,
        },
        "data": {
            "transform": {
                "action": {
                    "field": {
                        "action_norm_type": "bounds_q99",
                        "use_norm_mask": False,
                    }
                }
            }
        },
    }


def _schedule() -> VlaFlowSchedule:
    """Return a small valid non-uniform schedule."""
    return VlaFlowSchedule(
        model_timesteps=torch.tensor([1000.0, 500.0], device="cpu"),
        sigmas=torch.tensor([1.0, 0.5, 0.0], device="cpu"),
    )


def _config(root: Path) -> cosmos_model.VlaPsiPPOConfig:
    """Build a fully pinned test config."""
    schedule = _schedule()
    return cosmos_model.VlaPsiPPOConfig(
        policy_eval_root=str(root),
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
        flow_noise_level=0.4,
        flow_ignore_last=True,
        rtc_max_delay_exclusive=8,
        vlm_hidden_dim=4,
        critic_hidden_sizes=[8, 4],
    )


def _parallel_dims(**overrides: int) -> SimpleNamespace:
    """Build Cosmos-shaped parallel dimensions for tests."""
    values = {
        "dp_replicate": 1,
        "dp_shard": 1,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "world_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("rtc", "max_delay", "message"),
    (
        (False, 8, "RTC-trained"),
        (True, 20, "max_delay=8"),
    ),
)
def test_attested_run_config_pins_native_rtc_contract(
    rtc: bool,
    max_delay: int,
    message: str,
) -> None:
    run_config = _run_config()
    model = run_config["model"]
    assert isinstance(model, dict)
    model["rtc"] = rtc
    model["max_delay"] = max_delay
    with pytest.raises(ValueError, match=message):
        cosmos_model._rtc_max_delay_exclusive(run_config)

    valid = _run_config()
    assert cosmos_model._rtc_max_delay_exclusive(valid) == 8


def test_training_contract_accepts_direct_finetune_transform_schema(
    tmp_path: Path,
) -> None:
    run_config = _run_config()
    transform = run_config["data"]["transform"]
    assert isinstance(transform, dict)
    action = transform.pop("action")
    assert isinstance(action, dict)
    action["model"] = {}
    transform.update(action)

    cosmos_model._validate_training_contract(run_config, _config(tmp_path))


@pytest.fixture
def tiny_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]]:
    """Install a verified tiny source bundle and architecture constructor."""
    root = tmp_path / "policy_eval"
    model_root = root / "models" / MODEL_ID
    checkpoint = (
        model_root / "checkpoints" / f"ckpt_{CHECKPOINT_STEP}" / "model.safetensors"
    )
    stats = model_root / "stats" / "meta" / "stats_psi0.json"
    source = root / "src" / "psi"
    checkpoint.parent.mkdir(parents=True)
    stats.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (model_root / "run_config.json").write_text(
        json.dumps(_run_config()), encoding="utf-8"
    )
    stats.write_text(
        json.dumps(
            {
                "states": {"q01": [-1.0] * 29, "q99": [1.0] * 29},
                "action": {"q01": [-2.0] * 38, "q99": [2.0] * 38},
            }
        ),
        encoding="utf-8",
    )
    with torch.device("cpu"):
        tiny = _TinyPsi()
    state = {
        name: torch.full_like(tensor, float(index + 1))
        for index, (name, tensor) in enumerate(tiny.state_dict().items())
    }
    save_file(state, checkpoint)
    bundle = VlaSourceBundle(
        policy_eval_root=root,
        model_root=model_root,
        checkpoint_path=checkpoint,
        stats_path=stats,
        psi_source_root=source,
    )
    monkeypatch.setattr(
        cosmos_model.VlaSourceBundle,
        "verify",
        classmethod(lambda cls, policy_eval_root, **kwargs: bundle),
    )
    monkeypatch.setattr(
        cosmos_model.VlaSourceBundle,
        "verify_model_root",
        classmethod(lambda cls, model_root: bundle),
    )
    monkeypatch.setattr(
        cosmos_model,
        "_construct_psi_architecture",
        lambda **kwargs: _TinyPsi(),
    )
    cosmos_model.register_vla_psi_ppo_model()
    return root, bundle, state


def _build_materialized(
    root: Path,
) -> cosmos_model.VlaPsiPPOModel:
    """Emulate Cosmos meta construction and CPU materialization."""
    config = _config(root)
    with torch.device("meta"):
        model = cosmos_model.VlaPsiPPOModel(config)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    model._apply(
        lambda tensor: (
            torch.empty_like(tensor, device="cpu")
            if tensor.device.type == "meta"
            else tensor.to("cpu")
        )
    )
    return model


def test_meta_lifecycle_strict_load_and_trainable_ownership(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
) -> None:
    """Materialization loads exact Psi weights and initializes only new critic."""
    root, bundle, expected = tiny_source
    model = _build_materialized(root)
    # The real hook receives Cosmos's top-level Config. ParallelDims are passed
    # later to load_hf_weights by the trainer, not nested at Config.parallelism.
    cosmos_config = SimpleNamespace(
        policy=SimpleNamespace(model_name_or_path=str(bundle.model_root))
    )
    model.post_to_empty_hook(cosmos_config)
    model.load_hf_weights(str(bundle.model_root), _parallel_dims(), torch.device("cpu"))

    for name, tensor in model.actor_critic.psi_model.state_dict().items():
        assert torch.equal(tensor, expected[name])
    vlm_model = model.actor_critic.psi_model.get_submodule("vlm_model")
    action_header = model.actor_critic.psi_model.get_submodule("action_header")
    assert all(not parameter.requires_grad for parameter in vlm_model.parameters())
    assert action_header.weight.requires_grad
    assert action_header.bias.requires_grad
    assert not action_header.fixed_encoding.requires_grad
    assert all(
        parameter.requires_grad for parameter in model.actor_critic.critic.parameters()
    )
    assert all(
        torch.isfinite(parameter).all()
        for parameter in model.actor_critic.critic.parameters()
    )
    assert model.actor_critic.schedule.sha256 == model.config.flow_schedule_sha256
    assert model.actor_critic.rtc_max_delay_exclusive == 8
    assert torch.equal(
        model.actor_critic.normalizer.action_low,
        torch.full((38,), -2.0),
    )
    critic_after_first_hook = {
        name: tensor.detach().clone()
        for name, tensor in model.actor_critic.critic.state_dict().items()
    }
    model.post_to_empty_hook(cosmos_config)
    for name, tensor in model.actor_critic.critic.state_dict().items():
        assert torch.equal(tensor, critic_after_first_hook[name])
    assert not model.actor_critic.psi_model.action_header.fixed_encoding.requires_grad


def test_post_to_empty_restores_nonpersistent_qwen_rotary_buffers(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
) -> None:
    """Meta materialization cannot leave either Qwen3-VL RoPE table empty."""
    root, bundle, _expected = tiny_source
    model = _build_materialized(root)
    vlm = model.actor_critic.psi_model.vlm_model
    vision_rotary = vlm.model.visual.rotary_pos_emb
    text_rotary = vlm.model.language_model.rotary_emb
    with torch.no_grad():
        vision_rotary.inv_freq.zero_()
        text_rotary.inv_freq.zero_()
    text_rotary.original_inv_freq = torch.full_like(text_rotary.inv_freq, -1.0)

    model.post_to_empty_hook(
        SimpleNamespace(
            policy=SimpleNamespace(model_name_or_path=str(bundle.model_root)),
        )
    )

    assert torch.equal(
        vision_rotary.inv_freq,
        torch.tensor([1.0, 0.01], dtype=vision_rotary.inv_freq.dtype),
    )
    assert torch.equal(
        text_rotary.inv_freq,
        torch.tensor([1.0, 0.25, 0.0625], dtype=text_rotary.inv_freq.dtype),
    )
    assert text_rotary.original_inv_freq is text_rotary.inv_freq


def test_post_to_empty_aligns_bf16_accumulation_across_replay_processes(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout and learner use one BF16 GEMM accumulation contract."""
    root, bundle, _expected = tiny_source
    model = _build_materialized(root)
    monkeypatch.setattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
        True,
    )

    model.post_to_empty_hook(
        SimpleNamespace(
            policy=SimpleNamespace(model_name_or_path=str(bundle.model_root)),
        )
    )

    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction


def test_optimizer_parts_exactly_partition_trainable_parameters(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
) -> None:
    root, bundle, _expected = tiny_source
    model = _build_materialized(root)
    model.post_to_empty_hook(
        SimpleNamespace(
            policy=SimpleNamespace(model_name_or_path=str(bundle.model_root)),
        )
    )
    model.load_hf_weights(str(bundle.model_root), _parallel_dims(), torch.device("cpu"))

    parts = model.separate_model_parts()
    assert parts == [
        model.actor_critic.psi_model.action_header,
        model.actor_critic.critic,
    ]
    part_trainable_parameter_ids = [
        {id(parameter) for parameter in part.parameters() if parameter.requires_grad}
        for part in parts
    ]
    assert part_trainable_parameter_ids[0].isdisjoint(part_trainable_parameter_ids[1])
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert (
        part_trainable_parameter_ids[0] | part_trainable_parameter_ids[1]
        == trainable_parameter_ids
    )
    assert not model.actor_critic.psi_model.action_header.fixed_encoding.requires_grad


def test_critic_initialization_is_independent_of_process_rng(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
) -> None:
    """Disaggregated policy and rollout processes start from identical critics."""
    root, bundle, _expected = tiny_source
    critics: list[dict[str, torch.Tensor]] = []
    for seed in (11, 29):
        torch.manual_seed(seed)
        model = _build_materialized(root)
        model.post_to_empty_hook(
            SimpleNamespace(
                policy=SimpleNamespace(model_name_or_path=str(bundle.model_root)),
            )
        )
        critics.append(
            {
                name: tensor.detach().clone()
                for name, tensor in model.actor_critic.critic.state_dict().items()
            }
        )

    assert critics[0].keys() == critics[1].keys()
    assert all(torch.equal(critics[0][name], critics[1][name]) for name in critics[0])


def test_checkpoint_key_mismatch_fails_closed(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
) -> None:
    """A missing safetensors field never leaves a partially accepted model."""
    root, bundle, expected = tiny_source
    missing_key = next(iter(expected))
    save_file(
        {name: tensor for name, tensor in expected.items() if name != missing_key},
        bundle.checkpoint_path,
    )
    model = _build_materialized(root)
    with pytest.raises(RuntimeError, match="strictly match"):
        model.load_hf_weights(
            str(bundle.model_root),
            _parallel_dims(),
            torch.device("cpu"),
        )
    assert model._weights_loaded is False


def test_config_and_cosmos_registration(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
    tmp_path: Path,
) -> None:
    """Transformers, Cosmos, and the no-config model root resolve one type."""
    root, bundle, _expected = tiny_source
    assert (
        ModelRegistry._MODEL_REGISTRY[cosmos_model.VLA_PSI_PPO_MODEL_TYPE]
        is cosmos_model.VlaPsiPPOModel
    )
    config_dir = tmp_path / "hf_config"
    _config(root).save_pretrained(config_dir)
    loaded = AutoConfig.from_pretrained(config_dir)
    assert isinstance(loaded, cosmos_model.VlaPsiPPOConfig)

    local = load_model_config(str(bundle.model_root))
    assert isinstance(local, cosmos_model.VlaPsiPPOConfig)
    assert local.policy_eval_root == str(root)
    assert local.flow_noise_level == 0.4
    assert local.flow_ignore_last is True


@pytest.mark.parametrize("dimension", ["dp_replicate", "cp", "tp", "pp", "ep"])
def test_parallelism_larger_than_one_fails_closed(dimension: str) -> None:
    """The identity layout cannot silently run a multi-rank configuration."""
    parallel_dims = _parallel_dims(**{dimension: 2, "world_size": 2})
    with pytest.raises(NotImplementedError, match="single-GPU"):
        cosmos_model._identity_parallelize(None, parallel_dims, SimpleNamespace())


def test_index_free_cuda_device_resolves_to_worker_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cosmos's logical `cuda` spelling denotes the current concrete GPU."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert cosmos_model._canonical_torch_device(torch.device("cuda")) == torch.device(
        "cuda:3"
    )
    assert cosmos_model._canonical_torch_device(torch.device("cuda:2")) == torch.device(
        "cuda:2"
    )
    assert cosmos_model._canonical_torch_device(torch.device("cpu")) == torch.device(
        "cpu"
    )


def test_forward_surfaces_delegate_without_abi_translation(
    tiny_source: tuple[Path, VlaSourceBundle, dict[str, torch.Tensor]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cosmos wrapper leaves replay, value, and lease calls on the core."""
    root, _bundle, _expected = tiny_source
    model = _build_materialized(root)
    forward_result = {"log_probs": torch.tensor([3.0])}
    values = torch.tensor([4.0])
    lease = object()
    monkeypatch.setattr(model.actor_critic, "forward", lambda **kwargs: forward_result)
    monkeypatch.setattr(model.actor_critic, "forward_values", lambda **kwargs: values)
    monkeypatch.setattr(model.actor_critic, "clone_for_inference_lease", lambda: lease)

    assert model(example=torch.tensor(1)) is forward_result
    assert model.forward_values(example=torch.tensor(1)) is values
    assert model.clone_for_inference_lease() is lease


def test_rollout_loader_rejects_unattested_runtime_shape(tmp_path: Path) -> None:
    """The public rollout loader rejects unsupported device and dtype up front."""
    with pytest.raises(ValueError, match="bfloat16"):
        cosmos_model.load_vla_rollout_model(
            tmp_path, torch.device("cuda"), torch.float32
        )
    with pytest.raises(ValueError, match="CUDA"):
        cosmos_model.load_vla_rollout_model(
            tmp_path, torch.device("cpu"), torch.bfloat16
        )


def test_rollout_loader_matches_trainer_float32_master_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollout must not quantize weights before Cosmos weight synchronization."""
    observed: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, config: object) -> None:
            observed["config"] = config
            observed["construction_dtype"] = torch.get_default_dtype()

        def _apply(self, function: object) -> _FakeModel:
            observed["apply"] = function
            return self

        def post_to_empty_hook(self, config: object) -> None:
            observed["post_config"] = config

        def load_hf_weights(
            self, path: str, parallel_dims: object, device: torch.device
        ) -> None:
            observed["load"] = (path, parallel_dims, device)

        def eval(self) -> _FakeModel:
            observed["eval"] = True
            return self

    marker = object()
    monkeypatch.setattr(cosmos_model, "register_vla_psi_ppo_model", lambda: None)
    monkeypatch.setattr(
        cosmos_model, "_config_from_attested_model_root", lambda _: marker
    )
    monkeypatch.setattr(cosmos_model, "VlaPsiPPOModel", _FakeModel)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    result = cosmos_model.load_vla_rollout_model(
        tmp_path, torch.device("cuda:0"), torch.bfloat16
    )

    assert isinstance(result, _FakeModel)
    assert observed["config"] is marker
    assert observed["construction_dtype"] is torch.float32
    assert observed["eval"] is True


def test_rollout_meta_context_preserves_nonpersistent_buffers() -> None:
    """Rollout construction retains rotary buffers excluded from checkpoints."""

    class _TinyRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer(
                "inv_freq",
                torch.tensor([1.0, 0.5], dtype=torch.float32),
                persistent=False,
            )

    with cosmos_model.init_on_device("meta", include_buffers=False):
        module = _TinyRotary()

    assert module.weight.device.type == "meta"
    assert module.inv_freq.device.type == "cpu"
    assert torch.equal(module.inv_freq, torch.tensor([1.0, 0.5]))


def test_attested_psi_namespace_package_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second colocated load accepts Psi namespace packages from one root."""
    source_root = tmp_path / "src" / "psi"
    namespace_root = source_root / "models"
    namespace_root.mkdir(parents=True)
    init_file = source_root / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    (namespace_root / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = canonical_tree_snapshot(
        source_root,
        format_name="wenhao-psi-source-tree.v1",
        suffix=".py",
    )

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in tuple(sys.modules):
        if name == "psi" or name.startswith("psi."):
            monkeypatch.delitem(sys.modules, name)
    cosmos_model._import_attested_psi(
        source_root,
        expected_tree_sha256=snapshot.sha256,
    )
    namespace_module = cosmos_model.importlib.import_module("psi.models")

    assert namespace_module.__file__ is None
    assert tuple(namespace_module.__path__) == (str(namespace_root),)
    cosmos_model._import_attested_psi(
        source_root,
        expected_tree_sha256=snapshot.sha256,
    )


def test_unattested_psi_namespace_package_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A namespace search path outside the verified Psi tree fails closed."""
    source_root = tmp_path / "src" / "psi"
    source_root.mkdir(parents=True)
    init_file = source_root / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    rival_root = tmp_path / "rival" / "psi" / "models"
    rival_root.mkdir(parents=True)
    snapshot = canonical_tree_snapshot(
        source_root,
        format_name="wenhao-psi-source-tree.v1",
        suffix=".py",
    )

    namespace_module = ModuleType("psi.models")
    namespace_module.__file__ = None
    namespace_module.__path__ = [str(rival_root)]
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in tuple(sys.modules):
        if name == "psi" or name.startswith("psi."):
            monkeypatch.delitem(sys.modules, name)
    cosmos_model._import_attested_psi(
        source_root,
        expected_tree_sha256=snapshot.sha256,
    )
    monkeypatch.setitem(sys.modules, "psi.models", namespace_module)

    with pytest.raises(RuntimeError, match="unverified Psi search paths"):
        cosmos_model._import_attested_psi(
            source_root,
            expected_tree_sha256=snapshot.sha256,
        )


def test_attested_psi_loader_ignores_forged_timestamp_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid-looking cache cannot replace the attested source at execution."""
    source_root = tmp_path / "src" / "psi"
    source_root.mkdir(parents=True)
    init_file = source_root / "__init__.py"
    init_file.write_text("EXECUTION_SOURCE = 'source'\n", encoding="utf-8")
    snapshot = canonical_tree_snapshot(
        source_root,
        format_name="wenhao-psi-source-tree.v1",
        suffix=".py",
    )
    py_compile.compile(str(init_file), doraise=True)
    cache_path = Path(cosmos_model.importlib.util.cache_from_source(str(init_file)))
    valid_header = cache_path.read_bytes()[:16]
    forged_code = compile(
        "EXECUTION_SOURCE = 'unattested-bytecode'\n",
        str(init_file),
        "exec",
    )
    cache_path.write_bytes(valid_header + marshal.dumps(forged_code))
    assert (
        canonical_tree_snapshot(
            source_root,
            format_name="wenhao-psi-source-tree.v1",
            suffix=".py",
        ).sha256
        == snapshot.sha256
    )

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in tuple(sys.modules):
        if name == "psi" or name.startswith("psi."):
            monkeypatch.delitem(sys.modules, name)
    cosmos_model._import_attested_psi(
        source_root,
        expected_tree_sha256=snapshot.sha256,
    )

    module = sys.modules["psi"]
    assert module.EXECUTION_SOURCE == "source"
    assert module.__cached__ is None


def test_attested_psi_import_is_serialized_and_rejects_another_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent colocated construction cannot race global Psi import state."""
    source_roots = [tmp_path / name / "psi" for name in ("first", "second")]
    snapshots = []
    for index, source_root in enumerate(source_roots):
        source_root.mkdir(parents=True)
        (source_root / "__init__.py").write_text(
            f"SOURCE_ROOT = {index!r}\n",
            encoding="utf-8",
        )
        snapshots.append(
            canonical_tree_snapshot(
                source_root,
                format_name="wenhao-psi-source-tree.v1",
                suffix=".py",
            )
        )

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in tuple(sys.modules):
        if name == "psi" or name.startswith("psi."):
            monkeypatch.delitem(sys.modules, name)

    first_loader_entered = Event()
    release_first_loader = Event()
    second_snapshot_started = Event()
    original_exec_module = cosmos_model._AttestedPsiSourceLoader.exec_module
    original_snapshot = cosmos_model.canonical_tree_snapshot

    def controlled_exec_module(
        loader: cosmos_model._AttestedPsiSourceLoader,
        module: ModuleType,
    ) -> None:
        if loader.source_root == source_roots[0]:
            first_loader_entered.set()
            if not release_first_loader.wait(timeout=5.0):
                raise TimeoutError("test did not release the first Psi loader")
        original_exec_module(loader, module)

    def observed_snapshot(
        root: Path,
        *,
        format_name: str,
        suffix: str | None = None,
    ):
        if Path(root).resolve() == source_roots[1].resolve():
            second_snapshot_started.set()
        return original_snapshot(root, format_name=format_name, suffix=suffix)

    monkeypatch.setattr(
        cosmos_model._AttestedPsiSourceLoader,
        "exec_module",
        controlled_exec_module,
    )
    monkeypatch.setattr(cosmos_model, "canonical_tree_snapshot", observed_snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            cosmos_model._import_attested_psi,
            source_roots[0],
            expected_tree_sha256=snapshots[0].sha256,
        )
        try:
            assert first_loader_entered.wait(timeout=5.0)
            second = executor.submit(
                cosmos_model._import_attested_psi,
                source_roots[1],
                expected_tree_sha256=snapshots[1].sha256,
            )
            assert not second_snapshot_started.wait(timeout=0.2)
        finally:
            release_first_loader.set()
        first.result(timeout=5.0)
        with pytest.raises(RuntimeError, match="unverified Psi source"):
            second.result(timeout=5.0)

    assert second_snapshot_started.is_set()
    assert sys.modules["psi"].SOURCE_ROOT == 0
    active_finders = [
        finder
        for finder in sys.meta_path
        if isinstance(finder, cosmos_model._AttestedPsiSourceFinder)
    ]
    assert len(active_finders) == 1
    assert active_finders[0].source_root == source_roots[0]
