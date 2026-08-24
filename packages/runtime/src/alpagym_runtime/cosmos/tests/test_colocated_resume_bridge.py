# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for colocated checkpoint-resume step bootstrapping."""

import importlib
import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tomllib
from alpagym_host.checkpoint_resume import (
    canonical_json_sha256,
    checkpoint_tree_snapshot,
    validate_native_checkpoint_files,
)
from alpagym_host.config import CosmosRLMode
from alpagym_host.config import register_config_schema
from alpagym_host.run_artifacts import (
    build_artifact_paths,
    build_run_config,
    write_run_artifacts,
)
from hydra import compose, initialize_config_module

from alpagym_runtime.cosmos.colocated_resume_bridge import (
    install_colocated_resume_bootstrap_bridge,
)


def _install_fake_cosmos_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type, type, type]:
    """Install focused Cosmos command/controller fakes for one bridge test."""

    class FakeDataFetchCommand:
        """Minimal DataFetch protocol payload."""

        def __init__(self, *, global_step: object, total_steps: object) -> None:
            """Store the two fields consumed by the bridge."""
            self.global_step = global_step
            self.total_steps = total_steps

    class FakeRolloutToRolloutBroadcastCommand:
        """Minimal R2R protocol payload."""

        def __init__(self, *, weight_step: object, total_steps: object) -> None:
            """Store the two fields consumed by the bridge."""
            self.weight_step = weight_step
            self.total_steps = total_steps

    class FakeColocatedController:
        """Exercise the bridge around Cosmos-owned synchronization behavior."""

        def __init__(
            self,
            *,
            data_fetch_command: object,
            r2r_command: object,
            current_step: int = 0,
        ) -> None:
            """Create a controller with deterministic initial commands."""
            self.data_fetch_command = data_fetch_command
            self.r2r_command = r2r_command
            self.current_step = current_step
            self.total_steps = 100
            self.rollout = object()
            self.synchronization_state: tuple[int, int] | None = None

        def init_commands(self) -> None:
            """Represent the upstream method replaced by the bridge."""
            raise AssertionError("bridge was not installed")

        def policy_consume_one_step_commands_util_data_fetch(self) -> object:
            """Return the initial remote DataFetch command."""
            return self.data_fetch_command

        def wait_for_remote_command(
            self,
            replica: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Return the initial remote R2R command."""
            del replica, args, kwargs
            return self.r2r_command

        def rollout_consume_one_step_commands_util_r2r(self) -> bool:
            """Mirror the upstream R2R wait and observe its local step metadata."""
            self.wait_for_remote_command(self.rollout)
            self.synchronization_state = (self.current_step, self.total_steps)
            return True

    colocated_package = types.ModuleType("cosmos_rl.colocated")
    colocated_package.__path__ = []
    controller_module = types.ModuleType("cosmos_rl.colocated.controller")
    controller_module.ColocatedController = FakeColocatedController
    command_module = types.ModuleType("cosmos_rl.dispatcher.command")
    command_module.DataFetchCommand = FakeDataFetchCommand
    command_module.RolloutToRolloutBroadcastCommand = (
        FakeRolloutToRolloutBroadcastCommand
    )
    monkeypatch.setitem(sys.modules, "cosmos_rl.colocated", colocated_package)
    monkeypatch.setitem(
        sys.modules,
        "cosmos_rl.colocated.controller",
        controller_module,
    )
    monkeypatch.setitem(sys.modules, "cosmos_rl.dispatcher.command", command_module)
    return (
        FakeColocatedController,
        FakeDataFetchCommand,
        FakeRolloutToRolloutBroadcastCommand,
    )


@pytest.mark.parametrize(
    ("global_step", "total_steps", "weight_step", "r2r_total_steps"),
    [
        (1, 4, 1, None),
        (6, 10, 6, 10),
    ],
)
def test_bootstrap_labels_fresh_and_resumed_weights_with_behavior_step(
    monkeypatch: pytest.MonkeyPatch,
    global_step: int,
    total_steps: int,
    weight_step: int,
    r2r_total_steps: int | None,
) -> None:
    """Upcoming-step R2R metadata bootstraps local behavior version N."""
    controller_type, data_fetch_type, r2r_type = _install_fake_cosmos_modules(
        monkeypatch
    )
    install_colocated_resume_bootstrap_bridge()
    data_fetch_command = data_fetch_type(
        global_step=global_step,
        total_steps=total_steps,
    )
    controller = controller_type(
        data_fetch_command=data_fetch_command,
        r2r_command=r2r_type(
            weight_step=weight_step,
            total_steps=r2r_total_steps,
        ),
    )

    controller.init_commands()

    assert controller.current_step == global_step - 1
    assert controller.total_steps == total_steps
    assert controller.synchronization_state == (global_step - 1, total_steps)
    assert controller.init_data_fetch_command is data_fetch_command
    assert "wait_for_remote_command" not in controller.__dict__


@pytest.mark.parametrize(
    ("global_step", "total_steps", "weight_step", "r2r_total_steps", "message"),
    [
        (0, 10, 0, None, "global_step >= 1"),
        (6, 5, 5, None, "total_steps >= DataFetch.global_step"),
        (6, 10, 5, None, "step mismatch"),
        (6, 10, 6, 9, "total-step mismatch"),
    ],
)
def test_bootstrap_rejects_inconsistent_remote_step_metadata(
    monkeypatch: pytest.MonkeyPatch,
    global_step: int,
    total_steps: int,
    weight_step: int,
    r2r_total_steps: int | None,
    message: str,
) -> None:
    """Invalid DataFetch/R2R relationships fail before rollout generation."""
    controller_type, data_fetch_type, r2r_type = _install_fake_cosmos_modules(
        monkeypatch
    )
    install_colocated_resume_bootstrap_bridge()
    controller = controller_type(
        data_fetch_command=data_fetch_type(
            global_step=global_step,
            total_steps=total_steps,
        ),
        r2r_command=r2r_type(
            weight_step=weight_step,
            total_steps=r2r_total_steps,
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        controller.init_commands()


@pytest.mark.parametrize(
    ("mode", "expected_install_count"),
    [
        (CosmosRLMode.colocated, 1),
        (CosmosRLMode.disaggregated, 0),
    ],
)
def test_entrypoint_installs_bridge_only_for_colocated_policy_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: CosmosRLMode,
    expected_install_count: int,
) -> None:
    """Only the process that owns the local colocated controller installs the bridge."""
    entrypoint = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    cosmos_config_path = tmp_path / "cosmos.toml"
    cosmos_config_path.write_text(
        "[custom]\n"
        f"resolved_config_path = {json.dumps(str(tmp_path / 'resolved.yaml'))}\n",
        encoding="utf-8",
    )
    run_config = SimpleNamespace(
        logging_level="DEBUG",
        cosmos=SimpleNamespace(mode=mode),
        policy=SimpleNamespace(model=SimpleNamespace(kind="fake")),
    )
    install_calls: list[None] = []
    data_packer = SimpleNamespace(close=lambda: None)
    policy_bundle = SimpleNamespace(
        build_data_packer=lambda config, cosmos_role: data_packer,
    )
    monkeypatch.setenv("COSMOS_ROLE", "Policy")
    monkeypatch.setattr(entrypoint, "load_run_config", lambda path: run_config)
    monkeypatch.setattr(
        entrypoint,
        "install_colocated_resume_bootstrap_bridge",
        lambda: install_calls.append(None),
    )
    monkeypatch.setattr(
        entrypoint, "get_policy_bundle", lambda model_kind: policy_bundle
    )
    monkeypatch.setattr(entrypoint, "launch_worker", lambda **kwargs: None)

    entrypoint.main(["--config", str(cosmos_config_path)])

    assert len(install_calls) == expected_install_count


def test_step_five_native_restore_bootstraps_step_six_and_next_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """A sealed stage-five state restores before step six uses seed base plus ten."""

    del cosmos_stubs
    # The focused Cosmos test shim exposes the real utils package but omits this
    # rank helper. CheckpointMananger does not call it for a single rank.
    cosmos_util = sys.modules["cosmos_rl.utils.util"]
    monkeypatch.setattr(
        cosmos_util,
        "is_master_rank",
        lambda parallel_dims, global_rank: global_rank == 0,
        raising=False,
    )
    from cosmos_rl.utils.checkpoint import CheckpointMananger

    prior_run = tmp_path / "20260823T120000Z-11111111111111111111111111111111"
    prior_output = prior_run / "cosmos" / "20260823120000"
    checkpoint_config = SimpleNamespace(
        train=SimpleNamespace(
            output_dir=str(prior_output),
            resume=False,
            ckpt=SimpleNamespace(
                enable_checkpoint=True,
                save_mode="sync",
                max_keep=11,
                upload_s3=False,
            ),
        ),
        model_dump=lambda: {"test": "staged-native-resume"},
    )
    prior_manager = CheckpointMananger(checkpoint_config, global_rank=0)
    prior_model = torch.nn.Linear(1, 1, bias=False)
    prior_optimizer = torch.optim.Adam(prior_model.parameters(), lr=0.01)
    prior_scheduler = torch.optim.lr_scheduler.LambdaLR(
        prior_optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    prior_model(torch.ones((1, 1))).sum().backward()
    prior_optimizer.step()
    prior_optimizer.zero_grad()
    prior_scheduler.step()
    saved_weight = prior_model.weight.detach().clone()
    saved_optimizer_state = prior_optimizer.state_dict()
    saved_scheduler_state = prior_scheduler.state_dict()
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    saved_rng_state = prior_manager.get_rng_state()
    prior_manager.save_checkpoint(
        model=prior_model,
        optimizer=prior_optimizer,
        scheduler=prior_scheduler,
        step=5,
        total_steps=50,
        remain_samples_num=90,
        is_final=False,
    )
    prior_manager.save_check(step=5)
    checkpoint = prior_output / "checkpoints" / "step_5" / "policy"
    snapshot = checkpoint_tree_snapshot(checkpoint)
    ranks = validate_native_checkpoint_files(snapshot)
    postrun_body = {
        "schema_id": "alpagym.formal_run_postrun.v2",
        "run_completed": True,
        "cleanup_succeeded": True,
        "cleanup_failure": None,
        "runtime_ready_captured": True,
        "source_watch_clean": True,
        "sources_unchanged": True,
        "configs_unchanged": True,
        "formal_run_valid": True,
        "native_checkpoints": [
            {
                "cosmos_output_relative_path": prior_output.relative_to(
                    prior_run
                ).as_posix(),
                "policy_relative_path": checkpoint.relative_to(prior_run).as_posix(),
                "step": 5,
                "tree_sha256": snapshot.tree_sha256,
                "file_count": snapshot.file_count,
                "total_size_bytes": snapshot.total_size_bytes,
                "files": [identity.to_dict() for identity in snapshot.files],
                "ranks": list(ranks),
            }
        ],
    }
    postrun_sha256 = canonical_json_sha256(postrun_body)
    provenance = prior_run / "provenance"
    provenance.mkdir()
    (provenance / "postrun.json").write_text(
        json.dumps(
            {**postrun_body, "receipt_sha256": postrun_sha256},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        authored = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path / 'current'}",
                "experiment=g1_vla_hq_stairs_local_1gpu_train2_staged",
                "cosmos.train.max_num_steps=10",
                "cosmos.train.resume.enabled=true",
                f"cosmos.train.resume.prior_formal_run_id={prior_run.name}",
                "cosmos.train.resume.checkpoint_step=5",
                f"cosmos.train.resume.checkpoint_path={checkpoint}",
                f"cosmos.train.resume.checkpoint_tree_sha256={snapshot.tree_sha256}",
                f"cosmos.train.resume.prior_postrun_receipt_sha256={postrun_sha256}",
                "cosmos.train.resume.expected_next_training_step=6",
                f"policy.model.path={tmp_path / 'model'}",
                f"alpasim.repo_path={tmp_path / 'alpasim'}",
                "alpasim.repo_url=null",
                "alpasim.repo_ref=null",
                f"alpasim.humanoid.repo_path={tmp_path / 'humanoid'}",
                f"alpasim.humanoid.scene_store_path={tmp_path / 'scenes'}",
                "alpasim.humanoid.visual_controller_release_path="
                f"{tmp_path / 'controller'}",
                f"alpasim.humanoid.scene_cache_path={tmp_path / 'scene-cache'}",
                f"alpasim.humanoid.runtime_cache_path={tmp_path / 'runtime-cache'}",
            ],
        )
    artifact_paths = build_artifact_paths(authored)
    run_config = build_run_config(authored, artifact_paths)
    import alpagym_host.config_validation as config_validation

    config_validation._validate_checkpoint_resume_config(
        config=run_config,
        requested_command="run",
    )
    write_run_artifacts(run_config)
    generated = tomllib.loads(
        artifact_paths.cosmos_config_path.read_text(encoding="utf-8")
    )
    assert generated["train"]["resume"] == str(checkpoint)
    assert generated["train"]["max_num_steps"] == 10
    assert generated["train"]["epoch"] == 50

    trainer_module = importlib.import_module("alpagym_runtime.cosmos.trainer")
    current_output = artifact_paths.run_dir / "cosmos" / "20260823130000"
    runtime_config = SimpleNamespace(
        custom={"resolved_config_path": str(artifact_paths.resolved_config_path)},
        train=SimpleNamespace(
            output_dir=str(current_output),
            timestamp=current_output.name,
            resume=str(checkpoint),
            train_policy=SimpleNamespace(kl_beta=0.0),
            ckpt=SimpleNamespace(
                enable_checkpoint=True,
                save_mode="sync",
                max_keep=11,
                upload_s3=False,
            ),
        ),
        policy=SimpleNamespace(
            model_name_or_path=str(tmp_path / "model"),
            model_revision=None,
        ),
        model_dump=lambda: {"test": "staged-native-resume"},
    )
    trainer = object.__new__(trainer_module.AlpagymGRPOTrainer)
    trainer.config = runtime_config
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight.data.zero_()
    trainer.optimizers = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
    trainer.lr_schedulers = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizers,
        lr_lambda=lambda _step: 1.0,
    )
    trainer.ckpt_manager = CheckpointMananger(runtime_config, global_rank=0)
    trainer.map_w_from_policy_to_rollout = {"weight": trainer.model.weight}
    trainer.model_load_from_hf = lambda: pytest.fail(
        "sealed native resume fell back to base weights"
    )

    def model_resume_from_checkpoint() -> dict[str, object]:
        """Load through Cosmos's native checkpoint manager for this CPU model."""

        restored, trainer.lr_schedulers = trainer.ckpt_manager.load_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizers,
            scheduler=trainer.lr_schedulers,
            model_name_or_path=runtime_config.policy.model_name_or_path,
            revision=runtime_config.policy.model_revision,
        )
        return restored

    trainer.model_resume_from_checkpoint = model_resume_from_checkpoint
    trainer.set_model_train = trainer.model.train
    torch.manual_seed(9999)
    np.random.seed(9999)
    random.seed(9999)

    restored = trainer.weight_resume()

    assert restored == {
        "step": 5,
        "total_steps": 50,
        "remain_samples_num": 90,
        "is_final": False,
    }
    torch.testing.assert_close(trainer.model.weight, saved_weight)
    assert trainer_module._nested_state_equal(
        trainer.optimizers.state_dict(),
        saved_optimizer_state,
    )
    assert trainer.lr_schedulers.state_dict() == saved_scheduler_state
    assert trainer_module._nested_state_equal(
        trainer.ckpt_manager.get_rng_state(),
        saved_rng_state,
    )
    restore_receipt = json.loads(
        (
            artifact_paths.artifacts_dir / "checkpoint_resume_receipts" / "rank_0.json"
        ).read_text(encoding="utf-8")
    )
    assert restore_receipt["restored_training_step"] == 5
    assert restore_receipt["expected_next_training_step"] == 6
    assert restore_receipt["native_restore"]["loader_completed"] is True

    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    assert (
        rollout_module._effective_humanoid_rollout_seed_base(run_config)
        == 202608240100010
    )
    controller_type, data_fetch_type, r2r_type = _install_fake_cosmos_modules(
        monkeypatch
    )
    install_colocated_resume_bootstrap_bridge()
    data_fetch = data_fetch_type(global_step=6, total_steps=10)
    controller = controller_type(
        data_fetch_command=data_fetch,
        r2r_command=r2r_type(weight_step=6, total_steps=10),
    )

    controller.init_commands()

    assert controller.current_step == 5
    assert controller.total_steps == 10
    assert controller.init_data_fetch_command is data_fetch
