# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from alpagym_host.config import register_config_schema
from hydra import compose, initialize_config_module


def test_deploy_local_public_preset(tmp_path: Path) -> None:
    """deploy=local owns the public AlpaSim site; the paired topology owns the backend."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                f"run_root={tmp_path.as_posix()}",
                "deploy=local",
                "topology=local_colocated_1gpu",
            ],
        )

    # The local_colocated_1gpu topology selects local-process execution.
    assert cfg.execution.backend == "local_process"
    # repo_url comes from the shared default and carries the matching humanoid ABI;
    # deploy=local selects the public Wizard deploy preset.
    assert cfg.alpasim.repo_url == "ssh://git@github.com/chenyx09/alpasim.git"
    assert cfg.alpasim.wizard_args.deploy == "local"
    # renderer is unset so the public AlpaSim mirror's own default NRE renderer applies.
    assert cfg.alpasim.wizard_args.renderer is None
