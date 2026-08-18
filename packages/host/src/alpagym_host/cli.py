# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from alpagym_host.config import (
    RunConfig,
    alpagym_project_root,
    load_run_config,
    register_config_schema,
)
from alpagym_host.config_validation import validate_run_config
from alpagym_host.huggingface_validation import validate_huggingface_access
from alpagym_host.humanoid_scene_identity import freeze_humanoid_scene_fingerprints
from alpagym_host.run_artifacts import (
    build_artifact_paths,
    build_run_config,
    normalize_generated_policy_model_path,
    write_run_artifacts,
)
from alpagym_host.run_lifecycle import execute_run
from alpagym_host.slurm import submit_slurm_job


def load_or_create_run_config(cfg: DictConfig) -> RunConfig:
    """Load an existing resolved config or create and write a new one."""
    should_write_artifacts = cfg.execution.resolved_config_path is None
    if should_write_artifacts:
        artifact_paths = build_artifact_paths(cfg)
        run_config = build_run_config(cfg, artifact_paths)
        run_config = normalize_generated_policy_model_path(run_config)
        run_config = freeze_humanoid_scene_fingerprints(run_config)
    else:
        run_config = load_run_config(cfg.execution.resolved_config_path)

    validate_run_config(run_config, cfg.command)
    if should_write_artifacts:
        write_run_artifacts(run_config)
    return run_config


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> object:
    """Run or submit an AlpaGym host command."""
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[str(cfg.logging_level)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    run_config = load_or_create_run_config(cfg)
    # Validate HuggingFace access before AV/NuRec dispatch, so submit fails before
    # queuing a Slurm allocation and run fails before resolving the AlpaSim
    # checkout. Humanoid dynamics-only runs do not consume HuggingFace NuRec assets.
    if (
        getattr(getattr(run_config, "alpasim", None), "simulation_domain", "av")
        != "humanoid"
    ):
        validate_huggingface_access()
    if cfg.command == "run":
        execute_run(run_config)
        return run_config

    elif cfg.command == "submit":
        job_id = submit_slurm_job(
            execution=run_config.execution,
            artifact_paths=run_config.artifact_paths,
            project_root=alpagym_project_root(),
            deploy=str(HydraConfig.get().runtime.choices["deploy"]),
            topology=str(HydraConfig.get().runtime.choices["topology"]),
        )
        print(f"Submitted Slurm job {job_id}")
        print(f"Run directory: {run_config.artifact_paths.run_dir}")
        return run_config

    raise ValueError(f"Unsupported command: {cfg.command!r}")


if __name__ == "__main__":
    register_config_schema()
    main()
