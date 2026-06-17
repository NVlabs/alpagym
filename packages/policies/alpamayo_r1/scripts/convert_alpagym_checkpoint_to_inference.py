# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an AlpaGym training export into an inference-format Alpamayo checkpoint.

This is the inverse of ``convert_release_to_alpagym_checkpoint.py``. It turns a
Cosmos-RL safetensors export (an ``ExpertModel`` checkpoint, e.g.
``.../cosmos/<run>/safetensors/step_N``) into an inference-format checkpoint that
the AlpaSim ``alpamayo1_5`` driver can load for closed-loop evaluation.

Usage::

    uv run --package alpagym-alpamayo-r1 python \
      packages/policies/alpamayo_r1/scripts/convert_alpagym_checkpoint_to_inference.py \
      --input /PATH/TO/cosmos/<run>/safetensors/step_N \
      --output /PATH/TO/alpamayo-1.5-10B_clrl_inference

Two things differ between the training export and the inference format:

* ``config.json`` carries training-side ``model_type``/``architectures`` and
  ``_target_`` paths in the RL packages (``alpagym_alpamayo_r1``,
  ``alpamayo1_x_rl``, ``alpamayo_r1``). We rewrite it back to the ``alpamayo1_5``
  schema.
* The VLM tensors are flattened at the top level (``model.*``, ``visual.*``,
  ``lm_head.weight``) instead of nested under ``vlm.``. We rename the tensor
  keys shard-by-shard and rewrite the safetensors index. The expert and
  action-head tensors are already in inference naming and pass through unchanged.

The tokenizer/processor and generation config are already present in the export
(written by the forward conversion) and are copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from alpamayo.utils.checkpoint_utils import collect_targets, prepare_output_dir, remap_targets
from safetensors.torch import load_file, save_file

# Inverse of the forward script's TARGET_REMAP, mapping the RL ``_target_``
# paths back to the inference ``alpamayo1_5`` modules. ``remap_target`` uses the
# first matching prefix, so the explicit unicycle entry must precede any broader
# ``action_space`` prefix.
TARGET_REMAP = {
    "alpagym_alpamayo_r1.submodules.action_in_proj.": "alpamayo1_5.models.action_in_proj.",
    "alpamayo1_x_rl.diffusion.": "alpamayo1_5.diffusion.",
    "alpamayo_r1.models.delta_tokenizer.": "alpamayo1_5.models.delta_tokenizer.",
    "alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace": (
        "alpamayo1_5.action_space.UnicycleAccelCurvatureActionSpace"
    ),
    "alpamayo_r1.action_space.discrete_action_space.": (
        "alpamayo1_5.action_space.discrete_action_space."
    ),
}

# Keys the forward script injects via EXPERT_DEFAULTS that the inference config
# does not carry. Everything else (include_camera_ids, include_frame_nums,
# padding_side) is shared and stays.
EXPERT_ONLY_KEYS = (
    "cotrain_vlm",
    "stop_grad_from_vlm",
    "legacy_inference_image_input_format",
    "loss_weights",
    "traj_loss_weight",
    "expert_hist_traj_tokenizer_cfg",
    "hist_traj_embed_cfg",
)


def build_inference_config(expert_config: dict) -> dict:
    """Return an ``alpamayo1_5`` config.json derived from an ExpertModel config."""
    cfg = json.loads(json.dumps(expert_config))

    cfg["model_type"] = "alpamayo1_5"
    cfg["architectures"] = ["Alpamayo1_5"]
    remap_targets(cfg, TARGET_REMAP)

    for key in EXPERT_ONLY_KEYS:
        cfg.pop(key, None)
    cfg["add_special_tokens"] = True

    return cfg


def to_inference_weight_key(key: str) -> str:
    """Map an ExpertModel export tensor name to its inference ``vlm.``-nested name.

    The expert and action-head tensors keep their names; only the VLM tensors,
    flattened at the top level in the export, are renamed back under ``vlm.``.
    """
    if key == "lm_head.weight":
        return "vlm.lm_head.weight"
    if key.startswith("visual."):
        return "vlm.model.visual." + key[len("visual.") :]
    if key.startswith("model."):
        return "vlm.model.language_model." + key[len("model.") :]
    return key


def validate_checkpoint(output_dir: Path) -> None:
    """Fail fast if the converted checkpoint would not load in the AlpaSim driver."""
    cfg = json.loads((output_dir / "config.json").read_text())

    if cfg["model_type"] != "alpamayo1_5":
        raise ValueError(f"model_type not rewritten: {cfg['model_type']!r}")

    for target in sorted(set(collect_targets(cfg).values())):
        if target.startswith(("alpagym_alpamayo_r1.", "alpamayo1_x_rl.", "alpamayo_r1.")):
            raise ValueError(f"RL _target_ left in inference config: {target}")

    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (output_dir / name).is_file():
            raise FileNotFoundError(f"tokenizer file missing from checkpoint: {name}")


def main() -> None:
    """Parse CLI args, rewrite the config, rename weights, copy aux files, and validate."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cosmos-RL safetensors export dir (ExpertModel config + sharded safetensors).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination inference-format checkpoint dir."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output dir.")
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    if input_dir == output_dir:
        raise ValueError(f"--input and --output resolve to the same directory: {input_dir}")

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    print(f"Converting {input_dir} -> {output_dir}")

    inference_config = build_inference_config(json.loads((input_dir / "config.json").read_text()))
    (output_dir / "config.json").write_text(json.dumps(inference_config, indent=2) + "\n")

    # Rename the flattened VLM tensors back under `vlm.`, shard by shard, and
    # rewrite the safetensors index to match.
    index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    print(f"Renaming VLM tensor keys across {len(shards)} shards")
    for shard in shards:
        tensors = load_file(input_dir / shard)
        save_file(
            {to_inference_weight_key(k): v for k, v in tensors.items()},
            output_dir / shard,
            metadata={"format": "pt"},
        )
    index["weight_map"] = {to_inference_weight_key(k): v for k, v in index["weight_map"].items()}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    # The export already carries the tokenizer/processor and generation config.
    skip = {"config.json", "model.safetensors.index.json"}
    for src in sorted(input_dir.iterdir()):
        if src.is_file() and src.suffix != ".safetensors" and src.name not in skip:
            shutil.copyfile(src, output_dir / src.name)

    validate_checkpoint(output_dir)
    print(f"Inference-format checkpoint ready: {output_dir}")


if __name__ == "__main__":
    main()
