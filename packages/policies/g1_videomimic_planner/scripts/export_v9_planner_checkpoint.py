#!/usr/bin/env python3
"""Export a VideoMimic V9 actor as a 49-step AlpaGym planner checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpagym_g1_videomimic_planner.export import export_planner_checkpoint
from alpagym_g1_videomimic_planner.model import G1VideoMimicPlannerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[1024, 512, 256, 128],
    )
    parser.add_argument("--init-std", type=float, default=0.25)
    args = parser.parse_args()
    config = G1VideoMimicPlannerConfig(
        hidden_dims=args.hidden_dims,
        init_std=args.init_std,
    )
    export_planner_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
