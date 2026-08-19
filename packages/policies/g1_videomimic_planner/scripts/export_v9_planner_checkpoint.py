#!/usr/bin/env python3
"""Export a VideoMimic V9 actor for current-policy H70 planning."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpagym_g1_videomimic_planner.export import (
    DEFAULT_EXPORT_ACTION_STD_MAX,
    DEFAULT_EXPORT_ACTION_STD_MIN,
    export_planner_checkpoint,
)
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
    parser.add_argument(
        "--min-action-std",
        type=float,
        default=DEFAULT_EXPORT_ACTION_STD_MIN,
        help="Export-time exploration std floor (native V9 default: 0.05).",
    )
    parser.add_argument(
        "--max-action-std",
        type=float,
        default=DEFAULT_EXPORT_ACTION_STD_MAX,
        help="Export-time exploration std ceiling (native V9 default: 0.15).",
    )
    parser.add_argument(
        "--critic-seed",
        type=int,
        default=0,
        help="Deterministic initialization seed for the macro-boundary critic.",
    )
    args = parser.parse_args()
    config = G1VideoMimicPlannerConfig(
        hidden_dims=args.hidden_dims,
        init_std=args.init_std,
    )
    export_planner_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        config=config,
        critic_seed=args.critic_seed,
        min_action_std=args.min_action_std,
        max_action_std=args.max_action_std,
    )


if __name__ == "__main__":
    main()
