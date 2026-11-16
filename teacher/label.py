"""Generate one shared pseudo-label pool from the three calibrated teachers."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--teachers", type=Path, default=Path("runs/teachers"))
    parser.add_argument("--output", type=Path, default=Path("data/pseudo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from pipeline.prediction import generate_pseudo_labels
    from teacher.train import TEACHER_SEEDS

    checkpoints = [args.teachers / f"T{seed}" / "checkpoint" for seed in TEACHER_SEEDS]
    generate_pseudo_labels(args.data, checkpoints, args.output,
                           device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
