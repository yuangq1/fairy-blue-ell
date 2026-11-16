"""Fit the three scorers and calibrate their variance heads."""

import argparse
from pathlib import Path

TEACHER_SEEDS = (2006, 7, 24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("runs/teachers"))
    parser.add_argument("--model-path", default="microsoft/deberta-v3-large")
    parser.add_argument("--seed", choices=TEACHER_SEEDS, type=int, help="Fit just this teacher")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    from pipeline.training import train_teacher

    for seed in (TEACHER_SEEDS if args.seed is None else (args.seed,)):
        train_teacher(args.data, args.model_path, args.output / f"T{seed}", seed,
                      device=args.device, micro_batch=args.micro_batch,
                      effective_batch=args.effective_batch, epochs=args.epochs,
                      max_length=args.max_length)


if __name__ == "__main__":
    main()
