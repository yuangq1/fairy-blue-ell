"""Fit S0 on human scores or S1/S2 on mixed human and pseudo labels."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("S0", "S1", "S2"), required=True)
    parser.add_argument("--seed", type=int, choices=(4486, 4487, 4488), default=4486)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--pseudo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--model-path", default="microsoft/deberta-v3-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.model == "S0" and args.pseudo is not None:
        parser.error("S0 uses only human labels")
    if args.model == "S0" and args.seed != 4486:
        parser.error("Use seed 4486 for the S0 control")
    pseudo = args.pseudo or (Path("data/pseudo/pseudo.jsonl") if args.model != "S0" else None)

    from pipeline.training import train_student

    train_student(args.data, args.model_path, args.output / f"{args.model}_{args.seed}",
                  args.seed, args.model, pseudo_path=pseudo, device=args.device,
                  micro_batch=args.micro_batch, effective_batch=args.effective_batch,
                  epochs=args.epochs, max_length=args.max_length)


if __name__ == "__main__":
    main()
