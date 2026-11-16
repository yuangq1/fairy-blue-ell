"""Predict six scores per essay from a teacher, a student, or a seed ensemble."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with text_id and full_text")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from pipeline.prediction import predict_csv

    predict_csv(args.input, args.checkpoints, args.output,
                device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
