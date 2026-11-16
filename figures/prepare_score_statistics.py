"""Summarize the official ELL training labels and optionally plot them."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
SCORES = tuple(i / 2 for i in range(2, 11))


def load_scores(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text_id", *TRAITS}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected text_id and the six official score columns")
        ids, rows = set(), []
        for line, row in enumerate(reader, start=2):
            text_id = (row["text_id"] or "").strip()
            if not text_id or text_id in ids:
                raise ValueError(f"Missing or duplicate text_id at CSV line {line}")
            ids.add(text_id)
            try:
                scores = [float(row[trait]) for trait in TRAITS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid score at CSV line {line}") from exc
            if any(score not in SCORES for score in scores):
                raise ValueError(f"Scores must be in 0.5 increments from 1 to 5 at CSV line {line}")
            rows.append(scores)
    if len(rows) < 2:
        raise ValueError("At least two essays are required")
    return np.asarray(rows, dtype=np.float64)


def summarize(scores):
    if (scores.std(axis=0) == 0).any():
        raise ValueError("Pearson correlations require variation in each trait")
    statistics = {}
    for i, trait in enumerate(TRAITS):
        counts = Counter(scores[:, i])
        statistics[trait] = {"frequencies": {str(score): counts[score] for score in SCORES}}
    return {
        "records": len(scores),
        "target_statistics": statistics,
        "pearson_correlations": {
            "trait_order": list(TRAITS),
            "matrix": np.corrcoef(scores, rowvar=False).tolist(),
        },
    }


def save_plots(data, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [trait.title() for trait in TRAITS]
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    for trait, label in zip(TRAITS, labels):
        frequencies = data["target_statistics"][trait]["frequencies"]
        shares = [frequencies[str(score)] / data["records"] for score in SCORES]
        ax.plot(SCORES, shares, marker="o", markersize=4, label=label)
    ax.set(title="Human score distributions", xlabel="Score", ylabel="Share of essays",
           xticks=SCORES, ylim=(0, None))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncols=2, frameon=False)
    fig.savefig(output_dir / "score_distribution.png", dpi=200)
    plt.close(fig)

    matrix = np.asarray(data["pearson_correlations"]["matrix"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(title="Pearson correlations between scores",
           xticks=range(len(TRAITS)), yticks=range(len(TRAITS)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.65 else "black")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.savefig(output_dir / "score_correlation.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Official ELL train.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/figure_data.json"))
    parser.add_argument("--plots-dir", type=Path, help="Write two PNG plots into this directory")
    args = parser.parse_args()
    data = summarize(load_scores(args.train))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved six-trait summaries for {data['records']} essays to {args.output}")
    if args.plots_dir is not None:
        save_plots(data, args.plots_dir)
        print(f"Saved score distribution and Pearson correlation plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
