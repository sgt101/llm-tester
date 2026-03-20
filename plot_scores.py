#!/usr/bin/env python3
"""
Render a bar chart from score-*.json files in a composite directory.

Usage:
    uv run plot_scores.py <composite_dir> [--save path]

Each score file contributes one group of bars (one bar per object type plus
"overall").  Multiple score files (i.e. multiple models) are shown side by
side for easy comparison.

Options:
    --save <path>   Save the chart to a file instead of opening a window.
                    Format is inferred from the extension (png, pdf, svg …).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_scores(composite_dir: Path) -> dict[str, dict[str, float]]:
    """
    Return {model_label: {metric: score}} for every score-*.json file found.
    Scores are taken from the __summary__ record.
    Metrics are all keys except 'filename', 'total_images', and 'target'.
    """
    score_files = sorted(composite_dir.glob("score-*.json"))
    if not score_files:
        print(f"No score-*.json files found in {composite_dir}")
        sys.exit(1)

    results: dict[str, dict[str, float]] = {}
    for path in score_files:
        records: list[dict] = json.loads(path.read_text())
        summary = next((r for r in records if r.get("filename") == "__summary__"), None)
        if summary is None:
            print(f"  SKIP {path.name}: no __summary__ record")
            continue

        # Derive a short label from the filename: strip "score-" prefix and ".json"
        label = path.stem.removeprefix("score-")

        metrics = {
            k: float(v)
            for k, v in summary.items()
            if k not in ("filename", "total_images", "target")
        }
        results[label] = metrics

    return results


def plot(composite_dir: Path, save_path: Path | None = None) -> None:
    scores = load_scores(composite_dir)
    if not scores:
        print("Nothing to plot.")
        sys.exit(1)

    # Collect all metric names (preserve insertion order, put "overall" last)
    all_metrics: list[str] = []
    for metrics in scores.values():
        for k in metrics:
            if k != "overall" and k not in all_metrics:
                all_metrics.append(k)
    if any("overall" in m for m in scores.values()):
        all_metrics.append("overall")

    models = list(scores.keys())
    n_models = len(models)
    n_metrics = len(all_metrics)

    x = np.arange(n_metrics)
    bar_width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.4), 5))

    for i, model in enumerate(models):
        offsets = x - 0.4 + bar_width * i + bar_width / 2
        values = [scores[model].get(m, 0.0) for m in all_metrics]
        bars = ax.bar(offsets, values, width=bar_width * 0.9, label=model)
        # Label each bar with its value
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, rotation=25, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Model scores — {composite_dir.name}")
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")
    # Shade the "overall" column
    if "overall" in all_metrics:
        idx = all_metrics.index("overall")
        ax.axvspan(idx - 0.5, idx + 0.5, color="lightyellow", zorder=0)
    if n_models > 1:
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    composite_dir = Path(args[0]).resolve()
    if not composite_dir.is_dir():
        print(f"Error: {composite_dir} is not a directory")
        sys.exit(1)

    save_path: Path | None = None
    if "--save" in args:
        idx = args.index("--save")
        try:
            save_path = Path(args[idx + 1])
        except IndexError:
            print("Error: --save requires a path argument")
            sys.exit(1)

    plot(composite_dir, save_path)


if __name__ == "__main__":
    main()
