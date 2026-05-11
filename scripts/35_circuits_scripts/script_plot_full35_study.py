#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_summary_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {
        "benchmark_id",
        "top_group_magic",
        "top_group_entanglement",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def benchmark_to_grid_position(benchmark_id: str) -> tuple[int, int]:
    """
    Map benchmark ids to the original 35-circuit grid coordinates.

    Convention:
      - M1..M5  -> top row      : (x=1..5, y=0)
      - E1..E5  -> left column  : (x=0,   y=1..5)
      - Iij     -> interior     : (x=j,   y=i)

    So the full plot spans x,y in {0,1,2,3,4,5}.
    """
    benchmark_id = benchmark_id.strip()

    if benchmark_id.startswith("M") and benchmark_id[1:].isdigit():
        j = int(benchmark_id[1:])
        return j, 0

    if benchmark_id.startswith("E") and benchmark_id[1:].isdigit():
        i = int(benchmark_id[1:])
        return 0, i

    if benchmark_id.startswith("I") and len(benchmark_id) == 3 and benchmark_id[1:].isdigit():
        i = int(benchmark_id[1])
        j = int(benchmark_id[2])
        return j, i

    raise ValueError(f"Unrecognized benchmark_id format: {benchmark_id}")


def add_grid_positions(df: pd.DataFrame) -> pd.DataFrame:
    coords = df["benchmark_id"].apply(benchmark_to_grid_position)
    df = df.copy()
    df["x"] = coords.apply(lambda t: t[0])
    df["y"] = coords.apply(lambda t: t[1])
    return df


# ============================================================
# Plot
# ============================================================

def plot_dominant_group_positions(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Create one figure with two panels:
      1) dominant coalition for magic value
      2) dominant coalition for entanglement value
    """

    color_map = {
        "M": "#59a14f",  # green
        "E": "#4e79a7",  # blue
        "X": "#f28e2b",  # orange
    }

    label_map = {
        "M": "Magic coalition",
        "E": "Entanglement coalition",
        "X": "Mix coalition",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    panels = [
        ("top_group_magic", "Dominant coalition for magic value"),
        ("top_group_entanglement", "Dominant coalition for entanglement value"),
    ]

    for ax, (group_col, title) in zip(axes, panels):
        # scatter one color per dominant group
        for group_code in ["M", "E", "X"]:
            sub = df[df[group_col] == group_code]
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["x"],
                sub["y"],
                s=700,
                c=color_map[group_code],
                edgecolors="black",
                linewidths=1.0,
                alpha=0.95,
                zorder=3,
                label=label_map[group_code],
            )

            # annotate each point with benchmark id
            for _, row in sub.iterrows():
                ax.text(
                    row["x"],
                    row["y"],
                    row["benchmark_id"],
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    zorder=4,
                )

        # axes, ticks, grid
        ax.set_title(title, fontsize=16, pad=12)
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(5.5, -0.5)   # invert so top row is y=0 visually at top
        ax.set_xticks(range(0, 6))
        ax.set_yticks(range(0, 6))
        ax.set_xticklabels(["-", "1", "2", "3", "4", "5"], fontsize=12)
        ax.set_yticklabels(["-", "1", "2", "3", "4", "5"], fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.35, zorder=1)
        ax.set_xlabel("SRE / magic level", fontsize=14)
        ax.set_ylabel("Entanglement level", fontsize=14)

    # figure title
    fig.suptitle(
        "Dominant coalition across the original 35-circuit benchmark layout",
        fontsize=20,
        y=0.98,
    )

    # single legend for whole figure
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=label_map[g],
            markerfacecolor=color_map[g],
            markeredgecolor="black",
            markersize=14,
        )
        for g in ["M", "E", "X"]
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dominant coalition positions on the original 35-circuit grid."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="CSV containing benchmark_id, top_group_magic, top_group_entanglement",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/plots/full35_estimated_study/paper_figures"),
        help="Output directory",
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)

    df = load_summary_csv(args.summary_csv)
    df = add_grid_positions(df)

    out_path = args.outdir / "fig_dominant_group_positions.png"
    plot_dominant_group_positions(df, out_path)


if __name__ == "__main__":
    main()