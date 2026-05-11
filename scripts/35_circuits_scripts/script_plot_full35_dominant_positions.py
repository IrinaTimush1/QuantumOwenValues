#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


GROUP_TO_COLOR: Dict[str, str] = {
    "M": "#63B356",
    "E": "#5B8CC7",
    "X": "#F39A32",
}

GROUP_TO_LABEL: Dict[str, str] = {
    "M": "Magic coalition",
    "E": "Entanglement coalition",
    "X": "Mix coalition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dominant coalition positions across the original 35-circuit benchmark."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figsize", type=float, nargs=2, default=(16, 9), metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalize_group(value: str) -> str:
    if pd.isna(value):
        return "?"
    value = str(value).strip().upper()
    if value in {"M", "E", "X"}:
        return value
    return "?"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"benchmark_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    df = df.copy()

    # Accept either naming convention
    if "magic_top_group" in df.columns:
        pass
    elif "top_group_magic" in df.columns:
        df["magic_top_group"] = df["top_group_magic"]
    else:
        raise ValueError("Missing magic top-group column. Expected 'magic_top_group' or 'top_group_magic'.")

    if "entanglement_top_group" in df.columns:
        pass
    elif "top_group_entanglement" in df.columns:
        df["entanglement_top_group"] = df["top_group_entanglement"]
    else:
        raise ValueError(
            "Missing entanglement top-group column. Expected 'entanglement_top_group' or 'top_group_entanglement'."
        )

    # Accept either position convention
    if "magic_level" not in df.columns:
        if "grid_col" in df.columns:
            df["magic_level"] = df["grid_col"]
        else:
            raise ValueError("Missing magic position column. Expected 'magic_level' or 'grid_col'.")

    if "entanglement_level" not in df.columns:
        if "grid_row" in df.columns:
            df["entanglement_level"] = df["grid_row"]
        else:
            raise ValueError("Missing entanglement position column. Expected 'entanglement_level' or 'grid_row'.")

    df["magic_top_group"] = df["magic_top_group"].apply(normalize_group)
    df["entanglement_top_group"] = df["entanglement_top_group"].apply(normalize_group)

    return df


def draw_background_grid(ax: plt.Axes) -> None:
    for x in range(0, 7):
        ax.axvline(x - 0.5, color="#BDBDBD", linewidth=1, zorder=0)
    for y in range(0, 7):
        ax.axhline(y - 0.5, color="#BDBDBD", linewidth=1, zorder=0)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)

    ax.set_xticks(range(6))
    ax.set_xticklabels(["", "M1", "M2", "M3", "M4", "M5"], fontsize=13)

    ax.set_yticks(range(6))
    ax.set_yticklabels(["", "E1", "E2", "E3", "E4", "E5"], fontsize=13)

    ax.set_xlabel("Magic / SRE level", fontsize=14)
    ax.set_ylabel("Entanglement level", fontsize=14)
    ax.set_aspect("equal")


def plot_panel(ax: plt.Axes, df: pd.DataFrame, group_col: str, title: str) -> None:
    draw_background_grid(ax)
    ax.set_title(title, fontsize=18, pad=16)

    for _, row in df.iterrows():
        benchmark_id = str(row["benchmark_id"])
        group = normalize_group(row[group_col])

        x = int(row["magic_level"])
        y = int(row["entanglement_level"])

        color = GROUP_TO_COLOR.get(group, "#D9D9D9")

        ax.scatter(
            x, y,
            s=900,
            c=color,
            edgecolors="black",
            linewidths=1.6,
            zorder=3,
        )

        ax.text(
            x, y,
            benchmark_id,
            ha="center", va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=4,
        )


def build_legend_handles():
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=GROUP_TO_COLOR[key],
            markeredgecolor="black",
            markeredgewidth=1.3,
            markersize=16,
            label=GROUP_TO_LABEL[key],
        )
        for key in ["M", "E", "X"]
    ]


def main() -> None:
    args = parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {args.summary_csv}")

    df = pd.read_csv(args.summary_csv)
    df = prepare_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize))

    plot_panel(axes[0], df, "magic_top_group", "Dominant coalition for the magic value")
    plot_panel(axes[1], df, "entanglement_top_group", "Dominant coalition for the entanglement value")

    fig.suptitle(
        "Dominant coalition positions across the original 35-circuit benchmark",
        fontsize=24,
        y=0.985,
    )

    fig.text(
        0.5,
        0.925,
        "Each circuit is placed at its original benchmark position and colored by its dominant coalition",
        ha="center",
        va="center",
        fontsize=16,
    )

    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.82])

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()