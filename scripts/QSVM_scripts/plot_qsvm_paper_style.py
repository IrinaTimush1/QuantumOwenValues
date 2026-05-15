#!/usr/bin/env python3
"""
plot_qsvm_paper_style.py

Paper-style plotting script for the QSVM Owen-value experiment.

Expected input structure:

results/qsvm_owen_all_datasets/
  all_run_summaries.csv
  dataset_0/r1/gate_owen_values.csv
  dataset_0/r1/group_owen_values.csv
  dataset_0/r1/qsvm_value_cache.csv
  dataset_0/r1/coalition_values_exact_owen.csv       # optional fallback/alternative
  dataset_0/r2/...
  ...
  dataset_4/r3/...

Main outputs:

1. paperstyle_dataset0_gate_owen_r123.{png,pdf}
   One paper-Fig.-6-style scatter plot for dataset 0, overlaying r=1, r=2, r=3.
   Passive H gates are shown as grey x markers at 0.

2. paperstyle_gate_owen_mean_std_r1.{png,pdf}
   paperstyle_gate_owen_mean_std_r2.{png,pdf}
   paperstyle_gate_owen_mean_std_r3.{png,pdf}
   Paper-Fig.-9-style mean +/- one sample standard deviation over dataset_0..dataset_4.

3. paperstyle_gate_owen_mean_std_r123_panels.{png,pdf}
   Same mean +/- std information in one 3-panel figure.

4. paperstyle_value_distributions_dataset0_r123.{png,pdf}
   Paper-Fig.-7-style boxplots of coalition value-function accuracies by number of
   included total gates k for dataset 0. These are based on the coalition values
   available in qsvm_value_cache.csv / coalition_values_exact_owen.csv. If you did
   not run the all-subset distribution, this plot shows the coalition values visited
   during exact Owen evaluation, not the complete 2^N active-subset distribution.

5. The previous aggregate bar plots and statistics CSVs are also retained.

Notes:
- Error bars in the gate mean/std plots are sample standard deviations across dataset splits.
- Passive H gates are always included in the circuit and are not Owen players, so their
  Owen values are plotted as 0 markers only for visual alignment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


GROUP_ORDER = ["E", "M", "X"]
GROUP_LABELS = {
    "E": "E: CNOT scaffold",
    "M": "M: local rotations",
    "X": "X: entangling-motif rotations",
    "passive": "passive H",
}
GROUP_COLORS = {
    "E": "tab:blue",
    "M": "tab:orange",
    "X": "tab:green",
    "passive": "0.55",
}

R_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
}
R_COLORS = {
    1: "#005270",
    2: "#C04C8A",
    3: "#FF9300",
}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.65,
        "lines.linewidth": 1.1,
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run_summaries(results_dir: Path) -> pd.DataFrame:
    summary_csv = results_dir / "all_run_summaries.csv"
    if summary_csv.exists():
        return pd.read_csv(summary_csv)

    rows = []
    for summary_path in sorted(results_dir.glob("dataset_*/r*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(
            f"Could not find all_run_summaries.csv or dataset_*/r*/summary.json under {results_dir}"
        )
    return pd.DataFrame(rows)


def load_gate_values(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("dataset_*/r*/gate_owen_values.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No gate_owen_values.csv files found under {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["r"] = df["r"].astype(int)
    df["data_index"] = df["data_index"].astype(int)
    df["gate_index_1based"] = df["gate_index_1based"].astype(int)
    df["owen_value"] = pd.to_numeric(df["owen_value"], errors="coerce")
    # Passive gates are not players; stored Owen values may be NaN. Plot them as 0.
    df["owen_value_plot"] = df["owen_value"].fillna(0.0)
    return df


def load_group_values(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("dataset_*/r*/group_owen_values.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No group_owen_values.csv files found under {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["r"] = df["r"].astype(int)
    df["data_index"] = df["data_index"].astype(int)
    df["group_owen_value"] = pd.to_numeric(df["group_owen_value"], errors="coerce")
    return df


def load_coalition_values_for_dataset(results_dir: Path, dataset_index: int) -> pd.DataFrame:
    """Load available coalition values for value-distribution boxplots."""
    frames = []
    for r_dir in sorted((results_dir / f"dataset_{dataset_index}").glob("r*")):
        # Prefer qsvm_value_cache.csv because it contains every cached coalition value.
        candidates = [
            r_dir / "qsvm_value_cache.csv",
            r_dir / "coalition_values_exact_owen.csv",
            r_dir / "value_distribution.csv",
        ]
        chosen = next((p for p in candidates if p.exists()), None)
        if chosen is None:
            continue
        df = pd.read_csv(chosen)
        if "value_accuracy" not in df.columns:
            continue
        if "r" not in df.columns:
            # Infer r from directory name, e.g. r2.
            df["r"] = int(r_dir.name.replace("r", ""))
        if "data_index" not in df.columns:
            df["data_index"] = dataset_index
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"Could not find qsvm_value_cache.csv or coalition_values_exact_owen.csv under "
            f"{results_dir / f'dataset_{dataset_index}'}"
        )
    df = pd.concat(frames, ignore_index=True)
    df["r"] = df["r"].astype(int)
    df["data_index"] = df["data_index"].astype(int)
    df["value_accuracy"] = pd.to_numeric(df["value_accuracy"], errors="coerce")
    if "n_total_gates" in df.columns:
        df["n_total_gates"] = pd.to_numeric(df["n_total_gates"], errors="coerce").astype("Int64")
    elif "included_gate_indices_total" in df.columns:
        df["n_total_gates"] = df["included_gate_indices_total"].apply(count_gate_list).astype("Int64")
    elif "n_active_gates" in df.columns:
        # Less ideal, but keep the plot available.
        df["n_total_gates"] = pd.to_numeric(df["n_active_gates"], errors="coerce").astype("Int64")
    else:
        raise ValueError("Coalition-value CSV needs n_total_gates, included_gate_indices_total, or n_active_gates.")
    return df.dropna(subset=["value_accuracy", "n_total_gates"])


def count_gate_list(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if text == "" or text.lower() == "empty":
        return 0
    # Handles strings like "1-2-3" or "[1, 2, 3]" robustly enough for this script.
    cleaned = text.replace("[", "").replace("]", "").replace(",", "-").replace(" ", "")
    parts = [p for p in cleaned.split("-") if p]
    return len(parts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_std_se_ci(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "se": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan, "ci95_half_width": np.nan}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and scipy_stats is not None:
        tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    else:
        tcrit = 1.96
    half = tcrit * se
    return {"n": n, "mean": mean, "std": std, "se": se,
            "ci95_low": mean - half, "ci95_high": mean + half, "ci95_half_width": half}


def paired_test_rows(
    df: pd.DataFrame,
    value_col: str,
    condition_col: str,
    conditions: list,
    pair_col: str = "data_index",
    context: dict | None = None,
) -> list[dict]:
    context = context or {}
    rows: list[dict] = []
    for a, b in itertools.combinations(conditions, 2):
        wide = df[df[condition_col].isin([a, b])].pivot_table(
            index=pair_col, columns=condition_col, values=value_col, aggfunc="first"
        )
        if a not in wide.columns or b not in wide.columns:
            continue
        wide = wide[[a, b]].dropna()
        x = wide[a].astype(float).to_numpy()
        y = wide[b].astype(float).to_numpy()
        diff = y - x
        n = len(diff)
        row = {
            **context,
            "comparison": f"{b} - {a}",
            "condition_a": a,
            "condition_b": b,
            "n_pairs": n,
            "mean_difference": float(np.mean(diff)) if n else np.nan,
            "std_difference": float(np.std(diff, ddof=1)) if n > 1 else np.nan,
        }
        if scipy_stats is not None and n > 1:
            t_res = scipy_stats.ttest_rel(y, x)
            row["paired_t_statistic"] = float(t_res.statistic)
            row["paired_t_p_two_sided"] = float(t_res.pvalue)
            try:
                w_two = scipy_stats.wilcoxon(y, x, alternative="two-sided", zero_method="wilcox")
                row["wilcoxon_p_two_sided"] = float(w_two.pvalue)
            except Exception:
                row["wilcoxon_p_two_sided"] = np.nan
        else:
            row["paired_t_statistic"] = np.nan
            row["paired_t_p_two_sided"] = np.nan
            row["wilcoxon_p_two_sided"] = np.nan
        rows.append(row)
    return rows


def add_group_shares(group_df: pd.DataFrame) -> pd.DataFrame:
    df = group_df.copy()
    totals = (
        df.groupby(["data_index", "r"], as_index=False)["group_owen_value"]
        .sum()
        .rename(columns={"group_owen_value": "total_group_owen"})
    )
    df = df.merge(totals, on=["data_index", "r"], how="left")
    df["share"] = df["group_owen_value"] / df["total_group_owen"]
    df["share_percent"] = 100.0 * df["share"]
    return df


def make_stats_tables(summary_df: pd.DataFrame, gate_df: pd.DataFrame, group_df: pd.DataFrame, out_dir: Path) -> None:
    acc_stats = []
    for r, sub in summary_df.groupby("r", sort=True):
        acc_stats.append({"r": int(r), **mean_std_se_ci(sub["full_accuracy"])})
    pd.DataFrame(acc_stats).to_csv(out_dir / "qsvm_accuracy_stats.csv", index=False)

    acc_tests = paired_test_rows(
        summary_df,
        value_col="full_accuracy",
        condition_col="r",
        conditions=sorted(summary_df["r"].unique()),
        pair_col="data_index",
    )
    pd.DataFrame(acc_tests).to_csv(out_dir / "qsvm_accuracy_paired_tests.csv", index=False)

    gate_stats = []
    active_gate_df = gate_df[gate_df["group"].isin(GROUP_ORDER)].copy()
    for (r, gate_idx), sub in active_gate_df.groupby(["r", "gate_index_1based"], sort=True):
        row0 = sub.iloc[0]
        gate_stats.append({
            "r": int(r),
            "gate_index_1based": int(gate_idx),
            "gate_name": row0["gate_name"],
            "group": row0["group"],
            **mean_std_se_ci(sub["owen_value"]),
        })
    pd.DataFrame(gate_stats).to_csv(out_dir / "qsvm_gate_owen_stats.csv", index=False)

    group_stats = []
    for (r, group), sub in group_df.groupby(["r", "group"], sort=True):
        group_stats.append({"r": int(r), "group": group, **mean_std_se_ci(sub["group_owen_value"])})
    pd.DataFrame(group_stats).to_csv(out_dir / "qsvm_group_owen_stats.csv", index=False)

    group_tests = []
    for r, sub in group_df.groupby("r", sort=True):
        group_tests.extend(paired_test_rows(
            sub, value_col="group_owen_value", condition_col="group",
            conditions=GROUP_ORDER, pair_col="data_index", context={"r": int(r)}
        ))
    pd.DataFrame(group_tests).to_csv(out_dir / "qsvm_group_owen_paired_tests.csv", index=False)

    share_df = add_group_shares(group_df)
    share_stats = []
    for (r, group), sub in share_df.groupby(["r", "group"], sort=True):
        share_stats.append({"r": int(r), "group": group, **mean_std_se_ci(sub["share_percent"])})
    pd.DataFrame(share_stats).to_csv(out_dir / "qsvm_group_share_stats.csv", index=False)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    fig.tight_layout(pad=1.2, h_pad=2.6, w_pad=1.2)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def gate_names_for_r(r: int) -> list[str]:
    return ["H", "P", "H", "P", "CX", "P", "CX"] * int(r)


def set_top_gate_axis(ax: plt.Axes, max_gate: int) -> None:
    names = gate_names_for_r(max_gate // 7)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(np.arange(1, max_gate + 1))
    ax_top.set_xticklabels(names, fontsize=11)
    ax_top.tick_params(axis="x", which="major", length=7, width=0.9, pad=5)


def force_times_1e_minus_2(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, -2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
    # Matplotlib's automatic offset text sits at the upper-left inside the axes,
    # where it can collide with the first top-axis gate label. Keep the same
    # multiplier, but place it just outside the plotting box for cleaner output.
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        -0.065,
        1.015,
        r"$\times 10^{-2}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams.get("ytick.labelsize", 11),
        clip_on=False,
    )


def active_and_passive_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = df[df["group"].isin(GROUP_ORDER)].copy()
    passive = df[df["group"].astype(str).str.lower().eq("passive")].copy()
    return active, passive


# ---------------------------------------------------------------------------
# Paper-style plots requested by the user
# ---------------------------------------------------------------------------

def plot_dataset_gate_owen_scatter_r123(gate_df: pd.DataFrame, out_dir: Path, dataset_index: int = 0) -> None:
    """One Fig.-6-style scatter plot, overlaying r=1/r=2/r=3 for one dataset."""
    sub_all = gate_df[gate_df["data_index"] == dataset_index].copy()
    if sub_all.empty:
        raise ValueError(f"No gate values found for data_index={dataset_index}")

    max_r = int(sub_all["r"].max())
    max_gate = 7 * max_r
    fig, ax = plt.subplots(figsize=(13.6, 3.8))

    for r in sorted(sub_all["r"].unique()):
        sub = sub_all[sub_all["r"] == r].sort_values("gate_index_1based")
        active, _ = active_and_passive_split(sub)
        ax.scatter(
            active["gate_index_1based"],
            active["owen_value"],
            s=34,
            marker=R_MARKERS.get(int(r), "o"),
            color=R_COLORS.get(int(r), None),
            label=rf"$r={int(r)}$, exact",
            zorder=3,
        )

    # Show passive H gates from the largest r, like the paper-style overlay.
    max_r_df = sub_all[sub_all["r"] == max_r].copy()
    _, passive = active_and_passive_split(max_r_df)
    if not passive.empty:
        ax.scatter(
            passive["gate_index_1based"],
            np.zeros(len(passive)),
            s=34,
            marker="x",
            color="0.45",
            linewidths=1.0,
            label="passive H",
            zorder=4,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xlim(0.5, max_gate + 0.5)
    ax.set_xticks(np.arange(1, max_gate + 1))
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\Phi_{\mathrm{Owen}}^{(g)}$")
    force_times_1e_minus_2(ax)
    set_top_gate_axis(ax, max_gate)
    ax.grid(True, color="0.75", alpha=0.85)
    ax.legend(frameon=True, fancybox=False, loc="upper right")
    save_fig(fig, out_dir / f"paperstyle_dataset{dataset_index}_gate_owen_r123")


def plot_gate_owen_mean_std_single_r(gate_df: pd.DataFrame, out_dir: Path, r: int) -> None:
    """Fig.-9-style mean +/- std over datasets for one r."""
    sub = gate_df[gate_df["r"] == r].copy()
    if sub.empty:
        raise ValueError(f"No gate values found for r={r}")

    stats_rows = []
    for gate_idx, gsub in sub.groupby("gate_index_1based", sort=True):
        row0 = gsub.iloc[0]
        values = gsub["owen_value_plot"].to_numpy(dtype=float)
        # Passive H values are all plotted as zero; active gates use their real Owen values.
        stats_rows.append({
            "gate_index_1based": int(gate_idx),
            "gate_name": row0["gate_name"],
            "group": row0["group"],
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("gate_index_1based")

    fig, ax = plt.subplots(figsize=(8.0 if r <= 2 else 11.2, 3.2))
    active = stats_df[stats_df["group"].isin(GROUP_ORDER)].copy()
    passive = stats_df[stats_df["group"].astype(str).str.lower().eq("passive")].copy()

    ax.errorbar(
        active["gate_index_1based"],
        active["mean"],
        yerr=active["std"],
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=0.9,
        color=R_COLORS.get(r, "black"),
        label="mean ± one std over datasets",
        zorder=3,
    )
    if not passive.empty:
        ax.scatter(
            passive["gate_index_1based"],
            np.zeros(len(passive)),
            marker="x",
            s=28,
            color="0.45",
            linewidths=1.0,
            label="passive H",
            zorder=4,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xlim(0.5, 7 * r + 0.5)
    ax.set_xticks(np.arange(1, 7 * r + 1))
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\bar{\Phi}_{\mathrm{Owen}}^{(g)}$")
    force_times_1e_minus_2(ax)
    set_top_gate_axis(ax, 7 * r)
    ax.grid(True, color="0.75", alpha=0.85)
    # No top title here: the top x-axis gate labels are intentionally kept clear,
    # matching the paper style where the explanation goes in the figure caption.
    save_fig(fig, out_dir / f"paperstyle_gate_owen_mean_std_r{r}")


def plot_gate_owen_mean_std_panels(gate_df: pd.DataFrame, out_dir: Path) -> None:
    """Three-panel version of the mean +/- std over datasets."""
    r_values = sorted(gate_df["r"].unique())
    fig, axes = plt.subplots(len(r_values), 1, figsize=(12.0, 3.35 * len(r_values)), sharex=False)
    if len(r_values) == 1:
        axes = [axes]

    for ax, r in zip(axes, r_values):
        sub = gate_df[gate_df["r"] == r].copy()
        stats_rows = []
        for gate_idx, gsub in sub.groupby("gate_index_1based", sort=True):
            row0 = gsub.iloc[0]
            values = gsub["owen_value_plot"].to_numpy(dtype=float)
            stats_rows.append({
                "gate_index_1based": int(gate_idx),
                "gate_name": row0["gate_name"],
                "group": row0["group"],
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            })
        stats_df = pd.DataFrame(stats_rows).sort_values("gate_index_1based")
        active = stats_df[stats_df["group"].isin(GROUP_ORDER)].copy()
        passive = stats_df[stats_df["group"].astype(str).str.lower().eq("passive")].copy()

        ax.errorbar(
            active["gate_index_1based"],
            active["mean"],
            yerr=active["std"],
            fmt="o",
            markersize=5,
            capsize=3,
            elinewidth=0.9,
            color=R_COLORS.get(int(r), "black"),
            zorder=3,
        )
        if not passive.empty:
            ax.scatter(passive["gate_index_1based"], np.zeros(len(passive)), marker="x", s=28,
                       color="0.45", linewidths=1.0, zorder=4)
        ax.axhline(0.0, color="0.55", linewidth=0.8)
        ax.set_xlim(0.5, 7 * int(r) + 0.5)
        ax.set_xticks(np.arange(1, 7 * int(r) + 1))
        ax.set_ylabel(r"$\bar{\Phi}_{\mathrm{Owen}}^{(g)}$")
        force_times_1e_minus_2(ax)
        set_top_gate_axis(ax, 7 * int(r))
        ax.grid(True, color="0.75", alpha=0.85)
        ax.set_title(rf"({chr(96+int(r))}) $r={int(r)}$, exact Owen, mean $\pm$ std over 5 datasets",
                     fontweight="bold", y=-0.52)
        ax.set_xlabel(r"$g$")

    save_fig(fig, out_dir / "paperstyle_gate_owen_mean_std_r123_panels")


def plot_value_distribution_dataset_r123(results_dir: Path, out_dir: Path, dataset_index: int = 0) -> None:
    """Fig.-7-style value-function distribution by number of total included gates k."""
    val_df = load_coalition_values_for_dataset(results_dir, dataset_index=dataset_index)
    r_values = sorted(val_df["r"].unique())

    # Layout like the screenshot/paper: r=1 and r=2 on top, r=3 full width below.
    if set(r_values) >= {1, 2, 3}:
        fig = plt.figure(figsize=(11.5, 8.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.62, wspace=0.32)
        ax_map = {1: fig.add_subplot(gs[0, 0]), 2: fig.add_subplot(gs[0, 1]), 3: fig.add_subplot(gs[1, :])}
    else:
        fig, axs = plt.subplots(len(r_values), 1, figsize=(10.5, 3.0 * len(r_values)))
        if len(r_values) == 1:
            axs = [axs]
        ax_map = dict(zip(r_values, axs))

    for r in r_values:
        ax = ax_map[int(r)]
        sub = val_df[val_df["r"] == r].copy()
        max_k = 7 * int(r)
        all_k_desc = list(range(max_k, -1, -1))
        data = []
        positions = []
        for k in all_k_desc:
            vals = sub.loc[sub["n_total_gates"] == k, "value_accuracy"].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                data.append(vals)
                positions.append(k)

        color = R_COLORS.get(int(r), "black")
        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
                medianprops={"color": color, "linewidth": 1.1},
                boxprops={"edgecolor": color, "linewidth": 0.9},
                whiskerprops={"color": color, "linewidth": 0.8},
                capprops={"color": color, "linewidth": 0.8},
                flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": color,
                            "markersize": 3.2, "linestyle": "none"},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.16)

        # Dotted maximum value per k, analogous to the paper's Pareto-frontier guide.
        max_by_k = sub.groupby("n_total_gates")["value_accuracy"].max().sort_index(ascending=False)
        if len(max_by_k) > 0:
            ax.plot(max_by_k.index.to_numpy(dtype=float), max_by_k.to_numpy(dtype=float),
                    linestyle=(0, (1, 2.4)), color="black", linewidth=1.2)

        ax.set_xlim(max_k + 0.55, -0.55)
        ax.set_xticks(all_k_desc)
        ax.set_xlabel(r"number of gates $k$")
        ax.set_ylabel(r"$\mathcal{W}_k$")
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, color="0.75", alpha=0.85)
        caption = rf"({chr(96+int(r))}) $r={int(r)}$, exact Owen"
        ax.set_title(caption, fontweight="bold", y=-0.34)

    fig.text(0.5, 0.035,
             "H gates are passive and always included; k is total included gates. "
             "Boxplots use coalition values available from the exact-Owen cache.",
             ha="center", fontsize=10)
    save_fig(fig, out_dir / f"paperstyle_value_distributions_dataset{dataset_index}_r123")


# ---------------------------------------------------------------------------
# Original aggregate plots retained
# ---------------------------------------------------------------------------

def plot_accuracy_mean_std(summary_df: pd.DataFrame, out_dir: Path) -> None:
    stats_df = (summary_df.groupby("r")["full_accuracy"]
                .agg(mean="mean", std=lambda x: np.std(x, ddof=1), n="count")
                .reset_index().sort_values("r"))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(stats_df))
    ax.bar(x, stats_df["mean"], yerr=stats_df["std"], capsize=5, alpha=0.85, label="mean ± std")
    for i, r in enumerate(stats_df["r"]):
        vals = summary_df.loc[summary_df["r"] == r, "full_accuracy"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=34, zorder=3,
                   edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={int(r)}" for r in stats_df["r"]])
    ax.set_ylabel("Full-circuit QSVM test accuracy")
    ax.set_title("QSVM accuracy across five dataset splits")
    ax.set_ylim(0.45, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, out_dir / "qsvm_accuracy_mean_std")


def plot_group_owen_mean_std(group_df: pd.DataFrame, out_dir: Path) -> None:
    stats_df = (group_df.groupby(["r", "group"])["group_owen_value"]
                .agg(mean="mean", std=lambda x: np.std(x, ddof=1), n="count")
                .reset_index())
    r_values = sorted(group_df["r"].unique())
    x = np.arange(len(r_values))
    width = 0.23
    offsets = {g: (i - 1) * width for i, g in enumerate(GROUP_ORDER)}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for group in GROUP_ORDER:
        means, stds = [], []
        for r in r_values:
            row = stats_df[(stats_df["r"] == r) & (stats_df["group"] == group)]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]))
        xpos = x + offsets[group]
        ax.bar(xpos, means, width=width, yerr=stds, capsize=4, alpha=0.86,
               color=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for j, r in enumerate(r_values):
            vals = group_df.loc[(group_df["r"] == r) & (group_df["group"] == group),
                                "group_owen_value"].to_numpy(dtype=float)
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xpos[j]) + jitter, vals, s=22, zorder=3,
                       edgecolor="black", linewidth=0.35, color=GROUP_COLORS[group])
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={int(r)}" for r in r_values])
    ax.set_ylabel("Group Owen value")
    ax.set_title("E/M/X group Owen values across five dataset splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=1)
    save_fig(fig, out_dir / "qsvm_group_owen_mean_std")


def plot_group_share_mean_std(group_df: pd.DataFrame, out_dir: Path) -> None:
    share_df = add_group_shares(group_df)
    stats_df = (share_df.groupby(["r", "group"])["share_percent"]
                .agg(mean="mean", std=lambda x: np.std(x, ddof=1), n="count")
                .reset_index())
    r_values = sorted(share_df["r"].unique())
    x = np.arange(len(r_values))
    width = 0.23
    offsets = {g: (i - 1) * width for i, g in enumerate(GROUP_ORDER)}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for group in GROUP_ORDER:
        means, stds = [], []
        for r in r_values:
            row = stats_df[(stats_df["r"] == r) & (stats_df["group"] == group)]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]))
        xpos = x + offsets[group]
        ax.bar(xpos, means, width=width, yerr=stds, capsize=4, alpha=0.86,
               color=GROUP_COLORS[group], label=GROUP_LABELS[group])
        for j, r in enumerate(r_values):
            vals = share_df.loc[(share_df["r"] == r) & (share_df["group"] == group),
                                "share_percent"].to_numpy(dtype=float)
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), xpos[j]) + jitter, vals, s=22, zorder=3,
                       edgecolor="black", linewidth=0.35, color=GROUP_COLORS[group])
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={int(r)}" for r in r_values])
    ax.set_ylabel("Share of total explained gain (%)")
    ax.set_title("Normalized E/M/X contribution shares across five dataset splits")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=1)
    save_fig(fig, out_dir / "qsvm_group_share_mean_std")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style QSVM Owen plots.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/qsvm_owen_all_datasets"),
        help="Directory containing dataset_*/r*/ outputs and all_run_summaries.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots/CSVs. Default: <results-dir>/paperstyle_plots.",
    )
    parser.add_argument(
        "--dataset-index-for-gate-plot",
        type=int,
        default=0,
        help="Dataset index for dataset-specific per-gate and value-distribution plots.",
    )
    parser.add_argument(
        "--skip-value-distribution",
        action="store_true",
        help="Skip the value-distribution boxplot if you only want Owen-value plots.",
    )
    args = parser.parse_args()

    configure_matplotlib()

    results_dir = args.results_dir
    out_dir = args.output_dir if args.output_dir is not None else results_dir / "paperstyle_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_run_summaries(results_dir)
    gate_df = load_gate_values(results_dir)
    group_df = load_group_values(results_dir)

    for df in (summary_df, gate_df, group_df):
        if "r" in df.columns:
            df["r"] = df["r"].astype(int)
        if "data_index" in df.columns:
            df["data_index"] = df["data_index"].astype(int)
    summary_df["full_accuracy"] = pd.to_numeric(summary_df["full_accuracy"], errors="coerce")

    # Requested paper-like figures.
    plot_dataset_gate_owen_scatter_r123(
        gate_df, out_dir, dataset_index=args.dataset_index_for_gate_plot
    )
    for r in sorted(gate_df["r"].unique()):
        plot_gate_owen_mean_std_single_r(gate_df, out_dir, int(r))
    plot_gate_owen_mean_std_panels(gate_df, out_dir)

    if not args.skip_value_distribution:
        plot_value_distribution_dataset_r123(
            results_dir, out_dir, dataset_index=args.dataset_index_for_gate_plot
        )

    # Retained aggregate summary plots and CSVs.
    plot_accuracy_mean_std(summary_df, out_dir)
    plot_group_owen_mean_std(group_df, out_dir)
    plot_group_share_mean_std(group_df, out_dir)
    make_stats_tables(summary_df, gate_df, group_df, out_dir)

    print(f"Saved QSVM paper-style plots and statistics to: {out_dir}")
    for path in sorted(out_dir.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
