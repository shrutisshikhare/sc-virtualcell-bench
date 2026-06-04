"""Generate leaderboard table and radar chart from results/leaderboard.csv."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.metrics import mean_rank_score

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"

METRIC_COLS = ["PDS", "DES", "MAE", "PDC", "SLC", "AUP", "SES"]
# MAE: lower is better; all others: higher is better
ASCENDING = {"MAE"}


def load_leaderboard(path: str | Path = LEADERBOARD_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="model")
    return df


def print_leaderboard(df: pd.DataFrame) -> None:
    available = [c for c in METRIC_COLS if c in df.columns]
    display = df[available].copy()
    display["mean_rank"] = mean_rank_score(display)
    display = display.sort_values("mean_rank")
    print("\n=== Leaderboard (ranked by mean rank across 7 metrics) ===")
    print(display.round(4).to_string())
    print()


def plot_radar(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: str = "Model Comparison — Cell-Eval 7 Metrics",
) -> Path:
    """Radar chart: each model is a polygon over the 7 normalised metrics."""
    out_path = out_path or FIGURES_DIR / "radar_chart.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    available = [c for c in METRIC_COLS if c in df.columns]
    sub = df[available].copy()

    # normalise each metric to [0, 1] (flip MAE so higher = better)
    normed = pd.DataFrame(index=sub.index)
    for col in available:
        col_min, col_max = sub[col].min(), sub[col].max()
        rng = col_max - col_min
        if rng < 1e-10:
            normed[col] = 0.5
        elif col in ASCENDING:
            normed[col] = 1.0 - (sub[col] - col_min) / rng  # flip
        else:
            normed[col] = (sub[col] - col_min) / rng

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    for i, (model, row) in enumerate(normed.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=1.8, label=model, color=colors[i % 10])
        ax.fill(angles, values, alpha=0.07, color=colors[i % 10])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title(title, pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar chart saved → {out_path}")
    return out_path


def plot_pds_vs_des(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> Path:
    """Scatter plot of PDS vs DES showing the trade-off landscape."""
    out_path = out_path or FIGURES_DIR / "pds_vs_des.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "PDS" not in df.columns or "DES" not in df.columns:
        raise ValueError("DataFrame must contain PDS and DES columns.")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10.colors
    for i, (model, row) in enumerate(df.iterrows()):
        ax.scatter(row["PDS"], row["DES"], s=120, color=colors[i % 10], zorder=3)
        ax.annotate(model, (row["PDS"], row["DES"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=8)
    ax.set_xlabel("PDS (higher = better)", fontsize=11)
    ax.set_ylabel("DES (higher = better)", fontsize=11)
    ax.set_title("PDS vs DES Trade-off", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PDS vs DES scatter saved → {out_path}")
    return out_path


def generate_leaderboard(
    csv_path: str | Path = LEADERBOARD_CSV,
    save_plots: bool = True,
) -> pd.DataFrame:
    """Load CSV, print ranked table, optionally save visualisations."""
    df = load_leaderboard(csv_path)
    print_leaderboard(df)
    if save_plots:
        plot_radar(df)
        if "PDS" in df.columns and "DES" in df.columns:
            plot_pds_vs_des(df)
    return df


if __name__ == "__main__":
    generate_leaderboard()
