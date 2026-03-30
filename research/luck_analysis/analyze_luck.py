"""
analyze_luck.py — Variance decomposition: opening quality vs. dice luck.

Joins luck_data.csv (from parse_luck.py) with regret_data.csv.
Fits three OLS models and reports R²:

  Model A: won ~ opening features only
  Model B: won ~ luck_ratio only
  Model C: won ~ opening features + luck_ratio

Headline: "X% of Catan outcomes are explained by luck vs Y% by the opening."

Figures:
  fig1_luck_distribution.png   — luck_ratio histogram + per-quartile win rate
  fig2_luck_vs_win.png         — luck_ratio vs win rate (binned)
  fig3_variance_decomposition.png — R² bar chart for A / B / C
  fig4_luck_opening_scatter.png   — luck vs v2_regret coloured by won
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "src"))

LUCK_CSV   = Path(__file__).parent / "luck_data.csv"
REGRET_CSV = REPO / "research" / "opening_regret" / "regret_data.csv"
FIG_DIR    = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    luck   = pd.read_csv(LUCK_CSV)
    regret = pd.read_csv(REGRET_CSV)

    luck["game_id"]   = luck["game_id"].astype(str)
    regret["game_id"] = regret["game_id"].astype(str)

    df = luck.merge(regret, on=["game_id", "seat", "won"], how="inner", suffixes=("", "_r"))

    # keep only 4-player complete games
    game_counts = df.groupby("game_id")["seat"].count()
    complete    = game_counts[game_counts == 4].index
    df = df[df["game_id"].isin(complete)].copy()

    print(f"Merged: {len(df):,} rows from {df['game_id'].nunique():,} games")
    return df


# ---------------------------------------------------------------------------
# OLS R²  (linear probability model — standard for binary outcomes + R²)
# ---------------------------------------------------------------------------

def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Return R² of OLS y ~ X (adds intercept automatically)."""
    X_c = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
    y_hat = X_c @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame) -> None:
    y = df["won"].values.astype(float)

    # Opening features: v2_regret (negative — lower regret = better) + actual_pips
    opening_X = df[["v2_regret", "actual_pips"]].values
    luck_X    = df[["luck_ratio"]].values
    combined_X = np.column_stack([opening_X, luck_X])

    r2_opening  = ols_r2(opening_X, y)
    r2_luck     = ols_r2(luck_X,    y)
    r2_combined = ols_r2(combined_X, y)

    print(f"\n=== Variance decomposition ===")
    print(f"Model A (opening only):      R² = {r2_opening*100:.3f}%")
    print(f"Model B (luck only):         R² = {r2_luck*100:.3f}%")
    print(f"Model C (opening + luck):    R² = {r2_combined*100:.3f}%")
    print(f"Incremental luck (C-A):      dR2 = {(r2_combined-r2_opening)*100:.3f}%")

    # Descriptive stats on luck_ratio
    lr = df["luck_ratio"]
    print(f"\n=== Luck ratio descriptives ===")
    print(f"Mean:   {lr.mean():.4f}  (1.0 = exactly as lucky as expected)")
    print(f"Std:    {lr.std():.4f}")
    print(f"P25/P75: {lr.quantile(0.25):.4f} / {lr.quantile(0.75):.4f}")
    print(f"Min/Max: {lr.min():.4f} / {lr.max():.4f}")

    # Win rate by luck quartile
    df["luck_q"] = pd.qcut(lr, 4, labels=["Q1\n(unlucky)", "Q2", "Q3", "Q4\n(lucky)"])
    wr_by_q = df.groupby("luck_q", observed=True)["won"].mean()
    print(f"\n=== Win rate by luck quartile ===")
    for q, wr in wr_by_q.items():
        print(f"  {q}: {wr:.1%}")

    # Pearson r: luck vs won, regret vs won
    r_luck,   p_luck   = stats.pearsonr(df["luck_ratio"], y)
    r_regret, p_regret = stats.pearsonr(df["v2_regret"],  y)
    print(f"\n=== Correlations with winning ===")
    print(f"Luck ratio:   r={r_luck:.4f}  p={p_luck:.4f}")
    print(f"v2 regret:    r={r_regret:.4f}  p={p_regret:.4f}")

    plot_all(df, r2_opening, r2_luck, r2_combined, wr_by_q)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BLUE   = "#4878CF"
ORANGE = "#E68B33"
GREEN  = "#6ACC65"
GREY   = "#999999"

def plot_all(
    df: pd.DataFrame,
    r2_a: float, r2_b: float, r2_c: float,
    wr_by_q: pd.Series,
) -> None:

    # --- Fig 1: luck distribution + win rate by quartile ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(df["luck_ratio"], bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Expected (1.0)")
    ax.set_xlabel("Luck ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of luck ratios")
    ax.legend(fontsize=9)

    ax = axes[1]
    qs = wr_by_q.index.tolist()
    wrs = wr_by_q.values
    bars = ax.bar(qs, wrs, color=[BLUE, BLUE, ORANGE, ORANGE], edgecolor="white", linewidth=0.5)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.2, label="Null (25%)")
    ax.set_xlabel("Luck quartile")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate by luck quartile")
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=9)
    for bar, wr in zip(bars, wrs):
        ax.text(bar.get_x() + bar.get_width()/2, wr + 0.005,
                f"{wr:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_luck_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved fig1_luck_distribution.png")

    # --- Fig 2: luck ratio vs win rate (binned scatter) ---
    df2 = df.copy()
    df2["lr_bin"] = pd.cut(df2["luck_ratio"], bins=20)
    binned = df2.groupby("lr_bin", observed=True).agg(
        win_rate=("won", "mean"),
        n=("won", "count"),
        mid=("luck_ratio", "mean"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(binned["mid"], binned["win_rate"],
                    s=binned["n"] / binned["n"].max() * 300 + 20,
                    c=binned["mid"], cmap="RdYlGn", vmin=0.5, vmax=1.5,
                    edgecolors="grey", linewidth=0.4, alpha=0.85)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.2, label="Null (25%)")
    ax.axvline(1.0,  color="grey",  linestyle=":",  linewidth=1.0)
    ax.set_xlabel("Luck ratio (actual hits / expected hits)")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate vs. dice luck")
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Luck ratio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_luck_vs_win.png", dpi=150)
    plt.close(fig)
    print("Saved fig2_luck_vs_win.png")

    # --- Fig 3: Variance decomposition bar chart ---
    labels  = ["Opening\nonly", "Luck\nonly", "Opening\n+ Luck"]
    r2_vals = [r2_a * 100, r2_b * 100, r2_c * 100]
    colours = [BLUE, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, r2_vals, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("R² (%)")
    ax.set_title("Variance in winning explained by opening vs. luck")
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.3f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_variance_decomposition.png", dpi=150)
    plt.close(fig)
    print("Saved fig3_variance_decomposition.png")

    # --- Fig 4: luck vs regret scatter (coloured by won) ---
    sample = df.sample(min(5000, len(df)), random_state=42)
    won_mask  = sample["won"] == 1
    lost_mask = sample["won"] == 0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample.loc[lost_mask, "v2_regret"],
               sample.loc[lost_mask, "luck_ratio"],
               s=6, alpha=0.25, color=GREY,  label="Lost")
    ax.scatter(sample.loc[won_mask,  "v2_regret"],
               sample.loc[won_mask,  "luck_ratio"],
               s=6, alpha=0.35, color=ORANGE, label="Won")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("v2 regret (higher = worse opening)")
    ax.set_ylabel("Luck ratio")
    ax.set_title("Opening quality vs. luck (sample n=5,000)")
    ax.legend(fontsize=9, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_luck_opening_scatter.png", dpi=150)
    plt.close(fig)
    print("Saved fig4_luck_opening_scatter.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not LUCK_CSV.exists():
        print(f"ERROR: {LUCK_CSV} not found. Run parse_luck.py first.")
        sys.exit(1)

    df = load_data()
    run_analysis(df)
