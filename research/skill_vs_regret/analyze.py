"""
Skill vs Opening Regret Analysis
==================================
Core question: within a game, does the player who made the best opening win more often?

Player IDs are anonymised per-game so we can't track individuals across games.
Instead we use a cleaner design: within each 4-player game, rank players by
their opening regret and ask whether regret rank predicts final outcome.

Analyses:
  1. Win rate by within-game regret rank (best/2nd/3rd/worst opener)
  2. Spearman correlation: regret rank vs VP rank within each game
  3. "Best opener wins" rate vs. null (25%)
  4. Effect size: how many extra VP does a good opening buy?
  5. Does the effect vary by board type or game length?
  6. Regret dose-response: win rate vs regret percentile (continuous)
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT   = "#e94560"
ACCENT2  = "#44aaff"
TEXT_COL = "#eaeaea"
GRID_COL = "#2a2a4a"


# ── load and prepare data ──────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = ROOT / "research" / "opening_regret" / "regret_data.csv"
    df = pd.read_csv(path)
    # keep only complete 4-player games
    game_sizes = df.groupby("game_id").size()
    full_games = game_sizes[game_sizes == 4].index
    df = df[df.game_id.isin(full_games)].copy()
    log.info("Loaded %d rows from %d complete 4-player games", len(df), df.game_id.nunique())

    # Within-game regret rank (1 = lowest regret = best opener)
    df["regret_rank"] = df.groupby("game_id")["v2_regret"].rank(method="first")

    # Within-game VP rank (1 = most VPs = winner)
    df["vp_rank"] = df.groupby("game_id")["final_vp"].rank(method="average", ascending=False)

    # Game length bucket
    df["game_length"] = pd.cut(df["total_turns"],
                                bins=[0, 60, 80, 100, 999],
                                labels=["short (<60)", "medium (60-80)",
                                        "long (80-100)", "very long (>100)"])
    return df


# ── analysis functions ─────────────────────────────────────────────────────────

def win_rate_by_regret_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate for each within-game regret rank (1=best opener, 4=worst)."""
    result = (df.groupby("regret_rank")["won"]
                .agg(["mean", "count", "sum"])
                .rename(columns={"mean": "win_rate", "count": "n", "sum": "wins"}))
    result["se"] = np.sqrt(result.win_rate * (1 - result.win_rate) / result.n)
    return result


def spearman_per_game(df: pd.DataFrame) -> dict:
    """Compute Spearman correlation between regret rank and VP rank per game,
    then aggregate."""
    corrs = []
    for _, g in df.groupby("game_id"):
        if len(g) == 4:
            r, _ = stats.spearmanr(g["regret_rank"], g["vp_rank"])
            if not np.isnan(r):
                corrs.append(r)
    corrs = np.array(corrs)
    t_stat, p_val = stats.ttest_1samp(corrs, 0)
    return {
        "mean_rho":  corrs.mean(),
        "median_rho": np.median(corrs),
        "std_rho":   corrs.std(),
        "pct_positive": (corrs > 0).mean(),
        "t_stat":    t_stat,
        "p_val":     p_val,
        "n_games":   len(corrs),
        "corrs":     corrs,
    }


def best_opener_win_rate(df: pd.DataFrame) -> dict:
    """What fraction of games does the best opener (rank 1) win?
    Null expectation = 0.25."""
    best = df[df.regret_rank == 1.0]  # only games where there's a clear rank-1
    n = len(best)
    wins = best.won.sum()
    rate = wins / n
    # binomial test vs 0.25
    binom = stats.binomtest(int(wins), n, p=0.25, alternative="greater")
    return {"rate": rate, "wins": int(wins), "n": n, "p_val": binom.pvalue}


def regret_dose_response(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Win rate vs. regret percentile (continuous dose-response)."""
    df = df.copy()
    df["regret_pct"] = df.groupby("game_id")["v2_regret"].rank(pct=True)
    df["pct_bin"] = pd.cut(df["regret_pct"], bins=n_bins, labels=False)
    result = df.groupby("pct_bin")["won"].agg(["mean", "count"]).reset_index()
    result["bin_centre"] = (result["pct_bin"] + 0.5) / n_bins
    result["se"] = np.sqrt(result["mean"] * (1 - result["mean"]) / result["count"])
    return result


def vp_effect(df: pd.DataFrame) -> dict:
    """Average VP difference: best opener vs worst opener in same game."""
    game_stats = []
    for _, g in df.groupby("game_id"):
        if len(g) == 4:
            g_sorted = g.sort_values("regret_rank")
            best_vp  = g_sorted.iloc[0].final_vp
            worst_vp = g_sorted.iloc[-1].final_vp
            game_stats.append(best_vp - worst_vp)
    arr = np.array(game_stats)
    t, p = stats.ttest_1samp(arr, 0)
    return {"mean_diff": arr.mean(), "std": arr.std(),
            "t": t, "p": p, "n": len(arr)}


def game_length_moderation(df: pd.DataFrame) -> pd.DataFrame:
    """Does regret-win relationship vary by game length?"""
    rows = []
    for length, g in df.groupby("game_length", observed=True):
        best = g[g.regret_rank == 1.0]
        if len(best) < 50:
            continue
        rate = best.won.mean()
        se   = np.sqrt(rate * (1 - rate) / len(best))
        rows.append({"game_length": length, "win_rate": rate, "se": se, "n": len(best)})
    return pd.DataFrame(rows)


# ── figures ────────────────────────────────────────────────────────────────────

def fig1_win_rate_by_rank(rank_stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ranks  = rank_stats.index.values
    rates  = rank_stats.win_rate.values * 100
    errors = rank_stats.se.values * 100
    colours = [ACCENT2, "#88ccff", "#ffaa44", ACCENT]

    bars = ax.bar(ranks, rates, color=colours, width=0.6, zorder=3)
    ax.errorbar(ranks, rates, yerr=errors * 1.96, fmt="none",
                color="white", capsize=4, lw=1.5, zorder=4)
    ax.axhline(25, color="white", lw=1.2, ls="--", alpha=0.6, label="Null (25%)", zorder=2)

    for bar, r, n in zip(bars, rates, rank_stats.n.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{r:.1f}%", ha="center", va="bottom", color=TEXT_COL, fontsize=10, fontweight="bold")

    ax.set_xticks(ranks)
    ax.set_xticklabels(["Best opener\n(rank 1)", "2nd best\n(rank 2)",
                         "2nd worst\n(rank 3)", "Worst opener\n(rank 4)"],
                        color=TEXT_COL, fontsize=9)
    ax.set_ylabel("Win rate (%)", color=TEXT_COL)
    ax.set_title("Win Rate by Within-Game Opening Quality Rank", color=TEXT_COL, pad=10)
    ax.set_ylim(0, 38)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)

    fig.tight_layout()
    p = OUT_DIR / "fig1_win_rate_by_rank.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig2_spearman_dist(spearman: dict):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    corrs = spearman["corrs"]
    ax.hist(corrs, bins=60, color=ACCENT2, edgecolor="none", alpha=0.85)
    ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.6)
    ax.axvline(spearman["mean_rho"], color=ACCENT, lw=2,
               label=f'Mean r = {spearman["mean_rho"]:.3f}  (p={spearman["p_val"]:.2e})')

    ax.set_xlabel("Spearman r (regret rank vs VP rank, per game)", color=TEXT_COL)
    ax.set_ylabel("Number of games", color=TEXT_COL)
    ax.set_title("Per-Game Correlation: Opening Regret Rank vs Final VP Rank", color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.legend(fontsize=9, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)

    pct_pos = spearman["pct_positive"] * 100
    ax.text(0.98, 0.95, f"{pct_pos:.0f}% of games: better opener finishes higher",
            transform=ax.transAxes, ha="right", va="top",
            color=TEXT_COL, fontsize=8, alpha=0.8)

    fig.tight_layout()
    p = OUT_DIR / "fig2_spearman_dist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig3_dose_response(dose: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    x = dose.bin_centre.values * 100
    y = dose["mean"].values * 100
    e = dose["se"].values * 100 * 1.96

    ax.fill_between(x, y - e, y + e, color=ACCENT2, alpha=0.2)
    ax.plot(x, y, color=ACCENT2, lw=2, marker="o", ms=5, zorder=3)
    ax.axhline(25, color="white", lw=1.2, ls="--", alpha=0.6, label="Null (25%)")

    # Linear fit
    slope, intercept, r, p, _ = stats.linregress(x, y)
    xfit = np.linspace(x.min(), x.max(), 100)
    ax.plot(xfit, intercept + slope * xfit, color=ACCENT, lw=1.5, ls="--",
            label=f"Linear fit (r={r:.2f}, p={p:.3f})")

    ax.set_xlabel("Opening regret percentile within game (0=best, 100=worst)", color=TEXT_COL)
    ax.set_ylabel("Win rate (%)", color=TEXT_COL)
    ax.set_title("Win Rate vs. Opening Regret: Dose-Response", color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)

    fig.tight_layout()
    p = OUT_DIR / "fig3_dose_response.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig4_game_length(length_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    colours = [ACCENT2, "#88ccff", "#ffaa44", ACCENT]
    y = np.arange(len(length_df))
    rates = length_df.win_rate.values * 100
    errors = length_df.se.values * 100 * 1.96

    bars = ax.barh(y, rates, color=colours[:len(length_df)], height=0.5)
    ax.errorbar(rates, y, xerr=errors, fmt="none", color="white", capsize=3, lw=1.2)
    ax.axvline(25, color="white", lw=1.2, ls="--", alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(length_df.game_length.values, color=TEXT_COL, fontsize=9)
    ax.set_xlabel("Win rate of best opener (%)", color=TEXT_COL)
    ax.set_title("Effect of Opening Quality by Game Length\n(best opener in game)", color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.xaxis.grid(True, color=GRID_COL, zorder=0)

    for bar, r, n in zip(bars, rates, length_df.n.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{r:.1f}%  (n={n})", va="center", color=TEXT_COL, fontsize=8)

    fig.tight_layout()
    p = OUT_DIR / "fig4_game_length.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig5_summary(rank_stats, spearman, best_opener, vp_eff):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=DARK_BG)
    fig.suptitle("Does Opening Quality Predict Winning?", color=TEXT_COL, fontsize=13)

    # Left: win rate by rank (compact)
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ranks  = rank_stats.index.values
    rates  = rank_stats.win_rate.values * 100
    errors = rank_stats.se.values * 100
    colours = [ACCENT2, "#88ccff", "#ffaa44", ACCENT]
    bars = ax.bar(ranks, rates, color=colours, width=0.6, zorder=3)
    ax.errorbar(ranks, rates, yerr=errors * 1.96, fmt="none",
                color="white", capsize=3, lw=1.2, zorder=4)
    ax.axhline(25, color="white", lw=1.2, ls="--", alpha=0.6)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{r:.1f}%", ha="center", color=TEXT_COL, fontsize=9, fontweight="bold")
    ax.set_xticks(ranks)
    ax.set_xticklabels(["Best", "2nd", "3rd", "Worst"], color=TEXT_COL)
    ax.set_ylabel("Win rate (%)", color=TEXT_COL)
    ax.set_title("Win Rate by Opening Rank", color=TEXT_COL)
    ax.set_ylim(0, 36)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)

    # Right: key stats panel
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    stats_text = [
        ("Best opener win rate",     f"{best_opener['rate']*100:.1f}%",  f"vs 25% null  p={best_opener['p_val']:.3f}"),
        ("Spearman r (regret→rank)", f"{spearman['mean_rho']:.3f}",      f"p={spearman['p_val']:.2e}"),
        ("% games: better opener\nfinishes higher", f"{spearman['pct_positive']*100:.0f}%", ""),
        ("VP advantage: best vs\nworst opener",  f"+{vp_eff['mean_diff']:.2f} VP",  f"p={vp_eff['p']:.3f}"),
        ("Games analysed",           f"{spearman['n_games']:,}", "complete 4-player games"),
    ]

    y_pos = 0.88
    for label, value, note in stats_text:
        ax.text(0.05, y_pos, label, transform=ax.transAxes,
                color=TEXT_COL, fontsize=9, alpha=0.7)
        ax.text(0.55, y_pos, value, transform=ax.transAxes,
                color=ACCENT2, fontsize=11, fontweight="bold")
        if note:
            ax.text(0.55, y_pos - 0.065, note, transform=ax.transAxes,
                    color=TEXT_COL, fontsize=7.5, alpha=0.6)
        y_pos -= 0.17

    fig.tight_layout()
    p = OUT_DIR / "fig5_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    log.info("Running analyses ...")
    rank_stats   = win_rate_by_regret_rank(df)
    spearman     = spearman_per_game(df)
    best_opener  = best_opener_win_rate(df)
    dose         = regret_dose_response(df)
    vp_eff       = vp_effect(df)
    length_df    = game_length_moderation(df)

    log.info("=== Key Results ===")
    log.info("Win rate by opening rank:")
    for idx, row in rank_stats.iterrows():
        log.info("  Rank %d: %.1f%%  (n=%d)", idx, row.win_rate * 100, row.n)
    log.info("Best opener win rate: %.1f%%  p=%.4f", best_opener["rate"] * 100, best_opener["p_val"])
    log.info("Spearman r (regret->VP rank): %.3f  p=%.2e", spearman["mean_rho"], spearman["p_val"])
    log.info("Pct games: better opener finishes higher: %.0f%%", spearman["pct_positive"] * 100)
    log.info("VP advantage best vs worst opener: +%.2f  p=%.4f", vp_eff["mean_diff"], vp_eff["p"])

    log.info("Generating figures ...")
    fig1_win_rate_by_rank(rank_stats)
    fig2_spearman_dist(spearman)
    fig3_dose_response(dose)
    fig4_game_length(length_df)
    fig5_summary(rank_stats, spearman, best_opener, vp_eff)
    log.info("Done. Figures saved to %s", OUT_DIR)

    return {
        "rank_stats": rank_stats,
        "spearman": spearman,
        "best_opener": best_opener,
        "dose": dose,
        "vp_eff": vp_eff,
        "length_df": length_df,
    }


if __name__ == "__main__":
    main()
