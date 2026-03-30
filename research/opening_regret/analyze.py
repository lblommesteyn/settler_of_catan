"""
Opening Regret Analysis
=======================
Loads regret_data.csv and answers:
  1. How often are players near-optimal? (regret distribution)
  2. What systematic biases explain the gap? (pip, expansion, port)
  3. Which seats are hardest for humans to play well?
  4. Are there "trap openings"? (popular but underperforming choices)
  5. Does regret increase on complex / weird boards?
  6. Do winners make different opening tradeoffs?

Columns in regret_data.csv:
  game_id, seat, won, final_vp, total_turns,
  actual_score, actual_pips, actual_resources, actual_expansion, actual_ports, actual_adj3,
  v2_regret, v2_rank, n_v2_alts, best_v2_score,
  pip_diff, exp_diff, port_diff,       <- actual vs best-v2 alternative
  v1_pip_rank, v1_n_alts, v1_pip_pct   <- how good was v1 by pip count?
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SEP = "=" * 72

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def load(path="research/opening_regret/regret_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pct_rank_v2"]  = df["v2_rank"]     / df["n_v2_alts"].clip(lower=1)
    df["pct_rank_v1"]  = df["v1_pip_rank"] / df["v1_n_alts"].clip(lower=1)
    df["top1_v2"]      = (df["v2_rank"]    == 1).astype(int)
    df["top3_v2"]      = (df["v2_rank"]    <= 3).astype(int)
    df["top5_v2"]      = (df["v2_rank"]    <= 5).astype(int)
    df["top1_v1pip"]   = (df["v1_pip_rank"] == 1).astype(int)
    df["top3_v1pip"]   = (df["v1_pip_rank"] <= 3).astype(int)
    df["seat_label"]   = df["seat"].map({0:"Seat 0\n(1st/8th)", 1:"Seat 1\n(2nd/7th)",
                                         2:"Seat 2\n(3rd/6th)", 3:"Seat 3\n(4th/5th)"})
    return df


# ─────────────────────────────────────────────────────────────────
# 1. Regret distribution
# ─────────────────────────────────────────────────────────────────

def analysis_1_regret_distribution(df):
    section("1. OVERALL REGRET DISTRIBUTION")

    n = len(df)
    print(f"\n  N = {n:,} player-openings across {df['game_id'].nunique():,} games\n")

    print("  [V2 regret: given actual v1, how much better was the best v2?]\n")
    print(f"  Mean regret:    {df['v2_regret'].mean():.4f}  (model score units, 0–1 range)")
    print(f"  Median regret:  {df['v2_regret'].median():.4f}")
    print(f"  p75 regret:     {df['v2_regret'].quantile(0.75):.4f}")
    print(f"  p90 regret:     {df['v2_regret'].quantile(0.90):.4f}")

    print(f"\n  [Rank: how often do players pick the optimal v2?]\n")
    print(f"  Top-1 (optimal):  {df['top1_v2'].mean():.1%}  of second picks")
    print(f"  Top-3:            {df['top3_v2'].mean():.1%}")
    print(f"  Top-5:            {df['top5_v2'].mean():.1%}")
    print(f"  Median pct rank:  {df['pct_rank_v2'].median():.1%}  (0=best, 1=worst; random=50%)")

    print(f"\n  [V1 quality (by pip count rank among legal alternatives)]\n")
    print(f"  Top-1 pip:  {df['top1_v1pip'].mean():.1%}  of first picks (highest pip available)")
    print(f"  Top-3 pip:  {df['top3_v1pip'].mean():.1%}")
    print(f"  Median v1 pip pct rank: {df['pct_rank_v1'].median():.1%}")

    print(f"\n  [Does v2 regret correlate with losing?]\n")
    corr = df[["v2_regret","won"]].corr().iloc[0,1]
    print(f"  Pearson r(v2_regret, won): {corr:.4f}")
    df["regret_q"] = pd.qcut(df["v2_regret"], q=4,
                              labels=["Q1 low","Q2","Q3","Q4 high"])
    rq = df.groupby("regret_q", observed=True)["won"].mean()
    print("  Win rate by v2 regret quartile (Q1=near-optimal, Q4=large mistake):")
    for q, wr in rq.items():
        print(f"    {q}: {wr:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────
# 2. Bias decomposition
# ─────────────────────────────────────────────────────────────────

def analysis_2_bias_decomposition(df):
    section("2. BIAS DECOMPOSITION — What kinds of mistakes do players make?")

    print("\n  pip_diff  = actual_pips − best_v2_pips  (positive = chose more pips than optimal)")
    print("  exp_diff  = actual_expansion − best_v2_expansion  (negative = undervalued optionality)")
    print("  port_diff = actual_ports − best_v2_ports  (positive = chased ports)\n")

    for col, label in [
        ("pip_diff",  "Pip bias (actual − best-v2 pips)"),
        ("exp_diff",  "Expansion bias (actual − best-v2 expansion)"),
        ("port_diff", "Port bias (actual − best-v2 ports)"),
    ]:
        vals = df[col].dropna()
        t_stat, p_val = stats.ttest_1samp(vals, 0)
        sig = "  ← SIGNIFICANT" if p_val < 0.01 else ""
        print(f"  {label}")
        print(f"    mean={vals.mean():+.4f}  median={vals.median():+.4f}  "
              f"std={vals.std():.3f}  p={p_val:.2e}{sig}")

    print("\n  [Bias by regret quartile: what do high-regret players do differently?]\n")
    high = df[df["regret_q"] == "Q4 high"]
    low  = df[df["regret_q"] == "Q1 low"]
    for col, label in [("pip_diff","pip bias"), ("exp_diff","exp bias"), ("port_diff","port bias")]:
        hi, lo = high[col].mean(), low[col].mean()
        print(f"    {label}: high-regret={hi:+.4f}  low-regret={lo:+.4f}  delta={hi-lo:+.4f}")

    print("\n  [Which bias best predicts v2_regret?  (Pearson r)]\n")
    for col in ["pip_diff","exp_diff","port_diff"]:
        r, p = stats.pearsonr(df[col].fillna(0), df["v2_regret"])
        print(f"    {col:<15}: r={r:+.4f}  p={p:.2e}")

    return df


# ─────────────────────────────────────────────────────────────────
# 3. Seat difficulty
# ─────────────────────────────────────────────────────────────────

def analysis_3_seat_difficulty(df):
    section("3. SEAT DIFFICULTY — Which seat is hardest to play well?")

    print()
    seat_stats = df.groupby("seat").agg(
        n=("v2_regret","count"),
        mean_v2_regret=("v2_regret","mean"),
        top1_v2=("top1_v2","mean"),
        top3_v2=("top3_v2","mean"),
        median_v2_pct_rank=("pct_rank_v2","median"),
        top3_v1pip=("top3_v1pip","mean"),
        n_v1_alts_mean=("v1_n_alts","mean"),
        pip_bias=("pip_diff","mean"),
        exp_bias=("exp_diff","mean"),
    ).round(4)
    print(seat_stats.to_string())

    print("\n  [Does v2 regret penalise win rate differently by seat?]\n")
    for s in range(4):
        sub = df[df["seat"] == s]
        hi = sub[sub["regret_q"] == "Q4 high"]["won"].mean()
        lo = sub[sub["regret_q"] == "Q1 low"]["won"].mean()
        print(f"    Seat {s}: low-regret win={lo:.3f}  high-regret win={hi:.3f}  "
              f"penalty={hi-lo:+.3f}")

    return df


# ─────────────────────────────────────────────────────────────────
# 4. Trap openings
# ─────────────────────────────────────────────────────────────────

def analysis_4_trap_openings(df):
    section("4. TRAP OPENINGS — Patterns that look good but underperform")

    print("\n  'Trap opening': player leaves little v2 regret (thinks they chose well)")
    print("  but the pattern still underperforms the baseline win rate.\n")

    df["score_q"] = pd.qcut(df["actual_score"], q=4,
                             labels=["low_score","mid_low","mid_high","high_score"])
    print("  Win rate and biases by model score quartile:\n")
    trap_stats = df.groupby("score_q", observed=True).agg(
        n=("won","count"),
        win_rate=("won","mean"),
        pip_bias=("pip_diff","mean"),
        exp_bias=("exp_diff","mean"),
        port_bias=("port_diff","mean"),
        mean_pips=("actual_pips","mean"),
        mean_exp=("actual_expansion","mean"),
    ).round(3)
    print(trap_stats.to_string())

    baseline = df["won"].mean()
    print(f"\n  Baseline win rate: {baseline:.3f}\n")

    # Specific trap patterns
    trap_pip_low_exp = df[
        (df["actual_pips"] >= df["actual_pips"].quantile(0.8)) &
        (df["actual_expansion"] <= df["actual_expansion"].quantile(0.3))
    ]
    print(f"  High pips (>p80) + low expansion (<p30):  "
          f"n={len(trap_pip_low_exp):,}  win={trap_pip_low_exp['won'].mean():.3f}  "
          f"delta={(trap_pip_low_exp['won'].mean()-baseline):+.4f}")

    trap_port = df[df["port_diff"] > 0]
    print(f"  Chose more ports than best-v2 alt:        "
          f"n={len(trap_port):,}  win={trap_port['won'].mean():.3f}  "
          f"delta={(trap_port['won'].mean()-baseline):+.4f}")

    trap_pip_chaser = df[(df["pip_diff"] > 1) & (df["exp_diff"] < -1)]
    print(f"  Chose higher pips, lost expansion:        "
          f"n={len(trap_pip_chaser):,}  win={trap_pip_chaser['won'].mean():.3f}  "
          f"delta={(trap_pip_chaser['won'].mean()-baseline):+.4f}")

    trap_exp_good = df[(df["exp_diff"] > 0) & (df["pip_diff"] < 0)]
    print(f"  Chose more expansion, fewer pips:         "
          f"n={len(trap_exp_good):,}  win={trap_exp_good['won'].mean():.3f}  "
          f"delta={(trap_exp_good['won'].mean()-baseline):+.4f}")

    trap_top1 = df[df["top1_v2"] == 1]
    print(f"\n  Players who picked optimal v2 (rank 1):   "
          f"n={len(trap_top1):,}  win={trap_top1['won'].mean():.3f}  "
          f"delta={(trap_top1['won'].mean()-baseline):+.4f}")

    return df


# ─────────────────────────────────────────────────────────────────
# 5. Board complexity and regret
# ─────────────────────────────────────────────────────────────────

def analysis_5_board_complexity(df):
    section("5. BOARD COMPLEXITY — Do players fail more on weird boards?")

    print("\n  v1_n_alts = number of legal v1 alternatives (proxy for seat constraint level)")
    print("  Lower alts = heavily constrained seat (later in draft with occupied neighbours)\n")

    try:
        df["complexity"] = pd.qcut(df["v1_n_alts"], q=4,
                                    labels=["tight","mid_tight","mid_free","free"],
                                    duplicates="drop")
    except ValueError:
        # fallback if not enough distinct values
        df["complexity"] = pd.cut(df["v1_n_alts"], bins=4,
                                   labels=["tight","mid_tight","mid_free","free"])
    c_stats = df.groupby("complexity", observed=True).agg(
        n=("v2_regret","count"),
        mean_v2_regret=("v2_regret","mean"),
        top1_v2=("top1_v2","mean"),
        med_n_alts=("v1_n_alts","median"),
        pip_bias=("pip_diff","mean"),
    ).round(4)
    print(c_stats.to_string())

    # Pip spread within game
    game_spread = df.groupby("game_id")["actual_pips"].agg(lambda x: x.max()-x.min())
    df["game_pip_spread"] = df["game_id"].map(game_spread)
    df["spread_q"] = pd.qcut(df["game_pip_spread"], q=4,
                              labels=["equal","mild","spread","chaotic"])

    print("\n  Regret by within-game pip spread:\n")
    s_stats = df.groupby("spread_q", observed=True).agg(
        n=("v2_regret","count"),
        mean_v2_regret=("v2_regret","mean"),
        top1_v2=("top1_v2","mean"),
        pip_bias=("pip_diff","mean"),
        v1_pct_rank=("pct_rank_v1","mean"),
    ).round(4)
    print(s_stats.to_string())

    print("\n  [Hypothesis: players do better on simple/spread boards]")
    eq  = df[df["spread_q"]=="equal"]["v2_regret"].mean()
    cha = df[df["spread_q"]=="chaotic"]["v2_regret"].mean()
    print(f"  Equal board regret:   {eq:.4f}")
    print(f"  Chaotic board regret: {cha:.4f}")
    t, p = stats.ttest_ind(df[df["spread_q"]=="equal"]["v2_regret"],
                            df[df["spread_q"]=="chaotic"]["v2_regret"])
    print(f"  t-test p={p:.3e}")

    return df


# ─────────────────────────────────────────────────────────────────
# 6. Winners vs losers
# ─────────────────────────────────────────────────────────────────

def analysis_6_winners_vs_losers(df):
    section("6. WINNERS vs LOSERS — Do better players make different opening tradeoffs?")

    winners = df[df["won"] == 1]
    losers  = df[df["won"] == 0]
    print(f"\n  Winners: {len(winners):,}   Losers: {len(losers):,}\n")

    metrics = {
        "v2_regret":       "V2 regret",
        "top1_v2":         "Top-1 v2 rate",
        "top3_v2":         "Top-3 v2 rate",
        "pct_rank_v2":     "V2 pct rank (lower=better)",
        "pct_rank_v1":     "V1 pip pct rank (lower=better)",
        "actual_pips":     "Actual pip count",
        "actual_expansion":"Actual expansion score",
        "actual_ports":    "Actual ports",
        "pip_diff":        "Pip bias (actual−optimal)",
        "exp_diff":        "Expansion bias",
        "port_diff":       "Port bias",
        "actual_adj3":     "Interior hex count",
        "actual_resources":"Unique resources",
    }

    print(f"  {'Metric':<32}  {'Winners':>8}  {'Losers':>8}  {'Delta':>8}  {'p-val':>10}")
    print(f"  {'-'*32}  {'-------':>8}  {'------':>8}  {'-----':>8}  {'-------':>10}")
    for col, label in metrics.items():
        if col not in df.columns:
            continue
        w, l = winners[col].mean(), losers[col].mean()
        try:
            _, p = stats.mannwhitneyu(winners[col].dropna(), losers[col].dropna())
            p_str = f"{p:.2e}"
        except Exception:
            p_str = "     n/a"
        sig = " **" if float(p_str.replace("n/a","1")) < 0.01 else "   "
        print(f"  {label:<32}  {w:>8.4f}  {l:>8.4f}  {w-l:>+8.4f}  {p_str:>10}{sig}")

    print("\n  [VP quartile: does higher final VP correlate with lower regret?]\n")
    df["vp_q"] = pd.qcut(df["final_vp"].clip(lower=2), q=4,
                          labels=["low_vp","mid_low","mid_high","high_vp"],
                          duplicates="drop")
    vp_stats = df.groupby("vp_q", observed=True).agg(
        n=("v2_regret","count"),
        mean_regret=("v2_regret","mean"),
        top1_v2=("top1_v2","mean"),
        pip_bias=("pip_diff","mean"),
        exp_bias=("exp_diff","mean"),
    ).round(4)
    print(vp_stats.to_string())

    return df


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    print("Loading regret data...")
    df = load()
    print(f"  {len(df):,} rows from {df['game_id'].nunique():,} games")

    df = analysis_1_regret_distribution(df)
    df = analysis_2_bias_decomposition(df)
    df = analysis_3_seat_difficulty(df)
    df = analysis_4_trap_openings(df)
    df = analysis_5_board_complexity(df)
    df = analysis_6_winners_vs_losers(df)

    print(f"\n{SEP}\n  DONE\n{SEP}")
    df.to_csv("research/opening_regret/regret_enriched.csv", index=False)
    print("  Saved → research/opening_regret/regret_enriched.csv")
