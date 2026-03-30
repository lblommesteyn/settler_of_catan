"""
Catan Opening Intelligence — Research Analysis
===============================================

Three investigations:
  1. Seat dynamics — does position in the snake draft change what matters?
  2. Human vs optimal — how far below model-optimal do real players place?
  3. Board variance — on which boards does opening choice matter most?

Run:
    python analysis.py

Produces analysis output to stdout (redirect to a file for capture).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from catan.features.opening_features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(path: str = "data/processed/dataset.npz") -> pd.DataFrame:
    d = np.load(path, allow_pickle=True)
    df = pd.DataFrame(d["X"], columns=FEATURE_NAMES)
    df["won"] = d["y"].astype(int)
    df["seat"] = d["seats"].astype(int)
    if "final_vps" in d:
        df["final_vp"] = d["final_vps"].astype(int)
        df["final_rank"] = d["final_ranks"].astype(int)
        df["game_id"] = d["game_ids"]
    else:
        df["final_vp"] = np.nan
        df["final_rank"] = np.nan
        df["game_id"] = np.arange(len(df)).astype(str)
    return df


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def identify_archetype(row) -> str:
    scores = {
        "ore_wheat":   row["ore_wheat_score"],
        "road_race":   row["road_race_score"],
        "port_engine": row["port_engine_score"],
        "balanced":    row["balanced_score"],
        "high_pip":    row["high_pip_score"],
    }
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# 1. Seat dynamics
# ---------------------------------------------------------------------------

def analyze_seat_dynamics(df: pd.DataFrame):
    section("1. SEAT DYNAMICS — Does draft position change what matters?")

    # 1a. Win rate by seat
    print("\n[1a] Win rate by seat position")
    print("     (Expected ~25% for 4 players; early seats pick first)\n")
    for s in range(4):
        mask = df["seat"] == s
        n = mask.sum()
        wr = df.loc[mask, "won"].mean()
        print(f"  Seat {s} (picks {'1st,8th' if s==0 else '2nd,7th' if s==1 else '3rd,6th' if s==2 else '4th,5th'}): "
              f"n={n:,}  win_rate={wr:.3f}")

    # 1b. What do each seat actually pick? (pip counts, archetypes)
    print("\n[1b] Opening choices by seat (what humans actually place)\n")
    df["archetype"] = df.apply(identify_archetype, axis=1)
    for s in range(4):
        sub = df[df["seat"] == s]
        pip_med = sub["combined_pip_count"].median()
        pip_mean = sub["combined_pip_count"].mean()
        top_arch = sub["archetype"].value_counts().index[0]
        port_rate = sub["num_ports"].gt(0).mean()
        print(f"  Seat {s}: pips median={pip_med:.1f} mean={pip_mean:.1f}  "
              f"port_rate={port_rate:.2f}  top_archetype={top_arch}")

    # 1c. Feature importance per seat (logistic regression coefficients)
    print("\n[1c] Which features predict wins differently by seat?\n")
    feature_cols = [c for c in FEATURE_NAMES if c != "seat"]
    key_features = [
        "combined_pip_count", "unique_resource_count", "expansion_pip_sum",
        "num_ports", "ore_wheat_score", "road_race_score",
        "v1_num_adjacent_hexes", "v2_num_adjacent_hexes",
        "combined_city_pips", "combined_settlement_pips",
    ]

    seat_coefs = {}
    for s in range(4):
        sub = df[df["seat"] == s]
        X = sub[key_features].values
        y = sub["won"].values
        if y.mean() == 0 or y.mean() == 1:
            continue
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1, max_iter=500))])
        pipe.fit(X, y)
        seat_coefs[s] = dict(zip(key_features, pipe.named_steps["lr"].coef_[0]))

    print(f"  {'Feature':<35}", end="")
    for s in range(4):
        print(f"  Seat {s}", end="")
    print()
    print(f"  {'-'*35}", end="")
    for s in range(4):
        print(f"  ------", end="")
    print()

    for feat in key_features:
        print(f"  {feat:<35}", end="")
        for s in range(4):
            coef = seat_coefs.get(s, {}).get(feat, 0)
            marker = " *" if abs(coef) > 0.05 else "  "
            print(f"  {coef:+.3f}{marker}", end="")
        print()

    # 1d. Pip count sweet spot by seat
    print("\n[1d] Win rate by pip-count decile per seat\n")
    df["pip_decile"] = pd.qcut(df["combined_pip_count"], q=5,
                               labels=["very_low","low","mid","high","very_high"])
    pivot = df.groupby(["seat", "pip_decile"])["won"].mean().unstack("pip_decile")
    pivot.columns = [str(c) for c in pivot.columns]
    print(pivot.round(3).to_string())

    # 1e. Do early seats need higher pips to compensate?
    print("\n[1e] Optimal pip level by seat (pip-count at peak win rate)\n")
    for s in range(4):
        sub = df[df["seat"] == s].copy()
        # bin pips into 4-pip buckets
        sub["pip_bin"] = (sub["combined_pip_count"] // 4) * 4
        grp = sub.groupby("pip_bin")["won"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 50]
        if grp.empty:
            continue
        best_bin = grp["mean"].idxmax()
        best_wr = grp["mean"].max()
        baseline = sub["won"].mean()
        print(f"  Seat {s}: peak win_rate={best_wr:.3f} at pips ~{best_bin}-{best_bin+3}  "
              f"(baseline={baseline:.3f}, delta={best_wr-baseline:+.3f})")


# ---------------------------------------------------------------------------
# 2. Human vs Optimal
# ---------------------------------------------------------------------------

def analyze_human_vs_optimal(df: pd.DataFrame):
    section("2. HUMAN vs OPTIMAL — How far below best do real players place?")

    has_game_id = "game_id" in df.columns and df["game_id"].dtype == object

    # 2a. Pip distribution of actual human choices
    print("\n[2a] Distribution of combined pip counts (actual human choices)\n")
    pip_bins = [0, 10, 14, 18, 22, 26, 30, 100]
    pip_labels = ["<10", "10-13", "14-17", "18-21", "22-25", "26-29", "30+"]
    df["pip_group"] = pd.cut(df["combined_pip_count"], bins=pip_bins, labels=pip_labels)
    grp = df.groupby("pip_group", observed=True)["won"].agg(["count", "mean"])
    grp.columns = ["n_openings", "win_rate"]
    grp["pct_of_total"] = grp["n_openings"] / len(df) * 100
    print(grp.round(3).to_string())

    # 2b. Win rate vs pip count (are humans leaving value on the table?)
    print("\n[2b] Win rate by pip count — where is the value?\n")
    sub = df.groupby(df["combined_pip_count"].round(0))["won"].agg(["mean", "count"])
    sub = sub[sub["count"] >= 100]
    print(f"  {'Pips':>6}  {'Win%':>6}  {'N':>7}  Bar")
    print(f"  {'----':>6}  {'----':>6}  {'---':>7}")
    for pips, row in sub.iterrows():
        bar = "#" * int(row["mean"] * 100)
        print(f"  {pips:>6.0f}  {row['mean']*100:>5.1f}%  {row['count']:>7,}  {bar}")

    # 2c. Archetype frequency vs win rate (popular ≠ optimal)
    print("\n[2c] Archetype: frequency vs win rate (popular-but-suboptimal check)\n")
    df["archetype"] = df.apply(identify_archetype, axis=1)
    arch_stats = df.groupby("archetype")["won"].agg(["count", "mean"]).sort_values("count", ascending=False)
    arch_stats.columns = ["n_openings", "win_rate"]
    arch_stats["pct_of_total"] = arch_stats["n_openings"] / len(df) * 100
    arch_stats["vs_baseline"] = arch_stats["win_rate"] - df["won"].mean()
    print(arch_stats.round(3).to_string())

    # 2d. Are humans over-indexing on production resources vs city path?
    print("\n[2d] Settlement vs city resource focus — human tendency and value\n")
    df["focus"] = pd.cut(
        df["combined_city_pips"] / (df["combined_pip_count"].clip(lower=1)),
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["pure_settle", "settle_lean", "balanced", "city_lean", "pure_city"],
    )
    focus_stats = df.groupby("focus", observed=True)["won"].agg(["count", "mean"])
    focus_stats.columns = ["n_openings", "win_rate"]
    focus_stats["pct"] = focus_stats["n_openings"] / len(df) * 100
    print(focus_stats.round(3).to_string())

    # 2e. Port value — do ports actually help?
    print("\n[2e] Port access — does having a port help?\n")
    for ports in [0, 1, 2]:
        mask = df["num_ports"] == ports
        n = mask.sum()
        if n < 100:
            continue
        wr = df.loc[mask, "won"].mean()
        pip_med = df.loc[mask, "combined_pip_count"].median()
        print(f"  {ports} port(s): n={n:,}  win_rate={wr:.3f}  median_pips={pip_med:.1f}")

    print("\n  Port synergy (matched port+resource vs unmatched):")
    synergy_mask = df["port_synergy_score"] > 0.5
    for label, mask in [("synergy_port", synergy_mask), ("no_synergy_port", ~synergy_mask & (df["num_ports"] > 0)), ("no_port", df["num_ports"] == 0)]:
        n = mask.sum()
        if n < 100:
            continue
        wr = df.loc[mask, "won"].mean()
        print(f"    {label:<20}: n={n:,}  win_rate={wr:.3f}")

    # 2f. Resource diversity premium
    print("\n[2f] Unique resource count vs win rate\n")
    for n_res in range(1, 6):
        mask = df["unique_resource_count"] == n_res
        n = mask.sum()
        if n < 100:
            continue
        wr = df.loc[mask, "won"].mean()
        pip_med = df.loc[mask, "combined_pip_count"].median()
        print(f"  {n_res} unique resources: n={n:,}  win_rate={wr:.3f}  median_pips={pip_med:.1f}")

    # 2g. Per-game gap analysis (requires game_id)
    if has_game_id and df["game_id"].notna().any():
        print("\n[2g] Within-game gap: best vs worst opening in same game\n")
        # For each game, find pip spread between best and worst
        game_stats = df.groupby("game_id")["combined_pip_count"].agg(["max", "min", "std"])
        game_stats["pip_spread"] = game_stats["max"] - game_stats["min"]
        # Does pip spread predict outcome? (high spread = someone got worse deal)
        df2 = df.merge(game_stats["pip_spread"].reset_index(), on="game_id")
        df2["got_best"] = df2.groupby("game_id")["combined_pip_count"].transform(
            lambda x: x == x.max()
        )
        got_best_wr = df2[df2["got_best"]]["won"].mean()
        got_worst_wr = df2[~df2["got_best"]]["won"].mean()
        print(f"  Player with highest pips in game: win_rate={got_best_wr:.3f}")
        print(f"  Other players:                    win_rate={got_worst_wr:.3f}")

        # Pip spread quartile analysis
        df2["spread_q"] = pd.qcut(df2["pip_spread"], q=4,
                                   labels=["tight","moderate","spread","wide"])
        print("\n  Win rate for top pip player by game spread:")
        for q_label in ["tight","moderate","spread","wide"]:
            sub = df2[(df2["spread_q"] == q_label) & df2["got_best"]]
            if len(sub) < 50:
                continue
            wr = sub["won"].mean()
            print(f"    {q_label:<10}: win_rate={wr:.3f} (n={len(sub):,})")


# ---------------------------------------------------------------------------
# 3. Board Variance
# ---------------------------------------------------------------------------

def analyze_board_variance(df: pd.DataFrame):
    section("3. BOARD VARIANCE — On which boards does opening matter most?")

    has_game_id = "game_id" in df.columns and df["game_id"].dtype == object

    if not has_game_id or not df["game_id"].notna().any():
        print("\n  [skipped — requires game_id metadata; reprocess with updated loader]")
        return

    # 3a. Per-game pip spread (proxy for board quality variance)
    print("\n[3a] Per-game pip spread distribution\n")
    game_pips = df.groupby("game_id")["combined_pip_count"].agg(["max", "min", "mean", "std"])
    game_pips["spread"] = game_pips["max"] - game_pips["min"]
    print(f"  Games analyzed: {len(game_pips):,}")
    print(f"  Pip spread  mean={game_pips['spread'].mean():.1f}  "
          f"median={game_pips['spread'].median():.1f}  "
          f"p75={game_pips['spread'].quantile(0.75):.1f}  "
          f"p90={game_pips['spread'].quantile(0.90):.1f}  "
          f"max={game_pips['spread'].max():.1f}")

    # 3b. Does opening variance predict faster games? (more decisive outcomes)
    if "final_vp" in df.columns and df["final_vp"].notna().any():
        print("\n[3b] Does high opening-quality spread -> more decisive outcomes?\n")
        spread_df = game_pips["spread"].reset_index()
        df2 = df.merge(spread_df, on="game_id")
        df2["spread_q"] = pd.qcut(df2["spread"], q=4,
                                   labels=["equal","mild","spread","extreme"])
        print("  Mean final VP by opening-spread quartile:")
        vp_by_spread = df2.groupby("spread_q", observed=True)["final_vp"].mean()
        for q_label, vp in vp_by_spread.items():
            print(f"    {q_label:<10}: avg_final_vp={vp:.2f}")

    # 3c. High-variance boards: where opening selection has most impact
    print("\n[3c] Board entropy: boards where player pip spreads are extreme\n")
    high_spread = game_pips[game_pips["spread"] >= game_pips["spread"].quantile(0.9)]
    low_spread = game_pips[game_pips["spread"] <= game_pips["spread"].quantile(0.1)]

    df_high = df[df["game_id"].isin(high_spread.index)]
    df_low = df[df["game_id"].isin(low_spread.index)]

    # Does having the best opening on a high-spread board confer more advantage?
    for label, subset in [("high_spread_boards", df_high), ("low_spread_boards", df_low)]:
        sub = subset.copy()
        sub["got_best"] = sub.groupby("game_id")["combined_pip_count"].transform(
            lambda x: x == x.max()
        )
        wr_best = sub[sub["got_best"]]["won"].mean()
        wr_rest = sub[~sub["got_best"]]["won"].mean()
        n_games = sub["game_id"].nunique()
        print(f"  {label} (n={n_games:,} games):")
        print(f"    Top-pip player win_rate:   {wr_best:.3f}")
        print(f"    Others win_rate:           {wr_rest:.3f}")
        print(f"    Advantage of best opening: {wr_best - wr_rest:+.3f}\n")

    # 3d. Feature stability across board types
    print("[3d] Does feature importance shift between balanced vs unbalanced boards?\n")
    key_features = [
        "combined_pip_count", "unique_resource_count", "expansion_pip_sum",
        "combined_city_pips", "num_ports",
    ]
    results = {}
    for label, subset in [("balanced_boards", df_low), ("unbalanced_boards", df_high)]:
        X = subset[key_features].values
        y_sub = subset["won"].values
        if y_sub.mean() in (0, 1) or len(subset) < 500:
            continue
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1, max_iter=500))])
        pipe.fit(X, y_sub)
        results[label] = dict(zip(key_features, pipe.named_steps["lr"].coef_[0]))

    if results:
        print(f"  {'Feature':<35}", end="")
        for label in results:
            print(f"  {label[:20]}", end="")
        print()
        for feat in key_features:
            print(f"  {feat:<35}", end="")
            for label in results:
                coef = results[label].get(feat, 0)
                print(f"  {coef:+.4f}             ", end="")
            print()


# ---------------------------------------------------------------------------
# 4. Model performance deep-dive
# ---------------------------------------------------------------------------

def analyze_model_performance(df: pd.DataFrame):
    section("4. MODEL PERFORMANCE — Where does the model add (and lose) value?")

    feature_cols = FEATURE_NAMES
    X = df[feature_cols].values
    y = df["won"].values

    # 4a. AUC by seat
    print("\n[4a] AUC-ROC per seat (same model trained globally)\n")
    from sklearn.metrics import roc_auc_score
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1, max_iter=500))])
    pipe.fit(X, y)
    scores = pipe.predict_proba(X)[:, 1]
    df["pred_score"] = scores

    for s in range(4):
        mask = df["seat"] == s
        try:
            auc = roc_auc_score(y[mask], scores[mask])
        except Exception:
            auc = float("nan")
        print(f"  Seat {s}: AUC={auc:.4f}  (n={mask.sum():,})")

    # 4b. Calibration: does predicted win prob match actual win rate?
    print("\n[4b] Calibration — predicted probability vs actual win rate\n")
    df["score_decile"] = pd.qcut(df["pred_score"], q=10, duplicates="drop",
                                  labels=False)
    cal = df.groupby("score_decile").agg(
        pred_mean=("pred_score", "mean"),
        actual_mean=("won", "mean"),
        n=("won", "count"),
    )
    print(f"  {'Decile':>7}  {'Pred%':>6}  {'Actual%':>8}  {'Error':>7}  {'N':>7}")
    for dec, row in cal.iterrows():
        err = row["actual_mean"] - row["pred_mean"]
        print(f"  {dec:>7}  {row['pred_mean']*100:>5.1f}%  "
              f"{row['actual_mean']*100:>7.1f}%  {err:>+7.3f}  {row['n']:>7,}")

    # 4c. Top predicted openings — do they actually win more?
    print("\n[4c] Win rate by model score quintile\n")
    df["score_q"] = pd.qcut(df["pred_score"], q=5,
                             labels=["Q1_worst","Q2","Q3","Q4","Q5_best"])
    q_stats = df.groupby("score_q", observed=True)["won"].agg(["count", "mean"])
    q_stats.columns = ["n", "win_rate"]
    q_stats["vs_baseline"] = q_stats["win_rate"] - y.mean()
    print(q_stats.round(3).to_string())


# ---------------------------------------------------------------------------
# 5. Bonus: Pip ceiling effect
# ---------------------------------------------------------------------------

def analyze_pip_ceiling(df: pd.DataFrame):
    section("5. PIP CEILING — Do very high pip counts show diminishing returns?")

    print("\n[5a] Win rate vs combined pip count (smoothed)\n")
    sub = df[df["combined_pip_count"].between(8, 35)].copy()
    sub["pip_bin"] = sub["combined_pip_count"].round(0).astype(int)
    grp = sub.groupby("pip_bin")["won"].agg(["mean", "count"])
    grp = grp[grp["count"] >= 200]

    print(f"  {'Pips':>5}  {'Win%':>6}  {'Count':>7}")
    prev_wr = None
    for pips, row in grp.iterrows():
        delta = f" ({row['mean']-prev_wr:+.3f})" if prev_wr is not None else ""
        print(f"  {pips:>5}  {row['mean']*100:>5.1f}%{delta}  {row['count']:>7,}")
        prev_wr = row["mean"]

    # Find peak
    if not grp.empty:
        peak_pips = grp["mean"].idxmax()
        peak_wr = grp["mean"].max()
        print(f"\n  Peak: {peak_pips} pips -> {peak_wr*100:.1f}% win rate")
        print(f"  Baseline: {df['won'].mean()*100:.1f}%  "
              f"Uplift: {(peak_wr - df['won'].mean())*100:+.1f}pp")

    print("\n[5b] Pip ceiling by archetype — does 'more pips' saturate faster for some?\n")
    for arch in ["balanced", "ore_wheat", "road_race", "port_engine", "high_pip"]:
        sub2 = df[df.apply(identify_archetype, axis=1) == arch].copy()
        if len(sub2) < 200:
            continue
        sub2["pip_bin"] = sub2["combined_pip_count"].round(0).astype(int)
        grp2 = sub2.groupby("pip_bin")["won"].agg(["mean", "count"])
        grp2 = grp2[grp2["count"] >= 50]
        if grp2.empty:
            continue
        peak2 = grp2["mean"].idxmax()
        baseline2 = sub2["won"].mean()
        peak_wr2 = grp2["mean"].max()
        print(f"  {arch:<15}: peak at {peak2} pips  ({peak_wr2*100:.1f}% vs {baseline2*100:.1f}% base)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df):,} records  win_rate={df['won'].mean():.3f}")
    print(f"Has game_id metadata: {'game_id' in df.columns and df['game_id'].dtype == object}")
    if "final_vp" in df.columns:
        print(f"Has VP metadata:       {df['final_vp'].notna().mean():.0%}")

    analyze_seat_dynamics(df)
    analyze_human_vs_optimal(df)
    analyze_board_variance(df)
    analyze_model_performance(df)
    analyze_pip_ceiling(df)

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)
