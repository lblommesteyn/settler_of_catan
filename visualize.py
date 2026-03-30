"""
Catan Opening Intelligence — Research Visualizations
Generates figures/research_*.png
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from catan.features.opening_features import FEATURE_NAMES

# ── style ────────────────────────────────────────────────────────────────────
CATAN_PALETTE = ["#C0392B", "#2980B9", "#27AE60", "#F39C12"]   # red/blue/green/gold
SEAT_COLORS   = CATAN_PALETTE
ARCH_COLORS   = {
    "balanced":    "#5D6D7E",
    "road_race":   "#27AE60",
    "ore_wheat":   "#F39C12",
    "port_engine": "#2980B9",
    "high_pip":    "#E74C3C",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "font.family": "sans-serif",
})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

BASELINE = 0.256


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(path="data/processed/dataset.npz"):
    d = np.load(path, allow_pickle=True)
    df = pd.DataFrame(d["X"], columns=FEATURE_NAMES)
    df["won"]        = d["y"].astype(int)
    df["seat"]       = d["seats"].astype(int)
    df["final_vp"]   = d["final_vps"].astype(int)
    df["final_rank"] = d["final_ranks"].astype(int)
    df["game_id"]    = d["game_ids"]
    return df

def identify_archetype(row) -> str:
    scores = {k: row[f"{k}_score"] for k in
              ["ore_wheat","road_race","port_engine","balanced","high_pip"]}
    return max(scores, key=scores.get)

def add_baseline(ax, value=BASELINE, label="Baseline 25.6%"):
    ax.axhline(value, color="grey", lw=1.2, ls="--", alpha=0.7, zorder=0)
    ax.text(ax.get_xlim()[1]*0.98, value + 0.001, label,
            ha="right", va="bottom", fontsize=8, color="grey")

def pct_fmt(x, _):
    return f"{x*100:.1f}%"

def save(name):
    p = OUT / f"{name}.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Seat win rates + what each seat picks
# ═══════════════════════════════════════════════════════════════════════════════

def fig_seat_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Seat Dynamics  ·  43,947 Colonist.io games", fontsize=13, y=1.02)

    # 1a: win rate by seat
    ax = axes[0]
    wr = [df[df["seat"]==s]["won"].mean() for s in range(4)]
    bars = ax.bar(["Seat 0\n(1st/8th)","Seat 1\n(2nd/7th)",
                   "Seat 2\n(3rd/6th)","Seat 3\n(4th/5th)"],
                  wr, color=SEAT_COLORS, width=0.55, zorder=3)
    add_baseline(ax)
    ax.set_ylim(0.23, 0.29)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_title("Win Rate by Seat", fontweight="bold")
    ax.set_ylabel("Win rate")
    for bar, v in zip(bars, wr):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0008,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 1b: pip distribution by seat (violin)
    ax = axes[1]
    data_by_seat = [df[df["seat"]==s]["combined_pip_count"].values for s in range(4)]
    parts = ax.violinplot(data_by_seat, positions=range(4), showmedians=True,
                          showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(SEAT_COLORS[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Seat 0","Seat 1","Seat 2","Seat 3"])
    ax.set_title("Pip Count Distribution by Seat", fontweight="bold")
    ax.set_ylabel("Combined pip count")

    # 1c: feature coefficients heatmap
    ax = axes[2]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    key_features = [
        "combined_pip_count", "unique_resource_count",
        "expansion_pip_sum", "v1_num_adjacent_hexes",
        "v2_num_adjacent_hexes", "combined_city_pips",
        "ore_wheat_score", "road_race_score",
    ]
    feat_labels = [
        "Combined pips", "Unique resources",
        "Expansion pips", "V1 interior hex",
        "V2 interior hex", "City pips",
        "Ore/wheat score", "Road-race score",
    ]
    coef_matrix = []
    for s in range(4):
        sub = df[df["seat"]==s]
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=1, max_iter=500))])
        pipe.fit(sub[key_features].values, sub["won"].values)
        coef_matrix.append(pipe.named_steps["lr"].coef_[0])

    coef_df = pd.DataFrame(coef_matrix, columns=feat_labels,
                           index=["Seat 0","Seat 1","Seat 2","Seat 3"])
    vmax = max(abs(coef_df.values.max()), abs(coef_df.values.min()))
    sns.heatmap(coef_df, ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, annot=True, fmt=".3f",
                annot_kws={"size": 7.5}, linewidths=0.4,
                cbar_kws={"shrink": 0.7, "label": "LR coef"})
    ax.set_title("Feature Importance by Seat\n(logistic regression)", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save("fig1_seat_dynamics")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Human choices: pips, archetypes, ports
# ═══════════════════════════════════════════════════════════════════════════════

def fig_human_choices(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Human Opening Choices  ·  What Players Actually Pick", fontsize=13, y=1.02)

    # 2a: pip count histogram + win rate dual-axis
    ax1 = axes[0]
    ax2 = ax1.twinx()

    pip_grp = df.groupby(df["combined_pip_count"].astype(int))
    counts = pip_grp["won"].count()
    wrates = pip_grp["won"].mean()
    mask   = counts >= 300

    ax1.bar(counts[mask].index, counts[mask].values,
            color="#BDC3C7", alpha=0.8, width=0.8, label="# openings")
    ax2.plot(wrates[mask].index, wrates[mask].values,
             color="#E74C3C", lw=2.2, marker="o", ms=4, label="Win rate")
    ax2.axhline(BASELINE, color="grey", ls="--", lw=1, alpha=0.6)

    ax1.set_xlabel("Combined pip count")
    ax1.set_ylabel("# openings", color="#7F8C8D")
    ax2.set_ylabel("Win rate", color="#E74C3C")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax2.set_ylim(0.22, 0.32)
    ax1.set_title("Pip Count: Frequency & Win Rate", fontweight="bold")

    lines1, _ = ax1.get_legend_handles_labels()
    lines2, _ = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, ["# openings","Win rate"], fontsize=8, loc="upper right")

    # 2b: archetype frequency vs win rate
    ax = axes[1]
    df["archetype"] = df.apply(identify_archetype, axis=1)
    arch_stats = df.groupby("archetype")["won"].agg(["count","mean"]).reset_index()
    arch_stats.columns = ["archetype","count","win_rate"]
    arch_stats = arch_stats.sort_values("count", ascending=True)

    colors = [ARCH_COLORS.get(a,"grey") for a in arch_stats["archetype"]]
    bars = ax.barh(arch_stats["archetype"], arch_stats["count"],
                   color=colors, alpha=0.85, height=0.55)
    ax.set_xlabel("# openings")
    ax.set_title("Opening Frequency by Archetype", fontweight="bold")

    ax2b = ax.twiny()
    ax2b.scatter(arch_stats["win_rate"], arch_stats["archetype"],
                 color="black", zorder=5, s=60, marker="D")
    ax2b.axvline(BASELINE, color="grey", ls="--", lw=1, alpha=0.6)
    ax2b.set_xlabel("Win rate", color="#2C3E50")
    ax2b.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax2b.set_xlim(0.24, 0.27)

    # 2c: port access win rate
    ax = axes[2]
    port_data = []
    for p in range(3):
        mask = df["num_ports"] == p
        if mask.sum() < 100:
            continue
        port_data.append({
            "ports": f"{p} port{'s' if p!=1 else ''}",
            "win_rate": df.loc[mask, "won"].mean(),
            "n": mask.sum(),
        })

    synergy_cases = [
        ("Port + synergy",    (df["num_ports"] > 0) & (df["port_synergy_score"] > 0.5)),
        ("Port, no synergy",  (df["num_ports"] > 0) & (df["port_synergy_score"] <= 0.5)),
        ("No port",           df["num_ports"] == 0),
    ]
    all_bars = []
    labels, wrates_port, colors_port = [], [], []
    for lbl, mask in synergy_cases:
        n = mask.sum()
        if n < 100:
            continue
        wr = df.loc[mask, "won"].mean()
        labels.append(lbl)
        wrates_port.append(wr)
        colors_port.append(
            "#2980B9" if "synergy" in lbl else
            "#85C1E9" if "no synergy" in lbl else "#BDC3C7"
        )

    bars = ax.barh(labels, wrates_port, color=colors_port, height=0.45)
    ax.axvline(BASELINE, color="grey", ls="--", lw=1.2, alpha=0.7)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_xlim(0.245, 0.265)
    ax.set_title("Port Access vs Win Rate\n(ports are overvalued)", fontweight="bold")
    ax.set_xlabel("Win rate")
    for bar, v in zip(bars, wrates_port):
        ax.text(v + 0.0002, bar.get_y()+bar.get_height()/2,
                f"{v*100:.2f}%", va="center", fontsize=9)

    plt.tight_layout()
    save("fig2_human_choices")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Board variance: where does opening matter?
# ═══════════════════════════════════════════════════════════════════════════════

def fig_board_variance(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Board Variance  ·  When Does Opening Choice Actually Matter?", fontsize=13, y=1.02)

    game_pips = df.groupby("game_id")["combined_pip_count"].agg(["max","min","mean","std"])
    game_pips["spread"] = game_pips["max"] - game_pips["min"]
    spread_df = game_pips["spread"].reset_index()
    df2 = df.merge(spread_df, on="game_id")
    df2["got_best"] = df2.groupby("game_id")["combined_pip_count"].transform(
        lambda x: x == x.max()
    )

    # 3a: Spread distribution
    ax = axes[0]
    ax.hist(game_pips["spread"].values, bins=30, color="#5D6D7E", alpha=0.8, edgecolor="white")
    ax.axvline(game_pips["spread"].mean(), color="#E74C3C", lw=2,
               label=f"Mean: {game_pips['spread'].mean():.1f}")
    ax.axvline(game_pips["spread"].median(), color="#F39C12", lw=2, ls="--",
               label=f"Median: {game_pips['spread'].median():.1f}")
    ax.set_xlabel("Pip spread (best − worst opening in game)")
    ax.set_ylabel("Number of games")
    ax.set_title("Per-Game Opening Quality Spread\n(43,669 games)", fontweight="bold")
    ax.legend(fontsize=9)

    # 3b: Win rate for top-pip player by spread quartile
    ax = axes[1]
    labels_q = ["Tight\n(≤7 pips)", "Moderate\n(8–10)", "Spread\n(11–14)", "Wide\n(15+)"]
    df2["spread_q"] = pd.qcut(df2["spread"], q=4,
                               labels=["tight","moderate","spread","wide"])

    best_wrs, other_wrs = [], []
    for q_lbl in ["tight","moderate","spread","wide"]:
        sub = df2[df2["spread_q"] == q_lbl]
        best_wrs.append(sub[sub["got_best"]]["won"].mean())
        other_wrs.append(sub[~sub["got_best"]]["won"].mean())

    x = np.arange(4)
    w = 0.35
    bars_best  = ax.bar(x - w/2, best_wrs,  width=w, label="Best opening in game",
                        color="#27AE60", alpha=0.85)
    bars_other = ax.bar(x + w/2, other_wrs, width=w, label="Other players",
                        color="#BDC3C7", alpha=0.85)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels_q)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_ylim(0.23, 0.33)
    ax.set_title("Opening Advantage by Board Spread\n(tight boards = skill matters most)", fontweight="bold")
    ax.set_ylabel("Win rate")
    ax.legend(fontsize=8.5)

    for bar, v in zip(bars_best, best_wrs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0015,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#27AE60")

    # 3c: Advantage delta by spread quartile (bar chart of gap)
    ax = axes[2]
    deltas = [b - o for b, o in zip(best_wrs, other_wrs)]
    bar_colors = ["#27AE60" if d > 0 else "#E74C3C" for d in deltas]
    bars = ax.bar(labels_q, deltas, color=bar_colors, width=0.5, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:+.1f}pp"))
    ax.set_title("Best-Opening Advantage by Board Spread\n(equal boards: 3.6pp; chaos: near zero)", fontweight="bold")
    ax.set_ylabel("Win rate advantage (best vs others)")

    for bar, v in zip(bars, deltas):
        ypos = v + 0.0008 if v > 0 else v - 0.0015
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{v*100:+.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save("fig3_board_variance")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Model performance: calibration + score quintiles + pip ceiling
# ═══════════════════════════════════════════════════════════════════════════════

def fig_model_performance(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance  ·  What the Logistic Regression Learns", fontsize=13, y=1.02)

    X = df[FEATURE_NAMES].values
    y = df["won"].values
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1, max_iter=500))])
    pipe.fit(X, y)
    df["pred"] = pipe.predict_proba(X)[:,1]

    # 4a: calibration
    ax = axes[0]
    df["decile"] = pd.qcut(df["pred"], q=10, duplicates="drop", labels=False)
    cal = df.groupby("decile").agg(pred_mean=("pred","mean"), actual=("won","mean"))
    ax.scatter(cal["pred_mean"], cal["actual"], s=80, color="#2980B9", zorder=4, label="Decile")
    lo, hi = cal["pred_mean"].min()*0.98, cal["pred_mean"].max()*1.02
    ax.plot([lo, hi],[lo, hi], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Model Calibration\n(predicted vs actual per decile)", fontweight="bold")
    ax.legend(fontsize=8)

    # 4b: win rate by score quintile
    ax = axes[1]
    df["score_q"] = pd.qcut(df["pred"], q=5,
                             labels=["Q1\nworst","Q2","Q3","Q4","Q5\nbest"])
    q_wr = df.groupby("score_q", observed=True)["won"].mean()
    colors_q = sns.color_palette("RdYlGn", 5)
    bars = ax.bar(q_wr.index, q_wr.values, color=colors_q, width=0.55)
    add_baseline(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_ylim(0.22, 0.30)
    ax.set_title("Win Rate by Model Score Quintile\n(3pp spread, Q5 = +1.6pp)", fontweight="bold")
    ax.set_ylabel("Win rate")
    ax.set_xlabel("Score quintile")
    for bar, v in zip(bars, q_wr.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4c: pip ceiling
    ax = axes[2]
    sub = df[df["combined_pip_count"].between(6, 27)].copy()
    sub["pip_bin"] = sub["combined_pip_count"].round(0).astype(int)
    grp = sub.groupby("pip_bin")["won"].agg(["mean","count"])
    grp = grp[grp["count"] >= 250]

    ax.fill_between(grp.index, BASELINE, grp["mean"],
                    where=(grp["mean"] >= BASELINE),
                    alpha=0.25, color="#27AE60", label="Above baseline")
    ax.fill_between(grp.index, BASELINE, grp["mean"],
                    where=(grp["mean"] < BASELINE),
                    alpha=0.25, color="#E74C3C", label="Below baseline")
    ax.plot(grp.index, grp["mean"], color="#2C3E50", lw=2.2, marker="o", ms=4)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1.2, alpha=0.7, label="Baseline 25.6%")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_xlabel("Combined pip count")
    ax.set_ylabel("Win rate")
    ax.set_title("Pip Ceiling Effect\n(peak at ~24 pips, +1.6pp over baseline)", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save("fig4_model_performance")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Summary infographic
# ═══════════════════════════════════════════════════════════════════════════════

def fig_summary(df):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#F8F9FA")
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.4,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)

    fig.suptitle("Catan Opening Intelligence  ·  43,947 games  ·  Key Findings",
                 fontsize=15, fontweight="bold", y=0.96)

    # A: seat win rates
    ax = fig.add_subplot(gs[0, 0])
    seats_wr = [df[df["seat"]==s]["won"].mean() for s in range(4)]
    colors = ["#27AE60","#27AE60","#E74C3C","#E74C3C"]
    bars = ax.bar(["S0","S1","S2","S3"], seats_wr, color=colors, width=0.6)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1)
    ax.set_ylim(0.235, 0.275); ax.set_title("Seat Advantage", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, seats_wr):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                f"{v*100:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Seat (S0=first pick)")

    # B: archetype win rate delta
    ax = fig.add_subplot(gs[0, 1])
    df["archetype"] = df.apply(identify_archetype, axis=1)
    arch = df.groupby("archetype")["won"].mean().sort_values(ascending=True)
    deltas = arch - BASELINE
    colors_a = ["#27AE60" if v >= 0 else "#E74C3C" for v in deltas.values]
    ax.barh(deltas.index, deltas.values, color=colors_a, height=0.5)
    ax.axvline(0, color="grey", lw=1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:+.1f}pp"))
    ax.set_title("Archetype vs Baseline", fontweight="bold")
    ax.set_xlabel("Win rate delta")

    # C: port myth
    ax = fig.add_subplot(gs[0, 2])
    port_labels  = ["No port", "Has port"]
    port_wrs     = [df[df["num_ports"]==0]["won"].mean(),
                    df[df["num_ports"]>0]["won"].mean()]
    bars = ax.bar(port_labels, port_wrs,
                  color=["#BDC3C7","#2980B9"], width=0.5)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1)
    ax.set_ylim(0.24, 0.27); ax.set_title("Port Overvaluation", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, port_wrs):
        delta_lbl = f"({(v-BASELINE)*100:+.2f}pp)"
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0005,
                f"{v*100:.2f}%\n{delta_lbl}", ha="center", fontsize=8.5, fontweight="bold")

    # D: board spread advantage
    ax = fig.add_subplot(gs[0, 3])
    game_pips = df.groupby("game_id")["combined_pip_count"].agg(["max","min"])
    game_pips["spread"] = game_pips["max"] - game_pips["min"]
    df2 = df.merge(game_pips["spread"].reset_index(), on="game_id")
    df2["got_best"] = df2.groupby("game_id")["combined_pip_count"].transform(
        lambda x: x == x.max()
    )
    df2["spread_q"] = pd.qcut(df2["spread"], q=4,
                               labels=["Equal","Mild","Spread","Chaotic"])
    best_wrs_sum  = []
    other_wrs_sum = []
    for q in ["Equal","Mild","Spread","Chaotic"]:
        sub = df2[df2["spread_q"]==q]
        best_wrs_sum.append(sub[sub["got_best"]]["won"].mean())
        other_wrs_sum.append(sub[~sub["got_best"]]["won"].mean())

    x = np.arange(4)
    ax.bar(x-0.2, best_wrs_sum,  0.35, color="#27AE60", alpha=0.85, label="Best opening")
    ax.bar(x+0.2, other_wrs_sum, 0.35, color="#BDC3C7", alpha=0.85, label="Others")
    ax.axhline(BASELINE, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["Equal","Mild","Spread","Chaotic"], fontsize=8)
    ax.set_ylim(0.23, 0.32)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_title("Skill > Luck on Equal Boards", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    # E: pip ceiling (bottom left, wide)
    ax = fig.add_subplot(gs[1, :2])
    sub = df[df["combined_pip_count"].between(6, 27)].copy()
    sub["pip_bin"] = sub["combined_pip_count"].round(0).astype(int)
    grp = sub.groupby("pip_bin")["won"].agg(["mean","count"])
    grp = grp[grp["count"] >= 250]
    ax.plot(grp.index, grp["mean"], "o-", color="#2C3E50", lw=2, ms=5)
    ax.fill_between(grp.index, BASELINE, grp["mean"],
                    where=(grp["mean"]>=BASELINE), alpha=0.2, color="#27AE60")
    ax.fill_between(grp.index, BASELINE, grp["mean"],
                    where=(grp["mean"]<BASELINE), alpha=0.2, color="#E74C3C")
    ax.axhline(BASELINE, color="grey", ls="--", lw=1.2, label="Baseline 25.6%")
    ax.set_xlabel("Combined pip count"); ax.set_ylabel("Win rate")
    ax.set_title("Pip Count vs Win Rate  (max value ~+1.6pp at 24 pips)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.legend(fontsize=9)

    # F: model Q quintile + seat AUC (bottom right, wide)
    ax = fig.add_subplot(gs[1, 2:])

    # Model score quintile
    X = df[FEATURE_NAMES].values
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1, max_iter=500))])
    pipe.fit(X, df["won"].values)
    df["pred"] = pipe.predict_proba(X)[:,1]
    df["score_q"] = pd.qcut(df["pred"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
    q_wr = df.groupby("score_q", observed=True)["won"].mean()
    colors_q = sns.color_palette("RdYlGn", 5)
    bars = ax.bar(q_wr.index, q_wr.values, color=colors_q, width=0.55, alpha=0.9)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1.2, label="Baseline")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.set_ylim(0.22, 0.30)
    ax.set_title("Model Score Quintile vs Win Rate\n(3pp spread from worst to best predicted opening)",
                 fontweight="bold")
    ax.set_ylabel("Win rate"); ax.set_xlabel("Model score quintile (Q1=worst, Q5=best)")
    ax.legend(fontsize=9)
    for bar, v in zip(bars, q_wr.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    save("fig5_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} records loaded\n")

    print("Generating figures...")
    fig_seat_overview(df)
    fig_human_choices(df)
    fig_board_variance(df)
    fig_model_performance(df)
    fig_summary(df)

    print(f"\nAll figures saved to {OUT.resolve()}/")
