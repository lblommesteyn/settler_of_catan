"""
Opening Regret Visualizations
Generates figures in research/opening_regret/figures/
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

BASELINE = 0.255
SEAT_COLORS = ["#2ECC71", "#3498DB", "#E67E22", "#E74C3C"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130,
                     "axes.spines.right": False, "axes.spines.top": False})

pct = mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
pp  = mticker.FuncFormatter(lambda x, _: f"{x*100:+.2f}pp")


def save(name):
    p = FIGS / f"{name}.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  {p.name}")


def load():
    df = pd.read_csv(Path(__file__).parent / "regret_enriched.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Regret distribution + rank histogram
# ─────────────────────────────────────────────────────────────────────────────

def fig1_regret_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("How Much Value Do Catan Players Leave on the Table?\n"
                 "V2 Regret Distribution  |  9,927 Colonist.io games  |  38,670 player-openings",
                 fontsize=12, y=1.02)

    # 1a: v2 regret histogram
    ax = axes[0]
    ax.hist(df["v2_regret"].clip(0, 0.12), bins=50, color="#3498DB", alpha=0.8, edgecolor="white")
    ax.axvline(df["v2_regret"].mean(), color="#E74C3C", lw=2, label=f"Mean {df['v2_regret'].mean():.3f}")
    ax.axvline(df["v2_regret"].median(), color="#F39C12", lw=2, ls="--",
               label=f"Median {df['v2_regret'].median():.3f}")
    ax.set_xlabel("V2 regret (model score lost vs best alternative)")
    ax.set_ylabel("Count")
    ax.set_title("V2 Regret Distribution\n(how much better could v2 have been?)", fontweight="bold")
    ax.legend(fontsize=9)

    # 1b: v2 rank distribution (how often near top?)
    ax = axes[1]
    rank_counts = df["v2_rank"].value_counts().sort_index()
    rank_counts = rank_counts[rank_counts.index <= 20]
    colors_ = ["#E74C3C" if i == 1 else "#3498DB" if i <= 3 else "#BDC3C7"
               for i in rank_counts.index]
    ax.bar(rank_counts.index, rank_counts.values / len(df),
           color=colors_, width=0.7, alpha=0.85)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("V2 rank (1 = chose best available)")
    ax.set_ylabel("Fraction of picks")
    ax.set_title(f"V2 Pick Rank Distribution\n(only {df['top1_v2'].mean():.1%} choose #1 optimal)",
                 fontweight="bold")
    ax.text(1, df["top1_v2"].mean() + 0.001, f"{df['top1_v2'].mean():.1%}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#E74C3C")

    # 1c: win rate by regret quartile
    ax = axes[2]
    df["regret_q"] = pd.qcut(df["v2_regret"], q=4, labels=["Q1\n(least regret)","Q2","Q3","Q4\n(most regret)"])
    rq = df.groupby("regret_q", observed=True)["won"].mean()
    colors_ = ["#27AE60","#3498DB","#E67E22","#E74C3C"]
    bars = ax.bar(rq.index, rq.values, color=colors_, width=0.55, alpha=0.85)
    ax.axhline(BASELINE, color="grey", ls="--", lw=1.2, label=f"Baseline {BASELINE:.1%}")
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.235, 0.285)
    ax.set_title("Win Rate by V2 Regret Quartile\n(low regret = more wins)",
                 fontweight="bold")
    ax.set_ylabel("Win rate")
    for bar, v in zip(bars, rq.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save("fig1_regret_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Bias decomposition: pips, expansion, ports
# ─────────────────────────────────────────────────────────────────────────────

def fig2_bias_decomposition(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Systematic Biases in Human Opening Choices\n"
                 "(actual choice vs model-optimal alternative, relative to best v2)",
                 fontsize=12, y=1.02)

    # 2a: pip_diff distribution
    ax = axes[0]
    vals = df["pip_diff"].clip(-15, 15)
    ax.hist(vals[vals < 0], bins=40, color="#E74C3C", alpha=0.7, label=f"Chose fewer pips ({(df['pip_diff']<0).mean():.1%})")
    ax.hist(vals[vals > 0], bins=40, color="#27AE60", alpha=0.7, label=f"Chose more pips ({(df['pip_diff']>0).mean():.1%})")
    ax.hist(vals[vals == 0], bins=5, color="#BDC3C7", alpha=0.7, label=f"Same pips ({(df['pip_diff']==0).mean():.1%})")
    ax.axvline(df["pip_diff"].mean(), color="black", lw=2,
               label=f"Mean {df['pip_diff'].mean():+.2f}")
    ax.set_xlabel("pip_diff = actual pips - best-v2 pips")
    ax.set_ylabel("Count")
    ax.set_title(f"Pip Bias\nMean {df['pip_diff'].mean():+.3f} (p<0.001)", fontweight="bold")
    ax.legend(fontsize=8.5)

    # 2b: expansion_diff distribution
    ax = axes[1]
    vals = df["exp_diff"].clip(-80, 40)
    ax.hist(vals[vals < 0], bins=50, color="#E74C3C", alpha=0.7,
            label=f"Less expansion ({(df['exp_diff']<0).mean():.1%})")
    ax.hist(vals[vals > 0], bins=25, color="#27AE60", alpha=0.7,
            label=f"More expansion ({(df['exp_diff']>0).mean():.1%})")
    ax.axvline(df["exp_diff"].mean(), color="black", lw=2,
               label=f"Mean {df['exp_diff'].mean():+.1f}")
    ax.set_xlabel("exp_diff = actual expansion - best-v2 expansion")
    ax.set_title(f"Expansion Bias\nMean {df['exp_diff'].mean():+.2f} (p<0.001) ← KEY FINDING",
                 fontweight="bold", color="#C0392B")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8.5)

    # 2c: port_diff distribution
    ax = axes[2]
    port_vc = df["port_diff"].value_counts().sort_index()
    port_vc = port_vc[abs(port_vc.index) <= 2]
    colors_ = ["#E74C3C" if x > 0 else "#27AE60" if x < 0 else "#BDC3C7"
               for x in port_vc.index]
    ax.bar(port_vc.index.astype(str), port_vc.values / len(df),
           color=colors_, width=0.5, alpha=0.85)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("port_diff = actual ports - best-v2 ports")
    ax.set_ylabel("Fraction")
    ax.set_title(f"Port Bias\nMean {df['port_diff'].mean():+.4f} (p<0.001)\nPlayers systematically chase ports",
                 fontweight="bold")
    # annotation
    pos_port = (df["port_diff"] > 0).mean()
    ax.text(2, pos_port * 0.5 / len(port_vc) * len(df) / len(df),
            f"{pos_port:.1%}\nchased\nports", ha="center", fontsize=8.5, color="#C0392B",
            fontweight="bold")

    plt.tight_layout()
    save("fig2_bias_decomposition")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Seat difficulty
# ─────────────────────────────────────────────────────────────────────────────

def fig3_seat_difficulty(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Seat Difficulty  —  Which Draft Position Is Hardest to Play Well?",
                 fontsize=12, y=1.02)

    seat_labels = ["Seat 0\n(1st pick)", "Seat 1\n(2nd pick)",
                   "Seat 2\n(3rd pick)", "Seat 3\n(4th pick)"]

    # 3a: mean v2 regret by seat
    ax = axes[0]
    seat_regret = [df[df["seat"]==s]["v2_regret"].mean() for s in range(4)]
    bars = ax.bar(seat_labels, seat_regret, color=SEAT_COLORS, width=0.55, alpha=0.85)
    ax.set_title("Mean V2 Regret by Seat\n(similar across all seats)", fontweight="bold")
    ax.set_ylabel("Mean v2 regret")
    for bar, v in zip(bars, seat_regret):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3b: win penalty for high vs low regret per seat
    ax = axes[1]
    df["regret_q"] = pd.qcut(df["v2_regret"], q=4, labels=["Q1","Q2","Q3","Q4"])
    penalties, low_wrs, high_wrs = [], [], []
    for s in range(4):
        sub = df[df["seat"]==s]
        lo = sub[sub["regret_q"]=="Q1"]["won"].mean()
        hi = sub[sub["regret_q"]=="Q4"]["won"].mean()
        low_wrs.append(lo)
        high_wrs.append(hi)
        penalties.append(hi - lo)

    x = np.arange(4)
    w = 0.35
    ax.bar(x-w/2, low_wrs,  width=w, color="#27AE60", alpha=0.85, label="Low regret (Q1)")
    ax.bar(x+w/2, high_wrs, width=w, color="#E74C3C", alpha=0.85, label="High regret (Q4)")
    ax.axhline(BASELINE, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(seat_labels)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.22, 0.30)
    ax.set_title("Win Rate: Low vs High Regret by Seat\n(Seat 0 & 3 penalised most)",
                 fontweight="bold")
    ax.set_ylabel("Win rate")
    ax.legend(fontsize=9)

    # 3c: expansion bias by seat
    ax = axes[2]
    seat_exp_bias = [df[df["seat"]==s]["exp_diff"].mean() for s in range(4)]
    colors_ = ["#E74C3C" if v < 0 else "#27AE60" for v in seat_exp_bias]
    bars = ax.bar(seat_labels, seat_exp_bias, color=colors_, width=0.55, alpha=0.85)
    ax.axhline(0, color="grey", lw=1)
    ax.set_title("Expansion Bias by Seat\n(all seats undervalue expansion)",
                 fontweight="bold")
    ax.set_ylabel("Mean expansion bias (actual - optimal)")
    for bar, v in zip(bars, seat_exp_bias):
        ax.text(bar.get_x()+bar.get_width()/2, v-0.5,
                f"{v:.1f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    save("fig3_seat_difficulty")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Trap openings
# ─────────────────────────────────────────────────────────────────────────────

def fig4_trap_openings(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Trap Openings  —  Patterns Players Overvalue",
                 fontsize=12, y=1.02)

    baseline = df["won"].mean()

    # 4a: win rate by model score quartile
    ax = axes[0]
    df["score_q"] = pd.qcut(df["actual_score"], q=4,
                             labels=["Q1\nweak","Q2","Q3","Q4\nstrong"])
    sq = df.groupby("score_q", observed=True)["won"].mean()
    colors_ = sns.color_palette("RdYlGn", 4)
    bars = ax.bar(sq.index, sq.values, color=colors_, width=0.55, alpha=0.9)
    ax.axhline(baseline, color="grey", ls="--", lw=1.2)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.23, 0.29)
    ax.set_title("Win Rate by Opening Quality Score\n(Q4 openings: +2.0pp over baseline)",
                 fontweight="bold")
    ax.set_ylabel("Win rate")
    for bar, v in zip(bars, sq.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4b: specific traps vs non-traps
    ax = axes[1]
    patterns = {
        "High pips\n+ low expansion":   (df["actual_pips"] >= df["actual_pips"].quantile(0.8)) &
                                          (df["actual_expansion"] <= df["actual_expansion"].quantile(0.3)),
        "Port chaser\n(>optimal ports)": df["port_diff"] > 0,
        "Chosen\noptimal v2":            df["v2_rank"] == 1,
        "High expansion\n+ fewer pips":  (df["exp_diff"] > 0) & (df["pip_diff"] < 0),
    }
    wrs = [(label, df[mask]["won"].mean() - baseline, len(df[mask]))
           for label, mask in patterns.items()]
    labels_, deltas, ns = zip(*wrs)
    colors_ = ["#E74C3C" if d < 0 else "#27AE60" for d in deltas]
    bars = ax.barh(labels_, deltas, color=colors_, height=0.45, alpha=0.85)
    ax.axvline(0, color="grey", lw=1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:+.1f}pp"))
    ax.set_title("Win Rate Delta vs Baseline\nby Opening Pattern", fontweight="bold")
    ax.set_xlabel("Win rate delta vs baseline")
    for bar, n in zip(bars, ns):
        x = bar.get_width()
        ax.text(x + 0.0002 if x >= 0 else x - 0.0002, bar.get_y()+bar.get_height()/2,
                f"n={n:,}", va="center", ha="left" if x >= 0 else "right", fontsize=8)

    # 4c: pip count vs win rate (ceiling effect + trap zone)
    ax = axes[2]
    sub = df[df["actual_pips"].between(6, 28)]
    grp = sub.groupby(sub["actual_pips"].round(0).astype(int))["won"].agg(["mean","count"])
    grp = grp[grp["count"] >= 200]
    ax.plot(grp.index, grp["mean"], "o-", color="#2C3E50", lw=2, ms=5)
    ax.fill_between(grp.index, baseline, grp["mean"],
                    where=(grp["mean"] >= baseline), alpha=0.2, color="#27AE60", label="Above baseline")
    ax.fill_between(grp.index, baseline, grp["mean"],
                    where=(grp["mean"] < baseline), alpha=0.2, color="#E74C3C", label="Below baseline")
    ax.axhline(baseline, color="grey", ls="--", lw=1.2, label=f"Baseline {baseline:.1%}")
    # Mark the "trap zone": high pips but low expansion players tend to land here
    peak_pip = grp["mean"].idxmax()
    ax.axvline(peak_pip, color="#F39C12", ls=":", lw=1.5, label=f"Peak at {peak_pip} pips")
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Combined pip count")
    ax.set_ylabel("Win rate")
    ax.set_title("Pip Count vs Win Rate\n(peak ~24 pips; high pips alone don't guarantee wins)",
                 fontweight="bold")
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    save("fig4_trap_openings")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Board complexity and regret
# ─────────────────────────────────────────────────────────────────────────────

def fig5_board_complexity(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Does Board Complexity Affect Decision Quality?",
                 fontsize=12, y=1.02)

    # 5a: regret by within-game pip spread
    ax = axes[0]
    df["game_pip_spread"] = df["game_id"].map(
        df.groupby("game_id")["actual_pips"].agg(lambda x: x.max()-x.min()))
    df["spread_q"] = pd.qcut(df["game_pip_spread"], q=4,
                              labels=["Equal\n(≤4)", "Mild\n(5-7)", "Spread\n(8-11)", "Chaotic\n(12+)"])
    sp = df.groupby("spread_q", observed=True)["v2_regret"].mean()
    bars = ax.bar(sp.index, sp.values, color=["#5D6D7E","#3498DB","#E67E22","#E74C3C"],
                  width=0.55, alpha=0.85)
    ax.set_title("V2 Regret by Board Pip Spread\n(slight increase on chaotic boards)",
                 fontweight="bold")
    ax.set_ylabel("Mean v2 regret")
    for bar, v in zip(bars, sp.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # 5b: pip bias by board spread
    ax = axes[1]
    sp_pb = df.groupby("spread_q", observed=True)["pip_diff"].mean()
    colors_ = ["#E74C3C" if v < 0 else "#27AE60" for v in sp_pb.values]
    bars = ax.bar(sp_pb.index, sp_pb.values, color=colors_, width=0.55, alpha=0.85)
    ax.axhline(0, color="grey", lw=1)
    ax.set_title("Pip Bias by Board Spread\n(equal boards: stronger pip-chasing bias)",
                 fontweight="bold")
    ax.set_ylabel("Mean pip bias (actual - optimal)")
    for bar, v in zip(bars, sp_pb.values):
        ax.text(bar.get_x()+bar.get_width()/2, v-0.05,
                f"{v:+.3f}", ha="center", va="top", fontsize=9, fontweight="bold")

    # 5c: top-1 v2 rate by seat constraint level
    ax = axes[2]
    bins = [41, 44, 47, 51, 55]
    df["v1_constraint_q"] = pd.cut(df["v1_n_alts"], bins=bins,
                                    labels=["Tight\n(42-44)","Mid-tight\n(45-47)",
                                            "Mid-free\n(48-51)","Free\n(=54)"])
    cq = df.groupby("v1_constraint_q", observed=True)["top1_v2"].mean()
    bars = ax.bar(cq.index, cq.values,
                  color=["#E74C3C","#E67E22","#3498DB","#27AE60"],
                  width=0.55, alpha=0.85)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("Optimal V2 Rate by V1 Constraint Level\n(tighter seat = slightly better v2 choices)",
                 fontweight="bold")
    ax.set_ylabel("% of players who chose optimal v2")
    for bar, v in zip(bars, cq.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.0002,
                f"{v:.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save("fig5_board_complexity")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Winners vs losers
# ─────────────────────────────────────────────────────────────────────────────

def fig6_winners_vs_losers(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Winners vs Losers  —  How Do Opening Choices Differ?",
                 fontsize=12, y=1.02)

    winners = df[df["won"] == 1]
    losers  = df[df["won"] == 0]

    # 6a: key metric comparison (bar chart of deltas)
    ax = axes[0]
    metrics_for_plot = [
        ("v2_regret",        "V2 regret",         -1),  # lower is better for winner
        ("pct_rank_v2",      "V2 pct rank",        -1),  # lower is better
        ("actual_expansion", "Expansion score",    +1),
        ("actual_ports",     "Port count",         -1),  # winners have fewer
        ("exp_diff",         "Expansion bias",     +1),
        ("actual_adj3",      "Interior hexes",     +1),
    ]
    labels_, deltas, desired_colors = [], [], []
    for col, label, direction in metrics_for_plot:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        # standardize delta by std
        std = df[col].std()
        delta_std = (w_mean - l_mean) / std if std > 0 else 0
        labels_.append(label)
        deltas.append(delta_std * direction)   # positive = winner is better
        desired_colors.append("#27AE60" if delta_std * direction > 0 else "#E74C3C")

    bars = ax.barh(labels_, deltas, color=desired_colors, height=0.5, alpha=0.85)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Standardized delta (winner - loser, positive = winners better)")
    ax.set_title("Winners vs Losers\n(standardized effect sizes)", fontweight="bold")

    # 6b: expansion score distribution, winners vs losers
    ax = axes[1]
    ax.hist(losers["actual_expansion"].clip(0, 200), bins=50,
            color="#E74C3C", alpha=0.6, label=f"Losers (μ={losers['actual_expansion'].mean():.1f})",
            density=True)
    ax.hist(winners["actual_expansion"].clip(0, 200), bins=50,
            color="#27AE60", alpha=0.6, label=f"Winners (μ={winners['actual_expansion'].mean():.1f})",
            density=True)
    ax.set_xlabel("Expansion pip score (reachable future sites)")
    ax.set_ylabel("Density")
    ax.set_title("Expansion Score: Winners vs Losers\n(winners prioritise growth lanes)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    t, p = stats.mannwhitneyu(winners["actual_expansion"], losers["actual_expansion"])
    ax.text(0.97, 0.97, f"p = {p:.2e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="grey")

    # 6c: port_diff distribution, winners vs losers
    ax = axes[2]
    port_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    w_port = pd.cut(winners["port_diff"].clip(-2, 2), bins=port_bins).value_counts().sort_index()
    l_port = pd.cut(losers["port_diff"].clip(-2, 2),  bins=port_bins).value_counts().sort_index()
    x = np.arange(len(w_port))
    width = 0.35
    ax.bar(x - width/2, w_port.values / len(winners), width, color="#27AE60",
           alpha=0.8, label="Winners")
    ax.bar(x + width/2, l_port.values / len(losers),  width, color="#E74C3C",
           alpha=0.8, label="Losers")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(iv.left)+1}" for iv in w_port.index], fontsize=9)
    ax.set_xlabel("Port bias (actual - best-v2 ports)\n< 0: chose fewer ports than optimal")
    ax.yaxis.set_major_formatter(pct)
    ax.set_title("Port Bias: Winners vs Losers\n(winners chase ports slightly less)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    winner_port_bias = winners["port_diff"].mean()
    loser_port_bias  = losers["port_diff"].mean()
    ax.text(0.03, 0.97, f"Winners mean: {winner_port_bias:+.4f}\nLosers mean: {loser_port_bias:+.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="grey")

    plt.tight_layout()
    save("fig6_winners_vs_losers")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Summary: the key narrative
# ─────────────────────────────────────────────────────────────────────────────

def fig7_summary(df):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.42,
                           left=0.06, right=0.97, top=0.88, bottom=0.06)
    fig.suptitle("Opening Regret in Catan  ·  9,927 Real Games  ·  Key Findings",
                 fontsize=15, fontweight="bold", y=0.96)

    baseline = df["won"].mean()
    df["regret_q"] = pd.qcut(df["v2_regret"], q=4, labels=["Q1","Q2","Q3","Q4"])

    # A: regret distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["v2_regret"].clip(0, 0.12), bins=40, color="#3498DB", alpha=0.8, edgecolor="white")
    ax.axvline(df["v2_regret"].mean(), color="#E74C3C", lw=2,
               label=f"Mean: {df['v2_regret'].mean():.3f}")
    ax.set_xlabel("V2 regret"); ax.set_ylabel("Count")
    ax.set_title("1. Regret Distribution\nAvg: 2.6% model score lost", fontweight="bold")
    ax.legend(fontsize=8)

    # B: top-K rate
    ax = fig.add_subplot(gs[0, 1])
    ks = [1, 3, 5, 10, 15]
    rates = [(df["v2_rank"] <= k).mean() for k in ks]
    ax.bar([str(k) for k in ks], rates, color="#3498DB", alpha=0.85, width=0.5)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Top-K threshold"); ax.set_ylabel("Fraction of picks")
    ax.set_title("2. Pick Quality\nOnly 1.6% choose optimal v2", fontweight="bold")
    for x, v in zip(range(len(ks)), rates):
        ax.text(x, v+0.002, f"{v:.1%}", ha="center", fontsize=8.5, fontweight="bold")

    # C: expansion bias (the key finding)
    ax = fig.add_subplot(gs[0, 2])
    vals = df["exp_diff"].clip(-60, 30)
    ax.hist(vals[vals < 0], bins=40, color="#E74C3C", alpha=0.7,
            label=f"Underused expansion: {(df['exp_diff']<0).mean():.1%}")
    ax.hist(vals[vals >= 0], bins=20, color="#27AE60", alpha=0.7,
            label=f"Good/over expansion: {(df['exp_diff']>=0).mean():.1%}")
    ax.axvline(df["exp_diff"].mean(), color="black", lw=2)
    ax.set_xlabel("Expansion bias")
    ax.set_title(f"3. The Key Bias\nPlayers undervalue expansion\n(mean {df['exp_diff'].mean():.1f} pts)",
                 fontweight="bold", color="#C0392B")
    ax.legend(fontsize=8)

    # D: port trap
    ax = fig.add_subplot(gs[0, 3])
    no_port_trap = df[df["port_diff"] == 0]["won"].mean()
    port_trap    = df[df["port_diff"] > 0]["won"].mean()
    ax.bar(["No port bias", "Chased more\nports than best"],
           [no_port_trap, port_trap],
           color=["#27AE60","#E74C3C"], width=0.45, alpha=0.85)
    ax.axhline(baseline, color="grey", ls="--", lw=1)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.24, 0.27)
    ax.set_title(f"4. The Port Trap\n37.2% of players chase ports;\nit costs -0.7pp win rate",
                 fontweight="bold")
    for x, v in zip([0, 1], [no_port_trap, port_trap]):
        ax.text(x, v+0.0005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # E: win rate by regret quartile (bottom left, wide)
    ax = fig.add_subplot(gs[1, :2])
    rq = df.groupby("regret_q", observed=True)["won"].mean()
    x_labels = ["Q1: Optimal\n(low regret)", "Q2", "Q3", "Q4: High regret\n(big mistake)"]
    colors_ = ["#27AE60","#3498DB","#E67E22","#E74C3C"]
    bars = ax.bar(x_labels, rq.values, color=colors_, width=0.55, alpha=0.9)
    ax.axhline(baseline, color="grey", ls="--", lw=1.5, label=f"Baseline {baseline:.1%}")
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.235, 0.285)
    ax.set_title("5. Cost of Regret  —  Low vs High Regret Win Rate by Quartile\n"
                 "(near-optimal v2 choice → +2.6pp win rate vs high-regret players)",
                 fontweight="bold")
    ax.set_ylabel("Win rate")
    ax.legend(fontsize=10)
    for bar, v in zip(bars, rq.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # F: winners vs losers expansion (bottom right, wide)
    ax = fig.add_subplot(gs[1, 2:])
    winners = df[df["won"]==1]
    losers  = df[df["won"]==0]
    ax.hist(losers["actual_expansion"].clip(0,200), bins=50,
            color="#E74C3C", alpha=0.55, label=f"Losers  (μ={losers['actual_expansion'].mean():.1f})",
            density=True)
    ax.hist(winners["actual_expansion"].clip(0,200), bins=50,
            color="#27AE60", alpha=0.55, label=f"Winners (μ={winners['actual_expansion'].mean():.1f})",
            density=True)
    ax.set_xlabel("Expansion pip score (value of reachable future settlement sites)")
    ax.set_ylabel("Density")
    ax.set_title("6. Winners Prioritise Expansion\n"
                 "Winners score +0.46 pts more on expansion vs losers (p<0.05)",
                 fontweight="bold")
    ax.legend(fontsize=10)

    save("fig7_summary")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading enriched data...")
    df = load()
    print(f"  {len(df):,} rows\n")

    print("Generating figures:")
    fig1_regret_distribution(df)
    fig2_bias_decomposition(df)
    fig3_seat_difficulty(df)
    fig4_trap_openings(df)
    fig5_board_complexity(df)
    fig6_winners_vs_losers(df)
    fig7_summary(df)

    print(f"\nAll figures saved to {FIGS.resolve()}/")
