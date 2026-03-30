"""
Longest Road win rate graphic.
"""
import tarfile, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── collect data ─────────────────────────────────────────────────────────────
REPO = Path(__file__).parents[2]

rows = []
game_lr_counts = []

with tarfile.open(REPO / "games.tar.gz", "r:gz") as tf:
    for member in tf:
        if not member.name.endswith(".json") or not member.isfile():
            continue
        try:
            f = tf.extractfile(member)
            raw = json.loads(f.read())
            data = raw.get("data", raw)
            play_order = data.get("playOrder", [])[:4]
            if len(play_order) != 4:
                continue
            events = data.get("eventHistory", {}).get("events", [])
            end    = data.get("eventHistory", {}).get("endGameState", {})

            lr_final = None
            lr_ever  = set()
            for ev in events:
                sc  = ev.get("stateChange", {})
                lrs = sc.get("mechanicLongestRoadState", {})
                if lrs:
                    for cs, info in lrs.items():
                        if info.get("hasLongestRoad") == True:
                            c = int(cs)
                            lr_final = c
                            lr_ever.add(c)

            winner = None
            for cs, pd in end.get("players", {}).items():
                if isinstance(pd, dict) and pd.get("winningPlayer"):
                    winner = int(cs)
            if winner is None:
                continue

            game_lr_counts.append(len(lr_ever))
            for color in play_order:
                rows.append(dict(
                    ever_lr   = color in lr_ever,
                    final_lr  = color == lr_final,
                    contested = len(lr_ever) > 1,
                    won       = color == winner,
                ))
        except Exception:
            continue

# ── compute stats ─────────────────────────────────────────────────────────────
n_games = len(game_lr_counts)

def wr(subset):
    if not subset: return 0, 0
    wins = sum(r["won"] for r in subset)
    return wins / len(subset), len(subset)

never_held   = [r for r in rows if not r["ever_lr"]]
ever_held    = [r for r in rows if r["ever_lr"]]
final_holder = [r for r in rows if r["final_lr"]]
uncontested  = [r for r in rows if r["final_lr"] and not r["contested"]]
contested    = [r for r in rows if r["final_lr"] and r["contested"]]

stats = [
    ("Never held\nLongest Road",     *wr(never_held),   "#aab4c8"),
    ("Ever held\nLongest Road",      *wr(ever_held),    "#5b8db8"),
    ("Final holder\n(uncontested)",  *wr(uncontested),  "#f5a623"),
    ("Final holder\n(contested)",    *wr(contested),    "#e05a2b"),
    ("Final holder\n(all games)",    *wr(final_holder), "#c0392b"),
]

# ── build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor("#f7f4f0")

# Left: bar chart
ax1 = fig.add_axes([0.05, 0.13, 0.56, 0.72])
ax1.set_facecolor("#f7f4f0")

labels  = [s[0] for s in stats]
rates   = [s[1] for s in stats]
ns      = [s[2] for s in stats]
colours = [s[3] for s in stats]

x = np.arange(len(labels))
bars = ax1.bar(x, [r * 100 for r in rates], color=colours,
               width=0.58, zorder=3,
               edgecolor="white", linewidth=1.2)

# Null baseline
ax1.axhline(25, color="#555", linestyle="--", linewidth=1.4,
            zorder=4, label="Null baseline (25%)")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel("Win rate (%)", fontsize=12)
ax1.set_ylim(0, 80)
ax1.set_title("Does Longest Road predict winning?",
              fontsize=14, fontweight="bold", pad=10)
ax1.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[["top","right","left","bottom"]].set_visible(False)
ax1.tick_params(axis="both", length=0)

# Annotate bars
for bar, rate, n in zip(bars, rates, ns):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1.2,
             f"{rate:.1%}", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#222")
    ax1.text(bar.get_x() + bar.get_width()/2,
             2.5,
             f"n={n//1000:.0f}k", ha="center", va="bottom",
             fontsize=9, color="white", alpha=0.85)

ax1.legend(fontsize=10, frameon=False)

# Right: donut — games by LR outcome
ax2 = fig.add_axes([0.65, 0.10, 0.33, 0.78])
ax2.set_facecolor("#f7f4f0")

no_lr   = sum(1 for c in game_lr_counts if c == 0)
one_lr  = sum(1 for c in game_lr_counts if c == 1)
many_lr = sum(1 for c in game_lr_counts if c > 1)

donut_vals   = [no_lr, one_lr, many_lr]
donut_labels = [
    f"No LR built\n{no_lr/n_games:.1%}",
    f"1 holder\n{one_lr/n_games:.1%}",
    f"LR changed\nhands\n{many_lr/n_games:.1%}",
]
donut_colours = ["#ccc", "#f5a623", "#e05a2b"]

wedges, _ = ax2.pie(
    donut_vals,
    colors=donut_colours,
    startangle=90,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
)

# Centre text
ax2.text(0, 0.07, f"{n_games:,}", ha="center", va="center",
         fontsize=16, fontweight="bold", color="#222")
ax2.text(0, -0.18, "games", ha="center", va="center",
         fontsize=10, color="#555")

# Legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(donut_colours, donut_labels)]
ax2.legend(handles=patches, loc="lower center",
           bbox_to_anchor=(0.5, -0.08),
           fontsize=9.5, frameon=False, labelspacing=0.6)
ax2.set_title("Game breakdown\nby LR outcome",
              fontsize=12, fontweight="bold", pad=6)

# Footer
fig.text(0.5, 0.02,
         f"Based on {n_games:,} colonist.io games  •  Longest Road = 2 VP card awarded at 5+ road length",
         ha="center", fontsize=9, color="#777")

out = Path(__file__).parent / "longest_road_winrate.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved -> {out}")
