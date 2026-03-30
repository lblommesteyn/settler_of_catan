"""
VP path decomposition: how do winners actually reach 10 VP?

VP formula (confirmed on 94.4% of winners):
  settlements * 1  +  cities * 2  +  vp_cards * 1
  + longest_road * 2  +  largest_army * 2  = 10

Sources:
  cat 0 → settlements  (1 VP each, cities replace them)
  cat 1 → cities       (2 VP each)
  cat 2 → VP dev cards (1 VP each)
  cat 3 → Longest Road (1 = held, 2 VP)
  cat 4 → Largest Army (1 = held, 2 VP)
"""
import tarfile, json, sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO    = Path(__file__).parents[2]
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── collect ───────────────────────────────────────────────────────────────────

records = []   # one row per player in a complete 4-player game

with tarfile.open(REPO / "games.tar.gz", "r:gz") as tf:
    for member in tf:
        if not member.name.endswith(".json") or not member.isfile():
            continue
        try:
            f   = tf.extractfile(member)
            raw = json.loads(f.read())
            data = raw.get("data", raw)
            play_order = data.get("playOrder", [])[:4]
            if len(play_order) != 4:
                continue
            end = data.get("eventHistory", {}).get("endGameState", {})
            total_turns = int(end.get("totalTurnCount", 0))

            for color_str, pdata in end.get("players", {}).items():
                if not isinstance(pdata, dict):
                    continue
                vp_raw = pdata.get("victoryPoints", {})
                if not isinstance(vp_raw, dict) or not vp_raw:
                    continue
                cats = {int(k): int(v) for k, v in vp_raw.items()
                        if v is not None}
                settlements  = cats.get(0, 0)
                cities       = cats.get(1, 0)
                vp_cards     = cats.get(2, 0)
                longest_road = cats.get(3, 0)
                largest_army = cats.get(4, 0)

                total = (settlements * 1 + cities * 2 + vp_cards * 1
                         + longest_road * 2 + largest_army * 2)

                records.append(dict(
                    won          = bool(pdata.get("winningPlayer")),
                    rank         = pdata.get("rank", 0),
                    total_vp     = total,
                    settlements  = settlements,
                    cities       = cities,
                    vp_cards     = vp_cards,
                    longest_road = longest_road,
                    largest_army = largest_army,
                    # VP contributions
                    vp_settle    = settlements * 1,
                    vp_city      = cities * 2,
                    vp_vpcard    = vp_cards * 1,
                    vp_lr        = longest_road * 2,
                    vp_la        = largest_army * 2,
                    total_turns  = total_turns,
                ))
        except Exception:
            continue

winners = [r for r in records if r["won"] and r["total_vp"] >= 9]
losers  = [r for r in records if not r["won"]]
print(f"Total players: {len(records):,}  Winners: {len(winners):,}  Losers: {len(losers):,}")

# ── source breakdown ───────────────────────────────────────────────────────────

SOURCES = [
    ("vp_settle", "Settlements",   "#5b8db8"),
    ("vp_city",   "Cities",        "#e05a2b"),
    ("vp_vpcard", "VP Dev Cards",  "#6acc65"),
    ("vp_lr",     "Longest Road",  "#f5a623"),
    ("vp_la",     "Largest Army",  "#9b59b6"),
]

def mean(lst): return sum(lst) / len(lst) if lst else 0

print("\n=== Mean VP contribution per source ===")
print(f"{'Source':<16} {'Winners':>10} {'Losers':>10}")
for key, label, _ in SOURCES:
    wm = mean([r[key] for r in winners])
    lm = mean([r[key] for r in losers])
    print(f"  {label:<14} {wm:>10.2f} {lm:>10.2f}")

# ── winning recipes ────────────────────────────────────────────────────────────

def classify(r):
    """Classify a winner's VP path into a named recipe."""
    vc = r["vp_city"]
    vs = r["vp_settle"]
    lr = r["vp_lr"]
    la = r["vp_la"]
    vk = r["vp_vpcard"]
    ach = lr + la   # achievement VPs

    if vc >= 6:
        return "City machine\n(6+ VP from cities)"
    if vc >= 4 and ach >= 2:
        return "Cities + achievement"
    if vs >= 5 and ach >= 4:
        return "Settlements + both cards"
    if vs >= 5 and lr >= 2 and la == 0:
        return "Settlements + Longest Road"
    if vs >= 5 and la >= 2 and lr == 0:
        return "Settlements + Largest Army"
    if vs >= 5 and ach == 0:
        return "Pure settlement spread\n(no achievements)"
    if vc >= 4 and ach == 0:
        return "Pure city builder\n(no achievements)"
    if ach >= 4 and vk >= 2:
        return "Dev card heavy\n(cards + achievements)"
    if ach >= 4:
        return "Achievement double\n(LR + LA)"
    if vk >= 3:
        return "VP card heavy"
    return "Balanced"

recipe_counts = Counter(classify(r) for r in winners)
print("\n=== Winning recipe breakdown ===")
total_w = len(winners)
for recipe, n in recipe_counts.most_common():
    print(f"  {n:>5,} ({n/total_w:.1%})  {recipe!r}")

# ── rate: does holding LR/LA correlate with city vs. settlement paths? ─────────

lr_w   = [r for r in winners if r["vp_lr"] > 0]
la_w   = [r for r in winners if r["vp_la"] > 0]
both_w = [r for r in winners if r["vp_lr"] > 0 and r["vp_la"] > 0]
neither_w = [r for r in winners if r["vp_lr"] == 0 and r["vp_la"] == 0]

print(f"\n=== Achievement paths among winners ===")
print(f"  Held LR only:      {sum(1 for r in winners if r['vp_lr']>0 and r['vp_la']==0):,} ({sum(1 for r in winners if r['vp_lr']>0 and r['vp_la']==0)/total_w:.1%})")
print(f"  Held LA only:      {sum(1 for r in winners if r['vp_la']>0 and r['vp_lr']==0):,} ({sum(1 for r in winners if r['vp_la']>0 and r['vp_lr']==0)/total_w:.1%})")
print(f"  Held both LR+LA:   {len(both_w):,} ({len(both_w)/total_w:.1%})")
print(f"  Neither card:      {len(neither_w):,} ({len(neither_w)/total_w:.1%})")

# Win rate given achievement combo (among all players)
all_lr    = [r for r in records if r["vp_lr"] > 0]
all_la    = [r for r in records if r["vp_la"] > 0]
all_both  = [r for r in records if r["vp_lr"] > 0 and r["vp_la"] > 0]
all_neither = [r for r in records if r["vp_lr"] == 0 and r["vp_la"] == 0]

print(f"\n=== Win rate by achievement combo ===")
for label, subset in [("LR only (end)", all_lr), ("LA only (end)", all_la),
                       ("Both LR+LA (end)", all_both), ("Neither card", all_neither)]:
    wr = mean([r["won"] for r in subset])
    print(f"  {label:<22}: {wr:.1%}  (n={len(subset):,})")

# ── FIGURES ────────────────────────────────────────────────────────────────────

BG = "#f7f4f0"

# ── Fig 1: stacked bar — mean VP contribution by source, winners vs losers ──
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

groups  = ["Losers\n(all)", "Winners"]
subsets = [losers, winners]

bar_width = 0.42
x = np.array([0.0, 1.0])
bottoms = [np.zeros(2)]

for i, (key, label, color) in enumerate(SOURCES):
    vals = np.array([mean([r[key] for r in s]) for s in subsets])
    bottom = sum(b for b in bottoms)
    ax.bar(x, vals, bar_width, bottom=bottom,
           color=color, label=label, edgecolor="white", linewidth=0.8, zorder=3)
    # Label in segment if tall enough
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 0.25:
            ax.text(x[xi], b + v/2, f"{v:.1f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    bottoms.append(vals)

# Total labels on top
for xi, subset in enumerate(subsets):
    total = mean([r["total_vp"] for r in subset])
    ax.text(x[xi], total + 0.15, f"{total:.1f} VP",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222")

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13)
ax.set_ylabel("Average VP contribution", fontsize=12)
ax.set_title("Where do winners get their VP?", fontsize=14, fontweight="bold", pad=10)
ax.yaxis.grid(True, color="white", linewidth=1.5, zorder=0)
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.tick_params(length=0)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.set_xlim(-0.4, 1.4)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_vp_sources.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\nSaved fig1_vp_sources.png")

# ── Fig 2: winning recipe donut ───────────────────────────────────────────────
top_recipes = recipe_counts.most_common(7)
other_n     = total_w - sum(n for _, n in top_recipes)
if other_n > 0:
    top_recipes.append(("Other", other_n))

recipe_labels = [f"{r}\n({n/total_w:.0%})" for r, n in top_recipes]
recipe_vals   = [n for _, n in top_recipes]
palette = ["#5b8db8","#e05a2b","#f5a623","#6acc65","#9b59b6","#e67e22","#1abc9c","#aaa"]

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

wedges, _ = ax.pie(
    recipe_vals,
    colors=palette[:len(recipe_vals)],
    startangle=110,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
)
ax.text(0, 0.06, f"{total_w:,}", ha="center", fontsize=16, fontweight="bold")
ax.text(0, -0.16, "winners", ha="center", fontsize=10, color="#555")

patches = [mpatches.Patch(color=palette[i], label=recipe_labels[i])
           for i in range(len(top_recipes))]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.22), fontsize=9.5, frameon=False,
          ncol=2, labelspacing=0.7)
ax.set_title("Winning recipe breakdown", fontsize=14, fontweight="bold", pad=8)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_recipes.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved fig2_recipes.png")

# ── Fig 3: achievement combo win rates ────────────────────────────────────────
ach_labels = ["Neither\ncard", "Longest\nRoad only", "Largest\nArmy only", "Both\ncards"]
ach_subsets = [all_neither,
               [r for r in records if r["vp_lr"]>0 and r["vp_la"]==0],
               [r for r in records if r["vp_la"]>0 and r["vp_lr"]==0],
               all_both]
ach_wrs  = [mean([r["won"] for r in s]) for s in ach_subsets]
ach_ns   = [len(s) for s in ach_subsets]
ach_cols = ["#aab4c8", "#f5a623", "#9b59b6", "#c0392b"]

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.bar(ach_labels, [w*100 for w in ach_wrs],
              color=ach_cols, edgecolor="white", linewidth=1.2,
              width=0.55, zorder=3)
ax.axhline(25, color="#555", linestyle="--", linewidth=1.4, label="Null (25%)", zorder=4)
ax.set_ylabel("Win rate (%)", fontsize=12)
ax.set_title("Win rate by achievement card combo", fontsize=14, fontweight="bold", pad=8)
ax.set_ylim(0, 90)
ax.yaxis.grid(True, color="white", linewidth=1.5, zorder=0)
ax.spines[["top","right","left","bottom"]].set_visible(False)
ax.tick_params(length=0)
ax.legend(fontsize=10, frameon=False)

for bar, wr, n in zip(bars, ach_wrs, ach_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{wr:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, 2,
            f"n={n//1000:.0f}k", ha="center", va="bottom",
            fontsize=9, color="white", alpha=0.85)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_achievement_winrates.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved fig3_achievement_winrates.png")

# ── Fig 4: combined summary panel ────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.5))
fig.patch.set_facecolor(BG)

# Panel A: VP source stacked bars (re-draw compact)
ax1 = fig.add_axes([0.03, 0.12, 0.27, 0.76])
ax1.set_facecolor(BG)
bottoms2 = np.zeros(2)
for key, label, color in SOURCES:
    vals = np.array([mean([r[key] for r in s]) for s in subsets])
    ax1.bar(x, vals, bar_width, bottom=bottoms2, color=color,
            label=label, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, (v, b) in enumerate(zip(vals, bottoms2)):
        if v >= 0.3:
            ax1.text(x[xi], b + v/2, f"{v:.1f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
    bottoms2 += vals
for xi, subset in enumerate(subsets):
    total = mean([r["total_vp"] for r in subset])
    ax1.text(x[xi], total + 0.1, f"{total:.1f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(groups, fontsize=11)
ax1.set_ylabel("Avg VP contribution", fontsize=11)
ax1.set_title("VP sources", fontsize=12, fontweight="bold")
ax1.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
ax1.spines[["top","right","left","bottom"]].set_visible(False)
ax1.tick_params(length=0); ax1.set_xlim(-0.4, 1.4)
ax1.legend(fontsize=8.5, frameon=False, loc="upper left")

# Panel B: winning recipes donut
ax2 = fig.add_axes([0.33, 0.05, 0.33, 0.90])
ax2.set_facecolor(BG)
wedges2, _ = ax2.pie(recipe_vals, colors=palette[:len(recipe_vals)],
                     startangle=110,
                     wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2))
ax2.text(0, 0.07, f"{total_w:,}", ha="center", fontsize=14, fontweight="bold")
ax2.text(0, -0.13, "winners", ha="center", fontsize=9, color="#555")
patches2 = [mpatches.Patch(color=palette[i], label=recipe_labels[i])
            for i in range(len(top_recipes))]
ax2.legend(handles=patches2, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), fontsize=8.5, frameon=False,
           ncol=2, labelspacing=0.5)
ax2.set_title("Winning recipes", fontsize=12, fontweight="bold")

# Panel C: achievement win rates bar
ax3 = fig.add_axes([0.70, 0.12, 0.28, 0.76])
ax3.set_facecolor(BG)
bars3 = ax3.bar(ach_labels, [w*100 for w in ach_wrs],
                color=ach_cols, edgecolor="white", linewidth=1.0,
                width=0.58, zorder=3)
ax3.axhline(25, color="#555", linestyle="--", linewidth=1.3, zorder=4)
ax3.set_ylabel("Win rate (%)", fontsize=11)
ax3.set_title("Win rate by card combo", fontsize=12, fontweight="bold")
ax3.set_ylim(0, 90)
ax3.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
ax3.spines[["top","right","left","bottom"]].set_visible(False)
ax3.tick_params(length=0, axis="x", labelsize=10)
for bar, wr in zip(bars3, ach_wrs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f"{wr:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.suptitle("VP Path Decomposition: How Do Winners Reach 10 VP?",
             fontsize=14, fontweight="bold", y=1.01)
fig.text(0.5, -0.01,
         f"Based on {len(records):,} player records from {len(records)//4:,} games  "
         f"(VP formula: settlements + cities×2 + VP cards + LR×2 + LA×2)",
         ha="center", fontsize=9, color="#777")

fig.savefig(FIG_DIR / "fig4_summary.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved fig4_summary.png")
