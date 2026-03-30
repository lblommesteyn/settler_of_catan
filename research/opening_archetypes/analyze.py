"""
Opening Archetype Clustering
==============================
Cluster all 169k openings into strategic types based on their resource profile,
then compare win rates across archetypes.

Features per opening = v1_features + v2_features (element-wise sum):
  wood_pips, brick_pips, sheep_pips, wheat_pips, ore_pips  (raw pip counts)
  has_port                                                   (0/1 each vertex)
  num_adj_hexes                                              (interior-ness)

Pipeline:
  1. Extract features from tensor cache (fast, no re-loading games)
  2. K-means clustering (k=6)
  3. Name clusters by dominant resource profile
  4. Compute win rates by cluster, controlling for seat
  5. Visualise: radar charts, win rate bars, PCA scatter, seat heatmap
"""
from __future__ import annotations

import sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COL = "#eaeaea"
GRID_COL = "#2a2a4a"

CLUSTER_COLOURS = ["#e94560", "#44aaff", "#44cc88", "#ffaa44", "#cc88ff", "#ffdd44"]
RESOURCES = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]

K = 6   # number of archetypes


# ── feature extraction ─────────────────────────────────────────────────────────

def extract_opening_features(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load tensor cache and compute per-opening resource features.

    Returns:
        feats  (N, 7)  — wood/brick/sheep/wheat/ore/has_port/adj3 summed over v1+v2
        y_win  (N,)    — binary win label
        seat   (N,)    — seat 0-3
    """
    log.info("Loading tensor cache ...")
    cache = torch.load(cache_path, weights_only=True)
    X      = cache["X"].numpy()        # (N, 54, 16)  — note: features 0-6 are board, 14/15 are flags
    v1_idx = cache["v1"].numpy()       # (N,)
    v2_idx = cache["v2"].numpy()       # (N,)
    y_win  = cache["y_win"].numpy()    # (N,)
    seat   = cache["seat"].numpy()     # (N,)
    N = X.shape[0]
    log.info("Loaded %d openings", N)

    # Feature indices in X[i, v, :]:
    #   0  wood_pips/10    1  brick_pips/10   2  sheep_pips/10
    #   3  wheat_pips/10   4  ore_pips/10     5  total_pips/30
    #   6  num_adj_hexes/3  7  has_port
    FEAT_IDX = [0, 1, 2, 3, 4, 6, 7]  # skip total_pips (redundant)
    FEAT_NAMES = ["wood", "brick", "sheep", "wheat", "ore", "adj3", "port"]

    feats = np.zeros((N, len(FEAT_IDX)), dtype=np.float32)
    for fi, xi in enumerate(FEAT_IDX):
        feats[:, fi] = (X[np.arange(N), v1_idx, xi] +
                        X[np.arange(N), v2_idx, xi])

    # Un-normalise pips back to actual values for interpretability
    feats[:, :5] *= 10    # pips were divided by 10
    feats[:, 5]  *= 3     # adj3 was divided by 3
    # port stays 0-2 (sum of two binary values)

    return feats, y_win, seat, FEAT_NAMES


# ── clustering ─────────────────────────────────────────────────────────────────

def cluster_openings(feats: np.ndarray, k: int = K, seed: int = 42):
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(feats_scaled)
    centres = scaler.inverse_transform(km.cluster_centers_)
    log.info("K-means converged. Cluster sizes: %s",
             np.bincount(labels).tolist())
    return labels, centres, scaler


def name_clusters(centres: np.ndarray, feat_names: list[str]) -> list[str]:
    """
    Auto-name each cluster based on its dominant resource profile.
    centres: (K, 7) in original (un-scaled) units.
    """
    names = []
    res_idx = [feat_names.index(r) for r in ["wood", "brick", "sheep", "wheat", "ore"]]
    port_idx = feat_names.index("port")
    adj_idx  = feat_names.index("adj3")

    for c in centres:
        res_pips  = c[res_idx]
        total     = res_pips.sum()
        top2_idx  = res_pips.argsort()[-2:][::-1]
        top_res   = [RESOURCES[i] for i in top2_idx]
        has_port  = c[port_idx] > 0.6
        is_int    = c[adj_idx] > 4.5   # most vertices adj to 3 hexes
        is_high_pip = total > 18

        # Dominant resource pair
        r1, r2 = top_res
        if has_port and c[port_idx] > 1.0:
            name = f"Double-port ({r1})"
        elif has_port:
            name = f"Port + {r1}"
        elif {r1, r2} == {"Ore", "Wheat"}:
            name = "Ore+Wheat turtle"
        elif {r1, r2} == {"Wood", "Brick"}:
            name = "Wood+Brick rush"
        elif r1 in ("Sheep", "Wheat") and r2 in ("Sheep", "Wheat"):
            name = "Sheep+Wheat"
        elif is_high_pip:
            name = f"High-pip ({r1})"
        else:
            name = f"Balanced ({r1}+{r2})"
        names.append(name)
    return names


# ── analysis ───────────────────────────────────────────────────────────────────

def win_rates_by_cluster(labels, y_win, seat, k=K):
    rows = []
    for c in range(k):
        mask = labels == c
        n    = mask.sum()
        wins = y_win[mask].sum()
        rate = wins / n
        se   = np.sqrt(rate * (1 - rate) / n)
        # Seat-adjusted: expected win rate if seat distribution were uniform
        seat_win_rates = []
        for s in range(4):
            sm = mask & (seat == s)
            if sm.sum() > 0:
                seat_win_rates.append(y_win[sm].mean())
        adj_rate = np.mean(seat_win_rates) if seat_win_rates else rate
        rows.append({"cluster": c, "n": n, "win_rate": rate,
                     "adj_rate": adj_rate, "se": se})
    return pd.DataFrame(rows).set_index("cluster")


def seat_distribution(labels, seat, k=K):
    """Fraction of each cluster that belongs to each seat."""
    result = np.zeros((k, 4))
    for c in range(k):
        mask = labels == c
        for s in range(4):
            result[c, s] = (seat[mask] == s).sum() / mask.sum()
    return result


# ── figures ────────────────────────────────────────────────────────────────────

def fig1_radar(centres, names, feat_names):
    """Radar chart of resource profile per cluster."""
    res_idx   = [feat_names.index(r) for r in ["wood", "brick", "sheep", "wheat", "ore"]]
    res_names = RESOURCES
    n_res     = len(res_names)
    angles    = np.linspace(0, 2 * np.pi, n_res, endpoint=False).tolist()
    angles   += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                             subplot_kw=dict(polar=True),
                             facecolor=DARK_BG)
    fig.suptitle("Opening Archetype Resource Profiles", color=TEXT_COL, fontsize=13)

    for i, (ax, name, c) in enumerate(zip(axes.flat, names, CLUSTER_COLOURS)):
        vals = centres[i][res_idx].tolist()
        vals += vals[:1]
        ax.set_facecolor(PANEL_BG)
        ax.plot(angles, vals, color=c, lw=2)
        ax.fill(angles, vals, color=c, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(res_names, color=TEXT_COL, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(name, color=c, fontsize=9, pad=12)
        ax.spines["polar"].set_color(GRID_COL)
        ax.tick_params(colors=GRID_COL)
        # annotate total pips
        total = centres[i][res_idx].sum()
        ax.text(0.5, -0.08, f"Total pips: {total:.1f}", transform=ax.transAxes,
                ha="center", color=TEXT_COL, fontsize=7.5, alpha=0.8)

    fig.tight_layout()
    p = OUT_DIR / "fig1_radar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig2_win_rates(wr_df, names):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    x      = np.arange(K)
    rates  = wr_df.win_rate.values * 100
    adj    = wr_df.adj_rate.values * 100
    errors = wr_df.se.values * 100 * 1.96
    ns     = wr_df.n.values

    bars = ax.bar(x - 0.2, rates, width=0.35, color=CLUSTER_COLOURS,
                  label="Raw win rate", zorder=3)
    ax.bar(x + 0.2, adj, width=0.35, color=CLUSTER_COLOURS,
           alpha=0.45, label="Seat-adjusted", zorder=3)
    ax.errorbar(x - 0.2, rates, yerr=errors, fmt="none",
                color="white", capsize=3, lw=1.2, zorder=4)
    ax.axhline(25, color="white", lw=1.2, ls="--", alpha=0.6, label="Null (25%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={ns[i]/1000:.0f}k)" for i, n in enumerate(names)],
                        color=TEXT_COL, fontsize=8)
    ax.set_ylabel("Win rate (%)", color=TEXT_COL)
    ax.set_title("Win Rate by Opening Archetype", color=TEXT_COL, pad=10)
    ax.set_ylim(22, 29)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, zorder=0)
    ax.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{r:.1f}%", ha="center", color=TEXT_COL, fontsize=8, fontweight="bold")

    fig.tight_layout()
    p = OUT_DIR / "fig2_win_rates.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig3_pca(feats, labels, names, sample=8000):
    """PCA scatter of openings coloured by cluster."""
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(feats), sample, replace=False)
    pca  = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(StandardScaler().fit_transform(feats[idx]))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for c, name, col in zip(range(K), names, CLUSTER_COLOURS):
        mask = labels[idx] == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=col, alpha=0.25, s=8, label=name)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)", color=TEXT_COL)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)", color=TEXT_COL)
    ax.set_title("Opening Archetypes in PCA Space", color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.legend(fontsize=7.5, labelcolor=TEXT_COL, facecolor=DARK_BG,
              edgecolor=GRID_COL, markerscale=3)

    fig.tight_layout()
    p = OUT_DIR / "fig3_pca.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig4_seat_heatmap(seat_dist, names):
    """Heatmap: which seats favour which archetypes?"""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    im = ax.imshow(seat_dist.T * 100, aspect="auto", cmap="YlOrRd",
                   vmin=20, vmax=35)
    ax.set_xticks(range(K))
    ax.set_xticklabels(names, color=TEXT_COL, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Seat {s}" for s in range(4)], color=TEXT_COL)
    ax.set_title("Archetype Usage by Seat (%)", color=TEXT_COL, pad=10)

    for i in range(K):
        for j in range(4):
            ax.text(i, j, f"{seat_dist[i, j]*100:.0f}%",
                    ha="center", va="center", color="black", fontsize=8, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("% of seat's openings", color=TEXT_COL, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    fig.tight_layout()
    p = OUT_DIR / "fig4_seat_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig5_profile_bars(centres, names, feat_names, wr_df):
    """Horizontal stacked bar: resource composition + win rate annotation."""
    res_idx = [feat_names.index(r) for r in ["wood", "brick", "sheep", "wheat", "ore"]]
    RES_COLS = ["#44aa44", "#cc6622", "#eeeeaa", "#dddd44", "#888888"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG,
                              gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Opening Archetype Profiles and Win Rates", color=TEXT_COL, fontsize=13)

    # Left: stacked bar of resource pips
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    y = np.arange(K)
    lefts = np.zeros(K)
    for ri, (res, col) in enumerate(zip(RESOURCES, RES_COLS)):
        vals = centres[:, res_idx[ri]]
        ax.barh(y, vals, left=lefts, color=col, label=res, height=0.55)
        for yi, (v, l) in enumerate(zip(vals, lefts)):
            if v > 0.8:
                ax.text(l + v / 2, yi, f"{v:.1f}", ha="center", va="center",
                        color="black", fontsize=7.5, fontweight="bold")
        lefts += vals

    # Port indicator
    port_idx = feat_names.index("port")
    for yi, c in enumerate(range(K)):
        if centres[c, port_idx] > 0.5:
            ax.text(lefts[yi] + 0.3, yi, f"P({centres[c,port_idx]:.1f})",
                    va="center", color="#ffdd44", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TEXT_COL, fontsize=9)
    ax.set_xlabel("Combined pip count (v1 + v2)", color=TEXT_COL)
    ax.set_title("Resource Composition", color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[:].set_color(GRID_COL)
    ax.legend(loc="lower right", fontsize=8, labelcolor=TEXT_COL,
              facecolor=DARK_BG, edgecolor=GRID_COL)
    ax.xaxis.grid(True, color=GRID_COL, zorder=0)

    # Right: win rate bars
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    rates  = wr_df.win_rate.values * 100
    errors = wr_df.se.values * 100 * 1.96
    bars = ax2.barh(y, rates, color=CLUSTER_COLOURS, height=0.55)
    ax2.errorbar(rates, y, xerr=errors, fmt="none", color="white", capsize=3, lw=1.2)
    ax2.axvline(25, color="white", lw=1.2, ls="--", alpha=0.6)
    for bar, r in zip(bars, rates):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{r:.1f}%", va="center", color=TEXT_COL, fontsize=8, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Win rate (%)", color=TEXT_COL)
    ax2.set_title("Win Rate", color=TEXT_COL)
    ax2.set_xlim(22, 29)
    ax2.tick_params(colors=TEXT_COL)
    ax2.spines[:].set_color(GRID_COL)
    ax2.xaxis.grid(True, color=GRID_COL, zorder=0)

    fig.tight_layout()
    p = OUT_DIR / "fig5_profile_bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    cache_path = ROOT / "data" / "gnn" / "tensors_cache.pt"

    feats, y_win, seat, feat_names = extract_opening_features(cache_path)

    log.info("Clustering into %d archetypes ...", K)
    labels, centres, scaler = cluster_openings(feats, k=K)

    names = name_clusters(centres, feat_names)
    log.info("Archetype names: %s", names)

    wr_df    = win_rates_by_cluster(labels, y_win, seat)
    seat_dist = seat_distribution(labels, seat)

    log.info("=== Win Rates by Archetype ===")
    for c, name in enumerate(names):
        r = wr_df.loc[c]
        log.info("  %-30s  win=%.1f%%  n=%d", name, r.win_rate*100, r.n)

    log.info("Generating figures ...")
    fig1_radar(centres, names, feat_names)
    fig2_win_rates(wr_df, names)
    fig3_pca(feats, labels, names)
    fig4_seat_heatmap(seat_dist, names)
    fig5_profile_bars(centres, names, feat_names, wr_df)
    log.info("Done.")

    return names, centres, wr_df, seat_dist


if __name__ == "__main__":
    main()
