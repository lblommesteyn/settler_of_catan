"""
GNN Vertex Importance Analysis
================================
Core question: where do humans leave value on the table in the opening,
and is the divergence between human choices and GNN-optimal spatially
structured on the board?

For each test-set opening we:
  1. Hold v1 fixed at the human's actual choice.
  2. Enumerate all 54 possible v2 placements.
  3. Score each with the trained StaticBoardGCN.
  4. Compute per-vertex statistics:
       - frequency   : how often humans choose this vertex as v2
       - gnn_value   : average GNN score when v2 is placed here
       - regret      : average (best_available_v2 - actual) for openings where v2=here

Outputs (research/gnn_analysis/figures/):
  fig1_v2_frequency.png     — human v2 placement frequency heatmap
  fig2_v2_gnn_value.png     — average GNN win-prob by v2 vertex
  fig3_v2_regret.png        — average regret by v2 vertex
  fig4_human_vs_gnn.png     — side-by-side human freq vs GNN value + mismatch
  fig5_best_worst_boards.png — example boards: optimal vs human choice
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from catan.board.board import CatanBoard
from catan.models.gnn.board_to_graph import _sorted_vertices, _get_edge_index
from catan.models.gnn.static_gcn import StaticBoardGCN, build_norm_adj, N_VERTS, NODE_DIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT_DIR  = Path(__file__).parent / "figures"
DATA_DIR = ROOT / "data" / "gnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: str = "cpu") -> StaticBoardGCN:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg  = ckpt.get("config", {})
    model = StaticBoardGCN(
        hidden    = cfg.get("hidden",  64),
        n_layers  = cfg.get("layers",  3),
        dropout   = 0.0,
        multitask = True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_adj(ckpt["A_hat"].to(device))
    model.eval()
    return model


def get_vertex_coords() -> np.ndarray:
    """Return (54, 2) array of vertex (x, y) coords from the canonical board."""
    board = CatanBoard.random(seed=0)
    verts = _sorted_vertices(board)
    return np.array([[v[0], v[1]] for v in verts], dtype=np.float32)


def get_hex_coords() -> tuple[np.ndarray, float]:
    """Return hex centres (N_hex, 2) and approximate hex radius."""
    board = CatanBoard.random(seed=0)
    seen, centres = set(), []
    for hexes in board.vertex_hexes.values():
        for hx in hexes:
            key = (hx.q, hx.r)
            if key not in seen:
                seen.add(key)
                xy = hx.cartesian
                centres.append([xy[0], xy[1]])
    centres = np.array(centres, dtype=np.float32)
    # estimate hex radius from shortest inter-centre distance
    diffs = []
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            d = np.linalg.norm(centres[i] - centres[j])
            if 0.5 < d < 4.0:
                diffs.append(d)
    # shortest distance between adjacent hex centres = sqrt(3) * radius
    min_d  = np.percentile(diffs, 5) if diffs else np.sqrt(3)
    radius = min_d / np.sqrt(3)
    return centres, radius


# ── score all v2 alternatives for test set ───────────────────────────────────

@torch.no_grad()
def score_v2_alternatives(
    model:    StaticBoardGCN,
    X_all:    torch.Tensor,   # (N_total, 54, 16)
    v1_all:   torch.Tensor,   # (N_total,)
    v2_all:   torch.Tensor,   # (N_total,)
    seat_all: torch.Tensor,   # (N_total,)
    test_idx: torch.Tensor,
    batch_size: int = 256,
    device:   str = "cpu",
) -> dict:
    """
    For each test opening, enumerate all 54 v2 alternatives (keeping v1 fixed).
    Returns dict with arrays indexed by test opening.
    """
    N_test = len(test_idx)
    log.info("Scoring v2 alternatives for %d test openings ...", N_test)

    # Storage
    actual_scores  = np.zeros(N_test, dtype=np.float32)
    best_scores    = np.zeros(N_test, dtype=np.float32)
    actual_v2_idx  = v2_all[test_idx].numpy()    # (N_test,)
    actual_v1_idx  = v1_all[test_idx].numpy()
    actual_seats   = seat_all[test_idx].numpy()
    all_v2_scores  = np.zeros((N_test, N_VERTS), dtype=np.float32)  # per-vertex scores

    for batch_start in range(0, N_test, batch_size):
        batch_end = min(batch_start + batch_size, N_test)
        B = batch_end - batch_start
        global_idx = test_idx[batch_start:batch_end]  # indices into full dataset

        # Base features for this batch: (B, 54, 16) — zero out v1/v2 flags
        X_base = X_all[global_idx].clone()
        X_base[:, :, 14] = 0.0  # is_v1
        X_base[:, :, 15] = 0.0  # is_v2

        # Set v1 flags back
        for b in range(B):
            X_base[b, actual_v1_idx[batch_start + b], 14] = 1.0

        # Expand: each opening × 54 v2 candidates → (B*54, 54, 16)
        X_exp   = X_base.unsqueeze(1).expand(B, N_VERTS, N_VERTS, NODE_DIM)
        X_flat  = X_exp.reshape(B * N_VERTS, N_VERTS, NODE_DIM).clone()
        v2_try  = torch.arange(N_VERTS, device=device).unsqueeze(0).expand(B, -1).reshape(-1)

        # Set v2 flags
        batch_offset = torch.arange(B, device=device).unsqueeze(1).expand(B, N_VERTS).reshape(-1)
        X_flat[torch.arange(B * N_VERTS), v2_try] = X_flat[torch.arange(B * N_VERTS), v2_try].clone()
        for i in range(B * N_VERTS):
            X_flat[i, v2_try[i].item(), 15] = 1.0

        v1_exp  = torch.tensor(actual_v1_idx[batch_start:batch_end], device=device, dtype=torch.long)
        v1_exp  = v1_exp.unsqueeze(1).expand(B, N_VERTS).reshape(-1)
        seat_exp = torch.tensor(actual_seats[batch_start:batch_end], device=device, dtype=torch.long)
        seat_exp = seat_exp.unsqueeze(1).expand(B, N_VERTS).reshape(-1)

        scores = model.predict_proba(X_flat.to(device), v1_exp, v2_try, seat_exp).cpu().numpy()
        scores = scores.reshape(B, N_VERTS)

        all_v2_scores[batch_start:batch_end] = scores
        best_scores[batch_start:batch_end]   = scores.max(axis=1)
        for b in range(B):
            actual_scores[batch_start + b] = scores[b, actual_v2_idx[batch_start + b]]

        if batch_start % (batch_size * 20) == 0:
            log.info("  %d / %d openings processed", batch_start, N_test)

    regret = best_scores - actual_scores

    return {
        "actual_scores":  actual_scores,
        "best_scores":    best_scores,
        "regret":         regret,
        "all_v2_scores":  all_v2_scores,  # (N_test, 54)
        "actual_v2_idx":  actual_v2_idx,
        "actual_v1_idx":  actual_v1_idx,
        "actual_seats":   actual_seats,
    }


# ── per-vertex aggregation ────────────────────────────────────────────────────

def aggregate_per_vertex(results: dict) -> dict:
    """Compute per-vertex frequency, average GNN value, average regret."""
    freq     = np.zeros(N_VERTS, dtype=np.float32)
    gnn_val  = np.zeros(N_VERTS, dtype=np.float32)
    reg_sum  = np.zeros(N_VERTS, dtype=np.float32)
    reg_cnt  = np.zeros(N_VERTS, dtype=np.int32)

    # Mean GNN value across ALL openings for each v2 position
    gnn_val_all = results["all_v2_scores"].mean(axis=0)  # (54,)

    for i, v2 in enumerate(results["actual_v2_idx"]):
        freq[v2]    += 1
        reg_sum[v2] += results["regret"][i]
        reg_cnt[v2] += 1

    avg_regret = np.where(reg_cnt > 0, reg_sum / np.maximum(reg_cnt, 1), np.nan)
    freq_norm  = freq / freq.sum()

    return {
        "freq":         freq,
        "freq_norm":    freq_norm,
        "gnn_val":      gnn_val_all,      # average GNN score if placing v2 here
        "avg_regret":   avg_regret,       # average regret when human places v2 here
        "overall_regret_mean": results["regret"].mean(),
        "overall_regret_p25":  np.percentile(results["regret"], 25),
        "overall_regret_p75":  np.percentile(results["regret"], 75),
        "pct_optimal": (results["regret"] < 0.001).mean(),
    }


# ── visualisation helpers ─────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#e94560"
TEXT_COL  = "#eaeaea"
GRID_COL  = "#2a2a4a"

def board_heatmap(
    coords:    np.ndarray,        # (54, 2)
    values:    np.ndarray,        # (54,)  — colour
    sizes:     np.ndarray | None, # (54,)  — dot size (optional)
    title:     str,
    cbar_label: str,
    cmap:      str  = "RdYlGn",
    vmin:      float | None = None,
    vmax:      float | None = None,
    ax=None,
    fig=None,
    annotate:  bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Draw subtle hex grid lines
    hx_centres, hx_r = get_hex_coords()
    for cx, cy in hx_centres:
        hex_patch = mpatches.RegularPolygon(
            (cx, cy), numVertices=6, radius=hx_r * 0.98,
            orientation=0, linewidth=0.5,
            edgecolor=GRID_COL, facecolor="none", zorder=1,
        )
        ax.add_patch(hex_patch)

    # Draw edges between adjacent vertices
    board = CatanBoard.random(seed=0)
    verts = _sorted_vertices(board)
    v_idx = {v: i for i, v in enumerate(verts)}
    for v in verts:
        for nb in board.graph.vertex_neighbors[v]:
            if nb in v_idx and v_idx[nb] > v_idx[v]:
                x1, y1 = coords[v_idx[v]]
                x2, y2 = coords[v_idx[nb]]
                ax.plot([x1, x2], [y1, y2], color=GRID_COL, lw=0.8, zorder=2)

    norm = Normalize(vmin=vmin or np.nanmin(values), vmax=vmax or np.nanmax(values))
    cmap_obj = cm.get_cmap(cmap)
    colours  = [cmap_obj(norm(v)) if not np.isnan(v) else (0.3, 0.3, 0.3, 1.0)
                for v in values]

    base_s = 180
    if sizes is not None:
        s_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
        dot_sizes = base_s * 0.4 + s_norm * base_s * 1.6
    else:
        dot_sizes = np.full(len(values), base_s)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colours,
                    s=dot_sizes, zorder=5, edgecolors="white", linewidths=0.3)

    if annotate:
        for i, (x, y) in enumerate(coords):
            if not np.isnan(values[i]):
                ax.text(x, y, f"{values[i]:.2f}", fontsize=4.5,
                        ha="center", va="center", color="black", zorder=6)

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, color=TEXT_COL, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cb.outline.set_edgecolor(GRID_COL)

    ax.set_title(title, color=TEXT_COL, fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ── main figures ──────────────────────────────────────────────────────────────

def fig1_frequency(coords, stats):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    board_heatmap(
        coords, stats["freq_norm"] * 100,
        sizes=None,
        title="Human v2 Placement Frequency (%)",
        cbar_label="% of openings",
        cmap="YlOrRd",
        ax=ax, fig=fig,
        annotate=True,
    )
    fig.tight_layout()
    p = OUT_DIR / "fig1_v2_frequency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig2_gnn_value(coords, stats):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    board_heatmap(
        coords, stats["gnn_val"],
        sizes=None,
        title="GNN Win Probability by v2 Vertex",
        cbar_label="Avg P(win)",
        cmap="RdYlGn",
        ax=ax, fig=fig,
        annotate=True,
    )
    fig.tight_layout()
    p = OUT_DIR / "fig2_v2_gnn_value.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig3_regret(coords, stats):
    valid = ~np.isnan(stats["avg_regret"])
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    board_heatmap(
        coords, stats["avg_regret"],
        sizes=stats["freq"],
        title="Average GNN Regret by v2 Vertex\n(dot size = placement frequency)",
        cbar_label="Avg regret (P(win) gap)",
        cmap="RdYlGn_r",
        vmin=0.0,
        ax=ax, fig=fig,
        annotate=False,
    )
    fig.tight_layout()
    p = OUT_DIR / "fig3_v2_regret.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig4_mismatch(coords, stats):
    """Side-by-side: human frequency vs GNN value, plus mismatch score."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)
    fig.suptitle("Human Choice vs GNN Optimal: v2 Placement",
                 color=TEXT_COL, fontsize=14, y=1.01)

    board_heatmap(coords, stats["freq_norm"] * 100, sizes=None,
                  title="Human Frequency (%)", cbar_label="% chosen",
                  cmap="YlOrRd", ax=axes[0], fig=fig)

    board_heatmap(coords, stats["gnn_val"] * 100, sizes=None,
                  title="GNN Win Probability (%)", cbar_label="GNN P(win) %",
                  cmap="RdYlGn", ax=axes[1], fig=fig)

    # Mismatch = standardised (human_freq - gnn_val normalised)
    freq_z = (stats["freq_norm"] - stats["freq_norm"].mean()) / (stats["freq_norm"].std() + 1e-9)
    gnn_z  = (stats["gnn_val"]  - stats["gnn_val"].mean())   / (stats["gnn_val"].std()  + 1e-9)
    mismatch = freq_z - gnn_z   # positive = humans overvalue, negative = humans undervalue

    board_heatmap(coords, mismatch, sizes=None,
                  title="Human–GNN Mismatch\n(+ve = humans overvalue, −ve = undervalue)",
                  cbar_label="z-score diff",
                  cmap="RdBu_r",
                  ax=axes[2], fig=fig)

    fig.tight_layout()
    p = OUT_DIR / "fig4_human_vs_gnn.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


def fig5_summary_stats(results: dict, stats: dict):
    """Summary statistics panel."""
    fig = plt.figure(figsize=(12, 4), facecolor=DARK_BG)
    fig.suptitle("GNN Opening Regret: Summary", color=TEXT_COL, fontsize=14)

    # Panel 1: regret distribution
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(PANEL_BG)
    reg = results["regret"]
    ax1.hist(reg, bins=60, color=ACCENT, edgecolor="none", alpha=0.85)
    ax1.axvline(reg.mean(), color="white", lw=1.5, ls="--", label=f"Mean={reg.mean():.4f}")
    ax1.axvline(0, color="#44ff88", lw=1.5, ls="-", label="Optimal")
    ax1.set_xlabel("GNN Regret (P(win) gap)", color=TEXT_COL)
    ax1.set_ylabel("Count", color=TEXT_COL)
    ax1.set_title("v2 Regret Distribution", color=TEXT_COL)
    ax1.tick_params(colors=TEXT_COL)
    ax1.spines[:].set_color(GRID_COL)
    ax1.legend(fontsize=8, labelcolor=TEXT_COL, facecolor=DARK_BG, edgecolor=GRID_COL)

    # Panel 2: % optimal by seat
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(PANEL_BG)
    seats = results["actual_seats"]
    pct_opt = [(results["regret"][seats == s] < 0.001).mean() * 100 for s in range(4)]
    mean_reg = [results["regret"][seats == s].mean() for s in range(4)]
    bars = ax2.bar(range(4), pct_opt, color=[ACCENT, "#44aaff", "#ffaa44", "#44ff88"])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"Seat {s}" for s in range(4)], color=TEXT_COL)
    ax2.set_ylabel("% choosing optimal v2", color=TEXT_COL)
    ax2.set_title("Optimal v2 Rate by Seat", color=TEXT_COL)
    ax2.tick_params(colors=TEXT_COL)
    ax2.spines[:].set_color(GRID_COL)
    for bar, v in zip(bars, pct_opt):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", color=TEXT_COL, fontsize=9)

    # Panel 3: top-5 over/undervalued vertices
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(PANEL_BG)
    freq_z = (stats["freq_norm"] - stats["freq_norm"].mean()) / (stats["freq_norm"].std() + 1e-9)
    gnn_z  = (stats["gnn_val"]  - stats["gnn_val"].mean())   / (stats["gnn_val"].std()  + 1e-9)
    mismatch = freq_z - gnn_z

    top_over  = np.argsort(mismatch)[-5:][::-1]
    top_under = np.argsort(mismatch)[:5]
    labels = [f"V{i}" for i in top_over] + [f"V{i}" for i in top_under]
    values = list(mismatch[top_over]) + list(mismatch[top_under])
    colours = [ACCENT] * 5 + ["#44aaff"] * 5
    y_pos = range(len(labels))
    ax3.barh(list(y_pos), values, color=colours)
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(labels, color=TEXT_COL, fontsize=9)
    ax3.axvline(0, color="white", lw=0.8)
    ax3.set_xlabel("Mismatch z-score", color=TEXT_COL)
    ax3.set_title("Most Over/Undervalued Vertices\n(red=over, blue=under)", color=TEXT_COL)
    ax3.tick_params(colors=TEXT_COL)
    ax3.spines[:].set_color(GRID_COL)

    fig.tight_layout()
    p = OUT_DIR / "fig5_summary_stats.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    device = "cpu"

    # Load model
    ckpt_path = DATA_DIR / "model_static_gcn.pt"
    log.info("Loading model from %s", ckpt_path)
    model = load_model(ckpt_path, device)

    # Load tensor cache + split
    log.info("Loading tensor cache ...")
    cache = torch.load(DATA_DIR / "tensors_cache.pt", weights_only=True)
    X_all    = cache["X"]
    v1_all   = cache["v1"]
    v2_all   = cache["v2"]
    seat_all = cache["seat"]

    split     = torch.load(DATA_DIR / "split.pt", weights_only=False)
    test_idx  = torch.tensor(split["test"], dtype=torch.long)
    log.info("Test set: %d openings", len(test_idx))

    # Vertex coordinates from canonical board
    coords = get_vertex_coords()

    # Score all v2 alternatives
    results = score_v2_alternatives(
        model, X_all, v1_all, v2_all, seat_all,
        test_idx, batch_size=256, device=device,
    )

    # Aggregate per-vertex stats
    stats = aggregate_per_vertex(results)

    log.info("=== Results ===")
    log.info("  Mean GNN v2 regret:   %.4f", stats["overall_regret_mean"])
    log.info("  P25 / P75 regret:     %.4f / %.4f",
             stats["overall_regret_p25"], stats["overall_regret_p75"])
    log.info("  Pct optimal v2:       %.1f%%", stats["pct_optimal"] * 100)

    # Figures
    log.info("Generating figures ...")
    fig1_frequency(coords, stats)
    fig2_gnn_value(coords, stats)
    fig3_regret(coords, stats)
    fig4_mismatch(coords, stats)
    fig5_summary_stats(results, stats)

    log.info("All figures saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
