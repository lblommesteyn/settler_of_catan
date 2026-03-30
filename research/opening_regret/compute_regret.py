"""
Opening Regret Engine
=====================

For each game, reconstructs the board and computes per-player regret metrics.
Streaming from games.tar.gz (no extraction needed).

Metrics computed:
  v2_regret   — exact: given actual v1, how much better was the best v2?
  v1_pip_rank — how did actual v1's pip count rank among all legal v1s?
  v2_rank     — percentile rank of actual v2 choice among all legal v2s
  biases      — pip/expansion/port differences: actual vs best-v2 alternative

Full-pair enumeration is skipped (too slow: 54×50=2700 pairs/player×4=10k/game).
Instead the research questions are answered via:
  - exact v2 regret (the choice that matters most strategically)
  - v1 pip rank (proxy for v1 quality)
  - combined score gap (actual vs best-achievable-from-actual-v1)

Usage:
    python research/opening_regret/compute_regret.py [--games N]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from catan.board.board import CatanBoard
from catan.data.loader import stream_from_tarfile, colonist_record_to_board
from catan.features.opening_features import (
    compute_opening_features,
    opening_features_to_array,
    compute_all_vertex_features,
)
from catan.features.vertex_features import VertexFeatures
from catan.models.ml_model import LogisticOpeningModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(path: Path) -> LogisticOpeningModel:
    return LogisticOpeningModel.load(path)


def safe_features(v1, v2, seat, board, vf_cache) -> np.ndarray | None:
    try:
        of = compute_opening_features(v1, v2, seat, board, vf_cache)
        return opening_features_to_array(of)
    except Exception:
        return None


def safe_of(v1, v2, seat, board, vf_cache):
    try:
        return compute_opening_features(v1, v2, seat, board, vf_cache)
    except Exception:
        return None


def legal_v1_available(board: CatanBoard, occupied: set) -> list:
    too_close: set = set()
    for occ in occupied:
        too_close.add(occ)
        too_close.update(board.graph.vertex_neighbors[occ])
    return [v for v in board.legal_starting_vertices() if v not in too_close]


def compute_game_regret(record, board: CatanBoard, model) -> list[dict]:
    """
    Returns one dict per player per game.

    Strategy: only enumerate v2 alternatives (fast, ~18/player).
    For v1 quality, use pip-count rank over all legal v1 alts (O(54) lookups).
    Batch model predictions per game.
    """
    n_players = 4
    all_verts = sorted(board.all_vertices(),
                       key=lambda v: (round(v[1], 6), round(v[0], 6)))

    def corner_to_vert(cid):
        return all_verts[cid] if 0 <= cid < len(all_verts) else None

    # Reconstruct per-player placements
    player_placements: dict[int, list] = {}
    for p in record.opening_placements:
        player_placements.setdefault(p.player_index, []).append(p)
    for pid in player_placements:
        player_placements[pid].sort(key=lambda p: p.action_seq)

    color_to_seat = {color: seat for seat, color in enumerate(record.play_order[:n_players])}

    seat_vertices: dict[int, tuple] = {}
    for color, placements in player_placements.items():
        seat = color_to_seat.get(color)
        if seat is None or len(placements) < 2:
            continue
        v1 = corner_to_vert(placements[0].vertex_index)
        v2 = corner_to_vert(placements[1].vertex_index)
        if v1 is not None and v2 is not None:
            seat_vertices[seat] = (v1, v2)

    if len(seat_vertices) < 2:
        return []

    vf_cache = compute_all_vertex_features(board)

    # Occupied-at-v1-time for each seat (snake draft)
    draft_order = board.snake_draft_order(n_players)
    occupied_at_v1: dict[int, set] = {}
    occ_so_far: set = set()
    for seat in draft_order[:n_players]:
        occupied_at_v1[seat] = set(occ_so_far)
        v1 = seat_vertices.get(seat, (None, None))[0]
        if v1 is not None:
            occ_so_far.add(v1)

    all_v1s = {seat_vertices[s][0] for s in seat_vertices}

    # ── collect all arrays to batch-score ────────────────────────────────────
    all_arrays: list[np.ndarray] = []
    per_seat_meta: dict[int, dict] = {}

    for seat_idx in range(n_players):
        if seat_idx not in seat_vertices:
            continue
        actual_v1, actual_v2 = seat_vertices[seat_idx]
        occ_v2 = set(all_v1s)

        # actual pair
        arr_actual = safe_features(actual_v1, actual_v2, seat_idx, board, vf_cache)
        if arr_actual is None:
            continue
        actual_of = safe_of(actual_v1, actual_v2, seat_idx, board, vf_cache)
        if actual_of is None:
            continue

        actual_idx = len(all_arrays)
        all_arrays.append(arr_actual)

        # v2 alternatives (given actual v1)
        v2_alts = board.legal_second_vertices(actual_v1, occ_v2)
        v2_start = len(all_arrays)
        v2_valid: list = []
        for v2_alt in v2_alts:
            arr = safe_features(actual_v1, v2_alt, seat_idx, board, vf_cache)
            if arr is not None:
                all_arrays.append(arr)
                v2_valid.append(v2_alt)

        # v1 alternatives: just count/rank by pip count (cheap — uses vf_cache)
        occ_v1 = occupied_at_v1.get(seat_idx, set())
        v1_alts = legal_v1_available(board, occ_v1)
        # pip count for each v1 alt from vf_cache (index 0 = total_pips)
        v1_pip_counts = {v: float(vf_cache[v].total_pips) for v in v1_alts if v in vf_cache}
        actual_v1_pips = float(vf_cache[actual_v1].total_pips) if actual_v1 in vf_cache else 0.0
        # rank: how many v1 alternatives have strictly higher pip count
        v1_pip_rank = sum(1 for p in v1_pip_counts.values() if p > actual_v1_pips) + 1
        v1_n_alts   = len(v1_alts)

        per_seat_meta[seat_idx] = {
            "actual_v1":   actual_v1,
            "actual_v2":   actual_v2,
            "actual_of":   actual_of,
            "actual_idx":  actual_idx,
            "v2_alts":     v2_valid,
            "v2_start":    v2_start,
            "v2_end":      len(all_arrays),
            "v1_pip_rank": v1_pip_rank,
            "v1_n_alts":   v1_n_alts,
        }

    if not all_arrays:
        return []

    # ── single batch prediction ───────────────────────────────────────────────
    scores = model.predict_batch(np.vstack(all_arrays))

    # ── extract results ───────────────────────────────────────────────────────
    results = []
    for seat_idx, sd in per_seat_meta.items():
        actual_score = float(scores[sd["actual_idx"]])
        actual_of    = sd["actual_of"]
        actual_v2    = sd["actual_v2"]

        v2_score_slice = scores[sd["v2_start"]:sd["v2_end"]]
        v2_valid       = sd["v2_alts"]
        if len(v2_score_slice) == 0:
            continue

        # best v2
        best_v2_pos   = int(np.argmax(v2_score_slice))
        best_v2_score = float(v2_score_slice[best_v2_pos])
        best_v2       = v2_valid[best_v2_pos]
        v2_regret     = best_v2_score - actual_score

        # rank of actual_v2 among v2 alternatives
        try:
            av2_pos = v2_valid.index(actual_v2)
            av2_score = float(v2_score_slice[av2_pos])
            v2_rank = int(np.sum(v2_score_slice >= av2_score))
        except ValueError:
            v2_rank = len(v2_valid)          # not found — treat as last

        # bias features: actual vs best-v2 alternative
        best_v2_of = safe_of(sd["actual_v1"], best_v2, seat_idx, board, vf_cache) or actual_of

        pip_diff  = actual_of.combined_pip_count - best_v2_of.combined_pip_count
        exp_diff  = actual_of.expansion_pip_sum  - best_v2_of.expansion_pip_sum
        port_diff = actual_of.num_ports          - best_v2_of.num_ports

        won = (record.winner_index == record.play_order[seat_idx]
               if seat_idx < len(record.play_order) else False)
        vp  = record.final_vps.get(record.play_order[seat_idx], 0) \
              if seat_idx < len(record.play_order) else 0

        results.append({
            "game_id":        record.game_id,
            "seat":           seat_idx,
            "won":            int(won),
            "final_vp":       vp,
            "total_turns":    record.total_turns,
            # actual opening attributes
            "actual_score":       actual_score,
            "actual_pips":        actual_of.combined_pip_count,
            "actual_resources":   actual_of.unique_resource_count,
            "actual_expansion":   actual_of.expansion_pip_sum,
            "actual_ports":       actual_of.num_ports,
            "actual_adj3":        int(actual_of.v1_arr[19] == 3) + int(actual_of.v2_arr[19] == 3),
            # v2 regret (exact)
            "v2_regret":      v2_regret,
            "v2_rank":        v2_rank,
            "n_v2_alts":      len(v2_valid),
            "best_v2_score":  best_v2_score,
            "pip_diff":       pip_diff,
            "exp_diff":       exp_diff,
            "port_diff":      port_diff,
            # v1 quality proxy
            "v1_pip_rank":    sd["v1_pip_rank"],
            "v1_n_alts":      sd["v1_n_alts"],
            "v1_pip_pct":     sd["v1_pip_rank"] / max(sd["v1_n_alts"], 1),
        })

    return results


COLUMNS = [
    "game_id", "seat", "won", "final_vp", "total_turns",
    "actual_score", "actual_pips", "actual_resources", "actual_expansion",
    "actual_ports", "actual_adj3",
    "v2_regret", "v2_rank", "n_v2_alts", "best_v2_score",
    "pip_diff", "exp_diff", "port_diff",
    "v1_pip_rank", "v1_n_alts", "v1_pip_pct",
]


def run(n_games: int, out_path: Path, tar_path: Path, model_path: Path):
    model = load_model(model_path)
    logger.info("Model loaded  →  %s", model_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=COLUMNS)
    writer.writeheader()

    ok = skip = fail = 0
    t0 = time.time()

    for record in stream_from_tarfile(tar_path, max_games=n_games):
        board = colonist_record_to_board(record)
        if board is None:
            skip += 1
            continue
        try:
            rows = compute_game_regret(record, board, model)
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in COLUMNS})
            if rows:
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            logger.debug("Failed %s: %s", record.game_id, exc)
            fail += 1

        if ok % 500 == 0 and ok > 0:
            elapsed = time.time() - t0
            rate = ok / elapsed
            logger.info("Progress: %d/%d  (%.1f/s  ETA %.0fs)",
                        ok, n_games, rate, (n_games - ok) / rate)
            fout.flush()

    fout.close()
    logger.info("Done: %d games in %.1fs  (%d skipped, %d failed)",
                ok, time.time() - t0, skip, fail)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games",  type=int, default=5000)
    p.add_argument("--out",    default="research/opening_regret/regret_data.csv")
    p.add_argument("--data",   default="games.tar.gz")
    p.add_argument("--model",  default="data/processed/model_logreg.pkl")
    args = p.parse_args()
    run(args.games, Path(args.out), Path(args.data), Path(args.model))
