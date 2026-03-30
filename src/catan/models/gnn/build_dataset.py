"""
Build the PyG graph dataset from games.tar.gz.

Streams all games, converts each (board, v1, v2, seat) → Data object,
and saves to data/gnn/dataset.pt as a list of PyG Data objects.

Also saves the train/val/test split indices (stratified by game_id so
all openings from the same game stay in the same split — no data leakage).

Usage:
    python -m catan.models.gnn.build_dataset [--max-games N] [--out DIR]

Runtime: ~10-15 min for all 43,947 games.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from catan.data.loader import stream_from_tarfile, colonist_record_to_board
from catan.models.gnn.board_to_graph import (
    board_to_node_features,
    opening_to_graph,
    _sorted_vertices,
    _get_edge_index,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build(
    tar_path: Path,
    out_dir: Path,
    max_games: int = 999_999,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset: list[Data] = []
    game_ids: list[str]  = []

    ok = fail = skip = 0
    t0 = time.time()

    for record in stream_from_tarfile(tar_path, max_games=max_games):
        board = colonist_record_to_board(record)
        if board is None:
            skip += 1
            continue

        # Map corner IDs → sorted vertex list
        all_verts = _sorted_vertices(board)

        def corner_to_vert(cid):
            return all_verts[cid] if 0 <= cid < len(all_verts) else None

        # Precompute board node features once per game
        try:
            node_feats = board_to_node_features(board)
            # Warm the edge_index cache (same topology every game)
            _get_edge_index(board)
        except Exception as e:
            logger.debug("board feats failed %s: %s", record.game_id, e)
            fail += 1
            continue

        # Per-player placements
        player_placements: dict = {}
        for p in record.opening_placements:
            player_placements.setdefault(p.player_index, []).append(p)
        for pid in player_placements:
            player_placements[pid].sort(key=lambda p: p.action_seq)

        color_to_seat = {color: seat
                         for seat, color in enumerate(record.play_order[:4])}

        added = 0
        for color, placements in player_placements.items():
            if len(placements) < 2:
                continue
            seat = color_to_seat.get(color)
            if seat is None:
                continue

            v1 = corner_to_vert(placements[0].vertex_index)
            v2 = corner_to_vert(placements[1].vertex_index)
            if v1 is None or v2 is None:
                continue

            won = int(record.winner_index == color)
            rank = record.final_ranks.get(color, 4) if hasattr(record, 'final_ranks') else 4

            try:
                data = opening_to_graph(
                    board=board,
                    v1=v1, v2=v2,
                    seat=seat,
                    label_win=float(won),
                    label_rank=float(rank),
                    node_features_base=node_feats,
                )

                # Store v1/v2 local indices so model can look up their embeddings
                v_idx = {v: i for i, v in enumerate(all_verts)}
                data.v1_local = torch.tensor([v_idx.get(v1, 0)], dtype=torch.long)
                data.v2_local = torch.tensor([v_idx.get(v2, 0)], dtype=torch.long)
                data.game_id  = record.game_id
                data.seat     = torch.tensor([seat], dtype=torch.long)

                dataset.append(data)
                game_ids.append(record.game_id)
                added += 1
            except Exception as e:
                logger.debug("opening failed %s seat%d: %s", record.game_id, seat, e)

        if added > 0:
            ok += 1
        else:
            skip += 1

        if ok % 2000 == 0 and ok > 0:
            elapsed = time.time() - t0
            rate = ok / elapsed
            logger.info("Progress: %d games  %d graphs  %.1f g/s  ETA %.0fs",
                        ok, len(dataset), rate,
                        (max_games - ok) / rate if rate > 0 else 0)

    elapsed = time.time() - t0
    logger.info("Done: %d games → %d graphs in %.0fs  (%d skip %d fail)",
                ok, len(dataset), elapsed, skip, fail)

    # ── save dataset ──────────────────────────────────────────────────────────
    out_data = out_dir / "dataset.pt"
    torch.save(dataset, out_data)
    logger.info("Saved %d graphs → %s", len(dataset), out_data)

    # ── train/val/test split by game_id (no leakage) ─────────────────────────
    import numpy as np
    unique_games = list(dict.fromkeys(game_ids))   # preserve order, deduplicate
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(unique_games))
    n_val  = int(len(unique_games) * 0.10)
    n_test = int(len(unique_games) * 0.10)

    val_games  = set(unique_games[i] for i in perm[:n_val])
    test_games = set(unique_games[i] for i in perm[n_val:n_val+n_test])

    train_idx = [i for i, gid in enumerate(game_ids) if gid not in val_games and gid not in test_games]
    val_idx   = [i for i, gid in enumerate(game_ids) if gid in val_games]
    test_idx  = [i for i, gid in enumerate(game_ids) if gid in test_games]

    split = {"train": train_idx, "val": val_idx, "test": test_idx}
    torch.save(split, out_dir / "split.pt")
    logger.info("Split: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

    return dataset, split


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="games.tar.gz")
    p.add_argument("--out",        default="data/gnn")
    p.add_argument("--max-games",  type=int, default=999_999)
    args = p.parse_args()

    build(Path(args.data), Path(args.out), args.max_games)
