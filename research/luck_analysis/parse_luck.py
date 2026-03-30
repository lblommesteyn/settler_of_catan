"""
parse_luck.py — Extract per-player dice luck scores from games.tar.gz.

Tracks the FULL game state: settlements AND cities built throughout the game.

For each dice roll at time t:
  expected_pips_this_roll(P) = sum over (settlement/city of P, adjacent hex) of
                                  PIP_VALUES[hex.number] / 36  x  multiplier
                               where multiplier = 1 (settlement) or 2 (city)

  actual_pips_this_roll(P)   = sum over (settlement/city of P adjacent to hex
                                  whose number == dice_sum) of
                                  PIP_VALUES[dice_sum]  x  multiplier

Over the whole game:
  luck_ratio = total_actual_pips / total_expected_pips
  luck_diff  = (total_actual_pips - total_expected_pips) / n_rolls

This correctly accounts for:
  - Settlements built during the game (not just opening v1/v2)
  - Cities (2x multiplier)
  - The timing of builds (a city built on turn 30 only counts from turn 30 on)

Simplification: robber blocking is not modelled.

Output CSV: game_id, seat, won, n_rolls,
            total_expected_pips, total_actual_pips,
            luck_ratio, luck_diff
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "src"))

from catan.data.loader import (
    _parse_game_dict,
    colonist_record_to_board,
    _corner_id_to_vertex,
)
from catan.board.board import PIP_VALUES

TAR_PATH = REPO / "games.tar.gz"
OUT_CSV  = Path(__file__).parent / "luck_data.csv"


# ---------------------------------------------------------------------------
# Full-game luck computation
# ---------------------------------------------------------------------------

def compute_luck_rows(game_id: str, raw: dict) -> list[dict]:
    record = _parse_game_dict(raw, game_id=game_id)
    if record is None:
        return []

    board = colonist_record_to_board(record)
    if board is None:
        return []

    # Pre-cache: corner_id → list of (pip_value, hex_number) for adjacent hexes
    n_verts = len(sorted(board.all_vertices()))
    corner_hex_cache: dict[int, list[tuple[int, int]]] = {}
    for corner_id in range(n_verts):
        v_key = _corner_id_to_vertex(corner_id, board)
        if v_key is None:
            corner_hex_cache[corner_id] = []
            continue
        entries = []
        for htile in board.vertex_hexes.get(v_key, []):
            if htile.number is not None:
                entries.append((PIP_VALUES.get(htile.number, 0), htile.number))
        corner_hex_cache[corner_id] = entries

    data = raw.get("data", raw)
    play_order: list[int] = data.get("playOrder", [])[:4]
    events: list[dict]    = data.get("eventHistory", {}).get("events", [])

    # Running board state: corner_id → (owner_color, multiplier)
    # multiplier = 1 for settlement, 2 for city
    corner_state: dict[int, tuple[int, int]] = {}

    # Per-player accumulators keyed by player color
    total_actual:   dict[int, float] = {}
    total_expected: dict[int, float] = {}
    n_rolls: dict[int, int]          = {}

    for color in play_order:
        total_actual[color]   = 0.0
        total_expected[color] = 0.0
        n_rolls[color]        = 0

    for event in events:
        sc = event.get("stateChange", {})

        # --- Update corner state (settlements / upgrades to city) ---
        ms = sc.get("mapState", {})
        for corner_str, info in ms.get("tileCornerStates", {}).items():
            try:
                corner_id = int(corner_str)
            except (ValueError, TypeError):
                continue
            owner = info.get("owner")
            btype = info.get("buildingType", 0)
            if owner is not None and btype in (1, 2):
                corner_state[corner_id] = (int(owner), btype)

        # --- Process dice roll ---
        ds = sc.get("diceState", {})
        if not (isinstance(ds, dict) and ds.get("diceThrown")):
            continue

        d1 = ds.get("dice1") or 0
        d2 = ds.get("dice2") or 0
        dice_sum = int(d1) + int(d2)
        if not (2 <= dice_sum <= 12):
            continue

        # Attribute this roll to each player who has pieces on the board
        for color in play_order:
            n_rolls[color] += 1

        # Iterate over all built corners and attribute expected + actual resources.
        # Both are in "resource unit" terms: 1 per settlement hit, 2 per city hit.
        # expected += P(this hex is rolled) x mult  = pip_val/36 x mult
        # actual   += mult  (when the hex number matches the roll)
        for corner_id, (owner_color, mult) in corner_state.items():
            if owner_color not in total_actual:
                continue  # color not in play_order (shouldn't happen)
            hex_entries = corner_hex_cache.get(corner_id, [])
            for pip_val, hex_num in hex_entries:
                total_expected[owner_color] += pip_val / 36.0 * mult
                if hex_num == dice_sum:
                    total_actual[owner_color] += mult

    # --- Build output rows ---
    rows = []
    for seat, color in enumerate(play_order):
        nr = n_rolls.get(color, 0)
        if nr < 5:
            continue
        exp = total_expected.get(color, 0.0)
        act = total_actual.get(color, 0.0)
        if exp <= 0:
            continue

        luck_ratio = act / exp          # >1 = luckier than expected
        luck_diff  = (act - exp) / nr  # extra resources per roll vs. expectation

        won = int(color == record.winner_index)

        rows.append(dict(
            game_id             = game_id,
            seat                = seat,
            won                 = won,
            n_rolls             = nr,
            total_expected_pips = round(exp, 4),
            total_actual_pips   = round(act, 4),
            luck_ratio          = round(luck_ratio, 6),
            luck_diff           = round(luck_diff, 6),
        ))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "game_id", "seat", "won", "n_rolls",
    "total_expected_pips", "total_actual_pips",
    "luck_ratio", "luck_diff",
]


def main() -> None:
    log.info("Streaming %s (full-game luck metric)", TAR_PATH)
    n_games = n_rows = n_skip = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()

        with tarfile.open(TAR_PATH, "r:gz") as tf:
            for member in tf:
                if not member.name.endswith(".json") or not member.isfile():
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    raw = json.loads(f.read().decode("utf-8"))
                except Exception:
                    continue

                game_id = Path(member.name).stem
                rows = compute_luck_rows(game_id, raw)

                if not rows:
                    n_skip += 1
                else:
                    writer.writerows(rows)
                    n_rows  += len(rows)
                    n_games += 1

                if n_games % 5000 == 0 and n_games > 0:
                    log.info("  %d games, %d rows (skipped: %d)", n_games, n_rows, n_skip)

    log.info("Done: %d games -> %d rows  (skipped: %d)", n_games, n_rows, n_skip)
    log.info("Saved -> %s", OUT_CSV)


if __name__ == "__main__":
    main()
