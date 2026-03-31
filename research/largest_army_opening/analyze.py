"""Analyze whether chasing Largest Army early is a viable opening plan."""

from __future__ import annotations

import argparse
import math
import tarfile
from collections import Counter
from pathlib import Path

try:
    import orjson  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    orjson = None
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).parents[2]
OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "summary.json"
FIG_PATH = FIG_DIR / "fig1_largest_army_off_rip.png"

BASELINE_WIN_RATE = 0.25
OFF_RIP_CUTOFF = 40
EARLY_CUTOFF = 56
MID_CUTOFF = 80

BUCKET_ORDER = [
    "Never",
    "Off-rip\n(<=40)",
    "Early\n(41-56)",
    "Mid\n(57-80)",
    "Late\n(81+)",
]
BUCKET_COLORS = {
    "Never": "#a8b3c7",
    "Off-rip\n(<=40)": "#7a4fd3",
    "Early\n(41-56)": "#2e9f9b",
    "Mid\n(57-80)": "#e07a2d",
    "Late\n(81+)": "#b33a3a",
}


def _loads(raw_bytes: bytes) -> dict:
    if orjson is not None:
        return orjson.loads(raw_bytes)
    return json.loads(raw_bytes)


def bucket_for_claim(claim_completed_turns: int | None) -> str:
    if claim_completed_turns is None:
        return "Never"
    if claim_completed_turns <= OFF_RIP_CUTOFF:
        return "Off-rip\n(<=40)"
    if claim_completed_turns <= EARLY_CUTOFF:
        return "Early\n(41-56)"
    if claim_completed_turns <= MID_CUTOFF:
        return "Mid\n(57-80)"
    return "Late\n(81+)"


def ci95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    radius = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total)
    return max(0.0, p - radius), min(1.0, p + radius)


def collect_rows(source: Path, max_games: int | None = None) -> tuple[list[dict], int]:
    rows: list[dict] = []
    processed_games = 0

    with tarfile.open(source, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            extracted = tf.extractfile(member)
            if extracted is None:
                continue

            try:
                raw = _loads(extracted.read())
            except Exception:
                continue

            data = raw.get("data", raw)
            play_order = data.get("playOrder", [])[:4]
            if len(play_order) != 4:
                continue

            end_state = data.get("eventHistory", {}).get("endGameState", {})
            players = end_state.get("players", {})
            winner_color = next(
                (
                    int(color)
                    for color, pdata in players.items()
                    if isinstance(pdata, dict) and pdata.get("winningPlayer")
                ),
                None,
            )
            if winner_color is None:
                continue

            first_claim_by_player: dict[int, int] = {}
            current_completed_turns = 0
            for event in data.get("eventHistory", {}).get("events", []):
                state_change = event.get("stateChange", {})
                current_state = state_change.get("currentState", {})
                if current_state.get("completedTurns") is not None:
                    current_completed_turns = int(current_state["completedTurns"])

                game_log = state_change.get("gameLogState", {})
                if not isinstance(game_log, dict):
                    continue

                for entry in game_log.values():
                    if not isinstance(entry, dict):
                        continue
                    text = entry.get("text", {})
                    if not isinstance(text, dict) or text.get("achievementEnum") != 1:
                        continue

                    player = None
                    event_type = text.get("type")
                    if event_type == 66:
                        player = text.get("playerColor")
                    elif event_type == 68:
                        player = text.get("playerColorNew")

                    if player is not None:
                        first_claim_by_player.setdefault(int(player), current_completed_turns)

            for seat, color in enumerate(play_order):
                claim_turn = first_claim_by_player.get(color)
                rows.append(
                    {
                        "seat": seat,
                        "won": int(color == winner_color),
                        "claim_completed_turns": claim_turn,
                        "bucket": bucket_for_claim(claim_turn),
                    }
                )

            processed_games += 1
            if max_games is not None and processed_games >= max_games:
                break

    return rows, processed_games


def build_summary(rows: list[dict], games: int) -> dict:
    winners = [row for row in rows if row["won"] == 1]
    off_rip_rows = [row for row in rows if row["claim_completed_turns"] is not None and row["claim_completed_turns"] <= OFF_RIP_CUTOFF]

    bucket_stats = {}
    for label in BUCKET_ORDER:
        subset = [row for row in rows if row["bucket"] == label]
        wins = sum(row["won"] for row in subset)
        total = len(subset)
        low, high = ci95(wins, total)
        bucket_stats[label] = {
            "players": total,
            "wins": wins,
            "win_rate": wins / total if total else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        }

    winner_share = {}
    total_winners = len(winners)
    for label in BUCKET_ORDER:
        subset = [row for row in winners if row["bucket"] == label]
        winner_share[label] = {
            "winners": len(subset),
            "share_of_winners": len(subset) / total_winners if total_winners else 0.0,
        }

    seat_stats = {}
    for seat in range(4):
        seat_rows = [row for row in rows if row["seat"] == seat]
        seat_off_rip = [
            row
            for row in seat_rows
            if row["claim_completed_turns"] is not None and row["claim_completed_turns"] <= OFF_RIP_CUTOFF
        ]
        seat_stats[f"S{seat + 1}"] = {
            "baseline_players": len(seat_rows),
            "baseline_win_rate": sum(row["won"] for row in seat_rows) / len(seat_rows) if seat_rows else 0.0,
            "off_rip_players": len(seat_off_rip),
            "off_rip_win_rate": sum(row["won"] for row in seat_off_rip) / len(seat_off_rip) if seat_off_rip else 0.0,
        }

    cumulative = {}
    for cutoff in (40, 48, 56):
        subset = [
            row
            for row in rows
            if row["claim_completed_turns"] is not None and row["claim_completed_turns"] <= cutoff
        ]
        wins = sum(row["won"] for row in subset)
        winners_subset = [
            row
            for row in winners
            if row["claim_completed_turns"] is not None and row["claim_completed_turns"] <= cutoff
        ]
        cumulative[str(cutoff)] = {
            "players": len(subset),
            "wins": wins,
            "win_rate": wins / len(subset) if subset else 0.0,
            "share_of_winners": len(winners_subset) / total_winners if total_winners else 0.0,
        }

    return {
        "games": games,
        "players": len(rows),
        "winners": total_winners,
        "baseline_win_rate": sum(row["won"] for row in rows) / len(rows) if rows else 0.0,
        "ever_claimers": sum(row["claim_completed_turns"] is not None for row in rows),
        "ever_claimer_win_rate": (
            sum(row["won"] for row in rows if row["claim_completed_turns"] is not None)
            / max(sum(row["claim_completed_turns"] is not None for row in rows), 1)
        ),
        "off_rip_cutoff_completed_turns": OFF_RIP_CUTOFF,
        "bucket_stats": bucket_stats,
        "winner_share": winner_share,
        "seat_stats": seat_stats,
        "cumulative_claim_cutoffs": cumulative,
        "verdict": (
            "Largest Army is a viable midgame pivot, but not a default opening plan to force from turn 1."
        ),
    }


def plot_summary(summary: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    bg = "#f7f4f0"
    fig = plt.figure(figsize=(15, 5.5))
    fig.patch.set_facecolor(bg)

    ax1 = fig.add_axes([0.05, 0.16, 0.39, 0.72])
    ax2 = fig.add_axes([0.50, 0.16, 0.20, 0.72])
    ax3 = fig.add_axes([0.75, 0.16, 0.20, 0.72])
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(bg)

    labels = BUCKET_ORDER
    rates = np.array([summary["bucket_stats"][label]["win_rate"] * 100.0 for label in labels])
    lows = np.array([summary["bucket_stats"][label]["ci95_low"] * 100.0 for label in labels])
    highs = np.array([summary["bucket_stats"][label]["ci95_high"] * 100.0 for label in labels])
    errors = np.vstack([rates - lows, highs - rates])
    counts = [summary["bucket_stats"][label]["players"] for label in labels]
    colors = [BUCKET_COLORS[label] for label in labels]

    bars = ax1.bar(labels, rates, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax1.errorbar(np.arange(len(labels)), rates, yerr=errors, fmt="none", ecolor="#333333", capsize=4, linewidth=1.1, zorder=4)
    ax1.axhline(BASELINE_WIN_RATE * 100.0, color="#444444", linestyle="--", linewidth=1.3, zorder=4)
    ax1.text(
        len(labels) - 0.25,
        BASELINE_WIN_RATE * 100.0 + 0.8,
        "25% baseline",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    for bar, rate, count in zip(bars, rates, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            1.0,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#ffffff" if rate > 18.0 else "#333333",
        )
    ax1.set_ylabel("Win rate (%)", fontsize=11)
    ax1.set_title("Win rate by first Largest Army claim timing", fontsize=12.5, fontweight="bold")
    ax1.set_ylim(0, 72)
    ax1.yaxis.grid(True, color="white", linewidth=1.3, zorder=0)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax1.tick_params(length=0)

    winner_shares = np.array([summary["winner_share"][label]["share_of_winners"] * 100.0 for label in labels])
    winner_bars = ax2.bar(labels, winner_shares, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for bar, share in zip(winner_bars, winner_shares):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_title("Share of winners", fontsize=12.5, fontweight="bold")
    ax2.set_ylim(0, 62)
    ax2.yaxis.grid(True, color="white", linewidth=1.3, zorder=0)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax2.tick_params(length=0, labelleft=False)

    seat_labels = list(summary["seat_stats"].keys())
    baseline = np.array([summary["seat_stats"][label]["baseline_win_rate"] * 100.0 for label in seat_labels])
    off_rip = np.array([summary["seat_stats"][label]["off_rip_win_rate"] * 100.0 for label in seat_labels])
    x = np.arange(len(seat_labels))
    width = 0.36
    ax3.bar(x - width / 2.0, baseline, width, color="#b6beca", edgecolor="white", linewidth=1.1, label="Seat baseline", zorder=3)
    ax3.bar(x + width / 2.0, off_rip, width, color=BUCKET_COLORS["Off-rip\n(<=40)"], edgecolor="white", linewidth=1.1, label="Off-rip LA", zorder=3)
    for xi, base, off in zip(x, baseline, off_rip):
        ax3.text(xi - width / 2.0, base + 0.8, f"{base:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax3.text(xi + width / 2.0, off + 0.8, f"{off:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(seat_labels)
    ax3.set_ylim(0, 40)
    ax3.set_title("Off-rip LA by seat", fontsize=12.5, fontweight="bold")
    ax3.yaxis.grid(True, color="white", linewidth=1.3, zorder=0)
    ax3.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax3.tick_params(length=0, labelleft=False)
    ax3.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(
        "Largest Army Is a Viable Pivot, Not a Default Off-Rip Plan",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    footer = (
        f"Dataset: {summary['games']:,} four-player Colonist games ({summary['players']:,} player records). "
        f"\"Off-rip\" means first Largest Army claim by completed turn <= {summary['off_rip_cutoff_completed_turns']} "
        f"(about the first 10 table rounds)."
    )
    fig.text(
        0.05,
        0.05,
        footer,
        fontsize=9.2,
        color="#555555",
    )
    fig.savefig(FIG_PATH, dpi=170, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def save_summary(summary: dict) -> None:
    if orjson is not None:
        SUMMARY_PATH.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    else:
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def load_summary() -> dict:
    if orjson is not None:
        return orjson.loads(SUMMARY_PATH.read_bytes())
    return json.loads(SUMMARY_PATH.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "games.tar.gz",
        help="Path to the Colonist archive",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional limit for faster debugging",
    )
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Skip the archive scan and regenerate outputs from summary.json",
    )
    args = parser.parse_args()

    if args.from_summary:
        summary = load_summary()
    else:
        rows, games = collect_rows(args.source, max_games=args.max_games)
        summary = build_summary(rows, games)
        save_summary(summary)
    plot_summary(summary)

    off_rip = summary["bucket_stats"]["Off-rip\n(<=40)"]
    print(f"games={summary['games']:,} players={summary['players']:,}")
    print(f"off-rip LA: n={off_rip['players']:,} win_rate={off_rip['win_rate']:.3%}")
    print(f"baseline: {summary['baseline_win_rate']:.3%}")
    print(f"summary -> {SUMMARY_PATH}")
    print(f"figure  -> {FIG_PATH}")


if __name__ == "__main__":
    main()
