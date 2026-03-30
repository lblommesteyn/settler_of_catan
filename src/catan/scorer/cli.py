"""
Interactive CLI for the Catan Opening Intelligence scorer.

Usage:
  python -m catan.scorer.cli                         # random board, interactive
  python -m catan.scorer.cli --seat 2                # seat 2 on random board
  python -m catan.scorer.cli --rank-all              # rank all openings
  python -m catan.scorer.cli --model gbc             # use gradient boosting model
  python -m catan.scorer.cli --data path/to/games/   # load Colonist dataset + train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..board.board import CatanBoard, Resource, PortType
from ..features.opening_features import (
    compute_opening_features,
    compute_all_vertex_features,
    OpeningFeatures,
    FEATURE_NAMES,
)
from ..models.heuristic import PipCountHeuristic, WeightedHeuristic
from ..models.base_model import OpeningModel
from .explainer import explain_opening, OpeningExplanation


try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print(msg: str, style: str = "") -> None:
    if HAS_RICH and console:
        console.print(msg, style=style or None)
    else:
        print(msg)


def display_board_summary(board: CatanBoard) -> None:
    """Print a text summary of the board."""
    _print("\n[bold]Board Summary[/bold]" if HAS_RICH else "\nBoard Summary")
    resource_counts = {}
    pip_counts = {}
    for tile in board.tiles.values():
        r = tile.resource.value
        resource_counts[r] = resource_counts.get(r, 0) + 1
        if tile.pips > 0:
            pip_counts[r] = pip_counts.get(r, 0) + tile.pips

    for r, count in sorted(resource_counts.items()):
        pips = pip_counts.get(r, 0)
        _print(f"  {r:8s}: {count} tiles, {pips} total pips")

    n_ports = len(board.ports)
    _print(f"  Ports: {n_ports}")


def display_vertex_table(
    board: CatanBoard,
    top_k: int = 20,
    seat: int = 0,
) -> list[tuple[float, float]]:
    """Print top vertices by pip count and return sorted list."""
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()

    rows = []
    for i, vk in enumerate(sorted(legal, key=lambda v: -board.pip_count(v))):
        vf = vf_cache[vk]
        resources = "/".join(
            r.value[:2].upper()
            for r in [Resource.WOOD, Resource.BRICK, Resource.SHEEP,
                      Resource.WHEAT, Resource.ORE]
            if getattr(vf, f"has_{r.value}")
        )
        port_str = f"{vf.port_type.value}" if vf.has_port and vf.port_type else "-"
        rows.append((i + 1, vk, vf.total_pips, resources, port_str))

    rows_display = rows[:top_k]

    if HAS_RICH:
        table = Table(title=f"Top {top_k} Vertices (by pips)")
        table.add_column("#", style="dim")
        table.add_column("Index", style="cyan")
        table.add_column("Position (x,y)", style="white")
        table.add_column("Pips", style="bold green")
        table.add_column("Resources")
        table.add_column("Port")
        for rank, vk, pips, res, port in rows_display:
            table.add_row(
                str(rank), str(rank),
                f"({vk[0]:.3f}, {vk[1]:.3f})",
                str(pips), res, port,
            )
        console.print(table)
    else:
        print(f"\n{'#':4} {'Position':20} {'Pips':6} {'Resources':20} {'Port':12}")
        print("-" * 70)
        for rank, vk, pips, res, port in rows_display:
            print(f"{rank:4} ({vk[0]:7.3f},{vk[1]:7.3f}) {pips:6} {res:20} {port:12}")

    return [r[1] for r in rows]


def display_explanation(exp: OpeningExplanation) -> None:
    """Pretty-print an opening explanation."""
    pct = f"{exp.percentile:.0f}th"
    prob_pct = f"{exp.win_probability * 100:.1f}%"

    if HAS_RICH:
        title = f"Opening Analysis: [bold cyan]{exp.archetype}[/bold cyan]"
        lines = [
            f"[bold]Win Probability:[/bold] {prob_pct}  ({pct} percentile)",
            "",
        ]
        if exp.strengths:
            lines.append("[bold green]Strengths:[/bold green]")
            for s in exp.strengths:
                lines.append(f"  [green]+ {s}[/green]")
        if exp.weaknesses:
            lines.append("")
            lines.append("[bold red]Weaknesses:[/bold red]")
            for w in exp.weaknesses:
                lines.append(f"  [red]- {w}[/red]")
        if exp.counterfactuals:
            lines.append("")
            lines.append("[bold yellow]Better Alternatives:[/bold yellow]")
            for i, cf in enumerate(exp.counterfactuals, 1):
                delta_str = f"+{cf.delta * 100:.1f}%"
                lines.append(
                    f"  [yellow]{i}.[/yellow] "
                    f"({cf.v1[0]:.2f},{cf.v1[1]:.2f}) + ({cf.v2[0]:.2f},{cf.v2[1]:.2f})"
                    f"  [bold yellow]{delta_str}[/bold yellow]"
                )
                lines.append(f"     Gain: {cf.gain_desc} | Cost: {cf.cost_desc}")
        console.print(Panel("\n".join(lines), title=title, expand=False))
    else:
        print(f"\n{'='*60}")
        print(f"Opening: {exp.archetype}")
        print(f"Win Probability: {prob_pct}  ({pct} percentile)")
        print()
        if exp.strengths:
            print("Strengths:")
            for s in exp.strengths:
                print(f"  + {s}")
        if exp.weaknesses:
            print("\nWeaknesses:")
            for w in exp.weaknesses:
                print(f"  - {w}")
        if exp.counterfactuals:
            print("\nBetter Alternatives:")
            for i, cf in enumerate(exp.counterfactuals, 1):
                print(
                    f"  {i}. ({cf.v1[0]:.2f},{cf.v1[1]:.2f}) + ({cf.v2[0]:.2f},{cf.v2[1]:.2f})"
                    f"  +{cf.delta*100:.1f}%"
                )
                print(f"     Gain: {cf.gain_desc}")


def display_ranked_openings(
    ranked: list[tuple[OpeningFeatures, float]],
    top_k: int = 10,
) -> None:
    """Print top-k ranked openings in a table."""
    if HAS_RICH:
        table = Table(title=f"Top {top_k} Opening Pairs")
        table.add_column("#", style="dim")
        table.add_column("v1 (x,y)", style="cyan")
        table.add_column("v2 (x,y)", style="cyan")
        table.add_column("Win Prob", style="bold green")
        table.add_column("Pips")
        table.add_column("Resources")
        table.add_column("Ports")
        table.add_column("Archetype")
        for i, (f, score) in enumerate(ranked[:top_k], 1):
            from ..features.opening_features import identify_archetype
            table.add_row(
                str(i),
                f"({f.v1[0]:.2f},{f.v1[1]:.2f})",
                f"({f.v2[0]:.2f},{f.v2[1]:.2f})",
                f"{score * 100:.1f}%",
                str(f.combined_pip_count),
                str(f.unique_resource_count),
                str(f.num_ports),
                identify_archetype(f),
            )
        console.print(table)
    else:
        print(f"\n{'#':3} {'v1':20} {'v2':20} {'WinProb':8} {'Pips':5} {'Res':4} {'Ports':5}")
        print("-" * 75)
        for i, (f, score) in enumerate(ranked[:top_k], 1):
            print(
                f"{i:3} ({f.v1[0]:6.2f},{f.v1[1]:6.2f}) "
                f"({f.v2[0]:6.2f},{f.v2[1]:6.2f}) "
                f"{score*100:7.1f}% {f.combined_pip_count:5} "
                f"{f.unique_resource_count:4} {f.num_ports:5}"
            )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_type: str, model_path: Optional[Path] = None) -> OpeningModel:
    """Load a model by type name."""
    if model_type == "pip":
        return PipCountHeuristic()
    if model_type == "weighted":
        return WeightedHeuristic()

    # ML models require a saved file or training data
    if model_path and model_path.exists():
        if model_type == "logreg":
            from ..models.ml_model import LogisticOpeningModel
            return LogisticOpeningModel.load(model_path)
        if model_type in ("gbc", "lgbm"):
            from ..models.ml_model import GradientBoostingOpeningModel
            return GradientBoostingOpeningModel.load(model_path)

    # Fall back to weighted heuristic
    _print(
        f"[yellow]No trained {model_type} model found — using weighted heuristic.[/yellow]"
        if HAS_RICH else f"No trained {model_type} model found — using weighted heuristic."
    )
    return WeightedHeuristic()


# ---------------------------------------------------------------------------
# Interactive vertex selection
# ---------------------------------------------------------------------------

def select_vertex_interactive(
    board: CatanBoard,
    prompt: str,
    exclude: Optional[set] = None,
) -> tuple[float, float]:
    """Prompt the user to pick a vertex by index from the sorted list."""
    vf_cache = compute_all_vertex_features(board)
    legal = sorted(board.legal_starting_vertices(), key=lambda v: -board.pip_count(v))
    if exclude:
        legal = [v for v in legal if v not in (exclude or set())]

    for i, vk in enumerate(legal, 1):
        vf = vf_cache[vk]
        res = "/".join(
            r.value[:2].upper()
            for r in [Resource.WOOD, Resource.BRICK, Resource.SHEEP,
                      Resource.WHEAT, Resource.ORE]
            if getattr(vf, f"has_{r.value}")
        )
        port = f" [{vf.port_type.value}]" if vf.has_port and vf.port_type else ""
        print(f"  {i:3}. pips={vf.total_pips:2}  res={res:12}  pos=({vk[0]:6.2f},{vk[1]:6.2f}){port}")

    while True:
        raw = input(f"\n{prompt} (1-{len(legal)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(legal):
                return legal[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(legal)}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catan Opening Intelligence — score and explain opening placements"
    )
    parser.add_argument(
        "--board", default="random",
        help="'random' or path to a Colonist.io JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for board generation",
    )
    parser.add_argument(
        "--seat", type=int, default=0, choices=[0, 1, 2, 3],
        help="Seat position (0=first, 3=last)",
    )
    parser.add_argument(
        "--v1", default=None,
        help="First settlement position as 'x,y'",
    )
    parser.add_argument(
        "--v2", default=None,
        help="Second settlement position as 'x,y'",
    )
    parser.add_argument(
        "--model", default="weighted",
        choices=["pip", "weighted", "logreg", "gbc"],
        help="Scoring model",
    )
    parser.add_argument(
        "--model-path", default=None, type=Path,
        help="Path to a saved ML model file",
    )
    parser.add_argument(
        "--rank-all", action="store_true",
        help="Rank all legal opening pairs (may be slow)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top openings to display",
    )
    parser.add_argument(
        "--data", default=None, type=Path,
        help="Path to Colonist.io data directory (for inspect/train mode)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect sample game files in --data directory",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Evaluate opening quality via Monte Carlo simulation",
    )
    parser.add_argument(
        "--n-sims", type=int, default=50,
        help="Number of simulations for --simulate mode",
    )
    args = parser.parse_args()

    # --- Inspect mode ---
    if args.inspect and args.data:
        from ..data.loader import inspect_sample_files
        inspect_sample_files(args.data, n=3)
        return

    # --- Load board ---
    if args.board == "random":
        board = CatanBoard.random(seed=args.seed)
        _print(
            "\n[bold]Generated random Catan board[/bold]" if HAS_RICH
            else "\nGenerated random Catan board"
        )
    else:
        from ..data.loader import parse_game_file, colonist_record_to_board
        record = parse_game_file(Path(args.board))
        if record is None:
            print(f"Failed to parse board from {args.board}")
            sys.exit(1)
        board = colonist_record_to_board(record)
        if board is None:
            print("Failed to map Colonist board")
            sys.exit(1)

    display_board_summary(board)
    model = load_model(args.model, args.model_path)
    vf_cache = compute_all_vertex_features(board)

    # --- Rank all openings ---
    if args.rank_all:
        _print("\n[bold]Ranking all legal opening pairs...[/bold]" if HAS_RICH else "\nRanking all opening pairs...")
        legal = board.legal_starting_vertices()
        all_features = []
        import itertools
        for v1 in legal:
            for v2 in board.legal_second_vertices(v1):
                try:
                    f = compute_opening_features(v1, v2, args.seat, board, vf_cache)
                    all_features.append(f)
                except Exception:
                    pass
        ranked = model.rank_openings(all_features)
        _print(f"Evaluated {len(ranked)} opening pairs.")
        display_ranked_openings(ranked, top_k=args.top_k)
        return

    # --- Specific or interactive opening ---
    if args.v1 and args.v2:
        try:
            x1, y1 = map(float, args.v1.split(","))
            x2, y2 = map(float, args.v2.split(","))
            # Snap to nearest board vertex
            all_verts = board.all_vertices()
            v1 = min(all_verts, key=lambda v: (v[0] - x1) ** 2 + (v[1] - y1) ** 2)
            v2 = min(all_verts, key=lambda v: (v[0] - x2) ** 2 + (v[1] - y2) ** 2)
        except ValueError:
            print("Invalid vertex format. Use '--v1 x,y'")
            sys.exit(1)
    else:
        # Interactive selection
        _print(
            "\n[bold]Select your opening settlements:[/bold]" if HAS_RICH
            else "\nSelect your opening settlements:"
        )
        display_vertex_table(board, top_k=30, seat=args.seat)
        v1 = select_vertex_interactive(board, "Select first settlement")
        _print(f"  → First settlement: ({v1[0]:.3f}, {v1[1]:.3f})")

        legal_second = board.legal_second_vertices(v1)
        _print(f"\nLegal second settlements ({len(legal_second)} options):")
        v2 = select_vertex_interactive(board, "Select second settlement", exclude={v1})
        _print(f"  → Second settlement: ({v2[0]:.3f}, {v2[1]:.3f})")

    # --- Compute and display features ---
    try:
        features = compute_opening_features(v1, v2, args.seat, board, vf_cache)
    except Exception as exc:
        print(f"Feature computation failed: {exc}")
        sys.exit(1)

    # --- Simulation mode ---
    if args.simulate:
        from ..simulation.simulator import run_opening_evaluation
        _print(f"\n[bold]Running {args.n_sims} simulations...[/bold]" if HAS_RICH
               else f"\nRunning {args.n_sims} simulations...")
        eval_res = run_opening_evaluation(
            board, 0, v1, v2, args.seat, n_simulations=args.n_sims
        )
        _print(f"  Simulated win rate: {eval_res.win_rate * 100:.1f}%")
        _print(f"  Top-2 rate: {eval_res.top2_rate * 100:.1f}%")
        _print(f"  Avg final VP: {eval_res.avg_final_vp:.1f}")

    # --- Explain ---
    explanation = explain_opening(features, model, board)
    display_explanation(explanation)


if __name__ == "__main__":
    main()
