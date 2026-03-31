"""CLI for the Colonist CV assistant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ..board.board import CatanBoard, PortType, Resource
from ..board.hex_grid import AXIAL_POSITIONS
from ..features.opening_features import compute_all_vertex_features, compute_opening_features
from ..full_solver.state import DevCardType, TurnPhase, empty_hand, playable_resources
from ..models.heuristic import WeightedHeuristic
from ..scorer.explainer import explain_opening
from .advisor import HeuristicActionAdvisor
from .bootstrap import auto_bootstrap_board
from .context_ocr import read_screen_context
from .detector import DetectionError
from .geometry import BoardCalibration
from .opening_live import OpeningLiveRunner, analyze_opening_screen, suggestion_target_text
from .runtime import LiveAdvisorRunner, ScreenRegion, format_strategy_lines, grab_screen, save_calibration_interactive
from .schema import PlayerColor, PrivateObservation, PublicStructures, VisionFrameObservation
from .tracker import build_state_from_observation


def _default_geometry_board() -> CatanBoard:
    """Use any deterministic standard board when only geometry is needed."""
    return CatanBoard.random(seed=0)


def _write_board_template(path: Path) -> None:
    payload = {
        "_instructions": [
            "Fill resources using: wood, brick, sheep, wheat, ore, desert",
            "Fill numbers with the 18 non-desert tokens in AXIAL_POSITIONS order",
            "Fill ports using: 3:1, 2:1_wood, 2:1_brick, 2:1_sheep, 2:1_wheat, 2:1_ore",
            "AXIAL_POSITIONS order is included below for reference",
        ],
        "_axial_positions": AXIAL_POSITIONS,
        "resources": [None] * len(AXIAL_POSITIONS),
        "numbers": [None] * 18,
        "ports": [None] * 9,
    }
    path.write_text(json.dumps(payload, indent=2))


def _validate_board_payload(payload: dict, path: Path) -> None:
    required_keys = ("resources", "numbers", "ports")
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"Board file {path} is missing '{key}'. Run 'python -m catan.colonist_cv.cli init-board --output {path}'.")

    if len(payload["resources"]) != len(AXIAL_POSITIONS):
        raise ValueError(f"Board file {path} must have {len(AXIAL_POSITIONS)} resources.")
    if len(payload["numbers"]) != 18:
        raise ValueError(f"Board file {path} must have 18 number tokens.")
    if len(payload["ports"]) != 9:
        raise ValueError(f"Board file {path} must have 9 ports.")

    if any(item in (None, "") for item in payload["resources"]):
        raise ValueError(f"Board file {path} still has unfilled resource entries.")
    if any(item in (None, "") for item in payload["numbers"]):
        raise ValueError(f"Board file {path} still has unfilled number entries.")
    if any(item in (None, "") for item in payload["ports"]):
        raise ValueError(f"Board file {path} still has unfilled port entries.")


def _load_board(path: Path) -> CatanBoard:
    if not path.exists():
        raise FileNotFoundError(
            f"Board file {path} does not exist. Run 'python -m catan.colonist_cv.cli init-board --output {path}'."
        )
    payload = json.loads(path.read_text())
    _validate_board_payload(payload, path)
    resources = [Resource(item) for item in payload["resources"]]
    numbers = [int(item) for item in payload["numbers"]]
    ports = [PortType(item) for item in payload["ports"]]
    return CatanBoard.from_tiles(AXIAL_POSITIONS, resources, numbers, ports)


def _load_observation(path: Path, board: CatanBoard) -> VisionFrameObservation:
    payload = json.loads(path.read_text())
    public_players = []
    for player in payload["public_players"]:
        public_players.append(
            PublicStructures(
                player_id=int(player["player_id"]),
                color=PlayerColor.from_value(player["color"]) if player.get("color") else None,
                settlements=frozenset(tuple(vertex) for vertex in player.get("settlements", [])),
                cities=frozenset(tuple(vertex) for vertex in player.get("cities", [])),
                roads=frozenset(frozenset(tuple(vertex) for vertex in edge) for edge in player.get("roads", [])),
                visible_vp=player.get("visible_vp"),
                played_knights=int(player.get("played_knights", 0)),
                dev_cards_bought=int(player.get("dev_cards_bought", 0)),
            )
        )

    private_pov = None
    if payload.get("private_pov") is not None:
        private = payload["private_pov"]
        resources = empty_hand()
        for name, count in private.get("resources", {}).items():
            resources[Resource(name)] = int(count)
        mature = {DevCardType(card_name): int(count) for card_name, count in private.get("dev_cards_in_hand", {}).items()}
        fresh = {DevCardType(card_name): int(count) for card_name, count in private.get("new_dev_cards_in_hand", {}).items()}
        private_pov = PrivateObservation(
            player_id=int(private["player_id"]),
            resources=resources,
            dev_cards_in_hand=mature,
            new_dev_cards_in_hand=fresh,
            hidden_vp_cards=int(private.get("hidden_vp_cards", 0)),
        )

    return VisionFrameObservation(
        board=board,
        robber_hex=tuple(payload["robber_hex"]),
        public_players=tuple(public_players),
        current_player=int(payload["current_player"]),
        phase=TurnPhase(payload["phase"]),
        private_pov=private_pov,
        turn_number=int(payload.get("turn_number", 0)),
        setup_step=int(payload.get("setup_step", 0)),
        pending_setup_vertex=tuple(payload["pending_setup_vertex"]) if payload.get("pending_setup_vertex") else None,
        pending_discarders=tuple(int(player_id) for player_id in payload.get("pending_discarders", [])),
        free_roads_remaining=int(payload.get("free_roads_remaining", 0)),
        dev_card_played_this_turn=bool(payload.get("dev_card_played_this_turn", False)),
        dice_rolled_this_turn=bool(payload.get("dice_rolled_this_turn", False)),
        last_roll=int(payload["last_roll"]) if payload.get("last_roll") is not None else None,
        winner_id=int(payload["winner_id"]) if payload.get("winner_id") is not None else None,
    )


def _load_calibration(path: Path, board: CatanBoard) -> BoardCalibration:
    payload = json.loads(path.read_text())
    return BoardCalibration.from_serialized(board, payload)


def _load_capture_region(path: Path) -> ScreenRegion | None:
    payload = json.loads(path.read_text())
    bbox = payload.get("capture_bbox")
    if bbox is None:
        return None
    return ScreenRegion(
        left=int(bbox["left"]),
        top=int(bbox["top"]),
        right=int(bbox["right"]),
        bottom=int(bbox["bottom"]),
    )


def _optional_player_color(raw: str | None) -> PlayerColor | None:
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized == "auto":
        return None
    return PlayerColor.from_value(normalized)


def _cmd_advise(args: argparse.Namespace) -> int:
    board = _load_board(Path(args.board))
    _ = _load_calibration(Path(args.calibration), board)
    observation = _load_observation(Path(args.observation), board)
    state = build_state_from_observation(observation)
    advisor = HeuristicActionAdvisor()
    for line in format_strategy_lines(advisor.strategy_plan(state)):
        print(line)
    for rank, advice in enumerate(advisor.suggest(state, top_k=args.top_k), start=1):
        print(f"{rank}. score={advice.score:.2f} action={advice.action.action_type.value}")
        print(f"   {advice.summary}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    board = _load_board(Path(args.board)) if args.board else _default_geometry_board()
    region = ScreenRegion.parse(args.bbox) if args.bbox else None
    frame = grab_screen(region)
    save_calibration_interactive(frame, board, Path(args.output), capture_region=region)
    print(f"Saved calibration to {args.output}")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    board = _load_board(Path(args.board))
    calibration_path = Path(args.calibration)
    calibration = _load_calibration(calibration_path, board)
    region = ScreenRegion.parse(args.bbox) if args.bbox else _load_capture_region(calibration_path)
    context_path = Path(args.context) if args.context else None
    if context_path is not None and not context_path.exists():
        raise FileNotFoundError(f"Missing context file {args.context}.")

    color_order = [PlayerColor.from_value(part.strip()) for part in args.color_order.split(",")]
    if len(color_order) != 4:
        raise ValueError("--color-order must contain exactly 4 comma-separated colors.")
    color_to_player = {color: player_id for player_id, color in enumerate(color_order)}

    runner = LiveAdvisorRunner(
        board=board,
        calibration=calibration,
        advisor=HeuristicActionAdvisor(),
        color_to_player=color_to_player,
    )
    runner.run(
        region=region,
        context_path=context_path,
        my_color=_optional_player_color(args.my_color),
        top_k=args.top_k,
        interval_s=args.interval,
        once=args.once,
        slow_loop_ms=args.slow_loop_ms,
        stale_seconds=args.stale_seconds,
    )
    return 0


def _cmd_context_screen(args: argparse.Namespace) -> int:
    region = ScreenRegion.parse(args.bbox) if args.bbox else None
    frame = grab_screen(region)
    color_order = [PlayerColor.from_value(part.strip()) for part in args.color_order.split(",")]
    if len(color_order) != 4:
        raise ValueError("--color-order must contain exactly 4 comma-separated colors.")
    color_to_player = {color: player_id for player_id, color in enumerate(color_order)}
    my_color = _optional_player_color(args.my_color)
    player_id_hint = color_to_player[my_color] if my_color is not None else None
    detected = read_screen_context(frame, my_color=my_color, color_to_player=color_to_player, player_id_hint=player_id_hint)

    print(f"my_color={detected.my_color.value if detected.my_color is not None else 'unknown'}")
    print(f"prompt={detected.prompt_text or '<none>'}")
    print(f"current_player={detected.current_player if detected.current_player is not None else 'unknown'}")
    print(f"phase={detected.phase.value if detected.phase is not None else 'unknown'}")
    print(f"dice_rolled={detected.dice_rolled_this_turn if detected.dice_rolled_this_turn is not None else 'unknown'}")
    if detected.private_pov is None:
        print("hand=<unavailable>")
    else:
        hand = detected.private_pov.canonical().resources
        print("hand=" + ", ".join(f"{resource.value}:{hand[resource]}" for resource in playable_resources()))
    return 0


def _cmd_bootstrap_screen(args: argparse.Namespace) -> int:
    region = ScreenRegion.parse(args.bbox) if args.bbox else None
    frame = grab_screen(region)
    bootstrap = auto_bootstrap_board(frame)
    payload = {
        "desert_hex": list(bootstrap.desert_hex),
        "resources": [bootstrap.resources_by_hex[hex_coord].value for hex_coord in AXIAL_POSITIONS],
        "numbers": [bootstrap.numbers_by_hex[hex_coord] for hex_coord in AXIAL_POSITIONS if hex_coord != bootstrap.desert_hex],
        "ports": ["3:1"] * 9,
        "_note": "Ports are conservative generic placeholders in the automatic screen bootstrap.",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Saved auto-bootstrapped board to {args.output}")
    return 0


def _cmd_opening_screen(args: argparse.Namespace) -> int:
    region = ScreenRegion.parse(args.bbox) if args.bbox else None
    frame = grab_screen(region)
    analysis = analyze_opening_screen(
        frame,
        my_color=_optional_player_color(args.my_color),
        top_k=args.top_k,
    )
    print(
        f"Prompt={analysis.prompt.kind.value}  color={analysis.my_color.value}  seat=S{analysis.seat + 1}  "
        f"ports={'generic' if analysis.bootstrap.ports_are_conservative_generic else 'detected'}"
    )
    for index, suggestion in enumerate(analysis.suggestions, start=1):
        print(f"{index}. {suggestion.score:.3f} -> {suggestion_target_text(analysis, suggestion)}")
        print(f"   {suggestion.summary}")
        if suggestion.plan:
            print(f"   {suggestion.plan}")
    if args.output:
        bgr = cv2.cvtColor(analysis.annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(args.output)), bgr)
        print(f"Saved annotated frame to {args.output}")
    return 0


def _cmd_opening_live(args: argparse.Namespace) -> int:
    runner = OpeningLiveRunner()
    runner.run(
        region=ScreenRegion.parse(args.bbox) if args.bbox else None,
        my_color=_optional_player_color(args.my_color),
        top_k=args.top_k,
        interval_s=args.interval,
        once=args.once,
        show=not args.no_show,
    )
    return 0


def _add_opening_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    parser.add_argument("--my-color", default="auto", help="Local color or 'auto'")
    parser.add_argument("--top-k", type=int, default=3, help="Number of suggestions to track")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Capture and analyze once")
    parser.add_argument("--no-show", action="store_true", help="Do not open the overlay window")
    parser.set_defaults(func=_cmd_opening_live)


def _cmd_init_context(args: argparse.Namespace) -> int:
    payload = {
        "current_player": 0,
        "phase": "pre_roll",
        "private_pov": {
            "player_id": args.player_id,
            "resources": {resource.value: 0 for resource in playable_resources()},
            "dev_cards_in_hand": {},
            "new_dev_cards_in_hand": {},
            "hidden_vp_cards": 0,
        },
        "public_overrides": [],
        "turn_number": 0,
        "setup_step": 0,
        "pending_setup_vertex": None,
        "pending_discarders": [],
        "free_roads_remaining": 0,
        "dev_card_played_this_turn": False,
        "dice_rolled_this_turn": False,
        "last_roll": None,
        "winner_id": None,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Saved context template to {args.output}")
    return 0


def _cmd_init_board(args: argparse.Namespace) -> int:
    _write_board_template(Path(args.output))
    print(f"Saved board template to {args.output}")
    return 0


def _cmd_opening(args: argparse.Namespace) -> int:
    board = _load_board(Path(args.board))
    seat = int(args.seat)
    model = WeightedHeuristic()
    vf_cache = compute_all_vertex_features(board)

    ranked = []
    for v1 in board.legal_starting_vertices():
        for v2 in board.legal_second_vertices(v1):
            features = compute_opening_features(v1, v2, seat, board, vf_cache)
            ranked.append((features, model.predict_win_probability(features)))
    ranked.sort(key=lambda item: item[1], reverse=True)

    top = ranked[: args.top_k]
    print(f"Top {len(top)} opening pairs for seat {seat}")
    for index, (features, score) in enumerate(top, start=1):
        print(
            f"{index}. {score*100:.1f}%  "
            f"v1=({features.v1[0]:.3f},{features.v1[1]:.3f})  "
            f"v2=({features.v2[0]:.3f},{features.v2[1]:.3f})  "
            f"pips={features.combined_pip_count}  "
            f"resources={features.unique_resource_count}  "
            f"ports={features.num_ports}"
        )

    if top:
        print("\nBest opening explanation:")
        explanation = explain_opening(top[0][0], model, board)
        print(f"Win probability: {explanation.win_probability * 100:.1f}%")
        print(f"Percentile: {explanation.percentile:.1f}")
        print(f"Archetype: {explanation.archetype}")
        if explanation.strengths:
            print("Strengths:")
            for item in explanation.strengths:
                print(f"  + {item}")
        if explanation.weaknesses:
            print("Weaknesses:")
            for item in explanation.weaknesses:
                print(f"  - {item}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Colonist CV assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    advise = subparsers.add_parser("advise", help="Rank legal actions from a mapped observation JSON file")
    advise.add_argument("--board", required=True, help="Path to a static board JSON file")
    advise.add_argument("--calibration", required=True, help="Path to a board calibration JSON file")
    advise.add_argument("--observation", required=True, help="Path to a mapped frame observation JSON file")
    advise.add_argument("--top-k", type=int, default=5)
    advise.set_defaults(func=_cmd_advise)

    calibrate = subparsers.add_parser("calibrate", help="Capture the screen and click four anchor hexes")
    calibrate.add_argument("--board", help="Optional board JSON file; not required for calibration geometry")
    calibrate.add_argument("--output", required=True, help="Where to write the calibration JSON")
    calibrate.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    calibrate.set_defaults(func=_cmd_calibrate)

    live = subparsers.add_parser("live", help="Run the live Colonist advisor loop")
    live.add_argument("--board", required=True, help="Path to a static board JSON file")
    live.add_argument("--calibration", required=True, help="Path to a calibration JSON file")
    live.add_argument("--context", help="Optional fallback context JSON file for values OCR cannot read")
    live.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    live.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    live.add_argument("--top-k", type=int, default=5)
    live.add_argument("--color-order", default="red,blue,orange,green", help="Seat colors in player-id order")
    live.add_argument("--my-color", default="auto", help="Local color or 'auto'")
    live.add_argument("--slow-loop-ms", type=float, default=350.0, help="Warn when one loop exceeds this latency target")
    live.add_argument("--stale-seconds", type=float, default=30.0, help="Report when advice has not changed for this long")
    live.add_argument("--once", action="store_true", help="Capture and advise once instead of polling")
    live.set_defaults(func=_cmd_live)

    init_context = subparsers.add_parser("init-context", help="Write an editable live context JSON template")
    init_context.add_argument("--output", required=True, help="Where to write the context JSON")
    init_context.add_argument("--player-id", type=int, default=0, help="Your player id for private hand data")
    init_context.set_defaults(func=_cmd_init_context)

    init_board = subparsers.add_parser("init-board", help="Write an editable board JSON template")
    init_board.add_argument("--output", required=True, help="Where to write the board JSON")
    init_board.set_defaults(func=_cmd_init_board)

    opening = subparsers.add_parser("opening", help="Rank opening settlement pairs from a board JSON file")
    opening.add_argument("--board", required=True, help="Path to a filled board JSON file")
    opening.add_argument("--seat", type=int, default=0, choices=[0, 1, 2, 3], help="Seat position")
    opening.add_argument("--top-k", type=int, default=10, help="Number of opening pairs to print")
    opening.set_defaults(func=_cmd_opening)

    bootstrap_screen = subparsers.add_parser(
        "bootstrap-screen",
        help="Infer the current Colonist board directly from the live screen and write a board JSON",
    )
    bootstrap_screen.add_argument("--output", required=True, help="Where to write the inferred board JSON")
    bootstrap_screen.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    bootstrap_screen.set_defaults(func=_cmd_bootstrap_screen)

    opening_screen = subparsers.add_parser(
        "opening-screen",
        help="Capture the live Colonist screen once, infer the board, and print opening advice",
    )
    opening_screen.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    opening_screen.add_argument("--my-color", default="auto", help="Local color or 'auto'")
    opening_screen.add_argument("--top-k", type=int, default=3, help="Number of suggestions to print")
    opening_screen.add_argument("--output", help="Optional path for the annotated screenshot")
    opening_screen.set_defaults(func=_cmd_opening_screen)

    opening_live = subparsers.add_parser(
        "opening-live",
        help="Continuously watch Colonist and overlay setup placement suggestions from the screen",
    )
    _add_opening_live_arguments(opening_live)

    context_screen = subparsers.add_parser(
        "context-screen",
        help="Capture the current Colonist screen once and print OCR-derived turn context",
    )
    context_screen.add_argument("--bbox", help="Optional capture box: left,top,right,bottom")
    context_screen.add_argument("--my-color", default="auto", help="Local color or 'auto'")
    context_screen.add_argument("--color-order", default="red,blue,orange,green", help="Seat colors in player-id order")
    context_screen.set_defaults(func=_cmd_context_screen)

    args = parser.parse_args()
    try:
        return args.func(args)
    except DetectionError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


def opening_live_main() -> int:
    parser = argparse.ArgumentParser(description="Run the Colonist opening overlay directly")
    _add_opening_live_arguments(parser)
    args = parser.parse_args()
    try:
        return args.func(args)
    except DetectionError as exc:
        parser.error(str(exc))
    return 2
