"""Live screen-capture runtime for the Colonist CV assistant."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import ImageGrab

from ..board.board import Resource
from ..full_solver.state import DevCardType, TurnPhase, empty_hand
from .advisor import ActionAdvice, HeuristicActionAdvisor, StrategyPlan
from .context_ocr import ScreenContextDetection, read_screen_context
from .detector import ColonistVisionDetector, DetectionError
from .geometry import BoardCalibration
from .schema import PlayerColor, PrivateObservation, PublicStructures, VisionFrameObservation
from .tracker import ColonistVisionTracker


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    right: int
    bottom: int

    def as_bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def parse(cls, raw: str) -> "ScreenRegion":
        parts = [int(part.strip()) for part in raw.split(",")]
        if len(parts) != 4:
            raise ValueError("Screen region must be 'left,top,right,bottom'.")
        left, top, right, bottom = parts
        if right <= left or bottom <= top:
            raise ValueError("Screen region must have right > left and bottom > top.")
        return cls(left=left, top=top, right=right, bottom=bottom)


@dataclass(frozen=True)
class LiveContext:
    current_player: int
    phase: TurnPhase
    private_pov: Optional[PrivateObservation] = None
    public_overrides: tuple[PublicStructures, ...] = field(default_factory=tuple)
    turn_number: int = 0
    setup_step: int = 0
    pending_setup_vertex: Optional[tuple[float, float]] = None
    pending_discarders: tuple[int, ...] = field(default_factory=tuple)
    free_roads_remaining: int = 0
    dev_card_played_this_turn: bool = False
    dice_rolled_this_turn: bool = False
    last_roll: Optional[int] = None
    winner_id: Optional[int] = None


def grab_screen(region: Optional[ScreenRegion]) -> np.ndarray:
    image = ImageGrab.grab(bbox=region.as_bbox() if region is not None else None, all_screens=True)
    return np.asarray(image.convert("RGB"))


def save_calibration_interactive(
    image: np.ndarray,
    board,
    output_path: Path,
    capture_region: Optional[ScreenRegion] = None,
) -> BoardCalibration:
    anchors = [(-2, 0), (2, 0), (0, 2), (0, -2)]
    labels = [
        "Click center of hex (-2, 0)",
        "Click center of hex (2, 0)",
        "Click center of hex (0, 2)",
        "Click center of hex (0, -2)",
    ]
    clicked: list[tuple[float, float]] = []
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    window_name = "Colonist Calibration"

    def redraw() -> None:
        display = canvas.copy()
        for index, point in enumerate(clicked):
            cv2.circle(display, (int(point[0]), int(point[1])), 8, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(index + 1),
                (int(point[0]) + 10, int(point[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        prompt = labels[min(len(clicked), len(labels) - 1)]
        cv2.putText(display, prompt, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, "Press backspace to undo, q to quit", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

    def on_click(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(anchors):
            clicked.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_click)
    redraw()

    try:
        while len(clicked) < len(anchors):
            key = cv2.waitKey(25) & 0xFF
            if key in (8, 127) and clicked:
                clicked.pop()
                redraw()
            elif key == ord("q"):
                raise RuntimeError("Calibration cancelled.")
        calibration = BoardCalibration.from_hex_anchors(board, dict(zip(anchors, clicked)))
    finally:
        cv2.destroyWindow(window_name)

    payload = calibration.to_serialized()
    if capture_region is not None:
        payload["capture_bbox"] = {
            "left": capture_region.left,
            "top": capture_region.top,
            "right": capture_region.right,
            "bottom": capture_region.bottom,
        }
    output_path.write_text(json.dumps(payload, indent=2))
    return calibration


def load_live_context(path: Path) -> LiveContext:
    payload = json.loads(path.read_text())
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

    public_overrides = []
    for player in payload.get("public_overrides", []):
        public_overrides.append(
            PublicStructures(
                player_id=int(player["player_id"]),
                color=PlayerColor.from_value(player["color"]) if player.get("color") else None,
                visible_vp=player.get("visible_vp"),
                played_knights=int(player.get("played_knights", 0)),
                dev_cards_bought=int(player.get("dev_cards_bought", 0)),
            )
        )

    return LiveContext(
        current_player=int(payload["current_player"]),
        phase=TurnPhase(payload["phase"]),
        private_pov=private_pov,
        public_overrides=tuple(public_overrides),
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


def apply_context_overrides(
    observation: VisionFrameObservation,
    context: LiveContext,
) -> VisionFrameObservation:
    overrides = {player.player_id: player for player in context.public_overrides}
    merged_players = []
    for player in observation.public_players:
        override = overrides.get(player.player_id)
        if override is None:
            merged_players.append(player)
            continue
        merged_players.append(
            replace(
                player,
                visible_vp=override.visible_vp if override.visible_vp is not None else player.visible_vp,
                played_knights=override.played_knights,
                dev_cards_bought=override.dev_cards_bought,
            )
        )

    return replace(
        observation,
        public_players=tuple(merged_players),
        current_player=context.current_player,
        phase=context.phase,
        private_pov=context.private_pov,
        turn_number=context.turn_number,
        setup_step=context.setup_step,
        pending_setup_vertex=context.pending_setup_vertex,
        pending_discarders=context.pending_discarders,
        free_roads_remaining=context.free_roads_remaining,
        dev_card_played_this_turn=context.dev_card_played_this_turn,
        dice_rolled_this_turn=context.dice_rolled_this_turn,
        last_roll=context.last_roll,
        winner_id=context.winner_id,
    )


def fingerprint_observation(observation: VisionFrameObservation) -> str:
    public_payload = []
    for player in observation.public_players:
        public_payload.append(
            {
                "player_id": player.player_id,
                "settlements": sorted(player.settlements),
                "cities": sorted(player.cities),
                "roads": sorted(sorted(tuple(vertex) for vertex in edge) for edge in player.roads),
                "visible_vp": player.visible_vp,
                "played_knights": player.played_knights,
                "dev_cards_bought": player.dev_cards_bought,
            }
        )
    private_payload = None
    if observation.private_pov is not None:
        canonical = observation.private_pov.canonical()
        private_payload = {
            "player_id": canonical.player_id,
            "resources": {resource.value: canonical.resources[resource] for resource in canonical.resources},
            "dev_cards_in_hand": {card_type.value: count for card_type, count in canonical.dev_cards_in_hand.items()},
            "new_dev_cards_in_hand": {card_type.value: count for card_type, count in canonical.new_dev_cards_in_hand.items()},
            "hidden_vp_cards": canonical.hidden_vp_cards,
        }
    payload = {
        "robber_hex": observation.robber_hex,
        "current_player": observation.current_player,
        "phase": observation.phase.value,
        "public_players": public_payload,
        "private_pov": private_payload,
        "turn_number": observation.turn_number,
        "last_roll": observation.last_roll,
        "free_roads_remaining": observation.free_roads_remaining,
    }
    return json.dumps(payload, sort_keys=True)


def format_advice_lines(advice: list[ActionAdvice]) -> list[str]:
    lines = []
    for index, item in enumerate(advice, start=1):
        lines.append(f"{index}. [{item.score:.2f}] {item.action.action_type.value}")
        lines.append(f"   {item.summary}")
    return lines


def format_strategy_lines(plan: StrategyPlan) -> list[str]:
    return [
        f"Lean: {plan.lean}",
        f"Buy: {plan.buy_priority}",
        f"Goal: {plan.hand_goal}",
        f"Risk: {plan.risk}",
    ]


def _merge_screen_context(
    detected: ScreenContextDetection,
    fallback: Optional[LiveContext],
) -> LiveContext:
    current_player = detected.current_player
    if current_player is None and fallback is not None:
        current_player = fallback.current_player
    phase = detected.phase
    if phase is None and fallback is not None:
        phase = fallback.phase
    if current_player is None or phase is None:
        raise DetectionError(
            "Could not read the active turn from the screen. Pass --my-color and keep the action prompt visible, or keep a context JSON as fallback."
        )

    private_pov = detected.private_pov if detected.private_pov is not None else (fallback.private_pov if fallback is not None else None)
    dice_rolled = detected.dice_rolled_this_turn
    if dice_rolled is None and fallback is not None:
        dice_rolled = fallback.dice_rolled_this_turn

    return LiveContext(
        current_player=current_player,
        phase=phase,
        private_pov=private_pov,
        public_overrides=fallback.public_overrides if fallback is not None else tuple(),
        turn_number=fallback.turn_number if fallback is not None else 0,
        setup_step=fallback.setup_step if fallback is not None else 0,
        pending_setup_vertex=fallback.pending_setup_vertex if fallback is not None else None,
        pending_discarders=fallback.pending_discarders if fallback is not None else tuple(),
        free_roads_remaining=fallback.free_roads_remaining if fallback is not None else 0,
        dev_card_played_this_turn=fallback.dev_card_played_this_turn if fallback is not None else False,
        dice_rolled_this_turn=bool(dice_rolled),
        last_roll=fallback.last_roll if fallback is not None else None,
        winner_id=fallback.winner_id if fallback is not None else None,
    )


class LiveAdvisorRunner:
    """Continuous screenshot polling loop that emits updated move suggestions."""

    def __init__(
        self,
        board,
        calibration: BoardCalibration,
        detector: ColonistVisionDetector | None = None,
        advisor: HeuristicActionAdvisor | None = None,
        color_to_player: Optional[dict[PlayerColor, int]] = None,
    ):
        self.board = board
        self.calibration = calibration
        self.detector = detector or ColonistVisionDetector()
        self.advisor = advisor or HeuristicActionAdvisor()
        self.color_to_player = color_to_player or {
            PlayerColor.RED: 0,
            PlayerColor.BLUE: 1,
            PlayerColor.ORANGE: 2,
            PlayerColor.GREEN: 3,
        }
        self.tracker = ColonistVisionTracker()

    def run(
        self,
        region: Optional[ScreenRegion],
        context_path: Optional[Path],
        my_color: Optional[PlayerColor] = None,
        top_k: int = 5,
        interval_s: float = 1.0,
        once: bool = False,
    ) -> None:
        previous_fingerprint = None
        while True:
            frame = grab_screen(region)
            fallback = load_live_context(context_path) if context_path is not None and context_path.exists() else None
            player_id_hint = None
            if fallback is not None and fallback.private_pov is not None:
                player_id_hint = fallback.private_pov.player_id
            elif my_color is not None and my_color in self.color_to_player:
                player_id_hint = self.color_to_player[my_color]
            detected = read_screen_context(
                frame,
                my_color=my_color,
                color_to_player=self.color_to_player,
                player_id_hint=player_id_hint,
            )
            context = _merge_screen_context(detected, fallback)
            observation = self.detector.detect_frame(
                frame,
                board=self.board,
                calibration=self.calibration,
                color_to_player=self.color_to_player,
                current_player=context.current_player,
                phase=context.phase,
                private_pov=context.private_pov,
            )
            observation = apply_context_overrides(observation, context)
            fingerprint = fingerprint_observation(observation)
            if fingerprint != previous_fingerprint:
                state = self.tracker.ingest(observation)
                plan = self.advisor.strategy_plan(state)
                advice = self.advisor.suggest(state, top_k=top_k)
                print(f"\n[{time.strftime('%H:%M:%S')}] player={observation.current_player} phase={observation.phase.value}")
                for line in format_strategy_lines(plan):
                    print(line)
                for line in format_advice_lines(advice):
                    print(line)
                previous_fingerprint = fingerprint
            if once:
                return
            time.sleep(interval_s)
