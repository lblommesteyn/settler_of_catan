"""Structured parsing for Colonist game-log events."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..board.board import Resource
from ..full_solver.state import DevCardType, TurnPhase
from .schema import PlayerColor


DATASET_COLOR_NAMES: dict[int, str] = {
    1: "blue",
    2: "red",
    3: "orange",
    4: "brown",
    5: "white",
}

COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red",),
    "blue": ("blue",),
    "orange": ("orange", "brown"),
    "green": ("green",),
    "white": ("white",),
}

RESOURCE_ENUM_TO_RESOURCE: dict[int, Resource] = {
    1: Resource.BRICK,
    2: Resource.SHEEP,
    3: Resource.WHEAT,
    4: Resource.ORE,
    5: Resource.WOOD,
}

DEV_CARD_ENUM_TO_TYPE: dict[int, DevCardType] = {
    11: DevCardType.KNIGHT,
    13: DevCardType.MONOPOLY,
    14: DevCardType.ROAD_BUILDING,
    15: DevCardType.YEAR_OF_PLENTY,
    17: DevCardType.VICTORY_POINT,
}

PIECE_ENUM_TO_NAME: dict[int, str] = {
    0: "road",
    2: "settlement",
    3: "city",
    5: "robber",
}

ACHIEVEMENT_ENUM_TO_NAME: dict[int, str] = {
    0: "longest_road",
    1: "largest_army",
}

LOG_COLORS = tuple(sorted({alias for aliases in COLOR_ALIASES.values() for alias in aliases}, key=len, reverse=True))
RESOURCE_WORD_TO_RESOURCE: dict[str, Resource] = {
    "brick": Resource.BRICK,
    "sheep": Resource.SHEEP,
    "wool": Resource.SHEEP,
    "wheat": Resource.WHEAT,
    "ore": Resource.ORE,
    "wood": Resource.WOOD,
    "lumber": Resource.WOOD,
}

RESOURCE_PATTERN = re.compile(
    r"(?:(?P<count>\d+)\s*x?\s*)?(?P<name>brick|ore|wheat|sheep|wool|wood|lumber)s?\b"
)

SETUP_ORDER = (0, 1, 2, 3, 3, 2, 1, 0)


class ColonistEventType(str, Enum):
    TURN_START = "turn_start"
    TURN_BOUNDARY = "turn_boundary"
    END_TURN = "end_turn"
    ROLL = "roll"
    BUILD = "build"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_DEV_CARD = "play_dev_card"
    RESOURCE_DISTRIBUTION = "resource_distribution"
    MOVE_ROBBER = "move_robber"
    STEAL_CARD = "steal_card"
    TRADE_OFFER = "trade_offer"
    TRADE_ACCEPT = "trade_accept"
    MARITIME_TRADE = "maritime_trade"
    DISCARD_REQUEST = "discard_request"
    DISCARD = "discard"
    CLAIM_ACHIEVEMENT = "claim_achievement"
    TRANSFER_ACHIEVEMENT = "transfer_achievement"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ColonistGameEvent:
    event_type: ColonistEventType
    source: str
    text: str = ""
    raw_type: Optional[int] = None
    sequence: Optional[int] = None
    completed_turns: Optional[int] = None
    player_name: Optional[str] = None
    player_id: Optional[int] = None
    other_player_name: Optional[str] = None
    other_player_id: Optional[int] = None
    piece: Optional[str] = None
    dev_card: Optional[DevCardType] = None
    achievement: Optional[str] = None
    dice_total: Optional[int] = None
    amount: Optional[int] = None
    given_resources: tuple[Resource, ...] = field(default_factory=tuple)
    received_resources: tuple[Resource, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        actor = self.player_name or (f"p{self.player_id}" if self.player_id is not None else "unknown")
        target = self.other_player_name or (f"p{self.other_player_id}" if self.other_player_id is not None else "unknown")
        if self.event_type == ColonistEventType.ROLL:
            return f"{actor} rolled {self.dice_total}"
        if self.event_type == ColonistEventType.BUILD:
            return f"{actor} built {self.piece}"
        if self.event_type == ColonistEventType.BUY_DEV_CARD:
            return f"{actor} bought a development card"
        if self.event_type == ColonistEventType.PLAY_DEV_CARD:
            return f"{actor} played {self.dev_card.value if self.dev_card is not None else 'a development card'}"
        if self.event_type == ColonistEventType.TRADE_OFFER:
            return f"{actor} offered {_format_resource_multiset(self.given_resources)} for {_format_resource_multiset(self.received_resources)}"
        if self.event_type == ColonistEventType.TRADE_ACCEPT:
            return f"{actor} traded with {target}"
        if self.event_type == ColonistEventType.MARITIME_TRADE:
            return f"{actor} bank-traded {_format_resource_multiset(self.given_resources)} for {_format_resource_multiset(self.received_resources)}"
        if self.event_type == ColonistEventType.MOVE_ROBBER:
            return f"{actor} moved the robber"
        if self.event_type == ColonistEventType.STEAL_CARD:
            return f"{actor} stole from {target}"
        if self.event_type == ColonistEventType.DISCARD_REQUEST:
            return f"{actor} needs to discard"
        if self.event_type == ColonistEventType.DISCARD:
            return f"{actor} discarded {_format_resource_multiset(self.given_resources) or self.amount or 'cards'}"
        if self.event_type == ColonistEventType.CLAIM_ACHIEVEMENT:
            return f"{actor} claimed {self.achievement}"
        if self.event_type == ColonistEventType.TRANSFER_ACHIEVEMENT:
            return f"{actor} took {self.achievement} from {target}"
        if self.event_type == ColonistEventType.END_TURN:
            return f"{actor} ended the turn"
        if self.event_type == ColonistEventType.TURN_START:
            return f"{actor}'s turn"
        if self.event_type == ColonistEventType.GAME_OVER:
            return f"{actor} won the game"
        if self.event_type == ColonistEventType.RESOURCE_DISTRIBUTION:
            resources = _format_resource_multiset(self.received_resources)
            return f"{actor} received {resources}" if resources else f"{actor} received resources"
        return self.text or self.event_type.value


@dataclass(frozen=True)
class InferredTurnContext:
    current_player: Optional[int]
    phase: Optional[TurnPhase]
    dice_rolled_this_turn: Optional[bool]
    setup_step: Optional[int] = None
    pending_discarders: tuple[int, ...] = field(default_factory=tuple)
    last_roll: Optional[int] = None


def parse_dataset_state_change(
    state_change: dict,
    *,
    sequence: Optional[int] = None,
    color_to_player: Optional[dict[PlayerColor, int]] = None,
) -> tuple[ColonistGameEvent, ...]:
    """Parse a raw Colonist dataset `stateChange.gameLogState` block."""

    completed_turns = None
    current_state = state_change.get("currentState", {})
    if isinstance(current_state, dict) and current_state.get("completedTurns") is not None:
        completed_turns = int(current_state["completedTurns"])

    game_log = state_change.get("gameLogState", {})
    if not isinstance(game_log, dict):
        return tuple()

    events: list[ColonistGameEvent] = []
    for _, entry in sorted(game_log.items(), key=_numeric_key):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", {})
        if not isinstance(text, dict):
            continue
        parsed = _parse_dataset_log_entry(
            text,
            sequence=sequence,
            completed_turns=completed_turns,
            color_to_player=color_to_player,
        )
        if parsed is not None:
            events.append(parsed)
    return tuple(events)


def parse_visible_log_lines(
    lines: list[str] | tuple[str, ...],
    *,
    color_to_player: Optional[dict[PlayerColor, int]] = None,
) -> tuple[ColonistGameEvent, ...]:
    """Parse OCR'd visible Colonist log lines into structured events."""

    events: list[ColonistGameEvent] = []
    for sequence, raw_line in enumerate(lines):
        normalized = _normalize_log_text(raw_line)
        if not normalized:
            continue
        event = _parse_visible_log_line(normalized, sequence=sequence, color_to_player=color_to_player)
        if event is not None:
            events.append(event)
    return tuple(events)


def infer_turn_context_from_events(
    events: tuple[ColonistGameEvent, ...] | list[ColonistGameEvent],
    *,
    color_to_player: Optional[dict[PlayerColor, int]] = None,
) -> InferredTurnContext:
    """Infer active player and phase from recent structured log events."""

    if not events:
        return InferredTurnContext(current_player=None, phase=None, dice_rolled_this_turn=None)

    setup_context = _infer_setup_context(events)
    if setup_context.phase is not None:
        return setup_context

    latest = events[-1]
    if latest.event_type == ColonistEventType.GAME_OVER:
        return InferredTurnContext(
            current_player=latest.player_id,
            phase=TurnPhase.GAME_OVER,
            dice_rolled_this_turn=True,
        )

    if latest.event_type == ColonistEventType.END_TURN:
        return InferredTurnContext(
            current_player=_next_player_id(latest.player_id, color_to_player),
            phase=TurnPhase.PRE_ROLL,
            dice_rolled_this_turn=False,
        )

    if latest.event_type == ColonistEventType.TURN_START:
        return InferredTurnContext(
            current_player=latest.player_id,
            phase=TurnPhase.PRE_ROLL,
            dice_rolled_this_turn=False,
        )

    current_player = latest.player_id
    if current_player is None:
        current_player = next((event.player_id for event in reversed(events) if event.player_id is not None), None)

    window = _current_turn_window(events)
    last_roll_event = next(
        (
            event
            for event in reversed(window)
            if event.event_type == ColonistEventType.ROLL
            and (current_player is None or event.player_id == current_player)
        ),
        None,
    )
    last_roll = last_roll_event.dice_total if last_roll_event is not None else None

    if latest.event_type == ColonistEventType.TRADE_OFFER:
        return InferredTurnContext(
            current_player=latest.player_id,
            phase=TurnPhase.PENDING_TRADE,
            dice_rolled_this_turn=True,
            last_roll=last_roll,
        )

    if latest.event_type == ColonistEventType.ROLL:
        return InferredTurnContext(
            current_player=latest.player_id,
            phase=TurnPhase.RESOLVE_SEVEN if latest.dice_total == 7 else TurnPhase.MAIN,
            dice_rolled_this_turn=True,
            last_roll=latest.dice_total,
        )

    if latest.event_type in {ColonistEventType.DISCARD_REQUEST, ColonistEventType.DISCARD}:
        pending_discarders = tuple(
            player_id
            for player_id in _unique_in_order(
                event.player_id
                for event in window
                if event.event_type == ColonistEventType.DISCARD_REQUEST and event.player_id is not None
            )
        )
        roller = None
        if last_roll_event is not None and last_roll_event.dice_total == 7:
            roller = last_roll_event.player_id
        return InferredTurnContext(
            current_player=roller if roller is not None else current_player,
            phase=TurnPhase.RESOLVE_SEVEN,
            dice_rolled_this_turn=True,
            pending_discarders=pending_discarders,
            last_roll=last_roll or 7,
        )

    if latest.event_type in {ColonistEventType.MOVE_ROBBER, ColonistEventType.STEAL_CARD}:
        if last_roll_event is not None and last_roll_event.dice_total == 7:
            pending_discarders = tuple(
                player_id
                for player_id in _unique_in_order(
                    event.player_id
                    for event in window
                    if event.event_type == ColonistEventType.DISCARD_REQUEST and event.player_id is not None
                )
            )
            return InferredTurnContext(
                current_player=last_roll_event.player_id,
                phase=TurnPhase.RESOLVE_SEVEN,
                dice_rolled_this_turn=True,
                pending_discarders=pending_discarders,
                last_roll=7,
            )
        return InferredTurnContext(
            current_player=current_player,
            phase=TurnPhase.MAIN if last_roll_event is not None else TurnPhase.PRE_ROLL,
            dice_rolled_this_turn=last_roll_event is not None,
            last_roll=last_roll,
        )

    if latest.event_type == ColonistEventType.PLAY_DEV_CARD:
        return InferredTurnContext(
            current_player=current_player,
            phase=TurnPhase.MAIN if last_roll_event is not None else TurnPhase.PRE_ROLL,
            dice_rolled_this_turn=last_roll_event is not None,
            last_roll=last_roll,
        )

    if latest.event_type in {
        ColonistEventType.BUILD,
        ColonistEventType.BUY_DEV_CARD,
        ColonistEventType.TRADE_ACCEPT,
        ColonistEventType.MARITIME_TRADE,
        ColonistEventType.CLAIM_ACHIEVEMENT,
        ColonistEventType.TRANSFER_ACHIEVEMENT,
        ColonistEventType.RESOURCE_DISTRIBUTION,
    }:
        return InferredTurnContext(
            current_player=current_player,
            phase=TurnPhase.MAIN,
            dice_rolled_this_turn=True,
            last_roll=last_roll,
        )

    return InferredTurnContext(
        current_player=current_player,
        phase=TurnPhase.MAIN if last_roll_event is not None else None,
        dice_rolled_this_turn=last_roll_event is not None if last_roll_event is not None else None,
        last_roll=last_roll,
    )


def _parse_dataset_log_entry(
    text: dict,
    *,
    sequence: Optional[int],
    completed_turns: Optional[int],
    color_to_player: Optional[dict[PlayerColor, int]],
) -> Optional[ColonistGameEvent]:
    raw_type = text.get("type")
    if raw_type is None:
        return None
    raw_type = int(raw_type)

    if raw_type == 44:
        return ColonistGameEvent(
            event_type=ColonistEventType.TURN_BOUNDARY,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            text="turn boundary",
        )

    if raw_type == 10:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        first = int(text.get("firstDice", 0))
        second = int(text.get("secondDice", 0))
        return ColonistGameEvent(
            event_type=ColonistEventType.ROLL,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            dice_total=first + second,
            text=f"{player_name or player_id} rolled {first + second}",
        )

    if raw_type in {4, 5}:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        piece = PIECE_ENUM_TO_NAME.get(int(text.get("pieceEnum", -1)))
        if piece is None:
            return None
        return ColonistGameEvent(
            event_type=ColonistEventType.BUILD,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            piece=piece,
            text=f"{player_name or player_id} built {piece}",
        )

    if raw_type == 1:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.END_TURN,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            text=f"{player_name or player_id} ended the turn",
        )

    if raw_type in {117, 118}:
        actor_name, actor_id = _dataset_player(
            text.get("playerColorCreator", text.get("playerColor")),
            color_to_player,
        )
        other_name, other_id = _dataset_player(text.get("playerColorOffered"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.TRADE_OFFER,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            given_resources=_decode_resource_enums(text.get("offeredCardEnums", [])),
            received_resources=_decode_resource_enums(text.get("wantedCardEnums", [])),
            text="trade offer",
        )

    if raw_type == 115:
        actor_name, actor_id = _dataset_player(text.get("playerColor"), color_to_player)
        other_name, other_id = _dataset_player(text.get("acceptingPlayerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.TRADE_ACCEPT,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            given_resources=_decode_resource_enums(text.get("givenCardEnums", [])),
            received_resources=_decode_resource_enums(text.get("receivedCardEnums", [])),
            text="domestic trade accepted",
        )

    if raw_type == 116:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.MARITIME_TRADE,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            given_resources=_decode_resource_enums(text.get("givenCardEnums", [])),
            received_resources=_decode_resource_enums(text.get("receivedCardEnums", [])),
            text="maritime trade",
        )

    if raw_type == 16:
        thief_name, thief_id = _dataset_player(text.get("playerColorThief"), color_to_player)
        victim_name, victim_id = _dataset_player(text.get("playerColorVictim"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.STEAL_CARD,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=thief_name,
            player_id=thief_id,
            other_player_name=victim_name,
            other_player_id=victim_id,
            amount=len(text.get("cardBacks", [])),
            text="stole card",
        )

    if raw_type == 11:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.MOVE_ROBBER,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            piece="robber",
            text="moved robber",
        )

    if raw_type == 20:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        dev_card = DEV_CARD_ENUM_TO_TYPE.get(int(text.get("cardEnum", -1)))
        return ColonistGameEvent(
            event_type=ColonistEventType.PLAY_DEV_CARD,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            dev_card=dev_card,
            text="played development card",
        )

    if raw_type in {21, 47}:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        key = "cardEnums" if raw_type == 21 else "cardsToBroadcast"
        return ColonistGameEvent(
            event_type=ColonistEventType.RESOURCE_DISTRIBUTION,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            received_resources=_decode_resource_enums(text.get(key, [])),
            text="received resources",
        )

    if raw_type == 55:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        cards = _decode_resource_enums(text.get("cardEnums", []))
        return ColonistGameEvent(
            event_type=ColonistEventType.DISCARD,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            given_resources=cards,
            amount=len(cards),
            text="discarded cards",
        )

    if raw_type == 66:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.CLAIM_ACHIEVEMENT,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            achievement=ACHIEVEMENT_ENUM_TO_NAME.get(int(text.get("achievementEnum", -1))),
            text="claimed achievement",
        )

    if raw_type == 68:
        new_name, new_id = _dataset_player(text.get("playerColorNew"), color_to_player)
        old_name, old_id = _dataset_player(text.get("playerColorOld"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.TRANSFER_ACHIEVEMENT,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=new_name,
            player_id=new_id,
            other_player_name=old_name,
            other_player_id=old_id,
            achievement=ACHIEVEMENT_ENUM_TO_NAME.get(int(text.get("achievementEnum", -1))),
            text="transferred achievement",
        )

    if raw_type == 0:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.TURN_START,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            text="turn start",
        )

    if raw_type == 45:
        player_name, player_id = _dataset_player(text.get("playerColor"), color_to_player)
        return ColonistGameEvent(
            event_type=ColonistEventType.GAME_OVER,
            source="dataset",
            raw_type=raw_type,
            sequence=sequence,
            completed_turns=completed_turns,
            player_name=player_name,
            player_id=player_id,
            text="won the game",
        )

    return None


def _parse_visible_log_line(
    line: str,
    *,
    sequence: int,
    color_to_player: Optional[dict[PlayerColor, int]],
) -> Optional[ColonistGameEvent]:
    player_names = _extract_player_names(line)
    actor_name = player_names[0] if player_names else None
    other_name = player_names[1] if len(player_names) > 1 else None
    actor_id = _resolve_player_id(actor_name, color_to_player)
    other_id = _resolve_player_id(other_name, color_to_player)

    if "won the game" in line or "wins the game" in line:
        return ColonistGameEvent(
            event_type=ColonistEventType.GAME_OVER,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
        )

    roll_match = re.search(r"\brolled(?: a)? (?P<total>\d+)\b", line)
    if roll_match is not None:
        return ColonistGameEvent(
            event_type=ColonistEventType.ROLL,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            dice_total=int(roll_match.group("total")),
        )

    if "played" in line:
        for phrase, dev_card in (
            ("road building", DevCardType.ROAD_BUILDING),
            ("year of plenty", DevCardType.YEAR_OF_PLENTY),
            ("monopoly", DevCardType.MONOPOLY),
            ("knight", DevCardType.KNIGHT),
        ):
            if phrase in line:
                return ColonistGameEvent(
                    event_type=ColonistEventType.PLAY_DEV_CARD,
                    source="ocr",
                    sequence=sequence,
                    text=line,
                    player_name=actor_name,
                    player_id=actor_id,
                    dev_card=dev_card,
                )

    if "bought" in line and "development" in line:
        return ColonistGameEvent(
            event_type=ColonistEventType.BUY_DEV_CARD,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
        )

    if "moved" in line and "robber" in line:
        return ColonistGameEvent(
            event_type=ColonistEventType.MOVE_ROBBER,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            piece="robber",
        )

    if "stole" in line and " from " in line:
        amount_match = re.search(r"\bstole (?P<count>\d+)\b", line)
        return ColonistGameEvent(
            event_type=ColonistEventType.STEAL_CARD,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            amount=int(amount_match.group("count")) if amount_match is not None else None,
        )

    if "needs to discard" in line or "must discard" in line:
        return ColonistGameEvent(
            event_type=ColonistEventType.DISCARD_REQUEST,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
        )

    if "discarded" in line:
        cards = _parse_resource_mentions(line)
        return ColonistGameEvent(
            event_type=ColonistEventType.DISCARD,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            given_resources=cards,
            amount=len(cards) or _extract_first_integer(line),
        )

    if "largest army" in line or "longest road" in line:
        achievement = "largest_army" if "largest army" in line else "longest_road"
        event_type = ColonistEventType.CLAIM_ACHIEVEMENT
        if " from " in line or "took" in line or "stole" in line:
            event_type = ColonistEventType.TRANSFER_ACHIEVEMENT
        return ColonistGameEvent(
            event_type=event_type,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            achievement=achievement,
        )

    if "wants to give" in line or ("offer" in line and " for " in line):
        give_part, receive_part = _split_trade_line(line)
        return ColonistGameEvent(
            event_type=ColonistEventType.TRADE_OFFER,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            given_resources=_parse_resource_mentions(give_part),
            received_resources=_parse_resource_mentions(receive_part),
        )

    if ("accepted" in line and "trade" in line) or (" traded " in line and " with " in line):
        give_part, receive_part = _split_trade_line(line)
        event_type = ColonistEventType.TRADE_ACCEPT
        if "bank" in line or "maritime" in line:
            event_type = ColonistEventType.MARITIME_TRADE
        return ColonistGameEvent(
            event_type=event_type,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            other_player_name=other_name,
            other_player_id=other_id,
            given_resources=_parse_resource_mentions(give_part),
            received_resources=_parse_resource_mentions(receive_part),
        )

    if " with the bank" in line or "maritime trade" in line:
        give_part, receive_part = _split_trade_line(line)
        return ColonistGameEvent(
            event_type=ColonistEventType.MARITIME_TRADE,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            given_resources=_parse_resource_mentions(give_part),
            received_resources=_parse_resource_mentions(receive_part),
        )

    if "ended the turn" in line or "ended turn" in line or "ends the turn" in line:
        return ColonistGameEvent(
            event_type=ColonistEventType.END_TURN,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
        )

    if (line.endswith("'s turn") or re.search(r"\bturn\b", line)) and "turn to" not in line and actor_name is not None:
        return ColonistGameEvent(
            event_type=ColonistEventType.TURN_START,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
        )

    build_match = re.search(r"\b(?:built|placed|upgraded(?: to)?) (?:a |an )?(?P<piece>road|settlement|city)\b", line)
    if build_match is not None:
        return ColonistGameEvent(
            event_type=ColonistEventType.BUILD,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            piece=build_match.group("piece"),
        )

    resources = _parse_resource_mentions(line)
    if ("got" in line or "received" in line or "collected" in line) and resources:
        return ColonistGameEvent(
            event_type=ColonistEventType.RESOURCE_DISTRIBUTION,
            source="ocr",
            sequence=sequence,
            text=line,
            player_name=actor_name,
            player_id=actor_id,
            received_resources=resources,
        )

    return None


def _normalize_log_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("|", " ")
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^a-z0-9' ]+", " ", lowered)
    return " ".join(lowered.split())


def _extract_player_names(text: str) -> list[str]:
    names: list[str] = []
    for color in LOG_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            names.append(color)
    return names


def _parse_resource_mentions(text: str) -> tuple[Resource, ...]:
    resources: list[Resource] = []
    for match in RESOURCE_PATTERN.finditer(text):
        resource = RESOURCE_WORD_TO_RESOURCE[match.group("name")]
        count = int(match.group("count")) if match.group("count") else 1
        resources.extend([resource] * count)
    return tuple(resources)


def _split_trade_line(text: str) -> tuple[str, str]:
    if " wants to give " in text and " for " in text:
        after_give = text.split(" wants to give ", 1)[1]
        left, right = after_give.split(" for ", 1)
        return left, right
    if " traded " in text and " for " in text:
        after_traded = text.split(" traded ", 1)[1]
        left, right = after_traded.split(" for ", 1)
        return left, right
    if " offered " in text and " for " in text:
        after_offered = text.split(" offered ", 1)[1]
        left, right = after_offered.split(" for ", 1)
        return left, right
    return text, text


def _dataset_player(
    raw_color: object,
    color_to_player: Optional[dict[PlayerColor, int]],
) -> tuple[Optional[str], Optional[int]]:
    if raw_color is None:
        return None, None
    try:
        color_name = DATASET_COLOR_NAMES[int(raw_color)]
    except (KeyError, TypeError, ValueError):
        color_name = None
    return color_name, _resolve_player_id(color_name, color_to_player)


def _resolve_player_id(
    player_name: Optional[str],
    color_to_player: Optional[dict[PlayerColor, int]],
) -> Optional[int]:
    if player_name is None or color_to_player is None:
        return None
    normalized = player_name.strip().lower()
    for enum_color, player_id in color_to_player.items():
        aliases = COLOR_ALIASES.get(enum_color.value, (enum_color.value,))
        if normalized == enum_color.value or normalized in aliases:
            return player_id
    if normalized == "brown":
        for enum_color, player_id in color_to_player.items():
            if enum_color == PlayerColor.ORANGE:
                return player_id
    return None


def _decode_resource_enums(values: object) -> tuple[Resource, ...]:
    if not isinstance(values, list):
        return tuple()
    resources: list[Resource] = []
    for value in values:
        try:
            resource = RESOURCE_ENUM_TO_RESOURCE[int(value)]
        except (KeyError, TypeError, ValueError):
            continue
        resources.append(resource)
    return tuple(resources)


def _numeric_key(item: tuple[object, object]) -> tuple[int, str]:
    key = str(item[0])
    return (_extract_first_integer(key) or 0, key)


def _extract_first_integer(text: str) -> Optional[int]:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def _unique_in_order(values) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _current_turn_window(events: tuple[ColonistGameEvent, ...] | list[ColonistGameEvent]) -> list[ColonistGameEvent]:
    events_list = list(events)
    for index in range(len(events_list) - 1, -1, -1):
        if events_list[index].event_type in {ColonistEventType.END_TURN, ColonistEventType.TURN_START}:
            return events_list[index + 1 :]
    return events_list


def _infer_setup_context(
    events: tuple[ColonistGameEvent, ...] | list[ColonistGameEvent],
) -> InferredTurnContext:
    build_events = [
        event
        for event in events
        if event.event_type == ColonistEventType.BUILD and event.piece in {"settlement", "road"}
    ]
    if not build_events:
        return InferredTurnContext(current_player=None, phase=None, dice_rolled_this_turn=None)

    if any(
        event.event_type
        in {
            ColonistEventType.ROLL,
            ColonistEventType.PLAY_DEV_CARD,
            ColonistEventType.BUY_DEV_CARD,
            ColonistEventType.TRADE_OFFER,
            ColonistEventType.TRADE_ACCEPT,
            ColonistEventType.MARITIME_TRADE,
            ColonistEventType.END_TURN,
            ColonistEventType.MOVE_ROBBER,
            ColonistEventType.STEAL_CARD,
            ColonistEventType.DISCARD,
            ColonistEventType.DISCARD_REQUEST,
            ColonistEventType.CLAIM_ACHIEVEMENT,
            ColonistEventType.TRANSFER_ACHIEVEMENT,
            ColonistEventType.GAME_OVER,
        }
        for event in events
    ):
        return InferredTurnContext(current_player=None, phase=None, dice_rolled_this_turn=None)

    roads = [event for event in build_events if event.piece == "road"]
    settlements = [event for event in build_events if event.piece == "settlement"]
    if len(roads) > 8 or len(settlements) > 8:
        return InferredTurnContext(current_player=None, phase=None, dice_rolled_this_turn=None)
    if len(settlements) not in {len(roads), len(roads) + 1}:
        return InferredTurnContext(current_player=None, phase=None, dice_rolled_this_turn=None)

    setup_step = len(roads)
    if setup_step >= 8:
        return InferredTurnContext(current_player=0, phase=TurnPhase.PRE_ROLL, dice_rolled_this_turn=False, setup_step=8)

    if len(settlements) == len(roads) + 1 and settlements:
        return InferredTurnContext(
            current_player=settlements[-1].player_id,
            phase=TurnPhase.SETUP,
            dice_rolled_this_turn=False,
            setup_step=setup_step,
        )

    current_player = SETUP_ORDER[setup_step]
    return InferredTurnContext(
        current_player=current_player,
        phase=TurnPhase.SETUP,
        dice_rolled_this_turn=False,
        setup_step=setup_step,
    )


def _next_player_id(
    player_id: Optional[int],
    color_to_player: Optional[dict[PlayerColor, int]],
) -> Optional[int]:
    if player_id is None:
        return None
    n_players = max(len(color_to_player) if color_to_player is not None else 4, 1)
    return (player_id + 1) % n_players


def _format_resource_multiset(resources: tuple[Resource, ...]) -> str:
    if not resources:
        return ""
    counts: dict[Resource, int] = {}
    for resource in resources:
        counts[resource] = counts.get(resource, 0) + 1
    parts = []
    for resource in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE):
        count = counts.get(resource, 0)
        if count:
            parts.append(f"{count} {resource.value}")
    return ", ".join(parts)
