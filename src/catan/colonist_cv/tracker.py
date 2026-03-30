"""Observation-to-state reconstruction for the Colonist CV pipeline."""

from __future__ import annotations

from typing import Optional

from ..full_solver.rules import refresh_public_state
from ..full_solver.state import (
    DevCardType,
    ExactGameState,
    PrivatePlayerState,
    PublicPlayerState,
    empty_hand,
    make_bank_resources,
    make_dev_deck,
)
from .schema import PrivateObservation, PublicStructures, VisionFrameObservation


def _public_player_from_structures(player: PublicStructures) -> PublicPlayerState:
    return PublicPlayerState(
        player_id=player.player_id,
        settlements=player.settlements,
        cities=player.cities,
        roads=player.roads,
        hand_size=0,
        visible_vp=player.visible_vp or 0,
        played_knights=int(player.played_knights),
        dev_cards_bought=int(player.dev_cards_bought),
        settlements_left=5 - len(player.settlements),
        cities_left=4 - len(player.cities),
        roads_left=15 - len(player.roads),
    )


def _private_player_from_observation(observation: PrivateObservation) -> PrivatePlayerState:
    canonical = observation.canonical()
    return PrivatePlayerState(
        player_id=canonical.player_id,
        resources=canonical.resources,
        dev_cards_in_hand=canonical.dev_cards_in_hand,
        new_dev_cards_in_hand=canonical.new_dev_cards_in_hand,
        hidden_vp_cards=canonical.hidden_vp_cards,
    )


def _empty_private_player(player_id: int) -> PrivatePlayerState:
    return PrivatePlayerState(
        player_id=player_id,
        resources=empty_hand(),
        dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
        new_dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
        hidden_vp_cards=0,
    )


def build_state_from_observation(
    observation: VisionFrameObservation,
    previous_state: Optional[ExactGameState] = None,
) -> ExactGameState:
    """
    Convert a mapped frame observation into the solver's exact state container.

    Hidden state is preserved from `previous_state` when available. When booting
    from a single frame, opponent private state, bank counts, and deck order are
    necessarily approximate until live tracking fills them in.
    """

    n_players = len(observation.public_players)
    public_players = tuple(_public_player_from_structures(player) for player in observation.public_players)

    private_players = []
    private_override = observation.private_pov
    for player_id in range(n_players):
        if previous_state is not None and player_id < len(previous_state.private_players):
            private_player = previous_state.private_players[player_id].canonical()
        else:
            private_player = _empty_private_player(player_id)
        if private_override is not None and player_id == private_override.player_id:
            private_player = _private_player_from_observation(private_override)
        private_players.append(private_player)

    state = ExactGameState(
        board=observation.board,
        public_players=public_players,
        private_players=tuple(private_players),
        current_player=observation.current_player,
        phase=observation.phase,
        robber_hex=observation.robber_hex,
        bank_resources=previous_state.bank_resources if previous_state is not None else make_bank_resources(),
        dev_deck=previous_state.dev_deck if previous_state is not None else make_dev_deck(seed=0),
        pending_trade=observation.pending_trade,
        turn_number=observation.turn_number,
        setup_step=observation.setup_step,
        pending_setup_vertex=observation.pending_setup_vertex,
        pending_discarders=observation.pending_discarders,
        free_roads_remaining=observation.free_roads_remaining,
        dev_card_played_this_turn=observation.dev_card_played_this_turn,
        dice_rolled_this_turn=observation.dice_rolled_this_turn,
        last_roll=observation.last_roll,
        winner_id=observation.winner_id,
    )
    return refresh_public_state(state)


class ColonistVisionTracker:
    """Stateful bridge that preserves hidden information across CV snapshots."""

    def __init__(self, initial_state: Optional[ExactGameState] = None):
        self._state = initial_state

    @property
    def state(self) -> Optional[ExactGameState]:
        return self._state

    def reset(self) -> None:
        self._state = None

    def ingest(self, observation: VisionFrameObservation) -> ExactGameState:
        self._state = build_state_from_observation(observation, previous_state=self._state)
        return self._state
