"""Policy/value and opponent-model interfaces for the ideal solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .action_space import Action
from .belief import BeliefSummary
from .state import Observation, PlayerId


@dataclass(frozen=True)
class AuxiliaryPredictions:
    eta_to_win: float | None = None
    longest_road_prob: float | None = None
    largest_army_prob: float | None = None
    trade_acceptance_prob: float | None = None
    robber_target_probs: dict[PlayerId, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyInference:
    """
    Model output consumed by search.

    `action_priors` is sparse and only needs to cover the current legal actions.
    """

    action_priors: dict[Action, float]
    state_value: float
    auxiliary: AuxiliaryPredictions


class PolicyValueModel(Protocol):
    """
    Learned policy/value model used for priors and leaf evaluation.

    The model should condition on:
    - public board/game state
    - private acting-player hand
    - compact belief summaries over opponents
    - action masks
    """

    def infer(
        self,
        observation: Observation,
        belief: BeliefSummary,
        legal_actions: list[Action],
    ) -> PolicyInference:
        """Return priors, value, and auxiliary predictions."""


class OpponentModel(Protocol):
    """
    Predictive model for non-stationary multiplayer behavior.

    This should cover the places where a naive fixed opponent prior performs
    badly in Catan: trades, robber targeting, and race commitments.
    """

    def trade_acceptance_probability(
        self,
        responder_id: PlayerId,
        observation: Observation,
        offer: Action,
    ) -> float:
        """Predict whether one opponent will accept a proposed trade."""

    def robber_target_distribution(
        self,
        acting_player: PlayerId,
        observation: Observation,
    ) -> dict[PlayerId, float]:
        """Predict likely victims when the robber is moved."""
