"""Belief-state scaffolding for hidden-information reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .action_space import Action
from .state import DevCardType, FullGameState, Observation, PlayerId, Resource, ResourceCounts


@dataclass(frozen=True)
class HiddenPlayerHypothesis:
    resources: ResourceCounts
    dev_cards_in_hand: dict[DevCardType, int]
    hidden_vp_cards: int


@dataclass(frozen=True)
class BeliefParticle:
    """
    One sampled hidden-state completion consistent with public evidence.

    The public state is shared across particles; only hidden player inventories
    and deck order differ.
    """

    hidden_players: dict[PlayerId, HiddenPlayerHypothesis]
    remaining_dev_deck: tuple[DevCardType, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class BeliefSummary:
    """
    Compact statistics fed to the policy/value model.

    This is the learned-model view of the belief state, not the full particle
    set required by search.
    """

    expected_resources: dict[PlayerId, dict[Resource, float]]
    expected_dev_cards: dict[PlayerId, dict[DevCardType, float]]
    hidden_vp_expectation: dict[PlayerId, float]


@dataclass
class BeliefState:
    actor_id: PlayerId
    particles: list[BeliefParticle] = field(default_factory=list)

    def normalized(self) -> "BeliefState":
        total = sum(p.weight for p in self.particles) or 1.0
        self.particles = [
            BeliefParticle(
                hidden_players=p.hidden_players,
                remaining_dev_deck=p.remaining_dev_deck,
                weight=p.weight / total,
            )
            for p in self.particles
        ]
        return self


class BeliefTracker(Protocol):
    """
    Hidden-information tracker used by search and learning.

    A practical implementation should use exact constraints where possible and a
    particle filter where exact counting becomes intractable.
    """

    def initialize(self, observation: Observation) -> BeliefState:
        """Return an initial belief state for a new game."""

    def update(
        self,
        prior: BeliefState,
        previous_observation: Observation,
        public_action: Action,
        next_observation: Observation,
    ) -> BeliefState:
        """Condition the belief on a public transition."""

    def sample_determinization(self, belief: BeliefState, seed: int | None = None) -> BeliefParticle:
        """Sample one hidden-state completion for root sampling."""

    def summarize(self, belief: BeliefState) -> BeliefSummary:
        """Convert particles to compact model features."""
