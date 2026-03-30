"""Training and evaluation scaffold for the ideal 4-player solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .action_space import Action
from .belief import BeliefState
from .state import Observation, PayoffVector


@dataclass(frozen=True)
class TrainingExample:
    observation: Observation
    belief_state: BeliefState
    legal_actions: tuple[Action, ...]
    improved_policy: dict[Action, float]
    terminal_payoff: PayoffVector


@dataclass(frozen=True)
class LeagueConfig:
    games_per_matchup: int = 200
    seats_per_board: bool = True
    board_reuse_factor: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    supervised_warm_start_epochs: int = 4
    self_play_games_per_round: int = 512
    policy_update_rounds: int = 100
    checkpoint_interval: int = 5
    league: LeagueConfig = field(default_factory=LeagueConfig)


class EpisodeGenerator(Protocol):
    """Produces search-improved training examples from self-play."""

    def generate_round(self, num_games: int) -> list[TrainingExample]:
        """Run self-play and return learning targets."""


class LeagueEvaluator(Protocol):
    """Evaluates checkpoints against historical and scripted baselines."""

    def evaluate(self) -> dict[str, float]:
        """Return aggregated league metrics."""


class IdealSolverTrainer:
    """
    Intended training loop for the solver stack.

    Order of operations:

    1. warm start policy/value model on human traces
    2. fit opponent models from public histories
    3. run self-play with search-improved policies
    4. evaluate new checkpoints in a seat-balanced league
    5. keep only checkpoints that improve league strength
    """

    def __init__(
        self,
        episode_generator: EpisodeGenerator,
        league_evaluator: LeagueEvaluator,
        config: TrainingConfig | None = None,
    ) -> None:
        self.episode_generator = episode_generator
        self.league_evaluator = league_evaluator
        self.config = config or TrainingConfig()

    def train_round(self) -> dict[str, float]:
        """
        Skeleton training round.

        The actual implementation should generate self-play data, update model
        weights, then run league evaluation before promoting a new checkpoint.
        """
        _ = self.episode_generator.generate_round(self.config.self_play_games_per_round)
        return self.league_evaluator.evaluate()
