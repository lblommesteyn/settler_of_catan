"""Hybrid information-set search scaffold for 4-player Catan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .action_space import Action, group_actions_by_type
from .belief import BeliefState, BeliefTracker
from .policy import OpponentModel, PolicyValueModel
from .state import Observation, PayoffVector, PlayerId, RulesEngine


@dataclass(frozen=True)
class SearchConfig:
    root_simulations: int = 800
    max_depth: int = 64
    max_branch_per_type: int = 12
    c_puct: float = 1.5
    rollout_temperature: float = 0.8
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_mix: float = 0.25


@dataclass(frozen=True)
class ActionStats:
    visits: int
    q_value: float
    prior: float


@dataclass(frozen=True)
class SearchResult:
    selected_action: Action
    root_value: float
    action_stats: dict[Action, ActionStats]


@dataclass
class SearchNode:
    player_id: PlayerId
    visits: int = 0
    value_sum: float = 0.0
    children: dict[Action, "SearchNode"] = field(default_factory=dict)

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class HybridInformationSetSolver:
    """
    The intended decision-time planner for the ideal 4-player solver.

    High-level algorithm:

    1. Build an observation and belief state for the acting player.
    2. Sample hidden-state determinizations from the belief tracker.
    3. Expand only the most promising action types and options using policy
       priors and branching caps.
    4. Evaluate leaves with the value head or short tactical rollouts.
    5. Back up a 4-player payoff vector through the tree.
    6. Choose the root action that maximizes the acting player's backed-up EV.

    This class is a blueprint only; it pins down the dependencies and the
    intended search behavior before implementation work starts in `src/`.
    """

    def __init__(
        self,
        engine: RulesEngine,
        belief_tracker: BeliefTracker,
        policy_model: PolicyValueModel,
        opponent_model: OpponentModel,
        config: SearchConfig | None = None,
    ) -> None:
        self.engine = engine
        self.belief_tracker = belief_tracker
        self.policy_model = policy_model
        self.opponent_model = opponent_model
        self.config = config or SearchConfig()

    def select_action(
        self,
        observation: Observation,
        belief_state: BeliefState,
    ) -> SearchResult:
        """
        Select an action from an information set.

        Expected implementation:

        - get legal actions from the engine
        - summarize the belief for the learned model
        - use priors to rank action types and per-type options
        - run root-sampled information-set simulations
        - return root visit statistics and the selected action
        """
        legal_actions = self.engine.legal_actions(self._determinize_placeholder())
        belief_summary = self.belief_tracker.summarize(belief_state)
        inference = self.policy_model.infer(observation, belief_summary, legal_actions)

        grouped = group_actions_by_type(legal_actions)
        pruned_actions = self._top_actions_by_type(grouped, inference.action_priors)
        if not pruned_actions:
            raise RuntimeError("Search received no legal actions.")

        selected = max(pruned_actions, key=lambda action: inference.action_priors.get(action, 0.0))
        action_stats = {
            action: ActionStats(visits=0, q_value=0.0, prior=inference.action_priors.get(action, 0.0))
            for action in pruned_actions
        }
        return SearchResult(
            selected_action=selected,
            root_value=inference.state_value,
            action_stats=action_stats,
        )

    def _top_actions_by_type(
        self,
        grouped_actions: dict,
        priors: dict[Action, float],
    ) -> list[Action]:
        """Cap the branching factor independently inside each action type."""
        kept: list[Action] = []
        for actions in grouped_actions.values():
            ranked = sorted(actions, key=lambda action: priors.get(action, 0.0), reverse=True)
            kept.extend(ranked[: self.config.max_branch_per_type])
        return kept

    def _determinize_placeholder(self):
        """
        Placeholder used by the blueprint.

        Real implementation should sample a full hidden state from the belief
        tracker and expose it only to the engine, never to the acting policy.
        """
        raise NotImplementedError("Hook this solver up to a real determinization path.")
