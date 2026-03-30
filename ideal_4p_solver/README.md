# Ideal 4-Player Catan Solver

This folder is a concrete blueprint for the strongest practical 4-player Catan
solver worth building in this repo.

It is not an exact "solve" in the chess/checkers sense. Standard 4-player Catan
is a stochastic, imperfect-information, general-sum game with negotiation, so
an exact equilibrium computation for the full ruleset is not a realistic target.
The right target is an approximate information-set solver with learned priors,
belief tracking, structured trading, and online search.

## Target

Given a public game state, a private hand for the acting player, and a history
of public actions, choose the action that maximizes:

- win probability
- seat-robust long-run EV across a league of strong opponents
- low regret under board, seat, and negotiation variation

Mathematically, the acting player optimizes over an information set `I_t`, not a
fully observed state:

- choose `a_t = argmax_a E[u_i | I_t, a, pi_-i]`
- where `u_i` is terminal win utility or a calibrated payoff proxy
- and the expectation integrates over hidden cards, dice, dev deck order, and
  opponent behavior

## What The Ideal Solver Needs

- Exact rules engine for the official base game and FAQ edge cases
- Public/private state split with legal action generation
- Hidden-information belief tracker over opponent resources and dev cards
- Structured domestic-trade model, not free-form chat
- Hierarchical action abstraction to control branching
- Policy/value network with auxiliary heads
- Information-set search at decision time
- Population self-play and league evaluation

## Recommended Solver Stack

### 1. Exact engine

The engine must model:

- full dev-card deck and timing restrictions
- exact longest-road path logic
- largest army transitions
- robber discard, move, and steal rules
- bank depletion and resource-shortage rules
- maritime trade, structured domestic trade, and turn sequencing
- hidden VP cards and win-on-your-turn timing

### 2. Belief layer

The agent should maintain a particle or factored belief over:

- opponent resource hands
- opponent dev-card hands
- plausible future trade willingness

Beliefs are updated from:

- dice outcomes
- visible builds
- bank trades
- robber steals
- dev-card purchases and plays
- trade offers, accepts, and rejects

### 3. Action abstraction

Catan has a badly imbalanced action space because trade options dominate raw
enumeration. Search should choose:

1. action type
2. parameterized option inside that type

Typical action types:

- roll / discard / move robber / steal
- build road / settlement / city
- buy dev / play knight / monopoly / road building / year of plenty
- maritime trade
- domestic trade offer / accept / reject
- end turn

### 4. Policy/value model

The model should consume:

- board tensors or graph features
- public player state
- private acting-player state
- belief summaries over opponents
- phase flags and legal-action masks

And produce:

- policy prior over legal actions
- state value / win probability
- auxiliary heads:
- ETA-to-win
- probability of securing longest road
- probability of securing largest army
- trade acceptance likelihood
- opponent robber-target likelihood

### 5. Online search

The strongest practical solver here is a hybrid:

- root-sampled belief determinization
- information-set MCTS / POMCP-style planning
- policy prior for progressive widening
- learned value head for leaf evaluation
- vector-valued backups for 4-player general-sum outcomes

The important point is that backups should retain a 4-player payoff vector,
while action choice at each node is made from the current player's component.

### 6. Training pipeline

The training order should be:

1. supervised warm start on human traces
2. opponent-model pretraining
3. restricted-trade self-play
4. full structured-trade self-play
5. league training against historical checkpoints and scripted exploiters

## Why This Folder Exists

The current repo already has useful pieces for openings:

- board representation
- opening features
- learned opening scorers
- Colonist-derived data parsing

But the current simulator is intentionally simplified and not suitable as the
core of a full-game solver. This folder defines the missing architecture before
we wire any of it into `src/`.

## File Guide

- `state.py`: exact solver state model and engine interfaces
- `action_space.py`: parameterized action definitions and hierarchical grouping
- `belief.py`: belief particles, summaries, and tracker interfaces
- `policy.py`: policy/value and opponent-model interfaces
- `search.py`: hybrid information-set search skeleton
- `training.py`: self-play, league, and data-generation skeleton
- `ROADMAP.md`: staged build order

## Design Principles

- Keep rules exact before making the policy clever
- Separate public state, private state, and beliefs cleanly
- Treat trading as first-class, but structured
- Use search to handle tactics and learned models to handle priors
- Measure strength by league win rate, not one-off anecdotes
