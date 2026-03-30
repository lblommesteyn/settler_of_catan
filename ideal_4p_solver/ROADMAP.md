# Ideal 4-Player Solver Roadmap

## Stage 0: Hard reset on scope

Goal:

- stop treating the current opening simulator as a candidate full solver

Deliverables:

- exact rules checklist
- action taxonomy
- public/private state schema

## Stage 1: Exact engine

Build first:

- official turn sequencing
- full dev-card effects
- exact longest-road computation
- exact largest-army transitions
- exact discard / robber / steal behavior
- maritime trade
- structured domestic trade legality
- bank resource scarcity behavior

Exit criteria:

- deterministic replay of curated rule fixtures
- explicit regression tests for FAQ edge cases

## Stage 2: Observation and belief

Build:

- acting-player observation object
- belief tracker over hidden resources and dev cards
- particle resampling and consistency checks

Exit criteria:

- belief tracker stays consistent on replayed real games
- impossible hidden states are eliminated after public actions

## Stage 3: Action abstraction

Build:

- action-type-first enumeration
- top-k target generation per type
- structured trade templates

Exit criteria:

- legal action counts stay computationally bounded
- search can run without trade action explosion

## Stage 4: Baseline search

Build:

- root-sampled information-set search
- vector-valued backup statistics
- policy-prior guided expansion
- leaf rollout or value-head evaluation

Exit criteria:

- solver beats heuristic bots on fixed seeds
- search remains stable across all seats

## Stage 5: Learned models

Build:

- policy/value network
- auxiliary heads for tactical forecasts
- opponent-model heads for trade and robber choices

Exit criteria:

- learned priors improve search efficiency
- value head calibrates well on held-out games

## Stage 6: Training

Build:

- supervised warm start from real traces
- self-play data generation
- checkpoint league evaluation
- exploitability-style restricted-game tests where possible

Exit criteria:

- new checkpoints beat previous league median
- no regression on seat-conditioned win rate

## Stage 7: Trade realism

Build:

- offer generation policy
- acceptance model per opponent
- search-time trade expectation

Exit criteria:

- trade-enabled agent beats no-trade ablations
- search does not spam dominated offers

## Stage 8: Production-quality evaluation

Track:

- overall win rate
- win rate by seat
- win rate by board class
- regret against hindsight-best opening
- average game length
- calibration of value estimates
- tactical conversion rates for longest road / largest army / dev pivots

The solver is only worth calling "ideal" if it is strong across all seats and
board classes, not just good at greedy opening production.
