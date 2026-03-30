Product Requirements Document
Product name

Catan Opening Intelligence

One-line summary

A system that evaluates Catan opening settlement placements using historical game data and large-scale simulation, then explains why certain openings are strong.

Goal

Build a data-driven model that predicts the quality of an opening settlement pair in standard 4-player Catan, measured primarily by eventual win probability, and secondarily by downstream game quality metrics like early production, expansion success, and tempo.

Core user problem

Most players rely on rough heuristics:

total pips
resource diversity
“must have brick/wood”
“ore/wheat is good”
“port starts can work”

Those heuristics are useful, but incomplete. Opening strength also depends on:

seat order,
snake draft interaction,
expansion lanes,
board topology,
port timing,
robber vulnerability,
and how contested your intended growth path is.

There is no single clean model that combines all of these into a general opening evaluator. That is the gap this project fills.

Target users

Primary:

serious Catan players
board game analytics people
ML / DS portfolio reviewers
competition judges / hackathon judges

Secondary:

casual players who want a placement assistant
board game researchers
game AI developers
Success criteria

The project is successful if it does three things:

Predictive value
It predicts opening quality materially better than naive heuristics like raw pip count or simple diversity score.
Interpretability
It explains opening strength in understandable terms:
“This opening is strong because it preserves two expansion lanes, activates a 3:1 port quickly, and avoids overexposure to the central 6-hex robber.”
Generalization
It works across many board layouts and seat positions, not just memorized patterns from a few boards.
Scope
In scope
Standard 4-player base Catan
Randomized standard boards
Opening phase: first two settlements + first two roads
Seat order effects
Board-state encoding
Win probability modeling
Feature engineering for opening evaluation
Human-data analysis
Simulation-based counterfactuals
Interactive opening scorer
Out of scope for V1
Seafarers / Cities & Knights
5–6 player extension
Full trade-language modeling
Hidden-information negotiation modeling
Real-time voice/chat parsing
Perfect game-solving of full Catan
Product vision

The final system should let a user input or upload a board and get:

Ranked legal opening pairs
Predicted win probability by seat
Feature-based explanation
Comparison to human heuristics
Suggested placement archetype
balanced expansion
ore/wheat/dev
road race
port engine
high-risk high-roll
Counterfactual analysis
“If you give up 2 pips for a better port lane, your projected win rate rises by 1.8%”
“This spot looks strong on production but has high blocking risk”
Key research questions
Primary question

What makes an opening settlement placement actually strong in standard Catan?

Secondary questions
How much does seat order change the value of the same opening?
Does raw pip count overestimate openings with poor expansion lanes?
When does resource diversity beat concentrated strength?
How much is early port access worth?
Which features matter most: production, diversity, connectivity, or uncontested growth?
Do strong human players choose openings that match model-optimal openings?
Can a simple interpretable model get close to a more complex ML model?
Hypotheses

You want explicit hypotheses because they give the project a real research backbone.

H1

Raw pip count alone is a weak predictor of opening quality compared with a richer model including diversity, connectivity, and expansion features.

H2

The best openings differ significantly by seat position because of snake-draft placement order and contest dynamics.

H3

Openings with slightly lower production but better expansion optionality outperform “greedy pip” starts over large samples.

H4

Port value is highly nonlinear: ports matter most when paired with concentrated production and fast activation paths.

H5

Model-derived opening rankings will beat common human heuristics on held-out boards.

Data you could use

This is the most important part.

1. Best real-world dataset: public Colonist.io-derived game dataset

A public GitHub dataset now exists with 43,947 anonymized four-player online Catan games, one JSON per game, with the full starting board state, play order, event history, and end-game winner. The repo says the files include initialState, playOrder, the event stream, and final winner/rank information, with game events like builds, trades, dev card plays, robber moves, and dice-resource receipts. It is standard 4-player Catan and about 6.9 GB uncompressed.

Why this is huge:

enough scale for real supervised learning
includes real human or human-like online behavior
gives you actual opening choices and actual outcomes
lets you compare observed openings to model-evaluated alternatives

This should be your backbone dataset.

2. Smaller Kaggle datasets

There are smaller Kaggle datasets for Catan game records, including one specifically described as “Game Records and Statistics,” plus older personal datasets with around 50 games and analyses built from similarly small samples. These are useful for prototyping schemas and EDA, but they are far too small for a serious general opening model.

Use these only if you want:

fast prototyping,
demo notebooks,
or baseline exploration.
3. Simulation data you generate yourself

You should absolutely generate synthetic data too.

Why:

the human dataset only contains chosen openings, not all legal alternatives
you need counterfactuals: what would have happened if the player had opened elsewhere?
you want coverage across board topologies and legal opening combinations

A Python simulator like PyCatan or a similar Catan AI environment can be used to run self-play / bot-play rollouts and save traces in JSON. One simulator repo explicitly says it is for AI agents, supports custom agents, and saves game traces in JSON for later analysis.

4. RL / AI implementations for inspiration, not as your primary data

There are existing RL efforts that built full Catan simulators and state encodings because Catan is a difficult 4-player imperfect-information environment with trading, hidden information, and long horizons. One public writeup explains a custom simulator, player state features, public-information bounds on opponent resources, and the complexity of learning in full 4-player Catan. That is useful as architectural inspiration for your state representation.

Recommended data strategy

Use a hybrid data pipeline:

Dataset A: observed real games

From the 43,947-game dataset:

board layout
number tokens
ports
seat order
opening placements
game winner
early event sequence
final points / rank

This teaches:

what humans actually pick
which real openings win
seat-conditioned empirical priors
Dataset B: counterfactual simulation

For each real board:

enumerate legal opening pairs
simulate bot games from many candidate openings
estimate opening EV / win probability

This teaches:

what could have happened
opening quality independent of human selection bias
Dataset C: generated random boards

Randomly generate boards and run simulations over all or sampled legal openings.

This teaches:

generalization to unseen boards
robustness to board-type distribution shifts
Exact problem framing
Problem type

Primary:

supervised learning / ranking

Secondary:

simulation-based decision analysis
causal-ish comparative evaluation
graph feature engineering
Prediction target

You have a few options.

Main target for V1

Opening win probability

binary label: did this player win?
or multiclass: final rank 1–4
or expected final VP / finish percentile
Better auxiliary targets

Use several:

win probability
top-2 finish probability
VP at turn 30
first city timing
first expansion success
longest road acquisition probability
dev-card pivot likelihood

These auxiliary outcomes help explain why an opening is good.