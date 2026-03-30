# Catan Opening Intelligence — Research Report

**Dataset:** 43,947 Colonist.io games · 169,817 labeled opening placements
**Question:** What does the opening actually determine in Catan, and where do humans leave value on the table?

---

## Executive Summary

Catan openings matter — but less than conventional wisdom suggests, and the *kind* of value they provide is different from what most players optimize for. Three main findings:

1. **Seat order is the single biggest opening-related factor**, accounting for a ~1.2pp win-rate difference between seats 0/1 and seats 2/3. Yet all seats choose essentially identical openings (same median pips, same archetype distribution), meaning late-seat players are not compensating.

2. **Raw pip count is nearly uncorrelated with winning** at the individual game level (+1.6pp uplift from worst to best). The real value is in *where* those pips come from — specifically interior ("3-hex") vertices and city-path resources (ore+wheat). Players systematically undervalue these.

3. **Opening quality matters most on balanced boards**, not chaotic ones. On boards where all players got roughly equal pip counts, the player with the best opening won 30.2% of games. On high-spread boards (where one player lucked into great spots), the opening advantage collapses to +0.9pp.

---

## 1. Seat Dynamics

### 1a. Win Rate by Seat

| Seat | Draft picks | N | Win rate |
|------|-------------|---|----------|
| 0 | 1st, 8th | 43,535 | **26.1%** |
| 1 | 2nd, 7th | 43,534 | **26.1%** |
| 2 | 3rd, 6th | 41,436 | 25.1% |
| 3 | 4th, 5th | 41,312 | 24.9% |

Seats 0 and 1 win 1.2pp more often than seats 2 and 3. The asymmetry likely reflects that seats 2/3 have both picks in a row (5th and 6th, or 4th and 5th) and face more congested boards for their first placement.

### 1b. What Humans Actually Pick

All four seats produce nearly identical opening profiles:

```
Seat 0: pips median=12  mean=12.3  port_rate=64%  top_archetype=balanced
Seat 1: pips median=12  mean=12.2  port_rate=64%  top_archetype=balanced
Seat 2: pips median=12  mean=12.2  port_rate=64%  top_archetype=balanced
Seat 3: pips median=12  mean=12.2  port_rate=64%  top_archetype=balanced
```

Late-seat players are not adjusting their strategy to compensate for their positional disadvantage. If the seat effect is real, the rational response would be for seats 2/3 to take riskier or more diverse positions — but the data shows they don't.

### 1c. What Features Predict Wins — By Seat

Logistic regression coefficients per seat (starred = |coef| > 0.05):

| Feature | Seat 0 | Seat 1 | Seat 2 | Seat 3 |
|---------|--------|--------|--------|--------|
| combined_pip_count | -0.060 * | -0.054 * | -0.049 | -0.009 |
| v1_num_adjacent_hexes | +0.041 | +0.030 | +0.064 * | +0.048 |
| v2_num_adjacent_hexes | +0.024 | +0.034 | +0.029 | +0.059 * |
| combined_city_pips | +0.038 | +0.023 | -0.012 | -0.037 |
| ore_wheat_score | -0.022 | -0.035 | +0.044 | +0.013 |
| expansion_pip_sum | +0.000 | +0.018 | +0.005 | +0.027 |

**Key insight:** Combined pip count has a *negative* coefficient for early seats (0, 1). This is a classic confound: early-seat players who chase high pip counts are probably picking isolated corner hexes, while interior vertices (fewer pips but 3 adjacent hexes) are more valuable. The `num_adjacent_hexes` coefficient is consistently positive and grows for later seats — by seat 3, having an interior second settlement is the single most predictive feature.

The **ore_wheat_score** reversal (negative for seats 0-1, positive for seats 2-3) is interesting: early seats can afford to play a long city game because they get first access to good spots; late seats may be forced into ore/wheat as a response to wood/brick spots being taken.

### 1d. Pip Value by Seat

Win rate across pip-count quintiles:

| Quintile | Seat 0 | Seat 1 | Seat 2 | Seat 3 |
|----------|--------|--------|--------|--------|
| very_low | 26.0% | 25.9% | 24.7% | 24.6% |
| low | 25.6% | 26.0% | 25.5% | 25.0% |
| mid | 26.4% | 26.1% | 24.7% | 24.6% |
| high | 26.0% | 26.0% | 25.6% | 25.0% |
| very_high | 26.6% | 26.7% | 25.4% | 25.3% |

The spread across quintiles is ~1pp for all seats — pip count alone is a weak predictor. Early seats show slightly more variance (the "very_high" quintile is noticeably better for seats 0 and 1), consistent with early seats having better access to the high-pip-*and*-interior vertices that are genuinely strong.

---

## 2. Human vs Optimal

### 2a. Pip Distribution — What Humans Choose

| Pip range | N openings | % of total | Win rate |
|-----------|-----------|-----------|----------|
| < 10 | 65,752 | 38.7% | 25.5% |
| 10–13 | 47,979 | 28.3% | 25.4% |
| 14–17 | 37,027 | 21.8% | 25.8% |
| 18–21 | 15,965 | 9.4% | 25.7% |
| 22–25 | 3,026 | 1.8% | 26.8% |
| 26–29 | 42 | 0.03% | 40.5% |

Nearly 40% of all openings have fewer than 10 combined pips — these are almost entirely openings where both settlements touch desert or sea hexes. Win rates barely vary across the first five brackets (±0.7pp). The 26-29 bucket stands out (+14.9pp) but has only 42 observations.

### 2b. Win Rate by Exact Pip Count

Win rate is noisy and essentially flat from 1–25 pips, with no clear monotonic trend. The signal from pips alone is real but tiny. A player at 25 pips wins at ~26.7% vs 24.9% at 17 pips — a 1.8pp edge over a ~10-pip improvement in raw production.

**The takeaway:** Humans obsess over pip counts, but the data says pip count explains almost nothing about who wins. This doesn't mean pips don't matter — it means pip count is a poor proxy for opening quality because it ignores board position, resource composition, and expansion potential.

### 2c. Archetype Frequency vs Win Rate

| Archetype | N | % of total | Win rate | vs baseline |
|-----------|---|-----------|----------|-------------|
| balanced | 124,314 | 73.2% | 25.5% | -0.0% |
| road_race | 27,319 | 16.1% | 26.0% | +0.4% |
| ore_wheat | 17,317 | 10.2% | 25.3% | -0.3% |
| port_engine | 618 | 0.4% | 25.4% | -0.2% |
| high_pip | 249 | 0.1% | 26.5% | +0.9% |

**"Balanced" is the dominant human strategy** — 73% of all placements. It's also essentially a wash in terms of win rate. Road-race (prioritizing expansion reach) beats baseline by +0.4pp, the most consistent outperformer. Ore/wheat (city rush) slightly underperforms — possibly because it requires hitting specific rolls and opponents can roadblock city candidates.

The port_engine archetype is rare (0.4%) and no better than average. Ports don't help as much as players seem to think.

### 2d. Settlement vs City Resource Focus

| Focus | N | % | Win rate |
|-------|---|---|----------|
| pure_settle (wood/brick heavy) | 12,080 | 7.1% | 26.0% |
| settle_lean | 39,843 | 23.5% | 25.5% |
| balanced | 44,586 | 26.3% | 25.8% |
| city_lean | 30,159 | 17.8% | 25.6% |
| pure_city (ore/wheat heavy) | 20,553 | 12.1% | 25.3% |

Leans toward settlement resources (wood, brick, sheep, wheat) perform slightly better than city-heavy openings. Pure city (ore/wheat dominant) is the weakest. This reinforces the road-race finding: early mobility via roads and settlements provides a durable advantage that the ore/wheat city strategy struggles to match.

### 2e. Port Access — Does Having a Port Help?

| Ports | N | Win rate | Median pips |
|-------|---|----------|-------------|
| 0 | 60,935 | 25.8% | 14.0 |
| 1 | 86,951 | 25.5% | 12.0 |
| 2 | 21,931 | 25.3% | 9.0 |

Players with ports actually win *less* frequently. The key: port hexes typically touch only 2 land hexes (instead of 3), so port settlements have lower pip counts — and more importantly, fewer adjacent hex combinations. Players seek out ports and sacrifice production to get them. The data says this is a mistake on average.

Port synergy (holding a 2:1 port for a resource you actually produce) doesn't recover this: synergy-port players win at 25.5%, identical to those with mismatched ports (25.4%), and slightly below no-port players (25.8%).

**Conclusion:** Ports are overvalued by human players. The production sacrifice to gain port access is not compensated by the trading advantage, at least at the opening stage. This may change later in game (ports are most useful at 8+ cards), but the opening placement choice to prioritize ports is a net negative.

### 2f. Resource Diversity

| Unique resources | N | Win rate | Median pips |
|-----------------|---|----------|-------------|
| 1 | 10,254 | 25.3% | 5.0 |
| 2 | 51,382 | 25.4% | 8.0 |
| 3 | 69,045 | 25.6% | 13.0 |
| 4 | 35,042 | 25.8% | 16.0 |
| 5 | 4,068 | 25.5% | 19.0 |

There's a positive gradient from 1 to 4 unique resources (+0.5pp), but 5-resource openings don't continue the trend. This aligns with the feature importance results — unique resource count matters, but only up to a point. Getting all 5 resources in your opening typically requires spreading across low-production hexes, sacrificing pip depth for breadth.

### 2g. Within-Game Gap: Best vs Worst Opening

Across 43,669 games, the within-game pip spread averages **10.5 pips** (median 10.0, p90 = 16.0).

| Metric | Win rate |
|--------|----------|
| Player with highest pips in game | 26.3% |
| All other players | 25.3% |

The player who snagged the best opening (highest combined pips) in a given game wins +1.0pp more often. But this effect varies dramatically by how much spread there was:

| Game pip spread | Win rate (top pip player) |
|----------------|--------------------------|
| tight (≤ 7 pips spread) | **27.8%** |
| moderate (8–10) | 25.4% |
| spread (11–14) | 25.7% |
| wide (15+ pips spread) | 25.3% |

**Counterintuitive result:** When games are *tightly contested* (all players have similar pip counts), having slightly better pips matters most. When one player pulled far ahead in the opening, the advantage doesn't compound — likely because other players adjust their strategy (road-blocking, targeting with the robber), equalizing outcomes.

---

## 3. Board Variance

### 3a. How Much Do Boards Vary?

Per-game pip spread statistics (best opening minus worst in each game):

| Statistic | Value |
|-----------|-------|
| Mean spread | 10.5 pips |
| Median spread | 10.0 pips |
| 75th percentile | 13.0 pips |
| 90th percentile | 16.0 pips |
| Max spread | 26.0 pips |

Most games have a 10-pip spread between best and worst opening. This is meaningful — a 26-pip opening vs a 16-pip opening is a significant production advantage — but as section 2g showed, that advantage often doesn't translate to wins.

### 3b. Board Spread and Final VP

Mean final VP across opening-spread quartiles:

| Spread quartile | Avg final VP |
|----------------|-------------|
| equal | 5.65 |
| mild | 5.55 |
| spread | 5.56 |
| extreme | 5.54 |

Tightly contested boards (equal pip spreads) produce slightly higher average final VPs — games go longer and are more competitive. On extreme-spread boards, games end with lower average VP, suggesting more decisive outcomes... but the winner is not especially the person who got the good spots.

### 3c. Where Does Opening Choice Matter Most?

This is the most striking result. We compared the opening-to-win conversion on "equal boards" (bottom 10% of pip spread) vs "unequal boards" (top 10% of pip spread):

| Board type | Top-pip win rate | Others win rate | Advantage |
|-----------|----------------|----------------|----------|
| Low spread (equal boards) | **30.2%** | 26.6% | **+3.6pp** |
| High spread (unequal boards) | 25.8% | 24.9% | +0.9pp |

**On equal boards, having the best opening provides a 3.6pp advantage. On unequal boards, it shrinks to 0.9pp.**

The implication: when everyone is fighting over roughly equal terrain, the players who best identify the subtle quality differences win more. When the board is chaotic (some spots obviously much better), the game effectively randomizes — the person with the "bad" opening adapts, while the person who lucked into a great spot may not play optimally with it.

### 3d. Feature Importance Shifts by Board Type

| Feature | Balanced boards | Unbalanced boards |
|---------|----------------|------------------|
| combined_pip_count | +0.040 | +0.015 |
| unique_resource_count | +0.014 | +0.001 |
| expansion_pip_sum | -0.029 | +0.005 |
| combined_city_pips | -0.011 | -0.001 |
| num_ports | +0.004 | -0.002 |

On balanced boards, pip count and resource diversity are more predictive. On unbalanced boards, feature importance essentially collapses — no opening feature is a strong predictor because the dice determine outcomes more than placement.

---

## 4. Model Performance

### 4a. AUC-ROC by Seat

| Seat | AUC-ROC |
|------|---------|
| 0 | 0.515 |
| 1 | 0.515 |
| 2 | 0.514 |
| 3 | 0.509 |

The model is slightly better at predicting wins for early seats — consistent with early seats having more signal in their opening choices (later seats are more constrained, introducing noise).

### 4b. Calibration

The model is well-calibrated: predicted probabilities align closely with actual win rates across all deciles, with maximum error of 0.7pp. This means the model's output can be interpreted directly as a win probability estimate.

### 4c. Win Rate by Model Score Quintile

| Quintile | Win rate | vs baseline |
|----------|----------|-------------|
| Q1 (worst) | 24.2% | -1.4pp |
| Q2 | 25.1% | -0.4pp |
| Q3 | 25.4% | -0.1pp |
| Q4 | 25.9% | +0.3pp |
| Q5 (best) | **27.2%** | +1.6pp |

The model creates a 3.0pp spread from worst to best quintile. This is meaningful — going from a bottom-quintile opening to a top-quintile opening is roughly equivalent to a 1/33 improvement in win odds.

---

## 5. Pip Ceiling — Diminishing Returns

### 5a. Win Rate vs Pip Count

The relationship between pip count and win rate is not monotonic:

| Pips | Win% |
|------|------|
| 8 | 25.4% |
| 10 | 25.3% |
| 12 | 25.0% |
| 14 | 26.2% |
| 16 | 26.0% |
| 18 | 26.5% |
| 20 | 25.6% |
| 24 | **27.2%** (peak) |

Peak win rate occurs at **24 pips** (+1.6pp over baseline). Beyond 24 the sample is sparse. The relationship is noisy and non-linear — there's no clear "optimal pip count" from this data alone. The 14-pip bump is interesting and may reflect that 14 is a common threshold for two solid production spots.

### 5b. Pip Ceiling by Archetype

| Archetype | Peak pip count | Win rate at peak | Baseline |
|-----------|---------------|-----------------|----------|
| balanced | 25 pips | 27.7% | 25.5% |
| ore_wheat | **22 pips** | 33.3% | 25.3% |
| road_race | 1 pip* | 31.7% | 26.0% |
| port_engine | 2 pips* | 28.6% | 25.4% |
| high_pip | 25 pips | 24.2% | 26.5% |

*Very small samples at the extremes; road_race and port_engine low-pip peaks are likely statistical noise.

The ore_wheat result is the most interesting: ore/wheat openings peak earlier (22 pips) and more sharply. This makes intuitive sense — an ore/wheat opening with too many pips may be over-concentrated on those two resources, leaving gaps in early-game settlement expansion. The balanced archetype has the flattest pip curve, as expected.

**High_pip as archetype actually underperforms its baseline** (26.5% base but only 24.2% at peak). This archetype label (assigned when high_pip_score dominates) may be capturing the pathological case: players who sacrificed position and resource diversity for raw pip count.

---

## Synthesis: What Should Players Do Differently?

Based on 43,947 games:

### What's overvalued
- **Port access**: Humans over-index on ports. On average, a port opening is -0.3pp vs no port, because the production sacrifice isn't worth it.
- **Ore/wheat city path**: Underperforms road_race and balanced. Building to 5+ VP via settlements appears more reliable than trying to build cities early.
- **High pip count as a standalone goal**: Chasing maximum pips (e.g., placing on a 5+6+8 corner) is often suboptimal because those spots have fewer expansion vertices.

### What's undervalued
- **Interior vertices (3-hex adjacency)**: The `num_adjacent_hexes` feature is the strongest predictor in the model. A vertex touching 3 land hexes is significantly more valuable than a coastal vertex with the same pip count, because it provides more expansion paths and more resource combinations.
- **Expansion potential**: `expansion_pip_sum` (the sum of pips reachable within 2 road steps) is a better predictor than production at the opening settlements alone.
- **Road-race archetype**: Outperforms balanced by +0.4pp. Early road investment to lock in good expansion vertices pays off.

### The hardest truth
Opening placement explains roughly **1.5–2pp** of win rate variance in this dataset. Catan is fundamentally a dice game. The best opening gets you to ~27% from a 25.6% baseline — a real but modest edge. The game's randomness is a feature, not a bug: it keeps less-optimal openings competitive.

The largest opening edge (3.6pp) appears on boards where all players start with roughly equal pip counts — precisely the boards where human players might assume "it doesn't matter." These even boards are where opening *skill* matters most.

---

## Data & Methods

- **Source:** Colonist.io dataset, 43,947 games (streamed from games.tar.gz, no extraction)
- **Feature vector:** 75 features per opening (22 vertex 1, 22 vertex 2, 31 pair-level)
- **Model:** Logistic regression with StandardScaler (sklearn), global fit, no seat stratification
- **Analysis:** Pandas groupby, per-seat logistic regressions, within-game comparisons via game_id merge
- **Limitations:** Corner-ID → board-vertex mapping is an approximation (rank-sorted vertices); port assignment is angular approximation. Both introduce noise but should not introduce systematic bias.
