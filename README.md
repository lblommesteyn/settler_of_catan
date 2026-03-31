# settler_of_catan

Computer-vision and solver tooling for analyzing Catan openings and running a live assistant alongside Colonist.io.

## What is here

- Opening evaluation models and feature engineering
- Exact-rules base-game engine scaffolding
- Colonist.io screen parsing and live advice loops
- Tests for board logic, solver rules, and CV integration

## Current status

The repo already supports:

- offline opening ranking from a board JSON
- live opening suggestions from a Colonist.io screen
- an approximate live midgame advisor loop driven by CV and heuristic action ranking

The repo does not yet provide a production-grade full-game assistant. The main remaining work is robust real-screen parsing, state tracking from game events, hidden-information belief tracking, and a stronger search-based advisor.

## Quick start

```bash
pip install -e .[dev,vision]
pytest -q
```

Primary CLI entrypoints:

```bash
catan-score
catan-colonist-cv
catan-opening-live
```

## Opening overlay

Fastest way to bring up the live setup assistant:

```bash
catan-opening-live
```

Useful flags:

```bash
catan-opening-live --my-color red
catan-opening-live --bbox 100,100,1600,1000
catan-opening-live --top-k 3
```

This mode watches the Colonist setup screen directly and does not need a board JSON or calibration file.

## Full live advisor

The midgame advisor still needs a mapped board and calibration:

```bash
catan-colonist-cv init-board --output board.json
catan-colonist-cv calibrate --board board.json --output calibration.json
catan-colonist-cv live --board board.json --calibration calibration.json --my-color red
```

Useful diagnostics:

```bash
catan-colonist-cv context-screen --my-color red
catan-colonist-cv bootstrap-screen --output board.json
```

The live loop now reports per-stage latency and will keep running through transient detection failures instead of exiting immediately.

## Notes

- Large local datasets, generated screenshots, and the vendored OCR environment are intentionally excluded from git.
- See `prd.md`, `RESEARCH.md`, and `ideal_4p_solver/ROADMAP.md` for project direction.
