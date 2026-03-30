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
```

## Notes

- Large local datasets, generated screenshots, and the vendored OCR environment are intentionally excluded from git.
- See `prd.md`, `RESEARCH.md`, and `ideal_4p_solver/ROADMAP.md` for project direction.
