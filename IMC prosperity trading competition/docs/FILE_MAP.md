# File Map

This document explains the folder layout for a reviewer.

## Start Here

- `README.md`: high-level project overview.
- `docs/COMPETITION_PRIMER.md`: explains the competition mechanics.
- `docs/STRATEGY_EXPLAINER.md`: explains the finance logic.
- `docs/CODE_WALKTHROUGH.md`: explains the main code structure.
- `HOW_TO_RUN.txt`: commands for local backtests.

## Final Strategy Files

- `R2 - trader (final).py`: final Round 2 strategy. This is the most important
  implementation file.
- `R1 - trader (final).py`: final Round 1 strategy. This is the baseline that
  the Round 2 strategy extends.

## Iteration Snapshots

- `R0 - trader (*.py)`: tutorial or early competition versions.
- `R1 - trader (*.py)`: Round 1 iterations.
- `R2 - trader (*.py)`: Round 2 iterations.

These files are intentionally not deleted. They show how the strategy evolved
and make it possible to audit which ideas were tested before the final version.

Interpretation rule:

- Files ending in `(final).py` are submission snapshots.
- Numbered files are research snapshots.
- The highest number is not automatically the best file; some late experiments
  were rejected for robustness reasons.

## Data

- `data/round0/`: tutorial-round local market data.
- `data/round1/`: Round 1 local market data.
- `data/round2/`: Round 2 local market data.
- Each day folder contains a `prices` CSV and a `trades` CSV.

The backtester also has copied resources under:

```text
imc-prosperity-4-backtester/prosperity4bt/resources/
```

Those are the data files used by the packaged backtester when no external data
directory is provided.

## Backtester

- `imc-prosperity-4-backtester/`: local simulation engine.
- `imc-prosperity-4-backtester/prosperity4bt/__main__.py`: command-line entry
  point.
- `imc-prosperity-4-backtester/prosperity4bt/back_tester.py`: top-level
  backtest controller.
- `imc-prosperity-4-backtester/prosperity4bt/tools/order_match_maker.py`:
  order matching logic.
- `imc-prosperity-4-backtester/backtests/`: generated log output if present.

The backtester is a tool, not the strategy itself.

## Analysis Scripts

- `analyze_r2_osmium.py`: investigates OSMIUM order-book structure, hot quotes,
  wall stability, and post-event reversion.
- `breakthrough_analysis.py`: searches for unused predictive signals such as
  cross-product correlations, book features, and trade-tape effects.

These scripts are exploratory research tools. They are not uploaded to the
competition engine.

## Support Files

- `datamodel.py`: local copy of the official-style data classes used by the
  trader files.
- `Writing an Algorithm in Python.pdf`: competition guidance material.
- `strategy_journey.txt`: long-form original research diary. It is useful for
  understanding thought process, but the docs in this folder are cleaner entry
  points for an external reader.
- `SETUP_GUIDE.md`: environment setup.
- `HOW_TO_RUN.txt`: current command runbook.

## What I Chose Not To Reorganize

The strategy snapshots are still in the top-level folder. A cleaner archive
folder would look nicer, but moving them now could break existing backtest
commands, IDE references, and comparison workflows. The safer organization is to
add this file map and make the intended reading order explicit.
