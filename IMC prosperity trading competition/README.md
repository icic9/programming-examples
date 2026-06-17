# IMC Prosperity Trading Competition Project

This repository is my research and implementation workspace for the IMC
Prosperity algorithmic trading competition. It contains the full path from
early tutorial-round ideas to later Round 1 and Round 2 submissions, including
market data, backtesting tools, analysis scripts, and many strategy snapshots.

The main purpose of the project is to design a self-contained trading algorithm
that reads an order book every timestamp, estimates fair value, manages
inventory risk, and sends limit orders under strict position and runtime
constraints.

## Best Starting Point

Read these in order:

1. `docs/COMPETITION_PRIMER.md` - explains the competition for someone who has
   never seen IMC Prosperity.
2. `docs/STRATEGY_EXPLAINER.md` - explains the finance and market
   microstructure logic behind the strategies.
3. `docs/CODE_WALKTHROUGH.md` - maps the explanation onto the main Python file.
4. `docs/FILE_MAP.md` - explains what each part of the folder is for.
5. `docs/BACKTEST_RESULTS.md` - records the latest verified local backtest.
6. `HOW_TO_RUN.txt` - gives commands for running local backtests.

## Main Strategy Files

The two most important files are:

- `R2 - trader (final).py` - final Round 2 strategy and the best single file to
  review as the current submission-quality implementation.
- `R1 - trader (final).py` - final Round 1 strategy and the baseline that the
  Round 2 strategy extends.

The numbered files such as `R2 - trader (4).py` or `R1 - trader (29).py` are
iteration snapshots. They are intentionally kept because they show the research
process: ideas tested, bugs fixed, and rejected approaches. They should be read
as an audit trail, not as separate production modules.

## What The Final Round 2 Strategy Does

At a high level, `R2 - trader (final).py` is a market-making and statistical
arbitrage algorithm for two fictional products:

- `INTARIAN_PEPPER_ROOT`: treated as a drifting asset with a persistent
  upward clock-driven component. The strategy carries a long core inventory,
  recycles a smaller trading sleeve, and uses trend/flow guards to flatten when
  the drift appears to break.
- `ASH_COATED_OSMIUM`: treated as a range-bound, mean-reverting asset. The
  strategy estimates fair value from the visible book, microprice, and learned
  range anchor; then expresses most alpha passively through quote placement and
  side sizing rather than paying the spread.

The code also implements competition-specific mechanics:

- `Trader.run(state)` is the method called by the exchange simulator at each
  timestamp.
- `traderData` is the only persistent memory between timestamps, so all rolling
  state is serialized and returned each tick.
- Position limits are enforced in the strategy before orders are submitted.
- Round 2 includes a Market Access Fee bid through `Trader.bid()`, used here to
  win extra order-book visibility.

## Project Structure

```text
.
|-- R0 - trader (*.py)              Tutorial / early strategy versions
|-- R1 - trader (*.py)              Round 1 strategy iterations
|-- R2 - trader (*.py)              Round 2 strategy iterations
|-- R1 - trader (final).py          Round 1 final submission snapshot
|-- R2 - trader (final).py          Round 2 final submission snapshot
|-- data/                           Local CSV market data by round and day
|-- docs/                           Reviewer-facing explanations
|-- imc-prosperity-4-backtester/    Local backtesting engine
|-- analyze_r2_osmium.py            Focused OSMIUM order-book analysis
|-- breakthrough_analysis.py        Search for unused predictive signals
|-- datamodel.py                    Official-style data classes used by traders
|-- HOW_TO_RUN.txt                  Backtest commands and troubleshooting
|-- SETUP_GUIDE.md                  Environment and setup notes
|-- strategy_journey.txt            Long-form original research diary
```

## Attribution

The local backtester is based on the open-source `imc-prosperity-4-backtester`
project by Kevin Fu, itself inspired by earlier Prosperity backtesters. The
visualizer referenced by the backtester is also from the Prosperity community.
The trading strategies, analysis scripts, and explanatory documentation in this
folder are my own work unless otherwise stated.
