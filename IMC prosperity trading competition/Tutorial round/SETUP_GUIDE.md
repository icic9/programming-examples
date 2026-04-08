# IMC Prosperity 4 — Backtester Setup & Workflow Guide

## What You Need

- Python 3.11 or 3.12 installed
- A terminal (Mac Terminal, Windows PowerShell, or VS Code integrated terminal)
- The sample CSV files from IMC (prices + trades for round 0)

## Step 1: Set Up Your Project Folder

Create a folder structure like this on your computer:

```
prosperity4/
├── trader.py                ← your algorithm (the file you also upload to IMC)
├── data/
│   └── round0/
│       └── day-1/
│           ├── prices_round_0_day_-1.csv
│           └── trades_round_0_day_-1.csv
│       └── day-2/
│           ├── prices_round_0_day_-2.csv
│           └── trades_round_0_day_-2.csv
```

## Step 2: Install the Backtester

Open your terminal, navigate to your project folder, and run:

```bash
pip install -U imc-prosperity-4-backtester
```

If that package name doesn't work (it may be new), try cloning directly:

```bash
git clone https://github.com/kevin-fu1/imc-prosperity-4-backtester.git
cd imc-prosperity-4-backtester
pip install -e .
```

The `-e` flag installs it in "editable mode" so you can tweak the backtester
code if needed.

## Step 3: Run a Backtest

The exact command depends on how the backtester is configured, but based on
the repo structure it should be something like:

```bash
# If installed as a pip package with a CLI entry point:
prosperity4bt trader.py 0

# Or if running from the cloned repo:
python -m backtester trader.py --round 0 --day -1
```

Check the repo's README for the exact CLI syntax — it may differ slightly
from the Prosperity 3 backtester. The key inputs are always:
  1. Path to your trader.py file
  2. Which round/day of data to test against

## Step 4: Understand Backtester Output

The backtester will print a PnL (profit and loss) summary per product.
A typical output looks like:

```
EMERALDS: 15,230 XIRECs
TOMATOES:  3,870 XIRECs
Total:    19,100 XIRECs
```

It may also generate a log file you can feed into a visualizer.

## How the Backtester Works (conceptually)

The backtester replays your CSV data and simulates the engine:

```
For each timestamp in the CSV data:

  1. BUILD TradingState
     - Read the order book (bid/ask prices & volumes) from prices CSV
     - Read bot trades from trades CSV
     - Include your current position and traderData from last iteration

  2. CALL your Trader.run(state)
     - Your code returns: orders, conversions, traderData

  3. MATCH your orders against the order book
     - If your buy price >= an ask price → trade happens
     - If your sell price <= a bid price → trade happens
     - Remaining unmatched orders: check if any bot trades from the
       trades CSV could fill them
     - Update your position

  4. RECORD the PnL
     - Track how much you spent and earned from each trade
```

Key difference from the real IMC engine: the backtester can't simulate
bot *reactions* to your quotes. In the real engine, bots might decide to
trade against your outstanding orders. The backtester approximates this
using the trades CSV, but it's not perfect.

## Your Development Loop

This is the workflow you should follow:

```
  ┌─────────────────────────────────────────────┐
  │  1. Edit trader.py (tweak strategy params)  │
  └────────────────────┬────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────┐
  │  2. Run backtester locally on sample data   │
  │     → See PnL, check if it improved         │
  └────────────────────┬────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────┐
  │  3. Upload trader.py to IMC official site   │
  │     → See PnL on their hidden test data     │
  └────────────────────┬────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────┐
  │  4. Compare backtester PnL vs official PnL  │
  │     → If they're close, your backtest is    │
  │       reliable. If not, investigate why.    │
  └────────────────────┬────────────────────────┘
                       │
                       └──→ repeat from step 1
```

## What to Tweak in trader.py

Here are the parameters you can experiment with, roughly in order
of expected impact:

### EMERALDS:
- FAIR value: fixed at 10000 (confirmed by data — don't change)
- Quote offset: currently ±2 from fair. Try ±1, ±3, ±4.
  Tighter = more fills but more adverse selection risk.
  Wider = fewer fills but safer.

### TOMATOES:
- Rolling window size: currently 20. Try 5, 10, 50, 100.
  Shorter = faster reaction to drift but more noise.
  Longer = smoother estimate but lags behind real moves.
- Quote offset: currently ±2 from fair. Same tradeoff as above.
- You could try a weighted average (more weight on recent prices)
  instead of a simple average.

### Both products:
- How aggressively you use position room. The current code uses
  ALL remaining room for passive quotes. You might want to cap
  passive quote size (e.g., max 20 units) to leave room for
  aggressive opportunities.

## Common Gotchas

1. **Position limit violation → ALL orders rejected.**
   The engine checks aggregated buy volume and sell volume
   SEPARATELY. If either side alone would breach the limit,
   every order on that side is cancelled. Not partially — ALL.

2. **traderData has a 50,000 character limit.**
   Don't store too much history. 20 mid-prices is fine.
   Storing 10,000 would blow the limit.

3. **No external libraries** beyond pandas, numpy, statistics,
   math, typing, jsonpickle. No scipy, no sklearn, no requests.

4. **900ms time limit per run() call.** Keep it simple.
   Don't do heavy computation. The average should be ~100ms.

5. **The file you upload to IMC must be self-contained.**
   Everything goes in one .py file. You import from `datamodel`
   (which IMC provides) and standard libraries only.
