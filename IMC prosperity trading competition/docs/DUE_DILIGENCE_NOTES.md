# Due Diligence Notes For A Finance Reviewer

These are the questions I would expect from someone reviewing the project from
a trading or quant perspective.

## What Is The Edge?

There are two main edges:

- PEPPER: persistent drift/carry, implemented with a long core position and
  defensive regime-break logic.
- OSMIUM: range reversion and microstructure behavior, implemented through
  fair-value estimation, passive quote improvement, and toxicity-aware sizing.

The edge is not a single black-box forecast. It is a collection of small,
interpretable decisions about fair value, execution, and inventory.

## How Does The Strategy Avoid Overfitting?

- It uses simple signals with direct market interpretation.
- It rejects several locally profitable ideas when they are brittle.
- It uses hidden-data robustness as a design criterion.
- It avoids hardcoded absolute price anchors where possible.
- It expresses weak signals passively rather than paying the spread.

## What Are The Main Failure Modes?

- A structural regime change can invalidate PEPPER's carry assumption.
- OSMIUM's visible order book can be noisy or randomly masked.
- Local backtests may not perfectly model how other agents react to passive
  quotes.
- Inventory limits can prevent the strategy from acting on a valid signal.
- A small signal can become negative EV if implemented as an aggressive taker
  trade.

The final strategy addresses these with drift guards, one-sided-book handling,
position-aware sizing, and the maker/taker split.

## Why Keep So Many Files?

The numbered files are part of the research audit trail. They show how decisions
changed, which bugs were fixed, and which ideas were rejected. In a professional
research setting, this is closer to a lab notebook than a polished package.

## What Should Be Reviewed In The Code?

Focus on:

- `Trader.run`: state persistence and product dispatch.
- `_trade_pepper`: drift/carry logic and adverse-carry guard.
- `_trade_osmium`: fair value, range logic, hot-quote detection, toxicity, and
  passive order sizing.
- `bid`: Round 2 Market Access Fee decision.
- `_penny_passive_prices`: quote-placement discipline.

## What Should Not Be Over-Interpreted?

Local PnL is useful but not definitive. A backtest can confirm that the code
does what it is supposed to do on known data, but it cannot fully prove hidden
data robustness or model other competitors' reactions.

The best evidence is the combination of:

- clear market rationale,
- reasonable risk controls,
- reproducible backtests,
- and documented rejection of fragile local optimizations.
