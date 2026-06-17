# Strategy Explainer

This document explains the final strategy in finance terms.

## One-Sentence Summary

The final Round 2 strategy is an inventory-aware market-making system with two
separate books: a drift-carry strategy for `INTARIAN_PEPPER_ROOT` and a
mean-reversion microstructure strategy for `ASH_COATED_OSMIUM`.

## Core Philosophy

The code is built around three questions asked every timestamp:

1. What is fair value now?
2. Is any visible quote mispriced enough to cross the spread?
3. If not, where should I post passive liquidity and how much inventory should
   I be willing to hold?

The implementation deliberately keeps alpha signals and risk controls separate.
A signal can affect:

- fair value,
- taker permission,
- passive quote price,
- passive quote size,
- inventory target,
- or a defensive guard.

The main improvement across the project was learning not to let every signal do
every job. In later versions, small microstructure signals mostly affect passive
quotes, while crossing the spread is reserved for cleaner fair-value gaps.

## Product 1: INTARIAN_PEPPER_ROOT

PEPPER is treated as a drifting asset. The final strategy assumes there is a
persistent clock-like component in price evolution, so the strategy wants to own
a core long position while conditions remain favorable.

Main components:

- Fast EMA: tracks the current level.
- Slow EMA: provides a trend baseline.
- Clock drift estimate: models the stable upward drift observed in Round 2.
- Signed trade flow: detects whether prints are occurring above or below the
  previous reference price.
- Carry core: keeps a long inventory allocation while the drift regime is
  intact.
- Trading sleeve: buys and sells around the core to capture shorter-term edge.
- Adverse-carry guard: flattens or turns defensive if trend, clock gap, drift,
  or flow suggest the carry has broken.

Finance interpretation:

The PEPPER logic is not pure market making. It is closer to a carry strategy
wrapped in market-making execution. The strategy wants structural long exposure,
but it still uses passive quotes and position gates to avoid overpaying.

## Product 2: ASH_COATED_OSMIUM

OSMIUM is treated as range-bound and mean-reverting. The final strategy tries to
earn spread and reversion edge without overreacting to noisy visible depth.

Main components:

- Wall-mid: midpoint of the outer visible bid and ask levels. This is used as a
  more stable center for a range-bound book.
- Microprice: short-horizon pressure from best bid and best ask volumes.
- Learned range anchor: a slowly updated estimate of the center of the observed
  range.
- Round anchor: a soft long-horizon level inferred from the first valid mid,
  rounded to the nearest 100. This is used as an inventory permission gate, not
  as a direct fair-value term.
- Hot-quote signal: detects an unusually aggressive best bid or ask sitting far
  away from the next level and its own EMA.
- Toxicity score: combines level-1 imbalance, microprice deviation, and a small
  negative weight on deeper imbalance.
- Fill-quality feedback: evaluates whether recent fills were followed by
  favorable or adverse price movement.

Finance interpretation:

The OSMIUM logic is a microstructure strategy. It is trying to distinguish
between real fair-value movement and temporary pressure caused by aggressive
participants. When it sees a small expected reversion edge, it usually improves
its passive quote rather than crossing the market.

## Maker/Taker Separation

This is one of the most important design choices.

Taker reservation:

- Used when deciding whether to buy an ask or sell a bid immediately.
- Should include only robust fair-value and inventory information.
- Should not include small range or hot-quote edges that are weaker than the
  spread.

Maker reservation:

- Used when choosing passive bid and ask prices.
- Can include smaller microstructure signals because the strategy is not paying
  the spread.
- Can express directional views through quote placement and side sizing.

This split reduces the risk of converting small statistical signals into
expensive taker trades.

## Risk Controls

The main risk controls are:

- Position limits: all orders are sized within the product limit.
- Inventory skew: quotes become less aggressive on the side that would increase
  an already large position.
- Carry-break logic: PEPPER reduces long exposure when drift and trend signals
  deteriorate.
- Toxicity scaling: OSMIUM reduces quote size on the side most exposed to
  immediate adverse selection.
- Warmup periods: OSMIUM range signals are disabled until enough observations
  have accumulated.
- One-sided-book handling: early state estimates are adjusted when only bids or
  only asks are visible.

## Why The Project Does Not Use Black-Box ML

The competition environment is small, non-stationary, and adversarial. Hidden
test data can punish overfit models. The strategy therefore uses interpretable
statistical and microstructure signals:

- EMAs for level and drift tracking,
- simple regression slope as a diagnostic,
- book imbalance and microprice for immediate pressure,
- range anchors and wall-mid for mean reversion,
- explicit inventory gates for risk.

This makes the strategy easier to debug and easier to defend to a finance
reader: every signal has a direct market interpretation.

## Development Story

The numbered files show the research path:

- Early versions used intuitive market-making and simple fair-value estimates.
- Middle versions added econometric reasoning, such as stationarity and
  autocorrelation analysis.
- Later Round 1 versions introduced robust carry and inventory management.
- Round 2 versions adapted the Round 1 framework to market-access visibility,
  one-sided books, hot-quote behavior, and a stricter maker/taker split.

The final result is not simply the locally highest-PnL version. Some locally
profitable ideas were rejected because they were brittle or hard to justify on
hidden data.
