# Competition Primer

This document explains the IMC Prosperity setting without assuming the reader
has seen the competition before.

## The Basic Setup

IMC Prosperity is a simulated electronic trading competition. Each participant
writes a Python class named `Trader`. At every market timestamp, the exchange
simulator sends the current market state into:

```python
Trader.run(state)
```

The strategy returns:

```python
orders, conversions, traderData
```

For this project, the important return value is `orders`: buy and sell limit
orders for the current timestamp.

The goal is to maximize simulated profit and loss while obeying position limits
and runtime constraints.

## What The Strategy Sees

Each call to `run()` receives a `TradingState` object. The important fields are:

- `timestamp`: the current point in the simulation.
- `order_depths`: visible buy and sell orders for each product.
- `market_trades`: trades printed by other market participants.
- `own_trades`: fills from this strategy's previous orders.
- `position`: current net inventory by product.
- `traderData`: a string returned by the strategy on the prior timestamp.

The competition does not let a strategy keep ordinary in-memory state across
timestamps in a reliable way. Persistent state must be serialized into
`traderData`, then decoded on the next call. That is why the main strategy uses
`jsonpickle` heavily.

## Order Book Conventions

The order book contains price levels and volumes.

- Buy orders are bids.
- Sell orders are asks.
- In the official data model, sell-side volumes are represented as negative
  quantities. The strategy usually converts them with `-volume` when it wants a
  positive available size.

Example:

```text
Best bid:  9998 for 12 units
Best ask: 10002 for 8 units
Midprice: 10000
Spread:   4 ticks
```

If the strategy submits a buy order at 10002, it can cross the spread and buy
from the ask. If it submits a buy order at 9999, it rests passively in the book
and may be filled later if someone sells into it.

## Maker Versus Taker

The project separates two execution styles:

- Taker logic: cross the spread only when the visible quote is clearly
  mispriced relative to fair value. This is expensive because crossing pays the
  spread.
- Maker logic: post passive limit orders near, but not through, fair value. This
  earns edge if filled, but exposes the strategy to adverse selection.

One of the important Round 2 lessons is that small alpha signals should usually
change passive quote placement and size, not taker thresholds. Paying a full
spread to capture a one-tick signal is usually negative expected value.

## Position Limits

Each product has a maximum allowed long or short position. The final strategies
use a limit of 80 units for the Round 2 products. The code checks remaining
buying room and selling room before every order.

This matters because a profitable signal can still be unusable if the strategy
is already close to its inventory limit.

## Backtesting

The folder includes a local backtester under:

```text
imc-prosperity-4-backtester/
```

The backtester replays historical CSV data and calls the strategy the same way
the competition engine does. It then simulates matching orders against the
historical book and trade tape.

Important caveat: no local backtester can perfectly model how other competitors
would react to this strategy's quotes. The local backtester is useful for
research, debugging, and rejecting weak ideas, but official hidden-data results
are still the final robustness check.

## Round 2 Market Access Fee

Round 2 introduced a Market Access Fee mechanism. The strategy can expose:

```python
Trader.bid()
```

In `R2 - trader (final).py`, this returns `200`. The purpose is to bid for extra
order-book visibility. The strategy treats this as a fixed operational cost
worth paying because several OSMIUM signals depend on book structure.

## Glossary

- Fair value: the strategy's estimate of the asset's current economic value.
- Midprice: average of best bid and best ask.
- Microprice: bid/ask midpoint adjusted for level-1 volume imbalance.
- Spread: best ask minus best bid.
- Inventory skew: quote adjustment that discourages adding to an already large
  position.
- OBI: order book imbalance, usually bid volume minus ask volume divided by
  total volume.
- Passive quote: a limit order that waits in the book.
- Crossing: buying at the ask or selling at the bid immediately.
- Adverse selection: getting filled because a better-informed counterparty
  knows the price is about to move against the quote.
