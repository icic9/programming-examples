# Code Walkthrough

This guide maps the final strategy explanation onto `R2 - trader (final).py`.

## File-Level Structure

The final file is self-contained because the competition upload format expects
one Python strategy file. It imports only the official-style `datamodel` classes
and `jsonpickle` for state serialization.

Major sections:

- `Logger`: compresses state, orders, and diagnostic output into the format used
  by the visualizer.
- `Trader`: the class called by the exchange simulator.
- Strategy constants: product names, limits, alpha weights, thresholds, and
  sizing parameters.
- Helper functions: microprice, imbalance, wall-mid, passive price selection,
  rolling slope, hot-quote detection, and fill-quality tracking.
- `run()`: entry point called each timestamp.
- `_trade_pepper()`: PEPPER-specific strategy.
- `_trade_osmium()`: OSMIUM-specific strategy.

## Entry Point: `Trader.run`

`run(state)` performs four jobs:

1. Decode `state.traderData` into a dictionary named `saved`.
2. Initialize missing rolling-state variables.
3. Dispatch each product to its product-specific strategy function.
4. Encode the updated state back into `traderData` and return orders.

This is the key competition constraint: if a statistic needs to persist from one
timestamp to the next, it must live in `saved` and be returned through
`traderData`.

## PEPPER Function: `_trade_pepper`

The PEPPER code follows this sequence:

1. Read the visible order book.
2. Estimate a reference price using large quotes, microprice, and one-sided-book
   adjustments when needed.
3. Update fast EMA, slow EMA, flow, volatility, drift, and history.
4. Build a forecast from EMA level, trend carry, flow, and clock drift.
5. Decide whether the carry regime is healthy, flat-warning, or short-warning.
6. Use taker logic to build or reduce the carry core.
7. Use passive quotes to recycle the remaining trading sleeve.

The important concept is the carry core. The strategy is willing to keep a
structural long allocation while the drift regime remains intact, then reduces
that exposure when multiple warning signals confirm deterioration.

## OSMIUM Function: `_trade_osmium`

The OSMIUM code follows this sequence:

1. Read the visible order book and handle missing bid or ask sides.
2. Compute mid, spread, microprice, and wall-mid.
3. Update near-bid and near-ask EMAs for hot-quote detection.
4. Detect wall dislocations and hot bids/asks.
5. Infer a round-number anchor from the first valid mid.
6. Update observed range, range anchor, extreme-trade signal, and spike signal.
7. Build fair value from wall-mid, microprice, and learned anchor.
8. Build a taker reservation and cross only clear mispricings.
9. Build a maker reservation with range and dislocation bias.
10. Select passive prices with pennying logic.
11. Scale buy and sell sizes using inventory, toxicity, sweep risk, and
    dislocation direction.

The important concept is passive expression of small alpha. OSMIUM's range and
hot-quote signals are useful, but usually too small to justify crossing the
spread. The code therefore uses them mostly to improve passive quote placement
and side sizing.

## Helper Functions

- `_microprice`: estimates short-term pressure from best bid/ask volumes.
- `_l1_imbalance`: level-1 bid/ask volume imbalance.
- `_deep_imbalance`: imbalance excluding the best bid and ask.
- `_wall_mid`: uses outer visible levels as a stable range center.
- `_penny_passive_prices`: posts one tick better than competing passive orders
  while preserving minimum edge around fair value.
- `_two_level_orders`: converts target buy/sell capacity into actual `Order`
  objects. In the final Round 2 strategy, OSMIUM uses only the first level.
- `_hot_quote_signal`: detects transient aggressive best bids or asks.
- `_update_osmium_fill_quality`: tracks whether fills were followed by favorable
  or adverse mark-to-market movement.

## What To Look For In Review

For a finance review, the most important design choices are:

- Fair-value estimation is product-specific rather than generic.
- Position limits are treated as first-class constraints.
- Taker and maker decisions are separated.
- Small signals are expressed passively.
- Hidden-data robustness is prioritized over local backtest maximization.

For an engineering review, the most important design choices are:

- The uploaded strategy remains one self-contained file.
- Persistent state is explicit in `traderData`.
- Logging is compact enough for the competition output limit.
- The strategy uses simple standard-library computations rather than heavy
  dependencies that may be unavailable in the official environment.
