"""
IMC Prosperity 4 — Strategy v1: First Attempt (ML-Inspired Heuristics)
=======================================================================
Products: EMERALDS (stationary ~10000) | TOMATOES (drifting, unknown fair value)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONTEXT: MY STARTING POINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  My first attempt at a market-making algorithm. Before doing any formal
  statistical analysis, I assembled four intuitive ML-inspired signals.

  The overall structure — used in all subsequent versions — is:
    Phase 1 (Aggressive): scan the order book for resting orders that
      are mispriced relative to my fair value estimate, and take them
      immediately for certain edge.
    Phase 2 (Passive): post limit orders (bids and asks) around my fair
      value estimate to earn the spread from other participants.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SIGNAL DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  EMERALDS — fair value is visibly around 10,000.
    Passive quote midpoint adjusted by:
      - OBI: shifts in direction of current buy/sell pressure.
      - Inventory skew: leans quotes against our position.
    Aggressive fills always use the raw 10,000 — no adjustment.

  TOMATOES — fair value unknown, drifts over time.
    Four signals combined into a single fair value estimate:

    [1] EMA (Exponential Moving Average) — base fair value.
        SMA weights all window observations equally. For a drifting
        series, recent prices matter more than old ones, so SMA is
        suboptimal. EMA fixes this with geometric decay:
            ema_t = alpha * price_t + (1 - alpha) * ema_{t-1}
        I chose alpha = 0.25 by intuition (~7-period effective window).
        NOTE: v2 replaces this guess with a theoretically derived value.

    [2] Linear Regression Slope — trend component.
        OLS fit of recent midprices vs time index t=0,...,n-1 over a
        rolling 30-tick window. Slope estimates the current drift rate.
            fair += trend_weight * slope * horizon_ticks
        Positive slope → trending up → raise fair. Negative → lower.
        NOTE: This assumes persistent trends. If TOMATOES is actually a
        random walk, the slope is pure noise. I had not yet tested this.

    [3] Order Book Imbalance (OBI) — short-term flow signal.
        OBI = (total bid vol - total ask vol) / (total bid + ask vol)
        Range [-1, +1]. Positive = buy pressure → prices likely rise.
        Passive quote midpoint shifted by obi_weight * OBI.

    [4] Inventory Skew — position risk management.
        If long, lower both bid and ask to make selling more attractive.
        If short, raise both. Prevents runaway inventory build-up.
            skew = -max_skew * (position / limit)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LIMITATIONS (→ motivate v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - alpha=0.25 and all weights are unvalidated guesses.
  - Regression slope adds noise if TOMATOES is a random walk —
    not yet formally tested with a unit root test.
  - No principled basis for comparing this model to alternatives.
"""

import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


# ---------------------------------------------------------------------------
# Logger (unchanged from starter)
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing.symbol, listing.product, listing.denomination]
            )
        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders,
                order_depth.sell_orders,
            ]
        return compressed

    def compress_trades(
        self, trades: dict[Symbol, list[Trade]]
    ) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(
        self, orders: dict[Symbol, list[Order]]
    ) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def ema(prices: list, alpha: float = 0.3) -> float:
    """
    Exponential Moving Average — fair value estimate for a drifting series.

    Why EMA over SMA:
      SMA gives equal weight to all observations in the window. For a price
      series that drifts, the price from 30 ticks ago is less informative
      than the price from 1 tick ago. EMA decays weights geometrically:
        w_k = alpha * (1 - alpha)^k   (k = ticks ago)
      so recent observations dominate. The 'effective window' is ~1/alpha ticks.
    """
    if not prices:
        return 0.0
    val = prices[0]
    for p in prices[1:]:
        val = alpha * p + (1.0 - alpha) * val
    return val


def linreg_slope(prices: list) -> float:
    """
    OLS slope of prices against time index t = 0, 1, ..., n-1.

    This estimates the current rate of drift in the price series.
      slope > 0  → upward drift  → raise fair value estimate
      slope < 0  → downward drift → lower fair value estimate

    We project this slope forward by TOMATO_TREND_HORIZON ticks:
      trend_adj = trend_weight * slope * horizon
    and add it to the EMA fair value, giving a trend-adjusted estimate.

    Limitation: this is sensible only if the price series has persistent
    trends. If it is a random walk (differenced prices are white noise),
    the slope carries no predictive power. This was not tested in v1.
    """
    n = len(prices)
    if n < 3:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(prices) / n
    num = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def order_book_imbalance(od: OrderDepth) -> float:
    """
    Order Book Imbalance (OBI) — measures directional pressure in the book.

    OBI = (total bid volume - total ask volume) / (total bid + total ask)
    Range: [-1, +1].
      OBI > 0 → more resting buy volume → price likely to rise short-term.
      OBI < 0 → more resting sell volume → price likely to fall short-term.

    We shift our passive quote midpoint by obi_weight * OBI so that we
    lean in the direction the market is about to move — getting filled on
    the side with more adverse selection protection.
    """
    bid_vol = sum(od.buy_orders.values())
    ask_vol = sum(abs(v) for v in od.sell_orders.values())
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def inventory_skew(position: int, limit: int, max_skew: float = 3.0) -> float:
    """
    Inventory skew — adjusts our quote reference against our current position.

    A market maker accumulating a large position takes on directional risk.
    If we are long and the price falls, we lose money. To mitigate this:
      Long position → negative skew → lower both bid and ask slightly
        → counterparties are more likely to buy from us, reducing our long.
      Short position → positive skew → raise both quotes
        → counterparties more likely to sell to us, reducing our short.

    Formula: skew = -max_skew * (position / limit)
    Returns a price adjustment in ticks, applied to the passive quote midpoint.
    """
    return -max_skew * (position / limit)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:
    """
    v1: ML-inspired market maker using EMA, OBI, linreg slope, and inventory skew.

    Two-phase structure (used in all versions):
      Phase 1 — Aggressive: take any resting order mispriced vs our fair value.
      Phase 2 — Passive: post bid/ask limit orders around our fair value estimate.
    """

    # ── EMERALDS parameters (all chosen by intuition in this version) ──
    EMERALD_FAIR = 10_000
    EMERALD_LIMIT = 80
    EMERALD_BASE_SPREAD = 2       # passive half-spread from adjusted fair
    EMERALD_MAX_SKEW = 2.0        # max inventory skew in ticks at full position
    EMERALD_OBI_WEIGHT = 1.5      # ticks of quote adjustment per unit of OBI

    # ── TOMATOES parameters (all chosen by intuition in this version) ──
    TOMATO_LIMIT = 80
    TOMATO_EMA_ALPHA = 0.25       # EMA smoothing factor (~7-period effective window)
    TOMATO_WINDOW = 30            # rolling window length for slope and EMA
    TOMATO_TREND_HORIZON = 3      # ticks ahead to project the regression slope
    TOMATO_TREND_WEIGHT = 0.6     # dampening factor applied to the raw slope signal
    TOMATO_OBI_WEIGHT = 1.0       # ticks of quote adjustment per unit of OBI
    TOMATO_MAX_SKEW = 3.0         # max inventory skew in ticks at full position
    TOMATO_BASE_SPREAD = 2        # passive half-spread from fair value

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {"tomato_mids": []}

        for product in state.order_depths:
            od: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(od, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(
                    od, position, saved["tomato_mids"]
                )

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ------------------------------------------------------------------
    # EMERALDS
    # ------------------------------------------------------------------
    def trade_emeralds(self, od: OrderDepth, position: int) -> List[Order]:
        """
        EMERALDS: fair value is known to be ~10,000 from observation.

        Aggressive phase: buy any resting ask < 10,000; sell any resting bid > 10,000.
          These are guaranteed-profit trades — we take them at full available size.
          We use the raw 10,000, not the adjusted_fair, so inventory never stops
          us from capturing certain edge.

        Passive phase: post quotes around an OBI- and inventory-adjusted midpoint.
          - OBI shifts the midpoint toward the direction of current order flow.
          - Inventory skew shifts both quotes against our position to reduce it
            over time without actively crossing the spread.
          Quote prices: adjusted_fair ± EMERALD_BASE_SPREAD.
        """
        orders: List[Order] = []
        fair = self.EMERALD_FAIR
        limit = self.EMERALD_LIMIT

        # Compute adjustments for passive quoting only
        obi = order_book_imbalance(od)
        inv_sk = inventory_skew(position, limit, self.EMERALD_MAX_SKEW)

        # adjusted_fair is used only for passive quote placement.
        # Aggressive fills still check against raw fair=10,000.
        adjusted_fair = fair + obi * self.EMERALD_OBI_WEIGHT + inv_sk

        # --- aggressive: take mispriced resting orders ---
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair:
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            buy_qty = min(-od.sell_orders[ask_price], can_buy)
            orders.append(Order("EMERALDS", ask_price, buy_qty))
            position += buy_qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair:
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            sell_qty = min(od.buy_orders[bid_price], can_sell)
            orders.append(Order("EMERALDS", bid_price, -sell_qty))
            position -= sell_qty

        # --- passive: post quotes around adjusted fair ---
        buy_price = round(adjusted_fair) - self.EMERALD_BASE_SPREAD
        sell_price = round(adjusted_fair) + self.EMERALD_BASE_SPREAD

        remaining_buy = limit - position
        remaining_sell = limit + position

        if remaining_buy > 0:
            orders.append(Order("EMERALDS", buy_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("EMERALDS", sell_price, -remaining_sell))

        return orders

    # ------------------------------------------------------------------
    # TOMATOES
    # ------------------------------------------------------------------
    def trade_tomatoes(
        self, od: OrderDepth, position: int, mid_history: list
    ) -> List[Order]:
        """
        TOMATOES: fair value is unknown and drifts — must be estimated each tick.

        Combined fair value estimate (all four signals summed):
            fair = ema_fair                         ← base estimate (Signal 1)
                 + trend_weight * slope * horizon   ← drift projection (Signal 2)
                 + obi_weight * obi                 ← order flow signal (Signal 3)
                 + inventory_skew                   ← risk management (Signal 4)

        Aggressive phase: take any ask < fair or bid > fair (certain edge).
          Note: inventory skew is already baked into 'fair' here — this is a bug
          that gets fixed in v3. The skew means we refuse buys at fair-N to fair-1
          when we are long, missing genuinely profitable fills.

        Passive phase: post quotes at int(fair) ± dynamic_spread.
          Spread widens when the regression slope is large (high trend uncertainty).

        Persisted state (traderData): mid_history list, passed by reference and
          updated each tick so the EMA and slope use the most recent prices.
        """
        orders: List[Order] = []
        limit = self.TOMATO_LIMIT

        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2.0

        # Update history
        mid_history.append(current_mid)
        if len(mid_history) > self.TOMATO_WINDOW:
            mid_history.pop(0)

        # Signal 1: EMA fair value — base estimate, more reactive than SMA
        ema_fair = ema(mid_history, alpha=self.TOMATO_EMA_ALPHA)

        # Signal 2: Trend projection — OLS slope extrapolated TOMATO_TREND_HORIZON ticks
        # Rationale: if prices have been drifting up, raise our fair value estimate.
        slope = linreg_slope(mid_history)
        trend_adj = self.TOMATO_TREND_WEIGHT * slope * self.TOMATO_TREND_HORIZON

        # Signal 3: Order book imbalance — short-term directional nudge
        obi = order_book_imbalance(od)
        obi_adj = self.TOMATO_OBI_WEIGHT * obi

        # Signal 4: Inventory skew — lean our entire fair value against our position
        # (BUG: applying this to aggressive fills too; fixed in v3)
        inv_sk = inventory_skew(position, limit, self.TOMATO_MAX_SKEW)

        # All four adjustments combined into a single fair value
        fair = ema_fair + trend_adj + obi_adj + inv_sk

        logger.print(
            f"TOMATOES | mid={current_mid:.1f} ema={ema_fair:.1f} "
            f"slope={slope:.3f} trend_adj={trend_adj:.2f} "
            f"obi={obi:.3f} obi_adj={obi_adj:.2f} "
            f"inv_sk={inv_sk:.2f} fair={fair:.2f} pos={position}"
        )

        # Phase 1: Aggressive — take any resting order mispriced vs our fair estimate.
        # (BUG: 'fair' includes inventory skew here, so we miss fills near true fair)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair:
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            buy_qty = min(-od.sell_orders[ask_price], can_buy)
            orders.append(Order("TOMATOES", ask_price, buy_qty))
            position += buy_qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair:
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            sell_qty = min(od.buy_orders[bid_price], can_sell)
            orders.append(Order("TOMATOES", bid_price, -sell_qty))
            position -= sell_qty

        # Phase 2: Passive quotes around fair, with spread that widens under strong trend.
        # Rationale: a large slope means high uncertainty → wider spread for protection.
        trend_magnitude = abs(slope) * self.TOMATO_TREND_HORIZON
        dynamic_spread = max(
            self.TOMATO_BASE_SPREAD,
            int(self.TOMATO_BASE_SPREAD + trend_magnitude * 0.5)
        )

        buy_price = int(fair) - dynamic_spread
        sell_price = int(fair) + dynamic_spread

        remaining_buy = limit - position
        remaining_sell = limit + position

        if remaining_buy > 0:
            orders.append(Order("TOMATOES", buy_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("TOMATOES", sell_price, -remaining_sell))

        return orders
