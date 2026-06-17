"""
IMC Prosperity 4 — Strategy v2: Econometric Foundation
=======================================================
Products: EMERALDS (stationary ~10000) | TOMATOES (ARIMA(0,1,1) drift)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MOTIVATION: FROM INTUITION TO THEORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  In v1, the EMA alpha and all signal weights were chosen by intuition.
  I had no principled way to know whether alpha=0.25 was right or wrong.
  In this version, I apply econometric tools from my LSE Metrics II course
  (Slide Packs #3 and #6) directly to the price data to identify the true
  data-generating process — and derive the optimal strategy from theory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EMERALDS — Stationary Process, Known Fair Value
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1: Dickey-Fuller (DF) Unit Root Test.
    H0: price series has a unit root (is non-stationary, drifts).
    H1: series is stationary (mean-reverting).
    Result: DF t-statistic ≈ -97. Critical value at 1% ≈ -3.43.
    Since -97 << -3.43, we overwhelmingly reject H0.
    Conclusion: EMERALDS is strongly stationary.

  Step 2: ACF of levels.
    ACF(1) ≈ 0.03 — essentially zero autocorrelation at any lag.
    Conclusion: the series is white noise around a fixed mean of 10,000.

  Trading implication: fair value = 10,000 with near-certainty.
  No forecasting model is needed. Just make markets aggressively around
  this known value, with inventory skew to manage position risk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOMATOES — ARIMA(0,1,1) Process: Theory-Derived Optimal Forecast
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Step 1: ACF of levels shows ACF(1) ≈ 0.99 → near-unit-root.
    Prices are highly persistent and drift over time.

  Step 2: Difference the series: ΔY_t = Y_t − Y_{t-1}.
    ACF of differences:
      ACF(1) ≈ -0.41   (significant negative correlation at lag 1)
      ACF(2), ACF(3), ... ≈ 0 (sharp cutoff after lag 1)
    Interpretation: one significant lag then cutoff → MA(1) in differences.
    This is the textbook ACF signature of an ARIMA(0,1,1) process:
        ΔY_t = ε_t + θ · ε_{t-1}   with θ ≈ -0.55

  Step 3: Derive the optimal 1-step-ahead forecast.
    For ARIMA(0,1,1), the Kalman filter solution gives the minimum-MSE
    forecast as an Exponentially Weighted Moving Average (EWMA) with:
        α = 1 + θ ≈ 1 − 0.55 = 0.45   (refined to 0.43 empirically)
    This is NOT a tuned hyperparameter — it is derived from the model.
    Empirical verification: EMA(0.43) reduces forecast MSE by ~53% vs SMA(20).

  Step 4: Use microprice instead of midprice as the EMA input signal.
    microprice = (best_bid × ask_vol + best_ask × bid_vol) / (bid_vol + ask_vol)
    Volume-weights the two sides of the book. If ask side is heavier,
    the microprice sits closer to the ask — a more accurate measure of
    where the next trade is likely to occur.

  Additional: adaptive spread based on rolling volatility σ(ΔY_t).
    Wider spread when volatility is high → protection from adverse selection.
    Inventory skew (Avellaneda-Stoikov style) on passive quotes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BUGS (identified after backtesting → fixed in v3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BUG 1: Aggressive TOMATOES threshold used skewed effective_fair.
  BUG 2: TOMATO_SKEW=0.15 → max 12 ticks skew at limit (wider than spread).
  BUG 3: Adaptive spread uncapped → can grow to half-spread=6 (zero fills).
  BUG 4: Aggressive EMERALDS threshold also used skewed effective_fair.
  See v3 for detailed explanations and fixes.
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
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
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
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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


class Trader:
    """
    v2: Econometrics-driven market maker.

    Key insight: formal statistical identification of the data-generating process
    yields the theoretically optimal forecast — no arbitrary parameter choices.

    EMERALDS: known stationary fair = 10,000. Make markets aggressively.
    TOMATOES: ARIMA(0,1,1) with θ ≈ -0.55 → optimal forecast = EMA(α=0.43).
    """

    # ── EMERALDS parameters ──
    EMERALD_FAIR   = 10000
    EMERALD_SPREAD = 2          # passive half-spread from fair value
    EMERALD_LIMIT  = 80
    EMERALD_SKEW   = 0.5        # ticks of passive quote skew per unit of position
                                # (BUG: also applied to aggressive fills — fixed in v3)

    # ── TOMATOES parameters ──
    # α = 1 + θ where θ ≈ -0.55 is the MA(1) coefficient from ARIMA(0,1,1)
    # Derivation: for ARIMA(0,1,1), optimal EWMA alpha = 1 + theta
    TOMATO_EMA_ALPHA  = 0.43
    TOMATO_SPREAD     = 2       # base passive half-spread
    TOMATO_LIMIT      = 80
    TOMATO_SKEW       = 0.15    # ticks per unit of position
                                # (BUG: at limit=80 → 12 ticks, wider than spread)
    TOMATO_VOL_WINDOW = 15      # rolling window for σ(ΔY_t) volatility estimate

    def run(self, state: TradingState):
        result = {}

        # ── Restore persisted state ──
        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {
                "tomato_ema": None,      # EMA fair value (ARIMA-optimal)
                "tomato_last_mid": None,  # for volatility tracking
                "tomato_diffs": [],       # recent ΔY for vol estimation
            }

        for product in state.order_depths:
            od: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(od, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ──────────────────────────────────────────────────────────────────
    #  EMERALDS — Stationary process, known fair value = 10,000
    #  DF t-stat ≈ -97 → overwhelmingly reject unit root
    #  ACF(1) of levels ≈ 0.03 → white noise → fair = 10,000 is certain
    # ──────────────────────────────────────────────────────────────────
    def trade_emeralds(self, od: OrderDepth, position: int) -> List[Order]:
        """
        EMERALDS strategy: aggressive market making around the known fair value.

        Inventory skew (Avellaneda-Stoikov style):
          Shifts our effective fair value against our current position.
          Long → lower effective_fair → quote cheaper to sell, more expensive to buy.
          Short → raise effective_fair → quote cheaper to buy, more expensive to sell.
          skew = EMERALD_SKEW * position / limit   (max 0.5 ticks at limit)

        BUG (fixed in v3): we use effective_fair (skewed) as the aggressive threshold.
          At position=80, effective_fair = 10000 - 0.5 = 9999.5
          So we refuse to buy at 9999 even though fair=10000 → that is a guaranteed
          profitable trade we are passing up. Aggressive fills should never be skewed.

        Phase 1: Aggressive — take any ask < effective_fair, sell into any bid > effective_fair.
        Phase 2: Passive — post at effective_fair ± adaptive_spread.
          Spread widens slightly with position fraction to reduce fill rate when risky.
        """
        orders: List[Order] = []
        fair = self.EMERALD_FAIR
        limit = self.EMERALD_LIMIT
        half_spread = self.EMERALD_SPREAD

        # Inventory skew shifts our reference price against our position
        skew = self.EMERALD_SKEW * position / limit
        effective_fair = fair - skew   # BUG: should not be used for aggressive phase

        # Phase 1: Aggressive — take resting orders mispriced vs effective_fair
        # (BUG: should use raw fair=10000 here, not effective_fair)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= effective_fair:
                break
            ask_vol = od.sell_orders[ask_price]  # negative quantity in sell_orders
            can_buy = limit - position
            if can_buy <= 0:
                break
            buy_qty = min(-ask_vol, can_buy)
            orders.append(Order("EMERALDS", ask_price, buy_qty))
            position += buy_qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= effective_fair:
                break
            bid_vol = od.buy_orders[bid_price]
            can_sell = limit + position
            if can_sell <= 0:
                break
            sell_qty = min(bid_vol, can_sell)
            orders.append(Order("EMERALDS", bid_price, -sell_qty))
            position -= sell_qty

        # Phase 2: Passive — post quotes around effective_fair with adaptive spread
        # Spread widens by 1 when position is large (reduces fill rate → less inventory risk)
        pos_frac = abs(position) / limit
        adaptive_spread = half_spread + round(pos_frac * 1)

        buy_price  = round(effective_fair) - adaptive_spread
        sell_price = round(effective_fair) + adaptive_spread

        remaining_buy  = limit - position
        remaining_sell = limit + position

        if remaining_buy > 0:
            orders.append(Order("EMERALDS", buy_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("EMERALDS", sell_price, -remaining_sell))

        return orders

    # ──────────────────────────────────────────────────────────────────
    #  TOMATOES — ARIMA(0,1,1) process, theory-derived optimal forecast
    #
    #  Identification (Box-Jenkins methodology):
    #    ACF(levels) ≈ 0.99 → near unit root → difference the series
    #    ACF(ΔY_t): ACF(1) ≈ -0.41, ACF(k≥2) ≈ 0 → MA(1) in differences
    #    → ARIMA(0,1,1): ΔY_t = ε_t + θ·ε_{t-1}, θ ≈ -0.55
    #    → Optimal 1-step EWMA: α = 1 + θ ≈ 0.43
    #    → Empirical check: EMA(0.43) has 53% lower MSE than SMA(20)
    #
    #  Input signal: microprice (not raw midprice)
    #    microprice = (bid × ask_vol + ask × bid_vol) / (bid_vol + ask_vol)
    #    Better estimate of next trade price than the geometric midpoint.
    # ──────────────────────────────────────────────────────────────────
    def trade_tomatoes(
        self, od: OrderDepth, position: int, saved: dict
    ) -> List[Order]:
        """
        TOMATOES strategy: ARIMA(0,1,1)-optimal EMA forecast with inventory skewing.

        State persisted in saved dict across ticks:
          tomato_ema:      current EMA value (our fair value estimate)
          tomato_last_mid: previous midprice (to compute ΔY for vol estimation)
          tomato_diffs:    recent list of ΔY values for rolling volatility

        Microprice is used instead of midprice as the EMA input because it
        accounts for order book imbalance — if there is more ask-side volume,
        the next trade is more likely to occur at the ask, so microprice
        weights the ask more heavily.

        Adaptive spread:
          half_spread = base + vol_adj + inv_adj
          vol_adj = max(0, round(recent_vol - 1.0)) — widen in volatile periods
          inv_adj = round(pos_frac * 2)              — widen with large inventory

        BUG 1 (fixed in v3): aggressive threshold uses effective_fair (skewed).
          At position=80, skew = 0.15*80 = 12 ticks → we miss buys at fair-11
          to fair-1, all of which are profitable. Should use raw fair.

        BUG 2 (fixed in v3): TOMATO_SKEW=0.15 → 12 ticks max skew.
          Spread = 4 ticks total. Skew alone exceeds spread → nonsensical quotes.

        BUG 3 (fixed in v3): half_spread can reach 6 → full spread 12 ticks.
          Wider than the order book → passive quotes never fill.
        """
        orders: List[Order] = []
        limit = self.TOMATO_LIMIT

        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        # Microprice: volume-weighted fair value signal (better than raw midprice)
        # Rationale: heavier ask side → price more likely to trade at ask → weight ask more
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]
        if bid_vol + ask_vol > 0:
            microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            microprice = mid

        # EMA update: ARIMA(0,1,1)-optimal forecast with α = 1 + θ ≈ 0.43
        # This is derived from theory, not fitted to the data
        alpha = self.TOMATO_EMA_ALPHA
        if saved["tomato_ema"] is None:
            saved["tomato_ema"] = microprice
        else:
            saved["tomato_ema"] = alpha * microprice + (1 - alpha) * saved["tomato_ema"]

        fair = saved["tomato_ema"]   # raw unskewed EMA fair value

        # Rolling volatility: σ(ΔY_t) over the last TOMATO_VOL_WINDOW ticks
        # Used to adaptively widen the spread when the market is more volatile
        if saved["tomato_last_mid"] is not None:
            diff = mid - saved["tomato_last_mid"]
            saved["tomato_diffs"].append(diff)
            if len(saved["tomato_diffs"]) > self.TOMATO_VOL_WINDOW:
                saved["tomato_diffs"].pop(0)
        saved["tomato_last_mid"] = mid

        if len(saved["tomato_diffs"]) >= 5:
            recent_vol = (sum(d**2 for d in saved["tomato_diffs"]) / len(saved["tomato_diffs"])) ** 0.5
        else:
            recent_vol = 1.3   # prior from empirical data analysis

        # Inventory skew: shift effective_fair against our position (Avellaneda-Stoikov)
        # BUG 2: TOMATO_SKEW=0.15 → max 12 ticks at limit — far too large
        skew = self.TOMATO_SKEW * position
        effective_fair = fair - skew

        # Phase 1: Aggressive — take resting orders vs effective_fair (BUG 1: should use raw fair)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= effective_fair:   # BUG: should be >= fair
                break
            ask_vol = od.sell_orders[ask_price]
            can_buy = limit - position
            if can_buy <= 0:
                break
            buy_qty = min(-ask_vol, can_buy)
            orders.append(Order("TOMATOES", ask_price, buy_qty))
            position += buy_qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= effective_fair:   # BUG: should be <= fair
                break
            bid_vol = od.buy_orders[bid_price]
            can_sell = limit + position
            if can_sell <= 0:
                break
            sell_qty = min(bid_vol, can_sell)
            orders.append(Order("TOMATOES", bid_price, -sell_qty))
            position -= sell_qty

        # Phase 2: Passive quotes — adaptive spread around effective_fair
        # BUG 3: no cap on half_spread → can grow very wide in volatile periods
        pos_frac = abs(position) / limit
        vol_adj    = max(0, round(recent_vol - 1.0))   # wider when σ(ΔY) > 1
        inv_adj    = round(pos_frac * 2)                # wider when position is large
        half_spread = self.TOMATO_SPREAD + vol_adj + inv_adj   # BUG: uncapped

        buy_price  = int(round(effective_fair)) - half_spread
        sell_price = int(round(effective_fair)) + half_spread

        remaining_buy  = limit - position
        remaining_sell = limit + position

        if remaining_buy > 0:
            orders.append(Order("TOMATOES", buy_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("TOMATOES", sell_price, -remaining_sell))

        return orders
