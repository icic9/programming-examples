"""
IMC Prosperity 4 — Round 1 Trader v1
=====================================
Products:
  INTARIAN_PEPPER_ROOT  — ARIMA(0,1,1) with strong uptrend (δ ≈ +0.10 ticks/tick)
  ASH_COATED_OSMIUM     — ARIMA(0,1,1) zero-drift (δ ≈ 0.00)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATISTICAL ANALYSIS OF 3-DAY TRAINING DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ACF(1) of first differences:
    PEPPER:  −0.5006  →  optimal EMA alpha = 1 + ACF(1) = 0.4994 ≈ 0.50
    OSMIUM:  −0.4951  →  optimal EMA alpha = 1 + ACF(1) = 0.5049 ≈ 0.50

  Both products: ARIMA(0,1,1) with θ ≈ −0.50, MA coefficient θ ≈ −0.50.
  Optimal EMA formula: α = 1 + θ (Muth 1960 equivalence).

  In Tutorial round (TOMATOES), ACF(1) ≈ −0.57 → α = 0.43 was used.
  For Round 1, α = 0.50 is directly derived from the actual price data.

  Drift per timestamp:
    PEPPER:  +0.1002 ticks  →  +3,001 ticks over 3 training days (strong uptrend)
    OSMIUM:  −0.0001 ticks  →  effectively zero (random walk, no directional bias)

  Bid-ask spread statistics (level-1):
    PEPPER:  day -2 mean=12.0, day 0 mean=14.1  (range 2–21)
    OSMIUM:  day -2 mean=16.2, day 0 mean=16.2  (range 5–22)

  Diff standard deviation:
    PEPPER:  3.10   OSMIUM:  3.73

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRATEGY: INTARIAN_PEPPER_ROOT (TRENDING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Inherits all innovations from Tutorial v6 (TOMATOES):
    - ARIMA(0,1,1)-optimal EMA for fair value
    - Dual-EMA trend signal (fast − slow)
    - Trend-carry fair value (forecasts ahead of the lagged EMA)
    - Urgency position skew (2× multiplier at 70% of limit)
    - Exponential moving variance (replaces rolling list)
    - Book-adaptive pennying (Innovation 1)
    - Two-level passive layering (Innovation 2)
    - Against-trend passive capacity throttling

  Round 1 recalibrations vs Tutorial:

  1. alpha = 0.50 (vs 0.43):
       ACF(1) = −0.501 → θ = −0.501 → optimal α = 0.499 ≈ 0.50.
       Tutorial had ACF(1) = −0.43 (approximately); Round 1 analysis is direct.

  2. MOMENTUM_BIAS = 40 (vs 5 in Tutorial):
       Steady-state trend signal calculation:
         fast_lag = δ × (1−α_fast)/α_fast = 0.10 × (0.50/0.50) = 0.10 ticks
         slow_lag = δ × (1−α_slow)/α_slow = 0.10 × (0.94/0.06) = 1.567 ticks
         trend_signal ≈ slow_lag − fast_lag = 1.467 ticks in steady uptrend

       With MOMENTUM_BIAS = 40:
         biased_pos = position − 40 × 1.467 = position − 58.7
         At position=0: biased_pos ≈ −59 > urgency threshold (56 = 70% × 80)
         → urgency zone activated even at zero position in steady trend
         → skew × 2 = −4 ticks → eff_ema = forecast_ema + 4 → max buy bias
         → drives position accumulation toward long limit in uptrend

  3. TREND_GATE = 1.0 (vs 1.5 in Tutorial):
       Steady-state signal ≈ 1.47 ticks. Gate of 1.5 would barely ever activate.
       Gate of 1.0 activates when trend is about 68% of steady-state magnitude,
       ensuring against-trend throttling engages promptly.

  4. AGAINST_TREND_CAP = 0.55 (vs 0.65):
       In uptrend, passive sell capacity capped at 55% (vs 65%).
       More aggressive position accumulation on trend confirmation.
       Balanced to retain 55% spread income even in strong trend.

  5. VOL_THRESH = 4.0 (vs 2.5):
       diff_std = 3.10 for PEPPER. Threshold of 2.5 would trigger almost always,
       making min_edge=3 permanent. At 4.0, only exceptionally volatile moments
       (above 1.3σ) trigger the wider min_edge=2 fallback.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRATEGY: ASH_COATED_OSMIUM (STATIONARY / ZERO-DRIFT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Analogous to EMERALDS in Tutorial, but the true fair value is unknown.

  Tutorial EMERALDS: Dickey-Fuller t-stat ≈ −97 → overwhelmingly stationary
    → fixed fair = 10,000 used as anchor.

  OSMIUM: ARIMA(0,1,1) with zero drift. Not strictly stationary in level
    (unit root present), but has no directional bias. The optimal forecast
    is the current EMA(α=0.50), which adapts rapidly to any level shift.

  Architecture: identical to EMERALDS v4 logic, but:
    - fair = EMA(α=0.50) instead of fixed 10,000
    - Robust to one-sided book at market open (day -2 t=0: only ask side)
    - No trend signal, no momentum bias, no against-trend throttling
    - Max inventory skew: ±1 tick at position limit (via rounding)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COMBINED EFFECT IN STEADY UPTREND (PEPPER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Aggressive buys: any ask < ema + 0.37 (trend-carry of 0.25 × 1.47)
     — catches asks priced at the stale (lagged) ema level.

  2. Passive buys: full capacity (no throttle in uptrend direction).
     Posted at comp_bid+1 (penny, 5–6 ticks below fair given 12-tick spread).
     Strong fill rate due to queue priority.

  3. Passive sells: capacity throttled to 55% of normal.
     Posted at comp_ask−1 (penny). Only 55% of sell capacity used.

  4. Net inventory drift: buy fills >> sell fills → position ramps to limit.

  5. Near limit (position=72): biased_pos = 72 − 59 = 13 → neutral skew
     → normal market-making resumes near the position ceiling.

  In steady state with position=limit=80:
    biased_pos = 80 − 59 = 21 → positive skew (sell bias) → eff_ema < fair
    → sell orders post slightly below fair → natural unwinding at limit.
    Prevents hard position-limit breaches while maintaining spread income.
"""

import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from Tutorial v6) ────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations):
        conv = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                    o.importTariff, o.sugarPrice, o.sunlightIndex]
                for p, o in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity]
                for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out, lo = candidate, mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ── Trader ──────────────────────────────────────────────────────────────────
class Trader:
    """
    Round 1 v1: Dual-EMA trend-riding for PEPPER + EMA market-making for OSMIUM.

    PEPPER: All Tutorial v6 innovations retained; parameters recalibrated from
      3-day price data (ACF analysis gives α=0.50; drift=0.10 gives BIAS=40).
    OSMIUM: EMERALDS-analog with adaptive EMA fair value (no fixed 10,000 known).
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── PEPPER (INTARIAN_PEPPER_ROOT) ─────────────────────────────────────
    PEPPER_LIMIT          = 80
    PEPPER_EMA_ALPHA      = 0.28    # v2 sweep: slower fair catches the gradual Round 1 trend better
    PEPPER_EMA_SLOW_ALPHA = 0.015   # v2 sweep: much slower trend baseline for PEPPER structure
    PEPPER_SKEW           = 0.015   # v2 sweep: gentler inventory skew avoids fading the trend too hard
    PEPPER_SPREAD_BASE    = 2
    PEPPER_SPREAD_HIGH    = 1       # +1 min_edge in high-vol moments
    PEPPER_VOL_THRESH     = 4.0     # raised from Tutorial 2.5; diff_std=3.10 → only spikes trigger
    PEPPER_L1_FRAC        = 1.0     # one strategic passive price beat two-level splitting in R1 data
    PEPPER_L2_FRAC        = 0.4
    PEPPER_EMV_ALPHA      = 0.15    # exponential moving variance decay
    PEPPER_MOMENTUM_BIAS    = 0     # Round 1 sweep: position skew outperformed trend-to-position forcing
    PEPPER_URGENCY_THRESH   = 0.70  # |biased_pos| / limit threshold
    PEPPER_URGENCY_MULT     = 3.0   # paired with lower base skew; only pushes when inventory is stretched
    PEPPER_TREND_CARRY      = 0.55  # v2 sweep optimum with the slower EMA pair
    PEPPER_TREND_GATE       = 0.0   # always apply against-trend throttling once trend direction is nonzero
    PEPPER_AGAINST_TREND_CAP = 0.55 # vs 0.65; stronger throttle on against-trend passive side
    PEPPER_CARRY_EDGE       = 0.0   # tested paid carry; passive accumulation was better net of spread
    PEPPER_LONG_FLOOR       = 65    # keep core long carry; only sell PEPPER inventory above this floor
    PEPPER_MM_VOLUME        = 18    # large-quote fair: ignore small top-of-book noise
    PEPPER_JOIN_EDGE        = 0.0

    # ── OSMIUM (ASH_COATED_OSMIUM) ────────────────────────────────────────
    OSMIUM_LIMIT     = 80
    OSMIUM_EMA_ALPHA = 0.40         # Slightly slower EMA improved the three Round 1 backtest days
    OSMIUM_SKEW      = 3.0          # Stronger inventory skew added a small, stable OSMIUM gain
    OSMIUM_L1_FRAC   = 0.75
    OSMIUM_L2_FRAC   = 0.4
    OSMIUM_MM_VOLUME  = 0           # use plain quote mid; better than microprice on R1 OSMIUM
    OSMIUM_JOIN_EDGE  = 0.0

    # ── shared helpers (identical to Tutorial v6) ─────────────────────────────

    @staticmethod
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,
        fair: float,
        min_buy_edge: int,
        min_sell_edge: int,
        join_edge: float,
    ):
        """
        Book-adaptive pennying (Innovation 1, unchanged).

        Scan competitor resting bids below raw fair → post 1 tick above.
        Scan competitor resting asks above raw fair → post 1 tick below.
        Hard clamps prevent crossing the (possibly skewed) reference.
        """
        comp_bids = [p for p in od.buy_orders  if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        if comp_bids:
            best_bid_below = max(comp_bids)
            penny_buy = best_bid_below if fair - best_bid_below <= join_edge else best_bid_below + 1
        else:
            penny_buy = int(ref) - min_buy_edge - 1

        passive_buy = min(penny_buy, int(ref) - min_buy_edge)

        if comp_asks:
            best_ask_above = min(comp_asks)
            penny_sell = best_ask_above if best_ask_above - fair <= join_edge else best_ask_above - 1
        else:
            penny_sell = int(ref) + min_sell_edge + 1

        passive_sell = max(penny_sell, int(ref) + min_sell_edge)

        return passive_buy, passive_sell

    @staticmethod
    def _two_level_orders(
        symbol: str,
        passive_buy: int, passive_sell: int,
        remaining_buy: int, remaining_sell: int,
        l1_frac: float, l2_frac: float,
    ) -> List[Order]:
        """
        Two-level passive layering (Innovation 2, unchanged).

        L1 (60%): penny price — queue priority.
        L2 (40%): penny ± 1 tick — backup edge, survives bot tightening.
        """
        orders = []

        if remaining_buy > 0:
            l1_qty = max(1, round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            orders.append(Order(symbol, passive_buy,     l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))

        if remaining_sell > 0:
            l1_qty = max(1, round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            orders.append(Order(symbol, passive_sell,     -l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))

        return orders

    # ── run ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _large_quote_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        large_bids = [p for p, v in od.buy_orders.items() if abs(v) >= min_volume]
        large_asks = [p for p, v in od.sell_orders.items() if abs(v) >= min_volume]
        if large_bids and large_asks:
            return (max(large_bids) + min(large_asks)) / 2.0
        return fallback

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {
                "pepper_ema":      None,
                "pepper_ema_slow": None,
                "pepper_last_mid": None,
                "pepper_emv":      None,
                "osmium_ema":      None,
            }

        saved.setdefault("pepper_ema",      None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv",      None)
        saved.setdefault("osmium_ema",      None)

        for product in state.order_depths:
            od       = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == self.PEPPER:
                result[product] = self._trade_pepper(od, position, saved)
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OSMIUM ───────────────────────────────────────────────────────────────

    def _trade_osmium(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        """
        OSMIUM v1: EMERALDS-analog with EMA-based fair value.

        Unlike Tutorial EMERALDS (known fair=10,000), OSMIUM is an ARIMA(0,1,1)
        random walk with unknown mean. We use EMA(α=0.50) — the ARIMA-optimal
        predictor — as the adaptive fair value. The market-making logic is
        otherwise identical to EMERALDS v4.

        Robust to one-sided order book at market open (day -2, t=0: ask only).

        Phase 1 — Aggressive:
          Fill any resting order that crosses our EMA fair value.
          Profitable fills regardless of current inventory.

        Phase 2 — Passive:
          Inventory skew: round(SKEW × position / limit) ∈ {−1, 0, +1}.
          Max ±1 tick at position limit. Book-adaptive pennying + two-level layering.
        """
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT

        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        # Level-1 microprice (robust to one-sided book)
        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            mid = (best_bid + best_ask) / 2.0
            microprice = (best_bid * av + best_ask * bv) / (bv + av) if bv + av > 0 else mid
        elif has_bid:
            best_bid = max(od.buy_orders)
            microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            microprice = float(best_ask)
        ref_price = self._large_quote_mid(od, microprice, self.OSMIUM_MM_VOLUME)

        # EMA fair value update
        alpha = self.OSMIUM_EMA_ALPHA
        if saved["osmium_ema"] is None:
            saved["osmium_ema"] = ref_price
        else:
            saved["osmium_ema"] = alpha * ref_price + (1.0 - alpha) * saved["osmium_ema"]

        fair = saved["osmium_ema"]

        # Phase 1: Aggressive
        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= fair:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                orders.append(Order(self.OSMIUM, ask, qty))
                position += qty

        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= fair:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                orders.append(Order(self.OSMIUM, bid, -qty))
                position -= qty

        # Phase 2: Passive — inventory skew max ±1 tick
        skew     = round(self.OSMIUM_SKEW * position / limit)
        eff_fair = float(fair - skew)

        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_fair, fair=fair,
            min_buy_edge=1, min_sell_edge=1,
            join_edge=self.OSMIUM_JOIN_EDGE,
        )

        orders += self._two_level_orders(
            self.OSMIUM,
            passive_buy, passive_sell,
            limit - position, limit + position,
            self.OSMIUM_L1_FRAC, self.OSMIUM_L2_FRAC,
        )

        logger.print(
            f"OSM | mp={microprice:.1f} ema={fair:.2f} "
            f"skew={skew} eff={eff_fair:.2f} "
            f"pbuy={passive_buy} psell={passive_sell} pos={position}"
        )

        return orders

    # ── PEPPER ───────────────────────────────────────────────────────────────

    def _trade_pepper(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        """
        PEPPER v1: Tutorial v6 TOMATOES logic with Round 1 recalibration.

        Changes from Tutorial TOMATOES v6:
          α = 0.50    (vs 0.43): directly derived from ACF(1)=−0.501.
          MOMENTUM_BIAS = 40  (vs 5):  steady-state signal ≈ 1.47 ticks →
            biased_pos shift = 58.7 units → urgency zone at position=0.
          TREND_GATE = 1.0  (vs 1.5): throttle activates at 68% of steady signal.
          AGAINST_TREND_CAP = 0.55  (vs 0.65): stronger sell throttle in uptrend.
          VOL_THRESH = 4.0  (vs 2.5): avoids permanent high-vol mode.

        All v6 innovations retained (unchanged logic):
          - Trend-carry fair value (forecast_ema = ema + 0.25 × trend_signal)
          - Dual-EMA trend signal and urgency position skew
          - Exponential moving variance (EMV)
          - Book-adaptive pennying + two-level layering
          - Against-trend passive capacity throttling

        Phase 1 — Aggressive (uses forecast_ema):
          Buy any ask < forecast_ema; sell any bid > forecast_ema.
          Trend-carry ensures we don't buy at a stale lagged EMA.

        Phase 2 — Passive:
          biased_pos = position − MOMENTUM_BIAS × trend_signal
          In steady uptrend at position=0: biased_pos ≈ −59 → urgency zone.
          skew = 0.025 × biased_pos × 2 (urgency) ≈ −3 ticks.
          eff_ema = forecast_ema + 3 → willing to buy 2 ticks above raw ema.
          (In practice: comp_bid typically at ema−6, so penny = ema−5 dominates.)
          Against-trend throttle: remaining_sell × 0.55 in uptrend.
          Net effect: full buy capacity, 55% sell capacity → position ramps long.
        """
        orders: List[Order] = []
        limit  = self.PEPPER_LIMIT

        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        # Level-1 microprice (robust to one-sided book)
        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            mid = (best_bid + best_ask) / 2.0
            microprice = (best_bid * av + best_ask * bv) / (bv + av) if bv + av > 0 else mid
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)
        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)

        # Fast EMA — ARIMA(0,1,1) optimal (α=0.50)
        a_fast = self.PEPPER_EMA_ALPHA
        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            saved["pepper_ema"] = a_fast * ref_price + (1.0 - a_fast) * saved["pepper_ema"]
        ema = saved["pepper_ema"]

        # Slow EMA — sustained trend tracker
        a_slow = self.PEPPER_EMA_SLOW_ALPHA
        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            saved["pepper_ema_slow"] = (
                a_slow * ref_price + (1.0 - a_slow) * saved["pepper_ema_slow"]
            )
        ema_slow = saved["pepper_ema_slow"]

        # Trend signal and trend-carry fair value
        trend_signal = ema - ema_slow
        forecast_ema = ema + self.PEPPER_TREND_CARRY * trend_signal

        # Exponential moving variance
        if saved["pepper_last_mid"] is not None:
            diff_sq  = (mid - saved["pepper_last_mid"]) ** 2
            emv_prev = saved["pepper_emv"]
            if emv_prev is None:
                saved["pepper_emv"] = diff_sq
            else:
                a_v = self.PEPPER_EMV_ALPHA
                saved["pepper_emv"] = a_v * diff_sq + (1.0 - a_v) * emv_prev
        saved["pepper_last_mid"] = mid

        emv = saved["pepper_emv"]
        vol  = emv ** 0.5 if emv is not None else 2.0
        spread_extra = self.PEPPER_SPREAD_HIGH if vol > self.PEPPER_VOL_THRESH else 0

        # Phase 1: Aggressive — trend-carry fair value threshold
        carry_edge = self.PEPPER_CARRY_EDGE if trend_signal >= -0.5 else 0.0
        buy_threshold = forecast_ema + carry_edge
        sell_threshold = forecast_ema + carry_edge

        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= buy_threshold:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                orders.append(Order(self.PEPPER, ask, qty))
                position += qty

        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= sell_threshold:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                orders.append(Order(self.PEPPER, bid, -qty))
                position -= qty

        # Phase 2: Passive — trend-biased, urgency-scaled inventory skew

        biased_pos = position - self.PEPPER_MOMENTUM_BIAS * trend_signal
        biased_pos = max(-limit, min(limit, biased_pos))

        skew = self.PEPPER_SKEW * biased_pos
        if abs(biased_pos) > self.PEPPER_URGENCY_THRESH * limit:
            skew *= self.PEPPER_URGENCY_MULT

        eff_ema = forecast_ema - skew

        min_edge = 1 + spread_extra
        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_ema, fair=forecast_ema,
            min_buy_edge=min_edge, min_sell_edge=min_edge,
            join_edge=self.PEPPER_JOIN_EDGE,
        )

        # Against-trend passive capacity throttling
        remaining_buy  = limit - position
        remaining_sell = limit + position
        if trend_signal < -self.PEPPER_TREND_GATE:
            remaining_buy  = int(round(remaining_buy  * self.PEPPER_AGAINST_TREND_CAP))
        elif trend_signal > self.PEPPER_TREND_GATE:
            remaining_sell = int(round(remaining_sell * self.PEPPER_AGAINST_TREND_CAP))
            remaining_sell = min(remaining_sell, max(0, position - self.PEPPER_LONG_FLOOR))

        orders += self._two_level_orders(
            self.PEPPER,
            passive_buy, passive_sell,
            remaining_buy, remaining_sell,
            self.PEPPER_L1_FRAC, self.PEPPER_L2_FRAC,
        )

        logger.print(
            f"PEP | mid={mid:.1f} mp={microprice:.1f} ema={ema:.2f} "
            f"slow={ema_slow:.2f} sig={trend_signal:.2f} fcst={forecast_ema:.2f} "
            f"bpos={biased_pos:.1f} skew={skew:.2f} eff={eff_ema:.2f} "
            f"vol={vol:.2f} pbuy={passive_buy} psell={passive_sell} pos={position}"
        )

        return orders
