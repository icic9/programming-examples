"""
IMC Prosperity 4 — Strategy v3: Systematic Debugging
======================================================
Products: EMERALDS (stationary ~10000) | TOMATOES (ARIMA(0,1,1) drift)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONTEXT: RIGOUR IN EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The econometric foundation from v2 is sound, but backtesting and log
  analysis revealed four implementation bugs that silently reduced PnL.
  None of them raised exceptions — they just caused us to trade worse.
  This version identifies each bug precisely and fixes it.

  Econometric basis carried forward (unchanged from v2):
    EMERALDS : DF t-stat ≈ -97 → white noise → fair = 10,000 with certainty
    TOMATOES : ACF signature → ARIMA(0,1,1), θ ≈ -0.55
               Optimal forecast = EMA with α = 1 + θ ≈ 0.43
               Microprice as EMA input (better than raw midprice)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE PRINCIPLE ESTABLISHED HERE (carried into v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Aggressive fills and passive quoting serve fundamentally different purposes
  and must use different fair value references:

    AGGRESSIVE phase → use raw (unskewed) fair value.
      These are certain-edge trades: the order book is offering us a better
      price than our estimate of fair value. We should always take them,
      regardless of our current inventory level. Applying inventory skew
      here means we refuse profitable trades — that is never correct.

    PASSIVE phase → use skewed (inventory-adjusted) fair value.
      These are probabilistic trades: we post limit orders hoping to earn
      the spread. Here, skewing toward inventory reduction is appropriate
      because we are choosing where to advertise our willingness to trade.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BUG FIXES (all four bugs identified from log analysis and backtesting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  BUG 1 — TOMATOES: aggressive threshold used skewed effective_fair.
    SYMPTOM: At position=80 with TOMATO_SKEW=0.15, skew = 0.15×80 = 12.
             effective_fair = fair − 12.
             We only buy asks below fair-12. But asks at fair-11 to fair-1
             are all profitable (below our true fair value) — we miss them.
    FIX: aggressive threshold always compares against raw fair (unskewed).

  BUG 2 — TOMATOES: inventory skew coefficient was too large (0.15/unit).
    SYMPTOM: At position=80, skew = 12 ticks. Total spread ≈ 4 ticks.
             Passive sell price = eff_fair + 2 = (fair-12)+2 = fair-10.
             We were selling at 10 ticks BELOW fair — gifting edge away.
             Skew should always be smaller than the half-spread.
    FIX: TOMATO_SKEW reduced from 0.15 to 0.04 → max 3.2 ticks at limit.

  BUG 3 — TOMATOES: adaptive spread had no cap.
    SYMPTOM: vol_adj up to 2, inv_adj up to 2 → half_spread up to 6.
             Full spread = 12 ticks. Our passive quotes sit far outside
             any resting orders in the book → zero passive fills.
    FIX: hard cap of 4 ticks on the half-spread.

  BUG 4 — EMERALDS: aggressive fills used skewed effective_fair.
    SYMPTOM: EMERALDS fair is known with certainty = 10,000.
             At position=80, effective_fair ≈ 9,999.5.
             We refuse to buy at 9,999 even though fair=10,000.
             That is a guaranteed +0.5 tick of edge — we were leaving it.
    FIX: aggressive EMERALDS fills always use raw fair = 10,000. Skew is
         applied to passive quote prices only.
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

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity,
                                    t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations):
        conv = {}
        for product, obs in observations.conversionObservations.items():
            conv[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                             obs.exportTariff, obs.importTariff,
                             obs.sugarPrice, obs.sunlightIndex]
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


class Trader:
    """
    v3: Bug-fixed, econometrically-grounded market maker.

    All econometric foundations from v2 are preserved.
    The key architectural fix: aggressive fills use raw fair; passive quotes use skewed fair.
    """

    # ── EMERALDS ─────────────────────────────────────────────────────────
    EMERALD_FAIR   = 10_000
    EMERALD_LIMIT  = 80
    EMERALD_SPREAD = 2      # passive half-spread in ticks
    EMERALD_SKEW   = 1.0    # max ticks of passive quote skew at full position limit
                            # (BUG 4 fix: skew now applied ONLY to passive phase)

    # ── TOMATOES — from ARIMA(0,1,1) identification ───────────────────
    TOMATO_LIMIT      = 80
    TOMATO_EMA_ALPHA  = 0.43   # α = 1 + θ, θ ≈ -0.55 (derived from ACF)
    TOMATO_SPREAD     = 2      # base passive half-spread in ticks
    TOMATO_SKEW       = 0.04   # ticks per position unit; max = 0.04×80 = 3.2 ticks
                               # (BUG 2 fix: was 0.15 → 12 ticks, wider than spread)
    TOMATO_VOL_WINDOW = 15
    TOMATO_SPREAD_CAP = 4      # hard cap on half-spread (BUG 3 fix)

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {
                "tomato_ema":      None,
                "tomato_last_mid": None,
                "tomato_diffs":    [],
            }

        for product in state.order_depths:
            od       = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(od, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── EMERALDS ─────────────────────────────────────────────────────────
    def trade_emeralds(self, od: OrderDepth, position: int) -> List[Order]:
        """
        EMERALDS: known fair value = 10,000 (DF t-stat ≈ -97, white noise).

        Phase 1 — Aggressive (BUG 4 fix applied here):
          Always compare resting orders against raw fair = 10,000.
          We must never let inventory level stop us from taking a
          guaranteed-profit trade (e.g., buying at 9,999 when fair=10,000).

        Phase 2 — Passive:
          Apply a small inventory skew to the quote reference: max ±1 tick
          at the position limit. This gently nudges our quotes toward
          inventory reduction without sacrificing meaningful edge.
          skew = EMERALD_SKEW * position / limit   (∈ [-1, +1] ticks)
          Quote prices: round(eff_fair) ± EMERALD_SPREAD.
        """
        orders: List[Order] = []
        fair  = self.EMERALD_FAIR
        limit = self.EMERALD_LIMIT

        # Phase 1: aggressive — raw fair only, never skew ─────────────────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair:
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask_price], can_buy)
            orders.append(Order("EMERALDS", ask_price, qty))
            position += qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair:
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            qty = min(od.buy_orders[bid_price], can_sell)
            orders.append(Order("EMERALDS", bid_price, -qty))
            position -= qty

        # Phase 2: passive — small inventory skew on quote prices only ────
        # skew in [-1, +1] ticks; when long, lower both quotes slightly
        skew       = self.EMERALD_SKEW * position / limit   # max abs = 1 tick
        eff_fair   = fair - skew
        buy_price  = round(eff_fair) - self.EMERALD_SPREAD
        sell_price = round(eff_fair) + self.EMERALD_SPREAD

        if limit - position > 0:
            orders.append(Order("EMERALDS", buy_price,   limit - position))
        if limit + position > 0:
            orders.append(Order("EMERALDS", sell_price, -(limit + position)))

        return orders

    # ── TOMATOES ─────────────────────────────────────────────────────────
    def trade_tomatoes(
        self, od: OrderDepth, position: int, saved: dict
    ) -> List[Order]:
        """
        TOMATOES: ARIMA(0,1,1) process. Optimal forecast = EMA(α=0.43) of microprice.

        KEY ARCHITECTURAL SEPARATION (fixes BUG 1 and BUG 2):
          Aggressive threshold → raw EMA fair (unskewed).
            Any ask below fair is a profitable fill — take it regardless of inventory.
          Passive quote prices → skewed eff_fair (inventory-adjusted).
            Lean our limit order placement against our position to reduce it gently.

        This separation ensures we never miss a genuinely mispriced order while
        still managing inventory risk through our passive quote placement.

        Adaptive spread (fixes BUG 3):
          half_spread = min(base + vol_adj + inv_adj, TOMATO_SPREAD_CAP)
          vol_adj: 0, 1, or 2 depending on realised σ(ΔY_t) vs thresholds.
          inv_adj: 0 or 1 depending on whether |position|/limit > 0.6.
          Hard cap of 4 ensures quotes are always close enough to fill.

        Microprice as EMA input: better than raw midprice.
          microprice = (best_bid × ask_vol + best_ask × bid_vol) / total_vol
          If ask side is heavier, microprice sits closer to the ask price —
          accurately reflecting where the next trade will likely occur.
        """
        orders: List[Order] = []
        limit  = self.TOMATO_LIMIT

        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        # Microprice: volume-weighted, better than raw mid ─────────────────
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]
        microprice = (
            (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
            if bid_vol + ask_vol > 0 else mid
        )

        # EMA update — ARIMA(0,1,1)-optimal ───────────────────────────────
        a = self.TOMATO_EMA_ALPHA
        if saved["tomato_ema"] is None:
            saved["tomato_ema"] = microprice
        else:
            saved["tomato_ema"] = a * microprice + (1.0 - a) * saved["tomato_ema"]

        fair = saved["tomato_ema"]   # raw, unskewed fair value

        # Rolling volatility ───────────────────────────────────────────────
        if saved["tomato_last_mid"] is not None:
            saved["tomato_diffs"].append(mid - saved["tomato_last_mid"])
            if len(saved["tomato_diffs"]) > self.TOMATO_VOL_WINDOW:
                saved["tomato_diffs"].pop(0)
        saved["tomato_last_mid"] = mid

        if len(saved["tomato_diffs"]) >= 5:
            recent_vol = (
                sum(d ** 2 for d in saved["tomato_diffs"])
                / len(saved["tomato_diffs"])
            ) ** 0.5
        else:
            recent_vol = 1.3

        # Inventory skew — passive quotes only (BUG 2 fix: 0.04 not 0.15) ─
        # max abs = 0.04 * 80 = 3.2 ticks (always within the spread)
        skew     = self.TOMATO_SKEW * position
        eff_fair = fair - skew

        # Phase 1: Aggressive — always compare against raw fair (BUG 1 fix)
        # Any ask below fair = profitable trade, regardless of our inventory.
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair:            # raw fair, NOT eff_fair
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask_price], can_buy)
            orders.append(Order("TOMATOES", ask_price, qty))
            position += qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair:            # raw fair, NOT eff_fair
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            qty = min(od.buy_orders[bid_price], can_sell)
            orders.append(Order("TOMATOES", bid_price, -qty))
            position -= qty

        # Phase 2: Passive — skewed eff_fair with capped adaptive spread (BUG 3 fix)
        # Spread widens discretely based on volatility regime and inventory level.
        pos_frac    = abs(position) / limit
        vol_adj     = 0 if recent_vol < 1.5 else (1 if recent_vol < 2.5 else 2)
        inv_adj     = 1 if pos_frac > 0.6 else 0
        half_spread = min(
            self.TOMATO_SPREAD + vol_adj + inv_adj,
            self.TOMATO_SPREAD_CAP    # hard cap ensures quotes always fill (BUG 3 fix)
        )

        buy_price  = int(round(eff_fair)) - half_spread
        sell_price = int(round(eff_fair)) + half_spread

        remaining_buy  = limit - position
        remaining_sell = limit + position

        if remaining_buy > 0:
            orders.append(Order("TOMATOES", buy_price,   remaining_buy))
        if remaining_sell > 0:
            orders.append(Order("TOMATOES", sell_price, -remaining_sell))

        logger.print(
            f"TOM | mid={mid:.1f} mp={microprice:.1f} ema={fair:.2f} "
            f"skew={skew:.2f} eff={eff_fair:.2f} "
            f"vol={recent_vol:.2f} hs={half_spread} pos={position}"
        )

        return orders
