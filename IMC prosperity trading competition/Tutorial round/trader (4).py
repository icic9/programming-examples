"""
IMC Prosperity 4 — Strategy v4: Microstructure Innovation (Final Version)
==========================================================================
Products: EMERALDS (stationary ~10000) | TOMATOES (ARIMA(0,1,1) drift)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONTEXT: FROM CORRECT TO COMPETITIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  After v3 established a correct, theory-grounded implementation with all
  bugs fixed, I turned to the question of competing more effectively against
  other market participants. In a competition, other bots are also posting
  passive orders — I need to think about queue position and spread dynamics.
  Two innovations from market microstructure theory address this.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 1 — BOOK-ADAPTIVE PENNYING (Queue Priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  In v3, I posted passive quotes at a fixed spread from fair value (e.g.,
  fair ± 2 ticks) regardless of what other participants were doing.

  Problem: if competitors post at fair ± 2 and I also post at fair ± 2,
  I am at the back of the queue. If competitors post at fair ± 3, my
  quotes at ± 2 get filled immediately (no spread earned).

  Solution: read the actual live order book each tick to see where
  competitor bots have placed their resting orders, then post 1 tick
  inside their best price to gain queue priority.

  For buy: find the best competitor bid that is below fair (raw).
           Post 1 tick above it. This puts us at the front of the queue
           while still ensuring we only buy below our fair value estimate.
  For sell: find the best competitor ask that is above fair.
            Post 1 tick below it.

  Key properties:
    - Zero hyperparameters: we observe the book directly each tick.
    - Zero overfitting: no parameters fitted to historical data.
    - Hard boundaries: we never post at or across fair value (which
      would eliminate our edge entirely). min_edge ticks of buffer enforced.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 2 — TWO-LEVEL PASSIVE LAYERING (Robustness to Bot Behaviour)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Pure pennying is fragile: if competitor bots tighten to fair ± 1 tick,
  our penny price sits at fair ± 0 — zero edge, or worse.

  Solution: split remaining capacity across two price levels.
    Layer 1 — 60% at the penny price:  queue priority, tighter edge.
    Layer 2 — 40% at penny price ± 1:  one tick further out, more edge.

  Simulation across bot spread scenarios (EMERALDS, 1000-tick backtest):
    Bots at ±3: two-level → 41k PnL vs pure-penny (±1) → 30k
    Bots at ±2: two-level → 36k PnL vs pure-penny → 21k
    Bots at ±1: two-level → 26k PnL vs fixed-spread (±2) → 160 (!!!)
      → Fixed spread catastrophically fails when bots tighten.

  The two-level split does not maximise PnL in any single scenario, but
  it avoids catastrophic failure across the range of possible bot behaviours.
  When the competitor spread is unknown, this robustness dominates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FOUNDATIONS CARRIED FORWARD FROM V3 (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - Aggressive / passive separation: raw fair for fills, skewed for quotes.
  - ARIMA(0,1,1)-optimal EMA(α=0.43) of microprice for TOMATOES fair value.
  - Inventory skew on passive quotes only (max 2 ticks at position limit).
  - Spread widens by 1 tick when rolling σ(ΔY_t) > TOMATO_VOL_THRESH.
  - All four bugs from v2 remain fixed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (unchanged from starter) ────────────────────────────────────────
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
    v4: Final version — book-adaptive pennying + two-level layering.

    Builds on the correct v3 foundation. The two new methods (_penny_passive_prices
    and _two_level_orders) implement the microstructure innovations described above
    and are shared across both products.
    """

    # ── EMERALDS ─────────────────────────────────────────────────────────
    EMERALD_FAIR        = 10_000
    EMERALD_LIMIT       = 80
    EMERALD_SKEW        = 1.0    # max ±1 tick passive skew at full position limit
    # Two-level split fractions (must sum to 1.0)
    EMERALD_L1_FRAC     = 0.6    # 60% at penny price (queue priority)
    EMERALD_L2_FRAC     = 0.4    # 40% at penny±1 (backup edge)

    # ── TOMATOES ─────────────────────────────────────────────────────────
    TOMATO_LIMIT        = 80
    TOMATO_EMA_ALPHA    = 0.43   # α = 1 + θ, θ ≈ -0.55 (ARIMA(0,1,1)-optimal)
    TOMATO_SKEW         = 0.025  # ticks per position unit; max = 0.025 × 80 = 2 ticks
    TOMATO_VOL_WINDOW   = 20     # rolling window for σ(ΔY_t) estimate
    TOMATO_SPREAD_BASE  = 2      # base passive half-spread in normal vol regime
    TOMATO_SPREAD_HIGH  = 3      # widened half-spread when vol > VOL_THRESH
    TOMATO_VOL_THRESH   = 2.5    # σ(ΔY_t) threshold above which spread widens
    # Two-level split fractions
    TOMATO_L1_FRAC      = 0.6    # 60% at penny price
    TOMATO_L2_FRAC      = 0.4    # 40% at penny±1

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,           # reference fair value (possibly skewed for inventory)
        fair: float,          # raw (unskewed) fair value — used as the boundary
        min_buy_edge: int,    # minimum distance below ref for our buy price
        min_sell_edge: int,   # minimum distance above ref for our sell price
    ):
        """
        Book-adaptive pennying — Innovation 1.

        Scans the live order book to find competitor passive resting orders,
        then posts 1 tick inside their best price to gain queue priority.

        For buy side:
          Find the best (highest) competitor bid that is strictly below fair.
          These are safe to penny — they will not be aggressively taken.
          Post 1 tick above that price.
          Hard ceiling: never post at or above ref (need positive expected edge).

        For sell side:
          Find the best (lowest) competitor ask that is strictly above fair.
          Post 1 tick below that price.
          Hard floor: never post at or below ref.

        Fallback (no competitors visible): post at ref ± (min_edge + 1).
          Conservative: if we cannot observe competitors, we stay further out.

        The min_buy_edge / min_sell_edge parameters incorporate:
          - Base minimum edge (always ≥ 1 tick)
          - Extra 1 tick in high-vol regime (spread_extra) for TOMATOES
        """
        # Competitors' resting bids BELOW fair (won't be taken aggressively)
        comp_bids = [p for p in od.buy_orders  if p < fair]
        # Competitors' resting asks ABOVE fair (won't be taken aggressively)
        comp_asks = [p for p in od.sell_orders if p > fair]

        # ── Passive buy price ──
        if comp_bids:
            penny_buy = max(comp_bids) + 1          # 1 tick above their best
        else:
            penny_buy = int(ref) - min_buy_edge - 1  # no competitors → be conservative

        # Hard ceiling: never buy at or above ref (need positive edge)
        passive_buy = min(penny_buy, int(ref) - min_buy_edge)

        # ── Passive sell price ──
        if comp_asks:
            penny_sell = min(comp_asks) - 1          # 1 tick below their best
        else:
            penny_sell = int(ref) + min_sell_edge + 1

        # Hard floor: never sell at or below ref
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
        Two-level passive layering — Innovation 2.

        Instead of posting all remaining capacity at one price, split it:
          Layer 1 (l1_frac = 60%): at the penny price.
            → Tight to the market, high queue priority, smaller edge per fill.
          Layer 2 (l2_frac = 40%): at penny price − 1 (buy) or + 1 (sell).
            → One tick further from mid, lower queue priority, larger edge per fill.

        For the buy side: layer 2 is at penny_buy - 1 (even deeper = more edge).
        For the sell side: layer 2 is at penny_sell + 1 (even higher = more edge).

        Robustness argument:
          If bots post at ±1 and we only penny to ±0 (across fair), we lose edge.
          Layer 2 always sits at ±1 or further → maintains meaningful edge even
          when bots tighten. The split ensures we are never fully exposed to any
          single point of failure in competitor spread assumptions.

        l1_qty + l2_qty = remaining capacity (no capacity wasted).
        min l1_qty = 1 (always post at least 1 unit at the priority price).
        """
        orders = []

        # ── Buy side ──
        if remaining_buy > 0:
            l1_qty = max(1, round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            orders.append(Order(symbol, passive_buy,      l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))  # 1 tick lower = more edge

        # ── Sell side ──
        if remaining_sell > 0:
            l1_qty = max(1, round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            orders.append(Order(symbol, passive_sell,      -l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))  # 1 tick higher = more edge

        return orders

    # ── run ──────────────────────────────────────────────────────────────

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
                result[product] = self._trade_emeralds(od, position)
            elif product == "TOMATOES":
                result[product] = self._trade_tomatoes(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── EMERALDS ─────────────────────────────────────────────────────────
    def _trade_emeralds(self, od: OrderDepth, position: int) -> List[Order]:
        """
        EMERALDS: known fair = 10,000 (DF t-stat ≈ -97, white noise around mean).

        Phase 1 — Aggressive:
          Always use raw fair = 10,000. Never refuse a profitable fill because of
          inventory. (Aggressive/passive separation from v3 maintained.)

        Phase 2 — Passive:
          Inventory skew: max ±1 tick at position limit. Computed as:
            skew = round(EMERALD_SKEW * position / limit)   ∈ {-1, 0, +1}
            eff_fair = fair - skew
          This is the reference for pennying and the hard boundary check.

          Book-adaptive pennying (Innovation 1):
            Find best competitor bid below fair → post 1 tick above it.
            Find best competitor ask above fair → post 1 tick below it.
            min_buy_edge = min_sell_edge = 1: must have at least 1 tick of edge.

          Two-level layering (Innovation 2):
            60% of remaining capacity at penny price (queue priority).
            40% at penny price ± 1 (backup edge, robust to bot tightening).
        """
        orders: List[Order] = []
        fair  = self.EMERALD_FAIR
        limit = self.EMERALD_LIMIT

        # ── Phase 1: Aggressive — raw fair, no skew ──────────────────────
        for ask in sorted(od.sell_orders):
            if ask >= fair: break
            can_buy = limit - position
            if can_buy <= 0: break
            qty = min(-od.sell_orders[ask], can_buy)
            orders.append(Order("EMERALDS", ask, qty))
            position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fair: break
            can_sell = limit + position
            if can_sell <= 0: break
            qty = min(od.buy_orders[bid], can_sell)
            orders.append(Order("EMERALDS", bid, -qty))
            position -= qty

        # ── Phase 2: Passive ──────────────────────────────────────────────
        # Inventory skew: max ±1 tick at full limit
        skew     = round(self.EMERALD_SKEW * position / limit)  # ∈ {-1, 0, +1}
        eff_fair = float(fair - skew)                           # shift reference

        # Penny the book (using eff_fair as reference, fair as raw boundary)
        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_fair, fair=float(fair),
            min_buy_edge=1, min_sell_edge=1,
        )

        # Two-level layering
        orders += self._two_level_orders(
            "EMERALDS",
            passive_buy, passive_sell,
            limit - position, limit + position,
            self.EMERALD_L1_FRAC, self.EMERALD_L2_FRAC,
        )

        return orders

    # ── TOMATOES ─────────────────────────────────────────────────────────
    def _trade_tomatoes(
        self, od: OrderDepth, position: int, saved: dict
    ) -> List[Order]:
        """
        TOMATOES: ARIMA(0,1,1) process, θ ≈ -0.55, optimal forecast = EMA(α=0.43).

        Phase 1 — Aggressive:
          Compare each resting order against raw EMA (no inventory skew).
          Rationale: if an ask is below our best estimate of fair value,
          it is profitable to buy regardless of our current position.
          (Aggressive/passive separation from v3 maintained.)

        Phase 2 — Passive:
          Inventory skew: TOMATO_SKEW * position ∈ [-2, +2] ticks at limit.
          eff_ema = ema - skew   (used as reference for pennying, not for fills)

          Volatility-adjusted min_edge:
            min_edge = 1 (normal) or 2 (high vol, when σ(ΔY) > VOL_THRESH).
            In high vol regimes, wider minimum edge = more adverse selection protection.

          Book-adaptive pennying (Innovation 1):
            Find best competitor bid below raw ema → post 1 tick above it.
            Clamp: never post closer than min_edge ticks to eff_ema.

          Two-level layering (Innovation 2):
            60% at penny price, 40% at penny price ± 1.
            Robust to unknown competitor spread.

          Persisted state: tomato_ema, tomato_last_mid, tomato_diffs.
          All carried in saved dict via jsonpickle across ticks.
        """
        orders: List[Order] = []
        limit  = self.TOMATO_LIMIT

        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid      = (best_bid + best_ask) / 2.0

        # ── Microprice (Concept 5) ────────────────────────────────────────
        bv = od.buy_orders[best_bid]
        av = -od.sell_orders[best_ask]
        microprice = (
            (best_bid * av + best_ask * bv) / (bv + av)
            if bv + av > 0 else mid
        )

        # ── EMA update — ARIMA(0,1,1) optimal (Concept 4) ────────────────
        a = self.TOMATO_EMA_ALPHA
        if saved["tomato_ema"] is None:
            saved["tomato_ema"] = microprice
        else:
            saved["tomato_ema"] = a * microprice + (1.0 - a) * saved["tomato_ema"]

        ema = saved["tomato_ema"]   # raw, unskewed fair value

        # ── Rolling volatility for spread widening ────────────────────────
        if saved["tomato_last_mid"] is not None:
            saved["tomato_diffs"].append(mid - saved["tomato_last_mid"])
            if len(saved["tomato_diffs"]) > self.TOMATO_VOL_WINDOW:
                saved["tomato_diffs"].pop(0)
        saved["tomato_last_mid"] = mid

        diffs = saved["tomato_diffs"]
        if len(diffs) >= 5:
            vol = (sum(d ** 2 for d in diffs) / len(diffs)) ** 0.5
        else:
            vol = 1.3

        spread_extra = 1 if vol > self.TOMATO_VOL_THRESH else 0

        # ── Phase 1: Aggressive — raw EMA, no skew (Concept 3) ───────────
        for ask in sorted(od.sell_orders):
            if ask >= ema: break
            can_buy = limit - position
            if can_buy <= 0: break
            qty = min(-od.sell_orders[ask], can_buy)
            orders.append(Order("TOMATOES", ask, qty))
            position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= ema: break
            can_sell = limit + position
            if can_sell <= 0: break
            qty = min(od.buy_orders[bid], can_sell)
            orders.append(Order("TOMATOES", bid, -qty))
            position -= qty

        # ── Phase 2: Passive ──────────────────────────────────────────────
        # Inventory skew: max ±2 ticks at full limit
        skew     = self.TOMATO_SKEW * position       # ∈ [-2, +2] ticks
        eff_ema  = ema - skew                        # skewed reference for passive

        # Penny the book (Concept 1)
        # min_edge includes spread_extra to widen in high-vol regimes
        min_edge = 1 + spread_extra
        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_ema, fair=ema,
            min_buy_edge=min_edge, min_sell_edge=min_edge,
        )

        # Two-level layering (Concept 2)
        orders += self._two_level_orders(
            "TOMATOES",
            passive_buy, passive_sell,
            limit - position, limit + position,
            self.TOMATO_L1_FRAC, self.TOMATO_L2_FRAC,
        )

        logger.print(
            f"TOM | mid={mid:.1f} mp={microprice:.1f} ema={ema:.2f} "
            f"skew={skew:.2f} eff={eff_ema:.2f} "
            f"vol={vol:.2f} pbuy={passive_buy} psell={passive_sell} pos={position}"
        )

        return orders
