"""
IMC Prosperity 4 — Strategy v6: Trend-Carry Dual-EMA Signal
============================================================
Products: EMERALDS (stationary ~10000) | TOMATOES (ARIMA(0,1,1) drift)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RESEARCH PROCESS & DISCARDED CHANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This version is the product of deep backtesting research and a series
  of discarded candidates. What was tried and rejected:

  ✗  At-fair EMERALDS aggressive fills:
       Buying at ask=10000 (=fair) + passive sell at 10007 = +7 ticks.
       The capacity it displaces would complete a passive round-trip
       (buy at 9993, sell at 10007) = +14 ticks. Net: -7/unit.
       Backtester confirmed: -245 EMERALDS PnL over 2 training days.
       DISCARDED.

  ✗  Full-book microprice (total volume weights):
       Introduces noise (>1 tick on 3.6% of ticks) vs level-1 microprice.
       Tested both days: causes -160 total PnL baseline regression due to
       occasional EMA misdirection. DISCARDED (reverted to level-1).

  ✗  1-tick EMA velocity for momentum:
       ema_velocity = EMA_t − EMA_{t−1} is dominated by noise. Improved
       day -2 (+138 TOMATOES) but hurt day -1 (-131). Net: roughly neutral.
       DISCARDED in favour of dual-EMA signal.

  ✓  Dual-EMA momentum signal: RETAINED (see below).
  ✓  Urgency position skew: RETAINED (see below).
  ✓  Exponential moving variance: RETAINED (cleaner than windowed list).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 3 — TOMATOES: DUAL-EMA TREND SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Problem: in trending markets the single EMA(α=0.43) lags the true
  level, causing the passive loop to accumulate long positions precisely
  when price is falling. The 1-tick velocity fix was too noisy.

  Solution: run TWO EMAs in parallel:
    ema_fast (α=0.43): already used for fair-value estimation (unchanged)
    ema_slow (α=0.10): slower, sustained-trend tracker

  Trend signal = ema_fast − ema_slow.

  Why this is robust:
    - In a SUSTAINED downtrend: ema_fast pulls 5–15 ticks below ema_slow
      after 20–30 ticks. Signal is large, persistent, and confirmed.
    - In a FLAT / NOISY market: both EMAs orbit the same level. Signal
      stays near zero → essentially no momentum bias. Unlike 1-tick
      velocity, which fires on every tick even without real trend.
    - In the DAY -2 downtrend (5000→4960): signal grows to −10 over 40
      ticks. At MOMENTUM_BIAS=5, biased_pos = position + 50. This pushes
      us into the urgency zone for most real positions, creating strong
      and CORRECT sell pressure.
    - On DAY -1 (flat/volatile): signal stays near 0. Momentum bias is
      minimal. PnL is essentially identical to v4.

  Application:
    biased_pos = position − MOMENTUM_BIAS × trend_signal

    trend_signal < 0 (downtrend): biased_pos > position → act more long
      → larger skew → lower eff_ema → passive sell lower (fills faster)
      and passive buy lower (less buy-aggression in falling market). ✓

    trend_signal > 0 (uptrend): act more short → symmetric buy bias. ✓

  MOMENTUM_BIAS = 5. At max observed signal ≈ −10, biased_pos + 50 →
  activates urgency zone for typical positions in a strong trend.

  Note on backtester limitations:
    The TOMATOES competitor spread is 13–14 ticks wide. Because our
    passive quotes are always dominated by the book-adaptive penny
    price (comp_bid+1 / comp_ask-1), the skew clamp only activates
    when UNUSUAL HIGH BIDS or LOW ASKS appear close to fair value
    (occurs ~2–3 times per 10,000-tick day). In the 2-day training
    sample, the momentum signal provides a marginal but positive effect.
    With more competition data covering varied trend regimes, the
    dual-EMA signal's advantage over 1-tick velocity should be clear.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 4 — TOMATOES: URGENCY POSITION SKEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  When |biased_pos| > 70% of position limit (56 units), multiply the
  base inventory skew by 2×. Additive with the momentum signal:
  in a strong downtrend with real long position, both effects compound.
  Prevents hitting the position limit without warning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 5 — TOMATOES: EXPONENTIAL MOVING VARIANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Replaces the 20-element rolling list with exponential moving variance:
    emv_t = α_v × diff² + (1−α_v) × emv_{t−1},  α_v = 0.15

  Effective memory ≈ 7 ticks (vs 20-tick window). Detects vol regime
  changes faster. Eliminates the list storage. Note: the spread threshold
  (2.5 ticks) is rarely crossed in practice on tutorial round data
  (~0 times per day), so the primary benefit is code cleanliness.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 6 — TOMATOES: TREND-CARRY FAIR VALUE ADJUSTMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Problem: in v5, the dual-EMA signal only adjusted the passive skew via
  biased_pos. The aggressive threshold and the passive quote anchor were
  still the raw EMA — which lags the true price in a trend. The signal was
  saying "price is falling" but we were still buying aggressively up to
  a stale, too-high fair value.

  Solution: carry a small fraction of the trend signal into fair value:
    forecast_ema = ema + TOMATO_TREND_CARRY × trend_signal
    TOMATO_TREND_CARRY = 0.25 (25% of signal)

  forecast_ema is used as the threshold for aggressive fills AND as the
  anchor for passive quotes (before inventory skew is applied):
    Downtrend (signal = −10): forecast_ema = ema − 2.5 ticks.
      We stop buying at asks that are slightly above the momentum-adjusted
      fair value. Passive quotes also shift down with the trend.
    Uptrend: symmetric — forecast_ema rises → more willing to buy near
      the rising price, less willing to sell into a rising market.

  Also: TOMATO_EMA_SLOW_ALPHA reduced from 0.10 (v5) to 0.06 (v6).
    At 0.10, the slow EMA reacted to intra-day noise on the flat training
    day. At 0.06 (effective memory ≈ 17 ticks), it tracks only sustained
    multi-tick trends, suppressing spurious signals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INNOVATION 7 — TOMATOES: AGAINST-TREND PASSIVE CAPACITY THROTTLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Problem: even with carry-adjusted fair value and urgency skew, we still
  advertise full passive buy capacity in a downtrend. When those passive
  bids get hit, we accumulate long inventory against the trend direction —
  exactly the adverse outcome we want to avoid.

  Solution: when |trend_signal| exceeds TOMATO_TREND_GATE (1.5 ticks),
  cap passive size in the against-trend direction to 65%:
    Downtrend (signal < −1.5): remaining_buy × 0.65
    Uptrend   (signal > +1.5): remaining_sell × 0.65

  The 35% reduction meaningfully reduces adverse inventory accumulation
  while keeping 65% of spread income available in case the trend reverses.
  A complete shutdown would be too aggressive given that the competitor
  spread is 13–14 ticks and the trend signal magnitude varies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FOUNDATIONS CARRIED FORWARD FROM V5 (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - EMERALDS: identical to v4/v5 (no at-fair fills)
  - Book-adaptive pennying (Innovation 1)
  - Two-level passive layering (Innovation 2)
  - ARIMA(0,1,1)-optimal EMA(α=0.43) as TOMATOES fair value
  - Dual-EMA momentum signal and urgency skew (Innovations 3 & 4)
  - Exponential moving variance (Innovation 5)
  - Aggressive / passive separation for both products
  - Level-1 microprice for EMA input (cleaner than full-book)
  - All v3 bug fixes remain
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
    v6: Dual-EMA TOMATOES strategy with trend-carry fair value.

    EMERALDS is unchanged from v4/v5. TOMATOES uses a slower EMA
    (alpha=0.06) for a less noisy trend signal, applies a small fraction
    of that signal to fair value, then throttles passive size against
    sustained trend direction.
    """

    # ── EMERALDS (unchanged from v4) ─────────────────────────────────────
    EMERALD_FAIR        = 10_000
    EMERALD_LIMIT       = 80
    EMERALD_SKEW        = 1.0
    EMERALD_L1_FRAC     = 0.6
    EMERALD_L2_FRAC     = 0.4

    # ── TOMATOES ─────────────────────────────────────────────────────────
    TOMATO_LIMIT          = 80
    TOMATO_EMA_ALPHA      = 0.43   # ARIMA(0,1,1)-optimal; fast EMA = fair value
    TOMATO_EMA_SLOW_ALPHA = 0.06   # v6: slower trend EMA; less day -1 noise
    TOMATO_SKEW           = 0.025  # ticks per position unit; max 2 at limit
    TOMATO_SPREAD_BASE    = 2
    TOMATO_SPREAD_HIGH    = 3
    TOMATO_VOL_THRESH     = 2.5
    TOMATO_L1_FRAC        = 0.6
    TOMATO_L2_FRAC        = 0.4
    # ── v5/v6 additions ───────────────────────────────────────────────────
    TOMATO_EMV_ALPHA      = 0.15   # Innovation 5: EMV decay (≈7-tick memory)
    TOMATO_MOMENTUM_BIAS  = 5      # Innovation 3: position units per signal tick
    TOMATO_URGENCY_THRESH = 0.70   # Innovation 4: |biased_pos| / limit threshold
    TOMATO_URGENCY_MULT   = 2.0    # Innovation 4: skew multiplier in urgency zone
    TOMATO_TREND_CARRY      = 0.25   # Innovation 6: fraction of trend_signal added to fair value
    TOMATO_TREND_GATE       = 1.5    # Innovation 7: |trend_signal| threshold for capacity throttle
    TOMATO_AGAINST_TREND_CAP = 0.65  # Innovation 7: cap passive against-trend size to 65%

    # ── shared helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,           # reference fair value (possibly skewed for inventory)
        fair: float,          # raw (unskewed) fair value — used as boundary
        min_buy_edge: int,
        min_sell_edge: int,
    ):
        """
        Book-adaptive pennying — Innovation 1 (unchanged from v4).

        Scan competitor resting bids below raw fair → post 1 tick above.
        Scan competitor resting asks above raw fair → post 1 tick below.
        Hard clamps prevent crossing the skewed reference.
        """
        comp_bids = [p for p in od.buy_orders  if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        if comp_bids:
            penny_buy = max(comp_bids) + 1
        else:
            penny_buy = int(ref) - min_buy_edge - 1

        passive_buy = min(penny_buy, int(ref) - min_buy_edge)

        if comp_asks:
            penny_sell = min(comp_asks) - 1
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
        Two-level passive layering — Innovation 2 (unchanged from v4).

        L1 (60%): penny price — queue priority, tighter edge per fill.
        L2 (40%): penny ± 1 tick — one tick further, more edge per fill.
        """
        orders = []

        if remaining_buy > 0:
            l1_qty = max(1, round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            orders.append(Order(symbol, passive_buy,      l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))

        if remaining_sell > 0:
            l1_qty = max(1, round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            orders.append(Order(symbol, passive_sell,      -l1_qty))
            if l2_qty > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))

        return orders

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result = {}

        if state.traderData:
            saved = jsonpickle.decode(state.traderData)
        else:
            saved = {
                "tomato_ema":       None,   # fast EMA = primary fair value
                "tomato_ema_slow":  None,   # v5: slow EMA for trend signal
                "tomato_last_mid":  None,
                "tomato_emv":       None,   # v5: exponential moving variance
            }

        # Backwards-compatibility: add v5 keys if loading an older state blob.
        saved.setdefault("tomato_ema_slow", None)
        saved.setdefault("tomato_emv", None)
        # Remove legacy keys from earlier versions.
        saved.pop("tomato_diffs", None)
        saved.pop("tomato_ema_prev", None)

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

    # ── EMERALDS (unchanged from v4/v5) ──────────────────────────────────────
    def _trade_emeralds(self, od: OrderDepth, position: int) -> List[Order]:
        """
        EMERALDS v6: identical to v4/v5. Known fair = 10,000 (DF t-stat ≈ -97, white noise).

        At-fair aggressive fills were tested and discarded (see module docstring):
          buying at ask=10,000 earns +7 ticks but displaces a passive round-trip
          worth +14 ticks → net -7 ticks per unit of displaced capacity.

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

        # ── Phase 1: Aggressive ───────────────────────────────────────────
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

        # ── Phase 2: Passive ─────────────────────────────────────────────
        skew     = round(self.EMERALD_SKEW * position / limit)
        eff_fair = float(fair - skew)

        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_fair, fair=float(fair),
            min_buy_edge=1, min_sell_edge=1,
        )

        orders += self._two_level_orders(
            "EMERALDS",
            passive_buy, passive_sell,
            limit - position, limit + position,
            self.EMERALD_L1_FRAC, self.EMERALD_L2_FRAC,
        )

        return orders

    # ── TOMATOES ─────────────────────────────────────────────────────────────
    def _trade_tomatoes(
        self, od: OrderDepth, position: int, saved: dict
    ) -> List[Order]:
        """
        TOMATOES v6: dual-EMA momentum signal + urgency skew + EMV,
        extended with trend-carry fair value and against-trend throttling.

        Fair-value base (unchanged from v4):
          Level-1 microprice → EMA(α=0.43). Optimal for ARIMA(0,1,1).
          Full-book microprice was tested and discarded (introduces noise).

        Innovation 5 — Exponential moving variance:
          emv = 0.15 × diff² + 0.85 × emv_prev. Replaces rolling list.

        Innovation 3 — Dual-EMA trend signal (slow alpha: 0.06 in v6, 0.10 in v5):
          trend_signal = ema_fast − ema_slow.
          biased_pos = position − MOMENTUM_BIAS × trend_signal.
          Clamped to [−limit, +limit].

        Innovation 4 — Urgency zone:
          |biased_pos| > 70% of limit → multiply skew by 2×.

        Innovation 6 — Trend-carry fair value (new in v6):
          forecast_ema = ema + 0.25 × trend_signal.
          Used as the threshold for aggressive fills AND as the passive anchor.
          Downtrend: forecast_ema < ema → stop buying at the lagged EMA price.
          Uptrend: forecast_ema > ema → more willing to buy near rising price.

        Phase 1 — Aggressive:
          Compare each resting order against forecast_ema (trend-adjusted).
          Any ask below forecast_ema is profitable to buy; any bid above is
          profitable to sell — regardless of our current inventory.

        Phase 2 — Passive:
          eff_ema = forecast_ema − skew(biased_pos).  Inventory skew from
          biased_pos (max ±4 ticks in urgency zone). Pennying and layering
          as in v4.

        Innovation 7 — Against-trend passive capacity throttling (new in v6):
          When |trend_signal| > TOMATO_TREND_GATE (1.5 ticks), cap passive
          size in the against-trend direction at 65% of normal.
          Reduces adverse inventory accumulation without abandoning the spread.
        """
        orders: List[Order] = []
        limit  = self.TOMATO_LIMIT

        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid      = (best_bid + best_ask) / 2.0

        # ── Level-1 microprice (v4-identical, cleaner than full-book) ────
        bv = od.buy_orders[best_bid]
        av = -od.sell_orders[best_ask]
        microprice = (best_bid * av + best_ask * bv) / (bv + av) if bv + av > 0 else mid

        # ── Fast EMA update — ARIMA(0,1,1) optimal (unchanged from v4) ───
        a_fast = self.TOMATO_EMA_ALPHA
        if saved["tomato_ema"] is None:
            saved["tomato_ema"] = microprice
        else:
            saved["tomato_ema"] = a_fast * microprice + (1.0 - a_fast) * saved["tomato_ema"]

        ema = saved["tomato_ema"]   # primary fair value estimate

        # ── Innovation 3: Slow EMA update ────────────────────────────────
        a_slow = self.TOMATO_EMA_SLOW_ALPHA
        if saved["tomato_ema_slow"] is None:
            saved["tomato_ema_slow"] = microprice
        else:
            saved["tomato_ema_slow"] = a_slow * microprice + (1.0 - a_slow) * saved["tomato_ema_slow"]

        ema_slow = saved["tomato_ema_slow"]

        # Trend signal: fast − slow.
        # Positive (fast above slow): uptrend → act more short (buy pressure).
        # Negative (fast below slow): downtrend → act more long (sell pressure).
        trend_signal = ema - ema_slow

        # Innovation 6: trend-carry — nudge fair value in the trend direction ──
        # Prevents aggressive fills at a stale lagged EMA when the trend is clear.
        forecast_ema = ema + self.TOMATO_TREND_CARRY * trend_signal

        # ── Innovation 5: Exponential moving variance ─────────────────────
        if saved["tomato_last_mid"] is not None:
            diff_sq  = (mid - saved["tomato_last_mid"]) ** 2
            emv_prev = saved["tomato_emv"]
            if emv_prev is None:
                saved["tomato_emv"] = diff_sq
            else:
                a_v = self.TOMATO_EMV_ALPHA
                saved["tomato_emv"] = a_v * diff_sq + (1.0 - a_v) * emv_prev

        saved["tomato_last_mid"] = mid

        emv = saved["tomato_emv"]
        vol  = emv ** 0.5 if emv is not None else 1.3

        spread_extra = 1 if vol > self.TOMATO_VOL_THRESH else 0

        # ── Phase 1: Aggressive — trend-carry fair value ─────────────────
        for ask in sorted(od.sell_orders):
            if ask >= forecast_ema: break
            can_buy = limit - position
            if can_buy <= 0: break
            qty = min(-od.sell_orders[ask], can_buy)
            orders.append(Order("TOMATOES", ask, qty))
            position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= forecast_ema: break
            can_sell = limit + position
            if can_sell <= 0: break
            qty = min(od.buy_orders[bid], can_sell)
            orders.append(Order("TOMATOES", bid, -qty))
            position -= qty

        # ── Phase 2: Passive — trend-biased, urgency-scaled ──────────────

        # Innovation 3: momentum-biased effective position.
        # In downtrend (signal < 0): biased_pos > position → more sell pressure.
        # In uptrend (signal > 0): biased_pos < position → more buy pressure.
        biased_pos = position - self.TOMATO_MOMENTUM_BIAS * trend_signal
        biased_pos = max(-limit, min(limit, biased_pos))

        # Base inventory skew from biased position
        skew = self.TOMATO_SKEW * biased_pos      # ∈ [−2, +2] in normal zone

        # Innovation 4: urgency multiplier
        if abs(biased_pos) > self.TOMATO_URGENCY_THRESH * limit:
            skew *= self.TOMATO_URGENCY_MULT      # ∈ [−4, +4] in urgency zone

        eff_ema = forecast_ema - skew

        # Book-adaptive pennying
        min_edge = 1 + spread_extra
        passive_buy, passive_sell = self._penny_passive_prices(
            od, ref=eff_ema, fair=forecast_ema,
            min_buy_edge=min_edge, min_sell_edge=min_edge,
        )

        # Innovation 7: against-trend passive capacity throttling ─────────────
        # In a strong trend, reduce passive size in the against-trend direction.
        # Downtrend: fewer passive buys → less adverse inventory accumulation.
        # Uptrend: fewer passive sells → symmetric protection.
        remaining_buy = limit - position
        remaining_sell = limit + position
        if trend_signal < -self.TOMATO_TREND_GATE:
            remaining_buy = int(round(remaining_buy * self.TOMATO_AGAINST_TREND_CAP))
        elif trend_signal > self.TOMATO_TREND_GATE:
            remaining_sell = int(round(remaining_sell * self.TOMATO_AGAINST_TREND_CAP))

        # Two-level layering
        orders += self._two_level_orders(
            "TOMATOES",
            passive_buy, passive_sell,
            remaining_buy, remaining_sell,
            self.TOMATO_L1_FRAC, self.TOMATO_L2_FRAC,
        )

        logger.print(
            f"TOM | mid={mid:.1f} mp={microprice:.1f} ema={ema:.2f} "
            f"slow={ema_slow:.2f} sig={trend_signal:.2f} fcst={forecast_ema:.2f} "
            f"bpos={biased_pos:.1f} "
            f"skew={skew:.2f} eff={eff_ema:.2f} "
            f"vol={vol:.2f} pbuy={passive_buy} psell={passive_sell} pos={position}"
        )

        return orders
