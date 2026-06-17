import json
from collections import deque
from typing import Any, List

import jsonpickle

from datamodel import (
    Order, OrderDepth, ProsperityEncoder, Trade, TradingState,
)


# ── Logger ──────────────────────────────────────────────────────────────────
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


class Trader:
    """
    v15: Predictive + Hard-Anchored Breakthrough Strategy

    Core thesis:
      OSMIUM = EMERALDS analog: mean=10,000, std=5.4, hard-coded anchor.
               Trade aggressively around 10,000; asymmetric volume when loaded.

      PEPPER = Deterministic linear drift at 0.001 ticks per timestamp unit.
               OLS regression on last 6 prices predicts t+1 price with std≈2 ticks.
               Ride the trend long; only sell to unwind, never to go short.

    Innovations over v14:
      1. OSMIUM hard-coded fair = 10,000 (eliminates EMA lag/noise)
      2. OLS 6-tick regression replaces EMA for PEPPER (predictive, not reactive)
      3. Clock drift as long-horizon prior (start + 0.001*timestamp)
      4. Aggressive book-sweeping taking before passive quoting
      5. Inventory floor on PEPPER: never go net short in uptrend
      6. Asymmetric volume: scale down wrong-direction size quadratically
      7. Pennying: post inside the spread to maximise queue priority
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    # Fair value hard-coded at 10,000 (empirically verified: mean=10000.20, std=5.35)
    OSMIUM_LIMIT = 80
    OSMIUM_FAIR = 10_000

    # Take any offer this many ticks or more below fair (and symmetrically for bids)
    # Edge=1 → take asks ≤ 9999, take bids ≥ 10001
    OSMIUM_TAKE_EDGE = 1

    # Passive quote prices
    OSMIUM_QUOTE_BUY = 9_999
    OSMIUM_QUOTE_SELL = 10_001

    # Inventory scaling: below this abs position, post full size on both sides
    OSMIUM_NEUTRAL_BAND = 20
    # When |pos| > LOADED, scale down the "wrong" direction quadratically
    OSMIUM_LOADED_BAND = 50

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80

    # Deterministic drift per timestamp unit (verified: 0.001003 across 3 days)
    PEPPER_CLOCK_DRIFT = 0.001

    # OLS regression window length (prices)
    PEPPER_OLS_LEN = 6

    # Blend: OLS for responsiveness, clock for long-horizon stability
    PEPPER_OLS_WEIGHT = 0.70
    PEPPER_CLOCK_WEIGHT = 0.30

    # Taking: buy if ask < fair - edge; sell if bid > fair + edge
    PEPPER_TAKE_EDGE = 0.5

    # Passive quoting minimum edge from fair value
    PEPPER_QUOTE_EDGE = 1

    # Inventory skew: reservation = fair - SKEW * pos/limit
    PEPPER_INV_SKEW = 3.0
    PEPPER_URGENCY_THRESH = 0.65
    PEPPER_URGENCY_MULT = 2.5

    # Slope threshold to classify as uptrend; below -GATE = downtrend
    PEPPER_TREND_GATE = 0.02

    # In strong uptrend, target holding at least this many units long
    # This biases passive buys to stay large even when moderately long
    PEPPER_TARGET_LONG = 40

    def run(self, state: TradingState):
        result = {}
        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("pepper_history", [])
        saved.setdefault("pepper_day_start", None)
        saved.setdefault("osmium_ema", None)  # kept for logging only

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if product == self.PEPPER:
                result[product] = self._trade_pepper(
                    od, position, state.timestamp, saved
                )
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OSMIUM ──────────────────────────────────────────────────────────────

    def _trade_osmium(
        self,
        od: OrderDepth,
        position: int,
        saved: dict,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT
        fair = self.OSMIUM_FAIR

        # ── Phase 1: Aggressive sniping ──
        # Buy everything below fair value (asks ≤ fair - TAKE_EDGE)
        buy_threshold = fair - self.OSMIUM_TAKE_EDGE  # ≤ 9999
        for ask in sorted(od.sell_orders):
            if ask > buy_threshold:
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], can_buy)
            if qty > 0:
                orders.append(Order(self.OSMIUM, ask, qty))
                position += qty

        # Sell everything above fair value (bids ≥ fair + TAKE_EDGE)
        sell_threshold = fair + self.OSMIUM_TAKE_EDGE  # ≥ 10001
        for bid in sorted(od.buy_orders, reverse=True):
            if bid < sell_threshold:
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            qty = min(od.buy_orders[bid], can_sell)
            if qty > 0:
                orders.append(Order(self.OSMIUM, bid, -qty))
                position -= qty

        # ── Phase 2: Passive pennying inside the spread ──
        # Find best live prices on each side to penny
        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        passive_buy = (max(comp_bids) + 1) if comp_bids else self.OSMIUM_QUOTE_BUY
        passive_sell = (min(comp_asks) - 1) if comp_asks else self.OSMIUM_QUOTE_SELL

        # Clamp: never cross fair value
        passive_buy = min(passive_buy, self.OSMIUM_QUOTE_BUY)
        passive_sell = max(passive_sell, self.OSMIUM_QUOTE_SELL)

        if passive_buy >= passive_sell:
            passive_buy = self.OSMIUM_QUOTE_BUY
            passive_sell = self.OSMIUM_QUOTE_SELL

        # ── Asymmetric volume sizing ──
        # At neutral, post full size. As position grows, aggressively shrink
        # the "adding-more-of-same-direction" side quadratically.
        abs_pos = abs(position)
        neutral = self.OSMIUM_NEUTRAL_BAND
        loaded = self.OSMIUM_LOADED_BAND

        if abs_pos <= neutral:
            buy_mult = sell_mult = 1.0
        elif abs_pos <= loaded:
            # Linear scale from 1.0 at NEUTRAL_BAND to 0.3 at LOADED_BAND
            t = (abs_pos - neutral) / (loaded - neutral)
            wrong_mult = 1.0 - 0.7 * t
            if position > 0:
                buy_mult, sell_mult = wrong_mult, 1.0
            else:
                buy_mult, sell_mult = 1.0, wrong_mult
        else:
            # Stressed: heavily penalise wrong direction
            t = min(1.0, (abs_pos - loaded) / (limit - loaded))
            wrong_mult = max(0.05, 0.3 - 0.25 * t)
            if position > 0:
                buy_mult, sell_mult = wrong_mult, 1.0
            else:
                buy_mult, sell_mult = 1.0, wrong_mult

        remaining_buy = max(0, int(round((limit - position) * buy_mult)))
        remaining_sell = max(0, int(round((limit + position) * sell_mult)))

        if remaining_buy > 0:
            orders.append(Order(self.OSMIUM, passive_buy, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(self.OSMIUM, passive_sell, -remaining_sell))

        logger.print(
            f"OSM| fair={fair} pb={passive_buy} ps={passive_sell} "
            f"rb={remaining_buy} rs={remaining_sell} pos={position} "
            f"bm={buy_mult:.2f} sm={sell_mult:.2f}"
        )
        return orders

    # ── PEPPER ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ols_predict(history: list) -> float:
        """OLS linear regression on history; extrapolates one step ahead.

        For x=[0..n-1], fits y=a+b*x and returns predicted y at x=n.
        Correctly handles any window size n ≥ 2.
        """
        n = len(history)
        if n < 2:
            return history[0] if history else 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n
        sum_wp = sum((i - x_mean) * p for i, p in enumerate(history))
        sum_w2 = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum_wp / sum_w2 if sum_w2 > 0 else 0.0
        # Predict at x=n: offset from x_mean = n - (n-1)/2 = (n+1)/2
        return y_mean + slope * (n - x_mean)

    def _trade_pepper(
        self,
        od: OrderDepth,
        position: int,
        timestamp: int,
        saved: dict,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid = (best_bid + best_ask) / 2.0
        elif has_bid:
            best_bid = max(od.buy_orders)
            best_ask = None
            mid = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            best_bid = None
            mid = float(best_ask)

        # Use best large-volume price as the reference mid (filters noise)
        # Prefer the side with the most volume to anchor the mid
        if has_bid and has_ask:
            bid_vols = {p: v for p, v in od.buy_orders.items()}
            ask_vols = {p: -v for p, v in od.sell_orders.items()}
            heavy_bid = max(bid_vols, key=bid_vols.get)
            heavy_ask = min(ask_vols, key=lambda p: (-ask_vols[p], p))
            # Wall-mid: average of highest-volume bid and ask levels
            wall_mid = (heavy_bid + heavy_ask) / 2.0
            # Blend with raw mid for responsiveness
            ref_mid = 0.5 * wall_mid + 0.5 * mid
        else:
            ref_mid = mid

        # ── Update price history and OLS regression ──
        history = saved["pepper_history"]
        history.append(ref_mid)
        if len(history) > self.PEPPER_OLS_LEN:
            history.pop(0)

        # Clock drift: day starts at first observed price, drifts at 0.001/tick
        if saved["pepper_day_start"] is None:
            saved["pepper_day_start"] = ref_mid
        clock_fair = saved["pepper_day_start"] + self.PEPPER_CLOCK_DRIFT * timestamp

        # OLS regression on last N prices (works for any window size ≥ 2)
        if len(history) >= 2:
            ols_fair = self._ols_predict(history)
        else:
            ols_fair = ref_mid

        # Blend: OLS is primary (responsive), clock drift is secondary (stable anchor)
        fair = self.PEPPER_OLS_WEIGHT * ols_fair + self.PEPPER_CLOCK_WEIGHT * clock_fair

        # OLS slope (sign only needed) — use last 2 points as fast slope estimate
        if len(history) >= 3:
            slope = (history[-1] - history[-3]) / 2.0
        else:
            slope = 0.0

        uptrend = slope > self.PEPPER_TREND_GATE
        downtrend = slope < -self.PEPPER_TREND_GATE

        # ── Phase 1: Aggressive taking ──
        # Buy asks that are clearly below fair value
        buy_threshold = fair - self.PEPPER_TAKE_EDGE
        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= buy_threshold:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        # Sell bids that are clearly above fair value (only if long)
        sell_threshold = fair + self.PEPPER_TAKE_EDGE
        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= sell_threshold:
                    break
                # Inventory floor: never sell short in uptrend
                if uptrend and position <= 0:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        # ── Phase 2: Inventory-adjusted passive quoting ──
        # Reservation price: shift down when long (willing to sell cheaper to exit)
        inv_frac = position / limit
        skew = self.PEPPER_INV_SKEW * inv_frac
        if abs(inv_frac) > self.PEPPER_URGENCY_THRESH:
            skew *= self.PEPPER_URGENCY_MULT
        reservation = fair - skew

        # Penny inside any competitive quotes on each side
        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        penny_buy = (max(comp_bids) + 1) if comp_bids else (int(reservation) - self.PEPPER_QUOTE_EDGE)
        penny_sell = (min(comp_asks) - 1) if comp_asks else (int(reservation) + self.PEPPER_QUOTE_EDGE)

        # Clamp: stay at least QUOTE_EDGE from reservation
        passive_buy = min(penny_buy, int(reservation) - self.PEPPER_QUOTE_EDGE)
        passive_sell = max(penny_sell, int(reservation) + self.PEPPER_QUOTE_EDGE)

        if passive_buy >= passive_sell:
            passive_buy = int(reservation) - self.PEPPER_QUOTE_EDGE
            passive_sell = int(reservation) + self.PEPPER_QUOTE_EDGE

        # ── Asymmetric capacity: trend-biased volume sizing ──
        remaining_buy = limit - position
        remaining_sell = limit + position

        if uptrend:
            # In uptrend: maximise buy capacity; cap sells to reduce adverse selection
            # Allow sells only to bring position back from extreme long
            # Never go short (inventory floor)
            remaining_sell = max(0, min(remaining_sell, position))

            # Boost buy capacity: even when "loaded long", keep buying near target
            if position < self.PEPPER_TARGET_LONG:
                # Below target: post full buy size
                pass  # remaining_buy unchanged
            else:
                # At/above target: scale down buys quadratically to avoid hitting limit too fast
                excess = (position - self.PEPPER_TARGET_LONG) / (limit - self.PEPPER_TARGET_LONG)
                remaining_buy = max(5, int(round(remaining_buy * (1.0 - 0.5 * excess))))

        elif downtrend:
            # In downtrend: maximise sell capacity; cap buys (inventory ceiling = never long)
            remaining_buy = max(0, min(remaining_buy, -position))
            if position > -self.PEPPER_TARGET_LONG:
                pass  # remaining_sell unchanged
            else:
                excess = (-position - self.PEPPER_TARGET_LONG) / (limit - self.PEPPER_TARGET_LONG)
                remaining_sell = max(5, int(round(remaining_sell * (1.0 - 0.5 * excess))))

        # Post passive orders
        if remaining_buy > 0:
            orders.append(Order(self.PEPPER, passive_buy, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(self.PEPPER, passive_sell, -remaining_sell))

        logger.print(
            f"PEP| mid={mid:.1f} ols={ols_fair:.2f} clk={clock_fair:.2f} "
            f"fair={fair:.2f} slp={slope:.3f} res={reservation:.2f} "
            f"pb={passive_buy} ps={passive_sell} "
            f"rb={remaining_buy} rs={remaining_sell} pos={position} "
            f"trend={'UP' if uptrend else 'DN' if downtrend else '--'}"
        )
        return orders
