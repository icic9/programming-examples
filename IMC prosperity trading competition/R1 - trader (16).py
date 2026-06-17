import json
from typing import Any, List, Tuple

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
    v16: Predictive + Hard-Anchored Breakthrough Strategy (OLS fixed)

    Core thesis:
      OSMIUM ≡ EMERALDS: hard-coded fair value = 10,000. Trade tight around it.
      PEPPER: deterministic +0.001 ticks/timestamp uptrend.
              OLS(6) regression for fair value. Aggressive trend riding:
              always try to hold max long, buy every available ask in uptrend.

    Key fixes over v15:
      - OLS regression weights correct for any window size n (not just n=6)
      - OLS slope used for trend detection (stable vs noisy 2-point difference)
      - Uptrend mode: take ALL asks aggressively (no fair-value threshold),
        maximising position accumulation in a strong directional market.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    # Hard-coded anchor: empirically mean=10,000.20, std=5.35 across all 3 days
    OSMIUM_FAIR = 10_000

    # Take: buy all asks ≤ FAIR-1, sell all bids ≥ FAIR+1
    OSMIUM_TAKE_EDGE = 1

    # Passive: post inside any existing quotes, but always at worst 9999/10001
    OSMIUM_QUOTE_BUY = 9_999
    OSMIUM_QUOTE_SELL = 10_001

    # Asymmetric volume bands
    OSMIUM_NEUTRAL_BAND = 20
    OSMIUM_LOADED_BAND = 50

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80

    # Clock drift: day start + 0.001 per timestamp unit (verified across all 3 days)
    PEPPER_CLOCK_DRIFT = 0.001

    # OLS window length
    PEPPER_OLS_LEN = 6

    # Blend weights for fair value
    PEPPER_OLS_WEIGHT = 0.70
    PEPPER_CLOCK_WEIGHT = 0.30

    # Trend slope threshold (OLS slope units: ticks per window-step ≈ 100ms)
    # Expected uptrend slope ≈ 0.10 ticks/step. Gate at 0.03 = 30% of expected.
    PEPPER_TREND_GATE = 0.03

    # Taking behaviour per regime:
    #   STRONG uptrend → take ALL asks (no fair-value check), maximize long
    #   WEAK uptrend / neutral → only take clear edge (ask < fair - TAKE_EDGE)
    PEPPER_TAKE_EDGE = 0.5

    # Passive quoting minimum edge from reservation price
    PEPPER_QUOTE_EDGE = 1

    # Reservation-price inventory skew coefficient
    PEPPER_INV_SKEW = 3.0
    PEPPER_URGENCY_THRESH = 0.65
    PEPPER_URGENCY_MULT = 2.5

    # Target long position in uptrend (biases passive quotes toward buying)
    PEPPER_TARGET_LONG = 50

    def run(self, state: TradingState):
        result = {}
        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("pepper_history", [])
        saved.setdefault("pepper_day_start", None)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if product == self.PEPPER:
                result[product] = self._trade_pepper(
                    od, position, state.timestamp, saved
                )
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OLS helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _ols(history: list) -> Tuple[float, float]:
        """OLS linear regression on history.

        Returns (predicted_next, slope) where:
          predicted_next = fitted value at x = len(history)
          slope          = ticks per window-step (x units = 1 per data point)

        Correct for any window size n ≥ 2.
        """
        n = len(history)
        if n < 2:
            return (history[0] if history else 0.0), 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n
        sum_wp = sum((i - x_mean) * p for i, p in enumerate(history))
        sum_w2 = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum_wp / sum_w2 if sum_w2 > 0 else 0.0
        predicted = y_mean + slope * (n - x_mean)
        return predicted, slope

    # ── OSMIUM ──────────────────────────────────────────────────────────────

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT
        fair = self.OSMIUM_FAIR

        # ── Phase 1: Snipe mispricings ──
        for ask in sorted(od.sell_orders):
            if ask > fair - self.OSMIUM_TAKE_EDGE:  # ask > 9999 → stop
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], can_buy)
            if qty > 0:
                orders.append(Order(self.OSMIUM, ask, qty))
                position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fair + self.OSMIUM_TAKE_EDGE:  # bid < 10001 → stop
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            qty = min(od.buy_orders[bid], can_sell)
            if qty > 0:
                orders.append(Order(self.OSMIUM, bid, -qty))
                position -= qty

        # ── Phase 2: Passive penny inside the spread ──
        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        passive_buy = min(
            (max(comp_bids) + 1) if comp_bids else self.OSMIUM_QUOTE_BUY,
            self.OSMIUM_QUOTE_BUY,
        )
        passive_sell = max(
            (min(comp_asks) - 1) if comp_asks else self.OSMIUM_QUOTE_SELL,
            self.OSMIUM_QUOTE_SELL,
        )
        if passive_buy >= passive_sell:
            passive_buy = self.OSMIUM_QUOTE_BUY
            passive_sell = self.OSMIUM_QUOTE_SELL

        # ── Asymmetric volume: penalise "adding more of same direction" ──
        abs_pos = abs(position)
        neutral = self.OSMIUM_NEUTRAL_BAND
        loaded = self.OSMIUM_LOADED_BAND

        if abs_pos <= neutral:
            buy_mult = sell_mult = 1.0
        elif abs_pos <= loaded:
            t = (abs_pos - neutral) / (loaded - neutral)
            wrong_mult = 1.0 - 0.7 * t
            buy_mult, sell_mult = (wrong_mult, 1.0) if position > 0 else (1.0, wrong_mult)
        else:
            t = min(1.0, (abs_pos - loaded) / (limit - loaded))
            wrong_mult = max(0.05, 0.3 - 0.25 * t)
            buy_mult, sell_mult = (wrong_mult, 1.0) if position > 0 else (1.0, wrong_mult)

        remaining_buy = max(0, int(round((limit - position) * buy_mult)))
        remaining_sell = max(0, int(round((limit + position) * sell_mult)))

        if remaining_buy > 0:
            orders.append(Order(self.OSMIUM, passive_buy, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(self.OSMIUM, passive_sell, -remaining_sell))

        logger.print(
            f"OSM| pb={passive_buy} ps={passive_sell} "
            f"rb={remaining_buy} rs={remaining_sell} pos={position}"
        )
        return orders

    # ── PEPPER ──────────────────────────────────────────────────────────────

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

        # Wall-mid: filter noisy small orders by anchoring on highest-volume levels
        if has_bid and has_ask:
            heavy_bid = max(od.buy_orders, key=lambda p: od.buy_orders[p])
            heavy_ask = min(od.sell_orders, key=lambda p: (-od.sell_orders[p], p))
            ref_mid = 0.5 * ((heavy_bid + heavy_ask) / 2.0) + 0.5 * mid
        else:
            ref_mid = mid

        # ── Update price history ──
        history = saved["pepper_history"]
        history.append(ref_mid)
        if len(history) > self.PEPPER_OLS_LEN:
            history.pop(0)

        # Clock drift anchor: first observed price + deterministic drift rate
        if saved["pepper_day_start"] is None:
            saved["pepper_day_start"] = ref_mid
        clock_fair = saved["pepper_day_start"] + self.PEPPER_CLOCK_DRIFT * timestamp

        # OLS regression: correct weights for actual window size (any n ≥ 2)
        if len(history) >= 2:
            ols_fair, ols_slope = self._ols(history)
        else:
            ols_fair, ols_slope = ref_mid, 0.0

        # Blend: OLS primary (responsive) + clock drift secondary (stable anchor)
        fair = self.PEPPER_OLS_WEIGHT * ols_fair + self.PEPPER_CLOCK_WEIGHT * clock_fair

        # Trend classification using OLS slope (stable across 6 points)
        uptrend = ols_slope > self.PEPPER_TREND_GATE
        downtrend = ols_slope < -self.PEPPER_TREND_GATE

        # ── Phase 1: Aggressive taking ──
        if uptrend:
            # Strong uptrend: buy ALL available asks to maximise long accumulation.
            # Every ask we buy now will be worth more in future ticks.
            if has_ask:
                for ask in sorted(od.sell_orders):
                    can_buy = limit - position
                    if can_buy <= 0:
                        break
                    qty = min(-od.sell_orders[ask], can_buy)
                    if qty > 0:
                        orders.append(Order(self.PEPPER, ask, qty))
                        position += qty
        elif downtrend:
            # Strong downtrend: sell ALL available bids to maximise short accumulation.
            if has_bid:
                for bid in sorted(od.buy_orders, reverse=True):
                    can_sell = limit + position
                    if can_sell <= 0:
                        break
                    qty = min(od.buy_orders[bid], can_sell)
                    if qty > 0:
                        orders.append(Order(self.PEPPER, bid, -qty))
                        position -= qty
        else:
            # Neutral: take only when clear edge vs fair value
            buy_threshold = fair - self.PEPPER_TAKE_EDGE
            sell_threshold = fair + self.PEPPER_TAKE_EDGE
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
            if has_bid:
                for bid in sorted(od.buy_orders, reverse=True):
                    if bid <= sell_threshold:
                        break
                    can_sell = limit + position
                    if can_sell <= 0:
                        break
                    qty = min(od.buy_orders[bid], can_sell)
                    if qty > 0:
                        orders.append(Order(self.PEPPER, bid, -qty))
                        position -= qty

        # ── Phase 2: Passive quoting with inventory skew ──
        inv_frac = position / limit
        skew = self.PEPPER_INV_SKEW * inv_frac
        if abs(inv_frac) > self.PEPPER_URGENCY_THRESH:
            skew *= self.PEPPER_URGENCY_MULT
        reservation = fair - skew

        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        penny_buy = (max(comp_bids) + 1) if comp_bids else (int(reservation) - self.PEPPER_QUOTE_EDGE)
        penny_sell = (min(comp_asks) - 1) if comp_asks else (int(reservation) + self.PEPPER_QUOTE_EDGE)

        passive_buy = min(penny_buy, int(reservation) - self.PEPPER_QUOTE_EDGE)
        passive_sell = max(penny_sell, int(reservation) + self.PEPPER_QUOTE_EDGE)
        if passive_buy >= passive_sell:
            passive_buy = int(reservation) - self.PEPPER_QUOTE_EDGE
            passive_sell = int(reservation) + self.PEPPER_QUOTE_EDGE

        # ── Asymmetric capacity ──
        remaining_buy = limit - position
        remaining_sell = limit + position

        if uptrend:
            # Inventory floor: in uptrend, never go net short.
            remaining_sell = max(0, min(remaining_sell, position))
            # Below target long: post full buy capacity to accumulate faster
            if position < self.PEPPER_TARGET_LONG:
                pass  # full remaining_buy
            else:
                # Near/at limit: scale down buy slightly to avoid pinning at limit
                excess = (position - self.PEPPER_TARGET_LONG) / max(1, limit - self.PEPPER_TARGET_LONG)
                remaining_buy = max(5, int(round(remaining_buy * (1.0 - 0.4 * excess))))

        elif downtrend:
            # Mirror: never go net long in downtrend
            remaining_buy = max(0, min(remaining_buy, -position))
            if position > -self.PEPPER_TARGET_LONG:
                pass
            else:
                excess = (-position - self.PEPPER_TARGET_LONG) / max(1, limit - self.PEPPER_TARGET_LONG)
                remaining_sell = max(5, int(round(remaining_sell * (1.0 - 0.4 * excess))))

        if remaining_buy > 0:
            orders.append(Order(self.PEPPER, passive_buy, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(self.PEPPER, passive_sell, -remaining_sell))

        logger.print(
            f"PEP| mid={mid:.1f} ols={ols_fair:.2f} clk={clock_fair:.2f} "
            f"fair={fair:.2f} slp={ols_slope:.3f} res={reservation:.2f} "
            f"pb={passive_buy} ps={passive_sell} "
            f"rb={remaining_buy} rs={remaining_sell} pos={position} "
            f"trend={'UP' if uptrend else 'DN' if downtrend else '--'}"
        )
        return orders
