import json
from typing import Any, List, Tuple

import jsonpickle

from datamodel import (
    Order, OrderDepth, ProsperityEncoder, TradingState,
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
    v17: Pure trend-riding for PEPPER + hard-anchored OSMIUM

    Root-cause fix from v16:
      Passive sells at reservation+1 (≈ fair−2) were filling whenever
      bids rose to that level, cycling out of the long position too cheaply.
      In uptrend: remaining_sell = 0 — hold everything, profit from
      mark-to-market appreciation as price rises ~1000 ticks over each day.

    PEPPER strategy in uptrend:
      • Aggressively take ALL asks to accumulate max long (+80) fast.
      • Never post passive sells — ride the deterministic 0.001/tick drift.
      • Post passive buys to refill if position drops below target.

    PEPPER strategy in downtrend (mirror image):
      • Aggressively take ALL bids to accumulate max short (−80).
      • Never post passive buys.

    OSMIUM strategy (unchanged from v16):
      • Hard-coded fair = 10,000 (mean=10000.20, std=5.35 across 3 days).
      • Take asks ≤ 9999 and bids ≥ 10001 aggressively.
      • Passive quotes at 9999/10001 with asymmetric volume sizing.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_FAIR = 10_000
    OSMIUM_TAKE_EDGE = 1
    OSMIUM_QUOTE_BUY = 9_999
    OSMIUM_QUOTE_SELL = 10_001
    OSMIUM_NEUTRAL_BAND = 20
    OSMIUM_LOADED_BAND = 50

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_CLOCK_DRIFT = 0.001    # ticks per timestamp unit (verified)
    PEPPER_OLS_LEN = 6

    # OLS/clock blend weights
    PEPPER_OLS_WEIGHT = 0.70
    PEPPER_CLOCK_WEIGHT = 0.30

    # Trend gate: OLS slope threshold to classify as trending
    # Expected uptrend slope ≈ 0.10 ticks/step. Gate = 30% of expected.
    PEPPER_TREND_GATE = 0.03

    # When NOT trending: use fair-value edge for selective taking
    PEPPER_TAKE_EDGE = 0.5

    # Passive quote minimum edge from fair (used in neutral regime only)
    PEPPER_QUOTE_EDGE = 1

    # Inventory skew parameters (used only in neutral regime)
    PEPPER_INV_SKEW = 3.0
    PEPPER_URGENCY_THRESH = 0.65
    PEPPER_URGENCY_MULT = 2.5

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
        """OLS linear regression. Returns (predicted_next, slope).

        predicted_next = fitted value extrapolated one step beyond history.
        slope = ticks per data-point (each point ≈ 100 ms).
        Handles any window size n ≥ 2 correctly.
        """
        n = len(history)
        if n < 2:
            return (history[0] if history else 0.0), 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n
        sum_wp = sum((i - x_mean) * p for i, p in enumerate(history))
        sum_w2 = sum((i - x_mean) ** 2 for i in range(n))
        slope = sum_wp / sum_w2 if sum_w2 > 0 else 0.0
        return y_mean + slope * (n - x_mean), slope

    # ── OSMIUM ──────────────────────────────────────────────────────────────

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT
        fair = self.OSMIUM_FAIR

        # Phase 1: Snipe mispricings
        for ask in sorted(od.sell_orders):
            if ask > fair - self.OSMIUM_TAKE_EDGE:
                break
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], can_buy)
            if qty > 0:
                orders.append(Order(self.OSMIUM, ask, qty))
                position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fair + self.OSMIUM_TAKE_EDGE:
                break
            can_sell = limit + position
            if can_sell <= 0:
                break
            qty = min(od.buy_orders[bid], can_sell)
            if qty > 0:
                orders.append(Order(self.OSMIUM, bid, -qty))
                position -= qty

        # Phase 2: Passive penny inside the spread
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

        abs_pos = abs(position)
        if abs_pos <= self.OSMIUM_NEUTRAL_BAND:
            buy_mult = sell_mult = 1.0
        elif abs_pos <= self.OSMIUM_LOADED_BAND:
            t = (abs_pos - self.OSMIUM_NEUTRAL_BAND) / (self.OSMIUM_LOADED_BAND - self.OSMIUM_NEUTRAL_BAND)
            wrong_mult = 1.0 - 0.7 * t
            buy_mult, sell_mult = (wrong_mult, 1.0) if position > 0 else (1.0, wrong_mult)
        else:
            t = min(1.0, (abs_pos - self.OSMIUM_LOADED_BAND) / (limit - self.OSMIUM_LOADED_BAND))
            wrong_mult = max(0.05, 0.3 - 0.25 * t)
            buy_mult, sell_mult = (wrong_mult, 1.0) if position > 0 else (1.0, wrong_mult)

        remaining_buy = max(0, int(round((limit - position) * buy_mult)))
        remaining_sell = max(0, int(round((limit + position) * sell_mult)))

        if remaining_buy > 0:
            orders.append(Order(self.OSMIUM, passive_buy, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(self.OSMIUM, passive_sell, -remaining_sell))

        logger.print(f"OSM| pb={passive_buy} ps={passive_sell} rb={remaining_buy} rs={remaining_sell} pos={position}")
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
            # Wall-mid: anchor on highest-volume levels, filter small noisy quotes
            heavy_bid = max(od.buy_orders, key=lambda p: od.buy_orders[p])
            heavy_ask = min(od.sell_orders, key=lambda p: (-od.sell_orders[p], p))
            ref_mid = 0.5 * ((heavy_bid + heavy_ask) / 2.0) + 0.5 * mid
        elif has_bid:
            best_bid = max(od.buy_orders)
            best_ask = None
            mid = ref_mid = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            best_bid = None
            mid = ref_mid = float(best_ask)

        # Update price history
        history = saved["pepper_history"]
        history.append(ref_mid)
        if len(history) > self.PEPPER_OLS_LEN:
            history.pop(0)

        # Clock drift: anchor from day's first price observation
        if saved["pepper_day_start"] is None:
            saved["pepper_day_start"] = ref_mid
        clock_fair = saved["pepper_day_start"] + self.PEPPER_CLOCK_DRIFT * timestamp

        # OLS regression for predicted fair value + trend slope
        if len(history) >= 2:
            ols_fair, ols_slope = self._ols(history)
        else:
            ols_fair, ols_slope = ref_mid, 0.0

        fair = self.PEPPER_OLS_WEIGHT * ols_fair + self.PEPPER_CLOCK_WEIGHT * clock_fair

        uptrend = ols_slope > self.PEPPER_TREND_GATE
        downtrend = ols_slope < -self.PEPPER_TREND_GATE

        # ── Phase 1: Aggressive taking ──
        if uptrend:
            # Buy ALL available asks — every unit bought now appreciates at 0.1 tick/step.
            # Do not sell: remaining_sell = 0 in uptrend (see Phase 2).
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
            # Sell ALL available bids — mirror of uptrend.
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
            # Neutral: selective taking with fair-value edge
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

        # ── Phase 2: Passive quoting ──
        if uptrend:
            # Hold everything: never post passive sells.
            # Post aggressive passive buys to accumulate faster.
            remaining_buy = limit - position
            remaining_sell = 0  # key fix: no sells in uptrend

            if remaining_buy > 0:
                # Penny the best bid to maximise queue priority
                comp_bids = [p for p in od.buy_orders if p < fair]
                if comp_bids:
                    passive_buy = max(comp_bids) + 1
                else:
                    passive_buy = int(fair) - self.PEPPER_QUOTE_EDGE
                orders.append(Order(self.PEPPER, passive_buy, remaining_buy))

        elif downtrend:
            # Mirror of uptrend: hold short, no passive buys.
            remaining_buy = 0
            remaining_sell = limit + position

            if remaining_sell > 0:
                comp_asks = [p for p in od.sell_orders if p > fair]
                if comp_asks:
                    passive_sell = min(comp_asks) - 1
                else:
                    passive_sell = int(fair) + self.PEPPER_QUOTE_EDGE
                orders.append(Order(self.PEPPER, passive_sell, -remaining_sell))

        else:
            # Neutral: symmetric market making with inventory skew
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

            remaining_buy = limit - position
            remaining_sell = limit + position

            if remaining_buy > 0:
                orders.append(Order(self.PEPPER, passive_buy, remaining_buy))
            if remaining_sell > 0:
                orders.append(Order(self.PEPPER, passive_sell, -remaining_sell))

        logger.print(
            f"PEP| mid={mid:.1f} ols={ols_fair:.2f} clk={clock_fair:.2f} "
            f"fair={fair:.2f} slp={ols_slope:.3f} pos={position} "
            f"trend={'UP' if uptrend else 'DN' if downtrend else '--'}"
        )
        return orders
