import json
from typing import Any, List

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
    v18: Max-long PEPPER + Hard-anchored OSMIUM

    Root-cause analysis from v16/v17:
      Short-term OLS slope is too noisy (noise std ≈ ±0.48, signal ≈ 0.1)
      to reliably distinguish uptrend from downtrend. Noisy switching between
      "buy all" and "sell all" caused -1.4M losses (buy at ask ~10005,
      sell at bid ~9995, -10 ticks × thousands of cycles).

    PEPPER strategy (v18):
      Data shows ALL 3 days trend at exactly +0.001 ticks/timestamp = +1000 ticks/day.
      Buy-and-hold benchmark: 79,591/day. Optimal strategy: hold max long always.
      - Take: buy at best ask every tick until position = PEPPER_LIMIT.
      - Never sell. Mark-to-market gain from position × rising price is the PnL source.
      - Also post passive buy at best_bid+1 to accumulate even cheaper.

    OSMIUM strategy (v18 = v16 unchanged):
      Hard-coded fair = 10,000 (mean=10000.20, std=5.35 across 3 days ≡ EMERALDS).
      - Snipe: take asks ≤ 9999, take bids ≥ 10001.
      - Make: passive 9999/10001 with asymmetric volume sizing.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    PEPPER_LIMIT = 80
    OSMIUM_LIMIT = 80

    # OSMIUM: hard-coded anchor
    OSMIUM_FAIR = 10_000
    OSMIUM_TAKE_EDGE = 1
    OSMIUM_QUOTE_BUY = 9_999
    OSMIUM_QUOTE_SELL = 10_001
    OSMIUM_NEUTRAL_BAND = 20
    OSMIUM_LOADED_BAND = 50

    def run(self, state: TradingState):
        result = {}
        saved = {}  # no persistent state needed for v18 PEPPER

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if product == self.PEPPER:
                result[product] = self._trade_pepper(od, position)
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position)

        conversions = 0
        logger.flush(state, result, conversions, "")
        return result, conversions, ""

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

    def _trade_pepper(self, od: OrderDepth, position: int) -> List[Order]:
        """Max-long strategy: always accumulate to PEPPER_LIMIT, never sell.

        PEPPER drifts +0.001 ticks per timestamp across all 3 data days
        (+1000 ticks total per day). Holding 80 units from early in the day
        captures ~79,000 in mark-to-market appreciation.

        Never sell: any sell trades give away position that is worth ~1 tick
        more per step. Over 10,000 steps, each unit held is worth +1000 ticks.
        """
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        if not od.sell_orders:
            # No asks available; post passive buy inside the bid side
            if od.buy_orders and position < limit:
                best_bid = max(od.buy_orders)
                passive_buy = best_bid + 1
                remaining = limit - position
                orders.append(Order(self.PEPPER, passive_buy, remaining))
            return orders

        # Buy at each ask level from cheapest upward until limit is reached.
        # Buying cheap > buying fast: prefer best_ask, accept worse levels
        # only when needed to fill up to the limit.
        for ask in sorted(od.sell_orders):
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], can_buy)
            if qty > 0:
                orders.append(Order(self.PEPPER, ask, qty))
                position += qty

        # If still below limit, post a passive buy at best_bid+1 to
        # accumulate more cheaply via queue priority.
        if position < limit and od.buy_orders:
            remaining = limit - position
            best_bid = max(od.buy_orders)
            passive_buy = best_bid + 1
            # Only post if this is below the asks we just cleared
            best_remaining_ask = min(od.sell_orders) if od.sell_orders else passive_buy + 1
            if passive_buy < best_remaining_ask:
                orders.append(Order(self.PEPPER, passive_buy, remaining))

        logger.print(f"PEP| pos={position}")
        return orders
