
import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)


# ── Logger (same style as earlier trader files / visualizer-friendly) ──────
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
    Breakthrough version inspired directly by the pasted analysis.

    Core idea:
      1) OSMIUM behaves like a fixed-fair-value product around 10,000.
         Trade it like EMERALDS/AMETHYSTS:
           - aggressive take on clear mispricings
           - passive make at 9999 / 10001
           - asymmetric size when inventory is loaded

      2) PEPPER is treated as a directional hold-the-drift product for this
         backtest regime. The major breakthrough is:
           - do NOT market-make it symmetrically
           - do NOT passively sell it away
           - just accumulate to +80 and hold

    This preserves the same overall file/logger structure as the earlier
    trader.py files so it should visualize cleanly.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    PEPPER_LIMIT = 80
    OSMIUM_LIMIT = 80

    OSMIUM_FAIR = 10000
    OSMIUM_PASSIVE_BUY = 9999
    OSMIUM_PASSIVE_SELL = 10001

    @staticmethod
    def _best_bid(od: OrderDepth):
        return max(od.buy_orders) if od.buy_orders else None

    @staticmethod
    def _best_ask(od: OrderDepth):
        return min(od.sell_orders) if od.sell_orders else None

    @staticmethod
    def _two_level_orders(
        symbol: str,
        passive_buy: int,
        passive_sell: int,
        remaining_buy: int,
        remaining_sell: int,
        l1_frac: float,
        l2_frac: float,
    ) -> List[Order]:
        orders: List[Order] = []

        if remaining_buy > 0:
            l1_qty = max(1, round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            if l1_qty > 0:
                orders.append(Order(symbol, passive_buy, l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))

        if remaining_sell > 0:
            l1_qty = max(1, round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            if l1_qty > 0:
                orders.append(Order(symbol, passive_sell, -l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))

        return orders

    def run(self, state: TradingState):
        result = {}

        # Keep traderData structure compatible with previous versions.
        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("note", "v18_breakthrough")

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position)
            elif product == self.PEPPER:
                result[product] = self._trade_pepper(od, position)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OSMIUM: fixed fair at 10,000 ────────────────────────────────────────

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT
        fair = self.OSMIUM_FAIR

        if not od.buy_orders and not od.sell_orders:
            return orders

        # Phase 1: aggressive take on clear edge
        # buy clear discounts
        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                if ask > fair - 1:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, ask, qty))
                    position += qty

        # sell clear premiums
        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < fair + 1:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        # Phase 2: passive quoting at fixed anchor around 10,000
        # Asymmetric volume scaling:
        #   long -> smaller bids, larger asks
        #   short -> larger bids, smaller asks
        buy_scale = max(0.10, (limit - position) / (2 * limit))
        sell_scale = max(0.10, (limit + position) / (2 * limit))

        remaining_buy = min(limit - position, int(round(limit * buy_scale)))
        remaining_sell = min(limit + position, int(round(limit * sell_scale)))

        orders += self._two_level_orders(
            self.OSMIUM,
            self.OSMIUM_PASSIVE_BUY,
            self.OSMIUM_PASSIVE_SELL,
            remaining_buy,
            remaining_sell,
            0.80,
            0.20,
        )

        logger.print(
            f"OSM | fair={fair} pbuy={self.OSMIUM_PASSIVE_BUY} "
            f"psell={self.OSMIUM_PASSIVE_SELL} pos={position} "
            f"rb={remaining_buy} rs={remaining_sell}"
        )
        return orders

    # ── PEPPER: breakthrough = accumulate max long and never sell ───────────

    def _trade_pepper(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        if not od.sell_orders:
            logger.print(f"PEP | no asks pos={position}")
            return orders

        # Breakthrough logic from the pasted analysis:
        # buy aggressively until max long, then hold.
        remaining = limit - position
        if remaining <= 0:
            logger.print(f"PEP | holding max long pos={position}")
            return orders

        # Sweep available asks from best upward until the long limit is reached.
        for ask in sorted(od.sell_orders):
            can_buy = limit - position
            if can_buy <= 0:
                break
            qty = min(-od.sell_orders[ask], can_buy)
            if qty > 0:
                orders.append(Order(self.PEPPER, ask, qty))
                position += qty

        logger.print(f"PEP | accumulate-only pos={position}")
        return orders
