import json
import jsonpickle
from typing import Any, List, Tuple

from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
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
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state, trader_data):
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

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations):
        conv = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

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
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    LIMIT = 80

    # PEPPER is a drift product in the released R1 days. This block preserves
    # the strongest carry behavior from the earlier files and avoids selling
    # the core long unless inventory is already comfortably above the floor.
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_SKEW = 0.015
    PEPPER_SPREAD_HIGH = 1
    PEPPER_VOL_THRESH = 4.0
    PEPPER_EMV_ALPHA = 0.15
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 3.0
    PEPPER_TREND_CARRY = 0.55
    PEPPER_TREND_GATE = 0.0
    PEPPER_AGAINST_TREND_CAP = 0.55
    PEPPER_CARRY_EDGE = 0.0
    PEPPER_LONG_FLOOR = 65
    PEPPER_MM_VOLUME = 18

    # OSMIUM is mean-reverting, but the book is wide. The edge is to quote
    # near the touch when the reservation value allows it, then throttle the
    # crowded side before inventory becomes a problem.
    OSMIUM_FAST_ALPHA = 0.35
    OSMIUM_SLOW_ALPHA = 0.035
    OSMIUM_INV_SKEW = 6.0
    OSMIUM_WIDTH_BASE = 1.0
    OSMIUM_WIDTH_SPREAD_MULT = 0.15
    OSMIUM_TAKE_EDGE = 1.0
    OSMIUM_OBI_WEIGHT = 0.00
    OSMIUM_FLOW_ALPHA = 0.22
    OSMIUM_FLOW_WEIGHT = 0.00
    OSMIUM_L1_FRAC = 0.80
    OSMIUM_L2_FRAC = 0.20

    def run(self, state: TradingState):
        saved = jsonpickle.decode(state.traderData) if state.traderData else {}

        defaults = {
            "pepper_ema": None,
            "pepper_ema_slow": None,
            "pepper_last_mid": None,
            "pepper_emv": None,
            "osmium_fast": None,
            "osmium_slow": None,
            "osmium_flow": 0.0,
        }
        for key, value in defaults.items():
            saved.setdefault(key, value)

        result = {}
        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if product == self.PEPPER:
                result[product] = self._trade_pepper(od, position, saved)
            elif product == self.OSMIUM:
                prior_trades = state.market_trades.get(product, [])
                result[product] = self._trade_osmium(od, position, prior_trades, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    @staticmethod
    def _book_stats(od: OrderDepth) -> Tuple[bool, bool, float, float, int, int]:
        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
        elif has_bid:
            best_bid = max(od.buy_orders)
            best_ask = best_bid + 2
            mid = float(best_bid)
            spread = 2
        elif has_ask:
            best_ask = min(od.sell_orders)
            best_bid = best_ask - 2
            mid = float(best_ask)
            spread = 2
        else:
            return False, False, 0.0, 0.0, 0, 0
        return has_bid, has_ask, mid, spread, best_bid, best_ask

    @staticmethod
    def _microprice(od: OrderDepth, mid: float, best_bid: int, best_ask: int) -> float:
        if best_bid in od.buy_orders and best_ask in od.sell_orders:
            bid_volume = od.buy_orders[best_bid]
            ask_volume = -od.sell_orders[best_ask]
            if bid_volume + ask_volume > 0:
                return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        return mid

    @staticmethod
    def _obi(od: OrderDepth) -> float:
        bid_volume = sum(max(0, volume) for volume in od.buy_orders.values())
        ask_volume = sum(max(0, -volume) for volume in od.sell_orders.values())
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0

    @staticmethod
    def _large_quote_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        large_bids = [price for price, volume in od.buy_orders.items() if abs(volume) >= min_volume]
        large_asks = [price for price, volume in od.sell_orders.items() if abs(volume) >= min_volume]
        if large_bids and large_asks:
            return (max(large_bids) + min(large_asks)) / 2.0
        return fallback

    @staticmethod
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,
        fair: float,
        min_buy_edge: int,
        min_sell_edge: int,
    ) -> Tuple[int, int]:
        comp_bids = [price for price in od.buy_orders if price < fair]
        comp_asks = [price for price in od.sell_orders if price > fair]

        penny_buy = max(comp_bids) + 1 if comp_bids else int(ref) - min_buy_edge - 1
        penny_sell = min(comp_asks) - 1 if comp_asks else int(ref) + min_sell_edge + 1

        passive_buy = min(penny_buy, int(ref) - min_buy_edge)
        passive_sell = max(penny_sell, int(ref) + min_sell_edge)
        return passive_buy, passive_sell

    @staticmethod
    def _add_split_orders(
        orders: List[Order],
        symbol: str,
        buy_price: int,
        sell_price: int,
        buy_qty: int,
        sell_qty: int,
        l1_frac: float,
        l2_frac: float,
    ) -> None:
        def split(total: int) -> Tuple[int, int]:
            if total <= 0:
                return 0, 0
            l1 = round(total * l1_frac)
            l2 = round(total * l2_frac)
            if l1 + l2 == 0:
                l1 = total
            if l1 + l2 > total:
                overflow = l1 + l2 - total
                cut = min(l2, overflow)
                l2 -= cut
                overflow -= cut
                if overflow > 0:
                    l1 = max(0, l1 - overflow)
            if l1 + l2 < total:
                l1 += total - (l1 + l2)
            if l1 == 0:
                l1 = 1
                l2 = max(0, l2 - 1)
            return l1, l2

        if buy_qty > 0:
            l1, l2 = split(buy_qty)
            orders.append(Order(symbol, buy_price, l1))
            if l2 > 0:
                orders.append(Order(symbol, buy_price - 1, l2))

        if sell_qty > 0:
            l1, l2 = split(sell_qty)
            orders.append(Order(symbol, sell_price, -l1))
            if l2 > 0:
                orders.append(Order(symbol, sell_price + 1, -l2))

    def _trade_pepper(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        orders: List[Order] = []
        has_bid, has_ask, mid, _, best_bid, best_ask = self._book_stats(od)
        if not has_bid and not has_ask:
            return orders

        microprice = mid
        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)

        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            saved["pepper_ema"] = self.PEPPER_EMA_ALPHA * ref_price + (1.0 - self.PEPPER_EMA_ALPHA) * saved["pepper_ema"]

        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            saved["pepper_ema_slow"] = self.PEPPER_EMA_SLOW_ALPHA * ref_price + (1.0 - self.PEPPER_EMA_SLOW_ALPHA) * saved["pepper_ema_slow"]

        ema = saved["pepper_ema"]
        trend_signal = ema - saved["pepper_ema_slow"]
        forecast = ema + self.PEPPER_TREND_CARRY * trend_signal

        if saved["pepper_last_mid"] is not None:
            diff_sq = (mid - saved["pepper_last_mid"]) ** 2
            if saved["pepper_emv"] is None:
                saved["pepper_emv"] = diff_sq
            else:
                saved["pepper_emv"] = self.PEPPER_EMV_ALPHA * diff_sq + (1.0 - self.PEPPER_EMV_ALPHA) * saved["pepper_emv"]
        saved["pepper_last_mid"] = mid

        vol = saved["pepper_emv"] ** 0.5 if saved["pepper_emv"] is not None else 2.0
        min_edge = 1 + (self.PEPPER_SPREAD_HIGH if vol > self.PEPPER_VOL_THRESH else 0)
        carry_edge = self.PEPPER_CARRY_EDGE if trend_signal >= -0.5 else 0.0

        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= forecast + carry_edge:
                    break
                can_buy = self.LIMIT - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= forecast + carry_edge:
                    break
                can_sell = self.LIMIT + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        biased_pos = max(-self.LIMIT, min(self.LIMIT, position))
        skew = self.PEPPER_SKEW * biased_pos
        if abs(biased_pos) > self.PEPPER_URGENCY_THRESH * self.LIMIT:
            skew *= self.PEPPER_URGENCY_MULT

        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=forecast - skew,
            fair=forecast,
            min_buy_edge=min_edge,
            min_sell_edge=min_edge,
        )

        buy_qty = self.LIMIT - position
        sell_qty = self.LIMIT + position
        if trend_signal < -self.PEPPER_TREND_GATE:
            buy_qty = int(round(buy_qty * self.PEPPER_AGAINST_TREND_CAP))
        elif trend_signal > self.PEPPER_TREND_GATE:
            sell_qty = int(round(sell_qty * self.PEPPER_AGAINST_TREND_CAP))
            sell_qty = min(sell_qty, max(0, position - self.PEPPER_LONG_FLOOR))

        self._add_split_orders(orders, self.PEPPER, passive_buy, passive_sell, buy_qty, sell_qty, 1.0, 0.0)
        return orders

    def _trade_osmium(self, od: OrderDepth, position: int, prior_trades, saved: dict) -> List[Order]:
        orders: List[Order] = []
        has_bid, has_ask, mid, spread, best_bid, best_ask = self._book_stats(od)
        if not has_bid and not has_ask:
            return orders

        microprice = self._microprice(od, mid, best_bid, best_ask)
        book_imbalance = self._obi(od)

        signed_flow = 0.0
        for trade in prior_trades:
            if trade.price >= best_ask:
                signed_flow += trade.quantity
            elif trade.price <= best_bid:
                signed_flow -= trade.quantity
        saved["osmium_flow"] = self.OSMIUM_FLOW_ALPHA * signed_flow + (1.0 - self.OSMIUM_FLOW_ALPHA) * saved["osmium_flow"]

        if saved["osmium_fast"] is None:
            saved["osmium_fast"] = microprice
            saved["osmium_slow"] = microprice
        else:
            saved["osmium_fast"] = self.OSMIUM_FAST_ALPHA * microprice + (1.0 - self.OSMIUM_FAST_ALPHA) * saved["osmium_fast"]
            saved["osmium_slow"] = self.OSMIUM_SLOW_ALPHA * microprice + (1.0 - self.OSMIUM_SLOW_ALPHA) * saved["osmium_slow"]

        trend = saved["osmium_fast"] - saved["osmium_slow"]
        fair = (
            saved["osmium_fast"]
            + self.OSMIUM_OBI_WEIGHT * spread * book_imbalance
            + self.OSMIUM_FLOW_WEIGHT * saved["osmium_flow"]
            - 0.00 * trend
        )

        reservation = fair - self.OSMIUM_INV_SKEW * (position / self.LIMIT)
        half_width = max(1, round(self.OSMIUM_WIDTH_BASE + self.OSMIUM_WIDTH_SPREAD_MULT * spread))

        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= reservation - self.OSMIUM_TAKE_EDGE:
                    break
                can_buy = self.LIMIT - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, ask, qty))
                    position += qty

        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= reservation + self.OSMIUM_TAKE_EDGE:
                    break
                can_sell = self.LIMIT + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        reservation = fair - self.OSMIUM_INV_SKEW * (position / self.LIMIT)
        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=reservation,
            fair=reservation,
            min_buy_edge=half_width,
            min_sell_edge=half_width,
        )
        if passive_buy >= passive_sell:
            passive_buy = int(reservation) - half_width
            passive_sell = int(reservation) + half_width

        buy_scale = max(0.20, 1.0 - max(0.0, position / self.LIMIT))
        sell_scale = max(0.20, 1.0 - max(0.0, -position / self.LIMIT))

        if book_imbalance < 0.0:
            buy_scale *= 0.60
        elif book_imbalance > 0.0:
            sell_scale *= 0.60

        buy_qty = int(round((self.LIMIT - position) * buy_scale))
        sell_qty = int(round((self.LIMIT + position) * sell_scale))

        self._add_split_orders(
            orders,
            self.OSMIUM,
            passive_buy,
            passive_sell,
            buy_qty,
            sell_qty,
            self.OSMIUM_L1_FRAC,
            self.OSMIUM_L2_FRAC,
        )
        return orders
