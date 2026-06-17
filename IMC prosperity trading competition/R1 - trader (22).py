
import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
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
    v19 hybrid:

    OSMIUM
      - fixed fair around 10000
      - take clear edge
      - passive 9999 / 10001 with inventory-aware size

    PEPPER
      - long-biased trend trader / market maker hybrid
      - uses robust fast/slow EMA trend signal on large-quote fair
      - buys aggressively below forecast
      - DOES NOT sell inventory away cheaply
      - keeps a long floor in uptrend
      - only posts asks at a premium above forecast / reservation
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    PEPPER_LIMIT = 80
    OSMIUM_LIMIT = 80

    # PEPPER params
    PEPPER_MM_VOLUME = 18
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_EMV_ALPHA = 0.15
    PEPPER_TREND_CARRY = 0.55
    PEPPER_SKEW = 0.012
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 2.5
    PEPPER_VOL_THRESH = 4.0
    PEPPER_SPREAD_HIGH = 1

    # breakthrough idea: keep long exposure; do not sell it away
    PEPPER_LONG_FLOOR = 50
    PEPPER_STRONG_LONG_FLOOR = 65
    PEPPER_TREND_GATE = 0.20
    PEPPER_BUY_TAKE_EDGE = 1.0
    PEPPER_SELL_TAKE_EDGE = 3.0
    PEPPER_SELL_PREMIUM = 2
    PEPPER_L1_FRAC = 1.0
    PEPPER_L2_FRAC = 0.0

    # OSMIUM params
    OSMIUM_FAIR = 10000
    OSMIUM_TAKE_EDGE = 1
    OSMIUM_PASSIVE_BUY = 9999
    OSMIUM_PASSIVE_SELL = 10001
    OSMIUM_L1_FRAC = 0.80
    OSMIUM_L2_FRAC = 0.20

    @staticmethod
    def _large_quote_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        large_bids = [p for p, v in od.buy_orders.items() if abs(v) >= min_volume]
        large_asks = [p for p, v in od.sell_orders.items() if abs(v) >= min_volume]
        if large_bids and large_asks:
            return (max(large_bids) + min(large_asks)) / 2.0
        return fallback

    @staticmethod
    def _penny_passive_prices(od: OrderDepth, ref: float, fair: float,
                               min_buy_edge: int, min_sell_edge: int):
        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        penny_buy = (max(comp_bids) + 1) if comp_bids else (int(ref) - min_buy_edge - 1)
        penny_sell = (min(comp_asks) - 1) if comp_asks else (int(ref) + min_sell_edge + 1)

        passive_buy = min(penny_buy, int(ref) - min_buy_edge)
        passive_sell = max(penny_sell, int(ref) + min_sell_edge)

        if passive_buy >= passive_sell:
            passive_buy = int(ref) - min_buy_edge
            passive_sell = int(ref) + min_sell_edge

        return passive_buy, passive_sell

    @staticmethod
    def _two_level_orders(symbol: str, passive_buy: int, passive_sell: int,
                           remaining_buy: int, remaining_sell: int,
                           l1_frac: float, l2_frac: float) -> List[Order]:
        orders = []
        if remaining_buy > 0:
            l1_qty = max(1, round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            orders.append(Order(symbol, passive_buy, l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))
        if remaining_sell > 0:
            l1_qty = max(1, round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            orders.append(Order(symbol, passive_sell, -l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))
        return orders

    def run(self, state: TradingState):
        result = {}

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("pepper_ema", None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv", None)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position)
            elif product == self.PEPPER:
                result[product] = self._trade_pepper(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT
        fair = self.OSMIUM_FAIR

        if not od.buy_orders and not od.sell_orders:
            return orders

        # take clear edge
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

        # passive around fixed fair with asymmetric sizes
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
            self.OSMIUM_L1_FRAC,
            self.OSMIUM_L2_FRAC,
        )

        logger.print(
            f"OSM | fair={fair} pos={position} rb={remaining_buy} rs={remaining_sell}"
        )
        return orders

    def _trade_pepper(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        if not od.buy_orders and not od.sell_orders:
            return orders

        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            mid = (best_bid + best_ask) / 2.0
            microprice = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) > 0 else mid
        elif od.buy_orders:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)

        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)

        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            a = self.PEPPER_EMA_ALPHA
            saved["pepper_ema"] = a * ref_price + (1.0 - a) * saved["pepper_ema"]
        ema = saved["pepper_ema"]

        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            a = self.PEPPER_EMA_SLOW_ALPHA
            saved["pepper_ema_slow"] = a * ref_price + (1.0 - a) * saved["pepper_ema_slow"]
        ema_slow = saved["pepper_ema_slow"]

        if saved["pepper_last_mid"] is not None:
            diff_sq = (mid - saved["pepper_last_mid"]) ** 2
            emv_prev = saved["pepper_emv"]
            if emv_prev is None:
                saved["pepper_emv"] = diff_sq
            else:
                a_v = self.PEPPER_EMV_ALPHA
                saved["pepper_emv"] = a_v * diff_sq + (1.0 - a_v) * emv_prev
        saved["pepper_last_mid"] = mid

        emv = saved["pepper_emv"]
        vol = emv ** 0.5 if emv is not None else 2.0
        spread_extra = self.PEPPER_SPREAD_HIGH if vol > self.PEPPER_VOL_THRESH else 0

        trend_signal = ema - ema_slow
        forecast = ema + self.PEPPER_TREND_CARRY * trend_signal

        # aggressive buy below forecast
        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                if ask > forecast + self.PEPPER_BUY_TAKE_EDGE:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        # only aggressive sell on very rich prices, and never below held inventory
        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < forecast + self.PEPPER_SELL_TAKE_EDGE:
                    break
                can_sell = min(limit + position, max(0, position - self.PEPPER_LONG_FLOOR))
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        # passive
        skew = self.PEPPER_SKEW * position
        if abs(position) > self.PEPPER_URGENCY_THRESH * limit:
            skew *= self.PEPPER_URGENCY_MULT

        reservation = forecast - skew

        min_edge = 1 + spread_extra

        passive_buy, _ = self._penny_passive_prices(
            od,
            ref=reservation,
            fair=forecast,
            min_buy_edge=min_edge,
            min_sell_edge=min_edge,
        )

        # sell side intentionally premium-priced so we don't give the trend away
        premium_floor = self.PEPPER_STRONG_LONG_FLOOR if trend_signal > self.PEPPER_TREND_GATE else self.PEPPER_LONG_FLOOR
        passive_sell = int(round(max(forecast + self.PEPPER_SELL_PREMIUM, reservation + min_edge)))

        # buy size can still be large, but sell size is capped by long floor
        remaining_buy = limit - position
        remaining_sell = max(0, position - premium_floor)

        # in strong downtrend, stop adding longs aggressively
        if trend_signal < -self.PEPPER_TREND_GATE:
            remaining_buy = int(round(remaining_buy * 0.30))

        orders += self._two_level_orders(
            self.PEPPER,
            passive_buy,
            passive_sell,
            remaining_buy,
            remaining_sell,
            self.PEPPER_L1_FRAC,
            self.PEPPER_L2_FRAC,
        )

        logger.print(
            f"PEP | mid={mid:.1f} ref={ref_price:.1f} ema={ema:.2f} slow={ema_slow:.2f} "
            f"sig={trend_signal:.2f} fcst={forecast:.2f} res={reservation:.2f} "
            f"pb={passive_buy} ps={passive_sell} pos={position} rb={remaining_buy} rs={remaining_sell}"
        )
        return orders
