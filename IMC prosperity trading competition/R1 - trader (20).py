
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
    trader_v15_gemini_style.py

    Direct implementation of the requested design:
      - OSMIUM hard anchor at 10000
      - PEPPER predictive fair from large-quote "wall mid" + OBI shift + trend carry
      - Aggressive take-before-make on both products
      - Inventory-aware asymmetric volume quoting
      - Preserve logger/jsonpickle state style

    Important:
      This intentionally follows the requested "Gemini-style" assumptions,
      including the hard anchor for OSMIUM.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_EMV_ALPHA = 0.15
    PEPPER_TREND_CARRY = 0.55
    PEPPER_SKEW = 0.015
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 3.0
    PEPPER_TREND_GATE = 0.0
    PEPPER_AGAINST_TREND_CAP = 0.55
    PEPPER_LONG_FLOOR = 65
    PEPPER_VOL_THRESH = 4.0
    PEPPER_SPREAD_HIGH = 1
    PEPPER_MM_VOLUME = 18

    # Requested predictive additions
    PEPPER_WALL_MIN_VOL = 18
    PEPPER_OBI_SHIFT = 1.5
    PEPPER_TAKE_EDGE = 1
    PEPPER_L1_FRAC = 1.0
    PEPPER_L2_FRAC = 0.0

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_HARD_FAIR = 10000.0
    OSMIUM_SKEW = 6.0
    OSMIUM_TAKE_EDGE = 1
    OSMIUM_L1_FRAC = 0.80
    OSMIUM_L2_FRAC = 0.20

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mid(od: OrderDepth) -> float:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        if od.buy_orders:
            return float(max(od.buy_orders))
        if od.sell_orders:
            return float(min(od.sell_orders))
        return 0.0

    @staticmethod
    def _book_imbalance(od: OrderDepth) -> float:
        bid_volume = sum(max(0, v) for v in od.buy_orders.values())
        ask_volume = sum(max(0, -v) for v in od.sell_orders.values())
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    @staticmethod
    def _large_quote_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        large_bids = [p for p, v in od.buy_orders.items() if abs(v) >= min_volume]
        large_asks = [p for p, v in od.sell_orders.items() if abs(v) >= min_volume]
        if large_bids and large_asks:
            return (max(large_bids) + min(large_asks)) / 2.0
        return fallback

    @staticmethod
    def _wall_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        """
        "Wall-mid" variant:
        - Prefer the largest-volume bid/ask levels among quotes with meaningful size.
        - Falls back to large-quote mid or ordinary fallback.
        """
        bid_candidates = [(p, v) for p, v in od.buy_orders.items() if abs(v) >= min_volume]
        ask_candidates = [(p, -v) for p, v in od.sell_orders.items() if abs(v) >= min_volume]

        if bid_candidates and ask_candidates:
            wall_bid = max(bid_candidates, key=lambda x: (x[1], x[0]))[0]
            wall_ask = min(ask_candidates, key=lambda x: (-x[1], x[0]))[0]
            return (wall_bid + wall_ask) / 2.0

        return Trader._large_quote_mid(od, fallback, min_volume)

    @staticmethod
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,
        fair: float,
        min_buy_edge: int,
        min_sell_edge: int,
    ):
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
            l1_qty = int(round(remaining_buy * l1_frac))
            l2_qty = remaining_buy - l1_qty
            if l1_qty > 0:
                orders.append(Order(symbol, passive_buy, l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_buy - 1, l2_qty))

        if remaining_sell > 0:
            l1_qty = int(round(remaining_sell * l1_frac))
            l2_qty = remaining_sell - l1_qty
            if l1_qty > 0:
                orders.append(Order(symbol, passive_sell, -l1_qty))
            if l2_qty > 0 and l2_frac > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2_qty))

        return orders

    @staticmethod
    def _asymmetric_remaining(limit: int, position: int, wrong_mult: float, exit_mult: float):
        """
        Volume skew instead of only price skew.
        If long: buy side is "wrong side", sell side is exit side.
        If short: sell side is "wrong side", buy side is exit side.
        """
        max_buy = limit - position
        max_sell = limit + position

        if position > 0:
            remaining_buy = max(0, int(round(max_buy * wrong_mult)))
            remaining_sell = max(0, int(round(max_sell * exit_mult)))
        elif position < 0:
            remaining_buy = max(0, int(round(max_buy * exit_mult)))
            remaining_sell = max(0, int(round(max_sell * wrong_mult)))
        else:
            remaining_buy = max_buy
            remaining_sell = max_sell

        return remaining_buy, remaining_sell

    def run(self, state: TradingState):
        result = {}

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("pepper_ema", None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv", None)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if product == self.PEPPER:
                result[product] = self._trade_pepper(od, position, saved)
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OSMIUM: hard-anchor + take-then-make ───────────────────────────────

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT

        if not od.buy_orders and not od.sell_orders:
            return orders

        fair = self.OSMIUM_HARD_FAIR
        reservation = fair - self.OSMIUM_SKEW * (position / limit)

        # Take phase
        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                if ask > reservation - self.OSMIUM_TAKE_EDGE:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, ask, qty))
                    position += qty

        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < reservation + self.OSMIUM_TAKE_EDGE:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        # Passive phase
        reservation = fair - self.OSMIUM_SKEW * (position / limit)
        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=reservation,
            fair=fair,
            min_buy_edge=1,
            min_sell_edge=1,
        )

        # Asymmetric volume quoting per requested idea
        wrong_mult = 0.20 if abs(position) > 0.75 * limit else 0.40
        exit_mult = 1.00
        remaining_buy, remaining_sell = self._asymmetric_remaining(
            limit, position, wrong_mult, exit_mult
        )

        orders += self._two_level_orders(
            self.OSMIUM,
            passive_buy,
            passive_sell,
            remaining_buy,
            remaining_sell,
            self.OSMIUM_L1_FRAC,
            self.OSMIUM_L2_FRAC,
        )

        logger.print(
            f"OSM | fair={fair:.1f} res={reservation:.2f} "
            f"pb={passive_buy} ps={passive_sell} pos={position}"
        )
        return orders

    # ── PEPPER: wall-mid + OBI + take-then-make ────────────────────────────

    def _trade_pepper(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        if not od.buy_orders and not od.sell_orders:
            return orders

        mid = self._mid(od)
        wall_mid = self._wall_mid(od, mid, self.PEPPER_WALL_MIN_VOL)
        large_mid = self._large_quote_mid(od, mid, self.PEPPER_MM_VOLUME)
        ref_price = wall_mid if wall_mid != 0 else large_mid

        # EMA pair on large/wall fair rather than noisy mid
        a_fast = self.PEPPER_EMA_ALPHA
        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            saved["pepper_ema"] = a_fast * ref_price + (1.0 - a_fast) * saved["pepper_ema"]
        ema = saved["pepper_ema"]

        a_slow = self.PEPPER_EMA_SLOW_ALPHA
        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            saved["pepper_ema_slow"] = (
                a_slow * ref_price + (1.0 - a_slow) * saved["pepper_ema_slow"]
            )
        ema_slow = saved["pepper_ema_slow"]

        # Vol estimate
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

        # Predictive fair = trend carry + OBI shift
        trend_signal = ema - ema_slow
        forecast = ema + self.PEPPER_TREND_CARRY * trend_signal

        obi = self._book_imbalance(od)
        if obi > 0.5:
            forecast += self.PEPPER_OBI_SHIFT
        elif obi < -0.5:
            forecast -= self.PEPPER_OBI_SHIFT

        skew = self.PEPPER_SKEW * position
        if abs(position) > self.PEPPER_URGENCY_THRESH * limit:
            skew *= self.PEPPER_URGENCY_MULT

        eff_fair = forecast - skew
        min_edge = 1 + spread_extra

        # Take phase
        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                if ask > eff_fair - self.PEPPER_TAKE_EDGE:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < eff_fair + self.PEPPER_TAKE_EDGE:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        # Passive phase
        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=eff_fair,
            fair=forecast,
            min_buy_edge=min_edge,
            min_sell_edge=min_edge,
        )

        # Keep core long carry in uptrend; skew volume, not just price
        wrong_mult = 0.25 if abs(position) > self.PEPPER_URGENCY_THRESH * limit else 0.50
        exit_mult = 1.00
        remaining_buy, remaining_sell = self._asymmetric_remaining(
            limit, position, wrong_mult, exit_mult
        )

        if trend_signal < -self.PEPPER_TREND_GATE:
            remaining_buy = int(round(remaining_buy * self.PEPPER_AGAINST_TREND_CAP))
        elif trend_signal > self.PEPPER_TREND_GATE:
            remaining_sell = int(round(remaining_sell * self.PEPPER_AGAINST_TREND_CAP))
            remaining_sell = min(remaining_sell, max(0, position - self.PEPPER_LONG_FLOOR))

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
            f"PEP | mid={mid:.1f} wall={wall_mid:.1f} ema={ema:.2f} slow={ema_slow:.2f} "
            f"obi={obi:.2f} sig={trend_signal:.2f} fcst={forecast:.2f} "
            f"eff={eff_fair:.2f} pb={passive_buy} ps={passive_sell} pos={position}"
        )
        return orders
