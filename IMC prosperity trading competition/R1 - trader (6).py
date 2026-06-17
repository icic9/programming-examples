import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Order,
    OrderDepth,
    ProsperityEncoder,
    TradingState,
)


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

    # ── PEPPER: keep close to prior strong version ─────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_SKEW = 0.015
    PEPPER_SPREAD_HIGH = 1
    PEPPER_VOL_THRESH = 4.0
    PEPPER_L1_FRAC = 0.80
    PEPPER_L2_FRAC = 0.20
    PEPPER_EMV_ALPHA = 0.15
    PEPPER_MOMENTUM_BIAS = 0
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 3.0
    PEPPER_TREND_CARRY = 0.55
    PEPPER_TREND_GATE = 0.0
    PEPPER_AGAINST_TREND_CAP = 0.55
    PEPPER_CARRY_EDGE = 0.0
    PEPPER_LONG_FLOOR = 65
    PEPPER_MM_VOLUME = 18

    # ── OSMIUM: simplified, low-parameter MM ───────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_EMA_ALPHA = 0.35        # fair estimator
    OSMIUM_INV_SKEW = 6.0          # max reservation shift at full inventory ~6 ticks
    OSMIUM_WIDTH_BASE = 1.0        # baseline half-width
    OSMIUM_WIDTH_SPREAD_MULT = 0.20
    OSMIUM_TAKE_EDGE = 1.0
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
    def _penny_passive_prices(
        od: OrderDepth,
        ref: float,
        fair: float,
        min_buy_edge: int,
        min_sell_edge: int,
    ):
        comp_bids = [p for p in od.buy_orders if p < fair]
        comp_asks = [p for p in od.sell_orders if p > fair]

        penny_buy = (
            (max(comp_bids) + 1) if comp_bids else (int(ref) - min_buy_edge - 1)
        )
        penny_sell = (
            (min(comp_asks) - 1) if comp_asks else (int(ref) + min_sell_edge + 1)
        )

        passive_buy = min(penny_buy, int(ref) - min_buy_edge)
        passive_sell = max(penny_sell, int(ref) + min_sell_edge)

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

        def split_qty(total: int):
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
            if l1 == 0 and total > 0:
                l1 = 1
                if l2 > 0:
                    l2 -= 1
            return l1, l2

        if remaining_buy > 0:
            l1, l2 = split_qty(remaining_buy)
            if l1 > 0:
                orders.append(Order(symbol, passive_buy, l1))
            if l2 > 0:
                orders.append(Order(symbol, passive_buy - 1, l2))

        if remaining_sell > 0:
            l1, l2 = split_qty(remaining_sell)
            if l1 > 0:
                orders.append(Order(symbol, passive_sell, -l1))
            if l2 > 0:
                orders.append(Order(symbol, passive_sell + 1, -l2))

        return orders

    def run(self, state: TradingState):
        result = {}

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}

        saved.setdefault("pepper_ema", None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv", None)
        saved.setdefault("osmium_ema", None)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == self.PEPPER:
                result[product] = self._trade_pepper(od, position, saved)
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(od, position, saved)

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _trade_osmium(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        orders: List[Order] = []
        limit = self.OSMIUM_LIMIT

        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = float(best_bid)
            spread = 2
        else:
            best_ask = min(od.sell_orders)
            mid = float(best_ask)
            spread = 2

        # 1) fair value
        a = self.OSMIUM_EMA_ALPHA
        if saved["osmium_ema"] is None:
            saved["osmium_ema"] = mid
        else:
            saved["osmium_ema"] = a * mid + (1.0 - a) * saved["osmium_ema"]
        fair = saved["osmium_ema"]

        # 2) reservation price: one simple inventory control
        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

        # 3) adaptive width from live spread only
        half_width = max(
            1,
            round(self.OSMIUM_WIDTH_BASE + self.OSMIUM_WIDTH_SPREAD_MULT * spread),
        )

        # 4) aggressive only on obvious mispricings vs reservation
        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= reservation - self.OSMIUM_TAKE_EDGE:
                    break
                can_buy = limit - position
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
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        # recompute after fills
        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

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

        # simple inventory-aware size throttling
        buy_scale = max(0.20, 1.0 - max(0.0, position / limit))
        sell_scale = max(0.20, 1.0 - max(0.0, -position / limit))

        remaining_buy = int(round((limit - position) * buy_scale))
        remaining_sell = int(round((limit + position) * sell_scale))

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
            f"OSM | mid={mid:.1f} fair={fair:.2f} res={reservation:.2f} "
            f"spread={spread} hw={half_width} pbuy={passive_buy} "
            f"psell={passive_sell} pos={position}"
        )

        return orders

    def _trade_pepper(self, od: OrderDepth, position: int, saved: dict) -> List[Order]:
        orders: List[Order] = []
        limit = self.PEPPER_LIMIT

        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        if has_bid and has_ask:
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            bv = od.buy_orders[best_bid]
            av = -od.sell_orders[best_ask]
            mid = (best_bid + best_ask) / 2.0
            microprice = (
                (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) > 0 else mid
            )
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)

        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)

        a_fast = self.PEPPER_EMA_ALPHA
        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            saved["pepper_ema"] = (
                a_fast * ref_price + (1.0 - a_fast) * saved["pepper_ema"]
            )
        ema = saved["pepper_ema"]

        a_slow = self.PEPPER_EMA_SLOW_ALPHA
        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            saved["pepper_ema_slow"] = (
                a_slow * ref_price + (1.0 - a_slow) * saved["pepper_ema_slow"]
            )
        ema_slow = saved["pepper_ema_slow"]

        trend_signal = ema - ema_slow
        forecast_ema = ema + self.PEPPER_TREND_CARRY * trend_signal

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

        carry_edge = self.PEPPER_CARRY_EDGE if trend_signal >= -0.5 else 0.0
        buy_threshold = forecast_ema + carry_edge
        sell_threshold = forecast_ema + carry_edge

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

        biased_pos = position - self.PEPPER_MOMENTUM_BIAS * trend_signal
        biased_pos = max(-limit, min(limit, biased_pos))

        skew = self.PEPPER_SKEW * biased_pos
        if abs(biased_pos) > self.PEPPER_URGENCY_THRESH * limit:
            skew *= self.PEPPER_URGENCY_MULT

        eff_ema = forecast_ema - skew

        min_edge = 1 + spread_extra
        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=eff_ema,
            fair=forecast_ema,
            min_buy_edge=min_edge,
            min_sell_edge=min_edge,
        )

        remaining_buy = limit - position
        remaining_sell = limit + position

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
            f"PEP | mid={mid:.1f} mp={microprice:.1f} ema={ema:.2f} "
            f"slow={ema_slow:.2f} sig={trend_signal:.2f} fcst={forecast_ema:.2f} "
            f"bpos={biased_pos:.1f} skew={skew:.2f} eff={eff_ema:.2f} "
            f"vol={vol:.2f} pbuy={passive_buy} psell={passive_sell} pos={position}"
        )

        return orders
