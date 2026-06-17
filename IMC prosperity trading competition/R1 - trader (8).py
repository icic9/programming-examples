
import json
from typing import Any, List

import jsonpickle

from datamodel import (
    Observation, Order, OrderDepth,
    ProsperityEncoder, Trade, TradingState,
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
    Round 1 robust trader.

    Design goals:
      - no hardcoded absolute price anchors
      - no dependence on 10,000 or any fixed level
      - low-parameter, regime-based inventory control
      - all logic expressed relative to live book / EMA / spread

    PEPPER:
      - keep the strong trend-carry architecture

    OSMIUM:
      - adaptive EMA fair only
      - level-free inventory regimes
      - spread-relative quote width
      - asymmetric taking and sizing based on inventory
      - optional mean-reversion term uses EMA gap only (relative, not anchored)
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_SKEW = 0.015
    PEPPER_SPREAD_HIGH = 1
    PEPPER_VOL_THRESH = 4.0
    PEPPER_L1_FRAC = 1.0
    PEPPER_L2_FRAC = 0.0
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

    # ── OSMIUM: robust / level-free ────────────────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_EMA_ALPHA = 0.40

    # Reservation shift from inventory only, in ticks
    OSMIUM_INV_SKEW = 5.0

    # Regimes based only on inventory magnitude
    OSMIUM_NEUTRAL_POS = 25
    OSMIUM_EMERGENCY_POS = 60

    # Width from live spread only
    OSMIUM_BASE_WIDTH = 1.0
    OSMIUM_SPREAD_WIDTH = 0.18

    # Relative mean-reversion term: if mid deviates from EMA, lean back slightly
    # This is level-free because it only uses (mid - ema)
    OSMIUM_REVERT_WEIGHT = 0.35

    # Inventory-side width modifications
    OSMIUM_LOADED_EXIT_NARROW = 1
    OSMIUM_LOADED_WRONG_WIDEN = 1
    OSMIUM_EMERG_EXIT_NARROW = 2
    OSMIUM_EMERG_WRONG_WIDEN = 3

    # Aggressive thresholds
    OSMIUM_TAKE_NEUTRAL = 2
    OSMIUM_TAKE_LOADED = 1
    OSMIUM_TAKE_EMERG = 0

    # Passive size shaping
    OSMIUM_LOADED_WRONG_MULT = 0.35
    OSMIUM_EMERG_WRONG_MULT = 0.05
    OSMIUM_EXIT_MULT = 1.0

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
        passive_buy: int, passive_sell: int,
        remaining_buy: int, remaining_sell: int,
        l1_frac: float, l2_frac: float,
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

    @staticmethod
    def _single_level_orders(
        symbol: str,
        buy_price: int,
        sell_price: int,
        buy_qty: int,
        sell_qty: int,
    ) -> List[Order]:
        orders: List[Order] = []
        if buy_qty > 0:
            orders.append(Order(symbol, buy_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(symbol, sell_price, -sell_qty))
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
            spread = 16
        else:
            best_ask = min(od.sell_orders)
            mid = float(best_ask)
            spread = 16

        # Adaptive fair with no hardcoded level
        a = self.OSMIUM_EMA_ALPHA
        if saved["osmium_ema"] is None:
            saved["osmium_ema"] = mid
        else:
            saved["osmium_ema"] = a * mid + (1.0 - a) * saved["osmium_ema"]
        ema = saved["osmium_ema"]

        # Relative mean-reversion: if market stretches away from EMA, lean back slightly
        revert_term = -self.OSMIUM_REVERT_WEIGHT * (mid - ema)

        # Inventory-adjusted reservation price, fully level-free
        reservation = ema + revert_term - self.OSMIUM_INV_SKEW * (position / limit)

        abs_pos = abs(position)
        if abs_pos <= self.OSMIUM_NEUTRAL_POS:
            regime = "neutral"
        elif abs_pos <= self.OSMIUM_EMERGENCY_POS:
            regime = "loaded"
        else:
            regime = "emergency"

        base_half_width = max(
            1,
            round(self.OSMIUM_BASE_WIDTH + self.OSMIUM_SPREAD_WIDTH * spread),
        )

        if position > self.OSMIUM_NEUTRAL_POS:
            if regime == "loaded":
                buy_edge = base_half_width + self.OSMIUM_LOADED_WRONG_WIDEN
                sell_edge = max(1, base_half_width - self.OSMIUM_LOADED_EXIT_NARROW)
                buy_take = self.OSMIUM_TAKE_LOADED + 2
                sell_take = max(0, self.OSMIUM_TAKE_LOADED - 1)
                buy_mult = self.OSMIUM_LOADED_WRONG_MULT
                sell_mult = self.OSMIUM_EXIT_MULT
            else:
                buy_edge = base_half_width + self.OSMIUM_EMERG_WRONG_WIDEN
                sell_edge = max(1, base_half_width - self.OSMIUM_EMERG_EXIT_NARROW)
                buy_take = self.OSMIUM_TAKE_EMERG + 3
                sell_take = self.OSMIUM_TAKE_EMERG
                buy_mult = self.OSMIUM_EMERG_WRONG_MULT
                sell_mult = self.OSMIUM_EXIT_MULT
        elif position < -self.OSMIUM_NEUTRAL_POS:
            if regime == "loaded":
                sell_edge = base_half_width + self.OSMIUM_LOADED_WRONG_WIDEN
                buy_edge = max(1, base_half_width - self.OSMIUM_LOADED_EXIT_NARROW)
                sell_take = self.OSMIUM_TAKE_LOADED + 2
                buy_take = max(0, self.OSMIUM_TAKE_LOADED - 1)
                sell_mult = self.OSMIUM_LOADED_WRONG_MULT
                buy_mult = self.OSMIUM_EXIT_MULT
            else:
                sell_edge = base_half_width + self.OSMIUM_EMERG_WRONG_WIDEN
                buy_edge = max(1, base_half_width - self.OSMIUM_EMERG_EXIT_NARROW)
                sell_take = self.OSMIUM_TAKE_EMERG + 3
                buy_take = self.OSMIUM_TAKE_EMERG
                sell_mult = self.OSMIUM_EMERG_WRONG_MULT
                buy_mult = self.OSMIUM_EXIT_MULT
        else:
            buy_edge = sell_edge = base_half_width
            buy_take = sell_take = self.OSMIUM_TAKE_NEUTRAL
            buy_mult = sell_mult = 1.0

        # Aggressive only when clearly favorable relative to reservation
        if has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= reservation - buy_take:
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
                if bid <= reservation + sell_take:
                    break
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        reservation = ema + revert_term - self.OSMIUM_INV_SKEW * (position / limit)

        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=reservation,
            fair=reservation,
            min_buy_edge=buy_edge,
            min_sell_edge=sell_edge,
        )

        if passive_buy >= passive_sell:
            passive_buy = int(reservation) - buy_edge
            passive_sell = int(reservation) + sell_edge

        remaining_buy = max(0, int(round((limit - position) * buy_mult)))
        remaining_sell = max(0, int(round((limit + position) * sell_mult)))

        # Single-level OSMIUM quotes for max queue priority
        orders += self._single_level_orders(
            self.OSMIUM,
            passive_buy,
            passive_sell,
            remaining_buy,
            remaining_sell,
        )

        logger.print(
            f"OSM | mid={mid:.1f} ema={ema:.2f} rev={revert_term:.2f} "
            f"res={reservation:.2f} reg={regime} "
            f"buyE={buy_edge} sellE={sell_edge} "
            f"pbuy={passive_buy} psell={passive_sell} pos={position}"
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
