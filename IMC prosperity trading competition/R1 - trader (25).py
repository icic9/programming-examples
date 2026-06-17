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
    v25: robust protected PEPPER carry core + OSMIUM wall/regime maker

    Design principles:
      1) PEPPER is split into a strategic carry core and a spread-trading
         sleeve. The bot should not accidentally liquidate the carry core just
         because a market-making signal says sell.
      2) Inventory is managed by purpose: build core, recycle sleeve, clear
         risk. This is different from a single symmetric reservation price.
      3) OSMIUM fair uses a deep-liquidity wall mid, inspired by
         Frankfurt Hedgehogs' WallMid idea: ignore small noisy participants.
      4) Conservative anonymous-extrema detection tilts OSMIUM inventory only when
         repeated trade flow appears at local extremes.
      5) Take-clear-make execution frees risk capacity before posting quotes.
      6) PEPPER carry is the default prior, but a hysteresis brake disables it
         if the long-horizon clock trend is materially broken.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    # Lower than v14: PEPPER has a persistent upward carry, so inventory
    # should not push our reservation price away from the long side too early.
    PEPPER_SKEW = 0.008
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
    PEPPER_LONG_FLOOR = 60
    PEPPER_MM_VOLUME = 18
    PEPPER_FLOW_ALPHA = 0.25
    PEPPER_FLOW_WEIGHT = 1.15
    PEPPER_CLOCK_DRIFT_PER_STEP = 0.10
    PEPPER_CLOCK_DRIFT_WEIGHT = 0.40
    # Structural carry split:
    # - 68 units are treated as "do not recycle" carry inventory.
    # - the remaining 12 units are a trading sleeve for spread capture.
    # - buy the sleeve only on dips vs forecast; sell it only on premium bids.
    PEPPER_CORE_TARGET = 68
    PEPPER_REFILL_TARGET = 80
    PEPPER_CORE_BID_EDGE = -1
    PEPPER_SLEEVE_SELL_EDGE = 2
    PEPPER_CARRY_BREAK_TREND = -4.0
    PEPPER_CARRY_BREAK_CLOCK_GAP = -25.0
    PEPPER_CARRY_BREAK_CONFIRM = 3

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    # Slower than v14: wall-mid changes are noisy; the live book is used for
    # execution, while the fair estimate should avoid chasing short spikes.
    OSMIUM_EMA_ALPHA = 0.12

    # Toxicity model:
    # predict short-horizon adverse selection from imbalance and microprice.
    # The prediction controls size, not fair value, which is more robust.
    OSMIUM_TOX_OBI_WEIGHT = 0.65
    OSMIUM_TOX_MICRO_WEIGHT = 0.35
    OSMIUM_TOX_SIDE_MULT = 0.60
    OSMIUM_L1_FRAC = 0.80
    OSMIUM_L2_FRAC = 0.20

    # Reservation-price inventory penalty
    # Lower than v23: OSMIUM is range-bound, so excessive reservation-price
    # skew gives away profitable mean-reversion inventory too quickly.
    OSMIUM_INV_SKEW = 2.5

    # Width from live spread only
    OSMIUM_BASE_WIDTH = 1.0
    OSMIUM_SPREAD_WIDTH = 0.0

    # Inventory regimes
    OSMIUM_NEUTRAL_POS = 80
    OSMIUM_LOADED_POS = 80

    # Side adjustments
    OSMIUM_LOADED_EXIT_NARROW = 1
    OSMIUM_LOADED_WRONG_WIDEN = 1
    OSMIUM_STRESSED_EXIT_NARROW = 2
    OSMIUM_STRESSED_WRONG_WIDEN = 3

    # Taking thresholds
    OSMIUM_NEUTRAL_TAKE = 0
    OSMIUM_EXIT_TAKE = 1
    OSMIUM_STRESSED_EXIT_TAKE = 0

    # Passive size suppression
    OSMIUM_LOADED_WRONG_MULT = 0.30
    OSMIUM_STRESSED_WRONG_MULT = 0.05
    # Wider clearing is intentional: OSMIUM is range-bound/noisy, so freeing
    # inventory near fair has more value than waiting for a perfect exit.
    OSMIUM_CLEAR_WIDTH = 5
    OSMIUM_ADVERSE_VOLUME = 24
    OSMIUM_REQUIRE_TWO_SIDED_TAKE = True

    @staticmethod
    def _book_imbalance(od: OrderDepth) -> float:
        bid_volume = sum(max(0, v) for v in od.buy_orders.values())
        ask_volume = sum(max(0, -v) for v in od.sell_orders.values())
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0

    @staticmethod
    def _microprice(od: OrderDepth, mid: float, best_bid: int, best_ask: int) -> float:
        if best_bid in od.buy_orders and best_ask in od.sell_orders:
            bid_volume = od.buy_orders[best_bid]
            ask_volume = -od.sell_orders[best_ask]
            if bid_volume + ask_volume > 0:
                return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        return mid

    @staticmethod
    def _wall_mid(od: OrderDepth, fallback: float) -> float:
        if od.buy_orders and od.sell_orders:
            return (min(od.buy_orders) + max(od.sell_orders)) / 2.0
        return fallback

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

        l1_buy = int(round(remaining_buy * l1_frac))
        l2_buy = remaining_buy - l1_buy
        l1_sell = int(round(remaining_sell * l1_frac))
        l2_sell = remaining_sell - l1_sell

        if l1_buy > 0:
            orders.append(Order(symbol, passive_buy, l1_buy))
        if l2_buy > 0 and l2_frac > 0:
            orders.append(Order(symbol, passive_buy - 1, l2_buy))

        if l1_sell > 0:
            orders.append(Order(symbol, passive_sell, -l1_sell))
        if l2_sell > 0 and l2_frac > 0:
            orders.append(Order(symbol, passive_sell + 1, -l2_sell))

        return orders

    def run(self, state: TradingState):
        result = {}

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        saved.setdefault("pepper_ema", None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv", None)
        saved.setdefault("pepper_flow", 0.0)
        saved.setdefault("pepper_clock_origin", None)
        saved.setdefault("pepper_last_timestamp", None)
        saved.setdefault("pepper_carry_break_count", 0)
        saved.setdefault("osmium_ema", None)
        saved.setdefault("osmium_min", None)
        saved.setdefault("osmium_max", None)
        saved.setdefault("osmium_extreme_signal", 0.0)
        saved.setdefault("osmium_last_wall", None)
        saved.setdefault("osmium_diff_var", None)
        saved.setdefault("osmium_spike_signal", 0.0)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == self.PEPPER:
                prior_trades = state.market_trades.get(product, [])
                result[product] = self._trade_pepper(
                    od, position, prior_trades, state.timestamp, saved
                )
            elif product == self.OSMIUM:
                prior_trades = state.market_trades.get(product, [])
                result[product] = self._trade_osmium(
                    od, position, prior_trades, saved
                )

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── OSMIUM ──────────────────────────────────────────────────────────────

    def _trade_osmium(
        self,
        od: OrderDepth,
        position: int,
        prior_trades: List[Trade],
        saved: dict,
    ) -> List[Order]:
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

        microprice = mid
        if has_bid and has_ask:
            microprice = self._microprice(od, mid, best_bid, best_ask)
        wall_mid = self._wall_mid(od, mid)

        if saved["osmium_min"] is None:
            saved["osmium_min"] = wall_mid
            saved["osmium_max"] = wall_mid
        saved["osmium_min"] = min(saved["osmium_min"], wall_mid)
        saved["osmium_max"] = max(saved["osmium_max"], wall_mid)

        extreme_signal = 0.0
        for trade in prior_trades:
            if trade.quantity < 12:
                continue
            if trade.price <= saved["osmium_min"] + 1:
                extreme_signal += 1.0
            elif trade.price >= saved["osmium_max"] - 1:
                extreme_signal -= 1.0
        saved["osmium_extreme_signal"] = (
            0.90 * saved["osmium_extreme_signal"] + 0.10 * extreme_signal
        )

        fair_observation = 0.80 * wall_mid + 0.20 * microprice

        if saved["osmium_last_wall"] is not None:
            wall_diff = wall_mid - saved["osmium_last_wall"]
            diff_sq = wall_diff * wall_diff
            if saved["osmium_diff_var"] is None:
                saved["osmium_diff_var"] = diff_sq
            else:
                saved["osmium_diff_var"] = (
                    0.10 * diff_sq + 0.90 * saved["osmium_diff_var"]
                )
            vol = max(1.0, saved["osmium_diff_var"] ** 0.5)
            if abs(wall_diff) > 2.5 * vol:
                saved["osmium_spike_signal"] = -1.0 if wall_diff > 0 else 1.0
            else:
                saved["osmium_spike_signal"] *= 0.85
        saved["osmium_last_wall"] = wall_mid

        # WallMid suppresses small noisy orders; the small microprice blend
        # keeps the model responsive when the walls are stale.
        a = self.OSMIUM_EMA_ALPHA
        if saved["osmium_ema"] is None:
            saved["osmium_ema"] = fair_observation
        else:
            saved["osmium_ema"] = a * fair_observation + (1.0 - a) * saved["osmium_ema"]
        ema = saved["osmium_ema"]

        fair = (
            ema
            + 0.60 * saved["osmium_extreme_signal"]
            + 0.40 * saved["osmium_spike_signal"]
        )

        # Inventory-adjusted reservation price
        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

        half_spread = max(1.0, spread / 2.0)
        micro_signal = max(-1.0, min(1.0, (microprice - mid) / half_spread))
        toxicity = (
            self.OSMIUM_TOX_OBI_WEIGHT * self._book_imbalance(od)
            + self.OSMIUM_TOX_MICRO_WEIGHT * micro_signal
        )

        base_half_width = max(
            1,
            round(self.OSMIUM_BASE_WIDTH + self.OSMIUM_SPREAD_WIDTH * spread),
        )

        abs_pos = abs(position)
        if abs_pos <= self.OSMIUM_NEUTRAL_POS:
            buy_edge = sell_edge = base_half_width
            buy_take = sell_take = self.OSMIUM_NEUTRAL_TAKE
            buy_mult = sell_mult = 1.0
            regime = "N"
        elif abs_pos <= self.OSMIUM_LOADED_POS:
            regime = "L"
            if position > 0:
                buy_edge = base_half_width + self.OSMIUM_LOADED_WRONG_WIDEN
                sell_edge = max(1, base_half_width - self.OSMIUM_LOADED_EXIT_NARROW)
                buy_take = self.OSMIUM_NEUTRAL_TAKE + 2
                sell_take = self.OSMIUM_EXIT_TAKE
                buy_mult = self.OSMIUM_LOADED_WRONG_MULT
                sell_mult = 1.0
            else:
                sell_edge = base_half_width + self.OSMIUM_LOADED_WRONG_WIDEN
                buy_edge = max(1, base_half_width - self.OSMIUM_LOADED_EXIT_NARROW)
                sell_take = self.OSMIUM_NEUTRAL_TAKE + 2
                buy_take = self.OSMIUM_EXIT_TAKE
                sell_mult = self.OSMIUM_LOADED_WRONG_MULT
                buy_mult = 1.0
        else:
            regime = "S"
            if position > 0:
                buy_edge = base_half_width + self.OSMIUM_STRESSED_WRONG_WIDEN
                sell_edge = max(1, base_half_width - self.OSMIUM_STRESSED_EXIT_NARROW)
                buy_take = self.OSMIUM_NEUTRAL_TAKE + 3
                sell_take = self.OSMIUM_STRESSED_EXIT_TAKE
                buy_mult = self.OSMIUM_STRESSED_WRONG_MULT
                sell_mult = 1.0
            else:
                sell_edge = base_half_width + self.OSMIUM_STRESSED_WRONG_WIDEN
                buy_edge = max(1, base_half_width - self.OSMIUM_STRESSED_EXIT_NARROW)
                sell_take = self.OSMIUM_NEUTRAL_TAKE + 3
                buy_take = self.OSMIUM_STRESSED_EXIT_TAKE
                sell_mult = self.OSMIUM_STRESSED_WRONG_MULT
                buy_mult = 1.0

        # Aggressive phase:
        # neutral -> only obvious gifts
        # loaded/stressed -> much easier to cross on the exit side
        can_take_ask = has_ask and (
            has_bid or not self.OSMIUM_REQUIRE_TWO_SIDED_TAKE
        )
        if can_take_ask:
            for ask in sorted(od.sell_orders):
                if ask >= reservation - buy_take:
                    break
                available = -od.sell_orders[ask]
                if available > self.OSMIUM_ADVERSE_VOLUME and position >= 0:
                    continue
                can_buy = limit - position
                if can_buy <= 0:
                    break
                qty = min(available, can_buy)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, ask, qty))
                    position += qty

        can_take_bid = has_bid and (
            has_ask or not self.OSMIUM_REQUIRE_TWO_SIDED_TAKE
        )
        if can_take_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid <= reservation + sell_take:
                    break
                available = od.buy_orders[bid]
                if available > self.OSMIUM_ADVERSE_VOLUME and position <= 0:
                    continue
                can_sell = limit + position
                if can_sell <= 0:
                    break
                qty = min(available, can_sell)
                if qty > 0:
                    orders.append(Order(self.OSMIUM, bid, -qty))
                    position -= qty

        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

        # Borrowed from Linear Utility's take-clear-make structure:
        # if inventory is leaning one way, use visible liquidity at fair to
        # flatten part of the book before posting fresh passive quotes.
        fair_bid = round(fair - self.OSMIUM_CLEAR_WIDTH)
        fair_ask = round(fair + self.OSMIUM_CLEAR_WIDTH)
        if position > 0 and has_bid:
            clearable = sum(volume for price, volume in od.buy_orders.items() if price >= fair_ask)
            qty = min(position, clearable, limit + position)
            if qty > 0:
                orders.append(Order(self.OSMIUM, fair_ask, -qty))
                position -= qty
        elif position < 0 and has_ask:
            clearable = sum(-volume for price, volume in od.sell_orders.items() if price <= fair_bid)
            qty = min(-position, clearable, limit - position)
            if qty > 0:
                orders.append(Order(self.OSMIUM, fair_bid, qty))
                position += qty

        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

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

        buy_inventory_scale = max(0.20, 1.0 - max(0.0, position / limit))
        sell_inventory_scale = max(0.20, 1.0 - max(0.0, -position / limit))

        remaining_buy = max(
            0,
            int(round((limit - position) * buy_mult * buy_inventory_scale)),
        )
        remaining_sell = max(
            0,
            int(round((limit + position) * sell_mult * sell_inventory_scale)),
        )

        # If the short-horizon model says the next move is up, selling passive
        # is toxic. If it says down, buying passive is toxic. This keeps the
        # fair value stable while removing the side most likely to be picked off.
        if toxicity > 0:
            remaining_sell = int(round(remaining_sell * self.OSMIUM_TOX_SIDE_MULT))
        elif toxicity < 0:
            remaining_buy = int(round(remaining_buy * self.OSMIUM_TOX_SIDE_MULT))

        # Two-level quoting: first level keeps queue priority, second level
        # collects strict market-trade crosses when the book moves through us.
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
            f"OSM| mid={mid:.1f} wall={wall_mid:.1f} mp={microprice:.1f} "
            f"ema={ema:.2f} fair={fair:.2f} ext={saved['osmium_extreme_signal']:.2f} "
            f"spk={saved['osmium_spike_signal']:.2f} "
            f"res={reservation:.2f} spr={spread} reg={regime} "
            f"tox={toxicity:.2f} "
            f"be={buy_edge} se={sell_edge} bt={buy_take} st={sell_take} "
            f"pb={passive_buy} ps={passive_sell} pos={position}"
        )

        return orders

    # ── PEPPER ──────────────────────────────────────────────────────────────

    def _trade_pepper(
        self,
        od: OrderDepth,
        position: int,
        prior_trades: List[Trade],
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
            microprice = mid
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)

        # Defensive reset for environments that preserve traderData across a
        # day/session timestamp reset. It is inactive in the normal local runs.
        if saved["pepper_last_timestamp"] is not None and timestamp < saved["pepper_last_timestamp"]:
            saved["pepper_ema"] = None
            saved["pepper_ema_slow"] = None
            saved["pepper_last_mid"] = None
            saved["pepper_emv"] = None
            saved["pepper_flow"] = 0.0
            saved["pepper_clock_origin"] = None
            saved["pepper_carry_break_count"] = 0
        saved["pepper_last_timestamp"] = timestamp

        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)
        if saved["pepper_clock_origin"] is None:
            saved["pepper_clock_origin"] = ref_price

        if saved["pepper_last_mid"] is not None and prior_trades:
            signed_qty = 0
            total_qty = 0
            ref_mid = saved["pepper_last_mid"]
            for trade in prior_trades:
                total_qty += trade.quantity
                if trade.price > ref_mid:
                    signed_qty += trade.quantity
                elif trade.price < ref_mid:
                    signed_qty -= trade.quantity
            if total_qty > 0:
                flow_obs = signed_qty / total_qty
                saved["pepper_flow"] = (
                    self.PEPPER_FLOW_ALPHA * flow_obs
                    + (1.0 - self.PEPPER_FLOW_ALPHA) * saved["pepper_flow"]
                )

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
        clock_fair = (
            saved["pepper_clock_origin"]
            + self.PEPPER_CLOCK_DRIFT_PER_STEP * (timestamp / 100.0)
        )
        forecast_ema = (
            ema
            + self.PEPPER_TREND_CARRY * trend_signal
            + self.PEPPER_FLOW_WEIGHT * saved["pepper_flow"]
            + self.PEPPER_CLOCK_DRIFT_WEIGHT * (clock_fair - ema)
        )

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

        # PEPPER is a carry product first and a market-making product second.
        # Keep a strategic core long. Only inventory above that core is allowed
        # to be recycled for spread capture unless the trend regime breaks.
        clock_gap = ref_price - clock_fair
        if (
            trend_signal < self.PEPPER_CARRY_BREAK_TREND
            and clock_gap < self.PEPPER_CARRY_BREAK_CLOCK_GAP
        ):
            saved["pepper_carry_break_count"] += 1
        else:
            saved["pepper_carry_break_count"] = max(
                0, saved["pepper_carry_break_count"] - 1
            )

        carry_broken = (
            saved["pepper_carry_break_count"] >= self.PEPPER_CARRY_BREAK_CONFIRM
        )
        strong_reversal = carry_broken or trend_signal < -2.0

        if has_ask and not strong_reversal:
            for ask in sorted(od.sell_orders):
                can_buy = limit - position
                if can_buy <= 0:
                    break

                building_core = position < self.PEPPER_CORE_TARGET
                sleeve_edge = ask <= forecast_ema + self.PEPPER_CORE_BID_EDGE
                if not building_core and not sleeve_edge:
                    break

                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        if has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                if strong_reversal:
                    can_sell = limit + position
                else:
                    if bid <= forecast_ema + self.PEPPER_SLEEVE_SELL_EDGE:
                        break
                    can_sell = max(0, position - self.PEPPER_CORE_TARGET)
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

        if strong_reversal:
            remaining_buy = int(round(remaining_buy * self.PEPPER_AGAINST_TREND_CAP))
        else:
            # Build and maintain the core. The trading sleeve above the core is
            # the only inventory that can be passively offered.
            if position < self.PEPPER_CORE_TARGET:
                remaining_buy = limit - position
            elif position < self.PEPPER_REFILL_TARGET:
                remaining_buy = max(0, self.PEPPER_REFILL_TARGET - position)
            else:
                remaining_buy = max(0, limit - position)
            remaining_sell = max(0, position - self.PEPPER_CORE_TARGET)

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
            f"PEP| mid={mid:.1f} mp={microprice:.1f} ema={ema:.2f} "
            f"slow={ema_slow:.2f} sig={trend_signal:.2f} fcst={forecast_ema:.2f} "
            f"clk_gap={clock_gap:.1f} brk={saved['pepper_carry_break_count']} "
            f"flow={saved['pepper_flow']:.2f} eff={eff_ema:.2f} "
            f"vol={vol:.2f} pb={passive_buy} ps={passive_sell} pos={position}"
        )

        return orders
