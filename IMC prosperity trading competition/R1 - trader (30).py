
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
    v28: protected carry with confirmed 3-state PEPPER guard + learned OSMIUM range anchor.

    Design principles:
      - PEPPER carry remains the default because exposure is the alpha source.
        Flat/short regimes exist, but require multi-signal confirmation.
      - PEPPER signals are separated by role: EMA/clock for fair, flow for
        execution pressure, OLS slope as a diagnostic/brake input.
      - OSMIUM fair blends wall-mid, microprice, and a slow learned range
        anchor instead of a fixed price level.
      - Toxicity is contextual: less blunt near flat/wide spread, harsher when loaded.
    """

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # ── PEPPER ──────────────────────────────────────────────────────────────
    PEPPER_LIMIT = 80
    PEPPER_EMA_ALPHA = 0.28
    PEPPER_EMA_SLOW_ALPHA = 0.015
    PEPPER_EMV_ALPHA = 0.15
    PEPPER_MM_VOLUME = 18

    PEPPER_FLOW_ALPHA = 0.25
    PEPPER_FLOW_WEIGHT = 1.15
    PEPPER_TREND_CARRY = 0.55
    PEPPER_SLOPE_WINDOW = 12
    # Keep the slope as a diagnostic, not a direct fair-value term. The
    # short-window slope is noisy; putting it directly into price made v26
    # robust-looking but cost carry PnL.
    PEPPER_SLOPE_WEIGHT = 0.0

    PEPPER_SKEW = 0.008
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 3.0
    PEPPER_VOL_THRESH = 4.0
    PEPPER_SPREAD_HIGH = 1

    # Carry-prior sleeve. This is not a hard price level: it is a structural
    # inventory policy that stays long only while live signals do not veto it.
    PEPPER_MOMENTUM_BIAS = 0
    PEPPER_TREND_GATE = 0.0
    PEPPER_AGAINST_TREND_CAP = 0.55
    PEPPER_CARRY_EDGE = 0.0
    PEPPER_LONG_FLOOR = 60
    PEPPER_CLOCK_DRIFT_PER_STEP = 0.10
    PEPPER_CLOCK_DRIFT_WEIGHT = 0.40
    PEPPER_DRIFT_ALPHA = 0.015
    PEPPER_DRIFT_MIN = -0.05
    PEPPER_DRIFT_MAX = 0.18
    PEPPER_CORE_TARGET = 64
    PEPPER_REFILL_TARGET = 80
    PEPPER_CORE_BID_EDGE = -1
    PEPPER_SLEEVE_SELL_EDGE = 2
    PEPPER_CARRY_BREAK_TREND = -4.0
    PEPPER_CARRY_BREAK_CLOCK_GAP = -25.0
    PEPPER_CARRY_BREAK_CONFIRM = 3
    PEPPER_FLAT_CONFIRM = 5
    PEPPER_SHORT_CONFIRM = 7
    PEPPER_RECOVER_CONFIRM = 2
    PEPPER_L1_FRAC = 1.0
    PEPPER_L2_FRAC = 0.0
    PEPPER_CORE_TAKE_LEVELS = 1
    PEPPER_SLEEVE_SELL_LEVELS = 1

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_EMA_ALPHA = 0.12
    OSMIUM_INV_SKEW = 2.5

    OSMIUM_TOX_OBI_WEIGHT = 0.65
    OSMIUM_TOX_MICRO_WEIGHT = 0.35
    OSMIUM_TOX_SIDE_MULT = 0.60
    OSMIUM_L1_FRAC = 0.80
    OSMIUM_L2_FRAC = 0.20

    OSMIUM_BASE_WIDTH = 1
    OSMIUM_NEUTRAL_POS = 80
    OSMIUM_LOADED_POS = 80
    OSMIUM_NEUTRAL_TAKE = 0
    OSMIUM_EXIT_TAKE = 1
    OSMIUM_STRESSED_EXIT_TAKE = 0
    OSMIUM_LOADED_WRONG_MULT = 0.30
    OSMIUM_STRESSED_WRONG_MULT = 0.05
    OSMIUM_CLEAR_WIDTH = 5
    OSMIUM_ADVERSE_VOLUME = 24
    OSMIUM_REQUIRE_TWO_SIDED_TAKE = True
    OSMIUM_FILL_HORIZON = 100000
    OSMIUM_FILL_QUALITY_ALPHA = 0.025
    OSMIUM_FILL_BIAS_CAP = 2.0
    OSMIUM_FILL_FAIR_WEIGHT = 0.03

    @staticmethod
    def _book_imbalance(od: OrderDepth) -> float:
        bid_volume = sum(max(0, v) for v in od.buy_orders.values())
        ask_volume = sum(max(0, -v) for v in od.sell_orders.values())
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0

    @staticmethod
    def _microprice(od: OrderDepth, mid: float, best_bid: int, best_ask: int) -> float:
        bid_volume = od.buy_orders.get(best_bid, 0)
        ask_volume = -od.sell_orders.get(best_ask, 0)
        if bid_volume + ask_volume > 0:
            return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        return mid

    @staticmethod
    def _large_quote_mid(od: OrderDepth, fallback: float, min_volume: int) -> float:
        large_bids = [p for p, v in od.buy_orders.items() if abs(v) >= min_volume]
        large_asks = [p for p, v in od.sell_orders.items() if abs(v) >= min_volume]
        if large_bids and large_asks:
            return (max(large_bids) + min(large_asks)) / 2.0
        return fallback

    @staticmethod
    def _wall_mid(od: OrderDepth, fallback: float) -> float:
        """
        Robust institutional wall estimate.

        For this book the durable anchor is better represented by the deep
        outer walls than the largest single displayed size. A large displayed
        clip can be a transient queue, while the far bid/ask walls describe
        where the range-bound market maker is willing to warehouse risk.
        """
        if od.buy_orders and od.sell_orders:
            return (min(od.buy_orders) + max(od.sell_orders)) / 2.0
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
        orders: List[Order] = []
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

    @staticmethod
    def _rolling_ols_slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(values):
            x = i - x_mean
            num += x * (y - y_mean)
            den += x * x
        return num / den if den > 0 else 0.0

    def _update_osmium_fill_quality(
        self,
        saved: dict,
        own_trades: List[Trade],
        timestamp: int,
        mark_price: float,
    ) -> None:
        pending = saved["osmium_pending_fills"]
        seen = saved["osmium_seen_fills"]
        seen_set = set(seen)

        for trade in own_trades:
            side = 0
            if trade.buyer == "SUBMISSION":
                side = 1
            elif trade.seller == "SUBMISSION":
                side = -1
            if side == 0:
                continue

            key = f"{trade.timestamp}:{trade.price}:{trade.quantity}:{side}"
            if key in seen_set:
                continue
            pending.append([trade.timestamp, side, trade.price])
            seen.append(key)
            seen_set.add(key)

        if len(seen) > 180:
            del seen[:-180]

        alpha = self.OSMIUM_FILL_QUALITY_ALPHA
        keep = []
        for fill_ts, side, price in pending:
            if timestamp - fill_ts >= self.OSMIUM_FILL_HORIZON:
                quality = side * (mark_price - price)
                if side > 0:
                    saved["osmium_buy_fill_quality"] = (
                        alpha * quality
                        + (1.0 - alpha) * saved["osmium_buy_fill_quality"]
                    )
                else:
                    saved["osmium_sell_fill_quality"] = (
                        alpha * quality
                        + (1.0 - alpha) * saved["osmium_sell_fill_quality"]
                    )
            else:
                keep.append([fill_ts, side, price])
        saved["osmium_pending_fills"] = keep[-100:]

    def run(self, state: TradingState):
        result = {}

        saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        # PEPPER state
        saved.setdefault("pepper_ema", None)
        saved.setdefault("pepper_ema_slow", None)
        saved.setdefault("pepper_last_mid", None)
        saved.setdefault("pepper_emv", None)
        saved.setdefault("pepper_flow", 0.0)
        saved.setdefault("pepper_hist", [])
        saved.setdefault("pepper_regime", 1)   # -1 short core, 0 flat, +1 long core
        saved.setdefault("pepper_clock_origin", None)
        saved.setdefault("pepper_drift", None)
        saved.setdefault("pepper_last_timestamp", None)
        saved.setdefault("pepper_carry_break_count", 0)
        saved.setdefault("pepper_flat_count", 0)
        saved.setdefault("pepper_short_count", 0)
        saved.setdefault("pepper_recover_count", 0)
        # OSMIUM state
        saved.setdefault("osmium_ema", None)
        saved.setdefault("osmium_min", None)
        saved.setdefault("osmium_max", None)
        saved.setdefault("osmium_extreme_signal", 0.0)
        saved.setdefault("osmium_last_wall", None)
        saved.setdefault("osmium_diff_var", None)
        saved.setdefault("osmium_spike_signal", 0.0)
        saved.setdefault("osmium_anchor", None)
        saved.setdefault("osmium_pending_fills", [])
        saved.setdefault("osmium_seen_fills", [])
        saved.setdefault("osmium_buy_fill_quality", 0.0)
        saved.setdefault("osmium_sell_fill_quality", 0.0)

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == self.PEPPER:
                result[product] = self._trade_pepper(
                    od=od,
                    position=position,
                    prior_trades=state.market_trades.get(product, []),
                    timestamp=state.timestamp,
                    saved=saved,
                )
            elif product == self.OSMIUM:
                result[product] = self._trade_osmium(
                    od=od,
                    position=position,
                    prior_trades=state.market_trades.get(product, []),
                    own_trades=state.own_trades.get(product, []),
                    timestamp=state.timestamp,
                    saved=saved,
                )

        trader_data = jsonpickle.encode(saved)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

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
            microprice = self._microprice(od, mid, best_bid, best_ask)
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)

        # timestamp reset protection
        if saved["pepper_last_timestamp"] is not None and timestamp < saved["pepper_last_timestamp"]:
            saved["pepper_ema"] = None
            saved["pepper_ema_slow"] = None
            saved["pepper_last_mid"] = None
            saved["pepper_emv"] = None
            saved["pepper_flow"] = 0.0
            saved["pepper_hist"] = []
            saved["pepper_regime"] = 1
            saved["pepper_clock_origin"] = None
            saved["pepper_drift"] = None
            saved["pepper_carry_break_count"] = 0
            saved["pepper_flat_count"] = 0
            saved["pepper_short_count"] = 0
            saved["pepper_recover_count"] = 0
        saved["pepper_last_timestamp"] = timestamp

        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)
        if saved["pepper_clock_origin"] is None:
            saved["pepper_clock_origin"] = ref_price
        if saved["pepper_drift"] is None:
            saved["pepper_drift"] = self.PEPPER_CLOCK_DRIFT_PER_STEP

        # signed trade-flow signal
        if saved["pepper_last_mid"] is not None and prior_trades:
            signed_qty = 0.0
            total_qty = 0.0
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

        # EMA pair
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
        trend_signal = ema - ema_slow

        # rolling slope
        hist = saved["pepper_hist"]
        hist.append(ref_price)
        if len(hist) > self.PEPPER_SLOPE_WINDOW:
            hist.pop(0)
        slope = self._rolling_ols_slope(hist)

        # vol estimate
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

        # Forecast = online fair + weak structural carry prior. The clock term
        # is deliberately used as a prior, then guarded by the carry-break veto
        # below, instead of acting as an unconditional hard-coded rule.
        clock_step = timestamp / 100.0
        if clock_step > 30:
            drift_obs = (ref_price - saved["pepper_clock_origin"]) / clock_step
            drift_obs = max(self.PEPPER_DRIFT_MIN, min(self.PEPPER_DRIFT_MAX, drift_obs))
            a_drift = self.PEPPER_DRIFT_ALPHA
            saved["pepper_drift"] = (
                a_drift * drift_obs + (1.0 - a_drift) * saved["pepper_drift"]
            )
        clock_fair = saved["pepper_clock_origin"] + saved["pepper_drift"] * clock_step
        forecast = (
            ema
            + self.PEPPER_TREND_CARRY * trend_signal
            + self.PEPPER_SLOPE_WEIGHT * slope
            + self.PEPPER_FLOW_WEIGHT * saved["pepper_flow"]
            + self.PEPPER_CLOCK_DRIFT_WEIGHT * (clock_fair - ema)
        )

        # Philosophical change from v26's pure classifier:
        # profitable carry is the base case; online signals are a brake. A
        # short noisy OLS slope should not liquidate a strategic position.
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
        drift = saved["pepper_drift"]
        flat_warning = (
            (drift < 0.025 and trend_signal < -1.0)
            or (clock_gap < -20.0 and trend_signal < -2.0)
            or (saved["pepper_flow"] < -0.65 and slope < -0.35 and trend_signal < 0)
        )
        short_warning = (
            carry_broken
            or (drift < -0.010 and trend_signal < -3.0 and clock_gap < -25.0)
            or (clock_gap < -35.0 and trend_signal < -5.0)
        )
        recover_signal = (
            (drift > 0.050 and trend_signal > -0.5 and clock_gap > -12.0)
            or trend_signal > 1.0
        )

        if short_warning:
            saved["pepper_short_count"] += 1
        else:
            saved["pepper_short_count"] = max(0, saved["pepper_short_count"] - 1)

        if flat_warning:
            saved["pepper_flat_count"] += 1
        else:
            saved["pepper_flat_count"] = max(0, saved["pepper_flat_count"] - 1)

        if recover_signal:
            saved["pepper_recover_count"] += 1
        else:
            saved["pepper_recover_count"] = max(0, saved["pepper_recover_count"] - 1)

        prev_regime = saved["pepper_regime"]
        if saved["pepper_short_count"] >= self.PEPPER_SHORT_CONFIRM:
            regime = -1
        elif saved["pepper_flat_count"] >= self.PEPPER_FLAT_CONFIRM:
            regime = 0
        elif saved["pepper_recover_count"] >= self.PEPPER_RECOVER_CONFIRM:
            regime = 1
        else:
            regime = prev_regime if prev_regime in (-1, 0, 1) else 1

        strong_reversal = regime < 0
        target_core = (
            self.PEPPER_CORE_TARGET if regime > 0
            else -self.PEPPER_CORE_TARGET if regime < 0
            else 0
        )
        saved["pepper_regime"] = regime

        # Stage 1: build the carry core. The core buy rule is intentionally
        # more aggressive than a normal "ask below forecast" rule because the
        # alpha is exposure to carry, not one-tick spread capture.
        if regime > 0 and has_ask:
            for level_idx, ask in enumerate(sorted(od.sell_orders)):
                if position < target_core and level_idx >= self.PEPPER_CORE_TAKE_LEVELS:
                    break
                can_buy = limit - position
                if can_buy <= 0:
                    break

                building_core = position < target_core
                sleeve_edge = ask <= forecast + self.PEPPER_CORE_BID_EDGE
                if not building_core and not sleeve_edge:
                    break

                qty = min(-od.sell_orders[ask], can_buy)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        elif regime < 0 and has_bid:
            for bid in sorted(od.buy_orders, reverse=True):
                can_sell = limit + position
                if can_sell <= 0:
                    break

                building_core = position > target_core
                sleeve_edge = bid >= forecast - self.PEPPER_CORE_BID_EDGE
                if not building_core and not sleeve_edge:
                    break

                qty = min(od.buy_orders[bid], can_sell)
                if building_core:
                    qty = min(qty, position - target_core)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        # Stage 2: only recycle the sleeve above the protected core, unless the
        # carry brake has fired. This avoids selling the structural alpha.
        if regime > 0 and has_bid:
            for level_idx, bid in enumerate(sorted(od.buy_orders, reverse=True)):
                if level_idx >= self.PEPPER_SLEEVE_SELL_LEVELS:
                    break
                if bid <= forecast + self.PEPPER_SLEEVE_SELL_EDGE:
                    break
                can_sell = max(0, position - target_core)
                if can_sell <= 0:
                    break
                qty = min(od.buy_orders[bid], can_sell)
                if qty > 0:
                    orders.append(Order(self.PEPPER, bid, -qty))
                    position -= qty

        elif regime < 0 and has_ask:
            for ask in sorted(od.sell_orders):
                if ask >= forecast - self.PEPPER_SLEEVE_SELL_EDGE:
                    break
                can_buy = max(0, target_core - position)
                if can_buy <= 0:
                    break
                qty = min(-od.sell_orders[ask], can_buy, limit - position)
                if qty > 0:
                    orders.append(Order(self.PEPPER, ask, qty))
                    position += qty

        # Stage 3: passive quoting around target core
        biased_pos = position - self.PEPPER_MOMENTUM_BIAS * trend_signal
        biased_pos = max(-limit, min(limit, biased_pos))

        skew = self.PEPPER_SKEW * biased_pos
        if abs(biased_pos) > self.PEPPER_URGENCY_THRESH * limit:
            skew *= self.PEPPER_URGENCY_MULT
        eff = forecast - skew

        min_edge = 1 + spread_extra
        passive_buy, passive_sell = self._penny_passive_prices(
            od,
            ref=eff,
            fair=forecast,
            min_buy_edge=min_edge,
            min_sell_edge=min_edge,
        )

        remaining_buy = limit - position
        remaining_sell = limit + position

        if regime > 0:
            if position < target_core:
                remaining_buy = limit - position
            elif position < self.PEPPER_REFILL_TARGET:
                remaining_buy = max(0, self.PEPPER_REFILL_TARGET - position)
            else:
                remaining_buy = max(0, limit - position)
            remaining_sell = max(0, position - target_core)
        elif regime < 0:
            if position > target_core:
                remaining_sell = limit + position
            elif position > -self.PEPPER_REFILL_TARGET:
                remaining_sell = max(0, position + self.PEPPER_REFILL_TARGET)
            else:
                remaining_sell = max(0, limit + position)
            remaining_buy = max(0, target_core - position)
        else:
            remaining_buy = max(0, -position) + max(0, int(round((limit - position) * 0.20)))
            remaining_sell = max(0, position) + max(0, int(round((limit + position) * 0.20)))

        remaining_buy = min(remaining_buy, limit - position)
        remaining_sell = min(remaining_sell, limit + position)

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
            f"PEP| mid={mid:.1f} ref={ref_price:.1f} ema={ema:.2f} slow={ema_slow:.2f} "
            f"sig={trend_signal:.2f} slope={slope:.3f} flow={saved['pepper_flow']:.2f} "
            f"clk_gap={clock_gap:.1f} brk={saved['pepper_carry_break_count']} "
            f"flat={saved['pepper_flat_count']} short={saved['pepper_short_count']} "
            f"reg={regime} core={target_core} "
            f"fcst={forecast:.2f} eff={eff:.2f} pb={passive_buy} ps={passive_sell} pos={position}"
        )

        return orders

    # ── OSMIUM ──────────────────────────────────────────────────────────────

    def _trade_osmium(
        self,
        od: OrderDepth,
        position: int,
        prior_trades: List[Trade],
        own_trades: List[Trade],
        timestamp: int,
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
            microprice = self._microprice(od, mid, best_bid, best_ask)
        elif has_bid:
            best_bid = max(od.buy_orders)
            mid = microprice = float(best_bid)
            spread = 16
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)
            spread = 16

        wall_mid = self._wall_mid(od, mid)
        self._update_osmium_fill_quality(saved, own_trades, timestamp, mid)

        if saved["osmium_min"] is None:
            saved["osmium_min"] = wall_mid
            saved["osmium_max"] = wall_mid
        saved["osmium_min"] = min(saved["osmium_min"], wall_mid)
        saved["osmium_max"] = max(saved["osmium_max"], wall_mid)
        range_mid = (saved["osmium_min"] + saved["osmium_max"]) / 2.0
        if saved["osmium_anchor"] is None:
            saved["osmium_anchor"] = wall_mid
        else:
            saved["osmium_anchor"] = 0.01 * range_mid + 0.99 * saved["osmium_anchor"]

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

        fair_observation = (
            0.54 * wall_mid
            + 0.20 * microprice
            + 0.26 * saved["osmium_anchor"]
        )

        if saved["osmium_last_wall"] is not None:
            wall_diff = wall_mid - saved["osmium_last_wall"]
            diff_sq = wall_diff * wall_diff
            if saved["osmium_diff_var"] is None:
                saved["osmium_diff_var"] = diff_sq
            else:
                saved["osmium_diff_var"] = 0.10 * diff_sq + 0.90 * saved["osmium_diff_var"]
            vol = max(1.0, saved["osmium_diff_var"] ** 0.5)
            if abs(wall_diff) > 2.5 * vol:
                saved["osmium_spike_signal"] = -1.0 if wall_diff > 0 else 1.0
            else:
                saved["osmium_spike_signal"] *= 0.85
        saved["osmium_last_wall"] = wall_mid

        if saved["osmium_ema"] is None:
            saved["osmium_ema"] = fair_observation
        else:
            a = self.OSMIUM_EMA_ALPHA
            saved["osmium_ema"] = a * fair_observation + (1.0 - a) * saved["osmium_ema"]
        ema = saved["osmium_ema"]

        fair = (
            ema
            + 0.60 * saved["osmium_extreme_signal"]
            + 0.40 * saved["osmium_spike_signal"]
        )
        fill_bias = saved["osmium_sell_fill_quality"] - saved["osmium_buy_fill_quality"]
        fill_bias = max(-self.OSMIUM_FILL_BIAS_CAP, min(self.OSMIUM_FILL_BIAS_CAP, fill_bias))
        fair -= self.OSMIUM_FILL_FAIR_WEIGHT * fill_bias

        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)

        half_spread = max(1.0, spread / 2.0)
        micro_signal = max(-1.0, min(1.0, (microprice - mid) / half_spread))
        toxicity = (
            self.OSMIUM_TOX_OBI_WEIGHT * self._book_imbalance(od)
            + self.OSMIUM_TOX_MICRO_WEIGHT * micro_signal
        )

        abs_pos = abs(position)
        if abs_pos <= self.OSMIUM_NEUTRAL_POS:
            regime = "N"
            buy_edge = sell_edge = self.OSMIUM_BASE_WIDTH
            buy_take = sell_take = self.OSMIUM_NEUTRAL_TAKE
            buy_mult = sell_mult = 1.0
        elif abs_pos <= self.OSMIUM_LOADED_POS:
            regime = "L"
            if position > 0:
                buy_edge = self.OSMIUM_BASE_WIDTH + 1
                sell_edge = self.OSMIUM_BASE_WIDTH
                buy_take = self.OSMIUM_NEUTRAL_TAKE + 2
                sell_take = self.OSMIUM_EXIT_TAKE
                buy_mult = self.OSMIUM_LOADED_WRONG_MULT
                sell_mult = 1.0
            else:
                sell_edge = self.OSMIUM_BASE_WIDTH + 1
                buy_edge = self.OSMIUM_BASE_WIDTH
                sell_take = self.OSMIUM_NEUTRAL_TAKE + 2
                buy_take = self.OSMIUM_EXIT_TAKE
                sell_mult = self.OSMIUM_LOADED_WRONG_MULT
                buy_mult = 1.0
        else:
            regime = "S"
            if position > 0:
                buy_edge = self.OSMIUM_BASE_WIDTH + 2
                sell_edge = self.OSMIUM_BASE_WIDTH
                buy_take = self.OSMIUM_NEUTRAL_TAKE + 3
                sell_take = self.OSMIUM_STRESSED_EXIT_TAKE
                buy_mult = self.OSMIUM_STRESSED_WRONG_MULT
                sell_mult = 1.0
            else:
                sell_edge = self.OSMIUM_BASE_WIDTH + 2
                buy_edge = self.OSMIUM_BASE_WIDTH
                sell_take = self.OSMIUM_NEUTRAL_TAKE + 3
                buy_take = self.OSMIUM_STRESSED_EXIT_TAKE
                sell_mult = self.OSMIUM_STRESSED_WRONG_MULT
                buy_mult = 1.0

        # take obvious gifts
        can_take_ask = has_ask and (has_bid or not self.OSMIUM_REQUIRE_TWO_SIDED_TAKE)
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

        can_take_bid = has_bid and (has_ask or not self.OSMIUM_REQUIRE_TWO_SIDED_TAKE)
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

        # clear inventory near fair
        fair_bid = round(fair - self.OSMIUM_CLEAR_WIDTH)
        fair_ask = round(fair + self.OSMIUM_CLEAR_WIDTH)
        if position > 0 and has_bid:
            clearable = sum(v for p, v in od.buy_orders.items() if p >= fair_ask)
            qty = min(position, clearable, limit + position)
            if qty > 0:
                orders.append(Order(self.OSMIUM, fair_ask, -qty))
                position -= qty
        elif position < 0 and has_ask:
            clearable = sum(-v for p, v in od.sell_orders.items() if p <= fair_bid)
            qty = min(-position, clearable, limit - position)
            if qty > 0:
                orders.append(Order(self.OSMIUM, fair_bid, qty))
                position += qty

        reservation = fair - self.OSMIUM_INV_SKEW * (position / limit)
        passive_buy, passive_sell = self._penny_passive_prices(
            od, reservation, reservation, buy_edge, sell_edge
        )

        buy_inventory_scale = max(0.20, 1.0 - max(0.0, position / limit))
        sell_inventory_scale = max(0.20, 1.0 - max(0.0, -position / limit))

        remaining_buy = max(0, int(round((limit - position) * buy_mult * buy_inventory_scale)))
        remaining_sell = max(0, int(round((limit + position) * sell_mult * sell_inventory_scale)))

        tox_mult = self.OSMIUM_TOX_SIDE_MULT
        if spread >= 10 and abs(position) <= 24:
            tox_mult = 0.80
        elif abs(position) >= 56:
            tox_mult = 0.50

        if toxicity > 0:
            remaining_sell = int(round(remaining_sell * tox_mult))
        elif toxicity < 0:
            remaining_buy = int(round(remaining_buy * tox_mult))

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
            f"spk={saved['osmium_spike_signal']:.2f} res={reservation:.2f} "
            f"fillB={saved['osmium_buy_fill_quality']:.2f} "
            f"fillS={saved['osmium_sell_fill_quality']:.2f} tox={toxicity:.2f} "
            f"reg={regime} pb={passive_buy} ps={passive_sell} pos={position}"
        )

        return orders
