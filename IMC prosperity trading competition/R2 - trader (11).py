
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
    v2-R2: R1 v31 baseline + R2 mechanics + macro/micro separated alpha.

    R2 changes:
      - bid() returns MAF to win 25% extra order book visibility.
      - OSMIUM range-directional lean: when price is near the top of its
        observed range, reservation shifts down (lean sell); near bottom,
        reservation shifts up (lean buy). Converts passive MM into active
        mean-reversion harvesting at extremes without sacrificing robustness.
      - OSMIUM structural dislocation allocator: wall-mid and hot-quote
        events are used as passive quote/size alpha, not blind crossing alpha.
        Crossing L1 usually pays too much spread; the edge is sitting one tick
        inside the far side when same-timestamp market flow arrives.
      - OSMIUM toxicity is L1/microprice based. Full-book OBI is deliberately
        avoided because deep queues cancel the predictive L1 pressure.
      - OSMIUM passive liquidity is concentrated at the best inside quote. In
        the matcher, same-timestamp market-trade fills stop after the first
        matching buy/sell order, so a second passive level is structurally
        inferior rather than true queue diversification.
      - PEPPER fair is clock-dominant because R2 PEPPER has a stable
        +0.1/tick macro path; EMA/flow remain sleeve controls.
      - OSMIUM round-anchor mean reversion changes inventory appetite and
        crossing permission, not passive quote price. The anchor is inferred
        from the first valid mid instead of hardcoding a literal level.
      - All R1 robustness guards preserved unchanged (regime detection, carry
        brake, fill quality, spike/extreme signals, toxicity, two-sided gate).
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
    PEPPER_SLOPE_WEIGHT = 0.0

    PEPPER_SKEW = 0.008
    PEPPER_URGENCY_THRESH = 0.70
    PEPPER_URGENCY_MULT = 3.0
    PEPPER_VOL_THRESH = 4.0
    PEPPER_SPREAD_HIGH = 1
    PEPPER_ONE_SIDE_HALF_SPREAD = 7.0
    PEPPER_ONE_SIDE_OBS_WEIGHT = 0.50

    PEPPER_MOMENTUM_BIAS = 0
    PEPPER_TREND_GATE = 0.0
    PEPPER_AGAINST_TREND_CAP = 0.55
    PEPPER_CARRY_EDGE = 0.0
    PEPPER_LONG_FLOOR = 60
    PEPPER_CLOCK_DRIFT_PER_STEP = 0.10
    PEPPER_CLOCK_DRIFT_WEIGHT = 1.00
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
    PEPPER_ADVERSE_DRIFT_GATE = -0.015
    PEPPER_ADVERSE_TREND_GATE = -0.75
    PEPPER_EARLY_SHORT_DRIFT = -0.035
    PEPPER_EARLY_SHORT_TREND = -1.50
    PEPPER_KIN_ALPHA = 0.35
    PEPPER_KIN_BETA = 0.08
    PEPPER_DIP_SLOPE_THRESH = -1.0
    PEPPER_DIP_GAP_THRESH = -18.0
    PEPPER_DIP_CONFIRM = 2
    PEPPER_DIP_EXTRA_TARGET = 8
    PEPPER_FAST_BREAK_SLOPE = -2.5
    PEPPER_FAST_BREAK_GAP = -38.0

    # ── OSMIUM ──────────────────────────────────────────────────────────────
    OSMIUM_LIMIT = 80
    OSMIUM_EMA_ALPHA = 0.12
    OSMIUM_INV_SKEW = 2.5

    # R2 microstructure: total-book OBI is weak because deep queues cancel L1.
    # L1 imbalance and microprice are the short-horizon toxicity signal; deep
    # imbalance is used only as a small fade because it points the other way.
    OSMIUM_TOX_L1_WEIGHT = 0.58
    OSMIUM_TOX_MICRO_WEIGHT = 0.34
    OSMIUM_TOX_DEEP_FADE_WEIGHT = 0.08
    OSMIUM_TOX_SIDE_MULT = 0.60
    OSMIUM_L1_FRAC = 1.00
    OSMIUM_L2_FRAC = 0.00

    OSMIUM_BASE_WIDTH = 1
    OSMIUM_ONE_SIDE_HALF_SPREAD = 8.0
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
    OSMIUM_FILL_TICK_HORIZON = 1000
    OSMIUM_FILL_QUALITY_ALPHA = 0.025
    OSMIUM_FILL_BIAS_CAP = 2.0
    OSMIUM_FILL_FAIR_WEIGHT = 0.03

    # R2: range-directional lean — shifts reservation toward mean reversion at extremes
    OSMIUM_RANGE_SKEW = 1.8
    # R2: hot-quote signal — tracks transient aggressive bids/asks above EMA
    OSMIUM_HOT_SIGNAL_ALPHA = 0.20
    OSMIUM_HOT_TAKE_THRESH = 0.40   # signal magnitude to tighten edge one step
    OSMIUM_HOT_EDGE_REDUCTION = 1   # ticks to tighten edge on hot side
    OSMIUM_NEAR_EMA_ALPHA = 0.15    # speed of near-bid/ask EMA tracking
    OSMIUM_HOT_L1L2_THRESH = 5     # min L1-L2 gap to call it a hot quote
    OSMIUM_HOT_EMA_THRESH = 4      # min best_price - EMA gap to call it hot

    # R2 structural dislocation allocator.
    # Crossing L1 is usually negative EV after spread; the alpha belongs in
    # passive quote placement and side sizing.
    OSMIUM_WALL_DEV_THRESH = 2.0
    OSMIUM_WALL_REVERSION_SHIFT = 1.10
    OSMIUM_HOT_REVERSION_SHIFT = 0.75
    OSMIUM_DISLOCATION_MIN_FRAC = 0.72
    OSMIUM_DISLOCATION_OPPOSITE_MULT = 0.28

    # Same-timestamp sweep filter: avoid joining exactly one tick inside when
    # L1 pressure points through us and deep queues do not confirm a fade.
    OSMIUM_SWEEP_L1_THRESH = 0.25
    OSMIUM_SWEEP_DEEP_CONFIRM = 0.10
    OSMIUM_SWEEP_WALL_VOLUME = 24
    OSMIUM_SWEEP_SIZE_MULT = 0.25

    # Long-horizon OSMIUM mean reversion is around the product's round-number
    # center. Infer the center online to avoid brittle absolute hardcoding.
    OSMIUM_ROUND_ANCHOR_UNIT = 100.0
    OSMIUM_ABS_BAND = 6.0
    OSMIUM_ABS_INV_FRAC = 0.60
    OSMIUM_ABS_OPPOSITE_MULT = 0.35
    OSMIUM_ABS_CONFLICT_MULT = 0.55
    OSMIUM_ABS_CROSS_POS_ALLOW = 32

    # ── MAF ─────────────────────────────────────────────────────────────────
    def bid(self) -> int:
        """
        Market Access Fee bid.
        Top 50% of bids win extra quote visibility.  Keep this conservative:
        local research shows the strategy benefits from extra book structure,
        but the modeled PnL edge does not justify paying thousands in fees.
        """
        return 200

    # ── static helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _book_imbalance(od: OrderDepth) -> float:
        bid_volume = sum(max(0, v) for v in od.buy_orders.values())
        ask_volume = sum(max(0, -v) for v in od.sell_orders.values())
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0

    @staticmethod
    def _l1_imbalance(od: OrderDepth, best_bid: int, best_ask: int) -> float:
        bid_volume = max(0, od.buy_orders.get(best_bid, 0))
        ask_volume = max(0, -od.sell_orders.get(best_ask, 0))
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0

    @staticmethod
    def _deep_imbalance(od: OrderDepth, best_bid: int, best_ask: int) -> float:
        deep_bid_volume = sum(max(0, v) for p, v in od.buy_orders.items() if p != best_bid)
        deep_ask_volume = sum(max(0, -v) for p, v in od.sell_orders.items() if p != best_ask)
        total = deep_bid_volume + deep_ask_volume
        return (deep_bid_volume - deep_ask_volume) / total if total > 0 else 0.0

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
        Far bid/ask outer walls: where range-bound market makers warehouse risk.
        Degrades gracefully to L1 mid when outer levels are hidden (R2 80% visibility).
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

    @staticmethod
    def _hot_quote_signal(od: OrderDepth, bid_ema: float, ask_ema: float,
                          l1l2_thresh: int, ema_thresh: int) -> float:
        """
        Detect transient aggressive bids/asks that will exhaust and mean-revert.

        A "hot bid" is when L1 bid jumps significantly above L2 bid AND above
        its own EMA: this signals a buyer paying urgency premium.  In a
        mean-reverting market the right response is to sell to them.
        Returns +1 (hot bid → sell opportunity), -1 (hot ask → buy opportunity).
        Partial scores for moderate signals.
        """
        bids = sorted(od.buy_orders.keys(), reverse=True)
        asks = sorted(od.sell_orders.keys())
        signal = 0.0

        if len(bids) >= 2 and bid_ema > 0:
            l1l2_gap = bids[0] - bids[1]
            ema_gap = bids[0] - bid_ema
            if l1l2_gap >= l1l2_thresh and ema_gap >= ema_thresh:
                signal += 1.0
            elif l1l2_gap >= (l1l2_thresh - 2) and ema_gap >= (ema_thresh + 2):
                signal += 0.5

        if len(asks) >= 2 and ask_ema > 0:
            l1l2_gap = asks[1] - asks[0]
            ema_gap = ask_ema - asks[0]
            if l1l2_gap >= l1l2_thresh and ema_gap >= ema_thresh:
                signal -= 1.0
            elif l1l2_gap >= (l1l2_thresh - 2) and ema_gap >= (ema_thresh + 2):
                signal -= 0.5

        return signal

    def _update_osmium_fill_quality(
        self,
        saved: dict,
        own_trades: List[Trade],
        timestamp: int,
        mark_price: float,
        current_tick: int,
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
            pending.append([trade.timestamp, side, trade.price, current_tick])
            seen.append(key)
            seen_set.add(key)

        if len(seen) > 180:
            del seen[:-180]

        alpha = self.OSMIUM_FILL_QUALITY_ALPHA
        keep = []
        for record in pending:
            fill_ts, side, price = record[0], record[1], record[2]
            fill_tick = record[3] if len(record) > 3 else current_tick
            ready_by_time = timestamp - fill_ts >= self.OSMIUM_FILL_HORIZON
            ready_by_ticks = current_tick - fill_tick >= self.OSMIUM_FILL_TICK_HORIZON
            if ready_by_time or ready_by_ticks:
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
                keep.append([fill_ts, side, price, fill_tick])
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
        saved.setdefault("pepper_regime", 1)
        saved.setdefault("pepper_clock_origin", None)
        saved.setdefault("pepper_drift", None)
        saved.setdefault("pepper_last_timestamp", None)
        saved.setdefault("pepper_fast_level", None)
        saved.setdefault("pepper_fast_slope", None)
        saved.setdefault("pepper_fast_resid", 0.0)
        saved.setdefault("pepper_dip_count", 0)
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
        saved.setdefault("osmium_tick", 0)
        # R2: new OSMIUM state for granular mean-reversion signals
        saved.setdefault("osmium_near_bid_ema", None)   # EMA of best bid price
        saved.setdefault("osmium_near_ask_ema", None)   # EMA of best ask price
        saved.setdefault("osmium_hot_signal", 0.0)      # EWA hot-quote signal
        saved.setdefault("osmium_round_anchor", None)

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
        obs_weight = 1.0 if has_bid and has_ask else self.PEPPER_ONE_SIDE_OBS_WEIGHT

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
            saved["pepper_fast_level"] = None
            saved["pepper_fast_slope"] = None
            saved["pepper_fast_resid"] = 0.0
            saved["pepper_dip_count"] = 0
            saved["pepper_carry_break_count"] = 0
            saved["pepper_flat_count"] = 0
            saved["pepper_short_count"] = 0
            saved["pepper_recover_count"] = 0
        saved["pepper_last_timestamp"] = timestamp

        ref_price = self._large_quote_mid(od, microprice, self.PEPPER_MM_VOLUME)
        if has_bid and not has_ask:
            ref_price = best_bid + self.PEPPER_ONE_SIDE_HALF_SPREAD
        elif has_ask and not has_bid:
            ref_price = best_ask - self.PEPPER_ONE_SIDE_HALF_SPREAD
        if saved["pepper_clock_origin"] is None:
            saved["pepper_clock_origin"] = ref_price
        if saved["pepper_drift"] is None:
            saved["pepper_drift"] = self.PEPPER_CLOCK_DRIFT_PER_STEP
        if saved["pepper_fast_level"] is None:
            saved["pepper_fast_level"] = ref_price
        if saved["pepper_fast_slope"] is None:
            saved["pepper_fast_slope"] = self.PEPPER_CLOCK_DRIFT_PER_STEP

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

        if saved["pepper_ema"] is None:
            saved["pepper_ema"] = ref_price
        else:
            a = self.PEPPER_EMA_ALPHA * obs_weight
            saved["pepper_ema"] = a * ref_price + (1.0 - a) * saved["pepper_ema"]
        ema = saved["pepper_ema"]

        if saved["pepper_ema_slow"] is None:
            saved["pepper_ema_slow"] = ref_price
        else:
            a = self.PEPPER_EMA_SLOW_ALPHA * obs_weight
            saved["pepper_ema_slow"] = a * ref_price + (1.0 - a) * saved["pepper_ema_slow"]
        ema_slow = saved["pepper_ema_slow"]
        trend_signal = ema - ema_slow

        hist = saved["pepper_hist"]
        hist.append(ref_price)
        if len(hist) > self.PEPPER_SLOPE_WINDOW:
            hist.pop(0)
        slope = self._rolling_ols_slope(hist)

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

        clock_step = timestamp / 100.0
        if clock_step > 30:
            drift_obs = (ref_price - saved["pepper_clock_origin"]) / clock_step
            drift_obs = max(self.PEPPER_DRIFT_MIN, min(self.PEPPER_DRIFT_MAX, drift_obs))
            a_drift = self.PEPPER_DRIFT_ALPHA * obs_weight
            saved["pepper_drift"] = (
                a_drift * drift_obs + (1.0 - a_drift) * saved["pepper_drift"]
            )
        clock_fair = saved["pepper_clock_origin"] + saved["pepper_drift"] * clock_step
        fast_pred = saved["pepper_fast_level"] + saved["pepper_fast_slope"]
        fast_resid = ref_price - fast_pred
        fast_alpha = self.PEPPER_KIN_ALPHA * obs_weight
        fast_beta = self.PEPPER_KIN_BETA * obs_weight
        saved["pepper_fast_level"] = fast_pred + fast_alpha * fast_resid
        saved["pepper_fast_slope"] += fast_beta * fast_resid
        saved["pepper_fast_resid"] = fast_resid
        fast_slope = saved["pepper_fast_slope"]
        forecast = (
            ema
            + self.PEPPER_TREND_CARRY * trend_signal
            + self.PEPPER_SLOPE_WEIGHT * slope
            + self.PEPPER_FLOW_WEIGHT * saved["pepper_flow"]
            + self.PEPPER_CLOCK_DRIFT_WEIGHT * (clock_fair - ema)
        )

        clock_gap = ref_price - clock_fair
        fast_break = (
            clock_gap < self.PEPPER_FAST_BREAK_GAP
            and fast_slope < self.PEPPER_FAST_BREAK_SLOPE
            and trend_signal < -2.0
        )
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
        dip_signal = (
            clock_step > 120
            and drift > 0.035
            and clock_gap < self.PEPPER_DIP_GAP_THRESH
            and fast_slope < self.PEPPER_DIP_SLOPE_THRESH
            and not carry_broken
            and trend_signal > self.PEPPER_CARRY_BREAK_TREND
        )
        flat_warning = (
            (drift < 0.025 and trend_signal < -1.0)
            or (clock_gap < -20.0 and trend_signal < -2.0)
            or (saved["pepper_flow"] < -0.65 and slope < -0.35 and trend_signal < 0)
            or fast_break
        )
        short_warning = (
            carry_broken
            or (drift < -0.010 and trend_signal < -3.0 and clock_gap < -25.0)
            or (clock_gap < -35.0 and trend_signal < -5.0)
            or fast_break
        )
        if (
            clock_step > 120
            and drift < self.PEPPER_ADVERSE_DRIFT_GATE
            and trend_signal < self.PEPPER_ADVERSE_TREND_GATE
        ):
            flat_warning = True
        if (
            clock_step > 250
            and drift < self.PEPPER_EARLY_SHORT_DRIFT
            and trend_signal < self.PEPPER_EARLY_SHORT_TREND
        ):
            short_warning = True
        recover_signal = (
            (drift > 0.050 and trend_signal > -0.5 and clock_gap > -12.0)
            or trend_signal > 1.0
        )
        if dip_signal:
            saved["pepper_dip_count"] += 1
        else:
            saved["pepper_dip_count"] = max(0, saved["pepper_dip_count"] - 1)

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

        target_core = (
            self.PEPPER_CORE_TARGET if regime > 0
            else -self.PEPPER_CORE_TARGET if regime < 0
            else 0
        )
        dip_active = regime > 0 and saved["pepper_dip_count"] >= self.PEPPER_DIP_CONFIRM
        if dip_active:
            target_core = min(limit, target_core + self.PEPPER_DIP_EXTRA_TARGET)
        saved["pepper_regime"] = regime

        # Stage 1: build carry core
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

        # Stage 2: recycle sleeve above core only
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

        # Stage 3: passive quoting
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
            f"clk_gap={clock_gap:.1f} kin={fast_slope:.2f} kres={fast_resid:.1f} "
            f"dip={saved['pepper_dip_count']} brk={saved['pepper_carry_break_count']} "
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
            if saved["osmium_ema"] is None:
                mid = microprice = float(best_bid + self.OSMIUM_ONE_SIDE_HALF_SPREAD)
        else:
            best_ask = min(od.sell_orders)
            mid = microprice = float(best_ask)
            spread = 16
            if saved["osmium_ema"] is None:
                mid = microprice = float(best_ask - self.OSMIUM_ONE_SIDE_HALF_SPREAD)

        wall_mid = self._wall_mid(od, mid)
        saved["osmium_tick"] += 1
        self._update_osmium_fill_quality(
            saved, own_trades, timestamp, mid, saved["osmium_tick"]
        )

        # R2: update near-bid and near-ask EMAs for hot-quote detection
        alpha_near = self.OSMIUM_NEAR_EMA_ALPHA
        if has_bid:
            if saved["osmium_near_bid_ema"] is None:
                saved["osmium_near_bid_ema"] = float(best_bid)
            else:
                saved["osmium_near_bid_ema"] = (
                    alpha_near * best_bid
                    + (1.0 - alpha_near) * saved["osmium_near_bid_ema"]
                )
        if has_ask:
            if saved["osmium_near_ask_ema"] is None:
                saved["osmium_near_ask_ema"] = float(best_ask)
            else:
                saved["osmium_near_ask_ema"] = (
                    alpha_near * best_ask
                    + (1.0 - alpha_near) * saved["osmium_near_ask_ema"]
                )

        # R2: compute hot-quote signal (transient aggressive bids/asks above EMA)
        near_bid_ema = saved["osmium_near_bid_ema"] or 0.0
        near_ask_ema = saved["osmium_near_ask_ema"] or 0.0
        raw_hot = 0.0
        if has_bid and has_ask and near_bid_ema > 0 and near_ask_ema > 0:
            raw_hot = self._hot_quote_signal(
                od, near_bid_ema, near_ask_ema,
                self.OSMIUM_HOT_L1L2_THRESH,
                self.OSMIUM_HOT_EMA_THRESH,
            )
        saved["osmium_hot_signal"] = (
            self.OSMIUM_HOT_SIGNAL_ALPHA * raw_hot
            + (1.0 - self.OSMIUM_HOT_SIGNAL_ALPHA) * saved["osmium_hot_signal"]
        )
        hot = saved["osmium_hot_signal"]

        wall_dev = (mid - wall_mid) if has_bid and has_ask else 0.0
        wall_reversion = 0.0
        if wall_dev >= self.OSMIUM_WALL_DEV_THRESH:
            wall_reversion = -min(1.0, wall_dev / 4.0)
        elif wall_dev <= -self.OSMIUM_WALL_DEV_THRESH:
            wall_reversion = min(1.0, -wall_dev / 4.0)

        hot_reversion = -hot
        reversion_bias = (
            self.OSMIUM_WALL_REVERSION_SHIFT * wall_reversion
            + self.OSMIUM_HOT_REVERSION_SHIFT * hot_reversion
        )
        dislocation_vote = 0
        if wall_dev >= self.OSMIUM_WALL_DEV_THRESH:
            dislocation_vote -= 1
        elif wall_dev <= -self.OSMIUM_WALL_DEV_THRESH:
            dislocation_vote += 1
        if hot > self.OSMIUM_HOT_TAKE_THRESH:
            dislocation_vote -= 1
        elif hot < -self.OSMIUM_HOT_TAKE_THRESH:
            dislocation_vote += 1
        dislocation_side = 1 if dislocation_vote > 0 else -1 if dislocation_vote < 0 else 0

        if saved["osmium_round_anchor"] is None and mid > 1000:
            unit = self.OSMIUM_ROUND_ANCHOR_UNIT
            saved["osmium_round_anchor"] = round(mid / unit) * unit
        round_anchor = saved["osmium_round_anchor"] if saved["osmium_round_anchor"] is not None else mid
        round_dev = mid - round_anchor
        round_side = 0
        if round_dev >= self.OSMIUM_ABS_BAND:
            round_side = -1
        elif round_dev <= -self.OSMIUM_ABS_BAND:
            round_side = 1

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

        # R2: range-directional factor — normalized position in observed range
        anchor = saved["osmium_anchor"]
        half_up = max(1.0, saved["osmium_max"] - anchor)
        half_dn = max(1.0, anchor - saved["osmium_min"])
        if mid >= anchor:
            range_factor = min(1.0, (mid - anchor) / half_up)
        else:
            range_factor = max(-1.0, (mid - anchor) / half_dn)
        # Require some warmup before leaning (avoid noisy initial range estimates)
        if saved["osmium_tick"] < 50:
            range_factor = 0.0

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

        # R2: range-directional lean in reservation
        # At range top: lean sell (lower reservation → more willing to sell to bids)
        # At range bottom: lean buy (higher reservation → more willing to buy from asks)
        base_reservation = (
            fair
            - self.OSMIUM_INV_SKEW * (position / limit)
            - self.OSMIUM_RANGE_SKEW * range_factor
        )
        reservation = base_reservation

        half_spread = max(1.0, spread / 2.0)
        micro_signal = max(-1.0, min(1.0, (microprice - mid) / half_spread))
        l1_signal = self._l1_imbalance(od, best_bid, best_ask) if has_bid and has_ask else 0.0
        deep_signal = self._deep_imbalance(od, best_bid, best_ask) if has_bid and has_ask else 0.0
        toxicity = (
            self.OSMIUM_TOX_L1_WEIGHT * l1_signal
            + self.OSMIUM_TOX_MICRO_WEIGHT * micro_signal
            - self.OSMIUM_TOX_DEEP_FADE_WEIGHT * deep_signal
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

        # Structural dislocations are quote alpha, not take alpha.
        # Hot bid (hot > 0): transient buyer → tighten sell edge (quote closer to mid to capture)
        # Hot ask (hot < 0): transient seller → tighten buy edge
        # Tighten only passive edges; do not make L1 crossing easier.
        if dislocation_side < 0:
            sell_edge = max(1, sell_edge - self.OSMIUM_HOT_EDGE_REDUCTION)
        elif dislocation_side > 0:
            buy_edge = max(1, buy_edge - self.OSMIUM_HOT_EDGE_REDUCTION)

        # take obvious gifts
        can_take_ask = has_ask and (has_bid or not self.OSMIUM_REQUIRE_TWO_SIDED_TAKE)
        if round_side < 0 and position > -self.OSMIUM_ABS_CROSS_POS_ALLOW:
            can_take_ask = False
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
        if round_side > 0 and position < self.OSMIUM_ABS_CROSS_POS_ALLOW:
            can_take_bid = False
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

        # Recompute take reservation after any takes
        base_reservation = (
            fair
            - self.OSMIUM_INV_SKEW * (position / limit)
            - self.OSMIUM_RANGE_SKEW * range_factor
        )
        reservation = base_reservation

        # clear inventory near fair — slightly wider at range extremes for faster clearing
        dynamic_clear_width = self.OSMIUM_CLEAR_WIDTH + int(abs(range_factor) >= 0.6)
        fair_bid = round(fair - dynamic_clear_width)
        fair_ask = round(fair + dynamic_clear_width)
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

        base_reservation = (
            fair
            - self.OSMIUM_INV_SKEW * (position / limit)
            - self.OSMIUM_RANGE_SKEW * range_factor
        )
        quote_reservation = base_reservation + reversion_bias
        reservation = quote_reservation
        passive_buy, passive_sell = self._penny_passive_prices(
            od, quote_reservation, quote_reservation, buy_edge, sell_edge
        )
        if dislocation_side < 0 and has_ask:
            passive_sell = min(passive_sell, min(od.sell_orders) - 1)
            if passive_buy >= passive_sell:
                passive_buy = passive_sell - 1
        elif dislocation_side > 0 and has_bid:
            passive_buy = max(passive_buy, max(od.buy_orders) + 1)
            if passive_buy >= passive_sell:
                passive_sell = passive_buy + 1

        outer_bid_volume = od.buy_orders[min(od.buy_orders)] if has_bid else 0
        outer_ask_volume = -od.sell_orders[max(od.sell_orders)] if has_ask else 0
        adverse_sell_sweep = (
            dislocation_side == 0
            and l1_signal > self.OSMIUM_SWEEP_L1_THRESH
            and micro_signal > self.OSMIUM_SWEEP_L1_THRESH
            and deep_signal > -self.OSMIUM_SWEEP_DEEP_CONFIRM
            and outer_ask_volume >= self.OSMIUM_SWEEP_WALL_VOLUME
        )
        adverse_buy_sweep = (
            dislocation_side == 0
            and l1_signal < -self.OSMIUM_SWEEP_L1_THRESH
            and micro_signal < -self.OSMIUM_SWEEP_L1_THRESH
            and deep_signal < self.OSMIUM_SWEEP_DEEP_CONFIRM
            and outer_bid_volume >= self.OSMIUM_SWEEP_WALL_VOLUME
        )
        if adverse_sell_sweep and has_ask:
            passive_sell = max(passive_sell, min(od.sell_orders))
        if adverse_buy_sweep and has_bid:
            passive_buy = min(passive_buy, max(od.buy_orders))

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

        if adverse_sell_sweep:
            remaining_sell = int(round(remaining_sell * self.OSMIUM_SWEEP_SIZE_MULT))
        if adverse_buy_sweep:
            remaining_buy = int(round(remaining_buy * self.OSMIUM_SWEEP_SIZE_MULT))

        if dislocation_side < 0:
            min_sell = int(round((limit + position) * self.OSMIUM_DISLOCATION_MIN_FRAC))
            remaining_sell = max(remaining_sell, min_sell)
            remaining_buy = int(round(remaining_buy * self.OSMIUM_DISLOCATION_OPPOSITE_MULT))
        elif dislocation_side > 0:
            min_buy = int(round((limit - position) * self.OSMIUM_DISLOCATION_MIN_FRAC))
            remaining_buy = max(remaining_buy, min_buy)
            remaining_sell = int(round(remaining_sell * self.OSMIUM_DISLOCATION_OPPOSITE_MULT))

        if round_side > 0:
            if dislocation_side >= 0:
                min_buy = int(round((limit - position) * self.OSMIUM_ABS_INV_FRAC))
                remaining_buy = max(remaining_buy, min_buy)
                remaining_sell = int(round(remaining_sell * self.OSMIUM_ABS_OPPOSITE_MULT))
            else:
                remaining_sell = int(round(remaining_sell * self.OSMIUM_ABS_CONFLICT_MULT))
        elif round_side < 0:
            if dislocation_side <= 0:
                min_sell = int(round((limit + position) * self.OSMIUM_ABS_INV_FRAC))
                remaining_sell = max(remaining_sell, min_sell)
                remaining_buy = int(round(remaining_buy * self.OSMIUM_ABS_OPPOSITE_MULT))
            else:
                remaining_buy = int(round(remaining_buy * self.OSMIUM_ABS_CONFLICT_MULT))

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
            f"rfac={range_factor:.2f} wdev={wall_dev:.1f} hot={hot:.2f} "
            f"rdev={round_dev:.1f} rside={round_side} "
            f"rev={reversion_bias:.2f} l1={l1_signal:.2f} deep={deep_signal:.2f} "
            f"swpS={int(adverse_sell_sweep)} swpB={int(adverse_buy_sweep)} "
            f"fillB={saved['osmium_buy_fill_quality']:.2f} "
            f"fillS={saved['osmium_sell_fill_quality']:.2f} tox={toxicity:.2f} "
            f"reg={regime} pb={passive_buy} ps={passive_sell} pos={position}"
        )

        return orders
