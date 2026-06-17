"""
Breakthrough-hunting analysis: look for signals NOT yet exploited by the
current trader. Tests correlations, predictive features, and trade-tape info
that could unlock material alpha.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

BASE = Path(__file__).resolve().parent / "data" / "round2"
DAYS = [("day-1", -1), ("day0", 0), ("day1", 1)]


def load_book(day_dir, day_num, product):
    rows = []
    with open(BASE / day_dir / f"prices_round_2_day_{day_num}.csv") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["product"] != product:
                continue
            ts = int(r["timestamp"])
            bids = []
            asks = []
            for i in (1, 2, 3):
                if r[f"bid_price_{i}"]:
                    bids.append((int(r[f"bid_price_{i}"]), int(r[f"bid_volume_{i}"])))
                if r[f"ask_price_{i}"]:
                    asks.append((int(r[f"ask_price_{i}"]), int(r[f"ask_volume_{i}"])))
            rows.append({
                "ts": ts, "bids": bids, "asks": asks,
                "mid": float(r["mid_price"]),
            })
    return rows


def load_trades(day_dir, day_num, product):
    out = defaultdict(list)
    with open(BASE / day_dir / f"trades_round_2_day_{day_num}.csv") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["symbol"] != product:
                continue
            out[int(r["timestamp"])].append((int(float(r["price"])), int(r["quantity"])))
    return out


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def analyze_day(day_dir, day_num):
    print(f"\n===== Day {day_num} =====")
    pep = {r["ts"]: r for r in load_book(day_dir, day_num, "INTARIAN_PEPPER_ROOT")}
    osm = {r["ts"]: r for r in load_book(day_dir, day_num, "ASH_COATED_OSMIUM")}
    osm_trades = load_trades(day_dir, day_num, "ASH_COATED_OSMIUM")

    common_ts = sorted(set(pep.keys()) & set(osm.keys()))
    print(f"Common timestamps: {len(common_ts)}")

    # Build mid series for both
    pep_mids = []
    osm_mids = []
    for ts in common_ts:
        p = pep[ts]["mid"]
        o = osm[ts]["mid"]
        if p > 1000 and o > 1000:  # skip zero/garbage
            pep_mids.append(p)
            osm_mids.append(o)

    # === 1. PEPPER vs OSMIUM correlation (levels) ===
    print(f"\n[1] PEPPER vs OSMIUM mid-level corr: {corr(pep_mids, osm_mids):+.3f}")

    # === 2. First differences ===
    pep_d = [pep_mids[i+1] - pep_mids[i] for i in range(len(pep_mids)-1)]
    osm_d = [osm_mids[i+1] - osm_mids[i] for i in range(len(osm_mids)-1)]
    print(f"[2] PEPPER ΔMid vs OSMIUM ΔMid (same step) corr: {corr(pep_d, osm_d):+.3f}")

    # === 3. Lagged: PEPPER Δt predicts OSMIUM Δt+1? ===
    if len(pep_d) > 1:
        print(f"[3] PEPPER Δt predicts OSMIUM Δ(t+1) corr: {corr(pep_d[:-1], osm_d[1:]):+.3f}")
        print(f"[3b] OSMIUM Δt predicts PEPPER Δ(t+1) corr: {corr(osm_d[:-1], pep_d[1:]):+.3f}")

    # === 4. Predicting OSMIUM next-step move from current book features ===
    X_obi = []   # order book imbalance
    X_micro_minus_mid = []  # microprice skew vs mid (momentum/toxicity proxy)
    X_l1_vol_imb = []  # (bid1_vol - ask1_vol)/(sum)
    X_deep_imb = []  # deep level (L2/L3) vol imbalance
    X_spread = []
    X_mid_vs_wall = []   # mid - wall_mid (how much mid deviates from outer-wall center)
    X_hot_raw = []       # signed hot signal
    Y_next = []  # osm mid change at t+1

    for i, ts in enumerate(common_ts[:-1]):
        o = osm[ts]
        o_next = osm.get(common_ts[i+1])
        if not o_next or o_next["mid"] < 1000 or o["mid"] < 1000:
            continue
        bids = o["bids"]; asks = o["asks"]
        if not bids or not asks:
            continue
        bids_s = sorted(bids, key=lambda x: -x[0])
        asks_s = sorted(asks, key=lambda x: x[0])
        best_bid, best_bid_v = bids_s[0]
        best_ask, best_ask_v = asks_s[0]
        mid = o["mid"]
        total_bv = sum(v for _, v in bids)
        total_av = sum(v for _, v in asks)
        obi = (total_bv - total_av) / (total_bv + total_av) if (total_bv + total_av) > 0 else 0
        microprice = (best_bid * best_ask_v + best_ask * best_bid_v) / (best_bid_v + best_ask_v) if (best_bid_v + best_ask_v) > 0 else mid

        wall_bid = bids_s[-1][0]
        wall_ask = asks_s[-1][0]
        wall_mid = (wall_bid + wall_ask) / 2

        l1_imb = (best_bid_v - best_ask_v) / (best_bid_v + best_ask_v) if (best_bid_v + best_ask_v) > 0 else 0

        # deep imbalance: volumes at L2+L3 on each side
        deep_bv = sum(v for _, v in bids_s[1:])
        deep_av = sum(v for _, v in asks_s[1:])
        deep_imb = (deep_bv - deep_av) / (deep_bv + deep_av) if (deep_bv + deep_av) > 0 else 0

        spread = best_ask - best_bid
        mid_vs_wall = mid - wall_mid  # positive = mid above outer center

        # hot signal (raw current-tick): L1-L2 gaps
        hot = 0
        if len(bids_s) >= 2:
            gap = bids_s[0][0] - bids_s[1][0]
            if gap >= 5:
                hot += 1
        if len(asks_s) >= 2:
            gap = asks_s[1][0] - asks_s[0][0]
            if gap >= 5:
                hot -= 1

        X_obi.append(obi)
        X_micro_minus_mid.append(microprice - mid)
        X_l1_vol_imb.append(l1_imb)
        X_deep_imb.append(deep_imb)
        X_spread.append(spread)
        X_mid_vs_wall.append(mid_vs_wall)
        X_hot_raw.append(hot)

        Y_next.append(o_next["mid"] - mid)

    print(f"\n[4] Predicting OSMIUM ΔMid (t→t+1), n={len(Y_next)}:")
    print(f"    OBI (total vol imb)         → corr={corr(X_obi, Y_next):+.3f}")
    print(f"    Microprice - Mid            → corr={corr(X_micro_minus_mid, Y_next):+.3f}")
    print(f"    L1 vol imbalance            → corr={corr(X_l1_vol_imb, Y_next):+.3f}")
    print(f"    Deep vol imbalance (L2+)    → corr={corr(X_deep_imb, Y_next):+.3f}")
    print(f"    Spread (near)               → corr={corr(X_spread, Y_next):+.3f}")
    print(f"    Mid - Wall_mid              → corr={corr(X_mid_vs_wall, Y_next):+.3f}")
    print(f"    Hot signal (signed)         → corr={corr(X_hot_raw, Y_next):+.3f}")

    # === 5. Mid reversion to wall_mid? ===
    # If mid > wall_mid (above outer center), does mid tend to drop next tick?
    deviations = [(x, y) for x, y in zip(X_mid_vs_wall, Y_next)]
    above = [y for x, y in deviations if x > 2]
    below = [y for x, y in deviations if x < -2]
    if above:
        print(f"[5a] When mid > wall+2 (n={len(above)}): avg next Δmid={mean(above):+.2f}")
    if below:
        print(f"[5b] When mid < wall-2 (n={len(below)}): avg next Δmid={mean(below):+.2f}")

    # === 6. Trade tape direction ===
    # For each tick with trades, classify trade price vs mid as buying/selling
    # aggressive (trade near ask = market buy, trade near bid = market sell)
    trade_signals = []  # signed by qty
    next_osm_moves = []
    for i, ts in enumerate(common_ts[:-1]):
        trades_here = osm_trades.get(ts, [])
        if not trades_here:
            continue
        o = osm[ts]
        if o["mid"] < 1000:
            continue
        mid = o["mid"]
        signed = 0
        for p, q in trades_here:
            if p > mid:
                signed += q
            elif p < mid:
                signed -= q
        o_next = osm.get(common_ts[i+1])
        if o_next and o_next["mid"] > 1000:
            trade_signals.append(signed)
            next_osm_moves.append(o_next["mid"] - mid)
    if trade_signals:
        print(f"\n[6] Trade imbalance (signed qty) predicting next Δmid, n={len(trade_signals)}:")
        print(f"    corr={corr(trade_signals, next_osm_moves):+.3f}")
        # aggressive buy (pos trade signal) → next mid?
        pos_sig = [y for x, y in zip(trade_signals, next_osm_moves) if x >= 3]
        neg_sig = [y for x, y in zip(trade_signals, next_osm_moves) if x <= -3]
        if pos_sig:
            print(f"    Strong buy flow (≥3) (n={len(pos_sig)}): avg next Δmid={mean(pos_sig):+.2f}")
        if neg_sig:
            print(f"    Strong sell flow (≤-3) (n={len(neg_sig)}): avg next Δmid={mean(neg_sig):+.2f}")

    # === 7. Longer-horizon reversion: mid_vs_wall predicts Δmid over K steps? ===
    for K in [3, 10, 30, 100]:
        ys = []
        xs = []
        for i in range(len(common_ts) - K):
            ts = common_ts[i]
            ts_k = common_ts[i + K]
            o = osm.get(ts); o_k = osm.get(ts_k)
            if not o or not o_k or o["mid"] < 1000 or o_k["mid"] < 1000 or not o["bids"] or not o["asks"]:
                continue
            bids_s = sorted(o["bids"], key=lambda x: -x[0])
            asks_s = sorted(o["asks"], key=lambda x: x[0])
            wall_mid = (bids_s[-1][0] + asks_s[-1][0]) / 2
            xs.append(o["mid"] - wall_mid)
            ys.append(o_k["mid"] - o["mid"])
        if xs:
            print(f"[7] K={K}: corr(mid-wall, ΔmidK)={corr(xs, ys):+.3f}, |mid-wall|>2 n={sum(1 for x in xs if abs(x)>2)}")


for day_dir, day_num in DAYS:
    analyze_day(day_dir, day_num)
