"""
Granular per-timestamp OSMIUM order book analysis on R2 data.
Looks for structural patterns: hot quotes, transient levels, wall stability,
fair value divergence, trade-direction predictive signals.
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent / "data" / "round2"
DAYS = [("day-1", -1), ("day0", 0), ("day1", 1)]

def load_prices(day_dir, day_num):
    rows = []
    path = BASE / day_dir / f"prices_round_2_day_{day_num}.csv"
    with open(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r["product"] != "ASH_COATED_OSMIUM":
                continue
            ts = int(r["timestamp"])
            bids = []
            asks = []
            for i in (1, 2, 3):
                bp = r[f"bid_price_{i}"]
                bv = r[f"bid_volume_{i}"]
                if bp:
                    bids.append((int(bp), int(bv)))
                ap = r[f"ask_price_{i}"]
                av = r[f"ask_volume_{i}"]
                if ap:
                    asks.append((int(ap), int(av)))
            rows.append((ts, bids, asks, float(r["mid_price"])))
    return rows

def load_trades(day_dir, day_num):
    path = BASE / day_dir / f"trades_round_2_day_{day_num}.csv"
    trades = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r["symbol"] != "ASH_COATED_OSMIUM":
                continue
            trades[int(r["timestamp"])].append((int(float(r["price"])), int(r["quantity"])))
    return trades

def analyze(rows, trades, label):
    print(f"\n=== {label} ===")
    n = len(rows)
    mids = [r[3] for r in rows]
    avg_mid = sum(mids) / n
    print(f"  rows={n}, mid range=[{min(mids):.1f}, {max(mids):.1f}], mean={avg_mid:.1f}")

    # Book structure stats
    near_spreads = []
    l1_l2_bid_gaps = []
    l1_l2_ask_gaps = []
    full_spreads = []  # outermost bid to outermost ask
    l1_bid_vols = []
    l1_ask_vols = []
    outer_bid_vols = []
    outer_ask_vols = []

    hot_bid_events = []  # (ts, gap, price)
    hot_ask_events = []
    missing_bid_side = 0
    missing_ask_side = 0

    for ts, bids, asks, mid in rows:
        bids_sorted = sorted(bids, key=lambda x: -x[0])  # high→low
        asks_sorted = sorted(asks, key=lambda x: x[0])   # low→high

        if not bids:
            missing_bid_side += 1
            continue
        if not asks:
            missing_ask_side += 1
            continue

        best_bid = bids_sorted[0][0]
        best_ask = asks_sorted[0][0]
        near_spreads.append(best_ask - best_bid)
        full_spreads.append(asks_sorted[-1][0] - bids_sorted[-1][0])
        l1_bid_vols.append(bids_sorted[0][1])
        l1_ask_vols.append(asks_sorted[0][1])
        outer_bid_vols.append(bids_sorted[-1][1])
        outer_ask_vols.append(asks_sorted[-1][1])

        if len(bids) >= 2:
            gap = bids_sorted[0][0] - bids_sorted[1][0]
            l1_l2_bid_gaps.append(gap)
            if gap >= 5:
                hot_bid_events.append((ts, gap, best_bid, bids_sorted[0][1]))
        if len(asks) >= 2:
            gap = asks_sorted[1][0] - asks_sorted[0][0]
            l1_l2_ask_gaps.append(gap)
            if gap >= 5:
                hot_ask_events.append((ts, gap, best_ask, asks_sorted[0][1]))

    def stats(lst, name):
        if not lst:
            return
        lst_sorted = sorted(lst)
        avg = sum(lst) / len(lst)
        print(f"  {name}: mean={avg:.2f}, p10={lst_sorted[len(lst)//10]:.0f}, "
              f"p50={lst_sorted[len(lst)//2]:.0f}, p90={lst_sorted[9*len(lst)//10]:.0f}, "
              f"max={max(lst):.0f}")

    stats(near_spreads, "near_spread")
    stats(full_spreads, "full_spread (outer)")
    stats(l1_l2_bid_gaps, "L1-L2 bid gap")
    stats(l1_l2_ask_gaps, "L1-L2 ask gap")
    stats(l1_bid_vols, "L1 bid vol")
    stats(l1_ask_vols, "L1 ask vol")
    stats(outer_bid_vols, "outer bid vol")
    stats(outer_ask_vols, "outer ask vol")
    print(f"  missing bid side ticks: {missing_bid_side}, missing ask: {missing_ask_side}")
    print(f"  hot bid events (L1-L2 >=5): {len(hot_bid_events)}")
    print(f"  hot ask events (L1-L2 >=5): {len(hot_ask_events)}")

    # Post-hot-event behavior: does mid revert after a hot bid?
    # For each hot bid at ts, look at mid at ts+500, ts+1000, ts+2000 ahead
    mid_by_ts = {r[0]: r[3] for r in rows}
    ts_list = sorted(mid_by_ts.keys())
    def get_future_mid(ts_now, horizon):
        target = ts_now + horizon
        # find next ts >= target
        for t in ts_list:
            if t >= target:
                return mid_by_ts[t]
        return None

    def analyze_reversion(events, label, direction):
        """direction: +1 for hot bid (expect reversion down), -1 for hot ask (expect up)."""
        if not events:
            print(f"  [{label}] no events")
            return
        deltas_by_h = defaultdict(list)
        for ts, gap, price, vol in events:
            mid_now = mid_by_ts.get(ts)
            if mid_now is None:
                continue
            for h in [100, 300, 500, 1000, 2000]:
                f_mid = get_future_mid(ts, h)
                if f_mid is not None:
                    # signed reversion: negative = reverts (for hot bid, price went down)
                    deltas_by_h[h].append(direction * (f_mid - mid_now))
        print(f"  [{label}] n={len(events)}, avg signed dMid (negative = reversion predicted):")
        for h in [100, 300, 500, 1000, 2000]:
            v = deltas_by_h[h]
            if v:
                avg = sum(v) / len(v)
                # count reversions
                rev = sum(1 for x in v if x < 0)
                print(f"    h={h}: avg={avg:+.2f}, reversion rate={rev}/{len(v)}={rev/len(v):.2%}")

    analyze_reversion(hot_bid_events, "hot bid (expect mid DOWN)", +1)
    analyze_reversion(hot_ask_events, "hot ask (expect mid UP)", -1)

    # L2 volume imbalance signal
    l2_imb_events = []
    for ts, bids, asks, mid in rows:
        if len(bids) < 2 or len(asks) < 2:
            continue
        bids_sorted = sorted(bids, key=lambda x: -x[0])
        asks_sorted = sorted(asks, key=lambda x: x[0])
        b2_vol = bids_sorted[1][1]
        a2_vol = asks_sorted[1][1]
        if b2_vol + a2_vol > 0:
            imb = (b2_vol - a2_vol) / (b2_vol + a2_vol)
            if abs(imb) > 0.4:
                l2_imb_events.append((ts, imb))
    print(f"  strong L2 imbalance events (|imb|>0.4): {len(l2_imb_events)}")

    # Large trade analysis
    total_trades = sum(len(v) for v in trades.values())
    trade_qtys = []
    for tlist in trades.values():
        for p, q in tlist:
            trade_qtys.append(q)
    if trade_qtys:
        trade_qtys.sort()
        print(f"  trades: n={len(trade_qtys)}, median qty={trade_qtys[len(trade_qtys)//2]}, "
              f"p90={trade_qtys[9*len(trade_qtys)//10]}, max={max(trade_qtys)}")


for day_dir, day_num in DAYS:
    rows = load_prices(day_dir, day_num)
    trades = load_trades(day_dir, day_num)
    analyze(rows, trades, f"Day {day_num}")
