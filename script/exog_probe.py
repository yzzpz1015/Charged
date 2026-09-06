# -*- coding: utf-8 -*-
"""Exogenous TS (weather / price) vs charging volume relationship probe.

Methodological notes:
- Volume, weather, price all share the diurnal cycle -> raw correlation is
  dominated by the common schedule.  We report both RAW and CYCLE-ADJUSTED
  (subtract hour-of-day mean) values; the adjusted one isolates "unusual for
  this hour" increments.  MI is computed on the cycle-adjusted pair (BOTH
  sides) so the shared schedule is not counted as mutual information.
- Forecast relevance is measured as OOS R^2 increment over the target's OWN
  past (lags 1..3) — the decisive number, mirroring the neighbor probe.
- Weather is treated as KNOWN-FUTURE (forecast available): test weather at
  t+h vs volume at t+h.  Price is treated as UNKNOWN-FUTURE: only price[t-h]
  usable.
- Nonlinearity: distance correlation on the key same-time pairs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE = "/home/data/yzzpz/Charged/data"
CITIES = ["MEL", "SZH"]
EXOG = ["temp", "humidity", "price"]


def load(city):
    vol = pd.read_csv(f"{BASE}/{city}/volume.csv", index_col=0, parse_dates=True).astype(float)
    vol.columns = vol.columns.astype(int)
    w = pd.read_csv(f"{BASE}/{city}/weather.csv", index_col=0)
    w.index = pd.to_datetime(w.index, format="%Y/%m/%d %H:%M")
    w = w.sort_index()
    try:
        pri = pd.read_csv(f"{BASE}/{city}/e_price.csv", index_col=0, parse_dates=True).astype(float)
        pri.columns = pri.columns.astype(int)
    except FileNotFoundError:
        pri = None
    return vol, w, pri


def cycle_adjust(s):
    """Subtract hour-of-day mean (for a Series indexed by datetime)."""
    hm = s.groupby(s.index.hour).transform("mean")
    return s - hm


def week_cycle_adjust(s):
    """Subtract day-of-week x hour-of-day mean."""
    key = s.index.dayofweek * 24 + s.index.hour
    hm = s.groupby(key).transform("mean")
    return s - hm


def safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a, float)[m], np.asarray(b, float)[m]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0 or np.isnan(sa) or np.isnan(sb):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def binned_mi(x, y, nbins=8, seed=0, nmax=8000):
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    if len(x) > nmax:
        idx = rng.choice(len(x), nmax, replace=False)
        x, y = x[idx], y[idx]
    qx = np.asarray(pd.qcut(x, nbins, labels=False, duplicates="drop"), dtype=float)
    qy = np.asarray(pd.qcut(y, nbins, labels=False, duplicates="drop"), dtype=float)
    if np.isnan(qx).any() or np.isnan(qy).any():
        return np.nan
    k = int(max(qx.max(), qy.max())) + 1
    H = np.histogram2d(qx, qy, bins=[k, k])[0] + 1.0
    pxy = H / H.sum()
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    return float((pxy * np.log2(pxy / (px * py))).sum())


def dcor(x, y, nmax=800, seed=0):
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    if len(x) > nmax:
        idx = rng.choice(len(x), nmax, replace=False)
        x, y = x[idx], y[idx]
    x = x - x.mean()
    y = y - y.mean()
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])
    A = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    B = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
    dcov2 = float((A * B).mean())
    denom = np.sqrt(float((A * A).mean()) * float((B * B).mean()))
    return dcov2 / denom if denom > 0 else np.nan


def oos_increment(Xa, Xb, y, split):
    la = LinearRegression().fit(Xa[:split], y[:split])
    lb = LinearRegression().fit(Xb[:split], y[:split])
    return lb.score(Xb[split:], y[split:]) - la.score(Xa[split:], y[split:])


def main():
    for city in CITIES:
        vol, w, pri = load(city)
        cols = [c for c in vol.columns if vol[c].std() > 0]
        vol = vol[cols]
        T = len(vol)
        split = int(0.7 * T)
        wA = {k: cycle_adjust(w[k]) for k in ["temp", "humidity"]}
        wW = {k: week_cycle_adjust(w[k]) for k in ["temp", "humidity"]}
        vA = {s: cycle_adjust(vol[s]) for s in cols}
        priA = None
        if pri is not None:
            priA = {s: cycle_adjust(pri[s]) for s in cols if s in pri.columns}

        print(f"\n===== {city} ({len(cols)} sites, {T} steps) =====")

        # ---- 1. same-time correlation: raw / cycleAdj / weekAdj / MI ----
        print("\n[1] same-time corr (site-averaged): raw | cycleAdj(hour) | cycleAdj(hour*DoW) | MI(cycleAdj)")
        for name, k in [("temp", "temp"), ("humidity", "humidity")]:
            raw_src = w[k]
            cyc_src = wA[k]
            wkw = week_cycle_adjust(raw_src)
            pc_raw = np.nanmean([safe_corr(vol[s], raw_src) for s in cols])
            pc_cyc = np.nanmean([safe_corr(vA[s], cyc_src) for s in cols])
            pc_wwk = np.nanmean([safe_corr(week_cycle_adjust(vol[s]), wkw) for s in cols])
            mi = np.nanmean([binned_mi(vA[s].values, cyc_src.values) for s in cols])
            print(f"{name:<8} pear {pc_raw:.4f} | cycleAdj {pc_cyc:.4f} | week*hr {pc_wwk:.4f} | MI {mi:.4f}")
        if pri is not None:
            pcols = [s for s in cols if s in pri.columns]
            pr = np.nanmean([safe_corr(vol[s], pri[s]) for s in pcols])
            pc = np.nanmean([safe_corr(vA[s], priA[s]) for s in pcols])
            mi = np.nanmean([binned_mi(vA[s].values, priA[s].values) for s in pcols])
            print(f"{'price':<8} pear {pr:.4f} | cycleAdj {pc:.4f} | MI {mi:.4f}")

        # ---- 2. cycle-adjusted lagged corr (exog[t-h] -> vol[t]) ----
        print("\n[2] lagged corr (cycleAdj, site-averaged): exog[t-h] vs vol[t]")
        for name, src in [("temp", wA["temp"]), ("humidity", wA["humidity"])] + \
                ([] if priA is None else [("price", None)]):
            row = []
            for h in [0, 1, 6, 12, 24]:
                if src is None:
                    row.append(np.nanmean([safe_corr(vA[s].shift(h), priA[s]) for s in cols]))
                else:
                    row.append(np.nanmean([safe_corr(vA[s].shift(h), src) for s in cols]))
            print(f"{name:<8} " + " ".join(f"h={h}:{r:.3f}" for h, r in zip([0, 1, 6, 12, 24], row)))

        # ---- 3. forecast increment ----
        print("\n[3] OOS R2 increment over own past (lags 1..3)")
        # weather KNOWN-FUTURE: weather[t+h] -> vol[t+h]
        for h in [1, 2, 3, 6, 24]:
            incs = []
            for s in cols:
                y = vA[s].values[h:]
                Xa = np.stack([vA[s].shift(l).values[h:] for l in range(1, 4)], axis=1)
                Xb = np.concatenate([Xa, wA["temp"].shift(-h).values[h:][:, None]], axis=1)
                m = np.isfinite(Xb).all(1)
                incs.append(oos_increment(Xa[m], Xb[m], y[m], split))
            print(f"temp  known-future h={h:<2} incr {np.nanmean(incs):+.4f}  (weather[t+h] -> vol[t+h])")
        # price UNKNOWN-FUTURE: price[t-1..t-3] -> vol[t]
        if priA is not None:
            incs_p = []
            for s in cols:
                y = vA[s].values[1:]
                Xa = np.stack([vA[s].shift(l).values[1:] for l in range(1, 4)], axis=1)
                p1 = priA[s].shift(1).values[1:]
                Xb = np.concatenate([Xa, p1[:, None]], axis=1)
                m = np.isfinite(Xb).all(1)
                incs_p.append(oos_increment(Xa[m], Xb[m], y[m], split))
            print(f"price known-past-only h=1 incr {np.nanmean(incs_p):+.4f}  (price[t-1] -> vol[t])")
            # humidity known-future
            for h in [1, 6, 24]:
                incs = []
                for s in cols:
                    y = vA[s].values[h:]
                    Xa = np.stack([vA[s].shift(l).values[h:] for l in range(1, 4)], axis=1)
                    Xb = np.concatenate([Xa, wA["humidity"].shift(-h).values[h:][:, None]], axis=1)
                    m = np.isfinite(Xb).all(1)
                    incs.append(oos_increment(Xa[m], Xb[m], y[m], split))
                print(f"hum   known-future h={h:<2} incr {np.nanmean(incs):+.4f}  (humidity[t+h] -> vol[t+h])")

        # ---- 4. nonlinearity (dCor) on cycle-adjusted same-time ----
        print("\n[4] dCor vs |Pearson| (cycleAdj same-time)")
        for name, src in [("temp", wA["temp"]), ("humidity", wA["humidity"])] + \
                ([] if priA is None else [("price", None)]):
            if src is None:
                pc = np.nanmean([abs(safe_corr(vA[s], priA[s])) for s in cols])
                dc = np.nanmean([dcor(vA[s].values, priA[s].values) for s in cols])
            else:
                pc = np.nanmean([abs(safe_corr(vA[s], src)) for s in cols])
                dc = np.nanmean([dcor(vA[s].values, src.values) for s in cols])
            print(f"{name:<8} |pear| {pc:.4f} | dCor {dc:.4f} | ratio {dc / pc:.2f}")


if __name__ == "__main__":
    main()
