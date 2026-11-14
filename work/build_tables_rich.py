"""Rich forecast tables (month x hour x day-of-week) from the master dataset.

Same format as params.json (prof_net / prof_pv / gamma_default) but the third
axis is dayofweek (Mon=0 ... Sun=6) instead of the weekend flag, giving
(12, 24, 7) tables.  Only non-hidden-week data is used, weighted by the real
observation fraction (1 - imputed_frac) for load and PV, so fabricated hours
(Aug 17 - Sep 1 tail, hidden days 25-31, filled outages) get zero weight.

Output: work/params_rich.json  -> raw material for the 48h-horizon experiment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "master_dataset.csv", parse_dates=["Fecha"], index_col="Fecha")
df["net"] = df["load_kw"] - df["pv_kw"]
df["dow"] = df.index.dayofweek
df["wknd"] = (df["dow"] >= 5).astype(int)
df["w_real"] = (1 - df["load_kw_imputed_frac"]) * (1 - df["pv_kw_imputed_frac"])
df = df[df["in_hidden_week"] == 0]


def wmean_table(frame, weights, keys, dim3):
    """Weighted mean of 'net' keyed by month x hour x dim3. weights: Series aligned to frame."""
    sub = frame[weights > 0]
    w = weights[weights > 0]
    out = np.zeros((12, 24, dim3))
    cnt = np.zeros((12, 24, dim3))
    for (m, h, d), g in sub.groupby(keys):
        out[m - 1, h, d] = np.average(g["net"], weights=w.loc[g.index])
        cnt[m - 1, h, d] = len(g)
    return out, cnt


def fill_fallback(tab, cnt, has_dow, frame, weights):
    """Fill empty cells with progressively coarser means (dow -> weekend -> hour -> global)."""
    sub = frame[weights > 0]
    w = weights[weights > 0]
    filled = 0
    for m in range(12):
        for h in range(24):
            for d in range(tab.shape[2]):
                if cnt[m, h, d] > 0:
                    continue
                filled += 1
                sel = (sub.index.hour == h)
                if has_dow:
                    cand = sub[sel & (sub["dow"] == d)]
                    if len(cand):
                        tab[m, h, d] = np.average(cand["net"], weights=w.loc[cand.index])
                        continue
                cand = sub[sel & (sub["wknd"] == int(d >= 5))]
                if len(cand):
                    tab[m, h, d] = np.average(cand["net"], weights=w.loc[cand.index])
                    continue
                cand = sub[sel]
                tab[m, h, d] = np.average(cand["net"], weights=w.loc[cand.index]) if len(cand) else np.average(sub["net"], weights=w)
    return filled


ws = df["w_real"].copy()
real_mask = ws.notna() & (ws > 0)
ws = ws.fillna(0.0)

rich_net, cnt_net = wmean_table(df, ws * df["net"].notna(), ["month", "hour", "dow"], 7)
rich_pv, cnt_pv = wmean_table(df.assign(net=df["pv_kw"]), ws * df["pv_kw"].notna(), ["month", "hour", "dow"], 7)

oldp = json.loads((HERE / "params.json").read_text())
gamma = oldp.get("gamma_default", 0.264)
w_col = df["co2_kg_per_kwh"] + gamma * df["precio_eur_per_kwh"]
rich_w, cnt_w = wmean_table(df.assign(net=w_col), ws * w_col.notna(), ["month", "hour", "dow"], 7)

n_fb_net = fill_fallback(rich_net, cnt_net, True, df, ws)
n_fb_pv = fill_fallback(rich_pv, cnt_pv, True, df, ws)
n_fb_w = fill_fallback(rich_w, cnt_w, True, df.assign(net=w_col), ws)

hard_net, hcnt = wmean_table(df, ws.gt(0).astype(float), ["month", "hour", "dow"], 7)
hard_pv, _ = wmean_table(df.assign(net=df["pv_kw"]), ws.gt(0).astype(float), ["month", "hour", "dow"], 7)

params = {
    "prof_net": np.round(rich_net, 2).tolist(),
    "prof_pv": np.round(rich_pv, 2).tolist(),
    "prof_w": np.round(rich_w, 4).tolist(),
    "gamma_default": oldp.get("gamma_default", 0.264),
    "kappa": oldp.get("kappa", 0.0),
    "beta": oldp.get("beta", 0.4),
    "beta_mem": oldp.get("beta_mem", 0.6),
    "axis3": "dow (Mon=0)",
    "cnt_net": cnt_net.astype(int).tolist(),
}
(HERE / "params_rich.json").write_text(json.dumps(params, separators=(",", ":")))

real = df[real_mask].copy()
day = real.index.day
dim = real.index.days_in_month
pseudo_test = real[(day >= dim - 10) & (day < 25)]

old_net = np.array(oldp["prof_net"])
old_pv = np.array(oldp["prof_pv"])


def mae(tab, sub, is_pv=False, dow_axis=False):
    col = "pv_kw" if is_pv else "net"
    dcol = "dow" if dow_axis else "wknd"
    fc = [tab[m - 1, h, d] for m, h, d in zip(sub["month"], sub["hour"], sub[dcol])]
    return float(np.abs(np.asarray(fc) - sub[col]).mean())

print(f"rows kept (real, non-hidden): {real_mask.sum()} / {len(df)} non-hidden")
print(f"rich table (12,24,7) net: fallbacks={n_fb_net}  | pv: fallbacks={n_fb_pv}  | w: fallbacks={n_fb_w}")
print(f"cnt per cell: min={int(cnt_net[cnt_net>0].min())} median={int(np.median(cnt_net[cnt_net>0]))}")
print()
print("net-load forecast MAE (kW)   all-real   pseudo-test(days18-24)")
print(f"  old  month x hour x wknd : {mae(old_net, real):7.1f}   {mae(old_net, pseudo_test):7.1f}")
print(f"  rich month x hour x dow  : {mae(rich_net, real, dow_axis=True):7.1f}   {mae(rich_net, pseudo_test, dow_axis=True):7.1f}")
print(f"  hard-drop variant        : {mae(hard_net, real, dow_axis=True):7.1f}   {mae(hard_net, pseudo_test, dow_axis=True):7.1f}")
print()
print("pv forecast MAE (kW)")
print(f"  old                      : {mae(old_pv, real, True):7.1f}   {mae(old_pv, pseudo_test, True):7.1f}")
print(f"  rich                     : {mae(rich_pv, real, True, True):7.1f}   {mae(rich_pv, pseudo_test, True, True):7.1f}")
print()
print("saved params_rich.json")
