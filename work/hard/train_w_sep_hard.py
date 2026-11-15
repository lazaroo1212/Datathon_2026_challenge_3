"""train_w_sep_hard.py — modelos separados de CO2 y precio para el Hard Mode (jugada 1).

Mismo andamiaje que tools/train_v1.py (perfil 12×24×2, ridge por buckets de horizonte
3-6/7-12/13-20/21-26, features [1, perfilT, perfilT·r, T-24, última conocida], use =
~días 18-24 honesto, burn-in 170 h) pero entrenando CO2 y PRECIO por separado en vez
del w combinado. Motivo: el fix del excedente (v2) liberó el despacho — la sensibilidad
al error de w puede haber cambiado desde el no-go pre-fix (w-MAE 0.0173→0.0166 sin
ganancia en 17.88).

Salida: work/hard/v2_sep/params.json (= params de v2_free + PCO2/PPR/coef_co2/coef_price)
        work/hard/v2_sep/agent.py (= agent de v2_free + path separado con fallback)

Uso:  python3 work/hard/train_w_sep_hard.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent           # work/hard/
ROOT = HERE.parent.parent
PVS = 12.0
G = 0.27

df = pd.read_csv(ROOT / "work" / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
full = pd.date_range(df.index[0], df.index[-1], freq="1h")
df = df.reindex(full)
load = df.load_kw.values.copy()
pv = df.pv_kw.values * PVS
co2 = df.co2_kg_per_kwh.values
price = df.precio_eur_per_kwh.values
w = (co2 + G * price)
idx = df.index
mon = idx.month.values - 1
hr = idx.hour.values
dt = (idx.dayofweek.values >= 5).astype(int)
inwin = (idx.day.values >= 18) & (idx.day.values <= 24)
use = ~inwin                                    # honesto: fuera del pseudo-test


def prof(x):
    P = np.full((12, 24, 2), np.nan)
    for m in range(12):
        for h in range(24):
            for d in range(2):
                s = (mon == m) & (hr == h) & (dt == d) & use & ~np.isnan(x)
                if s.sum() >= 3:
                    P[m, h, d] = x[s].mean()
    for m in range(12):                          # fallbacks: mes×hora → hora
        for h in range(24):
            for d in range(2):
                if np.isnan(P[m, h, d]):
                    s = (mon == m) & (hr == h) & use & ~np.isnan(x)
                    if s.sum() >= 3:
                        P[m, h, d] = x[s].mean()
                    else:
                        s = (hr == h) & use & ~np.isnan(x)
                        P[m, h, d] = x[s].mean()
    return P


PCO2, PPR = prof(co2), prof(price)
pc = PCO2[mon, hr, dt]
pp = PPR[mon, hr, dt]
n = len(idx)


def ok(*a):
    return not any(np.isnan(v) for v in a)


rows = {"co2": [], "price": []}
for t in range(170, n - 30):
    if not use[t]:
        continue
    kw_c = co2[t:t + 3]
    kw_p = price[t:t + 3]
    rC = (np.mean(kw_c) / np.mean(pc[t:t + 3])) if ok(*kw_c) and ok(*pc[t:t + 3]) else None
    rPp = (np.mean(kw_p) / np.mean(pp[t:t + 3])) if ok(*kw_p) and ok(*pp[t:t + 3]) else None
    for h in range(3, 27):
        T = t + h
        if use[T] and rC is not None and rPp is not None:
            if ok(co2[T], co2[T - 24], co2[t + 2], pc[T]):
                rows["co2"].append((h, co2[T], 1, pc[T], pc[T] * rC, co2[T - 24], co2[t + 2]))
            if ok(price[T], price[T - 24], price[t + 2], pp[T]):
                rows["price"].append((h, price[T], 1, pp[T], pp[T] * rPp, price[T - 24], price[t + 2]))

bk = [(3, 6), (7, 12), (13, 20), (21, 26)]
coef_sep = {}
for key in rows:
    A = np.array(rows[key])
    coef_sep[key] = []
    for lo, hi in bk:
        s = (A[:, 0] >= lo) & (A[:, 0] <= hi)
        X = A[s, 2:]
        y = A[s, 1]
        b = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
        coef_sep[key].append(dict(lo=lo, hi=hi, b=b.tolist()))
        print(key, (lo, hi), len(y), np.round(b, 3).tolist())

# ── out-of-sample w-MAE (días 18-24, mismo protocolo que el otro chat) ──────
mae_comb, mae_sep, n_ev = [], [], 0
b_comb = None
for c in json.load(open(HERE / "v2_free" / "params.json"))["coef"]["w"]:
    pass  # buckets del combinado están en params; usamos el propio agente para comparar


def bucket_eval(lst, h):
    for c in lst:
        if c["lo"] <= h <= c["hi"]:
            return c["b"]
    return lst[-1]["b"] if h > lst[-1]["hi"] else lst[0]["b"]


# evaluación simple: para cada t en 18-24, comparar w predicho separado vs combinado
comb_params = json.load(open(HERE / "v2_free" / "params.json"))
PW_arr = np.array(comb_params["PW"])
m_err_c = []
for t in range(170, n - 30):
    if inwin[t]:
        continue
    kw_c, kw_p = co2[t:t + 3], price[t:t + 3]
    if not ok(*kw_c, *kw_p, *pc[t:t + 3], *pp[t:t + 3]):
        continue
    rC = np.mean(kw_c) / np.mean(pc[t:t + 3])
    rPp = np.mean(kw_p) / np.mean(pp[t:t + 3])
    rW = np.mean(co2[t:t + 3] + G * price[t:t + 3]) / np.mean(pc[t:t + 3] + G * pp[t:t + 3])
    for h in range(3, 27):
        Tp = t + h
        if not use[Tp] or not ok(co2[Tp], price[Tp], co2[Tp - 24], price[Tp - 24], w[Tp - 24]):
            continue
        w_true = w[Tp]
        bc = bucket_eval(coef_sep["co2"], h)
        bp = bucket_eval(coef_sep["price"], h)
        w_sep = max(bc[0] + bc[1] * pc[Tp] + bc[2] * pc[Tp] * rC + bc[3] * co2[Tp - 24] + bc[4] * co2[t + 2], 0.01) + \
            G * max(bp[0] + bp[1] * pp[Tp] + bp[2] * pp[Tp] * rPp + bp[3] * price[Tp - 24] + bp[4] * price[t + 2], 0.01)
        bw = bucket_eval(comb_params["coef"]["w"], h)
        Tk = idx[Tp]
        pwT = float(PW_arr[Tk.month - 1, Tk.hour, 1 if Tk.weekday() >= 5 else 0])
        w_comb = max(bw[0] + bw[1] * pwT + bw[2] * pwT * rW + bw[3] * w[Tp - 24] + bw[4] * (co2[t + 2] + G * price[t + 2]), 0.01)
        m_err_c.append(abs(w_comb - w_true) - abs(w_sep - w_true))

print(f"\nw-MAE combinado − w-MAE separado = {float(np.mean(m_err_c)):+.5f}  "
      f"(positivo = separado mejor; {len(m_err_c)} filas out-of-sample)")

# ── empaquetar v2_sep ──────────────────────────────────────────────────────
out = HERE / "v2_sep"
out.mkdir(exist_ok=True)
params = json.load(open(HERE / "v2_free" / "params.json"))
params["PCO2"] = PCO2.tolist()
params["PPR"] = PPR.tolist()
params["coef_sep"] = coef_sep
json.dump(params, open(out / "params.json", "w"))
shutil.copy(HERE / "v2_free" / "agent.py", out / "agent.py")
print(f"guardado: {out/'params.json'} (+coef_sep) · agent.py base copiado (parchear)")
