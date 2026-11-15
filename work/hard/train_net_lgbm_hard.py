"""train_net_lgbm_hard.py — jugada 2: LGBM de load y pv_hard con features ALINEADAS
al runtime (corrige la causa del NO-GO del LGBM solar).

Clave del alineamiento:
  - En runtime el agente construye: pv_lag24 = pv_hist.get(T-24, ppT) → cuando no hay
    historial (inicio de simulación o si el servidor reinicia el agente por semana)
    el feature es el PERFIL de la hora objetivo. En train_v1 el ridge entrenó con el
    lag real de la serie → mismatch. Aquí: 70% de filas con lag real, 30% con el
    fallback de perfil (idéntico al servido) — el modelo aprende ambos regímenes.
  - Targets: load(t+k) y pv_hard(t+k) separados; net = lf − pf sin recortar (v2).
  - Honest: features t con día <= 17 y targets t+k con día <= 17.

Salida: work/hard/v3_net/{agent.py, params.json, lgbm_load.pkl, lgbm_pv.pkl, features.json}
Uso:  .venv/bin/python work/hard/train_net_lgbm_hard.py
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # work/hard/
ROOT = HERE.parent.parent
PVS = 12.0

df = pd.read_csv(ROOT / "work" / "master_dataset.csv", parse_dates=["Fecha"]).set_index("Fecha")
df["pv_hard"] = df["pv_kw"] * PVS
load_map = df["load_kw"].to_dict()
pv_map = df["pv_hard"].to_dict()

p = json.load(open(HERE / "v2_sep" / "params.json"))
PL = {(mi, h, d): float(v) for mi, mv in enumerate(p["PL"])
      for h, (d0, d1) in enumerate(mv) for d, v in ((0, d0), (1, d1))}
PP = {(mi, h, d): float(v) for mi, mv in enumerate(p["PP"])
      for h, (d0, d1) in enumerate(mv) for d, v in ((0, d0), (1, d1))}
KS = range(1, 25)
rng = np.random.default_rng(0)


def build(target: str, dia_max: int):
    """target: 'load' o 'pv'. Filas para horizontes k=1..24 desde t (día <= dia_max),
    target t+k con día <= dia_max (honesto), 30% de filas con lags = perfil (alineado)."""
    rows, targets, horizons = [], [], []
    series = df["load_kw"] if target == "load" else df["pv_hard"]
    smap = series.to_dict()
    for t, row in df.iterrows():
        if t.day > dia_max:
            continue
        if row["pv_kw_imputed_frac"] >= 0.5 or row["load_kw_imputed_frac"] >= 0.5:
            continue
        s_t = smap.get(t, np.nan)
        s_m3 = float(np.mean([smap.get(t - pd.Timedelta(hours=i), np.nan) for i in (0, 1, 2)]))
        s_l24 = smap.get(t - pd.Timedelta(hours=24), np.nan)
        fbase = {
            "mes": t.month, "dow": t.weekday(), "es_finde": int(t.weekday() >= 5),
            "hora_sin": np.sin(2 * np.pi * t.hour / 24),
            "hora_cos": np.cos(2 * np.pi * t.hour / 24),
            "s_t": s_t, "s_mean3": s_m3,
            "irr_t": row.get("irradiance", np.nan),
            "irr_mean3": float(df.loc[t - pd.Timedelta(hours=2): t, "irradiance"].mean())
            if t - pd.Timedelta(hours=2) in df.index else np.nan,
            "temp_panel_t": row.get("temp_panel", np.nan),
            "temp_amb_t": row.get("temp_ambient", np.nan),
            "wind_t": row.get("wind_speed", np.nan),
        }
        for k in KS:
            tk = t + pd.Timedelta(hours=k)
            if tk not in df.index or tk.day > dia_max:
                continue
            yrow = df.loc[tk]
            if yrow["pv_kw_imputed_frac"] >= 0.5 or yrow["load_kw_imputed_frac"] >= 0.5:
                continue
            if target == "load":
                prof_tk = PL.get((tk.month - 1, tk.hour, 1 if tk.weekday() >= 5 else 0), np.nan)
            else:
                prof_tk = PP.get((tk.month - 1, tk.hour, 1 if tk.weekday() >= 5 else 0), np.nan)
            f = dict(fbase)
            f["horizon"] = k
            f["hora_tk"] = tk.hour
            f["prof_tk"] = prof_tk
            # ── alineación train/servido: 30% de filas con lag = perfil del target ──
            fallback = rng.random() < 0.30
            f["s_lag24"] = prof_tk if (fallback or np.isnan(s_l24)) else s_l24
            rows.append(f)
            targets.append(float(yrow["load_kw"] if target == "load" else yrow["pv_hard"]))
            horizons.append(k)
    return pd.DataFrame(rows), pd.Series(targets), horizons


print("construyendo train (días 1–17)...")
Xt_l, yt_l, _ = build("load", 17)
Xt_p, yt_p, _ = build("pv", 17)
print("construyendo eval (días 18–24, target dentro de la semana)...")
Xe_l, ye_l, _ = build("load", 24)
Xe_p, ye_p, _ = build("pv", 24)

FEATS = list(Xt_l.columns)
Xt_l, Xe_l = Xt_l.fillna(-1.0), Xe_l.fillna(-1.0)
Xt_p, Xe_p = Xt_p.fillna(-1.0), Xe_p.fillna(-1.0)

import lightgbm as lgb

res = {}
for name, Xt, yt, Xe, ye in [("load", Xt_l, yt_l, Xe_l, ye_l),
                             ("pv", Xt_p, yt_p, Xe_p, ye_p)]:
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=48,
                          min_child_samples=40, verbose=-1, random_state=0)
    m.fit(Xt, yt)
    mae = float(np.mean(np.abs(ye - m.predict(Xe))))
    res[name] = (m, mae)
    print(f"LGBM {name}: MAE eval = {mae:.1f} kW ({len(Xe)} filas)")

out = HERE / "v3_net"
out.mkdir(exist_ok=True)
joblib.dump(res["load"][0], out / "lgbm_load.pkl")
joblib.dump(res["pv"][0], out / "lgbm_pv.pkl")
(out / "features.json").write_text(json.dumps({"features": FEATS}, ensure_ascii=False, indent=1))
shutil.copy(HERE / "v2_sep" / "agent.py", out / "agent.py")
shutil.copy(HERE / "v2_sep" / "params.json", out / "params.json")
print(f"guardado en {out}/ — parchear agent.py (jugada 2)")
