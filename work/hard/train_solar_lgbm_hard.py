"""train_solar_lgbm_hard.py v2 — LGBM directo multi-horizonte de solar Hard Mode.

Correcciones vs v1 de este script:
  - tk = t + k HORAS (no días). Para t en días 18–24, los targets con k grandes
    caen en días 25+ (imputados) → solo evaluamos horizontes cuyo target cabe
    dentro del día 24. El juez final de la variante será la SIMULACIÓN
    (tools/sim_hard.py), no el MAE directo.
  - Train honesto: features t con día <= 17 Y target t+k con día <= 17.
  - Eval: features t en días 18–24 con target t+k también en 18–24 (target real
    del train, nunca visto por el modelo).

Salida: work/hard/v1_lgbm/solar_lgbm.pkl + features.json (+ agent.py/params.json base).

Uso:  .venv/bin/python work/hard/train_solar_lgbm_hard.py
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # work/hard/
ROOT = HERE.parent.parent

df = pd.read_csv(ROOT / "work" / "master_dataset.csv", parse_dates=["Fecha"]).set_index("Fecha")
df["pv_hard"] = df["pv_kw"] * 12.0
pv_map = df["pv_hard"].to_dict()

# perfil PP del agente v1 (misma fuente que el ridge de la solar)
p = json.load(open(HERE / "v1_honest" / "params.json"))
PP = {(mi, h, d): float(v) for mi, mv in enumerate(p["PP"])
      for h, (d0, d1) in enumerate(mv) for d, v in ((0, d0), (1, d1))}
KS = range(1, 25)


def build_rows(dias_filter):
    rows, targets, horizons = [], [], []
    for t, row in df.iterrows():
        if not dias_filter(t):
            continue
        if row["pv_kw_imputed_frac"] >= 0.5 or row["load_kw_imputed_frac"] >= 0.5:
            continue
        pv_t = row["pv_hard"]
        pv_m3 = float(np.mean([pv_map.get(t - pd.Timedelta(hours=i), np.nan)
                               for i in (0, 1, 2)]))
        irr_m3 = float(df.loc[t - pd.Timedelta(hours=2): t, "irradiance"].mean())
        fbase = {
            "mes": t.month, "dow": t.weekday(), "es_finde": int(t.weekday() >= 5),
            "hora_sin": np.sin(2 * np.pi * t.hour / 24),
            "hora_cos": np.cos(2 * np.pi * t.hour / 24),
            "pv_t": pv_t,
            "pv_mean3": pv_m3,
            "pv_lag24": pv_map.get(t - pd.Timedelta(hours=24), np.nan),
            "irr_t": row.get("irradiance", np.nan),
            "irr_mean3": irr_m3,
            "temp_panel_t": row.get("temp_panel", np.nan),
            "temp_amb_t": row.get("temp_ambient", np.nan),
            "wind_t": row.get("wind_speed", np.nan),
        }
        for k in KS:
            tk = t + pd.Timedelta(hours=k)
            if tk not in df.index:
                continue
            if not dias_filter(tk):
                continue
            yrow = df.loc[tk]
            if yrow["pv_kw_imputed_frac"] >= 0.5:
                continue
            f = dict(fbase)
            f["horizon"] = k
            f["hora_tk"] = tk.hour
            f["pp_tk"] = PP.get((tk.month - 1, tk.hour, 1 if tk.weekday() >= 5 else 0), np.nan)
            rows.append(f)
            targets.append(float(yrow["pv_hard"]))
            horizons.append(k)
    return pd.DataFrame(rows), pd.Series(targets), horizons


train_filter = lambda t: t.day <= 17          # fuera del pseudo-test
eval_filter = lambda t: 18 <= t.day <= 24     # dentro (targets reales del train)

print("construyendo dataset de entrenamiento (días 1–17)...")
Xt, yt, _ = build_rows(train_filter)
print("construyendo dataset de evaluación (días 18–24, target dentro de la semana)...")
Xe, ye, he = build_rows(eval_filter)
Xt, Xe = Xt.fillna(-1.0), Xe.fillna(-1.0)
print(f"train: {len(Xt)} filas · eval: {len(Xe)} filas · features: {len(Xt.columns)}")

import lightgbm as lgb

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=48,
                          min_child_samples=40, verbose=-1, random_state=0)
model.fit(Xt, yt)
mae = float(np.mean(np.abs(ye - model.predict(Xe))))
print(f"MAE evaluación (días 18–24, target dentro de semana): {mae:.1f} kW")

# comparación inmediata: perfil PP (el feature central del ridge de v1)
pp_pred = np.array([PP.get((tk.month - 1, tk.hour, 1 if tk.weekday() >= 5 else 0))
                    for tk in [t + pd.Timedelta(hours=k) for t, k in
                               zip(Xe.index if hasattr(Xe, "index") else range(len(Xe)),
                                   [None] * len(Xe))]] if False else None)
# (la comparación justa contra el ridge se hace en SIMULACIÓN; aquí solo guardamos)

out = HERE / "v1_lgbm"
out.mkdir(exist_ok=True)
joblib.dump(model, out / "solar_lgbm.pkl")
(out / "features.json").write_text(json.dumps({"features": list(Xt.columns)},
                                              ensure_ascii=False, indent=1))
shutil.copy(HERE / "v1_honest" / "agent.py", out / "agent.py")
shutil.copy(HERE / "v1_honest" / "params.json", out / "params.json")
print(f"guardado: {out/'solar_lgbm.pkl'} · features.json · agent.py/params.json base copiados")
print("SIGUIENTE PASO: parchear el agent.py de v1_lgbm para usar el LGBM en el bloque solar")
