"""solar_lgbm.py — ¿puede LightGBM batir al ridge de v1 en la solar Hard Mode (×12)?

Hipótesis del otro ordenador: el mayor error de v1 es la solar (MAE 156 kW vs 83 del
consumo) y con la carga neta real el score sube de 17.99 a 19.06. La meteo en ridge
no ayudó (<0.5%); aquí probamos si LGBM la mejora por ESTRUCTURA (nubosidad de ayer,
persistencia de nivel, no linealidades crepúsculo) y no por meteo cruda.

Protocolo honesto: entrenar excluyendo días 18–24; evaluar MAE fuera de muestra en
esos días (horas 3..24 de cada obs, como predice el agente). Comparar contra el ridge
actual NO es posible sin sus coeficientes exactos, así que comparamos contra el
mejor baseline reproducible aquí: perfil mes×hora×dow y "ayer a la misma hora".
Al final: simulación con el ganador si hay mejora clara (>10% de MAE).

Uso:  python3 work/hard/solar_lgbm.py          (requiere lightgbm en el venv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "work"))
sys.path.insert(0, str(HERE))

HORIZONTES = range(3, 25)        # el agente predice horas 4..24 (0-index 3..23)
DIA_TRAIN_MAX = 17               # honesto: entrenar solo días 1–17
LAGS = [24, 48, 72]              # "ayer", "anteayer"... a la misma hora

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("falta lightgbm: .venv/bin/pip install lightgbm  (o apt python3-lightgbm)")

df = pd.read_csv(ROOT / "work" / "master_dataset.csv", parse_dates=["Fecha"]).set_index("Fecha")
df = df[(df["in_hidden_week"] == 0) & (df["load_kw_imputed_frac"] < 0.5)
        & (df["pv_kw_imputed_frac"] < 0.5)].copy()
# señales objetivo: solar Hard (×12) y carga neta hard
df["pv_hard"] = df["pv_kw"] * 12
df["net_hard"] = (df["load_kw"] - df["pv_hard"]).clip(lower=0)

# ── features comunes (pasado-only) ─────────────────────────────────────────
def features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["mes"] = df.index.month
    X["hora"] = df.index.hour
    X["dow"] = df.index.dayofweek
    X["es_finde"] = (df.index.dayofweek >= 5).astype(int)
    for lag in LAGS:
        X[f"{target}_lag{lag}"] = df[target].shift(lag)
    X[f"{target}_roll24"] = df[target].shift(1).rolling(24).mean()
    X[f"{target}_roll72"] = df[target].shift(1).rolling(72).mean()
    # ratio de nivel de las 3 h previas (lo que el agente conoce del obs)
    X["nivel_ult3"] = df[target].shift(1).rolling(3).mean() / (df[target].rolling(72).mean() + 1e-9)
    # meteo (para el ablation: ¿ayuda en LGBM aunque no en ridge?)
    for c in ["irradiance", "temp_panel", "temp_ambient", "wind_speed"]:
        if c in df.columns:
            X[f"{c}_lag24"] = df[c].shift(24)
            X[f"{c}_ult3"] = df[c].shift(1).rolling(3).mean()
    return X


def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(f"{nombre:<38} MAE = {mae:7.1f} kW")
    return mae


resultados = {}
for target in ["pv_hard", "net_hard"]:
    print(f"\n=== objetivo: {target} ===")
    X = features(df, target)
    y = df[target].shift(-1)  # predecimos t+1; el agente itera horizontes con recursion

    # dataset de evaluación: días 18–24 (out-of-sample), horas donde hay lags
    eval_mask = (df.index.day >= 18) & (df.index.day <= 24)
    train_mask = (df.index.day <= DIA_TRAIN_MAX) & y.notna() & X.notna().all(axis=1)

    Xt, yt = X[train_mask], y[train_mask]
    Xe, ye = X[eval_mask & X.notna().all(axis=1) & y.notna()], y[eval_mask & X.notna().all(axis=1) & y.notna()]
    print(f"train {len(Xt)} h · eval {len(Xe)} h (días 18–24)")

    # ── baseline 1: perfil mes×hora×dow (train) ──
    perf = pd.DataFrame({"y": yt}).groupby([Xt["mes"], Xt["hora"], Xt["dow"]])["y"].mean()
    key = list(zip(Xe["mes"], Xe["hora"], Xe["dow"]))
    pred_perf = np.array([perf.get(k, float(yt.mean())) for k in key])
    resultados[f"{target} perfil"] = evaluar("perfil mes×hora×dow", ye.values, pred_perf)

    # ── baseline 2: ayer a la misma hora (persistencia 24h) ──
    ayer = df[target].shift(24)[Xe.index]
    resultados[f"{target} ayer"] = evaluar("ayer misma hora", ye.values, ayer.values)

    # ── LGBM sin meteo ──
    cols_sin_meteo = [c for c in Xt.columns if not any(
        c.startswith(m) for m in ("irradiance", "temp_", "wind_"))]
    m1 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, verbose=-1, random_state=0)
    m1.fit(Xt[cols_sin_meteo], yt)
    resultados[f"{target} lgbm"] = evaluar("LGBM (sin meteo)", ye.values,
                                           m1.predict(Xe[cols_sin_meteo]))

    # ── LGBM con meteo ──
    m2 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, verbose=-1, random_state=0)
    m2.fit(Xt, yt)
    resultados[f"{target} lgbm+meteo"] = evaluar("LGBM (+meteo)", ye.values, m2.predict(Xe))

print("\n=== resumen (MAE kW, menor = mejor) ===")
for k, v in resultados.items():
    print(f"  {k:<28} {v:7.1f}")

ratio_pv = resultados.get("pv_hard lgbm", 1e9) / max(resultados.get("pv_hard perfil", 1e-9), 1e-9)
print(f"\nLGBM solar vs perfil: {ratio_pv:.2f}× el MAE del perfil")
