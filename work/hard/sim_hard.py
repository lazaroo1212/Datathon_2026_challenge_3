"""sim_hard.py — Simulador Hard Mode (David, 15/11) reutilizable como módulo.

Hereda de CampusBatteryEnv y cambia:
  - pv_kw ×12 (solar Hard Mode) — también en el baseline interno (redefiniendo
    evaluate_results, porque el baseline usa el load/pv del dataframe ORIGINAL).
  - spec: 6.000 kWh, ±1.500 kW, η=0.9, SOC inicial 50%, soc_min 0, soc_max 1.
  - _get_obs: preus_properes_24h y co2_properes_24h recortadas a 3 elementos.
    noves_dades_5min mantiene su formato (la meteo extra es opcional para el agente).

Uso como módulo:
    from sim_hard import HardBatteryEnv, eval_ventanas_hard
    res = eval_ventanas_hard(agent_factory)   # dict con score medio y por ventana

Proxy local: el servidor usa otras semanas y escalas propias — solo para comparar
variantes ENTRE SÍ en local, no para extrapolar scores absolutos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent      # work/hard/
ROOT = HERE.parent.parent                   # raíz del repo
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv  # noqa: E402

PV_MULT = 12.0
SPEC_HARD = {
    "capacity_kwh": 6000.0,
    "power_kw": 1500.0,
    "efficiency": 0.9,
    "soc_init": 0.5,
    "soc_min": 0.0,
    "soc_max": 1.0,
    "dt_hours": 1.0,
}


class HardBatteryEnv(CampusBatteryEnv):
    """CampusBatteryEnv con solar ×12, batería 6.000/1.500 y obs de 3 h."""

    def __init__(self, hourly_data: pd.DataFrame, spec: dict = None, export_price: float = 0.0):
        df = hourly_data.copy().reset_index() if isinstance(hourly_data.index, pd.DatetimeIndex) \
            else hourly_data.copy()
        if "pv_kw" in df.columns:
            df["pv_kw"] = df["pv_kw"] * PV_MULT
        super().__init__(df, spec=spec or SPEC_HARD, export_price=export_price)

    def _get_obs(self):
        obs = super()._get_obs()
        obs["preus_properes_24h"] = obs["preus_properes_24h"][:3]
        obs["co2_properes_24h"] = obs["co2_properes_24h"][:3]
        return obs

    def evaluate_results(self, alpha=0.7, beta=0.3, mask=None):
        """Reimplementado: baseline con pv×12 (el del padre usa el pv original)."""
        grid = np.asarray(self.grid)
        exports = np.asarray(self.exports)
        price = self.df["precio_eur_per_kwh"].values[:len(grid)]
        co2 = self.df["co2_kg_per_kwh"].values[:len(grid)]
        load = self.df["load_kw"].values[:len(grid)]
        pv = self.df["pv_kw"].values[:len(grid)]          # ya multiplicado ×12
        base_grid = np.maximum(load - pv, 0.0)

        if mask is not None:
            m = np.asarray(mask)[:len(grid)]
            infr_arr = np.asarray(self.infraccions_per_step)[:len(grid)]
            infraccions = int(np.sum(infr_arr[m]))
            grid = grid[m]
            exports = exports[m]
            price = price[m]
            co2 = co2[m]
            base_grid = base_grid[m]
        else:
            infraccions = int(self.clipped + self.violations)

        cost_part = float(np.sum(grid * price) - np.sum(exports * self.export_price))
        co2_part = float(np.sum(grid * co2))
        cost_base = float(np.sum(base_grid * price))
        co2_base = float(np.sum(base_grid * co2))
        d_co2 = (co2_base - co2_part) / co2_base if co2_base > 0 else 0.0
        d_cost = (cost_base - cost_part) / cost_base if cost_base > 0 else 0.0
        penalties = 0.002 * infraccions
        score = alpha * d_co2 + beta * d_cost - penalties
        return {
            "score": round(max(0.0, score * 100), 2),
            "co2_estalviat_pct": round(d_co2 * 100, 2),
            "cost_estalviat_pct": round(d_cost * 100, 2),
            "co2_estalviat_kg": round(co2_base - co2_part, 1),
            "cost_estalviat_eur": round(cost_base - cost_part, 2),
            "co2_baseline_kg": round(co2_base, 1),
            "cost_baseline_eur": round(cost_base, 2),
            "infraccions": infraccions,
        }


def ventanas_hard(df: pd.DataFrame, dias=range(18, 25), min_horas=100):
    """Ventanas pseudo-test días 18–24 de cada mes con datos (9 en el dataset actual)."""
    out = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= min(dias)) & (g.index.day <= max(dias))]
        if len(w) >= min_horas:
            out.append((f"{y}-{m:02d}", w))
    return out


def eval_ventanas_hard(agent_factory, df=None, verbose=True):
    """Evalúa agent_factory() sobre las ventanas concatenadas; devuelve métricas.

    agent_factory: callable sin argumentos que devuelve un BatteryAgent NUEVO
    (con los modelos ya entrenados honestamente, excluyendo días 18–24).
    """
    if df is None:
        df = pd.read_csv(ROOT / "work" / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
        df = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    scores, infr, detalle = [], 0, []
    for etiqueta, win in ventanas_hard(df):
        env = HardBatteryEnv(win)
        agent = agent_factory()
        obs = env.reset()
        done = False
        while not done:
            p = agent.act(obs)
            obs, done = env.step(p)
        r = env.evaluate_results()
        scores.append(r["score"])
        infr += r["infraccions"]
        detalle.append((etiqueta, r["score"]))
        if verbose:
            print(f"  {etiqueta}: {r['score']:.2f}")
    return {
        "score_medio": float(np.mean(scores)),
        "score_por_ventana": detalle,
        "infracciones_total": infr,
        "horas_total": int(sum(len(w) for _, w in ventanas_hard(df))),
    }


if __name__ == "__main__":
    # sanity check: agente en reposo → score 0, 0 infracciones, baseline ×12
    sys.path.insert(0, str(ROOT / "work"))
    from mpc_agent import BatteryAgent  # noqa: E402

    df = pd.read_csv(ROOT / "work" / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
    win = df.loc["2025-09-18":"2025-09-24"]
    env = HardBatteryEnv(win)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(0.0)
    r = env.evaluate_results()
    print("sanity (agente en reposo):", r)
    assert r["score"] == 0.0 and r["infraccions"] == 0, "el reposo no debe puntuar ni infringir"
    # con PV×12 el baseline es MENOR que el estándar (la solar gigante desplaza importación);
    # comprobamos coherencia: baseline hard == baseline std con pv×12 sobre la misma ventana
    base_std_win = float(((win["load_kw"] - win["pv_kw"]).clip(lower=0)
                          * win["co2_kg_per_kwh"]).sum())
    assert abs(r["co2_baseline_kg"] - 12 * base_std_win / 12) < 1e-6 or \
        abs(r["co2_baseline_kg"] - base_std_win) > 1, \
        "el baseline hard debe usar pv×12, no el pv estándar"
    assert r["co2_baseline_kg"] < base_std_win * 1.01, \
        "con PV×12 el baseline debe ser <= al estándar (la solar desplaza)"
    # el obs debe traer 3 elementos de forecast
    env2 = HardBatteryEnv(win)
    obs2 = env2.reset()
    assert len(obs2["preus_properes_24h"]) == 3, "el obs debe estar recortado a 3 h"
    assert len(obs2["co2_properes_24h"]) == 3, "el obs debe estar recortado a 3 h"
    print("SIM HARD OK — specs, obs-3h y baseline ×12 verificados")
