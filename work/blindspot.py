"""blindspot.py — Cuantifica el punto ciego del horizonte de 24 h del MPC (David).

El LP solo ve w[t : t+24] (precio+CO2 exactos del obs). Si hay un pico de w en
t+24..t+47 mayor que todo lo visible, el agente reparte la batería dentro del
horizonte visible y llega "ciego" al pico real -> energía mal colocada.

Métricas:
  1. Frecuencia de horas "ciegas" (picos fuera de horizonte >> picos visibles).
  2. Δw típico por episodio (agrupa horas ciegas consecutivas = 1 pico lejano).
  3. Estimación SUPERIOR del score perdido por episodio y en total.

Conversión a puntos oficiales:
  score = 0.7*ΔCO2% + 0.3*ΔCost%  con  CO2_base = 71,816 kg / Cost_base = 118,746.34 €
  mover 1 kWh de un hora con w_a a otra con w_b (Δw = w_b - w_a) vale
  100 * (0.7/CO2_base) * Δw  puntos  (γ = 0.3*CO2_base/(0.7*Cost_base) ya integrado).

Uso:  python3 work/blindspot.py [--ratio 1.3] [--energy 1000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HOURLY = HERE / "hourly.csv"

GAMMA = 0.264          # indiferente (recta iso-score probada en servidor)
CO2_BASE = 71816.0     # kg CO2 baseline oficial (12 semanas test)
COST_BASE = 118746.34  # € baseline oficial
PTS_FACTOR = 100.0 * 0.7 / CO2_BASE   # puntos por kWh por unidad de Δw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1.3,
                    help="umbral: pico lejano > ratio * pico visible -> ciego")
    ap.add_argument("--energy", type=float, default=1000.0,
                    help="kWh reposicionables por episodio (cota superior; 1 ciclo medio)")
    args = ap.parse_args()

    df = pd.read_csv(HOURLY, parse_dates=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    df = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"]).reset_index(drop=True)
    w = (df["co2_kg_per_kwh"] + GAMMA * df["precio_eur_per_kwh"]).to_numpy()
    n = len(w)
    print(f"dataset: {n} h  ({df['Fecha'].iloc[0]} -> {df['Fecha'].iloc[-1]})")
    print(f"gamma={GAMMA}  ratio={args.ratio}  energia_reposicionable={args.energy} kWh\n")

    blind = np.zeros(n, dtype=bool)
    delta = np.zeros(n)
    for t in range(n - 48):
        near_max = w[t:t + 24].max()
        far_max = w[t + 24:t + 48].max()
        if far_max > args.ratio * near_max:
            blind[t] = True
            delta[t] = far_max - near_max

    pct = 100.0 * blind.sum() / max(1, n - 48)
    print(f"horas ciegas: {blind.sum()} / {n - 48}  ({pct:.1f} %)")

    # agrupar horas ciegas consecutivas -> episodios (un pico lejano dura <=24 h ciegas)
    episodes = []
    t = 0
    while t < n - 48:
        if blind[t]:
            t0 = t
            while t < n - 48 and blind[t]:
                t += 1
            episodes.append((t0, t, float(delta[t0:t].max())))
        else:
            t += 1

    print(f"episodios (picos lejanos): {len(episodes)}")
    if not episodes:
        print("=> SIN PUNTOS CIEGOS: el horizonte de 24 h cubre todos los picos relevantes.")
        sys.exit(0)

    # distribución por hora del día del inicio de episodio
    hod = df["Fecha"].iloc[[e[0] for e in episodes]].dt.hour
    print("\nepisodios por hora de inicio (0-23):")
    print(" ".join(f"{h:02d}:{c}" for h, c in enumerate(np.bincount(hod, minlength=24)) if c))

    # estimación superior de puntos perdidos
    lost = [(args.energy * d * PTS_FACTOR) for _, _, d in episodes]
    top = sorted(episodes, key=lambda e: -e[2])[:15]
    print("\ntop-15 episodios por Δw (t0, w_visible_max, w_lejano_max, Δw, puntos_perdidos~):")
    for t0, _, d in top:
        print(f"  {df['Fecha'].iloc[t0]}  w_vis={w[t0:t0+24].max():.4f}  "
              f"w_lejos={d + w[t0:t0+24].max():.4f}  Δw={d:.4f}  "
              f"~{args.energy * d * PTS_FACTOR:.3f} pts")

    print(f"\ncota superior total de puntos perdidos (si en cada episodio se recoloca "
          f"{args.energy:.0f} kWh): {sum(lost):.2f} pts")
    print("cota REALISTA: la fracción de energía que el LP actual realmente mal coloca "
          "es menor (ya descarga en picos moderados visibles); tomar 20-50% de la cota "
          "como expectativa del experimento 48h.")


if __name__ == "__main__":
    main()
