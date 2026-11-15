"""eval_juez_portatil.py — Juez canónico del modo estándar, versión PORTÁTIL.

Autónomo: no depende de la estructura del repo. Solo necesita:
  - environment.py (starter kit)  → --env
  - hourly.csv                    → --hourly
  - tu agent.py                   → --agent

Reporta DOS protocolos (NO mezclarlos al comparar):
  1. MEDIA POR-VENTANA (canónico — replica el servidor): cada semana se simula
     por separado con SOC inicial 50% (enunciado: "SOC_0 = 50% a l'inici de
     cada període d'avaluació"). Baseline de referencia: 6.118 con el agente
     del repo (config congelada de Pei). Offset servidor medido: +0.31 (→6.43).
  2. CONCATENADA (informativa): una simulación continua de ~1.440 h, el SOC
     arrastra entre ventanas. No es comparable con el canónico.

CRITERIO DE ENVÍO: media por-ventana >= 6.138 (= baseline 6.118 + 0.02).

Uso:
  python3 eval_juez_portatil.py --agent mi_agente.py \
      --hourly hourly.csv --env paquet_participants_datathon_upc/starter_kit
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="ruta a agent.py a evaluar")
    ap.add_argument("--hourly", default="hourly.csv", help="ruta a hourly.csv")
    ap.add_argument("--env", default="paquet_participants_datathon_upc/starter_kit",
                    help="directorio con environment.py del starter kit")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.env).resolve()))
    from environment import CampusBatteryEnv  # noqa: E402

    spec = importlib.util.spec_from_file_location("agente_juez", Path(args.agent).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AgentCls = mod.BatteryAgent

    df = pd.read_csv(args.hourly, parse_dates=["Fecha"]).set_index("Fecha")
    df = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])

    ventanas = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= 18) & (g.index.day <= 24)]
        if len(w) >= 100:
            ventanas.append((f"{y}-{m:02d}", w))
    print(f"ventanas: {[n for n, _ in ventanas]}")

    # 1) por-ventana (canónico)
    por_ventana, infr = [], 0
    for nombre, w in ventanas:
        env = CampusBatteryEnv(w.copy().reset_index())
        obs = env.reset()
        done = False
        ag = AgentCls()
        while not done:
            p = ag.act(obs)
            obs, done = env.step(p)
        r = env.evaluate_results()
        por_ventana.append((nombre, r["score"]))
        infr += r["infraccions"]
        print(f"  {nombre}: {r['score']:.2f} (infr {r['infraccions']})")
    media = float(np.mean([s for _, s in por_ventana]))
    print(f"\n>>> MEDIA POR-VENTANA = {media:.3f}   (infracciones: {infr})")
    print("    baseline referencia: 6.118 · criterio envío: >= 6.138 (+0.02)")
    print("    offset servidor medido: +0.31 (local 6.12 -> servidor 6.43)")

    # 2) concatenada (informativa)
    concat = pd.concat([w for _, w in ventanas])
    env = CampusBatteryEnv(concat.copy().reset_index())
    obs = env.reset()
    done = False
    ag = AgentCls()
    while not done:
        p = ag.act(obs)
        obs, done = env.step(p)
    r_cat = env.evaluate_results()
    print(f"\n[info] CONCATENADA = {r_cat['score']:.3f} (infr {r_cat['infraccions']}) "
          "— protocolo distinto, NO comparable con el canónico")


if __name__ == "__main__":
    main()
