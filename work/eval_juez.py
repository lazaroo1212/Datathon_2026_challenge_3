"""eval_juez.py — Juez canónico del modo estándar (David, 15/11).

UNA sola fuente de verdad para comparar variantes. Reporta DOS protocolos:
  1. POR-VENTANA (canónico, replica el servidor): cada semana se simula por
     separado con SOC inicial 50% — como el enunciado ("SOC_0 = 50% a l'inici
     de cada període d'avaluació") y como la plataforma evalúa (12 semanas
     independientes; la máscara descubierta puntúa las primeras 84 h de cada
     una). El offset servidor-proxy medido con 4 envíos: +0.31 (6.12 → 6.43).
  2. CONCATENADO (informativo): una sola simulación continua, el SOC arrastra
     entre ventanas. Es el que usaron algunos experimentos previos (48h, 5
     mecanismos de Pei) — números NO comparables con el por-ventana.

Criterio de promoción a envío: +0.02 en la MEDIA POR-VENTANA del agente de
referencia, con las mismas ventanas (días 18–24 de cada mes con datos).

Uso:  python3 work/eval_juez.py <agent.py> [referencia]
      python3 work/eval_juez.py work/mpc_agent.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

REF = {"6.12": "work/mpc_agent.py (agente repo, config congelada)"}


def cargar(ruta: str):
    spec = importlib.util.spec_from_file_location("agente_juez", Path(ruta).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatteryAgent


def ventanas(df):
    out = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= 18) & (g.index.day <= 24)]
        if len(w) >= 100:
            out.append((f"{y}-{m:02d}", w))
    return out


def evaluar(agent_cls, df):
    wins = ventanas(df)
    # 1) por-ventana (canónico)
    por_ventana = []
    infr_pv = 0
    for nombre, w in wins:
        env = CampusBatteryEnv(w.copy().reset_index())
        obs = env.reset()
        done = False
        ag = agent_cls()
        while not done:
            p = ag.act(obs)
            obs, done = env.step(p)
        r = env.evaluate_results()
        por_ventana.append((nombre, r["score"]))
        infr_pv += r["infraccions"]
    media_pv = float(np.mean([s for _, s in por_ventana]))

    # 2) concatenada (informativa)
    concat = pd.concat([w for _, w in wins])
    env = CampusBatteryEnv(concat.copy().reset_index())
    obs = env.reset()
    done = False
    ag = agent_cls()
    while not done:
        p = ag.act(obs)
        obs, done = env.step(p)
    r_cat = env.evaluate_results()

    return media_pv, por_ventana, r_cat, infr_pv


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "work" / "mpc_agent.py")
    df = pd.read_csv(ROOT / "work" / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
    df = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    ag_cls = cargar(ruta)

    media_pv, por_ventana, r_cat, infr = evaluar(ag_cls, df)

    print(f"agente: {ruta}")
    print("\n=== 1) MEDIA POR-VENTANA (canónico — replica servidor) ===")
    for nombre, s in por_ventana:
        print(f"  {nombre}: {s:.2f}")
    print(f"  >>> MEDIA = {media_pv:.3f}   (infracciones acumuladas: {infr})")
    print("      referencia histórica: 6.12 (agente repo) · servidor 6.43 (offset +0.31)")

    print("\n=== 2) CONCATENADA (informativa — NO comparar con el canónico) ===")
    print(f"  score = {r_cat['score']:.3f}   (infracciones: {r_cat['infraccions']})")
    print("\nCRITERIO DE ENVÍO: >= +0.02 sobre la media por-ventana del agente de referencia.")
