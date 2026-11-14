"""retest48.py — David: re-test limpio del horizonte-48h SIN el artefacto de cola.

Problema del experimento original de Jan (PROGRESS_jan.md §3): las ventanas de
eval (días 18-24) truncan `preus_properes_24h` al llegar al día 24 porque los
días 25-31 NO existen en hourly.csv. Con extend=48, esas horas finales planifican
sobre w fabricado -> el LP retiene SOC para picos fantasma -> -0.59 contaminado.

En la plataforma real los datos son continuos (siempre hay 24 h reales por
delante). Réplica limpia: el env corre sobre días REALES 11-20 y solo se
puntúan los días 11-17 (mask de evaluate_results). Así el obs siempre tiene
24 h reales para cada hora puntuada, como en el servidor.

Variantes (mismo mask, mismo SOC inicial):
  A = agente actual (mpc_agent.py, tablas viejas 12x24x2, solo KNOWN)
  B = tablas ricas 12x24x7, solo KNOWN     (aisla el efecto tablas)
  C = tablas ricas + horizonte 48h + prof_w (el experimento en disputa)

Uso:  python3 work/retest48.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv  # noqa: E402
from mpc_agent import BatteryAgent as OldAgent  # noqa: E402
from mpc_agent_rich import BatteryAgent as RichAgent  # noqa: E402

SCORE_DAYS = set(range(11, 18))   # días puntuados: 11-17 (168 h/mes)
BLOCK_DAYS = set(range(11, 21))   # datos reales 11-20: continuación para el obs

df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")


def run(agent: object, block: pd.DataFrame) -> dict:
    env = CampusBatteryEnv(block.copy().reset_index())
    mask = env.df["Fecha"].dt.day.isin(SCORE_DAYS).to_numpy()
    obs = env.reset()
    done = False
    while not done:
        p = agent.act(obs)
        obs, done = env.step(p)
    res = env.evaluate_results(mask=mask)
    res["horas_puntuadas"] = int(mask.sum())
    return res


months = sorted({(t.year, t.month) for t in df.index if t.day in BLOCK_DAYS})
per_month = []
for (y, m) in months:
    block = df[(df.index.year == y) & (df.index.month == m)
               & (df.index.day.isin(BLOCK_DAYS))]
    if len(block) < 200:  # exige ~10 días completos (salta Nov25 ausente, Ago26 corto)
        continue
    a = run(OldAgent(), block)
    rb = RichAgent()
    rb.extend = False
    b = run(rb, block)
    rc = RichAgent()
    rc.extend = True
    c = run(rc, block)
    per_month.append((f"{y}-{m:02d}", a["score"], b["score"], c["score"]))
    print(f"{y}-{m:02d}:  A(actual)={a['score']:.2f}  B(ricas)={b['score']:.2f}  "
          f"C(48h)={c['score']:.2f}   [infr A/B/C: {a['infraccions']}/"
          f"{b['infraccions']}/{c['infraccions']}]")

if not per_month:
    sys.exit("sin bloques válidos")

arr = np.array([[a, b, c] for _, a, b, c in per_month])
mean = arr.mean(axis=0)
print(f"\n=== media sobre {len(per_month)} meses (días 11-17, continuación real) ===")
print(f"A actual (tablas viejas, KNOWN)     : {mean[0]:.3f}")
print(f"B tablas ricas, KNOWN               : {mean[1]:.3f}  (Δ vs A: {mean[1]-mean[0]:+.3f})")
print(f"C tablas ricas + 48h + prof_w       : {mean[2]:.3f}  (Δ vs B: {mean[2]-mean[1]:+.3f})")
