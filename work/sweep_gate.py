"""Sweep gated evening reserve (blindspot-only hedge)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv
from mpc_agent import BatteryAgent

df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
windows = []
for (y, m), g in df.groupby([df.index.year, df.index.month]):
    w = g[(g.index.day >= 18) & (g.index.day <= 24)]
    if len(w) >= 100:
        windows.append((f"{y}-{m:02d}", w))

def run(agent, w):
    env = CampusBatteryEnv(w); obs = env.reset(); done = False
    while not done: obs, done = env.step(agent.act(obs))
    return env.evaluate_results()

scores, infr = [], 0
for _, w in windows:
    r = run(BatteryAgent(), w)
    scores.append(r["score"]); infr += r["infraccions"]
base = np.mean(scores)
print(f"baseline: {base:.3f}  infr {infr}")

for mix in [0.0, 0.5, 1.0]:
    for ratio in [1.15, 1.25, 1.4]:
        for reserve in [500, 1000, 1500]:
            scores, infr = [], 0
            for _, w in windows:
                a = BatteryAgent()
                a.reserve_kwh, a.reserve_ratio, a.ext_mix = float(reserve), ratio, mix
                r = run(a, w)
                scores.append(r["score"]); infr += r["infraccions"]
            m = np.mean(scores)
            print(f"mix={mix:<4} ratio={ratio:<5} reserve={reserve:<5}: mean {m:.3f}  delta {m-base:+.3f}  infr {infr}")
