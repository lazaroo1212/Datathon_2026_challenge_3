"""Sweep horizon extension (h_extra) and terminal value (kappa)."""
import sys, time
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
    env = CampusBatteryEnv(w)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env.evaluate_results()

t0 = time.time()
base = np.mean([run(BatteryAgent(), w)["score"] for _, w in windows])
tb = time.time() - t0
print(f"baseline h_extra=0: {base:.3f}  ({tb/len(windows):.2f}s/window)")

for h_extra in [12, 24, 36]:
    for kappa in [0.0, 0.3]:
        scores, infr = [], 0
        t0 = time.time()
        for _, w in windows:
            a = BatteryAgent()
            a.h_extra, a.kappa = h_extra, kappa
            r = run(a, w)
            scores.append(r["score"]); infr += r["infraccions"]
        m = np.mean(scores)
        print(f"h_extra={h_extra:<3} kappa={kappa:<4}: mean {m:.3f}  delta {m-base:+.3f}  "
              f"infr {infr}  ({(time.time()-t0)/len(windows):.2f}s/window)")

print("\n--- persistence blend sweep (kappa=0) ---")
for mix in [0.0, 0.3, 0.5, 0.7, 1.0]:
    for h_extra in [24, 36]:
        scores, infr = [], 0
        for _, w in windows:
            a = BatteryAgent()
            a.h_extra, a.kappa, a.ext_mix = h_extra, 0.0, mix
            r = run(a, w)
            scores.append(r["score"]); infr += r["infraccions"]
        m = np.mean(scores)
        print(f"mix={mix:<4} h_extra={h_extra:<3}: mean {m:.3f}  delta {m-base:+.3f}  infr {infr}")
