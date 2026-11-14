"""Evaluate LGBM agent vs profile baseline over pseudo-test windows, sweep alpha blend."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv  # noqa: E402
from lgbm_agent import LGBMAgent  # noqa: E402
from mpc_agent import BatteryAgent  # noqa: E402

df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")

windows = []
for (y, m), g in df.groupby([df.index.year, df.index.month]):
    w = g[(g.index.day >= 18) & (g.index.day <= 24)]
    if len(w) >= 100:
        windows.append((f"{y}-{m:02d}", w))


def run(agent_cls, w, **hp):
    agent = agent_cls()
    for k, v in hp.items():
        setattr(agent, k, v)
    env = CampusBatteryEnv(w)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env.evaluate_results()


# baseline reference
base = [run(BatteryAgent, w) for _, w in windows]
base_m = np.mean([r["score"] for r in base])
print(f"baseline (tables+memory): mean {base_m:.3f}")

import time
for alpha in [1.0, 0.75, 0.5, 0.25]:
    scores, infr = [], 0
    t0 = time.time()
    for _, w in windows:
        r = run(LGBMAgent, w, alpha=alpha)
        scores.append(r["score"])
        infr += r["infraccions"]
    m = np.mean(scores)
    print(f"lgbm alpha={alpha:<5}: mean {m:.3f}  delta {m - base_m:+.3f}  infr {infr}  "
          f"({(time.time()-t0)/len(windows):.1f}s/window)")
