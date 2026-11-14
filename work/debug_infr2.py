"""Locate remaining infractions across pseudo-test windows."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv  # noqa: E402
from mpc_agent import BatteryAgent  # noqa: E402

df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")

windows = []
for (y, m), g in df.groupby([df.index.year, df.index.month]):
    w = g[(g.index.day >= 18) & (g.index.day <= 24)]
    if len(w) >= 100:
        windows.append((f"{y}-{m:02d}", w))

for name, w in windows:
    agent = BatteryAgent()
    agent.gamma, agent.kappa, agent.beta = 0.18, 0.3, 0.4
    env = CampusBatteryEnv(w)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    infr = np.asarray(env.infraccions_per_step)
    idx = np.where(infr > 0)[0]
    if len(idx):
        print(f"\n{name}: {len(idx)} infractions at steps {idx.tolist()}")
        ETA_C = ETA_D = 0.9 ** 0.5
        for i in idx:
            soc_b = env.soc_init if i == 0 else env.soc_series[i - 1]
            a = env.actions[i]
            max_c = (2000 - soc_b) / ETA_C
            max_d = soc_b * ETA_D
            print(f"  step {i} ts={env.df['Fecha'].iloc[i]} action={a:.6f} soc_before={soc_b:.8f} "
                  f"max_c={max_c:.6f} max_d={max_d:.6f} ok_c={a <= max_c + 1e-9} ok_d={-a <= max_d + 1e-9}")
        print("  score:", env.evaluate_results()["score"])
    else:
        print(f"{name}: clean, score {env.evaluate_results()['score']}")
