"""Trace exact steps where env registers infractions."""
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
w = df[(df.index.year == 2026) & (df.index.month == 5) & (df.index.day >= 18) & (df.index.day <= 24)]

agent = BatteryAgent()
agent.gamma, agent.kappa, agent.beta = 0.18, 0.3, 0.4
env = CampusBatteryEnv(w)
obs = env.reset()
done = False
i = 0
while not done:
    a = agent.act(obs)
    obs, done = env.step(a)
    if env.infraccions_per_step[-1] > 0:
        eta = 0.9 ** 0.5
        p_req = env.actions[-1]
        print(f"step {i} ts={env.df['Fecha'].iloc[i]} soc_now={env.soc:.10f} p_req={p_req:.8f}")
        print(f"   obs soc_kwh was {a and ''}", obs)
        break
    i += 1
