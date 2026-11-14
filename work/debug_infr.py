"""Debug: find infraction hours + speed benchmark for MPC agent."""
import sys
import time
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
w = df[(df.index.year == 2026) & (df.index.month == 3) & (df.index.day >= 18) & (df.index.day <= 24)]

agent = BatteryAgent()
env = CampusBatteryEnv(w)
obs = env.reset()
done = False
t0 = time.time()
lat = []
while not done:
    t1 = time.time()
    action = agent.act(obs)
    lat.append(time.time() - t1)
    obs, done = env.step(action)
elapsed = time.time() - t0
res = env.evaluate_results()
print(f"window 168h: wall={elapsed:.2f}s  mean act={np.mean(lat)*1000:.2f}ms  max={np.max(lat)*1000:.2f}ms")
print("results:", res)

# find infractions
infr = np.asarray(env.infraccions_per_step)
idx = np.where(infr > 0)[0]
for i in idx:
    print(f"step {i}: ts={env.df['Fecha'].iloc[i]} action={env.actions[i]:.4f} soc_before~? infr={infr[i]}")
    # reconstruct soc before this step
    if i == 0:
        soc_b = env.soc_init
    else:
        soc_b = env.soc_series[i - 1]
    print(f"   soc_before={soc_b:.6f}  max_c={(2000-soc_b)/ETA_C if False else (2000-soc_b)/(0.9**0.5):.4f} max_d={soc_b*(0.9**0.5):.4f}")
    pc, pd_ = (env.actions[i], None)
print("actions near infraction:", [round(a, 3) for a in env.actions[max(0, idx[0]-2):idx[0]+3]] if len(idx) else "none")
