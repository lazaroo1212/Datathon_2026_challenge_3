"""Hyperparameter sweep for the MPC agent over pseudo-test windows."""
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


def pseudo_test_windows():
    windows = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= 18) & (g.index.day <= 24)]
        if len(w) >= 100:
            windows.append((f"{y}-{m:02d}", w))
    return windows


def eval_window(agent_cls, w, **hp):
    agent = agent_cls()
    for k, v in hp.items():
        setattr(agent, k, v)
    env = CampusBatteryEnv(w)
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, done = env.step(action)
    return env.evaluate_results()


if __name__ == "__main__":
    windows = pseudo_test_windows()
    print(f"{len(windows)} windows: {[n for n, _ in windows]}")

    grid = []
    for gamma in [0.10, 0.18, 0.264, 0.35, 0.50]:
        for kappa in [0.0, 0.3, 0.5]:
            grid.append({"gamma": gamma, "kappa": kappa})

    results = []
    for hp in grid:
        scores, imgs = [], []
        for name, w in windows:
            r = eval_window(BatteryAgent, w, **hp)
            scores.append(r["score"])
            imgs.append(r["infraccions"])
        results.append((hp, np.mean(scores), sum(imgs)))
        print(f"gamma={hp['gamma']:<5} kappa={hp['kappa']:<4} -> mean score {np.mean(scores):6.2f}  total infr {sum(imgs)}")

    best = max(results, key=lambda r: r[1])
    print("\nBEST:", best)

# --- finer sweep 2 ---
print("\n--- sweep2 ---")
res2 = []
for gamma in [0.16, 0.20, 0.264]:
    for bm in [0.0, 0.3, 0.6, 0.9]:
        for b in [0.2, 0.4]:
            scores = []
            for name, w in windows:
                r = eval_window(BatteryAgent, w, gamma=gamma, kappa=0.0, beta_mem=bm, beta=b)
                scores.append(r["score"])
            m = np.mean(scores)
            res2.append((m, gamma, bm, b))
            print(f"gamma={gamma:<5} beta_mem={bm:<4} beta={b:<4} -> {m:6.3f}")
best2 = max(res2)
print("BEST2:", best2)
