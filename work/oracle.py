"""Oracle bound: LP with perfect knowledge of future load/pv."""
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


class OracleAgent(BatteryAgent):
    oracle_df = None

    def act(self, obs):
        if self.oracle_df is not None:
            ts = pd.Timestamp(obs["timestamp"]) if not hasattr(obs["timestamp"], "month") else obs["timestamp"]
            H = min(len(obs["preus_properes_24h"]), len(obs["co2_properes_24h"]))
            fut = self.oracle_df.loc[ts: ts + pd.Timedelta(hours=H - 1)]
            net_true = np.maximum((fut["load_kw"] - fut["pv_kw"]).values[:H], 0.0)
            self._net_override = net_true
        return super().act(obs)

    def _forecast(self, ts, hours_ahead, last_net, last_pv):
        if getattr(self, "_net_override", None) is not None:
            net = self._net_override.copy()
            self._net_override = None
            pv = np.zeros_like(net)
            return net, pv
        return super()._forecast(ts, hours_ahead, last_net, last_pv)


windows = []
for (y, m), g in df.groupby([df.index.year, df.index.month]):
    w = g[(g.index.day >= 18) & (g.index.day <= 24)]
    if len(w) >= 100:
        windows.append((f"{y}-{m:02d}", w))

scores = []
for gamma in [0.15, 0.2, 0.264, 0.35]:
    row = []
    for name, w in windows:
        ag = OracleAgent()
        ag.oracle_df = df
        ag.gamma, ag.kappa = gamma, 0.0
        env = CampusBatteryEnv(w)
        obs = env.reset()
        done = False
        while not done:
            obs, done = env.step(ag.act(obs))
        r = env.evaluate_results()
        row.append(r["score"])
    m = np.mean(row)
    scores.append((m, gamma))
    print(f"oracle gamma={gamma}: mean={m:.2f} per-window={[round(x,1) for x in row]}")
print("BEST ORACLE:", max(scores))
