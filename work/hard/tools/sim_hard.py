"""Simulador local de Hard Mode (PV x12, bateria 6000 kWh / 1500 kW, obs de 3 h).
Uso (desde la raiz del repo):
  python3 sim_hard.py --agent work/hard/v1_honest --hourly work/hourly.csv \
      --env paquet_participants_datathon_upc/starter_kit
Imprime el score sobre las ventanas dias 18-24 de cada mes (concatenadas) y la media por ventana.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

SPEC = dict(capacity_kwh=6000, power_kw=1500, efficiency=0.9, soc_init=0.5,
            soc_min=0.0, soc_max=1.0, dt_hours=1.0)

def load_env_module(env_dir):
    sys.path.insert(0, str(env_dir))
    import environment as E
    return E

def make_env_class(E):
    class HardEnv(E.CampusBatteryEnv):
        def _get_obs(self):
            o = super()._get_obs()
            if o is None:
                return None
            o["preus_properes_24h"] = o["preus_properes_24h"][:3]
            o["co2_properes_24h"] = o["co2_properes_24h"][:3]
            return o
    return HardEnv

def windows(df, a=18, b=24, min_len=100):
    out = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= a) & (g.index.day <= b)]
        if len(w) >= min_len:
            out.append((f"{y}-{m:02d}", w))
    return out

def load_hourly(path, pv_scale=12.0):
    df = pd.read_csv(path, parse_dates=["Fecha"]).set_index("Fecha")
    df["pv_kw"] = df["pv_kw"] * pv_scale
    return df

def load_agent_module(agent_dir):
    p = Path(agent_dir) / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run(env_cls, agent_factory, df):
    env = env_cls(df, spec=SPEC)
    obs = env.reset()
    ag = agent_factory()
    done = False
    while not done:
        obs, done = env.step(ag.act(obs))
    return env.evaluate_results()

def evaluate(agent_dir, hourly, env_dir):
    E = load_env_module(env_dir)
    Env = make_env_class(E)
    df = load_hourly(hourly)
    wins = windows(df)
    mod = load_agent_module(agent_dir)
    cat = pd.concat([w for _, w in wins])
    r = run(Env, mod.BatteryAgent, cat)
    per = [run(Env, mod.BatteryAgent, w)["score"] for _, w in wins]
    return r, per, [n for n, _ in wins]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--hourly", required=True)
    ap.add_argument("--env", required=True)
    a = ap.parse_args()
    r, per, names = evaluate(a.agent, a.hourly, a.env)
    print("ventanas:", names)
    print("CONCATENADAS:", {k: r[k] for k in ["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions"]})
    print("media por ventana: %.2f" % np.mean(per), [round(x, 1) for x in per])
