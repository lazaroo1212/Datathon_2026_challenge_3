"""Evaluate an agent file on pseudo-test windows of the train data."""
import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
from environment import run_agent_simulation  # noqa: E402

df = pd.read_csv(Path(__file__).parent / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")


def pseudo_test_windows():
    """Proxy of hidden test (last 7 days of month, absent from train):
    use days 18-24 of each month, closest availablewindows."""
    windows = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        mask = (g.index.day >= 18) & (g.index.day <= 24)
        w = g[mask]
        if len(w) >= 100:
            windows.append((f"{y}-{m:02d}", w))
    return windows


def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    agent_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paquet_participants_datathon_upc" / "starter_kit" / "agent.py")
    mod = load_agent(agent_path)

    windows = pseudo_test_windows()
    print(f"pseudo-test windows: {[n for n, _ in windows]}")

    # concatenated
    concat = pd.concat([w for _, w in windows])
    res = run_agent_simulation(mod.BatteryAgent(), concat)
    print("\n=== concatenated (2016h-like) ===")
    for k, v in res.items():
        print(f"  {k:24s}: {v}")

    # per window
    scores = []
    print("\n=== per window ===")
    for name, w in windows:
        r = run_agent_simulation(mod.BatteryAgent(), w)
        scores.append(r["score"])
        print(f"  {name}: score={r['score']:6.2f} co2%={r['co2_estalviat_pct']:6.2f} "
              f"cost%={r['cost_estalviat_pct']:6.2f} infr={r['infraccions']}")
    print(f"\nmean per-window score: {np.mean(scores):.2f}")
