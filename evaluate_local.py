"""evaluate_local.py — Evaluador oficial local (rol Elayne).

Corre un BatteryAgent sobre el environment.py del starter kit con datos reales
y calcula el SCORE OFICIAL EXACTO del enunciado:

    Score = 0.7 · dCO2% + 0.3 · dCost% - 1.00 · Infracciones

(env.evaluate_results usa una penalización interna de 0.2 pts/infracción; aquí
se recalcula con la penalización oficial 1.00, configurable con --penalty).

Uso:
    python evaluate_local.py                              # agent del starter kit
    python evaluate_local.py work/mpc_agent.py            # cualquier agente
    python evaluate_local.py agente.py --data work/hourly.csv --penalty 1.0
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
from environment import run_agent_simulation  # noqa: E402

DEFAULT_DATA = ROOT / "work" / "hourly.csv"
DEFAULT_AGENT = ROOT / "paquet_participants_datathon_upc" / "starter_kit" / "agent.py"


def load_agent(path: str):
    """Carga un fichero .py y devuelve su módulo (busca clase BatteryAgent)."""
    spec = importlib.util.spec_from_file_location("agent_under_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "BatteryAgent"):
        raise SystemExit(f"ERROR: {path} no define la clase BatteryAgent")
    return mod


def pseudo_test_windows(df: pd.DataFrame, day_lo: int = 18, day_hi: int = 24, min_hours: int = 100):
    """Ventanas pseudo-test (días 18-24 de cada mes): el proxy más cercano al
    test oculto (últimos 7 días del mes, ausentes en train)."""
    windows = []
    for (y, m), g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= day_lo) & (g.index.day <= day_hi)]
        if len(w) >= min_hours:
            windows.append((f"{y}-{m:02d}", w))
    return windows


def official_score(r: dict, penalty: float) -> float:
    """Recalcula el score oficial exacto desde los componentes del entorno."""
    return 0.7 * r["co2_estalviat_pct"] + 0.3 * r["cost_estalviat_pct"] - penalty * r["infraccions"]


def report(title: str, r: dict, penalty: float):
    off = official_score(r, penalty)
    print(f"\n=== {title} ===")
    print(f"  SCORE OFICIAL          : {off:6.2f}   (0.7*{r['co2_estalviat_pct']:.2f} "
          f"+ 0.3*{r['cost_estalviat_pct']:.2f} - {penalty}*{r['infraccions']})")
    print(f"  dCO2 / dCost           : {r['co2_estalviat_pct']:6.2f}% / {r['cost_estalviat_pct']:6.2f}%")
    print(f"  CO2 evitado / ahorro   : {r['co2_estalviat_kg']:>10,.1f} kg / {r['cost_estalviat_eur']:>10,.2f} EUR")
    print(f"  infracciones           : {r['infraccions']}")
    return off


def main():
    ap = argparse.ArgumentParser(description="Evaluador oficial local (Enunciat 0.7/0.3/-infraccions)")
    ap.add_argument("agent", nargs="?", default=str(DEFAULT_AGENT), help="ruta al agent .py")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="CSV horario (work/hourly.csv)")
    ap.add_argument("--penalty", type=float, default=1.0, help="puntos por infraccion (enunciat=1.0)")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Fecha"]).set_index("Fecha")
    mod = load_agent(args.agent)

    windows = pseudo_test_windows(df)
    concat = pd.concat([w for _, w in windows])
    print(f"agente: {args.agent}")
    print(f"datos : {args.data}  |  {len(windows)} ventanas pseudo-test {18}-24: "
          f"{[n for n, _ in windows]}")

    # Evaluación tipo plataforma: todo concatenado (como las 12 semanas del test)
    off_all = report("TODO CONCATENADO (estilo plataforma)", run_agent_simulation(mod.BatteryAgent(), concat), args.penalty)

    # Por ventana + media
    offs, best, worst = [], None, None
    print("\n=== por ventana ===")
    for name, w in windows:
        r = run_agent_simulation(mod.BatteryAgent(), w)
        off = official_score(r, args.penalty)
        offs.append(off)
        tag = ""
        if best is None or off > best[1]:
            best = (name, off)
        if worst is None or off < worst[1]:
            worst = (name, off)
        print(f"  {name}: {off:6.2f}  (CO2 {r['co2_estalviat_pct']:5.2f}% | costo {r['cost_estalviat_pct']:5.2f}% | infr {r['infraccions']})")
    print(f"\nmedia por ventana : {np.mean(offs):6.2f}   (mejor {best[0]}={best[1]:.2f}, peor {worst[0]}={worst[1]:.2f})")

    print("\nCRITERIO DE ENVIO: enviar solo si este score >= mejor agente ya publicado (ver SUBMISSIONS.md) y 0 infracciones.")
    print(f"score tipo plataforma: {off_all:.2f}")
    return off_all


if __name__ == "__main__":
    main()
