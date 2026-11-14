"""replica_72h.py — David: replica la trayectoria del diagnóstico oficial del servidor.

El dashboard oficial muestra las primeras 72 h del test (Sep 24 -> Sep 26, 2025)
con cargas ~+500 kW de mañana, descargas -500 kW en picos vespertinos y ciclos
SOC 0->100->0.

NOTA: Sep 24-26 son días ocultos (ausentes de hourly.csv). Replicamos la ventana
pseudo-test equivalente de 72 h: Sep 18-20 (días 18-24 de cada mes, el proxy más
cercano al test oculto). El "check de oro" exacto contra la primera consigna
oficial (+112.57 kW en Sep 24 00:00) no es posible sin los datos ocultos; validamos
el patrón cualitativo: cargas ~+500 kW de mañana, descargas -500 kW en picos
vespertinos, ciclos SOC 0->100->0, arranque en SOC 50%.

Uso:  python3 work/replica_72h.py [inicio] [fin]
      python3 work/replica_72h.py 2025-09-18 2025-09-20
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paquet_participants_datathon_upc" / "starter_kit"))
sys.path.insert(0, str(HERE))

from environment import CampusBatteryEnv  # noqa: E402
from mpc_agent import BatteryAgent  # noqa: E402

inicio = sys.argv[1] if len(sys.argv) > 1 else "2025-09-18"
fin = sys.argv[2] if len(sys.argv) > 2 else "2025-09-20 23:00"

df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
win = df.loc[inicio:fin].copy()
if len(win) == 0:
    sys.exit(f"ventana vacía: {inicio} -> {fin} no está en hourly.csv "
             f"(rango disponible: {df.index[0]} -> {df.index[-1]})")
print(f"ventana: {win.index[0]} -> {win.index[-1]}  ({len(win)} h)\n")

env = CampusBatteryEnv(win)
agent = BatteryAgent()
obs = env.reset()

rows = []
done = False
while not done:
    p = agent.act(obs)
    obs, done = env.step(p)
    t = len(rows) - 1
    rows.append({
        "Fecha": win.index[t],
        "load": float(win["load_kw"].iloc[t]),
        "pv": float(win["pv_kw"].iloc[t]),
        "consigna": p,
        "soc": env.soc_series[-1],
        "grid": env.grid[-1],
    })

r = pd.DataFrame(rows)
r["hora"] = r["Fecha"].dt.strftime("%d %H:%M")

# --- tabla compacta (solo horas con acción relevante para no inundar) ---
print("hora        load    pv  consigna    SOC%   grid")
for _, x in r.iterrows():
    mark = ""
    if x["consigna"] >= 400:
        mark = "  <- carga"
    elif x["consigna"] <= -400:
        mark = "  <- descarga"
    if abs(x["consigna"]) >= 50 or mark:
        print(f"{x['hora']}  {x['load']:6.0f} {x['pv']:5.0f}  {x['consigna']:7.1f}  "
              f"{100 * x['soc'] / 2000:5.1f}  {x['grid']:6.1f}{mark}")

# --- métricas de comparación con el dashboard oficial ---
cargas = r[r["consigna"] >= 400]
descargas = r[r["consigna"] <= -400]
soc_max = 100 * r["soc"].max() / 2000
soc_min = 100 * r["soc"].min() / 2000

print("\n=== comparación cualitativa con el dashboard oficial (72 h) ===")
print(f"primera consigna: {r['consigna'].iloc[0]:+.2f} kW  (SOC inicial 50%)")
print(f"cargas >=+400 kW: {len(cargas)}  en horas: {list(cargas['Fecha'].dt.hour)}")
print(f"descargas <=-400 kW: {len(descargas)}  en horas: {list(descargas['Fecha'].dt.hour)}")
print(f"SOC max {soc_max:.0f}% / min {soc_min:.0f}%   (oficial: ciclos 0->100->0)")
print(f"score del entorno en la ventana: {env.evaluate_results()}")
