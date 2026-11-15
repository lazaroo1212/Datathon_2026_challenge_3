"""test_robustez_hard.py — casos límite para agentes Hard Mode (David, 15/11).

Igual que work/test_robustez.py pero con el obs de Hard Mode:
  - preus_properes_24h / co2_properes_24h con 1, 2, 3 y 24 elementos,
  - NaNs en precio, CO2 y telemetría,
  - solar observada sin ×12 (la v1 autodetecta y escala),
  - SOC en 0 y 6.000,
  - noves_dades_5min vacío/None,
  - secuencia larga encadenada sin infracciones.

Uso:  python3 work/hard/test_robustez_hard.py work/hard/agent_hard.py [params.json]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def cargar_agente(ruta_agente: str):
    import importlib.util
    ruta = Path(ruta_agente).resolve()
    spec = importlib.util.spec_from_file_location("agent_hard", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatteryAgent


AGENT_CLS = cargar_agente(sys.argv[1]) if len(sys.argv) > 1 else None
if AGENT_CLS is None:
    sys.exit("uso: python3 work/hard/test_robustez_hard.py <ruta agent.py> [params.json]")

TS = pd.Timestamp("2026-01-15 12:00:00")


def obs_hard() -> dict:
    return {
        "timestamp": TS,
        "hour": 12,
        "day_of_week": 3,
        "soc_kwh": 3000.0,
        "soc_pct": 0.5,
        "preu_actual": 0.14,
        "co2_actual": 0.15,
        "preus_properes_24h": [0.14, 0.15, 0.16],
        "co2_properes_24h": [0.18, 0.17, 0.16],
        "noves_dades_5min": {
            "ultim_consum_kw": 1200.0,
            "ultima_solar_kw": 1440.0,   # ya ×12 en Hard Mode
            "irradiancia_wm2": 650.0,
            "temp_ambient_c": 18.0,
            "temp_panel_c": 32.0,
            "vent_ms": 3.0,
        },
    }


CASOS = {
    "1. timestamp STRING": lambda o: o.update(timestamp="2026-01-15 12:00:00"),
    "2. horizontes de 1 elemento": lambda o: o.update(
        preus_properes_24h=[0.14], co2_properes_24h=[0.18]),
    "3. horizontes de 2 elementos": lambda o: o.update(
        preus_properes_24h=[0.14, 0.15], co2_properes_24h=[0.18, 0.17]),
    "4. horizontes de 24 (por si el servidor no recorta)": lambda o: o.update(
        preus_properes_24h=[0.14] * 24, co2_properes_24h=[0.18] * 24),
    "5. NaN en precio futuro": lambda o: o["preus_properes_24h"].__setitem__(1, np.nan),
    "6. NaN en CO2 futuro": lambda o: o["co2_properes_24h"].__setitem__(2, np.nan),
    "7. arrays vacíos": lambda o: o.update(preus_properes_24h=[], co2_properes_24h=[]),
    "8. arrays None": lambda o: o.update(preus_properes_24h=None, co2_properes_24h=None),
    "9. noves_dades_5min vacío": lambda o: o.update(noves_dades_5min={}),
    "10. noves_dades_5min None": lambda o: o.update(noves_dades_5min=None),
    "11. NaN en consumo y solar": lambda o: o["noves_dades_5min"].update(
        ultim_consum_kw=np.nan, ultima_solar_kw=np.nan),
    "12. solar SIN ×12 (autodetección)": lambda o: o["noves_dades_5min"].update(
        ultima_solar_kw=120.0, irradiancia_wm2=650.0),
    "13. solar en 0 (noche)": lambda o: o["noves_dades_5min"].update(
        ultima_solar_kw=0.0, irradiancia_wm2=0.0),
    "14. consumo y solar en 0 (corte de consumo)": lambda o: o["noves_dades_5min"].update(
        ultim_consum_kw=0.0, ultima_solar_kw=0.0),
    "15. SOC en 0": lambda o: o.update(soc_kwh=0.0, soc_pct=0.0),
    "16. SOC en 6.000 (techo)": lambda o: o.update(soc_kwh=6000.0, soc_pct=1.0),
    "17. SOC fuera de rango (-100)": lambda o: o.update(soc_kwh=-100.0),
    "18. preu_actual/co2_actual NaN": lambda o: o.update(preu_actual=np.nan, co2_actual=np.nan),
    "19. meteo NaN": lambda o: o["noves_dades_5min"].update(
        irradiancia_wm2=np.nan, temp_ambient_c=np.nan, temp_panel_c=np.nan, vent_ms=np.nan),
    "20. meteo ausente (obs viejo estándar)": None,  # manejado aparte (asignación)
    "21. hora inconsistente con timestamp": lambda o: o.update(hour=23),
}


def _caso20(o):
    o["noves_dades_5min"] = {
        k: v for k, v in o["noves_dades_5min"].items()
        if k not in ("irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms")}
    return o


CASOS["20. meteo ausente (obs viejo estándar)"] = _caso20

agent = AGENT_CLS()
fallos = []
print(f"{'caso':<48} {'consigna':>10}  estado")
print("-" * 72)
for nombre, mutar in CASOS.items():
    o = obs_hard()
    mutar(o)
    try:
        p = float(agent.act(o))
        if np.isnan(p):
            estado, fallos = "FAIL: NaN", fallos + [nombre]
        elif not (-1500.0 - 1e-9 <= p <= 1500.0 + 1e-9):
            estado, fallos = f"FAIL: fuera de rango ({p})", fallos + [nombre]
        else:
            estado = "ok"
        print(f"{nombre:<48} {p:>10.2f}  {estado}")
    except Exception as e:
        estado = f"FAIL: {type(e).__name__}: {e}"
        fallos.append(nombre)
        print(f"{nombre:<48} {'--':>10}  {estado}")

print("\nsecuencia encadenada 168 pasos (Hard Mode, SOC evoluciona):")
o = obs_hard()
soc = 3000.0
ok = True
for i in range(168):
    try:
        p = float(agent.act(o))
    except Exception as e:
        ok = False
        fallos.append(f"secuencia paso {i}: {e}")
        break
    if np.isnan(p) or not (-1500 <= p <= 1500):
        ok = False
        fallos.append(f"secuencia paso {i}: valor {p}")
        break
    soc += p * 0.9487 if p > 0 else p / 0.9487
    soc = float(np.clip(soc, 0.0, 6000.0))
    o["soc_kwh"], o["soc_pct"] = soc, soc / 6000
    o["timestamp"] = o["timestamp"] + pd.Timedelta(hours=1)
    o["hour"], o["day_of_week"] = o["timestamp"].hour, o["timestamp"].dayofweek
print(f"  168 pasos ok: {ok}, SOC final {soc:.0f} kWh")

print("\n" + "=" * 72)
if fallos:
    print(f"RESULTADO: {len(fallos)} FALLOS -> arreglar antes de enviar:")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)
print("RESULTADO: TODOS LOS CASOS OK — agente Hard a prueba de bombas.")
