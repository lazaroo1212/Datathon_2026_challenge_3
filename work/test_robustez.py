"""test_robustez.py — David: batería de casos límite para BatteryAgent.

El servidor puede alimentar `act(obs)` con formas que el entorno local no produce:
timestamp como string, horizonte truncado (H<24), NaNs en precios/CO2,
`noves_dades_5min` vacío, SOC en extremos... Cualquier excepción aquí = envío
fallido en la plataforma. Esta batería verifica que el agente NUNCA explota y
siempre devuelve un float válido en [-500, +500].

Uso:  python3 work/test_robustez.py            (agente canónico work/)
      python3 work/test_robustez.py submission # el agente empaquetado
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import mpc_agent  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "submission":
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub_agent", HERE.parent / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AGENT_CLS = mod.BatteryAgent
    print("agente bajo test: submission/agent.py")
else:
    AGENT_CLS = mpc_agent.BatteryAgent
    print("agente bajo test: work/mpc_agent.py (params canónicos)")

BASE_TS = pd.Timestamp("2026-01-15 12:00:00")


def obs_ok() -> dict:
    """obs completo y sano, como el del enunciado."""
    return {
        "timestamp": BASE_TS,
        "hour": 12,
        "day_of_week": 3,
        "soc_kwh": 1000.0,
        "soc_pct": 0.5,
        "preu_actual": 0.14,
        "co2_actual": 0.15,
        "preus_properes_24h": [0.14] * 24,
        "co2_properes_24h": [0.18] * 24,
        "noves_dades_5min": {"ultim_consum_kw": 1200.0, "ultima_solar_kw": 80.0},
    }


CASOS = {
    "1. timestamp como STRING (como check_submission)": lambda o: o.update(timestamp="2026-01-15 12:00:00"),
    "2. horizonte truncado H=3 (fin de mes)": lambda o: (o.update(preus_properes_24h=[0.14] * 3, co2_properes_24h=[0.18] * 3)),
    "3. horizonte H=1": lambda o: (o.update(preus_properes_24h=[0.14], co2_properes_24h=[0.18])),
    "4. NaNs en precios y CO2 futuros": lambda o: o.update(
        preus_properes_24h=[np.nan] * 24, co2_properes_24h=[np.nan] * 24),
    "5. NaNs solo en 3 horas intermedias": lambda o: o.update(
        preus_properes_24h=[0.14] * 24, co2_properes_24h=[0.18 if i % 3 else np.nan for i in range(24)]),
    "6. noves_dades_5min vacío": lambda o: o.update(noves_dades_5min={}),
    "7. noves_dades_5min = None": lambda o: o.update(noves_dades_5min=None),
    "8. noves_dades_5min con NaNs dentro": lambda o: o.update(
        noves_dades_5min={"ultim_consum_kw": np.nan, "ultima_solar_kw": np.nan}),
    "9. SOC en el suelo (0 kWh)": lambda o: o.update(soc_kwh=0.0, soc_pct=0.0),
    "10. SOC en el techo (2000 kWh)": lambda o: o.update(soc_kwh=2000.0, soc_pct=1.0),
    "11. SOC fuera de rango (-50, por si el env filtra mal)": lambda o: o.update(soc_kwh=-50.0),
    "12. preus_properes_24h = None": lambda o: o.update(preus_properes_24h=None, co2_properes_24h=None),
    "13. preu_actual/co2_actual NaN": lambda o: o.update(preu_actual=np.nan, co2_actual=np.nan),
    "14. arrays de longitud desigual (24 vs 17)": lambda o: o.update(co2_properes_24h=[0.18] * 17),
    "15. timestamp sin hour (string raro)": lambda o: o.update(timestamp="2026-01-15"),
    "16. hora del obs inconsistente con timestamp": lambda o: o.update(hour=23),
    "17. consumo/solar 0 (campus apagado)": lambda o: o.update(
        noves_dades_5min={"ultim_consum_kw": 0.0, "ultima_solar_kw": 0.0}),
    "18. consumo negativo (sensor raro)": lambda o: o.update(
        noves_dades_5min={"ultim_consum_kw": -120.0, "ultima_solar_kw": -5.0}),
}

agent = AGENT_CLS()
fallos = []
print(f"\n{'caso':<55} {'consigna':>10}  estado")
print("-" * 80)
for nombre, mutar in CASOS.items():
    o = obs_ok()
    mutar(o)
    try:
        p = agent.act(o)
        p = float(p)
        if np.isnan(p):
            estado = "FAIL: NaN"
            fallos.append(nombre)
        elif not (-500.0 - 1e-9 <= p <= 500.0 + 1e-9):
            estado = f"FAIL: fuera de rango ({p})"
            fallos.append(nombre)
        else:
            estado = "ok"
        print(f"{nombre:<55} {p:>10.2f}  {estado}")
    except Exception as e:
        estado = f"FAIL: {type(e).__name__}: {e}"
        fallos.append(nombre)
        print(f"{nombre:<55} {'--':>10}  {estado}")

# --- secuencia continua: 72 pasos encadenados simulando un env real ---
print("\nsecuencia encadenada 72 pasos (con SOC que evoluciona):")
o = obs_ok()
soc = 1000.0
ok = True
for i in range(72):
    p = float(agent.act(o))
    if np.isnan(p) or not (-500 <= p <= 500):
        ok = False
        fallos.append(f"secuencia paso {i}")
        break
    soc += p * 0.9487 if p > 0 else p / 0.9487
    soc = float(np.clip(soc, 0.0, 2000.0))
    o["soc_kwh"] = soc
    o["soc_pct"] = soc / 2000
    ts = o["timestamp"] + pd.Timedelta(hours=1)
    o["timestamp"] = ts
    o["hour"] = ts.hour
    o["day_of_week"] = ts.dayofweek
print(f"  72 pasos ok: {ok}, SOC final {soc:.0f} kWh")

print("\n" + "=" * 80)
if fallos:
    print(f"RESULTADO: {len(fallos)} FALLOS -> arreglar antes de enviar:")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)
print("RESULTADO: TODOS LOS CASOS OK — el agente es a prueba de bombas.")
