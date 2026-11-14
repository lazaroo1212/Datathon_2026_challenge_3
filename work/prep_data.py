"""Build hourly dataset exactly as the official tutorial does."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "paquet_participants_datathon_upc" / "data" / "train"
OUT = Path(__file__).parent / "hourly.csv"

df_contador = pd.read_csv(DATA / "contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
load_5min = (-df_contador["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
load_hourly = load_5min.resample("1h").mean().dropna()

df_inversor = pd.read_csv(DATA / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
p1 = [c for c in df_inversor.columns if "INVERSOR 1" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
p2 = [c for c in df_inversor.columns if "INVERSOR 2" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
p3 = [c for c in df_inversor.columns if "INVERSOR 3" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
pv_total_5min = (df_inversor[p1].fillna(0) + df_inversor[p2].fillna(0) + df_inversor[p3].fillna(0)).clip(lower=0)
pv_hourly = pv_total_5min.resample("1h").mean().dropna()

df_pvpc = pd.read_csv(DATA / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
df_co2 = pd.read_csv(DATA / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

df = pd.DataFrame({
    "load_kw": load_hourly,
    "pv_kw": pv_hourly,
    "precio_eur_per_kwh": df_pvpc["precio_eur_per_kwh"],
    "co2_kg_per_kwh": df_co2["co2_kg_per_kwh"],
}).dropna()
df.index.name = "Fecha"
df.to_csv(OUT)

print(df.describe().round(2))
print("\nrows:", len(df), "range:", df.index[0], "->", df.index[-1])

d = df.index.to_series()
is_last7 = d.dt.day > d.dt.days_in_month - 7
print("rows inside last-7-day windows:", int(is_last7.sum()))
