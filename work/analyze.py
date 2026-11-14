import pandas as pd
import numpy as np

df = pd.read_csv(__file__.replace("analyze.py", "hourly.csv"), parse_dates=["Fecha"]).set_index("Fecha")
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = df["dow"] >= 5
df["net"] = df["load_kw"] - df["pv_kw"]

print("== zero/low load hours (gap-like) ==")
print((df["load_kw"] < 100).value_counts().head())
print("\n== load by hour & weekend ==")
print(df.groupby(["is_weekend", "hour"])["load_kw"].mean().unstack(0).round(0).to_string())
print("\n== co2 by hour ==")
print(df.groupby("hour")["co2_kg_per_kwh"].mean().round(3).to_string())
print("\n== price by hour ==")
print(df.groupby("hour")["precio_eur_per_kwh"].mean().round(3).to_string())
print("\n== pv by hour ==")
print(df.groupby("hour")["pv_kw"].mean().round(1).to_string())
print("\n== load by month ==")
print(df.groupby("month")["load_kw"].agg(["mean", "std"]).round(0).to_string())
print("\n== co2 stats overall ==")
print(df["co2_kg_per_kwh"].describe().round(3).to_string())
print("\n== price stats overall ==")
print(df["precio_eur_per_kwh"].describe().round(3).to_string())

# avg per-kWh import price/co2 (import hours only)
imp = df[df["net"] > 0]
print("\nimport-hours price mean:", round(imp["precio_eur_per_kwh"].mean(), 4),
      " co2 mean:", round(imp["co2_kg_per_kwh"].mean(), 4))

# month-hour patterns of co2 and price (for embedding in agent)
mh_co2 = df.groupby(["month", "hour"])["co2_kg_per_kwh"].mean().round(4)
mh_pr = df.groupby(["month", "hour"])["precio_eur_per_kwh"].mean().round(4)
print("\nco2 month x hour grid sample (Jan):", mh_co2.loc[1].values.tolist() if 1 in mh_co2.index.get_level_values(0) else "n/a")
print("months present:", sorted(df["month"].unique()))
