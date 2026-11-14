"""Compute forecast tables (month x hour x daytype net-load and PV profiles) for the agent."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "hourly.csv", parse_dates=["Fecha"]).set_index("Fecha")
df["net"] = df["load_kw"] - df["pv_kw"]
df["hour"] = df.index.hour
df["month"] = df.index.month
df["dt"] = (df.index.dayofweek >= 5).astype(int)  # 1 = weekend

# profiles: month(1-12) x hour(0-23) x daytype(0/1)
prof_net = np.zeros((12, 24, 2))
prof_pv = np.zeros((12, 24, 2))
cnt = np.zeros((12, 24, 2))
for (m, h, d), g in df.groupby(["month", "hour", "dt"]):
    prof_net[m - 1, h, d] = g["net"].mean()
    prof_pv[m - 1, h, d] = g["pv_kw"].mean()
    cnt[m - 1, h, d] = len(g)

# fill missing month cells (e.g., November absent, gaps) with global hour+daytype mean
glob = np.zeros((24, 2))
cntg = np.zeros((24, 2))
for (h, d), g in df.groupby(["hour", "dt"]):
    glob[h, d] = g["net"].mean()
    cntg[h, d] = len(g)
for m in range(12):
    for h in range(24):
        for d in range(2):
            if cnt[m, h, d] == 0:
                prof_net[m, h, d] = glob[h, d]
                prof_pv[m, h, d] = df[(df.month == m + 1) | True].groupby("hour")["pv_kw"].mean().get(h, 0.0)

# fallback month: nearest month with data (for November: average of Oct+Dec? use neighbors)
def monthly_mean(col):
    return df.groupby("month")[col].mean()

# monthly hour-mean price/co2 profiles for horizon extension (blindspot fix)
prof_co2 = np.zeros((12, 24))
prof_price = np.zeros((12, 24))
cntm = np.zeros(12)
for m, g in df.groupby("month"):
    cntm[m - 1] = len(g)
    prof_co2[m - 1] = g.groupby("hour")["co2_kg_per_kwh"].mean().values
    prof_price[m - 1] = g.groupby("hour")["precio_eur_per_kwh"].mean().values
glob_co2 = df.groupby("hour")["co2_kg_per_kwh"].mean().values
glob_price = df.groupby("hour")["precio_eur_per_kwh"].mean().values
for m in range(12):
    if cntm[m] == 0:  # missing month (e.g., November): interpolate neighbours
        lo = (m - 1) % 12
        hi = (m + 1) % 12
        src = [x for x in (lo, hi, lo, hi) if cntm[x] > 0]
        prof_co2[m] = np.mean([prof_co2[x] for x in src], axis=0)
        prof_price[m] = np.mean([prof_price[x] for x in src], axis=0)
# fill any residual NaN hours
for arr, gb in ((prof_co2, glob_co2), (prof_price, glob_price)):
    for m in range(12):
        for h in range(24):
            if np.isnan(arr[m, h]):
                arr[m, h] = gb[h]

# Damped weekly correction: recent 3 weeks vs profile mean per dow
params = {
    "prof_net": np.round(prof_net, 2).tolist(),
    "prof_pv": np.round(prof_pv, 2).tolist(),
    "prof_co2": np.round(prof_co2, 5).tolist(),
    "prof_price": np.round(prof_price, 5).tolist(),
    "gamma_default": 0.264,
    "kappa": 0.0,
    "beta": 0.4,
    "beta_mem": 0.6,
    "h_extra": 24,
}
(HERE / "params.json").write_text(json.dumps(params))
print("saved params.json")

# quick check: forecast MAE of net load using these profiles on train data
df["net_fc"] = [prof_net[m - 1, h, d] for m, h, d in zip(df["month"], df["hour"], df["dt"])]
err = df["net_fc"] - df["net"]
print("in-sample profile MAE kW:", round(np.abs(err).mean(), 1), " mean net:", round(df["net"].mean(), 1))
print("MAE rel:", round((np.abs(err).mean() / df["net"].mean()) * 100, 1), "%")
