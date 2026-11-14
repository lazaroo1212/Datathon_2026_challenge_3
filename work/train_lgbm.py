"""Train LightGBM direct multi-horizon forecasters for net load (and load/pv variant).

Rules (per README assignment to Pei):
- train only on in_hidden_week == 0 rows, excluding eval windows (days >= 18 of month)
- sample weight = 1 - max(load_frac, pv_frac)  (imputed rows count less)
- features must be agent-observable at test time (calendar, known price/co2, lags/rolls of
  observed load/pv). No meteo (not available in obs).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
from pathlib import Path

HERE = Path(__file__).parent

HORIZONS = list(range(1, 25))  # t+1 .. t+24

CAL_FEATS = ["hour", "day_of_week", "month", "week_of_year", "is_weekend", "is_holiday",
             "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
KNOWN_FEATS = ["precio_target", "co2_target", "precio_now", "co2_now"]
LAG_FEATS = ["net_lag", "net_lag2", "net_lag7", "net_roll24", "net_roll168"]
FEATS = CAL_FEATS + KNOWN_FEATS + LAG_FEATS + ["horizon"]


def prepare_base(df):
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["Fecha"])
    df["net_kw"] = df["load_kw"] - df["pv_kw"]
    return df


def cal_features(ts):
    return {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "month": ts.month,
        "week_of_year": ts.isocalendar().week,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_holiday": 0,  # placeholder; holiday names not derivable at test time
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
    }


def build_samples(df, eval_mask=None):
    """Direct multi-horizon samples. eval_mask: rows excluded as validation targets."""
    idx = df.set_index("ts")
    net = df["net_kw"].values
    price = df["precio_eur_per_kwh"].values
    co2 = df["co2_kg_per_kwh"].values
    imputed = np.maximum(df["load_kw_imputed_frac"].values, df["pv_kw_imputed_frac"].values)
    ts_all = df["ts"].values

    pos = {t: i for i, t in enumerate(df["ts"].values)}

    rows, targets, weights, hz = [], [], [], []
    for i in range(len(df)):
        if eval_mask is not None and eval_mask[i]:
            continue
        if imputed[i] > 0.5:
            continue
        t_i = df["ts"].iloc[i]
        base_cal = cal_features(t_i)
        for h in HORIZONS:
            j = i + h
            if j >= len(df) or imputed[j] > 0.5:
                continue
            gap_ok = (ts_all[j] - ts_all[i]) == np.timedelta64(h, "h")
            if not gap_ok:
                continue
            t_j = df["ts"].iloc[j]
            # lags relative to TARGET hour t_j (what agent would have observed by then)
            f = dict(base_cal)
            f.update(cal_features(t_j))
            f["horizon"] = h
            f["precio_target"] = price[j]
            f["co2_target"] = co2[j]
            f["precio_now"] = price[i]
            f["co2_now"] = co2[i]
            for lag, hh in [("net_lag", 24), ("net_lag2", 48), ("net_lag7", 168)]:
                k = pos.get(t_j - np.timedelta64(hh, "h"))
                f[lag] = net[k] if (k is not None and imputed[k] <= 0.5 and
                                    ts_all[k] <= ts_all[i]) else np.nan
            # rolling means of net over the 24/168h BEFORE t_j but only up to time i
            hist_mask = (df["ts"].values > ts_all[j] - np.timedelta64(24, "h")) & \
                        (df["ts"].values <= ts_all[i])
            v = net[hist_mask & (imputed <= 0.5)]
            f["net_roll24"] = v.mean() if len(v) >= 6 else np.nan
            hist_mask_w = (df["ts"].values > ts_all[j] - np.timedelta64(168, "h")) & \
                          (df["ts"].values <= ts_all[i])
            v2 = net[hist_mask_w & (imputed <= 0.5)]
            f["net_roll168"] = v2.mean() if len(v2) >= 24 else np.nan
            rows.append(f)
            targets.append(net[j])
            weights.append(1.0 - imputed[j])
            hz.append(h)
    X = pd.DataFrame(rows, columns=FEATS)
    return X, np.array(targets), np.array(weights)


def train_model(X, y, w, tag):
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=4, verbose=-1,
    )
    model.fit(X, y, sample_weight=w)
    joblib.dump(model, HERE / f"model_{tag}.pkl")
    print(f"[{tag}] trained. MAE on train (weighted-ish, indicative):",
          round(float(np.abs(model.predict(X) - y).mean()), 2), "kW")
    imp = sorted(zip(model.feature_name_, model.feature_importances_), key=lambda x: -x[1])
    print(f"[{tag}] top features:", imp[:8])
    return model


if __name__ == "__main__":
    df = prepare_base(pd.read_csv(HERE / "master_dataset.csv"))
    df = df[df["in_hidden_week"] == 0].reset_index(drop=True)
    # exclude pseudo-test windows (days 18-31 of each month) from training
    eval_mask = df["ts"].dt.day >= 18
    print("train rows:", int((~eval_mask).sum()), " eval-window rows excluded:", int(eval_mask.sum()))

    X, y, w = build_samples(df, eval_mask=eval_mask.values)
    print("samples:", X.shape)
    train_model(X, y, w, "net")
