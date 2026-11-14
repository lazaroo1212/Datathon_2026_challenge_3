"""Build the hourly master dataset (Jan - data & features).

Pipeline
--------
1. Load the 6 raw CSV files from ``data/train``.
2. Build the 5-min series:
   - ``load_kw``  = max(0, -Contador POTENCIA ACTIVA EXPORTADA)  (net grid import)
   - ``pv_kw``    = sum of the 3 inverters' POTENCIA ACTIVA
   - meteo: irradiance, ambient/panel temperature, wind speed, peak-sun hours
   - digital relay state (entradas_salidas)
3. Physical cleaning:
   - power < 0 -> 0 (only exports / sensor noise)
   - power > 5000 kW -> NaN (anomaly, later imputed)
4. Impute on the complete 5-min grid:
   - gaps <= 15 min (<= 3 samples): time interpolation
   - remaining gaps: seasonal mean (month x hour x minute x daytype, with
     coarser fallbacks down to the global mean)
5. Resample 5-min -> 1 h with the mean (official tutorial semantics).
6. Align the hourly PVPC price and CO2 series (same seasonal imputation for
   their own gaps).
7. Add calendar/holiday features, lags 24/48/168 h, past-only rolling means,
   irradiance and temperature.
8. Write ``work/master_dataset.csv`` (hourly) and ``work/base_5min.csv.gz``.

The hidden test weeks (last 7 days of each month) are intentionally absent from
the raw training files.  They are imputed here only to obtain a continuous
series; use the ``in_hidden_week`` and ``*_imputed_frac`` columns to exclude
them from model training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "paquet_participants_datathon_upc" / "data" / "train"
OUT_HOURLY = HERE / "master_dataset.csv"
OUT_5MIN = HERE / "base_5min.csv.gz"

ANOMALY_KW = 5000.0
SHORT_GAP_SAMPLES = 3  # 3 x 5 min = 15 min

# National + Catalonia holidays plus Barcelona local (La Merce).
HOLIDAYS = {
    "2025-09-11": "Diada Nacional de Catalunya",
    "2025-09-24": "La Merce (local BCN)",
    "2025-10-12": "Fiesta Nacional de Espana",
    "2025-11-01": "Tots Sants",
    "2025-12-06": "Dia de la Constitucio",
    "2025-12-08": "Immaculada Concepcio",
    "2025-12-25": "Nadal",
    "2025-12-26": "Sant Esteve",
    "2026-01-01": "Any Nou",
    "2026-01-06": "Reis",
    "2026-04-03": "Divendres Sant",
    "2026-04-06": "Dilluns de Pasqua",
    "2026-05-01": "Dia del Treball",
    "2026-06-24": "Sant Joan",
    "2026-08-15": "L'Assumpcio",
    "2026-09-11": "Diada Nacional de Catalunya",
    "2026-09-24": "La Merce (local BCN)",
}
HOLIDAYS = {pd.Timestamp(k): v for k, v in HOLIDAYS.items()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _find_col(columns, *needles, exclude=()):
    """Return the first column containing every needle and none of the excludes."""
    for col in columns:
        upper = col.upper()
        if all(n.upper() in upper for n in needles) and not any(
            e.upper() in upper for e in exclude
        ):
            return col
    raise KeyError(f"column not found: {needles} (exclude={exclude})")


def _clean_power(s: pd.Series, high: float = ANOMALY_KW):
    """Coerce to numeric, turn >high into NaN, clip negatives to zero."""
    s = pd.to_numeric(s, errors="coerce")
    n_high = int((s > high).sum())
    n_neg = int((s < 0).sum())
    s = s.mask(s > high).clip(lower=0.0)
    return s, n_high, n_neg


def _clean_meteo(s: pd.Series, low=-np.inf, high=np.inf):
    """Range check; out-of-range values become NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.mask((s < low) | (s > high))


def _fill_short_gaps(s: pd.Series):
    """Linear time interpolation for gaps of at most SHORT_GAP_SAMPLES."""
    filled = s.interpolate(
        method="time", limit=SHORT_GAP_SAMPLES, limit_area="inside"
    )
    short_mask = s.isna() & filled.notna()
    return filled, short_mask


def _seasonal_fill(s: pd.Series, key_levels):
    """Fill remaining NaN with hierarchical seasonal means.

    The means are always computed from the originally observed values, so each
    coarser level only sees real measurements.  ``key_levels`` is an ordered
    list of column-name lists, from finest to coarsest.
    """
    idx = s.index
    frame = pd.DataFrame(
        {
            "v": s.to_numpy(),
            "month": idx.month,
            "hour": idx.hour,
            "minute": idx.minute,
            "dow": idx.dayofweek,
            "weekend": (idx.dayofweek >= 5).astype(int),
        },
        index=idx,
    )
    out = frame["v"].copy()
    for keys in key_levels:
        means = frame.groupby(keys, dropna=False)["v"].transform("mean")
        out = out.fillna(means)
    out = out.fillna(frame["v"].mean())
    return out


def _power_key_levels():
    return [
        ["month", "hour", "minute", "weekend"],
        ["month", "hour", "weekend"],
        ["hour", "minute", "weekend"],
        ["hour", "weekend"],
    ]


def _hourly_key_levels():
    return [
        ["month", "hour", "weekend"],
        ["hour", "weekend"],
    ]


def _impute_5min(raw: pd.Series, full_index, key_levels):
    """Return (imputed series on full 5-min grid, short mask, seasonal mask)."""
    s = raw.reindex(full_index)
    s, short_mask = _fill_short_gaps(s)
    seasonal_mask = s.isna()
    s = _seasonal_fill(s, key_levels)
    return s, short_mask, seasonal_mask


# --------------------------------------------------------------------------- #
# Load & build the 5-min series
# --------------------------------------------------------------------------- #
def build_5min():
    contador = pd.read_csv(DATA / "contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
    inversor = pd.read_csv(DATA / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    meteo = pd.read_csv(DATA / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
    entradas = pd.read_csv(DATA / "entradas_salidas.csv", parse_dates=["Fecha"]).set_index("Fecha")

    # --- Load (P_consum = max(0, -exported)) -------------------------------- #
    exp_col = _find_col(contador.columns, "Contador", "POTENCIA ACTIVA EXPORTADA")
    load_raw = -pd.to_numeric(contador[exp_col], errors="coerce")
    load_5, load_high, load_neg = _clean_power(load_raw)

    # --- PV (sum of the 3 inverters) ---------------------------------------- #
    inv_cols = [c for c in inversor.columns if "POTENCIA ACTIVA (KW)" in c.upper()
                and "INVERSOR" in c.upper() and "FASE" not in c.upper()]
    if len(inv_cols) != 3:
        raise RuntimeError(f"expected 3 inverter power columns, found {inv_cols}")
    pv = inversor[inv_cols].apply(pd.to_numeric, errors="coerce")
    pv_5 = pv.sum(axis=1, min_count=1)
    pv_5, pv_high, pv_neg = _clean_power(pv_5)

    # --- Meteo -------------------------------------------------------------- #
    irr_col = _find_col(meteo.columns, "IRRADIANCIA (W/M2)")
    irr = _clean_meteo(meteo[irr_col], low=0.0, high=ANOMALY_KW)
    ta_col = _find_col(meteo.columns, "TEMPERATURA AMBIENTE")
    ta = _clean_meteo(meteo[ta_col], low=-40.0, high=70.0)
    tp_col = _find_col(meteo.columns, "TEMPERATURA PANEL")
    tp = _clean_meteo(meteo[tp_col], low=-40.0, high=100.0)
    wind_col = _find_col(meteo.columns, "VELOCIDAD VIENTO")
    wind = _clean_meteo(meteo[wind_col], low=0.0, high=60.0)
    hsp_col = _find_col(meteo.columns, "HORAS SOL PICO")
    hsp = _clean_meteo(meteo[hsp_col], low=0.0, high=24.0)

    # --- Digital inputs (relay 0 is the only populated signal) -------------- #
    relay_col = _find_col(entradas.columns, "SALIDA RELE 0", exclude=(".1",))
    relay = pd.to_numeric(entradas[relay_col], errors="coerce")

    base = pd.concat(
        {
            "load_kw": load_5,
            "pv_kw": pv_5,
            "irradiance": irr,
            "temp_ambient": ta,
            "temp_panel": tp,
            "wind_speed": wind,
            "hsp": hsp,
            "relay0": relay,
        },
        axis=1,
    )
    base.index.name = "Fecha"

    meta_anomalies = {
        "load_gt5000": load_high,
        "load_neg": load_neg,
        "pv_gt5000": pv_high,
        "pv_neg": pv_neg,
    }
    return base, meta_anomalies


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    base_raw, anomalies = build_5min()

    start = base_raw.index.min()
    end = base_raw.index.max()
    full_5min = pd.date_range(start, end, freq="5min")
    print(f"raw 5-min: {start} -> {end}  ({len(base_raw)} rows)")
    print(f"complete grid: {len(full_5min)} rows")
    print("anomalies:", anomalies)

    # Impute every 5-min column and keep track of what was fabricated.
    imputed = {}
    flags = {}
    for col in base_raw.columns:
        s, short_m, seas_m = _impute_5min(base_raw[col], full_5min, _power_key_levels())
        imputed[col] = s
        flags[f"{col}_imputed_short"] = short_m.astype(float)
        flags[f"{col}_imputed_seasonal"] = seas_m.astype(float)

    base_5min = pd.DataFrame(imputed)
    flags_5min = pd.DataFrame(flags, index=full_5min)

    # --- Resample 5 min -> 1 h with the mean -------------------------------- #
    hourly = base_5min.resample("1h").mean()
    frac_short = flags_5min.resample("1h").mean()
    frac_seas = flags_5min.resample("1h").mean()
    hourly.index.name = "Fecha"

    # Consolidate imputation fractions for the core channels.
    for src in ("load_kw", "pv_kw"):
        hourly[f"{src}_imputed_frac"] = (
            frac_short[f"{src}_imputed_short"]
            + frac_short[f"{src}_imputed_seasonal"]
        ).clip(upper=1.0)
    meteo_src = ("irradiance", "temp_ambient", "temp_panel", "wind_speed")
    meteo_frac = sum(
        frac_short[f"{c}_imputed_short"] + frac_seas[f"{c}_imputed_seasonal"]
        for c in meteo_src
    )
    hourly["meteo_imputed_frac"] = meteo_frac.clip(upper=1.0) / len(meteo_src)

    # --- Hourly price & CO2 (own gaps -> seasonal imputation) --------------- #
    pvpc = pd.read_csv(DATA / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(DATA / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    price = pd.to_numeric(pvpc["precio_eur_per_kwh"], errors="coerce").reindex(hourly.index)
    co2i = pd.to_numeric(co2["co2_kg_per_kwh"], errors="coerce").reindex(hourly.index)

    price_missing = price.isna()
    co2_missing = co2i.isna()
    price = _seasonal_fill(price, _hourly_key_levels())
    co2i = _seasonal_fill(co2i, _hourly_key_levels())
    hourly["precio_eur_per_kwh"] = price
    hourly["co2_kg_per_kwh"] = co2i
    hourly["precio_imputed"] = price_missing.astype(int)
    hourly["co2_imputed"] = co2_missing.astype(int)

    hourly["net_kw"] = hourly["load_kw"] - hourly["pv_kw"]

    # --- Calendar & holidays ------------------------------------------------ #
    idx = hourly.index
    hourly["hour"] = idx.hour
    hourly["day_of_week"] = idx.dayofweek  # 0 = Monday
    hourly["month"] = idx.month
    hourly["day"] = idx.day
    hourly["week_of_year"] = idx.isocalendar().week.to_numpy().astype(int)
    hourly["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    holiday_series = idx.normalize().map(HOLIDAYS)
    hourly["is_holiday"] = holiday_series.notna().astype(int)
    hourly["holiday_name"] = holiday_series.fillna("none").to_numpy()

    # Hidden test weeks: last 7 days of each month.
    is_last7 = idx.day > (idx.days_in_month - 7)
    hourly["in_hidden_week"] = is_last7.astype(int)

    # Cyclical encodings (handy for the forecast models).
    hourly["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    hourly["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    hourly["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # --- Lags 24 / 48 / 168 h ----------------------------------------------- #
    lag_cols = [
        "load_kw",
        "pv_kw",
        "net_kw",
        "irradiance",
        "temp_ambient",
        "temp_panel",
        "precio_eur_per_kwh",
        "co2_kg_per_kwh",
    ]
    for col in lag_cols:
        for lag in (24, 48, 168):
            hourly[f"{col}_lag{lag}"] = hourly[col].shift(lag)

    # --- Past-only rolling means (no leakage: shift(1) first) ---------------- #
    roll_cols = ["load_kw", "pv_kw", "net_kw", "irradiance", "temp_ambient"]
    for col in roll_cols:
        for window in (24, 168):
            hourly[f"{col}_roll{window}"] = (
                hourly[col].shift(1).rolling(window, min_periods=max(1, window // 2)).mean()
            )

    # --- Save --------------------------------------------------------------- #
    hourly = hourly.sort_index()
    hourly.to_csv(OUT_HOURLY, float_format="%.4f")
    base_5min.to_csv(OUT_5MIN, float_format="%.3f", compression="gzip")

    # --- Summary ------------------------------------------------------------ #
    print(f"\nhourly master: {len(hourly)} rows x {hourly.shape[1]} cols -> {OUT_HOURLY}")
    print(f"5-min base:    {len(base_5min)} rows x {base_5min.shape[1]} cols -> {OUT_5MIN}")
    print(f"range: {hourly.index.min()} -> {hourly.index.max()}")
    print("\nmissing values left in key columns:",
          {c: int(hourly[c].isna().sum()) for c in ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]})
    print("load imputed fraction (mean):", round(hourly["load_kw_imputed_frac"].mean(), 4))
    print("pv imputed fraction (mean):  ", round(hourly["pv_kw_imputed_frac"].mean(), 4))
    print("hours in hidden weeks:", int(hourly["in_hidden_week"].sum()))
    print("holiday hours:", int(hourly["is_holiday"].sum()))
    print("\ncolumns:")
    for c in hourly.columns:
        print("  ", c)


if __name__ == "__main__":
    main()
