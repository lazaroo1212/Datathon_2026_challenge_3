# PROGRESS — Datathon 2026, UPC Sostenible (Challenge 3)

**Team**: David Lázaro, Elayne Valverde, Jan Jardí, Zhipeng Pei
**Status**: MPC agent ready, validated locally (~6.12 vs starter 1.75). Not yet submitted.

---

## 1. Challenge summary

- Program `BatteryAgent.act(obs)` to schedule hourly charge/discharge of a
  **2 MWh / 500 kW / 90% round-trip efficiency** battery (η_c = η_d = √0.9 ≈ 0.9487)
  on the UPC campus PCC.
- **Score** = 0.70 × ΔCO₂% + 0.30 × ΔCost% − penalties, vs. baseline (no battery).
  Each physical infraction (|P| > 500 kW or SOC outside [0, 2000] kWh) subtracts points.
- Sequential eval on hidden test = **last 7 days of each month** (2016 h). At each hour
  the agent sees: current price/CO₂, **next 24 h of price/CO₂ (exact)**, SOC, and last
  hour's load + PV (5-min telemetry). Future load/PV unknown → must forecast.
- solved via `linprog` (HiGHS).

## 2. Data pipeline (`work/prep_data.py` → `work/hourly.csv`)

- Aggregated 5-min telemetry to hourly with the **official tutorial's exact semantics**:
  - `load_kw = clip(-"Contador - POTENCIA ACTIVA EXPORTADA (kW)", 0, 5000)` → 1h mean
    (i.e. net grid import at the PCC, already net of PV).
  - `pv_kw = sum of 3 inverter "POTENCIA ACTIVA (kW)"` → 1h mean.
  - Joined with `precio_pvpc.csv` (PVPC €/kWh) and `co2_red.csv` (kg CO₂/kWh), dropna.
- Result: 5993 rows, 2025-09-01 → 2026-08-16. True test days (last 7 of month) are
  fully absent from train → we validate on **pseudo-test windows = days 18–24** of
  each month (closest proxy).

## 3. Pattern analysis (`work/analyze.py`)

- Weekday load: ~990 kW base (night) → ~1550 kW midday; weekends flat ~790 kW.
- PV is small: mean ~26 kW, max ~144 kW (≈2% of load).
- CO₂ intensity: midday dip 0.052 → evening peak 0.110 kg/kWh.
- PVPC price: evening peak 19–21 h (0.17–0.235 €/kWh), midday cheapest (~0.077).
- **Score-implied trade-off**: score = 0.7·ΔCO₂ + 0.3·ΔCost with constant denominators
  ⟹ minimizing hourly import with weight `w[t] = co2[t] + γ·price[t]` is optimal,
  γ ≈ (0.3·CO2_base)/(0.7·Cost_base) ≈ **0.264** (validated by sweep, flat 6.04–6.12
  for γ ∈ [0.1, 0.5] → robust).

## 4. Baseline

- Starter heuristic agent (`starter_kit/agent.py`): **mean score 1.75** on pseudo-test
  windows (with occasional infractions).

## 5. Our agent — MPC with LP (`work/mpc_agent.py` + `work/params.json`)

Every hour, solve a linear program over the 24 h horizon and apply the first action:

- **Decision vars** (per hour k of horizon): charge `p_c[k]`, discharge `p_d[k]` ∈ [0, 500],
  grid import `g[k] ≥ 0`, SOC `soc[k]` ∈ [0, 2000].
- **Objective**: min Σ w[k]·g[k] with w[k] = co₂[k] + γ·price[k]; terminal value
  −κ·η_d·(mean of top-4 w) on final SOC (κ = 0 best in sweeps).
- **Constraints**: `g[k] ≥ net_forecast[k] + p_c[k] − p_d[k]` (hinge → exports wasted),
  SOC dynamics with η_c / η_d, SOC bounds, SOC₀ = observed.
- **Forecasts**: month × hour × daytype net-load profile tables from train data
  (in-sample MAE ≈ 8.3%), corrected by (a) damped last-hour ratio, (b) per-hour-of-day
  ratio memory learned during the simulation (β_mem).
- **Hard clamps with safety margins** on the returned setpoint → **0 infractions**
  across all pseudo-test windows (bug found & fixed: discharge clamp wasn't capped at
  ±500 kW; float noise e.g. −500.000000001 got clipped by env → counted as infraction).
- **Speed benchmark**: ~1.3 ms per `act()` → ≈ 2.6 s for the full 2016 h test
  (server limit 60 s). Uses only numpy/scipy/pandas/json/pathlib — all allowed.

## 6. Tuning (`work/sweep.py`)

| param | tested range | best |
|---|---|---|
| γ (price↔CO₂ weight) | 0.10–0.50 | 0.18–0.264 (flat) |
| κ (terminal SOC value) | 0–0.5 | 0.0 |
| β (last-hour damping) | 0.2–0.4 | 0.2–0.4 (flat) |
| β_mem (history damping) | 0–0.9 | 0.0–0.6 (flat) |

**Result: mean score ≈ 6.12** (per-window: Sep 5.6, Oct 6.1, Dec 3.7, Jan 5.3,
Mar 6.4, Apr 7.3, May 7.2, Jun 7.2, Jul 6.5), 0 infractions.

## 7. Oracle bound (`work/oracle.py`)

- Running the same LP with **perfect foresight of future load** gives **6.16**.
- ⟹ Forecasting already extracts ~99% of what this controller structure can achieve;
  the score is limited by battery size vs. campus demand, not by forecast quality.

## 8. Submission rules to respect

- `check_submission.py` AST-scans: agent may NOT import `os`, `sys`, `subprocess`, etc.
- Allowed: numpy, pandas, scipy, sklearn, joblib, xgboost/lightgbm, stdlib logic modules.
- Max 10 submissions per turn (10 until 21:00 today, 10 overnight, 10 tomorrow),
  60 s runtime, 25 MB zip.
- Pack: `agent.py` + `params.json` (+ any .pkl) in `submission.zip`; load via
  `Path(__file__).parent` — no absolute paths.

## 9. Remaining / next steps

1. Package `submission.zip` (`agent.py` + `params.json`) and run `check_submission.py`.
2. Make 1–2 platform submissions to calibrate vs. the hidden test
   (expect deviation from the 6.12 local proxy; hidden weeks are days 25–31).
3. Optional micro-gains: extend horizon past 24 h with monthly mean w-profiles,
   per-month γ, richer load model (hour×dow×month), fallback if scipy fails on server.
4. Keep a locked copy of the best agent before experimenting further.
