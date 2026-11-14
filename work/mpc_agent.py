"""MPC-LP BatteryAgent: model-predictive control over 24h horizon with linear programming.

Objective per hour: w[t] = co2[t] + gamma * price[t]  (co2 dominates score at 70% weight,
gamma converts EUR/kWh into kg-CO2-equivalents from the score's implied trade-off).
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ETA_C = 0.9 ** 0.5
ETA_D = 0.9 ** 0.5
CAP = 2000.0
PMAX = 500.0


class BatteryAgent:
    def __init__(self):
        here = Path(__file__).parent
        with open(here / "params.json") as f:
            params = json.load(f)
        self.prof_net = np.array(params["prof_net"])      # (12, 24, 2)
        self.prof_pv = np.array(params["prof_pv"])        # (12, 24, 2)
        self.gamma = params.get("gamma_default", 0.264)   # price->co2 conversion
        self.kappa = params.get("kappa", 0.5)             # terminal value scale
        self.beta = params.get("beta", 0.4)               # forecast correction damping
        self.beta_mem = params.get("beta_mem", 0.6)       # history-ratio damping
        self.h_extra = int(params.get("h_extra", 0))      # profile-only hours beyond the
                                                          # exact 24h price/co2 window
        self.prof_co2 = np.array(params.get("prof_co2", np.zeros((12, 24))))
        self.prof_price = np.array(params.get("prof_price", np.zeros((12, 24))))
        self.mem = {}                                     # hour-of-day -> list of observed ratios
        self.last_ts = None
        self.w_hist = {}                                  # ts -> known w (exact when within 24h window)
        self.ext_mix = params.get("ext_mix", 0.5)         # 0=monthly profile, 1=pure persistence
        self.reserve_kwh = params.get("reserve_kwh", 0.0) # evening SOC floor at horizon end
        self.reserve_hours = tuple(params.get("reserve_hours", (17, 18, 19, 20, 21, 22, 23)))
        self.reserve_ratio = params.get("reserve_ratio", 1.25)  # gate: far peak > ratio*near peak

    # ---------- forecasting ----------
    def _profile(self, ts):
        return (ts.month - 1, ts.hour, 1 if ts.dayofweek >= 5 else 0)

    def _update_memory(self, ts, last_net, last_pv):
        m, h, d = self._profile(ts)
        prof = self.prof_net[m, h, d]
        if prof > 1e-6 and last_net is not None:
            r = float(np.clip(last_net / prof, 0.5, 1.6))
            lst = self.mem.setdefault(h, [])
            lst.append(r)
            if len(lst) > 3:
                lst.pop(0)

    def _forecast(self, ts, hours_ahead, last_net, last_pv):
        months, hours, dts = [], [], []
        for k in range(hours_ahead):
            t = ts + pd_offset(k)
            months.append(t.month - 1)
            hours.append(t.hour)
            dts.append(1 if t.dayofweek >= 5 else 0)
        net = self.prof_net[months, hours, dts].copy()
        pv = self.prof_pv[months, hours, dts].copy()
        # per-hour-of-day memory correction
        for k in range(hours_ahead):
            lst = self.mem.get(hours[k])
            if lst:
                wt = np.array([0.5, 0.3, 0.2][-len(lst):])
                r = float(np.dot(wt, lst[-len(wt):]))
                net[k] *= 1.0 + self.beta_mem * (r - 1.0)
        # damped multiplicative correction from last observed hour (decays with horizon)
        prof_last = self.prof_net[months[0], hours[0], dts[0]]
        if prof_last > 1e-6 and last_net is not None:
            ratio = float(np.clip(last_net / prof_last, 0.55, 1.45))
            for k in range(hours_ahead):
                net[k] *= 1.0 + self.beta * (ratio - 1.0) * np.exp(-k / 8.0)
        prof_pv_last = self.prof_pv[months[0], hours[0], dts[0]]
        if prof_pv_last > 1e-6 and last_pv is not None:
            ratio_pv = float(np.clip(last_pv / prof_pv_last, 0.0, 2.0))
            pv *= 1.0 + 0.3 * (ratio_pv - 1.0)
        return np.maximum(net, 0.0), np.maximum(pv, 0.0)

    # ---------- LP ----------
    def _solve(self, w, net_f, soc0, cv, soc_end_min=0.0):
        H = len(w)
        n = 4 * H
        c = np.zeros(n)
        c[2 * H:3 * H] = w                      # grid import vars
        c[3 * H + H - 1] = -cv                  # terminal soc value
        c[:2 * H] = 1e-9                        # tie-break

        A_eq = np.zeros((H, n))
        b_eq = np.zeros(H)
        for k in range(H):
            A_eq[k, k] = -ETA_C                 # -p_c[k]
            A_eq[k, H + k] = 1.0 / ETA_D        # +p_d[k]/eta_d
            A_eq[k, 3 * H + k] = 1.0            # soc[k+1]
            if k == 0:
                b_eq[k] = soc0
            else:
                A_eq[k, 3 * H + k - 1] = -1.0   # -soc[k]

        A_ub = np.zeros((H + 1, n))
        b_ub = np.zeros(H + 1)
        for k in range(H):
            A_ub[k, k] = 1.0                    # p_c
            A_ub[k, H + k] = -1.0               # -p_d
            A_ub[k, 2 * H + k] = -1.0           # -g
            b_ub[k] = -net_f[k]                 # p_c - p_d - g <= -net  <=>  g >= net + p
        # soc[H] >= soc_end_min  ->  -soc[H] <= -soc_end_min
        A_ub[H, 3 * H + H - 1] = -1.0
        b_ub[H] = -soc_end_min

        bounds = ([(0, PMAX)] * H + [(0, PMAX)] * H + [(0, 5000)] * H +
                  [(0.0, CAP)] * H)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res is None or not res.success or res.x is None:
            return None
        return float(res.x[0] - res.x[H])       # p_c[0] - p_d[0]

    # ---------- agent ----------
    @staticmethod
    def _clean(x):
        """NaN/None/basura -> None (evita propagar NaN al LP: ValueError)."""
        try:
            x = float(x)
            return None if np.isnan(x) else x
        except (TypeError, ValueError):
            return None

    def act(self, obs):
        price_arr = obs.get("preus_properes_24h") or []
        co2_arr = obs.get("co2_properes_24h") or []
        H = min(len(price_arr), len(co2_arr))
        if H == 0:
            return 0.0
        price = np.asarray(price_arr[:H], dtype=float)
        co2 = np.asarray(co2_arr[:H], dtype=float)
        price = np.nan_to_num(price, nan=0.14)
        co2 = np.nan_to_num(co2, nan=0.085)

        ts = obs["timestamp"]
        if not hasattr(ts, "month"):
            import pandas as pd
            ts = pd.Timestamp(ts)
        d5 = obs.get("noves_dades_5min") or {}
        last_net = self._clean(d5.get("ultim_consum_kw", None))
        last_pv = self._clean(d5.get("ultima_solar_kw", None))
        # last completed hour is ts - 1h; update per-hour ratio memory
        if hasattr(ts, "hour"):
            if self.last_ts is None or ts - self.last_ts == pd_offset(1):
                self._update_memory(ts - pd_offset(1), last_net, last_pv)
            self.last_ts = ts

        soc0 = float(np.clip(obs.get("soc_kwh", 1000.0), 0.0, CAP))
        w = co2 + self.gamma * price
        w = np.nan_to_num(w, nan=0.10)
        for k in range(H):
            self.w_hist[ts + pd_offset(k)] = float(w[k])

        # extend horizon beyond the exact 24h window: blend monthly-mean profile with
        # persistence (w of yesterday-same-hour, which lies inside the exact window)
        H_total = H + int(self.h_extra)
        if H_total > H:
            ext = np.zeros(H_total - H)
            for j in range(len(ext)):
                tau = ts + pd_offset(H + j)
                w_month = (self.prof_co2[tau.month - 1, tau.hour]
                           + self.gamma * self.prof_price[tau.month - 1, tau.hour])
                w_pers = self.w_hist.get(tau - pd_offset(24))
                if w_pers is None:
                    ext[j] = w_month
                else:
                    ext[j] = self.ext_mix * w_pers + (1.0 - self.ext_mix) * w_month
            w = np.concatenate([w, ext])

        net_f, pv_f = self._forecast(ts, H_total, last_net, last_pv)
        net_f = np.maximum(net_f - pv_f, 0.0)

        # terminal value: worth of stored energy ~= discounted best future w
        if H_total >= 4:
            w_sorted = np.sort(w)
            ref = float(np.mean(w_sorted[-min(4, H_total):]))
        else:
            ref = float(np.max(w)) if H_total else 0.0
        cv = self.kappa * ETA_D * ref

        # evening reserve, gated on the blindspot condition: estimated far peak
        # (t+24..t+47) must clearly exceed the best remaining visible peak
        soc_end_min = 0.0
        if self.reserve_kwh > 0 and getattr(ts, "hour", None) in self.reserve_hours:
            far_vals = []
            for j in range(24):
                tau = ts + pd_offset(H + j)
                w_month = (self.prof_co2[tau.month - 1, tau.hour]
                           + self.gamma * self.prof_price[tau.month - 1, tau.hour])
                w_pers = self.w_hist.get(tau - pd_offset(24))
                far_vals.append(w_month if w_pers is None
                                else self.ext_mix * w_pers + (1.0 - self.ext_mix) * w_month)
            far = float(np.max(far_vals))
            near = float(np.max(w[:H]))
            if far > self.reserve_ratio * near:
                soc_end_min = self.reserve_kwh

        p = self._solve(w, net_f, soc0, cv, soc_end_min=soc_end_min)
        if p is None:
            # fallback greedy
            w0 = w[0]
            if w0 > np.quantile(w, 0.8) and soc0 > 50:
                p = -PMAX
            elif w0 < np.quantile(w, 0.25) and soc0 < 1900:
                p = PMAX
            else:
                p = 0.0

        # clamp to physical limits with safety margins (avoid any infraction)
        p_max_c = max(0.0, min(PMAX, (CAP - soc0) / ETA_C) - 0.02)
        p_max_d = max(0.0, min(PMAX, soc0 * ETA_D) - 0.02)
        p = float(min(p_max_c, max(-p_max_d, p)))
        return float(np.clip(p, -p_max_d, p_max_c))


def pd_offset(k):
    import datetime
    return datetime.timedelta(hours=k)
