"""MPC-LP agent variant: rich dow-indexed profiles + 48h horizon.

Like mpc_agent.py but:
- profiles are (12, 24, 7) month x hour x dayofweek from params_rich.json
  (real data only, hidden weeks excluded);
- the LP horizon is 48 h: the first hours use the exact price/CO2 given in obs,
  the rest use the monthly w profile (prof_w) from params_rich.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ETA_C = 0.9 ** 0.5
ETA_D = 0.9 ** 0.5
CAP = 2000.0
PMAX = 500.0


class BatteryAgent:
    def __init__(self):
        here = Path(__file__).parent
        with open(here / "params_rich.json") as f:
            params = json.load(f)
        self.H = int(params.get("horizon", 48))
        self.extend = bool(int(params.get("extend", 0)))
        self.prof_net = np.array(params["prof_net"])      # (12, 24, 7)
        self.prof_pv = np.array(params["prof_pv"])        # (12, 24, 7)
        self.prof_w = np.array(params["prof_w"])          # (12, 24, 7)
        self.gamma = params.get("gamma_default", 0.264)
        self.kappa = params.get("kappa", 0.0)
        self.beta = params.get("beta", 0.4)
        self.beta_mem = params.get("beta_mem", 0.6)
        self.mem = {}
        self.last_ts = None

    def _profile(self, ts):
        return (ts.month - 1, ts.hour, ts.dayofweek)

    def _update_memory(self, ts, last_net, last_pv):
        m, h, d = self._profile(ts)
        prof = self.prof_net[m, h, d]
        if prof > 1e-6 and last_net is not None:
            r = float(np.clip(last_net / prof, 0.5, 1.6))
            lst = self.mem.setdefault(h, [])
            lst.append(r)
            if len(lst) > 3:
                lst.pop(0)

    def _timestamps(self, ts, hours_ahead):
        import datetime
        return [ts + datetime.timedelta(hours=k) for k in range(hours_ahead)]

    def _forecast(self, ts, hours_ahead, last_net, last_pv):
        tss = self._timestamps(ts, hours_ahead)
        months = [t.month - 1 for t in tss]
        hours = [t.hour for t in tss]
        dows = [t.dayofweek for t in tss]
        net = self.prof_net[months, hours, dows].copy()
        pv = self.prof_pv[months, hours, dows].copy()
        for k in range(hours_ahead):
            lst = self.mem.get(hours[k])
            if lst:
                wt = np.array([0.5, 0.3, 0.2][-len(lst):])
                r = float(np.dot(wt, lst[-len(wt):]))
                net[k] *= 1.0 + self.beta_mem * (r - 1.0)
        prof_last = self.prof_net[months[0], hours[0], dows[0]]
        if prof_last > 1e-6 and last_net is not None:
            ratio = float(np.clip(last_net / prof_last, 0.55, 1.45))
            for k in range(hours_ahead):
                net[k] *= 1.0 + self.beta * (ratio - 1.0) * np.exp(-k / 8.0)
        prof_pv_last = self.prof_pv[months[0], hours[0], dows[0]]
        if prof_pv_last > 1e-6 and last_pv is not None:
            ratio_pv = float(np.clip(last_pv / prof_pv_last, 0.0, 2.0))
            pv *= 1.0 + 0.3 * (ratio_pv - 1.0)
        return np.maximum(net, 0.0), np.maximum(pv, 0.0)

    def _solve(self, w, net_f, soc0, cv):
        H = len(w)
        n = 4 * H
        c = np.zeros(n)
        c[2 * H:3 * H] = w
        c[3 * H + H - 1] = -cv
        c[:2 * H] = 1e-9

        A_eq = np.zeros((H, n))
        b_eq = np.zeros(H)
        for k in range(H):
            A_eq[k, k] = -ETA_C
            A_eq[k, H + k] = 1.0 / ETA_D
            A_eq[k, 3 * H + k] = 1.0
            if k == 0:
                b_eq[k] = soc0
            else:
                A_eq[k, 3 * H + k - 1] = -1.0

        A_ub = np.zeros((H, n))
        b_ub = np.zeros(H)
        for k in range(H):
            A_ub[k, k] = 1.0
            A_ub[k, H + k] = -1.0
            A_ub[k, 2 * H + k] = -1.0
            b_ub[k] = -net_f[k]

        bounds = ([(0, PMAX)] * H + [(0, PMAX)] * H + [(0, 5000)] * H +
                  [(0.0, CAP)] * H)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res is None or not res.success or res.x is None:
            return None
        return float(res.x[0] - res.x[H])

    def act(self, obs):
        price_arr = obs.get("preus_properes_24h") or []
        co2_arr = obs.get("co2_properes_24h") or []
        KNOWN = min(len(price_arr), len(co2_arr))
        if KNOWN == 0:
            return 0.0
        price = np.asarray(price_arr[:KNOWN], dtype=float)
        co2 = np.asarray(co2_arr[:KNOWN], dtype=float)
        price = np.nan_to_num(price, nan=0.14)
        co2 = np.nan_to_num(co2, nan=0.085)

        ts = obs["timestamp"]
        if not hasattr(ts, "dayofweek"):
            import pandas as pd
            ts = pd.Timestamp(ts)
        d5 = obs.get("noves_dades_5min") or {}
        last_net = d5.get("ultim_consum_kw", None)
        last_pv = d5.get("ultima_solar_kw", None)
        H = max(KNOWN, self.H) if self.extend else KNOWN
        if hasattr(ts, "hour"):
            import datetime
            if self.last_ts is None or ts - self.last_ts == datetime.timedelta(hours=1):
                self._update_memory(ts - datetime.timedelta(hours=1), last_net, last_pv)
            self.last_ts = ts
            net_f, pv_f = self._forecast(ts, H, last_net, last_pv)
        else:
            net_f, pv_f = self._forecast(ts, H, None, None)
        net_f = np.maximum(net_f - pv_f, 0.0)

        w = np.empty(H)
        w[:KNOWN] = co2 + self.gamma * price
        tss = self._timestamps(ts, H)
        for k in range(KNOWN, H):
            t = tss[k]
            w[k] = self.prof_w[t.month - 1, t.hour, t.dayofweek]
        w = np.nan_to_num(w, nan=0.10)

        soc0 = float(np.clip(obs.get("soc_kwh", 1000.0), 0.0, CAP))
        if H >= 4:
            w_sorted = np.sort(w)
            ref = float(np.mean(w_sorted[-4:]))
        else:
            ref = float(np.max(w))
        cv = self.kappa * ETA_D * ref

        p = self._solve(w, net_f, soc0, cv)
        if p is None:
            w0 = w[0]
            if w0 > np.quantile(w, 0.8) and soc0 > 50:
                p = -PMAX
            elif w0 < np.quantile(w, 0.25) and soc0 < 1900:
                p = PMAX
            else:
                p = 0.0

        p_max_c = max(0.0, min(PMAX, (CAP - soc0) / ETA_C) - 0.02)
        p_max_d = max(0.0, min(PMAX, soc0 * ETA_D) - 0.02)
        p = float(min(p_max_c, max(-p_max_d, p)))
        return float(np.clip(p, -p_max_d, p_max_c))
