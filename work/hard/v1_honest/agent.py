"""Hard Mode v1: MPC-LP con forecasts aprendidos (ridge) de w, carga y solar."""
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
from scipy.optimize import linprog

ETA_C = 0.9 ** 0.5
ETA_D = 0.9 ** 0.5
CAP = 6000.0
PMAX = 1500.0


def _to_dt(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    elif isinstance(ts, str):
        import pandas as pd
        ts = pd.Timestamp(ts).to_pydatetime()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _clean(x):
    try:
        x = float(x)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


class BatteryAgent:
    def __init__(self):
        p = json.load(open(Path(__file__).parent / "params.json"))
        self.PL = np.array(p["PL"])
        self.PP = np.array(p["PP"])
        self.PW = np.array(p["PW"])
        self.coef = p["coef"]
        self.gamma = float(p.get("gamma_default", 0.27))
        self.Htot = int(p.get("horizon", 24))
        self.w_hist, self.load_hist, self.pv_hist = {}, {}, {}
        self._pvq, self._mult = [], 1.0   # auto-deteccion de escala de la solar observada

    @staticmethod
    def _ix(t):
        return (t.month - 1, t.hour, 1 if t.weekday() >= 5 else 0)

    @staticmethod
    def _bucket(lst, k):
        for c in lst:
            if c["lo"] <= k <= c["hi"]:
                return c["b"]
        return lst[-1]["b"] if k > lst[-1]["hi"] else lst[0]["b"]

    def _solve(self, w, net_f, soc0):
        H = len(w)
        n = 4 * H
        c = np.zeros(n)
        c[2 * H:3 * H] = w
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
        bounds = ([(0, PMAX)] * H + [(0, PMAX)] * H + [(0, 20000)] * H + [(0.0, CAP)] * H)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res is None or not res.success or res.x is None:
            return None
        return float(res.x[0] - res.x[H])

    def act(self, obs):
        price_arr = obs.get("preus_properes_24h") or []
        co2_arr = obs.get("co2_properes_24h") or []
        H0 = min(len(price_arr), len(co2_arr))
        if H0 == 0:
            return 0.0
        price = np.nan_to_num(np.asarray(price_arr[:H0], dtype=float), nan=0.14)
        co2 = np.nan_to_num(np.asarray(co2_arr[:H0], dtype=float), nan=0.085)
        ts = _to_dt(obs["timestamp"])
        d5 = obs.get("noves_dades_5min") or {}
        last_load = _clean(d5.get("ultim_consum_kw"))
        last_pv = _clean(d5.get("ultima_solar_kw"))
        h1 = timedelta(hours=1)
        tprev = ts - h1
        if last_pv is not None:
            ipp = self._ix(tprev)
            if self.PP[ipp] > 150:
                self._pvq.append(last_pv / self.PP[ipp])
                self._pvq = self._pvq[-40:]
            if len(self._pvq) >= 6:
                med = float(np.median(self._pvq))
                self._mult = 12.0 if 0.03 <= med <= 0.13 else 1.0
            last_pv = last_pv * self._mult
        if last_load is not None:
            self.load_hist[tprev] = last_load
        if last_pv is not None:
            self.pv_hist[tprev] = last_pv
        soc0 = float(np.clip(_clean(obs.get("soc_kwh")) or 0.0, 0.0, CAP))

        w_known = np.nan_to_num(co2 + self.gamma * price, nan=0.10)
        for k in range(H0):
            self.w_hist[ts + timedelta(hours=k)] = float(w_known[k])
        Ht = max(H0, self.Htot)

        # ---- forecast w beyond known window ----
        pw_known = np.array([self.PW[self._ix(ts + timedelta(hours=k))] for k in range(H0)])
        rW = float(np.clip(np.mean(w_known) / max(np.mean(pw_known), 1e-6), 0.3, 3.0))
        w2 = float(w_known[-1])
        w = np.zeros(Ht)
        w[:H0] = w_known
        for j in range(H0, Ht):
            T = ts + timedelta(hours=j)
            pwT = float(self.PW[self._ix(T)])
            y24 = self.w_hist.get(T - timedelta(hours=24), pwT)
            b = self._bucket(self.coef["w"], j)
            w[j] = max(b[0] + b[1] * pwT + b[2] * pwT * rW + b[3] * y24 + b[4] * w2, 0.01)

        # ---- forecast net load ----
        i_prev = self._ix(tprev)
        rL = 1.0
        if last_load is not None and self.PL[i_prev] > 1:
            rL = float(np.clip(last_load / self.PL[i_prev], 0.4, 1.8))
        rP = 1.0
        if last_pv is not None and self.PP[i_prev] > 20:
            rP = float(np.clip(last_pv / self.PP[i_prev], 0.0, 2.5))
        net = np.zeros(Ht)
        for k in range(Ht):
            T = ts + timedelta(hours=k)
            ix = self._ix(T)
            plT = float(self.PL[ix])
            ppT = float(self.PP[ix])
            T24 = T - timedelta(hours=24)
            bl = self._bucket(self.coef["load"], k)
            lf = bl[0] + bl[1] * plT + bl[2] * plT * rL + bl[3] * self.load_hist.get(T24, plT)
            if ppT > 1.0:
                bp = self._bucket(self.coef["pv"], k)
                pf = bp[0] + bp[1] * ppT + bp[2] * ppT * rP + bp[3] * self.pv_hist.get(T24, ppT)
                pf = max(pf, 0.0)
            else:
                pf = 0.0
            net[k] = max(lf - pf, 0.0)

        p = self._solve(w, net, soc0)
        if p is None:
            w0 = w[0]
            if w0 > np.quantile(w, 0.8) and soc0 > 50:
                p = -PMAX
            elif w0 < np.quantile(w, 0.25) and soc0 < CAP - 100:
                p = PMAX
            else:
                p = 0.0
        p_max_c = max(0.0, min(PMAX, (CAP - soc0) / ETA_C) - 0.02)
        p_max_d = max(0.0, min(PMAX, soc0 * ETA_D) - 0.02)
        return float(np.clip(p, -p_max_d, p_max_c))
