"""Hard Mode v1-LGBM: idéntico a v1_honest pero la SOLAR se predice con LightGBM
directo multi-horizonte (entrenado honesto, días 18–24 excluidos del train).
El forecast de w y de consumo sigue siendo el ridge de v1 (para medir el Δ aislado).
Fallback: si no hay solar_lgbm.pkl, usa el ridge original.
"""
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
        self._pvq, self._mult = [], 1.0
        # ── LGBM solar ──
        self._solar_model = None
        self._solar_feats = None
        mp = Path(__file__).parent / "solar_lgbm.pkl"
        try:
            if mp.exists():
                import joblib
                self._solar_model = joblib.load(mp)
                self._solar_feats = json.load(open(
                    Path(__file__).parent / "features.json"))["features"]
        except Exception:
            self._solar_model = None
        self._pv_last3 = []          # últimas 3 observaciones de pv (escaladas)
        self._irr_last3 = []         # últimas 3 irradiancias
        self._meteo_last = {"temp_panel": None, "temp_amb": None, "wind": None}

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

        # ── historial para features del LGBM solar ──
        if last_pv is not None:
            self._pv_last3 = (self._pv_last3 + [last_pv])[-3:]
        irr_t = _clean(d5.get("irradiancia_wm2"))
        if irr_t is not None:
            self._irr_last3 = (self._irr_last3 + [irr_t])[-3:]
        self._meteo_last["temp_panel"] = _clean(d5.get("temp_panel_c"))
        self._meteo_last["temp_amb"] = _clean(d5.get("temp_ambient_c"))
        self._meteo_last["wind"] = _clean(d5.get("vent_ms"))

        w_known = np.nan_to_num(co2 + self.gamma * price, nan=0.10)
        for k in range(H0):
            self.w_hist[ts + timedelta(hours=k)] = float(w_known[k])
        Ht = max(H0, self.Htot)

        # ---- forecast w beyond known window (ridge v1, sin cambios) ----
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

        # ---- forecast net load (carga: ridge v1; solar: LGBM si disponible) ----
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
                if self._solar_model is not None:
                    pv_m3 = float(np.mean(self._pv_last3)) if self._pv_last3 else ppT
                    irr_m3 = (float(np.mean(self._irr_last3))
                              if self._irr_last3 else 0.0)
                    row = {
                        "mes": T.month, "dow": T.weekday(),
                        "es_finde": int(T.weekday() >= 5),
                        "hora_sin": np.sin(2 * np.pi * ts.hour / 24),
                        "hora_cos": np.cos(2 * np.pi * ts.hour / 24),
                        "pv_t": float(last_pv) if last_pv is not None else ppT,
                        "pv_mean3": pv_m3,
                        "pv_lag24": float(self.pv_hist.get(T24, ppT)),
                        "irr_t": irr_t if irr_t is not None else 0.0,
                        "irr_mean3": irr_m3,
                        "temp_panel_t": self._meteo_last["temp_panel"]
                        if self._meteo_last["temp_panel"] is not None else -1.0,
                        "temp_amb_t": self._meteo_last["temp_amb"]
                        if self._meteo_last["temp_amb"] is not None else -1.0,
                        "wind_t": self._meteo_last["wind"]
                        if self._meteo_last["wind"] is not None else -1.0,
                        "horizon": k,
                        "hora_tk": T.hour,
                        "pp_tk": ppT,
                    }
                    try:
                        X = np.array([[row.get(c, -1.0) for c in self._solar_feats]])
                        pf = max(float(self._solar_model.predict(X)[0]), 0.0)
                    except Exception:
                        pf = None
                else:
                    pf = None
                if pf is None:
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
