"""LGBM-forecast variant of the MPC agent (test-time-observable features only)."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from mpc_agent import BatteryAgent, pd_offset  # noqa: E402
from train_lgbm import FEATS, cal_features  # noqa: E402


class LGBMAgent(BatteryAgent):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.model = joblib.load(HERE / "model_net.pkl")
        self.alpha = alpha          # blend: alpha*model + (1-alpha)*profile
        self.raw_hist = {}          # ts -> net observed (last completed hours)
        self.hist_order = []        # timestamps in observation order

    def _remember(self, ts, last_net, last_pv):
        if last_net is None:
            return
        t_prev = ts - pd_offset(1)
        if t_prev not in self.raw_hist:
            self.raw_hist[t_prev] = float(last_net) - float(last_pv or 0.0)
            self.hist_order.append(t_prev)
            if len(self.hist_order) > 400:
                old = self.hist_order.pop(0)
                self.raw_hist.pop(old, None)

    def act(self, obs):
        price_arr = obs.get("preus_properes_24h") or []
        co2_arr = obs.get("co2_properes_24h") or []
        H = min(len(price_arr), len(co2_arr))
        self._price_buf = np.asarray(price_arr[:H] or [0.14], dtype=float)
        self._co2_buf = np.asarray(co2_arr[:H] or [0.085], dtype=float)
        ts = obs["timestamp"]
        if not hasattr(ts, "month"):
            ts = pd.Timestamp(ts)
        d5 = obs.get("noves_dades_5min") or {}
        self._remember(ts, d5.get("ultim_consum_kw"), d5.get("ultima_solar_kw"))
        return super().act(obs)

    def _feat_row(self, ts_now, tau, k, price_tau, co2_tau):
        """Feature row for target hour tau (k = tau - ts_now), decided at ts_now.
        Mirrors train_lgbm.build_samples: last observed hour = ts_now - 1h."""
        f = cal_features(tau)
        f["horizon"] = k + 1
        f["precio_target"] = price_tau
        f["co2_target"] = co2_tau
        f["precio_now"] = np.nan
        f["co2_now"] = np.nan
        for lag_name, hh in [("net_lag", 24), ("net_lag2", 48), ("net_lag7", 168)]:
            f[lag_name] = self.raw_hist.get(tau - pd_offset(hh), np.nan)
        # rolling means over (tau - hh, ts_now - 1h] observed window
        i_prev = ts_now - pd_offset(1)
        for roll_name, hh, min_n in [("net_roll24", 24, 6), ("net_roll168", 168, 24)]:
            vals = [self.raw_hist[t] for t in self.raw_hist
                    if i_prev - pd_offset(hh) < t <= i_prev]
            f[roll_name] = float(np.mean(vals)) if len(vals) >= min_n else np.nan
        return f

    def _forecast(self, ts, hours_ahead, last_net, last_pv):
        # parent's profile forecast for blending
        prof_net, prof_pv = super()._forecast(ts, hours_ahead, last_net, last_pv)
        price = self._price_buf
        co2 = self._co2_buf
        H = len(price)
        rows = []
        for k in range(H):
            tau = ts + pd_offset(k)
            rows.append(self._feat_row(ts, tau, k, price[k], co2[k]))
        X = pd.DataFrame(rows, columns=FEATS)
        pred = np.maximum(self.model.predict(X), 0.0)
        net = self.alpha * pred + (1.0 - self.alpha) * prof_net[:H]
        return np.maximum(net, 0.0), prof_pv[:H]
