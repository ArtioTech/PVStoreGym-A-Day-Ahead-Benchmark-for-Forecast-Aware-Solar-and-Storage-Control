from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BatteryConfig:
    e_max_kwh: float = 1075.0
    p_pcs_kw: float = 373.0
    soc_min: float = 0.05
    soc_max: float = 0.95
    aux_kw: float = 2.0
    self_discharge_per_day: float = 0.001
    eta_nom: float = 0.97
    eta_slope: float = 0.02
    taper_margin_soc: float = 0.03
    deg_eur_per_kwh_throughput: float = 0.0


class Battery:
    """Battery + PCS with:
      - SOC taper near bounds
      - auxiliary load
      - self discharge
      - efficiency curve vs loading
      - optional throughput degradation proxy
    """

    def __init__(self, cfg: BatteryConfig, soc_init: float):
        self.cfg = cfg
        self.soc = float(np.clip(soc_init, cfg.soc_min, cfg.soc_max))

    @property
    def e_kwh(self) -> float:
        return self.soc * self.cfg.e_max_kwh

    def eta(self, p_kw: float) -> float:
        frac = min(1.0, abs(p_kw) / max(1e-6, self.cfg.p_pcs_kw))
        return float(np.clip(self.cfg.eta_nom - self.cfg.eta_slope * frac, 0.85, 0.995))

    def pcs_limit_kw(self) -> float:
        s = self.soc
        lo, hi = self.cfg.soc_min, self.cfg.soc_max
        m = self.cfg.taper_margin_soc
        if s < lo + m:
            return self.cfg.p_pcs_kw * max(0.0, (s - lo) / m)
        if s > hi - m:
            return self.cfg.p_pcs_kw * max(0.0, (hi - s) / m)
        return self.cfg.p_pcs_kw

    def power_headroom(self) -> dict:
        lim = self.pcs_limit_kw()
        return {"discharge_kw": lim, "charge_kw": lim}

    def step(self, p_req_kw: float, dt_h: float) -> dict:
        lim = self.pcs_limit_kw()
        p_real = float(np.clip(p_req_kw, -lim, lim))
        p_net = p_real - self.cfg.aux_kw

        eta = self.eta(p_net)
        e0 = self.e_kwh

        if p_net >= 0:
            e1 = e0 - (p_net * dt_h) / eta
        else:
            e1 = e0 + ((-p_net) * dt_h) * eta

        e1 *= (1.0 - self.cfg.self_discharge_per_day * (dt_h / 24.0))
        e1 = float(np.clip(e1, self.cfg.soc_min * self.cfg.e_max_kwh, self.cfg.soc_max * self.cfg.e_max_kwh))
        self.soc = e1 / self.cfg.e_max_kwh

        throughput_kwh = abs(p_net) * dt_h
        deg_cost = throughput_kwh * self.cfg.deg_eur_per_kwh_throughput

        return {
            "p_req_kw": float(p_req_kw),
            "p_real_kw": float(p_real),
            "p_net_kw": float(p_net),
            "eta": float(eta),
            "p_lim_kw": float(lim),
            "e0_kwh": float(e0),
            "e1_kwh": float(e1),
            "soc": float(self.soc),
            "throughput_kwh": float(throughput_kwh),
            "deg_cost": float(deg_cost),
        }
