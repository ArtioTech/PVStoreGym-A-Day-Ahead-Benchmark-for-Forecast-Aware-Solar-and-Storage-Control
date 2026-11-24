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

    def __init__(
        self,
        cfg: BatteryConfig,
        soc_init: float,
        legacy_v04: bool = False,
        *,
        aux_drains: bool | None = None,
        directional_taper: bool | None = None,
        degradation_on_ac_power: bool | None = None,
        energy_limited_power: bool | None = None,
    ):
        """Three independent behaviour flags (``legacy_v04=True`` sets all to v0.4):

        ``aux_drains``
            True: the auxiliary load is supplied by the battery. False (v0.4):
            it was subtracted from the AC setpoint, so an idle battery *gained*
            ``aux_kw`` of free energy.
        ``directional_taper``
            True: the power taper near a SOC bound limits only the direction
            that approaches the bound. False (v0.4): it limited both, so a
            battery at ``soc_min`` could not be charged again.
        ``degradation_on_ac_power``
            True: degradation cost is charged on ``|p_bat|``. False (v0.4/v1.0):
            on the DC-side power including the auxiliary load.
        ``energy_limited_power``
            True: the realised power is also limited by the energy between the
            SOC bounds, so a step can never draw more than the battery holds.
            False (v0.4/v1.0): only the taper limited power; a discharge from
            just inside the taper margin could request several times the
            remaining energy, and the final SOC clip then created the missing
            energy out of nothing (up to about 60 kWh in one step).
        """
        self.cfg = cfg
        self.legacy_v04 = bool(legacy_v04)
        self.aux_drains = (not legacy_v04) if aux_drains is None else bool(aux_drains)
        self.directional_taper = (not legacy_v04) if directional_taper is None else bool(directional_taper)
        self.degradation_on_ac_power = (
            (not legacy_v04) if degradation_on_ac_power is None else bool(degradation_on_ac_power)
        )
        self.energy_limited_power = (
            (not legacy_v04) if energy_limited_power is None else bool(energy_limited_power)
        )
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

    def directional_limits_kw(self) -> tuple[float, float]:
        """Return (max_discharge_kw, max_charge_kw) at the current SOC.

        Discharge tapers to zero towards ``soc_min`` and charge tapers to zero
        towards ``soc_max``; the opposite direction stays at full PCS power.
        """
        cfg = self.cfg
        if not self.directional_taper:
            lim = self.pcs_limit_kw()
            return lim, lim
        m = max(cfg.taper_margin_soc, 1e-9)
        dis = cfg.p_pcs_kw * float(np.clip((self.soc - cfg.soc_min) / m, 0.0, 1.0))
        ch = cfg.p_pcs_kw * float(np.clip((cfg.soc_max - self.soc) / m, 0.0, 1.0))
        return dis, ch

    def energy_limits_kw(self, dt_h: float) -> tuple[float, float]:
        """Largest AC discharge / charge power that keeps the next SOC inside its window.

        Solves ``e1 == e_min`` (resp. ``e_max``) of :meth:`step` for the power,
        including auxiliary load, self-discharge and the load-dependent
        efficiency (fixed-point iteration, the efficiency moves by at most 2%).
        """
        cfg = self.cfg
        sd = 1.0 - cfg.self_discharge_per_day * (dt_h / 24.0)
        aux = cfg.aux_kw if self.aux_drains else -cfg.aux_kw
        e0 = self.e_kwh
        room_dis = e0 - (cfg.soc_min * cfg.e_max_kwh) / sd      # DC energy that may leave
        room_ch = (cfg.soc_max * cfg.e_max_kwh) / sd - e0       # DC energy that may enter
        dis = ch = 0.0
        eta_d = eta_c = cfg.eta_nom
        for _ in range(6):
            dis = max(0.0, room_dis * eta_d / dt_h - aux)
            eta_d = self.eta(dis + aux)
            ch = max(0.0, room_ch / (dt_h * eta_c) + aux)
            eta_c = self.eta(-ch + aux)
        return dis, ch

    def power_headroom(self) -> dict:
        dis, ch = self.directional_limits_kw()
        return {"discharge_kw": dis, "charge_kw": ch}

    def step(self, p_req_kw: float, dt_h: float) -> dict:
        dis_lim, ch_lim = self.directional_limits_kw()
        if self.energy_limited_power:
            dis_energy, ch_energy = self.energy_limits_kw(dt_h)
            dis_lim, ch_lim = min(dis_lim, dis_energy), min(ch_lim, ch_energy)
        lim = max(dis_lim, ch_lim)
        p_real = float(np.clip(p_req_kw, -ch_lim, dis_lim))
        # DC-side power: the auxiliary load is supplied by the battery.
        p_net = p_real + self.cfg.aux_kw if self.aux_drains else p_real - self.cfg.aux_kw

        e0 = self.e_kwh
        if self.energy_limited_power and p_net > 0.0:
            # An empty battery cannot supply the auxiliary load either: the DC
            # draw is capped by the energy above soc_min (the rest of the
            # auxiliary load is then not served by the battery).
            sd = 1.0 - self.cfg.self_discharge_per_day * (dt_h / 24.0)
            room = max(0.0, e0 - (self.cfg.soc_min * self.cfg.e_max_kwh) / sd)
            p_net = min(p_net, room * self.eta(p_net) / dt_h)
        eta = self.eta(p_net)

        if p_net >= 0:
            e1 = e0 - (p_net * dt_h) / eta
        else:
            e1 = e0 + ((-p_net) * dt_h) * eta

        e1 *= (1.0 - self.cfg.self_discharge_per_day * (dt_h / 24.0))
        e1 = float(np.clip(e1, self.cfg.soc_min * self.cfg.e_max_kwh, self.cfg.soc_max * self.cfg.e_max_kwh))
        self.soc = e1 / self.cfg.e_max_kwh

        throughput_kwh = abs(p_net) * dt_h
        # Degradation: c_thr * |p_bat| * dt (AC power), or the DC-side legacy basis.
        deg_basis_kwh = abs(p_real) * dt_h if self.degradation_on_ac_power else throughput_kwh
        deg_cost = deg_basis_kwh * self.cfg.deg_eur_per_kwh_throughput

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
