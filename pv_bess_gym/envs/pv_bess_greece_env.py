from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from pv_bess_gym.data.builders.auxiliary.synthetic_signals import simulate_afrr_activation
from pv_bess_gym.data.signals import SignalSchema
from pv_bess_gym.envs.information import ALWAYS, NEVER, InformationData, naive_dam_forecast
from pv_bess_gym.models.battery import Battery, BatteryConfig
from pv_bess_gym.models.grid import Grid, GridConfig
from pv_bess_gym.models.pv import PVConfig, PVModel
from pv_bess_gym.models.settlement import (
    SettlementBreakdown,
    eur_from_kwh,
    kwh_from_kw,
    reserve_capacity_eur,
    settle_energy_layers,
)


@dataclass
class PlantConfig:
    """Physical plant definition.

    The plant is split into three components:
    - PV field and inverter
    - Battery and PCS
    - Grid connection point
    """

    pv: PVConfig
    battery: BatteryConfig
    grid: GridConfig


@dataclass
class MarketConfig:
    """Simulation-time and settlement parameters."""

    dt_h: float = 0.25
    afrr_block_minutes: int = 60
    violation_penalty_eur_per_kw: float = 100.0
    fallback_shortfall_penalty_eur_per_mwh: float = 20.0
    fallback_unavailable_penalty_eur_per_mw: float = 0.0


@dataclass
class HierarchicalConfig:
    """Controls the slower strategic decision layer."""

    decision_block_minutes: int = 60
    expose_action_mask: bool = True


@dataclass
class DayAheadMarketTiming:
    """Timing abstraction for the unified Greek-market environment.

    The real market sequence is D-1 day-ahead commitment, intraday corrections
    closer to delivery, and real-time physical dispatch with imbalance
    settlement. This config keeps that sequence explicit while remaining simple
    enough for Gym benchmarking:
    - the full DAM schedule is committed once before the delivery episode;
    - intraday corrections are additive and respect a configurable lead time;
    - the low-level action only dispatches the battery, it does not create an
      artificial real-time trade.
    """

    intraday_block_minutes: int = 60
    intraday_gate_closure_minutes: int = 60
    expose_action_mask: bool = True


@dataclass
class SimOptions:
    """Behaviour switches of the simulator core, one flag per behaviour.

    The defaults are the v2 simulator. :meth:`legacy` returns the v0.4
    simulator of the SyNERGY MED 2026 conference paper (every flag at its
    v0.4 value). Each flag can also be set on its own.

    ``settlement``
        ``"v2"``: scheduled energy includes requested aFRR activation,
        ``E_sch = (q + a+ r+ - a- r-) dt``; activated energy is settled once,
        at the balancing energy price (received for up, paid back for down).
        ``"v1"`` (conference): requested activation is not in the scheduled
        energy, so activated energy is settled as imbalance *and* paid as
        activation, both directions with a positive sign.
    ``metering_losses``
        True: the grid loss factor is applied to the settled energy.
        False (v0.4): losses computed but not settled.
    ``battery_aux_drains`` / ``directional_taper``
        See :class:`pv_bess_gym.models.battery.Battery`.
    ``degradation_on_ac_power``
        True: degradation cost on ``|p_bat|``. False: on DC-side power.
    ``energy_limited_power``
        True: realised battery power is limited by the energy between the SOC
        bounds. False (v0.4/v1.0): the SOC clip could create energy.
    ``calendar_days``
        True: an episode is a calendar delivery day; DST-change days are
        skipped. False (v0.4): raw 96-row slices.
    ``truncate_at_day_end``
        True: the last step returns ``truncated=True`` and a regular
        observation. False (v0.4): ``terminated=True`` and zeros.
    ``random_start``
        True: a seeded ``reset()`` samples the delivery day.
        False (v0.4): ``reset()`` starts at row 0 unless told otherwise.
    ``id_position_limit``
        ``"off"`` (v0.4-v3): the DAM schedule and the intraday correction are
        each clipped to the grid limit, so the commercial position
        ``q = x + Y`` can reach twice the limit. ``"clip"`` (rev1 A2): the
        intraday correction of a block is clipped so that
        ``-P_imp <= x_k + Y_k <= P_exp`` on every step of the block.
    """

    settlement: str = "v2"
    metering_losses: bool = True
    battery_aux_drains: bool = True
    directional_taper: bool = True
    degradation_on_ac_power: bool = True
    energy_limited_power: bool = True
    calendar_days: bool = True
    truncate_at_day_end: bool = True
    random_start: bool = True
    id_position_limit: str = "off"

    def __post_init__(self) -> None:
        if self.settlement not in {"v1", "v2"}:
            raise ValueError("settlement must be 'v1' or 'v2'.")
        if self.id_position_limit not in {"off", "clip"}:
            raise ValueError("id_position_limit must be 'off' or 'clip'.")

    @classmethod
    def legacy(cls) -> "SimOptions":
        return cls(
            settlement="v1",
            metering_losses=False,
            battery_aux_drains=False,
            directional_taper=False,
            degradation_on_ac_power=False,
            energy_limited_power=False,
            calendar_days=False,
            truncate_at_day_end=False,
            random_start=False,
        )

    @property
    def legacy_v04(self) -> bool:
        """True when every flag is at its v0.4 value."""
        return self == SimOptions.legacy()


@dataclass
class InformationConfig:
    """What the agent is told about the future (the difficulty knobs).

    ``pv_forecast_error_scale`` scales the *real* PV forecast error of the
    dataset: 0 gives a perfect PV forecast, 1 the archived forecast as is,
    2 doubles every error.

    ``price_noise_sigma`` adds seeded AR(1) noise to the price information in
    the observation, as a fraction of each price series' standard deviation.
    It applies to intraday and imbalance prices for the whole day, and to DAM
    prices until the day-ahead schedule is committed (after that the cleared
    DAM prices are public). 0 reproduces the v0.4 setting in which the agent
    observes the realised prices of the whole delivery day.

    The noise depends only on ``(noise_seed, episode start)``, so every
    controller sees the same forecasts for the same day.
    """

    pv_forecast_error_scale: float = 1.0
    price_noise_sigma: float = 0.0
    noise_ar_phi: float = 0.9
    noise_seed: int = 0
    # --- information filter (see pv_bess_gym.envs.information) -----------------
    # info_filter=True: every price in the observation was published at
    # decision time. Intraday entries show the newest auction already published
    # (IDA1 -> IDA2 -> IDA3); imbalance entries, never published in-day, show
    # the observable DAM price as a stand-in. Settlement is unaffected.
    info_filter: bool = False
    # What the D-1 layer sees as DAM prices before it commits: "realised" or
    # "forecast" (naive forecast, da_forecaster = "prev_day" or "prev_week").
    da_price_info: str = "realised"
    da_forecaster: str = "prev_day"
    # Weather information per layer (realism fixes R1, R2; defaults = version 4).
    # da_weather: forecast behind the D-1 decision. "short_lead" = the stitched
    #   Open-Meteo historical forecast (version 4); "day_ahead" = Open-Meteo
    #   Previous Runs `_previous_day2`, issued >= 48 h before the valid time.
    # id_weather: forecast behind intraday / real-time observations after the
    #   D-1 commitment. "same_as_da" (version 4) or "latest": short-lead values
    #   for valid times up to short_lead_visible_h ahead, else the fallback
    #   (`_previous_day1` when da_weather = "day_ahead", else short-lead).
    # pv_forecast_error_scale (kappa) applies to each series' own residual (R5).
    da_weather: str = "short_lead"
    id_weather: str = "same_as_da"
    short_lead_visible_h: float = 3.0

    def __post_init__(self) -> None:
        if self.da_price_info not in {"realised", "forecast"}:
            raise ValueError("da_price_info must be 'realised' or 'forecast'.")
        if self.da_weather not in {"short_lead", "day_ahead"}:
            raise ValueError("da_weather must be 'short_lead' or 'day_ahead'.")
        if self.id_weather not in {"same_as_da", "latest"}:
            raise ValueError("id_weather must be 'same_as_da' or 'latest'.")
        if self.info_filter and self.price_noise_sigma > 0.0:
            raise ValueError("price_noise_sigma cannot be combined with info_filter.")


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Stationary AR(1) sequence with unit marginal variance."""
    out = np.empty(n, dtype=float)
    innovations = rng.normal(0.0, 1.0, size=n)
    out[0] = innovations[0]
    scale = float(np.sqrt(max(1e-12, 1.0 - phi * phi)))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + scale * innovations[i]
    return out


class _BasePVBESSGreeceEnv(gym.Env):
    """Internal simulator core shared by higher-level environment interfaces.

    The simulator uses exogenous weather and market signals, a PV model, a
    battery model, a grid model, and a simplified commercial settlement logic.
    It is designed for algorithm benchmarking rather than market-compliant
    operation.

    All exogenous signals, including PV power, are pre-computed once as numpy
    arrays, so a step costs tens of microseconds instead of re-running the PV
    model for every observation entry.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
        options: Optional[SimOptions] = None,
        information: Optional[InformationConfig] = None,
        episode_days: Optional[Sequence[int]] = None,
        info_data: Optional[InformationData] = None,
    ):
        super().__init__()
        self.info_data = info_data
        self.df_all = signals.reset_index(drop=True)
        self.schema = schema
        self.plant = plant
        self.market = market
        self.episode_len = int(episode_len)
        self.soc_init = float(soc_init)
        self.include_forecasts = bool(include_forecasts)
        self.forecast_horizon_steps = int(forecast_horizon_steps)
        self.options = options or SimOptions()
        self.information = information or InformationConfig()
        if self.information.info_filter and info_data is None:
            raise ValueError(
                "info_filter=True needs the auction-price and publication-time sidecar files; "
                "run scripts/data/build_publication_sidecars.py."
            )

        self.pv_model = PVModel(plant.pv)
        self.grid = Grid(plant.grid)
        self.battery: Optional[Battery] = None

        self.afrr_block_steps = max(1, int(round(market.afrr_block_minutes / (market.dt_h * 60))))
        self.cap_up_commit_kw = 0.0
        self.cap_down_commit_kw = 0.0
        self.block_steps_left = 0

        self._precompute_signals()
        self._build_day_table()
        self.set_episode_days(episode_days)

        self._i0 = 0
        self._init_episode_arrays(0)
        self.t = 0
        self.cum_profit = 0.0
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------
    def _col(self, name: Optional[str], fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if name is not None and name in self.df_all.columns:
            return self.df_all[name].to_numpy(dtype=float)
        return fallback

    def _precompute_signals(self) -> None:
        s, df = self.schema, self.df_all
        self._ts = pd.DatetimeIndex(pd.to_datetime(df[s.ts_col]))
        self._ghi = self._col(s.ghi_wm2)
        self._temp = self._col(s.temp_air_c)
        self._wind = self._col(s.wind_mps)
        self._ghi_f = self._col(s.ghi_fcst_wm2, self._ghi)
        self._temp_f = self._col(s.temp_air_fcst_c, self._temp)
        self._wind_f = self._col(s.wind_fcst_mps, self._wind)
        self._dam = self._col(s.price_dam_eur_mwh)
        self._idp = self._col(s.price_id_eur_mwh, self._dam)
        self._imb = self._col(s.price_imb_eur_mwh)
        self._cup = self._col(s.afrr_cap_up_eur_mw_h)
        self._cdn = self._col(s.afrr_cap_down_eur_mw_h)
        self._act_up = self._col(s.afrr_act_up)
        self._act_dn = self._col(s.afrr_act_down)
        self._en_up = self._col(s.afrr_en_up_eur_mwh)
        self._en_dn = self._col(s.afrr_en_down_eur_mwh)
        self._pen_short = self._col(s.penalty_shortfall_eur_mwh)
        self._pen_unav = self._col(s.penalty_unavailable_eur_mw)

        self._pv_act = self.pv_model.ac_power_series(self._ts, self._ghi, self._temp, self._wind)
        # Forecast-residual scaling kappa acts on the weather (forecast minus
        # actual) *before* the PV model: kappa = 1 is the archived forecast,
        # 0 a perfect forecast, 2 doubles every weather error.
        kappa = float(self.information.pv_forecast_error_scale)
        if kappa != 1.0:
            self._ghi_f = np.clip(self._ghi + kappa * (self._ghi_f - self._ghi), 0.0, None)
            self._temp_f = self._temp + kappa * (self._temp_f - self._temp)
            if self._wind is not None and self._wind_f is not None:
                self._wind_f = np.clip(self._wind + kappa * (self._wind_f - self._wind), 0.0, None)
        self._pv_fc = self.pv_model.ac_power_series(self._ts, self._ghi_f, self._temp_f, self._wind_f)
        # Weather forecast series per lead: (pv_kw, ghi, temp), kappa on each residual.
        self._wx = {"short_lead": (self._pv_fc, self._ghi_f, self._temp_f)}
        info = self.information
        needs_lead = info.da_weather == "day_ahead" or info.id_weather == "latest"
        lead = None if self.info_data is None else self.info_data.weather_lead
        if needs_lead and lead is None:
            raise ValueError("da_weather='day_ahead' / id_weather='latest' need data/processed/weather_previous_runs_<year>.csv "
                             "(scripts/data/fetch_open_meteo_previous_runs.py).")
        if lead is not None:
            for d in (1, 2):
                ghi = np.clip(self._ghi + kappa * (lead[f"ghi_d{d}"] - self._ghi), 0.0, None)
                temp = self._temp + kappa * (lead[f"temp_air_d{d}"] - self._temp)
                wind = None if self._wind is None else np.clip(self._wind + kappa * (lead[f"wind_d{d}"] - self._wind), 0.0, None)
                self._wx[f"d{d}"] = (self.pv_model.ac_power_series(self._ts, ghi, temp, wind), ghi, temp)
        self._wx_da = "d2" if info.da_weather == "day_ahead" else "short_lead"
        self._wx_fallback = "d1" if info.da_weather == "day_ahead" else "short_lead"
        self._wx_visible_steps = int(round(info.short_lead_visible_h / self.market.dt_h))
        self._dam_fc = naive_dam_forecast(self._ts, self._dam, self.information.da_forecaster)

        local = self._ts if self._ts.tz is not None else self._ts.tz_localize("UTC")
        minutes = np.asarray(local.hour * 60 + local.minute, dtype=float)
        angle = 2.0 * np.pi * minutes / (24.0 * 60.0)
        self._tod_sin = np.sin(angle)
        self._tod_cos = np.cos(angle)
        self._price_scale = {
            "dam": float(np.nanstd(self._dam)),
            "id": float(np.nanstd(self._idp)),
            "imb": float(np.nanstd(self._imb)),
        }

    def _build_day_table(self) -> None:
        """Start rows of the episodes that ``reset(options={'day': k})`` addresses."""
        n, length = len(self.df_all), self.episode_len
        steps_per_day = int(round(24.0 / self.market.dt_h))
        if self.options.calendar_days and length == steps_per_day and n >= length:
            dates = np.asarray(self._ts.tz_localize(None).normalize()) if self._ts.tz is not None else np.asarray(self._ts.normalize())
            change = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
            sizes = np.diff(np.r_[change, n])
            # Days with a DST change (92 or 100 steps) and partial days are skipped,
            # and so are days with a missing market value (only possible after a
            # time-stamp correction, see pv_bess_gym.data.time_alignment).
            starts = change[sizes == length].astype(int)
            market = [a for a in (self._dam, self._idp, self._imb, self._cup, self._cdn) if a is not None]
            complete = [s for s in starts if all(np.isfinite(a[s:s + length]).all() for a in market)]
            self.day_starts = np.asarray(complete, dtype=int)
        else:
            self.day_starts = np.arange(0, max(1, n - length + 1), length, dtype=int)

    @property
    def num_days(self) -> int:
        return int(len(self.day_starts))

    def day_start_index(self, day: int) -> int:
        return int(self.day_starts[int(day)])

    def day_date(self, day: int) -> str:
        return self._ts[self.day_start_index(day)].date().isoformat()

    def set_episode_days(self, days: Optional[Sequence[int]]) -> None:
        """Restrict the days that a seeded ``reset()`` samples from (e.g. the train split)."""
        if days is None:
            self.episode_days = np.arange(self.num_days, dtype=int)
        else:
            arr = np.asarray(list(days), dtype=int)
            if arr.size == 0 or arr.min() < 0 or arr.max() >= self.num_days:
                raise ValueError(f"episode_days must be non-empty and within 0..{self.num_days - 1}.")
            self.episode_days = arr

    @property
    def episode_df(self) -> pd.DataFrame:
        return self.df_all.iloc[self._i0:self._i0 + self.episode_len].reset_index(drop=True)

    def activation_series(self, start_idx: int, activation_seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """aFRR activation ratios (up, down) of the episode starting at ``start_idx``.

        Without a seed: the dataset columns. With a seed: a fresh draw that
        depends only on ``(seed, start_idx)``, so every controller, and the
        oracle, see the same realisation.
        """
        length = self.episode_len
        sl = slice(start_idx, start_idx + length)
        if activation_seed is None:
            up = np.zeros(length) if self._act_up is None else self._act_up[sl]
            down = np.zeros(length) if self._act_dn is None else self._act_dn[sl]
            return up, down
        rng = np.random.default_rng(np.random.SeedSequence([int(activation_seed), int(start_idx)]))
        return simulate_afrr_activation(rng, length)

    def _init_episode_arrays(self, start_idx: int, activation_seed: Optional[int] = None) -> None:
        length = self.episode_len
        sl = slice(start_idx, start_idx + length)
        self.activation_seed = None if activation_seed is None else int(activation_seed)
        self._ep_act_up, self._ep_act_dn = self.activation_series(start_idx, activation_seed)
        info_cfg = self.information
        self._ep_wx = {k: tuple(np.asarray(a[sl], dtype=float) for a in v) for k, v in self._wx.items()}
        self._ep_pv_f = self._ep_wx[self._wx_da][0]          # forecast behind the D-1 decision
        self._wx_cache: Dict[Any, Any] = {}

        dam, idp, imb = self._dam[sl], self._idp[sl], self._imb[sl]
        sigma = float(info_cfg.price_noise_sigma)
        if sigma > 0.0:
            rng = np.random.default_rng(np.random.SeedSequence([int(info_cfg.noise_seed), int(start_idx)]))
            phi = float(info_cfg.noise_ar_phi)
            self._ep_dam_f = dam + sigma * self._price_scale["dam"] * _ar1(rng, length, phi)
            self._ep_id_f = idp + sigma * self._price_scale["id"] * _ar1(rng, length, phi)
            self._ep_imb_f = imb + sigma * self._price_scale["imb"] * _ar1(rng, length, phi)
        else:
            self._ep_dam_f, self._ep_id_f, self._ep_imb_f = dam, idp, imb
        if info_cfg.da_price_info == "forecast":
            self._ep_dam_f = self._dam_fc[sl]
        self._init_information_filter(start_idx)

    # ------------------------------------------------------------------
    # Information filter
    # ------------------------------------------------------------------
    def _init_information_filter(self, start_idx: int) -> None:
        """Per-episode publication steps and the auction prices behind the observation."""
        self._filter_on = bool(self.information.info_filter)
        self._obs_cache: Dict[Any, Any] = {}
        if not self._filter_on:
            return
        data, length = self.info_data, self.episode_len
        sl = slice(start_idx, start_idx + length)
        day_start = self._ts[start_idx]
        dt = self.market.dt_h
        self._pub_step = {m: data.publication_step(m, day_start, dt) for m in ("DAM", "IDA1", "IDA2", "IDA3", "ISP1")}
        self._auctions = [("ida1", "IDA1", data.ida1[sl]), ("ida2", "IDA2", data.ida2[sl]), ("ida3", "IDA3", data.ida3[sl])]
        if self._pub_step["DAM"] > 0 or self._pub_step["ISP1"] > 0:
            raise RuntimeError(f"DAM or ISP1 results of {day_start.date()} were published after 00:00 of the delivery day.")

    def _observed_dam(self, committed: bool) -> tuple[np.ndarray, str]:
        """DAM prices the agent may see: forecast before commitment if configured, else realised."""
        if not committed and self.information.da_price_info == "forecast":
            return self._ep_dam_f, "dam_forecast"
        return self._dam[self._i0:self._i0 + self.episode_len], "dam"

    def _observed_prices(self, t: int, committed: bool) -> Dict[str, Any]:
        """Observable DAM / intraday / imbalance-estimate vectors for the whole day at step ``t``.

        Returns values and, per delivery step, the label of the source series
        (a key of ``PUBLICATION_RULES``) so the observation can be audited.
        """
        n_public = sum(1 for _, market, _ in self._auctions if self._pub_step[market] <= t)
        key = (committed, n_public)
        cached = self._obs_cache.get(key)
        if cached is not None:
            return cached
        dam, dam_label = self._observed_dam(committed)
        intraday = np.array(dam, dtype=float, copy=True)
        id_label = np.full(self.episode_len, dam_label, dtype=object)
        for label, market, values in self._auctions:      # later auctions overwrite earlier ones
            if self._pub_step[market] <= t:
                have = ~np.isnan(values)
                intraday[have] = values[have]
                id_label[have] = label
        out = {
            "dam": np.asarray(dam, dtype=float), "dam_label": np.full(self.episode_len, dam_label, dtype=object),
            "id": intraday, "id_label": id_label,
            # The imbalance price is never public in-day: show the observable DAM price.
            "imb": np.asarray(dam, dtype=float), "imb_label": np.full(self.episode_len, dam_label, dtype=object),
        }
        self._obs_cache[key] = out
        return out

    _WX_LABEL = {"short_lead": "weather_short_lead", "d1": "weather_d1", "d2": "weather_d2"}

    def _observed_weather(self, t: int, committed: bool) -> Dict[str, Any]:
        """PV / GHI / temperature forecasts the agent may see at step ``t`` for every delivery step.

        Before the D-1 commitment, or with id_weather = "same_as_da": the D-1
        series. Otherwise ("latest"): the short-lead series for delivery steps
        up to ``short_lead_visible_h`` ahead of ``t``, the fallback series after.
        Also returns per step a label and the publication step of the value.
        """
        latest = committed and self.information.id_weather == "latest"
        key = (latest, int(t) if latest else -1)
        cached = self._wx_cache.get(key)
        if cached is not None:
            return cached
        n = self.episode_len
        v = np.arange(n)
        if not latest:
            pv, ghi, temp = self._ep_wx[self._wx_da]
            name = np.full(n, self._wx_da, dtype=object)
        else:
            near = v <= t + self._wx_visible_steps
            sl, fb = self._ep_wx["short_lead"], self._ep_wx[self._wx_fallback]
            pv, ghi, temp = (np.where(near, a, b) for a, b in zip(sl, fb))
            name = np.where(near, "short_lead", self._wx_fallback).astype(object)
        steps_per_day = int(round(24.0 / self.market.dt_h))
        pub_of = {"d2": v - 2 * steps_per_day, "d1": v - steps_per_day, "short_lead": v - self._wx_visible_steps}
        pub = np.array([pub_of[nm][k] for k, nm in enumerate(name)], dtype=np.int64)
        if not latest and self._wx_da == "short_lead":
            labels = np.full(n, "weather_forecast", dtype=object)     # version-4 assumption
            pub = np.full(n, ALWAYS, dtype=np.int64)
        else:
            labels = np.array([self._WX_LABEL[nm] for nm in name], dtype=object)
        out = {"pv": np.asarray(pv), "ghi": np.asarray(ghi), "temp": np.asarray(temp), "label": labels, "pub": pub}
        self._wx_cache[key] = out
        return out

    def observation_provenance(self) -> Dict[str, Any]:
        """Source label and publication step of every price in the current observation.

        Keys ``dam``, ``id``, ``imb``, ``afrr_cap``, ``pv_forecast``: arrays over
        the 96 delivery steps with the publication step of the value shown for
        that step (``ALWAYS`` for forecasts built from earlier days). Used by
        the observation audit (test U7).
        """
        t = min(self.t, self.episode_len - 1)
        committed = getattr(self, "day_ahead_committed", True)
        if not self._filter_on:
            raise RuntimeError("observation_provenance() needs info_filter=True.")
        seen = self._observed_prices(t, committed)
        step_of = {"dam": self._pub_step["DAM"], "dam_forecast": ALWAYS, "ida1": self._pub_step["IDA1"],
                   "ida2": self._pub_step["IDA2"], "ida3": self._pub_step["IDA3"]}
        to_steps = lambda labels: np.array([step_of[label] for label in labels], dtype=np.int64)  # noqa: E731
        return {
            "t": t,
            "dam_label": seen["dam_label"], "dam_pub": to_steps(seen["dam_label"]),
            "id_label": seen["id_label"], "id_pub": to_steps(seen["id_label"]),
            "imb_label": seen["imb_label"], "imb_pub": to_steps(seen["imb_label"]),
            "afrr_cap_pub": np.full(self.episode_len, self._pub_step["ISP1"], dtype=np.int64),
            "pv_forecast_label": self._observed_weather(t, committed)["label"],
            "pv_forecast_pub": self._observed_weather(t, committed)["pub"],
            "values": {"dam": seen["dam"], "id": seen["id"], "imb": seen["imb"]},
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        day: Optional[int] = None
        if "start_idx" in options:
            start_idx = int(options["start_idx"])
        elif "day" in options:
            day = int(options["day"])
            start_idx = self.day_start_index(day)
        elif self.options.random_start:
            day = int(self.episode_days[int(self.np_random.integers(len(self.episode_days)))])
            start_idx = self.day_start_index(day)
        else:
            start_idx = 0
        if start_idx + self.episode_len > len(self.df_all):
            start_idx = max(0, len(self.df_all) - self.episode_len)

        self._i0 = int(start_idx)
        self._init_episode_arrays(self._i0, options.get("activation_seed"))
        self.t = 0
        self.cum_profit = 0.0
        self.log = []
        self.battery = Battery(
            self.plant.battery,
            soc_init=float(options.get("soc_init", self.soc_init)),
            aux_drains=self.options.battery_aux_drains,
            directional_taper=self.options.directional_taper,
            degradation_on_ac_power=self.options.degradation_on_ac_power,
            energy_limited_power=self.options.energy_limited_power,
        )
        self.cap_up_commit_kw = 0.0
        self.cap_down_commit_kw = 0.0
        self.block_steps_left = 0
        info: Dict[str, Any] = {"start_idx": int(start_idx), "activation_seed": self.activation_seed}
        if day is not None:
            info["day"] = day
        return info

    # ------------------------------------------------------------------
    # Row-based helpers (kept for the legacy hierarchical environment)
    # ------------------------------------------------------------------
    def _row(self, t: int) -> pd.Series:
        return self.df_all.iloc[self._i0 + int(t)]

    def _time_features(self, ts: pd.Timestamp) -> tuple[float, float]:
        ts = pd.to_datetime(ts)
        minutes = ts.hour * 60 + ts.minute
        angle = 2 * np.pi * (minutes / (24 * 60))
        return float(np.sin(angle)), float(np.cos(angle))

    def _weather_power(self, row: pd.Series, forecast: bool = False) -> float:
        if forecast:
            ghi_col = self.schema.ghi_fcst_wm2 if self.schema.ghi_fcst_wm2 in row.index else self.schema.ghi_wm2
            temp_col = self.schema.temp_air_fcst_c if self.schema.temp_air_fcst_c in row.index else self.schema.temp_air_c
            wind_col = self.schema.wind_fcst_mps if (self.schema.wind_fcst_mps and self.schema.wind_fcst_mps in row.index) else self.schema.wind_mps
        else:
            ghi_col = self.schema.ghi_wm2
            temp_col = self.schema.temp_air_c
            wind_col = self.schema.wind_mps

        wind = None if (wind_col is None or wind_col not in row.index) else float(row[wind_col])
        return self.pv_model.ac_power_kw(
            pd.to_datetime(row[self.schema.ts_col]),
            float(row[ghi_col]),
            float(row[temp_col]),
            wind,
        )

    @staticmethod
    def _activation_from_signal(value: float, cap_kw: float) -> float:
        if 0.0 <= value <= 1.0:
            return value * cap_kw
        return min(abs(value), cap_kw)

    def _latch_afrr_block(self, cap_up_kw: float, cap_down_kw: float) -> None:
        if self.block_steps_left <= 0:
            self.cap_up_commit_kw = float(cap_up_kw)
            self.cap_down_commit_kw = float(cap_down_kw)
            self.block_steps_left = self.afrr_block_steps

    # ------------------------------------------------------------------
    # Settlement of one 15-minute step
    # ------------------------------------------------------------------
    def _settle_step(
        self,
        row: Any = None,
        *,
        dam_position_kw: float,
        id_adjustment_kw: float,
        p_batt_base_kw: float,
        cap_up_req_kw: float,
        cap_down_req_kw: float,
    ) -> tuple[float, Dict[str, Any]]:
        assert self.battery is not None
        dt = self.market.dt_h
        i = self._i0 + self.t
        self._latch_afrr_block(cap_up_req_kw, cap_down_req_kw)
        cap_up_kw = self.cap_up_commit_kw
        cap_down_kw = self.cap_down_commit_kw

        pv_kw = float(self._pv_act[i])
        pv_fcst_kw = float(self._observed_weather(self.t, bool(getattr(self, "day_ahead_committed", True)))["pv"][self.t])

        ppcs = self.plant.battery.p_pcs_kw
        violation_kw = 0.0
        if p_batt_base_kw + cap_up_kw > ppcs + 1e-6:
            violation_kw += p_batt_base_kw + cap_up_kw - ppcs
        if (-p_batt_base_kw) + cap_down_kw > ppcs + 1e-6:
            violation_kw += (-p_batt_base_kw) + cap_down_kw - ppcs

        act_up_kw = self._activation_from_signal(float(self._ep_act_up[self.t]), cap_up_kw)
        act_down_kw = self._activation_from_signal(float(self._ep_act_dn[self.t]), cap_down_kw)
        p_act_kw = act_up_kw - act_down_kw

        p_batt_req_kw = p_batt_base_kw + p_act_kw
        batt_res = self.battery.step(p_batt_req_kw, dt_h=dt)
        p_batt_real_kw = float(batt_res["p_real_kw"])

        shortfall_kw = max(0.0, abs(p_batt_req_kw) - abs(p_batt_real_kw))
        shortfall_mwh = (shortfall_kw * dt) / 1000.0

        net_injection_kw = pv_kw + p_batt_real_kw
        grid_res = self.grid.apply(net_injection_kw)
        export_kw = float(grid_res["export_kw"])
        import_kw = float(grid_res["import_kw"])

        price_dam = float(self._dam[i])
        price_id = float(self._idp[i])
        price_imb = float(self._imb[i])
        en_up = price_imb if self._en_up is None else float(self._en_up[i])
        en_down = price_imb if self._en_dn is None else float(self._en_dn[i])

        # What the meter sees, with or without the grid loss factor.
        if self.options.metering_losses:
            metered_kw = float(grid_res["inj_metered_kw"])
        else:
            metered_kw = export_kw - import_kw

        if self.options.settlement == "v1":
            # Conference settlement: requested activation is not part of the
            # scheduled energy, so activated energy is in the imbalance volume
            # *and* paid again as activation energy (both with a plus sign).
            delivered_vs_schedule_kw = metered_kw
            afrr_activation_eur = (
                eur_from_kwh(kwh_from_kw(act_up_kw, dt), en_up)
                + eur_from_kwh(kwh_from_kw(act_down_kw, dt), en_down)
            )
        else:
            # v2: E_sch = (q + a+ r+ - a- r-) dt. The requested balancing energy
            # is removed from the imbalance volume and settled once, at the
            # balancing energy price (received for up, paid back for down).
            delivered_vs_schedule_kw = metered_kw - p_act_kw
            afrr_activation_eur = (
                eur_from_kwh(kwh_from_kw(act_up_kw, dt), en_up)
                - eur_from_kwh(kwh_from_kw(act_down_kw, dt), en_down)
            )
        energy_layers = settle_energy_layers(
            dam_position_kw=dam_position_kw,
            id_adjustment_kw=id_adjustment_kw,
            actual_export_kw=delivered_vs_schedule_kw,
            actual_import_kw=0.0,
            price_dam_eur_mwh=price_dam,
            price_id_eur_mwh=price_id,
            price_imb_eur_mwh=price_imb,
            dt_h=dt,
        )

        afrr_cap_eur = (
            reserve_capacity_eur(cap_up_kw, float(self._cup[i]), dt)
            + reserve_capacity_eur(cap_down_kw, float(self._cdn[i]), dt)
        )

        shortfall_penalty = (
            float(self._pen_short[i]) if self._pen_short is not None else self.market.fallback_shortfall_penalty_eur_per_mwh
        )
        unavailable_penalty = (
            float(self._pen_unav[i]) if self._pen_unav is not None else self.market.fallback_unavailable_penalty_eur_per_mw
        )

        penalties_eur = (
            self.market.violation_penalty_eur_per_kw * violation_kw
            + shortfall_penalty * shortfall_mwh
            + unavailable_penalty * (violation_kw / 1000.0)
        )

        breakdown = SettlementBreakdown(
            dam_eur=energy_layers["dam_eur"],
            id_eur=energy_layers["id_eur"],
            imbalance_eur=energy_layers["imbalance_eur"],
            afrr_capacity_eur=afrr_cap_eur,
            afrr_activation_eur=afrr_activation_eur,
            degradation_eur=float(batt_res["deg_cost"]),
            penalties_eur=penalties_eur,
        )

        reward = breakdown.total_eur
        self.cum_profit += reward

        info = {
            "t": self.t,
            "timestamp": self._ts[i].isoformat(),
            "reward": float(reward),
            "cum_profit": float(self.cum_profit),
            "pv_kw": float(pv_kw),
            "pv_fcst_kw": float(pv_fcst_kw),
            "dam_position_kw": float(dam_position_kw),
            "id_adjustment_kw": float(id_adjustment_kw),
            "scheduled_position_kw": float(dam_position_kw + id_adjustment_kw),
            "p_batt_base_kw": float(p_batt_base_kw),
            "p_batt_req_kw": float(p_batt_req_kw),
            "p_batt_real_kw": float(p_batt_real_kw),
            "soc": float(self.battery.soc),
            "export_kw": export_kw,
            "import_kw": import_kw,
            "metered_kw": float(metered_kw),
            "imbalance_kwh": float(energy_layers["imbalance_kwh"]),
            "curtailed_kw": float(grid_res["curtailed_kw"]),
            "throughput_kwh": float(batt_res["throughput_kwh"]),
            "cap_up_commit_kw": float(cap_up_kw),
            "cap_down_commit_kw": float(cap_down_kw),
            "act_up_kw": float(act_up_kw),
            "act_down_kw": float(act_down_kw),
            "price_dam_eur_mwh": price_dam,
            "price_id_eur_mwh": price_id,
            "price_imb_eur_mwh": price_imb,
            "dam_eur": float(breakdown.dam_eur),
            "id_eur": float(breakdown.id_eur),
            "imbalance_eur": float(breakdown.imbalance_eur),
            "afrr_capacity_eur": float(breakdown.afrr_capacity_eur),
            "afrr_activation_eur": float(breakdown.afrr_activation_eur),
            "degradation_eur": float(breakdown.degradation_eur),
            "penalties_eur": float(breakdown.penalties_eur),
            "violation_kw": float(violation_kw),
            "shortfall_mwh": float(shortfall_mwh),
            "penalty_params": {
                "violation_penalty_eur_per_kw": self.market.violation_penalty_eur_per_kw,
                "shortfall_penalty_eur_per_mwh": float(shortfall_penalty),
                "unavailable_penalty_eur_per_mw": float(unavailable_penalty),
            },
        }
        return reward, info


class HierarchicalPVBESSGreeceEnv(_BasePVBESSGreeceEnv):
    """Hierarchical Gymnasium environment for PV+BESS market participation.

    The environment separates strategic and operational decisions:

    High-level action, applied only at decision boundaries:
    - day-ahead position at the point of common coupling
    - intraday adjustment
    - upward aFRR capacity commitment
    - downward aFRR capacity commitment

    Low-level action, applied at every step:
    - battery dispatch setpoint
    - real-time commercial bias, used here as a fast intraday correction term

    Observation dictionary:
    - ``physical``: current plant state, current prices, current activation signal,
      and timing features
    - ``forecast``: future forecast features over a finite horizon
    - ``action_mask``: tells the agent when a new high-level decision is due
    """

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
        hierarchy: Optional[HierarchicalConfig] = None,
        options: Optional[SimOptions] = None,
        information: Optional[InformationConfig] = None,
        episode_days: Optional[Sequence[int]] = None,
        info_data: Optional[InformationData] = None,
    ):
        self.hierarchy = hierarchy or HierarchicalConfig()
        self.hl_dam_position_kw = 0.0
        self.hl_id_adjustment_kw = 0.0
        self.hl_cap_up_kw = 0.0
        self.hl_cap_down_kw = 0.0
        self.hl_steps_left = 0
        super().__init__(
            signals=signals,
            schema=schema,
            plant=plant,
            market=market,
            episode_len=episode_len,
            soc_init=soc_init,
            include_forecasts=include_forecasts,
            forecast_horizon_steps=forecast_horizon_steps,
            options=options,
            information=information,
            episode_days=episode_days,
            info_data=info_data,
        )
        self._build_spaces()

    def _build_spaces(self) -> None:
        ppcs = self.plant.battery.p_pcs_kw
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        self.decision_block_steps = max(1, int(round(self.hierarchy.decision_block_minutes / (self.market.dt_h * 60))))

        self.action_space = spaces.Dict(
            {
                "high_level": spaces.Box(
                    low=np.array([-max_sched, -max_sched, 0.0, 0.0], dtype=np.float32),
                    high=np.array([max_sched, max_sched, ppcs, ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
                "low_level": spaces.Box(
                    low=np.array([-ppcs, -max_sched], dtype=np.float32),
                    high=np.array([ppcs, max_sched], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        physical_dim = 18
        forecast_dim = self.forecast_horizon_steps * 7 if self.include_forecasts else 0
        self.observation_space = spaces.Dict(
            {
                "physical": spaces.Box(
                    low=np.full(physical_dim, -1e9, dtype=np.float32),
                    high=np.full(physical_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "forecast": spaces.Box(
                    low=np.full(forecast_dim, -1e9, dtype=np.float32),
                    high=np.full(forecast_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.hl_dam_position_kw = 0.0
        self.hl_id_adjustment_kw = 0.0
        self.hl_cap_up_kw = 0.0
        self.hl_cap_down_kw = 0.0
        self.hl_steps_left = 0
        info = super().reset(seed=seed, options=options)
        return self._get_obs(), info

    def _strategic_due(self) -> bool:
        return getattr(self, "hl_steps_left", 0) <= 0

    def _forecast_matrix(self) -> List[float]:
        out: List[float] = []
        if not self.include_forecasts:
            return out
        for k in range(1, self.forecast_horizon_steps + 1):
            tt = min(self.t + k, self.episode_len - 1)
            row = self._row(tt)
            out.extend(
                [
                    self._weather_power(row, forecast=True),
                    float(row[self.schema.ghi_fcst_wm2]) if self.schema.ghi_fcst_wm2 in row.index else float(row[self.schema.ghi_wm2]),
                    float(row[self.schema.temp_air_fcst_c]) if self.schema.temp_air_fcst_c in row.index else float(row[self.schema.temp_air_c]),
                    float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.afrr_cap_up_eur_mw_h]),
                    float(row[self.schema.afrr_cap_down_eur_mw_h]),
                ]
            )
        return out

    def _get_obs(self) -> Dict[str, np.ndarray]:
        row = self._row(self.t)
        tod_sin, tod_cos = self._time_features(pd.to_datetime(row[self.schema.ts_col]))
        wind = 0.0 if (self.schema.wind_mps is None or self.schema.wind_mps not in row.index) else float(row[self.schema.wind_mps])
        act_up_signal = float(row[self.schema.afrr_act_up]) if (self.schema.afrr_act_up and self.schema.afrr_act_up in row.index) else 0.0
        act_down_signal = float(row[self.schema.afrr_act_down]) if (self.schema.afrr_act_down and self.schema.afrr_act_down in row.index) else 0.0

        physical = np.array(
            [
                float(self.battery.soc),
                float(self.battery.e_kwh),
                float(self._weather_power(row, forecast=False)),
                float(self._weather_power(row, forecast=True)),
                float(row[self.schema.ghi_wm2]),
                float(row[self.schema.temp_air_c]),
                float(wind),
                float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_imb_eur_mwh]),
                float(row[self.schema.afrr_cap_up_eur_mw_h]),
                float(row[self.schema.afrr_cap_down_eur_mw_h]),
                float(act_up_signal),
                float(act_down_signal),
                float(getattr(self, "hl_steps_left", 0)),
                float(self.block_steps_left),
                tod_sin,
                tod_cos,
            ],
            dtype=np.float32,
        )

        return {
            "physical": physical,
            "forecast": np.array(self._forecast_matrix(), dtype=np.float32),
            "action_mask": np.array([1.0 if self._strategic_due() else 0.0, 1.0], dtype=np.float32),
        }

    def _maybe_update_high_level(self, high_level: np.ndarray) -> bool:
        if self._strategic_due():
            hl = np.array(high_level, dtype=np.float32)
            self.hl_dam_position_kw = float(hl[0])
            self.hl_id_adjustment_kw = float(hl[1])
            self.hl_cap_up_kw = float(max(0.0, hl[2]))
            self.hl_cap_down_kw = float(max(0.0, hl[3]))
            self.hl_steps_left = self.decision_block_steps
            return True
        return False

    def step(self, action: Dict[str, np.ndarray]):
        row = self._row(self.t)
        high_level = action.get("high_level", np.zeros(4, dtype=np.float32))
        low_level = np.array(action.get("low_level", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        high_level_applied = self._maybe_update_high_level(high_level)

        p_batt_base_kw = float(low_level[0])
        realtime_trading_bias_kw = float(low_level[1])
        reward, info = self._settle_step(
            row=row,
            dam_position_kw=self.hl_dam_position_kw,
            id_adjustment_kw=self.hl_id_adjustment_kw + realtime_trading_bias_kw,
            p_batt_base_kw=p_batt_base_kw,
            cap_up_req_kw=self.hl_cap_up_kw,
            cap_down_req_kw=self.hl_cap_down_kw,
        )
        info.update(
            {
                "high_level_due": bool(self._strategic_due()),
                "high_level_applied": bool(high_level_applied),
                "hl_dam_position_kw": float(self.hl_dam_position_kw),
                "hl_id_adjustment_kw": float(self.hl_id_adjustment_kw),
                "hl_cap_up_kw": float(self.hl_cap_up_kw),
                "hl_cap_down_kw": float(self.hl_cap_down_kw),
                "realtime_trading_bias_kw": float(realtime_trading_bias_kw),
            }
        )
        self.log.append(info)

        self.block_steps_left = max(0, self.block_steps_left - 1)
        self.hl_steps_left = max(0, self.hl_steps_left - 1)
        self.t += 1
        terminated = self.t >= self.episode_len
        if terminated:
            next_obs = {
                "physical": np.zeros(self.observation_space["physical"].shape, dtype=np.float32),
                "forecast": np.zeros(self.observation_space["forecast"].shape, dtype=np.float32),
                "action_mask": np.zeros(self.observation_space["action_mask"].shape, dtype=np.float32),
            }
        else:
            next_obs = self._get_obs()
        return next_obs, float(reward), terminated, False, info



class DayAheadPVBESSGreeceEnv(_BasePVBESSGreeceEnv):
    """Greek-market-sequenced Gymnasium environment for PV+BESS operation.

    This class is kept as a backwards-compatible name. The canonical public
    class is :class:`UnifiedPVBESSGreeceEnv`, which inherits this implementation.

    1. Day-ahead layer: a full delivery-day DAM schedule and the aFRR capacity
       offer are committed once, at D-1.
    2. Intraday layer: corrections are additive to the DAM schedule and are
       written only into future periods after the configured gate closure.
    3. Real-time layer: the action dispatches the battery only. Deviations from
       DAM + intraday are settled at the imbalance price.

    Action dictionary:
    - ``day_ahead``: vector of length ``episode_len`` with the DAM schedule at
      the PCC in kW. Used only while ``action_mask[0] == 1``.
    - ``day_ahead_afrr``: ``[cap_up_kw, cap_down_kw]`` committed with it.
    - ``intraday``: ``[id_adjustment_kw]`` for the block that starts after the
      gate closure. Used only while ``action_mask[1] == 1``.
    - ``real_time``: ``[p_batt_base_kw]`` battery setpoint (discharge positive).
      The older key ``low_level`` is accepted as an alias.

    Observation dictionary:
    - ``physical`` (24,): current plant and market state.
    - ``forecast`` (H*9,): rolling look-ahead for intraday and real-time control.
    - ``day_ahead_context`` (episode_len*8,): full delivery-day view. Each row
      is ``[pv_fcst_kw, price_dam, price_id, price_imb, afrr_cap_up,
      afrr_cap_down, tod_sin, tod_cos]``, flattened row-major.
    - ``action_mask`` (3,): ``[day_ahead_due, intraday_due, real_time_due]``.

    Price entries of the observation follow :class:`InformationConfig`: with
    ``price_noise_sigma > 0`` the intraday and imbalance prices (and the DAM
    prices before commitment) are noisy forecasts, not realised values.
    """

    market_mode = "day_ahead"

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
        timing: Optional[DayAheadMarketTiming] = None,
        options: Optional[SimOptions] = None,
        information: Optional[InformationConfig] = None,
        episode_days: Optional[Sequence[int]] = None,
        info_data: Optional[InformationData] = None,
    ):
        self.timing = timing or DayAheadMarketTiming()
        self.day_ahead_schedule_kw = np.zeros(int(episode_len), dtype=float)
        self.id_adjustment_profile_kw = np.zeros(int(episode_len), dtype=float)
        self.day_ahead_committed = False
        self.day_ahead_source = "uncommitted"
        self.id_steps_left = 0
        self.current_id_offer_kw = 0.0
        self.d1_cap_up_kw = 0.0
        self.d1_cap_down_kw = 0.0
        self.last_intraday_target_start = 0
        self.last_intraday_target_stop = 0
        super().__init__(
            signals=signals,
            schema=schema,
            plant=plant,
            market=market,
            episode_len=episode_len,
            soc_init=soc_init,
            include_forecasts=include_forecasts,
            forecast_horizon_steps=forecast_horizon_steps,
            options=options,
            information=information,
            episode_days=episode_days,
            info_data=info_data,
        )
        self._build_spaces()

    def _build_spaces(self) -> None:
        ppcs = self.plant.battery.p_pcs_kw
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        self.intraday_block_steps = max(
            1,
            int(round(self.timing.intraday_block_minutes / (self.market.dt_h * 60))),
        )
        self.intraday_gate_steps = max(
            0,
            int(round(self.timing.intraday_gate_closure_minutes / (self.market.dt_h * 60))),
        )

        self.action_space = spaces.Dict(
            {
                "day_ahead": spaces.Box(
                    low=np.full(self.episode_len, -max_sched, dtype=np.float32),
                    high=np.full(self.episode_len, max_sched, dtype=np.float32),
                    dtype=np.float32,
                ),
                "day_ahead_afrr": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([ppcs, ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
                "intraday": spaces.Box(
                    low=np.array([-max_sched], dtype=np.float32),
                    high=np.array([max_sched], dtype=np.float32),
                    dtype=np.float32,
                ),
                "real_time": spaces.Box(
                    low=np.array([-ppcs], dtype=np.float32),
                    high=np.array([ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        physical_dim = 24
        forecast_dim = self.forecast_horizon_steps * 9 if self.include_forecasts else 0
        self.observation_space = spaces.Dict(
            {
                "physical": spaces.Box(
                    low=np.full(physical_dim, -1e9, dtype=np.float32),
                    high=np.full(physical_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "forecast": spaces.Box(
                    low=np.full(forecast_dim, -1e9, dtype=np.float32),
                    high=np.full(forecast_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "day_ahead_context": spaces.Box(
                    low=np.full(self.episode_len * 8, -1e9, dtype=np.float32),
                    high=np.full(self.episode_len * 8, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        info = super().reset(seed=seed, options=options)

        self.day_ahead_schedule_kw = np.zeros(self.episode_len, dtype=float)
        self.id_adjustment_profile_kw = np.zeros(self.episode_len, dtype=float)
        self.day_ahead_committed = False
        self.day_ahead_source = "uncommitted"
        self.id_steps_left = 0
        self.current_id_offer_kw = 0.0
        self.d1_cap_up_kw = 0.0
        self.d1_cap_down_kw = 0.0
        self.last_intraday_target_start = 0
        self.last_intraday_target_stop = 0
        self._build_context_cache()

        if "day_ahead_schedule_kw" in options:
            self._commit_day_ahead_schedule(options["day_ahead_schedule_kw"], source="reset_options")
        if "id_adjustment_profile_kw" in options:
            self.id_adjustment_profile_kw = self._coerce_profile(
                options["id_adjustment_profile_kw"],
                name="id_adjustment_profile_kw",
            )
        if "d1_cap_up_kw" in options:
            self.d1_cap_up_kw = float(options["d1_cap_up_kw"])
        if "d1_cap_down_kw" in options:
            self.d1_cap_down_kw = float(options["d1_cap_down_kw"])

        info.update(
            {
                "market_mode": self.market_mode,
                "day_ahead_committed": bool(self.day_ahead_committed),
                "day_ahead_source": self.day_ahead_source,
                "intraday_gate_closure_steps": int(self.intraday_gate_steps),
                "intraday_gate_closure_minutes": int(self.timing.intraday_gate_closure_minutes),
            }
        )
        return self._get_obs(), info

    def _build_context_cache(self) -> None:
        """Day-ahead context before and after the DAM schedule is committed."""
        sl = slice(self._i0, self._i0 + self.episode_len)
        n = len(self._ep_pv_f)
        ctx = np.empty((n, 8), dtype=np.float32)
        ctx[:, 0] = self._ep_pv_f
        ctx[:, 1] = self._ep_dam_f
        ctx[:, 2] = self._ep_id_f
        ctx[:, 3] = self._ep_imb_f
        ctx[:, 4] = self._cup[sl]
        ctx[:, 5] = self._cdn[sl]
        ctx[:, 6] = self._tod_sin[sl]
        ctx[:, 7] = self._tod_cos[sl]
        self._ctx_static = ctx.copy()
        self._ctx_before = ctx.reshape(-1)
        after = ctx.copy()
        after[:, 1] = self._dam[sl]
        self._ctx_after = after.reshape(-1)

    def _coerce_profile(self, values: Any, *, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size != self.episode_len:
            raise ValueError(
                f"{name} must contain exactly {self.episode_len} values for this delivery episode; "
                f"got {arr.size}."
            )
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        return np.clip(arr.astype(float), -max_sched, max_sched)

    def _commit_day_ahead_schedule(self, schedule_kw: Any, *, source: str) -> None:
        self.day_ahead_schedule_kw = self._coerce_profile(schedule_kw, name="day_ahead_schedule_kw")
        self.day_ahead_committed = True
        self.day_ahead_source = source

    def _day_ahead_due(self) -> bool:
        return self.t == 0 and not self.day_ahead_committed

    def _intraday_due(self) -> bool:
        return self.id_steps_left <= 0

    def _intraday_target_slice(self) -> slice:
        start = min(self.episode_len, self.t + self.intraday_gate_steps)
        stop = min(self.episode_len, start + self.intraday_block_steps)
        return slice(start, stop)

    def _maybe_commit_day_ahead_from_action(self, action: Dict[str, Any]) -> bool:
        if not self._day_ahead_due():
            return False
        if "day_ahead" not in action:
            raise ValueError(
                "Day-ahead schedule is due but action['day_ahead'] is missing. "
                "Provide a full episode-length DAM schedule, or pass "
                "reset(options={'day_ahead_schedule_kw': ...}) before stepping."
            )
        self._commit_day_ahead_schedule(action["day_ahead"], source="action")

        if "day_ahead_afrr" in action:
            afrr = np.asarray(action["day_ahead_afrr"], dtype=np.float32).reshape(-1)
            if afrr.size >= 2:
                ppcs = self.plant.battery.p_pcs_kw
                self.d1_cap_up_kw = float(np.clip(afrr[0], 0.0, ppcs))
                self.d1_cap_down_kw = float(np.clip(afrr[1], 0.0, ppcs))
        return True

    def _maybe_update_intraday(self, intraday: Any) -> bool:
        if not self._intraday_due():
            return False

        raw = np.zeros(1, dtype=np.float32) if intraday is None else np.asarray(intraday, dtype=np.float32).reshape(-1)
        if raw.size != 1:
            raise ValueError(
                "action['intraday'] must contain [id_adjustment_kw] only. "
                "aFRR capacity is committed D-1 via action['day_ahead_afrr']."
            )

        clipped = np.clip(raw, self.action_space["intraday"].low, self.action_space["intraday"].high)
        id_adjustment_kw = float(clipped[0])

        target = self._intraday_target_slice()
        if self.options.id_position_limit == "clip" and target.start < target.stop:
            x = self.day_ahead_schedule_kw[target]
            lo = float(np.max(-self.plant.grid.import_cap_kw - x))
            hi = float(np.min(self.plant.grid.export_cap_kw - x))
            id_adjustment_kw = float(np.clip(id_adjustment_kw, lo, hi))
        self.last_intraday_target_start = int(target.start)
        self.last_intraday_target_stop = int(target.stop)
        if target.start < target.stop:
            self.id_adjustment_profile_kw[target] = id_adjustment_kw

        self.current_id_offer_kw = id_adjustment_kw
        self.id_steps_left = self.intraday_block_steps
        return True

    def _forecast_matrix(self, t: int) -> np.ndarray:
        horizon = self.forecast_horizon_steps
        if not self.include_forecasts or horizon <= 0:
            return np.zeros(0, dtype=np.float32)
        tt = np.minimum(t + 1 + np.arange(horizon), self.episode_len - 1)
        gi = self._i0 + tt
        out = np.empty((horizon, 9), dtype=np.float32)
        wx = self._observed_weather(t, self.day_ahead_committed)
        out[:, 0] = wx["pv"][tt]
        out[:, 1] = wx["ghi"][tt]
        out[:, 2] = wx["temp"][tt]
        if self._filter_on:
            seen = self._observed_prices(t, self.day_ahead_committed)
            out[:, 3] = seen["dam"][tt]
            out[:, 4] = seen["id"][tt]
            out[:, 5] = seen["imb"][tt]
        else:
            out[:, 3] = self._dam[gi] if self.day_ahead_committed else self._ep_dam_f[tt]
            out[:, 4] = self._ep_id_f[tt]
            out[:, 5] = self._ep_imb_f[tt]
        out[:, 6] = self._cup[gi]
        out[:, 7] = self._cdn[gi]
        out[:, 8] = self.day_ahead_schedule_kw[tt] + self.id_adjustment_profile_kw[tt]
        return out.reshape(-1)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        t = min(self.t, self.episode_len - 1)
        i = self._i0 + t
        day_ahead_kw = float(self.day_ahead_schedule_kw[t])
        id_adjustment_kw = float(self.id_adjustment_profile_kw[t])
        committed = self.day_ahead_committed
        if self._filter_on:
            seen = self._observed_prices(t, committed)
            dam_now, id_now, imb_now = seen["dam"][t], seen["id"][t], seen["imb"][t]
            context = self._ctx_static.copy()
            context[:, 1], context[:, 2], context[:, 3] = seen["dam"], seen["id"], seen["imb"]
            context = context.reshape(-1)
        else:
            dam_now = self._dam[i] if committed else self._ep_dam_f[t]
            id_now, imb_now = self._idp[i], self._ep_imb_f[t]
            context = (self._ctx_after if committed else self._ctx_before).copy()
        wx = self._observed_weather(t, committed)
        context = context.reshape(self.episode_len, -1)
        context[:, 0] = wx["pv"]
        context = context.reshape(-1)

        physical = np.array(
            [
                self.battery.soc,
                self.battery.e_kwh,
                self._pv_act[i],
                wx["pv"][t],
                self._ghi[i],
                self._temp[i],
                0.0 if self._wind is None else self._wind[i],
                dam_now,
                id_now,
                imb_now,
                self._cup[i],
                self._cdn[i],
                self._ep_act_up[t],
                self._ep_act_dn[t],
                day_ahead_kw,
                id_adjustment_kw,
                day_ahead_kw + id_adjustment_kw,
                self.id_steps_left,
                self.block_steps_left,
                self._tod_sin[i],
                self._tod_cos[i],
                t / max(1, self.episode_len - 1),
                self.d1_cap_up_kw,
                self.d1_cap_down_kw,
            ],
            dtype=np.float32,
        )

        return {
            "physical": physical,
            "forecast": self._forecast_matrix(t),
            "day_ahead_context": context,
            "action_mask": np.array(
                [
                    1.0 if self._day_ahead_due() else 0.0,
                    1.0 if self._intraday_due() else 0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        }

    def step(self, action: Dict[str, Any]):
        action = action or {}
        day_ahead_due_before = self._day_ahead_due()
        intraday_due_before = self._intraday_due()

        day_ahead_applied = self._maybe_commit_day_ahead_from_action(action)
        intraday_applied = self._maybe_update_intraday(action.get("intraday"))

        real_time_action = action.get("real_time", action.get("low_level", np.zeros(1, dtype=np.float32)))
        real_time = np.array(real_time_action, dtype=np.float32).reshape(-1)
        if real_time.size != 1:
            raise ValueError("action['real_time'] must contain [p_batt_base_kw] in the unified environment.")
        real_time = np.clip(real_time, self.action_space["real_time"].low, self.action_space["real_time"].high)

        dam_position_kw = float(self.day_ahead_schedule_kw[self.t])
        id_adjustment_kw = float(self.id_adjustment_profile_kw[self.t])
        reward, info = self._settle_step(
            dam_position_kw=dam_position_kw,
            id_adjustment_kw=id_adjustment_kw,
            p_batt_base_kw=float(real_time[0]),
            cap_up_req_kw=self.d1_cap_up_kw,
            cap_down_req_kw=self.d1_cap_down_kw,
        )
        info.update(
            {
                "market_mode": self.market_mode,
                "day_ahead_due": bool(day_ahead_due_before),
                "day_ahead_applied": bool(day_ahead_applied),
                "day_ahead_source": self.day_ahead_source,
                "intraday_due": bool(intraday_due_before),
                "intraday_applied": bool(intraday_applied),
                "intraday_gate_closure_steps": int(self.intraday_gate_steps),
                "intraday_target_start": int(self.last_intraday_target_start),
                "intraday_target_stop": int(self.last_intraday_target_stop),
                "day_ahead_schedule_kw": dam_position_kw,
                "current_id_offer_kw": float(self.current_id_offer_kw),
                "id_adjustment_kw": id_adjustment_kw,
                "d1_cap_up_kw": float(self.d1_cap_up_kw),
                "d1_cap_down_kw": float(self.d1_cap_down_kw),
                "realtime_trading_bias_kw": 0.0,
            }
        )
        self.log.append(info)

        self.block_steps_left = max(0, self.block_steps_left - 1)
        self.id_steps_left = max(0, self.id_steps_left - 1)
        self.t += 1
        done = self.t >= self.episode_len
        if done and not self.options.truncate_at_day_end:
            next_obs = {
                key: np.zeros(space.shape, dtype=np.float32) for key, space in self.observation_space.spaces.items()
            }
            return next_obs, float(reward), True, False, info
        # The delivery day is a fixed horizon, not an absorbing state: the plant
        # keeps operating, so the episode is truncated and the final observation
        # is a regular one (last market row, final SOC).
        return self._get_obs(), float(reward), False, bool(done), info


class UnifiedPVBESSGreeceEnv(DayAheadPVBESSGreeceEnv):
    """Canonical Gymnasium environment for the PV+BESS Greek-market simulator.

    One Gym env with one ``reset()``, one ``step()``, one ``action_space`` and
    one ``observation_space``. The market hierarchy is represented by
    multi-rate action blocks and an action mask:

    - ``day_ahead`` and ``day_ahead_afrr``: DAM schedule and aFRR capacity,
      committed once at D-1.
    - ``intraday``: additive energy correction, updated on block boundaries
      and applied only after the gate-closure lead time.
    - ``real_time``: battery dispatch setpoint, applied every step.
    """

    market_mode = "unified"
