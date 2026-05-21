from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pvlib


@dataclass
class PVConfig:
    p_dc_kw: float = 550.0
    p_ac_kw: float = 550.0
    latitude: float = 37.98
    longitude: float = 23.72
    altitude_m: float = 70.0
    tz: str = "Europe/Athens"
    tilt_deg: float = 20.0
    azimuth_deg: float = 180.0
    albedo: float = 0.2
    system_losses_frac: float = 0.14
    gamma_pdc_per_c: float = -0.0035
    eta_inv_nom: float = 0.965
    eta_inv_ref: float = 0.963
    default_wind_mps: float = 1.0


class PVModel:
    """Weather-driven PV model using pvlib.

    Inputs:
      - timestamp
      - GHI [W/m²]
      - ambient temperature [°C]
      - wind speed [m/s] optional

    Outputs:
      - AC power at inverter output [kW]
    """

    def __init__(self, cfg: PVConfig):
        self.cfg = cfg
        self.location = pvlib.location.Location(
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            tz=cfg.tz,
            altitude=cfg.altitude_m,
        )

    def ac_power_kw(
        self,
        ts: pd.Timestamp,
        ghi_wm2: float,
        temp_air_c: float,
        wind_mps: float | None = None,
    ) -> float:
        cfg = self.cfg
        if ts.tzinfo is None:
            times = pd.DatetimeIndex([ts]).tz_localize("UTC").tz_convert(cfg.tz)
        else:
            times = pd.DatetimeIndex([ts]).tz_convert(cfg.tz)

        ghi = max(0.0, float(ghi_wm2))
        solar_position = self.location.get_solarposition(times)
        zenith = float(solar_position["apparent_zenith"].iloc[0])
        azimuth = float(solar_position["azimuth"].iloc[0])

        # GHI -> DNI/DHI decomposition
        erbs = pvlib.irradiance.erbs(
            ghi=pd.Series([ghi], index=times),
            zenith=pd.Series([zenith], index=times),
            datetime_or_doy=times,
        )
        dni = max(0.0, float(erbs["dni"].iloc[0]))
        dhi = max(0.0, float(erbs["dhi"].iloc[0]))

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=cfg.tilt_deg,
            surface_azimuth=cfg.azimuth_deg,
            solar_zenith=zenith,
            solar_azimuth=azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi,
            albedo=cfg.albedo,
        )
        poa_global = max(0.0, float(poa["poa_global"]))

        wind = cfg.default_wind_mps if wind_mps is None else max(0.0, float(wind_mps))
        temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

        temp_cell = float(
            pvlib.temperature.sapm_cell(
                poa_global=pd.Series([poa_global], index=times),
                temp_air=pd.Series([float(temp_air_c)], index=times),
                wind_speed=pd.Series([wind], index=times),
                a=temp_params["a"],
                b=temp_params["b"],
                deltaT=temp_params["deltaT"],
            ).iloc[0]
        )

        pdc_w = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=poa_global,
            pdc0=cfg.p_dc_kw * 1000.0,
            gamma_pdc=cfg.gamma_pdc_per_c,
            temp_cell=temp_cell,
        )
        pdc_kw = max(0.0, float(pdc_w / 1000.0))
        pdc_kw *= (1.0 - cfg.system_losses_frac)

        pac_w = pvlib.inverter.pvwatts(
            pdc=pdc_kw * 1000.0,
            pdc0=cfg.p_dc_kw * 1000.0,
            eta_inv_nom=cfg.eta_inv_nom,
            eta_inv_ref=cfg.eta_inv_ref,
        )
        pac_kw = max(0.0, float(pac_w / 1000.0))
        return min(pac_kw, cfg.p_ac_kw)
