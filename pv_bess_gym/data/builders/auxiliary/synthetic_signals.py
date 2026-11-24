from __future__ import annotations

import numpy as np
import pandas as pd


def build_weather_forecasts(actuals: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate reproducible synthetic forecasts from actual weather.

    These columns are synthetic by construction. The model uses lagged/smoothed
    actuals and correlated noise so the forecasts are plausible without being
    perfect. GHI gets larger, daylight-dependent errors than temperature or wind.
    """

    rng = np.random.default_rng(seed)
    idx = actuals.index

    ghi = actuals["ghi_wm2"].astype(float)
    temp = actuals["temp_air_c"].astype(float)
    wind = actuals["wind_mps"].astype(float)

    hour_angle = 2.0 * np.pi * ((idx.hour * 60 + idx.minute) / (24.0 * 60.0))
    daylight_weight = np.clip(ghi / max(float(ghi.max()), 1.0), 0.0, 1.0)

    ghi_smooth = ghi.rolling(8, min_periods=1).mean().shift(1).bfill()
    temp_smooth = temp.rolling(12, min_periods=1).mean().shift(1).bfill()
    wind_smooth = wind.rolling(6, min_periods=1).mean().shift(1).bfill()

    ghi_noise = _ar1_noise(rng, len(idx), phi=0.90, sigma=55.0) * (0.35 + 0.95 * daylight_weight.to_numpy())
    temp_noise = _ar1_noise(rng, len(idx), phi=0.93, sigma=0.55) + 0.35 * np.sin(hour_angle + 0.4)
    wind_noise = _ar1_noise(rng, len(idx), phi=0.82, sigma=0.65)

    ghi_fcst = np.clip(0.88 * ghi_smooth.to_numpy() + 0.12 * ghi.to_numpy() + ghi_noise, 0.0, None)
    temp_fcst = temp_smooth.to_numpy() + temp_noise
    wind_fcst = np.clip(0.75 * wind_smooth.to_numpy() + 0.25 * wind.to_numpy() + wind_noise, 0.0, None)

    return pd.DataFrame(
        {
            "ghi_fcst_wm2": ghi_fcst,
            "temp_air_fcst_c": temp_fcst,
            "wind_fcst_mps": wind_fcst,
        },
        index=idx,
    )


def simulate_afrr_activation(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` steps of sparse, autocorrelated aFRR activation ratios in [0, 1].

    Three-state Markov chain (idle / up episode / down episode) with a small
    leakage on the inactive side. The chain starts idle. Returns ``(up, down)``.
    """
    states = np.zeros(n, dtype=np.int8)
    up = np.zeros(n, dtype=float)
    down = np.zeros(n, dtype=float)

    level = 0.0
    for i in range(n):
        state = states[i - 1] if i > 0 else 0
        if state == 0:
            draw = rng.random()
            if draw < 0.018:
                state = 1
                level = rng.uniform(0.18, 0.55)
            elif draw < 0.036:
                state = -1
                level = rng.uniform(0.18, 0.55)
            else:
                level *= 0.20
        else:
            stay_prob = 0.82
            if rng.random() > stay_prob:
                if rng.random() < 0.08:
                    state = -state
                    level = rng.uniform(0.12, 0.40)
                else:
                    state = 0
                    level *= 0.35
            else:
                level = 0.80 * level + rng.uniform(0.04, 0.18)
        level = float(np.clip(level, 0.0, 1.0))
        states[i] = state
        leakage = float(np.clip(rng.beta(1.2, 18.0) * 0.10, 0.0, 0.12))
        if state > 0:
            up[i] = level
            down[i] = leakage
        elif state < 0:
            down[i] = level
            up[i] = leakage
        else:
            up[i] = leakage * 0.40
            down[i] = leakage * 0.40
    return np.clip(up, 0.0, 1.0), np.clip(down, 0.0, 1.0)


def build_afrr_activation_signals(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Create sparse, autocorrelated synthetic aFRR activations in [0, 1]."""
    up, down = simulate_afrr_activation(np.random.default_rng(seed), len(index))
    return pd.DataFrame({"afrr_act_up": up, "afrr_act_down": down}, index=index)


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    innovations = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovations[i]
    out[0] = innovations[0]
    return out
