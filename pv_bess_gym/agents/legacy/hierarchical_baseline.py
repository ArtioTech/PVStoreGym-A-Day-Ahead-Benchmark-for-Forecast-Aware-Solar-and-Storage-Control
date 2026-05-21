from __future__ import annotations

import numpy as np


class HierarchicalHeuristicAgent:
    """Simple two-layer benchmark controller.

    High level:
        At each strategic boundary, choose a commercial position and optional
        aFRR reserve capacities from forecasted PV and reserve prices.

    Low level:
        At every simulator step, move the battery based on price and SOC, then
        apply a small correction for expected PV forecast mismatch.

    This controller is intentionally simple. It is included only as a clear,
    deterministic baseline for sanity checks and algorithm comparisons.
    """

    def __init__(
        self,
        action_space,
        p_pcs_kw: float,
        reserve_fraction: float = 0.20,
        low_price: float = 45.0,
        high_price: float = 95.0,
        reserve_price_threshold: float = 20.0,
    ):
        self.action_space = action_space
        self.p_pcs_kw = float(p_pcs_kw)
        self.reserve_fraction = float(reserve_fraction)
        self.low_price = float(low_price)
        self.high_price = float(high_price)
        self.reserve_price_threshold = float(reserve_price_threshold)
        self.reset()

    def reset(self) -> None:
        self._last_high = np.zeros(4, dtype=np.float32)

    def _forecast_chunks(self, forecast_vec: np.ndarray) -> np.ndarray:
        if forecast_vec.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return forecast_vec.reshape(-1, 7)

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        physical = obs["physical"]
        forecast = obs["forecast"]
        action_mask = obs["action_mask"]

        soc = float(physical[0])
        pv_now = float(physical[2])
        pv_fcst_now = float(physical[3])
        price_dam = float(physical[7])
        cap_up_price = float(physical[10])
        cap_down_price = float(physical[11])

        forecast_matrix = self._forecast_chunks(forecast)

        if action_mask[0] > 0.5:
            pv_fcst_mean = float(forecast_matrix[:, 0].mean()) if forecast_matrix.size else pv_fcst_now
            dam_pos = min(pv_fcst_mean, 0.85 * self.p_pcs_kw)
            id_pos = 0.0

            reserve_kw = self.reserve_fraction * self.p_pcs_kw
            cap_up = reserve_kw if (cap_up_price >= self.reserve_price_threshold and soc > 0.35) else 0.0
            cap_down = reserve_kw if (cap_down_price >= self.reserve_price_threshold and soc < 0.75) else 0.0

            self._last_high = np.array([dam_pos, id_pos, cap_up, cap_down], dtype=np.float32)

        p_batt = 0.0
        if price_dam <= self.low_price and soc < 0.85:
            p_batt = -0.35 * self.p_pcs_kw
        elif price_dam >= self.high_price and soc > 0.20:
            p_batt = 0.35 * self.p_pcs_kw

        p_batt = min(p_batt, self.p_pcs_kw - float(self._last_high[2]))
        p_batt = max(p_batt, -(self.p_pcs_kw - float(self._last_high[3])))

        rt_bias = np.clip(
            pv_now - pv_fcst_now,
            -0.25 * self.p_pcs_kw,
            0.25 * self.p_pcs_kw,
        )

        return {
            "high_level": np.clip(
                self._last_high,
                self.action_space["high_level"].low,
                self.action_space["high_level"].high,
            ),
            "low_level": np.clip(
                np.array([p_batt, rt_bias], dtype=np.float32),
                self.action_space["low_level"].low,
                self.action_space["low_level"].high,
            ),
        }
