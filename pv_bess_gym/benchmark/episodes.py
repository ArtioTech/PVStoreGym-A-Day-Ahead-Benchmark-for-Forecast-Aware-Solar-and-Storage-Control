"""One-day benchmark episodes: run a controller and collect every metric of the brief."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

COMPONENTS = ["dam_eur", "id_eur", "afrr_capacity_eur", "afrr_activation_eur", "imbalance_eur",
              "degradation_eur", "penalties_eur"]


def run_episode(env: Any, agent: Any, day: int, activation_seed: int, soc_init: float) -> dict[str, Any]:
    """Run one delivery day and return one row of metrics.

    Definitions:

    - ``cost``: sum over steps of limit-violation plus shortfall magnitude [kW]
      (the constraint signal of the constrained RL baseline).
    - reserve deliverability = delivered / requested activation energy, where
      the delivered part of a step is the requested activation minus the
      shortfall of that step (a shortfall is attributed to activation first).
    - equivalent full cycles = battery throughput ``sum |p_bat| dt`` divided by
      twice the usable energy.
    - the ``xy_*`` columns are sufficient statistics for the correlation
      between step imbalance energy and the intraday-imbalance price spread.
    """
    dt = float(env.market.dt_h)
    bat = env.plant.battery
    usable_kwh = (bat.soc_max - bat.soc_min) * bat.e_max_kwh
    agent.reset()
    obs, _ = env.reset(options={"day": int(day), "soc_init": float(soc_init), "activation_seed": int(activation_seed)})
    row: dict[str, Any] = {"day": int(day), "date": env.day_date(day), "activation_seed": int(activation_seed)}
    sums = {key: 0.0 for key in COMPONENTS}
    profit = cost = act_req = act_short = throughput = curtailed = shortfall_mwh = violation = 0.0
    n = sx = sy = sxx = syy = sxy = sign_sum = imb_kwh = 0.0
    violation_steps = shortfall_steps = 0
    scheduled_kwh = metered_kwh = 0.0
    short_imb_steps = long_imb_steps = 0
    decision_seconds = 0.0
    done = False
    while not done:
        t0 = time.perf_counter()
        action = agent.act(obs)
        decision_seconds += time.perf_counter() - t0
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        profit += reward
        for key in COMPONENTS:
            sums[key] += info[key]
        short_kw = info["shortfall_mwh"] * 1000.0 / dt
        requested_kw = info["act_up_kw"] + info["act_down_kw"]
        act_req += requested_kw * dt
        act_short += min(requested_kw, short_kw) * dt
        cost += info["violation_kw"] + short_kw
        violation += info["violation_kw"]
        shortfall_mwh += info["shortfall_mwh"]
        violation_steps += info["violation_kw"] > 1e-3
        shortfall_steps += short_kw > 1e-3
        throughput += abs(info["p_batt_real_kw"]) * dt
        curtailed += info["curtailed_kw"] * dt
        x = info["imbalance_kwh"]
        y = info["price_id_eur_mwh"] - info["price_imb_eur_mwh"]
        n += 1
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
        sign_sum += np.sign(x) if abs(x) > 1e-9 else 0.0
        imb_kwh += x
        # scheduled energy as settled (v2): (DAM + ID + a+ r+ - a- r-) dt
        scheduled_kwh += (info["scheduled_position_kw"] + info["act_up_kw"] - info["act_down_kw"]) * dt
        metered_kwh += info["metered_kw"] * dt
        short_imb_steps += x < -1e-6
        long_imb_steps += x > 1e-6
    steps = int(n)
    row.update({
        "profit_eur": profit, **sums,
        "cost_kw": cost, "violation_kw_sum": violation, "violation_steps": int(violation_steps),
        "shortfall_mwh": shortfall_mwh, "shortfall_steps": int(shortfall_steps),
        "activation_requested_kwh": act_req, "activation_delivered_kwh": act_req - act_short,
        "throughput_kwh": throughput, "equivalent_full_cycles": throughput / (2.0 * usable_kwh),
        "curtailed_kwh": curtailed, "final_soc": float(env.battery.soc),
        "imbalance_kwh": imb_kwh, "imbalance_sign_mean": sign_sum / steps,
        "scheduled_kwh": scheduled_kwh, "metered_kwh": metered_kwh,
        "short_imbalance_steps": int(short_imb_steps), "long_imbalance_steps": int(long_imb_steps),
        "xy_n": n, "xy_sx": sx, "xy_sy": sy, "xy_sxx": sxx, "xy_syy": syy, "xy_sxy": sxy,
        "decision_seconds_per_step": decision_seconds / steps,
    })
    return row
