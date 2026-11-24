"""Perfect-foresight MILP oracle: a revenue upper bound for the unified env.

The oracle sees the realised PV, prices and aFRR activation ratios of a
delivery day (or of consecutive days with SOC carried) and chooses every
decision the environment offers: the DAM schedule, the D-1 aFRR capacities,
the gate-closure-delayed intraday corrections and the 15-minute battery
setpoints. It maximises the realised reward under the **v2 settlement**.

Why the optimum is an upper bound, by construction
--------------------------------------------------
Every trajectory the environment can produce is feasible for the MILP with
the same reward, because the MILP reproduces the settlement and the penalties
exactly and only *relaxes* the plant physics:

- conversion efficiency is fixed at its maximum (the zero-load efficiency);
  a free, non-negative energy-disposal variable ``w`` in the SOC equation
  absorbs the extra losses of the environment's load-dependent efficiency;
- power limits are untapered (PCS rating only), a superset of the tapered,
  energy-limited limits of the environment;
- PV may be curtailed freely, up to the PV output (the environment curtails
  only above the export limit);
- with ``id_position_limit="clip"`` (rev1 A2) the commercial position ``x + Y``
  is bounded by the grid limits, exactly as in the environment;
- the realised battery power ``p`` lies between 0 and the request
  ``b + a+ r+ - a- r-`` (the environment's projection always returns such a
  value); the undelivered part is the shortfall and is penalised as in the
  environment; setpoint plus reserve above the PCS rating is penalised as in
  the environment;
- when the battery sits at the SOC floor the auxiliary load is not served and
  self-discharge does not lower the energy further (as in the environment).

Hence ``U >= J(pi)`` for every admissible policy on the same realisation. The
reported ``upper_bound_eur`` is the solver's proven dual bound, so it remains
a bound even if the solve stops at a non-zero MIP gap.

``allow_imbalance=False`` adds the constraint that the plan carries no
imbalance volume: ORACLE-DELIVER (realism fix R6). It must meter its own
schedule, ``|metered - E_sch| <= deliver_epsilon_kwh`` per step, where
``E_sch = (DAM + ID + a+ r+ - a- r-) dt`` as in the v2 settlement. Its dual
bound ``U_D`` bounds every policy that delivers its schedule within that
tolerance; it is *not* a bound for policies that carry imbalance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class OracleResult:
    upper_bound_eur: float          # proven dual bound (the number to report as U)
    objective_eur: float            # value of the incumbent plan
    mip_gap: float
    status: str
    solve_seconds: float
    days: list[int]
    activation_seed: Optional[int] = None
    retries: int = 0                # re-solves with a 4x time limit after no incumbent
    plan: dict[str, np.ndarray] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)


class _Model:
    """Tiny helper to assemble a sparse MILP column by column."""

    def __init__(self) -> None:
        self.n = 0
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integer: list[int] = []
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []
        self.cost: dict[int, float] = {}

    def var(self, size: int, lb: float, ub: float, integer: bool = False) -> np.ndarray:
        idx = np.arange(self.n, self.n + size)
        self.n += size
        self.lb += [lb] * size
        self.ub += [ub] * size
        self.integer += [1 if integer else 0] * size
        return idx

    def row(self, terms: Sequence[tuple[int, float]], lb: float, ub: float) -> None:
        r = len(self.row_lb)
        for col, val in terms:
            if val != 0.0:
                self.rows.append(r)
                self.cols.append(int(col))
                self.vals.append(float(val))
        self.row_lb.append(lb)
        self.row_ub.append(ub)

    def add_cost(self, col: int, coeff: float) -> None:
        """Accumulate a *revenue* coefficient (the model maximises)."""
        if coeff != 0.0:
            self.cost[int(col)] = self.cost.get(int(col), 0.0) + float(coeff)


def solve_oracle(
    env: Any,
    days: Sequence[int] | int,
    *,
    soc_init: Optional[float] = None,
    activation_seed: Optional[int] = None,
    allow_imbalance: bool = True,
    deliver_epsilon_kwh: float = 0.0,
    time_limit_s: float = 120.0,
    mip_rel_gap: float = 1e-4,
) -> OracleResult:
    """Solve the perfect-foresight problem for ``days`` of ``env`` (one day, or
    consecutive days with SOC carried).

    Plant, market, gate-closure, degradation and settlement flags are read
    from ``env``, so the oracle is solved for the same task as the controllers
    it bounds. ``activation_seed`` selects the same activation realisation the
    environment draws with ``reset(options={"activation_seed": ...})``.
    """
    opts = env.options
    if opts.settlement != "v2":
        raise ValueError("The oracle models the v2 settlement; build the env with settlement='v2'.")
    days = [int(days)] if np.isscalar(days) else [int(d) for d in days]
    L, N = int(env.episode_len), len(days)
    T = L * N
    dt = float(env.market.dt_h)
    bat, grid, market = env.plant.battery, env.plant.grid, env.market
    P = float(bat.p_pcs_kw)
    aux = float(bat.aux_kw) if opts.battery_aux_drains else -float(bat.aux_kw)
    eta = float(np.clip(bat.eta_nom, 0.85, 0.995))            # maximum efficiency (zero load)
    sd = 1.0 - bat.self_discharge_per_day * (dt / 24.0)
    e_min, e_max = bat.soc_min * bat.e_max_kwh, bat.soc_max * bat.e_max_kwh
    soc0 = env.soc_init if soc_init is None else float(soc_init)
    e0 = float(np.clip(soc0, bat.soc_min, bat.soc_max) * bat.e_max_kwh)
    c_deg = float(bat.deg_eur_per_kwh_throughput)
    loss = float(grid.loss_frac) if opts.metering_losses else 0.0
    cap_exp = float(grid.export_cap_kw) if grid.allow_export else 0.0
    cap_imp = float(grid.import_cap_kw) if grid.allow_import else 0.0
    sched_max = max(grid.export_cap_kw, grid.import_cap_kw)
    gate, block = int(env.intraday_gate_steps), int(env.intraday_block_steps)
    pen_violation = float(market.violation_penalty_eur_per_kw) + float(market.fallback_unavailable_penalty_eur_per_mw) / 1000.0
    pen_shortfall = float(market.fallback_shortfall_penalty_eur_per_mwh)
    if env._pen_short is not None or env._pen_unav is not None:
        raise ValueError("The oracle does not support per-step penalty columns in the dataset.")

    starts = [env.day_start_index(d) for d in days]
    gi = np.concatenate([np.arange(s, s + L) for s in starts])
    pv = env._pv_act[gi]
    dam, idp, imb = env._dam[gi], env._idp[gi], env._imb[gi]
    cup, cdn = env._cup[gi], env._cdn[gi]
    acts = [env.activation_series(s, activation_seed) for s in starts]
    alpha = np.clip(np.concatenate([a[0] for a in acts]), 0.0, 1.0)
    beta = np.clip(np.concatenate([a[1] for a in acts]), 0.0, 1.0)
    en_up = imb if env._en_up is None else env._en_up[gi]
    en_dn = imb if env._en_dn is None else env._en_dn[gi]
    k = dt / 1000.0                                           # kW over one step -> MWh
    M = 3.0 * P                                               # |request| <= P + P, with margin

    m = _Model()
    s = m.var(T, -sched_max, sched_max)                       # DAM schedule
    b = m.var(T, -P, P)                                       # real-time setpoint
    p = m.var(T, -P, P)                                       # realised battery power
    sg = m.var(T, 0.0, 1.0, integer=True)                     # 1: request >= 0 (discharge side)
    short = m.var(T, 0.0, np.inf)                             # undelivered request [kW]
    v1 = m.var(T, 0.0, np.inf)                                # setpoint + up reserve above PCS [kW]
    v2 = m.var(T, 0.0, np.inf)                                # -setpoint + down reserve above PCS [kW]
    dis = m.var(T, 0.0, np.inf)                               # DC-side discharge
    ch = m.var(T, 0.0, np.inf)                                # DC-side charge
    e = m.var(T + 1, e_min, e_max)
    w = m.var(T, 0.0, np.inf)                                 # free energy disposal [kWh]
    unserved = m.var(T, 0.0, max(aux, 0.0))                   # auxiliary load not served at the floor [kW]
    floor = m.var(T, 0.0, 1.0, integer=True)                  # 1: battery at the SOC floor after the step
    g_max = (1.0 - sd) * e_max
    gain = m.var(T, 0.0, g_max)                               # self-discharge the floor clip gives back [kWh]
    curt = m.var(T, 0.0, np.inf)                              # PV curtailment [kW], at most the PV output
    vp = m.var(T, 0.0, cap_exp)                               # export
    vm = m.var(T, 0.0, cap_imp)                               # import
    deg = m.var(T, 0.0, np.inf)                               # |p| or DC throughput for degradation [kW]
    cap_u = m.var(N, 0.0, P)                                  # aFRR up capacity per day
    cap_d = m.var(N, 0.0, P)

    # Intraday decisions: taken at steps 0, block, 2*block, ... of each day and
    # written to [t + gate, t + gate + block) within the same day.
    id_of_step = np.full(T, -1, dtype=int)
    n_id = 0
    for day_i in range(N):
        for t_dec in range(0, L, block):
            lo, hi = min(L, t_dec + gate), min(L, t_dec + gate + block)
            if lo < hi:
                id_of_step[day_i * L + lo: day_i * L + hi] = n_id
                n_id += 1
    a = m.var(max(n_id, 1), -sched_max, sched_max)

    m.row([(e[0], 1.0)], e0, e0)
    for t in range(T):
        d = t // L
        u, dn = cap_u[d], cap_d[d]
        req = [(b[t], 1.0), (u, alpha[t]), (dn, -beta[t])]     # request = b + a+ r+ - a- r-
        # realised power between 0 and the request
        m.row([(p[t], 1.0), (sg[t], -P)], -np.inf, 0.0)                         # p <= P*sg
        m.row([(p[t], 1.0), (sg[t], -P)], -P, np.inf)                           # p >= -P*(1-sg)
        m.row([(p[t], 1.0), (sg[t], M)] + [(c, -v) for c, v in req], -np.inf, M)  # p <= req + M(1-sg)
        m.row([(p[t], 1.0), (sg[t], M)] + [(c, -v) for c, v in req], 0.0, np.inf)  # p >= req - M*sg
        # shortfall = |request - p|
        m.row([(short[t], 1.0), (p[t], 1.0)] + [(c, -v) for c, v in req], 0.0, np.inf)
        m.row([(short[t], 1.0), (p[t], -1.0)] + req, 0.0, np.inf)
        # reserve violation
        m.row([(v1[t], 1.0), (b[t], -1.0), (u, -1.0)], -P, np.inf)
        m.row([(v2[t], 1.0), (b[t], 1.0), (dn, -1.0)], -P, np.inf)
        # DC side: p + aux - unserved = dis - ch
        m.row([(p[t], 1.0), (unserved[t], -1.0), (dis[t], -1.0), (ch[t], 1.0)], -aux, -aux)
        # energy balance at maximum efficiency with free disposal
        m.row(
            [(e[t + 1], 1.0), (e[t], -sd), (dis[t], sd * dt / eta), (ch[t], -sd * dt * eta), (w[t], 1.0), (gain[t], -1.0)],
            0.0, 0.0,
        )
        # at the floor the environment neither serves the auxiliary load nor lets
        # self-discharge pull the energy below e_min
        m.row([(unserved[t], 1.0), (floor[t], -max(aux, 0.0))], -np.inf, 0.0)
        m.row([(gain[t], 1.0), (floor[t], -g_max)], -np.inf, 0.0)
        m.row([(e[t + 1], 1.0), (floor[t], e_max - e_min)], -np.inf, e_max)
        # grid: export - import = pv - curtailment + p
        m.row([(vp[t], 1.0), (vm[t], -1.0), (curt[t], 1.0), (p[t], -1.0)], pv[t], pv[t])
        m.row([(curt[t], 1.0)], 0.0, float(pv[t]))
        # degradation basis
        if opts.degradation_on_ac_power:
            m.row([(deg[t], 1.0), (p[t], -1.0)], 0.0, np.inf)
            m.row([(deg[t], 1.0), (p[t], 1.0)], 0.0, np.inf)
        else:
            m.row([(deg[t], 1.0), (dis[t], -1.0), (ch[t], -1.0)], 0.0, np.inf)

        # --- reward ---------------------------------------------------------
        # imbalance volume = metered - requested activation - DAM - ID
        m.add_cost(s[t], k * (dam[t] - imb[t]))
        if id_of_step[t] >= 0:
            m.add_cost(a[id_of_step[t]], k * (idp[t] - imb[t]))
        m.add_cost(vp[t], k * imb[t] * (1.0 - loss))
        m.add_cost(vm[t], -k * imb[t] * (1.0 + loss))
        m.add_cost(u, k * (cup[t] + alpha[t] * (en_up[t] - imb[t])))
        m.add_cost(dn, k * (cdn[t] - beta[t] * (en_dn[t] - imb[t])))
        m.add_cost(deg[t], -c_deg * dt)
        m.add_cost(v1[t], -pen_violation)
        m.add_cost(v2[t], -pen_violation)
        m.add_cost(short[t], -pen_shortfall * k)

        # rev1 A2: the environment bounds the commercial position x + Y by the grid
        # limits, so the oracle does too (keeps U a bound of the same feasible set, only tighter)
        if getattr(opts, "id_position_limit", "off") == "clip":
            pos = [(s[t], 1.0)] + ([(a[id_of_step[t]], 1.0)] if id_of_step[t] >= 0 else [])
            m.row(pos, -float(grid.import_cap_kw), float(grid.export_cap_kw))

        if not allow_imbalance:
            terms = [(vp[t], 1.0 - loss), (vm[t], -(1.0 + loss)), (u, -alpha[t]), (dn, beta[t]), (s[t], -1.0)]
            if id_of_step[t] >= 0:
                terms.append((a[id_of_step[t]], -1.0))
            eps_kw = float(deliver_epsilon_kwh) / dt
            m.row(terms, -eps_kw, eps_kw)

    c = np.zeros(m.n)
    for col, coeff in m.cost.items():
        c[col] = -coeff                                           # milp minimises
    A = sparse.csr_matrix((m.vals, (m.rows, m.cols)), shape=(len(m.row_lb), m.n))
    started = time.time()
    retries = 0
    limit = float(time_limit_s)
    while True:
        res = milp(
            c,
            constraints=LinearConstraint(A, np.array(m.row_lb), np.array(m.row_ub)),
            integrality=np.array(m.integer),
            bounds=Bounds(np.array(m.lb), np.array(m.ub)),
            options={"time_limit": limit, "mip_rel_gap": float(mip_rel_gap), "disp": False},
        )
        # No incumbent at the time limit (seen with the rev1 position bound): retry
        # once with four times the limit. Never triggered on the v3/v4 protocols,
        # where a missing incumbent raised.
        if res.x is not None or retries >= 1:
            break
        retries += 1
        limit *= 4.0
    elapsed = time.time() - started
    if res.x is None:
        raise RuntimeError(f"Oracle MILP returned no solution: {res.message}")

    x = res.x
    objective = float(-res.fun)
    dual = getattr(res, "mip_dual_bound", None)
    upper = float(-dual) if dual is not None and np.isfinite(dual) else objective
    upper = max(upper, objective)

    a_step = np.where(id_of_step >= 0, x[a][np.clip(id_of_step, 0, None)], 0.0)
    cap_u_step, cap_d_step = np.repeat(x[cap_u], L), np.repeat(x[cap_d], L)
    p_act = alpha * cap_u_step - beta * cap_d_step
    metered = (1.0 - loss) * x[vp] - (1.0 + loss) * x[vm]
    breakdown = {
        "dam_eur": float(np.sum(k * dam * x[s])),
        "id_eur": float(np.sum(k * idp * a_step)),
        "imbalance_eur": float(np.sum(k * imb * (metered - p_act - x[s] - a_step))),
        "afrr_capacity_eur": float(np.sum(k * (cup * cap_u_step + cdn * cap_d_step))),
        "afrr_activation_eur": float(np.sum(k * (en_up * alpha * cap_u_step - en_dn * beta * cap_d_step))),
        "degradation_eur": float(np.sum(c_deg * dt * x[deg])),
        "penalties_eur": float(np.sum(pen_violation * (x[v1] + x[v2]) + pen_shortfall * k * x[short])),
    }
    plan = {
        "day_ahead_kw": x[s].reshape(N, L),
        "id_adjustment_kw": a_step.reshape(N, L),
        "real_time_kw": x[b].reshape(N, L),
        "battery_kw": x[p].reshape(N, L),
        "cap_up_kw": x[cap_u].copy(),
        "cap_down_kw": x[cap_d].copy(),
        "curtailment_kw": x[curt].reshape(N, L),
        "disposal_kwh": x[w].reshape(N, L),
        "soc": x[e] / bat.e_max_kwh,
    }
    return OracleResult(
        upper_bound_eur=upper,
        objective_eur=objective,
        mip_gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
        status=str(res.message),
        solve_seconds=float(elapsed),
        days=days,
        activation_seed=activation_seed,
        retries=retries,
        plan=plan,
        breakdown=breakdown,
    )


class OraclePlanAgent:
    """Replays an oracle plan through the environment (gives ``U_rep``).

    The replayed profit falls short of the bound by exactly the physics the
    MILP relaxes: load-dependent efficiency, power taper and energy limits,
    and curtailment below the export limit, which the environment cannot do.
    """

    name = "oracle-replay"

    def __init__(self, env: Any, result: OracleResult):
        self.env = env
        self.result = result
        self._day = -1

    def reset(self) -> None:
        self._day += 1

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        plan, d, t = self.result.plan, self._day, int(self.env.t)
        L = int(self.env.episode_len)
        gate = int(self.env.intraday_gate_steps)
        target = min(L - 1, t + gate)
        intraday = plan["id_adjustment_kw"][d, target] if t + gate < L else 0.0
        return {
            "day_ahead": plan["day_ahead_kw"][d].astype(np.float32),
            # Actions are float32: shave 1 W off the capacities so rounding cannot
            # push setpoint + reserve above the PCS rating.
            "day_ahead_afrr": np.maximum(
                np.array([plan["cap_up_kw"][d], plan["cap_down_kw"][d]], dtype=np.float64) - 1e-3, 0.0
            ).astype(np.float32),
            "intraday": np.array([intraday], dtype=np.float32),
            "real_time": np.array([plan["real_time_kw"][d, t]], dtype=np.float32),
        }


def replay_oracle(env: Any, result: OracleResult, *, soc_init: float) -> float:
    """Profit of the oracle's actions when played through ``env`` (``U_rep``)."""
    agent = OraclePlanAgent(env, result)
    total, soc = 0.0, float(soc_init)
    for day in result.days:
        agent.reset()
        options = {"day": day, "soc_init": soc}
        if result.activation_seed is not None:
            options["activation_seed"] = result.activation_seed
        obs, _ = env.reset(options=options)
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
            total += reward
            done = terminated or truncated
        soc = float(env.battery.soc)
    return float(total)
