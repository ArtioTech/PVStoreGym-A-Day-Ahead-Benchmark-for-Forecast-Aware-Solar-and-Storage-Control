from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.agents import (  # noqa: E402
    build_registered_algorithm,
    default_algorithm_names,
    get_algorithm_spec,
    list_algorithm_specs,
    normalize_algorithm_name,
)
from pv_bess_gym.evaluation import EvaluationConfig, evaluate_agent, save_evaluation_outputs  # noqa: E402
from pv_bess_gym.envs import make_hierarchical_env, make_unified_env  # noqa: E402

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================
#
# Runs one or more registered benchmark algorithms and saves comparison outputs.
#
# Main benchmarks (unified mode):
#   da-opt-rbc    : D-1 co-optimized DAM+aFRR, RBC SIDC correction, RBC real-time
#   da-opt-mpc    : D-1 co-optimized DAM+aFRR, MPC SIDC correction, MPC real-time
#
# Ablation:
#   da-opt-rt-mpc : D-1 co-optimized DAM+aFRR, RBC SIDC correction, MPC real-time
#
# Add new algorithms in pv_bess_gym/agents/registry.py, then run:
#   python scripts/run_benchmarks.py --algorithm your-new-algorithm
# =============================================================================
DEFAULT_ALGORITHM = "da-opt-rbc"
DEFAULT_MARKET_MODE = "auto"
DEFAULT_START_DAY = 0
DEFAULT_NUM_DAYS = 7
DEFAULT_SOC_INIT = 0.10
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "simulations"


def _parse_days(value: str) -> list[int]:
    days = [chunk.strip() for chunk in value.split(",")]
    out = [int(day) for day in days if day]
    if not out:
        raise argparse.ArgumentTypeError("Expected a comma-separated list like '0,5,12'.")
    return out


def _default_signals_path() -> Path:
    processed = ROOT / "data" / "processed"
    if processed.exists() and any(processed.glob("greek_*.csv")):
        return processed
    return ROOT / "example_data"


def _list_algorithms() -> None:
    print("Available algorithms/controllers:\n")
    for market_mode in ("unified", "rolling"):
        print(f"{market_mode}:")
        for spec in list_algorithm_specs(market_mode=market_mode):
            marker = "benchmark" if spec.benchmark else "custom"
            print(f"  {spec.name:15s} [{marker}] {spec.description}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PV+BESS Gym simulator with a selected algorithm over any "
            "number of days."
        )
    )
    parser.add_argument(
        "--algorithm",
        default=DEFAULT_ALGORITHM,
        help="Algorithm/controller to run. Use 'all' to run all default algorithms for the selected market mode.",
    )
    parser.add_argument(
        "--market-mode",
        choices=["auto", "unified", "day-ahead", "rolling"],
        default=DEFAULT_MARKET_MODE,
        help="'unified' is the canonical Gym simulator. 'day-ahead' is accepted as an alias; 'rolling' is legacy.",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="Print available algorithms/controllers and exit.",
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=None,
        help="Signal file or directory. Defaults to data/processed if available, otherwise example_data.",
    )
    parser.add_argument(
        "--plant-config",
        type=Path,
        default=ROOT / "example_data" / "station_athens.json",
        help="Plant JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/simulations/<market_mode>_<algorithm>.",
    )
    parser.add_argument("--start-day", type=int, default=DEFAULT_START_DAY, help="First dataset day to evaluate.")
    parser.add_argument("--num-days", type=int, default=DEFAULT_NUM_DAYS, help="Number of consecutive days to evaluate.")
    parser.add_argument(
        "--days",
        type=_parse_days,
        help="Optional explicit day indices, e.g. '0,30,120'. Overrides start-day/num-days.",
    )
    parser.add_argument(
        "--carry-soc-between-days",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carry final SOC from one evaluated day to the next.",
    )
    parser.add_argument("--soc-init", type=float, default=DEFAULT_SOC_INIT, help="Initial SOC for the first evaluated day.")
    parser.add_argument("--dt-hours", type=float, default=0.25, help="Simulation step in hours.")
    parser.add_argument("--afrr-block-minutes", type=int, default=60, help="Reserve commitment block length.")
    parser.add_argument(
        "--decision-block-minutes",
        type=int,
        default=60,
        help="Legacy rolling mode only: high-level decision interval.",
    )
    parser.add_argument(
        "--intraday-block-minutes",
        type=int,
        default=60,
        help="Unified mode: intraday correction block length.",
    )
    parser.add_argument(
        "--intraday-gate-closure-minutes",
        type=int,
        default=60,
        help="Unified mode: lead time before an intraday correction can affect delivery.",
    )
    parser.add_argument(
        "--forecast-horizon-steps",
        type=int,
        default=8,
        help="Number of future forecast steps exposed in the observation.",
    )
    parser.add_argument("--violation-penalty", type=float, default=100.0, help="EUR/kW infeasibility penalty.")
    parser.add_argument("--shortfall-penalty", type=float, default=20.0, help="EUR/MWh fallback shortfall penalty.")
    parser.add_argument("--reserve-fraction", type=float, default=0.20, help="RBC reserve fraction.")
    parser.add_argument("--low-price", type=float, default=45.0, help="RBC charge threshold.")
    parser.add_argument("--high-price", type=float, default=95.0, help="RBC discharge threshold.")
    parser.add_argument("--reserve-price-threshold", type=float, default=20.0, help="RBC reserve threshold.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV logs only; skip plot generation.")
    return parser.parse_args()


def _resolve_market_mode(algorithm: str, requested_mode: str) -> str:
    if requested_mode == "day-ahead":
        return "unified"
    if requested_mode != "auto":
        return requested_mode
    return get_algorithm_spec(algorithm).market_mode


def _make_config(args: argparse.Namespace, *, algorithm: str, market_mode: str, output_dir: Path) -> EvaluationConfig:
    return EvaluationConfig(
        signals_path=args.signals or _default_signals_path(),
        plant_config_path=args.plant_config,
        output_dir=output_dir,
        start_day=args.start_day,
        num_days=args.num_days,
        day_indices=args.days,
        carry_soc_between_days=args.carry_soc_between_days,
        soc_init=args.soc_init,
        dt_hours=args.dt_hours,
        afrr_block_minutes=args.afrr_block_minutes,
        decision_block_minutes=args.decision_block_minutes,
        forecast_horizon_steps=args.forecast_horizon_steps,
        violation_penalty=args.violation_penalty,
        shortfall_penalty=args.shortfall_penalty,
        title=f"{market_mode} PV+BESS {algorithm}",
        agent_name=algorithm,
    )


def _build_env(args: argparse.Namespace, config: EvaluationConfig, market_mode: str) -> Any:
    if market_mode == "unified":
        return make_unified_env(
            signals_path=config.signals_path,
            plant_config_path=config.plant_config_path,
            tz=config.tz,
            dt_hours=config.dt_hours,
            afrr_block_minutes=config.afrr_block_minutes,
            intraday_block_minutes=args.intraday_block_minutes,
            intraday_gate_closure_minutes=args.intraday_gate_closure_minutes,
            violation_penalty=config.violation_penalty,
            shortfall_penalty=config.shortfall_penalty,
            soc_init=config.soc_init,
            include_forecasts=config.include_forecasts,
            forecast_horizon_steps=config.forecast_horizon_steps,
        )

    return make_hierarchical_env(
        signals_path=config.signals_path,
        plant_config_path=config.plant_config_path,
        tz=config.tz,
        dt_hours=config.dt_hours,
        afrr_block_minutes=config.afrr_block_minutes,
        decision_block_minutes=config.decision_block_minutes,
        violation_penalty=config.violation_penalty,
        shortfall_penalty=config.shortfall_penalty,
        soc_init=config.soc_init,
        include_forecasts=config.include_forecasts,
        forecast_horizon_steps=config.forecast_horizon_steps,
    )


def _algorithm_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reserve_fraction": args.reserve_fraction,
        "low_price": args.low_price,
        "high_price": args.high_price,
        "reserve_price_threshold": args.reserve_price_threshold,
    }


def _output_dir(args: argparse.Namespace, market_mode: str, algorithm: str, multiple: bool) -> Path:
    if args.output_dir is not None:
        return args.output_dir / f"{market_mode}_{algorithm}" if multiple else args.output_dir
    return DEFAULT_OUTPUT_ROOT / f"{market_mode}_{algorithm}"


def _save_outputs(result: Any, *, output_dir: Path, title: str, no_plots: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if no_plots:
        result.combined_df.to_csv(output_dir / "combined_log.csv", index=False)
        result.summary_df.to_csv(output_dir / "daily_summary.csv", index=False)
        return
    save_evaluation_outputs(result, output_dir=output_dir, title=title)


def _run_one(args: argparse.Namespace, *, algorithm: str, market_mode: str, output_dir: Path) -> dict[str, Any]:
    spec = get_algorithm_spec(algorithm)
    if spec.market_mode != market_mode:
        raise ValueError(
            f"Algorithm '{algorithm}' requires market mode '{spec.market_mode}', "
            f"but '{market_mode}' was requested."
        )

    config = _make_config(args, algorithm=algorithm, market_mode=market_mode, output_dir=output_dir)
    env = _build_env(args, config, market_mode)
    agent = build_registered_algorithm(algorithm, env, _algorithm_options(args))
    result = evaluate_agent(
        env,
        agent,
        start_day=config.start_day,
        num_days=config.num_days,
        day_indices=config.day_indices,
        carry_soc_between_days=config.carry_soc_between_days,
        soc_init=config.soc_init,
    )
    _save_outputs(
        result,
        output_dir=output_dir,
        title=f"{config.title} - {len(result.day_indices)} day(s)",
        no_plots=args.no_plots,
    )

    print(f"\nAlgorithm: {algorithm}")
    print(f"Market mode: {market_mode}")
    print(f"Signals: {config.signals_path}")
    for item in result.daily_summary:
        print(
            f"Day {item['day_index']:02d} | start_idx={item['start_idx']} | "
            f"profit={item['profit_eur']:.2f} EUR | final_soc={item['final_soc']:.3f}"
        )
    print(f"Total profit over {len(result.day_indices)} day(s): {result.total_profit:.2f} EUR")
    print(f"Saved outputs to: {output_dir}")

    return {
        "algorithm": algorithm,
        "market_mode": market_mode,
        "num_days": len(result.day_indices),
        "total_profit_eur": result.total_profit,
        "final_soc": result.final_soc,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    if args.list_algorithms:
        _list_algorithms()
        return

    requested = normalize_algorithm_name(args.algorithm)
    if requested == "all":
        market_mode = "unified" if args.market_mode in {"auto", "day-ahead"} else args.market_mode
        algorithms = default_algorithm_names(market_mode)
    else:
        market_mode = _resolve_market_mode(requested, args.market_mode)
        algorithms = [requested]

    summaries: list[dict[str, Any]] = []
    for algorithm in algorithms:
        output_dir = _output_dir(args, market_mode, algorithm, multiple=len(algorithms) > 1)
        summaries.append(_run_one(args, algorithm=algorithm, market_mode=market_mode, output_dir=output_dir))

    if len(summaries) > 1:
        summary_df = pd.DataFrame(summaries)
        base_dir = args.output_dir or DEFAULT_OUTPUT_ROOT
        base_dir.mkdir(parents=True, exist_ok=True)
        summary_path = base_dir / f"{market_mode}_simulation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nSimulation summary:")
        print(summary_df[["algorithm", "total_profit_eur", "final_soc", "output_dir"]].to_string(index=False))
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
