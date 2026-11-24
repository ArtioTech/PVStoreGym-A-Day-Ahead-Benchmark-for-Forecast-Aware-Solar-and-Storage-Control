"""Run the test suite and write results/tests.json (the Part B test table).

    python scripts/make_tests_json.py        # same as: make test
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.tasks import results_root  # noqa: E402

OUT = results_root()
OUT.mkdir(parents=True, exist_ok=True)

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402

# Items of the brief -> substring of the test name that covers it.
ITEMS = {
    "R4: balanced D-1 plan delivered exactly has zero imbalance": "test_R4_balanced_d1_plan_delivered_exactly_has_zero_imbalance",
    "R4 flag off reproduces version-4 planned imbalance": "test_R4_flag_off_reproduces_version_4_plan_imbalance",
    "U7 weather audit: no forecast before its issue time, full year": "test_U7_weather_full_year_no_forecast_before_its_issue_time",
    "U7 weather provenance in the observation audit": "test_U7_weather_provenance_is_exposed_in_the_observation_audit",
    "R5: kappa scales each layer's residual": "test_R5_kappa_scales_each_layer_residual",
    "R6: DA-ONLY <= ORACLE-DELIVER <= ORACLE": "test_R6_oracle_deliver_between_da_only_and_oracle",
    "check_env on every registered task": "test_check_env_on_every_registered_task",
    "Determinism: same seed and actions, identical step-log hash": "test_determinism_same_seed_same_actions_same_step_log_hash",
    "Determinism: different seeds differ only in stochastic signals": "test_different_seeds_differ_only_in_stochastic_signals",
    "U1 DAM position delivered exactly": "test_U1_",
    "U2 full up/down activation settled once (v2); fails under v1": "test_U2_",
    "U3 correction for a closed block goes to the first open block": "test_U3_",
    "U4 setpoint beyond PCS/SOC limit projected and penalised": "test_U4_",
    "U5 activation above available energy: shortfall recorded and penalised": "test_U5_",
    "U6 SOC energy-balance residual on random admissible days": "test_U6_",
    "U7 observation audit: nothing observed before publication": "test_U7_",
    "U8 inactive action keys ignored": "test_U8_",
    "Oracle >= every baseline on every day and seed": "test_oracle_bounds_every_baseline_on_every_day_and_seed",
}


def main() -> int:
    started = time.time()
    xml_path = OUT / "tests.junit.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.call([sys.executable, "-m", "pytest", "-q", "-W", "ignore", f"--junitxml={xml_path}", "tests"], cwd=ROOT)
    cases = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        cases.append({"test": f"{case.get('classname')}::{case.get('name')}", "outcome": outcome,
                      "seconds": float(case.get("time", 0.0))})
    table = []
    for item, pattern in ITEMS.items():
        matched = [c for c in cases if pattern in c["test"]]
        if not matched:
            status = "missing"
        elif any(c["outcome"] == "failed" for c in matched):
            status = "fail"
        elif all(c["outcome"] == "skipped" for c in matched):
            status = "skipped"
        else:
            status = "pass"
        table.append({"item": item, "status": status, "tests": [c["test"].split("::")[-1] for c in matched],
                      "n_passed": sum(c["outcome"] == "passed" for c in matched),
                      "n_skipped": sum(c["outcome"] == "skipped" for c in matched)})
    summary = {outcome: sum(c["outcome"] == outcome for c in cases) for outcome in ("passed", "failed", "skipped")}
    payload = {"meta": run_metadata(started=started, command="make test"), "summary": summary,
               "brief_items": table, "all_tests": cases}
    (OUT / "tests.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    xml_path.unlink(missing_ok=True)
    width = max(len(row["item"]) for row in table)
    for row in table:
        print(f"{row['item']:<{width}}  {row['status'].upper():7s} ({row['n_passed']} test(s))")
    print(f"\nAll tests: {summary}  ->  {OUT / 'tests.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
