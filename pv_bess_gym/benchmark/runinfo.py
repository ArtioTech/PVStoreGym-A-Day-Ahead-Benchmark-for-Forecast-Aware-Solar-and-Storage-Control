"""Metadata written into every results file: commit, versions, seed, wall-clock."""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_commit() -> dict[str, Any]:
    # "dirty" refers to code and configuration: generated outputs are excluded.
    status = _git("status", "--porcelain", "--", ".", ":!results", ":!figures", ":!outputs")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status) and status != "unknown"}


def versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"python": platform.python_version()}
    for module in ["pv_bess_gym", "numpy", "scipy", "pandas", "gymnasium", "cvxpy", "clarabel", "highspy",
                   "pvlib", "stable_baselines3", "torch", "omnisafe", "rliable"]:
        try:
            out[module] = getattr(__import__(module), "__version__", "unknown")
        except Exception:
            out[module] = None
    return out


def run_metadata(*, seed: Any = None, started: Optional[float] = None, **extra: Any) -> dict[str, Any]:
    """Dictionary to embed in a results file."""
    meta: dict[str, Any] = {
        **git_commit(),
        "versions": versions(),
        "solvers": {"qp": "CLARABEL (via cvxpy)", "milp": "HiGHS (via scipy.optimize.milp)"},
        "seed": seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine()},
    }
    if started is not None:
        meta["wall_clock_seconds"] = time.time() - started
    meta.update(extra)
    return meta
