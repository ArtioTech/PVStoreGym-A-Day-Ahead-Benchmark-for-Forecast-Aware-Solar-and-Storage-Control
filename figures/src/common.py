"""Shared style for the paper figures: vector PDF, 13.5 cm text width, fonts >= 7 pt, colour-blind-safe.

Run any figure script from the repository root, e.g.

    .venv/bin/python figures/src/fig_scores_T1.py

The active protocol ($PVBESS_PROTOCOL, default configs/journal.yaml) selects the
results directory (results/<subdir>/) and the output directory (figures/<subdir>/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pv_bess_gym.benchmark.tasks import figures_root, results_root  # noqa: E402

RESULTS, FIGURES = results_root(), figures_root()
WIDTH_IN = 13.5 / 2.54                   # MDPI text width
MIN_PT = 7.0
TASKS = ["T1-Nominal", "T2-Forecast", "T3-Gate", "T4-Ageing", "T5-Combined"]
ORDER = ["da-only", "da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc", "da-opt-dmpc",
         "da-opt-ppo", "da-opt-sac", "da-opt-ppol"]
LABELS = {"da-only": "DA-ONLY", "da-opt-rbc": "DA-OPT-RBC", "da-opt-rt-mpc": "DA-OPT-RT-MPC", "da-opt-id-mpc": "DA-OPT-ID-MPC",
          "da-opt-mpc": "DA-OPT-MPC", "da-opt-dmpc": "DA-OPT-DMPC", "da-opt-ppo": "DA-OPT-PPO", "da-opt-sac": "DA-OPT-SAC",
          "da-opt-ppol": "DA-OPT-PPOL"}
# One colour and marker per controller in every figure; hues chosen to stay
# distinguishable under the common colour-vision deficiencies, markers repeat the identity.
COLORS = {"da-only": "#6b6a63", "da-opt-rbc": "#2a78d6", "da-opt-rt-mpc": "#eb6834", "da-opt-id-mpc": "#1baf7a",
          "da-opt-mpc": "#eda100", "da-opt-dmpc": "#8c564b", "da-opt-ppo": "#e87ba4", "da-opt-sac": "#008300",
          "da-opt-ppol": "#6250d6"}
MARKERS = {"da-only": "x", "da-opt-rbc": "o", "da-opt-rt-mpc": "s", "da-opt-id-mpc": "^", "da-opt-mpc": "D",
           "da-opt-dmpc": "h", "da-opt-ppo": "v", "da-opt-sac": "P", "da-opt-ppol": "*"}
INK, MUTED, GRID = "#1a1a19", "#5a5953", "#e4e2da"

plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans", "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax, grid_axis: str = "x") -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def summary(task: str) -> dict:
    return json.loads((RESULTS / f"summary_{task}.json").read_text())


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    w, h = fig.get_size_inches()
    print(f"-> {(FIGURES / name).relative_to(ROOT)}.pdf  ({w * 2.54:.1f} x {h * 2.54:.1f} cm before tight crop)")
