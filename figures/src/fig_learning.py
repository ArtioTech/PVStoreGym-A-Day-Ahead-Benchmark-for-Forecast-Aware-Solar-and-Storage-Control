"""fig_learning - RL checkpoints on T1-Nominal: validation score vs steps; PPO-Lagrangian cost with budget.

Mean over training seeds with 95% CI (t-free normal approximation, 1.96 s / sqrt(n)).
Linear axes; points of a mean curve below the lower axis limit are drawn on the
limit as downward triangles, and a note under the figure gives their count and the lowest mean.
"""

import numpy as np
import pandas as pd

from common import COLORS, INK, LABELS, MARKERS, MUTED, RESULTS, WIDTH_IN, plt, save, style

curves = pd.read_csv(RESULTS / "rl_training" / "curves.csv")
curves = curves[curves.task == "T1-Nominal"]
hyper = pd.read_json(RESULTS / "rl_training" / "hyperparameters.json", typ="series")
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.4))

stats = {}
notes = []
for name, group in curves.groupby("controller"):
    wide = group.pivot(index="steps", columns="rl_seed", values="validation_score")
    stats[name] = (wide.index / 1000, wide.mean(axis=1), 1.96 * wide.std(axis=1, ddof=1) / np.sqrt(wide.shape[1]))
values = np.concatenate([m.to_numpy() for _, m, _ in stats.values()])
q1, q3 = np.percentile(values, [25, 75])
floor = min(q1 - 3.0 * (q3 - q1), values.max() - 1e-9)        # robust lower limit
floor = min(floor, float(np.min([m.min() for _, m, _ in stats.values()]))) if (q3 - q1) == 0 else floor
top = max(0.02, values.max() + 0.1 * (values.max() - floor))
for name, (x, m, h) in stats.items():
    a.plot(x, m.clip(lower=floor), color=COLORS[name], marker=MARKERS[name], markersize=3, linewidth=1.3, label=LABELS[name])
    a.fill_between(x, (m - h).clip(lower=floor), (m + h).clip(lower=floor), color=COLORS[name], alpha=0.18, linewidth=0)
    off = m < floor
    if off.any():
        a.scatter(x[off], np.full(off.sum(), floor), marker="v", color=COLORS[name], s=14, zorder=3, clip_on=False)
        notes.append(f"{LABELS[name]}: {int(off.sum())} checkpoints below the axis (lowest mean {m.min():.1f})")
a.axhline(0.0, color=INK, linewidth=0.6)
a.set_ylim(floor, top)
a.set_xlabel("environment steps [thousand]")
a.set_ylabel("validation score S")
handles, labels = a.get_legend_handles_labels()
style(a, "y")

ppol = curves[curves.controller == "da-opt-ppol"].pivot(index="steps", columns="rl_seed", values="validation_cost_kw")
if not ppol.empty:
    m, h = ppol.mean(axis=1), 1.96 * ppol.std(axis=1, ddof=1) / np.sqrt(ppol.shape[1])
    b.plot(ppol.index / 1000, m, color=COLORS["da-opt-ppol"], marker=MARKERS["da-opt-ppol"], markersize=4, linewidth=1.3)
    b.fill_between(ppol.index / 1000, m - h, m + h, color=COLORS["da-opt-ppol"], alpha=0.18, linewidth=0)
    budget = hyper["controllers"]["da-opt-ppol"]["cost_budget_kw_per_episode"]["T1-Nominal"]
    b.axhline(budget, color=INK, linewidth=0.8, linestyle="--")
    b.annotate(f"budget {budget:.0f}", (ppol.index.max() / 1000, budget), ha="right", va="bottom", fontsize=7, color=MUTED)
    b.set_ylim(0, max(float((m + h).max()), budget) * 1.12)
b.set_xlabel("environment steps [thousand]")
b.set_ylabel("PPOL validation cost\n[kW per episode]")
style(b, "y")
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3)
if notes:
    fig.text(0.01, 0.0, "Triangles on the lower edge: " + "; ".join(notes) + ".", fontsize=7, color=MUTED, ha="left", va="top")
save(fig, "fig_learning")
