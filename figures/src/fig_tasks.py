"""fig_tasks - score per controller and task, test split: IQM with 95% CI (S filled, S_D open)."""

import numpy as np

from common import COLORS, INK, LABELS, MARKERS, MUTED, ORDER, RESULTS, TASKS, WIDTH_IN, plt, save, style, summary

available = [t for t in TASKS if (RESULTS / f"summary_{t}.json").exists()]
names = [c for c in ORDER if c != "da-only" and any(c in summary(t)["splits"]["test"]["controllers"] for t in available)]
fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
width = 0.8 / max(1, len(names))
has_sd = False
for j, name in enumerate(names):
    off = (j - (len(names) - 1) / 2) * width
    for i, task in enumerate(TASKS):
        if task not in available:
            continue
        entry = summary(task)["splits"]["test"]["controllers"].get(name)
        if entry is None:
            continue
        q = entry["score"]["iqm"]
        ax.errorbar(i + off, q["value"], yerr=[[q["value"] - q["ci95"][0]], [q["ci95"][1] - q["value"]]],
                    fmt=MARKERS[name], color=COLORS[name], markersize=4, elinewidth=0.9, capsize=1.5,
                    label=LABELS[name] if i == 0 or task == available[0] else None)
        if "score_D" in entry:
            d = entry["score_D"]["iqm"]
            ax.errorbar(i + off, d["value"], yerr=[[d["value"] - d["ci95"][0]], [d["ci95"][1] - d["value"]]],
                        fmt=MARKERS[name], mfc="white", mec=COLORS[name], color=COLORS[name], markersize=4,
                        elinewidth=0.6, capsize=1.0, alpha=0.9)
            has_sd = True
ax.axhline(0.0, color=INK, linewidth=0.8)
ax.set_xticks(range(len(TASKS)), [t.replace("-", "\n") for t in TASKS])
ax.set_xlim(-0.55, len(TASKS) - 0.45)
ax.set_ylabel("IQM with 95% CI\n(0 = DA-ONLY, 1 = bound)")
style(ax, "y")
handles, labels = ax.get_legend_handles_labels()
seen = dict(zip(labels, handles))
ax.legend(seen.values(), seen.keys(), ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
          handletextpad=0.3, columnspacing=0.9)
if has_sd:
    ax.set_title("filled: S (bound ORACLE); open: S$_D$ (bound ORACLE-DELIVER)", loc="left", color=MUTED)
fig.tight_layout()
save(fig, "fig_tasks")
