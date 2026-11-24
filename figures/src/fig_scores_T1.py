"""fig_scores_T1 - normalised score per episode on T1-Nominal, test split (horizontal box plots).

S = (J - J_L)/(U - J_L): 0 = DA-ONLY, 1 = ORACLE. S_D uses the ORACLE-DELIVER bound.
Boxes: quartiles; whiskers: 5th-95th percentile; dots: episodes outside. The axis
is cut on the left; the number of episodes left of the cut is printed there.
"""

import numpy as np
import pandas as pd

from common import COLORS, INK, LABELS, MUTED, ORDER, RESULTS, WIDTH_IN, plt, save, style

df = pd.read_csv(RESULTS / "benchmark_T1-Nominal.csv")
df = df[df.split == "test"]
names = [c for c in ORDER if c in set(df.controller) and c != "da-only"]
panels = [("score", "score_valid", "S (bound: ORACLE)")]
if "score_D" in df.columns and df.score_D.notna().any():
    panels.append(("score_D", "score_D_valid", "S$_D$ (bound: ORACLE-DELIVER)"))

fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_IN, 0.30 * len(names) + 0.9), sharey=True, squeeze=False)
y = np.arange(len(names))[::-1]
for ax, (col, valid, title) in zip(axes[0], panels):
    part = df[df[valid].astype(bool)]
    data = [part[part.controller == c][col].to_numpy() for c in names]
    lo5 = min(float(np.percentile(v, 5)) for v in data)
    hi = max(float(np.percentile(v, 99)) for v in data)
    floor = lo5 - 0.15 * (hi - lo5)
    right = hi + 0.08 * (hi - lo5)
    boxes = ax.boxplot(data, positions=y, vert=False, widths=0.6, patch_artist=True, whis=(5, 95),
                       flierprops=dict(marker=".", markersize=2, alpha=0.3, markeredgewidth=0),
                       medianprops=dict(color=INK, linewidth=1.1), boxprops=dict(linewidth=0.7),
                       whiskerprops=dict(linewidth=0.7, color=MUTED), capprops=dict(linewidth=0.7, color=MUTED))
    for patch, name in zip(boxes["boxes"], names):
        patch.set_facecolor(COLORS[name]); patch.set_edgecolor(INK); patch.set_alpha(0.85)
    for yi, values in zip(y, data):
        below = int((values < floor).sum())
        if below:
            ax.annotate(f"{below}←", (floor, yi), xytext=(2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7, color=INK)
    ax.axvline(0.0, color=INK, linewidth=0.8)
    ax.set_xlim(floor, right)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("score per episode")
    style(ax, "x")
axes[0][0].set_yticks(y, [LABELS[c] for c in names])
n_base = len(df[(df.controller == names[0])])
fig.text(0.0, -0.02, f"Numbers at the left edge: episodes beyond the axis cut. {n_base} episodes per baseline "
         f"(91 days x 10 activation seeds); RL x 5 training seeds.", fontsize=7, color=MUTED, ha="left", va="top")
fig.tight_layout()
save(fig, "fig_scores_T1")
