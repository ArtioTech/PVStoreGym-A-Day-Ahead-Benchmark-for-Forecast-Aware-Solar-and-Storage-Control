"""fig_decomposition_T1 - mean daily profit by settlement component, T1-Nominal, test split (horizontal stacked bars)."""

import numpy as np
import pandas as pd

from common import INK, LABELS, ORDER, RESULTS, WIDTH_IN, plt, save, style, summary

PARTS = [("dam_eur", "DAM", "#2a78d6", 1), ("id_eur", "Intraday", "#eb6834", 1), ("afrr_capacity_eur", "aFRR capacity", "#eda100", 1),
         ("afrr_activation_eur", "aFRR activation", "#e87ba4", 1), ("imbalance_eur", "Imbalance", "#1baf7a", 1),
         ("degradation_eur", "Degradation", "#6250d6", -1), ("penalties_eur", "Penalties", "#c0392b", -1)]
test = summary("T1-Nominal")["splits"]["test"]
names = [c for c in ORDER if c in test["controllers"]]
rows = [[s * test["controllers"][c]["components_eur_per_day"][k] for k, _, _, s in PARTS] for c in names]
labels = [LABELS[c] for c in names]
oracle = pd.read_csv(RESULTS / "oracle_T1-Nominal.csv")
oracle = oracle[oracle.split == "test"]
rows.append([s * oracle[f"oracle_{k}"].mean() for k, _, _, s in PARTS]); labels.append("ORACLE")
if "oracle_D_dam_eur" in oracle.columns and oracle.oracle_D_dam_eur.notna().any():
    rows.append([s * oracle[f"oracle_D_{k}"].mean() for k, _, _, s in PARTS]); labels.append("ORACLE-DELIVER")
table = np.array(rows)

fig, ax = plt.subplots(figsize=(WIDTH_IN, 0.30 * len(labels) + 1.2))
y = np.arange(len(labels))[::-1]
pos, neg = np.zeros(len(labels)), np.zeros(len(labels))
for k, (_, name, color, _) in enumerate(PARTS):
    v = table[:, k]
    ax.barh(y, v, 0.66, left=np.where(v >= 0, pos, neg), color=color, edgecolor="white", linewidth=0.6, label=name)
    pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
ax.scatter(table.sum(axis=1), y, marker="D", s=16, color=INK, zorder=3, label="Net profit")
ax.axvline(0.0, color=INK, linewidth=0.6)
ax.set_yticks(y, labels)
ax.set_xlabel("EUR per day (costs negative), mean over test episodes")
style(ax, "x")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), handletextpad=0.4, columnspacing=1.0)
fig.tight_layout()
save(fig, "fig_decomposition_T1")
