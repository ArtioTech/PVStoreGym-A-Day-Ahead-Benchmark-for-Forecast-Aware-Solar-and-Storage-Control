from __future__ import annotations

import os
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_episode(log: List[Dict[str, Any]], title: str = "PV+BESS Episode", out_dir: str = "outputs"):
    if not log:
        raise ValueError("Empty log.")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(log)

    plt.figure()
    plt.plot(df["t"], df["pv_kw"], label="PV AC (kW)")
    plt.plot(df["t"], df["p_batt_real_kw"], label="Battery P (kW)")
    plt.plot(df["t"], df["export_kw"], label="Export (kW)")
    plt.plot(df["t"], df["import_kw"], label="Import (kW)")
    plt.xlabel("Step")
    plt.ylabel("kW")
    plt.title(title + " - Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "power.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["t"], df["soc"], label="SOC")
    plt.xlabel("Step")
    plt.ylabel("SOC")
    plt.title(title + " - State of charge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "soc.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["t"], df["reward"], label="Reward (€/step)")
    plt.plot(df["t"], df["cum_profit"], label="Cumulative profit (€)")
    plt.xlabel("Step")
    plt.ylabel("EUR")
    plt.title(title + " - Profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "profit.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["t"], df["dam_eur"], label="DAM")
    plt.plot(df["t"], df["id_eur"], label="IDM")
    plt.plot(df["t"], df["imbalance_eur"], label="Imbalance")
    plt.plot(df["t"], df["afrr_capacity_eur"], label="aFRR capacity")
    plt.plot(df["t"], df["afrr_activation_eur"], label="aFRR activation")
    plt.xlabel("Step")
    plt.ylabel("EUR/step")
    plt.title(title + " - Revenue stack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "revenue_stack.png"), dpi=150)
    plt.close()

    print(f"Saved plots in '{out_dir}/'")
