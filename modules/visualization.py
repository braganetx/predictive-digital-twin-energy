"""
modules/visualization.py — Publication-Grade Figure Generation Routines.
Renders 300-DPI publication figures adhering to IEEE/Elsevier formatting guidelines.
"""
from typing import Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR: str = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_figure1_energy_comparison(daily_reactive: np.ndarray,
                                   daily_predictive: np.ndarray,
                                   eta_energy: float) -> str:
    """
    Renders Figure 1: Daily Electrical Energy Consumption Comparison (56-Day Evaluation).
    Includes cumulative savings fill and daily difference bar chart.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    days = np.arange(1, len(daily_reactive) + 1)

    ax1.fill_between(days, daily_reactive, alpha=0.35, color="#E74C3C", label="Reactive Baseline (kWh)")
    ax1.fill_between(days, daily_predictive, alpha=0.55, color="#2ECC71", label="Predictive Digital Twin (kWh)")
    ax1.plot(days, daily_reactive, color="#C0392B", linewidth=1.3)
    ax1.plot(days, daily_predictive, color="#27AE60", linewidth=1.3)
    ax1.set_ylabel("Daily Electricity (kWh)")
    ax1.set_title(
        f"Figure 1: Daily Electrical Energy Consumption (56-Day Evaluation)\n"
        f"Reactive Baseline vs. Predictive Digital Twin (Gross Reduction: {eta_energy:.2f}%)"
    )
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.25)

    diff = daily_reactive - daily_predictive
    ax2.bar(days, diff, color=np.where(diff > 0, "#2ECC71", "#E74C3C"), alpha=0.8, width=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Simulation Day (Day 1 to 56)")
    ax2.set_ylabel("Daily Net Savings (kWh)")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure1_energy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_figure2_zone_temperatures(T_reactive: np.ndarray,
                                   T_predictive: np.ndarray,
                                   dataset: pd.DataFrame,
                                   zone_idx: int = 2,
                                   day_start: int = 14,
                                   n_days: int = 3) -> str:
    """
    Renders Figure 2: Indoor Thermal Trajectory in Specific Thermal Zone (Food Court).
    Demonstrates proactive pre-cooling and avoidance of reactive overshooting.
    """
    from config import T_COMFORT_LOW, T_COMFORT_HIGH, STEPS_PER_DAY
    s0 = day_start * STEPS_PER_DAY
    s1 = s0 + n_days * STEPS_PER_DAY
    hours = np.arange(n_days * STEPS_PER_DAY) * 15.0 / 60.0

    fig, ax = plt.subplots(figsize=(11, 4.8))

    ax.plot(hours, T_reactive[s0:s1, zone_idx], color="#E74C3C", linewidth=1.5, label="Reactive Baseline", alpha=0.85)
    ax.plot(hours, T_predictive[s0:s1, zone_idx], color="#27AE60", linewidth=1.8, label="Predictive Digital Twin (Pre-cooling)", alpha=0.95)
    ax.axhline(T_COMFORT_HIGH, color="#D35400", linestyle="--", linewidth=1.2, label=f"Comfort Upper Bound ({T_COMFORT_HIGH}°C)")
    ax.axhline(T_COMFORT_LOW, color="#2980B9", linestyle="--", linewidth=1.2, label=f"Comfort Lower Bound ({T_COMFORT_LOW}°C)")
    ax.fill_between(hours, T_COMFORT_LOW, T_COMFORT_HIGH, alpha=0.08, color="#2ECC71", label="ASHRAE/ISO Comfort Range")

    zone_names = ["Anchor Retail", "Specialty Boutiques", "Food Court", "Central Atrium"]
    z_name = zone_names[zone_idx] if zone_idx < len(zone_names) else f"Zone {zone_idx+1}"

    ax.set_xlabel("Time (Hours across Representative High-Occupancy Weekend)")
    ax.set_ylabel("Indoor Zone Air Temperature (°C)")
    ax.set_title(
        f"Figure 2: Indoor Thermal Trajectory in {z_name} During Occupancy Surge\n"
        f"(Reactive Lag & Thermal Overshooting vs. Predictive Proactive Pre-Cooling)"
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.set_ylim(20.5, 26.5)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_zone_temperatures.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_figure3_pareto_front(nsga2_F: np.ndarray,
                              moead_F: np.ndarray,
                              knee_F: np.ndarray) -> str:
    """
    Renders Figure 3: Non-Dominated Pareto Frontiers Derived by NSGA-II and MOEA/D.
    Highlights the autonomic knee point dispatch compromise setpoint.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    idx_n = np.argsort(nsga2_F[:, 0])
    idx_m = np.argsort(moead_F[:, 0])

    ax.scatter(nsga2_F[:, 0], nsga2_F[:, 1], s=35, alpha=0.75, color="#2980B9", label="NSGA-II Non-dominated Solutions", zorder=3)
    ax.plot(nsga2_F[idx_n, 0], nsga2_F[idx_n, 1], color="#2980B9", linewidth=1.2, alpha=0.5, zorder=2)

    ax.scatter(moead_F[:, 0], moead_F[:, 1], s=35, alpha=0.75, color="#E74C3C", marker="^", label="MOEA/D Non-dominated Solutions", zorder=3)
    ax.plot(moead_F[idx_m, 0], moead_F[idx_m, 1], color="#E74C3C", linewidth=1.2, alpha=0.5, linestyle="--", zorder=2)

    ax.scatter(knee_F[0], knee_F[1], s=220, color="#F39C12", marker="*", zorder=5, label="Autonomic Dispatch Setpoint (Knee Point K*)", edgecolors="black", linewidths=0.9)

    ax.set_xlabel("Objective 1: Building Electrical Energy Consumption — $f_1$ (kWh / 4-h Horizon)")
    ax.set_ylabel("Objective 2: Thermal Discomfort Violation Penalty — $f_2$ (°C·h)")
    ax.set_title(
        "Figure 3: Pareto-Optimal Frontiers Derived by Evolutionary Optimizers\n"
        "(Trade-off between Facility Energy Conservation and Occupant Thermal Quality)"
    )
    ax.legend(fontsize=9.5, frameon=True)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure3_pareto_front.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path
