"""
modules/statistics.py — Non-Parametric Hypothesis Testing & Quality-of-Service (QoS) Validation.
Implements the Wilcoxon signed-rank test and 10,000-sample bootstrap confidence intervals
to validate statistical significance without assuming normal distribution.
"""
from typing import Dict, Any, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from config import ALPHA, N_BOOTSTRAP, T_COMFORT_LOW, T_COMFORT_HIGH


def run_wilcoxon_test(daily_baseline: np.ndarray,
                      daily_predictive: np.ndarray,
                      alpha: float = ALPHA) -> Dict[str, Any]:
    """
    Executes a one-sided paired Wilcoxon signed-rank test to determine whether
    daily predictive facility consumption is systematically lower than conventional baseline.
    
    H0: Median paired difference (Baseline - Predictive) <= 0
    H1: Median paired difference (Baseline - Predictive) > 0
    
    Args:
        daily_baseline: Vector of daily baseline electricity consumption [kWh].
        daily_predictive: Vector of daily predictive electricity consumption [kWh].
        alpha: Statistical significance threshold (default: 0.05).
        
    Returns:
        Dict[str, Any]: Test statistics, p-values, and formal hypothesis decision.
    """
    diff = daily_baseline - daily_predictive
    res = wilcoxon(diff, alternative="greater")
    return {
        "W_statistic": float(res.statistic),
        "p_value_one_sided": float(res.pvalue),
        "reject_H0": bool(res.pvalue < alpha),
        "alpha": alpha,
        "n_samples": len(diff),
    }


def bootstrap_ci(data: np.ndarray,
                 stat_fn: Callable = np.mean,
                 n_iter: int = N_BOOTSTRAP,
                 alpha: float = ALPHA,
                 seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes non-parametric percentile bootstrap confidence intervals at (1 - alpha) level.
    
    Args:
        data: Array of sample observations.
        stat_fn: Statistical function to evaluate (default: np.mean).
        n_iter: Number of bootstrap iterations (default: 10,000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        Tuple[float, float, float]: (lower_bound, upper_bound, sample_mean).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    stats_boot = np.array([
        stat_fn(rng.choice(data, size=n, replace=True))
        for _ in range(n_iter)
    ])
    lower = float(np.percentile(stats_boot, 100 * alpha / 2))
    upper = float(np.percentile(stats_boot, 100 * (1 - alpha / 2)))
    return lower, upper, float(stat_fn(data))


def compute_qos(T_history: np.ndarray, dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the Thermal Comfort Quality-of-Service (QoS) satisfaction metric
    during commercial operating hours across all 4 building zones.
    
    QoS = percentage of operating 15-minute intervals where all zones
    remain strictly within the ASHRAE Standard 55 comfort band [22.0°C, 24.0°C].
    
    Args:
        T_history: Simulated temperature trajectories of shape (N_STEPS, N_ZONES).
        dataset: Telemetry DataFrame containing commercial schedule masks.
        
    Returns:
        Dict[str, Any]: QoS percentage and thermal discomfort violation hours.
    """
    operating_mask = dataset["is_open"].values.astype(bool)
    T_op = T_history[operating_mask]

    in_comfort = np.all((T_op >= T_COMFORT_LOW) & (T_op <= T_COMFORT_HIGH), axis=1)
    qos = float(in_comfort.mean()) * 100.0
    n_violations = int((~in_comfort).sum())

    return {
        "qos_percent": round(qos, 2),
        "n_open_steps": int(operating_mask.sum()),
        "n_violation_steps": n_violations,
        "violation_hours": round(n_violations * 15.0 / 60.0, 1),
    }
