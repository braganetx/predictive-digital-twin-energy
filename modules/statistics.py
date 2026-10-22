# modules/statistics.py
import numpy as np
from scipy import stats
from config import T_COMFORT_LOW, T_COMFORT_HIGH

def run_wilcoxon_test(daily_reactive: np.ndarray, daily_predictive: np.ndarray) -> dict:
    """
    Teste de Wilcoxon pareado de postos sinalizados (N=56 dias).
    Verifica se a reducao de consumo do preditivo e estatisticamente significativa.
    """
    differences = daily_reactive - daily_predictive
    stat_two, p_two = stats.wilcoxon(differences, alternative="two-sided")
    stat_one, p_one = stats.wilcoxon(differences, alternative="greater")

    return {
        "W_statistic": round(float(stat_two), 2),
        "p_value_two_sided": round(float(p_two), 6),
        "p_value_one_sided": round(float(p_one), 8),
        "n_pairs": len(differences),
        "reject_H0": bool(p_one < 0.05),
        "mean_diff_kwh": round(float(np.mean(differences)), 2),
        "median_diff_kwh": round(float(np.median(differences)), 2),
    }


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_iter: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    stats_boot = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_iter)
    ])
    lower = float(np.percentile(stats_boot, 100 * alpha / 2))
    upper = float(np.percentile(stats_boot, 100 * (1 - alpha / 2)))
    return lower, upper, float(stat_fn(data))


def compute_qos(T_history: np.ndarray, dataset) -> dict:
    """
    Calcula QoS de conforto termico durante as horas de funcionamento.
    QoS = % de passos operacionais dentro de [22°C, 24°C] em todas as 4 zonas.
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
