"""
modules/optimization.py — Multi-Objective Evolutionary Optimization Engine.
Implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II; Deb et al., 2002)
and the Decomposition-based Evolutionary Algorithm (MOEA/D; Zhang & Li, 2007)
to solve the Pareto trade-off between facility energy conservation and thermal comfort.
"""
from typing import Dict, Any
import numpy as np
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from config import (
    N_ZONES,
    FORECAST_HORIZON,
    T_SET_MIN,
    T_SET_MAX,
    STEP_MIN,
    LIGHTING_W_M2,
    ZONE_AREA,
    T_COMFORT_LOW,
    T_COMFORT_HIGH,
    OPTIM_POP_SIZE,
    OPTIM_N_GEN,
    OPTIM_CROSSOVER_PROB,
    OPTIM_CROSSOVER_ETA,
    OPTIM_MUTATION_PROB,
    OPTIM_MUTATION_ETA,
    MOEAD_NEIGHBORHOOD,
    SEED,
)
from modules.thermal_model import step_thermal_simulation, compute_hvac_power_kw


class HVACOptimizationProblem(Problem):
    """
    Formulation of the Bi-Objective Commercial Facility Optimization Problem:
    Minimize f_1: Total Electrical Energy Consumption [kWh] across the rolling lookahead horizon.
    Minimize f_2: Cumulative Occupant Thermal Discomfort Penalty [°C·h].
    """
    def __init__(self, T_in_init: np.ndarray,
                 Q_gains_horizon: np.ndarray,
                 t_amb_horizon: np.ndarray) -> None:
        self.T_in_init = T_in_init.copy()
        self.Q_gains_horizon = Q_gains_horizon
        self.t_amb_horizon = t_amb_horizon
        
        n_var = N_ZONES * FORECAST_HORIZON
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.full(n_var, T_SET_MIN),
            xu=np.full(n_var, T_SET_MAX)
        )

    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 2))

        for i, x in enumerate(X):
            T_set_matrix = x.reshape((FORECAST_HORIZON, N_ZONES))
            T_in = self.T_in_init.copy()
            f1, f2 = 0.0, 0.0

            for h in range(FORECAST_HORIZON):
                Q_gains = self.Q_gains_horizon[h]
                t_amb = self.t_amb_horizon[h]
                T_set = T_set_matrix[h]

                Q_cooling, P_elec = compute_hvac_power_kw(T_in, T_set, t_amb)
                P_lighting = (LIGHTING_W_M2 * ZONE_AREA.sum() / 1000.0)

                f1 += (P_elec.sum() + P_lighting) * (STEP_MIN / 60.0)
                T_in = step_thermal_simulation(T_in, Q_cooling, Q_gains, t_amb)

                discomfort = np.sum(
                    np.maximum(0.0, T_COMFORT_LOW - T_in) +
                    np.maximum(0.0, T_in - T_COMFORT_HIGH)
                )
                f2 += discomfort * (STEP_MIN / 60.0)

            F[i] = [f1, f2]

        out["F"] = F


def knee_point_selection(pareto_F: np.ndarray) -> int:
    """
    Selects the optimal balanced compromise setpoint vector (Knee Point K*)
    on the non-dominated Pareto frontier by minimizing Euclidean distance
    to the normalized utopia point.
    
    Args:
        pareto_F: Matrix of Pareto objective vectors of shape (N, 2).
        
    Returns:
        int: Index of the selected Knee Point solution.
    """
    f_min = pareto_F.min(axis=0)
    f_max = pareto_F.max(axis=0)
    f_norm = (pareto_F - f_min) / (f_max - f_min + 1e-9)
    dists = np.linalg.norm(f_norm, axis=1)
    return int(np.argmin(dists))


def run_optimization_step(T_in: np.ndarray,
                          Q_gains_horizon: np.ndarray,
                          t_amb_horizon: np.ndarray) -> Dict[str, Any]:
    """
    Executes evolutionary optimization over the forward prediction lookahead horizon.
    Computes Pareto frontiers using both NSGA-II and MOEA/D and extracts the Knee Point setpoint.
    
    Args:
        T_in: Current indoor zone temperature vector [°C].
        Q_gains_horizon: Forecasted zone heat gains [kW] across horizon H.
        t_amb_horizon: Forecasted ambient temperatures [°C] across horizon H.
        
    Returns:
        Dict[str, Any]: Optimized setpoint trajectory, Pareto fronts, and execution timings.
    """
    problem = HVACOptimizationProblem(T_in, Q_gains_horizon, t_amb_horizon)

    # 1. NSGA-II Execution
    algo_nsga2 = NSGA2(
        pop_size=OPTIM_POP_SIZE,
        crossover=SBX(prob=OPTIM_CROSSOVER_PROB, eta=OPTIM_CROSSOVER_ETA),
        mutation=PM(prob=OPTIM_MUTATION_PROB, eta=OPTIM_MUTATION_ETA),
        eliminate_duplicates=True
    )
    t0 = time.perf_counter()
    res_nsga2 = minimize(problem, algo_nsga2, ("n_gen", OPTIM_N_GEN), verbose=False, seed=SEED)
    t_nsga2 = time.perf_counter() - t0

    # 2. MOEA/D Execution
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=OPTIM_POP_SIZE - 1)
    algo_moead = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=MOEAD_NEIGHBORHOOD,
        crossover=SBX(prob=OPTIM_CROSSOVER_PROB, eta=OPTIM_CROSSOVER_ETA),
        mutation=PM(prob=OPTIM_MUTATION_PROB, eta=OPTIM_MUTATION_ETA),
    )
    t0 = time.perf_counter()
    res_moead = minimize(problem, algo_moead, ("n_gen", OPTIM_N_GEN), verbose=False, seed=SEED)
    t_moead = time.perf_counter() - t0

    pareto_F = res_nsga2.F
    pareto_X = res_nsga2.X
    best_idx = knee_point_selection(pareto_F)
    best_x = pareto_X[best_idx]

    return {
        "best_setpoints": best_x.reshape((FORECAST_HORIZON, N_ZONES)),
        "nsga2_pareto_F": pareto_F,
        "moead_pareto_F": res_moead.F if res_moead.F is not None else pareto_F,
        "t_nsga2_s": round(t_nsga2, 3),
        "t_moead_s": round(t_moead, 3),
    }
