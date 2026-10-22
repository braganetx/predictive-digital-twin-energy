# modules/optimization.py
import numpy as np
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from config import *
from modules.thermal_model import step_thermal_simulation, compute_hvac_power_kw

class HVACOptimizationProblem(Problem):
    """
    Problema de otimizacao multiobjetivo:
    f1 = consumo eletrico total [kWh] no horizonte
    f2 = penalidade de desconforto termico [°C·h]
    """
    def __init__(self, T_in_init, Q_gains_horizon, t_amb_horizon):
        self.T_in_init       = T_in_init.copy()
        self.Q_gains_horizon = Q_gains_horizon
        self.t_amb_horizon   = t_amb_horizon
        
        n_var = N_ZONES * FORECAST_HORIZON
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.full(n_var, T_SET_MIN),
            xu=np.full(n_var, T_SET_MAX)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 2))

        for i, x in enumerate(X):
            T_set_matrix = x.reshape((FORECAST_HORIZON, N_ZONES))
            T_in = self.T_in_init.copy()
            f1, f2 = 0.0, 0.0

            for h in range(FORECAST_HORIZON):
                Q_gains = self.Q_gains_horizon[h]
                t_amb   = self.t_amb_horizon[h]
                T_set   = T_set_matrix[h]

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
    f_min = pareto_F.min(axis=0)
    f_max = pareto_F.max(axis=0)
    f_norm = (pareto_F - f_min) / (f_max - f_min + 1e-9)
    dists = np.linalg.norm(f_norm, axis=1)
    return int(np.argmin(dists))


def run_optimization_step(T_in, Q_gains_horizon, t_amb_horizon):
    problem = HVACOptimizationProblem(T_in, Q_gains_horizon, t_amb_horizon)

    # 1. NSGA-II
    algo_nsga2 = NSGA2(
        pop_size=OPTIM_POP_SIZE,
        crossover=SBX(prob=OPTIM_CROSSOVER_PROB, eta=OPTIM_CROSSOVER_ETA),
        mutation=PM(prob=OPTIM_MUTATION_PROB, eta=OPTIM_MUTATION_ETA),
        eliminate_duplicates=True
    )
    t0 = time.perf_counter()
    res_nsga2 = minimize(problem, algo_nsga2, ("n_gen", OPTIM_N_GEN), verbose=False, seed=SEED)
    t_nsga2 = time.perf_counter() - t0

    # 2. MOEA/D
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
    best_x   = pareto_X[best_idx]

    return {
        "best_setpoints": best_x.reshape((FORECAST_HORIZON, N_ZONES)),
        "nsga2_pareto_F": pareto_F,
        "moead_pareto_F": res_moead.F if res_moead.F is not None else pareto_F,
        "t_nsga2_s": round(t_nsga2, 3),
        "t_moead_s": round(t_moead, 3),
    }
