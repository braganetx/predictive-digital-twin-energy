"""
Unit tests for multi-objective evolutionary optimization module (NSGA-II & MOEA/D).
Verifies objective evaluation, Pareto optimality criteria, and knee point selection.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import N_ZONES, FORECAST_HORIZON
from modules.optimization import HVACOptimizationProblem, knee_point_selection

class TestOptimization(unittest.TestCase):

    def test_knee_point_selection(self):
        """Verify knee point selection identifies the balanced compromise on a Pareto curve."""
        # Simulated convex Pareto front: f1 (energy) vs f2 (discomfort)
        pareto_F = np.array([
            [100.0, 10.0],  # Low energy, high discomfort
            [150.0, 3.0],   # Balanced compromise (knee point)
            [220.0, 0.5],   # High energy, near-zero discomfort
        ])
        knee_idx = knee_point_selection(pareto_F)
        self.assertEqual(knee_idx, 1, "Should select the middle point minimizing distance to normalized ideal vector")

    def test_problem_evaluation_dimensions_and_non_negativity(self):
        """Verify the pymoo optimization problem generates correct dimensions and non-negative costs."""
        T_init = np.full(N_ZONES, 23.0)
        Q_gains_horizon = np.full((FORECAST_HORIZON, N_ZONES), 150.0)
        t_amb_horizon = np.full(FORECAST_HORIZON, 30.0)

        problem = HVACOptimizationProblem(T_init, Q_gains_horizon, t_amb_horizon)
        
        # Test evaluation with a sample population of 4 candidate setpoint schedules
        pop_size = 4
        n_vars = N_ZONES * FORECAST_HORIZON
        X_sample = np.random.uniform(21.0, 25.0, size=(pop_size, n_vars))
        
        out = {}
        problem._evaluate(X_sample, out)

        self.assertIn("F", out)
        F = out["F"]
        self.assertEqual(F.shape, (pop_size, 2))
        self.assertTrue(np.all(F[:, 0] >= 0.0), "Energy consumption (f1) must be strictly non-negative")
        self.assertTrue(np.all(F[:, 1] >= 0.0), "Discomfort penalty (f2) must be strictly non-negative")

if __name__ == "__main__":
    unittest.main()
