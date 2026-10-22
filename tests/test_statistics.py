"""
Unit tests for the statistical verification module.
Verifies paired Wilcoxon signed-rank test and non-parametric bootstrap confidence intervals.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.statistics import run_wilcoxon_test, bootstrap_ci, compute_qos

class TestStatistics(unittest.TestCase):

    def test_wilcoxon_paired_test(self):
        """Verify Wilcoxon test rejects null hypothesis when systematic savings exist."""
        np.random.seed(42)
        n_samples = 56
        # Baseline series with mean 18,000 kWh/day
        baseline = np.random.normal(18000, 500, n_samples)
        # Predictive series systematically ~14% lower
        predictive = baseline * 0.86 + np.random.normal(0, 50, n_samples)

        res = run_wilcoxon_test(baseline, predictive)
        self.assertIn("p_value_one_sided", res)
        self.assertIn("reject_H0", res)
        self.assertTrue(res["reject_H0"], "Null hypothesis must be rejected for ~14% systematic reduction")
        self.assertLess(res["p_value_one_sided"], 0.001, "p-value must be highly significant (p < 0.001)")

    def test_bootstrap_ci_integrity(self):
        """Verify bootstrap confidence interval bounds satisfy ordering: lower <= mean <= upper."""
        np.random.seed(42)
        sample_savings = np.random.normal(14.2, 0.5, 56)
        ci_lo, ci_hi, ci_mean = bootstrap_ci(sample_savings, n_iter=1000)

        self.assertLessEqual(ci_lo, ci_mean, "Lower CI bound must be <= sample mean")
        self.assertGreaterEqual(ci_hi, ci_mean, "Upper CI bound must be >= sample mean")
        self.assertGreater(ci_hi, ci_lo, "Upper CI bound must be strictly greater than lower bound")

    def test_qos_calculation(self):
        """Verify Thermal Comfort QoS percentage computation."""
        # 4 steps in a single zone within comfort zone [22.0, 24.0]
        T_matrix = np.array([
            [22.5], [23.0], [23.5], [24.5] # 3 in comfort, 1 violation
        ])
        mock_df = pd.DataFrame({
            "is_open": [True, True, True, True]
        })
        qos = compute_qos(T_matrix, mock_df)
        self.assertEqual(qos["n_open_steps"], 4)
        self.assertEqual(qos["n_violation_steps"], 1)
        self.assertAlmostEqual(qos["qos_percent"], 75.0, places=1)
        self.assertAlmostEqual(qos["violation_hours"], 0.2, places=1)

if __name__ == "__main__":
    unittest.main()
