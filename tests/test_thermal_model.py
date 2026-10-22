"""
Unit tests for the thermodynamic simulation module (Clarke, 2001).
Tests energy conservation, chiller coefficient of performance (COP),
and multi-zone thermal differential equation behavior.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import N_ZONES, COP_CHILLER_REF
from modules.thermal_model import (
    get_chiller_cop,
    compute_zone_heat_gains,
    step_thermal_simulation,
    compute_hvac_power_kw,
    build_inter_zone_matrix
)

class TestThermalModel(unittest.TestCase):

    def test_chiller_cop_bounds(self):
        """Verify chiller COP conforms to physical thermodynamic limits (lift)."""
        cop_cool = get_chiller_cop(18.0)
        cop_ref  = get_chiller_cop(25.0)
        cop_hot  = get_chiller_cop(38.0)

        self.assertGreaterEqual(cop_cool, cop_ref, "Cooler ambient should yield equal or higher COP")
        self.assertGreaterEqual(cop_ref, cop_hot, "Reference ambient should yield equal or higher COP than hot ambient")
        self.assertGreaterEqual(cop_hot, 2.8, "COP must not drop below minimum physical cutoff (2.8)")
        self.assertLessEqual(cop_cool, 4.8, "COP must not exceed maximum physical chiller rating (4.8)")

    def test_zone_heat_gains(self):
        """Verify multi-zone heat gain synthesis from solar, occupancy, and equipment."""
        mock_step = {
            "solar_irr": 800.0,
            "occ_z1": 500,
            "occ_z2": 600,
            "occ_z3": 400,
            "occ_z4": 300,
            "is_open": 1.0
        }
        gains = compute_zone_heat_gains(mock_step)
        self.assertEqual(len(gains), N_ZONES)
        self.assertTrue(np.all(gains > 0), "All operational zones must have positive sensible heat gains")
        # Central Atrium (z4) has fenestration gains
        self.assertGreater(gains[3], 0.0)

    def test_thermal_differential_step(self):
        """Verify thermal step integration moves temperatures in expected physical direction."""
        T_init = np.full(N_ZONES, 23.0)
        Q_gains = np.full(N_ZONES, 200.0)
        t_amb = 32.0

        # Case A: No cooling -> temperatures must rise
        Q_cooling_zero = np.zeros(N_ZONES)
        T_after_no_cooling = step_thermal_simulation(T_init, Q_cooling_zero, Q_gains, t_amb)
        self.assertTrue(np.all(T_after_no_cooling > T_init), "Temperatures must increase with positive gains and no cooling")

        # Case B: Maximum cooling -> temperatures must drop relative to uncooled state
        Q_cooling_max = np.full(N_ZONES, 800.0)
        T_after_cooling = step_thermal_simulation(T_init, Q_cooling_max, Q_gains, t_amb)
        self.assertTrue(np.all(T_after_cooling < T_after_no_cooling), "Active cooling must reduce zone temperature")

    def test_inter_zone_coupling_symmetry(self):
        """Verify conductance matrix satisfies conservation property sum_j A_ij = 0 (Clarke, 2001)."""
        A = build_inter_zone_matrix()
        self.assertEqual(A.shape, (N_ZONES, N_ZONES))
        # Total off-diagonal heat exchange capacity must be non-zero
        self.assertGreater(np.sum(np.abs(A)), 0.0)
        # Net sum of each row must equal zero (heat entering one zone leaves another)
        row_sums = np.sum(A, axis=1)
        np.testing.assert_allclose(row_sums, np.zeros(N_ZONES), atol=1e-10)

if __name__ == "__main__":
    unittest.main()
