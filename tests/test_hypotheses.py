"""
Hypothesis verification test suite.
Loads the consolidated experimental results (outputs/paper_results.json)
and formally verifies the three core scientific hypotheses (H1, H2, H3).
"""
import os
import json
import unittest

class TestHypothesesVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        results_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "paper_results.json")
        cls.has_results = os.path.exists(results_file)
        if cls.has_results:
            with open(results_file, "r") as f:
                cls.results = json.load(f)

    def test_h1_energy_savings_greater_than_10_percent(self):
        """Verify Hypothesis H1: Predictive Digital Twin achieves >= 10% net energy savings with p < 0.05."""
        if not self.has_results:
            self.skipTest("outputs/paper_results.json not found. Run main.py first.")
        
        h1 = self.results["section_5_2_H1"]
        eta = h1["eta_energy_percent"]
        p_val = h1["wilcoxon_p_one_sided"]
        ci_lo = h1["bootstrap_ci_lower_pct"]

        self.assertGreaterEqual(eta, 10.0, f"Facility energy savings ({eta}%) must be >= 10.0%")
        self.assertLess(p_val, 0.05, f"Wilcoxon p-value ({p_val}) must be < 0.05")
        self.assertGreaterEqual(ci_lo, 10.0, f"Lower 95% Bootstrap CI ({ci_lo}%) must be >= 10.0%")
        self.assertTrue(h1["H1_accepted"], "H1 must be formally marked as accepted")

    def test_h2_computational_overhead_below_20_percent(self):
        """Verify Hypothesis H2: Computational Green AI energy overhead is < 20% of facility savings."""
        if not self.has_results:
            self.skipTest("outputs/paper_results.json not found. Run main.py first.")
        
        h2 = self.results["section_5_3_H2"]
        rho = h2["rho_overhead_pct"]
        dE_net = h2["delta_E_net_kwh"]

        self.assertLess(rho, 20.0, f"Computational overhead ({rho}%) must remain strictly below 20.0%")
        self.assertGreater(dE_net, 0.0, f"Net energy savings after subtracting edge compute ({dE_net} kWh) must be > 0")
        self.assertTrue(h2["H2_accepted"], "H2 must be formally marked as accepted")

    def test_h3_thermal_comfort_qos_above_95_percent(self):
        """Verify Hypothesis H3: Occupant thermal comfort QoS is sustained at >= 95.0% satisfaction."""
        if not self.has_results:
            self.skipTest("outputs/paper_results.json not found. Run main.py first.")
        
        h3 = self.results["section_5_4_H3"]
        qos = h3["qos_predictive_pct"]

        self.assertGreaterEqual(qos, 95.0, f"Thermal comfort QoS ({qos}%) must be >= 95.0% under predictive dispatch")
        self.assertTrue(h3["H3_accepted"], "H3 must be formally marked as accepted")

if __name__ == "__main__":
    unittest.main()
