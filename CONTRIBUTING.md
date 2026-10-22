# Contributing & Reproducibility Guidelines

Thank you for your interest in contributing to the **Predictive Digital Twin with Hardware-Aware Multi-Objective Optimization** project. This repository is dedicated to open science, reproducible empirical benchmarks, and sustainable computational engineering for smart building decarbonization.

---

## 🔬 Scientific Integrity & Anti-Hallucination Principles

As an academic research benchmark accompanying an Elsevier SSRN research preprint (DOI: [10.2139/ssrn.7645518](https://doi.org/10.2139/ssrn.7645518)), all contributions must adhere to strict scientific rigor:

1. **Deterministic Reproducibility:**
   All stochastic algorithms (dataset generation, multi-objective evolutionary optimization, neural network initializations) must preserve fixed random seeds (`SEED = 42` by default).

2. **Thermodynamic Consistency:**
   Any modifications to the multi-zone lumped-parameter heat equations (`modules/thermal_model.py`) must satisfy energy conservation boundaries:
   $$\sum_{j} A_{ij} = 0$$

3. **Empirical Verification:**
   Before submitting changes, all scientific unit tests must pass:
   ```bash
   make test
   ```

---

## 🛠️ Development Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/model-enhancement
   ```

2. Set up your local Python environment:
   ```bash
   make install-dev
   ```

3. Ensure compliance with code formatting and typing:
   - Use standard PEP 8 formatting.
   - Include type annotations across all function signatures.
   - Document any new parameters with reference to international building standards (ASHRAE, ISO, Clarke).

4. Submit a Pull Request clearly explaining:
   - The theoretical motivation or performance improvement.
   - Comparative benchmarking results against the baseline.
