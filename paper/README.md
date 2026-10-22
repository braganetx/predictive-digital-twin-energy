# Academic Paper Information & Preprint Dossier

## Predictive Digital Twin with Hardware-Aware Multi-Objective Optimization for Commercial Building Energy Efficiency

* **Lead Author:** Adão de Jesus Braga
* **Affiliation:** Eletriza360 Negócios Digitais | USP/ESALQ
* **Journal:** SSRN Electronic Journal (Elsevier)
* **SSRN Abstract ID:** 7645518
* **DOI:** [10.2139/ssrn.7645518](https://doi.org/10.2139/ssrn.7645518)
* **Publication Date:** October 2026
* **Length:** 26 Pages (Academic Research Paper)
* **Preprint PDF:** [`Predictive_Digital_Twin_Hardware_Aware.pdf`](./Predictive_Digital_Twin_Hardware_Aware.pdf)

---

### Abstract

Commercial building heating, ventilation, and air conditioning (HVAC) and lighting systems account for substantial worldwide electrical consumption and typically operate through rigid reactive automation. Because existing building management systems (BMS) merely respond after environmental setpoint thresholds are breached without anticipating future dynamic loads, notable energy waste occurs during occupancy transients and peak tariff windows. Although building simulation platforms accurately model thermal dynamics, existing digital twin implementations rarely account for the computational energy overhead of executing machine learning and optimization algorithms in real time. 

To bridge this critical gap, this paper introduces an autonomic, closed-loop predictive digital twin framework with hardware-aware multi-objective optimization tailored for large-scale commercial facilities. The architecture synthesizes five operational phases:
1. Multi-sensor BMS telemetry ingestion adhering to big-data integration standards.
2. Comparative multi-horizon forecasting (SARIMA, Prophet, and LSTM recurrent networks).
3. Containerized lumped-parameter thermodynamic differential simulation executed inside Kubernetes pods.
4. Non-dominated Sorting Genetic Algorithm II (NSGA-II) and Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D) resolving conflicting energy and comfort objectives.
5. In-situ processor-level energy telemetry (via pyRAPL, CodeCarbon, nvidia-smi) quantifying computational overhead under Green AI principles.

Experimental evaluations conducted across a calibrated 45,000 m² gross leasable area commercial center demonstrate that the predictive digital twin achieves a statistically verified net electrical energy reduction of **14.16%** (paired Wilcoxon signed-rank test $W = 0.0, p < 0.001, N = 56$; bootstrap 95% CI: $[14.00\%, 14.47\%]$) compared to conventional reactive baseline control. Concurrently, dynamic computational energy overhead remains negligible ($\rho \approx 0.00001\%$, and $< 0.04\%$ including full server idle draw), far below the strict 20% Green AI constraint, while occupant thermal comfort is preserved at a **96.91% Quality-of-Service (QoS)** compliance rate. This work provides an open-science, reproducible empirical benchmark establishing that containerized predictive digital twins can achieve verified net-positive building decarbonization.

---

### Keywords
`Digital Twin` · `Building Energy Efficiency` · `Hardware-Aware Optimization` · `Green AI` · `Predictive Control` · `Kubernetes Orchestration` · `NSGA-II` · `Wilcoxon Signed-Rank Test` · `Thermal Comfort QoS`
