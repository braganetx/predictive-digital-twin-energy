"""
config.py — Physical, Thermodynamic, and Experimental Parameters.
Calibrated for a large-scale commercial shopping center (45,000 m² GLA)
located in a tropical/sub-tropical climate (Central-West Brazil).

Standards & References:
- Envelope & Glass Properties: ANSI/ASHRAE/IES Standard 90.1-2022
- Thermal Comfort & QoS Limits: ANSI/ASHRAE Standard 55-2020 / ISO 7730
- Multi-Zone Thermal Modeling: Clarke, J.A. (2001)
- Multi-Objective Optimization: Deb et al. (2002), Zhang & Li (2007)
- Green AI Computational Telemetry: Schwartz et al. (2019)
"""
import numpy as np

# --- TEMPORAL SIMULATION HORIZONS ---
N_DAYS: int = 56              # 8 continuous operational weeks
STEP_MIN: int = 15            # Discrete sampling interval in minutes
STEPS_PER_DAY: int = 96       # 24 hours * 4 steps/hour
N_STEPS: int = N_DAYS * STEPS_PER_DAY   # 5,376 total simulation intervals
SEED: int = 42                # Random seed for deterministic reproducibility

# --- THERMAL ZONES (45,000 m² Gross Leasable Area - GLA) ---
N_ZONES: int = 4
ZONE_NAMES: list[str] = ["Anchor Retail", "Specialty Boutiques", "Food Court", "Central Atrium"]

# Envelope thermal resistance R_th [°C/kW] per zone
R_TH: np.ndarray = np.array([0.025, 0.020, 0.035, 0.030])

# Lumped thermal capacitance C_th [kJ/°C] per zone (concrete and masonry thermal mass)
C_TH: np.ndarray = np.array([2.1e6, 2.8e6, 0.9e6, 1.4e6])

# Indoor Thermal Comfort Thresholds [°C] (ASHRAE Standard 55 / ISO 7730)
T_COMFORT_LOW: float = 22.0
T_COMFORT_HIGH: float = 24.0
T_TARGET: float = 23.0

# --- CLIMATIC PROFILES (Goiania/Brasilia Metropolises) ---
T_AMB_MEAN: float = 27.2      # Diurnal mean ambient dry-bulb temperature [°C]
T_AMB_SWING: float = 8.85     # Diurnal temperature swing: (36.2°C - 18.5°C) / 2
SOLAR_PEAK: float = 950.0     # Peak direct solar irradiance [W/m²]
SOLAR_AREA_ATRIUM: float = 450.0  # Fenestrated skylight area in Central Atrium [m²]

# --- INTERNAL HEAT GAINS ---
Q_SENSIBLE_PER_PERSON: float = 0.100  # Sensible metabolic heat rate: 100 W/person (ASHRAE 55)
LIGHTING_W_M2: float = 15.0           # Commercial lighting power density [W/m²]

# Floor area allocation per zone [m²] (Sum = 45,000 m²)
ZONE_AREA: np.ndarray = np.array([12000, 18000, 6000, 9000])

# Internal commercial kitchen equipment heat load in Food Court [kW]
Q_FOOD_COURT_EQUIPMENT: float = 120.0

# --- STOCHASTIC OCCUPANCY DYNAMICS ---
PEAK_OCCUPANCY: int = 8500    # Peak simultaneous visitors on weekend surges
NOISE_STD_OCC: float = 0.04   # Gaussian occupancy perturbation noise

# Commercial operating schedules (15-min index in day)
OPEN_WEEKDAY: int = 40        # 10:00 AM
CLOSE_WEEKDAY: int = 88       # 10:00 PM
OPEN_SUNDAY: int = 48         # 12:00 PM
CLOSE_SUNDAY: int = 80        # 08:00 PM

# --- CENTRAL CHILLER PLANT (~3,700 kW Total Cooling Capacity) ---
# Design capacity with ASHRAE safety margin for extreme occupancy peaks
Q_HVAC_MAX_PER_ZONE: np.ndarray = np.array([950.0, 1150.0, 750.0, 850.0])  # Max cooling [kW]
T_SET_MIN: float = 20.0
T_SET_MAX: float = 26.0
DELTA_T_MAX_PER_STEP: float = 0.5   # Maximum setpoint slew rate [°C/step]

# Nominal Chiller Coefficient of Performance (COP) at reference rating
COP_CHILLER_REF: float = 3.8

# --- TIME-SERIES FORECASTING BENCHMARK ---
LOOKBACK_STEPS: int = 96      # 24-hour historical lookback window
FORECAST_HORIZON: int = 16    # 4-hour forward lookahead horizon
LSTM_HIDDEN_SIZE: int = 48
LSTM_NUM_LAYERS: int = 2
TRAIN_DAYS: int = 42          # 6-week training split
TEST_DAYS: int = 14           # 2-week out-of-sample evaluation split

# --- MULTI-OBJECTIVE EVOLUTIONARY OPTIMIZATION (NSGA-II & MOEA/D) ---
OPTIM_POP_SIZE: int = 80
OPTIM_N_GEN: int = 35
OPTIM_CROSSOVER_PROB: float = 0.90
OPTIM_CROSSOVER_ETA: float = 20.0
OPTIM_MUTATION_PROB: float = 0.10
OPTIM_MUTATION_ETA: float = 20.0
MOEAD_NEIGHBORHOOD: int = 15

# --- CONVENTIONAL REACTIVE CONTROL BASELINE ---
BASELINE_DEADBAND: float = 2.0      # 22.0°C to 24.0°C hysteresis band
BASELINE_HVAC_START: int = 36      # 09:00 AM fixed start schedule
BASELINE_HVAC_STOP: int = 90       # 10:30 PM fixed stop schedule

# --- STATISTICAL HYPOTHESIS TESTING ---
ALPHA: float = 0.05
N_BOOTSTRAP: int = 10000

# --- HARDWARE-AWARE ENERGY TELEMETRY (GREEN AI) ---
SERVER_TDP_WATTS: float = 170.0    # Dual-socket Intel Xeon Silver 4210R server baseline
IDLE_POWER_WATTS: float = 42.0     # Baseline chassis quiescent power draw [W]
GRID_CARBON_BR: float = 0.075      # Average grid carbon emission factor [kg CO2/kWh]
