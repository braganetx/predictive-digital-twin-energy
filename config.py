# config.py — Parametros Calibrados do Gemeo Digital Preditivo Hardware-Aware
import numpy as np

# --- SIMULACAO TEMPORAL ---
N_DAYS       = 56          # 8 semanas contiguas
STEP_MIN     = 15          # intervalo de amostragem em minutos
STEPS_PER_DAY = 96         # 24h * 4 passos/h
N_STEPS      = N_DAYS * STEPS_PER_DAY   # 5376 passos totais
SEED         = 42          # reprodutibilidade

# --- ZONAS TERMICAS (45.000 m² GLA) ---
N_ZONES = 4
ZONE_NAMES = ["Anchor Retail", "Specialty Boutiques", "Food Court", "Central Atrium"]

# Resistencia termica do envelope R_th [°C/kW] por zona
R_TH = np.array([0.025, 0.020, 0.035, 0.030])

# Capacitancia termica C_th [kJ/°C] por zona (massa termica de concreto e alvenaria)
C_TH = np.array([2.1e6, 2.8e6, 0.9e6, 1.4e6])

# Temperatura de conforto target [°C] (ASHRAE 55 / ISO 7730)
T_COMFORT_LOW  = 22.0
T_COMFORT_HIGH = 24.0
T_TARGET       = 23.0

# --- CLIMA (Goiania/Brasilia - Regiao Centro-Oeste) ---
T_AMB_MEAN   = 27.2    # media diaria [°C]
T_AMB_SWING  = 8.85    # amplitude diurna (36.2 - 18.5) / 2
SOLAR_PEAK   = 950.0   # W/m² — pico de irradiancia solar
SOLAR_AREA_ATRIUM = 450.0   # m² de claraboia envidracada no Atrio

# --- CARGAS TERMICAS INTERNAS ---
Q_SENSIBLE_PER_PERSON = 0.100  # kW por ocupante (100W/pessoa - ASHRAE)
LIGHTING_W_M2         = 15.0   # W/m² de iluminacao comercial

# Area por zona [m²]
ZONE_AREA = np.array([12000, 18000, 6000, 9000])  # total = 45.000 m²

# Carga termica de equipamentos da praca de alimentacao [kW]
Q_FOOD_COURT_EQUIPMENT = 120.0   # fornos, fritadeiras, exaustores

# --- OCUPACAO COMERCIAL ---
PEAK_OCCUPANCY = 8500   # visitantes simultaneos no pico de fim de semana
NOISE_STD_OCC  = 0.04   # ruido estocastico

# Horarios de funcionamento (passos de 15 min no dia)
OPEN_WEEKDAY  = 40   # 10:00
CLOSE_WEEKDAY = 88   # 22:00
OPEN_SUNDAY   = 48   # 12:00
CLOSE_SUNDAY  = 80   # 20:00

# --- HVAC CENTRAL (Chiller Plant ~3.700 kW total) ---
# Capacidade de projeto com fator de seguranca ASHRAE para picos de ocupacao
Q_HVAC_MAX_PER_ZONE = np.array([950.0, 1150.0, 750.0, 850.0])  # kW de resfriamento maximo
T_SET_MIN = 20.0
T_SET_MAX = 26.0
DELTA_T_MAX_PER_STEP = 0.5   # °C/passo

# Chiller COP de referencia
COP_CHILLER_REF = 3.8

# --- PREVISAO ---
LOOKBACK_STEPS   = 96    # 24h
FORECAST_HORIZON = 16   # 4h
LSTM_HIDDEN_SIZE = 48
LSTM_NUM_LAYERS  = 2
TRAIN_DAYS       = 42
TEST_DAYS        = 14

# --- OTIMIZACAO NSGA-II / MOEA/D ---
OPTIM_POP_SIZE   = 80
OPTIM_N_GEN      = 35
OPTIM_CROSSOVER_PROB = 0.90
OPTIM_CROSSOVER_ETA  = 20
OPTIM_MUTATION_PROB  = 0.10
OPTIM_MUTATION_ETA   = 20
MOEAD_NEIGHBORHOOD   = 15

# --- CONTROLE REATIVO (BASELINE) ---
BASELINE_DEADBAND   = 2.0    # 22.0°C a 24.0°C
BASELINE_HVAC_START = 36    # 09:00
BASELINE_HVAC_STOP  = 90    # 22:30

# --- VALIDACAO ESTATISTICA ---
ALPHA        = 0.05
N_BOOTSTRAP  = 10000

# --- HARDWARE MONITOR ---
IDLE_POWER_WATTS = 42.0
GRID_CARBON_BR   = 0.075   # kg CO2/kWh
