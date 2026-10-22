# modules/thermal_model.py
import numpy as np
from config import *

# Coeficiente de Ganho de Calor Solar (SHGC) para vidro duplo Low-E comercial (ASHRAE 90.1)
SHGC_ATRIUM_GLASS = 0.32

# Fracao de calor sensivel residual dos equipamentos da praca de alimentacao
# (60% do calor de fornos/fritadeiras e exaurido diretamente por coifas mecanicas externas)
FOOD_COURT_INTERNAL_FRACTION = 0.40


def get_chiller_cop(t_amb: float) -> float:
    """
    Calcula o Coeficiente de Performance (COP) do chiller central em funcao
    da temperatura externa de bulbo seco (temperatura de condensacao/lift).
    Manha fresca (20°C): COP ~ 4.2 - 4.5
    Tarde quente (36°C): COP ~ 3.0 - 3.2
    """
    cop = COP_CHILLER_REF - 0.075 * (t_amb - 25.0)
    return float(np.clip(cop, 2.8, 4.8))


def compute_zone_heat_gains(step_data: dict) -> np.ndarray:
    """Calcula vetor Q_total por zona [kW] para um passo de 15 minutos."""
    solar = float(step_data["solar_irr"])
    occ   = np.array([float(step_data[f"occ_z{z+1}"]) for z in range(N_ZONES)])
    is_open = float(step_data["is_open"])

    # 1. Calor metabolico sensivel dos ocupantes (100 W/pessoa)
    Q_occ = occ * Q_SENSIBLE_PER_PERSON

    # 2. Ganho solar fenestrado no Atrio (z4) com vidro Low-E (SHGC=0.32)
    Q_solar = np.zeros(N_ZONES)
    Q_solar[3] = (solar * SOLAR_AREA_ATRIUM * SHGC_ATRIUM_GLASS) / 1000.0   # W -> kW

    # 3. Iluminacao comercial
    Q_lighting = (LIGHTING_W_M2 * ZONE_AREA / 1000.0) * is_open

    # 4. Equipamentos da praca de alimentacao (z3) com exaustao de coifas
    Q_equip = np.zeros(N_ZONES)
    Q_equip[2] = (Q_FOOD_COURT_EQUIPMENT * FOOD_COURT_INTERNAL_FRACTION) * is_open

    return Q_occ + Q_solar + Q_lighting + Q_equip


def build_inter_zone_matrix() -> np.ndarray:
    """
    Matriz de intercambio convectivo e condutivo interzonal A_inter [kW/°C].
    Acoplamento termico entre as 4 zonas do shopping.
    """
    conductance = {
        (0, 1): 25.0,   # Anchor Retail <-> Specialty Boutiques
        (1, 2): 20.0,   # Specialty Boutiques <-> Food Court
        (1, 3): 30.0,   # Specialty Boutiques <-> Central Atrium
        (0, 3): 18.0,   # Anchor Retail <-> Central Atrium
    }
    A = np.zeros((N_ZONES, N_ZONES))
    for (i, j), g in conductance.items():
        A[i, j] += g
        A[j, i] += g
        A[i, i] -= g
        A[j, j] -= g
    return A


A_INTER = build_inter_zone_matrix()


def step_thermal_simulation(T_in: np.ndarray,
                            Q_hvac_cooling: np.ndarray,
                            Q_gains: np.ndarray,
                            t_amb: float,
                            dt_seconds: float = STEP_MIN * 60) -> np.ndarray:
    """
    Integra a equacao diferencial de balanco termico lumped multizona (Clarke, 2001):
    C_th * dT_in/dt = K_amb*(T_amb - T_in) + A_inter*T_in - Q_cooling + Q_gains
    """
    dt_hours = dt_seconds / 3600.0
    K_amb = 1.0 / R_TH   # [kW/°C]

    dT_dt = (K_amb * (t_amb - T_in) +
             A_INTER @ T_in -
             Q_hvac_cooling +
             Q_gains) * 3600.0 / C_TH

    T_new = T_in + dT_dt * dt_hours
    return np.clip(T_new, 18.0, 38.0)


def compute_hvac_power_kw(T_in: np.ndarray, T_setpoints: np.ndarray, t_amb: float) -> tuple:
    """
    Calcula o calor de resfriamento necessario [kW termico] e a potencia eletrica [kW eletrico].
    """
    cop = get_chiller_cop(t_amb)
    delta_T = np.maximum(0.0, T_in - T_setpoints)
    cooling_ratio = np.clip(delta_T / 0.8, 0.0, 1.0)
    Q_cooling = cooling_ratio * Q_HVAC_MAX_PER_ZONE
    P_electrical = Q_cooling / cop
    return Q_cooling, P_electrical
