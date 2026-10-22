"""
modules/thermal_model.py — Multi-Zone Lumped-Parameter Thermodynamic ODE Solver.
Implements the continuous-time thermal state equations and dynamic chiller models
formalized in Clarke (2001) and ANSI/ASHRAE Standard 90.1.
"""
from typing import Dict, Any, Tuple
import numpy as np
from config import (
    N_ZONES,
    R_TH,
    C_TH,
    ZONE_AREA,
    STEP_MIN,
    COP_CHILLER_REF,
    SOLAR_AREA_ATRIUM,
    Q_SENSIBLE_PER_PERSON,
    LIGHTING_W_M2,
    Q_FOOD_COURT_EQUIPMENT,
    Q_HVAC_MAX_PER_ZONE,
)

# Solar Heat Gain Coefficient (SHGC) for commercial Low-E double glazing (ASHRAE 90.1)
SHGC_ATRIUM_GLASS: float = 0.32

# Residual sensible internal load fraction for food court equipment
# (60% of appliance sensible heat is evacuated directly by commercial kitchen hoods)
FOOD_COURT_INTERNAL_FRACTION: float = 0.40


def get_chiller_cop(t_amb: float) -> float:
    """
    Computes the instantaneous Coefficient of Performance (COP) of the central chiller plant
    as a linear function of exterior ambient dry-bulb temperature (condenser lift).
    
    Morning mild conditions (20°C): COP ~ 4.2 - 4.5
    Peak afternoon heat (36°C): COP ~ 3.0 - 3.2
    
    Args:
        t_amb: Ambient outdoor temperature [°C].
        
    Returns:
        float: Calibrated chiller COP, clipped between physical bounds [2.8, 4.8].
    """
    cop = COP_CHILLER_REF - 0.075 * (t_amb - 25.0)
    return float(np.clip(cop, 2.8, 4.8))


def compute_zone_heat_gains(step_data: Dict[str, Any]) -> np.ndarray:
    """
    Computes the total sensible heat gain vector Q_gains [kW] across all 4 thermal zones
    for a single 15-minute simulation interval.
    
    Components:
    1. Occupant metabolic sensible load (100 W/person, ASHRAE 55).
    2. Fenestrated direct solar gains in Central Atrium skylight (z4).
    3. Commercial artificial lighting dissipation (15 W/m²).
    4. Food court internal cooking appliance sensible dissipation (z3).
    
    Args:
        step_data: Dictionary containing instantaneous sensor telemetry.
        
    Returns:
        np.ndarray: Vector of zone heat gains [kW] of shape (N_ZONES,).
    """
    solar = float(step_data["solar_irr"])
    occ = np.array([float(step_data[f"occ_z{z+1}"]) for z in range(N_ZONES)])
    is_open = float(step_data["is_open"])

    # 1. Metabolic sensible heat
    Q_occ = occ * Q_SENSIBLE_PER_PERSON

    # 2. Fenestrated solar heat gain in Atrium (z4)
    Q_solar = np.zeros(N_ZONES)
    Q_solar[3] = (solar * SOLAR_AREA_ATRIUM * SHGC_ATRIUM_GLASS) / 1000.0   # W -> kW

    # 3. Commercial artificial lighting load
    Q_lighting = (LIGHTING_W_M2 * ZONE_AREA / 1000.0) * is_open

    # 4. Food court commercial appliances (z3) with hood exhaust fraction
    Q_equip = np.zeros(N_ZONES)
    Q_equip[2] = (Q_FOOD_COURT_EQUIPMENT * FOOD_COURT_INTERNAL_FRACTION) * is_open

    return Q_occ + Q_solar + Q_lighting + Q_equip


def build_inter_zone_matrix() -> np.ndarray:
    """
    Constructs the inter-zonal conductive and convective coupling conductance matrix A_inter [kW/°C].
    Satisfies thermodynamic energy conservation where sum_j A_ij = 0 (Clarke, 2001).
    
    Returns:
        np.ndarray: Inter-zonal coupling matrix of shape (N_ZONES, N_ZONES).
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


A_INTER: np.ndarray = build_inter_zone_matrix()


def step_thermal_simulation(T_in: np.ndarray,
                            Q_hvac_cooling: np.ndarray,
                            Q_gains: np.ndarray,
                            t_amb: float,
                            dt_seconds: float = STEP_MIN * 60.0) -> np.ndarray:
    """
    Integrates the multi-zone lumped-parameter thermal differential equation (Clarke, 2001):
    C_th * (dT_in / dt) = K_amb * (T_amb - T_in) + A_inter * T_in - Q_cooling + Q_gains
    
    Args:
        T_in: Current indoor air temperature vector [°C].
        Q_hvac_cooling: Delivered HVAC sensible cooling capacity [kW].
        Q_gains: Total sensible internal and solar heat gains [kW].
        t_amb: Ambient outdoor air temperature [°C].
        dt_seconds: Time integration step in seconds (default: 900s / 15 min).
        
    Returns:
        np.ndarray: Updated indoor temperature vector [°C] of shape (N_ZONES,).
    """
    dt_hours = dt_seconds / 3600.0
    K_amb = 1.0 / R_TH   # Envelope thermal conductance [kW/°C]

    dT_dt = (K_amb * (t_amb - T_in) +
             A_INTER @ T_in -
             Q_hvac_cooling +
             Q_gains) * 3600.0 / C_TH

    T_new = T_in + dT_dt * dt_hours
    return np.clip(T_new, 18.0, 38.0)


def compute_hvac_power_kw(T_in: np.ndarray,
                          T_setpoints: np.ndarray,
                          t_amb: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates thermal cooling load [kW_th] and required electrical compressor draw [kW_e].
    
    Args:
        T_in: Current indoor temperature vector [°C].
        T_setpoints: Target temperature setpoints [°C].
        t_amb: Ambient outdoor temperature [°C].
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Q_cooling [kW_th], P_electrical [kW_e]).
    """
    cop = get_chiller_cop(t_amb)
    delta_T = np.maximum(0.0, T_in - T_setpoints)
    cooling_ratio = np.clip(delta_T / 0.8, 0.0, 1.0)
    Q_cooling = cooling_ratio * Q_HVAC_MAX_PER_ZONE
    P_electrical = Q_cooling / cop
    return Q_cooling, P_electrical
