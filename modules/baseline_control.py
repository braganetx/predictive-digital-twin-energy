"""
modules/baseline_control.py — Conventional Reactive Threshold Baseline Controller.
Implements the standard rule-based Building Management System (BMS) with
hysteresis deadband control and fixed operating time schedules.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from config import (
    N_ZONES,
    N_DAYS,
    N_STEPS,
    STEP_MIN,
    ZONE_AREA,
    LIGHTING_W_M2,
    T_COMFORT_LOW,
    T_COMFORT_HIGH,
    BASELINE_HVAC_START,
    BASELINE_HVAC_STOP,
    OPEN_WEEKDAY,
    CLOSE_WEEKDAY,
)
from modules.thermal_model import (
    compute_zone_heat_gains,
    step_thermal_simulation,
    compute_hvac_power_kw,
)


def run_baseline_simulation(dataset: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes the conventional reactive baseline controller across the 56-day evaluation window.
    
    Control Logic:
    - Fixed operating schedule: HVAC starts at 09:00 AM (step 36) and stops at 10:30 PM (step 90).
    - Thermostatic hysteresis: compressor activates when T_zone >= 24.0°C and cuts off when T_zone <= 22.0°C.
    - Night setback: HVAC shut off with setpoints relaxed to 28.0°C.
    
    Args:
        dataset: Synthesized environmental and occupancy telemetry DataFrame.
        
    Returns:
        Dict[str, Any]: Baseline performance metrics including daily kWh and zone thermal trajectories.
    """
    T_in = np.full(N_ZONES, 23.5)

    results = {
        "daily_kwh": np.zeros(N_DAYS),
        "daily_qos": np.zeros(N_DAYS),
        "T_history": np.zeros((N_STEPS, N_ZONES)),
        "power_history": np.zeros(N_STEPS),
    }

    # Hysteresis compressor state flag per zone
    cooling_active = np.zeros(N_ZONES, dtype=bool)

    for global_step in range(N_STEPS):
        row = dataset.iloc[global_step]
        day = int(row["day"])
        step_in_day = int(row["step_in_day"])
        t_amb = float(row["t_amb"])
        is_open = bool(row["is_open"])

        # Fixed scheduled operation: 09:00 AM to 10:30 PM
        hvac_scheduled = (BASELINE_HVAC_START <= step_in_day < BASELINE_HVAC_STOP)

        if hvac_scheduled:
            for z in range(N_ZONES):
                if T_in[z] >= T_COMFORT_HIGH:   # >= 24.0°C -> activate compressor
                    cooling_active[z] = True
                elif T_in[z] <= T_COMFORT_LOW:  # <= 22.0°C -> deactivate compressor
                    cooling_active[z] = False

            # Active cooling targets lower bound (22.0°C); otherwise floats towards 24.5°C
            T_setpoints = np.where(cooling_active, 22.0, 24.5)
        else:
            cooling_active[:] = False
            T_setpoints = np.full(N_ZONES, 28.0)   # Night setback: HVAC shut down

        Q_gains = compute_zone_heat_gains(row.to_dict())
        Q_cooling, P_elec = compute_hvac_power_kw(T_in, T_setpoints, t_amb)

        # Commercial artificial lighting consumption
        P_lighting = (LIGHTING_W_M2 * ZONE_AREA.sum() / 1000.0) * float(is_open)
        P_total = P_elec.sum() + P_lighting

        # Forward integration of thermal ODE
        T_in = step_thermal_simulation(T_in, Q_cooling, Q_gains, t_amb)

        results["T_history"][global_step] = T_in.copy()
        results["power_history"][global_step] = P_total
        results["daily_kwh"][day] += P_total * (STEP_MIN / 60.0)

        if is_open:
            in_comfort = np.all((T_in >= T_COMFORT_LOW) & (T_in <= T_COMFORT_HIGH))
            results["daily_qos"][day] += float(in_comfort)

    open_steps_per_day = max(1, CLOSE_WEEKDAY - OPEN_WEEKDAY)
    results["daily_qos"] /= open_steps_per_day

    return results
