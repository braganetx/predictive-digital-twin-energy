# modules/baseline_control.py
import numpy as np
from config import *
from modules.thermal_model import (
    compute_zone_heat_gains, step_thermal_simulation, compute_hvac_power_kw
)

def run_baseline_simulation(dataset) -> dict:
    """
    Executa o controlador reativo (baseline) por 56 dias.
    Logica: automacao predial convencional BMS com horario fixo (09:00 as 22:30)
    e termostato de histerese com deadband de 2.0°C em torno de 23.0°C (22.0°C a 24.0°C).
    """
    T_in = np.full(N_ZONES, 23.5)
    
    results = {
        "daily_kwh": np.zeros(N_DAYS),
        "daily_qos": np.zeros(N_DAYS),
        "T_history": np.zeros((N_STEPS, N_ZONES)),
        "power_history": np.zeros(N_STEPS),
    }

    # Estado de histerese por zona
    cooling_active = np.zeros(N_ZONES, dtype=bool)

    for global_step in range(N_STEPS):
        row = dataset.iloc[global_step]
        day = int(row["day"])
        step_in_day = int(row["step_in_day"])
        t_amb = float(row["t_amb"])
        is_open = bool(row["is_open"])

        # O HVAC baseline liga as 09:00 (passo 36) e desliga as 22:30 (passo 90)
        hvac_scheduled = (BASELINE_HVAC_START <= step_in_day < BASELINE_HVAC_STOP)

        if hvac_scheduled:
            for z in range(N_ZONES):
                if T_in[z] >= T_COMFORT_HIGH:   # >= 24.0°C -> liga compressor
                    cooling_active[z] = True
                elif T_in[z] <= T_COMFORT_LOW:  # <= 22.0°C -> desliga compressor
                    cooling_active[z] = False
            
            # Quando cooling ativo, tenta puxar para 22.0°C; senao mantem em flutuacao
            T_setpoints = np.where(cooling_active, 22.0, 24.5)
        else:
            cooling_active[:] = False
            T_setpoints = np.full(N_ZONES, 28.0)   # HVAC desligado a noite

        Q_gains = compute_zone_heat_gains(row.to_dict())
        Q_cooling, P_elec = compute_hvac_power_kw(T_in, T_setpoints, t_amb)

        # Iluminacao
        P_lighting = (LIGHTING_W_M2 * ZONE_AREA.sum() / 1000.0) * float(is_open)
        P_total = P_elec.sum() + P_lighting

        # Evolucao termica
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
