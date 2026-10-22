"""
modules/data_generator.py — Synthetic Environmental and Occupancy Telemetry Synthesis.
Generates an 8-week (56-day), high-resolution (15-min) calibrated dataset
representing commercial building physics and stochastic occupancy dynamics.
"""
import numpy as np
import pandas as pd
from config import (
    N_DAYS,
    STEPS_PER_DAY,
    STEP_MIN,
    N_STEPS,
    SEED,
    T_AMB_MEAN,
    T_AMB_SWING,
    SOLAR_PEAK,
    OPEN_WEEKDAY,
    CLOSE_WEEKDAY,
    OPEN_SUNDAY,
    CLOSE_SUNDAY,
    PEAK_OCCUPANCY,
    NOISE_STD_OCC,
    N_ZONES,
)


def generate_synthetic_dataset(seed: int = SEED) -> pd.DataFrame:
    """
    Synthesizes a calibrated 56-day, 15-minute telemetry dataset for a commercial facility.
    Generates ambient temperature profiles, solar irradiance, occupancy schedules,
    and indoor CO2 concentrations following ASHRAE Standard 62.1.
    
    Args:
        seed: Random seed for deterministic generation.
        
    Returns:
        pd.DataFrame: Complete continuous multi-sensor telemetry matrix.
    """
    rng = np.random.default_rng(seed)
    records = []

    for day in range(N_DAYS):
        weekday = day % 7   # 0=Monday ... 5=Saturday, 6=Sunday
        is_sunday = (weekday == 6)
        is_weekend = (weekday >= 5)

        open_step = OPEN_SUNDAY if is_sunday else OPEN_WEEKDAY
        close_step = CLOSE_SUNDAY if is_sunday else CLOSE_WEEKDAY

        for step in range(STEPS_PER_DAY):
            hour = step * STEP_MIN / 60.0
            global_step = day * STEPS_PER_DAY + step

            # Ambient Dry-Bulb Temperature (peak at 14:00, minimum at 05:00)
            t_amb = (T_AMB_MEAN +
                     T_AMB_SWING * np.sin(2 * np.pi * (hour - 5) / 24) +
                     rng.normal(0, 0.8))

            # Solar Irradiance Curve (Gaussian bell curve between 06:00 and 18:00)
            if 6 <= hour <= 18:
                solar = (SOLAR_PEAK *
                         np.exp(-((hour - 12) ** 2) / (2 * 4.0 ** 2)) +
                         rng.normal(0, 25))
                solar = max(0.0, solar)
            else:
                solar = 0.0

            # Multi-Zone Occupancy Synthesis
            is_open = (open_step <= step < close_step)
            if is_open:
                lunch_peak = np.exp(-((hour - 13) ** 2) / (2 * 1.2 ** 2))
                evening_peak = np.exp(-((hour - 19) ** 2) / (2 * 1.5 ** 2))
                base_occ = 0.30 + 0.55 * max(lunch_peak, evening_peak)
                if is_weekend:
                    base_occ *= 1.35   # 35% higher foot-traffic surge on weekends

                total_occ = min(PEAK_OCCUPANCY,
                                PEAK_OCCUPANCY * base_occ *
                                (1 + rng.normal(0, NOISE_STD_OCC)))
                total_occ = max(200.0, total_occ)

                # Distribution across thermal zones
                occ_weights = np.array([0.30, 0.35, 0.20, 0.15])
                zone_occ = (total_occ * occ_weights *
                            (1 + rng.normal(0, 0.03, N_ZONES)))
                zone_occ = np.maximum(50.0, zone_occ)
            else:
                zone_occ = np.zeros(N_ZONES)
                total_occ = 0.0

            # Indoor CO2 Concentration [ppm] (ASHRAE 62.1 ventilation dynamics)
            co2_base = 420.0
            co2 = co2_base + total_occ * 0.006 + rng.normal(0, 8)
            co2 = max(co2_base, co2)

            records.append({
                "global_step": global_step,
                "day": day,
                "step_in_day": step,
                "hour": round(hour, 4),
                "weekday": weekday,
                "is_weekend": int(is_weekend),
                "is_sunday": int(is_sunday),
                "is_open": int(is_open),
                "t_amb": round(t_amb, 3),
                "solar_irr": round(solar, 2),
                "total_occupancy": int(total_occ),
                "occ_z1": int(zone_occ[0]),
                "occ_z2": int(zone_occ[1]),
                "occ_z3": int(zone_occ[2]),
                "occ_z4": int(zone_occ[3]),
                "co2_ppm": round(co2, 1),
            })

    df = pd.DataFrame(records)
    df.index = pd.date_range("2026-08-01 00:00", periods=N_STEPS, freq="15min")
    return df
