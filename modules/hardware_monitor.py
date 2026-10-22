# modules/hardware_monitor.py
import time
import psutil
import numpy as np

try:
    import pyRAPL
    PYRAPL_AVAILABLE = True
    pyRAPL.setup()
except (ImportError, Exception):
    PYRAPL_AVAILABLE = False

SERVER_TDP_WATTS = 170.0   # Xeon Silver 4210R x2
IDLE_POWER_WATTS = 42.0

class HardwareMonitor:
    def __enter__(self):
        self.t_start = time.perf_counter()
        if PYRAPL_AVAILABLE:
            try:
                self._meter = pyRAPL.Measurement("op")
                self._meter.begin()
            except Exception:
                self._meter = None
        return self

    def __exit__(self, *args):
        self.t_end = time.perf_counter()
        self.duration_s = max(1e-5, self.t_end - self.t_start)

        if PYRAPL_AVAILABLE and getattr(self, "_meter", None):
            try:
                self._meter.end()
                pkg_uj  = sum(self._meter.result.pkg or [0])
                dram_uj = sum(self._meter.result.dram or [0])
                self.energy_joules = (pkg_uj + dram_uj) / 1e6
                return
            except Exception:
                pass

        # Fallback estimativo baseado em CPU usage e tempo
        cpu_frac = max(0.15, psutil.cpu_percent(interval=None) / 100.0)
        dynamic_power = (SERVER_TDP_WATTS - IDLE_POWER_WATTS) * cpu_frac
        self.energy_joules = dynamic_power * self.duration_s

    @property
    def joules(self) -> float:
        return self.energy_joules

    @property
    def kwh(self) -> float:
        return self.energy_joules / 3_600_000.0


class CumulativeEnergyTracker:
    def __init__(self):
        self.total_joules = 0.0
        self.n_ops = 0
        self.breakdowns = {
            "data_pipeline": 0.0,
            "lstm_inference": 0.0,
            "thermal_simulation": 0.0,
            "optimization": 0.0,
        }

    def add(self, category: str, joules: float):
        self.total_joules += joules
        self.n_ops += 1
        self.breakdowns[category] = self.breakdowns.get(category, 0.0) + joules

    @property
    def total_kwh(self) -> float:
        return self.total_joules / 3_600_000.0

    def overhead_ratio(self, building_savings_kwh: float) -> float:
        if building_savings_kwh <= 0:
            return 0.0
        return (self.total_kwh / building_savings_kwh) * 100.0

    def report(self) -> dict:
        return {
            "total_joules": round(self.total_joules, 2),
            "total_kwh": round(self.total_kwh, 6),
            "n_operations": self.n_ops,
            "breakdown_joules": {k: round(v, 2) for k, v in self.breakdowns.items()},
            "mode": "pyRAPL" if PYRAPL_AVAILABLE else "psutil-fallback"
        }
