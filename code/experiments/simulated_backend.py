"""
Synthetic backend for CPU-only smoke-testing of the sweep plumbing.

THIS BACKEND MEASURES NOTHING. It emits LatencyRecords built from the
hand-written constants below, plus Gaussian noise. Its output is an illustration
of the shape the framework expects — not data, not a reproduction, and not
evidence for any claim. Do not report numbers obtained from it. Use
`experiments/cuda_backend.py` on real hardware to obtain measurements.

Provenance of the constants, stated plainly because it has been misdescribed
before:

  - `_REGIME_COEFFICIENTS` and `_BASELINE_MULTIPLIERS` were written by hand.
    They were never fitted to measurements, on A100 or anywhere else.
  - They do not agree with the manuscript. `_BASELINE_MULTIPLIERS` puts FlexGen
    at 1.240 (a 24% latency *increase*) in the coordination-dominated regime,
    whereas the manuscript's Table 1 reports −18.4% (a decrease) for the same
    method and regime, and inverts the I/O-limited column likewise. Running
    `reproduce_table2.py` therefore contradicts the table it names. The
    disagreement is real and unresolved; do not "fix" it by editing these
    constants to match the paper, which would be fabrication in the other
    direction. It resolves only by measuring.
  - Earlier revisions claimed the noise CV was "calibrated ... to match the
    measurement variance reported in Supplementary Table B.2" and pointed at raw
    JSONL traces "archived on Zenodo". No such traces exist in this repository
    and no measurement campaign backs the CV; it is a guess.

The regime boundaries it classifies against did change: θ_B is now 1.0 under
R_B = T_comp/T_transfer (see orion/config.py). The constants below were invented
against the old θ_B = 0.40 and have not been revisited, which is one more reason
to treat every number here as arbitrary.
"""

from __future__ import annotations

import math
import random
import time

from orion.config import (
    THETA_C, THETA_B, Regime, HardwareProfile, A100_80GB, WINDOW_SEC,
)
from orion.profiler import LatencyRecord
from orion.ratios import classify_regime


# Hand-written latency coefficients. NOT calibrated on A100 or any device.
# Format: {regime: (t_comp, t_mem, t_swap, t_sync)} in seconds per 10-s window
_REGIME_COEFFICIENTS = {
    Regime.CAPACITY_LIMITED: (
        0.12,   # T_comp: compute is not the bottleneck
        0.55,   # T_mem:  dominant — frequent eviction/reload
        0.18,   # T_swap: moderate
        0.08,   # T_sync: low (no complex coordination)
    ),
    Regime.COORDINATION_DOMINATED: (
        0.20,   # T_comp: balanced
        0.18,   # T_mem:  moderate (good residency)
        0.22,   # T_swap: moderate (overlap profitable)
        0.14,   # T_sync: significant (active coordination)
    ),
    Regime.IO_LIMITED: (
        0.10,   # T_comp: partially hidden
        0.12,   # T_mem:  low (but irrelevant — BW saturated)
        0.68,   # T_swap: dominant — PCIe saturated
        0.16,   # T_sync: amplified by DMA backpressure
    ),
}

# Hand-written per-method multipliers (relative to PyTorch default = 1.0).
# > 1 → higher latency (worse); < 1 → lower (better). These contradict the
# manuscript's Table 1 in sign for FlexGen/DeepSpeed — see module docstring.
_BASELINE_MULTIPLIERS = {
    "orion":      {Regime.CAPACITY_LIMITED: 0.960, Regime.COORDINATION_DOMINATED: 0.795, Regime.IO_LIMITED: 0.895},
    "flexgen":    {Regime.CAPACITY_LIMITED: 0.985, Regime.COORDINATION_DOMINATED: 1.240, Regime.IO_LIMITED: 0.920},
    "deepspeed":  {Regime.CAPACITY_LIMITED: 0.980, Regime.COORDINATION_DOMINATED: 1.180, Regime.IO_LIMITED: 0.950},
    "swapadvisor":{Regime.CAPACITY_LIMITED: 0.995, Regime.COORDINATION_DOMINATED: 0.900, Regime.IO_LIMITED: 1.050},
    "vllm":       {Regime.CAPACITY_LIMITED: 1.010, Regime.COORDINATION_DOMINATED: 0.870, Regime.IO_LIMITED: 0.940},
    "pytorch":    {Regime.CAPACITY_LIMITED: 1.000, Regime.COORDINATION_DOMINATED: 1.000, Regime.IO_LIMITED: 1.000},
}

_CV = 0.04    # guessed coefficient of variation; not measured


class SimulatedBackend:
    """
    Simulated hardware backend for offline experiment reproduction.

    Args:
        hw:      HardwareProfile to simulate (default: A100_80GB).
        method:  Orchestration method name (used for multiplier lookup).
        seed:    RNG seed for reproducibility.
    """

    def __init__(
        self,
        hw: HardwareProfile = A100_80GB,
        method: str = "orion",
        seed: int = 42,
    ) -> None:
        self.hw     = hw
        self.method = method.lower()
        self._rng   = random.Random(seed)

    def warmup(self, warmup_sec: float = 60.0) -> None:
        """Simulate 60-s warmup (no-op in simulation)."""
        pass

    def measure(self, r_c: float, r_b: float) -> LatencyRecord:
        """Generate one synthetic 10-s LatencyRecord at (r_c, r_b)."""
        regime = classify_regime(r_c, r_b)
        coeffs = _REGIME_COEFFICIENTS[regime]
        mult   = _BASELINE_MULTIPLIERS.get(self.method, {}).get(regime, 1.0)

        def noisy(val: float) -> float:
            return max(0.0, val * mult * (1 + self._rng.gauss(0, _CV)))

        t_comp = noisy(coeffs[0])
        t_mem  = noisy(coeffs[1])
        t_swap = noisy(coeffs[2])
        t_sync = noisy(coeffs[3])
        t_wall = t_comp + t_mem + t_swap + t_sync

        # DMA utilisation: high in IO_LIMITED, moderate in coordination-dominated
        dma_util = {
            Regime.CAPACITY_LIMITED:       0.35,
            Regime.COORDINATION_DOMINATED: 0.52,
            Regime.IO_LIMITED:             0.91,
        }[regime]

        cache_hit = {
            Regime.CAPACITY_LIMITED:       0.55,
            Regime.COORDINATION_DOMINATED: 0.81,
            Regime.IO_LIMITED:             0.74,
        }[regime]

        return LatencyRecord(
            timestamp=time.time(),
            r_c=r_c, r_b=r_b,
            t_comp=t_comp, t_mem=t_mem, t_swap=t_swap, t_sync=t_sync,
            t_wall=t_wall,
            swap_to_comp_ratio=t_swap / t_comp if t_comp > 0 else 0,
            cache_hit_rate=cache_hit + self._rng.gauss(0, 0.02),
            dma_utilisation=dma_util + self._rng.gauss(0, 0.03),
        )
