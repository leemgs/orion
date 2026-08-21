"""
Conservative service-time lower bound on inference latency.

    T_total ≥ max(T_comp, ρ·W·max(0, 1 − R_C), D / B_slow)

Taking the maximum permits ideal overlap and avoids double-counting traffic
represented by both the non-resident-byte and compulsory-transfer terms.
Contention, synchronization, and incomplete overlap may increase observed
latency, so the bound is not a performance predictor and its residual must not
be attributed to a single mechanism without additional measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from orion.config import HardwareProfile
from orion.ratios import OperatingPoint


@dataclass
class LowerBoundResult:
    t_comp_min_s: float    # isolated computation time
    t_mem_min_s: float     # ρ·W·max(0, 1−R_C)
    t_swap_min_s: float    # D / B_slow
    t_lower_bound_s: float # max of compute, memory, and transfer terms
    # Comparison with measured value (optional)
    t_measured_s: float = 0.0

    @property
    def achievability(self) -> float:
        """τ = T_lower / T_measured ∈ (0,1].  1.0 = perfectly optimal."""
        if self.t_measured_s <= 0:
            return float("nan")
        return self.t_lower_bound_s / self.t_measured_s

    @property
    def residual_fraction(self) -> float:
        """Unattributed fraction above the conservative lower bound."""
        return 1.0 - self.achievability


def compute_lower_bound(
    op: OperatingPoint,
    rho: float,
    t_measured_s: float = 0.0,
) -> LowerBoundResult:
    """
    Compute the structural lower bound for a given operating point.

    Args:
        op:           OperatingPoint (contains R_C, R_B, w_gb, d_gb, b_slow_gbs).
        rho:          HBM miss-penalty per byte [s/byte] (platform constant).
        t_measured_s: Measured wall-clock T_total for achievability ratio.

    Returns:
        LowerBoundResult with bound components and achievability τ.
    """
    w_bytes = op.w_gb * 1e9
    d_bytes = op.d_gb * 1e9
    b_slow  = op.b_slow_gbs * 1e9

    t_comp_min = max(0.0, op.t_comp_s)
    t_mem_min  = rho * w_bytes * max(0.0, 1.0 - op.r_c)
    t_swap_min = d_bytes / b_slow if b_slow > 0 else 0.0

    return LowerBoundResult(
        t_comp_min_s=t_comp_min,
        t_mem_min_s=t_mem_min,
        t_swap_min_s=t_swap_min,
        t_lower_bound_s=max(t_comp_min, t_mem_min, t_swap_min),
        t_measured_s=t_measured_s,
    )


def compute_lower_bound_from_hw(
    hw: HardwareProfile,
    w_bytes: float,
    d_bytes: float,
    r_c: float,
    t_comp_s: float = 0.0,
    t_measured_s: float = 0.0,
) -> LowerBoundResult:
    """Convenience wrapper that takes raw hardware and workload parameters."""
    b_slow = hw.b_slow_bps
    t_comp_min = max(0.0, t_comp_s)
    t_mem_min  = hw.rho * w_bytes * max(0.0, 1.0 - r_c)
    t_swap_min = d_bytes / b_slow if b_slow > 0 else 0.0
    return LowerBoundResult(
        t_comp_min_s=t_comp_min,
        t_mem_min_s=t_mem_min,
        t_swap_min_s=t_swap_min,
        t_lower_bound_s=max(t_comp_min, t_mem_min, t_swap_min),
        t_measured_s=t_measured_s,
    )


def sharpness_coefficient(
    t_values: list[float],
    r_values: list[float],
) -> float:
    """
    Empirical sharpness coefficient S = |d ln T_total / d ln R| at boundary.

    Estimated by finite differences over log-log coordinates. Interpretation
    requires a prespecified neighbourhood and uncertainty analysis; this helper
    does not establish a transition or a universal decision threshold.

    Args:
        t_values: T_total measurements at consecutive R values.
        r_values: Corresponding R (R_C or R_B) values.

    Returns:
        Peak |Δ ln T / Δ ln R| in the series.
    """
    if len(t_values) != len(r_values) or len(t_values) < 2:
        raise ValueError("Need at least 2 paired (T, R) values.")

    s_values = []
    for i in range(1, len(t_values)):
        dt = math.log(t_values[i]) - math.log(t_values[i - 1])
        dr = math.log(r_values[i]) - math.log(r_values[i - 1])
        if abs(dr) > 1e-12:
            s_values.append(abs(dt / dr))

    return max(s_values) if s_values else 0.0
