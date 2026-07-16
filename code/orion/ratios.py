"""
Dimensionless control ratios R_C and R_B.

R_C = C_fast / W
    Measures how much of the active working set fits in fast memory.
    R_C ≥ 1 → fully resident; R_C < θ_C → capacity-limited regime.

    R_C has a floor. Activations and the KV cache must be resident for a step to
    run at all, so only parameter bytes are tradeable:

        R_C ≥ (W_act + W_kv) / W

    The floor rises with batch × seq and can exceed θ_C, at which point the
    capacity-limited regime is unreachable at that configuration. Use
    `rc_floor()` before designing a sweep.

R_B = B_slow * T_comp / D = T_comp / T_transfer
    An overlap ratio: how much of the compulsory per-step transfer fits behind
    computation.
        R_B ≥ 1 → transfer hides behind compute
        R_B < 1 → transfer is exposed; stall = T_transfer − T_comp
    R_B < θ_B = 1 → I/O-limited regime.

    NOTE — this is a correction. The paper defines R_B = B_slow·Δt/D with Δt the
    *step duration*, which is not well posed: steady state needs D/Δt ≤ B_slow,
    so that ratio is ≥ 1 by construction and its sub-1 range (including the
    paper's θ_B = 0.40 and every I/O-limited operating point) describes a
    diverging queue rather than a reachable state. Substituting Δt := T_comp
    fixes this and moves the boundary to a derived 1.0. See config.py.

R_C and R_B are not independently assignable. D grows as R_C falls, so R_B is
coupled to R_C; the throttle decouples them only downward. At each R_C the
reachable set is R_B ≤ B_hw·T_comp/D(R_C) — the probe region is a triangle, not
a rectangle. `rb_ceiling()` gives the bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from orion.config import (
    THETA_C, THETA_B, Regime, HardwareProfile, ModelSpec,
)


@dataclass
class OperatingPoint:
    """A fully characterised (R_C, R_B) operating point."""
    r_c: float            # fast-memory residency ratio
    r_b: float            # overlap ratio T_comp / T_transfer
    # Raw quantities (optional, for book-keeping)
    c_fast_gb: float = 0.0
    w_gb: float      = 0.0
    b_slow_gbs: float = 0.0
    t_comp_s: float   = 0.0   # was delta_t_s; step duration cannot define R_B
    d_gb: float       = 0.0

    @property
    def regime(self) -> Regime:
        return classify_regime(self.r_c, self.r_b)

    def __repr__(self) -> str:
        return (f"OperatingPoint(R_C={self.r_c:.3f}, R_B={self.r_b:.3f}, "
                f"regime={self.regime.name})")


def compute_rc(c_fast_bytes: float, w_bytes: float) -> float:
    """
    R_C = C_fast / W

    Args:
        c_fast_bytes: Fast-memory (HBM) capacity in bytes actually available
                      to the model (after OS/framework overhead).
        w_bytes:      Active working-set size W = W_param + W_act + W_kv.

    Returns:
        Dimensionless residency ratio.  > 1 means fully resident.
    """
    if w_bytes <= 0:
        raise ValueError("Working-set size W must be positive.")
    return c_fast_bytes / w_bytes


def compute_rb(b_slow_bps: float, t_comp_s: float, d_bytes: float) -> float:
    """
    R_B = B_slow * T_comp / D = T_comp / T_transfer

    The overlap ratio: computation time divided by the time to move the
    compulsory per-step transfer volume across the slow link.

    Args:
        b_slow_bps:  Sustained host-to-device bandwidth [bytes/s], measured via
                     NVML nvmlDeviceGetPcieThroughput over a 10-s window.
        t_comp_s:    Mean per-step *computation* time [seconds], isolated via
                     CUDA events. This is deliberately NOT the step duration:
                     step duration absorbs the transfer wait, which pins the
                     ratio at ≥ 1 and destroys its information content. See the
                     module docstring.
        d_bytes:     Compulsory data volume transferred per step [bytes],
                     derived from DMA transfer logs.

    Returns:
        Dimensionless overlap ratio. ≥ 1 means transfer hides behind compute.
    """
    if d_bytes <= 0:
        raise ValueError("Transfer volume D must be positive.")
    if t_comp_s <= 0:
        raise ValueError("Computation time T_comp must be positive.")
    return (b_slow_bps * t_comp_s) / d_bytes


def step_time(t_comp_s: float, d_bytes: float, b_slow_bps: float) -> float:
    """
    Steady-state step duration under perfect overlap: max(T_comp, T_transfer).

    This is what the paper's Δt actually is, and why it cannot appear in R_B:
    substituting it back gives B_slow·Δt/D ≥ 1 identically.
    """
    return max(t_comp_s, d_bytes / b_slow_bps)


def rc_floor(w_act_bytes: float, w_kv_bytes: float, w_bytes: float) -> float:
    """
    Smallest R_C the configuration admits: (W_act + W_kv) / W.

    Activations and the KV cache cannot be evicted mid-step, so residency below
    this floor is not a harder operating point — it is an impossible one.
    """
    if w_bytes <= 0:
        raise ValueError("Working-set size W must be positive.")
    return (w_act_bytes + w_kv_bytes) / w_bytes


def rb_ceiling(b_slow_bps: float, t_comp_s: float, d_bytes: float) -> float:
    """
    Largest R_B reachable at this R_C, given the link's unthrottled rate.

    The token bucket can only throttle *below* the hardware rate, so R_B is
    capped at B_hw·T_comp/D. Raising it further requires shrinking D (more
    residency, smaller batch): bandwidth cannot be added in software.
    """
    return compute_rb(b_slow_bps, t_comp_s, d_bytes)


def classify_regime(r_c: float, r_b: float,
                    theta_c: float = THETA_C,
                    theta_b: float = THETA_B) -> Regime:
    """
    Analytical regime assignment from the two ratios.

        if R_C < θ_C   →  CAPACITY_LIMITED
        elif R_B < θ_B →  IO_LIMITED
        else           →  COORDINATION_DOMINATED

    θ_B = 1 is derived (transfer stops hiding behind compute). θ_C = 0.50 is an
    unvalidated convention — see config.py.

    This is a definition, not a trained model: it has no accuracy figure because
    it cannot be wrong about its own thresholds. The learned feature tree in
    classifier.py is the thing that has an error rate, and it is uncalibrated.
    """
    if r_c < theta_c:
        return Regime.CAPACITY_LIMITED
    if r_b < theta_b:
        return Regime.IO_LIMITED
    return Regime.COORDINATION_DOMINATED


def from_hardware_model(
    hw: HardwareProfile,
    model: ModelSpec,
    t_comp_s: float,
    d_bytes: Optional[float] = None,
    c_fast_fraction: float = 1.0,
) -> OperatingPoint:
    """
    Construct an OperatingPoint from hardware + model descriptors.

    Args:
        hw:               HardwareProfile for the target platform.
        model:            ModelSpec for the deployed model.
        t_comp_s:         Measured mean per-step computation time [s]. Not the
                          step duration — see compute_rb().
        d_bytes:          Compulsory DMA transfer volume per step [bytes].
                          If None, estimated as max(0, W - C_fast).
        c_fast_fraction:  Fraction of hw.c_fast_bytes made available
                          (default 1.0; reduce to constrain HBM).

    Raises:
        ValueError: if the implied R_C lies below the configuration's floor,
            which would describe an operating point that cannot exist.
    """
    c_fast = hw.c_fast_bytes * c_fast_fraction
    w = model.w_total_bytes

    if d_bytes is None:
        # Conservative lower estimate: eviction volume when W > C_fast
        d_bytes = max(0.0, w - c_fast)
        if d_bytes == 0:
            d_bytes = 0.01 * w   # compulsory minimum even when fully resident

    r_c = compute_rc(c_fast, w)
    floor = rc_floor(model.w_act_gb * 1e9, model.w_kv_gb * 1e9, w)
    if r_c < floor:
        raise ValueError(
            f"R_C={r_c:.3f} is below the floor {floor:.3f} for {model.name} at "
            f"batch={model.batch}, seq={model.seq_len}: activations+KV alone "
            f"exceed the fast-memory budget, so this point cannot exist. "
            f"Reduce batch or sequence length."
        )
    r_b = compute_rb(hw.b_slow_bps, t_comp_s, d_bytes)

    return OperatingPoint(
        r_c=r_c, r_b=r_b,
        c_fast_gb=c_fast / 1e9,
        w_gb=w / 1e9,
        b_slow_gbs=hw.b_slow_gbs,
        t_comp_s=t_comp_s,
        d_gb=d_bytes / 1e9,
    )


def predict_theta_b(hw: HardwareProfile,
                    r_b_sat: Optional[float] = None) -> float:
    """
    Deprecated. θ_B is derived, not predicted.

    This implemented θ_B = R_B_sat·(1 + α_wb + α_q), fitted per platform against
    a claimed ≤4.7% prediction error. That construction presupposed the old
    step-duration definition of R_B, under which θ_B < 1 named an unreachable
    state; the amplification factors α_wb and α_q were free parameters fitted to
    reproduce a threshold that should never have been below 1.

    Under R_B = T_comp/T_transfer the boundary is exactly 1.0 on physical
    grounds, identically across platforms, with nothing left to calibrate.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(
        "predict_theta_b() is obsolete: θ_B = 1.0 is derived, not calibrated. "
        "Use config.THETA_B. See orion/config.py for the derivation."
    )
