#!/usr/bin/env python3
"""
Experiment planner: which (R_C, R_B) operating points physically exist?

Run this BEFORE booking GPU time. It needs no GPU. It answers the question the
regime framework cannot dodge: for a given model, batch and sequence length on
given hardware, which of the three regimes can actually be entered?

Two hard limits bound the grid.

  R_C floor.  Activations and the KV cache must be resident for a step to run,
              so only parameter bytes are tradeable:

                  R_C >= (W_act + W_kv) / W

              Below the floor the operating point does not exist. The floor
              rises with batch x seq, and above ~10k tokens it exceeds
              theta_C = 0.50 -- the capacity-limited regime becomes unreachable.

  R_B.        The paper defines R_B = B_slow * dt / D with dt the step duration.
              In steady state the link must deliver D bytes every dt seconds, so
              D/dt <= B_slow, hence R_B >= 1 ALWAYS: the paper's R_B < 1 points
              are diverging queues, not operating points.

              This planner therefore reports R_B under the repaired definition

                  R_B := T_comp / T_transfer = B_slow * T_comp / D

              an overlap ratio, with the physically grounded boundary
              theta_B = 1.0 (transfer stops hiding behind compute). See
              README §Reachability.

T_comp is a roofline ESTIMATE (2*N*tokens / effective FLOPS), so the R_B column
is a prediction to be confirmed by measurement, not a measurement. The R_C
column is exact: it follows from tensor shapes alone.

Usage:
    python experiments/plan_reachable_grid.py
    python experiments/plan_reachable_grid.py --mfu 0.35 --platform A100-80GB
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orion.config import (
    LLAMA3_8B, A100_80GB, ALL_PLATFORMS, THETA_C, THETA_B, ModelSpec,
    HardwareProfile,
)

# A100 SXM dense FP16 tensor-core peak.
A100_PEAK_FLOPS = 312e12


@dataclass
class GridPoint:
    batch: int
    seq: int
    w_gb: float
    rc_floor: float
    t_comp_s: float
    rb_max: float          # at the highest useful R_C
    rb_at_floor: float     # at the R_C floor (max D, min R_B)

    @property
    def capacity_reachable(self) -> bool:
        """Can we get below theta_C at all?"""
        return self.rc_floor < THETA_C

    @property
    def coord_reachable(self) -> bool:
        """Can transfer ever hide behind compute?"""
        return self.rb_max >= THETA_B

    @property
    def io_reachable(self) -> bool:
        """Can we push transfer past compute? (throttling can always lower R_B)"""
        return self.rb_at_floor < THETA_B or True

    @property
    def all_three(self) -> bool:
        return self.capacity_reachable and self.coord_reachable


def spec_at(base: ModelSpec, batch: int, seq: int) -> ModelSpec:
    return ModelSpec(
        name=base.name, w_param_gb=base.w_param_gb, batch=batch, seq_len=seq,
        n_layers=base.n_layers, d_model=base.d_model,
        dtype_bytes=base.dtype_bytes, n_heads=base.n_heads, d_head=base.d_head,
    )


def t_comp_estimate(spec: ModelSpec, mfu: float, peak_flops: float) -> float:
    """Roofline estimate: 2 * N_params * tokens / (peak * MFU)."""
    n_params = spec.w_param_gb * 1e9 / spec.dtype_bytes
    tokens = spec.batch * spec.seq_len
    return (2.0 * n_params * tokens) / (peak_flops * mfu)


def analyse(base: ModelSpec, hw: HardwareProfile, batch: int, seq: int,
            mfu: float, peak_flops: float) -> GridPoint:
    spec = spec_at(base, batch, seq)
    w = spec.w_total_gb
    nonparam = spec.w_act_gb + spec.w_kv_gb
    rc_floor = nonparam / w
    t_comp = t_comp_estimate(spec, mfu, peak_flops)

    def rb_at(r_c: float) -> float:
        d_gb = spec.w_param_gb - max(0.0, r_c * w - nonparam)
        if d_gb <= 0:
            return float("inf")     # nothing to stream: no transfer pressure
        return hw.b_slow_gbs * t_comp / d_gb

    # Highest R_C that still streams something: leave one layer's worth out.
    per_layer = spec.w_param_gb / spec.n_layers
    rc_hi = (spec.w_param_gb - per_layer + nonparam) / w
    return GridPoint(
        batch=batch, seq=seq, w_gb=w, rc_floor=rc_floor, t_comp_s=t_comp,
        rb_max=rb_at(rc_hi), rb_at_floor=rb_at(rc_floor),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan a reachable regime grid")
    ap.add_argument("--platform", default="A100-80GB")
    ap.add_argument("--mfu", type=float, default=0.40,
                    help="assumed model FLOPs utilisation for the T_comp estimate")
    ap.add_argument("--peak-flops", type=float, default=A100_PEAK_FLOPS)
    args = ap.parse_args()

    hw = next((p for p in ALL_PLATFORMS if p.name == args.platform), None)
    if hw is None:
        ap.error(f"unknown platform {args.platform!r}")

    print("=" * 78)
    print(f"Reachable regime grid — {LLAMA3_8B.name} on {hw.name}")
    print(f"  B_slow = {hw.b_slow_gbs} GB/s   "
          f"T_comp: roofline estimate at MFU = {args.mfu:.0%}")
    print(f"  theta_C = {THETA_C}   theta_B = {THETA_B} (repaired: "
          f"R_B = T_comp/T_transfer)")
    print("=" * 78)
    print(f"{'batch':>5} {'seq':>6} {'W (GB)':>8} {'R_C floor':>10} "
          f"{'T_comp':>9} {'R_B max':>9}  {'capacity?':>10} {'coord?':>8}")
    print("-" * 78)

    viable = []
    for batch in (1, 2, 4, 8, 16):
        for seq in (512, 1024, 2048, 4096):
            g = analyse(LLAMA3_8B, hw, batch, seq, args.mfu, args.peak_flops)
            cap = "YES" if g.capacity_reachable else "no"
            crd = "YES" if g.coord_reachable else "no"
            flag = "  <== all three" if g.all_three else ""
            print(f"{batch:>5} {seq:>6} {g.w_gb:>8.1f} {g.rc_floor:>10.3f} "
                  f"{g.t_comp_s * 1e3:>8.0f}ms {g.rb_max:>9.2f}  "
                  f"{cap:>10} {crd:>8}{flag}")
            if g.all_three:
                viable.append(g)

    print("-" * 78)
    if viable:
        print(f"\n{len(viable)} configuration(s) admit all three regimes.")
        # A good probe point needs room on BOTH sides of BOTH boundaries, and a
        # step short enough that a 14x14 grid finishes. Maximising R_B headroom
        # alone picks configs whose R_C floor sits right against theta_C.
        def score(g: GridPoint) -> tuple:
            room_c = THETA_C - g.rc_floor          # span below theta_C to sweep
            room_b = min(g.rb_max, 8.0)            # headroom, saturating
            return (room_c > 0.15 and room_b > 2.0, room_c, -g.t_comp_s)
        best = max(viable, key=score)
        print(f"Recommended probe point: batch={best.batch}, seq={best.seq}  "
              f"(W = {best.w_gb:.1f} GB)")
        print(f"  R_C: floor {best.rc_floor:.3f} -> 1.00, crossing theta_C="
              f"{THETA_C} with {THETA_C - best.rc_floor:.2f} of span below it")
        print(f"  R_B: throttle B_slow down from {best.rb_max:.1f} through "
              f"theta_B = 1.0 (est. T_comp = {best.t_comp_s * 1e3:.0f} ms)")
        print(f"  At the R_C floor, D = {LLAMA3_8B.w_param_gb:.0f} GB streams "
              f"per step and R_B = {best.rb_at_floor:.2f} — I/O-limited without "
              f"any throttling.")
        print("\nCAUTION: R_C and R_B are NOT independently variable by fiat.")
        print("  D grows as R_C falls, so R_B = B*T_comp/D is coupled to R_C.")
        print("  They decouple only via the throttle, and only downward: at each")
        print("  R_C the reachable set is R_B <= B_hw*T_comp/D(R_C). The probe")
        print("  region is a triangle, not the rectangle the sweep grid assumes.")
    else:
        print("\nNo configuration admits all three regimes on this platform.")
        print("The capacity-limited regime needs batch x seq small enough that")
        print("activations+KV stay under the parameter footprint; the")
        print("coordination-dominated regime needs compute long enough to hide")
        print("D bytes of transfer. On PCIe these pull in opposite directions.")

    print("\nNOTE: R_C floor is exact (tensor shapes). R_B is a roofline")
    print("      prediction and must be confirmed by measurement.")


if __name__ == "__main__":
    main()
