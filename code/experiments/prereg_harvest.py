#!/usr/bin/env python3
"""Harvest measurement records for the preregistered confirmatory experiment.

This is the *plumbing* for the campaign specified in
``paper/submission/preregistration.md``: it builds the preregistered operating-
point grid, assigns the train/held-out split deterministically **before** any
latency is seen, emits records in the locked JSONL schema, and validates them.
Its output is consumed directly by ``analyze_prereg.py``.

It never fabricates evidence. A ``--dry-run`` emits schema-correct rows tagged
``source="dryrun"`` (with placeholder timings) so the end-to-end pipeline and
the analyser's rejection of non-measured rows can be validated without hardware.
Real records (``source="measured"``) are produced only by wiring an actual
serving-stack measurement into ``measure_operating_point`` -- the integration
seam below. There is deliberately no built-in path that turns synthetic numbers
into ``source="measured"``.

Wiring a backend
----------------
``measure_operating_point`` must return a :class:`PointMeasurement` from a real
run: time compute and host-to-device transfer separately (CUDA events, as in
``experiments/cuda_backend.py``), realise the target residency by capping device
memory / offload fraction, and realise the target overlap by scaling compute.
Retain per-window and per-run raw records -- their absence in the submitted
accelerator runs is exactly why those runs carry no confidence interval.

Usage
-----
    # validate the pipeline without hardware
    python experiments/prereg_harvest.py --dry-run -o records_dryrun.jsonl
    python experiments/analyze_prereg.py records_dryrun.jsonl   # -> rejects dryrun

    # on real hardware, after wiring measure_operating_point:
    python experiments/prereg_harvest.py --backend torch-cuda -o records.jsonl
    python experiments/analyze_prereg.py records.jsonl --split heldout
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from orion.config import Regime  # noqa: E402
from orion.ratios import classify_regime  # noqa: E402
# PointMeasurement and the serving-stack backends live in prereg_backends so the
# measurement seam can be filled per framework without touching this plumbing.
from experiments.prereg_backends import PointMeasurement, get_backend  # noqa: E402,F401

# Preregistered label strings (must match preregistration.md and analyze_prereg).
REGIME_LABEL = {
    Regime.CAPACITY_LIMITED: "capacity-limited",
    Regime.IO_LIMITED: "I/O-limited",
    Regime.COORDINATION_DOMINATED: "coordination-dominated",
}

# Preregistered fixed policy set (see preregistration.md).
POLICIES = ["no-offload", "layer-pinned", "paged-KV-offload", "full-offload"]

# Default grid bracketing theta_C = 0.5 and theta_B = 1.0. A template: the real
# campaign fixes its own grid in advance and records it alongside the data.
DEFAULT_RC_GRID = [0.20, 0.35, 0.45, 0.50, 0.60, 0.80, 1.00, 1.50]
DEFAULT_RB_GRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.30, 2.00]

SPLIT_SEED = 20260818
HELDOUT_FRACTION = 0.40
RUNS_PER_POINT = 5   # preregistered minimum independent runs per (point, policy)


@dataclass(frozen=True)
class GridPoint:
    machine_id: str
    arch: str
    model: str
    framework: str
    r_c: float
    r_b: float

    @property
    def label(self) -> str:
        return REGIME_LABEL[classify_regime(self.r_c, self.r_b)]


def build_grid(machines: list[dict], models: list[dict],
               rc_grid: list[float] = DEFAULT_RC_GRID,
               rb_grid: list[float] = DEFAULT_RB_GRID) -> list[GridPoint]:
    points: list[GridPoint] = []
    for m in machines:
        for model in models:
            for rc in rc_grid:
                for rb in rb_grid:
                    points.append(GridPoint(
                        machine_id=m["machine_id"], arch=m["arch"],
                        model=model["model"], framework=model["framework"],
                        r_c=rc, r_b=rb))
    return points


def assign_splits(points: list[GridPoint], seed: int = SPLIT_SEED,
                  heldout_fraction: float = HELDOUT_FRACTION) -> dict[GridPoint, str]:
    """Deterministic train/held-out partition fixed before any latency is seen.

    Uses the standard library only so the assignment is reproducible from the
    seed alone and independent of NumPy version.
    """
    import random
    rng = random.Random(seed)
    order = list(range(len(points)))
    rng.shuffle(order)
    n_heldout = round(len(points) * heldout_fraction)
    heldout = set(order[:n_heldout])
    return {p: ("heldout" if i in heldout else "train")
            for i, p in enumerate(points)}


def _record(point: GridPoint, policy: str, split: str, run_idx: int,
            meas: PointMeasurement, source: str) -> dict:
    """Assemble one JSONL record in the locked schema."""
    return {
        "machine_id": point.machine_id, "arch": point.arch,
        "model": point.model, "framework": point.framework,
        "policy": policy, "split": split,
        "R_C": point.r_c, "R_B": point.r_b, "label": point.label,
        "C_fast_bytes": meas.c_fast_bytes, "W_bytes": meas.w_bytes,
        "D_bytes": meas.d_bytes, "D_nr_bytes": meas.d_nr_bytes,
        "B_slow_bytes_per_s": meas.b_slow_bytes_per_s,
        "T_comp_s": meas.t_comp_s, "T_transfer_s": meas.t_transfer_s,
        "T_total_s": meas.t_total_s,
        "run_idx": run_idx, "windows": meas.windows,
        "source": source,
        "provenance": meas.provenance,
    }


def measure_operating_point(point: GridPoint, policy: str, run_idx: int,
                            backend: str) -> PointMeasurement:
    """INTEGRATION SEAM -- dispatch to a serving-stack backend or refuse.

    Backends live in ``prereg_backends.py`` (vLLM / DeepSpeed / FlexGen /
    torch-cuda); each raises ``NotImplementedError`` until its real-measurement
    TODO is filled, so no synthetic value can ever be tagged ``source="measured"``.
    Use ``--dry-run`` to validate the pipeline without hardware.
    """
    return get_backend(backend).measure(point, policy, run_idx)


def _dry_run_measurement(point: GridPoint, policy: str) -> PointMeasurement:
    """Schema-correct placeholder for pipeline validation. NOT evidence."""
    return PointMeasurement(
        t_comp_s=0.0, t_transfer_s=0.0, t_total_s=0.0,
        c_fast_bytes=0.0, w_bytes=0.0, d_bytes=0.0, d_nr_bytes=0.0,
        b_slow_bytes_per_s=0.0, windows=0,
        provenance=("DRY RUN -- placeholder, not a measurement. Wire "
                    "measure_operating_point for real data."),
    )


def harvest(points: list[GridPoint], splits: dict[GridPoint, str],
            backend: str, runs_per_point: int = RUNS_PER_POINT,
            dry_run: bool = False):
    """Yield records for every (point, policy, run)."""
    for point in points:
        split = splits[point]
        for policy in POLICIES:
            for run_idx in range(runs_per_point):
                if dry_run:
                    meas = _dry_run_measurement(point, policy)
                    source = "dryrun"
                else:
                    meas = measure_operating_point(point, policy, run_idx, backend)
                    source = "measured"
                yield _record(point, policy, split, run_idx, meas, source)


REQUIRED_FIELDS = {
    "machine_id", "arch", "model", "framework", "policy", "split",
    "R_C", "R_B", "label", "C_fast_bytes", "W_bytes", "D_bytes", "D_nr_bytes",
    "B_slow_bytes_per_s", "T_comp_s", "T_transfer_s", "T_total_s",
    "run_idx", "windows", "source", "provenance",
}


def validate_record(rec: dict) -> None:
    missing = REQUIRED_FIELDS - rec.keys()
    if missing:
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if rec["label"] not in set(REGIME_LABEL.values()):
        raise ValueError(f"unknown label {rec['label']!r}")


def _default_targets() -> tuple[list[dict], list[dict]]:
    machines = [
        {"machine_id": "a100-node-1", "arch": "A100-80GB"},
        {"machine_id": "h100-node-1", "arch": "H100-80GB"},
    ]
    models = [
        {"model": "llama-3-8b", "framework": "vllm"},
        {"model": "retrieval-rag", "framework": "flexgen"},
    ]
    return machines, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSONL path")
    parser.add_argument("--backend", default="torch-cuda",
                        help="measurement backend id (must be wired)")
    parser.add_argument("--dry-run", action="store_true",
                        help="emit source='dryrun' placeholders (not evidence)")
    parser.add_argument("--runs", type=int, default=RUNS_PER_POINT)
    args = parser.parse_args()

    machines, models = _default_targets()
    points = build_grid(machines, models)
    splits = assign_splits(points)

    n = 0
    with args.output.open("w") as fh:
        for rec in harvest(points, splits, args.backend,
                           runs_per_point=args.runs, dry_run=args.dry_run):
            validate_record(rec)
            fh.write(json.dumps(rec) + "\n")
            n += 1
    kind = "dry-run" if args.dry_run else "measured"
    print(f"wrote {n} {kind} records for {len(points)} points to {args.output}")
    if args.dry_run:
        print("NOTE: dry-run rows are placeholders; analyze_prereg.py rejects them.")


if __name__ == "__main__":
    main()
