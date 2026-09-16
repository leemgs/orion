#!/usr/bin/env python3
"""Preregistered analysis for the decisive two-ratio experiment.

This implements the analysis **locked** in ``paper/submission/preregistration.md``
so that, once real records exist, the confirmatory test runs exactly as
specified with no post-hoc freedom. It computes:

* H3 -- a-priori best-policy prediction accuracy on the held-out split, with a
  point-level bootstrap CI, compared against the best fixed (label-blind) policy.
* H4 -- ranking-change detection for policy pairs across a predicted coordinate.
* H1 -- residency effect-size ratio with a bootstrap CI.

It reports numbers; it does not decide acceptance and it never fabricates data.
Records tagged ``"source": "simulated"`` are rejected as evidence, matching the
repository's measured/simulated separation. This module runs on data that does
not yet exist in the repository; it is validated by ``tests/test_analyze_prereg.py``
against a small labelled fixture.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- Preregistered a-priori label -> predicted-best-policy map (a HYPOTHESIS
# under test by H3, not a validated result). Overridable via --policy-map so the
# map itself is declared before data collection, never tuned on the outcome.
DEFAULT_POLICY_MAP = {
    "capacity-limited": "paged-KV-offload",
    "I/O-limited": "layer-pinned",
    "coordination-dominated": "no-offload",
}

THETA_C = 0.5
THETA_B = 1.0
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 10000


class SimulatedRecordError(ValueError):
    """Raised when simulated records are offered as empirical evidence."""


def load_records(path: Path, split: str | None = "heldout") -> list[dict]:
    """Load JSONL records, rejecting simulated ones and (optionally) filtering
    to a split. Confirmatory analysis uses the held-out split only."""
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("source") != "measured":
            raise SimulatedRecordError(
                f"record with source={rec.get('source')!r} cannot be used as "
                "evidence; only source='measured' is admissible"
            )
        if split is not None and rec.get("split") != split:
            continue
        records.append(rec)
    return records


def _point_key(rec: dict) -> tuple:
    """Identity of an operating point (a grid cell), independent of policy/run."""
    return (rec["machine_id"], rec["model"], round(rec["R_C"], 4),
            round(rec["R_B"], 4))


def _mean_latency_by_policy(recs: list[dict]) -> dict[str, float]:
    by_policy: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_policy[r["policy"]].append(r["T_total_s"])
    return {p: float(np.mean(v)) for p, v in by_policy.items()}


def _points(records: list[dict]) -> dict[tuple, dict]:
    """Group records into points, each carrying its label and per-policy means."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[_point_key(r)].append(r)
    points = {}
    for key, recs in grouped.items():
        means = _mean_latency_by_policy(recs)
        points[key] = {
            "label": recs[0]["label"],
            "R_C": recs[0]["R_C"],
            "R_B": recs[0]["R_B"],
            "policy_latency": means,
            "observed_best": min(means, key=means.get),
        }
    return points


def _accuracy(points: dict, policy_map: dict) -> float:
    hits = sum(
        1 for p in points.values()
        if policy_map.get(p["label"]) == p["observed_best"]
    )
    return hits / len(points) if points else float("nan")


def _best_fixed_policy_accuracy(points: dict) -> tuple[str, float]:
    """Label-blind baseline: the single policy that is best at the most points."""
    wins: dict[str, int] = defaultdict(int)
    for p in points.values():
        wins[p["observed_best"]] += 1
    best = max(wins, key=wins.get)
    return best, wins[best] / len(points)


def _bootstrap_accuracy_ci(points: dict, policy_map: dict,
                           seed: int = BOOTSTRAP_SEED,
                           resamples: int = BOOTSTRAP_RESAMPLES,
                           alpha: float = 0.05) -> tuple[float, float]:
    keys = list(points.keys())
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    accs = np.empty(resamples)
    for b in range(resamples):
        pick = rng.choice(idx, size=idx.size, replace=True)
        hits = sum(
            1 for i in pick
            if policy_map.get(points[keys[i]]["label"])
            == points[keys[i]]["observed_best"]
        )
        accs[b] = hits / idx.size
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def h3_prediction(points: dict, policy_map: dict) -> dict:
    acc = _accuracy(points, policy_map)
    lo, hi = _bootstrap_accuracy_ci(points, policy_map)
    baseline_policy, baseline_acc = _best_fixed_policy_accuracy(points)
    return {
        "n_points": len(points),
        "accuracy": acc,
        "accuracy_ci95": [lo, hi],
        "baseline_policy": baseline_policy,
        "baseline_accuracy": baseline_acc,
        # Primary preregistered success criterion for H3.
        "beats_baseline": lo > baseline_acc,
    }


def h4_ranking_change(points: dict, coordinate: str = "R_C",
                      seed: int = BOOTSTRAP_SEED,
                      resamples: int = BOOTSTRAP_RESAMPLES) -> list[dict]:
    """Detect a mean-latency ranking sign change for policy pairs across a
    predicted coordinate, with non-overlapping per-side bootstrap CIs."""
    theta = THETA_C if coordinate == "R_C" else THETA_B
    below = [p for p in points.values() if p[coordinate] < theta]
    above = [p for p in points.values() if p[coordinate] >= theta]
    policies = sorted({pol for p in points.values() for pol in p["policy_latency"]})
    rng = np.random.default_rng(seed)

    def side_diff_ci(side: list[dict], a: str, b: str):
        diffs = [p["policy_latency"][a] - p["policy_latency"][b]
                 for p in side
                 if a in p["policy_latency"] and b in p["policy_latency"]]
        if not diffs:
            return None
        arr = np.asarray(diffs)
        boot = np.empty(resamples)
        for i in range(resamples):
            boot[i] = rng.choice(arr, size=arr.size, replace=True).mean()
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return float(arr.mean()), float(lo), float(hi)

    findings = []
    for i, a in enumerate(policies):
        for b in policies[i + 1:]:
            lo_side, hi_side = side_diff_ci(below, a, b), side_diff_ci(above, a, b)
            if lo_side is None or hi_side is None:
                continue
            m_lo, l_lo, h_lo = lo_side
            m_hi, l_hi, h_hi = hi_side
            sign_change = (m_lo < 0) != (m_hi < 0)
            ci_separated = h_lo < l_hi or h_hi < l_lo
            findings.append({
                "coordinate": coordinate, "pair": [a, b],
                "below": {"mean_diff": m_lo, "ci95": [l_lo, h_lo]},
                "above": {"mean_diff": m_hi, "ci95": [l_hi, h_hi]},
                "ranking_change": bool(sign_change and ci_separated),
            })
    return findings


def h1_residency_ratio(points: dict, seed: int = BOOTSTRAP_SEED,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Best-policy latency ratio: capacity-limited points over resident points."""
    def best_lat(p):
        return p["policy_latency"][p["observed_best"]]
    cap = [best_lat(p) for p in points.values() if p["R_C"] < THETA_C]
    res = [best_lat(p) for p in points.values() if p["R_C"] >= 1.0]
    if not cap or not res:
        return {"ratio": None, "note": "insufficient points on one side"}
    rng = np.random.default_rng(seed)
    cap_a, res_a = np.asarray(cap), np.asarray(res)
    boot = np.empty(resamples)
    for i in range(resamples):
        c = rng.choice(cap_a, size=cap_a.size, replace=True).mean()
        r = rng.choice(res_a, size=res_a.size, replace=True).mean()
        boot[i] = c / r
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"ratio": float(cap_a.mean() / res_a.mean()),
            "ratio_ci95": [float(lo), float(hi)],
            "n_capacity": len(cap), "n_resident": len(res)}


def analyze(path: Path, policy_map: dict | None = None,
            split: str | None = "heldout") -> dict:
    policy_map = policy_map or DEFAULT_POLICY_MAP
    records = load_records(path, split=split)
    if not records:
        raise ValueError(f"no admissible records in {path} for split={split!r}")
    points = _points(records)
    return {
        "split": split,
        "policy_map": policy_map,
        "H3_prediction": h3_prediction(points, policy_map),
        "H4_ranking_change_RC": h4_ranking_change(points, "R_C"),
        "H4_ranking_change_RB": h4_ranking_change(points, "R_B"),
        "H1_residency_ratio": h1_residency_ratio(points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSONL of measured records")
    parser.add_argument("--split", default="heldout",
                        help="split to analyse (default: heldout; 'all' = no filter)")
    parser.add_argument("--policy-map", type=Path, default=None,
                        help="JSON file overriding the a-priori label->policy map")
    args = parser.parse_args()
    policy_map = (json.loads(args.policy_map.read_text())
                  if args.policy_map else None)
    split = None if args.split == "all" else args.split
    report = analyze(args.records, policy_map=policy_map, split=split)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
