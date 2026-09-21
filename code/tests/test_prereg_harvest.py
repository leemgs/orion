"""Validate the preregistered harvest plumbing and its handoff to the analyser.

No hardware and no fabricated evidence: dry-run records are placeholders, and
the one place we synthesise ``source="measured"`` rows is this test, to confirm
the schema flows into analyze_prereg. The harness itself refuses to emit
measured rows without a wired backend.
"""
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments import prereg_harvest as ph  # noqa: E402
from experiments.analyze_prereg import (  # noqa: E402
    SimulatedRecordError,
    analyze,
    load_records,
)


def _targets():
    machines = [{"machine_id": "a100-node-1", "arch": "A100-80GB"}]
    models = [{"model": "llama-3-8b", "framework": "vllm"}]
    return machines, models


def test_labels_follow_the_classifier():
    machines, models = _targets()
    pts = ph.build_grid(machines, models, rc_grid=[0.2, 0.6], rb_grid=[0.7, 1.5])
    labels = {(p.r_c, p.r_b): p.label for p in pts}
    assert labels[(0.2, 0.7)] == "capacity-limited"      # R_C < 0.5
    assert labels[(0.6, 0.7)] == "I/O-limited"           # R_C >= 0.5, R_B < 1
    assert labels[(0.6, 1.5)] == "coordination-dominated"  # both pass


def test_split_is_deterministic_and_fixed_fraction():
    machines, models = _targets()
    pts = ph.build_grid(machines, models)
    s1 = ph.assign_splits(pts)
    s2 = ph.assign_splits(pts)
    assert s1 == s2  # reproducible from the seed alone
    n_heldout = sum(1 for v in s1.values() if v == "heldout")
    assert n_heldout == round(len(pts) * ph.HELDOUT_FRACTION)


def test_measure_operating_point_refuses_without_backend():
    machines, models = _targets()
    p = ph.build_grid(machines, models, rc_grid=[0.6], rb_grid=[1.5])[0]
    with pytest.raises(NotImplementedError):
        ph.measure_operating_point(p, "no-offload", 0, "torch-cuda")


def test_dry_run_records_validate_but_are_rejected_as_evidence(tmp_path):
    machines, models = _targets()
    pts = ph.build_grid(machines, models, rc_grid=[0.2, 0.6], rb_grid=[0.7, 1.5])
    splits = ph.assign_splits(pts)
    out = tmp_path / "dry.jsonl"
    with out.open("w") as fh:
        for rec in ph.harvest(pts, splits, "torch-cuda", runs_per_point=2,
                              dry_run=True):
            ph.validate_record(rec)  # schema-complete
            assert rec["source"] == "dryrun"
            fh.write(json.dumps(rec) + "\n")
    # The analyser must refuse dry-run rows as evidence.
    with pytest.raises(SimulatedRecordError):
        load_records(out)


def test_measured_schema_flows_into_analyzer(tmp_path):
    """Convert harvest rows to source='measured' HERE (test-only) and confirm
    they are accepted and analysable. Latencies are chosen so the a-priori map
    predicts every held-out point."""
    machines, models = _targets()
    pts = ph.build_grid(machines, models, rc_grid=[0.2, 0.6, 1.5],
                        rb_grid=[0.7, 1.5])
    # Force all points into the held-out split for a self-contained analysis.
    splits = {p: "heldout" for p in pts}
    best_by_label = {
        "capacity-limited": "paged-KV-offload",
        "I/O-limited": "layer-pinned",
        "coordination-dominated": "no-offload",
    }
    out = tmp_path / "measured.jsonl"
    with out.open("w") as fh:
        for rec in ph.harvest(pts, splits, "torch-cuda", runs_per_point=2,
                              dry_run=True):
            winner = best_by_label[rec["label"]]
            rec["T_total_s"] = 0.02 if rec["policy"] == winner else 0.05
            rec["source"] = "measured"
            rec["provenance"] = "test-only synthetic; not repository evidence"
            ph.validate_record(rec)
            fh.write(json.dumps(rec) + "\n")

    recs = load_records(out, split="heldout")
    assert recs and all(r["source"] == "measured" for r in recs)
    report = analyze(out, split="heldout")
    assert report["H3_prediction"]["accuracy"] == pytest.approx(1.0)
    assert report["H3_prediction"]["beats_baseline"] is True
