"""Validate the preregistered analysis against a small labelled fixture.

The fixture is synthetic and used only to check the analysis arithmetic; it is
never presented as evidence. It is constructed so the a-priori policy map
predicts every held-out point (accuracy 1.0) and one policy pair reverses its
latency ranking across R_C = 0.5.
"""
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments.analyze_prereg import (  # noqa: E402
    DEFAULT_POLICY_MAP,
    SimulatedRecordError,
    analyze,
    load_records,
)

# T_total per (point, policy). Points chosen so observed_best matches the
# DEFAULT_POLICY_MAP prediction for each label.
POINTS = [
    # (machine, model, R_C, R_B, label, {policy: latency})
    ("m1", "llama", 0.30, 0.70, "capacity-limited",
     {"paged-KV-offload": 0.040, "layer-pinned": 0.050, "no-offload": 0.060}),
    ("m2", "llama", 0.20, 0.60, "capacity-limited",
     {"paged-KV-offload": 0.045, "layer-pinned": 0.055, "no-offload": 0.065}),
    ("m1", "llama", 0.70, 0.70, "I/O-limited",
     {"layer-pinned": 0.030, "no-offload": 0.040, "paged-KV-offload": 0.050}),
    ("m2", "llama", 1.00, 2.00, "coordination-dominated",
     {"no-offload": 0.035, "layer-pinned": 0.045, "paged-KV-offload": 0.055}),
]


def _write_fixture(path: Path, *, include_simulated=False, include_train=False):
    lines = []
    for machine, model, rc, rb, label, lats in POINTS:
        for policy, lat in lats.items():
            for run in range(3):  # identical runs -> exact means
                lines.append({
                    "machine_id": machine, "arch": "A100-80GB", "model": model,
                    "framework": "vllm", "policy": policy, "split": "heldout",
                    "R_C": rc, "R_B": rb, "label": label,
                    "T_total_s": lat, "run_idx": run, "source": "measured",
                })
    if include_train:
        lines.append({
            "machine_id": "m1", "arch": "A100-80GB", "model": "llama",
            "framework": "vllm", "policy": "no-offload", "split": "train",
            "R_C": 0.9, "R_B": 1.5, "label": "coordination-dominated",
            "T_total_s": 0.01, "run_idx": 0, "source": "measured",
        })
    if include_simulated:
        lines.append({
            "machine_id": "sim", "arch": "sim", "model": "llama",
            "framework": "sim", "policy": "no-offload", "split": "heldout",
            "R_C": 0.9, "R_B": 1.5, "label": "coordination-dominated",
            "T_total_s": 0.01, "run_idx": 0, "source": "simulated",
        })
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_simulated_records_are_rejected(tmp_path):
    fixture = tmp_path / "recs.jsonl"
    _write_fixture(fixture, include_simulated=True)
    with pytest.raises(SimulatedRecordError):
        load_records(fixture)


def test_train_split_excluded_from_heldout(tmp_path):
    fixture = tmp_path / "recs.jsonl"
    _write_fixture(fixture, include_train=True)
    heldout = load_records(fixture, split="heldout")
    assert all(r["split"] == "heldout" for r in heldout)
    assert not any(r["split"] == "train" for r in heldout)


def test_h3_accuracy_and_beats_baseline(tmp_path):
    fixture = tmp_path / "recs.jsonl"
    _write_fixture(fixture)
    report = analyze(fixture)
    h3 = report["H3_prediction"]
    assert h3["n_points"] == 4
    assert h3["accuracy"] == pytest.approx(1.0)
    # Every point is a hit, so the point-bootstrap is degenerate at 1.0.
    assert h3["accuracy_ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]
    # Best fixed policy wins only 2/4 points.
    assert h3["baseline_accuracy"] == pytest.approx(0.5)
    assert h3["beats_baseline"] is True
    assert report["policy_map"] == DEFAULT_POLICY_MAP


def test_h4_detects_ranking_change_across_rc(tmp_path):
    fixture = tmp_path / "recs.jsonl"
    _write_fixture(fixture)
    report = analyze(fixture)
    pko_no = [f for f in report["H4_ranking_change_RC"]
              if set(f["pair"]) == {"paged-KV-offload", "no-offload"}]
    assert pko_no and pko_no[0]["ranking_change"] is True
    # The mean latency difference reverses sign across theta_C.
    assert pko_no[0]["below"]["mean_diff"] * pko_no[0]["above"]["mean_diff"] < 0


def test_h1_residency_ratio_gt_one(tmp_path):
    fixture = tmp_path / "recs.jsonl"
    _write_fixture(fixture)
    h1 = analyze(fixture)["H1_residency_ratio"]
    assert h1["ratio"] > 1.0
    assert h1["n_capacity"] == 2 and h1["n_resident"] == 1
