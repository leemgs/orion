"""Validate the vLLM strategy-inversion harness's pure-Python logic.

No GPU or vLLM is exercised here: these tests cover record assembly, the locked
schema, the policy/engine-kwargs mapping, and the dry-run/evidence separation.
Any ``source="measured"`` row constructed below is synthetic and used only to
check the arithmetic and the analysis wiring; it is never presented as evidence
(the same convention as ``test_analyze_prereg.py``).
"""
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments.prereg_backends import PointMeasurement  # noqa: E402
from experiments.prereg_harvest import REQUIRED_FIELDS, validate_record  # noqa: E402
from experiments import vllm_strategy_inversion as inv  # noqa: E402


def test_dry_run_rows_are_schema_valid_and_not_evidence(tmp_path):
    out = tmp_path / "dry.jsonl"
    n = inv.run(model="m", machine_id="mach", arch="A100-80GB",
                points=inv.DEFAULT_POINTS, runs=2, output=out, dry_run=True)
    assert n == len(inv.DEFAULT_POINTS) * len(inv.POLICIES) * 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    for r in rows:
        validate_record(r)                      # locked schema
        assert REQUIRED_FIELDS <= r.keys()
        assert r["source"] == "dryrun"          # never 'measured'
        assert r["framework"] == "vllm"
        assert r["policy"] in inv.POLICIES


def test_make_record_computes_ratios_from_measurement():
    # Synthetic measurement: C_fast=30, W=100 -> R_C=0.3; T_comp=0.6,
    # T_transfer=1.0 -> R_B=0.6 -> capacity-limited (R_C<0.5).
    meas = PointMeasurement(
        t_comp_s=0.6, t_transfer_s=1.0, t_total_s=2.0,
        c_fast_bytes=30.0, w_bytes=100.0, d_bytes=70.0, d_nr_bytes=70.0,
        b_slow_bytes_per_s=70.0, windows=10, provenance="synthetic test")
    rec = inv.make_record(inv.DEFAULT_POINTS[0], "no-offload", 0,
                          "mach", "A100-80GB", "m", meas, "measured")
    assert rec["R_C"] == pytest.approx(0.3)
    assert rec["R_B"] == pytest.approx(0.6)
    assert rec["label"] == "capacity-limited"
    validate_record(rec)


def test_make_record_no_transfer_is_coordination_dominated():
    # Fully resident: C_fast>=W so R_C>=1, and no transfer (t_transfer tiny,
    # t_comp>0) -> R_B very large -> coordination-dominated.
    meas = PointMeasurement(
        t_comp_s=0.5, t_transfer_s=1e-12, t_total_s=0.5,
        c_fast_bytes=200.0, w_bytes=100.0, d_bytes=0.0, d_nr_bytes=0.0,
        b_slow_bytes_per_s=0.0, windows=10, provenance="synthetic resident")
    rec = inv.make_record(inv.DEFAULT_POINTS[2], "no-offload", 0,
                          "mach", "A100-80GB", "m", meas, "measured")
    assert rec["R_C"] >= 1.0
    assert rec["R_B"] > 1.0
    assert rec["label"] == "coordination-dominated"


class _LlamaCfg:
    num_hidden_layers = 32
    num_attention_heads = 32
    hidden_size = 4096
    num_key_value_heads = 8
    head_dim = 128
    vocab_size = 128256
    max_position_embeddings = 8192


def test_policy_engine_kwargs_distinct_offload():
    weights = 16e9  # 16 GB
    pt = inv.DEFAULT_POINTS[0]
    cfg = _LlamaCfg()
    k_noff = inv.policy_engine_kwargs("no-offload", pt, weights, cfg=cfg)
    k_full = inv.policy_engine_kwargs("full-offload", pt, weights, cfg=cfg)
    k_kv = inv.policy_engine_kwargs("paged-KV-offload", pt, weights, cfg=cfg)
    k_pin = inv.policy_engine_kwargs("layer-pinned", pt, weights, cfg=cfg)
    # no-offload: nothing offloaded, no KV connector
    assert k_noff["cpu_offload_gb"] == 0.0 and "kv_transfer_config" not in k_noff
    # weight offload ordering: full > pinned > none
    assert k_full["cpu_offload_gb"] > k_pin["cpu_offload_gb"] > 0.0
    # KV offload goes through the V1 offloading connector by default, not
    # swap_space (which the V1 engine ignores)
    for k in (k_kv, k_full):
        assert k["kv_transfer_config"]["kv_connector"] == "OffloadingConnector"
        assert k["kv_transfer_config"]["kv_connector_extra_config"]["num_cpu_blocks"] > 0
        assert k["swap_space"] == 0.0
    assert k_kv["cpu_offload_gb"] == 0.0  # weights stay resident


def test_policy_engine_kwargs_legacy_swap_mode_and_budget():
    pt = inv.DEFAULT_POINTS[0]
    k = inv.policy_engine_kwargs("paged-KV-offload", pt, 16e9, kv_mode="swap",
                                 budget=0.9)
    assert k["swap_space"] > 0.0 and "kv_transfer_config" not in k
    assert k["gpu_memory_utilization"] == 0.9
    with pytest.raises(ValueError):
        inv.policy_engine_kwargs("no-offload", pt, 16e9, kv_mode="bogus")
    with pytest.raises(ValueError):  # connector sizing needs the config
        inv.policy_engine_kwargs("paged-KV-offload", pt, 16e9)


def test_kv_connector_extra_override():
    spec = inv.kv_connector_spec(_LlamaCfg(), extra={"cpu_bytes_to_use": 1})
    assert spec["kv_connector_extra_config"] == {"cpu_bytes_to_use": 1}


def test_kv_cache_bytes_formula():
    class Cfg:
        num_hidden_layers = 32
        num_attention_heads = 32
        hidden_size = 4096
        num_key_value_heads = 8
        head_dim = 128
    # 2 * 32 * 8 * 128 * 2 bytes * (batch*(seq+out))
    b, s, o = 2, 100, 10
    expected = 2 * 32 * 8 * 128 * inv.DTYPE_BYTES * (b * (s + o))
    assert inv.kv_cache_bytes(Cfg(), b, s, o) == pytest.approx(expected)


def test_summary_detects_inversion_on_synthetic_measured(tmp_path, capsys):
    """Feed synthetic *measured* rows (for arithmetic only) through the real
    analysis and confirm an engineered rank inversion is reported."""
    # Two points: capacity-limited (R_C=0.3) where paged-KV-offload wins, and
    # coordination-dominated (R_C=1.0, R_B=2.0) where no-offload wins -> the
    # no-offload vs paged-KV-offload ranking inverts across R_C=0.5.
    rows = []
    specs = [
        (0.30, 0.70, "capacity-limited",
         {"no-offload": 0.060, "paged-KV-offload": 0.040, "layer-pinned": 0.050,
          "full-offload": 0.045}),
        (1.00, 2.00, "coordination-dominated",
         {"no-offload": 0.030, "paged-KV-offload": 0.055, "layer-pinned": 0.045,
          "full-offload": 0.050}),
    ]
    for rc, rb, label, lats in specs:
        for policy, lat in lats.items():
            for run in range(3):
                rows.append({
                    "machine_id": "mach", "arch": "A100-80GB", "model": "m",
                    "framework": "vllm", "policy": policy, "split": "all",
                    "R_C": rc, "R_B": rb, "label": label,
                    "C_fast_bytes": 1.0, "W_bytes": 1.0, "D_bytes": 1.0,
                    "D_nr_bytes": 1.0, "B_slow_bytes_per_s": 1.0,
                    "T_comp_s": 1.0, "T_transfer_s": 1.0, "T_total_s": lat,
                    "run_idx": run, "windows": 10, "source": "measured",
                    "provenance": "synthetic arithmetic fixture, not evidence",
                })
    out = tmp_path / "meas.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    inv.summarize(out)
    printed = capsys.readouterr().out
    assert "Strategy-inversion summary" in printed
    assert "ranking inversion across R_C" in printed
    # no-offload vs paged-KV-offload should be flagged as inverting.
    assert "no-offload" in printed and "paged-KV-offload" in printed
