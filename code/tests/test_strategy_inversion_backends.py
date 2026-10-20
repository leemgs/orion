"""Validate the shared strategy-inversion core and the DeepSpeed/FlexGen adapters.

No GPU, DeepSpeed, FlexGen or vLLM is exercised: these tests cover the pure
mapping/arithmetic code, the dry-run/evidence separation, and that missing
frameworks raise instead of producing numbers. Any ``source="measured"`` row
built here is synthetic and only checks arithmetic; it is never evidence.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments import strategy_inversion_common as common  # noqa: E402
from experiments import deepspeed_strategy_inversion as ds  # noqa: E402
from experiments import flexgen_strategy_inversion as fg  # noqa: E402
from experiments import vllm_strategy_inversion as vl  # noqa: E402
from experiments.prereg_harvest import validate_record  # noqa: E402


class LlamaCfg:
    num_hidden_layers = 32
    num_attention_heads = 32
    hidden_size = 4096
    num_key_value_heads = 8
    head_dim = 128
    vocab_size = 128256
    max_position_embeddings = 8192
    torch_dtype = "bfloat16"
    eos_token_id = [128001, 128009]
    pad_token_id = None


class OptCfg:  # OPT has no num_key_value_heads / head_dim attributes
    num_hidden_layers = 32
    num_attention_heads = 32
    hidden_size = 4096
    ffn_dim = 16384
    vocab_size = 50272
    max_position_embeddings = 2048
    torch_dtype = "float16"
    eos_token_id = 2
    pad_token_id = 1


ADAPTERS = [vl.VLLMAdapter(), ds.DeepSpeedAdapter(), fg.FlexGenAdapter()]


def _installed(*mods: str) -> bool:
    return any(importlib.util.find_spec(m) is not None for m in mods)


# These must only run where the framework is absent: with it installed (and a
# GPU) a "real" run would actually start measuring.
FRAMEWORK_MODULES = {"deepspeed": ("deepspeed",), "flexgen": ("flexllmgen", "flexgen")}


# --- common core ------------------------------------------------------------ #
@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.framework)
def test_dry_run_is_schema_valid_tagged_and_not_evidence(adapter, tmp_path):
    out = tmp_path / f"{adapter.framework}.jsonl"
    n = common.run(adapter, "m", "mach", "A100-80GB", list(adapter.default_points),
                   runs=2, output=out, dry_run=True, run_offset=3)
    assert n == len(adapter.default_points) * len(common.POLICIES) * 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    for r in rows:
        validate_record(r)
        assert r["source"] == "dryrun"
        assert r["framework"] == adapter.framework
        assert r["run_idx"] in (3, 4)            # --run-offset honoured
    from experiments.analyze_prereg import SimulatedRecordError, load_records
    with pytest.raises(SimulatedRecordError):
        load_records(out, split=None)


@pytest.mark.parametrize("adapter", [ds.DeepSpeedAdapter(), fg.FlexGenAdapter()],
                         ids=lambda a: a.framework)
def test_real_run_without_framework_raises_not_fabricates(adapter, tmp_path):
    if _installed(*FRAMEWORK_MODULES[adapter.framework]):
        pytest.skip(f"{adapter.framework} is installed; this checks the absent case")
    out = tmp_path / "real.jsonl"
    with pytest.raises(common.MeasurementUnavailable):
        common.run(adapter, "m", "mach", "A100-80GB", list(adapter.default_points),
                   runs=1, output=out, dry_run=False)
    assert not out.exists()  # nothing written


def test_demand_is_policy_invariant_and_contract_consistent():
    pt = common.OperatingPoint("p", gpu_mem_fraction=0.25, batch=4,
                               seq_len=1000, out_len=24)
    dev, b_slow, weights = 80e9, 25e9, 16e9
    d = common.demand_for(LlamaCfg(), pt, weights, dev, b_slow)
    kv = common.kv_cache_bytes(LlamaCfg(), 4, 1000, 24)
    assert d.w_bytes == pytest.approx(weights + kv)
    assert d.c_fast == pytest.approx(0.25 * dev)
    assert d.d_nr == pytest.approx(max(0.0, weights + kv - 0.25 * dev))
    assert d.t_transfer == pytest.approx(d.d_nr / b_slow)   # T_transfer = D/B_slow
    resident = common.demand_for(LlamaCfg(), pt, weights, dev * 10, b_slow)
    assert resident.d_nr == 0.0 and resident.t_transfer == 0.0


def test_kv_dims_fall_back_for_opt_configs():
    # OPT: kv_heads = attention heads, head_dim = hidden / heads
    assert common.kv_bytes_per_token(OptCfg()) == 2 * 32 * 32 * 128 * common.DTYPE_BYTES


def test_prompts_exact_length_and_identical_across_policies():
    pt = fg.FLEXGEN_POINTS[1]
    a = common.synthetic_prompt_ids(OptCfg(), pt)
    b = common.synthetic_prompt_ids(OptCfg(), pt)
    assert a == b                                   # same inputs for every policy/run
    assert len(a) == pt.batch and all(len(x) == pt.seq_len for x in a)
    assert all(100 <= t < OptCfg.vocab_size - 100 for x in a for t in x)


def test_context_check_and_flexgen_points_fit_opt():
    assert all(common.point_fits_context(OptCfg(), p) for p in fg.FLEXGEN_POINTS)
    too_long = common.OperatingPoint("x", 0.5, 1, seq_len=2048, out_len=1)
    assert not common.point_fits_context(OptCfg(), too_long)


def test_policy_spec_matches_preregistered_set():
    from experiments.prereg_harvest import POLICIES as PREREG
    from experiments.analyze_prereg import DEFAULT_POLICY_MAP
    assert set(common.POLICY_SPEC) == set(common.POLICIES) == set(PREREG)
    assert set(DEFAULT_POLICY_MAP.values()) <= set(common.POLICIES)


# --- DeepSpeed --------------------------------------------------------------- #
def test_ds_config_offload_and_persistence_by_policy():
    cfg = LlamaCfg()
    h = cfg.hidden_size
    noff = ds.ds_config_for("no-offload", cfg)["zero_optimization"]
    kv = ds.ds_config_for("paged-KV-offload", cfg)["zero_optimization"]
    pin = ds.ds_config_for("layer-pinned", cfg)["zero_optimization"]
    full = ds.ds_config_for("full-offload", cfg)["zero_optimization"]
    for z in (noff, kv, pin, full):
        assert z["stage"] == 3                          # one engine for all policies
    assert "offload_param" not in noff and "offload_param" not in kv
    assert pin["offload_param"] == {"device": "cpu", "pin_memory": True}
    assert full["offload_param"] == {"device": "cpu", "pin_memory": True}
    # layer-pinned keeps every hidden x hidden projection resident; full does not
    assert pin["stage3_param_persistence_threshold"] > h * h
    assert full["stage3_param_persistence_threshold"] < h * h


def test_ds_dtype_and_generate_kwargs():
    cfg = LlamaCfg()
    conf = ds.ds_config_for("no-offload", cfg)
    assert conf["bf16"]["enabled"] and not conf["fp16"]["enabled"]
    assert ds.ds_config_for("no-offload", OptCfg())["fp16"]["enabled"]
    pt = common.DEFAULT_POINTS[0]
    g_res = ds.generate_kwargs_for("no-offload", cfg, pt)
    g_kv = ds.generate_kwargs_for("paged-KV-offload", cfg, pt)
    assert g_res["max_new_tokens"] == g_res["min_new_tokens"] == pt.out_len
    assert g_res["pad_token_id"] == 128001            # first eos when no pad token
    assert "cache_implementation" not in g_res
    assert g_kv["cache_implementation"] == "offloaded"
    assert ds.generate_kwargs_for("full-offload", OptCfg(), pt)["pad_token_id"] == 1


# --- FlexGen ----------------------------------------------------------------- #
def test_flexgen_policy_percentages_follow_spec():
    pt = fg.FLEXGEN_POINTS[0]
    k = {p: fg.flexgen_policy_kwargs(p, pt) for p in common.POLICIES}
    for p, kw in k.items():
        assert kw["w_gpu_percent"] + kw["w_cpu_percent"] == 100
        assert kw["cache_gpu_percent"] + kw["cache_cpu_percent"] == 100
        assert kw["gpu_batch_size"] * kw["num_gpu_batches"] == pt.batch
        assert not kw["compress_weight"] and not kw["compress_cache"]
        frac = common.POLICY_SPEC[p]["weight_offload_frac"]
        assert kw["w_cpu_percent"] == round(100 * frac)
    assert k["no-offload"]["w_gpu_percent"] == 100 and not k["no-offload"]["overlap"]
    assert k["paged-KV-offload"]["cache_cpu_percent"] == 100
    assert k["paged-KV-offload"]["w_gpu_percent"] == 100
    assert k["layer-pinned"]["w_gpu_percent"] == 75
    assert k["full-offload"]["w_gpu_percent"] == 40
    assert k["full-offload"]["cache_cpu_percent"] == 100


@pytest.mark.skipif(_installed("flexllmgen", "flexgen"),
                    reason="FlexGen is installed; this checks the absent case")
def test_flexgen_api_missing_raises():
    with pytest.raises(common.MeasurementUnavailable):
        fg._flexgen_api()
