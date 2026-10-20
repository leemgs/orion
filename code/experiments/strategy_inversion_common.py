#!/usr/bin/env python3
"""Framework-neutral core of the strategy-ranking inversion harnesses.

The vLLM, DeepSpeed-Inference and FlexGen harnesses
(``vllm_strategy_inversion.py``, ``deepspeed_strategy_inversion.py``,
``flexgen_strategy_inversion.py``) share everything except how an engine is
built for a (operating point, policy) and how one batch is generated. That
shared part lives here so the three frameworks are measured identically:

* the preregistered policy set and a framework-neutral policy spec;
* operating points (the *demand*: GPU budget, batch, prompt/output length);
* the reporting-contract quantities -- ``W``, ``C_fast``, ``D``, ``D_nr`` from
  the model config and the declared budget, ``B_slow`` from a real
  host->device bandwidth probe, isolated ``T_comp`` from a resident baseline,
  and ``T_transfer = D / B_slow`` (manuscript Table 3);
* the timing loop (warm-up windows discarded), the locked JSONL record schema
  of ``paper/submission/preregistration.md``, and the H3/H4 summary computed
  by reusing ``analyze_prereg.py``.

No-fabrication contract
-----------------------
Nothing here invents a number. A missing library or GPU raises
``MeasurementUnavailable`` before any record is written; a configuration that
cannot run (out of memory, unsupported by the installed framework version) is
skipped and logged, never filled in. ``--dry-run`` writes ``source="dryrun"``
placeholders that ``analyze_prereg.py`` refuses as evidence.

Operating point vs. policy
--------------------------
``(R_C, R_B)`` describe the demand and are identical for every policy at one
operating point; a policy only changes how the engine copes, i.e. ``T_total``.
``T_comp`` is measured per framework (each framework's own resident execution),
so analyse frameworks separately or keep the ``framework`` field in mind when
merging files.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from orion.config import Regime  # noqa: E402
from orion.ratios import classify_regime  # noqa: E402
from experiments.prereg_backends import (  # noqa: E402
    MeasurementUnavailable,
    PointMeasurement,
)
from experiments.prereg_harvest import REGIME_LABEL, validate_record  # noqa: E402

# Preregistered fixed policy set (prereg_harvest.POLICIES, analyze_prereg map).
POLICIES = ["no-offload", "layer-pinned", "paged-KV-offload", "full-offload"]

# Framework-neutral meaning of each policy. Each adapter maps this onto its own
# knobs; where a framework cannot express it exactly, the adapter says so in
# its docstring and in every record's provenance.
POLICY_SPEC = {
    "no-offload":       dict(weight_offload_frac=0.0,  kv_offload=False),
    "paged-KV-offload": dict(weight_offload_frac=0.0,  kv_offload=True),
    "layer-pinned":     dict(weight_offload_frac=0.25, kv_offload=False),
    "full-offload":     dict(weight_offload_frac=0.60, kv_offload=True),
}

KV_OFFLOAD_GB = 8.0   # host space reserved for offloaded KV where a size is needed
DTYPE_BYTES = 2       # fp16/bf16 weights and KV
RUNS_PER_CELL = 5
WINDOWS = 10
WARMUP = 2
RESIDENT_BUDGET = 0.90  # GPU budget for the isolated-compute (resident) baseline


class PolicyUnsupported(RuntimeError):
    """The installed framework cannot realise this policy; the cell is skipped."""


@dataclass(frozen=True)
class OperatingPoint:
    """One demand configuration; every policy is run *at* this point.

    ``gpu_mem_fraction`` is the usable GPU budget (fraction of device memory).
    With ``batch``, ``seq_len`` (prompt tokens) and ``out_len`` (generated
    tokens) it fixes the demand and hence the target regime.
    """
    name: str
    gpu_mem_fraction: float
    batch: int
    seq_len: int
    out_len: int
    target_regime: str = ""  # author's intent (documentation only)


# Template grid. The author must confirm in the output that the *measured*
# (R_C, R_B) land one point per regime on their hardware/model and adjust the
# budget / lengths until they do (protocol step "Operating points").
DEFAULT_POINTS = [
    OperatingPoint("cap",   gpu_mem_fraction=0.30, batch=16, seq_len=512,
                   out_len=128, target_regime="capacity-limited"),
    OperatingPoint("io",    gpu_mem_fraction=0.70, batch=8,  seq_len=2048,
                   out_len=64,  target_regime="I/O-limited"),
    OperatingPoint("coord", gpu_mem_fraction=0.90, batch=4,  seq_len=4096,
                   out_len=256, target_regime="coordination-dominated"),
]


# --------------------------------------------------------------------------- #
# Model-config quantities (pure Python; no GPU needed)
# --------------------------------------------------------------------------- #
def _kv_dims(cfg) -> tuple[int, int, int]:
    layers = int(getattr(cfg, "num_hidden_layers"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    hidden = int(getattr(cfg, "hidden_size"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden // n_heads)
    return layers, kv_heads, head_dim


def kv_bytes_per_token(cfg) -> float:
    """K and V bytes stored per token: 2 * layers * kv_heads * head_dim * dtype."""
    layers, kv_heads, head_dim = _kv_dims(cfg)
    return 2.0 * layers * kv_heads * head_dim * DTYPE_BYTES


def kv_cache_bytes(cfg, batch: int, seq_len: int, out_len: int) -> float:
    """Exact KV-cache footprint of the workload from the model config."""
    return kv_bytes_per_token(cfg) * batch * (seq_len + out_len)


def max_context(cfg) -> int | None:
    v = getattr(cfg, "max_position_embeddings", None)
    return int(v) if v else None


def point_fits_context(cfg, point: OperatingPoint) -> bool:
    ctx = max_context(cfg)
    return ctx is None or point.seq_len + point.out_len <= ctx


def synthetic_prompt_ids(cfg, point: OperatingPoint) -> list[list[int]]:
    """Exactly ``seq_len`` token ids per request, identical for every policy and
    run at a point (string-seeded RNG is reproducible across processes).

    Token ids rather than text keep the prompt length exact, so the demand --
    and therefore (R_C, R_B) -- is the same in every framework.
    """
    vocab = int(getattr(cfg, "vocab_size", 32000))
    lo, hi = 100, max(101, vocab - 100)  # stay clear of special-token ranges
    rng = random.Random(f"{point.name}:{point.batch}:{point.seq_len}")
    return [[rng.randrange(lo, hi) for _ in range(point.seq_len)]
            for _ in range(point.batch)]


def checkpoint_weight_bytes(model: str, cfg=None) -> tuple[float, str]:
    """Weight bytes: on-disk checkpoint size if reachable (a measurement), else
    a config-based parameter estimate that is labelled as such."""
    try:
        from transformers.utils import cached_file  # type: ignore
        for index_name in ("model.safetensors.index.json",
                           "pytorch_model.bin.index.json"):
            idx = cached_file(model, index_name,
                              _raise_exceptions_for_missing_entries=False)
            if idx:
                shards = set(json.loads(Path(idx).read_text())
                             .get("weight_map", {}).values())
                total = 0
                for shard in shards:
                    p = cached_file(model, shard,
                                    _raise_exceptions_for_missing_entries=False)
                    if p and Path(p).exists():
                        total += Path(p).stat().st_size
                if total > 0:
                    return float(total), f"sum of {len(shards)} checkpoint shards"
        for single in ("model.safetensors", "pytorch_model.bin"):
            p = cached_file(model, single,
                            _raise_exceptions_for_missing_entries=False)
            if p and Path(p).exists():
                return float(Path(p).stat().st_size), f"checkpoint file {single}"
    except Exception:
        pass
    if cfg is not None:
        h = int(getattr(cfg, "hidden_size"))
        n_layers = int(getattr(cfg, "num_hidden_layers"))
        vocab = int(getattr(cfg, "vocab_size", 32000))
        inter = int(getattr(cfg, "intermediate_size", None)
                    or getattr(cfg, "ffn_dim", None) or 4 * h)
        params = n_layers * (4 * h * h + 3 * h * inter) + 2 * vocab * h
        return float(params * DTYPE_BYTES), "config parameter-count estimate (approx)"
    raise MeasurementUnavailable(
        f"cannot determine weight bytes for {model!r}; pass --weights-bytes")


@dataclass(frozen=True)
class Demand:
    """Policy-invariant demand of one operating point (reporting-contract fields)."""
    kv_bytes: float
    w_bytes: float       # active working set = weights + KV
    c_fast: float        # usable fast-tier bytes = budget * device memory
    d_nr: float          # compulsorily non-resident bytes
    d_bytes: float       # per-step compulsory cross-tier traffic
    t_transfer: float    # D / B_slow


def demand_for(cfg, point: OperatingPoint, weights_bytes: float,
               device_bytes: float, b_slow: float) -> Demand:
    kv = kv_cache_bytes(cfg, point.batch, point.seq_len, point.out_len)
    w = weights_bytes + kv
    c_fast = point.gpu_mem_fraction * device_bytes
    d_nr = max(0.0, w - c_fast)
    t_transfer = (d_nr / b_slow) if (d_nr > 0 and b_slow > 0) else 0.0
    return Demand(kv, w, c_fast, d_nr, d_nr, t_transfer)


# --------------------------------------------------------------------------- #
# Real sub-measurements (need a CUDA GPU)
# --------------------------------------------------------------------------- #
def measure_h2d_bandwidth(size_mb: int = 512, iters: int = 20) -> tuple[float, str]:
    """Sustained pinned host->device bandwidth (bytes/s) with CUDA events.

    This is the contract's ``B_slow`` (achieved on the limiting link, not the
    datasheet peak). Buffers are released afterwards so the probe does not eat
    into the budget of the engine measured next.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - env dependent
        raise MeasurementUnavailable("PyTorch is not installed") from exc
    if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
        raise MeasurementUnavailable("no CUDA GPU for the H2D bandwidth probe")

    dev = torch.device("cuda")
    n = size_mb * 1024 * 1024
    host = torch.empty(n, dtype=torch.uint8).pin_memory()
    dst = torch.empty(n, dtype=torch.uint8, device=dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        dst.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    secs = []
    for _ in range(iters):
        start.record()
        dst.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        secs.append(start.elapsed_time(end) / 1e3)
    name = torch.cuda.get_device_name(dev)
    del host, dst
    torch.cuda.empty_cache()
    bps = n / (sum(secs) / len(secs))
    return bps, f"{bps/1e9:.2f} GB/s sustained H2D on {name} ({size_mb} MiB x {iters})"


def device_total_bytes() -> float:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise MeasurementUnavailable("PyTorch is not installed") from exc
    if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
        raise MeasurementUnavailable("no CUDA GPU to read total memory")
    return float(torch.cuda.get_device_properties(0).total_memory)


def limit_gpu_budget(fraction: float) -> None:
    """Cap this process's CUDA caching allocator at ``fraction`` of device memory.

    Used by in-process engines (DeepSpeed, FlexGen) to impose an operating
    point's GPU budget; vLLM enforces its own via ``gpu_memory_utilization``.
    """
    import torch
    torch.cuda.set_per_process_memory_fraction(min(max(fraction, 0.05), 1.0), 0)


def release_cuda() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def gpu_resident_bytes() -> float | None:
    """Bytes currently allocated by this process on the GPU (a measurement)."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated())
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Engine adapter interface
# --------------------------------------------------------------------------- #
class EngineAdapter:
    """One serving framework. Subclasses build an engine for (point, policy)
    and generate one batch; the common code does everything else."""

    framework = "unknown"
    default_model = ""
    default_points = DEFAULT_POINTS
    #: whether the engine runs in this process (so GPU memory can be read here)
    in_process = True

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Register framework-specific CLI options."""

    def configure(self, args: argparse.Namespace) -> None:
        """Read framework-specific CLI options."""

    def require(self) -> None:
        """Raise MeasurementUnavailable if the framework is not installed."""
        raise NotImplementedError

    def build(self, model: str, cfg, point: OperatingPoint, policy: str,
              weights_bytes: float, budget: float) -> Any:
        """Construct the engine; raise PolicyUnsupported if it cannot be realised."""
        raise NotImplementedError

    def prepare(self, handle: Any, point: OperatingPoint,
                prompt_ids: list[list[int]]) -> Any:
        """Build per-batch inputs once, outside the timed region."""
        return prompt_ids

    def window(self, handle: Any, point: OperatingPoint, inputs: Any) -> float:
        """Generate one batch synchronously; return its wall time in seconds."""
        raise NotImplementedError

    def close(self, handle: Any) -> None:
        release_cuda()

    def describe(self, handle: Any) -> str:
        """Engine configuration actually used, for provenance."""
        return ""


def _timed_latency(adapter: EngineAdapter, handle: Any, point: OperatingPoint,
                   inputs: Any) -> float:
    """Mean per-request latency over WINDOWS batches after WARMUP discarded."""
    for _ in range(WARMUP):
        adapter.window(handle, point, inputs)
    lat = [adapter.window(handle, point, inputs) / max(1, point.batch)
           for _ in range(WINDOWS)]
    return sum(lat) / len(lat)


def _skip(point: OperatingPoint, policy: str, run_idx: int, what: str,
          exc: BaseException) -> None:
    print(f"  [skip] {point.name}/{policy} run{run_idx}: {what} "
          f"({type(exc).__name__}: {exc}); recording no value.", file=sys.stderr)


def isolated_compute_time(adapter: EngineAdapter, model: str, cfg,
                          point: OperatingPoint, weights_bytes: float
                          ) -> tuple[float, str]:
    """``T_comp``: the same workload fully resident (no offload, no transfer).

    Compute is policy-invariant for a fixed model/batch/lengths, so this is the
    isolated compute time of the point. If the resident baseline does not fit,
    returns 0.0 -- the point is then capacity-limited by R_C, and R_B is
    reported as unavailable rather than invented.
    """
    budget = max(point.gpu_mem_fraction, RESIDENT_BUDGET)
    try:
        handle = adapter.build(model, cfg, point, "no-offload", weights_bytes, budget)
    except Exception as exc:
        print(f"  [warn] isolated T_comp for {point.name} not measurable "
              f"({type(exc).__name__}: {exc}); R_B unavailable.", file=sys.stderr)
        release_cuda()
        return 0.0, "unavailable (resident baseline did not fit)"
    try:
        inputs = adapter.prepare(handle, point, synthetic_prompt_ids(cfg, point))
        t = _timed_latency(adapter, handle, point, inputs)
    except Exception as exc:
        print(f"  [warn] isolated T_comp for {point.name} failed "
              f"({type(exc).__name__}: {exc}); R_B unavailable.", file=sys.stderr)
        return 0.0, "unavailable (resident baseline failed)"
    finally:
        adapter.close(handle)
    return t, (f"{adapter.framework} resident no-offload baseline, budget "
               f"{budget:.2f}, {WINDOWS} windows")


def measure_cell(adapter: EngineAdapter, model: str, cfg, point: OperatingPoint,
                 policy: str, run_idx: int, dem: Demand, weights_bytes: float,
                 notes: dict, b_slow: float, t_comp: float
                 ) -> PointMeasurement | None:
    """One (point, policy, run). Returns None when the cell cannot run."""
    try:
        handle = adapter.build(model, cfg, point, policy, weights_bytes,
                               point.gpu_mem_fraction)
    except Exception as exc:  # OOM, PolicyUnsupported, bad config
        _skip(point, policy, run_idx, "engine init failed", exc)
        release_cuda()
        return None
    try:
        resident = gpu_resident_bytes() if adapter.in_process else None
        inputs = adapter.prepare(handle, point, synthetic_prompt_ids(cfg, point))
        t_total = _timed_latency(adapter, handle, point, inputs)
        engine_desc = adapter.describe(handle)
    except Exception as exc:
        _skip(point, policy, run_idx, "generate failed", exc)
        return None
    finally:
        adapter.close(handle)

    resident_note = (f"; GPU-resident after init={resident/1e9:.2f} GB (measured)"
                     if resident is not None else "")
    return PointMeasurement(
        t_comp_s=t_comp, t_transfer_s=max(dem.t_transfer, 1e-12), t_total_s=t_total,
        c_fast_bytes=dem.c_fast, w_bytes=dem.w_bytes, d_bytes=dem.d_bytes,
        d_nr_bytes=dem.d_nr, b_slow_bytes_per_s=b_slow, windows=WINDOWS,
        provenance=(f"{adapter.framework} {policy} at point {point.name}; "
                    f"T_total=mean request latency over {WINDOWS} windows "
                    f"({WARMUP} warm-up discarded); B_slow[{notes['b_slow']}]; "
                    f"T_comp[{notes['t_comp']}]; W=weights[{notes['weights']}]"
                    f"+KV(exact); T_transfer=D/B_slow (contract Table 3)"
                    f"{resident_note}; engine={engine_desc}"),
        extra={"point": point.name, "kv_bytes": dem.kv_bytes},
    )


# --------------------------------------------------------------------------- #
# Records, driver, summary
# --------------------------------------------------------------------------- #
def make_record(point: OperatingPoint, policy: str, run_idx: int,
                machine_id: str, arch: str, model: str,
                meas: PointMeasurement, source: str, framework: str) -> dict:
    """Assemble one record in the locked schema.

    For measured rows the ratios are the realised ones. ``t_transfer_s`` is
    >= 1e-12 by construction, so a fully-resident point (D=0) gets a very large
    R_B and is coordination-dominated -- correct, there is no exposed transfer.
    If the resident baseline was unavailable (t_comp=0) the point could not be
    held resident, i.e. R_C < theta_C, and classify_regime labels it from R_C.
    """
    if source == "measured" and meas.w_bytes > 0:
        r_c = meas.c_fast_bytes / meas.w_bytes
        r_b = meas.t_comp_s / meas.t_transfer_s
        label = REGIME_LABEL[classify_regime(r_c, r_b)]
    else:
        r_c, r_b, label = 0.0, 0.0, REGIME_LABEL[Regime.CAPACITY_LIMITED]
    return {
        "machine_id": machine_id, "arch": arch, "model": model,
        "framework": framework, "policy": policy, "split": "all",
        "R_C": r_c, "R_B": r_b, "label": label,
        "C_fast_bytes": meas.c_fast_bytes, "W_bytes": meas.w_bytes,
        "D_bytes": meas.d_bytes, "D_nr_bytes": meas.d_nr_bytes,
        "B_slow_bytes_per_s": meas.b_slow_bytes_per_s,
        "T_comp_s": meas.t_comp_s, "T_transfer_s": meas.t_transfer_s,
        "T_total_s": meas.t_total_s,
        "run_idx": run_idx, "windows": meas.windows,
        "source": source, "provenance": meas.provenance,
    }


def dry_run_measurement(point: OperatingPoint, policy: str) -> PointMeasurement:
    """Schema-correct placeholder (zeros). NOT evidence; the analyser rejects it."""
    return PointMeasurement(
        t_comp_s=0.0, t_transfer_s=0.0, t_total_s=0.0, c_fast_bytes=0.0,
        w_bytes=0.0, d_bytes=0.0, d_nr_bytes=0.0, b_slow_bytes_per_s=0.0,
        windows=0, provenance=(f"DRY RUN placeholder for {point.name}/{policy}; "
                               "not a measurement."))


def run(adapter: EngineAdapter, model: str, machine_id: str, arch: str,
        points: list[OperatingPoint], runs: int, output: Path, dry_run: bool,
        weights_bytes_override: float | None = None, run_offset: int = 0) -> int:
    records: list[dict] = []
    run_ids = range(run_offset, run_offset + runs)
    if dry_run:
        for point in points:
            for policy in POLICIES:
                for run_idx in run_ids:
                    records.append(make_record(
                        point, policy, run_idx, machine_id, arch, model,
                        dry_run_measurement(point, policy), "dryrun",
                        adapter.framework))
    else:
        adapter.require()
        b_slow, b_slow_note = measure_h2d_bandwidth()
        device_bytes = device_total_bytes()
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model)
        except Exception as exc:
            raise MeasurementUnavailable(
                f"cannot load the config of {model!r}: {exc}") from exc
        if weights_bytes_override is not None:
            weights_bytes, weights_note = weights_bytes_override, "user-supplied --weights-bytes"
        else:
            weights_bytes, weights_note = checkpoint_weight_bytes(model, cfg)
        print(f"[{adapter.framework}] weights {weights_bytes/1e9:.2f} GB "
              f"({weights_note}); B_slow {b_slow_note}", file=sys.stderr)

        for point in points:
            if not point_fits_context(cfg, point):
                print(f"  [skip] point {point.name}: seq_len+out_len="
                      f"{point.seq_len + point.out_len} exceeds the model context "
                      f"({max_context(cfg)}).", file=sys.stderr)
                continue
            dem = demand_for(cfg, point, weights_bytes, device_bytes, b_slow)
            t_comp, t_comp_note = isolated_compute_time(
                adapter, model, cfg, point, weights_bytes)
            notes = {"b_slow": b_slow_note, "t_comp": t_comp_note,
                     "weights": weights_note}
            for policy in POLICIES:
                for run_idx in run_ids:
                    meas = measure_cell(adapter, model, cfg, point, policy,
                                        run_idx, dem, weights_bytes, notes,
                                        b_slow, t_comp)
                    if meas is None:
                        continue  # infeasible cell: no record, not a fake number
                    records.append(make_record(point, policy, run_idx, machine_id,
                                               arch, model, meas, "measured",
                                               adapter.framework))

    with output.open("w") as fh:
        for rec in records:
            validate_record(rec)
            fh.write(json.dumps(rec) + "\n")
    kind = "dry-run" if dry_run else "measured"
    print(f"wrote {len(records)} {kind} {adapter.framework} records to {output}")
    return len(records)


def summarize(output: Path) -> None:
    """Print the H3/H4 summary by reusing the locked analysis. Non-measured rows
    are refused by ``analyze_prereg.load_records``, as they must be."""
    try:
        from experiments.analyze_prereg import (
            DEFAULT_POLICY_MAP, _points, h3_prediction, h4_ranking_change,
            load_records,
        )
    except Exception as exc:  # pragma: no cover
        print(f"(summary unavailable: {exc})", file=sys.stderr)
        return
    try:
        records = load_records(output, split=None)
    except Exception as exc:
        print(f"\nNo measured evidence to summarise ({exc}).")
        print("Run on a real GPU to produce source='measured' records.")
        return
    frameworks = sorted({r.get("framework", "?") for r in records})
    points = _points(records)
    print(f"\n=== Strategy-inversion summary (measured; framework={','.join(frameworks)}) ===")
    print(f"operating points: {len(points)}")
    for p in points.values():
        lats = ", ".join(f"{k}={v*1e3:.1f}ms" for k, v in
                         sorted(p["policy_latency"].items(), key=lambda kv: kv[1]))
        print(f"  {p['label']:<22} R_C={p['R_C']:.3f} R_B={p['R_B']:.3f} "
              f"best={p['observed_best']} | {lats}")
    h3 = h3_prediction(points, DEFAULT_POLICY_MAP)
    print(f"\nH3 a-priori best-policy accuracy: {h3['accuracy']:.2f} "
          f"CI95={[round(x, 2) for x in h3['accuracy_ci95']]} "
          f"vs baseline '{h3['baseline_policy']}'={h3['baseline_accuracy']:.2f} "
          f"-> beats_baseline={h3['beats_baseline']}")
    for coord in ("R_C", "R_B"):
        changes = [f for f in h4_ranking_change(points, coord) if f["ranking_change"]]
        if changes:
            print(f"H4 ranking inversion across {coord}:")
            for c in changes:
                print(f"  {c['pair'][0]} vs {c['pair'][1]}: "
                      f"below={c['below']['mean_diff']*1e3:+.1f}ms "
                      f"above={c['above']['mean_diff']*1e3:+.1f}ms (CIs separated)")
        else:
            print(f"H4 ranking inversion across {coord}: none detected.")


def cli_main(adapter: EngineAdapter, doc: str) -> None:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSONL (locked schema; feeds analyze_prereg.py)")
    parser.add_argument("--model", default=adapter.default_model,
                        help=f"HF model id or local path (default {adapter.default_model})")
    parser.add_argument("--machine-id", default="local-gpu-1")
    parser.add_argument("--arch", default="local-gpu",
                        help="e.g. A100-80GB, H100-80GB, MI250")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CELL,
                        help=f"independent runs per (point, policy) (default {RUNS_PER_CELL})")
    parser.add_argument("--run-offset", type=int, default=0,
                        help="first run_idx; run one process per run and merge "
                             "the JSONL files if engines leak memory in-process")
    parser.add_argument("--weights-bytes", type=float, default=None,
                        help="override measured weight bytes")
    parser.add_argument("--dry-run", action="store_true",
                        help="emit source='dryrun' placeholders (no hardware needed)")
    parser.add_argument("--no-summary", action="store_true")
    adapter.add_args(parser)
    args = parser.parse_args()
    adapter.configure(args)

    n = run(adapter, args.model, args.machine_id, args.arch,
            list(adapter.default_points), args.runs, args.output, args.dry_run,
            weights_bytes_override=args.weights_bytes, run_offset=args.run_offset)
    if args.dry_run:
        print("NOTE: dry-run rows are placeholders; analyze_prereg.py rejects them "
              "as evidence. Run on a real GPU for source='measured'.")
    elif not args.no_summary and n:
        summarize(args.output)
