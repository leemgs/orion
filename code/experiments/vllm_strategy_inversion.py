#!/usr/bin/env python3
"""Strategy-ranking inversion measurement harness for vLLM (author-run, real GPU).

This is the runnable harness for the single result the referee panel identifies
as the route to acceptance -- **the ranking of orchestration strategies inverts
across regimes** (``paper/submission/strategy_inversion_protocol.md``,
preregistration hypothesis H4; the predictive corollary is H3). It drives real
vLLM on a real accelerator, emits records in the locked schema of
``paper/submission/preregistration.md`` (so ``analyze_prereg.py`` consumes them
unchanged), and prints the inversion / prediction summary with bootstrap CIs.

No-fabrication contract
-----------------------
Every number comes from a real run or a real sub-measurement:

* ``T_total``  -- measured end-to-end request latency from vLLM (warm-up
  windows discarded), per (operating point, policy).
* ``B_slow``   -- measured *sustained* host->device bandwidth from a pinned-
  memory copy benchmark on the same GPU (not a datasheet peak).
* ``T_comp``   -- measured isolated compute: the same workload run fully
  resident, where there is no cross-tier transfer to hide behind. Compute work
  is policy-invariant for a fixed (model, batch, sequence length), so this is a
  legitimate isolated-compute measurement.
* ``T_transfer = D / B_slow`` -- the reporting contract's definition (Table 3),
  with ``D`` the per-step compulsory non-resident bytes forced by the capacity
  deficit. Timed/derived independently of compute, never from ``T_total``.
* ``W, C_fast, D, D_nr`` -- from the model's own config (exact KV formula +
  checkpoint weight bytes) and the declared GPU memory budget.

If vLLM or a CUDA GPU is absent, the harness raises ``MeasurementUnavailable``;
it never substitutes invented values. ``--dry-run`` emits schema-correct rows
tagged ``source="dryrun"`` (which ``analyze_prereg.py`` rejects as evidence) so
the whole pipeline -- schema, split, analysis -- can be validated without
hardware. If ``no-offload`` cannot fit a capacity-limited point, that
(point, policy) cell is skipped with a warning rather than faked.

Design: operating point vs. policy
----------------------------------
``(R_C, R_B)`` describe the *demand* and are held fixed across policies at one
operating point: the GPU memory budget and the workload (model, batch, sequence
length) set ``C_fast`` and ``W`` (hence ``R_C``) and the isolated compute vs.
compulsory transfer (hence ``R_B``). A *policy* is how the engine copes with
that demand (resident / KV-swap / layer-pin / weight-offload); it changes
``T_total`` but not the operating point. This is exactly what
``analyze_prereg.py`` expects: it groups records by (machine, model, R_C, R_B)
and compares policies within each point.

Usage
-----
    # 1) validate the pipeline end-to-end without a GPU
    python experiments/vllm_strategy_inversion.py --dry-run -o inv_dryrun.jsonl
    python experiments/analyze_prereg.py inv_dryrun.jsonl --split all  # rejects dryrun

    # 2) on a real GPU, with vLLM installed and a real model
    python experiments/vllm_strategy_inversion.py \
        --model meta-llama/Meta-Llama-3-8B --arch A100-80GB \
        --machine-id a100-node-1 --runs 5 -o records_a100.jsonl
    python experiments/analyze_prereg.py records_a100.jsonl --split all

Then merge the per-machine JSONL files and run ``analyze_prereg.py`` on the
held-out split, per the preregistration. Deposit the raw per-run records
(this file's output) under a Zenodo DOI (addresses referee R9).

Version note
------------
vLLM's offload knobs (``cpu_offload_gb``, ``swap_space``, ``gpu_memory_
utilization``, ``max_num_seqs``, ``max_model_len``) have been stable across
recent releases but their exact semantics can shift between major versions.
Verify ``policy_engine_kwargs`` against your installed vLLM before a campaign;
the realised ``(R_C, R_B)`` are recorded from what actually happened, so a
mis-set knob shows up as an off-target point rather than a silent error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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

# Preregistered fixed policy set (must match prereg_harvest.POLICIES and the
# a-priori map in analyze_prereg.DEFAULT_POLICY_MAP).
POLICIES = ["no-offload", "layer-pinned", "paged-KV-offload", "full-offload"]


@dataclass(frozen=True)
class OperatingPoint:
    """One demand configuration. The policies are run *at* this point.

    ``gpu_mem_fraction`` is the usable GPU budget (vLLM ``gpu_memory_
    utilization``); with ``batch`` and ``seq_len`` it fixes the demand and hence
    the target regime. ``compute_scale`` lengthens the output (more decode steps
    per request) to raise isolated compute relative to transfer, i.e. to move
    ``R_B`` -- used to place points on both sides of ``R_B = 1``.
    """
    name: str
    gpu_mem_fraction: float      # vLLM gpu_memory_utilization in (0, 1]
    batch: int                   # concurrent requests (vLLM max_num_seqs)
    seq_len: int                 # prompt tokens per request
    out_len: int                 # generated tokens per request
    target_regime: str = ""      # author's intended regime (documentation only)


# Template grid: author must choose points whose *measured* (R_C, R_B) land one
# per regime on their hardware/model (protocol step "Operating points"). These
# defaults bracket the two coordinates; confirm the realised ratios in the
# output and adjust gpu_mem_fraction / seq_len / out_len until each regime is hit.
DEFAULT_POINTS = [
    OperatingPoint("cap",   gpu_mem_fraction=0.30, batch=16, seq_len=512,
                   out_len=128, target_regime="capacity-limited"),
    OperatingPoint("io",    gpu_mem_fraction=0.70, batch=8,  seq_len=2048,
                   out_len=64,  target_regime="I/O-limited"),
    OperatingPoint("coord", gpu_mem_fraction=0.90, batch=4,  seq_len=4096,
                   out_len=256, target_regime="coordination-dominated"),
]

# Policy -> vLLM engine kwargs. These are real, distinct offload configurations.
# ``offload_gb`` is sized at call time from the model's weight bytes (see
# policy_engine_kwargs); ``swap_gb`` is CPU space for paged KV blocks.
POLICY_SPEC = {
    # all resident: no weight offload, no KV swap (only feasible where W<=C_fast)
    "no-offload":       dict(weight_offload_frac=0.0, swap_gb=0.0),
    # KV cache may page to CPU; weights stay resident
    "paged-KV-offload": dict(weight_offload_frac=0.0, swap_gb=8.0),
    # pin the hot fraction of weights on device, offload the cold remainder
    "layer-pinned":     dict(weight_offload_frac=0.25, swap_gb=0.0),
    # maximal weight offload to host plus KV swap
    "full-offload":     dict(weight_offload_frac=0.60, swap_gb=8.0),
}

DTYPE_BYTES = 2  # fp16/bf16 weights and KV
RUNS_PER_CELL = 5
WINDOWS = 10
WARMUP = 2


# --------------------------------------------------------------------------- #
# Real sub-measurements
# --------------------------------------------------------------------------- #
def measure_h2d_bandwidth(size_mb: int = 512, iters: int = 20) -> tuple[float, str]:
    """Measure sustained pinned host->device bandwidth (bytes/s) with CUDA events.

    Real measurement; raises if no CUDA GPU is present rather than returning a
    datasheet number. This is the ``B_slow`` of the reporting contract (achieved
    bandwidth on the limiting link, not peak).
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

    for _ in range(3):  # warm-up
        dst.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    secs = []
    for _ in range(iters):
        start.record()
        dst.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        secs.append(start.elapsed_time(end) / 1e3)
    mean_s = sum(secs) / len(secs)
    bps = n / mean_s
    name = torch.cuda.get_device_name(dev)
    return bps, f"{bps/1e9:.2f} GB/s sustained H2D on {name} ({size_mb} MiB x {iters})"


def kv_cache_bytes(cfg, batch: int, seq_len: int, out_len: int) -> float:
    """Exact KV-cache footprint for the workload from the model config.

    2 (K and V) * layers * kv_heads * head_dim * dtype * batch * (seq_len+out_len).
    """
    layers = int(getattr(cfg, "num_hidden_layers"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    hidden = int(getattr(cfg, "hidden_size"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
    tokens = batch * (seq_len + out_len)
    return 2.0 * layers * kv_heads * head_dim * DTYPE_BYTES * tokens


def checkpoint_weight_bytes(model: str, cfg=None) -> tuple[float, str]:
    """Weight bytes: sum the model's safetensors shards if reachable, else a
    config-based parameter estimate. Returns (bytes, how-derived)."""
    # Prefer the real on-disk size (a measurement, not an estimate).
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        from transformers.utils import cached_file  # type: ignore
        idx = cached_file(model, "model.safetensors.index.json",
                          _raise_exceptions_for_missing_entries=False)
        if idx:
            meta = json.loads(Path(idx).read_text())
            shards = set(meta.get("weight_map", {}).values())
            total = 0
            root = Path(idx).parent
            for shard in shards:
                p = root / shard
                if p.exists():
                    total += p.stat().st_size
            if total > 0:
                return float(total), f"sum of {len(shards)} safetensors shards"
    except Exception:
        pass
    # Fallback: parameter-count estimate from config (clearly labelled).
    if cfg is not None:
        h = int(getattr(cfg, "hidden_size"))
        L = int(getattr(cfg, "num_hidden_layers"))
        vocab = int(getattr(cfg, "vocab_size", 32000))
        inter = int(getattr(cfg, "intermediate_size", 4 * h))
        # attention (qkv+o ~ 4 h^2) + MLP (~3 h*inter) per layer + embeddings
        params = L * (4 * h * h + 3 * h * inter) + 2 * vocab * h
        return float(params * DTYPE_BYTES), "config parameter-count estimate (approx)"
    raise MeasurementUnavailable(
        f"cannot determine weight bytes for {model!r}; pass --weights-bytes")


def device_total_bytes() -> float:
    import torch
    if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
        raise MeasurementUnavailable("no CUDA GPU to read total memory")
    return float(torch.cuda.get_device_properties(0).total_memory)


# --------------------------------------------------------------------------- #
# vLLM measurement
# --------------------------------------------------------------------------- #
def policy_engine_kwargs(policy: str, point: OperatingPoint,
                         weights_bytes: float) -> dict:
    """Map a policy + operating point to concrete vLLM ``LLM(...)`` kwargs."""
    spec = POLICY_SPEC[policy]
    offload_gb = spec["weight_offload_frac"] * weights_bytes / 1e9
    kwargs = dict(
        gpu_memory_utilization=point.gpu_mem_fraction,
        max_num_seqs=point.batch,
        max_model_len=point.seq_len + point.out_len,
        swap_space=float(spec["swap_gb"]),       # GiB CPU space for KV blocks
        cpu_offload_gb=float(offload_gb),         # GiB of weights offloaded to CPU
        enforce_eager=True,                       # exclude CUDA-graph capture from timing
        dtype="auto",
    )
    return kwargs


def _mean_request_latency(llm, prompts, sampling) -> float:
    """One timed window: mean end-to-end latency per request (seconds)."""
    t0 = time.perf_counter()
    llm.generate(prompts, sampling, use_tqdm=False)
    dt = time.perf_counter() - t0
    return dt / max(1, len(prompts))


def measure_cell(model: str, point: OperatingPoint, policy: str,
                 run_idx: int, weights_bytes: float, weights_note: str,
                 b_slow: float, b_slow_note: str,
                 t_comp_s: float, t_comp_note: str) -> PointMeasurement | None:
    """Measure one (point, policy, run). Returns None if the config is infeasible
    (e.g. no-offload OOM at a capacity-limited point) -- never a faked number."""
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - env dependent
        raise MeasurementUnavailable(
            "vLLM is not installed; cannot measure. Install vLLM to run the "
            "real campaign, or use --dry-run to validate the pipeline."
        ) from exc

    # Working set and capacity from config + declared budget.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    kv_bytes = kv_cache_bytes(cfg, point.batch, point.seq_len, point.out_len)
    w_bytes = weights_bytes + kv_bytes  # active working set (weights + KV)
    c_fast = point.gpu_mem_fraction * device_total_bytes()
    d_nr = max(0.0, w_bytes - c_fast)   # compulsory non-resident bytes
    d_bytes = d_nr                      # per-step compulsory cross-tier traffic
    t_transfer = (d_bytes / b_slow) if (d_bytes > 0 and b_slow > 0) else 0.0

    kwargs = policy_engine_kwargs(policy, point, weights_bytes)
    prompts = [f"Summarise passage {i}: " + ("word " * point.seq_len)
               for i in range(point.batch)]
    sampling = SamplingParams(max_tokens=point.out_len, ignore_eos=True,
                              temperature=0.0)

    try:
        llm = LLM(model=model, **kwargs)
    except Exception as exc:  # OOM / infeasible config for this policy+point
        print(f"  [skip] {point.name}/{policy} run{run_idx}: engine init failed "
              f"({type(exc).__name__}: {exc}); recording no value.",
              file=sys.stderr)
        return None

    try:
        for _ in range(WARMUP):
            _mean_request_latency(llm, prompts, sampling)
        lat = [_mean_request_latency(llm, prompts, sampling) for _ in range(WINDOWS)]
    except Exception as exc:  # pragma: no cover - runtime failure on real HW
        print(f"  [skip] {point.name}/{policy} run{run_idx}: generate failed "
              f"({type(exc).__name__}: {exc}); recording no value.",
              file=sys.stderr)
        return None
    finally:
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    t_total = sum(lat) / len(lat)
    return PointMeasurement(
        t_comp_s=t_comp_s, t_transfer_s=max(t_transfer, 1e-12), t_total_s=t_total,
        c_fast_bytes=c_fast, w_bytes=w_bytes, d_bytes=d_bytes, d_nr_bytes=d_nr,
        b_slow_bytes_per_s=b_slow, windows=WINDOWS,
        provenance=(f"vLLM {policy} at point {point.name}; T_total=mean req "
                    f"latency over {WINDOWS} windows ({WARMUP} warm-up discarded); "
                    f"B_slow[{b_slow_note}]; T_comp[{t_comp_note}]; "
                    f"W=weights[{weights_note}]+KV(exact); "
                    f"T_transfer=D/B_slow (contract Table 3); "
                    f"kwargs={kwargs}"),
        extra={"point": point.name, "kv_bytes": kv_bytes},
    )


def isolated_compute_time(model: str, point: OperatingPoint,
                          weights_bytes: float) -> tuple[float, str]:
    """Measure T_comp: the workload run fully resident (no offload, no swap), so
    there is no cross-tier transfer. Compute is policy-invariant, so this is the
    isolated compute time for the point. Raises/returns 0 only if infeasible."""
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover
        raise MeasurementUnavailable("vLLM is not installed") from exc
    # Give the compute baseline enough budget to stay fully resident.
    kwargs = dict(gpu_memory_utilization=max(point.gpu_mem_fraction, 0.9),
                  max_num_seqs=point.batch,
                  max_model_len=point.seq_len + point.out_len,
                  swap_space=0.0, cpu_offload_gb=0.0, enforce_eager=True,
                  dtype="auto")
    prompts = [f"Summarise passage {i}: " + ("word " * point.seq_len)
               for i in range(point.batch)]
    sampling = SamplingParams(max_tokens=point.out_len, ignore_eos=True,
                              temperature=0.0)
    try:
        llm = LLM(model=model, **kwargs)
    except Exception as exc:  # pragma: no cover
        # Fully-resident baseline does not fit: report 0 (T_comp unavailable),
        # which makes R_B undefined for this point rather than fabricated.
        print(f"  [warn] isolated T_comp for {point.name} not measurable "
              f"({type(exc).__name__}); R_B will be unavailable.", file=sys.stderr)
        return 0.0, "unavailable (resident baseline did not fit)"
    try:
        for _ in range(WARMUP):
            _mean_request_latency(llm, prompts, sampling)
        lat = [_mean_request_latency(llm, prompts, sampling) for _ in range(WINDOWS)]
    finally:
        del llm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return sum(lat) / len(lat), f"resident no-offload baseline, {WINDOWS} windows"


# --------------------------------------------------------------------------- #
# Record assembly (locked schema) + driver
# --------------------------------------------------------------------------- #
def make_record(point: OperatingPoint, policy: str, run_idx: int,
                machine_id: str, arch: str, model: str,
                meas: PointMeasurement, source: str) -> dict:
    if source == "measured" and meas.w_bytes > 0:
        # Realised ratios from what actually happened (policy-invariant per point).
        # t_transfer_s is >= 1e-12 by construction, so a fully-resident point
        # (D=0) yields a very large R_B and classifies as coordination-dominated,
        # which is correct: there is no exposed transfer to hide. When the
        # isolated-compute baseline was unavailable (t_comp=0) the point is
        # capacity-limited by R_C (it could not be held resident), so
        # classify_regime still labels it correctly from R_C alone.
        r_c = meas.c_fast_bytes / meas.w_bytes
        r_b = meas.t_comp_s / meas.t_transfer_s
        label = REGIME_LABEL[classify_regime(r_c, r_b)]
        r_b_out = r_b
    else:
        r_c, r_b_out, label = 0.0, 0.0, REGIME_LABEL[Regime.CAPACITY_LIMITED]
    return {
        "machine_id": machine_id, "arch": arch, "model": model,
        "framework": "vllm", "policy": policy, "split": "all",
        "R_C": r_c, "R_B": r_b_out, "label": label,
        "C_fast_bytes": meas.c_fast_bytes, "W_bytes": meas.w_bytes,
        "D_bytes": meas.d_bytes, "D_nr_bytes": meas.d_nr_bytes,
        "B_slow_bytes_per_s": meas.b_slow_bytes_per_s,
        "T_comp_s": meas.t_comp_s, "T_transfer_s": meas.t_transfer_s,
        "T_total_s": meas.t_total_s,
        "run_idx": run_idx, "windows": meas.windows,
        "source": source, "provenance": meas.provenance,
    }


def _dry_run_meas(point: OperatingPoint, policy: str) -> PointMeasurement:
    """Schema-correct placeholder (zeros). NOT evidence; analyzer rejects it."""
    return PointMeasurement(
        t_comp_s=0.0, t_transfer_s=0.0, t_total_s=0.0, c_fast_bytes=0.0,
        w_bytes=0.0, d_bytes=0.0, d_nr_bytes=0.0, b_slow_bytes_per_s=0.0,
        windows=0, provenance=(f"DRY RUN placeholder for {point.name}/{policy}; "
                               "not a measurement."))


def run(model: str, machine_id: str, arch: str, points: list[OperatingPoint],
        runs: int, output: Path, dry_run: bool,
        weights_bytes_override: float | None = None) -> int:
    records: list[dict] = []
    if dry_run:
        for point in points:
            for policy in POLICIES:
                for run_idx in range(runs):
                    meas = _dry_run_meas(point, policy)
                    records.append(make_record(point, policy, run_idx, machine_id,
                                               arch, model, meas, "dryrun"))
    else:
        # Measure the shared, policy-invariant quantities once per machine/point.
        b_slow, b_slow_note = measure_h2d_bandwidth()
        if weights_bytes_override is not None:
            weights_bytes = weights_bytes_override
            weights_note = "user-supplied --weights-bytes"
        else:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(model)
            except Exception:
                cfg = None
            weights_bytes, weights_note = checkpoint_weight_bytes(model, cfg)
        print(f"model weights: {weights_bytes/1e9:.2f} GB ({weights_note}); "
              f"B_slow: {b_slow_note}", file=sys.stderr)

        for point in points:
            t_comp, t_comp_note = isolated_compute_time(model, point, weights_bytes)
            for policy in POLICIES:
                for run_idx in range(runs):
                    meas = measure_cell(model, point, policy, run_idx,
                                        weights_bytes, weights_note,
                                        b_slow, b_slow_note, t_comp, t_comp_note)
                    if meas is None:
                        continue  # infeasible cell: no record, not a fake number
                    records.append(make_record(point, policy, run_idx, machine_id,
                                               arch, model, meas, "measured"))

    with output.open("w") as fh:
        for rec in records:
            validate_record(rec)
            fh.write(json.dumps(rec) + "\n")
    kind = "dry-run" if dry_run else "measured"
    print(f"wrote {len(records)} {kind} records to {output}")
    return len(records)


def summarize(output: Path) -> None:
    """Print the inversion / prediction summary by reusing the locked analysis.

    Skipped for dry-run output (analyze_prereg correctly rejects non-measured
    rows as evidence)."""
    try:
        from experiments.analyze_prereg import (
            _points, h3_prediction, h4_ranking_change, DEFAULT_POLICY_MAP,
            load_records,
        )
    except Exception as exc:  # pragma: no cover
        print(f"(summary unavailable: {exc})", file=sys.stderr)
        return
    try:
        records = load_records(output, split=None)
    except Exception as exc:
        print(f"\nNo measured evidence to summarise ({exc}).")
        print("Run on a real GPU with vLLM to produce source='measured' records.")
        return
    points = _points(records)
    print("\n=== Strategy-inversion summary (measured) ===")
    print(f"operating points: {len(points)}")
    for key, p in points.items():
        best = p["observed_best"]
        lats = ", ".join(f"{k}={v*1e3:.1f}ms" for k, v in
                         sorted(p["policy_latency"].items(), key=lambda kv: kv[1]))
        print(f"  {p['label']:<22} R_C={p['R_C']:.3f} R_B={p['R_B']:.3f} "
              f"best={best} | {lats}")
    h3 = h3_prediction(points, DEFAULT_POLICY_MAP)
    print(f"\nH3 a-priori best-policy accuracy: {h3['accuracy']:.2f} "
          f"CI95={[round(x,2) for x in h3['accuracy_ci95']]} "
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSONL (locked schema; feeds analyze_prereg.py)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="HF model id or local path (real backends only)")
    parser.add_argument("--machine-id", default="local-gpu-1")
    parser.add_argument("--arch", default="local-gpu",
                        help="e.g. A100-80GB, H100-80GB, MI250")
    parser.add_argument("--runs", type=int, default=RUNS_PER_CELL,
                        help=f"independent runs per (point, policy) (default {RUNS_PER_CELL})")
    parser.add_argument("--weights-bytes", type=float, default=None,
                        help="override measured weight bytes (e.g. from the checkpoint)")
    parser.add_argument("--dry-run", action="store_true",
                        help="emit source='dryrun' placeholders (no hardware needed)")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    n = run(args.model, args.machine_id, args.arch, DEFAULT_POINTS,
            args.runs, args.output, args.dry_run,
            weights_bytes_override=args.weights_bytes)
    if args.dry_run:
        print("NOTE: dry-run rows are placeholders; analyze_prereg.py rejects them "
              "as evidence. Run on a real GPU with vLLM for source='measured'.")
    elif not args.no_summary and n:
        summarize(args.output)


if __name__ == "__main__":
    main()
