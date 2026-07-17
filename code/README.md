# Orion — Source Code

Reference implementation for:

> **"Hierarchical memory orchestration in AI inference exhibits intrinsic regime-dependent limits"**  
> Primary submission target: *Nature Machine Intelligence*

This directory contains the Orion measurement framework and the scripts used to reproduce the manuscript's figures, tables, and regime-classification results. The code documentation follows the current Nature Machine Intelligence (NMI) framing: ORION is presented as evidence for a general principle of memory-bound machine intelligence, while preserving a strict distinction between simulated reproduction and real-hardware measurements.

### NMI manuscript alignment

- **Scientific claim:** hierarchical memory orchestration exhibits three intrinsic operating regimes separated by abrupt, phase-like transitions.
- **Generality:** the framework uses dimensionless ratios, R_C and R_B, to compare hardware platforms and AI workloads.
- **Strategy inversion:** an optimization that improves latency in one regime can degrade it in another.
- **Reproducibility:** CPU-only simulation reproduces the qualitative regime structure; quantitative values reported as hardware measurements must come from live traces.
- **Transparency:** unavailable counters are reported as `NaN` or `MeasurementUnavailable`; live mode never falls back silently to simulation.

The NMI-framed manuscript sources are [`../paper/section/006_abstract_nmi.tex`](../paper/section/006_abstract_nmi.tex) and [`../paper/section/010_introduction_nmi.tex`](../paper/section/010_introduction_nmi.tex). Build and submission instructions are in [`../paper/README.md`](../paper/README.md).

---

## Directory structure

```
code/
├── orion/                      # Core Orion framework
│   ├── config.py               #   Thresholds, hardware profiles, model specs
│   ├── ratios.py               #   R_C, R_B computation & regime classification
│   ├── classifier.py           #   Depth-3 CART regime classifier (<0.1 ms)
│   ├── strategies.py           #   Per-regime orchestration strategies
│   ├── lower_bound.py          #   Structural lower bound & sharpness coefficient S
│   ├── profiler.py             #   Latency decomposition & hardware profiling
│   └── orchestrator.py         #   Main control loop (Orion_HW / Orion_Full)
├── experiments/
│   ├── simulated_backend.py    #   CPU-only synthetic backend for offline reproduction
│   ├── run_regime_sweep.py     #   Full R_C / R_B probing sweep (→ JSONL logs)
│   ├── reproduce_table2.py     #   Table 2: regime-dependent strategy inversion
│   ├── reproduce_table3.py     #   Table 3: workload generality (BLIP2/RAG/YOLOv8)
│   ├── reproduce_figure2.py    #   Figure 2: regime map + latency decomposition
│   └── reproduce_classifier_ablation.py  # Table D.1: Orion_HW vs Orion_Full
├── utils/
│   ├── stats.py                #   Bootstrap CI, Wilcoxon test, SweepStats
│   └── logging.py              #   JSONL record writer/reader
├── requirements.txt
└── setup.py
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- For **simulated reproduction** (no GPU): no additional dependencies required
- For **live GPU experiments**: CUDA 12.x, PyTorch ≥ 2.4, NVML (pynvml ≥ 11.0)

### Install (simulated mode)

```bash
cd code/
pip install -e ".[plot]"
# or without pip install:
pip install numpy scipy matplotlib
```

### Install (live GPU mode)

```bash
pip install -e ".[gpu,plot]"
# Install inference backends (as needed):
pip install vllm>=0.5.3
pip install deepspeed>=0.14
# FlexGen: follow https://github.com/FMInference/FlexGen
```

---

## Quick start — verify installation

```python
import sys; sys.path.insert(0, 'code')
from orion import from_hardware_model, A100_80GB, LLAMA3_8B

# Reproduce paper worked example (Introduction):
# Llama-3 8B at batch=8 on 80 GB HBM server
op = from_hardware_model(A100_80GB, LLAMA3_8B, t_comp_s=0.120, d_bytes=1.2e9)
print(f"R_C = {op.r_c:.2f}   (paper: 1.91)")  # 1.92
print(f"R_B = {op.r_b:.2f}   (paper: 1.61)")  # 1.61
print(f"Regime: {op.regime.name}")              # COORDINATION_DOMINATED
```

---

## Reproducing the NMI manuscript results

All experiments can be reproduced **without GPU** using the simulated backend.  
The simulated backend reproduces the qualitative regime structure and exercises the complete analysis pipeline; it must not be presented as real-hardware evidence. Exact quantitative reproduction requires the raw JSONL hardware traces. The Zenodo DOI remains a placeholder until the archive is published.

### Table 2 — Regime-dependent strategy inversion

```bash
cd code/
python experiments/reproduce_table2.py
```

Expected output: FlexGen shows −24% in coordination-dominated regime but +8–12% in I/O-limited.

### Table 3 — Workload generality (BLIP2 / RAG / YOLOv8)

```bash
python experiments/reproduce_table3.py
```

### Figure 2 — Regime map and latency decomposition

```bash
# Save as PDF (requires matplotlib):
python experiments/reproduce_figure2.py --save-pdf figure2_reproduced.pdf

# Save raw CSV data (no matplotlib required):
python experiments/reproduce_figure2.py --save-csv
# → results/figure2/fig2a_rc_sweep.csv
#   results/figure2/fig2b_decomposition.csv
#   results/figure2/fig2c_rb_sweep.csv
```

### Table D.1 — Classifier ablation (Orion_HW vs Orion_Full)

```bash
python experiments/reproduce_classifier_ablation.py
```

Expected: hardware-only gain −3.8 to −8.3%; classifier adds −1.1 to −18.8% (21–76% of total).

### Full regime probing sweep

```bash
# Simulated (CPU-only, ~2 min):
python experiments/run_regime_sweep.py --mode simulate

# Live A100 hardware:
python experiments/run_regime_sweep.py --mode live --platform A100-80GB
```

Output is written to `results/regime_sweep/`:
- `sweep_XX.jsonl` — raw 10-second window records (Zenodo archive format)
- `summary.json` — boundary estimates and sharpness coefficients

---

## Running on real hardware

### A100 80 GB (primary platform)

Software requirements: PyTorch 2.4, CUDA 12.2, vLLM 0.5.3, pynvml ≥ 11.0.

```bash
python experiments/run_regime_sweep.py \
    --mode live \
    --platform A100-80GB \
    --n-sweeps 5 \
    --n-windows 30 \
    --output-dir results/a100_sweep
```

The sweep varies R_C ∈ {0.10 … 2.00} and R_B ∈ {0.10 … 2.00} (14 grid points each).
A 60-second warm-up period precedes each operating point.
Grid points that the hardware cannot realise are skipped and recorded, with a
reason, under `rc_unreachable` / `rb_unreachable` in `summary.json` — see
*Reachability* below, which affects far more of the grid than one might expect.

### Live measurement caveats

These are properties of the measurement, not bugs, and they constrain what the
live backend can report.

**Nothing is synthesised.** Every field of a live `LatencyRecord` is a device
measurement. When a quantity cannot be measured the backend raises
`MeasurementUnavailable` or records `NaN`. `--mode live` will *not* silently
fall back to `--mode simulate`: simulated records are synthetic and must never
be reported as measurements.

**T_mem needs a separate pass.** Nsight Compute replays kernels to read HBM
counters, which perturbs the very timings under study, so counters cannot be
collected inside a timed window. Live windows carry `t_mem = NaN` until
`CUDABackend.attach_mem_counters()` supplies a miss volume from an `ncu` pass at
the same operating point. `ncu` also needs GPU performance-counter permissions
(NVIDIA `ERR_NVGPUCTRPERM`).

**T_comp and T_mem overlap.** Kernel time already contains the HBM stall cycles
that T_mem quantifies, so `t_comp + t_mem` double-counts them. The additive
four-term decomposition needs a compute-only definition of T_comp to be well
posed. `LatencyRecord.completeness_error` surfaces this rather than hiding it.

**T_swap is exposed stall, not transfer time.** Transfers that fully overlap
compute cost nothing end-to-end, so the backend charges only the time the
compute stream is *blocked* on a staging copy. This is what makes an additive
decomposition physically meaningful.

**ρ is measured, not assumed.** `calibrate_rho()` runs a streaming microbenchmark
whose working set far exceeds L2 and derives ρ = 1/effective_HBM_bandwidth from
the achieved rate. The `HardwareProfile.rho` constants in `config.py` are used
only if calibration is explicitly disabled; they have not been reconciled with
measured bandwidth.

**Weights are random by default.** Latency depends on tensor shapes, dtypes and
residency, not on weight values, so this is sound for regime characterisation —
but such a run measures the architecture, not a specific checkpoint.

### Reachability: which operating points exist

Two hard limits bound the (R_C, R_B) grid, and both bite at realistic
batch/sequence settings:

**R_C floor.** Activations and the KV cache must be resident for a step to run,
so only parameter bytes are tradeable. This puts a floor under R_C:

```
R_C ≥ (W_act + W_kv) / W
```

For Llama-3 8B at batch=8, seq=2048 (`config.py`: W=41.8 GB, W_act=17.2 GB,
W_kv=8.6 GB) the floor is **0.617** — above θ_C = 0.50. The capacity-limited
regime is not reachable at this configuration; reaching it requires a smaller
batch or sequence length, or offloading activations too.

**R_B ceiling.** The token bucket can only throttle *below* the link's sustained
rate, so:

```
R_B ≤ B_hw · T_comp / D
```

Raising R_B past that requires shrinking D (more residency, smaller batch) —
bandwidth cannot be added in software. The backend raises
`MeasurementUnavailable` with the implied cap rather than silently running at a
different R_B than requested.

**R_C and R_B are coupled.** D grows as R_C falls, so R_B is a function of R_C.
They decouple only via the throttle, and only downward: at each R_C the reachable
set is R_B ≤ B_hw·T_comp/D(R_C). The probe region is a **triangle**, not the
rectangle a naive grid sweep assumes.

Run `python experiments/plan_reachable_grid.py` (no GPU needed) to see which
(batch, seq) admit all three regimes on your platform before booking GPU time.
For Llama-3 8B on an A100, batch=8/seq=2048 does **not** — batch=1/seq=512 does.

### Correction to R_B (θ_B = 1.0, not 0.40)

`config.THETA_B` is 1.0. Some manuscript results still use 0.40. The change is not a
recalibration; the old definition was not well posed.

The paper defines R_B = B_slow·Δt/D with Δt the **step duration**. Steady state
requires the link to deliver D bytes every Δt seconds, so D/Δt ≤ B_slow and
therefore:

```
R_B = B_slow·Δt/D ≥ 1        always, on any hardware
```

R_B < 1 is not an operating point but a diverging queue: the backlog grows, Δt
stretches, and R_B returns to 1. R_B = 1 is an attractor, not a boundary. Under
that definition θ_B = 0.40 named a state the system cannot occupy, and every
I/O-limited operating point in the manuscript (R_B = 0.55, 0.18, 0.22, 0.26) sits
in infeasible territory. The one self-consistent R_B in the paper is the
Introduction's 1.61.

Substituting Δt := T_comp makes it an **overlap ratio**:

```
R_B = B_slow·T_comp/D = T_comp / T_transfer
```

well posed, reachable on both sides, and with the boundary at exactly **1.0** —
where transfer stops fitting behind compute. That value is derived, not fitted,
and is identical across platforms. `ratios.predict_theta_b()`, which fitted θ_B
per platform from free parameters α_wb and α_q, is correspondingly obsolete and
now raises.

θ_C = 0.50 is **not** repaired by this and remains unvalidated — its
majority-eviction "derivation" restates the definition rather than arguing
physics, and the structural lower bound implies sharpness S ≤ 1 at R_C = 0.50,
so the bound cannot produce an abrupt transition there. See `orion/config.py`.

### Other platforms

Backend adapters for these platforms are **not implemented**. Each needs a class
exposing `measure(r_c, r_b) -> LatencyRecord` and `warmup(sec)`, registered via
`HardwareProfiler.register_backend()`; `experiments/cuda_backend.py` is the
reference implementation to port.

| Platform | Software stack | Status |
|----------|---------------|--------|
| Google TPU v4 | JAX 0.4.26 | adapter not implemented |
| AWS Inferentia2 | NeuronSDK 2.18 | adapter not implemented |
| AMD MI250 | ROCm 6.0 | adapter not implemented; ROCm exposes rocm-smi and rocprof rather than NVML/CUPTI |
| Intel Xeon + Optane-PMem | PyTorch 2.4 | adapter not implemented; DRAM as fast tier, no GPU HBM |

Platform-specific backend adapters are not included (hardware unavailable for open release). Implement `measure(r_c, r_b) → LatencyRecord` and register via `HardwareProfiler.register_backend()`.

---

## Programmatic API

### Compute R_C, R_B from hardware and model parameters

```python
from orion import compute_rc, compute_rb, classify_regime

r_c = compute_rc(c_fast_bytes=80e9, w_bytes=41.8e9)   # 1.91
r_b = compute_rb(b_slow_bps=16.1e9, t_comp_s=0.120, d_bytes=1.2e9)  # 1.61
regime = classify_regime(r_c, r_b)                    # COORDINATION_DOMINATED
```

### Structural lower bound

```python
from orion import compute_lower_bound, A100_80GB
from orion.ratios import OperatingPoint

op = OperatingPoint(r_c=1.91, r_b=1.61, w_gb=41.8, d_gb=1.2, b_slow_gbs=16.1)
lb = compute_lower_bound(op, rho=A100_80GB.rho, t_measured_s=0.93)
print(f"Achievability τ = {lb.achievability:.2f}")   # 0.86–0.92 range
print(f"Residual (T_sync) = {lb.residual_fraction:.1%}")
```

### Runtime regime classifier

```python
from orion import RegimeClassifier, RuntimeFeatures

clf = RegimeClassifier()
# Features collected from DMA logs + CUPTI counters:
features = RuntimeFeatures(
    swap_to_comp_ratio=0.08,
    cache_hit_rate=0.82,
    dma_utilisation=0.45,
    r_c=1.91, r_b=1.61,      # pass ratios directly for highest accuracy
)
regime = clf.classify(features)
print(f"Regime: {regime.name}")
print(f"Latency: {clf.mean_latency_ms:.3f} ms")    # < 0.1 ms
```

### Regime-aware orchestrator

```python
from orion import OrionOrchestrator, OrionMode

# Orion_Full: hardware + 100-ms classifier loop
orc = OrionOrchestrator(mode=OrionMode.FULL)
orc.start()

for request in workload:
    latency = orc.infer(request, r_c=current_r_c, r_b=current_r_b)

orc.stop()
print(f"Mean latency: {orc.metrics.mean_latency_ms:.1f} ms")
print(f"Regime switches: {orc.metrics.n_regime_switches}")
```

---

## Data availability

Raw measurement logs use JSONL format with one entry per 10-second window. The planned Zenodo record is:

**DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**  
*(Placeholder; replace with the deposited DOI before submission or publication.)*

Each JSONL entry contains: `timestamp`, `r_c`, `r_b`, `t_comp`, `t_mem`, `t_swap`, `t_sync`, `t_wall`, `platform`, `model`, `sweep_id`, `window_id`.

Load with the provided utility:

```python
from utils.logging import load_jsonl
records = list(load_jsonl("path/to/sweep_00.jsonl"))
```

---

## Reproducing sharpness coefficient S

```python
from orion.lower_bound import sharpness_coefficient

# T_total measurements across R_C grid near θ_C = 0.50
t_values = [0.95, 0.93, 0.92, 0.88, 0.72, 0.54, 0.52, 0.51]
r_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
S = sharpness_coefficient(t_values, r_values)
print(f"S = {S:.2f}   (paper: 4.12 at θ_C, S* = 2.0)")
```

---

## Statistical tests

```python
from utils.stats import bootstrap_ci, wilcoxon_one_sided, SweepStats

# Aggregate across 5 sweeps
stats = SweepStats(sweep_means=[0.93, 0.91, 0.94, 0.92, 0.93])
print(stats)
# SweepStats(mean=0.9260, 95% CI=[0.9100, 0.9400], n_eff≈150)

# Wilcoxon test: H0: S ≤ S* = 2.0
s_estimates = [4.05, 4.12, 3.98, 4.18, 4.09]
p = wilcoxon_one_sided(s_estimates, null_value=2.0)
print(f"p = {p:.4f}   (paper: p < 0.05 near boundaries)")
```

---

## Plotting (Figure 2)

```bash
# Interactive:
python experiments/reproduce_figure2.py

# PDF at 300 DPI:
python experiments/reproduce_figure2.py --save-pdf fig2.pdf
```

For custom plots, export CSV and use your preferred tool:

```bash
python experiments/reproduce_figure2.py --save-csv
# Files: results/figure2/fig2a_rc_sweep.csv
#        results/figure2/fig2b_decomposition.csv
#        results/figure2/fig2c_rb_sweep.csv
```

---

## Citation

```bibtex
@article{orion2026nature,
  title   = {Hierarchical memory orchestration in {AI} inference exhibits
             intrinsic regime-dependent limits},
  author  = {Lim, Geunsik and others},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Manuscript in preparation},
}
```
