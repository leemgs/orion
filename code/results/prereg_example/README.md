# Example records — the format you must hand back

`records_EXAMPLE.jsonl` shows the **exact shape** of the file you produce on real
hardware for the preregistered experiment (`paper/submission/preregistration.md`).

**This is a template, not data.** Every row has `"source": "example"`, so
`analyze_prereg.py` refuses it as evidence. Your real file must set
`"source": "measured"` and contain numbers from real runs — nothing here is a
measurement, and the latencies are illustrative only.

## One JSON object per line = one (machine, model, operating point, policy, run)

The example has 2 operating points × 4 policies × 2 runs = 16 rows. Your real
file needs **≥ 5 runs per (point, policy)** and the full preregistered grid.

## Field dictionary

| Field | Type | What to put | Who fills it |
|-------|------|-------------|--------------|
| `machine_id` | str | Physical machine name, e.g. `"a100-node-1"` | you |
| `arch` | str | Accelerator arch, e.g. `"A100-80GB"` | you |
| `model` | str | Model, e.g. `"llama-3-8b"` | you |
| `framework` | str | Serving stack: `"vllm"`, `"deepspeed"`, `"flexgen"` | you |
| `policy` | str | Orchestration policy for this run (see below) | you |
| `split` | str | `"train"` or `"heldout"` — **use what the harvest tool assigned; do not choose per run** | tool |
| `R_C` | float | Residency `C_fast/W` at this point | you (by construction) |
| `R_B` | float | Overlap `T_comp/T_transfer` at this point | you (by construction) |
| `label` | str | `"capacity-limited"` / `"I/O-limited"` / `"coordination-dominated"` — from the classifier, **not** hand-set | tool |
| `C_fast_bytes` | int | Usable fast-tier bytes (capped device memory), not datasheet | you |
| `W_bytes` | int | Active working set = weights + activations + KV | you |
| `D_bytes` | int | Compulsory cross-tier bytes per step | you |
| `D_nr_bytes` | int | Non-resident bytes compulsorily accessed per step | you |
| `B_slow_bytes_per_s` | int | Sustained bandwidth on the limiting link (measured, not peak) | you |
| `T_comp_s` | float | Isolated compute time (transfer disabled / timed separately) | you |
| `T_transfer_s` | float | Isolated transfer time = `D/B_slow` | you |
| `T_total_s` | float | Measured end-to-end step latency — **the outcome** | you |
| `run_idx` | int | 0,1,2,… independent repeat of this (point, policy) | you/tool |
| `windows` | int | Timing windows averaged within the run | you |
| `source` | str | **Must be `"measured"`** for real data (`"example"`/`"dryrun"`/`"simulated"` are rejected) | you |
| `provenance` | str | One line: node, driver, date, how measured | you |

Consistency the reviewers will check: `R_C ≈ C_fast_bytes/W_bytes`,
`T_transfer_s ≈ D_bytes/B_slow_bytes_per_s`, `R_B ≈ T_comp_s/T_transfer_s`, and
`T_total_s ≥ max(T_comp_s, T_transfer_s)`.

## Policies (the fixed set)

`no-offload`, `layer-pinned`, `paged-KV-offload`, `full-offload`. Run **every**
policy at **every** operating point so the analysis can find the fastest one and
detect ranking changes.

## Don't hand-build this file

`code/experiments/prereg_harvest.py` builds the grid, assigns the split, and
writes this schema for you — you only wire the real measurement into its
`measure_operating_point` seam. See `../../../paper/submission/handoff_checklist.md`.
