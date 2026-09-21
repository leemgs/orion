# Preregistration — decisive test of the two-ratio framework

> Preregistered analysis plan for the confirmatory experiment requested by the
> referee panel (`nc_external_referee_simulation.md`, Items 1–3). It fixes the
> hypotheses, design, data schema, analysis, and decision rules **before** data
> collection so that the test is confirmatory, not exploratory. The analysis is
> implemented and locked in `code/experiments/analyze_prereg.py`; running it on
> real records is the only step left. This document claims **no result** — it
> defines how a result would be judged, including how the framework is rejected.

Status: **planned / not yet executed.** No accelerator campaign has been run.
The manuscript continues to report only the committed proof-of-concept evidence.

---

## 1. Motivation and scope

The submitted manuscript establishes that residency (`R_C`) and overlap (`R_B`)
are separately measurable and directionally consistent on small probes, but it
does **not** show that the operating point predicts a *decision* — which
orchestration policy to use. This preregistration specifies the confirmatory
test of exactly that claim, because it is the advance the target venue requires
and the hypothesis the Discussion currently leaves open.

## 2. Hypotheses

Let an operating point be `x = (R_C, R_B)` with the released classifier label
`L(x) in {capacity-limited, I/O-limited, coordination-dominated}`. Let a
*policy* be a fixed orchestration strategy (e.g. `full-offload`,
`layer-pinned`, `paged-KV-offload`, `no-offload`) run without changing any other
workload parameter.

- **H1 (residency effect).** At fixed access pattern, decreasing `R_C` below
  `theta_C` increases exposed memory-access cost when it increases `D_nr`.
- **H2 (exposed-transfer floor).** When `R_C >= theta_C`, points with `R_B < 1`
  expose at least `max(0, T_transfer - T_comp)` even under best-effort overlap.
- **H3 (decision relevance — the decisive claim).** The label `L(x)`, computed
  *a priori* from datasheet/measured inputs, predicts the latency-minimising
  policy on held-out operating points better than a label-blind baseline.
- **H4 (ranking change).** There exists at least one policy pair whose latency
  ranking changes sign across a predicted coordinate (`R_C = theta_C` or
  `R_B = 1`), with non-overlapping bootstrap CIs on both sides.

H3 is the confirmatory primary hypothesis. H4 is a secondary, stronger
demonstration. H1–H2 are directional replications of the submitted probes at
production scale.

## 3. Design

- **Hardware:** >= 2 accelerator architectures, >= 2 physical machines each
  (e.g. NVIDIA A100 80GB and NVIDIA H100 or AMD MI250; a TPU arm optional and
  end-to-end only, given XLA lazy timing). Record full node/driver provenance.
- **Workloads:** >= 2 model families to support the breadth claim, e.g.
  (a) an autoregressive LLM at a size that forces host offload on the smaller
  device (Llama-class), with paged KV cache and continuous batching; and
  (b) a retrieval-augmented or vision pipeline. Real frameworks (vLLM,
  DeepSpeed-Inference, FlexGen), not synthetic layer stacks; workloads sized to
  escape launch/compile overhead (warm-up iterations discarded).
- **Operating-point grid:** preregistered grids bracketing `R_C = 0.5` and
  `R_B = 1`, each point realised by construction (device memory cap, offload
  fraction, batch/sequence length), with continuous `(R_C, R_B)` recorded.
- **Policies:** the fixed set above, each run on every grid point.
- **Replication:** >= 5 independent runs per (machine, point, policy);
  compute/transfer timed separately (CUDA events); raw per-window and per-run
  records **retained** (the gap that prevented accelerator CIs in the submission).
- **Split:** operating points partitioned into a preregistered train and
  held-out set before any latency is inspected; the classifier constants are
  fixed (not fitted on evaluation data).

## 4. Record schema (locked)

One JSON object per (machine, model, point, policy, run), JSONL:

```json
{
  "machine_id": "a100-node-3", "arch": "A100-80GB", "model": "llama-…",
  "framework": "vllm", "policy": "paged-KV-offload", "split": "heldout",
  "R_C": 0.31, "R_B": 0.72, "label": "capacity-limited",
  "C_fast_bytes": …, "W_bytes": …, "D_bytes": …, "D_nr_bytes": …,
  "B_slow_bytes_per_s": …, "T_comp_s": …, "T_transfer_s": …,
  "T_total_s": 0.0412, "run_idx": 3, "windows": 10,
  "source": "measured", "provenance": "…"
}
```

`source` must be `"measured"`; records tagged `"simulated"` or `"dryrun"` are
rejected by the analysis as evidence (consistent with the repository's
measured/simulated separation). Every field in the manuscript's reporting
contract (Table 3) is mandatory.

**Collection tooling.** `code/experiments/prereg_harvest.py` builds this grid,
fixes the train/held-out split from a seed before any latency is seen, emits
this schema, and validates it. It has no path that turns synthetic numbers into
`source="measured"`: real records come only from wiring a serving-stack
measurement into its integration seam (compute and transfer timed separately,
per `experiments/cuda_backend.py`), and a `--dry-run` emits rejected `"dryrun"`
placeholders so the pipeline can be checked without hardware.

## 5. Analysis (locked in `analyze_prereg.py`)

Computed only on the held-out split, exactly as implemented:

1. **Best-policy prediction accuracy (H3).** For each held-out point, the
   predicted best policy is the a-priori rule keyed on `label`; the observed
   best policy is the one with lowest mean `T_total`. Report accuracy and its
   95% bootstrap CI (resampling points), against a label-blind baseline
   (single globally-best policy). Primary success: accuracy CI lower bound
   exceeds the baseline accuracy.
2. **Ranking-change detection (H4).** For each policy pair and each predicted
   coordinate, flag a sign change in mean-latency difference across the boundary
   with non-overlapping per-side bootstrap CIs.
3. **Directional replications (H1, H2).** Effect-size ratios with bootstrap CIs;
   exposed-interval check `T_total >= max(T_comp, T_transfer)` per point.
4. **Reporting.** All continuous `(R_C, R_B)` retained; categorical summaries
   repeated under alternative `theta_C in {0.4, 0.5, 0.6}` to show label
   stability, not a tuned cut-off.

## 6. Decision rules and falsification

- **Framework supported (toward the decisive claim):** H3 primary success
  **and** H1 replicated, on >= 2 architectures.
- **Framework rejected / substantially revised:** H3 accuracy CI overlaps the
  label-blind baseline, or H1 fails at production scale, or H2's exposed interval
  is not observed where `R_B < 1`. A null result is a reportable outcome and
  will be reported.
- **Inconclusive:** effect present but CIs too wide given the replication
  budget; report as such, do not re-slice to significance.

No optional stopping: the replication budget in §3 is fixed in advance. No
post-hoc grid changes, threshold tuning, or metric substitution.

## 7. What executing this changes

Success on H3 (+H1) converts the manuscript's central open hypothesis into a
demonstrated, preregistered result across real hardware and workloads — the
advance the referee panel identifies as the route to acceptance. Until it is
executed, the manuscript's scope and claims are unchanged.
