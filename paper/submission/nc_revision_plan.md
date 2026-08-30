# Nature Communications — staged preregistered revision plan

This plan converts the current proof-of-concept into a manuscript that can
support significant, general conclusions. It is the plan referenced in the
cover letter. Each stage names (a) the experiment, (b) the reviewer concern it
defends (cross-referenced to `reviewer_readiness_plan.md`), and (c) the specific
manuscript claim it would unlock. Analysis code, protocols, and integration are
assistant-supportable; running the accelerators is the author's task.

## Gate 0 — decide the target before spending accelerator budget

The current evidence fits a specialised systems venue (MLSys, IEEE TPDS/TC)
without further experiments. Committing to Nature Communications means committing
to Stages 1–3 below *before* submission, because the framework's value claim is
predictive and is presently untested. Recommendation: obtain an editor
pre-submission read on scope fit (cover letter already invites this) before the
full campaign.

## Stage 1 — real-workload substance (defends R1, R7; unlocks generality)

- Replace the d=2048 synthetic matmul with token-generation on a production
  serving stack (vLLM primary; FlexGen or DeepSpeed-Inference as a second stack
  to show the audit is framework-agnostic).
- Models spanning three workload classes: autoregressive LLM (Llama-3 8B and
  70B, Mixtral), vision / vision-language (ViT, BLIP-2), retrieval-augmented
  (FAISS index + generator).
- Deliverable: end-to-end latency with the four-term decomposition
  (T_comp, T_mem, T_swap, T_sync) at declared operating points; per-workload
  (R_C, R_B) computed from datasheets and confirmed by measurement.

## Stage 2 — the two results that beat "roofline renaming"

### 2a. Strategy-ranking inversion (defends R2 — highest priority; unlocks novelty)
- Compare >=3 orchestration policies (e.g. full-offload, layer-pinned, KV-paged)
  at >=3 operating points crossing all three regimes.
- Success criterion, prespecified: the policy that wins in the
  coordination-dominated regime is beaten in the I/O- or capacity-limited regime
  (a documented rank inversion), reported with bootstrap CIs.

### 2b. Held-out predictive test (defends R6; unlocks "predictive", not just "descriptive")
- For N real configurations, precompute (R_C, R_B) and predict the regime and
  the fastest of the candidate policies *before* measuring.
- Report hit rate (e.g. correct fastest-policy prediction in k of N) with a
  naive-baseline comparison.

## Stage 3 — statistical rigour and independent boundary evidence

- **Multi-machine replication (R5):** >=2–3 independent machines (or cloud
  instances) per accelerator class; bootstrap CIs; a significance test on each
  directional prediction; preregistered operating-point grid.
- **Independent boundary validation (R3):** show that an outcome metric *not*
  definitionally tied to R_B (achievable speedup, marginal latency, or energy)
  bends or breaks near R_B=1, on a workload large enough that the boundary is
  physically reachable and launch/compile overhead no longer dominates (fixes
  the A100/TPU artifact noted in Results).

## Stage 4 — reproducibility and motivation closure

- **Raw traces (R9):** deposit per-window / per-op raw device-event timestamps
  and a pinned container under a Zenodo DOI; cite the DOI in Data availability.
- **Energy (R8):** NVML power integration for throughput-per-watt by regime on
  >=2 accelerators, closing the loop with the introduction's energy motivation;
  if infeasible, downscope the energy framing in the introduction.
- **Prior-result re-reading (R10):** reproduce two previously conflicting
  offloading/serving results and show they sit in different regimes — direct
  empirical proof that the reporting contract resolves inconsistency.

## Minimal set to convert desk-reject risk into external review

Stages **1 + 2a + 2b + 3(multi-machine) + 4(raw traces)**. With these, the core
"insufficient empirical substance" objection is largely defended; R3-boundary,
R7-generality breadth, R8-energy, and R10-reinterpretation strengthen a revision.

## Division of labour

- Assistant: measurement protocols and scripts (vLLM/FlexGen integration,
  inversion and held-out-prediction designs, preregistration document, trace
  export), and integration of confirmed measurements into tables/figures/text.
- Author: execution on real accelerators, DOI deposition, portal submission.
