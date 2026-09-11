# Measurement protocol — strategy-ranking inversion (Major Point 1)

Preregisterable design for the single result most likely to move the paper from
"reporting discipline" to a substantial advance: **the ranking of orchestration
strategies inverts across regimes.** Author-run on real hardware; the assistant
can turn this into a runnable harness once a target stack is chosen.

## Falsifiable hypothesis (preregister before measuring)
For a fixed model and hardware, and for orchestration strategies
S = {S1, S2, S3, ...}, the latency-optimal strategy **depends on the regime**:
there exist operating points A and B such that
`argmin_S T(S, A) != argmin_S T(S, B)`,
i.e. a strategy that is best in one regime is not best in another
(strong form: it becomes the *worst*). Null hypothesis: one strategy dominates
in all regimes (no inversion).

## Factors
- **Hardware (>= 2 accelerators, ideally 2 vendors):** e.g. A100-80GB and one of
  {L4, H100, MI300, TPU v5e}. Record device, driver, CUDA/XLA, power cap.
- **Model (>= 1 real, ideally 2):** e.g. Llama-3 8B and 70B (fp16), plus one
  non-LLM (ViT-H or a RAG retriever) for breadth.
- **Operating points (>= 3, one per regime):** choose configs whose *measured*
  `(R_C, R_B)` land in capacity-limited (`R_C<0.5`), I/O-limited
  (`R_C>=0.5, R_B<1`), and coordination-dominated (`R_C>=0.5, R_B>=1`). Vary via
  HBM cap, batch size, sequence length, and sustained-bandwidth throttle.
- **Strategies (>= 3 real policies):**
  1. Baseline resident / no offload optimisation (e.g. vLLM default).
  2. Compute–transfer overlap / pipelined offload (FlexGen- or
     DeepSpeed-Inference-style prefetch with CUDA streams).
  3. Working-set reduction (weight quantisation or KV-cache compression).
  4. (optional) Fine-grained swap / selective layer offload.

## What to measure, per (hardware, model, operating point, strategy)
- `T_total` end-to-end per-token (or per-step) latency; also throughput.
- The four decomposition terms via CUDA events / CUPTI (`T_comp`, `T_mem`,
  `T_swap`, `T_sync`), isolated `T_comp` and `T_transfer` for `R_B`.
- `C_fast`, `W`, `D`, `D_nr`, sustained `B_slow` (the reporting-contract fields).
- NVML power integration -> energy/token (also answers the energy critique, R8).

## Statistics (preregister)
- >= 5 independent runs per cell; **>= 2 independent machines** per accelerator
  class. Report bootstrap 95% CIs.
- Primary test: for each hardware x model, a rank-reversal test across the >= 3
  operating points (e.g. Friedman + posthoc, or a preregistered contrast on the
  best-strategy identity). Report effect sizes, not just significance.
- Fix seeds; discard a warm-up window (already in the harness); no threshold
  tuning on evaluation data.

## Pass/fail (what "success" looks like)
- **Inversion confirmed** if the best strategy differs across >= 2 operating
  points with non-overlapping CIs on the deciding pairwise gap, replicated on
  >= 2 machines. Report the crossover magnitude (e.g. "S2 is -X% vs baseline in
  coordination-dominated but +Y% in I/O-limited").
- **Predictive corollary (Major Point 3):** compute `(R_C,R_B)` a priori for
  held-out configs, predict the winning strategy, report accuracy (k/N).

## Deliverables to fold into the manuscript
- A regime x strategy latency table with CIs and the inversion highlighted.
- A figure: latency (or speed-up vs baseline) per strategy across the regime
  axis, showing lines crossing.
- Raw event traces deposited on Zenodo with a DOI (addresses R9).

## Notes / honesty
- The synthetic layered-matrix harness cannot demonstrate true async overlap on
  CPU; genuine overlap requires CUDA streams on real GPUs, which is why this is
  an author-run, real-hardware protocol rather than a CPU-validated script.
- Report continuous `(R_C, R_B)`; the categorical labels are indexing only.
