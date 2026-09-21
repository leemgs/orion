# External referee simulation — *Nature Communications*

> Working document, **not for submission**. A simulated three-referee panel plus
> handling-editor summary, written to anticipate the decision and prioritise the
> revision that would actually move the manuscript across the acceptance bar.
> Companion to `nc_referee_report.md` (single-referee self-review),
> `reviewer_readiness_plan.md`, and `preregistration.md` (the decisive
> experiment this panel asks for). Nothing here claims a result the manuscript
> does not support.

## Manuscript under review
"A two-ratio framework separates residency and transfer overlap in
hierarchical-memory inference." Contribution as stated by the authors:
(i) two independently measurable dimensionless ratios — residency
`R_C = C_fast / W` and overlap `R_B = T_comp / T_transfer` (non-circular by
construction); (ii) a conservative max-form service-time lower bound and two
normalized floors, `f_nonresident >= max(0, 1 - R_C)` and
`f_exposed >= max(0, 1 - R_B)`; (iii) an auditable reporting contract; and
(iv) proof-of-concept measurements on one CPU and three accelerators (T4, A100,
TPU v5e). The authors explicitly disclaim production, strategy-ranking, energy,
per-device-boundary, and population-level hardware claims.

---

## Handling editor — summary assessment

The manuscript is honest, reproducible, and methodologically disciplined. The
decisive question for *Nature Communications* is not correctness but **advance
size and breadth of interest**. As written, the self-declared contribution is a
reporting discipline plus proof-of-concept microbenchmarks, and nearly every
substantive empirical claim is disclaimed. That combination risks an editorial
decline before external review on significance grounds, and — if sent out —
would require new experiments (below) to clear the bar. The work is an excellent
fit for a strong systems/measurement venue (MLSys, SC, ISCA, IISWC, IMC) in its
current form. A single decisive, preregistered positive result (Reviewer 2,
Major 1) is the most direct route to competitiveness at this journal.

**Provisional recommendation as-is: Reject on significance/advance grounds
(encourage transfer); revision path to Accept requires new empirical work, not
text changes.**

---

## Referee 1 — computer architecture / memory systems

**Recommendation: Reject (transfer), or Major Revision requiring new experiments.**

Strengths. The non-circular framing of `R_B` (separately timed compute and
transfer intervals rather than total step time in both metric and outcome) is a
genuine and useful clarification. The max-form bound is correctly conservative,
and the paper is careful to keep `theta_C = 0.5` a declared convention and
`theta_B = 1` a definitional equality.

Major concerns.
1. **Conceptual novelty is incremental.** `R_C` is working-set/cache residency
   (Denning); `R_B` is the compute-vs-transfer overlap implicit in Roofline and
   in async-copy analysis. Naming and jointly plotting them is a clarification,
   not the conceptual step a *Nature Communications* article is expected to make.
2. **The floors are near-tautological and, as noted by the authors, `R_C` does
   not determine `D_nr` without an access-coverage model** — the very quantity
   needed to instantiate the storage floor is outside the framework. The added
   value of the max-form bound over a single-tier roofline bound should be stated
   explicitly (the revised text now does this: it adds an explicit non-resident
   residency floor `rho * D_nr` and uses sustained rather than peak bandwidth).
3. **No independent test of the boundaries.** The operating labels are computed
   from the ratios, so observing them is not an independent test (acknowledged).
   A convincing test needs an outcome *not* definitionally tied to the ratio
   crossing its coordinate.
4. **A known measurement artifact remains in the headline table.** The
   fully-resident point sits above the trend on the A100/TPU and is diagnosed
   post hoc as a compile/launch artifact; a warm-up fix was added but the runs
   were not repeated, so Table 2's resident denominator is the artifact-affected
   point. Re-measure with the warm-up in place before relying on those ratios.

---

## Referee 2 — ML systems / inference serving

**Recommendation: Reject as-is; strong potential after one decisive experiment.**

1. **The paper's own impact thesis is not tested.** The cover letter and
   Discussion argue that inconsistent offloading/swapping/caching results stem
   from an uncontrolled operating regime. That is the compelling story, and no
   experiment demonstrates it. Show, on a real serving stack (vLLM /
   DeepSpeed-Inference / FlexGen) with a real model, that regime membership
   predicts which orchestration policy wins — including a case where the ranking
   of two policies changes across a predicted coordinate. This is the single
   highest-value addition; `preregistration.md` specifies it.
2. **No real workload.** `d = 2048`, 24 synthetic layers do not represent LLM
   inference (paged KV cache, continuous batching, IO-aware attention, PCIe/
   NVLink offloading). On fast devices, launch/compile overhead dominates the
   signal (the authors' own finding), so the regime of interest is not exercised.
3. **"Broad interest" is asserted, not shown.** The claim that one audit applies
   to weights, KV caches, embeddings, feature maps, and retrieval indices is
   attractive but undemonstrated. At least two model families (e.g. an
   autoregressive LLM plus retrieval or vision) should show the framework making
   a useful, checkable prediction.
4. The 70B worked example is arithmetic illustration, not evidence (correctly
   labelled; keep it that way).

---

## Referee 3 — measurement methodology / statistics

**Recommendation: Major Revision.** Statistics and reproducibility are a genuine
strength; the weight of the conclusion is light relative to the venue.

1. **Effect sizes are weak and confounded.** The 1.23–2.83x ratios average over
   points spanning several hierarchy transitions (the authors state the
   pointer-chain endpoints cross multiple transitions), on `n = 1` machine, with
   within-run windows rather than independent replicates. `R_C` is therefore not
   a clean single knob.
2. **Uncertainty quantification was thin.** The revision now reports 95%
   percentile bootstrap CIs for the mean of each selected pointer-chain point,
   derived from the committed seven trials, and states plainly that this is
   within-machine trial variability, not a population interval. This is the
   right honest step for the CPU probe; the accelerator runs still lack retained
   per-window records and so carry no CI. A competitive submission needs multiple
   machines, bootstrap CIs on the accelerator effect sizes, and preregistration.
3. **Strength (retain).** Open code, committed raw JSONL, a diagnosed artifact,
   candid limitations, and a clean measured/simulated separation are exemplary.

Minor.
- Keep title/abstract from implying an empirically discovered limit (`theta_C`
  convention, `theta_B` definition) — currently well controlled.
- Confirm authorship/CRediT against the Acknowledgements contributors (the
  manuscript already carries a CRediT-style Author Contributions statement).

---

## Consolidated requirements to reach Accept

Ranked by leverage. Items 1–3 are the ones that change the recommendation.

1. **Decisive, preregistered strategy-selection result** on a real serving stack
   and real model, across >= 2 accelerator architectures, including at least one
   ranking change across a predicted coordinate. (`preregistration.md`.)
2. **Boundary test decoupled from label definition**, using raw event traces on
   >= 2 architectures.
3. **Held-out predictive accuracy**: predict regime / best strategy a priori and
   report accuracy with bootstrap CIs, not post hoc fitting.
4. **Workload scale and breadth**: escape launch overhead; >= 2 model families.
5. **Statistics**: multiple machines, bootstrap CIs on all reported effect
   sizes, significance tests, preregistration.

## What would change the recommendation
If Items 1–3 were demonstrated on >= 2 real accelerators with real workloads and
proper statistics, this becomes a solid, potentially high-impact contribution
appropriate for the journal. Until then it is a framework-and-microbenchmark
note better served by a strong systems/measurement venue — the same conclusion
the authors reached in their own internal self-review.
