# Internal self-review — simulated *Nature Communications* referee report

> Working document, **not for submission**. A pre-submission self-review written
> in the voice of a Nature Communications referee, to anticipate the decision
> and prioritise revisions. Companion to `reviewer_readiness_plan.md`.

**Recommendation:** Reject for *Nature Communications* on scope/advance grounds
(not a defect-driven rejection); encourage transfer to a specialised systems or
measurement venue. The work is careful, honest, and reproducible, but does not
meet the bar for a substantial, broadly significant advance.

## Summary
The manuscript describes a hierarchical-memory operating point with two
dimensionless ratios — residency `R_C = C_fast/W` and overlap
`R_B = T_comp/T_transfer` (non-circular by construction) — a conservative
max-form service-time lower bound, three operational labels (`theta_C = 0.5`
declared convention, `theta_B = 1` definitional), and an auditable reporting
contract. Evidence: a CPU pointer-chase probe and layered-matrix probes on three
accelerators (T4, A100, TPU v5e). The authors explicitly disclaim production,
strategy-ranking, energy, per-device-boundary, and population-level claims.

## Assessment against journal criteria
1. **Significance / advance — below threshold (decisive).** By its own framing
   the contribution is a "reporting discipline" plus proof of concept; the
   significant result (regime membership predicting a *reversal in orchestration
   strategy ranking*) is motivation, not demonstrated.
2. **Novelty — incremental.** Recombines Roofline, working-set/thrashing, and
   queueing overlap. The non-circular `R_B` and joint coordinate are a useful
   clarification, but the paper concedes it does not predict speed-up magnitude
   or policy ranking.
3. **Evidence — insufficient and partly tautological.** Synthetic micro-probes
   (`d=2048`), weak/variable effect sizes (1.23–2.83x), and labels computed from
   the ratios (so "observing" them is not an independent test — acknowledged).
   No real model/serving, no strategy comparison, no cross-machine statistics.
4. **Rigor / reproducibility / honesty — strength.** Open code, committed data,
   a diagnosed measurement artifact, candid limitations. Commendable, but rigor
   about a modest result.

## Major points (needed to be competitive at this tier)
1. Demonstrate strategy-ranking inversion on a real serving stack + real model.
2. Independent test of the boundaries (an outcome not definitionally tied to the
   ratio changing at the predicted coordinate).
3. Predictive hold-out: predict regime/best-strategy a priori, report accuracy.
4. Real workload scale/breadth (LLM + vision + retrieval; escape launch overhead).
5. Statistics: multiple machines, bootstrap CIs, significance tests; preregister.

## Minor points
- Keep title/abstract from implying an empirically discovered "limit"
  (`theta_C` is a convention; `theta_B` a definition).
- Clarify what the max-form bound adds beyond a Roofline-style bound.
- The 70B worked example is arithmetic illustration, not evidence (label as such).
- Confirm authorship criteria vs the Acknowledgements contributors.

## What would change the recommendation
If Major Points 1–3 were demonstrated on >= 2 real accelerators with real
workloads and proper statistics, this becomes a solid, potentially high-impact
contribution suitable for the journal. As a framework-and-microbenchmark note it
is better served by a strong systems/measurement venue.
