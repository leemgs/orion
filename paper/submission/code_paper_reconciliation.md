# Code ↔ Manuscript Reconciliation — ORION

**Date:** 2026-08-18
**Scope:** Reconcile the public code (`code/`) with the manuscript (`paper/`)
and establish the status of the empirical ("real-hardware, five-platform")
claims ahead of a Nature Communications submission.

---

## 0. Executive summary

Two independent problems block submission. Neither is a formatting issue.

1. **No measurement data exists, and none can be produced in the current
   environment.** The repository contains zero measurement files
   (`.jsonl`/`.csv`/`.npy`), and the execution environment is a 4-core CPU
   container with **no NVIDIA/AMD GPU, no TPU, no Inferentia2, no Optane-PMem**
   and no `torch`. The manuscript's quantitative spine — θ boundaries with
   standard deviations, strategy-inversion magnitudes, per-platform shifts,
   classifier accuracy — is therefore **unsupported**. It cannot be
   substantiated here, and the simulated backend must not be used as a
   substitute (see §2).

2. **The public code encodes a *different, corrected theory* than the
   manuscript, and running it *contradicts* the manuscript.** The definitions
   of the control ratio `R_B` and the boundary `θ_B` in `code/` deliberately
   diverge from the paper; the reproduction script prints a strategy-inversion
   table that is the **reverse** of the paper's Table 2 (see §1, §2).

The honest consequence: the manuscript cannot be made both self-consistent and
truthful by text editing alone. It requires (a) the real measurement campaign,
and (b) author decisions on which theory (paper's or code's) is correct.

---

## 1. Divergences (code is the corrected analysis; paper is stale)

The code files carry extensive, self-authored notes stating that the paper's
math is wrong and that the code is the correction. Verbatim pointers:

### 1.1 Control ratio `R_B`

| | Manuscript | Code |
|---|---|---|
| Definition | `R_B = B_slow·Δt / D` (`section/025_results_ncs.tex:78`) | `R_B = B_slow·T_comp / D = T_comp / T_transfer` (`code/orion/ratios.py:17`) |
| Meaning | "fraction of sustained transfer-demand that bandwidth can serve" | overlap ratio: how much per-step transfer hides behind compute |

`code/orion/ratios.py:24` states the paper's definition is **degenerate**:
with `Δt` the step duration, `B_slow·Δt ≥ D` always holds, so `R_B ≥ 1` on any
hardware, and `R_B < 1` "is not an operating point but a diverging queue."
Under the paper's own definition, an I/O-limited regime at `R_B < 0.40`
**cannot occur**.

### 1.2 Bandwidth boundary `θ_B`

| Manuscript | Code |
|---|---|
| `θ_B ≈ 0.40`, "empirically calibrated across five platforms (s.d. ≤ 0.03)" (`section/025_results_ncs.tex:124,127`) | `θ_B = 1.00`, **derived** — "the point where transfer stops fitting behind compute" (`code/orion/config.py:6,48`) |

The running code classifies against `θ_B = 1.0` (verified — see §2), not 0.40.

### 1.3 Capacity boundary `θ_C`

| Manuscript | Code |
|---|---|
| `θ_C = 0.50` "follows analytically from the majority-eviction condition … independently of any platform constant" (`section/025_results_ncs.tex:135–139`) | `θ_C = 0.50` is an "empirical convention," **not** analytically forced (`code/orion/config.py:28–32`) |

`code/orion/config.py:28` argues the majority-eviction condition is "a statement
about miss volume, not latency"; miss volume `W(1−R_C)` is linear and continuous
in `R_C`, so "nothing distinguishes 1/2 as critical," and the structural lower
bound's sharpness `S = ρW·R_C/T` is monotonic and `≤ 1` at `R_C = 0.50` — i.e.
the bound does **not** create a sharp transition there. This directly undercuts
the abstract's claim that the lower bound proves "transitions are inevitable."

---

## 2. Runtime evidence (what the public code actually produces)

Environment: `pip install numpy scipy`; CPU-only; `--mode simulate` (the only
mode runnable without hardware).

**`python experiments/run_regime_sweep.py --mode simulate`** — the R_B sweep
flips regime at **R_B = 1.0** (I/O-limited below 1.0, coordination-dominated at
≥ 1.0), confirming the code uses `θ_B = 1.0`, not the paper's 0.40.

**`python experiments/reproduce_table2.py`** prints:

```
Method            CAPACITY LIMITED   COORDINATION DOMINATED   IO LIMITED
flexgen                 -1.5%              +24.0%  ←!            -8.0%
deepspeed               -2.0%              +18.0%  ←!            -5.0%
orion                   -4.0%              -20.5%  ←!           -10.5% ←!
```

This is the **opposite** of the manuscript: the paper says orchestration
(FlexGen/DeepSpeed) *helps* in the coordination-dominated regime and *worsens*
latency by 8–12% only in the I/O-limited regime; the script shows FlexGen/
DeepSpeed *worse* (+24%, +18%) in coordination-dominated and *better* in
I/O-limited. The script's own footer line ("FlexGen −24% coord-dominated →
+8–12% I/O-limited") also contradicts the table it just printed.

`code/experiments/simulated_backend.py` (docstring) states the cause plainly:
its coefficients "were written by hand … never fitted to measurements, on A100
or anywhere else," "do not agree with the manuscript," and the previously
claimed Zenodo traces / calibrated noise "do not exist … it is a guess." Its
numbers must not be reported as results.

> The generated `code/results/*.jsonl` from these runs were **deleted**, not
> committed: they are simulator smoke-test output, not measurements.

---

## 3. What reconciliation requires (and who must do it)

Editing text cannot close these gaps. Required actions, in order:

1. **Author decision on the correct theory.** Adopt the code's corrected
   `R_B = T_comp/T_transfer`, `θ_B = 1.0` (derived), and `θ_C = 0.50` as an
   operational convention — and correspondingly **soften the abstract's
   "phase transition … inevitable" claim** to a finite-size crossover whose
   sharpness is bounded (which the Discussion already half-concedes). This is a
   scientific reframing the author must own; it is not a mechanical edit.

2. **Real measurement campaign.** Run `experiments/cuda_backend.py` (and the
   platform equivalents) on the actual A100 / TPU v4 / Inferentia2 / MI250 /
   Optane-PMem hardware to produce the raw `.jsonl` traces. Every quantitative
   value in `section/025_results_ncs.tex` (θ ± s.d., `S = 4.12 ± 0.31`, 41.2%,
   the ±0.03/±0.05 per-platform shifts, Table 2 magnitudes, 93.4% classifier
   accuracy) must be recomputed from those traces. If some platforms are not
   actually available, the claims must be **scoped down** to the platforms that
   were measured.

3. **Regenerate Figure 1** (`figures/orion_regime_map.png`) once (1)–(2) are
   settled: the current PNG has `θ_B = 0.40` baked in and no generator script
   exists in the repo, so the figure and any corrected text will otherwise
   disagree. Figure 2 (`reproduce_figure2.py`) must likewise be rebuilt from
   real traces, not the simulated backend.

4. **Deposit the traces on Zenodo before submission** (see
   `submission/zenodo_deposit.md`) and update the Data availability statement
   to a concrete DOI, replacing "upon acceptance / on request."

Until (1)–(4) are done, the code and manuscript remain in conflict and the
empirical claims remain unbacked. No text-only change in this repository can
resolve that.

---

## 3a. Applied in this revision (paper → code, definitions only)

Per the decision to align the manuscript to the corrected code, the following
**definitional/derivational** edits were applied to the main-build sources
(the manuscript builds cleanly, 20 pp.):

- `R_B` redefined as `T_comp/T_transfer` (overlap ratio) in
  `025_results_ncs.tex` (Eq. 2) and `070_methods.tex`; the degeneracy of the
  old `B_slow·Δt/D` form is stated explicitly.
- `θ_B` set to `1.0` (**derived** overlap boundary) and `θ_C = 0.50`
  reframed as a **majority-residency convention** (not analytically forced) in
  Results, Methods, Introduction, Discussion, and the abstract.
- The abstract's "structural lower bounds proving transitions are inevitable"
  softened to bounds on irreducible costs with boundaries derived from the
  ratios; the capacity "transition" is described as a finite-size crossover.
- Numeric regime-condition captions (`R_B ≥ 0.4`) symbolised to `R_B ≥ θ_B`;
  claims that hardware "shifts θ_B" rewritten as operating points moving
  relative to the fixed, derived boundary.

**Not changed (still pending real measurement — not fabricated):** every
empirical magnitude (`S = 4.12 ± 0.31`, 41.2%, 8–14%, 78–84%, per-platform
values, the specific `R_B = 0.75/0.55/0.18/…` operating points, Table 2
inversion magnitudes, 93.4% classifier accuracy) and the **empirical panels of
Figure 2**. These
were produced under the superseded definition and must be recomputed from the
measurement campaign. The analytical regime-map asset has now been regenerated
from the released `THETA_C=0.50` and `THETA_B=1.0` constants; it contains no
empirical measurements and is reproducible with
`experiments/reproduce_regime_map.py`.

## 4. Note for the submission decision

Formatting is submission-ready (`submission/ncomms_compliance_report.md`), but
formatting is not the binding constraint. Submitting with the current data gap
and the code↔paper contradiction in a public repository risks not only desk
rejection but post-review integrity concerns, because a referee who runs the
linked code obtains results that contradict the paper. The measurement campaign
is the gating item.
