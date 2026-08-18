# Nature Communications — Submission Compliance Report

**Manuscript:** *Hierarchical memory orchestration in AI inference exhibits intrinsic regime-dependent limits* (ORION)
**Target journal:** Nature Communications (Article)
**Report date:** 2026-08-18
**Build:** `./run.sh --submission` → `main.pdf` (19 pp.) + `supplementary.pdf` (15 pp.)

This report audits the manuscript against the Nature Communications *Article*
author guidelines (nature.com/ncomms/submit/article). It records what is
already compliant, what was fixed in this revision, and what the author must
still do outside the LaTeX source before submitting.

---

## 1. Formatting compliance (hard limits)

| # | Requirement (NComms Article) | Limit | This manuscript | Status |
|---|------------------------------|-------|-----------------|--------|
| 1 | Abstract, unreferenced, single paragraph | ≤ 150 words | ~146 words | ✅ |
| 2 | Main text (Intro + Results + Discussion) | ≤ 5,000 words | ~3,354 words | ✅ |
| 3 | Display items (figures + tables) | ≤ 10 | 6 (2 figures + 4 tables) | ✅ |
| 4 | References | guideline ≤ ~70 | 25 cited | ✅ |
| 5 | Title | ≤ 15 words, avoid abbreviations/punctuation | 10 words (contains "AI") | ⚠️ minor |
| 6 | Methods placed after main text, not in word count | required | Yes (starred section) | ✅ |
| 7 | Continuous line numbers on the review copy | required | Added (`lineno`) | ✅ (new) |

Word counts measured on the source with captions, tables, and equations
excluded (Nature Communications excludes abstract, Methods, references, and
figure legends from the main-text count).

## 2. Required editorial sections

| Section | Required for an Article | Present | Notes |
|---------|-------------------------|---------|-------|
| Data availability statement | Yes | ✅ | Standalone statement |
| Code availability statement | Yes (custom code) | ✅ | GitHub + simulation scripts |
| Author Contributions (CRediT) | Yes | ✅ | CRediT roles listed for the sole author |
| Competing Interests | Yes | ✅ | "declares no competing interests" |
| Acknowledgements | Optional | ✅ | Present |
| Reporting Summary | Life-sciences only | n/a | Not a life-sciences study |

## 3. What changed in this revision

1. **Journal retargeting** — `main.tex` and `supplementary.tex` headers now
   name Nature Communications as the primary target and record the Article
   limits above. The broad-significance framing (a general principle rather
   than a systems-engineering optimisation) is retained, as it suits the
   multidisciplinary Nature Communications readership.
2. **Continuous line numbers** — added the `lineno` package with an
   amsmath-compatible patch so numbering runs through the display equations
   (verified on Eqs. 1–4). This satisfies the Nature Communications review
   requirement. Comment out `\linenumbers` for a clean camera-ready copy.
3. **Rebuilt `main.pdf`** in submission mode (reference URLs stripped).

## 4. Action items the author must complete outside the LaTeX source

These are **not** LaTeX/formatting issues; they are submission-system and
policy items that only the author can satisfy.

- **Cover letter** — write a Nature Communications cover letter (the existing
  `submission/nmi_presubmission_inquiry.md` is NMI-specific). State the
  advance/significance for a broad audience and why it fits Nature Communications.
- **Open-access APC** — Nature Communications is fully open access; the
  ~US$7,350 APC (or an institutional Read & Publish waiver) applies on
  acceptance. There is no subscription (free) route.
- **Data at review time** — the manuscript claims real-hardware measurements on
  five platforms but defers raw logs to "Zenodo upon acceptance / on request."
  Nature Communications increasingly expects data to be accessible **during**
  review. Strongly consider depositing the pre-processed traces (and, ideally,
  the raw `.jsonl` logs) with a DOI **before** submission and citing it in the
  Data availability statement.
- **Title advisory** — the title contains the abbreviation "AI." It is widely
  understood and unlikely to block, but Nature Communications prefers titles
  without abbreviations; a spelled-out alternative is worth considering.
- **Reporting/analysis** — no life-sciences reporting summary is needed, but be
  ready to complete any general editorial checklist the submission system
  presents.
- **ORCID / authorship** — single-author submission is permitted; ensure the
  ORCID and affiliation in `005_author_nature.tex` are current.

## 5. Publishability assessment (honest)

**Format readiness: ready to submit.** No blocking compliance issues remain;
all hard limits and required sections are satisfied and the PDF builds cleanly.

**Editorial outlook (content, not format):** a credible but not guaranteed
submission. Strengths: a clear, testable central claim (regime-dependent
limits with strategy *inversion*), an analytical lower bound plus empirical
calibration across five platforms, generality across LLM/vision/RAG and
scientific-computing workloads, and solid statistics (bootstrap CIs, Wilcoxon,
power analysis). Main substantive risks a reviewer/editor is likely to probe:
(1) verifiability of the real-hardware five-platform data at review time given
that the open repository ships a *simulated* backend — mitigate per §4;
(2) desk-rejection on scope/novelty — the roofline/working-set lineage is
well known, so the "phase-like regime" contribution must read as a genuine
general principle, which the current framing supports but reviewers will test.
Nature Communications desk-rejects a majority of submissions, so a strong
cover letter and review-time data availability materially improve the odds.
