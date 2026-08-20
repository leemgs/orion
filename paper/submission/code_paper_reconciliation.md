# Code–manuscript reconciliation

## Submission state

The manuscript now uses the same definitions as the released code throughout:

- `R_C = C_fast / W`;
- `R_B = T_comp / T_transfer`;
- `theta_C = 0.50`, an explicitly declared majority-residency convention; and
- `theta_B = 1.0`, the derived compute–transfer overlap boundary.

The analytical regime-map asset is generated from these constants by
`code/experiments/reproduce_regime_map.py`.

## Evidence boundary

The submitted claims are restricted to the compiled CPU pointer-chain
measurement and the separate CPU harness check whose raw JSONL records and JSON
summaries are committed under `code/results/`. Simulated
backend outputs are not empirical evidence. CUDA and XLA paths are available as
future measurement protocols, but the manuscript does not claim that GPU, TPU,
Inferentia, MI250, or Optane experiments were performed.

Earlier draft-only claims about five-platform calibration, strategy inversion,
classifier accuracy, energy, and cross-workload generality have been removed
from the submission sources and supplementary information because supporting
raw traces are not present.

## Checks before upload

1. Build `main.tex` and `supplementary.tex` from a clean checkout.
2. Confirm that the PDFs contain no draft notes or unresolved references.
3. Upload the exact repository commit cited in the submission system.
4. Archive the committed CPU data and code with a DOI if the journal requests a
persistent repository before review.
