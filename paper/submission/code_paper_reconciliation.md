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

The submitted claims cover the compiled CPU pointer-chain measurement, the CPU
harness check, and per-point summary files from NVIDIA T4, NVIDIA A100, and
Google TPU v5e runs committed under `code/results/`. Simulated backend outputs
are not empirical evidence. The accelerator files are summary-level records,
not raw device-event traces or independent hardware replicates. No Inferentia,
MI250, Optane, energy, or production-model experiment is claimed.

Earlier draft-only claims about five-platform calibration, strategy inversion,
classifier accuracy, energy, and cross-workload generality have been removed
because supporting measurements are not present.

## Checks before upload

1. Build `main.tex` and `supplementary.tex` from a clean checkout.
2. Confirm that the PDFs contain no draft notes or unresolved references.
3. Upload the exact repository commit cited in the submission system.
4. Archive the committed CPU data and code with a DOI if the journal requests a
persistent repository before review.
