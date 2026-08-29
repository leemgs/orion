# ORION manuscript

This directory contains the Nature Communications submission sources for
**“A two-ratio framework separates residency and transfer overlap in
hierarchical-memory inference.”** The submission is intentionally scoped to an analytical
formulation plus a proof of concept measured on one CPU and three accelerators
of two vendors (NVIDIA Tesla T4, NVIDIA A100, Google TPU v5e). It does not claim
per-device boundary values or cross-workload validation.

## Submission files

- `main.tex` — main manuscript entry point.
- `supplementary.tex` — standalone Supplementary Information.
- `section/001_title.tex`, `005_author_nature.tex`,
  `006_abstract_nature.tex`, `010_introduction.tex`,
  `025_results_ncs.tex`, `060_discussion.tex`, `070_methods.tex`,
  `090_ack.tex`, and `095_reference_nature.tex` — main-manuscript components.
- `section/900_appendix.tex` — supplementary derivations and reproduction
  instructions.
- `figures/orion_regime_map.png` — analytical, not empirical, regime map.
- `check_submission.py` — guard against known draft-only claims and markers.
- `submission/` — reconciliation, pre-submission audit, and cover-letter draft.

The CPU data and code live under `../code/`. In particular:

- `../code/results/cpu_probe/` contains the pointer-chain records and summary;
- `../code/results/colab_probe/` contains the layered-matrix CPU records and
  summary; and
- `../code/experiments/reproduce_regime_map.py` regenerates the analytical map.

## Build

A TeX distribution with `pdflatex` and `bibtex` is required.

```bash
cd paper
python check_submission.py
./run.sh
```

`./run.sh --submission` removes ordinary bibliography URL fields for the
submission rendering and restores the bibliography afterward. It builds both
`main.pdf` and `supplementary.pdf`.

## Reproduce reported results

From the repository root:

```bash
python code/experiments/cpu_hierarchy_probe.py
python code/experiments/colab_regime_measurement.py --backend numpy --quick
python code/experiments/export_paper_results.py
python code/experiments/reproduce_regime_map.py
```

The first two commands execute real CPU measurements. Exact timings vary by
machine. The third exports reported values from JSON into LaTeX macros. The
fourth generates an analytical schematic and does not execute an experiment.

## Evidence boundary

The repository also includes a separately labelled simulated analysis path,
which is useful for software testing but is not evidence for claims in the
submitted manuscript. The committed CUDA/XLA summaries support only the
accelerator proof-of-concept reported in the paper. Do not reintroduce earlier
draft claims about five-platform calibration, strategy inversion, classifier
accuracy, or accelerator energy unless the corresponding raw traces, analysis,
and Methods are added and independently checked.

## Final author checks

Before uploading to the journal portal:

1. Run `python check_submission.py` and build both PDFs from a clean checkout.
2. Inspect every PDF page for unresolved references, overfull content, and
   author metadata.
3. Verify the title, author name, ORCID, affiliation, email, contributions,
   acknowledgements, and competing-interest declaration.
4. Confirm current article-type, formatting, data-repository, and reporting
   requirements in the live Nature Communications submission portal.
5. Preferably archive the exact code/data release in a persistent repository
   and add its DOI to the Data availability statement.

No checklist can guarantee editorial acceptance. The manuscript states its
small synthetic, non-replicated scope so that editors and reviewers can
evaluate the evidence actually available.
