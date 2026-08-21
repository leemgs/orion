# ORION

**Two dimensionless ratios define operational limits in hierarchical-memory inference**

ORION is an analytical formulation and open measurement protocol for describing
hierarchical-memory operating points with two dimensionless ratios:
**R_C** (fast-memory capacity / active working set) and **R_B** (isolated compute
time / compulsory-transfer time). The submitted evidence is deliberately
limited to two proof-of-concept measurements on one CPU. It does not establish
accelerator generality, policy rankings, or sharp regime transitions.

Target journal: **Nature Communications**. See [paper/README.md](paper/README.md)
for the build, evidence boundary, and final author checks.

---

## Repository Structure

The repository is organized into three areas for easier maintenance.

| Directory | Contents | Documentation |
|-----------|----------|---------------|
| [`code/`](code/) | ORION measurement framework and scripts for reproducing the paper's results (Python) | [code/README.md](code/README.md) |
| [`paper/`](paper/) | Nature Communications manuscript, supplementary information, figures, references, and build scripts | [paper/README.md](paper/README.md) |
| [`ppt/`](ppt/) | Presentation materials (Korean and English slides, including the NCS presentation) | — |

---

## Quick Start

- **Paper build instructions, submission strategy, and journal priorities** → [paper/README.md](paper/README.md)
- **Result reproduction and code usage** → [code/README.md](code/README.md)

## Current submission state

The review package is evidence-aligned: the manuscript reports only the
committed CPU pointer-chain and layered-matrix measurements, while the regime
map is explicitly analytical. Reported values are exported from the committed
JSON/JSONL records into `paper/generated_results.tex`; optional CUDA, XLA, and
simulated paths are not presented as completed experiments.

Run the release checks from the repository root:

```bash
python -m pytest code/tests -q
python paper/check_submission.py
cd paper && ./run.sh
```

The build produces the line-numbered review manuscript (`paper/main.pdf`) and
standalone Supplementary Information (`paper/supplementary.pdf`). Exact CPU
timings are machine-dependent; regenerating the checked-in numerical macros is
only appropriate after intentionally replacing the corresponding measurement
records.

## Evidence and availability

Raw records and machine-readable summaries are version controlled in
`code/results/cpu_probe/` and `code/results/colab_probe/`. The public repository
is the review-time data and code location; no unassigned archive DOI is claimed.
Before journal upload, the corresponding author should archive the exact
submitted commit in a persistent repository if required and add the resulting
DOI consistently to the manuscript and documentation.

## Submission responsibility

Automated checks cannot establish novelty or guarantee acceptance. The
corresponding author must verify author metadata, declarations, article type,
and the journal's current portal requirements, and must visually inspect both
PDFs before submission. The principal known limitation remains the deliberately
stated single-platform, synthetic proof-of-concept evidence boundary.
