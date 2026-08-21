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
