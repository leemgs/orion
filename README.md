# ORION

**Regime-Dependent Limits of Hierarchical Memory Orchestration in Large-Scale AI Inference**
*Hierarchical memory orchestration in AI inference exhibits intrinsic regime-dependent limits*

ORION demonstrates that hierarchical memory orchestration in large-scale AI inference has **fundamentally different limits across hardware and workload regimes**. The optimal orchestration strategy is determined by only two dimensionless ratios: **R_C** (compute-to-memory ratio) and **R_B** (bandwidth-to-capacity ratio). In certain regimes, the optimal strategy **inverts**.

Target journal: **Nature Machine Intelligence** (confirmed first choice) — fallback: Nature Computational Science → Nature Communications → npj. See §§8–9 of [paper/README.md](paper/README.md) for the detailed submission strategy.

---

## Repository Structure

The repository is organized into three areas for easier maintenance.

| Directory | Contents | Documentation |
|-----------|----------|---------------|
| [`code/`](code/) | ORION measurement framework and scripts for reproducing the paper's results (Python) | [code/README.md](code/README.md) |
| [`paper/`](paper/) | LaTeX manuscript (IEEE and Nature templates), figures, references, and build scripts | [paper/README.md](paper/README.md) |
| [`ppt/`](ppt/) | Presentation materials (Korean and English slides, including the NCS presentation) | — |

---

## Quick Start

- **Paper build instructions, submission strategy, and journal priorities** → [paper/README.md](paper/README.md)
- **Result reproduction and code usage** → [code/README.md](code/README.md)
