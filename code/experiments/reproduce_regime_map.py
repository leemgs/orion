#!/usr/bin/env python3
"""Generate the analytical ORION regime map used by the manuscript.

The plot is a schematic of the definitions in ``orion.config``; it does not
contain, or imply, empirical accelerator measurements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orion.config import THETA_B, THETA_C


def plot(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    x_max, y_max = 2.0, 2.0
    colours = {"capacity": "#f4cccc", "coordination": "#d9ead3", "io": "#cfeaf3"}

    # Capacity is the first decision in the classifier. The I/O boundary is
    # therefore displayed only on the R_C >= theta_C half-plane.
    ax.axvspan(0, THETA_C, color=colours["capacity"], zorder=0)
    ax.fill_between([THETA_C, x_max], 0, THETA_B, color=colours["io"], zorder=0)
    ax.fill_between([THETA_C, x_max], THETA_B, y_max,
                    color=colours["coordination"], zorder=0)
    ax.axvline(THETA_C, color="#d9534f", linewidth=2)
    ax.plot([THETA_C, x_max], [THETA_B, THETA_B], color="#337ab7", linewidth=2)

    box = dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82)
    ax.text(THETA_C / 2, 1.1, "Capacity-limited\n$R_C < \\theta_C$",
            ha="center", va="center", fontsize=14, color="#c9302c", bbox=box)
    ax.text(1.25, 1.48,
            "Coordination-dominated\n$R_C \\geq \\theta_C,\\ R_B \\geq \\theta_B$",
            ha="center", va="center", fontsize=14, color="#238b22", bbox=box)
    ax.text(1.25, 0.48, "I/O-limited\n$R_C \\geq \\theta_C,\\ R_B < \\theta_B$",
            ha="center", va="center", fontsize=14, color="#2171b5", bbox=box)

    ax.set(xlim=(0, x_max), ylim=(0, y_max),
           xlabel=r"$R_C=C_{\mathrm{fast}}/W$ (fast-memory residency ratio)",
           ylabel=r"$R_B=T_{\mathrm{comp}}/T_{\mathrm{transfer}}$ (overlap ratio)",
           title="Operational regime map for hierarchical-memory inference")
    ax.legend(handles=[
        Patch(facecolor=colours["capacity"], label="Capacity-limited"),
        Patch(facecolor=colours["coordination"], label="Coordination-dominated"),
        Patch(facecolor=colours["io"], label="I/O-limited"),
    ], loc="upper right")
    ax.text(THETA_C + 0.02, 1.97, rf"$\theta_C={THETA_C:.2f}$", color="#c9302c", va="top")
    ax.text(1.97, THETA_B + 0.03, rf"$\theta_B={THETA_B:.1f}$", color="#2171b5", ha="right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        "paper/figures/orion_regime_map.png")
    args = parser.parse_args()
    plot(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
