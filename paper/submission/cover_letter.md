# Draft cover letter

Dear Editors,

Please consider our manuscript, “Two dimensionless ratios define operational
limits in hierarchical-memory inference,” for publication in *Nature
Communications*.

The manuscript separates fast-memory residency from compute–transfer overlap
using two dimensionless ratios. It derives a conservative service-time bound,
states an auditable three-label operating convention, and releases the code and
raw records for the measurements. The formulation also identifies why using
total step duration inside a transfer-pressure ratio is circular.

We have deliberately limited the empirical claim to a proof of concept. The
evidence is a compiled dependent-load experiment on one virtualised Intel Xeon
CPU and layered-matrix runs on three accelerators of two vendors (NVIDIA T4 and
A100, and Google TPU v5e). The residency direction holds on all three; the T4
also crosses the derived overlap boundary that the other sweeps did not
reliably resolve. We do not claim production-model, energy, strategy-ranking,
per-device-boundary, or population-level hardware validation. The manuscript
states these limitations in the abstract, Results,
Methods, and Discussion, and the public repository separates measured from
simulated output.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
