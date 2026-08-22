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
CPU and a layered-matrix run on a single NVIDIA Tesla T4 GPU; on the GPU both
falsifiable predictions hold—latency rises as residency falls below the
majority-residency convention, and the I/O-limited label appears exactly at the
derived overlap boundary that the CPU harness could not resolve. We do not
claim multi-accelerator, production-model, energy, or strategy-ranking
validation. The manuscript states these limitations in the abstract, Results,
Methods, and Discussion, and the public repository separates measured from
simulated output.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
