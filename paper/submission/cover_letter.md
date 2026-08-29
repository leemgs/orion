# Draft cover letter

Dear Editors,

Please consider our manuscript, “Two dimensionless ratios define operational
limits in hierarchical-memory inference,” for publication in *Nature
Communications*.

Memory-bound inference is now the dominant cost of deploying foundation models,
yet reported gains from offloading, swapping, caching, and compression are
notoriously inconsistent across studies. A recurring reason is that the operating
regime is not controlled, and that popular bandwidth-pressure ratios are
circular—they place total step time in both the metric and the outcome. Our
manuscript addresses this directly. It separates two independently measurable
quantities—fast-memory residency and compute–transfer overlap—into two
dimensionless ratios; proves a non-circular definition; derives a conservative
service-time lower bound and, from it, an overlap boundary that is fixed by
definition rather than fitted; and states an auditable reporting contract that
makes otherwise incomparable systems studies comparable. We see the principal
contribution as this measurement discipline for a problem of broad and growing
importance, delivered with open code and raw records.

We were deliberate about matching the empirical claim to the evidence. Beyond a
compiled dependent-load CPU experiment, we test the framework's two falsifiable
predictions on three accelerators of two vendors (NVIDIA T4 and A100, Google TPU
v5e): the residency prediction holds on all three, and the T4 crosses the derived
overlap boundary. We do not claim production-model, energy, strategy-ranking,
per-device-boundary, or population-level hardware validation, and we say so
plainly in the abstract, Results, Methods, and Discussion; the repository
separates measured from simulated output. We would welcome the editors' guidance
on whether this scope fits Nature Communications or a more specialised sister
journal, and we are prepared to extend the accelerator campaign during revision.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
