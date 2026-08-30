# Draft cover letter

Dear Editors,

Please consider our manuscript, "A two-ratio framework separates residency and
transfer overlap in hierarchical-memory inference," for publication in *Nature
Communications*.

Inference, not training, is now the dominant recurring cost and energy draw of
deployed foundation models, and its performance is increasingly set by how data
moves across a hierarchical memory system rather than by arithmetic. Yet
reported gains from offloading, swapping, caching, and compression are
notoriously inconsistent between studies. A recurring and under-recognised
cause is that the operating regime is uncontrolled, and that popular
bandwidth-pressure ratios are circular—they place total step time in both the
metric and the outcome being explained. Our manuscript addresses this at the
level of measurement principle. We separate two independently measurable
quantities—fast-memory residency ($R_C$) and compute–transfer overlap
($R_B$)—into two dimensionless, scale-invariant ratios; we prove that the
overlap ratio is non-circular; we derive a conservative service-time lower
bound and, from it, an exposed-transfer floor and a non-resident-storage floor
that are fixed by definition rather than fitted; and we specify an auditable
reporting contract that makes otherwise incomparable systems studies directly
comparable. The generality of this discipline—one audit applies equally to
weights, KV caches, embeddings, feature maps, and retrieval indices—is what we
believe fits the multidisciplinary readership of Nature Communications.

We have deliberately matched every empirical claim to the evidence at hand.
Beyond a compiled dependent-load CPU experiment, we exercise the framework on
three accelerators of two vendors (NVIDIA T4 and A100, and Google TPU v5e). The
residency direction is descriptively consistent on all three, and the T4 sweep
reaches both sides of the derived overlap coordinate. Because the operating
labels follow deductively from the classification rule, we do **not** present
this as independent boundary validation, and we make no production-model,
energy, strategy-ranking, per-device-boundary, or population-level hardware
claim. These boundaries are stated plainly in the abstract, Results, Methods,
and Discussion, and the public repository separates measured from simulated
output. We regard this scoping as a feature of the work's rigour, not a gap
concealed.

We recognise that the decisive tests of the framework's predictive value are
experimental, and we are committed to delivering them during revision on a
preregistered plan: (i) end-to-end four-term latency decomposition on a
production serving stack (vLLM / FlexGen) with real models spanning
autoregressive LLM, vision, and retrieval-augmented workloads; (ii) a
strategy-ranking-inversion test across the three regimes; (iii) a held-out
predictive test in which $(R_C,R_B)$ computed from datasheets forecasts the
fastest policy before measurement; (iv) multi-machine replication with bootstrap
confidence intervals; and (v) deposition of raw device-event traces under a
persistent DOI. We would welcome the editors' early guidance on whether this
staged scope fits Nature Communications or a more specialised sister journal
before we commit the full accelerator campaign.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
