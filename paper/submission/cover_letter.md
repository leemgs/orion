# Draft cover letter

Dear Editors,

Please consider our manuscript, "A two-ratio framework separates residency and
transfer overlap in hierarchical-memory inference," for *Nature Communications*.

**What this paper contributes is a measurement standard, not a benchmark
result.** Inference now dominates the recurring cost and energy of deployed
foundation models, and its speed is increasingly set by how data moves across a
hierarchical memory system rather than by arithmetic. Yet reported gains from
offloading, swapping, caching, and compression are notoriously hard to compare
across studies. A recurring, under-recognised cause is methodological: the
operating point is left unreported, and popular bandwidth-pressure ratios are
circular—they place total step time in both the metric and the outcome being
explained. Our primary contribution is a remedy at the level of measurement
principle—an auditable **reporting contract** that makes otherwise incomparable
systems studies directly comparable.

Concretely, the contract rests on two independently measurable, scale-invariant
coordinates—fast-memory residency $R_C$ and compute–transfer overlap $R_B$. We
prove $R_B$ is non-circular (compute and transfer are timed separately), derive
a conservative service-time lower bound, and from it two floors that are fixed
by definition rather than fitted: a non-resident-storage floor $\max(0,1-R_C)$
and an exposed-transfer floor $\max(0,1-R_B)$. The reusable output is the
minimal field set that lets any reader recompute $(R_C,R_B)$ and audit the
denominator choices behind a reported speed-up. Its breadth is what we believe
suits the multidisciplinary readership: the same audit applies without change to
weights, key–value caches, embeddings, feature maps, and retrieval indices, so
one reporting layer sits under otherwise incomparable offloading, compression,
scheduling, and memory-pooling work.

**We have deliberately matched every empirical claim to the evidence at hand,
and we present the measurements as a demonstration that the protocol is
executable and directionally sound—not as its validation.** Beyond a compiled
dependent-load CPU experiment, we exercise the framework on three accelerators of
two vendors (NVIDIA T4 and A100, and Google TPU v5e). The residency direction is
descriptively consistent on all three, and the T4 sweep reaches both sides of the
derived overlap coordinate. Because the operating labels follow deductively from
the classification rule, we explicitly do **not** present this as independent
boundary validation, and we make no production-model, energy, strategy-ranking,
per-device-boundary, or population-level hardware claim. These boundaries are
stated plainly in the abstract, Results, Methods, and Discussion; per-trial
bootstrap intervals accompany the CPU points; and the public repository keeps
measured output separate from a clearly labelled simulated backend. We regard
this scoping as a feature of the work's rigour, not a gap concealed.

We also recognise that the decisive tests of the framework's *predictive* value
are experimental, and we have preregistered them so they are confirmatory rather
than exploratory (analysis locked in the repository). On a preregistered plan we
will, during revision: (i) time the four-term latency decomposition end-to-end on
production serving stacks (vLLM / DeepSpeed / FlexGen) with real models spanning
autoregressive LLM, vision, and retrieval-augmented workloads; (ii) test whether
regime membership predicts the fastest orchestration policy a priori on a
held-out grid, including whether a policy pair reverses its ranking across a
predicted coordinate; (iii) replicate across multiple machines with bootstrap
confidence intervals; and (iv) deposit raw device-event traces under a persistent
DOI. Given that our present contribution is methodological, we would welcome the
editors' early guidance on whether this staged scope fits *Nature Communications*
or a more specialised sister journal before we commit the full accelerator
campaign.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
