# Draft cover letter — *Nature*

Dear Editors,

Please consider the manuscript "Separating memory capacity from data movement
in neural network inference" for publication as an Article in *Nature*.

**The problem.** Running large neural networks is now limited less by
arithmetic than by how data move between fast and slow memory, and inference
dominates the recurring cost and energy of deployed models. A large literature
on offloading, swapping, caching and compression reports gains that are
notoriously hard to compare: the operating condition of each measurement is
usually unreported, and popular bandwidth-pressure ratios are circular, placing
total step time in both the metric and the outcome it is meant to explain.

**What the manuscript contributes.** It proposes a measurement standard rather
than a benchmark result. Any operating point is described by two independently
measurable, scale-invariant ratios: fast-memory residency and compute–transfer
overlap. The overlap ratio is non-circular by construction, because compute and
transfer are timed separately. A conservative service-time bound yields two
floors that no placement or scheduling policy can remove, and a minimal set of
reported fields lets any reader recompute both ratios and audit the denominator
choices behind a published speed-up. The same audit applies without change to
weights, key–value caches, embeddings and retrieval indices, so one reporting
layer can sit beneath otherwise incomparable systems studies.

**What the evidence does and does not show.** Proof-of-concept probes on a
server processor and on three accelerators from two vendors (NVIDIA T4 and
A100, Google TPU v5e) show that the protocol is executable and that latency
rises in the predicted direction as residency falls; one accelerator sweep
reaches both sides of the derived overlap boundary. We state plainly that this
is not independent validation of the boundaries, and we make no
production-model, energy, policy-ranking, per-device-boundary or
population-level hardware claim. 95% bootstrap intervals for each processor
point's mean accompany the processor table, and the public repository keeps
measured output separate from a clearly labelled simulated backend.

**The decisive test, and a question for the editors.** We recognise that a
*Nature* Article must show broad consequence, and that the framework's
predictive value — whether the operating point identifies the fastest
orchestration policy on production systems — is experimental. That test is
preregistered, with its analysis locked in the repository, and measurement
harnesses for three production serving stacks (vLLM, DeepSpeed-Inference and
FlexGen) are released; no results from them are reported in this manuscript.
Given that the present contribution is methodological, we would value the
editors' view on whether it falls within *Nature*'s scope as submitted, or
whether the editors would prefer to see the preregistered production-scale
results first.

All reported data and analysis code are available during review at
https://github.com/leemgs/orion. The work is original, is not under
consideration elsewhere, and the author has approved the submission.

Sincerely,

Geunsik Lim
