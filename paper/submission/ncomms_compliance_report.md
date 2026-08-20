# Nature Communications pre-submission audit

## Scientific reporting

- The title and abstract state the limited contribution: an analytical
  formulation and single-CPU proof of concept.
- The Methods identify trial counts, grids, timing procedures, seeds,
  summaries, and the absence of hypothesis testing.
- Limitations explicitly exclude accelerator, production-workload, energy,
  strategy-ranking, and classifier-generalisation claims.
- Every reported number is traceable to committed JSON/JSONL data; generated
  LaTeX macros prevent manual transcription drift.
- The CPU summary records processor, visible CPU count, operating-system
  kernel, Python, NumPy, compiler, timer, seed, cache-size source, and trials.

## Availability statements

The main manuscript contains standalone Data availability and Code availability
sections. Data are available during review in the public repository rather than
being promised only after acceptance. Simulated results are explicitly excluded
from the evidence.

## Submission-package checks still requiring the corresponding author

- Verify author name, affiliation, ORCID, email, contribution statement, and
  competing-interest declaration.
- Select the appropriate article type and provide editor/reviewer suggestions
  in the journal portal.
- Confirm the journal's current formatting and repository policies in the live
  submission portal; those policies can change.
- Consider depositing the exact release in a persistent archive and inserting
  its DOI before submission.

No editorial outcome can be guaranteed. The remaining primary scientific
limitation is the single-platform proof-of-concept scope, which is stated
without qualification or hidden placeholder claims.
