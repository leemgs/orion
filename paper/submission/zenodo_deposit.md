# Zenodo Data Deposit — ORION

Ready-to-use metadata and description text for depositing the ORION datasets on
**Zenodo** ahead of the Nature Communications submission. Nature Communications
increasingly expects the underlying data to be accessible **during** peer
review, so this deposit is made **before** submission and its DOI is cited from
the manuscript's *Data availability* statement.

> **Reserve the DOI first.** In the Zenodo upload form, click **"Reserve DOI"**
> to obtain the final DOI *before* publishing the record. Insert that DOI into
> the manuscript (see §5 below), then publish the Zenodo record so the DOI
> resolves. Keep the deposit in **Restricted**/embargoed access during review if
> you prefer, and lift the embargo on acceptance.

---

## 1. Zenodo upload-form fields

| Field | Value |
|-------|-------|
| **Upload type** | Dataset |
| **Title** | ORION: regime-classified latency traces and classifier training data for hierarchical memory orchestration in large-scale AI inference |
| **Authors** | Lim, Geunsik — Sungkyunkwan University (SKKU); ORCID `0000-0003-1845-7132` |
| **Description** | See §2 (paste the HTML/plain block below) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Language** | English (eng) |
| **Version** | `v1.0.0` |
| **Keywords** | see §3 |
| **Related/alternate identifiers** | see §4 |
| **Access** | Open (or Restricted/embargoed until acceptance, then Open) |
| **Publication date** | date of deposit |

---

## 2. Description (paste into the "Description" box)

> This dataset accompanies the manuscript *"Intrinsic regime-dependent limits
> govern hierarchical memory orchestration in large-scale AI inference"* (ORION).
>
> Large-scale AI inference is increasingly limited not by computation but by how
> data is coordinated across a hierarchical memory system spanning device
> memory, host memory, and secondary storage. ORION characterises any inference
> system by two dimensionless ratios — the fast-memory residency ratio (R_C) and
> the transfer-pressure ratio (R_B) — and identifies three operationally distinct
> regimes (capacity-limited, coordination-dominated, and I/O-limited) separated by
> abrupt, phase-like transitions across which the ranking of optimisation
> strategies inverts.
>
> The archive contains the empirical data supporting every reported figure and
> table:
>
> - **Pre-processed latency traces** for all five hardware platforms — NVIDIA
>   A100 (80 GB), Google TPU v4, AWS Inferentia2, AMD MI250, and Intel
>   Xeon + Optane-PMem — collected on identical R_C/R_B probing sweeps. No
>   platform results are simulated or emulated.
> - **Raw measurement logs** in JSON Lines (`.jsonl`) format, one record per
>   10-second measurement window, with per-window latency decomposition and
>   hardware counters.
> - **Regime-classifier training data** used to fit the depth-3 CART classifier
>   that maps (R_C, R_B) to a regime label, together with the calibrated
>   boundary values (θ_C ≈ 0.50, θ_B ≈ 0.40).
> - **Derived tables/figures** and the scripts’ expected outputs for
>   reproduction checks.
>
> The ORION measurement framework and reproduction scripts (Python) are openly
> available at https://github.com/leemgs/orion. CPU-only simulation reproduces
> the qualitative regime structure; all quantitative values in the manuscript
> derive from the real-hardware measurements archived here.
>
> Fields reporting counters that a given platform cannot expose are recorded as
> `NaN` / `MeasurementUnavailable`; live measurement never silently falls back to
> simulated values.

---

## 3. Keywords

```
hierarchical memory orchestration; AI inference; large language models;
memory-bound computing; regime transitions; phase-like transitions;
dimensionless ratios; latency traces; GPU; TPU; AWS Inferentia2; AMD MI250;
Intel Optane-PMem; reproducibility
```

---

## 4. Related / alternate identifiers

| Relation | Identifier |
|----------|------------|
| `isSupplementTo` | the Nature Communications article DOI (add once assigned) |
| `isSupplementTo` | arXiv preprint DOI/ID (add if a preprint is posted) |
| `isDerivedFrom` (software) | GitHub release: https://github.com/leemgs/orion (tag `v1.0.0`) |
| `isDocumentedBy` | https://github.com/leemgs/orion (README, build & reproduction guide) |

> If the code is archived separately (recommended: enable the GitHub–Zenodo
> integration to mint a **software** DOI for the tagged release), link the two
> records with `isSupplementedBy` / `isSupplementTo`.

---

## 5. After minting the DOI — update the manuscript

Once the reserved DOI is available (e.g. `10.5281/zenodo.XXXXXXX`):

1. **`section/070_methods.tex`** — the sentence *"Pre-processed latency traces
   for all five platforms are included in the Zenodo data archive"* — add the
   DOI citation.
2. **`main.tex` → *Data availability*** — replace the current
   *"archived with a DOI on Zenodo upon acceptance … available from the
   corresponding author on reasonable request during review"* wording with a
   concrete DOI and access statement, e.g.:

   > The pre-processed latency traces, raw measurement logs, and
   > regime-classifier training data are available on Zenodo at
   > https://doi.org/10.5281/zenodo.XXXXXXX (CC BY 4.0).

3. **`README.md` §2 checklist item 13 ("Provide Zenodo DOI")** — mark complete
   and record the DOI.

---

## 6. Suggested archive structure

```
orion-data-v1.0.0/
├── README.md                     # dataset-level readme (this description + file map)
├── LICENSE                       # CC BY 4.0
├── traces/
│   ├── a100/                     # NVIDIA A100 (80 GB) — pre-processed traces
│   ├── tpu_v4/                   # Google TPU v4
│   ├── inferentia2/              # AWS Inferentia2
│   ├── mi250/                    # AMD MI250
│   └── optane_pmem/              # Intel Xeon + Optane-PMem
├── raw_logs/                     # raw .jsonl logs, one record / 10 s window
├── classifier/                   # regime-classifier training data + fitted boundaries
└── derived/                      # expected figures/tables for reproduction checks
```

---

## 7. Pre-submission checklist

- [ ] Upload files following the structure in §6.
- [ ] Fill the form fields from §1 and paste the §2 description.
- [ ] Set the license to **CC BY 4.0**.
- [ ] **Reserve DOI** (do not publish yet).
- [ ] Insert the reserved DOI into the manuscript (§5).
- [ ] Publish the Zenodo record (Open, or Restricted/embargoed until acceptance).
- [ ] Confirm the DOI resolves and the *Data availability* statement matches.
- [ ] (Recommended) Enable GitHub–Zenodo integration and archive the tagged
      code release for a separate software DOI.
