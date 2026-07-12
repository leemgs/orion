# Nature Machine Intelligence — Presubmission Inquiry

> **용도:** 정식 투고 전 NMI 에디터에게 스코프 적합성을 문의하는 커버레터입니다.
> Nature 자매지의 presubmission enquiry는 초록 수준의 요약 + 짧은 커버레터만으로
> 보통 **1~2주 내** "심사 진행 의향 있음 / 스코프 밖" 회신을 받습니다.
> 리젝 사이클(4~6개월)을 태우지 않고 NMI 적합성을 검증하는 안전장치입니다.
>
> 제출 경로: NMI 온라인 시스템의 *Presubmission enquiry* 또는 편집부 이메일.
> 아래 [ ] 표시는 투고 전 실제 값으로 채우세요.

---

## Cover letter (편집장 앞)

Dear Editor,

We would like to enquire whether the following study falls within the scope of
*Nature Machine Intelligence* before submitting a full manuscript.

**The finding.** As AI models scale beyond fast-memory capacity, the performance
of large-scale inference is increasingly limited not by computation but by how
intelligent systems coordinate data across a hierarchical memory system. We show
that this coordination is **not continuously optimizable**: it is governed by
**intrinsic, regime-dependent limits**. Characterizing any inference system by
two dimensionless ratios — the fast-memory residency ratio (R_C) and the
transfer-pressure ratio (R_B) — we identify three operational regimes separated
by **abrupt, phase-like transitions**. Across these boundaries the ranking of
optimization strategies **inverts**: a method that reduces latency by 24% in one
regime increases it by 8–12% in another. We derive structural lower bounds
proving the transitions are inevitable, calibrate the boundaries across five
hardware platforms, and validate the principle across diverse AI workloads
(autoregressive language models, vision question answering, retrieval-augmented
generation).

**Why *Nature Machine Intelligence*.** This is not a systems-engineering
optimization report but a **general principle governing the behaviour of
memory-bound machine intelligence at scale**. It reframes how the field should
reason about deploying and scaling AI inference — from continuous tuning toward
**regime-aware design** — and gives concrete, computable boundary targets that any
practitioner can evaluate from hardware datasheets and model cards before running
a single experiment. We believe this speaks directly to the readership of
*Nature Machine Intelligence*, which spans the science of intelligent systems and
their real-world deployment.

**Significance and novelty.** Prior work (offloading, GPU–CPU swapping, KV-cache
management) explains performance heterogeneity through workload-specific factors
without a unifying account of *when* and *why* orchestration fails. Our
regime-based framework provides that account analytically, with a proven lower
bound and multi-platform empirical confirmation.

We would be grateful to know whether you would consider a full submission on this
topic. A preprint is available at [arXiv ID], and we can provide the complete
manuscript immediately upon request.

Thank you for your time and consideration.

Sincerely,
[Corresponding author name], on behalf of all authors
[Affiliation]
[Email]

---

## Enquiry summary (≈100 words, 에디터 판단용 압축본)

Large-scale AI inference is increasingly memory-bound: performance is set by how
intelligent systems coordinate data across hierarchical memory, not by compute.
We show this coordination has intrinsic, regime-dependent limits. Two
dimensionless ratios (R_C, R_B) define three operational regimes separated by
abrupt, phase-like transitions, across which the ranking of optimization
strategies inverts. We prove the transitions are structurally inevitable,
calibrate the boundaries across five hardware platforms, and confirm the
principle on language, vision, and retrieval-augmented workloads. The result
establishes regime-dependent behaviour as a general principle of memory-bound
machine intelligence, reframing AI-inference scaling from continuous tuning to
regime-aware design.

---

## 체크리스트 (투고 전)

- [ ] arXiv 프리프린트 공개 및 ID 기입
- [ ] `[ ]` 자리표시자(저자·소속·이메일) 전부 채움
- [ ] 커버레터에서 "engineering optimization"이 아니라 "principle / discovery"로 읽히는지 최종 확인
- [ ] NMI 최신 투고 지침(단어 수·초록 형식) 재확인
- [ ] 산업 CPS·IEC 61508 안전 내용은 "부가 검증"으로만 언급 (주 서사는 원리)
