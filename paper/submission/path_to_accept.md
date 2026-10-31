# Path to Accept — *Nature* (Article) 기준 통합 우선순위 TODO

> 투고처: **Nature 본지 (Article)**. 이 문서가 최상위 기준이며, Nature
> Communications를 전제로 작성된 `nc_referee_report.md`,
> `nc_external_referee_simulation.md`, `nc_revision_plan.md`,
> `ncomms_compliance_report.md`, `reviewer_readiness_plan.md`는 **참고용
> 이력**이다(실험 설계 세부는 여전히 유효). 원칙: 근거 없는 주장·결과를 만들지
> 않는다. 텍스트·형식으로 해결 가능한 것은 즉시 반영했고, 실험이 필요한 것은
> "저자 실행 필요"로 구분한다.
>
> 형식 기준 출처: Nature formatting guide (nature.com/nature/for-authors/
> formatting-guide), 2026-10 확인.

---

## 0. 정직한 판정 — Nature 편집자·리뷰어 관점

**형식: 충족.** Nature Article 형식 요건(아래 §1)을 모두 맞췄고, 자동 점검으로
검증된다.

**과학적 판정: 현 상태로는 desk reject 가능성이 매우 높다.** Nature의 핵심
기준은 "탁월한 과학적 중요성"과 "분야 밖 독자에게 의미 있는 결론"이다. 이
원고는 측정 규약(reporting contract)과 소규모 proof-of-concept이며, 스스로
production 효과크기·정책 순위·에너지·장치별 경계값·모집단 일반화를 유보한다.
Nature Communications 기준에서도 자체 referee 시뮬레이션은 "유의성 부족으로
reject"였고, Nature 본지의 문턱은 그보다 훨씬 높다. 이 판단은 텍스트 수정으로
바뀌지 않는다.

**권고.** (1) 본 투고 전에 Nature의 **presubmission enquiry**(편집자 사전
문의, 공식 제도)를 먼저 보내 범위 적합성을 확인한다. 커버레터에도 같은 질문을
넣어 두었다. (2) 아래 P0 실험 결과를 확보한 뒤 본 투고하는 것이 현실적인
경로다. (3) P0 확보 전 투고가 꼭 필요하면 Nature Communications 또는 전문
학회(MLSys, ASPLOS, ISCA, SC 등)가 현 증거 수준에 맞는다.

### 예상 거절 사유 (엄격한 리뷰어 관점, 심각도순)

| # | 예상 지적 | 근거 | 텍스트로 해결? |
|---|---|---|---|
| R1 | **새 현상·결과가 없다.** 기여가 측정 규약이며 결론의 귀결(consequence)이 입증되지 않음 | 원고가 예측력·정책 순위를 "open test"로 명시 | 불가 — P0 |
| R2 | **신규성 점진적.** R_C=working-set 상주율, R_B=Roofline식 compute/transfer 균형의 재조합 | 서론이 "narrow gap"이라 스스로 기술 | 불가 — P0이 신규성 근거 |
| R3 | **증거가 합성 마이크로 프로브.** d=2048 layered matrix, pointer chase; 라벨은 규칙에서 연역되어 독립 검증 아님 | Results·Discussion에 명시 | 불가 — P0/P1 |
| R4 | **분야 밖 독자 관심 부족.** 컴퓨터 시스템 내부 문제로 읽힘 | Nature는 CS 논문을 광범위한 파급이 있을 때만 게재 | 일부 — 에너지·재현성 위기와 연결은 P0-1·P2-8 결과가 있어야 설득력 |
| R5 | **통계적 근거 약함.** 장치당 1대, 머신 내 반복만, 가속기 CI 없음, 유의성 검정 없음 | Methods·Discussion에 명시 | 불가 — P1 |
| R6 | **측정 artifact가 표에 남음.** A100/TPU의 fully-resident 점; warm-up 수정 후 재측정 안 함 | Methods에 명시 | 불가 — P1-5 |
| R7 | **"규약이 상충 결과를 해소한다"는 핵심 동기 미입증** | 커버레터·Discussion의 동기 | 불가 — P0-1 |

---

## 1. 완료됨 — 이번 Nature 재정비에서 직접 수정 (DONE)

- [x] **제목 교체 (95 → 73자).** "Separating memory capacity from data movement
      in neural network inference". Nature 규정(≤75자, 숫자·약어·구두점 없음,
      분야 밖 독자 이해 가능) 충족, 원고가 실제로 하는 일(두 제약의 분리)만
      서술하고 성능 결과를 주장하지 않음. SI·README·코드 문서도 동기화.
- [x] **Summary paragraph 재작성 (199단어).** Nature 지정 구조(기초 도입 →
      배경·근거 → "Here we introduce…" → 일반적 맥락), **참고문헌 인용 포함**,
      수치·약어·기호 제거. 주장 범위는 본문 증거와 동일.
- [x] **본문 분량 3,050 → 약 2,170단어** (Nature: 표시항목 4개 기준 약 2,500).
      삭제가 아니라 기술적 세부를 Methods로 이동: CPU layered-matrix 결과,
      fully-resident 측정 artifact 상세, reproducibility boundary. 모든 한계·
      단서 유지, 수치는 생성 매크로 그대로.
- [x] **Nature 구조.** 서론 제목 없음; 주제형 소제목(≤40자); Discussion;
      **main references → Methods(Data/Code availability 포함) →
      Acknowledgements → Author contributions → Competing interests** 순서.
      번호 없는 절로 바꾸면서 생긴 "Section N" 교차참조를 문장으로 교체.
- [x] **그림.** 그림 내부 제목 제거(Nature는 캡션에 제목), 300 dpi로 재생성
      (`reproduce_regime_map.py`, 정의·라벨 동일).
- [x] **저장소 정합성(리뷰어가 실제로 여는 곳).** 코드 README의 "three intrinsic
      operating regimes separated by abrupt, phase-like transitions" 및 과거
      과장 제목("Intrinsic regime-dependent limits govern…")을 원고 주장과
      일치하도록 수정. 원고는 불연속을 명시적으로 부정한다.
- [x] **자동 점검 강화(`check_submission.py`).** Nature 기준: 제목 ≤75자,
      summary ≤200단어 + 인용 필수, 본문 ≤3,000단어, 소제목 ≤40자, main 참고문헌
      ≤50; 금지 표현 검사를 README 3개까지 확대. citation key를 단어로 세던
      버그 수정.
- [x] **커버레터 Nature용 재작성.** 증거 범위를 정직하게 기술하고 편집자에게
      범위 적합성을 직접 질문.
- [x] (이전 세션) 표·그림 폭 초과 0, 참고문헌 44편 실재·메타데이터 검증
      (날조 1건 교체 포함), 링크·데이터 가용성 실검증.

## 2. 저자가 직접 해야 하는 비실험 항목

- [ ] **Funding statement** — Nature 순서상 필수 항목. 연구비 정보는 저자만 알
      수 있어 비워 두었다(추측 기재 금지). 없으면 "This research received no
      specific funding." 등 사실대로 기재.
- [ ] **Presubmission enquiry** 발송 (권고, §0).
- [ ] 포털: article type, 추천/배제 리뷰어, ORCID·소속·이메일 최종 확인.
      Reporting Summary는 Nature 지침상 생명과학·행동/사회과학 논문 필수로
      명시되어 있으므로 본 논문은 해당 여부를 포털에서 확인.
- [ ] 코드·데이터 릴리스를 Zenodo 등에 예치하고 **DOI를 Data availability에 추가**.

## 3. Accept를 위한 필수 실험 TODO — 우선순위 (저자 실행 필요)

> 텍스트로 해결 불가. 측정 하네스는 이미 저장소에 있다
> (`code/experiments/{vllm,deepspeed,flexgen}_strategy_inversion.py`,
> 분석은 `analyze_prereg.py`에 고정). 각 항목은 사전등록된 성공 기준으로
> 판정하며, null 결과도 그대로 보고한다.

### P0 — Nature 편집자 단계를 넘기 위한 최소 조건

1. **상충하던 기존 결과의 해소 (Nature급 "귀결")** — 서로 다른 결론을 낸
   공개된 오프로딩/서빙 연구 2건 이상을 재현해, 동작점 (R_C, R_B)을 보고하면
   두 결론이 서로 다른 regime에 있어 모순이 아님을 보인다. 이것이 "측정
   규약이 분야의 재현성 문제를 푼다"는 주장의 직접 증거이며 R1·R4·R7의 핵심 방어.
2. **예측력 홀드아웃 시험 (H3)** — 실제 서빙 스택 ≥2개, 가속기 아키텍처 ≥2개,
   실제 모델(예: 8B·70B급 LLM)에서 (R_C, R_B)를 측정 전 계산해 최속 정책을
   예측하고, label-blind baseline 대비 정확도의 bootstrap CI 하한이 baseline을
   넘는지 보고.
3. **정책 순위 역전 (H4)** — 예측된 좌표(R_C=θ_C 또는 R_B=1)를 사이에 두고
   정책 쌍의 순위가 CI가 분리된 채 뒤집히는지.
4. **실제 규모와 폭** — 합성 matmul이 아닌 실제 prefill/decode, ≥2개 워크로드
   계열(자기회귀 LLM + 검색증강 또는 비전).

### P1 — 통계·측정 엄밀성

5. **가속기 재측정** — warm-up이 들어간 하네스로 재실행해 fully-resident
   artifact(R6) 제거 후 Table 2 갱신.
6. **다중 머신 복제** — 아키텍처당 독립 머신 2–3대, 모든 효과크기에 bootstrap
   CI, 방향성 예측에 대한 사전등록 검정.
7. **경계의 독립 검증** — 라벨 정의와 무관한 결과 지표(달성 speedup,
   marginal latency)가 R_B=1 부근에서 꺾이는지.

### P2 — 분야 밖 독자 설득력 강화

8. **에너지** — 레짐별 토큰당 에너지(NVML)를 측정해 서론의 에너지 동기와
   연결. Nature 독자층에 대한 파급을 보여 주는 가장 직접적인 경로.
9. **Raw 트레이스 공개** — 윈도우·연산 단위 이벤트 타임스탬프를 DOI로 예치.
10. **Extended Data** (최대 10개) — 전체 sweep, 민감도(θ_C ∈ {0.4, 0.5, 0.6}),
    프레임워크별 결과.

## 4. 최소 실행 세트

Nature 본 투고 전에: **P0-1 + P0-2 + P0-3 (≥2 아키텍처, 실제 모델) + P1-6**.
이 결과가 확보되면 원고의 중심 주장을 "측정 규약 제안"에서 "동작점 보고가
상충 결과를 해소하고 최적 정책을 예측한다"로 올릴 수 있고, 그때 제목·summary를
그 결과에 맞게 다시 쓴다(현재 제목은 결과가 없는 상태에 맞춰 의도적으로
보수적이다).

## 5. 역할 분담

- **어시스턴트:** 하네스·분석 유지보수, 확보된 실측의 표/그림/Extended
  Data/통계 통합, 결과에 맞춘 제목·summary·본문 재작성, 형식 점검.
- **저자:** 실제 가속기 실행, 기존 논문 재현 대상 선정, DOI 예치, funding
  statement, presubmission enquiry, 포털 제출.

> 어떤 편집 결정도 보장할 수 없다. 현재 원고는 증거 범위를 숨김없이 기술하고
> 있으며, Accept 가능성을 실질적으로 바꾸는 것은 §3의 실험이다.
