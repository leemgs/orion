# Path to Accept — 통합 우선순위 TODO (Nature Communications)

> 최상위 통합 문서. 기존 개별 문서(`nc_referee_report.md`, `nc_external_referee_simulation.md`,
> `reviewer_readiness_plan.md`, `nc_revision_plan.md`, `preregistration.md`)의
> 지적을 하나의 우선순위 체크리스트로 합친 것. **근거 없는 결과를 만들지 않는다**는
> 원칙에 따라, 텍스트로 방어 가능한 항목만 즉시 수정했고 실측이 필요한 항목은
> "저자 실행 필요"로 명시한다.

---

## 0. 정직한 현재 판정 (reviewer 관점 요약)

- **형식·정직성·재현성: 충족.** desk-reject을 유발하는 형식 결함은 남아 있지 않다
  (아래 §1). 제목/초록은 발견된 "한계"를 주장하지 않으며, 모든 수치는 커밋된
  데이터에서 추적 가능하고, 측정/시뮬레이션이 분리되어 있다.
- **핵심 약점은 유의성(significance/advance)이다 — 텍스트로 해결 불가.** 현재
  증거는 소규모 합성 프로브의 proof-of-concept이며, 논문 스스로 production·전략
  순위·에너지·장치별 경계값·모집단 일반화 주장을 **명시적으로 유보**한다. 이
  조합은 Nature Communications에서 외부심사 전 editorial decline(유의성 사유)
  위험이 크다. 이는 자체 referee 시뮬레이션의 결론과 일치한다
  (`nc_external_referee_simulation.md`).
- **결론:** 제목 그대로의 "최종 제출 가능 수준"은 **형식적으로는 도달**했으나,
  Accept 확률을 실질적으로 높이려면 §3의 실험이 필요하다. 실험 없이 텍스트만으로
  Accept를 보장한다고 말하는 것은 근거 없는 주장이므로 하지 않는다. 자원 제약 시
  전문 시스템/측정 학회(MLSys, SC, ISCA, IISWC, IMC, IEEE TPDS/TC)가 현 상태의
  적합 투고처다.

---

## 1. 완료됨 — 형식/무결성 (DONE, 이번 및 직전 세션에서 수정)

- [x] **표·그림 가로 폭 초과 제거.** 본문·부록 모두 overfull hbox/vbox = 0.
      Table 2 컬럼 간격 축소로 7.77pt 초과 해소. 그림 해상도 ≈380 dpi(≥300 dpi).
- [x] **참고문헌 무결성.** 인용 44편 전수 실재 확인(arXiv API, Crossref, DBLP,
      OpenAlex, IEEE Xplore, USENIX).
  - 날조/비해석 DOI 1건 적발·교체: `wang2023modeloffloading`(DOI 404, 제목
    미존재) → Xiaofei Wang 외, *IEEE Comm. Surveys & Tutorials* 22(2):869–904,
    2020, DOI 10.1109/COMST.2020.2970550 (검증).
  - BibTeX가 역순 파싱하던 저자명 16건, 연도 오류 1건(He 2023→2024), 제목
    오타/약어 대소문자, 중복 bib 4건 정리.
- [x] **인용 키 44 = 인쇄 참고문헌 44** 1:1 정합, undefined ref/citation 0.
- [x] **제출 감사(`check_submission.py`) 통과** — draft-only 표현/placeholder 없음,
      초록 ≤150단어.
- [x] **필수 statement 존재** — Data/Code availability, Competing interests,
      Author Contributions(CRediT-스타일), Acknowledgements.
- [x] **70B 예시를 "illustrative calculation, not an empirical measurement"로
      명시** (referee minor point 직접 반영).
- [x] **내부 불일치 수정.** CPU 소절이 "previously planned A100, MI250, TPU v4,
      Inferentia2, Optane campaign"을 미실행으로 기술했으나 A100은 바로 다음
      소절에서 실제 실행·보고됨(Table 2). 유령 하드웨어 열거를 제거하고
      "preregistered, multi-platform campaign"으로 일반화하여 모순 해소.
- [x] **Supplementary Information(`supplementary.pdf`)를 저장소에서 추적**하도록
      `.gitignore` 예외 추가 및 커밋.

---

## 2. 텍스트/컴플라이언스로 해결 가능 — 상태

즉시 수정 가능한 항목(실험 불필요). 이번 세션에서 처리했거나 저자 포털 작업만 남은 것.

- [x] 제목/초록이 경험적 "한계" 발견을 함의하지 않도록 유지 (현재 통제됨:
      θ_C=0.5는 관례, θ_B=1은 정의로 명시).
- [x] max-form 하한이 단일-tier roofline 대비 무엇을 더하는지 Results에서 명시
      (ρ·D_nr 잔류 floor + sustained(비peak) 대역폭).
- [x] 70B 예시 = 설명용 산술임을 명시 (§1).
- [x] 저자권 vs Acknowledgements 정합성 확인: 단독저자 G.L.; Acknowledgements의
      피드백 제공자들은 authorship 기준 미충족 → acknowledgement 처리 적절. 변경 불필요.
- [ ] **(저자, 포털)** article type 선택, editor/reviewer 추천, competing-interest/
      ORCID/이메일 최종 확인, 저널의 현행 포맷·저장소 정책 재확인
      (`ncomms_compliance_report.md`).
- [ ] **(저자, 선택)** 서론의 에너지·경제 stakes는 현재 인용 기반 "동기"로 한정되어
      있음. §3의 에너지 실측(R8)을 하지 않을 경우 현 문장 유지가 안전(결과 주장이
      아니라 동기이므로 과장 아님). 에너지 실측을 추가하면 서론과 결과가 연결됨.

---

## 3. Accept 바를 넘기 위한 필수 실험 TODO — 우선순위별 (저자 실행 필요)

> 아래는 **텍스트로 해결 불가**하며 실제 가속기 실험이 필요하다. 어시스턴트는
> 프로토콜·스크립트·통합을 지원할 수 있으나, 실행·데이터 생성은 저자 몫이다.
> 각 항목에 **사전지정 성공 기준**을 둔다(사후 fitting 금지).
> 상세: `nc_revision_plan.md`(단계별), `reviewer_readiness_plan.md`(지적↔작업),
> `preregistration.md`(사전등록 설계).

### P0 — 이게 없으면 Accept 불가 (desk/major-reject 방어의 핵심)

1. **전략 순위 역전(strategy-ranking inversion) 실측** — *단일 최고 레버리지*.
   - 작업: 실제 서빙 스택(vLLM 기본; FlexGen/DeepSpeed-Inference 2차)에서 실제
     모델로, ≥3개 오케스트레이션 정책을 3개 레짐을 가로지르는 ≥3 동작점에서 비교.
   - 성공 기준(사전지정): coordination-dominated에서 이긴 정책이 I/O- 또는
     capacity-limited에서 짐(문서화된 rank inversion), bootstrap CI 동반.
   - 해제되는 주장: "roofline 재명명"을 넘어서는 신규성·처방력.
   - **[측정 스크립트 제공됨 — vLLM · DeepSpeed-Inference · FlexGen]**
     `code/experiments/{vllm,deepspeed,flexgen}_strategy_inversion.py`
     (공통 코어 `strategy_inversion_common.py`). 세 스택을 동일한 방식으로 측정:
     4개 정책을 동작점별로 구동해 `T_total` + reporting-contract 필드(실측 H2D
     대역폭 `B_slow`, resident baseline의 isolated `T_comp`, `T_transfer=D/B_slow`)를
     locked JSONL 스키마로 기록, `analyze_prereg.py`로 H3/H4 요약 출력.
     GPU/프레임워크 부재 시 예외(날조 없음), 실행 불가 셀은 기록 없이 건너뜀,
     `--dry-run`으로 하드웨어 없이 검증. vLLM의 KV offload는 V1 엔진에서
     무시되는 `swap_space` 대신 KV-offloading connector 사용(엔진 로그로 활성
     확인 필요). **실제 가속기 구동·데이터 생성은 저자 몫.**

2. **실제 워크로드 substance** — d=2048 합성 matmul 대체.
   - 작업: 토큰 생성(prefill+decode), KV 캐시/continuous batching/IO-aware
     attention/PCIe·NVLink 오프로드 포함. 4항 분해(T_comp,T_mem,T_swap,T_sync)를
     선언된 동작점에서 측정; (R_C,R_B)를 datasheet로 예측 후 실측 확인.

3. **홀드아웃 예측 시험(predictive, not descriptive)**.
   - 작업: N개 실제 구성에 대해 (R_C,R_B)를 사전 계산 → 레짐/최속 정책을 **측정 전**
     예측 → 적중률 보고(naive baseline 대비). 예: "N개 중 k개에서 최속 정책 정확 예측".

### P1 — 통계적 엄밀성 (P0 결과의 신뢰성 확보)

4. **다중 머신 복제 + 유의성.** 가속기 클래스당 독립 머신/인스턴스 2–3대,
   모든 효과크기에 bootstrap CI, 방향성 예측에 유의성 검정, 동작점 그리드 사전등록.
   (현재 s.d.는 머신내 반복이며 모집단 분산 아님 — referee R5.)

5. **가속기 artifact 재측정.** warm-up fix를 넣고 재실행하여 Table 2의 fully-resident
   분모(현재 compile/launch artifact 영향)를 정정; 깨끗한 단조 전환을 CI와 함께 제시.

6. **독립 경계 검증(boundary decoupled from label).** R_B에 정의상 묶이지 않은
   결과 지표(달성 speedup, marginal latency, 에너지)가 R_B=1 근방에서 꺾이는지를,
   런치/컴파일 오버헤드를 벗어난 큰 워크로드에서 측정하고 sharpness를 정량화.

### P2 — 일반성·재현성·동기 폐루프 (리비전 강화)

7. **워크로드 일반성.** 3-레짐 거동을 ≥2(이상적으로 3) 모델 클래스에서 재현:
   자기회귀 LLM(Llama-3 8B/70B, Mixtral), 비전/VL(ViT, BLIP-2), RAG(FAISS+생성기).

8. **Raw 트레이스 공개.** 윈도우별·op별 raw device-event 타임스탬프 + 고정 컨테이너를
   Zenodo DOI로 예치하고 Data availability에 DOI 인용(현재 요약 JSON만 존재 — R9).

9. **에너지(선택).** NVML 전력적분으로 레짐별 throughput-per-watt를 ≥2 가속기에서
   측정 → 서론의 에너지 동기와 연결(R8). 불가 시 서론 에너지 프레이밍 축소.

10. **선행 상충결과 재해석.** 서로 어긋났던 기존 오프로딩/서빙 결과 2건을 재현하여
    서로 다른 레짐에 있음을 보임 → "reporting contract"의 유용성 실증(R10).

---

## 4. 최소 실행 세트 (자원 제약 시)

desk-reject 위험 → 외부심사 진입으로 전환하는 최소 조합:
**P0 전체(1·2·3) + P1-4(다중 머신 CI) + P2-8(raw 트레이스 DOI).**
이 조합이면 "실증 substance 부족"이라는 핵심 지적이 대부분 방어된다. P1-6(경계),
P2-7(일반성), P2-9(에너지), P2-10(재해석)은 리비전에서 강화.

## 5. 역할 분담

- **어시스턴트:** 측정 프로토콜·스크립트(vLLM/FlexGen 통합, 역전·홀드아웃 설계,
  사전등록 문서, 트레이스 export), 확보된 실측의 표/그림/수치/통계 통합, 텍스트 정비.
- **저자:** 실제 가속기 실행, DOI 예치, 포털 제출, 저자권/competing-interest 최종 확인.

> 어떤 편집 결정도 보장할 수 없다. 남은 1차 과학적 한계는 소규모 합성·비복제
> proof-of-concept 범위이며, 이는 은폐 없이 원고에 그대로 기술되어 있다.
