# Colab 레짐 측정 하네스 — 실행 결과 (CPU 검증 + 실제 GPU 측정)

`experiments/colab_regime_measurement.py` 는 논문의 R_C(잔류)·R_B(오버랩)
방법론을 **실제 하드웨어**에서 측정한다. GPU(CUDA Event)/TPU(XLA)/CPU를
자동 감지하며, 모든 시간은 실측이다(합성 없음).

## 실제 GPU 측정 — NVIDIA Tesla T4 (Google Colab)

`tesla_t4_summary.json` 는 무료 Colab T4 런타임에서 CUDA-event 타이밍으로
얻은 **실측 가속기 데이터**다(d=2048, 24층, batch 8, 10 windows).

| 예측 | 실측 결과 |
|------|-----------|
| **예측 1 (잔류)** | 완전 상주(R_C=1) **14.53 ms** → 용량 제한(R_C<0.5) 평균 **41.08 ms** = **2.83×** |
| **예측 2 (오버랩)** | 연산량 스윕에서 R_B<1 → 모두 I/O-limited, R_B≥1 → 모두 coordination-dominated. 전환이 **R_B=0.90~1.19 사이**, 즉 유도 경계 **θ_B=1.0**에서 발생 |

→ CPU 하네스로는 잡음 때문에 못 보였던 **오버랩(I/O) 경계가 실제 GPU에서
선명하게 재현**됨. 두 예측 모두 실리콘에서 확인.

원시 요약: `tesla_t4_summary.json` (per-point 평균; Colab 세션의 per-window
raw 레코드는 보존되지 않아 요약만 저장).

## Colab에서 재현 (회원님이 직접)

```python
# 런타임 → 런타임 유형 변경 → GPU (또는 TPU) 선택 후:
!git clone https://github.com/leemgs/orion.git
%cd orion/code
!pip -q install torch                 # TPU면: !pip -q install torch torch_xla
!python experiments/colab_regime_measurement.py
```

끝나면 화면의 `[논문 붙여넣기용 요약]` 블록을 전달해 주시면 본문/그림에 반영합니다.

## 로컬 CPU 검증 (이 저장소에서 실제 수행)

`summary.json` / `records.jsonl` 는 numpy 백엔드 CPU 검증 실행 결과다
(용량 전환 2.15×). 스크립트 정상 동작 확인용이며, GPU 결과가 본 측정이다.

## 정직한 한계

- T4 결과는 **단일 가속기 proof-of-concept**다. 5개 플랫폼(A100/TPU v4/
  Inferentia2/MI250/Optane) 전면 campaign, 전략 역전, 분류기 정확도, 에너지
  주장을 대체하지 않는다.
