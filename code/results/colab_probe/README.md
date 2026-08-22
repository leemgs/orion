# Colab 레짐 측정 하네스 — 실행 결과 (CPU 검증 + 실제 GPU 측정)

`experiments/colab_regime_measurement.py` 는 논문의 R_C(잔류)·R_B(오버랩)
방법론을 **실제 하드웨어**에서 측정한다. GPU(CUDA Event)/TPU(XLA)/CPU를
자동 감지하며, 모든 시간은 실측이다(합성 없음).

## 실제 가속기 측정 — 3종·2개 벤더

각 `accel_<장치>.json`는 CUDA-event(또는 TPU는 XLA) 타이밍으로 얻은 **실측
가속기 데이터**다(d=2048, 24층, batch 8, 10 windows).

| 장치 | 파일 | 용량 전환(예측 1) | 오버랩 경계(예측 2) |
|------|------|------------------|--------------------|
| NVIDIA Tesla T4 | `accel_tesla-t4.json` | 14.53 → 41.08 ms = **2.83×** | R_B=1.0에서 I/O 경계 **명확 확인** |
| NVIDIA A100 | `accel_a100.json` | 52.40 → 70.55 ms = **1.35×** | sweep B가 compute-bound(R_B>2) → 경계 미탐침 |
| Google TPU v5e | `accel_tpu-xla.json` | 220.6 → 270.6 ms = **1.23×** | XLA 지연실행 → R_B 신뢰도 낮음(참고치) |

→ **예측 1(잔류 낮추면 지연 증가)은 3종 모두에서 성립**(방향; 크기는 장치 의존,
논문은 크기를 예측하지 않음). **예측 2(θ_B=1.0)는 T4에서 명확 확인.**
빠른 장치(A100/TPU)는 d=2048 워크로드가 작아 per-op 타이밍에 런치/측정
오버헤드가 섞여 R_C≈1 부근이 비단조적이다(정직하게 표기).

per-point 평균만 저장(Colab 세션의 per-window raw는 미보존).

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
