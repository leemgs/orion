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

### R_C≈1 비단조성 원인 조사 (결론)

지연은 **오프로딩 층 수(전송 횟수)에 대해 n_off≥2 구간에서 3종 모두 단조**
감소한다. 유일한 이탈은 **완전 상주 점(R_C=1, n_off=0)** 으로, A100(+9 ms)·
TPU(+42 ms)에서만 추세 위로 튀고 느린 T4에서는 튀지 않는다. 원인은 **측정
아티팩트**로 규명됨:
- R_C=1은 오프로딩 루프가 통째로 생략되는 **유일하게 구조가 다른 실행**(TPU에선
  별도 컴파일되는 XLA 그래프)이다.
- d=2048에서 빠른 장치는 실제 메모리 신호가 작아, 이 점의 1회성 컴파일·런치
  오버헤드가 잔여 신호를 초과한다(느린 T4는 신호가 커서 단조 유지).
- **Eager(NumPy) 대조 실험**: 동일 스윕에서 R_C=1이 **최저 지연**(단조) →
  아티팩트가 워크로드 로직이 아니라 가속기 실행/타이밍 경로에서 발생함을 확인.
- 하네스에 **워밍업 1스텝(측정 제외)** 추가 → 향후 실행에서 점별 컴파일 비용 제외.

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
