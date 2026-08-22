# 2-가속기 실측 확보 계획 (desk-reject 리스크 완화)

## 목적
단일 가속기(현재 Tesla T4) proof-of-concept는 "단일 장치 아티팩트"라는 리뷰어
지적에 취약하다. **서로 다른 아키텍처의 가속기 2개 이상**에서 두 예측
(잔류 → 용량 전환, R_B<1 → I/O 노출)이 모두 성립함을 보이면, 경계가 특정
장치의 산물이 아니라는 **교차 아키텍처 증거**가 되어 desk-reject 리스크가
실질적으로 줄어든다.

경계 θ_C=0.5, θ_B=1.0은 **유도값**이라 하드웨어에 따라 움직이지 않는다.
따라서 검증의 요점은 "경계 위치"가 아니라 **동작점이 각 장치에서 예측한 쪽에
떨어지는가**이다.

## 현재 상태 (완료)
서로 다른 아키텍처 **3종·2개 벤더** 실측 완료·논문 반영됨:
- **NVIDIA Tesla T4** — 용량 전환 2.83×, R_B=1.0에서 I/O 경계 확인.
  `code/results/colab_probe/accel_tesla-t4.json`
- **NVIDIA A100** — 용량 전환 1.35×(방향 성립). sweep B는 compute-bound이라
  I/O 경계 미탐침. `code/results/colab_probe/accel_a100.json`
- **Google TPU v5e** — 용량 전환 1.23×(방향 성립). XLA 지연실행으로 R_B 분해
  신뢰도 낮음(참고치). `code/results/colab_probe/accel_tpu-xla.json`

핵심: **예측 1(잔류)은 3종 모두에서 성립**(방향; 크기는 장치 의존, 논문은 크기를
예측하지 않음). **예측 2(오버랩 θ_B=1)는 T4에서 명확 확인.** 이로써 "단일 장치
아티팩트" 지적을 완화하는 교차 아키텍처 증거를 확보.

## 가속기 #2 — 회원님이 셀 하나 실행 (권장 순서)

하네스는 장치를 자동 감지하고, 실행 후 장치명이 붙은
`results/colab_probe/accel_<장치>.json` 파일을 저장한다.

### 옵션 1 (권장·가장 신뢰도 높음): 다른 NVIDIA GPU — L4 또는 A100
Colab 런타임 유형을 **L4**(무료/Pro) 또는 **A100**(Pro)로 바꾼 뒤:
```python
!git clone https://github.com/leemgs/orion.git
%cd orion/code
!pip -q install torch
!python experiments/colab_regime_measurement.py     # GPU 자동 감지, CUDA-event 실측
```
T4와 다른 대역폭/아키텍처의 GPU라 CUDA-event 경로가 그대로 신뢰 가능.

### 옵션 2 (교차 벤더로 가장 강한 주장): Google TPU
런타임 유형을 **TPU**로 바꾼 뒤:
```python
!git clone https://github.com/leemgs/orion.git
%cd orion/code
!pip -q install torch torch_xla
!python experiments/colab_regime_measurement.py     # XLA 자동 감지
```
주의: XLA는 지연 실행이라 per-op(T_comp/T_transfer) 분해 타이밍의 신뢰도가
낮다. TPU에서는 **용량 전환(T_total, 예측 1)** 을 1차 증거로 삼고, R_B 분해는
참고치로 다룬다.

## 실행 후 저에게 보내주실 것
1. 화면의 **`[논문 붙여넣기용 요약]` 블록** 전체, 그리고
2. `results/colab_probe/accel_<장치>.json` **파일 내용** (per-point 원시값 보존용).

Colab에서 파일 내용 출력:
```python
print(open("results/colab_probe/" +
      [f for f in __import__("os").listdir("results/colab_probe")
       if f.startswith("accel_")][ -1]).read())
```

## 수령 후 제가 완료할 작업 (2-가속기 반영)
1. `accel_<장치2>.json`을 저장소에 추가.
2. `export_paper_results.py`가 두 가속기를 읽어 **교차 가속기 비교 표**와
   매크로를 생성하도록 확장.
3. Results의 GPU 절을 **복수 가속기**로 갱신(장치별 용량 전환 배수 + θ_B=1.0
   경계 확인 표), 초록·논의·부록을 "two accelerators"로 정정.
4. 재빌드·제출 감사 통과 확인 후 `main`에 푸시.

## 정직한 한계
- 저(어시스턴트)는 Colab/가속기를 직접 구동할 수 없어 이 한 번의 실행은
  회원님이 수행해야 한다. 수치를 지어내지 않는다.
- 가속기 2개라도 여전히 프로덕션 모델·동시요청·에너지·전략 역전 주장은 하지
  않는다(범위 정직 유지). 목표는 "경계가 단일 장치 산물이 아님"의 교차 증거다.
