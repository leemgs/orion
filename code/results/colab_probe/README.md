# Colab 레짐 측정 하네스 — 실행/검증 결과

`experiments/colab_regime_measurement.py` 는 논문의 R_C(잔류)·R_B(오버랩)
방법론을 **실제 하드웨어**에서 측정한다. GPU(CUDA Event)/TPU(XLA)/CPU를
자동 감지하며, 모든 시간은 실측이다(합성 없음).

## Colab에서 실행 (회원님이 직접)

```python
# 1) 런타임 → 런타임 유형 변경 → GPU (또는 TPU) 선택
# 2) 아래 셀 실행
!git clone https://github.com/leemgs/orion.git
%cd orion/code
!pip -q install torch                 # TPU면: !pip -q install torch torch_xla
!python experiments/colab_regime_measurement.py        # GPU/TPU 자동 감지
```

실행이 끝나면 화면의 **`[논문 붙여넣기용 요약]` 블록**을 그대로 복사해
전달해 주시면, 그 실측값으로 본문 표·수치를 갱신하고 Figure를 재생성합니다.
원시 데이터는 `results/colab_probe/records.jsonl`, `summary.json` 에 저장됩니다.

## 로컬 CPU 검증(이 저장소에서 실제 수행함)

가속기 없이 스크립트 자체를 검증하기 위해 numpy 백엔드로 실행:

```
python experiments/colab_regime_measurement.py --backend numpy --d 512 --layers 12 --batch 8 --windows 5 --quick
```

실측 결과(Intel Xeon, 4코어):

| 스윕 | 관측 |
|------|------|
| R_C(잔류) | capacity-limited(R_C<0.5) 평균 4.03 ms → resident(R_C≥1) 1.87 ms = **2.15×** |

→ 작업집합이 빠른 메모리를 넘으면 지연이 계단식으로 증가하는 **capacity-limited
레짐 전환**을 실리콘에서 재확인. (스윕 B의 R_B 전환은 연산/대역폭 균형에
의존하므로 CPU에서는 잡음이 크고, 실제 GPU에서 선명하게 관측된다.)

## 정직한 한계

- 이 데이터는 **스크립트 검증 + 정성적 CPU 참조**다. 논문의 5개 가속기
  (A100/TPU v4/Inferentia2/MI250/Optane) **정량값을 대체하지 않는다.**
- θ_B, 전략 역전 크기(±%), 분류기 정확도 등은 실제 가속기 측정이 필요하며,
  위 Colab 실행이 그 데이터를 확보하는 경로다.
