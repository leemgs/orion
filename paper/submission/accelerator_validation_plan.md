# 가속기 후속 검증 계획

## 현재 증거 수준

현재 원고에는 NVIDIA T4, NVIDIA A100, Google TPU v5e의 단일 실행별
operating-point 요약이 포함된다. 세 장치에서 low-residency 지연시간의 방향은
예측과 일치하며, T4 sweep은 측정된 $R_B$ 값이 1의 양쪽에 도달한다. 그러나
라벨은 선언된 분류 규칙에서 자동으로 정해지므로 이를 경계의 독립 검증으로
해석하지 않는다. 저장된 JSON은 per-point 요약이며 raw device-event trace가
아니다.

## 제출 후 강한 검증에 필요한 항목

1. 최소 두 아키텍처에서 독립적인 전체 sweep을 반복하고 run-to-run 변동을
   보고한다.
2. 사전 등록한 operating-point grid로 $R_B=1$ 주변을 조밀하게 표집한다.
3. compute, transfer, end-to-end 구간의 raw event timestamp와 warm-up 및 제외
   규칙을 보존한다.
4. $R_C$는 연속값으로 보고하고, 0.5 외의 합리적인 residency convention에서도
   범주 요약의 민감도를 제시한다.
5. XLA는 명시적 동기화 전후의 trace로 deferred execution을 분리한다.
6. 장치, 드라이버, 런타임, 라이브러리, 전력·클럭 설정과 sustained bandwidth
   측정 절차를 기록한다.

이 검증이 완료되기 전에는 per-device 경계값, 프로덕션 모델 일반화, 에너지,
전략 순위, 분류기 정확도 또는 모집단 수준의 하드웨어 일반화를 주장하지 않는다.
