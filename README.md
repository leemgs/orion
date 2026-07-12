# ORION

**대규모 AI 추론에서 계층적 메모리 오케스트레이션의 레짐 의존적 한계**
*Hierarchical memory orchestration in AI inference exhibits intrinsic regime-dependent limits*

ORION은 대규모 AI 추론에서 계층적 메모리 오케스트레이션이 **하드웨어·워크로드 레짐(regime)에 따라 근본적으로 다른 한계**를 갖는다는 것을 규명한 연구입니다. 두 개의 무차원 비율 **R_C**(계산-메모리 비)와 **R_B**(대역폭-용량 비)만으로 최적 오케스트레이션 전략이 결정되며, 특정 레짐에서는 전략이 **역전(inversion)** 됩니다.

투고 목표: **Nature Computational Science** (1순위) / **Nature Communications** (2순위).

---

## 저장소 구성

관리 용이성을 위해 저장소를 성격이 다른 세 축으로 분리했습니다.

| 폴더 | 내용 | 상세 문서 |
|------|------|-----------|
| [`code/`](code/) | Orion 측정 프레임워크 및 논문 결과 재현 스크립트 (Python) | [code/README.md](code/README.md) |
| [`paper/`](paper/) | LaTeX 논문 원고 (IEEE·Nature 템플릿), 그림, 참고문헌, 빌드 스크립트 | [paper/README.md](paper/README.md) |
| [`ppt/`](ppt/) | 발표 자료 (한국어·영어 슬라이드, NCS 발표본) | — |

---

## 빠른 시작

- **논문 빌드 방법 · 투고 전략 · 저널 우선순위** → [paper/README.md](paper/README.md)
- **결과 재현 · 코드 실행 방법** → [code/README.md](code/README.md)
