# ORION — 대규모 AI 추론에서 계층적 메모리 오케스트레이션의 레짐 의존적 한계

LaTeX 논문 원고. **Nature Computational Science** (1순위) 또는 **Nature Communications** (2순위) 투고를 목표로 합니다.

---

## 목차

1. [저장소 구조](#1-저장소-구조)
2. [의존 패키지 설치 (Ubuntu 24.04)](#2-의존-패키지-설치-ubuntu-2404)
3. [빌드 방법](#3-빌드-방법)
4. [PDF 열람](#4-pdf-열람)
5. [익명 처리 스위치](#5-익명-처리-스위치)
6. [Nature 저널 구조 이해](#6-nature-저널-구조-이해)
7. [저널 추천 우선순위](#7-저널-추천-우선순위)
8. [단계별 투고 전략](#8-단계별-투고-전략)
9. [리뷰 절차 및 게재료](#9-리뷰-절차-및-게재료)

---

## 1. 저장소 구조

```
.
├── main.tex                    # IEEEtran 템플릿 (초안 / 내부 검토용)
├── main_nature.tex             # Springer Nature sn-jnl 템플릿 (투고용)
├── sn-jnl.cls                  # Springer Nature 공식 저널 클래스
├── IEEEtran.cls                # IEEE 클래스 (main.tex 용)
├── IEEEtranDOI.bst             # IEEE BibTeX 스타일
├── reference-data.bib          # 참고문헌 데이터베이스
├── latexmkrc                   # latexmk 설정 (타임존)
├── run.sh                      # 빌드 스크립트
├── figures/                    # 그림 파일 (PNG)
└── section/                    # 섹션별 .tex 파일
    ├── 001_title.tex
    ├── 005_author.tex          # IEEEtran 저자 블록
    ├── 005_author_nature.tex   # sn-jnl 저자 블록
    ├── 006_abstract.tex
    ├── 006_abstract_nature.tex
    ├── 010_introduction.tex
    ├── 020_regime_principle.tex
    ├── 030_transfer_model.tex
    ├── 040_experimental_validation.tex
    ├── 050_implications.tex
    ├── 060_discussion.tex
    ├── 070_methods.tex
    ├── 080_conclusion.tex
    ├── 090_ack.tex
    ├── 095_reference.tex
    ├── 095_reference_nature.tex
    └── 900_appendix.tex
```

---

## 2. 의존 패키지 설치 (Ubuntu 24.04)

```bash
# TeX Live 핵심 패키지 설치
sudo apt-get update
sudo apt-get install -y \
    texlive-base \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-pictures \
    texlive-bibtex-extra \
    bibtex

# PDF 뷰어 설치
sudo apt-get install -y evince
```

> **참고:** `texlive-science` 패키지에 이 논문에서 필요한 `algorithm.sty` 및 `algorithmicx.sty`가 포함되어 있습니다.

---

## 3. 빌드 방법

### Nature 템플릿 (투고용)

```bash
./run.sh nature
```

`main_nature.pdf` 파일이 생성됩니다.

### IEEEtran 템플릿 (내부 초안용)

```bash
./run.sh
```

`main.pdf` 파일이 생성됩니다.

### `run.sh` 내부 동작 순서

```
pdflatex  →  bibtex  →  pdflatex  →  pdflatex
```

---

## 4. PDF 열람

```bash
# Nature 투고용 PDF
evince main_nature.pdf

# IEEEtran 초안 PDF
evince main.pdf
```

기타 뷰어:

```bash
xdg-open main_nature.pdf     # 시스템 기본 뷰어
okular main_nature.pdf        # KDE 뷰어
zathura main_nature.pdf       # 경량 뷰어
```

---

## 5. 익명 처리 스위치

`main.tex` / `main_nature.tex` 상단의 `\anonymous` 값으로 저자 공개 여부를 제어합니다.

| 값 | 효과 |
|----|------|
| `1` | 실제 저자명 및 소속 표시 |
| `0` | 블라인드 리뷰용 익명 처리 |

---

## 6. Nature 저널 구조 이해

**Nature (본지)** 는 1869년 창간된 단일 저널이고, 그 아래 특화 자매지들이 별도 저널로 운영됩니다. 구조는 다음과 같습니다.

```
Springer Nature (출판사)
│
├── Nature  ←── 본지 (주1회 발행, 모든 과학 분야 최상위)
│
├── Nature Research Journals (자매지 — 분야별)
│   ├── Nature Medicine
│   ├── Nature Machine Intelligence
│   ├── Nature Computational Science   ← 이 논문 1순위
│   ├── Nature Electronics
│   ├── Nature Communications          ← 이 논문 2순위
│   ├── Nature Biotechnology
│   ├── Nature Physics  ... 등 50여개
│
└── npj (Nature Partner Journals) — 외부 기관과 공동 발행
    ├── npj Computational Intelligence ← 이 논문 3순위
    └── npj Digital Medicine ... 등
```

**핵심 차이점:**

- **Nature 본지** 투고 = 노벨상급 발견 수준 요구. CS 논문은 사실상 불가에 가까움
- **자매지** 투고 = 각 저널의 편집팀이 독립적으로 운영. 본지와 별도 심사
- 같은 "Nature" 브랜드지만 **편집위원회, 심사 기준, APC가 모두 다름**
- 자매지 탈락 후 다른 자매지로 **원고 이전(manuscript transfer) 서비스** 제공
- 구독 방식(Subscription)으로 제출 시 **게재료 무료**
- Open Access 선택 시 약 $11,690 USD (2024 기준)
- 삼성전자의 Springer Nature 기관 협약 여부는 사내 도서관 확인 권장

---

## 7. 저널 추천 우선순위

### 1순위: Nature Computational Science ★★★★★

```
선택 이유:
- "대규모 시뮬레이션·HPC·데이터 기반 과학 연구" → Nature Computational Science
  (선택 가이드 직접 해당)
- 계산 과학 + 수학적 모델링 + 실험 검증 구조가 저널 성격과 정확히 일치
- Phase transition 발견이라는 다학제적 언어가 이 저널 심사위원에게 친숙
- Nature Machine Intelligence보다 AI 학계 보이콧 영향 없음
- 삼성전자 SAIT의 Nature Communications 선례(유현승, 함돈회)가 심사 신뢰도에 긍정적
```

### 2순위: Nature Communications ★★★★

```
선택 이유:
- 오픈 액세스 → 피인용 접근성 극대화 (H-Index 300+ 저널)
- 삼성전자 직원 1저자 게재 선례 명확히 존재
  · 유현승(SAIT) → Nature Communications, 2023
  · 안중권(SAIT) → Nature Communications, 2020
- 심사 난이도가 상대적으로 낮아 게재 가능성 현실적
- 다분야 융합 논문에 유리 (AI + 시스템 + 물리 유사 현상)
- 탈락 시 Nature Computational Science → Nature Communications 순으로
  동일 원고를 빠르게 재투고 가능
```

### 3순위: npj Computational Intelligence ★★★

```
선택 이유:
- 비교적 새로운 저널로 AI·CS 모두 수용
- 1·2순위 탈락 시 안전망
- Impact Factor 축적 중 → 지금 게재 시 선도 논문으로 인용 효과 기대
```

### 제외 대상: Nature Machine Intelligence

AI 학계 전체 보이콧 운동이 진행 중이며, 오픈 액세스를 중시하는 AI 커뮤니티에서 배척받고 있음. 피인용 파급력 측면에서 불리.

---

## 8. 단계별 투고 전략

### Step 1 — arXiv 선공개 (즉시 가능)

```
Nature Medicine 사례(arXiv 2024 → Nature Medicine 2025)처럼
preprint를 먼저 공개하여 커뮤니티 반응 수집 및 선점 효과 확보.
투고 시 preprint 사실을 투명하게 고지 (자기표절 아님 — 정상적 관행).
```

### Step 2 — 영어 편집 서비스

```
Nature 투고 전 전문 영어 편집 필수.
- Springer Nature Author Services (공식)
- Editage (editage.co.kr)
```

### Step 3 — 투고 순서

```
[1차] Nature Computational Science
       ↓ (탈락 시, 약 2~4개월 후)
[2차] Nature Communications
       ↓ (탈락 시)
[3차] npj Computational Intelligence
```

### Step 4 — 논문 프레이밍 강화 포인트

```
현재 abstract의 "Here we show..." 구조는 Nature 스타일에 이미 맞음.
심사 통과를 위해 강조해야 할 요소:

1. "Phase transition" 유사성을 물리학 언어로 더 명시
   → Nature Computational Science 심사위원 설득력 ↑

2. "General principle beyond LLM" 확장성 강조
   → AI를 넘어 스토리지 시스템, 뇌-신경 계산 등으로 연결

3. 삼성전자 실제 인프라 규모 데이터 인용
   → 산업적 임팩트 명시
```

---

## 9. 리뷰 절차 및 게재료

### 리뷰 절차 (통상 4~6개월)

| 단계 | 소요 기간 | 내용 |
|------|-----------|------|
| Desk review (편집장 사전 검토) | 1~2주 | Scope 부적합 시 즉시 반려 |
| Peer review (외부 심사) | 8~14주 | 2~3명 전문가 심사 |
| 1차 결정 | — | Accept / Major revision / Minor revision / Reject |
| 수정 및 재심사 | 4~8주 | 통상 1~2 라운드 |
| 최종 승인 | 1~2주 | 게재 확정 |
| **총 소요** | **4~6개월** | 빠르면 3개월 |

### 게재료 (APC)

| 출판 방식 | 게재료 |
|-----------|--------|
| 구독 방식 (Subscription) | **무료** (저자 부담 없음) |
| Open Access | 약 $11,690 USD (2024 기준) |

> **결론:** 구독 방식으로 제출하면 **게재료 무료**입니다.
> 단, 독자는 구독 없이 열람 불가.
> 삼성전자는 Springer Nature와 기관 협약(Read & Publish)을 맺고 있을 가능성이 높으므로 소속 도서관에 확인 권장.
