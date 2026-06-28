# ORION — Regime-dependent limits of hierarchical memory orchestration in large-scale AI inference

LaTeX manuscript targeting **Nature Computational Science** (1순위) or **Nature Communications** (2순위).

---

## Repository structure

```
.
├── main.tex                  # IEEEtran template (draft / internal review)
├── main_nature.tex           # Springer Nature sn-jnl template (submission)
├── sn-jnl.cls                # Springer Nature journal class (required)
├── IEEEtran.cls              # IEEE class (for main.tex)
├── IEEEtranDOI.bst           # IEEE BibTeX style
├── reference-data.bib        # Bibliography database
├── latexmkrc                 # latexmk configuration (timezone)
├── run.sh                    # Build script
├── figures/                  # All figures (PNG)
└── section/                  # Per-section .tex files
    ├── 001_title.tex
    ├── 005_author.tex        # IEEEtran author block
    ├── 005_author_nature.tex # sn-jnl author block
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

## 1. Install dependencies (Ubuntu 24.04)

```bash
# Core TeX Live packages
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

# PDF viewer
sudo apt-get install -y evince
```

> **Note:** `texlive-science` provides `algorithm.sty` and `algorithmicx.sty`
> required by this manuscript.

---

## 2. Build

### Nature template (for submission)

```bash
./run.sh nature
```

Produces `main_nature.pdf`.

### IEEEtran template (internal draft)

```bash
./run.sh
```

Produces `main.pdf`.

### What `run.sh` does internally

```
pdflatex  →  bibtex  →  pdflatex  →  pdflatex
```

---

## 3. View the PDF

```bash
# Nature submission PDF
evince main_nature.pdf

# IEEEtran draft PDF
evince main.pdf
```

Other viewers:

```bash
xdg-open main_nature.pdf     # system default viewer
okular main_nature.pdf        # KDE viewer
zathura main_nature.pdf       # lightweight viewer
```

---

## 4. Anonymity switch

The `\anonymous` flag in `main.tex` / `main_nature.tex` controls author visibility:

| Value | Effect |
|-------|--------|
| `1`   | Real author name and affiliation shown |
| `0`   | "Anonymous Author(s)" for blind review |

---

## 5. Nature 저널 구조 이해

Nature 저널은 **본지(flagship)** 와 **자매지(sister journals)** 로 구성됩니다.
투고 전 이 구조를 이해하는 것이 중요합니다.

```
Springer Nature (출판사)
│
├── Nature  ←── 본지 (1869년 창간, 모든 과학 분야 최상위)
│               노벨상급 발견 수준 요구. CS 논문은 사실상 투고 불가.
│
├── Nature Research Journals (분야별 자매지, 각각 독립 편집위원회)
│   ├── Nature Computational Science   ← 이 논문 1순위 투고 대상
│   ├── Nature Communications          ← 이 논문 2순위 투고 대상
│   ├── Nature Machine Intelligence    (AI 학계 보이콧 진행 중 — 제외)
│   ├── Nature Medicine
│   ├── Nature Electronics
│   ├── Nature Biotechnology
│   └── Nature Physics ... 등 50여 개
│
└── npj (Nature Partner Journals) — 외부 기관과 공동 발행
    ├── npj Computational Intelligence ← 이 논문 3순위 안전망
    └── npj Digital Medicine ... 등
```

**핵심 포인트:**
- 자매지는 본지와 **편집위원회·심사 기준·APC가 모두 독립적**으로 운영됨
- 자매지 탈락 후 다른 자매지로 **원고 이전(manuscript transfer) 서비스** 제공
- 구독 방식(Subscription)으로 제출 시 **게재료 무료**
- Open Access 선택 시 약 $11,690 USD (2024 기준)
- 삼성전자의 Springer Nature 기관 협약 여부는 사내 도서관 확인 권장

---

## 6. 저널 추천 (우선순위)

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

AI 학계 전체 보이콧 운동이 진행 중이며, 오픈 액세스를 중시하는 AI 커뮤니티에서
배척받고 있음. 피인용 파급력 측면에서 불리.

---

## 7. 단계별 투고 전략

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

## 8. 리뷰 절차 및 게재료

| 단계 | 소요 기간 | 내용 |
|------|-----------|------|
| Desk review (편집장 사전 검토) | 1~2주 | Scope 부적합 시 즉시 반려 |
| Peer review (외부 심사) | 8~14주 | 2~3명 전문가 심사 |
| 1차 결정 | — | Accept / Major revision / Minor revision / Reject |
| 수정 및 재심사 | 4~8주 | 통상 1~2 라운드 |
| 최종 승인 | 1~2주 | 게재 확정 |
| **총 소요** | **4~6개월** | |

| 출판 방식 | 게재료 |
|-----------|--------|
| 구독 방식 (Subscription) | **무료** |
| Open Access | 약 $11,690 USD (2024 기준) |
