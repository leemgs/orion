# 실행 순서 (copy / paste)

아래 블록을 위에서부터 그대로 붙여넣어 실행하면 됩니다. **STEP 0–2 는 GPU 없이**
(파이프라인 점검), **STEP 3 부터 GPU 필요** (실제 측정). 모든 경로는 저장소 루트
(`orion/`) 기준입니다.

---

## STEP 0 — 저장소 최신화

```bash
git clone https://github.com/leemgs/orion.git   # 이미 있으면 생략
cd orion
git checkout main
git pull origin main
```

## STEP 1 — 파이썬 환경 (GPU 불필요)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pytest
```

## STEP 2 — 배관 점검 (GPU 불필요)

```bash
# 2-1) 전체 테스트
python -m pytest code/tests -q

# 2-2) dry-run 으로 스키마/파이프라인 확인 (이 파일은 '증거'가 아니라 거부됨)
python code/experiments/prereg_harvest.py --dry-run -o records_dryrun.jsonl

# 2-3) 분석기가 dry-run 을 거부하면 정상 (안전장치 확인)
python code/experiments/analyze_prereg.py records_dryrun.jsonl || echo "OK: dry-run 거부됨(정상)"
```

## STEP 3 — 실제 측정 (GPU 필요, 코드 작성 없음)

`torch-reference` 백엔드가 **로컬 GPU에서 실제 연산·전송을 CUDA 이벤트로 측정**합니다.
코드를 쓸 필요 없이 명령만 실행하면 됩니다.

```bash
# 3-1) CUDA용 PyTorch 설치 (본인 CUDA 버전에 맞게; 예: cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3-2) GPU 인식 확인 (True 여야 함)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# 3-3) 이 머신에서 캠페인 실행 (--arch 는 실제 GPU 이름으로 정직하게)
python code/experiments/prereg_harvest.py \
  --backend torch-reference \
  --machine-id node-A --arch A100-80GB --model ref-workload \
  --runs 5 -o records_nodeA.jsonl
```

> **아키텍처 ≥2종 요건**: 다른 GPU(예: H100, MI250) 머신에서 STEP 3-3 을 `--arch`,
> `--machine-id` 만 바꿔 반복한 뒤, 파일을 합칩니다:
>
> ```bash
> cat records_nodeA.jsonl records_nodeB.jsonl > records.jsonl
> ```
>
> 머신이 하나뿐이면 우선 `records.jsonl` 로 이름만 바꿔 진행해도 됩니다:
>
> ```bash
> cp records_nodeA.jsonl records.jsonl
> ```

## STEP 4 — 검증 + 미리보기 (GPU 불필요)

```bash
# 4-1) 스키마 검증 (필드 누락/라벨 오류 잡기)
python - <<'PY'
import sys, json
sys.path.insert(0, "code")
from experiments.prereg_harvest import validate_record
n = 0
for line in open("records.jsonl"):
    validate_record(json.loads(line)); n += 1
print(f"schema OK: {n} rows")
PY

# 4-2) 사전등록 분석 미리보기 (held-out 결과)
python code/experiments/analyze_prereg.py records.jsonl --split heldout
```

`H3_prediction.beats_baseline` 이 `true` 로 나오면 좋은 신호입니다(반드시 그래야
하는 건 아니며, 음성이어도 정직하게 보고합니다).

## STEP 5 — 저에게 넘기기

```bash
# records.jsonl 을 저장소에 커밋 (원본 증거 보존)
mkdir -p code/results/prereg_campaign
cp records.jsonl code/results/prereg_campaign/records.jsonl
git add code/results/prereg_campaign/records.jsonl
git commit -m "Add preregistered campaign records"
git push origin main
```

그런 다음 채팅으로 **"records 커밋했어: code/results/prereg_campaign/records.jsonl"**
라고만 알려주세요. 이후 분석 실행 → 표/그림/CI/본문 반영 → 커밋·병합은 제가 합니다.

---

## (선택) 더 강력한 근거로 업그레이드

`torch-reference` 는 실제 하드웨어 측정이지만 통제된 마이크로 워크로드입니다.
심사에서 가장 강한 근거는 실제 서빙 스택 + 실제 모델입니다. 원할 때
`code/experiments/prereg_backends.py` 의 `VLLMBackend` / `DeepSpeedBackend` /
`FlexGenBackend` 중 하나의 `measure()` 를 채운 뒤:

```bash
python code/experiments/prereg_harvest.py \
  --backend vllm --machine-id node-A --arch A100-80GB --model llama-3-8b \
  --runs 5 -o records_vllm_nodeA.jsonl
```

작성이 헷갈리면 `handoff_checklist.md`(STEP 1)와
`code/results/prereg_example/README.md`(필드 사전)를 보세요.
