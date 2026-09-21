# 핸드오프 체크리스트 — 당신이 직접 하고, 저에게 넘길 것

> 헷갈리지 않도록 정리했습니다. **결국 당신이 저에게 줘야 하는 것은 딱 하나입니다:
> 실제 측정으로 만든 `records.jsonl` 파일(각 줄 `"source": "measured"`).**
> 형식이 헷갈리면 `code/results/prereg_example/records_EXAMPLE.jsonl`(채워진 샘플)과
> 그 폴더의 `README.md`(필드 설명)를 그대로 보고 따라 하면 됩니다.

---

## 한눈에 보기

| 단계 | 누가 | 무엇을 | 산출물 |
|------|------|--------|--------|
| STEP 1 | **당신** | 실제 서빙 스택 측정 코드를 seam에 연결 | 수정된 `prereg_harvest.py` |
| STEP 2 | **당신** | 캠페인 실행 → 레코드 생성 | `records.jsonl` (`source="measured"`) |
| STEP 3 | **당신** | 검증 후 저에게 전달 | 파일 1개 |
| STEP 4 | **제가** | 분석·CI·표·그림·본문 자동 반영 | 갱신된 원고 |

당신 몫은 STEP 1–3 (실제 하드웨어가 필요한 부분)뿐입니다. 나머지는 제가 합니다.

---

## STEP 1 — 측정 코드 연결 (프레임워크별 뼈대 채우기)

측정 코드는 `code/experiments/prereg_backends.py`에 프레임워크별 **뼈대 클래스**로
준비돼 있습니다 — 쓰시는 스택 하나의 `measure()`만 채우면 됩니다:

| 백엔드 id | 클래스 | 채울 것 |
|-----------|--------|---------|
| `vllm` | `VLLMBackend` | policy→offload 설정 매핑 + 실제 실행/타이밍 |
| `deepspeed` | `DeepSpeedBackend` | policy→ZeRO-Inference offload 매핑 + 타이밍 |
| `flexgen` | `FlexGenBackend` | policy→GPU/CPU/disk 퍼센트 split + per-stage 타이머 |
| `torch-cuda` | `TorchCudaBackend` | 모델 로드 + compute/transfer 함수 정의 |

각 클래스는 지금 일부러 `NotImplementedError`를 던집니다(합성 숫자를 증거로
둔갑시키는 경로를 원천 차단). compute와 host→device transfer를 **따로** 재는 부분은
`time_compute_transfer_cuda(compute_fn, transfer_fn, windows=10)`으로 **이미 구현돼**
있으니 그대로 호출하면 됩니다(warm-up 1회 자동 폐기 → 컴파일/런치 artifact 제거).
`measure()`가 **실제 1회 측정 결과**를 담은 `PointMeasurement`를 돌려주게 만드세요:

```python
def measure_operating_point(point, policy, run_idx, backend):
    # 1) point.r_c 를 실현: device 메모리 상한/offload 비율로 residency 맞추기
    # 2) point.r_b 를 실현: compute 규모를 조절해 overlap 맞추기
    # 3) policy 대로 오케스트레이션 실행 (vLLM / DeepSpeed / FlexGen)
    # 4) compute 와 host→device transfer 를 **따로** 타이밍 (CUDA events)
    #    → 참고: code/experiments/cuda_backend.py 에 CUDA-event 타이밍 예시 있음
    return PointMeasurement(
        t_comp_s=...,        # 분리 측정한 compute 시간
        t_transfer_s=...,    # 분리 측정한 transfer 시간 (= D / B_slow)
        t_total_s=...,       # end-to-end step 지연 (결과값)
        c_fast_bytes=...,    # 실제 사용 가능한 fast-tier 용량 (datasheet 아님)
        w_bytes=...,         # active working set (weights+activations+KV)
        d_bytes=...,         # step당 compulsory cross-tier 바이트
        d_nr_bytes=...,      # 그 중 non-resident 바이트
        b_slow_bytes_per_s=...,  # limiting link 의 sustained 대역폭 (peak 아님)
        windows=10,
        provenance="a100-node-1, driver 550.x, 2026-..., CUDA events",
    )
```

라벨(`label`)과 split(`train`/`heldout`)은 도구가 자동으로 채웁니다 — **손대지
마세요.**

---

## STEP 2 — 캠페인 실행

```bash
# (a) 하드웨어 없이 배관만 먼저 점검 — dry-run 은 증거가 아니라 거부됩니다
python code/experiments/prereg_harvest.py --dry-run -o dry.jsonl
python code/experiments/analyze_prereg.py dry.jsonl   # → dryrun 거부되면 정상

# (b) STEP 1 연결 후, 실제 측정 캠페인 (--backend 는 채운 프레임워크로: vllm/deepspeed/flexgen/torch-cuda)
python code/experiments/prereg_harvest.py --backend vllm --runs 5 -o records.jsonl
```

`records.jsonl` 이 나오면 당신 몫의 결과물입니다.

---

## STEP 3 — 검증 후 저에게 전달

전달 전에 스스로 두 가지만 확인하면 좋습니다(안 해도 제가 확인합니다):

```bash
# 스키마 검증 (필드 누락/라벨 오류 잡기) — 저장소 루트에서 실행
python - <<'PY'
import sys, json
sys.path.insert(0, "code")           # 'code' 는 패키지가 아니므로 경로에 추가
from experiments.prereg_harvest import validate_record
for line in open("records.jsonl"):
    validate_record(json.loads(line))
print("schema OK")
PY

# 예비 분석 (held-out 결과 미리보기)
python code/experiments/analyze_prereg.py records.jsonl --split heldout
```

그런 다음 **저에게는 `records.jsonl` 경로만 알려주시거나 파일을 올려주시면
됩니다.** (예: "records.jsonl 커밋했어" 또는 파일 첨부)

---

## 최소 요건 (이게 충족돼야 심사에서 통과)

- [ ] 가속기 아키텍처 **≥ 2종**, 각 **≥ 2대**의 물리 머신
- [ ] 모델 계열 **≥ 2종** (예: autoregressive LLM + retrieval 또는 vision)
- [ ] 실제 서빙 스택 (vLLM / DeepSpeed / FlexGen), 합성 layer 스택 아님
- [ ] `R_C=0.5`, `R_B=1` 를 감싸는 그리드 (도구가 생성)
- [ ] (point, policy)당 **독립 실행 ≥ 5회**
- [ ] **per-window raw 기록 보존** (제출본에 CI가 없었던 이유가 바로 이게 없어서였음)
- [ ] 모든 줄 `"source": "measured"`

---

## 자주 헷갈리는 지점 (미리 방지)

1. **`source` 태그**: 실제 데이터는 반드시 `"measured"`. `example`/`dryrun`/`simulated`
   은 분석기가 자동 거부합니다(안전장치).
2. **`split` 임의 변경 금지**: 도구가 시드로 고정합니다. 결과를 본 뒤 바꾸면
   사전등록이 깨집니다.
3. **`label` 손대지 말기**: `R_C`, `R_B` 로부터 분류기가 계산합니다.
4. **datasheet peak 금지**: `C_fast_bytes`, `B_slow_bytes_per_s` 는 실제 사용
   가능/실측 sustained 값으로.
5. **compute/transfer 분리 타이밍**: `T_comp_s` 와 `T_transfer_s` 를 합쳐서 total
   로 쓰면 안 됩니다(순환성 회피의 핵심).

---

## STEP 4 — 그 다음은 제가 합니다

`records.jsonl` 을 주시면 제가:
- 사전등록된 분석 실행 (H3 예측정확도+CI, H4 순위역전, H1 residency 비율),
- 결과를 `generated_results.tex` / 표 / 그림에 반영,
- Results·Discussion 본문을 결과에 맞게 갱신(양성/음성/불충분 모두 정직하게),
- `check_submission.py` 정직성 게이트와 테스트 통과 확인,
- 지정 브랜치 커밋 후 `main` 병합까지.

즉, **실제 하드웨어가 필요한 부분만 당신이, 그 앞뒤 문서·분석·원고 작업은 제가**
맡습니다.
