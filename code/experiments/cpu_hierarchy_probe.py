#!/usr/bin/env python3
"""
CPU 메모리 계층 레짐 프로브 — 실측(real measurement)

이 스크립트는 ORION 논문의 핵심 주장(작업집합 W가 빠른 메모리 용량을 넘어서면
capacity-limited 레짐으로 급격히 전환된다)을 *실제 실리콘*에서 검증한다.
가속기(A100/TPU/…)가 아니라, 이 실행 환경에 실재하는 CPU 캐시 계층
(L2 → L3(LLC) → DRAM)을 대상으로 한다.

정직성 원칙(코드 cuda_backend.py와 동일):
  - 모든 값은 time.perf_counter_ns 로 측정한 실제 시간이다. 합성/모델링 없음.
  - 빠른 메모리 용량 C_fast 는 이 CPU 의 LLC(L3) 실측 용량으로 고정한다.
  - R_C = C_fast / W. W 가 LLC 를 넘으면(R_C < 1) 접근이 DRAM 바운드가 되며
    접근당 지연이 계단식으로 증가한다 = capacity-limited 레짐.

측정 방법: 포인터 체이싱(pointer chasing).
  크기 W 의 배열에 무작위 순열로 만든 단일 사이클 순회 링크를 심어,
  각 접근이 다음 접근 주소에 데이터 의존하도록 만든다. 이렇게 하면 CPU 의
  하드웨어 프리페처와 명령어 수준 병렬성이 무력화되어, 측정값이 순수한
  메모리 계층 지연을 반영한다(캐시 히트 vs 미스).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import statistics
import tempfile
from pathlib import Path

import numpy as np


def load_chase_kernel():
    """Compile and load the small C kernel that removes Python-loop overhead."""
    source = Path(__file__).with_name("cpu_pointer_chase.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    library = Path(tempfile.gettempdir()) / f"orion_pointer_chase_{digest}.so"
    if not library.exists():
        subprocess.run(
            ["cc", "-O3", "-std=c11", "-fPIC", "-shared", str(source),
             "-o", str(library)], check=True
        )
    lib = ctypes.CDLL(str(library))
    fn = lib.chase_latency_ns
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    fn.restype = ctypes.c_double
    return fn


def detect_llc_bytes() -> int:
    """Read the last-level unified/data cache size exposed by Linux sysfs."""
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    candidates: list[tuple[int, int]] = []
    for index in cache_root.glob("index*"):
        try:
            cache_type = (index / "type").read_text().strip()
            level = int((index / "level").read_text().strip())
            raw_size = (index / "size").read_text().strip().upper()
            multiplier = 1024 if raw_size.endswith("K") else 1024**2
            size = int(raw_size[:-1]) * multiplier
        except (OSError, ValueError):
            continue
        if cache_type in {"Unified", "Data"}:
            candidates.append((level, size))
    if not candidates:
        raise RuntimeError("LLC size unavailable in Linux sysfs; pass --c-fast-mib")
    return max(candidates)[1]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def compiler_version() -> str:
    try:
        result = subprocess.run(["cc", "--version"], check=True, text=True,
                                capture_output=True)
        return result.stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def make_chase(n_elems: int, rng: np.random.Generator) -> np.ndarray:
    """n_elems 길이의 단일 사이클 순열을 next-index 배열로 만든다."""
    perm = rng.permutation(n_elems)
    nxt = np.empty(n_elems, dtype=np.int64)
    # perm[0] -> perm[1] -> ... -> perm[n-1] -> perm[0] 인 하나의 큰 사이클
    nxt[perm[:-1]] = perm[1:]
    nxt[perm[-1]] = perm[0]
    return nxt


def chase_latency_ns(kernel, nxt: np.ndarray, n_hops: int) -> float:
    """n_hops 번 포인터 체이싱하고 hop 당 평균 지연(ns)을 반환한다."""
    pointer = nxt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    return float(kernel(pointer, n_hops))


def run(c_fast_bytes: int, out_dir: Path, n_trials: int = 7) -> dict:
    rng = np.random.default_rng(20260818)
    kernel = load_chase_kernel()
    bytes_per_elem = 8  # int64

    # 작업집합 크기를 L2(수 MiB) 아래부터 DRAM(수백 MiB)까지 스윕
    working_sets_mib = [0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    records = []
    summary = []

    print(f"[실측] 플랫폼=CPU  C_fast(LLC)={c_fast_bytes/2**20:.0f} MiB  "
          f"trials={n_trials}")
    print(f"{'W (MiB)':>9} {'R_C':>7} {'regime':>18} {'lat/access (ns)':>18}")

    for w_mib in working_sets_mib:
        w_bytes = int(w_mib * 2**20)
        n_elems = max(64, w_bytes // bytes_per_elem)
        nxt = make_chase(n_elems, rng)
        # hop 수: 배열을 여러 번 순회해 통계적으로 안정화(단, 시간 상한)
        n_hops = min(4_000_000, max(400_000, n_elems * 4))

        # 워밍업 1회 후 n_trials 회 측정
        chase_latency_ns(kernel, nxt, min(n_hops, 200_000))
        trials = [chase_latency_ns(kernel, nxt, n_hops) for _ in range(n_trials)]

        r_c = c_fast_bytes / w_bytes
        regime = "resident/coord" if r_c >= 0.5 else "capacity-limited"
        mean = statistics.mean(trials)
        sd = statistics.pstdev(trials)
        print(f"{w_mib:>9.1f} {r_c:>7.3f} {regime:>18} "
              f"{mean:>10.2f} ± {sd:.2f}")

        for t in trials:
            records.append({
                "platform": "CPU (L3 LLC + DRAM)",
                "w_bytes": w_bytes, "r_c": r_c,
                "regime": regime, "lat_per_access_ns": t,
            })
        summary.append({"w_mib": w_mib, "r_c": r_c, "regime": regime,
                        "lat_ns_mean": mean, "lat_ns_sd": sd,
                        "n_trials": n_trials})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cpu_hierarchy_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    metadata = {
        "cpu_model": cpu_model(),
        "logical_cpus_visible": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "c_compiler": compiler_version(),
        "timer": "clock_gettime(CLOCK_MONOTONIC_RAW) inside compiled C kernel",
        "seed": 20260818,
        "warmup_hops_max": 200_000,
        "c_fast_bytes": c_fast_bytes,
        "c_fast_source": "Linux sysfs LLC auto-detection or --c-fast-mib override",
        "n_trials": n_trials,
    }
    (out_dir / "cpu_hierarchy_summary.json").write_text(
        json.dumps({"metadata": metadata, "c_fast_bytes": c_fast_bytes,
                    "points": summary}, indent=2))

    # 계단 전환(crossover) 정량화: LLC 상주 구간 대비 DRAM 바운드 구간의 지연 비
    resident = [s["lat_ns_mean"] for s in summary if s["r_c"] >= 1.0]
    dram = [s["lat_ns_mean"] for s in summary if s["r_c"] <= 0.1]
    if resident and dram:
        ratio = statistics.mean(dram) / statistics.mean(resident)
        print(f"\n[결과] LLC 상주 대비 DRAM 바운드 지연 배수 = {ratio:.1f}x "
              f"(상주 {statistics.mean(resident):.1f} ns → "
              f"DRAM {statistics.mean(dram):.1f} ns)")
    print(f"[저장] {out_dir}/cpu_hierarchy_records.jsonl (+summary.json)")
    return {"summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-fast-mib", type=float,
                    help="빠른 메모리(LLC) 용량 MiB (기본값: Linux sysfs 자동 감지)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "cpu_probe")
    ap.add_argument("--trials", type=int, default=7)
    args = ap.parse_args()
    c_fast_bytes = (int(args.c_fast_mib * 2**20)
                    if args.c_fast_mib is not None else detect_llc_bytes())
    run(c_fast_bytes, args.out, args.trials)


if __name__ == "__main__":
    main()
