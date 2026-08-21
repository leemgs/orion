#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORION 레짐 실측 스크립트 (Colab GPU/TPU · 로컬 CPU 겸용)
========================================================

무엇을: 논문의 두 무차원 비율에 따른 레짐 전환을 *실제 하드웨어*에서 측정한다.
  - R_C = C_fast / W          (잔류 비율; 빠른 메모리에 든 작업집합 비율)
  - R_B = T_comp / T_transfer (오버랩 비율; 전송이 연산 뒤에 숨는 정도)
논문 정합화 후 정의(code/orion/{ratios,config}.py)와 동일하다:
  θ_C = 0.50 (다수-잔류 관례),  θ_B = 1.0 (유도된 오버랩 경계).

어떻게(정직성): 모든 시간은 실제 계측이다.
  - GPU: torch.cuda.Event 로 커널/복사 구간을 실측(synchronize 포함).
  - TPU: torch_xla + time.perf_counter (mark_step 동기화).
  - CPU: numpy + time.perf_counter (torch 불필요, 로컬 검증용).
합성·모델링 값은 없다. 측정 불가한 항목은 메모리에서 NaN으로 다루되,
결과 파일에는 표준 JSON의 null로 기록한다.
가중치는 무작위 초기화한다 — 메모리 계층 프로브의 지연은 텐서 shape/dtype/
잔류 상태에 좌우되지, 가중치 '값'에 좌우되지 않으므로 레짐 특성화에 타당하다
(cuda_backend.py 의 동일 원칙).

실행:
  # Colab (GPU 런타임 선택 후): 셀에서
  #   !pip -q install torch
  #   !python colab_regime_measurement.py
  # Colab (TPU): !pip -q install torch torch_xla 후 동일
  # 로컬 CPU 검증:
  #   python colab_regime_measurement.py --backend numpy --quick

출력:
  results/colab_probe/records.jsonl   (실측 원시 레코드)
  results/colab_probe/summary.json    (경계 추정 + 요약)
  화면에 표 + '논문 붙여넣기용' 요약 블록.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

NAN = float("nan")


# ----------------------------------------------------------------------------
# 백엔드 추상화: 어떤 백엔드든 "실제" 연산/전송/타이밍을 반환한다.
# ----------------------------------------------------------------------------
class Backend:
    name = "base"
    device = "?"

    def new_weight(self, d: int):
        """디바이스 상의 (d,d) 무작위 가중치 행렬."""
        raise NotImplementedError

    def new_host_weight(self, d: int):
        """호스트(오프로딩 원본) 상의 (d,d) 가중치."""
        raise NotImplementedError

    def transfer_in(self, host_w):
        """호스트→디바이스 복사. (device_w, 실측_초) 반환."""
        raise NotImplementedError

    def matmul(self, x, w, repeats: int):
        """x@w 를 repeats 회. (결과, 실측_초) 반환."""
        raise NotImplementedError

    def new_activation(self, batch: int, d: int):
        raise NotImplementedError


class NumpyBackend(Backend):
    name, device = "numpy", "CPU"

    def __init__(self):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(20260818)

    def new_weight(self, d):
        # 1/sqrt(d) 스케일 → 반복 matmul 이 대략 노름 보존(오버플로 방지)
        return (self.rng.standard_normal((d, d), dtype="float32")
                / self.np.float32(d ** 0.5))

    new_host_weight = new_weight

    def transfer_in(self, host_w):
        t0 = time.perf_counter()
        dev = host_w.copy()          # 실제 memcpy
        t1 = time.perf_counter()
        return dev, (t1 - t0)

    def matmul(self, x, w, repeats):
        t0 = time.perf_counter()
        y = x
        for _ in range(repeats):
            y = self.np.matmul(y, w)
        t1 = time.perf_counter()
        if y.shape[0] < 0:            # 결과 소비(최적화 방지)
            raise RuntimeError
        return y, (t1 - t0)

    def new_activation(self, batch, d):
        return self.rng.standard_normal((batch, d), dtype="float32")


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, want_tpu=False):
        import torch
        self.torch = torch
        self.is_xla = False
        if want_tpu:
            import torch_xla.core.xla_model as xm  # noqa
            self.xm = xm
            self.dev = xm.xla_device()
            self.is_xla = True
            self.device = "TPU(XLA)"
        elif torch.cuda.is_available():
            self.dev = torch.device("cuda")
            self.device = f"GPU:{torch.cuda.get_device_name(0)}"
        else:
            self.dev = torch.device("cpu")
            self.device = "CPU(torch)"
        torch.manual_seed(20260818)

    def _sync(self):
        if self.is_xla:
            self.xm.mark_step()
            self.xm.wait_device_ops()
        elif self.dev.type == "cuda":
            self.torch.cuda.synchronize()

    def new_weight(self, d):
        return self.torch.randn(d, d, device=self.dev,
                                dtype=self.torch.float32) / (d ** 0.5)

    def new_host_weight(self, d):
        w = self.torch.randn(d, d, dtype=self.torch.float32) / (d ** 0.5)
        if self.dev.type == "cuda":
            w = w.pin_memory()       # 고정 메모리 → 실제 DMA 경로
        return w

    def transfer_in(self, host_w):
        t = self.torch
        if self.dev.type == "cuda":
            s = t.cuda.Event(enable_timing=True); e = t.cuda.Event(enable_timing=True)
            t.cuda.synchronize(); s.record()
            dev = host_w.to(self.dev, non_blocking=True)
            e.record(); t.cuda.synchronize()
            return dev, s.elapsed_time(e) / 1e3   # ms→s, 실측
        t0 = time.perf_counter()
        dev = host_w.to(self.dev)
        self._sync()
        return dev, (time.perf_counter() - t0)

    def matmul(self, x, w, repeats):
        t = self.torch
        if self.dev.type == "cuda":
            s = t.cuda.Event(enable_timing=True); e = t.cuda.Event(enable_timing=True)
            t.cuda.synchronize(); s.record()
            y = x
            for _ in range(repeats):
                y = y @ w
            e.record(); t.cuda.synchronize()
            return y, s.elapsed_time(e) / 1e3
        t0 = time.perf_counter()
        y = x
        for _ in range(repeats):
            y = y @ w
        self._sync()
        return y, (time.perf_counter() - t0)

    def new_activation(self, batch, d):
        return self.torch.randn(batch, d, device=self.dev, dtype=self.torch.float32)


def make_backend(kind: str) -> Backend:
    if kind == "numpy":
        return NumpyBackend()
    if kind == "tpu":
        return TorchBackend(want_tpu=True)
    if kind == "torch":
        return TorchBackend(want_tpu=False)
    # auto
    try:
        import torch_xla.core.xla_model  # noqa
        return TorchBackend(want_tpu=True)
    except Exception:
        pass
    try:
        import torch  # noqa
        return TorchBackend(want_tpu=False)
    except Exception:
        return NumpyBackend()


# ----------------------------------------------------------------------------
# 측정 커널: 한 (R_C, R_B) 동작점에서 T_comp / T_transfer / T_total 실측
# ----------------------------------------------------------------------------
def measure_point(be: Backend, d: int, n_layers: int, batch: int,
                  resident_frac: float, comp_repeats: int, n_windows: int):
    """
    L=n_layers 개 층 중 resident_frac 만큼만 디바이스에 상주.
    나머지는 매 스텝 호스트→디바이스로 복사(오프로딩 전송량 D).
    반환: dict(r_c 근사, t_comp, t_transfer, t_total, r_b) — 모두 실측 평균.
    """
    n_res = max(0, min(n_layers, round(resident_frac * n_layers)))
    n_off = n_layers - n_res
    resident = [be.new_weight(d) for _ in range(n_res)]
    host_off = [be.new_host_weight(d) for _ in range(n_off)]

    t_comp_l, t_tx_l, t_tot_l = [], [], []
    for _ in range(n_windows):
        x = be.new_activation(batch, d)
        step_t0 = time.perf_counter()
        # (1) 오프로딩 층: 전송 후 연산
        tx = 0.0; comp = 0.0
        for hw in host_off:
            dev_w, dt = be.transfer_in(hw); tx += dt
            x, ct = be.matmul(x, dev_w, comp_repeats); comp += ct
        # (2) 상주 층: 연산만
        for rw in resident:
            x, ct = be.matmul(x, rw, comp_repeats); comp += ct
        t_tot_l.append(time.perf_counter() - step_t0)
        t_comp_l.append(comp); t_tx_l.append(tx)

    t_comp = statistics.mean(t_comp_l)
    t_tx = statistics.mean(t_tx_l)
    t_tot = statistics.mean(t_tot_l)
    # W ~ 전체 층, C_fast ~ 상주 층  →  R_C = 상주/전체
    r_c = n_res / n_layers if n_layers else NAN
    # R_B = T_comp / T_transfer (오프로딩이 있을 때만 정의)
    r_b = (t_comp / t_tx) if t_tx > 0 else NAN
    return {"r_c": r_c, "t_comp_s": t_comp, "t_transfer_s": t_tx,
            "t_total_s": t_tot, "r_b": r_b,
            "t_total_sd": statistics.pstdev(t_tot_l)}


def classify(r_c, r_b, theta_c=0.50, theta_b=1.0):
    if r_c < theta_c:
        return "capacity-limited"
    if r_b == r_b and r_b < theta_b:   # r_b not NaN
        return "io-limited"
    return "coordination-dominated"


def json_safe(value):
    """Recursively replace non-finite floats so output is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


# ----------------------------------------------------------------------------
# 스윕 드라이버
# ----------------------------------------------------------------------------
def run(be: Backend, d, n_layers, batch, n_windows, quick):
    print(f"[실측] backend={be.name} device={be.device} "
          f"d={d} layers={n_layers} batch={batch} windows={n_windows}")
    records, summary = [], []

    # --- 스윕 A: 잔류 R_C (θ_C=0.50 근방 capacity 전환) : R_B 높게 유지 ---
    print("\n=== 스윕 A: R_C (잔류) — comp_repeats 큼(전송이 연산 뒤 숨음) ===")
    print(f"{'R_C':>6} {'regime':>20} {'T_total(ms)':>13} {'R_B':>7}")
    rc_grid = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0] if quick else \
              [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 0.9, 1.0]
    for rc in rc_grid:
        m = measure_point(be, d, n_layers, batch, resident_frac=rc,
                          comp_repeats=(3 if quick else 6), n_windows=n_windows)
        reg = classify(m["r_c"], m["r_b"])
        print(f"{m['r_c']:>6.2f} {reg:>20} {m['t_total_s']*1e3:>13.2f} "
              f"{m['r_b']:>7.2f}")
        m.update(sweep="R_C", regime=reg, device=be.device); summary.append(m)
        records.append(dict(m))

    # --- 스윕 B: 오버랩 R_B (θ_B=1.0 근방 I/O 전환) : R_C 높게(상주) 유지 ---
    print("\n=== 스윕 B: R_B (오버랩) — 일부 오프로딩, 연산량을 스케일 ===")
    print(f"{'comp_rep':>8} {'regime':>20} {'T_total(ms)':>13} {'R_B':>7}")
    rep_grid = [1, 2, 4, 8] if quick else [1, 2, 3, 4, 6, 8, 12]
    for rep in rep_grid:
        # 상주 0.6(용량 충분) + 소량 오프로딩으로 전송 존재 → R_B 를 rep 로 조절
        m = measure_point(be, d, n_layers, batch, resident_frac=0.6,
                          comp_repeats=rep, n_windows=n_windows)
        reg = classify(m["r_c"], m["r_b"])
        print(f"{rep:>8} {reg:>20} {m['t_total_s']*1e3:>13.2f} {m['r_b']:>7.2f}")
        m.update(sweep="R_B", regime=reg, comp_repeats=rep, device=be.device)
        summary.append(m); records.append(dict(m))

    out = Path(__file__).resolve().parent.parent / "results" / "colab_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(
        "\n".join(json.dumps(json_safe(r), allow_nan=False) for r in records)
        + "\n")
    (out / "summary.json").write_text(json.dumps(json_safe(
        {"device": be.device, "backend": be.name, "d": d,
         "n_layers": n_layers, "batch": batch, "python": platform.python_version(),
         "theta_C": 0.50, "theta_B": 1.0, "points": summary}),
         indent=2, allow_nan=False))

    # --- 논문 붙여넣기용 요약 ---
    a = [s for s in summary if s["sweep"] == "R_C"]
    cap = [s["t_total_s"] for s in a if s["r_c"] < 0.5]
    res = [s["t_total_s"] for s in a if s["r_c"] >= 1.0]
    print("\n" + "=" * 60)
    print("[논문 붙여넣기용 요약 — 이 블록을 그대로 전달해 주세요]")
    print(f"device = {be.device}")
    if cap and res:
        print(f"capacity-limited(R_C<0.5) 평균 T_total = "
              f"{statistics.mean(cap)*1e3:.2f} ms")
        print(f"resident(R_C>=1.0)      평균 T_total = "
              f"{statistics.mean(res)*1e3:.2f} ms")
        print(f"용량 전환 배수 = {statistics.mean(cap)/statistics.mean(res):.2f}x")
    b = [s for s in summary if s["sweep"] == "R_B"]
    io = [s["t_total_s"] for s in b if s["r_b"] == s["r_b"] and s["r_b"] < 1.0]
    hid = [s["t_total_s"] for s in b if s["r_b"] == s["r_b"] and s["r_b"] >= 1.0]
    if io and hid:
        print(f"io-limited(R_B<1)  평균 T_total = {statistics.mean(io)*1e3:.2f} ms")
        print(f"overlapped(R_B>=1) 평균 T_total = {statistics.mean(hid)*1e3:.2f} ms")
    print(f"저장: {out}/records.jsonl , summary.json")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "torch", "tpu", "numpy"],
                    default="auto")
    ap.add_argument("--d", type=int, default=2048, help="은닉 차원(정사각 행렬)")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="격자 축소(빠른 검증)")
    args = ap.parse_args()
    be = make_backend(args.backend)
    run(be, args.d, args.layers, args.batch, args.windows, args.quick)


if __name__ == "__main__":
    main()
