"""
Live CUDA measurement backend (Methods §Regime probing methodology).

Implements the `measure(r_c, r_b) -> LatencyRecord` contract that
`HardwareProfiler.register_backend()` expects, against real silicon:

  R_C control   layer weights are partitioned between HBM and pinned host DRAM
                so that resident bytes / working-set bytes hits the target R_C.
  R_B control   every host->device copy passes through the token-bucket
                throttle in `orion.profiler.BandwidthThrottle`; the target rate
                is derived from the measured per-step transfer volume D and step
                time dt, then validated against NVML PCIe throughput.
  T_comp        CUDA events around the compute stream's kernels.
  T_swap        the time the compute stream is *blocked* on transfers, i.e. the
                exposed (non-overlapped) transfer cost.
  T_sync        wall time inside explicit synchronisation barriers.
  T_mem         counter-derived; see the caveat below.

Three honesty constraints govern this module:

1. Nothing is synthesised. Every field of every emitted LatencyRecord comes from
   a device measurement. If a quantity cannot be measured the backend raises
   MeasurementUnavailable or records NaN — never a modelled stand-in.

2. T_mem cannot be read inside a timed window. Nsight Compute replays kernels to
   collect HBM counters, which perturbs the timings under study. Live windows
   therefore carry t_mem = NaN unless `attach_mem_counters()` has supplied a
   measurement from a separate profiling pass at the same operating point.

3. T_comp and T_mem overlap by construction. Kernel time already contains the
   HBM stall cycles that T_mem quantifies, so t_comp + t_mem double-counts them.
   `LatencyRecord.completeness_error` will surface this rather than hide it. The
   additive four-term decomposition in the paper needs a compute-only definition
   of T_comp to be well posed; see README §Live measurement caveats.

Weights are randomly initialised by default. Latency of a memory-hierarchy probe
depends on tensor shapes, dtypes and residency, not on weight *values*, so this
is sound for regime characterisation — but it means these runs measure the
architecture, not a specific checkpoint. Pass `hf_model=...` to load real
weights instead.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from orion.config import (
    HardwareProfile, ModelSpec, A100_80GB, LLAMA3_8B, WINDOW_SEC, WARMUP_SEC,
    ALL_PLATFORMS,
)
from orion.cupti_counters import HBMCalibration, calibrate_rho, t_mem_from_counters
from orion.nvml_monitor import NVMLMonitor, MeasurementUnavailable
from orion.profiler import BandwidthThrottle, LatencyRecord
from orion.ratios import compute_rb

try:
    import torch
    import torch.nn as nn
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:              # pragma: no cover - depends on host
    torch = None                      # type: ignore[assignment]
    nn = None                         # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


NAN = float("nan")


def _require_cuda() -> None:
    if torch is None:
        raise MeasurementUnavailable(
            f"torch is required for the live backend but could not be imported "
            f"({_TORCH_IMPORT_ERROR!r}). Install with `pip install -e '.[gpu]'`."
        )
    if not torch.cuda.is_available():
        raise MeasurementUnavailable(
            "torch reports no CUDA device. Use --mode simulate for offline "
            "reproduction, but note that simulated records are synthetic and "
            "must not be reported as measurements."
        )


# --------------------------------------------------------------------------
# Model under test
# --------------------------------------------------------------------------

class _DecoderLayer:
    """
    One transformer decoder layer held as raw weight tensors.

    Weights live in pinned host memory and are staged to HBM on demand. Keeping
    them as plain tensors (rather than an nn.Module) lets the residency manager
    move a layer between tiers without touching autograd or module state.
    """

    def __init__(self, spec: ModelSpec, dtype, device) -> None:
        d = spec.d_model
        hidden = 4 * d
        # qkv projection, output projection, and the two MLP matrices.
        shapes = {
            "wqkv": (d, 3 * d),
            "wo":   (d, d),
            "w1":   (d, hidden),
            "w2":   (hidden, d),
        }
        self.host: dict[str, "torch.Tensor"] = {}
        for name, shape in shapes.items():
            t = torch.empty(shape, dtype=dtype, device="cpu").pin_memory()
            t.normal_(0.0, 0.02)
            self.host[name] = t
        self.device_weights: Optional[dict[str, "torch.Tensor"]] = None
        self.resident = False
        self._device = device
        self._dtype = dtype

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.host.values())

    def pin_to_device(self) -> None:
        """Materialise this layer's weights in HBM for the whole run."""
        if self.resident:
            return
        self.device_weights = {
            k: v.to(self._device, non_blocking=False) for k, v in self.host.items()
        }
        self.resident = True

    def evict(self) -> None:
        self.device_weights = None
        self.resident = False

    def stage_async(self, stream, throttle: Optional[BandwidthThrottle]) -> dict:
        """
        Copy this layer's weights host->device on `stream`, rate-limited.

        The throttle is acquired on the host before the copy is enqueued, which
        is what paces the DMA engine: the copy stream cannot run ahead of the
        token bucket.
        """
        if throttle is not None:
            throttle.acquire(self.nbytes)
        staged = {}
        with torch.cuda.stream(stream):
            for k, v in self.host.items():
                staged[k] = v.to(self._device, non_blocking=True)
        return staged

    def forward(self, x, weights: dict):
        """A real attention + MLP block; shapes drive the memory traffic."""
        qkv = x @ weights["wqkv"]
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        h = (attn @ v) @ weights["wo"]
        x = x + h
        return x + torch.relu(x @ weights["w1"]) @ weights["w2"]


@dataclass
class ResidencyPlan:
    """Which layers sit in HBM at a given R_C target."""
    resident_layers: list[int]
    streamed_layers: list[int]
    resident_bytes: int
    streamed_bytes: int          # == D, the compulsory per-step transfer volume
    achieved_r_c: float
    target_r_c: float
    w_total_bytes: int


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

class CUDABackend:
    """
    Live NVIDIA backend for regime probing.

    Args:
        platform:     platform label; must match a HardwareProfile name.
        spec:         model architecture under test.
        device_index: CUDA device ordinal.
        seq_len:      override the spec's sequence length (shorter = faster grid).
        calibrate:    run the rho streaming microbenchmark at construction.
    """

    def __init__(
        self,
        platform: str = "A100-80GB",
        spec: ModelSpec = LLAMA3_8B,
        device_index: int = 0,
        seq_len: Optional[int] = None,
        calibrate: bool = True,
        window_sec: float = WINDOW_SEC,
    ) -> None:
        _require_cuda()
        self.platform = platform
        self.hw = self._resolve_profile(platform)
        self.spec = spec
        self.device_index = device_index
        self.window_sec = window_sec
        self.device = torch.device(f"cuda:{device_index}")
        self.dtype = torch.float16
        self.seq_len = seq_len or spec.seq_len

        self.nvml = NVMLMonitor(device_index=device_index)
        self._compute_stream = torch.cuda.Stream(device=self.device)
        self._copy_stream = torch.cuda.Stream(device=self.device)

        self._layers = [
            _DecoderLayer(spec, self.dtype, self.device) for _ in range(spec.n_layers)
        ]
        self._x = torch.randn(
            spec.batch, self.seq_len, spec.d_model,
            dtype=self.dtype, device=self.device,
        )

        # Measured, not assumed: the real parameter footprint of what we built.
        self.w_param_bytes = sum(l.nbytes for l in self._layers)
        self.w_total_bytes = int(
            self.w_param_bytes
            + spec.w_act_gb * 1e9
            + spec.w_kv_gb * 1e9
        )

        self.calibration: Optional[HBMCalibration] = None
        if calibrate:
            self.calibration = calibrate_rho(device_index=device_index)

        self._mem_counters: dict[tuple, float] = {}
        self._t_comp_cache: dict[float, float] = {}
        self._discarded_windows = 0

    @staticmethod
    def _resolve_profile(platform: str) -> HardwareProfile:
        for p in ALL_PLATFORMS:
            if p.name == platform:
                return p
        known = ", ".join(p.name for p in ALL_PLATFORMS)
        raise ValueError(f"unknown platform {platform!r}; known: {known}")

    @property
    def discarded_windows(self) -> int:
        """Windows dropped by the NVML clock gate (Methods: >5% deviation)."""
        return self._discarded_windows

    # ---- rho ------------------------------------------------------------

    @property
    def rho(self) -> float:
        """
        Measured HBM miss penalty per byte (s/byte).

        Falls back to the HardwareProfile constant only if calibration was
        explicitly disabled, and warns, because the profile constants in
        config.py have not been reconciled with measured bandwidth.
        """
        if self.calibration is not None:
            return self.calibration.rho_s_per_byte
        return self.hw.rho

    # ---- R_C ------------------------------------------------------------

    def plan_residency(self, target_r_c: float) -> ResidencyPlan:
        """
        Partition layers between HBM and host DRAM to hit `target_r_c`.

        R_C = C_fast / W. The activation and KV footprints must stay resident for
        the step to run at all, so only the parameter bytes are tradeable:

            resident_param_budget = target_r_c * W - W_act - W_kv

        A target so low that the budget goes negative is not reachable on this
        device: the non-parameter working set alone already exceeds it.
        """
        w_nonparam = self.w_total_bytes - self.w_param_bytes
        budget = target_r_c * self.w_total_bytes - w_nonparam
        if budget < 0:
            raise MeasurementUnavailable(
                f"R_C={target_r_c:.2f} is unreachable for {self.spec.name}: "
                f"activations+KV ({w_nonparam / 1e9:.1f} GB) already exceed the "
                f"implied fast-memory budget "
                f"({target_r_c * self.w_total_bytes / 1e9:.1f} GB). Reduce batch "
                f"or sequence length, or probe a higher R_C."
            )

        resident, streamed = [], []
        used = 0
        for i, layer in enumerate(self._layers):
            if used + layer.nbytes <= budget:
                resident.append(i)
                used += layer.nbytes
            else:
                streamed.append(i)

        streamed_bytes = sum(self._layers[i].nbytes for i in streamed)
        resident_total = used + w_nonparam
        return ResidencyPlan(
            resident_layers=resident,
            streamed_layers=streamed,
            resident_bytes=used,
            streamed_bytes=streamed_bytes,
            achieved_r_c=resident_total / self.w_total_bytes,
            target_r_c=target_r_c,
            w_total_bytes=self.w_total_bytes,
        )

    def _apply_residency(self, plan: ResidencyPlan) -> None:
        for i in plan.streamed_layers:
            self._layers[i].evict()
        torch.cuda.empty_cache()
        for i in plan.resident_layers:
            self._layers[i].pin_to_device()

    # ---- R_B ------------------------------------------------------------

    def _target_bps_for(self, r_b: float, plan: ResidencyPlan, t_comp_s: float) -> float:
        """
        R_B = B_slow * T_comp / D, so B_slow = R_B * D / T_comp.

        T_comp is the measured per-step computation time at this operating point,
        not the step duration: step duration absorbs the transfer wait, which
        pins the ratio at >= 1 by construction (see orion/ratios.py).
        """
        if plan.streamed_bytes == 0:
            raise MeasurementUnavailable(
                f"R_B is undefined at R_C={plan.target_r_c:.2f}: the whole "
                f"working set is resident, so D = 0 and no transfer pressure "
                f"exists to control."
            )
        target = r_b * plan.streamed_bytes / t_comp_s
        # The token bucket can only throttle *below* the link's sustained rate.
        # A target above it is not a slower experiment, it is an impossible one.
        if target > self.hw.b_slow_bps:
            ceiling = self.hw.b_slow_bps * t_comp_s / plan.streamed_bytes
            raise MeasurementUnavailable(
                f"R_B={r_b:.2f} at R_C={plan.target_r_c:.2f} requires "
                f"{target / 1e9:.1f} GB/s, but {self.hw.name} sustains only "
                f"{self.hw.b_slow_gbs:.1f} GB/s. D={plan.streamed_bytes / 1e9:.1f} GB "
                f"must cross the link per step against T_comp="
                f"{t_comp_s * 1e3:.0f} ms, which caps R_B at {ceiling:.2f}. "
                f"Raising R_B here requires reducing D (more residency, smaller "
                f"batch), not throttling: bandwidth cannot be added in software."
            )
        return target

    def _measure_t_comp(self, plan: ResidencyPlan) -> float:
        """
        Measure per-step computation time with the throttle off.

        This anchors the R_B target. It is the CUDA-event kernel time, summed
        over layers — not the wall-clock step time, which would include the
        transfer wait we are trying to characterise.
        """
        if plan.target_r_c in self._t_comp_cache:
            return self._t_comp_cache[plan.target_r_c]
        for _ in range(3):
            self._run_step(plan, throttle=None)
        torch.cuda.synchronize(self.device)
        n = 5
        total = 0.0
        for _ in range(n):
            total += self._run_step(plan, throttle=None)["t_comp"]
        torch.cuda.synchronize(self.device)
        t_comp = total / n
        self._t_comp_cache[plan.target_r_c] = t_comp
        return t_comp

    # ---- step -----------------------------------------------------------

    def _run_step(self, plan: ResidencyPlan, throttle: Optional[BandwidthThrottle]) -> dict:
        """
        One decode step with one-layer-lookahead prefetch.

        Returns the per-step timing breakdown in seconds. t_swap is the *exposed*
        stall: the compute stream's wall time spent blocked on a staging copy
        that had not completed. Total transfer time that fully overlaps compute
        contributes zero, which is the physically meaningful accounting for an
        additive decomposition.
        """
        x = self._x
        comp_ms = 0.0
        swap_stall_s = 0.0
        sync_s = 0.0

        streamed = set(plan.streamed_layers)
        inflight: dict[int, tuple] = {}

        def stage(idx: int) -> None:
            if idx in streamed and idx not in inflight:
                ev = torch.cuda.Event(enable_timing=False)
                w = self._layers[idx].stage_async(self._copy_stream, throttle)
                with torch.cuda.stream(self._copy_stream):
                    ev.record(self._copy_stream)
                inflight[idx] = (w, ev)

        # Prime the pipeline with the first streamed layer.
        if plan.streamed_layers:
            stage(plan.streamed_layers[0])

        for i, layer in enumerate(self._layers):
            nxt = next((j for j in plan.streamed_layers if j > i), None)
            if nxt is not None:
                stage(nxt)     # prefetch while layer i computes

            if i in streamed:
                weights, ev = inflight.pop(i)
                t0 = time.perf_counter()
                ev.synchronize()          # exposed transfer stall, if any
                swap_stall_s += time.perf_counter() - t0
            else:
                weights = layer.device_weights

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._compute_stream):
                start.record(self._compute_stream)
                x = layer.forward(x, weights)
                end.record(self._compute_stream)
            t0 = time.perf_counter()
            end.synchronize()
            sync_s += time.perf_counter() - t0
            comp_ms += start.elapsed_time(end)

            if i in streamed:
                del weights

        t0 = time.perf_counter()
        torch.cuda.synchronize(self.device)
        sync_s += time.perf_counter() - t0

        return {
            "t_comp": comp_ms / 1000.0,
            "t_swap": swap_stall_s,
            "t_sync": sync_s,
        }

    # ---- public API -----------------------------------------------------

    def warmup(self, warmup_sec: float = WARMUP_SEC) -> None:
        """
        Drive the device to thermal and allocator steady state (Methods: 60 s).

        Unlike the simulated backend's no-op, this actually runs the workload.
        """
        plan = self.plan_residency(min(1.0, self.hw.c_fast_bytes / self.w_total_bytes))
        self._apply_residency(plan)
        deadline = time.perf_counter() + warmup_sec
        while time.perf_counter() < deadline:
            self._run_step(plan, throttle=None)

    def attach_mem_counters(self, r_c: float, r_b: float, miss_volume_bytes: float) -> None:
        """
        Supply HBM miss volume for an operating point from a separate `ncu` pass.

        Without this, `measure()` records t_mem = NaN at that point. See the
        module docstring for why counters cannot be read inside a live window.
        """
        self._mem_counters[(round(r_c, 4), round(r_b, 4))] = miss_volume_bytes

    def measure(self, r_c: float, r_b: float) -> LatencyRecord:
        """
        Collect one measurement window at the (r_c, r_b) operating point.

        The record reports *achieved* ratios, not requested ones: residency
        quantises to whole layers and the throttle is validated against NVML.

        Raises:
            MeasurementUnavailable: if the operating point is unreachable or the
                clock gate rejects the window.
        """
        plan = self.plan_residency(r_c)
        self._apply_residency(plan)

        throttle: Optional[BandwidthThrottle] = None
        if plan.streamed_bytes > 0:
            t_comp_ref = self._measure_t_comp(plan)
            throttle = BandwidthThrottle(
                self._target_bps_for(r_b, plan, t_comp_ref)
            )

        if not self.nvml.clock_is_stable():
            self._discarded_windows += 1
            raise MeasurementUnavailable(
                f"SM clock deviates {self.nvml.clock_deviation():.1%} from base "
                f"(gate: {5.0:.0f}%); window discarded before it started."
            )

        self.nvml.start_power_sampling()
        agg = {"t_comp": 0.0, "t_swap": 0.0, "t_sync": 0.0}
        steps = 0
        transferred = 0
        wall0 = time.perf_counter()
        deadline = wall0 + self.window_sec
        while time.perf_counter() < deadline:
            step = self._run_step(plan, throttle)
            for k in agg:
                agg[k] += step[k]
            transferred += plan.streamed_bytes
            steps += 1
        t_wall = time.perf_counter() - wall0
        self.nvml.stop_power_sampling()

        if not self.nvml.clock_is_stable():
            self._discarded_windows += 1
            raise MeasurementUnavailable(
                "SM clock drifted beyond the gate during the window; discarded."
            )
        if steps == 0:
            raise MeasurementUnavailable(
                f"no step completed within the {self.window_sec}s window"
            )

        # t_mem only if a counter pass supplied the miss volume for this point.
        key = (round(r_c, 4), round(r_b, 4))
        if key in self._mem_counters and self.calibration is not None:
            t_mem = t_mem_from_counters(
                self._mem_counters[key] * steps, self.calibration.effective_bw_bps
            )
        else:
            t_mem = NAN

        # Achieved R_B = T_comp / T_transfer, both per step and both measured:
        # T_comp from CUDA events, T_transfer from D over the rate the throttle
        # actually sustained.
        if plan.streamed_bytes > 0 and throttle is not None:
            t_comp_step = agg["t_comp"] / steps
            achieved_r_b = compute_rb(
                throttle.target_bps, t_comp_step, plan.streamed_bytes
            )
        else:
            achieved_r_b = NAN

        rec = LatencyRecord(
            timestamp=time.time(),
            r_c=plan.achieved_r_c,
            r_b=achieved_r_b,
            t_comp=agg["t_comp"],
            t_mem=t_mem,
            t_swap=agg["t_swap"],
            t_sync=agg["t_sync"],
            t_wall=t_wall,
            cache_hit_rate=NAN,      # requires an ncu pass; never guessed
            dma_utilisation=(
                self.nvml.measured_h2d_bps() / throttle.target_bps
                if throttle is not None else NAN
            ),
            platform=self.platform,
            model=self.spec.name,
        )
        return rec

    def close(self) -> None:
        self.nvml.shutdown()
        for layer in self._layers:
            layer.evict()
        torch.cuda.empty_cache()

    def __enter__(self) -> "CUDABackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
