#!/usr/bin/env python3
"""Measurement backends for the preregistered campaign — skeletons to fill.

Each backend turns one preregistered operating point + policy into one real
:class:`PointMeasurement`. The plumbing (grid, split, schema, validation) lives
in ``prereg_harvest.py``; this module is the part that touches a real serving
stack and must be completed on hardware.

**No-fabrication contract.** Every backend below raises ``NotImplementedError``
at the exact spot where a real measurement must be produced, so an unfinished
backend can never emit numbers that ``prereg_harvest`` would tag
``source="measured"``. Fill the marked ``TODO`` regions with real runs; do not
return invented values.

The reusable part that is *not* a stub is :func:`time_compute_transfer_cuda`,
which does correct CUDA-event timing of compute vs. host-to-device transfer and
is what the CUDA backend uses once its model-loading TODO is filled.

Realising a target operating point (all backends):
  * target ``point.r_c`` (residency): cap usable device memory and/or set the
    offload fraction so that ``C_fast/W`` hits the target;
  * target ``point.r_b`` (overlap): scale the per-step compute (sequence length,
    batch, or hidden work) so that ``T_comp/T_transfer`` hits the target;
  * ``policy``: map to the framework's orchestration setting (see each backend).
Record ``T_comp`` and ``T_transfer`` **separately**; never derive one from
``T_total``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class MeasurementUnavailable(RuntimeError):
    """Raised when the hardware/counters needed for a real measurement are absent."""


@dataclass
class PointMeasurement:
    """One real measurement for a (point, policy, run). All times in seconds."""
    t_comp_s: float
    t_transfer_s: float
    t_total_s: float
    c_fast_bytes: float
    w_bytes: float
    d_bytes: float
    d_nr_bytes: float
    b_slow_bytes_per_s: float
    windows: int
    provenance: str = ""
    extra: dict = field(default_factory=dict)


def time_compute_transfer_cuda(compute_fn: Callable[[], None],
                               transfer_fn: Callable[[], None],
                               windows: int, warmup: int = 1) -> tuple[float, float]:
    """Time compute and host-to-device transfer separately with CUDA events.

    ``compute_fn`` and ``transfer_fn`` each perform one step's work when called.
    Returns ``(mean_t_comp_s, mean_t_transfer_s)`` over ``windows`` timed
    iterations after ``warmup`` discarded iterations (excludes one-off
    compilation/allocation — the artifact diagnosed in the submitted runs).

    This is real timing code, safe to reuse; it raises if CUDA is unavailable
    rather than silently timing something else.
    """
    import torch
    if not torch.cuda.is_available():
        raise MeasurementUnavailable("CUDA is not available for event timing")

    def _time(fn: Callable[[], None]) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1e3  # ms -> s

    for _ in range(max(0, warmup)):
        transfer_fn()
        compute_fn()
    torch.cuda.synchronize()

    comp, xfer = [], []
    for _ in range(windows):
        xfer.append(_time(transfer_fn))
        comp.append(_time(compute_fn))
    return sum(comp) / len(comp), sum(xfer) / len(xfer)


class MeasurementBackend:
    """Base class. Subclasses implement :meth:`measure` for one serving stack."""

    #: framework tag written into each record's ``framework`` field
    framework = "unknown"

    def measure(self, point, policy: str, run_idx: int) -> PointMeasurement:
        raise NotImplementedError


class TorchCudaBackend(MeasurementBackend):
    """Generic PyTorch/CUDA backend. Fill model loading + step definition.

    Once the TODO below is filled, reuse :func:`time_compute_transfer_cuda` for
    the separated timing; the rest (assembling PointMeasurement) is done for you.
    """
    framework = "torch"

    def measure(self, point, policy: str, run_idx: int) -> PointMeasurement:
        # TODO(you): load the model, realise point.r_c (device-memory cap /
        # offload fraction) and point.r_b (compute scale), and define:
        #   compute_fn()  -> runs one step's compute
        #   transfer_fn() -> stages this step's non-resident bytes host->device
        # Then:
        #   t_comp, t_xfer = time_compute_transfer_cuda(compute_fn, transfer_fn,
        #                                                windows=10)
        #   t_total = <measured end-to-end step latency for `policy`>
        #   return PointMeasurement(t_comp, t_xfer, t_total,
        #       c_fast_bytes=..., w_bytes=..., d_bytes=..., d_nr_bytes=...,
        #       b_slow_bytes_per_s=..., windows=10, provenance="...")
        raise NotImplementedError(
            "TorchCudaBackend.measure is a skeleton: load a model and define "
            "compute_fn/transfer_fn, then use time_compute_transfer_cuda. See "
            "experiments/cuda_backend.py for an existing CUDA-event example."
        )


class VLLMBackend(MeasurementBackend):
    """vLLM serving backend.

    Policy map (fill to match your deployment):
      * ``no-offload``      -> keep all weights + KV resident (no CPU offload);
      * ``paged-KV-offload``-> enable paged KV-cache CPU offload
                               (``--swap-space`` / ``cpu_offload_gb``);
      * ``layer-pinned``    -> pin hot layers, offload the rest;
      * ``full-offload``    -> maximal weight/KV offload to host.
    Realise ``point.r_c`` via ``gpu_memory_utilization`` / offload size so that
    usable ``C_fast`` over the working set ``W`` hits the target; realise
    ``point.r_b`` via sequence length / batch. Time prefill compute separately
    from host-staging transfer (instrument the engine's copy stream, or run a
    transfer-only control), and read sustained PCIe/NVLink bandwidth for
    ``B_slow`` rather than the datasheet peak.
    """
    framework = "vllm"

    def measure(self, point, policy: str, run_idx: int) -> PointMeasurement:
        try:
            import vllm  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            raise NotImplementedError(
                "vLLM is not installed/wired. Install vLLM, map the policy to an "
                "offload configuration, and return a PointMeasurement from a real "
                "run (see this class's docstring)."
            ) from exc
        # TODO(you): construct LLM(...) per `policy`, realise r_c/r_b, run, and
        # time compute vs. host-staging transfer separately.
        raise NotImplementedError(
            "VLLMBackend.measure: fill the policy->offload map and real timing."
        )


class DeepSpeedBackend(MeasurementBackend):
    """DeepSpeed-Inference backend.

    Policy map via ZeRO-Inference / offload config: ``no-offload`` = all resident;
    ``full-offload`` = weights+KV to CPU (``offload_param``/``offload_kv``);
    ``layer-pinned`` = partial layer offload; ``paged-KV-offload`` = KV to host.
    Realise r_c through the offload ratio, r_b through compute scale, and time
    the prefetch/transfer stream separately from compute.
    """
    framework = "deepspeed"

    def measure(self, point, policy: str, run_idx: int) -> PointMeasurement:
        try:
            import deepspeed  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            raise NotImplementedError(
                "DeepSpeed is not installed/wired. Install it, map the policy to "
                "a ZeRO-Inference offload config, and return a real PointMeasurement."
            ) from exc
        raise NotImplementedError(
            "DeepSpeedBackend.measure: fill the policy->offload map and real timing."
        )


class FlexGenBackend(MeasurementBackend):
    """FlexGen offloading backend.

    FlexGen's policy is a percentage split of weights/KV/activations across
    GPU/CPU/disk. Map: ``no-offload`` = 100% GPU; ``full-offload`` = push to
    CPU/disk; ``layer-pinned`` / ``paged-KV-offload`` = intermediate splits that
    keep hot tensors on GPU. Realise r_c through the GPU percentage and r_b
    through generation length; FlexGen already separates compute and I/O stages,
    so read its per-stage timers for T_comp and T_transfer.
    """
    framework = "flexgen"

    def measure(self, point, policy: str, run_idx: int) -> PointMeasurement:
        try:
            import flexgen  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            raise NotImplementedError(
                "FlexGen is not installed/wired. Install it, map the policy to a "
                "GPU/CPU/disk percentage split, and return a real PointMeasurement."
            ) from exc
        raise NotImplementedError(
            "FlexGenBackend.measure: fill the policy->split map and real timing."
        )


BACKENDS: dict[str, MeasurementBackend] = {
    "torch-cuda": TorchCudaBackend(),
    "vllm": VLLMBackend(),
    "deepspeed": DeepSpeedBackend(),
    "flexgen": FlexGenBackend(),
}


def get_backend(backend_id: str) -> MeasurementBackend:
    try:
        return BACKENDS[backend_id]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend_id!r}; choose one of {sorted(BACKENDS)}"
        )
