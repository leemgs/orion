"""
HBM traffic counters and rho calibration (Methods §Measurement protocol, T_mem).

The paper defines the locality term as

    T_mem = miss_volume / effective_HBM_bandwidth

and the lower-bound constant rho as the HBM miss penalty per byte. This module
measures both against real silicon:

  - `calibrate_rho()` runs a streaming microbenchmark whose working set is far
    larger than L2, so every read misses to HBM, and derives
    rho = 1 / effective_HBM_bandwidth from the achieved bandwidth.
  - `NsightDramCounter` reads DRAM/L2 sector counters via Nsight Compute (`ncu`),
    which is the supported CUPTI-based path for these metrics.

Neither path has a modelled fallback. If a counter cannot be read the call
raises MeasurementUnavailable, and the caller must record t_mem as NaN rather
than substitute an estimate.

Note on `ncu`: Nsight Compute serialises and replays kernels, so it cannot run
inside a live latency window without perturbing the very timings under study.
T_mem is therefore collected in a *separate profiling pass* over an identical
workload configuration, as `collect_dram_bytes()` documents.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

from orion.nvml_monitor import MeasurementUnavailable

try:
    import torch
    _TORCH_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:              # pragma: no cover - depends on host
    torch = None                      # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


# L2 cache line / sector size in bytes on NVIDIA parts; ncu reports L2 traffic
# in sectors, not bytes.
L2_SECTOR_BYTES = 32


def _require_torch() -> None:
    if torch is None:
        raise MeasurementUnavailable(
            f"torch is required for live measurement but could not be imported "
            f"({_TORCH_IMPORT_ERROR!r})."
        )
    if not torch.cuda.is_available():
        raise MeasurementUnavailable(
            "torch reports no CUDA device; live measurement is not possible."
        )


@dataclass
class HBMCalibration:
    """Result of the streaming microbenchmark used to fix rho."""
    effective_bw_bps: float     # achieved HBM read bandwidth, bytes/s
    rho_s_per_byte: float       # 1 / effective_bw_bps
    buffer_bytes: int
    n_iters: int
    device_name: str

    @property
    def effective_bw_gbs(self) -> float:
        return self.effective_bw_bps / 1e9

    @property
    def rho_ps_per_byte(self) -> float:
        return self.rho_s_per_byte * 1e12


def calibrate_rho(
    device_index: int = 0,
    buffer_bytes: int = 2 << 30,      # 2 GiB, far beyond any current L2
    n_iters: int = 50,
    n_warmup: int = 10,
) -> HBMCalibration:
    """
    Measure the HBM miss penalty per byte via a streaming-read microbenchmark.

    A buffer of `buffer_bytes` (>> L2) is read end-to-end each iteration, so the
    achieved rate is the effective HBM read bandwidth under cache-missing loads
    (Methods: "a synthetic streaming kernel that issues controlled cache-miss
    loads"). rho is its reciprocal.

    Returns:
        HBMCalibration with the measured bandwidth and rho.

    Raises:
        MeasurementUnavailable: if no CUDA device is present.
    """
    _require_torch()
    dev = torch.device(f"cuda:{device_index}")

    n_elems = buffer_bytes // 4
    try:
        buf = torch.empty(n_elems, dtype=torch.float32, device=dev)
    except RuntimeError as exc:
        raise MeasurementUnavailable(
            f"could not allocate a {buffer_bytes / 2**30:.1f} GiB calibration "
            f"buffer on cuda:{device_index}: {exc}"
        ) from exc
    buf.uniform_()

    # A full reduction touches every byte exactly once and is bandwidth-bound.
    for _ in range(n_warmup):
        buf.sum()
    torch.cuda.synchronize(dev)

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(n_iters):
        buf.sum()
    end.record()
    torch.cuda.synchronize(dev)

    elapsed_s = start.elapsed_time(end) / 1000.0
    total_bytes = buf.numel() * buf.element_size() * n_iters
    bw = total_bytes / elapsed_s

    cal = HBMCalibration(
        effective_bw_bps=bw,
        rho_s_per_byte=1.0 / bw,
        buffer_bytes=buf.numel() * buf.element_size(),
        n_iters=n_iters,
        device_name=torch.cuda.get_device_name(device_index),
    )
    del buf
    torch.cuda.empty_cache()
    return cal


class NsightDramCounter:
    """
    DRAM/L2 traffic counters via Nsight Compute (`ncu`), the CUPTI-based path.

    `ncu` replays each kernel to collect counters, so this must be run as a
    separate profiling pass — never inside a timed measurement window.
    """

    #: DRAM read bytes and L2 read misses. dram__bytes_read gives the HBM
    #: traffic directly; the L2 miss sectors are kept as a cross-check.
    METRICS = (
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors_op_read_lookup_miss.sum",
    )

    def __init__(self, ncu_path: Optional[str] = None) -> None:
        self.ncu_path = ncu_path or shutil.which("ncu")
        if self.ncu_path is None:
            raise MeasurementUnavailable(
                "Nsight Compute (`ncu`) not found on PATH. It is required to "
                "measure T_mem from HBM counters. Install the CUDA Toolkit's "
                "Nsight Compute, or pass --mem-counter none to record t_mem as "
                "NaN (windows will then fail the completeness check)."
            )

    @staticmethod
    def available() -> bool:
        return shutil.which("ncu") is not None

    def collect_dram_bytes(self, command: Sequence[str], timeout_s: float = 3600.0) -> dict:
        """
        Profile `command` and return aggregate DRAM traffic.

        Args:
            command: the workload to profile, as an argv sequence.
            timeout_s: hard limit on the profiling pass.

        Returns:
            dict with dram_read_bytes, dram_write_bytes, l2_read_miss_sectors,
            and l2_read_miss_bytes.

        Raises:
            MeasurementUnavailable: if ncu fails or emits no counters.
        """
        argv = [
            self.ncu_path,
            "--csv",
            "--target-processes", "all",
            "--metrics", ",".join(self.METRICS),
            *command,
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeasurementUnavailable(
                f"ncu profiling pass exceeded {timeout_s}s"
            ) from exc

        if proc.returncode != 0:
            raise MeasurementUnavailable(
                f"ncu exited {proc.returncode}. Counter collection usually "
                f"requires elevated GPU performance-counter permissions "
                f"(see NVIDIA ERR_NVGPUCTRPERM). stderr:\n{proc.stderr[:2000]}"
            )

        totals = self._parse_csv(proc.stdout)
        if not totals:
            raise MeasurementUnavailable(
                "ncu produced no counter rows; the workload may not have "
                "launched any kernels."
            )
        miss_sectors = totals.get("lts__t_sectors_op_read_lookup_miss.sum", 0.0)
        return {
            "dram_read_bytes":      totals.get("dram__bytes_read.sum", 0.0),
            "dram_write_bytes":     totals.get("dram__bytes_write.sum", 0.0),
            "l2_read_miss_sectors": miss_sectors,
            "l2_read_miss_bytes":   miss_sectors * L2_SECTOR_BYTES,
        }

    @staticmethod
    def _parse_csv(stdout: str) -> dict:
        """Sum each metric over all profiled kernels in ncu's CSV output."""
        # ncu prints banner lines before the CSV header; skip to the header row.
        lines = stdout.splitlines()
        start = next(
            (i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None
        )
        if start is None:
            return {}

        totals: dict[str, float] = {}
        reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
        for row in reader:
            name = (row.get("Metric Name") or "").strip()
            raw = (row.get("Metric Value") or "").strip().replace(",", "")
            if not name or not raw:
                continue
            try:
                totals[name] = totals.get(name, 0.0) + float(raw)
            except ValueError:
                # ncu emits a units row and occasional "n/a" values; skip them.
                continue
        return totals


def t_mem_from_counters(miss_volume_bytes: float, effective_bw_bps: float) -> float:
    """
    T_mem = miss_volume / effective_HBM_bandwidth (Methods §T_mem).

    Raises:
        MeasurementUnavailable: on a non-positive bandwidth, which would
            otherwise silently produce an infinite or negative T_mem.
    """
    if effective_bw_bps <= 0:
        raise MeasurementUnavailable(
            f"effective HBM bandwidth must be positive, got {effective_bw_bps}"
        )
    return miss_volume_bytes / effective_bw_bps
