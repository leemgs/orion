"""
NVML-based device monitoring for live hardware runs (Methods §Measurement protocol).

Provides:
  - NVMLMonitor: GPU clock / power / PCIe throughput sampling
  - clock-deviation gate (discard windows where SM clock drifts > 5% from base)
  - PCIe throughput readback used to validate the software bandwidth throttle

Every reading here comes from NVML. If NVML or the device is unavailable this
module raises MeasurementUnavailable rather than substituting a modelled value:
an unmeasured quantity must never enter a LatencyRecord.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from orion.config import CLOCK_DEVIATION_GATE


class MeasurementUnavailable(RuntimeError):
    """Raised when a required hardware counter cannot be read.

    Callers must propagate this. Falling back to a modelled or default value
    would silently turn a measurement into a simulation.
    """


try:
    import pynvml
    _NVML_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:              # pragma: no cover - depends on host
    pynvml = None                     # type: ignore[assignment]
    _NVML_IMPORT_ERROR = exc


@dataclass
class PowerSample:
    """One NVML power reading."""
    timestamp: float      # perf_counter seconds
    watts: float


@dataclass
class DeviceState:
    """Instantaneous device state sampled from NVML."""
    sm_clock_mhz: int
    mem_clock_mhz: int
    power_w: float
    pcie_tx_kbs: int      # nvmlDeviceGetPcieThroughput, KB/s
    pcie_rx_kbs: int
    hbm_used_bytes: int
    hbm_total_bytes: int


class NVMLMonitor:
    """
    Thin NVML wrapper scoped to what the measurement protocol requires.

    Args:
        device_index: CUDA device ordinal to monitor.
        power_sample_ms: NVML power sampling interval (Methods: 100 ms).
    """

    def __init__(self, device_index: int = 0, power_sample_ms: float = 100.0) -> None:
        if pynvml is None:
            raise MeasurementUnavailable(
                f"pynvml is required for live measurement but could not be "
                f"imported ({_NVML_IMPORT_ERROR!r}). Install with "
                f"`pip install -e '.[gpu]'`."
            )
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:
            raise MeasurementUnavailable(f"NVML init failed: {exc!r}") from exc

        self.device_index = device_index
        self.power_sample_ms = power_sample_ms
        self._base_sm_clock = self._read_base_sm_clock()
        self._samples: list[PowerSample] = []
        self._sampling = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- static device facts -------------------------------------------

    def _read_base_sm_clock(self) -> int:
        """Base (nominal) SM clock in MHz, used as the clock-gate reference."""
        try:
            return pynvml.nvmlDeviceGetDefaultApplicationsClock(
                self._handle, pynvml.NVML_CLOCK_SM
            )
        except Exception:
            # Not all parts expose default application clocks; fall back to the
            # maximum supported SM clock, which is a valid nominal reference.
            try:
                return pynvml.nvmlDeviceGetMaxClockInfo(
                    self._handle, pynvml.NVML_CLOCK_SM
                )
            except Exception as exc:
                raise MeasurementUnavailable(
                    f"cannot read a base SM clock for the clock gate: {exc!r}"
                ) from exc

    @property
    def base_sm_clock_mhz(self) -> int:
        return self._base_sm_clock

    def device_name(self) -> str:
        name = pynvml.nvmlDeviceGetName(self._handle)
        return name.decode() if isinstance(name, bytes) else name

    def hbm_total_bytes(self) -> int:
        return int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).total)

    # ---- instantaneous state -------------------------------------------

    def read_state(self) -> DeviceState:
        """Sample all NVML counters the protocol depends on."""
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return DeviceState(
                sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(
                    self._handle, pynvml.NVML_CLOCK_SM),
                mem_clock_mhz=pynvml.nvmlDeviceGetClockInfo(
                    self._handle, pynvml.NVML_CLOCK_MEM),
                power_w=pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
                pcie_tx_kbs=pynvml.nvmlDeviceGetPcieThroughput(
                    self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES),
                pcie_rx_kbs=pynvml.nvmlDeviceGetPcieThroughput(
                    self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES),
                hbm_used_bytes=int(mem.used),
                hbm_total_bytes=int(mem.total),
            )
        except Exception as exc:
            raise MeasurementUnavailable(f"NVML read failed: {exc!r}") from exc

    def clock_deviation(self) -> float:
        """Fractional deviation of the current SM clock from base."""
        cur = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
        return abs(cur - self._base_sm_clock) / self._base_sm_clock

    def clock_is_stable(self, gate: float = CLOCK_DEVIATION_GATE) -> bool:
        """
        True if the SM clock is within `gate` of base (Methods: 5%).

        Windows failing this gate are discarded — thermal or power throttling
        confounds the latency decomposition.
        """
        return self.clock_deviation() <= gate

    def measured_h2d_bps(self) -> float:
        """
        Host-to-device PCIe throughput in bytes/s (NVML RX from the GPU's view).

        Used to validate that the software token-bucket actually achieves its
        target bandwidth (Methods: deviation from target must be <= 3%).
        """
        return self.read_state().pcie_rx_kbs * 1000.0

    # ---- power integration ---------------------------------------------

    def start_power_sampling(self) -> None:
        """Begin background NVML power sampling at `power_sample_ms`."""
        if self._thread is not None:
            return
        self._samples = []
        self._sampling.set()

        def _loop() -> None:
            while self._sampling.is_set():
                try:
                    w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    self._samples.append(PowerSample(time.perf_counter(), w))
                except Exception:
                    # A transient NVML read failure must not kill the run; the
                    # gap shows up as a reduced sample count, which the caller
                    # can inspect via `power_samples`.
                    pass
                time.sleep(self.power_sample_ms / 1000.0)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_power_sampling(self) -> list[PowerSample]:
        """Stop sampling and return the collected samples."""
        self._sampling.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return list(self._samples)

    @property
    def power_samples(self) -> list[PowerSample]:
        return list(self._samples)

    def mean_power_w(self) -> float:
        """Mean power over the collected samples."""
        if not self._samples:
            raise MeasurementUnavailable(
                "no power samples collected; call start_power_sampling() first"
            )
        return sum(s.watts for s in self._samples) / len(self._samples)

    def energy_j(self) -> float:
        """
        Energy in joules, trapezoidally integrated over the power samples
        (Methods: NVML power sampled at 100 ms, integrated per request).
        """
        if len(self._samples) < 2:
            raise MeasurementUnavailable(
                "at least two power samples are required to integrate energy"
            )
        total = 0.0
        for a, b in zip(self._samples, self._samples[1:]):
            total += 0.5 * (a.watts + b.watts) * (b.timestamp - a.timestamp)
        return total

    def shutdown(self) -> None:
        self.stop_power_sampling()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def __enter__(self) -> "NVMLMonitor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
