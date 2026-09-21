"""Validate the measurement-backend registry and its no-fabrication contract.

These run without any serving framework or GPU: every backend must refuse
cleanly (NotImplementedError) rather than return invented numbers, and the
registry must resolve the expected ids.
"""
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments import prereg_backends as pb  # noqa: E402
from experiments import prereg_harvest as ph  # noqa: E402


def _point():
    machines = [{"machine_id": "a100-node-1", "arch": "A100-80GB"}]
    models = [{"model": "llama-3-8b", "framework": "vllm"}]
    return ph.build_grid(machines, models, rc_grid=[0.6], rb_grid=[1.5])[0]


def test_registry_has_expected_backends():
    assert set(pb.BACKENDS) == {
        "torch-reference", "torch-cuda", "vllm", "deepspeed", "flexgen"}


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError):
        pb.get_backend("does-not-exist")


@pytest.mark.parametrize("backend_id", ["torch-cuda", "vllm", "deepspeed", "flexgen"])
def test_skeleton_backends_refuse_without_wiring(backend_id):
    with pytest.raises(NotImplementedError):
        pb.get_backend(backend_id).measure(_point(), "no-offload", 0)


def test_reference_backend_needs_a_gpu():
    # The reference backend is real, not a skeleton: without torch or a CUDA
    # GPU it must refuse cleanly (MeasurementUnavailable) rather than invent
    # numbers. It never raises NotImplementedError, since it is complete.
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        pytest.skip("CUDA present; refusal path not exercised here")
    with pytest.raises(pb.MeasurementUnavailable):
        pb.get_backend("torch-reference").measure(_point(), "paged-KV-offload", 0)


def test_harvest_seam_dispatches_to_backend():
    # The plumbing seam must route to the registry and inherit the refusal.
    with pytest.raises(NotImplementedError):
        ph.measure_operating_point(_point(), "no-offload", 0, "vllm")


def test_cuda_timing_helper_refuses_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present; refusal path not exercised")
    with pytest.raises(pb.MeasurementUnavailable):
        pb.time_compute_transfer_cuda(lambda: None, lambda: None, windows=2)
