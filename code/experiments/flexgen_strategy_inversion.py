#!/usr/bin/env python3
"""Strategy-ranking inversion measurement harness for FlexGen / FlexLLMGen
(author-run, real GPU).

Same protocol, record schema and measurement core as the vLLM and DeepSpeed
harnesses (``strategy_inversion_common.py``): ``T_total`` is measured per
(operating point, policy), ``T_comp`` from a resident baseline, ``B_slow`` from
a real host->device bandwidth probe, ``T_transfer = D/B_slow``. Nothing is
invented; a missing library or GPU raises, an unrunnable cell is skipped.

Scope of FlexGen
----------------
FlexGen (FMInference, packaged as ``flexllmgen``; older ``flexgen``) runs the
OPT model family only, and OPT's context is 2048 tokens. The default model is
therefore ``facebook/opt-6.7b`` and this harness uses its own operating points
that fit the context (``FLEXGEN_POINTS``). Weights are read from FlexGen's
NumPy weight directory (``--weights-dir``), which FlexGen downloads and
converts on first use. ``W`` and ``C_fast`` still come from the Hugging Face
config and checkpoint, exactly as for the other frameworks.

Policy -> FlexGen ``Policy`` (placement percentages GPU/CPU)
------------------------------------------------------------
FlexGen expresses placement natively as percentages of weights, KV cache and
activations, so the neutral spec maps almost exactly:

* ``no-offload``       weights 100/0, cache 100/0, activations 100/0, no overlap.
* ``paged-KV-offload`` weights 100/0, cache 0/100 (KV in host memory), overlap.
* ``layer-pinned``     weights 75/25, cache 100/0, overlap, pinned host weights.
* ``full-offload``     weights 40/60, cache 0/100, overlap, pinned host weights.

FlexGen splits each layer's tensors by cumulative size, so a weight percentage
is a per-layer byte share rather than whole layers pinned; the measured
GPU-resident bytes after initialisation are recorded in every provenance. The
operating point's GPU budget is imposed with
``torch.cuda.set_per_process_memory_fraction``.

API note
--------
Written against FlexLLMGen's ``flex_opt`` API (``Policy``, ``OptLM``,
``ExecutionEnv``, ``CompressionConfig``, ``get_opt_config``). ``require()``
constructs a test ``Policy`` with keyword arguments, so a version whose
signature differs fails up front with a clear message instead of mid-campaign.

Usage
-----
    python experiments/flexgen_strategy_inversion.py --dry-run -o fg_dryrun.jsonl
    python experiments/flexgen_strategy_inversion.py \
        --model facebook/opt-6.7b --arch A100-80GB \
        --machine-id a100-node-1 --runs 5 -o records_flexgen_a100.jsonl
    python experiments/analyze_prereg.py records_flexgen_a100.jsonl --split all
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments import strategy_inversion_common as common  # noqa: E402
from experiments.strategy_inversion_common import (  # noqa: E402
    POLICY_SPEC,
    MeasurementUnavailable,
    OperatingPoint,
)

# Operating points that fit OPT's 2048-token context. As with the defaults in
# the common module, confirm the measured (R_C, R_B) land one point per regime
# and adjust before a campaign.
FLEXGEN_POINTS = [
    OperatingPoint("cap",   gpu_mem_fraction=0.20, batch=16, seq_len=512,
                   out_len=128, target_regime="capacity-limited"),
    OperatingPoint("io",    gpu_mem_fraction=0.60, batch=8,  seq_len=1024,
                   out_len=64,  target_regime="I/O-limited"),
    OperatingPoint("coord", gpu_mem_fraction=0.90, batch=4,  seq_len=1536,
                   out_len=256, target_regime="coordination-dominated"),
]


def flexgen_policy_kwargs(policy: str, point: OperatingPoint) -> dict:
    """FlexGen ``Policy`` keyword arguments for a policy. Pure (unit-testable).

    The two ``CompressionConfig`` fields are added at build time; compression is
    disabled in every policy so placement is the only thing that varies.
    """
    spec = POLICY_SPEC[policy]
    w_gpu = round(100 * (1.0 - spec["weight_offload_frac"]))
    cache_gpu = 0 if spec["kv_offload"] else 100
    offloads = spec["weight_offload_frac"] > 0 or spec["kv_offload"]
    return dict(
        gpu_batch_size=point.batch, num_gpu_batches=1,
        w_gpu_percent=w_gpu, w_cpu_percent=100 - w_gpu,
        cache_gpu_percent=cache_gpu, cache_cpu_percent=100 - cache_gpu,
        act_gpu_percent=100, act_cpu_percent=0,
        overlap=offloads, sep_layer=True, pin_weight=True,
        cpu_cache_compute=False, attn_sparsity=1.0,
        compress_weight=False, compress_cache=False,
    )


def _flexgen_api() -> SimpleNamespace:
    """Locate the FlexGen API under either package name."""
    errors = []
    for pkg in ("flexllmgen", "flexgen"):
        try:
            flex_opt = importlib.import_module(f"{pkg}.flex_opt")
            opt_config = importlib.import_module(f"{pkg}.opt_config")
            compression = importlib.import_module(f"{pkg}.compression")
            env_cls = getattr(flex_opt, "ExecutionEnv", None)
            if env_cls is None:
                env_cls = importlib.import_module(f"{pkg}.utils").ExecutionEnv
            return SimpleNamespace(
                pkg=pkg, Policy=flex_opt.Policy, OptLM=flex_opt.OptLM,
                ExecutionEnv=env_cls,
                CompressionConfig=compression.CompressionConfig,
                get_opt_config=opt_config.get_opt_config)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{pkg}: {exc}")
    raise MeasurementUnavailable(
        "FlexGen is not installed or its API differs (tried flexllmgen, "
        f"flexgen): {'; '.join(errors)}. Install FlexLLMGen, or use --dry-run.")


def _make_policy(api: SimpleNamespace, kwargs: dict):
    comp_w = api.CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    comp_c = api.CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False)
    return api.Policy(**kwargs, comp_weight_config=comp_w, comp_cache_config=comp_c)


class FlexGenAdapter(common.EngineAdapter):
    framework = "flexgen"
    default_model = "facebook/opt-6.7b"
    default_points = FLEXGEN_POINTS
    in_process = True

    def __init__(self, weights_dir: str = "~/opt_weights",
                 offload_dir: str = "~/flexgen_offload_dir"):
        self.weights_dir = weights_dir
        self.offload_dir = offload_dir
        self._api: SimpleNamespace | None = None

    def add_args(self, parser) -> None:
        parser.add_argument("--weights-dir", default="~/opt_weights",
                            help="FlexGen NumPy weight directory (downloaded/converted "
                                 "on first use)")
        parser.add_argument("--offload-dir", default="~/flexgen_offload_dir",
                            help="FlexGen disk offload directory (unused by these "
                                 "GPU/CPU-only policies, but required by the env)")

    def configure(self, args) -> None:
        self.weights_dir = args.weights_dir
        self.offload_dir = args.offload_dir

    def require(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            raise MeasurementUnavailable("PyTorch is not installed") from exc
        api = _flexgen_api()
        probe = flexgen_policy_kwargs("no-offload", FLEXGEN_POINTS[0])
        try:
            _make_policy(api, probe)
        except TypeError as exc:
            raise MeasurementUnavailable(
                f"{api.pkg}.Policy signature differs from the one this harness "
                f"targets ({exc}); update flexgen_policy_kwargs.") from exc
        self._api = api

    def build(self, model, cfg, point, policy, weights_bytes, budget):
        api = self._api or _flexgen_api()
        common.limit_gpu_budget(budget)
        kwargs = flexgen_policy_kwargs(policy, point)
        env = api.ExecutionEnv.create(str(Path(self.offload_dir).expanduser()))
        try:
            lm = api.OptLM(api.get_opt_config(model), env,
                           str(Path(self.weights_dir).expanduser()),
                           _make_policy(api, kwargs))
        except Exception:
            env.close_copy_threads()
            raise
        return {"lm": lm, "env": env, "policy_kwargs": kwargs}

    def window(self, handle, point, inputs) -> float:
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        handle["lm"].generate(inputs, max_new_tokens=point.out_len, do_sample=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def close(self, handle) -> None:
        if isinstance(handle, dict):
            env = handle.pop("env", None)
            handle.pop("lm", None)
            if env is not None:
                try:
                    env.close_copy_threads()
                except Exception:
                    pass
        common.release_cuda()

    def describe(self, handle) -> str:
        k = handle["policy_kwargs"]
        return (f"FlexGen Policy weights GPU/CPU={k['w_gpu_percent']}/"
                f"{k['w_cpu_percent']}, cache GPU/CPU={k['cache_gpu_percent']}/"
                f"{k['cache_cpu_percent']}, overlap={k['overlap']}")


def main() -> None:
    common.cli_main(FlexGenAdapter(), __doc__)


if __name__ == "__main__":
    main()
