#!/usr/bin/env python3
"""Strategy-ranking inversion measurement harness for DeepSpeed-Inference
(ZeRO-Inference; author-run, real GPU).

Same protocol, record schema and measurement core as the vLLM and FlexGen
harnesses (``strategy_inversion_common.py``): ``T_total`` is measured per
(operating point, policy), ``T_comp`` from a resident baseline, ``B_slow`` from
a real host->device bandwidth probe, ``T_transfer = D/B_slow``. Nothing is
invented; a missing library or GPU raises, an unrunnable cell is skipped.

Engine
------
All four policies use the *same* engine -- DeepSpeed ZeRO stage 3 wrapping a
Hugging Face causal LM (the ZeRO-Inference recipe), driven by HF ``generate``
-- and differ only in placement. Using one engine keeps kernel differences
out of the policy comparison. The operating point's GPU budget is imposed with
``torch.cuda.set_per_process_memory_fraction``.

Policy -> ZeRO-Inference configuration
--------------------------------------
* ``no-offload``       ZeRO-3, parameters resident; KV resident.
* ``paged-KV-offload`` ZeRO-3, parameters resident; KV offloaded to host with
                       Hugging Face's ``OffloadedCache``
                       (``cache_implementation="offloaded"``).
* ``layer-pinned``     ``offload_param`` to pinned CPU memory with
                       ``stage3_param_persistence_threshold = hidden^2 + 1``, so
                       every hidden x hidden projection stays resident and only
                       the larger tensors (MLP) are fetched, with prefetch.
* ``full-offload``     ``offload_param`` to pinned CPU memory with the default
                       small persistence threshold, plus ``OffloadedCache``.

Approximation, stated plainly: ZeRO places parameters by size threshold, not by
an exact byte fraction, so ``layer-pinned``/``full-offload`` cannot hit the
neutral spec's 25%/60% offload exactly. Each record's provenance therefore
carries the measured GPU-resident bytes after initialisation, so the realised
placement is visible. KV offload needs a transformers version that provides
``OffloadedCache``; otherwise those cells are skipped with the reason logged.

Run with plain ``python`` (single process): the harness initialises a
one-rank process group itself. If repeated engine construction leaks memory on
your stack, use ``--runs 1 --run-offset k`` per process and merge the files.

Usage
-----
    python experiments/deepspeed_strategy_inversion.py --dry-run -o ds_dryrun.jsonl
    python experiments/deepspeed_strategy_inversion.py \
        --model meta-llama/Meta-Llama-3-8B --arch A100-80GB \
        --machine-id a100-node-1 --runs 5 -o records_ds_a100.jsonl
    python experiments/analyze_prereg.py records_ds_a100.jsonl --split all
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments import strategy_inversion_common as common  # noqa: E402
from experiments.strategy_inversion_common import (  # noqa: E402
    POLICY_SPEC,
    MeasurementUnavailable,
    PolicyUnsupported,
)


def model_dtype_name(cfg) -> str:
    """'bf16' if the checkpoint declares bfloat16, else 'fp16'."""
    dt = str(getattr(cfg, "torch_dtype", "") or "")
    return "bf16" if "bfloat16" in dt else "fp16"


def ds_config_for(policy: str, cfg) -> dict:
    """ZeRO-Inference config for a policy. Pure function (unit-testable).

    ``train_*batch_size`` entries are required by ``deepspeed.initialize`` even
    for inference; they do not affect generation.
    """
    spec = POLICY_SPEC[policy]
    h = int(getattr(cfg, "hidden_size"))
    zero = {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": h * h,
        "stage3_prefetch_bucket_size": int(0.9 * h * h),
        "stage3_param_persistence_threshold": 10 * h,
    }
    if spec["weight_offload_frac"] > 0:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        if policy == "layer-pinned":
            zero["stage3_param_persistence_threshold"] = h * h + 1
    dtype = model_dtype_name(cfg)
    return {
        "fp16": {"enabled": dtype == "fp16"},
        "bf16": {"enabled": dtype == "bf16"},
        "zero_optimization": zero,
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }


def generate_kwargs_for(policy: str, cfg, point) -> dict:
    """HF ``generate`` kwargs: fixed output length; KV offload where required."""
    eos = getattr(cfg, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    pad = getattr(cfg, "pad_token_id", None)
    kw = dict(max_new_tokens=point.out_len, min_new_tokens=point.out_len,
              do_sample=False, pad_token_id=pad if pad is not None else (eos or 0))
    if POLICY_SPEC[policy]["kv_offload"]:
        kw["cache_implementation"] = "offloaded"
    return kw


def _init_single_rank() -> None:
    import torch
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29512",
                 "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}.items():
        os.environ.setdefault(k, v)
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")


def _hf_deepspeed_config(ds_config: dict):
    try:
        from transformers.integrations import HfDeepSpeedConfig
    except Exception:  # older transformers
        from transformers.deepspeed import HfDeepSpeedConfig  # type: ignore
    return HfDeepSpeedConfig(ds_config)


class DeepSpeedAdapter(common.EngineAdapter):
    framework = "deepspeed"
    default_model = "meta-llama/Meta-Llama-3-8B"
    in_process = True

    def require(self) -> None:
        try:
            import deepspeed  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            raise MeasurementUnavailable(
                "DeepSpeed/PyTorch/transformers not installed; install them to "
                "run the real campaign, or use --dry-run.") from exc

    def build(self, model, cfg, point, policy, weights_bytes, budget):
        import torch
        import deepspeed
        import transformers
        from transformers import AutoModelForCausalLM

        gen_kwargs = generate_kwargs_for(policy, cfg, point)
        if "cache_implementation" in gen_kwargs and not hasattr(transformers, "OffloadedCache"):
            raise PolicyUnsupported(
                f"transformers {transformers.__version__} has no OffloadedCache; "
                "KV offload is unavailable for this policy")

        common.limit_gpu_budget(budget)
        _init_single_rank()
        ds_config = ds_config_for(policy, cfg)
        # Must exist (and stay referenced) before from_pretrained so the weights
        # are loaded directly into ZeRO-3 partitions instead of onto the GPU.
        dschf = _hf_deepspeed_config(ds_config)
        torch_dtype = torch.bfloat16 if model_dtype_name(cfg) == "bf16" else torch.float16
        hf_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
        engine = deepspeed.initialize(model=hf_model, config=ds_config)[0]
        engine.module.eval()
        return {"engine": engine, "dschf": dschf, "gen_kwargs": gen_kwargs,
                "zero": ds_config["zero_optimization"]}

    def prepare(self, handle, point, prompt_ids):
        import torch
        ids = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        return ids, torch.ones_like(ids)

    def window(self, handle, point, inputs) -> float:
        import torch
        ids, mask = inputs
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            handle["engine"].module.generate(input_ids=ids, attention_mask=mask,
                                             **handle["gen_kwargs"])
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def close(self, handle) -> None:
        if isinstance(handle, dict):
            for key in ("engine", "dschf"):
                handle.pop(key, None)
        common.release_cuda()

    def describe(self, handle) -> str:
        cache = handle["gen_kwargs"].get("cache_implementation", "default")
        return f"DeepSpeed ZeRO-3 zero_optimization={handle['zero']}; kv_cache={cache}"


def main() -> None:
    common.cli_main(DeepSpeedAdapter(), __doc__)


if __name__ == "__main__":
    main()
