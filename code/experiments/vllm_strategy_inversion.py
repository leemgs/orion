#!/usr/bin/env python3
"""Strategy-ranking inversion measurement harness for vLLM (author-run, real GPU).

Implements ``paper/submission/strategy_inversion_protocol.md`` (preregistration
H4, with the H3 predictive corollary) on vLLM. The measurement core, record
schema and analysis are shared with the DeepSpeed-Inference and FlexGen
harnesses in ``strategy_inversion_common.py``, so all three frameworks are
measured identically; see that module for the no-fabrication contract and the
definitions of ``T_total``, ``T_comp``, ``T_transfer = D/B_slow``, ``B_slow``,
``W``, ``C_fast``, ``D`` and ``D_nr``.

Policy -> vLLM engine arguments
-------------------------------
* ``no-offload``       weights and KV resident (``cpu_offload_gb=0``).
* ``paged-KV-offload`` weights resident; KV blocks offloaded to host memory.
* ``layer-pinned``     25% of weight bytes offloaded (``cpu_offload_gb``), rest
                       resident.
* ``full-offload``     60% of weight bytes offloaded plus KV offload.

The GPU budget of an operating point is ``gpu_memory_utilization``.

KV offload and vLLM versions (important)
----------------------------------------
vLLM's V1 engine, the default in current releases, does not swap KV blocks to
CPU: ``swap_space`` only affected the legacy V0 engine. Using it under V1 would
make ``paged-KV-offload`` silently run as a resident configuration. The default
``--kv-offload connector`` therefore uses vLLM's CPU KV-offloading connector via
``kv_transfer_config`` (``OffloadingConnector``). Its extra-config keys have
changed between releases, so they are overridable with ``--kv-connector-extra``;
check the engine log that the offloading connector is active before a campaign.
``--kv-offload swap`` keeps the legacy ``swap_space`` path for V0 only. If the
installed vLLM cannot build the connector, the KV-offload cells are skipped
with a logged reason -- never run as something else under the policy's name.

Usage
-----
    python experiments/vllm_strategy_inversion.py --dry-run -o inv_dryrun.jsonl
    python experiments/vllm_strategy_inversion.py \
        --model meta-llama/Meta-Llama-3-8B --arch A100-80GB \
        --machine-id a100-node-1 --runs 5 -o records_vllm_a100.jsonl
    python experiments/analyze_prereg.py records_vllm_a100.jsonl --split all
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments import strategy_inversion_common as common  # noqa: E402
from experiments.strategy_inversion_common import (  # noqa: E402,F401
    DEFAULT_POINTS,
    DTYPE_BYTES,
    KV_OFFLOAD_GB,
    POLICIES,
    POLICY_SPEC,
    MeasurementUnavailable,
    OperatingPoint,
    PolicyUnsupported,
    kv_bytes_per_token,
    kv_cache_bytes,
    summarize,
)

VLLM_BLOCK_SIZE = 16  # vLLM's default KV block size (tokens per block)
KV_MODES = ("connector", "swap")


def kv_connector_spec(cfg, extra: dict | None = None) -> dict:
    """``kv_transfer_config`` for vLLM's CPU KV-offloading connector.

    ``num_cpu_blocks`` sizes the host KV pool to ``KV_OFFLOAD_GB``. The key name
    follows vLLM's CPU offloading spec at the time of writing; override it with
    ``--kv-connector-extra`` if your version differs.
    """
    blocks = int(KV_OFFLOAD_GB * 1e9 // (VLLM_BLOCK_SIZE * kv_bytes_per_token(cfg)))
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": dict(extra) if extra else
                                     {"num_cpu_blocks": max(1, blocks)},
    }


def policy_engine_kwargs(policy: str, point: OperatingPoint, weights_bytes: float,
                         kv_mode: str = "connector", cfg=None,
                         budget: float | None = None,
                         kv_extra: dict | None = None) -> dict:
    """Map a policy + operating point to concrete vLLM ``LLM(...)`` kwargs.

    Pure function (no vLLM import) so the mapping is unit-testable. A
    connector-mode ``kv_transfer_config`` is returned as a plain dict and turned
    into vLLM's ``KVTransferConfig`` when the engine is built.
    """
    if kv_mode not in KV_MODES:
        raise ValueError(f"kv_mode must be one of {KV_MODES}, got {kv_mode!r}")
    spec = POLICY_SPEC[policy]
    kwargs = dict(
        gpu_memory_utilization=point.gpu_mem_fraction if budget is None else budget,
        max_num_seqs=point.batch,
        max_model_len=point.seq_len + point.out_len,
        cpu_offload_gb=float(spec["weight_offload_frac"] * weights_bytes / 1e9),
        swap_space=0.0,
        enforce_eager=True,  # exclude CUDA-graph capture from the timed region
        dtype="auto",
    )
    if spec["kv_offload"]:
        if kv_mode == "swap":
            kwargs["swap_space"] = float(KV_OFFLOAD_GB)  # legacy V0 engine only
        else:
            if cfg is None:
                raise ValueError("connector KV offload needs the model config")
            kwargs["kv_transfer_config"] = kv_connector_spec(cfg, kv_extra)
    return kwargs


class VLLMAdapter(common.EngineAdapter):
    framework = "vllm"
    default_model = "meta-llama/Meta-Llama-3-8B"
    # vLLM's V1 engine runs the model in a worker process, so this process's
    # allocator does not see the engine's GPU memory.
    in_process = False

    def __init__(self, kv_mode: str = "connector", kv_extra: dict | None = None):
        self.kv_mode = kv_mode
        self.kv_extra = kv_extra

    def add_args(self, parser) -> None:
        parser.add_argument("--kv-offload", choices=KV_MODES, default="connector",
                            help="KV offload mechanism: 'connector' (V1, default) "
                                 "or legacy 'swap' (V0 engine only)")
        parser.add_argument("--kv-connector-extra", default=None,
                            help="JSON for kv_connector_extra_config, if your "
                                 "vLLM version uses different keys")

    def configure(self, args) -> None:
        self.kv_mode = args.kv_offload
        self.kv_extra = json.loads(args.kv_connector_extra) if args.kv_connector_extra else None

    def require(self) -> None:
        try:
            import vllm  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            raise MeasurementUnavailable(
                "vLLM is not installed; install it to run the real campaign, "
                "or use --dry-run to validate the pipeline.") from exc

    def build(self, model, cfg, point, policy, weights_bytes, budget):
        from vllm import LLM
        kwargs = policy_engine_kwargs(policy, point, weights_bytes, self.kv_mode,
                                      cfg=cfg, budget=budget, kv_extra=self.kv_extra)
        engine_kwargs = dict(kwargs)
        if "kv_transfer_config" in engine_kwargs:
            try:
                from vllm.config import KVTransferConfig
            except Exception as exc:
                raise PolicyUnsupported(
                    "this vLLM has no KVTransferConfig; KV offload via connector "
                    "is unavailable") from exc
            engine_kwargs["kv_transfer_config"] = KVTransferConfig(
                **engine_kwargs["kv_transfer_config"])
        llm = LLM(model=model, **engine_kwargs)
        return {"llm": llm, "kwargs": kwargs}

    def prepare(self, handle, point, prompt_ids):
        from vllm import SamplingParams
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
        sampling = SamplingParams(max_tokens=point.out_len, ignore_eos=True,
                                  temperature=0.0)
        return prompts, sampling

    def window(self, handle, point, inputs) -> float:
        prompts, sampling = inputs
        t0 = time.perf_counter()
        handle["llm"].generate(prompts, sampling, use_tqdm=False)
        return time.perf_counter() - t0

    def close(self, handle) -> None:
        if isinstance(handle, dict):
            handle.pop("llm", None)
        common.release_cuda()

    def describe(self, handle) -> str:
        return f"vLLM kwargs={handle['kwargs']}"


# --- module-level API (kept for callers and tests) ------------------------- #
def make_record(point, policy, run_idx, machine_id, arch, model, meas, source):
    return common.make_record(point, policy, run_idx, machine_id, arch, model,
                              meas, source, VLLMAdapter.framework)


def run(model, machine_id, arch, points, runs, output, dry_run,
        weights_bytes_override=None, run_offset=0, kv_mode="connector",
        kv_extra=None) -> int:
    return common.run(VLLMAdapter(kv_mode, kv_extra), model, machine_id, arch,
                      points, runs, output, dry_run,
                      weights_bytes_override=weights_bytes_override,
                      run_offset=run_offset)


def main() -> None:
    common.cli_main(VLLMAdapter(), __doc__)


if __name__ == "__main__":
    main()
