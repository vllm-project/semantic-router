"""Same-panel collector for LoRA checkpoints on official MoE bases (eval runner adapter).

Runs the current ``training.model.infer`` as the 27B kernel adapter does (base via
``--source-path``, no truncation, one prompt per call), with the runtime check chosen by
the checkpoint's architecture:

- Qwen3.5-MoE (gated-delta layers): ``typed_collect_kernel.kernel_runtime`` first (image FLA
  site, dense bindings, an existing ``TRITON_CACHE_DIR``), then every gated-delta and
  causal-conv1d binding of ``modeling_qwen3_5_moe`` must be the image kernel, before and
  after the collection;
- Gemma 4 (no gated-delta layers): SDPA only, FLA is not required.

The experts implementation comes from the checkpoint's pinned metadata. A runtime sidecar
lands next to the predictions. ``--max-items N`` is the runner's smoke (first N prompts).
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from training.model import infer

SCHEMA = "decision2-27b-moe-collect-runtime/1"


def moe_bindings() -> dict[str, str | None]:
    runtime_check = importlib.import_module("v2.dec.runtime_check")
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

    return runtime_check.kernel_bindings(modeling_qwen3_5_moe)


def architecture(argv: list[str]) -> str:
    checkpoint = Path(argv[argv.index("--checkpoint") + 1])
    metadata = json.loads(
        (checkpoint / "decision_config.json").read_text(encoding="utf-8")
    )
    return str(metadata.get("architecture"))


def main(argv: list[str] | None = None) -> None:
    kernel = importlib.import_module("v2.27b.typed_collect_kernel")
    raw = list(sys.argv[1:] if argv is None else argv)
    max_items = int(raw[raw.index("--max-items") + 1]) if "--max-items" in raw else None
    arch = architecture(raw)
    gated_delta = arch.startswith("qwen3.5-moe")
    runtime: dict[str, Any] = {"architecture": arch}
    if gated_delta:
        runtime.update(kernel.kernel_runtime())
        before = moe_bindings()
        if not before or not all(before.values()):
            raise SystemExit(
                f"Qwen3.5-MoE kernel bindings are not the image kernels: {before}"
            )
        runtime["moe_kernel_bindings"] = before
    elif not arch.startswith("gemma4-moe"):
        raise SystemExit(f"not an MoE checkpoint: {arch}")
    args = kernel.smoke_argv(raw)
    started = time.perf_counter()
    sys.argv = [infer.__file__, *args]
    infer.main()
    wall = time.perf_counter() - started
    output = Path(args[args.index("--output") + 1])
    record = {
        "schema": SCHEMA,
        **{k: v for k, v in runtime.items() if k != "kernel_bindings"},
        "gated_delta": (
            "kernel (image FLA / causal-conv1d)"
            if gated_delta
            else "none (Gemma 4: SDPA only)"
        ),
        "fla_loaded": any(
            name == "fla" or name.startswith("fla.") for name in sys.modules
        ),
        "versions": kernel.versions(),
        "max_length": int(args[args.index("--max-length") + 1]),
        "max_items": max_items,
        "infer_wall_seconds": wall,
        "memory": kernel.memory(),
    }
    if gated_delta:
        after = moe_bindings()
        record["moe_kernel_bindings_after"] = after
        if not record["fla_loaded"] or after != runtime["moe_kernel_bindings"]:
            raise SystemExit("the kernel path was not used for the whole collection")
    output.with_name(output.name + ".runtime.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
