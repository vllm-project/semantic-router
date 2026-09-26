"""GPU load/forward preflight for a pinned Qwen3.5-family text backbone.

This is a load and runtime check only. It produces no benchmark result and
does not train or select a checkpoint. Model files must already be local.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("GPU preflight needs a visible accelerator")
    dtype = getattr(torch, args.dtype)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    full, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(
        info.get(field)
        for field in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
    ):
        raise RuntimeError(
            "Incomplete model load: " + json.dumps(info, default=str)[:4000]
        )
    backbone = full.model.language_model
    backbone.config.use_cache = False
    backbone.to(args.device)
    tokens = tokenizer("Decide which option is supported.", return_tensors="pt")[
        "input_ids"
    ].to(args.device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = backbone(input_ids=tokens, use_cache=False)
    torch.cuda.synchronize(args.device)
    print(
        json.dumps(
            {
                "model_revision": args.model_revision,
                "model_path": str(args.model_path),
                "architecture": type(backbone).__name__,
                "text_parameters": sum(
                    parameter.numel() for parameter in backbone.parameters()
                ),
                "parameter_dtype": args.dtype,
                "hidden_shape": list(result.last_hidden_state.shape),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu_arch": getattr(
                    torch.cuda.get_device_properties(args.device), "gcnArchName", None
                ),
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(args.device)
                / 2**30,
                "load_and_forward_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
