"""Write a BF16-storage copy of a full Qwen3 / Qwen3.5-text Decision 2.0 checkpoint (answers meant to stay identical).

The release runtime loads every parameter as FP32 and runs the backbone under
BF16 autocast, so each Linear projection weight is rounded to BF16
(round-to-nearest-even) before every matmul. Only those weights are stored as
BF16, with the same cast; everything the runtime uses in FP32 (embedding table,
RMS norms, gated-delta ``A_log`` / ``dt_bias``, depthwise conv filters, the
decision head) is copied bit for bit. Whether answers really stay identical is
decided by ``release.sh --parity`` on every scored prompt, not by this tool.
Other files are copied verbatim; the shard index gets the new ``total_size``.
The backbone may be sharded (``model.safetensors.index.json``) or one
``model.safetensors``.

Score offsets are bound to a model hash, so ``--score-bias`` also writes the
same offsets rebound to the copy's hash (outside the checkpoint directory).

    python3 -m v2.release.bf16_copy --source SOUP_DIR --output NEW_DIR --receipt receipt.json \
        [--score-bias SCORED_BIAS.json --score-bias-output BF16_BIAS.json]
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import re
import shutil
from pathlib import Path
from typing import Any

from v2.release.layout import sha_file, write_json

BF16_WEIGHT = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_(qkv|z|a|b)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)\.weight"
)
SHARDS = ("model-*.safetensors", "model.safetensors.index.json", "model.safetensors")
REBIND_RULE = (
    "same offsets; this checkpoint stores the scored weights' Linear projection "
    "matrices in BF16 exactly as BF16 autocast rounds them (v2.release.bf16_copy)"
)


def backbone_layout(source: Path) -> tuple[str, list[str], dict[str, Any] | None]:
    """("sharded", shards, index) or ("single", ["model.safetensors"], None)."""
    index_path = source / "backbone" / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return "sharded", sorted(set(index["weight_map"].values())), index
    if (source / "backbone" / "model.safetensors").is_file():
        return "single", ["model.safetensors"], None
    raise FileNotFoundError(f"No backbone weights under {source / 'backbone'}")


def backbone_weights(source: Path) -> Any:
    """copytree ignore: skip weight files in the backbone directory only."""
    backbone = (source / "backbone").resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != backbone:
            return set()
        return {n for n in names if any(fnmatch.fnmatch(n, p) for p in SHARDS)}

    return ignore


def rebind_score_bias(
    report: dict[str, Any],
    model_sha256: str,
    source_file_sha256: str,
    source_model_sha256: str,
) -> dict[str, Any]:
    """The same Score offsets bound to the BF16 copy's model hash."""
    if report.get("model_sha256") != source_model_sha256:
        raise ValueError("Score offsets are not bound to the source checkpoint")
    rebound = copy.deepcopy(report)
    rebound["model_sha256"] = model_sha256
    rebound["fit"]["bf16_rebind"] = {
        "source_file_sha256": source_file_sha256,
        "source_model_sha256": source_model_sha256,
        "rule": REBIND_RULE,
    }
    return rebound


def storage_dtype(name: str, shape: list[int], dtype: str) -> str:
    if dtype != "F32":
        raise ValueError(f"Source tensor is not FP32: {name} ({dtype})")
    return "BF16" if len(shape) == 2 and BF16_WEIGHT.fullmatch(name) else "F32"


def convert(source: Path, output: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists():
        raise FileExistsError(output)
    kind, shards, index = backbone_layout(source)
    pending = output.with_name(output.name + ".pending")
    shutil.copytree(source, pending, ignore=backbone_weights(source))
    numel = {"BF16": 0, "F32": 0}
    tensors_by_dtype = {"BF16": 0, "F32": 0}
    total_size = 0
    for shard in shards:
        with safe_open(str(source / "backbone" / shard), framework="pt") as src:
            metadata = src.metadata()
            names = list(src.keys())
            converted = {}
            for name in names:
                tensor = src.get_tensor(name)
                dtype = "F32" if tensor.dtype == torch.float32 else str(tensor.dtype)
                target = storage_dtype(name, list(tensor.shape), dtype)
                if target == "BF16":
                    tensor = tensor.to(torch.bfloat16)
                converted[name] = tensor.contiguous()
                numel[target] += tensor.numel()
                tensors_by_dtype[target] += 1
                total_size += tensor.numel() * tensor.element_size()
        save_file(converted, str(pending / "backbone" / shard), metadata=metadata)
        with safe_open(
            str(source / "backbone" / shard), framework="pt"
        ) as src, safe_open(str(pending / "backbone" / shard), framework="pt") as out:
            if sorted(out.keys()) != sorted(names):
                raise ValueError(f"Tensor names changed in {shard}")
            for name in names:
                original, stored = src.get_tensor(name), out.get_tensor(name)
                expected = (
                    original.to(torch.bfloat16)
                    if stored.dtype == torch.bfloat16
                    else original
                )
                if stored.dtype != expected.dtype or not torch.equal(stored, expected):
                    raise ValueError(
                        f"Stored tensor differs from its planned cast: {name}"
                    )
    if index is not None:
        index.setdefault("metadata", {})["total_size"] = total_size
        (pending / "backbone" / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    pending.rename(output)
    files = sorted(
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    )
    source_files = sorted(
        p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
    )
    if files != source_files:
        raise ValueError("Output file set differs from the source")
    return {
        "layout": kind,
        "numel_by_storage_dtype": numel,
        "tensors_by_storage_dtype": tensors_by_dtype,
        "tensor_bytes": total_size,
        "files": {
            name: {
                "source_sha256": sha_file(source / name),
                "sha256": sha_file(output / name),
                "bytes": (output / name).stat().st_size,
                "verbatim": not name.startswith("backbone/model"),
            }
            for name in files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--score-bias", type=Path)
    parser.add_argument("--score-bias-output", type=Path)
    args = parser.parse_args()
    if (args.score_bias is None) != (args.score_bias_output is None):
        parser.error("--score-bias and --score-bias-output go together")
    if args.score_bias_output and args.score_bias_output.resolve().is_relative_to(
        args.output.resolve()
    ):
        parser.error("--score-bias-output must be outside the output checkpoint")
    import safetensors
    import torch

    from training.model.infer import checkpoint_fingerprint
    from training.model.score_bias import load_score_bias

    source_model = checkpoint_fingerprint(args.source, None)["model_sha256"]
    if args.score_bias:
        load_score_bias(args.score_bias, source_model)
    result = convert(args.source, args.output)
    for name, entry in result["files"].items():
        if entry["verbatim"] and entry["sha256"] != entry["source_sha256"]:
            raise ValueError(f"Verbatim file changed: {name}")
    model = checkpoint_fingerprint(args.output, None)["model_sha256"]
    if args.score_bias:
        report = json.loads(args.score_bias.read_text(encoding="utf-8"))
        rebound = rebind_score_bias(
            report, model, sha_file(args.score_bias), source_model
        )
        write_json(args.score_bias_output, rebound)
        offsets, _ = load_score_bias(args.score_bias_output, model)
        result["score_bias"] = {
            "source_sha256": sha_file(args.score_bias),
            "sha256": sha_file(args.score_bias_output),
            "offsets": {str(k): v for k, v in sorted(offsets.items())},
        }
    write_json(
        args.receipt,
        {
            "schema": "dev2-release-bf16-copy/1",
            "rule": "BF16 storage for Linear projection weights (rounded exactly as BF16 autocast does); FP32 bit-exact for all other tensors",
            "bf16_pattern": BF16_WEIGHT.pattern,
            "source_model_sha256": source_model,
            "model_sha256": model,
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            **result,
        },
    )


if __name__ == "__main__":
    main()
