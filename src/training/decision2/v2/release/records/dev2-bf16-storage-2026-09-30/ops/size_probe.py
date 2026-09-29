"""Read-only: exact storage bytes of a v2.release.bf16_copy of a checkpoint backbone, from safetensors headers."""

import json
import re
import struct
import sys
from pathlib import Path

BF16_WEIGHT = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_(qkv|z|a|b)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)\.weight"
)
SIZE = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "U8": 1}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def main():
    ckpt = Path(sys.argv[1])
    out = {
        "checkpoint": str(ckpt),
        "shards": 0,
        "source_bytes": 0,
        "bf16copy_tensor_bytes": 0,
        "allbf16_tensor_bytes": 0,
        "bf16copy_file_bytes": 0,
        "fp32_kept": {},
        "dtypes": {},
        "numel": 0,
        "other_files": {},
    }
    kept = {}
    for shard in sorted((ckpt / "backbone").glob("model*.safetensors")):
        n, hdr = header(shard)
        out["shards"] += 1
        out["source_bytes"] += shard.stat().st_size
        shard_copy = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            numel = 1
            for d in meta["shape"]:
                numel *= d
            out["numel"] += numel
            out["dtypes"][meta["dtype"]] = out["dtypes"].get(meta["dtype"], 0) + numel
            bf16 = (
                meta["dtype"] == "F32"
                and len(meta["shape"]) == 2
                and BF16_WEIGHT.fullmatch(name)
            )
            b = numel * (2 if bf16 else SIZE[meta["dtype"]])
            out["bf16copy_tensor_bytes"] += b
            out["allbf16_tensor_bytes"] += numel * 2
            shard_copy += b
            if not bf16:
                key = re.sub(r"\.\d+\.", ".N.", name)
                kept.setdefault(key, [0, 0])
                kept[key][0] += 1
                kept[key][1] += numel
        out["bf16copy_file_bytes"] += shard_copy
    out["fp32_kept"] = {
        k: {"tensors": v[0], "numel": v[1], "bytes": v[1] * 4}
        for k, v in sorted(kept.items(), key=lambda kv: -kv[1][1])
    }
    for p in sorted(ckpt.rglob("*")):
        if p.is_file() and not (
            p.parent.name == "backbone" and p.name.startswith("model")
        ):
            out["other_files"][p.relative_to(ckpt).as_posix()] = p.stat().st_size
    print(json.dumps(out, indent=1))


main()
