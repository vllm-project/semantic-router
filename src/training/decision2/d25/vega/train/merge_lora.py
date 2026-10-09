"""Merge the Decision 2.0 Vega LoRA into Qwen3.8-27B in FP32 -> a base-equivalent checkpoint.

vllm-sr/Decision-2.0-Vega-27B is a PEFT LoRA (r 512, alpha 1024, all 496 text projections of a
``Qwen3_5TextModel``) plus a candidate head on frozen Qwen/Qwen3.8-27B @ 1d4bf0f2. The head is
ignored. Every targeted weight becomes ``W + (alpha / r) * B @ A`` computed in FP32 from the BF16
base, then cast to the output dtype (BF16 default; ``--dtype float32`` keeps the exact merge for
FP32 master weights). Everything else (embeddings, norms, lm_head, vision, MTP) is copied
unchanged, so the output has the base's layout and loads anywhere the base loads; the readout of
a warm-start arm is initialised from its (unchanged) lm_head rows.

The manifest records, per tensor, how much of the update survives the cast:
``kept = 1 - |cast(W + dW) - (W + dW)| / |dW|`` (Frobenius), the reason the 2.0 lesson says never to
merge a LoRA into BF16 weights for inference.

    python -m d25.vega.train.merge_lora --base /models/base --out /data/d25/shared/models/vega20-merged-bf16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ADAPTER_REPO = "vllm-sr/Decision-2.0-Vega-27B"
ADAPTER_SHA256 = "14c25f457edcdacbc1d9e8416b926b3421db9ba18c12f57febcf0adf8054accd"
CONFIG_SHA256 = "c2bb24361a5b318668ffd3998170f14b78851ca5e5f8e7bb586921366bc9029b"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def fetch_adapter(revision: str) -> tuple[Path, Path, str]:
    from huggingface_hub import HfApi, hf_hub_download

    resolved = HfApi().model_info(ADAPTER_REPO, revision=revision).sha
    config = Path(
        hf_hub_download(ADAPTER_REPO, "adapter/adapter_config.json", revision=resolved)
    )
    weights = Path(
        hf_hub_download(
            ADAPTER_REPO, "adapter/adapter_model.safetensors", revision=resolved
        )
    )
    return config, weights, resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", required=True, help="Qwen3.8-27B @ 1d4bf0f2 directory"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--revision", default="main", help="Revision of vllm-sr/Decision-2.0-Vega-27B"
    )
    parser.add_argument(
        "--adapter-dir",
        help="Local dir with adapter_config.json + adapter_model.safetensors",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    began = time.time()
    base = Path(args.base)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    if args.adapter_dir:
        config_path = Path(args.adapter_dir) / "adapter_config.json"
        weights_path = Path(args.adapter_dir) / "adapter_model.safetensors"
        resolved = "local"
    else:
        config_path, weights_path, resolved = fetch_adapter(args.revision)
    hashes = {
        "adapter_config.json": sha256(config_path),
        "adapter_model.safetensors": sha256(weights_path),
    }
    if (
        hashes["adapter_model.safetensors"] != ADAPTER_SHA256
        or hashes["adapter_config.json"] != CONFIG_SHA256
    ):
        raise SystemExit(
            f"adapter files differ from the released Decision-2.0-Vega-27B manifest: {hashes}"
        )
    adapter_config = json.loads(config_path.read_text())
    rank, alpha = adapter_config["r"], adapter_config["lora_alpha"]
    if adapter_config.get("use_rslora") or adapter_config.get("use_dora"):
        raise SystemExit("rsLoRA/DoRA adapters are not supported")
    scale = alpha / rank
    lora = load_file(str(weights_path))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora.items():
        if ".lora_A." in key:
            module, part = key.split(".lora_A.")[0], "A"
        elif ".lora_B." in key:
            module, part = key.split(".lora_B.")[0], "B"
        else:
            raise SystemExit(f"unexpected adapter tensor {key}")
        module = module.removeprefix("base_model.model.")
        pairs.setdefault(module, {})[part] = tensor
    targets = {
        f"model.language_model.{module}.weight": parts
        for module, parts in pairs.items()
    }
    index = json.loads((base / "model.safetensors.index.json").read_text())
    missing = [k for k in targets if k not in index["weight_map"]]
    if missing:
        raise SystemExit(f"adapter modules not in the base: {missing[:5]}")
    out_dtype = getattr(torch, args.dtype)
    stats = {}
    total_bytes = 0
    files = sorted(set(index["weight_map"].values()))
    for name in files:
        tensors = {}
        with safe_open(str(base / name), framework="pt") as handle:
            for key in handle.keys():
                weight = handle.get_tensor(key)
                if key in targets:
                    a = targets[key]["A"].to(args.device, torch.float32)
                    b = targets[key]["B"].to(args.device, torch.float32)
                    w = weight.to(args.device, torch.float32)
                    delta = (b @ a) * scale
                    merged = w + delta
                    cast = merged.to(out_dtype)
                    lost = (cast.float() - merged).norm()
                    dnorm = delta.norm()
                    stats[key] = {
                        "delta_norm": float(dnorm),
                        "weight_norm": float(w.norm()),
                        "kept_fraction": float(1 - lost / dnorm) if dnorm > 0 else 1.0,
                        "cos_cast_update": float(
                            torch.nn.functional.cosine_similarity(
                                (cast.float() - w).flatten(), delta.flatten(), dim=0
                            )
                        ),
                    }
                    tensors[key] = cast.cpu().contiguous()
                    del a, b, w, delta, merged, cast
                else:
                    tensors[key] = (
                        weight.contiguous()
                        if args.dtype == "bfloat16" or not weight.is_floating_point()
                        else weight.to(out_dtype).contiguous()
                    )
                total_bytes += tensors[key].numel() * tensors[key].element_size()
        save_file(tensors, str(partial / name), metadata={"format": "pt"})
        print(f"wrote {name} ({len(tensors)} tensors)", flush=True)
        del tensors
    if len(stats) != len(targets):
        raise SystemExit(f"merged {len(stats)} of {len(targets)} targets")
    index = dict(index)
    index["metadata"] = {**index.get("metadata", {}), "total_size": total_bytes}
    (partial / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for path in base.iterdir():
        if (
            path.is_file()
            and not path.name.endswith(".safetensors")
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copyfile(path, partial / path.name)
    kept = [s["kept_fraction"] for s in stats.values()]
    weighted = sum(
        s["kept_fraction"] * s["delta_norm"] ** 2 for s in stats.values()
    ) / sum(s["delta_norm"] ** 2 for s in stats.values())
    manifest = {
        "kind": "decision-2.0-vega-lora-merged",
        "base": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": BASE_REVISION,
            "path": str(base),
        },
        "adapter": {
            "repo": ADAPTER_REPO,
            "revision": resolved,
            "sha256": hashes,
            "r": rank,
            "alpha": alpha,
            "scale": scale,
            "modules": len(stats),
        },
        "merge": {
            "compute_dtype": "float32",
            "output_dtype": args.dtype,
            "head": "ignored (2.0 candidate head)",
        },
        "update_kept_after_cast": {
            "min": min(kept),
            "mean": sum(kept) / len(kept),
            "delta_norm_weighted": weighted,
        },
        "per_tensor": stats,
        "seconds": time.time() - began,
    }
    (partial / "MERGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for path in partial.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    os.replace(partial, out)
    summary = {k: v for k, v in manifest.items() if k != "per_tensor"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
