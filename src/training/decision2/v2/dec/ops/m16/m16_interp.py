"""Decoder M16 weight interpolation between a released checkpoint R and an arm X (prereg dec-m16-prereg-2026-10-01.md,
"Points"): W(alpha) = (1 - alpha) * R + alpha * X over every backbone tensor and the decision head, per tensor in FP32.

  lineage  R and X must be full decoder checkpoints with the same architecture, prompt version, head dimension,
           checkpoint format, maximum options and full_training_source, identical tokenizer files, and the same tensor
           names and shapes in the backbone and the head (shard layouts may differ). Prints the check as JSON; exit 1
           when the pair fails.
  build    the lineage check, then W(alpha) written in X's shard layout with X's tokenizer and backbone config;
           decision_config.json records "interpolation" (alpha, formula, both paths and model_sha256).

usage: m16_interp.py lineage --release R --arm X [--output J]
       m16_interp.py build --release R --arm X --alpha A --output DIR
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCHEMA = "dec-m16-interp/1"
SAME = (
    "architecture",
    "prompt_version",
    "head_dim",
    "checkpoint_format",
    "max_options",
    "full_training_source",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
HEAD = "decision_head.safetensors"


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_layout(root: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """name -> (shard file, shape) over the backbone shards."""
    layout = {}
    for shard in sorted((root / "backbone").glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                if key in layout:
                    raise ValueError(f"{root}: tensor {key} appears in two shards")
                layout[key] = (shard.name, tuple(handle.get_slice(key).get_shape()))
    return layout


def head_layout(root: Path) -> dict[str, tuple[int, ...]]:
    path = root / HEAD
    if not path.is_file():
        return {}
    with safe_open(str(path), framework="pt") as handle:
        return {k: tuple(handle.get_slice(k).get_shape()) for k in handle.keys()}


def lineage(release: Path, arm: Path) -> dict[str, Any]:
    reasons = []
    metas = [
        json.loads((p / "decision_config.json").read_text()) for p in (release, arm)
    ]
    for key in SAME:
        if json.dumps(metas[0].get(key), sort_keys=True) != json.dumps(
            metas[1].get(key), sort_keys=True
        ):
            reasons.append(f"decision_config differs in {key}")
    if metas[0].get("checkpoint_format") != "full":
        reasons.append("interpolation is defined for full checkpoints")
    tokenizer = {}
    for name in TOKENIZER_FILES:
        present = [(p / name).is_file() for p in (release, arm)]
        if present[0] != present[1]:
            reasons.append(f"only one side has {name}")
        elif present[0]:
            hashes = [file_sha256(p / name) for p in (release, arm)]
            tokenizer[name] = hashes[1]
            if hashes[0] != hashes[1]:
                reasons.append(f"{name} differs")
    layouts = [tensor_layout(p) for p in (release, arm)]
    if set(layouts[0]) != set(layouts[1]):
        reasons.append(
            f"backbone tensor names differ ({len(set(layouts[0]) ^ set(layouts[1]))} not shared)"
        )
    else:
        bad = [k for k in layouts[0] if layouts[0][k][1] != layouts[1][k][1]]
        if bad:
            reasons.append(
                f"backbone tensor shapes differ for {len(bad)} tensors, e.g. {bad[0]}"
            )
    heads = [head_layout(p) for p in (release, arm)]
    if not heads[0] or heads[0] != heads[1]:
        reasons.append("decision heads are missing or differ in names / shapes")
    from v2.dec.dec_model import dec_fingerprint

    return {
        "schema": SCHEMA,
        "release": str(release),
        "arm": str(arm),
        "release_model_sha256": dec_fingerprint(release, None)["model_sha256"],
        "arm_model_sha256": dec_fingerprint(arm, None)["model_sha256"],
        "same_fields": list(SAME),
        "full_training_source": metas[1].get("full_training_source"),
        "tokenizer_sha256": tokenizer,
        "backbone_tensors": len(layouts[1]),
        "release_shards": sorted({s for s, _ in layouts[0].values()}),
        "arm_shards": sorted({s for s, _ in layouts[1].values()}),
        "head_tensors": len(heads[1]),
        "pass": not reasons,
        "reasons": reasons,
    }


def mix(r: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    return ((1.0 - alpha) * r.float() + alpha * x.float()).contiguous()


def build(release: Path, arm: Path, alpha: float, output: Path) -> dict[str, Any]:
    from v2.dec.dec_model import dec_fingerprint

    if not (0.0 < alpha < 1.0) or not math.isfinite(alpha):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if output.exists():
        raise FileExistsError(output)
    check = lineage(release, arm)
    if not check["pass"]:
        raise ValueError(f"lineage check failed: {check['reasons']}")
    r_layout = tensor_layout(release)
    pending = output.with_name(output.name + ".pending")
    shutil.copytree(arm, pending, ignore=shutil.ignore_patterns("*.safetensors"))
    handles = {}
    try:
        for shard in sorted((arm / "backbone").glob("*.safetensors")):
            x = load_file(str(shard))
            out = {}
            for key, tensor in x.items():
                r_shard = r_layout[key][0]
                if r_shard not in handles:
                    handles[r_shard] = safe_open(
                        str(release / "backbone" / r_shard), framework="pt"
                    )
                out[key] = mix(handles[r_shard].get_tensor(key), tensor, alpha)
            save_file(
                out, str(pending / "backbone" / shard.name), metadata={"format": "pt"}
            )
            del x, out
    finally:
        handles.clear()
    r_head, x_head = load_file(str(release / HEAD)), load_file(str(arm / HEAD))
    save_file(
        {k: mix(r_head[k], v, alpha) for k, v in x_head.items()}, str(pending / HEAD)
    )
    ids = {release: check["release_model_sha256"], arm: check["arm_model_sha256"]}
    meta = json.loads((arm / "decision_config.json").read_text())
    meta["interpolation"] = {
        "alpha": alpha,
        "formula": "W = (1 - alpha) * release + alpha * arm, per tensor in FP32 (backbone and decision head)",
        "release": {"path": str(release), "model_sha256": ids[release]},
        "arm": {
            "path": str(arm),
            "model_sha256": ids[arm],
            "soup": meta.pop("soup", None),
        },
    }
    meta["initialization"] = "linear-interpolation-of-full-checkpoints"
    (pending / "decision_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if (pending / "checkpoint.json").exists():
        (pending / "checkpoint.json").unlink()
    pending.rename(output)
    return {
        "output": str(output),
        "alpha": alpha,
        "release_model_sha256": ids[release],
        "arm_model_sha256": ids[arm],
        "model_sha256": dec_fingerprint(output, None)["model_sha256"],
        "head_sha256": file_sha256(output / HEAD),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    lin = sub.add_parser("lineage")
    lin.add_argument("--release", type=Path, required=True)
    lin.add_argument("--arm", type=Path, required=True)
    lin.add_argument("--output", type=Path)
    bld = sub.add_parser("build")
    bld.add_argument("--release", type=Path, required=True)
    bld.add_argument("--arm", type=Path, required=True)
    bld.add_argument("--alpha", type=float, required=True)
    bld.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "lineage":
        result = lineage(args.release, args.arm)
        text = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        print(json.dumps({"pass": result["pass"], "reasons": result["reasons"]}))
        return 0 if result["pass"] else 1
    print(json.dumps(build(args.release, args.arm, args.alpha, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
