"""Exact uniform soup of same-recipe ``peft-lora/1`` seed checkpoints.

LoRA factors are concatenated along the rank: A' = [A_1; ...; A_N],
B' = [B_1, ..., B_N] / N, r' = N r, alpha' = N alpha, so the scale alpha'/r'
equals alpha/r and B'A' * scale = mean_i(B_i A_i * scale) exactly (up to FP32
rounding). The decision head, the only other trainable tensor set in LoRA mode,
is averaged uniformly. Dropout, targets and the source contract are unchanged.
No calibration is copied: a soup needs its own fit.

Before the output is renamed into place the tool checks, on CPU in row chunks,
that every projection satisfies max|dW_soup - mean dW_i| <= 1e-6 max|mean dW_i|
(FP32), that ``verify_adapter_config`` accepts the new adapter, and that the
reloaded head equals the mean; ``soup_manifest.json`` records the inputs, the
output identity, per-file SHA-256 and the verification statistics.

    python3 -m v2.27b.lora_soup --member CKPT_S1 --member CKPT_S2 \
        --source-path BASE --output SOUP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from training.model.data import canonical, file_sha256
from training.model.infer import MODEL_ROOT_FILES, checkpoint_fingerprint
from training.model.lora import LORA_FORMAT, verify_adapter_config

TOOL_VERSION = "dev2-27b-lora-soup/1"
TOLERANCE = 1e-6
CHUNK_ELEMENTS = 1 << 24
ADAPTER_WEIGHTS = "adapter/adapter_model.safetensors"
ADAPTER_CONFIG = "adapter/adapter_config.json"
HEAD = "decision_head.safetensors"
MANIFEST = "soup_manifest.json"
EXPECTED_ADAPTER_CONFIG = {
    "peft_type": "LORA",
    "bias": "none",
    "use_rslora": False,
    "use_dora": False,
    "rank_pattern": {},
    "alpha_pattern": {},
    "modules_to_save": None,
}
UNSUPPORTED_ENTRIES = ("backbone", "dec_residual.safetensors", "calibration.json")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _stem(name: str) -> str:
    return f"base_model.model.{name}"


def _tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in sorted(handle.keys())}


def check_members(members: list[Path]) -> dict[str, Any]:
    """Refuse members that differ in anything but their trained tensors."""
    if len(members) < 2:
        raise ValueError("A soup needs at least two members")
    if len({str(m.resolve()) for m in members}) != len(members):
        raise ValueError("Soup members must be distinct checkpoints")
    metas = [_json(m / "decision_config.json") for m in members]
    configs = [_json(m / ADAPTER_CONFIG) for m in members]
    for config in configs:
        # PEFT serializes its target set in hash order, which differs per process.
        if isinstance(config.get("target_modules"), list):
            config["target_modules"] = sorted(config["target_modules"])
    first, config = metas[0], configs[0]
    if first.get("checkpoint_format") != LORA_FORMAT:
        raise ValueError(f"Soup members must be {LORA_FORMAT} checkpoints")
    for member, meta in zip(members, metas):
        if canonical(meta) != canonical(first):
            raise ValueError(f"{member}: decision_config differs from the first member")
        present = [name for name in UNSUPPORTED_ENTRIES if (member / name).exists()]
        if present:
            raise ValueError(f"{member}: unsupported soup inputs {present}")
    for member, value in zip(members, configs):
        if canonical(value) != canonical(config):
            raise ValueError(f"{member}: adapter_config differs from the first member")
    for key, expected in EXPECTED_ADAPTER_CONFIG.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"Adapter config {key}={config.get(key)!r} is unsupported")
    contract = first["lora"]
    for member in members:
        verify_adapter_config(member / "adapter", contract)
    roots = [
        {
            name: file_sha256(m / name)
            for name in sorted(MODEL_ROOT_FILES - {"decision_config.json", HEAD})
            if (m / name).is_file()
        }
        for m in members
    ]
    if any(r != roots[0] for r in roots):
        raise ValueError("Soup members have different tokenizer files")
    heads = []
    for member in members:
        with safe_open(str(member / HEAD), framework="pt", device="cpu") as handle:
            heads.append(
                {
                    key: (
                        str(handle.get_slice(key).get_dtype()),
                        handle.get_slice(key).get_shape(),
                    )
                    for key in handle.keys()
                }
            )
    if any(h != heads[0] for h in heads):
        raise ValueError("Soup members have different decision head layouts")
    if any(dtype != "F32" for dtype, _ in heads[0].values()):
        raise ValueError("Decision head tensors must be FP32")
    return {"metadata": first, "adapter_config": config, "root_files": sorted(roots[0])}


def soup_adapter(members: list[Path], targets: list[str]) -> dict[str, torch.Tensor]:
    count = len(members)
    loaded = [_tensors(m / ADAPTER_WEIGHTS) for m in members]
    out: dict[str, torch.Tensor] = {}
    for name in targets:
        a, b = f"{_stem(name)}.lora_A.weight", f"{_stem(name)}.lora_B.weight"
        out[a] = torch.cat([t[a] for t in loaded], dim=0).contiguous()
        out[b] = torch.cat([t[b] / count for t in loaded], dim=1).contiguous()
    return out


def mean_head(members: list[Path]) -> dict[str, torch.Tensor]:
    heads = [_tensors(m / HEAD) for m in members]
    return {
        key: (sum(h[key].double() for h in heads) / len(heads)).float().contiguous()
        for key in heads[0]
    }


def projection_stats(
    factors: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
    soup: tuple[torch.Tensor, torch.Tensor],
    soup_scale: float,
) -> tuple[float, float]:
    """(max|dW_soup - mean dW_i|, max|mean dW_i|) in FP32, computed in row chunks."""
    soup_a, soup_b = soup
    rows = soup_b.shape[0]
    step = max(1, CHUNK_ELEMENTS // soup_a.shape[1])
    worst = peak = 0.0
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        reference = torch.zeros(stop - start, soup_a.shape[1], dtype=torch.float32)
        for a, b in factors:
            reference += (b[start:stop] @ a) * scale
        reference /= len(factors)
        delta = (soup_b[start:stop] @ soup_a) * soup_scale
        worst = max(worst, (delta - reference).abs().max().item())
        peak = max(peak, reference.abs().max().item())
    return worst, peak


def verify(members: list[Path], output: Path) -> dict[str, Any]:
    """Recompute every projection update and the head from the written files."""
    meta = _json(output / "decision_config.json")
    config = _json(output / ADAPTER_CONFIG)
    contract = meta["lora"]
    verify_adapter_config(output / "adapter", contract)
    member_config = _json(members[0] / ADAPTER_CONFIG)
    scale = member_config["lora_alpha"] / member_config["r"]
    soup_scale = config["lora_alpha"] / config["r"]
    if soup_scale != scale:
        raise ValueError("Soup LoRA scale differs from the members' scale")
    loaded = [_tensors(m / ADAPTER_WEIGHTS) for m in members]
    soup = _tensors(output / ADAPTER_WEIGHTS)
    per_projection, failures = [], []
    worst_ratio = 0.0
    for name in contract["target_modules"]:
        a, b = f"{_stem(name)}.lora_A.weight", f"{_stem(name)}.lora_B.weight"
        diff, peak = projection_stats(
            [(t[a], t[b]) for t in loaded], scale, (soup[a], soup[b]), soup_scale
        )
        ratio = diff / peak if peak > 0 else (0.0 if diff == 0 else float("inf"))
        worst_ratio = max(worst_ratio, ratio)
        per_projection.append([name, diff, peak])
        if not diff <= TOLERANCE * peak:
            failures.append(name)
    if failures:
        raise ValueError(
            f"Soup update differs from the mean update on {len(failures)} projections: {failures[:5]}"
        )
    expected_head = mean_head(members)
    head = _tensors(output / HEAD)
    if set(head) != set(expected_head) or any(
        not torch.equal(head[k], expected_head[k]) for k in head
    ):
        raise ValueError("Soup head differs from the members' mean head")
    heads = [_tensors(m / HEAD) for m in members]
    head_dev = max(
        (head[k].double() - sum(h[k].double() for h in heads) / len(heads))
        .abs()
        .max()
        .item()
        for k in head
    )
    return {
        "projections": len(per_projection),
        "tolerance_relative": TOLERANCE,
        "max_abs_diff": max(p[1] for p in per_projection),
        "max_relative_diff": worst_ratio,
        "per_projection": per_projection,
        "scale": scale,
        "verify_adapter_config": True,
        "head_tensors": len(head),
        "head_max_abs_dev_from_float64_mean": head_dev,
        "compute": "FP32 CPU, row chunks",
    }


def _without_source(files: dict[str, str]) -> dict[str, str]:
    return {
        name.removeprefix("checkpoint/"): sha
        for name, sha in files.items()
        if not name.startswith("source/")
    }


def _write_json(path: Path, value: Any, *, sort_keys: bool) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def build(members: list[Path], source_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    checked = check_members(members)
    identities = [checkpoint_fingerprint(m, source_path) for m in members]
    count = len(members)
    meta = json.loads(json.dumps(checked["metadata"]))
    contract = meta["lora"]
    rank, alpha = contract["rank"], contract["alpha"]
    contract["rank"], contract["alpha"] = rank * count, alpha * count
    meta["soup"] = {
        "method": "uniform LoRA soup: factors concatenated along the rank, B scaled by 1/N; head averaged",
        "members": [ident["model_sha256"] for ident in identities],
        "member_rank": rank,
        "member_alpha": alpha,
    }
    config = dict(checked["adapter_config"])
    config["r"], config["lora_alpha"] = rank * count, alpha * count
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    (pending / "adapter").mkdir(parents=True)
    for name in checked["root_files"]:
        shutil.copyfile(members[0] / name, pending / name)
    readme = members[0] / "adapter" / "README.md"
    if readme.is_file():
        shutil.copyfile(readme, pending / "adapter" / "README.md")
    _write_json(pending / ADAPTER_CONFIG, config, sort_keys=True)
    save_file(
        soup_adapter(members, contract["target_modules"]),
        str(pending / ADAPTER_WEIGHTS),
        metadata={"format": "pt"},
    )
    save_file(mean_head(members), str(pending / HEAD))
    _write_json(pending / "decision_config.json", meta, sort_keys=False)
    verification = verify(members, pending)
    identity = checkpoint_fingerprint(pending, source_path)
    files = sorted(p for p in pending.rglob("*") if p.is_file())
    manifest = {
        "tool_version": TOOL_VERSION,
        "code_sha256": {"lora_soup.py": file_sha256(Path(__file__))},
        "members": [
            {
                "path": str(m),
                "model_sha256": ident["model_sha256"],
                "checkpoint_sha256": _digest(_without_source(ident["files_sha256"])),
                "files_sha256": _without_source(ident["files_sha256"]),
            }
            for m, ident in zip(members, identities)
        ],
        "source_files_sha256": _digest(
            {
                name.removeprefix("source/"): sha
                for name, sha in identity["files_sha256"].items()
                if name.startswith("source/")
            }
        ),
        "lora": {
            "members": count,
            "member_rank": rank,
            "member_alpha": alpha,
            "rank": rank * count,
            "alpha": alpha * count,
            "dropout": contract["dropout"],
        },
        "output": {
            "model_sha256": identity["model_sha256"],
            "checkpoint_sha256": _digest(_without_source(identity["files_sha256"])),
            "files_sha256": {
                p.relative_to(pending).as_posix(): file_sha256(p) for p in files
            },
        },
        "calibration": "not copied; fit a new calibration for this soup",
        "verification": verification,
    }
    _write_json(pending / MANIFEST, manifest, sort_keys=True)
    for path in [*files, pending / MANIFEST]:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    os.replace(pending, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.member, args.source_path, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_sha256": manifest["output"]["model_sha256"],
                "members": [m["model_sha256"] for m in manifest["members"]],
                "projections": manifest["verification"]["projections"],
                "max_relative_diff": manifest["verification"]["max_relative_diff"],
            }
        )
    )


if __name__ == "__main__":
    main()
