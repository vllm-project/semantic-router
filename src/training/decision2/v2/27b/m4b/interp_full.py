"""Streaming weighted combinations of full ``DecisionModel`` checkpoints (CPU).

A full checkpoint (``checkpoint_format: full``) holds ``backbone/`` (config files and
FP32 safetensors shards with ``model.safetensors.index.json``), ``decision_head.safetensors``,
tokenizer files and ``decision_config.json``.

``soup`` is the uniform mean of N members; ``interp`` is alpha*S + (1 - alpha)*B for a
rational alpha = p/q. Weights are exact fractions k_i / q: every output element is
float32(sum_i k_i x_i / q), accumulated in float64 over row chunks of one tensor at a
time (members read with ``get_slice``, the output streamed to disk), so memory stays
bounded by the chunk size. Members with weight 0 are not read, and the sum starts from
the first weighted term, so alpha in {0, 1} reproduces an input bit for bit (signed
zeros included). Tensor names, shapes and dtypes, the backbone config files, the
tokenizer files and the architecture keys of ``decision_config.json`` must agree;
non-float tensors must be identical and are copied. The output keeps the first member's
shard layout (S for ``interp``), configs and tokenizer, and a ``decision_config.json``
whose ``initialization`` and ``combination`` record the method, the weights and every
member's identity; ``combination_manifest.json`` records inputs, outputs and the
verification.

``verify`` recomputes every tensor from the members and compares it bitwise with the
output, checks the copied files, and reports the largest relative deviation of the
output from the float64 combination (a soup must stay within 1e-6); a weight vector
with a single term (alpha in {0, 1}) must equal that member bitwise. Run after writing.

``merge-check`` verifies a materialized LoRA checkpoint (``training.model.materialize``)
in float64 against its pinned base and adapter: every target projection within 1e-6
relative (of max|expected|) of base + (alpha/r)·B@A, every other backbone tensor equal
to the base tensor cast to FP32, and the head copied bitwise.

    python3 -m v2.27b.m4b.interp_full soup --member A --member B --output OUT
    python3 -m v2.27b.m4b.interp_full interp --s S --b B --alpha 1/3 --output OUT
    python3 -m v2.27b.m4b.interp_full verify --output OUT [--report R.json]
    python3 -m v2.27b.m4b.interp_full merge-check --merged M --lora L --source-path BASE --output J
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

from training.model.data import canonical, file_sha256
from training.model.infer import MODEL_ROOT_FILES, checkpoint_fingerprint

TOOL_VERSION = "dev2-27b-m4b-interp-full/1"
SOUP_TOLERANCE = 1e-6
MERGE_TOLERANCE = 1e-6
CHUNK_ELEMENTS = 1 << 24
HEAD = "decision_head.safetensors"
INDEX = "model.safetensors.index.json"
MANIFEST = "combination_manifest.json"
SAME_KEYS = (
    "architecture",
    "prompt_version",
    "head_dim",
    "head_variant",
    "max_options",
    "checkpoint_format",
    "backbone_model_type",
)
DTYPES = {"F32": (torch.float32, 4), "F64": (torch.float64, 8), "BF16": (torch.bfloat16, 2),
          "F16": (torch.float16, 2), "I64": (torch.int64, 8), "I32": (torch.int32, 4),
          "I8": (torch.int8, 1), "U8": (torch.uint8, 1), "BOOL": (torch.bool, 1)}  # fmt: skip


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(size))


class Checkpoint:
    """Tensor name -> (file, dtype, shape) over the backbone shards and the head."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta = read_json(path / "decision_config.json")
        if self.meta.get("checkpoint_format", "full") != "full":
            raise ValueError(f"{path}: not a full checkpoint")
        backbone = path / "backbone"
        shards = sorted(backbone.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"{path}: no backbone shards")
        index = backbone / INDEX
        self.weight_map = (
            read_json(index)["weight_map"]
            if index.is_file()
            else {k: shards[0].name for k in header(shards[0]) if k != "__metadata__"}
        )
        self.files: dict[str, dict[str, tuple[str, list[int]]]] = {}
        for shard in shards:
            self.files[f"backbone/{shard.name}"] = {
                k: (v["dtype"], v["shape"])
                for k, v in header(shard).items()
                if k != "__metadata__"
            }
        mapped = {k: f"backbone/{v}" for k, v in self.weight_map.items()}
        listed = {k: f for f, names in self.files.items() for k in names}
        if mapped != listed:
            raise ValueError(f"{path}: {INDEX} disagrees with the shard headers")
        self.files[HEAD] = {
            k: (v["dtype"], v["shape"])
            for k, v in header(path / HEAD).items()
            if k != "__metadata__"
        }
        self.location = {k: f for f, names in self.files.items() for k in names}

    def layout(self) -> dict[str, tuple[str, list[int]]]:
        """Tensor name (head tensors prefixed ``head:``) -> (dtype, shape)."""
        out = {}
        for f, names in self.files.items():
            for k, v in names.items():
                out[("head:" if f == HEAD else "") + k] = v
        return out

    def side_files(self) -> list[str]:
        """Copied, non-weight files: backbone configs and the tokenizer files at the root."""
        files = [
            f"backbone/{p.name}"
            for p in sorted((self.path / "backbone").iterdir())
            if p.is_file() and p.suffix != ".safetensors"
        ]
        files += [
            p.name
            for p in sorted(self.path.iterdir())
            if p.is_file()
            and p.name in MODEL_ROOT_FILES - {"decision_config.json", HEAD}
        ]
        return files


def side_digest(path: Path) -> str:
    if path.suffix == ".json":
        return hashlib.sha256(canonical(read_json(path)).encode("utf-8")).hexdigest()
    return file_sha256(path)


def check_members(members: list[Checkpoint]) -> None:
    first = members[0]
    for member in members[1:]:
        for key in SAME_KEYS:
            if member.meta.get(key) != first.meta.get(key):
                raise ValueError(f"{member.path}: decision_config {key} differs")
        if member.layout() != first.layout():
            raise ValueError(f"{member.path}: tensor names, shapes or dtypes differ")
        if member.side_files() != first.side_files():
            raise ValueError(f"{member.path}: config / tokenizer file sets differ")
        for name in first.side_files():
            if side_digest(member.path / name) != side_digest(first.path / name):
                raise ValueError(f"{member.path}: {name} differs")


def parse_alpha(text: str) -> Fraction:
    alpha = Fraction(text)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    return alpha


def common(weights: list[Fraction]) -> tuple[list[int], int]:
    if sum(weights) != 1 or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative and sum to 1")
    den = math.lcm(*(w.denominator for w in weights))
    return [int(w * den) for w in weights], den


def row_chunks(shape: list[int]) -> Iterator[tuple[int, int] | None]:
    if not shape:
        yield None
        return
    inner = math.prod(shape[1:]) or 1
    step = max(1, CHUNK_ELEMENTS // inner)
    for start in range(0, shape[0], step):
        yield start, min(shape[0], start + step)


class Readers:
    """Open safetensors handles, one per (checkpoint, file)."""

    def __init__(self) -> None:
        self.handles: dict[tuple[str, str], Any] = {}

    def slice(
        self, ckpt: Checkpoint, name: str, rows: tuple[int, int] | None
    ) -> torch.Tensor:
        local = name.removeprefix("head:")
        file = HEAD if name.startswith("head:") else ckpt.location[local]
        key = (str(ckpt.path), file)
        if key not in self.handles:
            self.handles[key] = safe_open(
                str(ckpt.path / file), framework="pt", device="cpu"
            )
        handle = self.handles[key]
        if rows is None:
            return handle.get_tensor(local)
        return handle.get_slice(local)[rows[0] : rows[1]]


def combine(parts: list[tuple[int, torch.Tensor]], den: int) -> torch.Tensor:
    """float64 sum of k_i * x_i / den over the weighted terms, in order."""
    acc = None
    for k, x in parts:
        term = x.double() if k == 1 else x.double() * k
        acc = term if acc is None else acc + term
    return acc if den == 1 else acc / den


def chunk_values(
    readers: Readers, members: list[Checkpoint], nums: list[int], den: int, name: str,
    dtype: str, rows: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:  # fmt: skip
    """(output chunk, float64 combination or None for a copied non-float tensor)."""
    if dtype != "F32":
        tensors = [readers.slice(m, name, rows) for m in members]
        if any(not torch.equal(t, tensors[0]) for t in tensors[1:]):
            raise ValueError(f"{name}: non-FP32 tensor differs between members")
        return tensors[0], None
    parts = [(k, readers.slice(m, name, rows)) for k, m in zip(nums, members) if k]
    exact = combine(parts, den)
    return exact.float(), exact


def write_file(
    path: Path, names: list[str], layout: dict[str, tuple[str, list[int]]], produce
) -> None:
    """Stream one safetensors file: header first, then each tensor's chunks in name order."""
    offset, entries = 0, {}
    for name in names:
        dtype, shape = layout[name]
        size = math.prod(shape) * DTYPES[dtype][1]
        entries[name.removeprefix("head:")] = {
            "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]
        }  # fmt: skip
        offset += size
    head = json.dumps(
        {"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")
    )
    raw = head.encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        for name in names:
            dtype, shape = layout[name]
            for rows in row_chunks(shape):
                chunk = produce(name, dtype, rows).contiguous()
                stream.write(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def identity(path: Path) -> dict[str, str]:
    ident = checkpoint_fingerprint(path)
    return {"path": str(path), "model_sha256": ident["model_sha256"]}


def build(
    paths: list[Path], weights: list[Fraction], output: Path, method: str
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if len({str(p.resolve()) for p in paths}) != len(paths):
        raise ValueError("members must be distinct checkpoints")
    members = [Checkpoint(p) for p in paths]
    check_members(members)
    nums, den = common(weights)
    primary = members[0]
    layout = primary.layout()
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    (pending / "backbone").mkdir(parents=True)
    for name in primary.side_files():
        shutil.copyfile(primary.path / name, pending / name)
    readers = Readers()

    def produce(name: str, dtype: str, rows: tuple[int, int] | None) -> torch.Tensor:
        return chunk_values(readers, members, nums, den, name, dtype, rows)[0]

    for file, names in primary.files.items():
        keys = sorted(("head:" if file == HEAD else "") + k for k in names)
        write_file(pending / file, keys, layout, produce)
    weight_text = [str(w) for w in weights]
    meta = json.loads(json.dumps(primary.meta))
    meta.pop("lora_origin", None)
    meta.pop("soup", None)
    meta["initialization"] = (
        "uniform-soup-of-full-checkpoints"
        if method == "soup"
        else "interpolation-of-full-checkpoints"
    )
    meta["combination"] = {
        "tool_version": TOOL_VERSION,
        "method": method,
        "weights": weight_text,
        "members": [identity(m.path) for m in members],
        "precision": "float64 accumulation over row chunks, FP32 output",
    }
    (pending / "decision_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "tool_version": TOOL_VERSION,
        "code_sha256": {"interp_full.py": file_sha256(Path(__file__))},
        "method": method,
        "weights": weight_text,
        "members": meta["combination"]["members"],
        "primary": str(primary.path),
    }
    manifest["verification"] = verify_dir(pending, manifest)
    files = sorted(p for p in pending.rglob("*") if p.is_file())
    manifest["output"] = {
        "model_sha256": checkpoint_fingerprint(pending)["model_sha256"],
        "files_sha256": {
            p.relative_to(pending).as_posix(): file_sha256(p) for p in files
        },
    }
    (pending / MANIFEST).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(pending, output)
    return manifest


def verify_dir(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    members = [Checkpoint(Path(m["path"])) for m in manifest["members"]]
    check_members(members)
    weights = [Fraction(w) for w in manifest["weights"]]
    nums, den = common(weights)
    out = Checkpoint(output)
    if (
        out.layout() != members[0].layout()
        or out.files.keys() != members[0].files.keys()
    ):
        raise ValueError("output tensor layout differs from the members'")
    if out.weight_map != members[0].weight_map:
        raise ValueError(f"output {INDEX} differs from the primary member's")
    for name in members[0].side_files():
        if file_sha256(output / name) != file_sha256(members[0].path / name):
            raise ValueError(f"output {name} differs from the primary member's")
    single = (
        [i for i, k in enumerate(nums) if k] if sum(1 for k in nums if k) == 1 else []
    )
    readers, got = Readers(), Readers()
    worst, tensors, copied = 0.0, 0, 0
    for name, (dtype, shape) in sorted(out.layout().items()):
        tensors += 1
        peak = diff = 0.0
        for rows in row_chunks(shape):
            value, exact = chunk_values(readers, members, nums, den, name, dtype, rows)
            written = got.slice(out, name, rows)
            if written.dtype != value.dtype or not torch.equal(
                written.reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            ):
                raise ValueError(
                    f"{name}: output differs from the recomputed combination"
                )
            if single:
                member = readers.slice(members[single[0]], name, rows)
                if not torch.equal(
                    member.reshape(-1).view(torch.uint8),
                    written.reshape(-1).view(torch.uint8),
                ):
                    raise ValueError(
                        f"{name}: a single-term output differs from its member"
                    )
            if exact is None:
                copied += 1
                continue
            if exact.numel():
                diff = max(diff, (written.double() - exact).abs().max().item())
                peak = max(peak, exact.abs().max().item())
        ratio = diff / peak if peak > 0 else (0.0 if diff == 0 else float("inf"))
        worst = max(worst, ratio)
    if manifest["method"] == "soup" and not worst <= SOUP_TOLERANCE:
        raise ValueError(f"soup deviates {worst} from the float64 mean")
    return {
        "tensors": tensors,
        "copied_non_fp32_chunks": copied,
        "bitwise_recomputed": True,
        "single_term_bitwise_member": (
            members[single[0]].path.as_posix() if single else None
        ),
        "max_relative_dev_from_float64": worst,
        "soup_tolerance_relative": SOUP_TOLERANCE,
    }


def verify(output: Path) -> dict[str, Any]:
    manifest = read_json(output / MANIFEST)
    result = verify_dir(output, manifest)
    for name, digest in manifest["output"]["files_sha256"].items():
        if file_sha256(output / name) != digest:
            raise ValueError(f"{name} changed after the build")
    if (
        checkpoint_fingerprint(output)["model_sha256"]
        != manifest["output"]["model_sha256"]
    ):
        raise ValueError("output identity changed after the build")
    return {
        "output": str(output),
        "model_sha256": manifest["output"]["model_sha256"],
        **result,
    }


TEXT_PREFIXES = ("model.language_model.", "language_model.", "model.")


def text_name(name: str) -> str | None:
    """A text-backbone tensor name without its wrapper prefix; None outside the text model."""
    if name.startswith(("model.visual.", "visual.", "mtp.", "lm_head.")):
        return None
    for prefix in TEXT_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def base_tensors(source: Path) -> dict[str, tuple[Path, str]]:
    """Text name -> (shard, base tensor name) for the base's text-backbone tensors."""
    index = source / INDEX
    weight_map = (
        read_json(index)["weight_map"]
        if index.is_file()
        else {
            k: "model.safetensors"
            for k in header(source / "model.safetensors")
            if k != "__metadata__"
        }
    )
    out = {}
    for name, shard in weight_map.items():
        if name.startswith(TEXT_PREFIXES) and text_name(name) is not None:
            out[text_name(name)] = (source / shard, name)
    return out


def merge_check(merged: Path, lora: Path, source: Path) -> dict[str, Any]:
    out = Checkpoint(merged)
    meta = read_json(lora / "decision_config.json")
    contract = meta["lora"]
    config = read_json(lora / "adapter" / "adapter_config.json")
    scale = config["lora_alpha"] / config["r"]
    base = base_tensors(source)
    backbone = {k: v for k, v in out.layout().items() if not k.startswith("head:")}
    texts = {k: text_name(k) for k in backbone}
    if None in texts.values() or sorted(texts.values()) != sorted(base):
        raise ValueError("merged backbone tensors differ from the base text tensors")
    targets = {f"{text_name(t)}.weight": t for t in contract["target_modules"]}
    if not set(targets) <= set(texts.values()):
        raise ValueError("a LoRA target has no merged tensor")
    adapter = safe_open(
        str(lora / "adapter" / "adapter_model.safetensors"), "pt", device="cpu"
    )
    handles: dict[Path, Any] = {}
    readers = Readers()
    worst, per_projection, others, params = 0.0, [], 0, 0
    for name, (dtype, shape) in sorted(backbone.items()):
        if dtype != "F32":
            raise ValueError(f"{name}: merged tensor is {dtype}, expected F32")
        params += math.prod(shape)
        shard, base_name = base[texts[name]]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), "pt", device="cpu")
        if list(handles[shard].get_slice(base_name).get_shape()) != shape:
            raise ValueError(f"{name}: shape differs from the base")
        target = targets.get(texts[name])
        stem = f"base_model.model.{target}" if target else None
        a = adapter.get_tensor(f"{stem}.lora_A.weight").double() if stem else None
        b = adapter.get_slice(f"{stem}.lora_B.weight") if stem else None
        diff = peak = 0.0
        for rows in row_chunks(shape):
            written = readers.slice(out, name, rows)
            if rows is None:
                reference = handles[shard].get_tensor(base_name)
            else:
                reference = handles[shard].get_slice(base_name)[rows[0] : rows[1]]
            if stem is None:
                if not torch.equal(written, reference.float()):
                    raise ValueError(f"{name}: merged tensor differs from the base")
                continue
            expected = reference.double() + scale * (b[rows[0] : rows[1]].double() @ a)
            diff = max(diff, (written.double() - expected).abs().max().item())
            peak = max(peak, expected.abs().max().item())
        if stem is None:
            others += 1
            continue
        ratio = diff / peak if peak > 0 else (0.0 if diff == 0 else float("inf"))
        worst = max(worst, ratio)
        per_projection.append([target, diff, peak])
        if not diff <= MERGE_TOLERANCE * peak:
            raise ValueError(
                f"{name}: merged projection deviates {ratio} from base + s*BA"
            )
    head = {k: v for k, v in out.layout().items() if k.startswith("head:")}
    source_head = safe_open(str(lora / HEAD), "pt", device="cpu")
    if sorted(k.removeprefix("head:") for k in head) != sorted(source_head.keys()):
        raise ValueError("merged head tensors differ from the LoRA checkpoint's")
    for name in head:
        local = name.removeprefix("head:")
        if not torch.equal(
            readers.slice(out, name, None).reshape(-1).view(torch.uint8),
            source_head.get_tensor(local).reshape(-1).view(torch.uint8),
        ):
            raise ValueError(f"{name}: head is not copied bitwise")
        params += math.prod(head[name][1])
    return {
        "schema": "decision2-27b-m4b-merge-check/1",
        "merged": str(merged),
        "merged_model_sha256": checkpoint_fingerprint(merged)["model_sha256"],
        "lora": str(lora),
        "lora_model_sha256": checkpoint_fingerprint(lora, source)["model_sha256"],
        "scale": scale,
        "projections": len(per_projection),
        "other_backbone_tensors_equal_base_fp32": others,
        "head_tensors_bitwise": len(head),
        "tolerance_relative": MERGE_TOLERANCE,
        "max_relative_diff": worst,
        "per_projection": per_projection,
        "loaded_parameters": params,
        "compute": "float64 CPU, row chunks",
        "code_sha256": file_sha256(Path(__file__)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("soup")
    p.add_argument("--member", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("interp")
    p.add_argument("--s", type=Path, required=True, help="the alpha end (S)")
    p.add_argument("--b", type=Path, required=True, help="the 1 - alpha end (B)")
    p.add_argument("--alpha", required=True, help="rational, e.g. 1/3")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    p = sub.add_parser("merge-check")
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--source-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "soup":
        if len(args.member) < 2:
            parser.error("a soup needs at least two members")
        n = len(args.member)
        result = build(args.member, [Fraction(1, n)] * n, args.output, "soup")
    elif args.mode == "interp":
        alpha = parse_alpha(args.alpha)
        result = build([args.s, args.b], [alpha, 1 - alpha], args.output, "interp")
    elif args.mode == "verify":
        result = verify(args.output)
        if args.report:
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = merge_check(args.merged, args.lora, args.source_path)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
        result = {
            k: result[k]
            for k in ("projections", "max_relative_diff", "loaded_parameters")
        }
    if "verification" in result:
        result = {
            "output": str(args.output),
            "model_sha256": result["output"]["model_sha256"],
            "weights": result["weights"],
            **{
                k: result["verification"][k]
                for k in ("tensors", "max_relative_dev_from_float64")
            },
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
