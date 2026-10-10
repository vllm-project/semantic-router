"""Weights check of an export before it becomes a release: the vision encoder is bit-equal to stock, the readout and
the inference fields of ``decision_config.json`` match the released export, every backbone tensor is BF16.

Tensors are compared as raw bytes from the safetensors headers (dtype, shape and data), so no framework is loaded.
``--stock`` may hold only the shards with vision tensors (plus the index); stock keys ``model.visual.*`` are matched to
``visual.*``.

    python -m d25.vega.release.weights_check --export <soup dir> --release <released export> --stock <stock dir> \
        --out weights-check.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

VISION_TENSORS = 333
VISION_PARAMETERS = 460_730_096
# decision_config.json fields that describe how the export was made, not how it is read.
RECORD_FIELDS = {
    "provenance",
    "parity",
    "export",
    "created",
    "source",
    "run",
    "step",
    "notes",
}


def header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        meta = json.loads(stream.read(size))
    meta.pop("__metadata__", None)
    return meta, 8 + size


def tensors(root: Path, keep=lambda key: True) -> dict[str, tuple[Path, int, dict]]:
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    headers: dict[str, tuple[dict, int]] = {}
    out = {}
    for key, name in weight_map.items():
        if not keep(key):
            continue
        if name not in headers:
            headers[name] = header(root / name)
        meta, base = headers[name]
        out[key] = (root / name, base, meta[key])
    return out


def data(entry: tuple[Path, int, dict]) -> bytes:
    path, base, meta = entry
    start, end = meta["data_offsets"]
    with open(path, "rb") as stream:
        stream.seek(base + start)
        return stream.read(end - start)


def same(a, b) -> bool:
    return (
        a[2]["dtype"] == b[2]["dtype"]
        and a[2]["shape"] == b[2]["shape"]
        and data(a) == data(b)
    )


def compare(ours: dict, theirs: dict) -> dict:
    common = sorted(set(ours) & set(theirs))
    differing = [k for k in common if not same(ours[k], theirs[k])]
    return {
        "compared": len(common),
        "equal": len(common) - len(differing),
        "differing": differing[:10],
        "missing": sorted(set(theirs) - set(ours))[:10],
        "extra": sorted(set(ours) - set(theirs))[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--stock", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    def visual(key: str) -> bool:
        return key.startswith(("visual.", "model.visual."))

    def strip(table: dict) -> dict:
        return {
            k[len("model.") :] if k.startswith("model.visual.") else k: v
            for k, v in table.items()
        }

    ours = tensors(args.export)
    vision = {k: v for k, v in ours.items() if visual(k)}
    parameters = sum(math.prod(v[2]["shape"]) for v in vision.values())
    stock = strip(tensors(args.stock, visual))
    release_vision = tensors(args.release, visual)
    backbone_dtypes = sorted({v[2]["dtype"] for k, v in ours.items() if not visual(k)})

    readout = {}
    for name, root in (("export", args.export), ("release", args.release)):
        meta, _ = header(root / "readout.safetensors")
        readout[name] = {k: (m["dtype"], m["shape"]) for k, m in meta.items()}
    readout_bytes_equal = (args.export / "readout.safetensors").read_bytes() == (
        args.release / "readout.safetensors"
    ).read_bytes()

    configs = [
        json.loads((root / "decision_config.json").read_text())
        for root in (args.export, args.release)
    ]
    inference_differs = sorted(
        k
        for k in set(configs[0]) | set(configs[1])
        if k not in RECORD_FIELDS and configs[0].get(k) != configs[1].get(k)
    )
    files_equal = {
        name: (args.export / name).read_bytes() == (args.release / name).read_bytes()
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        )
        if (args.export / name).exists() and (args.release / name).exists()
    }
    report = {
        "export": str(args.export),
        "tensors": len(ours),
        "vision": {
            "tensors": len(vision),
            "parameters": parameters,
            "dtypes": sorted({v[2]["dtype"] for v in vision.values()}),
        },
        "vision_vs_stock": compare(vision, stock),
        "vision_vs_release": compare(vision, release_vision),
        "backbone_dtypes": backbone_dtypes,
        "readout": readout,
        "readout_layout_as_release": readout["export"] == readout["release"],
        "readout_bytes_equal_release": readout_bytes_equal,
        "decision_config_inference_fields_differing": inference_differs,
        "decision_config_record_fields": sorted(
            k for k in configs[0] if k in RECORD_FIELDS
        ),
        "files_equal_release": files_equal,
    }
    report["vision_stock_ok"] = (
        len(vision) == VISION_TENSORS
        and parameters == VISION_PARAMETERS
        and report["vision_vs_stock"]["equal"] == VISION_TENSORS == len(stock)
    )
    report["ok"] = (
        report["vision_stock_ok"]
        and report["readout_layout_as_release"]
        and not inference_differs
        and backbone_dtypes == ["BF16"]
        and all(files_equal.values())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
