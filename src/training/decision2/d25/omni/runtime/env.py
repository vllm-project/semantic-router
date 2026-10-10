"""Environment of the evidence pod and processor equivalence (no model weights loaded).

Records library versions and the kernels transformers binds, then checks that the package's processor (stock
``preprocessor_config.json`` with the 1.6 MP budget set in code) and the processor the engine loads from the
assembled evaluation checkpoint (``processor_config.json`` with the budget saved) render identical tensors
for a sample of suite rows.

    python -m d25.omni.runtime.env --package PKG --engine-ckpt GRAFT --suite SUITE --out env.json
"""

from __future__ import annotations

import argparse
import importlib
import platform
from pathlib import Path

from d25.omni.runtime.common import load_runtime, read_jsonl, stratified, write_json


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "torch",
        "transformers",
        "torchvision",
        "PIL",
        "fla",
        "triton",
        "fastapi",
        "httpx",
    ):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"missing ({type(exc).__name__})"
    import torch

    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--engine-ckpt", required=True, type=Path)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from d25.omni.common import vision_format
    from d25.omni.model import inputs
    from transformers import AutoProcessor

    rt = load_runtime(args.package)
    report = {"versions": versions(), "kernels": rt.kernel_report()}
    package_proc = rt.load_processor(args.package)
    engine_proc = inputs.setup_processor(
        AutoProcessor.from_pretrained(str(args.engine_ckpt)), vision_format.MAX_PIXELS
    )
    report["processors"] = {
        "package": type(package_proc.image_processor).__name__,
        "engine": type(engine_proc.image_processor).__name__,
        "package_size": dict(package_proc.image_processor.size),
        "engine_size": dict(engine_proc.image_processor.size),
    }
    codes = rt.answer_codes(package_proc.tokenizer)[0]
    rows = stratified(read_jsonl(args.suite / "rows.jsonl.gz"), args.rows, min_multi=10)
    mismatches, keys = [], set()
    for row in rows:
        ((qid, question),) = row["questions"].items()
        paths = [str(args.suite / ref) for ref in row["images"]]
        images = [rt.load_image(p) for p in paths]
        text_engine = inputs.render(
            engine_proc, "d25-vega", row.get("state"), question, codes, len(images)
        )
        text_package = package_proc.apply_chat_template(
            rt.image_messages(
                "d25-vega", row.get("state"), question, codes, len(images)
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        a = engine_proc(
            text=[text_engine], images=images, padding=True, return_tensors="pt"
        )
        b = package_proc(
            text=[text_package], images=images, padding=True, return_tensors="pt"
        )
        keys |= set(a) | set(b)
        bad = [
            k
            for k in sorted(set(a) | set(b))
            if k not in a
            or k not in b
            or a[k].dtype != b[k].dtype
            or not torch.equal(a[k], b[k])
        ]
        if text_engine != text_package or bad:
            mismatches.append(
                {
                    "id": row["id"],
                    "text_equal": text_engine == text_package,
                    "tensors": bad,
                }
            )
    report["processor_equivalence"] = {
        "rows": len(rows),
        "multi_image_rows": sum(len(r["images"]) > 1 for r in rows),
        "tensor_keys": sorted(keys),
        "mismatches": len(mismatches),
        "examples": mismatches[:10],
    }
    write_json(args.out, report)
    print(
        report["versions"],
        report["processors"],
        report["processor_equivalence"]["mismatches"],
    )


if __name__ == "__main__":
    main()
