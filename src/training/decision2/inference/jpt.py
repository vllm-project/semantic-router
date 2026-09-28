"""Collect JPT's native llm2jev answers from gold-free typed prompts.

The model is loaded from an immutable Hugging Face download. Its published
llm2jev prompt and label-logprob scorer are used without task conversion.
``--size`` selects the pinned JPT-0.8B, JPT-4B or (default) JPT-9B release with
the temperature published on its model card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

SOURCE_REVISION = "2b252d504972764211ef172c1155ac0fedc9c3de"
VARIANTS = {
    "9b": (
        "kirp/jpt-9b",
        "7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e",
        1.087,
        "jpt-9b-llm2jev-hf-v1",
    ),
    "4b": (
        "kirp/jpt-4b",
        "78312f855b8bebf83ae7e9e8f05b5a0f73f519a9",
        1.036,
        "jpt-4b-llm2jev-hf-v1",
    ),
    "0.8b": (
        "kirp/jpt-0.8b",
        "1431c0509bbc10772cd59964cc7af5835c8720d6",
        1.140,
        "jpt-0.8b-llm2jev-hf-v1",
    ),
}
MODEL_ID, MODEL_REVISION, TEMPERATURE, ADAPTER_VERSION = VARIANTS["9b"]


def verify_source(source_path: Path) -> None:
    source_path = source_path.resolve(strict=True)
    git = ["git", "-c", f"safe.directory={source_path}", "-C", str(source_path)]
    actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_REVISION:
        raise ValueError(f"llm2jev source revision mismatch: {actual}")
    if subprocess.check_output([*git, "status", "--porcelain"], text=True):
        raise ValueError("llm2jev source checkout is modified")


def model_fingerprint(model_path: Path) -> dict[str, str]:
    required = [model_path / "config.json", model_path / "tokenizer.json"]
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards or any(not path.is_file() for path in required):
        raise ValueError("JPT-9B package lacks configuration, tokenizer or weights")
    return {path.name: file_digest(path) for path in [*required, *shards]}


def collect(
    *,
    model_path: Path,
    source_path: Path,
    model_revision: str,
    prompts: Path,
    output: Path,
    size: str = "9b",
) -> dict[str, Any]:
    if size not in VARIANTS:
        raise ValueError(f"Unknown JPT size: {size}")
    model_id, pinned_revision, temperature, adapter_version = VARIANTS[size]
    if model_revision != pinned_revision:
        raise ValueError(f"JPT {size} requires the pinned model revision")
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError("Refusing to overwrite JPT predictions")
    rows = load_prompts(prompts)
    model_path, source_path = (
        model_path.resolve(strict=True),
        source_path.resolve(strict=True),
    )
    if not local_revision(model_path, pinned_revision):
        raise ValueError(f"JPT {size} download does not attest its pinned revision")
    verify_source(source_path)
    files = model_fingerprint(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(source_path))

    import torch
    import transformers
    from llm2jev import LLM2Jev
    from llm2jev.backends import HF
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path))
    backend = HF(model=str(model_path))
    jev = LLM2Jev(processor, backend, temperature=temperature)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"items": 0, "questions": 0, "invalid_questions": 0}
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            synchronize("cuda:0")
            started = time.perf_counter()
            answers, usage = jev.run(row["state"], row["questions"])
            synchronize("cuda:0")
            if answers.keys() != row["questions"].keys():
                raise ValueError(f"{row['id']}: native scorer omitted a question")
            counts["items"] += 1
            counts["questions"] += len(answers)
            receipt = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": usage,
                "source_input_sha256": digest(
                    {"state": row["state"], "questions": row["questions"]}
                ),
                "backend": "jpt-llm2jev-hf",
                "model_id": model_id,
                "model_revision": pinned_revision,
                "adapter_version": adapter_version,
            }
            target.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            target.flush()
    manifest = {
        "model_id": model_id,
        "model_revision": pinned_revision,
        "source_revision": SOURCE_REVISION,
        "temperature": temperature,
        "adapter_version": adapter_version,
        "model_files_sha256": files,
        "input_sha256": file_digest(prompts),
        "output_sha256": file_digest(output),
        "counts": counts,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "backend": "llm2jev.backends.HF",
        },
    }
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=tuple(VARIANTS), default="9b")
    args = parser.parse_args()
    manifest = collect(
        model_path=args.model_path,
        source_path=args.source_path,
        model_revision=args.model_revision,
        prompts=args.input,
        output=args.output,
        size=args.size,
    )
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
