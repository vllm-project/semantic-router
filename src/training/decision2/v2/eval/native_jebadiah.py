"""Collect Jebadiah 27B's native answers (bundled ``scripts/`` Scorer) on gold-free prompts.

The release's ``scripts/`` modules are the trainer's own renderer (AINode's decide
prompt, copied verbatim) and label-logit read, used here exactly as
``scripts/decide_standalone.py`` does: merged BF16 weights, fp32 candidate logits and
the shipped per-type temperatures. Panel requests already use the release's
``{state, questions}`` format, so nothing is mapped on the way in, and
``answer_from_probs`` already emits the scorer's fields.

The release renderer cuts states over its 2,048-token budget. This collector never
scores a cut prompt: such questions, and questions with more options than the release
has single-token labels, are recorded as invalid. Any other error stops the run. The
release targets CUDA/MPS; ROCm runs are labelled ``unvalidated_rocm``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from v2.eval.native_jet import attested_revision
from v2.eval.same_panel import input_digest, sha_file

MODEL_ID = "frontier-infra/jebadiah-27b"
MODEL_REVISION = "c68db2b5570f3c2bdab511088885a88bd8b3a814"
ADAPTER_VERSION = "jebadiah-27b-native-v1"
RUNTIME_FILES = (
    "scripts/jebadiah_model.py",
    "scripts/jebadiah_prompt.py",
    "scripts/ainode_prompt_verbatim.py",
    "temperatures.json",
    "prompt_contract.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def verify_release(model_path: Path, revision: str) -> dict[str, Any]:
    if revision != MODEL_REVISION or attested_revision(model_path) != revision:
        raise ValueError("Jebadiah requires its attested, pinned Hugging Face revision")
    missing = [name for name in RUNTIME_FILES if not (model_path / name).is_file()]
    if missing:
        raise ValueError(f"Jebadiah release lacks {', '.join(missing)}")
    index = json.loads((model_path / "model.safetensors.index.json").read_text("utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    absent = [name for name in shards if not (model_path / name).is_file()]
    if absent:
        raise ValueError(f"Jebadiah weights incomplete: {len(absent)} shard(s) missing")
    return {
        "runtime_sha256": {name: sha_file(model_path / name) for name in RUNTIME_FILES},
        "weight_shards": len(shards),
    }


def invalid(question: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"type": question["type"], "invalid_reason": reason}


def collect(
    *,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda",
    max_items: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_items]
    model_path = model_path.resolve(strict=True)
    identity = verify_release(model_path, revision)
    config_sha = sha_file(model_path / "prompt_contract.json")
    sys.path.insert(0, str(model_path / "scripts"))
    import torch
    from jebadiah_model import (
        Scorer,
        load_base,
        load_tokenizer,
        read_temperatures,
        template_sha256,
    )
    from jebadiah_prompt import answer_from_probs

    contract = json.loads((model_path / "prompt_contract.json").read_text("utf-8"))
    tokenizer = load_tokenizer(str(model_path))
    if template_sha256(tokenizer) != contract["chat_template_sha256"]:
        raise ValueError(
            "Jebadiah tokenizer chat template differs from its prompt contract"
        )
    model = load_base(str(model_path), dtype=torch.bfloat16, device=device)
    temperatures = read_temperatures(str(model_path))
    scorer = Scorer(model, tokenizer, temperatures=temperatures, device=device)
    counts = {"over_budget": 0, "native_rejection": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            questions = row["questions"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            answers: dict[str, Any] = {}
            ready = []
            for name, question in questions.items():
                try:
                    rendered = scorer.render(row["state"], question)
                except ValueError as exc:
                    if "single-token labels" not in str(exc):
                        raise
                    answers[name] = invalid(question, "native_rejection")
                    counts["native_rejection"] += 1
                    continue
                if rendered.truncated:
                    answers[name] = invalid(question, "over_budget")
                    counts["over_budget"] += 1
                    continue
                ready.append((name, question, rendered))
            for start in range(0, len(ready), 8):
                chunk = ready[start : start + 8]
                probs = scorer.score_rendered([(r, q["type"]) for _, q, r in chunk])
                for (name, question, rendered), p in zip(chunk, probs):
                    answers[name] = answer_from_probs(question, rendered.keys, p)
            torch.cuda.synchronize()
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": {name: answers[name] for name in questions},
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "source_input_sha256": input_digest(row["state"], questions),
                        "backend": "jebadiah-scorer",
                        "model_id": MODEL_ID,
                        "model_revision": revision,
                        "temperatures": temperatures,
                        "adapter_version": ADAPTER_VERSION,
                        "model_config_sha256": config_sha,
                        "runtime_qualification": "unvalidated_rocm",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "items": len(rows),
        "invalid": counts,
        "weight_shards": identity["weight_shards"],
        "runtime_sha256": identity["runtime_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
                device=args.device,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
