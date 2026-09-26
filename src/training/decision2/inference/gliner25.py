"""Gold-free, native-classification baseline for pinned GLiNER2.5-Decide.

Each Decision question becomes one exclusive GLiNER classification schema. Its
instruction and option descriptions use the native schema fields. The official
``Classifier.score`` path supplies per-label logits and native softmax
probabilities. No benchmark labels enter this collector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

MODEL_ID = "fastino/GLiNER2.5-Decide"
REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
LIBRARY_COMMIT = "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"
ADAPTER_VERSION = "gliner25-native-exclusive-v3"
MODEL_FILES = {
    "model.safetensors": "40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997",
    "config.json": "e748e5b80575471c91b3f0dd00f513ba58242544fcb7e1236e0021e61abd7673",
    "tokenizer.json": "3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8",
    "tokenizer_config.json": "323199a4e946039410899f3779f2aa3eaef1500213c512727ad0f623d4f21309",
    "special_tokens_map.json": "84ea70143f533d7e99b393d87f20010887a9ac2cba955828ef313886e4e83f4f",
}
_FORBIDDEN = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)


class NativeContextOverflow(ValueError):
    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(
            f"Native GLiNER input needs {tokens} tokens; encoder limit is {limit}"
        )
        self.tokens = tokens
        self.limit = limit


def verify_release(path: Path, revision: str) -> dict[str, Any]:
    if revision != REVISION or not local_revision(path, revision):
        raise ValueError("GLiNER2.5-Decide requires its attested pinned revision")
    for name, expected in MODEL_FILES.items():
        candidate = path / name
        if not candidate.is_file() or file_digest(candidate) != expected:
            raise ValueError(f"GLiNER model file mismatch: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("architecture") != "span" or config.get("architectures") != [
        "SpanExtractor"
    ]:
        raise ValueError("Unexpected GLiNER source architecture")
    return {
        "model_weights_sha256": MODEL_FILES["model.safetensors"],
        "model_config_sha256": MODEL_FILES["config.json"],
        "tokenizer_sha256": MODEL_FILES["tokenizer.json"],
    }


def _clean_alias(key: str, index: int, used: set[str]) -> str:
    """Keep native semantic label names when GLiNER's schema accepts them."""
    if (
        not key.strip()
        or len(key) > 64
        or any(token in key for token in _FORBIDDEN)
        or key in used
    ):
        alias = f"candidate_{index}"
    else:
        alias = key
    suffix = 0
    while alias in used:
        alias = f"candidate_{index}_{suffix}"
        suffix += 1
    used.add(alias)
    return alias


def _schema_safe(value: str) -> str:
    """Keep text semantics while escaping syntax reserved by GLiNER's schema."""
    safe = value.replace("(", ": ").replace(")", "")
    for marker in _FORBIDDEN[:-2]:
        safe = safe.replace(marker, marker[1:-1])
    return safe.strip()


def prepare_question(
    state: Any, question: dict[str, Any]
) -> tuple[str, list[tuple[str, str]]]:
    """Return native document text and ``(schema alias, benchmark key)`` pairs."""
    kind = question.get("type")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("Choice requires at least two options")
        options = [
            (str(key), str(description)) for key, description in criteria.items()
        ]
    elif kind == "noul":
        options = [
            ("yes", "The decision instruction is true"),
            ("no", "The decision instruction is false"),
        ]
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score requires 2-10 ordered levels")
        options = [(str(i), str(description)) for i, description in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind!r}")
    instruction = question.get("instructions")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Decision instruction must be nonempty text")
    state_text = (
        state
        if isinstance(state, str)
        else json.dumps(
            state, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
    )
    if not isinstance(state_text, str):
        raise ValueError("Invalid state")
    used: set[str] = set()
    aliases = [(_clean_alias(key, i, used), key) for i, (key, _) in enumerate(options)]
    remapped = [
        f"{alias} means original key {key}" for alias, key in aliases if alias != key
    ]
    if remapped:
        state_text += "\n" + "\n".join(remapped)
    return state_text, aliases


def schema_labels(
    question: dict[str, Any], aliases: list[tuple[str, str]]
) -> list[str] | dict[str, str]:
    if question["type"] == "noul":
        return [alias for alias, _ in aliases]
    descriptions = (
        list(question["criteria"].values())
        if question["type"] == "choice"
        else question["criteria"]
    )
    return {
        alias: _schema_safe(str(description)) or f"Option {key}"
        for (alias, key), description in zip(aliases, descriptions)
    }


def project(
    question: dict[str, Any],
    aliases: list[tuple[str, str]],
    probabilities: dict[str, float],
) -> dict[str, Any]:
    if set(probabilities) != {alias for alias, _ in aliases}:
        raise ValueError("Native GLiNER scorer did not return every option label")
    values = [float(probabilities[alias]) for alias, _ in aliases]
    if (
        any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
        or abs(sum(values) - 1.0) > 1e-4
    ):
        raise ValueError("Native exclusive GLiNER probabilities are malformed")
    best = max(range(len(values)), key=values.__getitem__)
    kind = question["type"]
    if kind == "choice":
        return {
            "type": kind,
            "choice": aliases[best][1],
            "probabilities": {key: probabilities[alias] for alias, key in aliases},
            "confidence": values[best],
        }
    if kind == "noul":
        yes = next(alias for alias, key in aliases if key == "yes")
        return {"type": kind, "noul": probabilities[yes], "confidence": values[best]}
    return {
        "type": kind,
        "score": sum(i * value for i, value in enumerate(values)),
        "probabilities": {str(i): value for i, value in enumerate(values)},
        "confidence": values[best],
        "native_level": best,
        "score_projection": "native_ordinal_expected_from_class_probabilities",
    }


def load_native(path: Path, device: str):
    if os.environ.get("GLINER2_SOURCE_COMMIT") != LIBRARY_COMMIT:
        raise RuntimeError("GLiNER2 library source commit is not pinned by runtime")
    import gliner2
    import torch
    from gliner2.classification.engine import Classifier

    if gliner2.__version__ != "2.0.0":
        raise RuntimeError("Unexpected GLiNER2 library version")
    return (
        Classifier.from_pretrained(str(path), device=device, dtype=torch.float32)
        .to(device)
        .eval()
    )


def score_question(native: Any, state: Any, question: dict[str, Any]) -> dict[str, Any]:
    from gliner2.classification.schema import ClassificationSchema

    text, aliases = prepare_question(state, question)
    schema = ClassificationSchema().single(
        "decision",
        schema_labels(question, aliases),
        instruction=_schema_safe(question["instructions"]),
    )
    compiled = native.compile_schema(schema)
    native_tokens = len(
        native.model.processor.transform_record(text, compiled.build()).input_ids
    )
    max_positions = int(native.model.encoder.config.max_position_embeddings)
    if native_tokens > max_positions:
        raise NativeContextOverflow(native_tokens, max_positions)
    scores = native.score(text, compiled)
    returned = scores.tasks.get("decision")
    if returned is None or set(returned) != {alias for alias, _ in aliases}:
        raise ValueError("Native GLiNER schema labels were dropped or altered")
    probabilities = {
        alias: scores.probability("decision", alias) for alias, _ in aliases
    }
    return project(question, aliases, probabilities)


def _completed(
    path: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(f"{path}:{line_number}: unknown or duplicate ID")
            if any(item.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    f"{path}:{line_number}: stale model or adapter identity"
                )
            if (
                item.get("source_input_sha256") != expected[item_id][0]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id][1]
            ):
                raise ValueError(f"{path}:{line_number}: stale or incomplete answers")
            seen.add(item_id)
    return seen


def collect(
    *,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    model_path = model_path.resolve(strict=True)
    release = verify_release(model_path, revision)
    rows = load_prompts(prompts)
    identity = {
        "backend": "gliner25",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "library_commit": LIBRARY_COMMIT,
        "adapter_version": ADAPTER_VERSION,
        "prompt_projection": "state-text; instruction-and-criteria-native-schema",
        **release,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        seen = _completed(output, rows, identity)
    else:
        seen = set()
    pending = [row for row in rows if row["id"] not in seen]
    if max_items is not None:
        pending = pending[:max_items]
    native = load_native(model_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as stream:
        for row in pending:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            answers = {}
            invalid_reason = None
            for name, question in row["questions"].items():
                try:
                    answers[name] = score_question(native, row["state"], question)
                except NativeContextOverflow as exc:
                    answers[name] = {
                        "type": question["type"],
                        "error": "context_overflow",
                        "native_input_tokens": exc.tokens,
                        "native_max_positions": exc.limit,
                    }
                    invalid_reason = "context_overflow"
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency_ms):
                raise ValueError("Nonfinite GLiNER request latency")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": latency_ms,
                "usage": None,
                "source_input_sha256": digest(payload),
                "model": f"{MODEL_ID}@{revision}",
                "runtime_qualification": "pytorch_fp32_rocm_unvalidated",
                "context_policy": "native-tokenizer-default",
                "invalid_reason": invalid_reason,
                **identity,
            }
            stream.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    return {
        "input_items": len(rows),
        "previously_completed": len(seen),
        "collected_now": len(pending),
        "output": str(output),
        **identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
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
                resume=args.resume,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
