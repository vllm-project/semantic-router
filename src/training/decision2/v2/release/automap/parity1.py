"""Decision 1.0 remote code vs the stored native predictions of the scored panels.

Loads a staged repository directory exactly as a user would
(``AutoModel.from_pretrained(dir, trust_remote_code=True)``), answers every
gold-free prompt with ``system_one`` and compares each answer with the sealed
native prediction using the release track's ``compare_answers`` (category
changes, missing answers, maximum absolute probability / score drift). A native
over-budget row (all answers null) must be rejected by the remote code too.

Receipts hold hashes, counts and drift only, never panel text; changed prompt
IDs go to a separate private file when ``--changes`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "dev1-automap-parity/1"
HERE = Path(__file__).resolve().parent


def _examples():
    spec = importlib.util.spec_from_file_location(
        "_release_examples", HERE.parent / "examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def runtime_versions() -> dict[str, Any]:
    import torch
    import transformers

    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    for name, distribution in (
        ("tokenizers", "tokenizers"),
        ("safetensors", "safetensors"),
        ("huggingface_hub", "huggingface_hub"),
        ("fla", "flash-linear-attention"),
        ("triton", "triton"),
    ):
        loaded = getattr(sys.modules.get(name), "__version__", None)
        try:
            versions[name] = loaded or metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="staged repository directory"
    )
    parser.add_argument(
        "--panel", action="append", required=True, help="NAME:PROMPTS:PREDICTIONS[:N]"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--changes", type=Path, help="private JSONL of changed prompt IDs"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    if args.threads:
        torch.set_num_threads(args.threads)
    examples = _examples()
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(args.model), trust_remote_code=True, device=args.device
    )
    load_seconds = time.perf_counter() - started
    loaded = sum(parameter.numel() for parameter in model.parameters())
    panels, changes = {}, []
    for spec in args.panel:
        name, prompts_path, predictions_path, *limit = spec.split(":")
        prompts = read_jsonl(Path(prompts_path))
        if limit:
            prompts = prompts[: int(limit[0])]
        sealed = {row["id"]: row for row in read_jsonl(Path(predictions_path))}
        totals = {
            "prompts": 0,
            "slots": 0,
            "category_changes": 0,
            "missing": 0,
            "max_abs_drift": 0.0,
            "input_mismatch": 0,
            "native_over_budget": 0,
            "over_budget_agree": 0,
            "input_tokens_mismatch": 0,
        }
        drift_by_type: dict[str, float] = {}
        panel_started = time.perf_counter()
        for prompt in prompts:
            payload = {"state": prompt["state"], "questions": prompt["questions"]}
            reference = sealed.get(prompt["id"]) or {}
            if reference.get("source_input_sha256") != payload_sha(payload):
                totals["input_mismatch"] += 1
            native = reference.get("answers") or {}
            native_over = bool(native) and all(
                value is None or (isinstance(value, dict) and "error" in value)
                for value in native.values()
            )
            totals["native_over_budget"] += native_over
            response = model.system_one(**payload)
            answers = response["answers"]
            ours_over = all(
                isinstance(value, dict) and value.get("error") == "max_length_exceeded"
                for value in answers.values()
            )
            tokens = None if ours_over else response["usage"]["input_tokens"]
            if native_over or ours_over:
                totals["prompts"] += 1
                totals["slots"] += len(prompt["questions"])
                if native_over and ours_over:
                    totals["over_budget_agree"] += 1
                else:
                    totals["category_changes"] += len(prompt["questions"])
                    changes.append(
                        {
                            "panel": name,
                            "id": prompt["id"],
                            "over_budget": [native_over, ours_over],
                        }
                    )
                continue
            result = examples.compare_answers(answers, native)
            usage = reference.get("usage") or {}
            if tokens is not None and usage.get("input_tokens") not in (None, tokens):
                totals["input_tokens_mismatch"] += 1
            totals["prompts"] += 1
            for key in ("slots", "category_changes", "missing"):
                totals[key] += result[key]
            totals["max_abs_drift"] = max(
                totals["max_abs_drift"], result["max_abs_drift"]
            )
            for qid, question in prompt["questions"].items():
                if answers.get(qid) is not None and native.get(qid) is not None:
                    single = examples.compare_answers(
                        {qid: answers[qid]}, {qid: native[qid]}
                    )
                    kind = question.get("type")
                    drift_by_type[kind] = max(
                        drift_by_type.get(kind, 0.0), single["max_abs_drift"]
                    )
            if result["category_changes"] or result["missing"]:
                changes.append({"panel": name, "id": prompt["id"], **result})
        panels[name] = {
            **totals,
            "max_abs_drift_by_type": drift_by_type,
            "prompts_sha256": sha_file(Path(prompts_path)),
            "predictions_sha256": sha_file(Path(predictions_path)),
            "ids_sha256": hashlib.sha256(
                json.dumps([p["id"] for p in prompts]).encode("utf-8")
            ).hexdigest(),
            "seconds": time.perf_counter() - panel_started,
        }
    passed = all(
        p["category_changes"] == 0
        and p["missing"] == 0
        and p["input_mismatch"] == 0
        and p["over_budget_agree"] == p["native_over_budget"]
        for p in panels.values()
    )
    receipt = {
        "schema": SCHEMA,
        "model_dir_config_sha256": sha_file(args.model / "config.json"),
        "remote_code_sha256": {
            path.name: sha_file(path) for path in sorted(args.model.glob("*.py"))
        },
        "model_name": model.config.model_name,
        "loaded_parameters": loaded,
        "device": str(next(model.parameters()).device),
        "threads": torch.get_num_threads(),
        "runtime": runtime_versions(),
        "environment": {
            key: os.environ.get(key)
            for key in ("FLA_CACHE_MODE", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
            if os.environ.get(key)
        },
        "load_seconds": load_seconds,
        "panels": panels,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.changes is not None:
        args.changes.write_text(
            "".join(json.dumps(c) + "\n" for c in changes), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                name: {
                    k: p[k]
                    for k in (
                        "prompts",
                        "category_changes",
                        "missing",
                        "max_abs_drift",
                        "input_mismatch",
                    )
                }
                for name, p in panels.items()
            }
        )
    )
    print("PASSED" if passed else "FAILED")


if __name__ == "__main__":
    main()
