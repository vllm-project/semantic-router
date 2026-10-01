"""Native reference predictions for a Kai-architecture fine-tune without a scored run.

Decision-1.0-Route-0.6B is a fine-tune of Kai with Kai's model-only layout, so its
native reference is the published Kai runtime (revision ``7185f514``:
``native/*.py``, ``decision_runtime``, ``decision_inference``) run on the
fine-tune's weights exactly as Kai's ``systemone.py`` runs it (two CPU threads,
no TF32, no MHA fast path, default B8 typed scheduling, 1,024-token complete
inputs). A native export directory is assembled from the runtime files and the
fine-tune's ``native/`` files, with a fresh ``MANIFEST.json``; the runtime then
verifies every file before loading. Rows follow ``inference.kai_lex``.

Run with Transformers 4.57.6 (the runtime pins it), on one ROCm GPU or the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

DECISION2 = Path(__file__).resolve().parents[3]
NATIVE_WEIGHT_FILES = (
    "decision_config.json",
    "INVENTORY.json",
    "STATE_LAYOUT.json",
    "encoder/config.json",
    "encoder/model.safetensors",
    "choice_encoder.safetensors",
    "score_encoder.safetensors",
    "decision_heads.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(kai_code: Path, weights: Path, target: Path) -> str:
    """Runtime files from Kai's revision plus the fine-tune's native files; returns the manifest SHA-256."""
    sys.path.insert(0, str(kai_code))
    from decision_runtime._compat import RUNTIME_SHA256

    for relative in RUNTIME_SHA256:
        source = kai_code / "native" / relative
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target / relative)
        if sha256(target / relative) != RUNTIME_SHA256[relative]:
            raise ValueError(
                f"Kai runtime file differs from its pinned hash: {relative}"
            )
    for relative in NATIVE_WEIGHT_FILES:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            (weights / "native" / relative).resolve(strict=True), target / relative
        )
    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    manifest = target / "MANIFEST.json"
    manifest.write_text(
        json.dumps({"schema": "decision.files.v1", "files": files}, indent=2) + "\n"
    )
    return sha256(manifest)


def load_native(directory: Path, manifest_sha256: str, device: str):
    """``decision_runtime.load_native``; on the CPU without its ROCm-only device assertion."""
    from decision_runtime import native as runtime

    if device != "cpu":
        return runtime.load_native(
            directory, expected_manifest_sha256=manifest_sha256, device=device
        )
    root, manifest = runtime.verify_files(directory, manifest_sha256)
    namespace = "_decision_native_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        namespace, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    spec.loader.exec_module(package)
    artifacts = importlib.import_module(namespace + ".artifacts")
    contract = importlib.import_module(namespace + ".contract")
    api = importlib.import_module(namespace + ".model")
    model, collator, cfg = artifacts.load_export(root, device=device)
    if type(model) is not api.DecisionModel or artifacts.verify_native(root) != (
        manifest,
        cfg,
    ):
        raise ValueError("Loaded class or native identity mismatch")
    return runtime.Native(
        root,
        manifest_sha256,
        model,
        collator,
        cfg,
        api,
        contract,
        artifacts,
        package_name=namespace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kai-code", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, required=True, help="fine-tune repository snapshot"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--panel", action="append", required=True, help="NAME:PROMPTS")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(DECISION2))
    from inference.kai_lex import is_context_overflow
    from inference.run import digest, load_prompts

    native_dir = args.work / "native"
    native_dir.mkdir(parents=True, exist_ok=False)
    manifest_sha256 = assemble(args.kai_code, args.weights, native_dir)

    import torch

    if args.device != "cpu":
        torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    from decision_inference import SystemOne

    native = load_native(native_dir, manifest_sha256, args.device)
    client = SystemOne(native, model=args.model_name, batching="default")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "native_manifest_sha256": manifest_sha256,
        "device": args.device,
        "panels": {},
    }
    for spec in args.panel:
        name, prompts_path = spec.split(":")
        overflow = 0
        with (args.output_dir / f"{name}.predictions.jsonl").open(
            "x", encoding="utf-8"
        ) as target:
            for row in load_prompts(Path(prompts_path)):
                payload = {"state": row["state"], "questions": row["questions"]}
                try:
                    response = client.system_one(**payload)
                    reason = None
                except ValueError as error:
                    if not is_context_overflow(error):
                        raise
                    overflow += 1
                    reason = "context_overflow"
                    response = {
                        "answers": {
                            qid: {"type": question["type"], "error": reason}
                            for qid, question in row["questions"].items()
                        },
                        "usage": None,
                    }
                record = {
                    "id": row["id"],
                    "answers": response["answers"],
                    "usage": response.get("usage"),
                    "model": args.model_name,
                    "source_input_sha256": digest(payload),
                    "invalid_reason": reason,
                }
                target.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        summary["panels"][name] = {"overflow_rows": overflow}
    (args.output_dir / "REFERENCE.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
