"""Gold-free predictions from a verified existing Kai checkpoint export.

Research only: this never presents an alternate checkpoint as trainer-selected
or release-qualified. Its exact input, model, and ancestry hashes travel with
each prediction so the ordinary development scorers can verify alignment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from inference.kai_lex import MODELS, runtime_report, verify_native_bundle
from inference.run import digest, file_digest, load_prompts, synchronize

MODEL_ID = "llm-semantic-router/dev-2.0-0.6b-research"
MODEL_NAME = "dev-2.0-0.6b"


def collect(
    *,
    base: Path,
    run: Path,
    native: Path,
    manifest_sha: str,
    checkpoint_sha: str,
    step: int,
    prompts: Path,
    output: Path,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    base, run, native = (path.resolve(strict=True) for path in (base, run, native))
    if not native.is_relative_to(run):
        raise ValueError("Checkpoint export must belong to the source run")
    receipt_path = native.with_suffix(".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    complete_path, run_path = run / "COMPLETE.json", run / "RUN.json"
    complete, run_receipt = json.loads(complete_path.read_text()), json.loads(
        run_path.read_text()
    )
    if (
        receipt.get("schema_version") != "kai06b-existing-checkpoint-export/1"
        or receipt.get("research_only") is not True
        or receipt.get("release_qualified") is not False
        or receipt.get("export_path") != str(native)
        or receipt.get("export_manifest_sha256") != manifest_sha
        or file_digest(native / "MANIFEST.json") != manifest_sha
        or receipt.get("checkpoint_step") != step
        or receipt.get("checkpoint_manifest_sha256") != checkpoint_sha
        or receipt.get("source_run_complete_sha256") != file_digest(complete_path)
        or receipt.get("source_run_json_sha256") != file_digest(run_path)
        or receipt.get("source_native_manifest_sha256")
        != complete["identity"]["native_manifest_sha256"]
        or complete["identity"] != run_receipt["identity"]
    ):
        raise ValueError("Research checkpoint/export receipt differs")
    verify_native_bundle(base, "kai", MODELS["kai"]["revision"])
    sys.path.insert(0, str(base))
    from decision_finetune import state

    checkpoint = run / "attempts" / "0001" / f"checkpoint-{step:06d}"
    _, metadata = state.verify_checkpoint(checkpoint, checkpoint_sha)
    if metadata.get("identity") != complete["identity"] or metadata.get("step") != step:
        raise ValueError("Checkpoint training identity differs")
    runtime = runtime_report("cuda:0", check_gpu=True)
    if not runtime["runtime_matches_validated"]:
        raise RuntimeError("Kai runtime differs from qualified source")
    import torch
    from decision_inference import SystemOne
    from decision_runtime import load_native

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    rows = load_prompts(prompts)
    model = load_native(native, expected_manifest_sha256=manifest_sha, device="cuda:0")
    client = SystemOne(model, model=MODEL_NAME, batching="default")
    output.parent.mkdir(parents=True, exist_ok=True)
    overflow = 0
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize("cuda:0")
            start = time.perf_counter()
            try:
                response = client.system_one(**payload)
                invalid = None
            except ValueError as error:
                if "exceeds 1024 tokens; no implicit truncation" not in str(error):
                    raise
                invalid = "context_overflow"
                overflow += 1
                response = {
                    "model": MODEL_NAME,
                    "usage": None,
                    "answers": {
                        qid: {"type": q["type"], "error": invalid}
                        for qid, q in row["questions"].items()
                    },
                }
            synchronize("cuda:0")
            latency = (time.perf_counter() - start) * 1000
            if (
                set(response["answers"]) != set(row["questions"])
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("Malformed checkpoint result")
            item = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency,
                "usage": response.get("usage"),
                "model": MODEL_NAME,
                "backend": "kai-reselection-research",
                "model_id": MODEL_ID,
                "model_revision": manifest_sha,
                "model_config_sha256": manifest_sha,
                "source_checkpoint_sha256": checkpoint_sha,
                "source_run_complete_sha256": file_digest(complete_path),
                "ablation_receipt_sha256": file_digest(receipt_path),
                "source_input_sha256": digest(payload),
                "invalid_reason": invalid,
                "research_only": True,
                "release_qualified": False,
            }
            stream.write(
                json.dumps(
                    item, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    return {
        "rows": len(rows),
        "overflow": overflow,
        "output": str(output),
        "output_sha256": file_digest(output),
        "export_manifest_sha256": manifest_sha,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "ablation_receipt_sha256": file_digest(receipt_path),
        "research_only": True,
        **runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                base=args.base_model,
                run=args.run,
                native=args.native,
                manifest_sha=args.manifest_sha256,
                checkpoint_sha=args.checkpoint_sha256,
                step=args.step,
                prompts=args.input,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
