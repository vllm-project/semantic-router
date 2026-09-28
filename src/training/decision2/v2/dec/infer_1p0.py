"""Gold-blind panel inference for an untouched Decision 1.0 package.

Loads the package through the shared ``from_decision1`` path (the same
rendering and head the decoder-track students train with) and applies the
package's own per-type temperatures when it ships them. Used for same-runtime
1.0 control readouts and for teacher-fidelity checks; not a training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.model.data import canonical, file_sha256
from training.model.infer import load_prompts, run_prompts, write_output
from training.model.source import source_fingerprint

ADAPTER_VERSION = "dec-decision1-package-shared-renderer/1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--package-temperatures", action="store_true")
    args = parser.parse_args()
    rows = load_prompts(args.input)

    import torch

    from training.model.decision_model import DecisionModel, collate, encode

    temperature: float | dict[str, float] = 1.0
    if args.package_temperatures:
        report = json.loads((args.package / "temperature.json").read_text())
        temperature = {k: float(v) for k, v in report["temperatures"].items()}
    source = source_fingerprint(args.package)
    model_sha = hashlib.sha256(
        canonical(source["files_sha256"]).encode("utf-8")
    ).hexdigest()
    here = Path(__file__).resolve()
    shared = here.parents[2] / "training" / "model"
    sources = {
        **{
            f"training/model/{n}": file_sha256(shared / n)
            for n in ("infer.py", "decision_model.py", "data.py", "source.py")
        },
        "v2/dec/infer_1p0.py": file_sha256(here),
    }
    adapter_sha = hashlib.sha256(canonical(sources).encode("utf-8")).hexdigest()
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_decision1(args.package, 256)
    model = model.float().to(device).eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in collate(encoded, pad_id).items()
        }
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            logits = model(**batch)
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    predictions, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        temperature=temperature,
        encode_fn=encode,
        predict_fn=predict,
        model_sha256=model_sha,
        adapter_sha256=adapter_sha,
    )
    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_sha256": model_sha,
        "model_files_sha256": source["files_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_files_sha256": sources,
        "input_sha256": file_sha256(args.input),
        "input_items": len(rows),
        "max_length": args.max_length,
        "temperature": temperature,
        "execution": "one benchmark item at a time; BF16 backbone, FP32 head",
        "truncation_policy": "none; over-budget questions produce an invalid answer",
        "counts": counts,
        "torch_version": torch.__version__,
    }
    write_output(args.output, predictions, manifest)
    print(json.dumps({"output": str(args.output), "counts": counts}), flush=True)


if __name__ == "__main__":
    main()
