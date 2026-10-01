"""9B M9 retention ceiling: Qwen3.5-9B-Base read on a panel through its own (untied) LM head.

Amendment 1 of the M9 preregistration. The preregistered B0 (a label-token zero-step checkpoint) cannot be built at
9B: ``v2.dec.label_token`` requires a tied LM head and Qwen3.5-9B-Base's head is untied. This diagnostic reads the
same label-token prompt (``decision2-label-token-v1``, ``encode_label``) with option logit i = h_last ·
W[label_i] in FP32, where W is the base's own ``lm_head.weight``; nothing is trained or saved. The predictions use the
shared ``run_prompts`` / ``write_output`` format, so ``m10_probes.py score`` reads them unchanged. Diagnostic only;
never a release score.

usage: python3 -m lux9b.m9_base_probes --base DIR --revision REV --input PROMPTS --output PRED [--max-length N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from training.model.data import canonical, file_sha256
from training.model.decision_model import DecisionModel, collate
from training.model.infer import (
    ADAPTER_VERSION,
    load_prompts,
    run_prompts,
    write_output,
)
from training.model.source import source_fingerprint
from v2.dec.label_token import LABEL_PROMPT_VERSION, LabelTokenModel, encode_label
from v2.dec.runtime_check import require_runtime

READER = "lux9b-m9-untied-label-token/1"


class UntiedLabelReader(LabelTokenModel):
    """The label-token readout with the base's untied LM head instead of the input embeddings."""

    def __init__(self, backbone: torch.nn.Module, lm_head: torch.Tensor):
        super().__init__(backbone, {"readout": "label_token", "reader": READER})
        self.register_buffer("untied_lm_head", lm_head, persistent=False)

    def lm_head_weight(self) -> torch.Tensor:
        return self.untied_lm_head


def load_lm_head(base: Path) -> torch.Tensor:
    index = json.loads((base / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(str(base / index["lm_head.weight"]), framework="pt") as stream:
        return stream.get_tensor("lm_head.weight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_prompts(args.input)
    runtime = require_runtime()
    source = source_fingerprint(args.base)
    model_sha = hashlib.sha256(
        canonical(
            {"source": source, "revision": args.revision, "reader": READER}
        ).encode("utf-8")
    ).hexdigest()
    sources = {
        name: file_sha256(Path(__file__).resolve().parents[2] / name)
        for name in ("9b/lux9b/m9_base_probes.py", "dec/label_token.py")
    }
    adapter_sha = hashlib.sha256(canonical(sources).encode("utf-8")).hexdigest()
    device = torch.device("cuda:0")
    base, tokenizer = DecisionModel.from_base(args.base, args.revision, 256)
    reader = (
        UntiedLabelReader(base.backbone, load_lm_head(args.base))
        .float()
        .to(device)
        .eval()
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            key: (
                value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            )
            for key, value in collate(encoded, pad_id).items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = reader(**batch)
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    predictions, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        temperature=1.0,
        encode_fn=encode_label,
        predict_fn=predict,
        model_sha256=model_sha,
        adapter_sha256=adapter_sha,
        calibration_sha256=None,
    )
    manifest = {
        "adapter_version": ADAPTER_VERSION,
        "dec_loader": "lux9b.m9_base_probes",
        "reader": READER,
        "model_sha256": model_sha,
        "base_revision": args.revision,
        "source_files_sha256": source["files_sha256"],
        "readout": "label_token (untied base LM head)",
        "prompt_version": LABEL_PROMPT_VERSION,
        "adapter_sha256": adapter_sha,
        "adapter_files_sha256": sources,
        "input_sha256": file_sha256(args.input),
        "input_items": len(rows),
        "max_length": args.max_length,
        "temperature": 1.0,
        "counts": counts,
        "torch_version": torch.__version__,
        "runtime": runtime,
    }
    write_output(args.output, predictions, manifest)
    print(
        json.dumps(
            {"output": str(args.output), "counts": counts, "model_sha256": model_sha}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
