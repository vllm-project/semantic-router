#!/usr/bin/env python3
"""Export Vela Halu through the maintained classifier exporter and freeze pair references.

The output is a local derived artifact. It retains the published 8192-token task
policy and supplies UTF-8 byte spans for router parity; exporting does not claim
AMD qualification. Use an immutable local checkpoint with the pinned revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "llm-semantic-router/Vela-1.0-Encoder-307M-Halu"
REVISION = "ca87531211e414ac21c641b2faa8b8e21619de8f"
POLICY = {
    "max_input_tokens": 8192,
    "token_threshold": 0.5,
    "threshold_comparison": "strictly_greater",
    "label2id": {"supported": 0, "hallucinated": 1},
    "input_pair": ["User request: {question}\n\n{context}", "answer"],
    "answer_offsets": "Unicode code points",
}
PROBES = [
    {
        "name": "supported",
        "context": "The museum opens at 10:00 on Tuesday.",
        "question": "When does the museum open on Tuesday?",
        "answer": "The museum opens at 10:00 on Tuesday.",
    },
    {
        "name": "unsupported_time",
        "context": "The museum opens at 10:00 on Tuesday.",
        "question": "When does the museum open on Tuesday?",
        "answer": "The museum opens at 09:00 on Tuesday.",
    },
    {
        "name": "unicode",
        "context": "Élodie lives in Paris. 王明住在北京。",
        "question": "Where do Élodie and 王明 live?",
        "answer": "Élodie lives in Berlin. 王明住在上海。",
    },
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_source(directory: Path) -> dict:
    source = json.loads(Path(__file__).with_name("source.json").read_text())
    for name, expected in source["files"].items():
        path = directory / name
        if expected["algorithm"] == "sha256":
            actual = digest(path)
        else:
            data = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected["digest"]:
            raise ValueError(f"Source file does not match pinned Halu revision: {name}")
    policy = json.loads((directory / "operating_point.json").read_text())
    if any(policy.get(key) != value for key, value in POLICY.items()):
        raise ValueError("Unsupported Halu operating point")
    config = json.loads((directory / "config.json").read_text())
    if config.get("id2label") != {"0": "supported", "1": "hallucinated"}:
        raise ValueError("Halu requires supported=0, hallucinated=1 labels")
    return policy


def answer_spans(answer, sequences, offsets, scores):
    spans, current = [], None
    for sequence, (start, end), score in zip(sequences, offsets, scores, strict=True):
        if sequence != 1 or end <= start:
            continue
        if score > POLICY["token_threshold"]:
            if current is None:
                current = {"start": start, "end": end, "confidence": score}
            else:
                current["end"] = max(current["end"], end)
                current["confidence"] = max(current["confidence"], score)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    for span in spans:
        start, end = span["start"], span["end"]
        span["text"] = answer[start:end]
        span["start"], span["end"] = (
            len(answer[:start].encode()),
            len(answer[:end].encode()),
        )
    return spans


def references(source: Path, device: str) -> list:
    # Offline contract tests import this module without model execution dependencies.
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.reference_compile = False
    model = (
        AutoModelForTokenClassification.from_pretrained(
            source,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    probes = [
        *PROBES,
        {
            **PROBES[1],
            "name": "long_evidence",
            "context": "The library is quiet. " * 150 + PROBES[1]["context"],
        },
    ]
    results = []
    for probe in probes:
        encoded = tokenizer(
            f"User request: {probe['question']}\n\n{probe['context']}",
            probe["answer"],
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if encoded.input_ids.shape[1] > POLICY["max_input_tokens"]:
            raise ValueError("Reference exceeds the published task limit")
        sequences = encoded.sequence_ids(0)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            scores = (
                model(**{key: value.to(device) for key, value in encoded.items()})
                .logits.float()
                .softmax(-1)[0, :, 1]
                .tolist()
            )
        results.append(
            {
                **probe,
                "input_tokens": encoded.input_ids.shape[1],
                "spans": answer_spans(probe["answer"], sequences, offsets, scores),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    policy = validate_source(args.model)
    if args.model.resolve() == args.output.resolve():
        raise ValueError("Keep the immutable source separate from derived artifacts")
    if not args.reference_only:
        exporter = (
            Path(__file__).resolve().parents[3]
            / "onnx-binding/scripts/export_classifier.py"
        )
        command = [
            sys.executable,
            str(exporter),
            "--model",
            str(args.model),
            "--output",
            str(args.output),
            "--dtype",
            args.dtype,
            "--device",
            args.device,
        ]
        if args.device == "cuda":
            command.append("--export-only")
        subprocess.run(command, check=True)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        args.model / "operating_point.json", args.output / "operating_point.json"
    )
    receipt = {
        "schema": "vela_halu_reference.v1",
        "source": {"repo_id": MODEL_ID, "revision": REVISION},
        "policy": policy,
        "offset_unit": "utf8_bytes",
        "reference_device": args.device,
        "source_files": {
            name: digest(args.model / name)
            for name in (
                "config.json",
                "tokenizer.json",
                "model.safetensors",
                "operating_point.json",
            )
        },
        "probes": references(args.model, args.device),
    }
    (args.output / "halu_reference.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
