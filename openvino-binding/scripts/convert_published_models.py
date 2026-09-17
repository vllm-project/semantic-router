#!/usr/bin/env python3
"""Convert the provisioner's pinned local Vela ONNX artifacts to OpenVINO IR."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
from pathlib import Path

REQUIRED_MODELS = {"Domain", "Embedding"}
REQUIRED_INPUTS = {"input_ids", "attention_mask", "position_ids"}


def owned_tokenizer_contract(tokenizer) -> dict:
    """Export original token counts; the owned handle enforces each call budget."""
    pad = tokenizer.pad_token_id
    suffix = sorted(
        {
            value
            for value in (tokenizer.sep_token_id, tokenizer.eos_token_id)
            if value is not None
        }
    )
    if (
        not isinstance(pad, int)
        or pad < 0
        or not suffix
        or any(not isinstance(value, int) or value < 0 for value in suffix)
    ):
        raise ValueError(
            "published tokenizer must declare padding and suffix token IDs"
        )
    # The pinned OpenVINO Tokenizers exporter inserts hidden truncation when
    # model_max_length is populated, even with tokenizer.json truncation=null.
    tokenizer.model_max_length = None
    tokenizer.init_kwargs["model_max_length"] = None
    tokenizer.backend_tokenizer.no_truncation()
    return {"pad_token_id": pad, "end_token_ids": suffix}


def read_sources(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    models = manifest["models"]
    if (
        manifest["provider"] != "ort"
        or {model["name"] for model in models} != REQUIRED_MODELS
        or len(models) != len(REQUIRED_MODELS)
    ):
        raise ValueError(
            "OpenVINO qualification requires exactly the registered Domain and Embedding ONNX artifacts"
        )
    for model in models:
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"{model['name']}: source revision must be immutable")
    return models


def graph_contract(model, name: str, config: dict) -> dict:
    inputs = []
    for value in model.inputs:
        # OpenVINO may retain an internal attention-mask alias while folding
        # the published graph. Resolve the public name from all tensor names.
        names = value.get_names() & REQUIRED_INPUTS
        if len(names) != 1:
            raise ValueError(f"{name}: unsupported graph inputs: {value.get_names()}")
        public_name = names.pop()
        if value.get_element_type().get_type_name() != "i64":
            raise ValueError(f"{name}: {public_name} must consume int64 tokens")
        value.get_tensor().set_names({public_name})
        inputs.append(public_name)
    if set(inputs) != REQUIRED_INPUTS or len(inputs) != len(REQUIRED_INPUTS):
        raise ValueError(f"{name}: unsupported graph inputs: {inputs}")
    if len(model.outputs) != 1:
        raise ValueError(f"{name}: expected exactly one graph output")
    output = model.output(0)
    if output.get_element_type().get_type_name() != "f32":
        raise ValueError(f"{name}: native binding requires float32 output")
    rank = output.get_partial_shape().rank
    if rank.is_dynamic:
        raise ValueError(f"{name}: output rank must be known")
    expected_ranks = (2,) if name == "Domain" else (2, 3)
    if rank.get_length() not in expected_ranks:
        raise ValueError(f"{name}: unsupported graph output rank {rank}")
    size = output.get_partial_shape()[-1]
    dimension = len(config["id2label"]) if name == "Domain" else config["hidden_size"]
    if size.is_dynamic or size.get_length() != dimension:
        raise ValueError(
            f"{name}: output size does not match source config: {size} != {dimension}"
        )
    if name == "Domain":
        # The C++ classifier consumes logits by their public tensor name.
        output.get_tensor().set_names({"logits"})
    # The native embedding binding supplies inputs in this exact order.
    return {"inputs": inputs, "output_rank": rank.get_length(), "dimension": dimension}


def convert_sources(source_manifest: Path, output_root: Path, receipt: Path) -> None:
    # Source-manifest validation does not need the optional conversion runtime.
    import openvino as ov  # noqa: PLC0415
    from openvino_tokenizers import convert_tokenizer  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    models = read_sources(source_manifest)
    converted = []
    for source in models:
        source_dir = Path(source["path"])
        graph = source_dir / "onnx/model.onnx"
        config = json.loads((source_dir / "config.json").read_text())
        # ONNX is already an inference graph. Read it directly through the
        # runtime frontend instead of retaining framework-export state.
        model = ov.Core().read_model(graph)
        contract = graph_contract(model, source["name"], config)
        if contract["inputs"] != ["input_ids", "attention_mask", "position_ids"]:
            raise ValueError(
                f"{source['name']}: graph inputs must be ordered input_ids, attention_mask, position_ids"
            )
        destination = output_root / source["name"].lower()
        destination.mkdir(parents=True, exist_ok=True)
        # Keep FP32 parameters; qualification must not silently quantize weights.
        ov.save_model(model, destination / "openvino_model.xml", compress_to_fp16=False)
        tokenizer = AutoTokenizer.from_pretrained(
            source_dir, local_files_only=True, trust_remote_code=False
        )
        tokenizer_contract = owned_tokenizer_contract(tokenizer)
        tokenizer.save_pretrained(destination)
        shutil.copy2(source_dir / "config.json", destination / "config.json")
        converted_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)
        ov.save_model(
            converted_tokenizer,
            destination / "openvino_tokenizer.xml",
            compress_to_fp16=False,
        )
        converted.append(
            {
                **source,
                **contract,
                **tokenizer_contract,
                "ir_path": str((destination / "openvino_model.xml").resolve()),
                "config_path": str((destination / "config.json").resolve()),
                "onnx_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
            }
        )
        del model, converted_tokenizer, tokenizer
        gc.collect()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {"provider": "openvino", "version": ov.__version__, "models": converted},
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    convert_sources(args.sources, args.output, args.manifest)


if __name__ == "__main__":
    main()
