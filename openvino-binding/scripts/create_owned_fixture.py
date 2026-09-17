#!/usr/bin/env python3
"""Generate tiny deterministic OpenVINO models without downloading weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops
from openvino_tokenizers import convert_tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def make_tokenizer(offset: int) -> PreTrainedTokenizerFast:
    vocab = {f"unused{i}": i for i in range(32)}
    for word, index in {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "alpha": offset,
        "beta": offset + 1,
        "gamma": offset + 2,
        "delta": offset + 3,
    }.items():
        del vocab[f"unused{index}"]
        vocab[word] = index
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    # OpenVINO Tokenizers 2025.1 adds a ragged-token truncation step whenever
    # model_max_length is not None, even if tokenizer.json has truncation=null.
    tokenizer.model_max_length = None
    tokenizer.init_kwargs["model_max_length"] = None
    tokenizer.backend_tokenizer.no_truncation()
    return tokenizer


def inspect_tokenizer(tokenizer_ir: ov.Model, offset: int) -> dict:
    operations = [node.get_type_name() for node in tokenizer_ir.get_ordered_ops()]
    # This tiny WordPiece graph needs none of these operations. Minimum is the
    # pinned exporter's ragged-token truncation operation, not a Slice.
    assert not {"Minimum", "Slice", "StridedSlice"}.intersection(operations)
    compiled = ov.Core().compile_model(tokenizer_ir, "CPU")
    text = " ".join(["alpha"] * 96)
    result = compiled([np.array([text])])
    ids = result[compiled.output("input_ids")][0].tolist()
    assert ids == [1] + [offset] * 96 + [2], ids
    return {
        "operations": operations,
        "probe_tokens": len(ids),
        "declared_model_limit": 64,
    }


def make_model(kind: str, variant: str) -> ov.Model:
    ids = ops.parameter([1, -1], ov.Type.i64, name="input_ids")
    mask = ops.parameter([1, -1], ov.Type.i64, name="attention_mask")
    ids.output(0).set_names({"input_ids"})
    mask.output(0).set_names({"attention_mask"})
    floats = ops.convert(ids, ov.Type.f32)
    weights = ops.convert(mask, ov.Type.f32)
    if kind == "embedding":
        scale = [1, 2, 3] if variant == "a" else [-1, 0.5, 2]
        bias = [0, 1, 2] if variant == "a" else [11, -7, 5]
        output = ops.add(
            ops.multiply(
                ops.unsqueeze(floats, [-1]),
                ops.constant(np.array(scale, dtype=np.float32)),
            ),
            ops.constant(np.array(bias, dtype=np.float32)),
        )
        output = ops.multiply(output, ops.unsqueeze(weights, [-1]))
        output.output(0).set_names({"last_hidden_state"})
    else:
        total = ops.reduce_sum(ops.multiply(floats, weights), [1], keep_dims=True)
        count = ops.reduce_sum(weights, [1], keep_dims=True)
        zero = ops.multiply(total, ops.constant(np.float32(0)))
        one = ops.add(zero, ops.constant(np.float32(1)))
        values = (
            [
                ops.multiply(total, ops.constant(np.float32(0.01))),
                ops.multiply(count, ops.constant(np.float32(0.1))),
                one,
            ]
            if variant == "a"
            else [
                one,
                ops.multiply(total, ops.constant(np.float32(-0.02))),
                ops.multiply(count, ops.constant(np.float32(0.05))),
            ]
        )
        output = ops.concat(values, 1)
        output.output(0).set_names({"logits"})
    return ov.Model([output], [ids, mask], f"owned_{kind}_{variant}")


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"openvino_version": ov.__version__, "fixtures": {}}
    for variant, offset in [("a", 4), ("b", 14)]:
        tokenizer = make_tokenizer(offset)
        expected = [1, offset, offset + 1, offset + 2, 2]
        assert tokenizer.backend_tokenizer.encode("alpha beta gamma").ids == expected
        assert tokenizer.backend_tokenizer.encode("alpha [PAD] beta").ids == [
            1,
            offset,
            0,
            offset + 1,
            2,
        ]
        tokenizer_ir = convert_tokenizer(tokenizer, with_detokenizer=False)
        tokenizer_evidence = inspect_tokenizer(tokenizer_ir, offset)
        for kind in ["embedding", "classifier"]:
            name = f"{kind}_{variant}"
            directory = output / name
            directory.mkdir(exist_ok=True)
            tokenizer.save_pretrained(directory)
            configuration = {
                "model_type": "bert",
                "hidden_size": 3,
                "max_position_embeddings": 64,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "pad_token_id": 0,
                "cls_token_id": 1,
                "sep_token_id": 2,
            }
            if kind == "classifier":
                configuration["id2label"] = {"0": "first", "1": "second", "2": "third"}
            (directory / "config.json").write_text(
                json.dumps(configuration, indent=2) + "\n"
            )
            ov.save_model(
                tokenizer_ir,
                directory / "openvino_tokenizer.xml",
                compress_to_fp16=False,
            )
            ov.save_model(
                make_model(kind, variant),
                directory / "openvino_model.xml",
                compress_to_fp16=False,
            )
            receipt["fixtures"][name] = {
                "plain_token_ids": expected,
                "tokenizer_evidence": tokenizer_evidence,
                "pad_token_id": 0,
                "end_token_ids": [2],
                "sha256": {
                    file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in sorted(directory.iterdir())
                    if file.is_file()
                },
            }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Created four tiny OpenVINO fixtures at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
