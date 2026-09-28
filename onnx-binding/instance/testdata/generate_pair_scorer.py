"""Generate an input-dependent paired ONNX fixture with onnx==1.18.0.

The graph returns a masked sum, so pair token IDs and full-input coverage have
exact expectations. It verifies the actual native ABI, not model quality.
"""

import argparse
import base64
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def artifacts():
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "sequence"])
        for name in ("input_ids", "attention_mask")
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
            helper.make_node("Cast", ["masked"], ["values"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["values"], ["logits"], axes=[1], keepdims=1),
        ],
        "owned_pair_sum",
        inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8
    )
    model.metadata_props.add(
        key="semantic_router.pair_scorer",
        value=json.dumps(
            {"version": 1, "layer": 2, "dimension": 4, "score_type": "relevance_logit"}
        ),
    )
    onnx.checker.check_model(model)
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "decoder": None,
        "post_processor": {
            "type": "BertProcessing",
            "sep": ["[SEP]", 4],
            "cls": ["[UNK]", 0],
        },
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "秘密": 3, "[SEP]": 4},
            "unk_token": "[UNK]",
        },
    }
    config = {
        "architectures": ["ModernBertModel"],
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 32768,
        "pad_token_id": 0,
        "representation_contract": {
            "version": 1,
            "pooling": "cls",
            "intermediate_normalization": "final_norm",
            "final_normalization": "final_norm",
            "head_dtype": "float32",
        },
    }
    layout = {
        "layer_indices": [1, 2],
        "dim_indices": [2, 4],
        "hidden_size": 4,
        "num_layers": 2,
        "pooling_strategy": "cls",
        "has_final_norm": True,
    }
    return {
        "model.onnx": model.SerializeToString(),
        **{
            name: (json.dumps(value, indent=2) + "\n").encode()
            for name, value in (
                ("tokenizer.json", tokenizer),
                ("config.json", config),
                ("matryoshka_config.json", layout),
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    files = artifacts()
    if args.stdout:
        print(
            json.dumps(
                {name: base64.b64encode(data).decode() for name, data in files.items()}
            )
        )
    else:
        destination = Path(__file__).parent / "pair_scorer"
        destination.mkdir(exist_ok=True)
        for name, data in files.items():
            (destination / name).write_bytes(data)
