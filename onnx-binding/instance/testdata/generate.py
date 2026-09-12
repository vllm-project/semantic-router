"""Generate tiny, input-dependent ONNX fixtures for ownership integration tests.

Run with onnx==1.18.0. These graphs test the real ORT boundary, not model quality;
maintained-checkpoint CPU/AMD regression remains a separate requirement.
"""

import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).parent
VOCAB = {"[UNK]": 0, "hello": 1, "world": 2, "秘密": 3, "test": 4}
TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "[UNK]"},
}


def save(kind, name, nodes, inputs, output_name, output_shape, initializers=()):
    directory = ROOT / kind
    directory.mkdir(exist_ok=True)
    graph = helper.make_graph(
        nodes,
        f"owned_{kind}_{name}",
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
        list(initializers),
    )
    model = helper.make_model(
        graph,
        producer_name="semantic-router-owned-instance-tests",
        opset_imports=[helper.make_opsetid("", 12)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, directory / name)
    (directory / "tokenizer.json").write_text(json.dumps(TOKENIZER, indent=2) + "\n")
    config = {
        "vocab_size": len(VOCAB),
        "hidden_size": 3,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "max_position_embeddings": 4096,
        "num_labels": 2,
        "id2label": {"0": "negative", "1": "positive"},
        "pad_token_id": 0,
    }
    if kind == "token":
        config["id2label"] = {"0": "O", "1": "B-SECRET"}
    if kind == "multimodal":
        config.update(
            embedding_dim=3,
            image_encoder={"image_size": 2},
            audio_encoder={"n_mels": 2},
            text_encoder={"max_seq_len": 512},
        )
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


text_inputs = [
    helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "tokens"])
    for name in ("input_ids", "attention_mask")
]
text_nodes = [
    helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
    helper.make_node("Cast", ["masked"], ["x"], to=TensorProto.FLOAT),
]
for kind in ("sequence", "token"):
    nodes = list(text_nodes)
    if kind == "sequence":
        nodes += [
            helper.make_node("ReduceMean", ["x"], ["score"], axes=[1], keepdims=1)
        ]
        score, axis, shape = "score", 1, ["batch", 2]
    else:
        nodes += [helper.make_node("Unsqueeze", ["x"], ["score"], axes=[2])]
        score, axis, shape = "score", 2, ["batch", "tokens", 2]
    nodes += [
        helper.make_node("Neg", [score], ["negative"]),
        helper.make_node("Concat", ["negative", score], ["logits"], axis=axis),
    ]
    save(kind, "model.onnx", nodes, text_inputs, "logits", shape)

constants = [
    helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
    helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]),
]
embedding_nodes = list(text_nodes) + [
    helper.make_node("Unsqueeze", ["x"], ["x3"], axes=[2]),
    helper.make_node("Add", ["x3", "one"], ["plus_one"]),
    helper.make_node("Add", ["x3", "two"], ["plus_two"]),
    helper.make_node(
        "Concat", ["x3", "plus_one", "plus_two"], ["last_hidden_state"], axis=2
    ),
]
save(
    "embedding",
    "model.onnx",
    embedding_nodes,
    text_inputs,
    "last_hidden_state",
    ["batch", "tokens", 3],
    constants,
)
save(
    "multimodal",
    "text_encoder.onnx",
    embedding_nodes
    + [
        helper.make_node(
            "ReduceMean", ["last_hidden_state"], ["embedding"], axes=[1], keepdims=0
        )
    ],
    text_inputs,
    "embedding",
    ["batch", 3],
    constants,
)
for name, input_name, shape in (
    ("image_encoder.onnx", "pixel_values", [1, 3, 2, 2]),
    ("audio_encoder.onnx", "mel_spectrogram", [1, 2, 3000]),
):
    nodes = [
        helper.make_node(
            "ReduceMean",
            [input_name],
            ["mean"],
            axes=list(range(1, len(shape))),
            keepdims=0,
        ),
        helper.make_node("Unsqueeze", ["mean"], ["score"], axes=[1]),
        helper.make_node("Add", ["score", "one"], ["plus_one"]),
        helper.make_node("Add", ["score", "two"], ["plus_two"]),
        helper.make_node(
            "Concat", ["score", "plus_one", "plus_two"], ["embedding"], axis=1
        ),
    ]
    save(
        "multimodal",
        name,
        nodes,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)],
        "embedding",
        [1, 3],
        constants,
    )
