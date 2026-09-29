"""Replace proved padding/window predicates with compact broadcast masks.

Attention scores, scaling, finite mask fills, softmax and NaN guards remain
unchanged. Global masks broadcast over queries instead of materializing a
quadratic tensor. Local windows retain their exact inclusive radius.

This optional graph transform does not qualify a model or execution provider.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from attention_masks import mask_contract, prune_unused
from onnx import TensorProto, helper, numpy_helper
from rewrite_graph import build_maps, find_attention_blocks, scalar_value

PREFIX = "__broadcast_mask_"
MIN_OPSET = 15  # Shape start/end attributes.
MASK_RANK = 2
ATTENTION_RANK = 4
MAX_MASK_FILL = -1024.0


def _plans(model):
    graph = model.graph
    if next((o.version for o in model.opset_import if not o.domain), 0) < MIN_OPSET:
        raise ValueError("Broadcast masks require ONNX opset >=15")
    if any(
        n.domain not in {"", "ai.onnx"} or n.op_type in {"Loop", "If", "Scan"}
        for n in graph.node
    ):
        raise ValueError("Only standard straight-line graphs are supported")
    names = {x.name for x in [*graph.input, *graph.initializer, *graph.value_info]}
    names.update(value for node in graph.node for value in node.output)
    if any(name.startswith(PREFIX) for name in names):
        raise ValueError("Reserved broadcast-mask value name already exists")
    producers, _ = build_maps(graph)
    blocks = find_attention_blocks(graph, producers)
    if not blocks or len(blocks) != sum(n.op_type == "Softmax" for n in graph.node):
        raise ValueError("Every attention Softmax must be recognized")
    mask = next((v for v in graph.input if v.name == "attention_mask"), None)
    if (
        mask is None
        or mask.type.tensor_type.elem_type != TensorProto.INT64
        or len(mask.type.tensor_type.shape.dim) != MASK_RANK
    ):
        raise ValueError("Expected a rank-two int64 attention_mask")
    plans = {}
    outputs = {v.name for v in graph.output}
    info = {
        x.name: x.type.tensor_type
        for x in [*graph.input, *graph.value_info, *graph.output]
    }
    for block in blocks:
        for value in [
            block["q_mul"].output[0],
            block["k_mul"].output[0],
            block["v_tensor"],
        ]:
            tensor = info.get(value)
            if (
                tensor is None
                or tensor.elem_type != TensorProto.FLOAT
                or len(tensor.shape.dim) != ATTENTION_RANK
            ):
                raise ValueError("Attention needs explicit rank-four FP32 value_info")
        name = block["mask_tensor"]
        node = producers[name]
        if (
            node.op_type != "Where"
            or scalar_value(graph, producers, node.input[1]) != 0
        ):
            raise ValueError("Expected a proved positive Where mask")
        radius, padding_fill, local_fill, _ = mask_contract(graph, producers, name)
        if not (
            padding_fill == local_fill
            and np.isfinite(padding_fill)
            and padding_fill <= MAX_MASK_FILL
        ):
            raise ValueError("Only identical finite negative mask fills are supported")
        if name in outputs:
            raise ValueError("The mask is externally observable")
        approved = {b["add_mask"].output[0] for b in blocks if b["mask_tensor"] == name}
        if any(
            n.op_type != "Add"
            or n.output[0] not in approved
            or list(n.input).count(name) != 1
            for n in graph.node
            if name in n.input
        ):
            raise ValueError("The mask has shape-observing or other consumers")
        plans[name] = (radius, padding_fill)
    return blocks, plans


class _Masks:
    def __init__(self):
        self.nodes = []
        self.initializers = []

    def constant(self, name, value, dtype=np.int64):
        name = PREFIX + name
        self.initializers.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
        )
        return name

    def op(self, kind, inputs, name, **attributes):
        name = PREFIX + name
        self.nodes.append(
            helper.make_node(kind, inputs, [name], name=name, **attributes)
        )
        return name

    def build(self, plans):
        zero = self.constant("zero", 0)
        one = self.constant("one", 1)
        float_zero = self.constant("float_zero", 0, np.float32)
        padding = self.op("Cast", ["attention_mask"], "bool", to=TensorProto.BOOL)
        padding = self.op(
            "Unsqueeze", [padding, self.constant("padding_axes", [1, 2])], "padding"
        )
        if any(radius >= 0 for radius, _ in plans.values()):
            size = self.op("Shape", ["attention_mask"], "shape", start=1, end=2)
            size = self.op("Squeeze", [size], "size")
            positions = self.op("Range", [zero, size, one], "positions")
            query = self.op(
                "Unsqueeze",
                [positions, self.constant("query_axes", [0, 1, 3])],
                "query",
            )
            key = self.op(
                "Unsqueeze", [positions, self.constant("key_axes", [0, 1, 2])], "key"
            )
            difference = self.op("Sub", [query, key], "delta")
            absolute = self.op("Abs", [difference], "absolute")
        for index, (name, (radius, fill)) in enumerate(plans.items()):
            keep = padding
            if radius >= 0:
                window = self.op(
                    "LessOrEqual",
                    [absolute, self.constant("radius" + str(index), radius)],
                    "window" + str(index),
                )
                keep = self.op("And", [padding, window], "local" + str(index))
            self.nodes.append(
                helper.make_node(
                    "Where",
                    [
                        keep,
                        float_zero,
                        self.constant("fill" + str(index), fill, np.float32),
                    ],
                    [name],
                    name=PREFIX + "mask" + str(index),
                )
            )


def rewrite_model(model):
    """Return a new model and proof receipt; never mutate input, including on error."""
    blocks, plans = _plans(model)
    result = copy.deepcopy(model)
    masks = _Masks()
    masks.build(plans)
    remaining = [n for n in result.graph.node if not any(v in plans for v in n.output)]
    del result.graph.node[:]
    result.graph.node.extend(masks.nodes + remaining)
    result.graph.initializer.extend(masks.initializers)
    infos = [v for v in result.graph.value_info if v.name not in plans]
    del result.graph.value_info[:]
    result.graph.value_info.extend(infos)
    prune_unused(result.graph)
    producers, _ = build_maps(result.graph)
    rewritten = find_attention_blocks(result.graph, producers)
    before = [n.SerializeToString() for b in blocks for n in b["probability_nodes"]]
    after = [n.SerializeToString() for b in rewritten for n in b["probability_nodes"]]
    if len(blocks) != len(rewritten) or before != after:
        raise ValueError("Attention or NaN guards changed during mask canonicalization")
    return result, {
        "attention_blocks": len(blocks),
        "masks": {n: {"radius": r, "fill": f} for n, (r, f) in plans.items()},
        "nan_guards_unchanged": True,
        "old_nodes": len(model.graph.node),
        "new_nodes": len(result.graph.node),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output_files = [
        args.output,
        Path(str(args.output) + ".data"),
        Path(str(args.output) + ".json"),
    ]
    if any(path.exists() for path in output_files):
        raise ValueError("Refusing to overwrite an existing graph, weights or receipt")
    model, receipt = rewrite_model(onnx.load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
    )
    onnx.checker.check_model(str(args.output))
    output_files[-1].write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
