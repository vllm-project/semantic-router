"""Tiny deterministic ONNX graphs with the tensor contracts of the packages' real graphs.

Built with the ``onnx`` package (a test dependency). Weights are small and
seeded, so every test sees the same numbers; graphs can share an external
weight file the way exported Vela graphs do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ONNX Runtime 1.20+ reads IR version 10; newer onnx releases write 11 by default.
IR_VERSION = 10
OPSET = 18


def _onnx():
    import onnx

    return onnx


def weights(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.5, shape).astype(np.float32)


def _save(
    model, path: Path, metadata: dict[str, str] | None, external: str | None
) -> Path:
    onnx = _onnx()
    for key, value in (metadata or {}).items():
        model.metadata_props.add(key=key, value=value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if external:
        # Weights go to the external file; small shape constants stay inline, as in real exports.
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external,
            size_threshold=256,
        )
    else:
        onnx.save_model(model, str(path))
    return path


def token_graph(
    path: Path,
    *,
    vocab: int = 32,
    hidden: int = 8,
    output: str = "last_hidden_state",
    scorer: int | None = None,
    position_ids: str = "row",
    metadata: dict[str, str] | None = None,
    external: str | None = None,
    seed: int = 0,
) -> Path:
    """``input_ids``, ``attention_mask``, ``position_ids`` -> hidden states or ``logits``.

    ``hidden = embed[ids] * mask + 0.01 * position``; with ``scorer`` the graph
    reads the first token's first ``scorer`` channels into one logit per row.
    ``position_ids`` is ``row`` ([1, sequence]) or ``batch`` ([batch, sequence]).
    """
    onnx = _onnx()
    helper, numpy_helper, types = onnx.helper, onnx.numpy_helper, onnx.TensorProto
    table = numpy_helper.from_array(weights((vocab, hidden), seed), "embed.weight")
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")
    scale = numpy_helper.from_array(np.array(0.01, dtype=np.float32), "scale")
    nodes = [
        helper.make_node("Gather", ["embed.weight", "input_ids"], ["tokens"]),
        helper.make_node("Cast", ["attention_mask"], ["mask"], to=types.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "axes"], ["mask3"]),
        helper.make_node("Mul", ["tokens", "mask3"], ["masked"]),
        helper.make_node("Cast", ["position_ids"], ["positions"], to=types.FLOAT),
        helper.make_node("Unsqueeze", ["positions", "axes"], ["positions3"]),
        helper.make_node("Mul", ["positions3", "scale"], ["offset"]),
    ]
    initializers = [table, axes, scale]
    if scorer is None:
        nodes.append(helper.make_node("Add", ["masked", "offset"], [output]))
        result = helper.make_tensor_value_info(
            output, types.FLOAT, ["batch", "sequence", hidden]
        )
    else:
        head = numpy_helper.from_array(weights((scorer, 1), seed + 1), "head.weight")
        start = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), "start")
        end = numpy_helper.from_array(np.array([1, scorer], dtype=np.int64), "end")
        slice_axes = numpy_helper.from_array(
            np.array([1, 2], dtype=np.int64), "slice_axes"
        )
        squeeze_axes = numpy_helper.from_array(
            np.array([1], dtype=np.int64), "squeeze_axes"
        )
        initializers += [head, start, end, slice_axes, squeeze_axes]
        nodes += [
            helper.make_node("Add", ["masked", "offset"], ["hidden"]),
            helper.make_node(
                "Slice", ["hidden", "start", "end", "slice_axes"], ["first"]
            ),
            helper.make_node("Squeeze", ["first", "squeeze_axes"], ["cls"]),
            helper.make_node("MatMul", ["cls", "head.weight"], [output]),
        ]
        result = helper.make_tensor_value_info(output, types.FLOAT, ["batch", 1])
    position_shape = [1, "sequence"] if position_ids == "row" else ["batch", "sequence"]
    graph = helper.make_graph(
        nodes,
        "token_graph",
        [
            helper.make_tensor_value_info(
                "input_ids", types.INT64, ["batch", "sequence"]
            ),
            helper.make_tensor_value_info(
                "attention_mask", types.INT64, ["batch", "sequence"]
            ),
            helper.make_tensor_value_info("position_ids", types.INT64, position_shape),
        ],
        [result],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )
    return _save(model, Path(path), metadata, external)


def projection_graph(
    path: Path,
    inputs: dict[str, list[int | str]],
    dimension: int,
    *,
    seed: int = 0,
    normalize: bool = True,
) -> Path:
    """Named float inputs -> ``embedding [1, dimension]``: the flattened inputs projected and L2-normalized.

    Integer inputs (token IDs and masks) are cast to float first, so the same
    builder stands in for the Omni text, image, CLAP and audio graphs.
    """
    onnx = _onnx()
    helper, numpy_helper, types = onnx.helper, onnx.numpy_helper, onnx.TensorProto
    nodes, initializers, terms = [], [], []
    for position, (name, shape) in enumerate(inputs.items()):
        integer = name in ("input_ids", "attention_mask")
        source = name
        if integer:
            nodes.append(
                helper.make_node("Cast", [name], [f"{name}_f"], to=types.FLOAT)
            )
            source = f"{name}_f"
        # Mean over every axis but the first keeps the graph shape-agnostic.
        nodes.append(
            helper.make_node(
                "ReduceMean",
                [source, f"{name}_axes"],
                [f"{name}_mean"],
                keepdims=0,
            )
        )
        reduce_axes = list(range(1, len(shape)))
        initializers.append(
            numpy_helper.from_array(
                np.array(reduce_axes, dtype=np.int64), f"{name}_axes"
            )
        )
        nodes.append(
            helper.make_node("Unsqueeze", [f"{name}_mean", "one"], [f"{name}_col"])
        )
        initializers.append(
            numpy_helper.from_array(
                weights((1, dimension), seed + position), f"{name}_w"
            )
        )
        nodes.append(
            helper.make_node("MatMul", [f"{name}_col", f"{name}_w"], [f"{name}_t"])
        )
        terms.append(f"{name}_t")
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "one"))
    bias = numpy_helper.from_array(weights((1, dimension), seed + 99), "bias")
    initializers.append(bias)
    nodes.append(helper.make_node("Sum", [*terms, "bias"], ["raw"]))
    if normalize:
        nodes.append(
            helper.make_node("LpNormalization", ["raw"], ["embedding"], axis=-1, p=2)
        )
    else:
        nodes.append(helper.make_node("Identity", ["raw"], ["embedding"]))
    graph_inputs = [
        helper.make_tensor_value_info(
            name,
            (types.INT64 if name in ("input_ids", "attention_mask") else types.FLOAT),
            shape,
        )
        for name, shape in inputs.items()
    ]
    graph = helper.make_graph(
        nodes,
        "projection_graph",
        graph_inputs,
        [helper.make_tensor_value_info("embedding", types.FLOAT, [1, dimension])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], ir_version=IR_VERSION
    )
    return _save(model, Path(path), None, None)
