"""Bound FP32 attention with query blocks and guarded local key/value cropping.

Global attention and guarded fallback retain complete keys. Local fast paths
remove only terms whose masked exponential is provably zero in FP32.

This preserves the export's separate Q/K scale operations and mask fill values.
It is an optional graph variant; numerical and execution-provider qualification
of a real model remains separate from successful graph rewriting.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from rewrite_graph import (
    attention_window,
    build_maps,
    find_attention_blocks,
    inverted_padding_fill,
    scalar_value,
)

INT32_ELEMENTS = 2**31 - 1
DEFAULT_SCORE_BYTES = 512 * 1024 * 1024
FLOAT32_BYTES = 4
ATTENTION_RANK = 4
MASK_RANK = 2
MIN_OPSET = 13
LOCAL_CROP_MAX_FINITE_FILL = -1024.0


def _array(graph, producers, name):
    value = next((x for x in graph.initializer if x.name == name), None)
    if value is None:
        node = producers.get(name)
        if node is not None and node.op_type == "Constant":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
    return None if value is None else numpy_helper.to_array(value)


def _ancestors(producers, name):
    result = {}
    pending = [name]
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is not None and node.output[0] not in result:
            result[node.output[0]] = node
            pending.extend(node.input)
    return result


def _mask_contract(graph, producers, name):
    window = attention_window(graph, producers, name)
    ancestors = _ancestors(producers, name)
    for node in ancestors.values():
        if node.op_type == "Cast" and any(
            a.name == "to"
            and a.i not in {TensorProto.FLOAT, TensorProto.BOOL, TensorProto.INT64}
            for a in node.attribute
        ):
            raise ValueError(
                "Narrow integer or mixed precision mask casts are unsupported"
            )
        if node.op_type == "Range" and (
            scalar_value(graph, producers, node.input[0]) != 0
            or scalar_value(graph, producers, node.input[2]) != 1
        ):
            raise ValueError(
                "Only zero-based unit-step self-attention positions are supported"
            )

    def position_view(value):
        node = producers.get(value)
        if node is None:
            raise ValueError("Unknown position view")
        if node.op_type == "Range":
            return ["position"]
        if node.op_type == "Cast" and not any(
            a.name == "to" and a.i == TensorProto.INT64 for a in node.attribute
        ):
            raise ValueError("Position differences must remain exact integers")
        if node.op_type in {"Identity", "Cast"}:
            return position_view(node.input[0])
        dims = position_view(node.input[0])
        if node.op_type == "Unsqueeze":
            axes = _array(graph, producers, node.input[1])
            if axes is not None:
                rank = len(dims) + len(axes)
                if any(not -rank <= int(axis) < rank for axis in axes):
                    raise ValueError("Position unsqueeze axis is out of range")
                axes = sorted(int(axis) % rank for axis in axes)
                if len(set(axes)) == len(axes):
                    for axis in axes:
                        dims.insert(axis, 1)
                    return dims
        elif node.op_type == "Transpose":
            perm = next(
                (list(a.ints) for a in node.attribute if a.name == "perm"), None
            )
            if perm is not None and sorted(perm) == list(range(len(dims))):
                return [dims[axis] for axis in perm]
        elif node.op_type == "Reshape":
            shape = _array(graph, producers, node.input[1])
            if (
                shape is not None
                and list(shape).count(-1) == 1
                and all(x in {1, -1} for x in shape)
            ):
                return ["position" if x == -1 else 1 for x in shape]
        raise ValueError("Unsupported positional broadcast")

    def predicate(value):
        if value == "attention_mask":
            return {"padding_2d"}
        node = producers.get(value)
        if node is None:
            array = _array(graph, producers, value)
            if array is not None and array.size == 1 and bool(array.item()):
                return {"true"}
            raise ValueError(f"Unknown attention-mask predicate {value!r}")
        if node.op_type in {"Identity", "Expand", "Cast"}:
            if node.op_type == "Cast" and not any(
                a.name == "to" and a.i == TensorProto.BOOL for a in node.attribute
            ):
                raise ValueError("Positive mask predicate must remain boolean")
            return predicate(node.input[0])
        if node.op_type == "Unsqueeze":
            flags = predicate(node.input[0])
            axes = _array(graph, producers, node.input[1])
            if flags == {"padding_2d"}:
                if axes is None or list(axes) != [1, 2]:
                    raise ValueError("Padding must broadcast over keys, not queries")
                return {"padding"}
            if "padding_2d" in flags:
                raise ValueError("Unsupported padding broadcast")
            return flags
        if node.op_type == "And":
            return predicate(node.input[0]) | predicate(node.input[1])
        if node.op_type in {"Less", "LessOrEqual"}:
            # The shared recognizer proves abs(q-k) and the radius. Also prove
            # these are the query and key axes, not two views of the same axis.
            absolute = producers[node.input[0]]
            difference = producers[absolute.input[0]]
            views = [position_view(value) for value in difference.input]
            views = [tuple([1] * (4 - len(view)) + view) for view in views]
            if set(views) != {(1, 1, "position", 1), (1, 1, 1, "position")}:
                raise ValueError(
                    "Local mask must compare absolute query and key positions"
                )
            return {"window"}
        if node.op_type == "GreaterOrEqual":
            position = producers.get(node.input[0])
            while position is not None and position.op_type in {
                "Cast",
                "Identity",
                "Unsqueeze",
                "Squeeze",
                "Reshape",
                "Transpose",
            }:
                position = producers.get(position.input[0])
            if (
                scalar_value(graph, producers, node.input[1]) == 0
                and position is not None
                and position.op_type == "Range"
                and scalar_value(graph, producers, position.input[0]) == 0
                and scalar_value(graph, producers, position.input[2]) == 1
            ):
                return {"true"}
            raise ValueError("Unproved non-negative attention position predicate")
        if node.op_type == "ConstantOfShape":
            fill = next((a.t for a in node.attribute if a.name == "value"), None)
            if fill is not None and numpy_helper.to_array(fill).item() == 1:
                return {"true"}
        if node.op_type == "Constant":
            array = _array(graph, producers, value)
            if array is not None and array.size == 1 and bool(array.item()):
                return {"true"}
        raise ValueError(f"Unknown attention-mask operation {node.op_type!r}")

    mask = producers[name]
    first = scalar_value(graph, producers, mask.input[1])
    last = scalar_value(graph, producers, mask.input[2])
    if first == 0:
        flags = predicate(mask.input[0])
        expected = {"padding"} | ({"window"} if window[0] >= 0 else set())
        if flags - {"true"} != expected:
            raise ValueError(
                "Attention mask is not the supported key-padding/window contract"
            )
        return window[0], last, last, ancestors
    if inverted_padding_fill(graph, producers, mask):
        return -1, first, first, ancestors
    condition = producers.get(mask.input[0])
    if condition is None or condition.op_type != "Not":
        raise ValueError("Unknown negative-fill attention predicate")
    inner_radius, padding_fill, _, _ = _mask_contract(graph, producers, mask.input[2])
    if inner_radius != -1 or predicate(condition.input[0]) != {"window"}:
        raise ValueError("Unsupported nested local attention mask")
    return window[0], padding_fill, first, ancestors


class _Nodes:
    def __init__(self, prefix):
        self.prefix = prefix
        self.nodes = []

    def op(self, kind, inputs, tag, **attributes):
        name = self.prefix + tag
        self.nodes.append(
            helper.make_node(kind, inputs, [name], name=name, **attributes)
        )
        return name

    def constant(self, tag, value, dtype=np.int64):
        return self.op(
            "Constant",
            [],
            tag,
            value=numpy_helper.from_array(np.asarray(value, dtype=dtype)),
        )


def _local_crop_guard(outer, q, k, v, depth, integer_mask, local_fill):
    """Permit removal only for all-valid padding and finite, bounded dot products.

    L1(Q)*L1(K)*depth overbounds every dot product. The extra factor eight
    covers FP32 accumulation error for depth <= 4096. With the finite-fill
    threshold below, even opposite-sign extrema leave a > 256 softmax gap,
    so every removed term has an exactly-zero FP32 exponential. Infinity,
    NaN, reduction overflow, small mask fills, and any padding use full K/V.
    V must also be finite: removing 0*NaN/Inf would change invalid-input math.
    """
    if np.isneginf(local_fill):
        limit = np.finfo(np.float32).max / 16.0
    elif np.isfinite(local_fill) and local_fill <= LOCAL_CROP_MAX_FINITE_FILL:
        limit = (-float(local_fill) - 256.0) / 4.0
    else:
        return outer.constant("crop_eligible", False, np.bool_)
    limit = np.nextafter(np.float32(limit), np.float32(0))
    totals = []
    for tag, value in [("q", q), ("k", k), ("v", v)]:
        absolute = outer.op("Abs", [value], "crop_" + tag + "_abs")
        totals.append(
            outer.op("ReduceSum", [absolute], "crop_" + tag + "_l1", keepdims=0)
        )
    product = outer.op("Mul", totals[:2], "crop_l1_product")
    dimension = outer.op("Cast", [depth], "crop_depth_float", to=TensorProto.FLOAT)
    factor = outer.op(
        "Mul",
        [dimension, outer.constant("crop_roundoff_factor", 8.0, np.float32)],
        "crop_dimension_factor",
    )
    bound = outer.op("Mul", [product, factor], "crop_dot_bound")
    bounded = outer.op(
        "LessOrEqual",
        [bound, outer.constant("crop_limit", limit, np.float32)],
        "crop_bounded",
    )
    value_finite = outer.op(
        "LessOrEqual",
        [
            totals[2],
            outer.constant("crop_float_max", np.finfo(np.float32).max, np.float32),
        ],
        "crop_value_finite",
    )
    minimum = outer.op("ReduceMin", [integer_mask], "crop_mask_min", keepdims=0)
    all_valid = outer.op(
        "Equal", [minimum, outer.constant("crop_mask_one", 1)], "crop_all_valid"
    )
    positive_depth = outer.op(
        "Greater", [depth, outer.constant("crop_depth_zero", 0)], "crop_positive_depth"
    )
    bounded_depth = outer.op(
        "LessOrEqual",
        [depth, outer.constant("crop_depth_max", 4096)],
        "crop_bounded_depth",
    )
    eligible = bounded
    for tag, value in [
        ("values", value_finite),
        ("padding", all_valid),
        ("positive_depth", positive_depth),
        ("bounded_depth", bounded_depth),
    ]:
        eligible = outer.op("And", [eligible, value], "crop_guard_" + tag)
    return outer.op("Identity", [eligible], "crop_eligible")


def _replacement(block, index, radius, padding_fill, local_fill, block_size, elements):
    outer = _Nodes(f"__blocked_attention_{index}_")
    zero = outer.constant("zero", 0)
    one = outer.constant("one", 1)
    two = outer.constant("two", 2)
    axes0 = outer.constant("axes0", [0])
    axes2 = outer.constant("axes2", [2])
    true = outer.constant("true", True, np.bool_)
    float_zero = outer.constant("float_zero", 0, np.float32)
    q = block["q_mul"].output[0]
    k = block["k_mul"].output[0]
    v = block["v_tensor"]
    qshape = outer.op("Shape", [q], "qshape")
    kshape = outer.op("Shape", [k], "kshape")
    vshape = outer.op("Shape", [v], "vshape")
    mshape = outer.op("Shape", ["attention_mask"], "mshape")

    def dimension(shape, axis, tag):
        return outer.op(
            "Gather", [shape, outer.constant(tag + "_axis", axis)], tag, axis=0
        )

    batch = dimension(qshape, 0, "batch")
    heads = dimension(qshape, 1, "heads")
    queries = dimension(qshape, 2, "queries")
    depth = dimension(qshape, 3, "depth")
    keys = dimension(kshape, 3, "keys")
    values = dimension(vshape, 3, "values")
    bh = outer.op("Mul", [batch, heads], "batch_heads")
    bhk = outer.op("Mul", [bh, keys], "batch_heads_keys")
    denominator = outer.op("Max", [bhk, one], "nonzero_denominator")
    capacity = outer.op(
        "Div", [outer.constant("element_limit", elements), denominator], "capacity"
    )
    valid = outer.op("Greater", [capacity, zero], "capacity_valid")
    for tag, value in [
        ("batch", batch),
        ("heads", heads),
        ("queries", queries),
        ("keys", keys),
    ]:
        positive = outer.op("Greater", [value, zero], tag + "_positive")
        valid = outer.op("And", [valid, positive], tag + "_valid")
    for tag, shape, axis, expected in [
        ("kb", kshape, 0, batch),
        ("kh", kshape, 1, heads),
        ("kd", kshape, 2, depth),
        ("vb", vshape, 0, batch),
        ("vh", vshape, 1, heads),
        ("vk", vshape, 2, keys),
        ("mb", mshape, 0, batch),
        ("mk", mshape, 1, keys),
    ]:
        actual = dimension(shape, axis, tag)
        equal = outer.op("Equal", [actual, expected], tag + "_equal")
        valid = outer.op("And", [valid, equal], tag + "_valid")
    equal_length = outer.op("Equal", [queries, keys], "self_attention_length")
    valid = outer.op("And", [valid, equal_length], "shape_valid")
    integer_mask = outer.op(
        "Cast", ["attention_mask"], "integer_mask", to=TensorProto.INT64
    )
    is_zero = outer.op("Equal", [integer_mask, zero], "mask_zero")
    is_one = outer.op("Equal", [integer_mask, one], "mask_one")
    binary = outer.op("Or", [is_zero, is_one], "binary_mask")
    binary = outer.op("Cast", [binary], "binary_integer", to=TensorProto.INT64)
    binary = outer.op("ReduceMin", [binary], "all_binary", keepdims=0)
    binary = outer.op("Equal", [binary, one], "binary_valid")
    valid = outer.op("And", [valid, binary], "input_valid")
    # ONNX has no Assert. An invalid shape deterministically fails Reshape before
    # score allocation; never divide by zero or silently exceed the memory cap.
    guard_size = outer.op("Where", [valid, one, two], "guard_size")
    guard_shape = outer.op("Unsqueeze", [guard_size, axes0], "guard_shape")
    guard = outer.op(
        "Reshape", [outer.constant("guard_token", [1]), guard_shape], "guard"
    )
    guard_scalar = outer.op("Gather", [guard, zero], "guard_scalar", axis=0)
    requested = outer.constant("requested_block", block_size)
    limited = outer.op("Min", [requested, capacity], "limited_block")
    safe = outer.op("Max", [limited, one], "nonzero_block")
    chunk = outer.op("Mul", [safe, guard_scalar], "block")
    ceiling = outer.op("Add", [queries, chunk], "ceiling_sum")
    ceiling = outer.op("Sub", [ceiling, one], "ceiling_numerator")
    trips = outer.op("Div", [ceiling, chunk], "trips")
    padding = outer.op("Cast", ["attention_mask"], "padding", to=TensorProto.BOOL)
    padding = outer.op(
        "Unsqueeze", [padding, outer.constant("pad_axes", [1, 2])], "padding4d"
    )
    padding_bias = outer.op(
        "Where",
        [padding, float_zero, outer.constant("padding_fill", padding_fill, np.float32)],
        "padding_bias",
    )
    crop = (
        _local_crop_guard(outer, q, k, v, depth, integer_mask, local_fill)
        if radius >= 0
        else None
    )

    body = _Nodes(outer.prefix + "body_")
    iteration, condition = body.prefix + "iteration", body.prefix + "condition"
    start = body.op("Mul", [iteration, chunk], "start")
    end = body.op("Add", [start, chunk], "unclipped_end")
    end = body.op("Min", [end, queries], "end")
    start_vector = body.op("Unsqueeze", [start, axes0], "start_vector")
    end_vector = body.op("Unsqueeze", [end, axes0], "end_vector")
    sliced_q = body.op("Slice", [q, start_vector, end_vector, axes2], "queries")
    selected_k, selected_v = k, v
    bias = padding_bias
    if radius >= 0:
        radius_value = body.constant("radius", radius)
        local_start = body.op("Sub", [start, radius_value], "local_key_start_unclipped")
        local_start = body.op("Max", [local_start, zero], "local_key_start")
        local_end = body.op("Add", [end, radius_value], "local_key_end_unclipped")
        local_end = body.op("Min", [local_end, keys], "local_key_end")
        key_start = body.op("Where", [crop, local_start, zero], "key_start")
        key_end = body.op("Where", [crop, local_end, keys], "key_end")
        key_start_vector = body.op("Unsqueeze", [key_start, axes0], "key_start_vector")
        key_end_vector = body.op("Unsqueeze", [key_end, axes0], "key_end_vector")
        axes3 = body.constant("axes3", [3])
        selected_k = body.op(
            "Slice", [k, key_start_vector, key_end_vector, axes3], "keys"
        )
        selected_v = body.op(
            "Slice", [v, key_start_vector, key_end_vector, axes2], "values"
        )
        bias = body.op(
            "Slice",
            [padding_bias, key_start_vector, key_end_vector, axes3],
            "padding_bias",
        )
        key_positions = body.op("Range", [key_start, key_end, one], "key_positions")
    scores = body.op("MatMul", [sliced_q, selected_k], "scores")
    if radius >= 0:
        positions = body.op("Range", [start, end, one], "query_positions")
        positions = body.op(
            "Unsqueeze", [positions, body.constant("column_axis", [1])], "query_column"
        )
        keys_row = body.op("Unsqueeze", [key_positions, axes0], "key_row")
        difference = body.op("Sub", [positions, keys_row], "difference")
        distance = body.op("Abs", [difference], "distance")
        keep = body.op("LessOrEqual", [distance, radius_value], "local_keep")
        bias = body.op(
            "Where",
            [keep, bias, body.constant("local_fill", local_fill, np.float32)],
            "local_bias",
        )
    masked = body.op("Add", [scores, bias], "masked")
    probability = body.op("Softmax", [masked], "probability", axis=-1)
    if block["probability_nodes"]:
        isnan = body.op("IsNaN", [probability], "isnan")
        probability = body.op(
            "Where", [isnan, float_zero, probability], "guarded_probability"
        )
    result = body.op("MatMul", [probability, selected_v], "result")
    length = body.op("Sub", [end, start], "length")
    remaining = body.op("Sub", [chunk, length], "remaining")
    remaining = body.op("Unsqueeze", [remaining, axes0], "remaining_vector")
    pads = body.op(
        "Concat",
        [
            body.constant("six_zeros", [0] * 6),
            remaining,
            body.constant("last_zero", [0]),
        ],
        "pads",
        axis=0,
    )
    scan = body.op("Pad", [result, pads, float_zero], "scan")
    continued = body.op("Identity", [condition], "continue")
    body_graph = helper.make_graph(
        body.nodes,
        body.prefix,
        [
            helper.make_tensor_value_info(iteration, TensorProto.INT64, []),
            helper.make_tensor_value_info(condition, TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info(continued, TensorProto.BOOL, []),
            helper.make_tensor_value_info(scan, TensorProto.FLOAT, [None] * 4),
        ],
    )
    scanned = outer.op("Loop", [trips, true], "scan", body=body_graph)
    ordered = outer.op("Transpose", [scanned], "ordered", perm=[1, 2, 0, 3, 4])
    dimensions = [
        outer.op("Unsqueeze", [value, axes0], tag + "_vector")
        for tag, value in [("b", batch), ("h", heads), ("v", values)]
    ]
    shape = outer.op(
        "Concat",
        [
            dimensions[0],
            dimensions[1],
            outer.constant("infer_length", [-1]),
            dimensions[2],
        ],
        "result_shape",
        axis=0,
    )
    flat = outer.op("Reshape", [ordered, shape], "flat")
    length_vector = outer.op("Unsqueeze", [queries, axes0], "query_length_vector")
    outer.nodes.append(
        helper.make_node(
            "Slice",
            [flat, outer.constant("slice_start", [0]), length_vector, axes2],
            [block["output_tensor"]],
            name=outer.prefix + "output",
        )
    )
    return outer.nodes


def _dependencies(node):
    values = set(node.input)
    for attribute in node.attribute:
        if attribute.type != onnx.AttributeProto.GRAPH:
            continue
        graph = attribute.g
        defined = {x.name for x in graph.input} | {x.name for x in graph.initializer}
        defined.update(output for child in graph.node for output in child.output)
        used = {name for child in graph.node for name in _dependencies(child)}
        values.update(used - defined)
    return values


def _prune(graph):
    producers, _ = build_maps(graph)
    needed = {x.name for x in graph.output}
    pending = list(needed)
    while pending:
        node = producers.get(pending.pop())
        if node is not None:
            for name in _dependencies(node) - needed:
                needed.add(name)
                pending.append(name)
    nodes = [n for n in graph.node if any(x in needed for x in n.output)]
    del graph.node[:]
    graph.node.extend(nodes)
    initializers = [x for x in graph.initializer if x.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    infos = [x for x in graph.value_info if x.name in needed]
    del graph.value_info[:]
    graph.value_info.extend(infos)
    return needed


def rewrite_model(model, block_size=256, max_score_bytes=DEFAULT_SCORE_BYTES):
    """Return a new, validated model and receipt; do not mutate input on failure."""
    if (
        block_size not in {256, 512}
        or not FLOAT32_BYTES <= max_score_bytes <= INT32_ELEMENTS * FLOAT32_BYTES
    ):
        raise ValueError(
            "Use block cap 256/512 and a positive FP32 score budget below int32"
        )
    if next((o.version for o in model.opset_import if not o.domain), 0) < MIN_OPSET:
        raise ValueError("Blocked attention requires ONNX opset >=13")
    graph = model.graph
    if any(
        n.op_type
        in {
            "Loop",
            "Attention",
            "MultiHeadAttention",
            "FlashAttention",
            "CKFlashAttention",
        }
        or n.domain not in {"", "ai.onnx"}
        or n.name.startswith("__blocked_attention_")
        for n in graph.node
    ):
        raise ValueError("Refusing an already blocked or control-flow model")
    producers, _ = build_maps(graph)
    blocks = find_attention_blocks(graph, producers)
    if not blocks:
        raise ValueError("No fully recognized attention blocks")
    info = {
        x.name: x.type.tensor_type
        for x in [*graph.input, *graph.value_info, *graph.output]
    }
    floating = {TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
    constant_types = [
        a.t.data_type
        for n in graph.node
        for a in n.attribute
        if a.type == onnx.AttributeProto.TENSOR
    ]
    if (
        any(x.data_type in floating for x in graph.initializer)
        or any(dtype in floating for dtype in constant_types)
        or any(t.elem_type in floating for t in info.values())
    ):
        raise ValueError("Only FP32 weights and attention are supported")
    if any(
        n.op_type == "Cast"
        and any(a.name == "to" and a.i in floating for a in n.attribute)
        for n in graph.node
    ):
        raise ValueError("Mixed precision attention graphs are refused")
    mask_info = info.get("attention_mask")
    if (
        mask_info is None
        or mask_info.elem_type not in {TensorProto.INT64, TensorProto.INT32}
        or len(mask_info.shape.dim) != MASK_RANK
    ):
        raise ValueError("Expected a rank-two integer attention_mask")
    matched = {block["softmax"].output[0] for block in blocks}
    for node in graph.node:
        if node.op_type == "Softmax" and node.output[0] not in matched:
            tensor = info.get(node.output[0])
            if tensor is None or len(tensor.shape.dim) != MASK_RANK:
                raise ValueError(
                    "Unrecognized attention Softmax; refusing a partial rewrite"
                )
    plans, removed, quadratic = [], set(), set()
    for block in blocks:
        for name in [
            block["q_mul"].output[0],
            block["k_mul"].output[0],
            block["v_tensor"],
        ]:
            tensor = info.get(name)
            if (
                tensor is None
                or tensor.elem_type != TensorProto.FLOAT
                or len(tensor.shape.dim) != ATTENTION_RANK
            ):
                raise ValueError(
                    f"Attention tensor {name!r} needs explicit rank-four FP32 value_info"
                )
        if any(
            a.name == "axis" and a.i not in {-1, 3} for a in block["softmax"].attribute
        ):
            raise ValueError("Softmax must normalize the complete key dimension")
        radius, pad_fill, local_fill, ancestors = _mask_contract(
            graph, producers, block["mask_tensor"]
        )
        plans.append((radius, pad_fill, local_fill))
        old = [
            block[x] for x in ["qk_matmul", "add_mask", "softmax", "av_matmul"]
        ] + block["probability_nodes"]
        names = {x.output[0] for x in old}
        for node in old:
            if node is block["av_matmul"]:
                continue
            if any(
                output in {x.name for x in graph.output} for output in node.output
            ) or any(
                n.output[0] not in names
                and any(output in n.input for output in node.output)
                for n in graph.node
            ):
                raise ValueError("Attention intermediates have additional consumers")
        removed.update(names)
        quadratic.update(
            name
            for name, node in ancestors.items()
            if node.op_type
            in {"Where", "And", "Less", "LessOrEqual", "Abs", "Sub", "Expand"}
        )
    result = copy.deepcopy(model)
    replacement = {
        b["output_tensor"]: _replacement(
            b, i, *plans[i], block_size, max_score_bytes // FLOAT32_BYTES
        )
        for i, b in enumerate(blocks)
    }
    nodes = []
    for node in result.graph.node:
        if node.output[0] in replacement:
            nodes.extend(replacement[node.output[0]])
        elif node.output[0] not in removed:
            nodes.append(node)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    live = _prune(result.graph)
    if quadratic & live:
        raise ValueError("Original dense mask computation has external consumers")
    onnx.checker.check_model(result)
    receipt = {
        "attention_blocks": len(blocks),
        "query_block_cap": block_size,
        "maximum_score_bytes": max_score_bytes,
        "score_elements_below_int32": True,
        "precision": "float32",
        "keys_and_values": "global: complete; local: guarded absolute query interval plus radius; otherwise complete",
        "local_crop_guard": "all batch keys valid, finite V, conservative Q/K FP32 bound; otherwise full K/V",
        "local_crop_version": 2,
        "memory_cap": "unchanged conservative full-key score cap",
        "cropped_local_blocks": sum(p[0] >= 0 for p in plans),
        "q_k_scaling": "original separate operations preserved",
        "windows": [p[0] for p in plans],
        "real_model_qualification": False,
    }
    return result, receipt


def specialize_batch(model, batch_size):
    """Restrict a token-ID export to an explicitly qualified batch size."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("Static batch size must be a positive integer")
    if {value.name for value in model.graph.input} != {"input_ids", "attention_mask"}:
        raise ValueError(
            "Static batch specialization requires token IDs and padding only"
        )
    for value in model.graph.input:
        tensor = value.type.tensor_type
        if len(tensor.shape.dim) != MASK_RANK:
            raise ValueError("Token IDs and padding must have rank two")
        batch = tensor.shape.dim[0]
        if batch.HasField("dim_value") and batch.dim_value != batch_size:
            raise ValueError("Cannot change a pre-existing fixed batch contract")
    result = copy.deepcopy(model)
    for value in result.graph.input:
        value.type.tensor_type.shape.dim[0].dim_value = batch_size
    # Keep inferred intermediates symbolic; the runtime propagates the fixed
    # input shape. No operator, weight, or attention computation is changed.
    onnx.checker.check_model(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--max-score-bytes", type=int, default=DEFAULT_SCORE_BYTES)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Restrict token-ID inputs to this separately qualified static batch size",
    )
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".data").exists():
        raise ValueError("Refusing to overwrite an existing graph or external weights")
    model, receipt = rewrite_model(
        onnx.load(args.input), args.block_size, args.max_score_bytes
    )
    if args.batch_size is not None:
        model = specialize_batch(model, args.batch_size)
        receipt["static_batch_size"] = args.batch_size
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
    Path(str(args.output) + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
