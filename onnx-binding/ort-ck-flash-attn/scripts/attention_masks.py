"""Prove key-padding/window mask semantics shared by attention graph transforms.

Unknown predicates and index layouts fail closed. These checks establish mask
values; each caller must separately protect observable shapes and consumers.
"""

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from rewrite_graph import (
    attention_window,
    build_maps,
    inverted_padding_fill,
    scalar_value,
)

MASK_RANK = 2


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


def _gathered_padding(graph, producers, gather):
    """Prove the export's GatherND is precisely mask[batch, key].

    Its two index ranges broadcast to [B, 1, 1, K], then concatenate
    on a new final axis. No query-dependent indices or arbitrary gather
    expressions are accepted, even when their example output agrees.
    """

    def node(value, kind):
        result = producers.get(value)
        if result is None or result.op_type != kind:
            raise ValueError(f"GatherND padding requires {kind}")
        return result

    def attributes(value):
        return {a.name: a.i for a in value.attribute}

    def unsqueeze(value, axes):
        result = node(value, "Unsqueeze")
        actual = _array(graph, producers, result.input[1])
        if actual is None or actual.dtype != np.int64 or actual.tolist() != axes:
            raise ValueError("Unproved GatherND index axis")
        return result.input[0]

    def index_range(value, axis):
        result = node(value, "Range")
        for name, expected in [(result.input[0], 0), (result.input[2], 1)]:
            constant = _array(graph, producers, name)
            if (
                constant is None
                or constant.dtype != np.int64
                or constant.ndim != 0
                or constant.item() != expected
            ):
                raise ValueError("GatherND indices require exact int64 ranges")
        squeeze = node(result.input[1], "Squeeze")
        axes_inputs = squeeze.input[1:]
        if axes_inputs:
            if len(axes_inputs) != 1:
                raise ValueError("Unproved GatherND range dimension squeeze")
            axes = _array(graph, producers, axes_inputs[0])
            if axes is None or axes.dtype != np.int64 or axes.tolist() != [0]:
                raise ValueError("Unproved GatherND range dimension squeeze")
        shape = node(squeeze.input[0], "Shape")
        if list(shape.input) != ["attention_mask"] or attributes(shape) != {
            "start": axis,
            "end": axis + 1,
        }:
            raise ValueError("GatherND range must span the corresponding mask axis")

    if attributes(gather) not in ({}, {"batch_dims": 0}):
        raise ValueError("GatherND padding requires batch_dims=0")
    cast = node(gather.input[0], "Cast")
    if list(cast.input) != ["attention_mask"] or attributes(cast) != {
        "to": TensorProto.BOOL
    }:
        raise ValueError("GatherND padding must read the boolean attention_mask")
    concat = node(gather.input[1], "Concat")
    if len(concat.input) != MASK_RANK or attributes(concat) not in (
        {"axis": -1},
        {"axis": 4},
    ):
        raise ValueError("GatherND padding requires final-axis [batch, key] pairs")
    expanded = []
    for name in concat.input:
        wrapper = node(name, "Unsqueeze")
        axes = _array(graph, producers, wrapper.input[1])
        if axes is None or axes.dtype != np.int64 or axes.tolist() not in ([-1], [4]):
            raise ValueError("GatherND index pairs require a new final axis")
        expanded.append(node(wrapper.input[0], "Expand"))
    if expanded[0].input[1] != expanded[1].input[1]:
        raise ValueError("GatherND index ranges must share their broadcast shape")
    shape = node(expanded[0].input[1], "Shape")
    if attributes(shape) not in ({}, {"start": 0}):
        raise ValueError("GatherND broadcast shape must preserve all dimensions")
    broadcast = node(shape.input[0], "Max")
    views = [item.input[0] for item in expanded]
    if len(broadcast.input) != MASK_RANK or set(broadcast.input) != set(views):
        raise ValueError("GatherND broadcast shape must come from its index ranges")
    # Exact exporter axes prove [B,1,1,1] and [1,1,1,K].
    index_range(unsqueeze(unsqueeze(views[0], [3]), [1, 2]), 0)
    index_range(unsqueeze(unsqueeze(views[1], [2]), [0, 1]), 1)


def mask_contract(graph, producers, name):
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
        if node.op_type == "GatherND":
            _gathered_padding(graph, producers, node)
            return {"padding"}
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
    inner_radius, padding_fill, _, _ = mask_contract(graph, producers, mask.input[2])
    if inner_radius != -1 or predicate(condition.input[0]) != {"window"}:
        raise ValueError("Unsupported nested local attention mask")
    return window[0], padding_fill, first, ancestors


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


def prune_unused(graph):
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
