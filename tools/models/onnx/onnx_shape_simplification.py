"""Simplify proven batch-copy Reshape targets without changing tensor data."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

_RESHAPE_INPUT_COUNT = 2
_MIN_CONCAT_INPUTS = 2


def simplify_batch_reshapes(model) -> dict:
    """Replace a proven first-dimension copy with ONNX Reshape's zero syntax.

    Match structure and declared dimensions, never operator names. The source
    shape must be Concat(Shape(tensor)[0:1], constant nonzero dimensions), and
    that tensor's first dimension must equal the reshaped data's first dimension.
    Unknown or different dimension identities are left unchanged. Original
    initializers, including external tensor references, are always preserved.
    """
    graph = model.graph
    producers = {value: node for node in graph.node for value in node.output}
    initializers = {value.name: value for value in graph.initializer}
    dimensions = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        if value.type.HasField("tensor_type"):
            dimensions[value.name] = [
                (
                    ("symbol", dim.dim_param)
                    if dim.dim_param
                    else (
                        ("static", dim.dim_value)
                        if dim.HasField("dim_value") and dim.dim_value > 0
                        else None
                    )
                )
                for dim in value.type.tensor_type.shape.dim
            ]

    def attributes(node):
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    def constant(name):
        tensor = initializers.get(name)
        node = producers.get(name)
        if (
            tensor is None
            and node is not None
            and not node.domain
            and node.op_type == "Constant"
        ):
            tensor = attributes(node).get("value")
        if (
            not isinstance(tensor, TensorProto)
            or tensor.data_type != TensorProto.INT64
            or tensor.data_location == TensorProto.EXTERNAL
            or len(tensor.dims) != 1
        ):
            return None
        return numpy_helper.to_array(tensor).tolist()

    names = set(producers) | set(initializers) | {x.name for x in graph.input}
    targets = {}
    changed = []
    removable = set()
    for reshape in graph.node:
        if (
            reshape.domain
            or reshape.op_type != "Reshape"
            or len(reshape.input) != _RESHAPE_INPUT_COUNT
        ):
            continue
        concat = producers.get(reshape.input[1])
        if (
            concat is None
            or concat.domain
            or concat.op_type != "Concat"
            or attributes(concat) != {"axis": 0}
            or len(concat.input) < _MIN_CONCAT_INPUTS
        ):
            continue
        shape = producers.get(concat.input[0])
        if (
            shape is None
            or shape.domain
            or shape.op_type != "Shape"
            or attributes(shape).get("start", 0) != 0
            or attributes(shape).get("end") != 1
            or len(shape.input) != 1
        ):
            continue
        source_dims = dimensions.get(shape.input[0], [])
        data_dims = dimensions.get(reshape.input[0], [])
        if (
            not source_dims
            or not data_dims
            or source_dims[0] is None
            or source_dims[0] != data_dims[0]
        ):
            continue
        pieces = [constant(value) for value in concat.input[1:]]
        if any(value is None for value in pieces):
            continue
        tail = [item for piece in pieces for item in piece]
        if not tail or tail.count(-1) > 1 or any(x == 0 or x < -1 for x in tail):
            continue
        attrs = attributes(reshape)
        if set(attrs) - {"allowzero"} or attrs.get("allowzero", 0) not in (0, 1):
            continue
        values = (0, *tail)
        if values not in targets:
            name = reshape.input[1] + "_copy_batch"
            while name in names:
                name += "_"
            names.add(name)
            graph.initializer.append(
                numpy_helper.from_array(np.array(values, dtype=np.int64), name)
            )
            targets[values] = name
        reshape.input[1] = targets[values]
        reshape.ClearField("attribute")
        reshape.attribute.extend([helper.make_attribute("allowzero", 0)])
        changed.append(reshape.name)
        removable.update((*concat.output, *shape.output))

    # Remove only the matched shape machinery, and only when no consumer remains.
    removed = []
    while True:
        used = {x for node in graph.node for x in node.input}
        used.update(x.name for x in graph.output)
        dead = [
            node
            for node in graph.node
            if node.output
            and all(x in removable and x not in used for x in node.output)
        ]
        if not dead:
            break
        for node in dead:
            removed.extend(node.output)
            graph.node.remove(node)
    infos = [value for value in graph.value_info if value.name not in removed]
    graph.ClearField("value_info")
    graph.value_info.extend(infos)
    return {"rewritten_reshapes": len(changed), "removed_shape_values": sorted(removed)}
