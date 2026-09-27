"""Replace proven Reshape dimension copies without runtime shape concatenation.

Only graph-input dimensions and known operator semantics establish provenance;
intermediate value_info is not proof. Equal symbolic input dimensions retain the
source graph's shared input contract. Unknown provenance leaves the graph alone.
"""

from collections import defaultdict

import numpy as np
from onnx import TensorProto, helper, numpy_helper

BINARY_INPUTS = 2
MATRIX_RANK = 2
SLICE_REQUIRED_INPUTS = 3
SLICE_AXES_INPUTS = 4
SLICE_STEPS_INPUTS = 5


def _attribute(node, name, default=None):
    return next(
        (helper.get_attribute_value(a) for a in node.attribute if a.name == name),
        default,
    )


def _broadcast(left, right):
    if left is None or right is None:
        return None
    rank = max(len(left), len(right))
    result = []
    for a, b in zip(
        (1,) * (rank - len(left)) + left,
        (1,) * (rank - len(right)) + right,
        strict=True,
    ):
        if a == 1:
            result.append(b)
        elif b == 1 or (a is not None and a == b):
            result.append(a)
        else:
            result.append(None)
    return tuple(result)


class _Dimensions:
    def __init__(self, graph):
        self.producers = {v: n for n in graph.node for v in n.output}
        self.initializers = {t.name: t for t in graph.initializer}
        self.inputs = {}
        self._shapes = {}
        self._values = {}
        for value in graph.input:
            tensor = value.type.tensor_type
            if value.type.HasField("tensor_type") and tensor.HasField("shape"):
                self.inputs[value.name] = tuple(
                    d.dim_value if d.HasField("dim_value") else d.dim_param or None
                    for d in tensor.shape.dim
                )

    def integers(self, name):
        if name in self.inputs:
            return None  # Graph-input initializers may be overridden by callers.
        tensor = self.initializers.get(name)
        node = self.producers.get(name)
        if (
            tensor is None
            and node is not None
            and node.op_type == "Constant"
            and not node.domain
        ):
            tensor = _attribute(node, "value")
        if (
            tensor is None
            or tensor.data_type != TensorProto.INT64
            or len(tensor.dims) > 1
        ):
            return None
        return tuple(int(v) for v in numpy_helper.to_array(tensor).reshape(-1))

    def shape_values(self, name):
        if name not in self._values:
            self._values[name] = self._shape_values(name)
        return self._values[name]

    def _shape_values(self, name):
        values = self.integers(name)
        if values is not None:
            return values
        node = self.producers.get(name)
        if node is None or node.domain:
            return None
        if node.op_type == "Shape" and len(node.input) == 1:
            source = self.shape(node.input[0])
            if source is not None:
                return source[
                    _attribute(node, "start", 0) : _attribute(node, "end", len(source))
                ]
        if node.op_type == "Concat" and _attribute(node, "axis") == 0:
            pieces = [self.shape_values(v) for v in node.input]
            if all(piece is not None for piece in pieces):
                return tuple(v for piece in pieces for v in piece)
        return None

    def shape(self, name):
        if name not in self._shapes:
            self._shapes[name] = self._shape(name)
        return self._shapes[name]

    def _shape(self, name):
        if name in self.inputs:
            return self.inputs[name]
        if name in self.initializers:
            return tuple(self.initializers[name].dims)
        node = self.producers.get(name)
        if node is None:
            return None
        if node.domain == "com.ck" and node.op_type == "CKFlashAttention":
            # The installed custom op allocates output with the exact Q shape.
            return self.shape(node.input[0]) if len(node.input) in (3, 4) else None
        if node.domain:
            return None
        if node.op_type == "Constant":
            tensor = _attribute(node, "value")
            return tuple(tensor.dims) if tensor is not None else None
        shapes = [self.shape(v) for v in node.input]
        if not shapes or shapes[0] is None:
            return None
        first = shapes[0]
        rank = len(first)
        if (
            node.op_type
            in {"Identity", "Cast", "Neg", "Sin", "Cos", "Erf", "LayerNormalization"}
            and name == node.output[0]
        ):
            return first
        if (
            node.op_type in {"Add", "Sub", "Mul", "Div"}
            and len(shapes) == BINARY_INPUTS
        ):
            return _broadcast(*shapes)
        if node.op_type == "MatMul" and len(shapes) == BINARY_INPUTS:
            right = shapes[1]
            if rank >= MATRIX_RANK and right is not None and len(right) >= MATRIX_RANK:
                batch = _broadcast(first[:-2], right[:-2])
                return (*batch, first[-2], right[-1])
        if node.op_type == "Transpose":
            perm = _attribute(node, "perm", list(reversed(range(rank))))
            if sorted(perm) == list(range(rank)):
                return tuple(first[i] for i in perm)
        if node.op_type == "Reshape" and len(node.input) == BINARY_INPUTS:
            target = self.shape_values(node.input[1])
            if target is not None:
                allowzero = _attribute(node, "allowzero", 0)
                return tuple(
                    (
                        first[i]
                        if d == 0 and not allowzero and i < rank
                        else None if d == -1 else d
                    )
                    for i, d in enumerate(target)
                )
        if (
            node.op_type == "Gather"
            and len(shapes) == BINARY_INPUTS
            and shapes[1] is not None
            and rank
        ):
            axis = _attribute(node, "axis", 0) % rank
            return first[:axis] + shapes[1] + first[axis + 1 :]
        if (
            node.op_type == "Concat"
            and all(s is not None and len(s) == rank for s in shapes)
            and rank
        ):
            axis = _attribute(node, "axis", 0) % rank
            result = list(first)
            for i in range(rank):
                values = [s[i] for s in shapes]
                result[i] = (
                    sum(values)
                    if i == axis and all(isinstance(v, int) for v in values)
                    else (
                        values[0]
                        if i != axis and all(v == values[0] for v in values)
                        else None
                    )
                )
            return tuple(result)
        if node.op_type == "Slice" and len(node.input) >= SLICE_REQUIRED_INPUTS:
            starts, ends = (self.integers(v) for v in node.input[1:3])
            if starts is None or ends is None or len(starts) != len(ends):
                return None
            axes = (
                self.integers(node.input[3])
                if len(node.input) >= SLICE_AXES_INPUTS and node.input[3]
                else tuple(range(len(starts)))
            )
            steps = (
                self.integers(node.input[4])
                if len(node.input) >= SLICE_STEPS_INPUTS and node.input[4]
                else (1,) * len(starts)
            )
            if (
                axes is None
                or steps is None
                or len(axes) != len(starts)
                or len(steps) != len(starts)
            ):
                return None
            result = list(first)
            for axis, start, end, step in zip(axes, starts, ends, steps, strict=True):
                if not -rank <= axis < rank or step == 0:
                    return None
                dimension_index = axis % rank
                dimension = first[dimension_index]
                result[dimension_index] = (
                    len(range(*slice(start, end, step).indices(dimension)))
                    if isinstance(dimension, int)
                    else None
                )
            return tuple(result)
        if node.op_type == "Split" and rank:
            axis = _attribute(node, "axis", 0) % rank
            parts = (
                self.integers(node.input[1])
                if len(node.input) == BINARY_INPUTS
                else _attribute(node, "split")
            )
            result = list(first)
            if parts is not None and len(parts) == len(node.output):
                result[axis] = parts[list(node.output).index(name)]
            elif (
                len(node.input) == 1
                and isinstance(first[axis], int)
                and first[axis] % len(node.output) == 0
            ):
                result[axis] = first[axis] // len(node.output)
            else:
                result[axis] = None
            return tuple(result)
        if node.op_type in {"Squeeze", "Unsqueeze"}:
            axes = (
                self.integers(node.input[1])
                if len(node.input) == BINARY_INPUTS
                else _attribute(node, "axes")
            )
            if axes is None:
                return None
            output_rank = rank + len(axes) if node.op_type == "Unsqueeze" else rank
            axes = {i % output_rank for i in axes}
            if node.op_type == "Squeeze":
                if any(first[i] != 1 for i in axes):
                    return None
                return tuple(d for i, d in enumerate(first) if i not in axes)
            source = iter(first)
            return tuple(1 if i in axes else next(source) for i in range(output_rank))
        return None


def copy_reshape_dimensions(graph):
    """Return the number of shared shape Concat nodes safely replaced.

    Every dynamic prefix dimension must copy the corresponding data dimension
    for every consumer. Literal zero constants are excluded because changing
    allowzero would change their meaning. Shared targets with other consumers
    or graph outputs are never modified.
    """
    # Control-flow subgraphs can capture outer values without a node input.
    # Their implicit consumers require a separate proof; leave these graphs alone.
    if any(a.type in (a.GRAPH, a.GRAPHS) for n in graph.node for a in n.attribute):
        return 0
    dimensions = _Dimensions(graph)
    consumers = defaultdict(list)
    for node in graph.node:
        for slot, value in enumerate(node.input):
            consumers[value].append((node, slot))
    outputs = {value.name for value in graph.output}
    replacements = []
    for node in graph.node:
        if (
            node.domain
            or node.op_type != "Concat"
            or len(node.output) != 1
            or _attribute(node, "axis") != 0
        ):
            continue
        output = node.output[0]
        uses = consumers[output]
        if output in outputs or not uses:
            continue
        prefix, tail = [], []
        for name in node.input:
            source = dimensions.producers.get(name)
            if (
                source is not None
                and not source.domain
                and source.op_type == "Shape"
                and not tail
            ):
                values = dimensions.shape_values(name)
                if values is None or not values or any(v is None for v in values):
                    break
                prefix.extend(values)
            else:
                values = dimensions.integers(name)
                if (
                    values is None
                    or not values
                    or any(v == 0 or v < -1 for v in values)
                ):
                    break
                tail.extend(values)
        else:
            if not prefix or not tail or tail.count(-1) > 1:
                continue
            for use, slot in uses:
                data = dimensions.shape(use.input[0])
                if (
                    use.domain
                    or use.op_type != "Reshape"
                    or slot != 1
                    or len(use.input) != BINARY_INPUTS
                    or data is None
                    or data[: len(prefix)] != tuple(prefix)
                ):
                    break
            else:
                replacements.append((node, uses, [0] * len(prefix) + tail))
    for node, uses, target in replacements:
        graph.initializer.append(
            numpy_helper.from_array(np.array(target, dtype=np.int64), node.output[0])
        )
        for use, _slot in uses:
            attributes = [a for a in use.attribute if a.name != "allowzero"]
            del use.attribute[:]
            use.attribute.extend(attributes)
            use.attribute.append(helper.make_attribute("allowzero", 0))
    removed = {node.output[0] for node, _uses, _target in replacements}
    kept = [node for node in graph.node if not any(v in removed for v in node.output)]
    del graph.node[:]
    graph.node.extend(kept)
    return len(replacements)
