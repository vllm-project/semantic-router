"""Specialize token input dimensions and fold only bounded shape-derived values.

Declared ONNX dimensions and shape transformations are the proof boundary.
Runtime token/mask/activation values and control-flow bodies are never evaluated.
"""

import copy
import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

TOKEN_INPUT_RANK = 2
SLICE_STARTS = 1
SLICE_ENDS = 2
SLICE_AXES = 3
SLICE_STEPS = 4
MAX_VALUE_ELEMENTS = 1024
DEFAULT_CONSTANT_BYTES = 8 * 1024 * 1024


class _UnprovenError(ValueError):
    pass


def _attribute(node, name, default=None):
    return next(
        (helper.get_attribute_value(a) for a in node.attribute if a.name == name),
        default,
    )


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _bind_inputs(model, batch, sequence):
    _positive(batch, "Batch size")
    if sequence is not None:
        _positive(sequence, "Sequence length")
    names = [value.name for value in model.graph.input]
    if len(names) != len(set(names)) or set(names) not in (
        {"input_ids", "attention_mask"},
        {"input_ids", "attention_mask", "position_ids"},
    ):
        raise ValueError(
            "Specialization requires standard token IDs, padding and optional positions"
        )
    bindings = {}
    result = copy.deepcopy(model)
    for value in result.graph.input:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != TensorProto.INT64
            or len(tensor.shape.dim) != TOKEN_INPUT_RANK
        ):
            raise ValueError("Token inputs must be rank-two int64 tensors")
        if value.name == "position_ids" and tensor.shape.dim[0].dim_value != 1:
            raise ValueError("Position IDs must declare one broadcast row")
        for axis, requested in enumerate(
            [1 if value.name == "position_ids" else batch, sequence]
        ):
            dim = tensor.shape.dim[axis]
            if dim.HasField("dim_value") and dim.dim_value < 1:
                raise ValueError("Fixed input dimensions must be positive")
            if requested is None:
                continue
            if dim.HasField("dim_value") and dim.dim_value != requested:
                raise ValueError("Cannot change a pre-existing fixed input contract")
            if dim.dim_param:
                symbol = dim.dim_param
                if symbol in bindings and bindings[symbol] != requested:
                    raise ValueError("Contradictory shared input dimension")
                bindings[symbol] = requested
            dim.dim_value = requested
    for value in list(result.graph.value_info) + list(result.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param in bindings:
                dim.dim_value = bindings[dim.dim_param]
    return result, bindings


class _ShapeValues:
    def __init__(self, model):
        self.nodes = {name: node for node in model.graph.node for name in node.output}
        self.initializers = {value.name: value for value in model.graph.initializer}
        self.metadata = {
            value.name: [
                dim.dim_value if dim.HasField("dim_value") else None
                for dim in value.type.tensor_type.shape.dim
            ]
            for value in list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        }
        self.values, self.shapes, self.roots, self.active = {}, {}, {}, set()

    @staticmethod
    def bounded(value):
        value = np.asarray(value)
        if value.size > MAX_VALUE_ELEMENTS or value.dtype.kind not in "biuf":
            raise _UnprovenError("Not a small numeric shape value")
        return value

    def shape(self, name):
        if name in self.shapes:
            return self.shapes[name]
        declared = self.metadata.get(name)
        if declared is not None and all(
            dim is not None and dim > 0 for dim in declared
        ):
            shape = declared
        elif name in self.initializers:
            shape = list(self.initializers[name].dims)
        else:
            node = self.nodes.get(name)
            if node is None or node.domain:
                raise _UnprovenError("Unknown tensor shape")
            if node.op_type == "Transpose":
                source = self.shape(node.input[0])
                permutation = _attribute(
                    node, "perm", list(reversed(range(len(source))))
                )
                if sorted(permutation) != list(range(len(source))):
                    raise _UnprovenError("Invalid transpose")
                shape = [source[axis] for axis in permutation]
            elif node.op_type == "Reshape":
                source = self.shape(node.input[0])
                target = self.value(node.input[1])
                if (
                    target.dtype != np.int64
                    or target.ndim != 1
                    or _attribute(node, "allowzero", 0)
                ):
                    raise _UnprovenError("Unsupported reshape shape")
                shape = [
                    source[axis] if dim == 0 else int(dim)
                    for axis, dim in enumerate(target)
                ]
                if shape.count(-1) > 1 or any(dim <= 0 and dim != -1 for dim in shape):
                    raise _UnprovenError("Invalid reshape dimensions")
                if -1 in shape:
                    divisor = math.prod(dim for dim in shape if dim != -1)
                    if not divisor or math.prod(source) % divisor:
                        raise _UnprovenError("Nonintegral reshape")
                    shape[shape.index(-1)] = math.prod(source) // divisor
                if math.prod(shape) != math.prod(source):
                    raise _UnprovenError("Reshape changes element count")
            elif node.op_type in ("Mul", "Add", "Sub", "Div"):
                shape = list(
                    np.broadcast_shapes(*(tuple(self.shape(i)) for i in node.input))
                )
            else:
                raise _UnprovenError("Shape depends on unsupported computation")
        self.shapes[name] = shape
        return shape

    def value(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.active:
            raise _UnprovenError("Cyclic shape dependency")
        self.active.add(name)
        try:
            return self._value(name)
        finally:
            self.active.remove(name)

    def _value(self, name):
        if name in self.initializers:
            tensor = self.initializers[name]
            if (
                tensor.data_location == TensorProto.EXTERNAL
                or math.prod(tensor.dims) > MAX_VALUE_ELEMENTS
            ):
                raise _UnprovenError(
                    "Initializer payload is not a small inline constant"
                )
            value, roots = numpy_helper.to_array(tensor), set()
        else:
            node = self.nodes.get(name)
            if node is None or node.domain or len(node.output) != 1:
                raise _UnprovenError("Runtime or custom-op value")
            if node.op_type == "Constant":
                tensor = _attribute(node, "value")
                if (
                    not isinstance(tensor, TensorProto)
                    or tensor.data_location == TensorProto.EXTERNAL
                    or math.prod(tensor.dims) > MAX_VALUE_ELEMENTS
                ):
                    raise _UnprovenError("Unsupported constant")
                value, roots = numpy_helper.to_array(tensor), set()
            elif node.op_type == "Shape":
                shape = self.shape(node.input[0])
                value = np.asarray(
                    shape[
                        _attribute(node, "start", 0) : _attribute(
                            node, "end", len(shape)
                        )
                    ],
                    dtype=np.int64,
                )
                roots = {node.input[0]}
            else:
                args = [self.value(i) for i in node.input if i]
                roots = set().union(*(self.roots[i] for i in node.input if i))
                value = self.evaluate(node, args)
        self.values[name] = self.bounded(value)
        self.roots[name] = roots
        return self.values[name]

    def evaluate(self, node, args):
        op = node.op_type
        binary = {
            "And": np.logical_and,
            "Equal": np.equal,
            "Greater": np.greater,
            "LessOrEqual": np.less_equal,
            "Mul": np.multiply,
            "Add": np.add,
            "Sub": np.subtract,
        }
        if op in binary or op in ("Min", "Max", "Div"):
            if (
                math.prod(np.broadcast_shapes(*(a.shape for a in args)))
                > MAX_VALUE_ELEMENTS
            ):
                raise _UnprovenError("Broadcast exceeds shape-value bound")
            if op in binary:
                return binary[op](*args)
            if op == "Div":
                if (
                    any(a.dtype.kind not in "iu" for a in args)
                    or not np.all(args[0] >= 0)
                    or not np.all(args[1] > 0)
                ):
                    raise _UnprovenError("Only nonnegative integer division is proved")
                return args[0] // args[1]
            value = args[0]
            for operand in args[1:]:
                value = (np.minimum if op == "Min" else np.maximum)(value, operand)
            return value
        if op == "Gather":
            axis = _attribute(node, "axis", 0)
            if args[0].size // args[0].shape[axis] * args[1].size > MAX_VALUE_ELEMENTS:
                raise _UnprovenError("Gather exceeds shape-value bound")
            return np.take(args[0], args[1], axis=axis)
        if op == "Concat":
            if sum(a.size for a in args) > MAX_VALUE_ELEMENTS:
                raise _UnprovenError("Concat exceeds shape-value bound")
            return np.concatenate(args, axis=_attribute(node, "axis"))
        if op == "Unsqueeze":
            return np.expand_dims(args[0], tuple(args[1].tolist()))
        if op == "Cast":
            return args[0].astype(
                helper.tensor_dtype_to_np_dtype(_attribute(node, "to"))
            )
        if op == "Slice":
            slices = [slice(None)] * args[0].ndim
            axes = (
                args[SLICE_AXES]
                if len(args) > SLICE_AXES
                else range(len(args[SLICE_STARTS]))
            )
            steps = (
                args[SLICE_STEPS]
                if len(args) > SLICE_STEPS
                else np.ones(len(args[SLICE_STARTS]), dtype=np.int64)
            )
            for start, end, axis, step in zip(
                args[SLICE_STARTS], args[SLICE_ENDS], axes, steps, strict=True
            ):
                if step == 0:
                    raise _UnprovenError("Zero slice step")
                slices[int(axis)] = slice(int(start), int(end), int(step))
            return args[0][tuple(slices)]
        raise _UnprovenError("Runtime-dependent or unsupported operation")


def specialize_token_shapes(
    model, batch_size, sequence_length=None, max_constant_bytes=DEFAULT_CONSTANT_BYTES
):
    """Return a copy with fixed token inputs and a bounded shape-only proof.

    Batch-only mode retains dynamic sequence computation. With a sequence bound,
    only small values whose complete ancestry is constants or known Shape nodes
    are replaced. Initializers and control-flow bodies remain byte-identical.
    """
    _positive(max_constant_bytes, "Constant byte limit")
    result, bindings = _bind_inputs(model, batch_size, sequence_length)
    rows, payload = [], 0
    if sequence_length is not None:
        evaluator = _ShapeValues(result)
        for node in result.graph.node:
            if node.op_type == "Constant" or len(node.output) != 1:
                continue
            try:
                array = evaluator.value(node.output[0])
            except (
                ValueError,
                IndexError,
                KeyError,
                TypeError,
                OverflowError,
                ZeroDivisionError,
            ):
                continue
            roots = evaluator.roots[node.output[0]]
            if not roots:
                continue
            payload += array.nbytes
            if payload > max_constant_bytes:
                raise ValueError("Shape constants exceed the total byte limit")
            rows.append(
                {
                    "node": node.name,
                    "output": node.output[0],
                    "op": node.op_type,
                    "shape_roots": sorted(roots),
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "value": array.tolist(),
                }
            )
            node.CopyFrom(
                helper.make_node(
                    "Constant",
                    [],
                    list(node.output),
                    name=node.name,
                    value=numpy_helper.from_array(array),
                )
            )
    receipt = {
        "fixed_batch_size": batch_size,
        "fixed_sequence_length": sequence_length,
        "dimension_bindings": bindings,
        "folded_nodes": len(rows),
        "constant_payload_bytes": payload,
        "proofs": rows,
        "runtime_qualified": False,
    }
    return result, receipt
