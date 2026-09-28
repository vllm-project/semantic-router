"""Keep an FP16 classifier's masked mean finite at long sequence lengths.

Only the typed, mask-derived ReduceSum/Div path into a dense head is changed.
The encoder, mask multiplication, head, and public tensor dtypes stay intact.
Apply this after any whole-model FP16 conversion so its FP32 reduction and
division are not narrowed again.
"""

from collections import defaultdict
from dataclasses import dataclass

from onnx import NodeProto, TensorProto, helper, numpy_helper


def _standard(node, op):
    return (
        node is not None
        and node.op_type == op
        and node.domain in ("", "ai.onnx")
        and len(node.output) == 1
    )


def _attribute(node, name, default=None):
    return next(
        (helper.get_attribute_value(a) for a in node.attribute if a.name == name),
        default,
    )


class _GraphView:
    def __init__(self, graph):
        self.nodes = list(graph.node)
        self.producers = {out: node for node in self.nodes for out in node.output}
        self.consumers = defaultdict(list)
        for node in self.nodes:
            for name in node.input:
                self.consumers[name].append(node)
        self.initializers = {tensor.name: tensor for tensor in graph.initializer}
        self.tensors = {}
        for value in (*graph.input, *graph.value_info, *graph.output):
            if value.type.HasField("tensor_type"):
                tensor = value.type.tensor_type
                if tensor.HasField("shape"):
                    shape = [
                        (
                            dim.dim_value
                            if dim.HasField("dim_value")
                            else dim.dim_param or None
                        )
                        for dim in tensor.shape.dim
                    ]
                    self.tensors[value.name] = (tensor.elem_type, shape)
        for tensor in graph.initializer:
            self.tensors[tensor.name] = (tensor.data_type, list(tensor.dims))
        self.outputs = {value.name for value in graph.output}
        self.names = set(self.tensors) | set(self.producers)
        self.names.update(node.name for node in self.nodes)

    def typed(self, name, dtype, rank):
        info = self.tensors.get(name)
        return info is not None and info[0] == dtype and len(info[1]) == rank

    def axes(self, node, rank):
        axes = _attribute(node, "axes")
        # ReduceSum/Unsqueeze have an optional second, axes input.
        if axes is None and len(node.input) == 2:  # noqa: PLR2004
            tensor = self.initializers.get(node.input[1])
            if tensor is None:
                producer = self.producers.get(node.input[1])
                if _standard(producer, "Constant"):
                    tensor = _attribute(producer, "value")
            if (
                tensor is None
                or tensor.data_type != TensorProto.INT64
                or list(tensor.dims) != [1]
            ):
                return None
            try:
                axes = numpy_helper.to_array(tensor).tolist()
            except (ValueError, OSError):
                return None
        if axes is None or len(axes) != 1 or not -rank <= axes[0] < rank:
            return None
        return tuple(axis % rank for axis in axes)

    def fresh(self, base):
        name, suffix = base, 0
        while name in self.names:
            suffix += 1
            name = f"{base}_{suffix}"
        self.names.add(name)
        return name


@dataclass
class _MeanPool:
    divide: NodeProto
    numerator: NodeProto
    denominator_cast: NodeProto
    count: NodeProto


def _masked_product(view, product):
    """Return the original rank-two integer mask, accepting either Mul order."""
    if (
        not _standard(product, "Mul")
        or len(product.input) != 2  # noqa: PLR2004 - binary op
    ):
        return None
    if not view.typed(product.output[0], TensorProto.FLOAT16, 3):
        return None
    for hidden, mask in (product.input, list(reversed(product.input))):
        cast = view.producers.get(mask)
        if (
            not view.typed(hidden, TensorProto.FLOAT16, 3)
            or not _standard(cast, "Cast")
            or len(cast.input) != 1
            or _attribute(cast, "to") != TensorProto.FLOAT16
        ):
            continue
        unsqueeze = view.producers.get(cast.input[0])
        if (
            not _standard(unsqueeze, "Unsqueeze")
            or len(unsqueeze.input) not in (1, 2)
            or view.axes(unsqueeze, 3) != (2,)
        ):
            continue
        original_mask = unsqueeze.input[0]
        if not view.typed(original_mask, TensorProto.INT64, 2):
            continue
        hidden_length = view.tensors[hidden][1][1]
        mask_length = view.tensors[original_mask][1][1]
        # A single-token mask broadcasting across the sequence is not this
        # mean: its denominator would count one token, rather than the sequence.
        if (
            hidden_length is not None
            and mask_length is not None
            and hidden_length != mask_length
        ):
            continue
        return original_mask
    return None


def _match_mean_pool(view, divide):
    if (
        not _standard(divide, "Div")
        or len(divide.input) != 2  # noqa: PLR2004 - binary op
        or not view.typed(divide.output[0], TensorProto.FLOAT16, 2)
    ):
        return None
    # This repair belongs to a sequence classifier, not an arbitrary attention
    # normalization or a reduction feeding the encoder.
    heads = view.consumers[divide.output[0]]
    if not any(
        _standard(head, "Gemm")
        and len(head.input) >= 2  # noqa: PLR2004 - Gemm inputs A/B, optional C
        and head.input[0] == divide.output[0]
        and _attribute(head, "transA", 0) == 0
        and view.typed(head.input[1], TensorProto.FLOAT16, 2)
        for head in heads
    ):
        return None
    numerator = view.producers.get(divide.input[0])
    denominator_cast = view.producers.get(divide.input[1])
    if (
        not _standard(numerator, "ReduceSum")
        or len(numerator.input) not in (1, 2)
        or view.axes(numerator, 3) != (1,)
        or _attribute(numerator, "keepdims", 1) != 0
        or not view.typed(numerator.output[0], TensorProto.FLOAT16, 2)
        or not _standard(denominator_cast, "Cast")
        or len(denominator_cast.input) != 1
        or _attribute(denominator_cast, "to") != TensorProto.FLOAT16
        or not view.typed(denominator_cast.output[0], TensorProto.FLOAT16, 2)
    ):
        return None
    mask = _masked_product(view, view.producers.get(numerator.input[0]))
    count = view.producers.get(denominator_cast.input[0])
    if (
        mask is None
        or not _standard(count, "ReduceSum")
        or len(count.input) not in (1, 2)
        or count.input[0] != mask
        or view.axes(count, 2) != (1,)
        or _attribute(count, "keepdims", 1) != 1
        or not view.typed(count.output[0], TensorProto.INT64, 2)
    ):
        return None
    return _MeanPool(divide, numerator, denominator_cast, count)


def stabilize_mean_pooling(graph):
    """Repair recognized FP16 masked means in place; return the number changed.

    Required type/rank evidence comes from graph inputs, initializers and
    value_info. Unrecognized, untyped and FP32 paths are left byte-for-byte
    unchanged. Shared FP16 intermediates remain available to other consumers.
    The original mean output name/type is preserved, making the pass idempotent.
    """
    view = _GraphView(graph)
    pools = [pool for node in view.nodes if (pool := _match_mean_pool(view, node))]
    if not pools:
        return 0

    replacements, removed, new_info = {}, set(), []
    for pool in pools:
        original_mean = pool.divide.output[0]
        prefix = original_mean + "__stable_pool"
        product = view.fresh(prefix + "_product_fp32")
        total = view.fresh(prefix + "_sum_fp32")
        count = view.fresh(prefix + "_count_fp32")
        mean = view.fresh(prefix + "_mean_fp32")
        nodes = [
            helper.make_node(
                "Cast",
                [pool.numerator.input[0]],
                [product],
                name=view.fresh(prefix + "_cast_product"),
                to=TensorProto.FLOAT,
            ),
        ]
        wider_sum = type(pool.numerator)()
        wider_sum.CopyFrom(pool.numerator)
        wider_sum.name = view.fresh(prefix + "_sum")
        wider_sum.input[0] = product
        wider_sum.output[0] = total
        nodes.extend(
            [
                wider_sum,
                helper.make_node(
                    "Cast",
                    [pool.count.output[0]],
                    [count],
                    name=view.fresh(prefix + "_cast_count"),
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Div", [total, count], [mean], name=view.fresh(prefix + "_divide")
                ),
                helper.make_node(
                    "Cast",
                    [mean],
                    [original_mean],
                    name=view.fresh(prefix + "_cast_mean"),
                    to=TensorProto.FLOAT16,
                ),
            ]
        )
        replacements[original_mean] = nodes
        for node in (pool.numerator, pool.denominator_cast):
            output = node.output[0]
            if output not in view.outputs and view.consumers[output] == [pool.divide]:
                removed.add(output)
        for name, source in (
            (product, pool.numerator.input[0]),
            (total, pool.numerator.output[0]),
            (count, pool.denominator_cast.output[0]),
            (mean, original_mean),
        ):
            new_info.append(
                helper.make_tensor_value_info(
                    name, TensorProto.FLOAT, view.tensors[source][1]
                )
            )

    rewritten = []
    for node in view.nodes:
        output = node.output[0] if len(node.output) == 1 else None
        if output in replacements:
            rewritten.extend(replacements[output])
        elif output not in removed:
            rewritten.append(node)
    kept_info = [value for value in graph.value_info if value.name not in removed]
    del graph.node[:]
    graph.node.extend(rewritten)
    del graph.value_info[:]
    graph.value_info.extend(kept_info + new_info)
    return len(pools)
