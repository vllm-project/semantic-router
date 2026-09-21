"""CPU reference tests for dimension-copy shape rewrites; no model or GPU."""

import copy
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from reshape_dimensions import copy_reshape_dimensions


def fixture():
    tensors = [
        numpy_helper.from_array(
            np.arange(72, dtype=np.float32).reshape(3, 24) / 100, "weight"
        ),
        numpy_helper.from_array(np.array([3, -1, 2], np.int64), "qkv_tail"),
        numpy_helper.from_array(np.array([-1], np.int64), "output_tail"),
    ]
    inputs = [
        helper.make_tensor_value_info(
            "hidden", TensorProto.FLOAT, ["batch", "sequence", 3]
        ),
        helper.make_tensor_value_info("mask", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info(
            "attention", TensorProto.FLOAT, ["batch", 4, "sequence", 2]
        ),
    ]
    nodes = [
        helper.make_node("Shape", ["mask"], ["batch_shape"], start=0, end=1),
        helper.make_node("Shape", ["mask"], ["sequence_shape"], start=1, end=2),
        helper.make_node(
            "Concat",
            ["batch_shape", "sequence_shape", "qkv_tail"],
            ["qkv_shape"],
            axis=0,
        ),
        helper.make_node(
            "Concat",
            ["batch_shape", "sequence_shape", "output_tail"],
            ["output_shape"],
            axis=0,
        ),
        helper.make_node("MatMul", ["hidden", "weight"], ["qkv"]),
        helper.make_node("Reshape", ["qkv", "qkv_shape"], ["qkv_output"], allowzero=1),
        helper.make_node(
            "Transpose", ["attention"], ["attention_transposed"], perm=[0, 2, 1, 3]
        ),
        helper.make_node(
            "Reshape",
            ["attention_transposed", "output_shape"],
            ["attention_output"],
            allowzero=1,
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "qkv_output", TensorProto.FLOAT, ["batch", "sequence", 3, 4, 2]
        ),
        helper.make_tensor_value_info(
            "attention_output", TensorProto.FLOAT, ["batch", "sequence", 8]
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes, "dimension_copy", inputs, outputs, initializer=tensors
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def producer(graph, value):
    return next(n for n in graph.node if value in n.output)


class ReshapeDimensions(unittest.TestCase):
    def test_dynamic_dimensions_preserve_values_and_declared_input_contract(self):
        original = fixture()
        fixed = copy.deepcopy(original)
        self.assertEqual(copy_reshape_dimensions(fixed.graph), 2)
        onnx.checker.check_model(fixed, full_check=True)
        self.assertEqual(fixed.graph.input, original.graph.input)
        self.assertEqual(fixed.graph.output, original.graph.output)
        self.assertEqual(fixed.graph.initializer[:3], original.graph.initializer)
        for batch, length in [(1, 1), (2, 7), (3, 513)]:
            with self.subTest(batch=batch, length=length):
                feeds = {
                    "hidden": np.arange(batch * length * 3, dtype=np.float32).reshape(
                        batch, length, 3
                    ),
                    "mask": np.ones((batch, length), np.int64),
                    "attention": np.arange(
                        batch * 4 * length * 2, dtype=np.float32
                    ).reshape(batch, 4, length, 2),
                }
                expected = ReferenceEvaluator(original).run(None, feeds)
                actual = ReferenceEvaluator(fixed).run(None, feeds)
                for before, after in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(before, after)
        self.assertEqual(copy_reshape_dimensions(fixed.graph), 0)
        self.assertEqual(
            [numpy_helper.to_array(t).tolist() for t in fixed.graph.initializer[-2:]],
            [[0, 0, 3, -1, 2], [0, 0, -1]],
        )
        for n in fixed.graph.node:
            if n.op_type == "Reshape":
                self.assertEqual(
                    next(a.i for a in n.attribute if a.name == "allowzero"), 0
                )

    def test_every_consumer_must_copy_the_same_dimensions(self):
        for kind in ["unrelated", "different_dimensions", "data_slot", "public_output"]:
            with self.subTest(kind=kind):
                graph = fixture().graph
                original_target = copy.deepcopy(producer(graph, "qkv_shape"))
                if kind == "unrelated":
                    graph.node.append(
                        helper.make_node("Identity", ["qkv_shape"], ["extra"])
                    )
                elif kind == "different_dimensions":
                    graph.input.append(
                        helper.make_tensor_value_info(
                            "other",
                            TensorProto.FLOAT,
                            ["different_batch", "sequence", 24],
                        )
                    )
                    graph.node.append(
                        helper.make_node("Reshape", ["other", "qkv_shape"], ["extra"])
                    )
                elif kind == "data_slot":
                    graph.node.append(
                        helper.make_node(
                            "Reshape", ["qkv_shape", "output_tail"], ["extra"]
                        )
                    )
                else:
                    graph.output.append(
                        helper.make_tensor_value_info(
                            "qkv_shape", TensorProto.INT64, [5]
                        )
                    )
                self.assertEqual(copy_reshape_dimensions(graph), 1)
                self.assertEqual(producer(graph, "qkv_shape"), original_target)

    def test_control_flow_implicit_consumers_are_not_ignored(self):
        graph = fixture().graph
        branch = helper.make_graph(
            [helper.make_node("Identity", ["qkv_shape"], ["captured"])],
            "capture",
            [],
            [helper.make_tensor_value_info("captured", TensorProto.INT64, [5])],
        )
        graph.node.append(
            helper.make_node(
                "If",
                ["condition"],
                ["branch_result"],
                then_branch=branch,
                else_branch=branch,
            )
        )
        before = graph.SerializeToString()
        self.assertEqual(copy_reshape_dimensions(graph), 0)
        self.assertEqual(graph.SerializeToString(), before)

    def test_unknown_operator_cannot_be_proven_by_intermediate_value_info(self):
        graph = fixture().graph
        node = producer(graph, "qkv")
        node.op_type = "Unproven"
        node.domain = "other"
        graph.value_info.append(
            helper.make_tensor_value_info(
                "qkv", TensorProto.FLOAT, ["batch", "sequence", 24]
            )
        )
        self.assertEqual(copy_reshape_dimensions(graph), 1)
        self.assertEqual(producer(graph, "qkv_shape").op_type, "Concat")

    def test_shared_target_can_serve_different_inferred_tail_dimensions(self):
        graph = fixture().graph
        graph.input.append(
            helper.make_tensor_value_info(
                "other", TensorProto.FLOAT, ["batch", "sequence", 36]
            )
        )
        graph.node.append(
            helper.make_node("Reshape", ["other", "qkv_shape"], ["extra"], allowzero=1)
        )
        self.assertEqual(copy_reshape_dimensions(graph), 2)
        self.assertEqual(producer(graph, "extra").input[1], "qkv_shape")
        self.assertEqual(
            next(
                a.i for a in producer(graph, "extra").attribute if a.name == "allowzero"
            ),
            0,
        )

    def test_missing_or_reordered_dimension_provenance_is_not_a_copy(self):
        for kind in ["missing", "different_symbol", "reordered", "late_shape"]:
            with self.subTest(kind=kind):
                graph = fixture().graph
                if kind == "missing":
                    graph.input[0].type.tensor_type.shape.dim[0].Clear()
                elif kind == "different_symbol":
                    graph.input[0].type.tensor_type.shape.dim[0].dim_param = "other"
                elif kind == "reordered":
                    node = producer(graph, "qkv_shape")
                    node.input[0], node.input[1] = node.input[1], node.input[0]
                else:
                    node = producer(graph, "qkv_shape")
                    node.input[:] = ["batch_shape", "qkv_tail", "sequence_shape"]
                self.assertEqual(copy_reshape_dimensions(graph), 1)
                self.assertEqual(producer(graph, "qkv_shape").op_type, "Concat")

    def test_literal_zero_target_retains_allowzero_one_and_empty_result(self):
        model = fixture()
        graph = model.graph
        graph.input[0].CopyFrom(
            helper.make_tensor_value_info(
                "hidden", TensorProto.FLOAT, ["batch", "sequence", 0, 2]
            )
        )
        linear = producer(graph, "qkv")
        linear.op_type = "Identity"
        linear.input[:] = ["hidden"]
        graph.initializer[1].CopyFrom(
            numpy_helper.from_array(np.array([0, 2], np.int64), "qkv_tail")
        )
        graph.output[0].CopyFrom(
            helper.make_tensor_value_info(
                "qkv_output", TensorProto.FLOAT, ["batch", "sequence", 0, 2]
            )
        )
        original = copy.deepcopy(model)
        self.assertEqual(copy_reshape_dimensions(graph), 1)
        self.assertEqual(
            producer(graph, "qkv_shape"), producer(original.graph, "qkv_shape")
        )
        feeds = {
            "hidden": np.empty((2, 3, 0, 2), np.float32),
            "mask": np.ones((2, 3), np.int64),
            "attention": np.zeros((2, 4, 3, 2), np.float32),
        }
        self.assertEqual(
            ReferenceEvaluator(model).run(None, feeds)[0].shape, (2, 3, 0, 2)
        )

    def test_overridable_initializer_and_dynamic_tail_are_not_constants(self):
        for kind in ["overridable", "dynamic"]:
            with self.subTest(kind=kind):
                graph = fixture().graph
                graph.input.append(
                    helper.make_tensor_value_info("qkv_tail", TensorProto.INT64, [3])
                )
                if kind == "dynamic":
                    del graph.initializer[1]
                self.assertEqual(copy_reshape_dimensions(graph), 1)
                self.assertEqual(producer(graph, "qkv_shape").op_type, "Concat")

    def test_dynamic_split_sizes_cannot_be_assumed_equal(self):
        graph = fixture().graph
        graph.input[0].CopyFrom(
            helper.make_tensor_value_info(
                "hidden", TensorProto.FLOAT, [4, "sequence", 3]
            )
        )
        graph.input[1].type.tensor_type.shape.dim[0].dim_value = 2
        graph.input[2].type.tensor_type.shape.dim[0].dim_value = 2
        graph.input.append(
            helper.make_tensor_value_info("sizes", TensorProto.INT64, [2])
        )
        graph.node.insert(
            0,
            helper.make_node(
                "Split", ["hidden", "sizes"], ["piece", "unused_piece"], axis=0
            ),
        )
        producer(graph, "qkv").input[0] = "piece"
        self.assertEqual(copy_reshape_dimensions(graph), 1)
        self.assertEqual(producer(graph, "qkv_shape").op_type, "Concat")

    def test_attention_slice_squeeze_and_custom_output_follow_query_shape(self):
        graph = fixture().graph
        graph.initializer.extend(
            [
                numpy_helper.from_array(np.array([0], np.int64), "start"),
                numpy_helper.from_array(np.array([1], np.int64), "end"),
                numpy_helper.from_array(np.array([2], np.int64), "axis"),
            ]
        )
        qkv_view = producer(graph, "qkv_output")
        qkv_view.output[0] = "view"
        graph.node.extend(
            [
                helper.make_node(
                    "Slice", ["view", "start", "end", "axis"], ["one_query"]
                ),
                helper.make_node("Squeeze", ["one_query", "axis"], ["query_bshd"]),
                helper.make_node(
                    "Transpose", ["query_bshd"], ["query"], perm=[0, 2, 1, 3]
                ),
                helper.make_node(
                    "CKFlashAttention",
                    ["query", "query", "query"],
                    ["ck_output"],
                    domain="com.ck",
                ),
            ]
        )
        producer(graph, "attention_transposed").input[0] = "ck_output"
        self.assertEqual(copy_reshape_dimensions(graph), 2)


if __name__ == "__main__":
    unittest.main()
