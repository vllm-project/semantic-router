"""Executable ONNX reference tests for stable classifier mean pooling.

Run with the rewriter's existing onnx/numpy dependencies:
    python -m unittest test_stable_pooling
"""

import copy
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from stable_pooling import stabilize_mean_pooling


def mean_classifier(dtype=TensorProto.FLOAT16, swap_product=False):
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    initializers = [
        numpy_helper.from_array(np.array([1], np.int64), "sequence_axis"),
        numpy_helper.from_array(np.array([2], np.int64), "last_axis"),
        numpy_helper.from_array(np.ones((1, 1), np_dtype), "head_weight"),
    ]
    inputs = [
        helper.make_tensor_value_info("encoder_input", dtype, [1, "sequence", 1]),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, [1, "sequence"]
        ),
    ]
    nodes = [
        helper.make_node("Identity", ["encoder_input"], ["hidden"], name="encoder"),
        helper.make_node("Unsqueeze", ["attention_mask", "last_axis"], ["mask_3d"]),
        helper.make_node("Cast", ["mask_3d"], ["floating_mask"], to=dtype),
        helper.make_node(
            "Mul",
            (
                ["floating_mask", "hidden"]
                if swap_product
                else ["hidden", "floating_mask"]
            ),
            ["masked"],
        ),
        helper.make_node("ReduceSum", ["masked", "sequence_axis"], ["sum"], keepdims=0),
        helper.make_node(
            "ReduceSum", ["attention_mask", "sequence_axis"], ["count"], keepdims=1
        ),
        helper.make_node("Cast", ["count"], ["floating_count"], to=dtype),
        helper.make_node("Div", ["sum", "floating_count"], ["mean"]),
        helper.make_node(
            "Gemm", ["mean", "head_weight"], ["logits"], name="head", transB=1
        ),
    ]
    infos = [
        helper.make_tensor_value_info(name, tensor_dtype, shape)
        for name, tensor_dtype, shape in (
            ("hidden", dtype, [1, "sequence", 1]),
            ("floating_mask", dtype, [1, "sequence", 1]),
            ("masked", dtype, [1, "sequence", 1]),
            ("sum", dtype, [1, 1]),
            ("count", TensorProto.INT64, [1, 1]),
            ("floating_count", dtype, [1, 1]),
            ("mean", dtype, [1, 1]),
        )
    ]
    graph = helper.make_graph(
        nodes,
        "mean_classifier",
        inputs,
        [helper.make_tensor_value_info("logits", dtype, [1, 1])],
        initializer=initializers,
        value_info=infos,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def output_node(graph, output):
    return next(node for node in graph.node if output in node.output)


def evaluate(model, hidden, mask, outputs=None):
    with np.errstate(over="ignore", invalid="ignore"):
        return ReferenceEvaluator(model).run(
            outputs, {"encoder_input": hidden, "attention_mask": mask}
        )


class StableMeanNumerics(unittest.TestCase):
    def test_long_fp16_mean_is_finite_without_changing_encoder_or_head(self):
        original = mean_classifier()
        fixed = copy.deepcopy(original)
        self.assertEqual(stabilize_mean_pooling(fixed.graph), 1)
        onnx.checker.check_model(fixed, full_check=True)
        self.assertEqual(
            output_node(original.graph, "hidden"), output_node(fixed.graph, "hidden")
        )
        self.assertEqual(
            output_node(original.graph, "logits"), output_node(fixed.graph, "logits")
        )
        self.assertEqual(original.graph.initializer, fixed.graph.initializer)
        for length, finite_before in (
            (512, True),
            (8192, True),
            (16384, False),
            (32768, False),
        ):
            with self.subTest(length=length):
                hidden = np.full((1, length, 1), 5, np.float16)
                mask = np.ones((1, length), np.int64)
                (old,) = evaluate(original, hidden, mask)
                (new,) = evaluate(fixed, hidden, mask)
                self.assertEqual(new.dtype, np.dtype(np.float16))
                np.testing.assert_array_equal(new, [[5]])
                self.assertEqual(bool(np.isfinite(old).all()), finite_before)

    def test_padding_uses_integer_count_before_float32_cast(self):
        model = mean_classifier()
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        count_name = next(
            node.output[0]
            for node in model.graph.node
            if node.op_type == "Cast" and list(node.input) == ["count"]
        )
        for length, valid in ((16384, 1025), (32768, 2051), (32768, 16384)):
            for side in ("left", "right"):
                with self.subTest(length=length, valid=valid, side=side):
                    mask = np.zeros((1, length), np.int64)
                    mask[:, :valid] = 1
                    if side == "left":
                        mask = np.ascontiguousarray(mask[:, ::-1])
                    # Ignored padding differs sharply from the valid values.
                    hidden = np.where(mask[..., None], 1, 2000).astype(np.float16)
                    result, count = evaluate(
                        model, hidden, mask, ["logits", count_name]
                    )
                    np.testing.assert_array_equal(result, [[1]])
                    self.assertEqual(count.dtype, np.dtype(np.float32))
                    np.testing.assert_array_equal(count, [[valid]])
        # Casting the count via FP16 would round 2051 to 2052 and fail this test.
        self.assertNotEqual(int(np.float16(2051)), 2051)

    def test_nonconstant_masked_mean_matches_float64_reference(self):
        model = mean_classifier(swap_product=True)
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        rng = np.random.default_rng(42)
        hidden = rng.uniform(-12, 16, (1, 32768, 1)).astype(np.float16)
        padding_probability = 0.2
        mask = (rng.random((1, 32768)) > padding_probability).astype(np.int64)
        (result,) = evaluate(model, hidden, mask)
        expected = (
            (hidden.astype(np.float64) * mask[..., None]).sum(axis=1)
            / mask.sum(axis=1, keepdims=True)
        ).astype(np.float16)
        np.testing.assert_array_equal(result, expected)

    def test_all_masked_input_is_not_silently_made_valid(self):
        model = mean_classifier()
        stabilize_mean_pooling(model.graph)
        (result,) = evaluate(
            model, np.ones((1, 8, 1), np.float16), np.zeros((1, 8), np.int64)
        )
        self.assertTrue(np.isnan(result).all())


class StableMeanGraphContract(unittest.TestCase):
    def assert_unchanged(self, model):
        before = model.SerializeToString()
        self.assertEqual(stabilize_mean_pooling(model.graph), 0)
        self.assertEqual(model.SerializeToString(), before)

    def test_fp32_graph_is_unchanged(self):
        self.assert_unchanged(mean_classifier(TensorProto.FLOAT))

    def test_pass_is_idempotent(self):
        model = mean_classifier()
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        self.assert_unchanged(model)

    def test_preserves_shared_sum_and_count_cast(self):
        model = mean_classifier()
        for name in ("sum", "floating_count"):
            model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 1])
            )
        old_sum = copy.deepcopy(output_node(model.graph, "sum"))
        old_cast = copy.deepcopy(output_node(model.graph, "floating_count"))
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        self.assertEqual(output_node(model.graph, "sum"), old_sum)
        self.assertEqual(output_node(model.graph, "floating_count"), old_cast)
        onnx.checker.check_model(model, full_check=True)
        logits, total, count = evaluate(
            model, np.full((1, 32768, 1), 5, np.float16), np.ones((1, 32768), np.int64)
        )
        np.testing.assert_array_equal(logits, [[5]])
        self.assertTrue(np.isinf(total).all())
        np.testing.assert_array_equal(count, [[32768]])

    def test_other_reductions_are_unchanged(self):
        model = mean_classifier()
        unrelated = helper.make_node(
            "ReduceSum", ["encoder_input", "last_axis"], ["unrelated_sum"], keepdims=0
        )
        model.graph.node.insert(0, unrelated)
        model.graph.output.append(
            helper.make_tensor_value_info(
                "unrelated_sum", TensorProto.FLOAT16, [1, "sequence"]
            )
        )
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        self.assertEqual(output_node(model.graph, "unrelated_sum"), unrelated)
        onnx.checker.check_model(model, full_check=True)

    def test_preserves_intermediate_with_another_consumer(self):
        model = mean_classifier()
        alias = helper.make_node("Identity", ["sum"], ["debug_sum"])
        model.graph.node.append(alias)
        model.graph.output.append(
            helper.make_tensor_value_info("debug_sum", TensorProto.FLOAT16, [1, 1])
        )
        original_sum = copy.deepcopy(output_node(model.graph, "sum"))
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        self.assertEqual(output_node(model.graph, "sum"), original_sum)
        self.assertEqual(output_node(model.graph, "debug_sum"), alias)
        onnx.checker.check_model(model, full_check=True)

    def test_constant_axes_and_negative_unsqueeze_axis(self):
        model = mean_classifier()
        constants = []
        weights = []
        for tensor in model.graph.initializer:
            if tensor.name in {"sequence_axis", "last_axis"}:
                value = 1 if tensor.name == "sequence_axis" else -1
                constant = numpy_helper.from_array(np.array([value], np.int64))
                constants.append(
                    helper.make_node("Constant", [], [tensor.name], value=constant)
                )
            else:
                weights.append(tensor)
        del model.graph.initializer[:]
        model.graph.initializer.extend(weights)
        nodes = constants + list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        onnx.checker.check_model(model, full_check=True)
        (result,) = evaluate(
            model, np.full((1, 64, 1), 3, np.float16), np.ones((1, 64), np.int64)
        )
        np.testing.assert_array_equal(result, [[3]])

    def test_existing_tensor_names_do_not_collide(self):
        model = mean_classifier()
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array(1, np.int64), "mean__stable_pool_product_fp32"
            )
        )
        self.assertEqual(stabilize_mean_pooling(model.graph), 1)
        onnx.checker.check_model(model, full_check=True)

    def test_different_denominator_mask_is_not_a_match(self):
        model = mean_classifier()
        model.graph.input.append(
            helper.make_tensor_value_info(
                "other_mask", TensorProto.INT64, [1, "sequence"]
            )
        )
        output_node(model.graph, "count").input[0] = "other_mask"
        self.assert_unchanged(model)

    def test_mask_broadcast_over_sequence_is_not_a_mean(self):
        model = mean_classifier()
        mask = next(
            value for value in model.graph.input if value.name == "attention_mask"
        )
        mask.type.tensor_type.shape.dim[1].dim_value = 1
        self.assert_unchanged(model)

    def test_requires_known_intermediate_dtype(self):
        model = mean_classifier()
        kept = [info for info in model.graph.value_info if info.name != "masked"]
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)
        self.assert_unchanged(model)

    def test_dynamic_reduction_axes_are_not_a_match(self):
        model = mean_classifier()
        model.graph.input.append(
            helper.make_tensor_value_info("runtime_axes", TensorProto.INT64, [1])
        )
        output_node(model.graph, "sum").input[1] = "runtime_axes"
        self.assert_unchanged(model)

    def test_custom_reduction_is_not_a_match(self):
        model = mean_classifier()
        output_node(model.graph, "sum").domain = "vendor"
        self.assert_unchanged(model)

    def test_other_reduction_axis_is_not_a_match(self):
        model = mean_classifier()
        output_node(model.graph, "sum").input[1] = "last_axis"
        self.assert_unchanged(model)

    def test_requires_sequence_classifier_head(self):
        model = mean_classifier()
        output_node(model.graph, "logits").op_type = "MatMul"
        self.assert_unchanged(model)

    def test_cls_pooling_is_unchanged(self):
        model = mean_classifier()
        nodes = [node for node in model.graph.node if "mean" not in node.output]
        nodes.insert(
            -1,
            helper.make_node("Gather", ["hidden", "sequence_axis"], ["mean"], axis=1),
        )
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        self.assert_unchanged(model)


if __name__ == "__main__":
    unittest.main()
