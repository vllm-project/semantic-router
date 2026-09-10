"""Unit tests for rewrite_graph.py that need no model.

Run from this directory with the rewriter's own dependencies installed:

    python3 -m unittest test_rewrite_graph
"""

import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from rewrite_graph import output_precision_conflict, weight_precision


def graph_with_weights(*weights):
    """A minimal graph whose only content is the given `(name, array)` weights.

    No value_info is attached, the way an exporter that records no intermediate
    tensor types leaves a graph.
    """
    initializers = [numpy_helper.from_array(a, name) for name, a in weights]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    return helper.make_graph([], "weights", [x], [x], initializer=initializers)


class WeightPrecision(unittest.TestCase):
    def test_fp16_weights_read_as_fp16_without_value_info(self):
        graph = graph_with_weights(
            ("w1", np.zeros((2, 2), np.float16)),
            ("w2", np.zeros((2,), np.float16)),
            ("shape", np.array([1, 4], np.int64)),
        )
        self.assertEqual(weight_precision(graph.initializer), TensorProto.FLOAT16)
        self.assertIsNone(
            output_precision_conflict(
                "out/model_fa_fp16.onnx",
                weight_precision(graph.initializer) == TensorProto.FLOAT16,
            )
        )

    def test_fp32_weights_read_as_fp32(self):
        graph = graph_with_weights(
            ("w1", np.zeros((2, 2), np.float32)),
            ("shape", np.array([1, 4], np.int64)),
        )
        self.assertEqual(weight_precision(graph.initializer), TensorProto.FLOAT)
        self.assertIsNotNone(
            output_precision_conflict(
                "out/model_fa_fp16.onnx",
                weight_precision(graph.initializer) == TensorProto.FLOAT16,
            )
        )

    def test_mixed_weights_are_refused(self):
        graph = graph_with_weights(
            ("w1", np.zeros((2, 2), np.float32)),
            ("w2", np.zeros((2, 2), np.float32)),
            ("w3", np.zeros((2,), np.float16)),
        )
        with self.assertRaises(ValueError) as refused:
            weight_precision(graph.initializer)
        self.assertIn("2 FLOAT", str(refused.exception))
        self.assertIn("1 FLOAT16", str(refused.exception))

    def test_graph_without_float_weights_is_refused(self):
        graph = graph_with_weights(("shape", np.array([1, 4], np.int64)))
        with self.assertRaises(ValueError):
            weight_precision(graph.initializer)

    def test_bfloat16_weights_are_refused(self):
        tensor = helper.make_tensor("w", TensorProto.BFLOAT16, [1], [0])
        with self.assertRaises(ValueError) as refused:
            weight_precision([tensor])
        self.assertIn("BFLOAT16", str(refused.exception))


class OutputPrecisionConflict(unittest.TestCase):
    def test_fp32_graph_under_an_fp16_name_is_refused(self):
        message = output_precision_conflict(
            "out/model_fa_fp16.onnx", model_is_fp16=False
        )
        self.assertIsNotNone(message)
        self.assertIn("FP32 weights under an fp16 name", message)
        self.assertIn("model_fa.onnx", message)

    def test_fp16_graph_under_an_fp16_name_passes(self):
        self.assertIsNone(
            output_precision_conflict("out/model_fa_fp16.onnx", model_is_fp16=True)
        )

    def test_fp32_graph_under_model_fa_passes(self):
        self.assertIsNone(
            output_precision_conflict("out/model_fa.onnx", model_is_fp16=False)
        )

    def test_name_check_ignores_case_and_directories(self):
        self.assertIsNotNone(
            output_precision_conflict("MODEL_FA_FP16.ONNX", model_is_fp16=False)
        )
        self.assertIsNone(
            output_precision_conflict("fp16/model_fa.onnx", model_is_fp16=False)
        )


if __name__ == "__main__":
    unittest.main()
