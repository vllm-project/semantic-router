"""Unit tests for rewrite_graph.py that need no model.

Run from this directory with the rewriter's own dependencies installed:

    python3 -m unittest test_rewrite_graph
"""

import unittest

from rewrite_graph import output_precision_conflict


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
