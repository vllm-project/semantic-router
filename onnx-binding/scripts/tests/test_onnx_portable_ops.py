"""Actual IEEE edge-case parity, nested scope safety, and storage preservation."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
    from onnx_portable_ops import lower_nan_predicates
except ImportError:
    ort = None


@unittest.skipIf(ort is None, "requires ONNX and actual ORT CPU")
class PortableNanTest(unittest.TestCase):
    def model(self, dtype=None, opset=18):
        dtype = T.FLOAT if dtype is None else dtype
        return h.make_model(
            h.make_graph(
                [h.make_node("IsNaN", ["x"], ["out"], name="predicate")],
                "guard",
                [h.make_tensor_value_info("x", dtype, ["B", "S"])],
                [h.make_tensor_value_info("out", T.BOOL, ["B", "S"])],
            ),
            opset_imports=[h.make_opsetid("", opset)],
            ir_version=10,
        )

    def run_model(self, model, feeds):
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return session.run(None, feeds)

    def test_ieee_edges_dynamic_batches_and_idempotence(self):
        for dtype, numpy_type in [
            (T.FLOAT16, np.float16),
            (T.FLOAT, np.float32),
            (T.DOUBLE, np.float64),
        ]:
            original = self.model(dtype)
            candidate = copy.deepcopy(original)
            result = lower_nan_predicates(candidate)
            self.assertEqual(result["rewritten_nan_predicates"], 1)
            self.assertEqual(result["unsupported_nan_predicates"], 0)
            saved = candidate.SerializeToString()
            self.assertEqual(
                lower_nan_predicates(candidate)["rewritten_nan_predicates"], 0
            )
            self.assertEqual(saved, candidate.SerializeToString())
            for batch in [1, 2]:
                values = np.tile(
                    np.array(
                        [0, -0.0, 1, -1, np.inf, -np.inf, np.nan, -np.nan], numpy_type
                    ),
                    (batch, 1),
                )
                expected = self.run_model(original, {"x": values})[0]
                actual = self.run_model(candidate, {"x": values})[0]
                np.testing.assert_array_equal(expected, np.isnan(values))
                np.testing.assert_array_equal(actual, expected)

    def test_signaling_nan_payloads_and_subnormals(self):
        payloads = np.array(
            [0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000, 1, 0x80000001],
            np.uint32,
        ).view(np.float32)[None, :]
        model = self.model()
        lower_nan_predicates(model)
        actual = self.run_model(model, {"x": payloads})[0]
        np.testing.assert_array_equal(actual, [[True, True, True, True, False, False]])

    def test_masked_softmax_keeps_nan_to_zero_guard(self):
        model = self.model()
        model.graph.node.insert(0, h.make_node("Softmax", ["x"], ["p"], axis=-1))
        model.graph.node[1].input[0] = "p"
        model.graph.node.append(h.make_node("Where", ["out", "zero", "p"], ["safe"]))
        model.graph.initializer.append(nh.from_array(np.array(0, np.float32), "zero"))
        model.graph.output.append(h.make_tensor_value_info("safe", T.FLOAT, ["B", "S"]))
        feeds = {"x": np.array([[-np.inf] * 4, [0, -np.inf, 1, -np.inf]], np.float32)}
        expected = self.run_model(model, feeds)
        lower_nan_predicates(model)
        actual = self.run_model(model, feeds)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(np.isfinite(actual[1]).all())
        np.testing.assert_array_equal(actual[1][0], np.zeros(4, np.float32))

    def test_subgraph_captures_and_global_name_collisions(self):
        model = self.model()
        branch = copy.deepcopy(model.graph)
        branch.ClearField("input")
        model.graph.ClearField("node")
        model.graph.node.append(h.make_node("Identity", ["x"], ["out_self_equal"]))
        model.graph.node.append(
            h.make_node(
                "If", ["condition"], ["out"], then_branch=branch, else_branch=branch
            )
        )
        model.graph.input.append(h.make_tensor_value_info("condition", T.BOOL, []))
        original = copy.deepcopy(model)
        self.assertEqual(lower_nan_predicates(model)["rewritten_nan_predicates"], 2)
        names = []
        for attribute in model.graph.node[1].attribute:
            self.assertEqual([n.op_type for n in attribute.g.node], ["Equal", "Not"])
            names.append(attribute.g.node[0].output[0])
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("out_self_equal", names)
        for condition in [True, False]:
            feeds = {
                "x": np.array([[np.nan, -np.inf]], np.float32),
                "condition": np.array(condition),
            }
            np.testing.assert_array_equal(
                self.run_model(model, feeds)[0], self.run_model(original, feeds)[0]
            )

    def test_local_function_uses_its_own_opset(self):
        model = self.model()
        model.graph.node[0].op_type = "Predicate"
        model.graph.node[0].domain = "local"
        model.opset_import.append(h.make_opsetid("local", 1))
        function = h.make_function(
            "local",
            "Predicate",
            ["x"],
            ["out"],
            [h.make_node("IsNaN", ["x"], ["out"])],
            opset_imports=[h.make_opsetid("", 18)],
        )
        model.functions.append(function)
        original = copy.deepcopy(model)
        self.assertEqual(lower_nan_predicates(model)["rewritten_nan_predicates"], 1)
        feeds = {"x": np.array([[np.nan, np.inf, -0.0]], np.float32)}
        np.testing.assert_array_equal(
            self.run_model(model, feeds)[0], self.run_model(original, feeds)[0]
        )
        model = original
        model.functions[0].opset_import[0].version = 9
        saved = model.SerializeToString()
        self.assertEqual(lower_nan_predicates(model)["unsupported_nan_predicates"], 1)
        self.assertEqual(saved, model.SerializeToString())

    def test_unsupported_type_contracts_and_custom_domains_are_preserved(self):
        for version in [9, 10, 20]:
            model = self.model(opset=version)
            saved = model.SerializeToString()
            result = lower_nan_predicates(model)
            self.assertEqual(result["rewritten_nan_predicates"], 0)
            self.assertEqual(result["unsupported_nan_predicates"], 1)
            self.assertEqual(saved, model.SerializeToString())
        model = self.model()
        model.graph.node[0].domain = "custom"
        saved = model.SerializeToString()
        self.assertEqual(lower_nan_predicates(model)["rewritten_nan_predicates"], 0)
        self.assertEqual(saved, model.SerializeToString())

    def test_inline_and_external_storage_are_byte_preserved(self):
        model = self.model()
        inline = nh.from_array(np.array([np.nan, -0.0], np.float32), "inline")
        external = h.make_tensor("weights", T.FLOAT, [1], [0])
        external.ClearField("float_data")
        external.data_location = T.EXTERNAL
        for key, value in [
            ("location", "weights.data"),
            ("offset", "4096"),
            ("length", "4"),
        ]:
            external.external_data.add(key=key, value=value)
        model.graph.initializer.extend([inline, external])
        before = [x.SerializeToString() for x in model.graph.initializer]
        lower_nan_predicates(model)
        self.assertEqual(
            before, [x.SerializeToString() for x in model.graph.initializer]
        )


if __name__ == "__main__":
    unittest.main()
