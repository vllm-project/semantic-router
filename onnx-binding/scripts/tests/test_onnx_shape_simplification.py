"""Structural proof boundaries and actual dynamic ONNX CPU equivalence."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from modernbert_inputs import ort_inputs
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
    from onnx_shape_simplification import simplify_batch_reshapes
except ImportError:
    ort = None


@unittest.skipIf(ort is None, "requires ONNX and actual ORT CPU")
class ShapeSimplificationTest(unittest.TestCase):
    def graph(self, unpack=True):
        tail = [-1, 3, 2, 4] if unpack else [-1, 8]
        shape = ["B", "S", 24] if unpack else ["B", "S", 2, 4]
        nodes = [
            h.make_node("Shape", ["mask"], ["batch"], start=0, end=1),
            h.make_node("Concat", ["batch", "tail"], ["target"], axis=0),
            h.make_node("Reshape", ["data", "target"], ["out"], allowzero=1),
        ]
        model = h.make_model(
            h.make_graph(
                nodes,
                "arbitrary-names",
                [
                    h.make_tensor_value_info("mask", T.INT64, ["B", "S"]),
                    h.make_tensor_value_info("data", T.FLOAT, shape),
                ],
                [h.make_tensor_value_info("out", T.FLOAT, ["B", "S", *tail[1:]])],
                [nh.from_array(np.array(tail, np.int64), "tail")],
            ),
            opset_imports=[h.make_opsetid("", 18)],
            ir_version=10,
        )
        return model

    def session(self, graph):
        onnx.checker.check_model(graph)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        return ort.InferenceSession(
            graph.SerializeToString(),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def test_actual_dynamic_copy_shape_and_initializer_preservation(self):
        for unpack in [False, True]:
            original = self.graph(unpack)
            graph = copy.deepcopy(original)
            initializer = graph.graph.initializer[0].SerializeToString()
            result = simplify_batch_reshapes(graph)
            self.assertEqual(result["rewritten_reshapes"], 1)
            self.assertEqual(
                graph.graph.initializer[0].SerializeToString(), initializer
            )
            self.assertEqual([x.op_type for x in graph.graph.node], ["Reshape"])
            before = graph.SerializeToString()
            self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)
            self.assertEqual(before, graph.SerializeToString())
            old, new = self.session(original), self.session(graph)
            for batch in [1, 2]:
                for length in [1, 17, 129, 32768]:
                    shape = (batch, length, 24) if unpack else (batch, length, 2, 4)
                    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                    feeds = {"data": data, "mask": np.ones((batch, length), np.int64)}
                    np.testing.assert_array_equal(
                        old.run(None, feeds)[0], new.run(None, feeds)[0]
                    )

    def test_missing_or_contradictory_dimension_identity_is_not_rewritten(self):
        for dim in [None, "OTHER", 3]:
            graph = self.graph()
            dimension = graph.graph.input[1].type.tensor_type.shape.dim[0]
            dimension.Clear()
            if isinstance(dim, str):
                dimension.dim_param = dim
            elif dim:
                dimension.dim_value = dim
            before = graph.SerializeToString()
            self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)
            self.assertEqual(before, graph.SerializeToString())

    def test_zero_multiple_inferred_nonconstant_and_custom_ops_are_not_rewritten(self):
        for tail in [[-1, 0, 2, 4], [-1, -1, 2, 4], [-2, 3, 2, 4]]:
            graph = self.graph()
            graph.graph.initializer[0].CopyFrom(
                nh.from_array(np.array(tail, np.int64), "tail")
            )
            self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)
        for index in [0, 1, 2]:
            graph = self.graph()
            graph.graph.node[index].domain = "custom"
            self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)
        graph = self.graph()
        graph.graph.initializer[0].data_type = T.INT32
        self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)
        graph = self.graph()
        graph.graph.ClearField("initializer")
        self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 0)

    def test_shared_shape_consumers_remain_intact(self):
        graph = self.graph()
        graph.graph.output.append(h.make_tensor_value_info("target", T.INT64, [5]))
        self.assertEqual(simplify_batch_reshapes(graph)["rewritten_reshapes"], 1)
        self.assertEqual(
            [x.op_type for x in graph.graph.node], ["Shape", "Concat", "Reshape"]
        )
        self.session(graph)

    def test_runtime_inputs_preserve_legacy_and_reject_unsupported_contracts(self):
        def session(positions=None):
            inputs = [
                SimpleNamespace(name=n, type="tensor(int64)", shape=["B", "S"])
                for n in ["input_ids", "attention_mask"]
            ]
            if positions:
                inputs.append(positions)
            return SimpleNamespace(get_inputs=lambda: inputs)

        ids = np.ones((2, 5), np.int64)
        mask = ids.copy()
        mask[1, :3] = 0
        self.assertEqual(
            set(ort_inputs(session(), ids, mask)), {"input_ids", "attention_mask"}
        )
        valid = SimpleNamespace(
            name="position_ids", type="tensor(int64)", shape=[1, "S"]
        )
        np.testing.assert_array_equal(
            ort_inputs(session(valid), ids, mask)["position_ids"], [[0, 1, 2, 3, 4]]
        )
        for field, value in [
            ("name", "unexpected"),
            ("type", "tensor(float)"),
            ("shape", []),
            ("shape", [2, "S"]),
            ("shape", [1, 8]),
        ]:
            bad = copy.copy(valid)
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                ort_inputs(session(bad), ids, mask)
        with self.assertRaises(ValueError):
            ort_inputs(session(valid), ids.astype(np.int32), mask)
        with self.assertRaises(ValueError):
            ort_inputs(session(valid), ids, mask[:1])


if __name__ == "__main__":
    unittest.main()
