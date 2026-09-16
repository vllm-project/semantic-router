"""Execute broadcast-mask transformations and reject unproved consumers/predicates."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from canonicalize_attention_masks import PREFIX, rewrite_model
from onnx import TensorProto, helper
from test_rewrite_blocked_attention import attention_model, feeds, ort, session


@unittest.skipIf(ort is None, "onnxruntime is required for numerical execution")
class BroadcastMaskExecutionTests(unittest.TestCase):
    def test_dynamic_padding_windows_and_ieee_values(self):
        for radius in [None, 0, 64]:
            for gathered in [False, True]:
                model = attention_model(
                    radius=radius, gathered=gathered, negative=np.finfo(np.float32).min
                )
                original = model.SerializeToString()
                rewritten, receipt = rewrite_model(model)
                self.assertEqual(original, model.SerializeToString())
                self.assertTrue(receipt["nan_guards_unchanged"])
                runners = [session(m) for m in [model, rewritten]]
                for batch, length in [(1, 1), (2, 3), (2, 65), (1, 129)]:
                    for variant in ["padding", "all_masked", "nan", "infinity"]:
                        with self.subTest(
                            radius=radius,
                            gathered=gathered,
                            batch=batch,
                            length=length,
                            variant=variant,
                        ):
                            values = feeds(
                                batch, length, all_masked=variant == "all_masked"
                            )
                            # Boolean conversion, not equality with one, defines padding.
                            values["attention_mask"][values["attention_mask"] == 1] = -3
                            if variant == "nan":
                                values["q"][0, 0, 0, 0] = np.nan
                                values["v"][0, 1, -1, 0] = np.nan
                            if variant == "infinity":
                                values["q"][0, 0, 0, 0] = np.inf
                                values["k"][0, 0, -1, 0] = -np.inf
                            before, after = [r.run(None, values) for r in runners]
                            for left, right in zip(before, after, strict=True):
                                np.testing.assert_array_equal(left, right)

    def test_global_mask_broadcast_shape_and_nan_guard_bytes(self):
        model = attention_model(gathered=True, negative=np.finfo(np.float32).min)
        rewritten, _ = rewrite_model(model)

        def guards(model):
            return [
                n.SerializeToString()
                for n in model.graph.node
                if n.name in {"isnan", "guarded"}
            ]

        self.assertEqual(guards(model), guards(rewritten))
        self.assertFalse(any(n.op_type == "GatherND" for n in rewritten.graph.node))
        # Expose only in the test *after* transformation to verify its actual shape.
        rewritten.graph.output.append(
            helper.make_tensor_value_info(
                "global_mask", TensorProto.FLOAT, ["batch", 1, 1, "sequence"]
            )
        )
        mask = session(rewritten).run(None, feeds(2, 129))[-1]
        self.assertEqual(mask.shape, (2, 1, 1, 129))

    def test_cli_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = [
                Path(directory) / name for name in ["source.onnx", "output.onnx"]
            ]
            onnx.save(
                attention_model(gathered=True, negative=np.finfo(np.float32).min),
                source,
            )
            original = source.read_bytes()
            command = [
                sys.executable,
                str(Path(__file__).with_name("canonicalize_attention_masks.py")),
                str(source),
                str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout)["attention_blocks"], 1)
            self.assertEqual(source.read_bytes(), original)
            actual = session(onnx.load(output)).run(None, feeds(1, 3))
            expected = session(onnx.load(source)).run(None, feeds(1, 3))
            np.testing.assert_array_equal(actual[0], expected[0])
            frozen = output.read_bytes()
            again = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("Refusing to overwrite", again.stderr)
            self.assertEqual(output.read_bytes(), frozen)


class BroadcastMaskRefusalTests(unittest.TestCase):
    def test_unknown_predicates_shapes_axes_and_fills_are_rejected(self):
        for case in [
            "shape_consumer",
            "output",
            "unknown",
            "negative_fill",
            "infinite_fill",
            "gather_axis",
            "key_axis",
            "old_opset",
            "reserved_initializer",
        ]:
            with self.subTest(case=case):
                model = attention_model(
                    radius=64, gathered=True, negative=np.finfo(np.float32).min
                )
                nodes = {n.name: n for n in model.graph.node}
                if case == "shape_consumer":
                    model.graph.node.append(
                        helper.make_node("Shape", ["local_mask"], ["observed_shape"])
                    )
                elif case == "output":
                    model.graph.output.append(
                        helper.make_tensor_value_info(
                            "local_mask", TensorProto.FLOAT, ["b", 1, "s", "s"]
                        )
                    )
                elif case == "unknown":
                    nodes["local_mask"].input[0] = "unknown_predicate"
                elif case == "negative_fill":
                    nodes["local_mask"].input[1] = "negative"
                elif case == "infinite_fill":
                    model = attention_model(radius=64, gathered=True)
                elif case == "gather_axis":
                    nodes["global_keep"].attribute[0].i = 1
                elif case == "key_axis":
                    axes = next(x for x in model.graph.initializer if x.name == "axes2")
                    axes.CopyFrom(
                        onnx.numpy_helper.from_array(np.array([1], np.int64), "axes2")
                    )
                elif case == "old_opset":
                    model.opset_import[0].version = 13
                else:
                    item = copy.deepcopy(model.graph.initializer[0])
                    item.name = PREFIX + "reserved"
                    model.graph.initializer.append(item)
                original = model.SerializeToString()
                with self.assertRaises(ValueError):
                    rewrite_model(model)
                self.assertEqual(model.SerializeToString(), original)


if __name__ == "__main__":
    unittest.main()
