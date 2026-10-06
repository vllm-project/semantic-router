"""Actual ORT storage parity and fail-closed external-data packing contracts."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pack_shared_weights import pack_models
except ImportError:
    onnx = None


def fixture(path, prefix="a", half=False, bias=0.25, external=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if half else np.float32
    onnx_type = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    weights = numpy_helper.from_array(
        np.array([[0.125, -0.5], [0.75, 0.375]], dtype), prefix + "_weight"
    )
    repeated = numpy_helper.from_array(
        np.array([[0.125, -0.5], [0.75, 0.375]], dtype), prefix + "_duplicate"
    )
    offset = numpy_helper.from_array(np.array([bias, -bias], dtype), "bias")
    nodes = [
        helper.make_node("MatMul", ["input", weights.name], ["a"]),
        helper.make_node("MatMul", ["input", repeated.name], ["b"]),
        helper.make_node("Add", ["a", "b"], ["c"]),
        helper.make_node("Add", ["c", "bias"], ["d"]),
        helper.make_node(
            "Constant",
            [],
            ["zero"],
            value=numpy_helper.from_array(np.array([0], dtype)),
        ),
        helper.make_node("Add", ["d", "zero"], ["output"]),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "storage",
            [helper.make_tensor_value_info("input", onnx_type, [None, 2])],
            [helper.make_tensor_value_info("output", onnx_type, [None, 2])],
            [weights, repeated, offset],
        ),
        opset_imports=[helper.make_opsetid("", 16)],
        ir_version=9,
    )
    if external:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".data",
            size_threshold=0,
        )
    else:
        onnx.save(model, path)


def run(path, values):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    ).run(None, {"input": values})[0]


@unittest.skipIf(onnx is None, "requires numpy, onnx and an actual ORT CPU runtime")
class SharedWeightsTests(unittest.TestCase):
    def test_actual_float_and_half_ort_outputs_and_every_blob_range(self):
        for half in [False, True]:
            with self.subTest(half=half), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources = {
                    "exit-1.onnx": root / "one/model.onnx",
                    "exit-2.onnx": root / "two/model.onnx",
                }
                fixture(sources["exit-1.onnx"], half=half)
                fixture(
                    sources["exit-2.onnx"],
                    prefix="different_name",
                    half=half,
                    bias=0.5,
                    external=True,
                )
                old = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
                output = root / "packed"
                receipt = pack_models(sources, output)
                self.assertLess(
                    receipt["unique_tensors"], receipt["tensor_occurrences"]
                )
                self.assertTrue(receipt["all_tensor_ranges_round_trip_verified"])
                blob = (output / "weights.data").read_bytes()
                self.assertEqual(
                    hashlib.sha256(blob).hexdigest(), receipt["shared_blob_sha256"]
                )
                offsets = []
                for graph in receipt["graphs"]:
                    for tensor in graph["tensors"]:
                        data = blob[
                            tensor["offset"] : tensor["offset"] + tensor["length"]
                        ]
                        self.assertEqual(len(data), tensor["length"])
                        self.assertEqual(
                            hashlib.sha256(data).hexdigest(), tensor["sha256"]
                        )
                        if tensor["shape"] == [2, 2]:
                            offsets.append(tensor["offset"])
                    for batch in [1, 3]:
                        values = (
                            np.arange(
                                batch * 2, dtype=np.float16 if half else np.float32
                            ).reshape(batch, 2)
                            / 3
                        )
                        np.testing.assert_array_equal(
                            run(output / graph["graph"], values),
                            run(sources[graph["graph"]], values),
                        )
                self.assertEqual(len(offsets), 4)
                self.assertEqual(
                    len(set(offsets)), 1
                )  # same bytes, different names and graphs
                for path, raw in old.items():
                    self.assertEqual(path.read_bytes(), raw)

    def test_equal_bytes_with_different_shape_or_dtype_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            fixture(source, external=False)
            model = onnx.load(source)
            raw = model.graph.initializer[0].raw_data
            for name, dtype, shape in [
                ("flat", TensorProto.FLOAT, [4]),
                ("integer", TensorProto.INT32, [2, 2]),
            ]:
                model.graph.initializer.append(
                    helper.make_tensor(name, dtype, shape, raw, raw=True)
                )
            source = root / "typed.onnx"
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="typed.data",
                size_threshold=0,
            )
            receipt = pack_models({"result.onnx": source}, root / "packed")
            selected = [
                row
                for row in receipt["graphs"][0]["tensors"]
                if row["name"] in {"a_weight", "flat", "integer"}
            ]
            self.assertEqual(len({row["offset"] for row in selected}), 3)

    def test_dangerous_locations_are_rejected_without_partial_output(self):
        for location in [
            "../outside.data",
            "/tmp/outside.data",
            "C:\\outside.data",
            "folder\\..\\outside.data",
            "",
            "escape.data",
        ]:
            with self.subTest(
                location=location
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source/model.onnx"
                fixture(source)
                outside = root / "outside.data"
                outside.write_bytes(b"\0" * 64)
                (source.parent / "escape.data").symlink_to(outside)
                model = onnx.load(source, load_external_data=False)
                tensor = model.graph.initializer[0]
                next(e for e in tensor.external_data if e.key == "location").value = (
                    location
                )
                source.write_bytes(model.SerializeToString())
                with self.assertRaises(ValueError):
                    pack_models({"out.onnx": source}, root / "packed")
                self.assertFalse((root / "packed").exists())
                self.assertFalse(list(root.glob(".packed-packing-*")))

    def test_ort_shape_inference_constants_stay_inline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/model.onnx"
            fixture(source)
            model = onnx.load(source, load_external_data=False)
            # Externalizing this Constant causes real ORT ShapeInferenceError:
            # Cannot parse data from external tensors (val_384).
            axis = numpy_helper.from_array(np.array([1], np.int64), "val_384")
            model.graph.node.append(
                helper.make_node("Constant", [], ["axis"], value=axis)
            )
            model.graph.node.append(
                helper.make_node("Unsqueeze", ["output", "axis"], ["expanded"])
            )
            shape = numpy_helper.from_array(np.array([0, 2], np.int64), "shape")
            model.graph.initializer.append(shape)
            model.graph.node.append(
                helper.make_node("Reshape", ["expanded", "shape"], ["restored"])
            )
            model.graph.output[0].name = "restored"
            source.write_bytes(model.SerializeToString())
            output = root / "packed"
            receipt = pack_models({"result.onnx": source}, output)
            self.assertGreater(receipt["preserved_inline_tensor_occurrences"], 0)
            loaded = onnx.load(output / "result.onnx", load_external_data=False)
            self.assertEqual(
                loaded.graph.initializer[-1].SerializeToString(),
                shape.SerializeToString(),
            )
            for batch in [1, 3]:
                values = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                np.testing.assert_array_equal(
                    run(source, values), run(output / "result.onnx", values)
                )

    def test_bad_range_fails_then_valid_retry_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/model.onnx"
            fixture(source)
            original = source.read_bytes()
            model = onnx.load(source, load_external_data=False)
            next(
                e for e in model.graph.initializer[0].external_data if e.key == "length"
            ).value = "999999"
            source.write_bytes(model.SerializeToString())
            with self.assertRaises(ValueError):
                pack_models({"out.onnx": source}, root / "packed")
            self.assertFalse((root / "packed").exists())
            source.write_bytes(original)
            pack_models({"out.onnx": source}, root / "packed")
            self.assertTrue((root / "packed/packing-manifest.json").is_file())

    def test_existing_output_flat_names_and_duplicate_metadata_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            fixture(source)
            existing = root / "existing"
            existing.mkdir()
            (existing / "keep").write_text("unchanged")
            with self.assertRaises(FileExistsError):
                pack_models({"out.onnx": source}, existing)
            self.assertEqual((existing / "keep").read_text(), "unchanged")
            for name in ["../out.onnx", "folder/out.onnx", "C:\\out.onnx", "/out.onnx"]:
                with self.assertRaises(ValueError):
                    pack_models({name: source}, root / "new")
            model = onnx.load(source, load_external_data=False)
            entry = model.graph.initializer[0].external_data.add()
            entry.key = "location"
            entry.value = "model.onnx.data"
            source.write_bytes(model.SerializeToString())
            with self.assertRaises(ValueError):
                pack_models({"out.onnx": source}, root / "new")

    def test_non_storage_fields_and_inline_empty_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.onnx"
            fixture(source, external=False)
            model = onnx.load(source)
            model.doc_string = "public description"
            model.graph.node[0].doc_string = "operation metadata"
            model.graph.initializer.append(
                numpy_helper.from_array(np.empty((0,), np.float32), "empty")
            )
            onnx.save(model, source)
            pack_models({"out.onnx": source}, root / "packed")
            packed = onnx.load(root / "packed/out.onnx", load_external_data=False)
            self.assertEqual(packed.doc_string, model.doc_string)
            self.assertEqual(
                packed.graph.node[0].SerializeToString(),
                model.graph.node[0].SerializeToString(),
            )
            self.assertEqual(
                packed.graph.initializer[-1].SerializeToString(),
                model.graph.initializer[-1].SerializeToString(),
            )


if __name__ == "__main__":
    unittest.main()
