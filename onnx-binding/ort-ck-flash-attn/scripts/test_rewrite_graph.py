"""Unit tests for rewrite_graph.py that need no model.

Run from this directory with the rewriter's own dependencies installed:

    python3 -m unittest test_rewrite_graph
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from rewrite_graph import (
    attention_window,
    build_maps,
    find_attention_blocks,
    output_precision_conflict,
    rewrite,
    rotary_fp32_initializers,
    weight_precision,
)


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


def graph_with_fp32_rotary_constant():
    graph = graph_with_weights(
        ("encoder.weight", np.ones((2, 2), np.float16)),
        ("frequencies", np.ones((1, 32, 1), np.float32)),
        ("start", np.array(0, np.int64)),
        ("length", np.array(64, np.int64)),
        ("step", np.array(1, np.int64)),
        ("axes", np.array([0, 1], np.int64)),
    )
    for op, inputs, output, attrs in (
        ("Range", ["start", "length", "step"], "positions", {}),
        ("Unsqueeze", ["positions", "axes"], "position_3d", {}),
        ("Cast", ["position_3d"], "position_float", {"to": TensorProto.FLOAT}),
        ("MatMul", ["frequencies", "position_float"], "angles", {}),
        ("Transpose", ["angles"], "angles_t", {"perm": [0, 2, 1]}),
        ("Concat", ["angles_t", "angles_t"], "angles_cat", {"axis": -1}),
        ("Cos", ["angles_cat"], "cos", {}),
        ("Sin", ["angles_cat"], "sin", {}),
        ("Cast", ["cos"], "cos_half", {"to": TensorProto.FLOAT16}),
        ("Cast", ["sin"], "sin_half", {"to": TensorProto.FLOAT16}),
    ):
        graph.node.append(helper.make_node(op, inputs, [output], name=output, **attrs))
    return graph


class RotaryPrecision(unittest.TestCase):
    def test_fp32_rotary_island_is_preserved_but_not_counted_as_weight(self):
        graph = graph_with_fp32_rotary_constant()
        before = graph.SerializeToString()
        exempt = rotary_fp32_initializers(graph, build_maps(graph)[0])
        self.assertEqual(exempt, {"frequencies"})
        self.assertEqual(
            weight_precision(graph.initializer, position_constants=exempt),
            TensorProto.FLOAT16,
        )
        self.assertEqual(before, graph.SerializeToString())

    def test_fp32_weight_with_extra_consumer_is_still_refused(self):
        graph = graph_with_fp32_rotary_constant()
        graph.node.append(
            helper.make_node("MatMul", ["frequencies", "x"], ["other"], name="other")
        )
        exempt = rotary_fp32_initializers(graph, build_maps(graph)[0])
        self.assertEqual(exempt, set())
        with self.assertRaisesRegex(ValueError, "mixes weight precisions"):
            weight_precision(graph.initializer, position_constants=exempt)

    def test_non_position_or_non_fp16_output_is_not_exempt(self):
        for variant in ("activation", "fp32_output"):
            graph = graph_with_fp32_rotary_constant()
            if variant == "activation":
                next(n for n in graph.node if n.name == "position_float").input[0] = "x"
            else:
                cast = next(n for n in graph.node if n.name == "sin_half")
                cast.attribute[0].i = TensorProto.FLOAT
            self.assertEqual(
                rotary_fp32_initializers(graph, build_maps(graph)[0]), set()
            )


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


def exported_attention_graph(layers=22, guard=True, inverted_fill=False):
    """Small, weight-free reproduction of the published intent FP16 export.

    Preserve its K reshape/transpose chain, scalar Q/K pre-scaling,
    Softmax/IsNaN/Where guard and inclusive positional mask. Names intentionally
    resemble neither exporter: classification must follow mask computation.
    """
    initializers = [
        numpy_helper.from_array(np.array(value, dtype=dtype), name)
        for name, value, dtype in (
            ("zero", 0, np.float16),
            ("negative", -65504, np.float16),
            ("qk_scale", np.sqrt(1 / 8), np.float16),
            ("start", 0, np.int64),
            ("length", 130, np.int64),
            ("step", 1, np.int64),
            ("radius", 64, np.int64),
            ("row_axis", [0], np.int64),
            ("column_axis", [1], np.int64),
            ("pad_axes", [1, 2], np.int64),
            ("dense_shape", [1, 1, 130, 130], np.int64),
            ("k_flat_shape", [1, 130, 64], np.int64),
            ("k_shape", [1, 1, 64, 130], np.int64),
        )
    ]
    nodes = []

    def add(op, inputs, output, **attrs):
        nodes.append(
            helper.make_node(op, inputs, [output], name="node_" + output, **attrs)
        )

    add("Cast", ["attention_mask"], "padding", to=TensorProto.BOOL)
    add("Unsqueeze", ["padding", "pad_axes"], "padding_4d")
    add("Expand", ["padding_4d", "dense_shape"], "global_keep")
    add("Range", ["start", "length", "step"], "positions")
    add("Unsqueeze", ["positions", "row_axis"], "keys")
    add("Unsqueeze", ["positions", "column_axis"], "queries")
    add("Sub", ["queries", "keys"], "difference")
    add("Abs", ["difference"], "distance")
    add("LessOrEqual", ["distance", "radius"], "within_window")
    add("And", ["global_keep", "within_window"], "local_keep")
    add("Where", ["global_keep", "zero", "negative"], "arbitrary_global")
    add("Where", ["local_keep", "zero", "negative"], "arbitrary_local")
    if inverted_fill:
        # Transformers 4.57.6 / Torch 2.10 TokenClassification export uses
        # Where(bool(1-padding), negative, 1-padding), then a separate window
        # masked_fill. Preserve key-only padding axes [1,2] and predicate polarity.
        initializers.append(
            numpy_helper.from_array(np.array(1, dtype=np.float16), "one")
        )
        nodes = [
            n
            for n in nodes
            if n.output[0]
            not in {"padding", "local_keep", "arbitrary_global", "arbitrary_local"}
        ]
        next(n for n in nodes if n.output[0] == "padding_4d").input[
            0
        ] = "attention_mask"
        queries = next(n for n in nodes if n.output[0] == "queries")
        queries.CopyFrom(
            helper.make_node(
                "Transpose", ["keys"], ["queries"], name="node_queries", perm=[1, 0]
            )
        )
        add("Cast", ["global_keep"], "float_padding", to=TensorProto.FLOAT16)
        add("Sub", ["one", "float_padding"], "inverted_padding")
        add("Cast", ["inverted_padding"], "is_padding", to=TensorProto.BOOL)
        add(
            "Where",
            ["is_padding", "negative", "inverted_padding"],
            "arbitrary_global",
        )
        add("Not", ["within_window"], "outside_window")
        add(
            "Where",
            ["outside_window", "negative", "arbitrary_global"],
            "arbitrary_local",
        )
    inputs = [
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 130])
    ]
    outputs = []
    for layer in range(layers):
        p = f"layer_{layer}_"
        for name in ("q", "k", "v"):
            inputs.append(
                helper.make_tensor_value_info(
                    p + name, TensorProto.FLOAT16, [1, 1, 130, 64]
                )
            )
        add("Reshape", [p + "k", "k_flat_shape"], p + "k_flat")
        add("Transpose", [p + "k_flat"], p + "k_transposed", perm=[0, 2, 1])
        add("Reshape", [p + "k_transposed", "k_shape"], p + "kt")
        add("Mul", [p + "q", "qk_scale"], p + "qs")
        add("Mul", [p + "kt", "qk_scale"], p + "ks")
        add("MatMul", [p + "qs", p + "ks"], p + "qk")
        mask = "arbitrary_global" if layer % 3 == 0 else "arbitrary_local"
        add("Add", [p + "qk", mask], p + "masked")
        add("Softmax", [p + "masked"], p + "probability", axis=-1)
        probability = p + "probability"
        if guard:
            add("IsNaN", [probability], p + "isnan")
            add("Where", [p + "isnan", "zero", probability], p + "guarded")
            probability = p + "guarded"
        add("MatMul", [probability, p + "v"], p + "out")
        outputs.append(
            helper.make_tensor_value_info(
                p + "out", TensorProto.FLOAT16, [1, 1, 130, 64]
            )
        )
    return helper.make_model(
        helper.make_graph(nodes, "published_sdpa_shape", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )


class PublishedExportShape(unittest.TestCase):
    def test_nan_guard_and_reshape_k_match_all_22_layers(self):
        model = exported_attention_graph()
        blocks = find_attention_blocks(model.graph, build_maps(model.graph)[0])
        self.assertEqual(len(blocks), 22)
        self.assertEqual(blocks[0]["k_tensor"], "layer_0_k")
        self.assertEqual(len(blocks[0]["probability_nodes"]), 2)

    def test_direct_softmax_remains_supported(self):
        model = exported_attention_graph(guard=False)
        self.assertEqual(
            len(find_attention_blocks(model.graph, build_maps(model.graph)[0])), 22
        )

    def test_semantic_window_does_not_reverse_global_and_local(self):
        graph = exported_attention_graph().graph
        out2node, _ = build_maps(graph)
        self.assertEqual(
            attention_window(graph, out2node, "arbitrary_global"), (-1, -1)
        )
        self.assertEqual(attention_window(graph, out2node, "arbitrary_local"), (64, 64))

    def test_unsupported_mask_is_not_silently_global(self):
        graph = exported_attention_graph().graph
        next(n for n in graph.node if n.op_type == "LessOrEqual").op_type = "Greater"
        with self.assertRaisesRegex(
            ValueError, "unsupported attention-mask comparison"
        ):
            attention_window(graph, build_maps(graph)[0], "arbitrary_local")

    def test_inverted_padding_mask_is_refused(self):
        graph = exported_attention_graph().graph
        mask = next(n for n in graph.node if n.output[0] == "arbitrary_global")
        mask.input[1], mask.input[2] = mask.input[2], mask.input[1]
        with self.assertRaisesRegex(ValueError, "allowed positions to zero"):
            attention_window(graph, build_maps(graph)[0], "arbitrary_global")

    def test_token_reexport_masked_fill_matches_global_and_local(self):
        graph = exported_attention_graph(inverted_fill=True).graph
        out2node, _ = build_maps(graph)
        self.assertEqual(
            attention_window(graph, out2node, "arbitrary_global"), (-1, -1)
        )
        self.assertEqual(attention_window(graph, out2node, "arbitrary_local"), (64, 64))

    def test_token_reexport_rejects_wrong_inversion_and_query_padding(self):
        for variant in ("zero_minus_padding", "query_axes", "not_padding"):
            with self.subTest(variant=variant):
                graph = exported_attention_graph(inverted_fill=True).graph
                if variant == "zero_minus_padding":
                    next(
                        n for n in graph.node if n.output[0] == "inverted_padding"
                    ).input[0] = "zero"
                elif variant == "query_axes":
                    axes = next(t for t in graph.initializer if t.name == "pad_axes")
                    axes.CopyFrom(numpy_helper.from_array(np.array([1, 3]), "pad_axes"))
                else:
                    next(n for n in graph.node if n.output[0] == "is_padding").input[
                        0
                    ] = "float_padding"
                with self.assertRaisesRegex(ValueError, "allowed positions to zero"):
                    attention_window(graph, build_maps(graph)[0], "arbitrary_global")

    def test_token_reexport_all_22_dense_masks_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_sdpa_fp16.onnx"
            target = Path(directory) / "model_fa_fp16.onnx"
            onnx.save(exported_attention_graph(guard=False, inverted_fill=True), source)
            rewrite(source, target)
            model = onnx.load(target)
        onnx.checker.check_model(model)
        self.assertEqual(
            sum(n.op_type == "CKFlashAttention" for n in model.graph.node), 22
        )
        self.assertFalse(
            {"Softmax", "Where", "Expand", "Not", "Abs", "LessOrEqual"}
            & {n.op_type for n in model.graph.node}
        )
        self.assertFalse(
            {"inverted_padding", "difference"}
            & {o for n in model.graph.node for o in n.output}
        )

    def test_rewrite_removes_dense_masks_and_guards_in_every_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_sdpa_fp16.onnx"
            target = Path(directory) / "model_fa_fp16.onnx"
            onnx.save(exported_attention_graph(), source)
            rewrite(source, target)
            model = onnx.load(target)
        onnx.checker.check_model(model)
        flash = [n for n in model.graph.node if n.op_type == "CKFlashAttention"]
        self.assertEqual(len(flash), 22)
        windows = [
            tuple(
                helper.get_attribute_value(a)
                for a in n.attribute
                if a.name.startswith("window_size")
            )
            for n in flash
        ]
        self.assertEqual(windows.count((-1, -1)), 8)
        self.assertEqual(windows.count((64, 64)), 14)
        self.assertTrue(
            all(n.input[3] == "/model/_fa_pad_bias/Unsqueeze_output_0" for n in flash)
        )
        self.assertFalse(
            {"Softmax", "IsNaN", "Where", "Expand", "Abs", "LessOrEqual"}
            & {n.op_type for n in model.graph.node}
        )

    def test_partial_match_and_nonzero_nan_replacement_are_refused(self):
        model = exported_attention_graph()
        guard = next(n for n in model.graph.node if n.output[0] == "layer_0_guarded")
        guard.input[1] = "negative"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_sdpa_fp16.onnx"
            onnx.save(model, source)
            with self.assertRaisesRegex(ValueError, "matched 21 of 22"):
                rewrite(source, Path(directory) / "model_fa_fp16.onnx")

    def test_requested_window_must_match_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_sdpa_fp16.onnx"
            onnx.save(exported_attention_graph(), source)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                rewrite(
                    source, Path(directory) / "model_fa_fp16.onnx", local_attention=256
                )

    def test_exported_dense_mask_output_prevents_unsafe_memory_claim(self):
        model = exported_attention_graph()
        model.graph.output.append(
            helper.make_tensor_value_info(
                "arbitrary_global", TensorProto.FLOAT16, [1, 1, 130, 130]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_sdpa_fp16.onnx"
            onnx.save(model, source)
            with self.assertRaisesRegex(
                ValueError, "dense attention mask is still live"
            ):
                rewrite(source, Path(directory) / "model_fa_fp16.onnx")


if __name__ == "__main__":
    unittest.main()
