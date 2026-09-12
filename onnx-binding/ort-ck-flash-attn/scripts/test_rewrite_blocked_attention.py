"""Execute tiny FP32 ONNX attention graphs with dynamic batch/sequence shapes."""

import copy
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from rewrite_blocked_attention import rewrite_model, specialize_batch

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def attention_model(radius=None, guard=True, inverted=False, negative=-np.inf):
    nodes, initializers = [], []

    def const(name, value, dtype=np.int64):
        initializers.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        )
        return name

    def op(kind, inputs, name, **attrs):
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    zero = const("zero", 0, np.float32)
    neg = const("negative", negative, np.float32)
    const("q_scale", 0.37, np.float32)
    const("k_scale", 0.29, np.float32)
    const("pad_axes", [1, 2])
    const("zero_i", 0)
    const("one_i", 1)
    const("two_i", 2)
    const("axes0", [0])
    const("axes1", [1])
    shape = op("Shape", ["q"], "q_shape")
    length = op("Gather", [shape, "two_i"], "length", axis=0)
    batch = op("Gather", [shape, "zero_i"], "batch", axis=0)
    b = op("Unsqueeze", [batch, "axes0"], "batch_vector")
    n = op("Unsqueeze", [length, "axes0"], "length_vector")
    dense = op("Concat", [b, const("one_vector", [1]), n, n], "dense_shape", axis=0)
    if inverted:
        expanded = op("Unsqueeze", ["attention_mask", "pad_axes"], "expanded_padding")
        expanded = op("Expand", [expanded, dense], "global_keep")
        padding = op("Cast", [expanded], "float_padding", to=TensorProto.FLOAT)
        inverse = op(
            "Sub", [const("one_float", 1, np.float32), padding], "inverted_padding"
        )
        is_pad = op("Cast", [inverse], "is_padding", to=TensorProto.BOOL)
        mask = op("Where", [is_pad, neg, inverse], "global_mask")
    else:
        padding = op("Cast", ["attention_mask"], "bool_padding", to=TensorProto.BOOL)
        expanded = op("Unsqueeze", [padding, "pad_axes"], "expanded_padding")
        keep = op("Expand", [expanded, dense], "global_keep")
        mask = op("Where", [keep, zero, neg], "global_mask")
    if radius is not None:
        position = op("Range", ["zero_i", length, "one_i"], "positions")
        queries = op("Unsqueeze", [position, "axes1"], "query_positions")
        keys = op("Unsqueeze", [position, "axes0"], "key_positions")
        distance = op("Sub", [queries, keys], "difference")
        distance = op("Abs", [distance], "distance")
        keep = op("LessOrEqual", [distance, const("radius", radius)], "local_keep")
        if inverted:
            outside = op("Not", [keep], "outside")
            # Different finite fills exercise exact nested mask reconstruction.
            local_negative = const("local_negative", -100000000.0, np.float32)
            mask = op("Where", [outside, local_negative, mask], "local_mask")
        else:
            keep = op("And", ["global_keep", keep], "combined_keep")
            mask = op("Where", [keep, zero, neg], "local_mask")
    kt = op("Transpose", ["k"], "kt", perm=[0, 1, 3, 2])
    qs = op("Mul", ["q", "q_scale"], "qs")
    ks = op("Mul", [kt, "k_scale"], "ks")
    scores = op("MatMul", [qs, ks], "scores")
    scores = op("Add", [scores, mask], "masked")
    probability = op("Softmax", [scores], "probability", axis=-1)
    if guard:
        isnan = op("IsNaN", [probability], "isnan")
        probability = op("Where", [isnan, zero, probability], "guarded")
    op("MatMul", [probability, "v"], "output")
    inputs = [
        helper.make_tensor_value_info(
            name, TensorProto.FLOAT, ["batch", 2, "sequence", 4]
        )
        for name in ["q", "k", "v"]
    ]
    inputs.append(
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch", "sequence"]
        )
    )
    graph = helper.make_graph(
        nodes,
        "attention",
        inputs,
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, ["batch", 2, "sequence", 4]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=9
    )
    return onnx.shape_inference.infer_shapes(model)


def session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def feeds(batch, length, all_masked=False):
    rng = np.random.default_rng(770 + batch + length)
    values = {
        name: rng.normal(size=(batch, 2, length, 4)).astype(np.float32)
        for name in ["q", "k", "v"]
    }
    mask = np.ones((batch, length), np.int64)
    padding_stride = 3
    if length > padding_stride:
        mask[0, 1::padding_stride] = (
            0  # irregular padding exercises key-axis broadcasting
        )
    if batch > 1:
        mask[1, length // 2 :] = 0
    if all_masked:
        mask[-1] = 0
    values["attention_mask"] = mask
    return values


@unittest.skipIf(ort is None, "onnxruntime CPU is needed for execution tests")
class BlockedNumerics(unittest.TestCase):
    def test_static_batch_rejects_unqualified_batches(self):
        inputs = [
            helper.make_tensor_value_info(
                name, TensorProto.INT64, ["batch", "sequence"]
            )
            for name in ["input_ids", "attention_mask"]
        ]
        output = helper.make_tensor_value_info(
            "output", TensorProto.INT64, ["batch", "sequence"]
        )
        graph = helper.make_graph(
            [helper.make_node("Mul", [value.name for value in inputs], ["output"])],
            "batch_contract",
            inputs,
            [output],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=9
        )
        original = model.SerializeToString()
        restricted = specialize_batch(model, 1)
        self.assertEqual(model.SerializeToString(), original)
        self.assertTrue(
            all(
                value.type.tensor_type.shape.dim[0].dim_value == 1
                for value in restricted.graph.input
            )
        )
        runtime = session(restricted)
        valid = {value.name: np.ones((1, 5), np.int64) for value in inputs}
        np.testing.assert_array_equal(
            runtime.run(None, valid)[0], np.ones((1, 5), np.int64)
        )
        for wrong_input in ["input_ids", "attention_mask"]:
            invalid = dict(valid)
            invalid[wrong_input] = np.ones((2, 5), np.int64)
            with self.assertRaises(Exception) as error:
                runtime.run(None, invalid)
            self.assertIn("Expected: 1", str(error.exception))
        with self.assertRaises(ValueError):
            specialize_batch(restricted, 2)
        with self.assertRaises(ValueError):
            specialize_batch(model, 0)
        with self.assertRaises(ValueError):
            specialize_batch(attention_model(), 1)

    def test_dynamic_batch_tail_and_local_absolute_positions(self):
        for radius, inverted in [(None, False), (2, False), (2, True)]:
            with self.subTest(radius=radius, inverted=inverted):
                original = attention_model(radius, inverted=inverted)
                frozen = original.SerializeToString()
                blocked, receipt = rewrite_model(original)
                self.assertEqual(original.SerializeToString(), frozen)
                self.assertEqual(receipt["attention_blocks"], 1)
                reference, candidate = session(original), session(blocked)
                for batch, length in [(1, 1), (2, 255), (1, 256), (2, 257), (1, 513)]:
                    inputs = feeds(batch, length)
                    before = reference.run(None, inputs)[0]
                    after = candidate.run(None, inputs)[0]
                    np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-6)
                    self.assertEqual(after.shape, (batch, 2, length, 4))

    def test_adaptive_memory_limit_still_uses_complete_keys(self):
        original = attention_model(radius=3)
        blocked, _ = rewrite_model(original, block_size=512, max_score_bytes=16384)
        inputs = feeds(2, 257)
        # B*H*K=1028, hence the effective query block is three rows, not 512.
        np.testing.assert_allclose(
            session(blocked).run(None, inputs)[0],
            session(original).run(None, inputs)[0],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_all_masked_rows_preserve_nan_guard_and_finite_fill(self):
        for negative, guard in [
            (-np.inf, True),
            (-np.finfo(np.float32).max, True),
            (-1000000000.0, False),
        ]:
            with self.subTest(negative=negative, guard=guard):
                original = attention_model(guard=guard, negative=negative)
                blocked, _ = rewrite_model(original)
                inputs = feeds(2, 257, all_masked=True)
                before = session(original).run(None, inputs)[0]
                after = session(blocked).run(None, inputs)[0]
                np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-6)
                self.assertTrue(np.isfinite(after).all())
                if negative == -np.inf:
                    self.assertTrue((after[-1] == 0).all())

    def test_nonbinary_padding_is_rejected(self):
        blocked, _ = rewrite_model(attention_model())
        for invalid in [-1, 2]:
            inputs = feeds(1, 17)
            inputs["attention_mask"][0, 3] = invalid
            with self.assertRaises(Exception) as error:
                session(blocked).run(None, inputs)
            self.assertIn("Reshape", str(error.exception))

    def test_impossible_budget_fails_before_score_allocation(self):
        blocked, _ = rewrite_model(attention_model(), max_score_bytes=16)
        with self.assertRaises(Exception) as error:
            session(blocked).run(None, feeds(2, 17))
        self.assertIn("Reshape", str(error.exception))

    def test_mismatched_self_attention_shapes_are_rejected_at_runtime(self):
        blocked, _ = rewrite_model(attention_model())
        inputs = feeds(1, 17)
        inputs["attention_mask"] = np.ones((1, 16), np.int64)
        with self.assertRaises(Exception) as error:
            session(blocked).run(None, inputs)
        self.assertIn("Reshape", str(error.exception))


class BlockedContracts(unittest.TestCase):
    def test_scales_survive_and_dense_mask_is_removed(self):
        original = attention_model(2)
        blocked, _ = rewrite_model(original)
        original_nodes = {n.name: n for n in original.graph.node}
        nodes = {n.name: n for n in blocked.graph.node}
        for name in ["qs", "ks"]:
            self.assertEqual(
                nodes[name].SerializeToString(),
                original_nodes[name].SerializeToString(),
            )
        self.assertTrue(any(n.op_type == "Loop" for n in blocked.graph.node))
        for name in [
            "global_keep",
            "global_mask",
            "difference",
            "distance",
            "local_keep",
            "local_mask",
        ]:
            self.assertNotIn(name, nodes)
        onnx.checker.check_model(blocked, full_check=True)

    def test_unknown_mask_and_precision_fail_without_mutating_input(self):
        for variant in [
            "or",
            "query_padding",
            "nonzero_positions",
            "half_weight",
            "missing_type",
        ]:
            with self.subTest(variant=variant):
                model = attention_model(2)
                if variant == "or":
                    next(
                        n for n in model.graph.node if n.name == "combined_keep"
                    ).op_type = "Or"
                elif variant == "query_padding":
                    next(
                        x for x in model.graph.initializer if x.name == "pad_axes"
                    ).CopyFrom(
                        numpy_helper.from_array(np.array([1, 3], np.int64), "pad_axes")
                    )
                elif variant == "nonzero_positions":
                    next(n for n in model.graph.node if n.op_type == "Range").input[
                        0
                    ] = "one_i"
                elif variant == "half_weight":
                    model.graph.initializer.append(
                        numpy_helper.from_array(np.ones(2, np.float16), "half")
                    )
                else:
                    items = [x for x in model.graph.value_info if x.name != "qs"]
                    del model.graph.value_info[:]
                    model.graph.value_info.extend(items)
                before = model.SerializeToString()
                with self.assertRaises(ValueError):
                    rewrite_model(model)
                self.assertEqual(model.SerializeToString(), before)

    def test_external_dense_mask_or_probability_consumers_are_refused(self):
        for output in ["global_mask", "probability"]:
            model = attention_model()
            model.graph.output.append(
                helper.make_tensor_value_info(output, TensorProto.FLOAT, [None] * 4)
            )
            with self.assertRaises(ValueError):
                rewrite_model(model)

    def test_position_bounds_require_unmodified_nonnegative_range(self):
        for variant in ["valid", "nonzero_bound", "multiply", "offset", "narrow_cast"]:
            with self.subTest(variant=variant):
                model = attention_model(2)
                index = next(
                    i
                    for i, n in enumerate(model.graph.node)
                    if n.name == "combined_keep"
                )
                nodes = []
                position = "query_positions"
                if variant in {"multiply", "offset"}:
                    nodes.append(
                        helper.make_node(
                            "Mul" if variant == "multiply" else "Add",
                            [position, "one_i"],
                            ["modified_positions"],
                            name="modified_positions",
                        )
                    )
                    position = "modified_positions"
                elif variant == "narrow_cast":
                    nodes.append(
                        helper.make_node(
                            "Cast",
                            [position],
                            ["narrow_positions"],
                            name="narrow_positions",
                            to=TensorProto.INT8,
                        )
                    )
                    position = "narrow_positions"
                bound = "one_i" if variant == "nonzero_bound" else "zero_i"
                nodes.append(
                    helper.make_node(
                        "GreaterOrEqual", [position, bound], ["bounds"], name="bounds"
                    )
                )
                nodes.append(
                    helper.make_node(
                        "And",
                        ["local_keep", "bounds"],
                        ["bounded_keep"],
                        name="bounded_keep",
                    )
                )
                old = list(model.graph.node)
                old[index].input[1] = "bounded_keep"
                del model.graph.node[:]
                model.graph.node.extend(old[:index] + nodes + old[index:])
                if variant == "valid":
                    rewrite_model(model)
                else:
                    with self.assertRaises(ValueError):
                        rewrite_model(model)

    def test_local_position_axes_cannot_be_identical(self):
        model = attention_model(2)
        next(n for n in model.graph.node if n.name == "difference").input[
            1
        ] = "query_positions"
        with self.assertRaises(ValueError):
            rewrite_model(model)

    def test_unmatched_or_partially_rewritten_attention_is_refused(self):
        for kind, domain in [
            ("Softmax", ""),
            ("Attention", ""),
            ("CKFlashAttention", "com.ck"),
        ]:
            with self.subTest(kind=kind):
                model = attention_model()
                model.graph.node.append(
                    helper.make_node(
                        kind, ["q"], ["unmatched"], name="unmatched", domain=domain
                    )
                )
                model.graph.value_info.append(
                    helper.make_tensor_value_info(
                        "unmatched", TensorProto.FLOAT, [None] * 4
                    )
                )
                before = model.SerializeToString()
                with self.assertRaises(ValueError):
                    rewrite_model(model)
                self.assertEqual(model.SerializeToString(), before)

    def test_invalid_options_and_already_blocked_are_refused(self):
        for options in [
            {"block_size": 0},
            {"block_size": 1024},
            {"max_score_bytes": 0},
            {"max_score_bytes": 2**33},
        ]:
            with self.assertRaises(ValueError):
                rewrite_model(attention_model(), **options)
        blocked, _ = rewrite_model(attention_model())
        with self.assertRaises(ValueError):
            rewrite_model(blocked)


def instrument(model):
    model = copy.deepcopy(model)
    prefix = "__blocked_attention_0_"
    model.graph.output.append(
        helper.make_tensor_value_info(prefix + "crop_eligible", TensorProto.BOOL, [])
    )
    loop = next(node for node in model.graph.node if node.op_type == "Loop")
    body = next(attribute.g for attribute in loop.attribute if attribute.name == "body")
    for name in ("key_start", "key_end"):
        body.output.append(
            helper.make_tensor_value_info(
                prefix + "body_" + name, TensorProto.INT64, []
            )
        )
        output = prefix + name + "_debug_scan"
        loop.output.append(output)
        model.graph.output.append(
            helper.make_tensor_value_info(output, TensorProto.INT64, [None])
        )
    onnx.checker.check_model(model)
    return model


@unittest.skipIf(ort is None, "onnxruntime CPU is needed for execution tests")
class LocalCropNumerics(unittest.TestCase):
    def check(self, model, inputs, expected_fast, **options):
        candidate, receipt = rewrite_model(model, **options)
        runtime = session(instrument(candidate))
        after, fast, starts, ends = runtime.run(None, inputs)
        reference = session(model).run(None, inputs)[0]
        np.testing.assert_allclose(
            after, reference, atol=2e-6, rtol=2e-6, equal_nan=True
        )
        self.assertEqual(bool(fast), expected_fast)
        if not expected_fast:
            np.testing.assert_array_equal(starts, np.zeros_like(starts))
            np.testing.assert_array_equal(
                ends, np.full_like(ends, inputs["q"].shape[2])
            )
        self.assertFalse(receipt["real_model_qualification"])
        return after, starts, ends

    def test_unpadded_b1_b2_radius64_and_odd_tail(self):
        for negative in (-np.inf, -np.finfo(np.float32).max):
            for batch in (1, 2):
                for length in (1, 63, 64, 65, 255, 256, 257, 513, 769):
                    with self.subTest(negative=negative, batch=batch, length=length):
                        inputs = feeds(batch, length)
                        inputs["attention_mask"][:] = 1
                        after, starts, ends = self.check(
                            attention_model(64, negative=negative), inputs, True
                        )
                        queries = np.arange(0, length, 256)
                        np.testing.assert_array_equal(
                            starts, np.maximum(0, queries - 64)
                        )
                        np.testing.assert_array_equal(
                            ends, np.minimum(length, queries + 256 + 64)
                        )
                        self.assertLessEqual(int(np.max(ends - starts)), 256 + 128)
                        self.assertEqual(after.shape, (batch, 2, length, 4))

    def test_local_padding_falls_back_for_entire_batch(self):
        for negative in (-np.inf, -np.finfo(np.float32).max, -1e8):
            for batch in (1, 2):
                for padding in ("left", "right", "all", "irregular"):
                    with self.subTest(negative=negative, batch=batch, padding=padding):
                        inputs = feeds(batch, 513)
                        if padding != "irregular":
                            inputs["attention_mask"][:] = 1
                            region = (
                                slice(None, 200)
                                if padding == "left"
                                else slice(200, None)
                            )
                            inputs["attention_mask"][-1, region] = 0
                            if padding == "all":
                                inputs["attention_mask"][-1] = 0
                        self.check(
                            attention_model(64, negative=negative), inputs, False
                        )

    def test_nested_finite_fill_preserves_scaling(self):
        inputs = feeds(2, 513)
        inputs["attention_mask"][:] = 1
        inputs["q"] *= 0.001
        inputs["k"] *= 0.001
        self.check(attention_model(64, inverted=True, negative=-np.inf), inputs, True)
        self.check(attention_model(64, negative=-65504.0), inputs, True)

    def test_small_finite_fill_remains_refused_by_shared_matcher(self):
        with self.assertRaises(ValueError):
            rewrite_model(attention_model(64, negative=-5.0))

    def test_radius_zero_retains_only_current_query_block_keys(self):
        inputs = feeds(1, 513)
        inputs["attention_mask"][:] = 1
        _, starts, ends = self.check(attention_model(0), inputs, True)
        np.testing.assert_array_equal(starts, [0, 256, 512])
        np.testing.assert_array_equal(ends, [256, 512, 513])

    def test_large_qk_and_nonfinite_input_fall_back(self):
        for field, value in (
            ("q", 1e20),
            ("k", 1e20),
            ("q", np.nan),
            ("k", np.inf),
            ("v", np.nan),
            ("v", np.inf),
        ):
            with self.subTest(field=field, value=value):
                inputs = feeds(1, 513)
                inputs["attention_mask"][:] = 1
                inputs[field][0, 0, 11, 1] = value
                self.check(attention_model(64, negative=-65504.0), inputs, False)

    def test_adaptive_full_key_budget_retained(self):
        inputs = feeds(2, 257)
        inputs["attention_mask"][:] = 1
        _, starts, ends = self.check(
            attention_model(2), inputs, True, block_size=512, max_score_bytes=16384
        )
        queries = np.arange(0, 257, 3)
        np.testing.assert_array_equal(starts, np.maximum(0, queries - 2))
        np.testing.assert_array_equal(ends, np.minimum(257, queries + 3 + 2))
        self.assertLessEqual(int(np.max(ends - starts)), 7)

    def test_global_attention_keeps_complete_keys_and_values(self):
        for negative in (-np.inf, -np.finfo(np.float32).max):
            rewritten, _ = rewrite_model(attention_model(negative=negative))
            self.assertFalse(any("crop_" in node.name for node in rewritten.graph.node))
            loop = next(node for node in rewritten.graph.node if node.op_type == "Loop")
            body = next(
                attribute.g for attribute in loop.attribute if attribute.name == "body"
            )
            products = [node for node in body.node if node.op_type == "MatMul"]
            self.assertEqual([node.input[1] for node in products], ["ks", "v"])
            self.assertEqual(sum(node.op_type == "Slice" for node in body.node), 1)

    def test_original_is_not_mutated(self):
        original = attention_model(64)
        before = original.SerializeToString()
        rewrite_model(original)
        self.assertEqual(before, original.SerializeToString())


if __name__ == "__main__":
    unittest.main()
