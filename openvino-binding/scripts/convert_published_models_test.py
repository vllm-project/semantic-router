"""Guard the boundary between published ONNX outputs and the C++ binding."""

import json
import tempfile
import unittest
from pathlib import Path

import openvino as ov
from convert_published_models import (
    graph_contract,
    owned_tokenizer_contract,
    read_sources,
)
from create_owned_fixture import inspect_tokenizer, make_tokenizer
from openvino import opset13 as ops
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer


class PublishedModelConversionTest(unittest.TestCase):
    def test_saved_tokenizer_reloads_for_published_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            make_tokenizer(4).save_pretrained(directory)
            tokenizer = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
            self.assertEqual(
                owned_tokenizer_contract(tokenizer),
                {"pad_token_id": 0, "end_token_ids": [2]},
            )
            evidence = inspect_tokenizer(
                convert_tokenizer(tokenizer, with_detokenizer=False), 4
            )
            self.assertEqual(evidence["probe_tokens"], 98)

    def test_owned_tokenizer_export_preserves_original_counts_and_declared_ids(self):
        tokenizer = make_tokenizer(4)
        tokenizer.model_max_length = 64
        tokenizer.init_kwargs["model_max_length"] = 64
        tokenizer.backend_tokenizer.enable_truncation(7)
        self.assertEqual(
            owned_tokenizer_contract(tokenizer),
            {"pad_token_id": 0, "end_token_ids": [2]},
        )
        evidence = inspect_tokenizer(
            convert_tokenizer(tokenizer, with_detokenizer=False), 4
        )
        self.assertEqual(evidence["probe_tokens"], 98)
        tokenizer.pad_token = None
        with self.assertRaisesRegex(ValueError, "padding"):
            owned_tokenizer_contract(tokenizer)

    def graph(self, shape, inputs=("input_ids", "attention_mask", "position_ids")):
        parameters = [ops.parameter([1, -1], ov.Type.i64, name=name) for name in inputs]
        output = ops.constant(0.0, ov.Type.f32)
        output = ops.broadcast(output, ops.constant(shape, ov.Type.i64))
        return ov.Model([output], parameters)

    def test_token_embeddings_and_pooled_embeddings_keep_their_dimensions(self):
        for shape in ([1, 4, 8], [1, 8]):
            with self.subTest(shape=shape):
                receipt = graph_contract(
                    self.graph(shape), "Embedding", {"hidden_size": 8}
                )
                self.assertEqual(receipt["output_rank"], len(shape))
                self.assertEqual(receipt["dimension"], 8)
        with self.assertRaisesRegex(ValueError, "output size"):
            graph_contract(self.graph([1, 4, 7]), "Embedding", {"hidden_size": 8})

    def test_classifier_requires_label_sized_logits_and_explicit_positions(self):
        graph = self.graph([1, 2])
        graph_contract(graph, "Domain", {"id2label": {"0": "a", "1": "b"}})
        self.assertEqual(graph.output(0).get_any_name(), "logits")
        with self.assertRaisesRegex(ValueError, "unsupported graph inputs"):
            graph_contract(
                self.graph([1, 2], ("input_ids", "attention_mask")),
                "Domain",
                {"id2label": {"0": "a", "1": "b"}},
            )

    def test_internal_aliases_preserve_the_published_input_contract(self):
        graph = self.graph([1, 4, 8])
        graph.input(1).get_tensor().set_names(
            {"attention_mask", "__blocked_attention_0_integer_mask"}
        )
        receipt = graph_contract(graph, "Embedding", {"hidden_size": 8})
        self.assertEqual(
            receipt["inputs"], ["input_ids", "attention_mask", "position_ids"]
        )
        self.assertEqual(graph.input(1).get_names(), {"attention_mask"})

    def test_manifest_rejects_missing_duplicate_and_unpinned_models(self):
        models = [
            {"name": name, "revision": "a" * 40} for name in ("Domain", "Embedding")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps({"provider": "ort", "models": models}))
            self.assertEqual(read_sources(path), models)
            for invalid in (
                models[:1],
                [models[0], models[0]],
                [models[0], {"name": "Embedding", "revision": "main"}],
            ):
                path.write_text(json.dumps({"provider": "ort", "models": invalid}))
                with self.assertRaises(ValueError):
                    read_sources(path)


if __name__ == "__main__":
    unittest.main()
