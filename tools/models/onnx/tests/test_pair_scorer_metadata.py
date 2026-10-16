"""Real portable pair scores carry verified physical exit identity."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import onnx
    import onnxruntime  # noqa: F401 -- actual CPU validation dependency
    import onnxscript  # noqa: F401 -- actual export dependency
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast
except ImportError:
    torch = None


@unittest.skipIf(
    torch is None, "requires torch, transformers, onnx, ORT and onnxscript"
)
class PairScorerMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "pair_exporter",
            Path(__file__).resolve().parents[1] / "export_2d_matryoshka.py",
        )
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source"
        backbone = cls.root / "backbone"
        cls.config = ModernBertConfig(
            vocab_size=32,
            pad_token_id=0,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            global_attn_every_n_layers=2,
            local_attention=4,
            reference_compile=False,
        )
        contract_module = importlib.import_module("mmbert_32k.representation_contract")
        contract_module.set_vela_representation_contract(cls.config, "reranker")
        torch.manual_seed(7)
        ModernBertModel(cls.config).save_pretrained(backbone)
        cls.model = cls.exporter.Matryoshka2DReranker(
            str(backbone),
            layer_indices=[1, 2],
            dim_indices=[4, 8],
            use_flash_attn=False,
            torch_dtype=torch.float32,
        ).eval()
        cls.model.save_pretrained(str(cls.source))
        vocab = {f"token_{index}": index for index in range(32)}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="token_1")),
            pad_token="token_0",
            unk_token="token_1",
        )
        tokenizer.save_pretrained(cls.source)
        cls.base_output = cls.root / "original-export"
        cls.args = SimpleNamespace(
            model=str(cls.source),
            revision=None,
            task="reranker",
            layers=[1],
            dimensions=[4],
            precision="fp32",
            device="cpu",
            output=str(cls.base_output),
            opset=18,
            resume=False,
            verify_only=False,
            validation_lengths=[2, 7, 17],
        )
        with mock.patch.object(cls.exporter, "parse_args", return_value=cls.args):
            cls.exporter.main()
        prefix = cls.exporter.PhysicalPrefixEncoder(
            copy.deepcopy(cls.model.encoder), 1, "reranker"
        )
        cls.reference = cls.exporter.RerankerExit(
            prefix, copy.deepcopy(cls.model.layer_heads["1"]["4"]), 4
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "export"
        shutil.copytree(self.base_output, self.output)
        self.path = self.output / "layer-1/dim-4/model.onnx"

    def verify(self):
        return self.exporter.verify_graph(
            self.reference,
            self.config,
            self.path,
            lengths=[2, 7, 17],
            dimensions=[4],
            task="reranker",
            precision="fp32",
        )

    def test_actual_export_identity_and_padded_cpu_scores(self):
        graph = onnx.load(self.path, load_external_data=False)
        entry = [
            value
            for value in graph.metadata_props
            if value.key == self.exporter.PAIR_SCORER_METADATA_KEY
        ]
        self.assertEqual(len(entry), 1)
        self.assertEqual(
            json.loads(entry[0].value),
            {
                "version": 1,
                "layer": 1,
                "dimension": 4,
                "score_type": "relevance_logit",
            },
        )
        self.assertFalse(
            any(node.doc_string or node.metadata_props for node in graph.graph.node)
        )
        results = self.verify()["cases"]
        self.assertEqual(len(results), 6)
        for row in results:
            self.assertLess(row["max_abs_output_error"], 2e-4)

    def test_resume_and_verify_only_accept_matching_metadata_without_rewrite(self):
        for mode in ("resume", "verify_only"):
            args = copy.copy(self.args)
            args.output = str(self.output)
            setattr(args, mode, True)
            before = self.path.read_bytes()
            with mock.patch.object(self.exporter, "parse_args", return_value=args):
                self.exporter.main()
            self.assertEqual(self.path.read_bytes(), before)

    def test_resume_and_verify_only_reject_unlabeled_existing_graph(self):
        graph = onnx.load(self.path, load_external_data=False)
        del graph.metadata_props[:]
        onnx.save_model(graph, self.path)
        before = self.path.read_bytes()
        for mode in ("resume", "verify_only"):
            args = copy.copy(self.args)
            args.output = str(self.output)
            setattr(args, mode, True)
            with (
                mock.patch.object(self.exporter, "parse_args", return_value=args),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                self.exporter.main()
            self.assertEqual(self.path.read_bytes(), before)

    def test_wrong_layer_dimension_score_type_and_json_are_rejected(self):
        expected = self.reference.pair_scorer_contract()
        values = [
            "not json",
            "null",
            json.dumps({**expected, "layer": 2}),
            json.dumps({**expected, "dimension": 8}),
            json.dumps({**expected, "score_type": "probability"}),
            json.dumps({**expected, "version": True}),
            json.dumps({**expected, "extra": "unverified"}),
        ]
        for value in values:
            with self.subTest(value=value):
                graph = onnx.load(self.path, load_external_data=False)
                graph.metadata_props[0].value = value
                onnx.save_model(graph, self.path)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    self.verify()

    def test_duplicate_metadata_is_rejected(self):
        graph = onnx.load(self.path, load_external_data=False)
        graph.metadata_props.add().CopyFrom(graph.metadata_props[0])
        onnx.save_model(graph, self.path)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.verify()

    def test_head_dimension_and_actual_prefix_depth_must_match(self):
        prefix = self.exporter.PhysicalPrefixEncoder(
            copy.deepcopy(self.model.encoder), 1, "reranker"
        )
        with self.assertRaisesRegex(ValueError, "scalar head"):
            self.exporter.RerankerExit(prefix, torch.nn.Linear(8, 1), 4)
        prefix.encoder.config.num_hidden_layers = 2
        with self.assertRaisesRegex(ValueError, "physical depth"):
            self.exporter.RerankerExit(prefix, torch.nn.Linear(4, 1), 4)

    def test_embedding_export_does_not_claim_pair_semantics(self):
        encoder = copy.deepcopy(self.model.encoder)
        contract_module = importlib.import_module("mmbert_32k.representation_contract")
        contract_module.set_vela_representation_contract(encoder.config, "embedding")
        prefix = self.exporter.PhysicalPrefixEncoder(encoder, 1, "embedding")
        path = self.output / "embedding.onnx"
        self.exporter.export_graph(prefix, self.config, path, opset=18, device="cpu")
        graph = onnx.load(path, load_external_data=False)
        self.assertFalse(
            any(
                item.key == self.exporter.PAIR_SCORER_METADATA_KEY
                for item in graph.metadata_props
            )
        )
        graph.metadata_props.add(
            key=self.exporter.PAIR_SCORER_METADATA_KEY,
            value=json.dumps(self.reference.pair_scorer_contract()),
        )
        with self.assertRaisesRegex(ValueError, "Embedding graph"):
            self.exporter.validate_graph_metadata(graph, prefix)


if __name__ == "__main__":
    unittest.main()
