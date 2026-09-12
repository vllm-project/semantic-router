"""Task-head parity and real dynamic exports, independent of remote checkpoints."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import onnxruntime  # noqa: F401 -- require an actual runtime for export tests
    import onnxscript  # noqa: F401 -- required by the dynamo exporter
    import torch
    from transformers import (
        ModernBertConfig,
        ModernBertForSequenceClassification,
        ModernBertForTokenClassification,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    exporter = importlib.import_module("export_classifier")
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch, transformers, onnxscript and ORT")
class ClassifierExportTest(unittest.TestCase):
    def test_token_validation_preserves_decisions_and_confidence(self):
        expected = torch.tensor([[5.0, 0.0, -50.0]]).numpy()
        dormant_rounding = expected.copy()
        dormant_rounding[0, 2] += 0.01
        exporter.verify_task_outputs(dormant_rounding, expected, True, False, "float32")
        changed_confidence = expected.copy()
        changed_confidence[0, 0] = 0.1
        with self.assertRaises(AssertionError):
            exporter.verify_task_outputs(
                changed_confidence, expected, True, False, "float32"
            )
        # A near tie can flip BIO tags while passing the probability tolerance.
        tied = torch.tensor([[0.0, 1e-6, -50.0]]).numpy()
        flipped = tied.copy()
        flipped[0, :2] = flipped[0, :2][::-1]
        for dtype in ("float32", "float16"):
            with self.assertRaises(AssertionError):
                exporter.verify_task_outputs(flipped, tied, True, False, dtype)

    def test_precision_variants_cannot_mix_source_artifacts(self):
        source = {"config.json": "config-a", "model.safetensors": "weights-a"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exporter.verify_output_source(output, source)
            (output / "export-float32.json").write_text(
                json.dumps({"source_artifacts": source})
            )
            exporter.verify_output_source(output, source)
            for name in source:
                different = dict(source, **{name: "changed"})
                with self.assertRaisesRegex(ValueError, "different or unpinned"):
                    exporter.verify_output_source(output, different)

    def test_half_encoder_keeps_mean_and_head_in_float32(self):
        class Encoder(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                hidden = torch.tensor(
                    [[[1, 0], [1.0009765625, 0], [1, 0]]], dtype=torch.float16
                )
                return SimpleNamespace(last_hidden_state=hidden)

        model = torch.nn.Module()
        model.config = SimpleNamespace(classifier_pooling="mean")
        model.model = Encoder()
        model.head = torch.nn.Identity()
        model.drop = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
        wrapper = exporter.ClassifierLogits(model, False)
        ids = torch.ones((1, 3), dtype=torch.long)
        actual = wrapper(ids, ids)
        reference = model.model(ids, ids).last_hidden_state.float().mean(1)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertGreater((actual - reference.half().float()).abs().max().item(), 0)

    def test_native_tasks_and_dynamic_padded_exports(self):
        torch.set_num_threads(2)
        for task in ("mean", "cls", "token"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(17)
                config = ModernBertConfig(
                    vocab_size=32,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    max_position_embeddings=256,
                    local_attention=4,
                    global_attn_every_n_layers=2,
                    num_labels=3,
                    pad_token_id=0,
                    classifier_pooling="cls" if task == "cls" else "mean",
                    reference_compile=False,
                )
                kind = (
                    ModernBertForTokenClassification
                    if task == "token"
                    else ModernBertForSequenceClassification
                )
                model = kind(config).eval()
                wrapper = exporter.ClassifierLogits(model, task == "token").eval()
                ids, mask = exporter.make_input([1, 6, 7, 2], 16, 2, 0)
                with torch.inference_mode():
                    torch.testing.assert_close(
                        wrapper(ids, mask), model(ids, mask).logits
                    )
                    batch = torch.export.Dim("batch", min=1, max=8)
                    length = torch.export.Dim("sequence", min=2, max=256)
                    graph = Path(directory) / "model.onnx"
                    torch.onnx.export(
                        wrapper,
                        (ids, mask),
                        str(graph),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["logits"],
                        dynamic_shapes=(
                            {0: batch, 1: length},
                            {0: batch, 1: length},
                        ),
                        opset_version=18,
                        dynamo=True,
                        external_data=True,
                    )
                results = exporter.verify_graph(
                    wrapper,
                    graph,
                    [1, 6, 7, 2],
                    SimpleNamespace(pad_token_id=0),
                    [2, 5, 65, 129],
                    task == "token",
                    False,
                    "float32",
                )
                self.assertEqual(len(results), 8)


if __name__ == "__main__":
    unittest.main()
