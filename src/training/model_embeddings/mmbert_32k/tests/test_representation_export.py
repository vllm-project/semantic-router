"""Actual small dynamic ONNX graphs, checked against native CPU execution."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401 -- numerical validation dependency
    import onnxscript  # noqa: F401 -- exporter dependency checked here
    import torch
    from onnx.reference import ReferenceEvaluator
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.representation_contract import (
        set_vela_representation_contract,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch, transformers, onnx and onnxscript")
class RepresentationExportTest(unittest.TestCase):
    def test_export_variants_cannot_mix_source_weights_or_tokenizers(self):
        root = Path(__file__).resolve().parents[5]
        spec = importlib.util.spec_from_file_location(
            "vela_artifacts", root / "onnx-binding/scripts/onnx_artifacts.py"
        )
        artifacts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(artifacts)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "onnx"
            source.mkdir()
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (source / name).write_bytes(b"original")
            identity = {
                "files_sha256": artifacts.source_snapshot_sha256(source),
                "layers": [1, 2],
            }
            artifacts.prepare_export_directory(output, identity)
            artifacts.prepare_export_directory(output, identity)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (source / name).write_bytes(b"changed")
                changed = dict(
                    identity, files_sha256=artifacts.source_snapshot_sha256(source)
                )
                with self.assertRaisesRegex(ValueError, "source or exit inventory"):
                    artifacts.prepare_export_directory(output, changed)
                (source / name).write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "source or exit inventory"):
                artifacts.prepare_export_directory(output, dict(identity, layers=[1]))
            (output / "source_identity.json").unlink()
            (output / "model.onnx").write_bytes(b"untracked")
            with self.assertRaisesRegex(ValueError, "no source identity"):
                artifacts.prepare_export_directory(output, identity)

    def test_external_data_hashes_nested_tensors_and_rejects_escapes(self):
        root = Path(__file__).resolve().parents[5]
        spec = importlib.util.spec_from_file_location(
            "vela_artifacts", root / "onnx-binding/scripts/onnx_artifacts.py"
        )
        artifacts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(artifacts)
        tensor = onnx.TensorProto()
        tensor.name = "external"
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = "location", "model.data"
        node = onnx.helper.make_node("Constant", [], ["value"], value=tensor)
        graph = onnx.helper.make_graph([node], "nested_tensor", [], [])
        model = onnx.helper.make_model(graph)
        model.graph.node[0].doc_string = "debug stack must be removed"
        model.graph.node[0].metadata_props.add(key="debug", value="private path")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            weights = Path(directory) / "model.data"
            weights.write_bytes(b"weights")
            artifacts.strip_debug_annotations(model)
            self.assertFalse(model.graph.node[0].doc_string)
            self.assertFalse(model.graph.node[0].metadata_props)
            self.assertEqual(
                artifacts.external_data_sha256(model, path),
                {"model.data": artifacts.sha256(weights)},
            )
            location = model.graph.node[0].attribute[0].t.external_data[0]
            for invalid in ("../secret", "/secret", "C:\\secret", ""):
                location.value = invalid
                with self.assertRaises(ValueError):
                    artifacts.external_data_sha256(model, path)
            with tempfile.TemporaryDirectory() as outside:
                (Path(directory) / "escape").symlink_to(
                    outside, target_is_directory=True
                )
                location.value = "escape/secret"
                with self.assertRaises(ValueError):
                    artifacts.external_data_sha256(model, path)

    def test_dynamic_embedding_and_reranker_match_native(self):
        root = Path(__file__).resolve().parents[5]
        spec = importlib.util.spec_from_file_location(
            "vela_export", root / "onnx-binding/scripts/export_2d_matryoshka.py"
        )
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        config = ModernBertConfig(
            vocab_size=32,
            pad_token_id=0,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=256,
            global_attn_every_n_layers=2,
            local_attention=4,
            reference_compile=False,
        )
        torch.manual_seed(7)
        encoder = ModernBertModel(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            for task in ("embedding", "reranker"):
                set_vela_representation_contract(encoder.config, task)
                prefix = exporter.PhysicalPrefixEncoder(copy.deepcopy(encoder), 1, task)
                model = (
                    prefix
                    if task == "embedding"
                    else exporter.RerankerExit(prefix, torch.nn.Linear(4, 1), 4)
                )
                path = Path(directory) / task / "model.onnx"
                result = exporter.export_graph(
                    model, config, path, opset=18, device="cpu"
                )
                self.assertEqual(len(result["graph_sha256"]), 64)
                verification = exporter.verify_graph(
                    model,
                    config,
                    path,
                    lengths=[3, 17, 129],
                    dimensions=[4],
                    task=task,
                    precision="fp32",
                )
                self.assertEqual(len(verification["cases"]), 6)
                graph = onnx.load(path)
                self.assertFalse(
                    any(
                        node.doc_string or node.metadata_props
                        for node in graph.graph.node
                    )
                )
                runtime = ReferenceEvaluator(str(path))
                for batch, length in ((1, 3), (2, 17), (3, 129)):
                    ids = torch.full((batch, length), 10, dtype=torch.long)
                    mask = torch.ones_like(ids)
                    if batch > 1:
                        ids[-1, length // 2 :] = 0
                        mask[-1, length // 2 :] = 0
                    with torch.inference_mode():
                        expected = model(ids, mask).numpy()
                    actual = runtime.run(
                        None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
                    )[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
                with self.assertRaises(FileExistsError):
                    exporter.export_graph(model, config, path, opset=18, device="cpu")


if __name__ == "__main__":
    unittest.main()
