#!/usr/bin/env python3
"""Export explicit mmBERT exits; structural export is not a quality benchmark.

Embedding graphs return token representations [batch, sequence, hidden]. The
runtime applies FP32 masked mean, dimension truncation, then L2 normalization.
Reranker graphs return one [batch, 1] logit from an independent trained head.
The source must declare representation_contract; no old contract is guessed.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_precision import cast_parameters_preserving_buffers
from onnx_artifacts import (
    external_data_sha256,
    prepare_export_directory,
    sha256,
    source_snapshot_sha256,
    strip_debug_annotations,
)

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "src/training/model_embeddings"
sys.path.insert(0, str(TRAINING_ROOT))
from mmbert_32k.representation_contract import (  # noqa: E402
    read_representation_contract,
)
from mmbert_32k.representation_outputs import PhysicalPrefixEncoder  # noqa: E402
from mmbert_32k.reranker_model import Matryoshka2DReranker  # noqa: E402

MIN_CONTEXT_TOKENS = 2


class RerankerExit(nn.Module):
    """Physical prefix and selected CLS head; head computation stays FP32."""

    def __init__(self, encoder, head, dimension: int):
        super().__init__()
        self.encoder, self.head, self.dimension = encoder, head.float(), dimension

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids, attention_mask)
        return self.head(hidden[:, 0, : self.dimension].float())


def export_graph(model, config, path: Path, *, opset: int, device: str) -> dict:
    """Export dynamic axes and hash the graph plus all external tensor data."""
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    ids = torch.full((2, 16), 10, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    mask[1, 12:] = 0
    ids[1, 12:] = config.pad_token_id
    batch = torch.export.Dim("batch", min=1, max=32)
    sequence = torch.export.Dim("sequence", min=2, max=config.max_position_embeddings)
    with torch.inference_mode():
        example = model(ids, mask)
    torch.onnx.export(
        model,
        (ids, mask),
        str(path),
        dynamo=True,
        external_data=True,
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=[
            "logits" if isinstance(model, RerankerExit) else "last_hidden_state"
        ],
        dynamic_shapes={
            "input_ids": {0: batch, 1: sequence},
            "attention_mask": {0: batch, 1: sequence},
        },
    )
    graph = onnx.load(str(path), load_external_data=False)
    strip_debug_annotations(graph)
    onnx.save_model(graph, str(path))
    onnx.checker.check_model(str(path))
    return {
        "graph_sha256": sha256(path),
        "external_data_sha256": external_data_sha256(graph, path),
        "example_output_shape": list(example.shape),
        "example_output_dtype": str(example.dtype),
        "numerical_validation": "pending; ONNX checker is structural validation only",
    }


def verify_graph(reference, config, path, *, lengths, dimensions, task, precision):
    """Compare portable graphs with native FP32 weights, including padded batches."""
    import onnxruntime as ort  # noqa: PLC0415 -- optional validation dependency

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    reference = reference.cpu().float().eval()
    results = []
    for length in lengths:
        for batch in (1, 2):
            ids = torch.full((batch, length), 10, dtype=torch.long)
            mask = torch.ones_like(ids)
            if batch > 1:
                mask[-1, max(1, length * 2 // 3) :] = 0
                ids[mask == 0] = config.pad_token_id
            with torch.inference_mode():
                expected = reference(ids, mask).float().numpy()
            actual = session.run(
                None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
            )[0].astype(np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Graph returned invalid shape or nonfinite values")
            valid = mask.numpy().astype(bool) if task == "embedding" else slice(None)
            if precision == "fp32":
                np.testing.assert_allclose(
                    actual[valid], expected[valid], atol=2e-4, rtol=1e-4
                )
            entry = {
                "batch": batch,
                "sequence_length": length,
                "valid_tokens": mask.sum(1).tolist(),
                "output_shape": list(actual.shape),
                "max_abs_output_error": float(
                    np.max(np.abs(actual[valid] - expected[valid]))
                ),
            }
            if task == "embedding":
                pooled = []
                for values in (expected, actual):
                    attended = mask.numpy()[..., None].astype(np.float32)
                    pooled.append((values * attended).sum(1) / attended.sum(1))
                entry["dimension_checks"] = {}
                for dimension in dimensions:
                    vectors = []
                    for values in pooled:
                        sliced = values[:, :dimension]
                        norms = np.linalg.norm(sliced, axis=1, keepdims=True)
                        if not np.isfinite(sliced).all() or np.any(norms <= 0):
                            raise ValueError("Cannot normalize invalid/zero embeddings")
                        vectors.append(sliced / norms)
                    np.testing.assert_allclose(
                        vectors[1],
                        vectors[0],
                        atol=3e-3 if precision == "fp16" else 2e-4,
                        rtol=1e-3 if precision == "fp16" else 1e-4,
                    )
                    entry["dimension_checks"][str(dimension)] = {
                        "minimum_cosine": float(
                            (vectors[0] * vectors[1]).sum(-1).min()
                        ),
                        "max_abs_embedding_error": float(
                            np.abs(vectors[1] - vectors[0]).max()
                        ),
                    }
            else:
                expected_score = torch.from_numpy(expected).sigmoid().numpy()
                actual_score = torch.from_numpy(actual).sigmoid().numpy()
                np.testing.assert_allclose(
                    actual_score,
                    expected_score,
                    atol=3e-3 if precision == "fp16" else 2e-4,
                    rtol=1e-3 if precision == "fp16" else 1e-4,
                )
                entry["max_abs_sigmoid_error"] = float(
                    np.abs(actual_score - expected_score).max()
                )
            results.append(entry)
    return {
        "reference": "native FP32 source weights",
        "cases": results,
        "scope": "Numerical agreement on fixed inputs; not task quality or 32K validation",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision", help="Required full HF commit SHA for remote models"
    )
    parser.add_argument(
        "--task", choices=("embedding", "reranker"), default="embedding"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 11, 22])
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[768, 512, 256, 128, 64]
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", required=True, help="Separate ONNX artifact directory"
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate existing graphs and complete missing exits",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Revalidate all existing graphs; never regenerate them",
    )
    parser.add_argument(
        "--validation-lengths",
        type=int,
        nargs="+",
        default=[2, 63, 64, 65, 127, 128, 129, 512],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(8)
    source = Path(args.model)
    local = source.is_dir()
    if not local and re.fullmatch(r"[0-9a-f]{40}", args.revision or "") is None:
        raise ValueError("Remote model requires an immutable 40-character revision")
    output = Path(args.output)
    if local and source.resolve() == output.resolve():
        raise ValueError(
            "ONNX output must not overwrite the native checkpoint directory"
        )
    options = {
        "local_files_only": local,
        "revision": args.revision,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(args.model, **options)
    config.reference_compile = False
    contract = read_representation_contract(config, args.task)
    if contract is None:
        raise ValueError(
            "Source config must declare representation_contract before export"
        )
    if config.model_type != "modernbert":
        raise ValueError("Only native ModernBERT is supported")
    total_layers = config.num_hidden_layers
    if any(
        not MIN_CONTEXT_TOKENS <= length <= config.max_position_embeddings
        for length in args.validation_lengths
    ):
        raise ValueError("Validation lengths must fit the native context budget")
    if len(set(args.layers)) != len(args.layers) or any(
        not 1 <= value <= total_layers for value in args.layers
    ):
        raise ValueError("Requested layers must be unique and within source depth")
    if len(set(args.dimensions)) != len(args.dimensions) or any(
        not 1 <= value <= config.hidden_size for value in args.dimensions
    ):
        raise ValueError("Requested dimensions must be unique and within hidden size")
    source_identity = {
        "version": 1,
        "source": (
            {"files_sha256": source_snapshot_sha256(source)}
            if local
            else {
                "repository": args.model,
                "revision": args.revision,
            }
        ),
        "task": args.task,
        "available_layers": sorted(args.layers),
        "dimensions": sorted(args.dimensions),
        "opset": args.opset,
        "representation_contract": contract,
    }
    prepare_export_directory(output, source_identity)
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    if args.task == "reranker":
        if not local:
            raise ValueError(
                "Download the pinned reranker encoder and heads together before export"
            )
        source_model = Matryoshka2DReranker.from_pretrained(
            args.model, use_flash_attn=False, torch_dtype=torch.float32
        )
        encoder = source_model.encoder
        if set(args.layers) - set(source_model.layer_indices) or set(
            args.dimensions
        ) - set(source_model.dim_indices):
            raise ValueError("Requested exit has no trained independent head")
    else:
        source_model = None
        encoder = AutoModel.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            **options,
        )
    encoder.config.reference_compile = False
    encoder = encoder.float().eval()
    public_config = copy.deepcopy(config)
    public_config._name_or_path = ""
    public_config.save_pretrained(output)
    AutoTokenizer.from_pretrained(args.model, **options).save_pretrained(output)
    metadata = {
        "available_layers": args.layers,
        "dimensions": args.dimensions,
        "total_layers": total_layers,
        "hidden_size": config.hidden_size,
        "representation_contract": contract,
        "task": args.task,
        "precision": args.precision,
        "source_revision": args.revision,
        "source_identity": source_identity,
        "source_weights_sha256": (
            {
                path.name: sha256(path)
                for pattern in ("*.safetensors", "classification_heads.pt")
                for path in source.glob(pattern)
            }
            if local
            else {}
        ),
        "exporter_sha256": sha256(Path(__file__)),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
        },
        "models": {},
    }
    receipt = output / f"export-{args.precision}.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    filename = "model.onnx" if args.precision == "fp32" else "model_sdpa_fp16.onnx"
    for layer in args.layers:
        for dimension in args.dimensions if args.task == "reranker" else [None]:
            reference_prefix = PhysicalPrefixEncoder(
                copy.deepcopy(encoder), layer, args.task
            )
            prefix = cast_parameters_preserving_buffers(
                copy.deepcopy(reference_prefix), dtype
            )
            directory = output / f"layer-{layer}"
            model = prefix
            reference = reference_prefix
            if dimension is not None:
                directory /= f"dim-{dimension}"
                model = RerankerExit(
                    prefix,
                    copy.deepcopy(source_model.layer_heads[str(layer)][str(dimension)]),
                    dimension,
                )
                reference = RerankerExit(
                    reference_prefix,
                    copy.deepcopy(source_model.layer_heads[str(layer)][str(dimension)]),
                    dimension,
                )
            path = directory / filename
            relative_path = str(path.relative_to(output))
            if path.exists() and (args.resume or args.verify_only):
                graph = onnx.load(str(path), load_external_data=False)
                onnx.checker.check_model(str(path))
                result = {
                    "graph_sha256": sha256(path),
                    "external_data_sha256": external_data_sha256(graph, path),
                    "generation": "existing graph revalidated",
                }
                prior = previous.get("models", {}).get(relative_path)
                if prior is not None and any(
                    prior[key] != result[key]
                    for key in ("graph_sha256", "external_data_sha256")
                ):
                    raise ValueError(
                        "Existing graph/data differ from the recorded artifact hashes"
                    )
            elif args.verify_only:
                raise FileNotFoundError(f"Required graph is missing: {path}")
            else:
                result = export_graph(
                    model, config, path, opset=args.opset, device=args.device
                )
            result["numerical_validation"] = verify_graph(
                reference,
                config,
                path,
                lengths=args.validation_lengths,
                dimensions=args.dimensions,
                task=args.task,
                precision=args.precision,
            )
            metadata["models"][relative_path] = result
            receipt.write_text(json.dumps(metadata, indent=2) + "\n")
            print(
                json.dumps({"layer": layer, "dimension": dimension, **result}),
                flush=True,
            )
            del model, prefix, reference, reference_prefix
    inventory = {
        key: metadata[key]
        for key in (
            "available_layers",
            "dimensions",
            "total_layers",
            "hidden_size",
            "task",
            "representation_contract",
        )
    }
    (output / "model_config.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
