#!/usr/bin/env python3
"""Export a local, merged ModernBERT task checkpoint and verify dynamic logits.

The portable FP32 graph is model.onnx; FP16 is model_sdpa_fp16.onnx.
CK Flash Attention conversion and GPU parity are separate, explicit steps.
An export receipt proves numerical agreement, not task quality at that length.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from model_precision import cast_parameters_preserving_buffers
from onnx_artifacts import external_data_sha256, sha256, strip_debug_annotations
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

MIN_VALIDATION_TOKENS = 2


class ClassifierLogits(torch.nn.Module):
    """Keep pooling and the complete task head in FP32, including FP16 exports."""

    def __init__(self, model, token_classification):
        super().__init__()
        self.model = model
        self.token_classification = token_classification
        self.model.head.float()
        self.model.classifier.float()
        if not token_classification and model.config.classifier_pooling not in {
            "mean",
            "cls",
        }:
            raise ValueError("Only mean or cls sequence pooling is supported")

    def forward(self, input_ids, attention_mask):
        hidden = self.model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if self.token_classification:
            return self.model.classifier(
                self.model.drop(self.model.head(hidden.float()))
            )
        if self.model.config.classifier_pooling == "cls":
            pooled = hidden[:, 0].float()
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1)
        return self.model.classifier(self.model.drop(self.model.head(pooled)))


def prepare_classifier(model, token_classification, dtype):
    """Cast encoder weights without rounding its native position buffers."""
    cast_parameters_preserving_buffers(model.model, dtype)
    return ClassifierLogits(model.eval(), token_classification).eval()


def make_input(seed_ids, length, batch, pad_token_id):
    repeated = (seed_ids * (length // len(seed_ids) + 1))[:length]
    ids = torch.tensor(repeated, dtype=torch.long).repeat(batch, 1)
    mask = torch.ones_like(ids)
    if batch > 1:
        pad = max(1, length // 3)
        mask[0, -pad:] = 0
        ids[0, -pad:] = pad_token_id
    return ids, mask


def verify_output_source(output, source_artifacts):
    """A directory's precision variants must describe one native checkpoint."""
    for path in output.glob("export-*.json"):
        prior = json.loads(path.read_text())
        if prior.get("source_artifacts") != source_artifacts:
            raise ValueError(
                "Export directory contains a different or unpinned source; "
                "use a fresh directory"
            )


def task_probabilities(logits, multi_label):
    values = torch.from_numpy(logits).float()
    return (values.sigmoid() if multi_label else values.softmax(-1)).numpy()


def verify_task_outputs(actual, expected, token_task, multi_label, dtype):
    """Check task semantics while retaining raw-logit errors in the receipt."""
    scores = task_probabilities(actual, multi_label)
    reference_scores = task_probabilities(expected, multi_label)
    if dtype == "float32" and not token_task:
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)
    # Token consumers use the complete softmax vector and BIO argmax. Dormant
    # class logits can amplify FP32 kernel rounding without changing either;
    # raw-logit error is diagnostic, not a token-decision tolerance.
    np.testing.assert_allclose(
        scores,
        reference_scores,
        atol=3e-3 if dtype == "float16" else 2e-4,
        rtol=1e-3 if dtype == "float16" else 1e-4,
    )
    if token_task:
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    return scores, reference_scores


def verify_graph(
    reference, graph_path, seed_ids, tokenizer, lengths, token_task, multi_label, dtype
):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(graph_path), options, providers=["CPUExecutionProvider"]
    )
    results = []
    for length in lengths:
        for batch in (1, 2):
            ids, mask = make_input(seed_ids, length, batch, tokenizer.pad_token_id)
            with torch.inference_mode():
                expected = reference(ids, mask).float().numpy()
            actual = session.run(
                None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
            )[0].astype(np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Graph returned invalid shape or nonfinite logits")
            # A padded token's query output has no task meaning. Its key is still
            # masked, so agreement on valid tokens verifies padding isolation.
            valid = mask.numpy().astype(bool) if token_task else slice(None)
            delta = np.abs(actual[valid] - expected[valid])
            # Quantization error is measured against full-precision native
            # inference, never against platform-dependent CPU half kernels.
            scores, reference_scores = verify_task_outputs(
                actual[valid], expected[valid], token_task, multi_label, dtype
            )
            results.append(
                {
                    "batch": batch,
                    "sequence_length": length,
                    "valid_tokens": mask.sum(1).tolist(),
                    "output_shape": list(actual.shape),
                    "max_abs_logit_error": float(delta.max()),
                    "max_abs_probability_error": float(
                        np.abs(scores - reference_scores).max()
                    ),
                    "argmax_agreement": float(
                        (actual[valid].argmax(-1) == expected[valid].argmax(-1)).mean()
                    ),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Immutable local merged checkpoint"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Recheck an existing export against the checkpoint",
    )
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--validation-lengths",
        type=int,
        nargs="+",
        default=[2, 63, 64, 65, 127, 128, 129, 512],
    )
    args = parser.parse_args()
    if not args.model.is_dir() or args.model.resolve() == args.output.resolve():
        raise ValueError("Use a local checkpoint and a separate export directory")
    torch.set_num_threads(8)
    source_artifacts = {
        path.name: sha256(path)
        for path in sorted(args.model.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }
    verify_output_source(args.output, source_artifacts)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "modernbert":
        raise ValueError("This exporter only supports ModernBERT task heads")
    architectures = config.architectures or []
    token_task = architectures == ["ModernBertForTokenClassification"]
    if not token_task and architectures != ["ModernBertForSequenceClassification"]:
        raise ValueError(f"A merged task checkpoint is required, got {architectures}")
    if set(config.id2label) != set(range(config.num_labels)):
        raise ValueError("id2label must exactly cover the task head")
    if len(set(config.id2label.values())) != config.num_labels:
        raise ValueError("Task labels must be unique")
    if any(
        length < MIN_VALIDATION_TOKENS or length > config.max_position_embeddings
        for length in args.validation_lengths
    ):
        raise ValueError("Validation lengths must fit the model's context")
    kind = (
        AutoModelForTokenClassification
        if token_task
        else AutoModelForSequenceClassification
    )
    model, loading = kind.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        reference_compile=False,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete task checkpoint: {loading}")
    wrapper = prepare_classifier(model, token_task, getattr(torch, args.dtype))
    reference = wrapper
    if args.dtype == "float16":
        reference = ClassifierLogits(
            kind.from_pretrained(
                args.model,
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
                reference_compile=False,
            ).eval(),
            token_task,
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed_ids = tokenizer.encode(
        "Example context. 中文测试。 Contact alice@example.org."
    )
    ids, mask = make_input(seed_ids, 64, 2, tokenizer.pad_token_id)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / (
        "model.onnx" if args.dtype == "float32" else "model_sdpa_fp16.onnx"
    )
    if args.verify_only and not path.is_file():
        raise FileNotFoundError("No exported graph to verify")
    if path.exists() and not args.verify_only:
        raise FileExistsError(
            "Use a fresh export directory to preserve earlier evidence"
        )
    batch = torch.export.Dim("batch", min=1, max=16)
    sequence = torch.export.Dim("sequence", min=2, max=config.max_position_embeddings)
    if not args.verify_only:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (ids, mask),
                str(path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=18,
                dynamo=True,
                external_data=True,
                dynamic_shapes=({0: batch, 1: sequence}, {0: batch, 1: sequence}),
                optimize=True,
            )
    graph = onnx.load(path, load_external_data=False)
    external_files = external_data_sha256(graph, path)
    strip_debug_annotations(graph)
    onnx.save(graph, path)
    onnx.checker.check_model(str(path))
    expected_rank = 3 if token_task else 2
    if len(graph.graph.output[0].type.tensor_type.shape.dim) != expected_rank:
        raise ValueError("Exported graph has the wrong task rank")
    results = verify_graph(
        reference,
        path,
        seed_ids,
        tokenizer,
        args.validation_lengths,
        token_task,
        config.problem_type == "multi_label_classification",
        args.dtype,
    )
    model.config._name_or_path = ""
    model.config.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    receipt = {
        "task": "token-classification" if token_task else "text-classification",
        "dtype": args.dtype,
        "head_dtype": "float32",
        "pooling_accumulation_dtype": "float32",
        "encoder_buffers": "native dtypes and values preserved",
        "reference_dtype": "float32",
        "output_validation": (
            "all valid-token probabilities and exact BIO argmax; raw logits reported"
            if token_task
            else "all class probabilities; raw logits also checked for FP32"
        ),
        "id2label": config.id2label,
        "max_position_embeddings": config.max_position_embeddings,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
        },
        "source_weights": {
            p.name: sha256(p) for p in sorted(args.model.glob("*.safetensors"))
        },
        "source_artifacts": source_artifacts,
        "exported_files": {
            p.name: sha256(p)
            for p in sorted(args.output.iterdir())
            if p.is_file() and not p.match("export-*.json")
        },
        "external_data": external_files,
        "cpu_dynamic_parity": results,
        "scope": "Numerical parity only; task quality and long-context GPU behavior require separate evaluations.",
    }
    (args.output / f"export-{args.dtype}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"task": receipt["task"], "dtype": args.dtype, "parity_cases": len(results)}
        )
    )


if __name__ == "__main__":
    main()
