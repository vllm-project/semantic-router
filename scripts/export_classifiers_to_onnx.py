#!/usr/bin/env python3
"""
Export merged LoRA classifier models to ONNX format.

Models:
- mmbert32k-intent-classifier: 14-class sequence classification
- mmbert32k-jailbreak-detector: 2-class sequence classification
- mmbert32k-pii-detector: 35-label token classification
- mmbert32k-factcheck-classifier: binary fact-check routing
- mmbert32k-feedback-detector: 4-class satisfaction (user feedback)

Uses optimum for ONNX export with proper handling of ModernBERT architecture.
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from onnxconverter_common import float16
except ImportError:
    float16 = None

MAX_CLASSIFICATION_SEQ_LEN = 512
TOKEN_OPTIMIZED_ARTIFACT_FP32 = "model_token_sdpa.onnx"
TOKEN_OPTIMIZED_ARTIFACT_FP16 = "model_token_sdpa_fp16.onnx"
TOKEN_EAGER_ARTIFACT_FP32 = "model_token_eager.onnx"
TOKEN_EAGER_ARTIFACT_FP16 = "model_token_eager_fp16.onnx"
TOKEN_LOGITS_RANK = 3
TOKEN_ARTIFACT_OPSET = 18
LOGIT_DIFF_WARN_THRESHOLD = 1e-4


class TokenLogitsWrapper(torch.nn.Module):
    """Expose only token logits for torch.onnx.export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_tokenizer(model_path: str):
    """Load tokenizer with the same tokenizer.json fallback used by classifiers."""
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (ValueError, OSError, AttributeError) as e:
        if (
            "TokenizersBackend" in str(e)
            or "does not exist" in str(e)
            or "has no attribute" in str(e)
        ):
            tokenizer_file = Path(model_path) / "tokenizer.json"
            if tokenizer_file.exists():
                return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        raise


def load_token_model(model_path: str, *, torch_dtype, attn_implementation: str | None):
    """Load a token classifier and request a specific attention implementation."""
    kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
    except (TypeError, ValueError) as e:
        if not attn_implementation:
            raise
        print(
            f"  Warning: {attn_implementation} attention load failed ({e}); "
            "retrying with model defaults"
        )
        kwargs.pop("attn_implementation", None)
        model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
    if attn_implementation:
        model.config._attn_implementation = attn_implementation
        base_model = getattr(model, "model", None)
        if base_model is not None and hasattr(base_model, "config"):
            base_model.config._attn_implementation = attn_implementation
    disable_reference_compile(model)
    return model


def disable_reference_compile(model):
    """Disable ModernBERT torch.compile paths that legacy ONNX tracing cannot export."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "reference_compile"):
            config.reference_compile = False


def copy_model_assets(model_path: str, output_path: str, tokenizer, mapping_files):
    """Save tokenizer/config and copy classifier mapping sidecars."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    config_src = Path(model_path) / "config.json"
    if config_src.exists():
        shutil.copy(config_src, output_dir / "config.json")
    for mapping_file in mapping_files:
        src = Path(model_path) / mapping_file
        if src.exists():
            shutil.copy(src, output_dir / mapping_file)
            print(f"  Copied {mapping_file}")


def convert_onnx_to_fp16(input_path: Path, output_path: Path):
    """Convert an exported fp32 ONNX graph to fp16 weights/operators."""
    if onnx is None or float16 is None:
        raise RuntimeError(
            "onnx and onnxconverter-common are required to create "
            f"{TOKEN_OPTIMIZED_ARTIFACT_FP16} or {TOKEN_EAGER_ARTIFACT_FP16}. "
            "Install with: "
            "pip install onnx onnxconverter-common"
        )

    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(
        model, keep_io_types=False, disable_shape_infer=True, op_block_list=[]
    )
    del model_fp16.graph.value_info[:]
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, output_path)


def verify_token_artifact_shape(
    artifact_path: Path,
    tokenizer,
    model,
    *,
    batch_size: int,
    seq_len: int,
):
    """Run a shape/parity smoke check for a token-classification ONNX artifact."""
    if np is None or ort is None:
        raise RuntimeError(
            "numpy and onnxruntime are required to verify token ONNX artifacts"
        )

    test_text = "John Smith's email is john@example.com and SSN is 123-45-6789."
    texts = [test_text] * batch_size
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"].detach().cpu().numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].detach().cpu().numpy().astype(
            np.int64
        ),
    }

    session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, ort_inputs)
    logits = np.asarray(outputs[0])
    expected_shape = (batch_size, seq_len, int(model.config.num_labels))
    if logits.shape != expected_shape:
        raise RuntimeError(
            f"{artifact_path} produced logits shape {logits.shape}; "
            f"expected {expected_shape}"
        )
    if logits.ndim != TOKEN_LOGITS_RANK:
        raise RuntimeError(
            f"{artifact_path} produced rank {logits.ndim}; expected token logits rank 3"
        )

    with torch.no_grad():
        device = next(model.parameters()).device
        pt_inputs = {name: value.to(device) for name, value in inputs.items()}
        pt_outputs = model(**pt_inputs)
    pt_logits = pt_outputs.logits.detach().cpu().numpy()
    active_tokens = ort_inputs["attention_mask"].astype(bool)
    diff = abs(pt_logits.astype("float32") - logits.astype("float32"))[active_tokens].max()
    print(f"  Token artifact shape: {logits.shape}")
    print(f"  Max PyTorch/ONNX active-token logit difference: {diff:.6f}")


def export_token_logits_onnx(wrapper, inputs, output_path: Path, *, opset: int):
    """Export token logits with the legacy tracer when available."""
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
    }
    try:
        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError as e:
        if "dynamo" not in str(e):
            raise
        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(output_path),
            **export_kwargs,
        )


def export_sequence_classifier(model_path: str, output_path: str, opset: int = 14):
    """Export a sequence classification model to ONNX."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_path)

    # Load model
    print("Loading PyTorch model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Get model info
    config = model.config
    print(f"  Architecture: {config.architectures}")
    print(f"  Num labels: {config.num_labels}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Export to ONNX using optimum
    print("Exporting to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True,
    )

    # Save ONNX model
    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy label mappings if they exist
    for mapping_file in [
        "label_mapping.json",
        "category_mapping.json",
        "jailbreak_type_mapping.json",
        "fact_check_mapping.json",
    ]:
        src = Path(model_path) / mapping_file
        if src.exists():
            shutil.copy(src, Path(output_path) / mapping_file)
            print(f"  Copied {mapping_file}")

    # Verify the exported model
    print("Verifying ONNX model...")
    ort_model_loaded = ORTModelForSequenceClassification.from_pretrained(output_path)

    # Test inference
    test_text = "This is a test sentence for verification."
    inputs = tokenizer(
        test_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        pt_outputs = model(**inputs)

    ort_outputs = ort_model_loaded(**inputs)

    # Compare outputs
    pt_logits = pt_outputs.logits.numpy()
    ort_logits = ort_outputs.logits.numpy()

    diff = abs(pt_logits - ort_logits).max()
    print(f"  Max logit difference: {diff:.6f}")

    if diff < LOGIT_DIFF_WARN_THRESHOLD:
        print("  ONNX model verified successfully!")
    else:
        print(f"  ⚠ Warning: Logit difference {diff} is larger than expected")

    # Print ONNX file size
    onnx_path = Path(output_path) / "model.onnx"
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX model size: {size_mb:.1f} MB")

    return output_path


def export_token_sdpa_artifact(
    model_path: str,
    output_path: str,
    *,
    batch_size: int,
    seq_len: int,
    opset: int,
    precision: str,
    attn_implementation: str,
):
    """Export a token-classification ONNX candidate for MIGraphX validation."""
    artifact_names = {
        ("sdpa", "fp32"): TOKEN_OPTIMIZED_ARTIFACT_FP32,
        ("sdpa", "fp16"): TOKEN_OPTIMIZED_ARTIFACT_FP16,
        ("eager", "fp32"): TOKEN_EAGER_ARTIFACT_FP32,
        ("eager", "fp16"): TOKEN_EAGER_ARTIFACT_FP16,
    }
    artifact_name = artifact_names.get((attn_implementation, precision))
    if artifact_name is None:
        raise ValueError(f"unsupported token artifact precision: {precision}")

    print(f"\nExporting token-specific optimized artifact: {artifact_name}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / artifact_name
    tmp_path = output_dir / f"model_token_{attn_implementation}_fp32.tmp.onnx"

    tokenizer = load_tokenizer(model_path)
    copy_model_assets(
        model_path,
        output_path,
        tokenizer,
        ["label_mapping.json", "pii_type_mapping.json", "pii_mapping.json"],
    )
    use_gpu = torch.cuda.is_available()
    direct_fp16_export = precision == "fp16" and use_gpu
    convert_fp32_to_fp16 = precision == "fp16" and not use_gpu
    dtype = torch.float16 if direct_fp16_export else torch.float32
    device = torch.device("cuda" if use_gpu else "cpu")
    model = load_token_model(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device).eval()
    wrapper = TokenLogitsWrapper(model).eval()

    test_text = "John Smith's email is john@example.com and SSN is 123-45-6789."
    inputs = tokenizer(
        [test_text] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    export_inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.no_grad():
        pt_logits = wrapper(export_inputs["input_ids"], export_inputs["attention_mask"])
    expected_shape = (batch_size, seq_len, int(model.config.num_labels))
    if tuple(pt_logits.shape) != expected_shape:
        raise RuntimeError(
            f"PyTorch token classifier produced logits shape {tuple(pt_logits.shape)}; "
            f"expected {expected_shape}"
        )

    if direct_fp16_export:
        print("  Exporting token artifact directly from FP16 CUDA/ROCm weights")
        export_token_logits_onnx(
            wrapper,
            export_inputs,
            artifact_path,
            opset=opset,
        )
    elif convert_fp32_to_fp16:
        print("  Exporting fp32 token graph before fp16 conversion")
        export_token_logits_onnx(
            wrapper,
            export_inputs,
            tmp_path,
            opset=opset,
        )
        convert_onnx_to_fp16(tmp_path, artifact_path)
    else:
        print("  Exporting token artifact from FP32 weights")
        export_token_logits_onnx(
            wrapper,
            export_inputs,
            artifact_path,
            opset=opset,
        )
    verify_token_artifact_shape(
        artifact_path,
        tokenizer,
        model,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if convert_fp32_to_fp16:
        tmp_path.unlink(missing_ok=True)

    size_mb = artifact_path.stat().st_size / (1024 * 1024)
    print(f"  Token optimized ONNX model size: {size_mb:.1f} MB")
    return artifact_path


def export_token_classifier(
    model_path: str,
    output_path: str,
    opset: int = 14,
    *,
    export_optimized: bool = True,
    optimized_batch_size: int = 1,
    optimized_seq_len: int = MAX_CLASSIFICATION_SEQ_LEN,
    optimized_opset: int = TOKEN_ARTIFACT_OPSET,
    optimized_precision: str = "fp32",
    optimized_attn: str = "sdpa",
    only_optimized: bool = False,
):
    """Export a token classification model to ONNX."""
    print(f"\n{'='*60}")
    print(f"Exporting: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(model_path)

    if not only_optimized:
        # Load model
        print("Loading PyTorch model...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        model.eval()

        # Get model info
        config = model.config
        print(f"  Architecture: {config.architectures}")
        print(f"  Num labels: {config.num_labels}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Export to ONNX using optimum
        print("Exporting to ONNX...")
        ort_model = ORTModelForTokenClassification.from_pretrained(
            model_path,
            export=True,
        )

        # Save ONNX model
        ort_model.save_pretrained(output_path)
        copy_model_assets(
            model_path,
            output_path,
            tokenizer,
            ["label_mapping.json", "pii_type_mapping.json", "pii_mapping.json"],
        )

        # Verify the exported model
        print("Verifying ONNX model...")
        ort_model_loaded = ORTModelForTokenClassification.from_pretrained(output_path)

        # Test inference
        test_text = "John Smith's email is john@example.com and SSN is 123-45-6789."
        inputs = tokenizer(
            test_text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            pt_outputs = model(**inputs)

        ort_outputs = ort_model_loaded(**inputs)

        # Compare outputs
        pt_logits = pt_outputs.logits.numpy()
        ort_logits = ort_outputs.logits.numpy()

        diff = abs(pt_logits - ort_logits).max()
        print(f"  Max logit difference: {diff:.6f}")

        if diff < LOGIT_DIFF_WARN_THRESHOLD:
            print("  ONNX model verified successfully!")
        else:
            print(f"  ⚠ Warning: Logit difference {diff} is larger than expected")

        # Print ONNX file size
        onnx_path = Path(output_path) / "model.onnx"
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"  ONNX model size: {size_mb:.1f} MB")

    if export_optimized:
        export_token_sdpa_artifact(
            model_path,
            output_path,
            batch_size=optimized_batch_size,
            seq_len=optimized_seq_len,
            opset=optimized_opset,
            precision=optimized_precision,
            attn_implementation=optimized_attn,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export classifier models to ONNX")
    parser.add_argument(
        "--model",
        choices=["intent", "jailbreak", "pii", "factcheck", "feedback", "all"],
        default="all",
        help="Which model to export",
    )
    parser.add_argument("--output-dir", default=".", help="Base output directory")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override input model path for the selected --model.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override output path for the selected --model.",
    )
    parser.add_argument(
        "--skip-token-optimized",
        action="store_true",
        help="Do not export token-specific SDPA ONNX artifacts for token classifiers.",
    )
    parser.add_argument(
        "--only-token-optimized",
        action="store_true",
        help="Only export the token-specific SDPA artifact; skip baseline model.onnx.",
    )
    parser.add_argument(
        "--token-optimized-batch-size",
        type=int,
        default=1,
        help="Static batch size for token-specific SDPA artifacts.",
    )
    parser.add_argument(
        "--token-optimized-seq-len",
        type=int,
        default=MAX_CLASSIFICATION_SEQ_LEN,
        help="Static sequence length for token-specific SDPA artifacts.",
    )
    parser.add_argument(
        "--token-optimized-opset",
        type=int,
        default=TOKEN_ARTIFACT_OPSET,
        help="ONNX opset for token-specific SDPA artifacts.",
    )
    parser.add_argument(
        "--token-optimized-precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help=(
            "Precision for token-specific SDPA artifacts. FP32 is the default "
            "because PII BIO boundaries are sensitive to FP16 logit drift."
        ),
    )
    parser.add_argument(
        "--token-optimized-attn",
        choices=["sdpa", "eager"],
        default="sdpa",
        help=(
            "Attention implementation for token-specific ONNX artifacts. Use "
            "eager to generate a non-SDPA candidate when debugging MIGraphX "
            "parity."
        ),
    )
    args = parser.parse_args()
    if args.input_path is not None and args.model == "all":
        parser.error("--input-path requires selecting one --model, not --model all")

    base_dir = Path(args.output_dir)

    models = {
        "intent": {
            "input": "mmbert32k-intent-classifier-merged-r32",
            "output": "mmbert32k-intent-classifier-onnx",
            "type": "sequence",
        },
        "jailbreak": {
            "input": "mmbert32k-jailbreak-detector-merged-r32",
            "output": "mmbert32k-jailbreak-detector-onnx",
            "type": "sequence",
        },
        "pii": {
            "input": "mmbert32k-pii-detector-merged-r32",
            "output": "mmbert32k-pii-detector-onnx",
            "type": "token",
        },
        "factcheck": {
            "input": "mmbert32k-factcheck-classifier-merged",
            "output": "mmbert32k-factcheck-classifier-merged-onnx",
            "type": "sequence",
        },
        "feedback": {
            "input": "mmbert32k-feedback-detector-merged",
            "output": "mmbert32k-feedback-detector-merged-onnx",
            "type": "sequence",
        },
    }

    to_export = (
        [args.model]
        if args.model != "all"
        else ["intent", "jailbreak", "pii", "factcheck", "feedback"]
    )

    for model_name in to_export:
        model_info = models[model_name]
        input_path = args.input_path or base_dir / model_info["input"]
        output_path = args.output_path or base_dir / model_info["output"]

        if not input_path.exists():
            print(f"⚠ Skipping {model_name}: {input_path} not found")
            continue

        if model_info["type"] == "sequence":
            export_sequence_classifier(str(input_path), str(output_path))
        else:
            export_token_classifier(
                str(input_path),
                str(output_path),
                export_optimized=not args.skip_token_optimized,
                optimized_batch_size=args.token_optimized_batch_size,
            optimized_seq_len=args.token_optimized_seq_len,
            optimized_opset=args.token_optimized_opset,
            optimized_precision=args.token_optimized_precision,
            optimized_attn=args.token_optimized_attn,
            only_optimized=args.only_token_optimized,
        )

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
