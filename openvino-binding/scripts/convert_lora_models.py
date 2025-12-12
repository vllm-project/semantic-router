#!/usr/bin/env python3
"""
Convert LoRA HuggingFace models to OpenVINO IR format

This script converts BERT and ModernBERT LoRA models from HuggingFace format
to OpenVINO Intermediate Representation (IR) format for inference.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import openvino as ov
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
import numpy as np


class LoRAModelConverter:
    """Converts LoRA models from HuggingFace to OpenVINO format"""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the HuggingFace model and tokenizer"""
        print(f"Loading model from {self.model_path}...")

        try:
            self.config = AutoConfig.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Detect model type from config
            self.model_type = "base"

            # Check if it's a token classification model (for NER, PII, etc.)
            if hasattr(self.config, "architectures") and self.config.architectures:
                arch = self.config.architectures[0]
                if "ForTokenClassification" in arch:
                    self.model_type = "token_classification"
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        self.model_path, torchscript=True
                    )
                    print(
                        f"✓ Loaded as TokenClassification model ({self.config.num_labels} labels)"
                    )
                elif "ForSequenceClassification" in arch:
                    self.model_type = "sequence_classification"
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path, torchscript=True
                    )
                    print(
                        f"✓ Loaded as SequenceClassification model ({self.config.num_labels} classes)"
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        self.model_path, torchscript=True
                    )
                    print("✓ Loaded as base model (no classifier head)")
            else:
                # Try sequence classification first, then fall back
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path, torchscript=True
                    )
                    self.model_type = "sequence_classification"
                    print("✓ Loaded as SequenceClassification model")
                except:
                    self.model = AutoModel.from_pretrained(
                        self.model_path, torchscript=True
                    )
                    print("✓ Loaded as base model")

            self.model.eval()
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def create_dummy_input(self):
        """Create dummy input for tracing"""
        # Create dummy inputs matching model's expected input
        seq_length = 128
        batch_size = 1

        input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Add token type ids for BERT models
        if hasattr(self.config, "type_vocab_size") and self.config.type_vocab_size > 0:
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_to_onnx(self):
        """Convert PyTorch model to ONNX format"""
        onnx_path = self.output_dir / "model.onnx"
        print(f"Converting to ONNX: {onnx_path}")

        try:
            dummy_input = self.create_dummy_input()

            # Determine input names based on model type
            input_names = ["input_ids", "attention_mask"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
            }

            if "token_type_ids" in dummy_input:
                input_names.append("token_type_ids")
                dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence"}

            # Determine output names and dynamic axes based on model type
            if self.model_type == "token_classification":
                # Token classification: logits shape is [batch, seq_len, num_labels]
                output_names = ["logits"]
                dynamic_axes["logits"] = {0: "batch_size", 1: "sequence"}
                print(
                    f"  Token classification model: logits shape [batch, seq_len, {self.config.num_labels}]"
                )
            elif self.model_type == "sequence_classification" or hasattr(
                self.model, "classifier"
            ):
                # Sequence classification: logits shape is [batch, num_classes]
                output_names = ["logits"]
                dynamic_axes["logits"] = {0: "batch_size"}
                print(
                    f"  Sequence classification model: logits shape [batch, {self.config.num_labels}]"
                )
            elif hasattr(self.model, "pooler"):
                # Base model with pooler
                output_names = ["last_hidden_state", "pooler_output"]
                print("  Base model with pooler, exporting hidden states")
            else:
                # Base model without pooler (e.g., ModernBERT)
                output_names = ["last_hidden_state"]
                print("  Base model, exporting hidden states only")

            # Export to ONNX
            torch.onnx.export(
                self.model,
                tuple(dummy_input.values()),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
            )

            print("✓ ONNX conversion successful")
            return str(onnx_path)
        except Exception as e:
            print(f"✗ ONNX conversion failed: {e}")
            return None

    def convert_to_openvino(self, onnx_path: str):
        """Convert ONNX model to OpenVINO IR format"""
        print(f"Converting ONNX to OpenVINO IR...")

        try:
            # Load ONNX model
            ov_model = ov.convert_model(onnx_path)

            # Save OpenVINO IR
            xml_path = self.output_dir / "openvino_model.xml"
            ov.save_model(ov_model, xml_path)

            print(f"✓ OpenVINO IR saved: {xml_path}")
            print(f"  - Model: openvino_model.xml")
            print(f"  - Weights: openvino_model.bin")
            return True
        except Exception as e:
            print(f"✗ OpenVINO conversion failed: {e}")
            return False

    def save_tokenizer(self):
        """Save tokenizer in OpenVINO-compatible format"""
        try:
            # Save tokenizer files
            tokenizer_path = self.output_dir / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)

            self.tokenizer.save_pretrained(tokenizer_path)
            print(f"✓ Tokenizer saved to {tokenizer_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save tokenizer: {e}")
            return False

    def convert(self):
        """Complete conversion pipeline"""
        print(f"\n{'='*60}")
        print(f"Converting LoRA model: {self.model_path.name}")
        print(f"{'='*60}\n")

        # Load model
        if not self.load_model():
            return False

        # Convert to ONNX
        onnx_path = self.convert_to_onnx()
        if not onnx_path:
            return False

        # Convert to OpenVINO
        if not self.convert_to_openvino(onnx_path):
            return False

        # Save tokenizer
        if not self.save_tokenizer():
            print("Warning: Tokenizer save failed, but model conversion succeeded")

        # Clean up ONNX file (optional)
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            print(f"✓ Cleaned up intermediate ONNX file")

        print(f"\n✓✓✓ Conversion complete! ✓✓✓")
        print(f"Output directory: {self.output_dir}\n")
        return True


def convert_lora_adapter(adapter_path: str, output_dir: str):
    """Convert a LoRA adapter (just the adapter weights)"""
    print(f"\nConverting LoRA adapter: {adapter_path}")

    try:
        # Load adapter weights
        adapter_state = torch.load(
            os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu"
        )

        # Create a simple model wrapper for the adapter
        class LoRAAdapterModel(torch.nn.Module):
            def __init__(self, adapter_state, hidden_size=768, rank=16):
                super().__init__()
                # LoRA A matrix (rank x hidden_size)
                self.lora_A = torch.nn.Linear(hidden_size, rank, bias=False)
                # LoRA B matrix (hidden_size x rank)
                self.lora_B = torch.nn.Linear(rank, hidden_size, bias=False)

                # Load weights from state dict
                if "lora_A.weight" in adapter_state:
                    self.lora_A.weight.data = adapter_state["lora_A.weight"]
                if "lora_B.weight" in adapter_state:
                    self.lora_B.weight.data = adapter_state["lora_B.weight"]

            def forward(self, x):
                # LoRA forward: B(A(x))
                return self.lora_B(self.lora_A(x))

        # Determine hidden size and rank from weights
        hidden_size = 768  # Default for BERT-base
        rank = 16  # Default rank

        for key, value in adapter_state.items():
            if "lora_A" in key and "weight" in key:
                rank, hidden_size = value.shape
                break

        adapter_model = LoRAAdapterModel(adapter_state, hidden_size, rank)
        adapter_model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, hidden_size)

        # Export to ONNX
        onnx_path = os.path.join(output_dir, "adapter_temp.onnx")
        torch.onnx.export(
            adapter_model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14,
        )

        # Convert to OpenVINO
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, os.path.join(output_dir, "openvino_model.xml"))

        # Clean up
        os.remove(onnx_path)

        print(f"✓ LoRA adapter converted successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to convert LoRA adapter: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA models to OpenVINO format"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input HuggingFace model directory"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for OpenVINO IR"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["base", "adapter"],
        default="base",
        help="Model type: base model or LoRA adapter",
    )
    parser.add_argument("--batch", action="store_true", help="Convert multiple models")

    args = parser.parse_args()

    if args.batch:
        # Batch conversion mode
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1

        # Find all model directories
        model_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

        if not model_dirs:
            print(f"No models found in {input_dir}")
            return 1

        print(f"Found {len(model_dirs)} models to convert")

        success_count = 0
        for model_dir in model_dirs:
            output_dir = Path(args.output) / model_dir.name
            converter = LoRAModelConverter(str(model_dir), str(output_dir))
            if converter.convert():
                success_count += 1

        print(f"\n{'='*60}")
        print(
            f"Batch conversion complete: {success_count}/{len(model_dirs)} successful"
        )
        print(f"{'='*60}")

    else:
        # Single model conversion
        if args.type == "adapter":
            success = convert_lora_adapter(args.input, args.output)
        else:
            converter = LoRAModelConverter(args.input, args.output)
            success = converter.convert()

        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
