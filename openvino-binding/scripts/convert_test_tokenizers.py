#!/usr/bin/env python3
"""
Convert HuggingFace tokenizers to OpenVINO native format for test models.
This script is called by 'make convert-openvino-test-models'.
"""
import os
import sys
from pathlib import Path

# Check for required dependencies
try:
    from transformers import AutoTokenizer
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: transformers not installed")
    print("=" * 70)
    print("Please install: pip install transformers")
    sys.exit(1)

try:
    from openvino_tokenizers import convert_tokenizer
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: openvino_tokenizers not installed")
    print("=" * 70)
    print("OpenVINO tokenizers is required for native tokenizer conversion.")
    print("\nInstall with:")
    print("  pip install openvino-tokenizers>=2025.3.0.0")
    print("\nAlternatively, skip tokenizer conversion (tests will still work):")
    print("  export SKIP_TOKENIZER_CONVERSION=1")
    print("  make convert-openvino-test-models")
    print("=" * 70)
    sys.exit(1)

try:
    import openvino as ov
except ImportError:
    print("\n" + "=" * 70)
    print("ERROR: openvino not installed")
    print("=" * 70)
    print("Please install: pip install openvino>=2024.0.0")
    sys.exit(1)


def convert_tokenizer_to_ov(model_name_or_path, output_dir):
    """Convert a HuggingFace tokenizer to OpenVINO format"""
    print(f"\n{'='*70}")
    print(f"Converting tokenizer: {model_name_or_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load HuggingFace tokenizer
        print("  → Loading HuggingFace tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"  ✓ Loaded: {type(tokenizer).__name__}")

        # Convert to OpenVINO
        print("  → Converting to OpenVINO format...")
        ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)

        # Print model info
        print(f"  ✓ Inputs:  {[inp.get_any_name() for inp in ov_tokenizer.inputs]}")
        print(f"  ✓ Outputs: {[out.get_any_name() for out in ov_tokenizer.outputs]}")

        # Save
        output_path = os.path.join(output_dir, "tokenizer.xml")
        ov.save_model(ov_tokenizer, output_path)

        # Verify files exist
        bin_path = output_path.replace(".xml", ".bin")
        if os.path.exists(output_path) and os.path.exists(bin_path):
            xml_size = os.path.getsize(output_path) / 1024  # KB
            bin_size = os.path.getsize(bin_path) / 1024  # KB
            print(f"  ✓ Saved: tokenizer.xml ({xml_size:.1f} KB)")
            print(f"  ✓ Saved: tokenizer.bin ({bin_size:.1f} KB)")
            return True
        else:
            print(f"  ✗ Error: Output files not created")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    script_dir = Path(__file__).parent.parent
    test_models_dir = script_dir / "test_models"

    print("\n" + "=" * 70)
    print("OpenVINO Test Tokenizer Conversion")
    print("=" * 70)
    print(f"Test models directory: {test_models_dir}")

    # Models to convert (these should already exist from optimum-cli)
    conversions = [
        # (HuggingFace model, output directory)
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            str(test_models_dir / "all-MiniLM-L6-v2"),
        ),
        (
            "LLM-Semantic-Router/category_classifier_modernbert-base_model",
            str(test_models_dir / "category_classifier_modernbert"),
        ),
    ]

    print(f"Tokenizers to convert: {len(conversions)}\n")

    results = []
    for model_name, output_dir in conversions:
        # Check if the model directory exists (should be created by optimum-cli)
        if not os.path.exists(output_dir):
            print(f"\n{'='*70}")
            print(f"Skipping: {model_name}")
            print(f"  ⚠️  Model directory not found: {output_dir}")
            print(f"  Run optimum-cli first to convert the model")
            print("=" * 70)
            results.append((model_name, False))
            continue

        # Check if tokenizer already exists
        tokenizer_path = os.path.join(output_dir, "tokenizer.xml")
        if os.path.exists(tokenizer_path):
            print(f"\n{'='*70}")
            print(f"Skipping: {model_name}")
            print(f"  ✓ Tokenizer already exists: {tokenizer_path}")
            print("=" * 70)
            results.append((model_name, True))
            continue

        success = convert_tokenizer_to_ov(model_name, output_dir)
        results.append((model_name, success))

    # Summary
    print("\n" + "=" * 70)
    print("TOKENIZER CONVERSION SUMMARY")
    print("=" * 70)

    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        short_name = model_name.split("/")[-1]
        print(f"{status}: {short_name}")

    total_success = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_success}/{len(results)} successful")

    if total_success == len(results):
        print("\n✓ All tokenizers ready!")
        print("\nYou can now run OpenVINO binding tests:")
        print("  cd openvino-binding && make test")
        return 0
    else:
        print("\n✗ Some conversions failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
