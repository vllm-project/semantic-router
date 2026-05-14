#!/usr/bin/env python3
"""
Convert HuggingFace tokenizers to OpenVINO native format.
This is a one-time conversion - the resulting .xml/.bin files are used by C++.
"""
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import openvino as ov


def convert_tokenizer_to_ov(model_name_or_path, output_dir):
    """Convert a HuggingFace tokenizer to OpenVINO format"""
    print(f"\n{'='*70}")
    print(f"Converting: {model_name_or_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load HuggingFace tokenizer
        print("  Loading HuggingFace tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"  ✓ Loaded: {type(tokenizer).__name__}")

        # Convert to OpenVINO
        print("  Converting to OpenVINO format...")
        ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)

        # Print model info
        print(f"  Inputs:  {[inp.get_any_name() for inp in ov_tokenizer.inputs]}")
        print(f"  Outputs: {[out.get_any_name() for out in ov_tokenizer.outputs]}")

        # Save
        output_path = os.path.join(output_dir, "tokenizer.xml")
        ov.save_model(ov_tokenizer, output_path)

        # Verify files exist
        bin_path = output_path.replace(".xml", ".bin")
        if os.path.exists(output_path) and os.path.exists(bin_path):
            xml_size = os.path.getsize(output_path) / 1024  # KB
            bin_size = os.path.getsize(bin_path) / 1024  # KB
            print(f"  ✓ Saved: {output_path} ({xml_size:.1f} KB)")
            print(f"  ✓ Saved: {bin_path} ({bin_size:.1f} KB)")
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
    models_dir = script_dir / "models"

    # Models to convert
    conversions = [
        # (HuggingFace model, output directory)
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            str(models_dir / "minilm_tokenizer"),
        ),
        # Add more models as needed
    ]

    print("OpenVINO Tokenizer Conversion")
    print("=" * 70)
    print(f"Models directory: {models_dir}")
    print(f"Conversions to perform: {len(conversions)}")

    results = []
    for model_name, output_dir in conversions:
        success = convert_tokenizer_to_ov(model_name, output_dir)
        results.append((model_name, success))

    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {model_name}")

    total_success = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_success}/{len(results)} successful")

    if total_success == len(results):
        print("\n✓ All tokenizers converted successfully!")
        print("\nConverted tokenizers can now be used by C++ code:")
        print('  - Load with ov::Core::read_model("path/to/tokenizer.xml")')
        print("  - Run inference with string input")
        print("  - Get token IDs, attention masks, etc.")
        return 0
    else:
        print("\n✗ Some conversions failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
