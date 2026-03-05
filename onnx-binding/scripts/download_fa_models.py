#!/usr/bin/env python3
"""Download FA FP16 ONNX models from HuggingFace for integration testing."""
import os
import sys

from huggingface_hub import snapshot_download

MODELS_DIR = os.environ.get("FA_MODELS_DIR", "/models")

REPOS = {
    # Classification models (merged only)
    "mmbert32k-intent-classifier-merged": "llm-semantic-router/mmbert32k-intent-classifier-merged",
    "mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
    "mmbert32k-pii-detector-merged": "llm-semantic-router/mmbert32k-pii-detector-merged",
    "mmbert32k-factcheck-classifier-merged": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
    "mmbert32k-feedback-detector-merged": "llm-semantic-router/mmbert32k-feedback-detector-merged",
    # Embedding models
    "mmbert-embed-32k-2d-matryoshka": "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
}

ALLOW_PATTERNS = [
    "onnx/*",
    "onnx/**/*",
    "*.json",
]

IGNORE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.h5",
    "model.onnx",
    "model.onnx.data",
    "model_sdpa_fp16.onnx",
    "1_Pooling/*",
]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    subset = sys.argv[1:] if len(sys.argv) > 1 else list(REPOS.keys())

    for name in subset:
        if name not in REPOS:
            print(f"Unknown model: {name}")
            continue
        repo_id = REPOS[name]
        local_dir = os.path.join(MODELS_DIR, name)

        print(f"\n{'='*60}")
        print(f"  Downloading {repo_id} -> {local_dir}")
        print(f"{'='*60}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
            )
            print(f"  OK: {local_dir}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
