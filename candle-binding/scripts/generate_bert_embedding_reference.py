#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import sentence_transformers
from sentence_transformers import SentenceTransformer, models

OUTPUT = Path(__file__).resolve().parents[1] / "test_data/bert_embedding_reference.json"

# Short memory-style texts. The checkpoint's tokenizer.json pads every input to
# 128 tokens, so these are the inputs where padding can reach the mean.
TEXTS = [
    "Which city do I live in now?",
    "I just moved to Denver, and I live there now.",
    "I live in Boston, near the Charles River.",
    "I work as a nurse at the children's hospital.",
    "My budget for the Japan trip is $4,000.",
    "Don't, won't, shouldn't, couldn't've.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Write sentence-transformers cosines to {OUTPUT.name}."
    )
    parser.add_argument("--model", default="models/mom-embedding-light")
    args = parser.parse_args()

    # Mirrors the checkpoint's modules.json: mean pooling, then L2 normalization.
    word = models.Transformer(args.model, max_seq_length=128)
    pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(
        modules=[word, pooling, models.Normalize()], device="cpu"
    )
    vectors = model.encode(TEXTS, batch_size=1, convert_to_tensor=True)
    cosine = (vectors @ vectors.T).tolist()

    reference = {
        "model": "sentence-transformers/all-MiniLM-L12-v2",
        "generator": f"sentence-transformers {sentence_transformers.__version__}",
        "texts": TEXTS,
        "cosine": [[round(value, 5) for value in row] for row in cosine],
    }
    OUTPUT.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
