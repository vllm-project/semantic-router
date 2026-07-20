"""Cross-encoder reranker parity check against HuggingFace transformers.

This is the ground-truth oracle for the candle `BertCrossEncoder`: it scores the
same (query, document) pairs used by the Go test
(`candle-binding/cross_encoder_live_test.go`) with HuggingFace `transformers` and
prints / emits the resulting relevance scores.

It is a developer tool, NOT part of CI: it needs torch + transformers, which the
Go/Rust binding tests deliberately do not depend on. Instead, run this once to
(re)generate the committed golden file, and the env-gated Go test
`TestCrossEncoderRerankMatchesTransformersGolden` asserts the candle scores match
that golden within tolerance, with no torch at test time.

Usage:
    pip install -r requirements.txt
    # print a human-readable parity table
    python verify_parity.py
    # (re)generate the committed golden file used by the Go parity test
    python verify_parity.py --emit-golden ../../testdata/reranker_parity_golden.json

The scoring matches the Rust implementation: for a single-logit head the score is
sigmoid(logit); for a multi-label head it is softmax over labels, positive (last)
class.
"""

import argparse
import json
import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512

# Mirrors `rerankCases` in candle-binding/cross_encoder_live_test.go. Keep in sync:
# documents[answer_index] is the doc that truly answers the query.
CASES = [
    {
        "query": "How do I cancel my subscription?",
        "documents": [
            "To cancel your subscription, open Settings > Billing and click Cancel plan.",
            "Our subscription plans come in monthly and annual billing options.",
            "Orders cannot be cancelled once they have shipped.",
            "Subscriptions renew automatically unless turned off.",
        ],
        "answer_index": 0,
    },
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe known for its cuisine.",
            "The French Riviera is a popular travel destination.",
            "Berlin is the capital of Germany.",
        ],
        "answer_index": 0,
    },
    {
        "query": "How do I get a refund for a cancelled flight?",
        "documents": [
            "Submit the refund request form within 30 days and the fare returns to your original payment method.",
            "Flights may be cancelled by the airline because of weather or operational issues.",
            "Travelers should arrive at the airport at least two hours before a flight.",
            "Seat upgrades can be purchased at check-in subject to availability.",
        ],
        "answer_index": 0,
    },
    {
        "query": "What is the return policy for opened electronics?",
        "documents": [
            "Opened electronics may be returned within 15 days as long as all original accessories are included.",
            "All electronics are covered by a one-year manufacturer warranty against defects.",
            "Most unopened items can be returned within 30 days for a full refund.",
            "Please recycle old electronics at a certified e-waste collection center.",
        ],
        "answer_index": 0,
    },
    {
        "query": "How many bags can I check for free on an international flight?",
        "documents": [
            "Economy passengers traveling abroad may check two pieces at no extra cost.",
            "Baggage rules differ between domestic and international flights.",
            "International flights generally begin boarding 45 minutes before departure.",
            "Checked bags must not exceed 23 kg or they incur an overweight fee.",
        ],
        "answer_index": 0,
    },
]


def score_from_logits(logits):
    """Map a logit row to a [0, 1] relevance score, matching cross_encoder.rs."""
    if len(logits) <= 1:
        return 1.0 / (1.0 + math.exp(-logits[0]))
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return exps[-1] / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-golden",
        metavar="PATH",
        help="write the transformers scores to a JSON golden file instead of a table",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    golden = {"model": MODEL, "max_length": MAX_LENGTH, "cases": []}
    correct = 0
    for case in CASES:
        query, documents, answer_index = (
            case["query"],
            case["documents"],
            case["answer_index"],
        )
        scores = []
        with torch.no_grad():
            for doc in documents:
                enc = tokenizer(
                    query,
                    doc,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                logits = model(**enc).logits[0].tolist()
                scores.append(score_from_logits(logits))

        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        top1_correct = order[0] == answer_index
        correct += top1_correct
        golden["cases"].append({"query": query, "scores": scores})

        if not args.emit_golden:
            print(f"QUERY: {query}")
            for rank, idx in enumerate(order, start=1):
                marker = "  <-- answer" if idx == answer_index else ""
                print(f"  #{rank}  score={scores[idx]:.4f}  [doc {idx}]{marker}")
            print(f"  => top-1 {'CORRECT' if top1_correct else 'WRONG'}\n")

    if args.emit_golden:
        with open(args.emit_golden, "w") as f:
            json.dump(golden, f, indent=2)
            f.write("\n")
        print(f"wrote golden file: {args.emit_golden} ({len(golden['cases'])} cases)")
    else:
        print(f"transformers top-1 accuracy: {correct}/{len(CASES)}")


if __name__ == "__main__":
    main()
