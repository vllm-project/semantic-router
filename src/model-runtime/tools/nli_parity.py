"""Compare a running NLI decision service against local Transformers weights.

Weights must be supplied explicitly; this tool never downloads packages or
runs package code. Start the model with vllm-sr serve, then run:

    python tools/nli_parity.py --package DIR --url http://127.0.0.1:8100 \
        --repo MoritzLaurer/ModernBERT-base-zeroshot-v2.0 --revision SHA --output record.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import httpx
import torch
import transformers

CASES = [
    ("code", "Write a Python function to sort a list.", "code"),
    ("math", "Solve the equation 2x + 5 = 13.", "math"),
    ("travel", "Plan a three-day holiday in Paris.", "travel"),
    ("negation", "I do not want any code. Plan a vacation in Tokyo.", None),
    (
        "mixed",
        "Write Python code to calculate the roots of a quadratic equation.",
        None,
    ),
    ("unrelated", "Explain how photosynthesis works.", None),
]
CRITERIA = {"code": "programming", "math": "mathematics", "travel": "travel planning"}
HYPOTHESES = [f"This request is about {label}." for label in CRITERIA.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.package, local_files_only=True, trust_remote_code=False
    )
    config = transformers.AutoConfig.from_pretrained(
        args.package, local_files_only=True, trust_remote_code=False
    )
    config.reference_compile = False
    reference = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.package,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    positive = next(
        index
        for index, label in config.id2label.items()
        if label.casefold() == "entailment"
    )
    standard = transformers.pipeline(
        "zero-shot-classification", model=reference, tokenizer=tokenizer, device=-1
    )
    records = []
    pipeline_maximum = 0.0
    maximum = 0.0
    with httpx.Client(base_url=args.url, timeout=120) as client:
        client.get("/health").raise_for_status()
        cards = client.get("/v1/models").json()["data"]
        if len(cards) != 1:
            raise ValueError("serve exactly one model for this parity run")
        card = cards[0]
        for case_id, state, expected in CASES:
            encoded = tokenizer(
                [state] * len(HYPOTHESES),
                HYPOTHESES,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            with torch.inference_mode():
                logits = reference(**encoded).logits
            reference_choice = logits[:, positive].softmax(0).tolist()
            reference_noul = logits[0].softmax(0)[positive].item()
            body = {
                "state": state,
                "questions": {
                    "domain": {
                        "type": "choice",
                        "instructions": "This request is about {label}.",
                        "criteria": CRITERIA,
                    },
                    "code": {"type": "noul", "instructions": HYPOTHESES[0]},
                },
                "options": {"return_meta": True},
            }
            started = time.perf_counter()
            response = client.post("/v1/decisions", json=body)
            response.raise_for_status()
            native = response.json()
            elapsed = 1000 * (time.perf_counter() - started)
            choice, noul = native["answers"]["domain"], native["answers"]["code"]
            actual = [choice["probabilities"][key] for key in CRITERIA]
            errors = [abs(a - b) for a, b in zip(actual, reference_choice, strict=True)]
            errors.append(abs(noul["noul"] - reference_noul))
            pipeline_choice = standard(
                state,
                candidate_labels=list(CRITERIA.values()),
                hypothesis_template="This request is about {}.",
                multi_label=False,
            )
            pipeline_scores = dict(
                zip(pipeline_choice["labels"], pipeline_choice["scores"], strict=True)
            )
            pipeline_noul = standard(
                state,
                candidate_labels=[CRITERIA["code"]],
                hypothesis_template="This request is about {}.",
                multi_label=True,
            )["scores"][0]
            pipeline_errors = [
                abs(choice["probabilities"][key] - pipeline_scores[label])
                for key, label in CRITERIA.items()
            ]
            pipeline_errors.append(abs(noul["noul"] - pipeline_noul))
            pipeline_maximum = max(pipeline_maximum, *pipeline_errors)
            classified = client.post(
                "/v1/classify",
                json={"input": [{"text": state, "text_pair": h} for h in HYPOTHESES]},
            )
            classified.raise_for_status()
            probabilities = torch.tensor(
                [row["probabilities"] for row in classified.json()["results"]]
            )
            errors.append((probabilities - logits.softmax(-1)).abs().max().item())
            reference_winner = list(CRITERIA)[
                reference_choice.index(max(reference_choice))
            ]
            reordered = json.loads(json.dumps(body))
            reordered["questions"]["domain"]["criteria"] = dict(
                reversed(list(CRITERIA.items()))
            )
            reordered_response = client.post("/v1/decisions", json=reordered)
            reordered_response.raise_for_status()
            reordered_answer = reordered_response.json()["answers"]["domain"]
            alias = client.post("/v1/systemone", json=body)
            alias.raise_for_status()
            passed = (
                max(errors) <= args.tolerance
                and max(pipeline_errors) <= args.tolerance
                and CRITERIA[choice["choice"]] == pipeline_choice["labels"][0]
                and choice["choice"] == reference_winner
                and reordered_answer["probabilities"] == choice["probabilities"]
                and reordered_answer["choice"] == choice["choice"]
                and alias.json()["answers"] == native["answers"]
            )
            maximum = max(maximum, *errors)
            records.append(
                {
                    "id": case_id,
                    "state": state,
                    "expected_smoke_choice": expected,
                    "native_answers": native["answers"],
                    "reference_choice": dict(
                        zip(CRITERIA, reference_choice, strict=True)
                    ),
                    "reference_noul": reference_noul,
                    "pipeline_choice": {
                        key: pipeline_scores[label] for key, label in CRITERIA.items()
                    },
                    "pipeline_noul": pipeline_noul,
                    "maximum_pipeline_difference": max(pipeline_errors),
                    "maximum_absolute_difference": max(errors),
                    "request_ms": elapsed,
                    "parity_passed": passed,
                }
            )
    report = {
        "model": args.repo,
        "revision": args.revision,
        "model_sha256": card["model_sha256"],
        "device": card["device"],
        "profile": "exact",
        "golden_status": card["golden"]["status"],
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "reference_attention": "eager",
        "dtype": "float32",
        "tolerance": args.tolerance,
        "maximum_absolute_difference": maximum,
        "maximum_pipeline_difference": pipeline_maximum,
        "cases": records,
        "passed": all(row["parity_passed"] for row in records),
        "note": "Parity and a small smoke panel, not a routing quality benchmark or calibrated confidence evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(records),
                "maximum_absolute_difference": maximum,
                "maximum_pipeline_difference": pipeline_maximum,
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
