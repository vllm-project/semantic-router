"""Smoke test of a served model: chat, reasoning + JSON schema + thinking budget, code logprobs.

python -m d25.vega.data.synth.smoke --url http://d25-vega-synth-llm:8000 --model NAME --tokenizer DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from d25.vega.common import decision_format as df
from d25.vega.data.synth.llm import LLM, parse_json

STATE = {
    "order": "R-1182",
    "charges": [
        {"amount": 42.0, "date": "2026-09-30"},
        {"amount": 42.0, "date": "2026-09-30"},
    ],
    "message": "I was charged twice for the same order. Please refund one of the charges.",
}
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "Charges, refunds and invoices",
        "shipping": "Deliveries and tracking",
        "returns": "Exchanges of received items",
    },
}


async def main_async(a) -> None:
    llm = LLM(a.url, a.model, concurrency=8)
    t = time.time()
    plain = await llm.chat(
        [
            {
                "role": "user",
                "content": "Reply with one word: what colour is a clear daytime sky?",
            }
        ],
        max_tokens=16,
        temperature=0.0,
    )
    print(
        "plain:", plain.get("content"), plain.get("finish"), f"{time.time() - t:.1f}s"
    )
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(QUESTION["criteria"])},
            "why": {"type": "string"},
        },
        "required": ["label", "why"],
        "additionalProperties": False,
    }
    t = time.time()
    reasoned = await llm.chat(
        [
            {
                "role": "user",
                "content": json.dumps({"state": STATE, "question": QUESTION}),
            }
        ],
        schema=schema,
        thinking=True,
        budget=a.budget,
        max_tokens=a.budget + 400,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=1,
    )
    print(
        "reasoned:",
        reasoned.get("content"),
        reasoned.get("finish"),
        "reasoning chars",
        len(reasoned.get("reasoning", "")),
        reasoned.get("usage"),
        f"{time.time() - t:.1f}s",
    )
    print("parsed:", parse_json(reasoned["content"]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    codes, ids = df.answer_codes(tokenizer)
    print("codes:", codes[:30], "...", codes[-3:], "n", len(codes))
    prompt = df.render(tokenizer, STATE, QUESTION, codes)
    probs = await llm.code_probs(prompt, ids[:3])
    print("code probs:", probs)
    noul = {
        "type": "noul",
        "instructions": "Was the customer charged more than once for order R-1182?",
    }
    probs = await llm.code_probs(df.render(tokenizer, STATE, noul, codes), ids[:2])
    print("noul probs [false, true]:", probs)
    await llm.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--budget", type=int, default=512)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
