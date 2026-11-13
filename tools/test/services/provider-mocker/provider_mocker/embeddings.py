"""A deterministic OpenAI embeddings API for the Router's embedding deployments.

It runs no model. A text carrying ``__EMBED_TOPIC__ <key>`` embeds as that
key alone, so every text of one topic has similarity 1 and a case pins a
cache hit or a polarity flip without depending on a model's paraphrase
quality. Any other text embeds as its lowercased words hashed into
DIMENSIONS buckets, so the same text repeats exactly, texts that share words
are similar and unrelated texts are not.
"""

import hashlib
import math
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

MODEL = "e2e/fixture-embedding"
DIMENSIONS = 256
TOPIC_MARKER = "__EMBED_TOPIC__"
TOPIC = re.compile(re.escape(TOPIC_MARKER) + r"\s+(\S+)")
WORD = re.compile(r"\w+")

router = APIRouter()


def embed(text: str) -> list[float]:
    topic = TOPIC.search(text)
    words = [topic.group(1)] if topic else WORD.findall(text.lower())
    vector = [0.0] * DIMENSIONS
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        body = await request.json()
    except ValueError:
        body = None
    inputs = body.get("input") if isinstance(body, dict) else None
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "input must be text or a list of text"}},
        )
    if body.get("dimensions") not in (None, DIMENSIONS):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"dimensions must be {DIMENSIONS}"}},
        )
    return {
        "object": "list",
        "model": body.get("model", MODEL),
        "data": [
            {"object": "embedding", "index": index, "embedding": embed(text)}
            for index, text in enumerate(inputs)
        ],
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in inputs),
            "total_tokens": sum(len(text.split()) for text in inputs),
        },
    }
