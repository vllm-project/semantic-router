"""Time to first token for a prefix computed from scratch against the same prefix served from a cache.

Sends prompts of an exact token length as token ids through /v1/completions, so no tokenizer is involved, each
twice. The first send computes and stores the KV, and the second is served from whatever cache the server has:
vLLM's GPU prefix cache, or LMCache's CPU tier when the server runs with the GPU prefix cache off. A short
unrelated prompt between the two sends keeps the second from reusing the first's in-flight state. TTFT is the
time to the first streamed token with max_tokens=1.
Usage: prefix_ttft.py --model MODEL --label ARM [--url URL] [--lengths 4096,8192,16384,32768] [--reps 5]
"""

import argparse
import json
import random
import time
import urllib.request


def ttft(url, model, ids):
    body = json.dumps(
        {
            "model": model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                return time.perf_counter() - t0
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--out", default="prefix_ttft.jsonl")
    a = ap.parse_args()
    rng = random.Random(0)
    ttft(a.url, a.model, [rng.randrange(1000, a.vocab) for _ in range(64)])  # warm-up
    with open(a.out, "a") as f:
        for n in map(int, a.lengths.split(",")):
            for rep in range(a.reps):
                ids = [rng.randrange(1000, a.vocab) for _ in range(n)]
                first = ttft(a.url, a.model, ids)
                ttft(a.url, a.model, [rng.randrange(1000, a.vocab) for _ in range(64)])
                second = ttft(a.url, a.model, ids)
                row = {
                    "arm": a.label,
                    "tokens": n,
                    "rep": rep,
                    "ttft_first": first,
                    "ttft_second": second,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
