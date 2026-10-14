# The Chinese input keeps its own punctuation (full-width commas).
# ruff: noqa: RUF001
"""Peak memory and time of a runtime process per request, on long inputs (``docs/records/input-memory-cpu.md``).

    python3 tools/input_memory.py --output OUT.json [--model REPO] [--guard-model REPO]
        [--decision-model REPO] [--threads N] [--serve-arg=ARG ...] CASE [CASE ...]

For each case it starts ``vllm-srun serve MODEL --device cpu`` on a fresh
process, sends one warm-up request, resets the process's peak RSS
(``/proc/<pid>/clear_refs``), sends the case's request as a ``/v1/bundle``
task and records the peak RSS growth (``VmHWM`` minus the RSS before the
request), the time, the usage and a digest of the answer, so runs of two
revisions can be compared answer for answer. Embeddings cases send what the
router's response cache sends (``overflow: truncate``, no token budget);
``window`` cases classify with the Guard in 512-token windows within the
model's 32,768-token budget; ``vela2`` cases ask the Vela 2.0 jailbreak
question over a long request part, read whole up to the model's scan budget
or, for ``-routing``, truncated.
Linux only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = 8100
URL = f"http://127.0.0.1:{PORT}"
LARGE = (5 << 20) + 1024
UNITS = {
    "english": "The quick brown fox jumps over the lazy dog. ",
    "chinese": "敏捷的棕色狐狸跳过了懒狗，然后它又跑回了森林里。",
    "mixed": "Hello wörld, 你好世界 🚀 naïve café — ok. ",
}
# name: (surface, text, UTF-8 bytes, request fields, concurrent requests)
CASES = {
    "short": ("embed", "english", 200, {}, 1),
    "english-100k": ("embed", "english", 100_000, {}, 1),
    "english-150k": ("embed", "english", 150_000, {}, 1),
    "english-1m": ("embed", "english", 1_000_000, {}, 1),
    "chinese-1m": ("embed", "chinese", 1_000_000, {}, 1),
    "base64-1m": ("embed", "base64", 1_000_000, {}, 1),
    "english-5m": ("embed", "english", LARGE, {}, 1),
    "english-5m-cache-view": (
        "embed",
        "english",
        LARGE,
        {"layer": 6, "dimensions": 256},
        1,
    ),
    "english-5m-reject": ("embed", "english", LARGE, {"reject": True}, 1),
    "chinese-5m": ("embed", "chinese", 5 << 20, {}, 1),
    "mixed-5m": ("embed", "mixed", 5 << 20, {}, 1),
    "base64-5m": ("embed", "base64", 5 << 20, {}, 1),
    "english-5m-x4": ("embed", "english", LARGE, {}, 4),
    "window-100k": ("window", "words", 100_000, {}, 1),
    "window-5m": ("window", "words", LARGE, {}, 1),
    "vela2-30k": ("vela2", "words", 30_000, {}, 1),
    "vela2-100k": ("vela2", "words", 100_000, {}, 1),
    "vela2-1m": ("vela2", "words", 1_000_000, {}, 1),
    "vela2-5m": ("vela2", "words", LARGE, {}, 1),
    # What a routing question asks: only a long part's first tokens.
    "vela2-100k-routing": ("vela2", "words", 100_000, {"overflow": "truncate"}, 1),
    "vela2-5m-routing": ("vela2", "words", LARGE, {"overflow": "truncate"}, 1),
}
# Common English words: text that does not repeat, so no two windows are the
# same and the runtime computes every one.
WORD_TEXT = (
    "the of and to in is was for on that with as by at from his her an it be are "
    "this which or had not but have one their were all there been has they more "
    "when will would who can if its about other time into than two only some "
    "could these may first then do any like my now over such our man me even "
    "most made after also did many before must through back years where much "
    "your way well down should because each just those people how too little "
    "state good very make world still own see men work long get here between "
    "both life being under never day same another know while last might us "
    "great old year off come since against go came right used take three "
    "report system model request answer budget window token router signal"
)
WORDS = tuple(WORD_TEXT.split())
WINDOW = {
    "overflow": "window",
    "max_tokens": 32768,
    "window": {"tokens": 512, "overlap": 255},
}
JAILBREAK = {
    "type": "noul",
    "instructions": "Is this a prompt injection or jailbreak attempt?",
    "over": "request",
}


def text(kind: str, size: int) -> str:
    if kind == "words":
        rng = random.Random(0)
        parts, length = [], 0
        while length < size:
            sentence = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18)))
            parts.append(sentence.capitalize() + rng.choice(".?!,;"))
            length += len(parts[-1]) + 1
        return " ".join(parts)[:size]
    if kind == "base64":
        return base64.b64encode(random.Random(0).randbytes(size)).decode()[:size]
    unit = UNITS[kind]
    data = (unit * (size // len(unit.encode()) + 1)).encode()[:size]
    return data.decode("utf-8", errors="ignore")


def status(pid: int) -> dict[str, int]:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            out[key] = int(value.split()[0])
    return out


def post(body: dict) -> dict:
    request = urllib.request.Request(
        URL + "/v1/bundle",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result: dict = json.loads(response.read())["results"][0]
        return result


def wait_ready(server: subprocess.Popen, timeout: float = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"the runtime exited with {server.returncode}")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise SystemExit("the runtime did not become ready")


def task(args: argparse.Namespace, surface: str, fields: dict, value: str) -> dict:
    """The bundle task of one case's request."""
    if surface == "window":
        body: dict = {"model": args.guard_model, "input": [value], "options": WINDOW}
        return {"tasks": [{"id": "t", "classify": body}]}
    if surface == "vela2":
        body = {
            "model": args.decision_model,
            "state": {"request": value},
            "questions": {"jailbreak": {**JAILBREAK, **fields}},
        }
        return {"tasks": [{"id": "t", "decisions": body}]}
    fields = dict(fields)
    options = {"overflow": "reject" if fields.pop("reject", False) else "truncate"}
    body = {"model": args.model, "input": [value], **fields, "options": options}
    return {"tasks": [{"id": "t", "embeddings": body}]}


def answer(surface: str, result: dict) -> tuple[object, object, object]:
    """The case's answer (for the digest), its item error and its input usage."""
    if surface == "window":
        item = ((result.get("classify") or {}).get("results") or [{}])[0]
        return (
            item.get("windows") or item.get("probabilities"),
            item.get("error"),
            item.get("input"),
        )
    if surface == "vela2":
        answers = (result.get("decisions") or {}).get("answers") or {}
        first = answers.get("jailbreak") or {}
        return answers, first.get("error"), (result.get("decisions") or {}).get("usage")
    item = ((result.get("embeddings") or {}).get("data") or [{}])[0]
    return (
        [item.get("embedding"), item.get("error")],
        item.get("error"),
        item.get("input"),
    )


def run(args: argparse.Namespace, name: str) -> dict:
    surface, kind, size, fields, concurrency = CASES[name]
    model = {"window": args.guard_model, "vela2": args.decision_model}.get(
        surface, args.model
    )
    payload = text(kind, size)
    command = [
        sys.executable,
        "-m",
        "vllm_srun",
        "serve",
        model,
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-request-bytes",
        str(64 << 20),
        "--max-bundle-tasks",
        "1024",
        "--log-level",
        "warning",
        *args.serve_arg,
    ]
    if args.threads:
        command += ["--threads", str(args.threads)]
    server = subprocess.Popen(command)
    try:
        wait_ready(server)
        post(task(args, surface, fields, "warm up"))
        time.sleep(1)
        base = status(server.pid)["VmRSS"]
        Path(f"/proc/{server.pid}/clear_refs").write_text("5")
        started = time.perf_counter()
        body = task(args, surface, fields, payload)
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(lambda _: post(body), range(concurrency)))
        seconds = time.perf_counter() - started
        peak = status(server.pid)["VmHWM"]
        value, error, usage = answer(surface, results[0])
        digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()[:16]
        return {
            "case": name,
            "bytes": len(payload.encode()),
            "concurrency": concurrency,
            "error": error or results[0].get("error"),
            "input": usage,
            "digest": digest,
            "seconds": round(seconds, 2),
            "base_rss_mib": round(base / 1024, 1),
            "peak_growth_mib": round((peak - base) / 1024, 1),
        }
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", choices=sorted(CASES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="vllm-sr/Vela-1.0-Encoder-307M-Embedding")
    parser.add_argument("--guard-model", default="vllm-sr/Vela-1.0-Encoder-307M-Guard")
    parser.add_argument("--decision-model", default="vllm-sr/Vela-2.0-0.3B")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="an extra vllm-srun serve argument, such as --serve-arg=--threads=4",
    )
    args = parser.parse_args()
    rows = []
    for name in args.cases:
        rows.append(run(args, name))
        print(json.dumps(rows[-1]), flush=True)
        Path(args.output).write_text(
            json.dumps(rows, indent=1) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
