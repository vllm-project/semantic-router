"""Planning, sharding and worker plumbing shared by ``run_suite`` and ``run_rows``.

A plan tokenizes every prompt once on CPU (``tokens.npy`` int32 + per-item offsets/lengths) and is cached
by a key over the inputs, the prompt family, the tokenizer files and a rendered probe prompt, so every
checkpoint with the same tokenizer and prompt reuses it. Items are assigned to shards by a deterministic
longest-processing-time split of token counts; each GPU worker process owns one shard, appends results
to its own JSONL files and skips finished items on restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PLAN_VERSION = 1
_CODEC = None


def stable_sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def tokenizer_files_sha(directory: Path) -> str:
    h = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = directory / name
        h.update(name.encode())
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def codec_key(codec) -> dict[str, Any]:
    probe = {
        "state": {"k": "v"},
        "question": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"a": None, "b": "x"},
        },
    }
    noul = {"state": "s", "question": {"type": "noul", "instructions": "Yes?"}}
    text = codec.text(probe["state"], probe["question"]) + codec.text(
        noul["state"], noul["question"]
    )
    return {
        "prompt": codec.prompt,
        "tokenizer": tokenizer_files_sha(codec.dir),
        "codes": stable_sha(codec.codes),
        "probe": hashlib.sha256(text.encode()).hexdigest(),
    }


def _init_codec(source: str, prompt: str) -> None:
    global _CODEC
    from d25.vega.eval.engine import PromptCodec

    _CODEC = PromptCodec(source, prompt=prompt)


def _encode_chunk(rows: list[dict[str, Any]]):
    import numpy as np

    ids = _CODEC.encode(rows)
    lengths = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
    flat = np.fromiter(
        (t for x in ids for t in x), dtype=np.int32, count=int(lengths.sum())
    )
    return lengths, flat


def tokenize(
    rows: Sequence[dict[str, Any]],
    source: str,
    prompt: str,
    out_dir: Path,
    workers: int,
    chunk: int = 256,
    log: Callable[[str], None] = print,
):
    """Tokenize ``{state, question}`` rows in parallel; writes ``tokens.npy`` and returns (lengths, offsets)."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [list(rows[i : i + chunk]) for i in range(0, len(rows), chunk)]
    lengths_parts, flats, total = [], [], 0
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=max(1, workers), initializer=_init_codec, initargs=(source, prompt)
    ) as pool:
        for n, (lengths, flat) in enumerate(pool.map(_encode_chunk, chunks)):
            lengths_parts.append(lengths)
            flats.append(flat)
            total += int(lengths.sum())
            if n % 100 == 0:
                log(
                    json.dumps(
                        {
                            "event": "tokenize",
                            "chunks": n + 1,
                            "of": len(chunks),
                            "tokens": total,
                            "seconds": round(time.time() - started, 1),
                        }
                    )
                )
    lengths = (
        np.concatenate(lengths_parts) if lengths_parts else np.zeros(0, dtype=np.int64)
    )
    offsets = (np.cumsum(lengths) - lengths).astype(np.int64)
    np.save(
        out_dir / "tokens.npy",
        np.concatenate(flats) if flats else np.zeros(0, dtype=np.int32),
    )
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "offsets.npy", offsets)
    return lengths, offsets


def lpt_shards(costs: Sequence[int], keys: Sequence[str], n: int) -> list[int]:
    """Deterministic longest-processing-time assignment of items to ``n`` shards."""
    import heapq

    order = sorted(range(len(costs)), key=lambda i: (-costs[i], keys[i]))
    heap = [(0, s) for s in range(n)]
    shard = [0] * len(costs)
    for i in order:
        load, s = heapq.heappop(heap)
        shard[i] = s
        heapq.heappush(heap, (load + int(costs[i]), s))
    return shard


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            if line.strip():
                yield json.loads(line)


def truncate_partial(path: Path) -> None:
    """Drop a trailing partial line left by an interrupted writer."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def spawn_workers(
    module: str,
    shards: Sequence[int],
    devices: Sequence[str],
    common: list[str],
    log_dir: Path,
    log: Callable[[str], None] = print,
) -> list[int]:
    """One worker process per (shard, device); returns exit codes."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard, device in zip(shards, devices):
        cmd = [
            sys.executable,
            "-m",
            module,
            "worker",
            "--shard",
            str(shard),
            "--device",
            device,
        ] + common
        handle = open(log_dir / f"worker-{shard:03d}.log", "a")
        procs.append(
            (
                shard,
                subprocess.Popen(
                    cmd, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy()
                ),
                handle,
            )
        )
        log(json.dumps({"event": "spawn", "shard": shard, "device": device}))
    codes = []
    for shard, proc, handle in procs:
        codes.append(proc.wait())
        handle.close()
        log(json.dumps({"event": "worker_exit", "shard": shard, "code": codes[-1]}))
    return codes


def parse_ids(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
