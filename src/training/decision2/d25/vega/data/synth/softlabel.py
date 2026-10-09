"""Big-LLM soft labels (S3): option-code probabilities from a served model for training rows.

The prompt is fixed: ``decision_format.render`` (system prompt, state, question, options under the
model's own 255 single-token answer codes, thinking disabled), and the readout is the next-token
probability of each option's code, renormalised over the question's options. Results go to
``meta.teachers.<name>``; the gold target is kept (``meta.gold_target`` is added when missing).
Rows that do not fit the server context are passed through unchanged and counted as unsupported.

    python -m d25.vega.data.synth.softlabel --rows IN.jsonl.gz --out OUT_DIR --name qwen3.5-397b-a17b-fp8 \
        --url http://d25-vega-synth-llm:8000 --model qwen3.5-397b-a17b-fp8 --tokenizer MODEL_DIR
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from d25.vega.common import decision_format as df
from d25.vega.data import util
from d25.vega.data.synth.llm import LLM


def prompt_record() -> dict:
    sample = df.user_prompt(
        "<state>",
        {
            "type": "choice",
            "instructions": "<instructions>",
            "criteria": {"<key>": "<description>"},
        },
        ["A"],
    )
    return {
        "format_id": df.FORMAT_ID,
        "system": df.SYSTEM_PROMPT,
        "user_template": sample,
        "thinking": False,
        "readout": "next-token probability of each option code, renormalised",
    }


async def label_part(
    llm: LLM, tokenizer, codes, ids, rows: list[dict], name: str, max_chars: int
) -> list[dict]:
    async def one(row: dict) -> dict:
        n = len(df.options(row["question"])[0])
        prompt = df.render(tokenizer, row["state"], row["question"], codes)
        meta = dict(row.get("meta") or {})
        if len(prompt) > max_chars:
            meta.setdefault("teacher_unsupported", {})[name] = "too_long"
            return dict(row, meta=meta)
        result = await llm.code_probs(prompt, ids[:n])
        if "error" in result:
            meta.setdefault("teacher_unsupported", {})[name] = str(
                result["error"].get("status")
            )
            return dict(row, meta=meta)
        teachers = dict(meta.get("teachers") or {})
        teachers[name] = [round(p, 6) for p in result["probs"]]
        meta["teachers"] = teachers
        meta.setdefault("gold_target", row["target"])
        return dict(row, meta=meta)

    return await asyncio.gather(*(one(r) for r in rows))


def score(rows: list[dict], name: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        probs = (row.get("meta") or {}).get("teachers", {}).get(name)
        label = row.get("label", -1)
        source = row.get("source", "?")
        if probs is None:
            groups[source].append(None)
            continue
        if label is None or label < 0:
            continue
        best = max(range(len(probs)), key=probs.__getitem__)
        groups[source].append((best == label, -math.log(max(probs[label], 1e-9))))

    def block(items):
        scored = [x for x in items if x is not None]
        return {
            "rows": len(items),
            "unsupported": sum(x is None for x in items),
            "scored": len(scored),
            "accuracy": round(sum(a for a, _ in scored) / max(len(scored), 1), 4),
            "nll": round(sum(b for _, b in scored) / max(len(scored), 1), 4),
        }

    everything = [x for v in groups.values() for x in v]
    return {
        "overall": block(everything),
        "by_source": {k: block(v) for k, v in sorted(groups.items())},
    }


async def run(a) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    codes, ids = df.answer_codes(tokenizer)
    llm = LLM(a.url, a.model, concurrency=a.concurrency)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for pattern in a.rows for p in glob.glob(pattern))
    part, buffer, done_rows = 0, [], []

    async def flush(index: int, chunk: list[dict]):
        target = out / f"part-{index:05d}.jsonl.gz"
        if target.exists():
            labelled = list(util.read_jsonl(target))
        else:
            labelled = await label_part(
                llm, tokenizer, codes, ids, chunk, a.name, a.max_chars
            )
            util.write_jsonl(target, labelled)
        print(
            f"part {index}: {len(labelled)} rows {json.dumps(score(labelled, a.name)['overall'])}",
            flush=True,
        )
        return labelled

    for path in paths:
        for row in util.read_jsonl(path):
            buffer.append(row)
            if len(buffer) == a.part_size:
                done_rows += await flush(part, buffer)
                part, buffer = part + 1, []
    if buffer:
        done_rows += await flush(part, buffer)
    report = {
        "teacher": {"name": a.name, "repo": a.repo, "revision": a.revision},
        "prompt": prompt_record(),
        "codes_sha": hashlib.sha256(json.dumps(codes).encode()).hexdigest()[:16],
        "inputs": paths,
        **score(done_rows, a.name),
        "server_stats": llm.stats.snapshot(),
    }
    util.write_json(out / "report.json", report)
    print(json.dumps(report["overall"]))
    await llm.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--name", required=True, help="teacher name used under meta.teachers"
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--revision", default="ea5b4f81096f3901c91dea97f81324302495781d")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--part-size", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument(
        "--max-chars", type=int, default=100000, help="skip prompts longer than this"
    )
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
