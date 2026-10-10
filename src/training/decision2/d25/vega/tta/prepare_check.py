"""Tokenizer-level checks of the permutation-average flag (no weights, CPU).

For every request: the released runtime's ``prepare`` vs this runtime's with the flag off must produce the same
token sequences, errors and runnable questions; with the flag on, every choice question with two or more options
gets a reversed-option prompt. Records per-request forward-pass shapes (questions, padded tokens per pass) for OFF
and ON, which ``latency_model.py`` turns into a latency prediction.

    python -m d25.vega.tta.prepare_check --old <v3.0.2 package dir (code + tokenizer)> \
        --new <package_d3 dir> --rows latency-760.jsonl.gz parity-600.jsonl.gz --out prepare-check.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path


def load_runtime(package: Path, name: str):
    for cached in ("d3_format", "d3_runtime"):
        sys.modules.pop(cached, None)
    sys.path.insert(0, str(package))
    try:
        spec = importlib.util.spec_from_file_location(name, package / "d3_runtime.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(package))
    return module


def tokenizer_only(module, tokenizer_dir: Path, flag: bool | None):
    """A D3 with the real tokenizer and answer codes but no weights (enough for ``prepare``)."""
    from transformers import AutoTokenizer

    model = object.__new__(module.D3)
    config = json.loads((tokenizer_dir / "decision_config.json").read_text())
    model.config = config
    model.prompt = config.get("prompt", "d3")
    model.max_length = int(config["max_length"]) if config.get("max_length") else None
    model.batch_size = module.DEFAULT_BATCH_SIZE
    model.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model.tokenizer.padding_side = "left"
    model.codes, model.token_ids = module.answer_codes(model.tokenizer)
    if config.get("codes") != model.codes or config.get("token_ids") != model.token_ids:
        raise SystemExit("answer codes differ from decision_config.json")
    model.processor = None
    model.image_unavailable = "tokenizer-only check"
    if flag is not None:
        model.permutation_average = flag
    return model


def passes(prepared, batch_size: int, merge_tokens: int) -> list[dict]:
    """Forward passes of a prepared text request (the runtime's merge rule): sequences, tokens, padded tokens."""
    keys = prepared.runnable
    out = []

    def record(seqs):
        out.append(
            {
                "sequences": len(seqs),
                "tokens": sum(map(len, seqs)),
                "padded": len(seqs) * max(map(len, seqs)),
            }
        )

    for start in range(0, len(keys), batch_size):
        chunk = keys[start : start + batch_size]
        forward = [prepared.sequences[k] for k in chunk]
        backward = [
            prepared.reversed_sequences[k]
            for k in chunk
            if k in prepared.reversed_sequences
        ]
        both = forward + backward
        if backward and len(both) * max(map(len, both)) > merge_tokens:
            record(forward)
            record(backward)
        else:
            record(both)
    return out


def read(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--rows", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    old_rt = load_runtime(args.old.absolute(), "d3_runtime_old")
    new_rt = load_runtime(args.new.absolute(), "d3_runtime_new")
    old = tokenizer_only(old_rt, args.old, None)
    off = tokenizer_only(new_rt, args.old, False)
    on = tokenizer_only(new_rt, args.old, True)
    report = {"old": str(args.old), "new": str(args.new), "sets": {}, "requests": {}}
    for path in args.rows:
        name = path.name.split(".")[0]
        stats = {
            "requests": 0,
            "identical_off": 0,
            "questions": 0,
            "reversed": 0,
            "choice_2plus": 0,
            "off_tokens": 0,
            "on_tokens": 0,
            "off_padded": 0,
            "on_padded": 0,
            "differing": [],
        }
        for row in read(path):
            rid = row["_evaluation"]["run_id"]
            a = old.prepare(row["state"], row["questions"])
            b = off.prepare(row["state"], row["questions"])
            c = on.prepare(row["state"], row["questions"])
            same = (a.keys, a.sequences, a.errors, a.runnable) == (
                b.keys,
                b.sequences,
                b.errors,
                b.runnable,
            )
            same = same and not b.reversed_sequences
            stats["requests"] += 1
            stats["identical_off"] += same
            if not same and len(stats["differing"]) < 20:
                stats["differing"].append(rid)
            stats["questions"] += len(row["questions"])
            stats["choice_2plus"] += sum(
                q.get("type") == "choice" and len(q.get("criteria") or {}) >= 2
                for q in row["questions"].values()
            )
            stats["reversed"] += len(c.reversed_sequences)
            if c.sequences != b.sequences:
                raise SystemExit(f"{rid}: ON changed the original-order prompts")
            p_off = passes(b, off.batch_size, new_rt.MERGE_TOKENS)
            p_on = passes(c, on.batch_size, new_rt.MERGE_TOKENS)
            for key, value in (("off", p_off), ("on", p_on)):
                stats[f"{key}_tokens"] += sum(p["tokens"] for p in value)
                stats[f"{key}_padded"] += sum(p["padded"] for p in value)
            report["requests"][rid] = {
                "set": name,
                "questions": len(row["questions"]),
                "off": p_off,
                "on": p_on,
            }
        stats["on_over_off_tokens"] = round(
            stats["on_tokens"] / max(1, stats["off_tokens"]), 4
        )
        stats["on_over_off_padded"] = round(
            stats["on_padded"] / max(1, stats["off_padded"]), 4
        )
        stats["pass"] = (
            stats["identical_off"] == stats["requests"]
            and stats["reversed"] == stats["choice_2plus"]
        )
        report["sets"][name] = stats
        print(
            json.dumps({name: {k: v for k, v in stats.items() if k != "differing"}}),
            flush=True,
        )
    args.out.write_text(json.dumps(report) + "\n")
    return 0 if all(s["pass"] for s in report["sets"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
