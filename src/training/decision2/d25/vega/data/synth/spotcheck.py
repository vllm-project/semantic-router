"""Independent quality audit of SYN1: blind relabel by a second-family judge, then adjudication.

1. Stratified sample (per archetype) of accepted rows, fixed seed.
2. The judge (a different model family from the generator/verifier, with a different prompt and a
   different item layout) answers each item blind and says whether it is well posed.
3. Every disagreement (or "not well posed") is adjudicated by the judge: it sees the item and the
   two candidate answers in random order, without knowing which one is the dataset's.
4. Report: raw disagreement, adjudicated label-error rate (dataset answer wrong or item broken),
   ambiguity rate, per archetype / condition / question kind, with Wilson 95% intervals.

    python -m d25.vega.data.synth.spotcheck --rows 'DIR/shards/*/rows.jsonl.gz' --per-archetype 60 \
        --url http://d25-vega-synth-judge:8000 --model glm-5.3-flash --out DIR/audit
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from d25.vega.common import decision_format as df
from d25.vega.data import util
from d25.vega.data.synth import prompts as P
from d25.vega.data.synth.llm import LLM, RequestCache, parse_json


def audit_layout(row: dict) -> str:
    question = row["question"]
    payload = {
        "state": row["state"] if row["state"] not in (None, "") else "(empty)",
        "question": question.get("instructions") or "Choose the best matching option.",
    }
    if question["type"] == "noul":
        payload["answer_format"] = "yes or no"
        criteria = question.get("criteria") or {}
        if criteria:
            payload["meaning"] = {
                "yes": criteria.get("true"),
                "no": criteria.get("false"),
            }
    else:
        payload["options"] = [
            {"key": k, "definition": v} for k, v in question["criteria"].items()
        ]
    return json.dumps(payload, ensure_ascii=False, indent=1)


def gold_label(row: dict) -> str:
    labels = P.labels_for(row["question"])
    return labels[row["label"]]


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def sample(paths: list[str], per_archetype: int, seed: int) -> list[dict]:
    by_archetype: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for row in util.read_jsonl(path):
            by_archetype[row["meta"]["archetype"]].append(row)
    rng = random.Random(seed)
    chosen = []
    for archetype in sorted(by_archetype):
        rows = sorted(by_archetype[archetype], key=lambda r: r["id"])
        rng.shuffle(rows)
        chosen += rows[:per_archetype]
    return chosen


async def judge(
    llm: LLM, row: dict, cache: RequestCache, budget: int, extra: dict
) -> dict:
    labels = P.labels_for(row["question"])
    messages = [
        {"role": "system", "content": P.AUDIT_SYSTEM},
        {"role": "user", "content": audit_layout(row)},
    ]
    result = await llm.chat(
        messages,
        cache=cache,
        schema=P.audit_schema(labels),
        thinking=True,
        budget=budget,
        max_tokens=budget + 1000,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=int(util.sha(row["id"] + ":audit", 8), 16),
        extra=extra,
    )
    if "error" in result or result["finish"] != "stop":
        return {"ok": False}
    try:
        out = parse_json(result["content"])
    except (json.JSONDecodeError, ValueError):
        return {"ok": False}
    return {"ok": True, **out}


async def adjudicate(
    llm: LLM,
    row: dict,
    gold: str,
    other: str,
    cache: RequestCache,
    budget: int,
    extra: dict,
) -> dict:
    rng = random.Random(row["id"])
    first_is_gold = rng.random() < 0.5
    first, second = (gold, other) if first_is_gold else (other, gold)
    content = (
        audit_layout(row)
        + "\n\n"
        + json.dumps({"first_candidate": first, "second_candidate": second})
    )
    messages = [
        {"role": "system", "content": P.ADJUDICATE_SYSTEM},
        {"role": "user", "content": content},
    ]
    result = await llm.chat(
        messages,
        cache=cache,
        schema=P.adjudicate_schema(),
        thinking=True,
        budget=budget,
        max_tokens=budget + 800,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=int(util.sha(row["id"] + ":adjudicate", 8), 16),
        extra=extra,
    )
    if "error" in result or result["finish"] != "stop":
        return {"ok": False}
    try:
        out = parse_json(result["content"])
    except (json.JSONDecodeError, ValueError):
        return {"ok": False}
    verdict = out.get("verdict")
    mapping = {"both": "ambiguous", "neither": "broken"}
    if verdict in ("first", "second"):
        outcome = (
            "gold_correct" if (verdict == "first") == first_is_gold else "gold_wrong"
        )
    else:
        outcome = mapping.get(verdict, "unknown")
    return {
        "ok": True,
        "verdict": verdict,
        "outcome": outcome,
        "reason": out.get("reason"),
    }


def summarise(records: list[dict]) -> dict:
    def block(items: list[dict]) -> dict:
        n = len(items)
        disagree = sum(r["disagree"] for r in items)
        wrong = sum(r["outcome"] in ("gold_wrong", "broken") for r in items)
        ambiguous = sum(r["outcome"] == "ambiguous" for r in items)
        return {
            "n": n,
            "disagreement": round(disagree / max(n, 1), 4),
            "label_error": round(wrong / max(n, 1), 4),
            "label_error_ci95": wilson(wrong, n),
            "ambiguous": round(ambiguous / max(n, 1), 4),
            "error_or_ambiguous": round((wrong + ambiguous) / max(n, 1), 4),
        }

    out = {"overall": block(records)}
    for field in (
        "archetype",
        "condition",
        "kind",
        "nota",
        "question_form",
        "state_language",
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[str(r["meta"].get(field))].append(r)
        out[f"by_{field}"] = {k: block(v) for k, v in sorted(groups.items())}
    out["outcomes"] = dict(Counter(r["outcome"] for r in records))
    return out


async def run(a) -> None:
    paths = sorted(p for pattern in a.rows for p in glob.glob(pattern))
    rows = sample(paths, a.per_archetype, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = RequestCache(out / "cache.jsonl")
    llm = LLM(a.url, a.model, concurrency=a.concurrency)
    extra = json.loads(a.extra) if a.extra else {}
    judged = await asyncio.gather(
        *(judge(llm, r, cache, a.budget, extra) for r in rows)
    )
    records, pending = [], []
    for row, j in zip(rows, judged):
        gold = gold_label(row)
        record = {
            "id": row["id"],
            "meta": row["meta"],
            "gold": gold,
            "judge": j.get("answer"),
            "well_posed": j.get("well_posed"),
            "judge_reason": j.get("reason"),
            "judge_ok": j.get("ok"),
        }
        record["disagree"] = (
            (not j.get("ok"))
            or j.get("answer") != gold
            or j.get("well_posed") is not True
        )
        record["outcome"] = "agree" if not record["disagree"] else "pending"
        records.append(record)
        if record["disagree"]:
            pending.append((record, row))

    async def settle(record, row):
        other = (
            record["judge"]
            if record["judge"] and record["judge"] != record["gold"]
            else None
        )
        if other is None:  # judge agreed on the label but called the item ill-posed
            labels = [x for x in P.labels_for(row["question"]) if x != record["gold"]]
            probs = row["meta"].get("teachers", {})
            other = labels[0] if labels else record["gold"]
            if probs:
                teacher = next(iter(probs.values()))
                order = sorted(range(len(teacher)), key=lambda i: -teacher[i])
                labels_all = P.labels_for(row["question"])
                other = next(
                    (labels_all[i] for i in order if labels_all[i] != record["gold"]),
                    other,
                )
        result = await adjudicate(
            llm, row, record["gold"], other, cache, a.budget, extra
        )
        record["adjudication"] = result
        record["outcome"] = (
            result.get("outcome", "unknown") if result.get("ok") else "unknown"
        )

    await asyncio.gather(*(settle(r, row) for r, row in pending))
    util.write_jsonl(out / "records.jsonl.gz", records)
    report = {
        "judge": {"model": a.model, "repo": a.repo, "revision": a.revision},
        "sample_seed": a.seed,
        "per_archetype": a.per_archetype,
        "rows_files": len(paths),
        "prompts": P.prompts_sha(),
        **summarise(records),
    }
    util.write_json(out / "report.json", report)
    print(json.dumps(report["overall"]), json.dumps(report["outcomes"]))
    cache.close()
    await llm.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--rows", nargs="+", required=True, help="glob(s) of accepted row files"
    )
    ap.add_argument("--per-archetype", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--revision", default="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument(
        "--extra", help="JSON merged into every request (for example reasoning_effort)"
    )
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
