"""Frozen multilingual diagnostic panel (MLX-diag v1): native Choice / Noul / Score.

Built only from publisher TEST splits that no project artifact has used:

* Choice: MASSIVE 1.1 TEST (CC BY 4.0), scenario of an assistant request, 18 fixed
  options, locales en ar de es fr ja zh, parallel source IDs, localisation-quality
  filter applied in all six non-English locales.
* Noul: PAWS-X TEST (Google; free use with attribution), "same meaning?", languages
  en de es fr ja ko zh, parallel IDs, 50/50 labels.
* Score: XNLI TEST (CC BY-NC 4.0; internal diagnostic only), three ordered support
  levels (contradicts < neither < supports), languages en ar de es fr ru zh, one
  hypothesis per premise, balanced levels.

Instructions and option descriptions are English; the state is in the target
language, so this measures cross-lingual reading with English-instructed native
questions. It is a development diagnostic, not a release score or a sealed test.
Rows are selected by a hash of the source ID (plus label balance), never by model
behaviour. The independence unit is the source item (shared across languages).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval.same_panel import input_digest, read_jsonl, sha_file, utc_now, write_json

VERSION = "dev2-mlx-diag/1"
SALT = "mlx-diag-v1"
MASSIVE_LOCALES = ("en-US", "ar-SA", "de-DE", "es-ES", "fr-FR", "ja-JP", "zh-CN")
PAWSX_LANGS = ("en", "de", "es", "fr", "ja", "ko", "zh")
XNLI_LANGS = ("en", "ar", "de", "es", "fr", "ru", "zh")
N_MASSIVE, N_PAWSX, N_XNLI = 126, 100, 99
SOURCE_SHA256 = {
    "massive/amazon-massive-dataset-1.1.tar.gz": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    "massive/1.1/data/en-US.jsonl": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "massive/1.1/data/ar-SA.jsonl": "b604b44d3bb94e4f71f64d041229e16ceb42d7b89385df7c82873d2514737186",
    "massive/1.1/data/de-DE.jsonl": "5e09cc550b38d37faf002e0ff42103acd330906e23032689e8f071e1cf3ff621",
    "massive/1.1/data/es-ES.jsonl": "310462a79fa181ff83c643a8d356c7b8155fd37a25e80a77ba3ca9b29305c4a5",
    "massive/1.1/data/fr-FR.jsonl": "f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49",
    "massive/1.1/data/ja-JP.jsonl": "c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79",
    "massive/1.1/data/zh-CN.jsonl": "992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace",
    "paws-x/de/test-00000-of-00001.parquet": "9f50249f813edbde93aa86668775779671457aee6b0fa334d1acd5d8d9f83fe6",
    "paws-x/en/test-00000-of-00001.parquet": "fdb1e43f1068cf0cec955327cfdedc2d3065f15d3cd673c9d0002e6e87cc7915",
    "paws-x/es/test-00000-of-00001.parquet": "5ba872e113a79c783616e59d3d28ec0e922b6db745d797e5eb95010a32928bee",
    "paws-x/fr/test-00000-of-00001.parquet": "98844ef252bf38c42cbbf144ec30984c9e54c4c204d0e777c083831130593f4b",
    "paws-x/ja/test-00000-of-00001.parquet": "a75eb349a59de7c0db1c408c4db29f8d9fbbc75cbd57d637f8a2454ae89abfd8",
    "paws-x/ko/test-00000-of-00001.parquet": "b27692abb0b1be9f9a3a22f80cee749d9dd35870a6f2874c594777aac3aa6988",
    "paws-x/zh/test-00000-of-00001.parquet": "cdae3237c12a49165aacf489d3ab60b832b7672e7775abf937a1495f3ab75c80",
    "xnli/all_languages/test-00000-of-00001.parquet": "599e3a0191403f19cbe802afdf69841152000b41eaed725e4f463d432c0ffb49",
}
REVISIONS = {
    "massive": "MASSIVE 1.1 official archive",
    "paws-x": "google-research-datasets/paws-x@4cd8187c404bda33cb1f62b49b001115862acf37",
    "xnli": "facebook/xnli@b8dd5d7af51114dbda02c0e3f6133f332186418e",
}
SCENARIOS = {
    "alarm": "Setting, changing or checking alarms.",
    "audio": "Changing device audio such as volume or mute.",
    "calendar": "Calendar events, reminders and schedules.",
    "cooking": "Recipes and cooking help.",
    "datetime": "Current date, time or time zones.",
    "email": "Reading, writing or managing email.",
    "general": "General chit-chat, jokes or questions to the assistant.",
    "iot": "Controlling smart-home devices such as lights, plugs or vacuum.",
    "lists": "Creating or editing lists such as shopping or to-do lists.",
    "music": "Playing or managing music.",
    "news": "News headlines or topics.",
    "play": "Playing media such as radio, podcasts, audiobooks or games.",
    "qa": "Factual questions, definitions, maths or currency.",
    "recommendation": "Recommendations for places, events or movies.",
    "social": "Social media posts or complaints to companies.",
    "takeaway": "Ordering takeaway food or checking an order.",
    "transport": "Transport tickets, taxis, trains or traffic.",
    "weather": "Weather conditions or forecasts.",
}
CHOICE_Q = {
    "type": "choice",
    "instructions": "The state is a request a user said to a voice assistant. Which scenario does the request belong to?",
    "criteria": SCENARIOS,
}
NOUL_Q = {
    "type": "noul",
    "instructions": "Do Sentence 1 and Sentence 2 in the state have the same meaning, i.e. is one a paraphrase of the other?",
    "criteria": {
        "true": "The two sentences express the same meaning.",
        "false": "The sentences differ in meaning, for example in who did what, when or where.",
    },
}
SCORE_Q = {
    "type": "score",
    "instructions": "How strongly does the premise in the state support the hypothesis?",
    "criteria": [
        "The premise contradicts the hypothesis.",
        "The premise neither supports nor contradicts the hypothesis.",
        "The premise supports (entails) the hypothesis.",
    ],
}


def key(*parts: str) -> str:
    return hashlib.sha256(":".join((SALT, *parts)).encode()).hexdigest()


def massive_passes(record: dict[str, Any]) -> bool:
    judgments = record.get("judgments") or []
    intent_ok = sum(j.get("intent_score") in (1, 2) for j in judgments) >= 2
    grammar = [
        j.get("grammar_score") for j in judgments if type(j.get("grammar_score")) is int
    ]
    language = sum(j.get("language_identification") == "target" for j in judgments) >= 2
    return intent_ok and bool(grammar) and statistics.median(grammar) >= 3 and language


def build_massive(root: Path) -> list[dict[str, Any]]:
    data = {
        loc: {
            r["id"]: r
            for r in read_jsonl(root / "1.1" / "data" / f"{loc}.jsonl")
            if r["partition"] == "test"
        }
        for loc in MASSIVE_LOCALES
    }
    ids = set(data["en-US"])
    for loc in MASSIVE_LOCALES[1:]:
        ids &= {i for i, r in data[loc].items() if massive_passes(r)}
    ids = {
        i
        for i in ids
        if all(
            data[loc][i]["scenario"] == data["en-US"][i]["scenario"]
            for loc in MASSIVE_LOCALES
        )
    }
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids, key=lambda i: key("massive", i)):
        by_scenario[data["en-US"][i]["scenario"]].append(i)
    per = N_MASSIVE // len(SCENARIOS)
    chosen = [i for s in sorted(SCENARIOS) for i in by_scenario[s][:per]]
    rows = []
    for i in chosen:
        for loc in MASSIVE_LOCALES:
            rows.append(
                {
                    "source": "massive",
                    "source_id": i,
                    "language": loc.split("-")[0],
                    "type": "choice",
                    "state": data[loc][i]["utt"],
                    "question": CHOICE_Q,
                    "value": data[loc][i]["scenario"],
                }
            )
    return rows


def build_pawsx(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    tables = {
        lang: {
            r["id"]: r
            for r in pq.read_table(
                root / lang / "test-00000-of-00001.parquet"
            ).to_pylist()
        }
        for lang in PAWSX_LANGS
    }
    ids = set(tables["en"])
    for lang in PAWSX_LANGS:
        ids &= {
            i
            for i, r in tables[lang].items()
            if r["sentence1"].strip() and r["sentence2"].strip()
        }
    ids = {
        i for i in ids if len({tables[lang][i]["label"] for lang in PAWSX_LANGS}) == 1
    }
    by_label: dict[int, list] = defaultdict(list)
    for i in sorted(ids, key=lambda i: key("pawsx", str(i))):
        by_label[tables["en"][i]["label"]].append(i)
    chosen = by_label[0][: N_PAWSX // 2] + by_label[1][: N_PAWSX // 2]
    rows = []
    for i in chosen:
        for lang in PAWSX_LANGS:
            r = tables[lang][i]
            rows.append(
                {
                    "source": "pawsx",
                    "source_id": str(i),
                    "language": lang,
                    "type": "noul",
                    "state": f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
                    "question": NOUL_Q,
                    "value": bool(r["label"]),
                }
            )
    return rows


def build_xnli(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        root / "all_languages" / "test-00000-of-00001.parquet"
    ).to_pylist()
    by_premise: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(table):
        by_premise[row["premise"]["en"]].append(index)
    candidates = [
        min(members, key=lambda m: key("xnli-row", str(m)))
        for members in by_premise.values()
    ]
    level = {2: 0, 1: 1, 0: 2}  # contradiction, neutral, entailment -> ordered support
    by_level: dict[int, list[int]] = defaultdict(list)
    for index in sorted(candidates, key=lambda m: key("xnli", str(m))):
        by_level[level[table[index]["label"]]].append(index)
    chosen = [i for lvl in (0, 1, 2) for i in by_level[lvl][: N_XNLI // 3]]
    rows = []
    for index in chosen:
        row = table[index]
        hypotheses = dict(
            zip(row["hypothesis"]["language"], row["hypothesis"]["translation"])
        )
        for lang in XNLI_LANGS:
            rows.append(
                {
                    "source": "xnli",
                    "source_id": str(index),
                    "language": lang,
                    "type": "score",
                    "state": f"Premise: {row['premise'][lang]}\nHypothesis: {hypotheses[lang]}",
                    "question": SCORE_Q,
                    "value": level[row["label"]],
                }
            )
    return rows


def build(sources: Path, output: Path) -> dict[str, Any]:
    observed = {}
    for relative, expected in SOURCE_SHA256.items():
        path = sources / relative
        digest = sha_file(path)
        if digest != expected:
            raise ValueError(f"{relative}: {digest} differs from pinned {expected}")
        observed[relative] = digest
    rows = (
        build_massive(sources / "massive")
        + build_pawsx(sources / "paws-x")
        + build_xnli(sources / "xnli")
    )
    rows.sort(key=lambda r: key("order", r["source"], r["source_id"], r["language"]))
    prompts, gold = [], []
    for n, row in enumerate(rows):
        item_id = f"mlx-{row['source']}-{row['language']}-{key('id', row['source'], row['source_id'])[:12]}"
        questions = {"q": row["question"]}
        prompts.append({"id": item_id, "state": row["state"], "questions": questions})
        gold.append(
            {
                "id": item_id,
                "source": row["source"],
                "source_id": row["source_id"],
                "language": row["language"],
                "type": row["type"],
                "value": row["value"],
                "input_sha256": input_digest(row["state"], questions),
            }
        )
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("duplicate item id")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("prompts.jsonl", prompts), ("gold.jsonl", gold)):
        (output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
            encoding="utf-8",
        )
    manifest = {
        "version": VERSION,
        "created_utc": utc_now(),
        "sources": observed,
        "revisions": REVISIONS,
        "items": len(prompts),
        "counts": {
            f"{t}/{lang}": n
            for (t, lang), n in sorted(
                Counter((g["type"], g["language"]) for g in gold).items()
            )
        },
        "source_items": {
            s: len({g["source_id"] for g in gold if g["source"] == s})
            for s in ("massive", "pawsx", "xnli")
        },
        "prompts_sha256": sha_file(output / "prompts.jsonl"),
        "gold_sha256": sha_file(output / "gold.jsonl"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _shingles(text: str, n: int = 8) -> set[str]:
    tokens = re.findall(r"\w+", text.casefold())
    if len(tokens) < n:
        chars = re.sub(r"\s+", "", text.casefold())
        return (
            {chars[i : i + 16] for i in range(max(1, len(chars) - 15))}
            if len(chars) >= 16
            else set()
        )
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def audit(panel: Path, against: list[Path]) -> dict[str, Any]:
    gold = {g["id"]: g for g in read_jsonl(panel / "gold.jsonl")}
    prompts = read_jsonl(panel / "prompts.jsonl")
    exact: set[str] = set()
    shingles: set[str] = set()
    corpus = {}
    for path in against:
        n = 0
        for row in read_jsonl(path):
            for text in _strings(row):
                if len(text) >= 12:
                    exact.add(_normalize(text))
                    shingles |= _shingles(text)
                    n += 1
        corpus[str(path.name) if len(against) < 40 else path.as_posix()[-80:]] = n
    flagged = []
    for prompt in prompts:
        parts = [p.split(": ", 1)[-1] for p in prompt["state"].split("\n")]
        for part in parts:
            sh = _shingles(part)
            hit = _normalize(part) in exact
            share = (len(sh & shingles) / len(sh)) if sh else 0.0
            if hit or share >= 0.5:
                flagged.append(
                    {
                        "id": prompt["id"],
                        "source": gold[prompt["id"]]["source"],
                        "exact": hit,
                        "shingle_share": round(share, 3),
                    }
                )
                break
    return {
        "schema": "dev2-mlx-isolation-audit/1",
        "corpus_strings": corpus,
        "flagged": len(flagged),
        "flagged_by_source": dict(Counter(f["source"] for f in flagged)),
        "flagged_items": flagged[:200],
    }


def score(panel: Path, predictions: Path) -> dict[str, Any]:
    from benchmark.score import evaluate_answer

    gold = {g["id"]: g for g in read_jsonl(panel / "gold.jsonl")}
    prompts = {p["id"]: p for p in read_jsonl(panel / "prompts.jsonl")}
    preds = {r["id"]: r for r in read_jsonl(predictions)}
    if set(preds) - set(gold):
        raise ValueError("predictions contain unknown ids")
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    brier: dict[str, list[float]] = defaultdict(list)
    answers_by_source: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for item_id, g in gold.items():
        prompt, pred = prompts[item_id], preds.get(item_id)
        question = prompt["questions"]["q"]
        if pred is not None and pred.get("source_input_sha256") != g["input_sha256"]:
            raise ValueError(f"{item_id}: prediction made from different input")
        answer = ((pred or {}).get("answers") or {}).get("q")
        if g["type"] == "choice":
            target = {
                "type": "choice",
                "value": g["value"],
                "label_to_semantic": {k: k for k in question["criteria"]},
                "semantic_value": g["value"],
            }
        else:
            target = {"type": g["type"], "value": g["value"]}
        result = (
            evaluate_answer(question, target, answer)
            if answer is not None
            else {"status": "missing"}
        )
        cell = cells[(g["type"], g["language"])]
        cell[0] += bool(result.get("correct"))
        cell[1] += 1
        cell[2] += result.get("status") != "ok"
        if "probability" in result:
            brier[g["type"]].append(result["probability"]["brier"])
        answers_by_source[(g["source"], g["source_id"])][g["language"]] = (
            result.get("semantic_point") if result.get("status") == "ok" else None
        )
    by_type: dict[str, dict[str, Any]] = {}
    for kind in ("choice", "noul", "score"):
        langs = {
            lang: {"correct": c, "n": n, "invalid": inv, "accuracy": c / n}
            for (t, lang), (c, n, inv) in sorted(cells.items())
            if t == kind
        }
        en = langs.get("en", {}).get("accuracy")
        non_en = [v["accuracy"] for lang, v in langs.items() if lang != "en"]
        by_type[kind] = {
            "languages": langs,
            "english_accuracy": en,
            "non_english_mean_accuracy": statistics.mean(non_en) if non_en else None,
            "gap_vs_english": (
                (statistics.mean(non_en) - en) if non_en and en is not None else None
            ),
            "brier": statistics.mean(brier[kind]) if brier[kind] else None,
        }
    consistent = [
        len(set(v.values())) == 1 and None not in v.values()
        for v in answers_by_source.values()
    ]
    languages = sorted({lang for (_t, lang) in cells})
    per_language = {
        lang: statistics.mean(
            by_type[t]["languages"][lang]["accuracy"]
            for t in by_type
            if lang in by_type[t]["languages"]
        )
        for lang in languages
    }
    return {
        "schema": "dev2-mlx-diag-score/1",
        "scope": "multilingual development diagnostic; not a release score",
        "predictions_sha256": sha_file(predictions),
        "gold_sha256": sha_file(panel / "gold.jsonl"),
        "items": len(gold),
        "invalid_or_missing": sum(c[2] for c in cells.values()),
        "type_macro_accuracy": statistics.mean(
            statistics.mean(v["accuracy"] for v in by_type[t]["languages"].values())
            for t in by_type
        ),
        "non_english_type_macro_accuracy": statistics.mean(
            by_type[t]["non_english_mean_accuracy"] for t in by_type
        ),
        "english_type_macro_accuracy": statistics.mean(
            by_type[t]["english_accuracy"] for t in by_type
        ),
        "per_language_mean_accuracy": per_language,
        "cross_language_consistency": sum(consistent) / len(consistent),
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--sources", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    a = commands.add_parser("audit")
    a.add_argument("--panel", type=Path, required=True)
    a.add_argument("--against", type=Path, action="append", required=True)
    a.add_argument("--output", type=Path, required=True)
    s = commands.add_parser("score")
    s.add_argument("--panel", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.sources, args.output), indent=2))
    elif args.command == "audit":
        result = audit(args.panel, args.against)
        write_json(args.output, result, exclusive=False)
        print(json.dumps({k: result[k] for k in ("flagged", "flagged_by_source")}))
    else:
        result = score(args.panel, args.predictions)
        write_json(args.output, result, exclusive=False)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "type_macro_accuracy",
                        "english_type_macro_accuracy",
                        "non_english_type_macro_accuracy",
                        "cross_language_consistency",
                        "invalid_or_missing",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
