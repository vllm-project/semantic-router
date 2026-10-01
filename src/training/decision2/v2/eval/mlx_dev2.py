"""MLX-DEV2: a held-out development twin of mlx-diag's card-eligible part (Choice + Noul).

Built exactly as `v2.eval.multilingual_panel` builds mlx-diag v1, from the publisher's
non-test splits:

* Choice: MASSIVE 1.1 (CC BY 4.0) **dev** partition, voice-assistant scenario, the same 18
  options and English instructions, locales en ar de es fr ja zh, parallel source IDs,
  the same localisation-quality filter in all six non-English locales.
* Noul: PAWS-X (free use with attribution) **validation** split, "same meaning?", languages
  en de es fr ja ko zh, parallel IDs, 50/50 labels.

There is no Score part: mlx-diag's Score part (XNLI) is CC BY-NC and is not card-eligible.
Rows are chosen by a salted hash of the source ID (plus scenario / label balance), never by
model behaviour. A source item flagged by the isolation scan in any language is dropped in
every language and the quota is refilled from the next hashed IDs.

    python3 -m v2.eval.mlx_dev2 pool --sources DIR --output POOL.jsonl
    python3 -m v2.eval.mlx_dev2 scan --pool POOL.jsonl --files LIST [--workers N] --output SCAN.json
    python3 -m v2.eval.mlx_dev2 build --sources DIR --exclude SCAN.json [...] --output PANEL_DIR
    python3 -m v2.eval.mlx_dev2 predictions --panel PANEL_DIR --answers ANSWERS.jsonl --output PRED.jsonl
    python3 -m v2.eval.mlx_dev2 score --panel PANEL_DIR --predictions PRED.jsonl --output SCORE.json
    python3 -m v2.eval.mlx_dev2 compare --panel PANEL_DIR --candidate A.jsonl --reference B.jsonl \
        [--candidate-name A] [--reference-name B] --output CMP.json

``scan`` reads every JSON string of every listed corpus file (``.jsonl`` or ``.jsonl.gz``)
and flags a pool part when (a) its normalised text equals a corpus string, (b) at least half
of its shingles (word 8-grams, or 16-character shingles for text under eight words) occur in
the corpus (the mlx-diag v1 audit rule), or (c) a part of 4–7 words occurs as a contiguous
word sequence inside a corpus string. ``predictions`` turns ``examples.py parity --answers``
output into scorer input. ``compare`` is the paired bootstrap of `v2.dec.mlx_paired` (paired
item draws within type x language, 5,000 replicates, seed 20260927, statistic = mean over
types of the mean over languages of accuracy) on Choice + Noul; the development guard passes
when its 95% upper bound is >= 0. Outputs hold aggregates and hashes only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from v2.eval.multilingual_panel import (
    CHOICE_Q,
    MASSIVE_LOCALES,
    NOUL_Q,
    PAWSX_LANGS,
    SCENARIOS,
    _normalize,
    _shingles,
    _strings,
    massive_passes,
)
from v2.eval.same_panel import input_digest, read_jsonl, sha_file, utc_now, write_json

VERSION = "dev2-mlx-dev2/1"
SALT = "mlx-dev2-v1"
TYPES = ("choice", "noul")
N_MASSIVE_PER_SCENARIO = 25
N_PAWSX = 400
SHORT_WORDS = range(4, 8)
SOURCE_SHA256 = {
    "massive/amazon-massive-dataset-1.1.tar.gz": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    "massive/1.1/data/en-US.jsonl": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
    "massive/1.1/data/ar-SA.jsonl": "b604b44d3bb94e4f71f64d041229e16ceb42d7b89385df7c82873d2514737186",
    "massive/1.1/data/de-DE.jsonl": "5e09cc550b38d37faf002e0ff42103acd330906e23032689e8f071e1cf3ff621",
    "massive/1.1/data/es-ES.jsonl": "310462a79fa181ff83c643a8d356c7b8155fd37a25e80a77ba3ca9b29305c4a5",
    "massive/1.1/data/fr-FR.jsonl": "f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49",
    "massive/1.1/data/ja-JP.jsonl": "c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79",
    "massive/1.1/data/zh-CN.jsonl": "992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace",
}
PAWSX_REVISION = (
    "google-research-datasets/paws-x@4cd8187c404bda33cb1f62b49b001115862acf37"
)


def key(*parts: str) -> str:
    return hashlib.sha256(":".join((SALT, *parts)).encode()).hexdigest()


def massive_pool(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Source ID -> locale -> record for dev-partition items usable in all seven locales."""
    data = {
        loc: {
            r["id"]: r
            for r in read_jsonl(root / "1.1" / "data" / f"{loc}.jsonl")
            if r["partition"] == "dev"
        }
        for loc in MASSIVE_LOCALES
    }
    ids = set(data["en-US"])
    for loc in MASSIVE_LOCALES[1:]:
        ids &= {i for i, r in data[loc].items() if massive_passes(r)}
    return {
        i: {loc: data[loc][i] for loc in MASSIVE_LOCALES}
        for i in ids
        if all(
            data[loc][i]["scenario"] == data["en-US"][i]["scenario"]
            for loc in MASSIVE_LOCALES
        )
    }


def pawsx_pool(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Source ID -> language -> record for validation pairs present and agreeing in all seven."""
    import pyarrow.parquet as pq

    tables = {
        lang: {
            str(r["id"]): r
            for r in pq.read_table(
                root / lang / "validation-00000-of-00001.parquet"
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
    return {
        i: {lang: tables[lang][i] for lang in PAWSX_LANGS}
        for i in ids
        if len({tables[lang][i]["label"] for lang in PAWSX_LANGS}) == 1
    }


def check_sources(sources: Path) -> dict[str, str]:
    observed = {}
    for relative, expected in SOURCE_SHA256.items():
        digest = sha_file(sources / relative)
        if digest != expected:
            raise ValueError(f"{relative}: {digest} differs from pinned {expected}")
        observed[relative] = digest
    for lang in PAWSX_LANGS:
        relative = f"paws-x/{lang}/validation-00000-of-00001.parquet"
        observed[relative] = sha_file(sources / relative)
    return observed


def pool(sources: Path) -> list[dict[str, Any]]:
    """Every candidate part (one text each) of the eligible pool, for the isolation scan."""
    parts = []
    for i, locales in massive_pool(sources / "massive").items():
        for loc, r in locales.items():
            parts.append(
                {
                    "source": "massive",
                    "source_id": i,
                    "language": loc.split("-")[0],
                    "text": r["utt"],
                }
            )
    for i, langs in pawsx_pool(sources / "paws-x").items():
        for lang, r in langs.items():
            for field in ("sentence1", "sentence2"):
                parts.append(
                    {
                        "source": "pawsx",
                        "source_id": i,
                        "language": lang,
                        "text": r[field],
                    }
                )
    return parts


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _index(parts: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, list[int]] = defaultdict(list)
    shingle: dict[str, list[int]] = defaultdict(list)
    short: dict[int, dict[str, list[int]]] = {n: defaultdict(list) for n in SHORT_WORDS}
    for n, part in enumerate(parts):
        exact[_normalize(part["text"])].append(n)
        for s in _shingles(part["text"]):
            shingle[s].append(n)
        words = _words(part["text"])
        if len(words) in SHORT_WORDS:
            short[len(words)][" ".join(words)].append(n)
    return {"exact": exact, "shingle": shingle, "short": short}


_STATE: dict[str, Any] = {}


def _init(pool_path: str) -> None:
    _STATE["index"] = _index(read_jsonl(Path(pool_path)))


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _scan_file(path: str) -> dict[str, Any]:
    index = _STATE["index"]
    exact, shingle, short = index["exact"], index["shingle"], index["short"]
    lengths = [n for n in SHORT_WORDS if short[n]]
    hits: dict[str, set[int]] = {"exact": set(), "contained": set()}
    seen: set[str] = set()
    strings = rows = 0
    try:
        with _open(Path(path)) as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
                for text in _strings(row):
                    if len(text) < 12:
                        continue
                    strings += 1
                    found = exact.get(_normalize(text))
                    if found:
                        hits["exact"].update(found)
                    for s in _shingles(text):
                        if s in shingle:
                            seen.add(s)
                    words = _words(text)
                    for n in lengths:
                        table = short[n]
                        for i in range(len(words) - n + 1):
                            found = table.get(" ".join(words[i : i + n]))
                            if found:
                                hits["contained"].update(found)
    except (OSError, EOFError, UnicodeError) as exc:
        return {"path": path, "error": type(exc).__name__}
    return {
        "path": path,
        "rows": rows,
        "strings": strings,
        "exact": sorted(hits["exact"]),
        "contained": sorted(hits["contained"]),
        "shingles": sorted(seen),
    }


def scan(pool_path: Path, files: list[Path], workers: int) -> dict[str, Any]:
    parts = read_jsonl(pool_path)
    own = [_shingles(p["text"]) for p in parts]
    exact: set[int] = set()
    contained: set[int] = set()
    seen: set[str] = set()
    corpus = {"files": 0, "rows": 0, "strings": 0, "unreadable": 0}
    with ProcessPoolExecutor(
        workers, initializer=_init, initargs=(str(pool_path),)
    ) as pool_:
        for result in pool_.map(_scan_file, [str(f) for f in files], chunksize=1):
            if "error" in result:
                corpus["unreadable"] += 1
                continue
            corpus["files"] += 1
            corpus["rows"] += result["rows"]
            corpus["strings"] += result["strings"]
            exact.update(result["exact"])
            contained.update(result["contained"])
            seen.update(result["shingles"])
    shared = {n for n, sh in enumerate(own) if sh and len(sh & seen) / len(sh) >= 0.5}
    reasons: dict[int, list[str]] = defaultdict(list)
    for label, members in (
        ("exact", exact),
        ("shingle", shared),
        ("contained", contained),
    ):
        for n in members:
            reasons[n].append(label)
    flagged: dict[str, set[str]] = defaultdict(set)
    for n in reasons:
        flagged[parts[n]["source"]].add(parts[n]["source_id"])
    by_reason = Counter(
        (parts[n]["source"], label) for n, labels in reasons.items() for label in labels
    )
    return {
        "schema": "dev2-mlx-dev2-scan/1",
        "pool_sha256": sha_file(pool_path),
        "pool_parts": len(parts),
        "pool_source_items": {
            s: len({p["source_id"] for p in parts if p["source"] == s})
            for s in ("massive", "pawsx")
        },
        "corpus": corpus,
        "corpus_files_sha256": hashlib.sha256(
            "\n".join(sorted(str(f) for f in files)).encode()
        ).hexdigest(),
        "flagged_parts": len(reasons),
        "flagged_parts_by_source_and_reason": {
            f"{s}/{r}": n for (s, r), n in sorted(by_reason.items())
        },
        "excluded_source_items": {s: sorted(v) for s, v in sorted(flagged.items())},
    }


def build(sources: Path, output: Path, exclude: dict[str, set[str]]) -> dict[str, Any]:
    observed = check_sources(sources)
    massive = massive_pool(sources / "massive")
    pawsx = pawsx_pool(sources / "paws-x")
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for i in sorted(
        set(massive) - exclude.get("massive", set()), key=lambda i: key("massive", i)
    ):
        by_scenario[massive[i]["en-US"]["scenario"]].append(i)
    chosen_massive = [
        i for s in sorted(SCENARIOS) for i in by_scenario[s][:N_MASSIVE_PER_SCENARIO]
    ]
    by_label: dict[int, list[str]] = defaultdict(list)
    for i in sorted(
        set(pawsx) - exclude.get("pawsx", set()), key=lambda i: key("pawsx", i)
    ):
        by_label[pawsx[i]["en"]["label"]].append(i)
    chosen_pawsx = by_label[0][: N_PAWSX // 2] + by_label[1][: N_PAWSX // 2]
    rows = []
    for i in chosen_massive:
        for loc in MASSIVE_LOCALES:
            record = massive[i][loc]
            rows.append(
                {
                    "source": "massive",
                    "source_id": i,
                    "language": loc.split("-")[0],
                    "type": "choice",
                    "state": record["utt"],
                    "question": CHOICE_Q,
                    "value": record["scenario"],
                }
            )
    for i in chosen_pawsx:
        for lang in PAWSX_LANGS:
            record = pawsx[i][lang]
            rows.append(
                {
                    "source": "pawsx",
                    "source_id": i,
                    "language": lang,
                    "type": "noul",
                    "state": f"Sentence 1: {record['sentence1']}\nSentence 2: {record['sentence2']}",
                    "question": NOUL_Q,
                    "value": bool(record["label"]),
                }
            )
    rows.sort(key=lambda r: key("order", r["source"], r["source_id"], r["language"]))
    prompts, gold = [], []
    for row in rows:
        item_id = f"mlx2-{row['source']}-{row['language']}-{key('id', row['source'], row['source_id'])[:12]}"
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
        "revisions": {
            "massive": "MASSIVE 1.1 official archive (dev partition)",
            "paws-x": f"{PAWSX_REVISION} (validation split)",
        },
        "pool_source_items": {"massive": len(massive), "pawsx": len(pawsx)},
        "excluded_source_items": {k: len(v) for k, v in sorted(exclude.items())},
        "items": len(prompts),
        "counts": {
            f"{t}/{lang}": n
            for (t, lang), n in sorted(
                Counter((g["type"], g["language"]) for g in gold).items()
            )
        },
        "per_scenario": {
            s: len(by_scenario[s][:N_MASSIVE_PER_SCENARIO]) for s in sorted(SCENARIOS)
        },
        "source_items": {
            s: len({g["source_id"] for g in gold if g["source"] == s})
            for s in ("massive", "pawsx")
        },
        "prompts_sha256": sha_file(output / "prompts.jsonl"),
        "gold_sha256": sha_file(output / "gold.jsonl"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def predictions(panel: Path, answers: Path) -> list[dict[str, Any]]:
    """``examples.py parity --answers`` lines -> prediction rows the scorer checks."""
    prompts = {p["id"]: p for p in read_jsonl(panel / "prompts.jsonl")}
    out = []
    for row in read_jsonl(answers):
        prompt = prompts.get(row["id"])
        if prompt is None:
            continue
        out.append(
            {
                "id": row["id"],
                "answers": row["answers"],
                "source_input_sha256": input_digest(
                    prompt["state"], prompt["questions"]
                ),
            }
        )
    return out


def outcomes(panel: Path, predictions_path: Path) -> dict[str, dict[str, Any]]:
    from v2.eval.overlap_effects import mlx_outcomes

    gold = {g["id"]: g for g in read_jsonl(panel / "gold.jsonl")}
    prompts = {p["id"]: p for p in read_jsonl(panel / "prompts.jsonl")}
    preds = {r["id"]: r for r in read_jsonl(predictions_path)}
    return mlx_outcomes(gold, prompts, preds)


def score(panel: Path, predictions_path: Path) -> dict[str, Any]:
    rows = outcomes(panel, predictions_path)
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows.values():
        cell = cells[(row["type"], row["language"])]
        cell[0] += row["correct"]
        cell[1] += 1
    by_type = {
        kind: {lang: c / n for (t, lang), (c, n) in sorted(cells.items()) if t == kind}
        for kind in TYPES
    }
    preds = {r["id"] for r in read_jsonl(predictions_path)}
    return {
        "schema": "dev2-mlx-dev2-score/1",
        "scope": "multilingual development guard; not a release score",
        "predictions_sha256": sha_file(predictions_path),
        "gold_sha256": sha_file(panel / "gold.jsonl"),
        "items": len(rows),
        "missing": len(set(rows) - preds),
        "card_type_macro_accuracy": statistics.mean(
            statistics.mean(by_type[t].values()) for t in TYPES
        ),
        "per_type": {
            t: {"macro": statistics.mean(v.values()), "languages": v}
            for t, v in by_type.items()
        },
    }


def compare(
    panel: Path,
    candidate: Path,
    reference: Path,
    names: tuple[str, str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    from v2.dec.mlx_paired import statistic
    from v2.eval.overlap_effects import mlx_strata

    cand, ref = outcomes(panel, candidate), outcomes(panel, reference)
    strata, kinds = mlx_strata(cand, ref, set())
    card = statistic(cand, ref, strata, kinds, TYPES, replicates, seed)
    return {
        "schema": "dev2-mlx-dev2-compare/1",
        "scope": "MLX-DEV2 development guard; paired candidate minus reference; aggregates only",
        "candidate_name": names[0],
        "reference_name": names[1],
        "gold_sha256": sha_file(panel / "gold.jsonl"),
        "predictions_sha256": {
            "candidate": sha_file(candidate),
            "reference": sha_file(reference),
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "method": "paired item draws within type x language (v2.eval.overlap_effects.strata_bootstrap); "
            "statistic = mean over types of the mean over languages of accuracy",
        },
        "card_eligible": card,
        "per_type": {
            t: statistic(cand, ref, strata, kinds, (t,), replicates, seed)
            for t in TYPES
        },
        "guard_pass": card["ci95"]["high"] >= 0,
    }


def main() -> None:
    from v2.eval.same_panel import PAIRED_REPLICATES, PAIRED_SEED

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("scan")
    s.add_argument("--pool", type=Path, required=True)
    s.add_argument("--files", type=Path, required=True, help="one corpus path per line")
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, required=True)
    b.add_argument("--exclude", type=Path, action="append", default=[])
    b.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("predictions")
    r.add_argument("--panel", type=Path, required=True)
    r.add_argument("--answers", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("--panel", type=Path, required=True)
    c.add_argument("--predictions", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--panel", type=Path, required=True)
    m.add_argument("--candidate", type=Path, required=True)
    m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--candidate-name", default="candidate")
    m.add_argument("--reference-name", default="reference")
    m.add_argument("--replicates", type=int, default=PAIRED_REPLICATES)
    m.add_argument("--seed", type=int, default=PAIRED_SEED)
    m.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pool":
        parts = pool(args.sources)
        args.output.write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in parts),
            encoding="utf-8",
        )
        print(json.dumps({"parts": len(parts), "sha256": sha_file(args.output)}))
    elif args.command == "scan":
        files = [Path(line) for line in args.files.read_text().splitlines() if line]
        result = scan(args.pool, files, args.workers)
        write_json(args.output, result, exclusive=False)
        print(
            json.dumps(
                {
                    "corpus": result["corpus"],
                    "flagged_parts": result["flagged_parts"],
                    "excluded": {
                        k: len(v) for k, v in result["excluded_source_items"].items()
                    },
                }
            )
        )
    elif args.command == "build":
        exclude: dict[str, set[str]] = defaultdict(set)
        for path in args.exclude:
            for source, ids in json.loads(path.read_text())[
                "excluded_source_items"
            ].items():
                exclude[source] |= set(map(str, ids))
        print(json.dumps(build(args.sources, args.output, exclude), indent=2))
    elif args.command == "predictions":
        rows = predictions(args.panel, args.answers)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"predictions": len(rows)}))
    elif args.command == "score":
        result = score(args.panel, args.predictions)
        write_json(args.output, result, exclusive=False)
        print(
            json.dumps(
                {k: result[k] for k in ("items", "missing", "card_type_macro_accuracy")}
            )
        )
    else:
        result = compare(
            args.panel,
            args.candidate,
            args.reference,
            (args.candidate_name, args.reference_name),
            args.replicates,
            args.seed,
        )
        write_json(args.output, result, exclusive=False)
        card = result["card_eligible"]
        print(
            json.dumps(
                {
                    "delta": round(card["delta"], 4),
                    "ci95": [
                        round(card["ci95"]["low"], 4),
                        round(card["ci95"]["high"], 4),
                    ],
                    "guard_pass": result["guard_pass"],
                }
            )
        )


if __name__ == "__main__":
    main()
