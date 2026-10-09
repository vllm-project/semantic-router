"""Build the same-skill proxy (S-proxy): fresh items for the 34 benchmarks with a private counterpart.

Every builder renders items exactly like the kit builder of its benchmark (same state layout, question
wording, option keys, scoring blocks) from data the public suite does not use; sources and row slices
are listed in ``holdouts.json``. Output: one ``<id>.jsonl.gz`` per benchmark in kit suite-row format
(with ``_evaluation``) plus ``manifest.json``.

    python -m d25.vega.eval.proxy.build_s --out /data/d25/vega/proxy/build/s [--only 25 30 ...]
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import random
import sqlite3
import sys
import time
import traceback
import unicodedata
import zipfile
from pathlib import Path

from d25.vega.eval.proxy.common import (
    NAMES,
    finish,
    mc_row,
    norm_text,
    question_count,
    rank,
    read_jsonl,
    rng,
    selected,
    sha256_file,
    stratified,
    take,
    write_jsonl,
)

WORK = Path("/work021")
RAW = WORK / "artifacts/benchmark-suite/raw"
SOURCES = WORK / "data/sources"
SUITE = Path("/data/d25/shared/index-suite-0.3")
KIT_DATA = Path("/data/d25/shared/decision-index-kit/decision_index/data")
N = 250  # target cases per benchmark unless noted


def hf_file(repo: str, filename: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def hf_list(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo, repo_type="dataset")


def parquet(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


class Public:
    """Fingerprints of public-suite rows per catalog id (for exact de-duplication and id exclusion)."""

    def __init__(self):
        self.texts = collections.defaultdict(set)
        self.ids = collections.defaultdict(set)
        self.rows = collections.defaultdict(list)
        keep_rows = {9, 31, 20, 22, 37, 36, 2, 50, 64, 44, 30, 56}
        files = [
            SUITE / "selected-rows.jsonl.gz",
            SUITE / "added-rows.jsonl.gz",
            SUITE / "gsm8k-rows.jsonl.gz",
        ]
        for path in files:
            for r in read_jsonl(path):
                n = r["_evaluation"]["catalog_id"]
                self.texts[n].add(norm_text(r["state"]))
                for q in r["questions"].values():
                    ins = q.get("instructions")
                    if isinstance(ins, dict):
                        ins = ins.get("candidate") or json.dumps(ins, sort_keys=True)
                    if ins and len(str(ins)) > 40:
                        self.texts[n].add(norm_text(ins))
                self.ids[n].add(str(r["_evaluation"]["group_id"]))
                for k in ("provenance", "metadata"):
                    sid = (r.get(k) or {}).get("source_id")
                    if sid is not None:
                        self.ids[n].add(str(sid))
                if n in keep_rows:
                    self.rows[n].append(r)

    def seen(self, n: int, *values) -> bool:
        return any(
            norm_text(v) in self.texts[n] for v in values if v not in (None, "", {})
        )


PUBLIC: Public | None = None


def public() -> Public:
    global PUBLIC
    if PUBLIC is None:
        t = time.time()
        PUBLIC = Public()
        print(
            json.dumps(
                {
                    "event": "public_index",
                    "seconds": round(time.time() - t, 1),
                    "benchmarks": len(PUBLIC.ids),
                }
            ),
            flush=True,
        )
    return PUBLIC


# ---------------------------------------------------------------- Knowledge & Reasoning


def b25_gpqa():
    with zipfile.ZipFile(SOURCES / "gpqa/dataset.zip") as z:
        read = lambda m: list(
            csv.DictReader(
                io.StringIO(z.read(m, pwd=b"deserted-untie-orchid").decode("utf-8-sig"))
            )
        )
        diamond = {norm_text(r["Question"]) for r in read("dataset/gpqa_diamond.csv")}
        extended = read("dataset/gpqa_extended.csv")
    pool = [
        (i, r)
        for i, r in enumerate(extended)
        if norm_text(r["Question"]) not in diamond
        and not public().seen(25, r["Question"])
    ]
    rows = []
    for i, r in take(pool, N, "gpqa", key=lambda x: x[1]["Question"]):
        options = [r["Correct Answer"]] + [
            r[f"Incorrect Answer {n}"] for n in (1, 2, 3)
        ]
        order = list(range(4))
        rng("GPQA", i).shuffle(order)
        criteria = {chr(65 + k): options[j] for k, j in enumerate(order)}
        rows.append(
            finish(
                {
                    "id": f"GPQA-extended:{i}",
                    "family": "GPQA-Diamond",
                    "split": "proxy",
                    "state": "",
                    "questions": {
                        "answer": {
                            "type": "choice",
                            "instructions": r["Question"],
                            "criteria": criteria,
                        }
                    },
                    "expected": {"answer": chr(65 + order.index(0))},
                    "metadata": {
                        "group_id": f"GPQA-extended:{i}",
                        "subdomain": r.get("Subdomain"),
                        "handling": "LOCAL EVALUATION ONLY",
                    },
                },
                25,
                track="GPQA-Diamond",
            )
        )
    return rows


def b30_gsm8k():
    from decision_index.suite.build import gsm8k_v3 as G3

    train = parquet(hf_file("openai/gsm8k", "main/train-00000-of-00001.parquet"))
    sl = [
        r
        for r in train
        if selected("gsm8k", r["question"], 16, 1)
        and not public().seen(30, r["question"])
    ]
    golds = {i: G3.gold_of(r["answer"]) for i, r in enumerate(sl)}
    sets = G3.option_sets(golds)
    rows = []
    for i, r in enumerate(sl):
        for count in (4, 10):
            opts = [G3.num(v) for v in sets[i][count]]
            rng("GSM", f"{i}:{count}").shuffle(opts)
            correct = opts.index(G3.num(golds[i]))
            rows.append(
                finish(
                    {
                        "id": f"GSM8K-{count}:train-slice:{i}",
                        "family": f"GSM8K-{count}",
                        "split": "proxy",
                        "state": {
                            "question": r["question"],
                            "task": f"GSM8K deterministic {count}-choice numeric selection",
                        },
                        "questions": {
                            "answer": {
                                "type": "choice",
                                "instructions": "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.",
                                "criteria": {
                                    f"option_{j}": v for j, v in enumerate(opts)
                                },
                            }
                        },
                        "expected": {"answer": f"option_{correct}"},
                        "metadata": {
                            "group_id": f"GSM8K:{i}",
                            "gold_numeric": G3.num(golds[i]),
                            "distractor_method": G3.METHOD
                            + " (within the proxy slice)",
                        },
                    },
                    30,
                    track=f"GSM8K-{count}choice",
                )
            )
    return rows


def b31_chess():
    import chess
    from decision_index.scoring.metrics import chess_score
    from decision_index.suite.build.adapters_scored import bag_records, decode_bag
    from decision_index.suite.build.freeze import priority

    by_fen = collections.defaultdict(dict)
    for rec in bag_records(RAW / "downloads/chessbench-test-action-value.bag"):
        fen, move, value = decode_bag(rec)
        by_fen[fen][move] = value
    pub = public().ids[31]
    order = sorted(
        by_fen, key=lambda fen: priority(31, hashlib.sha256(fen.encode()).hexdigest())
    )
    rows = []
    for fen in order:
        sid = hashlib.sha256(fen.encode()).hexdigest()
        if sid in pub:
            continue
        values = by_fen[fen]
        board = chess.Board(fen)
        moves = {m.uci(): m for m in board.legal_moves}
        if set(moves) != set(values) or not 2 <= len(moves) <= 255:
            continue
        best = max(values.values())
        accepted = sorted(k for k, v in values.items() if v == best)
        mo = sorted(
            moves, key=lambda m: hashlib.sha256((fen + ":" + m).encode()).digest()
        )
        row = {
            "id": "ChessBench:proxy:" + sid,
            "family": "ChessBench-legal-move",
            "split": "proxy",
            "state": {
                "fen": fen,
                "board": str(board),
                "side_to_move": "white" if board.turn else "black",
            },
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. Uppercase pieces are white. All legal moves are supplied.",
                    "criteria": {m: {"uci": m, "san": board.san(moves[m])} for m in mo},
                }
            },
            "expected": {"move": accepted[0]},
            "scoring": {
                "type": "chess_reference_value",
                "values": values,
                "accepted": accepted,
                "tie_tolerance": 0.0,
            },
            "metadata": {"group_id": sid, "legal_move_count": len(moves)},
        }
        assert all(chess_score(row, m)["best_move"] for m in accepted)
        rows.append(finish(row, 31, track="ChessBench-legal-move"))
        if len(rows) >= N:
            break
    return rows


def b44_cladder():
    with zipfile.ZipFile(RAW / "cladder/data/cladder-v1.zip") as z:
        qs = json.loads(z.read("cladder-v1-q-balanced.json"))
        models = {
            x["model_id"]: x for x in json.loads(z.read("cladder-v1-meta-models.json"))
        }
    pub = public().ids[44]
    pool = []
    for q in qs:
        if (
            str(q["question_id"]) in pub
            or f"CLadder:balanced:{q['question_id']}" in pub
        ):
            continue
        model = models[q["meta"]["model_id"]]
        prompt = "\n\n".join([model["background"], q["given_info"], q["question"]])
        if public().seen(44, prompt):
            continue
        pool.append((q, prompt))
    picked = stratified(
        pool,
        N,
        "cladder",
        stratum=lambda x: (x[0]["meta"].get("rung"), x[0]["answer"]),
        key=lambda x: x[0]["question_id"],
    )
    rows = []
    for q, prompt in picked:
        r = mc_row(
            "CLadder",
            "balanced-unselected",
            q["question_id"],
            prompt,
            ["yes", "no"],
            0 if q["answer"] == "yes" else 1,
            query_type=q["meta"].get("query_type"),
            rung=q["meta"].get("rung"),
        )
        rows.append(finish(r, 44, track="CLadder"))
    return rows


def b57_superg():
    data = list(read_jsonl(hf_file("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl")))
    pool = [r for r in data if 4 <= len(r["options"]) <= 26]
    rows = []
    for r in stratified(
        pool,
        300,
        "supergpqa",
        stratum=lambda r: r.get("discipline"),
        key=lambda r: r["uuid"],
    ):
        keys = [chr(65 + i) for i in range(len(r["options"]))]
        gold = r["answer_letter"].strip()
        if gold not in keys or norm_text(r["options"][keys.index(gold)]) != norm_text(
            r["answer"]
        ):
            continue
        rows.append(
            finish(
                {
                    "id": f"57:supergpqa:{r['uuid']}",
                    "family": "MMLU-Pro",
                    "split": "proxy",
                    "state": r["question"],
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "Which option is the correct answer?",
                            "criteria": dict(zip(keys, r["options"])),
                        }
                    },
                    "expected": {"q": gold},
                    "metadata": {
                        "group_id": f"57:supergpqa:{r['uuid']}",
                        "discipline": r.get("discipline"),
                        "field": r.get("field"),
                        "difficulty": r.get("difficulty"),
                    },
                },
                57,
                track=r.get("discipline"),
            )
        )
    return rows


def b28_winogrande():
    train = parquet(
        hf_file("allenai/winogrande", "winogrande_xl/train-00000-of-00001.parquet")
    )
    pool = [
        (i, r)
        for i, r in enumerate(train)
        if selected("winogrande", r["sentence"], 32, 1)
        and not public().seen(
            28, "Which option correctly fills the blank?\n" + r["sentence"]
        )
    ]
    rows = []
    for i, r in take(pool, 300, "winogrande", key=lambda x: x[1]["sentence"]):
        rows.append(
            finish(
                mc_row(
                    "WinoGrande",
                    "train-slice",
                    i,
                    "Which option correctly fills the blank?\n" + r["sentence"],
                    [r["option1"], r["option2"]],
                    int(r["answer"]) - 1,
                ),
                28,
                track="WinoGrande",
            )
        )
    return rows


# ---------------------------------------------------------------- Language Understanding


def b12_anli():
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    pool = []
    for rnd in (1, 2, 3):
        for r in parquet(
            hf_file("facebook/anli", f"plain_text/dev_r{rnd}-00000-of-00001.parquet")
        ):
            pool.append((rnd, r))
    rows = []
    for rnd, r in stratified(
        pool, 300, "anli", stratum=lambda x: x[0], key=lambda x: x[1]["uid"]
    ):
        ins = (
            "Classify the relationship between the premise and hypothesis.\nPremise: "
            + r["premise"]
            + "\nHypothesis: "
            + r["hypothesis"]
        )
        rows.append(
            finish(
                mc_row(
                    "ANLI",
                    f"dev_r{rnd}",
                    r["uid"],
                    ins,
                    list(mapping.values()),
                    int(r["label"]),
                ),
                12,
                track="ANLI",
            )
        )
    return rows


def b29_hellaswag():
    train = parquet(hf_file("Rowan/hellaswag", "data/train-00000-of-00001.parquet"))
    clean = lambda s: " ".join(str(s).split())
    pool = [
        (i, r) for i, r in enumerate(train) if selected("hellaswag", r["ctx"], 64, 1)
    ]
    pool = [
        (i, r)
        for i, r in pool
        if not public().seen(
            29, "Which continuation is most plausible?\n" + clean(r["ctx"])
        )
    ]
    rows = []
    for i, r in take(pool, 300, "hellaswag", key=lambda x: x[1]["ctx"]):
        rows.append(
            finish(
                mc_row(
                    "HellaSwag",
                    "train-slice",
                    f"{i}:{r['ind']}",
                    "Which continuation is most plausible?\n" + clean(r["ctx"]),
                    [clean(x) for x in r["endings"]],
                    int(r["label"]),
                ),
                29,
                track="HellaSwag",
            )
        )
    return rows


def b38_acos():
    from decision_index.suite.build.adapters_mechanical import (
        acos_sources,
        parse_acos_line,
    )

    class L:  # minimal layout shim for acos_sources
        repos = RAW / "repos"

    sources = acos_sources(L)
    parsed = {
        p: [
            parse_acos_line(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line
        ]
        for p in sources
    }
    by_domain = {}
    for p in sources:
        by_domain.setdefault(p.parent.name, set()).update(
            pair[0] for _, pairs in parsed[p] for pair in pairs
        )
    by_domain = {
        d: {(c, s) for c in cats for s in ("0", "1", "2")}
        for d, cats in by_domain.items()
    }
    pool = []
    for p in sources:
        if p.name.endswith("_dev.tsv"):
            for line_no, (review, gold) in enumerate(parsed[p], 1):
                if not public().seen(
                    38,
                    {
                        "review": review,
                        "task": "ACOS fixed category/sentiment presence",
                    },
                ):
                    pool.append((p, line_no, review, gold))
    want = {"Laptop-ACOS": 48, "Restaurant-ACOS": 52}
    rows = []
    for dom, k in want.items():
        sel = take(
            [x for x in pool if x[0].parent.name == dom],
            k,
            f"acos:{dom}",
            key=lambda x: x[2],
        )
        inventories = [
            sorted(by_domain[dom])[s : s + 64]
            for s in range(0, len(by_domain[dom]), 64)
        ]
        for p, line_no, review, gold in sel:
            for gi, inv in enumerate(inventories, 1):
                qs, ex = {}, {}
                for cat, sent in inv:
                    name = {"0": "negative", "1": "neutral", "2": "positive"}[sent]
                    key = f"{cat}__sentiment_{name}"
                    qs[key] = {
                        "type": "choice",
                        "instructions": (
                            "Given the review below, decide whether it expresses at least one "
                            f"aspect-category/sentiment pair ({cat}, {name}) anywhere "
                            "in the review. Answer yes only when that exact category and "
                            "sentiment pair is present; answer no otherwise. Multiple opposite "
                            "sentiments are separate valid questions."
                        ),
                        "criteria": {
                            "yes": "The exact category/sentiment pair is present.",
                            "no": "The exact category/sentiment pair is absent.",
                        },
                    }
                    ex[key] = "yes" if (cat, sent) in gold else "no"
                gid = f"{dom}:{p.stem}:line-{line_no}"
                rows.append(
                    finish(
                        {
                            "id": f"ACOS:category-sentiment:{gid}:group-{gi}",
                            "family": "ACOS-category-sentiment",
                            "split": "dev",
                            "state": {
                                "review": review,
                                "task": "ACOS fixed category/sentiment presence",
                            },
                            "questions": qs,
                            "expected": ex,
                            "metadata": {"group_id": gid, "inventory_group": gi},
                        },
                        38,
                        track="ACOS-category-sentiment",
                        group=gid,
                    )
                )
    return rows


def b39_fiqa():
    files = [
        f
        for f in hf_list("pauri32/fiqa-2018")
        if f.endswith((".parquet", ".jsonl", ".json", ".csv"))
    ]
    data = []
    for f in files:
        path = hf_file("pauri32/fiqa-2018", f)
        recs = (
            parquet(path)
            if f.endswith(".parquet")
            else (
                list(read_jsonl(path))
                if f.endswith(".jsonl")
                else list(csv.DictReader(open(path, encoding="utf-8")))
            )
        )
        for r in recs:
            r["_file"] = f
        data += recs
    docs = collections.defaultdict(list)
    for r in data:
        text = r.get("sentence") or r.get("text")
        target = r.get("target") or r.get("entity")
        score = r.get("sentiment_score", r.get("score"))
        if not text or not target or score is None:
            continue
        s = float(score)
        label = "Negative" if s <= -0.1 else "Positive" if s >= 0.1 else "Neutral"
        start = text.lower().find(str(target).lower())
        if start < 0:
            continue
        docs[text].append(
            (
                start,
                start + len(str(target)),
                text[start : start + len(str(target))],
                label,
            )
        )
    pool = [
        (t, sorted(set(v)))
        for t, v in docs.items()
        if not public().seen(
            39, {"document": t, "task": "FinEntity entity-given sentiment"}
        )
    ]
    rows = []
    for t, ents in take(pool, N, "fiqa", key=lambda x: x[0]):
        qs, ex = {}, {}
        seen_span = {}
        for j, (a, b, v, lab) in enumerate(ents):
            if (a, b) in seen_span:
                continue
            seen_span[(a, b)] = lab
            key = f"entity_{j:04d}_{a}_{b}"
            qs[key] = {
                "type": "choice",
                "instructions": (
                    "Classify the sentiment toward the supplied financial entity span. "
                    f"The entity span is given explicitly as {a}:{b} ({v}); "
                    "do not extract or alter it. Choose exactly one of Negative, Neutral, or Positive."
                ),
                "criteria": {x: x for x in ["Negative", "Neutral", "Positive"]},
            }
            ex[key] = lab
        sid = hashlib.sha256(t.encode()).hexdigest()[:16]
        rows.append(
            finish(
                {
                    "id": f"FinEntity:fiqa:{sid}",
                    "family": "FinEntity-entity-given",
                    "split": "proxy",
                    "state": {
                        "document": t,
                        "task": "FinEntity entity-given sentiment",
                    },
                    "questions": qs,
                    "expected": ex,
                    "metadata": {"group_id": f"fiqa:{sid}"},
                },
                39,
                track="FinEntity-entity-given",
            )
        )
    return rows


def b40_isarcasm():
    with open(
        RAW / "repos/isarcasm/train/train.En.csv", encoding="utf-8-sig", newline=""
    ) as f:
        data = [r for r in csv.DictReader(f) if r.get("tweet")]
    pool = [
        r
        for r in data
        if selected("isarcasm-en", r["tweet"], 4, 1)
        and r["sarcastic"] in ("0", "1")
        and not public().seen(40, r["tweet"])
    ]
    pos = [r for r in pool if r["sarcastic"] == "1"]
    neg = [r for r in pool if r["sarcastic"] == "0"]
    npos = min(len(pos), 60)
    picked = take(pos, npos, "isarcasm:pos", key=lambda r: r["tweet"]) + take(
        neg, min(len(neg), npos * 6), "isarcasm:neg", key=lambda r: r["tweet"]
    )
    rows = []
    for r in picked:
        sid = hashlib.sha256(r["tweet"].encode()).hexdigest()[:16]
        rows.append(
            finish(
                {
                    "id": f"iSarcasmEval-A-En:train-slice:{sid}",
                    "family": "iSarcasmEval-A-En",
                    "split": "train-slice",
                    "state": r["tweet"],
                    "questions": {
                        "sarcastic": {
                            "type": "choice",
                            "instructions": "Is this text intended to be sarcastic?",
                            "criteria": {"no": "No", "yes": "Yes"},
                        }
                    },
                    "expected": {"sarcastic": "yes" if r["sarcastic"] == "1" else "no"},
                    "metadata": {"group_id": sid},
                },
                40,
                track="iSarcasmEval-A-En",
            )
        )
    return rows


def b41_vast():
    labels = ["against", "favor", "neutral"]
    with open(RAW / "vast/data/VAST/vast_dev.csv", newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    pool = []
    for r in data:
        prompt = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
        if not public().seen(41, prompt):
            pool.append((r, prompt))
    rows = []
    for r, prompt in stratified(
        pool, 300, "vast", stratum=lambda x: x[0]["label"], key=lambda x: x[0]["new_id"]
    ):
        rows.append(
            finish(
                mc_row("VAST", "dev", r["new_id"], prompt, labels, int(r["label"])),
                41,
                track="VAST",
            )
        )
    return rows


def b42_nli4ct():
    criteria = {
        "Entailment": "The clinical trial evidence entails the statement.",
        "Contradiction": "The clinical trial evidence contradicts the statement.",
    }
    rows, pool = [], []
    with zipfile.ZipFile(RAW / "repos/nli4ct/training_data.zip") as z:
        names = {
            Path(n).name: n
            for n in z.namelist()
            if n.endswith(".json") and "MACOSX" not in n and "CT json" not in n
        }
        for split in ("dev", "train"):
            data = json.loads(z.read(names[f"{split}.json"]))
            for uid, row in data.items():
                if split == "train" and not selected("nli4ct-train", uid, 8, 1):
                    continue
                if row.get("Label") not in criteria or public().seen(
                    42,
                    "Classify the statement against the supplied clinical trial evidence:\n"
                    + row["Statement"],
                ):
                    continue
                pool.append((split, uid, row))
        picked = stratified(
            pool,
            300,
            "nli4ct",
            stratum=lambda x: (x[0], x[2]["Label"]),
            key=lambda x: x[1],
        )
        for split, uid, row in picked:
            section = row["Section_id"]
            state = {}
            for key, role in [
                ("Primary_id", "primary_trial"),
                ("Secondary_id", "secondary_trial"),
            ]:
                if key in row:
                    trial = json.loads(z.read("CT json/" + row[key] + ".json"))
                    state[role] = {
                        "id": row[key],
                        "section": section,
                        "text": trial[section],
                    }
            rows.append(
                finish(
                    {
                        "id": f"NLI4CT-2024:{split}:{uid}",
                        "family": "NLI4CT-2024",
                        "split": split,
                        "state": state,
                        "questions": {
                            "answer": {
                                "type": "choice",
                                "instructions": "Classify the statement against the supplied clinical trial evidence:\n"
                                + row["Statement"],
                                "criteria": criteria,
                            }
                        },
                        "expected": {"answer": row["Label"]},
                        "metadata": {"group_id": f"NLI4CT:{uid}"},
                    },
                    42,
                    track="NLI4CT-2024",
                )
            )
    return rows


def b59_ragtruth():
    from decision_index.suite.build.adapters_added import (
        RAGTRUTH_CRITERIA,
        RAGTRUTH_INSTRUCTIONS,
    )

    folder = RAW / "repos/cand-RAGTruth/dataset"
    sources = {
        x["source_id"]: x
        for x in map(json.loads, (folder / "source_info.jsonl").open(encoding="utf-8"))
    }
    pool = [
        r
        for r in map(json.loads, (folder / "response.jsonl").open(encoding="utf-8"))
        if r["split"] == "train" and selected("ragtruth", r["source_id"], 10, 1)
    ]
    picked = stratified(
        pool,
        300,
        "ragtruth",
        stratum=lambda r: (sources[r["source_id"]]["task_type"], len(r["labels"]) > 0),
        key=lambda r: r["id"],
    )
    rows = []
    for r in picked:
        s = sources[r["source_id"]]
        rows.append(
            finish(
                {
                    "id": f"59:proxy:{r['id']}",
                    "benchmark": "RAGTruth",
                    "family": "RAGTruth",
                    "split": "train-slice",
                    "state": {"prompt": s["prompt"], "response": r["response"]},
                    "questions": {
                        "q": {
                            "type": "noul",
                            "instructions": RAGTRUTH_INSTRUCTIONS,
                            "criteria": RAGTRUTH_CRITERIA,
                        }
                    },
                    "expected": {"q": len(r["labels"]) > 0},
                    "metadata": {
                        "group_id": f"59:{r['id']}",
                        "task_type": s["task_type"],
                    },
                },
                59,
                track=s["task_type"],
            )
        )
    return rows


# ---------------------------------------------------------------- Retrieval & Classification


def b04_banking77():
    import urllib.request

    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    text = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    data = list(csv.DictReader(io.StringIO(text)))
    labels = json.loads((RAW / "banking77/categories.json").read_text())
    pool = [
        (i, r)
        for i, r in enumerate(data)
        if selected("banking77", r["text"], 32, 1)
        and r["category"] in labels
        and not public().seen(
            4, "Classify the banking intent of this user request:\n" + r["text"]
        )
    ]
    rows = []
    for i, r in take(pool, 300, "banking77", key=lambda x: x[1]["text"]):
        rows.append(
            finish(
                mc_row(
                    "BANKING77",
                    "train-slice",
                    i,
                    "Classify the banking intent of this user request:\n" + r["text"],
                    labels,
                    labels.index(r["category"]),
                ),
                4,
                track="BANKING77",
            )
        )
    return rows


def b05_clinc():
    obj = json.loads((RAW / "clinc150/data_full.json").read_text())
    labels = sorted({p[1] for p in obj["train"]} | {"oos"})
    desc = [
        (
            "out of scope: none of the listed intents"
            if lab == "oos"
            else lab.replace("_", " ")
        )
        for lab in labels
    ]
    ins = "Classify the intent of this user request, or choose out of scope if none applies:\n"
    pool_in = [
        (i, t, lab)
        for i, (t, lab) in enumerate(obj["val"])
        if not public().seen(5, ins + t)
    ]
    pool_oos = [
        (i, t, lab)
        for i, (t, lab) in enumerate(obj["oos_val"])
        if not public().seen(5, ins + t)
    ]
    picked = [
        ("val",) + x for x in take(pool_in, 246, "clinc:in", key=lambda x: x[1])
    ] + [("oos_val",) + x for x in take(pool_oos, 54, "clinc:oos", key=lambda x: x[1])]
    return [
        finish(
            mc_row("CLINC150+OOS", split, i, ins + t, desc, labels.index(lab)),
            5,
            track="CLINC150+OOS",
        )
        for split, i, t, lab in picked
    ]


def _retrieval(n: int, family: str, n_queries: int):
    import tiktoken
    from decision_index.scoring.metrics import score_query
    from decision_index.suite.build.adapters_retrieval import project, retrieve
    import pyarrow.parquet as pq

    enc = tiktoken.get_encoding("cl100k_base")
    side = {}
    for line in open(
        RAW.parent / "retrieval-queries" / f"{family}.jsonl", encoding="utf-8"
    ):
        rec = json.loads(line)
        side[rec["id"]] = rec
    subsets = json.loads((KIT_DATA / "release-v2/retrieval-subsets.json").read_text())
    pub_ids = public().ids[n]
    toolret = family.startswith("ToolRet")
    if toolret:
        cfg = ast.parse((RAW / "repos/toolret/toolret/config.py").read_text())
        mapping = next(
            ast.literal_eval(x.value)
            for x in cfg.body
            if isinstance(x, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_TASK_2_CATEGORY"
                for t in x.targets
            )
        )
        paths = sorted((RAW / "toolret_queries").glob("*/*.parquet"))
    else:
        paths = sorted((RAW / "bright/examples").glob("*.parquet"))
    cands = []
    for p in paths:
        domain = p.parent.name if toolret else p.stem.replace("-00000-of-00001", "")
        for q in pq.read_table(p).to_pylist():
            rid = domain + ":" + str(q["id"])
            if rid in pub_ids or rid not in side:
                continue
            rec = side[rid]
            if not any(rec["qrels"].get(d, 0) > 0 for d in rec["scorable_ids"]):
                continue
            cands.append((domain, q, rid))
    picked = stratified(
        cands, n_queries, family, stratum=lambda x: x[0], key=lambda x: x[2]
    )
    rows = []
    conns = {}
    for domain, q, rid in picked:
        category = mapping[domain] if toolret else domain
        index_name = ("toolret_" if toolret else "bright_") + category
        if index_name not in conns:
            conns[index_name] = sqlite3.connect(
                f"file:{RAW.parent / 'retrieval-indexes-v2' / (index_name + '.sqlite')}?mode=ro",
                uri=True,
            )
        excluded = set(str(x) for x in q.get("excluded_ids", []) or [])
        qrels = (
            {str(x["id"]): 1 for x in json.loads(q["labels"])}
            if toolret
            else {str(x): 1 for x in q["gold_ids"]}
        )
        hits = retrieve(conns[index_name], q["query"], excluded, 32)
        meta = {"domain": domain, "category": category, "corpus_index": index_name}
        rs, record = project(family, rid, q["query"], hits, qrels, meta, enc)
        assert record["scorable_ids"] == side[rid]["scorable_ids"], rid
        for r in rs:
            rows.append(finish(r, n, track=family, group=rid))
    for c in conns.values():
        c.close()
    return rows


def b36_bright():
    return _retrieval(36, "BRIGHT-retrieval", 110)


def b02_toolret():
    return _retrieval(2, "ToolRet-retrieval", 160)


def b37_esci():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base = RAW / "repos/esci/shopping_queries_dataset"
    ex = pq.read_table(
        base / "shopping_queries_dataset_examples.parquet",
        filters=[("split", "=", "test")],
    )
    pub_rows = public().rows[37]
    pub_q = {r["metadata"]["query_id"] for r in pub_rows}
    pub_ids = {str(r["metadata"]["source_id"]) for r in pub_rows}
    keep = [
        i
        for i, (q, e) in enumerate(
            zip(ex["query_id"].to_pylist(), ex["example_id"].to_pylist())
        )
        if q not in pub_q and str(e) not in pub_ids
    ]
    ex = ex.take(keep)
    exs = ex.to_pylist()
    picked = stratified(
        exs,
        300,
        "esci",
        stratum=lambda r: (r["product_locale"], r["esci_label"]),
        key=lambda r: r["example_id"],
    )
    need = {(r["product_locale"], r["product_id"]) for r in picked}
    prods = {}
    for b in pq.ParquetFile(
        base / "shopping_queries_dataset_products.parquet"
    ).iter_batches(batch_size=65536):
        for r in b.to_pylist():
            k = (r["product_locale"], r["product_id"])
            if k in need:
                prods[k] = r
    criteria = {
        "E": "Exact: the product satisfies the search query.",
        "S": "Substitute: a product that could substitute for the requested product.",
        "C": "Complement: a product that complements the requested product.",
        "I": "Irrelevant: the product does not address the requested product need.",
    }
    rows = []
    for r in picked:
        p = prods[(r["product_locale"], r["product_id"])]
        product = {
            k.removeprefix("product_"): p[k]
            for k in [
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            ]
            if p.get(k) is not None
        }
        rows.append(
            finish(
                {
                    "id": f"Amazon-ESCI:test-unselected:{r['example_id']}",
                    "family": "Amazon-ESCI",
                    "split": "test-unselected",
                    "state": {"search_query": r["query"], "product": product},
                    "questions": {
                        "answer": {
                            "type": "choice",
                            "instructions": "Classify the relevance of this product to the search query using the ESCI categories.",
                            "criteria": criteria,
                        }
                    },
                    "expected": {"answer": r["esci_label"]},
                    "metadata": {
                        "group_id": f"esci:{r['example_id']}",
                        "query_id": r["query_id"],
                        "locale": r["product_locale"],
                    },
                },
                37,
                track="Amazon-ESCI",
            )
        )
    return rows


def b56_phish():
    from decision_index.suite.build.adapters_added import assignments

    questions = assignments(RAW / "repos/jev-phishing-bench/run_jev.py", {"QUESTIONS"})[
        "QUESTIONS"
    ]
    for q in questions.values():
        q.setdefault("criteria", {})
    pub_text = public().texts[56]
    pool = []
    for f in (
        "cross_domain_legitimate_v5.csv",
        "infrastructure_phishing_expanded.csv",
        "real_phishing_validation.csv",
    ):
        for r in csv.DictReader(
            open(hf_file("AreLit/PhishNChips", f), encoding="utf-8")
        ):
            try:
                email = json.loads(r["email_content"]) if "email_content" in r else r
            except (json.JSONDecodeError, TypeError):
                continue
            state = {
                k: email.get(k)
                for k in [
                    "sender",
                    "from",
                    "subject",
                    "body",
                    "link_display_text",
                    "link_url",
                ]
            }
            if not state.get("body") or norm_text(state) in pub_text:
                continue
            label = r.get("phish_label", r.get("label"))
            if label in (None, ""):
                continue
            pool.append(
                (
                    f,
                    r.get("id")
                    or hashlib.sha256(
                        json.dumps(state, sort_keys=True).encode()
                    ).hexdigest()[:16],
                    state,
                    bool(int(float(label))),
                )
            )
    picked = stratified(
        pool, N, "phish", stratum=lambda x: (x[0], x[3]), key=lambda x: str(x[1])
    )
    rows = []
    for f, sid, state, gold in picked:
        expected = {k: None for k in questions}
        expected.update(
            verdict="phishing" if gold else "legitimate",
            is_phishing=gold,
            verdict_alt_click="do_not_click" if gold else "click",
            verdict_alt_minimal="phishing" if gold else "legitimate",
        )
        rows.append(
            finish(
                {
                    "id": f"56:proxy:{f.split('.')[0]}:{sid}",
                    "benchmark": "PhishNChips",
                    "family": "PhishNChips",
                    "split": "proxy",
                    "state": state,
                    "questions": json.loads(json.dumps(questions)),
                    "expected": expected,
                    "metadata": {"group_id": f"56:{f}:{sid}", "source_file": f},
                    "scoring": {
                        "type": "calibration_decisions",
                        "primary_field": "verdict",
                        "unscored_fields": [
                            k for k in questions if k.startswith("sig_")
                        ],
                    },
                },
                56,
                track="native-nine-question",
            )
        )
    return rows


def b61_hover():
    from decision_index.suite.build.adapters_added import (
        HOVER_CRITERIA,
        HOVER_INSTRUCTIONS,
    )

    data = json.loads(
        (RAW / "repos/cand-hover/data/hover/hover_train_release_v1.1.json").read_text(
            encoding="utf-8"
        )
    )
    pool = [r for r in data if selected("hover", r["uid"], 50, 1)]
    picked = stratified(
        pool,
        300,
        "hover",
        stratum=lambda r: (r["num_hops"], r["label"]),
        key=lambda r: r["uid"],
    )
    db = sqlite3.connect(f"file:{RAW / 'hover/wiki_wo_links.db'}?mode=ro", uri=True)
    rows = []
    for r in picked:
        evidence = []
        for t in dict.fromkeys(t for t, _ in r["supporting_facts"]):
            found = db.execute(
                "SELECT id, text FROM documents WHERE id=?",
                (unicodedata.normalize("NFD", t),),
            ).fetchall()
            if len(found) != 1:
                evidence = None
                break
            evidence.append({"title": t, "text": found[0][1]})
        if evidence is None or public().seen(
            61, {"claim": r["claim"], "evidence": evidence}
        ):
            continue
        rows.append(
            finish(
                {
                    "id": f"61:proxy:{r['uid']}",
                    "benchmark": "HoVer",
                    "family": "HoVer",
                    "split": "train-slice",
                    "state": {"claim": r["claim"], "evidence": evidence},
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": HOVER_INSTRUCTIONS,
                            "criteria": HOVER_CRITERIA,
                        }
                    },
                    "expected": {"q": r["label"]},
                    "metadata": {
                        "group_id": f"61:{r['uid']}",
                        "num_hops": r["num_hops"],
                    },
                },
                61,
                track=f"{r['num_hops']}-hop",
            )
        )
    db.close()
    return rows


# ---------------------------------------------------------------- Tools & Automation


def b01_bfcl():
    from decision_index.suite.build.adapters_mechanical import _tool_names
    import re

    base = RAW / "repos/bfcl/berkeley-function-call-leaderboard/data"
    cats = [
        "multiple",
        "parallel",
        "parallel_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "java",
        "javascript",
    ]
    pool = []
    for cat in cats:
        data, goldp = (
            base / f"BFCL_v3_{cat}.json",
            base / "possible_answer" / f"BFCL_v3_{cat}.json",
        )
        if not data.exists() or not goldp.exists():
            continue
        gs = {x["id"]: x["ground_truth"] for x in (json.loads(l) for l in goldp.open())}
        for raw in data.open():
            item = json.loads(raw)
            truth = gs.get(item["id"])
            if not truth or not isinstance(item.get("function"), list):
                continue
            needed = _tool_names(truth)
            cands = [
                f["name"]
                for f in item["function"]
                if isinstance(f, dict) and f.get("name")
            ]
            if needed and cands and needed <= set(cands):
                pool.append((cat, item, needed, cands))
    picked = stratified(
        pool, N, "bfcl", stratum=lambda x: x[0], key=lambda x: x[1]["id"]
    )
    ins = (
        "Given the complete user conversation and the published function schemas in state, mark each "
        "candidate tool yes if it should be invoked to answer the request, or no otherwise. "
        "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n"
    )
    rows = []
    for cat, item, needed, cands in picked:
        qs, ex = {}, {}
        for name in cands:
            key = "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
            qs[key] = {
                "type": "choice",
                "instructions": ins + f"\n\nCandidate tool: {name}",
                "criteria": {
                    "yes": "Invoke this tool.",
                    "no": "Do not invoke this tool.",
                },
            }
            ex[key] = "yes" if name in needed else "no"
        rows.append(
            finish(
                {
                    "id": f"BFCL-tool-selection:{cat}:{item['id']}",
                    "family": "BFCL-tool-selection",
                    "split": "test",
                    "state": {
                        "conversation": item["question"],
                        "functions": item["function"],
                        "task": "BFCL tool-name selection",
                    },
                    "questions": qs,
                    "expected": ex,
                    "metadata": {
                        "group_id": f"bfcl:{cat}:{item['id']}",
                        "category": cat,
                    },
                },
                1,
                track="BFCL-tool-selection",
            )
        )
    return rows


def b09_home():
    from decision_index.suite.build import home_appliance as HA

    seen = {norm_text(r["state"]) for r in public().rows[9]}
    rows = []
    for seed in range(2026100901, 2026100921):
        HA.SEED = seed
        for h in range(len(HA.HOUSEHOLDS)):
            for i in range(10):
                r = HA.make_row(h, i)
                key = norm_text(r["state"])
                if key in seen:
                    continue
                seen.add(key)
                r["id"] = f"home-appliance:proxy-{seed}:{HA.HOUSEHOLDS[h]}:{i:02d}"
                r["split"] = "proxy"
                r["metadata"]["group_id"] = r["id"]
                rows.append(r)
    picked = take(rows, N, "home", key=lambda r: r["id"])
    return [finish(r, 9, track="Home-Appliance") for r in picked]


# ---------------------------------------------------------------- Arts & Human Taste


def b20_bpomp():
    from decision_index.suite.build.freeze import priority

    pairs = []
    for p in sorted((RAW / "downloads").glob("BPoMP_p*.json")):
        for variant, items in json.loads(p.read_text()).items():
            for i, pair in enumerate(items):
                if isinstance(pair, list) and len(pair) >= 2 and pair[0] != pair[1]:
                    pairs.append((p.stem, variant, i, pair[0], pair[1]))
    counts = collections.Counter(
        hashlib.sha256(a.encode()).hexdigest() for _, _, _, a, _ in pairs
    )
    used, chosen = 0, set()
    for g in sorted(counts, key=lambda g: priority(20, g)):
        if used + counts[g] <= 5000:
            chosen.add(g)
            used += counts[g]
    pool = [x for x in pairs if hashlib.sha256(x[3].encode()).hexdigest() not in chosen]
    picked = stratified(
        pool, N, "bpomp", stratum=lambda x: x[1], key=lambda x: f"{x[0]}:{x[1]}:{x[2]}"
    )
    rows = []
    for stem, variant, i, a, b in picked:
        key = hashlib.sha256(a.encode()).hexdigest()
        order = [0, 1]
        rng("BPoMP", f"{key}:{variant}:{i}").shuffle(order)
        ordered = [[a, b][j] for j in order]
        rows.append(
            finish(
                {
                    "id": f"BPoMP:original-limerick:{stem}:{variant}:{i}",
                    "family": "BPoMP-original-limerick",
                    "split": "unselected",
                    "state": {"task": "BPoMP original-versus-perturbed discrimination"},
                    "questions": {
                        "answer": {
                            "type": "choice",
                            "instructions": "Which candidate is the original released limerick? Judge rhyme, meter, and linguistic coherence. The original is not defined by subjective human preference.",
                            "criteria": {f"option_{j}": ordered[j] for j in range(2)},
                        }
                    },
                    "expected": {"answer": f"option_{ordered.index(a)}"},
                    "metadata": {"group_id": key, "variant": variant},
                },
                20,
                track="BPoMP-original-limerick",
                group=key,
            )
        )
    return rows


def b21_humicroedit():
    import re

    with zipfile.ZipFile(RAW / "downloads/humicroedit-full.zip") as z:
        data = list(
            csv.DictReader(
                io.StringIO(
                    z.read("semeval-2020-task-7-dataset/subtask-2/dev.csv").decode(
                        "utf-8-sig"
                    )
                )
            )
        )
    pool = []
    for r in data:
        if r["label"] not in ("1", "2"):
            continue
        crit = {}
        for n in (1, 2):
            rendered, count = re.subn(
                r"<[^<>]+/>", lambda _: r[f"edit{n}"], r[f"original{n}"]
            )
            if count != 1:
                break
            crit[f"headline_{n}"] = rendered
        if len(crit) == 2:
            pool.append((r, crit))
    rows = []
    for r, crit in take(pool, 300, "humicroedit", key=lambda x: x[0]["id"]):
        rows.append(
            finish(
                {
                    "id": f"Humicroedit:dev:{r['id']}",
                    "family": "Humicroedit",
                    "split": "dev",
                    "state": "",
                    "questions": {
                        "answer": {
                            "type": "choice",
                            "instructions": "Which edited news headline is funnier?",
                            "criteria": crit,
                        }
                    },
                    "expected": {"answer": "headline_" + r["label"]},
                    "metadata": {"group_id": f"humicroedit:{r['id']}"},
                },
                21,
                track="Humicroedit",
            )
        )
    return rows


def b22_pop909():
    from decision_index.suite.build.adapters_creative import (
        EXCLUDED_SONGS,
        VOCAB,
        parse_piece,
    )

    labels = {pc: "chord_" + str(i) for i, pc in enumerate(VOCAB)}
    criteria = {
        labels[pc]: (
            "No chord sounding at the target beat."
            if pc == "NoChord"
            else (
                "A chord pitch-class set outside this vocabulary."
                if pc == "Other"
                else {"names": VOCAB[pc], "pitch_classes": list(pc)}
            )
        )
        for pc in VOCAB
    }
    reverse = {name: pc for pc, name in VOCAB.items()}
    pub = collections.defaultdict(list)
    for r in public().rows[22]:
        pub[r["metadata"]["song_id"]].append(r["state"]["target_time_beats"])
    pool = []
    for path in sorted((RAW / "repos/pop909cl/POP909_processed").glob("*.mid")):
        if path.stem in EXCLUDED_SONGS:
            continue
        for r in parse_piece(path):
            if any(
                abs(r["target_time_beats"] - b) < 8 for b in pub.get(r["song_id"], [])
            ):
                continue
            pool.append(r)
    from decision_index.suite.build.freeze import priority

    pool.sort(key=lambda r: priority(22, r["group_id"]))
    picked, used = [], collections.defaultdict(list)
    for r in pool:
        if any(abs(r["target_time_beats"] - b) < 8 for b in used[r["song_id"]]):
            continue
        used[r["song_id"]].append(r["target_time_beats"])
        picked.append(r)
        if len(picked) >= 300:
            break
    rows = []
    for r in picked:
        rows.append(
            finish(
                {
                    "id": "POP909:proxy:" + r["group_id"],
                    "family": "POP909-chord-pitch-class",
                    "split": "unselected",
                    "state": {
                        k: r[k]
                        for k in (
                            "target_time_beats",
                            "context_notes",
                            "public_context",
                        )
                    },
                    "questions": {
                        "chord": {
                            "type": "choice",
                            "instructions": "Infer the chord sounding at the target beat from the score notes and musical context. Choose its pitch-class set; MIDI pitch classes are C=0 through B=11. Equivalent chord names are grouped.",
                            "criteria": criteria,
                        }
                    },
                    "expected": {"chord": labels[reverse[r["expected"]]]},
                    "metadata": {
                        "group_id": r["group_id"],
                        "song_id": r["song_id"],
                        "gold_class": r["expected"],
                    },
                },
                22,
                track="POP909-chord-pitch-class",
            )
        )
    return rows


def b50_habermas():
    import pyarrow.parquet as pq
    from decision_index.scoring.metrics import consensus_score

    source = RAW / "repos/habermas_machine/hm_all_candidate_comparisons.parquet"
    columns = [
        "metadata.id",
        "metadata.version",
        "metadata.status",
        "round_id",
        "iteration_index",
        "metadata.participant_id",
        "question.id",
        "question.split",
        "question.text",
        "rankings.metadata.status",
        "rankings.candidate_ids",
        "rankings.numerical_ranks",
        "candidates.metadata.id",
        "candidates.text",
        "own_opinion.metadata.id",
        "own_opinion.text",
        "other_opinions.metadata.id",
        "other_opinions.text",
    ]
    raw = pq.read_table(source, columns=columns).to_pylist()
    pub_q = {r["state"]["policy_question"] for r in public().rows[50]}
    groups = collections.defaultdict(list)
    for row in raw:
        if row["metadata.version"].startswith("EVAL") and row["question.split"] in {
            "IID_TEST",
            "OOD_TEST",
        }:
            continue
        if (
            row["metadata.status"] != "COMPLETED"
            or row["rankings.metadata.status"] != "COMPLETED"
            or row["iteration_index"] != 0
        ):
            continue
        if row["question.text"] in pub_q:
            continue
        groups[
            (row["round_id"], tuple(sorted(row["rankings.candidate_ids"] or [])))
        ].append(row)
    rows = []
    for (round_id, cands), panel in sorted(groups.items()):
        ref = panel[0]
        expected_panel = set(ref["other_opinions.metadata.id"]) | {
            ref["own_opinion.metadata.id"]
        }
        actual = [r["own_opinion.metadata.id"] for r in panel]
        if (
            set(actual) != expected_panel
            or len(actual) != len(set(actual))
            or not 2 <= len(cands) <= 255
        ):
            continue
        text_map = dict(zip(ref["candidates.metadata.id"], ref["candidates.text"]))
        if not all(
            isinstance(text_map.get(c), str) and text_map[c].strip() for c in cands
        ) or len({" ".join(text_map[c].split()) for c in cands}) != len(cands):
            continue
        sums, opinions, ok = dict.fromkeys(cands, 0), {}, True
        for r in panel:
            ranks = dict(
                zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"])
            )
            if set(ranks) != set(cands):
                ok = False
                break
            opinions[r["own_opinion.metadata.id"]] = r["own_opinion.text"]
            for c, v in ranks.items():
                sums[c] += v
        if not ok:
            continue
        order = sorted(
            cands, key=lambda c: hashlib.sha256((round_id + c).encode()).digest()
        )
        keys = {c: f"option_{i}" for i, c in enumerate(order)}
        accepted = [keys[c] for c in order if sums[c] == min(sums.values())]
        row = {
            "id": "Habermas-consensus:proxy:" + round_id,
            "family": "Habermas-consensus",
            "split": ref["question.split"],
            "state": {
                "policy_question": ref["question.text"],
                "participant_opinions": [opinions[k] for k in sorted(opinions)],
            },
            "questions": {
                "consensus": {
                    "type": "choice",
                    "instructions": "Choose the consensus statement you predict this group would rank highest on average, given their expressed opinions. Each participant has equal weight. Assess the group's preferences, rather than your own policy preference.",
                    "criteria": {keys[c]: text_map[c] for c in order},
                }
            },
            "expected": {"consensus": accepted[0]},
            "scoring": {
                "type": "human_group_rank",
                "summed_ranks": {keys[c]: sums[c] for c in order},
                "accepted": accepted,
                "panel_size": len(panel),
            },
            "metadata": {"group_id": round_id, "cohort": ref["metadata.version"]},
        }
        assert all(consensus_score(row, k)["group_preferred"] for k in accepted)
        rows.append(row)
    picked = stratified(
        rows,
        N,
        "habermas",
        stratum=lambda r: r["metadata"]["cohort"][:8],
        key=lambda r: r["id"],
    )
    return [finish(r, 50, track="Habermas-consensus") for r in picked]


def b64_newyorker():
    from decision_index.suite.build.adapters_added import NEWYORKER_INSTRUCTIONS
    import pyarrow.parquet as pq

    path = hf_file(
        "jmhessel/newyorker_caption_contest",
        "matching/validation-00000-of-00001.parquet",
    )
    t = pq.read_table(path)
    data = t.drop_columns([c for c in ("image",) if c in t.column_names]).to_pylist()
    pub_contests = {
        r.get("metadata", {}).get("contest_number") for r in public().rows[64]
    }
    pool = [r for r in data if r["contest_number"] not in pub_contests]
    rows = []
    for r in take(pool, 300, "newyorker", key=lambda r: r["instance_id"]):
        keys = [chr(65 + i) for i in range(len(r["caption_choices"]))]
        rows.append(
            finish(
                {
                    "id": f"64:proxy:{r['instance_id']}",
                    "benchmark": "New Yorker caption matching",
                    "family": "New Yorker caption matching",
                    "split": "validation",
                    "state": {
                        "scene": r["image_location"],
                        "description": r["image_description"],
                        "uncanny_description": r["image_uncanny_description"],
                        "entities": r["entities"],
                    },
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": NEWYORKER_INSTRUCTIONS,
                            "criteria": dict(zip(keys, r["caption_choices"])),
                        }
                    },
                    "expected": {"q": r["label"]},
                    "metadata": {
                        "group_id": f"64:{r['instance_id']}",
                        "contest_number": r["contest_number"],
                    },
                },
                64,
                track="matching-validation",
            )
        )
    return rows


BUILDERS = {
    25: b25_gpqa,
    30: b30_gsm8k,
    31: b31_chess,
    44: b44_cladder,
    57: b57_superg,
    28: b28_winogrande,
    12: b12_anli,
    29: b29_hellaswag,
    38: b38_acos,
    39: b39_fiqa,
    40: b40_isarcasm,
    41: b41_vast,
    42: b42_nli4ct,
    59: b59_ragtruth,
    4: b04_banking77,
    5: b05_clinc,
    36: b36_bright,
    2: b02_toolret,
    37: b37_esci,
    56: b56_phish,
    61: b61_hover,
    1: b01_bfcl,
    9: b09_home,
    20: b20_bpomp,
    21: b21_humicroedit,
    22: b22_pop909,
    50: b50_habermas,
    64: b64_newyorker,
}


def register(n):
    def deco(fn):
        BUILDERS[n] = fn
        return fn

    return deco


def main(argv=None):
    from d25.vega.eval.proxy import (
        build_s as module,
        build_s2,
    )  # noqa: F401  (registers into the importable module)

    builders = module.BUILDERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/d25/vega/proxy/build/s")
    ap.add_argument("--only", nargs="*", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = (
        json.loads(man_path.read_text()) if man_path.exists() else {"benchmarks": {}}
    )
    for n in sorted(a.only or builders):
        path = out / f"{n:02d}.jsonl.gz"
        if (
            path.exists()
            and not a.force
            and "error" not in manifest["benchmarks"].get(str(n), {"error": 1})
        ):
            continue
        t = time.time()
        try:
            rows = builders[n]()
            if not rows:
                raise ValueError("builder returned no rows")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            manifest["benchmarks"][str(n)] = {
                "name": NAMES[n],
                "error": f"{type(e).__name__}: {e}",
            }
            man_path.write_text(json.dumps(manifest, indent=1))
            continue
        assert len({r["_evaluation"]["run_id"] for r in rows}) == len(
            rows
        ), f"duplicate run ids in {n}"
        write_jsonl(path, rows)
        info = {
            "name": NAMES[n],
            "requests": len(rows),
            "cases": len({r["_evaluation"]["group_id"] for r in rows}),
            "questions": question_count(rows),
            "sha256": sha256_file(path, gunzip=True),
            "seconds": round(time.time() - t, 1),
        }
        manifest["benchmarks"][str(n)] = info
        man_path.write_text(json.dumps(manifest, indent=1))
        print(json.dumps({"event": "built", "n": n, **info}), flush=True)
    print(
        json.dumps(
            {
                "event": "done",
                "built": sorted(
                    int(k)
                    for k, v in manifest["benchmarks"].items()
                    if "error" not in v
                ),
                "failed": {
                    k: v["error"]
                    for k, v in manifest["benchmarks"].items()
                    if "error" in v
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
