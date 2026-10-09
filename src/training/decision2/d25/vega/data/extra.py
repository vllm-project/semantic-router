"""M2 converters: knowledge multiple choice and in-distribution TRAIN splits of index benchmark sources.

    python -m d25.vega.data.extra knowledge|indist --holdouts holdouts.json

Rows go to DATA_ROOT/rows/{knowledge,indist}/<source>.jsonl.gz with a report.json. Only train splits are
read. Rows that fall in a holdouts.json slice are dropped here (and carry meta.hf_ids / meta.slice_keys so
the mixture builder re-checks them against newer holdouts files). In-distribution rows are rendered like
the Decision Index kit renders the benchmark (state / instructions / option keys), mixed with a natural
rendering so the model does not rely on one template.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from d25.vega.data.holdouts import Holdouts
from d25.vega.data.util import (
    DATA_ROOT,
    choice_question,
    make_row,
    noul_question,
    rank,
    read_jsonl,
    rng_for,
    write_json,
    write_jsonl,
)

RAW = DATA_ROOT / "raw/m2"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REFERENTIAL = re.compile(
    r"\b(all|none|both|neither)\b.{0,20}\b(above|these|of the|options?|a and b)\b|^[a-e] (and|&) [a-e]$|^(all|none) of",
    re.I,
)
LETTER_COMBO = re.compile(r"^\s*(\(?[a-e1-5]\)?[\s,&/]*(and\s+)?)+\s*$", re.I)


def usable_distractor(text: str) -> bool:
    """Pool options for distractor expansion: no letter combinations ("acd", "a and c"), no references, >= 2 chars."""
    t = (text or "").strip()
    return len(t) >= 2 and not LETTER_COMBO.match(t) and not REFERENTIAL.search(t)


GENERIC_MCQ = [
    "Which option is the correct answer?",
    "Choose the correct answer.",
    "Which supplied option best answers the question?",
    "Choose the criterion that best answers the question.",
    "Select the best answer to the question.",
]


def revision_of(name: str) -> str:
    report = json.loads((DATA_ROOT / "raw/fetch-m2.json").read_text())
    return (
        report["hf"][name]["revision"][:8]
        if name in report["hf"]
        else report["urls"][name]["sha256"][:8]
    )


def parquet_rows(pattern: str) -> list[dict[str, Any]]:
    files = sorted(RAW.glob(pattern))
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


class Emitter:
    def __init__(
        self,
        name: str,
        kind: str,
        dataset: str,
        licence: str,
        hf_ids: list[str],
        holdouts: Holdouts,
    ):
        self.name, self.kind, self.dataset, self.licence, self.hf_ids = (
            name,
            kind,
            dataset,
            licence,
            hf_ids,
        )
        self.holdouts = holdouts
        self.rows: list[dict[str, Any]] = []
        self.stats: Counter = Counter()

    def add(
        self,
        row_id: str,
        family: str,
        state: Any,
        question: dict[str, Any],
        target: list[float],
        group: str,
        slice_keys: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        meta = {
            "dataset": self.dataset,
            "licence": self.licence,
            "licence_use": "commercial",
            "source_split": "train",
            "group": f"{self.name}:{group}",
            "hf_ids": self.hf_ids,
            "orig_kind": question["type"],
            "soft": False,
        }
        if slice_keys:
            meta["slice_keys"] = slice_keys
        meta.update(extra or {})
        try:
            row = make_row(
                row_id=f"{self.name}:{row_id}",
                source=f"{self.kind}:{self.name}",
                family=family,
                state=state,
                question=question,
                target=target,
                meta=meta,
            )
        except ValueError as error:
            self.stats[f"drop_invalid:{str(error)[:30]}"] += 1
            return
        reason = self.holdouts.check(row)
        if reason:
            self.stats[f"holdout:{reason}"] += 1
            return
        self.stats["out"] += 1
        self.rows.append(row)

    def write(self) -> dict[str, Any]:
        out = DATA_ROOT / f"rows/{self.kind}/{self.name}.jsonl.gz"
        write_jsonl(out, self.rows)
        return {
            "file": str(out),
            "dataset": self.dataset,
            "licence": self.licence,
            **dict(self.stats),
        }


def mcq_question(
    rng: random.Random, stem: str, options: list[str], gold: int, title: str
) -> tuple[Any, dict[str, Any], list[float]]:
    """Render one MCQ in a randomly chosen style: suite MMLU-Pro, suite GPQA, tasksource, named state."""
    target = [0.0] * len(options)
    target[gold] = 1.0
    style = rng.random()
    if style < 0.35:
        return (
            stem,
            choice_question(
                "Which option is the correct answer?",
                {LETTERS[i]: o for i, o in enumerate(options)},
            ),
            target,
        )
    if style < 0.55:
        return (
            "",
            choice_question(stem, {LETTERS[i]: o for i, o in enumerate(options)}),
            target,
        )
    if style < 0.8 and len(set(options)) == len(options):
        return (
            stem,
            choice_question(rng.choice(GENERIC_MCQ), dict.fromkeys(options)),
            target,
        )
    state = {"question": stem, "task": f"{title} multiple-choice question"}
    return (
        state,
        choice_question(
            "Choose the correct answer to the question in state.",
            {f"option_{i}": o for i, o in enumerate(options)},
        ),
        target,
    )


def shuffled(
    rng: random.Random, options: list[str], gold: int
) -> tuple[list[str], int]:
    if any(REFERENTIAL.search(o) for o in options):
        return options, gold
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(gold)


def expand(
    rng: random.Random, options: list[str], gold: int, pool: list[str], k: int = 10
) -> tuple[list[str], int] | None:
    """MMLU-Pro-like 10 options: add distractors that are options of other questions of the same subject."""
    if (
        any(REFERENTIAL.search(o) or LETTER_COMBO.match(o) for o in options)
        or len(pool) < 50
    ):
        return None
    seen = {o.strip().casefold() for o in options}
    extra: list[str] = []
    for _ in range(60):
        cand = pool[rng.randrange(len(pool))]
        key = cand.strip().casefold()
        if key in seen or not usable_distractor(cand):
            continue
        seen.add(key)
        extra.append(cand)
        if len(options) + len(extra) == k:
            break
    if len(options) + len(extra) != k:
        return None
    combined = options + extra
    return shuffled(rng, combined, gold)


# ------------------------------------------------------------------------------------------- knowledge


def knowledge(holdouts: Holdouts, seed: int) -> dict[str, Any]:
    reports = {}

    def emitter(name: str, repo: str, licence: str) -> Emitter:
        return Emitter(
            name, "knowledge", f"{repo}@{revision_of(name)}", licence, [repo], holdouts
        )

    # MedMCQA (Apache-2.0): 182k train; sample 40k, 30% of the sample expanded to 10 options within a subject.
    em = emitter("medmcqa", "openlifescienceai/medmcqa", "apache-2.0")
    rows = parquet_rows("medmcqa/data/train-*.parquet")
    pools: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for key in ("opa", "opb", "opc", "opd"):
            pools[r["subject_name"] or "?"].append(r[key] or "")
    rows = sorted(rows, key=lambda r: rank(r["id"], seed))[:40000]
    for r in rows:
        rng = rng_for(r["id"], seed)
        options = [r["opa"], r["opb"], r["opc"], r["opd"]]
        if (
            not all(o and o.strip() for o in options)
            or not isinstance(r["cop"], int)
            or len(set(options)) < 4
        ):
            em.stats["drop_bad_options"] += 1
            continue
        options, gold = shuffled(rng, options, r["cop"])
        extra = {}
        if rng.random() < 0.3:
            expanded = expand(rng, options, gold, pools[r["subject_name"] or "?"])
            if expanded:
                options, gold = expanded
                extra = {"aug": "distractor_expansion"}
        state, question, target = mcq_question(
            rng, r["question"], options, gold, "Medical entrance exam"
        )
        em.add(
            r["id"],
            f"knowledge/medmcqa/{r['subject_name']}",
            state,
            question,
            target,
            r["id"],
            extra=extra,
        )
    reports["medmcqa"] = em.write()

    # AQuA-RAT (Apache-2.0): algebra word problems, 5 options; sample 25k.
    em = emitter("aqua_rat", "deepmind/aqua_rat", "apache-2.0")
    rows = parquet_rows("aqua_rat/raw/train-*.parquet")
    for i, r in enumerate(
        sorted(rows, key=lambda r: rank(r["question"], seed))[:25000]
    ):
        rng = rng_for(r["question"], seed)
        options = [re.sub(r"^\s*[A-E]\s*\)\s*", "", o).strip() for o in r["options"]]
        if len(options) != 5 or r["correct"] not in "ABCDE" or len(set(options)) != 5:
            em.stats["drop_bad_options"] += 1
            continue
        options, gold = shuffled(rng, options, "ABCDE".index(r["correct"]))
        state, question, target = mcq_question(
            rng, r["question"], options, gold, "Algebra word problem"
        )
        key = rank(r["question"], 0)[:16]
        em.add(f"{i}:{key}", "knowledge/aqua_rat", state, question, target, key)
    reports["aqua_rat"] = em.write()

    # MathQA (Apache-2.0): 29.8k train; take 20k.
    em = emitter("math_qa", "allenai/math_qa", "apache-2.0")
    rows = parquet_rows("math_qa/**/*.parquet")
    pattern = re.compile(r"([a-e]) \) (.*?)(?= , [a-e] \) |$)")
    for n, r in enumerate(sorted(rows, key=lambda r: rank(r["Problem"], seed))[:20000]):
        rng = rng_for(r["Problem"], seed)
        parsed = pattern.findall(r["options"].strip().strip("[]'\""))
        options = [o.strip() for _, o in parsed]
        letters = [l for l, _ in parsed]
        if len(options) != 5 or r["correct"] not in letters or len(set(options)) != 5:
            em.stats["drop_bad_options"] += 1
            continue
        options, gold = shuffled(rng, options, letters.index(r["correct"]))
        state, question, target = mcq_question(
            rng, r["Problem"], options, gold, "Math word problem"
        )
        key = rank(r["Problem"], 0)[:16]
        em.add(
            f"{n}:{key}",
            f"knowledge/math_qa/{r['category']}",
            state,
            question,
            target,
            key,
        )
    reports["math_qa"] = em.write()

    # CommonsenseQA (MIT), QASC (CC BY 4.0), ARC train (CC BY-SA 4.0): all train rows.
    for name, repo, licence, pattern_, title in (
        (
            "commonsense_qa",
            "tau/commonsense_qa",
            "mit",
            "commonsense_qa/data/train-*.parquet",
            "Commonsense question",
        ),
        (
            "qasc",
            "allenai/qasc",
            "cc-by-4.0",
            "qasc/data/train-*.parquet",
            "Grade-school science question",
        ),
        (
            "ai2_arc",
            "allenai/ai2_arc",
            "cc-by-sa-4.0",
            "ai2_arc/*/train-*.parquet",
            "Grade-school science exam",
        ),
    ):
        em = emitter(name, repo, licence)
        for r in parquet_rows(pattern_):
            rng = rng_for(r["id"], seed)
            labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if r["answerKey"] not in labels or len(set(texts)) != len(texts):
                em.stats["drop_bad_options"] += 1
                continue
            options, gold = shuffled(rng, texts, labels.index(r["answerKey"]))
            stem = r["question"]
            if name == "qasc" and rng.random() < 0.3:
                stem = f"Fact 1: {r['fact1']}\nFact 2: {r['fact2']}\nQuestion: {r['question']}"
            state, question, target = mcq_question(rng, stem, options, gold, title)
            em.add(r["id"], f"knowledge/{name}", state, question, target, r["id"])
        reports[name] = em.write()

    # BoolQ (CC BY-SA 3.0): passage yes/no questions as noul.
    em = emitter("boolq", "google/boolq", "cc-by-sa-3.0")
    for n, r in enumerate(parquet_rows("boolq/data/train-*.parquet")):
        key = rank(r["question"] + r["passage"][:200], 0)[:16]
        question = r["question"].strip()
        question = (
            question[:1].upper()
            + question[1:]
            + ("" if question.endswith("?") else "?")
        )
        em.add(
            f"{n}:{key}",
            "knowledge/boolq",
            r["passage"],
            noul_question(question),
            [0.0, 1.0] if r["answer"] else [1.0, 0.0],
            key,
        )
    reports["boolq"] = em.write()
    return reports


# ------------------------------------------------------------------------------------------- in-distribution


def num(value: Decimal) -> str:
    return format(value.normalize(), "f")


def gsm8k_groups(golds: dict[int, Decimal], seed: int) -> dict[int, list[Decimal]]:
    """Answer groups of 10 problems with distinct golds of similar size (like the kit's answer-groups-v1)."""
    order = sorted(golds, key=lambda i: (golds[i], rank(str(i), seed)))
    groups: dict[int, list[Decimal]] = {}
    pending = list(order)
    while len(pending) >= 10:
        group, rest, values = [], [], set()
        for i in pending:
            if len(group) < 10 and golds[i] not in values:
                group.append(i)
                values.add(golds[i])
            else:
                rest.append(i)
        if len(group) < 10:
            break
        for i in group:
            groups[i] = sorted(values)
        pending = rest
    return groups


def indist(holdouts: Holdouts, seed: int) -> dict[str, Any]:
    reports = {}

    # BANKING77 train (CC BY 4.0), kit style: empty state, query in instructions, option_i = label names.
    categories = json.loads((RAW / "urls/categories.json").read_text())
    em = Emitter(
        "banking77",
        "indist",
        f"github PolyAI-LDN/task-specific-datasets@master ({revision_of('banking77_train')})",
        "cc-by-4.0",
        ["PolyAI/banking77"],
        holdouts,
    )
    with open(RAW / "urls/train.csv", newline="", encoding="utf-8") as stream:
        for i, r in enumerate(csv.DictReader(stream)):
            text, label = r["text"], r["category"]
            if label not in categories:
                em.stats["drop_label"] += 1
                continue
            rng = rng_for(text, seed)
            gold = categories.index(label)
            target = [0.0] * len(categories)
            target[gold] = 1.0
            if rng.random() < 0.7:
                state, question = {}, choice_question(
                    f"Classify the banking intent of this user request:\n{text}",
                    {f"option_{j}": c for j, c in enumerate(categories)},
                )
            else:
                state, question = text, choice_question(
                    "Which banking intent does this customer message express?",
                    {c.replace("_", " "): None for c in categories},
                )
            em.add(
                str(i),
                "indist/banking77",
                state,
                question,
                target,
                str(i),
                slice_keys={"banking77": text},
            )
    reports["banking77"] = em.write()

    # CLINC150 train incl. OOS train (CC BY 3.0); validation splits are held out by ws-proxy.
    readme = (RAW / "clinc_oos/README.md").read_text()
    block = readme[readme.find("config_name: plus") :]
    names = re.findall(r"'(\d+)': (\S+)", block[: block.find("splits")])
    names = [n for _, n in sorted(names, key=lambda x: int(x[0]))]
    labels = sorted(
        {("out of scope" if n == "oos" else n.replace("_", " ")) for n in names}
    )
    em = Emitter(
        "clinc150",
        "indist",
        f"clinc/clinc_oos@{revision_of('clinc_oos')}:plus",
        "cc-by-3.0",
        ["clinc/clinc_oos"],
        holdouts,
    )
    for i, r in enumerate(parquet_rows("clinc_oos/plus/train-*.parquet")):
        name = names[r["intent"]]
        label = "out of scope" if name == "oos" else name.replace("_", " ")
        rng = rng_for(r["text"], seed)
        target = [0.0] * len(labels)
        target[labels.index(label)] = 1.0
        if rng.random() < 0.7:
            state, question = {}, choice_question(
                f"Classify the intent of this user request, or choose out of scope if none applies:\n{r['text']}",
                {f"option_{j}": c for j, c in enumerate(labels)},
            )
        else:
            state, question = r["text"], choice_question(
                "What does the user want? Choose out of scope if no listed intent applies.",
                dict.fromkeys(labels),
            )
        em.add(str(i), "indist/clinc150", state, question, target, str(i))
    reports["clinc150"] = em.write()

    # HellaSwag train (MIT); 1/64 slice held out (key ctx). Sample 25k.
    em = Emitter(
        "hellaswag",
        "indist",
        f"Rowan/hellaswag@{revision_of('hellaswag')}",
        "mit",
        ["Rowan/hellaswag"],
        holdouts,
    )
    rows = parquet_rows("hellaswag/data/train-*.parquet")
    for n, r in enumerate(
        sorted(
            rows, key=lambda r: rank(f"{r['source_id']}:{r['ind']}:{r['ctx']}", seed)
        )[:26000]
    ):
        rng = rng_for(f"{r['source_id']}:{r['ind']}", seed)
        endings = [e.strip() for e in r["endings"]]
        if len(endings) != 4 or not str(r["label"]).isdigit():
            em.stats["drop_bad"] += 1
            continue
        gold = int(r["label"])
        target = [0.0] * 4
        target[gold] = 1.0
        if rng.random() < 0.7:
            state, question = {}, choice_question(
                f"Which continuation is most plausible?\n{r['ctx']}",
                dict(zip("ABCD", endings)),
            )
        else:
            state = {"activity": r["activity_label"], "context": r["ctx"]}
            question = choice_question(
                "Which ending best continues the context?", dict(zip("ABCD", endings))
            )
        em.add(
            f"{n}:{r['ind']}",
            "indist/hellaswag",
            state,
            question,
            target,
            str(r["source_id"]),
            slice_keys={"hellaswag": r["ctx"]},
        )
    reports["hellaswag"] = em.write()

    # WinoGrande train_xl; 1/32 slice held out (key sentence). Take 22k.
    em = Emitter(
        "winogrande",
        "indist",
        f"allenai/winogrande@{revision_of('winogrande')}:winogrande_xl",
        "apache-2.0 (allenai/winogrande GitHub LICENSE)",
        ["allenai/winogrande"],
        holdouts,
    )
    rows = parquet_rows("winogrande/winogrande_xl/train-*.parquet")
    for n, r in enumerate(
        sorted(rows, key=lambda r: rank(r["sentence"], seed))[:22000]
    ):
        rng = rng_for(r["sentence"], seed)
        if r["answer"] not in ("1", "2"):
            em.stats["drop_bad"] += 1
            continue
        target = [1.0, 0.0] if r["answer"] == "1" else [0.0, 1.0]
        if rng.random() < 0.7:
            state, question = {}, choice_question(
                f"Which option correctly fills the blank?\n{r['sentence']}",
                {"A": r["option1"], "B": r["option2"]},
            )
        else:
            state, question = r["sentence"], choice_question(
                "Which option correctly fills the blank (_) in the sentence?",
                (
                    {r["option1"]: None, r["option2"]: None}
                    if r["option1"] != r["option2"]
                    else {"A": r["option1"], "B": r["option2"]}
                ),
            )
        key = rank(r["sentence"], 0)[:16]
        em.add(
            f"{n}:{key}",
            "indist/winogrande",
            state,
            question,
            target,
            key,
            slice_keys={"winogrande": r["sentence"]},
        )
    reports["winogrande"] = em.write()

    # GSM8K train (MIT) as 4- and 10-choice numeric selection with answer-group distractors; 1/16 held out.
    train = [json.loads(line) for line in open(DATA_ROOT / "raw/gsm8k/train.jsonl")]
    golds: dict[int, Decimal] = {}
    for i, r in enumerate(train):
        try:
            golds[i] = Decimal(r["answer"].split("####")[-1].strip().replace(",", ""))
        except InvalidOperation:
            continue
    groups = gsm8k_groups(golds, seed)
    em = Emitter(
        "gsm8k_mcq",
        "indist",
        "openai/gsm8k@main (train)",
        "mit",
        ["openai/gsm8k"],
        holdouts,
    )
    for i, values in groups.items():
        q = train[i]["question"]
        rng = rng_for(q, seed)
        ring = values
        pos = ring.index(golds[i])
        for count in (4, 10):
            if count == 4:
                shift = rng.randrange(4)
                opts = [ring[(pos - shift + t) % len(ring)] for t in range(4)]
            else:
                opts = list(ring)
            texts = [num(v) for v in opts]
            rng.shuffle(texts)
            target = [0.0] * count
            target[texts.index(num(golds[i]))] = 1.0
            state = {
                "question": q,
                "task": f"GSM8K deterministic {count}-choice numeric selection",
            }
            question = choice_question(
                "Choose the numeric answer to the problem in state. Do not provide reasoning. "
                "This is a named multiple-choice adaptation of GSM8K.",
                {f"option_{j}": t for j, t in enumerate(texts)},
            )
            em.add(
                f"{i}:{count}",
                "indist/gsm8k_mcq",
                state,
                question,
                target,
                str(i),
                slice_keys={"gsm8k": q},
            )
    reports["gsm8k_mcq"] = em.write()

    # When2Call train (CC BY 4.0); slice key = user question text (no uuid in the train files); 1/10 held out.
    em = Emitter(
        "when2call",
        "indist",
        f"nvidia/When2Call@{revision_of('when2call')}",
        "cc-by-4.0",
        ["nvidia/When2Call"],
        holdouts,
    )
    kinds = {
        "direct": "Answer the question directly without calling a tool.",
        "tool_call": "Call one of the available tools with the needed arguments.",
        "request_info": "Ask the user for the missing information needed to call a tool.",
        "cannot_answer": "Say that it cannot answer or perform this request with the available tools.",
    }

    def kind_of(text: str) -> str:
        t = text.strip()
        if "<TOOLCALL>" in t:
            return "tool_call"
        low = t.lower()
        if (
            re.search(
                r"\b(unable|can't|cannot|not able|don't have (the )?(ability|capability|access)|sorry)\b",
                low,
            )
            and "?" not in t[-3:]
        ):
            return "cannot_answer"
        if t.endswith("?") or re.search(
            r"\b(could you|can you) (please )?(provide|tell|specify|share)\b", low
        ):
            return "request_info"
        return "direct"

    for path, mode in (
        (RAW / "when2call/train/when2call_train_sft.jsonl", "sft"),
        (RAW / "when2call/train/when2call_train_pref.jsonl", "pref"),
    ):
        for i, r in enumerate(read_jsonl(path)):
            user = next(
                (m["content"] for m in r["messages"] if m["role"] == "user"), None
            )
            if not user:
                continue
            tools = [
                json.loads(t) if isinstance(t, str) and t.strip().startswith("{") else t
                for t in r.get("tools", [])
            ]
            state = {"tools": tools, "question": user}
            key = f"{mode}:{i}"
            if mode == "sft":
                reply = next(
                    (m["content"] for m in r["messages"] if m["role"] == "assistant"),
                    "",
                )
                kind = kind_of(reply)
                order = list(kinds)
                target = [1.0 if k == kind else 0.0 for k in order]
                em.add(
                    key,
                    "indist/when2call_kind",
                    state,
                    choice_question(
                        "Which kind of response should the assistant give to the user's question, given the available tools?",
                        {k: kinds[k] for k in order},
                    ),
                    target,
                    key,
                    slice_keys={"when2call": user},
                )
            else:
                rng = rng_for(user + key, seed)
                pair = [
                    r["chosen_response"]["content"],
                    r["rejected_response"]["content"],
                ]
                gold = 0
                if rng.random() < 0.5:
                    pair, gold = pair[::-1], 1
                target = [1.0 - gold, float(gold)]
                em.add(
                    key,
                    "indist/when2call_pref",
                    state,
                    choice_question(
                        "Which response should the assistant give to the user's question, given the available tools?",
                        {"A": pair[0], "B": pair[1]},
                    ),
                    target,
                    key,
                    slice_keys={"when2call": user},
                )
    reports["when2call"] = em.write()

    # New Yorker caption matching (CC BY 4.0), train split; contests of the matching validation fold are excluded.
    em = Emitter(
        "newyorker",
        "indist",
        f"jmhessel/newyorker_caption_contest@{revision_of('newyorker')}:matching",
        "cc-by-4.0",
        ["jmhessel/newyorker_caption_contest"],
        holdouts,
    )
    import pyarrow.parquet as pq2

    val_contests = set()
    for path in sorted(RAW.glob("newyorker/matching/validation-*.parquet")):
        val_contests |= set(
            pq2.read_table(path, columns=["contest_number"])
            .column("contest_number")
            .to_pylist()
        )
    cols = [
        "contest_number",
        "image_location",
        "image_description",
        "image_uncanny_description",
        "entities",
        "caption_choices",
        "label",
        "instance_id",
    ]
    for path in sorted(RAW.glob("newyorker/matching/train-*.parquet")):
        for r in pq2.read_table(path, columns=cols).to_pylist():
            if r["contest_number"] in val_contests:
                em.stats["drop_validation_contest"] += 1
                continue
            choices = list(r["caption_choices"])
            if len(choices) != 5 or r["label"] not in "ABCDE":
                em.stats["drop_bad"] += 1
                continue
            target = [0.0] * 5
            target["ABCDE".index(r["label"])] = 1.0
            state = {
                "scene": r["image_location"],
                "description": r["image_description"],
                "uncanny_description": r["image_uncanny_description"],
                "entities": list(r["entities"] or []),
            }
            em.add(
                r["instance_id"],
                "indist/newyorker",
                state,
                choice_question(
                    "Which caption was written for this cartoon?",
                    dict(zip("ABCDE", choices)),
                ),
                target,
                str(r["contest_number"]),
            )
    reports["newyorker"] = em.write()

    # Glaive function calling v2 (Apache-2.0) -> BFCL-style tool selection, one yes/no question per candidate tool.
    em = Emitter(
        "glaive_bfcl",
        "indist",
        f"glaiveai/glaive-function-calling-v2@{revision_of('glaive_fc')}",
        "apache-2.0",
        ["glaiveai/glaive-function-calling-v2"],
        holdouts,
    )
    data = json.loads((RAW / "glaive_fc/glaive-function-calling-v2.json").read_text())
    all_functions: list[dict[str, Any]] = []
    parsed_rows = []
    decoder = json.JSONDecoder()
    for i, r in enumerate(data):
        system = r.get("system", "")
        start = system.find("{")
        functions = []
        while 0 <= start < len(system):
            try:
                obj, end = decoder.raw_decode(system, start)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict) and "name" in obj:
                functions.append(obj)
            nxt = system.find("{", end)
            start = nxt
        chat = r.get("chat", "")
        m_user = re.search(r"USER:\s*(.*?)\s*(?=ASSISTANT:)", chat, re.S)
        if not functions or not m_user:
            continue
        user = m_user.group(1).strip()
        m_asst = re.search(
            r"ASSISTANT:\s*(.*?)\s*(?:<\|endoftext\|>|$)", chat[m_user.end() :], re.S
        )
        reply = m_asst.group(1) if m_asst else ""
        called = None
        if reply.startswith("<functioncall>"):
            nm = re.search(r'"name"\s*:\s*"([^"]+)"', reply)
            called = nm.group(1) if nm else None
            if called is None:
                continue
        parsed_rows.append((i, user, functions, called))
        all_functions.extend(functions)
    instructions = (
        "Given the complete user conversation and the published function schemas in state, mark each candidate "
        "tool yes if it should be invoked to answer the request, or no otherwise."
    )
    for i, user, functions, called in sorted(
        parsed_rows, key=lambda x: rank(str(x[0]), seed)
    )[:15000]:
        rng = rng_for(str(i), seed)
        candidates = list({f["name"]: f for f in functions}.values())
        names = {f["name"] for f in candidates}
        for _ in range(rng.choice([0, 0, 1, 2, 3])):
            extra = all_functions[rng.randrange(len(all_functions))]
            if extra["name"] not in names:
                candidates.append(extra)
                names.add(extra["name"])
        rng.shuffle(candidates)
        state = {
            "conversation": [[{"role": "user", "content": user}]],
            "functions": candidates,
            "task": "BFCL tool-name selection",
        }
        for position, f in enumerate(candidates):
            yes = f["name"] == called
            em.add(
                f"{i}:{position}:{f['name']}",
                "indist/glaive_bfcl",
                state,
                choice_question(
                    instructions,
                    {"yes": "Invoke this tool.", "no": "Do not invoke this tool."},
                ),
                [1.0, 0.0] if yes else [0.0, 1.0],
                str(i),
            )
    reports["glaive_bfcl"] = em.write()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["knowledge", "indist", "all"])
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    holdouts = Holdouts(args.holdouts)
    for what in ["knowledge", "indist"] if args.what == "all" else [args.what]:
        reports = (
            knowledge(holdouts, args.seed)
            if what == "knowledge"
            else indist(holdouts, args.seed)
        )
        write_json(
            DATA_ROOT / f"rows/{what}/report.json",
            {"holdouts": holdouts.report(), "sources": reports},
        )
        for name, rep in reports.items():
            print(
                what,
                name,
                {k: v for k, v in rep.items() if k not in ("file",)},
                flush=True,
            )


if __name__ == "__main__":
    main()
