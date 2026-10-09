"""Second half of the S-proxy builders (registered into ``build_s.BUILDERS``).

Builders here either read the kit's own normalized pools (rows the public freeze did not select) or
recast held-out datasets of the same competence (MuSR, SATA-Bench, CRUXEval, BBH, API-Bank, When2Call).
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import re

from d25.vega.eval.proxy.build_s import N, RAW, hf_file, parquet, public, register
from d25.vega.eval.proxy.common import (
    finish,
    norm_text,
    read_jsonl,
    rng,
    selected,
    stratified,
    take,
)

NORMALIZED = RAW.parent / "normalized"


def normalized(name):
    return read_jsonl(NORMALIZED / f"{name}.jsonl")


def _unselected(
    n, name, gid=lambda r: str(r.get("metadata", {}).get("group_id", r["id"]))
):
    pub = public().ids[n]
    return [r for r in normalized(name) if gid(r) not in pub and r["id"] not in pub]


@register(31)
def b31_chess():
    from decision_index.suite.build.freeze import priority

    pool = _unselected(31, "ChessBench-legal-move")
    pool.sort(key=lambda r: priority(31, r["metadata"]["group_id"]))
    return [finish(r, 31, track="ChessBench-legal-move") for r in pool[:N]]


@register(44)
def b44_cladder():
    pool = [
        r
        for r in _unselected(44, "CLadder")
        if not public().seen(44, r["questions"]["q1"]["instructions"])
    ]
    picked = stratified(
        pool,
        N,
        "cladder",
        stratum=lambda r: (r["provenance"].get("rung"), r["gold"]["q1"]),
        key=lambda r: r["id"],
    )
    return [finish(r, 44, track="CLadder") for r in picked]


@register(20)
def b20_bpomp():
    pool = _unselected(20, "BPoMP-original-limerick")
    picked = stratified(
        pool,
        N,
        "bpomp",
        stratum=lambda r: r["metadata"]["variant"],
        key=lambda r: r["id"],
    )
    return [finish(r, 20, track="BPoMP-original-limerick") for r in picked]


@register(22)
def b22_pop909():
    from decision_index.suite.build.freeze import priority

    pub = collections.defaultdict(list)
    for r in public().rows[22]:
        pub[r["metadata"]["song_id"]].append(r["state"]["target_time_beats"])
    pool = [
        r
        for r in normalized("POP909-chord-pitch-class")
        if not any(
            abs(r["state"]["target_time_beats"] - b) < 8
            for b in pub.get(r["metadata"]["song_id"], [])
        )
    ]
    pool.sort(key=lambda r: priority(22, r["metadata"]["group_id"]))
    picked, used = [], collections.defaultdict(list)
    for r in pool:
        song, beat = r["metadata"]["song_id"], r["state"]["target_time_beats"]
        if any(abs(beat - b) < 8 for b in used[song]):
            continue
        used[song].append(beat)
        picked.append(r)
        if len(picked) >= 300:
            break
    return [finish(r, 22, track="POP909-chord-pitch-class") for r in picked]


@register(37)
def b37_esci():
    pub_q = {r["metadata"]["query_id"] for r in public().rows[37]}
    pool = []
    for r in normalized("Amazon-ESCI"):
        if r["metadata"]["query_id"] in pub_q:
            continue
        pool.append(r)
    picked = stratified(
        pool,
        300,
        "esci",
        stratum=lambda r: (r["metadata"]["locale"], r["expected"]["answer"]),
        key=lambda r: r["id"],
    )
    return [finish(r, 37, track="Amazon-ESCI") for r in picked]


def _retrieval(n, family, k):
    side = {
        json.loads(l)["id"]: json.loads(l)
        for l in open(
            RAW.parent / "retrieval-queries" / f"{family}.jsonl", encoding="utf-8"
        )
    }
    pub = public().ids[n]
    by_q = collections.defaultdict(list)
    for r in normalized(family):
        gid = r["metadata"]["group_id"]
        if gid not in pub:
            by_q[gid].append(r)
    ok = [
        g
        for g in by_q
        if g in side
        and any(side[g]["qrels"].get(d, 0) > 0 for d in side[g]["scorable_ids"])
        and {d for r in by_q[g] for d in r["scoring"]["field_to_document"].values()}
        == set(side[g]["scorable_ids"])
    ]
    picked = stratified(
        ok, k, family, stratum=lambda g: g.split(":", 1)[0], key=lambda g: g
    )
    return [finish(r, n, track=family, group=g) for g in picked for r in by_q[g]]


@register(36)
def b36_bright():
    return _retrieval(36, "BRIGHT-retrieval", 110)


@register(2)
def b02_toolret():
    return _retrieval(2, "ToolRet-retrieval", 160)


@register(39)
def b39_fiqa():
    labels = {"0": "Positive", "1": "Neutral", "2": "Negative"}
    docs = collections.defaultdict(set)
    for split in ("train", "validation", "test"):
        import csv

        for r in csv.DictReader(
            open(hf_file("pauri32/fiqa-2018", f"{split}.csv"), encoding="utf-8")
        ):
            text, target = r["sentence"], r["target"]
            if r.get("label") not in labels or not target:
                continue
            m = re.search(re.escape(target), text, flags=re.I)
            if not m:
                continue
            docs[text].add(
                (m.start(), m.end(), text[m.start() : m.end()], labels[r["label"]])
            )
    pool = [
        (t, sorted(v))
        for t, v in docs.items()
        if not public().seen(
            39, {"document": t, "task": "FinEntity entity-given sentiment"}
        )
    ]
    rows = []
    for t, ents in take(pool, N, "fiqa", key=lambda x: x[0]):
        qs, ex, seen = {}, {}, set()
        for j, (a, b, v, lab) in enumerate(ents):
            if (a, b) in seen:
                continue
            seen.add((a, b))
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


@register(32)
def b32_opentom():
    data = json.load(
        open(hf_file("SeacowX/OpenToM", "opentom_long.json"), encoding="utf-8")
    )
    pool = []
    flat = []
    for i, e in enumerate(data):
        qs = e.get("question")
        qs = qs if isinstance(qs, list) else [qs]
        for j, q in enumerate(qs):
            if isinstance(q, dict) and e.get("narrative"):
                flat.append((f"{i}.{j}", e, q))
    for i, e, q in flat:
        info = (
            json.loads(e["plot_info"])
            if isinstance(e.get("plot_info"), str)
            else e.get("plot_info") or {}
        )
        qtext, ans, qtype = (
            q.get("question"),
            str(q.get("answer", "")).strip(),
            q.get("type", ""),
        )
        if not qtext:
            continue
        if qtype.startswith("location") and "precisely" in qtext:
            opts = [info.get("original_place"), info.get("move_to_place")]
        elif qtype.startswith("location"):
            opts = ["Yes", "No"]
        elif qtype.startswith("multihop") and "accessib" in qtext:
            opts = ["more accessible", "equally accessible", "less accessible"]
        elif qtype.startswith("multihop"):
            opts = ["more full", "equally full", "less full"]
        elif qtype.startswith("attitude"):
            opts = ["positive", "neutral", "negative"]
        else:
            continue
        opts = [o for o in opts if o]
        norm = {o.lower().strip(" ."): o for o in opts}
        gold = norm.get(ans.lower().strip(" ."))
        if gold is None or len(set(opts)) < 2:
            continue
        pool.append((i, e, qtext, opts, gold, qtype.split("-")[0]))
    picked = stratified(
        pool, N, "opentom", stratum=lambda x: x[5], key=lambda x: f"{x[0]}:{x[2]}"
    )
    rows = []
    for i, e, qtext, opts, gold, qtype in picked:
        r = rng("opentom", f"{i}:{qtext}")
        order = list(opts)
        r.shuffle(order)
        row = {
            "id": f"MuSR:opentom:{i}:{hashlib.sha256(qtext.encode()).hexdigest()[:10]}",
            "benchmark": "MuSR",
            "family": "MuSR",
            "split": "proxy",
            "state": {},
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": e["narrative"] + "\n\n" + qtext,
                    "criteria": {chr(65 + k): o for k, o in enumerate(order)},
                }
            },
            "expected": {"q1": chr(65 + order.index(gold))},
            "metadata": {"group_id": f"opentom:{i}:{qtext}", "qtype": qtype},
        }
        rows.append(finish(row, 32, track="MuSR"))
    return rows


@register(33)
def b33_sata():
    data = parquet(
        hf_file("aps/super_glue", "multirc/validation-00000-of-00001.parquet")
    )
    groups = collections.defaultdict(list)
    for r in data:
        idx = json.loads(r["idx"]) if isinstance(r["idx"], str) else r["idx"]
        groups[(idx["paragraph"], idx["question"])].append((idx["answer"], r))
    pool = []
    for key, items in groups.items():
        items.sort()
        if not 3 <= len(items) <= 16:
            continue
        para, question = items[0][1]["paragraph"], items[0][1]["question"]
        if public().seen(
            33,
            {
                "paragraph": para,
                "question": question,
                "options": [x[1]["answer"] for x in items],
            },
        ):
            continue
        pool.append(
            (
                key,
                para,
                question,
                [(x[1]["answer"], int(x[1]["label"]) == 1) for x in items],
            )
        )
    rows = []
    for key, para, question, choices in take(
        pool, N, "sata-multirc", key=lambda x: f"{x[0]}"
    ):
        choices = list(choices)
        rng("SATA", key).shuffle(choices)
        qs, ex = {}, {}
        for j, (text, label) in enumerate(choices):
            qs[f"option_{j}"] = {
                "type": "choice",
                "instructions": f"Question: {question}\nDoes this candidate correctly answer the question?\nCandidate: {text}",
                "criteria": {"no": "No", "yes": "Yes"},
            }
            ex[f"option_{j}"] = "yes" if label else "no"
        rows.append(
            finish(
                {
                    "id": f"SATA-Bench:multirc:{key[0]}:{key[1]}",
                    "family": "SATA-Bench",
                    "split": "proxy",
                    "state": {
                        "paragraph": para,
                        "question": question,
                        "options": [c[0] for c in choices],
                    },
                    "questions": qs,
                    "expected": ex,
                    "metadata": {
                        "group_id": f"multirc:{key[0]}:{key[1]}",
                        "source_subset": "multirc-validation",
                    },
                },
                33,
                track="SATA-Bench",
            )
        )
    return rows


def _mutations(value, r):
    out = []
    if isinstance(value, bool):
        out = [not value]
    elif isinstance(value, int):
        out = [value + 1, value - 1, value * 2, -value, value + 2, 0 if value else 1]
    elif isinstance(value, float):
        out = [value + 1.0, value / 2, -value, round(value * 2, 6)]
    elif isinstance(value, str):
        out = [
            value[::-1],
            value.upper(),
            value[:-1],
            value + value[-1:] if value else "a",
            value.lower(),
            value.strip() + " ",
            "",
        ]
    elif isinstance(value, (list, tuple)):
        v = list(value)
        cands = [v[::-1], v[:-1], v + v[-1:], sorted(v, key=repr), v[1:], []]
        out = [type(value)(c) for c in cands]
    elif value is None:
        out = [0, [], ""]
    elif isinstance(value, dict):
        items = list(value.items())
        out = [dict(items[:-1]), dict(items[::-1][:-1]) if items else {"a": 1}]
    out = [
        x
        for x in out
        if x != value and not (isinstance(x, type(value)) is False and x == value)
    ]
    uniq = []
    for x in out:
        if all(repr(x) != repr(y) for y in uniq) and repr(x) != repr(value):
            uniq.append(x)
    r.shuffle(uniq)
    return uniq


@register(43)
def b43_crux():
    data = parquet(
        hf_file("livecodebench/execution-v2", "data/test-00000-of-00001.parquet")
    )
    rows = []
    pool = []
    for d in data:
        try:
            gold = ast.literal_eval(d["output"])
        except (ValueError, SyntaxError):
            continue
        if public().seen(
            43,
            {
                "code": d["code"],
                "input": d["input"],
                "task": "CRUXEval output selection",
            },
        ):
            continue
        pool.append((d, gold))
    for d, gold in take(
        pool, N, "lcb-exec", key=lambda x: f"{x[0]['id']}:{x[0]['question_id']}"
    ):
        r = rng("CRUX", f"{d['id']}:{d['question_id']}")
        k = r.choice([2, 2, 3, 3, 3, 4])
        vals = [gold] + _mutations(gold, r)[: k - 1]
        if len(vals) < 2:
            continue
        r.shuffle(vals)
        choices = [repr(v) for v in vals]
        correct = next(i for i, v in enumerate(vals) if repr(v) == repr(gold))
        call = d["input"]
        rows.append(
            finish(
                {
                    "id": f"CRUXEval-output-choice:lcb:{d['question_id']}:{d['id']}",
                    "family": "CRUXEval-output-choice",
                    "split": "proxy",
                    "state": {
                        "code": d["code"],
                        "input": call,
                        "task": "CRUXEval output selection",
                    },
                    "questions": {
                        "answer": {
                            "type": "choice",
                            "instructions": "Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values.",
                            "criteria": {
                                f"option_{i}": v for i, v in enumerate(choices)
                            },
                        }
                    },
                    "expected": {"answer": f"option_{correct}"},
                    "metadata": {
                        "group_id": f"lcb:{d['question_id']}:{d['id']}",
                        "candidate_count": len(choices),
                    },
                },
                43,
                track="CRUXEval-output-choice",
            )
        )
    return rows


BBH_TASKS = [
    "boolean_expressions",
    "causal_judgment",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies_syllogisms_negation",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction",
    "movie_recommendation",
    "navigate",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects",
    "web_of_lies",
]
BBEH_MC = {
    "disambiguation_qa": None,
    "geometric_shapes": None,
    "hyperbaton": None,
    "movie_recommendation": None,
    "shuffled_objects": None,
    "causal_understanding": ["Yes", "No"],
    "boardgame_qa": ["proved", "disproved", "unknown"],
}
OPT = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _bbh_row(sid, text, criteria, gold, track, source):
    return finish(
        {
            "id": f"58:proxy:{sid}",
            "benchmark": "BBH fixed-option tasks",
            "family": "BBH fixed-option tasks",
            "split": "proxy",
            "state": text,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Which option is the correct answer?",
                    "criteria": criteria,
                }
            },
            "expected": {"q": gold},
            "metadata": {"group_id": f"58:{sid}", "source": source},
        },
        58,
        track=track,
    )


@register(58)
def b58_bbh():
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files("tasksource/bigbench", repo_type="dataset")
    prefixes = {t[:160] for t in public().texts[58]}
    bb = []
    for task in BBH_TASKS:
        for f in [
            x for x in files if x.startswith(task + "/") and x.endswith(".parquet")
        ]:
            for i, r in enumerate(parquet(hf_file("tasksource/bigbench", f))):
                mct = (
                    json.loads(r["multiple_choice_targets"])
                    if isinstance(r["multiple_choice_targets"], str)
                    else list(r["multiple_choice_targets"])
                )
                scores = (
                    json.loads(r["multiple_choice_scores"])
                    if isinstance(r["multiple_choice_scores"], str)
                    else list(r["multiple_choice_scores"])
                )
                if (
                    not 2 <= len(mct) <= 26
                    or sum(scores) != 1
                    or len(set(mct)) != len(mct)
                ):
                    continue
                text = r["inputs"].strip()
                if public().seen(58, text) or norm_text(text)[:160] in prefixes:
                    continue
                gold = mct[scores.index(1)]
                bb.append((task, f"{f}:{i}", text, mct, gold))
    picked = stratified(
        bb, 150, "bbh-bigbench", stratum=lambda x: x[0], key=lambda x: x[1]
    )
    rows = []
    for task, sid, text, mct, gold in picked:
        if len(mct) <= 3 and all(len(m) < 40 for m in mct):
            criteria = {m: m for m in mct}
            g = gold
            state = text
        else:
            letters = [f"({chr(65 + k)})" for k in range(len(mct))]
            criteria = dict(zip(letters, mct))
            g = letters[mct.index(gold)]
            state = (
                text
                if "Options:" in text
                else text.rstrip()
                + "\nOptions:\n"
                + "\n".join(f"{k} {v}" for k, v in criteria.items())
            )
        rows.append(
            _bbh_row(
                f"bigbench:{task}:{sid}", state, criteria, g, task, "bigbench-non-bbh"
            )
        )
    ext = []
    for task, fixed in BBEH_MC.items():
        for i, r in enumerate(
            parquet(
                hf_file(
                    "hubert233/BigBenchExtraHard", f"data/{task}-00000-of-00001.parquet"
                )
            )
        ):
            text, target = r["input"], str(r["target"]).strip()
            if fixed:
                if target not in fixed:
                    continue
                criteria, g = {x: x for x in fixed}, target
            else:
                opts = [OPT.match(line.strip()) for line in text.splitlines()]
                opts = [(m.group(1), m.group(2)) for m in opts if m]
                letters = [k for k, _ in opts]
                if len(opts) < 2 or letters != [chr(65 + k) for k in range(len(opts))]:
                    continue
                t = target.strip("() .")
                if t not in letters:
                    continue
                criteria, g = {f"({k})": v for k, v in opts}, f"({t})"
            ext.append((task, f"{task}:{i}", text, criteria, g))
    for task, sid, text, criteria, g in stratified(
        ext, 150, "bbh-bbeh", stratum=lambda x: x[0], key=lambda x: x[1]
    ):
        rows.append(_bbh_row(f"bbeh:{sid}", text, criteria, g, "bbeh-" + task, "bbeh"))
    return rows


@register(3)
def b03_apibank():
    data = json.load(
        open(
            hf_file("liminghao1630/API-Bank", "training-data/lv1-api-train.json"),
            encoding="utf-8",
        )
    )
    call = re.compile(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    items, catalog = [], {}
    for i, d in enumerate(data):
        if not selected("apibank-lv1-train", i, 25, 1):
            continue
        m = call.search(d.get("output", ""))
        if not m:
            continue
        apis, dialogue = [], []
        for line in d.get("input", "").splitlines():
            s = line.strip()
            if s.startswith("{"):
                try:
                    a = json.loads(s)
                except json.JSONDecodeError:
                    continue
                name = a.get("apiCode") or a.get("name")
                if name:
                    tool = {
                        "name": name,
                        "description": a.get("description", ""),
                        "input_parameters": a.get(
                            "parameters", a.get("input_parameters", {})
                        ),
                        "output_parameters": a.get(
                            "response", a.get("output_parameters", {})
                        ),
                    }
                    apis.append(tool)
                    catalog.setdefault(name, tool)
            elif (
                s.startswith(("User:", "AI:", "API-Request:"))
                and "Generate API Request" not in s
            ):
                role, _, text = s.partition(":")
                dialogue.append({"role": role, "text": text.strip()})
        gold = m.group(1)
        if gold in {a["name"] for a in apis} and dialogue:
            items.append((i, apis, dialogue, gold))
    names = sorted(catalog)
    rows = []
    for i, apis, dialogue, gold in take(items, N, "apibank", key=lambda x: str(x[0])):
        r = rng("APIBANK", i)
        own = [a["name"] for a in apis]
        others = [x for x in names if x not in own]
        r.shuffle(others)
        pool = own + others[: max(0, 53 - len(own))]
        r.shuffle(pool)
        labels = {f"option_{k}": v for k, v in enumerate(pool)}
        goldkey = next(k for k, v in labels.items() if v == gold)
        rows.append(
            finish(
                {
                    "id": f"API-Bank-tool-selection:lv1-train:{i}",
                    "family": "API-Bank-tool-selection",
                    "split": "train-slice",
                    "state": {
                        "dialogue": dialogue,
                        "available_tools": [catalog[x] for x in pool],
                        "task": "API-Bank current API tool selection",
                    },
                    "questions": {
                        "tool": {
                            "type": "choice",
                            "instructions": "Select the single API tool to invoke for the dialogue prefix in state. Use the published catalog in state. Do not infer arguments or use the current/future API result.",
                            "criteria": labels,
                        }
                    },
                    "expected": {"tool": goldkey},
                    "metadata": {"group_id": f"apibank:lv1:{i}", "gold_tool": gold},
                },
                3,
                track="API-Bank-tool-selection",
            )
        )
    return rows


def _w2c_kind(text):
    t = text.lower()
    if "<toolcall>" in t:
        return "tool_call"
    if any(
        x in t
        for x in (
            "unable",
            "can't",
            "cannot",
            "not able",
            "don't have the ability",
            "apologies",
        )
    ):
        return "cannot_answer"
    if "?" in t:
        return "request_for_info"
    return "direct"


@register(62)
def b62_when2call():
    """The llm-judge test file overlaps the public MCQ test, so items come from a 1/10 slice of train_pref:
    the preferred and the rejected assistant turn as two options (same competence: answer, call, ask or decline).
    """
    from decision_index.suite.build.adapters_added import WHEN2CALL_INSTRUCTIONS

    pool = []
    for i, r in enumerate(
        read_jsonl(hf_file("nvidia/When2Call", "train/when2call_train_pref.jsonl"))
    ):
        msgs = (
            json.loads(r["messages"])
            if isinstance(r["messages"], str)
            else r["messages"]
        )
        if len(msgs) != 1 or msgs[0].get("role") != "user":
            continue
        question = msgs[0]["content"]
        if not selected("when2call", question, 10, 1):
            continue
        tools = [
            json.loads(t) if isinstance(t, str) else t
            for t in (
                json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            )
        ]
        chosen = (
            json.loads(r["chosen_response"])
            if isinstance(r["chosen_response"], str)
            else r["chosen_response"]
        )
        rejected = (
            json.loads(r["rejected_response"])
            if isinstance(r["rejected_response"], str)
            else r["rejected_response"]
        )
        if public().seen(62, {"tools": tools, "question": question}):
            continue
        pool.append((i, tools, question, chosen["content"], rejected["content"]))
    rows = []
    for i, tools, question, good, bad in stratified(
        pool, 300, "when2call", stratum=lambda x: _w2c_kind(x[3]), key=lambda x: x[2]
    ):
        opts = [good, bad]
        rng("W2C", question).shuffle(opts)
        rows.append(
            finish(
                {
                    "id": f"62:train-pref:{i}",
                    "benchmark": "When2Call MCQ",
                    "family": "When2Call MCQ",
                    "split": "train-slice",
                    "state": {"tools": tools, "question": question},
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": WHEN2CALL_INSTRUCTIONS,
                            "criteria": {"A": opts[0], "B": opts[1]},
                        }
                    },
                    "expected": {"q": "A" if opts[0] == good else "B"},
                    "metadata": {"group_id": f"62:pref:{i}", "kind": _w2c_kind(good)},
                },
                62,
                track=_w2c_kind(good),
            )
        )
    return rows
