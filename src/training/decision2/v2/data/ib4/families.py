"""IB4 families: pinned publisher rows -> native candidate rows (prereg ``records/ib4-prereg-2026-10-02.md`` §2).

Every family is balanced inside every declared cell (``audit_metadata["ib4"]["cell"]``): Noul families yes = no,
fixed-option Choice families one row per class. ``hash_key`` is the stable upstream key hash order uses. Rows whose
state duplicates an IB1-r3 / IB2 / IB3-r2 row, or whose declared key leaf appears in one, are removed before any
balancing (``prior``), so balance holds after the dedupe.
"""

from __future__ import annotations

import ast
import collections
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from v2.data.ib2 import families as ib2
from v2.data.sources.common import choice_options, make_row, noul_options, sha
from v2.data.textnorm import normalize

ARM = "ib4"
MAX_STATE_CHARS = 12_000
Rows = list[dict[str, Any]]

SQA2_INSTRUCTIONS = (
    "Does the passage support the proposed answer to the question, that is, does the passage show that this "
    "answer is correct?"
)
SMISH_INSTRUCTIONS = (
    "Is this text message a phishing or scam attempt, that is, one that tries to trick the recipient into "
    "handing over money, logins or personal data, or into opening a malicious link?"
)
ISARC_INSTRUCTIONS = (
    "Is the author of this post being sarcastic, that is, saying something other than what they mean in order "
    "to mock, criticise or complain?"
)
SENTFIN3_INSTRUCTIONS = (
    "What sentiment does the headline express toward the named entity itself (not toward other companies or the "
    "market in general)?"
)
SENTFIN3_OPTIONS = (
    "negative: the headline is bad news for the entity or reports it doing badly",
    "neutral: the headline mentions the entity without good or bad news for it",
    "positive: the headline is good news for the entity or reports it doing well",
)
SENTFIN3_LABELS = ("negative", "neutral", "positive")
W2C_INSTRUCTIONS = (
    "What should the assistant do next with this request, given the tools it has?"
)
W2C_OPTIONS = (
    "Call one of the available tools now",
    "Ask the user for the missing information first",
    "Tell the user the request cannot be handled with the available tools",
)
FCPICK_INSTRUCTIONS = (
    "Should the assistant call the candidate function to carry out the user's request?"
)
TEMPLATE_STRINGS = frozenset(
    [
        SQA2_INSTRUCTIONS,
        SMISH_INSTRUCTIONS,
        ISARC_INSTRUCTIONS,
        SENTFIN3_INSTRUCTIONS,
        *SENTFIN3_OPTIONS,
        W2C_INSTRUCTIONS,
        *W2C_OPTIONS,
        FCPICK_INSTRUCTIONS,
        "No",
        "Yes",
    ]
)
# The state field each construction could leak the label through (G4 hypothesis view, prereg §3).
HYPOTHESIS_FIELDS = {
    "sqa2": "answer",
    "sentfin3": "entity",
    "w2c_act": "tools",
    "fc_pick": "candidate",
}
# Key leaves (prereg §2.0): a candidate whose leaf in this field equals a leaf of an IB1-3 row is removed.
KEY_LEAVES = {"smish": "message", "w2c_act": "request", "fc_pick": "request"}
SQA2_QUESTIONS = 4000
SQA2_PER_TITLE = 20
SENTFIN3_PER_CLASS = 3000
W2C_PER_CLASS = 3000
FCPICK_CAP = 8000

Prior = Callable[[Mapping[str, Any]], bool]

# --------------------------------------------------------------------------- helpers


def order(salt: str, key: str) -> str:
    return sha(f"{salt}:{key}")


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> Rows:
    return sorted(
        rows,
        key=lambda r: (order(salt, r["audit_metadata"]["ib4"]["hash_key"]), r["id"]),
    )


def collapse(text: Any) -> str:
    return " ".join(str(text or "").split())


def words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def bucket(value: int, bounds: Sequence[int], labels: Sequence[str]) -> str:
    for bound, label in zip(bounds, labels, strict=False):
        if value <= bound:
            return label
    return labels[-1]


def row(
    *,
    family: str,
    source: str,
    task_type: str,
    group_key: str,
    key: str,
    state: dict[str, Any],
    instructions: str,
    options: list[dict[str, Any]],
    label: int,
    cell: str,
    language: str = "en",
    **extra: Any,
) -> dict[str, Any]:
    return make_row(
        arm=ARM,
        source=source,
        family=family,
        task_type=task_type,
        language=language,
        group_key=group_key,
        local_id=f"{family}:{key}",
        state=state,
        instructions=instructions,
        options=options,
        label=label,
        render_template=f"ib4_{family}_v1",
        audit={"ib4": {"hash_key": key, "cell": cell, **extra}},
    )


def noul(*, yes: bool, **kwargs: Any) -> dict[str, Any]:
    return row(
        task_type="noul", options=noul_options("en"), label=1 if yes else 0, **kwargs
    )


def resolve(rows: Rows, report: collections.Counter, prior: Prior) -> Rows:
    """Drop over-long states and IB1-3 duplicates; one row per id; conflicting copies dropped."""
    copies: dict[str, Rows] = collections.defaultdict(list)
    for item in rows:
        if sum(len(str(v)) for v in item["state"].values()) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        if prior(item):
            report["drop_prior_ib"] += 1
            continue
        copies[item["id"]].append(item)
    kept = []
    for members in copies.values():
        if len({(r["input_sha256"], r["label"]) for r in members}) > 1:
            report["drop_conflicting_duplicates"] += len(members)
            continue
        report["drop_exact_duplicates"] += len(members) - 1
        kept.append(members[0])
    return kept


def cell_of(item: Mapping[str, Any]) -> str:
    return str(item["audit_metadata"]["ib4"]["cell"])


def cell_balance(rows: Rows, labels: Sequence[int], cap: int | None, salt: str) -> Rows:
    """Equal rows per label inside every cell (hash order); a cap scales every cell down proportionally."""
    by: dict[str, dict[int, Rows]] = collections.defaultdict(
        lambda: {label: [] for label in labels}
    )
    for item in ranked(rows, salt):
        by[cell_of(item)][item["label"]].append(item)
    sizes = {c: min(len(v[label]) for label in labels) for c, v in by.items()}
    total = len(labels) * sum(sizes.values())
    if cap is not None and total > cap:
        sizes = {c: math.floor(n * cap / total) for c, n in sizes.items()}
    out: Rows = []
    for c in sorted(by):
        for label in labels:
            out += by[c][label][: sizes[c]]
    return out


def finish(
    rows: Rows,
    report: collections.Counter,
    prior: Prior,
    select: Callable[[Rows], Rows],
) -> Rows:
    rows = resolve(rows, report, prior)
    report["eligible"] = len(rows)
    chosen = select(rows)
    report["selected"] = len(chosen)
    report["selected_by_label"] = dict(
        sorted(collections.Counter(r["label"] for r in chosen).items())
    )
    return chosen


# --------------------------------------------------------------------------- sqa2: answer support (SQuAD 2.0 train)


def sqa2(
    records: Sequence[Mapping[str, Any]], report: collections.Counter, prior: Prior
) -> Rows:
    """Per answerable question: the gold answer (yes) and, from the same passage, another question's gold answer of
    the same kind that is no answer to it (no). Both answers are spans of the passage, so neither the answer alone
    nor its presence in the passage tells the label."""
    by_ctx: dict[tuple[str, str], list[tuple[str, str, list[str]]]] = (
        collections.defaultdict(list)
    )
    for record in records:
        report["read"] += 1
        texts = [collapse(t) for t in (record["answers"] or {}).get("text") or []]
        texts = [t for t in texts if t]
        if not texts:
            report["skip_unanswerable"] += 1
            continue
        by_ctx[(str(record["title"]), str(record["context"]))].append(
            (str(record["id"]), collapse(record["question"]), texts)
        )
    rows: Rows = []
    for (title, context), items in by_ctx.items():
        kinds: dict[str, list[tuple[str, str, list[str]]]] = collections.defaultdict(
            list
        )
        for item in items:
            kinds["number" if re.search(r"\d", item[2][0]) else "text"].append(item)
        for kind, members in kinds.items():
            members.sort(key=lambda m: order("ib4-sqa2-cycle-v1", m[0]))
            for i, (qid, question, golds) in enumerate(members):
                gold_norm = [normalize(g) for g in golds]
                other = None
                for step in range(1, len(members)):
                    cand = members[(i + step) % len(members)][2][0]
                    c = normalize(cand)
                    if c and all(
                        c != g and c not in g and g not in c for g in gold_norm
                    ):
                        other = cand
                        break
                if other is None:
                    report["skip_no_distractor"] += 1
                    continue
                for yes, answer in ((True, golds[0]), (False, other)):
                    rows.append(
                        noul(
                            yes=yes,
                            family="sqa2",
                            source="squad_v2_train",
                            group_key="sqa2:" + normalize(title),
                            key=f"{qid}:{'gold' if yes else 'other'}",
                            state={
                                "passage": context,
                                "question": question,
                                "answer": answer,
                            },
                            instructions=SQA2_INSTRUCTIONS,
                            cell=qid,
                            kind=kind,
                            title=title,
                        )
                    )

    def select(rs: Rows) -> Rows:
        twins = cell_balance(rs, (0, 1), None, "ib4-sqa2-bal-v1")
        by_q: dict[str, Rows] = collections.defaultdict(list)
        for item in twins:
            by_q[cell_of(item)].append(item)
        per_title: collections.Counter = collections.Counter()
        chosen: Rows = []
        for qid in sorted(by_q, key=lambda q: order("ib4-sqa2-pick-v1", q)):
            title = by_q[qid][0]["audit_metadata"]["ib4"]["title"]
            if per_title[title] >= SQA2_PER_TITLE:
                continue
            per_title[title] += 1
            chosen += by_q[qid]
            if len(chosen) >= 2 * SQA2_QUESTIONS:
                break
        return chosen

    return finish(rows, report, prior, select)


# --------------------------------------------------------------------------- smish: SMS phishing (Mendeley)

IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SMISH_LENGTHS = ((10, 20, 30, 45), ("<=10", "11-20", "21-30", "31-45", ">45"))


def smish(
    records: Sequence[Mapping[str, str]], report: collections.Counter, prior: Prior
) -> Rows:
    """yes = the publishers' ``Smishing`` class; no = ``ham``. Their ``spam`` (advertising) and the lower-case
    ``smishing`` rows (re-labelled UCI spam) are left out; balanced inside word-length cells.
    """
    rows: Rows = []
    for record in records:
        report["read"] += 1
        label = str(record.get("LABEL") or "").strip()
        text = collapse(record.get("TEXT"))
        if label not in ("Smishing", "ham"):
            report[f"skip_label_{label or 'empty'}"] += 1
            continue
        if words(text) < 3:
            report["drop_short"] += 1
            continue
        if IPV4.search(text):
            report["drop_ipv4"] += 1
            continue
        yes = label == "Smishing"
        rows.append(
            noul(
                yes=yes,
                family="smish",
                source="mendeley_sms_phishing",
                group_key="smish:" + normalize(text),
                key=sha(normalize(text))[:24],
                state={"message": text},
                instructions=SMISH_INSTRUCTIONS,
                cell=bucket(words(text), *SMISH_LENGTHS),
            )
        )
    return finish(
        rows,
        report,
        prior,
        lambda rs: cell_balance(rs, (0, 1), None, "ib4-smish-bal-v1"),
    )


# --------------------------------------------------------------------------- isarc2: sarcasm (iSarcasmEval train, task A)

ISARC_LENGTHS = ((10, 20, 30), ("<=10", "11-20", "21-30", ">30"))


def isarc2(
    files: Mapping[str, Sequence[Mapping[str, str]]],
    report: collections.Counter,
    prior: Prior,
) -> Rows:
    """Each train tweet with the author's own label (``sarcastic``); balanced inside language x length cells."""
    rows: Rows = []
    for language, records in files.items():
        for record in records:
            report["read"] += 1
            text = collapse(record.get("tweet"))
            flag = str(record.get("sarcastic") or "").strip()
            if not text or flag not in ("0", "1"):
                report["drop_shape"] += 1
                continue
            if IPV4.search(text):
                report["drop_ipv4"] += 1
                continue
            rows.append(
                noul(
                    yes=flag == "1",
                    family="isarc2",
                    source=f"isarcasmeval_train_{language}",
                    group_key="isarc2:" + normalize(text),
                    key=f"{language}:{sha(normalize(text))[:24]}",
                    state={"text": text},
                    instructions=ISARC_INSTRUCTIONS,
                    cell=f"{language}:{bucket(words(text), *ISARC_LENGTHS)}",
                    language=language,
                )
            )
    return finish(
        rows,
        report,
        prior,
        lambda rs: cell_balance(rs, (0, 1), None, "ib4-isarc2-bal-v1"),
    )


# --------------------------------------------------------------------------- sentfin3: entity-level sentiment


def sentfin3(
    records: Sequence[Mapping[str, str]], report: collections.Counter, prior: Prior
) -> Rows:
    """Every (headline, entity, decision); three fixed options; equal rows per class."""
    rows: Rows = []
    for record in records:
        report["read"] += 1
        title = collapse(record.get("Title"))
        try:
            decisions = ast.literal_eval(str(record.get("Decisions") or ""))
        except (ValueError, SyntaxError):
            report["drop_decisions_unparsed"] += 1
            continue
        if not title or not isinstance(decisions, dict):
            report["drop_shape"] += 1
            continue
        for entity, decision in decisions.items():
            entity = collapse(entity)
            if decision not in SENTFIN3_LABELS or not entity:
                report["drop_label"] += 1
                continue
            if normalize(entity) not in normalize(title):
                report["drop_entity_not_in_headline"] += 1
                continue
            rows.append(
                row(
                    family="sentfin3",
                    source="sentfin_v1",
                    task_type="choice",
                    group_key="sentfin3:" + normalize(title),
                    key=sha(normalize(title) + "\x1f" + normalize(entity))[:24],
                    state={"headline": title, "entity": entity},
                    instructions=SENTFIN3_INSTRUCTIONS,
                    options=choice_options(SENTFIN3_OPTIONS, SENTFIN3_LABELS),
                    label=SENTFIN3_LABELS.index(decision),
                    cell="all",
                    entities=len(decisions),
                )
            )
    return finish(
        rows,
        report,
        prior,
        lambda rs: cell_balance(
            rs, (0, 1, 2), 3 * SENTFIN3_PER_CLASS, "ib4-sentfin3-bal-v1"
        ),
    )


# --------------------------------------------------------------------------- w2c_act: call, ask or decline (When2Call train)

APOLOGY = re.compile(
    r"(?i)\b(sorry|apolog|unfortunately|unable|can't|cannot|can not|not able)\b"
)
ASK = re.compile(
    r"(?i)(\?|please provide|please specify|could you (please )?(provide|specify|tell|share)|i need|i'll need|i will need)"
)
TOOL_COUNTS = ((1, 2, 3, 4), ("1", "2", "3", "4", "5+"))


def w2c_category(text: str, names: set[str]) -> str | None:
    text = text.strip()
    if "<TOOLCALL>" in text:
        body = text.split("<TOOLCALL>", 1)[1].split("</TOOLCALL>", 1)[0]
        try:
            calls = json.loads(body)
        except json.JSONDecodeError:
            return None
        ok = (
            isinstance(calls, list)
            and calls
            and all(isinstance(c, dict) and c.get("name") in names for c in calls)
        )
        return "call" if ok else None
    apology, ask = bool(APOLOGY.search(text)), bool(ASK.search(text))
    if ask and not apology:
        return "ask"
    if apology and "?" not in text:
        return "unable"
    return None


def w2c_act(
    sft: Sequence[Mapping[str, Any]],
    pref: Sequence[Mapping[str, Any]],
    report: collections.Counter,
    prior: Prior,
) -> Rows:
    """The publishers' target behaviour of single-turn requests with at least one tool (SFT targets and the
    preference split's chosen responses); balanced per class inside tool-count cells."""
    rows: Rows = []
    for split, records in (("sft", sft), ("pref", pref)):
        for record in records:
            report["read"] += 1
            users = [m for m in record.get("messages") or [] if m.get("role") == "user"]
            if split == "sft":
                target = (record.get("messages") or [{}])[-1]
                text = (
                    target.get("content") if target.get("role") == "assistant" else None
                )
            else:
                chosen = record.get("chosen_response")
                text = chosen.get("content") if isinstance(chosen, dict) else chosen
            if len(users) != 1 or not isinstance(text, str):
                report["drop_turn_shape"] += 1
                continue
            try:
                tools = [
                    json.loads(t) if isinstance(t, str) else t
                    for t in record.get("tools") or []
                ]
            except json.JSONDecodeError:
                report["drop_tool_json"] += 1
                continue
            if not tools:
                report["drop_no_tools"] += 1
                continue
            names = {t.get("name") for t in tools if isinstance(t, dict)}
            category = w2c_category(text, names)
            if category is None:
                report[f"drop_uncategorised_{split}"] += 1
                continue
            request = collapse(users[0].get("content"))
            if not request or IPV4.search(request) or IPV4.search(json.dumps(tools)):
                report["drop_empty_or_ipv4"] += 1
                continue
            report[f"cat_{split}_{category}"] += 1
            rows.append(
                row(
                    family="w2c_act",
                    source="when2call_train",
                    task_type="choice",
                    group_key="w2c_act:" + normalize(request),
                    key=f"{split}:{sha(normalize(request) + json.dumps(tools, sort_keys=True))[:24]}",
                    state={
                        "tools": "\n".join(
                            json.dumps(t, ensure_ascii=False, indent=1) for t in tools
                        ),
                        "request": request,
                    },
                    instructions=W2C_INSTRUCTIONS,
                    options=choice_options(W2C_OPTIONS, ("call", "ask", "decline")),
                    label=("call", "ask", "unable").index(category),
                    cell=bucket(len(tools), *TOOL_COUNTS),
                    split=split,
                )
            )
    return finish(
        rows,
        report,
        prior,
        lambda rs: cell_balance(rs, (0, 1, 2), 3 * W2C_PER_CLASS, "ib4-w2c-bal-v1"),
    )


# --------------------------------------------------------------------------- fc_pick: per-function call decision (Glaive)


def fc_pick(
    records: Sequence[Mapping[str, Any]], report: collections.Counter, prior: Prior
) -> Rows:
    """Conversations listing at least two functions whose first assistant turn calls one: the called function as the
    candidate (yes) and another listed function (no). Balanced per candidate name, so the name alone is at chance.
    """
    rows: Rows = []
    for conv in ib2.glaive_conversations(records, report):
        if len(conv["functions"]) < 2 or not conv["call1"]:
            report["skip_not_multi_function_call"] += 1
            continue
        called = conv["call1"][0]
        others = sorted(
            (f["name"] for f in conv["functions"] if f["name"] != called),
            key=lambda n: order("ib4-fcpick-other-v1", f"{conv['key']}:{n}"),
        )
        request = collapse(conv["u1"])
        functions = ib2.dumps(conv["functions"], 1)
        if IPV4.search(request) or IPV4.search(functions):
            report["drop_ipv4"] += 1
            continue
        for yes, name in ((True, called), (False, others[0])):
            rows.append(
                noul(
                    yes=yes,
                    family="fc_pick",
                    source="glaive_function_calling_v2",
                    group_key="glaive:" + normalize(request),
                    key=f"{conv['key']}:{'called' if yes else 'other'}",
                    state={
                        "functions": functions,
                        "request": request,
                        "candidate": name,
                    },
                    instructions=FCPICK_INSTRUCTIONS,
                    cell=name,
                )
            )

    def select(rs: Rows) -> Rows:
        # One twin pair per request group first, then names balanced.
        seen: set[str] = set()
        firsts: Rows = []
        pairs: dict[str, Rows] = collections.defaultdict(list)
        for item in rs:
            pairs[item["audit_metadata"]["ib4"]["hash_key"].rsplit(":", 1)[0]].append(
                item
            )
        for conv_key in sorted(pairs, key=lambda k: order("ib4-fcpick-conv-v1", k)):
            group = pairs[conv_key][0]["group_id"]
            if group in seen:
                continue
            seen.add(group)
            firsts += pairs[conv_key]
        return cell_balance(firsts, (0, 1), FCPICK_CAP, "ib4-fcpick-bal-v1")

    return finish(rows, report, prior, select)
