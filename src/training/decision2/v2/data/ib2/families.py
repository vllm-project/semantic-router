"""IB2 families: pinned publisher rows -> native candidate rows (prereg ``records/ib2-prereg-2026-10-01.md`` §2).

Every family function returns validated TRAIN rows with its cap and balance applied. Balancing fields live in
``audit_metadata["ib2"]``: ``hash_key`` (the stable upstream key hash order uses), ``cell`` (the balance cell) and
``gold_position`` (rotated families).
"""

from __future__ import annotations

import collections
import html
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from v2.data.sources.common import make_row, noul_options, rotate, sha
from v2.data.textnorm import normalize

ARM = "ib2"
MAX_STATE_CHARS = 12_000
PER_GROUP = 2
Rows = list[dict[str, Any]]

FC_REL_INSTRUCTIONS = "Can one of the available functions carry out the user's request?"
FC_SEL_INSTRUCTIONS = (
    "Which function should the assistant call to carry out this request?"
)
FC_ARGS_INSTRUCTIONS = "Which arguments should the assistant pass to this function to carry out the request?"
FC_READY_INSTRUCTIONS = (
    "Does the user's request already contain every required argument for this function?"
)
FC_CAP = 8000
# Amendment 2: `fc_sel` and `fc_args` (run c2) failed G4 and are replaced by frequency-matched redesigns.
FC_FAMILIES = ("fc_rel", "fc_sel2", "fc_args2", "fc_ready")
YTSPAM_INSTRUCTIONS = (
    "Is this YouTube comment spam, that is advertising, self-promotion, a scam, or a link or request to "
    "visit, subscribe or follow that is unrelated to the video?"
)
ARGQ_INSTRUCTIONS = "Does the argument support or oppose the statement?"
ARGQ_OPTIONS = (
    {"key": "support", "description": "The argument supports the statement"},
    {"key": "oppose", "description": "The argument opposes the statement"},
)
ARGQ_CAP = 6000
HOVER_INSTRUCTIONS = "Do these Wikipedia passages support every part of the claim?"
HOVER_HOPS = (2, 3)
HOVER_CAP = 6000
HOVER_YES, HOVER_NO = 3, 2
QASC_INSTRUCTIONS = "Choose the best answer to the science question."
QASC_CAP = 6000
ARC_INSTRUCTIONS = "Choose the correct answer to the science exam question."
GSM_INSTRUCTIONS = "Is the stated answer to this math word problem correct?"
CNLI_INSTRUCTIONS = "Does this contract entail the following statement? {hypothesis}"
# ContractNLI's 17 fixed hypotheses (train.json "labels"); the builder checks the file against them.
CNLI_HYPOTHESES = {
    "nda-1": "All Confidential Information shall be expressly identified by the Disclosing Party.",
    "nda-10": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
    "nda-11": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
    "nda-12": "Receiving Party may independently develop information similar to Confidential Information.",
    "nda-13": "Receiving Party may acquire information similar to Confidential Information from a third party.",
    "nda-15": "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "nda-16": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "nda-17": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "nda-18": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    "nda-19": "Some obligations of Agreement may survive termination of Agreement.",
    "nda-2": "Confidential Information shall only include technical information.",
    "nda-20": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
    "nda-3": "Confidential Information may include verbally conveyed information.",
    "nda-4": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "nda-5": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "nda-7": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
    "nda-8": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
}

# Fixed instruction and option strings (G0 reports their hits separately; they are not data).
TEMPLATE_STRINGS = frozenset(
    [
        FC_REL_INSTRUCTIONS,
        FC_SEL_INSTRUCTIONS,
        FC_ARGS_INSTRUCTIONS,
        FC_READY_INSTRUCTIONS,
        YTSPAM_INSTRUCTIONS,
        ARGQ_INSTRUCTIONS,
        HOVER_INSTRUCTIONS,
        QASC_INSTRUCTIONS,
        ARC_INSTRUCTIONS,
        GSM_INSTRUCTIONS,
        "No",
        "Yes",
    ]
    + [o["description"] for o in ARGQ_OPTIONS]
    + [CNLI_INSTRUCTIONS.format(hypothesis=h) for h in CNLI_HYPOTHESES.values()]
)


# --------------------------------------------------------------------------- helpers


def order(salt: str, key: str) -> str:
    return sha(f"{salt}:{key}")


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> Rows:
    return sorted(
        rows, key=lambda r: order(salt, r["audit_metadata"]["ib2"]["hash_key"])
    )


def state_chars(state: Mapping[str, Any]) -> int:
    return sum(len(str(value)) for value in state.values())


def row(
    *,
    source: str,
    family: str,
    task_type: str,
    group_key: str,
    key: str,
    state: dict[str, Any],
    instructions: str,
    options: Sequence[Mapping[str, Any]],
    label: int,
    template: str,
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
        options=[dict(o) for o in options],
        label=label,
        render_template=template,
        audit={"ib2": {"hash_key": key, "cell": cell, **extra}},
    )


def noul(*, yes: bool, **kwargs: Any) -> dict[str, Any]:
    return row(
        task_type="noul",
        options=noul_options("en"),
        label=1 if yes else 0,
        cell=kwargs.pop("cell", "yes" if yes else "no"),
        **kwargs,
    )


def rotated(
    *, descriptions: Sequence[str], gold: int, key: str, **kwargs: Any
) -> dict[str, Any]:
    """Choice row whose options are rotated by a per-row seed (gold position near-uniform)."""
    options = [
        {"key": f"o{i + 1}", "description": text} for i, text in enumerate(descriptions)
    ]
    shifted, label = rotate(options, gold, f"ib2-rot-v1:{kwargs['family']}:{key}")
    shifted = [
        {"key": f"o{i + 1}", "description": o["description"]}
        for i, o in enumerate(shifted)
    ]
    return row(
        task_type="choice",
        options=shifted,
        label=label,
        key=key,
        gold_position=label,
        **kwargs,
    )


def resolve(rows: Rows, report: collections.Counter) -> Rows:
    """Drop over-long states; one row per id; ids whose copies disagree on input or label are dropped."""
    copies: dict[str, Rows] = collections.defaultdict(list)
    for item in rows:
        if state_chars(item["state"]) > MAX_STATE_CHARS:
            report["drop_length"] += 1
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


def per_cell_cap(rows: Rows, cell: Any, cap: int, salt: str) -> Rows:
    kept, taken = [], collections.Counter()
    for item in ranked(rows, salt):
        name = cell(item)
        if taken[name] < cap:
            kept.append(item)
            taken[name] += 1
    return kept


def per_group(rows: Rows, salt: str) -> Rows:
    return per_cell_cap(rows, lambda r: r["group_id"], PER_GROUP, salt)


def balance_labels(rows: Rows, cap: int | None, salt: str) -> Rows:
    """Equal rows per label (hash order), at most ``cap`` in total."""
    by: dict[int, Rows] = collections.defaultdict(list)
    for item in ranked(rows, salt):
        by[item["label"]].append(item)
    if not by or len(by) < len(rows[0]["options"]):
        return []
    size = min(len(members) for members in by.values())
    if cap is not None:
        size = min(size, cap // len(rows[0]["options"]))
    return [item for label in sorted(by) for item in by[label][:size]]


def ratio_labels(rows: Rows, yes: int, no: int, cap: int | None, salt: str) -> Rows:
    """Noul rows with yes : no = ``yes : no`` (hash order), at most ``cap`` in total."""
    pos = [r for r in ranked(rows, salt) if r["label"] == 1]
    neg = [r for r in ranked(rows, salt) if r["label"] == 0]
    units = min(len(pos) // yes, len(neg) // no)
    if cap is not None:
        units = min(units, cap // (yes + no))
    return pos[: units * yes] + neg[: units * no]


def cap_total(rows: Rows, cap: int, salt: str) -> Rows:
    return ranked(rows, salt)[:cap]


# --------------------------------------------------------------------------- Glaive function calling

GLAIVE_MARKER = "Use them if required -"
TURN_RE = re.compile(r"(?:^|\n)(USER|ASSISTANT|FUNCTION RESPONSE):[ \t]?")
CALL_RE = re.compile(
    r'^<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}\s*$',
    re.S,
)
APOLOGY = ("sorry", "apologize", "unfortunately")
INABILITY = (
    "capabilit",
    "can't assist",
    "cannot assist",
    "unable to",
    "not able to",
    "don't have the ability",
    "do not have the ability",
    "can't perform",
    "cannot perform",
    "not equipped",
    "limited to",
)
REQUEST = (
    "please provide",
    "could you provide",
    "can you provide",
    "could you please",
    "can you please tell",
    "could you tell",
    "may i know",
    "i need the",
    "i need some",
    "i need more",
)
GENERIC_TOKENS = frozenset(
    "get set create calculate check find search generate convert add update delete list send book make "
    "fetch retrieve analyze track schedule play translate recommend suggest info information details "
    "data".split()
)
NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split()
    )
}
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORD_RE = re.compile(r"\w+")


def glaive_functions(system: str) -> list[dict[str, Any]] | None:
    if GLAIVE_MARKER not in system:
        return None
    text = system.split(GLAIVE_MARKER, 1)[1]
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = 0
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text):
            break
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(obj, dict)
            or not isinstance(obj.get("name"), str)
            or not isinstance(obj.get("description"), str)
            or not obj["name"].strip()
        ):
            return None
        out.append(obj)
    names = [f["name"] for f in out]
    return out if out and len(set(names)) == len(names) else None


def glaive_turns(chat: str) -> list[tuple[str, str]]:
    parts = TURN_RE.split(chat)
    return [
        (parts[k], parts[k + 1].replace("<|endoftext|>", "").strip())
        for k in range(1, len(parts) - 1, 2)
    ]


def glaive_call(text: str, names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if not text.startswith("<functioncall>"):
        return None
    match = CALL_RE.match(text)
    try:
        if match:
            name, args = match.group(1), json.loads(match.group(2))
        else:
            obj = json.loads(text[len("<functioncall>") :].strip())
            if not isinstance(obj, dict):
                return None
            name, args = obj.get("name"), obj.get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
    except json.JSONDecodeError:
        return None
    if name not in names or not isinstance(args, dict):
        return None
    return name, args


def numbers_in(text: str) -> set[float]:
    out = set()
    for match in NUM_RE.finditer(text):
        try:
            out.add(float(match.group().replace(",", "")))
        except ValueError:
            pass
    for word in WORD_RE.findall(text.casefold()):
        if word in NUMBER_WORDS:
            out.add(float(NUMBER_WORDS[word]))
    return out


class Request:
    """A request text prepared for grounding checks."""

    def __init__(self, text: str) -> None:
        self.norm = normalize(text)
        self.words = set(WORD_RE.findall(self.norm))
        self.numbers = numbers_in(text)

    def grounded(self, value: Any) -> bool:
        if isinstance(value, bool) or value is None:
            return False
        if isinstance(value, (int, float)):
            return float(value) in self.numbers
        if isinstance(value, str):
            v = normalize(value)
            if not v:
                return False
            return v in self.words if len(v) <= 2 else v in self.norm
        if isinstance(value, list):
            return bool(value) and all(self.grounded(v) for v in value)
        return False


def marker(text: str, markers: Sequence[str]) -> bool:
    low = text.casefold().replace("\u2019", "'")
    return any(m in low for m in markers)


def is_refusal(text: str) -> bool:
    return (
        "<functioncall>" not in text
        and marker(text, APOLOGY)
        and marker(text, INABILITY)
        and not marker(text, REQUEST)
    )


def is_request(text: str) -> bool:
    return "<functioncall>" not in text and marker(text, REQUEST)


def name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {
        t
        for t in re.split(r"[^A-Za-z0-9]+", spaced.casefold())
        if t and t not in GENERIC_TOKENS
    }


def required(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    params = schema.get("parameters") if isinstance(schema, Mapping) else None
    names = params.get("required") if isinstance(params, Mapping) else None
    if isinstance(names, list) and names and all(isinstance(n, str) for n in names):
        return list(names)
    return list(args)


def dumps(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def glaive_conversations(
    records: Sequence[Mapping[str, Any]], report: collections.Counter
) -> list[dict[str, Any]]:
    """Parsed conversations with the facts every ``fc_*`` family needs."""
    out = []
    for record in records:
        report["read"] += 1
        functions = glaive_functions(str(record.get("system") or ""))
        if not functions:
            report["drop_no_functions"] += 1
            continue
        turns = glaive_turns(str(record.get("chat") or ""))
        if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT":
            report["drop_turn_shape"] += 1
            continue
        names = {f["name"] for f in functions}
        u1, a1 = turns[0][1], turns[1][1]
        u2 = turns[2][1] if len(turns) > 2 and turns[2][0] == "USER" else None
        a2 = turns[3][1] if len(turns) > 3 and turns[3][0] == "ASSISTANT" else None
        if not u1:
            report["drop_turn_shape"] += 1
            continue
        out.append(
            {
                "key": sha(
                    str(record.get("system")) + "\x1f" + str(record.get("chat"))
                )[:24],
                "functions": functions,
                "by_name": {f["name"]: f for f in functions},
                "u1": u1,
                "u2": u2,
                "call1": glaive_call(a1, names),
                "refusal": is_refusal(a1),
                "ask": is_request(a1),
                "call2": glaive_call(a2, names) if a2 and u2 else None,
                "calls": [
                    c
                    for role, text in turns
                    if role == "ASSISTANT" and (c := glaive_call(text, names))
                ],
            }
        )
    return out


STOPWORDS = frozenset(
    "the and for with from that this your into about given which what when where there their them they "
    "have will would could should based using user users".split()
)


def content_words(text: str) -> set[str]:
    return {
        w
        for w in WORD_RE.findall(normalize(text))
        if len(w) >= 4 and w not in STOPWORDS
    }


def unrelated(
    function: Mapping[str, Any], gold: Mapping[str, Any], request: str
) -> bool:
    """Amendment 1: no name token and no description word shared with the called function or the request."""
    words = set(WORD_RE.findall(normalize(request)))
    if function["name"] == gold["name"]:
        return False
    if name_tokens(function["name"]) & (name_tokens(gold["name"]) | words):
        return False
    described = content_words(function["description"])
    return not described & (content_words(gold["description"]) | words)


def fc_row(conv: Mapping[str, Any], family: str, **kwargs: Any) -> dict[str, Any]:
    return dict(
        source="glaive_fc_v2",
        family=family,
        group_key="glaive:" + normalize(conv["u1"]),
        template=f"ib2_{family}_v1",
        **kwargs,
    )


def glaive(
    records: Sequence[Mapping[str, Any]], reports: Mapping[str, collections.Counter]
) -> dict[str, Rows]:
    convs = glaive_conversations(records, reports["fc_rel"])
    # Pools over every parsed conversation: the most frequent description per function name and every value
    # of every (function, parameter) seen in a call.
    descriptions: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    values: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(dict)
    for conv in convs:
        for f in conv["functions"]:
            descriptions[f["name"]][" ".join(f["description"].split())] += 1
        for name, args in conv["calls"]:
            for param, value in args.items():
                values[(name, param)].setdefault(dumps(value), value)
    pool = sorted(descriptions, key=lambda n: order("ib2-fcsel-pool-v1", n))
    describe = {
        n: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for n, c in descriptions.items()
    }
    lists = [(c["key"], c["functions"]) for c in convs if c["call1"] or c["refusal"]]
    lists.sort(key=lambda item: (order("ib2-fcrel-lists-v1", item[0]), item[0]))
    # Amendment 2: frequency-matched pools, one entry per call instance (not per distinct name or value).
    called = sorted(
        ((c["key"], c["call1"][0]) for c in convs if c["call1"]),
        key=lambda item: (order("ib2-fcsel2-pool-v1", item[0]), item[0]),
    )
    instances: dict[tuple[str, str], list[tuple[str, Any]]] = collections.defaultdict(
        list
    )
    for conv in convs:
        for index, (name, args) in enumerate(conv["calls"]):
            for param, value in args.items():
                instances[(name, param)].append((f"{conv['key']}:{index}", value))
    for entries in instances.values():
        entries.sort(key=lambda item: (order("ib2-fcargs2-pool-v1", item[0]), item[0]))
    out: dict[str, Rows] = {name: [] for name in FC_FAMILIES}
    makers = {
        "fc_rel": fc_rel,
        "fc_sel2": fc_sel2,
        "fc_args2": fc_args2,
        "fc_ready": fc_ready,
    }
    for conv in convs:
        for family, make in makers.items():
            made = make(
                conv,
                reports[family],
                pool=pool,
                describe=describe,
                values=values,
                lists=lists,
                called=called,
                instances=instances,
            )
            for item in made if isinstance(made, list) else [made]:
                if item is not None:
                    out[family].append(item)
    for family in FC_FAMILIES:
        report = reports[family]
        rows = per_group(resolve(out[family], report), f"ib2-{family}-group-v1")
        if family == "fc_args2":
            rows = per_cell_cap(
                rows,
                lambda r: r["audit_metadata"]["ib2"]["cell"],
                PER_GROUP,
                "ib2-fc_args2-gold-v1",
            )
        report["eligible"] = len(rows)
        if family in ("fc_rel", "fc_ready"):
            rows = balance_labels(rows, FC_CAP, f"ib2-{family}-v1")
        else:
            rows = cap_total(rows, FC_CAP, f"ib2-{family}-v1")
        report["selected"] = len(rows)
        out[family] = rows
    return out


def fc_rel(
    conv: Mapping[str, Any],
    report: collections.Counter,
    *,
    lists: Sequence[tuple[str, list[dict[str, Any]]]],
    **_: Any,
) -> list[dict[str, Any]]:
    """Own list: yes iff A1 calls, no iff A1 refuses; amendment 1 adds a constructed no row per call."""

    def make(key: str, functions: list[dict[str, Any]], yes: bool, kind: str) -> dict:
        return noul(
            yes=yes,
            key=key,
            state={"functions": dumps(functions, 1), "request": conv["u1"]},
            instructions=FC_REL_INSTRUCTIONS,
            kind=kind,
            **fc_row(conv, "fc_rel"),
        )

    if conv["refusal"]:
        return [make(conv["key"], conv["functions"], False, "refusal")]
    if not conv["call1"]:
        report["drop_neither_call_nor_refusal"] += 1
        return []
    rows = [make(conv["key"], conv["functions"], True, "call")]
    gold = conv["by_name"][conv["call1"][0]]
    start = int(order("ib2-fcrel-start-v1", conv["key"]), 16) % len(lists)
    for step in range(min(len(lists), 200)):
        other, functions = lists[(start + step) % len(lists)]
        if other != conv["key"] and all(
            unrelated(f, gold, conv["u1"]) for f in functions
        ):
            rows.append(make(conv["key"] + ":x", functions, False, "constructed"))
            return rows
    report["no_constructed_negative"] += 1
    return rows


def fc_sel(
    conv: Mapping[str, Any],
    report: collections.Counter,
    *,
    pool: Sequence[str],
    describe: Mapping[str, str],
    **_: Any,
) -> dict | None:
    if not conv["call1"]:
        report["drop_no_first_call"] += 1
        return None
    gold = conv["call1"][0]
    blocked = name_tokens(gold) | set(WORD_RE.findall(normalize(conv["u1"])))
    start = int(order("ib2-fcsel-start-v1", conv["key"]), 16) % len(pool)
    picked: list[str] = []
    for step in range(len(pool)):
        name = pool[(start + step) % len(pool)]
        if name in conv["by_name"] or name_tokens(name) & blocked:
            continue
        if not name_tokens(name) or describe[name] in {describe[p] for p in picked}:
            continue
        picked.append(name)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        report["drop_no_distractors"] += 1
        return None
    gold_text = f"{gold}: {' '.join(conv['by_name'][gold]['description'].split())}"
    texts = [gold_text] + [f"{n}: {describe[n]}" for n in picked]
    if len({normalize(t) for t in texts}) != 4:
        report["drop_duplicate_options"] += 1
        return None
    return rotated(
        descriptions=texts,
        gold=0,
        key=conv["key"],
        state={"request": conv["u1"]},
        instructions=FC_SEL_INSTRUCTIONS,
        cell=gold,
        **fc_row(conv, "fc_sel"),
    )


def swap_values(
    args: Mapping[str, Any],
    name: str,
    request: Request,
    values: Mapping[tuple[str, str], Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]] | None:
    """Two copies of ``args`` with one value each replaced by an ungrounded value of the same parameter."""
    params = sorted(args, key=lambda p: order("ib2-fcargs-param-v1", key + p))
    copies: list[dict[str, Any]] = []
    used: set[str] = set()

    def alternative(param: str, skip: set[str]) -> Any:
        gold = args[param]
        pool = [
            v
            for k, v in sorted(
                values.get((name, param), {}).items(),
                key=lambda kv: order("ib2-fcargs-value-v1", key + param + kv[0]),
            )
            if k != dumps(gold) and k not in skip
        ]
        for value in pool:
            if type(value) is type(gold) and not request.grounded(value):
                return value
        return None

    for param in params:
        value = alternative(param, set())
        if value is not None:
            copies.append({**args, param: value})
            used.add(param)
            break
    if not copies:
        return None
    for param in params:
        if param in used:
            continue
        value = alternative(param, set())
        if value is not None:
            copies.append({**args, param: value})
            break
    if len(copies) == 1:
        param = next(iter(used))
        value = alternative(param, {dumps(copies[0][param])})
        if value is None:
            return None
        copies.append({**args, param: value})
    return copies


def fc_args(
    conv: Mapping[str, Any],
    report: collections.Counter,
    *,
    values: Mapping[tuple[str, str], Mapping[str, Any]],
    **_: Any,
) -> dict | None:
    if not conv["call1"]:
        report["drop_no_first_call"] += 1
        return None
    name, args = conv["call1"]
    request = Request(conv["u1"])
    if not args or not all(request.grounded(v) for v in args.values()):
        report["drop_not_grounded"] += 1
        return None
    copies = swap_values(args, name, request, values, conv["key"])
    if copies is None:
        report["drop_no_alternative_value"] += 1
        return None
    texts = [dumps(args)] + [dumps(c) for c in copies]
    if len(set(texts)) != 3:
        report["drop_duplicate_options"] += 1
        return None
    return rotated(
        descriptions=texts,
        gold=0,
        key=conv["key"],
        state={"function": dumps(conv["by_name"][name], 1), "request": conv["u1"]},
        instructions=FC_ARGS_INSTRUCTIONS,
        cell=name,
        **fc_row(conv, "fc_args"),
    )


def fc_sel2(
    conv: Mapping[str, Any],
    report: collections.Counter,
    *,
    called: Sequence[tuple[str, str]],
    describe: Mapping[str, str],
    **_: Any,
) -> dict | None:
    """Amendment 2: ``fc_sel`` with distractors drawn per call instance and canonical descriptions for all options."""
    if not conv["call1"]:
        report["drop_no_first_call"] += 1
        return None
    gold = conv["call1"][0]
    blocked = name_tokens(gold) | set(WORD_RE.findall(normalize(conv["u1"])))
    start = int(order("ib2-fcsel2-start-v1", conv["key"]), 16) % len(called)
    picked: list[str] = []
    for step in range(len(called)):
        name = called[(start + step) % len(called)][1]
        if name in picked or name in conv["by_name"] or name_tokens(name) & blocked:
            continue
        if not name_tokens(name) or describe[name] in {describe[p] for p in picked}:
            continue
        picked.append(name)
        if len(picked) == 3:
            break
    texts = [f"{n}: {describe[n]}" for n in [gold] + picked]
    if len(picked) < 3 or len({normalize(t) for t in texts}) != 4:
        report["drop_no_distractors"] += 1
        return None
    return rotated(
        descriptions=texts,
        gold=0,
        key=conv["key"],
        state={"request": conv["u1"]},
        instructions=FC_SEL_INSTRUCTIONS,
        cell=gold,
        **fc_row(conv, "fc_sel2"),
    )


def fc_args2(
    conv: Mapping[str, Any],
    report: collections.Counter,
    *,
    instances: Mapping[tuple[str, str], Sequence[tuple[str, Any]]],
    **_: Any,
) -> dict | None:
    """Amendment 2: ``fc_args`` with replacement values drawn per call instance (frequency-matched)."""
    if not conv["call1"]:
        report["drop_no_first_call"] += 1
        return None
    name, args = conv["call1"]
    request = Request(conv["u1"])
    if not args or not all(request.grounded(v) for v in args.values()):
        report["drop_not_grounded"] += 1
        return None
    params = sorted(args, key=lambda p: order("ib2-fcargs2-param-v1", conv["key"] + p))

    def alternative(param: str, skip: set[str]) -> Any:
        entries = instances.get((name, param), [])
        if not entries:
            return None
        start = int(order("ib2-fcargs2-start-v1", conv["key"] + param), 16)
        gold = args[param]
        for step in range(len(entries)):
            value = entries[(start + step) % len(entries)][1]
            text = dumps(value)
            if (
                type(value) is type(gold)
                and text != dumps(gold)
                and text not in skip
                and not request.grounded(value)
            ):
                return value
        return None

    copies: list[dict[str, Any]] = []
    for param in params:
        value = alternative(param, set())
        if value is not None:
            copies.append({**args, param: value})
        if len(copies) == 2:
            break
    if len(copies) == 1:
        param = next(p for p in params if dumps(copies[0][p]) != dumps(args[p]))
        value = alternative(param, {dumps(copies[0][param])})
        if value is not None:
            copies.append({**args, param: value})
    texts = [dumps(args)] + [dumps(c) for c in copies]
    if len(copies) < 2 or len(set(texts)) != 3:
        report["drop_no_alternative_value"] += 1
        return None
    return rotated(
        descriptions=texts,
        gold=0,
        key=conv["key"],
        state={"function": dumps(conv["by_name"][name], 1), "request": conv["u1"]},
        instructions=FC_ARGS_INSTRUCTIONS,
        cell=f"{name}|{sha(json.dumps(args, sort_keys=True, ensure_ascii=False))[:16]}",
        **fc_row(conv, "fc_args2"),
    )


def fc_ready(
    conv: Mapping[str, Any], report: collections.Counter, **_: Any
) -> dict | None:
    request = Request(conv["u1"])
    if conv["call1"]:
        name, args = conv["call1"]
        schema = conv["by_name"][name]
        needed = required(schema, args)
        if not needed or not all(
            p in args and request.grounded(args[p]) for p in needed
        ):
            report["drop_call_not_grounded"] += 1
            return None
        yes = True
    elif conv["ask"] and conv["call2"]:
        name, args = conv["call2"]
        schema = conv["by_name"][name]
        later = Request(conv["u2"])
        missing = [
            p
            for p in required(schema, args)
            if p in args
            and (
                (isinstance(args[p], (int, float)) and not isinstance(args[p], bool))
                or (isinstance(args[p], str) and len(normalize(args[p])) >= 3)
            )
            and not request.grounded(args[p])
            and later.grounded(args[p])
        ]
        if not missing:
            report["drop_ask_not_supplied_later"] += 1
            return None
        yes = False
    else:
        report["drop_neither"] += 1
        return None
    return noul(
        yes=yes,
        key=conv["key"],
        state={"function": dumps(schema, 1), "request": conv["u1"]},
        instructions=FC_READY_INSTRUCTIONS,
        **fc_row(conv, "fc_ready"),
    )


# --------------------------------------------------------------------------- spam


def clean_comment(text: str) -> str:
    return " ".join(html.unescape(str(text or "")).replace("\ufeff", " ").split())


def ytspam(
    tables: Mapping[str, Sequence[Mapping[str, str]]], report: collections.Counter
) -> Rows:
    rows: Rows = []
    for video, records in sorted(tables.items()):
        for record in records:
            report["read"] += 1
            text = clean_comment(record.get("CONTENT"))
            if not text or record.get("CLASS") not in ("0", "1"):
                report["drop_empty_or_label"] += 1
                continue
            rows.append(
                noul(
                    yes=record["CLASS"] == "1",
                    source="uci_youtube_spam",
                    family="ytspam",
                    group_key="ytspam:" + normalize(text),
                    key=sha(video + "\x1f" + str(record.get("COMMENT_ID")))[:24],
                    state={"comment": text},
                    instructions=YTSPAM_INSTRUCTIONS,
                    template="ib2_ytspam_v1",
                    video=video,
                )
            )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, None, "ib2-ytspam-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- stance


def argq(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        argument = " ".join(str(record.get("argument") or "").split())
        topic = " ".join(str(record.get("topic") or "").split())
        try:
            confident = float(record.get("stance_WA_conf") or 0) >= 0.9999
        except ValueError:
            confident = False
        stance = str(record.get("stance_WA") or "").strip()
        if not confident:
            report["drop_stance_not_unanimous"] += 1
            continue
        if not argument or not topic or stance not in ("1", "-1"):
            report["drop_shape"] += 1
            continue
        rows.append(
            row(
                source="ibm_argq_30k",
                family="argq",
                task_type="choice",
                group_key="argq:" + normalize(argument),
                key=sha(topic + "\x1f" + argument)[:24],
                state={"statement": topic, "argument": argument},
                instructions=ARGQ_INSTRUCTIONS,
                options=ARGQ_OPTIONS,
                label=0 if stance == "1" else 1,
                template="ib2_argq_v1",
                cell=normalize(topic),
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = topic_twins(rows, "ib2-argq-v1", ARGQ_CAP)
    report["selected"] = len(chosen)
    return chosen


def topic_twins(rows: Rows, salt: str, cap: int | None) -> Rows:
    """Per topic (``cell``) equal support and oppose rows (hash order), at most ``cap`` in total."""
    by: dict[str, dict[int, Rows]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for item in ranked(rows, salt):
        by[item["audit_metadata"]["ib2"]["cell"]][item["label"]].append(item)
    per = math.inf if cap is None else cap // (2 * max(1, len(by)))
    out: Rows = []
    for topic in sorted(by):
        size = int(min(len(by[topic][0]), len(by[topic][1]), per))
        out += by[topic][0][:size] + by[topic][1][:size]
    return out


# --------------------------------------------------------------------------- fact verification


def hover(
    claims: Sequence[Mapping[str, Any]],
    lookup: Any,
    report: collections.Counter,
) -> Rows:
    rows: Rows = []
    for record in claims:
        report["read"] += 1
        if record.get("num_hops") not in HOVER_HOPS:
            report["drop_hops"] += 1
            continue
        label = record.get("label")
        claim = " ".join(str(record.get("claim") or "").split())
        if label not in ("SUPPORTED", "NOT_SUPPORTED") or not claim:
            report["drop_shape"] += 1
            continue
        titles: list[str] = []
        for fact in record.get("supporting_facts") or []:
            if fact and fact[0] not in titles:
                titles.append(fact[0])
        passages = []
        for title in titles:
            text = lookup(title)
            if not text:
                break
            passages.append(f"{title}\n{' '.join(text.split())}")
        if not titles or len(passages) != len(titles):
            report["drop_title_missing"] += 1
            continue
        rows.append(
            noul(
                yes=label == "SUPPORTED",
                source="hover_train",
                family="hover",
                group_key="hover:" + str(record.get("hpqa_id")),
                key=str(record["uid"]),
                state={"claim": claim, "evidence": "\n\n".join(passages)},
                instructions=HOVER_INSTRUCTIONS,
                template="ib2_hover_v1",
                hops=record["num_hops"],
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = ratio_labels(rows, HOVER_YES, HOVER_NO, HOVER_CAP, "ib2-hover-v1")
    report["selected"] = len(chosen)
    return chosen


def hover_lookup(path: str) -> Any:
    import sqlite3

    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)

    def lookup(title: str) -> str | None:
        for name in (
            title,
            unicodedata.normalize("NFD", title),
            unicodedata.normalize("NFC", title),
        ):
            found = con.execute(
                "SELECT text FROM documents WHERE id = ?", (name,)
            ).fetchone()
            if found and found[0]:
                return str(found[0])
        return None

    return lookup


# --------------------------------------------------------------------------- knowledge MCQ


def qasc(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        question = " ".join(str(record.get("question") or "").split())
        labels = list(record["choices"]["label"])
        texts = [" ".join(str(t).split()) for t in record["choices"]["text"]]
        fact = normalize(str(record.get("combinedfact") or ""))
        if (
            not question
            or len(texts) != 8
            or record.get("answerKey") not in labels
            or len({normalize(t) for t in texts}) != 8
            or any(not t for t in texts)
        ):
            report["drop_shape"] += 1
            continue
        gold = labels.index(record["answerKey"])
        others = [normalize(t) for i, t in enumerate(texts) if i != gold]
        if normalize(texts[gold]) not in fact or any(
            len(o) >= 3 and o in fact for o in others
        ):
            report["drop_key_not_verified"] += 1
            continue
        rows.append(
            rotated(
                descriptions=texts,
                gold=gold,
                key=str(record["id"]),
                source="qasc_train",
                family="qasc",
                group_key="qasc:" + normalize(question),
                state={"question": question},
                instructions=QASC_INSTRUCTIONS,
                template="ib2_qasc_v1",
                cell="qasc",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = cap_total(rows, QASC_CAP, "ib2-qasc-v1")
    report["selected"] = len(chosen)
    return chosen


def arc(
    records: Sequence[tuple[str, Mapping[str, Any]]], report: collections.Counter
) -> Rows:
    rows: Rows = []
    for subset, record in records:
        report["read"] += 1
        question = " ".join(str(record.get("question") or "").split())
        labels = list(record["choices"]["label"])
        texts = [" ".join(str(t).split()) for t in record["choices"]["text"]]
        if (
            not question
            or not 3 <= len(texts) <= 5
            or record.get("answerKey") not in labels
            or len({normalize(t) for t in texts}) != len(texts)
            or any(not t for t in texts)
        ):
            report["drop_shape"] += 1
            continue
        rows.append(
            rotated(
                descriptions=texts,
                gold=labels.index(record["answerKey"]),
                key=str(record["id"]),
                source="arc_train",
                family="arc",
                group_key="arc:" + normalize(question),
                state={"question": question},
                instructions=ARC_INSTRUCTIONS,
                template="ib2_arc_v1",
                cell=subset,
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    report["selected"] = len(rows)
    return rows


# --------------------------------------------------------------------------- maths

CALC_RE = re.compile(r"<<[^<>]*=([^<>]*)>>")


def number_text(text: str) -> str | None:
    """Canonical text of a plain number (commas removed), or None."""
    raw = text.strip().replace(",", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return None
    value = float(raw)
    if value.is_integer():
        return str(int(value))
    return raw.rstrip("0").rstrip(".") if "." in raw else raw


def gsm(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        question = " ".join(str(record.get("question") or "").split())
        answer = str(record.get("answer") or "")
        if "####" not in answer or not question:
            report["drop_shape"] += 1
            continue
        gold = number_text(answer.rsplit("####", 1)[1])
        if gold is None:
            report["drop_answer_not_numeric"] += 1
            continue
        key = sha(question)[:24]
        distractors = sorted(
            {
                n
                for value in CALC_RE.findall(answer.rsplit("####", 1)[0])
                if (n := number_text(value)) is not None and float(n) != float(gold)
            },
            key=lambda n: order("ib2-gsm-d-v1", key + ":" + n),
        )
        yes = int(order("ib2-gsm-v1", key), 16) % 2 == 0 or not distractors
        shown = gold if yes else distractors[0]
        rows.append(
            noul(
                yes=yes,
                source="gsm8k_train",
                family="gsm",
                group_key="gsm:" + normalize(question),
                key=key,
                state={"problem": question, "claim": f"The answer is {shown}."},
                instructions=GSM_INSTRUCTIONS,
                template="ib2_gsm_v1",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, None, "ib2-gsm-v1")
    report["selected"] = len(chosen)
    return chosen


def gsm2(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    """Amendment 2: a no row states another problem's final answer, so yes and no answers share one distribution."""
    problems = []
    for record in records:
        report["read"] += 1
        question = " ".join(str(record.get("question") or "").split())
        answer = str(record.get("answer") or "")
        gold = number_text(answer.rsplit("####", 1)[1]) if "####" in answer else None
        if not question or gold is None:
            report["drop_shape"] += 1
            continue
        problems.append((sha(question)[:24], question, gold))
    pool = sorted(problems, key=lambda p: (order("ib2-gsm2-pool-v1", p[0]), p[0]))
    rows: Rows = []
    for key, question, gold in problems:
        yes = int(order("ib2-gsm2-v1", key), 16) % 2 == 0
        shown = gold
        if not yes:
            start = int(order("ib2-gsm2-start-v1", key), 16) % len(pool)
            shown = next(
                pool[(start + step) % len(pool)][2]
                for step in range(len(pool))
                if float(pool[(start + step) % len(pool)][2]) != float(gold)
            )
        rows.append(
            noul(
                yes=yes,
                source="gsm8k_train",
                family="gsm2",
                group_key="gsm:" + normalize(question),
                key=key,
                state={"problem": question, "claim": f"The answer is {shown}."},
                instructions=GSM_INSTRUCTIONS,
                template="ib2_gsm2_v1",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, None, "ib2-gsm2-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- contracts


def cnli(data: Mapping[str, Any], report: collections.Counter) -> Rows:
    labels = {k: v["hypothesis"] for k, v in data["labels"].items()}
    if labels != CNLI_HYPOTHESES:
        raise ValueError("ContractNLI hypotheses differ from the pinned list")
    rows: Rows = []
    for doc in data["documents"]:
        text = str(doc.get("text") or "").strip()
        annotations = doc["annotation_sets"][0]["annotations"]
        for hyp, item in sorted(annotations.items()):
            report["read"] += 1
            if len(text) > MAX_STATE_CHARS:
                report["drop_length"] += 1
                continue
            choice = item.get("choice")
            if choice not in ("Entailment", "Contradiction", "NotMentioned"):
                report["drop_shape"] += 1
                continue
            rows.append(
                noul(
                    yes=choice == "Entailment",
                    source="contractnli_train",
                    family="cnli",
                    group_key="cnli:" + str(doc["id"]),
                    key=f"{doc['id']}:{hyp}",
                    state={"contract": text},
                    instructions=CNLI_INSTRUCTIONS.format(
                        hypothesis=CNLI_HYPOTHESES[hyp]
                    ),
                    template="ib2_cnli_v1",
                    cell=hyp,
                    choice=choice,
                )
            )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = per_hypothesis(rows, "ib2-cnli-v1")
    report["selected"] = len(chosen)
    return chosen


def per_hypothesis(rows: Rows, salt: str) -> Rows:
    by: dict[str, Rows] = collections.defaultdict(list)
    for item in rows:
        by[item["audit_metadata"]["ib2"]["cell"]].append(item)
    out: Rows = []
    for hyp in sorted(by):
        out += balance_labels(by[hyp], None, salt)
    return out
