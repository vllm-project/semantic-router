"""IB1 families: pinned publisher rows -> native candidate rows (prereg §2).

Every family function returns validated TRAIN rows with its cap and balance applied. Balancing fields live in
``audit_metadata["ib1"]``: ``hash_key`` (the stable upstream key hash order uses), ``cell`` (the balance cell),
``gold_longer`` (A / B candidate families) and ``gold_position`` (rotated families).
"""

from __future__ import annotations

import ast
import collections
import csv
import json
import re
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from v2.data.sources.common import make_row, noul_options, rotate, sha
from v2.data.textnorm import normalize

ARM = "ib1"
MAX_STATE_CHARS = 12_000
Rows = list[dict[str, Any]]

AB_TEXT = (
    {"key": "a", "description": "Text A"},
    {"key": "b", "description": "Text B"},
)
AB_REPLY = (
    {"key": "a", "description": "Reply A"},
    {"key": "b", "description": "Reply B"},
)
AB_STANZA = (
    {"key": "a", "description": "Stanza A"},
    {"key": "b", "description": "Stanza B"},
)

SUMEDIT_DOMAINS = ("billsum", "sales_call", "sales_email", "shakespeare")
SUMEDIT_INSTRUCTIONS = "Is every statement in the summary supported by the document?"
SUMEDIT_CAP = 3000
EXPQA_INSTRUCTIONS = "Does the evidence fully support the claim?"
EXPQA_MAX_EVIDENCE = 6000
EXPQA_PER_GROUP = 2
EXPQA_CAP = 4000
SMS_INSTRUCTIONS = "Is this text message spam, that is unsolicited advertising, a scam or a phishing attempt?"
W2C_INSTRUCTIONS = "Which reply should the assistant give, given the available tools?"
W2C_PER_KIND = 2000
SNIPS_SEL_INSTRUCTIONS = (
    "Which function should the assistant call to handle this request?"
)
SNIPS_REL_INSTRUCTIONS = "Can one of these functions handle the request?"
SNIPS_CAP = 3000
SNIPS_FUNCTIONS = {
    "AddToPlaylist": {
        "name": "add_to_playlist",
        "description": "Add a song, an album or an artist's music to one of the user's playlists.",
        "parameters": ["playlist", "music_item", "artist", "entity_name"],
    },
    "BookRestaurant": {
        "name": "book_restaurant",
        "description": "Reserve a table at a restaurant for a party at a given time and place.",
        "parameters": [
            "restaurant_name",
            "restaurant_type",
            "cuisine",
            "party_size",
            "time",
            "location",
        ],
    },
    "GetWeather": {
        "name": "get_weather",
        "description": "Get the weather forecast or current conditions for a place and time.",
        "parameters": ["location", "time", "condition"],
    },
    "PlayMusic": {
        "name": "play_music",
        "description": "Start playing music by artist, album, track, genre, year or playlist on a music service.",
        "parameters": [
            "artist",
            "album",
            "track",
            "genre",
            "year",
            "service",
            "playlist",
        ],
    },
    "RateBook": {
        "name": "rate_book",
        "description": "Record the user's rating of a book or another written work.",
        "parameters": ["object_name", "object_type", "rating_value", "best_rating"],
    },
    "SearchCreativeWork": {
        "name": "search_creative_work",
        "description": "Find a creative work such as a film, TV show, song, album, book, game or painting.",
        "parameters": ["object_name", "object_type"],
    },
    "SearchScreeningEvent": {
        "name": "search_screening_event",
        "description": "Find cinema showtimes, or which cinemas show a film, at a place and time.",
        "parameters": ["movie_name", "object_location_type", "location", "time"],
    },
}
ARGS_DOMAINS = ("idebate.org", "debatewise.org", "www.debatepedia.org")
ARGS_INSTRUCTIONS = "Does the argument support or oppose the statement?"
ARGS_OPTIONS = (
    {"key": "support", "description": "The argument supports the statement"},
    {"key": "oppose", "description": "The argument opposes the statement"},
)
ARGS_MAX_PREMISE = 3000
ARGS_CAP = 6000
ISARC_INSTRUCTIONS = (
    "One of these texts is sarcastic and the other says the same thing without sarcasm. "
    "Which one is sarcastic?"
)
PROCB_GENERATORS = (
    "Qwen2-1.5B-Instruct",
    "Qwen2-7B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-Math-7B-Instruct",
)
PROCB_INSTRUCTIONS = "Is every step of this solution correct?"
SENTFIN_INSTRUCTIONS = (
    "What sentiment does the headline express toward the named entity?"
)
SENTFIN_LABELS = ("negative", "neutral", "positive")
SENTFIN_OPTIONS = (
    {"key": "negative", "description": "Negative toward the entity"},
    {"key": "neutral", "description": "Neutral toward the entity"},
    {"key": "positive", "description": "Positive toward the entity"},
)
SENTFIN_CAP = 8000
WANDS_INSTRUCTIONS = "How well does this product match the shopper's search query?"
WANDS_LABELS = ("Exact", "Partial", "Irrelevant")
WANDS_OPTIONS = (
    {
        "key": "exact",
        "description": "Exact match: the product is what the query asks for and meets every stated requirement",
    },
    {
        "key": "partial",
        "description": "Partial match: the product is related to the query but misses some of its requirements",
    },
    {
        "key": "irrelevant",
        "description": "Irrelevant: the product is not what the query asks for",
    },
)
WANDS_MAX_DESCRIPTION = 1500
WANDS_MAX_FEATURES = 800
WANDS_PER_QUERY = 40
WANDS_CAP = 8000
MAUD_MAX_TEXT = 5000
MAUD_MAX_OPTIONS = 6
MAUD_PER_QUESTION = 250
MAUD_CAP = 6000
COPA_INSTRUCTIONS = "Which alternative is more plausibly the {relation} of the premise?"
CSQA_INSTRUCTIONS = "Choose the best answer to the question."
CSQA_CAP = 6000
MEDMCQA_INSTRUCTIONS = "Choose the correct answer to the medical exam question."
MEDMCQA_CAP = 8000
WANLI_INSTRUCTIONS = "What is the relation between the premise and the hypothesis?"
WANLI_LABELS = ("entailment", "neutral", "contradiction")
WANLI_OPTIONS = (
    {
        "key": "entailment",
        "description": "Entailment: the hypothesis must be true if the premise is true",
    },
    {
        "key": "neutral",
        "description": "Neutral: the hypothesis may or may not be true given the premise",
    },
    {
        "key": "contradiction",
        "description": "Contradiction: the hypothesis cannot be true if the premise is true",
    },
)
WANLI_CAP = 9000
POEM_INSTRUCTIONS = (
    "One stanza is the original text of a published poem; in the other one word was replaced. "
    "Which is the original? Consider rhyme, metre and sense."
)
POEM_PER_BOOK = 20
POEM_CAP = 4000
WORD_RE = re.compile(r"^(.*?)([A-Za-z]+)([^A-Za-z]*)$")

# Fixed instruction and option strings (G0 reports their hits separately; they are not data).
TEMPLATE_STRINGS = frozenset(
    [
        SUMEDIT_INSTRUCTIONS,
        EXPQA_INSTRUCTIONS,
        SMS_INSTRUCTIONS,
        W2C_INSTRUCTIONS,
        SNIPS_SEL_INSTRUCTIONS,
        SNIPS_REL_INSTRUCTIONS,
        ARGS_INSTRUCTIONS,
        ISARC_INSTRUCTIONS,
        PROCB_INSTRUCTIONS,
        SENTFIN_INSTRUCTIONS,
        WANDS_INSTRUCTIONS,
        CSQA_INSTRUCTIONS,
        MEDMCQA_INSTRUCTIONS,
        WANLI_INSTRUCTIONS,
        POEM_INSTRUCTIONS,
        COPA_INSTRUCTIONS.format(relation="cause"),
        COPA_INSTRUCTIONS.format(relation="effect"),
        "No",
        "Yes",
    ]
    + [o["description"] for group in (AB_TEXT, AB_REPLY, AB_STANZA) for o in group]
    + [o["description"] for o in ARGS_OPTIONS + SENTFIN_OPTIONS + WANDS_OPTIONS]
    + [o["description"] for o in WANLI_OPTIONS]
    + [f"{f['name']}: {f['description']}" for f in SNIPS_FUNCTIONS.values()]
)


# --------------------------------------------------------------------------- helpers


def order(salt: str, key: str) -> str:
    return sha(f"{salt}:{key}")


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> Rows:
    return sorted(
        rows, key=lambda r: order(salt, r["audit_metadata"]["ib1"]["hash_key"])
    )


def state_chars(state: Mapping[str, Any]) -> int:
    return sum(len(str(value)) for value in state.values())


def meta(key: str, cell: str, **extra: Any) -> dict[str, Any]:
    return {"ib1": {"hash_key": key, "cell": cell, **extra}}


def longer(gold: str, other: str) -> bool | None:
    if len(gold) == len(other):
        return None
    return len(gold) > len(other)


def ab(key: str, gold: str, other: str) -> tuple[str, str, int]:
    """(A, B, label): the gold side is A unless the salted key hash is odd."""
    if int(order("ib1-ab-v1", key), 16) % 2:
        return other, gold, 1
    return gold, other, 0


def row(
    *,
    source: str,
    family: str,
    task_type: str,
    language: str,
    group_key: str,
    key: str,
    state: dict[str, Any],
    instructions: str,
    options: Sequence[Mapping[str, Any]],
    label: int,
    template: str,
    cell: str,
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
        audit=meta(key, cell, **extra),
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
    *,
    descriptions: Sequence[str],
    gold: int,
    key: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Choice row whose options are rotated by a per-row seed (gold position near-uniform)."""
    options = [
        {"key": f"o{i + 1}", "description": text} for i, text in enumerate(descriptions)
    ]
    shifted, label = rotate(options, gold, f"ib1-rot-v1:{kwargs['family']}:{key}")
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
    """One row per id; ids whose copies disagree on input or label are dropped entirely."""
    copies: dict[str, Rows] = collections.defaultdict(list)
    for item in rows:
        copies[item["id"]].append(item)
    kept = []
    for members in copies.values():
        if len({(r["input_sha256"], r["label"]) for r in members}) > 1:
            report["drop_conflicting_duplicates"] += len(members)
            continue
        report["drop_exact_duplicates"] += len(members) - 1
        kept.append(members[0])
    return kept


def balance_labels(rows: Rows, cap: int | None, salt: str) -> Rows:
    """Equal rows per label (hash order), at most ``cap`` in total."""
    by: dict[int, Rows] = collections.defaultdict(list)
    for item in ranked(rows, salt):
        by[item["label"]].append(item)
    if not by:
        return []
    size = min(len(members) for members in by.values())
    if len(by) < len(rows[0]["options"]):
        size = 0
    if cap is not None:
        size = min(size, cap // len(rows[0]["options"]))
    return [item for label in sorted(by) for item in by[label][:size]]


def take_length_balanced(rows: Rows, count: int, salt: str) -> Rows:
    """Up to ``count`` rows with gold-longer and gold-shorter in equal numbers (equal lengths free)."""
    ordered = ranked(rows, salt)
    yes = [r for r in ordered if r["audit_metadata"]["ib1"]["gold_longer"] is True]
    no = [r for r in ordered if r["audit_metadata"]["ib1"]["gold_longer"] is False]
    tie = [r for r in ordered if r["audit_metadata"]["ib1"]["gold_longer"] is None]
    chosen = tie[:count]
    half = min(len(yes), len(no), (count - len(chosen)) // 2)
    return chosen + yes[:half] + no[:half]


def per_cell_cap(
    rows: Rows, cell: Callable[[dict[str, Any]], str], cap: int, salt: str
) -> Rows:
    kept, taken = [], collections.Counter()
    for item in ranked(rows, salt):
        name = cell(item)
        if taken[name] < cap:
            kept.append(item)
            taken[name] += 1
    return kept


def cap_total(rows: Rows, cap: int, salt: str) -> Rows:
    return ranked(rows, salt)[:cap]


def fit(rows: Rows, report: collections.Counter, family: str) -> Rows:
    kept = []
    for item in rows:
        if state_chars(item["state"]) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        kept.append(item)
    return kept


# --------------------------------------------------------------------------- faithfulness


def sumedit(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        if record.get("domain") not in SUMEDIT_DOMAINS:
            report["drop_domain"] += 1
            continue
        doc, summary = (
            str(record.get("doc") or "").strip(),
            str(record.get("summary") or "").strip(),
        )
        if not doc or not summary or record.get("label") not in (0, 1):
            report["drop_empty_or_label"] += 1
            continue
        key = sha(str(record["id"]) + "\x1f" + summary)[:24]
        rows.append(
            noul(
                yes=record["label"] == 1,
                source="summedits",
                family="sumedit",
                language="en",
                group_key="sumedit:" + normalize(doc),
                key=key,
                state={"document": doc, "summary": summary},
                instructions=SUMEDIT_INSTRUCTIONS,
                template="ib1_sumedit_v1",
                cell=record["domain"],
                domain=record["domain"],
            )
        )
    rows = resolve(fit(rows, report, "sumedit"), report)
    report["eligible"] = len(rows)
    chosen: Rows = []
    per_domain = SUMEDIT_CAP // len(SUMEDIT_DOMAINS)
    for domain in SUMEDIT_DOMAINS:
        members = [r for r in rows if r["audit_metadata"]["ib1"]["domain"] == domain]
        if members:
            chosen += balance_labels(members, per_domain, "ib1-sumedit-v1")
    report["selected"] = len(chosen)
    return chosen


def expqa(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        question = str(record.get("question") or "").strip()
        for system, answer in sorted((record.get("answers") or {}).items()):
            for claim in answer.get("claims") or []:
                report["read"] += 1
                support = claim.get("support")
                if support not in ("Complete", "Missing"):
                    report["drop_support_" + str(support)] += 1
                    continue
                evidence = [
                    str(e).strip()
                    for e in claim.get("evidence") or []
                    if str(e).strip()
                ]
                text = str(claim.get("claim_string") or "").strip()
                if not evidence or not text or not question:
                    report["drop_no_evidence"] += 1
                    continue
                joined = "\n\n".join(evidence)
                if len(joined) > EXPQA_MAX_EVIDENCE:
                    report["drop_evidence_length"] += 1
                    continue
                key = sha(question + "\x1f" + system + "\x1f" + text)[:24]
                rows.append(
                    noul(
                        yes=support == "Complete",
                        source="expertqa_r2",
                        family="expqa",
                        language="en",
                        group_key="expqa:" + normalize(question),
                        key=key,
                        state={"evidence": joined, "claim": text},
                        instructions=EXPQA_INSTRUCTIONS,
                        template="ib1_expqa_v1",
                        system=system,
                    )
                )
    rows = resolve(fit(rows, report, "expqa"), report)
    report["eligible"] = len(rows)
    rows = per_cell_cap(
        rows,
        lambda r: f"{r['group_id']}|{r['label']}",
        EXPQA_PER_GROUP,
        "ib1-expqa-group-v1",
    )
    chosen = balance_labels(rows, EXPQA_CAP, "ib1-expqa-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- phishing / fraud


def sms(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        text = str(record.get("sms") or "").strip()
        if not text or record.get("label") not in (0, 1):
            report["drop_empty_or_label"] += 1
            continue
        key = sha(text)[:24]
        rows.append(
            noul(
                yes=record["label"] == 1,
                source="sms_spam_collection",
                family="sms",
                language="en",
                group_key="sms:" + normalize(text),
                key=key,
                state={"message": text},
                instructions=SMS_INSTRUCTIONS,
                template="ib1_sms_v1",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, None, "ib1-sms-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- tool calls


def w2c_content(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("content") or "").strip()
    return str(message or "").strip()


def w2c(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        messages = record.get("messages") or []
        if len(messages) != 1 or messages[0].get("role") != "user":
            report["drop_not_single_user_turn"] += 1
            continue
        user = w2c_content(messages[0])
        chosen, rejected = w2c_content(record.get("chosen_response")), w2c_content(
            record.get("rejected_response")
        )
        if not user or not chosen or not rejected or chosen == rejected:
            report["drop_empty_or_equal"] += 1
            continue
        tools = [
            json.loads(t) if isinstance(t, str) else t
            for t in record.get("tools") or []
        ]
        key = sha(user + "\x1f" + chosen + "\x1f" + rejected)[:24]
        a, b, label = ab(key, chosen, rejected)
        kind = "toolcall" if "<TOOLCALL>" in chosen else "text"
        rows.append(
            row(
                source="when2call_train_pref",
                family="w2c",
                task_type="choice",
                language="en",
                group_key="w2c:" + normalize(user),
                key=key,
                state={
                    "tools": json.dumps(tools, ensure_ascii=False, indent=1),
                    "user_message": user,
                    "reply_a": a,
                    "reply_b": b,
                },
                instructions=W2C_INSTRUCTIONS,
                options=AB_REPLY,
                label=label,
                template="ib1_w2c_v1",
                cell=kind,
                gold_longer=longer(chosen, rejected),
                chosen_kind=kind,
            )
        )
    rows = resolve(fit(rows, report, "w2c"), report)
    report["eligible"] = len(rows)
    chosen_rows: Rows = []
    for kind in ("toolcall", "text"):
        members = [r for r in rows if r["audit_metadata"]["ib1"]["chosen_kind"] == kind]
        chosen_rows += take_length_balanced(members, W2C_PER_KIND, "ib1-w2c-v1")
    report["selected"] = len(chosen_rows)
    return chosen_rows


def snips_utterances(
    files: Mapping[str, Any], report: collections.Counter
) -> list[tuple[str, str]]:
    out = []
    seen: dict[str, set[str]] = collections.defaultdict(set)
    for intent, data in sorted(files.items()):
        for item in data[intent]:
            report["read"] += 1
            text = " ".join("".join(seg["text"] for seg in item["data"]).split())
            if text:
                seen[normalize(text)].add(intent)
                out.append((intent, text))
    kept = []
    for intent, text in out:
        if len(seen[normalize(text)]) > 1:
            report["drop_conflicting_intents"] += 1
            continue
        kept.append((intent, text))
    return kept


def snips_function(intent: str) -> str:
    spec = SNIPS_FUNCTIONS[intent]
    return f"{spec['name']}: {spec['description']}"


def snips_schema(intent: str) -> dict[str, Any]:
    spec = SNIPS_FUNCTIONS[intent]
    return {
        "name": spec["name"],
        "description": spec["description"],
        "parameters": list(spec["parameters"]),
    }


def snips(
    files: Mapping[str, Any],
    report_sel: collections.Counter,
    report_rel: collections.Counter,
) -> tuple[Rows, Rows]:
    utterances = snips_utterances(files, report_sel)
    intents = sorted(SNIPS_FUNCTIONS)
    sel: Rows = []
    rel: Rows = []
    for intent, text in utterances:
        key = sha(intent + "\x1f" + text)[:24]
        others = sorted(
            (i for i in intents if i != intent),
            key=lambda i: order("ib1-snips-other-v1", key + i),
        )
        group = "snips:" + normalize(text)
        if int(order("ib1-snips-v1", normalize(text)), 16) % 2 == 0:
            picked = [intent] + others[:3]
            sel.append(
                rotated(
                    descriptions=[snips_function(i) for i in picked],
                    gold=0,
                    key=key,
                    source="snips_2017_custom_intents",
                    family="snips_sel",
                    language="en",
                    group_key=group,
                    state={"request": text},
                    instructions=SNIPS_SEL_INSTRUCTIONS,
                    template="ib1_snips_sel_v1",
                    cell=intent,
                    intent=intent,
                )
            )
        else:
            yes = int(order("ib1-snips-rel-v1", key), 16) % 2 == 0
            names = ([intent] + others[:2]) if yes else others[:3]
            names = sorted(names, key=lambda i: order("ib1-snips-order-v1", key + i))
            rel.append(
                noul(
                    yes=yes,
                    source="snips_2017_custom_intents",
                    family="snips_rel",
                    language="en",
                    group_key=group,
                    key=key,
                    state={
                        "request": text,
                        "functions": json.dumps(
                            [snips_schema(i) for i in names], indent=1
                        ),
                    },
                    instructions=SNIPS_REL_INSTRUCTIONS,
                    template="ib1_snips_rel_v1",
                    cell=f"{intent}|{'yes' if yes else 'no'}",
                    intent=intent,
                )
            )
    sel = resolve(sel, report_sel)
    rel = resolve(rel, report_rel)
    report_sel["eligible"], report_rel["eligible"] = len(sel), len(rel)
    sel = per_cell_cap(
        sel,
        lambda r: r["audit_metadata"]["ib1"]["intent"],
        SNIPS_CAP // len(intents),
        "ib1-snips-sel-v1",
    )
    rel = per_cell_cap(
        rel,
        lambda r: r["audit_metadata"]["ib1"]["cell"],
        SNIPS_CAP // (2 * len(intents)),
        "ib1-snips-rel-cap-v1",
    )
    rel = balance_labels(rel, None, "ib1-snips-rel-bal-v1")
    report_sel["selected"], report_rel["selected"] = len(sel), len(rel)
    return sel, rel


# --------------------------------------------------------------------------- stance


def args(lines: Iterable[str], report: collections.Counter) -> Rows:
    by_statement: dict[str, dict[str, list[tuple[str, str, str]]]] = (
        collections.defaultdict(lambda: {"PRO": [], "CON": []})
    )
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        report["read"] += 1
        domain = urllib.parse.urlparse(
            record.get("context", {}).get("sourceUrl", "")
        ).netloc
        if domain not in ARGS_DOMAINS:
            report["drop_portal"] += 1
            continue
        premises = record.get("premises") or []
        statement = " ".join(str(record.get("conclusion") or "").split())
        if len(premises) != 1 or len(statement.split()) < 3:
            report["drop_shape"] += 1
            continue
        text = str(premises[0].get("text") or "").strip()
        stance = premises[0].get("stance")
        if not text or stance not in ("PRO", "CON") or len(text) > ARGS_MAX_PREMISE:
            report["drop_premise"] += 1
            continue
        by_statement[normalize(statement)][stance].append(
            (record["id"], statement, text)
        )
    rows: Rows = []
    twins = [s for s, sides in by_statement.items() if sides["PRO"] and sides["CON"]]
    report["twin_statements"] = len(twins)
    for norm in sorted(twins, key=lambda s: order("ib1-args-statement-v1", s)):
        if len(rows) >= ARGS_CAP:
            break
        pair = []
        for stance in ("PRO", "CON"):
            ident, statement, text = min(
                by_statement[norm][stance], key=lambda t: order("ib1-args-v1", t[0])
            )
            pair.append(
                row(
                    source="args_me_portals",
                    family="args",
                    task_type="choice",
                    language="en",
                    group_key="args:" + norm,
                    key=sha(ident)[:24],
                    state={"statement": statement, "argument": text},
                    instructions=ARGS_INSTRUCTIONS,
                    options=ARGS_OPTIONS,
                    label=0 if stance == "PRO" else 1,
                    template="ib1_args_v1",
                    cell=stance,
                )
            )
        if pair[0]["input_sha256"] != pair[1]["input_sha256"]:
            rows += pair
    report["selected"] = len(rows)
    return rows


# --------------------------------------------------------------------------- sarcasm


def isarc(
    tables: Mapping[str, Sequence[Mapping[str, str]]], report: collections.Counter
) -> Rows:
    rows: Rows = []
    for lang, records in sorted(tables.items()):
        text_field = "tweet" if lang == "en" else "text"
        for record in records:
            report["read"] += 1
            if record.get("sarcastic") != "1":
                report["drop_not_sarcastic"] += 1
                continue
            tweet = str(record.get(text_field) or "").strip()
            rephrase = str(record.get("rephrase") or "").strip()
            if not tweet or not rephrase or normalize(tweet) == normalize(rephrase):
                report["drop_no_rephrase"] += 1
                continue
            key = sha(lang + "\x1f" + tweet)[:24]
            a, b, label = ab(key, tweet, rephrase)
            rows.append(
                row(
                    source="isarcasmeval_train",
                    family="isarc",
                    task_type="choice",
                    language=lang,
                    group_key="isarc:" + normalize(tweet),
                    key=key,
                    state={"text_a": a, "text_b": b},
                    instructions=ISARC_INSTRUCTIONS,
                    options=AB_TEXT,
                    label=label,
                    template="ib1_isarc_v1",
                    cell=lang,
                    gold_longer=longer(tweet, rephrase),
                )
            )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = take_length_balanced(rows, len(rows), "ib1-isarc-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- math verification


def procb(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        if record.get("generator") not in PROCB_GENERATORS:
            report["drop_generator_licence"] += 1
            continue
        problem = str(record.get("problem") or "").strip()
        steps = [str(s).strip() for s in record.get("steps") or []]
        if not problem or not steps or any(not s for s in steps):
            report["drop_empty"] += 1
            continue
        solution = "\n\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
        rows.append(
            noul(
                yes=int(record["label"]) == -1,
                source="processbench_math_apache",
                family="procb",
                language="en",
                group_key="procb:" + normalize(problem),
                key=str(record["id"]),
                state={"problem": problem, "solution": solution},
                instructions=PROCB_INSTRUCTIONS,
                template="ib1_procb_v1",
                generator=record["generator"],
            )
        )
    rows = resolve(fit(rows, report, "procb"), report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, None, "ib1-procb-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- financial entity sentiment


def sentfin(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        title = " ".join(str(record.get("Title") or "").split())
        try:
            decisions = ast.literal_eval(record.get("Decisions") or "{}")
        except (SyntaxError, ValueError):
            report["drop_unparsed"] += 1
            continue
        for entity, decision in sorted(decisions.items()):
            report["read"] += 1
            entity = " ".join(str(entity).split())
            if not title or not entity or decision not in SENTFIN_LABELS:
                report["drop_empty_or_label"] += 1
                continue
            key = sha(title + "\x1f" + entity)[:24]
            rows.append(
                row(
                    source="sentfin_v1",
                    family="sentfin",
                    task_type="choice",
                    language="en",
                    group_key="sentfin:" + normalize(title),
                    key=key,
                    state={"headline": title, "entity": entity},
                    instructions=SENTFIN_INSTRUCTIONS,
                    options=SENTFIN_OPTIONS,
                    label=SENTFIN_LABELS.index(decision),
                    template="ib1_sentfin_v1",
                    cell=decision,
                )
            )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, SENTFIN_CAP, "ib1-sentfin-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- product search


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def wands_product(product: Mapping[str, str]) -> str:
    features = "; ".join(
        part.strip()
        for part in str(product.get("product_features") or "").split("|")
        if part.strip()
    )
    lines = [
        f"Name: {' '.join(str(product.get('product_name') or '').split())}",
        f"Class: {' '.join(str(product.get('product_class') or '').split())}",
        f"Category: {' '.join(str(product.get('category hierarchy') or '').split())}",
        f"Description: {clip(product.get('product_description') or '', WANDS_MAX_DESCRIPTION)}",
        f"Features: {clip(features, WANDS_MAX_FEATURES)}",
    ]
    return "\n".join(lines)


def wands(
    queries: Sequence[Mapping[str, str]],
    products: Sequence[Mapping[str, str]],
    labels: Sequence[Mapping[str, str]],
    report: collections.Counter,
) -> Rows:
    query_text = {
        q["query_id"]: " ".join(str(q.get("query") or "").split()) for q in queries
    }
    product_by = {p["product_id"]: p for p in products}
    rows: Rows = []
    for record in labels:
        report["read"] += 1
        query = query_text.get(record.get("query_id"))
        product = product_by.get(record.get("product_id"))
        if not query or product is None or record.get("label") not in WANDS_LABELS:
            report["drop_missing"] += 1
            continue
        key = sha(record["query_id"] + "\x1f" + record["product_id"])[:24]
        rows.append(
            row(
                source="wands",
                family="wands",
                task_type="choice",
                language="en",
                group_key="wands:" + record["query_id"],
                key=key,
                state={"query": query, "product": wands_product(product)},
                instructions=WANDS_INSTRUCTIONS,
                options=WANDS_OPTIONS,
                label=WANDS_LABELS.index(record["label"]),
                template="ib1_wands_v1",
                cell=record["label"],
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    rows = per_cell_cap(
        rows,
        lambda r: f"{r['group_id']}|{r['label']}",
        WANDS_PER_QUERY // 3,
        "ib1-wands-query-v1",
    )
    chosen = balance_labels(rows, WANDS_CAP, "ib1-wands-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- contracts


def maud_question(question: str, subquestion: str) -> str:
    q = re.sub(r"[-\s]*Answer$", "", question.strip()).strip()
    sub = subquestion.strip()
    if sub and sub != "<NONE>":
        return f"{q} ({sub})"
    return q


def maud(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    answers: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for record in records:
        answers[(record["question"], record["subquestion"])].add(
            record["answer"].strip()
        )
    rows: Rows = []
    for record in records:
        report["read"] += 1
        if record.get("data_type") != "main":
            report["drop_data_type"] += 1
            continue
        options = sorted(
            a for a in answers[(record["question"], record["subquestion"])] if a
        )
        text = str(record.get("text") or "").strip()
        answer = record["answer"].strip()
        if not 2 <= len(options) <= MAUD_MAX_OPTIONS or answer not in options:
            report["drop_option_count"] += 1
            continue
        if not text or len(text) > MAUD_MAX_TEXT:
            report["drop_text_length"] += 1
            continue
        question = maud_question(record["question"], record["subquestion"])
        key = sha(record["contract_name"] + "\x1f" + question + "\x1f" + text)[:24]
        rows.append(
            rotated(
                descriptions=options,
                gold=options.index(answer),
                key=key,
                source="maud_train_main",
                family="maud",
                language="en",
                group_key="maud:" + record["contract_name"],
                state={"excerpt": text},
                instructions=(
                    f"Merger-agreement deal point: {question}. Which answer does the excerpt support?"
                ),
                template="ib1_maud_v1",
                cell=question,
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    rows = per_cell_cap(
        rows,
        lambda r: r["audit_metadata"]["ib1"]["cell"],
        MAUD_PER_QUESTION,
        "ib1-maud-q-v1",
    )
    chosen = cap_total(rows, MAUD_CAP, "ib1-maud-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- commonsense and knowledge MCQ


def copa(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        premise = str(record.get("premise") or "").strip()
        c1, c2 = (
            str(record.get("choice1") or "").strip(),
            str(record.get("choice2") or "").strip(),
        )
        relation = record.get("question")
        if (
            not premise
            or not c1
            or not c2
            or relation not in ("cause", "effect")
            or record.get("label") not in ("0", "1")
        ):
            report["drop_shape"] += 1
            continue
        label = int(record["label"])
        key = sha(premise + "\x1f" + c1 + "\x1f" + c2)[:24]
        rows.append(
            row(
                source="balanced_copa_train",
                family="copa",
                task_type="choice",
                language="en",
                group_key="copa:" + "|".join(sorted((normalize(c1), normalize(c2)))),
                key=key,
                state={"premise": premise},
                instructions=COPA_INSTRUCTIONS.format(relation=relation),
                options=[
                    {"key": "a", "description": c1},
                    {"key": "b", "description": c2},
                ],
                label=label,
                template="ib1_copa_v1",
                cell=relation,
                gold_longer=longer((c1, c2)[label], (c1, c2)[1 - label]),
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    report["selected"] = len(rows)
    return rows


def csqa(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        question = str(record.get("question") or "").strip()
        labels = list(record["choices"]["label"])
        texts = [str(t).strip() for t in record["choices"]["text"]]
        if (
            not question
            or len(texts) != 5
            or record.get("answerKey") not in labels
            or len({normalize(t) for t in texts}) != 5
        ):
            report["drop_shape"] += 1
            continue
        rows.append(
            rotated(
                descriptions=texts,
                gold=labels.index(record["answerKey"]),
                key=str(record["id"]),
                source="commonsenseqa_train",
                family="csqa",
                language="en",
                group_key="csqa:"
                + normalize(str(record.get("question_concept") or question)),
                state={"question": question},
                instructions=CSQA_INSTRUCTIONS,
                template="ib1_csqa_v1",
                cell="csqa",
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = cap_total(rows, CSQA_CAP, "ib1-csqa-v1")
    report["selected"] = len(chosen)
    return chosen


def medmcqa_consistent(options: Sequence[str], gold: int, explanation: str) -> bool:
    exp = normalize(explanation)
    norms = [normalize(o) for o in options]
    if len(norms[gold]) < 3 or norms[gold] not in exp:
        return False
    return not any(len(n) >= 3 and n in exp for i, n in enumerate(norms) if i != gold)


def medmcqa(records: Iterable[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        if record.get("choice_type") != "single":
            report["drop_multi"] += 1
            continue
        question = str(record.get("question") or "").strip()
        options = [
            str(record.get(k) or "").strip() for k in ("opa", "opb", "opc", "opd")
        ]
        gold = record.get("cop")
        if (
            not question
            or any(not o for o in options)
            or len({normalize(o) for o in options}) != 4
            or gold not in (0, 1, 2, 3)
        ):
            report["drop_shape"] += 1
            continue
        if not medmcqa_consistent(options, gold, str(record.get("exp") or "")):
            report["drop_explanation_inconsistent"] += 1
            continue
        rows.append(
            rotated(
                descriptions=options,
                gold=gold,
                key=str(record["id"]),
                source="medmcqa_train",
                family="medmcqa",
                language="en",
                group_key="medmcqa:" + normalize(question),
                state={"question": question},
                instructions=MEDMCQA_INSTRUCTIONS,
                template="ib1_medmcqa_v1",
                cell=str(record.get("subject_name") or ""),
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = cap_total(rows, MEDMCQA_CAP, "ib1-medmcqa-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- adversarial NLI


def wanli(
    records: Iterable[Mapping[str, Any]],
    annotations: Iterable[Mapping[str, Any]],
    report: collections.Counter,
) -> Rows:
    labels: dict[int, list[str]] = collections.defaultdict(list)
    for item in annotations:
        labels[int(item["id"])].append(str(item.get("label")))
    rows: Rows = []
    for record in records:
        report["read"] += 1
        premise, hypothesis = (
            str(record.get("premise") or "").strip(),
            str(record.get("hypothesis") or "").strip(),
        )
        gold = record.get("gold")
        votes = labels.get(int(record["id"]), [])
        if not premise or not hypothesis or gold not in WANLI_LABELS:
            report["drop_shape"] += 1
            continue
        if len(votes) < 2 or any(v != gold for v in votes):
            report["drop_not_unanimous"] += 1
            continue
        rows.append(
            row(
                source="wanli_train",
                family="wanli",
                task_type="choice",
                language="en",
                group_key="wanli:" + normalize(premise),
                key=str(record["id"]),
                state={"premise": premise, "hypothesis": hypothesis},
                instructions=WANLI_INSTRUCTIONS,
                options=WANLI_OPTIONS,
                label=WANLI_LABELS.index(gold),
                template="ib1_wanli_v1",
                cell=gold,
            )
        )
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = balance_labels(rows, WANLI_CAP, "ib1-wanli-v1")
    report["selected"] = len(chosen)
    return chosen


# --------------------------------------------------------------------------- poems (BPoMP-like)


def final_word(line: str) -> tuple[str, str, str] | None:
    match = WORD_RE.match(line.rstrip())
    if not match:
        return None
    head, word, tail = match.groups()
    if not word.isalpha() or not word.isascii():
        return None
    return head, word, tail


def poem_windows(
    lines: Sequence[tuple[str, int]], report: collections.Counter
) -> list[tuple[int, int, list[str]]]:
    """(book, start, four lines) windows whose 2nd and 4th lines end in an orthographic rhyme."""
    by_book: dict[int, list[str]] = collections.defaultdict(list)
    for text, book in lines:
        by_book[int(book)].append(" ".join(str(text).split()))
    windows = []
    for book, book_lines in sorted(by_book.items()):
        for start in range(0, len(book_lines) - 3, 4):
            window = book_lines[start : start + 4]
            report["windows"] += 1
            if any(not 4 <= len(line.split()) <= 12 for line in window):
                continue
            second, fourth = final_word(window[1]), final_word(window[3])
            if not second or not fourth:
                continue
            w2, w4 = second[1].lower(), fourth[1].lower()
            if len(w2) < 3 or len(w4) < 3 or w2 == w4 or w2[-3:] != w4[-3:]:
                continue
            windows.append((book, start, window))
    return windows


def poem_replacement(
    pool: Mapping[int, list[tuple[str, int]]],
    key: str,
    word: str,
    rhyme: str,
    book: int,
) -> str | None:
    """Walk the sorted pool of other-book line-final words of length within ±1 from a key-hashed start."""
    candidates = [
        c
        for size in (len(word) - 1, len(word), len(word) + 1)
        for c in pool.get(size, [])
    ]
    if not candidates:
        return None
    start = int(order("ib1-poem-sub-v1", key), 16) % len(candidates)
    for step in range(len(candidates)):
        other, other_book = candidates[(start + step) % len(candidates)]
        if other_book != book and other[-2:] != rhyme[-2:] and other != word.lower():
            return other
    return None


def poem(lines: Sequence[tuple[str, int]], report: collections.Counter) -> Rows:
    windows = poem_windows(lines, report)
    report["rhyme_windows"] = len(windows)
    pool: dict[int, list[tuple[str, int]]] = collections.defaultdict(list)
    for item in sorted({(final_word(w[3])[1].lower(), book) for book, _, w in windows}):
        pool[len(item[0])].append(item)
    rows: Rows = []
    per_book: collections.Counter = collections.Counter()
    for book, start, window in sorted(
        windows, key=lambda w: order("ib1-poem-window-v1", f"{w[0]}:{w[1]}")
    ):
        if per_book[book] >= POEM_PER_BOOK or len(rows) >= POEM_CAP:
            continue
        key = sha(f"{book}:{start}:" + "\n".join(window))[:24]
        head, word, tail = final_word(window[3])
        rhyme = final_word(window[1])[1].lower()
        replacement = poem_replacement(pool, key, word, rhyme, book)
        if replacement is None:
            report["drop_no_replacement"] += 1
            continue
        if word[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        perturbed = window[:3] + [head + replacement + tail]
        original_text, perturbed_text = "\n".join(window), "\n".join(perturbed)
        a, b, label = ab(key, original_text, perturbed_text)
        rows.append(
            row(
                source="gutenberg_poetry_cc0",
                family="poem",
                task_type="choice",
                language="en",
                group_key=f"poem:{book}:{start}",
                key=key,
                state={"stanza_a": a, "stanza_b": b},
                instructions=POEM_INSTRUCTIONS,
                options=AB_STANZA,
                label=label,
                template="ib1_poem_v1",
                cell=str(book),
                gold_longer=longer(original_text, perturbed_text),
                book=book,
            )
        )
        per_book[book] += 1
    rows = resolve(rows, report)
    report["selected"] = len(rows)
    return rows


# --------------------------------------------------------------------------- readers


def read_csv(path: Any, **kwargs: Any) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, **kwargs))
