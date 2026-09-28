"""H1 short cross-domain human families (English) and H5 MTOP intent Choice.

Rules: data-arms-v2-prereg-2026-09-28.md sections 3 and 4. MultiWOZ 2.2
mirrors the v1 SGD builder (``v2.data.sources.sgd``): one activation or request
event per dialogue, Choice balanced per intent within a service and Noul
balanced per (service, asked slot). Taskmaster-2 asks one present and one
absent slot per conversation, each picked against the running per-slot
true-minus-false gap of its domain, which never leaves [-1, 1] (a conversation
that would push it out is dropped), so slot names carry no label. DBpedia-14
and MTOP are C6, WinoGrande is C7 (source option order), CommonsenseQA, ARC,
OpenBookQA, AQuA-RAT, QuaRTz and ROPES are C8, SciTail is Noul balanced per
hypothesis (groups join rows sharing a premise or a hypothesis) and GSM8K uses
the C9 numeric twins. Builders read TRAIN files only and return rows before
the whole-group cap; ordering comes from sorted inputs, SHA-256 and
string-seeded ``random.Random``.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import json
import random
import re
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.m2.common import choice_options, make_row, noul_options, read_jsonl, sha
from v2.data.m2.constructions import numeric_twins
from v2.data.m2.spec import FamilySpec
from v2.data.sources import ordinal, sgd
from v2.data.textnorm import normalize

Rows = list[dict[str, Any]]
Report = dict[str, Any]
Dirs = Mapping[str, Path]
Drops = collections.Counter[str]
Record = Mapping[str, Any]

MULTIWOZ_ROOT = "data/MultiWOZ_2.2"
WH_SLOT = re.compile(r"(?:what|which|how|when|where|who)\s", re.IGNORECASE)
TASKMASTER_ROOT = "TM-2-2020/data"
TASKMASTER_UTTERANCES = 12
TASKMASTER_SPEAKERS = {"USER": "User", "ASSISTANT": "Assistant"}
TASKMASTER_SUFFIX = re.compile(r"\.(?:accept|reject)$")
TASKMASTER_QUESTION = "In this conversation, does the user specify the {}?"
TASKMASTER_MAX_GAP = 1
DBPEDIA_CLASSES = (
    "Company",
    "Educational institution",
    "Artist",
    "Athlete",
    "Office holder",
    "Mean of transportation",
    "Building",
    "Natural place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written work",
)
DBPEDIA_PER_CLASS = 300
DBPEDIA_INSTRUCTIONS = "Which category does the subject of this text belong to?"
WINOGRANDE_INSTRUCTIONS = "Which option correctly fills the blank (_) in the sentence?"
CHOICE_INSTRUCTIONS = "Which option best answers the question in the state?"
AQUA_LETTERS = "ABCDE"
AQUA_PREFIX = re.compile(r"\s*([A-Ea-e])\s*\)\s*")
OR_WORD = re.compile(r"(?<!\w)or(?!\w)", re.IGNORECASE)
OR_SPLIT = re.compile(r"\s+or\s+", re.IGNORECASE)
DELIMITERS = ",;:"
INNER_BREAKS = ",;:?!"
EDGE = " ,;:?!.\"'"
ARTICLES = ("the ", "a ", "an ")
FUNCTION_WORDS = frozenset(
    "a an the of in on at to for with by than from and "
    "his her their its my your our this that these those".split()
)
QUESTION_WORDS = frozenset(
    "which what who whom whose where when why how is are was were do does did "
    "will would can could should has have had".split()
)
MAX_CANDIDATE_WORDS = 6
MAX_PARALLEL_WORDS = 4
SCITAIL_INSTRUCTIONS = "Does the premise support the hypothesis?"
SCITAIL_LABELS = {"entailment": 1, "entails": 1, "neutral": 0}
GSM8K_INSTRUCTIONS = "Is the final answer to this problem {}?"
GSM8K_INTEGER = re.compile(r"-?\d+")
MTOP_ARCHIVE = "mtop.zip"
MTOP_MEMBER = "mtop/{}/train.txt"
MTOP_LANGUAGES = ("de", "es", "fr", "hi", "th")
MTOP_COLUMNS = 8
MTOP_MIN_INTENTS = 3
MTOP_BALANCE = Fraction(6, 5)
MTOP_INSTRUCTIONS = "Which intent does the user's request express?"
MTOP_LICENCE = re.compile(r"licen[cs]e|copying", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class Job:
    """One family: source directory short name (also the group namespace),
    family name, licence-registry source id and seed."""

    name: str
    family: str
    source: str
    seed: str

    def row(self, **fields: Any) -> dict[str, Any]:
        return make_row(
            source=self.source,
            family=self.family,
            namespace=self.name,
            template=f"m2/{self.family}/v1",
            **fields,
        )


def _clean(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _files(root: Path, *patterns: str) -> list[Path]:
    found = {
        path for pattern in patterns for path in root.glob(pattern) if path.is_file()
    }
    if not found:
        raise FileNotFoundError(f"{root}: no TRAIN file matches {', '.join(patterns)}")
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def _inputs(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def _jsonl(paths: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in paths:
        for item in read_jsonl(path):
            yield path, item


def _report(
    inputs: Mapping[str, str],
    rows: Rows,
    items: int,
    drops: Mapping[str, int],
    **notes: Any,
) -> Report:
    return {
        "inputs": dict(inputs),
        "candidates": len(rows),
        "items": items,
        "drops": {reason: count for reason, count in sorted(drops.items()) if count},
        **notes,
    }


def _balanced(by_label: Mapping[Any, Rows], labels: Sequence[Any], seed: str) -> Rows:
    """The same number of rows per label (the rarest label's count), hash order."""
    quota = min(len(by_label.get(label, [])) for label in labels)
    kept: Rows = []
    for label in labels:
        members = sorted(
            by_label.get(label, []), key=lambda row: sha(f"{seed}:balance:{row['id']}")
        )
        kept.extend(members[:quota])
    return kept


# MultiWOZ 2.2 (SGD format)


def slot_question(description: str) -> str:
    """The SGD slot question; descriptions phrased as questions ("what is the
    type of the hotel") are asked directly."""
    text = _clean(description).rstrip(".")
    if WH_SLOT.match(text):
        return f"In the latest message, does the user ask {text[:1].lower()}{text[1:]}?"
    return sgd.slot_question(text)


def _multiwoz(
    dirs: Dirs, job: Job
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    root = dirs[job.name]
    schema = root / MULTIWOZ_ROOT / "schema.json"
    files = _files(root, f"{MULTIWOZ_ROOT}/train/dialogues_*.json")
    services = {service["service_name"]: service for service in _json(schema)}
    dialogues = [dialogue for path in files for dialogue in _json(path)]
    ids = [dialogue["dialogue_id"] for dialogue in dialogues]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate MultiWOZ dialogue_id")
    return _inputs(root, [schema, *files]), services, dialogues


def build_multiwoz_intent(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    inputs, services, dialogues = _multiwoz(dirs, job)
    usable: dict[str, list[str]] = {}
    for name, service in sorted(services.items()):
        described = [normalize(item["description"]) for item in service["intents"]]
        if len(described) >= 2 and len(set(described)) == len(described):
            usable[name] = [item["name"] for item in service["intents"]]
    drops: Drops = collections.Counter()
    picked: dict[str, dict[str, Rows]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for dialogue in dialogues:
        events = []
        for event in sgd.activations(dialogue):
            if event[1] not in usable:
                continue
            if event[2] not in usable[event[1]]:
                drops["activation_intent_not_in_schema"] += 1
                continue
            events.append(event)
        if not events:
            drops["dialogue_without_activation"] += 1
            continue
        dialogue_id = dialogue["dialogue_id"]
        index, service, intent = sgd._pick_event(
            events, f"{job.seed}:choice:{dialogue_id}"
        )
        picked[service][intent].append(
            job.row(
                task_type="choice",
                language="en",
                group_key=dialogue_id,
                local_id=f"{dialogue_id}:{index}:{service}",
                state=sgd.window(dialogue["turns"], index),
                instructions=sgd.CHOICE_INSTRUCTIONS,
                options=choice_options(
                    [
                        _clean(item["description"])
                        for item in services[service]["intents"]
                    ]
                ),
                label=usable[service].index(intent),
                audit={"turn_index": index, "service": service, "intent": intent},
                rotate_choice=True,
            )
        )
    rows: Rows = []
    per_intent: dict[str, int] = {}
    missing = []
    for service in sorted(picked):
        total = sum(len(members) for members in picked[service].values())
        kept = _balanced(picked[service], usable[service], job.seed)
        if not kept:
            missing.append(service)
            drops["service_with_unchosen_intent"] += total
            continue
        drops["intent_balance"] += total - len(kept)
        per_intent[service] = len(kept) // len(usable[service])
        rows.extend(kept)
    return rows, _report(
        inputs,
        rows,
        len(dialogues),
        drops,
        services_eligible=sorted(usable),
        services_ineligible=sorted(set(services) - set(usable)),
        services_with_unchosen_intent=missing,
        rows_per_intent=per_intent,
    )


def build_multiwoz_slot(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    inputs, services, dialogues = _multiwoz(dirs, job)
    drops: Drops = collections.Counter()
    requestable: dict[str, set[str]] = collections.defaultdict(set)
    accepted = []
    for dialogue in dialogues:
        events = []
        for index, service, requested in sgd.requests(dialogue):
            if service not in services:
                drops["request_unknown_service"] += 1
                continue
            names = {slot["name"] for slot in services[service]["slots"]}
            if not set(requested) <= names:
                drops["requested_slot_not_in_schema"] += 1
                continue
            events.append((index, service, requested))
            requestable[service].update(requested)
        accepted.append((dialogue, events))
    strata: dict[tuple[str, str], dict[int, Rows]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for dialogue, events in accepted:
        if not events:
            drops["dialogue_without_request"] += 1
            continue
        dialogue_id = dialogue["dialogue_id"]
        index, service, requested = sgd._pick_event(
            events, f"{job.seed}:noul:{dialogue_id}"
        )
        slots = {
            slot["name"]: _clean(slot["description"])
            for slot in services[service]["slots"]
        }
        local_id = f"{dialogue_id}:{index}:{service}"
        truth = int(sha(f"{job.seed}:truth:{dialogue_id}"), 16) % 2
        taken = {normalize(slots[name]) for name in requested}
        pool = (
            list(requested)
            if truth
            else sorted(
                name
                for name in requestable[service] - set(requested)
                if normalize(slots[name]) not in taken
            )
        )
        if not pool:
            drops["no_false_slot"] += 1
            continue
        asked = sgd._pick(pool, f"{job.seed}:slot:{local_id}")
        strata[(service, asked)][truth].append(
            job.row(
                task_type="noul",
                language="en",
                group_key=dialogue_id,
                local_id=local_id,
                state=sgd.window(dialogue["turns"], index),
                instructions=slot_question(slots[asked]),
                options=noul_options("en"),
                label=truth,
                audit={
                    "turn_index": index,
                    "service": service,
                    "asked_slot": asked,
                    "requested_slots": list(requested),
                },
            )
        )
    rows: Rows = []
    for key in sorted(strata):
        kept = _balanced(strata[key], [0, 1], job.seed)
        drops["slot_balance"] += sum(len(v) for v in strata[key].values()) - len(kept)
        rows.extend(kept)
    per_service = collections.Counter(row["audit_metadata"]["service"] for row in rows)
    return rows, _report(
        inputs,
        rows,
        len(dialogues),
        drops,
        requestable_slots={
            name: len(requestable[name]) for name in sorted(requestable)
        },
        rows_per_service=dict(sorted(per_service.items())),
    )


# Taskmaster-2


def readable_slot(name: str) -> str:
    return " ".join(name.replace("_", " ").replace(".", " ").split())


def _conversation(
    conversation: Mapping[str, Any],
) -> tuple[str, list[str], dict[str, None]] | str:
    """(state, slots a user utterance carries, slots any utterance carries)
    over the rendered first utterances, or the reason to drop the conversation."""
    shown = list(conversation.get("utterances") or [])[:TASKMASTER_UTTERANCES]
    lines: list[str] = []
    present: dict[str, None] = {}
    annotated: dict[str, None] = {}
    for utterance in shown:
        speaker = TASKMASTER_SPEAKERS.get(utterance.get("speaker"))
        if speaker is None:
            return "unknown_speaker"
        lines.append(f"{speaker}: {_clean(utterance.get('text'))}")
        for segment in utterance.get("segments") or []:
            for annotation in segment.get("annotations") or []:
                slot = TASKMASTER_SUFFIX.sub("", _clean(annotation.get("name")))
                if slot:
                    annotated[slot] = None
                    if utterance["speaker"] == "USER":
                        present[slot] = None
    if not present:
        return "no_user_slot"
    return "\n".join(lines), sorted(present), annotated


def _least(
    slots: Sequence[str], gap: Mapping[str, int], sign: int, rng: random.Random
) -> str | None:
    """A slot minimizing ``sign * gap`` among those still below the gap bound
    (ties broken by the seeded RNG), or None when every slot is at the bound."""
    allowed = [slot for slot in slots if sign * gap[slot] < TASKMASTER_MAX_GAP]
    if not allowed:
        return None
    best = min(sign * gap[slot] for slot in allowed)
    return rng.choice([slot for slot in allowed if sign * gap[slot] == best])


def build_taskmaster(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, f"{TASKMASTER_ROOT}/*.json")
    drops: Drops = collections.Counter()
    rows: Rows = []
    domains: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    items = 0
    for path in files:
        domain = path.stem
        conversations = _json(path)
        items += len(conversations)
        usable = []
        for conversation in conversations:
            ident = str(conversation["conversation_id"])
            if ident in seen:
                drops["duplicate_conversation"] += 1
                continue
            seen.add(ident)
            parsed = _conversation(conversation)
            if isinstance(parsed, str):
                drops[parsed] += 1
                continue
            usable.append((ident, *parsed))
        inventory = sorted({slot for _, _, present, _ in usable for slot in present})
        gap: collections.Counter[str] = collections.Counter()
        made = 0
        for ident, state, present, annotated in sorted(
            usable, key=lambda item: sha(f"{job.seed}:{domain}:{item[0]}")
        ):
            blocked = {readable_slot(slot) for slot in annotated}
            absent = [slot for slot in inventory if readable_slot(slot) not in blocked]
            if not absent:
                drops["no_absent_slot"] += 1
                continue
            rng = random.Random(f"{job.seed}:{ident}")
            true_slot = _least(present, gap, 1, rng)
            false_slot = _least(absent, gap, -1, rng)
            if true_slot is None or false_slot is None:
                drops["slot_balance"] += 1
                continue
            gap[true_slot] += 1
            gap[false_slot] -= 1
            for label, slot, twin in ((1, true_slot, "true"), (0, false_slot, "false")):
                rows.append(
                    job.row(
                        task_type="noul",
                        language="en",
                        group_key=ident,
                        local_id=f"{ident}:{twin}",
                        state=state,
                        instructions=TASKMASTER_QUESTION.format(readable_slot(slot)),
                        options=noul_options("en"),
                        label=label,
                        audit={"domain": domain, "slot": slot, "twin": twin},
                    )
                )
            made += 2
        domains[domain] = {
            "conversations": len(conversations),
            "usable": len(usable),
            "rows": made,
            "inventory": len(inventory),
            "asked_slots": len(gap),
            "slot_gap_total": sum(abs(value) for value in gap.values()),
            "slot_gap_max": max((abs(value) for value in gap.values()), default=0),
        }
    return rows, _report(_inputs(root, files), rows, items, drops, domains=domains)


# DBpedia-14 (C6)


def build_dbpedia(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, "dbpedia_14/train-*.jsonl")
    drops: Drops = collections.Counter()
    buckets: list[list[tuple[str, str, str]]] = [[] for _ in DBPEDIA_CLASSES]
    items = 0
    for _, item in _jsonl(files):
        local_id = f"db-{items}"
        items += 1
        label = item.get("label")
        if type(label) is not int or not 0 <= label < len(DBPEDIA_CLASSES):
            drops["unknown_label"] += 1
            continue
        title, content = _clean(item.get("title")), _clean(item.get("content"))
        if not title or not content:
            drops["empty_text"] += 1
            continue
        buckets[label].append((local_id, title, content))
    quota = min(DBPEDIA_PER_CLASS, *map(len, buckets))
    rows: Rows = []
    for label, bucket in enumerate(buckets):
        chosen = sorted(bucket, key=lambda entry: sha(f"{job.seed}:{entry[0]}"))
        drops["class_quota"] += len(bucket) - quota
        for local_id, title, content in chosen[:quota]:
            rows.append(
                job.row(
                    task_type="choice",
                    language="en",
                    group_key=normalize(title),
                    local_id=local_id,
                    state=f"{title}\n\n{content}",
                    instructions=DBPEDIA_INSTRUCTIONS,
                    options=choice_options(list(DBPEDIA_CLASSES)),
                    label=label,
                    rotate_choice=True,
                )
            )
    return rows, _report(
        _inputs(root, files),
        rows,
        items,
        drops,
        per_class=quota,
        class_sizes=[len(bucket) for bucket in buckets],
    )


# WinoGrande (C7)


def _winogrande_item(item: Mapping[str, Any]) -> bool:
    first, second = _clean(item.get("option1")), _clean(item.get("option2"))
    return (
        str(item.get("answer")) in ("1", "2")
        and "_" in _clean(item.get("sentence"))
        and bool(first and second)
        and normalize(first) != normalize(second)
    )


def winogrande_twins(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Consecutive rows with the same option pair, flipped answers and
    different sentences."""
    return (
        _winogrande_item(first)
        and _winogrande_item(second)
        and _clean(first["option1"]) == _clean(second["option1"])
        and _clean(first["option2"]) == _clean(second["option2"])
        and str(first["answer"]) != str(second["answer"])
        and _clean(first["sentence"]) != _clean(second["sentence"])
    )


def build_winogrande(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, "winogrande_xl/train-*.jsonl")
    items = [item for _, item in _jsonl(files)]
    drops: Drops = collections.Counter()
    rows: Rows = []
    index = 0
    while index < len(items):
        if index + 1 < len(items) and winogrande_twins(items[index], items[index + 1]):
            pair = f"wg-{index}"
            for offset in (0, 1):
                item = items[index + offset]
                rows.append(
                    job.row(
                        task_type="choice",
                        language="en",
                        group_key=pair,
                        local_id=f"wg-{index + offset}",
                        state=_clean(item["sentence"]),
                        instructions=WINOGRANDE_INSTRUCTIONS,
                        options=choice_options(
                            [_clean(item["option1"]), _clean(item["option2"])]
                        ),
                        label=int(item["answer"]) - 1,
                        audit={"pair": pair, "twin": offset},
                    )
                )
            index += 2
        else:
            drops["unpaired_row"] += 1
            index += 1
    return rows, _report(
        _inputs(root, files), rows, len(items), drops, pairs=len(rows) // 2
    )


# Generic multiple choice (C8)


def _gold(labels: Sequence[str], answer: Any) -> int | None:
    key = answer.strip() if isinstance(answer, str) else ""
    return labels.index(key) if key and labels.count(key) == 1 else None


def _choice_problem(state: str, options: Sequence[str], gold: int | None) -> str | None:
    if not state:
        return "empty_question"
    if gold is None:
        return "answer_not_in_labels"
    if len(options) < 2:
        return "too_few_options"
    if not all(options):
        return "empty_option"
    if len({normalize(option) for option in options}) != len(options):
        return "duplicate_options"
    return None


def _choice_rows(job: Job, records: Iterable[Record | str], drops: Drops) -> Rows:
    """Rotated Choice rows from normalized records; a string record is a drop."""
    rows: Rows = []
    seen: set[str] = set()
    for record in records:
        if isinstance(record, str):
            drops[record] += 1
            continue
        local_id = record["local_id"]
        if local_id in seen:
            drops["duplicate_id"] += 1
            continue
        seen.add(local_id)
        problem = _choice_problem(record["state"], record["options"], record["gold"])
        if problem:
            drops[problem] += 1
            continue
        rows.append(
            job.row(
                task_type="choice",
                language="en",
                group_key=record["group_key"],
                local_id=local_id,
                state=record["state"],
                instructions=record.get("instructions", CHOICE_INSTRUCTIONS),
                options=choice_options(record["options"]),
                label=record["gold"],
                audit=record.get("audit"),
                rotate_choice=True,
            )
        )
    return rows


def _labelled_records(
    files: Sequence[Path], *, question: str, context: str | None, subset: bool
) -> Iterator[Record | str]:
    """One record or drop reason per source item."""
    for path, item in _jsonl(files):
        choices = item.get("choices")
        labels = choices.get("label") if isinstance(choices, Mapping) else None
        texts = choices.get("text") if isinstance(choices, Mapping) else None
        if (
            not isinstance(labels, list)
            or not isinstance(texts, list)
            or len(labels) != len(texts)
        ):
            yield "malformed_choices"
            continue
        ident = str(item["id"])
        text = _clean(item.get(question))
        record = {
            "local_id": ident,
            "group_key": ident,
            "state": text,
            "options": [_clean(option) for option in texts],
            "gold": _gold(
                [str(label).strip() for label in labels], item.get("answerKey")
            ),
            "audit": {"subset": path.parent.name} if subset else None,
        }
        if context is not None:
            passage = _clean(item.get(context))
            if not passage:
                yield "empty_context"
                continue
            record["state"] = f"{passage}\n\n{text}" if text else ""
            record["group_key"] = normalize(passage)
        yield record


def build_labelled(
    dirs: Dirs,
    *,
    job: Job,
    patterns: Sequence[str],
    question: str,
    context: str | None = None,
    subset: bool = False,
) -> tuple[Rows, Report]:
    """CommonsenseQA, ARC, OpenBookQA and QuaRTz: ``choices{label,text}`` plus
    ``answerKey``; the state is the question, after the context when given."""
    root = dirs[job.name]
    files = _files(root, *patterns)
    drops: Drops = collections.Counter()
    records = list(
        _labelled_records(files, question=question, context=context, subset=subset)
    )
    rows = _choice_rows(job, records, drops)
    return rows, _report(_inputs(root, files), rows, len(records), drops)


def aqua_options(raw: Any) -> list[str] | None:
    """The five option texts without their "A)".."E)" prefixes (repeated
    prefixes too), or None unless every option starts with its own letter."""
    if not isinstance(raw, list) or len(raw) != len(AQUA_LETTERS):
        return None
    texts = []
    for letter, option in zip(AQUA_LETTERS, raw):
        if not isinstance(option, str):
            return None
        text, stripped = option, False
        while (match := AQUA_PREFIX.match(text)) and match.group(1).upper() == letter:
            text, stripped = text[match.end() :], True
        if not stripped:
            return None
        texts.append(_clean(text))
    return texts


def build_aquarat(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, "raw/train-*.jsonl")
    drops: Drops = collections.Counter()
    records: list[Record] = []
    seen: set[tuple[Any, ...]] = set()
    items = 0
    for _, item in _jsonl(files):
        local_id = f"aqua-{items}"
        items += 1
        options = aqua_options(item.get("options"))
        if options is None:
            drops["malformed_options"] += 1
            continue
        question = _clean(item.get("question"))
        correct = _clean(item.get("correct")).upper()
        gold = (
            AQUA_LETTERS.index(correct)
            if len(correct) == 1 and correct in AQUA_LETTERS
            else None
        )
        key = (normalize(question), tuple(map(normalize, options)), gold)
        if key in seen:
            drops["duplicate_item"] += 1
            continue
        seen.add(key)
        records.append(
            {
                "local_id": local_id,
                "group_key": normalize(question),
                "state": question,
                "options": options,
                "gold": gold,
            }
        )
    rows = _choice_rows(job, records, drops)
    return rows, _report(_inputs(root, files), rows, items, drops)


def candidate_key(text: str) -> str:
    """Comparison form of an answer candidate: normalized, edge punctuation
    and one leading article removed."""
    value = normalize(text).strip(EDGE)
    for article in ARTICLES:
        if value.startswith(article):
            return value[len(article) :].strip()
    return value


def _candidate_pair(first: str, second: str, gold: str) -> bool:
    keys = (candidate_key(first), candidate_key(second))
    if not all(keys) or keys[0] == keys[1] or keys.count(gold) != 1:
        return False
    for text in (first, second):
        words = text.split()
        if (
            not 1 <= len(words) <= MAX_CANDIDATE_WORDS
            or text in FUNCTION_WORDS
            or words[0].lower() in QUESTION_WORDS
        ):
            return False
    return True


def or_candidates(question: str, answer: str) -> tuple[list[str], int] | str:
    """The two alternatives joined by the question's only "or" and the gold
    position, or the reason to drop the question.

    Alternatives are either delimited ("..., A or B?" after the last , ; or :
    before "or", at most two words apart in length) or parallel (the n words
    on each side of "or", n <= 4, sharing a word when n > 1). Exactly one
    distinct pair may satisfy: exactly one alternative equals the gold answer,
    neither is a function word or starts with a question word, and neither
    crosses a clause break.
    """
    text = _clean(question)
    count = len(OR_WORD.findall(text))
    if count != 1:
        return "no_or" if count == 0 else "multiple_or"
    parts = OR_SPLIT.split(text)
    if len(parts) != 2:
        return "or_not_between_words"
    left, right = parts
    gold = candidate_key(answer)
    if not gold:
        return "empty_answer"
    found: dict[tuple[str, str], list[str]] = {}

    def offer(first: str, second: str) -> None:
        if _candidate_pair(first, second, gold):
            found.setdefault(
                (candidate_key(first), candidate_key(second)), [first, second]
            )

    cut = max(left.rfind(mark) for mark in DELIMITERS)
    tail = right.rstrip(" ?!.")
    if cut >= 0 and not any(mark in tail for mark in INNER_BREAKS):
        head = left[cut + 1 :].strip()
        if head.lower().startswith("either "):
            head = head[len("either ") :]
        if abs(len(head.split()) - len(tail.split())) <= 2:
            offer(head.strip(EDGE), tail.strip(EDGE))
    before, after = left.split(), right.split()
    for size in range(1, MAX_PARALLEL_WORDS + 1):
        if size > len(before) or size > len(after):
            break
        first, second = before[-size:], after[:size]
        if any(
            mark in word for word in first[:-1] + second[:-1] for mark in INNER_BREAKS
        ):
            break
        if size > 1 and not {w.casefold() for w in first} & {
            w.casefold() for w in second
        }:
            continue
        offer(" ".join(first).strip(EDGE), " ".join(second).strip(EDGE))
    if not found:
        return "gold_not_a_candidate"
    if len(found) > 1:
        return "ambiguous_candidates"
    options = next(iter(found.values()))
    return options, [candidate_key(option) for option in options].index(gold)


def build_ropes(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, "plain_text/train-*.jsonl")
    drops: Drops = collections.Counter()
    records: list[Record] = []
    items = 0
    for _, item in _jsonl(files):
        items += 1
        answers = item.get("answers")
        texts = answers.get("text") if isinstance(answers, Mapping) else None
        texts = [
            text
            for text in texts or []
            if isinstance(text, str) and candidate_key(text)
        ]
        if len(dict.fromkeys(map(candidate_key, texts))) != 1:
            drops["answer_count"] += 1
            continue
        background = _clean(item.get("background"))
        situation = _clean(item.get("situation"))
        if not background or not situation:
            drops["empty_context"] += 1
            continue
        found = or_candidates(item.get("question"), texts[0])
        if isinstance(found, str):
            drops[found] += 1
            continue
        options, gold = found
        records.append(
            {
                "local_id": str(item["id"]),
                "group_key": f"{normalize(background)}\n\n{normalize(situation)}",
                "state": f"{background}\n\n{situation}",
                "instructions": _clean(item.get("question")),
                "options": options,
                "gold": gold,
            }
        )
    rows = _choice_rows(job, records, drops)
    return rows, _report(_inputs(root, files), rows, items, drops)


# SciTail (Noul)


def build_scitail(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    """Entails vs neutral, balanced within every hypothesis; a group is a
    connected set of rows sharing premises or hypotheses."""
    root = dirs[job.name]
    files = _files(root, "snli_format/train-*.jsonl")
    drops: Drops = collections.Counter()
    pairs: dict[tuple[str, str], list[tuple[str, str, str, int]]] = {}
    items = 0
    for _, item in _jsonl(files):
        local_id = f"scitail-{items}"
        items += 1
        label = SCITAIL_LABELS.get(_clean(item.get("gold_label")).lower())
        if label is None:
            drops["unknown_label"] += 1
            continue
        premise = _clean(item.get("sentence1"))
        hypothesis = _clean(item.get("sentence2"))
        if not premise or not hypothesis:
            drops["empty_text"] += 1
            continue
        pairs.setdefault((normalize(premise), normalize(hypothesis)), []).append(
            (local_id, premise, hypothesis, label)
        )
    by_hypothesis: dict[str, tuple[list[Any], list[Any]]] = {}
    for (_, hypothesis_key), members in pairs.items():
        if len({member[3] for member in members}) > 1:
            drops["conflicting_duplicate"] += len(members)
            continue
        drops["duplicate_pair"] += len(members) - 1
        by_hypothesis.setdefault(hypothesis_key, ([], []))[members[0][3]].append(
            members[0]
        )
    kept = []
    for buckets in by_hypothesis.values():
        quota = min(map(len, buckets))
        drops["hypothesis_balance"] += sum(map(len, buckets)) - 2 * quota
        for bucket in buckets:
            kept.extend(sorted(bucket, key=lambda m: sha(f"{job.seed}:{m[0]}"))[:quota])
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for _, premise, hypothesis, _ in kept:
        first = find("p:" + normalize(premise))
        second = find("h:" + normalize(hypothesis))
        if first != second:
            parent[max(first, second)] = min(first, second)
    group_of: dict[str, str] = {}
    for _, premise, _, _ in kept:
        node, key = find("p:" + normalize(premise)), normalize(premise)
        group_of[node] = min(group_of.get(node, key), key)
    rows = [
        job.row(
            task_type="noul",
            language="en",
            group_key=group_of[find("p:" + normalize(premise))],
            local_id=local_id,
            state={"premise": premise, "hypothesis": hypothesis},
            instructions=SCITAIL_INSTRUCTIONS,
            options=noul_options("en"),
            label=label,
        )
        for local_id, premise, hypothesis, label in kept
    ]
    sizes = collections.Counter(row["group_id"] for row in rows)
    return rows, _report(
        _inputs(root, files),
        rows,
        items,
        drops,
        hypotheses=sum(1 for buckets in by_hypothesis.values() if all(buckets)),
        components=len(sizes),
        largest_component=max(sizes.values(), default=0),
    )


# GSM8K (C9)


def gsm8k_answer(solution: Any) -> tuple[str | None, str | None]:
    """(canonical integer after the last "####", drop reason)."""
    if not isinstance(solution, str) or "####" not in solution:
        return None, "no_final_answer"
    value = solution.rsplit("####", 1)[1].strip().replace(",", "")
    if not GSM8K_INTEGER.fullmatch(value):
        return None, "non_integer_answer"
    return str(int(value)), None


def build_gsm8k(dirs: Dirs, *, job: Job) -> tuple[Rows, Report]:
    root = dirs[job.name]
    files = _files(root, "main/train-*.jsonl")
    drops: Drops = collections.Counter()
    records = []
    items = 0
    for _, item in _jsonl(files):
        local_id = f"gsm-{items}"
        items += 1
        problem = _clean(item.get("question"))
        if not problem:
            drops["empty_question"] += 1
            continue
        answer, reason = gsm8k_answer(item.get("answer"))
        if answer is None:
            drops[str(reason)] += 1
            continue
        records.append(
            {
                "local_id": local_id,
                "group_key": local_id,
                "problem": problem,
                "answer": answer,
            }
        )
    rows, twin_drops = numeric_twins(
        records,
        source=job.source,
        family=job.family,
        namespace=job.name,
        template=f"m2/{job.family}/v1",
        seed=job.seed,
        instructions=GSM8K_INSTRUCTIONS,
    )
    drops.update(twin_drops)
    return rows, _report(
        _inputs(root, files), rows, items, drops, problems=len(records)
    )


# MTOP (C6, H5)


def intent_description(intent: str) -> str:
    return intent.removeprefix("IN:").lower().replace("_", " ")


def _mtop_member(
    archive: zipfile.ZipFile, language: str
) -> tuple[list[list[str]], str, int]:
    """TSV rows of a language's TRAIN member, the member's SHA-256 and the
    number of lines without exactly eight columns."""
    data = archive.read(MTOP_MEMBER.format(language))
    rows, malformed = [], 0
    for line in data.decode("utf-8").split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != MTOP_COLUMNS:
            malformed += 1
            continue
        rows.append(fields)
    return rows, hashlib.sha256(data).hexdigest(), malformed


def build_mtop(dirs: Dirs, *, job: Job, language: str) -> tuple[Rows, Report]:
    """Intent Choice over the English TRAIN intents of the utterance's domain;
    at most 1.2x the rarest present intent per intent within the domain."""
    path = dirs[job.name] / MTOP_ARCHIVE
    with zipfile.ZipFile(path) as archive:
        english, english_sha, english_malformed = _mtop_member(archive, "en")
        native, native_sha, native_malformed = _mtop_member(archive, language)
        licences = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if MTOP_LICENCE.search(name.rsplit("/", 1)[-1])
        }
    inventory: dict[str, dict[str, None]] = {}
    for fields in english:
        if fields[1].startswith("IN:"):
            inventory.setdefault(fields[4], {})[fields[1]] = None
    options = {
        domain: sorted(intents)
        for domain, intents in sorted(inventory.items())
        if len(intents) >= MTOP_MIN_INTENTS
    }
    drops: Drops = collections.Counter({"malformed_line": native_malformed})
    candidates: list[tuple[str, str, str, str, str, str]] = []
    seen: set[str] = set()
    for fields in native:
        ident, intent, _, utterance, domain, locale = (
            field.strip() for field in fields[:6]
        )
        if ident in seen:
            drops["duplicate_id"] += 1
            continue
        seen.add(ident)
        if domain not in inventory:
            drops["domain_not_in_english"] += 1
        elif domain not in options:
            drops["domain_too_few_intents"] += 1
        elif intent not in options[domain]:
            drops["intent_not_in_english_domain"] += 1
        elif not _clean(utterance):
            drops["empty_utterance"] += 1
        else:
            candidates.append(
                (
                    f"{language}:{ident}",
                    ident,
                    _clean(utterance),
                    domain,
                    intent,
                    locale,
                )
            )
    present: dict[str, dict[str, None]] = {}
    for candidate in candidates:
        present.setdefault(candidate[3], {})[candidate[4]] = None
    levels = {
        domain: {intent: i for i, intent in enumerate(sorted(intents))}
        for domain, intents in present.items()
    }
    eligible = [c for c in candidates if len(levels[c[3]]) >= 2]
    drops["single_intent_domain"] += len(candidates) - len(eligible)
    kept, balance = ordinal.balance(
        eligible,
        cell=lambda c: c[3],
        level=lambda c: levels[c[3]][c[4]],
        levels=lambda c: len(levels[c[3]]),
        ident=lambda c: c[0],
        seed=job.seed,
        ratio=MTOP_BALANCE,
    )
    drops["intent_balance"] += len(eligible) - len(kept)
    rows = [
        job.row(
            task_type="choice",
            language=language,
            group_key=ident,
            local_id=local_id,
            state=utterance,
            instructions=MTOP_INSTRUCTIONS,
            options=choice_options(
                [intent_description(name) for name in options[domain]]
            ),
            label=options[domain].index(intent),
            audit={"domain": domain, "intent": intent, "locale": locale},
            rotate_choice=True,
        )
        for local_id, ident, utterance, domain, intent, locale in kept
    ]
    return rows, _report(
        {MTOP_ARCHIVE: file_sha256(path)},
        rows,
        len(native) + native_malformed,
        drops,
        members={
            MTOP_MEMBER.format("en"): english_sha,
            MTOP_MEMBER.format(language): native_sha,
        },
        english_malformed_lines=english_malformed,
        licence_members=licences,
        domains={
            domain: {
                "options": len(options[domain]),
                "intents": list(levels[domain]),
                **balance[domain],
            }
            for domain in sorted(balance)
        },
    )


def _spec(
    arm: str,
    cap_rows: int,
    builder: Callable[..., tuple[Rows, Report]],
    job: Job,
    **options: Any,
) -> FamilySpec:
    return FamilySpec(
        arm=arm,
        family=job.family,
        source=job.source,
        build=functools.partial(builder, job=job, **options),
        cap_rows=cap_rows,
        seed=job.seed,
    )


FAMILIES: tuple[FamilySpec, ...] = (
    _spec(
        "h1",
        4000,
        build_multiwoz_intent,
        Job("multiwoz", "multiwoz_intent", "multiwoz22_train", "h1-multiwoz-intent-v1"),
    ),
    _spec(
        "h1",
        4000,
        build_multiwoz_slot,
        Job("multiwoz", "multiwoz_slot", "multiwoz22_train", "h1-multiwoz-slot-v1"),
    ),
    _spec(
        "h1",
        4000,
        build_taskmaster,
        Job(
            "taskmaster",
            "taskmaster2_slot",
            "taskmaster2_train",
            "h1-taskmaster2-slot-v1",
        ),
    ),
    _spec(
        "h1",
        4200,
        build_dbpedia,
        Job("dbpedia14", "dbpedia14", "dbpedia14_train", "h1-dbpedia14-v1"),
    ),
    _spec(
        "h1",
        20000,
        build_winogrande,
        Job(
            "winogrande", "winogrande_twins", "winogrande_xl_train", "h1-winogrande-v1"
        ),
    ),
    _spec(
        "h1",
        9000,
        build_labelled,
        Job("csqa", "csqa", "csqa_train", "h1-csqa-v1"),
        patterns=("data/train-*.jsonl",),
        question="question",
    ),
    _spec(
        "h1",
        3500,
        build_labelled,
        Job("arc", "arc", "arc_train", "h1-arc-v1"),
        patterns=("ARC-Challenge/train-*.jsonl", "ARC-Easy/train-*.jsonl"),
        question="question",
        subset=True,
    ),
    _spec(
        "h1",
        5000,
        build_labelled,
        Job("obqa", "obqa", "obqa_main_train", "h1-obqa-v1"),
        patterns=("main/train-*.jsonl",),
        question="question_stem",
    ),
    _spec(
        "h1",
        10000,
        build_aquarat,
        Job("aquarat", "aquarat", "aquarat_train", "h1-aquarat-v1"),
    ),
    _spec(
        "h1",
        2700,
        build_labelled,
        Job("quartz", "quartz", "quartz_train", "h1-quartz-v1"),
        patterns=("data/train-*.jsonl",),
        question="question",
        context="para",
    ),
    _spec("h1", 8000, build_ropes, Job("ropes", "ropes", "ropes_train", "h1-ropes-v1")),
    _spec(
        "h1",
        10000,
        build_scitail,
        Job("scitail", "scitail", "scitail_train", "h1-scitail-v1"),
    ),
    _spec(
        "h1",
        16000,
        build_gsm8k,
        Job("gsm8k", "gsm8k_twins", "gsm8k_train", "h1-gsm8k-v1"),
    ),
    *(
        _spec(
            "h5",
            2000,
            build_mtop,
            Job(
                "mtop",
                f"mtop_intent_{language}",
                "mtop_train",
                f"h5-mtop-{language}-v1",
            ),
            language=language,
        )
        for language in MTOP_LANGUAGES
    ),
)
