"""Build the A7 encoder-family sub-arms (`a7-enc10-v3`) from pinned human-labelled sources.

Rules: `records/a7-enc10-prereg-2026-09-28.md` with amendment 1 (A7x
distractors from the same scenario only) and amendment 2 (SentiMix Hinglish in
A7s). States are rendered from the
upstream files; instructions and option descriptions are the fixed English
templates below; labels are the publishers' human labels. No 1.0 roster text is
read. Writes prelim/<sub>.{train,aho}.jsonl and a count-only build manifest for
the A7 screens (`run_nodeB.sh` lengths/screens/admit/post/freeze/isolation).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.a7.build_a7 import (
    Components,
    _external_sets,
    _json_bytes,
    _write_new,
    is_aho,
    keyed,
    sha256_file,
    state_key,
)
from v2.data.freeze import canonical_jsonl
from v2.data.textnorm import normalize

VERSION = "a7-enc10-v3"
ENC_SUB_ARMS = ("A7q", "A7k", "A7s", "A7x")
TIE_BAND = 0.1
BALANCE_RATIO = 1.2
MASSIVE_LOCALES = (
    "ar-SA", "en-US", "es-ES", "hi-IN", "id-ID", "ja-JP",
    "ko-KR", "sw-KE", "ta-IN", "tr-TR", "vi-VN", "zh-CN",
)  # fmt: skip
MASSIVE_PER_LOCALE = 4000

OASST_AXES = {
    "quality": (
        "Rate the overall quality of the final assistant reply in this conversation.",
        [
            "Very low quality",
            "Low quality",
            "Moderate quality",
            "High quality",
            "Very high quality",
        ],
    ),
    "helpfulness": (
        "Rate how helpful the final assistant reply is for what the user asked.",
        [
            "Not helpful",
            "Slightly helpful",
            "Moderately helpful",
            "Very helpful",
            "Extremely helpful",
        ],
    ),
    "humor": (
        "Rate how humorous the final assistant reply is.",
        [
            "Not humorous",
            "Slightly humorous",
            "Moderately humorous",
            "Very humorous",
            "Extremely humorous",
        ],
    ),
    "creativity": (
        "Rate how creative the final assistant reply is.",
        [
            "Not creative",
            "Slightly creative",
            "Moderately creative",
            "Very creative",
            "Extremely creative",
        ],
    ),
}
OASST_SUFFIX = (
    " The levels follow the average of human raters' judgments on a five-point scale."
)
STS_INSTRUCTIONS = (
    "Rate how similar the two sentences are in meaning. The levels follow the average "
    "of human annotators' similarity scores from 0 to 5."
)
STS_LEVELS = [
    "0: The two sentences are unrelated in meaning.",
    "1: The sentences share a topic but say different things.",
    "2: The sentences share some details, but their main meaning differs.",
    "3: The main meaning is roughly the same, but important information differs or is missing.",
    "4: The sentences mean nearly the same thing and differ only in minor details.",
    "5: The sentences mean exactly the same thing.",
]
SENTIMENT_INSTRUCTIONS = "Rate the overall sentiment that the message expresses."
SENTIMENT_LEVELS = [
    "Negative: the message expresses a negative feeling or opinion.",
    "Neutral: the message expresses neither a positive nor a negative feeling or opinion.",
    "Positive: the message expresses a positive feeling or opinion.",
]
SENTIMENT_LABELS = {"negative": 0, "neutral": 1, "positive": 2}
MASSIVE_INSTRUCTIONS = (
    "Which intent does the user's request to the voice assistant express?"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_verified(entry: Mapping[str, Any]) -> bytes:
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{entry['name']}: SHA-256 differs from the spec")
    return path.read_bytes()


def level_or_none(value: float, scale: int) -> int | None:
    """round(value) on [0, scale], or None within TIE_BAND of a level boundary."""
    if not 0 <= value <= scale:
        return None
    lower = math.floor(value)
    if lower < scale and abs(value - (lower + 0.5)) < TIE_BAND - 1e-9:
        return None
    return int(math.floor(value + 0.5))


def language_code(tag: str) -> str:
    return tag.split("-")[0].split("_")[0].lower()


def score_row(
    *,
    sub_arm: str,
    family: str,
    source: str,
    upstream_id: str,
    group: str,
    language: str,
    state: str,
    instructions: str,
    levels: Sequence[str],
    label: int,
    audit: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    return _row(
        sub_arm=sub_arm,
        family=family,
        source=source,
        upstream_id=upstream_id,
        group=group,
        language=language,
        state=state,
        instructions=instructions,
        options=[{"key": str(i), "description": text} for i, text in enumerate(levels)],
        label=label,
        task_type="score",
        audit=audit,
        entry=entry,
    )


def _row(
    *,
    sub_arm: str,
    family: str,
    source: str,
    upstream_id: str,
    group: str,
    language: str,
    state: str,
    instructions: str,
    options: list[dict[str, str]],
    label: int,
    task_type: str,
    audit: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "id": f"a7:{sub_arm}:{family}:{upstream_id}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": f"a7:{sub_arm}:{group}",
        "language": language,
        "split": "train",
        "source": source,
        "evaluation_role": "train",
        "render_template": f"a7_{family}_v1",
        "audit_metadata": {
            "a7": {
                "version": VERSION,
                "sub_arm": sub_arm,
                "source_file": entry["name"],
                "source_file_sha256": entry["sha256"],
                "upstream_id": upstream_id,
                **audit,
            }
        },
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    validate_row(row, "train")
    return row


# --- A7q: OASST1 reply ratings ---------------------------------------------


def oasst_rows(
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], collections.Counter]:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(read_verified(entry))).to_pylist()
    by_id = {message["message_id"]: message for message in table}
    skipped: collections.Counter[str] = collections.Counter()
    rows = []
    for message in table:
        if message["role"] != "assistant":
            continue
        if message["deleted"] or message["synthetic"] or not message["review_result"]:
            skipped["message_filtered"] += 1
            continue
        chain, parent = [], message["parent_id"]
        while parent is not None:
            ancestor = by_id.get(parent)
            if ancestor is None or ancestor["deleted"]:
                chain = None
                break
            chain.append(ancestor)
            parent = ancestor["parent_id"]
        if chain is None:
            skipped["ancestor_missing_or_deleted"] += 1
            continue
        labels = message["labels"] or {"name": [], "value": [], "count": []}
        named = {
            name: (value, count)
            for name, value, count in zip(
                labels["name"], labels["value"], labels["count"]
            )
        }
        if named.get("pii", (0, 0))[0] > 0 or named.get("spam", (0, 0))[0] >= 0.5:
            skipped["pii_or_spam"] += 1
            continue
        turns = [
            f"{'User' if turn['role'] == 'prompter' else 'Assistant'}: {turn['text'].strip()}"
            for turn in reversed(chain)
        ]
        state = (
            "Conversation:\n"
            + "\n".join(turns)
            + "\n\nFinal assistant reply:\n"
            + message["text"].strip()
        )
        for axis, (instructions, levels) in OASST_AXES.items():
            if axis not in named or named[axis][1] < 1:
                continue
            value, count = named[axis]
            level = level_or_none(4 * value, 4)
            if level is None:
                skipped[f"tie_{axis}"] += 1
                continue
            rows.append(
                score_row(
                    sub_arm="A7q",
                    family=f"oasst1_{axis}",
                    source="a7:oasst1_train",
                    upstream_id=message["message_id"],
                    group=f"oasst1:{message['message_tree_id']}",
                    language=language_code(message["lang"]),
                    state=state,
                    instructions=instructions + OASST_SUFFIX,
                    levels=levels,
                    label=level,
                    audit={
                        "message_tree_id": message["message_tree_id"],
                        "axis": axis,
                        "rater_count": count,
                        "rater_mean": value,
                        "upstream_lang": message["lang"],
                    },
                    entry=entry,
                )
            )
    return rows, skipped


# --- A7k: KLUE-STS and JSTS ---------------------------------------------------


def sts_pairs(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = read_verified(entry).decode("utf-8")
    if entry["name"] == "klue_sts":
        return [
            {
                "upstream_id": item["guid"],
                "s1": item["sentence1"],
                "s2": item["sentence2"],
                "score": float(item["labels"]["real-label"]),
                "language": "ko",
                "family": "klue_sts",
                "source": "a7:klue_sts_v1.1_train",
                "extra": {"klue_source": item["source"]},
            }
            for item in json.loads(raw)
        ]
    return [
        {
            "upstream_id": str(item["sentence_pair_id"]),
            "s1": item["sentence1"],
            "s2": item["sentence2"],
            "score": float(item["label"]),
            "language": "ja",
            "family": "jglue_jsts",
            "source": "a7:jglue_jsts_v1.3_train",
            "extra": {"yjcaptions_id": item["yjcaptions_id"]},
        }
        for item in map(json.loads, filter(str.strip, raw.splitlines()))
    ]


def sts_rows(
    entries: Sequence[Mapping[str, Any]], excluded_ids: Mapping[str, set[str]]
) -> tuple[list[dict[str, Any]], collections.Counter]:
    skipped: collections.Counter[str] = collections.Counter()
    pairs: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    blocked_sentences: set[str] = set()
    for entry in entries:
        for pair in sts_pairs(entry):
            if pair["upstream_id"] in excluded_ids.get(pair["family"], set()):
                blocked_sentences.update({normalize(pair["s1"]), normalize(pair["s2"])})
                skipped[f"{pair['family']}_used_by_data_track"] += 1
            else:
                pairs.append((pair, entry))
    components = Components()
    for pair, _ in pairs:
        components.union(normalize(pair["s1"]), normalize(pair["s2"]))
    rows = []
    for pair, entry in pairs:
        if {normalize(pair["s1"]), normalize(pair["s2"])} & blocked_sentences:
            skipped[f"{pair['family']}_shares_sentence_with_data_track"] += 1
            continue
        level = level_or_none(pair["score"], 5)
        if level is None:
            skipped[f"{pair['family']}_tie"] += 1
            continue
        root = components.find(normalize(pair["s1"]))
        rows.append(
            score_row(
                sub_arm="A7k",
                family=pair["family"],
                source=pair["source"],
                upstream_id=pair["upstream_id"],
                group=f"{pair['family']}:{_sha(root)[:24]}",
                language=pair["language"],
                state=f"Sentence 1: {pair['s1'].strip()}\nSentence 2: {pair['s2'].strip()}",
                instructions=STS_INSTRUCTIONS,
                levels=STS_LEVELS,
                label=level,
                audit={"mean_score": pair["score"], **pair["extra"]},
                entry=entry,
            )
        )
    return rows, skipped


# --- A7s: AfriSenti Swahili ----------------------------------------------------


def sentiment_rows(
    items: Sequence[tuple[str, str, str]],
    *,
    family: str,
    source: str,
    language: str,
    entry: Mapping[str, Any],
    skipped: collections.Counter,
) -> list[dict[str, Any]]:
    """(upstream id, text, label) -> A7s Score rows; conflicts and repeats dropped."""
    labels_by_text: dict[str, set[str]] = collections.defaultdict(set)
    for _, text, label in items:
        labels_by_text[normalize(text)].add(label)
    rows, seen = [], set()
    for upstream_id, text, label in items:
        key = normalize(text)
        if len(labels_by_text[key]) > 1:
            skipped["conflicting_labels"] += 1
            continue
        if key in seen:
            skipped["duplicate_text"] += 1
            continue
        seen.add(key)
        rows.append(
            score_row(
                sub_arm="A7s",
                family=family,
                source=source,
                upstream_id=upstream_id,
                group=f"{family}:{_sha(key)[:24]}",
                language=language,
                state=f"Message: {text}",
                instructions=SENTIMENT_INSTRUCTIONS,
                levels=SENTIMENT_LEVELS,
                label=SENTIMENT_LABELS[label],
                audit={"upstream_label": label},
                entry=entry,
            )
        )
    return rows


def afrisenti_rows(
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], collections.Counter]:
    reader = csv.DictReader(
        io.StringIO(read_verified(entry).decode("utf-8")), delimiter="\t"
    )
    skipped: collections.Counter[str] = collections.Counter()
    items = []
    for item in reader:
        text = (item.get("tweet") or "").strip()
        label = (item.get("label") or "").strip().lower()
        if not text or label not in SENTIMENT_LABELS:
            skipped["empty_or_unknown_label"] += 1
            continue
        items.append((item["ID"], text, label))
    rows = sentiment_rows(
        items,
        family="afrisenti_sw",
        source="a7:afrisenti_sw_train",
        language="sw",
        entry=entry,
        skipped=skipped,
    )
    return rows, skipped


SENTIMIX_MEMBER = "Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"


def sentimix_rows(
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], collections.Counter]:
    """Amendment 2: SentiMix Hinglish training tweets (CoNLL blocks) for A7s."""
    with zipfile.ZipFile(io.BytesIO(read_verified(entry))) as archive:
        text = archive.read(SENTIMIX_MEMBER).decode("utf-8")
    skipped: collections.Counter[str] = collections.Counter()
    items: list[tuple[str, str, str]] = []
    header: list[str] | None = None
    tokens: list[str] = []

    def close() -> None:
        if header is None:
            return
        label = header[2].strip().lower() if len(header) > 2 else ""
        if label not in SENTIMENT_LABELS or not tokens:
            skipped["empty_or_unknown_label"] += 1
        else:
            items.append((header[1], " ".join(tokens), label))

    for line in text.splitlines():
        if line.startswith("meta\t"):
            close()
            header, tokens = line.split("\t"), []
        elif line.strip() and header is not None:
            token = line.split("\t")[0]
            if token:
                tokens.append(token)
    close()
    rows = sentiment_rows(
        items,
        family="sentimix_hinglish",
        source="a7:sentimix_hinglish_train",
        language="hi-en",
        entry=entry,
        skipped=skipped,
    )
    return rows, skipped


# --- A7x: MASSIVE intents (ablation-only) --------------------------------------


def massive_items(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    with tarfile.open(fileobj=io.BytesIO(read_verified(entry)), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            name = member.name.rsplit("/", 1)[-1]
            if (
                not name.endswith(".jsonl")
                or name[: -len(".jsonl")] not in MASSIVE_LOCALES
            ):
                continue
            for line in archive.extractfile(member).read().decode("utf-8").splitlines():
                item = json.loads(line)
                if item["partition"] == "train":
                    items.append(item)
    return items


def massive_rows(
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], collections.Counter]:
    items = massive_items(entry)
    skipped: collections.Counter[str] = collections.Counter()
    intents_by_scenario: dict[str, set[str]] = collections.defaultdict(set)
    for item in items:
        intents_by_scenario[item["scenario"]].add(item["intent"])
    cells: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for item in items:
        cells[(item["locale"], item["scenario"])][item["intent"]].append(item)
    kept = []
    for key, by_intent in sorted(cells.items()):
        if len(by_intent) < 2:
            skipped["single_intent_scenario"] += sum(map(len, by_intent.values()))
            continue
        cap = int(BALANCE_RATIO * min(map(len, by_intent.values())))
        for intent, group in sorted(by_intent.items()):
            ordered = sorted(
                group,
                key=lambda item: _sha(f"massive_intent:{item['locale']}:{item['id']}"),
            )
            kept.extend(ordered[:cap])
            skipped["intent_balance"] += max(0, len(group) - cap)
    per_locale: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in kept:
        per_locale[item["locale"]].append(item)
    rows = []
    for locale, group in sorted(per_locale.items()):
        ordered = sorted(
            group, key=lambda item: _sha(f"massive-utterance:{item['id']}")
        )
        skipped["locale_cap"] += max(0, len(ordered) - MASSIVE_PER_LOCALE)
        for item in ordered[:MASSIVE_PER_LOCALE]:
            upstream_id = f"{item['locale']}:{item['id']}"
            same = sorted(intents_by_scenario[item["scenario"]] - {item["intent"]})
            distractors = sorted(
                same,
                key=lambda intent: _sha(f"massive-distractor:{upstream_id}:{intent}"),
            )[:3]
            position = int(_sha(f"massive-position:{upstream_id}"), 16) % (
                len(distractors) + 1
            )
            names = distractors[:position] + [item["intent"]] + distractors[position:]
            rows.append(
                _row(
                    sub_arm="A7x",
                    family="massive_intent",
                    source="a7:massive_1.1_train",
                    upstream_id=upstream_id,
                    group=f"massive:{item['id']}",
                    language=language_code(item["locale"]),
                    state=f"Request: {item['utt'].strip()}",
                    instructions=MASSIVE_INSTRUCTIONS,
                    options=[
                        {
                            "key": f"intent_{index}",
                            "description": name.replace("_", " "),
                        }
                        for index, name in enumerate(names)
                    ],
                    label=position,
                    task_type="choice",
                    audit={
                        "scenario": item["scenario"],
                        "intent": item["intent"],
                        "locale": item["locale"],
                    },
                    entry=entry,
                )
            )
    return rows, skipped


# --- balance, dedupe, isolation, partition ---------------------------------------


def balance_levels(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per family, cap every Score level at BALANCE_RATIO x the rarest level."""
    by_family: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    kept, dropped = [], {}
    for row in rows:
        if row["task_type"] == "score":
            by_family[row["family"]][row["label"]].append(row)
        else:
            kept.append(row)
    for family, levels in sorted(by_family.items()):
        size = len(next(iter(levels.values()))[0]["options"])
        counts = [len(levels.get(level, [])) for level in range(size)]
        cap = int(BALANCE_RATIO * min(counts))
        for level in range(size):
            ordered = sorted(
                levels.get(level, []), key=lambda row: _sha(f"{family}:{row['id']}")
            )
            kept.extend(ordered[:cap])
            if len(ordered) > cap:
                dropped[f"{family}|{level}"] = len(ordered) - cap
    return kept, dropped


def build(
    spec: Mapping[str, Any], excluded_ids: Mapping[str, set[str]]
) -> dict[str, Any]:
    if spec.get("version") != VERSION:
        raise ValueError(f"spec version must be {VERSION}")
    sources = {entry["name"]: entry for entry in spec["sources"]}
    skipped: dict[str, dict[str, int]] = {}
    candidates: list[dict[str, Any]] = []
    builders: list[
        tuple[str, Callable[[], tuple[list[dict[str, Any]], collections.Counter]]]
    ] = [
        ("A7q", lambda: oasst_rows(sources["oasst1"])),
        ("A7k", lambda: sts_rows([sources["klue_sts"], sources["jsts"]], excluded_ids)),
        ("A7s", lambda: afrisenti_rows(sources["afrisenti_sw"])),
        ("A7s-sentimix", lambda: sentimix_rows(sources["sentimix"])),
        ("A7x", lambda: massive_rows(sources["massive"])),
    ]
    for name, make in builders:
        rows, counts = make()
        skipped[name] = dict(sorted(counts.items()))
        candidates.extend(rows)
    balanced, balance_dropped = balance_levels(candidates)

    labels: dict[str, set[str]] = collections.defaultdict(set)
    for row in balanced:
        labels[row["input_sha256"]].add(canonical(row["options"][row["label"]]))
    kept: dict[str, dict[str, Any]] = {}
    duplicates = conflicts = 0
    for row in sorted(balanced, key=lambda row: row["id"]):
        digest_ = row["input_sha256"]
        if len(labels[digest_]) > 1:
            conflicts += 1
        elif digest_ in kept:
            duplicates += 1
        else:
            kept[digest_] = row

    holdout_groups, holdout_inputs, holdout_states = _external_sets(
        spec.get("isolation_holdout", [])
    )
    train_groups, train_inputs, train_states = _external_sets(
        spec.get("isolation_train", [])
    )
    isolation_dropped: collections.Counter[str] = collections.Counter()
    for digest_ in list(kept):
        row = kept[digest_]
        if digest_ in holdout_inputs or state_key(row["state"]) in holdout_states:
            isolation_dropped[row["family"]] += 1
            del kept[digest_]

    components = Components()
    members: dict[str, list[str]] = collections.defaultdict(list)
    for row in kept.values():
        members[f"input:{row['input_sha256']}"].append(row["group_id"])
        key = state_key(row["state"])
        if key is not None:
            members[f"state:{key}"].append(row["group_id"])
    components.link(members)
    roots: dict[str, set[str]] = collections.defaultdict(set)
    touched = set()
    for row in kept.values():
        root = components.find(row["group_id"])
        roots[root].add(row["group_id"])
        if (
            row["input_sha256"] in train_inputs
            or state_key(row["state"]) in train_states
        ):
            touched.add(root)
    parts: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"train": [], "aho": []} for name in ENC_SUB_ARMS
    }
    for row in kept.values():
        root = components.find(row["group_id"])
        sub_arm = row["audit_metadata"]["a7"]["sub_arm"]
        if root not in touched and is_aho(min(roots[root])):
            held = dict(row, split="select", evaluation_role="select")
            validate_row(held, "select")
            parts[sub_arm]["aho"].append(held)
        else:
            parts[sub_arm]["train"].append(row)
    manifest = {
        "schema": "decision2.v2.a7.build-enc.v1",
        "version": VERSION,
        "sources": [
            {
                key: entry[key]
                for key in ("name", "path", "sha256", "licence")
                if key in entry
            }
            for entry in spec["sources"]
        ],
        "isolation_inputs": {
            kind: [
                {key: e[key] for key in ("name", "sha256") if key in e}
                for e in spec.get(kind, [])
            ]
            for kind in ("isolation_train", "isolation_holdout")
        },
        "excluded_data_track_ids": {
            family: len(ids) for family, ids in sorted(excluded_ids.items())
        },
        "skipped": skipped,
        "balance_dropped": balance_dropped,
        "duplicates_dropped": duplicates,
        "label_conflicts_dropped": conflicts,
        "isolation_dropped": dict(sorted(isolation_dropped.items())),
        "sub_arms": {
            name: {
                part: {
                    "rows": len(rows),
                    "by_family_label": keyed(
                        collections.Counter(
                            (row["family"], row["label"]) for row in rows
                        )
                    ),
                    "by_language": dict(
                        sorted(
                            collections.Counter(row["language"] for row in rows).items()
                        )
                    ),
                }
                for part, rows in by_part.items()
            }
            for name, by_part in parts.items()
        },
    }
    return {"parts": parts, "manifest": manifest}


def data_track_ids(paths: Iterable[Path]) -> dict[str, set[str]]:
    """Upstream ids of KLUE-STS / JSTS rows in research & data files."""
    ids: dict[str, set[str]] = {"klue_sts": set(), "jglue_jsts": set()}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                local = str(row.get("audit_metadata", {}).get("source_local_id", ""))
                if row.get("source", "").startswith("klue_sts") and local:
                    ids["klue_sts"].add(local)
                    if local.isdigit():
                        ids["klue_sts"].add(f"klue-sts-v1_train_{int(local):05d}")
                elif row.get("source", "").startswith("jglue_jsts") and local:
                    ids["jglue_jsts"].add(local)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-track-file", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args(argv)
    if (args.out_dir / "prelim").exists():
        parser.error(f"refusing to reuse {args.out_dir / 'prelim'}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    excluded = data_track_ids(args.data_track_file)
    result = build(spec, excluded)
    (args.out_dir / "prelim").mkdir(parents=True, mode=0o700)
    (args.out_dir / "views").mkdir(mode=0o700, exist_ok=True)
    outputs = {}
    for name, by_part in result["parts"].items():
        for part, rows in by_part.items():
            if rows:
                data = canonical_jsonl(rows)
                path = args.out_dir / "prelim" / f"{name}.{part}.jsonl"
                _write_new(path, data)
                outputs[f"prelim/{path.name}"] = hashlib.sha256(data).hexdigest()
    manifest = dict(
        result["manifest"],
        build_commit=args.commit,
        data_track_files=[
            {"name": path.name, "sha256": sha256_file(path)}
            for path in args.data_track_file
        ],
        outputs=outputs,
    )
    _write_new(args.out_dir / "build-manifest.json", _json_bytes(manifest))
    print(
        json.dumps(
            {
                name: {part: len(rows) for part, rows in by_part.items()}
                for name, by_part in result["parts"].items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
