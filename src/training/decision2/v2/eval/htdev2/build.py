"""HT-DEV v2 builder: a held-out parallel form of the formal CSS15 tasks.

Preregistration `v2/eval/records/htdev2-prereg-2026-09-30.md` (amendment 1).

    python3 -m v2.eval.htdev2.build pool --sources <dir> --salt-root <LLMs_for_CSS> \
        --replication-root <decision-models-css> --output <work dir>
    python3 -m v2.eval.htdev2.build freeze --work <work dir> --hits <hits.jsonl> ... --output <panel dir>

`pool` rebuilds each task's SALT pool (the rows the formal sample was drawn from) with the
SALT loader's formatting, checks that the formal CSS15 items map onto it, removes every
formal and pilot item and every group touched by a formal item, and writes the candidates
for the overlap scans. `freeze` drops scan-flagged candidates, draws the class-balanced
sample and renders it with the authors' typed-choice mapping. Only counts and hashes are
printed; texts stay in the mode-600 work and panel files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from transfer.build import (
    criteria_for,
    json_text,
    normalized_context_sha256,
    sha_bytes,
    sha_value,
    source_maps,
)
from v2.eval import panels
from v2.eval.same_panel import sha_file, utc_now, write_json

PANEL = "ht-dev2"
PANEL_VERSION = "ht-dev2/1"
SEED_PREFIX = "ht-dev2/1"
TASKS = (
    "emotion",
    "ibc",
    "media_ideology",
    "raop",
    "talklife",
    "wiki_politeness",
    "tempowic",
    "flute",
    "mrf",
    "conv_go_awry",
    "persuasion",
    "reddit_humor",
    "wiki_corpus",
)
EXCLUDED = {
    "indian_english_dialect": "the formal sample is the whole annotated set",
    "tropes": "114 singleton labels incl. composites cannot be reproduced from held-out characters",
}
TEMPLATE_KEY = {
    "wiki_corpus": "power",
    "wiki_politeness": "politeness",
    "conv_go_awry": "toxicity",
    "reddit_humor": "humor",
    "flute": "flute-classification",
    "mrf": "mrf-classification",
}
TASK_CAP = 150
GROUP_CAP = 2
FLOOR = 60
PRESCAN_FACTOR = 40
MAP_MIN = 0.95
LABEL_MIN = 0.95
EMOTION_LETTERS = {4: "A", 3: "B", 1: "C", 0: "D", 2: "E", 5: "F"}
csv.field_size_limit(1 << 30)


@dataclass
class Row:
    key: str
    context: str
    prompt: str
    label: str
    groups: list[str] = field(default_factory=list)


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def salt_maps(salt_root: Path) -> dict[str, Any]:
    module = ast.parse((salt_root / "mappings.py").read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


HUMOR_PREFIX = " \n\nConstraint: "


def template(maps: dict[str, Any], task: str) -> str:
    text = maps["prompts_templates"][TEMPLATE_KEY.get(task, task)]
    if task == "reddit_humor":
        # the formal humor prompt predates this template's leading line and final newline
        if not (text.startswith(HUMOR_PREFIX) and text.endswith("\n")):
            raise ValueError("unexpected SALT humor template")
        text = text[len(HUMOR_PREFIX) : -1]
    return text


def salt_label(value: Any) -> str:
    """The label string the CSS builder sees: SALT's JSON value through str()."""
    return str(value)


def read_csv_rows(path: Path, header: bool = True) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[1:] if header else rows


def processed_pool(path: Path) -> Iterator[tuple[str, str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, context in data["context"].items():
        yield key, context, data["labels"][key]


# ---------------------------------------------------------------- per-task pools


def pool_emotion(src: Path, salt: Path, maps) -> Iterator[Row]:
    import pyarrow.parquet as pq

    table = pq.read_table(src / "emotion/unsplit/train-00000-of-00001.parquet")
    prompt = template(maps, "emotion")
    for index, row in enumerate(table.to_pylist()):
        text = row["text"]
        yield Row(
            f"row{index}", text, prompt, EMOTION_LETTERS[row["label"]], [norm(text)]
        )


def pool_csv_column(
    path: Path, text_col: str, label_col: str, prompt: str, group_cols=()
) -> Iterator[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            text = row.get(text_col) or ""
            if not text.strip():
                continue
            groups = [norm(text)] + [
                f"{col}:{norm(row[col])}" for col in group_cols if row.get(col)
            ]
            yield Row(f"row{index}", text, prompt, row[label_col], groups)


def pool_ibc(src, salt, maps):
    return pool_csv_column(
        salt / "css_data/ibc/ibc.csv", "sentence", "leaning", template(maps, "ibc")
    )


def pool_media(src, salt, maps):
    return pool_csv_column(
        salt / "css_data/media_ideology/media_ideology.csv",
        "content",
        "bias_text",
        template(maps, "media_ideology"),
        ("url", "title"),
    )


def pool_mrf(src, salt, maps):
    return pool_csv_column(
        salt / "css_data/mrf/mrf-classification.csv",
        "headline",
        "gold_label",
        template(maps, "mrf"),
    )


def pool_raop(src, salt, maps):
    prompt = template(maps, "raop")
    for key, context, label in processed_pool(salt / "css_data/raop/raop.json"):
        yield Row(f"row{key}", context, prompt, salt_label(label), [norm(context)])


def pool_talklife(src, salt, maps):
    prompt = template(maps, "talklife")
    with (salt / "css_data/talklife/talklife.csv").open(
        newline="", encoding="utf-8"
    ) as h:
        meta = list(csv.DictReader(h))
    for key, context, label in processed_pool(salt / "css_data/talklife/talklife.json"):
        row = meta[int(key)]
        expected = f"Seeker: {row['Seeker']}\n\nResponse: {row['Response']}"
        if expected == context:
            groups = [f"sp:{row['sp_id']}", f"rp:{row['rp_id']}"]
        else:
            groups = [f"ctx:{norm(context)}"]
        yield Row(f"row{key}", context, prompt, salt_label(label), groups)


def pool_flute(src, salt, maps):
    prompt = template(maps, "flute")
    path = salt / "css_data/flute/flute-classification.json"
    for key, context, label in processed_pool(path):
        premise, _, hypothesis = context.partition("\n\nhypothesis: ")
        groups = [f"p:{norm(premise)}", f"h:{norm(hypothesis)}"]
        yield Row(f"row{key}", context, prompt, salt_label(label), groups)


def pool_tempowic(src, salt, maps):
    prompt = template(maps, "tempowic")
    data = src / "tempowic/TempoWiC/data"
    splits = (
        ("validation", "validation.data.jl", "validation.labels.tsv"),
        ("train", "train.data.jl", "train.labels.tsv"),
        ("test", "test-codalab-10k.data.jl", "test.gold.tsv"),
    )
    for split, data_file, label_file in splits:
        labels = dict(
            line.split("\t")[:2]
            for line in (data / label_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        with (data / data_file).open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["id"] not in labels:
                    continue
                t1, t2, word = (
                    item["tweet1"]["text"],
                    item["tweet2"]["text"],
                    item["word"],
                )
                context = f"text1: {t1}\n\ntext2: {t2}\n\nword: {word}"
                label = "Same" if labels[item["id"]].strip() == "1" else "Different"
                groups = [f"pair:{item['id']}", f"t:{norm(t1)}", f"t:{norm(t2)}"]
                yield Row(f"{split}:{item['id']}", context, prompt, label, groups)


def pool_reddit_humor(src, salt, maps):
    prompt = template(maps, "reddit_humor")
    for name in ("test", "train"):
        for index, row in enumerate(
            read_csv_rows(src / f"reddit_humor/{name}.tsv", header=False)
        ):
            if len(row) != 4 or row[1] not in ("0", "1") or not row[3].strip():
                continue
            label = "True" if row[1] == "1" else "False"
            yield Row(f"{name}{index}", row[3], prompt, label, [norm(row[3])])


def convokit(src: Path, name: str):
    """Utterances (ConvoKit field names) and conversation metadata from a corpus ZIP."""
    with zipfile.ZipFile(src / f"convokit/{name}.zip") as archive:
        with archive.open(f"{name}/utterances.jsonl") as handle:
            utterances = []
            for line in handle:
                raw = json.loads(line)
                utterances.append(
                    {
                        "id": raw["id"],
                        "speaker": raw.get("speaker", raw.get("user")),
                        "conversation_id": raw.get("conversation_id", raw.get("root")),
                        "reply_to": raw.get("reply_to", raw.get("reply-to")),
                        "timestamp": raw.get("timestamp"),
                        "text": raw.get("text") or "",
                        "meta": raw.get("meta") or {},
                    }
                )
        members = set(archive.namelist())
        conv_path = f"{name}/conversations.json"
        conversations = (
            json.loads(archive.read(conv_path)) if conv_path in members else {}
        )
    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for utterance in utterances:
        by_conv[str(utterance["conversation_id"])].append(utterance)
    for conv in by_conv.values():
        conv.sort(key=lambda u: (u["timestamp"] is None, u["timestamp"] or 0))
    return dict(sorted(by_conv.items())), conversations


def line(utterance: dict[str, Any]) -> str:
    return f"{utterance['speaker']}: {utterance['text']}"


def pool_politeness(src, salt, maps):
    prompt = template(maps, "wiki_politeness")
    by_conv, _ = convokit(src, "wikipedia-politeness-corpus")
    for conv_id, conv in by_conv.items():
        context = "\n".join(line(u) for u in conv)
        label = conv[-1]["meta"].get("Binary")
        if label is None:
            continue
        yield Row(
            f"conv:{conv_id}", context, prompt, salt_label(label), [f"c:{conv_id}"]
        )


def pool_power(src, salt, maps):
    base = template(maps, "wiki_corpus")
    by_conv, _ = convokit(src, "wiki-corpus")
    for conv_id, conv in by_conv.items():
        context = "\n".join(line(u) for u in conv)
        first: dict[str, dict[str, Any]] = {}
        for utterance in conv:
            first.setdefault(str(utterance["speaker"]), utterance)
        for speaker, utterance in sorted(first.items()):
            label = utterance["meta"].get("is-admin")
            if label is None:
                continue
            prompt = base.replace("{$speaker}", speaker)
            yield Row(
                f"conv:{conv_id}|spk:{speaker}",
                context,
                prompt,
                salt_label(bool(label)),
                [f"c:{conv_id}"],
            )


def first_two(conv: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], str]]:
    first = conv[0]
    initial = line(first)
    for reply in conv:
        if reply["reply_to"] == first["id"]:
            yield reply, "\n".join([initial, line(reply)])


def pool_cga(src, salt, maps):
    prompt = template(maps, "conv_go_awry")
    by_conv, conversations = convokit(src, "conversations-gone-awry-corpus")
    for conv_id, conv in by_conv.items():
        meta = conversations.get(conv_id, {})
        meta = meta.get("meta", meta)
        label = meta.get("conversation_has_personal_attack")
        if label is None:
            continue
        groups = [f"c:{conv_id}"] + (
            [f"pair:{meta['pair_id']}"] if meta.get("pair_id") else []
        )
        for reply, context in first_two(conv):
            yield Row(
                f"conv:{conv_id}|utt:{reply['id']}",
                context,
                prompt,
                salt_label(bool(label)),
                groups,
            )


def pool_persuasion(src, salt, maps):
    prompt = template(maps, "persuasion")
    by_conv, _ = convokit(src, "winning-args-corpus")
    for conv_id, conv in by_conv.items():
        for reply, context in first_two(conv):
            success = reply["meta"].get("success")
            if success is None:
                continue
            yield Row(
                f"conv:{conv_id}|utt:{reply['id']}",
                context,
                prompt,
                salt_label(float(success)),
                [f"c:{conv_id}"],
            )


POOLS = {
    "emotion": pool_emotion,
    "ibc": pool_ibc,
    "media_ideology": pool_media,
    "raop": pool_raop,
    "talklife": pool_talklife,
    "wiki_politeness": pool_politeness,
    "tempowic": pool_tempowic,
    "flute": pool_flute,
    "mrf": pool_mrf,
    "conv_go_awry": pool_cga,
    "persuasion": pool_persuasion,
    "reddit_humor": pool_reddit_humor,
    "wiki_corpus": pool_power,
}


# ---------------------------------------------------------------- stage: pool


def order_key(task: str, key: str) -> str:
    return hashlib.sha256(f"{SEED_PREFIX}:{task}:{key}".encode()).hexdigest()


def group_hash(task: str, group: str) -> str:
    return sha_bytes(f"{task}\x00{group}".encode("utf-8"))


def write_private_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_text(row) + "\n")
    return sha_file(path)


def formal_rows(panel_root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in ("css15", "css-pilot"):
        with panels.path(panel_root, name, "gold").open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def build_pool(args: argparse.Namespace) -> dict[str, Any]:
    panels.verify(args.panel_root, ["css15", "css-pilot"])
    maps = salt_maps(args.salt_root)
    option_to_gold, binary_criteria = source_maps(
        args.replication_root / "pilot_jev.py"
    )
    formal = formal_rows(args.panel_root)
    formal_contexts = {row["normalized_context_sha256"] for row in formal}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formal:
        by_task[row["task"]].append(row)
    candidates, manifest = [], {}
    for task in TASKS:
        labels = by_task[task][0]["labels"]
        rows = list(POOLS[task](args.sources, args.salt_root, maps))
        index: dict[tuple[str, str], list[Row]] = defaultdict(list)
        by_context: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            ctx = normalized_context_sha256(row.context)
            index[(ctx, sha_bytes(row.prompt.encode("utf-8")))].append(row)
            by_context[ctx].append(row)
        mapped = label_ok = 0
        blocked_groups: set[str] = set()
        for item in by_task[task]:
            ctx, prompt = (
                item["normalized_context_sha256"],
                item["source_prompt_sha256"],
            )
            for row in by_context.get(ctx, []):
                blocked_groups.update(row.groups)
            hits = index.get((ctx, prompt), [])
            if hits:
                mapped += 1
                label_ok += any(salt_label(h.label) == item["gold"] for h in hits)
        n_formal = len(by_task[task])
        map_rate = mapped / n_formal
        label_rate = label_ok / mapped if mapped else 0.0
        admitted = map_rate >= MAP_MIN and label_rate >= LABEL_MIN
        counts: Counter = Counter()
        kept: list[tuple[str, Row]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            ctx = normalized_context_sha256(row.context)
            if ctx in formal_contexts:
                counts["formal_context"] += 1
            elif blocked_groups.intersection(row.groups):
                counts["formal_group"] += 1
            elif row.label not in labels:
                counts["label_outside_task"] += 1
            elif (ctx, row.prompt) in seen:
                counts["duplicate"] += 1
            else:
                seen.add((ctx, row.prompt))
                kept.append((order_key(task, row.key), row))
        kept.sort(key=lambda pair: pair[0])
        quota = TASK_CAP // len(labels)
        per_class: Counter = Counter()
        scanned = 0
        for _, row in kept:
            if per_class[row.label] >= PRESCAN_FACTOR * quota:
                continue
            per_class[row.label] += 1
            scanned += 1
            candidates.append(
                {
                    "task": task,
                    "source_item_id": row.key,
                    "source": task,
                    "state": row.context,
                    "prompt": row.prompt,
                    "label": row.label,
                    "groups": [group_hash(task, g) for g in row.groups],
                    "order": order_key(task, row.key),
                }
            )
        manifest[task] = {
            "pool_rows": len(rows),
            "formal_items": n_formal,
            "formal_mapped": mapped,
            "map_rate": map_rate,
            "label_agreement": label_rate,
            "admitted": admitted,
            "blocked_groups": len(blocked_groups),
            "excluded": dict(counts),
            "eligible": len(kept),
            "eligible_by_class": dict(Counter(row.label for _, row in kept)),
            "quota_per_class": quota,
            "labels": labels,
            "scan_candidates": scanned,
        }
        print(
            json.dumps(
                {
                    "task": task,
                    **{
                        k: manifest[task][k]
                        for k in (
                            "pool_rows",
                            "map_rate",
                            "label_agreement",
                            "admitted",
                            "eligible",
                            "scan_candidates",
                        )
                    },
                }
            ),
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    scan_rows = (
        {k: c[k] for k in ("task", "source_item_id", "source", "state")}
        for c in candidates
    )
    result = {
        "schema": "dev2-htdev2-pool/1",
        "created_utc": utc_now(),
        "salt_mappings_sha256": sha_file(args.salt_root / "mappings.py"),
        "pilot_jev_sha256": sha_file(args.replication_root / "pilot_jev.py"),
        "excluded_tasks": EXCLUDED,
        "tasks": manifest,
        "files": {
            "candidates": write_private_jsonl(
                args.output / "candidates.jsonl", candidates
            ),
            "scan": write_private_jsonl(
                args.output / "scan-candidates.jsonl", scan_rows
            ),
        },
    }
    write_json(args.output / "POOL.json", result)
    return result


# ---------------------------------------------------------------- stage: freeze


def flagged_ids(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    flagged: set[str] = set()
    stats: Counter = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_text in handle:
                hit = json.loads(line_text)
                verdict = hit.get("verdict")
                stats[f"{path.name}:{verdict}"] += 1
                if verdict != "CLEAN":
                    flagged.add(hit["id"])
    return flagged, dict(stats)


def render(task: str, row: dict[str, Any], option_to_gold, binary_criteria):
    instructions, criteria = criteria_for(
        task, row["prompt"], option_to_gold, binary_criteria
    )
    item_id = f"{PANEL}/{task}/{sha_bytes(row['source_item_id'].encode())[:16]}"
    payload = {
        "state": row["state"],
        "questions": {
            "label": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }
    gold = {
        "id": item_id,
        "panel_version": PANEL_VERSION,
        "role": "evaluation",
        "task": task,
        "gold": row["label"],
        "labels": list(criteria),
        "source_key_sha256": sha_bytes(row["source_item_id"].encode()),
        "group_sha256": row["groups"],
        "source_context_sha256": sha_bytes(row["state"].encode("utf-8")),
        "normalized_context_sha256": normalized_context_sha256(row["state"]),
        "source_prompt_sha256": sha_bytes(row["prompt"].encode("utf-8")),
        "input_sha256": sha_value(payload),
    }
    return {"id": item_id, **payload}, gold


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    pool = json.loads((args.work / "POOL.json").read_text(encoding="utf-8"))
    option_to_gold, binary_criteria = source_maps(
        args.replication_root / "pilot_jev.py"
    )
    flagged, hit_stats = flagged_ids(args.hits)
    with (args.work / "candidates.jsonl").open(encoding="utf-8") as handle:
        candidates = [json.loads(line_text) for line_text in handle]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_task[row["task"]].append(row)
    prompts, golds, summary = [], [], {}
    for task in TASKS:
        info = pool["tasks"][task]
        if not info["admitted"]:
            summary[task] = {"admitted": False, "reason": "reconstruction check"}
            continue
        quota = info["quota_per_class"]
        rows = sorted(by_task[task], key=lambda r: r["order"])
        chosen, per_class, per_group, dropped = [], Counter(), Counter(), 0
        for row in rows:
            if f"{task}|{row['source_item_id']}" in flagged:
                dropped += 1
                continue
            primary = row["groups"][0]
            if per_class[row["label"]] >= quota or per_group[primary] >= GROUP_CAP:
                continue
            per_class[row["label"]] += 1
            per_group[primary] += 1
            chosen.append(row)
        admitted = len(chosen) >= FLOOR and set(per_class) == set(info["labels"])
        summary[task] = {
            "admitted": admitted,
            "items": len(chosen),
            "by_class": dict(sorted(per_class.items())),
            "flagged_in_scan": dropped,
        }
        if not admitted:
            continue
        for row in chosen:
            prompt, gold = render(task, row, option_to_gold, binary_criteria)
            prompts.append(prompt)
            golds.append(gold)
    order = sorted(
        range(len(prompts)), key=lambda i: sha_bytes(prompts[i]["id"].encode())
    )
    prompts = [prompts[i] for i in order]
    golds = [golds[i] for i in order]
    out = args.output
    result = {
        "schema": "dev2-htdev2-build/1",
        "created_utc": utc_now(),
        "panel": PANEL,
        "panel_version": PANEL_VERSION,
        "pool_sha256": sha_file(args.work / "POOL.json"),
        "hits": {str(p): sha_file(p) for p in args.hits},
        "hit_stats": hit_stats,
        "tasks": summary,
        "items": len(prompts),
        "admitted_tasks": sorted(t for t, s in summary.items() if s.get("admitted")),
        "files": {
            "prompts": write_private_jsonl(
                out / "goldfree" / f"{PANEL}.prompts.jsonl", prompts
            ),
            "gold": write_private_jsonl(out / "gold" / f"{PANEL}.gold.jsonl", golds),
        },
    }
    os.chmod(out / "goldfree" / f"{PANEL}.prompts.jsonl", 0o644)
    write_json(out / "BUILD.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("pool")
    one.add_argument("--sources", type=Path, required=True)
    one.add_argument("--salt-root", type=Path, required=True)
    one.add_argument("--replication-root", type=Path, required=True)
    one.add_argument("--panel-root", type=Path, default=panels.DEFAULT_ROOT)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("freeze")
    two.add_argument("--work", type=Path, required=True)
    two.add_argument("--replication-root", type=Path, required=True)
    two.add_argument("--hits", type=Path, action="append", default=[])
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_pool(args) if args.command == "pool" else freeze(args)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("files", "items", "admitted_tasks")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
