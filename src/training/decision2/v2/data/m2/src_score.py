"""H6 human ordinal Score families: MLQE-PE, OneStopEnglish and A6h2 (C10).

Rules: data-arms-v2-prereg-2026-09-28.md section 3 (C10). A binned row draws
its level count L from 2..10 by ``sha256(f"{family}:{local_id}")``; cut points
are linear-interpolated quantiles of the value over every parsed TRAIN item of
the source (of the topic for ArgQ), one set per L; rows within the guard band
of a cut are dropped; each (family, L) cell keeps at most 6/5 of its rarest
level. The whole-group cap is applied here and followed by the same balance,
so the balance holds on the capped rows and the orchestrator's cap is a no-op.

OneStopEnglish is a fixed three-level Score with one group per article and one
row per reading level. Intermediate files of the pinned release lack every
non-ASCII character (typographic quotes, dashes and pound signs are deleted)
while the other levels keep them, so all versions are folded to ASCII the same
way and the character set carries no level information.

A6h2 reuses the v1 KLUE-STS, JSTS and ArgQ record parsing, so local ids and
group keys equal v1's. Items used by v1, items sharing a v1 item's group key
and items with a v1 item's normalized state are excluded.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import json
import math
import statistics
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import file_sha256
from v2.data.m2.build import v1_local_ids
from v2.data.m2.common import cap_groups, make_row, score_options
from v2.data.m2.spec import FamilySpec
from v2.data.sources import argq, jglue, klue, ordinal
from v2.data.textnorm import normalize

Rows = list[dict[str, Any]]
Report = dict[str, Any]

ARM = "h6"
LEVELS = tuple(range(2, 11))
C10_DROPS = ("guard_band", "level_balance", "cap", "post_cap_balance")

MLQEPE_SOURCE = "mlqepe_train"
MLQEPE_DIR = Path("data/direct-assessments/train")
MLQEPE_PAIRS = ("en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en")
MLQEPE_COLUMNS = ("index", "original", "translation", "scores", "mean")
MLQEPE_CAP = 3000
MLQEPE_DROPS = (
    "malformed_line",
    "duplicate_index",
    "empty_text",
    "invalid_mean",
    "duplicate_state",
)

ONESTOP_SOURCE = "onestop_english"
ONESTOP_FAMILY = "onestop_level"
ONESTOP_DIR = Path("Texts-SeparatedByReadingLevel")
ONESTOP_LEVELS = (("ele", "Ele-Txt"), ("int", "Int-Txt"), ("adv", "Adv-Txt"))
ONESTOP_HEADERS = {"elementary": "ele", "intermediate": "int", "advanced": "adv"}
ONESTOP_OPTIONS = (
    "Elementary: written for beginning readers of English.",
    "Intermediate: written for readers with some command of English.",
    "Advanced: written for fluent readers of English.",
)
ONESTOP_INSTRUCTIONS = "At which reading level is this text written?"
ONESTOP_CAP = 1000
ONESTOP_SEED = "h6-onestop-v1"
ONESTOP_DROPS = (
    "unexpected_name",
    "empty_text",
    "duplicate_article",
    "missing_level",
    "header_mismatch",
    "identical_versions",
)

A6H2_CAP = 1800
A6H2_DROPS = ("v1_item", "v1_group", "v1_state", "duplicate_state")


@dataclasses.dataclass(frozen=True)
class Scale:
    low: float
    high: float
    guard: float
    digits: int
    measure: str
    population: str
    instructions: str


MLQEPE_SCALE = Scale(
    low=0.0,
    high=100.0,
    guard=2.0,
    digits=1,
    measure="mean human quality rating",
    population="translations in this language pair",
    instructions="How good is the translation of the source sentence? "
    "Choose the quality level.",
)
STS_SCALE = Scale(
    low=0.0,
    high=5.0,
    guard=0.2,
    digits=1,
    measure="mean human similarity rating",
    population="sentence pairs in this dataset",
    instructions="How similar in meaning are the two sentences? "
    "Choose the similarity level.",
)
ARGQ_SCALE = Scale(
    low=0.0,
    high=1.0,
    guard=0.02,
    digits=2,
    measure="weighted human quality score",
    population="arguments on this topic",
    instructions="How strong is this argument for or against the topic? "
    "Choose the quality level.",
)
DESCRIPTION_TEMPLATE = (
    "Level {k+1} of {L} (ranked {p_k}–{p_k+1}% from the bottom among "
    "{population}): {measure} {band low}–{band high} on a {low}–{high} scale; "
    "p_k = 100k/L rounded half up, bands = [scale low, cuts..., scale high]"
)


@dataclasses.dataclass(frozen=True)
class Item:
    local_id: str
    group_key: str
    key: tuple[str, ...]
    state: dict[str, str]
    value: float
    audit: dict[str, Any]
    topic: str = ""
    check: float | None = None


def clean(text: str) -> str:
    return " ".join(text.split())


def percent(k: int, levels: int) -> int:
    """``100 * k / levels`` rounded half up."""
    return (200 * k + levels) // (2 * levels)


def describe(cuts: Sequence[float], scale: Scale) -> list[str]:
    levels = len(cuts) + 1
    span = f"{scale.low:g}–{scale.high:g}"
    return [
        f"Level {k + 1} of {levels} (ranked {percent(k, levels)}–"
        f"{percent(k + 1, levels)}% from the bottom among {scale.population}): "
        f"{scale.measure} {low + 0.0:.{scale.digits}f}–"
        f"{high + 0.0:.{scale.digits}f} on a {span} scale"
        for k, (low, high) in enumerate(ordinal.bands(cuts, scale.low, scale.high))
    ]


def fit(items: Sequence[Item], field: str) -> dict[str, dict[int, list[float]]]:
    values: dict[str, list[float]] = collections.defaultdict(list)
    for item in items:
        values[item.topic].append(getattr(item, field))
    return {
        topic: {size: ordinal.quantile_cuts(values[topic], size) for size in LEVELS}
        for topic in sorted(values)
    }


def eligible(
    items: Sequence[Item], used: set[str], drops: collections.Counter[str]
) -> list[Item]:
    """Items not used by v1 (by local id, group key or normalized state); the
    first item of every normalized state."""
    taken = [item for item in items if item.local_id in used]
    groups = {item.group_key for item in taken}
    states = {item.key for item in taken}
    seen: set[tuple[str, ...]] = set()
    kept = []
    for item in items:
        if item.local_id in used:
            reason = "v1_item"
        elif item.group_key in groups:
            reason = "v1_group"
        elif item.key in states:
            reason = "v1_state"
        elif item.key in seen:
            reason = "duplicate_state"
        else:
            seen.add(item.key)
            kept.append(item)
            continue
        drops[reason] += 1
    return kept


def by_size(counts: Mapping[int, int]) -> dict[str, int]:
    return {f"L{size}": counts.get(size, 0) for size in LEVELS}


def level_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    table = collections.Counter((len(row["options"]), row["label"]) for row in rows)
    return {
        f"L{size}": {str(level): table[size, level] for level in range(size)}
        for size in LEVELS
    }


def balance(rows: Rows, seed: str) -> tuple[Rows, dict[str, Any]]:
    kept, cells = ordinal.balance(
        rows,
        cell=lambda row: f"L{len(row['options'])}",
        level=lambda row: row["label"],
        levels=lambda row: len(row["options"]),
        ident=lambda row: row["id"],
        seed=f"{seed}:balance",
    )
    return kept, {
        f"L{size}": cells[f"L{size}"] for size in LEVELS if f"L{size}" in cells
    }


def binned(
    fitted: Sequence[Item],
    candidates: Sequence[Item],
    *,
    family: str,
    source: str,
    language: str,
    namespace: str,
    scale: Scale,
    seed: str,
    cap: int,
    drops: collections.Counter[str],
    agreement: bool = False,
) -> tuple[Rows, Report]:
    """C10 Score rows from ``candidates`` with cut points fitted on ``fitted``."""
    cuts = fit(fitted, "value")
    checks = fit(fitted, "check") if agreement else {}
    texts = {
        topic: {size: describe(points, scale) for size, points in sizes.items()}
        for topic, sizes in cuts.items()
    }
    assigned: collections.Counter[int] = collections.Counter()
    guarded: collections.Counter[int] = collections.Counter()
    disagreed: collections.Counter[int] = collections.Counter()
    rows: Rows = []
    for item in candidates:
        size = ordinal.level_count(family, item.local_id, LEVELS)
        assigned[size] += 1
        points = cuts[item.topic][size]
        if ordinal.near_cut(item.value, points, scale.guard):
            guarded[size] += 1
            continue
        label = ordinal.bin_index(item.value, points)
        if (
            agreement
            and ordinal.bin_index(item.check, checks[item.topic][size]) != label
        ):
            disagreed[size] += 1
            continue
        rows.append(
            make_row(
                source=source,
                family=family,
                task_type="score",
                language=language,
                namespace=namespace,
                group_key=item.group_key,
                local_id=item.local_id,
                state=item.state,
                instructions=scale.instructions,
                options=score_options(texts[item.topic][size]),
                label=label,
                template=f"m2/{family}/v1",
                audit={**item.audit, "levels": size},
            )
        )
    balanced, cells = balance(rows, seed)
    capped = cap_groups(balanced, cap, seed)
    final, post_cells = balance(capped, seed)
    kept = {row["id"] for row in final}
    final = [row for row in rows if row["id"] in kept]
    drops["guard_band"] += sum(guarded.values())
    if agreement:
        drops["mace_p_disagreement"] += sum(disagreed.values())
    drops["level_balance"] += len(rows) - len(balanced)
    drops["cap"] += len(balanced) - len(capped)
    drops["post_cap_balance"] += len(capped) - len(final)
    sizes_left = collections.Counter(len(row["options"]) for row in final)

    def table(sizes: Mapping[int, list[float]]) -> dict[str, list[float]]:
        return {f"L{size}": points for size, points in sizes.items()}

    topical = any(topic for topic in cuts)
    report: Report = {
        "level_counts": list(LEVELS),
        "level_count_rule": f"{LEVELS}[int(sha256(f'{family}:{{local_id}}'), 16)"
        f" % {len(LEVELS)}]",
        "cut_rule": "linear-interpolated quantiles k/L of the value over every "
        "parsed TRAIN item"
        + (" of the same topic" if topical else "")
        + "; level = number of cuts <= value",
        "cut_points": (
            {topic: table(sizes) for topic, sizes in cuts.items()}
            if topical
            else table(cuts.get("", {}))
        ),
        "guard_band": scale.guard,
        "guard_band_rule": f"drop if |value - cut| <= {scale.guard} "
        f"(+{ordinal.TOLERANCE}) for any cut",
        "levels_assigned": by_size(assigned),
        "guard_band_drops": by_size(guarded),
        "levels_before_balance": level_table(rows),
        "balance_rule": "per cell (family, L) keep at most floor(6/5 * rarest "
        "level) rows per level in sha256(seed:balance:id) order",
        "balance_seed": f"{seed}:balance",
        "balance_cells": cells,
        "empty_level_cells": [
            f"L{size}" for size in LEVELS if not cells.get(f"L{size}", {}).get("limit")
        ],
        "levels_after_balance": level_table(balanced),
        "cap_rule": "common.cap_groups(rows, cap, seed) on the balanced rows, then "
        "the same balance again (the orchestrator's cap is then a no-op)",
        "cap": cap,
        "cap_seed": seed,
        "rows_before_cap": len(balanced),
        "post_cap_balance_cells": post_cells,
        "levels_after_cap": level_table(final),
        "rows_by_level_count": by_size(sizes_left),
        "empty_level_cells_after_cap": [
            f"L{size}" for size in LEVELS if not sizes_left[size]
        ],
        "rows": len(final),
        "groups": len({row["group_id"] for row in final}),
        "description_template": DESCRIPTION_TEMPLATE,
    }
    if agreement:
        report["mace_p_disagreement_drops"] = by_size(disagreed)
        report["mace_p_cut_points"] = {
            topic: table(sizes) for topic, sizes in checks.items()
        }
        report["agreement_rule"] = (
            "kept only if the MACE-P level under the per-topic MACE-P cuts of "
            "the same L equals the value level"
        )
    if not topical and "" in texts:
        report["descriptions"] = {
            f"L{size}": texts[""][size] for size in LEVELS if size in texts[""]
        }
    return final, report


def read_member(path: Path, name: str) -> bytes:
    with tarfile.open(path, "r:gz") as archive:
        found = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.removeprefix("./") == name
        ]
        if len(found) != 1:
            raise ValueError(f"{path}: expected one member {name}, found {len(found)}")
        stream = archive.extractfile(found[0])
        if stream is None:
            raise ValueError(f"{path}: member {name} is not readable")
        return stream.read()


def number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def ratings(text: str) -> list[float]:
    try:
        values = json.loads(text)
    except ValueError:
        return []
    if not isinstance(values, list) or not all(
        type(value) in (int, float) for value in values
    ):
        return []
    return [float(value) for value in values]


def mlqepe_items(
    data: bytes, pair: str, where: str, drops: collections.Counter[str]
) -> tuple[list[Item], dict[str, int]]:
    """Rows of an unquoted tab-separated ``df.short.tsv`` member (fields are
    split on tabs only; quotes are literal text)."""
    lines = data.decode("utf-8-sig").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError(f"{where}: empty file")
    header = lines[0].removesuffix("\r").split("\t")
    missing = [name for name in MLQEPE_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}")
    column = {name: header.index(name) for name in MLQEPE_COLUMNS}
    stats = {"records": 0, "blank_lines": 0, "mean_differs_from_scores": 0}
    items: list[Item] = []
    indices: set[str] = set()
    for raw in lines[1:]:
        line = raw.removesuffix("\r")
        if not line.strip():
            stats["blank_lines"] += 1
            continue
        stats["records"] += 1
        fields = line.split("\t")
        index = fields[column["index"]].strip() if len(fields) == len(header) else ""
        if not index:
            drops["malformed_line"] += 1
            continue
        if index in indices:
            drops["duplicate_index"] += 1
            continue
        indices.add(index)
        original = clean(fields[column["original"]])
        translation = clean(fields[column["translation"]])
        if not original or not translation:
            drops["empty_text"] += 1
            continue
        mean = number(fields[column["mean"]])
        if mean is None or not 0 <= mean <= 100:
            drops["invalid_mean"] += 1
            continue
        scores = ratings(fields[column["scores"]])
        if scores and abs(sum(scores) / len(scores) - mean) > 1e-6:
            stats["mean_differs_from_scores"] += 1
        items.append(
            Item(
                local_id=f"{pair}:{index}",
                group_key=normalize(original),
                key=(normalize(original), normalize(translation)),
                state={"source": original, "translation": translation},
                value=mean,
                audit={"mean": mean, "raters": len(scores) if scores else None},
            )
        )
    return items, stats


def mlqepe(
    dirs: Mapping[str, Path],
    *,
    family: str,
    source: str,
    cap: int,
    seed: str,
    pair: str,
) -> tuple[Rows, Report]:
    root = dirs["mlqepe"]
    relative = MLQEPE_DIR / f"{pair}-train.tar.gz"
    member = f"{pair}-train/train.{pair.replace('-', '')}.df.short.tsv"
    data = read_member(root / relative, member)
    drops = collections.Counter(dict.fromkeys(MLQEPE_DROPS + C10_DROPS, 0))
    items, stats = mlqepe_items(data, pair, f"{relative}:{member}", drops)
    first, second = pair.split("-")
    language = first if second == "en" else second
    rows, binning = binned(
        items,
        eligible(items, set(), drops),
        family=family,
        source=source,
        language=language,
        namespace="mlqepe",
        scale=MLQEPE_SCALE,
        seed=seed,
        cap=cap,
        drops=drops,
    )
    return rows, {
        "inputs": {relative.as_posix(): file_sha256(root / relative)},
        "member": {"name": member, "sha256": hashlib.sha256(data).hexdigest()},
        "pair": pair,
        "language": language,
        "candidates": stats["records"],
        "items": stats["records"],
        "drops": dict(sorted(drops.items())),
        "blank_lines": stats["blank_lines"],
        "mean_differs_from_scores": stats["mean_differs_from_scores"],
        "value": "mean (0-100) of the raters' direct-assessment scores",
        "group_key": "normalized original sentence (namespace mlqepe)",
        **binning,
    }


@dataclasses.dataclass(frozen=True)
class Version:
    article: str
    text: str
    removed: int
    header: str | None


def reading_text(text: str) -> tuple[str, int, str | None]:
    """ASCII-folded, whitespace-normalized lines without blank lines or a
    leading level header; returns (text, removed characters, header level)."""
    lines, removed = [], 0
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        spaced = " ".join(raw.split())
        folded = "".join(char for char in spaced if char < "\x80")
        removed += len(spaced) - len(folded)
        line = " ".join(folded.split())
        if line:
            lines.append(line)
    header = None
    if lines and lines[0].casefold() in ONESTOP_HEADERS:
        header = ONESTOP_HEADERS[lines.pop(0).casefold()]
    return "\n".join(lines), removed, header


def onestop(
    dirs: Mapping[str, Path], *, family: str, source: str, cap: int, seed: str
) -> tuple[Rows, Report]:
    root = dirs["onestop"]
    drops = collections.Counter(dict.fromkeys(ONESTOP_DROPS, 0))
    inputs: dict[str, str] = {}
    ignored, replaced, replacement_characters = 0, [], 0
    headers: collections.Counter[str] = collections.Counter()
    articles: dict[str, dict[str, list[Version]]] = {}
    for code, folder in ONESTOP_LEVELS:
        directory = root / ONESTOP_DIR / folder
        for path in sorted(directory.iterdir(), key=lambda entry: entry.name):
            hidden = path.name.startswith(".")
            if hidden or not path.is_file() or path.suffix.lower() != ".txt":
                ignored += 1
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            inputs[relative] = hashlib.sha256(data).hexdigest()
            article, dash, suffix = path.stem.rpartition("-")
            article = clean(article)
            if not dash or not article or suffix.strip().casefold() != code:
                drops["unexpected_name"] += 1
                continue
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace").removeprefix("\ufeff")
                replaced.append(relative)
                replacement_characters += text.count("\ufffd")
            body, removed, header = reading_text(text)
            if header is not None:
                headers[code] += 1
            if not body:
                drops["empty_text"] += 1
                continue
            versions = articles.setdefault(normalize(article), {})
            versions.setdefault(code, []).append(
                Version(article, body, removed, header)
            )
    rows: Rows = []
    for key in sorted(articles):
        versions = articles[key]
        if any(len(found) > 1 for found in versions.values()):
            reason = "duplicate_article"
        elif len(versions) < len(ONESTOP_LEVELS):
            reason = "missing_level"
        elif any(
            found[0].header not in (None, code) for code, found in versions.items()
        ):
            reason = "header_mismatch"
        elif len({normalize(found[0].text) for found in versions.values()}) < len(
            ONESTOP_LEVELS
        ):
            reason = "identical_versions"
        else:
            reason = ""
        if reason:
            drops[reason] += sum(len(found) for found in versions.values())
            continue
        for label, (code, _) in enumerate(ONESTOP_LEVELS):
            version = versions[code][0]
            rows.append(
                make_row(
                    source=source,
                    family=family,
                    task_type="score",
                    language="en",
                    namespace="onestop",
                    group_key=key,
                    local_id=f"{version.article}:{code}",
                    state=version.text,
                    instructions=ONESTOP_INSTRUCTIONS,
                    options=score_options(ONESTOP_OPTIONS),
                    label=label,
                    template=f"m2/{family}/v1",
                    audit={
                        "article": version.article,
                        "characters": len(version.text),
                        "non_ascii_removed": version.removed,
                    },
                )
            )
    level_rows = {
        code: [row for row in rows if row["label"] == label]
        for label, (code, _) in enumerate(ONESTOP_LEVELS)
    }
    return rows, {
        "inputs": inputs,
        "candidates": len(inputs),
        "items": len(inputs),
        "drops": dict(sorted(drops.items())),
        "ignored_files": ignored,
        "articles": len(rows) // len(ONESTOP_LEVELS),
        "rows": len(rows),
        "level_histogram": {
            str(label): len(level_rows[code])
            for label, (code, _) in enumerate(ONESTOP_LEVELS)
        },
        "utf8_replacement": {
            "files": replaced,
            "characters": replacement_characters,
        },
        "header_lines_removed": {code: headers[code] for code, _ in ONESTOP_LEVELS},
        "non_ascii_removed": {
            code: {
                "rows_with_removals": sum(
                    1
                    for row in level_rows[code]
                    if row["audit_metadata"]["non_ascii_removed"]
                ),
                "characters": sum(
                    row["audit_metadata"]["non_ascii_removed"]
                    for row in level_rows[code]
                ),
            }
            for code, _ in ONESTOP_LEVELS
        },
        "mean_characters": {
            code: (
                round(
                    statistics.fmean(len(row["state"]) for row in level_rows[code]), 1
                )
                if level_rows[code]
                else 0.0
            )
            for code, _ in ONESTOP_LEVELS
        },
        "normalization": "UTF-8 (BOM stripped; errors='replace' only for invalid "
        "files); every non-ASCII character deleted in all levels; whitespace "
        "collapsed per line; blank lines removed; a leading Elementary / "
        "Intermediate / Advanced header line removed",
        "group_key": "normalized article name (namespace onestop)",
        "balance": "exact: one row per level for every article present at all levels",
    }


def sts_item(pair: klue.RatedPair) -> Item:
    first, second = clean(pair.first), clean(pair.second)
    audit: dict[str, Any] = {"mean_rating": pair.mean}
    if pair.stratum is not None:
        audit["stratum"] = pair.stratum
    return Item(
        local_id=pair.local_id,
        group_key=pair.group or pair.local_id,
        key=(normalize(first), normalize(second)),
        state={"sentence_1": first, "sentence_2": second},
        value=pair.mean,
        audit=audit,
    )


def klue_sts_items(root: Path) -> tuple[list[Item], Report]:
    """v1 ``klue.sts`` pairs: local id = guid, value = labels.real-label."""
    pairs = []
    for where, record in klue.read_jsonl(root / klue.STS_FILE):
        labels = record.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"{where}: labels must be an object")
        pairs.append(
            klue.RatedPair(
                local_id=klue.ident(record, "guid", where),
                first=klue.text(record, "sentence1", where),
                second=klue.text(record, "sentence2", where),
                mean=klue.rating(labels.get("real-label"), 0, klue.STS_TOP, where),
                stratum=klue.text(record, "source", where),
            )
        )
    return [sts_item(pair) for pair in pairs], {
        "inputs": {klue.STS_FILE.as_posix(): file_sha256(root / klue.STS_FILE)},
        "strata": dict(
            sorted(collections.Counter(str(pair.stratum) for pair in pairs).items())
        ),
    }


def jsts_items(root: Path) -> tuple[list[Item], Report]:
    """v1 ``jglue.jsts`` pairs: local id = sentence_pair_id, group = image id."""
    pairs = [
        klue.RatedPair(
            local_id=klue.ident(record, "sentence_pair_id", where),
            first=klue.text(record, "sentence1", where),
            second=klue.text(record, "sentence2", where),
            mean=klue.rating(record.get("label"), 0, klue.STS_TOP, where),
            group=jglue.image_id(record, where),
        )
        for where, record in klue.read_jsonl(root / jglue.JSTS_FILE)
    ]
    return [sts_item(pair) for pair in pairs], {
        "inputs": {jglue.JSTS_FILE.as_posix(): file_sha256(root / jglue.JSTS_FILE)},
        "image_ids": len({pair.group for pair in pairs}),
    }


def argq_items(root: Path) -> tuple[list[Item], Report]:
    """v1 ``argq.read`` arguments: local id = row<position>, value = WA."""
    arguments, skipped, records = argq.read(root / argq.FILE)
    items = []
    for argument in arguments:
        topic, text = clean(argument.topic), clean(argument.argument)
        items.append(
            Item(
                local_id=argument.local_id,
                group_key=normalize(argument.argument),
                key=(normalize(topic), normalize(text)),
                state={"topic": topic, "argument": text},
                value=argument.wa,
                audit={"wa": argument.wa, "mace_p": argument.mace},
                topic=argument.topic,
                check=argument.mace,
            )
        )
    return items, {
        "inputs": {argq.FILE.as_posix(): file_sha256(root / argq.FILE)},
        "records": records,
        "non_train_rows_skipped": skipped,
        "topics": len({argument.topic for argument in arguments}),
    }


@dataclasses.dataclass(frozen=True)
class Plan:
    directory: str
    language: str
    scale: Scale
    parse: Callable[[Path], tuple[list[Item], Report]]
    group_rule: str
    agreement: bool = False


def a6h2(
    dirs: Mapping[str, Path],
    *,
    family: str,
    source: str,
    cap: int,
    seed: str,
    plan: Plan,
) -> tuple[Rows, Report]:
    items, info = plan.parse(dirs[plan.directory])
    known = {item.local_id for item in items}
    if len(known) != len(items):
        raise ValueError(f"{family}: duplicate local ids in {source}")
    used = v1_local_ids(dirs, source)
    unknown = sorted(used - known)
    if unknown:
        raise ValueError(
            f"{family}: {len(unknown)} v1 local ids of {source} are not in the "
            f"TRAIN file, e.g. {unknown[:3]}"
        )
    reasons = (
        A6H2_DROPS + C10_DROPS + (("mace_p_disagreement",) if plan.agreement else ())
    )
    drops = collections.Counter(dict.fromkeys(reasons, 0))
    rows, binning = binned(
        items,
        eligible(items, used, drops),
        family=family,
        source=source,
        language=plan.language,
        namespace="a6h2",
        scale=plan.scale,
        seed=seed,
        cap=cap,
        drops=drops,
        agreement=plan.agreement,
    )
    report: Report = {
        **info,
        "candidates": len(items),
        "items": len(items),
        "drops": dict(sorted(drops.items())),
        "v1_local_ids": len(used),
        "v1_exclusion": "local id in v1 rows of the source, or the group key or "
        "normalized state of such an item",
        "group_key": f"{plan.group_rule} (namespace a6h2, as v1 A6H_GROUP_KEYS)",
        **binning,
    }
    if "strata" in info:
        table: dict[tuple[int, int], collections.Counter[str]] = (
            collections.defaultdict(collections.Counter)
        )
        for row in rows:
            cell = (len(row["options"]), row["label"])
            table[cell][row["audit_metadata"]["stratum"]] += 1
        report["strata_after_cap"] = {
            f"L{size}": {
                str(level): dict(sorted(table[size, level].items()))
                for level in range(size)
            }
            for size in LEVELS
        }
    return rows, report


A6H2_PLANS = (
    (
        "a6h2_klue_sts",
        "klue_sts_train",
        Plan("klue", "ko", STS_SCALE, klue_sts_items, "guid"),
    ),
    (
        "a6h2_jsts",
        "jglue_jsts_v1.3_train",
        Plan(
            "jglue",
            "ja",
            STS_SCALE,
            jsts_items,
            "image id (yjcaptions_id before the first '-')",
        ),
    ),
    (
        "a6h2_argq",
        "argq30k_train",
        Plan(
            "argq",
            "en",
            ARGQ_SCALE,
            argq_items,
            "normalized argument",
            agreement=True,
        ),
    ),
)


def _spec(
    family: str,
    source: str,
    cap: int,
    seed: str,
    builder: Callable[..., tuple[Rows, Report]],
    **options: Any,
) -> FamilySpec:
    return FamilySpec(
        arm=ARM,
        family=family,
        source=source,
        build=functools.partial(
            builder, family=family, source=source, cap=cap, seed=seed, **options
        ),
        cap_rows=cap,
        seed=seed,
    )


FAMILIES: tuple[FamilySpec, ...] = (
    *(
        _spec(
            f"mlqepe_{pair.replace('-', '')}",
            MLQEPE_SOURCE,
            MLQEPE_CAP,
            f"h6-mlqepe-{pair}-v1",
            mlqepe,
            pair=pair,
        )
        for pair in MLQEPE_PAIRS
    ),
    _spec(ONESTOP_FAMILY, ONESTOP_SOURCE, ONESTOP_CAP, ONESTOP_SEED, onestop),
    *(
        _spec(family, source, A6H2_CAP, f"h6-{family}-v1", a6h2, plan=plan)
        for family, source, plan in A6H2_PLANS
    ),
)
