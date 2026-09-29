"""Blind label review (PN1) and blind template spot-check (HS1) of published data arms.

Rules: ``v2/data/records/m4-dq-prereg-2026-09-30.md``.

    python3 -m v2.data.dq.blind_review sample-pn1 --rows pn1.train.jsonl --sha256 HEX --out-dir DIR
    python3 -m v2.data.dq.blind_review sample-hs1 --rows train.jsonl --sha256 HEX --out-dir DIR
    python3 -m v2.data.dq.blind_review splits-pn1 --packet DIR/pn1.packet.r1.jsonl \
        --r1 R1.jsonl --r2 R2.jsonl --out R3.packet.jsonl
    python3 -m v2.data.dq.blind_review score-pn1 --sample DIR/pn1.sample.json \
        --key DIR/pn1.key.jsonl --r1 R1.jsonl --r2 R2.jsonl [--r3 R3.jsonl] \
        --out REPORT.json --private PRIVATE.json
    python3 -m v2.data.dq.blind_review fix-pn1 --rows pn1.train.jsonl --sha256 HEX \
        --private PRIVATE.json [--drop-groups GROUPS.txt] --out-rows FIXED.jsonl --receipt FIX.json
    python3 -m v2.data.dq.blind_review rebuild-pn1r2 --rows pn1.train.jsonl --sha256 HEX \
        --private PRIVATE.json --out-rows PN1R2.jsonl --receipt REBUILD.json
    python3 -m v2.data.dq.blind_review sample-pn1r2 --rows PN1R2.jsonl --sha256 HEX \
        --exclude-key DIR/pn1.key.jsonl --out-dir DIR2
    python3 -m v2.data.dq.blind_review adjudication-hs1 --key DIR/hs1.key.jsonl \
        --packet DIR/hs1.packet.F.jsonl [...] --answers A.jsonl [...] \
        --out-packet ADJ.packet.jsonl --out-key ADJ.key.jsonl
    python3 -m v2.data.dq.blind_review score-hs1 --sample DIR/hs1.sample.json \
        --key DIR/hs1.key.jsonl --answers A.jsonl [...] --adj-key ADJ.key.jsonl \
        --adjudication ADJ.jsonl --confirmed CONFIRMED.json --out REPORT.json --private PRIVATE.json

A packet item carries only what a model sees at training time (instructions, state, option
descriptions) under an opaque review id. Keys (row ids, gold, family, audit metadata) stay on the
node until every blind answer is in. Every output file is created exclusively (never overwritten).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical
from v2.data.m4.pn1_text import GROUP_OF, LANGUAGE_NAMES

PN1_SALT = "dq-pn1-review-v1"
PN1_PACKET_SALT = "dq-pn1-packet-v1"
PN1_R2_SALT = "dq-pn1-r2-order-v1"
PN1_R3_SALT = "dq-pn1-r3-order-v1"
PN1_BALANCE_SALT = "dq-pn1-balance-v1"
PN1_PER_CELL = 8
HS1_SALT = "dq-hs1-spot-v1"
HS1_PACKET_SALT = "dq-hs1-packet-v1"
HS1_ADJ_SALT = "dq-hs1-adj-v1"
HS1_PER_CELL = 17
BOOT_SALT = "dq-boot-v1"
BOOT_REPS = 10_000

PN1_THRESHOLDS = {
    "error_max": 0.05,
    "error_upper_max": 0.08,
    "weighted_error_max": 0.05,
    "cell_fail_cp_lower": 0.05,
    "margin_yes_min": 0.80,
    "margin_no_max": 0.20,
}
YES, NO = "yes", "no"
CONFIDENCE = ("high", "medium", "low")
FLUENCY = ("ok", "awkward", "ungrammatical")
HS1_FLAGS = (
    "garbled",
    "contradiction",
    "ambiguous",
    "options_malformed",
    "gold_unclear",
    "other",
)
ADJ_VERDICTS = ("A", "B", "both", "neither")
PN1_PACKET_FIELDS = ("rid", "language", "instructions", "state", "options")
HS1_PACKET_FIELDS = ("rid", "task_type", "instructions", "state", "options")


# ------------------------------------------------------------------ io


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_new(path: Path, text: str) -> str:
    """Create ``path`` exclusively (mode 0600) and return the SHA-256 of ``text``."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    return sha(text)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    return write_new(path, "".join(_dumps(row) + "\n" for row in rows))


def write_json(path: Path, payload: Any) -> str:
    return write_new(
        path, json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    )


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: sha256 {actual} != pinned {expected_sha256}")
    return read_jsonl(path)


# ------------------------------------------------------------------ statistics

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    total = sum(
        math.exp(
            base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        )
        for i in range(k + 1)
    )
    return min(total, 1.0)


def _bisect(f: Callable[[float], float], increasing: bool) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(mid) < 0) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    lower = (
        0.0
        if k == 0
        else _bisect(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2, True)
    )
    upper = 1.0 if k == n else _bisect(lambda p: _binom_cdf(k, n, p) - alpha / 2, False)
    return (lower, upper)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("kappa needs two equal, nonempty label lists")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)


def _rng(salt: str) -> random.Random:
    return random.Random(int(sha(salt), 16))


def stratified_bootstrap(
    strata: Mapping[str, Sequence[Any]],
    statistic: Callable[[Mapping[str, Sequence[Any]]], float],
    reps: int = BOOT_REPS,
    salt: str = BOOT_SALT,
) -> tuple[float, float]:
    """Percentile 95% interval of ``statistic`` over within-stratum resamples."""
    rng = _rng(salt)
    keys = sorted(strata)
    values = []
    for _ in range(reps):
        draw = {
            key: [strata[key][rng.randrange(len(strata[key]))] for _ in strata[key]]
            for key in keys
        }
        values.append(statistic(draw))
    values.sort()
    return (values[int(0.025 * reps)], values[min(reps - 1, int(0.975 * reps))])


def proportion(k: int, n: int) -> dict[str, Any]:
    lo, hi = clopper_pearson(k, n)
    wlo, whi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 6) if n else None,
        "cp95": [round(lo, 6), round(hi, 6)],
        "wilson95": [round(wlo, 6), round(whi, 6)],
    }


# ------------------------------------------------------------------ PN1 sampling


def noul_gold(row: Mapping[str, Any]) -> str:
    if row.get("task_type") != "noul":
        raise ValueError(f"{row.get('id')}: not a Noul row")
    key = row["options"][row["label"]]["key"]
    if key not in ("true", "false"):
        raise ValueError(f"{row['id']}: unexpected Noul key {key!r}")
    return YES if key == "true" else NO


def pn1_cell(row: Mapping[str, Any]) -> str:
    return f"{row['language']}|{GROUP_OF[row['family']]}|{noul_gold(row)}"


@dataclasses.dataclass(frozen=True)
class Pn1Round:
    """Salts, allocation and review-id prefix of one PN1 review round."""

    name: str
    salt: str
    packet_salt: str
    r2_salt: str
    per_cell: int
    prefix: str


ROUND1 = Pn1Round("r1", PN1_SALT, PN1_PACKET_SALT, PN1_R2_SALT, PN1_PER_CELL, "p")
ROUND2 = Pn1Round(
    "pn1r2",
    "dq-pn1-r2-review-v1",
    "dq-pn1-r2-packet-v1",
    "dq-pn1-r2-r2-order-v1",
    12,
    "q",
)


def sample_pn1(
    rows: Sequence[Mapping[str, Any]],
    per_cell: int = PN1_PER_CELL,
    salt: str = PN1_SALT,
    exclude: Iterable[str] = (),
) -> tuple[list[Mapping[str, Any]], Counter]:
    """``per_cell`` rows per language x group x gold cell, in salted-hash order, one per group.
    ``exclude`` ids (an earlier round's sample) count in the population but are never drawn.
    """
    population: Counter = Counter()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"{row.get('id')}: not a TRAIN row")
        population[pn1_cell(row)] += 1
    skip = set(exclude)
    taken: Counter = Counter()
    used_groups: set[str] = set()
    picked = []
    for row in sorted(rows, key=lambda r: sha(f"{salt}:{r['id']}")):
        cell = pn1_cell(row)
        if (
            row["id"] in skip
            or taken[cell] >= per_cell
            or row["group_id"] in used_groups
        ):
            continue
        picked.append(row)
        taken[cell] += 1
        used_groups.add(row["group_id"])
    return picked, population


def pn1_packet_item(rid: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rid": rid,
        "language": LANGUAGE_NAMES[row["language"]],
        "instructions": row["instructions"],
        "state": row["state"],
        "options": [option["description"] for option in row["options"]],
    }


def pn1_key_item(rid: str, row: Mapping[str, Any]) -> dict[str, Any]:
    audit = row.get("audit_metadata") or {}
    judge = audit.get("judge") or {}
    return {
        "rid": rid,
        "id": row["id"],
        "group_id": row["group_id"],
        "language": row["language"],
        "group": GROUP_OF[row["family"]],
        "family": row["family"],
        "gold": noul_gold(row),
        "cell": pn1_cell(row),
        "judge_p_yes": judge.get("label_p_yes"),
        "fluency_p_yes": judge.get("fluency_p_yes"),
        "edited_position": audit.get("edited_position"),
    }


def assert_blind(
    items: Sequence[Mapping[str, Any]], fields: Sequence[str], secrets: Iterable[str]
) -> None:
    """Every packet item has exactly ``fields``; no secret string occurs in the packet."""
    for item in items:
        if tuple(sorted(item)) != tuple(sorted(fields)):
            raise ValueError(f"packet item {item.get('rid')} has fields {sorted(item)}")
    text = "".join(_dumps(item) for item in items)
    leaked = sorted({s for s in secrets if s and s in text})
    if leaked:
        raise ValueError(f"packet leaks {len(leaked)} key strings, e.g. {leaked[0]!r}")


def build_pn1(
    rows: Sequence[Mapping[str, Any]],
    rows_sha256: str,
    review: Pn1Round = ROUND1,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    skip = set(exclude)
    picked, population = sample_pn1(rows, review.per_cell, review.salt, skip)
    ordered = sorted(picked, key=lambda r: sha(f"{review.packet_salt}:{r['id']}"))
    rids = {
        row["id"]: f"{review.prefix}{index + 1:03d}"
        for index, row in enumerate(ordered)
    }
    packet = [pn1_packet_item(rids[row["id"]], row) for row in ordered]
    key = [pn1_key_item(rids[row["id"]], row) for row in ordered]
    secrets = (
        {row["id"] for row in ordered}
        | {row["group_id"] for row in ordered}
        | set(GROUP_OF)
    )
    assert_blind(packet, PN1_PACKET_FIELDS, secrets)
    packet_r2 = sorted(packet, key=lambda item: sha(f"{review.r2_salt}:{item['rid']}"))
    sampled = Counter(item["cell"] for item in key)
    return {
        "packet_r1": packet,
        "packet_r2": packet_r2,
        "key": key,
        "sample": {
            "schema": "dev2-dq-pn1-sample/1",
            "round": review.name,
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": review.salt,
            "per_cell": review.per_cell,
            "excluded": len(skip),
            "n": len(key),
            "population": dict(sorted(population.items())),
            "sampled": dict(sorted(sampled.items())),
            "thresholds": PN1_THRESHOLDS,
        },
    }


# ------------------------------------------------------------------ HS1 sampling


def option_key(row: Mapping[str, Any]) -> str:
    return row["options"][row["label"]]["key"]


def sample_hs1(
    rows: Sequence[Mapping[str, Any]], per_cell: int = HS1_PER_CELL
) -> tuple[list[Mapping[str, Any]], Counter]:
    """``per_cell`` rows per family x task type, in salted-hash order, one per group."""
    population: Counter = Counter()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"{row.get('id')}: not a TRAIN row")
        population[f"{row['family']}|{row['task_type']}"] += 1
    taken: Counter = Counter()
    used_groups: set[str] = set()
    picked = []
    for row in sorted(rows, key=lambda r: sha(f"{HS1_SALT}:{r['id']}")):
        cell = f"{row['family']}|{row['task_type']}"
        if taken[cell] >= per_cell or row["group_id"] in used_groups:
            continue
        picked.append(row)
        taken[cell] += 1
        used_groups.add(row["group_id"])
    return picked, population


def build_hs1(rows: Sequence[Mapping[str, Any]], rows_sha256: str) -> dict[str, Any]:
    picked, population = sample_hs1(rows)
    ordered = sorted(picked, key=lambda r: sha(f"{HS1_PACKET_SALT}:{r['id']}"))
    packets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = []
    for index, row in enumerate(ordered):
        rid = f"h{index + 1:03d}"
        audit = row.get("audit_metadata") or {}
        packets[row["family"]].append(
            {
                "rid": rid,
                "task_type": row["task_type"],
                "instructions": row["instructions"],
                "state": row["state"],
                "options": [
                    {"key": o["key"], "description": o["description"]}
                    for o in row["options"]
                ],
            }
        )
        key.append(
            {
                "rid": rid,
                "id": row["id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "task_type": row["task_type"],
                "render_template": row["render_template"],
                "kind": audit.get("kind"),
                "subtype": audit.get("subtype"),
                "slice": audit.get("slice"),
                "gold_key": option_key(row),
                "option_refs": audit.get("option_refs"),
            }
        )
    secrets = {row["id"] for row in ordered} | {row["group_id"] for row in ordered}
    for items in packets.values():
        assert_blind(items, HS1_PACKET_FIELDS, secrets)
    sampled = Counter(f"{k['family']}|{k['task_type']}" for k in key)
    return {
        "packets": dict(packets),
        "key": key,
        "sample": {
            "schema": "dev2-dq-hs1-sample/1",
            "rows_sha256": rows_sha256,
            "rows": len(rows),
            "salt": HS1_SALT,
            "per_cell": HS1_PER_CELL,
            "n": len(key),
            "population": dict(sorted(population.items())),
            "sampled": dict(sorted(sampled.items())),
        },
    }


# ------------------------------------------------------------------ answers


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def read_answers(
    paths: Sequence[Path], expected: Iterable[str], id_field: str = "rid"
) -> dict[str, dict[str, Any]]:
    """One answer per expected review id across ``paths``; anything else is an error."""
    answers: dict[str, dict[str, Any]] = {}
    for path in paths:
        for item in read_jsonl(path):
            rid = str(item.get(id_field, "")).strip()
            if rid in answers:
                raise ValueError(f"{path}: duplicate answer for {rid}")
            answers[rid] = item
    wanted = set(expected)
    missing, extra = sorted(wanted - set(answers)), sorted(set(answers) - wanted)
    if missing or extra:
        raise ValueError(
            f"answers: missing {missing[:5]} ({len(missing)}), extra {extra[:5]}"
        )
    return answers


def pn1_answer(item: Mapping[str, Any]) -> str:
    value = _norm(item.get("answer"))
    if value not in (YES, NO):
        raise ValueError(f"{item.get('rid')}: answer must be yes/no, got {value!r}")
    return value


def splits_pn1(
    packet: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    split = [
        item
        for item in packet
        if pn1_answer(r1[item["rid"]]) != pn1_answer(r2[item["rid"]])
    ]
    return sorted(split, key=lambda item: sha(f"{PN1_R3_SALT}:{item['rid']}"))


# ------------------------------------------------------------------ PN1 scoring


def pn1_items(
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
    r3: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    items = []
    for row in key:
        rid = row["rid"]
        a, b = pn1_answer(r1[rid]), pn1_answer(r2[rid])
        votes = [a, b]
        c = None
        if a != b:
            if r3 is None or rid not in r3:
                raise ValueError(f"{rid}: R1/R2 split without an R3 answer")
            c = pn1_answer(r3[rid])
            votes.append(c)
        majority = YES if votes.count(YES) > votes.count(NO) else NO
        fluency = [
            [_norm(x) for x in (r.get("fluency") or [])] for r in (r1[rid], r2[rid])
        ]
        items.append(
            {
                **row,
                "r1": a,
                "r2": b,
                "r3": c,
                "majority": majority,
                "error": majority != row["gold"],
                "r1_confidence": _norm(r1[rid].get("confidence")),
                "r2_confidence": _norm(r2[rid].get("confidence")),
                "fluency": fluency,
                "notes": [str(r1[rid].get("note", "")), str(r2[rid].get("note", ""))]
                + ([str(r3[rid].get("note", ""))] if c is not None and r3 else []),
            }
        )
    return items


def _both_flag(item: Mapping[str, Any], position: int) -> bool:
    return all(
        len(f) > position and f[position] == "ungrammatical" for f in item["fluency"]
    )


def weighted_error(
    by_cell: Mapping[str, Sequence[Mapping[str, Any]]], population: Mapping[str, int]
) -> float:
    total = sum(population[cell] for cell in by_cell)
    return sum(
        population[cell] / total * (sum(i["error"] for i in items) / len(items))
        for cell, items in by_cell.items()
        if items
    )


def pn1_verdict(
    items: Sequence[Mapping[str, Any]], population: Mapping[str, int]
) -> dict[str, Any]:
    n, errors = len(items), sum(i["error"] for i in items)
    _, upper = clopper_pearson(errors, n)
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    lang_group: Counter = Counter()
    lang_group_n: Counter = Counter()
    for item in items:
        by_cell[item["cell"]].append(item)
        lang_group[f"{item['language']}|{item['group']}"] += item["error"]
        lang_group_n[f"{item['language']}|{item['group']}"] += 1
    weighted = weighted_error(by_cell, population)
    failing = sorted(
        cell
        for cell, k in lang_group.items()
        if clopper_pearson(k, lang_group_n[cell])[0]
        > PN1_THRESHOLDS["cell_fail_cp_lower"]
    )
    p1 = (
        errors / n <= PN1_THRESHOLDS["error_max"]
        and upper <= PN1_THRESHOLDS["error_upper_max"]
    )
    p2 = weighted <= PN1_THRESHOLDS["weighted_error_max"]
    p3 = not failing
    return {
        "n": n,
        "errors": errors,
        "error": round(errors / n, 6),
        "error_cp95_upper": round(upper, 6),
        "weighted_error": round(weighted, 6),
        "failing_language_group_cells": failing,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "verdict": "PASS" if p1 and p2 and p3 else "FAIL",
    }


def _breakdown(items: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item[field])].append(item)
    return {
        name: proportion(sum(i["error"] for i in group), len(group))
        for name, group in sorted(groups.items())
    }


def score_pn1(
    sample: Mapping[str, Any],
    key: Sequence[Mapping[str, Any]],
    r1: Mapping[str, Mapping[str, Any]],
    r2: Mapping[str, Mapping[str, Any]],
    r3: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    items = pn1_items(key, r1, r2, r3)
    population = sample["population"]
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_cell[item["cell"]].append(item)
    weighted_ci = stratified_bootstrap(
        by_cell, lambda draw: weighted_error(draw, population)
    )
    kappa = cohen_kappa([i["r1"] for i in items], [i["r2"] for i in items])
    kappa_ci = stratified_bootstrap(
        by_cell,
        lambda draw: cohen_kappa(
            [i["r1"] for g in draw.values() for i in g],
            [i["r2"] for g in draw.values() for i in g],
        ),
        salt=BOOT_SALT + ":kappa",
    )
    n = len(items)
    splits = [i for i in items if i["r3"] is not None]
    swap = [i for i in items if i["group"] == "swap" and i["edited_position"] in (0, 1)]
    verdict = pn1_verdict(items, population)
    for item in items:
        item["language_group"] = f"{item['language']}|{item['group']}"
    report = {
        "schema": "dev2-dq-pn1-review/1",
        "sample": {
            k: sample[k] for k in ("rows_sha256", "rows", "salt", "per_cell", "n")
        },
        "verdict": verdict,
        "thresholds": PN1_THRESHOLDS,
        "error_unweighted": proportion(verdict["errors"], n),
        "error_weighted": {
            "rate": verdict["weighted_error"],
            "bootstrap95": [round(weighted_ci[0], 6), round(weighted_ci[1], 6)],
            "reps": BOOT_REPS,
        },
        "agreement_with_gold": {
            "r1": proportion(sum(i["r1"] == i["gold"] for i in items), n),
            "r2": proportion(sum(i["r2"] == i["gold"] for i in items), n),
            "majority": proportion(sum(not i["error"] for i in items), n),
            "r3_on_splits": proportion(
                sum(i["r3"] == i["gold"] for i in splits), len(splits)
            ),
        },
        "inter_reviewer": {
            "raw": proportion(sum(i["r1"] == i["r2"] for i in items), n),
            "kappa": round(kappa, 6),
            "kappa_bootstrap95": [round(kappa_ci[0], 6), round(kappa_ci[1], 6)],
            "splits": len(splits),
        },
        "direction": {
            "false_yes": sum(i["error"] and i["gold"] == NO for i in items),
            "false_no": sum(i["error"] and i["gold"] == YES for i in items),
        },
        "by_language": _breakdown(items, "language"),
        "by_group": _breakdown(items, "group"),
        "by_gold": _breakdown(items, "gold"),
        "by_family": _breakdown(items, "family"),
        "by_language_group": _breakdown(items, "language_group"),
        "fluency": {
            "edited_sentence_ungrammatical_both": proportion(
                sum(_both_flag(i, i["edited_position"]) for i in swap), len(swap)
            ),
            "any_sentence_ungrammatical_both": proportion(
                sum(_both_flag(i, 0) or _both_flag(i, 1) for i in items), n
            ),
        },
        "confidence": {
            "r1": dict(Counter(i["r1_confidence"] for i in items)),
            "r2": dict(Counter(i["r2_confidence"] for i in items)),
        },
    }
    private = {
        "schema": "dev2-dq-pn1-review-private/1",
        "errors": [i for i in items if i["error"]],
        "splits": splits,
        "items": items,
    }
    return report, private


# ------------------------------------------------------------------ PN1 fix rule (prereg §3.6)


def balance_cells(
    rows: Sequence[Mapping[str, Any]], protect: Iterable[str] = ()
) -> tuple[list[Mapping[str, Any]], set[str]]:
    """Exactly 50/50 labels per language x group cell: surplus rows of the larger label go
    in salted-hash order, ``protect`` ids (reviewed rows) only when nothing else is left.
    """
    keep_last = set(protect)
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[f"{row['language']}|{GROUP_OF[row['family']]}"].append(row)
    dropped: set[str] = set()
    for members in by_cell.values():
        yes = [r for r in members if noul_gold(r) == YES]
        no = [r for r in members if noul_gold(r) == NO]
        surplus = yes if len(yes) > len(no) else no
        excess = abs(len(yes) - len(no))
        order = sorted(
            surplus,
            key=lambda r: (r["id"] in keep_last, sha(f"{PN1_BALANCE_SALT}:{r['id']}")),
        )
        dropped.update(r["id"] for r in order[:excess])
    return [r for r in rows if r["id"] not in dropped], dropped


PN1R2_NEAR_DROP = ("es", "fr", "ar", "ru", "ko")
PN1R2_NAME_DROP = ("ru",)


def rebuild_pn1r2(
    rows: Sequence[Mapping[str, Any]], private: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Amendment 1: D1 (pn-near in PN1R2_NEAR_DROP), D2 (pn-name in PN1R2_NAME_DROP), F1 (the
    round-1 gold errors), then the balance repair. No margin filter."""
    items = private["items"]
    reviewed = {i["id"] for i in items}
    errors = {i["id"] for i in items if i["error"]}
    reasons: Counter = Counter()
    kept = []
    for row in rows:
        if row["family"] == "pn-near" and row["language"] in PN1R2_NEAR_DROP:
            reasons["D1_near"] += 1
        elif row["family"] == "pn-name" and row["language"] in PN1R2_NAME_DROP:
            reasons["D2_ru_name"] += 1
        elif row["id"] in errors:
            reasons["F1_error"] += 1
        else:
            kept.append(row)
    fixed, balance = balance_cells(kept, reviewed)
    reasons["balance"] = len(balance)

    def cells(members: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(pn1_cell(r) for r in members).items()))

    receipt = {
        "schema": "dev2-dq-pn1r2-rebuild/1",
        "rules": {"near_drop": PN1R2_NEAR_DROP, "name_drop": PN1R2_NAME_DROP},
        "dropped": dict(reasons),
        "rows_before": len(rows),
        "rows_after": len(fixed),
        "cells_before": cells(rows),
        "cells_after": cells(fixed),
        "languages_after": dict(sorted(Counter(r["language"] for r in fixed).items())),
    }
    return fixed, receipt


def fix_pn1(
    rows: Sequence[Mapping[str, Any]],
    private: Mapping[str, Any],
    population: Mapping[str, int],
    drop_groups: Iterable[str] = (),
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Apply F0 (whole groups from an audit), F1, F2 and, only if still failing, F3; then
    restore label balance per language x group cell. Re-evaluation uses the reviewed rows
    that survive the review-independent steps (F0, F2, F3)."""
    items = private["items"]
    reviewed = {i["id"] for i in items}
    error_ids = {i["id"] for i in items if i["error"]}
    dropped_groups = set(drop_groups)
    before = pn1_verdict(items, population)
    failing = set(before["failing_language_group_cells"])
    steps = {"F0_groups": len(dropped_groups), "F2_cells": sorted(failing)}

    def cell_of(row: Mapping[str, Any]) -> str:
        return f"{row['language']}|{GROUP_OF[row['family']]}"

    def keep_f02(row: Mapping[str, Any]) -> bool:
        return row["group_id"] not in dropped_groups and cell_of(row) not in failing

    kept_rows = [r for r in rows if keep_f02(r)]
    kept_ids = {r["id"] for r in kept_rows}
    survivors = [i for i in items if i["id"] in kept_ids]
    after = (
        pn1_verdict(survivors, Counter(pn1_cell(r) for r in kept_rows))
        if survivors
        else None
    )
    margin = False
    if after is None or not (after["P1"] and after["P2"]):
        margin = True

        def confident(row: Mapping[str, Any]) -> bool:
            p = ((row.get("audit_metadata") or {}).get("judge") or {}).get(
                "label_p_yes"
            )
            if p is None:
                return True
            if noul_gold(row) == YES:
                return p >= PN1_THRESHOLDS["margin_yes_min"]
            return p <= PN1_THRESHOLDS["margin_no_max"]

        kept_rows = [r for r in kept_rows if confident(r)]
        kept_ids = {r["id"] for r in kept_rows}
        survivors = [i for i in survivors if i["id"] in kept_ids]
        after = (
            pn1_verdict(survivors, Counter(pn1_cell(r) for r in kept_rows))
            if survivors
            else None
        )
    steps["F3_margin_applied"] = margin
    steps["reviewed_survivors"] = len(survivors)
    kept = [r for r in rows if r["id"] in kept_ids and r["id"] not in error_ids]
    fixed, balance_drop = balance_cells(kept, reviewed)
    receipt = {
        "schema": "dev2-dq-pn1-fix/1",
        "before": before,
        "after": after,
        "steps": {
            **steps,
            "F1_error_rows": len(error_ids),
            "balance_rows": len(balance_drop),
            "rows_before": len(rows),
            "rows_after": len(fixed),
        },
        "verdict": "FIXED-PASS" if after and after["verdict"] == "PASS" else "FAIL",
    }
    return fixed, receipt


# ------------------------------------------------------------------ HS1 adjudication and scoring


def adjudication_hs1(
    key: Sequence[Mapping[str, Any]],
    packet: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blind adjudication items: answer != gold (A/B in salted order) or a raised flag."""
    items, adj_key = [], []
    candidates = []
    for row in key:
        answer = answers[row["rid"]]
        given = str(answer.get("answer", "")).strip()
        flags = [_norm(f) for f in answer.get("flags") or []]
        if given != row["gold_key"]:
            candidates.append((row, "disagree", given, flags, answer))
        elif flags:
            candidates.append((row, "flag", given, flags, answer))
    candidates.sort(key=lambda c: sha(f"{HS1_ADJ_SALT}:{c[0]['rid']}"))
    for index, (row, kind, given, flags, answer) in enumerate(candidates):
        aid = f"a{index + 1:03d}"
        base = {
            k: packet[row["rid"]][k]
            for k in ("task_type", "instructions", "state", "options")
        }
        item: dict[str, Any] = {"aid": aid, "type": kind, **base}
        entry: dict[str, Any] = {"aid": aid, "rid": row["rid"], "type": kind}
        if kind == "disagree":
            gold_first = int(sha(f"{HS1_ADJ_SALT}:order:{row['rid']}"), 16) % 2 == 0
            a, b = (row["gold_key"], given) if gold_first else (given, row["gold_key"])
            item["candidates"] = {"A": a, "B": b}
            entry.update({"A": a, "B": b, "gold_is": "A" if gold_first else "B"})
        else:
            item["reported_issue"] = {
                "flags": flags,
                "note": str(answer.get("note", "")),
            }
        items.append(item)
        adj_key.append(entry)
    return items, adj_key


def score_hs1(
    sample: Mapping[str, Any],
    key: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    adj_key: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Mapping[str, Any]],
    confirmed: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_rid = {row["rid"]: row for row in key}
    adj_by_rid = {entry["rid"]: entry for entry in adj_key}
    rows = []
    for row in key:
        answer = answers[row["rid"]]
        entry = adj_by_rid.get(row["rid"])
        outcome = None
        against = False
        if entry is not None:
            verdict = adjudication[entry["aid"]]
            if entry["type"] == "disagree":
                outcome = str(verdict.get("verdict", "")).strip()
                if outcome not in ADJ_VERDICTS:
                    raise ValueError(f"{entry['aid']}: bad verdict {outcome!r}")
                against = outcome != entry["gold_is"]
            else:
                outcome = _norm(verdict.get("verdict"))
                if outcome not in ("defect", "no_defect"):
                    raise ValueError(f"{entry['aid']}: bad flag verdict {outcome!r}")
                against = outcome == "defect"
        status = confirmed.get(row["rid"], {}).get("status")
        if against and status not in ("defect", "not_defect"):
            raise ValueError(
                f"{row['rid']}: candidate defect without a root-cause status"
            )
        rows.append(
            {
                **row,
                "answer": str(answer.get("answer", "")).strip(),
                "agree": str(answer.get("answer", "")).strip() == row["gold_key"],
                "flags": [_norm(f) for f in answer.get("flags") or []],
                "adjudication": outcome,
                "candidate_defect": against,
                "confirmed_defect": against and status == "defect",
                "root_cause": confirmed.get(row["rid"]),
            }
        )
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    per_family = {}
    for family, members in sorted(families.items()):
        n = len(members)
        per_family[family] = {
            "n": n,
            "reviewer_agreement": proportion(sum(r["agree"] for r in members), n),
            "flagged": sum(bool(r["flags"]) for r in members),
            "adjudicated": sum(r["adjudication"] is not None for r in members),
            "candidate_defects": proportion(
                sum(r["candidate_defect"] for r in members), n
            ),
            "confirmed_defects": proportion(
                sum(r["confirmed_defect"] for r in members), n
            ),
            "by_task_type": {
                t: proportion(
                    sum(r["agree"] for r in members if r["task_type"] == t),
                    sum(r["task_type"] == t for r in members),
                )
                for t in sorted({r["task_type"] for r in members})
            },
        }
    total_confirmed = sum(r["confirmed_defect"] for r in rows)
    report = {
        "schema": "dev2-dq-hs1-spot/1",
        "sample": {
            k: sample[k] for k in ("rows_sha256", "rows", "salt", "per_cell", "n")
        },
        "per_family": per_family,
        "pooled": {
            "reviewer_agreement": proportion(sum(r["agree"] for r in rows), len(rows)),
            "candidate_defects": proportion(
                sum(r["candidate_defect"] for r in rows), len(rows)
            ),
            "confirmed_defects": proportion(total_confirmed, len(rows)),
        },
        "verdict": "PASS" if total_confirmed == 0 else "DEFECTS-FOUND",
    }
    private = {
        "schema": "dev2-dq-hs1-spot-private/1",
        "rows": rows,
        "unknown_rids": sorted(set(confirmed) - set(by_rid)),
    }
    return report, private


# ------------------------------------------------------------------ CLI


def _answers_for(
    paths: Sequence[Path], rids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    return read_answers(list(paths), rids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sample-pn1", "sample-hs1"):
        p = sub.add_parser(name)
        p.add_argument("--rows", type=Path, required=True)
        p.add_argument("--sha256", required=True)
        p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("splits-pn1")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--r1", type=Path, required=True)
    p.add_argument("--r2", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score-pn1")
    for flag in ("--sample", "--key", "--r1", "--r2", "--out", "--private"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--r3", type=Path)
    p = sub.add_parser("fix-pn1")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--drop-groups", type=Path)
    p.add_argument("--out-rows", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p = sub.add_parser("rebuild-pn1r2")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--out-rows", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p = sub.add_parser("sample-pn1r2")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--exclude-key", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("adjudication-hs1")
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--packet", type=Path, action="append", required=True)
    p.add_argument("--answers", type=Path, action="append", required=True)
    p.add_argument("--out-packet", type=Path, required=True)
    p.add_argument("--out-key", type=Path, required=True)
    p = sub.add_parser("score-hs1")
    for flag in (
        "--sample",
        "--key",
        "--adj-key",
        "--adjudication",
        "--confirmed",
        "--out",
        "--private",
    ):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--answers", type=Path, action="append", required=True)
    args = parser.parse_args(argv)

    if args.command in ("sample-pn1", "sample-hs1"):
        rows = load_rows(args.rows, args.sha256)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "sample-pn1":
            built = build_pn1(rows, args.sha256)
            hashes = {
                "packet_r1": write_jsonl(
                    args.out_dir / "pn1.packet.r1.jsonl", built["packet_r1"]
                ),
                "packet_r2": write_jsonl(
                    args.out_dir / "pn1.packet.r2.jsonl", built["packet_r2"]
                ),
                "key": write_jsonl(args.out_dir / "pn1.key.jsonl", built["key"]),
            }
            name = "pn1.sample.json"
        else:
            built = build_hs1(rows, args.sha256)
            hashes = {
                f"packet_{family}": write_jsonl(
                    args.out_dir / f"hs1.packet.{family}.jsonl", items
                )
                for family, items in sorted(built["packets"].items())
            }
            hashes["key"] = write_jsonl(args.out_dir / "hs1.key.jsonl", built["key"])
            name = "hs1.sample.json"
        sample = {**built["sample"], "files_sha256": hashes}
        write_json(args.out_dir / name, sample)
        print(json.dumps({k: sample[k] for k in ("n", "sampled")}, sort_keys=True))
        return 0
    if args.command == "splits-pn1":
        packet = read_jsonl(args.packet)
        rids = [item["rid"] for item in packet]
        r1 = _answers_for([args.r1], rids)
        r2 = _answers_for([args.r2], rids)
        split = splits_pn1(packet, r1, r2)
        write_jsonl(args.out, split)
        print(json.dumps({"items": len(packet), "splits": len(split)}))
        return 0
    if args.command == "score-pn1":
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
        key = read_jsonl(args.key)
        rids = [row["rid"] for row in key]
        r1 = _answers_for([args.r1], rids)
        r2 = _answers_for([args.r2], rids)
        split_rids = [rid for rid in rids if pn1_answer(r1[rid]) != pn1_answer(r2[rid])]
        r3 = _answers_for([args.r3], split_rids) if args.r3 else None
        report, private = score_pn1(sample, key, r1, r2, r3)
        report["inputs_sha256"] = {
            name: file_sha256(path)
            for name, path in (
                ("key", args.key),
                ("r1", args.r1),
                ("r2", args.r2),
                ("r3", args.r3),
            )
            if path is not None
        }
        write_json(args.out, report)
        write_json(args.private, private)
        print(json.dumps(report["verdict"], sort_keys=True))
        return 0
    if args.command == "fix-pn1":
        rows = load_rows(args.rows, args.sha256)
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
        private = json.loads(args.private.read_text(encoding="utf-8"))
        groups = (
            [
                g.strip()
                for g in args.drop_groups.read_text(encoding="utf-8").splitlines()
                if g.strip()
            ]
            if args.drop_groups
            else []
        )
        fixed, receipt = fix_pn1(rows, private, sample["population"], groups)
        receipt["rows_sha256_before"] = args.sha256
        receipt["rows_sha256_after"] = write_jsonl(args.out_rows, fixed)
        write_json(args.receipt, receipt)
        print(
            json.dumps(
                {"verdict": receipt["verdict"], **receipt["steps"]}, sort_keys=True
            )
        )
        return 0
    if args.command == "rebuild-pn1r2":
        rows = load_rows(args.rows, args.sha256)
        private = json.loads(args.private.read_text(encoding="utf-8"))
        fixed, receipt = rebuild_pn1r2(rows, private)
        ordered = sorted(fixed, key=lambda r: r["id"])
        receipt["rows_sha256_before"] = args.sha256
        receipt["rows_sha256_after"] = write_new(
            args.out_rows, "".join(canonical(row) + "\n" for row in ordered)
        )
        write_json(args.receipt, receipt)
        print(
            json.dumps(
                {k: receipt[k] for k in ("dropped", "rows_after", "languages_after")},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "sample-pn1r2":
        rows = load_rows(args.rows, args.sha256)
        exclude = [row["id"] for row in read_jsonl(args.exclude_key)]
        built = build_pn1(rows, args.sha256, ROUND2, exclude)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        hashes = {
            "packet_r1": write_jsonl(
                args.out_dir / "pn1r2.packet.r1.jsonl", built["packet_r1"]
            ),
            "packet_r2": write_jsonl(
                args.out_dir / "pn1r2.packet.r2.jsonl", built["packet_r2"]
            ),
            "key": write_jsonl(args.out_dir / "pn1r2.key.jsonl", built["key"]),
            "exclude_key": file_sha256(args.exclude_key),
        }
        sample = {**built["sample"], "files_sha256": hashes}
        write_json(args.out_dir / "pn1r2.sample.json", sample)
        print(json.dumps({k: sample[k] for k in ("n", "sampled")}, sort_keys=True))
        return 0
    if args.command == "adjudication-hs1":
        key = read_jsonl(args.key)
        packet = {
            item["rid"]: item for path in args.packet for item in read_jsonl(path)
        }
        answers = _answers_for(args.answers, [row["rid"] for row in key])
        items, adj_key = adjudication_hs1(key, packet, answers)
        write_jsonl(args.out_packet, items)
        write_jsonl(args.out_key, adj_key)
        print(json.dumps({"rows": len(key), "adjudication_items": len(items)}))
        return 0
    if args.command == "score-hs1":
        sample = json.loads(args.sample.read_text(encoding="utf-8"))
        key = read_jsonl(args.key)
        answers = _answers_for(args.answers, [row["rid"] for row in key])
        adj_key = read_jsonl(args.adj_key)
        adjudication = (
            read_answers(
                [args.adjudication], [e["aid"] for e in adj_key], id_field="aid"
            )
            if adj_key
            else {}
        )
        confirmed = json.loads(args.confirmed.read_text(encoding="utf-8"))
        report, private = score_hs1(
            sample, key, answers, adj_key, adjudication, confirmed
        )
        write_json(args.out, report)
        write_json(args.private, private)
        print(
            json.dumps(
                {"verdict": report["verdict"], **report["pooled"]}, sort_keys=True
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
