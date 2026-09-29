"""HS1 audits (prereg §5 as amended): balance, shortcuts, heuristics, lengths, names, structure.

    python3 -m v2.data.hs1.audit local --train T --dev D --out-dir DIR [--workers 8]
    python3 -m v2.data.hs1.audit struct --candidates T --candidates D --protected P.jsonl ... \
        --private-receipt PRIV.json --public-receipt PUB.json

``local`` needs only the HS1 files. ``struct`` reads protected panel files (on
node A only); its private receipt names protected rows, the public one holds
counts only.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest
from v2.data.hs1.core import mask

BANNED = (
    "long_policy",
    "multi_hop",
    "judge_hard",
    "temporal_numeric",
    "probability",
    "trap",
    "ambiguous",
    "tradeoff",
    "adversarial",
    "routing_hard",
)
NEGATION = re.compile(
    r"\b(?:not|no|never|without|missing|none|neither|nor|unable|lacks?)\b|n't",
    re.IGNORECASE,
)
CATCH_ALL = re.compile(
    r"^(?:all requirements are met|every requirement is met|none\b|no applicant)",
    re.IGNORECASE,
)
MARGIN = 0.05


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(dict(row)) + "\n")


def fold(group_id: str) -> int:
    return int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % 5


def share(values: Sequence[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# ---------------------------------------------------------------- A2 balance


def balance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fails: list[str] = []
    fam: dict[str, list] = collections.defaultdict(list)
    for row in rows:
        fam[row["family"]].append(row)
    for family, items in sorted(fam.items()):
        cell: dict[str, Any] = {"noul_true_by_kind": {}}
        noul = [r for r in items if r["task_type"] == "noul"]
        cell["noul_true"] = share([r["label"] == 1 for r in noul])
        if noul and not 0.47 <= cell["noul_true"] <= 0.53:
            fails.append(f"{family}: noul true share {cell['noul_true']}")
        by_kind = collections.defaultdict(list)
        for r in noul:
            by_kind[r["audit_metadata"]["kind"]].append(r["label"] == 1)
        for kind, labels in sorted(by_kind.items()):
            value = share(labels)
            cell["noul_true_by_kind"][kind] = [len(labels), value]
            if len(labels) >= 100 and not 0.47 <= value <= 0.53:
                fails.append(f"{family}/{kind}: noul true share {value}")
        positions: dict[int, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        quoted: dict[int, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for r in items:
            if r["task_type"] != "choice":
                continue
            k = len(r["options"])
            positions[k][r["label"]] += 1
            ref = r["audit_metadata"].get("option_refs", {}).get("quoted")
            if ref is not None:
                quoted[k][ref] += 1
        cell["choice_position_max_dev"] = {}
        for label, table in (("gold", positions), ("quoted", quoted)):
            for k, counts in sorted(table.items()):
                total = sum(counts.values())
                dev = max(abs(counts[i] / total - 1 / k) for i in range(k))
                cell["choice_position_max_dev"][f"{label}:{k}"] = [total, round(dev, 4)]
                if total >= 100 and dev > 0.03:
                    fails.append(
                        f"{family}: {label} position deviation {dev:.3f} at {k} options"
                    )
        levels: dict[int, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for r in items:
            if r["task_type"] == "score":
                positive = family != "hs1_unmet_condition" or r["label"] > 0
                if positive:
                    levels[len(r["options"])][r["label"]] += 1
        cell["score_levels"] = {
            str(k): dict(sorted(c.items())) for k, c in levels.items()
        }
        for k, counts in levels.items():
            used = (
                list(range(1, k)) if family == "hs1_unmet_condition" else list(range(k))
            )
            total = sum(counts[i] for i in used)
            if total < 100:
                continue
            for i in used:
                ratio = counts[i] / (total / len(used))
                if not 0.8 <= ratio <= 1.2:
                    fails.append(f"{family}: score level {i}/{k} ratio {ratio:.2f}")
        if family == "hs1_quote_check":
            flags = [r["audit_metadata"]["quote_correct"] for r in items]
            cell["quote_correct_share"] = share(flags)
            if cell["quote_correct_share"] != 0.5:
                fails.append(
                    f"{family}: quote-correct share {cell['quote_correct_share']}"
                )
        out[family] = cell
    return {"cells": out, "fails": fails, "pass": not fails}


# ---------------------------------------------------------------- A3c heuristics


def heuristics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tallies: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        a = r["audit_metadata"]
        fam, t = r["family"], r["task_type"]
        if fam == "hs1_quote_check":
            if a["interface"] == "noul_verify":
                tallies["f1:reject_quote:noul_verify"].append(r["label"] == 0)
            else:
                tallies[f"f1:adopt_quote:{a['interface']}"].append(
                    a["option_refs"]["quoted"] == r["label"]
                )
        elif fam == "hs1_policy_packet":
            refs = (
                a.get("option_refs", {}) if t == "choice" else a.get("heuristics", {})
            )
            for name in ("base_only", "latest_amendment", "first_match"):
                if name in refs:
                    tallies[f"f2:{name}:{t}"].append(refs[name] == r["label"])
        elif fam == "hs1_unmet_condition" and t == "noul":
            has_neg = bool(NEGATION.search(r["state"]))
            tallies["f3:yes_unless_negation:noul"].append(
                (0 if has_neg else 1) == r["label"]
            )
            tallies["f3:no_if_near_threshold:noul"].append(
                (0 if a.get("near_threshold") else 1) == r["label"]
            )
    report, fails = {}, []
    for name, values in sorted(tallies.items()):
        acc = share(values)
        report[name] = [len(values), acc]
        fam, _, t = name.split(":")
        if fam == "f1" and not 0.47 <= acc <= 0.53:
            fails.append(f"{name} {acc}")
        if fam == "f2" and acc > (0.60 if t == "noul" else 0.45):
            fails.append(f"{name} {acc}")
        if fam == "f3" and acc > 0.60:
            fails.append(f"{name} {acc}")
    return {"accuracy": report, "fails": fails, "pass": not fails}


# ---------------------------------------------------------------- A3d length probe


def length_probe(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Group-5-fold best single threshold on state chars or digit count, Noul rows per family."""
    report, fails = {}, []
    fam: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r["task_type"] == "noul":
            fam[r["family"]].append(r)
    for family, items in sorted(fam.items()):
        feats = [
            (len(r["state"]), sum(ch.isdigit() for ch in r["state"])) for r in items
        ]
        labels = [r["label"] for r in items]
        folds = [fold(r["group_id"]) for r in items]
        best_overall = 0.0
        for f_index in (0, 1):
            correct = 0
            for k in range(5):
                train = [
                    (feats[i][f_index], labels[i])
                    for i in range(len(items))
                    if folds[i] != k
                ]
                test = [
                    (feats[i][f_index], labels[i])
                    for i in range(len(items))
                    if folds[i] == k
                ]
                cuts = sorted({v for v, _ in train})
                step = max(1, len(cuts) // 200)
                best = (0, 0, 1)
                for cut in cuts[::step]:
                    for sign in (0, 1):
                        hits = sum(
                            ((v > cut) == bool(sign)) == bool(y) for v, y in train
                        )
                        if hits > best[0]:
                            best = (hits, cut, sign)
                _, cut, sign = best
                correct += sum(((v > cut) == bool(sign)) == bool(y) for v, y in test)
            best_overall = max(best_overall, correct / len(items))
        majority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
        report[family] = {
            "n": len(items),
            "accuracy": round(best_overall, 4),
            "majority": round(majority, 4),
        }
        if best_overall > majority + MARGIN:
            fails.append(f"{family}: {best_overall:.3f} vs majority {majority:.3f}")
    return {"families": report, "fails": fails, "pass": not fails}


# ---------------------------------------------------------------- A7 names


def banned_names(rows: Sequence[dict[str, Any]], package: Path) -> dict[str, Any]:
    hits: collections.Counter = collections.Counter()
    for path in sorted(package.glob("*.py")):
        if path.name == "audit.py":
            continue
        text = path.read_text(encoding="utf-8").lower()
        for name in BANNED:
            if re.search(rf"\b{name}\b", text) or ("_" in name and name in text):
                hits[f"code:{path.name}:{name}"] += 1
    for r in rows:
        a = r["audit_metadata"]
        fields = " ".join(
            str(v)
            for v in (
                r["family"],
                r["render_template"],
                a.get("kind"),
                a.get("subtype"),
            )
        ).lower()
        for name in BANNED:
            if name in fields:
                hits[f"metadata:{name}"] += 1
            if "_" in name and name in r["state"].lower():
                hits[f"text:{name}"] += 1
        for key in ("benchmark", "jev_arena", "jevbench"):
            if key in fields:
                hits[f"metadata:{key}"] += 1
    return {"hits": dict(hits), "pass": not hits}


# ---------------------------------------------------------------- A3a / A3b shortcut cells


def _derive(
    row: dict[str, Any], state: str, instructions: str, family_label: str
) -> dict[str, Any]:
    out = dict(row, state=state, instructions=instructions, family=family_label)
    out["input_sha256"] = digest({field: out[field] for field in INPUT_FIELDS})
    return out


def probe_rows(rows: Sequence[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r["family"] != family:
            continue
        view = r["audit_metadata"].get("probe_view", "")
        if family == "hs1_quote_check":
            derived = _derive(
                r,
                "(evidence removed)",
                mask(view) + "\n" + str(r["instructions"]),
                family,
            )
        elif family == "hs1_policy_packet":
            derived = _derive(
                r,
                "(policy removed)",
                mask(view) + "\n" + str(r["instructions"]),
                family,
            )
        else:
            derived = _derive(
                r,
                "(see question)",
                mask(r["state"]) + "\n" + str(r["instructions"]),
                family,
            )
        out.append(derived)
    return out


def catch_all_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    choice = [r for r in rows if r["task_type"] == "choice"]
    if not choice:
        return None
    hits = sum(
        bool(CATCH_ALL.match(str(r["options"][r["label"]]["description"])))
        for r in choice
    )
    return hits / len(choice)


def run_shortcut(cell: Path, receipt: Path, workers: int) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "v2.data.shortcut",
            "--rows",
            str(cell),
            "--receipt",
            str(receipt),
            "--workers",
            str(workers),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    data = json.loads(receipt.read_text())
    return {"verdict": data["verdict"], "gates": data["gates"]}


def judge(
    gates: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    family: str,
    *,
    significance: bool = False,
    adopt_baseline: bool = False,
) -> tuple[bool, list]:
    """Gate verdicts under amendment 1.

    F3 Choice compares against max(majority, catch-all gold rate) + .05; the F1
    quote-only probe compares Choice / Score against max(majority, .50) + .05
    (adopting the quote is .50 by design). The baseline is never below chance
    (mean 1/K over the cell's rows), because exactly balanced labels push the
    cross-validated majority below chance. With ``significance`` (per-kind
    diagnostic cells) a gate fails only if its Wilson 95% lower bound exceeds
    the threshold.
    """
    catch = catch_all_rate(rows) if family == "hs1_unmet_condition" else None
    chance = {}
    for task in ("choice", "score", "noul"):
        sizes = [len(r["options"]) for r in rows if r["task_type"] == task]
        chance[task] = sum(1 / k for k in sizes) / len(sizes) if sizes else 0.0
    summary = []
    ok = True
    for g in gates:
        baseline = max(g["majority"], chance.get(g["task_type"], 0.0))
        threshold = baseline + MARGIN
        if g["task_type"] == "choice" and catch is not None:
            threshold = max(baseline, catch) + MARGIN
        if adopt_baseline and g["task_type"] in ("choice", "score"):
            threshold = max(baseline, 0.5) + MARGIN
        value = (
            g["wilson95"][0] if significance and g.get("wilson95") else g["accuracy"]
        )
        passed = value <= threshold + 1e-9
        ok &= passed
        summary.append(
            {
                "task_type": g["task_type"],
                "view": g["view"],
                "n": g["n"],
                "accuracy": round(g["accuracy"], 4),
                "threshold": round(threshold, 4),
                "pass": passed,
            }
        )
    return ok, summary


def shortcut_audits(
    rows: Sequence[dict[str, Any]], out_dir: Path, workers: int
) -> dict[str, Any]:
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(mode=0o700, exist_ok=True)
    a3a, a3b = {}, {}
    by_cell: dict[tuple[str, str], list] = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["family"], r["audit_metadata"]["kind"])].append(r)
    for (family, kind), items in sorted(by_cell.items()):
        path = cells_dir / f"{family}.{kind}.jsonl"
        write_jsonl(path, items)
        result = run_shortcut(
            path, cells_dir / f"{family}.{kind}.shortcut.json", workers
        )
        ok, summary = judge(result["gates"], items, family, significance=True)
        a3a[f"{family}/{kind}"] = {"pass": ok, "gates": summary}
    pooled = {}
    for family in sorted({r["family"] for r in rows}):
        items = [r for r in rows if r["family"] == family]
        path = cells_dir / f"{family}.pooled.jsonl"
        write_jsonl(path, items)
        result = run_shortcut(
            path, cells_dir / f"{family}.pooled.shortcut.json", workers
        )
        ok, summary = judge(result["gates"], items, family)
        pooled[family] = {"pass": ok, "gates": summary}
    probes = {
        family: probe_rows(rows, family)
        for family in sorted({r["family"] for r in rows})
    }
    probes["hs1_quote_check.noul_verify"] = [
        r
        for r in probes.get("hs1_quote_check", [])
        if r["audit_metadata"]["interface"] == "noul_verify"
    ]
    for name, derived in probes.items():
        family = name.split(".")[0]
        path = cells_dir / f"{name}.probe.jsonl"
        write_jsonl(path, derived)
        result = run_shortcut(path, cells_dir / f"{name}.probe.shortcut.json", workers)
        # The probe view is carried in the question, so the "state_removed" view is the probe.
        gates = [g for g in result["gates"] if g["view"] == "state_removed"]
        ok, summary = judge(
            gates, derived, family, adopt_baseline=name == "hs1_quote_check"
        )
        verdict = "PASS"
        for g in summary:
            if g["accuracy"] > g["threshold"] - MARGIN + 0.10:
                verdict = "FAIL"
            elif not g["pass"] and verdict == "PASS":
                verdict = "WARN"
        a3b[name] = {"verdict": verdict, "gates": summary}
    return {
        "a3a_pooled": pooled,
        "a3a": a3a,
        "a3a_pass": all(v["pass"] for v in a3a.values())
        and all(v["pass"] for v in pooled.values()),
        "a3b": a3b,
        "a3b_pass": all(v["verdict"] != "FAIL" for v in a3b.values()),
    }


# ---------------------------------------------------------------- A4b structural scan

WORD = re.compile(r"[a-z#$]+")


def grams(text: str, n: int = 8) -> set[int]:
    words = WORD.findall(mask(text))
    return {
        int.from_bytes(
            hashlib.blake2b(
                " ".join(words[i : i + n]).encode(), digest_size=8
            ).digest(),
            "big",
        )
        for i in range(len(words) - n + 1)
    }


def leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in (
                "id",
                "gold",
                "group_id",
                "cluster_id",
                "source",
                "task",
                "language",
            ):
                continue
            yield from leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)


def structural_scan(
    candidates: Sequence[Path], protected: Sequence[Path], threshold: float = 0.5
) -> tuple[dict, dict]:
    index: dict[int, list[int]] = collections.defaultdict(list)
    sizes: list[int] = []
    names: list[str] = []
    for path in protected:
        for line_no, line in enumerate(path.open(encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            item_grams = set()
            for leaf in leaves(row):
                item_grams |= grams(leaf)
            pid = len(sizes)
            sizes.append(len(item_grams))
            names.append(f"{path.name}:{row.get('id', line_no)}")
            for g in item_grams:
                index[g].append(pid)
    flags = []
    scanned = 0
    for path in candidates:
        for row in read_jsonl(path):
            scanned += 1
            for field in ("state", "instructions"):
                cand = grams(str(row[field]))
                if not cand:
                    continue
                hits: collections.Counter = collections.Counter()
                for g in cand:
                    for pid in index.get(g, ()):
                        hits[pid] += 1
                for pid, count in hits.items():
                    fwd = count / len(cand)
                    rev = count / sizes[pid] if sizes[pid] >= 10 else 0.0
                    if fwd >= threshold or rev >= threshold:
                        flags.append(
                            {
                                "id": row["id"],
                                "group_id": row["group_id"],
                                "field": field,
                                "protected": names[pid],
                                "forward": round(fwd, 3),
                                "reverse": round(rev, 3),
                            }
                        )
    private = {"flags": flags, "protected_items": len(sizes), "scanned": scanned}
    public = {
        "schema": "decision2.hs1.struct.v1",
        "rule": f"masked word 8-grams; flag forward or reverse containment >= {threshold} (reverse needs >= 10 protected grams)",
        "protected_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected
        },
        "protected_items": len(sizes),
        "scanned_rows": scanned,
        "flagged_rows": len({f["id"] for f in flags}),
        "flagged_groups": len({f["group_id"] for f in flags}),
        "pass": not flags,
    }
    return private, public


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("local")
    local.add_argument("--train", type=Path, required=True)
    local.add_argument("--dev", type=Path, required=True)
    local.add_argument("--out-dir", type=Path, required=True)
    local.add_argument("--workers", type=int, default=8)
    struct = sub.add_parser("struct")
    struct.add_argument("--candidates", type=Path, action="append", required=True)
    struct.add_argument("--protected", type=Path, action="append", required=True)
    struct.add_argument("--private-receipt", type=Path, required=True)
    struct.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "struct":
        private, public = structural_scan(args.candidates, args.protected)
        for path, payload in (
            (args.private_receipt, private),
            (args.public_receipt, public),
        ):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, indent=1, sort_keys=True)
        print(json.dumps(public, indent=1))
        return 0 if public["pass"] else 1
    train, dev = read_jsonl(args.train), read_jsonl(args.dev)
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = {
        "schema": "decision2.hs1.audit.v1",
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.train, args.dev)
        },
        "a2_balance": {"train": balance(train), "dev": balance(dev)},
        "a3c_heuristics": {"train": heuristics(train), "dev": heuristics(dev)},
        "a3d_length": {"train": length_probe(train), "dev": length_probe(dev)},
        "a7_names": banned_names(train + dev, Path(__file__).resolve().parent),
        "shortcut": shortcut_audits(train, args.out_dir, args.workers),
    }
    fd = os.open(
        args.out_dir / "hs1.audit.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
    verdicts = {
        "a2_train": report["a2_balance"]["train"]["pass"],
        "a3c_train": report["a3c_heuristics"]["train"]["pass"],
        "a3d_train": report["a3d_length"]["train"]["pass"],
        "a7": report["a7_names"]["pass"],
        "a3a": report["shortcut"]["a3a_pass"],
        "a3b": report["shortcut"]["a3b_pass"],
    }
    print(json.dumps(verdicts, indent=1))
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
