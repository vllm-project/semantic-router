"""Build HS1 TRAIN, the dev slices and the ``hs1-dev`` panel files (prereg §2-§3).

    python3 -m v2.data.hs1.build --out-dir DIR [--scale 1.0]

Writes (O_EXCL, mode 0600): ``hs1.train.jsonl``, ``hs1.dev.jsonl`` (training
contract, split=select), ``hs1-dev.prompts.jsonl`` (gold-free panel prompts),
``hs1-dev.gold.jsonl`` and ``hs1.build.json`` (sizes, code hashes, content
hashes, statistics). An oracle / re-check disagreement stops the build.
"""

from __future__ import annotations

import argparse
import collections
import inspect
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical, check_partition_isolation
from training.model.infer import question_to_row
from v2.data.hs1 import core, f1_quote, f2_policy, f3_unmet

MODULES = {m.FAMILY: m for m in (f1_quote, f2_policy, f3_unmet)}
# family -> (TRAIN groups, dev-id groups, dev-ood groups); two rows per group.
SIZES = {
    "hs1_quote_check": (3600, 320, 160),
    "hs1_policy_packet": (1200, 160, 80),
    "hs1_unmet_condition": (5200, 320, 160),
}
LONG_CHARS = 3000
PANEL = "hs1-dev"


def plan(family: str, slice_name: str, groups: int) -> list[tuple[str, str, str, str]]:
    """(seed_key, kind, interface, length) for every group of one family slice."""
    module = MODULES[family]
    kinds = module.KINDS_OOD if slice_name == "dev-ood" else module.KINDS_TRAIN
    namespace = "hs1-train" if slice_name == "train" else "hs1-dev"
    per_kind = core.schedule(
        {kind: 1.0 for kind in kinds}, groups, f"{family}:{slice_name}:kinds"
    )
    counts = collections.Counter(per_kind)
    slots: list[tuple[str, str, str]] = []
    for kind in kinds:
        supported = module.supported(kind)
        shares = {i: s for i, s in module.INTERFACE_SHARES.items() if i in supported}
        interfaces = core.schedule(
            shares, counts[kind], f"{family}:{slice_name}:{kind}:if"
        )
        lengths = core.schedule(
            module.LENGTH_SHARES, counts[kind], f"{family}:{slice_name}:{kind}:len"
        )
        slots.extend((kind, i, length) for i, length in zip(interfaces, lengths))
    core.rng_for("hs1-plan", family, slice_name).shuffle(slots)
    offset = 0 if slice_name != "dev-ood" else 1_000_000
    return [
        (f"{namespace}:{family}:{offset + index}", kind, interface, length)
        for index, (kind, interface, length) in enumerate(slots)
    ]


def generate(family: str, slice_name: str, groups: int) -> list[dict[str, Any]]:
    module = MODULES[family]
    split = "train" if slice_name == "train" else "select"
    rows: list[dict[str, Any]] = []
    targets: collections.Counter = collections.Counter()
    counters: collections.Counter = collections.Counter()
    takes_index = "index" in inspect.signature(module.make_group).parameters
    for seed_key, kind, interface, length in plan(family, slice_name, groups):
        if takes_index:
            items = module.make_group(
                seed_key, kind, interface, length, index=counters[(kind, interface)]
            )
            counters[(kind, interface)] += 1
        else:
            items = module.make_group(seed_key, kind, interface, length)
        first_choice = next((it for it in items if it.task_type == "choice"), None)
        count = len(first_choice.choices) if first_choice else 1
        target = targets[count] % count
        targets[count] += 1
        rows.extend(
            core.build_rows(
                family,
                seed_key,
                items,
                split=split,
                slice_name=slice_name,
                gold_target=target,
            )
        )
    return rows


def panel_prompt(row: dict[str, Any]) -> dict[str, Any]:
    kind = row["task_type"]
    if kind == "score":
        criteria: Any = [o["description"] for o in row["options"]]
    else:
        criteria = {o["key"]: o["description"] for o in row["options"]}
    return {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            "decision": {
                "type": kind,
                "instructions": row["instructions"],
                "criteria": criteria,
            }
        },
    }


def panel_gold(row: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    kind = row["task_type"]
    key = row["options"][row["label"]]["key"]
    value: Any = (
        key if kind == "choice" else (key == "true" if kind == "noul" else int(key))
    )
    gold = {"type": kind, "value": value, "semantic_value": value}
    if kind == "choice":
        gold["label_to_semantic"] = {o["key"]: o["key"] for o in row["options"]}
    audit = row["audit_metadata"]
    return {
        "id": row["id"],
        "task": row["family"],
        "source": audit["kind"],
        "slice": audit["slice"],
        "interface": audit["interface"],
        "subtype": audit["subtype"],
        "group_id": row["group_id"],
        "cluster_id": row["group_id"],
        "language": row["language"],
        "long": len(row["state"]) >= LONG_CHARS,
        "input_chars": len(row["state"]),
        "questions": prompt["questions"],
        "gold": {"decision": gold},
        "audit": {
            k: audit[k]
            for k in ("option_refs", "quote_correct", "twin", "failing")
            if k in audit
        },
    }


def roundtrip(row: dict[str, Any], prompt: dict[str, Any]) -> None:
    back = question_to_row(prompt, "decision", prompt["questions"]["decision"])
    for field in ("state", "instructions", "options", "task_type"):
        if back[field] != row[field]:
            raise ValueError(f"{row['id']}: panel round-trip differs in {field}")


def quantiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "min": ordered[0],
        "q10": pick(0.1),
        "q50": pick(0.5),
        "q90": pick(0.9),
        "max": ordered[-1],
    }


def stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_family: dict[str, list] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family, items in sorted(by_family.items()):
        audit = [r["audit_metadata"] for r in items]
        noul = [r for r in items if r["task_type"] == "noul"]
        choice = [r for r in items if r["task_type"] == "choice"]
        score = [r for r in items if r["task_type"] == "score"]
        out[family] = {
            "rows": len(items),
            "groups": len({r["group_id"] for r in items}),
            "task_type": dict(collections.Counter(r["task_type"] for r in items)),
            "interface": dict(collections.Counter(a["interface"] for a in audit)),
            "kind": dict(sorted(collections.Counter(a["kind"] for a in audit).items())),
            "length_class": dict(collections.Counter(a["length"] for a in audit)),
            "slice": dict(collections.Counter(a["slice"] for a in audit)),
            "subtype": dict(
                sorted(collections.Counter(a["subtype"] for a in audit).items())
            ),
            "noul_true_share": (
                round(sum(r["label"] for r in noul) / len(noul), 4) if noul else None
            ),
            "choice_gold_position": {
                str(k): dict(
                    sorted(
                        collections.Counter(
                            r["label"] for r in choice if len(r["options"]) == k
                        ).items()
                    )
                )
                for k in sorted({len(r["options"]) for r in choice})
            },
            "score_levels": {
                str(k): dict(
                    sorted(
                        collections.Counter(
                            r["label"] for r in score if len(r["options"]) == k
                        ).items()
                    )
                )
                for k in sorted({len(r["options"]) for r in score})
            },
            "state_chars": quantiles([len(r["state"]) for r in items]),
            "rows_state_ge_3000": sum(len(r["state"]) >= LONG_CHARS for r in items),
            "approx_tokens_chars_div_4": sum(
                len(r["state"])
                + len(str(r["instructions"]))
                + sum(len(str(o["description"])) for o in r["options"])
                for r in items
            )
            // 4,
        }
    return out


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.py"))
    }


def write_new(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        canonical(r) + "\n" for r in sorted(rows, key=lambda r: r["id"])
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="smoke builds only; the release build uses 1.0",
    )
    args = parser.parse_args(argv)
    started = time.time()
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for family, (n_train, n_id, n_ood) in SIZES.items():
        scale = lambda n: max(1, round(n * args.scale))  # noqa: E731
        train += generate(family, "train", scale(n_train))
        dev += generate(family, "dev-id", scale(n_id))
        dev += generate(family, "dev-ood", scale(n_ood))
        print(
            f"{family}: train {len(train)} dev {len(dev)} ({time.time() - started:.0f}s)",
            file=sys.stderr,
        )
    check_partition_isolation({"train": train, "select": dev})
    for rows in (train, dev):
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (row["state"], canonical(row["instructions"]))
            if key in seen:
                raise ValueError(f"duplicate state+instructions: {row['id']}")
            seen.add(key)
    prompts, gold = [], []
    for row in dev:
        prompt = panel_prompt(row)
        roundtrip(row, prompt)
        prompts.append(prompt)
        gold.append(panel_gold(row, prompt))
    args.out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    hashes = {
        "hs1.train.jsonl": write_new(args.out_dir / "hs1.train.jsonl", jsonl(train)),
        "hs1.dev.jsonl": write_new(args.out_dir / "hs1.dev.jsonl", jsonl(dev)),
        f"{PANEL}.prompts.jsonl": write_new(
            args.out_dir / f"{PANEL}.prompts.jsonl", jsonl(prompts)
        ),
        f"{PANEL}.gold.jsonl": write_new(
            args.out_dir / f"{PANEL}.gold.jsonl", jsonl(gold)
        ),
    }
    build = {
        "schema": "decision2.hs1.build.v1",
        "generator": core.GENERATOR,
        "version": core.VERSION,
        "scale": args.scale,
        "sizes": SIZES,
        "code_sha256": code_hashes(),
        "files_sha256": hashes,
        "train": stats(train),
        "dev": stats(dev),
        "seconds": round(time.time() - started, 1),
    }
    write_new(
        args.out_dir / "hs1.build.json",
        (json.dumps(build, indent=1, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(hashes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
