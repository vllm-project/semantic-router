"""Freeze a Decision 2.0 v2 data arm: validate, canonicalize, hash, describe.

The canonical form is the rows sorted by id, one `canonical(row)` per line,
UTF-8, each line ending in a newline; `content_sha256` hashes those bytes.
`isolation_check` fails on any id, group_id or input-hash intersection
between partitions of different kinds. A partition named "kind/name" (for
example "train/A6", "aho/A6", "select/SELECT700") has kind "kind"; any other
name is its own kind, so every pair of plain names must be disjoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.textnorm import text_leaves

SCHEMA = "decision2.v2.freeze.v1"
ISOLATION_SCHEMA = "decision2.v2.isolation.v1"
ROLES = {"train": ("train", False), "aho": ("select", False), "replay": ("train", True)}
LICENSE_FIELDS = ("license", "attribution", "evidence", "redistribution")
TOKENIZER_KINDS = ("native_decoder", "raw")
CHOICE_BUCKETS = (
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 255),
)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
ISOLATION_FIELDS = ("id", "group_id", "input_sha256")

TokenizerLoader = Callable[[Mapping[str, Any]], Any]


class IsolationError(ValueError):
    def __init__(self, report: dict[str, Any]) -> None:
        pairs = sorted(
            {f"{item['a']}~{item['b']}:{item['field']}" for item in report["failures"]}
        )
        super().__init__(f"Partition isolation failed: {', '.join(pairs)}")
        self.report = report


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def parse_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    lines = data.split(b"\n")
    if lines and not lines[-1]:
        lines.pop()
    rows = []
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            raise ValueError(f"{label}:{number}: blank line")
        try:
            row = json.loads(
                raw,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except ValueError as exc:
            raise ValueError(f"{label}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{number}: every line must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label}: no rows")
    return rows


def canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: row["id"])
    return "".join(canonical(row) + "\n" for row in ordered).encode("utf-8")


def validate_rows(rows: Sequence[dict[str, Any]], role: str, label: str) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    partition, replay = ROLES[role]
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        try:
            validate_row(row, partition, replay=replay)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}:{number}: {exc}") from exc
        if row["id"] in seen:
            raise ValueError(f"{label}:{number}: duplicate id {row['id']}")
        seen.add(row["id"])


def _payload(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def quantiles(values: Sequence[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        f"p{round(q * 100)}": ordered[int(q * (len(ordered) - 1))] for q in QUANTILES
    }


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counts = collections.Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def _bucket(count: int) -> str:
    for low, high in CHOICE_BUCKETS:
        if low <= count <= high:
            return str(low) if low == high else f"{low}-{high}"
    raise ValueError(f"unsupported option count {count}")


def _nested(pairs: Iterable[tuple[Any, Any]]) -> dict[str, dict[str, int]]:
    nested: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for outer, inner in pairs:
        nested[str(outer)][str(inner)] += 1
    return {
        outer: dict(sorted(counts.items())) for outer, counts in sorted(nested.items())
    }


def describe(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts for the manifest; rows must already be valid."""
    score = [row for row in rows if row["task_type"] == "score"]
    choice = [row for row in rows if row["task_type"] == "choice"]
    noul = [row for row in rows if row["task_type"] == "noul"]
    group_sizes = collections.Counter(row["group_id"] for row in rows)
    noul_labels = collections.Counter(
        row["options"][row["label"]]["key"] for row in noul
    )
    return {
        "rows": len(rows),
        "groups": len(group_sizes),
        "counts": {
            field: _counter(row[field] for row in rows)
            for field in ("task_type", "language", "source", "family")
        },
        "score": {
            "by_levels": _counter(len(row["options"]) for row in score),
            "by_levels_grade": _nested(
                (len(row["options"]), row["options"][row["label"]]["key"])
                for row in score
            ),
        },
        "choice": {
            "by_option_bucket": _counter(
                _bucket(len(row["options"])) for row in choice
            ),
            "gold_position_by_bucket": _nested(
                (_bucket(len(row["options"])), row["label"]) for row in choice
            ),
        },
        "noul": {
            "labels": {key: noul_labels[key] for key in ("false", "true")},
            "true_fraction": (
                round(noul_labels["true"] / len(noul), 6) if noul else None
            ),
        },
        "rows_per_group": _counter(group_sizes.values()),
        "state_chars": quantiles([len(_payload(row["state"])) for row in rows]),
    }


def raw_text(row: Mapping[str, Any]) -> str:
    """State, instructions and option descriptions, newline-joined."""
    parts = [_payload(row["state"]), _payload(row["instructions"])]
    parts += [_payload(option["description"]) for option in row["options"]]
    return "\n".join(parts)


def load_tokenizer(spec: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"tokenizer {spec['name']!r}: token counting needs the transformers "
            "package, which is not installed"
        ) from exc
    return AutoTokenizer.from_pretrained(
        spec["path"],
        revision=spec.get("revision"),
        local_files_only=True,
        trust_remote_code=False,
    )


def _native_encode() -> Callable[..., dict[str, Any]]:
    try:
        from training.model.decision_model import encode
    except ImportError as exc:
        raise RuntimeError(
            "native_decoder token counting imports training.model.decision_model, "
            f"which needs torch and transformers: {exc}"
        ) from exc
    return encode


def _check_tokenizers(specs: Sequence[Mapping[str, Any]]) -> None:
    names = []
    for spec in specs:
        if (
            not isinstance(spec, Mapping)
            or not isinstance(spec.get("name"), str)
            or not isinstance(spec.get("path"), str)
            or spec.get("kind") not in TOKENIZER_KINDS
            or not isinstance(spec.get("revision"), (str, type(None)))
        ):
            raise ValueError(
                "tokenizer specs need string name and path, optional string revision, "
                f"and kind in {TOKENIZER_KINDS}"
            )
        names.append(spec["name"])
    if len(set(names)) != len(names):
        raise ValueError("tokenizer names must be unique")


def token_counts(
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    loader: TokenizerLoader,
) -> dict[str, Any]:
    if spec["kind"] == "native_decoder":
        encode = _native_encode()
        tokenizer = loader(spec)
        counts = [len(encode(row, tokenizer, sys.maxsize)["ids"]) for row in rows]
    else:
        tokenizer = loader(spec)
        counts = [
            len(tokenizer.encode(raw_text(row), add_special_tokens=False))
            for row in rows
        ]
    by_task: collections.Counter[str] = collections.Counter()
    by_language: collections.Counter[str] = collections.Counter()
    for row, count in zip(rows, counts):
        by_task[row["task_type"]] += count
        by_language[row["language"]] += count
    tokenizer_json = Path(spec["path"]) / "tokenizer.json"
    return {
        "name": spec["name"],
        "path": spec["path"],
        "revision": spec.get("revision"),
        "kind": spec["kind"],
        "tokenizer_json_sha256": (
            hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
            if tokenizer_json.is_file()
            else None
        ),
        "total": sum(counts),
        "padded8_total": sum(-(-count // 8) * 8 for count in counts),
        "quantiles": quantiles(counts),
        "by_task_type": dict(sorted(by_task.items())),
        "by_language": dict(sorted(by_language.items())),
    }


def license_table(
    rows: Iterable[Mapping[str, Any]], registry: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    sources = sorted({row["source"] for row in rows})
    missing = [
        source
        for source in sources
        if not isinstance(registry.get(source), Mapping)
        or any(
            registry[source].get(field) in (None, "", []) for field in LICENSE_FIELDS
        )
    ]
    if missing:
        raise ValueError(
            f"license registry lacks complete entries for sources {missing}"
        )
    return {
        source: {field: registry[source][field] for field in LICENSE_FIELDS}
        for source in sources
    }


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    ).encode("utf-8")


def canonical_path(rows_path: Path) -> Path:
    stem = (
        rows_path.name[: -len(".jsonl")]
        if rows_path.name.endswith(".jsonl")
        else rows_path.name
    )
    return rows_path.with_name(f"{stem}.canonical.jsonl")


def freeze(
    rows_path: str | Path,
    *,
    arm_id: str,
    role: str,
    license_registry: Mapping[str, Any] | str | Path,
    tokenizers: Sequence[Mapping[str, Any]],
    out_manifest: str | Path,
    tokenizer_loader: TokenizerLoader | None = None,
) -> dict[str, Any]:
    rows_path, out_manifest = Path(rows_path), Path(out_manifest)
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError("arm_id must be a nonempty string")
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if out_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {out_manifest}")
    _check_tokenizers(tokenizers)
    registry_sha256 = None
    if not isinstance(license_registry, Mapping):
        registry_bytes = Path(license_registry).read_bytes()
        registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
        license_registry = json.loads(registry_bytes)
        if not isinstance(license_registry, Mapping):
            raise ValueError("license registry must be a JSON object keyed by source")
    data = rows_path.read_bytes()
    rows = parse_jsonl(data, str(rows_path))
    validate_rows(rows, role, str(rows_path))
    licenses = license_table(rows, license_registry)
    content = canonical_jsonl(rows)
    content_sha256 = hashlib.sha256(content).hexdigest()
    ordered = parse_jsonl(content, "canonical")
    tokens = [
        token_counts(ordered, spec, tokenizer_loader or load_tokenizer)
        for spec in tokenizers
    ]
    written = None
    if data != content:
        written = canonical_path(rows_path)
        if written.exists():
            if written.read_bytes() != content:
                raise FileExistsError(f"refusing to overwrite {written}")
        else:
            _write_new(written, content)
    manifest = {
        "schema": SCHEMA,
        "arm_id": arm_id,
        "role": role,
        "partition": ROLES[role][0],
        "input": {
            "path": str(rows_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "is_canonical": written is None,
        },
        "canonical_path": str(written or rows_path),
        "content_sha256": content_sha256,
        "canonical_form": "rows sorted by id; canonical(row) + '\\n' per row; UTF-8",
        **describe(ordered),
        "tokens": tokens,
        "licenses": licenses,
        "license_registry_sha256": registry_sha256,
    }
    _write_new(out_manifest, _json_bytes(manifest))
    return manifest


def _kind(name: str) -> str:
    return name.split("/", 1)[0]


def _partition_rows(
    value: str | Path | Iterable[Mapping[str, Any]], name: str
) -> tuple[list[Mapping[str, Any]], str | None]:
    if isinstance(value, (str, Path)):
        data = Path(value).read_bytes()
        return parse_jsonl(data, name), hashlib.sha256(data).hexdigest()
    return list(value), None


def isolation_check(
    named_partitions: Mapping[str, str | Path | Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report pairwise id/group_id/input_sha256 intersections; raise
    IsolationError when partitions of different kinds intersect."""
    if len(named_partitions) < 2:
        raise ValueError("isolation_check needs at least two partitions")
    values: dict[str, dict[str, set[str]]] = {}
    partitions = {}
    for name in sorted(named_partitions):
        rows, sha256 = _partition_rows(named_partitions[name], name)
        fields: dict[str, set[str]] = {field: set() for field in ISOLATION_FIELDS}
        for number, row in enumerate(rows, 1):
            missing = [
                field for field in ("id", "group_id", *INPUT_FIELDS) if field not in row
            ]
            if missing:
                raise ValueError(f"{name}:{number}: missing {missing}")
            input_hash = digest({field: row[field] for field in INPUT_FIELDS})
            if row.get("input_sha256", input_hash) != input_hash:
                raise ValueError(
                    f"{name}:{number}: input_sha256 disagrees with payload"
                )
            fields["id"].add(row["id"])
            fields["group_id"].add(row["group_id"])
            fields["input_sha256"].add(input_hash)
        values[name] = fields
        partitions[name] = {
            "kind": _kind(name),
            "rows": len(rows),
            "groups": len(fields["group_id"]),
            "sha256": sha256,
        }
    intersections, failures = [], []
    names = sorted(values)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for field in ISOLATION_FIELDS:
                shared = sorted(values[first][field] & values[second][field])
                if not shared:
                    continue
                item = {
                    "a": first,
                    "b": second,
                    "field": field,
                    "count": len(shared),
                    "examples": shared[:10],
                    "allowed": _kind(first) == _kind(second),
                }
                intersections.append(item)
                if not item["allowed"]:
                    failures.append(item)
    report = {
        "schema": ISOLATION_SCHEMA,
        "partitions": partitions,
        "intersections": intersections,
        "failures": failures,
        "verdict": "FAIL" if failures else "PASS",
    }
    if failures:
        raise IsolationError(report)
    return report


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser(
        "freeze", help="validate, canonicalize and describe an arm"
    )
    frozen.add_argument("--rows", required=True, type=Path)
    frozen.add_argument("--arm-id", required=True)
    frozen.add_argument("--role", required=True, choices=sorted(ROLES))
    frozen.add_argument("--license-registry", required=True, type=Path)
    frozen.add_argument(
        "--tokenizers",
        type=Path,
        help="JSON list of {name, path, revision, kind: native_decoder|raw}",
    )
    frozen.add_argument("--out-manifest", required=True, type=Path)
    isolation = commands.add_parser("isolation", help="check partition isolation")
    isolation.add_argument(
        "--partition", action="append", required=True, type=_named_path
    )
    isolation.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if args.out_manifest.exists():
            parser.error(f"refusing to overwrite {args.out_manifest}")
        specs = (
            json.loads(args.tokenizers.read_text(encoding="utf-8"))
            if args.tokenizers
            else []
        )
        manifest = freeze(
            args.rows,
            arm_id=args.arm_id,
            role=args.role,
            license_registry=args.license_registry,
            tokenizers=specs,
            out_manifest=args.out_manifest,
        )
        print(
            json.dumps(
                {"content_sha256": manifest["content_sha256"], "rows": manifest["rows"]}
            )
        )
        return 0
    if args.report is not None and args.report.exists():
        parser.error(f"refusing to overwrite {args.report}")
    named = dict(args.partition)
    if len(named) != len(args.partition):
        parser.error("partition names must be unique")
    try:
        report = isolation_check(named)
        status = 0
    except IsolationError as exc:
        report, status = exc.report, 1
    if args.report is not None:
        _write_new(args.report, _json_bytes(report))
    print(
        json.dumps({"verdict": report["verdict"], "failures": len(report["failures"])})
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
