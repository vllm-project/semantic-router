"""Evaluation-only pools: the registry and the guard every mixture builder calls.

``eval_only_pools.json`` (next to this module) lists pools that hold evaluation items or their
outputs, which never enter a training mixture: the AutoJev-27B runtime-qualification sets (their
spot-check set S holds 128 typed FINAL, 231 public-231 and 128 CSS15 prompts; the outputs were
never targets) and the installed evaluation panels. A pool matches a file by path pattern or,
for files up to ``HASH_LIMIT_BYTES``, by SHA-256 (so a renamed copy is caught), and a row by the
SHA-256 of its id (a ``#`` resample suffix is ignored).

    from v2.common import eval_only
    eval_only.guard(args, spec)      # every path-like string in parsed arguments / specs
    eval_only.check_rows(rows)       # the rows a builder is about to write

Both raise ``EvalOnlyInputError``. The relabel command rewrites a training-corpora manifest
(``c1-corpora/1`` or ``htdev-training-manifest/1``) with the matching files moved from each
training label into a sibling ``<label>:evaluation-only`` label of kind ``evaluation-only``:

    python3 -m v2.common.eval_only relabel --manifest IN.json --out OUT.json --receipt R.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).with_name("eval_only_pools.json")
HASH_LIMIT_BYTES = 8 << 20
KIND = "evaluation-only"
_SPLIT = re.compile(r"[\s,=]+")


class EvalOnlyInputError(ValueError):
    """A mixture input or row belongs to an evaluation-only pool."""


@dataclasses.dataclass(frozen=True)
class Pool:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    file_sha256: frozenset[str]
    item_id_sha256: frozenset[str]


@functools.lru_cache(maxsize=None)
def load(path: str = str(REGISTRY)) -> tuple[Pool, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "dev2-eval-only-pools/1":
        raise ValueError(f"{path}: unexpected schema {data.get('schema')!r}")
    return tuple(
        Pool(
            name=pool["name"],
            patterns=tuple(re.compile(p) for p in pool["path_patterns"]),
            file_sha256=frozenset(pool["file_sha256"]),
            item_id_sha256=frozenset(pool["item_id_sha256"]),
        )
        for pool in data["pools"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_path(text: str, pools: Iterable[Pool] | None = None) -> str | None:
    """Name of the pool whose path pattern matches ``text``, else None."""
    for pool in load() if pools is None else pools:
        if any(p.search(text) for p in pool.patterns):
            return pool.name
    return None


def match_file(path: str | Path, pools: Iterable[Pool] | None = None) -> str | None:
    """Path pattern, or SHA-256 for an existing file up to ``HASH_LIMIT_BYTES``."""
    pools = load() if pools is None else tuple(pools)
    name = match_path(str(path), pools)
    if name is not None:
        return name
    target = Path(path)
    try:
        if not target.is_file() or target.stat().st_size > HASH_LIMIT_BYTES:
            return None
    except OSError:
        return None
    digest = _sha256(target)
    for pool in pools:
        if digest in pool.file_sha256:
            return pool.name
    return None


def check_file(
    path: str | Path, sha256: str | None = None, pools: Iterable[Pool] | None = None
) -> None:
    """Raise if ``path`` belongs to an evaluation-only pool (pattern, or ``sha256`` when the
    caller has already hashed the file)."""
    pools = load() if pools is None else tuple(pools)
    name = match_path(str(path), pools)
    if name is None and sha256 is not None:
        name = next((p.name for p in pools if sha256 in p.file_sha256), None)
    elif name is None:
        name = match_file(path, pools)
    if name is not None:
        raise EvalOnlyInputError(
            f"{path}: evaluation-only pool {name!r} cannot enter a training mixture"
        )


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        yield text
        for token in _SPLIT.split(text):
            if token and token != text:
                yield token


def guard(*objects: Any, pools: Iterable[Pool] | None = None) -> None:
    """Raise if any path-like string in ``objects`` (argparse namespaces, specs, paths, lists)
    names a file of an evaluation-only pool."""
    pools = load() if pools is None else tuple(pools)
    for text in _strings(objects):
        if "/" not in text and not Path(text).is_file():
            continue
        name = match_file(text, pools)
        if name is not None:
            raise EvalOnlyInputError(
                f"{text}: evaluation-only pool {name!r} cannot enter a training mixture"
            )


def check_rows(
    rows: Iterable[Mapping[str, Any]], pools: Iterable[Pool] | None = None
) -> int:
    """Raise if a row id belongs to an evaluation-only pool; return the rows checked."""
    pools = load() if pools is None else tuple(pools)
    ids = frozenset().union(*(pool.item_id_sha256 for pool in pools))
    count = 0
    for row in rows:
        count += 1
        ident = str(row.get("id", ""))
        for candidate in {ident, ident.split("#", 1)[0]}:
            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() in ids:
                raise EvalOnlyInputError(
                    f"row {ident}: an evaluation-only item cannot enter a training mixture"
                )
    return count


def relabel(
    manifest: Mapping[str, Any], pools: Iterable[Pool] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move matching files of each label into ``<label>:evaluation-only`` (kind evaluation-only)."""
    pools = load() if pools is None else tuple(pools)
    labels: dict[str, Any] = {}
    moved: dict[str, list[dict[str, str]]] = {}
    for label, entry in manifest["labels"].items():
        if entry.get("kind") == KIND or label.endswith(":" + KIND):
            labels[label] = entry
            continue
        keep, move = [], []
        for item in entry["files"]:
            name = match_path(item["path"], pools) or next(
                (p.name for p in pools if item.get("sha256") in p.file_sha256), None
            )
            (move if name else keep).append((item, name))
        labels[label] = {**entry, "files": [item for item, _ in keep]}
        if "file_count" in entry:
            labels[label].update(_totals([item for item, _ in keep]))
        if move:
            files = [item for item, _ in move]
            labels[f"{label}:{KIND}"] = {
                **{k: v for k, v in entry.items() if k not in ("files", "kind")},
                "kind": KIND,
                "pools": sorted({name for _, name in move}),
                "files": files,
                **(_totals(files) if "file_count" in entry else {}),
            }
            moved[label] = [
                {"path": item["path"], "sha256": item.get("sha256", ""), "pool": name}
                for item, name in move
            ]
    out = {**manifest, "labels": labels}
    if "totals" in manifest:
        out["totals"] = {
            **manifest["totals"],
            "labels": len(labels),
            "evaluation_only_files": sum(len(v) for v in moved.values()),
        }
    receipt = {
        "schema": "dev2-eval-only-relabel/1",
        "labels_before": len(manifest["labels"]),
        "labels_after": len(labels),
        "moved_by_label": {k: len(v) for k, v in sorted(moved.items())},
        "moved": moved,
    }
    return out, receipt


def _totals(files: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "file_count": len(files),
        "bytes": sum(int(f.get("bytes", 0)) for f in files),
        "rows": sum(max(int(f.get("rows", 0)), 0) for f in files),
    }


def _write_new(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=1) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("relabel")
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    out, receipt = relabel(source)
    receipt["manifest_sha256"] = _sha256(args.manifest)
    receipt["registry_sha256"] = _sha256(REGISTRY)
    receipt["out_sha256"] = _write_new(args.out, out)
    _write_new(args.receipt, receipt)
    print(json.dumps({k: receipt[k] for k in ("moved_by_label", "out_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
