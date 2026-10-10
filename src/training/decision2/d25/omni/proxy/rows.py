"""Proxy evaluation rows in the vision-suite layout.

A proxy set directory holds ``rows.jsonl.gz`` (sorted by id), ``images/<sha[:2]>/<sha>.<ext>``
(content-addressed stored bytes), ``manifest.json``, ``BUILD.json`` and ``SHA256SUMS``. Rows follow
``d25.omni.common.vision_format``: one lettered ``choice`` question per row, ``expected`` holds the
answer key, and ``metadata`` records ``benchmark``, ``subtask``, ``chance``, ``n_options``,
``n_images`` and per-image provenance.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import platform
import random
import string
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from d25.omni.common import vision_format as vf

LETTERS = string.ascii_uppercase
PROXY_SPLIT = "proxy-v1"


def item_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from any parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def rng(*parts: Any) -> random.Random:
    return random.Random(item_seed(*parts))


def allocate(
    sizes: Mapping[str, int], weights: Mapping[str, float], n: int, seed: int
) -> list[list[tuple[str, int]]]:
    """Per item, a disjoint list of ``(kind, pool index)`` candidates so no source item is used twice.

    Kinds are assigned to items by largest-remainder rounding of ``weights`` (kinds with an empty
    pool get none); each item of a kind owns a contiguous block of a shuffled pool and tries its
    candidates in order.
    """
    kinds = [k for k in weights if sizes.get(k, 0) > 0]
    total = sum(weights[k] for k in kinds)
    quotas = {k: n * weights[k] / total for k in kinds}
    counts = {k: int(quotas[k]) for k in kinds}
    for k in sorted(kinds, key=lambda k: quotas[k] - counts[k], reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[k] += 1
    r = random.Random(item_seed("allocate", seed, n))
    slots = [k for k in kinds for _ in range(counts[k])]
    r.shuffle(slots)
    orders = {k: r.sample(range(sizes[k]), sizes[k]) for k in kinds}
    taken = {k: 0 for k in kinds}
    plan = []
    for k in slots:
        block = max(1, sizes[k] // max(1, counts[k]))
        start = taken[k]
        taken[k] += block
        plan.append([(k, orders[k][j % sizes[k]]) for j in range(start, start + block)])
    return plan


def lettered(
    correct: str, distractors: Sequence[str], r: random.Random, keep_order: bool = False
) -> tuple[dict[str, str], str]:
    """Options ``{A: text, ...}`` with the correct text at a random (or first) position."""
    texts = [correct, *distractors]
    if len(set(texts)) != len(texts):
        raise ValueError(f"options must be distinct: {texts}")
    if not keep_order:
        r.shuffle(texts)
    criteria = {LETTERS[i]: t for i, t in enumerate(texts)}
    answer = LETTERS[texts.index(correct)]
    return criteria, answer


def encode_image(image, fmt: str = "PNG", quality: int = 95) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        image.convert("RGB").save(
            buffer, format="JPEG", quality=quality, subsampling=0, optimize=False
        )
        return buffer.getvalue(), "jpg"
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue(), "png"


class ProxyWriter:
    """Collects rows and images of one proxy set and writes the suite layout on ``close``."""

    def __init__(
        self, root: str | Path, name: str, benchmark: str, version: str, seed: int
    ):
        self.root = Path(root)
        self.name = name
        self.benchmark = benchmark
        self.version = version
        self.seed = seed
        self.rows: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self.started = dt.datetime.now(dt.timezone.utc)
        (self.root / "images").mkdir(parents=True, exist_ok=True)

    def source(self, key: str, **info: Any) -> None:
        """Register a source (licence, repo or URL, revision, file hashes) for the manifest."""
        self.sources[key] = dict(info)

    def image(self, payload: bytes, ext: str) -> tuple[str, str]:
        relative = vf.store_image(payload, self.root, ext)
        return relative, vf.sha256_bytes(payload)

    def add(
        self,
        item_id: str,
        subtask: str,
        images: Sequence[tuple[str, str]],
        instructions: str,
        criteria: Mapping[str, str],
        answer: str,
        state: Any = "",
        provenance: Sequence[Mapping[str, Any]] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if answer not in criteria:
            raise ValueError(f"{item_id}: answer {answer!r} is not an option")
        if not 0 < len(images) <= vf.MAX_IMAGES:
            raise ValueError(
                f"{item_id}: a proxy row takes 1 to {vf.MAX_IMAGES} images"
            )
        n = len(criteria)
        row = {
            "id": f"{self.name}:{item_id}",
            "family": self.benchmark,
            "split": PROXY_SPLIT,
            "images": [path for path, _ in images],
            "state": state,
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(criteria),
                }
            },
            "expected": {"q1": answer},
            "metadata": {
                "benchmark": self.benchmark,
                "proxy": self.name,
                "proxy_version": self.version,
                "subtask": subtask,
                "chance": 1.0 / n,
                "n_options": n,
                "n_images": len(images),
                "image_sha256": [digest for _, digest in images],
                "provenance": list(provenance),
                **(extra or {}),
            },
        }
        self.rows.append(row)

    def close(
        self, command: Sequence[str] | None = None, code_tag: str | None = None
    ) -> dict[str, Any]:
        ids = Counter(r["id"] for r in self.rows)
        dupes = [k for k, v in ids.items() if v > 1]
        if dupes:
            raise ValueError(f"duplicate row ids: {dupes[:5]}")
        rows = sorted(self.rows, key=lambda r: r["id"])
        payload = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        ).encode()
        rows_path = self.root / "rows.jsonl.gz"
        with open(rows_path, "wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as gz:
                gz.write(payload)
        answers = Counter(r["expected"]["q1"] for r in rows)
        subtasks = Counter(r["metadata"]["subtask"] for r in rows)
        images = sorted({p for r in rows for p in r["images"]})
        manifest = {
            "proxy": self.name,
            "benchmark": self.benchmark,
            "version": self.version,
            "split": PROXY_SPLIT,
            "seed": self.seed,
            "rows": len(rows),
            "images": len(images),
            "image_bytes": sum((self.root / p).stat().st_size for p in images),
            "subtasks": dict(sorted(subtasks.items())),
            "answer_keys": dict(sorted(answers.items())),
            "mean_chance": sum(r["metadata"]["chance"] for r in rows)
            / max(1, len(rows)),
            "rows_sha256": hashlib.sha256(payload).hexdigest(),
            "rows_gz_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "sources": self.sources,
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
        build = {
            "command": list(command or sys.argv),
            "code_tag": code_tag or os.environ.get("D25_CODE_TAG"),
            "python": platform.python_version(),
            "started_utc": self.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        (self.root / "BUILD.json").write_text(json.dumps(build, indent=1) + "\n")
        sums = []
        for path in ["rows.jsonl.gz", "manifest.json", *images]:
            digest = hashlib.sha256((self.root / path).read_bytes()).hexdigest()
            sums.append(f"{digest}  {path}")
        (self.root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        return manifest


def read_rows(root: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(root) / "rows.jsonl.gz", "rt") as handle:
        return [json.loads(line) for line in handle]


def verify(root: str | Path) -> list[str]:
    """Files whose sha256 differs from SHA256SUMS (empty when the set is intact)."""
    root = Path(root)
    bad = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        target = root / path
        if (
            not target.exists()
            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
        ):
            bad.append(path)
    return bad
