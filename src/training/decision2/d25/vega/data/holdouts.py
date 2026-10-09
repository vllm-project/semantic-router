"""Apply ws-proxy's holdouts.json (sources and row slices reserved for the private-part proxies).

Rows identify their upstream dataset through ``meta.hf_ids`` (list) or a lookup table supplied by the
caller (tasksource sources.yaml, Decision 2.0 origin map), ``meta.source_split`` and, for row-sliced
datasets, ``meta.slice_keys`` ({slice key name: exact upstream field value}). A row that belongs to a
row-sliced dataset but carries no slice key is dropped (conservative superset of the slice).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from d25.vega.data.util import sha256_file


def selected(name: str, key: str, mod: int, keep: int) -> bool:
    return (
        int(
            hashlib.sha256(f"d25-proxy-v1:{name}:{key}".encode("utf-8")).hexdigest()[
                :12
            ],
            16,
        )
        % mod
        < keep
    )


def norm_id(value: str) -> str:
    value = re.sub(r"\(.*?\)", "", value).strip().lower()
    value = value.split("@")[0].strip()
    return re.sub(r"\s+", " ", value)


class Holdouts:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.spec = json.loads(self.path.read_text())
        self.sha256 = sha256_file(self.path)
        self.version = self.spec.get("version")
        self.ts_sources = list(self.spec.get("tasksource_sources") or [])
        self.entries = []
        for entry in self.spec.get("datasets", []):
            ids = {norm_id(entry["hf_id"])} | {
                norm_id(a) for a in entry.get("aliases", [])
            }
            self.entries.append((entry, ids))
        self.counts: Counter = Counter()

    def match_entry(self, hf_ids: list[str]) -> list[dict[str, Any]]:
        wanted = {norm_id(h) for h in hf_ids if h}
        return [entry for entry, ids in self.entries if wanted & ids]

    def check(self, row: dict[str, Any], hf_ids: list[str] | None = None) -> str | None:
        """Return a drop reason or None."""
        source = row.get("source", "")
        if source.startswith("tasksource:"):
            name = source.split(":", 1)[1]
            for blocked in self.ts_sources:
                if name == blocked or name.startswith(blocked + "/"):
                    return f"tasksource_source:{blocked}"
        meta = row.get("meta") or {}
        ids = list(hf_ids or []) + list(meta.get("hf_ids") or [])
        if not ids:
            return None
        split = str(meta.get("source_split") or "")
        for entry in self.match_entry(ids):
            scope = entry.get("scope")
            label = f"{entry['use']}:{entry['hf_id']}"
            if scope == "all":
                return label
            splits = [s for s in entry.get("splits", [])]
            in_split = (
                "all" in splits
                or split in splits
                or (
                    split in ("validation", "dev", "val")
                    and any(s in ("validation", "dev", "val") for s in splits)
                )
            )
            if scope == "splits" and in_split:
                return label
            if scope == "rows" and in_split:
                sl = entry["slice"]
                key = (meta.get("slice_keys") or {}).get(sl["name"])
                if key is None:
                    return label + ":no_slice_key"
                if selected(sl["name"], str(key), int(sl["mod"]), int(sl["keep"])):
                    return label + ":slice"
        return None

    def report(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "version": self.version,
            "sha256": self.sha256,
            "dropped": dict(self.counts.most_common()),
        }
