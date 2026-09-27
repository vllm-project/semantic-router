"""Build one frozen, development-only Kai2 Choice/Noul recovery micro-arm.

The arm reuses verified native rows without rewriting instructions or labels.
TweetEval rows add human-labeled Choice contexts; the clean-v2 replay preserves
Noul and Score. Selection is deterministic at whole component-group level.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

INPUT_SHA256 = {
    "clean_train": "fc8496c76c84b37f0a9308b13a270b69aab608defb8a9e4a3f62011352b6976c",
    "clean_manifest": "a2591978481e019f482686df69d55f213a970f51ce465c6a45e6910bc9d3dca5",
    "human_train": "af03fc360392f6b521cf8659874cab98e3dcb6b76592e586f6d2a24b724f7300",
    "human_manifest": "563921757af8d8540cbc5dc6788a09087f22c893c48c95e070a3beab8e36b259",
}
SEED = "20260927-kai06-human-recovery1024-v1"
HUMAN_BUCKETS = ("hate", "offensive", "irony", "emotion", "sentiment")
HUMAN_PER_BUCKET = 64
REPLAY_QUOTAS = {"Choice": 256, "Noul": 320, "Score": 128}


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _read(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    if sha_file(path) != expected_sha:
        raise ValueError(f"Frozen native source changed: {path.name}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Native source has repeated row IDs")
    if any(
        set(row)
        != {
            "id",
            "component_id",
            "source_id",
            "hard_target_id",
            "state_text",
            "question",
            "target",
        }
        for row in rows
    ):
        raise ValueError("Unexpected native source schema")
    return rows


def _key(value: str) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode()).hexdigest()


def select_whole_groups(
    rows: list[dict[str, Any]], count: int, *, source: str, kind: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["component_id"]].append(row)
    candidates = []
    for group_id, group in groups.items():
        if all(
            row["source_id"] == source and row["question"]["type"] == kind
            for row in group
        ):
            candidates.append((_key(group_id), group_id, group))
    selected = []
    for _, _, group in sorted(candidates):
        if len(selected) + len(group) <= count:
            selected.extend(group)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Cannot select {count} whole groups for {source}/{kind}")
    return selected


def select_replay_groups(
    clean: list[dict[str, Any]],
    human: list[dict[str, Any]],
    *,
    kind: str,
    quota: int,
) -> list[dict[str, Any]]:
    """Select complete clean groups absent from the full human catalogue."""
    clean_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in clean:
        clean_groups[row["component_id"]].append(row)
    human_component_ids = {row["component_id"] for row in human}
    candidates = sorted(
        (_key(group_id), group_id, group)
        for group_id, group in clean_groups.items()
        if group_id not in human_component_ids
        and all(row["question"]["type"] == kind for row in group)
    )
    picked = []
    for _, _, group in candidates:
        if len(picked) + len(group) <= quota:
            picked.extend(group)
        if len(picked) == quota:
            break
    if len(picked) != quota:
        raise ValueError(f"Cannot select {quota} clean replay {kind} rows")
    return picked


def build(
    clean_train: Path,
    clean_manifest: Path,
    human_train: Path,
    human_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    inputs = {
        "clean_train": clean_train,
        "clean_manifest": clean_manifest,
        "human_train": human_train,
        "human_manifest": human_manifest,
    }
    if {name: sha_file(path) for name, path in inputs.items()} != INPUT_SHA256:
        raise ValueError("Frozen input manifest or TRAIN bytes changed")
    clean = _read(clean_train, INPUT_SHA256["clean_train"])
    human = _read(human_train, INPUT_SHA256["human_train"])
    if len(clean) != 6262 or len(human) != 5431:
        raise ValueError("Frozen native source cardinality changed")
    chosen = []
    for bucket in HUMAN_BUCKETS:
        chosen += select_whole_groups(
            human,
            HUMAN_PER_BUCKET,
            source=f"tweeteval_train:{bucket}",
            kind="Choice",
        )
    for kind, quota in REPLAY_QUOTAS.items():
        chosen += select_replay_groups(clean, human, kind=kind, quota=quota)
    ids = [row["id"] for row in chosen]
    components = [row["component_id"] for row in chosen]
    if len(chosen) != 1024 or len(set(ids)) != len(ids):
        raise ValueError("Expected 1,024 unique native rows")
    for component in set(components):
        original = [row for row in (*clean, *human) if row["component_id"] == component]
        actual = [row for row in chosen if row["component_id"] == component]
        if len(original) != len(actual):
            raise ValueError("A source component was split or duplicated")
    chosen.sort(key=lambda row: (_key("shuffle:" + row["component_id"]), row["id"]))
    output_dir.mkdir(parents=True)
    train_path = output_dir / "train.jsonl"
    with train_path.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    train_path.chmod(0o600)
    receipt = {
        "schema": "decision2-kai06-recovery1024/1",
        "seed": SEED,
        "input_sha256": INPUT_SHA256,
        "builder_sha256": sha_file(Path(__file__)),
        "train_sha256": sha_file(train_path),
        "rows": len(chosen),
        "component_groups": len(set(components)),
        "by_type": dict(
            sorted(
                collections.Counter(row["question"]["type"] for row in chosen).items()
            )
        ),
        "by_source": dict(
            sorted(collections.Counter(row["source_id"] for row in chosen).items())
        ),
        "rights_status": "development_only; human source terms and final weight redistribution must be reviewed separately",
        "raw_text_redistribution": "prohibited",
    }
    receipt_path = output_dir / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    receipt_path.chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "clean_train",
        "clean_manifest",
        "human_train",
        "human_manifest",
        "output_dir",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    receipt = build(
        args.clean_train,
        args.clean_manifest,
        args.human_train,
        args.human_manifest,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("train_sha256", "rows", "by_type", "component_groups")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
