"""Compile the pinned public pressure dataset into a development-only panel.

The upstream *data* are CC BY-SA 4.0. This module does not copy the upstream
unlicensed harness. Run it on an experiment host and keep the generated text
files there unless their redistribution terms have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PANEL_VERSION = "decision2-pressure-dev/1"
SOURCE_REVISION = "21820dc8b455e37aebaf5a83122233b6e564c109"
SOURCE_SHA256 = {
    "items.jsonl": "15de056cb01884b8ace78083a6702f0cad6454ab64d49ff45a9b27166e7bdb86",
    "manifest.json": "bb1d264ca2d9df2f379e786dc720f50fd23bd3122c25c30e6f012a71f43b7a49",
    "gates.json": "b45a2ac858fb916381ee23ec452197ce0d436ef64f483053fad3d30877360c0a",
    "splits/rq3_hardness.json": "ef1d5116478452a0feec1bface291fa98c21cbaf941f77c509d028e2b2f49559",
    "splits/rq2_order.json": "e3ed684f4588dbbc25a4a658f8515676d432c5de76be656ea6553e45235ed4d7",
    "splits/rq1_kscaling.json": "dc8696f5fc0595f0fdda5406d9cc094833b09d68858d243647a737098dbdc147",
}
SLICES = {
    "rq3_shared_hardness": {
        "split": "rq3_hardness",
        "uids": 400,
        "domains": ("clinc", "mtop"),
        "pools": ("near", "far"),
        "k": (2, 4, 8, 16),
        "permutations": 1,
    },
    "rq2_shared_order": {
        "split": "rq2_order",
        "uids": 800,
        "domains": ("clinc", "goemotions", "fintopic", "mtop"),
        "pools": ("near", "far"),
        "k": (16,),
        "permutations": 5,
    },
    "rq1_extended_capacity": {
        "split": "rq1_kscaling",
        "uids": 800,
        "domains": ("clinc", "mtop", "goemotions", "dbpedia"),
        "pools": ("ext",),
        "k": (2, 4, 8, 16, 32, 64, 128),
        "permutations": 1,
    },
}
INSTRUCTIONS = "Which label applies here?"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else compact(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_source(repository: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify the immutable source *and all* manifest-listed data files."""
    if _git(repository, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError("pressure source revision differs from frozen protocol")
    if _git(repository, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("pressure source has modified tracked files")
    data_root = repository / "dataset" / "v3"
    for relative, expected in SOURCE_SHA256.items():
        path = data_root / relative
        if file_sha256(path) != expected:
            raise ValueError(f"source SHA-256 mismatch: {relative}")
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    gates = json.loads((data_root / "gates.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "v3"
        or manifest.get("n_items") != 1000
        or manifest.get("gates_passed") is not False
        or gates.get("G3_textfree_gold_picker", {}).get("pass") is not False
    ):
        raise ValueError("source quality-gate manifest changed")
    for relative, receipt in manifest["files"].items():
        path = data_root / relative
        if path.stat().st_size != receipt["bytes"] or not file_sha256(path).startswith(
            receipt["sha256_16"]
        ):
            raise ValueError(f"manifest-listed source file changed: {relative}")
    return data_root, manifest, gates


def read_items(
    path: Path, expected_counts: dict[str, int]
) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            uid, domain, text, gold = (
                item.get(key) for key in ("uid", "domain", "text", "gold")
            )
            if not all(
                isinstance(value, str) and value for value in (uid, domain, text, gold)
            ):
                raise ValueError(f"items:{line_number}: malformed identity/text/gold")
            if uid in items or not uid.startswith(domain + ":"):
                raise ValueError(f"items:{line_number}: duplicate or inconsistent UID")
            if digest(text.encode("utf-8")) != item.get("text_sha256"):
                raise ValueError(f"items:{line_number}: text hash mismatch")
            if type(item.get("leaked")) is not bool or item.get("text_chars") != len(
                text
            ):
                raise ValueError(
                    f"items:{line_number}: malformed leakage/text metadata"
                )
            pools = item.get("distractors")
            if not isinstance(pools, dict) or set(pools) != {"near", "far", "ext"}:
                raise ValueError(f"items:{line_number}: missing distractor pool")
            for pool, expected_size in (("near", 63), ("far", 63), ("ext", 255)):
                labels = pools[pool]
                if (
                    not isinstance(labels, list)
                    or len(labels) != expected_size
                    or any(not isinstance(label, str) or not label for label in labels)
                    or len(set(labels)) != len(labels)
                    or gold in labels
                ):
                    raise ValueError(f"items:{line_number}: invalid {pool} pool")
            items[uid] = item
            counts[domain] += 1
    if len(items) != 1000 or dict(counts) != expected_counts:
        raise ValueError("source item count by domain differs from manifest")
    return items


def read_split(
    path: Path, spec: dict[str, Any], items: dict[str, dict[str, Any]]
) -> list[str]:
    split = json.loads(path.read_text(encoding="utf-8"))
    uids = split.get("uids")
    if (
        not isinstance(uids, list)
        or len(uids) != spec["uids"]
        or len(set(uids)) != len(uids)
        or split.get("n_items") != len(uids)
        or set(split.get("domains", [])) != set(spec["domains"])
        or not set(spec["pools"]) <= set(split.get("tiers", []))
        or not set(spec["k"]) <= set(split.get("k_grid", []))
    ):
        raise ValueError(f"{path.name}: split contract changed")
    if any(
        uid not in items or items[uid]["domain"] not in spec["domains"] for uid in uids
    ):
        raise ValueError(f"{path.name}: missing UID or unexpected domain")
    if set(items[uid]["domain"] for uid in uids) != set(spec["domains"]):
        raise ValueError(f"{path.name}: domain subset missing")
    if Counter(items[uid]["domain"] for uid in uids) != dict.fromkeys(
        spec["domains"], 200
    ):
        raise ValueError(
            f"{path.name}: domain UID counts differ from frozen 200-per-domain split"
        )
    return uids


def ordered_options(
    item: dict[str, Any], pool: str, k: int, permutation: int
) -> list[str]:
    options = [item["gold"], *item["distractors"][pool][: k - 1]]
    if len(options) != k or len(set(options)) != k:
        raise ValueError("option set is not complete and unique")
    # Stable across Python processes: no built-in hash(), global RNG, or source row order.
    seed = int(digest([item["uid"], pool, k, permutation]), 16)
    random.Random(seed).shuffle(options)
    return options


def compile_one(
    item: dict[str, Any],
    slice_name: str,
    pool: str,
    k: int,
    permutation: int,
    *,
    repeat: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if slice_name not in SLICES:
        raise ValueError("unknown pressure slice")
    spec = SLICES[slice_name]
    if (
        pool not in spec["pools"]
        or k not in spec["k"]
        or not 0 <= permutation < spec["permutations"]
    ):
        raise ValueError("pool/K/permutation is outside frozen slice")
    if repeat and (slice_name != "rq2_shared_order" or permutation != 0):
        raise ValueError("only order permutation zero has an identical-order repeat")
    options = ordered_options(item, pool, k, permutation)
    payload = {
        "state": item["text"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {label: label for label in options},
            }
        },
    }
    item_id = (
        "pr-"
        + digest(
            [
                slice_name,
                item["uid"],
                pool,
                k,
                permutation,
                "repeat" if repeat else "main",
            ]
        )[:24]
    )
    prompt = {"id": item_id, **payload}
    gold = {
        "panel_version": PANEL_VERSION,
        "id": item_id,
        "slice": slice_name,
        "uid": item["uid"],
        "domain": item["domain"],
        "leaked": item["leaked"],
        "pool": pool,
        "k": k,
        "permutation": permutation,
        "repeat": repeat,
        "gold": item["gold"],
        "labels": options,
        "text_sha256": item["text_sha256"],
        "option_set_sha256": digest(sorted(options)),
        "option_order_sha256": digest(options),
        "input_sha256": digest(payload),
    }
    return prompt, gold


def compile_slice(
    slice_name: str,
    uids: list[str],
    items: dict[str, dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    spec = SLICES[slice_name]
    for uid in uids:
        item = items[uid]
        for pool in spec["pools"]:
            for k in spec["k"]:
                variants = [
                    compile_one(item, slice_name, pool, k, permutation)
                    for permutation in range(spec["permutations"])
                ]
                if len({gold["option_order_sha256"] for _, gold in variants}) != len(
                    variants
                ):
                    raise ValueError(
                        f"{uid}/{pool}/K{k}: distinct permutations have identical order"
                    )
                yield from variants
                if slice_name == "rq2_shared_order":
                    yield compile_one(item, slice_name, pool, k, 0, repeat=True)


def build(repository: Path, output_dir: Path) -> dict[str, Any]:
    data_root, source_manifest, gates = verify_source(repository)
    items = read_items(data_root / "items.jsonl", source_manifest["n_items_by_domain"])
    splits = {
        name: read_split(data_root / "splits" / f"{spec['split']}.json", spec, items)
        for name, spec in SLICES.items()
    }
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pressure-build-", dir=output_dir.parent))
    manifest: dict[str, Any] = {
        "panel_version": PANEL_VERSION,
        "role": "public_development_diagnostic_only",
        "source_repository": "https://github.com/gazelle93/decision-models-under-pressure",
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_data_license": "CC BY-SA 4.0; attribute upstream datasets and preserve ShareAlike for redistributed prompts",
        "source_code_license": "no root code license found; upstream harness not copied",
        "quality_gate": {
            "source_gates_passed": False,
            "failed_gate": "G3_textfree_gold_picker",
            "detail": gates["G3_textfree_gold_picker"]["detail"],
        },
        "prompt_contract": {
            "fields": ["id", "state", "questions"],
            "question_id": "decision",
            "question_type": "choice",
            "instructions": INSTRUCTIONS,
            "payload_sha256": "SHA-256 of UTF-8 compact ensure_ascii=False JSON of exactly {state,questions} in that order",
            "option_seed": "integer SHA-256 of UTF-8 compact JSON [uid,pool,K,permutation]",
            "control": "RQ2 repeats permutation 0 once with a different opaque id and identical payload",
        },
        "selection_warning": "Public and format-leaky; never use as a final release gate or claim held-out transfer.",
        "compiler_sha256": file_sha256(Path(__file__)),
        "slices": {},
    }
    try:
        for name, uids in splits.items():
            paths = {
                kind: staging / f"{name}.{kind}.jsonl" for kind in ("prompts", "gold")
            }
            count = control_count = 0
            seen_ids: set[str] = set()
            with paths["prompts"].open("w", encoding="utf-8") as prompts, paths[
                "gold"
            ].open("w", encoding="utf-8") as gold:
                for prompt, label in compile_slice(name, uids, items):
                    if prompt["id"] in seen_ids:
                        raise ValueError(f"duplicate compiled ID: {prompt['id']}")
                    seen_ids.add(prompt["id"])
                    prompts.write(compact(prompt) + "\n")
                    gold.write(compact(label) + "\n")
                    count += 1
                    control_count += int(label["repeat"])
            spec = SLICES[name]
            expected_main = (
                spec["uids"]
                * len(spec["pools"])
                * len(spec["k"])
                * spec["permutations"]
            )
            expected_control = (
                spec["uids"] * len(spec["pools"]) if name == "rq2_shared_order" else 0
            )
            if (count - control_count, control_count) != (
                expected_main,
                expected_control,
            ):
                raise ValueError(f"{name}: wrong main/control request count")
            manifest["slices"][name] = {
                "source_split": f"splits/{spec['split']}.json",
                "uids": spec["uids"],
                "domains": list(spec["domains"]),
                "pools": list(spec["pools"]),
                "k": list(spec["k"]),
                "permutations": spec["permutations"],
                "main_requests": expected_main,
                "control_requests": expected_control,
                "files": {
                    kind: {"name": paths[kind].name, "sha256": file_sha256(paths[kind])}
                    for kind in paths
                },
            }
        (staging / "pressure-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="pinned upstream git checkout"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new private development panel directory",
    )
    args = parser.parse_args()
    manifest = build(args.source_root, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "slices": {
                    name: {
                        key: value[key] for key in ("main_requests", "control_requests")
                    }
                    for name, value in manifest["slices"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
