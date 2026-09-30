"""Freeze complete source-balanced query batches before either arm runs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .newbase_data import FrozenCorpus, GroupCycle, file_digest, read_jsonl


def seed_for(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def prepare_stream(corpus: FrozenCorpus, specification: dict, output: Path) -> dict:
    """Use only explicit admitted IDs and strata, never sample a raw dataset.

    A source has one or more fixed strata (language, length or layout cells).
    Strata cycle without replacement independently of the source schedule.
    ``candidate_component_ids`` is a previously frozen list on each retrieval
    record. This function never mines negatives or changes relevance labels.
    """
    if not corpus.manifest.get("split"):
        raise ValueError("Draws require an explicit source partition")
    records = {row["id"]: row for row in corpus.records}
    sources = specification["sources"]
    if (
        not sources
        or type(specification["cycles"]) is not int
        or specification["cycles"] <= 0
    ):
        raise ValueError("A stream needs positive cycles and explicit sources")
    samplers, strata_pending, strata_rng = {}, {}, {}
    for name, source in sources.items():
        if type(source["steps_per_cycle"]) is not int or source["steps_per_cycle"] <= 0:
            raise ValueError("Every included source needs positive cycle support")
        if not source["strata"]:
            raise ValueError("Missing explicit source strata")
        for stratum, identities in source["strata"].items():
            if (
                not identities
                or len(set(identities)) != len(identities)
                or any(key not in records for key in identities)
            ):
                raise ValueError("Strata require unique admitted record IDs")
            sampler = GroupCycle(
                [records[key] for key in identities],
                seed_for(specification["seed"], name + ":" + stratum),
            )
            if not 1 <= source["batch_queries"] <= len(sampler.groups):
                raise ValueError("Batch size exceeds distinct parent support")
            samplers[name, stratum] = sampler
        strata_pending[name] = []
        strata_rng[name] = random.Random(seed_for(specification["seed"], name))
    schedule_rng = random.Random(specification["seed"])
    counts, stratum_counts, record_counts = Counter(), Counter(), Counter()
    draws = []
    for _ in range(specification["cycles"]):
        schedule = [
            name
            for name, source in sources.items()
            for _ in range(source["steps_per_cycle"])
        ]
        schedule_rng.shuffle(schedule)
        for name in schedule:
            if not strata_pending[name]:
                strata_pending[name] = sorted(sources[name]["strata"])
                strata_rng[name].shuffle(strata_pending[name])
            stratum = strata_pending[name].pop()
            selected = samplers[name, stratum].draw(sources[name]["batch_queries"])
            draws.append(
                {
                    "step": len(draws) + 1,
                    "source": name,
                    "stratum": stratum,
                    "record_ids": [row["id"] for row in selected],
                }
            )
            counts[name] += 1
            stratum_counts[name + ":" + stratum] += 1
            record_counts.update(row["id"] for row in selected)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "draws.jsonl").open("w", encoding="utf-8") as stream:
        for row in draws:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    manifest = {
        "version": 1,
        "train_manifest_sha256": corpus.manifest_sha256,
        "source_split": corpus.manifest["split"],
        "specification": specification,
        "steps": len(draws),
        "draws_sha256": file_digest(output / "draws.jsonl"),
        "source_counts": dict(counts),
        "stratum_counts": dict(stratum_counts),
        "record_counts": dict(record_counts),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_stream(directory: Path, corpus: FrozenCorpus) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if (
        manifest["source_split"] != corpus.manifest["split"]
        or manifest["train_manifest_sha256"] != corpus.manifest_sha256
    ):
        raise ValueError("Draw stream belongs to a different admitted TRAIN corpus")
    if file_digest(directory / "draws.jsonl") != manifest["draws_sha256"]:
        raise ValueError("Frozen draw stream changed")
    draws = read_jsonl(directory / "draws.jsonl")
    records = {row["id"]: row for row in corpus.records}
    if len(draws) != manifest["steps"]:
        raise ValueError("Draw stream step count changed")
    for step, draw in enumerate(draws, start=1):
        identities = draw["record_ids"]
        if (
            draw["step"] != step
            or len(set(identities)) != len(identities)
            or not identities
        ):
            raise ValueError("Draws must be contiguous and contain unique record IDs")
        parents = set()
        for identity in identities:
            if identity not in records:
                raise ValueError("Draw refers to a record outside admitted TRAIN")
            groups = set(records[identity]["parent_groups"])
            if parents & groups:
                raise ValueError("One logical batch contains related parent groups")
            parents.update(groups)
    return draws, manifest
