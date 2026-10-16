"""Content identities and deterministic grouped sampling for encoder training."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_PAIR_SIZE = 2


def text_digest(text: str, *, normalize: bool = False) -> str:
    """Hash complete text; normalized identity never replaces original bytes."""
    if normalize:
        text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without requiring its complete contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _unique_strings(values, field: str) -> set[str]:
    if not isinstance(values, list) or any(
        not isinstance(x, str) or not x for x in values
    ):
        raise ValueError(f"{field} must contain nonempty string identities")
    result = set(values)
    if len(result) != len(values):
        raise ValueError(f"Duplicate {field}")
    return result


def _contrastive_preferences(record: dict) -> set[str]:
    preferences = _unique_strings(
        record.get("contrastive_preference_component_ids", []),
        "contrastive preferences",
    )
    if preferences and (
        not preferences <= set(record.get("unjudged_component_ids", []))
        or not preferences <= set(record.get("candidate_component_ids", []))
    ):
        raise ValueError("Contrastive preferences must be explicit unjudged candidates")
    return preferences


def _contrastive_ignored(record: dict) -> set[str]:
    ignored = _unique_strings(
        record.get("contrastive_ignored_component_ids", []),
        "contrastive ignored candidates",
    )
    if ignored and (
        not ignored <= set(record.get("unjudged_component_ids", []))
        or not ignored <= set(record.get("candidate_component_ids", []))
        or ignored & set(record.get("positive_component_ids", []))
        or ignored & set(record.get("judged_negative_component_ids", []))
        or ignored & _contrastive_preferences(record)
    ):
        raise ValueError(
            "Ignored candidates must be explicit unjudged candidates without preferences"
        )
    return ignored


def validate_record(record: dict, components: dict[str, dict], split: str) -> None:
    """Validate a query's complete positive/negative/unknown partition."""
    if not split or record.get("split") != split:
        raise ValueError("Record split differs from the requested frozen partition")
    if not record.get("id") or not record.get("source") or not record.get("language"):
        raise ValueError("Missing record identity, source or language")
    if not _unique_strings(record.get("parent_groups"), "parent_groups"):
        raise ValueError("Every record needs a parent group")
    preferences = _contrastive_preferences(record)
    ignored = _contrastive_ignored(record)
    weak_field = "unjudged_preference_component_ids"
    if weak_field in record:
        weak = _unique_strings(record[weak_field], "unjudged preferences")
        if (
            "query_component_id" not in record
            or not weak <= set(record.get("unjudged_component_ids", []))
            or not weak <= set(record.get("candidate_component_ids", []))
        ):
            raise ValueError("Weak preferences require explicit unjudged candidates")
    if "component_id" in record:
        if (
            preferences
            or ignored
            or any(
                key in record
                for key in ("pair_component_ids", "query_component_id", "label")
            )
        ):
            raise ValueError("Representation inputs cannot imply pair/relevance labels")
        refs = {record["component_id"]}
    elif "pair_component_ids" in record:
        if preferences or ignored:
            raise ValueError(
                "Semantic pairs cannot declare retrieval preferences or ignored candidates"
            )
        refs = _unique_strings(record.get("pair_component_ids"), "pair components")
        if len(refs) != _PAIR_SIZE or not 0 <= record.get("label", -1) <= 1:
            raise ValueError(
                "Semantic pairs need two distinct components and a [0,1] label"
            )
        if record.get("label_kind") == "binary" and record["label"] not in (0, 1):
            raise ValueError("Binary pair labels must be zero or one")
    else:
        positive = _unique_strings(record.get("positive_component_ids"), "positives")
        negative = _unique_strings(
            record.get("judged_negative_component_ids"), "negatives"
        )
        unknown = _unique_strings(record.get("unjudged_component_ids"), "unjudged")
        if not positive or positive & negative or (positive | negative) & unknown:
            raise ValueError("Missing positive or conflicting relevance partitions")
        refs = positive | negative | unknown | {record.get("query_component_id")}
    if any(key not in components for key in refs):
        raise ValueError("Record refers to a missing complete text component")


@dataclass(frozen=True)
class FrozenCorpus:
    """A hashed partition; source selection and permissions belong to its producer."""

    components: dict[str, dict]
    records: tuple[dict, ...]
    manifest_sha256: str
    manifest: dict

    @classmethod
    def load(cls, directory: Path, split: str) -> FrozenCorpus:
        manifest_path = directory / "manifest.json"
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if not split or manifest.get("split") != split:
            raise ValueError("Manifest split differs from the requested partition")
        for name in ("components.jsonl", "records.jsonl"):
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            if not expected or file_digest(directory / name) != expected:
                raise ValueError(f"Frozen corpus file changed: {name}")
        components = {}
        for row in read_jsonl(directory / "components.jsonl"):
            identity, text = row.get("id"), row.get("text")
            if (
                not identity
                or identity in components
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError("Duplicate identity or empty text component")
            if row.get("text_sha256") != text_digest(text):
                raise ValueError("Complete text hash differs from admitted component")
            if row.get("normalized_sha256") != text_digest(text, normalize=True):
                raise ValueError("Normalized component identity differs")
            if not _unique_strings(row.get("parent_groups"), "component parents"):
                raise ValueError("Missing component parents")
            components[identity] = row
        records = read_jsonl(directory / "records.jsonl")
        identities = set()
        for record in records:
            validate_record(record, components, split)
            if record["id"] in identities:
                raise ValueError("Duplicate query/pair record")
            identities.add(record["id"])
        if not records:
            raise ValueError("An admitted corpus cannot be empty")
        for name in ("components.jsonl", "records.jsonl"):
            if file_digest(directory / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Frozen corpus changed during loading: {name}")
        return cls(
            components, tuple(records), hashlib.sha256(raw).hexdigest(), manifest
        )


class GroupCycle:
    """Without-replacement parent sampling; translations share one cycle entry."""

    def __init__(self, records: list[dict], seed: int):
        # Group tuples are not sufficient: [A,B] and [B,C] are one family.
        roots: dict[str, str] = {}

        def find(key):
            roots.setdefault(key, key)
            while roots[key] != key:
                roots[key] = roots[roots[key]]
                key = roots[key]
            return key

        for record in records:
            parents = sorted(_unique_strings(record["parent_groups"], "parent groups"))
            if not parents:
                raise ValueError("Cannot cycle ungrouped examples")
            find(parents[0])
            for parent in parents[1:]:
                roots[find(parent)] = find(parents[0])
        families: dict[str, set[str]] = {}
        for parent in list(roots):
            families.setdefault(find(parent), set()).add(parent)
        self.groups: dict[tuple[str, ...], list[dict]] = {}
        for record in records:
            group = tuple(sorted(families[find(record["parent_groups"][0])]))
            self.groups.setdefault(group, []).append(record)
        if not self.groups:
            raise ValueError("Cannot sample an empty source/length/language pool")
        self.rng = random.Random(seed)
        self.pending: list[tuple[str, ...]] = []
        self.draw_counts = dict.fromkeys(self.groups, 0)
        self.total_draws = 0

    def draw(self, count: int) -> list[dict]:
        if not 1 <= count <= len(self.groups):
            raise ValueError("Logical batch must fit distinct available parent groups")
        selected = []
        used: set[tuple[str, ...]] = set()
        postponed = []
        while len(selected) < count:
            if not self.pending:
                self.pending = list(self.groups)
                self.rng.shuffle(self.pending)
            group = self.pending.pop()
            if group in used:
                postponed.append(group)
                continue
            used.add(group)
            selected.append(self.rng.choice(self.groups[group]))
            self.draw_counts[group] += 1
            self.total_draws += 1
        self.pending.extend(reversed(postponed))
        return selected


def retrieval_masks(
    records: list[dict], document_ids: list[str], components: dict[str, dict]
) -> tuple[list[list[bool]], list[list[bool]]]:
    """Mask related unknowns, retaining qrels and explicit contrastive preferences.

    A producer may declare an ordered, unjudged alternative even when it shares
    a source parent. This permits a contrastive denominator without assigning
    absolute relevance gold or changing BCE/Lambda masks.
    """
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Deduplicate physical documents before scoring")
    hashes = [components[key]["normalized_sha256"] for key in document_ids]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Identical document text cannot multiply in-batch negatives")
    positive_masks, valid_masks = [], []
    for record in records:
        positive = set(record["positive_component_ids"])
        negative = set(record["judged_negative_component_ids"])
        positive_hashes = {components[key]["normalized_sha256"] for key in positive}
        positive_parents = {
            group for key in positive for group in components[key]["parent_groups"]
        }
        preference_hashes = {
            components[key]["normalized_sha256"]
            for key in _contrastive_preferences(record)
        }
        ignored = _contrastive_ignored(record)
        row_positive, row_valid = [], []
        for key in document_ids:
            item = components[key]
            is_positive = (
                key in positive or item["normalized_sha256"] in positive_hashes
            )
            related = bool(positive_parents & set(item["parent_groups"]))
            row_positive.append(is_positive)
            row_valid.append(
                is_positive
                or key in negative
                or item["normalized_sha256"] in preference_hashes
                or (key not in ignored and not related)
            )
        if not any(row_positive):
            raise ValueError("Logical candidate batch omitted a query's positive")
        positive_masks.append(row_positive)
        valid_masks.append(row_valid)
    return positive_masks, valid_masks
