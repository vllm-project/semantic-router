"""Retrieve repository cases by observed symptom, without consulting task answers."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

CASE_FILE = Path(__file__).with_name("cases.json")


def load_cases():
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.split(r"[\s_-]+", text.strip()))


def search_cases(symptoms: str | Iterable[str]) -> list[dict]:
    """Return matching evidence and retest conditions, with exact tags first."""
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    queries = {_normalize(item) for item in symptoms if item.strip()}
    if not queries:
        return []
    cards = load_cases()
    aliases_by_tag: dict[str, set[str]] = {}
    for path in CASE_FILE.parent.glob("locales/*/casebook-search.json"):
        for tag, aliases in json.loads(path.read_text(encoding="utf-8")).items():
            aliases_by_tag.setdefault(tag, set()).update(map(_normalize, aliases))
    ranked = []
    for card in cards:
        tags = {_normalize(key) for key in [card["id"], *card["tags"]]}
        phrases = {_normalize(phrase) for phrase in card.pop("match_phrases")}
        for tag in card["tags"]:
            phrases.update(aliases_by_tag.get(tag, ()))
        exact = len(queries & tags)
        matched = sum(any(phrase in query for query in queries) for phrase in phrases)
        if exact or matched:
            ranked.append(((exact, matched), card))
    # Keep catalog order for ties so a shared symptom starts with the simpler case.
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    return [card for _, card in ranked]
