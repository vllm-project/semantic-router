"""Deterministic SYN1 seed plan: domain x archetype x question shape x conditions x planned answers.

A seed is one task family. It gets one DESIGN call and one INSTANCES call per condition; planned
answers control label balance (choice answers cycle through a shuffled option order, yes/no pairs
are balanced, shifted-prior groups are 3:1 skewed in either direction).

    python -m d25.vega.data.synth.plan --seeds 18000 --out /data/d25/vega/synth/plan/syn1-plan.jsonl.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from d25.vega.data.synth import taxonomy as tx

SEED = 20261009
KEY_STYLES = {
    "snake_case": "short snake_case identifiers (for example late_delivery)",
    "title_case": "short Title Case labels (for example Late Delivery)",
    "codes": "short codes natural to the domain (for example P1, Tier B, Code 4) with full descriptions",
    "natural_phrases": "brief natural-language phrases as keys (for example 'Refund approved')",
    "entity_values": "the entity names or values themselves as keys, descriptions only when needed",
}
ENTITY_ARCHETYPES = {
    "long_state_lookup",
    "comparison_choice",
    "answerability",
    "knowledge_application",
}


def rng_for(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def weighted(rng: random.Random, items: list, weights: list[float]):
    return rng.choices(items, weights=weights, k=1)[0]


def option_count(rng: random.Random, low: int, high: int) -> int:
    buckets = [
        (max(a, low), min(b, high), w)
        for a, b, w in tx.OPTION_BUCKETS
        if a <= high and b >= low
    ]
    a, b, _ = weighted(rng, buckets, [w for *_, w in buckets])
    return rng.randint(a, b)


def planned_answers(rng: random.Random, kind: str, n_options: int) -> dict[str, list]:
    """Per condition, the planned answer of each variant (option index, or 'yes'/'no')."""
    answers: dict[str, list] = {}
    if kind == "noul":
        for condition, (count, layout) in tx.CONDITION_LAYOUT.items():
            if layout == "pair":
                pair = ["yes", "no"]
                rng.shuffle(pair)
                answers[condition] = pair
            else:
                dominant = rng.choice(["yes", "no"])
                minority = "no" if dominant == "yes" else "yes"
                group = [dominant] * (count - 1) + [minority]
                rng.shuffle(group)
                answers[condition] = group
        return answers
    order = list(range(n_options))
    rng.shuffle(order)
    cursor = 0

    def take() -> int:
        nonlocal cursor
        value = order[cursor % n_options]
        cursor += 1
        return value

    for condition, (count, layout) in tx.CONDITION_LAYOUT.items():
        if layout == "pair":
            first = take()
            second = take()
            if second == first:
                second = (first + 1) % n_options
            answers[condition] = [first, second]
        else:
            dominant, minority = take(), take()
            if minority == dominant:
                minority = (dominant + 1) % n_options
            group = [dominant] * (count - 1) + [minority]
            rng.shuffle(group)
            answers[condition] = group
    return answers


def make_seed(
    index: int, domains: list[tuple[str, str]], archetypes: list[str]
) -> dict:
    rng = rng_for("seed", index)
    sector, domain = domains[index % len(domains)]
    weights = [tx.ARCHETYPES[a]["weight"] for a in archetypes]
    archetype = weighted(rng, archetypes, weights)
    spec = tx.ARCHETYPES[archetype]
    types = spec["types"]
    if types == ("multi",):
        kind = "multi"
    elif len(types) == 1:
        kind = types[0]
    else:
        kind = (
            "noul"
            if rng.random() < (0.75 if types[0] == "noul" else 0.45)
            else "choice"
        )
    low, high = spec["options"]
    n_options = (
        option_count(rng, max(low, 2), high)
        if kind == "choice"
        else (2 if kind == "noul" else 0)
    )
    if kind == "choice":
        styles = list(KEY_STYLES)
        key_weights = [3, 2, 1, 1.5, 3 if archetype in ENTITY_ARCHETYPES else 0.2]
        key_style = weighted(rng, styles, key_weights)
    else:
        key_style = "snake_case"
    language = weighted(
        rng, [lang for lang, _ in tx.LANGUAGES], [w for _, w in tx.LANGUAGES]
    )
    question_language = (
        language if (language == "English" or rng.random() < 0.5) else "English"
    )
    state_format = "json" if rng.random() < tx.JSON_STATE_SHARE else "text"
    styles = {c: rng.choice(tx.STYLES) for c in tx.CONDITIONS}
    if state_format == "json":
        styles = {c: "JSON record" for c in tx.CONDITIONS}
    return {
        "seed_id": f"s{index:06d}",
        "index": index,
        "sector": sector,
        "domain": domain,
        "archetype": archetype,
        "kind": kind,
        "n_options": n_options,
        "key_style": key_style,
        "ordinal": bool(spec["ordinal"]),
        "phenomenon": rng.choice(spec["phenomena"]),
        "state_language": language,
        "question_language": question_language,
        "state_format": state_format,
        "styles": styles,
        "answers": planned_answers(rng, kind, n_options) if kind != "multi" else {},
        "gen_seed": rng.randrange(2**31),
    }


def excluded(seed: dict, holdouts: dict | None) -> bool:
    """Skip seeds that would mimic proxy-holdout families (holdouts.json 'o-proxy' entries)."""
    if not holdouts:
        return False
    text = f"{seed['sector']} {seed['domain']} {seed['archetype']}".lower()
    for entry in holdouts.get("synth_exclude_keywords", []):
        if entry.lower() in text:
            return True
    return False


def build(count: int, holdouts: dict | None = None) -> list[dict]:
    domains = list(tx.DOMAINS)
    rng_for("domains").shuffle(domains)
    archetypes = list(tx.ARCHETYPES)
    seeds, index = [], 0
    while len(seeds) < count:
        seed = make_seed(index, domains, archetypes)
        index += 1
        if not excluded(seed, holdouts):
            seeds.append(seed)
    return seeds


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--holdouts",
        type=Path,
        help="holdouts.json with optional synth_exclude_keywords",
    )
    a = ap.parse_args(argv)
    holdouts = (
        json.loads(a.holdouts.read_text())
        if a.holdouts and a.holdouts.exists()
        else None
    )
    seeds = build(a.seeds, holdouts)
    from d25.vega.data import util

    util.write_jsonl(a.out, seeds)
    summary = {
        "seeds": len(seeds),
        "domains": len({s["domain"] for s in seeds}),
        "by_archetype": dict(Counter(s["archetype"] for s in seeds)),
        "by_kind": dict(Counter(s["kind"] for s in seeds)),
        "by_language": dict(Counter(s["state_language"] for s in seeds)),
        "by_state_format": dict(Counter(s["state_format"] for s in seeds)),
        "option_counts": dict(
            sorted(
                Counter(s["n_options"] for s in seeds if s["kind"] == "choice").items()
            )
        ),
    }
    util.write_json(a.out.with_suffix("").with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
