"""Shared core for the HS1 generators: seeds, the Item contract, rows and text helpers.

Every family module (``f1_quote``, ``f2_policy``, ``f3_unmet``) exposes::

    FAMILY: str
    KINDS_TRAIN: tuple[str, ...]      # kinds / domains used in TRAIN and dev-id
    KINDS_OOD: tuple[str, ...]        # kinds / domains used only in dev-ood
    INTERFACE_SHARES: dict[str, float]
    LENGTH_SHARES: dict[str, float]
    def supported(kind: str) -> tuple[str, ...]
    def make_group(seed_key: str, kind: str, interface: str, length: str) -> list[Item]

``make_group`` is a pure function of its arguments. It returns the rows of one
generated world (a pair for F1 / F3, two cases for F2) with the oracle label
already re-checked (``gold == recheck``). Labels come only from structure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.sources.common import choice_options, noul_options, score_options
from v2.data.verifiable.core import EN_FIRST, EN_LAST, MONTHS_EN, WEEKDAYS_EN

GENERATOR = "decision2-hs1"
VERSION = "1.0.0"
ARM = "hs1"
SOURCE = "decision2_hardskills_hs1"
LANG = "en"
FAMILIES = ("hs1_quote_check", "hs1_policy_packet", "hs1_unmet_condition")
TASK_TYPES = ("choice", "noul", "score")


class GenerationError(RuntimeError):
    """A world could not satisfy its constraints; the caller must not retry blindly."""


class RecheckMismatch(GenerationError):
    """Oracle and independent re-check disagree: stops the build (never redrawn)."""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rng_for(*parts: object) -> random.Random:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


@dataclasses.dataclass(frozen=True)
class Item:
    """One native decision produced by a family generator (before option permutation).

    ``task_type`` is choice / noul / score. ``choices`` holds Choice option
    descriptions in canonical order, or Score level criteria (level 0 first);
    it is empty for Noul. ``gold`` and ``recheck`` are an index into
    ``choices`` (Choice), a level (Score) or 0 / 1 for false / true (Noul).
    ``option_refs`` maps names (e.g. ``quoted``, ``base_only``) to option
    indices in the same encoding; the row builder remaps them with the Choice
    permutation. ``meta`` is extra JSON-serializable audit metadata and must
    contain ``interface`` and ``length`` (the class).
    """

    task_type: str
    state: str
    instructions: str
    choices: tuple[str, ...]
    gold: int
    recheck: int
    kind: str
    subtype: str
    variant: str
    facts: Mapping[str, Any]
    option_refs: Mapping[str, int] = dataclasses.field(default_factory=dict)
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    probe_view: str = ""

    def check(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise GenerationError(f"bad task_type {self.task_type}")
        if not self.state.strip() or not self.instructions.strip():
            raise GenerationError("empty state or instructions")
        if self.gold != self.recheck:
            raise RecheckMismatch(
                f"oracle {self.gold} != recheck {self.recheck} ({self.kind}/{self.subtype})"
            )
        if self.task_type == "noul":
            if self.choices or self.gold not in (0, 1):
                raise GenerationError("noul items carry no choices and a 0/1 gold")
        else:
            count = len(self.choices)
            low = 2 if self.task_type == "choice" else 2
            high = 8 if self.task_type == "choice" else 10
            if not low <= count <= high or not 0 <= self.gold < count:
                raise GenerationError(
                    f"bad option count or gold ({count}, {self.gold})"
                )
            if len(set(self.choices)) != count:
                raise GenerationError("duplicate option descriptions")
        for name, index in self.option_refs.items():
            bound = 2 if self.task_type == "noul" else len(self.choices)
            if not 0 <= index < bound:
                raise GenerationError(f"option_ref {name} out of range")
        for field in ("interface", "length"):
            if field not in self.meta:
                raise GenerationError(f"meta.{field} missing")
        canonical(dict(self.facts))
        canonical(dict(self.meta))


# ---------------------------------------------------------------- schedules


def schedule(shares: Mapping[str, float], count: int, seed: str) -> list[str]:
    """Exact largest-remainder allocation of ``count`` slots, deterministically shuffled."""
    total = sum(shares.values())
    if count <= 0 or total <= 0:
        return []
    raw = {key: count * value / total for key, value in shares.items()}
    alloc = {key: int(value) for key, value in raw.items()}
    rest = count - sum(alloc.values())
    for key in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[:rest]:
        alloc[key] += 1
    slots = [key for key in sorted(alloc) for _ in range(alloc[key])]
    rng_for("hs1-schedule", seed).shuffle(slots)
    return slots


# ---------------------------------------------------------------- rows


def _permutation(
    count: int, golds: Sequence[int], target: int, rng: random.Random
) -> list[int]:
    """Return ``order`` (new position -> old index) placing the distinct ``golds`` at
    consecutive positions ``target, target + 1, ...`` (mod count); others shuffled."""
    placed: dict[int, int] = {}
    for offset, gold in enumerate(dict.fromkeys(golds)):
        placed[(target + offset) % count] = gold
    others = [index for index in range(count) if index not in placed.values()]
    rng.shuffle(others)
    fill_iter = iter(others)
    return [placed[pos] if pos in placed else next(fill_iter) for pos in range(count)]


def build_rows(
    family: str,
    seed_key: str,
    items: Sequence[Item],
    *,
    split: str,
    slice_name: str,
    gold_target: int,
) -> list[dict[str, Any]]:
    """Turn one world's Items into validated training-contract rows.

    Choice options are permuted so that the first Choice item's gold lands at
    ``gold_target`` (mod its option count). Items of a world that share the
    same option descriptions share one permutation (true minimal pairs).
    """
    if split not in ("train", "select"):
        raise ValueError(split)
    group_id = f"{ARM}:{family}:{sha(seed_key)[:24]}"
    rng = rng_for("hs1-perm", seed_key)
    shared: dict[tuple[str, ...], list[int]] = {}
    rows = []
    for position, item in enumerate(items):
        item.check()
        refs = dict(item.option_refs)
        gold, recheck = item.gold, item.recheck
        if item.task_type == "choice":
            count = len(item.choices)
            order = shared.get(item.choices)
            if order is None:
                target = (gold_target + position) % count
                golds = [
                    other.gold
                    for other in items
                    if other.task_type == "choice" and other.choices == item.choices
                ]
                order = _permutation(count, golds, target, rng)
                shared[item.choices] = order
            new_of_old = {old: new for new, old in enumerate(order)}
            descriptions = [item.choices[old] for old in order]
            options = choice_options(descriptions)
            gold, recheck = new_of_old[gold], new_of_old[recheck]
            refs = {name: new_of_old[index] for name, index in refs.items()}
        elif item.task_type == "score":
            options = score_options(list(item.choices))
        else:
            options = noul_options(LANG)
        audit = {
            "generator": GENERATOR,
            "version": VERSION,
            "kind": item.kind,
            "subtype": item.subtype,
            "seed_group": seed_key,
            "slice": slice_name,
            "facts_sha256": digest(dict(item.facts)),
            "oracle_label": gold,
            "recheck_label": recheck,
            "option_refs": refs,
            **dict(item.meta),
        }
        if item.probe_view:
            audit["probe_view"] = item.probe_view
        row = {
            "id": f"{ARM}-{family}-{sha(seed_key + '|' + str(position))[:24]}",
            "state": item.state,
            "instructions": item.instructions,
            "options": options,
            "label": gold,
            "task_type": item.task_type,
            "family": family,
            "group_id": group_id,
            "language": LANG,
            "split": split,
            "evaluation_role": split,
            "source": SOURCE,
            "render_template": f"hs1/{family}/{item.kind}/{item.variant}",
            "audit_metadata": audit,
        }
        row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
        rows.append(validate_row(row, split))
    return rows


# ---------------------------------------------------------------- names

HS1_FIRST = (
    "Abena",
    "Adrian",
    "Aisha",
    "Alejandro",
    "Alina",
    "Anders",
    "Anika",
    "Arjun",
    "Beatriz",
    "Bilal",
    "Bronwyn",
    "Callum",
    "Camila",
    "Chidi",
    "Clara",
    "Damian",
    "Daria",
    "Declan",
    "Delphine",
    "Emeka",
    "Esther",
    "Farah",
    "Felix",
    "Fiona",
    "Gabriel",
    "Greta",
    "Hana",
    "Hamza",
    "Helena",
    "Ivan",
    "Isabel",
    "Jasper",
    "Jonas",
    "Joanna",
    "Kamal",
    "Katrin",
    "Keanu",
    "Lara",
    "Leon",
    "Lucia",
    "Magnus",
    "Maren",
    "Marisol",
    "Milan",
    "Nadia",
    "Nikolai",
    "Noor",
    "Oskar",
    "Paloma",
    "Pavel",
    "Quentin",
    "Rafael",
    "Rania",
    "Rosalind",
    "Samir",
    "Selin",
    "Soren",
    "Tamsin",
    "Teodor",
    "Thandiwe",
    "Ursula",
    "Valentina",
    "Viktor",
    "Wanjiru",
    "Xavier",
    "Yara",
    "Yusuf",
    "Zainab",
    "Zoran",
    "Elif",
    "Mirela",
    "Tobias",
)
HS1_LAST = (
    "Achterberg",
    "Alvarez",
    "Amundsen",
    "Bakshi",
    "Balogun",
    "Barros",
    "Castellano",
    "Chaudhry",
    "Dahl",
    "Delacroix",
    "Donnelly",
    "Eriksen",
    "Esposito",
    "Fagerlund",
    "Farouk",
    "Gallagher",
    "Gonzaga",
    "Halvorsen",
    "Hakimi",
    "Horvath",
    "Ibekwe",
    "Jablonski",
    "Kaminski",
    "Karimi",
    "Kowalczyk",
    "Laurent",
    "Lindahl",
    "Mbeki",
    "Moreau",
    "Nakamura",
    "Novak",
    "Okafor",
    "Olsen",
    "Petrov",
    "Quinlan",
    "Rahman",
    "Reyes",
    "Rinaldi",
    "Sandoval",
    "Schreiber",
    "Sorensen",
    "Takahashi",
    "Thornton",
    "Uddin",
    "Varga",
    "Vasquez",
    "Whitfield",
    "Wojcik",
    "Yilmaz",
    "Zamora",
    "Zielinski",
    "Abernathy",
    "Brennan",
    "Caldwell",
    "Duarte",
    "Ferreira",
)
FIRST_NAMES = tuple(dict.fromkeys(EN_FIRST + HS1_FIRST))
LAST_NAMES = tuple(dict.fromkeys(EN_LAST + HS1_LAST))


def people(rng: random.Random, count: int, exclude: Iterable[str] = ()) -> list[str]:
    """Distinct full names; no two share a first or a last name."""
    blocked: set[str] = set()
    for name in exclude:
        blocked.update(name.split())
    result: list[str] = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > 10_000:
            raise GenerationError("name pool exhausted")
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        if first in blocked or last in blocked:
            continue
        blocked.update((first, last))
        result.append(f"{first} {last}")
    return result


def surname(name: str) -> str:
    return name.split()[-1]


def given(name: str) -> str:
    return name.split()[0]


# ---------------------------------------------------------------- dates, times, money

DATE_STYLES = ("mdy", "dmy", "iso", "mdy_short", "dmy_weekday")


def fmt_date(value: date, style: str = "mdy") -> str:
    month = MONTHS_EN[value.month - 1]
    if style == "mdy":
        return f"{month} {value.day}, {value.year}"
    if style == "dmy":
        return f"{value.day} {month} {value.year}"
    if style == "iso":
        return value.isoformat()
    if style == "mdy_short":
        return f"{month[:3]} {value.day}, {value.year}"
    if style == "dmy_weekday":
        return f"{WEEKDAYS_EN[value.weekday()]} {value.day} {month} {value.year}"
    raise ValueError(style)


def weekday(value: date) -> str:
    return WEEKDAYS_EN[value.weekday()]


def random_date(
    rng: random.Random, start: date = date(2025, 1, 6), span_days: int = 600
) -> date:
    return start + timedelta(days=rng.randrange(span_days))


def is_business_day(value: date, holidays: Iterable[date] = ()) -> bool:
    return value.weekday() < 5 and value not in set(holidays)


def add_business_days(value: date, days: int, holidays: Iterable[date] = ()) -> date:
    """Move forward ``days`` business days (the start day itself is not counted)."""
    holiday_set = set(holidays)
    current = value
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if is_business_day(current, holiday_set):
            remaining -= 1
    return current


def business_days_between(start: date, end: date, holidays: Iterable[date] = ()) -> int:
    """Business days after ``start`` up to and including ``end`` (0 if end <= start)."""
    holiday_set = set(holidays)
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if is_business_day(current, holiday_set):
            count += 1
    return count


def fmt_time(hour: int, minute: int, style: str = "24h") -> str:
    if style == "24h":
        return f"{hour:02d}:{minute:02d}"
    suffix = "a.m." if hour < 12 else "p.m."
    display = hour % 12 or 12
    return f"{display}:{minute:02d} {suffix}"


def fmt_money(amount: float, style: str = "usd") -> str:
    """Two-decimal amounts unless whole; styles: usd ($1,250.00), eur (EUR 1,250.00), credits."""
    whole = abs(amount - round(amount)) < 1e-9
    text = f"{amount:,.0f}" if whole else f"{amount:,.2f}"
    if style == "usd":
        return f"${text}"
    if style == "eur":
        return f"EUR {text}"
    if style == "gbp":
        return f"£{text}"
    if style == "credits":
        return f"{text} credits"
    raise ValueError(style)


def fmt_int(value: int) -> str:
    return f"{value:,}"


# ---------------------------------------------------------------- text helpers


def join_list(parts: Sequence[str], conjunction: str = "and") -> str:
    parts = list(parts)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {conjunction} {parts[1]}"
    return ", ".join(parts[:-1]) + f", {conjunction} {parts[-1]}"


def pick(rng: random.Random, options: Sequence[Any]) -> Any:
    return options[rng.randrange(len(options))]


def fill(template: str, **slots: Any) -> str:
    """``str.format`` that fails loudly on a missing slot."""
    try:
        return template.format(**slots)
    except KeyError as exc:  # pragma: no cover - template bug
        raise GenerationError(f"template slot missing: {exc}") from exc


def require_present(state: str, needles: Iterable[str]) -> None:
    """Every decisive fact string must occur verbatim in the rendered state."""
    for needle in needles:
        if needle not in state:
            raise GenerationError(f"decisive fact not rendered: {needle!r}")


def within(text: str, low: int, high: int) -> bool:
    return low <= len(text) <= high


_MASK_RULES = (
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), " D "),
    (re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?"), " $ "),
    (re.compile(r"\d+(?:[.,:]\d+)*"), "#"),
)
_CAPITAL = re.compile(r"(?<=[^.!?\n] )[A-Z][A-Za-z'\-]+")


def mask(text: str) -> str:
    """Skeleton text for probes and structural scans: digits, dates, money, names masked."""
    out = text
    for pattern, repl in _MASK_RULES:
        out = pattern.sub(repl, out)
    out = _CAPITAL.sub("X", out)
    for month in MONTHS_EN:
        out = re.sub(rf"\b{month}\b", "M", out)
    return re.sub(r"\s+", " ", out).strip().lower()


def assemble(
    rng: random.Random,
    core_blocks: Sequence[str],
    filler_blocks: Sequence[str],
    low: int,
    high: int,
    *,
    separator: str = "\n\n",
    keep_order: bool = True,
) -> str:
    """Interleave filler blocks between core blocks until the text reaches [low, high] chars.

    Core blocks keep their relative order. Filler is consumed from a shuffled
    copy; if the core alone exceeds ``high`` or the filler cannot reach ``low``
    a GenerationError is raised so the caller can redraw its world.
    """
    core = list(core_blocks)
    base = separator.join(core)
    if len(base) > high:
        raise GenerationError(f"core text {len(base)} chars exceeds {high}")
    pool = list(filler_blocks)
    rng.shuffle(pool)
    chosen: list[str] = []
    length = len(base)
    for block in pool:
        if length >= low:
            break
        if length + len(separator) + len(block) > high:
            continue
        chosen.append(block)
        length += len(separator) + len(block)
    if length < low:
        raise GenerationError(f"filler could not reach {low} chars (got {length})")
    slots: list[list[str]] = [[] for _ in range(len(core) + 1)]
    for block in chosen:
        slots[rng.randrange(len(slots))].append(block)
    parts: list[str] = []
    for index in range(len(core) + 1):
        parts.extend(slots[index])
        if index < len(core):
            parts.append(core[index])
    text = separator.join(parts)
    if not low <= len(text) <= high:
        raise GenerationError("assembled text out of range")
    return text
