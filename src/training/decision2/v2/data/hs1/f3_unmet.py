"""F3 ``hs1_unmet_condition``: unmet-condition negatives with matched positive twins.

Design: ``records/hs1-prereg-2026-09-29.md`` §1 F3. A world is a rule text with
3-6 required conditions (0-2 of them "either ... or"), 0-3 recommended items, a
case narrative and one question. ``make_group`` returns the NEGATIVE row (one
required condition fails; two in 15% of worlds) and its POSITIVE twin (each
failing attribute moved just inside the rule).

Twins keep identical masked bags of words (``core.mask``): numbers and dates move
to values with the same printed length and digit count; a document, signer or
category flips by exchanging two phrases that both twins contain (the failing
document swaps status with a document the rule does not require, the signer
with the copied person, "from X to Y" with "from Y to X"); a holder name is
masked. The designated value keeps its distance class (within 5% / 3 days of
its boundary, or not) in both twins, so negation words, role and category
words, boundary proximity and length carry no label signal. ``self_check``
measures that on generated Noul worlds.

Gold: ``solve`` solves the structured world; ``Recheck`` re-reads the rendered
state with its own patterns and comparison code and never sees the facts.
"""

from __future__ import annotations

import copy
import math
import random
import re
import string
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from v2.data.hs1 import core
from v2.data.hs1.core import GenerationError, Item, fmt_date, fmt_money, rng_for
from v2.data.hs1.f3_domains import PACKS

FAMILY = "hs1_unmet_condition"
KINDS_TRAIN = (
    "grant_application",
    "rental_application",
    "equipment_loan",
    "travel_grant",
    "club_membership",
    "event_permit",
    "scholarship",
    "volunteer_clearance",
)
KINDS_OOD = ("claim_filing", "vehicle_inspection")
INTERFACE_SHARES = {"noul": 0.70, "choice": 0.20, "score": 0.10}
LENGTH_SHARES = {"short": 0.70, "medium": 0.30}
LENGTH_RANGES = {"short": (400, 1200), "medium": (1200, 3000)}
REQ_KINDS = (
    "minimum",
    "maximum",
    "date_window",
    "before_deadline",
    "document_provided",
    "signer_role",
    "category_match",
    "count_at_least",
    "same_party",
    "valid_on_date",
)
BRANCH_KINDS = (
    "minimum",
    "maximum",
    "count_at_least",
    "document_provided",
    "category_match",
)
LAYOUTS = ("numbered", "paragraph", "checklist", "email", "form")
TWO_FAIL_SHARE = 0.15
NEAR_REL = 0.05
NEAR_DAYS = 3
NEAR_SHARE = 0.5
NEAR_MISS_SHARE = 0.85
CRED_START_SHARE = 0.3
MAX_ATTEMPTS = 50
NEGATION_WORDS = (
    "not",
    "no",
    "never",
    "without",
    "missing",
    "n't",
    "none",
    "neither",
    "nor",
    "unable",
    "lacks",
)
NEAR_MISS_WORDS = ("deputy", "acting", "assistant", "interim", "vice")


class RecheckError(GenerationError):
    """The independent re-check could not read a rendered state in exactly one way."""


def supported(kind: str) -> tuple[str, ...]:
    if kind not in PACKS:
        raise KeyError(kind)
    return ("noul", "choice", "score")


# ---------------------------------------------------------------- template banks

_FRAMES = {
    "numbered": ("{i}. {NP}.",),
    "paragraph": (
        "{subj} needs {np}.",
        "{subj} must have {np}.",
        "The rules require {np}.",
        "In addition, the rules require {np}.",
        "{subj} must also have {np}.",
    ),
    "checklist": (
        "Check that the file shows {np}.",
        "Confirm that the file shows {np}.",
        "Make sure the file shows {np}.",
    ),
    "email": (
        "You will need {np}.",
        "We also need {np}.",
        "{Office} will look for {np}.",
        "Please make sure the file shows {np}.",
    ),
    "form": ("Condition {i} requires {np}.",),
}
_REC_FRAMES = (
    "{NP} is recommended but not required.",
    "It is recommended, though not required, to include {np}.",
    "Optional: {np}.",
    "{Office} also likes to see {np}, but this is optional.",
)
_RULE_INTROS = (
    "An application succeeds only if every required condition is met.",
    "Every requirement listed here must be satisfied; recommended items are optional.",
    "All of the conditions below are required unless marked as recommended.",
    "Requests that do not meet every required condition are not approved.",
    "Each required condition must hold on its own, and nothing else can make up for a missing one.",
)

_NP_MEASURE = {
    ">=": (
        "{a_noun} of at least {val}",
        "{a_noun} of no less than {val}",
        "{a_noun} of {val} or more",
        "a minimum {noun} of {val}",
    ),
    ">": ("{a_noun} of more than {val}", "{a_noun} above {val}"),
    "<=": (
        "{a_noun} of no more than {val}",
        "{a_noun} of at most {val}",
        "{a_noun} of {val} or less",
        "{a_noun} that does not exceed {val}",
        "a maximum {noun} of {val}",
    ),
    "<": (
        "{a_noun} of less than {val}",
        "{a_noun} below {val}",
        "{a_noun} under {val}",
    ),
}
_NP_COUNT = {
    ">=": (
        "at least {val}",
        "{n} or more {unit}",
        "no fewer than {val}",
        "a minimum of {val}",
    ),
    "<=": (
        "{head} no more than {val}",
        "{head} at most {val}",
        "{head} {val} or fewer",
    ),
    "<": ("{head} fewer than {val}",),
}
_NP_WINDOW = (
    "{a_noun} from {start} through {end}",
    "{a_noun} no earlier than {start} and no later than {end}",
    "{a_noun} between {start} and {end}, both dates included",
    "{a_noun} falling on or between {start} and {end}",
)
_NP_DEADLINE = {
    "<=": (
        "{a_thing} submitted on or before {date}",
        "{a_thing} submitted no later than {date}",
        "{a_thing} handed in by {date}",
    ),
    "<": ("{a_thing} submitted before {date}",),
}
_NP_SIGNER = (
    "a signature from the {role}{alt} on the {form}",
    "the {form} signed by the {role}{alt}",
    "sign-off on the {form} by the {role}{alt}",
)
_NP_PARTY = (
    "{a_thing} in the {person}'s own name",
    "{a_thing} held in the {person}'s own name",
    "{a_thing} whose named holder is the {person}",
)
_NP_CRED = (
    "{a_cred} that is valid on the {anchor}",
    "{a_cred} still in force on the {anchor}",
    "{a_cred} that covers the {anchor}",
)

_CL_MEASURE = (
    "the file lists {a_noun} of {val}",
    "the {noun} stated on the form is {val}",
    "the {noun} on record is {val}",
    "the documents show {a_noun} of {val}",
    "the form gives {a_noun} of {val}",
)
_CL_COUNT = (
    "the application lists {val}",
    "the file shows {val}",
    "the form records {val}",
    "{val} are recorded on the form",
)
_CL_WINDOW = (
    "the requested {noun} is {date}",
    "the form gives {a_noun} of {date}",
    "the {noun} on the form is {date}",
)
_CL_DEADLINE = (
    "the {thing} was submitted on {date}",
    "the {thing} reached {office} on {date}",
    "the {thing} is date-stamped {date}",
    "the {thing} came in on {date}",
)
_CL_SIGNER = (
    "the {form} was signed by the {signer}, with the {other} copied",
    "the signature on the {form} is that of the {signer}, and the {other} is named as the contact",
    "the {form} was signed by {sname}, the {signer}, while {oname}, the {other}, was copied",
    "the {signer} signed the {form} after a call with the {other}",
)
_CL_PARTY = (
    "the {thing} is in the name of {holder}",
    "the named holder of the {thing} is {holder}",
    "the {thing} is registered to {holder}",
    "the {thing} lists {holder} as the holder",
)
_CL_CRED_UNTIL = (
    "the {cred} is valid through {until}",
    "the {cred} remains valid through {until}",
)
_CL_CRED_BOTH = (
    "the {cred} runs from {start} through {until}",
    "the {cred} was issued on {start} and is valid through {until}",
    "the {cred} is valid from {start} through {until}",
)

_PROVIDED = (
    "was attached to the file",
    "is on file",
    "was uploaded on {date}",
    "came in with the form",
    "was handed in at the front desk",
    "was received last week",
    "is in the folder",
    "was enclosed with the form",
    "was uploaded with no problems",
    "came in without any delay",
)
_NOT_PROVIDED = (
    "is still pending",
    "has not arrived yet",
    "is due next week",
    "was promised but never sent",
    "was left out of the envelope",
    "is still being prepared",
    "was requested but has not been received",
    "will only arrive after the review",
    "is still missing",
    "has not been sent",
)

_RELATION = (
    "{given}'s {relation}, {other}, helped put the file together.",
    "The file also mentions {given}'s {relation}, {other}.",
    "{given}'s {relation}, {other}, dropped some of the papers off.",
)
_CONNECTORS = (
    "",
    "",
    "",
    "Also, ",
    "In addition, ",
    "Separately, ",
    "For the record, ",
)
_LEADS = (
    "The file for {name} shows the following.",
    "Next is {name}.",
    "{name} has also applied.",
    "Here is what the file says about {name}.",
)
_PREAMBLE_B = (
    "{count} {persons} are being considered.",
    "{office_cap} has files from {count} {persons}.",
    "There are {count} {persons} in this round.",
)
_COUNT_WORDS = {3: "three", 4: "four"}

_EMAIL_OPEN = (
    "Could you check the file below against our rules? The rules are as follows.",
    "Here are the rules for this round, followed by the file I would like you to check.",
    "Before we decide, please look over this file. First, the rules.",
)
_EMAIL_CASE = (
    "Here is where {given}'s file stands.",
    "Now the file itself.",
    "This is what we have on file for {given}.",
)
_EMAIL_CASE_B = ("Here are the files.", "Now the files themselves.")
_EMAIL_CLOSE = ("Thanks,", "Best regards,", "Many thanks,")
_FORM_NOTE = (
    "The conditions on this form are written out below as sentences.",
    "This summary restates each condition of the form in words.",
)
_FORM_CASE = ("Applicant section, as written up by {office}:", "Applicant section:")

_Q_NOUL = (
    (
        "meets",
        "Does {name}'s {request} meet every required condition in the rules above?",
    ),
    ("qualifies", "Based on the rules and the file, does {name}'s {request} qualify?"),
    ("proceed", "Can {name}'s {request} go ahead under these rules?"),
    ("allowed", "Under the stated rules, is {name}'s {request} allowed to proceed?"),
    ("approve", "Should {office} approve {name}'s {request} as it stands?"),
)
_Q_UNMET = (
    (
        "unmet",
        "Which required condition, if any, does {name}'s {request} fail to meet? "
        "If every requirement is satisfied, choose the option that says so.",
    ),
    (
        "unmet_b",
        "Looking at {name}'s {request}, which requirement is not met? "
        "Choose the option saying every requirement is met only if nothing fails.",
    ),
)
_Q_FIRST = (
    (
        "first",
        "Which is the first required condition, in the order the rules list them, that "
        "{name}'s {request} does not meet? If every requirement is met, choose the option "
        "that says so.",
    ),
    (
        "first_b",
        "More than one requirement may be unmet. Which unmet requirement comes first in "
        "the order the rules list them for {name}'s {request}? If none is unmet, pick the "
        "option saying every requirement is met.",
    ),
)
_Q_WHICH = (
    (
        "which",
        "Which of the {persons} described above meets every required condition? "
        "If none of them does, choose that option.",
    ),
    (
        "which_b",
        "Only {a_person} who meets every required condition can be accepted. "
        "Which {person}, if any, is that?",
    ),
)
_Q_SCORE = (
    ("grade", "Grade the compliance of {name}'s {request} on the scale below."),
    (
        "rate",
        "Using the compliance scale, how fully does {name}'s {request} satisfy the rules, "
        "including the recommended items?",
    ),
)
_ALL_MET = (
    "All requirements are met",
    "Every requirement is met",
    "None; every requirement is met",
)
_NONE_OF = (
    "None of them",
    "None of the {persons}",
    "No {person} meets every requirement",
)
SCORE_LEVELS = (
    "Not compliant: at least one required condition is not met",
    "Required conditions all met, but two or more recommended items are missing",
    "Required conditions all met, with exactly one recommended item missing",
    "Fully compliant: every required condition and every recommended item is met",
)

_GENERIC_FILLER = (
    "The office is closed on public holidays.",
    "Decisions are normally sent out by email.",
    "{given} asked to be contacted by phone rather than by post.",
    "The reviewer on duty this week works from the east building.",
    "The case was logged at {time} on a {weekday}.",
    "{given} prefers letters to go to a work address.",
    "Notes in the file are kept for two years and then archived.",
    "The front desk has moved to the ground floor for the summer.",
    "There was a short delay in scanning the post last month.",
    "Several files were reviewed at the same meeting.",
    "{given} has not asked for an interpreter.",
    "No complaints about the process were received this year.",
    "The office did not change its opening hours this season.",
    "A reminder about opening hours went out in the spring.",
    "{given} called once to check that the papers had reached the office.",
    "The reviewer's notes follow the order in which the papers were read.",
)


# ---------------------------------------------------------------- text helpers


def _a(phrase: str) -> str:
    word = phrase.split()[0].lower()
    vowel = word[0] in "aeiou" and not word.startswith(("uni", "use", "eu", "one"))
    return ("an " if vowel or word.startswith("hour") else "a ") + phrase


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lc(text: str) -> str:
    return text[:1].lower() + text[1:]


def _art(article: str, noun: str) -> str:
    return f"{article} {noun}" if article else noun


def _shape(text: str) -> tuple[int, int]:
    return len(text), sum(ch.isdigit() for ch in text)


def _iso(value: date) -> str:
    return value.isoformat()


def _day(text: str) -> date:
    return date.fromisoformat(text)


_SCALE = {"dec1": 10, "dec2": 100}


def _scale(inst: dict[str, Any]) -> int:
    return _SCALE.get(inst["fmt"], 1)


def fmt_value(inst: dict[str, Any], value: int, currency: str) -> str:
    """Printed form of a quantity value stored as an integer in the instance's scale."""
    fmt = inst["fmt"]
    if fmt == "money":
        return fmt_money(value, currency)
    if fmt == "pct":
        return f"{value}%"
    if fmt == "dec1":
        text = f"{value / 10:.1f}"
    elif fmt == "dec2":
        text = f"{value / 100:.2f}"
    else:
        text = f"{value:,}"
    return f"{text} {inst['unit']}" if inst["unit"] else text


def _vstep(inst: dict[str, Any], threshold: int) -> int:
    if inst["fmt"] in ("dec1", "dec2", "pct") or inst["style"] == "count":
        return 1
    big = 50 if inst["fmt"] == "money" else 100
    for bound, step in ((10_000, big), (1_000, 10), (100, 5)):
        if threshold >= bound:
            return step
    return 1


def _fails(op: str, value: Any, threshold: Any) -> bool:
    if op == ">=":
        return value < threshold
    if op == ">":
        return value <= threshold
    if op == "<=":
        return value > threshold
    if op == "<":
        return value >= threshold
    raise ValueError(op)


# ---------------------------------------------------------------- value designers


def _num_candidates(
    inst: dict[str, Any], op: str, threshold: int, region: str, klass: str
) -> list[int]:
    step = _vstep(inst, threshold)
    span = int(threshold * 0.42 / step) + 3
    upper = 100 if inst["fmt"] == "pct" else None
    floor = 2 if inst["fmt"] == "int" else 1
    out = []
    for k in range(-span, span + 1):
        value = threshold + k * step
        if value < floor or (upper is not None and value > upper):
            continue
        if (region == "fail") != _fails(op, value, threshold):
            continue
        rel = abs(value - threshold) / threshold
        if klass == "near" and rel > NEAR_REL:
            continue
        if klass == "far" and not 0.08 <= rel <= 0.40:
            continue
        out.append(value)
    return out


_NumTable = dict[str, dict[tuple[int, int], tuple[list[int], list[int]]]]
_NUM_TABLES: dict[tuple[Any, ...], _NumTable] = {}


def _num_table(inst: dict[str, Any], op: str, threshold: int) -> _NumTable:
    """Per distance class, the failing and passing values grouped by a printed shape that both sides share.

    Every value of a condition (twin or not) is drawn from this table, so a
    threshold, distance class or printed shape never marks the designated one.
    The shape does not depend on the currency (its prefix has no digits).
    """
    key = (inst["noun"], inst["fmt"], inst["unit"], op, threshold)
    table = _NUM_TABLES.get(key)
    if table is None:
        table = {}
        for klass in ("near", "far"):
            sides: tuple[dict[tuple[int, int], list[int]], ...] = ({}, {})
            for side, region in zip(sides, ("fail", "met")):
                for value in _num_candidates(inst, op, threshold, region, klass):
                    side.setdefault(_shape(fmt_value(inst, value, "usd")), []).append(
                        value
                    )
            shared = sorted(set(sides[0]) & set(sides[1]))
            if shared:
                table[klass] = {
                    shape: (sides[0][shape], sides[1][shape]) for shape in shared
                }
        _NUM_TABLES[key] = table
    return table


def twin_grid(inst: dict[str, Any], op: str) -> list[int]:
    """Grid thresholds that admit a failing and a passing value of one printed shape."""
    return [t for t in _grid(inst) if _num_table(inst, op, t)]


def _num_class(rng: random.Random, table: dict[str, Any]) -> str:
    """The distance class of a value: near with NEAR_SHARE when both classes admit twins."""
    if len(table) == 2:
        return "near" if rng.random() < NEAR_SHARE else "far"
    if not table:
        raise GenerationError("no twin-capable values")
    return next(iter(table))


def _num_twin(
    rng: random.Random, inst: dict[str, Any], op: str, threshold: int
) -> tuple[int, int]:
    """A failing value and a passing value of the same printed shape and distance class."""
    by_shape = _num_table(inst, op, threshold)[
        _num_class(rng, _num_table(inst, op, threshold))
    ]
    bad, shape = rng.choice(
        [(value, shape) for shape, (fails, _) in by_shape.items() for value in fails]
    )
    return bad, rng.choice(by_shape[shape][1])


def _num_single(
    rng: random.Random, inst: dict[str, Any], op: str, threshold: int, region: str
) -> int:
    by_shape = _num_table(inst, op, threshold)[
        _num_class(rng, _num_table(inst, op, threshold))
    ]
    side = 0 if region == "fail" else 1
    return rng.choice([value for pair in by_shape.values() for value in pair[side]])


def _offsets(klass: str, near: range, far: range) -> range:
    return near if klass == "near" else far


def _window_dates(
    start: date, end: date, region: str, klass: str
) -> list[tuple[date, str]]:
    """Dates with their side ('start' / 'end') of the window [start, end]."""
    out: list[tuple[date, str]] = []
    if region == "fail":
        for d in _offsets(klass, range(1, NEAR_DAYS + 1), range(6, 41)):
            out += [
                (start - timedelta(days=d), "start"),
                (end + timedelta(days=d), "end"),
            ]
        return out
    for d in _offsets(klass, range(0, NEAR_DAYS + 1), range(6, 41)):
        for day, side in (
            (start + timedelta(days=d), "start"),
            (end - timedelta(days=d), "end"),
        ):
            gap = min((day - start).days, (end - day).days)
            if start <= day <= end and (
                gap <= NEAR_DAYS if klass == "near" else gap >= 6
            ):
                out.append((day, side))
    return out


def _deadline_dates(deadline: date, op: str, region: str, klass: str) -> list[date]:
    if region == "fail":
        near = range(0, NEAR_DAYS) if op == "<" else range(1, NEAR_DAYS + 1)
        return [
            deadline + timedelta(days=d) for d in _offsets(klass, near, range(6, 31))
        ]
    near = range(1, NEAR_DAYS + 1) if op == "<" else range(0, NEAR_DAYS + 1)
    return [deadline - timedelta(days=d) for d in _offsets(klass, near, range(6, 31))]


def _cred_dates(anchor: date, mode: str, region: str, klass: str) -> list[date]:
    """Expiry dates (mode 'until') or start dates (mode 'start') around the anchor."""
    if mode == "until":
        if region == "fail":
            return [
                anchor - timedelta(days=d)
                for d in _offsets(klass, range(1, NEAR_DAYS + 1), range(6, 301))
            ]
        return [
            anchor + timedelta(days=d)
            for d in _offsets(klass, range(0, NEAR_DAYS + 1), range(6, 601))
        ]
    if region == "fail":
        return [
            anchor + timedelta(days=d)
            for d in _offsets(klass, range(1, NEAR_DAYS + 1), range(6, 61))
        ]
    return [
        anchor - timedelta(days=d)
        for d in _offsets(klass, range(0, NEAR_DAYS + 1), range(6, 201))
    ]


# ---------------------------------------------------------------- world builder


def _grid(inst: dict[str, Any]) -> list[int]:
    scale = _scale(inst)
    lo, hi, step = (round(inst[k] * scale) for k in ("lo", "hi", "step"))
    return list(range(lo, hi + 1, step))


def _path(*parts: object) -> str:
    return ".".join(str(part) for part in parts)


def _get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _set(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


class _Builder:
    """Draws one structured world: the negative twin's facts plus the positive-twin updates."""

    def __init__(
        self,
        rng: random.Random,
        domain: str,
        interface: str,
        length: str,
        design: dict[str, Any],
    ) -> None:
        self.rng = rng
        self.domain = domain
        self.pack = PACKS[domain]
        self.interface = interface
        self.length = length
        self.design = design
        self.compact = design["form"] == "which_applicant" and length == "short"
        self.style = rng.choice(("mdy", "dmy", "iso", "mdy_short"))
        self.currency = rng.choice(("usd", "usd", "gbp", "eur"))
        self.base = core.random_date(rng, date(2025, 2, 3), 900)
        self.anchor = self.base + timedelta(days=rng.randrange(20, 90))
        self.used: list[str] = []
        self.docs = list(self.pack["docs"])
        rng.shuffle(self.docs)
        self.required_docs = 0
        quantities = self.pack["quantities"]
        self.pools: dict[str, list[dict[str, Any]]] = {
            "minimum": [q for q in quantities if q["kind"] == "minimum"],
            "maximum": [q for q in quantities if q["kind"] == "maximum"],
            "count_at_least": [q for q in quantities if q["kind"] == "count_at_least"],
            "date_window": list(self.pack["windows"]),
            "before_deadline": list(self.pack["deadlines"]),
            "signer_role": list(self.pack["signers"]),
            "category_match": list(self.pack["categories"]),
            "same_party": list(self.pack["parties"]),
            "valid_on_date": list(self.pack["creds"]),
        }
        for pool in self.pools.values():
            rng.shuffle(pool)

    # -- small draws

    def _date(self, value: date) -> str:
        return fmt_date(value, self.style)

    def _names(self, count: int) -> list[str]:
        names = core.people(self.rng, count, exclude=self.used)
        self.used += names
        return names

    def _name_like(self, reference: str) -> str:
        for _ in range(400):
            candidate = core.people(self.rng, 1, exclude=self.used)[0]
            if len(candidate) == len(reference):
                self.used.append(candidate)
                return candidate
        raise GenerationError("no name of matching length")

    def _status(self, provided: bool) -> str:
        text = self.rng.choice(_PROVIDED if provided else _NOT_PROVIDED)
        if "{date}" in text:
            text = text.format(
                date=self._date(self.base - timedelta(days=self.rng.randrange(1, 20)))
            )
        return text

    def _pop_doc(self) -> tuple[str, str]:
        if not self.docs:
            raise GenerationError("document pool exhausted")
        return self.docs.pop()

    def _take(self, kind: str) -> dict[str, Any] | None:
        if kind == "document_provided":
            if self.required_docs >= 2 or len(self.docs) < 6:
                return None
            self.required_docs += 1
            article, noun = self._pop_doc()
            return {"key": noun, "doc": noun, "article": article, "label": _cap(noun)}
        pool = self.pools[kind]
        return pool.pop() if pool else None

    # -- requirements

    def _plan_requirements(
        self, n_req: int, n_either: int, fail_kinds: list[str]
    ) -> list[dict[str, Any]]:
        rng = self.rng
        plan: list[dict[str, Any]] = []
        either_left = n_either
        for kind in fail_kinds:
            kinds = [kind]
            if kind in BRANCH_KINDS and either_left and rng.random() < 0.4:
                kinds.append(rng.choice(BRANCH_KINDS))
                either_left -= 1
            plan.append({"fail": True, "kinds": kinds})
        while len(plan) < n_req:
            if either_left and rng.random() < 0.5:
                plan.append(
                    {
                        "fail": False,
                        "kinds": [rng.choice(BRANCH_KINDS), rng.choice(BRANCH_KINDS)],
                    }
                )
                either_left -= 1
            else:
                plan.append({"fail": False, "kinds": [rng.choice(REQ_KINDS)]})
        reserved = {}
        for index, entry in enumerate(plan):
            if entry["fail"]:
                reserved[index] = self._take(entry["kinds"][0])
                if reserved[index] is None:
                    raise GenerationError(
                        f"no instance for failing kind {entry['kinds'][0]}"
                    )
        requirements = []
        for index, entry in enumerate(plan):
            branches = []
            for position, kind in enumerate(entry["kinds"]):
                inst = (
                    reserved.pop(index)
                    if position == 0 and index in reserved
                    else self._take(kind)
                )
                tries = 0
                while inst is None:
                    tries += 1
                    if tries > 30:
                        raise GenerationError("instance pools exhausted")
                    kind = rng.choice(
                        BRANCH_KINDS if len(entry["kinds"]) > 1 else REQ_KINDS
                    )
                    inst = self._take(kind)
                branches.append(
                    {
                        "kind": kind,
                        "key": inst["key"],
                        "label": inst["label"],
                        "inst": inst,
                        "params": self._params(kind, inst),
                    }
                )
            requirements.append({"fail": entry["fail"], "branches": branches})
        rng.shuffle(requirements)
        for index, req in enumerate(requirements, 1):
            req["id"] = f"r{index}"
            for letter, branch in zip("ab", req["branches"]):
                branch["bid"] = f"r{index}{letter}"
            labels = [b["label"] for b in req["branches"]]
            req["label"] = (
                labels[0] if len(labels) == 1 else f"{labels[0]} or {_lc(labels[1])}"
            )
        return requirements

    def _params(self, kind: str, inst: dict[str, Any]) -> dict[str, Any]:
        rng = self.rng
        tpl = rng.randrange(60)
        if kind in ("minimum", "maximum", "count_at_least"):
            if kind == "minimum":
                op = ">" if rng.random() < 0.25 else ">="
            elif kind == "maximum":
                op = "<" if rng.random() < 0.25 else "<="
            else:
                op = ">="
            grid = twin_grid(inst, op)
            if not grid:
                raise GenerationError(
                    f"no twin-capable threshold for {inst['key']} {op}"
                )
            return {"op": op, "threshold": rng.choice(grid), "tpl": tpl}
        if kind == "date_window":
            start = self.base + timedelta(days=rng.randrange(10, 80))
            end = start + timedelta(days=rng.randrange(20, 70))
            return {"start": _iso(start), "end": _iso(end), "tpl": tpl}
        if kind == "before_deadline":
            deadline = self.base + timedelta(days=rng.randrange(5, 60))
            return {
                "op": "<" if rng.random() < 0.2 else "<=",
                "date": _iso(deadline),
                "tpl": tpl,
            }
        if kind == "document_provided":
            return {"doc": inst["doc"], "article": inst["article"]}
        if kind == "signer_role":
            near = list(inst["near"])
            rng.shuffle(near)
            roll = rng.random()
            alts = [near[0]] if roll < 0.4 else []
            note = (
                rng.choice(("inline", "may"))
                if alts
                else ("exclude" if roll < 0.7 and not self.compact else "")
            )
            excluded = near[:2] if note == "exclude" else []
            return {
                "role": inst["role"],
                "alts": alts,
                "note": note,
                "excluded": excluded,
                "form": inst["form"],
                "tpl": tpl,
            }
        if kind == "category_match":
            return {"required": rng.choice(inst["pair"]), "tpl": tpl}
        return {"tpl": tpl}

    # -- case values

    def _signer_roles(
        self, params: dict[str, Any], inst: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        met = [params["role"]] + list(params["alts"])
        near_fail = [r for r in inst["near"] if r not in params["alts"]]
        return met, near_fail, list(inst["wrong"])

    def _attr(
        self, branch: dict[str, Any], region: str, who: dict[str, Any]
    ) -> dict[str, Any]:
        """A value for one branch that is the same in both twins."""
        rng = self.rng
        kind, inst, params = branch["kind"], branch["inst"], branch["params"]
        tpl = rng.randrange(60)
        if kind in ("minimum", "maximum", "count_at_least"):
            value = _num_single(rng, inst, params["op"], params["threshold"], region)
            return {"value": value, "tpl": tpl}
        if kind in ("date_window", "before_deadline"):
            return {"date": _iso(self._date_single(branch, region)), "tpl": tpl}
        if kind == "document_provided":
            return {
                "status": self._status(region == "met"),
                "provided": region == "met",
            }
        if kind == "signer_role":
            met, near_fail, wrong = self._signer_roles(params, inst)
            if region == "met":
                signer = rng.choice(met)
                other = rng.choice([r for r in met + near_fail + wrong if r != signer])
            else:
                signer = rng.choice(near_fail + wrong)
                other = rng.choice(met)
            names = self._names(2) if tpl % len(_CL_SIGNER) == 2 else [None, None]
            return {
                "signer": signer,
                "other": other,
                "sname": names[0],
                "oname": names[1],
                "tpl": tpl,
            }
        if kind == "category_match":
            required = params["required"]
            wrong = [c for c in inst["pair"] if c != required][0]
            cur, prev = (required, wrong) if region == "met" else (wrong, required)
            attr = {
                "cur": cur,
                "prev": prev if rng.random() < 0.5 else None,
                "tpl": tpl,
            }
            attr.update(self._cat_slots())
            return attr
        if kind == "same_party":
            if region == "met":
                return {"holder": who["name"], "tpl": tpl}
            return {"holder": self._relative(who)["name"], "tpl": tpl}
        if kind == "valid_on_date":
            mode = "start" if rng.random() < CRED_START_SHARE else "until"
            both = mode == "start" or rng.random() < 0.5
            value = self._date_single(branch, region, mode)
            if mode == "start":
                until = self.anchor + timedelta(days=rng.randrange(100, 700))
                return {"start": _iso(value), "until": _iso(until), "tpl": tpl}
            start = (
                min(value, self.anchor) - timedelta(days=rng.randrange(180, 720))
                if both
                else None
            )
            return {
                "start": _iso(start) if start else None,
                "until": _iso(value),
                "tpl": tpl,
            }
        raise ValueError(kind)

    def _date_table(
        self, branch: dict[str, Any], mode: str | None
    ) -> dict[str, dict[tuple[int, int], tuple[list[tuple[date, str]], ...]]]:
        """Per distance class, failing and passing dates (with their window side) of a shared printed shape."""
        kind, p = branch["kind"], branch["params"]
        table = {}
        for klass in ("near", "far"):
            if kind == "date_window":
                start, end = _day(p["start"]), _day(p["end"])
                pools = (
                    _window_dates(start, end, "fail", klass),
                    _window_dates(start, end, "met", klass),
                )
            elif kind == "before_deadline":
                pools = tuple(
                    [
                        (d, "")
                        for d in _deadline_dates(
                            _day(p["date"]), p["op"], region, klass
                        )
                    ]
                    for region in ("fail", "met")
                )
            else:
                pools = tuple(
                    [(d, "") for d in _cred_dates(self.anchor, mode, region, klass)]
                    for region in ("fail", "met")
                )
            sides: tuple[dict[tuple[int, int], list[tuple[date, str]]], ...] = ({}, {})
            for side, pool in zip(sides, pools):
                for day, where in pool:
                    side.setdefault(_shape(self._date(day)), []).append((day, where))
            shared = sorted(set(sides[0]) & set(sides[1]))
            if shared:
                table[klass] = {
                    shape: (sides[0][shape], sides[1][shape]) for shape in shared
                }
        return table

    def _date_single(
        self, branch: dict[str, Any], region: str, mode: str | None = None
    ) -> date:
        """A date on one side of the rule, drawn exactly like one side of a twin (same class and shapes)."""
        table = self._date_table(branch, mode)
        by_shape = table[_num_class(self.rng, table)]
        side = 0 if region == "fail" else 1
        return self.rng.choice(
            [day for pair in by_shape.values() for day, _ in pair[side]]
        )

    def _date_twin(
        self, branch: dict[str, Any], mode: str | None = None
    ) -> tuple[date, date]:
        table = self._date_table(branch, mode)
        by_shape = table[_num_class(self.rng, table)]
        bad, where, shape = self.rng.choice(
            [
                (day, side, shape)
                for shape, (fails, _) in by_shape.items()
                for day, side in fails
            ]
        )
        mets = by_shape[shape][1]
        same = [day for day, side in mets if side == where]
        return bad, self.rng.choice(same or [day for day, _ in mets])

    def _cat_slots(self) -> dict[str, str]:
        past = self.base - timedelta(days=self.rng.randrange(30, 400))
        return {"month": core.MONTHS_EN[past.month - 1], "date": self._date(past)}

    def _relative(self, who: dict[str, Any], like: bool = False) -> dict[str, Any]:
        if who.get("relative") is None:
            name = self._name_like(who["name"]) if like else self._names(1)[0]
            who["relative"] = {
                "name": name,
                "relation": self.rng.choice(self.pack["relations"]),
                "tpl": self.rng.randrange(60),
            }
        elif like and len(who["relative"]["name"]) != len(who["name"]):
            raise GenerationError("relative name length differs")
        return who["relative"]

    def _twin(
        self, branch: dict[str, Any], who: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Negative (failing) and positive (passing) values of the designated branch."""
        rng = self.rng
        kind, inst, params = branch["kind"], branch["inst"], branch["params"]
        tpl = rng.randrange(60)
        if kind in ("minimum", "maximum", "count_at_least"):
            bad, good = _num_twin(rng, inst, params["op"], params["threshold"])
            return {"value": bad, "tpl": tpl}, {"value": good, "tpl": tpl}
        if kind in ("date_window", "before_deadline"):
            bad, good = self._date_twin(branch)
            return {"date": _iso(bad), "tpl": tpl}, {"date": _iso(good), "tpl": tpl}
        if kind == "valid_on_date":
            mode = "start" if rng.random() < CRED_START_SHARE else "until"
            both = mode == "start" or rng.random() < 0.5
            bad, good = self._date_twin(branch, mode)
            if mode == "until":
                start = (
                    min(bad, good) - timedelta(days=rng.randrange(180, 720))
                    if both
                    else None
                )
                shared = _iso(start) if start else None
                return (
                    {"start": shared, "until": _iso(bad), "tpl": tpl},
                    {"start": shared, "until": _iso(good), "tpl": tpl},
                )
            until = _iso(self.anchor + timedelta(days=rng.randrange(100, 700)))
            return (
                {"start": _iso(bad), "until": until, "tpl": tpl},
                {"start": _iso(good), "until": until, "tpl": tpl},
            )
        if kind == "signer_role":
            met, near_fail, wrong = self._signer_roles(params, inst)
            fail_role = (
                rng.choice(near_fail)
                if near_fail and rng.random() < NEAR_MISS_SHARE
                else rng.choice(wrong)
            )
            met_role = (
                params["alts"][0]
                if params["alts"] and rng.random() < 0.6
                else params["role"]
            )
            names = self._names(2) if tpl % len(_CL_SIGNER) == 2 else [None, None]
            negative = {
                "signer": fail_role,
                "other": met_role,
                "sname": names[0],
                "oname": names[1],
                "tpl": tpl,
            }
            positive = {
                "signer": met_role,
                "other": fail_role,
                "sname": names[1],
                "oname": names[0],
                "tpl": tpl,
            }
            return negative, positive
        if kind == "category_match":
            required = params["required"]
            wrong = [c for c in inst["pair"] if c != required][0]
            slots = self._cat_slots()
            return (
                {"cur": wrong, "prev": required, "tpl": tpl, **slots},
                {"cur": required, "prev": wrong, "tpl": tpl, **slots},
            )
        if kind == "same_party":
            relative = self._relative(who, like=True)
            return {"holder": relative["name"], "tpl": tpl}, {
                "holder": who["name"],
                "tpl": tpl,
            }
        if kind == "document_provided":
            return {"status": "", "provided": False}, {"status": "", "provided": True}
        raise ValueError(kind)

    # -- the world

    def build(self) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
        rng, pack = self.rng, self.pack
        n_fail, form, target_level = (
            self.design["n_fail"],
            self.design["form"],
            self.design["target_level"],
        )
        fail_kinds = list(self.design["fail_kinds"])
        multi = form == "which_applicant"
        short = self.length == "short"
        if multi:
            count = 3 if short else rng.choice((3, 3, 4))
            n_req = 3 if short else rng.choice((3, 4, 4, 5))
        elif short:
            count = 1
            n_req = rng.choice((3, 3, 4) if form == "score" else (3, 3, 4, 4, 5))
        else:
            count = 1
            n_req = rng.choice((4, 5, 5, 6, 6))
        n_either = min(rng.choices((0, 1, 2), weights=(55, 35, 10))[0], n_req - 1)
        if short and multi:
            n_either = 0
        elif short and form == "score":
            n_either = min(n_either, 1)
        if form == "score":
            n_rec = rng.choice((2, 3))
        elif multi:
            n_rec = 0 if short else rng.choice((0, 1))
        else:
            n_rec = (
                rng.choice((0, 0, 1, 1, 2)) if short else rng.choice((0, 1, 2, 2, 3))
            )
        requirements = self._plan_requirements(n_req, n_either, fail_kinds)
        recommended = []
        for index in range(n_rec):
            article, noun = self._pop_doc()
            recommended.append({"id": f"o{index + 1}", "doc": noun, "article": article})
        star = rng.randrange(count)
        applicants = [
            {
                "name": name,
                "star": index == star,
                "attrs": {},
                "rec": {},
                "extras": [],
                "relative": None,
            }
            for index, name in enumerate(self._names(count))
        ]
        designated: list[str] = []
        counterweights: list[str] = []
        updates: list[tuple[str, Any]] = []
        others = [a for a in range(count) if a != star]
        other_fail = dict(zip(others, rng.sample(range(n_req), len(others))))
        for a, who in enumerate(applicants):
            rec_missing: set[str] = set()
            if target_level is not None:
                missing = {3: 0, 2: 1, 1: rng.choice(range(2, n_rec + 1))}[target_level]
                rec_missing = {r["id"] for r in rng.sample(recommended, missing)}
            for rec in recommended:
                provided = (
                    rec["id"] not in rec_missing
                    if target_level is not None
                    else rng.random() < 0.5
                )
                who["rec"][rec["id"]] = {
                    "status": self._status(provided),
                    "provided": provided,
                }
            for r_index, req in enumerate(requirements):
                failing = req["fail"] if a == star else other_fail.get(a) == r_index
                if failing:
                    regions = ["fail"] * len(req["branches"])
                elif len(req["branches"]) == 1:
                    regions = ["met"]
                else:
                    regions = rng.choice(
                        (["met", "fail"], ["fail", "met"], ["met", "met"])
                    )
                for b_index, branch in enumerate(req["branches"]):
                    path = _path("applicants", a, "attrs", branch["bid"])
                    if a == star and req["fail"] and b_index == 0:
                        negative, positive = self._twin(branch, who)
                        who["attrs"][branch["bid"]] = negative
                        updates.append((path, positive))
                        designated.append(path)
                        if branch["kind"] == "document_provided":
                            counterweights.append(
                                self._counterweight(a, who, negative, positive, updates)
                            )
                    else:
                        who["attrs"][branch["bid"]] = self._attr(
                            branch, regions[b_index], who
                        )
            if not multi and rng.random() < 0.35:
                article, noun = self._pop_doc()
                provided = rng.random() < 0.5
                who["extras"].append(
                    {
                        "doc": noun,
                        "article": article,
                        "status": self._status(provided),
                        "provided": provided,
                    }
                )
            if not multi and who["relative"] is None and rng.random() < 0.6:
                self._relative(who)
        facts = {
            "family": FAMILY,
            "domain": self.domain,
            "interface": self.interface,
            "length": self.length,
            "form": form,
            "date_style": self.style,
            "currency": self.currency,
            "anchor": {
                "noun": pack["anchor"]["noun"],
                "date": _iso(self.anchor),
                "tpl": rng.randrange(60),
            },
            "requirements": requirements,
            "recommended": recommended,
            "applicants": applicants,
            "star": star,
            "n_fail": n_fail,
            "fail_kinds": fail_kinds,
            "target_level": target_level,
            "designated": designated,
            "counterweights": counterweights,
            "twin": "negative",
        }
        facts["render"] = self._plan_render(facts)
        self._fill(facts)
        return facts, updates

    def _counterweight(
        self,
        a: int,
        who: dict[str, Any],
        negative: dict[str, Any],
        positive: dict[str, Any],
        updates: list[tuple[str, Any]],
    ) -> str:
        """Balance the failing document's status words with a document the rule does not require."""
        rng = self.rng
        bad, good = self._status(False), self._status(True)
        negative.update(status=bad, provided=False)
        positive.update(status=good, provided=True)
        free = [rid for rid, rec in who["rec"].items() if rec["provided"]]
        if self.interface != "score" and free and rng.random() < 0.35:
            rid = rng.choice(free)
            who["rec"][rid] = {"status": good, "provided": True}
            path = _path("applicants", a, "rec", rid)
            updates.append((path, {"status": bad, "provided": False}))
            return path
        article, noun = self._pop_doc()
        entry = {"doc": noun, "article": article, "status": good, "provided": True}
        who["extras"].append(entry)
        path = _path("applicants", a, "extras", len(who["extras"]) - 1)
        updates.append(
            (path, {"doc": noun, "article": article, "status": bad, "provided": False})
        )
        return path

    # -- render plan (identical for both twins)

    def _plan_render(self, facts: dict[str, Any]) -> dict[str, Any]:
        rng, pack = self.rng, self.pack
        form = facts["form"]
        layout = rng.choice(
            [name for name in LAYOUTS if not (self.compact and name == "email")]
        )
        bank = {
            "noul": _Q_NOUL + tuple(pack["questions"]),
            "score": _Q_SCORE,
            "which_unmet": _Q_UNMET,
            "which_unmet_first": _Q_FIRST,
            "which_applicant": _Q_WHICH,
        }[form]
        question_id, question = rng.choice(bank)
        if form == "which_applicant":
            catch_all = rng.choice(_NONE_OF).format(
                persons=pack["persons"], person=pack["person"]
            )
        elif form.startswith("which_unmet"):
            catch_all = rng.choice(_ALL_MET)
        else:
            catch_all = ""
        count = len(facts["requirements"])
        program = rng.randrange(len(pack["titles"]))
        plan: dict[str, Any] = {
            "layout": layout,
            "question_id": question_id,
            "question": question,
            "catch_all": catch_all,
            "title": pack["titles"][program],
            "rule_intro": rng.choice(_RULE_INTROS),
            "frames": [rng.randrange(60) for _ in range(count)],
            "subjects": [rng.randrange(60) for _ in range(count)],
            "rec_frames": [rng.randrange(60) for _ in facts["recommended"]],
            "office": rng.randrange(60),
            "program": program,
            "intro": rng.randrange(60),
            "leads": [rng.randrange(60) for _ in facts["applicants"]],
            "preamble": rng.randrange(60),
            "form_note": rng.randrange(60),
            "form_case": rng.randrange(60),
            "email": None,
            "compact": self.compact,
        }
        if layout == "email":
            officer, colleague = self._names(2)
            plan["email"] = {
                "officer": officer,
                "title": rng.choice(pack["officers"]),
                "colleague": colleague,
                "open": rng.randrange(60),
                "case": rng.randrange(60),
                "close": rng.randrange(60),
            }
        plan["case"] = self._plan_case(facts, layout)
        return plan

    def _clauses(self, facts: dict[str, Any], a: int) -> list[list[Any]]:
        who = facts["applicants"][a]
        out: list[list[Any]] = [
            ["fact", a, b["bid"]] for r in facts["requirements"] for b in r["branches"]
        ]
        out += [["rec", a, rec["id"]] for rec in facts["recommended"]]
        out += [["extra", a, j] for j in range(len(who["extras"]))]
        return out

    def _sentences(self, clauses: list[list[Any]]) -> list[dict[str, Any]]:
        rng = self.rng
        out = []
        index = 0
        while index < len(clauses):
            parts = [clauses[index]]
            if index + 1 < len(clauses) and rng.random() < 0.15:
                parts.append(clauses[index + 1])
                index += 1
            index += 1
            connector = (
                rng.choice(_CONNECTORS[3:])
                if rng.random() < 0.2 and not self.compact
                else ""
            )
            out.append({"c": connector, "parts": parts})
        return out

    def _plan_case(
        self, facts: dict[str, Any], layout: str
    ) -> list[list[dict[str, Any]]]:
        rng = self.rng
        has_cred = any(
            b["kind"] == "valid_on_date"
            for r in facts["requirements"]
            for b in r["branches"]
        )
        anchor = has_cred or rng.random() < 0.3
        head = [{"c": "", "parts": [["email_case"]]}] if layout == "email" else []
        if facts["form"] == "which_applicant":
            preamble = head + [{"c": "", "parts": [["preamble"]]}]
            if anchor:
                preamble.append({"c": "", "parts": [["anchor"]]})
            paragraphs = [preamble]
            for a in range(len(facts["applicants"])):
                para = [{"c": "", "parts": [["lead", a]]}]
                clauses = self._clauses(facts, a)
                rng.shuffle(clauses)
                paragraphs.append(para + self._sentences(clauses))
            return paragraphs
        first = head + [{"c": "", "parts": [["intro"]]}]
        if facts["applicants"][0]["relative"]:
            first.append({"c": "", "parts": [["relation", 0]]})
        clauses = self._clauses(facts, 0) + ([["anchor"]] if anchor else [])
        rng.shuffle(clauses)
        sentences = self._sentences(clauses)
        chunks = 1 if len(sentences) <= 3 else rng.choice((1, 2, 2, 3))
        bounds = (
            [0]
            + sorted(rng.sample(range(1, len(sentences)), chunks - 1))
            + [len(sentences)]
        )
        parts = [sentences[bounds[i] : bounds[i + 1]] for i in range(chunks)]
        return [first + parts[0]] + parts[1:]

    def _fill(self, facts: dict[str, Any]) -> None:
        """Insert non-decisive filler sentences until the state reaches its length class."""
        rng = self.rng
        low, high = LENGTH_RANGES[self.length]
        multi = facts["form"] == "which_applicant"
        given = core.given(facts["applicants"][0]["name"])
        office = _pick(self.pack["offices"], facts["render"]["office"])
        pool = []
        for text in tuple(self.pack["filler"]) + _GENERIC_FILLER:
            if multi and "{given}" in text:
                continue
            clock = core.fmt_time(rng.choice((16, 17, 18)), rng.choice((0, 30)))
            pool.append(
                text.format(
                    given=given,
                    weekday=rng.choice(core.WEEKDAYS_EN[:5]),
                    time=clock,
                    office=office,
                )
            )
        rng.shuffle(pool)
        paragraphs = facts["render"]["case"]
        length = len(render_state(facts)[0])
        if length > high - 15:
            raise GenerationError(f"core text {length} chars exceeds {high}")
        floor = max(length, low + 30)
        target = rng.randint(floor, high - 30) if floor < high - 30 else length
        while length < target and pool:
            spec = {"c": "", "parts": [["filler", pool.pop()]]}
            if rng.random() < 0.2:
                where = rng.randrange(1, len(paragraphs) + 1)
                paragraphs.insert(where, [spec])
                undo = lambda w=where: paragraphs.pop(w)
            else:
                para = paragraphs[rng.randrange(len(paragraphs))]
                where = rng.randrange(1, len(para) + 1)
                para.insert(where, spec)
                undo = lambda p=para, w=where: p.pop(w)
            new_length = len(render_state(facts)[0])
            if new_length > high - 10:
                undo()
                continue
            length = new_length
        if length < low:
            raise GenerationError(f"filler could not reach {low} chars (got {length})")


# ---------------------------------------------------------------- renderer


def _pick(seq: Any, index: int) -> Any:
    return seq[index % len(seq)]


def _short_pick(seq: Any, index: int) -> Any:
    """One of the two shortest templates of ``seq`` (by template length, so the same for every value)."""
    order = sorted(range(len(seq)), key=lambda i: (len(seq[i]), i))
    return seq[order[index % min(2, len(seq))]]


QUANTITY_KINDS = ("minimum", "maximum", "count_at_least")


class _Renderer:
    """Deterministic text for one twin; everything variable comes from the facts."""

    def __init__(self, facts: dict[str, Any]) -> None:
        self.f = facts
        self.pack = PACKS[facts["domain"]]
        self.plan = facts["render"]
        self.needles: list[str] = []
        self.office = _pick(self.pack["offices"], self.plan["office"])
        self.branches = {
            b["bid"]: b for r in facts["requirements"] for b in r["branches"]
        }
        self.case_pick = _short_pick if self.plan["compact"] else _pick

    def _need(self, text: str) -> str:
        self.needles.append(text)
        return text

    def _capitalized(self, text: str, since: int) -> str:
        """``_cap(text)``; needles recorded since ``since`` that open ``text`` follow the capital."""
        for index in range(since, len(self.needles)):
            if text.startswith(self.needles[index]):
                self.needles[index] = _cap(self.needles[index])
        return _cap(text)

    def _d(self, iso: str) -> str:
        return self._need(fmt_date(_day(iso), self.f["date_style"]))

    # -- rule

    def branch_np(self, branch: dict[str, Any]) -> str:
        kind, inst, p = branch["kind"], branch["inst"], branch["params"]
        tpl = p.get("tpl", 0)
        if kind in QUANTITY_KINDS:
            val = fmt_value(inst, p["threshold"], self.f["currency"])
            if inst["style"] == "measure":
                return _pick(_NP_MEASURE[p["op"]], tpl).format(
                    a_noun=_a(inst["noun"]), noun=inst["noun"], val=self._need(val)
                )
            template = _pick(_NP_COUNT[p["op"]], tpl)
            text = template.format(
                val=val, n=f"{p['threshold']:,}", unit=inst["unit"], head=inst["head"]
            )
            self._need(val if "{val}" in template else text)
            return text
        if kind == "date_window":
            return _pick(_NP_WINDOW, tpl).format(
                a_noun=_a(inst["noun"]),
                start=self._d(p["start"]),
                end=self._d(p["end"]),
            )
        if kind == "before_deadline":
            return _pick(_NP_DEADLINE[p["op"]], tpl).format(
                a_thing=_a(inst["thing"]), date=self._d(p["date"])
            )
        if kind == "document_provided":
            return self._need(_art(p["article"], p["doc"]))
        if kind == "signer_role":
            alt = f" or the {p['alts'][0]}" if p["note"] == "inline" else ""
            note = ""
            if p["note"] == "may":
                note = f" (the {p['alts'][0]} may also sign)"
            elif p["note"] == "exclude":
                note = f" ({_a(p['excluded'][0])} or {_a(p['excluded'][1])} does not count)"
            text = _pick(_NP_SIGNER, tpl).format(
                role=self._need(p["role"]), alt=alt, form=p["form"]
            )
            return text + note
        if kind == "category_match":
            return _pick(inst["rule"], tpl).format(
                cat=self._need(p["required"]), a_cat=_a(p["required"])
            )
        if kind == "same_party":
            return _pick(_NP_PARTY, tpl).format(
                a_thing=_a(inst["thing"]), person=self.pack["person"]
            )
        if kind == "valid_on_date":
            return _pick(_NP_CRED, tpl).format(
                a_cred=_art(inst["article"], inst["cred"]),
                anchor=self.f["anchor"]["noun"],
            )
        raise ValueError(kind)

    def req_np(self, req: dict[str, Any]) -> str:
        nps = [self.branch_np(b) for b in req["branches"]]
        return nps[0] if len(nps) == 1 else f"either {nps[0]} or {nps[1]}"

    def _frame(self, layout: str, index: int, requirement: dict[str, Any]) -> str:
        frames = _FRAMES[layout]
        if layout == "paragraph" and index == 0:
            frames = frames[:3]
        subject = _pick(self.pack["subjects"], self.plan["subjects"][index])
        template = _pick(frames, self.plan["frames"][index])
        since = len(self.needles)
        np = self.req_np(requirement)
        cap = self._capitalized(np, since) if "{NP}" in template else np
        return template.format(
            i=index + 1, NP=cap, np=np, subj=subject, Office=_cap(self.office)
        )

    def _rule_blocks(self) -> list[str]:
        layout, plan = self.plan["layout"], self.plan
        items = [
            self._frame(layout, i, r) for i, r in enumerate(self.f["requirements"])
        ]
        recs = []
        for i, rec in enumerate(self.f["recommended"]):
            template = _pick(_REC_FRAMES, plan["rec_frames"][i])
            since = len(self.needles)
            np = self._need(_art(rec["article"], rec["doc"]))
            cap = self._capitalized(np, since) if "{NP}" in template else np
            recs.append(template.format(NP=cap, np=np, Office=_cap(self.office)))
        title, intro = plan["title"], plan["rule_intro"]
        if layout == "numbered":
            return ["\n".join([title, intro] + items + recs)]
        if layout == "paragraph":
            return [title, " ".join([intro] + items + recs)]
        if layout == "checklist":
            return ["\n".join([f"{title} (checklist)"] + items + recs)]
        if layout == "email":
            e = plan["email"]
            header = f"From: {e['officer']} ({e['title']})\nTo: {e['colleague']}\nSubject: {title}"
            opening = _pick(_EMAIL_OPEN, e["open"])
            return [
                header,
                f"Hi {core.given(e['colleague'])},",
                " ".join([opening] + items + recs),
            ]
        note = _pick(_FORM_NOTE, plan["form_note"])
        return ["\n".join([f"Form summary: {title}", note]), " ".join(items + recs)]

    # -- case

    def _fact(self, a: int, bid: str) -> str:
        who = self.f["applicants"][a]
        branch = self.branches[bid]
        attr = who["attrs"][bid]
        kind, inst, p = branch["kind"], branch["inst"], branch["params"]
        tpl = attr.get("tpl", 0)
        pick = self.case_pick
        if kind in QUANTITY_KINDS:
            val = self._need(fmt_value(inst, attr["value"], self.f["currency"]))
            if inst["style"] == "measure":
                return pick(_CL_MEASURE, tpl).format(
                    a_noun=_a(inst["noun"]), noun=inst["noun"], val=val
                )
            return pick(_CL_COUNT, tpl).format(val=val)
        if kind == "date_window":
            return pick(_CL_WINDOW, tpl).format(
                noun=inst["noun"], a_noun=_a(inst["noun"]), date=self._d(attr["date"])
            )
        if kind == "before_deadline":
            return pick(_CL_DEADLINE, tpl).format(
                thing=inst["thing"], office=self.office, date=self._d(attr["date"])
            )
        if kind == "document_provided":
            return f"the {self._need(p['doc'])} {attr['status']}"
        if kind == "signer_role":
            return pick(_CL_SIGNER, tpl).format(
                form=p["form"],
                signer=self._need(attr["signer"]),
                other=attr["other"],
                sname=attr["sname"],
                oname=attr["oname"],
            )
        if kind == "category_match":
            prev, cur = attr["prev"], attr["cur"]
            bank = inst["swap"] if prev else inst["single"]
            return pick(bank, tpl).format(
                prev=prev,
                cur=self._need(cur),
                a_prev=_a(prev) if prev else "",
                a_cur=_a(cur),
                given=core.given(who["name"]),
                month=attr["month"],
                date=attr["date"],
            )
        if kind == "same_party":
            return pick(_CL_PARTY, tpl).format(
                thing=inst["thing"], holder=self._need(attr["holder"])
            )
        if kind == "valid_on_date":
            if attr["start"]:
                return pick(_CL_CRED_BOTH, tpl).format(
                    cred=inst["cred"],
                    start=self._d(attr["start"]),
                    until=self._d(attr["until"]),
                )
            return pick(_CL_CRED_UNTIL, tpl).format(
                cred=inst["cred"], until=self._d(attr["until"])
            )
        raise ValueError(kind)

    def _part(self, spec: list[Any]) -> tuple[str, bool]:
        """Rendered text of one sentence part and whether it is a bare clause."""
        f, plan, kind = self.f, self.plan, spec[0]
        if kind == "fact":
            return self._fact(spec[1], spec[2]), True
        if kind == "rec":
            rec = next(r for r in f["recommended"] if r["id"] == spec[2])
            return (
                f"the {self._need(rec['doc'])} {f['applicants'][spec[1]]['rec'][spec[2]]['status']}",
                True,
            )
        if kind == "extra":
            extra = f["applicants"][spec[1]]["extras"][spec[2]]
            return f"the {extra['doc']} {extra['status']}", True
        if kind == "anchor":
            clause = _pick(self.pack["anchor"]["clauses"], f["anchor"]["tpl"])
            return clause.format(date=self._d(f["anchor"]["date"])), True
        if kind == "intro":
            program = _pick(self.pack["programs"], plan["program"])
            name = self._need(f["applicants"][0]["name"])
            return (
                _pick(self.pack["intros"], plan["intro"]).format(
                    name=name, program=program
                ),
                False,
            )
        if kind == "email_case":
            case = plan["email"]["case"]
            if f["form"] == "which_applicant":
                return _pick(_EMAIL_CASE_B, case), False
            return (
                _pick(_EMAIL_CASE, case).format(
                    given=core.given(f["applicants"][0]["name"])
                ),
                False,
            )
        if kind == "preamble":
            text = _pick(_PREAMBLE_B, plan["preamble"]).format(
                count=_COUNT_WORDS[len(f["applicants"])],
                persons=self.pack["persons"],
                office_cap=_cap(self.office),
            )
            return _cap(text), False
        if kind == "lead":
            name = self._need(f["applicants"][spec[1]]["name"])
            return (
                self.case_pick(_LEADS, plan["leads"][spec[1]]).format(name=name),
                False,
            )
        if kind == "relation":
            who = f["applicants"][spec[1]]
            rel = who["relative"]
            return (
                _pick(_RELATION, rel["tpl"]).format(
                    given=core.given(who["name"]),
                    relation=rel["relation"],
                    other=rel["name"],
                ),
                False,
            )
        if kind == "filler":
            return spec[1], False
        raise ValueError(kind)

    def _sentence(self, sentence: dict[str, Any]) -> str:
        since = len(self.needles)
        parts = [self._part(p) for p in sentence["parts"]]
        if len(parts) == 1 and not parts[0][1]:
            return parts[0][0]
        body = "; ".join(text for text, _ in parts)
        return (
            sentence["c"] + body if sentence["c"] else self._capitalized(body, since)
        ) + "."

    def state(self) -> str:
        blocks = self._rule_blocks()
        layout = self.plan["layout"]
        if layout == "form":
            blocks.append(
                _pick(_FORM_CASE, self.plan["form_case"]).format(office=self.office)
            )
        for para in self.plan["case"]:
            blocks.append(" ".join(self._sentence(s) for s in para))
        if layout == "email":
            e = self.plan["email"]
            blocks.append(
                f"{_pick(_EMAIL_CLOSE, e['close'])}\n{core.given(e['officer'])}"
            )
        return "\n\n".join(blocks)


def render_state(facts: dict[str, Any]) -> tuple[str, list[str]]:
    """The rendered state of one twin and the decisive strings it must contain."""
    renderer = _Renderer(facts)
    return renderer.state(), renderer.needles


def question_text(facts: dict[str, Any]) -> str:
    pack = PACKS[facts["domain"]]
    plan = facts["render"]
    name = facts["applicants"][0]["name"]
    return plan["question"].format(
        name=name,
        given=core.given(name),
        request=pack["request"],
        office=_pick(pack["offices"], plan["office"]),
        persons=pack["persons"],
        person=pack["person"],
        a_person=_a(pack["person"]),
    )


def choice_texts(facts: dict[str, Any]) -> tuple[str, ...]:
    form = facts["form"]
    if form == "noul":
        return ()
    if form == "score":
        return SCORE_LEVELS
    catch_all = facts["render"]["catch_all"]
    if form == "which_applicant":
        return tuple(a["name"] for a in facts["applicants"]) + (catch_all,)
    return tuple(r["label"] for r in facts["requirements"]) + (catch_all,)


# ---------------------------------------------------------------- oracle


def branch_met(
    branch: dict[str, Any], attr: dict[str, Any], who: dict[str, Any], anchor: str
) -> bool:
    kind, p = branch["kind"], branch["params"]
    if kind in QUANTITY_KINDS:
        return not _fails(p["op"], attr["value"], p["threshold"])
    if kind == "date_window":
        return p["start"] <= attr["date"] <= p["end"]
    if kind == "before_deadline":
        return (
            attr["date"] <= p["date"] if p["op"] == "<=" else attr["date"] < p["date"]
        )
    if kind == "document_provided":
        return bool(attr["provided"])
    if kind == "signer_role":
        return attr["signer"] == p["role"] or attr["signer"] in p["alts"]
    if kind == "category_match":
        return attr["cur"] == p["required"]
    if kind == "same_party":
        return attr["holder"] == who["name"]
    if kind == "valid_on_date":
        return (attr["start"] is None or attr["start"] <= anchor) and anchor <= attr[
            "until"
        ]
    raise ValueError(kind)


def failing(facts: dict[str, Any], a: int) -> list[str]:
    """Ids of the required conditions applicant ``a`` fails, in rule order."""
    who = facts["applicants"][a]
    anchor = facts["anchor"]["date"]
    return [
        r["id"]
        for r in facts["requirements"]
        if not any(
            branch_met(b, who["attrs"][b["bid"]], who, anchor) for b in r["branches"]
        )
    ]


def _missing_recommended(facts: dict[str, Any], a: int) -> int:
    return sum(
        1
        for rec in facts["recommended"]
        if not facts["applicants"][a]["rec"][rec["id"]]["provided"]
    )


def solve(facts: dict[str, Any]) -> int:
    """The oracle label in the Item encoding."""
    form = facts["form"]
    if form == "which_applicant":
        ok = [a for a in range(len(facts["applicants"])) if not failing(facts, a)]
        if len(ok) > 1:
            raise GenerationError("more than one applicant qualifies")
        return ok[0] if ok else len(facts["applicants"])
    fails = failing(facts, 0)
    if form == "noul":
        return 0 if fails else 1
    if form == "score":
        if fails:
            return 0
        missing = _missing_recommended(facts, 0)
        return 3 if missing == 0 else 2 if missing == 1 else 1
    if not fails:
        return len(facts["requirements"])
    if form == "which_unmet" and len(fails) > 1:
        raise GenerationError(
            "more than one unmet requirement for a single-answer question"
        )
    return [r["id"] for r in facts["requirements"]].index(fails[0])


def near_threshold(facts: dict[str, Any]) -> bool:
    """Whether any decisive value sits within 5% (numbers) or 3 days (dates) of its boundary."""
    anchor = _day(facts["anchor"]["date"])
    for req in facts["requirements"]:
        for branch in req["branches"]:
            kind, p = branch["kind"], branch["params"]
            for who in facts["applicants"]:
                attr = who["attrs"][branch["bid"]]
                if kind in QUANTITY_KINDS:
                    gap = (
                        abs(attr["value"] - p["threshold"]) / p["threshold"] <= NEAR_REL
                    )
                elif kind == "date_window":
                    day = _day(attr["date"])
                    gap = (
                        min(
                            abs((day - _day(p["start"])).days),
                            abs((day - _day(p["end"])).days),
                        )
                        <= NEAR_DAYS
                    )
                elif kind == "before_deadline":
                    gap = abs((_day(attr["date"]) - _day(p["date"])).days) <= NEAR_DAYS
                elif kind == "valid_on_date":
                    days = [abs((_day(attr["until"]) - anchor).days)]
                    if attr["start"]:
                        days.append(abs((_day(attr["start"]) - anchor).days))
                    gap = min(days) <= NEAR_DAYS
                else:
                    gap = False
                if gap:
                    return True
    return False


# ---------------------------------------------------------------- independent re-check

_RC_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
}
_RC_MONTHS.update({name[:3]: number for name, number in list(_RC_MONTHS.items())})
_RC_DATE = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})\b|\b([A-Z][a-z]{2,8}) (\d{1,2}), (\d{4})\b"
    r"|\b(\d{1,2}) ([A-Z][a-z]{2,8}) (\d{4})\b"
)
_RC_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_RC_NAME = re.compile(r"\b[A-Z][a-zA-Z'\-]+ [A-Z][a-zA-Z'\-]+\b")
_RC_CMP = (
    ("no more than", "<="),
    ("no less than", ">="),
    ("no fewer than", ">="),
    ("at least", ">="),
    ("at most", "<="),
    ("or more", ">="),
    ("or less", "<="),
    ("or fewer", "<="),
    ("does not exceed", "<="),
    ("a minimum", ">="),
    ("a maximum", "<="),
    ("more than", ">"),
    ("less than", "<"),
    ("fewer than", "<"),
    ("above", ">"),
    ("below", "<"),
    ("under", "<"),
)
_RC_HAS = (
    "attached",
    "on file",
    "uploaded",
    "came in",
    "handed in",
    "received last",
    "in the folder",
    "enclosed",
)
_RC_LACKS = (
    "pending",
    "not ",
    "never",
    "due next",
    "left out",
    "being prepared",
    "only arrive",
    "missing",
    "requested but",
)
_RC_CATCH_ALL = ("all requirements", "every requirement", "none;", "none of", "no ")


def _rc_dates(text: str) -> list[date]:
    out = []
    for m in _RC_DATE.finditer(text):
        if m.group(1):
            out.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        month, day, year = (
            (m.group(4), m.group(5), m.group(6))
            if m.group(4)
            else (m.group(8), m.group(7), m.group(9))
        )
        number = _RC_MONTHS.get(month.lower())
        if number:
            out.append(date(int(year), number, int(day)))
    return out


def _rc_numbers(text: str) -> list[Decimal]:
    for m in _RC_DATE.finditer(text):
        text = text.replace(m.group(0), " ")
    return [
        Decimal(m.group(0).rstrip(",").replace(",", ""))
        for m in _RC_NUMBER.finditer(text)
    ]


def _rc_word(term: str, text: str) -> re.Match[str] | None:
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text)


def _rc_terms(text: str, terms: list[str]) -> list[str]:
    """Non-overlapping occurrences of ``terms`` (longest first), in text order."""
    taken: list[tuple[int, int, str]] = []
    for term in sorted(terms, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text):
            if all(m.end() <= s or m.start() >= e for s, e, _ in taken):
                taken.append((m.start(), m.end(), term))
    return [term for _, _, term in sorted(taken)]


def _rc_comparator(text: str) -> str:
    hits = []
    for phrase, op in _RC_CMP:
        hits += [(m.start(), m.end(), op) for m in re.finditer(rf"\b{phrase}\b", text)]
    kept = [
        h
        for h in hits
        if not any(o != h and o[0] <= h[0] and h[1] <= o[1] for o in hits)
    ]
    if len(kept) != 1:
        raise RecheckError(f"comparator unreadable in {text!r}")
    return kept[0][2]


def _rc_holds(op: str, value: Any, limit: Any) -> bool:
    return {
        ">=": value >= limit,
        ">": value > limit,
        "<=": value <= limit,
        "<": value < limit,
    }[op]


def _rc_frame(
    template: str, subjects: tuple[str, ...], offices: tuple[str, ...]
) -> re.Pattern[str]:
    import string

    if template.startswith("{i}. "):
        return re.compile(r"(?m)^\d+\. (?P<np>.+)\.$")
    np_first = template.startswith("{NP}")
    pattern = ""
    for literal, field, _, _ in string.Formatter().parse(template):
        pattern += re.escape(literal)
        if field in ("np", "NP"):
            pattern += r"(?P<np>[A-Z][^.\n]*?)" if np_first else r"(?P<np>.+?)"
        elif field == "i":
            pattern += r"\d+"
        elif field == "subj":
            pattern += "(?:" + "|".join(re.escape(s) for s in subjects) + ")"
        elif field == "Office":
            pattern += "(?:" + "|".join(re.escape(_cap(o)) for o in offices) + ")"
    if pattern.endswith(r"\."):
        pattern = pattern[:-2] + r"\.(?=\s|$)"
    if np_first:
        pattern = r"(?:(?<=\. )|(?<=\n)|^)" + pattern
    return re.compile(pattern)


class Recheck:
    """Independent reader of a rendered F3 state.

    It knows the domain vocabulary and the rule frames, never the world facts:
    it finds the rule sentences, reads each condition's comparator, limits and
    roles from the words, reads the case clauses, and recomputes the answer
    with its own date, number and status parsing.
    """

    _cache: dict[str, "Recheck"] = {}

    @classmethod
    def for_domain(cls, domain: str) -> "Recheck":
        if domain not in cls._cache:
            cls._cache[domain] = cls(domain)
        return cls._cache[domain]

    def __init__(self, domain: str) -> None:
        pack = PACKS[domain]
        self.pack = pack
        self.frames = [
            (_rc_frame(t, pack["subjects"], pack["offices"]), False)
            for layout in LAYOUTS
            for t in _FRAMES[layout]
        ]
        self.frames += [
            (_rc_frame(t, pack["subjects"], pack["offices"]), True) for t in _REC_FRAMES
        ]
        self.doc_nps = {_art(article, noun): noun for article, noun in pack["docs"]}
        self.docs = [noun for _, noun in pack["docs"]]
        name = r"(?P<name>[A-Z][a-zA-Z'\-]+ [A-Z][a-zA-Z'\-]+)"
        programs = "(?:" + "|".join(re.escape(p) for p in pack["programs"]) + ")"
        self.intros = [
            re.compile(
                re.escape(t)
                .replace(r"\{name\}", name)
                .replace(r"\{program\}", programs)
            )
            for t in pack["intros"]
        ]
        self.anchor_noun = pack["anchor"]["noun"]

    # -- rule

    def _read_np(self, np: str) -> dict[str, Any]:
        readings = [
            r
            for r in (
                self._np_doc(np),
                self._np_quantity(np),
                self._np_window(np),
                self._np_deadline(np),
                self._np_signer(np),
                self._np_category(np),
                self._np_party(np),
                self._np_cred(np),
            )
            if r is not None
        ]
        if len(readings) != 1:
            raise RecheckError(f"{len(readings)} readings of {np!r}")
        return readings[0]

    def _np_doc(self, np: str) -> dict[str, Any] | None:
        doc = self.doc_nps.get(np)
        return (
            None
            if doc is None
            else {"kind": "document_provided", "key": doc, "label": _cap(doc)}
        )

    def _np_quantity(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["quantities"]:
            if not _rc_word(inst["noun"], np):
                continue
            numbers = _rc_numbers(np)
            if len(numbers) != 1:
                raise RecheckError(f"threshold unreadable in {np!r}")
            return {
                "kind": "quantity",
                "key": inst["key"],
                "label": inst["label"],
                "style": inst["style"],
                "noun": inst["noun"],
                "op": _rc_comparator(np),
                "limit": numbers[0],
            }
        return None

    def _np_window(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["windows"]:
            if not _rc_word(inst["noun"], np):
                continue
            days = _rc_dates(np)
            inclusive = any(
                k in np
                for k in (
                    "through",
                    "no earlier than",
                    "both dates included",
                    "on or between",
                )
            )
            if len(days) != 2 or not inclusive:
                raise RecheckError(f"window unreadable in {np!r}")
            return {
                "kind": "date_window",
                "key": inst["key"],
                "label": inst["label"],
                "noun": inst["noun"],
                "start": days[0],
                "end": days[1],
            }
        return None

    def _np_deadline(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["deadlines"]:
            if not _rc_word(inst["thing"], np):
                continue
            days = _rc_dates(np)
            if len(days) != 1:
                raise RecheckError(f"deadline unreadable in {np!r}")
            if (
                "on or before" in np
                or "no later than" in np
                or re.search(r"\bby\b", np)
            ):
                op = "<="
            elif re.search(r"\bbefore\b", np):
                op = "<"
            else:
                raise RecheckError(f"deadline wording unknown in {np!r}")
            return {
                "kind": "before_deadline",
                "key": inst["key"],
                "label": inst["label"],
                "thing": inst["thing"],
                "op": op,
                "date": days[0],
            }
        return None

    def _roles(self, inst: dict[str, Any]) -> list[str]:
        return [inst["role"], *inst["near"], *inst["wrong"]]

    def _np_signer(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["signers"]:
            if not _rc_word(inst["form"], np) or not re.search(r"\bsign", np):
                continue
            allowed: list[str] = []
            for paren in re.findall(r"\(([^)]*)\)", np):
                if "may also sign" in paren:
                    allowed += _rc_terms(paren, self._roles(inst))
                elif "does not count" not in paren:
                    raise RecheckError(f"unknown signer note in {np!r}")
            main = re.sub(r"\s*\([^)]*\)", "", np).replace(f"the {inst['form']}", "")
            named = _rc_terms(main, self._roles(inst))
            if not named or (len(named) > 1 and " or the " not in main):
                raise RecheckError(f"signer roles unreadable in {np!r}")
            return {
                "kind": "signer_role",
                "key": inst["key"],
                "label": inst["label"],
                "form": inst["form"],
                "allowed": named + allowed,
                "roles": self._roles(inst),
            }
        return None

    def _np_category(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["categories"]:
            found = set(_rc_terms(np, list(inst["pair"])))
            if not found:
                continue
            if len(found) != 1:
                raise RecheckError(f"two categories in {np!r}")
            return {
                "kind": "category_match",
                "key": inst["key"],
                "label": inst["label"],
                "pair": list(inst["pair"]),
                "required": found.pop(),
            }
        return None

    def _np_party(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["parties"]:
            if _rc_word(inst["thing"], np) and (
                "own name" in np or "named holder is the" in np
            ):
                return {
                    "kind": "same_party",
                    "key": inst["key"],
                    "label": inst["label"],
                    "thing": inst["thing"],
                }
        return None

    def _np_cred(self, np: str) -> dict[str, Any] | None:
        for inst in self.pack["creds"]:
            if _rc_word(inst["cred"], np) and _rc_word(self.anchor_noun, np):
                return {
                    "kind": "valid_on_date",
                    "key": inst["key"],
                    "label": inst["label"],
                    "cred": inst["cred"],
                }
        return None

    def _requirement(self, np: str) -> dict[str, Any]:
        np = _lc(np.strip())
        if not np.startswith("either "):
            branch = self._read_np(np)
            return {"branches": [branch], "label": branch["label"]}
        body = np[len("either ") :]
        splits = []
        for m in re.finditer(" or ", body):
            try:
                splits.append(
                    (self._read_np(body[: m.start()]), self._read_np(body[m.end() :]))
                )
            except RecheckError:
                continue
        if len(splits) != 1:
            raise RecheckError(f"either/or unreadable: {np!r}")
        first, second = splits[0]
        return {
            "branches": [first, second],
            "label": f"{first['label']} or {_lc(second['label'])}",
        }

    def read_rule(
        self, state: str
    ) -> tuple[list[dict[str, Any]], list[str], list[tuple[int, int]]]:
        hits = []
        for regex, recommended in self.frames:
            hits += [
                (m.start(), m.end(), m.group("np"), recommended)
                for m in regex.finditer(state)
            ]
        hits.sort()
        for left, right in zip(hits, hits[1:]):
            if right[0] < left[1]:
                raise RecheckError("overlapping rule sentences")
        requirements, recommended = [], []
        for _, _, np, is_rec in hits:
            if is_rec:
                doc = self.doc_nps.get(_lc(np.strip()))
                if doc is None:
                    raise RecheckError(f"unknown recommended item {np!r}")
                recommended.append(doc)
            else:
                requirements.append(self._requirement(np))
        return requirements, recommended, [(s, e) for s, e, _, _ in hits]

    # -- case

    @staticmethod
    def _clauses(text: str) -> list[tuple[int, str]]:
        out, start = [], 0
        for m in re.finditer(r"\.(?=\s|$)|;(?=\s)|\n", text):
            piece = text[start : m.start()]
            if piece.strip():
                out.append((start + len(piece) - len(piece.lstrip()), piece.strip()))
            start = m.end()
        if text[start:].strip():
            out.append((start, text[start:].strip()))
        return out

    def _fact(self, branch: dict[str, Any], clause: str) -> Any:
        """The branch's case fact if ``clause`` states it, else None."""
        kind = branch["kind"]
        if kind == "quantity":
            if branch["style"] == "count":
                if not re.search(rf"\d[\d,]* {re.escape(branch['noun'])}\b", clause):
                    return None
            elif not _rc_word(branch["noun"], clause):
                return None
            numbers = _rc_numbers(clause)
            if len(numbers) != 1:
                raise RecheckError(f"value unreadable in {clause!r}")
            return numbers[0]
        if kind in ("date_window", "before_deadline"):
            if not _rc_word(
                branch["noun"] if kind == "date_window" else branch["thing"], clause
            ):
                return None
            days = _rc_dates(clause)
            if len(days) != 1:
                raise RecheckError(f"date unreadable in {clause!r}")
            return days[0]
        if kind == "document_provided":
            m = re.search(rf"\b[Tt]he {re.escape(branch['key'])}(?![\w-])", clause)
            return None if m is None else self._provided(clause[m.end() :])
        if kind == "signer_role":
            if not _rc_word(branch["form"], clause) or "sign" not in clause:
                return None
            roles = "|".join(
                re.escape(r) for r in sorted(branch["roles"], key=len, reverse=True)
            )
            for pattern in (
                rf"signed by (?:[A-Z][a-zA-Z'\-]+ [A-Z][a-zA-Z'\-]+, )?the ({roles})(?![\w-])",
                rf"signature on the .+? is that of the ({roles})(?![\w-])",
                rf"\b[Tt]he ({roles}) signed the\b",
            ):
                m = re.search(pattern, clause)
                if m:
                    return m.group(1)
            raise RecheckError(f"signer unreadable in {clause!r}")
        if kind == "category_match":
            found = _rc_terms(clause, branch["pair"])
            if not found:
                return None
            if len(found) == 1:
                return found[0]
            for marker in (" to ", " now ", " until "):
                cut = clause.find(marker)
                if cut >= 0 and (marker != " to " or " from " in clause):
                    after = _rc_terms(clause[cut:], branch["pair"])
                    if after:
                        return after[0]
            raise RecheckError(f"current category unreadable in {clause!r}")
        if kind == "same_party":
            if not _rc_word(branch["thing"], clause):
                return None
            names = _RC_NAME.findall(clause)
            if len(names) != 1:
                raise RecheckError(f"holder unreadable in {clause!r}")
            return names[0]
        if kind == "valid_on_date":
            if not _rc_word(branch["cred"], clause):
                return None
            days = _rc_dates(clause)
            if len(days) == 2:
                return days[0], days[1]
            if len(days) == 1 and "through" in clause:
                return None, days[0]
            raise RecheckError(f"validity unreadable in {clause!r}")
        raise ValueError(kind)

    @staticmethod
    def _provided(rest: str) -> bool:
        has = any(k in rest for k in _RC_HAS)
        lacks = any(k in rest for k in _RC_LACKS)
        if has == lacks:
            raise RecheckError(f"document status unreadable: {rest!r}")
        return has

    def _one(self, branch: dict[str, Any], clauses: list[tuple[int, str]]) -> Any:
        found = [
            fact
            for fact in (self._fact(branch, c) for _, c in clauses)
            if fact is not None
        ]
        if len(found) != 1:
            raise RecheckError(f"{branch['key']}: {len(found)} case statements")
        return found[0]

    def _anchor(self, clauses: list[tuple[int, str]]) -> date:
        found = [_rc_dates(c) for _, c in clauses if _rc_word(self.anchor_noun, c)]
        if len(found) != 1 or len(found[0]) != 1:
            raise RecheckError("anchor date unreadable")
        return found[0][0]

    def _met(
        self, branch: dict[str, Any], fact: Any, name: str, anchor: date | None
    ) -> bool:
        kind = branch["kind"]
        if kind == "quantity":
            return _rc_holds(branch["op"], fact, branch["limit"])
        if kind == "date_window":
            return branch["start"] <= fact <= branch["end"]
        if kind == "before_deadline":
            return _rc_holds(branch["op"], fact, branch["date"])
        if kind == "document_provided":
            return fact
        if kind == "signer_role":
            return fact in branch["allowed"]
        if kind == "category_match":
            return fact == branch["required"]
        if kind == "same_party":
            return fact == name
        start, until = fact
        return (start is None or start <= anchor) and anchor <= until

    def _unmet(
        self,
        requirements: list[dict[str, Any]],
        clauses: list[tuple[int, str]],
        name: str,
        anchor: date | None,
    ) -> list[int]:
        return [
            index
            for index, req in enumerate(requirements)
            if not any(
                self._met(b, self._one(b, clauses), name, anchor)
                for b in req["branches"]
            )
        ]

    def solve(
        self, state: str, instructions: str, choices: tuple[str, ...], task_type: str
    ) -> int:
        requirements, recommended, spans = self.read_rule(state)
        case = list(state)
        for start, end in spans:
            case[start:end] = " " * (end - start)
        clauses = self._clauses("".join(case))
        needs_anchor = any(
            b["kind"] == "valid_on_date" for r in requirements for b in r["branches"]
        )
        anchor = self._anchor(clauses) if needs_anchor else None
        catch = [
            i for i, c in enumerate(choices) if c.lower().startswith(_RC_CATCH_ALL)
        ]
        names = [c for c in choices if _RC_NAME.fullmatch(c)]
        if task_type == "choice" and names:
            text = "".join(case)
            starts = sorted((text.find(n), n) for n in names)
            if any(pos < 0 for pos, _ in starts) or len(catch) != 1:
                raise RecheckError("applicant sections unreadable")
            qualified = []
            for index, (pos, name) in enumerate(starts):
                end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
                section = [c for c in clauses if pos <= c[0] < end]
                if not self._unmet(requirements, section, name, anchor):
                    qualified.append(name)
            if len(qualified) > 1:
                raise RecheckError("several applicants qualify")
            return choices.index(qualified[0]) if qualified else catch[0]
        found = [
            m.group("name") for rx in self.intros for m in rx.finditer("".join(case))
        ]
        if len(found) != 1:
            raise RecheckError("applicant name unreadable")
        unmet = self._unmet(requirements, clauses, found[0], anchor)
        if task_type == "noul":
            return 0 if unmet else 1
        if task_type == "score":
            if unmet:
                return 0
            missing = 0
            for doc in recommended:
                if not self._one({"kind": "document_provided", "key": doc}, clauses):
                    missing += 1
            return 3 if missing == 0 else 2 if missing == 1 else 1
        if len(catch) != 1:
            raise RecheckError("catch-all option unreadable")
        if not unmet:
            return catch[0]
        if len(unmet) > 1 and "first" not in instructions:
            raise RecheckError(
                "several unmet requirements for a single-answer question"
            )
        label = requirements[unmet[0]]["label"]
        if label not in choices:
            raise RecheckError(f"requirement label {label!r} not offered")
        return choices.index(label)


# ---------------------------------------------------------------- family API


class TwinError(GenerationError):
    """A twin invariant failed; this is a generator bug and is never retried."""


def negation_count(text: str) -> int:
    lower = text.lower()
    words = re.findall(
        r"\b(?:not|no|never|without|missing|none|neither|nor|unable|lacks)\b", lower
    )
    return len(words) + lower.count("n't")


def masked_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[#$]", core.mask(text))


def twin_facts(facts: dict[str, Any], updates: list[tuple[str, Any]]) -> dict[str, Any]:
    positive = copy.deepcopy(facts)
    for path, value in updates:
        _set(positive, path, copy.deepcopy(value))
    positive["twin"] = "positive"
    return positive


def _items(negative: dict[str, Any], positive: dict[str, Any]) -> list[Item]:
    checker = Recheck.for_domain(negative["domain"])
    items: list[Item] = []
    rendered = []
    for facts in (negative, positive):
        state, needles = render_state(facts)
        core.require_present(state, needles)
        rendered.append(state)
        instructions = question_text(facts)
        choices = choice_texts(facts)
        task = facts["interface"]
        gold = solve(facts)
        recheck = checker.solve(state, instructions, choices, task)
        if gold != recheck:
            raise core.RecheckMismatch(
                f"oracle {gold} != recheck {recheck} ({facts['domain']}/{facts['twin']})"
            )
        plan = facts["render"]
        star = facts["applicants"][facts["star"]]
        failing_ids = failing(facts, facts["star"])
        kinds = {
            b["bid"]: b["kind"] for r in facts["requirements"] for b in r["branches"]
        }
        failed_kinds = (
            [kinds[path.split(".")[-1]] for path in facts["designated"]]
            if failing_ids
            else []
        )
        refs = {"catch_all": len(choices) - 1} if task == "choice" else {}
        meta = {
            "interface": task,
            "length": facts["length"],
            "domain": facts["domain"],
            "form": facts["form"],
            "failing": failed_kinds,
            "twin": facts["twin"],
            "n_required": len(facts["requirements"]),
            "n_optional": len(facts["recommended"]),
            "near_threshold": near_threshold(facts),
            "layout": plan["layout"],
            "question_id": plan["question_id"],
            "n_fail": facts["n_fail"],
            "either_or": sum(
                1 for r in facts["requirements"] if len(r["branches"]) > 1
            ),
            "failing_ids": failing_ids,
            "applicants": len(facts["applicants"]),
            "star_name": star["name"] if facts["form"] == "which_applicant" else "",
        }
        subtype = facts["fail_kinds"][0] if facts["n_fail"] == 1 else "two_unmet"
        items.append(
            Item(
                task_type=task,
                state=state,
                instructions=instructions,
                choices=choices,
                gold=gold,
                recheck=recheck,
                kind=facts["domain"],
                subtype=subtype,
                variant=f"{plan['layout']}/{plan['question_id']}",
                facts=facts,
                option_refs=refs,
                meta=meta,
            )
        )
    for item in items:
        item.check()
    neg, pos = items
    if neg.gold == pos.gold or (
        neg.task_type in ("noul", "score") and (neg.gold != 0 or pos.gold == 0)
    ):
        raise TwinError(f"label does not flip ({neg.gold} -> {pos.gold})")
    if neg.choices != pos.choices or neg.instructions != pos.instructions:
        raise TwinError("twins differ in question or options")
    if len(rendered[0]) != len(rendered[1]):
        raise TwinError("twin lengths differ")
    if sorted(masked_tokens(rendered[0])) != sorted(masked_tokens(rendered[1])):
        raise TwinError("twin masked bags differ")
    if negation_count(rendered[0]) != negation_count(rendered[1]):
        raise TwinError("twin negation counts differ")
    return items


def design(seed_key: str, interface: str) -> dict[str, Any]:
    """The world's label design, drawn once per seed so that rejected draws cannot skew its shares."""
    rng = rng_for(seed_key, "design")
    n_fail = 2 if rng.random() < TWO_FAIL_SHARE else 1
    if interface in ("noul", "score"):
        form = interface
    else:
        form = rng.choice(("which_unmet", "which_applicant"))
        if form == "which_unmet" and (n_fail == 2 or rng.random() < 0.3):
            form = "which_unmet_first"
    return {
        "n_fail": n_fail,
        "form": form,
        "fail_kinds": rng.sample(REQ_KINDS, n_fail),
        "target_level": rng.choice((1, 2, 3)) if form == "score" else None,
    }


def build_world(
    seed_key: str, kind: str, interface: str, length: str
) -> tuple[dict[str, Any], list[tuple[str, Any]], int]:
    """The negative twin's facts, the positive-twin updates and the number of rejected draws."""
    if interface not in supported(kind) or length not in LENGTH_RANGES:
        raise ValueError(f"unsupported {kind}/{interface}/{length}")
    plan = design(seed_key, interface)
    last: GenerationError | None = None
    for attempt in range(MAX_ATTEMPTS):
        rng = rng_for(seed_key, "attempt", attempt)
        try:
            negative, updates = _Builder(rng, kind, interface, length, plan).build()
        except (RecheckError, TwinError):
            raise
        except GenerationError as exc:
            last = exc
            continue
        return negative, updates, attempt
    raise GenerationError(
        f"{kind}/{interface}/{length}: no world after {MAX_ATTEMPTS} attempts ({last})"
    )


def make_group(seed_key: str, kind: str, interface: str, length: str) -> list[Item]:
    """The NEGATIVE row and its POSITIVE twin for one world (a pure function of the arguments)."""
    negative, updates, _ = build_world(seed_key, kind, interface, length)
    return _items(negative, twin_facts(negative, updates))


# ---------------------------------------------------------------- pack validation


def pack_vocabulary(domain: str) -> list[str]:
    """Every phrase the re-check uses to locate a condition or a case fact."""
    pack = PACKS[domain]
    vocab = [q["noun"] for q in pack["quantities"]]
    vocab += [w["noun"] for w in pack["windows"]] + [
        d["thing"] for d in pack["deadlines"]
    ]
    vocab += [noun for _, noun in pack["docs"]] + [s["form"] for s in pack["signers"]]
    vocab += [p["thing"] for p in pack["parties"]] + [c["cred"] for c in pack["creds"]]
    vocab += [pack["anchor"]["noun"]]
    return vocab


def validate_pack(domain: str) -> list[str]:
    """Problems that would let the re-check misread a state of this domain (empty when sound)."""
    pack = PACKS[domain]
    problems = []
    vocab = pack_vocabulary(domain)
    roles = [r for s in pack["signers"] for r in (s["role"], *s["near"], *s["wrong"])]
    cats = [c for cat in pack["categories"] for c in cat["pair"]]
    for term in vocab:
        for other in vocab:
            if term != other and _rc_word(term, other):
                problems.append(f"{term!r} occurs inside {other!r}")
        for word in roles + cats:
            if _rc_word(word, term):
                problems.append(f"role or category {word!r} occurs inside {term!r}")
    free_text = (
        list(pack["filler"])
        + list(pack["titles"])
        + list(pack["intros"])
        + list(pack["programs"])
    )
    free_text += (
        list(pack["relations"]) + list(pack["subjects"]) + list(pack["offices"])
    )
    free_text += (
        [t for t in pack["anchor"]["clauses"]] + list(_GENERIC_FILLER) + list(_RELATION)
    )
    free_text += (
        list(_LEADS)
        + list(_PREAMBLE_B)
        + list(_EMAIL_OPEN)
        + list(_EMAIL_CASE)
        + list(_RULE_INTROS)
    )
    free_text += (
        list(_PROVIDED) + list(_NOT_PROVIDED) + list(_FORM_NOTE) + list(_FORM_CASE)
    )
    for text in free_text:
        lower = text.lower()
        for term in vocab + roles + cats:
            if term == pack["anchor"]["noun"] and text in pack["anchor"]["clauses"]:
                continue
            if _rc_word(term.lower(), lower):
                problems.append(f"{term!r} appears in free text {text!r}")
    for q in pack["quantities"]:
        if q["style"] == "count" and q["kind"] == "maximum" and not q["head"]:
            problems.append(f"count maximum {q['key']} needs a head phrase")
        for op in _QUANTITY_OPS[q["kind"]]:
            if not twin_grid(q, op):
                problems.append(
                    f"quantity {q['key']} has no twin-capable threshold for {op!r}"
                )
    available = {
        kind: bool(values)
        for kind, values in (
            ("minimum", [q for q in pack["quantities"] if q["kind"] == "minimum"]),
            ("maximum", [q for q in pack["quantities"] if q["kind"] == "maximum"]),
            (
                "count_at_least",
                [q for q in pack["quantities"] if q["kind"] == "count_at_least"],
            ),
            ("date_window", pack["windows"]),
            ("before_deadline", pack["deadlines"]),
            ("document_provided", pack["docs"]),
            ("signer_role", pack["signers"]),
            ("category_match", pack["categories"]),
            ("same_party", pack["parties"]),
            ("valid_on_date", pack["creds"]),
        )
    }
    problems += [f"no {kind} condition" for kind in REQ_KINDS if not available[kind]]
    return problems


_QUANTITY_OPS = {
    "minimum": (">=", ">"),
    "maximum": ("<=", "<"),
    "count_at_least": (">=",),
}


# ---------------------------------------------------------------- lexical self-check

SELF_CHECK_WORLDS = 1000
SELF_CHECK_TOLERANCE = 0.05
HEURISTIC_CEILING = 0.60


def probe_tokens(text: str, ngram: int = 1) -> list[str]:
    """Whitespace tokens of ``core.mask(text)`` with edge punctuation stripped, plus n-grams up to ``ngram``."""
    words = [
        word
        for word in (raw.strip(string.punctuation) for raw in core.mask(text).split())
        if word
    ]
    grams = list(words)
    for size in range(2, ngram + 1):
        grams += [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
    return grams


def naive_bayes_cv(rows: list[tuple[str, list[str], int]], folds: int = 5) -> float:
    """Group-``folds``-fold accuracy of a multinomial naive Bayes (add-one) on (group, tokens, label) rows."""
    fold_of = {group: int(core.sha(group), 16) % folds for group, _, _ in rows}
    correct = 0
    for fold in range(folds):
        counts: tuple[dict[str, int], dict[str, int]] = ({}, {})
        totals, priors = [0, 0], [0, 0]
        for group, tokens, label in rows:
            if fold_of[group] == fold:
                continue
            priors[label] += 1
            totals[label] += len(tokens)
            table = counts[label]
            for token in tokens:
                table[token] = table.get(token, 0) + 1
        vocab = len(counts[0].keys() | counts[1].keys()) + 1
        for group, tokens, label in rows:
            if fold_of[group] != fold:
                continue
            scores = []
            for side in (0, 1):
                denominator = math.log(totals[side] + vocab)
                score = math.log(priors[side] + 1)
                score += sum(
                    math.log(counts[side].get(token, 0) + 1) - denominator
                    for token in tokens
                )
                scores.append(score)
            correct += int(scores[1] > scores[0]) == label
    return correct / len(rows)


def self_check(
    worlds: int = SELF_CHECK_WORLDS,
    namespace: str = "hs1-f3-selfcheck",
    kinds: tuple[str, ...] = KINDS_TRAIN,
    ngram: int = 1,
) -> dict[str, Any]:
    """Label-blind probes over Noul worlds (both twins).

    ``naive_bayes`` is the group-5-fold unigram (or n-gram) naive Bayes on
    ``core.mask(state + " " + instructions)``; ``negation_rule`` answers yes
    unless the text contains a negation word; ``near_rule`` answers no when
    ``meta.near_threshold`` is set; ``length_rule`` answers yes when the state
    is longer than the median state of its length class.
    """
    lengths = core.schedule(LENGTH_SHARES, worlds, f"{namespace}:lengths")
    rows, items = [], []
    for index, length in enumerate(lengths):
        kind = kinds[index % len(kinds)]
        key = f"{namespace}:{kind}:{index}"
        for item in make_group(key, kind, "noul", length):
            rows.append(
                (
                    key,
                    probe_tokens(item.state + " " + item.instructions, ngram),
                    item.gold,
                )
            )
            items.append(item)
    medians = {}
    for length in LENGTH_RANGES:
        sizes = sorted(
            len(item.state) for item in items if item.meta["length"] == length
        )
        medians[length] = sizes[len(sizes) // 2] if sizes else 0

    def accuracy(predict: Any) -> float:
        return sum(int(predict(item)) == item.gold for item in items) / len(items)

    return {
        "worlds": worlds,
        "rows": len(items),
        "ngram": ngram,
        "yes_share": sum(item.gold for item in items) / len(items),
        "naive_bayes": naive_bayes_cv(rows),
        "negation_rule": accuracy(
            lambda item: negation_count(item.state + " " + item.instructions) == 0
        ),
        "near_rule": accuracy(lambda item: not item.meta["near_threshold"]),
        "length_rule": accuracy(
            lambda item: len(item.state) > medians[item.meta["length"]]
        ),
    }


def self_check_problems(
    result: dict[str, Any], tolerance: float = SELF_CHECK_TOLERANCE
) -> list[str]:
    problems = []
    if abs(result["naive_bayes"] - 0.5) > tolerance:
        problems.append(
            f"naive Bayes accuracy {result['naive_bayes']:.3f} outside 0.5 +/- {tolerance}"
        )
    for name in ("negation_rule", "near_rule", "length_rule"):
        if result[name] > HEURISTIC_CEILING:
            problems.append(
                f"{name} accuracy {result[name]:.3f} above {HEURISTIC_CEILING}"
            )
    return problems
