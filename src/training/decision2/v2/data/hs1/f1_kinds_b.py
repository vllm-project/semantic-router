"""F1 ``hs1_quote_check`` kinds, part B: project_timeline, directory_route, rubric_pick, subscription_bill.

Every builder draws a structured world, renders it as English prose, solves it
with an oracle and then re-solves it with separately written code on facts
re-parsed from the rendered evidence and distractor text (``_*_recheck``). The
builder also draws the error mechanism of the wrong claim. Structural features
that a mechanism needs (a revised estimate, a delegation, a disqualifying
requirement, a similarly named team, a promotional credit, a price-change
notice) are drawn first and independently of the target gold, and every world
carries them, decisive or not; the mechanism is then chosen among those the
drawn features allow. See ``f1_base`` for the kind contract.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from v2.data.hs1.core import (
    GenerationError,
    add_business_days,
    business_days_between,
    fmt_date,
    join_list,
    people,
    pick,
)
from v2.data.hs1.f1_base import Claim, F1World

ATTEMPTS = 250
LENGTHS = ("short", "long")
LONG_POOL_CHARS = 7300  # distractors + filler supplied for long worlds


class _Redraw(Exception):
    """Internal: this draw missed a constraint; draw again from the same rng."""


# ---------------------------------------------------------------- shared helpers

_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
_WORD_VALUE = {word: value for value, word in enumerate(_WORDS)}
_COUNT_RX = r"(?:\d+|" + "|".join(_WORDS) + r")"
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBER = {name: index + 1 for index, name in enumerate(_MONTHS)}
_MONTH_NUMBER.update({name[:3]: index + 1 for index, name in enumerate(_MONTHS)})
_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_DATE_RX = (
    r"(?:(?:"
    + "|".join(_DAY_NAMES)
    + r") )?\d{1,2} (?:"
    + "|".join(_MONTHS)
    + r") \d{4}"
    r"|(?:"
    + "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True))
    + r") \d{1,2}, \d{4}"
    r"|\d{4}-\d{2}-\d{2}"
)
_MONEY_RX = r"-?\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
_NAME_RX = r"[A-Z][\w\-]*(?: [A-Z][\w\-]*)*"
_LIST_RX = _NAME_RX + r"(?:(?:, and |, | and )" + _NAME_RX + r")*"
_PERSON_RX = r"[A-Z][a-z]+ [A-Z][a-z]+"
DATE_STYLES = ("dmy_weekday", "dmy_weekday", "dmy", "mdy")


def _count(value: int, words: bool) -> str:
    return _WORDS[value] if words and 0 <= value < len(_WORDS) else str(value)


def _unit(value: int, unit: str, words: bool = False) -> str:
    return f"{_count(value, words)} {unit}{'' if value == 1 else 's'}"


def _read_count(token: str) -> int:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if token in _WORD_VALUE:
        return _WORD_VALUE[token]
    raise GenerationError(f"unreadable count {token!r}")


def _read_date(text: str) -> date:
    text = text.strip()
    match = re.fullmatch(r"(?:[A-Z][a-z]+day )?(\d{1,2}) ([A-Z][a-z]+) (\d{4})", text)
    if match:
        return date(int(match[3]), _MONTH_NUMBER[match[2]], int(match[1]))
    match = re.fullmatch(r"([A-Z][a-z]+) (\d{1,2}), (\d{4})", text)
    if match:
        return date(int(match[3]), _MONTH_NUMBER[match[1]], int(match[2]))
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return date(int(match[1]), int(match[2]), int(match[3]))
    raise GenerationError(f"unreadable date {text!r}")


def _cents(cents: int) -> str:
    """Money with cents always shown: $1,204.50."""
    sign = "-" if cents < 0 else ""
    value = abs(cents)
    return f"{sign}${value // 100:,}.{value % 100:02d}"


def _price(cents: int) -> str:
    """List prices: whole dollars when whole ($49), else with cents."""
    return f"${cents // 100:,}" if cents % 100 == 0 else _cents(cents)


def _read_money(text: str) -> int:
    match = re.fullmatch(r"(-?)\$([\d,]+)(?:\.(\d{2}))?", text.strip())
    if not match:
        raise GenerationError(f"unreadable amount {text!r}")
    cents = int(match[2].replace(",", "")) * 100 + int(match[3] or 0)
    return -cents if match[1] else cents


def _rx(template: str, **groups: str) -> re.Pattern[str]:
    """Compile a fact template into a full-match pattern with one group per slot.

    A slot repeated in the template must repeat the same text (a backreference).
    """
    parts = []
    seen: set[str] = set()
    for literal, field, _, _ in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if field is not None:
            parts.append(
                f"(?P={field})"
                if field in seen
                else f"(?P<{field}>{groups.get(field, '.+?')})"
            )
            seen.add(field)
    return re.compile("".join(parts))


def _units(text: str) -> list[list[str]]:
    """Split text into blocks (blank lines), then into units: lines, then sentences."""
    blocks = []
    for block in text.split("\n\n"):
        units: list[str] = []
        for line in block.split("\n"):
            line = re.sub(r"^\s*(?:[-\u2022*]|\d+[.)])\s+", "", line).strip()
            if line:
                units.extend(
                    part for part in re.split(r"(?<=\.)\s+(?=[A-Z(])", line) if part
                )
        if units:
            blocks.append(units)
    return blocks


def _split_list(text: str) -> list[str]:
    return [part for part in re.split(r", and |, | and ", text) if part]


def _weekday_on_or_after(value: date) -> date:
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value


def _draw_day(
    rng: random.Random, first: date = date(2026, 1, 5), span: int = 520
) -> date:
    return first + timedelta(days=rng.randrange(span))


def _check_args(kind: str, base: str, target: int | None, length: str) -> None:
    if base not in BASES[kind]:
        raise GenerationError(f"{kind}: base {base!r} not supported")
    if length not in LENGTHS:
        raise GenerationError(f"{kind}: bad length {length!r}")
    if base == "choice" and target is not None:
        raise GenerationError(f"{kind}: choice worlds take no target")
    if base == "noul" and target not in (0, 1):
        raise GenerationError(f"{kind}: noul target must be 0 or 1")
    if base == "score" and target not in range(SCORE_LEVELS.get(kind, 0)):
        raise GenerationError(f"{kind}: score target out of range")


def _attempt_loop(kind: str, draw: Callable[[], F1World]) -> F1World:
    last: Exception | None = None
    for _ in range(ATTEMPTS):
        try:
            world = draw()
        except _Redraw as exc:
            last = exc
            continue
        world.check()
        return world
    raise GenerationError(f"{kind}: no draw met the constraints ({last})")


def _option_layout(rng: random.Random, under: bool) -> tuple[int, int]:
    """3-5 options and the gold's rank among their ascending values, drawn in the plan.

    The builders draw the wrong value below the gold (``under``) in half of the
    worlds and above it in the other half; the gold's rank is then drawn so that
    it is uniform over positions (an extreme rank is only reachable on one side),
    so the rank of the gold among the offered values carries no signal. A redraw
    keeps the plan, so ranks that are harder to fill keep their share.
    """
    count = rng.randint(3, 5)
    weights = [1] * count
    weights[0 if under else -1] = 0
    weights[-1 if under else 0] = 2
    return count, rng.choices(range(count), weights)[0]


def _ranked_options(
    rng: random.Random,
    gold: Any,
    wrong: Any,
    pool: Iterable[Any],
    layout: tuple[int, int],
) -> list[Any]:
    """Ordered option values in the planned ``layout``: gold, wrong and extras from ``pool``, ascending."""
    count, rank = layout
    under = wrong < gold
    need_below = rank - (1 if under else 0)
    need_above = count - 1 - rank - (0 if under else 1)
    if need_below < 0 or need_above < 0:
        raise _Redraw("the wrong value is on the other side of the planned rank")
    extras = [
        value for value in dict.fromkeys(pool) if value != gold and value != wrong
    ]
    rng.shuffle(extras)
    below = [value for value in extras if value < gold][:need_below]
    above = [value for value in extras if value > gold][:need_above]
    if len(below) < need_below or len(above) < need_above:
        raise _Redraw("not enough plausible option values")
    return sorted([gold, wrong] + below + above)


def _filler(
    rng: random.Random,
    templates: Sequence[str],
    slots: Callable[[], dict[str, str]],
    need: int,
) -> tuple[str, ...]:
    pool = list(templates) + list(_GENERIC_FILLER)
    rng.shuffle(pool)
    out: list[str] = []
    total = 0
    for template in pool:
        if total >= need:
            break
        text = template.format(**slots())
        out.append(text)
        total += len(text) + 2
    if total < need:
        raise _Redraw("filler pool too small")
    return tuple(out)


def _filler_slots(
    rng: random.Random, names: Sequence[str], extra: dict[str, str]
) -> Callable[[], dict[str, str]]:
    def draw() -> dict[str, str]:
        first, second = rng.sample(list(names), 2)
        slots = {
            "name": first,
            "name2": second,
            "room": pick(rng, _ROOMS),
            "floor": pick(rng, ("second", "third", "fourth", "fifth", "ground")),
            "landmark": pick(rng, _LANDMARKS),
            "street": pick(rng, _STREETS),
        }
        slots.update(extra)
        return slots

    return draw


_ROOMS = (
    "Birch room",
    "Harbour room",
    "Orchard room",
    "Lantern room",
    "Meridian room",
    "Willow room",
    "Atlas room",
    "Quarry room",
    "Beacon room",
    "Fern room",
)
_LANDMARKS = (
    "fountain in the square",
    "north car park",
    "old clock tower",
    "bike shelter",
    "bus shelter on the corner",
    "front lawn",
    "war memorial",
    "library steps",
    "canal bridge",
    "flagpole",
)
_STREETS = (
    "Ropewalk Lane",
    "Carver Street",
    "Mill Road",
    "Station Approach",
    "Tannery Row",
    "Bellfield Road",
    "Wharf Street",
    "Granary Lane",
    "Pottery Road",
    "Weaver Street",
)
_PLACES = (
    "Harbour Street",
    "Linden Park",
    "Riverside",
    "Northgate",
    "Elm Court",
    "Kestrel Way",
    "Marlow Road",
    "Ashby Lane",
    "Quayside",
    "Foundry Yard",
    "Beacon Hill",
    "Orchard Row",
    "Millbrook",
    "Castle Wharf",
    "Juniper House",
    "Sable Point",
    "Wexford Place",
    "Cobalt Square",
    "Heron Quay",
    "Larkspur Close",
    "Fenwick Road",
    "Granite Court",
    "Hollis Park",
    "Ivory Lane",
    "Oakmere",
    "Pembrook Yard",
)

_GENERIC_FILLER = (
    "Housekeeping: the {room} on the {floor} floor is being repainted, so the Thursday stand-up moves to the "
    "small kitchen until further notice. Please keep the corridor clear of boxes and take recycling down to "
    "the courtyard bins rather than leaving it by the lifts.",
    "{name} asked everyone to check that their out-of-office messages point to the shared mailbox rather than "
    "to a personal address. Messages sent to a personal inbox during leave are easy to miss, while the shared "
    "mailbox is read every morning by whoever is on duty.",
    "A reminder about the fire drill: when the alarm sounds, leave by the nearest stairwell, do not use the "
    "lifts, and gather at the assembly point by the {landmark}. Wardens will do a head count, so please stay "
    "at the assembly point until they have finished.",
    "The coffee machine on the {floor} floor has been repaired. {name} has put up a note about descaling it "
    "regularly; the tablets are in the drawer under the sink, and the instructions are taped inside the "
    "cupboard door.",
    "Notes from the last retrospective: people liked the shorter meetings and the shared checklist. The team "
    "agreed to keep the agenda to three items, to post minutes in the channel promptly, and to rotate the "
    "note-taker, starting with {name}.",
    "IT notice: laptops will prompt for a security update soon. Save your work before restarting, and leave "
    "the machine plugged in until the update finishes. If the update fails twice, raise a ticket and quote "
    "the error message in full.",
    "{name} has booked the {room} for the quarterly social. There will be snacks and a short quiz; the "
    "questions are being kept secret, but {name2} has promised they will not be about spreadsheets. Let the "
    "organisers know about any dietary requirements.",
    "Please remember to lock your screen when you step away from your desk. Visitors are regularly shown "
    "around the {floor} floor, and a screen full of customer details is exactly the kind of thing that ends "
    "up in an incident report.",
    "The shared drive is being tidied up. Folders that nobody has opened for a long time will move to the "
    "archive area, where they stay readable. If you rely on an old folder, add a line to the tracking sheet "
    "that {name} circulated and it will stay where it is.",
    "Travel reminder: book rail tickets through the usual portal and keep the receipts. For visits near the "
    "{landmark}, the bus from the station is often quicker than a taxi at rush hour, and the stop is right "
    "outside the main entrance.",
    "Wellbeing corner: the walking group meets at lunchtime by the {landmark} and does a loop of about half an "
    "hour. New walkers are welcome, and {name} usually brings a spare umbrella for anyone who forgets theirs.",
    "Feedback on the new meeting-room screens has been mixed. Wireless sharing works well from laptops but not "
    "from some phones, so for now please borrow the adapter kept in the {room} if you want to present from a "
    "phone.",
    "Printing: the large-format printer now sits on the {floor} floor next to the stationery cupboard. It takes "
    "a few minutes to warm up in the morning, and jobs sent before then wait in the queue until it is ready.",
    "{name} is the contact for the workplace survey. It asks about desk preferences, storage and quiet space; "
    "it takes a few minutes to complete, and answers stay anonymous unless you choose to add your name.",
    "A short note on style for shared documents: use sentence case for headings, write dates in full, and avoid "
    "abbreviations that people outside the team might not know. {name} keeps a list of preferred terms on the "
    "team wiki.",
    "Security desk: visitor badges now have to be handed back at reception on the way out. Lost badges should "
    "be reported the same day, and anyone working late should let the desk know before the evening shift "
    "changes over.",
    "Recycling has moved to a three-bin system: paper, mixed plastics and general waste. The labels are "
    "colour-coded, and {name} has offered to answer questions about which bin takes what, including the "
    "tricky coffee cups.",
    "The team photo will be retaken because half the team was away last time. {name} suggests a morning slot in "
    "the {room}, where the light is better; anyone who would rather not appear in it can simply say so.",
    "The mentoring scheme is looking for volunteers. Mentors meet their mentee for an hour or so every few weeks "
    "to talk about whatever is useful, from presentations to career plans. {name2} can pair people up.",
    "Deliveries to the post room are collected twice a day. Parcels that need a signature are held at reception, "
    "and {name} sends a message when something large arrives so that it does not block the corridor.",
    "The book club has picked a short novel this time, after complaints that the last one took months to finish. "
    "Meetings are in the {room} over lunch, and {name} has a few spare copies to lend.",
    "Accessibility check: the ramp by the {landmark} entrance is being resurfaced, so step-free access is through "
    "the side door on {street} for the moment. Signs have gone up, and {name} can arrange an escort if needed.",
    "{name} is collecting ideas for the charity day. Suggestions so far include a bake sale, a sponsored walk to "
    "the {landmark} and a quiz night; the vote closes when the list stops growing.",
    "Heating and cooling: the {floor} floor thermostat is now controlled by the building manager rather than from "
    "the panel by the lifts. If a room is too warm or too cold, log it with the facilities desk.",
    "New starters get a buddy for their first weeks. {name2} has put together a list of the questions people "
    "ask most, from where the spare chargers live to how to book a meeting room at short notice.",
    "The internal newsletter wants short pieces from every team. A paragraph and a photo is plenty, and {name} "
    "will tidy up the wording; the most popular section last time was the one about people's pets.",
    "Desk booking: hot desks on the {floor} floor can be reserved up to a week ahead. Please release a booking "
    "you no longer need, because empty reserved desks are the most common complaint on the feedback board.",
    "Kitchen etiquette, once more: label food in the shared fridge with your name, and clear out anything you "
    "have forgotten by Friday afternoon. {name} will empty the fridge after that without further warning.",
    "Password manager rollout: everyone will receive an invitation email with a setup link. The link expires "
    "after a while, so set it up when it arrives; {name2} is running short drop-in sessions in the {room}.",
    "The cycle-to-work scheme has reopened. The racks in the {street} basement have space again after the "
    "clear-out, and the showers on the ground floor are back in use following the plumbing repair.",
    "Customer thank-you notes are pinned on the board by the {room}. {name} reads a few out at the monthly "
    "all-hands, which has become many people's favourite part of the meeting.",
    "Meeting-free mornings are being trialled on Wednesdays. The idea is to keep a block of time for focused "
    "work; urgent calls are still fine, but routine meetings should move elsewhere in the week.",
    "Lost property: a blue umbrella, a set of keys on a lighthouse keyring and a pair of reading glasses are "
    "waiting at reception. Anything unclaimed goes to the charity shop on {street} at the end of the month.",
    "First-aid training has a few places left. The course lasts a day, covers the basics and is run in the "
    "{room}; {name} keeps the list of trained first-aiders up to date on the noticeboard.",
    "The plants on the {floor} floor are looked after by volunteers. {name2} has drawn up a watering rota, and "
    "the large fern by the window apparently prefers to be left alone.",
)


def _distinct_names(rng: random.Random, count: int) -> list[str]:
    return people(rng, count)


# ---------------------------------------------------------------- project_timeline

_PT_DOMAINS = (
    {
        "noun": "fit-out",
        "first": ("Site Survey", "Strip-Out"),
        "middle": (
            "Partition Framing",
            "Electrical First Fix",
            "Plumbing Rough-In",
            "Ceiling Grid",
            "Plastering",
            "Floor Screed",
            "Joinery",
            "Decoration",
            "Data Cabling",
        ),
        "last": ("Snagging Walkthrough", "Client Handover"),
        "sources": (
            "electrical contractor",
            "joinery supplier",
            "site foreman",
            "plastering crew",
        ),
        "roles": (
            "site manager",
            "fit-out coordinator",
            "client representative",
            "quantity surveyor",
        ),
    },
    {
        "noun": "release",
        "first": ("Scope Sign-Off", "Environment Setup"),
        "middle": (
            "API Changes",
            "Database Migration",
            "Front-End Update",
            "Integration Testing",
            "Security Review",
            "Load Testing",
            "Release Notes",
            "User Acceptance Testing",
        ),
        "last": ("Production Release", "Go-Live Check"),
        "sources": (
            "platform team",
            "test lead",
            "database administrator",
            "security analyst",
        ),
        "roles": (
            "release manager",
            "engineering lead",
            "product owner",
            "delivery manager",
        ),
    },
    {
        "noun": "lab move",
        "first": ("Equipment Inventory", "Hazard Assessment"),
        "middle": (
            "Packing",
            "Decommissioning",
            "Specialist Transport",
            "Bench Installation",
            "Reinstallation",
            "Calibration",
            "Validation Runs",
            "Safety Inspection",
        ),
        "last": ("Lab Reopening", "Sign-Off Inspection"),
        "sources": (
            "removals firm",
            "lab technician",
            "facilities team",
            "equipment vendor",
        ),
        "roles": (
            "lab manager",
            "facilities coordinator",
            "principal investigator",
            "safety officer",
        ),
    },
    {
        "noun": "campaign",
        "first": ("Brief Approval", "Audience Research"),
        "middle": (
            "Copywriting",
            "Visual Design",
            "Legal Review",
            "Translation",
            "Print Production",
            "Landing Page Build",
            "Media Booking",
            "Proofing",
        ),
        "last": ("Campaign Launch", "Launch Review"),
        "sources": ("design agency", "print house", "legal team", "translation vendor"),
        "roles": (
            "marketing manager",
            "campaign lead",
            "account director",
            "brand manager",
        ),
    },
    {
        "noun": "racking project",
        "first": ("Floor Survey", "Racking Order"),
        "middle": (
            "Anchor Drilling",
            "Upright Assembly",
            "Beam Fitting",
            "Load Testing",
            "Aisle Marking",
            "Sprinkler Adjustment",
            "Labelling",
            "Scanner Setup",
        ),
        "last": ("Go-Live", "Final Inspection"),
        "sources": (
            "racking installer",
            "fire engineer",
            "warehouse supervisor",
            "scanner vendor",
        ),
        "roles": (
            "warehouse manager",
            "operations planner",
            "site supervisor",
            "logistics lead",
        ),
    },
    {
        "noun": "site migration",
        "first": ("Content Audit", "Hosting Setup"),
        "middle": (
            "Template Build",
            "Content Migration",
            "Redirect Mapping",
            "Accessibility Check",
            "Search Indexing",
            "Form Rebuild",
            "Analytics Setup",
            "Editor Training",
        ),
        "last": ("Site Launch", "Launch Check"),
        "sources": (
            "web agency",
            "content team",
            "hosting provider",
            "accessibility auditor",
        ),
        "roles": (
            "web manager",
            "digital lead",
            "content strategist",
            "project coordinator",
        ),
    },
    {
        "noun": "refurbishment",
        "first": ("Condition Survey", "Room Clearance"),
        "middle": (
            "Wall Repairs",
            "Electrical Upgrade",
            "Vinyl Flooring",
            "Infection Control Fit",
            "Painting",
            "Equipment Fitting",
            "Signage",
            "Deep Clean",
        ),
        "last": ("Clinic Reopening", "Handover Inspection"),
        "sources": (
            "building contractor",
            "estates team",
            "flooring supplier",
            "infection control nurse",
        ),
        "roles": (
            "practice manager",
            "estates officer",
            "clinical lead",
            "project coordinator",
        ),
    },
    {
        "noun": "training rollout",
        "first": ("Needs Analysis", "Trainer Booking"),
        "middle": (
            "Course Design",
            "Slide Production",
            "Pilot Session",
            "Feedback Review",
            "Platform Upload",
            "Assessment Writing",
            "Session Scheduling",
            "Invitations",
        ),
        "last": ("Programme Launch", "Launch Briefing"),
        "sources": (
            "learning designer",
            "training vendor",
            "people team",
            "platform administrator",
        ),
        "roles": (
            "learning manager",
            "people partner",
            "programme lead",
            "training coordinator",
        ),
    },
)

_PT_START = (
    "Work on the {project} begins on {start}, which counts as working day 1.",
    "The {project} starts on {start}; that day is working day 1.",
    "Day 1 of the {project} is {start}.",
)
_PT_RULES = (
    "Only Monday to Friday are working days; nobody works on Saturdays or Sundays, and no public holidays fall "
    "in the period. A task of N working days occupies N consecutive working days and is complete at the end of "
    "the last of them. A task starts on the next working day after every task it depends on is complete, tasks "
    "with no prerequisites start on day 1, and tasks that do not depend on each other run in parallel.",
    "Scheduling convention: weekends (Saturday and Sunday) are not working days and there are no holidays in "
    "this window. Durations are in working days counted from the day a task starts, so a three-day task that "
    "starts on a Monday is done at the end of Wednesday. Each task begins on the first working day after all "
    "of its prerequisites are done, and independent tasks run side by side.",
    "We count working days only: Monday to Friday, with Saturdays and Sundays off and no bank holidays in the "
    "period. A task that starts on a given working day and needs N working days ends at the close of the Nth "
    "working day counted from that day. Nothing starts until all of its prerequisites have ended, but tasks "
    "with no dependency between them are worked on at the same time.",
)
_PT_RULES_SHORT = (
    "Working days are Monday to Friday; weekends are not worked and there are no holidays in the period. A task "
    "of N working days occupies N consecutive working days, starts on the next working day after all of its "
    "prerequisites are complete, and runs in parallel with tasks it does not depend on.",
    "Only weekdays count: Saturdays and Sundays are off, with no holidays in the window. Each task starts on the "
    "working day after its last prerequisite finishes (day 1 if it has none), takes its estimate in consecutive "
    "working days, and runs alongside independent tasks.",
    "Durations are in working days (Monday to Friday; no weekend work and no holidays in the period). A task "
    "begins on the first working day after all of its prerequisites are done and ends at the close of its last "
    "working day; unrelated tasks run side by side.",
)
_PT_FINISH_RULE = (
    "The project is finished when its last task is complete.",
    "The finish date of the project is the day its final task is done.",
    "The project ends on the day its last task ends.",
)
_PT_OTHER_RULE = "The same working-day rules apply to it."
_PT_HEAD_LIST = (
    "Task list for the {project}:",
    "Estimates for the {project}, in working days:",
    "Plan for the {project}:",
)
_PT_HEAD_PROSE = (
    "The {project} plan has {count} tasks.",
    "The {project} is broken into {count} tasks.",
)
_PT_TASK_DEP = (
    "{name}: {days}, after {deps}.",
    "{name} ({days}) follows {deps}.",
    "{name} takes {days} and can start once {deps} {be} finished.",
    "{name} needs {days}; it cannot begin until {deps} {have} finished.",
    "After {deps}, {name} runs for {days}.",
)
_PT_TASK_FREE = (
    "{name}: {days}, no prerequisites.",
    "{name} ({days}) can start on day 1.",
    "{name} takes {days} and can start straight away.",
    "{name} needs {days}; it depends on nothing else.",
    "{name} runs for {days} from the start date.",
)
_PT_REVISE = (
    "Update, {when}: {name} will now take {new}, not {old} as planned.",
    "Revised estimate ({when}): {name} goes from {old} to {new}.",
    "On {when}, the {source} confirmed that {name} needs {new} rather than the {old} in the plan.",
    "Estimate change, {when}: {name} is now {new} (previously {old}).",
)
_PT_REVISE_NAMED = (
    "Update on the {project}, {when}: {name} will now take {new}, not {old} as planned.",
    "Change to the {project} plan ({when}): {name} goes from {old} to {new}.",
)
_PT_DEADLINE = (
    "The {project} has to be finished by the end of {deadline}.",
    "The client needs the {project} completed by {deadline}.",
    "Deadline for the {project}: end of {deadline}.",
)
_PT_QUESTION = {
    "choice": (
        "On which date will the {project} be finished at the earliest?",
        "What is the earliest finish date for the {project}?",
        "When is the earliest the {project} can be completed?",
    ),
    ("noul", "by"): (
        "Can the {project} be finished by the end of {deadline}?",
        "Will the {project} be done by {deadline}?",
    ),
    ("noul", "past"): (
        "Will the {project} run past {deadline}?",
        "Will the {project} finish later than {deadline}?",
    ),
    "score": (
        "How much working-day slack will the {project} have against its deadline of {deadline}?",
        "How far ahead of its {deadline} deadline, if at all, will the {project} finish?",
    ),
}
_PT_CONCLUDE_CHOICE = (
    "the {project} will be finished on {finish}",
    "the earliest finish for the {project} is {finish}",
    "the {project} can be completed on {finish} at the earliest",
)
_PT_CONCLUDE_NOUL = {
    ("by", 1): (
        "the {project} can be finished by {deadline}",
        "the {project} will be done by {deadline}",
    ),
    ("by", 0): (
        "the {project} cannot be finished by {deadline}",
        "the {project} will not be done by {deadline}",
    ),
    ("past", 1): (
        "the {project} will run past {deadline}",
        "the {project} will finish later than {deadline}",
    ),
    ("past", 0): (
        "the {project} will not run past {deadline}",
        "the {project} will not finish later than {deadline}",
    ),
}
_PT_CONCLUDE_SCORE = (
    (
        "the {project} will finish after its {deadline} deadline",
        "the {project} will finish on its deadline day, {deadline}",
        "the {project} will finish one or two working days ahead of its {deadline} deadline",
        "the {project} will finish three or more working days ahead of its {deadline} deadline",
    ),
    (
        "the {project} will miss its {deadline} deadline",
        "the {project} will finish exactly on its {deadline} deadline",
        "the {project} will beat its {deadline} deadline by one or two working days",
        "the {project} will beat its {deadline} deadline by at least three working days",
    ),
)
_PT_LEVELS = (
    "It finishes after the deadline",
    "It finishes on the deadline day itself",
    "It finishes one or two working days before the deadline",
    "It finishes three or more working days before the deadline",
)
_PT_RATIONALE = (
    "the chain {chain} adds up to {total}",
    "the longest dependent run is {chain}, {total} in all",
    "{total} along {chain} ends on {finish}",
    "{chain} take {total} end to end, finishing on {finish}",
)
_PT_FILLER = (
    "Site logistics for the {project}: deliveries come in through the rear gate on {street}, and the skip is "
    "collected on request. {name} holds the keys to the store room, so ask before borrowing tools from it.",
    "{name} walked round with the client and took photos for the {project} progress report. The client asked "
    "mainly about finishes and signage; nothing in the conversation changes the plan or any estimate.",
    "Weekly risk review: no new risks were raised for the {project}. The register still lists supplier "
    "availability and weather as the main watch items, both rated low, and {name} will look at it again at "
    "the next meeting.",
    "Communication plan: progress on the {project} goes out as a short email on Friday afternoons. {name} "
    "drafts it, and anyone with a photo worth sharing should post it in the channel before lunch.",
    "Documents for the {project} live in the shared folder under Plans. Please save drawings as PDF, put the "
    "revision letter in the file name and never overwrite an earlier version; {name2} archives the old ones.",
    "The client's facilities team asked for a named contact during the {project}. {name} takes calls during "
    "office hours, and anything out of hours goes to the building's security desk, which has the contact sheet.",
    "Health and safety: an induction is mandatory for anyone working on the {project}. It takes about twenty "
    "minutes and covers access, fire exits and first aid, and {name} runs it on request.",
    "Budget note: spending on the {project} is tracked separately by finance and has no bearing on the task "
    "estimates in this thread. Questions about purchase orders should go to {name2}.",
    "Lessons learned from the last similar job, collected by {name}: label every box, photograph the before "
    "state, and keep one shared list of open questions instead of five private ones.",
    "Stakeholder note: the steering group has seen the {project} plan and had no comments beyond thanking the "
    "team. {name} will present a one-page summary at their next session.",
)


def _pt_level(slack: int) -> int:
    if slack < 0:
        return 0
    if slack == 0:
        return 1
    return 2 if slack <= 2 else 3


def _pt_wrong_level(rng: random.Random, gold: int) -> int:
    """Wrong levels chosen so that quoted levels are uniform over right and wrong quotes alike."""
    return rng.choice((1, 2)) if gold in (0, 3) else rng.choice((0, 3))


def _pt_graph(
    rng: random.Random, domain: dict[str, Any], count: int
) -> tuple[list[str], dict[str, set[str]]]:
    sources = 2 if count >= 5 and rng.random() < 0.4 else 1
    firsts = list(domain["first"])
    rng.shuffle(firsts)
    middle = rng.sample(domain["middle"], count - sources - 1)
    middle.sort(key=domain["middle"].index)
    names = firsts[:sources] + middle + [pick(rng, domain["last"])]
    preds: dict[str, set[str]] = {name: set() for name in names}
    for index in range(sources, count - 1):
        earlier = names[:index]
        window = earlier[-3:]
        size = 2 if len(window) >= 2 and rng.random() < 0.3 else 1
        preds[names[index]] = set(rng.sample(window, size))
    used = {p for name in names for p in preds[name]}
    preds[names[-1]] = {name for name in names[:-1] if name not in used}
    ancestors: dict[str, set[str]] = {}
    for name in names:
        ancestors[name] = set()
        for p in preds[name]:
            ancestors[name] |= ancestors[p] | {p}
    for name in names:
        preds[name] = {
            p
            for p in preds[name]
            if not any(p in ancestors[q] for q in preds[name] if q != p)
        }
    return names, preds


def _pt_paths(names: Sequence[str], preds: dict[str, set[str]]) -> list[list[str]]:
    paths: list[list[str]] = []

    def walk(node: str, tail: list[str]) -> None:
        if not preds[node]:
            paths.append([node] + tail)
            return
        for p in sorted(preds[node], key=names.index):
            walk(p, [node] + tail)

    walk(names[-1], [])
    return paths


def _pt_solve(
    names: Sequence[str], dur: dict[str, int], preds: dict[str, set[str]]
) -> tuple[int, list[str]]:
    """Oracle: forward pass in listed (topological) order; the critical chain must be unique."""
    end: dict[str, int] = {}
    for name in names:
        begin = max((end[p] for p in preds[name]), default=0) + 1
        end[name] = begin + dur[name] - 1
    finish = max(end.values())
    last = [name for name in names if end[name] == finish]
    if len(last) != 1:
        raise _Redraw("two tasks finish last")
    chain = [last[0]]
    while preds[chain[-1]]:
        best = max(end[p] for p in preds[chain[-1]])
        tops = [p for p in preds[chain[-1]] if end[p] == best]
        if len(tops) != 1:
            raise _Redraw("critical chain is not unique")
        chain.append(tops[0])
    chain.reverse()
    if len(chain) < 2:
        raise _Redraw("chain too short")
    return finish, chain


def _pt_try(names, dur, preds) -> tuple[int, list[str]] | None:
    try:
        return _pt_solve(names, dur, preds)
    except _Redraw:
        return None


def _pt_ancestors(
    names: Sequence[str], preds: dict[str, set[str]]
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in names:
        out[name] = set()
        for p in preds[name]:
            out[name] |= out[p] | {p}
    return out


def _pt_candidates(
    names, preds, original, revised, revised_task, chain_true, finish_true
):
    """Every mechanism output available in this world: (mechanism, variant, finish, chain)."""
    out: list[tuple[str, str, int, list[str]]] = []
    stale = _pt_try(names, {**revised, revised_task: original[revised_task]}, preds)
    if stale and stale[0] != finish_true:
        out.append(("stale_value", "original_estimate", stale[0], stale[1]))
    for first, second in zip(chain_true, chain_true[1:]):
        cut = {name: set(ps) for name, ps in preds.items()}
        cut[second].discard(first)
        model = _pt_try(names, revised, cut)
        if model and model[0] < finish_true:
            out.append(("arithmetic_slip", "dependent_as_parallel", model[0], model[1]))
    ancestors = _pt_ancestors(names, preds)
    for x in names[:-1]:
        for y in names[:-1]:
            if x == y or x in ancestors[y] or y in ancestors[x]:
                continue
            if names.index(x) > names.index(y):
                continue
            joined = {name: set(ps) for name, ps in preds.items()}
            joined[y].add(x)
            model = _pt_try(names, revised, joined)
            if model and model[0] > finish_true and x in model[1] and y in model[1]:
                out.append(
                    ("arithmetic_slip", "parallel_as_sequential", model[0], model[1])
                )
    for path in _pt_paths(names, preds):
        if path != chain_true:
            total = sum(revised[name] for name in path)
            if total < finish_true:
                out.append(("short_chain", "non_critical_path", total, path))
    return out


def _pt_render_tasks(
    rng: random.Random,
    project: str,
    names: Sequence[str],
    dur: dict[str, int],
    preds: dict[str, set[str]],
    style: str,
    words: bool,
) -> tuple[str, list[str]]:
    records = []
    for name in names:
        deps = sorted(preds[name], key=names.index)
        days = _unit(dur[name], "working day", words)
        if deps:
            template = (
                pick(rng, _PT_TASK_DEP)
                if style == "prose"
                else _PT_TASK_DEP[int(style[4:])]
            )
            plural = len(deps) > 1
            records.append(
                template.format(
                    name=name,
                    days=days,
                    deps=join_list(deps),
                    be="are" if plural else "is",
                    have="have" if plural else "has",
                )
            )
        else:
            template = (
                pick(rng, _PT_TASK_FREE)
                if style == "prose"
                else _PT_TASK_FREE[int(style[4:])]
            )
            records.append(template.format(name=name, days=days))
    if style == "prose":
        head = pick(rng, _PT_HEAD_PROSE).format(
            project=project, count=_count(len(names), words)
        )
        return head + " " + " ".join(records), records
    bullet = pick(rng, ("- ", "\u2022 ", "#"))
    lines = [
        (f"{index + 1}. " if bullet == "#" else bullet) + record
        for index, record in enumerate(records)
    ]
    head = pick(rng, _PT_HEAD_LIST).format(project=project)
    return head + "\n" + "\n".join(lines), records


def _pt_revision(rng, project, name, old, new, when, source, words, named=False) -> str:
    template = pick(rng, _PT_REVISE_NAMED if named else _PT_REVISE + _PT_REVISE_NAMED)
    return template.format(
        project=project,
        when=when,
        name=name,
        new=_unit(new, "working day", words),
        old=_count(old, words),
        source=source,
    )


def _pt_plan(rng: random.Random, base: str, target: int | None) -> dict[str, Any]:
    """Draw the visible features first, then the mechanism they allow for this target."""
    revdir = rng.choice(("longer", "shorter"))
    plan: dict[str, Any] = {"revdir": revdir, "polarity": None, "wrong_level": None}
    if base == "choice":
        # stale goes either way with the revision; short chains are early, so the
        # arithmetic slip is the sequential one here and the wrong date is early
        # in exactly half of the choice worlds
        mechanism = pick(rng, ("stale_value", "arithmetic_slip", "short_chain"))
        if mechanism == "stale_value":
            direction = "under" if revdir == "longer" else "over"
        else:
            direction = "under" if mechanism == "short_chain" else "over"
        plan["layout"] = _option_layout(rng, direction == "under")
    else:
        if base == "noul":
            plan["polarity"] = rng.choice(("by", "past"))
            under = (plan["polarity"] == "by") == (target == 0)
        else:
            plan["wrong_level"] = _pt_wrong_level(rng, target)
            under = plan["wrong_level"] > target
        direction = "under" if under else "over"
        stale_ok = (revdir == "longer") == under
        if stale_ok and rng.random() < 2 / 3:
            mechanism = "stale_value"
        elif under:
            mechanism = "short_chain" if rng.random() < 0.8 else "arithmetic_slip"
        else:
            mechanism = "arithmetic_slip"
    plan["mechanism"] = mechanism
    plan["direction"] = direction
    return plan


def _pt_deadline_index(rng, base, target, plan, f_gold, f_wrong) -> int | None:
    if base == "choice":
        return None
    if base == "noul":
        low, high = sorted((f_gold, f_wrong))
        return rng.randrange(low, high)  # low <= d < high
    feasible = [
        d
        for d in range(1, max(f_gold, f_wrong) + 7)
        if _pt_level(d - f_gold) == target
        and _pt_level(d - f_wrong) == plan["wrong_level"]
    ]
    if not feasible:
        raise _Redraw("no deadline fits the score levels")
    return pick(rng, feasible)


def _pt_answer(
    base: str, polarity: str | None, finish: int, deadline: int | None
) -> int:
    if base == "noul":
        return int(finish <= deadline) if polarity == "by" else int(finish > deadline)
    return _pt_level(deadline - finish)


def _pt_draw(
    rng: random.Random, base: str, target: int | None, length: str, plan: dict[str, Any]
) -> F1World:
    domain_index = rng.randrange(len(_PT_DOMAINS))
    domain = _PT_DOMAINS[domain_index]
    count = (
        rng.choice((4, 5, 5, 6, 6)) if length == "long" else rng.choice((4, 4, 5, 5, 5))
    )
    names, preds = _pt_graph(rng, domain, count)
    if len(_pt_paths(names, preds)) < 2:
        raise _Redraw("no parallel branch")
    original = {name: rng.randint(1, 8) for name in names}
    revised_task = pick(rng, names)
    step = rng.randint(1, 4)
    old = original[revised_task]
    new = old + step if plan["revdir"] == "longer" else old - step
    if not 1 <= new <= 11:
        raise _Redraw("revision out of range")
    revised = {**original, revised_task: new}
    # optional second, non-decisive revision (a distractor record)
    second = None
    finish_true, chain_true = _pt_solve(names, revised, preds)
    if rng.random() < 0.4:
        others = [
            name for name in names if name != revised_task and name not in chain_true
        ]
        if others:
            s_name = pick(rng, others)
            s_new = original[s_name] + rng.choice((-2, -1, 1, 2))
            if 1 <= s_new <= 11:
                trial = {**revised, s_name: s_new}
                model = _pt_try(names, trial, preds)
                if model and model[0] == finish_true:
                    second = (s_name, original[s_name], s_new)
                    revised = trial
                    finish_true, chain_true = model
    candidates = _pt_candidates(
        names, preds, original, revised, revised_task, chain_true, finish_true
    )
    direction = plan["direction"]
    fits = [
        c
        for c in candidates
        if c[0] == plan["mechanism"] and (c[2] < finish_true) == (direction == "under")
    ]
    if not fits:
        raise _Redraw("mechanism not available")
    rng.shuffle(fits)
    start = _weekday_on_or_after(_draw_day(rng))
    style = pick(rng, DATE_STYLES)
    words = rng.random() < 0.3

    def day(index: int) -> date:
        return add_business_days(start, index - 1)

    chosen = None
    deadline_index = None
    for candidate in fits:
        try:
            deadline_index = _pt_deadline_index(
                rng, base, target, plan, finish_true, candidate[2]
            )
        except _Redraw:
            continue
        chosen = candidate
        break
    if chosen is None:
        raise _Redraw("no candidate fits the target")
    mechanism, detail, f_wrong, chain_wrong = chosen
    if base == "noul":
        gold = _pt_answer(base, plan["polarity"], finish_true, deadline_index)
        if (
            gold != target
            or _pt_answer(base, plan["polarity"], f_wrong, deadline_index) != 1 - target
        ):
            raise _Redraw("noul answers do not split")
    elif base == "score":
        gold = _pt_level(deadline_index - finish_true)
        if gold != target:
            raise _Redraw("score gold mismatch")

    place, other_place = rng.sample(_PLACES, 2)
    project = f"{place} {domain['noun']}"
    other_domain = _PT_DOMAINS[
        (domain_index + rng.randrange(1, len(_PT_DOMAINS))) % len(_PT_DOMAINS)
    ]
    other_project = f"{other_place} {other_domain['noun']}"
    fmt = lambda value: fmt_date(value, style)  # noqa: E731
    deadline = day(deadline_index) if deadline_index is not None else None

    # choice options
    choices: tuple[str, ...] = ()
    if base == "choice":
        indices = [c[2] for c in candidates] + [
            finish_true + 1,
            f_wrong + 1,
            finish_true - 1,
            f_wrong - 1,
        ]
        pool = [day(index) for index in indices if index >= 1]
        pool += [
            start + timedelta(days=index - 1) for index in (finish_true, f_wrong)
        ]  # calendar-day count
        pool = [value for value in pool if value.weekday() < 5]
        values = _ranked_options(
            rng, day(finish_true), day(f_wrong), pool, plan["layout"]
        )
        choices = tuple(fmt(value) for value in values)
        gold = values.index(day(finish_true))
        wrong_answer = values.index(day(f_wrong))
    elif base == "noul":
        wrong_answer = 1 - gold
    else:
        wrong_answer = plan["wrong_level"]

    # evidence
    rule_bank = _PT_RULES if length == "long" else _PT_RULES_SHORT
    rule_index = rng.randrange(len(rule_bank))
    start_sentence = pick(rng, _PT_START).format(project=project, start=fmt(start))
    rules = [start_sentence, rule_bank[rule_index], pick(rng, _PT_FINISH_RULE)]
    deadline_sentence = None
    if deadline is not None:
        deadline_sentence = pick(rng, _PT_DEADLINE).format(
            project=project, deadline=fmt(deadline)
        )
        rules.append(deadline_sentence)
    style_kind = (
        "prose" if rng.random() < 0.35 else f"list{rng.randrange(len(_PT_TASK_DEP))}"
    )
    task_block, task_records = _pt_render_tasks(
        rng, project, names, original, preds, style_kind, words
    )
    issued = start - timedelta(days=rng.randint(14, 28))
    update_day = issued + timedelta(days=rng.randint(2, 9))
    revision = _pt_revision(
        rng,
        project,
        revised_task,
        old,
        new,
        fmt(update_day),
        pick(rng, domain["sources"]),
        words,
    )
    evidence = [" ".join(rules), task_block, revision]
    distractors: list[str] = []
    if second is not None:
        s_day = issued + timedelta(days=rng.randint(2, 11))
        distractors.append(
            _pt_revision(
                rng,
                project,
                second[0],
                second[1],
                second[2],
                fmt(s_day),
                pick(rng, domain["sources"]),
                words,
            )
        )
    other_names, other_preds = _pt_graph(
        rng, other_domain, 3 if length == "short" else 5
    )
    if set(other_names) & set(names):
        raise _Redraw("task names collide across projects")
    other_dur = {name: rng.randint(1, 8) for name in other_names}
    other_start = _weekday_on_or_after(start + timedelta(days=rng.randint(-10, 10)))
    other_rules = (
        pick(rng, _PT_START).format(project=other_project, start=fmt(other_start))
        + " "
        + _PT_OTHER_RULE
    )
    other_block, _ = _pt_render_tasks(
        rng,
        other_project,
        other_names,
        other_dur,
        other_preds,
        "prose" if rng.random() < 0.5 else f"list{rng.randrange(len(_PT_TASK_DEP))}",
        words,
    )
    other_task = pick(rng, other_names)
    other_new = max(1, other_dur[other_task] + rng.choice((-2, -1, 1, 2, 3)))
    other_revision = (
        _pt_revision(
            rng,
            other_project,
            other_task,
            other_dur[other_task],
            other_new,
            fmt(issued + timedelta(days=rng.randint(1, 12))),
            pick(rng, other_domain["sources"]),
            words,
            named=True,
        )
        if other_new != other_dur[other_task]
        else None
    )
    if length == "short":
        if rng.random() < 0.5 or not (distractors or other_revision):
            distractors.append(other_rules + "\n" + other_block)
        elif other_revision and (not distractors or rng.random() < 0.5):
            distractors.append(other_revision)
    else:
        distractors.append(other_rules + "\n" + other_block)
        if other_revision:
            distractors.append(other_revision)
        third_domain = _PT_DOMAINS[(domain_index + 1) % len(_PT_DOMAINS)]
        if third_domain is other_domain:
            third_domain = _PT_DOMAINS[(domain_index + 2) % len(_PT_DOMAINS)]
        third_names, third_preds = _pt_graph(rng, third_domain, 4)
        if not (set(third_names) & (set(names) | set(other_names))):
            third_place = pick(
                rng, [p for p in _PLACES if p not in (place, other_place)]
            )
            third_project = f"{third_place} {third_domain['noun']}"
            third_dur = {name: rng.randint(1, 8) for name in third_names}
            third_start = _weekday_on_or_after(
                start + timedelta(days=rng.randint(7, 30))
            )
            third_block, _ = _pt_render_tasks(
                rng, third_project, third_names, third_dur, third_preds, "prose", words
            )
            distractors.append(
                pick(rng, _PT_START).format(
                    project=third_project, start=fmt(third_start)
                )
                + " "
                + _PT_OTHER_RULE
                + " "
                + third_block
            )
    decisive = [start_sentence, *task_records, revision]
    if deadline_sentence:
        decisive.append(deadline_sentence)

    # question and claims
    fields = {"project": project}
    if deadline is not None:
        fields["deadline"] = fmt(deadline)
    if base == "noul":
        question = pick(rng, _PT_QUESTION[("noul", plan["polarity"])]).format(**fields)
    else:
        question = pick(rng, _PT_QUESTION[base]).format(**fields)
    conclusion_variant = rng.randrange(3 if base == "choice" else 2)
    rationale_variant = rng.randrange(len(_PT_RATIONALE))

    def claim(answer: int, finish: int, chain: Sequence[str], label: str) -> Claim:
        values = {**fields, "finish": fmt(day(finish))}
        if base == "choice":
            conclusion = _PT_CONCLUDE_CHOICE[conclusion_variant].format(**values)
        elif base == "noul":
            conclusion = _PT_CONCLUDE_NOUL[(plan["polarity"], answer)][
                conclusion_variant
            ].format(**values)
        else:
            conclusion = _PT_CONCLUDE_SCORE[conclusion_variant][answer].format(**values)
        rationale = _PT_RATIONALE[rationale_variant].format(
            chain=join_list(list(chain)),
            total=_unit(finish, "working day"),
            finish=values["finish"],
        )
        return Claim(
            answer=answer, conclusion=conclusion, rationale=rationale, mechanism=label
        )

    right = claim(gold, finish_true, chain_true, "correct")
    wrong = claim(wrong_answer, f_wrong, chain_wrong, mechanism)

    text = "\n\n".join(evidence + distractors)
    recheck = _pt_recheck(text, project, base, plan["polarity"], choices)
    if (
        _pt_recheck("\n\n".join(evidence), project, base, plan["polarity"], choices)
        != recheck
    ):
        raise GenerationError("project_timeline: a distractor changed the answer")
    need = (
        max(0, LONG_POOL_CHARS - sum(len(block) + 2 for block in distractors))
        if length == "long"
        else 0
    )
    filler_names = people(rng, 6)
    filler = (
        _filler(
            rng,
            _PT_FILLER,
            _filler_slots(rng, filler_names, {"project": project}),
            need,
        )
        if need
        else ()
    )
    stale_model = _pt_try(names, {**revised, revised_task: old}, preds)
    facts = {
        "project": project,
        "start": start.isoformat(),
        "tasks": [
            {
                "name": name,
                "estimate": original[name],
                "final": revised[name],
                "after": sorted(preds[name], key=names.index),
            }
            for name in names
        ],
        "revision": {
            "task": revised_task,
            "old": old,
            "new": new,
            "changes_finish": stale_model is None or stale_model[0] != finish_true,
        },
        "second_revision": list(second) if second else None,
        "finish_index": finish_true,
        "finish": day(finish_true).isoformat(),
        "chain": chain_true,
        "wrong": {
            "mechanism": mechanism,
            "detail": detail,
            "finish_index": f_wrong,
            "chain": chain_wrong,
        },
        "polarity": plan["polarity"],
        "deadline": deadline.isoformat() if deadline else None,
        "revision_direction": plan["revdir"],
        "error_direction": plan["direction"],
        "names": sorted(filler_names) if filler else [],
    }
    return F1World(
        kind="project_timeline",
        base=base,
        subject=f"the {project} schedule",
        question=question,
        choices=(
            choices if base == "choice" else (_PT_LEVELS if base == "score" else ())
        ),
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=filler,
        right=right,
        wrong=wrong,
        decisive=tuple(decisive),
        facts=facts,
        variant=f"{style_kind}-r{rule_index}-c{conclusion_variant}-q{rationale_variant}",
        roles=tuple(domain["roles"]),
    )


# re-check: separately written parser and solver for project_timeline
_PT_RX_START = [_rx(t, start=_DATE_RX) for t in _PT_START]
_PT_RX_DEADLINE = [_rx(t, deadline=_DATE_RX) for t in _PT_DEADLINE]
_PT_RX_HEAD = [_rx(t) for t in _PT_HEAD_LIST] + [
    _rx(t, count=_COUNT_RX) for t in _PT_HEAD_PROSE
]
_DAYS_RX = _COUNT_RX + r" working days?"
_PT_RX_TASK = [
    _rx(t, name=_NAME_RX, days=_DAYS_RX, deps=_LIST_RX, be="is|are", have="has|have")
    for t in _PT_TASK_DEP + _PT_TASK_FREE
]
_PT_RX_REVISE = [
    _rx(
        t,
        when=_DATE_RX,
        name=_NAME_RX,
        new=_DAYS_RX,
        old=_COUNT_RX,
        source=r"[a-z][a-z \-]*",
    )
    for t in _PT_REVISE + _PT_REVISE_NAMED
]


def _pt_recheck(
    text: str, project: str, base: str, polarity: str | None, options: Sequence[str]
) -> int:
    start = deadline = None
    tasks: dict[str, tuple[int, list[str]]] = {}
    changes: list[tuple[date, str, int, int]] = []
    for units in _units(text):
        owner = None
        for pattern in _PT_RX_HEAD:
            match = pattern.fullmatch(units[0])
            if match:
                owner = match["project"]
                break
        for position, unit in enumerate(units):
            for pattern in _PT_RX_START:
                match = pattern.fullmatch(unit)
                if match and match["project"] == project:
                    start = _read_date(match["start"])
            for pattern in _PT_RX_DEADLINE:
                match = pattern.fullmatch(unit)
                if match and match["project"] == project:
                    deadline = _read_date(match["deadline"])
            for pattern in _PT_RX_REVISE:
                match = pattern.fullmatch(unit)
                if match and match.groupdict().get("project", project) == project:
                    changes.append(
                        (
                            _read_date(match["when"]),
                            match["name"],
                            _read_count(match["old"]),
                            _read_count(match["new"].split()[0]),
                        )
                    )
            if owner == project and position > 0:
                for pattern in _PT_RX_TASK:
                    match = pattern.fullmatch(unit)
                    if match:
                        deps = _split_list(match.groupdict().get("deps") or "")
                        tasks[match["name"]] = (
                            _read_count(match["days"].split()[0]),
                            deps,
                        )
                        break
                else:
                    raise GenerationError(
                        f"project_timeline re-check: unparsed task record {unit!r}"
                    )
    if start is None or not tasks:
        raise GenerationError("project_timeline re-check: plan not found")
    if any(dep not in tasks for _, deps in tasks.values() for dep in deps):
        raise GenerationError(
            "project_timeline re-check: a prerequisite was not parsed"
        )
    length = {name: value[0] for name, value in tasks.items()}
    for _, name, old, new in sorted(changes):
        if name in length:
            if tasks[name][0] != old:
                raise GenerationError(
                    "project_timeline re-check: revision does not match the plan"
                )
            length[name] = new
    done: dict[str, int] = {}

    def complete(name: str) -> int:
        if name not in done:
            done[name] = (
                max((complete(dep) for dep in tasks[name][1]), default=0) + length[name]
            )
        return done[name]

    total = max(complete(name) for name in tasks)
    finish = start
    left = total - 1
    while left:
        finish += timedelta(days=1)
        if finish.weekday() < 5:
            left -= 1
    if base == "choice":
        hits = [
            index
            for index, option in enumerate(options)
            if _read_date(option) == finish
        ]
        if len(hits) != 1:
            raise GenerationError(
                "project_timeline re-check: finish date not offered once"
            )
        return hits[0]
    if deadline is None:
        raise GenerationError("project_timeline re-check: deadline not found")
    if base == "noul":
        return int(finish <= deadline) if polarity == "by" else int(finish > deadline)
    if finish > deadline:
        return 0
    spare = sum(
        1
        for k in range(1, (deadline - finish).days + 1)
        if (finish + timedelta(days=k)).weekday() < 5
    )
    return 1 if spare == 0 else (2 if spare <= 2 else 3)


def build_project_timeline(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    _check_args("project_timeline", base, target, length)
    plan = _pt_plan(rng, base, target)
    return _attempt_loop(
        "project_timeline", lambda: _pt_draw(rng, base, target, length, plan)
    )


# ---------------------------------------------------------------- directory_route

_DR_TWINS = (
    ("Payments Operations", "Payments Operations Support"),
    ("Customer Insights", "Customer Insight Tools"),
    ("Retail Pricing", "Retail Pricing Analytics"),
    ("Field Services North", "Field Services Northeast"),
    ("Clinical Systems", "Clinical Support Systems"),
    ("Freight Planning", "Freight Planning Systems"),
    ("Supplier Quality", "Supplier Quality Assurance"),
    ("Member Services", "Member Service Design"),
    ("Data Platform", "Data Platforms Engineering"),
    ("Store Operations", "Store Operations Support"),
    ("Partner Marketing", "Partner Marketing Events"),
    ("Product Analytics", "Product Analytics Tooling"),
    ("Digital Channels", "Digital Channel Services"),
    ("Warehouse Systems", "Warehouse Systems Support"),
    ("Brand Studio", "Brand Studio Production"),
    ("Treasury Operations", "Treasury Operations Control"),
)
_DR_OTHER_TEAMS = (
    "Legal Counsel",
    "Facilities",
    "Internal Audit",
    "Talent Acquisition",
    "Security Engineering",
    "Procurement",
    "Corporate Communications",
    "Tax Reporting",
    "Customer Support",
    "Quality Engineering",
    "Sustainability",
    "Research Operations",
    "Workplace Services",
    "Revenue Accounting",
)
_DR_ORGS = (
    "Brightwater Foods",
    "Calder Instruments",
    "Marlowe Health",
    "Orbis Freight",
    "Pinecrest Energy",
    "Quill Publishing",
    "Redfern Analytics",
    "Silverline Retail",
    "Tidewater Insurance",
    "Vantage Logistics",
    "Westbrook Utilities",
    "Ashgrove Pharma",
)
_DR_ITEMS = (
    ("a replacement laptop", 142900),
    ("two ergonomic chairs", 88600),
    ("a conference ticket", 69500),
    ("a design software licence", 54000),
    ("a label printer", 31950),
    ("an external training course", 124000),
    ("lab consumables", 47325),
    ("three monitor arms", 26970),
    ("a survey tool subscription", 90000),
    ("a courier service contract", 215000),
    ("noise-cancelling headsets", 39800),
    ("a portable projector", 61450),
)
_DR_MEMBER_TITLES = (
    "analyst",
    "coordinator",
    "engineer",
    "specialist",
    "associate",
    "planner",
    "officer",
)
_DR_OWNER_TITLES = (
    "finance director",
    "operations director",
    "head of department",
    "programme director",
    "commercial director",
    "division head",
)
_DR_DELEGATE_TITLES = (
    "deputy director",
    "senior manager",
    "finance manager",
    "operations manager",
    "business manager",
)
_DR_POLICY = (
    "Approval policy: a purchase request is approved by the budget owner of the cost centre that the "
    "requester's team is charged to. Team leads confirm the business need but do not approve spending. If the "
    "budget owner has delegated approval for that cost centre, the delegate approves instead for requests dated "
    "within the delegation period, and both the first and the last day of the period count.",
    "How approvals work: find the requester's team, then the cost centre that team is charged to; the budget "
    "owner of that cost centre signs off the request. A delegation notice hands this to the named delegate for "
    "requests dated from its first day to its last day inclusive. Sign-off by a team lead is not an approval.",
    "Spending approval rules: requests are signed off by the budget owner of the requester's team's cost centre, "
    "never by the team lead. Where a delegation is on file for that cost centre and the request date falls "
    "inside the delegation period (start and end dates included), the delegate signs off in the owner's place.",
)
_DR_POLICY_SHORT = (
    "Approval policy: purchases are approved by the budget owner of the cost centre the requester's team is "
    "charged to, not by team leads. A delegate approves instead for requests dated within the delegation "
    "period, first and last days included.",
    "Rule: follow the requester's team to its cost centre; that cost centre's budget owner approves. If a "
    "delegation for that cost centre covers the request date (both end dates count), the delegate approves. "
    "Team leads never approve spending.",
    "Spending is signed off by the budget owner of the requester's team's cost centre, or by the delegate in a "
    "delegation notice when the request date falls within its period, both ends included. Team leads do not "
    "sign off.",
)
_DR_PEOPLE = (
    "{person} \u2014 {title}, {team}",
    "{person} ({title}, {team})",
    "{person}: {title} in {team}",
)
_DR_TEAM = (
    "{team}: led by {lead}; charged to cost centre {code}.",
    "{team} \u2014 lead {lead}, cost centre {code}",
    "The {team} team is led by {lead} and charged to cost centre {code}.",
)
_DR_CC = (
    "Cost centre {code}: budget owner {owner}.",
    "{code} \u2014 budget owner: {owner}",
    "{owner} is the budget owner of cost centre {code}.",
)
_DR_DELEGATE = (
    "{owner} has delegated budget approval for cost centre {code} to {delegate} from {first} to {last}.",
    "Delegation notice: from {first} to {last}, approvals for cost centre {code} go to {delegate} instead of "
    "{owner}.",
    "While {owner} is away ({first} to {last}), {delegate} approves spending on cost centre {code}.",
)
_DR_REQUEST = (
    "Purchase request {rid}, dated {when}: {requester} asks to buy {item} for {amount}.",
    "On {when}, {requester} raised purchase request {rid} for {item} ({amount}).",
    "Request {rid} ({when}) from {requester}: {item}, {amount}.",
)
_DR_PAST = (
    "Request {rid} from {person} ({when}) was approved by {approver}.",
    "{approver} approved {person}'s request {rid} on {when}.",
)
_DR_MOVE = (
    "Directory change: {person} moved from {old} to {new} last quarter.",
    "{person} has joined {new}; the directory entry has been updated.",
)
_DR_HEAD_PEOPLE = ("Staff directory (extract):", "People:", "Directory entries:")
_DR_HEAD_TEAMS = ("Teams and cost centres:", "Team records:", "Teams:")
_DR_HEAD_CC = ("Budget owners:", "Cost centre register:", "Cost centres:")
_DR_QUESTION = {
    "choice": (
        "Who has to approve {requester}'s purchase request {rid}?",
        "Whose approval does purchase request {rid} from {requester} need?",
        "Who is the right approver for {requester}'s request {rid}?",
    ),
    "noul": (
        "Is {named} the right person to approve {requester}'s purchase request {rid}?",
        "Should purchase request {rid} from {requester} go to {named} for approval?",
    ),
}
_DR_CONCLUDE_CHOICE = (
    "{approver} must approve {requester}'s request",
    "{requester}'s request needs {approver}'s approval",
    "the approval for request {rid} has to come from {approver}",
)
_DR_CONCLUDE_NOUL = (
    (
        "{named} is not the right person to approve {requester}'s request",
        "{named} is the right person to approve {requester}'s request",
    ),
    (
        "request {rid} should not go to {named} for approval",
        "request {rid} should go to {named} for approval",
    ),
)
_DR_RATIONALE = (
    "on {when}, sign-off for {unit} rests with {approver}",
    "{approver} signs off {unit} dated {when}",
    "{approver} holds sign-off for {unit} on {when}",
)
_DR_FILLER = (
    "The finance system will be unavailable on Saturday morning for maintenance. Requests saved as drafts are "
    "kept, and nothing needs to be resubmitted once the system is back.",
    "{name} has updated the supplier list: two office furniture suppliers were added and one courier was removed "
    "after its contract ended. The list is in the procurement folder.",
    "Expense receipts: photograph the whole receipt, including the date and the supplier's name, before "
    "uploading it. Blurry photos are the most common reason a claim bounces back.",
    "Reminder from {name}: request titles should say what is being bought, not just 'misc', so that the monthly "
    "spending report stays readable for everyone who uses it.",
    "Supplier onboarding takes a little while because bank details are checked by phone, so new suppliers should "
    "be set up well before the first order. {name2} can start the paperwork.",
    "The finance helpdesk is staffed during office hours. {name} covers lunchtimes, and questions sent after "
    "hours are answered the next morning.",
    "Budget reports for the quarter are in the shared folder. They show commitments as well as actual spending, "
    "and {name} is happy to walk anyone through the layout.",
    "Company cards are meant for travel and small online purchases. A lost card must be reported to the bank's "
    "hotline straight away, and {name2} will arrange a replacement.",
    "The {org} intranet has a new search page. It covers policies, forms and the staff directory, and {name} "
    "would like feedback on anything that is hard to find.",
    "Asset tags: equipment worth keeping track of gets a tag when it arrives. {name} applies them in the post "
    "room, so please do not unpack large deliveries at your desk.",
    "Invoices from suppliers should be sent to the accounts payable mailbox, not to individuals. {name} checks "
    "the mailbox every day and matches each invoice to its order before it is paid.",
    "The {org} sustainability group asks that packaging from deliveries be flattened and taken to the recycling "
    "cage. {name2} has also started a shelf for reusable padded envelopes.",
    "Travel bookings made through the portal show up in the monthly report automatically. Bookings made "
    "elsewhere have to be entered by hand, which is slow, so {name} asks everyone to use the portal.",
    "The procurement team is running a short survey about the ordering screens. It takes a couple of minutes, "
    "and {name} will share the results along with the changes they lead to.",
    "Stationery orders go out once a week. Add what you need to the shared list by Thursday lunchtime and "
    "{name2} will place a single order, which keeps delivery charges down.",
    "A reminder that the directory lists work phone numbers only. {name} asks people to keep their own entry "
    "tidy, including the desk location, so that visitors can find them.",
    "The {org} audit team will visit next quarter as part of the routine cycle. They mostly look at how "
    "records are kept, and {name} will let teams know if anything needs preparing.",
    "Office equipment that is no longer needed can be offered on the internal marketplace page before it is "
    "recycled. {name2} collects unclaimed items from the {floor} floor once a month.",
)


def _dr_code(rng: random.Random, taken: set[str]) -> str:
    while True:
        code = f"CC-{rng.randint(1, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)}"
        if code not in taken and len(set(code[3:])) >= 3:
            return code


def _dr_twin_code(code: str) -> str:
    digits = code[3:]
    if digits[2] != digits[3]:
        return "CC-" + digits[:2] + digits[3] + digits[2]
    return "CC-" + digits[0] + digits[2] + digits[1] + digits[3]


def _dr_window(rng: random.Random, when: date, active: bool) -> tuple[date, date]:
    span = rng.randint(6, 24)
    if active:
        first = when - timedelta(days=rng.randint(1, span - 1))
        return first, first + timedelta(days=span)
    if rng.random() < 0.65:
        last = when - timedelta(days=rng.randint(2, 20))
        return last - timedelta(days=span), last
    first = when + timedelta(days=rng.randint(2, 20))
    return first, first + timedelta(days=span)


def _dr_plan(rng: random.Random) -> dict[str, Any]:
    plan = {
        "active": rng.random() < 0.5,
        "twin_delegation": rng.random() < 0.5,
        "twin_active": rng.random() < 0.5,
        "lead_delegate": rng.random() < 0.4,
        "mechanism": pick(rng, ("short_chain", "entity_swap", "stale_value")),
    }
    plan["unit"] = (
        "team" if plan["mechanism"] == "short_chain" or rng.random() < 0.5 else "code"
    )
    return plan


def _dr_draw(
    rng: random.Random, base: str, target: int | None, length: str, plan: dict[str, Any]
) -> F1World:
    long = length == "long"
    twin_pair = list(pick(rng, _DR_TWINS))
    rng.shuffle(twin_pair)
    team, twin_team = twin_pair
    others = rng.sample(_DR_OTHER_TEAMS, 5 if long else rng.choice((0, 1)))
    names = people(rng, 21 if long else 7 + 2 * len(others) + rng.choice((0, 1)))
    requester, lead, owner, delegate, twin_owner, twin_lead, *rest = names
    twin_delegate = rest.pop()
    teams = [team, twin_team] + others
    leads = {team: lead, twin_team: twin_lead}
    taken: set[str] = set()
    code = _dr_code(rng, taken)
    twin_code = _dr_twin_code(code)
    taken |= {code, twin_code}
    codes = {team: code, twin_team: twin_code}
    owners = {code: owner, twin_code: twin_owner}
    pool = list(rest)
    for other in others:
        leads[other] = pool.pop()
        codes[other] = _dr_code(rng, taken)
        taken.add(codes[other])
    for other in others:
        owners[codes[other]] = pool.pop() if pool else leads[pick(rng, others)]
    if len(set(owners.values())) != len(owners):
        raise _Redraw("an owner holds two cost centres")
    when = _weekday_on_or_after(_draw_day(rng))
    style = pick(rng, DATE_STYLES)
    fmt = lambda value: fmt_date(value, style)  # noqa: E731
    first, last = _dr_window(rng, when, plan["active"])
    delegations = [(owner, code, delegate, first, last)]
    distractor_delegations = []
    if plan["twin_delegation"]:
        t_first, t_last = _dr_window(rng, when, plan["twin_active"])
        distractor_delegations.append(
            (twin_owner, twin_code, twin_delegate, t_first, t_last)
        )
    other_codes = [codes[other] for other in others]
    if plan["lead_delegate"] and other_codes:
        o_code = pick(rng, other_codes)
        o_first, o_last = _dr_window(rng, when, rng.random() < 0.5)
        distractor_delegations.append((owners[o_code], o_code, lead, o_first, o_last))
    if long:
        for o_code in rng.sample(other_codes, 2):
            if any(entry[1] == o_code for entry in distractor_delegations):
                continue
            o_first, o_last = _dr_window(rng, when, rng.random() < 0.5)
            helper = pick(rng, [leads[t] for t in others if codes[t] != o_code])
            distractor_delegations.append(
                (owners[o_code], o_code, helper, o_first, o_last)
            )

    def approver(
        cost_centre: str, entries: Sequence[tuple[str, str, str, date, date]]
    ) -> str:
        for d_owner, d_code, d_delegate, d_first, d_last in entries:
            if (
                d_code == cost_centre
                and d_owner == owners[cost_centre]
                and d_first <= when <= d_last
            ):
                return d_delegate
        return owners[cost_centre]

    every = delegations + distractor_delegations
    gold_person = approver(code, every)
    outputs = {
        "short_chain": lead,
        "entity_swap": approver(twin_code, every),
        "stale_value": owner if plan["active"] else delegate,
    }
    mechanism = plan["mechanism"]
    wrong_person = outputs[mechanism]
    if wrong_person == gold_person or len(set(outputs.values()) | {gold_person}) != 4:
        raise _Redraw("mechanism outputs collide")
    # records
    titles = {
        requester: pick(rng, _DR_MEMBER_TITLES),
        lead: "team lead",
        twin_lead: "team lead",
    }
    person_team = {requester: team, lead: team, twin_lead: twin_team}
    for other in others:
        titles[leads[other]] = "team lead"
        person_team[leads[other]] = other
    for person in (owner, twin_owner) + tuple(owners[c] for c in other_codes):
        titles.setdefault(person, pick(rng, _DR_OWNER_TITLES))
        person_team.setdefault(person, pick(rng, others + [twin_team]))
    for person in (delegate, twin_delegate):
        if person == twin_delegate and not plan["twin_delegation"] and not long:
            continue
        titles.setdefault(person, pick(rng, _DR_DELEGATE_TITLES))
        person_team.setdefault(person, pick(rng, others or [twin_team, team]))
    extra = [p for p in names if p not in titles and p != twin_delegate]
    for person in extra:
        titles[person] = pick(rng, _DR_MEMBER_TITLES)
        person_team[person] = pick(rng, teams)
    if long:
        colleague = pick(rng, [p for p in extra] or [twin_delegate])
        person_team[colleague] = pick(rng, [team, twin_team])
    people_template = pick(rng, _DR_PEOPLE)
    listing = list(titles)
    if not long:
        senior = [
            p for p in (owner, delegate, twin_owner, twin_delegate) if p in titles
        ]
        keep = {
            requester,
            lead,
            twin_lead,
            *extra,
            *(leads[t] for t in others),
            *rng.sample(senior, 2),
        }
        listing = [p for p in listing if p in keep]
    rng.shuffle(listing)
    people_lines = {
        p: people_template.format(person=p, title=titles[p], team=person_team[p])
        for p in listing
    }
    team_template = pick(rng, _DR_TEAM)
    team_order = list(teams)
    rng.shuffle(team_order)
    team_lines = {
        t: team_template.format(team=t, lead=leads[t], code=codes[t])
        for t in team_order
    }
    cc_template = pick(rng, _DR_CC)
    cc_order = list(owners)
    rng.shuffle(cc_order)
    cc_lines = {c: cc_template.format(code=c, owner=owners[c]) for c in cc_order}
    bullet = pick(rng, ("- ", "\u2022 ", ""))

    def table(head_bank: Sequence[str], lines: Iterable[str]) -> str:
        return pick(rng, head_bank) + "\n" + "\n".join(bullet + line for line in lines)

    org = pick(rng, _DR_ORGS)
    rid = f"PR-{rng.randint(10000, 99999)}"
    item, amount = pick(rng, _DR_ITEMS)
    request = pick(rng, _DR_REQUEST).format(
        rid=rid,
        when=fmt(when),
        requester=requester,
        item=item,
        amount=_cents(amount),
    )
    delegation_template = rng.randrange(len(_DR_DELEGATE))

    def notice(entry: tuple[str, str, str, date, date]) -> str:
        d_owner, d_code, d_delegate, d_first, d_last = entry
        return _DR_DELEGATE[delegation_template].format(
            owner=d_owner,
            code=d_code,
            delegate=d_delegate,
            first=fmt(d_first),
            last=fmt(d_last),
        )

    relevant_notice = notice(delegations[0])
    policy = f"{org} \u2014 " + pick(rng, _DR_POLICY if long else _DR_POLICY_SHORT)
    evidence = [
        policy,
        table(_DR_HEAD_PEOPLE, people_lines.values()),
        table(_DR_HEAD_TEAMS, team_lines.values()),
        table(_DR_HEAD_CC, cc_lines.values()),
        relevant_notice,
        request,
    ]
    distractors = [notice(entry) for entry in distractor_delegations]
    if long:
        for _ in range(3):
            o_team = pick(rng, others)
            o_person = pick(
                rng,
                [p for p, t in person_team.items() if t == o_team] or [leads[o_team]],
            )
            o_when = when - timedelta(days=rng.randint(20, 90))
            distractors.append(
                pick(rng, _DR_PAST).format(
                    rid=f"PR-{rng.randint(10000, 99999)}",
                    person=o_person,
                    when=fmt(o_when),
                    approver=owners[codes[o_team]],
                )
            )
        mover = pick(rng, extra) if extra else twin_delegate
        distractors.append(
            pick(rng, _DR_MOVE).format(
                person=mover,
                old=pick(rng, [t for t in others if t != person_team[mover]] or others),
                new=person_team[mover],
            )
        )
    elif others and rng.random() < 0.5:
        o_team = others[0]
        distractors.append(
            pick(rng, _DR_PAST).format(
                rid=f"PR-{rng.randint(10000, 99999)}",
                person=leads[o_team],
                when=fmt(when - timedelta(days=rng.randint(20, 90))),
                approver=owners[codes[o_team]],
            )
        )
    rng.shuffle(distractors)
    decisive = [
        people_lines[requester],
        team_lines[team],
        cc_lines[code],
        relevant_notice,
        request,
    ]

    # question, options and claims
    if base == "choice":
        pool = [lead, outputs["entity_swap"], owner, delegate, twin_owner, twin_lead]
        extras = [
            p for p in dict.fromkeys(pool) if p not in (gold_person, wrong_person)
        ]
        rng.shuffle(extras)
        values = [gold_person, wrong_person] + extras[: rng.randint(1, 3)]
        rng.shuffle(values)
        choices = tuple(values)
        gold = values.index(gold_person)
        wrong_answer = values.index(wrong_person)
        named = None
        question = pick(rng, _DR_QUESTION["choice"]).format(
            requester=requester, rid=rid
        )
    else:
        choices = ()
        gold = target
        wrong_answer = 1 - target
        named = gold_person if target == 1 else wrong_person
        question = pick(rng, _DR_QUESTION["noul"]).format(
            requester=requester, rid=rid, named=named
        )
    conclusion_variant = rng.randrange(
        len(_DR_CONCLUDE_CHOICE) if base == "choice" else len(_DR_CONCLUDE_NOUL)
    )
    rationale_variant = rng.randrange(len(_DR_RATIONALE))
    true_unit = (
        f"the {team} team's requests"
        if plan["unit"] == "team"
        else f"requests on cost centre {code}"
    )
    swap_unit = (
        f"the {twin_team} team's requests"
        if plan["unit"] == "team"
        else f"requests on cost centre {twin_code}"
    )

    def claim(answer: int, person: str, unit: str, label: str) -> Claim:
        values = {
            "requester": requester,
            "rid": rid,
            "approver": person,
            "named": named,
        }
        if base == "choice":
            conclusion = _DR_CONCLUDE_CHOICE[conclusion_variant].format(**values)
        else:
            conclusion = _DR_CONCLUDE_NOUL[conclusion_variant][answer].format(**values)
        rationale = _DR_RATIONALE[rationale_variant].format(
            unit=unit, when=fmt(when), approver=person
        )
        return Claim(
            answer=answer, conclusion=conclusion, rationale=rationale, mechanism=label
        )

    right = claim(gold, gold_person, true_unit, "correct")
    wrong = claim(
        wrong_answer,
        wrong_person,
        swap_unit if mechanism == "entity_swap" else true_unit,
        mechanism,
    )
    text = "\n\n".join(evidence + distractors)
    recheck = _dr_recheck(text, requester, rid, base, choices, named)
    if (
        _dr_recheck("\n\n".join(evidence), requester, rid, base, choices, named)
        != recheck
    ):
        raise GenerationError("directory_route: a distractor changed the answer")
    need = (
        max(0, LONG_POOL_CHARS - sum(len(block) + 2 for block in distractors))
        if long
        else 0
    )
    filler_names = people(rng, 6, exclude=names)
    filler = (
        _filler(rng, _DR_FILLER, _filler_slots(rng, filler_names, {"org": org}), need)
        if need
        else ()
    )
    facts = {
        "org": org,
        "request": {
            "id": rid,
            "date": when.isoformat(),
            "requester": requester,
            "item": item,
            "cents": amount,
        },
        "team": team,
        "twin_team": twin_team,
        "code": code,
        "twin_code": twin_code,
        "teams": {t: {"lead": leads[t], "code": codes[t]} for t in teams},
        "owners": dict(owners),
        "delegations": [
            {
                "owner": e[0],
                "code": e[1],
                "delegate": e[2],
                "first": e[3].isoformat(),
                "last": e[4].isoformat(),
            }
            for e in every
        ],
        "relevant_delegation_active": plan["active"],
        "gold_person": gold_person,
        "wrong": {"mechanism": mechanism, "person": wrong_person, "unit": plan["unit"]},
        "outputs": outputs,
        "named": named,
        "names": sorted(set(names) | (set(filler_names) if filler else set())),
    }
    return F1World(
        kind="directory_route",
        base=base,
        subject=f"the approval route for request {rid}",
        question=question,
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=filler,
        right=right,
        wrong=wrong,
        decisive=tuple(decisive),
        facts=facts,
        variant=f"p{_DR_PEOPLE.index(people_template)}-t{_DR_TEAM.index(team_template)}"
        f"-c{_DR_CC.index(cc_template)}-d{delegation_template}-k{conclusion_variant}-q{rationale_variant}",
        roles=(
            "procurement officer",
            "finance business partner",
            "office manager",
            "team administrator",
            "purchasing coordinator",
            "budget analyst",
        ),
    )


_DR_RX_PEOPLE = [
    _rx(t, person=_PERSON_RX, title=r"[a-z][a-z ]*", team=r"[A-Z][\w ]*")
    for t in _DR_PEOPLE
]
_DR_RX_TEAM = [
    _rx(t, team=r"[A-Z][\w ]*", lead=_PERSON_RX, code=r"CC-\d{4}") for t in _DR_TEAM
]
_DR_RX_CC = [_rx(t, code=r"CC-\d{4}", owner=_PERSON_RX) for t in _DR_CC]
_DR_RX_DELEGATE = [
    _rx(
        t,
        owner=_PERSON_RX,
        code=r"CC-\d{4}",
        delegate=_PERSON_RX,
        first=_DATE_RX,
        last=_DATE_RX,
    )
    for t in _DR_DELEGATE
]
_DR_RX_REQUEST = [
    _rx(
        t,
        rid=r"PR-\d{5}",
        when=_DATE_RX,
        requester=_PERSON_RX,
        item=r"[^,()]+?",
        amount=_MONEY_RX,
    )
    for t in _DR_REQUEST
]


def _dr_recheck(
    text: str,
    requester: str,
    rid: str,
    base: str,
    options: Sequence[str],
    named: str | None,
) -> int:
    member: dict[str, str] = {}
    charged: dict[str, str] = {}
    budget: dict[str, str] = {}
    notices: list[tuple[str, str, str, int, int]] = []
    dated = None
    for units in _units(text):
        for unit in units:
            for pattern in _DR_RX_PEOPLE:
                match = pattern.fullmatch(unit)
                if match:
                    member[match["person"]] = match["team"]
                    break
            for pattern in _DR_RX_TEAM:
                match = pattern.fullmatch(unit)
                if match:
                    charged[match["team"]] = match["code"]
                    break
            for pattern in _DR_RX_CC:
                match = pattern.fullmatch(unit)
                if match:
                    budget[match["code"]] = match["owner"]
                    break
            for pattern in _DR_RX_DELEGATE:
                match = pattern.fullmatch(unit)
                if match:
                    notices.append(
                        (
                            match["owner"],
                            match["code"],
                            match["delegate"],
                            _read_date(match["first"]).toordinal(),
                            _read_date(match["last"]).toordinal(),
                        )
                    )
                    break
            for pattern in _DR_RX_REQUEST:
                match = pattern.fullmatch(unit)
                if match and match["rid"] == rid and match["requester"] == requester:
                    dated = _read_date(match["when"]).toordinal()
                    break
    if dated is None or requester not in member:
        raise GenerationError(
            "directory_route re-check: request or requester not found"
        )
    cost_centre = charged[member[requester]]
    holder = budget[cost_centre]
    in_force = [
        n
        for n in notices
        if n[1] == cost_centre and n[0] == holder and n[3] <= dated <= n[4]
    ]
    if len(in_force) > 1:
        raise GenerationError("directory_route re-check: overlapping delegations")
    person = in_force[0][2] if in_force else holder
    if base == "choice":
        if options.count(person) != 1:
            raise GenerationError("directory_route re-check: approver not offered once")
        return options.index(person)
    return int(person == named)


def build_directory_route(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    _check_args("directory_route", base, target, length)
    plan = _dr_plan(rng)
    return _attempt_loop(
        "directory_route", lambda: _dr_draw(rng, base, target, length, plan)
    )


# ---------------------------------------------------------------- rubric_pick

_RP_CONTRACTS = (
    "office cleaning contract",
    "payroll software tender",
    "staff catering contract",
    "website redesign tender",
    "fleet maintenance contract",
    "translation services framework",
    "security guarding contract",
    "print management contract",
    "lab equipment servicing contract",
    "event production contract",
    "grounds maintenance contract",
    "helpdesk outsourcing tender",
)
_RP_SUPPLIERS = (
    "Birchwood Services",
    "Crescent Facilities",
    "Dunmore Systems",
    "Eastgate Partners",
    "Foxglove Studio",
    "Greyhawk Consulting",
    "Hartley Group",
    "Ironbridge Works",
    "Juniper Digital",
    "Kingsway Supplies",
    "Larchmont Solutions",
    "Meadowbank Contracting",
    "Northwind Services",
    "Oakridge Partners",
    "Palisade Group",
    "Rookwood Technical",
    "Saltmarsh Associates",
    "Thistle Logistics",
    "Upland Engineering",
    "Vellum Creative",
)
_RP_CRITERIA = (
    "Quality",
    "Price",
    "Delivery",
    "Experience",
    "Sustainability",
    "Support",
    "Security",
    "Innovation",
    "Social Value",
    "Implementation",
)
_RP_WEIGHTS = {
    3: (
        (50, 30, 20),
        (40, 35, 25),
        (45, 35, 20),
        (40, 40, 20),
        (35, 35, 30),
        (60, 25, 15),
        (50, 25, 25),
        (45, 30, 25),
        (55, 25, 20),
        (40, 30, 30),
    ),
    4: (
        (40, 30, 20, 10),
        (30, 30, 20, 20),
        (35, 25, 25, 15),
        (40, 25, 20, 15),
        (30, 25, 25, 20),
        (35, 30, 20, 15),
        (45, 25, 15, 15),
        (30, 30, 25, 15),
    ),
}
# kind -> (higher value is better, attribute phrase, inclusive requirement, exclusive requirement, thresholds)
_RP_REQUIREMENTS = {
    "warranty": (
        True,
        "{v} months of warranty",
        "at least {t} months of warranty",
        "more than {t} months of warranty",
        (12, 18, 24, 36),
    ),
    "experience": (
        True,
        "{v} years of relevant experience",
        "at least {t} years of relevant experience",
        "more than {t} years of relevant experience",
        (3, 5, 8),
    ),
    "response": (
        False,
        "a {v}-hour response time",
        "a response time of no more than {t} hours",
        "a response time of under {t} hours",
        (4, 6, 8, 24),
    ),
    "delivery": (
        False,
        "a {v}-week delivery time",
        "a delivery time of no more than {t} weeks",
        "a delivery time of less than {t} weeks",
        (6, 8, 10, 12),
    ),
    "staff": (
        True,
        "{v} certified technicians",
        "at least {t} certified technicians",
        "more than {t} certified technicians",
        (3, 4, 5, 6),
    ),
}
_RP_BRIEF = (
    "Evaluation of proposals for the {contract}. Each proposal is scored from 1 to 10 on {criteria}.",
    "The panel scored every proposal for the {contract} from 1 to 10 on {criteria}.",
    "Proposals for the {contract} are marked out of 10 on each of {criteria}.",
)
_RP_WEIGHT_SENTENCE = (
    "The weights are {weights}.",
    "Criteria weights: {weights}.",
)
_RP_METHOD = (
    "A proposal's weighted score is the sum of each criterion score multiplied by its weight, so the maximum is "
    "10.00.",
    "The weighted score multiplies each criterion score by its weight and adds the results (out of 10.00).",
)
_RP_MANDATORY = (
    "Mandatory requirement: every proposal must offer {requirement}.",
    "Proposals must offer {requirement}; any that do not are excluded.",
    "A proposal is only eligible if it offers {requirement}.",
)
_RP_AWARD = (
    "A proposal that fails the mandatory requirement is excluded whatever its score, and the contract goes to "
    "the eligible proposal with the highest weighted score.",
    "Excluded proposals cannot win, however well they score; the award goes to the highest weighted score "
    "among the eligible ones.",
)
_RP_SHEET_HEAD = (
    "Scores for the {contract}:",
    "Panel scores, {contract}:",
    "Score sheet for the {contract}:",
)
_RP_LINE = (
    "{name}: {scores}; {attr}.",
    "{name} scored {scores}, and offers {attr}.",
    "{name} \u2014 {scores} \u2014 {attr}",
)
_RP_BAND_LABELS = (
    ("Band D", "Band C", "Band B", "Band A"),
    ("Limited", "Adequate", "Good", "Excellent"),
    ("Tier 4", "Tier 3", "Tier 2", "Tier 1"),
)
_RP_BOUNDS = (
    (500, 650, 800),
    (450, 600, 750),
    (550, 700, 850),
    (500, 600, 750),
    (400, 550, 700),
)
_RP_BAND_LEAD = (
    "Score bands for reporting: ",
    "The evaluation guide groups weighted scores into bands: ",
)
_RP_BANDS_LOWER = (
    "{l0}: below {b1}; {l1}: from {b1} up to but not including {b2}; {l2}: from {b2} up to but not "
    "including {b3}; {l3}: {b3} or higher."
)
_RP_BANDS_UPPER = (
    "{l0}: {b1} or lower; {l1}: above {b1} up to and including {b2}; {l2}: above {b2} up to and "
    "including {b3}; {l3}: above {b3}."
)
_RP_QUESTION = {
    "choice": (
        "Which proposal should be awarded the {contract}?",
        "Which bid should win the {contract}?",
        "Which proposal should the panel select for the {contract}?",
    ),
    "noul": (
        "Does {named} have the highest weighted score among the eligible proposals for the {contract}?",
        "Is {named} the eligible proposal with the highest weighted score for the {contract}?",
    ),
    "score": (
        "In which band of the evaluation guide does {named}'s weighted score fall?",
        "Which reporting band does {named}'s weighted score for the {contract} belong to?",
    ),
}
_RP_CONCLUDE_CHOICE = (
    "{winner} should be awarded the {contract}",
    "the {contract} should go to {winner}",
    "{winner} is the proposal the panel should select",
)
_RP_CONCLUDE_NOUL = (
    (
        "{named} does not have the highest weighted score among the eligible proposals",
        "{named} has the highest weighted score among the eligible proposals",
    ),
    (
        "{named} is not the strongest eligible proposal",
        "{named} is the strongest eligible proposal",
    ),
)
_RP_CONCLUDE_SCORE = (
    "{named}'s weighted score falls in {label}",
    "{named} belongs in {label}",
)
_RP_RATIONALE = (
    "{winner} posts the best weighted score among the eligible proposals, {top} against {next} for {runner}",
    "{winner} tops the eligible field on {top}, ahead of {runner} on {next}",
    "among the eligible proposals {winner} scores {top} and the next best, {runner}, scores {next}",
)
_RP_RATIONALE_SCORE = (
    "{named}'s weighted score comes to {total}",
    "applying the weights to {named}'s scores gives {total}",
    "{named} totals {total} on the weighted scale",
)
_RP_DISTRACT = (
    "A submission from {late} arrived after the closing date and was not evaluated.",
    "{name} chaired the {contract} panel; declarations of interest were collected and none were relevant.",
    "{supplier} asked whether the {contract} could start a week later. The panel noted the question, which has "
    "no effect on the scoring.",
    "References for {supplier} were followed up by {name} and came back without concerns.",
)
_RP_FILLER = (
    "The panel met in the {room}. {name} reminded members to keep their individual notes until the award is "
    "confirmed, in case a bidder asks for feedback.",
    "Moderation: scores were discussed criterion by criterion until the panel agreed a single mark for each. "
    "{name2} recorded the agreed marks on the sheet.",
    "Unsuccessful bidders receive a short feedback letter. {name} drafts the letters from the panel notes and "
    "shares them before they go out.",
    "The standstill period follows the award notice. During that time no contract is signed, and questions from "
    "bidders are answered in writing by {name}.",
    "Site visits were offered to all bidders on the same terms. {name2} accompanied each visit and answered only "
    "factual questions about access and working hours.",
    "Conflict-of-interest forms are kept with the tender file. {name} checked that every panel member had "
    "signed one before scoring began.",
    "The tender documents, clarification log and score sheets are archived together. {name} will close the "
    "file once the contract is signed.",
    "Bidders were told in advance how the scoring works. {name2} answered two general questions about the "
    "format of the response and posted the answers for everyone.",
    "Budget holders were briefed on the timetable for the award. {name} will confirm the start date with the "
    "successful bidder once the paperwork is complete.",
    "Lessons for the next tender: publish the question deadline more prominently and allow a little more time "
    "for site visits. {name} has added both to the checklist.",
)


def _rp_hundredths(value: int) -> str:
    return f"{value // 100}.{value % 100:02d}"


def _rp_total(weights: Sequence[int], scores: Sequence[int]) -> int:
    return sum(w * s for w, s in zip(weights, scores))


def _rp_band(total: int, bounds: Sequence[int], scheme: str) -> int:
    if scheme == "lower":
        return sum(1 for bound in bounds if total >= bound)
    return sum(1 for bound in bounds if total > bound)


def _rp_near(total: int, bounds: Sequence[int]) -> str:
    for bound in bounds:
        if total == bound:
            return "exact"
        if total == bound - 5:
            return "just_below"
        if total == bound + 5:
            return "just_above"
    return "none" if all(abs(total - bound) >= 15 for bound in bounds) else "close"


def _rp_plan(rng: random.Random, base: str, target: int | None) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "inclusive": rng.random() < 0.5,
        "scheme": rng.choice(("lower", "upper")),
    }
    if base == "score":
        wrong_level = _pt_wrong_level(rng, target)
        plan["wrong_level"] = wrong_level
        up = wrong_level > target
        if abs(wrong_level - target) == 1 and rng.random() < 0.85:
            plan["mechanism"] = "boundary_misread"
            lower = plan["scheme"] == "lower"
            plan["near"] = (
                ("just_below" if lower else "exact")
                if up
                else ("exact" if lower else "just_above")
            )
        else:
            plan["mechanism"] = "arithmetic_slip"
            exact_ok = target >= 1 if plan["scheme"] == "lower" else target <= 2
            kinds = ["none"] + (["exact"] if exact_ok else [])
            kinds += (["just_below"] if target <= 2 else []) + (
                ["just_above"] if target >= 1 else []
            )
            plan["near"] = pick(rng, kinds)
        return plan
    # the boundary proposal tops the eligible field in 1 of 3 worlds, as each of the
    # three eligible proposals would by chance; the disqualified one tops it in half
    plan["boundary_top"] = rng.random() < 1 / 3
    plan["disqualified_top"] = rng.random() < 0.5
    options = [("arithmetic_slip", 1.0)]
    if plan["disqualified_top"]:
        options.append(("overlooked_condition", 8.0))
    if plan["boundary_top"]:
        options.append(("boundary_misread", 30.0))
    plan["mechanism"] = rng.choices([o[0] for o in options], [o[1] for o in options])[0]
    return plan


def _rp_requirement(rng: random.Random, inclusive: bool, count: int) -> dict[str, Any]:
    kind = pick(rng, sorted(_RP_REQUIREMENTS))
    higher, _, _, _, thresholds = _RP_REQUIREMENTS[kind]
    threshold = pick(rng, thresholds)
    step = max(1, threshold // 6)
    sign = 1 if higher else -1
    fail = threshold - sign * step * rng.randint(1, 3)
    passes = [threshold + sign * step * rng.randint(1, 4) for _ in range(count - 2)]
    if min([fail, *passes]) < 1:
        raise _Redraw("attribute below one")
    return {
        "kind": kind,
        "threshold": threshold,
        "inclusive": inclusive,
        "higher": higher,
        "boundary": threshold,
        "fail": fail,
        "passes": passes,
    }


def _rp_complies(
    value: int, req: dict[str, Any], inclusive: bool | None = None
) -> bool:
    inclusive = req["inclusive"] if inclusive is None else inclusive
    if req["higher"]:
        return value >= req["threshold"] if inclusive else value > req["threshold"]
    return value <= req["threshold"] if inclusive else value < req["threshold"]


def _rp_leader(totals: dict[str, int], pool: Iterable[str]) -> tuple[str, str]:
    ranked = sorted(pool, key=lambda name: -totals[name])
    if len(ranked) < 2 or totals[ranked[0]] == totals[ranked[1]]:
        raise _Redraw("no clear leader")
    return ranked[0], ranked[1]


def _rp_draw_choice(rng, base, target, length, plan) -> F1World:
    count_criteria = (
        rng.choice((3, 3, 4)) if length == "short" else rng.choice((3, 4, 4))
    )
    weights = list(pick(rng, _RP_WEIGHTS[count_criteria]))
    rng.shuffle(weights)
    criteria = rng.sample(_RP_CRITERIA, count_criteria)
    names = rng.sample(_RP_SUPPLIERS, 4)
    boundary, disqualified, pass_a, pass_b = names
    req = _rp_requirement(rng, plan["inclusive"], 4)
    values = {
        boundary: req["boundary"],
        disqualified: req["fail"],
        pass_a: req["passes"][0],
        pass_b: req["passes"][1],
    }
    eligible = [n for n in names if _rp_complies(values[n], req)]
    pairs = [
        (i, j)
        for i in range(count_criteria)
        for j in range(i + 1, count_criteria)
        if weights[i] != weights[j]
    ]
    mechanism = plan["mechanism"]
    for _ in range(400):
        scores = {n: [rng.randint(3, 10) for _ in criteria] for n in names}
        totals = {n: _rp_total(weights, scores[n]) for n in names}
        if len(set(totals.values())) != 4:
            continue
        winner, runner = _rp_leader(totals, eligible)
        if plan["disqualified_top"] != (totals[disqualified] > totals[winner]):
            continue
        if plan["inclusive"]:
            if plan["boundary_top"] != (winner == boundary):
                continue
        elif plan["boundary_top"] != (totals[boundary] > totals[winner]):
            continue
        if mechanism == "overlooked_condition":
            slip_pool = names
            slip_totals = totals
            slip_detail = "requirement_ignored"
        elif mechanism == "boundary_misread":
            slip_pool = [
                n for n in names if _rp_complies(values[n], req, not req["inclusive"])
            ]
            slip_totals = totals
            slip_detail = (
                "inclusive_read_as_exclusive"
                if req["inclusive"]
                else "exclusive_read_as_inclusive"
            )
        else:
            if not pairs:
                raise _Redraw("no weights to swap")
            i, j = pick(rng, pairs)
            swapped = list(weights)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            slip_pool = eligible
            slip_totals = {n: _rp_total(swapped, scores[n]) for n in names}
            slip_detail = f"weights_swapped:{criteria[i]}<->{criteria[j]}"
        try:
            slip_winner, slip_runner = _rp_leader(slip_totals, slip_pool)
        except _Redraw:
            continue
        if slip_winner != winner:
            break
    else:
        raise _Redraw("no score table fits the plan")
    return _rp_render(
        rng,
        base,
        target,
        length,
        plan,
        criteria,
        weights,
        names,
        values,
        req,
        scores,
        totals,
        (winner, runner),
        (slip_winner, slip_runner, slip_totals, slip_detail),
        None,
    )


def _rp_draw_score(rng, base, target, length, plan) -> F1World:
    count_criteria = rng.choice((3, 3, 4)) if length == "short" else rng.choice((3, 4))
    weights = list(pick(rng, _RP_WEIGHTS[count_criteria]))
    rng.shuffle(weights)
    criteria = rng.sample(_RP_CRITERIA, count_criteria)
    count = rng.choice((3, 4)) if length == "short" else 4
    names = rng.sample(_RP_SUPPLIERS, count)
    req = _rp_requirement(rng, plan["inclusive"], count)
    roster = [req["boundary"], req["fail"], *req["passes"]]
    values = dict(zip(names, roster))
    named = pick(rng, names)
    bounds = pick(rng, _RP_BOUNDS)
    scheme = plan["scheme"]
    wrong_level = plan["wrong_level"]
    pairs = [
        (i, j)
        for i in range(count_criteria)
        for j in range(i + 1, count_criteria)
        if weights[i] != weights[j]
    ]
    grid = [()]
    for _ in criteria:
        grid = [prefix + (s,) for prefix in grid for s in range(1, 11)]
    rng.shuffle(
        grid
    )  # the first fit in a shuffled grid is a uniform draw among all fits
    found = None
    for vector in grid:
        total = _rp_total(weights, vector)
        if (
            _rp_band(total, bounds, scheme) != target
            or _rp_near(total, bounds) != plan["near"]
        ):
            continue
        if plan["mechanism"] == "boundary_misread":
            if plan["near"] == "exact":
                slipped = _rp_band(
                    total, bounds, "upper" if scheme == "lower" else "lower"
                )
            else:
                slipped = _rp_band(
                    total + (5 if plan["near"] == "just_below" else -5), bounds, scheme
                )
            if slipped == wrong_level:
                found = (vector, None)
                break
            continue
        swaps = [
            p
            for p in pairs
            if _rp_band(_rp_total(_rp_swap(weights, p), vector), bounds, scheme)
            == wrong_level
        ]
        if swaps:
            found = (vector, pick(rng, swaps))
            break
    if found is None:
        raise _Redraw("no score vector fits the band plan")
    vector, swap = found
    scores = {n: [rng.randint(2, 10) for _ in criteria] for n in names}
    scores[named] = list(vector)
    totals = {n: _rp_total(weights, scores[n]) for n in names}
    if plan["mechanism"] == "boundary_misread":
        slip_total = totals[named]
        slip_detail = (
            "boundary_inclusivity" if plan["near"] == "exact" else "treated_as_boundary"
        )
    else:
        slip_total = _rp_total(_rp_swap(weights, swap), vector)
        slip_detail = f"weights_swapped:{criteria[swap[0]]}<->{criteria[swap[1]]}"
    return _rp_render(
        rng,
        base,
        target,
        length,
        plan,
        criteria,
        weights,
        names,
        values,
        req,
        scores,
        totals,
        None,
        (None, None, {named: slip_total}, slip_detail),
        (named, bounds, scheme),
    )


def _rp_swap(weights: Sequence[int], pair: tuple[int, int]) -> list[int]:
    out = list(weights)
    out[pair[0]], out[pair[1]] = out[pair[1]], out[pair[0]]
    return out


def _rp_render(
    rng,
    base,
    target,
    length,
    plan,
    criteria,
    weights,
    names,
    values,
    req,
    scores,
    totals,
    truth,
    slip,
    band_info,
) -> F1World:
    long = length == "long"
    contract = pick(rng, _RP_CONTRACTS)
    _, attr_phrase, incl_phrase, excl_phrase, _ = _RP_REQUIREMENTS[req["kind"]]
    requirement = (incl_phrase if req["inclusive"] else excl_phrase).format(
        t=req["threshold"]
    )
    weight_text = join_list([f"{c} {w}%" for c, w in zip(criteria, weights)])
    brief = [
        pick(rng, _RP_BRIEF).format(contract=contract, criteria=join_list(criteria)),
        pick(rng, _RP_WEIGHT_SENTENCE).format(weights=weight_text),
        pick(rng, _RP_METHOD),
        pick(rng, _RP_MANDATORY).format(requirement=requirement),
        pick(rng, _RP_AWARD),
    ]
    line_template = pick(rng, _RP_LINE)
    order = list(names)
    rng.shuffle(order)
    lines = {}
    for name in order:
        parts = [f"{c} {s}" for c, s in zip(criteria, scores[name])]
        lines[name] = line_template.format(
            name=name,
            scores=join_list(parts),
            attr=attr_phrase.format(v=values[name]),
        )
    bullet = pick(rng, ("- ", "\u2022 ", ""))
    sheet = (
        pick(rng, _RP_SHEET_HEAD).format(contract=contract)
        + "\n"
        + "\n".join(bullet + lines[name] for name in order)
    )
    evidence = [" ".join(brief), sheet]
    decisive = [brief[1], brief[3], *lines.values()]
    labels = None
    if band_info is not None:
        named, bounds, scheme = band_info
        labels = pick(rng, _RP_BAND_LABELS)
        template = _RP_BANDS_LOWER if scheme == "lower" else _RP_BANDS_UPPER
        band_text = pick(rng, _RP_BAND_LEAD) + template.format(
            l0=labels[0],
            l1=labels[1],
            l2=labels[2],
            l3=labels[3],
            b1=_rp_hundredths(bounds[0]),
            b2=_rp_hundredths(bounds[1]),
            b3=_rp_hundredths(bounds[2]),
        )
        evidence.append(band_text)
        decisive.append(band_text)
    staff = people(rng, 8)
    others = [s for s in _RP_SUPPLIERS if s not in names]
    rng.shuffle(others)
    distractors = []
    templates = list(_RP_DISTRACT)
    rng.shuffle(templates)
    for template in templates[: (4 if long else rng.randint(1, 2))]:
        distractors.append(
            template.format(
                late=others.pop(),
                name=staff.pop(),
                contract=contract,
                supplier=pick(rng, names),
            )
        )
    if long or rng.random() < 0.15:
        other_contract = pick(rng, [c for c in _RP_CONTRACTS if c != contract])
        other_names = [others.pop() for _ in range(3)]
        other_lines = [
            line_template.format(
                name=n,
                scores=join_list([f"{c} {rng.randint(3, 10)}" for c in criteria]),
                attr=attr_phrase.format(
                    v=max(1, req["threshold"] + rng.randint(-3, 5))
                ),
            )
            for n in other_names
        ]
        distractors.append(
            pick(rng, _RP_SHEET_HEAD).format(contract=other_contract)
            + "\n"
            + "\n".join(bullet + line for line in other_lines)
        )
    rng.shuffle(distractors)

    fields = {"contract": contract}
    if base == "score":
        named, bounds, scheme = band_info
        fields["named"] = named
        question = pick(rng, _RP_QUESTION["score"]).format(**fields)
        refs = [
            label if label.startswith(("Band", "Tier")) else f"the {label} band"
            for label in labels
        ]
        choices = tuple(
            f"{ref[0].upper()}{ref[1:]} ({place})"
            for ref, place in zip(
                refs, ("lowest", "second lowest", "second highest", "highest")
            )
        )
        gold = _rp_band(totals[named], bounds, scheme)
        wrong_answer = plan["wrong_level"]
        conclusion_variant = rng.randrange(len(_RP_CONCLUDE_SCORE))
        rationale_variant = rng.randrange(len(_RP_RATIONALE_SCORE))

        def claim(answer: int, total: int, label: str) -> Claim:
            return Claim(
                answer=answer,
                conclusion=_RP_CONCLUDE_SCORE[conclusion_variant].format(
                    named=named, label=refs[answer]
                ),
                rationale=_RP_RATIONALE_SCORE[rationale_variant].format(
                    named=named, total=_rp_hundredths(total)
                ),
                mechanism=label,
            )

        right = claim(gold, totals[named], "correct")
        wrong = claim(wrong_answer, slip[2][named], plan["mechanism"])
        subject = f"{named}'s score band"
    else:
        winner, runner = truth
        slip_winner, slip_runner, slip_totals, _ = slip
        named = None
        if base == "choice":
            choices = tuple(order)
            gold = order.index(winner)
            wrong_answer = order.index(slip_winner)
            question = pick(rng, _RP_QUESTION["choice"]).format(**fields)
            conclusion_variant = rng.randrange(len(_RP_CONCLUDE_CHOICE))
        else:
            choices = ()
            gold = target
            wrong_answer = 1 - target
            named = winner if target == 1 else slip_winner
            fields["named"] = named
            question = pick(rng, _RP_QUESTION["noul"]).format(**fields)
            conclusion_variant = rng.randrange(len(_RP_CONCLUDE_NOUL))
        rationale_variant = rng.randrange(len(_RP_RATIONALE))

        def claim(
            answer: int, top: str, second: str, table: dict[str, int], label: str
        ) -> Claim:
            if base == "choice":
                conclusion = _RP_CONCLUDE_CHOICE[conclusion_variant].format(
                    winner=top, contract=contract
                )
            else:
                conclusion = _RP_CONCLUDE_NOUL[conclusion_variant][answer].format(
                    named=named
                )
            rationale = _RP_RATIONALE[rationale_variant].format(
                winner=top,
                runner=second,
                top=_rp_hundredths(table[top]),
                next=_rp_hundredths(table[second]),
            )
            return Claim(
                answer=answer,
                conclusion=conclusion,
                rationale=rationale,
                mechanism=label,
            )

        right = claim(gold, winner, runner, totals, "correct")
        wrong = claim(
            wrong_answer, slip_winner, slip_runner, slip_totals, plan["mechanism"]
        )
        subject = f"the {contract} evaluation"
    text = "\n\n".join(evidence + distractors)
    recheck = _rp_recheck(text, contract, base, choices, named)
    if _rp_recheck("\n\n".join(evidence), contract, base, choices, named) != recheck:
        raise GenerationError("rubric_pick: a distractor changed the answer")
    need = (
        max(0, LONG_POOL_CHARS - sum(len(block) + 2 for block in distractors))
        if long
        else 0
    )
    filler = (
        _filler(
            rng, _RP_FILLER, _filler_slots(rng, staff, {"contract": contract}), need
        )
        if need
        else ()
    )
    facts = {
        "contract": contract,
        "criteria": criteria,
        "weights": weights,
        "requirement": {k: v for k, v in req.items() if k != "passes"},
        "proposals": {
            n: {"scores": scores[n], "value": values[n], "total": totals[n]}
            for n in names
        },
        "wrong": {"mechanism": plan["mechanism"], "detail": slip[3]},
        "plan": {k: v for k, v in plan.items() if k != "mechanism"},
        "named": named,
        "names": sorted(staff) if filler else [],
    }
    if truth is not None:
        facts["winner"], facts["slip_winner"] = truth[0], slip[0]
    return F1World(
        kind="rubric_pick",
        base=base,
        subject=subject,
        question=question,
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=filler,
        right=right,
        wrong=wrong,
        decisive=tuple(decisive),
        facts=facts,
        variant=f"l{_RP_LINE.index(line_template)}-k{conclusion_variant}-q{rationale_variant}",
        roles=(
            "procurement lead",
            "panel member",
            "evaluation chair",
            "contracts manager",
            "category manager",
            "commercial analyst",
        ),
    )


_RP_RX_WEIGHTS = [_rx(t, weights=r"[A-Z].*\d+%") for t in _RP_WEIGHT_SENTENCE]
_RP_RX_HEAD = [_rx(t, contract=r"[a-z][a-z \-]*") for t in _RP_SHEET_HEAD]
_RP_RX_REQ = []
for _kind, (_higher, _attr, _incl, _excl, _) in _RP_REQUIREMENTS.items():
    for _inclusive, _phrase in ((True, _incl), (False, _excl)):
        for _template in _RP_MANDATORY:
            _RP_RX_REQ.append(
                (
                    _kind,
                    _higher,
                    _inclusive,
                    _rx(
                        _template.replace(
                            "{requirement}", _phrase.replace("{t}", "{threshold}")
                        ),
                        threshold=r"\d+",
                    ),
                )
            )
_RP_RX_LINE = [
    (
        kind,
        [
            _rx(
                t.replace("{attr}", phrase.replace("{v}", "{value}")),
                name=r"[A-Z][A-Za-z ]*?",
                scores=r"[A-Z][A-Za-z ]* \d+(?:(?:, and |, | and )[A-Z][A-Za-z ]* \d+)*",
                value=r"\d+",
            )
            for t in _RP_LINE
        ],
    )
    for kind, (_, phrase, _, _, _) in _RP_REQUIREMENTS.items()
]
_RP_RX_BANDS = [
    (
        scheme,
        _rx(
            lead + template,
            l0=r"[A-Z][\w ]*",
            l1=r"[A-Z][\w ]*",
            l2=r"[A-Z][\w ]*",
            l3=r"[A-Z][\w ]*",
            b1=r"\d+\.\d{2}",
            b2=r"\d+\.\d{2}",
            b3=r"\d+\.\d{2}",
        ),
    )
    for lead in _RP_BAND_LEAD
    for scheme, template in (("lower", _RP_BANDS_LOWER), ("upper", _RP_BANDS_UPPER))
]


def _rp_recheck(
    text: str, contract: str, base: str, options: Sequence[str], named: str | None
) -> int:
    weights: dict[str, int] = {}
    rule = None
    rows: dict[str, tuple[dict[str, int], int]] = {}
    bands = None
    for units in _units(text):
        head = next(
            (p.fullmatch(units[0]) for p in _RP_RX_HEAD if p.fullmatch(units[0])), None
        )
        ours = head is not None and head["contract"] == contract
        for position, unit in enumerate(units):
            for pattern in _RP_RX_WEIGHTS:
                match = pattern.fullmatch(unit)
                if match:
                    for label, share in re.findall(
                        r"([A-Z][A-Za-z ]*?) (\d+)%", match["weights"]
                    ):
                        weights[label] = int(share)
            for kind, higher, inclusive, pattern in _RP_RX_REQ:
                match = pattern.fullmatch(unit)
                if match:
                    rule = (kind, higher, inclusive, int(match["threshold"]))
            for scheme, pattern in _RP_RX_BANDS:
                match = pattern.fullmatch(unit)
                if match:
                    cut = [round(float(match[key]) * 100) for key in ("b1", "b2", "b3")]
                    bands = (scheme, cut)
            if ours and position > 0:
                parsed = None
                for kind, patterns in _RP_RX_LINE:
                    for pattern in patterns:
                        match = pattern.fullmatch(unit)
                        if match:
                            marks = {
                                label: int(v)
                                for label, v in re.findall(
                                    r"([A-Z][A-Za-z ]*?) (\d+)", match["scores"]
                                )
                            }
                            parsed = (match["name"], marks, int(match["value"]), kind)
                            break
                    if parsed:
                        break
                if parsed is None:
                    raise GenerationError(
                        f"rubric_pick re-check: unparsed score line {unit!r}"
                    )
                rows[parsed[0]] = (parsed[1], parsed[2])
    if not weights or rule is None or not rows or sum(weights.values()) != 100:
        raise GenerationError("rubric_pick re-check: brief or sheet not found")
    kind, higher, inclusive, threshold = rule
    score = {
        name: sum(weights[label] * mark for label, mark in marks.items())
        for name, (marks, _) in rows.items()
    }
    if base == "score":
        if bands is None:
            raise GenerationError("rubric_pick re-check: bands not found")
        scheme, cut = bands
        value = score[named]
        return len([c for c in cut if (value >= c if scheme == "lower" else value > c)])

    def ok(amount: int) -> bool:
        if higher:
            return amount > threshold or (inclusive and amount == threshold)
        return amount < threshold or (inclusive and amount == threshold)

    allowed = sorted(
        (name for name, (_, amount) in rows.items() if ok(amount)),
        key=lambda n: score[n],
        reverse=True,
    )
    if len(allowed) < 2 or score[allowed[0]] == score[allowed[1]]:
        raise GenerationError("rubric_pick re-check: no clear eligible leader")
    best = allowed[0]
    if base == "choice":
        return list(options).index(best)
    return int(best == named)


def build_rubric_pick(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    _check_args("rubric_pick", base, target, length)
    plan = _rp_plan(rng, base, target)
    draw = _rp_draw_score if base == "score" else _rp_draw_choice
    return _attempt_loop("rubric_pick", lambda: draw(rng, base, target, length, plan))


# ---------------------------------------------------------------- subscription_bill

_SB_SERVICES = (
    ("Lumenbox", "cloud storage", ("Basic", "Plus", "Pro", "Business")),
    ("Harbourline", "video meeting", ("Starter", "Standard", "Premium", "Enterprise")),
    ("Tallyworks", "bookkeeping", ("Solo", "Studio", "Growth", "Scale")),
    ("Beaconly", "web hosting", ("Lite", "Core", "Plus", "Max")),
    ("Fernway", "music streaming", ("Single", "Duo", "Family", "Studio")),
    ("Kestrel Mail", "newsletter", ("Starter", "Growth", "Pro", "Premier")),
    ("Orchardly", "project planning", ("Essentials", "Team", "Business", "Corporate")),
    ("Quillstack", "note-taking", ("Personal", "Plus", "Pro", "Teams")),
)
# plan prices always carry cents, so an unprorated month never looks like a round number
_SB_CENTS = (25, 40, 49, 50, 75, 90, 95, 99)
_SB_PROMO_AMOUNTS = (500, 600, 800, 1000, 1200, 1500, 2000)
_SB_WIDTHS = (200, 300, 500, 800, 1000, 1500, 2000)
_SB_WIDE = 1000
_SB_SEASONS = (
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
    "New Year",
    "Anniversary",
    "Midyear",
)
_SB_CREDIT_LABELS = (
    "service credit",
    "goodwill credit",
    "referral credit",
    "loyalty credit",
    "billing-error credit",
    "courtesy credit",
)
# primary notice: in force on the plan before (A) or after (B) the switch, or not decisive
_SB_NOTICE_ROLES = (("A", 0.425), ("B", 0.425), ("other", 0.075), ("future", 0.075))
# promotion eligibility of (plan before, plan after) the switch
_SB_PROMO_PATTERNS = (
    ((1, 0), 0.425),
    ((0, 1), 0.425),
    ((1, 1), 0.075),
    ((0, 0), 0.075),
)
_SB_POLARITIES = ("above", "at_most", "refund", "pay")
_SB_ROLES = (
    "billing specialist",
    "account manager",
    "support agent",
    "finance assistant",
    "subscriptions analyst",
    "customer success manager",
)
_SB_HEADER = (
    "Account {acct}, held by {name}, is on a monthly {company} {service} subscription billed in arrears; the "
    "current billing period runs from {start} to {end}, and its invoice is issued on {issue}.",
    "{company} {service} subscription for {name} (account {acct}): the current monthly billing period runs from "
    "{start} to {end} and is invoiced in arrears on {issue}.",
    "Billing period for account {acct} ({name}, {company} {service}): {start} to {end}, invoiced in arrears on "
    "{issue}.",
)
_SB_RULES = (
    "Billing rules. Each monthly billing period is invoiced in arrears. When the plan changes part-way through a "
    "period, each plan is charged for the days it was active: its monthly price is divided by the number of days "
    "in the period and multiplied by those days, and each plan's charge is rounded to the nearest cent. The day "
    "of the switch is billed on the new plan. A price change applies from the first billing period that starts "
    "on or after its effective date, so a period that is already running keeps the old price. Credits on the "
    "account are deducted from the next invoice; if they come to more than the charges, the difference is "
    "refunded to the card on file.",
    "How proration works: plans are charged by the day. For each plan used during a period, the charge is the "
    "monthly price multiplied by the number of days on that plan and divided by the number of days in the "
    "period, rounded to the cent. The switch date counts as a day on the new plan, not the old one. Price changes "
    "take effect from the first period that begins on or after the effective date and never change a period "
    "that has already started. Credits come off the next invoice, and credit larger than the charges is paid "
    "back to the card as a refund.",
    "Proration policy: a plan change during a billing period splits the period by day. The old plan is charged "
    "for the days before the switch date and the new plan for the days from the switch date to the end of the "
    "period, each at its monthly price times its days divided by the days in the period, rounded to the nearest "
    "cent per plan. New prices apply to billing periods that begin on or after the effective date of the change. "
    "Credits reduce the next invoice, and whatever credit is left over after the charges is refunded.",
)
_SB_RULES_SHORT = (
    "Billing rules: periods are invoiced in arrears. After a mid-period plan change, each plan is charged its "
    "monthly price times its days divided by the days in the period, rounded to the cent per plan; the switch "
    "day counts on the new plan. New prices apply from the first period starting on or after their effective "
    "date. Credits come off the next invoice, and any excess is refunded.",
    "Proration is by the day: each plan costs its monthly price times its days on the plan divided by the days "
    "in the period, rounded to the cent per plan, and the switch date is billed on the new plan. A price change "
    "applies to periods that start on or after its effective date. Credits reduce the next invoice; any excess "
    "is refunded to the card.",
    "Plans are billed in arrears and prorated daily: the monthly price times the days on the plan, divided by "
    "the days in the period, rounded to the cent for each plan, with the switch day on the new plan. Price "
    "changes start with the first period beginning on or after the effective date. Credits come off the next "
    "invoice; excess credit is refunded.",
)
_SB_PRICE_HEAD = (
    "{company} monthly plan prices (price list dated {listed}):",
    "Plan prices per month, from the {company} price list of {listed}:",
)
_SB_PRICE_LINE = ("{plan}: {price}", "{plan} plan \u2014 {price} a month")
_SB_PRICE_PROSE = (
    "The {company} price list dated {listed} gives these monthly prices: {pairs}.",
    "Monthly prices on the {company} price list of {listed}: {pairs}.",
)
_SB_NOTICE = (
    "Price change notice, sent {sent}: from {eff}, the {plan} plan costs {new} a month instead of {old}.",
    "On {sent}, {company} announced that the monthly price of the {plan} plan changes from {old} to {new} with "
    "effect from {eff}.",
    "Notice of new pricing ({sent}): the {plan} plan moves from {old} to {new} per month, effective {eff}.",
)
_SB_CHANGE = (
    "On {when}, account {acct} switched from the {old} plan to the {new} plan.",
    "Plan change on account {acct}: {old} to {new}, effective {when}.",
    "{holder} moved account {acct} from the {old} plan to the {new} plan on {when}.",
)
_SB_BOOKED = (
    "Account {acct} is booked to move from the {old} plan to the {new} plan on {when}.",
    "A further change is scheduled for account {acct}: {old} to {new} on {when}.",
)
_SB_CREDIT = (
    "Credit on account {acct}: {amount} ({label}), to be deducted from the next invoice.",
    "Account {acct} holds a {label} of {amount}, which comes off the next invoice.",
    "A {label} of {amount} was added to account {acct} and will be taken off the next invoice.",
)
_SB_CREDIT_USED = (
    "The {label} of {amount} on account {acct} was already deducted from the invoice of {when}.",
    "An earlier {label} of {amount} on account {acct} was used up on the invoice issued {when}.",
)
_SB_PROMO = (
    "{season} offer, for invoices issued from {first} to {last}: accounts on the {plans} plan on the invoice date "
    "get a one-off credit of {amount} on that invoice; other plans do not qualify.",
    "Promotion ({season} offer): a one-off {amount} credit on invoices issued from {first} to {last}, for "
    "accounts that are on the {plans} plan when the invoice is issued.",
    "{season} offer from {company}: every account on the {plans} plan on its invoice date receives {amount} off "
    "that invoice, for invoices dated {first} to {last}; other plans are not eligible.",
)
_SB_LAST = (
    "The previous invoice for account {acct}, issued on {when}, came to {amount}.",
    "Account {acct} was billed {amount} on its last invoice, dated {when}.",
)
_SB_ACTIVITY_HEAD = (
    "Recent activity on account {acct}:",
    "Account {acct} \u2014 changes and credits:",
)
_SB_NOTES = (
    "Support ticket {ticket}: {person} asked how the plan switch would appear on the invoice, and the agent sent "
    "a link to the billing help page.",
    "{person} replaced the payment card on account {acct} after the old card expired; the card on file does not "
    "change any charge.",
    "The usage report for account {acct} shows normal activity this period; usage does not affect the plan "
    "charges.",
)
_SB_QUESTION = {
    "choice": (
        "What amount will {name}'s next invoice show?",
        "How much will the next invoice for account {acct} come to?",
        "What will {name} be billed on the invoice of {issue}?",
    ),
    ("noul", "above"): (
        "Will {name}'s next invoice come to more than {threshold}?",
        "Is the next invoice for account {acct} going to be above {threshold}?",
    ),
    ("noul", "at_most"): (
        "Will {name}'s next invoice come to {threshold} or less?",
        "Will the next invoice for account {acct} be no more than {threshold}?",
    ),
    ("noul", "refund"): (
        "Will {name} be owed a refund when the next invoice is issued?",
        "Does the next invoice for account {acct} leave a refund due to {name}?",
    ),
    ("noul", "pay"): (
        "Will {name} have anything to pay on the next invoice?",
        "Does the next invoice for account {acct} leave an amount for {name} to pay?",
    ),
    "score": (
        "Compared with the previous invoice, how will {name}'s next invoice change?",
        "How does the next invoice for account {acct} compare with the previous one?",
    ),
}
_SB_CONCLUDE_CHOICE = (
    "{name}'s next invoice will come to {total}",
    "the next invoice for account {acct} is {total}",
    "{given} will be billed {total} on the next invoice",
)
_SB_CONCLUDE_NOUL = {
    ("above", 1): (
        "{name}'s next invoice will come to more than {threshold}",
        "the next invoice for account {acct} will be above {threshold}",
    ),
    ("above", 0): (
        "{name}'s next invoice will not come to more than {threshold}",
        "the next invoice for account {acct} will not be above {threshold}",
    ),
    ("at_most", 1): (
        "{name}'s next invoice will come to {threshold} or less",
        "the next invoice for account {acct} will be no more than {threshold}",
    ),
    ("at_most", 0): (
        "{name}'s next invoice will come to more than {threshold}",
        "the next invoice for account {acct} will be more than {threshold}",
    ),
    ("refund", 1): (
        "{name} will be owed a refund on the next invoice",
        "the next invoice for account {acct} leaves a refund due to {given}",
    ),
    ("refund", 0): (
        "{name} will not be owed a refund on the next invoice",
        "the next invoice for account {acct} leaves no refund due to {given}",
    ),
    ("pay", 1): (
        "{name} will have something to pay on the next invoice",
        "the next invoice for account {acct} leaves an amount for {given} to pay",
    ),
    ("pay", 0): (
        "{name} will have nothing to pay on the next invoice",
        "the next invoice for account {acct} leaves nothing for {given} to pay",
    ),
}
_SB_CONCLUDE_SCORE = (
    (
        "{name}'s next invoice will be more than {width} lower than the previous one",
        "{name}'s next invoice will be at most {width} lower than the previous one, or the same",
        "{name}'s next invoice will be higher than the previous one by {width} or less",
        "{name}'s next invoice will be more than {width} higher than the previous one",
    ),
    (
        "the next invoice for account {acct} will drop by more than {width}",
        "the next invoice for account {acct} will drop by {width} or less, if at all",
        "the next invoice for account {acct} will rise, by {width} or less",
        "the next invoice for account {acct} will rise by more than {width}",
    ),
)
_SB_LEVELS = (
    "More than {width} lower than the previous invoice",
    "Lower by {width} or less, or unchanged",
    "Higher by {width} or less",
    "More than {width} higher than the previous invoice",
)
_SB_RATIONALE = (
    "the plan charges for the period come to {charges} and the credits to {credits}",
    "the period's plan charges total {charges}, against {credits} of credits",
    "{charges} is due in plan charges for the period, less {credits} of credits",
)
_SB_FILLER = (
    "Invoices from {company} are emailed to the billing contact on file. {name} asked for that contact to be "
    "changed to the shared finance mailbox so that invoices are not missed during holidays.",
    "The payment card on an account can be changed from the billing page at any time. {name2} asked whether "
    "several cards can be kept on file; only one card is kept, and it is used for every invoice.",
    "{name} called the {company} help desk to ask where past invoices are kept. They are in the documents tab of "
    "the account page as PDF files, and each one can be downloaded as often as needed.",
    "Company details on invoices: {name2} asked for the registered address to be added to future invoices. The "
    "change was made on the account page and needs no action from the billing team.",
    "The {company} status page lists planned maintenance. Work is announced a week ahead and usually happens "
    "overnight, so most customers never notice it.",
    "Data export: the account owner can export everything from the settings page. {name} tested an export "
    "recently, and it finished within a few minutes.",
    "Billing questions go through the {company} help centre, where replies usually arrive within a working day. "
    "{name2} keeps a list of the ticket numbers in the shared folder.",
    "Security reminder: {company} never asks for card details by email. Suspicious messages should be forwarded "
    "to the security team and then deleted, and {name} can help if anyone is unsure.",
    "The admin panel shows who can see billing information. {name} is the account owner, and {name2} was given "
    "read-only access so that the finance team can check invoices without changing anything.",
    "Renewal reminders are sent by email a few days before each billing period ends. They name the plan on the "
    "account and link to the billing page; they are reminders only and do not change any charge.",
)


def _sb_share(price: int, days: int, period: int) -> int:
    """Oracle proration in cents: price x days / period, halves rounded up."""
    return (2 * price * days + period) // (2 * period)


def _sb_half(price: int, days: int, period: int) -> bool:
    return (2 * price * days) % (2 * period) == period


def _sb_band(change: int, width: int) -> int:
    if change < -width:
        return 0
    if change <= 0:
        return 1
    return 2 if change <= width else 3


def _sb_answer(polarity: str, net: int, threshold: int | None) -> int:
    if polarity == "above":
        return int(net > threshold)
    if polarity == "at_most":
        return int(net <= threshold)
    return int(net < 0) if polarity == "refund" else int(net > 0)


def _sb_month_after(value: date) -> date:
    """The first day of the month after ``value``'s month."""
    return date(value.year + value.month // 12, value.month % 12 + 1, 1)


def _sb_month_before(value: date) -> date:
    """The first day of the month before ``value``'s month."""
    return (value.replace(day=1) - timedelta(days=1)).replace(day=1)


def _sb_weighted(rng: random.Random, pairs: Sequence[tuple[Any, float]]) -> Any:
    return rng.choices([value for value, _ in pairs], [weight for _, weight in pairs])[
        0
    ]


def _sb_prices(rng: random.Random, tiers: Sequence[str]) -> dict[str, int]:
    dollars = rng.randint(6, 16)
    prices = {}
    for index, tier in enumerate(tiers):
        if index:
            dollars += rng.randint(*((7, 14), (9, 20), (14, 30))[index - 1])
        prices[tier] = dollars * 100 + pick(rng, _SB_CENTS)
    return prices


def _sb_notice(
    rng: random.Random,
    tiers: Sequence[str],
    listed: dict[str, int],
    plan_name: str,
    up: bool,
    eff: date,
) -> dict[str, Any]:
    old = listed[plan_name]
    step = rng.randint(max(1, old // 1250), max(2, old // 400)) * 100 + pick(
        rng, (0, 0, 0, 50)
    )
    new = old + step if up else old - step
    index = tiers.index(plan_name)
    if (
        new < 300
        or (index and new <= listed[tiers[index - 1]] + 200)
        or (index < len(tiers) - 1 and new >= listed[tiers[index + 1]] - 200)
    ):
        raise _Redraw("price change crosses a neighbouring plan")
    return {
        "plan": plan_name,
        "old": old,
        "new": new,
        "eff": eff,
        "sent": eff - timedelta(days=rng.randint(21, 50)),
    }


def _sb_split(rng: random.Random, total: int) -> list[int]:
    """One or two credit lines adding up to ``total`` cents (the first a whole-dollar amount)."""
    if total < 400 or rng.random() < 0.4:
        return [total]
    first = rng.randint(1, total // 100 - 1) * 100
    return [first, total - first]


def _sb_between(rng: random.Random, low: int, high: int) -> int | None:
    """A round amount strictly between two totals (cents), preferring coarse steps."""
    for step in (500, 100, 50, 10):
        values = [
            value
            for value in range((low // step + 1) * step, high, step)
            if low < value < high
        ]
        if values:
            return pick(rng, values)
    return None


def _sb_bands(
    rng: random.Random, net: int, wrong_net: int, gold: int, wrong: int, width: int
) -> tuple[int, int] | None:
    """The planned band width and a previous-invoice amount that put ``net`` in ``gold`` and
    ``wrong_net`` in ``wrong``."""
    spans = (
        (-(10**7), -width - 5),
        (-width + 5, -5),
        (5, width - 5),
        (width + 5, 10**7),
    )
    low, high = max(500, net - 6000), net + 6000
    for total, level in ((net, gold), (wrong_net, wrong)):
        least, most = spans[level]
        low, high = max(low, total - most), min(high, total - least)
    return (width, rng.randint(low, high)) if low <= high else None


def _sb_plan(rng: random.Random, base: str, target: int | None) -> dict[str, Any]:
    """Draw the visible features first (independent of the target), then the mechanism they allow."""
    plan: dict[str, Any] = {
        "upgrade": rng.random() < 0.5,
        "notice": _sb_weighted(rng, _SB_NOTICE_ROLES),
        "price_up": rng.random() < 0.5,
        "promo": _sb_weighted(rng, _SB_PROMO_PATTERNS),
        "polarity": None,
        "wrong_level": None,
    }
    if base == "choice":
        over = rng.random() < 0.5
        plan["layout"] = _option_layout(rng, not over)
    elif base == "noul":
        plan["polarity"] = pick(rng, _SB_POLARITIES)
        # a higher wrong total answers yes to "above" and "pay", and no to "at_most" and "refund"
        over = (target == 0) == (plan["polarity"] in ("above", "pay"))
    else:
        # the band width is on every option, so it is drawn before (and apart from) the target;
        # no slip moves an invoice across a whole wide band, so wide bands take the partner level
        # next to the gold (each gold has one such partner, so wrong levels stay uniform)
        plan["width"] = pick(rng, _SB_WIDTHS)
        if plan["width"] >= _SB_WIDE:
            plan["wrong_level"] = target + (1 if target in (0, 2) else -1)
        else:
            plan["wrong_level"] = _pt_wrong_level(rng, target)
        over = plan["wrong_level"] > target
    plan["direction"] = "over" if over else "under"
    available = []
    if plan["notice"] in ("A", "B") and plan["price_up"] != over:
        available.append("stale_value")
    if plan["promo"] == ((0, 1) if over else (1, 0)):
        available.append("scope_misapplied")
    plan["mechanism"] = pick(rng, available) if available else "arithmetic_slip"
    return plan


def _sb_draw(
    rng: random.Random, base: str, target: int | None, length: str, plan: dict[str, Any]
) -> F1World:
    long = length == "long"
    company, service, tiers = pick(rng, _SB_SERVICES)
    listed = _sb_prices(rng, tiers)
    low, high = sorted(rng.sample(range(len(tiers)), 2))
    plan_a, plan_b = (
        (tiers[low], tiers[high]) if plan["upgrade"] else (tiers[high], tiers[low])
    )
    unused = [tier for tier in tiers if tier not in (plan_a, plan_b)]
    start = _draw_day(rng)
    if start.day > 28:
        start = start.replace(day=rng.randint(1, 28))
    issue = date(start.year + start.month // 12, start.month % 12 + 1, start.day)
    end = issue - timedelta(days=1)
    period = (issue - start).days
    days_a = rng.randint(3, period - 3)
    days_b = period - days_a
    switch = start + timedelta(days=days_a)

    # the plan draws the primary notice; the second one is never decisive
    in_force = sorted({start, start.replace(day=1), _sb_month_before(start)})
    later = sorted({_sb_month_after(start), issue, _sb_month_after(issue)})
    role = plan["notice"]
    if role in ("A", "B"):
        covered, eff = (plan_a if role == "A" else plan_b), pick(rng, in_force)
    elif role == "future":
        covered, eff = pick(rng, (plan_a, plan_b)), pick(rng, later)
    else:
        covered, eff = pick(rng, unused), pick(rng, in_force + later)
    notices = [_sb_notice(rng, tiers, listed, covered, plan["price_up"], eff)]
    second = pick(rng, [tier for tier in tiers if tier != covered])
    second_eff = pick(rng, later if second in (plan_a, plan_b) else in_force + later)
    notices.append(
        _sb_notice(rng, tiers, listed, second, rng.random() < 0.5, second_eff)
    )
    price = dict(listed)
    for notice in notices:
        if notice["eff"] <= start:
            price[notice["plan"]] = notice["new"]
    if any(price[x] >= price[y] for x, y in zip(tiers, tiers[1:])):
        raise _Redraw("price changes reorder the plans")
    pa, pb = price[plan_a], price[plan_b]
    if _sb_half(pa, days_a, period) or _sb_half(pb, days_b, period):
        raise _Redraw("a prorated charge ends in half a cent")

    # the promotional credit follows the plan held on the invoice date
    flags = {plan_a: plan["promo"][0], plan_b: plan["promo"][1]}
    eligible = {tier for tier, flag in flags.items() if flag}
    if not eligible or (len(eligible) == 1 and rng.random() < 0.5):
        eligible.add(pick(rng, unused))
    promo_plans = [tier for tier in tiers if tier in eligible]
    promo_on, old_on = plan_b in eligible, plan_a in eligible
    ceiling = max(_SB_PROMO_AMOUNTS[0], min(pa, pb) * 45 // 100)
    promo = pick(rng, [value for value in _SB_PROMO_AMOUNTS if value <= ceiling])
    promo_first = pick(rng, (issue.replace(day=1), _sb_month_before(issue)))
    promo_last = pick(
        rng, (_sb_month_after(issue), _sb_month_after(_sb_month_after(issue)))
    ) - timedelta(days=1)

    def charge(
        price_a: int, price_b: int, first: int = days_a, last: int = days_b
    ) -> int:
        return _sb_share(price_a, first, period) + _sb_share(price_b, last, period)

    charges = charge(pa, pb)
    # mechanism outputs: (mechanism, plan charges, promotion counted)
    outputs: dict[str, tuple[str, int, bool]] = {
        "no_proration_new": ("arithmetic_slip", pb, promo_on),
        "no_proration_old": ("arithmetic_slip", pa, promo_on),
        "switch_day_twice": ("arithmetic_slip", charge(pa, pb, days_a + 1), promo_on),
        "switch_day_dropped": (
            "arithmetic_slip",
            charge(pa, pb, days_a, days_b - 1),
            promo_on,
        ),
    }
    if role in ("A", "B"):
        stale = {plan_a: pa, plan_b: pb, covered: notices[0]["old"]}
        outputs["old_price"] = (
            "stale_value",
            charge(stale[plan_a], stale[plan_b]),
            promo_on,
        )
    if old_on != promo_on:
        outputs["promo_by_old_plan"] = ("scope_misapplied", charges, old_on)
    early = []
    for notice in notices:
        if notice["plan"] in (plan_a, plan_b) and notice["eff"] > start:
            ahead = {plan_a: pa, plan_b: pb, notice["plan"]: notice["new"]}
            early.append(charge(ahead[plan_a], ahead[plan_b]))

    over = plan["direction"] == "over"

    def shift(name: str) -> int:
        _, value, flag = outputs[name]
        return value - charges - promo * (int(flag) - int(promo_on))

    fits = [
        name
        for name, output in outputs.items()
        if output[0] == plan["mechanism"]
        and abs(shift(name)) >= 40
        and (shift(name) > 0) == over
    ]
    if not fits:
        raise _Redraw("the planned mechanism has no output in the planned direction")
    rng.shuffle(fits)
    lead = "no_proration" if rng.random() < 0.55 else "switch_day"
    fits.sort(key=lambda name: not name.startswith(lead))

    polarity = plan["polarity"]
    straddle = base == "noul" and polarity in ("refund", "pay")
    moderate = rng.randint(150, max(200, min(2500, charges * 35 // 100)))
    moderate_parts = _sb_split(rng, moderate)
    used_credit = rng.randint(3, 15) * 100 + pick(rng, (0, 0, 50))
    old_promo = pick(rng, [value for value in _SB_PROMO_AMOUNTS if value != promo])
    found = None
    for name in fits:
        _, wrong_charges, wrong_flag = outputs[name]
        if straddle:
            # the other credits put zero strictly between the true and the slipped balance
            x_true, x_wrong = (
                charges - promo * promo_on,
                wrong_charges - promo * wrong_flag,
            )
            lo, hi = max(150, min(x_true, x_wrong) + 5), max(x_true, x_wrong) - 5
            if lo > hi:
                continue
            other = rng.randint(lo, hi)
            parts = _sb_split(rng, other)
        else:
            other, parts = moderate, moderate_parts
        net = charges - other - promo * promo_on
        wrong_net = wrong_charges - other - promo * wrong_flag
        if not straddle and min(net, wrong_net) < 100:
            continue
        if base == "choice":
            # other slips on the same records, on both sides of the true amount
            credit_true = other + promo * promo_on
            pool = [value - other - promo * flag for _, value, flag in outputs.values()]
            pool += [value - credit_true for value in early]
            pool += [net + part for part in parts] + [charges, net - old_promo]
            pool += [
                net - _sb_share(pa, days_a, period),
                net - _sb_share(pb, days_b, period),
            ]
            if used_credit not in parts:
                pool.append(net - used_credit)
            pool.append(net + promo if promo_on else net - promo)
            if period != 30:
                pool.append(
                    _sb_share(pa, days_a, 30) + _sb_share(pb, days_b, 30) - credit_true
                )
            spread: list[int] = []
            for value in pool:
                if value >= 100 and all(
                    abs(value - kept) > 5 for kept in spread + [net, wrong_net]
                ):
                    spread.append(value)
            try:
                detail: Any = _ranked_options(
                    rng, net, wrong_net, spread, plan["layout"]
                )
            except _Redraw:
                continue
        elif base == "noul":
            detail = (
                None
                if straddle
                else _sb_between(rng, min(net, wrong_net), max(net, wrong_net))
            )
            if not straddle and detail is None:
                continue
            if (
                _sb_answer(polarity, net, detail) != target
                or _sb_answer(polarity, wrong_net, detail) != 1 - target
            ):
                continue
        else:
            detail = _sb_bands(
                rng, net, wrong_net, target, plan["wrong_level"], plan["width"]
            )
            if detail is None:
                continue
        found = (name, wrong_charges, wrong_flag, other, parts, net, wrong_net, detail)
        break
    if found is None:
        raise _Redraw("no mechanism output fits the question")
    detail_name, wrong_charges, wrong_flag, other, parts, net, wrong_net, detail = found

    # rendering
    style = pick(rng, DATE_STYLES)
    fmt = lambda value: fmt_date(value, style)  # noqa: E731
    names = people(rng, 8)
    holder, *crowd = names
    acct, *other_accounts = [
        f"AC-{number}" for number in rng.sample(range(10000, 100000), 3)
    ]
    header_index = rng.randrange(len(_SB_HEADER))
    header = _SB_HEADER[header_index].format(
        acct=acct,
        name=holder,
        company=company,
        service=service,
        start=fmt(start),
        end=fmt(end),
        issue=fmt(issue),
    )
    rule_bank = _SB_RULES if long else _SB_RULES_SHORT
    rule_index = rng.randrange(len(rule_bank))
    listed_on = min(notice["sent"] for notice in notices) - timedelta(
        days=rng.randint(15, 80)
    )
    price_style = "list" if rng.random() < 0.6 else "prose"
    if price_style == "list":
        line = pick(rng, _SB_PRICE_LINE)
        lines = {
            tier: line.format(plan=tier, price=_cents(listed[tier])) for tier in tiers
        }
        bullet = pick(rng, ("- ", "\u2022 ", ""))
        price_block = (
            pick(rng, _SB_PRICE_HEAD).format(company=company, listed=fmt(listed_on))
            + "\n"
            + "\n".join(bullet + lines[tier] for tier in tiers)
        )
        price_facts = [lines[plan_a], lines[plan_b]]
    else:
        pairs = join_list([f"{tier} {_cents(listed[tier])}" for tier in tiers])
        price_block = pick(rng, _SB_PRICE_PROSE).format(
            company=company, listed=fmt(listed_on), pairs=pairs
        )
        price_facts = [price_block]

    def notice_text(notice: dict[str, Any]) -> str:
        return pick(rng, _SB_NOTICE).format(
            sent=fmt(notice["sent"]),
            eff=fmt(notice["eff"]),
            plan=notice["plan"],
            old=_cents(notice["old"]),
            new=_cents(notice["new"]),
            company=company,
        )

    primary_text, second_text = notice_text(notices[0]), notice_text(notices[1])
    season, old_season = rng.sample(_SB_SEASONS, 2)
    promo_text = pick(rng, _SB_PROMO).format(
        season=season,
        first=fmt(promo_first),
        last=fmt(promo_last),
        plans=join_list(promo_plans, "or"),
        amount=_price(promo),
        company=company,
    )
    labels = rng.sample(_SB_CREDIT_LABELS, len(parts) + 1)
    change_text = pick(rng, _SB_CHANGE).format(
        when=fmt(switch), acct=acct, old=plan_a, new=plan_b, holder=holder
    )
    credit_texts = [
        pick(rng, _SB_CREDIT).format(acct=acct, amount=_cents(amount), label=label)
        for amount, label in zip(parts, labels)
    ]
    previous = (
        detail[1]
        if base == "score"
        else max(500, net + pick(rng, (-1, 1)) * rng.randint(100, 2000))
    )
    last_text = pick(rng, _SB_LAST).format(
        acct=acct, when=fmt(start), amount=_cents(previous)
    )
    records = [change_text, *credit_texts] + ([last_text] if base == "score" else [])
    if rng.random() < 0.6:
        bullet = pick(rng, ("- ", "\u2022 "))
        activity = (
            pick(rng, _SB_ACTIVITY_HEAD).format(acct=acct)
            + "\n"
            + "\n".join(bullet + record for record in records)
        )
    else:
        activity = " ".join(records)
    top = [rule_bank[rule_index], price_block]
    rng.shuffle(top)
    rest = [primary_text, activity, promo_text]
    rng.shuffle(rest)
    evidence = [header, *top, *rest]
    decisive = [
        header,
        rule_bank[rule_index],
        *price_facts,
        change_text,
        *credit_texts,
        promo_text,
    ]
    if role in ("A", "B"):
        decisive.append(primary_text)
    if base == "score":
        decisive.append(last_text)

    # distractors: parsed by the re-check and rejected there (other plan, date, account or status)
    distractors = [second_text]
    for account, person in zip(other_accounts[: 2 if long else 1], crowd):
        x, y = rng.sample(tiers, 2)
        items = [
            pick(rng, _SB_CHANGE).format(
                when=fmt(start + timedelta(days=rng.randint(2, period - 2))),
                acct=account,
                old=x,
                new=y,
                holder=person,
            ),
            pick(rng, _SB_CREDIT).format(
                acct=account,
                amount=_cents(rng.randint(150, 2500)),
                label=pick(rng, _SB_CREDIT_LABELS),
            ),
        ]
        if long:
            items.append(
                pick(rng, _SB_LAST).format(
                    acct=account, when=fmt(start), amount=_cents(rng.randint(900, 9000))
                )
            )
        distractors.append(" ".join(items))
    distractors.append(
        pick(rng, _SB_CREDIT_USED).format(
            label=labels[-1],
            amount=_cents(used_credit),
            acct=acct,
            when=fmt(start),
        )
    )
    distractors.append(
        pick(rng, _SB_BOOKED).format(
            acct=acct,
            old=plan_b,
            new=pick(rng, [tier for tier in tiers if tier != plan_b]),
            when=fmt(issue + timedelta(days=rng.randint(3, 20))),
        )
    )
    old_last = promo_first - timedelta(days=rng.randint(1, 20))
    old_plans = rng.sample(tiers, rng.randint(1, 2))
    distractors.append(
        pick(rng, _SB_PROMO).format(
            season=old_season,
            first=fmt(_sb_month_before(old_last)),
            last=fmt(old_last),
            plans=join_list([tier for tier in tiers if tier in old_plans], "or"),
            amount=_price(old_promo),
            company=company,
        )
    )
    if base != "score":
        distractors.append(last_text)
    notes = [
        template.format(
            ticket=f"#{rng.randint(10000, 99999)}",
            person=pick(rng, crowd[2:]),
            acct=acct,
        )
        for template in _SB_NOTES
    ]
    distractors += notes if long else notes[:1]
    rng.shuffle(distractors)

    # question, options and claims
    given = holder.split()[0]
    fields = {"name": holder, "acct": acct, "issue": fmt(issue), "given": given}
    if base == "noul" and detail is not None:
        fields["threshold"] = _price(detail)
    if base == "score":
        fields["width"] = _price(detail[0])
    question = pick(
        rng, _SB_QUESTION[("noul", polarity) if base == "noul" else base]
    ).format(**fields)
    if base == "choice":
        choices = tuple(_cents(value) for value in detail)
        gold, wrong_answer = detail.index(net), detail.index(wrong_net)
    elif base == "noul":
        choices, gold, wrong_answer = (), target, 1 - target
    else:
        width, previous = detail
        choices = tuple(level.format(width=_price(width)) for level in _SB_LEVELS)
        gold, wrong_answer = target, plan["wrong_level"]
        if (
            _sb_band(net - previous, width) != gold
            or _sb_band(wrong_net - previous, width) != wrong_answer
        ):
            raise GenerationError(
                "subscription_bill: band placement does not match the plan"
            )
    conclusion_variant = rng.randrange(
        len(_SB_CONCLUDE_CHOICE) if base == "choice" else 2
    )
    rationale_variant = rng.randrange(len(_SB_RATIONALE))

    def claim(
        answer: int, total: int, charged: int, credited: int, label: str
    ) -> Claim:
        values = {**fields, "total": _cents(total)}
        if base == "choice":
            conclusion = _SB_CONCLUDE_CHOICE[conclusion_variant].format(**values)
        elif base == "noul":
            conclusion = _SB_CONCLUDE_NOUL[(polarity, answer)][
                conclusion_variant
            ].format(**values)
        else:
            conclusion = _SB_CONCLUDE_SCORE[conclusion_variant][answer].format(**values)
        rationale = _SB_RATIONALE[rationale_variant].format(
            charges=_cents(charged), credits=_cents(credited)
        )
        return Claim(
            answer=answer, conclusion=conclusion, rationale=rationale, mechanism=label
        )

    right = claim(gold, net, charges, other + promo * promo_on, "correct")
    wrong = claim(
        wrong_answer,
        wrong_net,
        wrong_charges,
        other + promo * wrong_flag,
        plan["mechanism"],
    )
    bare = "\n\n".join(evidence)
    recheck = _sb_recheck(
        "\n\n".join(evidence + distractors), acct, base, question, choices
    )
    if _sb_recheck(bare, acct, base, question, choices) != recheck or any(
        _sb_recheck(bare + "\n\n" + block, acct, base, question, choices) != recheck
        for block in distractors
    ):
        raise GenerationError("subscription_bill: a distractor changed the answer")
    need = (
        max(0, LONG_POOL_CHARS - sum(len(block) + 2 for block in distractors))
        if long
        else 0
    )
    filler_names = people(rng, 6, exclude=names)
    filler = (
        _filler(
            rng,
            _SB_FILLER,
            _filler_slots(rng, filler_names, {"company": company}),
            need,
        )
        if need
        else ()
    )
    facts = {
        "company": company,
        "service": service,
        "account": acct,
        "holder": holder,
        "period": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "issue": issue.isoformat(),
            "days": period,
        },
        "switch": {
            "date": switch.isoformat(),
            "before": plan_a,
            "after": plan_b,
            "days_before": days_a,
            "days_after": days_b,
            "upgrade": plan["upgrade"],
        },
        "list_prices": dict(listed),
        "prices_in_force": {plan_a: pa, plan_b: pb},
        "notices": [
            {
                "plan": n["plan"],
                "old": n["old"],
                "new": n["new"],
                "effective": n["eff"].isoformat(),
                "sent": n["sent"].isoformat(),
                "in_force": n["eff"] <= start,
            }
            for n in notices
        ],
        "notice_role": role,
        "promo": {
            "amount": promo,
            "plans": promo_plans,
            "first": promo_first.isoformat(),
            "last": promo_last.isoformat(),
            "applies": promo_on,
            "pattern": list(plan["promo"]),
        },
        "credits": parts,
        "charges": charges,
        "credit_total": other + promo * promo_on,
        "net": net,
        "wrong": {
            "mechanism": plan["mechanism"],
            "detail": detail_name,
            "charges": wrong_charges,
            "credit_total": other + promo * wrong_flag,
            "net": wrong_net,
        },
        "direction": plan["direction"],
        "polarity": polarity,
        "threshold": detail if base == "noul" else None,
        "previous_invoice": previous,
        "band_width": detail[0] if base == "score" else None,
        "names": sorted(filler_names) if filler else [],
    }
    return F1World(
        kind="subscription_bill",
        base=base,
        subject=f"{holder}'s next invoice",
        question=question,
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=filler,
        right=right,
        wrong=wrong,
        decisive=tuple(decisive),
        facts=facts,
        variant=f"h{header_index}-r{rule_index}-{price_style}-k{conclusion_variant}-q{rationale_variant}",
        roles=_SB_ROLES,
    )


# re-check: separately written parser and Decimal solver for subscription_bill
_SB_ACCT_RX = r"AC-\d{5}"
_SB_PLAN_RX = r"[A-Z][a-z]+"
_SB_COMPANY_RX = r"[A-Z][a-z]+(?: [A-Z][a-z]+)?"
_SB_LABEL_RX = r"[a-z][a-z\-]*(?: [a-z\-]+)*"
_SB_RX_HEADER = [
    _rx(
        t,
        acct=_SB_ACCT_RX,
        name=_PERSON_RX,
        company=_SB_COMPANY_RX,
        service=r"[a-z][a-z\-]*(?: [a-z\-]+)?",
        start=_DATE_RX,
        end=_DATE_RX,
        issue=_DATE_RX,
    )
    for t in _SB_HEADER
]
_SB_RX_PRICE_HEAD = [
    _rx(t, company=_SB_COMPANY_RX, listed=_DATE_RX) for t in _SB_PRICE_HEAD
]
_SB_RX_PRICE_LINE = [_rx(t, plan=_SB_PLAN_RX, price=_MONEY_RX) for t in _SB_PRICE_LINE]
_SB_RX_PRICE_PROSE = [
    _rx(t, company=_SB_COMPANY_RX, listed=_DATE_RX, pairs=r"[A-Z].*")
    for t in _SB_PRICE_PROSE
]
_SB_RX_NOTICE = [
    _rx(
        t,
        sent=_DATE_RX,
        eff=_DATE_RX,
        plan=_SB_PLAN_RX,
        old=_MONEY_RX,
        new=_MONEY_RX,
        company=_SB_COMPANY_RX,
    )
    for t in _SB_NOTICE
]
_SB_RX_MOVE = [
    _rx(
        t,
        when=_DATE_RX,
        acct=_SB_ACCT_RX,
        old=_SB_PLAN_RX,
        new=_SB_PLAN_RX,
        holder=_PERSON_RX,
    )
    for t in _SB_CHANGE + _SB_BOOKED
]
_SB_RX_CREDIT = [
    _rx(t, acct=_SB_ACCT_RX, amount=_MONEY_RX, label=_SB_LABEL_RX) for t in _SB_CREDIT
]
_SB_RX_USED = [
    _rx(t, acct=_SB_ACCT_RX, amount=_MONEY_RX, label=_SB_LABEL_RX, when=_DATE_RX)
    for t in _SB_CREDIT_USED
]
_SB_RX_PROMO = [
    _rx(
        t,
        season=r"[A-Z][a-z]+(?: [A-Z][a-z]+)?",
        first=_DATE_RX,
        last=_DATE_RX,
        plans=r"[A-Z][a-z]+(?: or [A-Z][a-z]+)*",
        amount=_MONEY_RX,
        company=_SB_COMPANY_RX,
    )
    for t in _SB_PROMO
]
_SB_RX_LAST = [
    _rx(t, acct=_SB_ACCT_RX, when=_DATE_RX, amount=_MONEY_RX) for t in _SB_LAST
]
_SB_RX_QUESTION = [
    (key[1], _rx(t, name=_PERSON_RX, acct=_SB_ACCT_RX, threshold=_MONEY_RX))
    for key, templates in _SB_QUESTION.items()
    if isinstance(key, tuple)
    for t in templates
]
_SB_RX_WIDTH = _rx(_SB_LEVELS[0], width=_MONEY_RX)


def _sb_money(text: str) -> Decimal:
    return Decimal(text.replace("$", "").replace(",", ""))


def _sb_first(patterns: Sequence[re.Pattern[str]], unit: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.fullmatch(unit)
        if match:
            return match
    return None


def _sb_recheck(
    text: str, acct: str, base: str, question: str, options: Sequence[str]
) -> int:
    period = None
    listed: dict[str, Decimal] = {}
    notices: list[tuple[date, str, Decimal, Decimal]] = []
    moves: list[tuple[date, str, str]] = []
    credits: list[Decimal] = []
    promos: list[tuple[date, date, list[str], Decimal]] = []
    previous = None
    for units in _units(text):
        price_list = _sb_first(_SB_RX_PRICE_HEAD, units[0]) is not None
        for position, unit in enumerate(units):
            if price_list and position > 0:
                match = _sb_first(_SB_RX_PRICE_LINE, unit)
                if match is None:
                    raise GenerationError(
                        f"subscription_bill re-check: unparsed price line {unit!r}"
                    )
                listed[match["plan"]] = _sb_money(match["price"])
                continue
            match = _sb_first(_SB_RX_PRICE_PROSE, unit)
            if match:
                for name, amount in re.findall(
                    r"([A-Z][a-z]+) (\$[\d,]+\.\d{2})", match["pairs"]
                ):
                    listed[name] = _sb_money(amount)
                continue
            match = _sb_first(_SB_RX_HEADER, unit)
            if match:
                if match["acct"] == acct:
                    period = (
                        _read_date(match["start"]),
                        _read_date(match["end"]),
                        _read_date(match["issue"]),
                    )
                continue
            match = _sb_first(_SB_RX_NOTICE, unit)
            if match:
                notices.append(
                    (
                        _read_date(match["eff"]),
                        match["plan"],
                        _sb_money(match["old"]),
                        _sb_money(match["new"]),
                    )
                )
                continue
            match = _sb_first(_SB_RX_MOVE, unit)
            if match:
                if match["acct"] == acct:
                    moves.append(
                        (_read_date(match["when"]), match["old"], match["new"])
                    )
                continue
            match = _sb_first(_SB_RX_CREDIT, unit)
            if match:
                if match["acct"] == acct:
                    credits.append(_sb_money(match["amount"]))
                continue
            if _sb_first(_SB_RX_USED, unit):
                continue
            match = _sb_first(_SB_RX_PROMO, unit)
            if match:
                promos.append(
                    (
                        _read_date(match["first"]),
                        _read_date(match["last"]),
                        match["plans"].split(" or "),
                        _sb_money(match["amount"]),
                    )
                )
                continue
            match = _sb_first(_SB_RX_LAST, unit)
            if match and match["acct"] == acct:
                previous = _sb_money(match["amount"])
    if period is None or not listed:
        raise GenerationError(
            "subscription_bill re-check: billing period or price list not found"
        )
    start, end, issue = period
    days = (end - start).days + 1
    inside = sorted(move for move in moves if start < move[0] <= end)
    if len(inside) != 1 or any(end < move[0] <= issue for move in moves):
        raise GenerationError(
            "subscription_bill re-check: expected exactly one plan change in the period"
        )
    switched, before, after = inside[0]

    def monthly(name: str) -> Decimal:
        value = listed[name]
        for effective, plan_name, old, new in sorted(notices):
            if plan_name == name and effective <= start:
                if old != value:
                    raise GenerationError(
                        "subscription_bill re-check: notice does not match the price list"
                    )
                value = new
        return value

    charged = sum(
        ((monthly(name) * used) / days).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        for name, used in (
            (before, (switched - start).days),
            (after, (end - switched).days + 1),
        )
    )
    live = [
        promo for promo in promos if promo[0] <= issue <= promo[1] and after in promo[2]
    ]
    if len(live) > 1:
        raise GenerationError("subscription_bill re-check: two promotions apply")
    balance = (
        charged
        - sum(credits, Decimal(0))
        - sum((promo[3] for promo in live), Decimal(0))
    )
    if base == "choice":
        hits = [
            index
            for index, option in enumerate(options)
            if _sb_money(option) == balance
        ]
        if len(hits) != 1:
            raise GenerationError(
                "subscription_bill re-check: invoice amount not offered once"
            )
        return hits[0]
    if base == "noul":
        for kind, pattern in _SB_RX_QUESTION:
            match = pattern.fullmatch(question)
            if match:
                break
        else:
            raise GenerationError("subscription_bill re-check: question not recognised")
        if kind in ("above", "at_most"):
            limit = _sb_money(match["threshold"])
            return int(balance > limit) if kind == "above" else int(balance <= limit)
        return int(balance < 0) if kind == "refund" else int(balance > 0)
    match = _SB_RX_WIDTH.fullmatch(options[0])
    if match is None or previous is None:
        raise GenerationError(
            "subscription_bill re-check: bands or previous invoice not found"
        )
    width, change = _sb_money(match["width"]), balance - previous
    return 0 if change < -width else 1 if change <= 0 else 2 if change <= width else 3


def build_subscription_bill(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    _check_args("subscription_bill", base, target, length)
    plan = _sb_plan(rng, base, target)
    return _attempt_loop(
        "subscription_bill", lambda: _sb_draw(rng, base, target, length, plan)
    )


# @@APPEND@@

BUILDERS = {
    "project_timeline": build_project_timeline,
    "directory_route": build_directory_route,
    "rubric_pick": build_rubric_pick,
    "subscription_bill": build_subscription_bill,
}
BASES = {
    "project_timeline": ("choice", "noul", "score"),
    "directory_route": ("choice", "noul"),
    "rubric_pick": ("choice", "noul", "score"),
    "subscription_bill": ("choice", "noul", "score"),
}
SCORE_LEVELS = {"project_timeline": 4, "rubric_pick": 4, "subscription_bill": 4}
