"""F1 kinds (part A): ``order_sla``, ``expense_total``, ``applicant_screen``, ``transit_connection``.

Every builder follows the contract in ``f1_base``: it draws a structured world,
solves it with an oracle, renders the evidence as English prose, re-parses the
rendered evidence and distractor text with patterns tied to the templates below
and re-solves the parsed facts with separately written code (the recheck), and
derives the wrong claim from exactly one error mechanism.

Anti-shortcut rule used throughout: the structural features a mechanism relies
on (a revision block, a holiday, a similarly numbered record, a cap that binds,
a high-cost list, an alternative route to a requirement, a timetable change) are
drawn *before* the target and the mechanism, at fixed rates, and are
always rendered (on the subject or, non-decisively, on another record). The
mechanism is then chosen among those that can produce a wrong answer in the
required direction, so no feature predicts the gold label.

What the options show is planned the same way: Score band layouts, approver
tiers and the gold's rank among ordered Choice values are drawn before the
world and apart from the target, and a draw that misses the plan is redrawn
under the same plan (``_plan_loop``), so the plan keeps the rates it was drawn at.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

from v2.data.hs1.core import (
    LAST_NAMES,
    MONTHS_EN,
    GenerationError,
    add_business_days,
    business_days_between,
    fmt_date,
    fmt_money,
    is_business_day,
    join_list,
    people,
    pick,
)
from v2.data.hs1.f1_base import Claim, F1World

# ---------------------------------------------------------------- shared helpers

_NUM_WORDS = (
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
_MONTH_SRC = (
    "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    "|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_DATE_SRC = rf"\d{{4}}-\d{{2}}-\d{{2}}|(?:{_MONTH_SRC}) \d{{1,2}}, \d{{4}}|\d{{1,2}} (?:{_MONTH_SRC}) \d{{4}}"
_DATE_RX = re.compile(_DATE_SRC)
_MONTH_INDEX = {name[:3]: index + 1 for index, name in enumerate(MONTHS_EN)}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#(])")
_MONEY_RX = re.compile(r"\$(\d[\d,]*(?:\.\d{2})?)")
_DATE_STYLES = ("mdy", "dmy", "dmy_weekday", "mdy_short", "iso")


def _num(rng: random.Random, value: int) -> str:
    """A small count as a word or as digits."""
    if 0 <= value < len(_NUM_WORDS) and rng.random() < 0.5:
        return _NUM_WORDS[value]
    return str(value)


def _read_num(token: str) -> int:
    token = token.lower()
    return _NUM_WORDS.index(token) if token in _NUM_WORDS else int(token)


def _to_date(text: str) -> date:
    text = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year, month, day = (int(part) for part in text.split("-"))
        return date(year, month, day)
    match = re.fullmatch(rf"({_MONTH_SRC}) (\d{{1,2}}), (\d{{4}})", text)
    if match:
        return date(int(match[3]), _MONTH_INDEX[match[1][:3]], int(match[2]))
    match = re.fullmatch(rf"(\d{{1,2}}) ({_MONTH_SRC}) (\d{{4}})", text)
    if match:
        return date(int(match[3]), _MONTH_INDEX[match[2][:3]], int(match[1]))
    raise GenerationError(f"unparseable date {text!r}")


def _dates_in(text: str) -> list[date]:
    return [_to_date(match.group(0)) for match in _DATE_RX.finditer(text)]


def _sentences(block: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT.split(block.strip()) if part]


def _cents_in(text: str) -> list[int]:
    return [round(float(raw.replace(",", "")) * 100) for raw in _MONEY_RX.findall(text)]


def _money(cents: int) -> str:
    return fmt_money(cents / 100, "usd")


def _weighted(rng: random.Random, pairs: Sequence[tuple[Any, float]]) -> Any:
    total = sum(weight for _, weight in pairs)
    mark = rng.random() * total
    for value, weight in pairs:
        mark -= weight
        if mark < 0:
            return value
    return pairs[-1][0]


def _an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _bdays_in(start: date, end: date, holidays: Sequence[date]) -> list[date]:
    """Business days d with start < d <= end."""
    out, current = [], start
    while current < end:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            out.append(current)
    return out


def _back_bdays(value: date, days: int, holidays: Sequence[date]) -> date:
    current, left = value, days
    while left > 0:
        current -= timedelta(days=1)
        if is_business_day(current, holidays):
            left -= 1
    return current


class _Redraw(GenerationError):
    """A draw missed a constraint; builders with a plan redraw the world and keep the plan."""


_PLAN_ATTEMPTS = 400


def _plan_loop(kind: str, draw: Callable[[], F1World]) -> F1World:
    """Redraw under one plan, so that the plan keeps the rates it was drawn at.

    Only ``_Redraw`` is retried here; any other error (an inconsistency) propagates.
    """
    last: Exception | None = None
    for _ in range(_PLAN_ATTEMPTS):
        try:
            return draw()
        except _Redraw as exc:
            last = exc
    raise GenerationError(f"{kind}: no draw met the plan ({last})")


def _rank_plan(rng: random.Random) -> tuple[int, int]:
    """Choice option count and the gold's rank among the ordered option values (uniform)."""
    count = rng.choice((3, 4, 4, 5))
    return count, rng.randrange(count)


def _rank_side_ok(rank: int, count: int, gold: Any, wrong: Any) -> bool:
    """Whether ``wrong`` can sit beside a gold of this rank (a lowest gold needs it above)."""
    return (rank > 0 or wrong > gold) and (rank < count - 1 or wrong < gold)


def _ranked(
    gold: Any, wrong: Any, pool: Sequence[Any], count: int, rank: int
) -> list[Any]:
    """``count`` distinct values with ``gold`` at ``rank`` in ascending order and ``wrong`` among
    them; the others come from ``pool`` in its order, skipping mirror images around the gold
    (which would mark the gold as the midpoint of two options)."""
    below = rank - (1 if wrong < gold else 0)
    above = count - 1 - rank - (1 if wrong > gold else 0)
    if below < 0 or above < 0:
        raise _Redraw("the wrong value is on the wrong side of the planned rank")
    values = [gold, wrong]
    left = {True: below, False: above}
    for value in dict.fromkeys(pool):
        side = value < gold
        if value in values or not left[side] or gold + (gold - value) in values:
            continue
        values.append(value)
        left[side] -= 1
    if any(left.values()):
        raise _Redraw("not enough option values on one side of the gold")
    return values


def _business_day(
    rng: random.Random, start: date = date(2025, 1, 13), span: int = 640
) -> date:
    for _ in range(20):
        value = start + timedelta(days=rng.randrange(span))
        if value.weekday() < 5:
            return value
    raise _Redraw("no business day drawn")


def _similar_code(rng: random.Random, digits: str, taken: set[str]) -> str:
    """A code that differs from ``digits`` by one swap of neighbours or one digit."""
    for _ in range(40):
        index = rng.randrange(len(digits) - 1)
        if rng.random() < 0.5 and digits[index] != digits[index + 1]:
            code = (
                digits[:index] + digits[index + 1] + digits[index] + digits[index + 2 :]
            )
        else:
            index = rng.randrange(1, len(digits))
            code = (
                digits[:index]
                + str((int(digits[index]) + rng.randrange(1, 10)) % 10)
                + digits[index + 1 :]
            )
        if code != digits and code[0] != "0" and code not in taken:
            return code
    raise _Redraw("no similar code")


def _fresh_code(rng: random.Random, width: int, taken: set[str]) -> str:
    for _ in range(100):
        code = str(rng.randrange(10 ** (width - 1), 10**width))
        if code not in taken:
            return code
    raise _Redraw("code pool exhausted")


def _mechanism_pick(
    rng: random.Random, candidates: Sequence[tuple[str, Any]], weights: dict[str, float]
) -> list[tuple[str, Any]]:
    """Order candidate (mechanism, payload) pairs: mechanisms by weight, variants shuffled."""
    by_mech: dict[str, list[tuple[str, Any]]] = {}
    for mech, payload in candidates:
        by_mech.setdefault(mech, []).append((mech, payload))
    ordered: list[tuple[str, Any]] = []
    pool = [(mech, weights.get(mech, 1.0)) for mech in sorted(by_mech)]
    while pool:
        mech = _weighted(rng, pool)
        pool = [pair for pair in pool if pair[0] != mech]
        variants = list(by_mech[mech])
        rng.shuffle(variants)
        ordered.extend(variants)
    return ordered


_PARTNERS = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}


def _partners(rng: random.Random, target: int) -> list[int]:
    """Wrong Score levels for a 4-level gold: middle levels pair with extremes and vice versa,
    so that the wrong claims' levels are as uniform as the gold levels."""
    levels = list(_PARTNERS[target])
    rng.shuffle(levels)
    return levels


# ---------------------------------------------------------------- filler

_SHARED_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Records and retention",
        (
            "Entries in this file are kept for seven years and then destroyed under the retention schedule.",
            "Scanned copies carry the same weight as originals once they have been checked against the source.",
            "Anyone who spots a gap in the file should raise it with the records desk rather than editing an entry.",
            "Corrections are added as new entries, and the entry being corrected stays visible for audit purposes.",
            "The records desk aims to answer queries within two working days.",
            "Access to the archive is logged, and every download is tied to a named user.",
        ),
    ),
    (
        "Data protection",
        (
            "Personal details in this file are used only for the purpose stated when they were collected.",
            "Please do not forward this material to personal email accounts or unapproved messaging apps.",
            "Printed copies should be collected from the printer straight away and shredded after use.",
            "If you believe personal data has been shared by mistake, report it to the privacy team the same day.",
            "Requests from individuals to see their own records go through the privacy team, not through this file.",
            "Screens showing case details should be locked whenever they are left unattended.",
        ),
    ),
    (
        "How to raise a concern",
        (
            "Questions about a decision in this file can be raised with the team lead in the first instance.",
            "If the team lead cannot resolve the point, it goes to the operations manager with a short summary.",
            "Please keep concerns factual and point to the entry you are relying on.",
            "A concern does not pause routine processing unless a manager says so in writing.",
            "Every concern receives a reference so that it can be tracked to a conclusion.",
            "Concerns that are resolved informally should still be noted in the file for future readers.",
        ),
    ),
    (
        "System maintenance",
        (
            "The case system is unavailable for planned maintenance on the first Sunday evening of each month.",
            "Work saved before the maintenance window is kept; unsaved drafts may be lost.",
            "After maintenance, users may need to sign in again and accept the updated terms of use.",
            "Slow searches are usually caused by very broad filters, so narrow the filter before reporting a fault.",
            "Faults can be reported through the service portal, with a screenshot where possible.",
            "The service desk posts a notice on the portal once the system is fully available again.",
        ),
    ),
    (
        "Feedback",
        (
            "We invite short feedback on how this process worked for you.",
            "Feedback is read by the process owner, who collects themes rather than individual comments.",
            "Suggestions that save time for the next reader are especially welcome.",
            "Please avoid including personal details of other people in feedback forms.",
            "A summary of the changes made in response to feedback is published every quarter.",
            "Feedback is optional and does not affect the handling of any individual case.",
        ),
    ),
    (
        "Training reminders",
        (
            "New team members complete the introductory module before handling cases on their own.",
            "Refresher sessions run each quarter and are recorded for anyone who cannot attend live.",
            "The module on reading source records carefully is recommended for everyone who signs off decisions.",
            "Completion is tracked automatically once the short quiz at the end is submitted.",
            "Managers can request a tailored session for their team through the learning portal.",
            "Training material is reviewed every year so that examples stay close to current practice.",
        ),
    ),
    (
        "Accessibility",
        (
            "This material is available in large print and in an audio version on request.",
            "If a format does not work for you, the team will arrange an alternative without delay.",
            "Meeting rooms used for reviews have hearing loops and step-free access.",
            "Documents are checked with a screen reader before they are published internally.",
            "Please tell the organiser in advance if you need any adjustment for a review meeting.",
            "Colour is never the only way information is shown in our forms.",
        ),
    ),
    (
        "Working hours and cover",
        (
            "The team works a rota so that at least two people are available during core hours.",
            "Out-of-hours messages are picked up at the start of the next working day.",
            "Planned absences should be entered in the shared calendar at least a week ahead.",
            "When a colleague is away, their open cases are reassigned by the team lead.",
            "Handovers include a short note on anything that is waiting for a reply.",
            "Urgent matters outside core hours go to the on-call manager by phone.",
        ),
    ),
)


def _paragraphs(
    rng: random.Random,
    topics: Sequence[tuple[str, Sequence[str]]],
    low: int = 4,
    high: int = 6,
) -> list[str]:
    """One filler paragraph per topic: an optional heading and 4-6 of its sentences."""
    out = []
    for title, lines in topics:
        count = min(len(lines), rng.randint(low, high))
        chosen = sorted(rng.sample(range(len(lines)), count))
        body = " ".join(lines[index] for index in chosen)
        style = rng.randrange(3)
        if style == 0:
            out.append(f"{title}. {body}")
        elif style == 1:
            out.append(f"{title}: {body}")
        else:
            out.append(body)
    rng.shuffle(out)
    return out


def _filler(
    rng: random.Random, kind_topics: Sequence[tuple[str, Sequence[str]]]
) -> tuple[str, ...]:
    return tuple(_paragraphs(rng, list(kind_topics) + list(_SHARED_TOPICS)))


def _claim_texts(
    rng: random.Random,
    conclusions: Sequence[str],
    rationales: Sequence[str],
) -> tuple[str, str]:
    """Pick one conclusion and one rationale template for the world (shared by both claims)."""
    return pick(rng, conclusions), pick(rng, rationales)


# ---------------------------------------------------------------- order_sla

_OS_SITES = (
    "the Riverside warehouse",
    "the north depot",
    "the Eastgate fulfilment centre",
    "the Harbour Road site",
    "the Millbrook warehouse",
    "the West Yard depot",
    "the Canal Street hub",
    "the Airport Park depot",
)
_OS_CARRIERS = (
    "Swiftline Parcel",
    "Northway Freight",
    "BlueArrow Logistics",
    "Meridian Courier",
    "Pinecrest Express",
    "Kestrel Delivery",
    "Lantern Post",
    "Corvid Couriers",
)
_OS_HUBS = (
    "the Lakeside cross-dock",
    "the Midlands sort centre",
    "the Southgate depot",
    "the Brookfield transfer point",
    "the Quarry Lane hub",
)
_OS_HOLIDAYS = (
    "Founders' Day",
    "Unity Day",
    "Heritage Day",
    "Charter Day",
    "Settlers' Day",
    "the regional civic holiday",
    "the midsummer holiday",
    "Harbour Day",
)
_OS_SUFFIX = (
    "Supplies",
    "Trading",
    "Interiors",
    "Workshop",
    "Clinics",
    "Outfitters",
    "Studio",
    "Bakery",
    "Garden Centre",
    "Hardware",
    "Opticians",
    "Print Shop",
)
_OS_FORMS = (("bdays", 0.3), ("cdays", 0.2), ("by", 0.25), ("before", 0.25))
_OS_EXTEND = (
    "after a reroute through {hub}",
    "because of a missed trunk connection at {hub}",
    "after the carrier reported a vehicle fault",
    "after a weather hold at {hub}",
)
_OS_ADVANCE = (
    "after {customer} upgraded the order to priority handling",
    "after the account team booked an express service",
    "after the order was moved to the carrier's next-day network",
)
_OS_ROLES = (
    "customer service agent",
    "logistics coordinator",
    "account manager",
    "dispatch supervisor",
    "operations analyst",
    "claims handler",
)
_OS_LEVELS = (
    "On time: delivered on or before the last on-time day",
    "Late by one business day",
    "Late by two or three business days",
    "Late by four or more business days",
)
_OS_LEVEL_WORDS = (
    "on time",
    "one business day late",
    "two or three business days late",
    "four or more business days late",
)
_OS_WEIGHTS = {
    "noul": {
        "stale_value": 3.0,
        "arithmetic_slip": 1.2,
        "boundary_misread": 2.2,
        "entity_swap": 0.25,
    },
    "score": {
        "stale_value": 1.6,
        "arithmetic_slip": 1.6,
        "boundary_misread": 1.5,
        "entity_swap": 0.3,
    },
    "choice": {
        "stale_value": 1.0,
        "arithmetic_slip": 1.4,
        "boundary_misread": 0.8,
        "entity_swap": 1.2,
    },
}

_OS_TERMS = (
    "Delivery terms. A window stated in business days counts the business days after the dispatch "
    "date, whatever the time of the dispatch scan; business days run Monday to Friday and exclude "
    "public holidays. A window stated in calendar days counts every day after the dispatch date. A "
    "promise of delivery by the end of a date includes that date, whereas a promise of delivery before "
    "a date must be met by the end of the previous day. When a promise is revised, the revised promise "
    "replaces the earlier one.",
    "How delivery promises are measured: the dispatch date itself is never counted, even for "
    "late-evening scans. Business-day windows skip Saturdays, Sundays and public holidays, while "
    "calendar-day windows count every day. With a promise of delivery by the end of a date, that date "
    "still counts; with a promise of delivery before a date, the day before is the last on-time day. "
    "Only the most recent revision of a promise is in force.",
    "Service level rules, in short. Counting starts on the day after dispatch, regardless of the scan "
    "time. Business days are weekdays other than public holidays, and calendar days are all days. "
    "Delivery on the date named in a promise of delivery by the end of that date is on time, and a "
    "promise of delivery before a date ends on the day before the date named. A revision notice "
    "supersedes the promise it revises.",
)
_OS_ORDER = (
    "Order #{id} for {customer} was dispatched from {site} on {dispatch} at {dtime}. The order "
    "confirmation promised delivery {promise}.",
    "Dispatch record: order #{id} ({customer}) was handed to {carrier} at {site} on {dispatch}, with "
    "the dispatch scan at {dtime}. Delivery was promised {promise}.",
    "{customer} placed order #{id}. The consignment was dispatched from {site} on {dispatch} at "
    "{dtime}. The confirmation email promised delivery {promise}.",
    "Order #{id}, {customer}: dispatched on {dispatch} at {dtime} with {carrier}. The customer was "
    "promised delivery {promise}.",
)
_OS_DELIVERY = (
    "Proof of delivery: order #{id} was signed for on {delivered} at {ttime}.",
    "Tracking for order #{id} shows the parcel delivered on {delivered} at {ttime}, signed for at the "
    "front desk.",
    "{carrier} confirmed that order #{id} was delivered to {customer} on {delivered} ({ttime}).",
    "Delivery scan for order #{id}: handed over to {customer} on {delivered} at {ttime}.",
)
_OS_REVISION = (
    "Update issued on {issued}: {reason}, the delivery promise for order #{id} was revised to "
    "delivery {promise}. The earlier promise no longer applies.",
    "Revision notice for order #{id}, issued {issued}: {reason}, the promise was revised to delivery "
    "{promise}.",
    "On {issued} the account team revised the delivery promise for order #{id} {reason}; it now "
    "reads delivery {promise}.",
)
_OS_HOLIDAY = (
    "Carrier notice from {carrier}: {hday} is a public holiday ({hname}). There are no collections "
    "or deliveries that day, and it does not count as a business day.",
    "Reminder: {hname} falls on {hday}. It is a public holiday in the delivery region, so it is not a "
    "business day and no deliveries are made.",
    "Planning note: depots and the carrier network close for {hname}, a public holiday, on {hday}.",
)
_OS_Q = {
    "noul": (
        "Was order #{id} delivered within the delivery promise in force for it?",
        "Did order #{id} arrive on time under its current delivery promise?",
        "Going by these records, did the delivery of order #{id} meet its promised window?",
    ),
    "score": (
        "How late was the delivery of order #{id} against the promise in force?",
        "Rate the lateness of order #{id} relative to its current delivery promise.",
        "Which lateness band does the delivery of order #{id} fall into?",
    ),
    "choice": (
        "What is the last day on which order #{id} could be delivered and still count as on time?",
        "Under the promise in force, by the end of which day did order #{id} have to be delivered?",
        "On which date does the promised delivery window for order #{id} close?",
    ),
}
_OS_NOUL_CONC = (
    (
        "order #{id} was delivered within its promised window",
        "order #{id} missed its promised window",
    ),
    ("order #{id} arrived on time", "order #{id} arrived late"),
    (
        "the delivery of order #{id} met the promise in force",
        "the delivery of order #{id} did not meet the promise in force",
    ),
    (
        "order #{id} made its delivery commitment",
        "order #{id} fell short of its delivery commitment",
    ),
)
_OS_TRACK_RAT = (
    "it was delivered on {delivered} and the window closed at the end of {deadline}",
    "the last on-time day was {deadline} and the delivery scan is dated {delivered}",
    "the promise in force ran to {deadline}, and the parcel was signed for on {delivered}",
    "the proof of delivery is dated {delivered}, against a last on-time day of {deadline}",
)
_OS_SCORE_CONC = (
    "order #{id} was {level}",
    "the delivery of order #{id} should be rated {level}",
    "order #{id} counts as {level}",
)
_OS_CHOICE_CONC = (
    "the last on-time day for order #{id} is {answer}",
    "order #{id} had to be delivered by the end of {answer}",
    "the promised window for order #{id} closes on {answer}",
)
_OS_CHOICE_RAT = (
    "it was dispatched on {dispatch} and the promise in force is delivery {promise}",
    "the count starts from dispatch on {dispatch}, with delivery promised {promise}",
    "dispatch was on {dispatch} and the applicable promise is delivery {promise}",
)
_OS_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Packaging standards",
        (
            "Fragile items are double-boxed with at least five centimetres of padding on every side.",
            "Liquids travel upright in sealed inner bags, with the orientation arrows facing up.",
            "Each parcel carries a packing slip inside and a shipping label on the largest face.",
            "Reused boxes are fine as long as old labels and barcodes are fully covered.",
            "Heavy parcels are marked with a team-lift sticker so the carrier can plan the drop.",
            "Packing stations restock tape and void fill at the start of every shift.",
        ),
    ),
    (
        "Returns",
        (
            "Customers can start a return from their account page or by replying to the confirmation email.",
            "Returned goods are inspected at the returns bench before any refund is issued.",
            "Items returned in their original packaging are restocked the same week.",
            "A prepaid return label is offered for faulty goods and for items sent in error.",
            "Refunds go back to the original payment method once inspection is complete.",
            "Returns that arrive without a reference are held for a short time while the team traces them.",
        ),
    ),
    (
        "Damaged or missing items",
        (
            "If a parcel arrives damaged, customers are asked to keep the packaging and send photographs.",
            "Claims for missing items are checked against the packing record and the weight at dispatch.",
            "Replacement stock is reserved while a claim is open so that it can ship quickly.",
            "Carrier claims are filed by the claims team, not by the customer.",
            "A short note on the outcome is added to the order history once a claim closes.",
            "Repeated damage on the same route is reported to the carrier account manager.",
        ),
    ),
    (
        "Address changes",
        (
            "Delivery addresses can be changed free of charge until the order is picked.",
            "After dispatch, address changes depend on the carrier and may not always be possible.",
            "The team never changes an address on the strength of an unverified phone call.",
            "Customers moving premises are encouraged to update their account before placing new orders.",
            "Redirected parcels keep their original tracking reference.",
            "Business addresses should include the floor or unit so that drivers find the right door.",
        ),
    ),
    (
        "Customs paperwork",
        (
            "Shipments leaving the customs area travel with a commercial invoice in three copies.",
            "Commodity codes are checked by the trade compliance team before booking.",
            "Customers importing goods may be contacted by the carrier to settle duties.",
            "Paperwork queries are answered by the trade desk rather than by the warehouse.",
            "Incomplete paperwork is the most common reason for a parcel being held at a border.",
            "Copies of export documents are stored with the order record.",
        ),
    ),
    (
        "Carrier contacts",
        (
            "Each carrier has a named account manager who handles escalations for our sites.",
            "Collection times vary by site and are posted on the loading bay door.",
            "Drivers sign the collection manifest, and a copy is kept in the dispatch office.",
            "Missed collections are reported to the carrier on the same day.",
            "Carrier performance is reviewed monthly using on-time and damage figures.",
            "Changes to carrier contracts are announced to the dispatch teams by email.",
        ),
    ),
    (
        "Sustainability",
        (
            "Our sites are moving to recycled void fill and paper tape across all packing stations.",
            "Consolidating several small orders for the same customer reduces the number of vehicle trips.",
            "Carriers report their fleet emissions to us every quarter.",
            "Customers can opt in to fewer, larger deliveries from their account settings.",
            "Pallet wrap is collected and returned to the supplier for recycling.",
            "Energy use at each warehouse is reviewed alongside the monthly performance report.",
        ),
    ),
    (
        "Stock and backorders",
        (
            "Items on backorder are listed separately on the confirmation so that customers can plan.",
            "Partial shipments are sent only when the customer agrees to them.",
            "Stock levels on the website refresh every few minutes from the warehouse system.",
            "Discontinued lines are marked clearly, and remaining stock is sold on a first-come basis.",
            "Customers are told by email when a backordered item is allocated.",
            "Seasonal lines are counted separately during the quarterly stock take.",
        ),
    ),
)


def _os_deadline(
    dispatch: date, promise: tuple[str, Any], holidays: Sequence[date]
) -> date:
    form, value = promise
    if form == "bdays":
        return add_business_days(dispatch, value, holidays)
    if form == "cdays":
        return dispatch + timedelta(days=value)
    if form == "by":
        return value
    return value - timedelta(days=1)


def _os_level(late: int) -> int:
    return 0 if late <= 0 else 1 if late == 1 else 2 if late <= 3 else 3


def _os_draw_promise(
    rng: random.Random, form: str, dispatch: date, holidays: Sequence[date]
) -> tuple[str, Any]:
    if form == "bdays":
        return ("bdays", rng.randint(3, 7))
    if form == "cdays":
        return ("cdays", rng.randint(4, 10))
    if form == "by":
        return ("by", add_business_days(dispatch, rng.randint(3, 7), holidays))
    return ("before", add_business_days(dispatch, rng.randint(4, 8), holidays))


def _os_promise_for(
    form: str, dispatch: date, holidays: Sequence[date], deadline: date
) -> tuple[str, Any] | None:
    """Express ``deadline`` as a promise of the given form, if that form can."""
    if form == "bdays":
        count = business_days_between(dispatch, deadline, holidays)
        if count >= 2 and add_business_days(dispatch, count, holidays) == deadline:
            return ("bdays", count)
        return None
    if form == "cdays":
        count = (deadline - dispatch).days
        return ("cdays", count) if count >= 2 else None
    if form == "by":
        return ("by", deadline) if is_business_day(deadline, holidays) else None
    stated = deadline + timedelta(days=1)
    return ("before", stated) if is_business_day(stated, holidays) else None


def _os_revised(
    rng: random.Random,
    dispatch: date,
    holidays: Sequence[date],
    deadline: date,
    later: bool,
    first: str,
) -> tuple[str, Any]:
    """A promise 1-3 business days later (or earlier) than ``deadline``, preferring form ``first``."""
    gap = rng.randint(1, 3)
    day = (
        add_business_days(deadline, gap, holidays)
        if later
        else _back_bdays(deadline, gap, holidays)
    )
    forms = [first] + [form for form, _ in _OS_FORMS if form != first]
    for form in forms:
        promise = _os_promise_for(form, dispatch, holidays, day)
        if promise is not None:
            return promise
    raise _Redraw("order_sla: no revised promise")


def _os_promise_text(rng: random.Random, promise: tuple[str, Any], style: str) -> str:
    form, value = promise
    if form == "bdays":
        return f"within {_num(rng, value)} business days of dispatch"
    if form == "cdays":
        return f"within {_num(rng, value)} calendar days of dispatch"
    if form == "by":
        return pick(rng, ("by the end of {d}", "by close of business on {d}")).format(
            d=fmt_date(value, style)
        )
    return f"before {fmt_date(value, style)}"


def _os_slips(
    dispatch: date, promise: tuple[str, Any], holidays: Sequence[date]
) -> list[tuple[str, Any]]:
    """Deadline-shifting slips on the promise in force: (mechanism, (variant, wrong deadline))."""
    form, value = promise
    true = _os_deadline(dispatch, promise, holidays)
    out: dict[str, date] = {}
    if form == "bdays":
        out["calendar_count"] = dispatch + timedelta(days=value)
        out["holiday_counted"] = add_business_days(dispatch, value, ())
        out["dispatch_day_counted"] = add_business_days(dispatch, value - 1, holidays)
        out["late_scan_start"] = add_business_days(dispatch, value + 1, holidays)
    elif form == "cdays":
        out["business_count"] = add_business_days(dispatch, value, holidays)
        out["dispatch_day_counted"] = dispatch + timedelta(days=value - 1)
    elif form == "by":
        out["read_as_before"] = value - timedelta(days=1)
    else:
        out["read_as_by"] = value
    mech = "boundary_misread" if form in ("by", "before") else "arithmetic_slip"
    return [(mech, (name, day)) for name, day in out.items() if day != true]


def _os_time(rng: random.Random, low: int, high: int) -> str:
    return f"{rng.randint(low, high):02d}:{rng.randrange(0, 60, 5):02d}"


def _os_order(rng: random.Random, code: str, dispatch: date) -> dict[str, Any]:
    return {
        "code": code,
        "customer": f"{pick(rng, LAST_NAMES)} {pick(rng, _OS_SUFFIX)}",
        "site": pick(rng, _OS_SITES),
        "carrier": pick(rng, _OS_CARRIERS),
        "dispatch": dispatch,
        "dtime": _os_time(rng, 7, 21),
        "ttime": _os_time(rng, 8, 18),
    }


def _os_order_text(
    rng: random.Random, order: dict[str, Any], style: str, phrase: str
) -> str:
    return pick(rng, _OS_ORDER).format(
        id=order["code"],
        customer=order["customer"],
        site=order["site"],
        carrier=order["carrier"],
        dispatch=fmt_date(order["dispatch"], style),
        dtime=order["dtime"],
        promise=phrase,
    )


def _os_delivery_text(rng: random.Random, order: dict[str, Any], style: str) -> str:
    return pick(rng, _OS_DELIVERY).format(
        id=order["code"],
        customer=order["customer"],
        carrier=order["carrier"],
        delivered=fmt_date(order["delivered"], style),
        ttime=order["ttime"],
    )


def _os_revision_text(
    rng: random.Random,
    order: dict[str, Any],
    style: str,
    issued: date,
    phrase: str,
    later: bool,
) -> str:
    reason = pick(rng, _OS_EXTEND if later else _OS_ADVANCE).format(
        hub=pick(rng, _OS_HUBS), customer=order["customer"]
    )
    return pick(rng, _OS_REVISION).format(
        issued=fmt_date(issued, style), reason=reason, id=order["code"], promise=phrase
    )


def _os_distractors(
    rng: random.Random,
    count: int,
    around: date,
    holidays: Sequence[date],
    style: str,
    taken: set[str],
) -> list[str]:
    blocks = []
    for _ in range(count):
        code = _fresh_code(rng, 5, taken)
        taken.add(code)
        dispatch = add_business_days(
            _back_bdays(around, rng.randint(0, 8), holidays),
            rng.randint(0, 8),
            holidays,
        )
        order = _os_order(rng, code, dispatch)
        promise = _os_draw_promise(rng, _weighted(rng, _OS_FORMS), dispatch, holidays)
        parts = [
            _os_order_text(rng, order, style, _os_promise_text(rng, promise, style))
        ]
        deadline = _os_deadline(dispatch, promise, holidays)
        if rng.random() < 0.3:
            later = rng.random() < 0.5
            new = _os_revised(
                rng, dispatch, holidays, deadline, later, _weighted(rng, _OS_FORMS)
            )
            issued = add_business_days(dispatch, 1, holidays)
            if issued < _os_deadline(dispatch, new, holidays):
                parts.append(
                    f"On {fmt_date(issued, style)} the promise was revised to delivery "
                    f"{_os_promise_text(rng, new, style)}."
                )
                deadline = _os_deadline(dispatch, new, holidays)
        order["delivered"] = rng.choice(
            _bdays_in(dispatch, deadline + timedelta(days=5), holidays)
        )
        parts.append(_os_delivery_text(rng, order, style))
        blocks.append(" ".join(parts))
    return blocks


# recheck: separately written parser and solver over the rendered text

_OS_ID_RX = re.compile(r"#(\d{5})\b")
_OS_PROMISE_RX = re.compile(
    rf"within (\w+) (business|calendar) days of dispatch"
    rf"|by (?:the end of|close of business on) (?:[A-Z][a-z]+day )?({_DATE_SRC})"
    rf"|before (?:[A-Z][a-z]+day )?({_DATE_SRC})"
)


def _os_parse_promise(match: re.Match[str]) -> tuple[str, Any]:
    if match.group(1):
        return (match.group(2), _read_num(match.group(1)))
    if match.group(3):
        return ("by", _to_date(match.group(3)))
    return ("before", _to_date(match.group(4)))


def _os_parse(text: str) -> tuple[dict[str, dict[str, Any]], set[date]]:
    orders: dict[str, dict[str, Any]] = {}
    holidays: set[date] = set()
    for block in text.split("\n\n"):
        if re.search(r"\bpublic holiday\b", block):
            holidays.update(_dates_in(block))
            continue
        found = _OS_ID_RX.search(block)
        if not found:
            continue
        record = orders.setdefault(found.group(1), {"revisions": []})
        for sentence in _sentences(block):
            promise = _OS_PROMISE_RX.search(sentence)
            if promise and "revised" in sentence:
                record["revisions"].append(
                    (_dates_in(sentence)[0], _os_parse_promise(promise))
                )
            elif promise and "promised" in sentence:
                record["promise"] = _os_parse_promise(promise)
            elif re.search(r"\bdelivered\b|signed for|handed over", sentence):
                record["delivered"] = _dates_in(sentence)[0]
            elif "dispatch" in sentence and _dates_in(sentence):
                record["dispatch"] = _dates_in(sentence)[0]
    return orders, holidays


def _os_recheck(text: str, code: str, base: str, choices: Sequence[str]) -> int:
    orders, holidays = _os_parse(text)
    record = orders.get(code, {})
    if "dispatch" not in record or "promise" not in record:
        return -1
    revisions = sorted(record["revisions"], key=lambda item: item[0])
    form, value = revisions[-1][1] if revisions else record["promise"]
    start = record["dispatch"]
    if form == "business":
        last, left = start, value
        while left:
            last += timedelta(days=1)
            if last.isoweekday() < 6 and last not in holidays:
                left -= 1
    elif form == "calendar":
        last = start + timedelta(days=value)
    elif form == "by":
        last = value
    else:
        last = value - timedelta(days=1)
    if base == "choice":
        hits = [
            index
            for index, text_ in enumerate(choices)
            if _dates_in(text_)[:1] == [last]
        ]
        return hits[0] if len(hits) == 1 else -1
    arrived = record.get("delivered")
    if arrived is None:
        return -1
    late, day = 0, last
    while day < arrived:
        day += timedelta(days=1)
        if day.isoweekday() < 6 and day not in holidays:
            late += 1
    if base == "noul":
        return 1 if late == 0 else 0
    return 0 if late == 0 else 1 if late == 1 else 2 if late < 4 else 3


def _build_order_sla(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    plan = _rank_plan(rng) if base == "choice" else None
    return _plan_loop(
        "order_sla", lambda: _draw_order_sla(rng, base, target, length, plan)
    )


def _draw_order_sla(
    rng: random.Random,
    base: str,
    target: int | None,
    length: str,
    plan: tuple[int, int] | None,
) -> F1World:
    style = pick(rng, _DATE_STYLES)
    eff_form = _weighted(rng, _OS_FORMS)
    orig_form = _weighted(rng, _OS_FORMS)
    rev_subject = rng.random() < 0.6
    rev_later = rng.random() < 0.5
    sim_later = rng.random() < 0.5
    hol_in = rng.random() < 0.6
    taken: set[str] = set()
    code = _fresh_code(rng, 5, taken)
    taken.add(code)
    sim_code = _similar_code(rng, code, taken)
    taken.add(sim_code)
    dispatch = _business_day(rng)

    promise = _os_draw_promise(rng, eff_form, dispatch, ())
    plain = _os_deadline(dispatch, promise, ())
    stated = promise[1] if eff_form in ("by", "before") else None
    if hol_in:
        slots = [day for day in _bdays_in(dispatch, plain, ()) if day != stated]
        if not slots:
            raise _Redraw("order_sla: no holiday slot")
        holiday = rng.choice(slots)
    elif rng.random() < 0.5:
        holiday = _back_bdays(dispatch, rng.randint(3, 9), ())
    else:
        holiday = add_business_days(plain, rng.randint(10, 16), ())
    hol = (holiday,)
    deadline = _os_deadline(dispatch, promise, hol)
    issued = add_business_days(dispatch, 1, hol)

    subject = _os_order(rng, code, dispatch)
    orig, d_orig = None, None
    if rev_subject:
        # the original promise was revised to ``promise``: later (extension) or earlier (upgrade)
        orig = _os_revised(rng, dispatch, hol, deadline, not rev_later, orig_form)
        d_orig = _os_deadline(dispatch, orig, hol)
        if not (issued < d_orig and issued < deadline) or d_orig == deadline:
            raise _Redraw("order_sla: revision too early")

    sim_dispatch = add_business_days(
        _back_bdays(dispatch, rng.randint(0, 2), hol), rng.randint(0, 2), hol
    )
    similar = _os_order(rng, sim_code, sim_dispatch)
    sim_promise = _os_draw_promise(rng, _weighted(rng, _OS_FORMS), sim_dispatch, hol)
    sim_new = None
    if not rev_subject:
        sim_new = _os_revised(
            rng,
            sim_dispatch,
            hol,
            _os_deadline(sim_dispatch, sim_promise, hol),
            sim_later,
            _weighted(rng, _OS_FORMS),
        )
        if add_business_days(sim_dispatch, 1, hol) >= _os_deadline(
            sim_dispatch, sim_new, hol
        ):
            raise _Redraw("order_sla: similar revision too early")
    d_sim = _os_deadline(sim_dispatch, sim_new or sim_promise, hol)

    slips = _os_slips(dispatch, promise, hol)
    stale = [("stale_value", ("original_promise", d_orig))] if rev_subject else []
    right_vals: dict[str, date] = {}
    wrong_vals: dict[str, date] = {}
    choice_dates: list[date] = []
    mech = variant = None
    delivered = sim_delivered = None
    if base == "noul":
        cands = [
            pair
            for pair in stale + slips
            if (target == 1 and pair[1][1] < deadline)
            or (target == 0 and pair[1][1] > deadline)
        ] + [("entity_swap", ("similar_order", None))]
        for mech, (variant, shifted) in _mechanism_pick(rng, cands, _OS_WEIGHTS[base]):
            if mech == "entity_swap":
                on_time = _bdays_in(dispatch, deadline, hol)
                late = _bdays_in(deadline, deadline + timedelta(days=8), hol)
                mine, theirs = (on_time, late) if target == 1 else (late, on_time)
                theirs = [day for day in theirs if day > sim_dispatch]
                if mine and theirs:
                    delivered, sim_delivered = rng.choice(mine), rng.choice(theirs)
                    wrong_vals = {"deadline": deadline, "delivered": sim_delivered}
                    break
            else:
                low, high = (shifted, deadline) if target == 1 else (deadline, shifted)
                pool = [day for day in _bdays_in(low, high, hol) if day > dispatch]
                if pool:
                    delivered = rng.choice(pool)
                    wrong_vals = {"deadline": shifted, "delivered": delivered}
                    break
        else:
            raise _Redraw("order_sla: no noul mechanism")
        gold, wrong_answer = target, 1 - target
    elif base == "score":

        def pool_for(level: int, after: date) -> list[date]:
            if level == 0:
                return _bdays_in(after, deadline, hol)
            steps = {1: (1,), 2: (2, 3), 3: (4, 5, 6)}[level]
            return [add_business_days(deadline, step, hol) for step in steps]

        mine = pool_for(target, dispatch)
        rng.shuffle(mine)
        found = False
        ordered = _mechanism_pick(
            rng,
            stale + slips + [("entity_swap", ("similar_order", None))],
            _OS_WEIGHTS[base],
        )
        for wanted in _partners(rng, target):
            for mech, (variant, shifted) in ordered:
                if mech == "entity_swap":
                    theirs = [
                        day
                        for day in pool_for(wanted, sim_dispatch)
                        if day > sim_dispatch
                    ]
                    if mine and theirs:
                        delivered, sim_delivered, wrong_answer = (
                            mine[0],
                            rng.choice(theirs),
                            wanted,
                        )
                        wrong_vals = {"deadline": deadline, "delivered": sim_delivered}
                        found = True
                        break
                    continue
                for day in mine:
                    if _os_level(business_days_between(shifted, day, hol)) == wanted:
                        delivered, wrong_answer = day, wanted
                        wrong_vals = {"deadline": shifted, "delivered": day}
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise _Redraw("order_sla: no score mechanism")
        gold = target
    else:
        count, rank = plan
        cands = (
            stale
            + slips
            + ([("entity_swap", ("similar_order", d_sim))] if d_sim != deadline else [])
        )
        ordered = _mechanism_pick(rng, cands, _OS_WEIGHTS[base])
        fits = [
            pair for pair in ordered if _rank_side_ok(rank, count, deadline, pair[1][1])
        ]
        if not fits:
            raise _Redraw(
                "order_sla: no choice mechanism on the side the gold's rank needs"
            )
        mech, (variant, shifted) = fits[0]
        others = [pair[1][1] for pair in ordered if pair is not fits[0]]
        rng.shuffle(others)
        # neighbouring promises of the same form, so that extra dates fall on the weekdays the
        # gold's form allows (a "before" deadline can be a Sunday, a business-day one cannot)
        form, value = promise
        if form in ("bdays", "cdays"):
            near = [
                (form, value + step)
                for step in (-3, -2, -1, 1, 2, 3)
                if value + step >= 1
            ]
        else:
            near = [(form, add_business_days(value, step, hol)) for step in (1, 2, 3)]
            near += [(form, _back_bdays(value, step, hol)) for step in (1, 2, 3)]
        extras = [_os_deadline(dispatch, alternative, hol) for alternative in near]
        rng.shuffle(extras)
        choice_dates = _ranked(deadline, shifted, others + extras, count, rank)
        rng.shuffle(choice_dates)
        delivered = rng.choice(_bdays_in(dispatch, deadline + timedelta(days=5), hol))
        gold, wrong_answer = choice_dates.index(deadline), choice_dates.index(shifted)
    if sim_delivered is None:
        sim_delivered = rng.choice(
            _bdays_in(sim_dispatch, d_sim + timedelta(days=5), hol)
        )
    subject["delivered"], similar["delivered"] = delivered, sim_delivered

    # render
    eff_phrase = _os_promise_text(rng, promise, style)
    orig_phrase = _os_promise_text(rng, orig, style) if orig else None
    sim_phrase = _os_promise_text(rng, sim_promise, style)
    sim_new_phrase = _os_promise_text(rng, sim_new, style) if sim_new else None
    subject_block = _os_order_text(rng, subject, style, orig_phrase or eff_phrase)
    if rev_subject:
        revision = _os_revision_text(rng, subject, style, issued, eff_phrase, rev_later)
    else:
        revision = _os_revision_text(
            rng,
            similar,
            style,
            add_business_days(sim_dispatch, 1, hol),
            sim_new_phrase,
            sim_later,
        )
    holiday_block = pick(rng, _OS_HOLIDAY).format(
        carrier=subject["carrier"],
        hday=fmt_date(holiday, style),
        hname=pick(rng, _OS_HOLIDAYS),
    )
    similar_block = (
        _os_order_text(rng, similar, style, sim_phrase)
        + " "
        + _os_delivery_text(rng, similar, style)
    )
    middle = [revision, holiday_block, similar_block]
    rng.shuffle(middle)
    evidence = [subject_block, *middle, _os_delivery_text(rng, subject, style)]
    terms = pick(rng, _OS_TERMS)
    evidence = [terms, *evidence] if rng.random() < 0.7 else [*evidence, terms]
    distractors = _os_distractors(
        rng, 9 if length == "long" else 3, dispatch, hol, style, taken
    )

    # claims
    ident = {"id": code}
    if base == "noul":
        pair = pick(rng, _OS_NOUL_CONC)
        rationale = pick(rng, _OS_TRACK_RAT)
        right_vals = {"deadline": deadline, "delivered": delivered}
        conc = {1: pair[0].format(**ident), 0: pair[1].format(**ident)}
        right_c, wrong_c = conc[gold], conc[wrong_answer]
    elif base == "score":
        template, rationale = pick(rng, _OS_SCORE_CONC), pick(rng, _OS_TRACK_RAT)
        right_vals = {"deadline": deadline, "delivered": delivered}
        right_c = template.format(level=_OS_LEVEL_WORDS[gold], **ident)
        wrong_c = template.format(level=_OS_LEVEL_WORDS[wrong_answer], **ident)
    else:
        template, rationale = pick(rng, _OS_CHOICE_CONC), pick(rng, _OS_CHOICE_RAT)
        right_c = template.format(answer=fmt_date(deadline, style), **ident)
        wrong_c = template.format(answer=fmt_date(shifted, style), **ident)
    if base == "choice":
        right_r = rationale.format(
            dispatch=fmt_date(dispatch, style), promise=eff_phrase
        )
        if mech == "stale_value":
            wrong_r = rationale.format(
                dispatch=fmt_date(dispatch, style), promise=orig_phrase
            )
        elif mech == "entity_swap":
            wrong_r = rationale.format(
                dispatch=fmt_date(sim_dispatch, style),
                promise=sim_new_phrase or sim_phrase,
            )
        else:
            wrong_r = right_r
    else:
        right_r = rationale.format(
            **{key: fmt_date(value, style) for key, value in right_vals.items()}
        )
        wrong_r = rationale.format(
            **{key: fmt_date(value, style) for key, value in wrong_vals.items()}
        )

    choices = (
        tuple(fmt_date(day, style) for day in choice_dates)
        if base == "choice"
        else _OS_LEVELS if base == "score" else ()
    )
    recheck = _os_recheck("\n\n".join(evidence + distractors), code, base, choices)
    decisive = [fmt_date(dispatch, style), eff_phrase, fmt_date(holiday, style)]
    if base != "choice":
        decisive.append(fmt_date(delivered, style))
    facts = {
        "order": code,
        "similar_order": sim_code,
        "date_style": style,
        "dispatch": dispatch.isoformat(),
        "promise_form": eff_form,
        "promise": [promise[0], str(promise[1])],
        "original_promise": [orig[0], str(orig[1])] if orig else None,
        "revision_on": "subject" if rev_subject else "similar",
        "revision_later": rev_later if rev_subject else sim_later,
        "holiday": holiday.isoformat(),
        "holiday_in_window": hol_in,
        "deadline": deadline.isoformat(),
        "delivered": delivered.isoformat(),
        "similar_delivered": sim_delivered.isoformat(),
        "mechanism_variant": variant,
        "options": [day.isoformat() for day in choice_dates],
    }
    return F1World(
        kind="order_sla",
        base=base,
        subject=f"order #{code}",
        question=pick(rng, _OS_Q[base]).format(**ident),
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=_filler(rng, _OS_TOPICS),
        right=Claim(gold, right_c, right_r, "correct"),
        wrong=Claim(wrong_answer, wrong_c, wrong_r, mech),
        decisive=tuple(decisive),
        facts=facts,
        variant=f"{base}-{eff_form}",
        roles=_OS_ROLES,
    )


# ---------------------------------------------------------------- expense_total

_EX_CITIES = (
    "Lisbon",
    "Oslo",
    "Denver",
    "Toronto",
    "Rotterdam",
    "Lyon",
    "Seattle",
    "Munich",
    "Dublin",
    "Melbourne",
    "Vancouver",
    "Zurich",
    "Boston",
    "Copenhagen",
)
_EX_DEPTS = (
    "Product Marketing",
    "Field Engineering",
    "Customer Success",
    "Research",
    "Procurement",
    "Regional Sales",
    "Quality Assurance",
    "Learning and Development",
)
_EX_PURPOSES = (
    "attend a trade fair",
    "run a customer workshop",
    "visit a supplier",
    "present at a conference",
    "train a partner team",
    "audit a regional office",
)
_EX_HOTELS = (
    "the Harbourview Hotel",
    "the Linden Inn",
    "the Grand Meridian",
    "the Station Lodge",
    "the Parkside Suites",
    "the Old Mill Hotel",
)
_EX_VENUES = (
    "the Copper Kettle",
    "Marlowe's Bistro",
    "the station cafe",
    "the hotel restaurant",
    "Olive and Thyme",
    "the Blue Door Diner",
    "Saffron House",
    "the Lantern Grill",
)
_EX_TRANSPORT = (
    ("a taxi from the airport to the hotel", 1800, 6500),
    ("a train ticket to the customer site", 2400, 12000),
    ("the airport shuttle", 1200, 3000),
    ("parking at the departure airport", 1500, 6000),
    ("a taxi to the venue", 1400, 4500),
)
_EX_SUPPLIES = (
    ("printing of workshop handouts", 1500, 5500),
    ("a replacement laptop adapter", 2500, 5900),
)
_EX_MEALS = (("breakfast", 900, 2200), ("lunch", 1200, 3200), ("dinner", 2200, 7000))
_EX_ROLES = (
    "finance assistant",
    "expenses administrator",
    "team lead",
    "accounts payable clerk",
    "travel coordinator",
    "budget analyst",
)
# Score bands: below LOW, two bands of WIDTH, then LOW + 2 * WIDTH or more (cents).
_EX_BAND_LOW = (50000, 55000, 60000, 65000, 70000, 75000, 80000)
_EX_BAND_WIDTH = (10000, 15000, 20000, 25000)
_EX_HEAD_WIDTH = (5000, 7500, 10000, 12500, 15000, 20000)
_EX_WEIGHTS = {
    "noul": {
        "overlooked_condition": 1.5,
        "arithmetic_slip": 0.5,
        "boundary_misread": 3.0,
        "scope_misapplied": 1.2,
    },
    "approver": {
        "overlooked_condition": 1.6,
        "arithmetic_slip": 0.5,
        "boundary_misread": 1.5,
        "scope_misapplied": 1.0,
    },
    "total": {
        "overlooked_condition": 1.0,
        "arithmetic_slip": 1.0,
        "scope_misapplied": 1.0,
    },
    "score": {
        "overlooked_condition": 2.0,
        "arithmetic_slip": 0.4,
        "scope_misapplied": 1.4,
    },
}
_EX_POLICY = (
    "Travel expense policy, extract. The traveller's own meals are reimbursed up to {meal} per day. "
    "Hotel rooms are reimbursed up to {std} per night, or up to {high} per night in a high-cost city; "
    "room tax is reimbursed in full on top of the capped room rate. The high-cost cities are {cities}. "
    "Client entertainment is not a meal expense: it is reimbursed in full with a guest list, and the "
    "meal cap does not apply to it. Ground transport, conference fees and small supplies are "
    "reimbursed at cost.",
    "Expense rules (summary). Meals: the traveller's own meals up to {meal} per day. Accommodation: "
    "the room rate up to {std} per night, rising to {high} per night in a high-cost city, with room "
    "tax paid in full on top of the capped rate. The high-cost list covers {cities}. Client "
    "entertainment with a guest list is paid in full and is outside the meal cap. Transport, "
    "conference registration and supplies are paid at cost.",
    "What the policy allows: up to {meal} per day for the traveller's own meals; up to {std} per night "
    "for the room rate, or {high} per night in a high-cost city, plus the room tax in full. The "
    "high-cost cities are {cities}. The meal cap covers the traveller's own meals only, so client "
    "entertainment backed by a guest list is reimbursed in full. Transport, registration fees and "
    "supplies are reimbursed as spent.",
)
_EX_Q = {
    "noul": (
        "Does {claimant}'s claim need the approval of the department head?",
        "Under the approval rules, must {head} approve {claimant}'s claim?",
        "Is {claimant}'s claim one that the department head has to approve?",
    ),
    "approver": (
        "Who has to approve {claimant}'s expense claim?",
        "Under the approval rules, who signs off {claimant}'s claim?",
        "Which approver does {claimant}'s claim go to?",
    ),
    "total": (
        "What is the reimbursable total of {claimant}'s claim?",
        "How much should {claimant} be reimbursed for this trip under the policy?",
        "Under the expense policy, what does {claimant}'s claim come to?",
    ),
    "score": (
        "Which reimbursement band does {claimant}'s claim fall into?",
        "Place the reimbursable total of {claimant}'s claim in the correct band.",
        "Rate {claimant}'s claim by its reimbursable total.",
    ),
}
_EX_NOUL_CONC = (
    (
        "{claimant}'s claim needs the department head's approval",
        "{claimant}'s claim can be approved by the line manager",
    ),
    (
        "the claim from {given} has to go to {head}",
        "the claim from {given} can stay with {manager}",
    ),
    (
        "{given}'s claim is at the department-head level",
        "{given}'s claim is at the line-manager level",
    ),
)
_EX_APPROVER_CONC = (
    "{claimant}'s claim must be approved by {answer}",
    "the approver for {given}'s claim is {answer}",
    "{given}'s claim has to be signed off by {answer}",
)
_EX_TOTAL_CONC = (
    "the reimbursable total for {given}'s claim is {answer}",
    "{given} should be reimbursed {answer}",
    "the claim from {given} comes to {answer} under the policy",
)
_EX_SCORE_CONC = (
    "{given}'s reimbursable total is {level}",
    "the claim from {given} lands {level}",
)
_EX_SUM_RAT = (
    "the reimbursable total comes to {total}",
    "after the caps the claim is worth {total}",
    "meals count for {meals} after the daily cap, which puts the total at {total}",
    "the hotel counts for {lodging}, so the whole claim is {total}",
)
_EX_PART_RAT = (
    "meals come to {meals} after the daily cap and the hotel to {lodging}",
    "the hotel nights count for {lodging} and the other items for {other}",
    "meals are allowed at {meals}, and transport and the other items add {other}",
)
_EX_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Receipts",
        (
            "Itemised receipts are needed for every line; card slips on their own are not enough.",
            "Photographs of receipts are accepted if the vendor, the date and the amount are readable.",
            "Receipts in another currency should show the amount charged to the card.",
            "Lost receipts can be replaced by a signed declaration, which the approver may query.",
            "Please keep original receipts until the claim has been paid.",
            "Receipts are matched to card transactions automatically where possible.",
        ),
    ),
    (
        "Booking travel",
        (
            "Flights and rail travel should be booked through the travel portal where possible.",
            "Economy fares are standard; exceptions need approval before booking.",
            "Booking early usually gives lower fares and more flexible tickets.",
            "Hotel bookings made through the portal are billed centrally and do not appear on claims.",
            "Travellers should check the destination guidance before booking.",
            "Changes to bookings are made through the portal so that the itinerary stays accurate.",
        ),
    ),
    (
        "Corporate cards",
        (
            "Corporate cards are for business expenses only and are reconciled every month.",
            "Personal spending on a corporate card must be repaid within the same cycle.",
            "Lost or stolen cards should be reported to the card provider straight away.",
            "Cash withdrawals on corporate cards are discouraged except where cards are not accepted.",
            "Card statements are reviewed by the finance team before month end.",
            "Card limits are set by role and reviewed once a year.",
        ),
    ),
    (
        "Submitting a claim",
        (
            "Claims should be submitted within a month of the end of the trip.",
            "Each claim covers one trip, so that approvers can see the full picture at once.",
            "The claim form adds up the lines, but the policy caps are applied during review.",
            "Claims are paid with the next payroll run after approval.",
            "Queries from the finance team are sent by email and should be answered within a week.",
            "Claims that mix several trips are returned to the traveller for splitting.",
        ),
    ),
    (
        "Duty of care",
        (
            "Travellers should register their trip so that the company can reach them in an emergency.",
            "The travel assistance line is available at all hours during a business trip.",
            "Travellers who feel unsafe may change their plans and explain the change afterwards.",
            "Long driving after a flight is discouraged; a taxi or an overnight stay is preferred.",
            "Health guidance for each destination is published on the travel page.",
            "Managers are expected to check in with team members travelling alone.",
        ),
    ),
    (
        "Currency and payments",
        (
            "Amounts are reimbursed in the currency of the traveller's payroll.",
            "Card conversions use the rate on the card statement.",
            "Where cash was used abroad, the rate on the day of withdrawal applies.",
            "Bank charges for currency conversion can be claimed with evidence.",
            "Reimbursements are paid into the account used for salary.",
            "The finance team can advise on unusual payment situations before travel.",
        ),
    ),
    (
        "Sustainable travel",
        (
            "Rail is preferred to flying for journeys under about four hours.",
            "Virtual meetings should be considered before booking any trip.",
            "Travellers are encouraged to combine visits in the same region into one trip.",
            "Public transport at the destination is preferred where it is practical and safe.",
            "Travel emissions are reported to the leadership team each quarter.",
            "Hotels close to the venue help reduce ground transport.",
        ),
    ),
    (
        "Audit sampling",
        (
            "A sample of approved claims is audited every month.",
            "Audits check that caps were applied and that receipts support each line.",
            "Findings are shared with approvers so that practice stays consistent.",
            "Repeated errors on claims can lead to extra review for a period.",
            "Audit results are summarised for the finance committee.",
            "Travellers may be asked to explain a line even after payment.",
        ),
    ),
)


def _ex_parts(
    items: Sequence[dict[str, Any]],
    lodging: dict[str, Any],
    meal_cap: int,
    room_cap: int,
    *,
    ent_as_meal: bool = False,
    cap_tax: bool = False,
    no_meal_cap: bool = False,
    no_room_cap: bool = False,
    omit: int | None = None,
    double: int | None = None,
) -> dict[str, int]:
    """Reimbursable parts (cents) under the policy, or under one slipped reading of it."""
    days: dict[date, int] = {}
    other = 0
    for index, item in enumerate(items):
        times = 0 if index == omit else 2 if index == double else 1
        if item["cat"] == "meal" or (ent_as_meal and item["cat"] == "ent"):
            days[item["day"]] = days.get(item["day"], 0) + times * item["cents"]
        else:
            other += times * item["cents"]
    meals = sum(
        total if no_meal_cap else min(total, meal_cap) for total in days.values()
    )
    rate, tax = lodging["rate"], lodging["tax"]
    if no_room_cap:
        night = rate + tax
    elif cap_tax:
        night = min(rate + tax, room_cap)
    else:
        night = min(rate, room_cap) + tax
    parts = {"meals": meals, "lodging": lodging["nights"] * night, "other": other}
    parts["total"] = parts["meals"] + parts["lodging"] + parts["other"]
    return parts


def _ex_nice_between(rng: random.Random, low: int, high: int) -> int | None:
    """A round threshold strictly between two totals (cents), preferring coarse steps."""
    for step in (5000, 2500, 1000, 500):
        values = [
            value
            for value in range((low // step + 1) * step, high, step)
            if low < value < high
        ]
        if values:
            return rng.choice(values)
    return None


def _ex_plan(rng: random.Random, form: str) -> dict[str, Any]:
    """What the question and its options show, drawn before the world and independent of the target:
    the score band layout, the approver tiers of the right and wrong claims (each uniform, so the
    quoted tier does not tell them apart) with the department head's band width, or the option ranks.
    """
    if form == "score":
        return {"low": pick(rng, _EX_BAND_LOW), "width": pick(rng, _EX_BAND_WIDTH)}
    if form == "approver":
        gold = rng.randrange(3)
        return {
            "gold": gold,
            "wrong": pick(rng, [tier for tier in range(3) if tier != gold]),
            "gap": pick(rng, _EX_HEAD_WIDTH),
        }
    if form == "total":
        count, rank = _rank_plan(rng)
        return {"count": count, "rank": rank}
    return {}


def _ex_place(
    rng: random.Random,
    total: int,
    wrong: int,
    tiers: tuple[int, int],
    gap: int,
    exact: bool,
) -> tuple[int, int] | None:
    """Thresholds ``(t1, t1 + gap)`` with ``total`` and ``wrong`` in the planned (gold, wrong) tiers.

    With ``exact`` the total itself is the threshold between the two tiers."""
    gold, other = tiers
    low, high = min(total, wrong), max(total, wrong)
    t1: int | None
    if {gold, other} == {0, 1}:
        t1 = total if exact else _ex_nice_between(rng, max(low, high - gap), high)
    elif {gold, other} == {1, 2}:
        t2 = total if exact else _ex_nice_between(rng, low, min(high, low + gap))
        t1 = None if t2 is None else t2 - gap
    elif exact:
        t1 = total if gold == 0 else total - gap
    else:
        t1 = _ex_nice_between(rng, low, high - gap)
    if t1 is None or t1 < 5000:
        return None
    return t1, t1 + gap


def _ex_rule(total: int, threshold: int, or_more: bool) -> bool:
    return total >= threshold if or_more else total > threshold


def _ex_tier(total: int, t1: int, t2: int, or_more: bool) -> int:
    return (
        2 if _ex_rule(total, t2, or_more) else 1 if _ex_rule(total, t1, or_more) else 0
    )


def _ex_approval_text(
    rng: random.Random, t1: int, t2: int | None, or_more: bool, names: dict[str, str]
) -> str:
    def phrase(value: int) -> str:
        return f"{_money(value)} or more" if or_more else f"more than {_money(value)}"

    manager, head, director = names["manager"], names["head"], names["director"]
    if t2 is None:
        return pick(
            rng,
            (
                f"Approval rules. Claims with a reimbursable total of {phrase(t1)} are approved by the "
                f"department head, {head}. Other claims are approved by the line manager, {manager}.",
                f"Sign-off: a claim whose reimbursable total is {phrase(t1)} needs the department head "
                f"({head}); anything else is signed off by the line manager ({manager}).",
                f"Who approves: the line manager, {manager}, approves claims in the normal course. Claims "
                f"with a reimbursable total of {phrase(t1)} go to the department head, {head}, instead.",
            ),
        )
    upper = f"below {_money(t2)}" if or_more else f"not more than {_money(t2)}"
    return pick(
        rng,
        (
            f"Approval rules. Claims with a reimbursable total of {phrase(t2)} are approved by the finance "
            f"director, {director}. Claims of {phrase(t1)} but {upper} are approved by the department head, "
            f"{head}. All other claims are approved by the line manager, {manager}.",
            f"Sign-off levels: the line manager, {manager}, signs off claims in the normal course. A "
            f"reimbursable total of {phrase(t1)} but {upper} needs the department head, {head}. A "
            f"reimbursable total of {phrase(t2)} needs the finance director, {director}.",
        ),
    )


_EX_THRESH_RX = re.compile(
    r"(\$[\d,]+(?:\.\d{2})?) or more|more than (\$[\d,]+(?:\.\d{2})?)"
)


def _ex_parse(text: str, claimant: str) -> dict[str, Any]:
    out: dict[str, Any] = {"tiers": {}, "days": {}, "other": 0, "lodging": []}
    for block in text.split("\n\n"):
        if "high-cost" in block and "per day" in block:
            out["meal_cap"] = _cents_in(
                re.search(r"\$[\d,.]+ per day", block).group(0)
            )[0]
            out["night_caps"] = sorted(
                _cents_in(" ".join(re.findall(r"\$[\d,.]+ per night", block)))
            )
            listing = re.search(
                r"high-cost (?:cities are|list covers) ([^.]+)\.", block
            ).group(1)
            out["cities"] = {
                part.strip() for part in re.split(r",| and ", listing) if part.strip()
            }
        elif "line manager" in block:
            for sentence in _sentences(block):
                role = next(
                    (
                        r
                        for r in ("finance director", "department head")
                        if r in sentence
                    ),
                    None,
                )
                found = [
                    (_cents_in(m.group(1) or m.group(2))[0], m.group(1) is not None)
                    for m in _EX_THRESH_RX.finditer(sentence)
                ]
                if role and found:
                    out["tiers"][role] = min(found)
        elif claimant in _sentences(block)[0] and "trip to" in block:
            out["city"] = re.search(r"trip to ([A-Z][a-z]+)", block).group(1)
            pieces = [piece for line in block.split("\n") for piece in _sentences(line)]
            for piece in pieces:
                amounts = _cents_in(piece)
                if not amounts:
                    continue
                if "room rate" in piece:
                    nights = _read_num(re.search(r"(\w+) nights?\b", piece).group(1))
                    out["lodging"].append((nights, amounts[0], amounts[1]))
                elif re.search(r"\bclient\b", piece, re.I):
                    out["other"] += amounts[-1]
                elif re.search(r"\b(?:breakfast|lunch|dinner)\b", piece, re.I):
                    day = _dates_in(piece)[0]
                    out["days"][day] = out["days"].get(day, 0) + amounts[-1]
                else:
                    out["other"] += amounts[-1]
    return out


def _ex_recheck(text: str, claimant: str, form: str, choices: Sequence[str]) -> int:
    parsed = _ex_parse(text, claimant)
    caps = parsed["night_caps"]
    room_cap = caps[-1] if parsed["city"] in parsed["cities"] else caps[0]
    total = parsed["other"]
    for nights, rate, tax in parsed["lodging"]:
        total += nights * (tax + (rate if rate < room_cap else room_cap))
    for spent in parsed["days"].values():
        total += spent if spent < parsed["meal_cap"] else parsed["meal_cap"]

    def meets(role: str) -> bool:
        if role not in parsed["tiers"]:
            return False
        bound, inclusive = parsed["tiers"][role]
        return total > bound or (inclusive and total == bound)

    if form == "noul":
        return int(meets("department head"))
    if form == "approver":
        role = (
            "finance director"
            if meets("finance director")
            else ("department head" if meets("department head") else "line manager")
        )
        hits = [index for index, option in enumerate(choices) if role in option]
        return hits[0] if len(hits) == 1 else -1
    if form == "total":
        hits = [
            index
            for index, option in enumerate(choices)
            if _cents_in(option) == [total]
        ]
        return hits[0] if len(hits) == 1 else -1
    edges = sorted({cents for option in choices for cents in _cents_in(option)})
    if len(edges) != 3:
        return -1
    return next((level for level, edge in enumerate(edges) if total < edge), 3)


def _ex_items(
    rng: random.Random, days: Sequence[date], flags: dict[str, bool], meal_cap: int
) -> list[dict[str, Any]]:
    for _ in range(40):
        items: list[dict[str, Any]] = []
        for index, day in enumerate(days):
            kinds = list(_EX_MEALS)
            if index == 0:
                kinds = kinds[1:]
            elif index == len(days) - 1:
                kinds = kinds[:2]
            chosen = [meal for meal in kinds if rng.random() < 0.75] or [
                pick(rng, kinds)
            ]
            for name, low, high in chosen:
                items.append(
                    {
                        "cat": "meal",
                        "day": day,
                        "cents": rng.randint(low, high),
                        "desc": f"{name} at {pick(rng, _EX_VENUES)}",
                    }
                )
        totals: dict[date, int] = {}
        for item in items:
            totals[item["day"]] = totals.get(item["day"], 0) + item["cents"]
        if any(total > meal_cap for total in totals.values()) == flags["meal_over"]:
            break
    else:
        raise _Redraw("expense_total: meal draw")
    for desc, low, high in rng.sample(_EX_TRANSPORT, rng.randint(2, 3)):
        items.append(
            {
                "cat": "transport",
                "day": pick(rng, days),
                "cents": rng.randint(low, high),
                "desc": desc,
            }
        )
    if flags["conf"]:
        items.append(
            {
                "cat": "conf",
                "day": days[0],
                "cents": rng.randrange(150, 460, 5) * 100,
                "desc": "the conference registration fee",
            }
        )
    if flags["ent"]:
        company = f"{pick(rng, LAST_NAMES)} {pick(rng, ('Holdings', 'Analytics', 'Partners', 'Engineering'))}"
        items.append(
            {
                "cat": "ent",
                "day": pick(rng, days),
                "cents": rng.randint(9000, 26000),
                "desc": f"a client dinner with guests from {company} at {pick(rng, _EX_VENUES)} (guest list attached)",
            }
        )
    if flags["supplies"]:
        desc, low, high = pick(rng, _EX_SUPPLIES)
        items.append(
            {
                "cat": "supplies",
                "day": pick(rng, days),
                "cents": rng.randint(low, high),
                "desc": desc,
            }
        )
    items.sort(key=lambda item: item["day"])
    return items


def _ex_claim_text(
    rng: random.Random,
    who: str,
    dept: str,
    city: str,
    days: Sequence[date],
    items: Sequence[dict[str, Any]],
    lodging: dict[str, Any] | None,
    style: str,
    purpose: str,
) -> str:
    start, end = fmt_date(days[0], style), fmt_date(days[-1], style)
    listed = rng.random() < 0.5
    stay = ""
    if lodging:
        count = lodging["nights"]
        stay = f"{_num(rng, count)} night{'s' if count > 1 else ''}"
    if listed:
        lines = [
            f"Expense claim from {who}, {dept}, for a trip to {city} from {start} to {end} to {purpose}."
        ]
        if lodging:
            lines.append(
                f"- {start} to {end}: {stay} at {lodging['hotel']}, room rate "
                f"{_money(lodging['rate'])} per night plus {_money(lodging['tax'])} room tax per night"
            )
        for item in items:
            lines.append(
                f"- {fmt_date(item['day'], style)}: {item['desc']}, {_money(item['cents'])}"
            )
        return "\n".join(lines)
    parts = [
        pick(
            rng,
            (
                f"{who} of {dept} claimed the following for a trip to {city} from {start} to {end} to {purpose}.",
                f"Claim summary for {who} ({dept}), trip to {city}, {start} to {end}, to {purpose}.",
            ),
        )
    ]
    if lodging:
        parts.append(
            f"The stay at {lodging['hotel']} ran {stay} at a room rate of "
            f"{_money(lodging['rate'])} per night plus {_money(lodging['tax'])} room tax per night."
        )
    for item in items:
        parts.append(
            pick(
                rng,
                (
                    "On {day}, {desc} cost {amount}.",
                    "{desc} on {day} came to {amount}.",
                ),
            ).format(
                day=fmt_date(item["day"], style),
                desc=item["desc"],
                amount=_money(item["cents"]),
            )
        )
    return " ".join(_cap(part) for part in parts)


def _build_expense_total(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    form = base if base != "choice" else ("approver" if rng.random() < 0.5 else "total")
    plan = _ex_plan(rng, form)
    return _plan_loop(
        "expense_total",
        lambda: _draw_expense_total(rng, base, target, length, form, plan),
    )


def _draw_expense_total(
    rng: random.Random,
    base: str,
    target: int | None,
    length: str,
    form: str,
    plan: dict[str, Any],
) -> F1World:
    style = pick(rng, ("mdy", "dmy", "mdy_short", "dmy_weekday"))
    flags = {
        "listed": rng.random() < 0.5,
        "ent": rng.random() < 0.5,
        "conf": rng.random() < 0.5,
        "supplies": rng.random() < 0.3,
        "or_more": rng.random() < 0.5,
        "exact": rng.random() < 0.4,
        "rate_high": rng.random() < 0.6,
        "meal_over": rng.random() < 0.75,
    }
    claimant, manager, head, director = people(rng, 4)
    names = {"manager": manager, "head": head, "director": director}
    meal_cap = rng.randrange(40, 80, 5) * 100
    std_cap = rng.randrange(140, 210, 10) * 100
    high_cap = std_cap + rng.randrange(40, 90, 10) * 100
    cities = list(_EX_CITIES)
    rng.shuffle(cities)
    high_list = cities[:3]
    city = high_list[rng.randrange(3)] if flags["listed"] else cities[3]
    room_cap = high_cap if flags["listed"] else std_cap
    nights = rng.randint(1, 3)
    first = date(2025, 1, 6) + timedelta(days=rng.randrange(600))
    days = [first + timedelta(days=offset) for offset in range(nights + 1)]
    if flags["rate_high"]:
        rate = std_cap + rng.randint(5, (high_cap - std_cap) // 100 + 30) * 100
    else:
        rate = std_cap - rng.randint(0, 45) * 100
    rate += pick(rng, (0, 0, 0, 50))
    lodging = {
        "nights": nights,
        "rate": rate,
        "tax": round(rate * pick(rng, (0.08, 0.1, 0.12, 0.135, 0.15))),
        "hotel": pick(rng, _EX_HOTELS),
    }
    items = _ex_items(rng, days, flags, meal_cap)
    adjustable = [
        index for index, item in enumerate(items) if item["cat"] == "transport"
    ]

    def parts(**kw: Any) -> dict[str, int]:
        return _ex_parts(items, lodging, meal_cap, kw.pop("room_cap", room_cap), **kw)

    # an exact total sits on the threshold, so "or more" puts it in the upper of the two tiers
    exact = flags["exact"] and (
        (form == "noul" and flags["or_more"] == (target == 1))
        or (form == "approver" and flags["or_more"] == (plan["gold"] > plan["wrong"]))
    )
    threshold = None
    if exact:
        total = parts()["total"]
        threshold = round(total / 5000) * 5000
        index = rng.choice(adjustable)
        amount = items[index]["cents"] + threshold - total
        if not 800 <= amount <= 18000:
            raise _Redraw("expense_total: exact adjustment out of range")
        items[index]["cents"] = amount
    right = parts()
    total = right["total"]

    variants = [
        ("overlooked_condition", "meal_cap_ignored", parts(no_meal_cap=True)),
        ("overlooked_condition", "room_cap_ignored", parts(no_room_cap=True)),
        ("scope_misapplied", "room_cap_on_tax", parts(cap_tax=True)),
    ]
    if flags["listed"]:
        variants.append(
            ("overlooked_condition", "high_cost_overlooked", parts(room_cap=std_cap))
        )
    else:
        variants.append(
            ("scope_misapplied", "high_cost_misapplied", parts(room_cap=high_cap))
        )
    if flags["ent"]:
        variants.append(
            ("scope_misapplied", "meal_cap_on_entertainment", parts(ent_as_meal=True))
        )
    for index, item in enumerate(items):
        if item["cat"] != "meal":
            variants.append(
                ("arithmetic_slip", f"omitted_{item['cat']}", parts(omit=index))
            )
            variants.append(
                ("arithmetic_slip", f"double_{item['cat']}", parts(double=index))
            )
    cands = [
        (mech, (variant, vals))
        for mech, variant, vals in variants
        if vals["total"] != total
    ]

    t1 = t2 = None
    choice_totals: list[int] = []
    if form == "noul":
        want_lower = target == 1
        pool = [pair for pair in cands if (pair[1][1]["total"] < total) == want_lower]
        if exact:
            pool.append(
                (
                    "boundary_misread",
                    (
                        "threshold_"
                        + ("inclusive" if flags["or_more"] else "exclusive"),
                        right,
                    ),
                )
            )
        for mech, (variant, wrong) in _mechanism_pick(rng, pool, _EX_WEIGHTS[form]):
            t1 = (
                threshold
                if exact
                else _ex_nice_between(
                    rng, min(total, wrong["total"]), max(total, wrong["total"])
                )
            )
            if t1 is not None:
                break
        else:
            raise _Redraw("expense_total: no noul mechanism")
        gold = int(_ex_rule(total, t1, flags["or_more"]))
        if gold != target:
            raise _Redraw("expense_total: noul target missed")
        wrong_answer = 1 - gold
    elif form == "approver":
        tiers = (plan["gold"], plan["wrong"])
        higher = tiers[1] > tiers[0]
        pool = [pair for pair in cands if (pair[1][1]["total"] > total) == higher]
        if exact and abs(tiers[0] - tiers[1]) == 1:
            pool.append(
                (
                    "boundary_misread",
                    (
                        "threshold_"
                        + ("inclusive" if flags["or_more"] else "exclusive"),
                        right,
                    ),
                )
            )
        for mech, (variant, wrong) in _mechanism_pick(rng, pool, _EX_WEIGHTS[form]):
            placed = _ex_place(rng, total, wrong["total"], tiers, plan["gap"], exact)
            if placed is None:
                continue
            t1, t2 = placed
            misread = (
                not flags["or_more"] if mech == "boundary_misread" else flags["or_more"]
            )
            if (
                _ex_tier(total, t1, t2, flags["or_more"]),
                _ex_tier(wrong["total"], t1, t2, misread),
            ) == tiers:
                break
        else:
            raise _Redraw("expense_total: no approver mechanism for the planned tiers")
        gold, wrong_answer = tiers
    elif form == "total":
        count, rank = plan["count"], plan["rank"]
        ordered = _mechanism_pick(rng, cands, _EX_WEIGHTS[form])
        fits = [
            pair
            for pair in ordered
            if _rank_side_ok(rank, count, total, pair[1][1]["total"])
        ]
        if not fits:
            raise _Redraw(
                "expense_total: no total slip on the side the gold's rank needs"
            )
        mech, (variant, wrong) = fits[0]
        others = [pair[1][1]["total"] for pair in ordered if pair is not fits[0]]
        rng.shuffle(others)
        claimed = parts(no_meal_cap=True, no_room_cap=True)["total"]
        extras = [claimed] + [
            total + pick(rng, (1, -1)) * item["cents"] for item in items
        ]
        rng.shuffle(extras)
        choice_totals = _ranked(total, wrong["total"], others + extras, count, rank)
        rng.shuffle(choice_totals)
        gold, wrong_answer = choice_totals.index(total), choice_totals.index(
            wrong["total"]
        )
    else:
        bounds = (
            plan["low"],
            plan["low"] + plan["width"],
            plan["low"] + 2 * plan["width"],
        )

        def band_of(value: int) -> int:
            return sum(value >= bound for bound in bounds)

        if band_of(total) != target:
            raise _Redraw("expense_total: total outside the target band")
        ordered = _mechanism_pick(rng, cands, _EX_WEIGHTS[form])
        found = None
        for wanted in _partners(rng, target):
            found = next(
                (pair for pair in ordered if band_of(pair[1][1]["total"]) == wanted),
                None,
            )
            if found:
                break
        if found is None:
            raise _Redraw("expense_total: no slip lands in a partner band")
        mech, (variant, wrong) = found
        gold, wrong_answer = target, band_of(wrong["total"])
    if t1 is None:
        t1 = rng.choice((50000, 75000, 100000, 150000))
    two_tier = form != "approver"

    # render
    policy = pick(rng, _EX_POLICY).format(
        meal=_money(meal_cap),
        std=_money(std_cap),
        high=_money(high_cap),
        cities=join_list(high_list),
    )
    approval = _ex_approval_text(
        rng, t1, None if two_tier else t2, flags["or_more"], names
    )
    dept = pick(rng, _EX_DEPTS)
    claim = _ex_claim_text(
        rng, claimant, dept, city, days, items, lodging, style, pick(rng, _EX_PURPOSES)
    )
    evidence = (
        [policy, approval, claim] if rng.random() < 0.5 else [claim, policy, approval]
    )
    others = people(
        rng, 9 if length == "long" else 3, exclude=[claimant, manager, head, director]
    )
    distractors = []
    for other in others:
        other_days = [date(2025, 1, 6) + timedelta(days=rng.randrange(600))]
        other_days.append(other_days[0] + timedelta(days=1))
        other_items = _ex_items(
            rng,
            other_days,
            {
                "meal_over": rng.random() < 0.5,
                "conf": False,
                "ent": rng.random() < 0.3,
                "supplies": False,
            },
            meal_cap,
        )
        distractors.append(
            _ex_claim_text(
                rng,
                other,
                pick(rng, _EX_DEPTS),
                pick(rng, _EX_CITIES),
                other_days,
                other_items[:5],
                None,
                style,
                pick(rng, _EX_PURPOSES),
            )
        )

    # claims
    ident = {
        "claimant": claimant,
        "given": claimant.split()[0],
        "head": head,
        "manager": manager,
    }
    labels = [
        f"{manager} (line manager)",
        f"{head} (department head)",
        f"{director} (finance director)",
    ]
    spoken = [
        f"{manager}, the line manager",
        f"{head}, the department head",
        f"{director}, the finance director",
    ]
    if form == "noul":
        pair = pick(rng, _EX_NOUL_CONC)
        right_c, wrong_c = (pair[1 - gold].format(**ident), pair[gold].format(**ident))
        choices: tuple[str, ...] = ()
    elif form == "approver":
        template = pick(rng, _EX_APPROVER_CONC)
        right_c = template.format(answer=spoken[gold], **ident)
        wrong_c = template.format(answer=spoken[wrong_answer], **ident)
        choices = tuple(labels)
    elif form == "total":
        template = pick(rng, _EX_TOTAL_CONC)
        right_c = template.format(answer=_money(total), **ident)
        wrong_c = template.format(answer=_money(wrong["total"]), **ident)
        choices = tuple(_money(value) for value in choice_totals)
    else:
        b1, b2, b3 = (_money(bound) for bound in bounds)
        words = (
            f"below {b1}",
            f"in the {b1} to {b2} band",
            f"in the {b2} to {b3} band",
            f"at {b3} or more",
        )
        template = pick(rng, _EX_SCORE_CONC)
        right_c = template.format(level=words[gold], **ident)
        wrong_c = template.format(level=words[wrong_answer], **ident)
        choices = (
            f"Reimbursable total below {b1}",
            f"Reimbursable total of {b1} up to but not including {b2}",
            f"Reimbursable total of {b2} up to but not including {b3}",
            f"Reimbursable total of {b3} or more",
        )
    templates = _EX_PART_RAT if form == "total" else _EX_SUM_RAT
    differing = [
        template
        for template in templates
        if any(right[key] != wrong[key] for key in re.findall(r"\{(\w+)\}", template))
    ]
    rationale = pick(rng, differing or templates)
    right_r = rationale.format(**{key: _money(value) for key, value in right.items()})
    wrong_r = rationale.format(**{key: _money(value) for key, value in wrong.items()})

    recheck = _ex_recheck("\n\n".join(evidence + distractors), claimant, form, choices)
    decisive = [
        _money(meal_cap),
        _money(room_cap),
        city,
        _money(lodging["rate"]),
        _money(lodging["tax"]),
    ]
    decisive += [_money(item["cents"]) for item in items]
    if form in ("noul", "approver"):
        decisive.append(_money(t1))
    facts = {
        "claimant": claimant,
        "form": form,
        "city": city,
        "high_cost": high_list,
        **flags,
        "meal_cap": meal_cap,
        "std_cap": std_cap,
        "high_cap": high_cap,
        "nights": nights,
        "rate": lodging["rate"],
        "tax": lodging["tax"],
        "items": [
            [item["cat"], item["day"].isoformat(), item["cents"]] for item in items
        ],
        "total": total,
        "wrong_total": wrong["total"],
        "threshold_1": t1,
        "threshold_2": t2,
        "exact": exact,
        "mechanism_variant": variant,
        "options": choice_totals,
    }
    return F1World(
        kind="expense_total",
        base=base,
        subject=f"{claimant}'s claim",
        question=pick(rng, _EX_Q[form]).format(**ident),
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=_filler(rng, _EX_TOPICS),
        right=Claim(gold, right_c, right_r, "correct"),
        wrong=Claim(wrong_answer, wrong_c, wrong_r, mech),
        decisive=tuple(decisive),
        facts=facts,
        variant=f"{form}-{'listed' if flags['listed'] else 'std'}",
        roles=_EX_ROLES,
    )


# ---------------------------------------------------------------- applicant_screen

_AP_POSTS = (
    (
        "warehouse operations lead",
        "warehouse operations",
        ("logistics", "supply chain management"),
        "Forklift Instructor",
        "SAP EWM",
    ),
    (
        "site safety adviser",
        "health and safety",
        ("occupational safety", "environmental science"),
        "Site Safety Supervisor",
        "Power BI",
    ),
    (
        "payroll specialist",
        "payroll",
        ("accounting", "finance"),
        "Certified Payroll Practitioner",
        "Workday",
    ),
    (
        "laboratory technician",
        "laboratory work",
        ("biomedical science", "chemistry"),
        "Laboratory Safety Practitioner",
        "LabWare",
    ),
    (
        "network engineer",
        "network engineering",
        ("computer science", "electrical engineering"),
        "Network Security Associate",
        "Ansible",
    ),
    (
        "food production supervisor",
        "food production",
        ("food science", "nutrition"),
        "Advanced Food Hygiene",
        "Tableau",
    ),
)
_AP_OTHER_DEGREES = (
    "history",
    "music",
    "philosophy",
    "geography",
    "graphic design",
    "marketing",
    "sociology",
)
_AP_LANGS = (
    "English",
    "Spanish",
    "German",
    "French",
    "Portuguese",
    "Dutch",
    "Polish",
    "Italian",
    "Turkish",
)
_AP_ORGS = (
    "Brightwater Foods",
    "Northfield Council",
    "Keystone Labs",
    "Halden Logistics",
    "Orchard Health",
    "Pioneer Networks",
)
_AP_ROLES = (
    "recruiter",
    "hiring manager",
    "HR business partner",
    "talent acquisition coordinator",
    "panel member",
    "resourcing officer",
)
_AP_LEVELS = (
    "Fails two or more requirements",
    "Fails exactly one requirement",
    "Meets every requirement",
    "Meets every requirement and has the preferred qualification",
)
_AP_LEVEL_WORDS = (
    "fails two or more of the requirements",
    "fails exactly one requirement",
    "meets every requirement",
    "meets every requirement and also has the preferred qualification",
)
_AP_WEIGHTS = {
    "noul": {
        "overlooked_condition": 1.0,
        "entity_swap": 0.6,
        "boundary_misread": 1.6,
        "stale_value": 2.6,
    },
    "score": {
        "overlooked_condition": 1.0,
        "entity_swap": 0.6,
        "boundary_misread": 1.6,
        "stale_value": 2.6,
    },
    "choice": {
        "overlooked_condition": 1.0,
        "entity_swap": 0.8,
        "boundary_misread": 1.5,
        "stale_value": 2.4,
    },
}
_AP_Q = {
    "noul": (
        "Does {name} meet all of the requirements for the {job} vacancy?",
        "Going by the stated requirements, does {name} qualify for the {job} post?",
        "Is {name} eligible for the {job} role on every requirement listed?",
    ),
    "score": (
        "What screening outcome should {name} receive for the {job} vacancy?",
        "How does {name} screen against the requirements for the {job} post?",
        "Rate {name}'s application against the {job} requirements.",
    ),
    "choice": (
        "Which applicant meets every requirement for the {job} vacancy?",
        "Exactly one of these applicants meets all the requirements for the {job} post. Who is it?",
        "Which of the applicants should be shortlisted as meeting every {job} requirement?",
    ),
}
_AP_NOUL_CONC = (
    (
        "{name} meets all of the requirements",
        "{name} does not meet all of the requirements",
    ),
    (
        "{given} qualifies for the {job} role",
        "{given} does not qualify for the {job} role",
    ),
    (
        "{given}'s application clears every requirement",
        "{given}'s application falls short of the requirements",
    ),
)
_AP_SCORE_CONC = ("{name} {level}", "on the screening, {given} {level}")
_AP_CHOICE_CONC = (
    "{answer} is the applicant who meets every requirement",
    "the only applicant who meets all the requirements is {answer}",
    "{answer} should be shortlisted as the one applicant meeting every requirement",
)
_AP_RAT = (
    "{given} has {v1} and {v2}",
    "the file shows {v1} and {v2}",
    "{given}'s file lists {v1} and {v2}",
)
_AP_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Interview process",
        (
            "Shortlisted applicants are invited to a structured interview with a panel of three.",
            "Every applicant is asked the same core questions so that answers can be compared fairly.",
            "Panel members score answers independently before discussing them.",
            "Interviews can take place in person or by video, at the applicant's choice.",
            "Applicants receive the interview format at least five working days in advance.",
            "Notes from the interview are kept with the application record.",
        ),
    ),
    (
        "Equal opportunities",
        (
            "We welcome applications from people of every background.",
            "Adjustments to the recruitment process are available on request.",
            "Monitoring information is collected separately and never seen by the panel.",
            "Decisions are based only on the published requirements and the evidence provided.",
            "Applicants who meet the requirements and declare a disability are offered an interview.",
            "Feedback on the process is reviewed by the people team every quarter.",
        ),
    ),
    (
        "References",
        (
            "References are requested only after a conditional offer has been made.",
            "Two references are needed, one of them from the most recent employer.",
            "Referees are asked to confirm dates of employment and the role held.",
            "An offer becomes final once satisfactory references have been received.",
            "Applicants are told before any referee is contacted.",
            "Reference requests are sent by the people team, not by the hiring manager.",
        ),
    ),
    (
        "Pay and benefits",
        (
            "The salary range is published with each vacancy and is not negotiated at screening.",
            "Benefits include a pension scheme, paid leave and a learning budget.",
            "Flexible working patterns are considered for every role.",
            "New starters join the pension scheme automatically and can opt out.",
            "The learning budget can be used for courses, books or conference places.",
            "Pay is reviewed each year in line with the published framework.",
        ),
    ),
    (
        "Hybrid working",
        (
            "Most roles combine office days with remote days agreed with the team.",
            "Some operational roles are based on site because of the equipment involved.",
            "Equipment for home working is provided after the probation period starts.",
            "Teams agree a regular day on which everyone meets in person.",
            "Hybrid arrangements are reviewed after six months.",
            "Travel between sites during the working day is paid by the employer.",
        ),
    ),
    (
        "Application data",
        (
            "Application records are kept for twelve months after the vacancy closes.",
            "Applicants can ask for their records to be deleted earlier.",
            "Only the people team and the panel can see application files.",
            "Application files are not used for any purpose other than this vacancy.",
            "Unsuccessful applicants may be asked whether they wish to hear about future vacancies.",
            "Copies of certificates are checked against the originals at the offer stage.",
        ),
    ),
    (
        "Onboarding",
        (
            "New starters spend their first week meeting the team and learning the main systems.",
            "Each new starter is paired with a buddy from another team.",
            "A short induction plan is agreed with the line manager on the first day.",
            "Mandatory training is completed during the first month.",
            "Probation reviews take place after three and six months.",
            "Questions about the first weeks can go to the buddy or the people team.",
        ),
    ),
    (
        "Screening practice",
        (
            "Screeners compare each file with the published requirements, one requirement at a time.",
            "Only evidence in the file is considered; assumptions are not recorded as facts.",
            "Where a document is unclear, the screener notes the question rather than guessing.",
            "Preferred qualifications never replace a stated requirement.",
            "A second screener checks a sample of files for consistency.",
            "Screening notes are written so that the applicant could read them.",
        ),
    ),
)
_AP_REQ_TEXT = {
    "exp": "{bound} {years} years of experience in {field}",
    "degree": "a degree in {a} or {b}, or at least {alt} years of experience in {field} in place of a degree",
    "cert": "{a_cert} {cert} certificate that is valid on the closing date",
    "lang": "professional fluency in {lang}",
    "start": "availability to start by {start}",
}


def _ap_span(months: int) -> str:
    years, rest = divmod(months, 12)
    head = f"{years} year{'s' if years != 1 else ''}"
    return head if rest == 0 else f"{head} and {rest} month{'s' if rest != 1 else ''}"


def _ap_ok(
    post: dict[str, Any], app: dict[str, Any], req: str, slip: str | None = None
) -> bool:
    if req == "exp":
        months = app["exp"]
        if slip == "round_up":
            months = round(months / 12) * 12
        strict = (
            True
            if slip == "exclusive"
            else False if slip == "inclusive" else post["strict"]
        )
        need = 12 * post["years"]
        return months > need if strict else months >= need
    if req == "degree":
        if app["degree"] in post["degrees"]:
            return True
        return slip != "alt_overlooked" and app["exp"] >= 12 * post["alt_years"]
    if req == "cert":
        until = app["cert_until"]
        if app["update"] and slip != "stale":
            kind, _, day = app["update"]
            until = day if kind == "renewed" else day - timedelta(days=1)
        return until >= post["close"]
    if req == "lang":
        return post["lang"] in app["fluent"]
    if slip == "exclusive":
        return app["start"] < post["start_by"]
    return app["start"] <= post["start_by"]


def _ap_status(
    post: dict[str, Any],
    app: dict[str, Any],
    slip: str | None = None,
    focus: str | None = None,
    source: dict[str, Any] | None = None,
) -> tuple[frozenset[str], bool]:
    fails = set()
    for req in post["reqs"]:
        if req == focus and slip == "ignore":
            continue
        holder = source if (req == focus and slip == "swap") else app
        if not _ap_ok(post, holder, req, slip if req == focus else None):
            fails.add(req)
    tool = app["tool"]
    if focus == "tool":
        tool = source["tool"] if slip == "swap" else False if slip == "ignore" else tool
    return frozenset(fails), tool


def _ap_level(status: tuple[frozenset[str], bool]) -> int:
    fails, tool = status
    return 0 if len(fails) >= 2 else 1 if fails else 3 if tool else 2


def _ap_minus(req: str, feats: dict[str, Any]) -> list[tuple[str, str]]:
    """Slips that make a failing requirement look met."""
    out = [("overlooked_condition", "ignore"), ("entity_swap", "swap")]
    if req == "exp" and not feats["alt"]:
        out.append(("boundary_misread", "inclusive" if feats["strict"] else "round_up"))
    if req == "cert" and feats["upd_subject"] and feats["upd_kind"] == "lapsed":
        out.append(("stale_value", "stale"))
    return out


def _ap_plus(req: str, feats: dict[str, Any]) -> list[tuple[str, str]]:
    """Slips that make a met requirement look failed."""
    out = [("entity_swap", "swap")]
    if req == "degree" and feats["alt"]:
        out.append(("overlooked_condition", "alt_overlooked"))
    if (req == "exp" and not feats["strict"] and not feats["alt"]) or req == "start":
        out.append(("boundary_misread", "exclusive"))
    if req == "cert" and feats["upd_subject"] and feats["upd_kind"] == "renewed":
        out.append(("stale_value", "stale"))
    return out


def _ap_candidates(
    rng: random.Random,
    reqs: Sequence[str],
    feats: dict[str, Any],
    base: str,
    target: int,
) -> list[tuple[str, tuple[str, str, frozenset[str], bool]]]:
    """(mechanism, (slip, focus, subject's failing set, subject's tool flag)) for the target."""
    allowed = [
        req
        for req in reqs
        if not (
            req == "cert" and feats["upd_subject"] and feats["upd_kind"] == "renewed"
        )
        and not (feats["alt"] and req in ("exp", "degree"))
    ]
    tool = rng.random() < 0.5
    out = []
    if base == "noul" and target == 0 or base == "score" and target == 1:
        for req in allowed:
            out += [
                (mech, (slip, req, frozenset({req}), tool))
                for mech, slip in _ap_minus(req, feats)
            ]
    if base == "noul" and target == 1 or base == "score" and target in (2, 3):
        tool = target == 3 if base == "score" else tool
        for req in reqs:
            out += [
                (mech, (slip, req, frozenset(), tool))
                for mech, slip in _ap_plus(req, feats)
            ]
    if base == "score" and target == 1:
        for req in allowed:
            others = [other for other in reqs if other != req]
            focus = rng.choice(others)
            out += [
                (mech, (slip, focus, frozenset({req}), tool))
                for mech, slip in _ap_plus(focus, feats)
            ]
    if base == "score" and target == 0 and len(allowed) >= 2:
        for req in allowed:
            second = rng.choice([other for other in allowed if other != req])
            out += [
                (mech, (slip, req, frozenset({req, second}), tool))
                for mech, slip in _ap_minus(req, feats)
            ]
    if base == "score" and target == 2:
        out.append(("entity_swap", ("swap", "tool", frozenset(), False)))
    if base == "score" and target == 3:
        out.append(("overlooked_condition", ("ignore", "tool", frozenset(), True)))
    return out


def _ap_applicant(
    rng: random.Random,
    post: dict[str, Any],
    name: str,
    code: str,
    fails: frozenset[str],
    tool: bool,
    flavors: dict[str, str],
) -> dict[str, Any]:
    """Build a file whose requirement statuses are exactly ``fails`` (with the given flavours)."""
    years, alt, close = post["years"], post["alt_years"], post["close"]
    need = 12 * years
    if flavors.get("degree") == "alt":
        exp = 12 * alt + rng.randint(0, 30)
    elif "exp" in fails:
        if flavors.get("exp") == "near":
            exp = need if post["strict"] else need - rng.randint(1, 3)
        else:
            exp = max(12, need - rng.randint(14, 40))
    elif flavors.get("exp") == "exact" and not post["strict"]:
        exp = need
    else:
        exp = need + rng.randint(3, 60)
    if "degree" in fails:
        exp = min(exp, 12 * alt - rng.randint(1, 20))
        degree = None if rng.random() < 0.5 else pick(rng, _AP_OTHER_DEGREES)
    elif flavors.get("degree") == "alt":
        degree = None if rng.random() < 0.5 else pick(rng, _AP_OTHER_DEGREES)
    else:
        degree = pick(rng, post["degrees"])
    update = None
    cert = flavors.get("cert")
    if "cert" in fails:
        if cert == "lapse":
            until = close + timedelta(days=rng.randint(90, 700))
            day = close - timedelta(days=rng.randint(3, 120))
            update = ("lapsed", day - timedelta(days=rng.randint(0, 20)), day)
        else:
            until = close - timedelta(days=rng.randint(5, 400))
    else:
        if cert == "renew":
            until = close - timedelta(days=rng.randint(5, 200))
            update = (
                "renewed",
                close - timedelta(days=rng.randint(1, 60)),
                close + timedelta(days=rng.randint(200, 1100)),
            )
        elif cert == "renew_nd":
            until = close + timedelta(days=rng.randint(20, 300))
            update = (
                "renewed",
                close - timedelta(days=rng.randint(1, 60)),
                until + timedelta(days=rng.randint(300, 900)),
            )
        elif cert == "lapse_nd":
            until = close + timedelta(days=rng.randint(200, 700))
            day = close + timedelta(days=rng.randint(5, 150))
            update = ("lapsed", day - timedelta(days=rng.randint(0, 20)), day)
        else:
            until = close + timedelta(days=rng.randint(20, 700))
    langs = [lang for lang in _AP_LANGS if lang != post["lang"]]
    rng.shuffle(langs)
    if "lang" in fails:
        fluent, basic = [langs[0]] if rng.random() < 0.5 else langs[:2], [
            post["lang"] or langs[2]
        ]
        if len(fluent) == 2:
            basic = []
    else:
        fluent, basic = [post["lang"] or langs[2], langs[0]], []
        if rng.random() < 0.4:
            fluent, basic = fluent[:1], [langs[1]]
    start_by = post["start_by"] or close + timedelta(days=30)
    if "start" in fails:
        start = start_by + timedelta(days=rng.randint(2, 30))
    elif flavors.get("start") == "exact":
        start = start_by
    else:
        start = start_by - timedelta(days=rng.randint(1, 25))
    return {
        "name": name,
        "code": code,
        "exp": exp,
        "degree": degree,
        "cert_until": until,
        "update": update,
        "fluent": fluent,
        "basic": basic,
        "start": start,
        "tool": tool,
        "tool_said": tool or rng.random() < 0.5,
    }


def _ap_random(
    rng: random.Random,
    post: dict[str, Any],
    name: str,
    code: str,
    *,
    min_fails: int = 0,
    need_fail: str | None = None,
    need_pass: str | None = None,
    update: str | None = None,
) -> dict[str, Any]:
    reqs = list(post["reqs"])
    for _ in range(40):
        fails = {req for req in reqs if rng.random() < 0.35}
        if need_fail:
            fails.add(need_fail)
        if need_pass:
            fails.discard(need_pass)
        if update == "renewed":
            fails.discard("cert")
        if len(fails) >= min_fails:
            break
    else:
        raise GenerationError("applicant_screen: no distractor file")
    flavors = (
        {"exp": "near" if "exp" in fails else "exact"} if rng.random() < 0.3 else {}
    )
    if update == "renewed":
        flavors["cert"] = "renew" if rng.random() < 0.5 else "renew_nd"
    elif update == "lapsed":
        flavors["cert"] = "lapse" if "cert" in fails else "lapse_nd"
    app = _ap_applicant(
        rng, post, name, code, frozenset(fails), rng.random() < 0.4, flavors
    )
    if _ap_status(post, app)[0] != frozenset(fails):
        raise GenerationError("applicant_screen: distractor statuses drifted")
    return app


def _ap_phrase(
    post: dict[str, Any],
    app: dict[str, Any],
    req: str,
    style: str,
    slip: str | None = None,
) -> str:
    if req == "exp":
        months = round(app["exp"] / 12) * 12 if slip == "round_up" else app["exp"]
        return f"{_ap_span(months)} of experience in {post['field']}"
    if req == "degree":
        if app["degree"] in post["degrees"]:
            return f"a degree in {app['degree']}"
        return f"no degree in {post['degrees'][0]} or {post['degrees'][1]}"
    if req == "cert":
        name = post["cert"]
        if app["update"] and slip != "stale":
            kind, _, day = app["update"]
            if kind == "renewed":
                return (
                    f"{_an(name)} {name} certificate valid until {fmt_date(day, style)}"
                )
            return f"{_an(name)} {name} certificate that stopped being valid on {fmt_date(day, style)}"
        until = app["cert_until"]
        if until >= post["close"]:
            return (
                f"{_an(name)} {name} certificate valid until {fmt_date(until, style)}"
            )
        return (
            f"{_an(name)} {name} certificate that expired on {fmt_date(until, style)}"
        )
    if req == "lang":
        lang = post["lang"]
        return (
            f"fluent {lang}"
            if lang in app["fluent"]
            else f"only basic {lang}" if lang in app["basic"] else f"no {lang}"
        )
    if req == "start":
        return f"a start date of {fmt_date(app['start'], style)}"
    if app["tool"] and slip != "ignore":
        return f"experience with {post['tool']}"
    return f"no experience with {post['tool']}"


def _ap_file_text(
    rng: random.Random, post: dict[str, Any], app: dict[str, Any], style: str
) -> str:
    given = app["name"].split()[0]
    head = pick(
        rng,
        (
            "{name} (applicant {code}).",
            "Applicant {code}: {name}.",
            "File for {name}, applicant {code}.",
        ),
    )
    span = _ap_span(app["exp"])
    lines = [
        pick(
            rng,
            (
                f"{given} has {span} of experience in {post['field']}.",
                f"{given} has worked in {post['field']} for {span}.",
                f"{given}'s CV shows {span} of experience in {post['field']}.",
            ),
        )
    ]
    if app["degree"]:
        lines.append(
            pick(
                rng,
                (
                    f"{given} holds a degree in {app['degree']}.",
                    f"{given} graduated with a degree in {app['degree']}.",
                ),
            )
        )
    else:
        lines.append(
            pick(
                rng,
                (
                    f"{given} does not hold a degree.",
                    f"{given} left university before completing a degree.",
                ),
            )
        )
    until = fmt_date(app["cert_until"], style)
    lines.append(
        pick(
            rng,
            (
                f"{given}'s {post['cert']} certificate is valid until {until}.",
                f"{given} holds {_an(post['cert'])} {post['cert']} certificate, valid until {until}.",
            ),
        )
    )
    fluent, basic = app["fluent"], app["basic"]
    if basic:
        lines.append(
            pick(
                rng,
                (
                    f"{given} is fluent in {fluent[0]}, with basic {basic[0]}.",
                    f"{given} speaks {fluent[0]} fluently and has basic {basic[0]}.",
                ),
            )
        )
    elif len(fluent) == 2:
        lines.append(
            pick(
                rng,
                (
                    f"{given} speaks {fluent[0]} and {fluent[1]} fluently.",
                    f"{given} is fluent in {fluent[0]} and {fluent[1]}.",
                ),
            )
        )
    else:
        lines.append(f"{given} is fluent in {fluent[0]}.")
    start = fmt_date(app["start"], style)
    lines.append(
        pick(
            rng,
            (
                f"{given} can start on {start}.",
                f"{given} is available to start from {start}.",
            ),
        )
    )
    if app["tool"]:
        lines.append(f"{given} has used {post['tool']} in previous roles.")
    elif app["tool_said"]:
        lines.append(f"{given} has not worked with {post['tool']}.")
    body = lines[:1] + [line for line in lines[1:]]
    if rng.random() < 0.5:
        middle = body[1:-1]
        rng.shuffle(middle)
        body = body[:1] + middle + body[-1:]
    return " ".join([head.format(name=app["name"], code=app["code"])] + body)


def _ap_update_text(
    rng: random.Random, post: dict[str, Any], app: dict[str, Any], style: str
) -> str:
    kind, issued, day = app["update"]
    slots = {
        "issued": fmt_date(issued, style),
        "name": app["name"],
        "cert": post["cert"],
        "day": fmt_date(day, style),
    }
    if kind == "renewed":
        return pick(
            rng,
            (
                "Records update, {issued}: {name}'s {cert} certificate was renewed and is now valid until {day}.",
                "Note added on {issued}: the issuing body has renewed {name}'s {cert} certificate, which is now "
                "valid until {day}.",
            ),
        ).format(**slots)
    return pick(
        rng,
        (
            "Records update, {issued}: the issuing body withdrew {name}'s {cert} certificate with effect from "
            "{day} after a missed renewal.",
            "Note added on {issued}: {name}'s {cert} certificate lapsed and stopped being valid on {day}.",
        ),
    ).format(**slots)


def _ap_posting_text(rng: random.Random, post: dict[str, Any], style: str) -> str:
    slots = {
        "bound": "at least" if not post["strict"] else "more than",
        "years": _num(rng, post["years"]),
        "field": post["field"],
        "a": post["degrees"][0],
        "b": post["degrees"][1],
        "alt": _num(rng, post["alt_years"]),
        "cert": post["cert"],
        "a_cert": _an(post["cert"]),
        "lang": post["lang"],
        "start": fmt_date(post["start_by"], style) if post["start_by"] else "",
    }
    items = [_AP_REQ_TEXT[req].format(**slots) for req in post["reqs"]]
    close = fmt_date(post["close"], style)
    preferred = pick(
        rng,
        (
            f"Experience with {post['tool']} is preferred but not required.",
            f"Preferred but not required: experience with {post['tool']}.",
        ),
    )
    if rng.random() < 0.5:
        return "\n".join(
            [
                f"Vacancy: {post['job']} at {post['org']}, reference {post['ref']}. Applications close on {close}. "
                "Candidates must meet every one of these requirements:"
            ]
            + [f"- {item}" for item in items]
            + [preferred]
        )
    return (
        f"The {post['job']} post at {post['org']} (reference {post['ref']}) is open until {close}. {preferred} "
        f"To be considered, a candidate needs: {'; '.join(items[:-1])}; and {items[-1]}."
    )


def _ap_parse(text: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    post: dict[str, Any] = {}
    files: dict[str, dict[str, Any]] = {}
    updates: dict[str, tuple[str, date]] = {}
    for block in text.split("\n\n"):
        if "Candidates must meet" in block or "To be considered" in block:
            post["close"] = _to_date(
                re.search(
                    rf"(?:close on|open until) (?:[A-Z][a-z]+day )?({_DATE_SRC})", block
                ).group(1)
            )
            tool = re.search(r"[Ee]xperience with ([A-Z][\w]*(?: [A-Z][\w]*)*)", block)
            post["tool"] = tool.group(1) if tool else None
            for piece in re.split(r"\n|;", block):
                if "degree" in piece:
                    found = re.search(
                        r"degree in (.+?) or (.+?), or at least (\w+) years", piece
                    )
                    post["degree"] = (
                        {found.group(1), found.group(2)},
                        _read_num(found.group(3)),
                    )
                elif "years of experience" in piece:
                    found = re.search(
                        r"(at least|more than) (\w+) years of experience", piece
                    )
                    post["exp"] = (
                        found.group(1) == "more than",
                        _read_num(found.group(2)),
                    )
                elif "certificate" in piece:
                    post["cert"] = True
                elif "fluency in" in piece:
                    post["lang"] = re.search(r"fluency in (\w+)", piece).group(1)
                elif "start by" in piece:
                    post["start"] = _to_date(
                        re.search(
                            rf"start by (?:[A-Z][a-z]+day )?({_DATE_SRC})", piece
                        ).group(1)
                    )
        elif block.startswith(("Records update", "Note added")):
            who = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)'s", block).group(1)
            updates[who] = (
                "renewed" if "renewed" in block else "lapsed",
                _dates_in(block)[-1],
            )
        else:
            head = re.match(
                r"(?:File for |Applicant [A-Z]-\d+: )?([A-Z][a-z]+ [A-Z][a-z]+)", block
            )
            if not head or not re.search(r"[Aa]pplicant [A-Z]-\d{4}", block):
                continue
            record: dict[str, Any] = {"tool": " has used " in block}
            for sentence in _sentences(block):
                if "experience in" in sentence or "worked in" in sentence:
                    found = re.search(r"(\d+) years?(?: and (\d+) months?)?", sentence)
                    record["months"] = 12 * int(found.group(1)) + int(
                        found.group(2) or 0
                    )
                elif "degree" in sentence:
                    found = re.search(r"degree in ([a-z ]+)\.", sentence)
                    record["degree"] = found.group(1) if found else None
                elif "valid until" in sentence:
                    record["until"] = _dates_in(sentence)[0]
                elif "fluent" in sentence:
                    if "basic" in sentence:
                        record["fluent"] = set(
                            re.findall(r"(?:fluent in|speaks) (\w+)", sentence)
                        )
                    else:
                        record["fluent"] = set(
                            re.findall(r"\b([A-Z][a-z]+)\b", sentence)
                        ) - {head.group(1).split()[0]}
                elif "start" in sentence:
                    record["start"] = _dates_in(sentence)[0]
            files[head.group(1)] = record
    for who, update in updates.items():
        if who in files:
            files[who]["update"] = update
    return post, files


def _ap_recheck_screen(
    post: dict[str, Any], record: dict[str, Any]
) -> tuple[int, bool]:
    missed = 0
    if "exp" in post:
        strict, years = post["exp"]
        missed += not (
            record["months"] > 12 * years if strict else record["months"] >= 12 * years
        )
    if "degree" in post:
        names, alt = post["degree"]
        missed += not (record["degree"] in names or record["months"] >= 12 * alt)
    if "cert" in post:
        last_valid = record["until"]
        if "update" in record:
            kind, day = record["update"]
            last_valid = day if kind == "renewed" else day - timedelta(days=1)
        missed += last_valid < post["close"]
    if "lang" in post:
        missed += post["lang"] not in record["fluent"]
    if "start" in post:
        missed += record["start"] > post["start"]
    return missed, record["tool"]


def _ap_recheck(text: str, base: str, subject: str, choices: Sequence[str]) -> int:
    post, files = _ap_parse(text)
    if base == "choice":
        clear = [
            index
            for index, name in enumerate(choices)
            if name in files and _ap_recheck_screen(post, files[name])[0] == 0
        ]
        return clear[0] if len(clear) == 1 else -1
    if subject not in files:
        return -1
    missed, tool = _ap_recheck_screen(post, files[subject])
    if base == "noul":
        return int(missed == 0)
    return 0 if missed >= 2 else 1 if missed == 1 else 3 if tool else 2


def _build_applicant_screen(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    style = pick(rng, ("mdy", "dmy", "mdy_short", "dmy_weekday"))
    job, field, degrees, cert, tool = pick(rng, _AP_POSTS)
    optional = [req for req in ("lang", "start") if rng.random() < 0.5]
    close = date(2025, 2, 3) + timedelta(days=rng.randrange(560))
    years = rng.randint(2, 6)
    post = {
        "job": job,
        "field": field,
        "degrees": degrees,
        "cert": cert,
        "tool": tool,
        "org": pick(rng, _AP_ORGS),
        "ref": f"{pick(rng, ('HR', 'VAC', 'REC'))}-{rng.randint(100, 999)}",
        "years": years,
        "strict": rng.random() < 0.3,
        "alt_years": years + rng.randint(3, 5),
        "close": close,
        "lang": pick(rng, _AP_LANGS) if "lang" in optional else None,
        "start_by": (
            close + timedelta(days=rng.randint(20, 75)) if "start" in optional else None
        ),
    }
    reqs = ["exp", "degree", "cert"] + optional
    rng.shuffle(reqs)
    post["reqs"] = tuple(reqs)
    feats = {
        "strict": post["strict"],
        "upd_subject": rng.random() < 0.6,
        "upd_kind": "renewed" if rng.random() < 0.5 else "lapsed",
        "alt": rng.random() < 0.3,
    }
    count = rng.choice((3, 4)) if base == "choice" else rng.choice((2, 3))
    names = people(rng, count + (6 if length == "long" else 2))
    codes = [
        f"{pick(rng, 'ABCDEFGHJKLMN')}-{code}"
        for code in rng.sample(range(1000, 9999), len(names))
    ]

    def flavors_for(
        fails: frozenset[str], slip: str | None, focus: str | None, on_subject: bool
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        if feats["alt"] and not ({"exp", "degree"} & fails):
            out["degree"] = "alt"
        if focus == "exp" and slip in ("round_up", "inclusive"):
            out["exp"] = "near"
        elif focus in ("exp", "start") and slip == "exclusive":
            out[focus] = "exact"
        elif rng.random() < 0.25:
            out["exp"] = "near" if "exp" in fails else "exact"
        if on_subject:
            if feats["upd_kind"] == "renewed":
                decisive = (slip == "stale" and focus == "cert") or rng.random() < 0.5
                out["cert"] = "renew" if decisive else "renew_nd"
            else:
                out["cert"] = "lapse" if "cert" in fails else "lapse_nd"
        return out

    if base == "choice":
        # the wrong claim picks an applicant who fails exactly one requirement
        feats_w = dict(feats, alt=False)
        allowed = [
            req
            for req in reqs
            if not (
                req == "cert"
                and feats["upd_subject"]
                and feats["upd_kind"] == "renewed"
            )
        ]
        cands = [
            (mech, (slip, req))
            for req in allowed
            for mech, slip in _ap_minus(req, feats_w)
        ]
        mech, (slip, focus) = _mechanism_pick(rng, cands, _AP_WEIGHTS["choice"])[0]
        gold_app = _ap_applicant(
            rng,
            post,
            names[0],
            codes[0],
            frozenset(),
            rng.random() < 0.4,
            flavors_for(frozenset(), None, None, False),
        )
        wrong_fails = frozenset({focus})
        wrong_app = _ap_applicant(
            rng,
            post,
            names[1],
            codes[1],
            wrong_fails,
            rng.random() < 0.4,
            dict(
                flavors_for(wrong_fails, slip, focus, feats["upd_subject"]), degree=""
            ),
        )
        others = [
            _ap_random(
                rng,
                post,
                names[index],
                codes[index],
                min_fails=1,
                update=(
                    None if feats["upd_subject"] or index != 2 else feats["upd_kind"]
                ),
            )
            for index in range(2, count)
        ]
        source = next(
            app for app in [gold_app] + others if focus not in _ap_status(post, app)[0]
        )
        cast = [gold_app, wrong_app] + others
        if (
            _ap_status(post, gold_app)[0]
            or _ap_status(post, wrong_app)[0] != wrong_fails
        ):
            raise GenerationError("applicant_screen: choice statuses drifted")
        if _ap_status(post, wrong_app, slip, focus, source)[0]:
            raise GenerationError(
                "applicant_screen: choice slip does not clear the applicant"
            )
        order = list(range(count))
        rng.shuffle(order)
        options = [cast[index]["name"] for index in order]
        gold, wrong_answer = options.index(gold_app["name"]), options.index(
            wrong_app["name"]
        )
        subject_app, subject_label = gold_app, "the shortlist"
        extra = [
            _ap_random(rng, post, name, code, min_fails=1)
            for name, code in zip(names[count:], codes[count:])
        ]
    else:
        cands = _ap_candidates(rng, reqs, feats, base, target)
        ordered = _mechanism_pick(rng, cands, _AP_WEIGHTS[base])
        wanted_levels = _partners(rng, target) if base == "score" else [None]
        for mech, (slip, focus, fails, has_tool), wanted in (
            (m, p, w) for w in wanted_levels for m, p in ordered
        ):
            subject_app = _ap_applicant(
                rng,
                post,
                names[0],
                codes[0],
                fails,
                has_tool,
                flavors_for(fails, slip, focus, feats["upd_subject"]),
            )
            status = _ap_status(post, subject_app)
            if status[0] != fails:
                continue
            if slip == "swap":
                if focus == "tool":
                    source = _ap_random(rng, post, names[1], codes[1])
                    source["tool"] = source["tool_said"] = True
                elif focus in fails:
                    source = _ap_random(rng, post, names[1], codes[1], need_pass=focus)
                else:
                    source = _ap_random(rng, post, names[1], codes[1], need_fail=focus)
            else:
                source = _ap_random(
                    rng,
                    post,
                    names[1],
                    codes[1],
                    update=None if feats["upd_subject"] else feats["upd_kind"],
                )
            wrong_status = _ap_status(post, subject_app, slip, focus, source)
            if base == "noul":
                gold, wrong_answer = int(not status[0]), int(not wrong_status[0])
                ok = gold == target and wrong_answer != gold
            else:
                gold, wrong_answer = _ap_level(status), _ap_level(wrong_status)
                ok = gold == target and wrong_answer == wanted
            if ok:
                break
        else:
            raise GenerationError("applicant_screen: no mechanism")
        cast = [subject_app, source]
        if not feats["upd_subject"] and source["update"] is None:
            cast.append(
                _ap_random(rng, post, names[2], codes[2], update=feats["upd_kind"])
            )
        cast += [
            _ap_random(rng, post, names[index], codes[index])
            for index in range(len(cast), count)
        ]
        options = []
        subject_label = subject_app["name"]
        extra = [
            _ap_random(rng, post, name, code)
            for name, code in zip(names[len(cast) :], codes[len(cast) :])
        ]

    # render
    evidence = [_ap_posting_text(rng, post, style)]
    files = [_ap_file_text(rng, post, app, style) for app in cast]
    rng.shuffle(files)
    evidence += files
    for app in cast:
        if app["update"]:
            evidence.append(_ap_update_text(rng, post, app, style))
    distractors = [_ap_file_text(rng, post, app, style) for app in extra]

    # claims
    named = cast[0] if base != "choice" else None
    if base == "choice":
        claim_right, claim_wrong = gold_app, wrong_app
    else:
        claim_right = claim_wrong = subject_app
    passing = [
        req
        for req in reqs
        if req not in _ap_status(post, claim_wrong)[0] and req != focus
    ]
    failing = sorted(_ap_status(post, claim_right)[0] - {focus})
    if slip == "alt_overlooked":
        context = "exp"
    else:
        context = failing[0] if failing else (passing[0] if passing else None)
    spare = [req for req in passing if req != context]
    if context is None or (slip == "ignore" and focus != "tool" and not spare):
        raise GenerationError("applicant_screen: no rationale context")
    right_v1 = _ap_phrase(post, claim_right, focus, style)
    if slip == "ignore" and focus != "tool":
        wrong_v1 = _ap_phrase(post, claim_wrong, spare[0], style)
    elif slip == "swap":
        wrong_v1 = _ap_phrase(post, source, focus, style)
    elif slip in ("stale", "ignore"):
        # the superseded certificate record, or the preferred item read as absent
        wrong_v1 = _ap_phrase(post, claim_wrong, focus, style, slip)
    else:
        # boundary and alternative-route slips cite the true value and misread the rule
        wrong_v1 = _ap_phrase(post, claim_wrong, focus, style)
    rationale = pick(rng, _AP_RAT)
    right_r = rationale.format(
        given=claim_right["name"].split()[0],
        v1=right_v1,
        v2=_ap_phrase(post, claim_right, context, style),
    )
    wrong_r = rationale.format(
        given=claim_wrong["name"].split()[0],
        v1=wrong_v1,
        v2=_ap_phrase(post, claim_wrong, context, style),
    )
    ident = {"job": job}
    if base == "noul":
        pair = pick(rng, _AP_NOUL_CONC)
        slots = {"name": named["name"], "given": named["name"].split()[0], **ident}
        right_c, wrong_c = pair[1 - gold].format(**slots), pair[gold].format(**slots)
        choices: tuple[str, ...] = ()
    elif base == "score":
        template = pick(rng, _AP_SCORE_CONC)
        slots = {"name": named["name"], "given": named["name"].split()[0]}
        right_c = template.format(level=_AP_LEVEL_WORDS[gold], **slots)
        wrong_c = template.format(level=_AP_LEVEL_WORDS[wrong_answer], **slots)
        choices = _AP_LEVELS
    else:
        template = pick(rng, _AP_CHOICE_CONC)
        right_c, wrong_c = template.format(answer=gold_app["name"]), template.format(
            answer=wrong_app["name"]
        )
        choices = tuple(options)
    recheck = _ap_recheck(
        "\n\n".join(evidence + distractors),
        base,
        named["name"] if named else "",
        choices,
    )
    decisive = [fmt_date(close, style), _ap_span(claim_right["exp"])]
    if focus != "tool":
        decisive.append(_ap_span(claim_wrong["exp"]))
    facts = {
        "posting": {
            key: (value.isoformat() if isinstance(value, date) else value)
            for key, value in post.items()
            if key != "degrees"
        },
        "degrees": list(degrees),
        "features": feats,
        "slip": slip,
        "focus": focus,
        "applicants": [
            {
                "name": app["name"],
                "fails": sorted(_ap_status(post, app)[0]),
                "tool": app["tool"],
                "months": app["exp"],
                "update": app["update"][0] if app["update"] else None,
            }
            for app in cast
        ],
        "options": options,
    }
    return F1World(
        kind="applicant_screen",
        base=base,
        subject=subject_label,
        question=pick(rng, _AP_Q[base]).format(
            name=named["name"] if named else "", job=job
        ),
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=_filler(rng, _AP_TOPICS),
        right=Claim(gold, right_c, right_r, "correct"),
        wrong=Claim(wrong_answer, wrong_c, wrong_r, mech),
        decisive=tuple(decisive),
        facts=facts,
        variant=f"{base}-{len(reqs)}req",
        roles=_AP_ROLES,
    )


# ---------------------------------------------------------------- transit_connection

_TC_CITIES = (
    "Port Aldren",
    "Kessa",
    "Mirovale",
    "Tarnhelm",
    "Oskarby",
    "Valdoro",
    "Brenmoor",
    "Iskra Bay",
    "Lunetta",
    "Corvell",
    "Sable Point",
    "Drovana",
    "Halvik",
    "Marrow Cove",
)
_TC_AIRLINES = ("NV", "KT", "OR", "SQ", "AV", "LM", "TZ", "HR")
_TC_ROLES = (
    "travel agent",
    "trip coordinator",
    "operations controller",
    "ground staff supervisor",
    "booking assistant",
    "travel desk officer",
)
_TC_LEVELS = (
    "The connection cannot be made under the minimum connection rule",
    "The connection can be made with less than 30 minutes to spare beyond the minimum",
    "The connection can be made with 30 to 89 minutes to spare beyond the minimum",
    "The connection can be made with 90 minutes or more to spare beyond the minimum",
)
_TC_LEVEL_WORDS = (
    "not possible under the connection rule",
    "tight, with less than 30 minutes to spare",
    "comfortable, with 30 to 89 minutes to spare",
    "long, with 90 minutes or more to spare",
)
_TC_WEIGHTS = {
    "noul": {"arithmetic_slip": 0.6, "boundary_misread": 1.6, "stale_value": 2.4},
    "score": {"arithmetic_slip": 0.7, "boundary_misread": 2.5, "stale_value": 2.0},
    "choice": {"arithmetic_slip": 1.0, "boundary_misread": 1.0, "stale_value": 1.0},
}
_TC_Q = {
    "noul": (
        "Can {traveller} make the connection onto flight {code}?",
        "Under the connection rule, does {traveller}'s transfer onto flight {code} work?",
        "Will {traveller} be able to board flight {code} at {hub}?",
    ),
    "score": (
        "How does {traveller}'s connection onto flight {code} at {hub} rate?",
        "Rate the transfer time {traveller} has at {hub} for flight {code}.",
        "Which band describes {traveller}'s connection onto flight {code}?",
    ),
    "choice": (
        "Which onward departure is the earliest one {traveller} can make at {hub}?",
        "Under the connection rule, which is the first onward flight from {hub} that {traveller} can take?",
        "Which onward flight should {traveller} be booked on, as the earliest that works?",
    ),
}
_TC_NOUL_CONC = (
    (
        "{traveller} will make the {code} connection",
        "{traveller} will miss the {code} connection",
    ),
    (
        "the transfer at {hub} works for {given}",
        "the transfer at {hub} does not work for {given}",
    ),
    (
        "{given} can make the connection onto flight {code}",
        "{given} cannot make the connection onto flight {code}",
    ),
)
_TC_SCORE_CONC = (
    "{given}'s connection at {hub} is {level}",
    "the transfer onto flight {code} is {level}",
)
_TC_CHOICE_CONC = (
    "the earliest departure {given} can make is flight {answer}",
    "{given} should be booked onto flight {answer}",
    "flight {answer} is the first onward departure that works for {given}",
)
_TC_RAT = (
    "{first} reaches {hub} at {arrival} local time and the onward flight leaves at {departure}",
    "arrival in {hub} is at {arrival}, which leaves {gap} minutes before the {departure} departure",
    "the first leg lands at {arrival} {hub} time, giving a {gap}-minute gap to the {departure} departure",
)
_TC_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Baggage",
        (
            "Checked bags are labelled through to the final destination on a single booking.",
            "Travellers on separate tickets must collect their bags and check in again.",
            "Cabin bags must fit the sizer at the gate; oversized items are placed in the hold.",
            "Liquids in cabin bags follow the security limits of the departure airport.",
            "Delayed bags are delivered to the address given on the report form.",
            "Sports equipment should be booked in advance so that space can be reserved.",
        ),
    ),
    (
        "Check-in",
        (
            "Online check-in opens a day before departure for most routes.",
            "Boarding passes can be shown on a phone or printed at home.",
            "Bag drop desks close well before departure, so plan to arrive early.",
            "Travellers needing assistance should tell the airline when they book.",
            "Passport details must match the booking exactly.",
            "Check-in staff cannot change the connection rules at a transfer airport.",
        ),
    ),
    (
        "Lounges",
        (
            "Lounge access depends on the fare and on frequent flyer status.",
            "Most lounges offer quiet areas, showers and light meals.",
            "Children travelling with an eligible adult are usually admitted free.",
            "Lounges may limit entry at busy times.",
            "Day passes can sometimes be bought at the lounge entrance.",
            "Flight announcements are not always made in lounges, so keep an eye on the screens.",
        ),
    ),
    (
        "Assistance at airports",
        (
            "Assistance for reduced mobility is free and should be requested in advance.",
            "Staff can meet travellers at the aircraft door and accompany them to the next gate.",
            "Quiet routes through security are available at several airports.",
            "Assistance requests are passed on to each airport in the itinerary.",
            "Travellers can bring their own mobility aids, which travel free of charge.",
            "Please arrive at the assistance desk at the time given in the confirmation.",
        ),
    ),
    (
        "Transfers",
        (
            "Transfer passengers follow the signs for connecting flights after leaving the aircraft.",
            "Some airports require a second security check for connecting passengers.",
            "Gate information for the onward flight appears on the screens in the transfer area.",
            "Walking times between terminals can be long at large airports.",
            "Transfer desks can help if a connection is missed.",
            "Onward boarding passes can often be collected at the transfer desk.",
        ),
    ),
    (
        "Seat selection",
        (
            "Seats can be chosen during booking or later in the booking area.",
            "Extra legroom seats carry a fee on most fares.",
            "Families are seated together where possible, at no extra charge for young children.",
            "Seat maps can change if the aircraft type is swapped.",
            "Exit row seats are only assigned to travellers who meet the safety requirements.",
            "Seat requests are not guaranteed until the booking is confirmed.",
        ),
    ),
    (
        "Ground transport",
        (
            "Trains and buses link most hub airports with their city centres.",
            "Taxi ranks are signposted outside the arrivals hall.",
            "Car hire desks are usually in a separate building reached by shuttle.",
            "Night services run less often, so check the timetable before travelling late.",
            "Some airports charge a fee for picking up passengers at the kerb.",
            "Journey planners on the airport website show the current options.",
        ),
    ),
    (
        "Travel documents",
        (
            "Entry rules depend on nationality and on the purpose of the trip.",
            "Some transfer airports require a transit visa even without leaving the airport.",
            "Passports should be valid for the whole trip and often for several months beyond it.",
            "Travellers are responsible for holding the documents their destination requires.",
            "Airlines may refuse boarding if documents are missing.",
            "Official government pages give the most reliable document information.",
        ),
    ),
)


def _tc_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _tc_utc(offset: int) -> str:
    return f"UTC{'+' if offset >= 0 else '-'}{abs(offset)}"


def _tc_dur(minutes: int) -> str:
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _tc_ok(gap: int, mct: int, strict: bool) -> bool:
    return gap > mct if strict else gap >= mct


def _tc_level(gap: int, mct: int, strict: bool) -> int:
    if not _tc_ok(gap, mct, strict):
        return 0
    spare = gap - mct
    return 1 if spare < 30 else 2 if spare < 90 else 3


def _tc_code(rng: random.Random, taken: set[str]) -> str:
    for _ in range(50):
        code = f"{pick(rng, _TC_AIRLINES)} {rng.randint(100, 1999)}"
        if code not in taken:
            taken.add(code)
            return code
    raise _Redraw("transit_connection: code pool")


def _tc_layout(
    rng: random.Random,
    plan: tuple[int, int],
    mech: str,
    shift: int,
    mct: int,
    strict: bool,
    change_subject: bool,
    change_gap: int,
    landed: int,
) -> tuple[list[int], list[int], int, int] | None:
    """Sorted onward gaps whose first workable one has the planned rank, the gaps before the
    timetable change, and the option the slipped reading picks; None if no layout is found.
    """
    count, rank = plan
    need = mct + 5 if strict else mct
    if mech == "boundary_misread" and rank == (0 if strict else count - 1):
        return None
    failing, working = range(-60, need, 5), range(need, 200, 5)
    for _ in range(60):
        gaps = sorted(rng.sample(failing, rank)) + sorted(
            rng.sample(working, count - rank)
        )
        if mech == "boundary_misread":
            # a gap of exactly the minimum reads differently: under a strict rule the last one that
            # fails (read as inclusive, it works), under an inclusive rule the gold's (it fails)
            gaps[rank - 1 if strict else rank] = mct
        if any(b - a < 15 for a, b in zip(gaps, gaps[1:])):
            continue
        changed = rng.randrange(count)
        olds = list(gaps)
        if mech == "stale_value" or change_subject:
            olds[changed] = gaps[changed] - change_gap
        works = [_tc_ok(g, mct, strict) for g in gaps]
        gold_i = works.index(True)
        if mech == "arithmetic_slip":
            wrong_works = [_tc_ok(g - shift, mct, strict) for g in gaps]
        elif mech == "boundary_misread":
            wrong_works = [_tc_ok(g, mct, not strict) for g in gaps]
        else:
            wrong_works = [_tc_ok(g, mct, strict) for g in olds]
        if True not in wrong_works:
            continue
        read = olds if mech == "stale_value" else gaps
        wrong_i = min(range(count), key=lambda i: read[i] if wrong_works[i] else 10**6)
        if mech != "stale_value" and change_subject:
            old_works = [_tc_ok(g, mct, strict) for g in olds]
            keep = min(range(count), key=lambda i: olds[i] if old_works[i] else 10**6)
            if True in old_works and keep != gold_i:
                olds[changed] = gaps[changed]
        if (
            wrong_i != gold_i
            and landed + max(gaps + olds) <= 23 * 60 + 30
            and landed + min(gaps + olds) > 5 * 60
        ):
            return gaps, olds, gold_i, wrong_i
    return None


_TC_TIME_RX = re.compile(r"\b(\d{2}):(\d{2})\b")
_TC_CODE_RX = re.compile(r"\b([A-Z]{2} \d{3,4})\b")


def _tc_parse(text: str, traveller: str) -> dict[str, Any]:
    out: dict[str, Any] = {"zones": {}, "changes": {}, "options": []}
    for block in text.split("\n\n"):
        for city, sign, hours in re.findall(
            r"([A-Z][a-z]+(?: [A-Z][a-z]+)?),? (?:is on |on )?UTC([+-])(\d{1,2})", block
        ):
            out["zones"][city] = int(hours) * (1 if sign == "+" else -1)
        if block.startswith("Connection rules at"):
            out["hub"] = re.match(
                r"Connection rules at ([A-Z][a-z]+(?: [A-Z][a-z]+)?):", block
            ).group(1)
            bound, minutes = re.search(
                r"(at least|more than) (\d+) minutes", block
            ).groups()
            out["mct"] = (int(minutes), bound == "more than")
        elif block.startswith(("Timetable change", "Schedule update")):
            code = _TC_CODE_RX.search(block).group(1)
            times = [60 * int(h) + int(m) for h, m in _TC_TIME_RX.findall(block)]
            out["changes"][code] = times[0] if "instead of" in block else times[1]
        elif block.startswith("Onward departures"):
            out["options"] = [
                (code, 60 * int(h) + int(m))
                for code, h, m in re.findall(
                    r"([A-Z]{2} \d{3,4}) at (\d{2}):(\d{2})", block
                )
            ]
        elif traveller in block and ("Itinerary" in block or "booking" in block):
            sentences = _sentences(block)
            first = next(
                sentence for sentence in sentences if "flying time" in sentence
            )
            second = next(
                (s for s in sentences if s is not first and _TC_CODE_RX.search(s)), ""
            )
            hours, minutes = re.search(r"(\d+) h (\d+) min", first).groups()
            out["leg"] = {
                "origin": re.search(
                    r"(?:leaves|departs) ([A-Z][a-z]+(?: [A-Z][a-z]+)?) at", first
                ).group(1),
                "dep": sum(
                    60 ** (1 - i) * int(v)
                    for i, v in enumerate(_TC_TIME_RX.search(first).groups())
                ),
                "dur": 60 * int(hours) + int(minutes),
            }
            if _TC_TIME_RX.search(second):
                out["onward"] = (
                    _TC_CODE_RX.search(second).group(1),
                    sum(
                        60 ** (1 - i) * int(v)
                        for i, v in enumerate(_TC_TIME_RX.search(second).groups())
                    ),
                )
    return out


def _tc_recheck(text: str, traveller: str, base: str, choices: Sequence[str]) -> int:
    parsed = _tc_parse(text, traveller)
    leg, zones, hub = parsed["leg"], parsed["zones"], parsed["hub"]
    landed = leg["dep"] + leg["dur"] + 60 * (zones[hub] - zones[leg["origin"]])
    mct, strict = parsed["mct"]

    def works(departure: int) -> bool:
        gap = departure - landed
        return gap > mct or (not strict and gap == mct)

    if base == "choice":
        timed = sorted(
            (parsed["changes"].get(code, when), code)
            for code, when in parsed["options"]
        )
        first = next((code for when, code in timed if works(when)), None)
        hits = [
            index for index, option in enumerate(choices) if option == f"flight {first}"
        ]
        return hits[0] if len(hits) == 1 else -1
    code, when = parsed["onward"]
    departure = parsed["changes"].get(code, when)
    if base == "noul":
        return int(works(departure))
    if not works(departure):
        return 0
    spare = departure - landed - mct
    return 1 if spare < 30 else 2 if spare < 90 else 3


def _build_transit_connection(
    rng: random.Random, base: str, target: int | None, length: str
) -> F1World:
    plan = None
    if base == "choice":
        count = rng.choice((3, 4))
        plan = (count, rng.randrange(count))
    return _plan_loop(
        "transit_connection",
        lambda: _draw_transit_connection(rng, base, target, length, plan),
    )


def _draw_transit_connection(
    rng: random.Random,
    base: str,
    target: int | None,
    length: str,
    plan: tuple[int, int] | None,
) -> F1World:
    style = pick(rng, ("mdy", "dmy", "dmy_weekday", "mdy_short"))
    delta = rng.choice((-1, 1)) * rng.randint(1, 5)
    strict = rng.random() < 0.5
    change_subject = rng.random() < 0.6
    change_later = rng.random() < 0.5
    step = rng.choice((15, 20, 25, 30, 40, 45, 60))
    origin_zone = rng.randint(-6, 6)
    hub_zone = origin_zone + delta
    cities = rng.sample(_TC_CITIES, 5)
    origin, hub, dest = cities[:3]
    dest_zone = hub_zone + rng.choice((-2, -1, 1, 2, 3))
    mct = rng.choice((40, 45, 50, 60, 75, 90))
    need = mct + 5 if strict else mct
    travel = date(2025, 1, 6) + timedelta(days=rng.randrange(620))
    traveller = people(rng, 1)[0]
    taken: set[str] = set()
    first_code = _tc_code(rng, taken)
    for _ in range(20):
        dep = 5 * rng.randint(6 * 12, 13 * 12)
        dur = 5 * rng.randint(10, 66)
        landed = dep + dur + 60 * delta
        if 7 * 60 <= landed <= 18 * 60:
            break
    else:
        raise _Redraw("transit_connection: no leg timing")
    slips = {
        "zone_ignored": -60 * delta,
        "zone_sign": -120 * delta,
        "zone_doubled": 60 * delta,
    }
    change_gap = step * (1 if change_later else -1)
    onward_code = _tc_code(rng, taken)
    options: list[tuple[str, int]] = []
    old_time = None
    right_gap = wrong_gap = None
    mech = variant = None

    def gaps_for(level: int) -> list[int]:
        if level == 0:
            low = [mct] if strict else []
            return low + [mct - 5 * k for k in range(1, 13)]
        low, high = {1: (0, 25), 2: (30, 85), 3: (90, 180)}[level]
        return [
            mct + spare
            for spare in range(low, high + 1, 5)
            if not (strict and spare == 0)
        ]

    if base in ("noul", "score"):
        cands: list[tuple[str, Any]] = [
            ("arithmetic_slip", (name, shift)) for name, shift in slips.items()
        ]
        cands.append(("boundary_misread", ("inclusive" if strict else "exclusive", 0)))
        if change_subject:
            cands.append(("stale_value", ("old_departure", -change_gap)))
        found = False
        ordered = _mechanism_pick(rng, cands, _TC_WEIGHTS[base])
        wanted_levels = _partners(rng, target) if base == "score" else [None]
        for mech, (variant, shift), wanted in (
            (m, p, w) for w in wanted_levels for m, p in ordered
        ):
            pool = (
                [mct]
                if mech == "boundary_misread"
                else (
                    list(range(max(5, need - 150), need + 150, 5))
                    if base == "noul"
                    else gaps_for(target)
                )
            )
            rng.shuffle(pool)
            if mech == "arithmetic_slip" and not 0 < landed + shift < 24 * 60:
                continue
            for gap in pool:
                if mech == "arithmetic_slip":
                    wrong = gap - shift
                    wrong_ok = _tc_ok(wrong, mct, strict)
                    wrong_level = _tc_level(wrong, mct, strict)
                elif mech == "boundary_misread":
                    wrong = gap
                    wrong_ok = _tc_ok(gap, mct, not strict)
                    wrong_level = _tc_level(gap, mct, not strict)
                else:
                    wrong = gap + shift
                    wrong_ok = _tc_ok(wrong, mct, strict)
                    wrong_level = _tc_level(wrong, mct, strict)
                if base == "noul":
                    good = (
                        int(_tc_ok(gap, mct, strict)) == target
                        and int(wrong_ok) != target
                    )
                    good = good and gap > 0 and 0 < wrong < 360
                else:
                    good = (
                        _tc_level(gap, mct, strict) == target and wrong_level == wanted
                    )
                    good = good and gap > 0 and wrong > 0
                if good:
                    right_gap, wrong_gap = gap, wrong
                    found = True
                    break
            if found:
                break
        if not found:
            raise _Redraw("transit_connection: no mechanism")
        departure = landed + right_gap
        if change_subject:
            old_time = departure - change_gap
        if not all(
            0 < value <= 23 * 60 + 30 for value in (departure, old_time or departure)
        ):
            raise _Redraw("transit_connection: onward outside the day")
        gold = (
            int(_tc_ok(right_gap, mct, strict))
            if base == "noul"
            else _tc_level(right_gap, mct, strict)
        )
        if base == "noul":
            wrong_answer = 1 - gold
        elif mech == "boundary_misread":
            wrong_answer = _tc_level(right_gap, mct, not strict)
        else:
            wrong_answer = _tc_level(wrong_gap, mct, strict)
        wrong_arrival = landed + (slips[variant] if mech == "arithmetic_slip" else 0)
        wrong_departure = old_time if mech == "stale_value" else departure
        onward_departure = departure
    else:
        cands = [("arithmetic_slip", (name, shift)) for name, shift in slips.items()]
        cands += [
            ("boundary_misread", ("inclusive" if strict else "exclusive", 0)),
            ("stale_value", ("old_departure", 0)),
        ]
        layout = None
        for mech, (variant, shift) in _mechanism_pick(rng, cands, _TC_WEIGHTS[base]):
            if 0 < landed + shift < 24 * 60:
                layout = _tc_layout(
                    rng,
                    plan,
                    mech,
                    shift,
                    mct,
                    strict,
                    change_subject,
                    change_gap,
                    landed,
                )
            if layout is not None:
                break
        if layout is None:
            raise _Redraw("transit_connection: no choice layout for the planned rank")
        gaps, olds, gold_i, wrong_i = layout
        count = len(gaps)
        codes = [_tc_code(rng, taken) for _ in range(count)]
        options = [(code, landed + gap) for code, gap in zip(codes, gaps)]
        old_options = [(code, landed + gap) for code, gap in zip(codes, olds)]
        gold, wrong_answer = gold_i, wrong_i
        right_gap, wrong_gap = gaps[gold_i], (
            olds[wrong_i] if mech == "stale_value" else gaps[wrong_i]
        )
        wrong_arrival = landed + (slips[variant] if mech == "arithmetic_slip" else 0)
        wrong_departure = (
            old_options[wrong_i][1] if mech == "stale_value" else options[wrong_i][1]
        )
        onward_code, onward_departure = options[gold_i]
        changed_code, changed_old = next(
            (
                (code, old)
                for (code, new), (_, old) in zip(options, old_options)
                if new != old
            ),
            (None, None),
        )

    # render
    date_text = fmt_date(travel, style)
    zones = {origin: origin_zone, hub: hub_zone, dest: dest_zone}
    zone_text = pick(
        rng,
        (
            "Time zones on {day}: {a} is on {za}, {h} is on {zh} and {d} is on {zd}.",
            "Local times are given in each city's own zone: {a}, {za}; {h}, {zh}; {d}, {zd}.",
        ),
    ).format(
        day=date_text,
        a=origin,
        h=hub,
        d=dest,
        za=_tc_utc(origin_zone),
        zh=_tc_utc(hub_zone),
        zd=_tc_utc(dest_zone),
    )
    rule = (
        f"Connection rules at {hub}: a transfer is only accepted when there are more than {mct} minutes "
        "between the scheduled arrival and the onward departure."
        if strict
        else f"Connection rules at {hub}: a transfer needs at least {mct} minutes between the scheduled arrival "
        "and the onward departure."
    )
    first = pick(
        rng,
        (
            f"The first leg, flight {first_code}, leaves {origin} at {_tc_clock(dep)} local time, and the scheduled "
            f"flying time to {hub} is {_tc_dur(dur)}.",
            f"Flight {first_code} departs {origin} at {_tc_clock(dep)} local time with a scheduled flying time of "
            f"{_tc_dur(dur)} to {hub}.",
        ),
    )
    if base == "choice":
        onward = f"The booked onward flight to {dest} was cancelled, so {traveller.split()[0]} needs a new connection."
        board = (
            f"Onward departures from {hub} to {dest} on {date_text}: "
            + join_list(
                [f"flight {code} at {_tc_clock(when)}" for code, when in old_options]
            )
            + " (all local times)."
        )
        change_code = changed_code if changed_code else None
        change_old = changed_old
        change_new = dict(options).get(changed_code) if changed_code else None
    else:
        onward = pick(
            rng,
            (
                f"The booked connection is flight {onward_code} from {hub} to {dest}, departing at "
                f"{_tc_clock(old_time if old_time is not None else departure)} local time.",
                f"The onward flight {onward_code} to {dest} is scheduled to leave {hub} at "
                f"{_tc_clock(old_time if old_time is not None else departure)} local time.",
            ),
        )
        board = None
        change_code = onward_code if change_subject else None
        change_old, change_new = (
            (old_time, departure) if change_subject else (None, None)
        )
    itinerary = (
        pick(
            rng,
            (
                f"Itinerary for {traveller}, travelling on {date_text}. ",
                f"{traveller}'s booking for {date_text}. ",
            ),
        )
        + first
        + " "
        + onward
    )
    if change_code is None:
        change_code = _tc_code(rng, taken)
        base_time = 5 * rng.randint(9 * 12, 20 * 12)
        change_old, change_new = base_time, base_time + change_gap
    effective = fmt_date(travel - timedelta(days=rng.randint(3, 40)), style)
    notice = pick(
        rng,
        (
            f"Timetable change effective {effective}: flight {change_code} from {hub} now departs at "
            f"{_tc_clock(change_new)} instead of {_tc_clock(change_old)}.",
            f"Schedule update ({effective}): the departure of flight {change_code} from {hub} moves from "
            f"{_tc_clock(change_old)} to {_tc_clock(change_new)}.",
        ),
    )
    evidence = [itinerary, zone_text, rule]
    if board:
        evidence.append(board)
    evidence.insert(rng.randint(1, len(evidence)), notice)
    others = people(rng, 8 if length == "long" else 3, exclude=[traveller])
    distractors = []
    for other in others:
        code_a, code_b = _tc_code(rng, taken), _tc_code(rng, taken)
        odep = 5 * rng.randint(6 * 12, 16 * 12)
        city = pick(rng, [c for c in _TC_CITIES if c not in (hub,)])
        distractors.append(
            pick(
                rng,
                (
                    f"Itinerary for {other}, travelling on {fmt_date(travel + timedelta(days=rng.randint(-5, 5)), style)}. "
                    f"Flight {code_a} leaves {city} at {_tc_clock(odep)} local time with a scheduled flying time of "
                    f"{_tc_dur(5 * rng.randint(10, 60))} to {pick(rng, _TC_CITIES)}. The onward flight {code_b} is "
                    f"scheduled to leave at {_tc_clock(min(odep + 300, 23 * 60))} local time.",
                    f"Departures board extract for {pick(rng, _TC_CITIES)}: flight {code_a} at {_tc_clock(odep)} and flight "
                    f"{code_b} at {_tc_clock(min(odep + 95, 23 * 60 + 30))}.",
                ),
            )
        )

    # claims
    given = traveller.split()[0]
    slots = {"traveller": traveller, "given": given, "hub": hub, "code": onward_code}
    if base == "noul":
        pair = pick(rng, _TC_NOUL_CONC)
        right_c, wrong_c = pair[1 - gold].format(**slots), pair[gold].format(**slots)
        choices: tuple[str, ...] = ()
    elif base == "score":
        template = pick(rng, _TC_SCORE_CONC)
        right_c = template.format(level=_TC_LEVEL_WORDS[gold], **slots)
        wrong_c = template.format(level=_TC_LEVEL_WORDS[wrong_answer], **slots)
        choices = _TC_LEVELS
    else:
        template = pick(rng, _TC_CHOICE_CONC)
        right_c = template.format(answer=options[gold][0], **slots)
        wrong_c = template.format(answer=options[wrong_answer][0], **slots)
        choices = tuple(f"flight {code}" for code, _ in options)
    rationale = pick(rng, _TC_RAT)
    right_r = rationale.format(
        first=first_code,
        hub=hub,
        arrival=_tc_clock(landed),
        departure=_tc_clock(onward_departure),
        gap=right_gap,
    )
    wrong_r = rationale.format(
        first=first_code,
        hub=hub,
        arrival=_tc_clock(wrong_arrival),
        departure=_tc_clock(wrong_departure),
        gap=wrong_departure - wrong_arrival,
    )
    recheck = _tc_recheck("\n\n".join(evidence + distractors), traveller, base, choices)
    decisive = [
        _tc_clock(dep),
        _tc_dur(dur),
        _tc_utc(origin_zone),
        _tc_utc(hub_zone),
        f"{mct} minutes",
    ]
    facts = {
        "traveller": traveller,
        "origin": origin,
        "hub": hub,
        "destination": dest,
        "zones": zones,
        "delta_hours": delta,
        "strict": strict,
        "mct": mct,
        "leg_departure": dep,
        "leg_duration": dur,
        "arrival_hub": landed,
        "onward": onward_code,
        "right_gap": right_gap,
        "wrong_gap": wrong_gap,
        "change_on_subject": change_subject,
        "change_later": change_later,
        "change_minutes": step,
        "options": [[code, when] for code, when in options],
        "mechanism_variant": variant,
    }
    return F1World(
        kind="transit_connection",
        base=base,
        subject=f"{traveller}'s connection at {hub}",
        question=pick(rng, _TC_Q[base]).format(
            traveller=traveller, code=onward_code, hub=hub
        ),
        choices=choices,
        gold=gold,
        recheck=recheck,
        evidence=tuple(evidence),
        distractors=tuple(distractors),
        filler=_filler(rng, _TC_TOPICS),
        right=Claim(gold, right_c, right_r, "correct"),
        wrong=Claim(wrong_answer, wrong_c, wrong_r, mech),
        decisive=tuple(decisive),
        facts=facts,
        variant=f"{base}-{'east' if delta > 0 else 'west'}",
        roles=_TC_ROLES,
    )


# ---------------------------------------------------------------- registry

BUILDERS: dict[str, Callable[[random.Random, str, int | None, str], F1World]] = {
    "order_sla": _build_order_sla,
    "expense_total": _build_expense_total,
    "applicant_screen": _build_applicant_screen,
    "transit_connection": _build_transit_connection,
}
BASES: dict[str, tuple[str, ...]] = {
    kind: ("choice", "noul", "score") for kind in BUILDERS
}
SCORE_LEVELS: dict[str, int] = {kind: 4 for kind in BUILDERS}
