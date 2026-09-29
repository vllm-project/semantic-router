"""F2 ``hs1_policy_packet``: long policy packets (records/hs1-prereg-2026-09-29.md §1 F2).

A packet is one English policy document of 4,000-12,000 characters: title and
edition line, definitions, scope, base provisions, 1-4 amendments with
effective dates (some replaced or withdrawn by a later amendment, some not yet
in force), optional annexes keyed on one scope attribute, an optional
temporary measure with an in-force window, an optional "notwithstanding"
exception, an interpretation section stating one of three precedence schemes
and the date that governs, and distractor sections. Four layouts are used:
numbered sections, a handbook with headings, a consolidated text with a change
log, and a memo with attachments.

Each outcome is set by the highest-precedence applicable provision under the
packet's scheme; within the main text the latest amendment in force on the
governing date replaces earlier text. The oracle works on ``Provision``
objects; ``recheck`` is a separately written brute-force evaluator that reads
only the JSON facts (ISO dates, conditions resolved through the definitions
table, a timeline replay for amendments and a literal order list per scheme).

A group is two different cases on the same packet and interface. Case designs
are enumerated for the packet (governing-date epoch x annex scope x exception
state, and the requested quantity interval for Noul) and one is selected per
case against group targets (Noul answer, Score level, differing golds, and for
Score whether first_match is right) and a drawn pattern of which naive
heuristic (base_only, latest_amendment, first_match) is right, which keeps
each heuristic right on roughly 15-40% of rows. When no design meets the
targets the packet's values are redrawn on the same structure. A build passes
``index`` (the group's position within its domain and interface) to balance
the Noul answers exactly and to cycle the first Score level and a question
template shared by both cases. The two Choice cases always have the same
number of options.
"""

from __future__ import annotations

import dataclasses
import operator
import random
from datetime import date, timedelta
from typing import Any, Callable

from v2.data.hs1 import core
from v2.data.hs1.f2_domains import (
    DOMAINS,
    GENERIC_DEFS,
    GENERIC_DISTRACTORS,
    LAYOUTS,
    Domain,
    Levels,
    Quantity,
)

FAMILY = "hs1_policy_packet"
KINDS_TRAIN = (
    "travel_expense",
    "returns_refund",
    "leave_policy",
    "library_lending",
    "it_access",
    "grant_funding",
    "parking_permits",
    "warranty_service",
)
KINDS_OOD = ("tuition_refund", "venue_booking")
INTERFACE_SHARES = {"choice": 0.45, "noul": 0.35, "score": 0.20}
LENGTH_SHARES = {"long": 1.0}
INTERFACES = ("choice", "noul", "score")

PACKET_MIN = 4000
PACKET_MAX = 12000
STATE_MAX = 12000
ATTEMPTS = 50

SCHEMES: dict[str, tuple[str, ...]] = {
    "S1": ("exception", "temporary", "annex", "main"),
    "S2": ("annex", "exception", "temporary", "main"),
    "S3": ("temporary", "exception", "annex", "main"),
}
HAZARDS = (
    "amendment_in_force",
    "amendment_not_yet_in_force",
    "superseded_amendment",
    "exception_applies",
    "exception_condition_unmet",
    "annex_in_scope",
    "annex_out_of_scope",
    "temporary_measure_in_force",
    "temporary_measure_not_yet_in_force",
    "temporary_measure_expired",
    "cross_reference",
)
HEURISTICS = ("base_only", "latest_amendment", "first_match")
POSITIVE = {
    "exception": "exception_applies",
    "temporary": "temporary_measure_in_force",
    "annex": "annex_in_scope",
}
NEGATIVE = {
    "exception": "exception_condition_unmet",
    "temporary": "temporary_measure_expired",
    "annex": "annex_out_of_scope",
}


def supported(kind: str) -> tuple[str, ...]:
    if kind not in DOMAINS:
        raise KeyError(kind)
    return INTERFACES


# ---------------------------------------------------------------- engine model


@dataclasses.dataclass(frozen=True)
class Cond:
    """``attr`` compared with ``arg`` (min: >=, max: <=, is: ==); ``term`` names the definition used, if any."""

    attr: str
    kind: str
    arg: Any
    term: str = ""

    def holds(self, attrs: dict[str, Any]) -> bool:
        value = attrs[self.attr]
        if self.kind == "min":
            return value >= self.arg
        if self.kind == "max":
            return value <= self.arg
        return value == self.arg


@dataclasses.dataclass(frozen=True)
class Provision:
    """One provision. ``cls`` is main / exception / temporary / annex.

    Main-class amendments carry ``number``, ``op`` (set / replace / withdraw),
    ``start`` (effective date) and, for replace / withdraw, the ``target`` pid.
    A temporary measure is in force from ``start`` to ``end`` inclusive.
    """

    pid: str
    cls: str
    value: Any = None
    start: date | None = None
    end: date | None = None
    number: int = 0
    op: str = ""
    target: str = ""
    adopted: date | None = None
    scope: str = ""
    conds: tuple[Cond, ...] = ()

    @property
    def sets_value(self) -> bool:
        return self.value is not None


@dataclasses.dataclass
class Packet:
    domain: str
    interface: str
    scheme: str
    governs: str
    gov_via_term: bool
    layout: str
    provisions: tuple[Provision, ...]
    order: tuple[str, ...]
    base_start: date
    horizon: date

    def __post_init__(self) -> None:
        self.by_pid = {p.pid: p for p in self.provisions}
        self.base = self.by_pid["base"]
        self.amendments = tuple(
            sorted(
                (p for p in self.provisions if p.cls == "main" and p.pid != "base"),
                key=lambda p: p.number,
            )
        )
        self.specials = tuple(p for p in self.provisions if p.cls != "main")
        self.withdrawn = {
            p.target: p.start
            for p in self.amendments
            if p.op in ("replace", "withdraw")
        }

    @property
    def latest_setter(self) -> Provision:
        setters = [p for p in self.amendments if p.sets_value]
        return setters[-1]


@dataclasses.dataclass
class Case:
    attrs: dict[str, Any]
    scope: str
    dates: dict[str, date]
    q: Any = None

    def governing(self, packet: Packet) -> date:
        return self.dates[packet.governs]


@dataclasses.dataclass
class Resolution:
    decider: Provision
    main: Provision
    applicable: dict[str, bool]


def amendment_status(
    packet: Packet, amendment: Provision, when: date, main: Provision
) -> str:
    withdrawn = packet.withdrawn.get(amendment.pid)
    if withdrawn is not None and withdrawn <= when:
        return "withdrawn"
    if amendment.start > when:
        return "not_yet"
    return "current" if amendment is main else "replaced"


def main_text(packet: Packet, when: date) -> Provision:
    """The main text as amended on ``when``: the latest value-setting amendment in force, else the base."""
    live = [
        p
        for p in packet.amendments
        if p.sets_value
        and p.start <= when
        and not (p.pid in packet.withdrawn and packet.withdrawn[p.pid] <= when)
    ]
    if not live:
        return packet.base
    return max(live, key=lambda p: (p.start, p.number))


def special_applies(provision: Provision, case: Case, when: date) -> bool:
    if provision.cls == "exception":
        return all(cond.holds(case.attrs) for cond in provision.conds)
    if provision.cls == "temporary":
        return provision.start <= when <= provision.end
    if provision.cls == "annex":
        return case.scope == provision.scope
    raise ValueError(provision.cls)


def temporary_status(provision: Provision, when: date) -> str:
    if when < provision.start:
        return "not_started"
    return "in_force" if when <= provision.end else "expired"


def _top(packet: Packet, applicable: dict[str, bool], main: Provision) -> Provision:
    for cls in SCHEMES[packet.scheme]:
        if cls == "main":
            return main
        for provision in packet.specials:
            if provision.cls == cls and applicable[provision.pid]:
                return provision
    return main


def resolve(packet: Packet, case: Case, when: date | None = None) -> Resolution:
    """Oracle: the provision that sets the outcome for ``case`` under the packet's scheme."""
    when = case.governing(packet) if when is None else when
    main = main_text(packet, when)
    applicable = {p.pid: special_applies(p, case, when) for p in packet.specials}
    return Resolution(_top(packet, applicable, main), main, applicable)


# ---------------------------------------------------------------- heuristics


def first_match(packet: Packet, case: Case, when: date | None = None) -> Provision:
    """First provision in document order whose own stated conditions hold, ignoring precedence and withdrawals."""
    when = case.governing(packet) if when is None else when
    for pid in packet.order:
        provision = packet.by_pid[pid]
        if not provision.sets_value:
            continue
        if provision.cls == "main":
            if provision.start <= when:
                return provision
        elif special_applies(provision, case, when):
            return provision
    return packet.base


def heuristic_provisions(packet: Packet, case: Case) -> dict[str, Provision]:
    return {
        "base_only": packet.base,
        "latest_amendment": packet.latest_setter,
        "first_match": first_match(packet, case),
    }


def answer_fn(interface: str, q: Any) -> Callable[[Any], int]:
    if interface == "noul":
        return lambda value: int(q <= value)
    return lambda value: value


# ---------------------------------------------------------------- hazards


def negative_label(provision: Provision, when: date) -> str:
    if (
        provision.cls == "temporary"
        and temporary_status(provision, when) == "not_started"
    ):
        return "temporary_measure_not_yet_in_force"
    return NEGATIVE[provision.cls]


def failing_conds(provision: Provision, case: Case) -> tuple[Cond, ...]:
    return tuple(cond for cond in provision.conds if not cond.holds(case.attrs))


def analyse(
    packet: Packet,
    case: Case,
    answer: Callable[[Any], int],
    res: Resolution | None = None,
) -> tuple[tuple[str, ...], str] | None:
    """Consequential hazards of one case and the deciding hazard; None if the design is unusable.

    A hazard is recorded when misjudging that provision would change the
    answer: the decider itself; an applicable but outranked provision with a
    different answer; a non-applicable provision whose inclusion would change
    the decider's answer; and, when the main text decides, every amendment
    other than the one in force whose value would give a different answer.
    """
    when = case.governing(packet)
    res = resolve(packet, case) if res is None else res
    gold = answer(res.decider.value)
    labels: set[str] = set()
    tempting: list[Provision] = []
    decider, main = res.decider, res.main
    if decider.cls != "main":
        labels.add(POSITIVE[decider.cls])
    elif decider is not packet.base:
        labels.add("amendment_in_force")
    for provision in packet.specials:
        if provision is decider:
            continue
        if res.applicable[provision.pid]:
            if answer(provision.value) != gold:
                labels.add(POSITIVE[provision.cls])
            continue
        widened = dict(res.applicable)
        if provision.cls == "annex":
            widened.update({p.pid: False for p in packet.specials if p.cls == "annex"})
        widened[provision.pid] = True
        if answer(_top(packet, widened, main).value) == gold:
            continue
        labels.add(negative_label(provision, when))
        tempting.append(provision)
    if decider.cls != "main":
        if main is not packet.base and answer(main.value) != gold:
            labels.add("amendment_in_force")
    else:
        for amendment in packet.amendments:
            if (
                not amendment.sets_value
                or amendment is main
                or answer(amendment.value) == gold
            ):
                continue
            status = amendment_status(packet, amendment, when, main)
            labels.add(
                "amendment_not_yet_in_force"
                if status == "not_yet"
                else "superseded_amendment"
            )
    exception = next((p for p in packet.specials if p.cls == "exception"), None)
    exception_via_term = False
    if exception is not None:
        applies = res.applicable[exception.pid]
        deciding = exception.conds if applies else failing_conds(exception, case)
        involved = (
            exception is decider
            or exception in tempting
            or (applies and answer(exception.value) != gold)
        )
        exception_via_term = involved and any(cond.term for cond in deciding)
    if exception_via_term:
        labels.add("cross_reference")
    if not labels:
        return None
    rank = {cls: index for index, cls in enumerate(SCHEMES[packet.scheme])}
    if decider.cls != "main":
        subtype = (
            "cross_reference"
            if decider.cls == "exception" and exception_via_term
            else POSITIVE[decider.cls]
        )
    else:
        latest = packet.latest_setter
        if (
            main is not packet.base
            and latest is not main
            and answer(latest.value) != gold
        ):
            status = amendment_status(packet, latest, when, main)
            subtype = (
                "amendment_not_yet_in_force"
                if status == "not_yet"
                else "superseded_amendment"
            )
        elif main is not packet.base and "amendment_in_force" in labels:
            subtype = "amendment_in_force"
        elif tempting:
            top = min(tempting, key=lambda p: rank[p.cls])
            subtype = (
                "cross_reference"
                if top.cls == "exception" and exception_via_term
                else negative_label(top, when)
            )
        elif latest is not main and answer(latest.value) != gold:
            status = amendment_status(packet, latest, when, main)
            subtype = (
                "amendment_not_yet_in_force"
                if status == "not_yet"
                else "superseded_amendment"
            )
        else:
            subtype = sorted(labels)[0]
    return tuple(sorted(labels)), subtype


# ---------------------------------------------------------------- independent re-check

RECHECK_ORDER = {
    "S1": ["exception", "temporary", "annex", "main"],
    "S2": ["annex", "exception", "temporary", "main"],
    "S3": ["temporary", "exception", "annex", "main"],
}
_COMPARE = {"min": operator.ge, "max": operator.le, "is": operator.eq}


def _recheck_main(provisions: list[dict[str, Any]], day: str) -> dict[str, Any]:
    """Replay the amendment history up to ``day`` (ISO) and return the main-text provision then in force."""
    base = [p for p in provisions if p["pid"] == "base"][0]
    events = []
    for p in provisions:
        if p["cls"] != "main" or p["pid"] == "base":
            continue
        if p["op"] in ("replace", "withdraw"):
            events.append((p["start"], 0, p["target"], "cancel"))
        if p["op"] in ("set", "replace"):
            events.append((p["start"], 1, p["pid"], "enter"))
    events.sort()
    active: dict[str, tuple[str, int]] = {}
    cancelled: set[str] = set()
    numbers = {p["pid"]: p["number"] for p in provisions}
    for when, _, pid, action in events:
        if when > day:
            break
        if action == "cancel":
            cancelled.add(pid)
            active.pop(pid, None)
        elif pid not in cancelled:
            active[pid] = (when, numbers[pid])
    if not active:
        return base
    chosen = sorted(active.items(), key=lambda item: item[1])[-1][0]
    return [p for p in provisions if p["pid"] == chosen][0]


def _recheck_holds(
    cond: dict[str, Any], definitions: dict[str, Any], attrs: dict[str, Any]
) -> bool:
    spec = definitions[cond["term"]] if "term" in cond else cond
    return bool(_COMPARE[spec["kind"]](attrs[spec["attr"]], spec["arg"]))


def recheck(facts: dict[str, Any]) -> int:
    """Brute-force label from the recorded facts alone (separately written from the oracle)."""
    packet, case = facts["packet"], facts["case"]
    day = case["dates"][packet["governs"]]
    provisions = packet["provisions"]
    applicable: dict[str, list[dict[str, Any]]] = {
        "exception": [],
        "temporary": [],
        "annex": [],
    }
    for p in provisions:
        if p["cls"] == "exception":
            if all(
                _recheck_holds(c, packet["definitions"], case["attrs"])
                for c in p["conds"]
            ):
                applicable["exception"].append(p)
        elif p["cls"] == "temporary":
            if p["start"] <= day <= p["end"]:
                applicable["temporary"].append(p)
        elif p["cls"] == "annex":
            if p["scope"] == case["scope"]:
                applicable["annex"].append(p)
    chosen = None
    for cls in RECHECK_ORDER[packet["scheme"]]:
        if cls == "main":
            chosen = _recheck_main(provisions, day)
            break
        if len(applicable[cls]) > 1:
            raise core.GenerationError(f"two applicable {cls} provisions")
        if applicable[cls]:
            chosen = applicable[cls][0]
            break
    value = chosen["value"]
    interface = facts["interface"]
    if interface == "noul":
        return 1 if case["q"] <= value else 0
    if interface == "score":
        return int(value)
    return facts["options"].index(value)


# ---------------------------------------------------------------- packet structure

N_AMENDMENTS = ((1, 0.22), (2, 0.33), (3, 0.27), (4, 0.18))
N_SPECIAL_CLASSES = ((1, 0.30), (2, 0.50), (3, 0.20))
MIN_GAP = 15


def _weighted(rng: random.Random, pairs: tuple[tuple[Any, float], ...]) -> Any:
    x = rng.random() * sum(weight for _, weight in pairs)
    for value, weight in pairs:
        x -= weight
        if x < 0:
            return value
    return pairs[-1][0]


def _days(rng: random.Random, low: int, high: int) -> timedelta:
    return timedelta(days=rng.randint(low, high))


def _draw_amendments(
    rng: random.Random, base_start: date, count: int
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    cancelled: set[str] = set()
    cursor = base_start + _days(rng, 40, 120)
    for number in range(1, count + 1):
        pid = f"A{number}"
        adopted = cursor
        live = [s for s in specs if s["op"] != "withdraw" and s["pid"] not in cancelled]
        op = (
            "set"
            if not live
            else _weighted(rng, (("set", 0.5), ("replace", 0.2), ("withdraw", 0.3)))
        )
        if op == "withdraw" and live[-1]["op"] != "set":
            op = "replace"
        target = live[-1]["pid"] if op != "set" else ""
        floor = max(
            [s["start"] for s in specs if s["op"] != "withdraw" and s["pid"] != target]
            + [s["start"] for s in specs if s["op"] == "withdraw"],
            default=None,
        )
        if op == "withdraw":
            start = adopted + _days(rng, 0, 20)
        else:
            long_lead = op == "set" and rng.random() < 0.3
            start = adopted + (_days(rng, 80, 150) if long_lead else _days(rng, 14, 45))
            if floor is not None and start <= floor + timedelta(days=20):
                start = floor + _days(rng, 21, 45)
        if target:
            cancelled.add(target)
        specs.append(
            {
                "pid": pid,
                "number": number,
                "op": op,
                "target": target,
                "adopted": adopted,
                "start": start,
            }
        )
        cursor = max(adopted, start - timedelta(days=150)) + _days(rng, 35, 110)
    return specs


def _key_dates(
    amendments: list[dict[str, Any]], window: tuple[date, date] | None
) -> list[date]:
    keys = {spec["start"] for spec in amendments}
    if window is not None:
        keys.update((window[0], window[1] + timedelta(days=1)))
    return sorted(keys)


def _spaced(keys: list[date]) -> bool:
    return all((b - a).days >= MIN_GAP for a, b in zip(keys, keys[1:]))


def _draw_window(
    rng: random.Random, base_start: date, amendments: list[dict[str, Any]]
) -> tuple[date, date] | None:
    last = max(spec["start"] for spec in amendments)
    span = (last - base_start).days + 90
    for _ in range(20):
        start = base_start + timedelta(days=rng.randint(20, max(21, span)))
        end = start + _days(rng, 44, 149)
        if _spaced(_key_dates(amendments, (start, end))):
            return start, end
    return None


def _balanced_levels(rng: random.Random, count: int, levels: int) -> list[int]:
    pool: list[int] = []
    while len(pool) < count:
        chunk = list(range(levels))
        rng.shuffle(chunk)
        pool.extend(chunk)
    pool = pool[:count]
    rng.shuffle(pool)
    return pool


def _assign_values(
    rng: random.Random,
    var: Quantity | Levels,
    interface: str,
    pids: list[str],
    latest: str,
    targets: dict[str, Any],
) -> dict[str, Any]:
    if interface != "score":
        if len(pids) > len(var.ladder):
            raise core.GenerationError("ladder too short for the provision set")
        return dict(zip(pids, rng.sample(list(var.ladder), len(pids))))
    levels = len(var.frags)
    need = sorted(set(targets["levels"]))
    for _ in range(60):
        values = dict(zip(pids, _balanced_levels(rng, len(pids), levels)))
        missing = [level for level in need if level not in values.values()]
        for level in missing:
            values[rng.choice(pids)] = level
        if (
            all(level in values.values() for level in need)
            and values["base"] != values[latest]
        ):
            return values
    raise core.GenerationError("could not place the target levels")


def _arrange(
    rng: random.Random,
    layout: str,
    has_x: bool,
    has_t: bool,
    annexes: list[str],
    amendments: list[str],
) -> list[tuple[str, list[str]]]:
    """Core sections in document order, each with the provisions it states."""
    embed = has_x and layout in ("numbered", "handbook") and rng.random() < 0.35
    main = ("main", ["base"] + (["X1"] if embed else []))
    specials: list[tuple[str, list[str]]] = []
    if has_x and not embed:
        specials.append(("exceptions", ["X1"]))
    if has_t:
        specials.append(("temporary", ["T1"]))
    if layout == "numbered":
        rng.shuffle(specials)
        middle = [("amendments", amendments)] + specials
        if rng.random() < 0.45:
            rng.shuffle(middle)
        early = rng.random() < 0.35
        secs = [("purpose", []), ("defs", []), ("scope", [])] + (
            [("interp", [])] if early else []
        )
        secs += [main] + middle + ([] if early else [("interp", [])])
        return secs + [(f"annex:{pid}", [pid]) for pid in annexes]
    if layout == "handbook":
        rng.shuffle(specials)
        news_first = rng.random() < 0.6
        early = rng.random() < 0.4
        secs = [("about", []), ("defs", [])] + ([("interp", [])] if early else [])
        secs += ([("amendments", amendments)] if news_first else []) + [main] + specials
        secs += ([] if news_first else [("amendments", amendments)]) + (
            [] if early else [("interp", [])]
        )
        return secs + ([("annexes", annexes)] if annexes else [])
    if layout == "consolidated":
        rng.shuffle(specials)
        secs = (
            [("preamble", []), ("purpose", []), ("defs", []), ("scope", []), main]
            + specials
            + [("interp", [])]
        )
        return (
            secs
            + [(f"annex:{pid}", [pid]) for pid in annexes]
            + [("changelog", amendments)]
        )
    latest, earlier = amendments[-1], amendments[:-1]
    t_first = has_t and rng.random() < 0.5
    secs = [
        ("memo", [latest]),
        ("att1", []),
        ("purpose", []),
        ("defs", []),
        ("scope", []),
        main,
    ]
    secs += [s for s in specials if s[0] == "exceptions"] + (
        [("temporary", ["T1"])] if t_first else []
    )
    secs.append(("interp", []))
    if earlier:
        secs += [("att2", []), ("amendments", earlier)]
    late_t = has_t and not t_first
    if annexes or late_t:
        secs.append(("att3", []))
        secs += ([("temporary", ["T1"])] if late_t else []) + [
            (f"annex:{pid}", [pid]) for pid in annexes
        ]
    return secs


def _draw_cond(rng: random.Random, spec: Any, via_term: bool) -> Cond:
    arg = rng.choice(spec.choices) if spec.kind == "is" else rng.choice(spec.thresholds)
    return Cond(spec.key, spec.kind, arg, spec.term if via_term else "")


def draw_packet(
    rng: random.Random, domain: Domain, interface: str, targets: dict[str, Any]
) -> tuple[Packet, dict]:
    """Draw the structured packet; returns the packet and its layout arrangement."""
    scheme = rng.choice(sorted(SCHEMES))
    governs = rng.choice(("a", "b"))
    gov_via_term = rng.random() < 0.35
    layout = rng.choice(LAYOUTS)
    base_start = date(2024, 1, 8) + timedelta(days=rng.randrange(540))
    amendments = _draw_amendments(rng, base_start, _weighted(rng, N_AMENDMENTS))
    setters = [spec["pid"] for spec in amendments if spec["op"] != "withdraw"]
    classes = set(rng.sample(["x", "t", "n"], _weighted(rng, N_SPECIAL_CLASSES)))
    n_annex = (2 if rng.random() < 0.3 else 1) if "n" in classes else 0
    floor = 4
    while 1 + len(setters) + len(classes - {"n"}) + n_annex < floor:
        missing = sorted({"x", "t", "n"} - classes)
        if missing:
            added = rng.choice(missing)
            classes.add(added)
            n_annex = max(n_annex, 1 if added == "n" else 0)
        elif n_annex < 2:
            n_annex += 1
        else:
            break
    has_x, has_t = "x" in classes, "t" in classes
    window = _draw_window(rng, base_start, amendments) if has_t else None
    has_t = window is not None
    keys = _key_dates(amendments, window)
    if not _spaced(keys):
        raise core.GenerationError("key dates too close")
    var = getattr(domain, interface)
    pids = ["base"] + setters + (["X1"] if has_x else []) + (["T1"] if has_t else [])
    pids += [f"N{i + 1}" for i in range(n_annex)]
    if len(pids) < floor or len(pids) == 1 + len(setters):
        raise core.GenerationError("too few value-setting provisions")
    values = _assign_values(rng, var, interface, pids, setters[-1], targets)
    provisions = [Provision("base", "main", values["base"], start=base_start)]
    for spec in amendments:
        provisions.append(
            Provision(
                spec["pid"],
                "main",
                values.get(spec["pid"]),
                start=spec["start"],
                number=spec["number"],
                op=spec["op"],
                target=spec["target"],
                adopted=spec["adopted"],
            )
        )
    if has_x:
        specs = rng.sample(list(domain.conditions), 1 if rng.random() < 0.55 else 2)
        conds = tuple(_draw_cond(rng, spec, rng.random() < 0.45) for spec in specs)
        provisions.append(Provision("X1", "exception", values["X1"], conds=conds))
    if has_t:
        provisions.append(
            Provision("T1", "temporary", values["T1"], start=window[0], end=window[1])
        )
    for index, scope in enumerate(rng.sample(list(domain.scope.values), n_annex)):
        provisions.append(
            Provision(f"N{index + 1}", "annex", values[f"N{index + 1}"], scope=scope)
        )
    annex_pids = [f"N{i + 1}" for i in range(n_annex)]
    arrangement = _arrange(
        rng, layout, has_x, has_t, annex_pids, [spec["pid"] for spec in amendments]
    )
    order = tuple(pid for _, pids_ in arrangement for pid in pids_ if pid != "base")
    horizon = keys[-1] + _days(rng, 100, 200)
    packet = Packet(
        domain.key,
        interface,
        scheme,
        governs,
        gov_via_term,
        layout,
        tuple(provisions),
        order,
        base_start,
        horizon,
    )
    return packet, {"arrangement": arrangement, "keys": keys}


def redraw_values(
    rng: random.Random, packet: Packet, domain: Domain, targets: dict[str, Any]
) -> Packet:
    """The same packet structure with a fresh draw of every provision's value."""
    pids = [p.pid for p in packet.provisions if p.sets_value]
    var = getattr(domain, packet.interface)
    values = _assign_values(
        rng, var, packet.interface, pids, packet.latest_setter.pid, targets
    )
    provisions = tuple(
        dataclasses.replace(p, value=values[p.pid]) if p.sets_value else p
        for p in packet.provisions
    )
    return dataclasses.replace(packet, provisions=provisions)


# ---------------------------------------------------------------- case designs


@dataclasses.dataclass
class Design:
    epoch: int
    scope: str
    exc: str
    interval: int
    gold: Any
    pattern: tuple[int, int, int]
    labels: tuple[str, ...]
    subtype: str
    decider: str
    heur: dict[str, Any]

    @property
    def key(self) -> tuple[int, str, str, int]:
        return (self.epoch, self.scope, self.exc, self.interval)


def epochs(packet: Packet, keys: list[date]) -> list[tuple[date, date]]:
    """Maximal date ranges on which every provision's in-force status is constant (3-day margins)."""
    low = packet.base_start + timedelta(days=10)
    edges = [low] + [k for k in keys if k > low] + [packet.horizon + timedelta(days=1)]
    spans = []
    for a, b in zip(edges, edges[1:]):
        first, last = a + timedelta(days=3), b - timedelta(days=4)
        if (last - first).days >= 2:
            spans.append((first, last))
    return spans


def _cond_value(rng: random.Random | None, spec: Any, cond: Cond, want: bool) -> Any:
    if cond.kind == "is":
        if want:
            return cond.arg
        others = [c for c in spec.choices if c != cond.arg]
        return rng.choice(others) if rng else others[0]
    t = cond.arg
    if cond.kind == "min":
        low, high = (t, min(spec.hi, t + 6)) if want else (max(spec.lo, t - 4), t - 1)
        if want and high > low:
            low = t + 1
    else:
        low, high = (max(spec.lo, t - 5), t) if want else (t + 1, min(spec.hi, t + 4))
        if want and high > low:
            high = t - 1
    if high < low:
        raise core.GenerationError(f"no value for {cond}")
    if rng is None:
        return low if cond.kind == "min" and want else high
    return rng.randint(low, high)


def _skeleton_attrs(
    domain: Domain, exception: Provision | None, state: str
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if exception is None:
        return attrs
    specs = {spec.key: spec for spec in domain.conditions}
    for index, cond in enumerate(exception.conds):
        attrs[cond.attr] = _cond_value(
            None, specs[cond.attr], cond, state != f"no:{index}"
        )
    return attrs


def enumerate_designs(
    packet: Packet, domain: Domain, spans: list[tuple[date, date]]
) -> list[Design]:
    exception = next((p for p in packet.specials if p.cls == "exception"), None)
    scopes = [p.scope for p in packet.specials if p.cls == "annex"] + [""]
    states = (
        (["yes"] + [f"no:{i}" for i in range(len(exception.conds))])
        if exception
        else ["none"]
    )
    values = sorted({p.value for p in packet.provisions if p.sets_value})
    designs: list[Design] = []
    for index, (first, _) in enumerate(spans):
        for scope in scopes:
            for state in states:
                attrs = _skeleton_attrs(domain, exception, state)
                case = Case(attrs, scope, {packet.governs: first})
                res = resolve(packet, case)
                heur = heuristic_provisions(packet, case)
                if packet.interface == "noul":
                    intervals = [
                        (i, (values[i - 1] + values[i]) / 2)
                        for i in range(1, len(values))
                    ]
                else:
                    intervals = [(-1, None)]
                for interval, q in intervals:
                    answer = answer_fn(packet.interface, q)
                    found = analyse(packet, case, answer, res)
                    if found is None:
                        continue
                    labels, subtype = found
                    gold = answer(res.decider.value)
                    answers = {name: answer(p.value) for name, p in heur.items()}
                    pattern = tuple(int(answers[name] == gold) for name in HEURISTICS)
                    designs.append(
                        Design(
                            index,
                            scope,
                            state,
                            interval,
                            gold,
                            pattern,
                            labels,
                            subtype,
                            res.decider.pid,
                            answers,
                        )
                    )
    return designs


# Selection weight of each heuristic pattern (base_only, latest_amendment, first_match right?) among the
# patterns a packet can realise; tuned so that each heuristic is right on roughly 15-40% of rows. Score
# fixes the first_match bit per case (FIRST_MATCH_RIGHT), so its weights only rank patterns sharing it.
PATTERN_WEIGHTS = {
    "choice": {
        (0, 0, 0): 2.5,
        (1, 0, 0): 3.5,
        (1, 0, 1): 0.3,
        (0, 1, 0): 3.5,
        (0, 1, 1): 0.4,
        (0, 0, 1): 0.2,
    },
    "noul": {
        (0, 0, 0): 4.0,
        (1, 0, 0): 1.5,
        (0, 1, 0): 1.5,
        (0, 0, 1): 0.3,
        (1, 1, 0): 0.5,
        (1, 0, 1): 0.1,
        (0, 1, 1): 0.1,
        (1, 1, 1): 0.05,
    },
    "score": {
        (0, 0, 0): 3.0,
        (1, 0, 0): 3.0,
        (0, 1, 0): 3.0,
        (0, 0, 1): 0.5,
        (1, 0, 1): 0.6,
        (0, 1, 1): 0.8,
        (1, 1, 0): 0.5,
        (1, 1, 1): 0.2,
    },
}


# Share of cases whose design must have first_match right (the rest must have it wrong). A Score case, with
# its level fixed and no request quantity to vary, otherwise has only first_match-right designs about half
# the time. Values are redrawn on the same structure up to VALUE_TRIES times; after that the target is
# dropped, since redrawing the structure would skew layouts. With those fallbacks (about 13% of cases)
# first_match is right on about 32% of Score rows.
FIRST_MATCH_RIGHT = {"score": 0.20}
VALUE_TRIES = 30


def group_targets(
    seed_key: str, interface: str, levels: int, index: int | None = None
) -> dict[str, Any]:
    """Group-level targets, fixed before any packet attempt.

    Without ``index`` the answers and levels are drawn from the seed. With ``index`` (the group's position
    among the groups of its domain and interface) the Noul cases answer ``index % 2`` and the other
    answer, so every group and every run of consecutive groups is balanced exactly; the first Score case
    takes level ``index % levels`` and the second a different level drawn from the seed; and both cases
    share one question template, cycled with the index on the period of the answers / first levels.
    ``first_match`` says per case whether the first matching provision must give the gold (None: free).
    """
    rng = core.rng_for(seed_key, "f2-targets")
    first = int(rng.random() < 0.5)
    answers = (first, 1 - first if rng.random() < 0.8 else first)
    low = rng.randrange(levels)
    other = (
        (low + 1 + rng.randrange(levels - 1)) % levels if rng.random() < 0.85 else low
    )
    targets: dict[str, Any] = {
        "answers": answers,
        "levels": (low, other),
        "differ": rng.random() < 0.9,
    }
    share = FIRST_MATCH_RIGHT.get(interface)
    targets["first_match"] = tuple(
        None if share is None else int(rng.random() < share) for _ in range(2)
    )
    targets["question"] = None
    if index is not None:
        if index < 0:
            raise ValueError(f"negative group index {index}")
        low = index % levels
        targets["answers"] = (index % 2, 1 - index % 2)
        # Cycling the second level as well would balance each domain's levels exactly, which pushes a
        # cross-validated majority baseline below chance (each held-out fold's rarest level wins training).
        targets["levels"] = (low, (low + 1 + rng.randrange(levels - 1)) % levels)
        targets["question"] = (
            index // {"choice": 1, "noul": 2, "score": levels}[interface]
        )
    return targets


def select_design(
    rng: random.Random,
    designs: list[Design],
    interface: str,
    targets: dict[str, Any],
    position: int,
    previous: Design | None,
    strict: bool = True,
) -> Design:
    """One case design meeting the group targets; ``strict=False`` gives up the first_match target if unmet."""
    pool = [d for d in designs if 1 <= len(d.labels) <= 4]
    if previous is not None:
        pool = [d for d in pool if d.key != previous.key]
    if interface == "noul":
        pool = [d for d in pool if d.gold == targets["answers"][position]]
    elif interface == "score":
        pool = [d for d in pool if d.gold == targets["levels"][position]]
    elif previous is not None and targets["differ"]:
        differing = [d for d in pool if d.gold != previous.gold]
        pool = differing or pool
    wanted = targets["first_match"][position]
    if wanted is not None:
        matching = [d for d in pool if d.pattern[2] == wanted]
        pool = matching if matching or strict else pool
    if not pool:
        raise core.GenerationError("no case design meets the group targets")
    weights = PATTERN_WEIGHTS[interface]
    patterns = sorted({d.pattern for d in pool})
    pattern = _weighted(rng, tuple((p, weights.get(p, 0.1)) for p in patterns))
    pool = [d for d in pool if d.pattern == pattern]
    subtypes = sorted({d.subtype for d in pool})
    chosen = rng.choice(subtypes)
    return rng.choice([d for d in pool if d.subtype == chosen])


# ---------------------------------------------------------------- text: values, provisions, clauses


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def units(unit: str, amount: Any, currency: str) -> str:
    if unit == "money":
        return core.fmt_money(amount, currency)
    shown = core.fmt_int(amount) if isinstance(amount, int) else str(amount)
    text = unit.replace("{n}", shown)
    if amount == 1 and text.endswith("s"):
        text = text[:-1]
    return text


@dataclasses.dataclass
class Style:
    domain: Domain
    interface: str
    date_style: str
    currency: str
    org: str
    title: str
    office: str
    code: str
    edition: int
    issued: date
    officials: list[str]
    slots: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        d = self.domain
        self.slots = {
            "org": self.org,
            "office": self.office,
            "title": self.title,
            "member": d.member,
            "members": d.members,
            "Member": _cap(d.member),
            "Members": _cap(d.members),
            "person": d.person,
            "Person": _cap(d.person),
            "case": d.case,
            "cases": d.cases,
            "Case": _cap(d.case),
            "Cases": _cap(d.cases),
            "a_case": article(d.case),
        }

    @property
    def var(self) -> Quantity | Levels:
        return getattr(self.domain, self.interface)

    def fill(self, template: str, **extra: Any) -> str:
        return core.fill(template, **{**self.slots, **extra})

    def d(self, value: date) -> str:
        return core.fmt_date(value, self.date_style)

    def value_text(self, value: Any) -> str:
        var = self.var
        if isinstance(var, Levels):
            return var.needles[value]
        return units(var.unit, value, self.currency)

    def option_text(self, value: Any) -> str:
        return self.fill(self.var.option.replace("{val}", self.value_text(value)))

    def frag(self, rng: random.Random, value: Any) -> str:
        var = self.var
        if isinstance(var, Levels):
            return self.fill(var.frags[value])
        return self.fill(rng.choice(var.frags).replace("{val}", self.value_text(value)))

    def cond_value(self, spec: Any, value: Any) -> str:
        return (
            str(value) if spec.kind == "is" else units(spec.unit, value, self.currency)
        )


def cond_phrase(style: Style, cond: Cond) -> str:
    spec = {s.key: s for s in style.domain.conditions}[cond.attr]
    if cond.term:
        return f"{style.fill(spec.term_use)} (as defined in <<def:{cond.term}>>)"
    return style.fill(spec.text.replace("{t}", style.cond_value(spec, cond.arg)))


def definition_text(style: Style, spec: Any, arg: Any) -> str:
    body = style.fill(spec.term_def.replace("{t}", style.cond_value(spec, arg)))
    return f'"{spec.term}" means {body}.'


def text_base(style: Style, rng: random.Random, provision: Provision) -> str:
    var = style.var
    if isinstance(var, Levels):
        lead = rng.choice(("", "As a general rule, ", "In the ordinary case, "))
        body = var.frags[provision.value]
        return style.fill(lead + body if lead else _cap(body)) + "."
    return style.fill(
        rng.choice(var.base).replace("{val}", style.value_text(provision.value))
    )


def text_amendment(
    style: Style, rng: random.Random, packet: Packet, p: Provision
) -> str:
    d, k, base = style.d, p.number, "<<base>>"
    if p.op == "set":
        f = style.frag(rng, p.value)
        options = [
            f"Amendment No. {k} (adopted {d(p.adopted)}) amends {base} with effect from {d(p.start)}. From that "
            f"date, {f}.",
            f'With effect from {d(p.start)}, {base} is amended to read: "{_cap(f)}." (Amendment No. {k}, adopted '
            f"{d(p.adopted)}.)",
            f"Amendment No. {k}, adopted on {d(p.adopted)}, takes effect on {d(p.start)}. It amends {base} so that "
            f"{f}.",
        ]
        return rng.choice(options)
    target = packet.by_pid[p.target]
    j = target.number
    if p.op == "replace":
        f = style.frag(rng, p.value)
        options = [
            f"Amendment No. {k} (adopted {d(p.adopted)}) replaces Amendment No. {j} in full with effect from "
            f"{d(p.start)}. From that date, {f}.",
            f"With effect from {d(p.start)}, Amendment No. {j} is replaced by Amendment No. {k} (adopted "
            f"{d(p.adopted)}), under which {f}.",
        ]
        text = rng.choice(options)
        if p.start <= target.start and rng.random() < 0.5:
            text += f" Amendment No. {j} therefore never takes effect."
        return text
    if p.start > target.start:
        options = [
            f"Amendment No. {k} (adopted {d(p.adopted)}) withdraws Amendment No. {j} with effect from {d(p.start)}. "
            f"From that date, the wording of {base} that applied before Amendment No. {j} took effect applies again.",
            f"Amendment No. {j} is withdrawn with effect from {d(p.start)} by Amendment No. {k}, adopted on "
            f"{d(p.adopted)}. From {d(p.start)}, {base} reads as it did before Amendment No. {j} took effect.",
        ]
    else:
        options = [
            f"Amendment No. {k} (adopted {d(p.adopted)}) withdraws Amendment No. {j} with effect from {d(p.start)}, "
            f"before it was due to take effect. Amendment No. {j} therefore never comes into force.",
            f"Amendment No. {j}, which was due to take effect on {d(target.start)}, is withdrawn by Amendment No. {k} "
            f"(adopted {d(p.adopted)}) with effect from {d(p.start)} and will not come into force.",
        ]
    return rng.choice(options)


def text_exception(style: Style, rng: random.Random, p: Provision) -> str:
    conds = core.join_list([cond_phrase(style, cond) for cond in p.conds])
    f = style.frag(rng, p.value)
    return rng.choice(
        [
            f"Notwithstanding <<base>>, where {conds}, {f}.",
            f"Notwithstanding anything in <<base>>, if {conds}, {f}.",
            f"The following exception applies notwithstanding <<base>>: where {conds}, {f}.",
        ]
    )


def text_temporary(style: Style, rng: random.Random, p: Provision) -> str:
    s, e, f = style.d(p.start), style.d(p.end), style.frag(rng, p.value)
    return rng.choice(
        [
            f"As a temporary measure, from {s} to {e} inclusive, {f}. The measure lapses at the end of {e}.",
            f"Temporary measure: for the period from {s} to {e}, both dates included, {f}. After {e} the measure no "
            f"longer applies.",
            f"From {s} until {e} inclusive, and as a temporary measure only, {f}.",
        ]
    )


def text_annex(style: Style, rng: random.Random, p: Provision) -> str:
    scope = style.fill(style.domain.scope.annex_scope.replace("{v}", p.scope))
    f = style.frag(rng, p.value)
    return rng.choice(
        [
            f"This annex applies to {scope}. Where it applies, {f}. In all other respects the main text applies.",
            f"The rules in this annex apply only to {scope}. Within that scope, {f}.",
            f"Scope of this annex: {scope}. Rule: {f}.",
        ]
    )


def annex_title(style: Style, p: Provision) -> str:
    return style.fill(style.domain.scope.annex_title.replace("{v}", p.scope))


CLASS_PHRASES = {
    "exception": ("an exception whose conditions are met", "an applicable exception"),
    "temporary": (
        "a temporary measure in force on {gov}",
        "a temporary measure then in force",
    ),
    "annex": (
        "an annex whose scope covers the {case}",
        "the annex that covers the {case}",
    ),
    "main": ("the main text as amended", "the main text, as amended"),
}


def precedence_texts(style: Style, rng: random.Random, packet: Packet) -> list[str]:
    dates = style.domain.dates
    gov_phrase, other_phrase = (
        (dates.a_phrase, dates.b_phrase)
        if packet.governs == "a"
        else (dates.b_phrase, dates.a_phrase)
    )
    gov = "the Relevant Date" if packet.gov_via_term else "the governing date"
    variant = rng.randrange(2)
    phrases = [
        style.fill(CLASS_PHRASES[cls][variant], gov=gov)
        for cls in SCHEMES[packet.scheme]
    ]
    c1, c2, c3, c4 = phrases
    scheme = rng.choice(
        [
            f"Where more than one provision of this policy could apply to the same {style.domain.case}, the first of "
            f"the following that applies governs: (a) {c1}; (b) {c2}; (c) {c3}; (d) {c4}.",
            f"Provisions rank in the following order, highest first: {c1}, then {c2}, then {c3}, and finally {c4}. A "
            f"provision of higher rank prevails over any provision of lower rank that would otherwise apply.",
            f"{_cap(c1)} prevails over {c2}; {c2} prevails over {c3}; and {c3} prevails over {c4}. Where only one "
            f"provision applies, that provision governs.",
        ]
    )
    a_case = article(style.domain.case)
    if packet.gov_via_term:
        timing = (
            f"Whether an amendment or a temporary measure is in force for {a_case} is decided by reference "
            f"to the Relevant Date (see <<def:Relevant Date>>)."
        )
    else:
        timing = rng.choice(
            [
                f"The governing date for {a_case} is {gov_phrase}; {other_phrase} does not affect which provisions are "
                f"in force.",
                f"Whether an amendment or a temporary measure is in force for {a_case} is decided by reference to "
                f"{gov_phrase} (the governing date), not {other_phrase}.",
            ]
        )
    within = rng.choice(
        [
            "Within the main text, an amendment replaces the earlier wording from its effective date. An amendment that "
            "has been replaced by a later amendment stops applying from the date on which the replacement takes effect, "
            "and a withdrawn amendment stops applying from the date of withdrawal. No amendment has any effect before "
            "its effective date.",
            "Each amendment changes the main text only from the date on which it takes effect. If an amendment is later "
            "replaced or withdrawn, it ceases to apply from the date of that replacement or withdrawal.",
        ]
    )
    limits = (
        "An exception applies only when every condition it states is met. An annex applies only within the "
        "scope it states. A temporary measure is in force from its first day to its last day inclusive and not "
        "at any other time."
    )
    texts = [style.fill(scheme), style.fill(timing), within, limits]
    if rng.random() < 0.5:
        texts[1], texts[2] = texts[2], texts[1]
    return texts


def definition_entries(
    style: Style, rng: random.Random, packet: Packet
) -> list[tuple[str, str]]:
    """(ref key, text) pairs for the definitions section; terms used by a provision come first in no fixed place."""
    domain = style.domain
    entries: list[tuple[str, str]] = []
    exception = packet.by_pid.get("X1")
    used = {cond.attr for cond in exception.conds} if exception else set()
    specs = {spec.key: spec for spec in domain.conditions}
    if exception:
        for cond in exception.conds:
            if cond.term:
                entries.append(
                    (
                        f"def:{cond.term}",
                        definition_text(style, specs[cond.attr], cond.arg),
                    )
                )
    if packet.gov_via_term:
        dates = domain.dates
        phrase = dates.a_phrase if packet.governs == "a" else dates.b_phrase
        entries.append(("def:Relevant Date", f'"Relevant Date" means {phrase}.'))
    spare = [spec for spec in domain.conditions if spec.key not in used]
    for spec in rng.sample(spare, min(len(spare), rng.randint(0, 2))):
        arg = (
            rng.choice(spec.choices)
            if spec.kind == "is"
            else rng.choice(spec.thresholds)
        )
        entries.append(("", definition_text(style, spec, arg)))
    extra = rng.sample(list(domain.extra_defs), rng.randint(2, 3)) + rng.sample(
        list(GENERIC_DEFS), rng.randint(1, 2)
    )
    entries += [("", style.fill(f'"{term}" means {body}.')) for term, body in extra]
    rng.shuffle(entries)
    return entries


# ---------------------------------------------------------------- text: sections and layouts


@dataclasses.dataclass
class Block:
    key: str
    heading: str
    paras: list[tuple[str, str]]
    kind: str = "section"
    hosts: bool = True


HEADINGS = {
    "formal": {
        "purpose": ("Purpose", "Purpose of this policy"),
        "defs": ("Definitions", "Definitions and interpretation of terms"),
        "scope": ("Scope", "Application of this policy"),
        "exceptions": ("Exceptions", "Exceptions to the general rules"),
        "temporary": ("Temporary measures", "Temporary provisions"),
        "amendments": ("Amendments", "Record of amendments"),
        "interp": ("Interpretation and precedence", "Order of precedence"),
    },
    "handbook": {
        "about": ("About this guide", "Introduction"),
        "defs": ("Key terms", "Words with a special meaning"),
        "exceptions": ("Exceptions to the standard rules", "Exceptions"),
        "temporary": ("Temporary measures", "Temporary arrangements now announced"),
        "amendments": ("Changes since the last edition", "What has changed"),
        "interp": ("When more than one rule could apply", "How the rules fit together"),
    },
}


def core_blocks(
    style: Style, rng: random.Random, packet: Packet, arrangement: list
) -> list[Block]:
    domain, layout = style.domain, packet.layout
    heads = HEADINGS["handbook" if layout == "handbook" else "formal"]
    variant = rng.randrange(2)

    def head(key: str) -> str:
        return heads[key][variant]

    letters = iter("ABCDEF")
    memo_latest = packet.amendments[-1]
    blocks: list[Block] = []
    for key, pids in arrangement:
        if key == "purpose":
            blocks.append(
                Block(
                    key,
                    head(key),
                    [("", style.fill(rng.choice(domain.purpose)))],
                    hosts=False,
                )
            )
        elif key == "about":
            paras = [
                ("", style.fill(rng.choice(domain.purpose))),
                ("", style.fill(rng.choice(domain.scope_text))),
            ]
            blocks.append(Block(key, head(key), paras, hosts=False))
        elif key == "scope":
            paras = [("", style.fill(rng.choice(domain.scope_text)))]
            if any(p.cls == "annex" for p in packet.specials):
                paras.append(
                    (
                        "",
                        "Each annex to this policy applies only within the scope stated in it.",
                    )
                )
            blocks.append(Block(key, head(key), paras))
        elif key == "defs":
            lead = rng.choice(
                (
                    "In this policy the following words have the meanings shown.",
                    "The following terms are used in this document.",
                )
            )
            blocks.append(
                Block(
                    key,
                    head(key),
                    [("", lead)] + definition_entries(style, rng, packet),
                    hosts=False,
                )
            )
        elif key == "main":
            statics = [
                ("", style.fill(text))
                for text in rng.sample(list(domain.static), rng.randint(1, 2))
            ]
            paras = [("base", text_base(style, rng, packet.base))]
            paras = (
                paras + statics
                if rng.random() < 0.6
                else statics[:1] + paras + statics[1:]
            )
            if "X1" in pids:
                paras.append(("X1", text_exception(style, rng, packet.by_pid["X1"])))
            blocks.append(Block(key, style.var.heading, paras))
        elif key == "exceptions":
            blocks.append(
                Block(
                    key,
                    head(key),
                    [("X1", text_exception(style, rng, packet.by_pid["X1"]))],
                )
            )
        elif key == "temporary":
            hosts = not (layout == "memo" and any(b.key == "att3" for b in blocks))
            blocks.append(
                Block(
                    key,
                    head(key),
                    [("T1", text_temporary(style, rng, packet.by_pid["T1"]))],
                    hosts=hosts,
                )
            )
        elif key == "amendments":
            paras = [
                (pid, text_amendment(style, rng, packet, packet.by_pid[pid]))
                for pid in pids
            ]
            title = "Earlier amendments" if layout == "memo" else head(key)
            blocks.append(Block(key, title, paras, hosts=layout != "memo"))
        elif key == "interp":
            blocks.append(
                Block(
                    key,
                    head(key),
                    [("", text) for text in precedence_texts(style, rng, packet)],
                )
            )
        elif key.startswith("annex:"):
            p = packet.by_pid[pids[0]]
            heading = f"Annex {next(letters)} — {annex_title(style, p)}"
            blocks.append(
                Block(
                    key,
                    heading,
                    [(p.pid, text_annex(style, rng, p))],
                    kind="annex",
                    hosts=False,
                )
            )
        elif key == "annexes":
            paras = []
            for pid in pids:
                p = packet.by_pid[pid]
                paras.append(
                    (
                        pid,
                        f"Annex {next(letters)} ({annex_title(style, p)}). {text_annex(style, rng, p)}",
                    )
                )
            blocks.append(Block(key, "Annexes", paras, hosts=False))
        elif key == "preamble":
            text = (
                f"This consolidated text reproduces the {style.title} as adopted on {style.d(packet.base_start)}. "
                f"Amendments made since then are recorded in the change log at the end of this document, with "
                f"the dates from which they take effect; they have not been written into the articles."
            )
            blocks.append(Block(key, "", [("", text)], kind="raw", hosts=False))
        elif key == "changelog":
            paras = [
                (pid, text_amendment(style, rng, packet, packet.by_pid[pid]))
                for pid in pids
            ]
            blocks.append(Block(key, "Change log", paras, kind="log", hosts=False))
        elif key == "memo":
            memo_day = memo_latest.adopted + _days(rng, 1, 6)
            header = (
                f"MEMORANDUM\nTo: All {domain.members}\nFrom: {style.officials[2]}, {style.office}\n"
                f"Date: {style.d(memo_day)}\nSubject: {style.title}, Amendment No. {memo_latest.number}"
            )
            intro = (
                f"This memorandum gives notice of Amendment No. {memo_latest.number} to the {style.title} of "
                f"{style.org}."
            )
            attached = ["the policy as originally adopted (Attachment 1)"]
            if len(packet.amendments) > 1:
                attached.append("the earlier amendments (Attachment 2)")
            if any(k == "att3" for k, _ in arrangement):
                attached.append("the temporary measures and annexes (Attachment 3)")
            closing = (
                f"For reference, this memorandum attaches {core.join_list(attached)}."
            )
            body = [
                ("", intro),
                (memo_latest.pid, text_amendment(style, rng, packet, memo_latest)),
                ("", closing),
            ]
            blocks.append(Block(key, header, body, kind="raw", hosts=False))
        elif key in ("att1", "att2", "att3"):
            names = {
                "att1": f"ATTACHMENT 1: {style.title} (as adopted on {style.d(packet.base_start)})",
                "att2": "ATTACHMENT 2: Earlier amendments",
                "att3": "ATTACHMENT 3: Temporary measures and annexes",
            }
            blocks.append(Block(key, names[key], [], kind="raw", hosts=False))
    return blocks


def title_block(style: Style, packet: Packet) -> str:
    d = style.d
    if packet.layout == "numbered":
        return f"{style.org}\n{style.title}\nEdition {style.edition}, issued {d(style.issued)}. Reference {style.code}."
    if packet.layout == "handbook":
        return (
            f"{style.title}: a guide for {style.domain.members}\n{style.org}. Edition {style.edition}, "
            f"{d(style.issued)}."
        )
    if packet.layout == "consolidated":
        return f"{style.org}\n{style.title}\nConsolidated text as at {d(style.issued)}. Reference {style.code}."
    return ""


def render_blocks(
    style: Style, packet: Packet, blocks: list[Block]
) -> tuple[str, dict[str, str]]:
    """Number the blocks for the layout, resolve cross-references and return (text, paragraph by ref key)."""
    layout = packet.layout
    labels: dict[str, str] = {}
    plan: list[tuple[str, list[tuple[str, str, str]]]] = []
    number = 0
    for block in blocks:
        rows: list[tuple[str, str, str]] = []
        if block.kind == "section":
            number += 1
            if layout in ("numbered", "memo"):
                heading = f"{number}. {block.heading}"
            elif layout == "consolidated":
                heading = f"Article {number}: {block.heading}"
            else:
                heading = block.heading
            for index, (ref, text) in enumerate(block.paras, 1):
                if layout in ("numbered", "memo"):
                    prefix, label = f"{number}.{index} ", f"clause {number}.{index}"
                elif layout == "consolidated":
                    prefix, label = f"({index}) ", f"Article {number}({index})"
                else:
                    prefix, label = "", f'the section "{block.heading}"'
                if ref:
                    labels[ref] = label
                rows.append((prefix, ref, text))
        else:
            heading = block.heading
            rows = [("", ref, text) for ref, text in block.paras]
        plan.append((heading, rows))
    rendered: dict[str, str] = {}
    parts = [title_block(style, packet)] if title_block(style, packet) else []
    for heading, rows in plan:
        lines = [heading] if heading else []
        for prefix, ref, text in rows:
            for token, label in labels.items():
                if token == "base" or token.startswith("def:"):
                    text = text.replace(f"<<{token}>>", label)
            if "<<" in text:
                raise core.GenerationError(f"unresolved reference in {text[:60]!r}")
            lines.append(prefix + text)
            if ref:
                rendered[ref] = text
        parts.append("\n".join(lines))
    return "\n\n".join(parts), rendered


def filler_blocks(style: Style, rng: random.Random) -> list[Block]:
    pool = list(style.domain.distractors) + list(GENERIC_DISTRACTORS)
    rng.shuffle(pool)
    blocks = []
    for heading, body in pool:
        numbers = {
            "wd": rng.randint(2, 10),
            "dd": rng.randint(14, 60),
            "mm": rng.randint(6, 36),
            "hh": rng.randint(2, 6),
            "nn": rng.randint(2, 9),
            "pct": rng.randint(5, 25),
            "ext": rng.randint(2000, 7999),
            "yy": style.issued.year,
            "name1": style.officials[0],
            "name2": style.officials[1],
        }
        blocks.append(
            Block(
                "filler",
                style.fill(heading, **numbers),
                [("", style.fill(body, **numbers))],
            )
        )
    return blocks


def draw_target_length(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.40:
        return rng.randint(4150, 4950)
    if u < 0.75:
        return rng.randint(5000, 7000)
    return 7000 + int((10900 - 7000) * rng.random() ** 1.4)


def assemble_packet(
    style: Style,
    rng: random.Random,
    packet: Packet,
    blocks: list[Block],
    target: int,
    cap: int,
) -> tuple[str, dict[str, str]]:
    """Insert distractor sections after host blocks until the packet length is within the target window."""
    cap = min(cap, PACKET_MAX)
    text, rendered = render_blocks(style, packet, blocks)
    low = max(PACKET_MIN, min(target, cap) - 250)
    high = min(cap, max(low + 500, target + 250))
    if len(text) > high:
        if len(text) > cap:
            raise core.GenerationError(f"core packet {len(text)} chars exceeds {cap}")
        return text, rendered
    hosts = [index for index, block in enumerate(blocks) if block.hosts]
    pool = filler_blocks(style, rng)
    chosen: list[tuple[int, float, Block]] = []

    def build() -> tuple[str, dict[str, str]]:
        merged: list[Block] = []
        for index, block in enumerate(blocks):
            merged.append(block)
            merged.extend(
                b
                for i, _, b in sorted(chosen, key=lambda item: (item[0], item[1]))
                if i == index
            )
        return render_blocks(style, packet, merged)

    estimate = len(text)
    while pool and estimate < target:
        block = pool.pop()
        chosen.append((rng.choice(hosts), rng.random(), block))
        estimate += len(block.heading) + sum(len(t) for _, t in block.paras) + 12
    for _ in range(40):
        text, rendered = build()
        if len(text) < low and pool:
            block = pool.pop()
            chosen.append((rng.choice(hosts), rng.random(), block))
        elif len(text) > high and chosen:
            chosen.pop(rng.randrange(len(chosen)))
        else:
            break
    if not PACKET_MIN <= len(text) <= cap:
        raise core.GenerationError(
            f"packet length {len(text)} outside [{PACKET_MIN}, {cap}]"
        )
    return text, rendered


# ---------------------------------------------------------------- text: case and question

CASE_HEADS = (
    "{Case} for assessment",
    "Case details",
    "{Case} under review",
    "Details of the {case}",
)


def render_case(
    style: Style,
    rng: random.Random,
    packet: Packet,
    case: Case,
    name: str,
    shown: list[str],
    noise: list[tuple],
) -> tuple[str, dict[str, str]]:
    """Render the case in one of four formats; returns text and the rendered item per attribute key."""
    domain = style.domain
    date_style = (
        style.date_style if rng.random() < 0.7 else rng.choice(core.DATE_STYLES)
    )
    specs = {spec.key: spec for spec in domain.conditions}
    items: list[tuple[str, str, str, str]] = []
    scope_sentence = rng.choice(domain.scope.case)
    middle = [("scope", domain.scope.label, case.scope, scope_sentence)]
    for key in shown:
        spec = specs[key]
        middle.append(
            (
                key,
                spec.label,
                style.cond_value(spec, case.attrs[key]),
                rng.choice(spec.case),
            )
        )
    for spec, value in noise:
        middle.append((f"noise:{spec.label}", spec.label, value, spec.case))
    rng.shuffle(middle)
    dates = domain.dates
    items.append(("claimant", domain.claimant_label, name, ""))
    items += middle
    items.append(
        (
            "date:a",
            dates.a_label,
            core.fmt_date(case.dates["a"], date_style),
            dates.a_case,
        )
    )
    items.append(
        (
            "date:b",
            dates.b_label,
            core.fmt_date(case.dates["b"], date_style),
            dates.b_case,
        )
    )
    if packet.interface == "noul":
        var = domain.noul
        q_text = units(var.unit, case.q, style.currency)
        item = ("q", var.ask_label, q_text, rng.choice(var.ask))
        items.insert(rng.randint(len(items) - 2, len(items)), item)
    form = rng.choice(("form", "bullets", "prose", "note"))
    rendered: dict[str, str] = {}
    if form in ("form", "bullets"):
        lines = [style.fill(rng.choice(CASE_HEADS))]
        for key, label, value, _ in items:
            line = f"{'- ' if form == 'bullets' else ''}{label}: {value}"
            rendered[key] = f"{label}: {value}"
            lines.append(line)
        return "\n".join(lines), rendered
    sentences = []
    for key, label, value, sentence in items:
        if key == "claimant":
            continue
        text = style.fill(sentence, name=name, v=value, d=value, q=value)
        rendered[key] = text
        sentences.append(text)
    if form == "note":
        opening = (
            f"Note from the {style.office}: we have received {article(domain.case)} from {name} and need to "
            f"check it against the policy."
        )
    else:
        opening = f"{style.fill(rng.choice(CASE_HEADS))}. The {domain.case} below has been received from {name}."
    rendered["claimant"] = name
    return opening + "\n" + " ".join(sentences), rendered


def render_question(
    style: Style, rng: random.Random, name: str, slot: int | None = None
) -> tuple[str, int]:
    """One question template: ``slot`` (a group target) modulo the template count, else a random one."""
    templates = style.domain.questions[style.interface]
    index = rng.randrange(len(templates)) if slot is None else slot % len(templates)
    return style.fill(templates[index], name=name), index


# ---------------------------------------------------------------- cases


def _pick_dates(
    rng: random.Random,
    packet: Packet,
    span: tuple[date, date],
    lag: int,
    keys: list[date],
    cross: bool,
) -> tuple[date, date]:
    """Governing date inside ``span``; with ``cross`` the other date is placed beyond a neighbouring key date."""
    first, last = span
    when = None
    if cross and packet.governs == "a":
        boundary = next((k for k in keys if k > last), None)
        if boundary is not None:
            low = max(first, boundary - timedelta(days=lag - 3))
            if low <= last:
                when = low + timedelta(days=rng.randint(0, (last - low).days))
    elif cross:
        boundary = max((k for k in keys if k <= first), default=None)
        if boundary is not None:
            high = min(last, boundary + timedelta(days=lag - 3))
            if high >= first:
                when = first + timedelta(days=rng.randint(0, (high - first).days))
    if when is None:
        when = first + timedelta(days=rng.randint(0, (last - first).days))
    step = timedelta(days=lag if packet.governs == "a" else -lag)
    other = when + step
    if any(abs((other - k).days) <= 2 for k in keys):
        other += timedelta(days=5 if packet.governs == "a" else -5)
    return when, other


def _pick_q(rng: random.Random, unit: str, low: Any, high: Any) -> Any:
    if unit == "money":
        q = low + rng.uniform(0.15, 0.85) * (high - low)
        q = round(q) if rng.random() < 0.6 else round(q, 2)
        return q if low < q < high else (low + high) / 2
    return rng.randint(low + 1, high - 1)


def realize(
    rng: random.Random,
    packet: Packet,
    domain: Domain,
    design: Design,
    spans: list,
    keys: list[date],
    style: Style,
) -> tuple[Case, dict[str, Any]]:
    exception = packet.by_pid.get("X1")
    specs = {spec.key: spec for spec in domain.conditions}
    attrs: dict[str, Any] = {}
    shown: list[str] = []
    if exception is not None:
        for index, cond in enumerate(exception.conds):
            attrs[cond.attr] = _cond_value(
                rng, specs[cond.attr], cond, design.exc != f"no:{index}"
            )
            shown.append(cond.attr)
    others = [spec for spec in domain.conditions if spec.key not in attrs]
    for spec in rng.sample(others, rng.randint(max(0, 2 - len(shown)), 3 - len(shown))):
        attrs[spec.key] = (
            rng.choice(spec.choices)
            if spec.kind == "is"
            else rng.randint(spec.lo, spec.hi)
        )
        shown.append(spec.key)
    for spec in others:
        attrs.setdefault(
            spec.key,
            (
                rng.choice(spec.choices)
                if spec.kind == "is"
                else rng.randint(spec.lo, spec.hi)
            ),
        )
    annexed = {p.scope for p in packet.specials if p.cls == "annex"}
    scope = design.scope or rng.choice(
        [v for v in domain.scope.values if v not in annexed]
    )
    q = None
    if packet.interface == "noul":
        values = sorted({p.value for p in packet.provisions if p.sets_value})
        q = _pick_q(
            rng, domain.noul.unit, values[design.interval - 1], values[design.interval]
        )
    base_count = 1 + len(shown) + (q is not None)
    noise_specs = rng.sample(
        list(domain.noise), rng.randint(max(0, 4 - base_count), min(2, 7 - base_count))
    )
    noise = []
    for spec in noise_specs:
        value = rng.choice(spec.values).replace("{n}", str(rng.randint(1000, 99999)))
        noise.append((spec, style.fill(value)))
    lag = rng.randint(*domain.dates.lag)
    cross = rng.random() < 0.5
    when, other = _pick_dates(rng, packet, spans[design.epoch], lag, keys, cross)
    dates = (
        {"a": when, "b": other} if packet.governs == "a" else {"a": other, "b": when}
    )
    case = Case(attrs, scope, dates, q)
    return case, {"shown": shown, "noise": noise, "other": other}


def _facts(packet: Packet, case: Case, name: str) -> dict[str, Any]:
    def iso(value: date | None) -> str | None:
        return value.isoformat() if value else None

    provisions = []
    definitions: dict[str, Any] = {}
    for p in packet.provisions:
        conds = []
        for cond in p.conds:
            if cond.term:
                conds.append({"term": cond.term})
                definitions[cond.term] = {
                    "attr": cond.attr,
                    "kind": cond.kind,
                    "arg": cond.arg,
                }
            else:
                conds.append({"attr": cond.attr, "kind": cond.kind, "arg": cond.arg})
        provisions.append(
            {
                "pid": p.pid,
                "cls": p.cls,
                "value": p.value,
                "start": iso(p.start),
                "end": iso(p.end),
                "number": p.number,
                "op": p.op,
                "target": p.target,
                "adopted": iso(p.adopted),
                "scope": p.scope,
                "conds": conds,
            }
        )
    return {
        "interface": packet.interface,
        "packet": {
            "domain": packet.domain,
            "scheme": packet.scheme,
            "governs": packet.governs,
            "gov_via_term": packet.gov_via_term,
            "layout": packet.layout,
            "provisions": provisions,
            "definitions": definitions,
            "order": list(packet.order),
            "base_start": iso(packet.base_start),
        },
        "case": {
            "claimant": name,
            "attrs": dict(case.attrs),
            "scope": case.scope,
            "dates": {key: iso(value) for key, value in case.dates.items()},
            "q": case.q,
        },
    }


def _hazards(
    packet: Packet, case: Case, answer: Callable[[Any], int], other: date
) -> tuple[tuple[str, ...], str, bool]:
    found = analyse(packet, case, answer)
    if found is None:
        raise core.GenerationError("realised case has no hazard")
    labels, subtype = found
    gold = answer(resolve(packet, case).decider.value)
    contrast = answer(resolve(packet, case, other).decider.value) != gold
    if contrast and packet.gov_via_term and "cross_reference" not in labels:
        labels = tuple(sorted(labels + ("cross_reference",)))
    if not 1 <= len(labels) <= 4:
        raise core.GenerationError(f"{len(labels)} hazards")
    return labels, subtype, contrast


def _required_values(packet: Packet, case: Case) -> list[Any]:
    """The gold value and the naive-heuristic values, which every Choice option set contains."""
    heur = heuristic_provisions(packet, case)
    return list(
        dict.fromkeys(
            [resolve(packet, case).decider.value] + [p.value for p in heur.values()]
        )
    )


def option_count(rng: random.Random, packet: Packet, cases: list[Case]) -> int:
    """One option count (4 or 5) for both cases of a group, so that gold positions balance per count."""
    available = len({p.value for p in packet.provisions if p.sets_value})
    need = max(len(_required_values(packet, case)) + 1 for case in cases)
    count = min(5, max(rng.choice((4, 5)), need), available)
    if count < 4:
        raise core.GenerationError("fewer than four option values in the packet")
    return count


def _options(rng: random.Random, packet: Packet, case: Case, count: int) -> list[Any]:
    must = _required_values(packet, case)
    others = [
        v
        for v in sorted({p.value for p in packet.provisions if p.sets_value})
        if v not in must
    ]
    rng.shuffle(others)
    chosen = must + others[: max(0, count - len(must))]
    if len(chosen) != count:
        raise core.GenerationError(
            f"{len(chosen)} option values for a {count}-option group"
        )
    return sorted(chosen)


def _item(
    rng: random.Random,
    style: Style,
    packet: Packet,
    packet_text: str,
    case: Case,
    info: dict[str, Any],
    name: str,
    case_text: str,
    case_items: dict[str, str],
    question: str,
    qid: int,
    position: str,
    n_options: int = 0,
) -> core.Item:
    domain, interface = style.domain, packet.interface
    answer = answer_fn(interface, case.q)
    res = resolve(packet, case)
    decider = res.decider
    labels, subtype, contrast = _hazards(packet, case, answer, info["other"])
    heur = heuristic_provisions(packet, case)
    facts = _facts(packet, case, name)
    option_refs: dict[str, int] = {}
    meta_heur = None
    if interface == "choice":
        values = _options(rng, packet, case, n_options)
        choices = tuple(style.option_text(v) for v in values)
        gold = values.index(decider.value)
        option_refs = {key: values.index(p.value) for key, p in heur.items()}
        facts["options"] = values
    elif interface == "noul":
        choices = ()
        gold = answer(decider.value)
        meta_heur = {key: answer(p.value) for key, p in heur.items()}
    else:
        choices = tuple(style.fill(text) for text in domain.score.criteria)
        gold = decider.value
        meta_heur = {key: p.value for key, p in heur.items()}
        facts["levels"] = len(choices)
    if position == "after":
        state = packet_text + "\n\n" + case_text
    else:
        lead = rng.choice(
            (
                f"The {domain.case} above is to be assessed under the following policy.",
                f"The policy that applies to this {domain.case} is reproduced below.",
                "The relevant policy document follows.",
            )
        )
        state = case_text + "\n\n" + lead + "\n\n" + packet_text
    needles = [style.value_text(decider.value), case_items[f"date:{packet.governs}"]]
    if decider.cls == "main" and decider is not packet.base:
        needles.append(style.d(decider.start))
    elif decider.cls == "temporary":
        needles += [style.d(decider.start), style.d(decider.end)]
    elif decider.cls == "annex":
        needles += [decider.scope, case_items["scope"]]
    specs = {spec.key: spec for spec in domain.conditions}
    exception = packet.by_pid.get("X1")
    if exception is not None and (
        "exception_applies" in labels or "exception_condition_unmet" in labels
    ):
        for cond in exception.conds:
            needles += [
                style.cond_value(specs[cond.attr], cond.arg),
                case_items[cond.attr],
            ]
    if interface == "noul":
        needles.append(case_items["q"])
    core.require_present(state, needles)
    if len(state) > STATE_MAX or not PACKET_MIN <= len(packet_text) <= PACKET_MAX:
        raise core.GenerationError("state length out of range")
    decider_kind = (
        decider.cls
        if decider.cls != "main"
        else ("base" if decider is packet.base else "amendment")
    )
    dates = domain.dates
    meta: dict[str, Any] = {
        "interface": interface,
        "length": "long",
        "domain": domain.key,
        "scheme": packet.scheme,
        "governing_date_rule": dates.a_key if packet.governs == "a" else dates.b_key,
        "hazards": list(labels),
        "n_amendments": len(packet.amendments),
        "packet_chars": len(packet_text),
        "layout": packet.layout,
        "decider": decider_kind,
        "governing_date_via_definition": packet.gov_via_term,
        "date_contrast": contrast,
        "case_position": position,
    }
    if meta_heur is not None:
        meta["heuristics"] = meta_heur
    else:
        meta["n_options"] = len(choices)
    return core.Item(
        task_type=interface,
        state=state,
        instructions=question,
        choices=choices,
        gold=gold,
        recheck=recheck(facts),
        kind=domain.key,
        subtype=subtype,
        variant=f"{packet.layout}/{packet.scheme}/q{qid + 1}",
        facts=facts,
        option_refs=option_refs,
        meta=meta,
        probe_view=case_text + "\n\n" + question,
    )


def _style(rng: random.Random, domain: Domain, interface: str, packet: Packet) -> Style:
    officials = core.people(rng, 3)
    issued = max(p.adopted for p in packet.amendments) + _days(rng, 5, 40)
    code = (
        "".join(word[0] for word in domain.key.split("_")).upper()
        + f"-{rng.randint(100, 999)}"
    )
    return Style(
        domain,
        interface,
        rng.choice(("mdy", "dmy", "iso", "dmy", "mdy_short")),
        rng.choice(domain.currency),
        rng.choice(domain.orgs),
        rng.choice(domain.titles),
        rng.choice(domain.offices),
        code,
        rng.randint(2, 9),
        issued,
        officials,
    )


def _build(
    rng: random.Random, domain: Domain, interface: str, targets: dict[str, Any]
) -> list[core.Item]:
    packet, extra = draw_packet(rng, domain, interface, targets)
    spans = epochs(packet, extra["keys"])
    for trial in range(VALUE_TRIES):
        if trial:
            packet = redraw_values(rng, packet, domain, targets)
        designs = enumerate_designs(packet, domain, spans)
        strict = trial < VALUE_TRIES - 1
        try:
            first = select_design(rng, designs, interface, targets, 0, None, strict)
            second = select_design(rng, designs, interface, targets, 1, first, strict)
            break
        except core.GenerationError:
            if not strict:
                raise
    style = _style(rng, domain, interface, packet)
    names = core.people(rng, 2, exclude=style.officials)
    cases = []
    for design, name in zip((first, second), names):
        case, info = realize(rng, packet, domain, design, spans, extra["keys"], style)
        answer = answer_fn(interface, case.q)
        if answer(resolve(packet, case).decider.value) != design.gold:
            raise core.GenerationError("realised case disagrees with its design")
        case_text, case_items = render_case(
            style, rng, packet, case, name, info["shown"], info["noise"]
        )
        question, qid = render_question(style, rng, name, targets["question"])
        position = "after" if rng.random() < 0.7 else "before"
        cases.append((case, info, name, case_text, case_items, question, qid, position))
    overhead = max(len(c[3]) + 160 for c in cases)
    blocks = core_blocks(style, rng, packet, extra["arrangement"])
    packet_text, rendered = assemble_packet(
        style, rng, packet, blocks, draw_target_length(rng), STATE_MAX - overhead
    )
    positions = [packet_text.find(rendered[pid]) for pid in packet.order]
    if min(positions, default=0) < 0 or positions != sorted(positions):
        raise core.GenerationError("document order differs from the recorded order")
    n_options = (
        option_count(rng, packet, [c[0] for c in cases]) if interface == "choice" else 0
    )
    items = [
        _item(rng, style, packet, packet_text, *case, n_options=n_options)
        for case in cases
    ]
    for item in items:
        item.check()
    return items


def make_group(
    seed_key: str, kind: str, interface: str, length: str, index: int | None = None
) -> list[core.Item]:
    """Two different cases on one packet, same interface; retried deterministically on GenerationError.

    ``index`` is the group's position among the groups of its (domain, interface) in a build; when given,
    the answer / level targets and the question template follow it (see ``group_targets``).
    """
    if kind not in DOMAINS:
        raise KeyError(kind)
    if interface not in INTERFACES or length not in LENGTH_SHARES:
        raise ValueError(f"unsupported interface/length {interface}/{length}")
    domain = DOMAINS[kind]
    targets = group_targets(seed_key, interface, len(domain.score.frags), index)
    error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            return _build(
                core.rng_for(seed_key, "attempt", attempt), domain, interface, targets
            )
        except core.RecheckMismatch:
            raise
        except core.GenerationError as exc:
            error = exc
    raise core.GenerationError(
        f"{kind}/{interface}: no valid group in {ATTEMPTS} attempts ({error})"
    )
