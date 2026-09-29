from __future__ import annotations

import collections
import dataclasses
import random
import re
import unittest
from datetime import date

from v2.data.hs1 import core, f2_policy
from v2.data.hs1.f2_policy import Case, Cond, Packet, Provision

GROUPS = 8
BALANCE_GROUPS = 200
INDEXED_GROUPS = 30
DOMAINS = f2_policy.KINDS_TRAIN + f2_policy.KINDS_OOD
# Upper limits of the A3c audit per interface, and the design aim for first_match
HEURISTIC_LIMITS = {"choice": 0.45, "noul": 0.60, "score": 0.45}
FIRST_MATCH_AIM = {"choice": 0.40, "score": 0.40}
HEURISTIC_FLOOR = 0.15
STATE = (4000, 12000)
# make_group redraws on any GenerationError; these messages mean an inconsistency, not a missed constraint
INCONSISTENT = re.compile(
    r"recheck|oracle|disagrees|two applicable|document order|unresolved reference|slot missing"
)


def _key(kind: str, interface: str, index: int) -> str:
    return f"hs1-test:{f2_policy.FAMILY}:{kind}:{interface}:{index}"


class F2FamilyTest(unittest.TestCase):
    built: dict[tuple[str, str], list[tuple[str, list[core.Item]]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.built = {}
        for kind in DOMAINS:
            for interface in f2_policy.supported(kind):
                cls.built[(kind, interface)] = [
                    (key, f2_policy.make_group(key, kind, interface, "long"))
                    for key in (_key(kind, interface, index) for index in range(GROUPS))
                ]

    def test_every_domain_and_interface_builds_rows(self) -> None:
        self.assertEqual(len(DOMAINS), 10)
        self.assertEqual(len(self.built), 30)
        for (kind, interface), cell in self.built.items():
            for index, (key, items) in enumerate(cell):
                with self.subTest(kind=kind, interface=interface, index=index):
                    self.assertEqual(len(items), 2)
                    for item in items:
                        item.check()
                        self.assertEqual((item.task_type, item.kind), (interface, kind))
                        self.assertEqual(item.gold, item.recheck)
                        self.assertEqual(item.gold, f2_policy.recheck(dict(item.facts)))
                        self.assertIn(item.subtype, f2_policy.HAZARDS)
                        self.assertTrue(1 <= len(item.meta["hazards"]) <= 4)
                        self.assertTrue(
                            set(item.meta["hazards"]) <= set(f2_policy.HAZARDS)
                        )
                        if interface != "choice":
                            self.assertEqual(
                                set(item.meta["heuristics"]), set(f2_policy.HEURISTICS)
                            )
                        else:
                            self.assertEqual(
                                set(item.option_refs), set(f2_policy.HEURISTICS)
                            )
                    split = "select" if kind in f2_policy.KINDS_OOD else "train"
                    rows = core.build_rows(
                        f2_policy.FAMILY,
                        key,
                        items,
                        split=split,
                        slice_name=split,
                        gold_target=index,
                    )
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(len({row["group_id"] for row in rows}), 1)
                    for row in rows:
                        audit = row["audit_metadata"]
                        self.assertEqual(row["label"], audit["oracle_label"])
                        self.assertEqual(audit["oracle_label"], audit["recheck_label"])
                    if interface == "choice":
                        self.assertEqual(
                            rows[0]["label"], index % len(rows[0]["options"])
                        )

    def test_state_and_packet_lengths(self) -> None:
        for (kind, interface), cell in self.built.items():
            for key, items in cell:
                for item in items:
                    self.assertTrue(
                        STATE[0] <= len(item.state) <= STATE[1],
                        f"{key}: {len(item.state)}",
                    )
                    self.assertTrue(
                        f2_policy.PACKET_MIN
                        <= item.meta["packet_chars"]
                        <= f2_policy.PACKET_MAX,
                        key,
                    )

    def test_no_unit_noun_is_rendered_twice(self) -> None:
        doubled = re.compile(
            r"\b\d[\d,.]* (nights|days|hours|weeks|months|years|minutes|sessions)"
            r" (?:[a-z]+ ){0,2}\1\b"
        )
        for (kind, interface), cell in self.built.items():
            for key, items in cell:
                for item in items:
                    with self.subTest(kind=kind, interface=interface, key=key):
                        self.assertIsNone(doubled.search(item.state))

    def test_two_cases_share_one_packet(self) -> None:
        for (kind, interface), cell in self.built.items():
            for key, (first, second) in cell:
                self.assertEqual(first.facts["packet"], second.facts["packet"], key)
                self.assertEqual(
                    first.meta["packet_chars"], second.meta["packet_chars"], key
                )
                self.assertNotEqual(first.facts["case"], second.facts["case"], key)
                self.assertNotEqual(first.state, second.state, key)

    def test_deterministic(self) -> None:
        for (kind, interface), cell in self.built.items():
            key, items = cell[0]
            again = f2_policy.make_group(key, kind, interface, "long")
            self.assertEqual(
                [dataclasses.asdict(i) for i in items],
                [dataclasses.asdict(i) for i in again],
            )
            rows = core.build_rows(
                f2_policy.FAMILY,
                key,
                items,
                split="train",
                slice_name="train",
                gold_target=0,
            )
            rows_again = core.build_rows(
                f2_policy.FAMILY,
                key,
                again,
                split="train",
                slice_name="train",
                gold_target=0,
            )
            self.assertEqual(rows, rows_again)
        states = [
            item.state
            for cell in self.built.values()
            for _, items in cell
            for item in items
        ]
        self.assertEqual(len(set(states)), len(states))

    def test_no_label_disagreement_is_redrawn(self) -> None:
        for (kind, interface), cell in self.built.items():
            domain = f2_policy.DOMAINS[kind]
            for key, _ in cell:
                targets = f2_policy.group_targets(
                    key, interface, len(domain.score.frags)
                )
                for attempt in range(f2_policy.ATTEMPTS):
                    try:
                        f2_policy._build(
                            core.rng_for(key, "attempt", attempt),
                            domain,
                            interface,
                            targets,
                        )
                        break
                    except core.GenerationError as exc:
                        self.assertIsNone(
                            INCONSISTENT.search(str(exc)),
                            f"{key} attempt {attempt}: {exc}",
                        )

    def test_noul_golds_are_balanced(self) -> None:
        golds, firsts = [], []
        for index in range(BALANCE_GROUPS):
            kind = DOMAINS[index % len(DOMAINS)]
            items = f2_policy.make_group(
                f"hs1-test:f2-balance:{index}", kind, "noul", "long"
            )
            golds += [item.gold for item in items]
            firsts.append(items[0].gold)
        self.assertTrue(
            0.45 <= sum(golds) / len(golds) <= 0.55, sum(golds) / len(golds)
        )
        self.assertTrue(
            0.40 <= sum(firsts) / len(firsts) <= 0.60, sum(firsts) / len(firsts)
        )

    def test_index_none_is_the_default(self) -> None:
        for kind, interface in (
            ("travel_expense", "noul"),
            ("grant_funding", "score"),
            ("it_access", "choice"),
        ):
            key = _key(kind, interface, 0)
            default = f2_policy.make_group(key, kind, interface, "long")
            explicit = f2_policy.make_group(key, kind, interface, "long", index=None)
            self.assertEqual(
                [dataclasses.asdict(i) for i in default],
                [dataclasses.asdict(i) for i in explicit],
            )
        with self.assertRaises(ValueError):
            f2_policy.make_group(
                _key("travel_expense", "noul", 0),
                "travel_expense",
                "noul",
                "long",
                index=-1,
            )


class F2IndexedBuildTest(unittest.TestCase):
    """Groups built as the build does: ``index`` counts the groups of each (domain, interface) from 0."""

    built: dict[tuple[str, str], list[list[core.Item]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.built = {
            (kind, interface): [
                f2_policy.make_group(
                    f"hs1-test:f2-indexed:{kind}:{interface}:{index}",
                    kind,
                    interface,
                    "long",
                    index=index,
                )
                for index in range(INDEXED_GROUPS)
            ]
            for kind in f2_policy.KINDS_TRAIN
            for interface in f2_policy.INTERFACES
        }

    def test_noul_answers_balance_exactly(self) -> None:
        for kind in f2_policy.KINDS_TRAIN:
            golds = []
            for index, items in enumerate(self.built[(kind, "noul")]):
                self.assertEqual(
                    [item.gold for item in items],
                    [index % 2, 1 - index % 2],
                    (kind, index),
                )
                golds += [item.gold for item in items]
            self.assertEqual(sum(golds) * 2, len(golds), kind)

    def test_first_score_level_cycles_with_the_index(self) -> None:
        for kind in f2_policy.KINDS_TRAIN:
            levels = len(f2_policy.DOMAINS[kind].score.frags)
            for index, (first, second) in enumerate(self.built[(kind, "score")]):
                self.assertEqual(first.gold, index % levels, (kind, index))
                self.assertNotEqual(second.gold, first.gold, (kind, index))

    def test_cases_share_the_question_template_and_option_count(self) -> None:
        for (kind, interface), groups in self.built.items():
            templates = collections.Counter()
            for first, second in groups:
                self.assertEqual(first.variant, second.variant, (kind, interface))
                self.assertEqual(
                    len(first.choices), len(second.choices), (kind, interface)
                )
                templates[first.variant.rsplit("/", 1)[1]] += 1
            count = len(f2_policy.DOMAINS[kind].questions[interface])
            self.assertEqual(len(templates), count, (kind, interface, templates))

    def test_naive_heuristics_stay_within_limits(self) -> None:
        tallies: dict[tuple[str, str], list[bool]] = collections.defaultdict(list)
        for (kind, interface), groups in self.built.items():
            for items in groups:
                for item in items:
                    refs = (
                        item.option_refs
                        if interface == "choice"
                        else item.meta["heuristics"]
                    )
                    for name in f2_policy.HEURISTICS:
                        tallies[(interface, name)].append(refs[name] == item.gold)
        for (interface, name), hits in sorted(tallies.items()):
            rate = sum(hits) / len(hits)
            with self.subTest(interface=interface, heuristic=name, rate=round(rate, 3)):
                self.assertGreaterEqual(rate, HEURISTIC_FLOOR)
                self.assertLessEqual(rate, HEURISTIC_LIMITS[interface])
                if name == "first_match" and interface in FIRST_MATCH_AIM:
                    self.assertLessEqual(rate, FIRST_MATCH_AIM[interface])


def _packet(scheme: str, interface: str = "noul") -> Packet:
    """A hand-built packet: base 100; amendment A1 sets 150 from 1 June 2025; A2 replaces A1 with 200 from
    1 January 2026; exception X1 (300) needs at least 5 nights; temporary T1 (120) runs 1 March-30 April 2025;
    annex N1 (80) covers the north region."""
    provisions = (
        Provision("base", "main", value=100, start=date(2025, 1, 1)),
        Provision(
            "A1",
            "main",
            value=150,
            start=date(2025, 6, 1),
            number=1,
            op="set",
            adopted=date(2025, 5, 1),
        ),
        Provision(
            "A2",
            "main",
            value=200,
            start=date(2026, 1, 1),
            number=2,
            op="replace",
            target="A1",
            adopted=date(2025, 11, 1),
        ),
        Provision("X1", "exception", value=300, conds=(Cond("nights", "min", 5),)),
        Provision(
            "T1", "temporary", value=120, start=date(2025, 3, 1), end=date(2025, 4, 30)
        ),
        Provision("N1", "annex", value=80, scope="north"),
    )
    return Packet(
        domain="travel_expense",
        interface=interface,
        scheme=scheme,
        governs="a",
        gov_via_term=False,
        layout="numbered",
        provisions=provisions,
        order=tuple(p.pid for p in provisions),
        base_start=date(2025, 1, 1),
        horizon=date(2026, 12, 31),
    )


def _case(nights: int, scope: str, when: date, q: object = None) -> Case:
    return Case(
        attrs={"nights": nights},
        scope=scope,
        dates={"a": when, "b": date(2025, 1, 15)},
        q=q,
    )


class F2EngineTest(unittest.TestCase):
    def test_main_text_follows_amendment_history(self) -> None:
        packet = _packet("S1")
        self.assertEqual(f2_policy.main_text(packet, date(2025, 3, 15)).pid, "base")
        self.assertEqual(f2_policy.main_text(packet, date(2025, 7, 15)).pid, "A1")
        self.assertEqual(f2_policy.main_text(packet, date(2026, 2, 1)).pid, "A2")
        a1, a2 = packet.by_pid["A1"], packet.by_pid["A2"]
        self.assertEqual(
            f2_policy.amendment_status(packet, a2, date(2025, 7, 15), a1), "not_yet"
        )
        self.assertEqual(
            f2_policy.amendment_status(packet, a1, date(2026, 2, 1), a2), "withdrawn"
        )
        self.assertEqual(
            f2_policy.temporary_status(packet.by_pid["T1"], date(2025, 7, 15)),
            "expired",
        )

    def test_precedence_schemes_by_hand(self) -> None:
        cases = (
            # nights, scope, governing date -> decider under S1, S2, S3
            (3, "south", date(2025, 7, 15), ("A1", "A1", "A1")),
            (6, "south", date(2025, 7, 15), ("X1", "X1", "X1")),
            (6, "north", date(2025, 3, 15), ("X1", "N1", "T1")),
            (3, "north", date(2025, 3, 15), ("T1", "N1", "T1")),
            (3, "north", date(2026, 2, 1), ("N1", "N1", "N1")),
            (3, "south", date(2026, 2, 1), ("A2", "A2", "A2")),
        )
        for nights, scope, when, deciders in cases:
            for scheme, expected in zip(("S1", "S2", "S3"), deciders):
                with self.subTest(nights=nights, scope=scope, when=when, scheme=scheme):
                    packet = _packet(scheme, "choice")
                    case = _case(nights, scope, when)
                    decider = f2_policy.resolve(packet, case).decider
                    self.assertEqual(decider.pid, expected)
                    facts = f2_policy._facts(packet, case, "Test Person")
                    facts["options"] = [80, 100, 120, 150, 200, 300]
                    self.assertEqual(
                        facts["options"][f2_policy.recheck(facts)], decider.value
                    )

    def test_noul_answer_and_heuristics_by_hand(self) -> None:
        packet = _packet("S1")
        case = _case(3, "south", date(2026, 2, 1), q=180)
        self.assertEqual(f2_policy.resolve(packet, case).decider.value, 200)
        self.assertEqual(
            f2_policy.recheck(f2_policy._facts(packet, case, "Test Person")), 1
        )
        answer = f2_policy.answer_fn("noul", 180)
        self.assertEqual((answer(200), answer(180), answer(150)), (1, 1, 0))
        heuristics = f2_policy.heuristic_provisions(packet, case)
        self.assertEqual(
            {name: p.pid for name, p in heuristics.items()},
            {"base_only": "base", "latest_amendment": "A2", "first_match": "base"},
        )

    def test_hazard_analysis_by_hand(self) -> None:
        # A1 (150) is in force; A2 (200) is not yet; X1 (300) would flip the answer but needs 5 nights;
        # T1 has expired and N1 is out of scope, and neither would change the answer for q = 180.
        packet = _packet("S1")
        case = _case(3, "south", date(2025, 7, 15), q=180)
        answer = f2_policy.answer_fn("noul", 180)
        self.assertEqual(answer(f2_policy.resolve(packet, case).decider.value), 0)
        labels, subtype = f2_policy.analyse(packet, case, answer)
        self.assertEqual(
            labels,
            (
                "amendment_in_force",
                "amendment_not_yet_in_force",
                "exception_condition_unmet",
            ),
        )
        self.assertEqual(subtype, "amendment_not_yet_in_force")

    def test_first_match_target_is_met_or_dropped(self) -> None:
        packet = _packet("S1", "choice")
        keys = [date(2025, 3, 1), date(2025, 5, 1), date(2025, 6, 1), date(2026, 1, 1)]
        domain = f2_policy.DOMAINS["travel_expense"]
        designs = f2_policy.enumerate_designs(
            packet, domain, f2_policy.epochs(packet, keys)
        )
        self.assertEqual({d.pattern[2] for d in designs}, {0, 1})
        targets = {
            "answers": (0, 1),
            "levels": (0, 1),
            "differ": False,
            "question": None,
        }
        for wanted in (0, 1):
            wants = {**targets, "first_match": (wanted, None)}
            chosen = f2_policy.select_design(
                random.Random(wanted), designs, "choice", wants, 0, None
            )
            self.assertEqual(chosen.pattern[2], wanted)
        only_right = [d for d in designs if d.pattern[2] == 1]
        wants_wrong = {**targets, "first_match": (0, None)}
        with self.assertRaises(core.GenerationError):
            f2_policy.select_design(
                random.Random(0), only_right, "choice", wants_wrong, 0, None
            )
        fallback = f2_policy.select_design(
            random.Random(0), only_right, "choice", wants_wrong, 0, None, False
        )
        self.assertEqual(fallback.pattern[2], 1)

    def test_redraw_values_keeps_the_structure(self) -> None:
        packet = _packet("S2", "score")
        domain = f2_policy.DOMAINS["travel_expense"]
        again = f2_policy.redraw_values(
            random.Random(1), packet, domain, {"levels": (0, 3)}
        )

        def skeleton(p: Packet) -> list[Provision]:
            return [dataclasses.replace(q, value=None) for q in p.provisions]

        self.assertEqual(skeleton(again), skeleton(packet))
        self.assertEqual(
            (again.scheme, again.layout, again.order),
            (packet.scheme, packet.layout, packet.order),
        )
        values = {p.pid: p.value for p in again.provisions}
        self.assertTrue(
            {0, 3} <= set(values.values()) <= set(range(len(domain.score.frags)))
        )
        self.assertNotEqual(values["base"], values[again.latest_setter.pid])


if __name__ == "__main__":
    unittest.main()
