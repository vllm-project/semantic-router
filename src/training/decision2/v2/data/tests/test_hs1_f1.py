from __future__ import annotations

import dataclasses
import random
import re
import unittest
from collections import Counter
from datetime import date
from fractions import Fraction

from v2.data.hs1 import core, f1_kinds_b, f1_quote
from v2.data.hs1.f1_base import MECHANISMS

GROUPS = 15
LENGTHS = ("short", "long")
KINDS = f1_quote.KINDS_TRAIN + f1_quote.KINDS_OOD
SCORE_KINDS = tuple(kind for kind in KINDS if "score" in f1_quote.BASES[kind])
# Choice kinds whose options are ordered values (dates, amounts, approver tiers, departure times)
RANKED_KINDS = (
    "order_sla",
    "expense_total",
    "project_timeline",
    "transit_connection",
    "subscription_bill",
)
SHORT = (600, 2000)
LONG = (3000, 7500)
SB = "subscription_bill"
# make_group redraws on any GenerationError; these messages mean an inconsistency, not a missed constraint
INCONSISTENT = re.compile(
    r"recheck|re-check|oracle|claim must|changed the answer|drifted|decisive fact|unpars|unread|slot missing"
)


def _key(kind: str, interface: str, length: str, index: int) -> str:
    return f"hs1-test:{f1_quote.FAMILY}:{kind}:{interface}:{length}:{index}"


def _fold(seed_key: str) -> int:
    return int(core.sha(seed_key), 16) % 5


def _lookup_accuracy(records: list[tuple[str, int, object]]) -> float:
    """5-fold CV accuracy of predicting the most frequent gold seen with a record's key
    (unseen keys: the training folds' majority gold)."""
    hits = 0
    for fold in range(5):
        table: dict[object, Counter] = {}
        overall: Counter = Counter()
        for seed_key, gold, key in records:
            if _fold(seed_key) != fold:
                table.setdefault(key, Counter())[gold] += 1
                overall[gold] += 1
        default = min(overall, key=lambda gold: (-overall[gold], gold))
        for seed_key, gold, key in records:
            if _fold(seed_key) == fold:
                seen = table.get(key)
                hits += (
                    min(seen, key=lambda g: (-seen[g], g)) if seen else default
                ) == gold
    return hits / len(records)


def _tolerance(records: list, chance: float) -> float:
    """The audit's 0.03 margin, widened to three standard errors on small samples."""
    return max(0.03, 3 * (chance * (1 - chance) / len(records)) ** 0.5)


def _gold_rank(kind: str, item: core.Item) -> tuple[tuple[object, int], int] | None:
    """((attribute, option count), the gold's rank among the options' ordered values)."""
    facts, count = item.facts, len(item.choices)
    if kind == "order_sla":
        values = list(facts["options"])
    elif kind == "expense_total":
        if (
            facts["form"] == "approver"
        ):  # options are the line manager, department head, finance director
            return ("tier", count), item.gold
        values = list(facts["options"])
    elif kind == "transit_connection":
        values = [departure for _, departure in facts["options"]]
    else:  # project_timeline, subscription_bill: options are listed in ascending order
        return ("value", count), item.gold
    return ("value", count), sorted(values).index(values[item.gold])


class F1FamilyTest(unittest.TestCase):
    built: dict[tuple[str, str, str], list[tuple[str, list[core.Item]]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.built = {}
        for kind in KINDS:
            for interface in f1_quote.supported(kind):
                for length in LENGTHS:
                    cls.built[(kind, interface, length)] = [
                        (key, f1_quote.make_group(key, kind, interface, length))
                        for key in (
                            _key(kind, interface, length, index)
                            for index in range(GROUPS)
                        )
                    ]

    def test_every_kind_is_registered(self) -> None:
        self.assertEqual(len(KINDS), 8)
        self.assertEqual(len(set(KINDS)), 8)
        for kind in KINDS:
            self.assertIn(kind, f1_quote.BUILDERS)
            self.assertIn("noul_verify", f1_quote.supported(kind))
            if "score" in f1_quote.BASES[kind]:
                self.assertIn(f1_quote.SCORE_LEVELS[kind], range(3, 6))
        self.assertEqual(f1_quote.BASES[SB], ("choice", "noul", "score"))
        self.assertEqual(f1_quote.SCORE_LEVELS[SB], 4)
        self.assertEqual(
            f1_quote.supported(SB), ("choice", "noul_direct", "noul_verify", "score")
        )

    def test_items_check_and_build_rows(self) -> None:
        for (kind, interface, length), cell in self.built.items():
            self.assertEqual(len(cell), GROUPS)
            for index, (key, items) in enumerate(cell):
                with self.subTest(
                    kind=kind, interface=interface, length=length, index=index
                ):
                    self.assertEqual(len(items), 2)
                    for item in items:
                        item.check()
                        self.assertEqual(item.task_type, f1_quote.TASK_TYPE[interface])
                        self.assertEqual(item.kind, kind)
                        self.assertEqual(item.gold, item.recheck)
                        self.assertEqual(
                            (item.meta["interface"], item.meta["length"]),
                            (interface, length),
                        )
                        self.assertIn(item.subtype, MECHANISMS)
                    split = "select" if kind in f1_quote.KINDS_OOD else "train"
                    rows = core.build_rows(
                        f1_quote.FAMILY,
                        key,
                        items,
                        split=split,
                        slice_name=split,
                        gold_target=index,
                    )
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(len({row["group_id"] for row in rows}), 1)
                    self.assertEqual(len({row["id"] for row in rows}), 2)
                    for row, item in zip(rows, items):
                        audit = row["audit_metadata"]
                        self.assertEqual(row["label"], audit["oracle_label"])
                        self.assertEqual(audit["oracle_label"], audit["recheck_label"])
                        self.assertEqual(row["source"], core.SOURCE)
                        self.assertTrue(
                            row["render_template"].startswith(
                                f"hs1/{f1_quote.FAMILY}/{kind}/"
                            )
                        )
                        if item.task_type == "choice":
                            options = [
                                option["description"] for option in row["options"]
                            ]
                            self.assertEqual(
                                options[row["label"]], item.choices[item.gold]
                            )
                            quoted = item.choices[item.option_refs["quoted"]]
                            self.assertEqual(
                                options[audit["option_refs"]["quoted"]], quoted
                            )
                    if items[0].task_type == "choice":
                        self.assertEqual(rows[0]["options"], rows[1]["options"])
                        self.assertEqual(
                            rows[0]["label"], index % len(rows[0]["options"])
                        )
                        self.assertEqual(rows[0]["label"], rows[1]["label"])

    def test_right_and_wrong_quotes(self) -> None:
        for (kind, interface, length), cell in self.built.items():
            for key, (right, wrong) in cell:
                with self.subTest(
                    kind=kind, interface=interface, length=length, key=key
                ):
                    self.assertTrue(right.meta["quote_correct"])
                    self.assertFalse(wrong.meta["quote_correct"])
                    self.assertEqual(right.meta["mechanism"], wrong.meta["mechanism"])
                    self.assertIn(wrong.meta["mechanism"], MECHANISMS)
                    self.assertNotEqual(right.probe_view, wrong.probe_view)
                    self.assertEqual(right.state.count(right.probe_view), 1)
                    self.assertEqual(
                        right.state.replace(right.probe_view, "<quote>"),
                        wrong.state.replace(wrong.probe_view, "<quote>"),
                    )
                    if interface == "noul_verify":
                        self.assertEqual((right.gold, wrong.gold), (1, 0))
                        self.assertEqual((right.recheck, wrong.recheck), (1, 0))
                        continue
                    self.assertEqual(right.instructions, wrong.instructions)
                    self.assertEqual(right.choices, wrong.choices)
                    self.assertEqual(right.gold, wrong.gold)
                    self.assertEqual(right.option_refs["quoted"], right.gold)
                    self.assertNotEqual(wrong.option_refs["quoted"], wrong.gold)
                    if interface == "noul_direct":
                        self.assertEqual(wrong.option_refs["quoted"], 1 - wrong.gold)
                    if interface == "score":
                        self.assertIn(
                            abs(wrong.option_refs["quoted"] - wrong.gold), (1, 2)
                        )

    def test_state_length_ranges(self) -> None:
        self.assertEqual(f1_quote.SHORT_CHARS, SHORT)
        self.assertEqual(f1_quote.LONG_CHARS, LONG)
        for (kind, interface, length), cell in self.built.items():
            low, high = SHORT if length == "short" else LONG
            for key, items in cell:
                for item in items:
                    self.assertTrue(
                        low <= len(item.state) <= high,
                        f"{key}: {len(item.state)} chars",
                    )

    def test_deterministic(self) -> None:
        for (kind, interface, length), cell in self.built.items():
            key, items = cell[0]
            again = f1_quote.make_group(key, kind, interface, length)
            self.assertEqual(
                [dataclasses.asdict(i) for i in items],
                [dataclasses.asdict(i) for i in again],
            )
            rows = core.build_rows(
                f1_quote.FAMILY,
                key,
                items,
                split="train",
                slice_name="train",
                gold_target=1,
            )
            rows_again = core.build_rows(
                f1_quote.FAMILY,
                key,
                again,
                split="train",
                slice_name="train",
                gold_target=1,
            )
            self.assertEqual(rows, rows_again)
        states = [items[0].state for cell in self.built.values() for _, items in cell]
        self.assertEqual(len(set(states)), len(states))
        world = f1_kinds_b.build_subscription_bill(
            random.Random(7), "choice", None, "short"
        )
        self.assertEqual(
            world,
            f1_kinds_b.build_subscription_bill(
                random.Random(7), "choice", None, "short"
            ),
        )

    def test_no_label_disagreement_is_redrawn(self) -> None:
        for (kind, interface, length), cell in self.built.items():
            for key, _ in cell[:5]:
                base = f1_quote.base_for(key, kind, interface)
                target = f1_quote.target_for(key, kind, base)
                for attempt in range(f1_quote.MAX_ATTEMPTS):
                    try:
                        world = f1_quote.BUILDERS[kind](
                            core.rng_for(key, "attempt", attempt), base, target, length
                        )
                        world.check()
                        f1_quote._items(
                            core.rng_for(key, "quote", attempt),
                            world,
                            interface,
                            length,
                        )
                        break
                    except core.GenerationError as exc:
                        self.assertIsNone(
                            INCONSISTENT.search(str(exc)),
                            f"{key} attempt {attempt}: {exc}",
                        )


class F1IndexTest(unittest.TestCase):
    """``make_group(..., index=i)``: i is the group's position in its (kind, interface) cell."""

    def test_index_sets_the_base_and_its_target(self) -> None:
        for kind in KINDS:
            bases = f1_quote.BASES[kind]
            for interface in f1_quote.supported(kind):
                verify = interface == "noul_verify"
                golds: dict[str, Counter] = {}
                for index in range(4 * len(bases) if verify else 8):
                    key = _key(kind, interface, "short", 500 + index)
                    right, wrong = f1_quote.make_group(
                        key, kind, interface, "short", index=index
                    )
                    base = right.meta["base"]
                    base_gold = right.meta[
                        "claim_answer"
                    ]  # the right claim answers the base question
                    counter = index // len(bases) if verify else index
                    expected = (
                        bases[index % len(bases)]
                        if verify
                        else f1_quote.INTERFACE_BASE[interface]
                    )
                    with self.subTest(kind=kind, interface=interface, index=index):
                        self.assertEqual(base, expected)
                        if base == "noul":
                            self.assertEqual(base_gold, counter % 2)
                        elif base == "score":
                            self.assertEqual(
                                base_gold, counter % f1_quote.SCORE_LEVELS[kind]
                            )
                        if not verify and base != "choice":
                            self.assertEqual(
                                (right.gold, wrong.gold), (base_gold, base_gold)
                            )
                        if verify:
                            self.assertEqual((right.gold, wrong.gold), (1, 0))
                    golds.setdefault(base, Counter())[base_gold] += 1
                # every base's targets are exactly balanced over the cell (with two bases, as for
                # directory_route, index % 2 alone would fix the Noul-verify target at 1)
                for base, seen in golds.items():
                    if base != "choice":
                        with self.subTest(kind=kind, interface=interface, base=base):
                            self.assertEqual(len(set(seen.values())), 1, seen)
                            self.assertEqual(
                                len(seen),
                                2 if base == "noul" else f1_quote.SCORE_LEVELS[kind],
                            )

    def test_without_index_the_seed_draws_base_and_target(self) -> None:
        def draw(key: str, kind: str, label: str, size: int) -> int:
            return int(core.sha(f"hs1-f1|{key}|{kind}|{label}"), 16) % size

        for kind in KINDS:
            bases = f1_quote.BASES[kind]
            for interface in f1_quote.supported(kind):
                key = _key(kind, interface, "short", 900)
                base = f1_quote.base_for(key, kind, interface)
                expected = (
                    bases[draw(key, kind, "verify-base", len(bases))]
                    if interface == "noul_verify"
                    else (f1_quote.INTERFACE_BASE[interface])
                )
                target = f1_quote.target_for(key, kind, base)
                with self.subTest(kind=kind, interface=interface):
                    self.assertEqual(base, expected)
                    if base == "noul":
                        self.assertEqual(target, draw(key, kind, "target-noul", 2))
                    elif base == "score":
                        self.assertEqual(
                            target,
                            draw(
                                key, kind, "target-score", f1_quote.SCORE_LEVELS[kind]
                            ),
                        )
                    else:
                        self.assertIsNone(target)
                    items = f1_quote.make_group(key, kind, interface, "short")
                    pinned = f1_quote.make_group(
                        key, kind, interface, "short", target=target
                    )
                    self.assertEqual(items[0].meta["base"], base)
                    self.assertEqual(
                        [dataclasses.asdict(i) for i in items],
                        [dataclasses.asdict(i) for i in pinned],
                    )

    def test_index_is_deterministic_and_validated(self) -> None:
        key = _key("order_sla", "score", "short", 0)
        first = f1_quote.make_group(key, "order_sla", "score", "short", index=5)
        self.assertEqual(
            first, f1_quote.make_group(key, "order_sla", "score", "short", index=5)
        )
        self.assertEqual(first[0].gold, 1)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                f1_quote.make_group(key, "order_sla", "score", "short", index=bad)


class F1IndependenceTest(unittest.TestCase):
    """Nothing but the evidence may tell the gold: not the Score criteria or question, and not the
    Choice gold's rank among ordered option values. Build-size figures are in the audit record;
    these are checks on a sample."""

    def test_score_criteria_are_drawn_before_the_target(self) -> None:
        # a band layout on the options is part of the plan: one seed gives one layout for every level
        builders = (
            ("expense_total", f1_quote.BUILDERS["expense_total"]),
            (SB, f1_kinds_b.build_subscription_bill),
        )
        for index in range(30):
            for kind, builder in builders:
                layouts = {
                    builder(
                        random.Random(f"hs1-test:bands:{kind}:{index}"),
                        "score",
                        target,
                        "short",
                    ).choices
                    for target in range(4)
                }
                self.assertEqual(len(layouts), 1, (kind, index, layouts))

    def test_score_options_and_question_do_not_predict_the_level(self) -> None:
        for kind in SCORE_KINDS:
            levels = f1_quote.SCORE_LEVELS[kind]
            records = []
            for index in range(160 if kind == "rubric_pick" else 320):
                key = f"hs1-test:independence:{kind}:score:{index}"
                length = "long" if (index // levels) % 4 == 3 else "short"
                item = f1_quote.make_group(key, kind, "score", length, index=index)[0]
                records.append((key, item.gold, item))
            majority = max(Counter(gold for _, gold, _ in records).values()) / len(
                records
            )
            for view, of in (
                ("options", lambda item: item.choices),
                ("question", lambda item: item.instructions),
                ("question skeleton", lambda item: core.mask(item.instructions)),
            ):
                accuracy = _lookup_accuracy(
                    [(key, gold, of(item)) for key, gold, item in records]
                )
                with self.subTest(kind=kind, view=view):
                    self.assertLessEqual(
                        accuracy, majority + _tolerance(records, majority)
                    )

    def test_choice_gold_rank_among_option_values_is_uniform(self) -> None:
        for kind in RANKED_KINDS:
            ranks, quoted = [], []
            for index in range(320):
                key = f"hs1-test:independence:{kind}:choice:{index}"
                length = "long" if index % 4 == 3 else "short"
                right, wrong = f1_quote.make_group(
                    key, kind, "choice", length, index=index
                )
                (attribute, count), rank = _gold_rank(kind, right)
                ranks.append((key, rank, (attribute, count)))
                if (
                    attribute == "tier"
                ):  # the wrong approver is as uniform as the right one
                    quoted.append(
                        (key, wrong.option_refs["quoted"], (attribute, count))
                    )
            for name, records in (("gold rank", ranks), ("quoted tier", quoted)):
                if not records:
                    continue
                chance = sum(1 / count for _, _, (_, count) in records) / len(records)
                with self.subTest(kind=kind, check=name):
                    self.assertLessEqual(
                        _lookup_accuracy(records), chance + _tolerance(records, chance)
                    )


# Hand-written evidence in the rendered templates. Period 18 March-17 April 2026 has 31 days; the switch on
# 30 March leaves 12 days on Plus and 19 on Pro. The Pro notice (from 1 March) is in force, the Plus notice
# (1 April, mid-period) is not. Plus: $24.90 x 12/31 = $9.64; Pro: $44.95 x 19/31 = $27.55; charges $37.19.
# Credits: $5.00, plus the $10 Spring offer (Pro on the invoice date); the Winter offer has ended, the $8.00
# credit is used up and the credit on AC-67890 belongs to another account. Net: $37.19 - $15.00 = $22.19.
HAND_TEXT = "\n\n".join(
    (
        "Account AC-12345, held by Ada Lovelace, is on a monthly Lumenbox cloud storage subscription billed in "
        "arrears; the current billing period runs from 18 March 2026 to 17 April 2026, and its invoice is issued on "
        "18 April 2026.",
        "Lumenbox monthly plan prices (price list dated 5 January 2026):\n- Basic: $12.40\n- Plus: $24.90\n"
        "- Pro: $39.95\n- Business: $59.50",
        "Price change notice, sent 2 February 2026: from 1 March 2026, the Pro plan costs $44.95 a month instead of "
        "$39.95.",
        "Price change notice, sent 20 March 2026: from 1 April 2026, the Plus plan costs $27.90 a month instead of "
        "$24.90.",
        "Recent activity on account AC-12345:\n"
        "- On 30 March 2026, account AC-12345 switched from the Plus plan to the Pro plan.\n"
        "- Credit on account AC-12345: $5.00 (goodwill credit), to be deducted from the next invoice.\n"
        "- The previous invoice for account AC-12345, issued on 18 March 2026, came to $30.00.",
        "The referral credit of $8.00 on account AC-12345 was already deducted from the invoice of 18 March 2026.",
        "Account AC-12345 is booked to move from the Pro plan to the Basic plan on 25 April 2026.",
        "Spring offer, for invoices issued from 1 April 2026 to 30 April 2026: accounts on the Pro plan on the "
        "invoice date get a one-off credit of $10 on that invoice; other plans do not qualify.",
        "Winter offer, for invoices issued from 1 January 2026 to 28 February 2026: accounts on the Pro plan on the "
        "invoice date get a one-off credit of $15 on that invoice; other plans do not qualify.",
        "Plan change on account AC-67890: Basic to Business, effective 2 April 2026. Credit on account AC-67890: "
        "$20.00 (service credit), to be deducted from the next invoice.",
    )
)
HAND_ACCOUNT = "AC-12345"


def _solve(facts: dict) -> tuple[int, int]:
    """Plan charges and net balance (cents) from world facts, with exact fractions."""
    start = date.fromisoformat(facts["period"]["start"])
    prices = dict(facts["list_prices"])
    for notice in sorted(facts["notices"], key=lambda n: n["effective"]):
        if date.fromisoformat(notice["effective"]) <= start:
            prices[notice["plan"]] = notice["new"]
    switch, days = facts["switch"], facts["period"]["days"]
    charges = sum(
        int(Fraction(prices[plan] * used, days) + Fraction(1, 2))
        for plan, used in (
            (switch["before"], switch["days_before"]),
            (switch["after"], switch["days_after"]),
        )
    )
    promo = (
        facts["promo"]["amount"] if switch["after"] in facts["promo"]["plans"] else 0
    )
    return charges, charges - sum(facts["credits"]) - promo


class SubscriptionBillTest(unittest.TestCase):
    def test_proration_oracle_by_hand(self) -> None:
        share = f1_kinds_b._sb_share
        self.assertEqual(share(2490, 12, 31), 964)
        self.assertEqual(share(4495, 19, 31), 2755)
        self.assertEqual(share(3995, 19, 31), 2449)
        self.assertEqual(share(1275, 1, 30), 43)
        self.assertTrue(f1_kinds_b._sb_half(1275, 1, 30))
        self.assertFalse(f1_kinds_b._sb_half(2490, 12, 31))

    def test_recheck_on_hand_written_evidence(self) -> None:
        recheck = f1_kinds_b._sb_recheck
        amounts = ("$19.13", "$22.19", "$37.19")
        self.assertEqual(recheck(HAND_TEXT, HAND_ACCOUNT, "choice", "", amounts), 1)
        without_notice = "\n\n".join(
            block for block in HAND_TEXT.split("\n\n") if "$44.95" not in block
        )
        self.assertEqual(
            recheck(without_notice, HAND_ACCOUNT, "choice", "", amounts), 0
        )
        for question, answer in (
            ("Will Ada Lovelace's next invoice come to more than $22?", 1),
            ("Will Ada Lovelace's next invoice come to $22 or less?", 0),
            ("Is the next invoice for account AC-12345 going to be above $22.50?", 0),
            ("Will Ada Lovelace be owed a refund when the next invoice is issued?", 0),
            ("Will Ada Lovelace have anything to pay on the next invoice?", 1),
        ):
            self.assertEqual(
                recheck(HAND_TEXT, HAND_ACCOUNT, "noul", question, ()), answer, question
            )
        large = HAND_TEXT.replace(
            "AC-12345: $5.00 (goodwill credit)", "AC-12345: $40.00 (goodwill credit)"
        )
        refund = "Will Ada Lovelace be owed a refund when the next invoice is issued?"
        self.assertEqual(recheck(large, HAND_ACCOUNT, "noul", refund, ()), 1)
        levels = tuple(level.format(width="$5") for level in f1_kinds_b._SB_LEVELS)
        self.assertEqual(recheck(HAND_TEXT, HAND_ACCOUNT, "score", "", levels), 0)
        for previous, level in (("$25.00", 1), ("$20.00", 2), ("$15.00", 3)):
            text = HAND_TEXT.replace("came to $30.00", f"came to {previous}")
            self.assertEqual(
                recheck(text, HAND_ACCOUNT, "score", "", levels), level, previous
            )
        with self.assertRaises(core.GenerationError):
            recheck(
                HAND_TEXT, HAND_ACCOUNT, "choice", "", ("$19.13", "$23.35", "$37.19")
            )

    def test_generated_worlds_match_an_independent_solver(self) -> None:
        for index in range(45):
            base = ("choice", "noul", "score")[index % 3]
            target = None if base == "choice" else index % (2 if base == "noul" else 4)
            length = "long" if index % 5 == 0 else "short"
            world = f1_kinds_b.build_subscription_bill(
                random.Random(f"sb-solver-{index}"), base, target, length
            )
            facts = world.facts
            with self.subTest(index=index, base=base):
                charges, net = _solve(facts)
                self.assertEqual((charges, net), (facts["charges"], facts["net"]))
                self.assertEqual(
                    facts["switch"]["days_before"] + facts["switch"]["days_after"],
                    facts["period"]["days"],
                )
                self.assertEqual(
                    facts["wrong"]["net"] > net, facts["direction"] == "over"
                )
                mechanism = facts["wrong"]["mechanism"]
                self.assertEqual(world.wrong.mechanism, mechanism)
                if mechanism == "stale_value":
                    self.assertIn(facts["notice_role"], ("A", "B"))
                if mechanism == "scope_misapplied":
                    self.assertNotEqual(*facts["promo"]["pattern"])
                if base == "choice":
                    self.assertEqual(
                        world.choices[world.gold], f"${net // 100:,}.{net % 100:02d}"
                    )
                elif base == "noul":
                    polarity, threshold = facts["polarity"], facts["threshold"]
                    expected = {
                        "above": lambda: net > threshold,
                        "at_most": lambda: net <= threshold,
                        "refund": lambda: net < 0,
                        "pay": lambda: net > 0,
                    }[polarity]()
                    self.assertEqual(world.gold, int(expected))
                else:
                    change, width = net - facts["previous_invoice"], facts["band_width"]
                    expected = (
                        0
                        if change < -width
                        else 1 if change <= 0 else 2 if change <= width else 3
                    )
                    self.assertEqual(world.gold, expected)
                bare = "\n\n".join(world.evidence)
                for block in world.distractors:
                    self.assertEqual(
                        f1_kinds_b._sb_recheck(
                            bare + "\n\n" + block,
                            facts["account"],
                            base,
                            world.question,
                            world.choices,
                        ),
                        world.gold,
                    )

    def test_long_worlds_supply_distractors_and_filler(self) -> None:
        for index in range(24):
            base = ("choice", "noul", "score")[index % 3]
            target = None if base == "choice" else index % (2 if base == "noul" else 4)
            world = f1_kinds_b.build_subscription_bill(
                random.Random(f"sb-long-{index}"), base, target, "long"
            )
            supplied = sum(len(block) + 2 for block in world.distractors + world.filler)
            self.assertGreaterEqual(supplied, 7000)

    def test_features_and_mechanisms_on_both_sides_of_the_gold(self) -> None:
        cells: Counter = Counter()
        for index in range(240):
            items = f1_quote.make_group(
                f"hs1-test:sb-noul:{index}", SB, "noul_direct", "short"
            )
            facts, gold = items[0].facts, items[0].gold
            cells[("notice decisive", facts["notice_role"] in ("A", "B"), gold)] += 1
            cells[("promo applies", facts["promo"]["applies"], gold)] += 1
            cells[("upgrade", facts["switch"]["upgrade"], gold)] += 1
            cells[("mechanism", facts["wrong"]["mechanism"], gold)] += 1
            cells[("polarity", facts["polarity"], gold)] += 1
        for feature in ("notice decisive", "promo applies", "upgrade"):
            for value in (True, False):
                for gold in (0, 1):
                    self.assertGreaterEqual(
                        cells[(feature, value, gold)], 5, (feature, value, gold)
                    )
        for gold in (0, 1):
            for mechanism in ("arithmetic_slip", "stale_value", "scope_misapplied"):
                self.assertGreaterEqual(
                    cells[("mechanism", mechanism, gold)], 15, (mechanism, gold)
                )
            for polarity in f1_kinds_b._SB_POLARITIES:
                self.assertGreaterEqual(
                    cells[("polarity", polarity, gold)], 10, (polarity, gold)
                )
        for interface in ("choice", "score"):
            directions: Counter = Counter()
            for index in range(120):
                items = f1_quote.make_group(
                    f"hs1-test:sb-{interface}:{index}", SB, interface, "short"
                )
                directions[
                    (items[0].facts["wrong"]["mechanism"], items[0].facts["direction"])
                ] += 1
            for mechanism in ("arithmetic_slip", "stale_value", "scope_misapplied"):
                for direction in ("over", "under"):
                    self.assertGreaterEqual(
                        directions[(mechanism, direction)],
                        5,
                        (interface, mechanism, direction),
                    )


if __name__ == "__main__":
    unittest.main()
