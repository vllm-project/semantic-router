from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path
from typing import Any

from training.model.data import digest
from v2.data.hs1 import core, f3_unmet
from v2.data.hs1.f3_domains import PACKS

DOMAINS = f3_unmet.KINDS_TRAIN + f3_unmet.KINDS_OOD
INTERFACES = ("noul", "choice", "score")
PER_CELL = 10
ROOT = Path(__file__).resolve().parents[3]


def _world(key: str, domain: str, interface: str, length: str) -> dict[str, Any]:
    negative, updates, rejected = f3_unmet.build_world(key, domain, interface, length)
    positive = f3_unmet.twin_facts(negative, updates)
    items = f3_unmet.make_group(key, domain, interface, length)
    rows = core.build_rows(
        f3_unmet.FAMILY, key, items, split="train", slice_name="train", gold_target=0
    )
    return {
        "key": key,
        "negative": negative,
        "positive": positive,
        "items": items,
        "rows": rows,
        "rejected": rejected,
    }


_WORLDS: list[dict[str, Any]] = []


def worlds() -> list[dict[str, Any]]:
    if not _WORLDS:
        for domain in DOMAINS:
            for interface in INTERFACES:
                for length in f3_unmet.LENGTH_RANGES:
                    for index in range(PER_CELL):
                        key = f"hs1-test:{f3_unmet.FAMILY}:{domain}:{interface}:{length}:{index}"
                        _WORLDS.append(_world(key, domain, interface, length))
    return _WORLDS


def leaf_diffs(left: Any, right: Any, path: str = "") -> list[str]:
    if (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.keys() == right.keys()
    ):
        return [
            d
            for key in left
            for d in leaf_diffs(
                left[key], right[key], f"{path}.{key}" if path else str(key)
            )
        ]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            d
            for i, (a, b) in enumerate(zip(left, right))
            for d in leaf_diffs(a, b, f"{path}.{i}")
        ]
    return [] if left == right else [path]


def branch_near(
    branch: dict[str, Any], attr: dict[str, Any], anchor: date
) -> bool | None:
    kind, p = branch["kind"], branch["params"]
    if kind in f3_unmet.QUANTITY_KINDS:
        return abs(attr["value"] - p["threshold"]) / p["threshold"] <= f3_unmet.NEAR_REL
    if kind == "date_window":
        day = date.fromisoformat(attr["date"])
        gaps = (
            day - date.fromisoformat(p["start"]),
            day - date.fromisoformat(p["end"]),
        )
        return min(abs(g.days) for g in gaps) <= f3_unmet.NEAR_DAYS
    if kind == "before_deadline":
        return (
            abs((date.fromisoformat(attr["date"]) - date.fromisoformat(p["date"])).days)
            <= f3_unmet.NEAR_DAYS
        )
    if kind == "valid_on_date":
        days = [abs((date.fromisoformat(attr["until"]) - anchor).days)]
        if attr["start"]:
            days.append(abs((date.fromisoformat(attr["start"]) - anchor).days))
        return min(days) <= f3_unmet.NEAR_DAYS
    return None


class PackTest(unittest.TestCase):
    def test_packs_validate(self) -> None:
        for domain in DOMAINS:
            self.assertEqual(f3_unmet.validate_pack(domain), [], domain)

    def test_every_grid_threshold_used_admits_twins(self) -> None:
        for domain in DOMAINS:
            for inst in PACKS[domain]["quantities"]:
                for op in f3_unmet._QUANTITY_OPS[inst["kind"]]:
                    grid = f3_unmet.twin_grid(inst, op)
                    self.assertGreaterEqual(len(grid), 2, (domain, inst["key"], op))
                    for threshold in grid:
                        bad, good = f3_unmet._num_twin(
                            core.rng_for("grid", threshold), inst, op, threshold
                        )
                        self.assertTrue(f3_unmet._fails(op, bad, threshold))
                        self.assertFalse(f3_unmet._fails(op, good, threshold))
                        self.assertEqual(
                            f3_unmet._shape(f3_unmet.fmt_value(inst, bad, "gbp")),
                            f3_unmet._shape(f3_unmet.fmt_value(inst, good, "gbp")),
                        )


class DeterminismTest(unittest.TestCase):
    CODE = (
        "import json\n"
        "from v2.data.hs1 import core, f3_unmet\n"
        "out = []\n"
        "for i, (d, t, l) in enumerate([('scholarship', 'noul', 'short'), ('event_permit', 'choice', 'medium'),\n"
        "                               ('claim_filing', 'score', 'short')]):\n"
        "    items = f3_unmet.make_group(f'det:{i}', d, t, l)\n"
        "    out.append(core.build_rows(f3_unmet.FAMILY, f'det:{i}', items, split='train', slice_name='train',\n"
        "                               gold_target=i))\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )

    def test_same_key_same_world(self) -> None:
        for domain, interface, length in (
            ("rental_application", "noul", "medium"),
            ("travel_grant", "choice", "short"),
            ("vehicle_inspection", "score", "medium"),
        ):
            first = f3_unmet.make_group("det:same", domain, interface, length)
            second = f3_unmet.make_group("det:same", domain, interface, length)
            self.assertEqual(first, second)
            other = f3_unmet.make_group("det:other", domain, interface, length)
            self.assertNotEqual(first[0].state, other[0].state)

    def test_independent_of_hash_seed(self) -> None:
        results = []
        for seed in ("0", "4242"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
            proc = subprocess.run(
                [sys.executable, "-c", self.CODE],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            results.append(json.loads(proc.stdout))
        self.assertEqual(results[0], results[1])


class CoverageTest(unittest.TestCase):
    def test_every_cell_builds_and_rechecks(self) -> None:
        cells = collections.Counter()
        for world in worlds():
            neg, pos = world["items"]
            for item in (neg, pos):
                item.check()
                self.assertEqual(item.gold, item.recheck)
            self.assertEqual(len(world["rows"]), 2)
            self.assertEqual(world["rows"][0]["group_id"], world["rows"][1]["group_id"])
            cells[(neg.kind, neg.task_type, neg.meta["length"])] += 1
        for domain in DOMAINS:
            for interface in INTERFACES:
                for length in f3_unmet.LENGTH_RANGES:
                    self.assertGreaterEqual(
                        cells[(domain, interface, length)], PER_CELL
                    )

    def test_few_rejected_draws(self) -> None:
        rejected = [world["rejected"] for world in worlds()]
        self.assertLessEqual(sum(r > 3 for r in rejected) / len(rejected), 0.02)
        self.assertLess(max(rejected), f3_unmet.MAX_ATTEMPTS // 2)

    def test_length_ranges(self) -> None:
        for world in worlds():
            neg, pos = world["items"]
            low, high = f3_unmet.LENGTH_RANGES[neg.meta["length"]]
            self.assertTrue(
                low <= len(neg.state) <= high, (world["key"], len(neg.state))
            )
            self.assertEqual(len(neg.state), len(pos.state), world["key"])

    def test_design_counts(self) -> None:
        for world in worlds():
            facts = world["negative"]
            self.assertTrue(3 <= len(facts["requirements"]) <= 6)
            self.assertLessEqual(
                sum(len(r["branches"]) > 1 for r in facts["requirements"]), 2
            )
            self.assertLessEqual(len(facts["recommended"]), 3)
            self.assertEqual(
                len(f3_unmet.failing(facts, facts["star"])), facts["n_fail"]
            )
            if facts["form"] == "which_applicant":
                self.assertIn(len(facts["applicants"]), (3, 4))


class TwinTest(unittest.TestCase):
    def test_labels_flip(self) -> None:
        for world in worlds():
            neg, pos = world["items"]
            self.assertEqual(
                (neg.choices, neg.instructions), (pos.choices, pos.instructions)
            )
            form = neg.meta["form"]
            if form == "noul":
                self.assertEqual((neg.gold, pos.gold), (0, 1))
            elif form == "score":
                self.assertEqual(neg.gold, 0)
                self.assertIn(pos.gold, (1, 2, 3))
            elif form == "which_applicant":
                self.assertEqual(neg.gold, len(neg.choices) - 1)
                star = world["negative"]["applicants"][world["negative"]["star"]]
                self.assertEqual(neg.choices[pos.gold], star["name"])
            else:
                self.assertEqual(pos.gold, len(pos.choices) - 1)
                failing = f3_unmet.failing(world["negative"], 0)
                ids = [r["id"] for r in world["negative"]["requirements"]]
                self.assertEqual(neg.gold, ids.index(failing[0]))

    def test_only_designated_attributes_differ(self) -> None:
        for world in worlds():
            negative, positive = world["negative"], world["positive"]
            allowed = set(negative["designated"]) | set(negative["counterweights"])
            diffs = leaf_diffs(negative, positive)
            self.assertIn("twin", diffs)
            for path in diffs:
                if path != "twin":
                    self.assertTrue(
                        any(path == a or path.startswith(a + ".") for a in allowed),
                        (world["key"], path),
                    )
            for path in negative["designated"]:
                self.assertTrue(
                    any(d.startswith(path) for d in diffs), (world["key"], path)
                )
            self.assertEqual(len(negative["designated"]), negative["n_fail"])
            self.assertEqual(f3_unmet.failing(positive, positive["star"]), [])

    def test_lexical_balance(self) -> None:
        for world in worlds():
            neg, pos = world["items"]
            pack = PACKS[neg.kind]
            words = list(f3_unmet.NEAR_MISS_WORDS)
            words += [
                role
                for s in pack["signers"]
                for role in (s["role"], *s["near"], *s["wrong"])
            ]
            words += [c for cat in pack["categories"] for c in cat["pair"]]
            self.assertEqual(
                f3_unmet.negation_count(neg.state),
                f3_unmet.negation_count(pos.state),
                world["key"],
            )
            for word in words:
                self.assertEqual(
                    neg.state.count(word), pos.state.count(word), (world["key"], word)
                )
            self.assertEqual(
                sorted(f3_unmet.masked_tokens(neg.state)),
                sorted(f3_unmet.masked_tokens(pos.state)),
            )
            self.assertEqual(
                sorted(f3_unmet.probe_tokens(neg.state)),
                sorted(f3_unmet.probe_tokens(pos.state)),
            )
            self.assertEqual(neg.meta["near_threshold"], pos.meta["near_threshold"])

    def test_boundary_proximity_matches_other_values(self) -> None:
        counts = collections.Counter()
        for world in worlds():
            facts = world["negative"]
            star = facts["applicants"][facts["star"]]
            designated = {path.split(".")[-1] for path in facts["designated"]}
            anchor = date.fromisoformat(facts["anchor"]["date"])
            for req in facts["requirements"]:
                for branch in req["branches"]:
                    near = branch_near(branch, star["attrs"][branch["bid"]], anchor)
                    if near is not None:
                        counts[(branch["bid"] in designated, near)] += 1
        rate = {
            side: counts[(side, True)] / (counts[(side, True)] + counts[(side, False)])
            for side in (True, False)
        }
        self.assertLess(abs(rate[True] - rate[False]), 0.10, rate)

    def test_score_levels_balanced(self) -> None:
        levels = collections.Counter()
        for index in range(240):
            domain = DOMAINS[index % len(DOMAINS)]
            length = "short" if index % 10 < 7 else "medium"
            neg, pos = f3_unmet.make_group(
                f"hs1-test:score:{index}", domain, "score", length
            )
            self.assertEqual(neg.gold, 0)
            levels[pos.gold] += 1
        for level in (1, 2, 3):
            self.assertTrue(0.8 * 80 <= levels[level] <= 1.2 * 80, levels)

    def test_two_fail_share_and_forms(self) -> None:
        designs = [
            f3_unmet.design(f"hs1-test:design:{i}", "choice") for i in range(2000)
        ]
        two = sum(d["n_fail"] == 2 for d in designs) / len(designs)
        self.assertLess(abs(two - f3_unmet.TWO_FAIL_SHARE), 0.025)
        forms = collections.Counter(d["form"] for d in designs)
        self.assertLess(abs(forms["which_applicant"] / len(designs) - 0.5), 0.04)
        self.assertTrue(
            all(
                d["form"] == "which_unmet_first"
                for d in designs
                if d["n_fail"] == 2 and d["form"] != "which_applicant"
            )
        )


class SelfCheckTest(unittest.TestCase):
    def test_label_blind_probes(self) -> None:
        result = f3_unmet.self_check(600, namespace="hs1-test-selfcheck")
        self.assertEqual(result["rows"], 1200)
        self.assertEqual(result["yes_share"], 0.5)
        self.assertEqual(
            f3_unmet.self_check_problems(result, tolerance=0.07), [], result
        )

    def test_naive_bayes_detects_a_planted_cue(self) -> None:
        rows = [
            (f"g{i}", ["common", "cue"] if label else ["common"], label)
            for i in range(200)
            for label in (0, 1)
        ]
        self.assertGreater(f3_unmet.naive_bayes_cv(rows), 0.95)


RENTAL_HEAD = (
    "Linden Court tenancy criteria\n"
    "All of the conditions below are required unless marked as recommended.\n"
)
RENTAL_INTRO = "Ravi Horvath has applied to rent a two-bedroom flat at Linden Court."
QUESTION = "Does Ravi Horvath's rental application meet every required condition in the rules above?"


def rental_state(rules: list[str], case: list[str], optional: str = "") -> str:
    lines = [f"{i}. {rule}." for i, rule in enumerate(rules, 1)] + (
        [f"Optional: {optional}."] if optional else []
    )
    return RENTAL_HEAD + "\n".join(lines) + "\n\n" + " ".join([RENTAL_INTRO, *case])


class HandCheckedTest(unittest.TestCase):
    """Boundary cases per requirement kind, for the oracle and for the re-check reader."""

    CASES = (
        # kind, rule, case sentence, expected (1 = met)
        (
            "minimum",
            "A gross monthly income of at least £2,500",
            "The file lists a gross monthly income of £2,500.",
            1,
        ),
        (
            "minimum",
            "A gross monthly income of at least £2,500",
            "The file lists a gross monthly income of £2,499.",
            0,
        ),
        (
            "minimum",
            "A credit score above 700 points",
            "The credit score on record is 700 points.",
            0,
        ),
        (
            "minimum",
            "A credit score above 700 points",
            "The credit score on record is 705 points.",
            1,
        ),
        (
            "maximum",
            "A household of no more than 4 occupants",
            "4 occupants are recorded on the form.",
            1,
        ),
        (
            "maximum",
            "A household of no more than 4 occupants",
            "The file shows 5 occupants.",
            0,
        ),
        (
            "maximum",
            "An outstanding debt below £5,000",
            "The outstanding debt on record is £5,000.",
            0,
        ),
        (
            "maximum",
            "An outstanding debt below £5,000",
            "The outstanding debt on record is £4,950.",
            1,
        ),
        (
            "count_at_least",
            "At least 12 months of rental history",
            "The form records 12 months of rental history.",
            1,
        ),
        (
            "count_at_least",
            "12 or more months of rental history",
            "The form records 11 months of rental history.",
            0,
        ),
        (
            "date_window",
            "A move-in date from June 1, 2026 through June 30, 2026",
            "The requested move-in date is June 30, 2026.",
            1,
        ),
        (
            "date_window",
            "A move-in date from June 1, 2026 through June 30, 2026",
            "The requested move-in date is July 1, 2026.",
            0,
        ),
        (
            "date_window",
            "A move-in date no earlier than 2026-06-01 and no later than 2026-06-30",
            "The move-in date on the form is 2026-05-31.",
            0,
        ),
        (
            "before_deadline",
            "A completed application form submitted before May 10, 2026",
            "The completed application form was submitted on May 10, 2026.",
            0,
        ),
        (
            "before_deadline",
            "A completed application form submitted on or before May 10, 2026",
            "The completed application form was submitted on May 10, 2026.",
            1,
        ),
        (
            "before_deadline",
            "A completed application form handed in by 10 May 2026",
            "The completed application form came in on 11 May 2026.",
            0,
        ),
        (
            "document_provided",
            "A landlord reference",
            "The landlord reference is on file.",
            1,
        ),
        (
            "document_provided",
            "A landlord reference",
            "The landlord reference was uploaded with no problems.",
            1,
        ),
        (
            "document_provided",
            "A landlord reference",
            "The landlord reference has not arrived yet.",
            0,
        ),
        (
            "signer_role",
            "The holding deposit form signed by the leasing manager (the deputy leasing manager may also sign)",
            "The holding deposit form was signed by the deputy leasing manager, with the leasing manager copied.",
            1,
        ),
        (
            "signer_role",
            "The holding deposit form signed by the leasing manager",
            "The holding deposit form was signed by the acting leasing manager, with the leasing manager copied.",
            0,
        ),
        (
            "signer_role",
            "A signature from the leasing manager on the holding deposit form "
            "(an acting leasing manager or a deputy leasing manager does not count)",
            "The deputy leasing manager signed the holding deposit form after a call with the leasing manager.",
            0,
        ),
        (
            "signer_role",
            "A signature from the leasing manager or the assistant leasing manager on the holding deposit form",
            "The signature on the holding deposit form is that of the assistant leasing manager, "
            "and the concierge is named as the contact.",
            1,
        ),
        (
            "category_match",
            "Full-time employment",
            "Ravi moved from part-time employment to full-time employment in March.",
            1,
        ),
        (
            "category_match",
            "Full-time employment",
            "Ravi moved from full-time employment to part-time employment in March.",
            0,
        ),
        (
            "category_match",
            "Employment on a part-time basis",
            "Ravi worked full-time until 2026-01-05 and has worked part-time since.",
            1,
        ),
        (
            "same_party",
            "A deposit account held in the applicant's own name",
            "The deposit account is in the name of Ravi Horvath.",
            1,
        ),
        (
            "same_party",
            "A deposit account held in the applicant's own name",
            "The named holder of the deposit account is Farida Vogel.",
            0,
        ),
        (
            "valid_on_date",
            "A photo ID that is valid on the lease signing date",
            "The lease signing date is June 1, 2026. The photo ID is valid through June 1, 2026.",
            1,
        ),
        (
            "valid_on_date",
            "A photo ID that is valid on the lease signing date",
            "The lease signing date is June 1, 2026. The photo ID is valid through May 31, 2026.",
            0,
        ),
        (
            "valid_on_date",
            "A photo ID that covers the lease signing date",
            "The lease signing date is June 1, 2026. The photo ID was issued on June 2, 2026 and is valid "
            "through June 1, 2028.",
            0,
        ),
    )

    def test_recheck_reads_each_kind(self) -> None:
        checker = f3_unmet.Recheck.for_domain("rental_application")
        kinds = set()
        for kind, rule, case, expected in self.CASES:
            state = rental_state([rule], [case])
            self.assertEqual(
                checker.solve(state, QUESTION, (), "noul"), expected, (rule, case)
            )
            kinds.add(kind)
        self.assertEqual(kinds, set(f3_unmet.REQ_KINDS))

    def test_recheck_multi_condition_forms(self) -> None:
        checker = f3_unmet.Recheck.for_domain("rental_application")
        rules = [
            "A credit score above 700 points",
            "Either full-time employment or at least 12 months of rental history",
            "A completed application form submitted before May 10, 2026",
            "A landlord reference",
        ]
        case = [
            "The credit score on record is 700 points.",
            "Ravi is in part-time employment.",
            "The form records 18 months of rental history.",
            "The completed application form was submitted on May 10, 2026.",
            "The landlord reference is on file.",
            "The pet reference is still pending.",
        ]
        state = rental_state(rules, case, optional="a pet reference")
        self.assertEqual(checker.solve(state, QUESTION, (), "noul"), 0)
        labels = (
            "Credit score",
            "Employment type or rental history",
            "Application deadline",
            "Landlord reference",
            "All requirements are met",
        )
        first = "Which is the first required condition, in the order the rules list them, that is not met?"
        self.assertEqual(checker.solve(state, first, labels, "choice"), 0)
        with self.assertRaises(f3_unmet.RecheckError):
            checker.solve(state, "Which requirement is not met?", labels, "choice")
        fixed = state.replace("is 700 points.", "is 705 points.").replace(
            "on May 10, 2026", "on May 9, 2026"
        )
        self.assertEqual(checker.solve(fixed, QUESTION, (), "noul"), 1)
        self.assertEqual(checker.solve(fixed, first, labels, "choice"), 4)
        self.assertEqual(
            checker.solve(fixed, "Grade it.", f3_unmet.SCORE_LEVELS, "score"), 2
        )

    def test_oracle_boundaries(self) -> None:
        who = {"name": "Ravi Horvath"}
        anchor = "2026-06-01"

        def met(kind: str, params: dict[str, Any], attr: dict[str, Any]) -> bool:
            return f3_unmet.branch_met(
                {"kind": kind, "params": params}, attr, who, anchor
            )

        for op, value, expected in (
            (">=", 3, True),
            (">=", 2, False),
            (">", 3, False),
            (">", 4, True),
        ):
            self.assertEqual(
                met("minimum", {"op": op, "threshold": 3}, {"value": value}), expected
            )
        for op, value, expected in (
            ("<=", 5, True),
            ("<=", 6, False),
            ("<", 5, False),
            ("<", 4, True),
        ):
            self.assertEqual(
                met("maximum", {"op": op, "threshold": 5}, {"value": value}), expected
            )
        self.assertTrue(
            met("count_at_least", {"op": ">=", "threshold": 3}, {"value": 3})
        )
        self.assertFalse(
            met("count_at_least", {"op": ">=", "threshold": 3}, {"value": 2})
        )
        window = {"start": "2026-03-01", "end": "2026-03-31"}
        for day, expected in (
            ("2026-03-01", True),
            ("2026-03-31", True),
            ("2026-02-28", False),
            ("2026-04-01", False),
        ):
            self.assertEqual(met("date_window", window, {"date": day}), expected)
        for op, day, expected in (
            ("<=", "2026-05-10", True),
            ("<=", "2026-05-11", False),
            ("<", "2026-05-10", False),
            ("<", "2026-05-09", True),
        ):
            self.assertEqual(
                met("before_deadline", {"op": op, "date": "2026-05-10"}, {"date": day}),
                expected,
            )
        self.assertTrue(met("document_provided", {}, {"provided": True}))
        self.assertFalse(met("document_provided", {}, {"provided": False}))
        signer = {"role": "chair", "alts": ["vice chair"]}
        for role, expected in (
            ("chair", True),
            ("vice chair", True),
            ("acting chair", False),
            ("treasurer", False),
        ):
            self.assertEqual(met("signer_role", signer, {"signer": role}), expected)
        self.assertFalse(
            met("signer_role", {"role": "chair", "alts": []}, {"signer": "vice chair"})
        )
        self.assertTrue(
            met("category_match", {"required": "non-profit"}, {"cur": "non-profit"})
        )
        self.assertFalse(
            met("category_match", {"required": "non-profit"}, {"cur": "for-profit"})
        )
        self.assertTrue(met("same_party", {}, {"holder": "Ravi Horvath"}))
        self.assertFalse(met("same_party", {}, {"holder": "Farida Vogel"}))
        for start, until, expected in (
            (None, "2026-06-01", True),
            (None, "2026-05-31", False),
            ("2026-06-01", "2027-01-01", True),
            ("2026-06-02", "2027-01-01", False),
        ):
            self.assertEqual(
                met("valid_on_date", {}, {"start": start, "until": until}), expected
            )

    def test_generated_world_matches_hand_reading(self) -> None:
        world = next(
            w
            for w in worlds()
            if w["negative"]["form"] == "noul" and w["negative"]["n_fail"] == 1
        )
        neg, pos = world["items"]
        facts = world["negative"]
        path = facts["designated"][0]
        bid = path.split(".")[-1]
        branch = next(
            b for r in facts["requirements"] for b in r["branches"] if b["bid"] == bid
        )
        anchor = facts["anchor"]["date"]
        who = facts["applicants"][0]
        self.assertFalse(f3_unmet.branch_met(branch, who["attrs"][bid], who, anchor))
        twin = world["positive"]["applicants"][0]
        self.assertTrue(f3_unmet.branch_met(branch, twin["attrs"][bid], twin, anchor))
        self.assertEqual((neg.gold, pos.gold), (0, 1))
        self.assertNotEqual(digest(dict(neg.facts)), digest(dict(pos.facts)))


if __name__ == "__main__":
    unittest.main()
