import hashlib
import importlib
import random
import unittest
from collections import Counter

mixtures = importlib.import_module("v2.27b.build_mixtures")

M2_OUTPUT_SHA256 = "ab108699026a8154061247e01b31dcc74452637b6ced508cc55aec827c1a8c82"
M2_REPORT_SHA256 = "b83d2377e24bb03cb230bb8492f01be3fb50d6c0c99fb542e2f957c601fbac05"


def rows(prefix, groups, per_group=2, source="s", families=3, low=50, high=400):
    out, lengths = [], {}
    rng = random.Random(prefix)
    for g in range(groups):
        for r in range(per_group):
            row_id = f"{prefix}-{g}-{r}"
            out.append(
                {
                    "id": row_id,
                    "group_id": f"{prefix}-g{g}",
                    "source": source,
                    "family": f"{prefix}-fam{g % families}",
                    "task_type": ("choice", "noul", "score")[(g + r) % 3],
                    "language": "en" if g % 4 else "zh",
                    "input_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
                }
            )
            lengths[row_id] = rng.randint(low, high)
    return out, lengths


def a7_pool():
    """Four sub-arms; family a7p-fam* are small enough to be exhausted."""
    pool, lengths = [], {}
    for prefix, groups, per_group, families in (
        ("a7g", 300, 1, 4),
        ("a7o", 200, 1, 5),
        ("a7p", 30, 1, 6),
        ("a7i", 120, 2, 2),
    ):
        part, part_len = rows(
            prefix, groups, per_group, source=prefix, families=families
        )
        pool += part
        lengths.update(part_len)
    return pool, lengths


def digest(value):
    return hashlib.sha256(mixtures.canonical(value).encode()).hexdigest()


class WaterfillTest(unittest.TestCase):
    def test_small_families_give_all_and_rest_resplits(self):
        quotas = mixtures.waterfill({"a": 10, "b": 100, "c": 1000, "d": 1000}, 600)
        self.assertEqual(quotas["a"], 10)
        self.assertEqual(quotas["b"], 100)
        self.assertEqual(quotas["c"] + quotas["d"], 490)
        self.assertEqual((quotas["c"], quotas["d"]), (245, 245))

    def test_remainder_goes_to_first_names(self):
        self.assertEqual(
            mixtures.waterfill({"x": 50, "y": 50, "z": 50}, 100),
            {"x": 34, "y": 33, "z": 33},
        )

    def test_target_above_pool_takes_everything(self):
        self.assertEqual(mixtures.waterfill({"a": 5, "b": 7}, 100), {"a": 5, "b": 7})


class GroupFamilyTest(unittest.TestCase):
    def test_group_spanning_families_counts_toward_first_name(self):
        members = [
            {"family": "stage4_replay_natural_high_k"},
            {"family": "stage4_replay_clinc_train"},
        ]
        self.assertEqual(
            mixtures.group_family("g", members), "stage4_replay_clinc_train"
        )


class FamilyEqualTest(unittest.TestCase):
    def setUp(self):
        self.base, base_len = rows("base", 60)
        self.a6, a6_len = rows("a6", 80, source="t")
        self.a7, a7_len = a7_pool()
        self.lengths = {**base_len, **a6_len, **a7_len}
        self.specs = [
            "a7=base:full,A6:rho,A7:tokens=40000:family-equal",
            "score_rep=base:full,A6:rho,repeat:match=a7",
        ]

    def build(self, seed="seed", a7=None):
        return mixtures.build(
            self.base,
            {"A6": self.a6, "A7": self.a7 if a7 is None else a7},
            dict(self.lengths),
            self.specs,
            seed,
        )

    def test_quotas_whole_groups_and_prefix_rule(self):
        out, report = self.build()
        part = report["a7"]["parts"]["A7"]
        families = part["families"]
        self.assertEqual(sum(f["quota"] for f in families.values()), 40000)
        open_quotas = {f["quota"] for f in families.values() if not f["exhausted"]}
        self.assertLessEqual(max(open_quotas) - min(open_quotas), 1)
        for name, family in families.items():
            if family["exhausted"]:
                self.assertEqual(family["tokens"], family["available_tokens"])
            if family["quota"] == family["available_tokens"]:
                self.assertTrue(family["exhausted"], name)
        self.assertTrue(any(f["exhausted"] for f in families.values()))
        self.assertTrue(any(not f["exhausted"] for f in families.values()))
        largest = max(
            sum(self.lengths[r["id"]] for r in g)
            for g in mixtures.group_rows(self.a7).values()
        )
        for family in families.values():
            self.assertLessEqual(abs(family["tokens"] - family["quota"]), largest)
        self.assertLessEqual(abs(part["tokens"] - 40000), largest * len(families))
        a7_rows = [r for r in out["a7"] if r["source"].startswith("a7")]
        sizes = Counter(r["group_id"] for r in a7_rows)
        full = Counter(r["group_id"] for r in self.a7)
        self.assertTrue(all(sizes[g] == full[g] for g in sizes))
        self.assertEqual(part["rows"], len(a7_rows))
        self.assertEqual(part["tokens"], sum(self.lengths[r["id"]] for r in a7_rows))

    def test_family_quota_matches_manual_waterfill(self):
        _, report = self.build()
        families = report["a7"]["parts"]["A7"]["families"]
        totals = {f: v["available_tokens"] for f, v in families.items()}
        self.assertEqual(
            {f: v["quota"] for f, v in families.items()},
            mixtures.waterfill(totals, 40000),
        )

    def test_duplicates_of_mixture_rows_are_dropped(self):
        pool = [dict(r) for r in self.a7]
        base_inputs = [r["input_sha256"] for r in self.base]
        a6_inputs = [r["input_sha256"] for r in self.a6]
        pool[0]["input_sha256"] = base_inputs[3]
        pool[1]["input_sha256"] = a6_inputs[0]
        pool[2]["group_id"] = self.base[0]["group_id"]
        a7i = [r for r in pool if r["source"] == "a7i"]
        a7i[0]["input_sha256"] = base_inputs[7]
        _, report = self.build(a7=pool)
        part = report["a7"]["parts"]["A7"]
        treat_a6 = {
            r["input_sha256"] for r in self.build()[0]["a7"] if r["source"] == "t"
        }
        expected_groups = 3 + (a6_inputs[0] in treat_a6)
        self.assertEqual(part["dropped_duplicate_groups"], expected_groups)
        self.assertEqual(part["dropped_duplicate_rows"], expected_groups + 1)
        out, _ = self.build(a7=pool)
        inputs = [r["input_sha256"] for r in out["a7"]]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertNotIn(a7i[0]["group_id"], {r["group_id"] for r in out["a7"]})

    def test_pool_self_duplicates_fail(self):
        pool = [dict(r) for r in self.a7]
        pool[1]["input_sha256"] = pool[0]["input_sha256"]
        with self.assertRaises(ValueError):
            self.build(a7=pool)

    def test_repeat_matches_and_pass1_equals_treatment_base_a6(self):
        out, report = self.build()
        target = report["a7"]["tokens"]
        self.assertLessEqual(
            abs(report["score_rep"]["tokens"] - target),
            mixtures.REPEAT_TOLERANCE * target,
        )
        pass1 = sorted(
            (r for r in out["score_rep"] if "#r" not in r["id"]), key=lambda r: r["id"]
        )
        treat = sorted(
            (r for r in out["a7"] if not r["source"].startswith("a7")),
            key=lambda r: r["id"],
        )
        self.assertEqual(pass1, treat)
        self.assertEqual(
            report["score_rep"]["parts"]["A6"], report["a7"]["parts"]["A6"]
        )
        repeated = [r for r in out["score_rep"] if "#r" in r["id"]]
        originals = {r["id"]: r for r in pass1}
        for row in repeated:
            original_id, _, suffix = row["id"].rpartition("#r")
            self.assertTrue(suffix.isdigit())
            self.assertEqual({**row, "id": original_id}, originals[original_id])
        passes = report["score_rep"]["parts"]["repeat"]["passes"]
        self.assertEqual(passes[-1]["kind"], "prefix")
        self.assertTrue(all(p["kind"] == "full" for p in passes[:-1]))
        last = [r for r in repeated if r["id"].endswith(passes[-1]["suffix"])]
        sizes = Counter(r["group_id"] for r in last)
        full = Counter(r["group_id"] for r in pass1)
        self.assertTrue(all(sizes[g] == full[g] for g in sizes))

    def test_repeat_uses_several_full_passes(self):
        base, lengths = rows("b", 40)
        big, big_len = rows("big", 300, source="x", families=4)
        specs = [
            "big=base:full,X:tokens=100000:family-equal",
            "rep=base:full,repeat:match=big",
        ]
        out, report = mixtures.build(
            base, {"X": big}, {**lengths, **big_len}, specs, "s"
        )
        passes = report["rep"]["parts"]["repeat"]["passes"]
        self.assertGreaterEqual(len(passes), 2)
        self.assertEqual(
            [p["suffix"] for p in passes], [f"#r{i + 1}" for i in range(len(passes))]
        )
        self.assertEqual(passes[0]["rows"], len(base))
        target = report["big"]["tokens"]
        self.assertLessEqual(abs(report["rep"]["tokens"] - target), 0.005 * target)
        self.assertEqual(len(out["rep"]), len({r["id"] for r in out["rep"]}))

    def test_repeat_rejects_unknown_or_smaller_target(self):
        base, lengths = rows("b", 20)
        with self.assertRaises(ValueError):
            mixtures.build(base, {}, dict(lengths), ["r=base:full,repeat:match=x"], "s")
        with self.assertRaises(ValueError):
            mixtures.build(
                base,
                {},
                dict(lengths),
                ["x=base:full", "r=base:full,resample:rho,repeat:match=x"],
                "s",
            )

    def test_bad_terms_fail(self):
        for spec in (
            "x=base:full,repeat:full",
            "x=base:full,A7:match=y",
            "x=base:full,A7:tokens=5:equal",
            "x=base:full,A7:tokens=abc:family-equal",
            "x=base:full,repeat:match=y,A6:rho",
        ):
            with self.assertRaises(ValueError):
                mixtures.parse_mixture(spec)
        with self.assertRaises(ValueError):
            mixtures.build(
                self.base,
                {},
                dict(self.lengths),
                ["x=base:full,base:tokens=5:family-equal"],
                "s",
            )

    def test_deterministic_and_seeded(self):
        first = self.build()
        self.assertEqual(digest(first), digest(self.build()))
        other = self.build(seed="other")
        self.assertNotEqual(digest(first[0]["a7"]), digest(other[0]["a7"]))

    def test_breakdown_totals(self):
        out, report = self.build()
        for name, r in report.items():
            for field, cells in r["breakdown"].items():
                self.assertEqual(
                    sum(c["rows"] for c in cells.values()), r["rows"], field
                )
                self.assertEqual(
                    sum(c["tokens"] for c in cells.values()), r["tokens"], field
                )


class M2CompatibilityTest(unittest.TestCase):
    def test_m2_specs_unchanged(self):
        base, base_len = rows("base", 60)
        a2, a2_len = rows("a2", 70, source="v")
        a6, a6_len = rows("a6", 80, source="t")
        specs = [
            "combined=base:full,A2:rho,A6:rho",
            "control=base:full,resample:rho",
            "score=base:full,A6:rho",
            "verifiable=base:full,A2:rho",
        ]
        out, report = mixtures.build(
            base,
            {"A2": a2, "A6": a6},
            {**base_len, **a2_len, **a6_len},
            specs,
            "decision2-27b-m2",
        )
        for r in report.values():
            r.pop("breakdown")
        self.assertEqual(digest(out), M2_OUTPUT_SHA256)
        self.assertEqual(digest(report), M2_REPORT_SHA256)


class AhoSampleTest(unittest.TestCase):
    def test_family_stratified_whole_groups_at_most_count(self):
        pool, _ = a7_pool()
        kept, report = mixtures.sample_aho(pool, 150, "seed")
        self.assertLessEqual(len(kept), 150)
        self.assertEqual(report["rows"], len(kept))
        by_family = Counter(r["family"] for r in kept)
        self.assertEqual(
            dict(by_family), {f: n for f, n in report["by_family"].items() if n}
        )
        self.assertEqual(len(by_family), len({r["family"] for r in pool}))
        sizes = Counter(r["group_id"] for r in kept)
        full = Counter(r["group_id"] for r in pool)
        self.assertTrue(all(sizes[g] == full[g] for g in sizes))
        self.assertEqual([r["id"] for r in kept], sorted(r["id"] for r in kept))
        again, _ = mixtures.sample_aho(list(reversed(pool)), 150, "seed")
        self.assertEqual(kept, again)
        self.assertNotEqual(kept, mixtures.sample_aho(pool, 150, "other")[0])

    def test_small_pool_is_taken_whole(self):
        pool, _ = rows("p", 10, per_group=1)
        kept, _ = mixtures.sample_aho(pool, 800, "seed")
        self.assertEqual(len(kept), 10)


if __name__ == "__main__":
    unittest.main()
