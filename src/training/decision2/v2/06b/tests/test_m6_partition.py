import importlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

m6 = importlib.import_module("v2.06b.m6_partition")
common = importlib.import_module("v2.06b.common")
row = importlib.import_module("v2.06b.tests.test_mixture_s2").row


def count_many(rows):
    return [r["_tokens"] for r in rows]


class Files:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()

    def write(self, name, lines):
        path = Path(self.dir.name) / name
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        return {
            "path": str(path),
            "sha256": common.file_sha256(path),
            "rows": len(lines),
        }


def pool_rows(prefix, groups, rows_per_group, tokens, **kw):
    return [
        row(
            f"{prefix}-{g}-{i}",
            group=f"{prefix}-g{g}",
            tokens=tokens,
            state=f"{prefix} {g} {i}",
            **kw,
        )
        for g in range(groups)
        for i in range(rows_per_group)
    ]


class Fixture:
    """A base pool, a shared QKS pool, and rest pools; family y lacks pool R2 and part of Q."""

    def __init__(self, budget=500, tolerance=0.2):
        f = Files()
        base = pool_rows("b", 5, 2, 10)
        q = pool_rows("q", 30, 2, 5)
        r1 = pool_rows("r1", 60, 2, 10)
        r2 = pool_rows("r2", 40, 1, 15)
        big = row("r1-big", group="r1-g0", tokens=99, state="big")
        r1.append(big)
        pools = {"B": base, "Q": q, "R1": r1, "R2": r2}
        self.pools = {
            name: f.write(f"{name}.jsonl", rows) for name, rows in pools.items()
        }

        def entries(pool, rows):
            return [
                {
                    "id": r["id"],
                    "pool": pool,
                    "source": r["source"],
                    "task_type": r["task_type"],
                    "language": r["language"],
                    "native": 0,
                }
                for r in rows
            ]

        x = entries("B", base) + entries("Q", q) + entries("R1", r1) + entries("R2", r2)
        y = entries("B", base) + entries("Q", q[:40]) + entries("R1", r1)
        self.recipes = {
            "x": {
                "name": "x",
                **f.write("x.ids.jsonl", sorted(x, key=lambda e: e["id"])),
            },
            "y": {
                "name": "y",
                **f.write("y.ids.jsonl", sorted(y, key=lambda e: e["id"])),
            },
        }
        self.files = f
        self.budget, self.tolerance = budget, tolerance

    def spec(self, family, seed, budget=None, tolerance=None):
        other = "y" if family == "x" else "x"
        return {
            "template": "M6",
            "unit": m6.mix.UNIT,
            "tokenizer": "unused",
            "max_row_tokens": 50,
            "name": f"{family}-s{seed}",
            "family": family,
            "seed": seed,
            "seeds": 3,
            "budget_tokens": budget or self.budget,
            "budget_tolerance": self.tolerance if tolerance is None else tolerance,
            "partition_salt": "m6-partition",
            "recipe": self.recipes[family],
            "shared_with": self.recipes[other],
            "base_pool": "B",
            "qks_pools": ["Q"],
            "pools": self.pools,
        }


class PartsTest(unittest.TestCase):
    def test_equal_parts_partition_whole_groups_in_order(self):
        parts = m6.equal_parts([5] * 9, 3)
        self.assertEqual(parts, [0, 0, 0, 1, 1, 1, 2, 2, 2])
        parts = m6.equal_parts([4, 4, 4, 1, 9, 2, 2, 2], 3)
        sums = [
            sum(s for s, p in zip([4, 4, 4, 1, 9, 2, 2, 2], parts) if p == k)
            for k in range(3)
        ]
        self.assertEqual(sum(sums), 28)
        self.assertTrue(all(3 * s <= 28 for s in sums[:2]))
        self.assertEqual(parts, sorted(parts))

    def test_sequential_fill_targets_then_stop(self):
        targets = [Fraction(10)] * 3
        self.assertEqual(
            m6.sequential_fill([4] * 9, targets), [0, 0, 1, 1, 2, 2, None, None, None]
        )
        self.assertEqual(m6.sequential_fill([4, 11, 4], targets), [0, None, None])
        self.assertEqual(m6.sequential_fill([4, 7, 3], targets), [0, 1, 1])


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.inputs = m6.Inputs(count_many)

    def built(self, family):
        return [m6.build(self.inputs, self.fx.spec(family, k)) for k in (1, 2, 3)]

    def test_seeds_disjoint_shared_base_and_qks(self):
        xs, ys = self.built("x"), self.built("y")
        base = {r["id"] for r in xs[0][0] if r["id"].startswith("b-")}
        self.assertEqual(len(base), 10)
        for family in (xs, ys):
            nonbase = [
                {r["id"] for r in rows if not r["id"].startswith("b-")}
                for rows, _, _ in family
            ]
            groups = [
                {r["group_id"] for r in rows if not r["id"].startswith("b-")}
                for rows, _, _ in family
            ]
            self.assertTrue(m6.disjoint(nonbase))
            self.assertTrue(m6.disjoint(groups))
            for rows, _, _ in family:
                self.assertEqual(
                    {r["id"] for r in rows if r["id"].startswith("b-")}, base
                )
        for k in range(3):
            qx = {r["id"] for r in xs[k][0] if r["id"].startswith("q-")}
            qy = {r["id"] for r in ys[k][0] if r["id"].startswith("q-")}
            self.assertEqual(qx, qy)
            self.assertTrue(qx)
        union = set().union(
            *({r["id"] for r in rows if r["id"].startswith("q-")} for rows, _, _ in xs)
        )
        self.assertEqual(len(union), 40)
        report = xs[0][2]
        third = report["qks"]["Q"]["admissible_tokens"] / 3
        self.assertTrue(all(t <= third for t in report["qks"]["Q"]["part_tokens"][:2]))
        self.assertEqual(sum(report["qks"]["Q"]["part_tokens"]), 200)

    def test_proportional_targets_whole_groups_and_cap(self):
        xs = self.built("x")
        for rows, tokens, report in xs:
            self.assertLessEqual(abs(report["budget_deviation"]), 0.2)
            self.assertEqual(sum(tokens), report["train_tokens"])
            self.assertNotIn("r1-big", {r["id"] for r in rows})
            self.assertNotIn("r1-0-0", {r["id"] for r in rows})
            rest = report["rest"]
            self.assertEqual(report["dropped"]["over_cap_groups"]["R1"]["groups"], 1)
            n1, n2 = rest["R1"]["admissible_tokens"], rest["R2"]["admissible_tokens"]
            self.assertAlmostEqual(
                rest["R1"]["target_tokens"][0] / rest["R2"]["target_tokens"][0],
                n1 / n2,
                places=3,
            )
            k = report["seed"] - 1
            for pool in ("R1", "R2"):
                self.assertLessEqual(
                    rest[pool]["part_tokens"][k], rest[pool]["target_tokens"][k]
                )
            by_group = {}
            for r in rows:
                by_group.setdefault(r["group_id"], set()).add(r["id"])
            pool = {r["id"]: r for r in pool_rows("r1", 60, 2, 10)}
            for group, ids in by_group.items():
                if group.startswith("r1-"):
                    self.assertEqual(
                        ids, {i for i, r in pool.items() if r["group_id"] == group}
                    )

    def test_deterministic_and_budget_tolerance(self):
        first = [r["id"] for r in m6.build(self.inputs, self.fx.spec("x", 2))[0]]
        again = [
            r["id"] for r in m6.build(m6.Inputs(count_many), self.fx.spec("x", 2))[0]
        ]
        self.assertEqual(first, again)
        with self.assertRaises(ValueError):
            m6.build(self.inputs, self.fx.spec("x", 1, tolerance=0.0001, budget=501))
        with self.assertRaises(ValueError):
            m6.build(self.inputs, self.fx.spec("x", 1, budget=5000))

    def test_verify_reports_pass(self):
        specs = [self.fx.spec(f, k) for f in ("x", "y") for k in (1, 2, 3)]
        built = {}
        for spec in specs:
            rows, _, report = m6.build(self.inputs, spec)
            built[spec["name"]] = (rows, report)
        teacher = {
            r["id"]: {
                "id": r["id"],
                "input_sha256": r["input_sha256"],
                "teacher_probs": {"a": 0.5, "b": 0.5},
            }
            for rows, _ in built.values()
            for r in rows
            if r["id"] != "q-1-0"
        }
        result = m6.verify(specs, built, self.inputs, {"x": teacher})
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["qks_identical_across_families"].values()))
        missing = sum(
            e["teacher_missing_by_pool"].get("Q", 0)
            for n, e in result["per_seed"].items()
            if n.startswith("x")
        )
        self.assertEqual(missing, 1)
        self.assertEqual(result["families"]["y"]["recipe_rows"], 10 + 40 + 121)

    def test_join_rejects_metadata_mismatch(self):
        fx = self.fx
        path = Path(fx.recipes["x"]["path"])
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        lines[0]["language"] = "zh"
        bad = {"name": "x", **fx.files.write("bad.ids.jsonl", lines)}
        spec = {**fx.spec("x", 1), "recipe": bad}
        with self.assertRaises(ValueError):
            m6.build(m6.Inputs(count_many), spec)


class TeacherMergePriorityTest(unittest.TestCase):
    def test_restrict_and_priority(self):
        merge = importlib.import_module("v2.06b.teacher_merge")
        f = Files()
        e = lambda i, p: {
            "id": i,
            "input_sha256": "0" * 64,
            "teacher_probs": {"a": p, "b": 1 - p},
        }  # noqa: E731
        a = f.write("a.jsonl", [e("x", 0.25), e("y", 0.5)])
        b = f.write("b.jsonl", [e("x", 0.25), e("y", 0.75), e("z", 0.5), e("w", 0.5)])
        pa, pb = (Path(a["path"]), a["sha256"]), (Path(b["path"]), b["sha256"])
        with self.assertRaises(ValueError):
            merge.merge([pa, pb])
        rows, report = merge.merge([pa, pb], keep={"x", "y", "z"}, priority=True)
        self.assertEqual([r["id"] for r in rows], ["x", "y", "z"])
        self.assertEqual(rows[1]["teacher_probs"]["a"], 0.5)
        self.assertEqual(report["repeats"], {"0>1:different": 1, "0>1:identical": 1})
        self.assertEqual(report["inputs"][1]["kept"], 1)


class SoupSpecTest(unittest.TestCase):
    def test_groups_differences_and_refusals(self):
        soupspec = importlib.import_module("v2.06b.m6_soupspec")
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)

            def run(arm, done=True, **change):
                spec = {
                    "arm": arm,
                    "family": "qwen-causal",
                    "start": {
                        "revision": "r",
                        "head_seed": 1,
                        "collapse_stop": {"step": 9},
                    },
                    "data": {"parent": "p", "mixture": {"sha256": arm[:4]}},
                    "seed": arm[-2:],
                    "teacher": None,
                }
                for dotted, value in change.items():
                    node = spec
                    *parents, leaf = dotted.split("__")
                    for key in parents:
                        node = node[key]
                    node[leaf] = value
                d = runs / arm / "full"
                d.mkdir(parents=True)
                (d / "RUN.json").write_text(json.dumps({"spec": spec}))
                (d / "BEST.json").write_text(json.dumps({"state_sha256": arm}))
                (d / "COMPLETE.json").write_text(
                    json.dumps({"status": "COMPLETE" if done else "FAILED"})
                )

            for k in (1, 2, 3):
                run(f"aaaa-s{k}")
            run("bbbb-s1", teacher={"x": 1}, start__collapse_stop={"step": 7})
            run("bbbb-s2", teacher={"x": 1}, start__collapse_stop={"step": 7})
            run("bbbb-s3", done=False)
            run("cccc-s1", start__head_seed=2)
            spec, info = soupspec.soup_spec("z", ["aaaa", "bbbb"], runs)
            self.assertEqual(len(spec["ingredients"]), 5)
            self.assertEqual(info["skipped_seeds"], {"aaaa": [], "bbbb": ["bbbb-s3"]})
            self.assertEqual(
                spec["allowed_differences"],
                ["arm", "data.mixture", "seed", "start.collapse_stop", "teacher"],
            )
            self.assertTrue(spec["recipe"].endswith("/aaaa-s1.json"))
            with self.assertRaises(ValueError):
                soupspec.soup_spec("z", ["aaaa-s1", "cccc-s1"], runs)
            with self.assertRaises(ValueError):
                soupspec.soup_spec("z", ["aaaa-s1", "bbbb-s3"], runs)


if __name__ == "__main__":
    unittest.main()
