from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.data.m2.common import sha
from v2.data.m3 import xl, xl_r2


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _r1_dir(
    root: Path, recipes: dict[str, list[dict]], extra: dict | None = None
) -> tuple[Path, str]:
    r1 = root / "r1"
    r1.mkdir()
    manifest = {"recipes": {}, **(extra or {})}
    for name in xl_r2.R1_RECIPES:
        path = _jsonl(r1 / f"{name}.ids.jsonl", recipes.get(name, []))
        manifest["recipes"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    raw = json.dumps(manifest).encode()
    (r1 / "mx-xl.manifest.json").write_bytes(raw)
    return r1, hashlib.sha256(raw).hexdigest()


def _m(i, group, pool="H1", source="s1", language="en", native=100, cap=None):
    return {
        "id": i,
        "group": group,
        "pool": pool,
        "source": source,
        "cap": cap or source,
        "task_type": "noul",
        "language": language,
        "levels": 0,
        "native": native,
    }


def _order(variant: str, pool: str, groups: list[str]) -> list[str]:
    return sorted(groups, key=lambda g: sha(f"mx-xl-r2:{variant}:{pool}:{g}"))


class RowsTest(unittest.TestCase):
    def test_resolves_the_union_with_pool_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_a = root / "a.jsonl"
            lines = [
                '{"id": "a1", "group_id": "g1", "state": "x"}\n',
                '{"state": "y",   "group_id": "g1", "id": "a2"}\n',
                '{"id": "a3", "group_id": "g2", "state": "z"}\n',
            ]
            pool_a.write_text("".join(lines), encoding="utf-8")
            _jsonl(root / "b.jsonl", [{"id": "b1", "group_id": "g1", "state": "w"}])
            specs = {
                "A": {"rows": [str(pool_a)]},
                "V1:B": {"rows": [str(root / "b.jsonl")]},
            }
            (root / "pools.json").write_text(json.dumps(specs))
            ids = lambda *pairs: [  # noqa: E731
                {"id": i, "pool": p, "native": 5} for i, p in pairs
            ]
            r1, digest = _r1_dir(
                root,
                {
                    "mx-xl-full": ids(("a2", "A")),
                    "cx-xl-v2v1-short": ids(("a1", "A"), ("b1", "V1:B")),
                },
            )
            argv = [
                "rows",
                "--pools",
                str(root / "pools.json"),
                "--r1-dir",
                str(r1),
                "--r1-manifest-sha256",
                digest,
                "--out-dir",
                str(root / "out"),
            ]
            xl_r2.main(argv)
            out = root / "out"
            self.assertEqual(
                (out / "rows" / "A.jsonl").read_text(), lines[0] + lines[1]
            )
            self.assertTrue((out / "rows" / "V1-B.jsonl").is_file())
            index = [
                json.loads(line)
                for line in (out / "index.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [(r["pool"], r["id"], r["group"]) for r in index],
                [("A", "a1", "g1"), ("A", "a2", "g1"), ("V1:B", "b1", "g1")],
            )
            receipt = json.loads((out / "rows.receipt.json").read_text())
            self.assertEqual(receipt["union"]["rows"], 3)
            self.assertEqual(receipt["union"]["group_ids_in_several_pools"], 1)
            with self.assertRaises(ValueError):
                xl_r2.main(argv[:-3] + ["0" * 64, "--out-dir", str(root / "o2")])

    def test_an_id_missing_from_its_pool_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _jsonl(root / "a.jsonl", [{"id": "a1", "group_id": "g1", "state": "x"}])
            specs = {"A": {"rows": [str(root / "a.jsonl")]}}
            recipes = {"mx-xl-full": [{"id": "a9", "pool": "A", "native": 1}]}
            with self.assertRaises(ValueError):
                xl_r2.resolve_rows(specs, recipes, root / "out")


class SelectionTest(unittest.TestCase):
    def _r1(self) -> dict[str, list[dict]]:
        full = [
            _m("a1", "ga", pool="A0s-strict", source="anchor"),
            _m("f1", "gf", native=50),
            _m("f2", "gf", native=50),
            _m("k1", "gk", native=100),
            _m("k2", "gk2", pool="G2", source="gen", cap="gen|fam", language="ja"),
        ]
        return {
            "mx-xl-full": full,
            "mx-xl-short": full[:1] + full[3:],
            "cx-xl-a7v1-full": [full[0], full[1], full[2]],
            "cx-xl-a7v1-short": [full[0]],
            "cx-xl-v2v1-full": [full[0], full[4]],
            "cx-xl-v2v1-short": [full[0], full[4]],
        }

    def test_r2_is_r1_minus_flagged_plus_gap_and_controls(self) -> None:
        r1 = self._r1()
        gap = {
            "H7": {"h1": [_m("h1a", "h1", pool="H7", source="hover", native=40)]},
            "H8": {
                "j1": [
                    _m("j1a", "j1", pool="H8", source="jcqa", language="ja", native=30)
                ]
            },
        }
        with mock.patch.object(xl, "SOURCE_CAP", 1.0), mock.patch.object(
            xl, "ENGLISH_CAP", 1.0
        ):
            recipes, selection = xl_r2.build_recipes(r1, {"gf"}, gap)
        full = {m["id"] for m in recipes["mx-xl-full-r2"]}
        self.assertEqual(full, {"a1", "k1", "k2", "h1a", "j1a"})
        self.assertEqual(
            {m["id"] for m in recipes["cx-xl-r2-nogap-full"]}, {"a1", "k1", "k2"}
        )
        self.assertEqual({m["id"] for m in recipes["cx-xl-r2-a7v1-full"]}, {"a1"})
        self.assertEqual(
            {m["id"] for m in recipes["cx-xl-r2-v2v1-short"]}, {"a1", "k2"}
        )
        self.assertEqual(selection["full"]["total_with_candidates"], 370)
        self.assertEqual(
            {m["id"] for m in recipes["mx-xl-short-r2"]},
            {"a1", "k1", "k2", "h1a", "j1a"},
        )
        with self.assertRaises(ValueError):
            xl_r2.build_recipes(r1, {"h1"}, gap)

    def test_order_and_target(self) -> None:
        groups = {
            f"g{i}": [_m(f"r{i}", f"g{i}", pool="H7", native=10)] for i in range(8)
        }
        for variant in ("full", "short"):
            self.assertEqual(
                xl_r2.gap_candidates(groups, variant, "H7", 30),
                _order(variant, "H7", list(groups))[:3],
            )

    def test_short_rule_takes_only_groups_within_the_limit(self) -> None:
        groups = {
            "ok": [_m("a", "ok", pool="H8", native=1024)],
            "long": [
                _m("b", "long", pool="H8", native=10),
                _m("c", "long", pool="H8", native=1025),
            ],
        }
        self.assertEqual(xl_r2.gap_candidates(groups, "short", "H8", 1e9), ["ok"])
        self.assertEqual(
            sorted(xl_r2.gap_candidates(groups, "full", "H8", 1e9)), ["long", "ok"]
        )

    def test_source_cap_skips_a_group_and_keeps_later_ones(self) -> None:
        base = [_m(f"b{i}", f"b{i}", source=f"s{i}", language="xx") for i in range(20)]
        big = [_m("x1", "gx", pool="H8", source="s0", language="ja", native=200)]
        small = [_m("y1", "gy", pool="H8", source="new", language="ja", native=10)]
        rows, report = xl_r2.select_gap(
            base, {"H7": {}, "H8": {"gx": big, "gy": small}}, "full"
        )
        self.assertEqual([m["id"] for m in rows], ["y1"])
        self.assertEqual(report["total_with_candidates"], 2210)
        self.assertEqual(report["pools"]["H8"]["groups_skipped"], {"source_cap:s0": 1})

    def test_english_cap_uses_the_r2_total(self) -> None:
        base = [_m(f"b{i}", f"b{i}", source=f"s{i}", language="en") for i in range(4)]
        base += [_m(f"c{i}", f"c{i}", source=f"t{i}", language="ja") for i in range(2)]
        en = {"e": [_m("e1", "e", pool="H7", source="hover", native=100)]}
        ja = {"j": [_m("j1", "j", pool="H8", source="jcqa", language="ja", native=90)]}
        with mock.patch.object(xl, "SOURCE_CAP", 1.0):
            rows, report = xl_r2.select_gap(base, {"H7": en, "H8": ja}, "full")
            self.assertEqual({m["id"] for m in rows}, {"j1"})
            self.assertEqual(
                report["pools"]["H7"]["groups_skipped"], {"english_cap": 1}
            )
            more_ja = {
                "j": ja["j"],
                "k": [
                    _m("k1", "k", pool="H8", source="jc2", language="ja", native=200)
                ],
            }
            rows, _ = xl_r2.select_gap(base, {"H7": en, "H8": more_ja}, "full")
        self.assertEqual({m["id"] for m in rows}, {"e1", "j1", "k1"})


class RescreenTest(unittest.TestCase):
    def test_flags_group_ids_from_either_pass_in_every_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = {
                "A": [{"id": "a1", "group_id": "g1"}, {"id": "a2", "group_id": "g2"}],
                "V1:B": [
                    {"id": "b1", "group_id": "g1"},
                    {"id": "b2", "group_id": "g3"},
                ],
            }
            specs = {}
            for pool, data in rows.items():
                path = _jsonl(root / f"{xl_r2.pool_file(pool)}.src.jsonl", data)
                specs[pool] = {"rows": [str(path)]}
            meta = lambda i, p: {  # noqa: E731
                "id": i,
                "pool": p,
                "source": "src",
                "native": 10,
            }
            recipes = {name: [] for name in xl_r2.R1_RECIPES}
            recipes["mx-xl-full"] = [
                meta("a1", "A"),
                meta("a2", "A"),
                meta("b1", "V1:B"),
                meta("b2", "V1:B"),
            ]
            out = root / "rescreen"
            out.mkdir()
            receipt = xl_r2.resolve_rows(specs, recipes, out)
            (out / "rows.receipt.json").write_text(json.dumps(receipt))
            for scan in ("scan", "scan-union"):
                (out / scan).mkdir()
            inv = "f" * 64

            def write(path: Path, files: list[str], groups: dict) -> None:
                private = {
                    "protected": {"inventory_sha256": inv},
                    "candidate_files": [{"sha256": s} for s in files],
                    "groups": groups,
                }
                path.write_text(json.dumps(private))
                public = {
                    "candidates": {"rows": 2, "groups": 2},
                    "flagged": {"boilerplate": {}},
                }
                Path(str(path).replace("private", "public")).write_text(
                    json.dumps(public)
                )

            hit = {"roles": ["v1_aho_a3"], "methods": ["N"]}
            sha_a = receipt["pools"]["A"]["candidates_sha256"]
            sha_b = receipt["pools"]["V1:B"]["candidates_sha256"]
            write(out / "scan" / "A.private.json", [sha_a], {"g2": hit})
            write(out / "scan" / "V1-B.private.json", [sha_b], {})
            write(
                out / "scan-union" / "union.private.json",
                [sha_b, sha_a],
                {"g1": dict(hit, roles=["css15_native"], methods=["S"])},
            )
            private, public = xl_r2.rescreen(out, recipes, inv)
            self.assertEqual(private["flagged_group_ids"], ["g1", "g2"])
            self.assertEqual(public["flagged"]["rows"], 3)
            self.assertEqual(public["flagged"]["by_pool"]["V1:B"]["rows"], 1)
            self.assertEqual(public["flagged"]["only_union_scan"]["groups"], 2)
            self.assertEqual(public["by_recipe"]["mx-xl-full"]["rows"], 3)
            self.assertEqual(public["flagged"]["by_role"]["css15_native"]["groups"], 2)
            with self.assertRaises(ValueError):
                xl_r2.rescreen(out, recipes, "0" * 64)


def _row(i: str, group: str, source: str, language: str, family: str = "f") -> dict:
    return {
        "id": i,
        "group_id": group,
        "input_sha256": sha("in:" + i),
        "family": family,
        "source": source,
        "task_type": "noul",
        "language": language,
        "options": ["No", "Yes"],
        "state": "s",
        "instructions": "i",
    }


class ExclusionTest(unittest.TestCase):
    RECEIPT = {
        "flagged_group_ids": ["g1", "g2", "g3"],
        "hits": {
            "H1": {
                "g1": {"roles": ["v1_aho_a3"], "methods": ["L", "N"]},
                "g2": {"roles": ["a7_aho_A7q", "decision_bench_v4"], "methods": ["S"]},
            },
            "H3": {
                "g1": {"roles": ["css15_goldfree"], "methods": ["N"]},
                "g3": {"roles": ["rights_clean_cal"], "methods": ["N"]},
            },
        },
    }

    def test_one_evaluation_hit_in_any_pool_excludes_the_group_id(self) -> None:
        out, kept, rule = xl_r2.exclusion(
            self.RECEIPT, ["v1_aho_*", "a7_aho_*", "rights_clean_cal"]
        )
        self.assertEqual((out, kept), ({"g1", "g2"}, {"g3"}))
        self.assertEqual(
            rule["hit_roles_excluding"], ["css15_goldfree", "decision_bench_v4"]
        )
        self.assertEqual(
            rule["disclosed_by_role_pool"], {"rights_clean_cal": {"H3": 1}}
        )
        self.assertEqual(rule["disclosed_with_E_or_L"], 0)

    def test_no_pattern_excludes_every_flagged_group(self) -> None:
        out, kept, _ = xl_r2.exclusion(self.RECEIPT, [])
        self.assertEqual((out, kept), ({"g1", "g2", "g3"}, set()))

    def test_hits_must_match_the_flagged_ids(self) -> None:
        bad = dict(self.RECEIPT, flagged_group_ids=["g1"])
        with self.assertRaises(ValueError):
            xl_r2.exclusion(bad, [])


class BuildCheckTest(unittest.TestCase):
    def test_build_then_check_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pools = {
                "A0s-strict": ("anchor", [("a1", "ga", "anc", "en", 900)]),
                "H1": (
                    "human",
                    [
                        ("h1", "g1", "sq", "en", 500),
                        ("h2", "g1", "sq", "en", 2500),
                        ("h3", "g2", "sq", "de", 300),
                    ],
                ),
                "A7g": ("generated", [("x1", "gx", "gen", "en", 3000)]),
                "H7": ("human", [("n1", "gn", "nq", "en", 2400)]),
                "H8": (
                    "human",
                    [
                        ("t1", "gt", "ty", "ja", 700),
                        ("t2", "gt", "ty", "ja", 800),
                        ("t3", "gt", "ty", "ja", 600),
                    ],
                ),
            }
            specs, gap_args, native = {}, [], {}
            for pool, (kind, data) in pools.items():
                made = [_row(i, g, s, lang) for i, g, s, lang, _ in data]
                for row in made:
                    if row["id"] == "t3":
                        row["input_sha256"] = sha("in:t1")
                rows = _jsonl(root / f"{pool}.rows.jsonl", made)
                tokens = _jsonl(
                    root / f"{pool}.tokens.jsonl",
                    [{"id": i, "native": n} for i, *_, n in data],
                )
                native.update({i: n for i, *_, n in data})
                if pool in xl_r2.GAP_TARGETS:
                    gap_args += ["--gap", f"{pool}={rows},{tokens}"]
                else:
                    specs[pool] = {
                        "rows": [str(rows)],
                        "tokens": [str(tokens)],
                        "kind": kind,
                    }
            (root / "pools.json").write_text(json.dumps(specs))
            loaded, _ = xl.load(specs)
            members = xl_r2.members_by_id(loaded)
            pick = lambda *ids: [  # noqa: E731
                {k: members[i][k] for k in ("id", "pool", "source")}
                | {k: members[i][k] for k in ("task_type", "language", "native")}
                for i in ids
            ]
            ids = {
                "mx-xl-full": pick("a1", "h1", "h2", "h3", "x1"),
                "mx-xl-short": pick("a1", "h3"),
                "cx-xl-a7v1-full": pick("a1", "x1"),
                "cx-xl-a7v1-short": pick("a1"),
                "cx-xl-v2v1-full": pick("a1", "h1", "h2"),
                "cx-xl-v2v1-short": pick("a1", "h3"),
            }
            lux = _jsonl(root / "lux.jsonl", [{"id": "a1"}, {"id": "h3"}])
            wave = _jsonl(root / "w1.jsonl", [{"id": "h1"}])
            pending = _jsonl(root / "c1.rows.jsonl", [{"id": "x1"}])
            summaries = {}
            for name, rows in ids.items():
                s = xl.summary([members[r["id"]] for r in rows])
                s["coverage"] = {
                    "lux1": {
                        "rows_with_targets": sum(r["id"] in ("a1", "h3") for r in rows)
                    }
                }
                summaries[name] = s
            r1, digest = _r1_dir(root, ids, {"schema": "decision2-mx-xl/1"})
            manifest = json.loads((r1 / "mx-xl.manifest.json").read_text())
            for name, s in summaries.items():
                manifest["recipes"][name].update(s)
            manifest["dropped"] = {}
            raw = json.dumps(manifest).encode()
            (r1 / "mx-xl.manifest.json").write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            common = ["--r1-dir", str(r1), "--r1-manifest-sha256", digest]
            xl_r2.main(
                ["rows", "--pools", str(root / "pools.json"), "--out-dir"]
                + [str(root / "rescreen")]
                + common
            )
            rescreen = root / "rescreen.private.json"
            hit = lambda roles, methods: {  # noqa: E731
                "roles": roles,
                "methods": methods,
                "passes": ["per_pool"],
            }
            rescreen.write_text(
                json.dumps(
                    {
                        "flagged_group_ids": ["g1", "g2", "ga"],
                        "inventory_sha256": "i",
                        "hits": {
                            "A0s-strict": {
                                "ga": hit(["rights_clean_select"], ["N"]),
                            },
                            "H1": {
                                "g1": hit(["v1_aho_a3"], ["E", "N"]),
                                "g2": hit(["css15_native"], ["N"]),
                            },
                        },
                    }
                )
            )
            disclose = ["v1_aho_*", "a7_aho_*", "rights_clean_select"]
            out = root / "out"
            report = root / "check.json"
            caps = (
                mock.patch.object(xl, "SOURCE_CAP", 1.0),
                mock.patch.object(xl, "ENGLISH_CAP", 1.0),
            )
            with caps[0], caps[1]:
                xl_r2.main(
                    ["build", "--pools", str(root / "pools.json"), "--rescreen"]
                    + [str(rescreen), "--out-dir", str(out)]
                    + [a for p in disclose for a in ("--disclose-role", p)]
                    + gap_args
                    + common
                    + ["--targets", f"lux1={lux}", "--extra-targets", f"lux1={wave}"]
                    + ["--pending", f"lux1=lux-xl-c-w1={pending}"]
                )
                code = xl_r2.main(
                    ["check", "--pools", str(root / "pools.json"), "--out-dir"]
                    + [str(out), "--rescreen", str(rescreen), "--rows-dir"]
                    + [str(root / "rescreen" / "rows"), "--report", str(report)]
                    + gap_args
                    + common
                )
            result = json.loads((out / "mx-xl-r2.manifest.json").read_text())
            full = result["recipes"]["mx-xl-full-r2"]
            self.assertEqual(full["rows"], 7)
            self.assertEqual(full["gap"]["H7"]["rows"], 1)
            self.assertEqual(full["gap"]["H8"]["groups"], 1)
            self.assertEqual(full["flagged_excluded"]["rows"], 1)
            self.assertEqual(full["flagged_disclosed"]["rows"], 3)
            rule = result["rescreen"]
            self.assertEqual(rule["disclosed_roles"], disclose)
            self.assertEqual(rule["hit_roles_excluding"], ["css15_native"])
            self.assertEqual(
                (rule["excluded_group_ids"], rule["disclosed_group_ids"]), (1, 2)
            )
            self.assertEqual(rule["disclosed_with_E_or_L"], 1)
            self.assertEqual(rule["r1_union_disclosed"]["rows"], 3)
            self.assertEqual(full["coverage"]["lux1"]["rows_with_targets"], 2)
            self.assertEqual(full["coverage"]["lux1"]["pending"], {"lux-xl-c-w1": 1})
            self.assertEqual(full["coverage"]["lux1"]["without_gap_rows"], 3)
            self.assertEqual(
                full["long_evidence"]["share"], round((2500 + 3000 + 2400) / 10800, 4)
            )
            self.assertEqual(
                full["long_evidence"]["share_human_pools_only"],
                round((2500 + 2400) / (500 + 2500 + 2400 + 1500), 4),
            )
            short = result["recipes"]["mx-xl-short-r2"]
            self.assertEqual(short["gap"]["H7"]["rows"], 0)
            self.assertEqual(short["gap"]["H8"]["rows"], 2)
            self.assertEqual(result["recipes"]["cx-xl-r2-nogap-short"]["rows"], 1)
            self.assertEqual(result["dropped"], {"H8|duplicate_of:H8": 1})
            checks = json.loads(report.read_text())
            self.assertEqual(
                [k for k, v in checks["checks"].items() if not v["pass"]], []
            )
            self.assertEqual(
                checks["checks"]["gap_rows_dropped_as_in_the_build"]["by_pool_family"],
                {"H8|f": 1},
            )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
