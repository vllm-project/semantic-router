import importlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

launch = importlib.import_module("v2.27b.launch")
m6_data = importlib.import_module("v2.27b.m6.m6_data")
m6_slices = importlib.import_module("v2.27b.m6.m6_slices")
ROOT = Path(__file__).resolve().parents[1]
M6 = ROOT / "m6"


def fake_sysfs(root: Path, render: str, pci: str) -> Path:
    device = root / "devices" / pci
    device.mkdir(parents=True)
    node = root / "drm" / render
    node.mkdir(parents=True)
    (node / "device").symlink_to(device)
    return root / "drm"


def noul_row(rid, family, language, label, group=None):
    return {
        "id": rid,
        "group_id": group or f"g-{rid}",
        "family": family,
        "language": language,
        "label": label,
        "task_type": "noul",
        "options": [{"key": "false"}, {"key": "true"}],
    }


def p_true(value):
    return [1.0 - value, value]


class M6LaunchTest(unittest.TestCase):
    def test_default_allocation_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sorted(launch.allowed_gpus("b")), [5, 6, 7])
            self.assertEqual(sorted(launch.allowed_gpus("a")), [2, 3, 4])
            self.assertEqual(launch.max_cap_hours(), 13.0)

    def test_m6_allocation(self):
        with mock.patch.dict(os.environ, {"DEV2_27B_ALLOC": "m6"}, clear=True):
            self.assertEqual(sorted(launch.allowed_gpus("b")), [0, 1, 5])
            self.assertEqual(sorted(launch.allowed_gpus("a")), [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(sorted(launch.allowed_gpus("c")), [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(launch.allowed_gpus("a")[2], launch.NODE_GPUS["a"][2])
            self.assertEqual(launch.max_cap_hours(), 22.0)
            with tempfile.TemporaryDirectory() as tmp:
                drm = fake_sysfs(Path(tmp), "renderD129", "0000:83:00.0")
                self.assertEqual(launch.render_node(0, drm).name, "renderD129")
                for gpu in (2, 3, 4, 6, 7):
                    with self.assertRaises(ValueError):
                        launch.render_node(gpu, drm)

    def test_unknown_allocation_refused(self):
        with mock.patch.dict(os.environ, {"DEV2_27B_ALLOC": "m5"}, clear=True):
            with self.assertRaises(ValueError):
                launch.allowed_gpus("b")

    def test_m6_node_d_map(self):
        with mock.patch.dict(
            os.environ, {"DEV2_27B_ALLOC": "m6", "DEV2_NODE": "d"}, clear=True
        ):
            self.assertEqual(launch.node_name(), "d")
            self.assertEqual(sorted(launch.allowed_gpus()), list(range(8)))
            self.assertEqual(launch.allowed_gpus()[7], ("0000:bb:00.0", "renderD185"))
            with tempfile.TemporaryDirectory() as tmp:
                drm = fake_sysfs(Path(tmp), "renderD153", "0000:9b:00.0")
                self.assertEqual(launch.render_node(3, drm).name, "renderD153")
        with mock.patch.dict(os.environ, {"DEV2_NODE": "d"}, clear=True):
            with self.assertRaises(ValueError):
                launch.node_name()

    def test_m6_node_e_map_excludes_external_gpus(self):
        with mock.patch.dict(
            os.environ, {"DEV2_27B_ALLOC": "m6", "DEV2_NODE": "e"}, clear=True
        ):
            self.assertEqual(launch.node_name(), "e")
            self.assertEqual(sorted(launch.allowed_gpus()), [0, 1, 2, 3, 6, 7])
            self.assertEqual(launch.allowed_gpus()[6], ("0000:b3:00.0", "renderD177"))
            with tempfile.TemporaryDirectory() as tmp:
                drm = fake_sysfs(Path(tmp), "renderD145", "0000:93:00.0")
                self.assertEqual(launch.render_node(2, drm).name, "renderD145")
                for gpu in (4, 5):
                    with self.assertRaises(ValueError):
                        launch.render_node(gpu, drm)

    def test_m6_node_f_map_excludes_k8s_gpus(self):
        with mock.patch.dict(
            os.environ, {"DEV2_27B_ALLOC": "m6", "DEV2_NODE": "f"}, clear=True
        ):
            self.assertEqual(sorted(launch.allowed_gpus()), [2, 3, 4, 5, 6, 7])
            with tempfile.TemporaryDirectory() as tmp:
                drm = fake_sysfs(Path(tmp), "renderD153", "0000:9b:00.0")
                self.assertEqual(launch.render_node(3, drm).name, "renderD153")
                for gpu in (0, 1):
                    with self.assertRaises(ValueError):
                        launch.render_node(gpu, drm)

    def test_read_lease_single_line_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = Path(tmp) / "owner"
            owner.write_text(
                "track=eval-ix1 status=released (IX1 follow-up complete) "
                "last_job_end_utc=2026-10-01T08:18:09Z\n"
            )
            self.assertEqual(
                launch.read_lease(owner),
                {
                    "track": "eval-ix1",
                    "status": "released (IX1 follow-up complete)",
                    "last_job_end_utc": "2026-10-01T08:18:09Z",
                },
            )
            owner.write_text("track=27b\npurpose=a b=c (kept whole)\nstatus=idle\n")
            self.assertEqual(launch.read_lease(owner)["purpose"], "a b=c (kept whole)")
            owner.write_text("purpose=27b M5 closed; reserved-idle for track 27b\n")
            self.assertEqual(
                launch.read_lease(owner),
                {"purpose": "27b M5 closed; reserved-idle for track 27b"},
            )

    def test_launch3_idle_status_free_text(self):
        launch3 = importlib.import_module("v2.27b.m4b.launch3")
        for ok in ("released", "released (IX1 complete)", "idle", "reserved-idle"):
            self.assertTrue(launch3.idle_status(ok), ok)
        for busy in ("running", "", None, "busy"):
            self.assertFalse(launch3.idle_status(busy), busy)

    def test_launch3_m6_allocations(self):
        code = (
            "import importlib, json; l = importlib.import_module('v2.27b.m4b.launch3');"
            " print(json.dumps([l.TRACK, sorted(l.ALLOWED_GPUS)]))"
        )
        decision2 = ROOT.parents[1]
        for alloc, gpus in (
            ("m6-b", [0, 1, 5]),
            ("m6-a", [2]),
            ("m6-d", list(range(8))),
        ):
            env = {
                **os.environ,
                "DEV2_27B_LAUNCH_ALLOC": alloc,
                "PYTHONPATH": str(decision2),
            }
            out = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            ).stdout
            self.assertEqual(json.loads(out), ["27b", gpus])


class M6SliceReadoutTest(unittest.TestCase):
    def test_parse_slices(self):
        kr = importlib.import_module("v2.27b.kernel_readout")
        sha = "a" * 64
        self.assertEqual(
            kr.parse_slices([f"pn1=/data/x=y.jsonl={sha}"]),
            [("pn1", Path("/data/x=y.jsonl"), sha)],
        )
        for bad in (
            [f"pn1=/x={'b' * 63}"],
            [f"p n=/x={sha}"],
            [f"pn1=/x={sha}", f"pn1=/y={sha}"],
            [],
        ):
            with self.assertRaises(SystemExit):
                kr.parse_slices(bad)

    def test_slice_rows_keeps_order_and_keys(self):
        kr = importlib.import_module("v2.27b.kernel_readout")
        rows = [noul_row("a", "pn-hop", "de", 1), noul_row("b", "pn-near", "de", 0)]
        records = [
            {"id": "a", "task_type": "noul", "label": 1, "logits": [0.0, 1.0]},
            {"id": "b", "task_type": "noul", "label": 0, "logits": [1.0, 0.0]},
        ]
        out = kr.slice_rows(records, rows)
        self.assertEqual([r["keys"] for r in out], [["false", "true"]] * 2)
        self.assertGreater(out[0]["probabilities"][1], 0.5)
        with self.assertRaises(SystemExit):
            kr.slice_rows(list(reversed(records)), rows)


class M6SlicesScoreTest(unittest.TestCase):
    def pn1_rows(self):
        rows = []
        for i in range(40):
            rows.append(noul_row(f"hop{i}", "pn-hop", "ja", 1))
            rows.append(noul_row(f"near{i}", "pn-near", "ja", 0))
            rows.append(noul_row(f"noisy{i}", "pn-near", "ko", 0))
        return rows

    def test_pn1_guard_flags_yes_bias(self):
        rows = self.pn1_rows()
        ref = {
            r["id"]: p_true(
                0.9 if r["label"] else (0.6 if r["id"] in ("near0", "near1") else 0.1)
            )
            for r in rows
        }
        biased = {
            r["id"]: p_true(
                0.9 if r["label"] else (0.6 if r["id"][-1] in "0123" else 0.1)
            )
            for r in rows
        }
        report = m6_slices.pn1_report(rows, ("cand", biased), ("ref", ref))
        self.assertGreater(report["delta"]["clean_no"], 0)
        self.assertFalse(report["pass"])
        same = m6_slices.pn1_report(rows, ("cand", ref), ("ref", ref))
        self.assertEqual(same["delta"]["clean_no"], 0)
        self.assertTrue(same["pass"])
        # noisy constructions (pn-near in ko) do not count toward the clean gold-no rate
        noisy = dict(ref, **{f"noisy{i}": p_true(0.9) for i in range(40)})
        self.assertTrue(
            m6_slices.pn1_report(rows, ("cand", noisy), ("ref", ref))["pass"]
        )

    def test_pn1_heldout_es_fr_is_reported_only(self):
        rows = self.pn1_rows()
        for i in range(20):
            rows.append(noul_row(f"eshop{i}", "pn-hop", "es", 1))
            rows.append(noul_row(f"esname{i}", "pn-name", "es", 0))
            rows.append(noul_row(f"frnear{i}", "pn-near", "fr", 0))
        ref = {r["id"]: p_true(0.9 if r["label"] else 0.1) for r in rows}
        report = m6_slices.pn1_report(rows, ("c", ref), ("r", ref))
        held = report["heldout_es_fr"]
        self.assertEqual(held["candidate"]["hop"]["n"], 20)
        # es / fr pn-near is a noisy construction: only the es name rows count as clean gold-no
        self.assertEqual(held["candidate"]["clean_no"]["n"], 20)
        self.assertEqual(held["delta"], {"hop": 0, "clean_no": 0})
        self.assertIsNotNone(held["delta_ci95"])
        biased = dict(ref, **{f"esname{i}": p_true(0.9) for i in range(10)})
        report = m6_slices.pn1_report(rows, ("c", biased), ("r", ref))
        self.assertAlmostEqual(report["heldout_es_fr"]["delta"]["clean_no"], 0.5)
        self.assertGreater(report["delta"]["clean_no"], 0)
        self.assertFalse(report["pass"])
        self.assertIsNone(
            m6_slices.pn1_report(self.pn1_rows(), ("c", ref), ("r", ref))[
                "heldout_es_fr"
            ]
        )

    def test_pn1_guard_hop_slack(self):
        rows = self.pn1_rows()
        ref = {r["id"]: p_true(0.9 if r["label"] else 0.1) for r in rows}
        drop = dict(ref, **{"hop0": p_true(0.2)})
        self.assertTrue(
            m6_slices.pn1_report(rows, ("c", drop), ("r", ref))["pass"]
        )  # hop -0.025
        drop = dict(ref, **{f"hop{i}": p_true(0.2) for i in range(2)})
        self.assertFalse(
            m6_slices.pn1_report(rows, ("c", drop), ("r", ref))["pass"]
        )  # hop -0.05

    def breadth_rows(self):
        rows = []
        for fam, n in (("args", 120), ("sms", 60), ("snips_rel", 12)):
            for i in range(n):
                rows.append(
                    {
                        "id": f"{fam}{i}",
                        "group_id": f"{fam}-g{i // 2}",
                        "family": fam,
                        "label": i % 2,
                        "task_type": "choice",
                        "options": [{"key": "a"}, {"key": "b"}],
                    }
                )
        return rows

    def test_breadth_gate(self):
        rows = self.breadth_rows()
        right = {r["id"]: ([0.2, 0.8] if r["label"] else [0.8, 0.2]) for r in rows}
        half = {
            r["id"]: ([0.2, 0.8] if int(r["id"][-1]) % 4 == 0 else [0.8, 0.2])
            for r in rows
        }
        report = m6_slices.breadth_report(
            rows, ("c", right), ("r", half), ["sms"], reps=200
        )
        self.assertEqual(report["eligible_families"], ["args", "sms"])
        self.assertTrue(report["pass"])
        self.assertTrue(report["by_family"]["sms"]["in_distribution"])
        self.assertIn("snips_rel", report["by_family"])
        same = m6_slices.breadth_report(rows, ("c", right), ("r", right), [], reps=200)
        self.assertEqual(same["delta"], 0)
        self.assertFalse(same["pass"])

    def test_read_probs_checks_keys(self):
        rows = [noul_row("a", "pn-hop", "de", 1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "a", "keys": ["true", "false"], "probabilities": [0.5, 0.5]}
                )
                + "\n"
            )
            with self.assertRaises(ValueError):
                m6_slices.read_probs(path, rows)
            path.write_text(
                json.dumps(
                    {"id": "a", "keys": ["false", "true"], "probabilities": [0.3, 0.7]}
                )
                + "\n"
            )
            self.assertEqual(m6_slices.read_probs(path, rows), {"a": [0.3, 0.7]})


class M6DataTest(unittest.TestCase):
    def test_drop_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "ib.jsonl", Path(tmp) / "ibx.jsonl"
            lines = [
                json.dumps({"id": str(i), "family": f}) + "\n"
                for i, f in enumerate(["args", "w2c", "isarc", "sms"])
            ]
            src.write_text("".join(lines))
            report = m6_data.drop_families(src, ["w2c", "isarc"], out)
            self.assertEqual(out.read_text(), lines[0] + lines[3])
            self.assertEqual(
                (report["rows_in"], report["rows_dropped"], report["rows_out"]),
                (4, 2, 2),
            )
            with self.assertRaises(FileExistsError):
                m6_data.drop_families(src, ["w2c"], out)
            with self.assertRaises(ValueError):
                m6_data.drop_families(src, ["nope"], Path(tmp) / "other.jsonl")

    def test_drop_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "pn1.jsonl", Path(tmp) / "pn1h.jsonl"
            lines = [
                json.dumps(
                    {
                        "id": str(i),
                        "group_id": g,
                        "language": "ja",
                        "family": "pn-hop",
                        "label": 1,
                    }
                )
                + "\n"
                for i, g in enumerate(["g1", "g2", "g2", "g3"])
            ]
            src.write_text("".join(lines))
            g0, scan = Path(tmp) / "g0.txt", Path(tmp) / "scan.txt"
            g0.write_text("")
            scan.write_text("g2\n")
            report = m6_data.drop_groups(src, [g0, scan], out)
            self.assertEqual(out.read_text(), lines[0] + lines[3])
            self.assertEqual(
                (
                    report["groups_dropped"],
                    report["rows_in"],
                    report["rows_dropped"],
                    report["rows_out"],
                ),
                (1, 4, 2, 2),
            )
            self.assertEqual(report["rows_dropped_by_cell"], {"ja/pn-hop/1": 2})
            with self.assertRaises(FileExistsError):
                m6_data.drop_groups(src, [scan], out)
            scan.write_text("g9\n")
            with self.assertRaises(ValueError):
                m6_data.drop_groups(src, [scan], Path(tmp) / "other.jsonl")

    def test_concat_dev_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, out = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "ab"
            a.write_text(json.dumps({"id": "1"}) + "\n")
            b.write_text(json.dumps({"id": "2"}))
            report = m6_data.concat([a, b], out)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(
                out.read_text().splitlines(), ['{"id": "1"}', '{"id": "2"}']
            )
            with self.assertRaises(ValueError):
                m6_data.concat([a, a], Path(tmp) / "aa")

    def test_soup_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            members = []
            for k in (1, 2):
                m = Path(tmp) / f"s{k}"
                m.mkdir()
                (m / "decision_config.json").write_text(
                    json.dumps({"lora": {"rank": 128, "alpha": 256}})
                )
                members.append(m)
            manifest = {
                "lora": {"members": 2, "rank": 256, "alpha": 512},
                "verification": {
                    "verify_adapter_config": True,
                    "max_relative_diff": 1e-7,
                    "tolerance_relative": 1e-6,
                },
                "members": [
                    {"path": str(m), "files_sha256": {"adapter/x": "ab"}}
                    for m in members
                ],
            }
            self.assertEqual(m6_data.soup_problems(manifest, {}, members), [])
            sums = Path(tmp) / "SHA256SUMS"
            sums.write_text("cd  ./adapter/x\n")
            self.assertTrue(
                m6_data.soup_problems(manifest, {str(members[1]): str(sums)}, members)
            )
            bad = dict(manifest, lora={"members": 2, "rank": 128, "alpha": 256})
            self.assertTrue(m6_data.soup_problems(bad, {}, members))


def typed_value(t_dev, choice, noul, score):
    def cell(acc):
        return {"accuracy": acc, "answer_categories": {"a": 1, "b": 1}}

    return {
        "T_dev": t_dev,
        "typed_dev": {
            "by_type": {
                "choice": cell(choice),
                "noul": cell(noul),
                "score": cell(score),
            }
        },
    }


class M6GatesTest(unittest.TestCase):
    def setUp(self):
        self.gates = importlib.import_module("v2.27b.m6.m6_devgates")
        self.l128 = typed_value(0.8875, 1.0, 0.55, 1.0)

    def test_floors_against_l128(self):
        ok = self.gates.floors(typed_value(0.8575, 0.97, 0.52, 0.97), self.l128)
        self.assertTrue(ok["typed"]["pass"])
        self.assertTrue(ok["noul"]["pass"])
        low_t = self.gates.floors(typed_value(0.857, 1.0, 0.6, 1.0), self.l128)
        self.assertFalse(low_t["typed"]["pass"])
        low_score = self.gates.floors(typed_value(0.9, 1.0, 0.6, 0.96), self.l128)
        self.assertFalse(low_score["typed"]["pass"])
        low_noul = self.gates.floors(typed_value(0.9, 1.0, 0.519, 1.0), self.l128)
        self.assertFalse(low_noul["noul"]["pass"])

    def test_pn1_validation_rule(self):
        report = {
            "candidate": "M5-L128",
            "reference": "A20r",
            "delta": {"clean_no": 0.0188},
        }
        self.assertTrue(self.gates.pn1_validated(report))
        self.assertFalse(
            self.gates.pn1_validated(dict(report, delta={"clean_no": 0.0}))
        )
        with self.assertRaises(ValueError):
            self.gates.pn1_validated(dict(report, candidate="M6-IB"))


class M6IndexFirstTest(unittest.TestCase):
    def test_gate_integrity_and_choice(self):
        rule = importlib.import_module("v2.27b.m6.m6_index_first")

        def boot(low):
            return {
                "headline": {
                    "delta": low + 1,
                    "ci95": [low, low + 2],
                    "se": 0.5,
                    "p_le_0": 0.01,
                }
            }

        ok = {"types": {t: {"verdict": "OK"} for t in ("choice", "noul", "score")}}
        bad = {
            "types": {"choice": {"verdict": "OK"}, "score": {"verdict": "COLLAPSED"}}
        }
        public, private = rule.decide(
            {
                "M6-IB": boot(0.4),
                "M6-IB2": boot(0.9),
                "M5-L128": boot(1.5),
                "M6-IBX": boot(-0.1),
            },
            {"M6-IB": ok, "M6-IB2": ok, "M5-L128": bad, "M6-IBX": ok},
            {
                "M6-IB2": {
                    "weighted_delta_sum": 2.0,
                    "benchmarks": {
                        "HoVer": {"weighted_delta": 1.2},
                        "BPoMP": {"weighted_delta": 0.3},
                        "BANKING77": {"weighted_delta": 0.1},
                        "MMLU": {"weighted_delta": 0.4},
                    },
                }
            },
        )
        self.assertEqual(public["order"], ["M6-IB2", "M6-IB"])
        self.assertEqual(public["choice"], "M6-IB2")
        self.assertFalse(public["candidates"]["M5-L128"]["eligible"])
        self.assertFalse(public["candidates"]["M6-IBX"]["index_gate"])
        self.assertNotIn("ci95", json.dumps(public))
        t = private["candidates"]["M6-IB2"]["transfer_only"]
        self.assertEqual(t["excluded"], ["HoVer", "BPoMP"])
        self.assertEqual(t["transfer_only_weighted_delta"], 0.5)
        self.assertEqual(
            rule.transfer_only(
                "M5-L128",
                {
                    "weighted_delta_sum": 1.0,
                    "benchmarks": {"BPoMP": {"weighted_delta": 1.0}},
                },
            )["excluded"],
            [],
        )
        public, _ = rule.decide({"M6-IB": boot(0.4)}, {}, {})
        self.assertIsNone(public["choice"])
        self.assertIsNone(public["candidates"]["M6-IB"]["no_type_collapsed"])


class M6ReportTest(unittest.TestCase):
    def test_devgates_and_heldout_tables(self):
        report = importlib.import_module("v2.27b.m6.m6_report")
        gates = {
            "G1_collapse": {"flags": [], "pass": True},
            "G2_htdev2_vs_A20r": {
                "delta": 0.004,
                "ci95": [-0.01, 0.02],
                "verdict": "TIE",
                "pass": True,
            },
            "G3_typed_floor_vs_L128": {
                "T_dev": 0.88,
                "T_dev_floor": 0.8575,
                "choice_accuracy": 0.97,
                "score_accuracy": 0.99,
                "pass": True,
            },
            "G4_noul_floor_vs_L128": {
                "noul_accuracy": 0.55,
                "floor": 0.52,
                "pass": True,
            },
            "G5_pn1_guard_vs_A20r": {
                "delta": {"clean_no": 0.012, "hop": 0.0},
                "delta_ci95": {"clean_no": [0.002, 0.022], "hop": [0.0, 0.0]},
                "pass": False,
            },
            "G6_breadth_vs_A20r": {
                "B_dev": {"candidate": 0.93, "reference": 0.92},
                "delta": 0.01,
                "delta_ci95": [0.002, 0.018],
                "pass": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            dg = Path(tmp) / "DEVGATES.json"
            dg.write_text(
                json.dumps(
                    {
                        "candidates": {"M6-IB": {"gates": gates, "pass": False}},
                        "finalists": [],
                    }
                )
            )
            lines = report.devgates_table([dg])
            self.assertIn(
                "+0.0120 [+0.0020, +0.0220] / +0.0000 [+0.0000, +0.0000] (**FAIL**)",
                lines[2],
            )
            self.assertIn("| 0.9300, +0.0100 [+0.0020, +0.0180] (pass) |", lines[2])
            self.assertEqual(lines[-1], "Finalists: none.")
            rows = [noul_row(f"es{i}", "pn-name", "es", 0) for i in range(4)]
            rows += [noul_row(f"hop{i}", "pn-hop", "fr", 1) for i in range(2)]
            for name, yes in (("A20r", 0.1), ("M6-IB", 0.9)):
                d = Path(tmp) / "slices" / name / "probs"
                d.mkdir(parents=True)
                (d / "pn1.probs.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": r["id"],
                                "keys": ["false", "true"],
                                "probabilities": p_true(0.9 if r["label"] else yes),
                            }
                        )
                        + "\n"
                        for r in rows
                    )
                )
            pn1 = Path(tmp) / "pn1.jsonl"
            pn1.write_text("".join(json.dumps(r) + "\n" for r in rows))
            held = report.heldout_table(Path(tmp), pn1, ["M6-IB"])
            self.assertTrue(held[2].startswith("| M6-IB | 1.0000 / 0.0000 | +1.0000"))
            self.assertIn("(4)", held[2])

    def test_verdicts_table(self):
        report = importlib.import_module("v2.27b.m6.m6_report")
        pair = {"delta": 2.1, "ci95": [0.3, 3.9], "H_ci95": [-0.02, 0.03]}
        items = {
            "1_v3_lower_bound_vs_A20r": {"pass": True},
            "2_H_not_below_A20r": {"pass": True},
            "3_no_type_collapsed": {"pass": True},
            "4_mlx_card_eligible_not_below_A20r": {
                "status": "PENDING (node A mlx-paired)",
                "pass": None,
            },
            "5_tier_gates": {"pass": True},
            "6_no_overlap_exposure": {"pass": True},
            "7_public231_not_regression": {"delta": -1, "verdict": "OK", "pass": True},
        }
        record = {
            "finalists": {
                "M6-IB2": {
                    "v3": 74.5,
                    "T": 0.95,
                    "H": 0.59,
                    "paired": {"A20r": pair, "autojev27": pair},
                    "successor_items": items,
                    "successor_items_1_7": None,
                    "beats_autojev27": {"pass": True},
                }
            },
            "successor_items_1_7": [],
            "beats_autojev_and_successor": [],
            "choice": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "VERDICTS.json"
            path.write_text(json.dumps(record))
            lines = report.verdicts_table(path)
        self.assertIn(
            "| M6-IB2 | 74.50 (0.950 / 0.590) | +2.10 [+0.30, +3.90] (pass)", lines[2]
        )
        self.assertIn("n/a (pending)", lines[2])
        self.assertIn("-1 OK (pass)", lines[2])
        self.assertEqual(
            lines[-1],
            "Items 1–7: none; beats AutoJev-27B and items 1–7: none; choice: none.",
        )


class M5VerdictsOptionsTest(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def paired(self, low):
        return {
            "point": {"delta": {"score": 2.0, "T": 0.05, "H": 0.01}},
            "ci95": {"low": low, "high": 4.0},
            "axis_ci95": {
                "H": {"delta": {"low": -0.02, "high": 0.03}},
                "T": {"delta": {"low": 0.03, "high": 0.07}},
            },
            "models": {"left": "x", "right": "y"},
        }

    def test_train_of_and_exposure_prefix(self):
        verdicts = importlib.import_module("v2.27b.m5.m5_verdicts")
        with tempfile.TemporaryDirectory() as tmp:
            gates, run = Path(tmp) / "gates", Path(tmp) / "run"
            self.write(
                run / "REPORT.json",
                {
                    "v3": {"score": 75.0, "T": 0.95, "H": 0.59},
                    "panels": {"public231": {"correct": 203}},
                },
            )
            for key in ("A20r", "autojev27", "eikos27b", "jebadiah27b", "F1"):
                self.write(gates / "M6-IB" / f"paired-vs-{key}.json", self.paired(0.5))
            self.write(
                gates / "M6-IB" / "types.json",
                {"types": {t: {"verdict": "OK"} for t in ("choice", "noul", "score")}},
            )
            self.write(
                gates / "M6-IB" / "public231-vs-A20r.json",
                {"delta": 0, "ci95": [-5, 5], "verdict": "OK"},
            )
            self.write(
                gates / "mlx" / "M6-IB-vs-A20r.json",
                {
                    "bootstrap": {"card_macro_ci95": [-0.01, 0.01]},
                    "delta": {"card_macro": 0.0, "type_macro": 0.0},
                    "R4": {"pass": True},
                },
            )
            self.write(
                gates / "overlap" / "exposure-m6-a20ib1.json",
                {"groups": [], "methods_agree": True},
            )
            out = verdicts.finalist(
                "M6-IB", run, gates, {}, {"M6-IB": ["a20ib1"]}, "exposure-m6-"
            )
            self.assertTrue(out["successor_items"]["6_no_overlap_exposure"]["pass"])
            self.assertTrue(out["successor_items_1_7"])
            default = verdicts.finalist("M6-IB", run, gates, {})
            self.assertFalse(
                default["successor_items"]["6_no_overlap_exposure"]["pass"]
            )


class M6IndexPathTest(unittest.TestCase):
    ITEMS = (
        "1_v3_lower_bound_vs_A20r",
        "2_H_not_below_A20r",
        "3_no_type_collapsed",
        "4_mlx_card_eligible_not_below_A20r",
        "5_tier_gates",
        "6_no_overlap_exposure",
        "7_public231_not_regression",
    )

    def finalist(self, ci, fail=()):
        items = {k: {"pass": k not in fail} for k in self.ITEMS}
        items["1_v3_lower_bound_vs_A20r"]["pass"] = ci[0] > 0
        return {
            "paired": {"A20r": {"ci95": list(ci)}, "autojev27": {"ci95": [-1.0, 2.0]}},
            "successor_items": items,
            "successor_items_1_7": all(v["pass"] for v in items.values()),
            "beats_autojev27": {"pass": False},
        }

    def verdicts(self, finalists):
        passing = [n for n, f in finalists.items() if f["successor_items_1_7"]]
        return {
            "finalists": finalists,
            "successor_items_1_7": passing,
            "beats_autojev_and_successor": [],
            "choice": passing[0] if passing else None,
        }

    @staticmethod
    def boot(low):
        return {
            "headline": {
                "delta": low + 0.2,
                "ci95": [low, low + 0.4],
                "se": 0.1,
                "p_le_0": 0.01,
            }
        }

    def test_item_1_prime_and_choice(self):
        m = importlib.import_module("v2.27b.m6.m6_index_path")
        finalists = {
            "M6-IB": self.finalist((0.5, 4.0)),
            "M6-IBX": self.finalist((-1.0, 3.0)),
            "M6-IB2": self.finalist(
                (-1.5, 2.5), fail=("4_mlx_card_eligible_not_below_A20r",)
            ),
            "M6-IB2PN": self.finalist((-2.0, -0.1)),
        }
        boots = {
            "M6-IB": self.boot(-0.1),
            "M6-IBX": self.boot(0.05),
            "M6-IB2": self.boot(0.3),
            "M6-IB2PN": self.boot(0.4),
        }
        public, private = m.decide(self.verdicts(finalists), boots, {})
        f = public["finalists"]
        self.assertTrue(f["M6-IB"]["classic_items_1_7"])
        self.assertFalse(f["M6-IB"]["index_path"])
        self.assertTrue(f["M6-IBX"]["index_path"])
        self.assertFalse(f["M6-IB2"]["index_path"])  # item 4 fails
        self.assertFalse(
            f["M6-IB2PN"]["item_1_prime"]["a_v3_not_significantly_below_A20r"]
        )
        self.assertEqual(public["index_path"], ["M6-IBX"])
        self.assertEqual(
            (public["choice"], public["choice_path"]), ("M6-IB", "classic")
        )
        for field in ('"ci95"', '"se"', '"p_le_0"', '"delta"', "0.45"):
            self.assertNotIn(field, json.dumps(public))
        self.assertEqual(
            private["finalists"]["M6-IBX"]["index_delta"]["ci95"], [0.05, 0.45]
        )
        del finalists["M6-IB"], boots["M6-IB"]
        public, _ = m.decide(self.verdicts(finalists), boots, {})
        self.assertEqual((public["choice"], public["choice_path"]), ("M6-IBX", "index"))
        boots.pop("M6-IBX")
        public, _ = m.decide(self.verdicts(finalists), boots, {})
        self.assertIsNone(
            public["finalists"]["M6-IBX"]["item_1_prime"][
                "b_index_delta_significantly_positive"
            ]
        )
        self.assertIsNone(public["choice"])

    def test_transfer_only_excludes_in_distribution_and_format_matched(self):
        m = importlib.import_module("v2.27b.m6.m6_index_path")
        names = (
            "When2Call",
            "iSarcasmEval",
            "HoVer",
            "GSM8K",
            "BPoMP",
            "BANKING77",
            "ANLI",
        )
        delta = {
            "weighted_delta_sum": 1.0,
            "benchmarks": {n: {"weighted_delta": 0.1} for n in names},
        }
        out = m.transfer_only("M6-IB2", delta)
        self.assertEqual(
            out["excluded"], ["When2Call", "iSarcasmEval", "HoVer", "GSM8K", "BPoMP"]
        )
        self.assertAlmostEqual(out["transfer_only_weighted_delta"], 0.5)
        self.assertEqual(out["shared_with_A20r_weighted_delta"], {"BANKING77": 0.1})
        self.assertEqual(m.transfer_only("M6-IBX", delta)["excluded"], ["BPoMP"])

    def test_private_output_must_be_private(self):
        m = importlib.import_module("v2.27b.m6.m6_index_path")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verdicts = root / "VERDICTS.json"
            verdicts.write_text(
                json.dumps(self.verdicts({"M6-IB": self.finalist((0.5, 4.0))}))
            )
            with self.assertRaises(SystemExit):
                m.main(
                    [
                        "--verdicts",
                        str(verdicts),
                        "--public",
                        str(root / "p.json"),
                        "--private",
                        str(root / "x.json"),
                    ]
                )
            (root / "private").mkdir()
            m.main(
                [
                    "--verdicts",
                    str(verdicts),
                    "--public",
                    str(root / "p.json"),
                    "--private",
                    str(root / "private" / "x.json"),
                ]
            )
            self.assertEqual(
                json.loads((root / "p.json").read_text())["choice"], "M6-IB"
            )


class M6ScriptTest(unittest.TestCase):
    def test_arm_caps(self):
        sha = "0" * 64

        def run(arm, mix, cap):
            return subprocess.run(
                [
                    "bash",
                    str(M6 / "m6-arm.sh"),
                    "d",
                    "2",
                    arm,
                    "s1",
                    mix,
                    sha,
                    "861",
                    cap,
                ],
                capture_output=True,
                text=True,
            )

        for arm, mix, cap in (
            ("M6-IB2", "a20ib12", "20.5"),
            ("M6-IB2PN", "a20ib12pn", "22.5"),
        ):
            out = run(arm, mix, cap)
            self.assertEqual(out.returncode, 2)
            self.assertIn("bad SAVE_EVERY", out.stderr)
        out = run("M6-IB2PN", "a20ib12pn", "22")
        self.assertEqual(out.returncode, 2)
        self.assertIn("mixtures-m6pn-1/a20ib12pn.train.jsonl is not", out.stderr)
        out = run("M6-IB3", "a20ib12", "20")
        self.assertIn("unknown arm", out.stderr)

    def test_index_entries_match_ix1_launcher(self):
        launcher = (ROOT.parent / "eval" / "ix1" / "launch.sh").read_text()
        index = (M6 / "m6-index.sh").read_text()
        self.assertIn("MD=/data/dev2/models/ix1/m6\n", index)
        self.assertIn("PKG=$MD/$ARM-re876fbe", index)
        for arm in (
            "M6-IB",
            "M6-IBX",
            "M6-IB2",
            "M6-IB2PN",
            "M6-IBxIB2-m50",
            "M6-IBxIB2-m67",
            "M7-IB124ML",
            "M7-IB14ML",
            "M8-IB14",
            "M8-IB124",
            "X7-IBxIB2xIB14ML",
            "X7-4ARM",
            "X8-IBxIB2-8",
            "X8-ML",
            "X9-LRH2",
            "X9-LRH2xM50",
            "X9-LRH",
            "X9-LRHxM50",
            "X9-ML0",
            "X9-IBxIB2-10",
        ):
            entry = (
                f'  [{arm}]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 '
                f'/data/dev2/models/ix1/m6/{arm}-re876fbe"\n'
            )
            self.assertIn(entry, launcher)

    def test_index_argument_checks(self):
        for script in ("m6-index.sh", "m6-index-run.sh"):
            for args, message in (
                (["abc", "M6-IB", "stage"], "full commit SHA"),
                (["0" * 40, "M6-IB3", "stage"], "bad ARM"),
                (["0" * 40, "M6-IBxIB2-m5", "stage"], "bad ARM"),
                (["0" * 40, "M6-IBxIB2", "stage"], "bad ARM"),
                (["0" * 40, "M7-IB24", "stage"], "bad ARM"),
                (["0" * 40, "X9-LRH4", "stage"], "bad ARM"),
                (["0" * 40, "M9-IB-lrh", "stage"], "bad ARM"),
            ):
                with self.subTest(script=script, args=args):
                    out = subprocess.run(
                        ["bash", str(M6 / script), *args],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(out.returncode, 2)
                    self.assertIn(message, out.stderr)
        out = subprocess.run(
            ["bash", str(M6 / "m6-stage-a.sh"), "M6-IB3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 2)
        self.assertIn("bad ARM", out.stderr)

    def index_run(self, *args, dry=True):
        env = dict(os.environ, M6_INDEX_DRY="1" if dry else "0")
        return subprocess.run(
            ["bash", str(M6 / "m6-index-run.sh"), "0" * 40, "M6-IB", *args],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_index_run_node_checks(self):
        for args, message in (
            (["f", "0", "4"], "NODE must be b, c, d or e"),
            (["b", "0", "2"], "node b: GPU 2 is not allowed"),
            (["e", "0", "4"], "node e: GPU 4 is not allowed"),
            (["e", "0", "5"], "node e: GPU 5 is not allowed"),
            (["d", "0,8", "4"], "SHARDS"),
            (["d", "0,0", "4"], "lists a shard twice"),
            (["d", "0,1"], "no GPU listed"),
            (["d", "0,1", "8"], "node d: GPU 8 is not allowed"),
            (["c", "0,1", "0"], "node c: GPU 0 is not allowed"),
            (["c", "0,1", "1", "1"], "listed twice"),
        ):
            with self.subTest(args=args):
                out = self.index_run(*args)
                self.assertEqual(out.returncode, 2)
                self.assertIn(message, out.stderr)

    def test_index_run_deals_shards_to_gpus(self):
        out = self.index_run("c", "4,5,6,7", "1", "2")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(
            out.stdout.splitlines(),
            [
                "shard 4: node c GPU1",
                "shard 5: node c GPU2",
                "shard 6: node c GPU1",
                "shard 7: node c GPU2",
                'launch.sh --gpus "9 9 9 9 1 2 1 2"',
            ],
        )
        out = self.index_run("d", "0,1", "4", "5", "6", "7")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('launch.sh --gpus "4 5 9 9 9 9 9 9"', out.stdout)
        out = self.index_run("e", "2,3", "0", "7")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('launch.sh --gpus "9 9 0 7 9 9 9 9"', out.stdout)
        env = dict(os.environ, M6_INDEX_DRY="1", M6_INDEX_STAGGER="2m")
        out = subprocess.run(
            ["bash", str(M6 / "m6-index-run.sh"), "0" * 40, "M6-IB", "e", "2", "0"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(out.returncode, 2)
        self.assertIn("M6_INDEX_STAGGER", out.stderr)
        env = dict(os.environ, M6_INDEX_DRY="1", M6_INDEX_AFTER="8")
        out = subprocess.run(
            ["bash", str(M6 / "m6-index-run.sh"), "0" * 40, "M6-IB", "d", "5", "0"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(out.returncode, 2)
        self.assertIn("M6_INDEX_AFTER", out.stderr)
        out = self.index_run("d", "0,1", "4", dry=False)
        self.assertEqual(out.returncode, 2)
        self.assertIn("missing mirror", out.stderr)

    def index_plan(self, gpus, shards=None):
        env = dict(os.environ, M6_INDEX_GPUS=gpus, DEV2_NODES_FILE="/nonexistent")
        if shards is not None:
            env["M6_INDEX_SHARDS"] = shards
        return subprocess.run(
            ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IB", "plan"],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_index_plan_splits_shards_across_nodes(self):
        out = self.index_plan("d4 d5 d6 d7 c1 c2 c3 c4")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertIn("node d: shards 0,1,2,3 on GPU 4 5 6 7", lines)
        self.assertIn("node c: shards 4,5,6,7 on GPU 1 2 3 4", lines)
        self.assertIn('  launch.sh --gpus "9 9 9 9 1 2 3 4"', lines)
        out = self.index_plan("d4 c1 d5 c2")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertIn("node d: shards 0,2,4,6 on GPU 4 5", lines)
        self.assertIn("node c: shards 1,3,5,7 on GPU 1 2", lines)
        self.assertIn('  launch.sh --gpus "4 9 5 9 4 9 5 9"', lines)
        self.assertIn('  launch.sh --gpus "9 1 9 2 9 1 9 2"', lines)
        self.assertIn("  shard 6: node d GPU5", lines)
        self.assertIn("  shard 7: node c GPU2", lines)
        out = self.index_plan("4 5 6 7")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(
            out.stdout.splitlines()[0], "node d: shards 0,1,2,3,4,5,6,7 on GPU 4 5 6 7"
        )
        out = self.index_plan("e0 e1 e2 e3 e6 e7", shards="2 3 4 5 6 7")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertEqual(lines[0], "node e: shards 2,3,4,5,6,7 on GPU 0 1 2 3 6 7")
        self.assertIn('  launch.sh --gpus "9 9 0 1 2 3 6 7"', lines)
        out = self.index_plan("b0 b1 b5 d0 d1", shards="2 3 4 5 6 7")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertIn("node d: shards 5,6 on GPU 0 1", lines)
        self.assertIn("node b: shards 2,3,4,7 on GPU 0 1 5", lines)
        self.assertIn('  launch.sh --gpus "9 9 0 1 5 9 9 0"', lines)
        for shards, message in (
            ("2 8", "not '8'"),
            ("2 2", "lists 2 twice"),
            (" ", "at least one shard"),
        ):
            with self.subTest(shards=shards):
                out = self.index_plan("e0", shards=shards)
                self.assertEqual(out.returncode, 2)
                self.assertIn(message, out.stderr)
        for gpus, message in (
            ("e4", "not 'e4'"),
            ("b2", "not 'b2'"),
            ("e5", "not 'e5'"),
            ("c0", "not 'c0'"),
            ("d8", "not 'd8'"),
            ("d4 d4", "lists d4 twice"),
            ("4 d4", "lists d4 twice"),
            (" ", "1-8 entries"),
            ("c1 c2 c3 c4 c5 c6 c7 d4 d5", "1-8 entries"),
        ):
            with self.subTest(gpus=gpus):
                out = self.index_plan(gpus)
                self.assertEqual(out.returncode, 2)
                self.assertIn(message, out.stderr)

    def test_index_run_launches_every_node_of_a_split_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir, log = Path(tmp) / "bin", Path(tmp) / "ssh.log"
            bin_dir.mkdir()
            (
                bin_dir / "ssh"
            ).write_text(  # like ssh: reads stdin, answers every check with the same line
                f'#!/usr/bin/env bash\necho "${{@: -1}}" >> {log}\ncat > /dev/null\necho ok\n'
            )
            (bin_dir / "sleep").write_text("#!/usr/bin/env bash\n")
            for stub in ("ssh", "sleep"):
                (bin_dir / stub).chmod(0o755)
            nodes = Path(tmp) / "nodes.env"
            nodes.write_text(
                "node-b=root@b\nnode-c=root@c\nnode-d=root@d\nnode-e=root@e\n"
            )
            out = subprocess.run(
                ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IB", "run"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=60,
                env=dict(
                    os.environ,
                    PATH=f"{bin_dir}:{os.environ['PATH']}",
                    DEV2_NODES_FILE=str(nodes),
                    M6_INDEX_GPUS="d4 c1 e0",
                    M6_INDEX_STAGGER="120",
                ),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            calls = log.read_text()
            self.assertIn("m6-index-run.sh " + "0" * 40 + " M6-IB d 0,3,6 4", calls)
            self.assertIn("m6-index-run.sh " + "0" * 40 + " M6-IB c 1,4,7 1", calls)
            self.assertIn("m6-index-run.sh " + "0" * 40 + " M6-IB e 2,5 0", calls)
            self.assertIn("M6_INDEX_STAGGER=120 M6_INDEX_AFTER= setsid", calls)
            self.assertIn(
                "tail -n 3 /data/dev2/private/eval/index021/ix1/logs/m6-index-M6-IB-c.log",
                calls,
            )

    def test_index_stage_from_soup_needs_opt_in(self):
        soup = "ab" * 32
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir, log = Path(tmp) / "bin", Path(tmp) / "ssh.log"
            bin_dir.mkdir()
            (
                bin_dir / "ssh"
            ).write_text(  # no PACKAGE.json on node B; the soup names its identity
                "#!/usr/bin/env bash\n"
                f'cmd="${{@: -1}}"; echo "$cmd" >> {log}; cat > /dev/null\n'
                'case "$cmd" in "test -f "*PACKAGE.json) exit 1 ;; '
                '"test -f "*soup_manifest.json) exit 0 ;; '
                "*rank*soup_manifest.json) echo 256 ;; "
                f"*soup_manifest.json) echo {soup} ;; *) echo ok ;; esac\n"
            )
            (bin_dir / "ssh").chmod(0o755)
            nodes = Path(tmp) / "nodes.env"
            nodes.write_text("node-b=root@b\nnode-c=root@c\nnode-d=root@d\n")
            env = dict(
                os.environ,
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                DEV2_NODES_FILE=str(nodes),
            )
            run = ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IBX", "stage"]
            out = subprocess.run(
                run, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env
            )
            self.assertEqual(out.returncode, 3)
            self.assertIn("M6_INDEX_FROM_SOUP=1", out.stderr)
            out = subprocess.run(
                run,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=dict(env, M6_INDEX_FROM_SOUP="1"),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn(f"--model-sha256 {soup}", log.read_text())
            self.assertIn(f"MODEL_MANIFEST.json {soup} 27497508864", log.read_text())

    def fake_nodes(self, tmp, answers):
        """A fake ssh answering each remote command by the first matching (glob, line) pair, else "ok"."""
        bin_dir, log = Path(tmp) / "bin", Path(tmp) / "ssh.log"
        bin_dir.mkdir()
        cases = " ".join(f"{glob}) echo {line} ;;" for glob, line in answers)
        (bin_dir / "ssh").write_text(
            "#!/usr/bin/env bash\n"
            f'cmd="${{@: -1}}"; echo "$cmd" >> {log}; cat > /dev/null\n'
            f'case "$cmd" in {cases} *) echo ok ;; esac\n'
        )
        (bin_dir / "ssh").chmod(0o755)
        nodes = Path(tmp) / "nodes.env"
        nodes.write_text("node-b=root@b\nnode-c=root@c\nnode-d=root@d\nnode-e=root@e\n")
        env = dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            DEV2_NODES_FILE=str(nodes),
        )
        return env, log

    def test_index_stage_takes_the_loaded_count_from_the_soup_rank(self):
        soup = "cd" * 32
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.fake_nodes(
                tmp,
                [
                    ('"test -f "*PACKAGE.json', "> /dev/null; exit 1"),
                    ('"test -f "*soup_manifest.json', "> /dev/null"),
                    ("*rank*soup_manifest.json", "512"),
                    ("*soup_manifest.json", soup),
                ],
            )
            out = subprocess.run(
                ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IBxIB2-m50", "stage"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=dict(env, M6_INDEX_FROM_SOUP="1"),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            calls = log.read_text()
            self.assertIn("/data/dev2/models/ix1/m6/M6-IBxIB2-m50-re876fbe", calls)
            self.assertIn(f"MODEL_MANIFEST.json {soup} 29365153792", calls)
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = self.fake_nodes(
                tmp,
                [
                    ('"test -f "*PACKAGE.json', "> /dev/null; exit 1"),
                    ('"test -f "*soup_manifest.json', "> /dev/null"),
                    ("*rank*soup_manifest.json", "abc"),
                    ("*soup_manifest.json", soup),
                ],
            )
            out = subprocess.run(
                ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M7-IB124ML", "stage"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=dict(env, M6_INDEX_FROM_SOUP="1"),
            )
            self.assertEqual(out.returncode, 3)
            self.assertIn("bad LoRA rank", out.stderr)

    def test_index_parity_gpu_and_score_base_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.fake_nodes(tmp, [])
            for gpu in ("e4", "e5", "c1", "d8", "b2"):
                with self.subTest(gpu=gpu):
                    out = subprocess.run(
                        [
                            "bash",
                            str(M6 / "m6-index.sh"),
                            "0" * 40,
                            "M6-IBxIB2-m67",
                            "parity",
                        ],
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        env=dict(env, M6_PARITY_GPU=gpu),
                    )
                    self.assertEqual(out.returncode, 2)
                    self.assertIn("M6_PARITY_GPU", out.stderr)
            for base in ("M6-IB3", "M6-IBxIB2-m67"):
                with self.subTest(base=base):
                    out = subprocess.run(
                        [
                            "bash",
                            str(M6 / "m6-index.sh"),
                            "0" * 40,
                            "M6-IBxIB2-m67",
                            "score",
                        ],
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        env=dict(env, M6_INDEX_BASE=base),
                    )
                    self.assertEqual(out.returncode, 2)
                    self.assertIn("M6_INDEX_BASE", out.stderr)

    def test_index_hold_and_xarm_argument_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.fake_nodes(tmp, [])
            for gpus in ("", "c1", "e4", "b2", "d8"):
                with self.subTest(gpus=gpus):
                    out = subprocess.run(
                        [
                            "bash",
                            str(M6 / "m6-index.sh"),
                            "0" * 40,
                            "M6-IBxIB2-m50",
                            "hold",
                        ],
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        env=dict(env, M6_INDEX_GPUS=gpus),
                    )
                    self.assertEqual(out.returncode, 2)
                    self.assertIn("M6_INDEX_GPUS", out.stderr)
            out = subprocess.run(
                ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IBxIB2-m50", "unhold"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=dict(env, M6_INDEX_GPUS="d5 e0 b1"),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(
                log.read_text().count("27B Index hold for M6-IBxIB2-m50"), 3
            )
        for args, message in (
            (["abc", "M6-IBxIB2-m50", "1"], "full commit SHA"),
            (["0" * 40, "M6-IB", "1"], "bad NAME"),
            (["0" * 40, "M6-IBxIB2-m50", "2"], "not an M6 node B GPU"),
            (["0" * 40, "M7-IB124ML", "1"], "missing mirror"),
            (["0" * 40, "X9-LRH2xM50", "1"], "missing mirror"),
            (["0" * 40, "X9-LRH3", "1"], "bad NAME"),
        ):
            with self.subTest(args=args):
                out = subprocess.run(
                    ["bash", str(M6 / "m6-xarm.sh"), *args],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(out.returncode, 2)
                self.assertIn(message, out.stderr)

    def test_index_parity_on_node_e_copies_the_record_to_node_d(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, log = self.fake_nodes(
                tmp,
                [('"test ! -e "*', "> /dev/null"), ("*sha256sum*parity.json", "same")],
            )
            (Path(tmp) / "bin" / "sleep").write_text("#!/usr/bin/env bash\n")
            (Path(tmp) / "bin" / "sleep").chmod(0o755)
            out = subprocess.run(
                ["bash", str(M6 / "m6-index.sh"), "0" * 40, "M6-IBxIB2-m50", "parity"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=60,
                env=dict(env, M6_PARITY_GPU="e6"),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            calls = log.read_text()
            self.assertIn("launch.sh parity --src", calls)
            self.assertRegex(calls, r"--model M6-IBxIB2-m50\s+--gpu 6 ")
            self.assertIn("parity record copied node E -> node D", out.stdout)

    def test_contrast_guard_moves_m4_contrast_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            gates = Path(tmp)
            (gates / "contrast.json").write_text(
                json.dumps({"pairs": {}, "runs": {}, "score_levels": {"M6-IB": {}}})
            )
            (gates / "contrast.log").write_text("log\n")
            chain = subprocess.run(  # not our child: init reaps it, so kill -0 fails once it ends
                ["bash", "-c", "sleep 2 > /dev/null 2>&1 & echo $!"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            out = subprocess.run(
                ["bash", str(M6 / "m6-contrast-guard.sh"), chain],
                capture_output=True,
                text=True,
                timeout=60,
                env=dict(os.environ, M6_GATES=tmp, M6_GUARD_LOG_DELAY="0"),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertFalse((gates / "contrast.json").exists())
            moved = sorted(p.name for p in gates.glob("contrast-M6-IB-*"))
            self.assertEqual([Path(n).suffix for n in moved], [".json", ".log"])
            self.assertIn("complete (no listed chain alive)", out.stdout)

    def test_relay_accepts_m9_arm_seeds(self):
        for name, message in (
            ("M8-IB124-s4", "no arm-seed run"),
            ("M9-IB12ML-s5", "no arm-seed run"),
            ("M9-IB2-lrh-s6", "no arm-seed run"),
            ("M9-IB-lrh-s5", "no arm-seed run"),
            ("M9-IB-lrq-s5", "NAME is an"),
            ("M9-IB-s7", "NAME is an"),
            ("X9-LRH-s5", "NAME is an"),
        ):
            with self.subTest(name=name):
                out = subprocess.run(
                    ["bash", str(M6 / "m6-relay.sh"), name, "1"],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, RELAY_NODE="c"),
                )
                self.assertEqual(out.returncode, 2)
                self.assertIn(message, out.stderr)

    def test_bash_n(self):
        scripts = sorted(M6.glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
