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
            self.assertEqual(sorted(launch.allowed_gpus("a")), [2])
            self.assertEqual(launch.max_cap_hours(), 20.0)
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


class M6ScriptTest(unittest.TestCase):
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
