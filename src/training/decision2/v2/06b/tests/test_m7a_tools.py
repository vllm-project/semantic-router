import hashlib
import importlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from training.model.data import digest, file_sha256
from training.model.infer import checkpoint_fingerprint, normalized_answer
from training.model.score_bias import SCORE_BIAS_FORMAT
from v2.eval import panels

aho = importlib.import_module("v2.06b.m7_aho")
sb = importlib.import_module("v2.06b.m7_scorebias")
package = importlib.import_module("v2.06b.m7_package")
common = importlib.import_module("v2.06b.common")


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


def training_row(key, group, levels, label, state, *, task="score", language="en"):
    options = (
        [{"key": str(i), "description": f"level {i}"} for i in range(levels)]
        if task == "score"
        else [{"key": k, "description": k} for k in ("a", "b")]
    )
    row = {
        "id": key,
        "state": state,
        "instructions": "Rate it",
        "options": options,
        "label": label,
        "task_type": task,
        "family": "synthetic",
        "group_id": group,
        "language": language,
        "split": "select",
        "source": "synthetic",
        "evaluation_role": "select",
        "render_template": "t",
        "audit_metadata": {},
    }
    row["input_sha256"] = digest(
        {f: row[f] for f in ("state", "instructions", "options", "task_type")}
    )
    return row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=False) + "\n")


def fake_checkpoint(path: Path, weights: bytes = b"weights") -> None:
    (path / "backbone").mkdir(parents=True)
    (path / "decision_config.json").write_text("{}")
    (path / "decision_head.safetensors").write_bytes(b"head")
    (path / "backbone" / "model.safetensors").write_bytes(weights)
    (path / "backbone" / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")


class SplitTest(unittest.TestCase):
    def test_split_rule_is_the_preregistered_hash(self):
        for group in ("g1", "a3:musique:01980c", "", "組"):
            value = int(
                hashlib.sha256(("m7a-aho-split:" + group).encode()).hexdigest()[:16], 16
            )
            self.assertEqual(aho.split_of(group), "fit" if value % 2 == 0 else "chk")
            self.assertEqual(aho.split_of(group), aho.split_of(group))
        splits = {aho.split_of(f"g{i}") for i in range(20)}
        self.assertEqual(splits, {"fit", "chk"})
        self.assertEqual(aho.group_of({"id": "x", "group_id": ""}), "x")
        self.assertEqual(aho.group_of({"id": "x", "group_id": "g"}), "g")

    def test_state_key_separates_text_and_structure(self):
        self.assertEqual(aho.state_key("a b"), "a b")
        self.assertEqual(
            aho.state_key({"b": 1, "a": 2}), aho.state_key({"a": 2, "b": 1})
        )
        self.assertNotEqual(aho.state_key('{"a":2}'), aho.state_key({"a": 2}))


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.saved = aho.CAL698_SCORE_ROWS
        aho.CAL698_SCORE_ROWS = 2

    def tearDown(self):
        aho.CAL698_SCORE_ROWS = self.saved

    def make_inputs(self, root: Path) -> list[str]:
        arm_a = [
            training_row(f"a{i}", f"ga{i // 2}", 5, i % 5, f"state a{i}")
            for i in range(40)
        ]
        arm_a.append(training_row("a-choice", "gac", 2, 0, "choice", task="choice"))
        arm_a.append(training_row("a-l6", "ga6", 6, 1, "six"))
        arm_a.append(training_row("a-mix", "gmix", 5, 1, "in mixture"))
        arm_a.append(training_row("a-mixgroup", "gmixg", 5, 1, "group in mixture"))
        arm_a.append(training_row("a-select", "gsel", 3, 1, "select row"))
        arm_a.append(training_row("a-excl", "a3:excluded", 4, 1, "excluded group"))
        arm_a.append(training_row("a-panel", "gpan", 3, 1, {"panel": "state"}))
        arm_a.append(training_row("a-cal", "gc4", 3, 1, "cal group", language="zh"))
        arm_b = [
            training_row(f"b{i}", f"gb{i}", 3, i % 3, f"state b{i}", language="zh")
            for i in range(30)
        ]
        write_rows(root / "A.jsonl", arm_a)
        write_rows(root / "B.jsonl", arm_b)
        write_rows(
            root / "mix.jsonl",
            [
                training_row("a-mix", "other", 5, 1, "in mixture"),
                training_row("m2", "gmixg", 5, 1, "x"),
            ],
        )
        write_rows(
            root / "select.jsonl", [training_row("s1", "gs", 3, 0, "select row")]
        )
        cal = [
            training_row("c1", "gc1", 5, 1, "cal one"),
            training_row("c2", "gc2", 5, 3, "cal two"),
            training_row("c3", "gc3", 2, 0, "cal three", task="choice"),
            training_row("c4", "gc4", 5, 0, "cal state", language="zh"),
        ]
        cal[3]["group_id"] = "gc4"
        write_rows(root / "cal700.jsonl", cal)
        write_rows(root / "cal698.jsonl", cal[:3])
        (root / "excluded.json").write_text(
            json.dumps(
                {
                    "schema": "dev2-overlap-excluded-groups/1",
                    "groups": {
                        "a3:excluded": {"input_sha256": [], "pools": [], "row_ids": []}
                    },
                }
            )
        )
        write_rows(
            root / "panels" / "goldfree" / "typed-dev.prompts.jsonl",
            [{"id": "p", "state": {"panel": "state"}, "questions": {}}],
        )
        return [
            "build",
            "--output",
            str(root / "out"),
            "--arm",
            f"A7q={root / 'A.jsonl'}",
            "--arm",
            f"H6={root / 'B.jsonl'}",
            "--mixture",
            str(root / "mix.jsonl"),
            "--select",
            str(root / "select.jsonl"),
            "--cal700",
            str(root / "cal700.jsonl"),
            "--cal698",
            str(root / "cal698.jsonl"),
            "--cal698-sha256",
            file_sha256(root / "cal698.jsonl"),
            "--excluded-groups",
            str(root / "excluded.json"),
            "--panel-root",
            str(root / "panels"),
            "--panel",
            "typed-dev",
            "--pi-manifest",
            "",
        ]

    def test_build_filters_drops_and_splits_by_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quiet(aho.main, self.make_inputs(root))
            out = root / "out"
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["filtered_not_score_L345"], {"A7q": 2})
            self.assertEqual(
                manifest["dropped"]["by_first_reason"],
                {
                    "in_m6_training_mixture": 2,
                    "in_select": 1,
                    "r2_excluded_group": 1,
                    "panel_state_match": 1,
                    "in_cal": 1,
                },
            )
            fit = (out / "fit_aho.jsonl").read_text().splitlines()
            chk = (out / "chk.jsonl").read_text().splitlines()
            self.assertEqual(len(fit) + len(chk), 70)
            original = {
                json.loads(line)["id"]: line
                for name in ("A.jsonl", "B.jsonl")
                for line in (root / name).read_text().splitlines()
            }
            groups = {}
            for split, lines in (("fit", fit), ("chk", chk)):
                for line in lines:
                    row = json.loads(line)
                    self.assertEqual(line, original[row["id"]])
                    self.assertEqual(aho.split_of(row["group_id"]), split)
                    groups.setdefault(row["group_id"], set()).add(split)
            self.assertTrue(all(len(s) == 1 for s in groups.values()))
            self.assertEqual(manifest["kept"]["fit"], len(fit))
            ids = json.loads((out / "cal698.score.ids.json").read_text())
            self.assertEqual(ids["ids"], ["c1", "c2"])
            index = common.read_jsonl(out / "index.jsonl")
            self.assertEqual({e["arm"] for e in index}, {"A7q", "H6"})
            self.assertNotIn("label", index[0])
            counts = manifest["counts_arm_levels_split_gold"]
            total = sum(
                n
                for arm in counts.values()
                for level in arm.values()
                for split in level.values()
                for n in split.values()
            )
            self.assertEqual(total, 70)
            for name, entry in manifest["outputs"].items():
                self.assertEqual(file_sha256(out / name), entry["sha256"])
            with self.assertRaises(FileExistsError):
                quiet(aho.main, self.make_inputs(root))


def cyclic_sample(levels, bias, bases, seed):
    """Rows whose true logits are all cyclic rotations of random vectors, so the
    expected label frequencies are uniform; the model adds `bias`."""
    rng = random.Random(seed)
    rows = []
    for _ in range(bases):
        v = [rng.gauss(0, 1.5) for _ in range(levels)]
        for shift in range(levels):
            t = v[shift:] + v[:shift]
            p = sb.softmax(t)
            y = rng.choices(range(levels), weights=p)[0]
            rows.append(([a + b for a, b in zip(t, bias)], y))
    return rows


class FitTest(unittest.TestCase):
    def test_constant_logits_are_undone_exactly(self):
        z = [0.3, -1.2, 0.9]
        rows = [(z, 0)] * 3 + [(z, 1)] + [(z, 2)] * 2
        result = sb.fit_level(rows, 3)
        mean = sum(z) / 3
        self.assertEqual(result["offsets"], [round(-(v - mean), 6) + 0.0 for v in z])
        self.assertLess(result["gradient_norm"], 1e-9)

    def test_recovers_known_offsets_on_generated_data(self):
        bias = [0.0, 0.8, -0.5, 1.1, -0.2]
        rows = cyclic_sample(5, bias, 1200, 7)
        result = sb.fit_level(rows, 5)
        mean = sum(bias) / 5
        for got, want in zip(result["offsets"], [-(b - mean) for b in bias]):
            self.assertAlmostEqual(got, want, delta=0.08)
        self.assertAlmostEqual(sum(result["offsets"]), 0.0, places=5)
        self.assertLess(result["gradient_norm"], 1e-9)
        self.assertLess(result["objective_end"], result["objective_start"])
        self.assertEqual(result["offsets"], [round(v, 6) for v in result["offsets"]])

    def test_missing_level_stops_the_fit(self):
        with self.assertRaisesRegex(sb.FitError, "misses level"):
            sb.fit_level([([0.0, 0.0, 0.0], 0), ([0.0, 0.0, 0.0], 1)], 3)

    def test_solve(self):
        x = sb.solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
        self.assertAlmostEqual(x[0], 0.8)
        self.assertAlmostEqual(x[1], 1.4)


class StatisticsTest(unittest.TestCase):
    def test_wilson_kappa_and_bootstrap_on_tiny_cases(self):
        low, high = sb.wilson(5, 10)
        half = 1.96 * math.sqrt(0.025 + 1.96**2 / 400) / (1 + 1.96**2 / 10)
        self.assertAlmostEqual(low, 0.5 - half)
        self.assertAlmostEqual(high, 0.5 + half)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(sb.kappa([0, 0, 1, 1], [0, 1, 1, 1]), 0.5)
        self.assertAlmostEqual(sb.kappa([None, 1], [0, 1]), (0.5 - 0.25) / 0.75)
        self.assertIsNone(sb.kappa([1, 1], [1, 1]))
        self.assertEqual(sb.bootstrap([1.0, 1.0, 1.0]), [1.0, 1.0])
        self.assertEqual(sb.bootstrap([0.0, 1.0]), [0.0, 1.0])
        self.assertEqual(sb.bootstrap([0.0, 1.0]), sb.bootstrap([0.0, 1.0]))
        rng = random.Random(20260929)
        draws = sorted(
            sum([0.0, 1.0, 1.0][rng.randrange(3)] for _ in range(3)) / 3
            for _ in range(5000)
        )
        self.assertEqual(sb.bootstrap([0.0, 1.0, 1.0])[0], draws[int(4999 * 0.025)])

    def test_summary_majority_modal_share_and_recall(self):
        gold = [0, 0, 0, 1, 2]
        pred = [0, None, 1, 1, 2]
        out = sb.summary(pred, gold, intervals=True)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["majority_level"], 0)
        self.assertAlmostEqual(out["majority_accuracy"], 0.6)
        self.assertAlmostEqual(out["delta_vs_majority"], 0.0)
        self.assertEqual(
            out["predicted_distribution"], {"0": 1, "1": 2, "2": 1, "invalid": 1}
        )
        self.assertEqual(out["top_value"], "1")
        self.assertAlmostEqual(out["top_share"], 0.4)
        self.assertEqual(out["recall_by_level"], {"0": 1 / 3, "1": 1.0, "2": 1.0})
        self.assertEqual(len(out["delta_vs_majority_ci95"]), 2)
        self.assertEqual(sb.majority_level([2, 1, 2, 1]), 1)
        self.assertEqual(sb.predicted([0.5, 0.5, 0.0]), None)
        self.assertEqual(sb.predicted([0.2, 0.5, 0.3]), 1)
        self.assertEqual(
            sb.confusion([0, None], [0, 1]), {"0": {"0": 1}, "1": {"invalid": 1}}
        )


def logged_record(model, bias_sha=None, s_logits=(0.0, 0.0, 1.0), choice=(1.0, 0.0)):
    record = {
        "id": "p1",
        "model_sha256": model,
        "answers": {
            "s": normalized_answer("score", ["0", "1", "2"], list(s_logits), 1.0),
            "c": normalized_answer("choice", ["a", "b"], list(choice), 1.0),
            "n": normalized_answer("noul", ["false", "true"], [0.0, 1.0], 1.0),
            "bad": {"type": "score", "error": "max_length_exceeded"},
        },
    }
    if bias_sha:
        record["score_bias_sha256"] = bias_sha
    return record


class ReplayTest(unittest.TestCase):
    def run_replay(self, root: Path, online_record: dict, bias_path: Path, model: str):
        logged = root / "logged.jsonl"
        write_rows(logged, [logged_record(model)])
        online = root / "online" / "dev.predictions.jsonl"
        write_rows(online, [online_record])
        (online.parent / "dev.predictions.jsonl.manifest.json").write_text(
            json.dumps(
                {"model_sha256": model, "score_bias_sha256": file_sha256(bias_path)}
            )
        )
        output = root / f"ad3-{len(list(root.glob('ad3-*')))}.json"
        quiet(
            sb.main,
            [
                "replay",
                "--logged",
                str(logged),
                "--online",
                str(online),
                "--score-bias",
                str(bias_path),
                "--output",
                str(output),
            ],
        )
        (online.parent / "dev.predictions.jsonl.manifest.json").unlink()
        return json.loads(output.read_text())

    def test_replay_passes_and_detects_mismatches(self):
        model = "d" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bias = root / "score_bias.json"
            bias.write_text(
                json.dumps(
                    {
                        "format": SCORE_BIAS_FORMAT,
                        "model_sha256": model,
                        "offsets": {"3": [1.5, 0.0, -1.5]},
                        "fit": {},
                    }
                )
            )
            sha = file_sha256(bias)
            good = logged_record(model, sha, s_logits=(1.5, 0.0, -0.5))
            result = self.run_replay(root, good, bias, model)
            self.assertTrue(result["A-D3"]["pass"], result)
            self.assertLess(result["max_abs_dp"]["score"], 1e-12)
            self.assertEqual(result["counts"]["errors"], 1)

            uncorrected = logged_record(model, sha)
            result = self.run_replay(root, uncorrected, bias, model)
            self.assertFalse(result["A-D3"]["pass"])
            self.assertEqual(result["mismatches"][0]["reason"], "score argmax")

            flipped = logged_record(
                model, sha, s_logits=(1.5, 0.0, -0.5), choice=(0.0, 1.0)
            )
            result = self.run_replay(root, flipped, bias, model)
            self.assertFalse(result["A-D3"]["pass"])
            self.assertIn("choice", [m["reason"] for m in result["mismatches"]])

            drift = logged_record(
                model, sha, s_logits=(1.5, 0.0, -0.5), choice=(1.0, 0.001)
            )
            result = self.run_replay(root, drift, bias, model)
            self.assertFalse(result["A-D3"]["checks"]["max_abs_dp_le_1e-4"])
            self.assertTrue(result["A-D3"]["checks"]["no_answer_mismatch"])

            unbound = logged_record(model, None, s_logits=(1.5, 0.0, -0.5))
            result = self.run_replay(root, unbound, bias, model)
            self.assertFalse(result["A-D3"]["checks"]["online_rows_score_bias_sha256"])


class PackageTest(unittest.TestCase):
    def make_source(self, root: Path) -> tuple[Path, Path]:
        source = root / "src" / "full"
        export = source / "best-export"
        fake_checkpoint(export)
        files = {
            item.relative_to(export).as_posix(): {
                "bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
            for item in sorted(export.rglob("*"))
            if item.is_file()
        }
        manifest = common.write_json(
            source / "best-export.MANIFEST.json",
            {"schema": "dev2-06b-causal-files/1", "files": files},
        )
        common.write_json(
            source / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "best_export_manifest_sha256": manifest,
                "state_sha256": "s",
            },
        )
        bias = root / "score_bias.json"
        bias.write_text(
            json.dumps(
                {
                    "format": SCORE_BIAS_FORMAT,
                    "model_sha256": checkpoint_fingerprint(export)["model_sha256"],
                    "offsets": {"5": [0.1, 0.0, 0.0, 0.0, -0.1]},
                    "fit": {},
                }
            )
        )
        return source, bias

    def test_package_is_byte_identical_plus_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias = self.make_source(root)
            dest = root / "arms" / "m7a" / "full"
            record = quiet(package.build, source, bias, dest)
            manifest_sha = file_sha256(dest / "best-export.MANIFEST.json")
            complete = json.loads((dest / "COMPLETE.json").read_text())
            self.assertEqual(complete["best_export_manifest_sha256"], manifest_sha)
            self.assertEqual(record["files_verified"], 5)
            files = json.loads((dest / "best-export.MANIFEST.json").read_text())[
                "files"
            ]
            self.assertEqual(files["score_bias.json"]["sha256"], file_sha256(bias))
            for name in files:
                if name != "score_bias.json":
                    self.assertEqual(
                        (dest / "best-export" / name).read_bytes(),
                        (source / "best-export" / name).read_bytes(),
                    )
            self.assertEqual(
                checkpoint_fingerprint(dest / "best-export"),
                checkpoint_fingerprint(source / "best-export"),
            )
            with self.assertRaises(FileExistsError):
                package.build(source, bias, dest)

    def test_package_refuses_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias = self.make_source(root)
            (source / "best-export" / "backbone" / "model.safetensors").write_bytes(
                b"other!!"
            )
            with self.assertRaisesRegex(ValueError, "source bytes differ"):
                package.build(source, bias, root / "d1")
            self.assertFalse((root / "d1").exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias = self.make_source(root)
            (source / "best-export" / "extra.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "differ from its manifest"):
                package.build(source, bias, root / "d2")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias = self.make_source(root)
            raw = json.loads(bias.read_text())
            raw["model_sha256"] = "e" * 64
            bias.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "model hash differs"):
                package.build(source, bias, root / "d3")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias = self.make_source(root)
            complete = json.loads((source / "COMPLETE.json").read_text())
            complete["best_export_manifest_sha256"] = "0" * 64
            (source / "COMPLETE.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "does not bind"):
                package.build(source, bias, root / "d4")


class PatchedGold:
    def __init__(self, panel, digest_value):
        self.panel, self.value = panel, digest_value

    def __enter__(self):
        self.saved = panels.ALL[self.panel]["gold_sha256"]
        panels.ALL[self.panel]["gold_sha256"] = self.value

    def __exit__(self, *exc):
        panels.ALL[self.panel]["gold_sha256"] = self.saved


def score_item(key, levels, gold_value):
    return {
        "id": key,
        "state": "s",
        "questions": {
            "q": {
                "type": "score",
                "instructions": "Rate",
                "criteria": [str(i) for i in range(levels)],
            }
        },
        "gold": {
            "q": {"type": "score", "value": gold_value, "semantic_value": gold_value}
        },
    }


class FitCheckIntegrationTest(unittest.TestCase):
    def test_fit_check_devcheck_end_to_end(self):
        bias5 = [0.0, -0.6, 0.2, 0.4, 1.5]
        bias3 = [0.0, -0.3, 1.2]
        dev_bias = [0.0, -0.3, 2.5]
        rng = random.Random(11)

        def probs(levels, y, bias):
            t = [2.2 if k == y else 0.0 for k in range(levels)]
            t = [v + rng.gauss(0, 0.7) for v in t]
            return sb.softmax([a + b for a, b in zip(t, bias)])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aho_dir = root / "aho"
            aho_dir.mkdir()
            rows = [
                training_row(f"r{i}", f"g{i}", 5 if i % 2 else 3, 0, f"s{i}")
                for i in range(1400)
            ]
            for row in rows:
                row["label"] = rng.randrange(len(row["options"]))
            fit_rows = [r for r in rows if aho.split_of(r["group_id"]) == "fit"]
            chk_rows = [r for r in rows if aho.split_of(r["group_id"]) == "chk"]
            write_rows(aho_dir / "fit_aho.jsonl", fit_rows)
            write_rows(aho_dir / "chk.jsonl", chk_rows)
            write_rows(
                aho_dir / "index.jsonl",
                [
                    {
                        "id": r["id"],
                        "arm": "A7q" if i % 3 else "H6",
                        "split": "x",
                        "levels": 0,
                    }
                    for i, r in enumerate(rows)
                ],
            )
            cal = [
                training_row(f"c{i}", f"cg{i}", 5, i % 5, f"c{i}") for i in range(20)
            ]
            write_rows(root / "cal.jsonl", cal)
            common.write_json(
                aho_dir / "cal698.score.ids.json",
                {
                    "ids": [r["id"] for r in cal],
                    "cal700_sha256": file_sha256(root / "cal.jsonl"),
                },
            )
            outputs = {}
            for name in (
                "fit_aho.jsonl",
                "chk.jsonl",
                "index.jsonl",
                "cal698.score.ids.json",
            ):
                outputs[name] = {"sha256": file_sha256(aho_dir / name), "rows": 0}
            common.write_json(aho_dir / "MANIFEST.json", {"outputs": outputs})

            probs_dir = root / "probs"
            probs_dir.mkdir()
            entries = []
            for name, part in (("fit_aho.jsonl", fit_rows), ("chk.jsonl", chk_rows)):
                out = probs_dir / name.replace(".jsonl", ".probs.jsonl")
                sha = common.write_jsonl(
                    out,
                    [
                        {
                            "id": r["id"],
                            "probabilities": probs(
                                len(r["options"]),
                                r["label"],
                                bias5 if len(r["options"]) == 5 else bias3,
                            ),
                        }
                        for r in part
                    ],
                )
                entries.append(
                    {
                        "input_sha256": file_sha256(aho_dir / name),
                        "output": f"/runs/m7/probs/{out.name}",
                        "output_sha256": sha,
                    }
                )
            common.write_json(
                probs_dir / "PROBS.json",
                {
                    "state_sha256": "state",
                    "cal_parity": {"pass": True, "max_abs_drift": 0.0},
                    "outputs": entries,
                },
            )
            soup = root / "soup"
            fake_checkpoint(soup / "best-export")
            cal_sha = common.write_jsonl(
                soup / "cal.probs.jsonl",
                [
                    {"id": r["id"], "probabilities": probs(5, r["label"], bias5)}
                    for r in cal
                ],
            )
            common.write_json(
                soup / "COMPLETE.json",
                {
                    "best_export_manifest_sha256": "m",
                    "state_sha256": "state",
                    "cal": {"probabilities_sha256": cal_sha},
                },
            )
            common_args = [
                "--aho-dir",
                str(aho_dir),
                "--probs",
                str(probs_dir / "PROBS.json"),
                "--soup-dir",
                str(soup),
                "--cal-gold",
                str(root / "cal.jsonl"),
            ]
            code = quiet(
                sb.main,
                [
                    "fit",
                    *common_args,
                    "--output",
                    str(root / "score_bias.json"),
                    "--report",
                    str(root / "fit.json"),
                ],
            )
            self.assertEqual(code, 0)
            bias = json.loads((root / "score_bias.json").read_text())
            model = checkpoint_fingerprint(soup / "best-export")["model_sha256"]
            self.assertEqual(bias["model_sha256"], model)
            self.assertEqual(bias["offsets"]["4"], [0.0] * 4)
            report = json.loads((root / "fit.json").read_text())
            self.assertEqual(report["levels"]["4"]["status"], "too_few_rows")
            self.assertEqual(report["levels"]["5"]["rows_by_source"]["CAL698"], 20)
            for key, true in (("5", bias5), ("3", bias3)):
                mean = sum(true) / len(true)
                for got, want in zip(bias["offsets"][key], [-(b - mean) for b in true]):
                    self.assertAlmostEqual(got, want, delta=0.35)

            dev_gold = root / "typed-dev.gold.jsonl"
            items = [score_item(f"d{i}", 3, i % 3) for i in range(30)]
            write_rows(dev_gold, items)
            dev_predictions = root / "dev.predictions.jsonl"
            write_rows(
                dev_predictions,
                [
                    {
                        "id": item["id"],
                        "model_sha256": model,
                        "answers": {
                            "q": normalized_answer(
                                "score",
                                ["0", "1", "2"],
                                [
                                    z + b
                                    for z, b in zip(
                                        [2.0 if k == i % 3 else 0.0 for k in range(3)],
                                        dev_bias,
                                    )
                                ],
                                1.0,
                            )
                        },
                    }
                    for i, item in enumerate(items)
                ],
            )
            with PatchedGold("typed-dev", file_sha256(dev_gold)):
                quiet(
                    sb.main,
                    [
                        "check",
                        *common_args,
                        "--score-bias",
                        str(root / "score_bias.json"),
                        "--dev-predictions",
                        str(dev_predictions),
                        "--dev-gold",
                        str(dev_gold),
                        "--output",
                        str(root / "check.json"),
                    ],
                )
            result = json.loads((root / "check.json").read_text())
            chk5 = result["chk"]["L5"]
            self.assertGreater(chk5["after"]["accuracy"], chk5["before"]["accuracy"])
            self.assertTrue(result["A-D1"]["pass"])
            dev = result["typed_dev_score"]
            self.assertEqual(dev["before"]["correct"], 10)
            self.assertEqual(dev["after"]["correct"], 30)
            self.assertFalse(result["A-D2"]["pass"])
            self.assertFalse(result["A-D2"]["checks"]["correct_ge_101"])
            self.assertIn("A7q", result["chk_by_arm"])
            self.assertIn("CAL698", result["fit_by_source"])

            quiet(
                sb.main,
                [
                    "devcheck",
                    "--aho-dir",
                    str(aho_dir),
                    "--probs",
                    str(probs_dir / "PROBS.json"),
                    "--label",
                    "m7-x",
                    "--output",
                    str(root / "bd1.json"),
                ],
            )
            bd1 = json.loads((root / "bd1.json").read_text())
            self.assertEqual(bd1["chk"]["L5"]["before"], result["chk"]["L5"]["before"])
            self.assertEqual(
                bd1["B-D1"]["pass"],
                chk5["before"]["top_share"] <= 0.9
                and chk5["before"]["delta_vs_majority_ci95"][0] > 0,
            )


class GateTest(unittest.TestCase):
    def run_gate(self, root: Path, answers: list[int | None], name: str) -> dict:
        panel_root = root / "panels"
        gold = panel_root / "gold" / "typed-final.gold.jsonl"
        items = [score_item(f"f{i}", 5, [0, 1, 2, 3, 4, 4][i % 6]) for i in range(60)]
        write_rows(gold, items)
        run = root / name
        predictions = run / "output" / "typed-final.predictions.jsonl"
        records = []
        for item, level in zip(items, answers):
            record = {"id": item["id"], "answers": {}}
            if level is not None:
                logits = [3.0 if k == level else 0.0 for k in range(5)]
                record["answers"]["q"] = normalized_answer(
                    "score", [str(k) for k in range(5)], logits, 1.0
                )
            records.append(record)
        write_rows(predictions, records)
        (run / "SEAL.json").write_text(
            json.dumps(
                {
                    "panels": {
                        "typed-final": {"predictions_sha256": file_sha256(predictions)}
                    }
                }
            )
        )
        (run / "M6-SUMMARY.json").write_text(
            json.dumps(
                {
                    "successor": {
                        "checks": {
                            "v3_lower_bound_gt_0": True,
                            "H_upper_bound_ge_0": True,
                        }
                    }
                }
            )
        )
        gates = root / f"{name}.gates"
        with PatchedGold("typed-final", file_sha256(gold)):
            quiet(
                sb.main,
                [
                    "gate",
                    "--run",
                    str(run),
                    "--gates",
                    str(gates),
                    "--label",
                    name,
                    "--output",
                    str(gates / "score-s3.json"),
                    "--panel-root",
                    str(panel_root),
                ],
            )
        return json.loads((gates / "score-s3.json").read_text())

    def test_gate_passes_a_good_run_and_fails_a_collapsed_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            levels = [[0, 1, 2, 3, 4, 4][i % 6] for i in range(60)]
            good = [level if i % 10 else None for i, level in enumerate(levels)]
            result = self.run_gate(root, good, "good")
            self.assertEqual(result["score"]["n"], 60)
            self.assertEqual(result["score"]["correct"], 54)
            self.assertEqual(result["score"]["predicted_distribution"]["invalid"], 6)
            self.assertEqual(result["score"]["majority_level"], 4)
            self.assertTrue(result["S3"]["pass"], result["S3"])
            self.assertTrue(result["successor"]["verdict"])

            collapsed = [4] * 57 + [0, 1, 2]
            result = self.run_gate(root, collapsed, "bad")
            self.assertFalse(result["S3"]["checks"]["score_top_share_lt_0.90"])
            self.assertFalse(result["S3"]["checks"]["no_type_collapsed"])
            self.assertFalse(result["successor"]["verdict"])


if __name__ == "__main__":
    unittest.main()
