import gzip
import hashlib
import importlib
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from benchmark.generate import generate, prompt_record
from v2.eval import leak_audit, panels, score5t
from v2.eval.htdev import score as htdev_score
from v2.eval.htdev.build import sha_bytes
from v2.eval.same_panel import read_jsonl
from v2.eval.sealed.score import input_digest

TEST_SEED = b"t" * 32
TEST_SEED_NAMES = {"fit": "test-score5t/fit", "check": "test-score5t/check"}
GROUPS, INDICES = 6, 40


def write_jsonl(path: Path, rows: list) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return sha_bytes(data)


def manifest_file(path: Path, labels: dict[str, list[Path]]) -> Path:
    def entry(p: Path) -> dict:
        size = p.stat().st_size if p.exists() else 0
        return {"path": str(p), "sha256": "0" * 64, "bytes": size}

    data = {
        "schema": "c1-corpora/1",
        "labels": {
            label: {"kind": "training", "files": [entry(p) for p in files]}
            for label, files in labels.items()
        },
    }
    path.write_text(json.dumps(data))
    return path


def model_text(prompt: dict) -> str:
    prefix, options, suffix = score5t.native_inputs()(prompt)[0]
    return prefix + "".join(options) + suffix


def empty_keys() -> dict:
    return {"k0": defaultdict(set), "k1": defaultdict(set)}


def answer(item_id: str, level) -> dict:
    return {"id": item_id, "answers": {"decision": {"type": "score", "score": level}}}


def gold_rows(levels: list[int], half: str = "fit") -> list[dict]:
    return [
        {
            "id": f"{half}{n}",
            "task": score5t.TASK,
            "source": score5t.SOURCE,
            "half": half,
            "group_id": f"{half}-g{n // 4}",
            "language": "en",
            "long": False,
            "questions": score5t.LEDGER_QUESTIONS,
            "gold": {"decision": {"type": "score", "value": v, "semantic_value": v}},
        }
        for n, v in enumerate(levels)
    ]


def predict(gold: list[dict], levels: list) -> dict:
    return {
        g["id"]: answer(g["id"], level)
        for g, level in zip(gold, levels)
        if level is not None
    }


class KeysTest(unittest.TestCase):
    def setUp(self):
        self.group = score5t.ledger_groups(TEST_SEED, 3)[0]
        self.base = self.group[0]

    def test_k1_ignores_event_ids_and_row_order(self):
        state, questions = self.base["state"], self.base["questions"]
        renamed = {
            **state,
            "events": [
                {**event, "id": f"Z{n}"}
                for n, event in enumerate(reversed(state["events"]))
            ],
        }
        self.assertEqual(score5t.ledger_key(state), score5t.ledger_key(renamed))
        self.assertEqual(
            score5t.ledger_key(state, questions), score5t.ledger_key(renamed, questions)
        )
        self.assertNotEqual(
            score5t.payload_key(state, questions),
            score5t.payload_key(renamed, questions),
        )
        keys = [score5t.ledger_key(i["state"], i["questions"]) for i in self.group]
        self.assertEqual(keys[0], keys[2])
        self.assertEqual(keys[0], keys[3])
        self.assertNotEqual(keys[0], keys[1])

    def test_k1_and_k2_change_with_the_removal_posted_flag(self):
        state = self.base["state"]
        flipped = {
            **state,
            "events": [
                {**e, "posted": not e["posted"]} if e["tick"] == 20 else e
                for e in state["events"]
            ],
        }
        self.assertNotEqual(score5t.ledger_key(state), score5t.ledger_key(flipped))
        k2, k2_flipped = score5t.structure_key(state), score5t.structure_key(flipped)
        self.assertEqual(k2[:3], k2_flipped[:3])
        self.assertEqual(k2[3], not k2_flipped[3])

    def test_question_part_and_non_ledgers(self):
        state, questions = self.base["state"], self.base["questions"]
        other = {"decision": {**questions["decision"], "instructions": "Other."}}
        self.assertNotEqual(
            score5t.ledger_key(state), score5t.ledger_key(state, questions)
        )
        self.assertNotEqual(
            score5t.ledger_key(state, questions), score5t.ledger_key(state, other)
        )
        for bad in (
            {"initial": 1, "capacity": 4},
            {**state, "initial": True},
            {**state, "events": [{"tick": 10, "kind": "add"}]},
            json.dumps(state),
            [state],
        ):
            self.assertIsNone(score5t.ledger_state(bad))

    def test_ledgers_in_nested_rows_and_embedded_json(self):
        state, questions = self.base["state"], self.base["questions"]
        row = {
            "messages": [{"role": "user", "content": model_text(self.base)}],
            "meta": {"items": [{"state": state, "questions": questions}]},
            "blob": json.dumps({"state": json.dumps(state)}),
        }
        found = list(score5t.ledgers(row))
        self.assertEqual(len(found), 3)
        self.assertEqual([q is not None for _, q in found], [False, True, False])
        k0, k1, count = score5t.row_keys(row)
        self.assertEqual(count, 3)
        self.assertEqual(k1, {score5t.ledger_key(state)})
        self.assertEqual(k0, {self.base["provenance"]["payload_sha256"]})

    def test_adapter_text_ignores_the_id(self):
        prompt = prompt_record(self.base)
        renamed = {**prompt, "id": score5t.panel_id(prompt["id"])}
        text = score5t.native_inputs()
        self.assertEqual(text(prompt), text(renamed))
        prefix, options, suffix = text(prompt)[0]
        self.assertTrue(prefix.startswith("Context:\n{"))
        self.assertIn(score5t.LEDGER_INSTRUCTIONS, prefix)
        self.assertEqual(len(options), 5)
        self.assertIn('"description":"Final amount is 0 units."', options[0])
        self.assertTrue(suffix.endswith("Decision:"))
        changed = self.group[1]
        self.assertNotEqual(text(prompt_record(changed)), text(prompt))

    def test_seed_generator_and_panel_id_rules(self):
        self.assertEqual(
            score5t.SEED_NAMES,
            {
                "fit": "decision2-score5t-dev-v1/fit",
                "check": "decision2-score5t-dev-v1/check",
            },
        )
        self.assertEqual(score5t.seed_bytes("x"), hashlib.sha256(b"x").digest())
        code = Path(score5t.generator.__file__).read_bytes()
        self.assertEqual(sha_bytes(code), score5t.FINAL_GENERATOR_SHA256)
        self.assertEqual(
            score5t.panel_id("td_x"),
            "score5t-" + hashlib.sha256(b"score5t-dev-v1:id:td_x").hexdigest()[:16],
        )


class SelectTest(unittest.TestCase):
    def test_whole_groups_in_index_order_with_reasons_and_abort(self):
        groups = score5t.ledger_groups(TEST_SEED, 12)
        self.assertEqual(sorted(groups), list(range(12)))
        keys = empty_keys()
        keys["k1"][score5t.ledger_key(groups[1][1]["state"])].add("final-k1")
        label = groups[3][3]
        keys["k0"][score5t.payload_key(label["state"], label["questions"])].add("dev")
        keys["k1"][score5t.ledger_key(groups[3][0]["state"])].add("select")

        def hit(group):
            return any(
                score5t.payload_key(i["state"], i["questions"]) in keys["k0"]
                or score5t.ledger_key(i["state"]) in keys["k1"]
                for i in group
            )

        expected = [i for i in range(12) if not hit(groups[i])][:5]
        accepted, log = score5t.select_groups(groups, keys, 5)
        self.assertEqual(
            [g[0]["provenance"]["instance_index"] for g in accepted], expected
        )
        self.assertTrue(all(len(g) == 4 for g in accepted))
        self.assertEqual(
            [e["instance_index"] for e in log], list(range(expected[-1] + 1))
        )
        entries = {e["instance_index"]: e for e in log}
        self.assertFalse(entries[1]["accepted"])
        self.assertIn("final-k1", entries[1]["reasons"])
        self.assertLessEqual({"dev", "select"}, set(entries[3]["reasons"]))
        for entry in log:
            self.assertEqual(
                entry["reasons"], sorted(entry["reasons"], key=score5t.REASONS.index)
            )
            self.assertEqual(entry["accepted"], not entry["reasons"])
            self.assertEqual(len(entry["k1"]), 2)
        taken = {i["id"] for g in accepted for i in g}
        self.assertFalse(taken & {i["id"] for i in groups[1] + groups[3]})
        with self.assertRaises(ValueError):
            score5t.select_groups(groups, keys, 11)


class SummaryTest(unittest.TestCase):
    def test_collapse_at_top_share_090_exactly_and_warn_just_below(self):
        levels = [4] * 90 + [0, 1, 2, 3] * 2 + [0, 1]
        gold = gold_rows(levels)
        out = score5t.summary(gold, predict(gold, levels), replicates=300)
        self.assertEqual(out["top_category"], "4")
        self.assertEqual(out["top_share"], 0.9)
        self.assertEqual(out["flags"], ["COLLAPSE"])
        self.assertEqual(out["always_majority_level"], 4)
        self.assertAlmostEqual(out["acc_minus_majority"], 0.1)
        below = [3] + levels[1:]
        out = score5t.summary(gold, predict(gold, below), replicates=300)
        self.assertEqual(out["top_share"], 0.89)
        self.assertGreaterEqual(out["top_share_wilson95"][1], score5t.TOP_SHARE)
        self.assertEqual(out["flags"], ["WARN"])

    def test_collapse_via_the_accuracy_wilson_lower_bound(self):
        levels = [0, 1, 2, 3, 4] * 20
        gold = gold_rows(levels)
        preds = [v if n < 20 else (v + 1) % 5 for n, v in enumerate(levels)]
        out = score5t.summary(gold, predict(gold, preds), replicates=300)
        self.assertEqual(out["top_share"], 0.2)
        self.assertEqual(out["accuracy"], 0.2)
        self.assertLessEqual(out["accuracy_wilson95"][0], score5t.CHANCE)
        self.assertEqual(out["always_majority_level"], 0)
        self.assertEqual(out["flags"], ["COLLAPSE", "NO-GAIN"])

    def test_warn_via_top_share_upper_bound_depends_on_n(self):
        levels = [4] * 85 + [0, 1, 2, 3] * 3 + [0, 1, 2]
        gold = gold_rows(levels)
        out = score5t.summary(gold, predict(gold, levels), replicates=300)
        self.assertEqual(out["top_share"], 0.85)
        self.assertEqual(out["flags"], ["WARN"])
        many = gold_rows(levels * 4)
        out = score5t.summary(many, predict(many, levels * 4), replicates=100)
        self.assertLess(out["top_share_wilson95"][1], score5t.TOP_SHARE)
        self.assertEqual(out["flags"], [])

    def test_no_gain_over_always_majority(self):
        levels = [4] * 40 + [0, 1, 2, 3] * 15
        gold = gold_rows(levels)
        preds = [v if n < 42 else (v + 1) % 5 for n, v in enumerate(levels)]
        out = score5t.summary(gold, predict(gold, preds), replicates=500)
        self.assertAlmostEqual(out["accuracy"], 0.42)
        self.assertAlmostEqual(out["always_majority_accuracy"], 0.40)
        self.assertLessEqual(out["acc_minus_majority_boot95"][0], 0)
        self.assertEqual(out["flags"], ["NO-GAIN"])

    def test_invalid_and_missing_answers_are_their_own_category(self):
        levels = [0, 1, 2, 3, 4] * 20
        gold = gold_rows(levels)
        preds = {g["id"]: answer(g["id"], 9) for g in gold[5:55]}
        preds.update(predict(gold[:5], levels[:5]))
        out = score5t.summary(gold, preds, replicates=200)
        self.assertEqual(out["histogram"]["invalid"], 95)
        self.assertEqual(out["invalid_or_missing"], 95)
        self.assertEqual(out["top_category"], "invalid")
        self.assertEqual(out["top_share"], 0.95)
        self.assertEqual(out["modal_level"], 0)
        self.assertEqual(out["rare_levels"], [0, 1, 2, 3, 4])
        self.assertEqual(out["correct"], 5)
        self.assertEqual(out["flags"], ["COLLAPSE", "NO-GAIN"])
        self.assertAlmostEqual(out["gold_l4_share"], 0.2)
        self.assertAlmostEqual(out["l4_share"], 0.01)

    def test_bootstrap_and_majority_mirror_m7(self):
        m7 = importlib.import_module("v2.06b.m7_scorebias")
        self.assertEqual((m7.SEED, m7.DRAWS), (score5t.BOOT_SEED, score5t.REPLICATES))
        correct = [int(n % 3 != 0) for n in range(40)]
        always = [int(n % 4 == 0) for n in range(40)]
        accuracy, gain = score5t.bootstrap(correct, always, score5t.REPLICATES)
        self.assertEqual(accuracy, m7.bootstrap([float(c) for c in correct]))
        self.assertEqual(gain, m7.bootstrap([c - a for c, a in zip(correct, always)]))
        for gold in ([4, 4, 1, 1, 0], [2, 3, 3, 2], [1]):
            self.assertEqual(score5t.majority_level(gold), m7.majority_level(gold))

    def test_flag_thresholds(self):
        self.assertEqual(score5t.flags(0.90, 0.95, 0.5, 0.1), ["COLLAPSE"])
        self.assertEqual(score5t.flags(0.89, 0.90, 0.5, 0.1), ["WARN"])
        self.assertEqual(score5t.flags(0.89, 0.899, 0.5, 0.1), [])
        self.assertEqual(score5t.flags(0.5, 0.6, 0.20, 0.1), ["COLLAPSE"])
        self.assertEqual(score5t.flags(0.5, 0.6, 0.21, 0.0), ["NO-GAIN"])
        self.assertEqual(score5t.flags(0.95, 1.0, 0.1, -0.1), ["COLLAPSE", "NO-GAIN"])

    def test_blocks_split_by_half_and_refuse_unknown_ids(self):
        gold = gold_rows([4, 3, 2, 1] * 5, "fit") + gold_rows([0, 1] * 10, "check")
        levels = [g["gold"]["decision"]["value"] for g in gold]
        out = score5t.blocks(gold, predict(gold, levels), replicates=50)
        self.assertEqual([out[b]["n"] for b in score5t.BLOCKS], [40, 20, 20])
        self.assertEqual(out["check"]["gold_histogram"]["4"], 0)
        self.assertEqual(out["fit"]["accuracy"], 1.0)
        with self.assertRaises(ValueError):
            score5t.summary(gold, {"x": answer("x", 1)}, replicates=10)


class ValidateCriteriaTest(unittest.TestCase):
    def block(self, **overrides) -> dict:
        rows = {
            "m7-mx-soup": (0.99, ["COLLAPSE"]),
            "m6-mxcx-soup": (0.95, ["COLLAPSE", "NO-GAIN"]),
            "m7-mxcx-soup": (0.85, ["WARN"]),
            "m6-mxcxa-soup": (0.80, []),
            "m4-t-a7-soup": (0.60, ["NO-GAIN"]),
            "kai1": (0.55, []),
        }
        rows.update(overrides)
        return {
            name: {"top_share": share, "flags": flags}
            for name, (share, flags) in rows.items()
            if share is not None
        }

    def test_pass_and_each_criterion_failing(self):
        ok = score5t.criteria(self.block())
        self.assertEqual((ok["V1"], ok["V2"], ok["V3"], ok["pass"]), (True,) * 4)
        self.assertEqual(ok["missing"], [])
        v1 = score5t.criteria(self.block(**{"m7-mx-soup": (0.99, ["WARN"])}))
        self.assertEqual((v1["V1"], v1["V3"], v1["pass"]), (False, True, False))
        v2 = score5t.criteria(self.block(**{"m6-mxcxa-soup": (0.80, ["COLLAPSE"])}))
        self.assertEqual((v2["V2"], v2["pass"]), (False, False))
        tie = score5t.criteria(self.block(**{"m7-mxcx-soup": (0.80, [])}))
        self.assertEqual((tie["V1"], tie["V2"], tie["V3"]), (True, True, False))
        gone = score5t.criteria(self.block(**{"m4-t-a7-soup": (None, [])}))
        self.assertEqual((gone["V2"], gone["V3"], gone["pass"]), (False, False, False))
        self.assertEqual(gone["missing"], ["m4-t-a7-soup"])

    def test_agreement_and_kai1_position(self):
        block = self.block()
        out = score5t.agreement(block)
        self.assertEqual(len(out["models"]), 6)
        self.assertAlmostEqual(out["kendall_tau_b"], 1.0)
        self.assertAlmostEqual(out["spearman_rho"], 1.0)
        self.assertAlmostEqual(
            out["panel_minus_final_top_share"]["m4-t-a7-soup"], -0.0425
        )
        self.assertEqual(score5t.kai1_position(block)["rank"], 6)
        self.assertTrue(score5t.kai1_position(block)["lowest"])
        high = self.block(kai1=(0.97, ["COLLAPSE"]))
        self.assertEqual(score5t.kai1_position(high)["rank"], 2)
        self.assertLess(score5t.agreement(high)["kendall_tau_b"], 1.0)
        self.assertIsNone(score5t.kai1_position(self.block(kai1=(None, []))))


class ScanTest(unittest.TestCase):
    def test_scan_finds_ledger_rows_and_counts_loose_rows(self):
        items = [g[0] for g in score5t.ledger_groups(TEST_SEED, 4).values()]
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            write_jsonl(
                corpus / "a.jsonl",
                [
                    prompt_record(items[0]),
                    {"messages": [{"role": "user", "content": model_text(items[1])}]},
                    {"text": "posted within capacity"},
                ],
            )
            (corpus / "b.json").write_text(
                json.dumps({"items": [prompt_record(items[2])]}, indent=2)
            )
            write_jsonl(corpus / "c.jsonl", [{"note": score5t.LEDGER_INSTRUCTIONS}])
            (corpus / "d.jsonl.gz").write_bytes(
                gzip.compress((json.dumps(prompt_record(items[3])) + "\n").encode())
            )
            (corpus / "e.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd")
            (corpus / "empty.jsonl").write_bytes(b"")
            write_jsonl(
                corpus / "f.jsonl", [{"t": "posted capacity"}] * 3 + [{"t": "posted"}]
            )
            names = ("a.jsonl", "b.json", "c.jsonl", "d.jsonl.gz", "e.jsonl.zst")
            files = [
                corpus / n for n in (*names, "empty.jsonl", "f.jsonl", "gone.jsonl")
            ]
            manifest = manifest_file(
                Path(tmp) / "manifest.json", {"x": files, "y": [corpus / "a.jsonl"]}
            )
            outs = []
            for workers in (1, 2):
                out = Path(tmp) / f"scan{workers}.json"
                with mock.patch("sys.stdout"):
                    score5t.main(
                        [
                            "scan",
                            "--manifest",
                            str(manifest),
                            "--output",
                            str(out),
                            "--workers",
                            str(workers),
                        ]
                    )
                self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)
                outs.append(json.loads(out.read_text()))
            one, two = outs
            for report in outs:
                report.pop("runtime_seconds")
                report.pop("workers")
            self.assertEqual(one, two)
            self.assertEqual(one["manifest"]["entries"], 9)
            self.assertEqual(one["manifest"]["files"], 8)
            self.assertEqual(one["files_scanned"], 6)
            self.assertEqual(one["missing"], [str(corpus / "gone.jsonl")])
            self.assertEqual(one["skipped_compressed"], [str(corpus / "e.jsonl.zst")])
            self.assertEqual(one["errors"], [])
            self.assertEqual(one["size_mismatch"], [])
            hits = {Path(h["path"]).name: h for h in one["hits"]}
            self.assertEqual(
                sorted(hits), ["a.jsonl", "b.json", "c.jsonl", "d.jsonl.gz"]
            )
            self.assertEqual(hits["a.jsonl"]["signature_rows"], 2)
            self.assertEqual(hits["a.jsonl"]["ledger_states"], 2)
            self.assertEqual(hits["a.jsonl"]["labels"], ["x", "y"])
            self.assertEqual(hits["b.json"]["signature_rows"], 2)
            self.assertEqual(hits["b.json"]["ledger_states"], 1)
            self.assertEqual(hits["c.jsonl"]["unparsed"], 1)
            self.assertEqual(one["unparsed"], 1)
            self.assertEqual(one["signature_rows"], 6)
            self.assertEqual(
                one["signature_rows_by"], {"instructions": 5, "criterion": 4}
            )
            self.assertEqual(
                set(one["k0"]),
                {items[i]["provenance"]["payload_sha256"] for i in (0, 2, 3)},
            )
            self.assertEqual(
                set(one["k1"]), {score5t.ledger_key(i["state"]) for i in items}
            )
            loose = {Path(r["path"]).name: r["rows"] for r in one["loose_top_files"]}
            self.assertEqual(loose["a.jsonl"], 3)
            self.assertEqual(loose["f.jsonl"], 3)
            self.assertEqual(loose["b.json"], 1)
            self.assertEqual(one["by_label"]["y"]["signature_rows"], 2)

    def test_scan_refuses_sealed_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = {"path": score5t.SEALED_ROOT + "c1/rows.jsonl", "bytes": 1}
            manifest = Path(tmp) / "m.json"
            manifest.write_text(
                json.dumps({"labels": {"x": {"kind": "training", "files": [entry]}}})
            )
            args = ["scan", "--manifest", str(manifest), "--output", f"{tmp}/s.json"]
            with self.assertRaises(SystemExit):
                score5t.main(args + ["--workers", "1"])
            self.assertFalse(Path(tmp, "s.json").exists())


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.halves = {
            half: score5t.ledger_groups(score5t.seed_bytes(name), INDICES)
            for half, name in TEST_SEED_NAMES.items()
        }
        fit, check = self.halves["fit"], self.halves["check"]
        panels_root = self.root / "panels"
        final = generate("final", b"f" * 32, 20)
        final_prompts = [prompt_record(i) for i in final]
        final_prompts.append({**prompt_record(fit[0][3]), "id": "td_forced"})
        final_gold = final + [{"id": "td_forced", "group_id": "g_forced"}]
        dev = generate("dev", b"d" * 32, 5)
        dev_prompts = [prompt_record(i) for i in dev]
        dev_prompts.append({**prompt_record(check[1][0]), "id": "td_dev_forced"})
        dev_gold = dev + [{"id": "td_dev_forced", "group_id": "g_dev_forced"}]
        registry = {}
        for name, prompts, gold in (
            ("typed-final", final_prompts, final_gold),
            ("typed-dev", dev_prompts, dev_gold),
        ):
            spec = dict(panels.ALL[name])
            spec["prompts_sha256"] = write_jsonl(panels_root / spec["prompts"], prompts)
            spec["gold_sha256"] = write_jsonl(panels_root / spec["gold"], gold)
            registry[name] = spec

        def training_row(n: int, state) -> dict:
            return {
                "id": f"r{n}",
                "family": "f",
                "task_type": "score",
                "state": state,
                "instructions": "Rate.",
                "options": [{"key": "0", "description": "a"}] * 2,
                "label": 0,
            }

        data = self.root / "data"
        plain = [training_row(n, f"text {n}") for n in range(5)]
        ledger_text = "Ledger:\n" + json.dumps(fit[2][1]["state"])
        nested = {**training_row(9, "x"), "audit": {"source": check[3][2]["state"]}}
        self.sources = {
            "select": write_jsonl(
                data / "select.jsonl", plain + [training_row(8, ledger_text)]
            ),
            "cal": write_jsonl(data / "cal.jsonl", plain),
            "cal698": write_jsonl(data / "cal698.jsonl", plain + [nested]),
        }
        write_jsonl(data / "corpus" / "t.jsonl", [prompt_record(fit[4][0]), {"t": "x"}])
        corpora = manifest_file(
            self.root / "corpora.json", {"train": [data / "corpus" / "t.jsonl"]}
        )
        self.scan = self.root / "scan.json"
        with mock.patch("sys.stdout"):
            score5t.main(
                [
                    "scan",
                    "--manifest",
                    str(corpora),
                    "--output",
                    str(self.scan),
                    "--workers",
                    "1",
                ]
            )
        self.panels_root = panels_root
        self.registry = registry
        self.patches = [
            mock.patch.multiple(
                score5t,
                SEED_NAMES=TEST_SEED_NAMES,
                GROUPS_PER_HALF=GROUPS,
                MAX_INDICES=INDICES,
            ),
            mock.patch.dict(panels.ALL, registry),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def build_args(self, out: Path, **shas) -> list[str]:
        data = self.root / "data"
        args = [
            "build",
            "--output-dir",
            str(out),
            "--panels-root",
            str(self.panels_root),
            "--training-scan",
            str(self.scan),
        ]
        for name in ("select", "cal", "cal698"):
            args += [f"--{name}", str(data / f"{name}.jsonl")]
            args += [f"--{name}-sha256", shas.get(name, self.sources[name])]
        return args

    def test_build_writes_the_panel_manifest_and_audit_trail(self):
        out = self.root / "build"
        with mock.patch("sys.stdout"):
            score5t.main(self.build_args(out))
        prompts = read_jsonl(out / "score5t-dev.prompts.jsonl")
        gold = read_jsonl(out / "score5t-dev.gold.jsonl")
        manifest = json.loads((out / "MANIFEST.json").read_text())
        report = json.loads((out / "exclusion-report.json").read_text())
        log = read_jsonl(out / "groups.jsonl")
        n = 2 * GROUPS * 4
        self.assertEqual(len(prompts), n)
        self.assertTrue(all(set(p) == {"id", "state", "questions"} for p in prompts))
        self.assertTrue(all(p["id"].startswith("score5t-") for p in prompts))
        self.assertEqual([p["id"] for p in prompts], [g["id"] for g in gold])
        self.assertEqual(
            [g["half"] for g in gold], ["fit"] * (n // 2) + ["check"] * (n // 2)
        )
        self.assertEqual({g["split"] for g in gold}, {"fit", "check"})
        self.assertEqual({g["provenance"]["generator_split"] for g in gold}, {"final"})
        self.assertEqual(
            [g["provenance"]["variant"] for g in gold[:4]], list(score5t.VARIANTS)
        )
        indices = [
            g["provenance"]["instance_index"] for g in gold if g["half"] == "fit"
        ]
        self.assertEqual(indices, sorted(indices))
        first = gold[0]
        self.assertEqual(first["cluster_id"], first["group_id"])
        self.assertEqual(first["task"], "score5t/resource_ledger")
        self.assertEqual(first["gold"]["decision"]["type"], "score")
        self.assertEqual(first["questions"], score5t.LEDGER_QUESTIONS)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(out / "score5t-dev.gold.jsonl").st_mode & 0o777, 0o600)
        by_half = {
            half: read_jsonl(out / f"score5t-dev.{half}.gold.jsonl")
            for half in TEST_SEED_NAMES
        }
        self.assertEqual(by_half["fit"] + by_half["check"], gold)

        forced = {
            ("fit", 0): "final-k0",
            ("fit", 2): "select",
            ("fit", 4): "training",
            ("check", 1): "dev",
            ("check", 3): "cal698",
        }
        entries = {(e["half"], e["instance_index"]): e for e in log}
        for key, reason in forced.items():
            self.assertIn(reason, entries[key]["reasons"])
            self.assertFalse(entries[key]["accepted"])
        self.assertIn("final-k1", entries[("fit", 0)]["reasons"])
        accepted = {(g["half"], g["provenance"]["instance_index"]) for g in gold}
        self.assertEqual(accepted, {k for k, e in entries.items() if e["accepted"]})
        for half in TEST_SEED_NAMES:
            stats = manifest["halves"][half]
            self.assertEqual(stats["groups_accepted"], GROUPS)
            self.assertEqual(
                stats["indices_walked"], sum(e["half"] == half for e in log)
            )
            self.assertEqual(sum(stats["gold_levels"].values()), GROUPS * 4)
            self.assertEqual(stats["variants"], {v: GROUPS for v in score5t.VARIANTS})
        self.assertGreaterEqual(
            manifest["halves"]["fit"]["dropped_by_reason"]["training"], 1
        )
        self.assertEqual(
            report["drops"]["fit"]["dropped_by_reason"],
            manifest["halves"]["fit"]["dropped_by_reason"],
        )
        sources = report["sources"]
        self.assertEqual(sources["final"]["instructions_rows"], 81)
        self.assertEqual(sources["final"]["ledger_rows"], 81)
        self.assertEqual(sources["final"]["ledger_rows_with_panel_question"], 81)
        self.assertEqual(sources["dev"]["ledger_rows"], 1)
        self.assertEqual(sources["select"]["ledger_rows"], 1)
        self.assertEqual(sources["select"]["k0"], 0)
        self.assertEqual(sources["cal"]["ledger_rows"], 0)
        self.assertEqual(sources["training"]["signature_rows"], 1)
        self.assertEqual(manifest["items"], n)
        self.assertEqual(
            manifest["generator"]["sha256"], score5t.FINAL_GENERATOR_SHA256
        )
        self.assertEqual(
            manifest["seeds"]["fit"]["commitment_sha256"],
            hashlib.sha256(score5t.seed_bytes("test-score5t/fit")).hexdigest(),
        )
        self.assertEqual(manifest["adapter_input_check"]["items_identical"], n)
        self.assertEqual(
            manifest["overlap"]["final_k2_distinct"], sources["final"]["k2"]
        )
        self.assertLessEqual(
            manifest["overlap"]["fit"]["k1_group_states_distinct"], GROUPS
        )
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(sha_bytes((out / name).read_bytes()), digest)
        questions = leak_audit.load_panel(
            self.root,
            "score5t-dev",
            (out / "score5t-dev.prompts.jsonl", out / "score5t-dev.gold.jsonl"),
        )
        self.assertEqual(len(questions), n)
        self.assertEqual(questions[0].gold, first["gold"]["decision"]["value"])
        with self.assertRaises(FileExistsError), mock.patch("sys.stdout"):
            score5t.main(self.build_args(out))

    def test_build_refuses_changed_sources_or_generator(self):
        with self.assertRaises(SystemExit):
            score5t.main(self.build_args(self.root / "b1", select="0" * 64))
        with mock.patch.object(score5t, "FINAL_GENERATOR_SHA256", "0" * 64):
            with self.assertRaises(SystemExit):
                score5t.main(self.build_args(self.root / "b2"))
        with mock.patch.object(score5t, "MAX_INDICES", GROUPS + 1):
            with self.assertRaises(ValueError):
                score5t.main(self.build_args(self.root / "b3"))
        self.assertFalse((self.root / "b1").exists())

    def test_validate_against_sealed_runs(self):
        out = self.root / "build"
        with mock.patch("sys.stdout"):
            score5t.main(self.build_args(out))
        root = self.root / "registered"
        spec = {
            "prompts": "goldfree/score5t-dev.prompts.jsonl",
            "gold": "gold/score5t-dev.gold.jsonl",
        }
        for kind in ("prompts", "gold"):
            target = root / spec[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((out / f"score5t-dev.{kind}.jsonl").read_bytes())
            spec[f"{kind}_sha256"] = sha_bytes(target.read_bytes())
        prompts = {p["id"]: p for p in read_jsonl(root / spec["prompts"])}
        gold = read_jsonl(root / spec["gold"])
        order = sorted(gold, key=lambda g: g["gold"]["decision"]["value"] != 4)
        n = len(gold)

        def rows(fours: int, zeros: int = 0) -> list[dict]:
            four = {g["id"] for g in order[:fours]}
            zero = {g["id"] for g in gold[:zeros]}
            out = []
            for g in gold:
                level = (
                    0
                    if g["id"] in zero
                    else 4 if g["id"] in four else g["gold"]["decision"]["value"]
                )
                p = prompts[g["id"]]
                out.append(
                    {
                        **answer(g["id"], level),
                        "source_input_sha256": input_digest(p["state"], p["questions"]),
                    }
                )
            return out

        plan = {
            "m7-mx-soup": rows(n),
            "m6-mxcx-soup": rows(n - 2),
            "m7-mxcx-soup": rows(40),
            "m6-mxcxa-soup": rows(36),
            "m4-t-a7-soup": rows(30),
            "kai1": rows(0, 20),
            "m4-t-a7-soup-cache": rows(30),
        }
        args = ["validate", "--panels-root", str(root), "--replicates", "50"]
        for name, predictions in plan.items():
            run = self.root / "runs" / name
            write_jsonl(run / "output" / "score5t-dev.predictions.jsonl", predictions)
            with mock.patch("sys.stdout"):
                htdev_score.main(
                    [
                        "seal",
                        "--prompts",
                        str(root / spec["prompts"]),
                        "--predictions",
                        str(run / "output" / "score5t-dev.predictions.jsonl"),
                        "--output",
                        str(run / score5t.SEAL),
                    ]
                )
            args += ["--run", f"{name}={run}"]
        result_path = self.root / "validate.json"
        with mock.patch.dict(panels.ALL, {"score5t-dev": spec}), mock.patch(
            "sys.stdout"
        ):
            score5t.main(args + ["--output", str(result_path)])
        result = json.loads(result_path.read_text())
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["criteria"]["missing"], [])
        self.assertEqual(result["secondary"]["extra_runs"], ["m4-t-a7-soup-cache"])
        self.assertTrue(result["secondary"]["kai1"]["lowest"])
        self.assertEqual(len(result["secondary"]["agreement"]["full"]["models"]), 6)
        self.assertEqual(set(result["secondary"]["per_half"]), {"fit", "check"})
        full = result["runs"]["m7-mx-soup"]["blocks"]["full"]
        self.assertEqual(
            (full["n"], full["top_category"], full["top_share"]), (n, "4", 1.0)
        )
        self.assertEqual(result["runs"]["m4-t-a7-soup"]["blocks"]["fit"]["n"], n // 2)

        tampered = (
            self.root / "runs" / "kai1" / "output" / "score5t-dev.predictions.jsonl"
        )
        tampered.write_text(tampered.read_text().replace('"score": 0', '"score": 1', 1))
        with mock.patch.dict(panels.ALL, {"score5t-dev": spec}), mock.patch(
            "sys.stdout"
        ):
            with self.assertRaises(ValueError):
                score5t.main(args + ["--output", str(self.root / "validate2.json")])
        with self.assertRaises(SystemExit):
            score5t.main(args + ["--output", str(self.root / "validate3.json")])


if __name__ == "__main__":
    unittest.main()
