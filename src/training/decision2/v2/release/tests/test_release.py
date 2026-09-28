"""CPU/stdlib tests for the Decision 2.0 release pipeline (no torch, no network)."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from v2.release import build, card, examples, layout, licence
from v2.release.runtime import api

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "v2/eval/records/m1-reports"
ROSTER = ROOT / "v2/eval/records/decision-index-peer-roster-2026-09-28.json"


def fake_safetensors(
    path: Path, shapes: dict[str, list[int]], payload: bytes = b""
) -> None:
    header = {
        name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in shapes.items()
    }
    data = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(data)) + data + payload)


class LayoutTest(unittest.TestCase):
    def test_tiers_follow_loaded_parameters(self):
        self.assertEqual(layout.name_for(571_909_635), "DEV2.0-0.6B")
        self.assertEqual(layout.name_for(753_446_208), "DEV2.0-0.8B")
        self.assertEqual(layout.name_for(7_940_895_744), "DEV2.0-9B")
        self.assertEqual(layout.name_for(25_688_227_840), "DEV2.0-27B")
        self.assertIsNone(layout.tier_for(1_300_000_000))
        with self.assertRaises(ValueError):
            layout.name_for(1_300_000_000)

    def test_repository_rules(self):
        layout.check_repo(
            "llm-semantic-router/dev2-release-staging", "DEV2.0-0.6B", staging=True
        )
        layout.check_repo("llm-semantic-router/DEV2.0-4B", "DEV2.0-4B", staging=False)
        for repo, name, staging in (
            ("llm-semantic-router/DEV2.0-4B", "DEV2.0-4B", True),
            ("llm-semantic-router/dev2-release-staging", "DEV2.0-4B", False),
            ("someone/DEV2.0-4B", "DEV2.0-4B", False),
            ("llm-semantic-router/DEV2.0-4B", "DEV2.0-9B", False),
        ):
            with self.assertRaises(ValueError):
                layout.check_repo(repo, name, staging=staging)

    def test_pointer_lists_only_present_files(self):
        files = [
            "decision_config.json",
            "decision_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "backbone/config.json",
            "backbone/model.safetensors",
        ]
        value = layout.pointer(
            "qwen-full",
            "DEV2.0-4B",
            files,
            calibration="calibration.json",
            base=None,
            max_input_tokens=8192,
        )
        self.assertEqual(
            (value["decision_format"], value["format_version"]), ("vllm-sr-decision", 2)
        )
        self.assertEqual(value["backbone"]["weights"], ["backbone/model.safetensors"])
        self.assertEqual(
            value["decision_weights"], {"decision_head": "decision_head.safetensors"}
        )
        self.assertNotIn("chat_template", value["tokenizer"])


class LicenceTest(unittest.TestCase):
    def setUp(self):
        self.roster = licence.load_roster(ROSTER)

    def check(self, repo, family="peer", label=None):
        return licence.card_eligibility(
            {"repo_id": repo, "family": family, "label": label}, self.roster
        )

    def test_card_filter_excludes_nc_research_unknown_and_internal(self):
        self.assertFalse(self.check("kirp/jpt-9b")["eligible"])
        self.assertFalse(self.check("HopitAI/hopper-g")["eligible"])
        self.assertFalse(self.check("unknown/model")["eligible"])
        self.assertFalse(
            self.check(None, "decision2", "DEV2.0-0.6B (private)")["eligible"]
        )
        self.assertTrue(self.check("Hanno-Labs/bosun-v3.1-0.6b")["eligible"])
        self.assertTrue(
            self.check("llm-semantic-router/Decision-1.0-Kai-0.6B", "decision1")[
                "eligible"
            ]
        )

    def test_package_licence_requires_compatible_lineage(self):
        self.assertEqual(
            licence.package_licence(
                [
                    {"component": "a", "licence": "apache-2.0"},
                    {"component": "b", "licence": "MIT"},
                ]
            )["spdx"],
            "apache-2.0",
        )
        mixed = licence.package_licence(
            [
                {"component": "a", "licence": "apache-2.0"},
                {"component": "tokenizer", "licence": "gemma"},
            ]
        )
        self.assertEqual((mixed["spdx"], mixed["apache_compatible"]), ("other", False))


def card_entries():
    return [
        {
            "key": "cand",
            "role": "candidate",
            "report": str(REPORTS / "lex.json"),
            "label": "DEV2.0-0.6B",
        },
        {
            "key": "kai1",
            "role": "own-1.0",
            "report": str(REPORTS / "kai1.json"),
            "repo_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
            "label": "Decision 1.0 Kai",
        },
        {
            "key": "bosun",
            "role": "peer",
            "report": str(REPORTS / "bosun.json"),
            "repo_id": "Hanno-Labs/bosun-v3.1-0.6b",
        },
        {
            "key": "gliner",
            "role": "peer",
            "report": str(REPORTS / "gliner25.json"),
            "repo_id": "fastino/GLiNER2.5-Decide",
        },
        {"key": "old", "role": "peer", "report": str(REPORTS / "dev20-06b.json")},
        {
            "key": "jpt",
            "role": "peer",
            "report": str(REPORTS / "jpt08b.json"),
            "repo_id": "kirp/jpt-0.8b",
        },
    ]


def facts(licence_spdx="apache-2.0"):
    return {
        "model_name": "DEV2.0-0.6B",
        "repo_id": "llm-semantic-router/dev2-release-staging",
        "profile": "kai-native",
        "parameters": {
            "loaded": 571_909_635,
            "components_text": "native encoder paths and heads 571,909,635",
        },
        "max_input_tokens": 8192,
        "origin": {
            "repo_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
            "revision": "7" * 40,
            "relation": "unchanged",
            "summary": "Weights unchanged.",
        },
        "licence": {
            "spdx": licence_spdx,
            "license_name": "decision-2.0-component-licences",
            "components": [],
        },
        "banner": "DEV2.0-0.6B-owl-banner.png",
        "calibration_text": "raw native probabilities.",
        "requirements_text": "Tested with Transformers 4.57.6.",
    }


class CardTest(unittest.TestCase):
    def test_card_from_same_panel_reports(self):
        with tempfile.TemporaryDirectory() as scratch:
            out, banner = Path(scratch) / "pkg", Path(scratch) / "banner.png"
            banner.write_bytes(b"\x89PNG\r\n\x1a\n")
            result = card.build_card(
                entries=card_entries(),
                roster=ROSTER,
                paired=None,
                facts=facts("other"),
                text={
                    "tagline": "A compact decision model.",
                    "staging_notice": "Staging dry run.",
                    "limitations": [],
                },
                banner=banner,
                work=Path(scratch) / "work",
                output=out,
            )
            readme = (out / "README.md").read_text()
            files = {
                p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
            }
            files |= {"LICENSE", "LICENSING.md", "NOTICE", "ATTRIBUTIONS.md"}
            self.assertEqual(card.check_rendered(readme, files), [])
            self.assertEqual({e["key"] for e in result["excluded"]}, {"old", "jpt"})
            self.assertIn(
                "Typed Choice (correct): 235/800 versus 277/800", result["tradeoffs"]
            )
            self.assertIn("| Typed Choice (correct) | 235/800 | 277/800 |", readme)
            self.assertIn("do not measure multilingual ability", readme)
            self.assertIn("license: other", readme)
            self.assertNotIn("Decision 2.0 collection", readme)
            for chart in layout.CHART_FILES:
                self.assertTrue((out / chart).is_file())
                self.assertNotIn("Pareto", (out / chart).read_text())
            code = readme.split("```python\n", 1)[1].split("```", 1)[0]
            compile(code, "card", "exec")
            self.assertIn('Decision2.from_pretrained("dev2-release-staging")', code)

    def test_card_needs_own_comparator(self):
        entries = [e for e in card_entries() if e["role"] != "own-1.0"]
        with self.assertRaises(ValueError):
            card.select_reports(entries, ROSTER)

    def test_lint(self):
        self.assertTrue(card.lint("A Pareto frontier"))
        self.assertTrue(card.lint("ran on node A"))
        self.assertTrue(card.lint("path /data/dev2/runs"))
        self.assertFalse(card.lint("Transformers 4.57.6 on one GPU"))


class ExamplesTest(unittest.TestCase):
    def test_compare_detects_drift(self):
        a = {
            "route": {
                "type": "choice",
                "choice": "x",
                "probabilities": {"x": 0.6, "y": 0.4},
            }
        }
        b = {
            "route": {
                "type": "choice",
                "choice": "x",
                "probabilities": {"x": 0.6000001, "y": 0.3999999},
            }
        }
        self.assertEqual(examples.compare_answers(a, a)["max_abs_drift"], 0.0)
        result = examples.compare_answers(a, b)
        self.assertEqual(result["category_changes"], 0)
        self.assertGreater(result["max_abs_drift"], 0)
        c = {
            "route": {
                "type": "choice",
                "choice": "y",
                "probabilities": {"x": 0.4, "y": 0.6},
            }
        }
        self.assertEqual(examples.compare_answers(a, c)["category_changes"], 1)

    def test_answer_checks(self):
        question = examples.EXAMPLES[0]["questions"]["urgency"]
        good = {
            "type": "score",
            "score": 1.2,
            "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
        }
        self.assertEqual(examples.answer_problems(question, good), [])
        self.assertTrue(
            examples.answer_problems(question, {"type": "score", "error": "x"})
        )

    def test_over_budget_example_is_long(self):
        state = examples.over_budget_example(100)["state"]
        self.assertGreater(len(state.split()), 200)


class BuildTest(unittest.TestCase):
    def test_screen_refuses_private_text(self):
        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            (stage / "README.md").write_text("see /data/dev2/runs/x")
            with self.assertRaises(ValueError):
                build.screen(stage)
            (stage / "README.md").write_text("token hf_" + "a" * 34)
            with self.assertRaises(ValueError):
                build.screen(stage)
            (stage / "README.md").write_text("fine")
            self.assertEqual(build.screen(stage)["text_files_screened"], 1)

    def test_qwen_full_build_and_verify(self):
        from training.model.data import canonical
        from training.model.infer import checkpoint_fingerprint

        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            ckpt = scratch / "ckpt"
            metadata = {
                "architecture": "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp",
                "prompt_version": "decision2-segmented-options-global-query-v1",
                "head_dim": 256,
                "checkpoint_format": "full",
                "text_parameter_count": 4_000_000_000,
            }
            (ckpt / "backbone").mkdir(parents=True)
            (ckpt / "decision_config.json").write_text(json.dumps(metadata))
            (ckpt / "backbone/config.json").write_text(
                json.dumps({"model_type": "qwen3_5_text"})
            )
            fake_safetensors(
                ckpt / "backbone/model.safetensors", {"w": [4_000_000_000]}
            )
            fake_safetensors(ckpt / "decision_head.safetensors", {"h": [1000, 256]})
            (ckpt / "tokenizer.json").write_text("{}")
            (ckpt / "tokenizer_config.json").write_text("{}")
            (ckpt / "trainer_state.pt").write_bytes(b"x")
            identity = checkpoint_fingerprint(ckpt)
            cal = scratch / "cal.json"
            cal.write_text(
                json.dumps(
                    {
                        "calibration_version": "decision2-per-type-temperature/1",
                        "model_sha256": identity["model_sha256"],
                        **{
                            k: "0" * 64
                            for k in (
                                "checkpoint_sha256",
                                "cal_sha256",
                                "best_sha256",
                                "complete_sha256",
                                "provenance_sha256",
                            )
                        },
                        "fit_split": "cal",
                        "selection_policy": "completed_run_best_only",
                        "temperature_by_type": {
                            "choice": 1.1,
                            "noul": 0.9,
                            "score": 1.3,
                        },
                        "inference": {"max_length": 8192},
                    }
                )
            )
            licence_file = scratch / "LICENSE"
            licence_file.write_text("Apache License 2.0\n")
            banner_dir = scratch / "brand"
            banner_dir.mkdir()
            (banner_dir / "DEV2.0-4B-owl-banner.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            entries = card_entries()
            spec = {
                "schema": build.SPEC_SCHEMA,
                "kind": "staging",
                "repo_id": "llm-semantic-router/dev2-release-staging",
                "model_name": "DEV2.0-4B",
                "profile": "qwen-full",
                "checkpoint": str(ckpt),
                "expected_identity": {"model_sha256": identity["model_sha256"]},
                "calibration": {"path": str(cal), "sha256": layout.sha_file(cal)},
                "max_input_tokens": 8192,
                "origin": {
                    "repo_id": "Qwen/Qwen3.5-4B-Base",
                    "revision": "a" * 40,
                    "relation": "finetune",
                    "summary": "Fine-tuned.",
                },
                "licence": {
                    "components": [
                        {"component": "Qwen3.5-4B-Base", "licence": "apache-2.0"}
                    ],
                    "files": [
                        {
                            "source": str(licence_file),
                            "path": "LICENSE",
                            "sha256": layout.sha_file(licence_file),
                        }
                    ],
                    "attributions": ["Qwen3.5-4B-Base (Apache-2.0)."],
                },
                "card": {
                    "reports": entries,
                    "roster": str(ROSTER),
                    "text": {"tagline": "t", "staging_notice": "Staging."},
                    "requirements_text": "Tested with Transformers 5.17.0.",
                },
            }
            spec_path = scratch / "spec.json"
            spec_path.write_text(json.dumps(spec))
            original = build.BRAND_DIR
            build.BRAND_DIR = banner_dir
            try:
                receipt = build.build(
                    spec_path, scratch / "out" / "dev2-release-staging"
                )
            finally:
                build.BRAND_DIR = original
            pkg = scratch / "out" / "dev2-release-staging"
            manifest = api.verify_bundle(pkg)
            self.assertEqual(manifest["parameters"]["loaded"], 4_000_256_000)
            self.assertEqual(
                manifest["identity"]["model_sha256"], identity["model_sha256"]
            )
            self.assertEqual(
                checkpoint_fingerprint(pkg)["model_sha256"], identity["model_sha256"]
            )
            self.assertNotIn("trainer_state.pt", manifest["files_sha256"])
            self.assertIn(
                "decision2/_vendor/dev2model/decision_model.py",
                manifest["files_sha256"],
            )
            self.assertEqual(receipt["licence"], "apache-2.0")
            pointer = json.loads((pkg / "config.json").read_text())
            self.assertEqual(
                pointer["calibration"], {"temperature_file": "calibration.json"}
            )
            (pkg / "README.md").write_text("tampered")
            with self.assertRaises(ValueError):
                api.verify_bundle(pkg)
            self.assertEqual(canonical({}), "{}")
            self.assertEqual(len(hashlib.sha256(b"").hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
