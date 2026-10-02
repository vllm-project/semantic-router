"""CPU/stdlib tests for the Decision 2.0 release pipeline (no torch, no network)."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from v2.release import automap, build, card, examples, layout, licence
from v2.release.runtime import api
from v2.release.tests import card_fixture

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
        self.assertEqual(layout.name_for(571_909_635), "Decision-2.0-Kai-0.6B")
        self.assertEqual(layout.name_for(753_446_208), "Decision-2.0-Eos-0.8B")
        self.assertEqual(layout.name_for(7_940_895_744), "Decision-2.0-Lux-9B")
        self.assertEqual(layout.name_for(25_688_227_840), "Decision-2.0-Vega-27B")
        self.assertIsNone(layout.tier_for(1_300_000_000))
        with self.assertRaises(ValueError):
            layout.name_for(1_300_000_000)

    def test_count_names_keep_the_released_names(self):
        # Loaded counts of the released Decision-2.0-Kai-0.6B, 0.8B, 2B and 4B packages.
        for loaded, name in (
            (597_103_104, "Decision-2.0-Kai-0.6B"),
            (753_446_208, "Decision-2.0-Eos-0.8B"),
            (1_883_930_944, "Decision-2.0-Sol-2B"),
            (4_208_383_488, "Decision-2.0-Nox-4B"),
        ):
            self.assertEqual(layout.release_name(loaded, "loaded-parameters"), name)
            self.assertEqual(layout.release_name(loaded), name)
        self.assertEqual(layout.tier_for(25_746_591_744), "27B")
        self.assertEqual(layout.release_name(25_746_591_744), "Decision-2.0-Vega-27B")
        self.assertEqual(
            layout.release_name(25_746_591_744, "loaded-parameters"),
            "Decision-2.0-Vega-26B",
        )
        layout.check_repo(
            "vllm-sr/Decision-2.0-Vega-26B",
            "Decision-2.0-Vega-26B",
            staging=False,
        )
        with self.assertRaises(ValueError):
            layout.count_name(1_300_000_000)
        with self.assertRaises(ValueError):
            layout.release_name(4_208_383_488, "rounded")

    def test_repository_rules(self):
        layout.check_repo(
            "vllm-sr/dev2-release-staging",
            "Decision-2.0-Kai-0.6B",
            staging=True,
        )
        layout.check_repo(
            "vllm-sr/Decision-2.0-Nox-4B",
            "Decision-2.0-Nox-4B",
            staging=False,
        )
        for repo, name, staging in (
            ("vllm-sr/Decision-2.0-Nox-4B", "Decision-2.0-Nox-4B", True),
            ("vllm-sr/dev2-release-staging", "Decision-2.0-Nox-4B", False),
            ("someone/Decision-2.0-Nox-4B", "Decision-2.0-Nox-4B", False),
            ("vllm-sr/Decision-2.0-Nox-4B", "Decision-2.0-Lux-9B", False),
            ("vllm-sr/DEV2.0-4B", "DEV2.0-4B", False),
            ("vllm-sr/Decision-2.0-Sol-4B", "Decision-2.0-Sol-4B", False),
            (
                "vllm-sr/Decision-2.0-Nox-1.3B",
                "Decision-2.0-Nox-1.3B",
                False,
            ),
            (
                "vllm-sr/Decision-2.0-Route-0.6B",
                "Decision-2.0-Route-0.6B",
                False,
            ),
        ):
            with self.assertRaises(ValueError):
                layout.check_repo(repo, name, staging=staging)
        for repo, name, staging, current in (
            (
                "llm-semantic-router/DEV2.0-9B",
                "Decision-2.0-Lux-9B",
                False,
                "vllm-sr/Decision-2.0-Lux-9B",
            ),
            (
                "llm-semantic-router/Decision-2.0-Nox-4B",
                "Decision-2.0-Nox-4B",
                False,
                "vllm-sr/Decision-2.0-Nox-4B",
            ),
            (
                "llm-semantic-router/dev2-release-staging",
                "Decision-2.0-Kai-0.6B",
                True,
                "vllm-sr/dev2-release-staging",
            ),
        ):
            with self.assertRaisesRegex(ValueError, f"renamed; use {current}\\Z"):
                layout.check_repo(repo, name, staging=staging)

    def test_former_repositories_map_to_the_renamed_ones(self):
        self.assertEqual(
            layout.FORMER_REPOS,
            {
                "llm-semantic-router/DEV2.0-0.6B": "vllm-sr/Decision-2.0-Kai-0.6B",
                "llm-semantic-router/DEV2.0-0.8B": "vllm-sr/Decision-2.0-Eos-0.8B",
                "llm-semantic-router/DEV2.0-2B": "vllm-sr/Decision-2.0-Sol-2B",
                "llm-semantic-router/DEV2.0-4B": "vllm-sr/Decision-2.0-Nox-4B",
                "llm-semantic-router/DEV2.0-9B": "vllm-sr/Decision-2.0-Lux-9B",
                "llm-semantic-router/DEV2.0-27B": "vllm-sr/Decision-2.0-Vega-27B",
            },
        )
        for new in layout.FORMER_REPOS.values():
            layout.check_repo(new, new.rsplit("/", 1)[1], staging=False)
            self.assertEqual(layout.current_repo(new), new)
        for former, current in (
            ("llm-semantic-router/DEV2.0-Route-0.6B", "vllm-sr/DEV2.0-Route-0.6B"),
            ("llm-semantic-router/Decision-2.0-Sol-2B", "vllm-sr/Decision-2.0-Sol-2B"),
            ("llm-semantic-router/Decision-1.0-Lux-9B", "vllm-sr/Decision-1.0-Lux-9B"),
            ("vllm-sr/DEV2.0-Route-0.6B", "vllm-sr/DEV2.0-Route-0.6B"),
            ("vllm-sr/DEV2.0-4B", "vllm-sr/DEV2.0-4B"),
            ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
            ("llm-semantic-router", "llm-semantic-router"),
        ):
            self.assertEqual(layout.current_repo(former), current)

    def test_current_ids_change_only_hub_references(self):
        cache = "/data/dev2/hf-cache/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e/LICENSE"
        spec = {
            "repo_id": "llm-semantic-router/Decision-2.0-Eos-0.8B",
            "origin": {
                "repo_id": "llm-semantic-router/Decision-1.0-Eos-0.8B",
                "revision": "a" * 40,
            },
            "base": {"repo_id": "Qwen/Qwen3.5-0.8B"},
            "licence": {
                "components": [
                    {"source": "llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e"},
                    {"source": "llm-semantic-router/DEV2.0-0.8B"},
                ],
                "files": [{"source": cache}],
            },
            "banner": {
                "source": "llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e:assets/x.png"
            },
            "_release": {
                "renamed_from": "llm-semantic-router/DEV2.0-0.8B",
                "note": "llm-semantic-router/DEV2.0-0.8B was moved",
            },
            "notes": [
                {
                    "source": "llm-semantic-router/decision-2.0-training-data m6/ib4 (see record)"
                }
            ],
        }
        self.assertEqual(
            layout.current_ids(spec),
            {
                **spec,
                "repo_id": "vllm-sr/Decision-2.0-Eos-0.8B",
                "origin": {
                    "repo_id": "vllm-sr/Decision-1.0-Eos-0.8B",
                    "revision": "a" * 40,
                },
                "licence": {
                    "components": [
                        {"source": "vllm-sr/Decision-1.0-Eos-0.8B@363c4a5e"},
                        {"source": "vllm-sr/Decision-2.0-Eos-0.8B"},
                    ],
                    "files": [{"source": cache}],
                },
                "banner": {
                    "source": "vllm-sr/Decision-1.0-Eos-0.8B@363c4a5e:assets/x.png"
                },
            },
        )
        self.assertEqual(spec["repo_id"], "llm-semantic-router/Decision-2.0-Eos-0.8B")

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
            "Decision-2.0-Nox-4B",
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
            self.check(None, "decision2", "Decision-2.0-Kai-0.6B (private)")["eligible"]
        )
        self.assertTrue(self.check("Hanno-Labs/bosun-v3.1-0.6b")["eligible"])
        self.assertTrue(
            self.check("vllm-sr/Decision-1.0-Kai-0.6B", "decision1")["eligible"]
        )
        # Reports scored before the organization rename name the former organization.
        self.assertEqual(
            self.check("llm-semantic-router/Decision-1.0-Kai-0.6B", "decision1")[
                "reason"
            ],
            "own Decision 1.0",
        )
        self.assertFalse(
            self.check("llm-semantic-router/DEV2.0-0.6B@5380e01e", "decision2")[
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
            "label": "Decision-2.0-Kai-0.6B",
        },
        {
            "key": "kai1",
            "role": "own-1.0",
            "report": str(REPORTS / "kai1.json"),
            "repo_id": "vllm-sr/Decision-1.0-Kai-0.6B",
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
        "model_name": "Decision-2.0-Kai-0.6B",
        "repo_id": "vllm-sr/dev2-release-staging",
        "profile": "kai-native",
        "parameters": {
            "loaded": 571_909_635,
            "components_text": "native encoder paths and heads 571,909,635",
        },
        "max_input_tokens": 8192,
        "origin": {
            "repo_id": "vllm-sr/Decision-1.0-Kai-0.6B",
            "revision": "7" * 40,
            "relation": "unchanged",
            "summary": "Weights unchanged.",
        },
        "licence": {
            "spdx": licence_spdx,
            "license_name": "decision-2.0-component-licences",
            "components": [],
        },
        "model_sha256": "b" * 64,
        "speed": {"median_ms": 12.34, "requests": 400},
        "remote_code": {"tested": ["5.17.0"], "base": None},
    }


TEXT: dict[str, str] = {}


def build_test_card(
    scratch: Path,
    entries: list[dict],
    values: dict,
    text: dict | None = None,
    paired: Path | None = None,
    paired_peers: dict[str, Path] | None = None,
    family: dict[str, float] | None = None,
) -> dict:
    """Build a card from synthetic Index values and placeholder assets; README, files and digests."""
    entries = card_fixture.relabel(entries, values["model_name"])
    index, assets = card_fixture.card_inputs(
        scratch,
        entries,
        values["model_name"],
        values["model_sha256"],
        values.get("comparison", "own-1.0"),
        family=family,
    )
    out = scratch / "pkg"
    result = card.build_card(
        entries=entries,
        roster=ROSTER,
        paired=paired,
        facts=values,
        text=TEXT if text is None else text,
        work=scratch / "work",
        output=out,
        index=index,
        assets=assets,
        paired_peers=paired_peers,
    )
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    readme = (out / "README.md").read_text()
    return {
        **result,
        "readme": readme,
        "files": files,
        "out": out,
        "index": index,
        "assets": assets,
        "problems": card.check_rendered(readme, files | {"LICENSE"}),
    }


class CardTest(unittest.TestCase):
    def test_product_card_from_same_panel_reports(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = build_test_card(
                Path(scratch),
                card_entries(),
                facts(),
                {"staging_notice": "Staging dry run."},
            )
            readme = result["readme"]
            self.assertEqual(result["problems"], [])
            self.assertEqual({e["key"] for e in result["excluded"]}, {"old", "jpt"})
            self.assertEqual(result["files"], {"README.md", *layout.CARD_ASSETS})
            self.assertIn(
                "Typed Choice (correct): 235/800 versus 277/800", result["tradeoffs"]
            )
            body = readme.split("\n---\n", 1)[1]
            self.assertTrue(
                body.startswith(
                    "\n![Decision-2.0-Kai-0.6B](assets/banner.png)\n\n# Decision-2.0-Kai-0.6B\n"
                )
            )
            headings = re.findall(r"^#{2} .+$", readme, flags=re.M)
            self.assertEqual(
                headings,
                [
                    "## Highlights",
                    "## Quickstart",
                    "## Evaluation",
                    "## License",
                    "## Citation",
                ],
            )
            self.assertIn("license: apache-2.0", readme)
            self.assertIn("base_model: vllm-sr/Decision-1.0-Kai-0.6B", readme)
            self.assertIn("| **Context length** | 8,192 tokens |", readme)
            self.assertIn("| **Decision types** | Choice · Yes / No · Score |", readme)
            self.assertIn("Apache-2.0 ([LICENSE](LICENSE)).", readme)
            self.assertIn(
                "**Speed:** a median of 12.3 ms per single-question request on a single GPU.",
                readme,
            )
            self.assertIn("**Many questions, one pass:**", readme)
            # Synthetic Index: Kai 0.6B 10.5, Decision 1.0 Kai 8.25.
            self.assertIn("+2.2 on the Jev Decision Index", readme)
            self.assertIn("| **10.5** |", readme)
            self.assertIn("| 8.2 |", readme)
            self.assertIn(f"<sub>{card_fixture.FOOTNOTE}</sub>", readme)
            for word in (
                "Limitations",
                "## Training data",
                "NOTICE",
                "ATTRIBUTIONS",
                "JevBench",
                "EVALUATION",
                "LoRA",
                "BF16",
                "stock",
                "Decision 2.0 collection",
            ):
                self.assertNotIn(word, readme)
            self.assertNotIn("](https://huggingface.co/collections/", readme)
            self.assertEqual(readme.count("```python"), 1)
            code = readme.split("```python\n", 1)[1].split("```", 1)[0]
            compile(code, "card", "exec")
            self.assertIn(
                'AutoModel.from_pretrained("vllm-sr/dev2-release-staging", trust_remote_code=True)',
                code,
            )
            self.assertEqual(code.count('pipeline("decision"'), 1)
            self.assertTrue(
                next(
                    l for l in code.splitlines() if 'pipeline("decision"' in l
                ).startswith("# ")
            )
            for name in layout.CARD_ASSETS:
                self.assertEqual(
                    (result["out"] / name).read_bytes(),
                    (result["assets"] / name).read_bytes(),
                )
            self.assertEqual(set(result["figures_sha256"]), set(layout.CARD_ASSETS))
            self.assertEqual(result["index_sha256"], layout.sha_file(result["index"]))
            receipt = (Path(scratch) / "work" / card.ASSETS_RECEIPT).read_text()
            self.assertNotIn("10.5", receipt)

    def test_index_gain_is_stated_only_when_positive(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = build_test_card(
                Path(scratch), card_entries(), facts(), family={"0.6B": 8.0}
            )
            self.assertEqual(result["problems"], [])
            self.assertNotIn("on the Jev Decision Index", result["readme"])
            self.assertIn(card.CHART_AREAS, result["readme"])
            self.assertIn("| **8.0** |", result["readme"])

    def test_released_card_links_the_collection_and_project(self):
        with tempfile.TemporaryDirectory() as scratch:
            readme = build_test_card(Path(scratch), card_entries(), facts())["readme"]
            self.assertIn(f"[Decision 2.0]({card.COLLECTION_URL})", readme)
            self.assertIn(f"[vLLM Semantic Router]({card.PROJECT_URL})", readme)

    def test_card_refuses_unknown_text_missing_remote_code_and_foreign_inputs(self):
        for values, text in (
            (facts(), {"model_type": "round-1 key"}),
            ({**facts(), "remote_code": None}, {}),
            ({**facts(), "model_name": "DEV2.0-0.6B"}, {}),
        ):
            with tempfile.TemporaryDirectory() as scratch, self.assertRaises(
                (ValueError, TypeError)
            ):
                build_test_card(Path(scratch), card_entries(), values, text)
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            entries = card_fixture.relabel(card_entries(), facts()["model_name"])
            index, assets = card_fixture.card_inputs(
                scratch, entries, facts()["model_name"], facts()["model_sha256"]
            )

            def attempt(values=None, **paths):
                card.build_card(
                    entries=entries,
                    roster=ROSTER,
                    paired=None,
                    facts=values or facts(),
                    text={},
                    work=scratch / "work",
                    output=scratch / "pkg",
                    index=paths.get("index", index),
                    assets=paths.get("assets", assets),
                )

            with self.assertRaisesRegex(ValueError, "other weights"):
                attempt({**facts(), "model_sha256": "c" * 64})
            other = card_fixture.index_file(
                scratch / "other-index.json",
                {"0.6B": facts()["model_sha256"]},
                {"0.6B": 99.5},
            )
            with self.assertRaisesRegex(ValueError, "another Index input"):
                attempt(index=other)
            (assets / layout.CHART_FILES[0]).write_bytes(b"\x89PNG\r\n\x1a\nchanged")
            with self.assertRaisesRegex(ValueError, "differs from its receipt"):
                attempt()

    def test_card_needs_own_comparator(self):
        entries = [e for e in card_entries() if e["role"] != "own-1.0"]
        with self.assertRaises(ValueError):
            card.select_reports(entries, ROSTER)

    def test_http_check_anchors_follow_hub_headings(self):
        from v2.release.tests.hub_card_http_check import anchor

        self.assertEqual(anchor("Model overview"), "model-overview")
        self.assertEqual(
            anchor("Results below Decision 1.0 Eos"),
            "results-below-decision-10-eos",
        )

    def test_lint(self):
        self.assertFalse(card.lint("A Pareto frontier"))
        self.assertTrue(card.lint("ran on node A"))
        self.assertTrue(card.lint("path /data/dev2/runs"))
        self.assertTrue(card.lint("JevArena v3 score"))
        self.assertFalse(card.lint("Transformers 4.57.6 on one GPU, Python 3.12"))
        self.assertFalse(card.lint("node Express"))
        self.assertEqual(card.lint_readme("post-key results"), ["post-key wording"])
        self.assertTrue(card.lint_readme("typed Brier / ECE"))
        self.assertTrue(card.lint_readme("revision " + "a" * 40))
        for removed in (
            "JevBench public 231",
            "rank-128 LoRA",
            "BF16 backbone",
            "## Limitations",
            "## Training data",
            "see ATTRIBUTIONS.md",
            "[NOTICE](NOTICE)",
            "evaluation/EVALUATION.md",
            "Runs with stock 🤗 Transformers",
        ):
            self.assertTrue(card.lint_readme(removed), removed)
        self.assertFalse(card.lint_readme("JevArena 43.22, Transformers 5.17"))
        self.assertEqual(
            card.lint_readme(
                'from_pretrained("llm-semantic-router/Decision-2.0-Kai-0.6B")'
            ),
            ["former Hugging Face organization"],
        )
        self.assertFalse(
            card.lint_readme('from_pretrained("vllm-sr/Decision-2.0-Kai-0.6B")')
        )
        self.assertIn("/collections/vllm-sr/", card.COLLECTION_URL)


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

    def test_kernel_gate_uses_the_decoder_runtime_check(self):
        self.assertTrue(examples.RUNTIME_CHECK.is_file())
        self.assertIsNone(examples.kernel_runtime(False))
        original = examples.RUNTIME_CHECK
        with tempfile.TemporaryDirectory() as scratch:
            stub = Path(scratch) / "runtime_check.py"
            try:
                examples.RUNTIME_CHECK = stub
                stub.write_text(
                    "def runtime_identity():\n    return {'kernel_bindings': {'f': 'fla.x'}}\n"
                    "def violations(identity):\n    return []\n"
                )
                result = examples.kernel_runtime(True)
                self.assertEqual(result["kernel_bindings"], {"f": "fla.x"})
                self.assertEqual(result["runtime_check_sha256"], layout.sha_file(stub))
                stub.write_text(
                    "def runtime_identity():\n    return {}\n"
                    "def violations(identity):\n    return ['reference path']\n"
                )
                with self.assertRaises(RuntimeError):
                    examples.kernel_runtime(True)
            finally:
                examples.RUNTIME_CHECK = original

    def test_card_example_imports_only_named_sites(self):
        import argparse

        answers = {"route": {"type": "noul", "noul": 0.25}}
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            site, package = scratch / "site", scratch / "pkg"
            site.mkdir()
            package.mkdir()
            (site / "sitemod.py").write_text(f"ANSWERS = {answers!r}\n")
            (package / "README.md").write_text(
                '```python\nimport json\nfrom sitemod import ANSWERS\nNAME = "pkg"\n'
                "print(json.dumps(ANSWERS))\n```\n"
            )
            reference = scratch / "reference.json"
            reference.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {
                                "id": examples.EXAMPLES[0]["id"],
                                "response": {"answers": answers},
                            }
                        ]
                    }
                )
            )
            args = argparse.Namespace(
                package=package, reference=reference, tolerance=0.0, site=[]
            )
            self.assertFalse(examples.card(args)["passed"])
            args.site = [str(site)]
            result = examples.card(args)
            self.assertTrue(result["passed"])
            self.assertEqual(result["interpreter_flags"], ["-s", "-B"])

    def test_launcher_refuses_secret_like_env(self):
        import subprocess

        script = ROOT / "v2/release/release.sh"
        common = ["bash", str(script), "--spec", "s", "--src", "x", "--work", "/tmp/w"]
        refused = subprocess.run(
            [*common, "--env", "HF_TOKEN=abc"], capture_output=True, text=True
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("non-secret", refused.stderr)
        accepted = subprocess.run(
            [*common, "--env", "TRITON_CACHE_DIR=/c"], capture_output=True, text=True
        )
        self.assertIn("work dir must be under", accepted.stderr)


class GateTest(unittest.TestCase):
    def test_decision_must_name_this_candidate(self):
        from v2.release import gate

        with tempfile.TemporaryDirectory() as scratch:
            paired = Path(scratch) / "paired.json"
            paired.write_text(
                json.dumps(
                    {
                        "point": {"delta": {"score": 3.0}},
                        "ci95": {"low": 1.0, "high": 5.0},
                    }
                )
            )
            spec = {
                "model_name": "Decision-2.0-Nox-4B",
                "repo_id": "vllm-sr/Decision-2.0-Nox-4B",
                "expected_identity": {"model_sha256": "a" * 64},
                "scored": {"report_sha256": "b" * 64},
                "card": {"paired": str(paired)},
            }
            decision = {
                "schema": gate.DECISION_SCHEMA,
                "decision": "release",
                "model_name": "Decision-2.0-Nox-4B",
                "repo_id": "vllm-sr/Decision-2.0-Nox-4B",
                "identity": {"model_sha256": "a" * 64},
                "report_sha256": "b" * 64,
                "paired_sha256": layout.sha_file(paired),
                "decided_by": "coordinator",
                "rationale": "paired interval excludes zero",
            }
            path = Path(scratch) / "decision.json"
            path.write_text(json.dumps(decision))
            self.assertEqual(gate.check(spec, path)["decision"], "release")
            path.write_text(
                json.dumps({**decision, "identity": {"model_sha256": "c" * 64}})
            )
            with self.assertRaises(ValueError):
                gate.check(spec, path)

            draft = {
                **{k: v for k, v in decision.items() if k != "decided_by"},
                "status": "draft",
                "prepared_by": "release engineering",
            }
            path.write_text(json.dumps(draft))
            self.assertEqual(gate.check(spec, path)["status"], "draft")
            with self.assertRaises(ValueError):
                gate.check(spec, path, final=True)
            for bad in (
                {**draft, "decided_by": "coordinator"},
                {k: v for k, v in draft.items() if k != "prepared_by"},
                {**draft, "status": "pending"},
                {k: v for k, v in decision.items() if k != "decided_by"},
            ):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    gate.check(spec, path)
            path.write_text(json.dumps({**decision, "status": "final"}))
            self.assertEqual(gate.check(spec, path, final=True)["status"], "final")

    def test_evaluate_items_and_legacy_download_receipt(self):
        from v2.release import gate

        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            receipts = work / "receipts"
            receipts.mkdir()
            paired = work / "paired.json"
            paired.write_text(json.dumps({"ci95": {"low": 3.6, "high": 13.3}}))
            decision = work / "decision.json"
            decision.write_text(json.dumps({"status": "draft"}))
            spec = {
                "kind": "release",
                "model_name": "Decision-2.0-Nox-4B",
                "repo_id": "vllm-sr/Decision-2.0-Nox-4B",
                "expected_identity": {"model_sha256": "c" * 64},
                "scored": {"report_sha256": "d" * 64},
                "card": {"paired": str(paired)},
                "gate_receipt": str(decision),
            }
            revision = "a" * 40
            files = {
                "spec": spec,
                "build": {"parameters": {"loaded": 7}, "card": {"tradeoffs": []}},
                "repeat-pre": {"passed": True},
                "card-pre": {"passed": True},
                "upload": {"revision": revision},
                "download": {"revision": revision, "files": 3},
                "tree": {"passed": True, "files": 3},
                "post": {"passed": True, "loaded_parameters": 7},
                "repeat-post": {"passed": True},
                "card-post": {"passed": True},
                "readback": {"passed": True, "card_problems": [], "card_data": {}},
            }
            for name, value in files.items():
                (receipts / f"{name}.json").write_text(json.dumps(value))
            result = gate.evaluate(work)
            self.assertTrue(result["passed"], result["items"])
            self.assertEqual(result["decision"]["status"], "draft")
            (receipts / "download.json").write_text(
                json.dumps({"revision": "b" * 40, "files": 3})
            )
            result = gate.evaluate(work)
            self.assertFalse(result["items"]["3_download_hash_parameters"]["passed"])
            with self.assertRaises(ValueError):
                gate.seal(work)


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
            entries = card_entries()
            spec = {
                "schema": build.SPEC_SCHEMA,
                "kind": "staging",
                "repo_id": "vllm-sr/dev2-release-staging",
                "model_name": "Decision-2.0-Nox-4B",
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
                "card": card_fixture.pinned_card(
                    scratch,
                    entries,
                    "Decision-2.0-Nox-4B",
                    identity["model_sha256"],
                    {"staging_notice": "Staging."},
                ),
            }
            spec_path = scratch / "spec.json"
            spec_path.write_text(json.dumps(spec))
            receipt = build.build(spec_path, scratch / "out" / "dev2-release-staging")
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
            self.assertEqual(pointer["auto_map"], automap.CONFIG_FIELDS["auto_map"])
            for name in automap.FILES:
                self.assertEqual(
                    (pkg / name).read_bytes(), (automap.SOURCE / name).read_bytes()
                )
                self.assertEqual(
                    manifest["files_sha256"][name],
                    manifest["remote_code"]["files"][name]["sha256"],
                )
            self.assertIn(card.TRANSFORMERS_HEADING, (pkg / "README.md").read_text())
            for name in ("NOTICE", "ATTRIBUTIONS.md", "evaluation/EVALUATION.md"):
                self.assertFalse((pkg / name).exists(), name)
            self.assertEqual(
                manifest["card"]["figures_sha256"],
                {n: manifest["files_sha256"][n] for n in layout.CARD_ASSETS},
            )
            (pkg / "README.md").write_text("tampered")
            with self.assertRaises(ValueError):
                api.verify_bundle(pkg)
            self.assertEqual(canonical({}), "{}")
            self.assertEqual(len(hashlib.sha256(b"").hexdigest()), 64)

            spec["scored"] = {"label": "scored run /data/dev2/runs/x"}  # manifest only
            spec_path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                build.build(spec_path, scratch / "out2" / "dev2-release-staging")
            self.assertFalse((scratch / "out2" / "dev2-release-staging").exists())


if __name__ == "__main__":
    unittest.main()
