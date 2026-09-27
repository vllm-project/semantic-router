"""Gold-free first-release v3 planning and receipt integrity checks."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run import digest as native_digest
from jev_arena.jevbench_public import FILES, SOURCE_REVISION, SOURCE_URL
from scripts.baseline_attestation_v3 import build_attestation
from scripts import baseline_repeat_smoke_v3 as smoke
from scripts.eikos_stable_runtime_v3 import FLA_BACKEND, TORCH_BACKEND
from scripts.plan_final_eval import BASELINES, sha_file
from scripts import plan_first_release_v3 as planner
from scripts.plan_first_release_v3 import (
    GATE_DOCUMENT,
    PLAN_VERSION,
    ROSTER_VERSION,
    _baseline_prediction_hashes,
    _baseline_repeatability,
    audit_prekey_predictions,
    build_plan,
    checked_roster,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class FirstReleasePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "models"
        self.external_root = self.root / "external"
        self.models = [
            next(model for model in BASELINES if model.key == key)
            for key in ("nox", "eikos4b")
        ]
        self.attestations = []
        for model in self.models:
            package = self.model_root / model.model_dir
            package.mkdir(parents=True)
            (package / "config.json").write_text("{}\n", encoding="utf-8")
            if model.key == "nox":
                (package / "bundle-manifest.json").write_text("{}\n", encoding="utf-8")
            else:
                (package / "decision_config.json").write_text("{}\n", encoding="utf-8")
                (package / "SHA256SUMS").write_text("fixture\n", encoding="utf-8")
            header = json.dumps(
                {"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
            ).encode()
            (package / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + b"\0" * 16
            )
            receipt = self.root / f"{model.key}.receipt.json"
            attestation = self.root / f"{model.key}.attestation.json"
            build_attestation(
                model,
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
                receipt_output=receipt,
                attestation_output=attestation,
                loaded_parameter_count=4,
            )
            self.attestations.append(json.loads(attestation.read_text()))
        self.candidate = {
            "key": "d2-4b",
            "label": "DEV2.0-4B",
            "size": "4B",
            "model_id": "llm-semantic-router/DEV2.0-4B",
            "selected_checkpoint": "checkpoint-0232",
            "architecture": "eikos_semif",
            "package_dir": str(self.root / "candidate-package"),
            "model_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "calibration": str(self.root / "candidate-cal.json"),
            "max_length": 16000,
        }
        choice32 = self.root / "choice32.prompts.jsonl"
        typed_dev = self.root / "typed-dev.prompts.jsonl"

        def prompt(index: int, kind: str) -> dict:
            question = (
                {"type": "choice", "options": {"A": "yes", "B": "no"}}
                if kind == "choice"
                else {"type": kind}
            )
            return {
                "id": f"{kind}-{index}",
                "state": f"state {index}",
                "questions": {"q": question},
            }

        choice32.write_text(
            "".join(json.dumps(prompt(i, "choice")) + "\n" for i in range(32))
        )
        typed_dev.write_text(
            "".join(
                json.dumps(prompt(i, ("choice", "noul", "score")[i % 3])) + "\n"
                for i in range(1600)
            )
        )
        for name, digest in (
            ("CSS32_SHA", sha_file(choice32)),
            ("TYPED_DEV_SHA", sha_file(typed_dev)),
        ):
            patcher = patch.object(smoke, name, digest)
            patcher.start()
            self.addCleanup(patcher.stop)
        smoke_prompts = self.root / "repeat32.prompts.jsonl"
        smoke_manifest = self.root / "repeat32.manifest.json"
        smoke.prepare(choice32, typed_dev, smoke_prompts, smoke_manifest)
        patcher = patch.object(
            planner, "BASELINE_SMOKE_PROMPTS_SHA", sha_file(smoke_prompts)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.repeat_panel = {
            "choice32_path": str(choice32),
            "typed_dev_path": str(typed_dev),
            "prompts_path": str(smoke_prompts),
            "manifest_path": str(smoke_manifest),
        }
        self.repeatability = []
        for model, attestation in zip(self.models, self.attestations, strict=True):
            row_hashes = _baseline_prediction_hashes(model, attestation)
            first = self.root / f"{model.key}.repeat-first.jsonl"
            second = self.root / f"{model.key}.repeat-second.jsonl"
            repeated = []
            for item in smoke.rows(smoke_prompts):
                kind = item["questions"]["q"]["type"]
                answer = (
                    {
                        "type": "choice",
                        "choice": "A",
                        "probabilities": {"A": 0.7, "B": 0.3},
                    }
                    if kind == "choice"
                    else (
                        {"type": "noul", "noul": 0.7}
                        if kind == "noul"
                        else {
                            "type": "score",
                            "score": 0.7,
                            "probabilities": {"0": 0.3, "1": 0.7},
                        }
                    )
                )
                repeated.append(
                    {
                        "id": item["id"],
                        "answers": {"q": answer},
                        "source_input_sha256": native_digest(
                            {"state": item["state"], "questions": item["questions"]}
                        ),
                        "model_id": model.model_id,
                        "model_revision": model.revision,
                        "backend": model.backend,
                        "adapter_version": "native-published-v2",
                        "revision_attested": True,
                        "runtime_matches_validated": True,
                        "usage": {"input_tokens": 100},
                        **row_hashes,
                    }
                )
            content = "".join(json.dumps(row) + "\n" for row in repeated)
            first.write_text(content)
            second.write_text(content)
            receipt = self.root / f"{model.key}.repeat-receipt.json"
            write_json(
                receipt,
                smoke.compare(
                    smoke_prompts,
                    first,
                    second,
                    model_id=model.model_id,
                    revision=model.revision,
                    backend=model.backend,
                    adapter_version="native-published-v2",
                ),
            )
            self.repeatability.append(
                {
                    "key": model.key,
                    "receipt_path": str(receipt),
                    "receipt_sha256": sha_file(receipt),
                    "first_predictions_path": str(first),
                    "second_predictions_path": str(second),
                }
            )
        self.roster_path = self.root / "roster.json"
        write_json(
            self.roster_path,
            {
                "schema_version": ROSTER_VERSION,
                "candidate_keys": ["d2-4b"],
                "candidate_size_b": {"d2-4b": 4.2e-9},
                "baseline_repeat_panel": self.repeat_panel,
                "baseline_keys": ["nox", "eikos4b"],
                "baseline_attestations": self.attestations,
                "baseline_repeatability": self.repeatability,
                "pairs": [
                    {
                        "candidate": "d2-4b",
                        "comparator": "nox",
                        "size_relation": "same",
                        "rationale": "",
                    }
                ],
                "gate_document_sha256": sha_file(SOURCE_ROOT / GATE_DOCUMENT),
            },
        )
        self.public = self.root / "public-panel"
        self.public.mkdir()
        self.public_prompts = self.public / "prompts.jsonl"
        self.public_prompts.write_text("", encoding="utf-8")
        write_json(
            self.public / "manifest.json",
            {
                "build_version": "jevarena-jevbench-public-build/1",
                "items": 231,
                "source_url": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "source_sha256": {
                    tier: digest for tier, (_, digest, _) in FILES.items()
                },
                "prompts_sha256": sha_file(self.public_prompts),
                "targets_sha256": "c" * 64,
            },
        )
        self.css = self.root / "css.prompts.jsonl"
        self.css.write_text("", encoding="utf-8")
        self.candidate_freeze = self.root / "candidate-freeze.json"
        write_json(self.candidate_freeze, {"frozen": True})
        self.stable_runtime = {"d2-4b": {"fixture": "verified upstream"}}

    def test_composite_native_row_digests_follow_adapter_recipes(self) -> None:
        cases = (
            (
                "kev",
                {
                    "provenance.json": "1" * 64,
                    "head.pt": "2" * 64,
                    "adapter_model.safetensors": "3" * 64,
                },
                {},
                {
                    "model_config_sha256": native_digest(
                        {
                            "provenance": "1" * 64,
                            "head": "2" * 64,
                            "adapter": "3" * 64,
                        }
                    )
                },
            ),
            (
                "jevk5-4b",
                {
                    "jevk5_config.json": "4" * 64,
                    "model.safetensors": "5" * 64,
                    "SHA256SUMS": "6" * 64,
                },
                {"jevk5/runtime.py": "7" * 64, "jevk5/prompt.py": "8" * 64},
                {
                    "model_config_sha256": "4" * 64,
                    "model_weight_sha256": "5" * 64,
                    "release_manifest_sha256": "6" * 64,
                    "runtime_source_sha256": native_digest(
                        {"jevk5/runtime.py": "7" * 64, "jevk5/prompt.py": "8" * 64}
                    ),
                },
            ),
        )
        for key, files, runtime_files, expected in cases:
            with self.subTest(key=key):
                path = self.root / f"{key}.native-receipt.json"
                write_json(path, {"files": files, "runtime_files": runtime_files})
                model = next(item for item in BASELINES if item.key == key)
                self.assertEqual(
                    _baseline_prediction_hashes(
                        model,
                        {"receipt_path": str(path), "receipt_sha256": sha_file(path)},
                    ),
                    expected,
                )

    def test_plan_is_separate_v3_two_axis_with_public_crosscheck(self) -> None:
        models, pairs, attestations, sizes, roster_sha = checked_roster(
            self.roster_path,
            candidates=[self.candidate],
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )
        plan = build_plan(
            models=models,
            pairs=pairs,
            attestations=attestations,
            candidate_freeze_path=self.candidate_freeze,
            candidate_freeze_sha256=sha_file(self.candidate_freeze),
            candidate_size_b=sizes,
            roster_path=self.roster_path,
            roster_sha256=roster_sha,
            css_prompts=self.css,
            css_info={"sha256": sha_file(self.css), "items": 6547, "task_count": 15},
            public_panel=self.public,
            source_root=SOURCE_ROOT,
            evaluation_root=self.root / "new-evaluation",
            model_root=self.model_root,
            external_root=self.external_root,
            python="python3",
            kai_lex_python="/private/kai-python",
            fla_path="/private/fla",
            stable_runtime=self.stable_runtime,
        )
        self.assertEqual(plan["plan_version"], PLAN_VERSION)
        self.assertEqual(plan["formula"], "100*sqrt(T*H)")
        self.assertEqual(len(plan["comparison_pairs_sha256"]), 64)
        self.assertEqual(plan["comparison_pairs"][0]["comparator"], "nox")
        self.assertEqual(len(plan["inference"]), 3)
        self.assertEqual(len(plan["paired_ci_commands_after_prekey_freeze"]), 1)
        self.assertEqual(plan["model_roster"][-1]["size_b"], 4.2e-9)
        inference = "\n".join(
            command for model in plan["inference"] for command in model["commands"]
        )
        self.assertIn("training.eikos.published_infer", inference)
        eikos = plan["inference"][-1]["commands"]
        self.assertEqual(len(eikos), 3)
        self.assertTrue(
            all("--deterministic-algorithms" in command for command in eikos)
        )
        self.assertTrue(
            all("--torch-reference-gated-delta" in command for command in eikos)
        )
        self.assertIn("inference.run", inference)
        self.assertIn("inference.eikos", inference)
        self.assertIn("--input", inference)
        self.assertIn("public-panel/prompts.jsonl", inference)
        score = "\n".join(
            command
            for model in plan["scoring_commands_after_prekey_freeze"]
            for command in model["commands"]
        )
        self.assertIn("benchmark.score", score)
        self.assertIn("transfer.score", score)
        self.assertIn("jev_arena.jevbench_public", score)
        eikos_score = plan["scoring_commands_after_prekey_freeze"][-1]["commands"][-1]
        self.assertIn("--prediction-manifest", eikos_score)
        self.assertNotIn("arena_v2", score)
        self.assertNotIn("decision_bench", score)
        self.assertNotIn("authored", score)

    def test_roster_rejects_missing_open_control_and_tampered_native_package(
        self,
    ) -> None:
        roster = json.loads(self.roster_path.read_text(encoding="utf-8"))
        roster["baseline_keys"] = ["nox"]
        roster["baseline_attestations"] = [self.attestations[0]]
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "open-model control"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )
        roster["baseline_keys"] = ["nox", "eikos4b"]
        roster["baseline_attestations"] = self.attestations
        write_json(self.roster_path, roster)
        (self.model_root / self.models[0].model_dir / "config.json").write_text(
            "changed\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ValueError, "package, runtime, or calibration differs"
        ):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )

    def test_roster_requires_pinned_two_process_smoke_for_every_baseline(self) -> None:
        roster = json.loads(self.roster_path.read_text())
        roster["baseline_repeatability"] = self.repeatability[:1]
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "Every selected baseline"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )
        roster["baseline_repeatability"] = self.repeatability
        roster["baseline_repeatability"][0]["second_predictions_path"] = roster[
            "baseline_repeatability"
        ][0]["first_predictions_path"]
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "two distinct native runs"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )

    def test_roster_rejects_changed_smoke_prompt_manifest(self) -> None:
        manifest = Path(self.repeat_panel["manifest_path"])
        contents = json.loads(manifest.read_text())
        contents["types"]["choice"] += 1
        write_json(manifest, contents)
        with self.assertRaisesRegex(ValueError, "pinned gold-free build"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )

    def test_roster_rejects_unattested_smoke_weights_and_failed_drift(self) -> None:
        item = self.repeatability[0]
        first = Path(item["first_predictions_path"])
        second = Path(item["second_predictions_path"])
        receipt = Path(item["receipt_path"])
        records = [json.loads(line) for line in first.read_text().splitlines()]
        records[0]["model_config_sha256"] = "0" * 64
        content = "".join(json.dumps(row) + "\n" for row in records)
        first.write_text(content)
        second.write_text(content)
        write_json(
            receipt,
            smoke.compare(
                Path(self.repeat_panel["prompts_path"]),
                first,
                second,
                model_id=self.models[0].model_id,
                revision=self.models[0].revision,
                backend=self.models[0].backend,
                adapter_version="native-published-v2",
            ),
        )
        roster = json.loads(self.roster_path.read_text())
        roster["baseline_repeatability"][0]["receipt_sha256"] = sha_file(receipt)
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "package differs from attestation"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )
        records[0]["model_config_sha256"] = _baseline_prediction_hashes(
            self.models[0], self.attestations[0]
        )["model_config_sha256"]
        records[0]["answers"]["q"]["probabilities"] = {"A": 0.69, "B": 0.31}
        first.write_text("".join(json.dumps(row) + "\n" for row in records))
        write_json(
            receipt,
            smoke.compare(
                Path(self.repeat_panel["prompts_path"]),
                first,
                second,
                model_id=self.models[0].model_id,
                revision=self.models[0].revision,
                backend=self.models[0].backend,
                adapter_version="native-published-v2",
            ),
        )
        roster["baseline_repeatability"][0]["receipt_sha256"] = sha_file(receipt)
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "repeatability gate failed"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )

    def test_kev_smoke_matches_composite_package_fingerprint(self) -> None:
        kev = next(model for model in BASELINES if model.key == "kev")
        package_receipt = self.root / "kev-package-receipt.json"
        write_json(
            package_receipt,
            {
                "files": {
                    "provenance.json": "1" * 64,
                    "head.pt": "2" * 64,
                    "adapter_model.safetensors": "3" * 64,
                },
                "runtime_files": {},
            },
        )
        attestation = {
            "receipt_path": str(package_receipt),
            "receipt_sha256": sha_file(package_receipt),
            "adapter_sha256": "4" * 64,
        }
        fingerprint = _baseline_prediction_hashes(kev, attestation)[
            "model_config_sha256"
        ]
        first, second = self.root / "kev-first.jsonl", self.root / "kev-second.jsonl"
        rows = [
            json.loads(line)
            for line in Path(self.repeatability[0]["first_predictions_path"])
            .read_text()
            .splitlines()
        ]
        for row in rows:
            row.update(
                model_id=kev.model_id,
                model_revision=kev.revision,
                backend=kev.backend,
                model_config_sha256=fingerprint,
            )
        content = "".join(json.dumps(row) + "\n" for row in rows)
        first.write_text(content)
        second.write_text(content)
        receipt = self.root / "kev-repeat-receipt.json"
        write_json(
            receipt,
            smoke.compare(
                Path(self.repeat_panel["prompts_path"]),
                first,
                second,
                model_id=kev.model_id,
                revision=kev.revision,
                backend=kev.backend,
                adapter_version="native-published-v2",
            ),
        )
        roster = {
            "baseline_repeat_panel": self.repeat_panel,
            "baseline_repeatability": [
                {
                    "key": kev.key,
                    "receipt_path": str(receipt),
                    "receipt_sha256": sha_file(receipt),
                    "first_predictions_path": str(first),
                    "second_predictions_path": str(second),
                }
            ],
        }
        verified = _baseline_repeatability(roster, [kev], {kev.key: attestation})
        self.assertEqual(verified[kev.key]["receipt_sha256"], sha_file(receipt))

    def test_same_size_requires_measured_parameter_ratio(self) -> None:
        roster = json.loads(self.roster_path.read_text(encoding="utf-8"))
        roster["candidate_size_b"]["d2-4b"] = 6e-9
        write_json(self.roster_path, roster)
        with self.assertRaisesRegex(ValueError, "Comparator size relation is false"):
            checked_roster(
                self.roster_path,
                candidates=[self.candidate],
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
            )
        roster["pairs"][0]["size_relation"] = "nearest"
        roster["pairs"][0][
            "rationale"
        ] = "Actual loaded parameters differ by more than the frozen ratio."
        write_json(self.roster_path, roster)
        _models, pairs, _attestations, _sizes, _roster_sha = checked_roster(
            self.roster_path,
            candidates=[self.candidate],
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )
        self.assertEqual(pairs[0]["size_relation"], "nearest")

    def test_prediction_audit_refuses_changed_source_before_any_label_access(
        self,
    ) -> None:
        models, pairs, attestations, sizes, roster_sha = checked_roster(
            self.roster_path,
            candidates=[self.candidate],
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )
        plan = build_plan(
            models=models,
            pairs=pairs,
            attestations=attestations,
            candidate_freeze_path=self.candidate_freeze,
            candidate_freeze_sha256=sha_file(self.candidate_freeze),
            candidate_size_b=sizes,
            roster_path=self.roster_path,
            roster_sha256=roster_sha,
            css_prompts=self.css,
            css_info={"sha256": sha_file(self.css), "items": 6547, "task_count": 15},
            public_panel=self.public,
            source_root=SOURCE_ROOT,
            evaluation_root=self.root / "new-evaluation",
            model_root=self.model_root,
            external_root=self.external_root,
            python="python3",
            kai_lex_python="/private/kai-python",
            fla_path="/private/fla",
            stable_runtime=self.stable_runtime,
        )
        plan["source_sha256"]["benchmark/score.py"] = "0" * 64
        with patch(
            "scripts.plan_first_release_v3.frozen_candidates",
            return_value=([self.candidate], "x"),
        ), patch(
            "scripts.plan_first_release_v3.verified_stable_runtime",
            return_value=self.stable_runtime,
        ):
            with self.assertRaisesRegex(ValueError, "Protocol source changed"):
                audit_prekey_predictions(plan)

    def test_gold_free_prediction_audit_full_panel_and_tamper(self) -> None:
        panels = {
            "typed": (self.root / "new-evaluation" / "typed-final.prompts.jsonl", 1600),
            "css": (self.css, 6547),
            "public": (self.public_prompts, 231),
        }
        prompts = {}
        for panel, (path, count) in panels.items():
            rows = [
                {
                    "id": f"{panel}-{i}",
                    "state": "state",
                    "questions": {
                        "q": {"type": "choice", "options": {"A": "yes", "B": "no"}}
                    },
                }
                for i in range(count)
            ]
            if panel != "typed":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "".join(
                        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )
            prompts[panel] = rows
        write_json(
            self.public / "manifest.json",
            {
                "build_version": "jevarena-jevbench-public-build/1",
                "items": 231,
                "source_url": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "source_sha256": {
                    tier: digest for tier, (_, digest, _) in FILES.items()
                },
                "prompts_sha256": sha_file(self.public_prompts),
                "targets_sha256": "c" * 64,
            },
        )
        models, pairs, attestations, sizes, roster_sha = checked_roster(
            self.roster_path,
            candidates=[self.candidate],
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )
        plan = build_plan(
            models=models,
            pairs=pairs,
            attestations=attestations,
            candidate_freeze_path=self.candidate_freeze,
            candidate_freeze_sha256=sha_file(self.candidate_freeze),
            candidate_size_b=sizes,
            roster_path=self.roster_path,
            roster_sha256=roster_sha,
            css_prompts=self.css,
            css_info={"sha256": sha_file(self.css), "items": 6547, "task_count": 15},
            public_panel=self.public,
            source_root=SOURCE_ROOT,
            evaluation_root=self.root / "new-evaluation",
            model_root=self.model_root,
            external_root=self.external_root,
            python="python3",
            kai_lex_python="/private/kai-python",
            fla_path="/private/fla",
            stable_runtime=self.stable_runtime,
        )
        typed_path = panels["typed"][0]
        typed_path.parent.mkdir(parents=True, exist_ok=True)
        typed_path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in prompts["typed"]
            ),
            encoding="utf-8",
        )
        hashes = []
        for model in plan["inference"]:
            for panel, name in model["paths"].items():
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for prompt in prompts[panel]:
                    payload = {
                        "state": prompt["state"],
                        "questions": prompt["questions"],
                    }
                    input_sha = hashlib.sha256(
                        json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                    row = {
                        "id": prompt["id"],
                        "answers": {"q": {"choice": "A"}},
                        "source_input_sha256": input_sha,
                        "model_id": model["model_id"],
                        "model_revision": model["revision"],
                    }
                    if model["group"] == "decision2":
                        row.update(
                            {
                                "model_sha256": self.candidate["model_sha256"],
                                "calibration_sha256": self.candidate[
                                    "calibration_sha256"
                                ],
                            }
                        )
                    else:
                        native = next(
                            item for item in self.models if item.key == model["key"]
                        )
                        package = self.model_root / native.model_dir
                        row.update(
                            {
                                "backend": native.backend,
                                "revision_attested": True,
                                "runtime_matches_validated": True,
                                "model_config_sha256": sha_file(
                                    package
                                    / (
                                        "bundle-manifest.json"
                                        if native.key == "nox"
                                        else "decision_config.json"
                                    )
                                ),
                            }
                        )
                        if native.key == "eikos4b":
                            row["release_manifest_sha256"] = sha_file(
                                package / "SHA256SUMS"
                            )
                    rows.append(row)
                path.write_text(
                    "".join(
                        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )
                hashes.append(f"{sha_file(path)}  {path}")
                if model["group"] == "decision2":
                    native_receipt = Path(name + ".manifest.json")
                    write_json(
                        native_receipt,
                        {
                            "model_id": model["model_id"],
                            "model_revision": model["revision"],
                            "model_sha256": self.candidate["model_sha256"],
                            "calibration": {
                                "file_sha256": self.candidate["calibration_sha256"]
                            },
                            "input_sha256": sha_file(panels[panel][0]),
                            "input_items": len(rows),
                            "evaluated_items": len(rows),
                            "counts": {"items": len(rows)},
                            "predictions_sha256": sha_file(path),
                            "calibration_sha256": self.candidate["calibration_sha256"],
                            "collector_source_sha256": sha_file(
                                SOURCE_ROOT / "training/eikos/published_infer.py"
                            ),
                            "runtime": {
                                "torch_deterministic_algorithms": True,
                                "gated_delta_backend_before": FLA_BACKEND,
                                "gated_delta_backend": TORCH_BACKEND,
                            },
                        },
                    )
                    hashes.append(f"{sha_file(native_receipt)}  {native_receipt}")
        (self.root / "new-evaluation" / "RAW_PREDICTIONS.sha256").write_text(
            "\n".join(hashes) + "\n", encoding="utf-8"
        )
        with patch(
            "scripts.plan_first_release_v3.frozen_candidates",
            return_value=([self.candidate], "x"),
        ), patch(
            "scripts.plan_first_release_v3.verified_stable_runtime",
            return_value=self.stable_runtime,
        ):
            audited = audit_prekey_predictions(plan)
            self.assertEqual(audited["status"], "gold_free_prekey_predictions_verified")
            self.assertEqual(
                audited["comparison_pairs_sha256"], plan["comparison_pairs_sha256"]
            )
            baseline_path = Path(plan["inference"][0]["paths"]["public"])
            baseline_rows = baseline_path.read_text()
            altered_rows = [json.loads(line) for line in baseline_rows.splitlines()]
            del altered_rows[0]["model_id"]
            baseline_path.write_text(
                "".join(json.dumps(row) + "\n" for row in altered_rows)
            )
            with self.assertRaisesRegex(ValueError, "model identity differs"):
                audit_prekey_predictions(plan)
            baseline_path.write_text(baseline_rows)
            native_path = Path(
                plan["inference"][-1]["paths"]["public"] + ".manifest.json"
            )
            native = json.loads(native_path.read_text(encoding="utf-8"))
            native["runtime"]["gated_delta_backend"] = FLA_BACKEND
            write_json(native_path, native)
            with self.assertRaisesRegex(ValueError, "stable native backend differs"):
                audit_prekey_predictions(plan)
            native["runtime"]["gated_delta_backend"] = TORCH_BACKEND
            write_json(native_path, native)
            bad_pair_plan = {**plan, "comparison_pairs_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "comparison pairs changed"):
                audit_prekey_predictions(bad_pair_plan)
            bad_smoke_plan = copy.deepcopy(plan)
            bad_smoke_plan["baseline_repeatability"]["nox"]["receipt_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "repeatability receipts changed"):
                audit_prekey_predictions(bad_smoke_plan)
            for field, wrong in (
                ("native_model_sha256", "0" * 64),
                ("adapter_sha256", "0" * 64),
                ("calibration_sha256", "0" * 64),
                ("revision", "wrong-revision"),
            ):
                with self.subTest(field=field):
                    bad_identity_plan = copy.deepcopy(plan)
                    bad_identity_plan["model_roster"][0][field] = wrong
                    with self.assertRaisesRegex(
                        ValueError, "planned baseline identity differs"
                    ):
                        audit_prekey_predictions(bad_identity_plan)
            first = Path(plan["inference"][0]["paths"]["typed"])
            original = first.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0].pop("revision_attested")
            first.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "native runtime differs"):
                audit_prekey_predictions(plan)
            rows[0]["revision_attested"] = True
            rows[0]["model_config_sha256"] = "0" * 64
            first.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "native runtime differs"):
                audit_prekey_predictions(plan)
            first.write_text(original, encoding="utf-8")
            open_control = Path(plan["inference"][1]["paths"]["typed"])
            open_original = open_control.read_text(encoding="utf-8")
            open_rows = [json.loads(line) for line in open_original.splitlines()]
            open_rows[0]["release_manifest_sha256"] = "0" * 64
            open_control.write_text(
                "".join(json.dumps(row) + "\n" for row in open_rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "native runtime differs"):
                audit_prekey_predictions(plan)
            open_control.write_text(open_original, encoding="utf-8")
            first.write_text(
                first.read_text(encoding="utf-8").replace(
                    "typed-0", "typed-tampered", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input or answer IDs differ"):
                audit_prekey_predictions(plan)


if __name__ == "__main__":
    unittest.main()
