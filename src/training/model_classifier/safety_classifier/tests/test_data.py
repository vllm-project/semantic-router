from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.training.model_classifier.safety_classifier.config import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
)
from src.training.model_classifier.safety_classifier.data import (
    DEFAULT_LEVEL1_PER_LABEL,
    DEFAULT_LEVEL2_PER_LABEL,
    InsufficientClassSamplesError,
    SchemaError,
    build_level1_dataset,
    build_level2_dataset,
    materialize_dataset,
    normalize_prompt,
    prepare_materialized_data,
    prompt_fingerprint,
    validate_aegis_row,
    validate_synth_row,
)
from src.training.model_classifier.safety_classifier.taxonomy import (
    LEVEL2_LABELS,
    UnknownCategoryError,
)


def _row_id(prompt: str, label: str, categories: str) -> str:
    value = json.dumps([prompt, label, categories], ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def aegis_row(
    prompt: str,
    *,
    label: str = "unsafe",
    categories: str = "Violence",
    row_id: str | None = None,
    response: str | None = "ignored response",
    response_label: str | None = "unsafe",
) -> dict[str, object]:
    return {
        "id": row_id or _row_id(prompt, label, categories),
        "reconstruction_id_if_redacted": None,
        "prompt": prompt,
        "response": response,
        "prompt_label": label,
        "response_label": response_label,
        "violated_categories": categories,
        "prompt_label_source": "fixture",
        "response_label_source": "fixture" if response_label else None,
    }


def synth_row(text: str, category: str) -> dict[str, object]:
    return {"text": text, "category": category, "label": 1}


def level2_fixture() -> (
    tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]
):
    aegis = {
        "train": [
            aegis_row("violent prompt", categories="Violence"),
            aegis_row("sexual prompt", categories="Sexual"),
            aegis_row("hate prompt", categories="Hate/Identity Hate"),
            aegis_row(
                "privacy first prompt",
                categories="Needs Caution, PII/Privacy, Violence",
            ),
        ],
        "validation": [],
        "test": [],
    }
    synth = [
        synth_row("crime prompt", "S2_non_violent_crimes"),
        synth_row("advice prompt", "S6_specialized_advice"),
        synth_row("weapons prompt", "S9_indiscriminate_weapons"),
        synth_row("self harm prompt", "S11_suicide_self_harm"),
        synth_row("election prompt", "S13_elections"),
    ]
    return aegis, synth


class NormalizationAndSchemaTest(unittest.TestCase):
    def test_fingerprint_uses_nfkc_whitespace_collapse_and_casefold(self) -> None:
        left = "  \uff28\uff25\uff2c\uff2c\uff2f\tStraße\n"
        right = "hello strasse"

        self.assertEqual(normalize_prompt(left), right)
        self.assertEqual(prompt_fingerprint(left), prompt_fingerprint(right))

    def test_aegis_schema_rejects_missing_and_extra_columns(self) -> None:
        missing = aegis_row("prompt")
        missing.pop("response")
        with self.assertRaisesRegex(SchemaError, "missing"):
            validate_aegis_row(missing)

        extra = aegis_row("prompt")
        extra["new_column"] = "taxonomy drift"
        with self.assertRaisesRegex(SchemaError, "unexpected"):
            validate_aegis_row(extra)

    def test_synthetic_schema_requires_integer_unsafe_label(self) -> None:
        row = synth_row("prompt", "S2_non_violent_crimes")
        row["label"] = True
        with self.assertRaisesRegex(SchemaError, "integer 1"):
            validate_synth_row(row)

    def test_unknown_category_fails_even_on_safe_row(self) -> None:
        rows = {
            "train": [
                aegis_row("safe", label="safe", categories="New Hazard"),
                aegis_row("unsafe"),
            ]
        }
        with self.assertRaisesRegex(UnknownCategoryError, "New Hazard"):
            build_level1_dataset(rows, per_label=1)


class Level1DataTest(unittest.TestCase):
    def test_prompt_only_and_prompt_filters(self) -> None:
        rows = {
            "train": [
                aegis_row("safe train", label="safe", categories=""),
                aegis_row("unsafe train"),
                aegis_row(" REDACTED ", row_id="redacted"),
                aegis_row(" \n\t ", row_id="empty"),
            ],
            "validation": [
                aegis_row(
                    "safe evaluation prompt",
                    label="safe",
                    categories="",
                    response="an unsafe response that must not become input",
                    response_label="unsafe",
                )
            ],
            "test": [],
        }

        build = build_level1_dataset(rows, per_label=1)
        evaluation = build.split_samples("validation")[0]

        self.assertEqual(evaluation.label_name, "safe")
        self.assertEqual(evaluation.text, "safe evaluation prompt")
        self.assertNotIn("response", evaluation.to_dict())
        self.assertEqual(evaluation.to_dict()["label"], "safe")
        self.assertEqual(evaluation.to_dict()["label_id"], 0)
        self.assertNotIn("label_name", evaluation.to_dict())
        self.assertEqual(build.manifest["audit"]["redacted_prompt_rows"], 1)
        self.assertEqual(build.manifest["audit"]["empty_prompt_rows"], 1)

    def test_conflicts_drop_groups_and_split_precedence_prevents_leakage(self) -> None:
        rows = {
            "train": [
                aegis_row("baseline safe", label="safe", categories=""),
                aegis_row("baseline unsafe"),
                aegis_row("Cross Split", label="safe", categories="", row_id="ct"),
                aegis_row("Conflict", label="safe", categories="", row_id="cf-t"),
            ],
            "validation": [
                aegis_row(
                    " cross\t split ",
                    label="safe",
                    categories="",
                    row_id="cv",
                )
            ],
            "test": [
                aegis_row("CROSS SPLIT", label="safe", categories="", row_id="ce"),
                aegis_row("conflict", label="unsafe", row_id="cf-e"),
            ],
        }

        build = build_level1_dataset(rows, per_label=1)
        all_samples = list(build.samples)
        cross = prompt_fingerprint("cross split")
        conflict = prompt_fingerprint("conflict")

        cross_samples = [
            sample for sample in all_samples if sample.fingerprint == cross
        ]
        self.assertEqual(len(cross_samples), 1)
        self.assertEqual(cross_samples[0].split, "test")
        self.assertFalse(any(sample.fingerprint == conflict for sample in all_samples))
        self.assertEqual(build.manifest["audit"]["label_conflict_groups"], 1)
        self.assertEqual(build.manifest["audit"]["label_conflict_rows"], 2)
        self.assertEqual(build.manifest["audit"]["lower_precedence_rows_dropped"], 2)

        split_fingerprints = {
            split: {sample.fingerprint for sample in build.split_samples(split)}
            for split in ("train", "validation", "test")
        }
        self.assertTrue(
            split_fingerprints["train"].isdisjoint(split_fingerprints["test"])
        )
        self.assertTrue(
            split_fingerprints["validation"].isdisjoint(split_fingerprints["test"])
        )

    def test_aegis_wins_same_label_duplicate_over_synthetic(self) -> None:
        rows = {
            "train": [
                aegis_row("safe", label="safe", categories=""),
                aegis_row("duplicate unsafe", row_id="aegis-winner"),
            ]
        }
        synth = [synth_row(" DUPLICATE\tUNSAFE ", "S2_non_violent_crimes")]

        build = build_level1_dataset(rows, synth, per_label=1)
        unsafe = next(
            sample
            for sample in build.split_samples("train")
            if sample.label_name == "unsafe"
        )
        self.assertEqual(unsafe.source, "aegis")
        self.assertEqual(unsafe.source_id, "aegis-winner")

    def test_stable_hash_sampling_is_exact_and_input_order_independent(self) -> None:
        train = [
            *(
                aegis_row(f"safe {index}", label="safe", categories="")
                for index in range(4)
            ),
            *(aegis_row(f"unsafe {index}") for index in range(4)),
        ]
        synth = [
            synth_row(f"synthetic unsafe {index}", "S2_non_violent_crimes")
            for index in range(2)
        ]

        first = build_level1_dataset({"train": train}, synth, per_label=2, seed=7)
        second = build_level1_dataset(
            {"train": list(reversed(train))},
            list(reversed(synth)),
            per_label=2,
            seed=7,
        )

        self.assertEqual(
            [sample.to_dict() for sample in first.samples],
            [sample.to_dict() for sample in second.samples],
        )
        self.assertEqual(first.manifest, second.manifest)
        counts = Counter(sample.label_name for sample in first.split_samples("train"))
        self.assertEqual(counts, {"safe": 2, "unsafe": 2})
        self.assertEqual(DEFAULT_LEVEL1_PER_LABEL, 10_000)

    def test_level1_never_oversamples(self) -> None:
        rows = {
            "train": [
                aegis_row("only safe", label="safe", categories=""),
                aegis_row("only unsafe"),
            ]
        }
        with self.assertRaisesRegex(InsufficientClassSamplesError, "requires 2"):
            build_level1_dataset(rows, per_label=2)


class Level2DataTest(unittest.TestCase):
    def test_first_mapped_label_and_deterministic_balancing(self) -> None:
        aegis, synth = level2_fixture()

        first = build_level2_dataset(aegis, synth, per_label=2, seed=11)
        second = build_level2_dataset(
            {
                split: list(reversed(rows))
                for split, rows in reversed(list(aegis.items()))
            },
            list(reversed(synth)),
            per_label=2,
            seed=11,
        )

        self.assertEqual(
            [sample.to_dict() for sample in first.samples],
            [sample.to_dict() for sample in second.samples],
        )
        self.assertEqual(first.manifest, second.manifest)
        counts = Counter(sample.label_name for sample in first.split_samples("train"))
        self.assertEqual(counts, dict.fromkeys(LEVEL2_LABELS, 2))
        self.assertEqual(first.manifest["audit"]["train_rows_oversampled"], 9)
        self.assertEqual(first.manifest["audit"]["level2_multitarget_rows"], 1)
        self.assertEqual(DEFAULT_LEVEL2_PER_LABEL, 2_000)

        privacy_samples = [
            sample for sample in first.samples if sample.text == "privacy first prompt"
        ]
        self.assertEqual(
            {sample.label_name for sample in privacy_samples}, {"S9_privacy"}
        )
        self.assertEqual({sample.occurrence for sample in privacy_samples}, {0, 1})
        self.assertTrue(
            all(sample.to_dict()["is_multitarget"] for sample in privacy_samples)
        )


class MaterializationTest(unittest.TestCase):
    def test_materializes_jsonl_and_auditable_manifest(self) -> None:
        rows = {
            "train": [
                aegis_row("safe", label="safe", categories=""),
                aegis_row("unsafe"),
            ],
            "validation": [],
            "test": [],
        }
        build = build_level1_dataset(rows, per_label=1)

        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = materialize_dataset(
                build,
                temporary_dir,
                provenance={"fixture": True},
            )
            target = Path(temporary_dir)
            stored_manifest = json.loads(
                (target / "data_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(stored_manifest, manifest)
            self.assertEqual(manifest["provenance"], {"fixture": True})
            for split in ("train", "validation", "test"):
                payload = (target / f"{split}.jsonl").read_bytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    manifest["files"][split]["sha256"],
                )
            train_rows = [
                json.loads(line)
                for line in (target / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(train_rows), 2)
            self.assertTrue(all("response" not in row for row in train_rows))

    def test_prepare_uses_fixture_downloader_without_network(self) -> None:
        aegis, synth = level2_fixture()
        aegis["train"].append(aegis_row("safe prompt", label="safe", categories=""))

        with tempfile.TemporaryDirectory() as temporary_dir:
            fixture_root = Path(temporary_dir) / "fixtures"
            fixture_root.mkdir()
            files: dict[tuple[str, str], Path] = {}
            contract = copy.deepcopy(load_contract(DEFAULT_CONTRACT_PATH))
            contract["data"]["binary_train_per_label"] = 1
            contract["data"]["hazard_train_per_label"] = 1

            for split in ("train", "validation", "test"):
                path = fixture_root / f"aegis-{split}.json"
                path.write_text(
                    json.dumps(aegis[split], ensure_ascii=False), encoding="utf-8"
                )
                spec = contract["datasets"]["aegis"]["files"][split]
                spec["path"] = path.name
                spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                files[(contract["datasets"]["aegis"]["id"], path.name)] = path

            synth_path = fixture_root / "synthetic.jsonl"
            synth_path.write_text(
                "".join(json.dumps(row) + "\n" for row in synth),
                encoding="utf-8",
            )
            synth_spec = contract["datasets"]["synthetic"]["files"]["train"]
            synth_spec["path"] = synth_path.name
            synth_spec["sha256"] = hashlib.sha256(synth_path.read_bytes()).hexdigest()
            files[(contract["datasets"]["synthetic"]["id"], synth_path.name)] = (
                synth_path
            )

            contract_path = fixture_root / "contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            calls: list[dict[str, str]] = []

            def fixture_downloader(**kwargs: str) -> str:
                calls.append(kwargs)
                return str(files[(kwargs["repo_id"], kwargs["filename"])])

            output = Path(temporary_dir) / "output"
            manifests = prepare_materialized_data(
                output,
                contract_path=contract_path,
                downloader=fixture_downloader,
            )

            self.assertEqual(len(calls), 4)
            self.assertTrue(all(call["revision"] for call in calls))
            self.assertEqual(manifests["level1"]["splits"]["train"]["rows"], 2)
            self.assertEqual(manifests["level2"]["splits"]["train"]["rows"], 9)
            for task in ("level1", "level2"):
                self.assertTrue((output / task / "train.jsonl").is_file())
                self.assertTrue((output / task / "data_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
