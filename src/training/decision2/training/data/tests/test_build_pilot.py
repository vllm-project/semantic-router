import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class PilotDataTests(unittest.TestCase):
    def test_generated_oracles_and_balance(self):
        for index in range(40):
            row = pilot.make_composition("fixed", index)
            actual = row["audit_metadata"]["start"]
            for operation in row["audit_metadata"]["operations"]:
                actual = pilot._apply(actual, operation)
            self.assertEqual(row["options"][row["label"]]["description"], actual)
            self.assertEqual(len({o["description"] for o in row["options"]}), 4)
            pilot.validate_train_row(row)
        for index in range(40):
            row = pilot.make_reading("fixed", index)
            if row["task_type"] == "choice":
                self.assertEqual(
                    row["options"][row["label"]]["description"],
                    row["audit_metadata"]["wooden_box_item"],
                )
            else:
                expected = "true" if row["audit_metadata"]["oracle_result"] else "false"
                self.assertEqual(row["options"][row["label"]]["key"], expected)
            pilot.validate_train_row(row)
        abstentions = [pilot.make_abstention("fixed", i) for i in range(40)]
        self.assertEqual(
            sum(
                row["audit_metadata"]["oracle_result"] == "not stated"
                for row in abstentions
            ),
            20,
        )
        for row in abstentions:
            self.assertEqual(
                row["options"][row["label"]]["description"],
                row["audit_metadata"]["oracle_result"],
            )
            pilot.validate_train_row(row)

    def test_seed_is_deterministic_and_split_guard(self):
        self.assertEqual(
            pilot.make_composition("stable", 12), pilot.make_composition("stable", 12)
        )
        self.assertNotEqual(
            pilot.make_composition("stable", 12), pilot.make_composition("changed", 12)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.train.jsonl"
            row = pilot.make_composition("stable", 0)
            row["split"] = "final"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "non-train"):
                pilot.read_legacy_train(path)
            row["split"] = "train"
            row["family"] = "exception_stack"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "benchmark family"):
                pilot.read_legacy_train(path)

    def test_structured_legacy_fields_preserve_candidate_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.jsonl"
            old = pilot.make_composition("old", 9)
            old["source"] = {"dataset": "example_corpus", "source_id": "p7"}
            old["render_template"] = None
            old["options"][0]["description"] = {"nested": ["A", "B"]}
            old["input_sha256"] = pilot.input_sha256(old)
            write_jsonl(path, [old])
            new = pilot.read_legacy_train(path)[0]
            self.assertEqual(new["source"], "legacy:example_corpus")
            self.assertEqual(new["options"][0]["description"], {"nested": ["A", "B"]})
            self.assertEqual(new["render_template"], "legacy_unspecified")
            self.assertEqual(new["input_sha256"], old["input_sha256"])
            self.assertEqual(new["audit_metadata"]["original_source"], old["source"])
            pilot.validate_train_row(new)

    def test_cross_split_group_and_near_duplicate_audit(self):
        first = pilot.make_composition("old", 2)
        second = pilot.make_composition("other", 5)
        second["group_id"] = first["group_id"]
        audit = pilot.overlap_audit([first], [second])
        self.assertEqual(audit["group_id"]["count"], 1)
        self.assertTrue(pilot.audit_has_exact_overlap(audit))
        near = dict(first)
        near["id"] = "different-id"
        near["group_id"] = "different-group"
        near["state"] = first["state"] + " "
        self.assertGreaterEqual(pilot.near_duplicates([first], [near])["count"], 1)
        clean, receipt = pilot.quarantine_holdout_neighbors([first], [near])
        self.assertEqual(clean, [])
        self.assertEqual(receipt["excluded_groups"], 1)
        self.assertEqual(
            receipt["excluded_group_details"][0]["reasons"], ["near_input"]
        )

    def test_matched_replacement_keeps_parent_groups_and_token_budget(self):
        base = [pilot.make_composition("old", i) for i in range(10)]
        base[0]["group_id"] = base[1]["group_id"]
        new = [pilot.make_reading("new", i) for i in range(2)]
        lengths = {
            row["id"]: value
            for row, value in zip(
                base, (100, 130, 150, 180, 200, 220, 240, 260, 280, 300)
            )
        }
        lengths.update({new[0]["id"]: 175, new[1]["id"]: 205})
        kept, receipt = pilot.matched_legacy_subset(
            base, 8, new, lambda row: lengths[row["id"]], "fixed"
        )
        self.assertEqual(len(kept), 8)
        self.assertEqual(
            {base[0]["id"], base[1]["id"]} <= {row["id"] for row in kept}, True
        )
        self.assertLessEqual(abs(receipt["replacement_token_delta"]), 10)
        self.assertEqual(receipt["removed_groups"], 2)

    def test_stratified_replacement_preserves_groups_and_source_mix(self):
        base = [pilot.make_composition("old", i) for i in range(40)]
        for i, row in enumerate(base):
            row["family"] = "legacy_a" if i < 20 else "legacy_b"
            row["task_type"] = "choice"
            row["language"] = "en"
        base[0]["group_id"] = base[1]["group_id"]
        new = [pilot.make_reading("new", i) for i in range(8)]
        lengths = {row["id"]: i + 100 for i, row in enumerate([*base, *new])}
        kept, receipt = pilot.stratified_legacy_subset(
            base, 32, new, lambda row: lengths[row["id"]], "fixed"
        )
        self.assertEqual(len(kept), 32)
        self.assertEqual(
            receipt["removed_family_counts"], {"legacy_a": 4, "legacy_b": 4}
        )
        self.assertEqual(
            receipt["strategy"], "whole-source-group-family-task-language-stratified-v1"
        )
        self.assertEqual(receipt["removed_rows"], 8)
        kept_ids = {row["id"] for row in kept}
        self.assertEqual(base[0]["id"] in kept_ids, base[1]["id"] in kept_ids)

    def test_source_evidence_requires_explicit_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "licenses.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": {
                            "legacy:a": {
                                "license": "CC-BY-4.0",
                                "attribution": "Example",
                                "evidence": "pinned source card",
                            }
                        }
                    }
                )
            )
            records, receipt = pilot.read_source_license_evidence(path)
            self.assertEqual(records["legacy:a"]["license"], "CC-BY-4.0")
            self.assertEqual(
                receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            path.write_text(
                json.dumps({"sources": {"legacy:a": {"license": "CC-BY-4.0"}}})
            )
            with self.assertRaisesRegex(ValueError, "attribution"):
                pilot.read_source_license_evidence(path)

    def test_build_writes_manifest_and_keeps_references_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path, select_path, cal_path = (
                root / name for name in ("legacy.jsonl", "select.jsonl", "cal.jsonl")
            )
            legacy = [pilot.make_composition("old", i) for i in range(3)]
            for row in legacy:
                row["family"] = "legacy_composition"
            select = pilot.make_reading("holdout", 2)
            select["split"] = select["evaluation_role"] = "select"
            cal = pilot.make_abstention("holdout", 3)
            cal["split"] = "cal"
            cal["evaluation_role"] = "calibrate"
            write_jsonl(legacy_path, legacy)
            write_jsonl(select_path, [select])
            write_jsonl(cal_path, [cal])
            args = argparse.Namespace(
                legacy_train=legacy_path,
                select=select_path,
                cal=cal_path,
                output_dir=root / "output",
                seed="new",
                arm=["baseline:2:0:0:0", "mixed:1:1:0:0"],
                tokenizer=None,
                tokenizer_revision=None,
                max_row_tokens=None,
                max_arm_tokens=None,
                match_arm_tokens_percent=None,
                near_duplicate_policy="error",
                overwrite=False,
                derive_holdouts=False,
                select_count=None,
                cal_count=None,
            )
            manifests = pilot.build(args)
            self.assertEqual([m["output"]["rows"] for m in manifests], [2, 2])
            self.assertEqual(manifests[1]["counts"]["family"][pilot.FAMILIES[0]], 1)
            for manifest in manifests:
                output = args.output_dir / manifest["output"]["file"]
                self.assertEqual(
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                    manifest["output"]["sha256"],
                )
                ids = {
                    json.loads(line)["id"] for line in output.read_text().splitlines()
                }
                self.assertNotIn(select["id"], ids)
                self.assertNotIn(cal["id"], ids)
                self.assertEqual(
                    manifest["leakage_audit"]["train_select"]["group_id"]["count"], 0
                )
                self.assertEqual(manifest["token_audit"]["method"], "not measured")

    def test_derived_holdouts_reserve_groups_before_arm_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / "legacy.jsonl"
            legacy = [pilot.make_composition("old", i) for i in range(8)]
            for row in legacy:
                row["family"] = "legacy_composition"
            write_jsonl(legacy_path, legacy)
            args = argparse.Namespace(
                legacy_train=legacy_path,
                select=None,
                cal=None,
                output_dir=root / "output",
                seed="new",
                arm=["baseline:6:0:0:0", "mixed:5:1:0:0"],
                tokenizer=None,
                tokenizer_revision=None,
                max_row_tokens=None,
                max_arm_tokens=None,
                match_arm_tokens_percent=None,
                near_duplicate_policy="report",
                overwrite=False,
                derive_holdouts=True,
                select_count=1,
                cal_count=1,
            )
            manifests = pilot.build(args)
            select = [
                json.loads(line)
                for line in (args.output_dir / "select.jsonl").read_text().splitlines()
            ]
            cal = [
                json.loads(line)
                for line in (args.output_dir / "cal.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                (select[0]["split"], select[0]["evaluation_role"]), ("select", "select")
            )
            self.assertEqual(
                (cal[0]["split"], cal[0]["evaluation_role"]), ("cal", "calibrate")
            )
            for manifest in manifests:
                train = [
                    json.loads(line)
                    for line in (args.output_dir / manifest["output"]["file"])
                    .read_text()
                    .splitlines()
                ]
                for ref in (select, cal):
                    self.assertFalse(
                        {row["id"] for row in train} & {row["id"] for row in ref}
                    )
                    self.assertFalse(
                        {row["group_id"] for row in train}
                        & {row["group_id"] for row in ref}
                    )
                    self.assertFalse(
                        {row["input_sha256"] for row in train}
                        & {row["input_sha256"] for row in ref}
                    )
                checked = {
                    "train": load_partition(
                        args.output_dir / manifest["output"]["file"], "train"
                    ),
                    "select": load_partition(
                        args.output_dir / "select.jsonl", "select"
                    ),
                    "cal": load_partition(args.output_dir / "cal.jsonl", "cal"),
                }
                check_partition_isolation(checked)
            holdouts = json.loads(
                (args.output_dir / "holdouts.manifest.json").read_text()
            )
            self.assertEqual(holdouts["outputs"]["select.jsonl"]["rows"], 1)
            self.assertEqual(holdouts["outputs"]["cal.jsonl"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
