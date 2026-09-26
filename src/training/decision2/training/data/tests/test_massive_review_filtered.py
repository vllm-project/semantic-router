"""Independent verdicts remove whole parallel groups without approving training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.data import build_massive_expanded_review as expanded
from training.data import build_massive_multilingual as massive
from training.data import build_massive_review_filtered as filtered


class ReviewFilteredTests(unittest.TestCase):
    def test_all_verdicts_required_and_rejected_group_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, review, rights, output = (
                root / "candidate",
                root / "expanded",
                root / "source",
                root / "filtered",
            )
            for path in (candidate, review, rights):
                path.mkdir()
            source_ids = sorted(filtered.MANDATORY_EXCLUSIONS, key=int)
            source_ids.extend(str(20000 + i) for i in range(600 - len(source_ids)))
            train = []
            for source_id in source_ids:
                for locale in massive.LOCALES:
                    train.append(
                        {
                            "id": f"massive-1.1:{source_id}:{locale}",
                            "group_id": f"massive-1.1:{source_id}",
                            "input_sha256": f"train-{source_id}-{locale}",
                            "audit_metadata": {
                                "source_id": source_id,
                                "intent": "alarm_set",
                                "source_locale": locale,
                            },
                        }
                    )
            dev = [
                {
                    "id": f"dev-{i}",
                    "group_id": f"dev-group-{i//7}",
                    "input_sha256": f"dev-hash-{i}",
                }
                for i in range(1400)
            ]
            massive.write_jsonl(candidate / "train.private.jsonl", train)
            massive.write_jsonl(candidate / "dev.private.jsonl", dev)
            manifest = {
                "training_approved": False,
                "selected_source_groups": {"train": 600, "dev": 200},
                "outputs": {
                    name: {"sha256": massive.sha(candidate / name)}
                    for name in ("train.private.jsonl", "dev.private.jsonl")
                },
                "intent_coverage": {"dev": 52},
                "source": {"attribution": ["MASSIVE", "SLURP"]},
            }
            (candidate / "manifest.json").write_text(json.dumps(manifest))
            manifest_sha = massive.sha(candidate / "manifest.json")
            packet = [
                {"review_id": expanded.review_id(source_id), "utterance": "synthetic"}
                for source_id in source_ids
            ]
            key = [
                {"review_id": expanded.review_id(source_id), "source_id": source_id}
                for source_id in source_ids
            ]
            massive.write_jsonl(review / "english-all600.blind.private.jsonl", packet)
            massive.write_jsonl(review / "english-all600.key.private.jsonl", key)
            receipt = {
                "candidate_manifest_sha256": manifest_sha,
                "candidate_train_sha256": manifest["outputs"]["train.private.jsonl"][
                    "sha256"
                ],
                "blind_review_rows": 600,
                "files": {
                    name: massive.sha(review / name)
                    for name in (
                        "english-all600.blind.private.jsonl",
                        "english-all600.key.private.jsonl",
                    )
                },
            }
            (review / "receipt.json").write_text(json.dumps(receipt))
            verdict_path = root / "verdicts.jsonl"
            bad_source = source_ids[-1]
            verdicts = [
                {
                    "review_id": expanded.review_id(source_id),
                    "verdict": "ambiguous" if source_id == bad_source else "pass",
                    "reason": "two actions" if source_id == bad_source else "clear",
                }
                for source_id in source_ids
            ]
            massive.write_jsonl(verdict_path, verdicts)
            excluded = set(filtered.MANDATORY_EXCLUSIONS) | {bad_source}
            exclusion_path = root / "exclusion.json"
            exclusion = {
                "blind_review_sha256": filtered.EXPANDED_BLIND_SHA,
                "key_sha256": receipt["files"]["english-all600.key.private.jsonl"],
                "force_exclude_prior_17_count": len(filtered.MANDATORY_EXCLUSIONS),
                "excluded_source_ids": sorted(excluded, key=int),
                "reason_by_source_id": dict.fromkeys(excluded, "review"),
                "union_excluded_count": len(excluded),
                "retained_source_groups": 582,
                "retained_intent_coverage": 1,
                "retained_per_intent": {"alarm_set": 582},
            }
            exclusion_path.write_text(json.dumps(exclusion))
            report_path = root / "review-report.json"
            report = {
                "training_approved": False,
                "blind_review_sha256": filtered.EXPANDED_BLIND_SHA,
                "packet_sha256": receipt["files"]["english-all600.blind.private.jsonl"],
                "key_sha256": receipt["files"]["english-all600.key.private.jsonl"],
                "verdict_sha256": massive.sha(verdict_path),
                "exclusion_sha256": massive.sha(exclusion_path),
                "excluded_source_groups": 18,
                "retained_source_groups": 582,
                "retained_intent_coverage": 1,
                "counts": {"pass": 599, "ambiguous": 1},
            }
            report_path.write_text(json.dumps(report))
            (rights / "LICENSE").write_text("CC BY 4.0")
            (rights / "NOTICE.md").write_text("MASSIVE and SLURP attribution")
            with mock.patch.object(
                massive, "LICENSE_SHA", massive.sha(rights / "LICENSE")
            ), mock.patch.object(
                massive, "NOTICE_SHA", massive.sha(rights / "NOTICE.md")
            ):
                self.assertRaisesRegex(
                    ValueError,
                    "cover every source group",
                    filtered.load_verdicts,
                    verdict_path,
                    massive.sha(verdict_path),
                    {expanded.review_id(source_id) for source_id in source_ids[:-1]},
                )
                result = filtered.build(
                    candidate,
                    manifest_sha,
                    review,
                    massive.sha(review / "receipt.json"),
                    verdict_path,
                    massive.sha(verdict_path),
                    exclusion_path,
                    massive.sha(exclusion_path),
                    report_path,
                    massive.sha(report_path),
                    rights,
                    output,
                )
            self.assertFalse(result["training_approved"])
            self.assertEqual(result["source_groups"]["kept_train"], 582)
            self.assertEqual(result["source_groups"]["excluded_train"], 18)
            self.assertEqual(result["rows"]["train"], 4074)
            self.assertEqual(result["rows"]["dev"], 1400)
            kept = filtered.load_jsonl(output / "train.private.jsonl")
            self.assertNotIn(
                bad_source, {row["audit_metadata"]["source_id"] for row in kept}
            )
            self.assertTrue(
                filtered.MANDATORY_EXCLUSIONS.isdisjoint(
                    {row["audit_metadata"]["source_id"] for row in kept}
                )
            )
            self.assertEqual(
                massive.sha(output / "dev.private.jsonl"),
                manifest["outputs"]["dev.private.jsonl"]["sha256"],
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertTrue(
                all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
            )


if __name__ == "__main__":
    unittest.main()
