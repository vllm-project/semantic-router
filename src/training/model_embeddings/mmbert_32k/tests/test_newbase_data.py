"""Family cycles, complete-text locks and multi-positive retrieval admission."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.training.model_embeddings.mmbert_32k.newbase_objectives import (
        multi_positive_loss,
    )

from src.training.model_embeddings.mmbert_32k.newbase_data import (
    FrozenCorpus,
    GroupCycle,
    file_digest,
    retrieval_masks,
    text_digest,
    validate_record,
)


class NewBaseDataTest(unittest.TestCase):
    def test_transitive_parent_overlap_is_one_family(self):
        records = [
            {"id": "a", "parent_groups": ["A", "B"]},
            {"id": "b", "parent_groups": ["B", "C"]},
            {"id": "c", "parent_groups": ["C"]},
            {"id": "d", "parent_groups": ["D"]},
        ]
        left, right = GroupCycle(records, 12), GroupCycle(records, 12)
        self.assertEqual(set(left.groups), {("A", "B", "C"), ("D",)})
        for _ in range(10):
            batch = left.draw(2)
            self.assertEqual(batch, right.draw(2))
            self.assertEqual(sum(row["id"] == "d" for row in batch), 1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            left.draw(3)

    def test_batch_crossing_cycle_boundary_preserves_without_replacement(self):
        records = [{"id": str(i), "parent_groups": [str(i)]} for i in range(5)]
        sampler = GroupCycle(records, 123)
        for _ in range(11):
            batch = sampler.draw(3)
            self.assertEqual(len({row["id"] for row in batch}), 3)
        counts = list(sampler.draw_counts.values())
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_related_unknowns_masked_but_judged_negatives_preserved(self):
        components = {
            key: {"normalized_sha256": key, "parent_groups": parents}
            for key, parents in {"p": ["A"], "q": ["A"], "n": ["A"], "u": ["B"]}.items()
        }
        rows = [
            {"positive_component_ids": ["p"], "judged_negative_component_ids": ["n"]}
        ]
        positive, valid = retrieval_masks(rows, ["p", "q", "n", "u"], components)
        self.assertEqual(positive, [[True, False, False, False]])
        self.assertEqual(valid, [[True, False, True, True]])
        components["q"]["normalized_sha256"] = "p"
        with self.assertRaisesRegex(ValueError, "Identical"):
            retrieval_masks(rows, ["p", "q"], components)

    def test_wrong_partition_and_modified_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            components = [
                {
                    "id": key,
                    "text": text,
                    "text_sha256": text_digest(text),
                    "normalized_sha256": text_digest(text, normalize=True),
                    "parent_groups": [key],
                }
                for key, text in (("q", "query"), ("p", "passage"))
            ]
            records = [
                {
                    "id": "r",
                    "source": "miracl",
                    "language": "en",
                    "split": "train",
                    "parent_groups": ["r"],
                    "query_component_id": "q",
                    "positive_component_ids": ["p"],
                    "judged_negative_component_ids": [],
                    "unjudged_component_ids": [],
                }
            ]
            for filename, values in (
                ("components.jsonl", components),
                ("records.jsonl", records),
            ):
                (root / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in values)
                )
            manifest = {
                "split": "train",
                "files": {
                    file: {"sha256": file_digest(root / file)}
                    for file in ("components.jsonl", "records.jsonl")
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "partition"):
                FrozenCorpus.load(root, "validation")
            self.assertEqual(len(FrozenCorpus.load(root, "train").records), 1)
            (root / "components.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                FrozenCorpus.load(root, "train")

    def test_weak_ranking_preferences_cannot_relabel_a_judged_candidate(self):
        components = {key: {} for key in ("q", "p", "n", "u")}
        row = {
            "id": "r",
            "source": "retrieval",
            "language": "en",
            "split": "train",
            "parent_groups": ["query"],
            "query_component_id": "q",
            "positive_component_ids": ["p"],
            "judged_negative_component_ids": ["n"],
            "unjudged_component_ids": ["u"],
            "candidate_component_ids": ["p", "n", "u"],
        }
        for allowed in ([], ["u"]):
            row["unjudged_preference_component_ids"] = allowed
            validate_record(row, components, "train")
        for invalid in (["p"], ["n"], ["absent"], ["u", "u"]):
            row["unjudged_preference_component_ids"] = invalid
            with self.assertRaises(ValueError):
                validate_record(row, components, "train")
        self.assertEqual(row["unjudged_component_ids"], ["u"])
        self.assertEqual(row["judged_negative_component_ids"], ["n"])

    def test_explicit_preference_retains_only_its_unjudged_alternative(self):
        components = {
            key: {"normalized_sha256": key, "parent_groups": ["same-article"]}
            for key in ("p", "alternative", "other")
        }
        row = {
            "positive_component_ids": ["p"],
            "judged_negative_component_ids": [],
            "unjudged_component_ids": ["alternative", "other"],
            "candidate_component_ids": ["p", "alternative", "other"],
        }
        self.assertEqual(
            retrieval_masks([row], list(components), components)[1],
            [[True, False, False]],
        )
        row["contrastive_preference_component_ids"] = ["alternative"]
        positive, valid = retrieval_masks([row], list(components), components)
        self.assertEqual(positive, [[True, False, False]])
        self.assertEqual(valid, [[True, True, False]])
        self.assertEqual(row["judged_negative_component_ids"], [])
        for invalid in (["p"], ["missing"], ["alternative", "alternative"]):
            row["contrastive_preference_component_ids"] = invalid
            with self.assertRaises(ValueError):
                retrieval_masks([row], list(components), components)

    def test_ignored_unknown_retains_pool_and_known_judgments(self):
        components = {
            key: {"normalized_sha256": key, "parent_groups": [key]}
            for key in ("q", "p", "n", "u", "other")
        }
        row = {
            "id": "r",
            "source": "retrieval",
            "language": "en",
            "split": "train",
            "parent_groups": ["query"],
            "query_component_id": "q",
            "positive_component_ids": ["p"],
            "judged_negative_component_ids": ["n"],
            "unjudged_component_ids": ["u", "other"],
            "candidate_component_ids": ["p", "n", "u", "other"],
        }
        documents = row["candidate_component_ids"][:]
        baseline = retrieval_masks([row], documents, components)
        row["contrastive_ignored_component_ids"] = []
        self.assertEqual(retrieval_masks([row], documents, components), baseline)
        row["contrastive_ignored_component_ids"] = ["u"]
        validate_record(row, components, "train")
        positive, valid = retrieval_masks([row], documents, components)
        self.assertEqual(positive, baseline[0])
        self.assertEqual(valid, [[True, True, False, True]])
        self.assertEqual(row["candidate_component_ids"], documents)
        self.assertEqual(row["unjudged_component_ids"], ["u", "other"])

        for invalid in (["p"], ["n"], ["absent"], ["u", "u"]):
            row["contrastive_ignored_component_ids"] = invalid
            with self.assertRaises(ValueError):
                validate_record(row, components, "train")
            with self.assertRaises(ValueError):
                retrieval_masks([row], documents, components)
        row["contrastive_ignored_component_ids"] = ["u"]
        row["contrastive_preference_component_ids"] = ["u"]
        with self.assertRaises(ValueError):
            retrieval_masks([row], documents, components)

    @unittest.skipIf(torch is None, "torch is required for gradient validation")
    def test_ignored_logit_has_zero_gradient_with_complete_candidate_pool(self):
        components = {
            key: {"normalized_sha256": key, "parent_groups": [key]}
            for key in ("p", "n", "u", "other")
        }
        row = {
            "positive_component_ids": ["p"],
            "judged_negative_component_ids": ["n"],
            "unjudged_component_ids": ["u", "other"],
            "candidate_component_ids": ["p", "n", "u", "other"],
            "contrastive_ignored_component_ids": ["u"],
        }
        positive, valid = retrieval_masks([row], list(components), components)
        scores = torch.tensor([[1.0, 2.0, 100.0, 3.0]], requires_grad=True)
        loss = multi_positive_loss(scores, torch.tensor(positive), torch.tensor(valid))
        loss.backward()
        self.assertEqual(scores.grad[0, 2].item(), 0)
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 1].item(), 0)
        self.assertGreater(scores.grad[0, 3].item(), 0)


if __name__ == "__main__":
    unittest.main()
