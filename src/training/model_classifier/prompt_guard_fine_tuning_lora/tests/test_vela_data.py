import unittest

from src.training.model_classifier.prompt_guard_fine_tuning_lora.vela_data import (
    annotation_conflicts,
    normalize_annotations,
    team_partition,
)


class VelaPromptGuardDataTests(unittest.TestCase):
    def test_boolean_annotations_and_phase_conflicts(self):
        first = {
            "A text": {"attack_attempt": True},
            " a text ": {"attack_attempt": False},
        }
        known, conflicts = annotation_conflicts(first)
        known, later = annotation_conflicts({"A text": {"attack_attempt": True}}, known)
        self.assertEqual(len(conflicts | later), 1)
        labels, audit = normalize_annotations({"A text": {"attack_attempt": True}})
        self.assertEqual(len(labels), 1)
        self.assertEqual(audit["source_boolean_annotations"], 1)

    def test_normalized_annotation_conflicts_are_removed(self):
        labels, audit = normalize_annotations(
            {
                "Do something": {"attack_attempt": "True"},
                " do something ": {"attack_attempt": "False"},
                "Uncertain": {"attack_attempt": "Unclear"},
            }
        )
        self.assertFalse(labels)
        self.assertEqual(audit["conflicting_normalized_annotations"], 1)
        self.assertEqual(audit["source_unclear_or_invalid"], 1)

    def test_team_is_always_in_same_partition(self):
        self.assertEqual(team_partition({"team-a"}), team_partition({"team-a"}))

    def test_shared_text_across_partition_teams_is_excluded(self):
        first = "team-a"
        other = next(
            f"team-{i}"
            for i in range(100)
            if team_partition({f"team-{i}"}) != team_partition({first})
        )
        self.assertIsNone(team_partition({first, other}))


if __name__ == "__main__":
    unittest.main()
