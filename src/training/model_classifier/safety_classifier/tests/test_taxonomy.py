from __future__ import annotations

import unittest

from src.training.model_classifier.safety_classifier.taxonomy import (
    EXCLUDED_AEGIS_CATEGORIES,
    LEVEL2_LABEL_TO_ID,
    LEVEL2_LABELS,
    SYNTH_CATEGORY_TO_LABEL,
    UnknownCategoryError,
    map_aegis_categories,
    map_synth_category,
)


class TaxonomyTest(unittest.TestCase):
    def test_legacy_label_ids_are_stable(self) -> None:
        self.assertEqual(
            LEVEL2_LABELS,
            (
                "S1_violent_crimes",
                "S2_nonviolent_crimes",
                "S3_sex_crimes",
                "S5_weapons_cbrne",
                "S6_self_harm",
                "S7_hate",
                "S8_specialized_advice",
                "S9_privacy",
                "S13_misinformation",
            ),
        )
        self.assertEqual(
            LEVEL2_LABEL_TO_ID,
            {label: label_id for label_id, label in enumerate(LEVEL2_LABELS)},
        )

    def test_aegis_uses_first_mapped_target_in_source_order(self) -> None:
        mapping = map_aegis_categories("Needs Caution, PII/Privacy, Violence, Threat")

        self.assertEqual(mapping.selected_label, "S9_privacy")
        self.assertEqual(
            mapping.mapped_targets,
            ("S9_privacy", "S1_violent_crimes"),
        )
        self.assertTrue(mapping.is_multilabel)
        self.assertTrue(mapping.is_multitarget)

    def test_categories_mapping_to_same_target_are_not_multitarget(self) -> None:
        mapping = map_aegis_categories("Threat, Violence")

        self.assertEqual(mapping.mapped_targets, ("S1_violent_crimes",))
        self.assertTrue(mapping.is_multilabel)
        self.assertFalse(mapping.is_multitarget)

    def test_explicit_exclusions_have_no_legacy_target(self) -> None:
        value = ", ".join(sorted(EXCLUDED_AEGIS_CATEGORIES))
        mapping = map_aegis_categories(value)

        self.assertIsNone(mapping.selected_label)
        self.assertEqual(mapping.mapped_targets, ())

    def test_unknown_aegis_category_fails_closed(self) -> None:
        with self.assertRaisesRegex(UnknownCategoryError, "New Hazard"):
            map_aegis_categories("Violence, New Hazard")

    def test_all_synthetic_categories_have_explicit_targets(self) -> None:
        expected = {
            "S2_non_violent_crimes": "S2_nonviolent_crimes",
            "S6_specialized_advice": "S8_specialized_advice",
            "S7_privacy": "S9_privacy",
            "S9_indiscriminate_weapons": "S5_weapons_cbrne",
            "S11_suicide_self_harm": "S6_self_harm",
            "S13_elections": "S13_misinformation",
        }
        self.assertEqual(SYNTH_CATEGORY_TO_LABEL, expected)
        self.assertEqual(
            {category: map_synth_category(category) for category in expected},
            expected,
        )

    def test_unknown_synthetic_category_fails_closed(self) -> None:
        with self.assertRaisesRegex(UnknownCategoryError, "S99_new_hazard"):
            map_synth_category("S99_new_hazard")


if __name__ == "__main__":
    unittest.main()
