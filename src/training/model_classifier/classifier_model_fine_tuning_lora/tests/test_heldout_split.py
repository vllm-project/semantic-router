"""Contract tests for the MMLU-Pro held-out reservation.

Guards the invariant behind
https://github.com/vllm-project/semantic-router/issues/3558: the intent
classifier's training pool is carved out of MMLU-Pro 'test', so nothing the
model is scored on may reach the gradient path.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter

from src.training.model_classifier.classifier_model_fine_tuning_lora.heldout_split import (
    HELDOUT_FRACTION,
    MANIFEST_NAME,
    SPLIT_SEED,
    build_manifest,
    drop_reserved_questions,
    reserve_heldout,
)

CATEGORIES = ["biology", "law", "math", "other"]

# A generic stem, the shape of question MMLU-Pro actually repeats across rows.
SHARED_STEM = "Which of the following statements is false?"


def corpus(rows_per_category: int = 100) -> tuple[list[str], list[str]]:
    """A stand-in split with unique question text in every row."""
    texts, labels = [], []
    for category in CATEGORIES:
        for i in range(rows_per_category):
            texts.append(f"{category} question {i}")
            labels.append(category)
    return texts, labels


class ReserveHeldoutTest(unittest.TestCase):
    def test_pool_and_heldout_partition_the_rows(self):
        texts, labels = corpus()
        pool, heldout = reserve_heldout(texts, labels)

        self.assertEqual(sorted(pool + heldout), list(range(len(texts))))
        self.assertFalse(set(pool) & set(heldout))
        self.assertEqual(pool, sorted(pool))
        self.assertEqual(heldout, sorted(heldout))

    def test_no_question_is_on_both_sides(self):
        texts, labels = corpus()
        pool, heldout = reserve_heldout(texts, labels)

        self.assertFalse({texts[i] for i in pool} & {texts[i] for i in heldout})

    def test_a_repeated_question_never_straddles_the_split(self):
        # MMLU-Pro repeats 200 question texts across 431 rows of 'test'. Reserving
        # by row index alone leaves those on both sides, which is the leak.
        #
        # Five identical rows in one category force the straddle whatever the
        # shuffle does: the category reserves round(5 * 0.2) = 1 of them, so an
        # index-only split always leaves the other four in the pool.
        texts, labels = corpus()
        texts += [SHARED_STEM] * 5
        labels += ["law"] * 5

        pool, heldout = reserve_heldout(texts, labels)
        pool_questions = {texts[i] for i in pool}
        heldout_questions = {texts[i] for i in heldout}

        self.assertIn(SHARED_STEM, heldout_questions)
        self.assertNotIn(SHARED_STEM, pool_questions)
        self.assertFalse(pool_questions & heldout_questions)

    def test_every_repeated_question_lands_wholly_on_one_side(self):
        texts, labels = corpus()
        for n, category in enumerate(CATEGORIES, start=2):
            texts += [f"repeated {category} stem"] * n
            labels += [category] * n

        pool, heldout = reserve_heldout(texts, labels)
        pool_rows = set(pool)
        heldout_rows = set(heldout)

        sides = {}
        for index, text in enumerate(texts):
            if index in heldout_rows:
                sides.setdefault(text, set()).add("heldout")
            elif index in pool_rows:
                sides.setdefault(text, set()).add("pool")
        straddling = {t for t, s in sides.items() if len(s) > 1}

        self.assertEqual(straddling, set())

    def test_a_repeated_question_costs_the_pool_not_the_heldout_slice(self):
        texts, labels = corpus()
        texts += [SHARED_STEM] * 5
        labels += ["law"] * 5

        pool, heldout = reserve_heldout(texts, labels)
        copies = {i for i, t in enumerate(texts) if t == SHARED_STEM}

        # One copy is scored; the rest are left out of the pool, not scored again.
        self.assertEqual(len(copies & set(heldout)), 1)
        self.assertEqual(copies & set(pool), set())

    def test_every_category_keeps_its_share(self):
        texts, labels = corpus(rows_per_category=100)
        _, heldout = reserve_heldout(texts, labels)

        held_by_category = Counter(labels[i] for i in heldout)
        self.assertEqual(set(held_by_category), set(CATEGORIES))
        for category in CATEGORIES:
            self.assertEqual(held_by_category[category], 100 * HELDOUT_FRACTION)

    def test_uneven_categories_each_keep_a_share(self):
        texts, labels = [], []
        for size, category in zip([500, 60, 11], CATEGORIES, strict=False):
            for i in range(size):
                texts.append(f"{category} question {i}")
                labels.append(category)

        _, heldout = reserve_heldout(texts, labels)
        held_by_category = Counter(labels[i] for i in heldout)
        self.assertEqual(held_by_category["biology"], 100)
        self.assertEqual(held_by_category["law"], 12)
        self.assertEqual(held_by_category["math"], 2)

    def test_the_same_seed_reserves_the_same_rows(self):
        texts, labels = corpus()

        self.assertEqual(reserve_heldout(texts, labels), reserve_heldout(texts, labels))

    def test_a_different_seed_reserves_different_rows(self):
        texts, labels = corpus()
        _, default = reserve_heldout(texts, labels)
        _, other = reserve_heldout(texts, labels, seed=SPLIT_SEED + 1)

        self.assertNotEqual(default, other)
        self.assertEqual(len(default), len(other))

    def test_reserved_rows_are_not_simply_the_first_rows(self):
        # The trainer's category sampler head-slices, so a reservation that also
        # took a contiguous prefix would correlate with what gets sampled.
        texts, labels = corpus()
        _, heldout = reserve_heldout(texts, labels)

        self.assertNotEqual(heldout, list(range(len(heldout))))

    def test_indices_are_plain_ints_so_the_manifest_serialises(self):
        texts, labels = corpus()
        _, heldout = reserve_heldout(texts, labels)

        self.assertTrue(all(type(i) is int for i in heldout))
        self.assertEqual(json.loads(json.dumps(heldout)), heldout)

    def test_mismatched_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            reserve_heldout(["a", "b"], ["only-one-label"])

    def test_a_fraction_outside_the_unit_interval_is_rejected(self):
        texts, labels = corpus(rows_per_category=10)
        for fraction in (0.0, 1.0, -0.5, 2.0):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                reserve_heldout(texts, labels, fraction=fraction)


class DropReservedQuestionsTest(unittest.TestCase):
    def test_supplement_repeats_of_a_reserved_question_are_dropped(self):
        supplement = [("hello there", "other"), ("law question 3", "law")]

        kept = drop_reserved_questions(supplement, ["law question 3"])

        self.assertEqual(kept, [("hello there", "other")])

    def test_unrelated_supplement_rows_survive(self):
        supplement = [("hello there", "other"), ("how are you", "other")]

        self.assertEqual(
            drop_reserved_questions(supplement, ["law question 3"]), supplement
        )

    def test_an_empty_supplement_is_handled(self):
        self.assertEqual(drop_reserved_questions([], ["anything"]), [])


class BuildManifestTest(unittest.TestCase):
    def test_the_manifest_records_what_reproduces_the_split(self):
        manifest = build_manifest(
            [3, 1, 2],
            {"eval_accuracy": 0.9, "eval_f1": 0.8},
            dataset="TIGER-Lab/MMLU-Pro",
        )

        self.assertEqual(manifest["dataset"], "TIGER-Lab/MMLU-Pro")
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(manifest["heldout_fraction"], HELDOUT_FRACTION)
        self.assertEqual(manifest["seed"], SPLIT_SEED)
        self.assertEqual(manifest["num_heldout_rows"], 3)
        self.assertEqual(manifest["heldout_row_indices"], [3, 1, 2])
        self.assertEqual(manifest["accuracy"], 0.9)
        self.assertEqual(manifest["f1"], 0.8)

    def test_the_manifest_round_trips_through_json(self):
        texts, labels = corpus()
        _, heldout = reserve_heldout(texts, labels)
        manifest = build_manifest(
            heldout, {"eval_accuracy": 0.5, "eval_f1": 0.5}, dataset="x/y"
        )

        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertTrue(MANIFEST_NAME.endswith(".json"))

    def test_the_recorded_count_matches_the_recorded_indices(self):
        texts, labels = corpus()
        _, heldout = reserve_heldout(texts, labels)
        manifest = build_manifest(
            heldout, {"eval_accuracy": 0.5, "eval_f1": 0.5}, dataset="x/y"
        )

        self.assertEqual(
            manifest["num_heldout_rows"], len(manifest["heldout_row_indices"])
        )
        self.assertEqual(
            len(set(manifest["heldout_row_indices"])), manifest["num_heldout_rows"]
        )


if __name__ == "__main__":
    unittest.main()
