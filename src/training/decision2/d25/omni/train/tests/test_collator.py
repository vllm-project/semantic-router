from __future__ import annotations

import random
import unittest

import torch

from d25.omni.model import checkpoint, inputs, tiny
from d25.omni.train.collator import MultimodalCollator, shuffle_options
from d25.omni.train.rows import measure, read_rows


class CollatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        paths = tiny.fixtures()
        init = paths["init_vega"]
        cls.decision = checkpoint.read_decision_config(init)
        cls.processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(init)))
        cls.collator = MultimodalCollator(cls.processor, cls.decision["codes"])
        cls.rows, dropped = read_rows(
            [paths["data"] / "mm.jsonl", paths["data"] / "text.jsonl"], "main"
        )
        assert not dropped
        for row, (tokens, reason) in zip(
            cls.rows,
            measure(cls.rows, cls.processor, cls.decision["codes"], "d25-vega", 16_384),
        ):
            assert reason is None
            row.tokens = tokens

    def test_mixed_text_and_image_batch(self) -> None:
        image_rows = [row for row in self.rows if row.images][:3]
        text_rows = [row for row in self.rows if not row.images][:2]
        rows = [text_rows[0], image_rows[0], image_rows[1], text_rows[1], image_rows[2]]
        batch = self.collator(rows)
        n_images = sum(len(row.images) for row in rows)
        self.assertEqual(batch.images, n_images)
        self.assertEqual(batch.inputs["image_grid_thw"].shape, (n_images, 3))
        patches = int(batch.inputs["image_grid_thw"].prod(-1).sum())
        self.assertEqual(batch.inputs["pixel_values"].shape[0], patches)
        ids, mask = batch.inputs["input_ids"], batch.inputs["attention_mask"]
        self.assertEqual(ids.shape[1], max(row.tokens for row in rows))
        self.assertEqual(mask.sum(1).tolist(), [row.tokens for row in rows])
        self.assertTrue(bool((mask[:, -1] == 1).all()))
        image_token = self.processor.tokenizer.convert_tokens_to_ids(inputs.IMAGE_TOKEN)
        visual = int((ids == image_token).sum())
        self.assertEqual(
            visual, patches // self.processor.image_processor.merge_size**2
        )
        self.assertEqual(batch.inputs["mm_token_type_ids"].sum().item(), visual)
        self.assertEqual(batch.targets.shape, (5, 255))
        for index, row in enumerate(rows):
            self.assertTrue(
                torch.allclose(
                    batch.targets[index, : row.n_options], torch.tensor(row.target)
                )
            )
            self.assertEqual(
                float(batch.targets[index, row.n_options :].abs().sum()), 0.0
            )
        self.assertEqual(batch.counts.tolist(), [row.n_options for row in rows])
        self.assertEqual(batch.weights.tolist(), [1.0] * 5)
        self.assertEqual(batch.tokens, sum(row.tokens for row in rows))

    def test_text_only_batch_has_no_pixels(self) -> None:
        batch = self.collator([row for row in self.rows if not row.images][:3])
        self.assertNotIn("pixel_values", batch.inputs)
        self.assertEqual(batch.images, 0)

    def test_dummy_batch_has_zero_weight(self) -> None:
        batch = self.collator(self.rows[:1], dummy=True)
        self.assertEqual(batch.weights.tolist(), [0.0])

    def test_shuffle_options_permutes_targets_with_keys(self) -> None:
        row = next(
            row
            for row in self.rows
            if row.question["type"] == "choice" and row.n_options >= 4
        )
        shuffled = shuffle_options(row, random.Random(4))
        before = dict(zip(row.question["criteria"], row.target))
        after = dict(zip(shuffled.question["criteria"], shuffled.target))
        self.assertEqual(before, after)
        self.assertEqual(row.question["criteria"], dict(row.question["criteria"]))
        orders = {
            tuple(shuffle_options(row, random.Random(seed)).question["criteria"])
            for seed in range(20)
        }
        self.assertGreater(len(orders), 1)
        noul = next(row for row in self.rows if row.question["type"] == "noul")
        self.assertIs(shuffle_options(noul, random.Random(0)), noul)

    def test_measure_drops_long_and_conflicting_rows(self) -> None:
        row = self.rows[0]
        long_row = type(row)(
            **{**row.__dict__, "id": "long", "state": "word " * 4000, "images": []}
        )
        conflict = type(row)(
            **{
                **row.__dict__,
                "id": "conflict",
                "state": "<|image_pad|>",
                "images": [r for r in self.rows if r.images][0].images[:1],
            }
        )
        results = measure(
            [long_row, conflict],
            self.processor,
            self.decision["codes"],
            "d25-vega",
            4096,
        )
        self.assertIn("over max_length", results[0][1])
        self.assertIn("placeholder", results[1][1])


if __name__ == "__main__":
    unittest.main()
