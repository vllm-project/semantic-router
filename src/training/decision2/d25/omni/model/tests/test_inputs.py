from __future__ import annotations

import os
import unittest

from PIL import Image

from d25.omni.common import vision_format
from d25.omni.model import inputs, tiny
from d25.vega.common import decision_format as text_format

QWEN38_DIR = os.environ.get("D25_QWEN38_DIR")


class InputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.processor = inputs.setup_processor(
            AutoProcessor.from_pretrained(str(tiny.fixtures()["init_stock"]))
        )
        cls.codes, _ = text_format.answer_codes(cls.processor.tokenizer)
        cls.question = {
            "type": "choice",
            "instructions": "Which is shown ?",
            "criteria": {"a": "cat", "b": None},
        }

    def test_cost_matches_processor_output(self) -> None:
        layouts = [
            [],
            [(2000, 1500)],
            [(300, 200), (64, 64)],
            [(1280, 1280), (3000, 400), (300, 200), (90, 120)],
        ]
        for sizes in layouts:
            images = [Image.new("RGB", size) for size in sizes]
            for prompt in inputs.PROMPTS:
                with self.subTest(images=len(images), prompt=prompt):
                    text = inputs.render(
                        self.processor,
                        prompt,
                        "state words",
                        self.question,
                        self.codes,
                        len(images),
                    )
                    cost = inputs.cost(
                        self.processor, text, [image.size for image in images]
                    )
                    encoded = inputs.encode(self.processor, [text], images)
                    self.assertEqual(encoded["input_ids"].shape[1], cost.tokens)
                    self.assertEqual(text.count(inputs.IMAGE_TOKEN), len(images))

    def test_board_pixel_cap(self) -> None:
        image_processor = self.processor.image_processor
        self.assertLessEqual(
            inputs.visual_tokens(image_processor, 4000, 3000), 1_638_400 // 1024
        )
        self.assertGreater(inputs.visual_tokens(image_processor, 4000, 3000), 1_500)
        self.assertEqual(inputs.visual_tokens(image_processor, 10, 10), 65_536 // 1024)
        with self.assertRaises(ValueError):
            inputs.visual_tokens(image_processor, 20_000, 50)

    def test_vega_text_rendering_is_unchanged(self) -> None:
        text = inputs.render(
            self.processor, "d25-vega", "s", self.question, self.codes, 0
        )
        self.assertEqual(
            text,
            text_format.render(
                self.processor.tokenizer, "s", self.question, self.codes
            ),
        )
        with_images = inputs.render(
            self.processor, "d25-vega", "s", self.question, self.codes, 2
        )
        self.assertTrue(
            with_images.split("<|im_start|>user\n", 1)[1].startswith(
                "<|vision_start|><|image_pad|><|vision_end|>" * 2 + "State:"
            )
        )

    def test_pplx_prompt(self) -> None:
        text = inputs.render(
            self.processor, "pplx", {"k": 1}, self.question, self.codes, 1
        )
        self.assertIn(inputs.PPLX_SYSTEM_PROMPT, text)
        self.assertIn(
            'State:\n{"k": 1}\n\nQuestion:\nWhich is shown ?\n\nOptions:\nA: a: cat\nB: b',
            text,
        )
        self.assertIn("Return only the letter code of the best option.", text)
        self.assertLess(text.index("<|image_pad|>"), text.index("State:"))

    def test_placeholder_conflict(self) -> None:
        self.assertFalse(inputs.placeholder_conflict("<|image_pad|> text", 1))
        self.assertTrue(inputs.placeholder_conflict("<|image_pad|><|image_pad|>", 1))
        self.assertTrue(inputs.placeholder_conflict("<|image_pad|> <|video_pad|>", 1))

    def test_check_codes(self) -> None:
        codes, ids = text_format.answer_codes(self.processor.tokenizer)
        inputs.check_codes(self.processor.tokenizer, codes, ids)
        with self.assertRaises(ValueError):
            inputs.check_codes(self.processor.tokenizer, codes[::-1], ids[::-1])

    def test_vision_format_limits(self) -> None:
        self.assertEqual(vision_format.MAX_IMAGES, 4)
        self.assertEqual(
            self.processor.image_processor.size["longest_edge"],
            vision_format.MAX_PIXELS,
        )


@unittest.skipUnless(
    QWEN38_DIR, "set D25_QWEN38_DIR to a Qwen3.8-27B tokenizer/processor directory"
)
class RealProcessorTest(unittest.TestCase):
    """Against the real Qwen3.8-27B tokenizer and processor files (no weights needed)."""

    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.processor = inputs.setup_processor(
            AutoProcessor.from_pretrained(QWEN38_DIR)
        )
        cls.codes, cls.ids = text_format.answer_codes(cls.processor.tokenizer)

    def test_answer_codes_match_the_released_decider(self) -> None:
        self.assertEqual(
            self.codes[:30], [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "AA", "AB", "AC", "AD"]
        )
        self.assertEqual(self.ids[:3], [32, 33, 34])
        self.assertEqual(self.codes[-3:], ["JR", "JS", "JT"])

    def test_cost_and_cap(self) -> None:
        question = {
            "type": "choice",
            "instructions": "Pick",
            "criteria": {"x": "first", "y": None},
        }
        images = [Image.new("RGB", (2000, 1500)), Image.new("RGB", (300, 200))]
        text = inputs.render(
            self.processor, "d25-vega", "hello", question, self.codes, len(images)
        )
        cost = inputs.cost(self.processor, text, [image.size for image in images])
        self.assertEqual(cost.visual_tokens, (1564, 70))
        encoded = inputs.encode(self.processor, [text], images)
        self.assertEqual(encoded["input_ids"].shape[1], cost.tokens)
        self.assertTrue(
            text.split("<|im_start|>user\n", 1)[1].startswith(
                "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>State:"
            )
        )


if __name__ == "__main__":
    unittest.main()
