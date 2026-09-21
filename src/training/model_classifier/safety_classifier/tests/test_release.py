import ast
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.config import load_contract
from src.training.model_classifier.safety_classifier.release import build_model_card


class ModelCardTest(unittest.TestCase):
    def test_card_uses_product_header_and_keeps_the_actual_training_base(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text(
                '{"test":{"test_accuracy":0.75,"test_runtime":1.0,'
                '"test_false_positives":20}}',
                encoding="utf-8",
            )
            (run_root / "training_manifest.json").write_text(
                '{"source_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                '"global_train_batch_size":64,'
                '"precision":"bf16"}',
                encoding="utf-8",
            )
            contract = load_contract()
            contract["base_model"] = {
                "id": "example/actual-training-base",
                "revision": "b" * 40,
            }
            card = build_model_card(
                "level1",
                "adapter",
                "example/model-lora",
                run_root,
                contract,
            )
            body = card.split("---", 2)[2].lstrip()
            self.assertTrue(body.startswith('<div align="center">'))
            self.assertIn('src="https://vllm-sr.ai/img/vllm-sr-logo.social.png"', body)
            self.assertEqual(
                re.findall(r'<a href="([^"]+)">', body),
                [
                    "https://vllm-sr.ai/",
                    "https://vllm-sr.ai/blog/",
                    "https://vllm-dev.slack.com/archives/C09CTGF8KCN",
                    "https://github.com/vllm-project/semantic-router",
                ],
            )
            self.assertIn("LoRA adapter", card)
            self.assertIn("example/actual-training-base", card)
            self.assertIn("b" * 40, card)
            self.assertNotIn("llm-semantic-router/mmbert-32k-yarn", card)
            self.assertIn("| `accuracy` | 0.750000 |", card)
            self.assertNotIn("test_runtime", card)
            self.assertNotIn("false_positives", card)
            self.assertNotIn("Technical Report", card)
            self.assertNotIn("TECHNICAL.md", card)
            self.assertNotIn("Global batch", card)

    def test_card_refuses_a_mutable_or_missing_source_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text(
                '{"test":{"test_accuracy":0.75}}',
                encoding="utf-8",
            )
            (run_root / "training_manifest.json").write_text(
                '{"source_commit":null,"global_train_batch_size":64,'
                '"precision":"bf16"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "immutable source commit"):
                build_model_card(
                    "level1",
                    "adapter",
                    "example/model-lora",
                    run_root,
                    load_contract(),
                )

    def test_adapter_card_constructs_the_task_specific_classifier_head(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text(
                '{"test":{"test_f1_macro":0.75}}', encoding="utf-8"
            )
            (run_root / "training_manifest.json").write_text(
                '{"source_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                '"global_train_batch_size":64,"precision":"bf16"}',
                encoding="utf-8",
            )

            card = build_model_card(
                "level2",
                "adapter",
                "example/level2-lora",
                run_root,
                load_contract(),
            )

            self.assertIn("num_labels=len(id2label)", card)
            self.assertIn("8: 'S13_misinformation'", card)
            self.assertIn("'S13_misinformation': 8", card)
            self.assertIn("not twelve independent Hazard scores", card)

    def test_all_card_variants_include_a_complete_bounded_cpu_example(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "metrics.json").write_text('{"test":{}}')
            (run_root / "training_manifest.json").write_text(
                json.dumps({"source_commit": "a" * 40})
            )
            for task in ("level1", "level2"):
                for artifact in ("adapter", "merged"):
                    with self.subTest(task=task, artifact=artifact):
                        card = build_model_card(
                            task, artifact, "example/model", run_root, load_contract()
                        )
                        example = card.split("```python\n", 1)[1].split("```", 1)[0]
                        ast.parse(example)
                        self.assertIn(".eval()", example)
                        self.assertIn("torch.inference_mode()", example)
                        self.assertIn("truncation=True", example)
                        self.assertIn("max_length=512", example)
                        self.assertIn("model(**inputs).logits.argmax", example)
                        self.assertIn("print(model.config.id2label[label_id])", example)
                        self.assertIn("does\nnot establish longer-input accuracy", card)
                        self.assertNotIn("32,768-token accuracy", card)


if __name__ == "__main__":
    unittest.main()
