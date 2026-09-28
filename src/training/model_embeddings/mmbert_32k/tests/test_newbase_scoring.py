"""Complete native DEV scoring preserves query support and artifact identity."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    from src.training.model_embeddings.mmbert_32k.newbase_data import FrozenCorpus
    from src.training.model_embeddings.mmbert_32k.newbase_model import NewBaseTask
    from src.training.model_embeddings.mmbert_32k.newbase_scoring import (
        exact_fixture_batches,
        score_corpus,
        tokenizer_semantics_digest,
    )
    from src.training.model_embeddings.mmbert_32k.tests import (
        test_newbase_batches as fixtures,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class NewBaseScoringTest(unittest.TestCase):
    def tokenizer(self):
        backend = Tokenizer(
            models.WordLevel(
                {
                    "[PAD]": 0,
                    "[CLS]": 1,
                    "[SEP]": 2,
                    "[UNK]": 3,
                    "one": 4,
                    "question": 5,
                },
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
        )

    def corpus(self):
        rows, components = fixtures.NewBaseBatchesTest().examples()
        for index, row in enumerate(rows):
            row.update(id=f"query-{index}", language="en")
        return FrozenCorpus(components, tuple(rows), "a" * 64, {"split": "known_dev"})

    def test_real_twenty_exit_scoring_round_trip(self):
        for task in ("embedding", "reranker"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                model = fixtures.NewBaseBatchesTest().model(task)
                corpus = self.corpus()

                def evaluate(current, corpus=corpus):
                    return score_corpus(
                        current,
                        fixtures.CompleteTokenizer(),
                        corpus,
                        device=torch.device("cpu"),
                        token_budget=1000,
                    )

                report, numeric = evaluate(model)
                self.assertEqual(report["precision"], "float32")
                self.assertEqual(report["evaluated_records"], 2)
                self.assertEqual(len(report["exits"]), 20)
                self.assertEqual(len(numeric), 40)
                for support in report["support"].values():
                    self.assertEqual(support["miracl"], 2)
                checkpoint = Path(directory) / "checkpoint"
                model.save(checkpoint)
                restored = NewBaseTask.resume(checkpoint)
                self.assertEqual(evaluate(restored), (report, numeric))

    def test_first_stage_miss_remains_zero_with_original_positive_gold(self):
        corpus = self.corpus()
        row = corpus.records[0]
        row["candidate_component_ids"] = ["n"]
        model = fixtures.NewBaseBatchesTest().model("reranker")
        options = {
            "device": torch.device("cpu"),
            "token_budget": 1000,
        }
        with self.assertRaisesRegex(ValueError, "omitted a known relevant"):
            score_corpus(model, fixtures.CompleteTokenizer(), corpus, **options)
        row["evaluation_scope"] = "first_stage_top100"
        _, numeric = score_corpus(
            model, fixtures.CompleteTokenizer(), corpus, **options
        )
        misses = [item for item in numeric if item["id"] == row["id"]]
        self.assertEqual(len(misses), 20)
        self.assertTrue(all(item["metric"] == 0 for item in misses))
        self.assertEqual(row["positive_component_ids"], ["p1"])

    def test_exact_fixture_never_decodes_and_rejects_semantic_or_token_mutation(self):
        corpus = self.corpus()
        row = corpus.records[0]
        tokenizer = self.tokenizer()
        model = fixtures.NewBaseBatchesTest().model("embedding")
        row.update(
            evaluation_kind="controlled_pair", length_bucket=4096, position="end"
        )
        content = {
            "task": "embedding",
            "tokenizer_semantics_sha256": tokenizer_semantics_digest(tokenizer),
            "query": tokenizer(corpus.components["q1"]["text"])["input_ids"],
            "candidates": {"p1": [1, 4, 4, 2], "n": [1, 5, 2]},
        }

        def seal():
            row["exact_token_inputs"] = {
                **content,
                "sha256": hashlib.sha256(
                    json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }

        seal()
        query, documents = exact_fixture_batches(
            row, tokenizer, model, corpus.components
        )
        self.assertEqual(query.rows[0], tuple(content["query"]))
        self.assertEqual(documents.rows, ((1, 4, 4, 2), (1, 5, 2)))
        tokenizer.backend_tokenizer.enable_padding(
            length=8, pad_id=0, pad_token="[PAD]"
        )
        self.assertEqual(
            content["tokenizer_semantics_sha256"], tokenizer_semantics_digest(tokenizer)
        )
        content["candidates"]["p1"][1] = 900
        with self.assertRaisesRegex(ValueError, "identity/task"):
            exact_fixture_batches(row, tokenizer, model, corpus.components)
        seal()
        with self.assertRaisesRegex(ValueError, "invalid token"):
            exact_fixture_batches(row, tokenizer, model, corpus.components)
        content["candidates"]["p1"][1] = 4
        content["tokenizer_semantics_sha256"] = "0" * 64
        seal()
        with self.assertRaisesRegex(ValueError, "identity/task"):
            exact_fixture_batches(row, tokenizer, model, corpus.components)


if __name__ == "__main__":
    unittest.main()
