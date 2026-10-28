import copy
import importlib.util
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.training.model_classifier.sequence_repair.train import (
    global_batch_loss,
    sample_source_balanced,
    source_sampling_pools,
    token_budget_microbatches,
)


class TokenBudgetTests(unittest.TestCase):
    def test_stable_groups_preserve_duplicate_draws_and_bound_padding(self):
        lengths = [3, 12, 32, 7, 3]
        sampled = [2, 4, 0, 1, 2, 3, 4, 1]
        groups = token_budget_microbatches(sampled, lengths, 32)
        flat = [index for group in groups for index in group]
        self.assertEqual(Counter(flat), Counter(sampled))
        self.assertEqual(flat, sorted(sampled, key=lengths.__getitem__))
        longest_index = 2
        self.assertEqual(
            [group for group in groups if longest_index in group],
            [[longest_index], [longest_index]],
        )
        for group in groups:
            self.assertLessEqual(len(group) * max(lengths[i] for i in group), 32)
        with self.assertRaises(ValueError):
            token_budget_microbatches(sampled, lengths, 31)
        with self.assertRaises(ValueError):
            token_budget_microbatches(sampled, lengths, 0)

    def test_prefetch_preserves_sampling_and_rng_across_optimizer_steps(self):
        rows = [
            {"source": source, "length_bucket": length, "label": label}
            for source in ["natural", "authored"]
            for length in [8, 32]
            for label in ["a", "b"]
        ]
        pools = source_sampling_pools(rows, True)
        sequential, prefetched = random.Random(17), random.Random(17)
        for _ in range(30):
            original = []
            for _ in range(2):
                original.extend(
                    sample_source_balanced(pools, sequential, True) for _ in range(4)
                )
            sampled = [
                sample_source_balanced(pools, prefetched, True) for _ in range(8)
            ]
            self.assertEqual(original, sampled)
            groups = token_budget_microbatches(
                sampled, [r["length_bucket"] for r in rows], 32
            )
            self.assertEqual(
                Counter(original), Counter(i for group in groups for i in group)
            )
        self.assertEqual(sequential.getstate(), prefetched.getstate())

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "Optional actual ModernBERT gradient check requires Torch and Transformers",
    )
    def test_fp32_modernbert_gradient_matches_full_batch_with_dropout_disabled(self):
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415 - optional dependency
            ModernBertConfig,
            ModernBertForSequenceClassification,
        )

        torch.set_num_threads(1)
        torch.manual_seed(11)
        config = ModernBertConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=8,
            global_attn_every_n_layers=2,
            num_labels=3,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            classifier_dropout=0.0,
            classifier_pooling="mean",
            reference_compile=False,
        )
        config._attn_implementation = "sdpa"
        full = ModernBertForSequenceClassification(config).float().train()
        grouped = copy.deepcopy(full)
        lengths = [3, 12, 32, 7, 5, 19, 8, 4]
        examples = [torch.randint(3, 64, (length,)) for length in lengths]
        labels = torch.arange(8) % 3

        def accumulate(model, indices):
            ids = torch.zeros(
                len(indices), max(lengths[i] for i in indices), dtype=torch.long
            )
            mask = torch.zeros_like(ids)
            for j, index in enumerate(indices):
                ids[j, : lengths[index]] = examples[index]
                mask[j, : lengths[index]] = 1
            logits = model(input_ids=ids, attention_mask=mask).logits
            global_batch_loss(logits, labels[indices], 8).backward()

        accumulate(full, list(range(8)))
        groups = token_budget_microbatches(list(range(8)), lengths, 32)
        self.assertGreater(len(groups), 2)
        for indices in groups:
            accumulate(grouped, indices)
        for (name, expected), (_, actual) in zip(
            full.named_parameters(), grouped.named_parameters(), strict=True
        ):
            self.assertIsNotNone(expected.grad, name)
            self.assertIsNotNone(actual.grad, name)
            torch.testing.assert_close(
                expected.grad, actual.grad, rtol=2e-5, atol=2e-6, msg=name
            )


if __name__ == "__main__":
    unittest.main()
