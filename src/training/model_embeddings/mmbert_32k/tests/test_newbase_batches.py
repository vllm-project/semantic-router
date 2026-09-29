"""Logical negative pools and loss gradients survive physical microbatching."""

import unittest

try:
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.newbase_batches import (
        ObjectiveConfig,
        TokenBatch,
        embedding_step,
        reranker_step,
    )
    from src.training.model_embeddings.mmbert_32k.newbase_data import text_digest
    from src.training.model_embeddings.mmbert_32k.newbase_model import (
        ExitSpec,
        NewBaseTask,
    )
except ImportError:
    torch = None


class CompleteTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, left, right=None, **options):
        if options["truncation"] or options["padding"]:
            raise AssertionError(
                "The whole-input encoder must not truncate or pad here"
            )
        values = []
        for index, text in enumerate(left):
            row = [1, *[3 + len(word) % 20 for word in text.split()], 2]
            if right is not None:
                row.extend([*[3 + len(word) % 20 for word in right[index].split()], 2])
            values.append(row)
        return {"input_ids": values}


@unittest.skipIf(torch is None, "requires torch and transformers")
class NewBaseBatchesTest(unittest.TestCase):
    def test_matching_layer_anchors_train_all_encoder_layers(self):
        class Cache:
            def lookup(self, inputs, components, batch, device):
                return torch.stack(
                    [
                        torch.stack(
                            [
                                torch.arange(1, 17).float().roll(i + layer)
                                for layer in (3, 6, 11, 22)
                            ]
                        )
                        for i in range(len(inputs))
                    ]
                ).to(device)

        _, components = self.examples()
        rows = [{"source": "natural", "component_id": key} for key in components]
        model = self.model("embedding")
        loss, _ = embedding_step(
            model,
            CompleteTokenizer(),
            rows,
            components,
            device=torch.device("cpu"),
            token_budget=8,
            maximum=128,
            objective=ObjectiveConfig("representation"),
            amp=False,
            anchor_cache=Cache(),
            anchor_weight=1.0,
            anchor_target_layers=[3, 6, 11, 22],
        )
        loss.backward()
        self.assertGreater(loss.item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for layer in model.encoder.layers:
            self.assertTrue(any(p.grad.abs().max() > 0 for p in layer.parameters()))

    def test_broad_anchors_train_every_layer_across_complete_microbatches(self):
        class Cache:
            def lookup(self, inputs, components, batch, device):
                return torch.stack(
                    [torch.arange(1, 17).float().roll(i) for i in range(len(inputs))]
                ).to(device)

        _, components = self.examples()
        rows = [{"source": "natural", "component_id": key} for key in components]
        model = self.model("embedding")
        results = []
        for budget in (1000, 8):
            model.zero_grad(set_to_none=True)
            loss, metrics = embedding_step(
                model,
                CompleteTokenizer(),
                rows,
                components,
                device=torch.device("cpu"),
                token_budget=budget,
                maximum=128,
                objective=ObjectiveConfig("representation"),
                amp=False,
                anchor_cache=Cache(),
                anchor_weight=1.0,
            )
            loss.backward()
            gradients = {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
            }
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients.values()))
            for layer in model.encoder.layers:
                self.assertTrue(any(p.grad.abs().max() > 0 for p in layer.parameters()))
            results.append((loss.detach(), gradients, metrics["token_sha256"]))
        torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-5)
        self.assertEqual(results[0][2], results[1][2])
        for name in results[0][1]:
            torch.testing.assert_close(
                results[0][1][name], results[1][1][name], atol=2e-5, rtol=2e-4
            )
        with self.assertRaisesRegex(ValueError, "explicit teacher"):
            embedding_step(
                model,
                CompleteTokenizer(),
                rows,
                components,
                device=torch.device("cpu"),
                token_budget=128,
                maximum=128,
                objective=ObjectiveConfig("representation"),
                amp=False,
            )

    def model(self, task):
        config = ModernBertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=22,
            num_attention_heads=2,
            max_position_embeddings=128,
            local_attention=4,
            reference_compile=False,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "sdpa"
        return NewBaseTask(
            ModernBertModel(config), task, ExitSpec(dimensions=(16, 12, 8, 4, 2)), {}
        ).eval()

    def examples(self):
        texts = {
            "q1": "one question",
            "q2": "second longer query",
            "p1": "first correct passage",
            "p2": "another complete relevant paragraph",
            "n": "a different document",
        }
        components = {
            key: {
                "text": text,
                "normalized_sha256": text_digest(text, normalize=True),
                "parent_groups": [key],
            }
            for key, text in texts.items()
        }
        rows = [
            {
                "source": "miracl",
                "query_component_id": f"q{i}",
                "positive_component_ids": [f"p{i}"],
                "judged_negative_component_ids": ["n"],
                "unjudged_component_ids": [],
                "candidate_component_ids": [f"p{i}", "n"],
            }
            for i in (1, 2)
        ]
        return rows, components

    def test_ranking_filter_preserves_full_candidates_and_judged_loss(self):
        rows, components = self.examples()
        model = self.model("reranker")

        def evaluate(records):
            return reranker_step(
                model,
                CompleteTokenizer(),
                records,
                components,
                device=torch.device("cpu"),
                token_budget=128,
                maximum=128,
                objective=ObjectiveConfig("ranking"),
                amp=False,
            )

        base, _ = evaluate(rows)
        expanded = []
        for index, row in enumerate(rows):
            unknown = "p2" if index == 0 else "p1"
            expanded.append(
                {
                    **row,
                    "candidate_component_ids": [
                        *row["candidate_component_ids"],
                        unknown,
                    ],
                    "unjudged_component_ids": [unknown],
                    "unjudged_preference_component_ids": [],
                }
            )
        filtered, metrics = evaluate(expanded)
        torch.testing.assert_close(filtered, base, atol=1e-6, rtol=1e-6)
        self.assertEqual(metrics["input_count"], 6)
        for row in expanded:
            del row["unjudged_preference_component_ids"]
        legacy, _ = evaluate(expanded)
        self.assertGreater(legacy.item(), filtered.item())
        expanded[0]["unjudged_preference_component_ids"] = ["n"]
        with self.assertRaisesRegex(ValueError, "unjudged"):
            evaluate(expanded)

    def test_loss_and_gradients_match_across_microbatch_budgets(self):
        rows, components = self.examples()
        for task, objective, function in (
            (
                "embedding",
                ObjectiveConfig("retrieval", distillation_weight=0.1),
                embedding_step,
            ),
            ("reranker", ObjectiveConfig("ranking", lambda_weight=0.2), reranker_step),
        ):
            with self.subTest(task=task):
                model = self.model(task)
                losses, gradients, receipts = [], [], []
                for budget in (1000, 12):
                    model.zero_grad(set_to_none=True)
                    loss, receipt = function(
                        model,
                        CompleteTokenizer(),
                        rows,
                        components,
                        device=torch.device("cpu"),
                        token_budget=budget,
                        maximum=128,
                        objective=objective,
                        amp=False,
                    )
                    loss.backward()
                    losses.append(loss.detach())
                    gradients.append(
                        {
                            name: value.grad.clone()
                            for name, value in model.named_parameters()
                        }
                    )
                    receipts.append(receipt)
                torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-5)
                self.assertEqual(
                    receipts[0]["token_sha256"], receipts[1]["token_sha256"]
                )
                for name in gradients[0]:
                    torch.testing.assert_close(
                        gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-4
                    )

    def test_long_complete_input_rejects_overflow_before_forward(self):
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            TokenBatch.encode(CompleteTokenizer(), ["one two three"], None, 4)
        batch = TokenBatch.encode(CompleteTokenizer(), ["one two three"], None, 8)
        with self.assertRaisesRegex(ValueError, "microbatch"):
            list(batch.partitions(4))

    def test_unpartitioned_candidate_cannot_become_implicit_negative(self):
        rows, components = self.examples()
        rows[0]["candidate_component_ids"].append("p2")
        for function, kind in (
            (embedding_step, "retrieval"),
            (reranker_step, "ranking"),
        ):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "explicit relevance membership"),
            ):
                function(
                    None,
                    CompleteTokenizer(),
                    rows,
                    components,
                    device=torch.device("cpu"),
                    token_budget=128,
                    maximum=128,
                    objective=ObjectiveConfig(kind),
                    amp=False,
                )


if __name__ == "__main__":
    unittest.main()
