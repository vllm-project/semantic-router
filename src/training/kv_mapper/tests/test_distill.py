from __future__ import annotations

import itertools
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Torch and Transformers are optional in the contract test environment.
# ruff: noqa: PLC0415

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import (  # noqa: E402
    CompatibilitySpec,
    Manifest,
    read_artifact,
    tensor_keys_for_layers,
    write_artifact,
)

HEADS, HEAD_DIM, LAYERS, VOCAB = 2, 16, 2, 100
WIDTH = HEADS * HEAD_DIM


def _manifest(mapper_id: str = "tiny-full_head-fp32-tp1-h2-sa-tb-b1") -> Manifest:
    compat = CompatibilitySpec(
        source_model="tiny/source",
        source_revision="a" * 40,
        target_model="tiny/target",
        target_revision="b" * 40,
        variant="full_head",
        precision="fp32",
        source_tp=1,
        target_tp=1,
        head_order="hf",
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
    )
    layers = {str(i): [i] for i in range(LAYERS)}
    return Manifest(
        mapper_id=mapper_id,
        compatibility=compat,
        topk=1,
        ridge_alpha=1.0,
        centered_inputs=True,
        rope_stripped_on_keys=True,
        source_layers_per_target={"k": layers, "v": dict(layers)},
    )


def _identity_tensors() -> dict[str, np.ndarray]:
    tensors = {}
    for name in tensor_keys_for_layers(LAYERS):
        if name.endswith(".W"):
            tensors[name] = np.eye(WIDTH, dtype=np.float32)
        else:
            tensors[name] = np.zeros(WIDTH, dtype=np.float32)
    return tensors


class DistillTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("torch or transformers not installed") from exc
        self.torch = torch

        def model(seed: int):
            torch.manual_seed(seed)
            # A wide init gives peaked predictions, so the cache visibly matters.
            config = Qwen3Config(
                initializer_range=0.2,
                vocab_size=VOCAB,
                hidden_size=64,
                num_hidden_layers=LAYERS,
                num_attention_heads=4,
                num_key_value_heads=HEADS,
                head_dim=HEAD_DIM,
                intermediate_size=128,
            )
            return Qwen3ForCausalLM(config).eval().requires_grad_(False)

        self.model = model

    def _samples(self, count: int, seed: int):
        rng = random.Random(seed)
        return [
            (
                [rng.randrange(VOCAB) for _ in range(rng.randrange(4, 12))],
                [rng.randrange(VOCAB) for _ in range(rng.randrange(1, 6))],
            )
            for _ in range(count)
        ]

    def test_identity_maps_reproduce_the_native_cache(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper, distillation_loss

        model = self.model(0)
        mapper = LinearMapper(_manifest(), _identity_tensors())
        for prefix, continuation in self._samples(4, 1):
            loss = distillation_loss(model, model, mapper, prefix, continuation)
            self.assertLess(float(loss.detach()), 1e-5)

    def test_gradient_reaches_only_the_maps(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper, distillation_loss

        source, target = self.model(0), self.model(1)
        mapper = LinearMapper(_manifest(), _identity_tensors())
        prefix, continuation = self._samples(1, 2)[0]
        loss = distillation_loss(source, target, mapper, prefix, continuation)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        grads = [p.grad for p in mapper.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_training_lowers_kl_on_its_samples(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper, mean_kl, train

        model = self.model(0)
        rng = np.random.default_rng(0)
        tensors = {
            name: value + rng.normal(0.0, 0.3, value.shape).astype(np.float32)
            for name, value in _identity_tensors().items()
        }
        mapper = LinearMapper(_manifest(), tensors)
        samples = self._samples(4, 3)
        before = mean_kl(model, model, mapper, samples)

        def stream():
            while True:
                yield from samples

        train(model, model, mapper, stream(), steps=60, batch=4, lr=1e-2, log_every=0)
        self.assertLess(mean_kl(model, model, mapper, samples), before * 0.5)

    def test_export_keeps_the_artifact_layout(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper

        mapper = LinearMapper(_manifest(), _identity_tensors())
        exported = mapper.export()
        self.assertEqual(set(exported), set(_identity_tensors()))
        with tempfile.TemporaryDirectory() as tmp:
            write_artifact(Path(tmp), _manifest(), exported)
            _manifest_read, tensors = read_artifact(Path(tmp))
        for name, value in tensors.items():
            np.testing.assert_array_equal(value, _identity_tensors()[name])

    def test_low_rank_mapper_starts_at_the_base_and_trains_the_correction(
        self,
    ) -> None:
        from src.training.kv_mapper.distill import LinearMapper, distillation_loss

        source, target = self.model(0), self.model(1)
        mapper = LinearMapper(_manifest(), _identity_tensors(), rank=4)
        for name, value in mapper.export().items():
            np.testing.assert_array_equal(value, _identity_tensors()[name])
        prefix, continuation = self._samples(1, 2)[0]
        distillation_loss(source, target, mapper, prefix, continuation).backward()
        trainable = dict(mapper.named_parameters())
        self.assertFalse(any(name.endswith("_W") for name in trainable))
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for n, p in trainable.items()
                if n.endswith("_B")
            ),
            0.0,
        )

    def test_relative_rates_follow_each_map_scale(self) -> None:
        from src.training.kv_mapper.distill import LinearMapper, relative_rates, train

        tensors = {
            name: value * (0.02 if ".v." in name else 1.0) + 0.5 * name.endswith(".b")
            for name, value in _identity_tensors().items()
        }
        rms = {
            name.replace(".", "_"): float(np.sqrt(np.mean(np.square(value))))
            for name, value in tensors.items()
        }
        mapper = LinearMapper(_manifest(), tensors)
        for group, name in zip(relative_rates(mapper), mapper.params, strict=True):
            self.assertAlmostEqual(group["scale"], rms[name], places=5)
        low_rank = LinearMapper(_manifest(), tensors, rank=4)
        scales = dict(zip(low_rank.params, relative_rates(low_rank), strict=True))
        self.assertEqual(scales["target_0_v_W_A"]["scale"], 1.0)
        self.assertAlmostEqual(
            scales["target_0_v_W_B"]["scale"], rms["target_0_v_W"], places=5
        )
        model = self.model(0)
        before = mapper.export()
        train(
            model,
            model,
            mapper,
            iter(self._samples(1, 4)),
            steps=1,
            batch=1,
            relative_lr=True,
            log_every=0,
        )
        moved = {
            name: float(np.abs(mapper.export()[name] - before[name]).max())
            for name in ("target.0.k.W", "target.0.v.W")
        }
        self.assertGreater(moved["target.0.k.W"], 10 * moved["target.0.v.W"])

    def test_cosine_schedule_warms_up_then_decays(self) -> None:
        from src.training.kv_mapper.distill import cosine_lr

        rates = [cosine_lr(step, 100, 1.0, 0.05) for step in range(100)]
        self.assertAlmostEqual(rates[4], 1.0)
        self.assertLess(rates[0], rates[4])
        self.assertLess(rates[-1], 0.01)
        self.assertTrue(all(a >= b for a, b in itertools.pairwise(rates[4:])))


class SplitConversationTests(unittest.TestCase):
    def test_prefix_ends_at_the_generation_prompt(self) -> None:
        try:
            from src.training.kv_mapper.distill_run import split_conversation
        except ImportError as exc:
            raise unittest.SkipTest("torch, transformers or datasets missing") from exc

        class Tokenizer:
            def apply_chat_template(
                self, messages, tokenize, add_generation_prompt=False
            ):
                text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
                return text + ("<assistant>" if add_generation_prompt else "")

            def encode(self, text, add_special_tokens):
                return [ord(c) for c in text]

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        prefix, continuation = split_conversation(
            Tokenizer(), messages, random.Random(0), 1000, 1000
        )
        self.assertEqual("".join(map(chr, prefix)), "<user>hi</><assistant>")
        self.assertEqual("".join(map(chr, continuation)), "hello</>")
        self.assertIsNone(
            split_conversation(Tokenizer(), messages[:1], random.Random(0), 1000, 1000)
        )


if __name__ == "__main__":
    unittest.main()
