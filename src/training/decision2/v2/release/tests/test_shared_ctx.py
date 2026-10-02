"""Shared-context prefill switch of the release runtime (``runtime/shared_ctx.py``).

On CPU every tensor is FP32, so a request answered with the switch on must
equal the exact path up to float rounding (the shared forwards only reorder
sums): that checks positions, masks, the gated-delta state and convolution
hand-off, and the suffix gathers of both modes. With the switch off the
runtime never imports the module and runs the batches it always ran. Needs
torch, Transformers (Qwen3; Qwen3.5 where installed) and tokenizers; run in
the pinned image. ``gpu_shared_ctx`` measures the BF16 GPU path on packages.
"""

from __future__ import annotations

import importlib
import inspect
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.release import build

try:
    import torch
except ImportError:
    torch = None

RUNTIME = build.SOURCE_ROOT / "v2/release/runtime"
VENDOR = (
    "calibration.py",
    "data.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "score_bias.py",
    "source.py",
)
QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
STATE = (
    "Support ticket 4821 (web chat). The customer ordered a stand mixer nine days "
    "ago and paid by card; the parcel was marked delivered but the box arrived "
    "damaged and one bowl is missing. She asks for a replacement or a refund, says "
    "this is her third contact and that she will cancel her membership otherwise. "
) * 2


def stage(scratch: Path):
    """The runtime package (``decision2``) with its vendored sources, imported."""
    root = scratch / "decision2"
    (root / "_vendor/dev2model").mkdir(parents=True)
    for name in ("__init__.py", "api.py", "qwen.py", "shared_ctx.py"):
        shutil.copyfile(RUNTIME / name, root / name)
    for name in ("_vendor/__init__.py", "_vendor/dev2model/__init__.py"):
        (root / name).write_text("", encoding="utf-8")
    for name in VENDOR:
        shutil.copyfile(
            build.SOURCE_ROOT / "training/model" / name,
            root / "_vendor/dev2model" / name,
        )
    for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
        del sys.modules[name]
    sys.path.insert(0, str(scratch))
    try:
        return importlib.import_module("decision2.qwen")
    finally:
        sys.path.remove(str(scratch))


def tokenizer():
    """A small byte-level BPE with Qwen2's pre-tokenizer, trained on the fly."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=480,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>"],
    )
    corpus = [STATE, "Task type: choice\nQuestion:\nOptions:", '{"key": "a"}'] * 20
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )


def questions(count: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    for index in range(count):
        kind = ("choice", "noul", "score")[index % 3]
        words = " ".join(
            rng.choice(["team", "refund", "late", "box", "card"])
            for _ in range(rng.randint(2, 12))
        )
        if kind == "choice":
            criteria = {
                f"k{j}": f"{words} option {j}" for j in range(rng.randint(2, 5))
            }
        elif kind == "noul":
            criteria = {"false": "No", "true": f"Yes, {words}"}
        else:
            criteria = [f"level {j} {words}" for j in range(rng.randint(2, 6))]
        out[f"q{index}"] = {
            "type": kind,
            "instructions": f"Q{index}: {words}?",
            "criteria": criteria,
        }
    return out


def probabilities(answer: dict) -> dict:
    if "noul" in answer:
        return {"true": answer["noul"]}
    return answer["probabilities"]


@unittest.skipIf(torch is None, "needs torch")
class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        stage(Path(cls.scratch.name))
        cls.sc = importlib.import_module("decision2.shared_ctx")

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def test_values(self):
        sc = self.sc
        self.assertIsNone(sc.resolve(None))
        self.assertIsNone(sc.resolve(False))
        self.assertEqual(sc.resolve(True), sc.DEFAULT_POLICY)
        self.assertEqual(sc.resolve({"tau": 0.1}).tau, 0.1)
        policy = sc.SharePolicy(mode="cache")
        self.assertIs(sc.resolve(policy), policy)
        for bad in (
            {"tau": 2},
            {"mode": "x"},
            {"fallback": "x"},
            {"nope": 1},
            {"min_questions": 1},
            "on",
        ):
            with self.assertRaises(ValueError):
                sc.resolve(bad)

    def test_shared_prefix_stops_before_the_first_option(self):
        sc = self.sc
        rows = [
            {"ids": [1, 2, 3, 4, 5, 6, 7], "candidate_positions": [5, 6]},
            {"ids": [1, 2, 3, 4, 5, 6, 8], "candidate_positions": [4, 6]},
        ]
        self.assertEqual(sc.shared_prefix(rows, 1), 4)
        self.assertEqual(sc.shared_prefix(rows, 4), 4)
        self.assertEqual(sc.shared_prefix(rows, 3), 3)
        rows[1]["ids"] = [9] + rows[1]["ids"][1:]
        self.assertEqual(sc.shared_prefix(rows, 1), 0)

    def test_buckets_split_by_length_only_when_it_pays(self):
        sc = self.sc
        lengths = [10, 100, 12, 98, 50]
        self.assertEqual(sc.buckets(lengths, 1, 0), [[0, 1, 2, 3, 4]])
        self.assertEqual(sc.buckets(lengths, 3, 0), [[0, 2], [4], [1, 3]])
        self.assertEqual(sc.buckets(lengths, 3, 10_000), [[0, 1, 2, 3, 4]])

    def test_prefix_tokenizer_equals_the_tokenizer(self):
        tok = tokenizer()
        rng = random.Random(1)
        tails = [
            "",
            " ",
            "\n",
            " \n\n",
            "é",
            "😀",
            "<|im_start|>",
            "x<|endoftext|>",
            "\t\t",
        ]
        heads = [STATE, STATE + "  ", "Context:\n" + STATE + "\n", "αβγ " + STATE]
        reused = 0
        for head in heads:
            for tail in tails:
                shared = self.sc.PrefixTokenizer(tok)
                for index in range(6):
                    text = (
                        head
                        + tail
                        + f"\n\nTask type: {rng.choice(['choice', 'noul'])} {index}"
                    )
                    self.assertEqual(
                        shared.encode(text), tok.encode(text, add_special_tokens=False)
                    )
                reused += shared.reused
        self.assertGreater(reused, 100)


def tiny(kind: str):
    from training.model.decision_model import CandidateHead, DecisionModel

    torch.manual_seed(11)
    if kind == "qwen3":
        from transformers import Qwen3Config, Qwen3Model

        config = Qwen3Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=2048,
        )
        backbone = Qwen3Model(config)
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        config = Qwen3_5TextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"] * 2,
            max_position_embeddings=2048,
        )
        backbone = Qwen3_5TextModel(config)
    backbone.config.use_cache = False
    model = DecisionModel(backbone, CandidateHead(64, 16), {"head_variant": "shared"})
    return model.float().eval()


def has_qwen3_5() -> bool:
    try:
        importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return False
    return True


class Recorder(torch.nn.Module if torch else object):
    """The model, recording the input shape of every forward."""

    def __init__(self, model):
        super().__init__()
        self.inner = model
        self.backbone = model.backbone
        self.head = model.head
        self.metadata = model.metadata
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, **batch):
        self.shapes.append(tuple(batch["input_ids"].shape))
        return self.inner(**batch)


@unittest.skipIf(torch is None, "needs torch")
class SwitchTest(unittest.TestCase):
    kind = "qwen3"

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.qwen = stage(Path(cls.scratch.name))
        try:
            cls.model = Recorder(tiny(cls.kind))
        except ImportError:
            raise unittest.SkipTest(f"needs Transformers with {cls.kind}")
        cls.tok = tokenizer()

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def backend(self, share=False, budget=None):
        if isinstance(share, dict):
            share = {"min_shared_tokens": 0, **share}
        return self.qwen.QwenDecision(
            self.model,
            self.tok,
            torch.device("cpu"),
            {"choice": 1.0, "noul": 1.0, "score": 1.0},
            4096,
            torch,
            batch_tokens=budget,
            share_context=share,
        )

    def close(self, left: dict, right: dict, tolerance: float = 2e-5):
        self.assertEqual(list(left), list(right))
        for qid in left:
            a, b = left[qid], right[qid]
            self.assertEqual(sorted(a), sorted(b), qid)
            if "error" in a:
                self.assertEqual(a, b)
                continue
            for key, value in probabilities(a).items():
                self.assertAlmostEqual(value, probabilities(b)[key], delta=tolerance)

    def test_off_is_the_exact_path(self):
        for name in [n for n in sys.modules if n.endswith("decision2.shared_ctx")]:
            del sys.modules[name]
        qs = questions(7)
        backend = self.backend()
        self.model.shapes.clear()
        answers, tokens = backend.system_one(STATE, qs)
        self.assertNotIn("decision2.shared_ctx", sys.modules)
        self.assertEqual(len(self.model.shapes), 1)
        self.assertEqual(self.model.shapes[0][0], 7)
        again, _ = backend.system_one(STATE, qs, share_context=False)
        self.assertEqual(again, answers)
        self.assertNotIn("decision2.shared_ctx", sys.modules)

    def test_on_equals_the_exact_path_up_to_rounding(self):
        for count, seed in ((2, 0), (3, 1), (9, 2), (16, 3)):
            qs = questions(count, seed)
            exact, tokens = self.backend().system_one(STATE, qs)
            for mode in ("tree", "cache"):
                backend = self.backend({"mode": mode})
                answers, shared_tokens = backend.system_one(STATE, qs)
                self.assertTrue(backend.share_stats["shared"], backend.share_stats)
                self.assertEqual(backend.share_stats["mode"], mode)
                self.assertGreater(backend.share_stats["prefix_tokens"], 64)
                self.assertEqual(shared_tokens, tokens)
                self.close(answers, exact)

    def test_runtime_default_and_request_override(self):
        qs = questions(5)
        exact, _ = self.backend().system_one(STATE, qs)
        backend = self.backend({})
        answers, _ = backend.system_one(STATE, qs)
        self.assertTrue(backend.share_stats["shared"])
        self.close(answers, exact)
        self.model.shapes.clear()
        off, _ = backend.system_one(STATE, qs, share_context=False)
        self.assertEqual(off, exact)
        self.assertEqual(self.model.shapes, [self.model.shapes[0]])

    def test_single_question_and_invalid_questions(self):
        backend = self.backend({})
        one = {"q0": questions(1)["q0"]}
        answers, _ = backend.system_one(STATE, one)
        self.assertEqual(backend.share_stats["reason"], "too few questions")
        self.assertEqual(answers, self.backend().system_one(STATE, one)[0])
        qs = {**questions(4), "bad": {"type": "rank", "instructions": "?"}}
        answers, _ = backend.system_one(STATE, qs)
        self.assertEqual(answers["bad"], {"type": "rank", "error": "invalid_question"})
        self.close(answers, self.backend().system_one(STATE, qs)[0])

    def test_long_prefix_and_the_break_even(self):
        long_state = {"ticket": STATE * 4, "history": [STATE[:80]] * 3}
        qs = questions(6, 4)
        probe = self.backend({})
        answers, _ = probe.system_one(long_state, qs)
        self.assertTrue(probe.share_stats["shared"], probe.share_stats)
        self.close(answers, self.backend().system_one(long_state, qs)[0])
        saved = 5 * probe.share_stats["prefix_tokens"]
        for threshold, shared in ((saved, True), (saved + 1, False)):
            backend = self.backend({"min_shared_tokens": threshold})
            backend.system_one(long_state, qs)
            self.assertEqual(backend.share_stats["shared"], shared)
        auto = sys.modules["decision2.shared_ctx"].auto_shared_tokens(
            self.model.backbone.config
        )
        backend = self.backend(True)
        backend.system_one(long_state, qs)
        self.assertEqual(backend.share_stats["shared"], saved >= auto)

    def test_over_budget_tree_runs_the_cache_mode(self):
        from decision2._vendor.dev2model.decision_model import encode
        from decision2._vendor.dev2model.infer import question_to_row

        qs = questions(8, 5)
        item = {"id": "r", "state": STATE}
        longest = max(
            len(encode(question_to_row(item, k, q), self.tok, 4096)["ids"])
            for k, q in qs.items()
        )
        budget = -(-longest // 8) * 8
        exact, _ = self.backend(budget=budget).system_one(STATE, qs)
        backend = self.backend({"mode": "tree"}, budget=budget)
        answers, _ = backend.system_one(STATE, qs)
        self.assertEqual(backend.share_stats["mode"], "cache")
        self.assertGreater(backend.share_stats["suffix_batches"], 1)
        self.close(answers, exact)

    def test_fallbacks(self):
        qs = questions(6, 6)
        exact, _ = self.backend().system_one(STATE, qs)
        backend = self.backend({"tau": 1.0})
        answers, _ = backend.system_one(STATE, qs)
        self.assertEqual(backend.share_stats["rescored"], 6)
        self.close(answers, exact, tolerance=1e-6)
        backend = self.backend({"tau": 1.0, "fallback": "request"})
        answers, _ = backend.system_one(STATE, qs)
        self.assertFalse(backend.share_stats["shared"])
        self.assertEqual(answers, exact)


@unittest.skipUnless(
    torch is not None and has_qwen3_5(), "needs Transformers with Qwen3.5"
)
class HybridSwitchTest(SwitchTest):
    kind = "qwen3_5"

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5 import modeling_qwen3_5

        # The image's causal-conv1d / FLA kernels are GPU-only and its CPU PyTorch lacks
        # the triangular solve of the chunked reference: use the torch references, with
        # the recurrent form of the gated-delta rule in both places.
        reference = {
            name: inspect.unwrap(getattr(modeling_qwen3_5, name))
            for name in (
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "torch_recurrent_gated_delta_rule",
            )
        }
        reference["torch_chunk_gated_delta_rule"] = reference[
            "torch_recurrent_gated_delta_rule"
        ]
        cls.patches = [
            mock.patch.object(modeling_qwen3_5, n, f) for n, f in reference.items()
        ]
        for patch in cls.patches:
            patch.start()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for patch in cls.patches:
            patch.stop()


if __name__ == "__main__":
    unittest.main()
