"""Verify measured accounting without downloading torch or model artifacts."""

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROMPT_TOKENS = 7


def load_verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "model_selection"
        / "rl_model_selection"
        / "automix_verifier.py"
    )
    spec = importlib.util.spec_from_file_location("automix_usage_fixture", path)
    module = importlib.util.module_from_spec(spec)
    fake_torch = SimpleNamespace(no_grad=contextlib.nullcontext)
    with patch.dict(
        sys.modules, {"torch": fake_torch, "transformers": None, spec.name: module}
    ):
        spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, _device):
        return self

    def __getitem__(self, index):
        tokens = list(range(self.shape[-1]))
        return tokens[index[1]] if isinstance(index, tuple) else tokens


class Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt, **_kwargs):
        self.prompt = prompt
        return {"input_ids": Tensor(PROMPT_TOKENS)}

    def decode(self, tokens, **_kwargs):
        return "YES" if tokens[0] >= PROMPT_TOKENS else self.prompt + "YES"


class Model:
    def __init__(self):
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return Tensor(PROMPT_TOKENS + self.calls + 1)


class AutoMixUsageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_verifier()

    def verifier(self):
        verifier = self.module.AutoMixVerifier.__new__(self.module.AutoMixVerifier)
        verifier.model = Model()
        verifier.tokenizer = Tokenizer()
        verifier.num_samples = 2
        verifier.temperature = 0.7
        verifier.device = "cpu"
        return verifier

    def test_counts_each_prompt_and_generated_tokens_before_decoding(self):
        confidence, samples, usage = self.verifier().verify_with_usage(
            "question", "answer"
        )
        self.assertEqual(confidence, 1.0)
        self.assertEqual(samples, ["YES", "YES"])
        self.assertEqual(
            usage, {"prompt_tokens": 14, "completion_tokens": 5, "total_tokens": 19}
        )

    def test_decodes_generated_tokens_after_prompt_truncation(self):
        class TruncatedTokenizer(Tokenizer):
            def decode(self, tokens, **kwargs):
                if tokens[0] == 0:
                    return "truncated promptYES"
                return super().decode(tokens, **kwargs)

        verifier = self.verifier()
        verifier.tokenizer = TruncatedTokenizer()
        _, samples, usage = verifier.verify_with_usage(
            "question", "answer", context="context " * 100
        )
        self.assertEqual(samples, ["YES", "YES"])
        self.assertEqual(usage["total_tokens"], 19)

    def test_existing_tuple_interface_is_preserved(self):
        self.assertEqual(
            self.verifier().verify("question", "answer"), (1.0, ["YES", "YES"])
        )

    def test_missing_model_has_unknown_usage(self):
        verifier = self.verifier()
        verifier.model = None
        _, _, usage = verifier.verify_with_usage("question", "answer")
        self.assertIsNone(usage)

    def test_http_handler_publishes_usage_only_when_measured(self):
        for measured in (True, False):
            with self.subTest(measured=measured):
                handler = self.module.AutoMixHandler.__new__(self.module.AutoMixHandler)
                handler.path = "/verify"
                body = json.dumps({"question": "question", "answer": "answer"}).encode()
                handler.headers = {"Content-Length": str(len(body))}
                handler.rfile = io.BytesIO(body)
                handler.verifier = self.verifier()
                if not measured:
                    handler.verifier.model = None
                responses = []
                handler._send_json = lambda status, payload, output=responses: (
                    output.append(payload)
                )
                handler.do_POST()
                self.assertEqual("usage" in responses[0], measured)
                if measured:
                    self.assertEqual(responses[0]["usage"]["total_tokens"], 19)


if __name__ == "__main__":
    unittest.main()
