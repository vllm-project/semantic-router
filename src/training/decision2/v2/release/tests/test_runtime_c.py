"""Runtime C: share_context through the remote code and pipeline, the card's speed and share_context lines,
the public many-question request, and the fast path's tree and residual dispatch (stdlib; torch parts skip).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from v2.release import card, examples, shared_request
from v2.release.tests.test_release import build_test_card, card_entries, facts

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)
SHARED = {"questions": 128, "off_ms": 228.53, "on_ms": 78.31}


class CardTest(unittest.TestCase):
    def build(self, details):
        scratch = Path(self.enterContext(tempfile.TemporaryDirectory()))
        details = {
            **facts(),
            "profile": "qwen-full",
            "remote_code": {"tested": ["5.17.0", "5.18.0"], "base": None},
            **details,
        }
        result = build_test_card(scratch, card_entries(), details, {})
        return result["readme"], result["files"] | {"LICENSE"}

    def test_speed_line_and_share_context_comment(self):
        readme, files = self.build({"speed_shared": SHARED})
        speed = next(line for line in readme.splitlines() if "**Speed:**" in line)
        self.assertEqual(
            speed,
            "- **Speed:** a median of 12.3 ms per single-question request on a single GPU; 128 questions "
            "about one input take 78.3 ms with `share_context=True` instead of 229 ms.",
        )
        code = examples.card_block(readme, transformers=True)
        compile(code, "card", "exec")
        self.assertIn(
            "# model.system_one(state=..., questions=..., share_context=True)\n", code
        )
        self.assertEqual(card.check_rendered(readme, files), [])
        quickstart = readme.split("## Quickstart\n", 1)[1].split("\n## ", 1)[0]
        self.assertEqual(re.sub(r"```.*?```", "", quickstart, flags=re.S).strip(), "")

    def test_without_shared_speed_the_card_is_unchanged(self):
        readme, _ = self.build({})
        self.assertIn(
            "- **Speed:** a median of 12.3 ms per single-question request on a single GPU.\n",
            readme,
        )
        self.assertNotIn("share_context", readme)


@unittest.skipUnless(HAS_TORCH, "needs torch (the inference sources import it)")
class SharedRequestTest(unittest.TestCase):
    def test_questions_are_distinct_and_valid(self):
        from training.model.infer import question_to_row

        request = shared_request.request()
        questions = request["questions"]
        self.assertEqual(len(questions), 128)
        self.assertEqual(
            len({json.dumps(q, sort_keys=True) for q in questions.values()}), 128
        )
        kinds = [q["type"] for q in list(questions.values())[:16]]
        self.assertEqual(set(kinds), {"choice", "noul", "score"})
        item = {"id": "request", "state": request["state"]}
        for qid, question in questions.items():
            question_to_row(item, qid, question)
        self.assertEqual(list(shared_request.questions(16)), list(questions)[:16])
        with self.assertRaises(ValueError):
            shared_request.questions(129)
        self.assertEqual(shared_request.request(), request)


class PipelineTest(unittest.TestCase):
    def load(self):
        transformers = types.ModuleType("transformers")

        class Pipeline:
            def __call__(self, inputs, **kwargs):
                _, forward, _ = self._sanitize_parameters(**kwargs)
                return self.postprocess(
                    self._forward(self.preprocess(inputs), **forward)
                )

        transformers.Pipeline = Pipeline
        path = Path(examples.__file__).with_name("automap") / "pipeline_decision2.py"
        spec = importlib.util.spec_from_file_location("pipeline_decision2_test", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            spec.loader.exec_module(module)
        return module

    def test_share_context_reaches_system_one_only_when_given(self):
        module = self.load()
        calls = []

        class Model:
            def system_one(self, **kwargs):
                calls.append(kwargs)
                return {"answers": {}}

        decide = module.Decision2Pipeline.__new__(module.Decision2Pipeline)
        decide.model = Model()
        request = {"state": "s", "questions": {"q": {}}}
        decide(request)
        decide(request, share_context=True)
        decide(state="s", questions={"q": {}}, share_context={"tau": 0.01})
        self.assertEqual(
            calls,
            [
                {"state": "s", "questions": {"q": {}}},
                {"state": "s", "questions": {"q": {}}, "share_context": True},
                {"state": "s", "questions": {"q": {}}, "share_context": {"tau": 0.01}},
            ],
        )
        with self.assertRaises(TypeError):
            decide(request, batch_tokens=4)


@unittest.skipUnless(HAS_TORCH, "needs torch and transformers")
class ModelingTest(unittest.TestCase):
    def test_runtime_options_and_per_request_switch(self):
        from v2.release.automap import modeling_decision2 as m

        self.assertTrue(
            {"graphs", "kernels", "share_context"} <= set(m.RUNTIME_OPTIONS)
        )
        calls = []

        class Runtime:
            def system_one(self, **kwargs):
                calls.append(kwargs)
                return {"model": "x", "answers": {}, "usage": {}}

        model = m.Decision2Model.__new__(m.Decision2Model)
        object.__setattr__(model, "runtime", Runtime())
        model.system_one(state="s", questions={"q": {}})
        model.system_one(state="s", questions={"q": {}}, share_context=True)
        model.forward("s", {"q": {}}, share_context=False)
        self.assertEqual(
            calls,
            [
                {"state": "s", "questions": {"q": {}}},
                {"state": "s", "questions": {"q": {}}, "share_context": True},
                {"state": "s", "questions": {"q": {}}, "share_context": False},
            ],
        )


@unittest.skipUnless(HAS_TORCH, "needs torch")
class FastDispatchTest(unittest.TestCase):
    def test_tree_and_residual(self):
        import torch

        from v2.release.runtime import fast

        class Tree:
            tree_attention = staticmethod(lambda *a, **k: None)

        tree = Tree()
        self.assertIs(fast._tree(tree), tree)
        for mask in (None, {"full_attention": None}, torch.ones(2, 3)):
            self.assertIsNone(fast._tree(mask))
        used = []
        kernels = types.SimpleNamespace(
            residual_add=lambda h, d: used.append(1) or h + d.float()
        )
        hidden = torch.randn(2, 4, 8)
        delta = torch.randn(2, 4, 8).to(torch.bfloat16)
        self.assertTrue(
            torch.equal(fast._residual(kernels, hidden, delta, torch), hidden + delta)
        )
        self.assertEqual(used, [1])
        strided = torch.randn(2, 8, 4).to(torch.bfloat16).transpose(1, 2)
        self.assertTrue(
            torch.equal(
                fast._residual(kernels, hidden, strided, torch), hidden + strided
            )
        )
        self.assertEqual(used, [1])


if __name__ == "__main__":
    unittest.main()
