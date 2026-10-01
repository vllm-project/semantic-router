"""Per-level Score offsets (score_bias.json) in the release runtime and builder.

The staged runtime runs over the real vendored inference sources with a stub
encoder and a fake model (stdlib only, no torch).
"""

from __future__ import annotations

import importlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.model.infer import checkpoint_fingerprint, product_answer, run_prompts
from training.model.score_bias import SCORE_BIAS_FORMAT, validate_offsets
from v2.release import build, layout
from v2.release.runtime import api
from v2.release.tests.test_release import ROSTER, card_entries, fake_safetensors

REAL_MODULES = (
    "calibration.py",
    "data.py",
    "infer.py",
    "lora.py",
    "source.py",
    "score_bias.py",
)
STUB_DECISION_MODEL = '''"""Stub: prompt encoding and batching without torch."""

FAKE = {}


class DecisionModel:
    @classmethod
    def from_checkpoint(cls, root, source_path=None):
        return FAKE["model"], FAKE["tokenizer"]


def encode(row, tokenizer, max_length):
    return {
        "id": row["id"],
        "ids": [1] * len(row["options"]),
        "keys": [option["key"] for option in row["options"]],
    }


def collate(encoded, pad_id):
    return {"ids": [item["id"] for item in encoded]}
'''
IDENTITY = {"head_variant": "shared", "dec_residual": False}
TEMPERATURES = {"choice": 1.1, "noul": 0.9, "score": 1.3}
QUESTIONS = {
    "c": {
        "type": "choice",
        "instructions": "Choose",
        "criteria": {"alpha": "A", "beta": "B", "gamma": None},
    },
    "n": {
        "type": "noul",
        "instructions": "Yes?",
        "criteria": {"false": "No", "true": "Yes"},
    },
    "s3": {"type": "score", "instructions": "Rate", "criteria": ["lo", "mid", "hi"]},
    "s4": {"type": "score", "instructions": "Rate", "criteria": ["a", "b", "c", "d"]},
    "s5": {
        "type": "score",
        "instructions": "Rate",
        "criteria": ["1", "2", "3", {"level": 4}, "5"],
    },
}
LOGITS = {
    "c": [0.2, -0.1, 0.4],
    "n": [-0.3, 0.4],
    "s3": [0.0, 0.3, 0.8],
    "s4": [0.5, 0.0, 0.1, -0.2],
    "s5": [0.1, 0.7, 0.2, -0.4, 0.9],
}
OFFSETS_5 = {"5": [0.0, 0.25, -0.125, 0.5, -0.625]}
OFFSETS_35 = {"3": [-0.5, 0.0, 0.5], **OFFSETS_5}
STATE = {"ticket": "printer on fire", "priority": None}


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeModel:
    def __init__(self, table):
        self.table = table

    def __call__(self, *, ids):
        return [FakeTensor(self.table[name.split("/")[-1]]) for name in ids]

    def float(self):
        return self

    def to(self, _device):
        return self

    def eval(self):
        return self


FAKE_TORCH = SimpleNamespace(
    is_tensor=lambda _value: False,
    inference_mode=lambda: _Null(),
    device=lambda name: SimpleNamespace(type=str(name).split(":")[0]),
    set_num_threads=lambda _count: None,
)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def bias_report(model_sha256, offsets, **changes):
    value = {
        "format": SCORE_BIAS_FORMAT,
        "model_sha256": model_sha256,
        "offsets": offsets,
        "fit": {"objective": "synthetic"},
    }
    value.update(changes)
    return value


def mirror_tree(scratch: Path, *, with_score_bias: bool = True) -> Path:
    mirror = scratch / f"{'c' * 40}-src_training_decision2"
    tree = mirror / "src/training/decision2"
    model = tree / "training/model"
    model.mkdir(parents=True)
    for name in REAL_MODULES:
        if name != "score_bias.py" or with_score_bias:
            shutil.copyfile(build.SOURCE_ROOT / "training/model" / name, model / name)
    (model / "decision_model.py").write_text(STUB_DECISION_MODEL, encoding="utf-8")
    (mirror / ".dev2-mirror.json").write_text(
        json.dumps({"commit": "c" * 40, "tree": "d" * 40})
    )
    return tree


class StagedRuntime:
    """Import a staged ``decision2`` package; purge it from sys.modules on exit."""

    def __init__(self, stage: Path):
        self.stage = str(stage)

    def _purge(self):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]

    def __enter__(self):
        self._purge()
        sys.path.insert(0, self.stage)
        return self

    def module(self, name: str):
        return importlib.import_module(f"decision2.{name}")

    def __exit__(self, *_exc):
        sys.path.remove(self.stage)
        self._purge()
        return False


def prompt_rows():
    return [{"id": "request", "state": STATE, "questions": QUESTIONS}]


def scored_answers(table, score_bias=None, temperature=TEMPERATURES):
    """Answers of the scoring tool (`training.model.infer.run_prompts`)."""

    def encode(row, _tokenizer, _max_length):
        return {"id": row["id"], "ids": [1], "keys": [o["key"] for o in row["options"]]}

    def predict(encoded):
        return [list(table[item["id"].split("/")[-1]]) for item in encoded]

    offsets = validate_offsets(score_bias) if score_bias else None
    predictions, _ = run_prompts(
        prompt_rows(),
        tokenizer=None,
        max_length=64,
        temperature=temperature,
        encode_fn=encode,
        predict_fn=predict,
        model_sha256="a" * 64,
        adapter_sha256="adapter",
        score_bias=offsets,
        score_bias_sha256="f" * 64 if offsets else None,
    )
    return predictions[0]["answers"]


class RuntimeScoreBiasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        scratch = Path(cls.scratch.name)
        cls.tree = mirror_tree(scratch)
        cls.stage = scratch / "stage"
        cls.stage.mkdir()
        spec = {"profile": "qwen-full", "vendor_source": str(cls.tree)}
        cls.records = build.vendor_runtime(spec, cls.stage, IDENTITY)
        cls.runtime = StagedRuntime(cls.stage).__enter__()
        cls.qwen = cls.runtime.module("qwen")

    @classmethod
    def tearDownClass(cls):
        cls.runtime.__exit__(None, None, None)
        cls.scratch.cleanup()

    def decision(self, table=LOGITS, score_bias=None):
        return self.qwen.QwenDecision(
            FakeModel(table),
            SimpleNamespace(pad_token_id=0, eos_token_id=None),
            SimpleNamespace(type="cpu"),
            TEMPERATURES,
            64,
            FAKE_TORCH,
            **(
                {"score_bias": validate_offsets(score_bias)}
                if score_bias is not None
                else {}
            ),
        )

    def answers(self, table=LOGITS, score_bias=None):
        answers, tokens = self.decision(table, score_bias).system_one(STATE, QUESTIONS)
        self.assertEqual(tokens, sum(len(q["criteria"]) for q in QUESTIONS.values()))
        return answers

    def test_vendors_score_bias_next_to_infer(self):
        self.assertIn("decision2/_vendor/dev2model/score_bias.py", self.records)
        infer = self.runtime.module("_vendor.dev2model.infer")
        self.assertTrue(hasattr(infer, "run_prompts"))

    def test_without_offsets_answers_are_unchanged(self):
        expected = {}
        for qid, question in QUESTIONS.items():
            keys = (
                [str(i) for i in range(len(question["criteria"]))]
                if question["type"] == "score"
                else list(question["criteria"])
            )
            descriptions = (
                list(question["criteria"])
                if question["type"] == "score"
                else list(question["criteria"].values())
            )
            expected[qid] = product_answer(
                question["type"],
                keys,
                LOGITS[qid],
                TEMPERATURES[question["type"]],
                descriptions,
            )
        plain = json.dumps(self.answers())
        self.assertEqual(plain, json.dumps(expected))
        self.assertEqual(json.dumps(self.answers(score_bias=None)), plain)
        self.assertEqual(json.dumps(self.answers(score_bias={"7": [0.0] * 7})), plain)

    def test_offsets_equal_the_scoring_tool(self):
        for offsets in (OFFSETS_5, OFFSETS_35):
            with self.subTest(offsets=sorted(offsets)):
                released = self.answers(score_bias=offsets)
                scored = scored_answers(LOGITS, offsets)
                self.assertEqual(set(released), set(scored))
                for qid, answer in scored.items():
                    self.assertEqual(
                        {key: released[qid][key] for key in answer}, answer, qid
                    )

    def test_only_listed_level_counts_change(self):
        plain = self.answers()
        biased = self.answers(score_bias=OFFSETS_5)
        for qid in ("c", "n", "s3", "s4"):
            self.assertEqual(biased[qid], plain[qid])
        self.assertNotEqual(biased["s5"]["score"], plain["s5"]["score"])
        self.assertEqual(biased["s5"]["legend"], plain["s5"]["legend"])
        both = self.answers(score_bias=OFFSETS_35)
        self.assertNotEqual(both["s3"]["probabilities"], plain["s3"]["probabilities"])
        self.assertEqual(both["s4"], plain["s4"])
        self.assertEqual(both["s5"], biased["s5"])
        score = both["s3"]
        self.assertTrue(
            math.isclose(
                score["score"],
                sum(int(k) * p for k, p in score["probabilities"].items()),
            )
        )

    def test_malformed_output_stays_invalid(self):
        table = {
            **LOGITS,
            "s5": [math.nan] * 5,
            "s3": [0.0, 1.0],
            "c": [math.inf, 0.0, 0.0],
        }
        for offsets in (None, OFFSETS_35):
            answers = self.answers(table, offsets)
            for qid in ("c", "s3", "s5"):
                self.assertEqual(
                    answers[qid],
                    {"type": QUESTIONS[qid]["type"], "error": "invalid_model_output"},
                )
            self.assertIn("score", answers["s4"])


class LoadBindingTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        scratch = Path(self.scratch.name)
        stage = scratch / "stage"
        stage.mkdir()
        spec = {"profile": "qwen-full", "vendor_source": str(mirror_tree(scratch))}
        build.vendor_runtime(spec, stage, IDENTITY)
        self.runtime = StagedRuntime(stage).__enter__()
        self.qwen = self.runtime.module("qwen")
        self.stub = self.runtime.module("_vendor.dev2model.decision_model")
        self.stub.FAKE.update(
            model=FakeModel(LOGITS),
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=None),
        )
        self.root = root = scratch / "pkg"
        (root / "backbone").mkdir(parents=True)
        (root / "decision_config.json").write_text('{"checkpoint_format": "full"}')
        (root / "decision_head.safetensors").write_bytes(b"head")
        (root / "backbone/model.safetensors").write_bytes(b"weights")
        self.model_sha256 = checkpoint_fingerprint(root)["model_sha256"]
        (root / "score_bias.json").write_text(
            json.dumps(bias_report(self.model_sha256, OFFSETS_35))
        )
        sha = layout.sha_file(root / "score_bias.json")
        self.manifest = {
            "profile": "qwen-full",
            "identity": {"model_sha256": self.model_sha256},
            "max_input_tokens": 64,
            "files_sha256": {"score_bias.json": sha},
            "score_bias": {
                "file": "score_bias.json",
                "sha256": sha,
                "offsets": {
                    k: [float(v) for v in row] for k, row in sorted(OFFSETS_35.items())
                },
            },
        }
        self.torch = sys.modules.get("torch")
        sys.modules["torch"] = FAKE_TORCH

    def tearDown(self):
        if self.torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.torch
        self.runtime.__exit__(None, None, None)
        self.scratch.cleanup()

    def load(self, manifest):
        return self.qwen.QwenDecision.load(
            self.root, manifest, device="cpu", base_path=None, threads=None
        )

    def test_bound_offsets_load_and_apply(self):
        decision = self.load(self.manifest)
        self.assertEqual(decision.score_bias, validate_offsets(OFFSETS_35))
        answers, _ = decision.system_one(STATE, QUESTIONS)
        self.assertEqual(
            decision.temperatures, {"choice": 1.0, "noul": 1.0, "score": 1.0}
        )
        scored = scored_answers(LOGITS, OFFSETS_35, temperature=1.0)
        for qid, answer in scored.items():
            self.assertEqual({key: answers[qid][key] for key in answer}, answer, qid)
        self.assertEqual(
            answers["s5"]["score"],
            product_answer(
                "score",
                list("01234"),
                [a + b for a, b in zip(LOGITS["s5"], OFFSETS_5["5"])],
                1.0,
                QUESTIONS["s5"]["criteria"],
            )["score"],
        )

    def test_manifest_without_offsets_loads_as_before(self):
        manifest = {k: v for k, v in self.manifest.items() if k != "score_bias"}
        decision = self.load(manifest)
        self.assertIsNone(decision.score_bias)
        self.assertIsNone(
            self.qwen.load_score_bias_entry(self.root, manifest, self.model_sha256)
        )

    def test_binding_failures_are_refused(self):
        entry = self.manifest["score_bias"]
        bad = {
            "inventory sha": {
                **self.manifest,
                "files_sha256": {"score_bias.json": "0" * 64},
            },
            "entry sha": {**self.manifest, "score_bias": {**entry, "sha256": "0" * 64}},
            "offsets": {
                **self.manifest,
                "score_bias": {**entry, "offsets": {"5": OFFSETS_5["5"]}},
            },
            "missing": {
                **self.manifest,
                "score_bias": {**entry, "file": "absent.json"},
            },
        }
        for name, manifest in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(manifest)
        path = self.root / "score_bias.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "packaged score_bias"):
            self.load(self.manifest)
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "model hash differs"):
            self.qwen.load_score_bias_entry(self.root, self.manifest, "b" * 64)
        offsets = self.qwen.load_score_bias_entry(
            self.root, self.manifest, self.model_sha256
        )
        self.assertEqual(offsets, validate_offsets(OFFSETS_35))


class VendorScoreBiasTest(unittest.TestCase):
    def test_mirror_without_score_bias_builds_as_before(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = mirror_tree(Path(scratch), with_score_bias=False)
            (tree / "training/model/infer.py").write_text("# pre-8730d9413 infer\n")
            stage = Path(scratch) / "stage"
            stage.mkdir()
            spec = {"profile": "qwen-full", "vendor_source": str(tree)}
            records = build.vendor_runtime(spec, stage, IDENTITY)
            self.assertNotIn("decision2/_vendor/dev2model/score_bias.py", records)
            spec["score_bias"] = {"path": "x", "sha256": "0" * 64}
            shutil.rmtree(stage)
            stage.mkdir()
            with self.assertRaisesRegex(ValueError, "score_bias.py"):
                build.vendor_runtime(spec, stage, IDENTITY)

    def test_vendored_infer_imports_only_with_score_bias(self):
        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch) / "stage"
            stage.mkdir()
            tree = mirror_tree(Path(scratch))
            spec = {"profile": "qwen-full", "vendor_source": str(tree)}
            build.vendor_runtime(spec, stage, IDENTITY)
            with StagedRuntime(stage) as runtime:
                runtime.module("_vendor.dev2model.infer")
            (stage / layout.VENDOR_DIR / "dev2model/score_bias.py").unlink()
            with StagedRuntime(stage) as runtime, self.assertRaises(ImportError):
                runtime.module("_vendor.dev2model.infer")


def without(value: dict, key: str) -> dict:
    return {k: v for k, v in value.items() if k != key}


def qwen_full_spec(scratch: Path) -> tuple[dict, str]:
    ckpt = scratch / "ckpt"
    (ckpt / "backbone").mkdir(parents=True)
    (ckpt / "decision_config.json").write_text(
        json.dumps({"checkpoint_format": "full", "text_parameter_count": 600_000_000})
    )
    (ckpt / "backbone/config.json").write_text('{"model_type": "qwen3"}')
    fake_safetensors(ckpt / "backbone/model.safetensors", {"w": [600_000_000]})
    fake_safetensors(ckpt / "decision_head.safetensors", {"h": [1000, 256]})
    (ckpt / "tokenizer.json").write_text("{}")
    (ckpt / "tokenizer_config.json").write_text("{}")
    model_sha256 = checkpoint_fingerprint(ckpt)["model_sha256"]
    licence_file = scratch / "LICENSE"
    licence_file.write_text("Apache License 2.0\n")
    spec = {
        "schema": build.SPEC_SCHEMA,
        "kind": "staging",
        "repo_id": "llm-semantic-router/dev2-release-staging",
        "model_name": "Decision-2.0-Kai-0.6B",
        "profile": "qwen-full",
        "checkpoint": str(ckpt),
        "expected_identity": {"model_sha256": model_sha256},
        "max_input_tokens": 8192,
        "origin": {
            "repo_id": "Qwen/Qwen3-0.6B",
            "revision": "a" * 40,
            "relation": "finetune",
            "summary": "Fine-tuned.",
        },
        "licence": {
            "components": [{"component": "Qwen3-0.6B", "licence": "apache-2.0"}],
            "files": [
                {
                    "source": str(licence_file),
                    "path": "LICENSE",
                    "sha256": layout.sha_file(licence_file),
                }
            ],
            "attributions": ["Qwen3-0.6B (Apache-2.0)."],
        },
        "card": {
            "reports": card_entries(),
            "roster": str(ROSTER),
            "text": {
                "model_type": "Decision model",
                "training_summary": "Fine-tuned on decision data.",
                "staging_notice": "Staging.",
            },
        },
    }
    return spec, model_sha256


class BuildScoreBiasTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.dir = Path(self.scratch.name)
        self.spec, self.model_sha256 = qwen_full_spec(self.dir)
        self.bias = self.dir / "fit" / "score_bias.json"
        self.bias.parent.mkdir()
        self.bias.write_text(json.dumps(bias_report(self.model_sha256, OFFSETS_35)))
        self.bias_sha = layout.sha_file(self.bias)
        self.native = self.dir / "native-manifest.json"
        self.write_native(self.bias_sha)
        self.spec["score_bias"] = {"path": str(self.bias), "sha256": self.bias_sha}
        self.spec["scored"] = {"native_manifest": str(self.native)}
        self.count = 0

    def tearDown(self):
        self.scratch.cleanup()

    def write_native(self, bias_sha, offsets=OFFSETS_35, *, entry=True):
        """A scored native manifest as training.model.infer writes it."""
        sources = {
            name: layout.sha_file(build.SOURCE_ROOT / "training/model" / name)
            for name in ("infer.py", "data.py", "score_bias.py")
        }
        native = {"model_sha256": self.model_sha256, "adapter_files_sha256": sources}
        if bias_sha is not None:
            native["score_bias_sha256"] = bias_sha
            if entry:
                native["score_bias"] = {
                    "file_sha256": bias_sha,
                    "offsets": {
                        str(k): row
                        for k, row in sorted(validate_offsets(offsets).items())
                    },
                }
        self.native.write_text(json.dumps(native))

    def build(self, spec):
        self.count += 1
        path = self.dir / f"spec{self.count}.json"
        path.write_text(json.dumps(spec))
        output = self.dir / f"out{self.count}" / "dev2-release-staging"
        build.build(path, output)
        return output

    def test_package_carries_bound_offsets(self):
        package = self.build(self.spec)
        manifest = api.verify_bundle(package)
        self.assertEqual(
            manifest["score_bias"],
            {
                "file": "score_bias.json",
                "sha256": self.bias_sha,
                "scored_sha256": self.bias_sha,
                "offsets": {
                    k: [float(v) for v in row] for k, row in sorted(OFFSETS_35.items())
                },
            },
        )
        self.assertEqual(manifest["files_sha256"]["score_bias.json"], self.bias_sha)
        vendored = "decision2/_vendor/dev2model/score_bias.py"
        self.assertEqual(
            manifest["files_sha256"][vendored],
            layout.sha_file(build.SOURCE_ROOT / "training/model/score_bias.py"),
        )
        checked = manifest["runtime"]["scored_runtime_check"]["checked"]
        self.assertEqual(
            checked["training/model/score_bias.py"],
            manifest["files_sha256"][vendored],
        )
        self.assertEqual(
            checkpoint_fingerprint(package)["model_sha256"], self.model_sha256
        )
        with StagedRuntime(package) as runtime:
            runtime.module("_vendor.dev2model.infer")
            qwen = runtime.module("qwen")
            self.assertEqual(
                qwen.load_score_bias_entry(package, manifest, self.model_sha256),
                validate_offsets(OFFSETS_35),
            )

    def test_without_offsets_the_manifest_has_no_entry(self):
        spec = {k: v for k, v in self.spec.items() if k != "score_bias"}
        self.write_native(None)
        manifest = api.verify_bundle(self.build(spec))
        self.assertNotIn("score_bias", manifest)
        self.assertNotIn("score_bias.json", manifest["files_sha256"])
        self.assertIn(
            "decision2/_vendor/dev2model/score_bias.py", manifest["files_sha256"]
        )

    def test_unbound_offsets_are_refused(self):
        wrong_model = self.dir / "wrong.json"
        wrong_model.write_text(json.dumps(bias_report("b" * 64, OFFSETS_35)))
        cases = {
            "file sha": {
                **self.spec,
                "score_bias": {"path": str(self.bias), "sha256": "0" * 64},
            },
            "model hash": {
                **self.spec,
                "score_bias": {
                    "path": str(wrong_model),
                    "sha256": layout.sha_file(wrong_model),
                },
            },
            "no scored manifest": without(self.spec, "scored"),
            "no spec offsets": without(self.spec, "score_bias"),
        }
        for name, spec in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.build(spec)
        natives = {
            "differing offsets": ((self.bias_sha, OFFSETS_5), {}),
            "one level differs": (
                (self.bias_sha, {**OFFSETS_35, "3": [-0.5, 0.0, 0.5000001]}),
                {},
            ),
            "no score_bias.offsets": ((self.bias_sha,), {"entry": False}),
            "no score_bias_sha256": ((None,), {}),
        }
        for name, (args, kwargs) in natives.items():
            self.write_native(*args, **kwargs)
            with self.subTest(name), self.assertRaisesRegex(
                ValueError, "scored run applied"
            ):
                self.build(self.spec)
        native = json.loads(self.native.read_text())
        native["score_bias"] = {"file_sha256": "e" * 64, "offsets": OFFSETS_35}
        self.native.write_text(json.dumps(native))
        with self.assertRaisesRegex(ValueError, "scored run applied"):
            self.build(without(self.spec, "score_bias"))
        self.write_native(self.bias_sha)
        native = json.loads(self.native.read_text())
        self.native.write_text(json.dumps({**native, "model_sha256": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "scored run applied"):
            self.build(self.spec)

    def test_path_free_copy_with_the_scored_offsets_builds(self):
        public = self.dir / "public" / "score_bias.json"
        public.parent.mkdir()
        public.write_text(
            json.dumps(
                bias_report(
                    self.model_sha256,
                    OFFSETS_35,
                    fit={"scored_file_sha256": self.bias_sha},
                ),
                indent=2,
            )
        )
        public_sha = layout.sha_file(public)
        self.assertNotEqual(public_sha, self.bias_sha)
        spec = {**self.spec, "score_bias": {"path": str(public), "sha256": public_sha}}
        package = self.build(spec)
        manifest = api.verify_bundle(package)
        self.assertEqual(manifest["score_bias"]["sha256"], public_sha)
        self.assertEqual(manifest["score_bias"]["scored_sha256"], self.bias_sha)
        self.assertEqual(manifest["files_sha256"]["score_bias.json"], public_sha)
        check = manifest["runtime"]["scored_runtime_check"]
        self.assertEqual(check["score_bias_sha256"], self.bias_sha)
        receipt = json.loads(
            (package.parent / f"{package.name}.build/BUILD.json").read_text()
        )
        self.assertEqual(receipt["score_bias"], manifest["score_bias"])
        with StagedRuntime(package) as runtime:
            qwen = runtime.module("qwen")
            self.assertEqual(
                qwen.load_score_bias_entry(package, manifest, self.model_sha256),
                validate_offsets(OFFSETS_35),
            )


class ScreenScoreBiasTest(unittest.TestCase):
    def test_screen_accepts_public_and_refuses_private_provenance(self):
        with tempfile.TemporaryDirectory() as scratch:
            stage = Path(scratch)
            path = stage / "score_bias.json"
            public = {"scored_file_sha256": "0" * 64, "inputs": {"gold": "1" * 64}}
            path.write_text(json.dumps(bias_report("a" * 64, OFFSETS_5, fit=public)))
            self.assertEqual(build.screen(stage)["text_files_screened"], 1)
            private = {
                "inputs": {
                    "gold": {
                        "path": "/data/dev2/private/gold.jsonl",
                        "sha256": "1" * 64,
                    }
                }
            }
            path.write_text(json.dumps(bias_report("a" * 64, OFFSETS_5, fit=private)))
            with self.assertRaisesRegex(ValueError, "Private path"):
                build.screen(stage)


if __name__ == "__main__":
    unittest.main()
