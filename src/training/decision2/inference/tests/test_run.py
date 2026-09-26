"""Collector integrity checks without model weights or local GPU inference."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.run import (
    collect,
    digest,
    eos_runtime_report,
    load_prompts,
    verify_eos_manifest,
    verify_model_family,
)


class CollectorTest(unittest.TestCase):
    def test_native_answers_pass_through_and_resume_checks_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "decider_config.json").write_text("{}")
            prompts = root / "prompts.jsonl"
            row = {
                "id": "one",
                "state": {"value": 1},
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {"a": "A", "b": "B"},
                    },
                    "flag": {"type": "noul", "instructions": "True?"},
                    "rank": {
                        "type": "score",
                        "instructions": "Rank",
                        "criteria": ["low", "high"],
                    },
                },
            }
            prompts.write_text(json.dumps(row) + "\n")
            answers = {
                "route": {
                    "type": "choice",
                    "choice": "b",
                    "probabilities": {"a": 0.25, "b": 0.75},
                },
                "flag": {"type": "noul", "noul": 0.7},
                "rank": {
                    "type": "score",
                    "score": 0.6,
                    "probabilities": {"0": 0.4, "1": 0.6},
                },
            }
            calls = []

            def fake_decide(**payload):
                calls.append(payload)
                return {"model": "decider-test", "answers": answers}

            output = root / "predictions.jsonl"
            with patch(
                "inference.run.load_decider", return_value=(object(), fake_decide, {})
            ):
                first = collect(
                    backend="decider",
                    model_path=model_dir,
                    revision="commit",
                    prompts=prompts,
                    output=output,
                    device="cpu",
                )
                second = collect(
                    backend="decider",
                    model_path=model_dir,
                    revision="commit",
                    prompts=prompts,
                    output=output,
                    device="cpu",
                    resume=True,
                )
            self.assertEqual(first["collected_now"], 1)
            self.assertEqual(second["collected_now"], 0)
            self.assertEqual(
                calls, [{"state": row["state"], "questions": row["questions"]}]
            )
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["answers"], answers)
            self.assertEqual(receipt["source_input_sha256"], digest(calls[0]))
            row["state"] = {"value": 2}
            prompts.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "stale input"):
                collect(
                    backend="decider",
                    model_path=model_dir,
                    revision="commit",
                    prompts=prompts,
                    output=output,
                    device="cpu",
                    resume=True,
                )

    def test_refuses_gold_bearing_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "one", "state": {}, "questions": {"q": {}}, "gold": {}}
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "gold-bearing"):
                load_prompts(path)

    def test_nox_sol_eos_identity_and_pinned_revision(self):
        families = {
            "nox": ("Decision-1.0-Nox", "Qwen/Qwen3.5-4B", "bundle-manifest.json"),
            "sol": ("Decision-1.0-Sol", "Qwen/Qwen3.5-2B", "bundle-manifest.json"),
            "eos": ("Decision-1.0-Eos", "Qwen/Qwen3.5-0.8B", "MODEL_MANIFEST.json"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "state": {},
                        "questions": {"q": {"type": "noul", "instructions": "True?"}},
                    }
                )
                + "\n"
            )
            for backend, (native_name, base_model, manifest_name) in families.items():
                model = root / backend
                model.mkdir()
                (model / "decision_config.json").write_text(
                    json.dumps({"base_model": base_model, "model_name": native_name})
                )
                (model / manifest_name).write_text("{}")
                metadata = (
                    model / ".cache/huggingface/download/decision_config.json.metadata"
                )
                metadata.parent.mkdir(parents=True)
                metadata.write_text("exact-commit\nlocal-etag\n")
                loader = (
                    "inference.run.load_eos"
                    if backend == "eos"
                    else "inference.run.load_pointer_bundle"
                )

                def fake_decide(*, native_name=native_name, **_payload):
                    return {
                        "model": native_name,
                        "answers": {"q": {"type": "noul", "noul": 0.75}},
                    }

                output = root / f"{backend}.predictions.jsonl"
                with patch(
                    loader,
                    return_value=(
                        object(),
                        fake_decide,
                        {"runtime_matches_validated": True, "runtime_differences": {}},
                    ),
                ):
                    first = collect(
                        backend=backend,
                        model_path=model,
                        revision="exact-commit",
                        prompts=prompts,
                        output=output,
                        device="cpu",
                    )
                    second = collect(
                        backend=backend,
                        model_path=model,
                        revision="exact-commit",
                        prompts=prompts,
                        output=output,
                        device="cpu",
                        resume=True,
                    )
                receipt = json.loads(output.read_text())
                self.assertEqual(first["collected_now"], 1)
                self.assertEqual(second["collected_now"], 0)
                self.assertEqual(
                    receipt["model_id"],
                    f"llm-semantic-router/Decision-1.0-{native_name.split('-')[-1]}-"
                    + {"nox": "4B", "sol": "2B", "eos": "0.8B"}[backend],
                )
                self.assertTrue(receipt["revision_attested"])
                self.assertTrue(receipt["runtime_matches_validated"])
                with self.assertRaisesRegex(ValueError, "attest requested revision"):
                    collect(
                        backend=backend,
                        model_path=model,
                        revision="wrong-commit",
                        prompts=prompts,
                        output=root / f"{backend}.wrong.jsonl",
                        device="cpu",
                    )
                if backend == "nox":
                    with patch(
                        loader,
                        return_value=(
                            object(),
                            lambda **_payload: {
                                "model": "Decision-1.0-Sol",
                                "answers": {"q": {"type": "noul", "noul": 0.75}},
                            },
                            {},
                        ),
                    ), self.assertRaisesRegex(ValueError, "native model identity"):
                        collect(
                            backend=backend,
                            model_path=model,
                            revision="exact-commit",
                            prompts=prompts,
                            output=root / "wrong-model.jsonl",
                            device="cpu",
                        )

    def test_family_mismatch_and_eos_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "decision_config.json").write_text(
                json.dumps({"base_model": "Qwen/Qwen3.5-2B"})
            )
            with self.assertRaisesRegex(ValueError, "expects base_model"):
                verify_model_family(root, "nox")
            files = dict.fromkeys(
                (
                    "decision/engine.py",
                    "decision/model.py",
                    "decision/types.py",
                    "decision_config.json",
                    "decision_head.safetensors",
                    "runtime.json",
                    "tokenizer.json",
                    "backbone/config.json",
                    "backbone/model.safetensors",
                ),
                b"content",
            )
            import hashlib

            manifest = {"model_name": "Decision-1.0-Eos", "files": {}}
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                manifest["files"][name] = {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            (root / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
            verify_eos_manifest(root)
            (root / "decision_head.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                verify_eos_manifest(root)

    def test_eos_runtime_is_strict_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qualified = {
                "torch": "2.12",
                "hip": "7.2",
                "transformers": "5.17",
                "flash_linear_attention": "0.5.2",
                "device_architecture": "gfx942",
                "attention": "sdpa",
            }
            (root / "runtime.json").write_text(
                json.dumps({"temperature": 1.04, "qualified_runtime": qualified})
            )
            (root / "decision_config.json").write_text(
                json.dumps({"attention": "sdpa"})
            )
            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "2.12"
            fake_torch.version = types.SimpleNamespace(hip="7.2")
            fake_torch.cuda = types.SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _device: types.SimpleNamespace(
                    gcnArchName="gfx942:sramecc+:xnack-"
                ),
            )
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.__version__ = "5.17"
            fake_fla = types.ModuleType("fla")
            fake_fla.__version__ = "0.5.2"
            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                    "fla": fake_fla,
                },
            ):
                self.assertTrue(
                    eos_runtime_report(root, "cuda:0", False)[
                        "runtime_matches_validated"
                    ]
                )
                fake_fla.__version__ = "different"
                with self.assertRaisesRegex(
                    RuntimeError, "differs from the qualified release"
                ):
                    eos_runtime_report(root, "cuda:0", False)
                exploratory = eos_runtime_report(root, "cuda:0", True)
                self.assertFalse(exploratory["runtime_matches_validated"])
                self.assertIn(
                    "flash_linear_attention", exploratory["runtime_differences"]
                )


if __name__ == "__main__":
    unittest.main()
