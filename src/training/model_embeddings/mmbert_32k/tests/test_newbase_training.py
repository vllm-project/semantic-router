"""Optimizer/RNG resume preserves a real next update and rejects other arms."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.newbase_data import (
        FrozenCorpus,
        file_digest,
        text_digest,
    )
    from src.training.model_embeddings.mmbert_32k.newbase_model import (
        ExitSpec,
        NewBaseTask,
        state_digest,
    )
    from src.training.model_embeddings.mmbert_32k.newbase_stream import prepare_stream
    from src.training.model_embeddings.mmbert_32k.newbase_teacher import (
        identity_digest,
        input_identity,
        record_inputs,
        token_digest,
    )
    from src.training.model_embeddings.mmbert_32k.newbase_training import (
        make_optimizer,
        restore_training_state,
        run,
        save_training_state,
        validate_config,
    )
    from src.training.model_embeddings.mmbert_32k.tests import (
        test_newbase_batches as batch_fixtures,
    )
    from src.training.model_embeddings.mmbert_32k.tests import (
        test_newbase_scoring as score_fixtures,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class NewBaseTrainingTest(unittest.TestCase):
    def test_resume_reproduces_dropout_optimizer_moments_and_next_update(self):
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
        torch.manual_seed(77)
        model = NewBaseTask(
            ModernBertModel(config),
            "reranker",
            ExitSpec(dimensions=(16, 12, 8, 4, 2)),
            {},
        ).train()
        optimizer, scheduler, groups = make_optimizer(model, steps=10, warmup=2)
        self.assertEqual(
            sum(row["parameters"] for row in groups),
            sum(parameter.numel() for parameter in model.parameters()),
        )
        ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])

        def update(current, opt, schedule):
            opt.zero_grad(set_to_none=True)
            values = current(ids, ids.ne(0).long())
            loss = sum(
                weight * torch.nn.functional.softplus(-values[key]).mean()
                for key, weight in current.exits.weighted()
            )
            loss.backward()
            opt.step()
            schedule.step()
            return loss.detach()

        update(model, optimizer, scheduler)
        identity = {"run_id": "first", "plan": "fixed"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot"
            save_training_state(
                model,
                None,
                optimizer,
                scheduler,
                path,
                identity=identity,
                step=1,
                elapsed=2.0,
                evaluations=[{"step": 0}],
            )
            expected_loss = update(model, optimizer, scheduler)
            expected_state = state_digest(model.state_dict())
            restored = NewBaseTask.resume(path).train()
            new_optimizer, new_scheduler, _ = make_optimizer(
                restored, steps=10, warmup=2
            )
            state = restore_training_state(
                path, new_optimizer, new_scheduler, identity=identity
            )
            self.assertEqual(state["evaluations"], [{"step": 0}])
            actual_loss = update(restored, new_optimizer, new_scheduler)
            torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
            self.assertEqual(state_digest(restored.state_dict()), expected_state)
            with self.assertRaisesRegex(ValueError, "different locked run"):
                restore_training_state(
                    path, new_optimizer, new_scheduler, identity={"run_id": "second"}
                )

    def test_missing_run_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "task and nonempty run_id"):
            validate_config({"status": "proposal"})

    def test_arbitrary_two_step_cpu_run_and_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            base = root / "encoder"
            encoder = batch_fixtures.NewBaseBatchesTest().model("embedding").encoder
            del encoder.config.representation_contract
            encoder.save_pretrained(base)
            tokenizer = score_fixtures.NewBaseScoringTest().tokenizer()
            tokenizer.save_pretrained(base)
            model_files = {
                p.name: file_digest(p) for p in base.iterdir() if p.is_file()
            }
            rows, components = batch_fixtures.NewBaseBatchesTest().examples()
            for identity, component in components.items():
                component.update(
                    id=identity, text_sha256=text_digest(component["text"])
                )
            datasets = {}
            for split in ("train", "validation"):
                directory = root / split
                directory.mkdir()
                selected = copy.deepcopy(rows)
                for index, row in enumerate(selected):
                    row.update(
                        id=f"example-{index}",
                        split=split,
                        source="catalog",
                        language="en",
                        parent_groups=[f"family-{index}"],
                    )
                for filename, data in (
                    ("components.jsonl", components.values()),
                    ("records.jsonl", selected),
                ):
                    (directory / filename).write_text(
                        "".join(json.dumps(row) + "\n" for row in data)
                    )
                manifest = {
                    "split": split,
                    "files": {
                        filename: {"sha256": file_digest(directory / filename)}
                        for filename in ("components.jsonl", "records.jsonl")
                    },
                }
                (directory / "manifest.json").write_text(json.dumps(manifest))
                datasets[split] = FrozenCorpus.load(directory, split)
            stream = prepare_stream(
                datasets["train"],
                {
                    "seed": 17,
                    "cycles": 1,
                    "sources": {
                        "catalog": {
                            "steps_per_cycle": 2,
                            "batch_queries": 2,
                            "strata": {
                                "all": [row["id"] for row in datasets["train"].records]
                            },
                        }
                    },
                },
                root / "draws",
            )
            code = Path(__file__).resolve().parents[1]
            lock = root / "code.json"
            lock.write_text(
                json.dumps(
                    {
                        "files": {
                            str(p.relative_to(code)): file_digest(p)
                            for p in code.rglob("*.py")
                        }
                    }
                )
            )
            config = {
                "task": "reranker",
                "run_id": "custom-cpu-check",
                "steps": 2,
                "maximum_seconds": 60,
                "eval_steps": [0, 2],
                "seed": 17,
                "training_precision": "float32",
                "gradient_clip_norm": 1.0,
                "training_token_budget": 128,
                "evaluation_token_budget": 128,
                "source_maximum_tokens": {"catalog": 128},
                "objectives": {"catalog": {"kind": "ranking", "lambda_weight": 0.2}},
                "exits": ExitSpec(dimensions=(16, 12, 8, 4, 2)).to_dict(),
                "code_root": str(code),
                "execution_lock_path": str(lock),
                "execution_lock_sha256": file_digest(lock),
                "base_directory": str(base),
                "base_files": model_files,
                "train_directory": str(root / "train"),
                "development_directory": str(root / "validation"),
                "train_manifest_sha256": datasets["train"].manifest_sha256,
                "development_manifest_sha256": datasets["validation"].manifest_sha256,
                "draws_directory": str(root / "draws"),
                "draws_sha256": stream["draws_sha256"],
                "gradient_checkpointing": False,
                "optimizer": {"warmup_steps": 1, "encoder_lr": 2e-5, "head_lr": 1e-4},
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            output = root / "run"
            run(config, file_digest(path), output, device=torch.device("cpu"))
            result = json.loads((output / "completion.json").read_bytes())
            self.assertEqual(result["run_id"], "custom-cpu-check")
            self.assertTrue(result["all_draws_complete"])
            self.assertEqual(result["completed_steps"], 2)
            self.assertIsNone(result["error_type"])
            self.assertIsNone(result["selection"])
            self.assertNotEqual(
                json.loads((output / "step-0/newbase_checkpoint.json").read_bytes())[
                    "state_sha256"
                ],
                json.loads((output / "step-2/newbase_checkpoint.json").read_bytes())[
                    "state_sha256"
                ],
            )
            self.assertEqual(len(NewBaseTask.resume(output / "step-2").layer_heads), 4)
            cache = root / "teacher"
            cache.mkdir()
            inputs = sorted(
                {
                    ids
                    for row in datasets["train"].records
                    for ids in record_inputs(row, "reranker")
                }
            )
            entries = []
            for index, ids in enumerate(inputs):
                identity = input_identity(ids, components)
                batch = batch_fixtures.TokenBatch.encode(
                    tokenizer,
                    [components[ids[0]]["text"]],
                    [components[ids[1]]["text"]],
                    128,
                )
                entries.append(
                    {
                        "key": identity_digest(identity),
                        "identity": identity,
                        "value_index": index,
                        "token_sha256": token_digest(batch.rows[0], batch.pad_id),
                    }
                )
            (cache / "entries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in entries)
            )
            save_file(
                {"values": torch.arange(len(inputs), dtype=torch.float32)[:, None]},
                cache / "values.safetensors",
            )
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "task": "reranker",
                        "source_split": "train",
                        "train_manifest_sha256": datasets["train"].manifest_sha256,
                        "draws_sha256": stream["draws_sha256"],
                        "dimensions": 1,
                        "teacher": {
                            "repo_id": "synthetic/teacher",
                            "revision": "fixed",
                            "files": {"model.safetensors": "fixed-digest"},
                            "representation": "synthetic pair scores",
                        },
                        "files": {
                            filename: file_digest(cache / filename)
                            for filename in ("entries.jsonl", "values.safetensors")
                        },
                    }
                )
            )
            config["teacher"] = {
                "objective": "query_order",
                "exit_supervision": "all",
                "weight": 0.25,
                "directory": str(cache),
                "manifest_sha256": file_digest(cache / "manifest.json"),
            }
            path.write_text(json.dumps(config))
            treated = root / "teacher-run"
            run(config, file_digest(path), treated, device=torch.device("cpu"))
            result = json.loads((treated / "completion.json").read_bytes())
            self.assertTrue(result["all_draws_complete"])
            self.assertEqual(
                result["teacher_manifest_sha256"], config["teacher"]["manifest_sha256"]
            )
            self.assertEqual(
                state_digest(NewBaseTask.resume(output / "step-0").state_dict()),
                state_digest(NewBaseTask.resume(treated / "step-0").state_dict()),
            )
            self.assertNotEqual(
                state_digest(NewBaseTask.resume(output / "step-2").state_dict()),
                state_digest(NewBaseTask.resume(treated / "step-2").state_dict()),
            )
            # A new optimizer run can continue the complete task checkpoint;
            # this is distinct from both fresh-Base initialization and resume.
            initial = treated / "step-2"
            config.update(
                initialization="continued_task",
                base_directory=str(initial),
                base_files={
                    str(p.relative_to(initial)): file_digest(p)
                    for p in initial.rglob("*")
                    if p.is_file()
                },
                provenance={
                    "parent_task_state": state_digest(
                        NewBaseTask.resume(initial).state_dict()
                    )
                },
            )
            path.write_text(json.dumps(config))
            continued = root / "continued-run"
            run(config, file_digest(path), continued, device=torch.device("cpu"))
            self.assertEqual(
                state_digest(NewBaseTask.resume(continued / "step-0").state_dict()),
                state_digest(NewBaseTask.resume(initial).state_dict()),
            )
            self.assertEqual(
                NewBaseTask.resume(continued / "step-2").lineage["initialization"],
                "continued_task",
            )


if __name__ == "__main__":
    unittest.main()
