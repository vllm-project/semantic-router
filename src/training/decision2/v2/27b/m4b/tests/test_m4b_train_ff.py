import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from training.model import train as reference_train
from training.model.data import load_partition
from training.model.decision_model import DecisionModel, encode
from training.model.train import learning_factor
from v2.dec.batching import padded, row_windows, token_batches
from v2.dec.train_dec import attach_teacher_probs, load_teacher

train_ff = importlib.import_module("v2.27b.m4b.train_ff")
reload_check = importlib.import_module("v2.27b.m4b.reload_check")
tiny = importlib.import_module("v2.27b.m4b.tiny")
ROOT = Path(train_ff.__file__).resolve().parents[3]
ARCHES = ("qwen3", "qwen3_5") if tiny.has_qwen3_5() else ("qwen3",)


def lengths_like_train(count: int, seed: int = 3) -> list[int]:
    rng = random.Random(seed)
    return [
        rng.choice((rng.randint(20, 400), rng.randint(400, 4096))) for _ in range(count)
    ]


class PlanTest(unittest.TestCase):
    def test_update_plan_is_token_batches_in_row_windows(self):
        lengths = lengths_like_train(25822)
        plan = train_ff.update_plan(
            lengths, seed=20260926, max_tokens=32768, max_rows=64, update_rows=64
        )
        self.assertEqual(len(plan), 404)
        self.assertEqual([len(rows) for rows in plan[:-1]], [64] * 403)
        self.assertEqual(len(plan[-1]), 25822 - 403 * 64)
        order = [
            i
            for batch in token_batches(
                lengths, seed=20260926, epoch=0, max_tokens=32768, max_rows=64
            )
            for i in batch
        ]
        self.assertEqual([i for rows in plan for i in rows], order)
        self.assertEqual(len(row_windows([[i] for i in order], 64)), 404)

    def test_split_covers_is_deterministic_and_balanced(self):
        lengths = lengths_like_train(2000)
        plan = train_ff.update_plan(
            lengths, seed=1, max_tokens=32768, max_rows=64, update_rows=64
        )
        for rows in plan[:40]:
            parts = train_ff.split_rows(rows, lengths, 3)
            self.assertEqual(parts, train_ff.split_rows(list(rows), list(lengths), 3))
            self.assertEqual(sorted(i for part in parts for i in part), sorted(rows))
            for part in parts:
                self.assertEqual(part, [i for i in rows if i in set(part)])
            loads = [sum(padded(lengths[i]) for i in part) for part in parts]
            self.assertLessEqual(
                max(loads) - min(loads), max(padded(lengths[i]) for i in rows)
            )

    def test_rank_schedule_limits_and_dummies(self):
        lengths = lengths_like_train(600)
        for rows in train_ff.update_plan(
            lengths, seed=2, max_tokens=32768, max_rows=64, update_rows=64
        ):
            schedule = train_ff.rank_schedule(
                rows, lengths, 3, max_tokens=32768, max_rows=64
            )
            self.assertEqual(len({len(part) for part in schedule}), 1)
            flat = [i for part in schedule for batch in part if batch for i in batch]
            self.assertEqual(sorted(flat), sorted(rows))
            for part in schedule:
                kinds = [batch is None for batch in part]
                self.assertEqual(kinds, sorted(kinds))
                for batch in part:
                    if batch:
                        self.assertLessEqual(len(batch), 64)
                        self.assertLessEqual(
                            train_ff.padded_tokens(batch, lengths), 32768
                        )
        schedule = train_ff.rank_schedule(
            [5], lengths, 3, max_tokens=32768, max_rows=64
        )
        self.assertEqual(schedule, [[[5]], [None], [None]])

    def test_eval_steps_and_shards(self):
        self.assertEqual(
            sorted(train_ff.eval_steps(404)), [51, 102, 153, 204, 255, 306, 357, 404]
        )
        self.assertEqual(sorted(train_ff.eval_steps(8)), list(range(1, 9)))
        self.assertEqual(
            train_ff.shard_positions(700, 3, 1), (list(range(1, 700, 3)), 234)
        )
        self.assertEqual(train_ff.shard_positions(1, 3, 2), ([], 1))


class ScheduleAndSelectionTest(unittest.TestCase):
    def test_learning_rate_shape(self):
        factors = [learning_factor(step, 404, 0.1) for step in range(404)]
        self.assertAlmostEqual(factors[0], 1 / 40)
        self.assertAlmostEqual(factors[39], 1.0)
        self.assertAlmostEqual(factors[40], 1.0)
        self.assertTrue(all(a < b for a, b in zip(factors[:39], factors[1:40])))
        self.assertTrue(all(a >= b for a, b in zip(factors[40:], factors[41:])))
        self.assertAlmostEqual(learning_factor(404, 404, 0.1), 0.1)
        self.assertGreater(factors[-1], 0.1)

    def test_best_rule(self):
        def m(accuracy, brier):
            return {"family_macro_accuracy": accuracy, "family_macro_brier": brier}

        records = [(51, m(0.6, 0.3))]
        self.assertTrue(train_ff.is_best(records, 51))
        records.append((102, m(0.6, 0.3)))
        self.assertFalse(train_ff.is_best(records, 102))
        records.append((153, m(0.6, 0.29)))
        self.assertTrue(train_ff.is_best(records, 153))
        records.append((204, m(0.61, 0.5)))
        self.assertTrue(train_ff.is_best(records, 204))
        records.append((255, m(0.59, 0.1)))
        self.assertFalse(train_ff.is_best(records, 255))


class TinyModelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.sources = {
            arch: tiny.write_source(cls.root / arch, arch=arch) for arch in ARCHES
        }
        cls.train = tiny.write_rows(
            cls.root / "train.jsonl", tiny.make_rows(30, "train")
        )
        cls.select = tiny.write_rows(
            cls.root / "select.jsonl", tiny.make_rows(36, "select")
        )
        cls.cal = tiny.write_rows(cls.root / "cal.jsonl", tiny.make_rows(6, "cal"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TeacherJoinTest(TinyModelCase):
    def write_teacher(self, name, records):
        path = self.root / name
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        return path

    @staticmethod
    def ramp(count):
        return [(1.0 + k) / sum(1.0 + j for j in range(count)) for k in range(count)]

    def teacher_records(self, rows):
        return [
            {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "teacher_probs": {
                    o["key"]: p
                    for o, p in zip(row["options"], self.ramp(len(row["options"])))
                },
            }
            for row in rows
        ]

    def test_partial_join_attaches_normalized_probabilities(self):
        rows = load_partition(self.train, "train")
        records = self.teacher_records(rows[:5])
        records[1]["input_sha256"] = "0" * 64
        records.append({**records[0], "id": "absent"})
        teacher = load_teacher(
            self.write_teacher("partial.jsonl", records), rows, partial=True
        )
        self.assertEqual(
            sorted(teacher), sorted(r["id"] for r in rows[:5] if r is not rows[1])
        )
        with self.assertRaises(ValueError):
            load_teacher(self.root / "partial.jsonl", rows)
        model_path = self.sources["qwen3"]
        _, tokenizer = train_ff.build_model(model_path, "rev", 16, 1)
        item = encode(rows[0], tokenizer, 512)
        attach_teacher_probs(item, teacher[rows[0]["id"]])
        for got, want in zip(item["teacher_probs"], self.ramp(len(rows[0]["options"]))):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(len(item["teacher_probs"]), len(rows[0]["options"]))

    def test_onestep_with_teacher_then_reload(self):
        rows = load_partition(self.train, "train")
        teacher = self.write_teacher("teacher.jsonl", self.teacher_records(rows[::3]))
        output = self.root / "onestep"
        argv = [
            "--model-path",
            str(self.sources["qwen3"]),
            "--revision",
            "rev",
            "--train",
            str(self.train),
            "--select",
            str(self.select),
            "--cal",
            str(self.cal),
            "--arm",
            "T",
            "--seed",
            "11",
            "--teacher",
            str(teacher),
            "--teacher-partial",
            "--teacher-kl-weight",
            "0.5",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--update-rows",
            "8",
            "--max-batch-tokens",
            "512",
            "--max-batch-rows",
            "4",
            "--head-dim",
            "16",
            "--max-length",
            "512",
        ]
        train_ff.main([*argv, "--onestep", "--output", str(output)])
        events = [json.loads(line) for line in (output / "train-metrics.jsonl").open()]
        update = events[0]
        self.assertEqual(update["rows"], 8)
        self.assertGreater(update["teacher_rows"], 0)
        self.assertGreater(update["kl"], 0)
        self.assertTrue(math.isfinite(update["grad_norm_preclip"]))
        self.assertEqual(update["lr_backbone"], 1e-5 * learning_factor(0, 4, 0.1))
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["contract"]["teacher_rows"], 10)
        self.assertEqual(
            len((output / "onestep.select32.jsonl").read_text().splitlines()), 32
        )
        for name in ("BEST.json", "COMPLETE.json", "LATEST.json"):
            self.assertTrue((output / name).is_file(), name)
        self.assertFalse((output / ".select-shards").exists())
        with self.assertRaises(SystemExit) as done:
            reload_check.main(
                [
                    "--run-dir",
                    str(output),
                    "--select",
                    str(self.select),
                    "--max-length",
                    "512",
                    "--output",
                    str(self.root / "reload.json"),
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(done.exception.code, 0)
        receipt = json.loads((self.root / "reload.json").read_text())
        self.assertEqual((receipt["argmax_changes"], receipt["rows"]), (0, 32))
        self.assertEqual(receipt["checkpoint_format"], "full")

        stray = self.write_teacher(
            "stray.jsonl", [{**self.teacher_records(rows[:1])[0], "id": "x"}]
        )
        with self.assertRaises(ValueError):
            train_ff.main(
                [str(stray) if arg == str(teacher) else arg for arg in argv]
                + ["--max-updates", "1", "--output", str(self.root / "stray")]
            )


class HeadInitTest(TinyModelCase):
    def reference_head(self, arch: str, seed: int) -> dict:
        captured = {}
        original = DecisionModel.from_base

        class Stop(Exception):
            pass

        def capture(*args, **kwargs):
            model, _ = original(*args, **kwargs)
            captured.update({k: v.clone() for k, v in model.head.state_dict().items()})
            raise Stop

        argv = [
            "train",
            "--model-path",
            str(self.sources[arch]),
            "--base-revision",
            "rev",
            "--init-kind",
            "posttrained",
            "--train",
            str(self.train),
            "--select",
            str(self.select),
            "--cal",
            str(self.cal),
            "--output",
            str(self.root / f"ref-{arch}-{seed}"),
            "--head-dim",
            "16",
            "--seed",
            str(seed),
            "--train-mode",
            "full",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch(
            "torch.cuda.is_available", return_value=True
        ), mock.patch(
            "torch.cuda.is_bf16_supported", return_value=True
        ), mock.patch.object(
            DecisionModel, "from_base", capture
        ):
            with self.assertRaises(Stop):
                reference_train.main()
        return captured

    def test_initial_head_is_bitwise_identical(self):
        for arch in ARCHES:
            for seed in (20260926, 20260928):
                with self.subTest(arch=arch, seed=seed):
                    expected = self.reference_head(arch, seed)
                    torch.manual_seed(0)
                    random.seed(0)
                    model, _ = train_ff.build_model(self.sources[arch], "rev", 16, seed)
                    actual = model.head.state_dict()
                    self.assertEqual(sorted(actual), sorted(expected))
                    for name, tensor in expected.items():
                        self.assertTrue(torch.equal(actual[name], tensor), name)
        other, _ = train_ff.build_model(self.sources["qwen3"], "rev", 16, 1)
        self.assertNotEqual(
            train_ff.state_sha256(other.head),
            train_ff.state_sha256(
                train_ff.build_model(self.sources["qwen3"], "rev", 16, 2)[0].head
            ),
        )


class SaveReloadTest(TinyModelCase):
    def test_save_full_matches_model_save_and_round_trips(self):
        for arch in ARCHES:
            with self.subTest(arch=arch):
                model, tokenizer = train_ff.build_model(
                    self.sources[arch], "rev", 16, 5
                )
                model = train_ff.prepare_model(model, train_ff.Group(), "fp32", False)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(torch.randn_like(parameter) * 1e-3)
                model.metadata.update(
                    {"checkpoint_format": "full", "training_mode": "full"}
                )
                direct, gathered = (
                    self.root / f"direct-{arch}",
                    self.root / f"gathered-{arch}",
                )
                model.save(direct, tokenizer)
                train_ff.save_full(
                    model,
                    tokenizer,
                    train_ff.gather_state(model, train_ff.Group()),
                    gathered,
                )
                files = sorted(
                    p.relative_to(direct) for p in direct.rglob("*") if p.is_file()
                )
                self.assertEqual(
                    files,
                    sorted(
                        p.relative_to(gathered)
                        for p in gathered.rglob("*")
                        if p.is_file()
                    ),
                )
                for name in files:
                    self.assertEqual(
                        (direct / name).read_bytes(),
                        (gathered / name).read_bytes(),
                        name,
                    )
                loaded, _ = DecisionModel.from_checkpoint(gathered)
                state, reloaded = model.state_dict(), loaded.state_dict()
                self.assertEqual(sorted(state), sorted(reloaded))
                for name in state:
                    self.assertTrue(
                        torch.equal(state[name], reloaded[name].float()), name
                    )


def unsplit_run_update(
    model,
    optimizer,
    group,
    *,
    micro_batches,
    rows,
    items,
    dummy,
    pad_id,
    precision,
    brier_weight,
    kl_weight,
    step,
    horizon,
    warmup_ratio,
    clip,
):
    """``train_ff.run_update`` as it was before the accumulate / step split."""
    factor = learning_factor(step, horizon, warmup_ratio)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["peak_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(5, dtype=torch.float64, device=group.device)
    for batch_ids in micro_batches:
        subset = [items[index] for index in batch_ids] if batch_ids else [dummy]
        logits, batch, terms = train_ff.batch_terms(
            model,
            subset,
            pad_id=pad_id,
            device=group.device,
            precision=precision,
            brier_weight=brier_weight,
            kl_weight=kl_weight,
        )
        (terms["total"].sum() * (1.0 / rows if batch_ids else 0.0)).backward()
        if batch_ids:
            sums += torch.stack(
                [
                    *(
                        terms[name].detach().double().sum()
                        for name in ("total", "ce", "brier", "replay_kl")
                    ),
                    (logits.detach().argmax(-1) == batch["labels"]).sum().double(),
                ]
            )
    group.all_reduce(sums)
    norm = train_ff.scalar(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    totals = sums.tolist()
    if not all(math.isfinite(value) for value in totals) or not math.isfinite(norm):
        raise RuntimeError(f"Nonfinite loss or gradient norm at update {step + 1}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    return {
        "loss": totals[0] / rows,
        "ce": totals[1] / rows,
        "brier": totals[2] / rows,
        "kl": totals[3] / rows,
        "accuracy": totals[4] / rows,
        "grad_norm_preclip": norm,
        "lr_backbone": lrs["backbone"],
        "lr_head": lrs["head"],
    }


class LossNormalizationTest(TinyModelCase):
    def test_run_update_matches_unsplit_update(self):
        _, tokenizer = train_ff.build_model(self.sources["qwen3"], "rev", 16, 9)
        items = [
            encode(row, tokenizer, 512)
            for row in load_partition(self.train, "train")[:7]
        ]
        updates = [([[0, 1], None, [2, 3, 4]], 5), ([[5], None, [6]], 2)]
        runs = []
        for update in (unsplit_run_update, train_ff.run_update):
            model, _ = train_ff.build_model(self.sources["qwen3"], "rev", 16, 9)
            model = train_ff.prepare_model(model, train_ff.Group(), "fp32", True)
            optimizer = train_ff.make_optimizer(model, 1e-3, 1e-2, 0.01)
            model.train()
            results = [
                update(
                    model,
                    optimizer,
                    train_ff.Group(),
                    micro_batches=micro_batches,
                    rows=rows,
                    items=items,
                    dummy=items[0],
                    pad_id=tokenizer.pad_token_id,
                    precision="fp32",
                    brier_weight=0.5,
                    kl_weight=0.0,
                    step=step,
                    horizon=4,
                    warmup_ratio=0.25,
                    clip=0.05,
                )
                for step, (micro_batches, rows) in enumerate(updates)
            ]
            state = {k: v.clone() for k, v in model.state_dict().items()}
            moments = [
                (s["exp_avg"].clone(), s["exp_avg_sq"].clone())
                for s in optimizer.state.values()
            ]
            grads = [p.grad for p in model.parameters()]
            runs.append((results, state, moments, grads))
        (old, old_state, old_moments, old_grads), (new, state, moments, grads) = runs
        self.assertEqual(old, new)
        self.assertTrue(all(g is None for g in old_grads + grads))
        self.assertEqual(sorted(old_state), sorted(state))
        for name in state:
            self.assertTrue(torch.equal(old_state[name], state[name]), name)
        self.assertEqual(len(old_moments), len(moments))
        for (a, b), (c, d) in zip(old_moments, moments):
            self.assertTrue(torch.equal(a, c) and torch.equal(b, d))

    def run_once(self, micro_batches, rows, items, dummy, pad_id):
        model, _ = train_ff.build_model(self.sources["qwen3"], "rev", 16, 9)
        model = train_ff.prepare_model(model, train_ff.Group(), "fp32", False)
        optimizer = train_ff.make_optimizer(model, 1e-5, 1e-4, 0.01)
        return train_ff.run_update(
            model,
            optimizer,
            train_ff.Group(),
            micro_batches=micro_batches,
            rows=rows,
            items=items,
            dummy=dummy,
            pad_id=pad_id,
            precision="fp32",
            brier_weight=0.5,
            kl_weight=0.0,
            step=0,
            horizon=10,
            warmup_ratio=0.1,
            clip=1e9,
        )

    def test_sharded_sum_equals_single_batch_mean(self):
        _, tokenizer = train_ff.build_model(self.sources["qwen3"], "rev", 16, 9)
        items = [
            encode(row, tokenizer, 512)
            for row in load_partition(self.train, "train")[:9]
        ]
        lengths = [len(item["ids"]) for item in items]
        rows = list(range(9))
        dummy = items[0]
        whole = self.run_once([rows], 9, items, dummy, tokenizer.pad_token_id)
        schedule = train_ff.rank_schedule(
            rows, lengths, 3, max_tokens=10**6, max_rows=2
        )
        # Summing every rank's micro-batch gradients in one process is what the SUM reduction does.
        ranks = [batch for part in schedule for batch in part] + [None]
        sharded = self.run_once(ranks, 9, items, dummy, tokenizer.pad_token_id)
        for key in ("loss", "ce", "brier", "accuracy", "grad_norm_preclip"):
            self.assertLess(
                abs(whole[key] - sharded[key]), 1e-5 * max(1.0, abs(whole[key])), key
            )
        self.assertEqual(whole["lr_head"], 1e-4 * learning_factor(0, 10, 0.1))


# Triples the sharded gradient of the update whose rows are fewer than the ranks.
SCALED_DUMMY_UPDATE = """
import importlib

train_ff = importlib.import_module("v2.27b.m4b.train_ff")
fsdp_parity = importlib.import_module("v2.27b.m4b.fsdp_parity")
accumulate = train_ff.accumulate_update


def scaled(model, group, **kwargs):
    if group.sharded and kwargs["rows"] < group.world:
        kwargs["rows"] = kwargs["rows"] / 3
    return accumulate(model, group, **kwargs)


train_ff.accumulate_update = scaled
fsdp_parity.main()
"""


class GlooParityTest(unittest.TestCase):
    GRADIENT_GATES = (
        "gradient_tensors_within_1e-4",
        "gradient_below_floor_within_1e-6",
        "gradient_global_within_1e-5",
    )

    def run_parity(self, tmp: Path, target: list[str]) -> tuple[int, dict, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
        )
        env["OMP_NUM_THREADS"] = "1"
        done = subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                "3",
                *target,
                "--device",
                "cpu",
                "--arch",
                ARCHES[-1],
                "--output",
                str(tmp / "parity"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        log = done.stdout[-3000:] + done.stderr[-3000:]
        self.assertTrue((tmp / "parity" / "parity.json").is_file(), log)
        parity = json.loads((tmp / "parity" / "parity.json").read_text())
        return done.returncode, parity, log

    def test_three_rank_fsdp_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, parity, log = self.run_parity(
                Path(tmp), ["-m", "v2.27b.m4b.fsdp_parity"]
            )
            self.assertEqual(code, 0, log)
            self.assertEqual(parity["status"], "PASS")
            self.assertTrue(all(parity["gates"].values()))
            for gate in self.GRADIENT_GATES:
                self.assertIn(gate, parity["gates"])
            for gate in ("parameters_within_1e-4", "updates_within_1e-4"):
                self.assertNotIn(gate, parity["gates"])
                self.assertIn(gate, parity["report_only_after_adamw"])
            self.assertIn(
                "zero_init_parameter_relative", parity["report_only_after_adamw"]
            )
            self.assertIn(1, parity["dummy_micro_batches_per_rank"][-1])
            self.assertEqual(len(parity["gradient_parity"]), 2)
            for update in parity["gradient_parity"]:
                self.assertEqual(
                    update["tensors"], parity["against_reference"]["tensors"]
                )
                self.assertGreater(update["reference_norm"], 0)
                self.assertLessEqual(update["global_relative"], 1e-5)
                self.assertLessEqual(update["tensor_relative_max"], 1e-4)
                self.assertTrue(update["worst_tensors"])

    def test_scaled_sharded_gradient_fails_the_gradient_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "scaled_dummy_update.py"
            script.write_text(SCALED_DUMMY_UPDATE, encoding="utf-8")
            code, parity, log = self.run_parity(Path(tmp), [str(script)])
            self.assertEqual(code, 1, log)
            self.assertEqual(parity["status"], "FAIL")
            first, second = parity["gradient_parity"]
            self.assertTrue(all(first["passed"].values()))
            self.assertFalse(second["passed"]["tensors"])
            self.assertFalse(second["passed"]["global"])
            self.assertAlmostEqual(second["global_relative"], 2.0, places=4)
            self.assertFalse(parity["gates"]["gradient_tensors_within_1e-4"])
            self.assertFalse(parity["gates"]["gradient_global_within_1e-5"])
            self.assertTrue(parity["gates"]["loss_within_1e-5"])


if __name__ == "__main__":
    unittest.main()
