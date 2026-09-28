"""CPU contract tests for the decoder-track trainer, readouts and dev readout.

Torch-dependent cases skip when torch is absent (the local workstation); run
the whole file inside the pinned training image before a GPU launch.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2.dec.dev_readout import css_H, paired_bootstrap, typed_T

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ReadoutMathTest(unittest.TestCase):
    def test_typed_T_is_family_macro(self) -> None:
        records = [
            {"group": "g1", "family": "a", "correct": True},
            {"group": "g1", "family": "a", "correct": False},
            {"group": "g2", "family": "b", "correct": True},
        ]
        self.assertAlmostEqual(typed_T(records), 0.75)
        self.assertAlmostEqual(typed_T(records, {"g1": 2, "g2": 0}), 0.5)

    def test_css_H_is_task_median_macro_f1(self) -> None:
        tasks = {
            "t1": {"labels": ["x", "y"], "gold": ["x", "y"], "choice": ["x", "y"]},
            "t2": {"labels": ["x", "y"], "gold": ["x", "y"], "choice": ["x", "x"]},
            "t3": {"labels": ["x", "y"], "gold": ["x", "y"], "choice": [None, None]},
        }
        self.assertAlmostEqual(css_H(tasks), 1 / 3)

    def test_identical_arms_have_zero_interval(self) -> None:
        records = [
            {"group": f"g{i}", "family": "a" if i % 2 else "b", "correct": bool(i % 3)}
            for i in range(12)
        ]
        tasks = {
            "t": {
                "labels": ["x", "y"],
                "gold": ["x", "y"] * 5,
                "choice": ["x", "x"] * 5,
            }
        }
        result = paired_bootstrap((records, tasks), (records, tasks))["delta_b_minus_a"]
        for name in ("T", "H", "proxy"):
            self.assertEqual(result[name]["lower95"], 0.0)
            self.assertEqual(result[name]["upper95"], 0.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ResidualReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from torch import nn

        from training.model.decision_model import (
            ARCHITECTURE,
            CandidateHead,
            DecisionModel,
        )

        torch.manual_seed(0)
        hidden, layers = 16, 3

        class StubBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(
                    hidden_size=hidden, num_hidden_layers=layers
                )
                self.embed = nn.Embedding(64, hidden)
                self.blocks = nn.ModuleList(
                    nn.Linear(hidden, hidden) for _ in range(layers)
                )

            def forward(
                self,
                input_ids,
                attention_mask,
                use_cache=False,
                output_hidden_states=False,
            ):
                h = self.embed(input_ids)
                states = [h]
                for block in self.blocks:
                    h = torch.tanh(block(h))
                    states.append(h)
                return SimpleNamespace(
                    last_hidden_state=h,
                    hidden_states=tuple(states) if output_hidden_states else None,
                )

        self.torch = torch
        self.base = DecisionModel(
            StubBackbone(),
            CandidateHead(hidden, 8),
            {"architecture": ARCHITECTURE, "head_dim": 8},
        ).eval()
        self.batch = {
            "input_ids": torch.randint(0, 64, (3, 12)),
            "attention_mask": torch.ones(3, 12, dtype=torch.long),
            "candidate_positions": torch.tensor([[2, 4, 6], [3, 5, 0], [1, 7, 9]]),
            "candidate_mask": torch.tensor(
                [[True, True, True], [True, True, False], [True, True, True]]
            ),
            "query_positions": torch.tensor([11, 11, 11]),
            "task_type_ids": torch.tensor([2, 0, 1]),
            "score_level_indices": torch.tensor([[0, 1, 2], [0, 1, 0], [0, 1, 2]]),
        }

    def _logits(self, model):
        with self.torch.no_grad():
            return model(**self.batch)

    def test_zero_gate_is_exact_parity(self) -> None:
        from v2.dec.dec_model import DecModel

        reference = self._logits(self.base)
        for residuals in (
            (),
            ("ordinal_score",),
            ("layer_mix",),
            ("layer_mix", "ordinal_score"),
        ):
            model = DecModel.wrap(self.base, residuals).eval()
            self.assertTrue(self.torch.equal(self._logits(model), reference), residuals)

    def test_ordinal_residual_touches_only_score_rows(self) -> None:
        from v2.dec.dec_model import DecModel

        reference = self._logits(self.base)
        model = DecModel.wrap(self.base, ("ordinal_score",)).eval()
        with self.torch.no_grad():
            model.ordinal_score.gate.fill_(0.5)
        logits = self._logits(model)
        self.assertTrue(self.torch.equal(logits[1:], reference[1:]))
        delta = (logits[0] - reference[0]).tolist()
        self.assertFalse(all(v == 0 for v in delta))
        self.assertTrue(all(v <= 0 for v in delta))
        self.assertGreaterEqual(delta[1] - delta[0], delta[2] - delta[1])

    def test_layer_mix_changes_logits_and_metadata(self) -> None:
        from v2.dec.dec_model import DEC_HEAD_VARIANT, DecModel

        model = DecModel.wrap(self.base, ("layer_mix",)).eval()
        self.assertEqual(model.metadata["head_variant"], DEC_HEAD_VARIANT)
        self.assertEqual(model.metadata["dec_residual"]["layer_count"], 4)
        self.assertNotIn("head_variant", self.base.metadata)
        with self.torch.no_grad():
            model.layer_mix.gate.fill_(1.0)
        self.assertFalse(self.torch.equal(self._logits(model), self._logits(self.base)))

    def test_residual_state_roundtrip(self) -> None:
        from safetensors.torch import load_file, save_file

        from v2.dec.dec_model import DecModel

        model = DecModel.wrap(self.base, ("layer_mix", "ordinal_score"))
        with self.torch.no_grad():
            model.ordinal_score.gate.fill_(0.25)
            model.layer_mix.mix.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "residual.safetensors"
            save_file(model.residual_state(), str(path))
            state = load_file(str(path))
        fresh = DecModel.wrap(self.base, ("layer_mix", "ordinal_score"))
        for name in ("ordinal_score", "layer_mix"):
            subset = {
                k.removeprefix(f"{name}."): v
                for k, v in state.items()
                if k.startswith(f"{name}.")
            }
            getattr(fresh, name).load_state_dict(subset, strict=True)
        self.assertEqual(fresh.ordinal_score.gate.item(), 0.25)
        self.assertTrue(self.torch.equal(fresh.layer_mix.mix, model.layer_mix.mix))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TrainerHelperTest(unittest.TestCase):
    def test_inverse_type_balance_equalizes_totals(self) -> None:
        from v2.dec.train_dec import type_weights

        rows = [{"task_type": t} for t in ["choice"] * 6 + ["noul"] * 3 + ["score"]]
        weights = type_weights(rows, "inverse")
        totals = {
            t: weights[t] * n for t, n in (("choice", 6), ("noul", 3), ("score", 1))
        }
        self.assertEqual(len({round(v, 12) for v in totals.values()}), 1)
        self.assertAlmostEqual(sum(totals.values()), len(rows))
        self.assertEqual(
            type_weights(rows, "none"), {"choice": 1.0, "noul": 1.0, "score": 1.0}
        )

    def test_teacher_file_must_match_train_rows(self) -> None:
        from v2.dec.train_dec import attach_teacher_probs, load_teacher

        rows = [
            {"id": "a", "input_sha256": "h1", "options": [{"key": "x"}, {"key": "y"}]},
            {
                "id": "b",
                "input_sha256": "h2",
                "options": [{"key": "false"}, {"key": "true"}],
            },
        ]
        good = [
            {"id": "a", "input_sha256": "h1", "teacher_probs": {"x": 0.25, "y": 0.75}},
            {
                "id": "b",
                "input_sha256": "h2",
                "teacher_probs": {"false": 0.5, "true": 0.5},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in good))
            teacher = load_teacher(path, rows)
            item = {"keys": ["y", "x"]}
            attach_teacher_probs(item, teacher["a"])
            self.assertEqual(item["teacher_probs"], [0.75, 0.25])
            bad = [dict(good[0], input_sha256="other"), good[1]]
            path.write_text("".join(json.dumps(r) + "\n" for r in bad))
            with self.assertRaises(ValueError):
                load_teacher(path, rows)
            path.write_text(json.dumps(good[0]) + "\n")
            with self.assertRaises(ValueError):
                load_teacher(path, rows)


if __name__ == "__main__":
    unittest.main()
