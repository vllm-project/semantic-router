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
        for name in ("T", "H", "proxy", "H_mean", "proxy_mean_H"):
            self.assertEqual(result[name]["lower95"], 0.0)
            self.assertEqual(result[name]["upper95"], 0.0)


class TemplateSTest(unittest.TestCase):
    def test_whole_group_stratified_deterministic_selection(self) -> None:
        from v2.dec.build_template_s import group_rows, select_groups

        rows = []
        for i in range(40):
            kind = "choice" if i % 4 else "score"
            for j in range(1 + i % 2):
                rows.append(
                    {
                        "id": f"r{i}-{j}",
                        "group_id": f"g{i}",
                        "source": "s",
                        "task_type": kind,
                        "language": "en",
                    }
                )
        groups = group_rows(rows)
        tokens = {g: 10 * len(members) for g, members in groups.items()}
        chosen = select_groups(groups, tokens, 300, "seed")
        self.assertEqual(chosen, select_groups(groups, tokens, 300, "seed"))
        self.assertNotEqual(chosen, select_groups(groups, tokens, 300, "other"))
        self.assertEqual(len(set(chosen)), len(chosen))
        taken = sum(tokens[g] for g in chosen)
        self.assertGreaterEqual(taken, 300)
        self.assertLess(taken, 300 + 2 * max(tokens.values()))
        score = [g for g in chosen if groups[g][0]["task_type"] == "score"]
        self.assertTrue(0 < len(score) < len(chosen))


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

    def test_matrix_v1_schedule_and_selection(self) -> None:
        from v2.dec.train_dec import checkpoint_steps, selection_key

        self.assertEqual(
            sorted(checkpoint_steps(466, "even8", 32)),
            [58, 116, 175, 233, 291, 350, 408, 466],
        )
        self.assertEqual(checkpoint_steps(1, "even8", 32), {1})
        self.assertEqual(max(checkpoint_steps(466, "every", 32)), 466)
        tie = {"family_macro_accuracy": 0.8, "family_macro_brier": 0.1}
        worse_brier = {"family_macro_accuracy": 0.8, "family_macro_brier": 0.2}
        self.assertGreater(
            selection_key(worse_brier, 58, "matrix-v1"),
            selection_key(tie, 116, "matrix-v1"),
        )
        self.assertGreater(
            selection_key(tie, 116, "shared"), selection_key(worse_brier, 58, "shared")
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
            self.assertEqual(set(load_teacher(path, rows, partial=True)), {"a"})
            extra = [good[0], dict(good[1], input_sha256="renumbered")]
            extra.append({"id": "gone", "input_sha256": "h9", "teacher_probs": {}})
            path.write_text("".join(json.dumps(r) + "\n" for r in extra))
            self.assertEqual(set(load_teacher(path, rows, partial=True)), {"a"})
            with self.assertRaises(ValueError):
                load_teacher(path, rows)


class RuntimeCheckTest(unittest.TestCase):
    @staticmethod
    def _decorated(implementation, is_new_implementation):
        def wrapped(*args, **kwargs):
            if is_new_implementation:
                return implementation(*args, **kwargs)
            return None

        def outer(*args, **kwargs):
            return wrapped(*args, **kwargs)

        return outer

    def test_binding_follows_decorator_closures(self) -> None:
        from v2.dec.runtime_check import _binding

        def chunk_gated_delta_rule():
            return None

        name, is_new = _binding(self._decorated(chunk_gated_delta_rule, True))
        self.assertTrue(name.endswith("chunk_gated_delta_rule"))
        self.assertTrue(is_new)
        self.assertFalse(_binding(self._decorated(chunk_gated_delta_rule, False))[1])

    def test_violations_need_kernels_and_persisted_cache(self) -> None:
        from v2.dec.runtime_check import BINDINGS, violations

        good = {
            "kernel_bindings": {
                name: prefix + "ops.impl" for name, prefix in BINDINGS.items()
            },
            "triton_cache_dir": "/triton-cache",
            "triton_cache_autotuning": "1",
        }
        self.assertEqual(violations(good), [])
        reference = dict(
            good,
            kernel_bindings=dict(
                good["kernel_bindings"], torch_chunk_gated_delta_rule=None
            ),
        )
        self.assertEqual(len(violations(reference)), 1)
        self.assertEqual(
            len(
                violations(
                    dict(good, triton_cache_autotuning=None, triton_cache_dir=None)
                )
            ),
            2,
        )


def _score_row(i: int, count: int, gold: int, group: str | None = None) -> dict:
    return {
        "id": f"s{count}-{i}",
        "group_id": group or f"g{count}-{i}",
        "task_type": "score",
        "options": [{"key": str(k), "description": f"level {k}"} for k in range(count)],
        "label": gold,
        "source": "s",
    }


class ScoreOverfitTest(unittest.TestCase):
    def test_selection_covers_every_gold_level_once_per_group(self) -> None:
        from v2.dec.build_score_overfit import gold_level, select_rows

        rows = [
            _score_row(i, count, i % count, group=f"g{count}-{i // 2}")
            for count in (2, 5, 10)
            for i in range(60)
        ]
        chosen = select_rows(rows, 20, "seed")
        self.assertEqual(chosen, select_rows(rows, 20, "seed"))
        for count in (2, 5, 10):
            subset = [r for r in chosen if len(r["options"]) == count]
            self.assertEqual(len(subset), 20)
            self.assertEqual({gold_level(r) for r in subset}, set(range(count)))
            self.assertEqual(len({r["group_id"] for r in subset}), len(subset))

    def test_fit_report_verdicts(self) -> None:
        from v2.dec.score_fit_report import fit_report

        rows = [_score_row(i, 3, i % 3) for i in range(30)]
        perfect = {r["id"]: {"prediction_key": str(r["label"])} for r in rows}
        self.assertEqual(fit_report(rows, perfect)["status"], "PASS")
        collapsed = {r["id"]: {"prediction_key": "0"} for r in rows}
        report = fit_report(rows, collapsed)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["gold_levels_never_predicted"], {"L3": [1, 2]})


class BatchingTest(unittest.TestCase):
    def test_token_batches_respect_budget_and_cover_rows_once(self) -> None:
        from v2.dec.batching import padded, row_windows, token_batches

        lengths = [(i * 37) % 900 + 20 for i in range(500)]
        batches = token_batches(lengths, seed=3, epoch=0, max_tokens=4096, max_rows=16)
        self.assertEqual(
            batches,
            token_batches(lengths, seed=3, epoch=0, max_tokens=4096, max_rows=16),
        )
        self.assertNotEqual(
            batches,
            token_batches(lengths, seed=3, epoch=1, max_tokens=4096, max_rows=16),
        )
        self.assertEqual(sorted(i for b in batches for i in b), list(range(500)))
        for batch in batches:
            self.assertLessEqual(len(batch), 16)
            self.assertLessEqual(
                max(padded(lengths[i]) for i in batch) * len(batch), 4096
            )
        windows = row_windows(batches, 64)
        self.assertEqual(sum(len(b) for w in windows for b in w), 500)
        self.assertTrue(all(sum(len(b) for b in w) >= 64 for w in windows[:-1]))
        with self.assertRaises(ValueError):
            token_batches([5000], seed=0, epoch=0, max_tokens=4096, max_rows=4)


class MixtureTest(unittest.TestCase):
    def test_rule_7d_rekey_renumbers_and_rehashes(self) -> None:
        from training.model.data import INPUT_FIELDS, digest
        from v2.dec.build_mixture import rekey

        row = {
            "task_type": "choice",
            "state": "s",
            "instructions": "q",
            "options": [
                {"key": "result_2", "description": "a"},
                {"key": "result_0", "description": "b"},
                {"key": "result_1", "description": "c"},
            ],
            "label": 1,
            "audit_metadata": {},
        }
        row["input_sha256"] = digest({f: row[f] for f in INPUT_FIELDS})
        out = rekey(row)
        self.assertEqual(
            [o["key"] for o in out["options"]], ["result_0", "result_1", "result_2"]
        )
        self.assertEqual([o["description"] for o in out["options"]], ["a", "b", "c"])
        self.assertEqual(out["label"], 1)
        self.assertEqual(
            out["audit_metadata"]["dec_rule_7d"]["original_keys"],
            ["result_2", "result_0", "result_1"],
        )
        self.assertEqual(out["input_sha256"], digest({f: out[f] for f in INPUT_FIELDS}))
        self.assertNotEqual(out["input_sha256"], row["input_sha256"])
        self.assertIs(rekey(out), out)
        other = dict(row, options=[{"key": "x", "description": "a"}] * 2)
        self.assertIs(rekey(other), other)


def _train_row(row_id: str, group: str, text: str, meta: dict | None = None) -> dict:
    from training.model.data import INPUT_FIELDS, digest

    row = {
        "id": row_id,
        "state": text,
        "instructions": "pick",
        "options": [
            {"key": "result_0", "description": "a"},
            {"key": "result_1", "description": "b"},
        ],
        "label": 0,
        "task_type": "choice",
        "family": "fam",
        "group_id": group,
        "language": "en",
        "split": "train",
        "source": "src",
        "evaluation_role": "train",
        "render_template": "t",
        "audit_metadata": meta or {},
    }
    row["input_sha256"] = digest({f: row[f] for f in INPUT_FIELDS})
    return row


class RecipeMixtureTest(unittest.TestCase):
    """Recipe id joins, summed budget ratios and original-hash dedup."""

    def _build(self, root: Path, spec: dict, rows: dict[str, list[dict]]):
        from unittest import mock

        from training.model.data import file_sha256
        from v2.dec import build_mixture

        files = {}
        for name, content in rows.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in content))
            files[name] = {"sha256": file_sha256(path)}
        (root / "registry.json").write_text(json.dumps({"files": files}))
        registries = {
            "d": build_mixture.Registry(root, "registry.json", ""),
            "mx": build_mixture.Registry(root, "-", ""),
        }
        with mock.patch.object(
            build_mixture,
            "token_lengths",
            lambda kept, tokenizer, workers: [10] * len(kept),
        ):
            return build_mixture.build(spec, registries, root, 1)

    def test_recipe_ids_budget_list_and_original_hash_dedup(self) -> None:
        from training.model.data import file_sha256

        base = [_train_row(f"a{i}", f"g{i}", f"s{i}") for i in range(4)]
        renumbered = _train_row(
            "a4",
            "g4",
            "renumbered",
            {"option_key_renumbering": {"original_input_sha256": "old-hash"}},
        )
        replay_twin = _train_row(
            "r0", "h0", "twin", {"a7": {"original_input_sha256": "old-hash"}}
        )
        replay = [replay_twin] + [
            _train_row(f"r{i}", f"h{i}", f"t{i}") for i in range(1, 9)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = root / "recipe.ids.jsonl"
            ids.write_text(
                "".join(
                    json.dumps({"id": r["id"], "pool": "P"}) + "\n"
                    for r in base[1:] + [renumbered]
                )
                + json.dumps({"id": "zz", "pool": "Q"})
                + "\n"
            )
            spec_ids = {
                "file": "mx:recipe.ids.jsonl",
                "sha256": file_sha256(ids),
                "pool": "P",
            }
            spec = {
                "name": "t",
                "seed": "t",
                "dedupe_original_hashes": True,
                "components": [
                    {"name": "P", "files": ["d:p.jsonl"], "ids": spec_ids},
                    {
                        "name": "R",
                        "files": ["d:r.jsonl"],
                        "budget_ratio": {"of": ["P"], "ratio": 1.5},
                    },
                ],
            }
            mixture, manifest = self._build(
                root, spec, {"p.jsonl": base + [renumbered], "r.jsonl": replay}
            )
            self.assertEqual(manifest["components"]["P"]["in_recipe"], 4)
            self.assertEqual(manifest["components"]["R"]["duplicate_input"], 1)
            self.assertEqual(manifest["components"]["R"]["budget_tokens"], 60)
            self.assertEqual(manifest["components"]["R"]["tokens"], 60)
            self.assertNotIn("r0", {r["id"] for r in mixture})
            self.assertNotIn("a0", {r["id"] for r in mixture})

            spec["components"][0]["ids"] = dict(spec_ids, pool="Q")
            with self.assertRaises(ValueError):
                self._build(
                    root, spec, {"p.jsonl": base + [renumbered], "r.jsonl": replay}
                )
            spec["components"][0]["ids"] = dict(spec_ids, sha256="0" * 64)
            with self.assertRaises(ValueError):
                self._build(
                    root, spec, {"p.jsonl": base + [renumbered], "r.jsonl": replay}
                )

    def test_original_hash_dedup_is_opt_in(self) -> None:
        from v2.dec.build_mixture import row_hashes

        row = _train_row("x", "g", "s", {"a7": {"original_input_sha256": "h"}})
        self.assertEqual(row_hashes(row, False), {row["input_sha256"]})
        self.assertEqual(row_hashes(row, True), {row["input_sha256"], "h"})


class MergeLabelsTest(unittest.TestCase):
    def test_shards_must_cover_train_once_and_overrides_are_recorded(self) -> None:
        import subprocess
        import sys

        rows = [_train_row(f"a{i}", f"g{i}", f"s{i}") for i in range(4)]

        def label(row: dict, p: float) -> dict:
            return {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "teacher_probs": {"result_0": p, "result_1": 1 - p},
            }

        root_dir = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train = tmp_path / "train.jsonl"
            train.write_text("".join(json.dumps(r) + "\n" for r in rows))
            shard0 = tmp_path / "s0.jsonl"
            shard1 = tmp_path / "s1.jsonl"
            override = tmp_path / "o.jsonl"
            shard0.write_text(
                "".join(json.dumps(label(r, 0.9)) + "\n" for r in rows[0::2])
            )
            shard1.write_text(
                "".join(json.dumps(label(r, 0.8)) + "\n" for r in rows[1::2])
            )
            override.write_text(json.dumps(label(rows[1], 0.2)) + "\n")

            def run(*extra: str) -> subprocess.CompletedProcess:
                out = tmp_path / f"m{len(list(tmp_path.glob('m*.jsonl')))}.jsonl"
                return (
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "v2.dec.merge_labels",
                            "--train",
                            str(train),
                            *extra,
                            "--output",
                            str(out),
                        ],
                        cwd=root_dir,
                        capture_output=True,
                        text=True,
                    ),
                    out,
                )

            done, out = run(
                "--part",
                str(shard0),
                "--part",
                str(shard1),
                "--override",
                str(override),
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            merged = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([m["id"] for m in merged], [r["id"] for r in rows])
            self.assertAlmostEqual(merged[1]["teacher_probs"]["result_0"], 0.2)
            manifest = json.loads(
                out.with_name(out.name + ".manifest.json").read_text()
            )
            self.assertEqual(manifest["rows_by_origin"], {"part": 3, "override0": 1})
            foreign = _train_row("zz", "gz", "other mixture")
            wider = tmp_path / "wider.jsonl"
            wider.write_text(
                json.dumps(label(rows[2], 0.3))
                + "\n"
                + json.dumps(label(foreign, 0.5))
                + "\n"
            )
            strict, _ = run(
                "--part", str(shard0), "--part", str(shard1), "--override", str(wider)
            )
            self.assertNotEqual(strict.returncode, 0)
            subset, out = run(
                "--part",
                str(shard0),
                "--part",
                str(shard1),
                "--override",
                str(wider),
                "--override-subset",
            )
            self.assertEqual(subset.returncode, 0, subset.stderr)
            manifest = json.loads(
                out.with_name(out.name + ".manifest.json").read_text()
            )
            self.assertEqual(manifest["rows_by_origin"], {"part": 3, "override0": 1})
            stale = tmp_path / "stale.jsonl"
            stale.write_text(
                json.dumps(dict(label(rows[2], 0.3), input_sha256="0" * 64)) + "\n"
            )
            mismatch, _ = run(
                "--part",
                str(shard0),
                "--part",
                str(shard1),
                "--override",
                str(stale),
                "--override-subset",
            )
            self.assertNotEqual(mismatch.returncode, 0)
            missing, _ = run("--part", str(shard0))
            self.assertNotEqual(missing.returncode, 0)
            repeated, _ = run(
                "--part", str(shard0), "--part", str(shard0), "--part", str(shard1)
            )
            self.assertNotEqual(repeated.returncode, 0)


HAS_QWEN35 = HAS_TORCH and importlib.util.find_spec("transformers") is not None


def _torch_reference(function, depth: int = 0):
    """The reference PyTorch function behind a Transformers kernel wrapper."""
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None) or ()
    if code is not None and "torch_function" in code.co_freevars:
        return cells[code.co_freevars.index("torch_function")].cell_contents
    for cell in cells if depth < 4 else ():
        value = cell.cell_contents
        if callable(value):
            found = _torch_reference(value, depth + 1)
            if found is not None:
                return found
    return None


@unittest.skipUnless(HAS_QWEN35, "torch/transformers not installed")
class PaddingEquivalenceTest(unittest.TestCase):
    """Right padding must not change loss or gradients on a real hybrid backbone.

    On CPU the image's CUDA/Triton kernels cannot run, so the gated-delta and
    causal-conv1d calls use Transformers' reference implementations; the GPU
    probe ``check_padding`` covers the kernels themselves.
    """

    def setUp(self) -> None:
        from unittest import mock

        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            reference = _torch_reference(getattr(modeling_qwen3_5, name))
            if reference is not None:
                patcher = mock.patch.object(modeling_qwen3_5, name, reference)
                patcher.start()
                self.addCleanup(patcher.stop)
        try:
            torch.linalg.solve_triangular(
                torch.eye(2), torch.ones(2, 1), upper=False, unitriangular=True
            )
        except RuntimeError:
            # ROCm builds without CPU BLAS: use the reference's own
            # forward-substitution branch (plain matmuls, same math).
            patcher = mock.patch.object(
                modeling_qwen3_5, "is_torchdynamo_exporting", lambda: True
            )
            patcher.start()
            self.addCleanup(patcher.stop)

        from training.model.decision_model import (
            ARCHITECTURE,
            CandidateHead,
            DecisionModel,
        )

        torch.manual_seed(0)
        config = Qwen3_5TextConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=256,
        )
        backbone = Qwen3_5TextModel(config).float()
        self.torch = torch
        self.model = DecisionModel(
            backbone, CandidateHead(32, 8), {"architecture": ARCHITECTURE}
        ).train()
        lengths_and_counts = [(13, 3), (37, 2), (22, 4), (9, 2), (30, 3)]
        self.items = []
        for index, (length, count) in enumerate(lengths_and_counts):
            ids = torch.randint(1, 97, (length,)).tolist()
            ends = sorted(torch.randperm(length - 2)[:count].add(1).tolist())
            self.items.append(
                {
                    "id": f"r{index}",
                    "ids": ids,
                    "candidate_positions": ends,
                    "query_position": length - 1,
                    "label": index % count,
                    "keys": [str(k) for k in range(count)],
                    "score_level_indices": list(range(count)),
                    "task_type": "score",
                    "teacher_probs": None,
                }
            )

    def _run(self, mode: str, groups: list[list[int]]):
        from training.model.loss import per_example_loss
        from v2.dec.check_padding import batch_for

        self.model.zero_grad(set_to_none=True)
        losses = {}
        for positions in groups:
            subset = [self.items[i] for i in positions]
            batch = batch_for(subset, 0, mode, 11)
            logits = self.model(**batch)
            terms = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )
            terms["total"].sum().backward()
            for position, value in zip(positions, terms["total"].tolist()):
                losses[position] = value
        grads = self.torch.cat(
            [p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
        )
        return [losses[i] for i in range(len(self.items))], grads

    def test_padded_microbatches_match_unpadded_rows(self) -> None:
        from v2.dec.check_padding import mixed_groups

        singles = [[i] for i in range(len(self.items))]
        exact_loss, exact_grad = self._run("exact", singles)
        grouped = mixed_groups(
            list(range(len(self.items))), [len(i["ids"]) for i in self.items], 3
        )
        self.assertTrue(any(len(g) > 1 for g in grouped))
        for mode, groups in (
            ("single", singles),
            ("extra", singles),
            ("grouped", grouped),
        ):
            loss, grad = self._run(mode, groups)
            for a, b in zip(exact_loss, loss):
                self.assertAlmostEqual(a, b, places=5, msg=mode)
            relative = (grad - exact_grad).norm() / exact_grad.norm()
            self.assertLess(relative.item(), 1e-5, mode)

    def test_mixed_groups_pair_short_and_long_rows(self) -> None:
        from v2.dec.check_padding import mixed_groups

        groups = mixed_groups([0, 1, 2, 3, 4, 5], [5, 60, 10, 50, 20, 40], 2)
        self.assertEqual(groups, [[1, 0], [3, 2], [5, 4]])


if __name__ == "__main__":
    unittest.main()
