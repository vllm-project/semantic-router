"""CPU tests for decoder M10: the label-token readout, head-less soups and the ops/m10 data and probe tools."""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from training.model.decision_model import collate
from v2.dec import label_token as lt

HERE = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m10" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


md = load("m10_data")
mp = load("m10_probes")


class FakeTokenizer:
    """Words ``yes`` / ``no``, one or two capitals, single digits and other runs are one token each."""

    PATTERN = re.compile(r"yes|no|[A-Z]{2}(?![A-Z])|[A-Z]|\d|\n|[^\sA-Z\d]+| ")

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.pad_token_id = 0
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [
            self.vocab.setdefault(t, len(self.vocab) + 1)
            for t in self.PATTERN.findall(text)
        ]


def row(kind="choice", n=3, label=0):
    if kind == "noul":
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
    elif kind == "score":
        options = [{"key": str(i), "description": f"level {i}"} for i in range(n)]
    else:
        options = [{"key": f"k{i}", "description": f"opt {i}"} for i in range(n)]
    return {
        "id": f"r-{kind}-{n}",
        "state": "the state",
        "instructions": "pick one",
        "options": options,
        "label": label,
        "task_type": kind,
        "family": "fam",
    }


class TinyBackbone(nn.Module):
    """last_hidden_state = the input embedding, so the label logit is E[query token] . E[label]."""

    def __init__(self, vocab=1000, hidden=8):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, hidden)
        self.config = SimpleNamespace(
            tie_word_embeddings=True, model_type="qwen3_5_text"
        )

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class LabelTokenEncodeTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_alphabet(self):
        alphabet = lt.choice_alphabet(self.tok)
        self.assertEqual(len(alphabet), 255)
        self.assertEqual(alphabet[:3], ["A", "B", "C"])
        self.assertEqual(alphabet[26:28], ["AA", "AB"])

    def test_labels_and_positions(self):
        for kind, n, expected in (
            ("choice", 3, ["A", "B", "C"]),
            ("noul", 2, ["no", "yes"]),
            ("score", 5, ["0", "1", "2", "3", "4"]),
            ("choice", 30, lt.choice_alphabet(self.tok)[:30]),
        ):
            enc = lt.encode_label(row(kind, n), self.tok, 4096)
            self.assertEqual(enc["labels"], expected)
            ids = enc["ids"]
            self.assertEqual(
                [ids[p] for p in enc["candidate_positions"]], enc["label_token_ids"]
            )
            self.assertEqual(len(set(enc["label_token_ids"])), n)
            self.assertEqual(enc["query_position"], len(ids) - 1)
            self.assertEqual(ids[-1], self.tok.encode("\n")[0])
            collate([enc], 0)  # the shared collate accepts the label positions

    def test_score_above_ten_levels_uses_letters(self):
        enc = lt.encode_label(row("score", 12), self.tok, 4096)
        self.assertEqual(enc["labels"][:2], ["A", "B"])

    def test_too_many_options_and_length(self):
        with self.assertRaises(ValueError):
            lt.labels_for(row("choice", 256), lt.choice_alphabet(self.tok))
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            lt.encode_label(row("choice", 3), self.tok, 10)


class LabelTokenModelTest(unittest.TestCase):
    def test_logits_are_tied_lm_head_logits(self):
        tok = FakeTokenizer()
        items = [
            lt.encode_label(row("choice", 4), tok, 4096),
            lt.encode_label(row("noul"), tok, 4096),
        ]
        batch = collate(items, 0)
        backbone = TinyBackbone()
        model = lt.LabelTokenModel(backbone, {"readout": "label_token"})
        logits = model(**batch)
        emb = backbone.embed.weight
        for b, item in enumerate(items):
            query = emb[item["ids"][-1]]
            for i, label_id in enumerate(item["label_token_ids"]):
                self.assertAlmostEqual(
                    logits[b, i].detach().item(), float(query @ emb[label_id]), places=5
                )
        self.assertTrue(torch.isinf(logits[1, 2:]).all())
        self.assertEqual(list(model.head.parameters()), [])

    def test_wrap_requires_tied_qwen35(self):
        tok = FakeTokenizer()
        backbone = TinyBackbone()
        base = SimpleNamespace(
            backbone=backbone, metadata={"head_dim": 256, "base_revision": "r"}
        )
        wrapped = lt.LabelTokenModel.wrap(base, tok)
        self.assertEqual(wrapped.metadata["readout"], "label_token")
        self.assertNotIn("head_dim", wrapped.metadata)
        self.assertEqual(
            wrapped.metadata["label_token"]["choice_alphabet_sha256"],
            lt.alphabet_sha256(lt.choice_alphabet(tok)),
        )
        backbone.config.tie_word_embeddings = False
        with self.assertRaises(ValueError):
            lt.LabelTokenModel.wrap(base, tok)


class HeadlessSoupTest(unittest.TestCase):
    def test_soup_of_label_checkpoints(self):
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            members = []
            for i in range(2):
                m = tmp / f"m{i}"
                (m / "backbone").mkdir(parents=True)
                save_file(
                    {"w": torch.full((2, 2), float(i))},
                    str(m / "backbone" / "model.safetensors"),
                )
                (m / "decision_config.json").write_text(
                    json.dumps(
                        {
                            "readout": "label_token",
                            "architecture": lt.LABEL_ARCHITECTURE,
                            "prompt_version": lt.LABEL_PROMPT_VERSION,
                            "checkpoint_format": "full",
                            "max_options": 255,
                            "full_training_source": {"kind": "base"},
                        }
                    )
                )
                members.append(m)
            out = tmp / "soup"
            cmd = [sys.executable, "-m", "v2.dec.soup", "--output", str(out)]
            for m in members:
                cmd += ["--member", str(m)]
            subprocess.run(cmd, check=True, cwd=HERE.parents[1], capture_output=True)
            self.assertFalse((out / "decision_head.safetensors").exists())
            self.assertTrue(
                torch.equal(
                    load_file(str(out / "backbone" / "model.safetensors"))["w"],
                    torch.full((2, 2), 0.5),
                )
            )


class DataTest(unittest.TestCase):
    def test_quarantine_and_teacher_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [
                {
                    "id": f"r{i}",
                    "group_id": "q" if i == 1 else f"g{i}",
                    "input_sha256": f"h{i}",
                    "task_type": "choice",
                    "source": "s",
                }
                for i in range(4)
            ]
            base = tmp / "base.jsonl"
            base.write_text("".join(json.dumps(r) + "\n" for r in rows))
            teacher = tmp / "t.jsonl"
            teacher.write_text(
                "".join(
                    json.dumps({"id": f"r{i}", "input_sha256": f"h{i}"}) + "\n"
                    for i in (0, 1, 2)
                )
            )
            report = md.build(base, teacher, {"q"}, tmp / "out")
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["quarantine_rows_dropped"], 1)
            self.assertEqual(report["teacher_rows_covering_train"], 2)
            self.assertEqual(report["teacher_rows_outside_train"], 1)
            self.assertEqual(report["gold_only_rows"], 1)
            lines = (tmp / "out" / "train.jsonl").read_bytes().splitlines()
            self.assertEqual(lines[0], json.dumps(rows[0]).encode())


class ProbeTest(unittest.TestCase):
    def test_gsm8k_options(self):
        q = "Tom has 3 apples and buys 4 more boxes of 5. How many?"
        sol = "4*5=<<4*5=20>>20 apples. 3+20=<<3+20=23>>23\n#### 23"
        opts, gold = mp.gsm8k_options(q, sol, random.Random(1))
        self.assertEqual(len(set(opts)), 4)
        self.assertEqual(opts[gold], "23")
        self.assertIn("20", opts)

    def test_overlap_finalize_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            long_q = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
            cands = [
                {
                    "id": "a",
                    "probe": "mmlu",
                    "stem": long_q,
                    "state": long_q,
                    "instructions": "i",
                    "options": ["x", "y", "z", "w"],
                    "gold": 1,
                },
                {
                    "id": "b",
                    "probe": "mmlu",
                    "stem": "short stem here",
                    "state": "s",
                    "instructions": "i",
                    "options": ["x", "y"],
                    "gold": 0,
                },
                {
                    "id": "c",
                    "probe": "gsm8k",
                    "stem": "unrelated words entirely",
                    "state": "s",
                    "instructions": "i",
                    "options": ["1", "2", "3", "4"],
                    "gold": 3,
                },
            ]
            cpath = tmp / "c.jsonl"
            cpath.write_text("".join(json.dumps(c) + "\n" for c in cands))
            corpus = tmp / "suite.jsonl.gz"
            with gzip.open(corpus, "wt") as f:
                f.write(
                    json.dumps(
                        {
                            "id": "s1",
                            "state": {"q": "Prefix " + long_q.upper() + " suffix"},
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "id": "s2",
                            "questions": {"q": {"instructions": "a short stem here!"}},
                        }
                    )
                    + "\n"
                )
            out = tmp / "o.json"
            mp.overlap(
                SimpleNamespace(
                    candidates=str(cpath),
                    corpus=[str(corpus)],
                    kind="suite",
                    output=str(out),
                )
            )
            self.assertEqual(json.loads(out.read_text())["hit_ids"], ["a", "b"])
            prompts, gold, report = tmp / "p.jsonl", tmp / "g.jsonl", tmp / "r.json"
            mp.finalize(
                SimpleNamespace(
                    candidates=str(cpath),
                    exclude=[str(out)],
                    gsm8k_max=10,
                    prompts=str(prompts),
                    gold=str(gold),
                    report=str(report),
                )
            )
            p = [json.loads(line) for line in prompts.read_text().splitlines()]
            self.assertEqual([x["id"] for x in p], ["c"])
            self.assertEqual(set(p[0]), {"id", "state", "questions"})
            self.assertEqual(p[0]["questions"]["q"]["criteria"]["D"], "4")
            preds = tmp / "pred.jsonl"
            preds.write_text(
                json.dumps(
                    {"id": "c", "answers": {"q": {"type": "choice", "choice": "D"}}}
                )
                + "\n"
            )
            scored = tmp / "s.json"
            mp.score(
                SimpleNamespace(
                    gold=str(gold),
                    predictions=str(preds),
                    reference=str(preds),
                    output=str(scored),
                    draws=50,
                )
            )
            result = json.loads(scored.read_text())
            self.assertEqual(result["gsm8k"]["accuracy"], 1.0)
            self.assertEqual(result["gsm8k"]["delta"], 0.0)


class TrainerArgsTest(unittest.TestCase):
    def test_label_token_rejects_head_init_seed(self):
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; sys.argv=['x','--model-path','m','--train','t','--select','s','--cal','c',"
                "'--output','o','--arm','a','--readout','label_token','--head-init-seed','1'];"
                "from v2.dec import train_dec; train_dec.parse_args()"
            ),
        ]
        result = subprocess.run(
            cmd, cwd=HERE.parents[1], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("label_token", result.stderr)


if __name__ == "__main__":
    unittest.main()
