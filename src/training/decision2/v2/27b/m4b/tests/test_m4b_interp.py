import contextlib
import importlib
import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

interp = importlib.import_module("v2.27b.m4b.interp_full")

SHARDS = {
    "model-00001-of-00002.safetensors": {
        "embed_tokens.weight": [7, 5],
        "layers.0.mlp.gate_proj.weight": [6, 5],
    },
    "model-00002-of-00002.safetensors": {
        "layers.0.input_layernorm.weight": [5],
        "norm.weight": [5],
        "layers.0.linear_attn.A_log": [3],
    },
}
HEAD = {"key.weight": [4, 5], "scalar": []}
META = {
    "architecture": "decision2-qwen3_5-dynamic-option",
    "prompt_version": "p1",
    "head_dim": 4,
    "head_variant": "shared",
    "max_options": 8,
    "checkpoint_format": "full",
}


def tensor(shape, gen, *, zeros=False):
    value = torch.randn(shape, generator=gen, dtype=torch.float32)
    if zeros and value.numel():
        value.view(-1)[0] = -0.0
    return value


def make_full(
    root: Path, name: str, seed: int, *, shards=SHARDS, meta=None, config=None
):
    path = root / name
    (path / "backbone").mkdir(parents=True)
    gen = torch.Generator().manual_seed(seed)
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(
            {k: tensor(s, gen, zeros=True) for k, s in tensors.items()},
            str(path / "backbone" / shard),
            metadata={"format": "pt"},
        )
        weight_map.update({k: shard for k in tensors})
    (path / "backbone" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (path / "backbone" / "config.json").write_text(
        json.dumps(config or {"model_type": "qwen3_5_text", "hidden_size": 5})
    )
    save_file(
        {k: tensor(s, gen) for k, s in HEAD.items()},
        str(path / "decision_head.safetensors"),
    )
    (path / "tokenizer.json").write_text('{"tok": 1}')
    (path / "decision_config.json").write_text(
        json.dumps({**META, "initialization": f"seed-{seed}", **(meta or {})})
    )
    return path


def tensors_of(ckpt: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted((ckpt / "backbone").glob("*.safetensors")):
        out.update(load_file(str(shard)))
    out.update(
        {
            f"head:{k}": v
            for k, v in load_file(str(ckpt / "decision_head.safetensors")).items()
        }
    )
    return out


def bits(t: torch.Tensor) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def run(argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        interp.main(argv)
    return json.loads(out.getvalue().splitlines()[-1])


class InterpFullTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.s = make_full(self.root, "S", 1)
        self.b = make_full(self.root, "B", 2)

    def tearDown(self):
        self.tmp.cleanup()

    def interp(self, alpha: str, name: str) -> Path:
        run(["interp", "--s", str(self.s), "--b", str(self.b), "--alpha", alpha,
             "--output", str(self.root / name)])  # fmt: skip
        return self.root / name

    def test_endpoints_are_bitwise_inputs(self):
        for alpha, source in (("1", self.s), ("0", self.b)):
            out = self.interp(alpha, f"end-{alpha}")
            got, want = tensors_of(out), tensors_of(source)
            self.assertEqual(sorted(got), sorted(want))
            for k in want:
                self.assertEqual(bits(got[k]), bits(want[k]), k)
            self.assertEqual(
                sorted(p.name for p in (out / "backbone").iterdir()),
                sorted(p.name for p in (source / "backbone").iterdir()),
            )
            report = run(["verify", "--output", str(out)])
            self.assertEqual(
                report["single_term_bitwise_member"],
                str(source if alpha == "1" else self.b),
            )

    def test_midpoint_and_third_are_exact(self):
        for alpha in ("1/2", "1/3"):
            out = self.interp(alpha, f"mid-{alpha.replace('/', '_')}")
            a = Fraction(alpha)
            s, b, got = tensors_of(self.s), tensors_of(self.b), tensors_of(out)
            for k in s:
                expected = ((s[k].double() * a.numerator + b[k].double() * (a.denominator - a.numerator))
                            / a.denominator).float()  # fmt: skip
                self.assertEqual(bits(got[k]), bits(expected), k)
            meta = json.loads((out / "decision_config.json").read_text())
            self.assertEqual(
                meta["initialization"], "interpolation-of-full-checkpoints"
            )
            self.assertEqual(meta["combination"]["weights"], [alpha, str(1 - a)])
            self.assertEqual(run(["verify", "--output", str(out)])["tensors"], 7)

    def test_soup_is_the_float64_mean(self):
        c = make_full(self.root, "C", 3)
        run(["soup", "--member", str(self.s), "--member", str(self.b), "--member", str(c),
             "--output", str(self.root / "soup")])  # fmt: skip
        members = [tensors_of(p) for p in (self.s, self.b, c)]
        got = tensors_of(self.root / "soup")
        for k in got:
            mean = (
                members[0][k].double() + members[1][k].double() + members[2][k].double()
            ) / 3
            self.assertEqual(bits(got[k]), bits(mean.float()), k)
            peak = mean.abs().max().item() if mean.numel() else 0
            self.assertLessEqual(
                (got[k].double() - mean).abs().max().item(), 1e-6 * max(peak, 1e-30)
            )
        manifest = json.loads(
            (self.root / "soup" / "combination_manifest.json").read_text()
        )
        self.assertLessEqual(
            manifest["verification"]["max_relative_dev_from_float64"], 1e-6
        )
        with safe_open(
            str(self.root / "soup" / "backbone" / "model-00001-of-00002.safetensors"),
            "pt",
        ) as h:
            self.assertEqual(h.metadata(), {"format": "pt"})

    def test_chunked_rows_match_whole_tensors(self):
        old = interp.CHUNK_ELEMENTS
        interp.CHUNK_ELEMENTS = 7
        try:
            out = self.interp("2/3", "chunked")
        finally:
            interp.CHUNK_ELEMENTS = old
        s, b, got = tensors_of(self.s), tensors_of(self.b), tensors_of(out)
        for k in s:
            expected = ((s[k].double() * 2 + b[k].double()) / 3).float()
            self.assertEqual(bits(got[k]), bits(expected), k)

    def test_verify_detects_a_changed_output(self):
        out = self.interp("1/2", "mid")
        shard = out / "backbone" / "model-00002-of-00002.safetensors"
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            interp.verify(out)

    def test_refuses_mismatches(self):
        cases = {
            "shape": dict(shards={**SHARDS, "model-00002-of-00002.safetensors": {
                **SHARDS["model-00002-of-00002.safetensors"], "norm.weight": [6]}}),
            "name": dict(shards={**SHARDS, "model-00002-of-00002.safetensors": {
                "layers.0.input_layernorm.weight": [5], "final_norm.weight": [5],
                "layers.0.linear_attn.A_log": [3]}}),
            "config": dict(config={"model_type": "qwen3_5_text", "hidden_size": 6}),
            "meta": dict(meta={"head_dim": 8}),
        }  # fmt: skip
        for label, kwargs in cases.items():
            other = make_full(self.root, f"bad-{label}", 4, **kwargs)
            with self.subTest(label), self.assertRaises(ValueError):
                interp.build(
                    [self.s, other],
                    [Fraction(1, 2)] * 2,
                    self.root / f"o-{label}",
                    "soup",
                )
            self.assertFalse((self.root / f"o-{label}").exists())
        lora = make_full(
            self.root, "lora", 5, meta={"checkpoint_format": "peft-lora/1"}
        )
        with self.assertRaises(ValueError):
            interp.build(
                [self.s, lora], [Fraction(1, 2)] * 2, self.root / "o-lora", "soup"
            )
        with self.assertRaises(ValueError):
            interp.build(
                [self.s, self.s], [Fraction(1, 2)] * 2, self.root / "o-same", "soup"
            )
        with self.assertRaises(ValueError):
            interp.parse_alpha("3/2")
        out = self.interp("1/2", "once")
        with self.assertRaises(FileExistsError):
            interp.build([self.s, self.b], [Fraction(1, 2)] * 2, out, "interp")


def make_lora(
    root: Path, base: Path, merged_like: Path, scale_alpha=4, rank=2, *, perturb=0.0
):
    """A LoRA checkpoint and the matching merged full checkpoint (FP32 merge of base + s*BA)."""
    gen = torch.Generator().manual_seed(9)
    lora = Path(tempfile.mkdtemp(dir=root)) / "lora"
    (lora / "adapter").mkdir(parents=True)
    target = "layers.0.mlp.gate_proj"
    a = torch.randn(rank, 5, generator=gen)
    b = torch.randn(6, rank, generator=gen)
    save_file(
        {
            f"base_model.model.{target}.lora_A.weight": a,
            f"base_model.model.{target}.lora_B.weight": b,
        },
        str(lora / "adapter" / "adapter_model.safetensors"),
    )
    (lora / "adapter" / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": scale_alpha})
    )
    head = load_file(str(merged_like / "decision_head.safetensors"))
    save_file(head, str(lora / "decision_head.safetensors"))
    (lora / "decision_config.json").write_text(
        json.dumps(
            {
                **META,
                "checkpoint_format": "peft-lora/1",
                "lora": {"target_modules": [target]},
            }
        )
    )
    tensors = load_file(str(base / "model.safetensors"))
    merged = {}
    for name, value in tensors.items():
        local = name.removeprefix("model.language_model.")
        if local.startswith(("model.visual", "mtp.")) or name == "lm_head.weight":
            continue
        merged[local] = value.float()
    key = f"{target}.weight"
    merged[key] = (
        merged[key].double() + scale_alpha / rank * (b.double() @ a.double())
    ).float()
    merged[key] += perturb
    return lora, merged


class MergeCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        gen = torch.Generator().manual_seed(3)
        self.base = self.root / "base"
        self.base.mkdir()
        names = [k for shard in SHARDS.values() for k in shard.items()]
        save_file(
            {
                **{
                    f"model.language_model.{k}": torch.randn(
                        s, generator=gen
                    ).bfloat16()
                    for k, s in names
                },
                "model.visual.patch.weight": torch.zeros(2).bfloat16(),
                "lm_head.weight": torch.zeros(3, 5).bfloat16(),
            },
            str(self.base / "model.safetensors"),
        )
        self.template = make_full(self.root, "template", 1)

    def tearDown(self):
        self.tmp.cleanup()

    def merged(self, name, tensors):
        path = make_full(self.root, name, 1)
        shards = {}
        for shard, keys in SHARDS.items():
            shards[shard] = {k: tensors[k] for k in keys}
            save_file(shards[shard], str(path / "backbone" / shard))
        return path

    def check(self, perturb=0.0, head_change=False):
        lora, tensors = make_lora(self.root, self.base, self.template, perturb=perturb)
        merged = self.merged(f"m-{perturb}-{head_change}", tensors)
        if head_change:
            head = load_file(str(merged / "decision_head.safetensors"))
            head["scalar"] = head["scalar"] + 1
            save_file(head, str(merged / "decision_head.safetensors"))
        return interp.merge_check(merged, lora, self.base)

    def test_exact_merge_passes(self):
        from unittest import mock

        with mock.patch.object(
            interp, "checkpoint_fingerprint", return_value={"model_sha256": "x"}
        ):
            report = self.check()
        self.assertEqual(report["projections"], 1)
        self.assertEqual(report["other_backbone_tensors_equal_base_fp32"], 4)
        self.assertLessEqual(report["max_relative_diff"], 1e-6)
        self.assertEqual(report["loaded_parameters"], 35 + 30 + 5 + 5 + 3 + 20 + 1)

    def test_bad_merge_or_head_fails(self):
        from unittest import mock

        with mock.patch.object(
            interp, "checkpoint_fingerprint", return_value={"model_sha256": "x"}
        ):
            with self.assertRaises(ValueError):
                self.check(perturb=1e-3)
            with self.assertRaises(ValueError):
                self.check(head_change=True)


if __name__ == "__main__":
    unittest.main()
