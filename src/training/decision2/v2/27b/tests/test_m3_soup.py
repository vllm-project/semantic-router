import importlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file

    lora_soup = importlib.import_module("v2.27b.lora_soup")
except ImportError:
    torch = None

from training.model.lora import verify_adapter_config
from training.model.source import source_fingerprint

TARGETS = {
    "layers.0.mlp.gate_proj": [12, 20],
    "layers.0.self_attn.q_proj": [12, 8],
    "layers.1.linear_attn.out_proj": [6, 12],
}
HEAD_SHAPES = {"key.weight": [4, 12], "query_norm.bias": [12], "scalar.weight": [1, 4]}


def make_source(root: Path, marker: bytes = b"base") -> Path:
    source = root / "base"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "qwen3_5"}\n')
    save_file(
        {"w": torch.zeros(2)},
        str(source / "model.safetensors"),
        metadata={"m": marker.decode()},
    )
    return source


def make_member(
    root: Path, name: str, source: Path, seed: int, *, rank=4, alpha=8, targets=TARGETS
) -> Path:
    path = root / name
    (path / "adapter").mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for module, (inputs, outputs) in targets.items():
        stem = f"base_model.model.{module}"
        tensors[f"{stem}.lora_A.weight"] = torch.randn(
            rank, inputs, generator=generator
        )
        tensors[f"{stem}.lora_B.weight"] = (
            torch.randn(outputs, rank, generator=generator) * 0.1
        )
    save_file(
        tensors,
        str(path / "adapter/adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": sorted({m.rsplit(".", 1)[-1] for m in targets}, reverse=True),
        "use_rslora": False,
        "use_dora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "base_model_name_or_path": None,
    }
    (path / "adapter/adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (path / "adapter/README.md").write_text("# adapter\n")
    head = {
        key: torch.randn(*shape, generator=generator)
        for key, shape in HEAD_SHAPES.items()
    }
    save_file(head, str(path / "decision_head.safetensors"))
    (path / "tokenizer.json").write_text('{"vocab": 1}\n')
    (path / "checkpoint.json").write_text(json.dumps({"step": seed}))
    (path / "trainer_state.pt").write_bytes(b"optimizer")
    meta = {
        "architecture": "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp",
        "prompt_version": "decision2-segmented-options-global-query-v1",
        "head_dim": 4,
        "checkpoint_format": "peft-lora/1",
        "training_mode": "lora",
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": 0.05,
            "target_modules": list(targets),
            "target_dimensions": {k: list(v) for k, v in targets.items()},
            "source_kind": "posttrained",
            "source_fingerprint": source_fingerprint(source),
            "base_revision": "1" * 40,
            "peft_version": "0.21.0",
        },
    }
    (path / "decision_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def delta(path: Path, module: str) -> torch.Tensor:
    config = json.loads((path / "adapter/adapter_config.json").read_text())
    tensors = load_file(str(path / "adapter/adapter_model.safetensors"))
    stem = f"base_model.model.{module}"
    b, a = (
        tensors[f"{stem}.lora_B.weight"].double(),
        tensors[f"{stem}.lora_A.weight"].double(),
    )
    return b @ a * (config["lora_alpha"] / config["r"])


@unittest.skipIf(torch is None, "torch and safetensors are required")
class LoraSoupTest(unittest.TestCase):
    def setUp(self):
        self._scratch = tempfile.TemporaryDirectory()
        self.root = Path(self._scratch.name)
        self.source = make_source(self.root)

    def tearDown(self):
        self._scratch.cleanup()

    def members(self, count):
        return [
            make_member(self.root, f"s{i}", self.source, 100 + i) for i in range(count)
        ]

    def test_update_is_the_mean_for_two_and_three_members(self):
        seeds = self.members(3)
        for count in (2, 3):
            members = seeds[:count]
            output = self.root / f"soup{count}"
            manifest = lora_soup.build(members, self.source, output)
            for module in TARGETS:
                mean = sum(delta(m, module) for m in members) / count
                diff = (delta(output, module) - mean).abs().max().item()
                self.assertLessEqual(diff, 1e-6 * mean.abs().max().item())
            meta = json.loads((output / "decision_config.json").read_text())
            config = json.loads((output / "adapter/adapter_config.json").read_text())
            self.assertEqual(
                (meta["lora"]["rank"], meta["lora"]["alpha"]), (4 * count, 8 * count)
            )
            self.assertEqual(
                (config["r"], config["lora_alpha"], config["lora_dropout"]),
                (4 * count, 8 * count, 0.05),
            )
            verify_adapter_config(output / "adapter", meta["lora"])
            self.assertEqual(manifest["verification"]["projections"], len(TARGETS))
            self.assertLessEqual(manifest["verification"]["max_relative_diff"], 1e-6)
            self.assertEqual(len(manifest["members"]), count)
            self.assertEqual(len(meta["soup"]["members"]), count)
            self.assertFalse((output / "trainer_state.pt").exists())
            self.assertFalse((output / "checkpoint.json").exists())
            self.assertFalse((output / "calibration.json").exists())
            self.assertFalse(output.with_name(output.name + ".pending").exists())

    def test_head_is_the_uniform_mean(self):
        members = self.members(3)
        output = self.root / "soup"
        lora_soup.build(members, self.source, output)
        head = load_file(str(output / "decision_head.safetensors"))
        heads = [load_file(str(m / "decision_head.safetensors")) for m in members]
        self.assertEqual(set(head), set(HEAD_SHAPES))
        for key in head:
            mean = sum(h[key].double() for h in heads) / 3
            self.assertLess((head[key].double() - mean).abs().max().item(), 1e-6)
            self.assertEqual(head[key].dtype, torch.float32)

    def test_manifest_identity_matches_the_fingerprint(self):
        from training.model.infer import checkpoint_fingerprint

        output = self.root / "soup"
        manifest = lora_soup.build(self.members(2), self.source, output)
        identity = checkpoint_fingerprint(output, self.source)
        self.assertEqual(manifest["output"]["model_sha256"], identity["model_sha256"])
        on_disk = json.loads((output / "soup_manifest.json").read_text())
        self.assertEqual(on_disk["output"]["model_sha256"], identity["model_sha256"])
        self.assertIn(
            "adapter/adapter_model.safetensors", on_disk["output"]["files_sha256"]
        )
        self.assertEqual(on_disk["tool_version"], lora_soup.TOOL_VERSION)

    def test_output_is_byte_identical_for_the_same_inputs(self):
        members = self.members(2)
        first, second = self.root / "a", self.root / "b"
        lora_soup.build(members, self.source, first)
        lora_soup.build(members, self.source, second)
        names = sorted(
            p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file()
        )
        self.assertEqual(
            names,
            sorted(
                p.relative_to(second).as_posix()
                for p in second.rglob("*")
                if p.is_file()
            ),
        )
        for name in names:
            self.assertEqual(
                (first / name).read_bytes(), (second / name).read_bytes(), name
            )

    def test_mismatched_members_are_refused(self):
        a = make_member(self.root, "a", self.source, 1)
        other_root = self.root / "other"
        other_root.mkdir()
        other_source = make_source(other_root, b"other")
        cases = {
            "rank": make_member(self.root, "rank", self.source, 2, rank=2, alpha=4),
            "base": make_member(self.root, "base", other_source, 3),
            "targets": make_member(
                self.root,
                "targets",
                self.source,
                4,
                targets={**TARGETS, "layers.0.mlp.gate_proj": [12, 24]},
            ),
        }
        fmt = make_member(self.root, "fmt", self.source, 5)
        meta = json.loads((fmt / "decision_config.json").read_text())
        meta["checkpoint_format"] = "full"
        (fmt / "decision_config.json").write_text(json.dumps(meta))
        cases["format"] = fmt
        head = make_member(self.root, "head", self.source, 6)
        save_file(
            {"key.weight": torch.zeros(4, 12)}, str(head / "decision_head.safetensors")
        )
        cases["head"] = head
        rslora = [
            make_member(self.root, f"rslora{i}", self.source, 7 + i) for i in range(2)
        ]
        for member in rslora:
            path = member / "adapter/adapter_config.json"
            config = json.loads(path.read_text())
            config["use_rslora"] = True
            path.write_text(json.dumps(config))
        cases["rslora"] = rslora[0]
        for label, member in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                lora_soup.build([a, member], self.source, self.root / f"out-{label}")
            self.assertFalse((self.root / f"out-{label}").exists())
        with self.assertRaisesRegex(ValueError, "use_rslora"):
            lora_soup.build(rslora, self.source, self.root / "x")
        with self.assertRaises(ValueError):
            lora_soup.build([a], self.source, self.root / "single")
        with self.assertRaises(ValueError):
            lora_soup.build([a, a], self.source, self.root / "dup")

    def test_serialized_target_order_does_not_matter(self):
        members = self.members(2)
        path = members[1] / "adapter/adapter_config.json"
        config = json.loads(path.read_text())
        config["target_modules"] = list(reversed(config["target_modules"]))
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        output = self.root / "soup"
        lora_soup.build(members, self.source, output)
        written = json.loads((output / "adapter/adapter_config.json").read_text())
        self.assertEqual(written["target_modules"], sorted(written["target_modules"]))

    def test_verification_rejects_a_corrupted_soup(self):
        members = self.members(2)
        output = self.root / "soup"
        lora_soup.build(members, self.source, output)
        weights = output / "adapter/adapter_model.safetensors"
        tensors = load_file(str(weights))
        key = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        tensors[key] = tensors[key] * 1.001
        save_file(tensors, str(weights), metadata={"format": "pt"})
        with self.assertRaises(ValueError):
            lora_soup.verify(members, output)

    def test_peft_forward_of_the_soup_is_the_mean_of_the_members(self):
        try:
            from peft import PeftModel
        except ImportError:
            self.skipTest("peft is not installed")
        from torch import nn

        class Block(nn.Module):
            def __init__(self, dims):
                super().__init__()
                for path, (inputs, outputs) in dims.items():
                    parent, leaf = path.rsplit(".", 1)
                    if not hasattr(self, parent):
                        self.add_module(parent, nn.Module())
                    getattr(self, parent).add_module(leaf, nn.Linear(inputs, outputs))

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                layers = {}
                for name, dims in TARGETS.items():
                    index, rest = name.split(".", 2)[1:]
                    layers.setdefault(int(index), {})[rest] = dims
                self.layers = nn.ModuleList(Block(layers[i]) for i in sorted(layers))

            def forward(self, inputs):
                modules = dict(self.named_modules())
                return [modules[name](inputs[name]) for name in TARGETS]

        generator = torch.Generator().manual_seed(9)
        inputs = {
            name: torch.randn(3, dims[0], generator=generator)
            for name, dims in TARGETS.items()
        }

        def outputs(path):
            model = PeftModel.from_pretrained(Tiny(), path / "adapter").eval()
            with torch.no_grad():
                return model(inputs)

        members = self.members(3)
        output = self.root / "soup"
        lora_soup.build(members, self.source, output)
        soup = outputs(output)
        each = [outputs(m) for m in members]
        for index, value in enumerate(soup):
            mean = sum(e[index].double() for e in each) / 3
            self.assertLess((value.double() - mean).abs().max().item(), 1e-5)

    def test_existing_output_is_refused(self):
        output = self.root / "soup"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            lora_soup.build(self.members(2), self.source, output)


if __name__ == "__main__":
    unittest.main()
