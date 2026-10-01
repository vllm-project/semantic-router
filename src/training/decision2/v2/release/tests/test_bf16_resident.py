"""BF16-resident Linear weights in the release runtime (``qwen.keep_linear_bf16``).

Needs torch; the model tests also need Transformers (Qwen3; Qwen3.5 where the
installed version has it) and PEFT, and skip otherwise, so run them in the
pinned image. CPU BF16 autocast rounds Linear inputs and weights to BF16 the
way CUDA autocast does, so forwards before and after the conversion must be
bitwise equal. ``gpu_bf16_resident`` repeats the check end to end through real
packages on one GPU.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from v2.release import build

try:
    import torch
except ImportError:
    torch = None

RUNTIME = build.SOURCE_ROOT / "v2/release/runtime"


def staged_qwen(scratch: Path):
    """The runtime's ``qwen`` module, imported from a minimal staged package."""
    stage = scratch / "decision2"
    (stage / "_vendor/dev2model").mkdir(parents=True)
    for name in ("__init__.py", "api.py", "qwen.py"):
        shutil.copyfile(RUNTIME / name, stage / name)
    for name in ("_vendor/__init__.py", "_vendor/dev2model/__init__.py"):
        (stage / name).write_text("", encoding="utf-8")
    shutil.copyfile(
        build.SOURCE_ROOT / "training/model/data.py",
        stage / "_vendor/dev2model/data.py",
    )
    for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
        del sys.modules[name]
    sys.path.insert(0, str(scratch))
    try:
        return importlib.import_module("decision2.qwen")
    finally:
        sys.path.remove(str(scratch))


def round_linear_weights(module) -> None:
    """Store every Linear weight as BF16 would, as the released BF16 packages do."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.copy_(layer.weight.to(torch.bfloat16).float())


def dtypes(module) -> dict[str, torch.dtype]:
    return {name: p.dtype for name, p in module.named_parameters()}


@unittest.skipIf(torch is None, "needs torch")
class KeepLinearBf16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.qwen = staged_qwen(Path(cls.scratch.name))

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def test_only_exact_linear_tensors_become_bf16(self):
        torch.manual_seed(1)
        module = torch.nn.Module()
        module.embed = torch.nn.Embedding(16, 8)
        module.tied = torch.nn.Linear(8, 16, bias=False)
        module.exact = torch.nn.Linear(8, 8)
        module.inexact = torch.nn.Linear(8, 8, bias=False)
        module.conv = torch.nn.Conv1d(8, 8, 4, groups=8)
        module.norm = torch.nn.LayerNorm(8)
        module.A_log = torch.nn.Parameter(torch.randn(4))
        module.dt_bias = torch.nn.Parameter(torch.randn(4))
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(parameter.to(torch.bfloat16).float())
            module.inexact.weight.add_(1e-6)
        module.tied.weight = module.embed.weight
        before = {k: v.detach().clone() for k, v in module.state_dict().items()}
        counts = self.qwen.keep_linear_bf16(module, torch)
        self.assertEqual(counts, {"linear_bf16": 1, "linear_fp32": 2})
        bf16 = {
            name for name, dtype in dtypes(module).items() if dtype == torch.bfloat16
        }
        self.assertEqual(bf16, {"exact.weight", "exact.bias"})
        self.assertIs(module.tied.weight, module.embed.weight)
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value.float(), before[name]), name)

    def test_already_bf16_and_repeat_calls_are_stable(self):
        layer = torch.nn.Linear(4, 4, bias=False)
        round_linear_weights(layer)
        first = self.qwen.keep_linear_bf16(layer, torch)
        second = self.qwen.keep_linear_bf16(layer, torch)
        self.assertEqual(first, second)
        self.assertEqual(first, {"linear_bf16": 1, "linear_fp32": 0})


def tiny_decision_model(kind: str):
    from training.model.decision_model import CandidateHead, DecisionModel

    if kind == "qwen3":
        from transformers import Qwen3Config, Qwen3Model

        config = Qwen3Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
        )
        backbone = Qwen3Model(config)
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        config = Qwen3_5TextConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=512,
        )
        backbone = Qwen3_5TextModel(config)
    backbone.config.use_cache = False
    model = DecisionModel(backbone, CandidateHead(64, 16), {"head_variant": "shared"})
    round_linear_weights(model.backbone)
    return model.float().eval()


def batch():
    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 512, (2, 24), generator=generator)
    mask = torch.ones_like(ids)
    mask[1, 18:] = 0
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "candidate_positions": torch.tensor([[5, 9, 13], [4, 8, 0]]),
        "candidate_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "query_positions": torch.tensor([20, 17]),
    }


def forward(model) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        return model(**batch())


def has_qwen3_5() -> bool:
    try:
        importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return False
    return True


@unittest.skipIf(torch is None, "needs torch")
class BackboneParityTest(unittest.TestCase):
    """Bitwise-equal outputs under BF16 autocast before and after the conversion."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.qwen = staged_qwen(Path(cls.scratch.name))

    @classmethod
    def tearDownClass(cls):
        for name in [n for n in sys.modules if n.split(".")[0] == "decision2"]:
            del sys.modules[name]
        cls.scratch.cleanup()

    def check(self, model, linear: int, fp32_linear: int = 0) -> dict:
        torch.manual_seed(3)
        expected = forward(model)
        counts = self.qwen.keep_linear_bf16(model.backbone, torch)
        self.assertEqual(counts, {"linear_bf16": linear, "linear_fp32": fp32_linear})
        self.assertTrue(torch.equal(forward(model), expected))
        kinds = dtypes(model)
        for name, dtype in kinds.items():
            if name.startswith("head."):
                self.assertEqual(dtype, torch.float32, name)
        self.assertEqual(
            model.backbone.get_input_embeddings().weight.dtype, torch.float32
        )
        return kinds

    def test_qwen3_dense(self):
        try:
            model = tiny_decision_model("qwen3")
        except ImportError:
            self.skipTest("needs Transformers with Qwen3")
        kinds = self.check(model, linear=14)
        self.assertTrue(
            all(d == torch.float32 for n, d in kinds.items() if "norm" in n)
        )

    @unittest.skipUnless(has_qwen3_5(), "needs Transformers with Qwen3.5")
    def test_qwen3_5_hybrid_keeps_gated_delta_tensors_fp32(self):
        model = tiny_decision_model("qwen3_5")
        linear = sum(isinstance(m, torch.nn.Linear) for m in model.backbone.modules())
        kinds = self.check(model, linear=linear)
        fp32 = [
            n
            for n, d in kinds.items()
            if n.startswith("backbone.") and d == torch.float32
        ]
        for marker in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
            self.assertTrue(any(marker in n for n in fp32), marker)

    def test_lora_adapter_stays_unmerged(self):
        try:
            importlib.import_module("peft")
            model = tiny_decision_model("qwen3")
        except ImportError:
            self.skipTest("needs Transformers with Qwen3 and PEFT")
        from peft import LoraConfig, get_peft_model

        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        model.backbone = get_peft_model(
            model.backbone, LoraConfig(r=4, lora_alpha=8, target_modules=targets)
        )
        with torch.no_grad():
            for name, parameter in model.backbone.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.05)
        model.eval()
        lora = 2 * len(targets) * 2
        kinds = self.check(model, linear=14, fp32_linear=lora)
        layers = [m for m in model.backbone.modules() if hasattr(m, "merged")]
        self.assertTrue(layers)
        self.assertFalse(any(m.merged for m in layers))
        for name, dtype in kinds.items():
            if "lora_" in name:
                self.assertEqual(dtype, torch.float32, name)
            elif "base_layer" in name:
                self.assertEqual(dtype, torch.bfloat16, name)


if __name__ == "__main__":
    unittest.main()
