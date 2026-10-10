from __future__ import annotations

import unittest

import torch

from d25.omni.model import checkpoint, inputs, tiny
from d25.omni.train import arms
from d25.omni.train.collator import MultimodalCollator
from d25.omni.train.loss import row_losses, step_loss
from d25.omni.train.model import OmniDecisionModel
from d25.omni.train.rows import measure, read_rows


def snapshot(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class ArmsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoProcessor

        cls.paths = tiny.fixtures()
        init = cls.paths["init_vega"]
        decision = checkpoint.read_decision_config(init)
        processor = inputs.setup_processor(AutoProcessor.from_pretrained(str(init)))
        cls.collator = MultimodalCollator(processor, decision["codes"])
        rows, _ = read_rows([cls.paths["data"] / "mm.jsonl"], "main")
        for row, (tokens, _) in zip(
            rows, measure(rows, processor, decision["codes"], "d25-vega", 16_384)
        ):
            row.tokens = tokens
        cls.image_batch = cls.collator([row for row in rows if row.images][:3])
        cls.text_batch = cls.collator([row for row in rows if not row.images][:3])

    def build(self, scales):
        model = OmniDecisionModel.from_checkpoint(self.paths["init_vega"])
        counts = arms.apply(model, scales)
        return model, counts

    def train(self, model, scales, batch, steps: int = 2) -> None:
        optimizer = torch.optim.AdamW(arms.param_groups(model, scales, 0.01), lr=1e-2)
        for _ in range(steps):
            logits = model(**batch.inputs)
            loss = step_loss(
                row_losses(logits, batch.targets, batch.counts),
                batch.weights,
                float(batch.weights.sum()),
                1,
            )
            loss.backward()
            for group in optimizer.param_groups:
                group["lr"] = 1e-2 * group["lr_scale"]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def changed(self, before, model, group: str) -> bool:
        return any(
            not torch.equal(before[n], p.detach())
            for n, p in model.named_parameters()
            if arms.group_of(n) == group
        )

    def test_group_assignment(self) -> None:
        model, counts = self.build(arms.ARMS["O-graft-frozen"].scales())
        groups = {arms.group_of(name) for name, _ in model.named_parameters()}
        self.assertEqual(groups, set(arms.GROUPS))
        self.assertEqual(counts["encoder"], 0)
        for name, parameter in model.named_parameters():
            frozen = name.startswith("backbone.visual.") and not name.startswith(
                "backbone.visual.merger."
            )
            self.assertEqual(parameter.requires_grad, not frozen, name)
        self.assertTrue(model.backbone.visual.merger.linear_fc1.weight.requires_grad)

    def test_param_groups(self) -> None:
        scales = arms.ARMS["O-graft-lowlr"].scales()
        model, _ = self.build(scales)
        groups = arms.param_groups(model, scales, 0.01)
        by_name = {g["name"]: g for g in groups}
        self.assertEqual(by_name["encoder"]["lr_scale"], 0.1)
        self.assertEqual(by_name["merger"]["lr_scale"], 1.0)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids), {id(p) for p in model.parameters() if p.requires_grad}
        )
        no_decay = arms.param_groups(model, scales, 0.01, decay_1d=False)
        self.assertTrue(
            any(
                g["name"].endswith("-no-decay") and g["weight_decay"] == 0
                for g in no_decay
            )
        )
        with self.assertRaises(ValueError):
            arms.apply(model, {"encoder": 1.0})

    def test_frozen_encoder_stays_bit_equal_and_merger_trains(self) -> None:
        scales = arms.ARMS["O-graft-frozen"].scales()
        model, _ = self.build(scales)
        before = snapshot(model)
        self.train(model, scales, self.image_batch)
        self.assertFalse(self.changed(before, model, "encoder"))
        self.assertTrue(self.changed(before, model, "merger"))
        self.assertTrue(self.changed(before, model, "language"))
        self.assertTrue(self.changed(before, model, "readout"))

    def test_low_lr_encoder_trains(self) -> None:
        scales = arms.ARMS["O-graft-lowlr"].scales()
        model, _ = self.build(scales)
        before = snapshot(model)
        self.train(model, scales, self.image_batch)
        self.assertTrue(self.changed(before, model, "encoder"))

    def test_fully_frozen_tower(self) -> None:
        scales = {**arms.ARMS["O-graft-frozen"].scales(), "merger": 0.0}
        model, counts = self.build(scales)
        self.assertEqual(counts["merger"], 0)
        before = snapshot(model)
        self.train(model, scales, self.image_batch)
        self.assertFalse(self.changed(before, model, "encoder"))
        self.assertFalse(self.changed(before, model, "merger"))

    def test_text_batch_runs_vision_sync_with_zero_gradient(self) -> None:
        scales = arms.ARMS["O-graft-lowlr"].scales()
        model, _ = self.build(scales)
        logits = model(**self.text_batch.inputs)
        loss = step_loss(
            row_losses(logits, self.text_batch.targets, self.text_batch.counts),
            self.text_batch.weights,
            3.0,
            1,
        )
        loss.backward()
        merger = model.backbone.visual.merger.linear_fc1.weight.grad
        self.assertIsNotNone(merger)
        self.assertEqual(float(merger.abs().sum()), 0.0)
        no_sync = OmniDecisionModel.from_checkpoint(
            self.paths["init_vega"], vision_sync=False
        )
        self.assertTrue(
            torch.allclose(
                no_sync(**self.text_batch.inputs), logits.detach(), atol=1e-6
            )
        )

    def test_arm_presets(self) -> None:
        self.assertEqual(arms.ARMS["O-fresh"].init, "stock")
        self.assertEqual(arms.ARMS["O-graft-frozen"].init, "vega")
        self.assertEqual(arms.ARMS["O-graft-lowlr"].scales()["encoder"], 0.1)


if __name__ == "__main__":
    unittest.main()
