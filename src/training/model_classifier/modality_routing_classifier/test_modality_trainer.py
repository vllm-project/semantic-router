"""Tests for the trainer's configuration, data preparation, loss and teacher handling."""

import json
import types

import modality_routing_fixed_split_trainer as trainer
import numpy as np
import pytest
import torch
from modality_losses import DistillationSettings, distillation_loss
from transformers import TrainingArguments


def write_split(path, counts):
    rows = [
        {
            "text": f"row {label} {i}",
            "label": label,
            "label_name": ["AR", "DIFFUSION", "BOTH"][label],
        }
        for label, n in counts.items()
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def make_config(tmp_path, **overrides):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_split(train, {0: 40, 1: 40, 2: 10})
    write_split(val, {0: 5, 1: 5, 2: 5})
    return trainer.TrainConfig(
        model_name="distilbert-base-uncased",
        train_file=str(train),
        val_file=str(val),
        **overrides,
    )


def test_distillation_settings_exist_only_with_a_teacher(tmp_path):
    assert make_config(tmp_path).distillation is None
    config = make_config(
        tmp_path, teacher_model_path="teacher", temperature=2.0, kd_alpha=0.3
    )
    assert config.distillation == DistillationSettings(2.0, 0.3)


def test_command_line_maps_onto_the_config():
    args = trainer.build_parser().parse_args(
        [
            "--model",
            "mmbert-32k",
            "--train-file",
            "t",
            "--val-file",
            "v",
            "--no-class-weights",
            "--seed",
            "3",
        ]
    )
    config = trainer.config_from_args(args)
    assert config.model_name == "mmbert-32k"
    assert config.use_class_weights is False
    assert config.seed == 3
    assert config.teacher_model_path is None


def test_training_rows_are_oversampled_reproducibly(tmp_path):
    config = make_config(tmp_path, seed=11)
    train_a, val_a, stats = trainer.prepare_training_rows(config)
    train_b, _, _ = trainer.prepare_training_rows(config)
    assert len(val_a) == 15
    assert stats.label_counts == {0: 40, 1: 40, 2: 10}
    assert [sum(r["label"] == c for r in train_a) for c in range(3)] == [40, 40, 40]
    assert train_a == train_b


def test_class_weighting_can_be_switched_off(tmp_path):
    train, _, stats = trainer.prepare_training_rows(
        make_config(tmp_path, use_class_weights=False)
    )
    assert stats is None
    assert len(train) == 90


def test_training_arguments_carry_seed_and_the_warmup_ratio(tmp_path):
    args = trainer.build_training_args(
        make_config(tmp_path, seed=5, num_epochs=4), str(tmp_path)
    )
    assert isinstance(args, TrainingArguments)
    assert args.seed == 5 and args.data_seed == 5
    # transformers >=5.15 reads the ratio from warmup_steps, older versions from warmup_ratio
    assert 0.06 in (args.warmup_steps, getattr(args, "warmup_ratio", None))
    assert args.remove_unused_columns is False


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, features):
        return types.SimpleNamespace(logits=self.linear(features))


def make_trainer(tmp_path, **kwargs):
    return trainer.ModalityTrainer(
        model=TinyModel(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        **kwargs,
    )


def batch(with_teacher):
    inputs = {
        "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "labels": torch.tensor([0, 2]),
    }
    if with_teacher:
        inputs["teacher_logits"] = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    return inputs


def test_loss_without_teacher_logits_is_the_hard_loss_only(tmp_path):
    t = make_trainer(
        tmp_path, focal_gamma=2.0, distillation=DistillationSettings(3.0, 0.5)
    )
    loss, outputs = t.compute_loss(
        t.model, batch(with_teacher=False), return_outputs=True
    )
    expected = t._hard_loss_fn(outputs.logits.device)(
        outputs.logits, torch.tensor([0, 2])
    )
    assert loss.item() == pytest.approx(expected.item())


def test_loss_with_teacher_logits_adds_the_soft_term(tmp_path):
    settings = DistillationSettings(3.0, 0.5)
    t = make_trainer(tmp_path, focal_gamma=2.0, distillation=settings)
    loss, outputs = t.compute_loss(
        t.model, batch(with_teacher=True), return_outputs=True
    )
    hard = t._hard_loss_fn(outputs.logits.device)(outputs.logits, torch.tensor([0, 2]))
    expected = distillation_loss(
        outputs.logits, batch(True)["teacher_logits"], hard, settings
    )
    assert loss.item() == pytest.approx(expected.item())


def test_teacher_column_is_ignored_when_distillation_is_off(tmp_path):
    t = make_trainer(tmp_path, focal_gamma=2.0, distillation=None)
    with_col = t.compute_loss(t.model, batch(with_teacher=True))
    without = t.compute_loss(t.model, batch(with_teacher=False))
    assert with_col.item() == pytest.approx(without.item())


def test_class_weights_reach_the_focal_loss(tmp_path):
    t = make_trainer(tmp_path, class_weights=[0.5, 1.0, 3.0], focal_gamma=2.0)
    fn = t._hard_loss_fn(torch.device("cpu"))
    assert fn.alpha.tolist() == [0.5, 1.0, 3.0]
    assert t._hard_loss_fn(torch.device("cpu")) is fn


class FakeDataset:
    def __init__(self):
        self.columns = {}

    def add_column(self, name, values):
        self.columns[name] = values
        return self


class FakeTeacher:
    config = types.SimpleNamespace(num_labels=3)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **enc):
        return types.SimpleNamespace(
            logits=torch.tensor([[0.1, 0.7, 0.2]] * len(enc["features"]))
        )


class FakeTeacherTokenizer:
    def __call__(self, batch, **kwargs):
        class Enc(dict):
            def to(self, device):
                return self

        return Enc(features=torch.zeros(len(batch), 2))


def load_fake_teacher(id2label):
    return lambda path, num_labels: (FakeTeacher(), FakeTeacherTokenizer(), id2label)


def test_teacher_logits_are_reordered_to_canonical_columns(monkeypatch):
    # teacher classes are (BOTH, AR, DIFFUSION); the student expects (AR, DIFFUSION, BOTH)
    monkeypatch.setattr(
        trainer,
        "load_sequence_classifier_for_inference",
        load_fake_teacher({0: "BOTH", 1: "AR", 2: "DIFFUSION"}),
    )
    monkeypatch.setattr(trainer, "clear_gpu_memory", lambda: None)
    dataset = trainer.add_teacher_logits(
        FakeDataset(), ["a", "b"], "teacher", batch_size=8, max_length=8, device="cpu"
    )
    assert np.allclose(dataset.columns["teacher_logits"], [[0.7, 0.2, 0.1]] * 2)


def test_a_teacher_with_incompatible_labels_is_rejected(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "load_sequence_classifier_for_inference",
        load_fake_teacher({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}),
    )
    with pytest.raises(ValueError, match="refusing to guess"):
        trainer.add_teacher_logits(
            FakeDataset(), ["a"], "teacher", batch_size=8, max_length=8, device="cpu"
        )
