"""The canonical label order must match the one the training script uses."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

import modality_routing_bert_finetuning_lora as baseline
from modality_label_mapping import MODALITY_LABELS


def test_canonical_labels_match_the_baseline_training_script():
    assert MODALITY_LABELS == baseline.MODALITY_LABELS
