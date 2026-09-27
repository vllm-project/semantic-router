"""CPU-only contract checks for the no-update 9B TRAIN replay."""

from __future__ import annotations

import importlib.util
from itertools import chain
from pathlib import Path

from training.model.plan import epoch_batches


def _probe():
    path = Path(__file__).resolve().parents[1] / "probe_qwen35_9b_base_backward.py"
    spec = importlib.util.spec_from_file_location("probe_qwen35_9b_base_backward", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_107_is_original_epoch_window() -> None:
    probe = _probe()
    lengths = [(i * 17) % 97 + 1 for i in range(7324)]
    scheduled = probe.scheduled_indices(lengths, 20260926)
    batches = epoch_batches(
        lengths, [], epoch=0, seed=20260926, microbatch=1, replay_fraction=0.0
    )
    for step in (105, 106, 107):
        assert scheduled[str(step)] == [
            batch[0][1] for batch in batches[(step - 1) * 16 : step * 16]
        ]
        assert len(scheduled[str(step)]) == 16
    assert len(set(chain.from_iterable(scheduled.values()))) == 48
