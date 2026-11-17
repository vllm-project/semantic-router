"""The batch limits a model leaves to the runtime: device classes and the startup calibration."""

import pytest
from vllm_srun.config import EngineArgs
from vllm_srun.plugins.base import DeviceInfo
from vllm_srun.scheduler import calibration

GIB = 1 << 30


@pytest.mark.parametrize(
    "memory_gib, limits",
    [
        (256, (32_768, 256)),
        (192, (32_768, 256)),
        (140, (16_384, 128)),
        (80, (16_384, 128)),
        (64, (16_384, 128)),
        (48, (8_192, 64)),
        (24, (4_096, 32)),
    ],
)
def test_gpus_take_the_limits_of_their_memory_class(memory_gib, limits):
    total = None if memory_gib is None else memory_gib * GIB
    device = DeviceInfo("rocm", 0, "gpu", total_memory=total)
    args, source = calibration.resolve(EngineArgs(), device)
    assert (args.max_num_batched_tokens, args.max_num_seqs) == limits
    assert source.endswith("device class")


def test_the_cpu_keeps_its_defaults_and_set_limits_are_kept():
    args, source = calibration.resolve(EngineArgs(), DeviceInfo("cpu", None, "cpu"))
    assert (args.max_num_seqs, args.max_num_batched_tokens) == (8_192, 65_536)
    assert source == "the defaults"
    for unsized in (DeviceInfo("mps", 0, "gpu"), DeviceInfo("example_host", 0, "gpu")):
        args, source = calibration.resolve(EngineArgs(), unsized)
        assert (args.max_num_seqs, args.max_num_batched_tokens) == (8_192, 65_536)
        assert source == "the defaults"
        assert calibration.device_class(unsized) is None
    gpu = DeviceInfo("cuda", 0, "gpu", total_memory=256 * GIB)
    args, _ = calibration.resolve(EngineArgs(max_num_seqs=8), gpu)
    assert (args.max_num_seqs, args.max_num_batched_tokens) == (8, 32_768)
    args, source = calibration.resolve(
        EngineArgs(max_num_seqs=8, max_num_batched_tokens=1024), gpu
    )
    assert (args.max_num_seqs, args.max_num_batched_tokens) == (8, 1024)
    assert source == "the engine arguments"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TimedModel:
    """Merged forwards that take ``overhead + tokens * per_token + tokens**2 * growth`` seconds of a fake clock and peak at ``bytes_per_token``."""

    def __init__(self, clock, overhead, per_token, bytes_per_token=0, growth=0.0):
        self.clock, self.overhead, self.per_token = clock, overhead, per_token
        self.bytes_per_token, self.growth = bytes_per_token, growth
        self.peak = 0
        self.sizes = []

    def calibration_items(self, rows, row_tokens):
        return [row_tokens] * rows

    def run_approximate(self, items):
        tokens = sum(items)
        self.sizes.append(tokens)
        self.clock.now += (
            self.overhead + tokens * self.per_token + tokens * tokens * self.growth
        )
        self.peak = tokens * self.bytes_per_token
        return [None] * len(items)


def run(model, clock, budget=32_768, available=None):
    memory = None
    if available is not None:

        def memory():
            return available, lambda: model.peak

    return calibration.calibrate(model, lambda work: work(), budget, memory, clock)


def test_a_steep_cost_curve_keeps_the_largest_budget():
    clock = Clock()
    # Per-forward overhead dominates small forwards: 32k tokens are cheapest per token.
    chosen = run(TimedModel(clock, overhead=0.02, per_token=2e-6), clock)
    assert [forward.tokens for forward in chosen.forwards] == [
        4096,
        8192,
        16_384,
        32_768,
    ]
    assert chosen.max_num_batched_tokens == chosen.knee == 32_768
    assert "knee 32768" in chosen.describe()


def test_a_flat_cost_curve_keeps_the_largest_budget_on_the_plateau():
    clock = Clock()
    chosen = run(TimedModel(clock, overhead=0.0002, per_token=5e-6), clock)
    assert chosen.max_num_batched_tokens == chosen.knee == 32_768


def test_a_cost_per_token_that_rises_stops_at_the_knee():
    clock = Clock()
    # 4,096 and 8,192 tokens cost the same per token, 16,384 3.5 % more and 32,768 12 % more,
    # although a 32,768-token forward fits in 0.2 s.
    chosen = run(TimedModel(clock, overhead=0.001, per_token=5e-6, growth=3e-11), clock)
    assert chosen.latency_cap == 32_768
    assert chosen.max_num_batched_tokens == chosen.knee == 8192


def test_a_slow_model_caps_one_forward_at_the_latency_target():
    clock = Clock()
    model = TimedModel(clock, overhead=0.0, per_token=4e-5)
    chosen = run(model, clock)
    # 8,192 tokens take 0.33 s: timing stops there, and 0.3 s holds 7,500 tokens
    # (7,424 in whole rows); the flat cost per token reaches past 4,096 tokens.
    assert [forward.tokens for forward in chosen.forwards] == [4096, 8192]
    assert chosen.latency_cap == chosen.knee == chosen.max_num_batched_tokens == 7424
    clock = Clock()
    slower = run(TimedModel(clock, overhead=0.0, per_token=1e-4), clock)
    # Even 4,096 tokens take 0.41 s: the budget scales down to what 0.3 s holds.
    assert slower.max_num_batched_tokens == 2816


def test_a_cost_still_falling_at_the_latency_target_keeps_the_target():
    clock = Clock()
    # A 4B model's forwards on one MI325X: 16,384 tokens in 258 ms, 32,768 in 504 ms.
    model = TimedModel(clock, overhead=0.0108, per_token=1.505e-5)
    chosen = run(model, clock)
    assert [forward.tokens for forward in chosen.forwards] == [
        4096,
        8192,
        16_384,
        32_768,
    ]
    # 0.3 s holds 19,216 tokens, 19,200 in whole rows; the cheapest forward does not fit.
    assert chosen.latency_cap == chosen.knee == chosen.max_num_batched_tokens == 19_200
    clock = Clock()
    # The same device on another start: 16,384 tokens within 2 % of 32,768 per token.
    model = TimedModel(clock, overhead=0.004, per_token=1.52e-5)
    chosen = run(model, clock)
    assert chosen.latency_cap == chosen.knee == chosen.max_num_batched_tokens == 19_456


def test_the_budget_stays_within_free_device_memory():
    clock = Clock()
    model = TimedModel(clock, overhead=0.02, per_token=2e-6, bytes_per_token=1 << 20)
    # 16,384 tokens peak at 16 GiB, over 85 % of 16 GiB free.
    chosen = run(model, clock, available=16 * GIB)
    assert chosen.memory_cap == chosen.max_num_batched_tokens == 8192
    assert [forward.tokens for forward in chosen.forwards] == [4096, 8192, 16_384]


def test_a_model_without_calibration_rows_keeps_its_class_limits():
    class Silent:
        def calibration_items(self, rows, row_tokens):
            return None

    assert calibration.calibrate(Silent(), lambda work: work(), 32_768) is None
