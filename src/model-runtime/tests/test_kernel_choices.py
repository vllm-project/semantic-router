"""Per-model FLA kernel choices: resolution, routing per thread, and pinned models sharing one process.

The CPU tests drive fake autotuners as Triton drives FLA's (``key in cache``, then ``cache[key]``,
timing only a missing key). The GPU tests check the resolver against FLA's own config-file lookup,
and the routing on FLA's real autotuners.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import replace

import pytest
from vllm_srun.accel import autotune
from vllm_srun.accel.autotune import (
    KernelChoices,
    PinnedKernel,
    RoutedCache,
    fla_key_hash,
)
from vllm_srun.registry import builtin

L2NORM = "l2norm_fwd_kernel"
KEY = [128, 1, "torch.bfloat16"]


def config(**kwargs):
    return {
        "kwargs": kwargs,
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
        "maxnreg": None,
        "ir_override": None,
    }


def recorded(*entries, fla="0.5.2", kernel=L2NORM):
    return {
        "fla": fla,
        "triton": "3.7.1",
        "kernels": {kernel: [{"key": key, "config": value} for key, value in entries]},
    }


class FakeAutotuner:
    """Triton's lookup of an autotuned kernel: ``key in cache``, then ``cache[key]``, timing a missing key."""

    def __init__(self):
        self.cache = {}
        self.timed = []

    def run(self, key):
        if key not in self.cache:
            self.timed.append(key)
            self.cache[key] = {"timed": list(key)}
        return self.cache[key]


@pytest.fixture()
def fla(monkeypatch):
    """FLA 0.5.2 with one autotuned kernel, ``l2norm_fwd_kernel``; configurations stay dicts."""
    tuner = FakeAutotuner()
    monkeypatch.setattr(autotune, "fla_version", lambda: "0.5.2")
    monkeypatch.setattr(autotune, "fla_autotuners", lambda: {L2NORM: [tuner]})
    monkeypatch.setattr(autotune, "triton_config", lambda value: value)
    return tuner


def test_a_key_takes_its_entry_else_the_first_numeric_match_by_hash_else_the_first_entry():
    one = [128, 1, False, "torch.bfloat16"]
    two = [128, 2, False, "torch.bfloat16"]
    varlen = [128, 1, True, "torch.bfloat16"]
    kernel = PinnedKernel(
        [
            {"key": two, "config": config(BT=8)},
            {"key": one, "config": config(BT=16)},
            {"key": varlen, "config": config(BT=32)},
        ]
    )
    assert kernel.resolve(tuple(one)) == config(BT=16)
    assert kernel.resolve(tuple(varlen)) == config(BT=32)
    first_by_hash = min([one, two], key=fla_key_hash)
    assert kernel.resolve((256, 7, False, "torch.bfloat16")) == (
        config(BT=16) if first_by_hash == one else config(BT=8)
    )
    assert kernel.resolve((64, 3, True, "torch.bfloat16")) == config(BT=32)
    # Booleans and strings must be equal, so nothing matches: the kernel's first entry.
    assert kernel.resolve((128, 1, False, "torch.float32")) == config(BT=8)
    assert kernel.resolve((128, 1, None, "torch.bfloat16")) == config(BT=8)


def test_the_key_hash_is_fla_s():
    # FLA's AutotuneKey.key_hash: MD5 of the compact, key-sorted JSON of the tuning key.
    expected = hashlib.md5(b'[128,2,"torch.bfloat16"]').hexdigest()
    assert fla_key_hash((128, 2, "torch.bfloat16")) == expected
    assert fla_key_hash([128, 2, "torch.bfloat16"]) == expected


def test_each_thread_runs_the_choices_of_the_model_it_runs(fla):
    first = KernelChoices(recorded((KEY, config(BT=8))))
    second = KernelChoices(recorded((KEY, config(BT=32))))
    assert first.install() is None
    assert second.install() is None
    assert isinstance(fla.cache, RoutedCache)
    key = tuple(KEY)
    rounds = threading.Barrier(2)
    seen: dict[str, set[str]] = {"first": set(), "second": set()}

    def serve(name, choices):
        with choices.scope():
            for _ in range(50):
                rounds.wait()
                seen[name].add(json.dumps(fla.run(key)))

    threads = [
        threading.Thread(target=serve, args=("first", first)),
        threading.Thread(target=serve, args=("second", second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {
        "first": {json.dumps(config(BT=8))},
        "second": {json.dumps(config(BT=32))},
    }
    assert fla.timed == []
    # Outside every scope the autotuner keeps its own cache: one timing, then reuse.
    assert fla.run(key) == {"timed": KEY}
    assert fla.run(key) == {"timed": KEY}
    assert fla.timed == [key]
    assert dict(fla.cache) == {key: {"timed": KEY}}
    with first.scope():
        assert fla.run(key) == config(BT=8)


def test_a_scope_restores_the_choices_around_it(fla):
    first = KernelChoices(recorded((KEY, config(BT=8))))
    second = KernelChoices(recorded((KEY, config(BT=32))))
    assert first.install() is None
    key = tuple(KEY)
    with first.scope():
        assert second.run(lambda: fla.run(key)) == config(BT=32)
        assert fla.run(key) == config(BT=8)
    assert fla.run(key) == {"timed": KEY}


def test_routing_keeps_what_the_autotuner_had_already_tuned(fla):
    key = (128, 9, "torch.bfloat16")
    assert fla.run(key) == {"timed": [128, 9, "torch.bfloat16"]}
    choices = KernelChoices(recorded((KEY, config(BT=8))))
    assert choices.install() is None
    assert choices.install() is None
    assert isinstance(fla.cache, RoutedCache)
    assert not isinstance(fla.cache.own, RoutedCache)
    assert fla.run(key) == {"timed": [128, 9, "torch.bfloat16"]}
    assert fla.timed == [key]


def test_choices_that_cannot_run_here_say_why(fla, monkeypatch):
    entry = (KEY, config(BT=8))
    assert (
        KernelChoices(recorded(entry, fla="0.6.0")).install()
        == "they were recorded with FLA 0.6.0, not 0.5.2"
    )
    assert KernelChoices(recorded(entry, kernel="chunk_fwd_kernel_o")).install() == (
        "FLA 0.5.2 has no autotuned kernel chunk_fwd_kernel_o"
    )

    def broken():
        raise ImportError("no triton")

    monkeypatch.setattr(autotune, "fla_autotuners", broken)
    assert KernelChoices(recorded(entry)).install() == (
        "FLA 0.5.2 failed to import (ImportError: no triton)"
    )
    monkeypatch.setattr(autotune, "fla_version", lambda: None)
    assert KernelChoices(recorded(entry)).install() == "FLA is not installed"
    assert not isinstance(fla.cache, RoutedCache)


@pytest.fixture()
def two_fla_models(tmp_path, monkeypatch, fla):
    """Two Decision 1.0 decoders whose recorded choices give one FLA key different configurations.

    Every forward launches the fake FLA kernel, as a Qwen3.5 forward launches FLA's, and records
    the configuration it ran with.
    """
    from vllm_srun import runtime as runtime_module
    from vllm_srun.families.decision1.family import Decision1Family
    from vllm_srun.families.decision1.model import QwenDecisionModel
    from vllm_srun.testing.decision1 import write_package

    roots = [
        write_package(tmp_path / name, seed=index, model_name=name)
        for index, name in enumerate(("first", "second"))
    ]
    choices = {
        "first": recorded((KEY, config(BT=8))),
        "second": recorded((KEY, config(BT=32))),
    }
    launched: dict[str, set[str]] = {"first": set(), "second": set()}
    run = QwenDecisionModel.run_shared

    def forward(self, items, shared_prefix):
        launched[self.info.id].add(json.dumps(fla.run(tuple(KEY))))
        return run(self, items, shared_prefix)

    golden = runtime_module.golden_check

    def matched(*args, **kwargs):
        return replace(golden(*args, **kwargs), status="matched")

    monkeypatch.setattr(
        Decision1Family,
        "kernel_choices",
        lambda self, package, device: choices[package.model_name],
    )
    monkeypatch.setattr(QwenDecisionModel, "run_shared", forward)
    monkeypatch.setattr(runtime_module, "golden_check", matched)
    return roots, choices, launched


def serve(roots):
    from vllm_srun.config import ModelConfig, ServeConfig
    from vllm_srun.runtime import Runtime

    runtime = Runtime(
        ServeConfig(
            models=tuple(
                ModelConfig(model=str(root), device="cpu", name=root.name)
                for root in roots
            )
        )
    )
    runtime.start(background=False)
    return runtime


def ask(runtime, model):
    import asyncio

    body = {
        "model": model,
        "state": "Is this request about cooking?",
        "questions": {"q": {"type": "noul", "instructions": "about cooking"}},
    }
    status, response = asyncio.run(runtime.call("decisions", body))
    assert status == 200, response


def test_two_pinned_models_in_one_process_each_run_their_own_choices(
    two_fla_models, fla
):
    roots, _, launched = two_fla_models
    runtime = serve(roots)
    try:
        for _ in range(3):
            ask(runtime, "first")
            ask(runtime, "second")
        assert [served.health.state for served in runtime.served] == ["ready"] * 2
        assert [served.health.golden.status for served in runtime.served] == [
            "matched"
        ] * 2
    finally:
        runtime.stop()
    assert launched == {
        "first": {json.dumps(config(BT=8))},
        "second": {json.dumps(config(BT=32))},
    }
    assert fla.timed == []


def test_a_model_whose_choices_cannot_run_loads_unverified_and_says_why(
    two_fla_models, fla, caplog
):
    roots, choices, launched = two_fla_models
    choices["second"] = recorded((KEY, config(BT=32)), fla="0.6.0")
    with caplog.at_level(logging.WARNING, logger="vllm_srun"):
        runtime = serve(roots)
    try:
        ask(runtime, "second")
        first, second = runtime.served
        reason = (
            "kernel choices not applied: they were recorded with FLA 0.6.0, not 0.5.2"
        )
        assert first.health.golden.status == "matched"
        assert first.health.reason is None
        assert second.health.state == "ready"
        assert second.health.golden.status == "unverified"
        assert second.health.golden.detail == reason
        card = second.card([])
        assert (card["status"], card["reason"]) == ("ready", reason)
        assert card["golden"]["status"] == "unverified"
    finally:
        runtime.stop()
    assert "second runs without its recorded kernel choices" in caplog.text
    assert launched["first"] == {json.dumps(config(BT=8))}
    assert launched["second"] == {json.dumps({"timed": KEY})}


def test_a_pinned_model_without_a_device_thread_runs_inline_in_its_scope(
    two_fla_models, fla, monkeypatch
):
    from vllm_srun.families.decision1.model import QwenDecisionModel

    roots, _, launched = two_fla_models
    threads = set()
    run = QwenDecisionModel.run_shared

    def inline(self, items, shared_prefix):
        threads.add(threading.current_thread().name)
        return run(self, items, shared_prefix)

    monkeypatch.setattr(QwenDecisionModel, "run_shared", inline)
    monkeypatch.setattr(QwenDecisionModel, "device_thread", False)
    runtime = serve(roots)
    try:
        threads.clear()
        for _ in range(3):
            ask(runtime, "first")
            ask(runtime, "second")
    finally:
        runtime.stop()
    # Requests ran on their planning threads (a request that arrives while the worker
    # still holds the model runs on the worker instead); every forward ran its own choices.
    assert threads - {"vllm-srun-worker"}
    assert not any(name.startswith("vllm-sr-cpu") for name in threads)
    assert launched == {
        "first": {json.dumps(config(BT=8))},
        "second": {json.dumps(config(BT=32))},
    }
    assert fla.timed == []


def test_models_without_choices_never_enter_a_scope(tmp_path, monkeypatch):
    from vllm_srun.families.decision1.model import QwenDecisionModel
    from vllm_srun.testing.decision1 import write_package

    scopes = []
    run = QwenDecisionModel.run_shared

    def forward(self, items, shared_prefix):
        scopes.append(autotune._SCOPE.choices)
        return run(self, items, shared_prefix)

    monkeypatch.setattr(QwenDecisionModel, "run_shared", forward)
    runtime = serve([write_package(tmp_path / "plain", model_name="plain")])
    try:
        ask(runtime, "plain")
        assert runtime.served[0].kernel_choices is None
    finally:
        runtime.stop()
    assert scopes and set(scopes) == {None}


# -- GPU: FLA itself ---------------------------------------------------------


def _variants(key):
    """Unrecorded keys near ``key``: other numbers, flipped booleans, another dtype."""
    out = [
        [value + 7 if isinstance(value, int) and not isinstance(value, bool) else value
         for value in key],
        [not value if isinstance(value, bool) else value for value in key],
        [("torch.float16" if value == "torch.bfloat16" else value) for value in key],
    ]  # fmt: skip
    return [variant for variant in out if variant != key]


@pytest.mark.gpu
def test_the_resolver_picks_what_fla_picks_from_the_entries_as_config_files(
    tmp_path, monkeypatch
):
    cache = pytest.importorskip("fla.ops.utils.cache")
    monkeypatch.setattr(cache, "FLA_CACHE_MODE", cache.FlaCacheMode.FULL)
    compared = 0
    for model in builtin.all_models():
        choices = model.kernel_choices.get("rocm:gfx942")
        if not choices:
            continue
        directory = tmp_path / model.repo_id.replace("/", "--")
        directory.mkdir()
        for name, entries in choices["kernels"].items():
            document = {
                "kernel_name": name,
                "autotune_entries": {
                    fla_key_hash(entry["key"]): {
                        "autotune_key": entry["key"],
                        "config": entry["config"],
                    }
                    for entry in entries
                },
                "default_config": entries[0]["config"],
            }
            (directory / f"{name}.json").write_text(
                json.dumps(document, sort_keys=True), encoding="utf-8"
            )
        monkeypatch.setenv("FLA_CONFIG_DIR", str(directory))
        for name, entries in choices["kernels"].items():
            kernel = PinnedKernel(entries)
            recorded_keys = [entry["key"] for entry in entries]
            unrecorded = [v for key in recorded_keys for v in _variants(key)]
            for key in recorded_keys + unrecorded:
                expected = cache.load_cached_config(name, cache.AutotuneKey(tuple(key)))
                assert kernel.resolve(tuple(key)) == expected, (
                    model.repo_id,
                    name,
                    key,
                )
                compared += 1
    assert compared > 100


@pytest.mark.gpu
def test_fla_s_gated_delta_rule_runs_each_scoped_model_s_configurations():
    torch = pytest.importorskip("torch")
    pytest.importorskip("fla")
    from vllm_srun.accel.rocm import ROCmAccelerator

    accelerator = ROCmAccelerator()
    if not accelerator.available():
        pytest.skip("needs a ROCm GPU")
    device = accelerator.devices()[0]
    delta = accelerator.kernels(device).select("chunk_gated_delta_rule")
    if delta.source != "fla":
        pytest.skip("FLA's chunked gated delta rule is not installed")
    for model in builtin.all_models():
        recorded = model.kernel_choices.get("rocm:gfx942")
        if recorded:
            assert KernelChoices(recorded).install() is None, model.repo_id
    eos = builtin.lookup("vllm-sr/Decision-2.0-Eos-0.8B").kernel_choices["rocm:gfx942"]
    sol = builtin.lookup("vllm-sr/Decision-2.0-Sol-2B").kernel_choices["rocm:gfx942"]
    first, second = KernelChoices(eos), KernelChoices(sol)
    assert first.install() is None
    assert second.install() is None
    tuners = [t for ts in autotune.fla_autotuners().values() for t in ts]
    generator = torch.Generator(device="cuda").manual_seed(0)

    def inputs(heads):
        shape = (1, 96, heads, 128)

        def normal(*size):
            return torch.randn(
                *size, generator=generator, device="cuda", dtype=torch.bfloat16
            )

        gate = -torch.rand(1, 96, heads, generator=generator, device="cuda")
        beta = torch.rand(1, 96, heads, generator=generator, device="cuda")
        return normal(*shape), normal(*shape), normal(*shape), gate, beta.bfloat16()

    for choices, heads in ((first, 16), (second, 16), (first, 16)):
        for tuner in tuners:
            tuner.best_config = None
        q, k, v, g, beta = inputs(heads)
        with torch.inference_mode(), choices.scope():
            delta.fn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        ran = 0
        for tuner in tuners:
            name = tuner.base_fn.__name__
            if tuner.best_config is None or name not in choices.kernels:
                continue
            assert isinstance(tuner.cache, RoutedCache)
            assert any(
                tuner.best_config is value
                for value in choices.kernels[name].configs.values()
            ), name
            ran += 1
        assert ran >= len(choices.kernels) - 1
