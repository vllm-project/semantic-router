"""Omni audio result-cache hits must not repeat feature preparation."""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.multimodal_embedding import audio
from vllm_srun.runtime import Runtime
from vllm_srun.testing.omni import write_snapshot


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_snapshot(
        tmp_path_factory.mktemp("audio-cache") / "mini", variant="mini", seed=5677
    )


@pytest.fixture
def runtime_factory(snapshot):
    running = []

    def start(entries=4):
        runtime = Runtime(
            ServeConfig(
                models=(
                    ModelConfig(
                        model=str(snapshot), name="omni", device="cpu", engine="native"
                    ),
                ),
                result_cache_entries=entries,
                threads=2,
            )
        )
        runtime.start(background=False)
        running.append(runtime)
        return runtime

    yield start
    for runtime in running:
        runtime.stop()


def sound(rate=44100, frequency=440, audio_format="wav"):
    frames = np.round(
        32767 * 0.2 * np.sin(2 * np.pi * frequency * np.arange(rate // 10) / rate)
    ).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames.tobytes())
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(buffer.getvalue()).decode(),
            "format": audio_format,
        },
    }


def call(runtime, inputs):
    return asyncio.run(runtime.call("embeddings", {"model": "omni", "input": inputs}))


def test_audio_result_hit_skips_preprocessing(runtime_factory):
    runtime = runtime_factory()
    model = runtime.lookup("omni").model
    body = {"model": "omni", "input": [sound()]}

    async def requests():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://localhost",
        ) as client:
            with patch.object(
                model.audio, "features", wraps=model.audio.features
            ) as prepare, patch.object(
                audio, "resample", wraps=audio.resample
            ) as resample, patch.object(
                audio, "features", wraps=audio.features
            ) as log_mel, patch.object(
                model, "_audio", wraps=model._audio
            ) as forward:
                first = await client.post("/v1/embeddings", json=body)
                assert first.status_code == 200
                assert (
                    prepare.call_count,
                    resample.call_count,
                    log_mel.call_count,
                    forward.call_count,
                ) == (1, 2, 2, 1)
                for spy in (prepare, resample, log_mel, forward):
                    spy.reset_mock()
                second = await client.post("/v1/embeddings", json=body)
                assert second.status_code == 200
                assert second.json()["data"] == first.json()["data"]
                assert second.json()["usage"] == first.json()["usage"]
                assert (
                    second.json()["meta"]["representation"]
                    == first.json()["meta"]["representation"]
                )
                assert (
                    prepare.call_count,
                    resample.call_count,
                    log_mel.call_count,
                    forward.call_count,
                ) == (0, 0, 0, 0)

    asyncio.run(requests())


def test_cache_disabled_keeps_the_original_audio_path(runtime_factory):
    runtime = runtime_factory(entries=0)
    model = runtime.lookup("omni").model
    inputs = [sound()]
    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(model, "_audio", wraps=model._audio) as forward:
        first = call(runtime, inputs)
        second = call(runtime, inputs)
    assert first[0] == second[0] == 200
    assert first[1]["data"] == second[1]["data"]
    assert prepare.call_count == forward.call_count == 2


@pytest.mark.parametrize(
    "audio_format", ["wave", "audio/wav", "audio/wave", "audio/x-wav"]
)
def test_supported_audio_aliases_reuse_the_same_result(runtime_factory, audio_format):
    runtime = runtime_factory()
    first = call(runtime, [sound()])
    model = runtime.lookup("omni").model
    with patch.object(model.audio, "features", wraps=model.audio.features) as prepare:
        second = call(runtime, [sound(audio_format=audio_format)])
    assert second[0] == 200
    assert second[1]["data"] == first[1]["data"]
    assert prepare.call_count == 0


@pytest.mark.parametrize("audio_format", ["mp3", "audio/ogg", ""])
def test_a_cached_wav_cannot_bypass_unsupported_format_validation(
    runtime_factory, audio_format
):
    runtime = runtime_factory()
    first = call(runtime, [sound()])
    model = runtime.lookup("omni").model
    with patch.object(model, "_audio", wraps=model._audio) as forward:
        rejected = call(runtime, [sound(audio_format=audio_format)])
    assert rejected[0] == 200
    assert rejected[1]["data"][0]["error"] == "invalid_input"
    assert forward.call_count == 0
    assert call(runtime, [sound()])[1]["data"] == first[1]["data"]


def test_mixed_hit_invalid_and_miss_keep_input_order(runtime_factory):
    runtime = runtime_factory()
    first = call(runtime, [sound()])
    invalid = {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(b"not a WAV").decode(),
            "format": "wav",
        },
    }
    model = runtime.lookup("omni").model
    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(model, "_audio", wraps=model._audio) as forward:
        response = call(runtime, [sound(), invalid, sound(frequency=880)])
    entries = response[1]["data"]
    assert response[0] == 200
    assert [entry["index"] for entry in entries] == [0, 1, 2]
    assert entries[0]["embedding"] == first[1]["data"][0]["embedding"]
    assert entries[1]["error"] == "invalid_input"
    assert len(entries[2]["embedding"]) == 768
    assert prepare.call_count == 2  # Invalid and missing inputs, not the hit.
    assert forward.call_count == 1
    assert runtime.lookup("omni").health.state == "ready"


def test_eviction_returns_to_the_original_miss_path(runtime_factory):
    runtime = runtime_factory(entries=1)
    first = call(runtime, [sound()])
    call(runtime, [sound(frequency=880)])
    model = runtime.lookup("omni").model
    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(model, "_audio", wraps=model._audio) as forward:
        repeated = call(runtime, [sound()])
    assert repeated[1]["data"] == first[1]["data"]
    assert prepare.call_count == forward.call_count == 1


def test_planning_retains_a_hit_when_the_cache_entry_is_evicted(runtime_factory):
    runtime = runtime_factory(entries=1)
    first = call(runtime, [sound()])
    prepared = runtime.prepare("embeddings", {"model": "omni", "input": [sound()]})
    assert prepared.plan.items[0].features == {}
    assert prepared.plan.cache_lookups[0][0]
    served = runtime.lookup("omni")
    served.cache.put("another-input", [0.0])
    with patch.object(served.cache, "get", wraps=served.cache.get) as lookup:
        retained = runtime._lookup([(0, prepared)])
    assert retained.misses == [[]]
    assert lookup.call_count == 0
    assert retained.values[(0, 0)] == first[1]["data"][0]["embedding"]
    response = runtime.finish(prepared, [retained.values[(0, 0)]], 0, 0)
    assert response["data"] == first[1]["data"]


def test_cache_metrics_are_not_counted_twice(runtime_factory):
    runtime = runtime_factory()
    served = runtime.lookup("omni")
    with patch.object(served.cache, "get", wraps=served.cache.get) as lookup:
        call(runtime, [sound()])
        call(runtime, [sound()])
    assert lookup.call_count == 3
    for outcome in ("hit", "miss"):
        metric = runtime.metrics.result_cache.labels(model="omni", outcome=outcome)
        assert metric._value.get() == 1


def test_concurrent_warm_hits_do_not_prepare_or_forward(runtime_factory):
    runtime = runtime_factory()
    inputs = [sound()]
    first = call(runtime, inputs)
    model = runtime.lookup("omni").model

    async def requests():
        return await asyncio.gather(
            *(
                runtime.call("embeddings", {"model": "omni", "input": inputs})
                for _ in range(8)
            )
        )

    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(model, "_audio", wraps=model._audio) as forward:
        responses = asyncio.run(requests())
    assert all(
        response[0] == 200 and response[1]["data"] == first[1]["data"]
        for response in responses
    )
    assert prepare.call_count == forward.call_count == 0


def test_concurrent_cold_inputs_remain_valid(runtime_factory):
    runtime = runtime_factory()

    async def requests():
        return await asyncio.gather(
            *(
                runtime.call(
                    "embeddings",
                    {"model": "omni", "input": [sound(frequency=frequency)]},
                )
                for frequency in (440, 440, 880, 1320)
            )
        )

    responses = asyncio.run(requests())
    assert all(
        response[0] == 200 and len(response[1]["data"][0]["embedding"]) == 768
        for response in responses
    )
    assert responses[0][1]["data"] == responses[1][1]["data"]
    assert runtime.lookup("omni").health.state == "ready"


def test_result_cache_is_local_to_each_loaded_model(runtime_factory):
    first_runtime = runtime_factory()
    reference = call(first_runtime, [sound()])
    second_runtime = runtime_factory()
    model = second_runtime.lookup("omni").model
    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(model, "_audio", wraps=model._audio) as forward:
        independent = call(second_runtime, [sound()])
    assert independent[1]["data"] == reference[1]["data"]
    assert prepare.call_count == forward.call_count == 1


def test_reference_forward_ignores_the_result_cache(runtime_factory):
    runtime = runtime_factory()
    first = call(runtime, [sound()])
    served = runtime.lookup("omni")
    model = served.model
    with patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(
        model, "_audio", wraps=model._audio
    ) as forward, patch.object(
        served.cache, "get", wraps=served.cache.get
    ) as lookup:
        reference = served.answer_alone("embeddings", {"input": [sound()]})
    assert reference["data"] == first[1]["data"]
    assert prepare.call_count == forward.call_count == 1
    assert lookup.call_count == 0


def test_cached_none_preserves_invalid_model_output(runtime_factory):
    runtime = runtime_factory()
    call(runtime, [sound()])
    served = runtime.lookup("omni")
    prepared = runtime.prepare("embeddings", {"model": "omni", "input": [sound()]})
    served.cache.put(prepared.plan.items[0].cache_key, None)
    with patch.object(
        served.model.audio, "features", wraps=served.model.audio.features
    ) as prepare:
        response = call(runtime, [sound()])
    assert response[0] == 200
    assert response[1]["data"][0]["error"] == "invalid_model_output"
    assert prepare.call_count == 0


def test_base_cached_planning_preserves_existing_plugins():
    from vllm_srun.plugins.base import LoadedModel, SurfacePlan, SurfaceRequest

    class LegacyModel(LoadedModel):
        def plan_surface(self, surface, request):
            return SurfacePlan(surface, [], 0, request.body)

        def run(self, items):
            return []

        def finish_surface(self, plan, results):
            return plan.state

    request = SurfaceRequest("embeddings", {"input": "text"}, None, True, 0)

    def lookup(key):
        pytest.fail("a legacy plugin must not call the early lookup")

    plan = LegacyModel().plan_cached_surface("embeddings", request, lookup)
    assert plan.state is request.body
    assert plan.cache_lookups == {}


def test_a_planned_miss_can_use_a_result_published_during_preparation(
    runtime_factory,
):
    runtime = runtime_factory()
    served = runtime.lookup("omni")
    prepared = runtime.prepare("embeddings", {"model": "omni", "input": [sound()]})
    item = prepared.plan.items[0]
    assert item.features and prepared.plan.cache_lookups == {}
    expected = served.answer_alone("embeddings", {"input": [sound()]})["data"][0][
        "embedding"
    ]
    served.cache.put(item.cache_key, expected)
    with patch.object(served.cache, "get", wraps=served.cache.get) as lookup:
        observed = runtime._lookup([(0, prepared)])
    assert lookup.call_count == 1
    assert observed.values == {(0, 0): expected} and observed.misses == [[]]


@pytest.mark.parametrize("warm", [False, True])
def test_duplicate_positions_and_invalid_sibling_keep_one_lookup_per_key(
    runtime_factory, warm
):
    runtime = runtime_factory(entries=1)
    served = runtime.lookup("omni")
    if warm:
        call(runtime, [sound()])
    with patch.object(served.cache, "get", wraps=served.cache.get) as lookup:
        prepared = runtime.prepare(
            "embeddings",
            {"model": "omni", "input": [sound(), sound(audio_format="mp3"), sound()]},
        )
    assert lookup.call_count == 1
    assert [item.index for item in prepared.plan.items] == [0, 2]
    assert set(prepared.plan.cache_lookups) == ({0, 1} if warm else set())
    assert all(result[0] for result in prepared.plan.cache_lookups.values())
    if warm:
        served.cache.put("replacement", [0.0])
    from vllm_srun.timing import ServerTiming

    outcomes = [None]
    asyncio.run(runtime._run_group([(0, prepared)], outcomes, ServerTiming()))
    status, body = outcomes[0]
    assert status == 200
    assert body["data"][0]["embedding"] == body["data"][2]["embedding"]
    assert body["data"][1]["error"] == "invalid_input"
    outcome = "hit" if warm else "miss"
    assert (
        runtime.metrics.result_cache.labels(model="omni", outcome=outcome)._value.get()
        == 2
    )


def test_cached_hit_survives_eviction_beside_an_expired_miss(runtime_factory):
    import time
    from dataclasses import replace

    from vllm_srun.timing import ServerTiming

    runtime = runtime_factory(entries=1)
    first = call(runtime, [sound()])
    served = runtime.lookup("omni")
    prepared = runtime.prepare(
        "embeddings", {"model": "omni", "input": [sound(), sound(frequency=880)]}
    )
    missed_key = prepared.plan.items[1].cache_key
    prepared.request = replace(prepared.request, deadline=time.monotonic() - 1)
    served.cache.put("replacement", [0.0])
    outcomes = [None]
    with patch.object(served.model, "_audio", wraps=served.model._audio) as forward:
        asyncio.run(runtime._run_group([(0, prepared)], outcomes, ServerTiming()))
    status, body = outcomes[0]
    assert status == 200
    assert body["data"][0]["embedding"] == first[1]["data"][0]["embedding"]
    assert body["data"][1]["error"] == "deadline_exceeded"
    assert forward.call_count == 0
    assert served.cache.get(missed_key)[0] is False


def test_runtime_preserves_the_default_legacy_planning_hook(runtime_factory):
    from vllm_srun.plugins.base import LoadedModel

    runtime = runtime_factory()
    served = runtime.lookup("omni")
    model = served.model
    legacy_hook = LoadedModel.plan_cached_surface.__get__(model, type(model))
    with patch.object(model, "plan_cached_surface", legacy_hook), patch.object(
        model, "plan_surface", wraps=model.plan_surface
    ) as planning, patch.object(
        model.audio, "features", wraps=model.audio.features
    ) as prepare, patch.object(
        served.cache, "get", wraps=served.cache.get
    ) as lookup:
        first = call(runtime, [sound()])
        second = call(runtime, [sound()])
    assert first[0] == second[0] == 200
    assert first[1]["data"] == second[1]["data"]
    assert planning.call_count == prepare.call_count == lookup.call_count == 2
