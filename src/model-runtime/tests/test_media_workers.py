"""Request media decoded in worker processes: the same patch rows and the same rejections as in-process decoding."""

import os
import pickle

import pytest
import torch
from vllm_srun import media
from vllm_srun.errors import MediaError
from vllm_srun.families.decision3 import images as img

SETTINGS = img.ProcessorSettings(
    patch_size=16,
    temporal_patch_size=2,
    merge_size=2,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


@pytest.fixture()
def workers():
    """Set the media worker count for one test; the pool stops after it."""

    def configure(count):
        media.stop()
        media.configure(count)

    yield configure
    media.stop()
    media.configure(None)


def test_a_rejection_keeps_its_reason_and_limit_across_processes():
    error = MediaError("image_too_large", "too big", 8_000_000)
    copy = pickle.loads(pickle.dumps(error))
    assert (copy.reason, str(copy), copy.limit) == (
        "image_too_large",
        "too big",
        8_000_000,
    )


def test_without_workers_decoding_runs_on_the_calling_thread(workers):
    workers(0)
    future = media.submit(divmod, 7, 2)
    assert future.done() and future.result() == (3, 1)
    with pytest.raises(MediaError) as caught:
        media.submit(img.load, "not a data url", SETTINGS).result()
    assert caught.value.reason == "image_format"


def test_stopping_the_workers_waits_for_them_to_exit(workers):
    workers(1)
    pid = media.outcome(media.submit(os.getpid), os.getpid)
    assert pid != os.getpid()
    media.stop()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_worker_processes_decode_images_bit_for_bit(workers, png_url):
    workers(1)
    value = png_url(97, 61, 5)
    picture, digest = img.decode(value)
    inline = img.preprocess(picture, digest, SETTINGS)
    pixels, worker_digest = media.outcome(
        media.submit(img.load, value, SETTINGS), img.load, value, SETTINGS
    )
    assert pixels.dtype.name == "uint8" and worker_digest == digest
    decoded = img.loaded(pixels, worker_digest, SETTINGS)
    assert (decoded.grid, decoded.tokens, decoded.digest) == (
        inline.grid,
        inline.tokens,
        inline.digest,
    )
    assert torch.equal(decoded.pixel_values, inline.pixel_values)
    with pytest.raises(MediaError) as caught:
        media.submit(img.load, "data:image/gif;base64,R0lGODlh", SETTINGS).result()
    assert caught.value.reason == "image_format"


def test_requests_read_the_same_images_with_and_without_workers(
    d3_runtime, png_url, workers
):
    model = d3_runtime.lookup(None).model
    urls = [png_url(80, 60, 51), png_url(64, 96, 52)]
    answers = {}
    for count in (0, 2):
        workers(count)
        model.media_cache = type(model.media_cache)(model.media_cache.max_bytes)
        plan = model.plan_images(
            "Shown?", {"q": {"type": "noul", "instructions": "Shown?"}}, urls
        )
        answers[count] = (
            [image.pixel_values for image in plan.items[0].media.images],
            model.run(list(plan.items)),
        )
    for inline, worker in zip(answers[0][0], answers[2][0], strict=True):
        assert torch.equal(inline, worker)
    assert answers[0][1] == answers[2][1]


def test_images_are_processed_on_the_cpu_device_thread(monkeypatch, png_url):
    """One OpenMP team: preprocessing shares the thread CPU forwards run on, from any caller."""
    import threading

    from vllm_srun.accel.cpu import device_thread

    threads = []
    patches = img._patches

    def recorded(*args):
        threads.append(threading.current_thread().name)
        return patches(*args)

    monkeypatch.setattr(img, "_patches", recorded)
    picture, digest = img.decode(png_url(97, 61, 5))
    img.preprocess(picture, digest, SETTINGS)
    pixels, _ = img.load(png_url(97, 61, 5), SETTINGS)
    img.loaded(pixels, digest, SETTINGS)
    device_thread().submit(img.preprocess, picture, digest, SETTINGS).result(timeout=60)
    device = device_thread().submit(lambda: threading.current_thread().name).result()
    assert threads == [device, device, device]
