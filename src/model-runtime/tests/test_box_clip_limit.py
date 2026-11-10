"""The video forms of Box read clips up to the card's limits.box_video_max_seconds: a
box + at or track question on a longer clip gets invalid_question / box_video_too_long before any frame window is
decoded, the request's other questions are answered, and a card without the limit keeps every clip up to 300 s.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from vllm_srun.errors import QUESTION_REASONS

from . import test_composed
from .test_composed import CLIP, decide

# test_composed's tiny state-once model, as this module's fixture too.
model = test_composed.model


@pytest.fixture()
def tracked(model, monkeypatch):
    """The card's Box and track at a 60-s limit, and a 75-s clip from a window decoder standing in for OpenCV."""
    import numpy as np
    from vllm_srun.families.decision3 import state_once
    from vllm_srun.families.decision3 import videos as vid

    seen = []

    def fake_windows(value, rate, max_seconds=None):
        fps, total = 4.0, 4 * 75
        seen.append(max_seconds)
        if max_seconds is not None and total / fps > max_seconds:
            raise vid.ClipTooLongError(total / fps, max_seconds)
        for number, window in enumerate(vid.window_plan(total, fps, rate)):
            frames = [
                np.full((64, 96, 3), (i * 7) % 255, dtype=np.uint8) for i in window
            ]
            yield vid._video(frames, list(window), fps, total, f"clip:w{rate}:{number}")

    monkeypatch.setattr(state_once.vid, "decode_windows", fake_windows)
    if model.video_settings is None:
        monkeypatch.setattr(
            model,
            "video_settings",
            vid.VideoSettings(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=(0.5,) * 3,
                image_std=(0.5,) * 3,
            ),
        )
        monkeypatch.setattr(
            model, "video_token", model.tokenizer.token_to_id("<|video_pad|>")
        )
        monkeypatch.setattr(model, "video_inputs", True)
    monkeypatch.setattr(
        model,
        "info",
        replace(
            model.info,
            question_types=(*model.info.question_types, "box", "track"),
            limits={**model.info.limits, "box_video_max_seconds": 60},
        ),
    )
    return seen


def test_the_reason_is_a_fixed_word():
    assert "box_video_too_long" in QUESTION_REASONS["invalid_question"]


def test_a_clip_over_the_limit_fails_the_video_box_questions(model, tracked):
    """Each video Box question gets its own error and reason; with no other question the request fails as a whole,
    as for every request whose questions are all invalid."""
    questions = {
        "track": {"type": "track", "instructions": "the dog", "rate": 2},
        "at": {"type": "box", "instructions": "the dog", "at": 10},
    }
    plan = model.plan_composed("A clip.", questions, [], [CLIP], {})
    for key in questions:
        error = plan.errors[key]
        assert (
            error["error"] == "invalid_question"
            and error["reason"] == "box_video_too_long"
        ), error
        assert (
            "at most 60 seconds" in error["message"]
            and "75.0 seconds" in error["message"]
        ), error
    assert not plan.items
    assert tracked and set(tracked) == {60}


def test_without_the_limit_the_clip_is_read(model, tracked, monkeypatch):
    limits = {
        k: v for k, v in model.info.limits.items() if k != "box_video_max_seconds"
    }
    monkeypatch.setattr(model, "info", replace(model.info, limits=limits))
    answers = decide(
        model,
        {
            "state": "A clip.",
            "videos": [CLIP],
            "questions": {"at": {"type": "box", "instructions": "the dog", "at": 70}},
        },
    )
    assert answers["at"]["type"] == "box" and "error" not in answers["at"], answers[
        "at"
    ]
    assert tracked == [None]


def test_the_decoder_refuses_before_reading_frames():
    """The real decoder (OpenCV) on a 75-s clip at 1 frame per second: ClipTooLongError over 60 s, windows within 90 s."""
    cv2 = pytest.importorskip("cv2")
    import base64
    import tempfile
    from pathlib import Path

    import numpy as np
    from vllm_srun.families.decision3 import videos as vid

    if vid.available() is not None:
        pytest.skip(vid.available())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (64, 48)
        )
        rng = np.random.default_rng(7)
        for _ in range(75):
            writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
        writer.release()
        url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
    with pytest.raises(vid.ClipTooLongError) as caught:
        next(iter(vid.decode_windows(url, 1, 60)))
    assert caught.value.limit == 60 and caught.value.seconds == pytest.approx(75, abs=1)
    assert next(iter(vid.decode_windows(url, 1, 90))) is not None

    real = cv2.VideoCapture

    class NoRate:
        """A container that reports no usable frame rate (its length cannot be read)."""

        def __init__(self, *args):
            self.reader = real(*args)

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else self.reader.get(prop)

        def __getattr__(self, name):
            return getattr(self.reader, name)

    cv2.VideoCapture = NoRate
    try:
        with pytest.raises(vid.ClipTooLongError) as unknown:
            next(iter(vid.decode_windows(url, 1, 90)))
        assert unknown.value.seconds is None and unknown.value.limit == 90
        assert next(iter(vid.decode_windows(url, 1, None))) is not None
    finally:
        cv2.VideoCapture = real
