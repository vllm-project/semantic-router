"""Composed images (``assets``) and Box answers on state-once Decision 3 packages.

A question's own images sit inline in its block and run as a row of their
own, so its answer never depends on the other questions; a Box question
branches from the state and reads its four edge markers in that forward, the
same values its row reads.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import replace

import pytest
import torch
from vllm_srun import assets as assetlib
from vllm_srun.config import EngineArgs, ServeConfig
from vllm_srun.families.decision3 import composed
from vllm_srun.families.decision3.state_once import StateOnceModel
from vllm_srun.plugins.base import SurfaceRequest

from .conftest import start_runtime

FP32_TOLERANCE = 2e-5


def png(width: int, height: int, colour: tuple[int, int, int]) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


PHOTO = png(64, 48, (200, 30, 30))
CROP_A = png(32, 32, (30, 200, 30))
CROP_B = png(48, 32, (30, 30, 200))
ASSETS = {"<photo>": PHOTO, "<a>": CROP_A, "<b>": CROP_B}
QUESTIONS = {
    "box": {"type": "box", "instructions": "the red square"},
    "mine": {
        "type": "choice",
        "instructions": "Which option looks like the photo?",
        "criteria": {"first": "<a>", "second": "<b>"},
    },
    "inline": {"type": "noul", "instructions": "Is <a> green?"},
    "plain": {"type": "noul", "instructions": "Is the photo red?"},
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from vllm_srun.testing.decision3 import write_fixture

    package = write_fixture(
        tmp_path_factory.mktemp("composed") / "pkg", "prefix_bidir", seed=31
    )
    runtime = start_runtime(package, engine_args=EngineArgs(dtype="float32"))
    found = runtime.lookup(None).model
    assert isinstance(found, StateOnceModel)
    yield found
    runtime.stop()


def plan(model, questions):
    return model.plan_composed({"photo": "<photo>"}, questions, [], None, ASSETS)


def test_box_and_question_images_answer(model):
    planned = plan(model, QUESTIONS)
    assert not planned.errors, planned.errors
    items = {item.question_id: item for item in planned.items}
    assert items["box"].readouts is not None and len(items["box"].readouts) == 4
    assert [
        len(i.block_images) for i in (items["mine"], items["inline"], items["plain"])
    ] == [2, 1, 0]
    values = model.run(planned.items)
    answers = {
        item.question_id: model.answer(item, value)
        for item, value in zip(planned.items, values, strict=True)
    }
    box = answers["box"]
    assert box["type"] == "box" and set(box["box"]) == set(composed.EDGES)
    assert (
        box["box"]["top"] < box["box"]["bottom"]
        and box["box"]["left"] < box["box"]["right"]
    )
    assert 0.0 <= box["confidence"] <= 1.0
    assert answers["mine"]["type"] == "choice" and answers["inline"]["type"] == "noul"
    # The state counts once, every block (images included) once.
    assert planned.input_tokens == sum(item.cost for item in planned.items)


def test_branched_box_readouts_equal_its_row(model):
    planned = plan(model, {"box": QUESTIONS["box"], "plain": QUESTIONS["plain"]})
    branched = model.run(planned.items)
    for item, value in zip(planned.items, branched, strict=True):
        state = item.state
        images = model._state_media(state)
        ids = torch.as_tensor([item.ids], dtype=torch.long)
        rows = model._read_rows(
            [item], ids, torch.ones_like(ids), images, [len(state.ids)]
        )[0]
        got = value if item.readouts else [value]
        want = rows if item.readouts else [rows]
        gap = max(
            abs(a - b)
            for g, w in zip(got, want, strict=True)
            for a, b in zip(g, w, strict=True)
        )
        assert gap <= FP32_TOLERANCE, (item.question_id, gap)


def test_question_images_never_depend_on_other_questions(model):
    alone = plan(model, {"mine": QUESTIONS["mine"]})
    together = plan(model, QUESTIONS)
    value_alone = model.run(alone.items)[0]
    index = [item.question_id for item in together.items].index("mine")
    value_together = model.run(together.items)[index]
    assert value_alone == value_together  # bit-identical in FP32


def test_box_block_text_is_the_packages_prompt_format(model):
    text, used = composed.box_block(
        "the dog in <photo>", model.codes, composed.check_references(ASSETS), 3.25, True
    )
    assert used == ["<photo>"]
    assert text.endswith(composed.GENERATION + "visible:\ntop:\nbottom:\nleft:\nright:")
    assert (
        "in the video frame at 3.2 seconds" in text
        or "in the video frame at 3.3 seconds" in text
    )


def test_video_track_and_at_box(model, monkeypatch):
    import numpy as np
    from vllm_srun.families.decision3 import state_once
    from vllm_srun.families.decision3 import videos as vid

    def fake_windows(value, rate):
        fps, total = 30.0, 30 * 20  # a 20-s clip
        for number, window in enumerate(vid.window_plan(total, fps, rate)):
            frames = [
                np.full((64, 96, 3), (i * 7) % 255, dtype=np.uint8) for i in window
            ]
            yield vid._video(frames, list(window), fps, total, f"clip:w{rate}:{number}")

    monkeypatch.setattr(state_once.vid, "decode_windows", fake_windows)
    if (
        model.video_settings is None
    ):  # OpenCV absent here: the fake decoder stands in for it
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
    planned = model.plan_composed(
        "a clip",
        {
            "track": {"type": "track", "instructions": "the dog"},
            "at": {"type": "box", "instructions": "the dog", "at": 7.3},
        },
        [],
        ["data:video/mp4;base64,AAAA"],
        {},
    )
    assert not planned.errors, planned.errors
    tracks = [item for item in planned.items if item.task_type == "track"]
    assert (
        len(tracks) == 20
    )  # 1 update per second over 20 s (2 windows of 16 s at 2 fps)
    values = model.run(planned.items)

    class _Plan:
        state = planned

    response = model.finish_surface(_Plan(), values)
    track = response["answers"]["track"]
    assert track["type"] == "track" and len(track["track"]) == 20
    times = [entry["time"] for entry in track["track"]]
    assert times == sorted(times) and times[0] == 0.25 and times[-1] == 19.25
    assert all(0.0 <= entry["visible"] <= 1.0 for entry in track["track"])
    at = response["answers"]["at"]
    assert at["type"] == "box" and at["time"] == 7.25


def _resolved_request():
    from dataclasses import dataclass

    from vllm_srun.assets import Segments, Use

    @dataclass(frozen=True)
    class Asset:
        reference: str
        url: str

        def data_url(self):
            return self.url

    @dataclass(frozen=True)
    class Request:
        state: object
        questions: dict
        images: list
        videos: list
        uses: list

    photo, a = Asset("<photo>", PHOTO), Asset("<a>", CROP_A)
    inline = Use(a, "instructions", "inline")
    return Request(
        {"photo": Segments((Use(photo, "state"),))},
        {
            "inline": {
                "type": "noul",
                "instructions": Segments(("Is ", inline, " green?")),
            }
        },
        [],
        [],
        [Use(photo, "state"), inline],
    )


def test_plan_resolved_reads_the_api_layers_segments(model, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(
        model,
        "info",
        replace(model.info, image_placements=("state", "instructions", "criteria")),
    )
    via_api = model.plan_resolved(_resolved_request())
    direct = model.plan_composed(
        {"photo": "<photo>"}, {"inline": QUESTIONS["inline"]}, [], None, ASSETS
    )
    assert [item.ids for item in via_api.items] == [item.ids for item in direct.items]


def test_plan_resolved_refuses_question_images_the_card_does_not_place(
    model, monkeypatch
):
    from dataclasses import replace

    monkeypatch.setattr(model, "info", replace(model.info, image_placements=("state",)))
    plan = model.plan_resolved(_resolved_request())
    assert plan.question_ids == ["inline"] and not plan.items
    assert plan.errors["inline"]["reason"] == "question_images_unsupported"


# ------------------------------------------------------------------ Box and track through the API layer, no assets

CLIP = "data:video/mp4;base64,AAAA"  # not decoded: the window decoder stands in


@pytest.fixture()
def located(model, monkeypatch):
    """The card's Box and track, and a 20-s clip at 4 frames per second from a window decoder standing in for OpenCV."""
    import numpy as np
    from vllm_srun.families.decision3 import state_once
    from vllm_srun.families.decision3 import videos as vid

    def fake_windows(value, rate):
        fps, total = 4.0, 4 * 20
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
            model.info, question_types=(*model.info.question_types, "box", "track")
        ),
    )


def decide(model, body):
    """A decisions body's answers as the server gives them: assets.prepare, plan_surface, run, finish_surface."""
    prepared = assetlib.prepare(body, ServeConfig.max_request_bytes)
    plan = model.plan_surface(
        "decisions",
        SurfaceRequest("decisions", prepared, None, False, time.monotonic()),
    )
    return model.finish_surface(
        plan, model.run(list(plan.items)) if plan.items else []
    )["answers"]


def check_box(box, confidence):
    assert set(box) == set(composed.EDGES) and all(
        0.0 <= edge <= 1.0 for edge in box.values()
    )
    assert box["top"] < box["bottom"] and box["left"] < box["right"]
    assert 0.0 <= confidence <= 1.0


def test_track_and_at_box_without_assets(model, located):
    """A track at either rate and a box at a moment, over the request's one video, in the contract's answer shapes."""
    answers = decide(
        model,
        {
            "state": "A clip of a dog.",
            "videos": [CLIP],
            "questions": {
                "default": {"type": "track", "instructions": "the dog"},
                "r1": {"type": "track", "instructions": "the dog", "rate": 1},
                "r2": {"type": "track", "instructions": "the dog", "rate": 2},
                "at": {"type": "box", "instructions": "the dog", "at": 7.3},
            },
        },
    )
    for key in ("default", "r1", "r2"):
        assert (
            set(answers[key]) == {"type", "track"} and answers[key]["type"] == "track"
        ), answers[key]
        times = [entry["time"] for entry in answers[key]["track"]]
        assert times == sorted(times) and times[0] >= 0.0 and times[-1] < 20.0
        for entry in answers[key]["track"]:
            assert (
                set(entry) == {"time", "box", "confidence", "visible"}
                and 0.0 <= entry["visible"] <= 1.0
            )
            check_box(entry["box"], entry["confidence"])
    # One update per second over the 20 s by default and at rate 1, two at rate 2.
    assert len(answers["default"]["track"]) == len(answers["r1"]["track"]) == 20
    assert len(answers["r2"]["track"]) == 40
    at = answers["at"]
    assert set(at) == {"type", "box", "confidence", "time"} and at["type"] == "box", at
    assert at["time"] == 7.25  # the nearest frame group
    check_box(at["box"], at["confidence"])


def test_image_box_reads_the_photo_in_images(model, located):
    """An image Box whose photo is in ``images`` (no assets) is the family's composed Box, beside a plain question."""
    questions = {"box": QUESTIONS["box"], "plain": QUESTIONS["plain"]}
    answers = decide(
        model, {"state": "A photo.", "images": [PHOTO], "questions": questions}
    )
    box = answers["box"]
    assert set(box) == {"type", "box", "confidence"} and box["type"] == "box", box
    check_box(box["box"], box["confidence"])
    assert answers["plain"]["type"] == "noul" and "error" not in answers["plain"]
    direct = model.plan_composed("A photo.", questions, [PHOTO], [], {})
    values = model.run(direct.items)
    expected = {
        item.question_id: model.answer(item, value)
        for item, value in zip(direct.items, values, strict=True)
    }
    assert box == expected["box"]


@pytest.mark.parametrize("rate", [0, 3, -1, 1.5, "2", True, [2]])
def test_a_track_rate_other_than_one_or_two_fails_alone(model, located, rate):
    answers = decide(
        model,
        {
            "state": "A clip.",
            "videos": [CLIP],
            "questions": {
                "bad": {"type": "track", "instructions": "the dog", "rate": rate},
                "at": {"type": "box", "instructions": "the dog", "at": 0},
            },
        },
    )
    bad = answers["bad"]
    assert bad["type"] == "track" and bad["error"] == "invalid_question", bad
    assert bad["reason"] == "question_malformed" and "rate" in bad["message"]
    assert answers["at"]["time"] == 0.25 and "error" not in answers["at"]


def test_an_at_past_the_clip_fails_alone_with_its_reason(model, located):
    answers = decide(
        model,
        {
            "state": "A clip.",
            "videos": [CLIP],
            "questions": {
                "late": {"type": "box", "instructions": "the dog", "at": 21.5},
                "last": {"type": "box", "instructions": "the dog", "at": 20},
            },
        },
    )
    assert (
        answers["late"]["error"] == "invalid_question"
        and answers["late"]["reason"] == "at_out_of_range"
    )
    assert answers["last"]["time"] == 19.25
