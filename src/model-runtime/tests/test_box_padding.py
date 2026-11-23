"""An image Box on a wide, short photo, padded on its short axis (family options ``box_pad_rows``, #7011).

The Box reads a state of the photo padded with ``PAD_FILL`` to ``rows`` merged
rows and answers in the photo's own coordinates; every other question of the
request reads the photo as sent.
"""

from __future__ import annotations

import base64
import io

import pytest
import torch
from vllm_srun.config import EngineArgs
from vllm_srun.errors import PackageError
from vllm_srun.families.decision3 import composed
from vllm_srun.families.decision3 import images as img
from vllm_srun.families.decision3.family import box_padding
from vllm_srun.families.decision3.state_once import StateOnceModel

from .conftest import start_runtime

PADDING = img.BoxPadding(rows=8, min_aspect=4.0)
BOX = {"type": "box", "instructions": "the bright bar"}
PLAIN = {"type": "noul", "instructions": "Is the photo red?"}


def noise(width: int, height: int, seed: int = 0):
    from PIL import Image

    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=generator)
    return Image.fromarray(pixels.to(torch.uint8).numpy(), "RGB")


def url(picture) -> str:
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def padded_by_hand(picture, rows: int):
    """``picture`` centred on a ``PAD_FILL`` canvas ``rows`` merged rows high, as a client would pad it."""
    from PIL import Image

    width, height = picture.size
    target = rows * 32
    canvas = Image.new("RGB", (width, target), (img.PAD_FILL,) * 3)
    canvas.paste(picture, (0, (target - height) // 2))
    return canvas


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from vllm_srun.testing.decision3 import write_fixture

    package = write_fixture(
        tmp_path_factory.mktemp("box-padding") / "pkg", "prefix_bidir", seed=31
    )
    runtime = start_runtime(package, engine_args=EngineArgs(dtype="float32"))
    found = runtime.lookup(None).model
    assert isinstance(found, StateOnceModel)
    yield found
    runtime.stop()


def answers(model, questions, images):
    planned = model.plan_composed("A photo.", questions, images, [], {})
    assert not planned.errors, planned.errors
    values = model.run(planned.items)
    return planned, {
        item.question_id: model.answer(item, value)
        for item, value in zip(planned.items, values, strict=True)
    }


def test_padding_keeps_the_photos_patches_and_fills_the_rest(model):
    picture = noise(1792, 224)
    original = model.image(url(picture), "images[0]")
    padded, frame = img.pad_short_axis(original, PADDING, model.settings)
    by_hand = model.image(url(padded_by_hand(picture, 8)), "images[0]")
    assert padded.grid == by_hand.grid == (1, 16, 112)
    assert torch.equal(padded.pixel_values, by_hand.pixel_values)
    assert padded.digest != original.digest
    assert frame == (0.0, 16 / 256, 1.0, 240 / 256)


@pytest.mark.parametrize(
    ("width", "height", "padded"),
    [
        (1792, 224, True),  # 8:1 at 7 rows
        (896, 224, True),  # exactly 4:1
        (864, 224, False),  # under 4:1
        (2048, 256, False),  # already 8 rows
        (224, 1792, True),  # tall
        (7360, 224, False),  # padded past the pixel budget
    ],
)
def test_which_photos_are_padded(model, width, height, padded):
    image = model.image(url(noise(width, height, seed=width)), "images[0]")
    found = img.pad_short_axis(image, PADDING, model.settings)
    assert (found is not None) == padded
    if found is not None:
        assert min(found[0].grid[1:]) == 16


def test_unpad_maps_back_and_clamps():
    frame = (0.0, 0.25, 1.0, 0.75)
    box = {"top": 0.375, "bottom": 0.625, "left": 0.1, "right": 0.2}
    assert composed.unpad(box, frame) == {
        "top": 0.25,
        "bottom": 0.75,
        "left": 0.1,
        "right": 0.2,
    }
    outside = composed.unpad({**box, "top": 0.1, "bottom": 0.9}, frame)
    assert (outside["top"], outside["bottom"]) == (0.0, 1.0)
    inside_pad = composed.unpad({**box, "top": 0.05, "bottom": 0.1}, frame)
    assert 0.0 <= inside_pad["top"] < inside_pad["bottom"] <= 1.0


def test_box_answers_as_on_the_hand_padded_photo_and_others_are_untouched(
    model, monkeypatch
):
    picture = noise(1792, 224, seed=5)
    questions = {"box": BOX, "plain": PLAIN}
    unpadded, before = answers(model, questions, [url(picture)])
    _, by_hand = answers(model, {"box": BOX}, [url(padded_by_hand(picture, 8))])
    monkeypatch.setattr(model, "box_padding", PADDING)
    planned, after = answers(model, questions, [url(picture)])
    items = {item.question_id: item for item in planned.items}
    assert items["box"].state is not items["plain"].state
    assert items["box"].box_frame == (0.0, 16 / 256, 1.0, 240 / 256)
    assert items["plain"].box_frame is None
    # The other question reads the photo as sent: the same tokens and features; its value moves only with the
    # forward's batch, as it does between requests with and without the Box.
    plain = next(item for item in unpadded.items if item.question_id == "plain")
    assert items["plain"].ids == plain.ids
    assert after["plain"]["noul"] == pytest.approx(before["plain"]["noul"], abs=2e-5)
    # The Box is the hand-padded photo's Box in the photo's coordinates.
    want = composed.unpad(by_hand["box"]["box"], items["box"].box_frame)
    for edge in composed.EDGES:
        assert after["box"]["box"][edge] == pytest.approx(want[edge], abs=2e-5)
    assert after["box"]["confidence"] == pytest.approx(
        by_hand["box"]["confidence"], abs=2e-5
    )
    # Each state counts once.
    assert planned.input_tokens == len(items["plain"].ids) + len(items["box"].ids)


def test_off_by_default_and_for_photos_that_do_not_need_it(model, monkeypatch):
    wide = url(noise(1792, 224, seed=6))
    planned = model.plan_composed("A photo.", {"box": BOX}, [wide], [], {})
    assert planned.items[0].box_frame is None
    monkeypatch.setattr(model, "box_padding", PADDING)
    square = url(noise(256, 256, seed=7))
    planned = model.plan_composed("A photo.", {"box": BOX}, [square], [], {})
    assert planned.items[0].box_frame is None
    # Two images: which one the Box reads is the model's call, so neither is padded.
    planned = model.plan_composed("Two photos.", {"box": BOX}, [wide, wide], [], {})
    assert planned.items[0].box_frame is None


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, None),
        ({"box_pad_rows": 8}, img.BoxPadding(8, 4.0)),
        ({"box_pad_rows": 12, "box_pad_min_aspect": 6}, img.BoxPadding(12, 6.0)),
    ],
)
def test_box_padding_options(options, expected):
    assert box_padding(options) == expected


@pytest.mark.parametrize(
    "options",
    [
        {"box_pad_rows": 0},
        {"box_pad_rows": True},
        {"box_pad_rows": "8"},
        {"box_pad_rows": 8, "box_pad_min_aspect": 0.5},
        {"box_pad_rows": 8, "box_pad_min_aspect": float("nan")},
    ],
)
def test_box_padding_options_refused(options):
    with pytest.raises(PackageError):
        box_padding(options)


def test_a_padded_box_over_the_question_cap_reads_the_photo_as_sent(model, monkeypatch):
    wide = url(noise(1792, 224, seed=8))
    sent = model.plan_composed("A photo.", {"box": BOX}, [wide], [], {})
    monkeypatch.setattr(model, "box_padding", PADDING)
    padded = model.plan_composed("A photo.", {"box": BOX}, [wide], [], {})
    assert len(padded.items[0].ids) > len(sent.items[0].ids)
    monkeypatch.setattr(model, "validated", len(sent.items[0].ids))
    planned = model.plan_composed("A photo.", {"box": BOX}, [wide], [], {})
    assert not planned.errors
    assert planned.items[0].box_frame is None
    assert planned.items[0].ids == sent.items[0].ids


def test_a_padded_state_over_the_request_cap_reads_the_photo_as_sent(
    model, monkeypatch
):
    from dataclasses import replace

    wide = url(noise(1792, 224, seed=9))
    questions = {"box": BOX, "plain": PLAIN}
    sent = model.plan_composed("A photo.", questions, [wide], [], {})
    limits = {**model.info.limits, "max_request_tokens": sent.input_tokens}
    monkeypatch.setattr(model, "info", replace(model.info, limits=limits))
    monkeypatch.setattr(model, "box_padding", PADDING)
    planned = model.plan_composed("A photo.", questions, [wide], [], {})
    assert not planned.errors
    assert all(item.box_frame is None for item in planned.items)
    assert planned.input_tokens == sent.input_tokens
