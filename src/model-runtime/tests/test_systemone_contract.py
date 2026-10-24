"""System One compatibility: wire forms, assets and references, Box, reasons, confidence and long context."""

import asyncio
import base64
import gzip
import json
import math
import struct
import zlib

import pytest
from vllm_srun import assets as assetlib
from vllm_srun.api import wire
from vllm_srun.errors import MediaError, RequestError, RuntimeServiceError
from vllm_srun.plugins.decisions import as_systemone
from vllm_srun.systemone import (
    choice_confidence,
    read_box,
    read_question,
    score_confidence,
)

from .conftest import start_runtime


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


def zstd_compress(data):
    from backports.zstd import compress

    return compress(data)


def png_bytes(png_url, width=64, height=48, seed=0):
    return base64.b64decode(png_url(width, height, seed).split(",", 1)[1])


def asset(data):
    return {
        "type": "image",
        "format": "png",
        "data_base64": base64.b64encode(data).decode(),
    }


NOUL = {"type": "noul", "instructions": "Is it red?"}


# ------------------------------------------------------------------ confidence


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ((0.6, 0.3, 0.1), 0.4),
        ((0.6, 0.2, 0.2), 0.4),
        ((0.9, 0.1), 0.8),
        ((0.40, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05), (0.40 - 1 / 8) / (1 - 1 / 8)),
        ((0.77, 0.12, 0.07, 0.04), (0.77 - 0.25) / 0.75),
        ((0.25, 0.25, 0.25, 0.25), 0.0),
    ],
)
def test_choice_confidence_is_system_ones_published_formula(probabilities, expected):
    assert math.isclose(choice_confidence(probabilities), expected, abs_tol=1e-12)


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ((0.0, 0.57, 0.43), 1 - 0.43 / (2 / 3)),
        ((0.0, 0.5, 0.5), 0.25),
        ((0.5, 0.0, 0.5), 0.0),
        ((0.0, 0.05, 0.9, 0.05, 0.0), 1 - 0.1 / 1.2),
        ((1.0, 0.0), 1.0),
    ],
)
def test_score_confidence_measures_the_spread_around_the_most_likely_level(
    probabilities, expected
):
    assert math.isclose(score_confidence(probabilities), expected, abs_tol=1e-12)


def test_answers_carry_the_published_confidence(d3_runtime):
    status, body = call(
        d3_runtime,
        {
            "state": "Merge two sorted lists.",
            "questions": {
                "domain": {
                    "type": "choice",
                    "instructions": "Domain?",
                    "criteria": {"code": None, "math": None, "other": None},
                },
                "level": {
                    "type": "score",
                    "instructions": "How hard?",
                    "criteria": ["easy", "medium", "hard"],
                },
            },
        },
    )
    assert status == 200
    choice, score = body["answers"]["domain"], body["answers"]["level"]
    assert math.isclose(
        choice["confidence"],
        choice_confidence(list(choice["probabilities"].values())),
        abs_tol=1e-12,
    )
    assert math.isclose(
        score["confidence"],
        score_confidence([score["probabilities"][str(i)] for i in range(3)]),
        abs_tol=1e-12,
    )


# ------------------------------------------------------------------ questions


def test_choice_and_score_may_omit_instructions(d3_runtime):
    status, body = call(
        d3_runtime,
        {
            "state": "Buy cheap watches now!!!",
            "questions": {
                "tone": {"type": "choice", "criteria": {"calm": None, "pushy": None}},
                "null": {
                    "type": "choice",
                    "instructions": None,
                    "criteria": {"calm": None, "pushy": None},
                },
                "level": {"type": "score", "criteria": ["low", "high"]},
                "spam": {
                    "type": "noul",
                    "instructions": None,
                    "criteria": {"true": "Unsolicited advertising"},
                },
                "bare": {"type": "noul"},
            },
        },
    )
    assert status == 200
    answers = body["answers"]
    assert answers["tone"]["choice"] in ("calm", "pushy")
    assert answers["null"]["probabilities"] == answers["tone"]["probabilities"]
    assert "score" in answers["level"] and 0 <= answers["spam"]["noul"] <= 1
    assert answers["bare"]["error"] == "invalid_question"
    assert answers["bare"]["reason"] == "question_malformed"


def test_box_questions_take_instructions_and_a_time():
    assert read_box({"type": "box", "instructions": "the mug", "at": 2}).at == 2.0
    assert read_box({"type": "box", "instructions": "the mug", "at": 0}).at == 0.0
    assert read_box({"type": "track", "instructions": "the mug"}).kind == "track"
    for bad in (
        {"type": "box"},
        {"type": "box", "instructions": "x", "criteria": {"a": None}},
        {"type": "box", "instructions": "x", "at": -1},
        {"type": "box", "instructions": "x", "at": True},
        {"type": "box", "instructions": "x", "at": "2"},
        {"type": "box", "instructions": "x", "at": math.inf},
        {"type": "box", "instructions": "x", "rate": 1},
        {"type": "track", "instructions": "x", "at": 1},
    ):
        with pytest.raises(ValueError) as caught:
            read_box(bad)
        assert caught.value.reason == "question_malformed"


def test_track_questions_take_a_rate_of_one_or_two_updates_per_second():
    track = {"type": "track", "instructions": "the mug"}
    assert read_box(track).rate == 1
    assert read_box({**track, "rate": None}).rate == 1
    assert read_box({**track, "rate": 1}).rate == 1
    assert read_box({**track, "rate": 2}).rate == 2
    assert read_box({**track, "rate": 2.0}).rate == 2
    for rate in (0, 3, -1, 1.5, math.nan, "2", True, [2], {"rate": 2}):
        with pytest.raises(ValueError, match="rate") as caught:
            read_box({**track, "rate": rate})
        assert caught.value.reason == "question_malformed"


def test_unknown_types_have_their_own_reason():
    with pytest.raises(ValueError) as caught:
        read_question({"type": "rank", "instructions": "x"})
    assert caught.value.reason == "type_unsupported"


def test_box_and_track_fail_alone_on_a_model_without_them(d3_runtime, png_url):
    status, body = call(
        d3_runtime,
        {
            "state": "A photo.",
            "images": [png_url(64, 48, 1)],
            "questions": {
                "mug": {"type": "box", "instructions": "The mug."},
                "path": {"type": "track", "instructions": "The mug."},
                "red": NOUL,
            },
        },
    )
    assert status == 200
    answers = body["answers"]
    assert list(answers) == ["mug", "path", "red"]
    assert answers["mug"] == {
        "type": "box",
        "error": "invalid_question",
        "reason": "box_unsupported",
        "message": answers["mug"]["message"],
    }
    assert answers["path"]["reason"] == "track_unsupported"
    assert 0 <= answers["red"]["noul"] <= 1


def test_every_failed_answer_has_a_fixed_reason(d3_runtime):
    status, body = call(
        d3_runtime,
        {
            "state": "x " * 5000,
            "questions": {
                "long": NOUL,
                "bad": {"type": "score", "instructions": "x", "criteria": ["one"]},
            },
        },
    )
    assert status == 200
    assert body["answers"]["long"]["reason"] == "question_tokens"
    assert body["answers"]["bad"]["reason"] == "question_malformed"


# ------------------------------------------------------------------ System One errors


def test_systemone_answers_a_failed_question_as_the_whole_request():
    body = {
        "model": "m",
        "answers": {
            "ok": {"type": "noul", "noul": 0.5},
            "bad": {
                "type": "box",
                "error": "invalid_question",
                "reason": "box_unsupported",
                "message": "no box",
            },
        },
        "usage": {"input_tokens": 3, "output_tokens": 0},
    }
    status, error = as_systemone(200, body)
    assert status == 422
    assert error["error"]["code"] == "question_failed"
    assert error["error"]["question_error"] == "invalid_question"
    assert error["error"]["reason"] == "box_unsupported"
    assert error["error"]["question"] == "bad"
    assert "questions.bad" in error["error"]["message"]
    long = {
        "answers": {
            "q": {
                "type": "noul",
                "error": "max_length_exceeded",
                "reason": "question_tokens",
            }
        }
    }
    assert as_systemone(200, long)[0] == 400
    states = {
        "answers": {"q": {"type": "noul", "noul": 0.1}},
        "states": {
            "s": {
                "answers": {
                    "r": {
                        "type": "noul",
                        "error": "deadline_exceeded",
                        "reason": "deadline",
                    }
                }
            }
        },
    }
    status, error = as_systemone(200, states)
    assert (status, error["error"]["question"]) == (504, "r")
    assert "states.s.questions.r" in error["error"]["message"]
    fine = {"answers": {"q": {"type": "noul", "noul": 0.1}}}
    assert as_systemone(200, fine) == (200, fine)


def test_a_request_without_a_valid_question_names_its_first_one(d3_runtime):
    status, body = call(
        d3_runtime,
        {"state": "x", "questions": {"a": {"type": "box", "instructions": "x"}}},
    )
    assert status == 400
    assert (
        body["error"]["reason"] == "box_unsupported"
        and body["error"]["question"] == "a"
    )
    status, body = as_systemone(status, body)
    assert status == 422 and body["error"]["question_error"] == "invalid_question"


# ------------------------------------------------------------------ wire forms


def test_content_encodings_follow_system_one():
    assert wire.content_encoding([]) is None
    assert wire.content_encoding(["identity"]) is None
    assert wire.content_encoding(["GZIP"]) == "gzip"
    assert wire.content_encoding(["x-gzip"]) == "gzip"
    assert wire.content_encoding([" zstd "]) == "zstd"
    for values in (["br"], ["gzip, gzip"], ["gzip", "zstd"], ["deflate"]):
        with pytest.raises(RuntimeServiceError) as caught:
            wire.content_encoding(values)
        assert (caught.value.status, caught.value.reason) == (
            415,
            "encoding_unsupported",
        )


def test_bodies_decode_within_the_limit_once_decompressed():
    data = json.dumps({"state": "x" * 4000}).encode()
    assert wire.decode(gzip.compress(data), "gzip", 1 << 20) == data
    assert wire.decode(zstd_compress(data), "zstd", 1 << 20) == data
    assert (
        wire.decode(zstd_compress(data) + zstd_compress(b" "), "zstd", 1 << 20)
        == data + b" "
    )
    with pytest.raises(RuntimeServiceError) as caught:
        wire.decode(zstd_compress(bytes(5000)), "zstd", 4096)
    assert caught.value.status == 413
    with pytest.raises(RuntimeServiceError) as caught:
        wire.decode(zstd_compress(data)[:-4], "zstd", 1 << 20)
    assert caught.value.reason == "encoding_malformed"
    with pytest.raises(RuntimeServiceError) as caught:
        wire.decode(gzip.compress(b"0" * 5000), "gzip", 4096)
    assert caught.value.status == 413
    for coding, raw in (
        ("gzip", b"\x1f\x8b" + bytes(30)),
        ("zstd", b"\x28\xb5\x2f\xfd" + bytes(30)),
    ):
        with pytest.raises(RuntimeServiceError) as caught:
            wire.decode(raw, coding, 1 << 20)
        assert (caught.value.status, caught.value.reason) == (400, "encoding_malformed")
    from backports.zstd import CompressionParameter, ZstdCompressor

    wide = ZstdCompressor(options={CompressionParameter.window_log: 24}).compress(
        bytes(1 << 25), mode=ZstdCompressor.FLUSH_FRAME
    )
    with pytest.raises(RuntimeServiceError) as caught:
        wire.decode(wide, "zstd", 1 << 26)
    assert (
        caught.value.reason == "encoding_malformed" and "8 MiB" in caught.value.message
    )


def multipart_body(parts, mark="b0undary"):
    chunks = []
    for name, content_type, data in parts:
        chunks.append(
            f'--{mark}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'.encode()
        )
        if content_type:
            chunks.append(f"Content-Type: {content_type}\r\n".encode())
        chunks += [b"\r\n", data, b"\r\n"]
    chunks.append(f"--{mark}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={mark}"


def test_multipart_reads_the_request_and_its_image_parts(png_url):
    image = png_bytes(png_url)
    request = {
        "state": {"photo": "<p>"},
        "questions": {"q": NOUL},
        "assets": {"<p>": {"type": "image", "part": "asset-0"}},
    }
    raw, content_type = multipart_body(
        [
            ("request", "application/json", json.dumps(request).encode()),
            ("asset-0", "image/png", image),
        ]
    )
    first, files = wire.multipart(raw, wire.boundary(content_type))
    body = json.loads(first)
    wire.attach_parts(body, files)
    assert body["assets"]["<p>"] == {"type": "image", "data": image}


@pytest.mark.parametrize(
    "parts, assets, message",
    [
        (
            [("asset-0", "image/png", b"x"), ("request", "application/json", b"{}")],
            {},
            "first multipart part",
        ),
        ([("request", "text/plain", b"{}")], {}, "application/json"),
        (
            [
                ("request", None, b"{}"),
                ("a", "image/png", b"x"),
                ("a", "image/png", b"y"),
            ],
            {"r": {"part": "a"}},
            "more than once",
        ),
        (
            [("request", None, b"{}"), ("a", "image/gif", b"x")],
            {"r": {"part": "a"}},
            "content type",
        ),
        (
            [("request", None, b"{}"), ("a", "image/png", b"x")],
            {},
            "claimed by no asset",
        ),
        (
            [("request", None, b"{}"), ("a", "image/png", b"x")],
            {"r": {"part": "b"}},
            "does not have",
        ),
        (
            [("request", None, b"{}"), ("a", "image/png", b"x")],
            {"r": {"part": "a"}, "s": {"part": "a"}},
            "claimed twice",
        ),
    ],
)
def test_malformed_multipart_bodies_are_refused(parts, assets, message):
    raw, content_type = multipart_body(parts)
    with pytest.raises(RequestError) as caught:
        _, files = wire.multipart(raw, wire.boundary(content_type))
        wire.attach_parts({"assets": assets}, files)
    assert caught.value.reason == "multipart_malformed" and message in str(caught.value)


# ------------------------------------------------------------------ assets


def test_assets_follow_the_reference_rules(png_url):
    image = png_bytes(png_url)
    body = {"state": "see", "questions": {"q": NOUL}}
    cases = [
        ({"assets": {}}, "asset_malformed"),
        ({"assets": {" ": asset(image)}, "state": " "}, "asset_reference"),
        ({"assets": {"r" * 1025: asset(image)}}, "asset_reference"),
        (
            {"assets": {f"{i:04d}" + "r" * 1016: asset(image) for i in range(65)}},
            "asset_reference",
        ),
        ({"assets": {"<unused>": asset(image)}}, "asset_unreferenced"),
        (
            {
                "assets": {"<p>": {"type": "video", "data_base64": "AA=="}},
                "state": "<p>",
            },
            "asset_malformed",
        ),
        ({"assets": {"<p>": {"type": "image"}}, "state": "<p>"}, "asset_malformed"),
        (
            {"assets": {"<p>": {"part": "asset-0"}}, "state": "<p>"},
            "multipart_malformed",
        ),
        (
            {"assets": {"<p>": {**asset(image), "extra": 1}}, "state": "<p>"},
            "asset_malformed",
        ),
    ]
    for change, reason in cases:
        with pytest.raises((RequestError, MediaError)) as caught:
            assetlib.prepare({**body, **change}, 1 << 26)
        assert caught.value.reason == reason, change.keys()


def test_assets_are_read_from_their_bytes(png_url):
    image = png_bytes(png_url, 300, 260)
    prepared = assetlib.prepare(
        {
            "state": "<p>",
            "questions": {"q": NOUL},
            "assets": {"<p>": {**asset(image), "format": "jpeg"}},
        },
        1 << 26,
    )
    checked = prepared["assets"]["<p>"]
    assert (checked.format, checked.width, checked.height) == ("png", 300, 260)
    gif = b"GIF89a" + bytes(20)
    animated = (
        image[:33]
        + struct.pack(">I", 8)
        + b"acTL"
        + bytes(8)
        + struct.pack(">I", zlib.crc32(b"acTL" + bytes(8)))
        + image[33:]
    )
    for data, reason in (
        (gif, "image_format"),
        (animated, "image_format"),
        (b"\xff\xd8\xff" + bytes(8), "image_malformed"),
        (bytes((16 << 20) + 1), "image_too_large"),
    ):
        with pytest.raises(MediaError) as caught:
            assetlib.prepare(
                {
                    "state": "<p>",
                    "questions": {"q": NOUL},
                    "assets": {"<p>": asset(data)},
                },
                1 << 26,
            )
        assert caught.value.reason == reason
    with pytest.raises(MediaError) as caught:
        assetlib.prepare(
            {
                "state": "<p>",
                "questions": {"q": NOUL},
                "assets": {"<p>": {"data_base64": "not base64!"}},
            },
            1 << 26,
        )
    assert caught.value.reason == "image_malformed"


def test_every_use_counts_toward_the_request_size(png_url):
    image = png_bytes(png_url)
    body = {
        "state": ["<p>"] * 4,
        "questions": {"q": NOUL},
        "assets": {"<p>": asset(image)},
    }
    assert assetlib.prepare(body, 4 * len(image))
    with pytest.raises(RequestError) as caught:
        assetlib.prepare(body, 4 * len(image) - 1)
    assert caught.value.reason == "asset_uses"


def test_references_match_whole_values_then_the_longest_left_to_right(png_url):
    image = png_bytes(png_url)
    checked = assetlib.prepare(
        {
            "state": "x",
            "questions": {"q": NOUL},
            "assets": {"<a>": asset(image), "<a>b": asset(image), "<c>": asset(image)},
            "states": {
                "s": {"state": "<a> <a>b<c><c> {<a>}", "questions": {"q": NOUL}}
            },
        },
        1 << 26,
    )["assets"]
    finder = assetlib.Finder(checked)
    whole = finder.split("<a>", "state", None)
    assert [part.asset.reference for part in whole.parts] == ["<a>"]
    mixed = finder.split("see <a> and <a>b<c><c>.", "state", None)
    assert [p if isinstance(p, str) else p.asset.reference for p in mixed.parts] == [
        "see ",
        "<a>",
        " and ",
        "<a>b",
        "<c>",
        "<c>",
        ".",
    ]
    assert finder.split("nothing here", "state", None) == "nothing here"
    resolved = assetlib.resolve(
        {"<a>": "<c>", "list": ["<a>"]},
        {
            "q": {
                "type": "choice",
                "instructions": "<c>?",
                "criteria": {"<a>": "<a>b", "x": None},
            },
            "s": {"type": "score", "levels": ["<c>", "low"]},
            "n": {"type": "noul", "choices": [{"key": "true", "description": "<a>"}]},
        },
        checked,
    )
    assert list(resolved.state) == ["<a>", "list"]
    assert [
        (use.placement, use.question, use.asset.reference) for use in resolved.uses
    ] == [
        ("state", None, "<c>"),
        ("state", None, "<a>"),
        ("instructions", "q", "<c>"),
        ("criteria", "q", "<a>b"),
        ("criteria", "s", "<c>"),
        ("criteria", "n", "<a>"),
    ]


def test_state_assets_read_as_the_same_images_in_front_of_the_state(
    d3_runtime, png_url
):
    first, second = png_url(64, 48, 3), png_url(48, 64, 4)
    questions = {
        "red": NOUL,
        "same": {"type": "noul", "instructions": "Are both photos the same?"},
    }
    as_images = call(
        d3_runtime,
        {
            "state": {"before": "[image 1]", "after": "see [image 2] here"},
            "images": [first, second],
            "questions": questions,
        },
    )
    as_assets = call(
        d3_runtime,
        {
            "state": {"before": "<b>", "after": "see <a> here"},
            "assets": {
                "<b>": {"data_base64": first.split(",", 1)[1]},
                "<a>": {"data_base64": second.split(",", 1)[1]},
            },
            "questions": questions,
        },
    )
    assert as_images == as_assets
    assert as_assets[0] == 200
    after_images = call(
        d3_runtime,
        {
            "state": "next to <a>",
            "images": [first],
            "assets": {"<a>": {"data_base64": second.split(",", 1)[1]}},
            "questions": questions,
        },
    )
    expected = call(
        d3_runtime,
        {
            "state": "next to [image 2]",
            "images": [first, second],
            "questions": questions,
        },
    )
    assert after_images == expected


def test_question_images_fail_alone_on_a_model_that_reads_images_in_the_state(
    d3_runtime, png_url
):
    image = png_url(64, 48, 5).split(",", 1)[1]
    status, body = call(
        d3_runtime,
        {
            "state": "A menu.",
            "assets": {"<lobster>": {"data_base64": image}},
            "questions": {
                "friendly": {
                    "type": "noul",
                    "instructions": "Is the <lobster> friendly?",
                },
                "menu": {"type": "noul", "instructions": "Is this a menu?"},
                "pick": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"a": "<lobster>", "b": None},
                },
            },
        },
    )
    assert status == 200
    assert body["answers"]["friendly"]["reason"] == "question_images_unsupported"
    assert body["answers"]["pick"]["reason"] == "question_images_unsupported"
    alone = call(
        d3_runtime,
        {
            "state": "A menu.",
            "questions": {"menu": {"type": "noul", "instructions": "Is this a menu?"}},
        },
    )[1]
    assert body["answers"]["menu"] == alone["answers"]["menu"]


def test_a_rejected_asset_is_named_by_its_reference(d3_runtime, png_url):
    tiny = png_url(64, 48, 6).split(",", 1)[1]
    wide = base64.b64encode(png_bytes(png_url, 4000, 10, 7)).decode()
    status, body = call(
        d3_runtime,
        {
            "state": "<w> and <t>",
            "assets": {"<t>": {"data_base64": tiny}, "<w>": {"data_base64": wide}},
            "questions": {"q": NOUL},
        },
    )
    assert status == 400
    assert body["error"]["reason"] == "image_aspect_ratio"
    assert body["error"]["message"].startswith("assets['<w>']")


def test_assets_need_a_model_that_reads_images(d3_text_package, png_url):
    runtime = start_runtime(d3_text_package)
    try:
        image = png_url(64, 48, 8).split(",", 1)[1]
        status, body = call(
            runtime,
            {
                "state": "<p>",
                "assets": {"<p>": {"data_base64": image}},
                "questions": {"q": NOUL},
            },
        )
        assert (status, body["error"]["reason"]) == (400, "image_unsupported")
    finally:
        runtime.stop()


def test_states_share_the_requests_assets(d3_runtime, png_url):
    image = png_url(64, 48, 9).split(",", 1)[1]
    status, body = call(
        d3_runtime,
        {
            "state": "plain text",
            "questions": {"q": NOUL},
            "assets": {"<p>": {"data_base64": image}},
            "states": {"s": {"state": "<p>", "questions": {"q": NOUL}}},
        },
    )
    assert status == 200 and "noul" in body["states"]["s"]["answers"]["q"]


def test_the_card_says_where_images_are_read(d3_runtime):
    card = d3_runtime.lookup(None).card([])
    assert card["image_placements"] == ["state"]
    assert card["limits"]["image_max_bytes"] == 16 << 20
    assert "validated_input_tokens" not in card["limits"]


# ------------------------------------------------------------------ long context


@pytest.fixture(scope="module")
def validated_runtime(fixture_root):
    from vllm_srun.testing.decision3 import VALIDATED_LENGTH, write_fixture

    runtime = start_runtime(
        write_fixture(fixture_root / "d3-validated", "validated", seed=14)
    )
    yield runtime, VALIDATED_LENGTH
    runtime.stop()


def test_questions_past_the_validated_length_need_long_context(validated_runtime):
    runtime, validated = validated_runtime
    card = runtime.lookup(None).card([])
    assert (
        card["limits"]["validated_input_tokens"]
        == validated
        < card["limits"]["max_input_tokens"]
    )
    body = {"state": "word " * validated, "questions": {"q": NOUL}}
    status, answer = call(runtime, body)
    assert status == 200 and answer["answers"]["q"]["reason"] == "question_tokens"
    assert "long context" in answer["answers"]["q"]["message"]
    status, answer = call(runtime, {**body, "options": {"long_context": True}})
    assert status == 200 and 0 <= answer["answers"]["q"]["noul"] <= 1
    status, answer = call(runtime, {**body, "options": {"long_context": "yes"}})
    assert status == 400


# ------------------------------------------------------------------ the HTTP layer


@pytest.fixture(scope="module")
def client(d3_runtime):
    from starlette.testclient import TestClient
    from vllm_srun.api.app import create_app

    with TestClient(create_app(d3_runtime)) as test_client:
        yield test_client


def test_the_server_reads_compressed_and_multipart_bodies(client, png_url):
    image = png_bytes(png_url, 64, 48, 10)
    request = {
        "state": {"photo": "<p>"},
        "questions": {"q": NOUL},
        "assets": {"<p>": {"type": "image", "part": "asset-0"}},
    }
    raw, content_type = multipart_body(
        [
            ("request", "application/json", json.dumps(request).encode()),
            ("asset-0", "image/png", image),
        ]
    )
    plain = client.post(
        "/v1/systemone", content=raw, headers={"Content-Type": content_type}
    )
    packed = client.post(
        "/v1/systemone",
        content=zstd_compress(raw),
        headers={"Content-Type": content_type, "Content-Encoding": "zstd"},
    )
    as_json = client.post(
        "/v1/systemone",
        json={
            **request,
            "assets": {"<p>": {"data_base64": base64.b64encode(image).decode()}},
        },
    )
    assert plain.status_code == packed.status_code == as_json.status_code == 200
    assert plain.json() == packed.json() == as_json.json()
    gzipped = client.post(
        "/v1/decisions",
        content=gzip.compress(
            json.dumps({"state": "x", "questions": {"q": NOUL}}).encode()
        ),
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert gzipped.status_code == 200
    refused = client.post(
        "/v1/decisions",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Encoding": "br"},
    )
    assert (
        refused.status_code,
        refused.json()["error"]["code"],
        refused.json()["error"]["reason"],
    ) == (415, "unsupported_media_type", "encoding_unsupported")


def test_systemone_fails_the_request_and_decisions_the_question(client):
    body = {
        "state": "x",
        "questions": {"q": NOUL, "b": {"type": "box", "instructions": "x"}},
    }
    decisions = client.post("/v1/decisions", json=body)
    systemone = client.post("/v1/systemone", json=body)
    assert (
        decisions.status_code == 200
        and decisions.json()["answers"]["b"]["reason"] == "box_unsupported"
    )
    assert systemone.status_code == 422
    assert systemone.json()["error"] | {"message": None} == {
        "code": "question_failed",
        "question_error": "invalid_question",
        "reason": "box_unsupported",
        "question": "b",
        "message": None,
    }


def test_long_context_reads_past_the_validated_length(validated_runtime):
    from starlette.testclient import TestClient
    from vllm_srun.api.app import create_app

    runtime, validated = validated_runtime
    body = {"state": "word " * validated, "questions": {"q": NOUL}}
    with TestClient(create_app(runtime)) as test_client:
        refused = test_client.post("/v1/systemone", json=body)
        answered = test_client.post(
            "/v1/systemone", json={**body, "options": {"long_context": True}}
        )
    assert (refused.status_code, refused.json()["error"]["reason"]) == (
        400,
        "question_tokens",
    )
    assert answered.status_code == 200


def test_models_lists_decision_models_in_the_system_one_shape(client):
    listing = client.get("/v1/models").json()
    assert listing["api_version"] == "3.2.0"
    assert [model["name"] for model in listing["models"]] == [
        card["id"] for card in listing["data"]
    ]
    assert set(listing["models"][0]) == {"name", "description", "release_date"}


def test_an_unknown_model_names_the_served_ones(client):
    answer = client.post(
        "/v1/systemone",
        json={"model": "unserved-model", "state": "x", "questions": {"q": NOUL}},
    )
    assert answer.status_code == 404
    assert "unserved-model" in answer.json()["error"]["message"]
