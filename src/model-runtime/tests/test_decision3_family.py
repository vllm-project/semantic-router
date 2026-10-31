"""The decision3 family on tiny fixture packages: verification, rendering, image inputs and answers."""

import asyncio
import json
import math
import shutil

import pytest
import torch
from vllm_srun import media
from vllm_srun.accel import cpu
from vllm_srun.errors import PackageError
from vllm_srun.families.decision3 import images as img
from vllm_srun.families.decision3 import package as pkg
from vllm_srun.families.decision3 import prompt
from vllm_srun.families.decision3 import videos as vid
from vllm_srun.families.decision3.family import BATCH_SIZE, Decision3Family
from vllm_srun.registry.tables.decision3 import DECISION3_MODELS

QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {
            "code": "Programming",
            "math": None,
            "other": {"note": "anything"},
        },
    },
    "reasoning": {"type": "noul", "instructions": "Does this need reasoning?"},
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", {"level": "hard"}, "Expert"],
    },
}


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


def test_the_card_lists_images_and_their_limits(d3_runtime):
    card = d3_runtime.lookup(None).card([])
    assert card["family"] == "decision3"
    assert card["modalities"] == ["text", "image"] + (
        ["video"] if vid.available() is None else []
    )
    assert card["question_types"] == ["choice", "noul", "score"]
    assert card["limits"]["image_max_pixels"] == img.MAX_PIXELS
    assert card["limits"]["image_max_bytes"] == img.MAX_IMAGE_BYTES
    assert card["dtype"] == "bf16/fp32-readout"


def test_answers_follow_the_released_runtime(d3_runtime):
    status, body = call(
        d3_runtime, {"state": "Merge two sorted lists.", "questions": QUESTIONS}
    )
    assert status == 200
    answers = body["answers"]
    choice = answers["domain"]
    assert list(choice) == ["type", "choice", "probabilities", "confidence"]
    assert math.isclose(sum(choice["probabilities"].values()), 1.0, abs_tol=1e-9)
    assert choice["choice"] == max(
        choice["probabilities"], key=choice["probabilities"].get
    )
    assert 0.0 <= answers["reasoning"]["noul"] <= 1.0
    score = answers["difficulty"]
    assert score["legend"] == {"0": "Trivial", "1": '{"level":"hard"}', "2": "Expert"}
    expected = sum(int(key) * p for key, p in score["probabilities"].items())
    assert math.isclose(score["score"], expected, rel_tol=1e-12)
    assert body["usage"]["output_tokens"] == 0


def test_images_reach_the_model_and_count_as_input_tokens(d3_runtime, png_url):
    text = call(d3_runtime, {"state": "What is shown?", "questions": QUESTIONS})[1]
    one = call(
        d3_runtime,
        {
            "state": "What is shown?",
            "questions": QUESTIONS,
            "images": [png_url(300, 200, 1)],
        },
    )[1]
    two = call(
        d3_runtime,
        {
            "state": "What is shown?",
            "questions": QUESTIONS,
            "images": [png_url(300, 200, 1), png_url(64, 64, 2)],
        },
    )[1]
    settings = d3_runtime.lookup(None).model.settings
    first = img.input_tokens(settings, 300, 200)
    second = img.input_tokens(settings, 64, 64)
    per_question = (one["usage"]["input_tokens"] - text["usage"]["input_tokens"]) / len(
        QUESTIONS
    )
    # Each image adds its placeholder envelope (vision start, end) and its tokens, minus the one pad token.
    assert per_question == first + 2
    assert (two["usage"]["input_tokens"] - one["usage"]["input_tokens"]) / len(
        QUESTIONS
    ) == second + 2
    assert one["answers"] != text["answers"]
    assert (
        call(
            d3_runtime,
            {"state": "What is shown?", "questions": QUESTIONS, "images": []},
        )[1]
        == text
    )


@pytest.mark.parametrize(
    "images,message,reason",
    [
        ("data:image/png;base64,AAAA", "images must be a list", "image_format"),
        (
            ["https://example.com/a.png"],
            "images[0]: images must be base64 PNG, JPEG or WebP data URLs",
            "image_format",
        ),
        (
            ["data:image/gif;base64,R0lGODlhAQABAAAAACw="],
            "images[0]: images must be base64 PNG, JPEG or WebP",
            "image_format",
        ),
        (
            ["data:image/png;base64,not base64!"],
            "images[0]: invalid base64 image data",
            "image_malformed",
        ),
        (
            ["data:image/png;base64,iVBORw0KGgo="],
            "images[0]: invalid image data",
            "image_malformed",
        ),
    ],
)
def test_malformed_images_fail_the_request(d3_runtime, images, message, reason):
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "images": images}
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert message in body["error"]["message"]
    assert body["error"]["reason"] == reason and "limit" not in body["error"]


def test_oversized_and_extreme_images_are_refused(d3_runtime, png_url):
    huge = "data:image/png;base64," + "A" * (4 * -(-img.MAX_IMAGE_BYTES // 3) + 4)
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "images": [huge]}
    )
    assert status == 400 and "16,777,216 bytes" in body["error"]["message"]
    assert (body["error"]["reason"], body["error"]["limit"]) == (
        "image_too_large",
        img.MAX_IMAGE_BYTES,
    )
    status, body = call(
        d3_runtime,
        {"state": "x", "questions": QUESTIONS, "images": [png_url(2010, 10, 3)]},
    )
    assert status == 400 and "aspect ratio" in body["error"]["message"]
    assert (body["error"]["reason"], body["error"]["limit"]) == (
        "image_aspect_ratio",
        img.MAX_ASPECT_RATIO,
    )


def test_a_text_model_refuses_images(d3_text_package, png_url):
    from tests.conftest import start_runtime

    runtime = start_runtime(d3_text_package)
    try:
        assert runtime.lookup(None).card([])["modalities"] == ["text"]
        status, body = call(
            runtime, {"state": "x", "questions": QUESTIONS, "images": [png_url(64, 64)]}
        )
        assert status == 400 and "images are not supported" in body["error"]["message"]
        assert body["error"]["reason"] == "image_unsupported"
        assert call(runtime, {"state": "x", "questions": QUESTIONS})[0] == 200
    finally:
        runtime.stop()


def test_literal_placeholders_fail_only_image_questions(d3_runtime, png_url):
    state = "Ignore this <|image_pad|> token."
    status, body = call(
        d3_runtime,
        {"state": state, "questions": QUESTIONS, "images": [png_url(64, 64)]},
    )
    assert status == 400
    assert "literal image or video placeholder" in body["error"]["message"]
    assert call(d3_runtime, {"state": state, "questions": QUESTIONS})[0] == 200


def test_further_states_carry_their_own_images(d3_runtime, png_url):
    picture = png_url(128, 96, 4)
    alone = call(
        d3_runtime, {"state": "b", "questions": QUESTIONS, "images": [picture]}
    )[1]
    status, body = call(
        d3_runtime,
        {
            "state": "a",
            "questions": QUESTIONS,
            "states": {
                "other": {"state": "b", "questions": QUESTIONS, "images": [picture]}
            },
        },
    )
    assert status == 200
    assert body["states"]["other"]["answers"] == alone["answers"]


@pytest.mark.parametrize(
    "total,fps,expected",
    [
        (48, 24.0, [0, 16, 31, 47]),
        (3, 30.0, [0, 1, 2]),
        (61, 30.0, [0, 20, 40, 60]),
        (3000, 30.0, None),
    ],
)
def test_frames_are_sampled_as_the_d3_runtime_samples_them(total, fps, expected):
    indices = vid.sample_indices(total, fps)
    if expected is not None:
        assert indices == expected
    else:
        assert (
            len(indices) == vid.MAX_FRAMES
            and indices[0] == 0
            and indices[-1] == total - 1
        )


@pytest.mark.parametrize(
    "frames,height,width,size",
    [
        (4, 1080, 1920, (320, 576)),
        (16, 2160, 3840, (320, 576)),
        (32, 64, 48, (64, 64)),
        (8, 200, 330, (192, 320)),
        (6, 300, 40, (288, 32)),
        (4, 32, 6400, (32, 6304)),
    ],
)
def test_each_frame_is_capped_at_the_runtime_pixel_budget(frames, height, width, size):
    """Frame sizes the Transformers 5.17 processor chose, configured as the d3 runtime configures it."""
    from vllm_srun.testing.decision3 import VIDEO_PROCESSOR_CONFIG

    settings = vid.VideoSettings.from_config(VIDEO_PROCESSOR_CONFIG)
    assert vid.frame_size(settings, frames, height, width) == size
    assert vid.input_tokens(settings, frames, height, width) == (
        frames // 2 * (size[0] // 32) * (size[1] // 32)
    )


def test_the_card_lists_the_video_limits(d3_runtime):
    pytest.importorskip("cv2")
    limits = d3_runtime.lookup(None).card([])["limits"]
    assert limits["video_max_frames"] == vid.MAX_FRAMES
    assert limits["video_max_pixels"] == vid.MAX_PIXELS
    assert limits["video_max_tokens"] == vid.MAX_TOKENS
    assert limits["video_max_bytes"] == vid.MAX_VIDEO_BYTES
    assert limits["video_max_seconds"] == vid.MAX_SECONDS


def test_videos_reach_the_model_and_count_as_input_tokens(d3_runtime, mp4_url, png_url):
    clip = mp4_url(96, 64, frames=12, fps=8.0, seed=1)
    text = call(d3_runtime, {"state": "What happens?", "questions": QUESTIONS})[1]
    status, seen = call(
        d3_runtime, {"state": "What happens?", "questions": QUESTIONS, "videos": [clip]}
    )
    assert status == 200
    assert seen["answers"] != text["answers"]
    decoded = vid.decode(clip)
    assert decoded.indices == (0, 4, 7, 11)
    settings = d3_runtime.lookup(None).model.video_settings
    processed = vid.preprocess(decoded, settings)
    expansion = processed.placeholder(
        "<|vision_start|>", "<|video_pad|>", "<|vision_end|>"
    )
    assert expansion.startswith("<0.2 seconds><|vision_start|>")
    per_question = (
        seen["usage"]["input_tokens"] - text["usage"]["input_tokens"]
    ) / len(QUESTIONS)
    tokenizer = d3_runtime.lookup(None).model.tokenizer
    expanded = len(tokenizer.encode(expansion, add_special_tokens=False).ids)
    # The template's own vision start and end around the video, plus the expansion in place of its pad token.
    assert per_question == expanded + 2
    status, both = call(
        d3_runtime,
        {
            "state": "What happens?",
            "questions": QUESTIONS,
            "images": [png_url(64, 64, 2)],
            "videos": [clip],
        },
    )
    assert status == 200 and both["answers"] != seen["answers"]
    again = call(
        d3_runtime, {"state": "What happens?", "questions": QUESTIONS, "videos": [clip]}
    )[1]
    assert again == seen


@pytest.mark.parametrize(
    "videos,message,reason",
    [
        ("data:video/mp4;base64,AAAA", "videos must be a list", "video_format"),
        (
            ["https://example.com/a.mp4"],
            "videos[0]: videos must be base64 MP4",
            "video_format",
        ),
        (
            ["data:video/avi;base64,AAAA"],
            "videos[0]: videos must be base64 MP4",
            "video_format",
        ),
        (
            ["data:video/mp4;base64,not base64!"],
            "videos[0]: invalid base64 video data",
            "video_malformed",
        ),
        (["data:video/mp4;base64,AAAAAAAA"], "videos[0]: ", "video_malformed"),
    ],
)
def test_malformed_videos_fail_the_request(d3_runtime, videos, message, reason):
    pytest.importorskip("cv2")
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "videos": videos}
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert message in body["error"]["message"]
    assert body["error"]["reason"] == reason and "limit" not in body["error"]


def test_oversized_videos_are_refused(d3_runtime):
    pytest.importorskip("cv2")
    huge = "data:video/mp4;base64," + "A" * (4 * -(-vid.MAX_VIDEO_BYTES // 3) + 4)
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "videos": [huge]}
    )
    assert status == 400 and "32,000,000 bytes" in body["error"]["message"]
    assert (body["error"]["reason"], body["error"]["limit"]) == (
        "video_too_large",
        vid.MAX_VIDEO_BYTES,
    )


def test_videos_over_the_duration_limit_are_refused(d3_runtime, mp4_url):
    clip = mp4_url(32, 32, frames=vid.MAX_SECONDS + 2, fps=1.0, seed=5)
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "videos": [clip]}
    )
    assert status == 400 and "300 seconds" in body["error"]["message"]
    assert (body["error"]["reason"], body["error"]["limit"]) == (
        "video_too_long",
        vid.MAX_SECONDS,
    )


def test_a_text_model_refuses_videos(d3_text_package, mp4_url):
    from tests.conftest import start_runtime

    runtime = start_runtime(d3_text_package)
    try:
        status, body = call(
            runtime,
            {"state": "x", "questions": QUESTIONS, "videos": [mp4_url(32, 32, 4)]},
        )
        assert status == 400 and "reads no videos" in body["error"]["message"]
        assert body["error"]["reason"] == "video_unsupported"
    finally:
        runtime.stop()


def test_further_states_carry_their_own_videos(d3_runtime, mp4_url):
    clip = mp4_url(64, 48, frames=6, seed=3)
    alone = call(d3_runtime, {"state": "b", "questions": QUESTIONS, "videos": [clip]})[
        1
    ]
    status, body = call(
        d3_runtime,
        {
            "state": "a",
            "questions": QUESTIONS,
            "states": {
                "other": {"state": "b", "questions": QUESTIONS, "videos": [clip]}
            },
        },
    )
    assert status == 200
    assert body["states"]["other"]["answers"] == alone["answers"]


def test_score_levels_must_be_distinct(d3_runtime):
    question = {
        "type": "score",
        "instructions": "How hard?",
        "criteria": ["Easy", "Easy"],
    }
    status, body = call(
        d3_runtime,
        {"state": "x", "questions": {"s": question, "n": QUESTIONS["reasoning"]}},
    )
    assert status == 200
    assert body["answers"]["s"]["error"] == "invalid_question"
    assert "distinct" in body["answers"]["s"]["message"]


def test_a_request_is_tokenized_at_once_and_reports_errors_in_question_order(
    d3_runtime, monkeypatch
):
    model = d3_runtime.lookup(None).model
    short = {"type": "noul", "instructions": "Is it short?"}
    long = {"type": "noul", "instructions": "word " * 400}
    invalid = {
        "type": "score",
        "instructions": "How hard?",
        "criteria": ["Easy", "Easy"],
    }
    alone = model.plan("state", {"a": short})
    monkeypatch.setattr(model, "limit", len(alone.items[0].ids) + 50)
    plan = model.plan("state", {"l": long, "a": short, "s": invalid})
    assert [item.question_id for item in plan.items] == ["a"]
    assert plan.items[0].ids == alone.items[0].ids
    assert list(plan.errors) == ["l", "s"]
    assert plan.errors["l"]["error"] == "max_length_exceeded"
    assert plan.errors["s"]["error"] == "invalid_question"
    text = prompt.render(
        prompt.user_prompt(
            "state",
            short["instructions"],
            prompt.option_texts("noul", None),
            model.codes,
        )
    )
    assert (
        alone.items[0].ids == model.tokenizer.encode(text, add_special_tokens=False).ids
    )


def test_questions_run_eight_per_pass_in_request_order(d3_runtime):
    model = d3_runtime.lookup(None).model
    questions = {f"q{i}": QUESTIONS["reasoning"] for i in range(BATCH_SIZE + 3)}
    plan = model.plan("state", questions)
    assert model.exact_batches(list(plan.items)) == [
        list(range(BATCH_SIZE)),
        list(range(BATCH_SIZE, BATCH_SIZE + 3)),
    ]
    assert model.shared_context(list(plan.items), None) == 0


def test_the_prompt_lists_codes_and_renders_the_chat_format(d3_runtime):
    model = d3_runtime.lookup(None).model
    user = prompt.user_prompt(
        {"a": 1}, {"ask": "x"}, ["code: Programming", "math"], model.codes
    )
    assert user == (
        'State:\n{"a": 1}\n\nQuestion:\n{"ask": "x"}\n\nOptions:\nA: code: Programming\nB: math\n\n'
        "Reply with only the code of the best option."
    )
    assert prompt.user_prompt("", None, ["x"], model.codes).startswith(
        "State:\n(empty)\n\nQuestion:\nChoose the best"
    )
    rendered = prompt.render(user, 2)
    assert rendered.startswith(
        f"<|im_start|>system\n{prompt.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
    )
    assert rendered.count(prompt.IMAGE_PLACEHOLDER) == 2
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert prompt.option_texts("noul", {"true": {}}) == ["No / false", "Yes / true"]


def test_the_tokenizer_is_built_as_transformers_builds_qwen2(d3_package):
    backend, pad = prompt.load_tokenizer(d3_package)
    config = json.loads((d3_package / "tokenizer_config.json").read_text())
    assert pad == backend.token_to_id(config["pad_token"])
    for key, entry in config["added_tokens_decoder"].items():
        assert backend.token_to_id(entry["content"]) == int(key)
    state = backend.pre_tokenizer.__getstate__().decode()
    assert prompt.QWEN2_PRETOKENIZE.replace("\\", "\\\\") in state
    assert backend.encode("<|audio_pad|>", add_special_tokens=False).ids == [
        backend.token_to_id("<|audio_pad|>")
    ]


def test_an_unknown_chat_template_is_refused(tmp_path, d3_package):
    copy = tmp_path / "d3"
    shutil.copytree(d3_package, copy)
    (copy / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    with pytest.raises(PackageError, match="chat template"):
        prompt.check_chat_template(copy)


@pytest.mark.parametrize(
    "name", ["readout.safetensors", "decision_config.json", "model.safetensors"]
)
def test_tampered_files_are_refused(tmp_path, d3_package, name):
    copy = tmp_path / "d3"
    shutil.copytree(d3_package, copy)
    path = copy / name
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(PackageError):
        pkg.verify(copy)


def test_identity_covers_the_inference_fields(tmp_path, d3_package):
    copy = tmp_path / "d3"
    shutil.copytree(d3_package, copy)
    manifest = json.loads((copy / pkg.MANIFEST_NAME).read_text())
    manifest["identity"]["model_sha256"] = "0" * 64
    (copy / pkg.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackageError, match="identity"):
        pkg.verify(copy)


def test_the_family_detects_only_d3_exports(d3_package, qwen3_package):
    from vllm_srun.plugins.base import PackageRef

    family = Decision3Family()
    assert family.detect(PackageRef(root=d3_package))
    assert not family.detect(PackageRef(root=qwen3_package))
    assert family.descriptor()["modalities"] == ["text", "image", "video"]


def test_preprocessing_matches_the_recorded_processor_output():
    """Pixel rows of a fixed image that needs no resize, as the Transformers 5.17 torchvision processor wrote them."""
    import hashlib

    from PIL import Image

    pixels = bytes(
        (x * 7 + y * 3 + c * 50) % 256
        for y in range(224)
        for x in range(320)
        for c in range(3)
    )
    image = Image.frombytes("RGB", (320, 224), pixels)
    settings = img.ProcessorSettings(16, 2, 2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    out = img.preprocess(image, "digest", settings)
    assert out.grid == (1, 14, 20)
    assert out.tokens == 70
    assert tuple(out.pixel_values.shape) == (280, 1536)
    digest = hashlib.sha256(out.pixel_values.contiguous().numpy().tobytes()).hexdigest()
    assert digest == "6c77827d418e10a4a4b15eeb95f32ec703815c693d1ca92d007ba02a466e71f3"


def test_video_preprocessing_matches_the_recorded_processor_output():
    """Patch rows of fixed frames that need no resize, as the Transformers 5.17 torchvision processor wrote them."""
    import hashlib

    import numpy as np
    from vllm_srun.testing.decision3 import VIDEO_PROCESSOR_CONFIG

    t, y, x, c = np.ogrid[:6, :64, :96, :3]
    frames = ((x * 7 + y * 3 + c * 50 + t * 11) % 256).astype(np.uint8)
    video = vid.Video(frames, 8.0, (0, 3, 6, 9, 11, 11), 12, "digest")
    out = vid.preprocess(video, vid.VideoSettings.from_config(VIDEO_PROCESSOR_CONFIG))
    assert out.grid == (3, 4, 6)
    assert out.tokens == 18
    assert out.timestamps == (0.1875, 0.9375, 1.375)
    assert out.placeholder("<S>", "p", "<E>") == "".join(
        f"<{seconds} seconds><S>pppppp<E>" for seconds in ("0.2", "0.9", "1.4")
    )
    assert tuple(out.pixel_values.shape) == (72, 1536)
    digest = hashlib.sha256(out.pixel_values.contiguous().numpy().tobytes()).hexdigest()
    assert digest == "511bf301e59904d9a101cda76505724a51e16b3203d3aff2583e9c915e77c69c"


@pytest.mark.parametrize(
    "size,expected",
    [
        ((123, 77), (224, 352)),
        ((1280, 1280), (1280, 1280)),
        ((4000, 3000), (1088, 1472)),
        ((32, 4800), (4800, 32)),
    ],
)
def test_smart_resize_keeps_the_released_pixel_budget(size, expected):
    width, height = size
    resized = img.smart_resize(height, width, 32, img.MIN_PIXELS, img.MAX_PIXELS)
    assert resized == expected
    assert img.MIN_PIXELS <= resized[0] * resized[1] <= img.MAX_PIXELS


def test_a_pruned_backbone_follows_its_layer_types(
    d3_pruned_runtime, d3_pruned_package
):
    text = json.loads((d3_pruned_package / "config.json").read_text())["text_config"]
    backbone = d3_pruned_runtime.lookup(None).model.engine_model.backbone
    assert [layer.kind for layer in backbone.layers] == text["layer_types"]
    assert text["full_attention_interval"] == 4


FLA_KERNELS = {
    "chunk_fwd_kernel_o",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "l2norm_fwd_kernel",
    "recompute_w_u_fwd_kernel",
}


@pytest.mark.parametrize("model", DECISION3_MODELS, ids=lambda model: model.repo_id)
def test_every_model_records_goldens_and_kernel_choices(model):
    expected = {"cpu", "rocm"}
    devices = {"rocm:gfx942"}
    if model.repo_id == "vllm-sr/d3-edge":
        expected.update({"cpu:x86_64:avx2:bf16_emulated", "cuda:sm_89"})
        devices.add("cuda:sm_89")
    elif model.repo_id in ("vllm-sr/d3", "vllm-sr/d3-turbo"):
        expected.add("cpu:x86_64:avx512:bf16_native")
    elif model.repo_id == "vllm-sr/d3-ultra":
        expected.update({"cpu:x86_64:avx512:bf16_native", "rocm:gfx942"})
    assert set(model.golden_answers) == expected
    for answers in model.golden_answers.values():
        assert set(answers) == {
            "difficulty",
            "domain",
            "reasoning",
            "image_colour",
            "image_photo",
        }
    assert set(model.kernel_choices) == devices
    for device in devices:
        choices = model.kernel_choices[device]
        assert choices["fla"] == "0.5.2"
        assert set(choices["kernels"]) == FLA_KERNELS
    if "cuda:sm_89" in devices:
        assert model.kernel_choices["cuda:sm_89"]["triton"] == "3.6.0"


def test_threads_stay_within_the_container_unless_configured(monkeypatch):
    monkeypatch.setattr(cpu, "container_cpus", lambda: 2)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert cpu.cap_threads(None) == {"from": 4, "to": 2}
        assert torch.get_num_threads() == 2
        torch.set_num_threads(4)
        assert cpu.cap_threads(3) is None
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert cpu.cap_threads(None) is None
    finally:
        torch.set_num_threads(before)


def test_repeated_media_are_processed_once_and_answer_alike(
    d3_runtime, png_url, mp4_url, monkeypatch
):
    model = d3_runtime.lookup(None).model
    image, clip = png_url(120, 90, 7), mp4_url(96, 64, frames=12, fps=8.0, seed=7)
    decoded = {"images": 0, "videos": 0}
    image_decode, video_decode = img.decode, vid.decode

    def count_image(value):
        decoded["images"] += 1
        return image_decode(value)

    def count_video(value):
        decoded["videos"] += 1
        return video_decode(value)

    monkeypatch.setattr(img, "decode", count_image)
    monkeypatch.setattr(vid, "decode", count_video)
    # with media workers (16 CPUs or more) the decoding is a load in a worker process
    submit = media.submit

    def count_submit(fn, *args):
        decoded["images"] += fn is img.load
        decoded["videos"] += fn is vid.load
        return submit(fn, *args)

    monkeypatch.setattr(media, "submit", count_submit)
    body = {"state": "What is shown?", "questions": QUESTIONS, "images": [image]}
    first, second = call(d3_runtime, body), call(d3_runtime, body)
    assert first == second and first[0] == 200
    assert decoded["images"] == 1
    video = {"state": "What happens?", "questions": QUESTIONS, "videos": [clip]}
    first, second = call(d3_runtime, video), call(d3_runtime, video)
    assert first == second and first[0] == 200
    assert decoded["videos"] == 1
    assert model.media_cache.stats()["hits"] >= 2


def test_passes_of_one_size_reuse_the_tower_features_of_their_images(
    d3_runtime, png_url, monkeypatch
):
    model = d3_runtime.lookup(None).model
    questions = {f"q{i}": QUESTIONS["reasoning"] for i in range(2 * BATCH_SIZE)}
    body = {
        "state": "What is shown?",
        "questions": questions,
        "images": [png_url(80, 60, 9)],
    }
    model.feature_cache = type(model.feature_cache)(model.feature_cache.max_bytes)
    calls = []
    tower = model.engine_model.tower_features

    def counted(name, pixel_values, grids):
        calls.append(len(grids))
        return tower(name, pixel_values, grids)

    monkeypatch.setattr(model.engine_model, "tower_features", counted)
    served = d3_runtime.lookup(None)
    cached = served.answer_alone("decisions", body)
    assert calls == [BATCH_SIZE]
    model.feature_cache = type(model.feature_cache)(model.feature_cache.max_bytes)
    again = served.answer_alone("decisions", body)
    assert again == cached and calls == [BATCH_SIZE, BATCH_SIZE]


def test_coalesced_batches_mix_requests_and_read_their_images_in_one_call(
    d3_runtime, png_url, monkeypatch
):
    model = d3_runtime.lookup(None).model
    plans = [
        model.plan_images("Shown?", QUESTIONS, [png_url(80, 60, 21)]),
        model.plan("Merge two sorted lists.", QUESTIONS),
        model.plan_images(
            "Shown?", QUESTIONS, [png_url(64, 96, 22), png_url(64, 64, 23)]
        ),
    ]
    items = [item for plan in plans for item in plan.items]
    exact = [value for plan in plans for value in model.run(list(plan.items))]
    calls = []
    tower = model.engine_model.tower_features

    def counted(name, pixel_values, grids, packed=False):
        calls.append((len(grids), packed))
        return tower(name, pixel_values, grids, packed=packed)

    model.feature_cache = type(model.feature_cache)(model.feature_cache.max_bytes)
    monkeypatch.setattr(model.engine_model, "tower_features", counted)
    mixed = model.run_approximate(items)
    assert calls == [(3, model.packs_merged_rows)]
    assert len(mixed) == len(exact)
    for got, want in zip(mixed, exact, strict=True):
        assert got is not None and want is not None
        assert max(abs(a - b) for a, b in zip(got, want, strict=True)) < 0.05
    # Features of a merged tower call never stand in for a request's own forwards,
    # which still read one copy of its images per question.
    calls.clear()
    assert [model.run(list(plan.items)) for plan in plans] == [
        exact[:3],
        exact[3:6],
        exact[6:],
    ]
    assert sorted(calls) == [(3, False), (6, False)]


def test_coalesced_forwards_follow_the_measured_cost_per_token_of_their_lane(
    d3_runtime, png_url
):
    from vllm_srun.families.decision3 import family

    model = d3_runtime.lookup(None).model
    saved = dict(model._token_seconds)
    try:
        model._token_seconds["text"] = family.COALESCED_FORWARD_SECONDS / 10_000
        model._token_seconds["media"] = 1.0
        limit = model.forward_token_budget()
        assert model.coalesced_token_budget("text") == (
            10_000 if limit is None else min(10_000, limit)
        )
        assert model.coalesced_token_budget("media") == family.MIN_COALESCED_TOKENS
        text = model.plan("Merge two sorted lists.", QUESTIONS).items
        image = model.plan_images("Shown?", QUESTIONS, [png_url(80, 60, 24)]).items
        assert {model.lane(item) for item in text} == {"text"}
        assert {model.lane(item) for item in image} == {"media"}
        # A media pass moves the media lane's cost only.
        model.run_approximate(list(image))
        assert model._token_seconds["text"] == family.COALESCED_FORWARD_SECONDS / 10_000
        assert model._token_seconds["media"] < 1.0
    finally:
        model._token_seconds = saved
