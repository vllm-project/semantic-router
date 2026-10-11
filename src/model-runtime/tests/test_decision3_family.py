"""The decision3 family on tiny fixture packages: verification, rendering, image inputs and answers."""

import asyncio
import json
import math
import shutil

import pytest
import torch
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
    "images,reason",
    [
        ("data:image/png;base64,AAAA", "images must be a list"),
        (
            ["https://example.com/a.png"],
            "images[0]: images must be base64 PNG, JPEG or WebP data URLs",
        ),
        (
            ["data:image/gif;base64,R0lGODlhAQABAAAAACw="],
            "images[0]: images must be base64 PNG, JPEG or WebP",
        ),
        (["data:image/png;base64,not base64!"], "images[0]: invalid base64 image data"),
        (["data:image/png;base64,iVBORw0KGgo="], "images[0]: invalid image data"),
    ],
)
def test_malformed_images_fail_the_request(d3_runtime, images, reason):
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "images": images}
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert reason in body["error"]["message"]


def test_oversized_and_extreme_images_are_refused(d3_runtime, png_url):
    huge = "data:image/png;base64," + "A" * (4 * -(-img.MAX_IMAGE_BYTES // 3) + 4)
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "images": [huge]}
    )
    assert status == 400 and "8,000,000 bytes" in body["error"]["message"]
    status, body = call(
        d3_runtime,
        {"state": "x", "questions": QUESTIONS, "images": [png_url(2010, 10, 3)]},
    )
    assert status == 400 and "aspect ratio" in body["error"]["message"]


def test_a_text_model_refuses_images(d3_text_package, png_url):
    from tests.conftest import start_runtime

    runtime = start_runtime(d3_text_package)
    try:
        assert runtime.lookup(None).card([])["modalities"] == ["text"]
        status, body = call(
            runtime, {"state": "x", "questions": QUESTIONS, "images": [png_url(64, 64)]}
        )
        assert status == 400 and "images are not supported" in body["error"]["message"]
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


def test_each_frame_is_capped_at_the_runtime_pixel_budget(d3_runtime):
    settings = d3_runtime.lookup(None).model.video_settings
    assert vid.frame_size(settings, 4, 1080, 1920) == (320, 576)
    assert vid.frame_size(settings, 32, 64, 48) == (64, 64)
    assert vid.input_tokens(settings, 4, 1080, 1920) == 2 * (320 // 32) * (576 // 32)


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
    "videos,reason",
    [
        ("data:video/mp4;base64,AAAA", "videos must be a list"),
        (["https://example.com/a.mp4"], "videos[0]: videos must be base64 MP4"),
        (["data:video/avi;base64,AAAA"], "videos[0]: videos must be base64 MP4"),
        (["data:video/mp4;base64,not base64!"], "videos[0]: invalid base64 video data"),
        (["data:video/mp4;base64,AAAAAAAA"], "videos[0]: "),
    ],
)
def test_malformed_videos_fail_the_request(d3_runtime, videos, reason):
    pytest.importorskip("cv2")
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "videos": videos}
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert reason in body["error"]["message"]


def test_oversized_videos_are_refused(d3_runtime):
    pytest.importorskip("cv2")
    huge = "data:video/mp4;base64," + "A" * (4 * -(-vid.MAX_VIDEO_BYTES // 3) + 4)
    status, body = call(
        d3_runtime, {"state": "x", "questions": QUESTIONS, "videos": [huge]}
    )
    assert status == 400 and "32,000,000 bytes" in body["error"]["message"]


def test_a_text_model_refuses_videos(d3_text_package, mp4_url):
    from tests.conftest import start_runtime

    runtime = start_runtime(d3_text_package)
    try:
        status, body = call(
            runtime,
            {"state": "x", "questions": QUESTIONS, "videos": [mp4_url(32, 32, 4)]},
        )
        assert status == 400 and "reads no videos" in body["error"]["message"]
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
def test_every_model_records_golden_answers_and_rocm_kernel_choices(model):
    assert set(model.golden_answers) == {"cpu", "rocm"}
    for answers in model.golden_answers.values():
        assert set(answers) == {
            "difficulty",
            "domain",
            "reasoning",
            "image_colour",
            "image_photo",
        }
    assert set(model.kernel_choices) == {"rocm:gfx942"}
    choices = model.kernel_choices["rocm:gfx942"]
    assert choices["fla"] == "0.5.2"
    assert set(choices["kernels"]) == FLA_KERNELS


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
