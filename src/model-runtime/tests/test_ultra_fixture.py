"""The ``ultra`` fixture serves d3-ultra's card: turbo's, with the ``header-v1`` prompt layout (a text-only request
lists its questions in front of the state) and a temperature per request size (``temperature_by_questions``).
"""

import asyncio
import hashlib
import json
import math
import shutil

import pytest
import torch
from vllm_srun.errors import PackageError
from vllm_srun.families.decision3 import package as pkg
from vllm_srun.families.decision3 import prompt
from vllm_srun.families.decision3.family import Decision3Model, _scaled
from vllm_srun.families.decision3.state_once import StateOnceModel

from .conftest import start_runtime

STATE = "Merge two sorted lists of integers into one sorted list."
QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": "Programming", "math": None, "other": {"note": "x"}},
    },
    "broken": {"type": "choice", "criteria": {}},
    "reasoning": {"type": "noul", "instructions": "Does this need reasoning?"},
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", {"level": "hard"}, "Expert"],
    },
}
# The question list of QUESTIONS, written out: the invalid question is left out of the numbering, a Noul question
# lists no options, and Choice and Score questions list their option texts without answer codes.
HEADER = (
    "Questions in this request:\n"
    "1. Which domain is this request about?\n"
    "   - code: Programming\n"
    "   - math\n"
    '   - other: {"note": "x"}\n'
    "2. Does this need reasoning?\n"
    "3. How difficult is it?\n"
    "   - Trivial\n"
    '   - {"level": "hard"}\n'
    "   - Expert\n"
    "\n"
)
ONE_QUESTION = {"reasoning": QUESTIONS["reasoning"]}


def call(runtime, body):
    return asyncio.run(runtime.call("decisions", body))


@pytest.fixture(scope="module")
def packages(fixture_root):
    from vllm_srun.testing.decision3 import write_fixture

    root = fixture_root / "d3-ultra"
    # One seed: the packages hold the same weights and differ only in their decision_config.
    return {
        "ultra": write_fixture(root / "ultra", "ultra", seed=23),
        "plain": write_fixture(
            root / "plain",
            "ultra",
            seed=23,
            overrides={"temperature_by_questions": None},
        ),
        "turbo": write_fixture(root / "turbo", "turbo", seed=23),
        "rows": write_fixture(
            root / "rows", "vision", seed=23, overrides={"prompt_layout": pkg.HEADER_V1}
        ),
    }


@pytest.fixture(scope="module")
def runtimes(packages):
    started = {name: start_runtime(package) for name, package in packages.items()}
    yield started
    for runtime in started.values():
        runtime.stop()


def model_of(runtimes, name):
    return runtimes[name].lookup(None).model


def rehash(root, name):
    """MODEL_MANIFEST.json's digest and size of ``name`` after an edit; the identity stays as built."""
    manifest = json.loads((root / pkg.MANIFEST_NAME).read_text())
    data = (root / name).read_bytes()
    manifest["files_sha256"][name] = hashlib.sha256(data).hexdigest()
    manifest["files_bytes"][name] = len(data)
    (root / pkg.MANIFEST_NAME).write_text(json.dumps(manifest))


def test_the_identity_covers_the_layout_and_the_temperatures(packages, tmp_path):
    details, _ = pkg.verify(packages["ultra"])
    config = details.decision_config
    assert config["prompt_layout"] == pkg.HEADER_V1
    assert pkg.temperature_table(config) == ((1, 4, 16), (1.046, 1.0, 1.0, 1.0))
    files = json.loads((packages["ultra"] / pkg.MANIFEST_NAME).read_text())["identity"][
        "files_sha256"
    ]
    for field in ("prompt_layout", "temperature_by_questions"):
        without = {key: value for key, value in config.items() if key != field}
        assert pkg.model_identity(files, without) != details.model_sha256
    copy = tmp_path / "temperatures-edited"
    shutil.copytree(packages["ultra"], copy)
    config = json.loads((copy / "decision_config.json").read_text())
    config["temperature_by_questions"]["temperature"][0] = 1.5
    (copy / "decision_config.json").write_text(json.dumps(config))
    rehash(copy, "decision_config.json")
    with pytest.raises(PackageError, match="model identity differs"):
        pkg.verify(copy)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_layout", "header-v2", "unknown prompt layout"),
        ("prompt_layout", 1, "unknown prompt layout"),
        ("temperature_by_questions", [1.0], "temperature_by_questions"),
        (
            "temperature_by_questions",
            {"max_questions": [1]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [1, 4], "temperature": [1.1, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [4, 4], "temperature": [1.1, 1.0, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [0], "temperature": [1.1, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [True], "temperature": [1.1, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [1], "temperature": [0.0, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [1], "temperature": [math.inf, 1.0]},
            "temperature_by_questions",
        ),
        (
            "temperature_by_questions",
            {"max_questions": [1], "temperature": [1.1, 1.0], "default": 1.0},
            "temperature_by_questions",
        ),
    ],
)
def test_a_malformed_layout_or_temperature_table_is_refused(
    packages, field, value, message
):
    config = json.loads((packages["ultra"] / "decision_config.json").read_text())
    config[field] = value
    with pytest.raises(PackageError, match=message):
        pkg.check_decision_config(config)


def test_the_temperature_follows_the_number_of_questions(runtimes):
    model = model_of(runtimes, "ultra")
    assert isinstance(model, StateOnceModel)
    assert [model.request_temperature(n) for n in (1, 2, 4, 5, 16, 17, 64)] == [
        1.046,
        *[1.0] * 6,
    ]
    table = ((1, 4, 16), (4.0, 3.0, 2.0, 1.0))
    assert [pkg.temperature_for(table, n) for n in (1, 2, 4, 5, 16, 17, 64)] == [
        4.0,
        3.0,
        3.0,
        2.0,
        2.0,
        1.0,
        1.0,
    ]
    assert model_of(runtimes, "turbo").request_temperature(1) is None


def test_a_text_request_lists_its_questions_before_the_state(runtimes):
    model = model_of(runtimes, "ultra")
    plan = model.plan(STATE, QUESTIONS)
    assert [item.question_id for item in plan.items] == [
        "domain",
        "reasoning",
        "difficulty",
    ]
    assert set(plan.errors) == {"broken"}
    head = prompt.state_block(STATE, header=HEADER)
    assert head.endswith(f"\n{HEADER}State:\n{STATE}\n\n")
    state = model.tokenizer.encode(head, add_special_tokens=False)
    split = prompt.state_tokens(state.offsets, head)
    for item, (instructions, texts) in zip(
        plan.items,
        [
            (
                QUESTIONS["domain"]["instructions"],
                ["code: Programming", "math", 'other: {"note": "x"}'],
            ),
            (QUESTIONS["reasoning"]["instructions"], list(prompt.NOUL_DESCRIPTIONS)),
            (
                QUESTIONS["difficulty"]["instructions"],
                ["Trivial", '{"level": "hard"}', "Expert"],
            ),
        ],
        strict=True,
    ):
        whole = head + prompt.question_block(instructions, texts, model.codes)
        assert item.ids == model.tokenizer.encode(whole, add_special_tokens=False).ids
        assert item.state is not None and list(item.state.ids) == state.ids[:split]
        assert item.temperature == 1.0


def test_requests_with_images_keep_the_d3_layout(runtimes, png_url):
    ultra, turbo = model_of(runtimes, "ultra"), model_of(runtimes, "turbo")
    images = [png_url(64, 48, 3)]
    with_header = ultra.plan_images(STATE, QUESTIONS, images)
    without = turbo.plan_images(STATE, QUESTIONS, images)
    assert [item.ids for item in with_header.items] == [
        item.ids for item in without.items
    ]
    assert prompt.HEADER_TITLE not in ultra.tokenizer.decode(with_header.items[0].ids)
    assert (
        ultra.plan(STATE, QUESTIONS).items[0].ids
        != turbo.plan(STATE, QUESTIONS).items[0].ids
    )


def test_composed_requests_keep_the_d3_layout_and_box_its_temperature(
    runtimes, png_url
):
    model = model_of(runtimes, "ultra")
    plan = model.plan_composed(
        STATE,
        {
            "where": {"type": "box", "instructions": "the brightest patch"},
            "reasoning": QUESTIONS["reasoning"],
        },
        [png_url(64, 48, 4)],
        None,
        {},
    )
    temperatures = {item.question_id: item.temperature for item in plan.items}
    assert temperatures == {"where": None, "reasoning": 1.046}
    assert prompt.HEADER_TITLE not in model.tokenizer.decode(plan.items[0].ids)


def test_a_one_question_request_is_answered_at_its_temperature(runtimes):
    """The same weights without the table answer at the package temperature (1.0): a one-question request's
    probabilities are those raised to 1 / 1.046 and renormalized, and a two-question request's are the same.
    """
    ultra, plain = runtimes["ultra"], runtimes["plain"]
    for questions in (ONE_QUESTION, {"domain": QUESTIONS["domain"]}):
        status, warm = call(ultra, {"state": STATE, "questions": questions})
        assert status == 200, warm
        status, base = call(plain, {"state": STATE, "questions": questions})
        assert status == 200, base
        (key,) = questions
        if questions[key]["type"] == "noul":
            got = [1 - warm["answers"][key]["noul"], warm["answers"][key]["noul"]]
            reference = [1 - base["answers"][key]["noul"], base["answers"][key]["noul"]]
        else:
            got = list(warm["answers"][key]["probabilities"].values())
            reference = list(base["answers"][key]["probabilities"].values())
        powered = [value ** (1 / 1.046) for value in reference]
        expected = [value / sum(powered) for value in powered]
        assert max(abs(a - b) for a, b in zip(got, expected, strict=True)) < 1e-5
        assert max(abs(a - b) for a, b in zip(got, reference, strict=True)) > 1e-4
    two = {"domain": QUESTIONS["domain"], "reasoning": QUESTIONS["reasoning"]}
    status, warm = call(ultra, {"state": STATE, "questions": two})
    assert status == 200, warm
    status, base = call(plain, {"state": STATE, "questions": two})
    assert warm["answers"] == base["answers"]


def test_rows_divide_by_their_own_temperature_as_their_own_pass_would():
    logits = torch.randn(4, pkg.MAX_OPTIONS)
    temperatures = [1.046, 1.0, 1.046, 0.75]
    scaled = _scaled(logits, temperatures)
    for row, value in enumerate(temperatures):
        assert torch.equal(scaled[row], (logits[row : row + 1] / value)[0])
    assert torch.equal(_scaled(logits, [0.75] * 4), logits / 0.75)


def test_a_per_question_package_lists_the_questions_too(runtimes):
    model = model_of(runtimes, "rows")
    assert type(model) is Decision3Model
    plan = model.plan(STATE, QUESTIONS)
    first = plan.items[0]
    text = prompt.render(
        prompt.user_prompt(
            STATE,
            QUESTIONS["domain"]["instructions"],
            ["code: Programming", "math", 'other: {"note": "x"}'],
            model.codes,
            HEADER,
        )
    )
    assert first.ids == model.tokenizer.encode(text, add_special_tokens=False).ids
