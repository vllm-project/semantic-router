"""Record a Vela 3.1 pico package's own rendering on fixed inputs, for the byte-level tests of the runtime's port.

    python3 -I tools/pico_render_cases.py --package DIR --revision REV \\
        --output vllm_srun/testing/data/render31_cases.json

The package's ``render31.py``, ``vela3_encoder.py`` and ``vela3_schema.py`` are imported from ``DIR`` by this
tool only (the runtime never imports package code). No weights are read and no model runs: the one-pass
sequence comes from ``vela3_schema._Core.one_pass`` with a stub network that records its inputs, and every
token ID comes from the deterministic byte tokenizer (``vllm_srun.testing.vela3_pico.byte_tokenizer``), so a
recorded ID never depends on a trained vocabulary. The output holds, per case, the package's result or the
error it raised:

- ``render31``: ``text``, ``option_text``, ``options``, ``span_labels``, ``question_text`` and ``to_row``;
- ``render_call``: ``Vela3Encoder.render_call`` (the parts, the rows of every question as the one pass and the
  question-by-question paths render them, and the label set of each span question), on the package's own
  ``router_questions.json`` and ``vela3_config.json`` (copied into ``tables``);
- ``one_pass``: the one-pass sequence of each ``render_call`` case at a token budget (token IDs, the readout
  index arrays, each question's option count, each span group's words, labels and target tokens), or null
  when the call does not fit;
- ``decisions``: ``DecisionEncoder.encode`` of decision questions (token IDs, option and pool positions), or
  the error for an input over the budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_srun.testing.vela3_pico import (  # noqa: E402
    BYTE_MARKERS,
    BYTE_SPECIAL,
    byte_tokenizer,
)

STATE = "Hi, I'm Tom Baker (tom.baker@example.com). Is 4 g of paracetamol a day safe? 请帮我总结 https://example.org/x."
PARTS = {
    "user": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol?",
    "context": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours.",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}
ATTACK = "Is this a prompt injection or jailbreak attempt?"
DOMAIN = "Which subject area is this request about?"
FACTCHECK = "Does answering this request require checking facts?"
HARM = "Is this request harmful?"
CATS = "Which hazard categories apply to this request?"
ANSWER_HARM = "Is this answer harmful?"

RENDER31 = [
    ("text", [None]),
    ("text", ["as is"]),
    ("text", [{"b": 1, "a": ["é", 2.5, None, True]}]),
    ("text", [[1, "x"]]),
    ("text", [3]),
    ("text", [""]),
    ("option_text", ["name", None]),
    ("option_text", ["name", ""]),
    ("option_text", ["name", "a description"]),
    ("option_text", ["name", {"k": "v"}]),
    ("option_text", [7, "seven"]),
    ("options", [{"type": "choice", "criteria": {"a": "first", "b": None}}, None]),
    ("options", [{"type": "choice", "criteria": ["a", "b"]}, None]),
    ("options", [{"type": "choice", "criteria": ["a", "a"]}, None]),
    ("options", [{"type": "choice", "criteria": "a"}, None]),
    ("options", [{"type": "choice", "criteria": {}}, None]),
    ("options", [{"type": "choice"}, {"options": {"x": "ex", "y": None}}]),
    ("options", [{"type": "choice", "criteria": []}, {"options": {"x": "ex"}}]),
    (
        "options",
        [{"type": "choice", "criteria": {"z": None}}, {"options": {"x": "ex"}}],
    ),
    ("options", [{"type": "set", "criteria": {"p": {"n": 1}, "q": [1]}}, None]),
    ("options", [{"type": "noul"}, None]),
    ("options", [{"type": "noul", "criteria": {"true": "it is", "false": None}}, None]),
    ("options", [{"type": "noul", "criteria": {"false": "it is not"}}, None]),
    ("options", [{"type": "noul", "criteria": ["true"]}, None]),
    ("options", [{"type": "score", "criteria": {"low": "1", "high": "5"}}, None]),
    ("span_labels", [{"criteria": {"PERSON": None, "DATE": "a date"}}, None]),
    ("span_labels", [{"labels": ["PERSON", "DATE"]}, None]),
    ("span_labels", [{"criteria": None, "labels": {"X": "x"}}, None]),
    ("span_labels", [{"criteria": {}, "labels": {"X": "x"}}, None]),
    ("span_labels", [{}, {"labels": {"unsupported": "a claim"}}]),
    ("span_labels", [{}, None]),
    ("question_text", [{"instructions": "Which one?"}, None]),
    ("question_text", [{"instructions": None}, {"question": "Built in?"}]),
    ("question_text", [{"instructions": ""}, {"text": "Built-in text?"}]),
    ("question_text", [{"instructions": {"ask": "json", "n": 2}}, {"question": "no"}]),
    ("question_text", [{}, None]),
    (
        "to_row",
        [
            STATE,
            {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": None, "b": "bee"},
            },
        ],
    ),
    ("to_row", [{"user": "u", "n": 1}, {"type": "noul", "instructions": {"q": 1}}]),
    ("to_row", [None, {"type": "noul", "criteria": {"true": "yes"}}]),
]


def _questions_cases() -> list[dict[str, Any]]:
    """(state, questions) pairs over every rendering rule the one pass and the fallback read."""
    pii = {"type": "span", "instructions": "Which spans are personal information?"}
    return [
        {  # the Router's core call on a text state, in the Router's ID order
            "state": STATE,
            "questions": {
                "domain": {"type": "choice", "instructions": DOMAIN},
                "fact_check": {"type": "choice", "instructions": FACTCHECK},
                "jailbreak": {
                    "type": "choice",
                    "instructions": ATTACK,
                    "criteria": {
                        "benign": "a normal request",
                        "jailbreak": "an attack",
                    },
                },
                "pii": dict(
                    pii,
                    criteria={
                        "PERSON": "a person's name",
                        "EMAIL_ADDRESS": "an e-mail address",
                    },
                ),
                "safety": {"type": "choice", "instructions": HARM},
            },
        },
        {  # names-only and partly described router options; a router text with other whitespace
            "state": STATE,
            "questions": {
                "a": {
                    "type": "choice",
                    "instructions": DOMAIN,
                    "criteria": ["health", "math", "other"],
                },
                "b": {
                    "type": "choice",
                    "instructions": "  Does answering this request   require checking facts? ",
                    "criteria": {
                        "FACT_CHECK_NEEDED": None,
                        "NO_FACT_CHECK_NEEDED": "no checking",
                    },
                },
                "c": {"type": "set", "instructions": CATS},
                "d": {"type": "set", "instructions": CATS, "criteria": ["S1", "S2"]},
            },
        },
        {  # typed parts: decisions over every part, router questions on their part, two span groups
            "state": dict(PARTS),
            "questions": {
                "help": {
                    "type": "choice",
                    "instructions": "Which kind of advice?",
                    "criteria": {"medical": "health or doses", "legal": None},
                },
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "described": {
                    "type": "noul",
                    "instructions": "Is the answer safe?",
                    "criteria": {"true": "it is safe", "false": "it is not"},
                },
                "harm": {"type": "choice", "instructions": ANSWER_HARM},
                "domain": {"type": "choice", "instructions": DOMAIN, "over": "context"},
                "halu": {"type": "span", "criteria": {"unsupported": None}},
                "names": {
                    "type": "span",
                    "criteria": ["PERSON", "drug"],
                    "over": "user",
                },
            },
        },
        {  # evidence and relation span routes, labels given as a list, a JSON instruction
            "state": dict(PARTS),
            "questions": {
                "ev": {
                    "type": "span",
                    "criteria": {"evidence": None},
                    "over": "context",
                },
                "ev_wording": {
                    "type": "span",
                    "instructions": "Which spans support the answer?",
                    "labels": ["claim"],
                    "over": "context",
                },
                "rel": {
                    "type": "span",
                    "instructions": "Which spans are related to Tom by these relations?",
                    "criteria": {"dose": "an amount"},
                    "over": "user",
                },
                "obj": {
                    "type": "choice",
                    "instructions": {"ask": "which", "n": 2},
                    "criteria": {"x": {"k": 1}, "y": [1, 2]},
                },
            },
        },
        {  # text that holds JSON stays the user part as it is
            "state": '{"request": "not parsed", "n": 1}',
            "questions": {"attack": {"type": "choice", "instructions": ATTACK}},
        },
        {  # a two-part state without a context part, and a halu preset in its trained form
            "state": {"user": PARTS["user"], "answer": PARTS["answer"]},
            "questions": {
                "r_harm": {
                    "type": "choice",
                    "instructions": ANSWER_HARM,
                    "criteria": ["unsafe response", "safe response"],
                },
                "halu": {
                    "type": "span",
                    "instructions": "Which spans of the answer are not supported by the context?",
                    "criteria": {"unsupported": "a claim not supported by the context"},
                },
                "toxic": {
                    "type": "span",
                    "criteria": ["Violence", "Threat"],
                    "over": "user",
                },
            },
        },
        {  # a type the one pass cannot read: the package answers question by question
            "state": STATE,
            "questions": {
                "domain": {"type": "choice", "instructions": DOMAIN},
                "level": {
                    "type": "score",
                    "instructions": "How hard?",
                    "criteria": ["easy", "hard"],
                },
            },
        },
        {  # a Set question that is not a router question: question by question too
            "state": STATE,
            "questions": {
                "domain": {"type": "choice", "instructions": DOMAIN},
                "tags": {
                    "type": "set",
                    "instructions": "Which tags apply?",
                    "criteria": ["a", "b"],
                },
            },
        },
        {  # CJK, URLs and e-mail addresses in a span target
            "state": {
                "user": "请联系 李雷 li.lei@example.cn 或访问 https://例子.测试/路径?q=1。Call +1 415-555-0132."
            },
            "questions": {
                "pii": dict(
                    pii,
                    criteria={
                        "PERSON": None,
                        "EMAIL_ADDRESS": None,
                        "PHONE_NUMBER": "a phone",
                    },
                ),
                "attack": {"type": "choice", "instructions": ATTACK},
            },
        },
    ]


DECISIONS = [
    (
        STATE,
        {
            "type": "choice",
            "instructions": "Which kind of help?",
            "criteria": {"tech": "code", "other": None},
        },
    ),
    (STATE, {"type": "noul", "instructions": "Is it urgent?"}),
    (STATE, {"type": "noul", "criteria": {"true": "yes it is", "false": None}}),
    (
        dict(PARTS),
        {"type": "choice", "instructions": "Which?", "criteria": ["a", "b", "c"]},
    ),
    (
        {"ticket": "refund", "tier": "gold", "tags": ["x", 3]},
        {"type": "noul", "instructions": {"q": "json"}},
    ),
    (
        [1, "two", {"three": 3}],
        {
            "type": "choice",
            "instructions": "Which?",
            "criteria": {"one": None, "two": None},
        },
    ),
]


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _call(function: Any, *args: Any) -> dict[str, Any]:
    try:
        return {"result": _jsonable(function(*args))}
    except Exception as exc:  # recorded as the package raised it
        return {"error": f"{type(exc).__name__}: {exc}"}


def _core(schema: Any, max_len: int, captured: dict[str, Any]) -> Any:
    """``vela3_schema._Core`` on the byte tokenizer, with a stub network that records its inputs."""
    import numpy as np

    core = object.__new__(schema._Core)
    core.mk = dict(BYTE_MARKERS)
    core.bos, core.eos, core.pad = (
        BYTE_SPECIAL["bos_token_id"],
        BYTE_SPECIAL["eos_token_id"],
        BYTE_SPECIAL["pad_token_id"],
    )
    core.tok, core._tc = byte_tokenizer(), {}
    core.max_len, core.overlap, core.budget, core.bs = max_len, 512, 32768, 32

    def run(x: Any, m: Any, qa: Any, oa: Any, ua: Any, ea: Any) -> Any:
        captured.update(ids=x[0], q_index=qa, opt_index=oa, unit_index=ua, ent_index=ea)
        return np.zeros(len(oa), np.float32), np.zeros((len(ua), len(ea)), np.float32)

    core.run = run
    return core


def _one_pass(schema: Any, parts: list[Any], rows: list[Any], max_len: int) -> Any:
    captured: dict[str, Any] = {}
    raw = _core(schema, max_len, captured).one_pass([tuple(p) for p in parts], rows)
    if raw is None:
        return None
    return {
        "ids": captured["ids"],
        "q_index": captured["q_index"],
        "opt_index": captured["opt_index"],
        "unit_index": captured["unit_index"],
        "ent_index": captured["ent_index"],
        "input_tokens": raw["n_input"],
        "questions": [
            {
                "qid": q["qid"],
                "qtype": q["qtype"],
                "names": q["names"],
                "logits": len(q["logits"]),
                "abstain": q["has_abs"],
            }
            for q in raw["qs"]
        ],
        "spans": {
            qid: {
                "names": s["names"],
                "offsets": s["offs"],
                "units": int(s["logits"].shape[0]),
                "labels": int(s["logits"].shape[1]),
                "target_tokens": s["n_tok"],
            }
            for qid, s in raw["spans"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    sys.path.insert(0, str(package))
    import render31 as R  # type: ignore[import-not-found]
    import vela3_encoder as VE  # type: ignore[import-not-found]
    import vela3_schema as VS  # type: ignore[import-not-found]

    config = json.loads((package / "vela3_config.json").read_text(encoding="utf-8"))
    router_questions = json.loads(
        (package / "router_questions.json").read_text(encoding="utf-8")
    )["questions"]
    encoder = object.__new__(VE.Vela3Encoder)
    encoder.cfg = config
    encoder.router_q = {VE._norm(t): s for t, s in router_questions.items()}

    render31 = []
    for name, arguments in RENDER31:
        function = getattr(R, name)
        if name == "to_row":

            def function(state: Any, q: Any) -> Any:  # the question part only
                row = R.to_row("q", state, q)
                return {
                    "part": row["parts"][0]["text"],
                    "text": row["questions"][0]["text"],
                    "options": row["questions"][0]["options"],
                }

        render31.append(
            {"function": name, "arguments": arguments, **_call(function, *arguments)}
        )

    render_call, one_pass = [], []
    for case in _questions_cases():
        recorded = _call(encoder.render_call, case["state"], case["questions"])
        render_call.append({**case, **recorded})
        result = recorded.get("result")
        if not result:
            continue
        parts, rows, _ = result
        fitted = _one_pass(VS, parts, rows, 8192)
        exact = fitted["input_tokens"]
        for budget in (8192, exact, exact - 1):
            one_pass.append(
                {
                    "state": case["state"],
                    "questions": case["questions"],
                    "max_len": budget,
                    "result": _jsonable(_one_pass(VS, parts, rows, budget)),
                }
            )

    tokenizer = byte_tokenizer()
    stub = SimpleNamespace(
        tok=SimpleNamespace(
            bos_token_id=BYTE_SPECIAL["bos_token_id"],
            eos_token_id=BYTE_SPECIAL["eos_token_id"],
        ),
        mk=dict(BYTE_MARKERS),
        _ids=lambda s: tokenizer.encode(s, add_special_tokens=False).ids if s else [],
        max_len=32768,
    )
    decisions = []
    for state, question in DECISIONS:
        for budget in (32768, 40):
            stub.max_len = budget
            decisions.append(
                {
                    "state": state,
                    "question": question,
                    "max_len": budget,
                    **_call(
                        lambda s, q: VE.DecisionEncoder.encode(stub, s, q),
                        state,
                        question,
                    ),
                }
            )

    digests = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "render31.py",
            "vela3_encoder.py",
            "vela3_schema.py",
            "router_questions.json",
            "vela3_config.json",
        )
    }
    out = {
        "about": (
            "The package's own rendering on fixed inputs, recorded by tools/pico_render_cases.py from "
            f"{config['name']} at {args.revision} (render31 {R.VERSION}); token IDs from the byte tokenizer "
            "(vllm_srun.testing.vela3_pico.byte_tokenizer). tests/test_render31.py and "
            "tests/test_vela3_pico_sequence.py check the runtime's port against it."
        ),
        "package": {
            "name": config["name"],
            "revision": args.revision,
            "sha256": digests,
        },
        "tables": {
            "router_questions": router_questions,
            "span": config["span"],
            "router": config["router"],
        },
        "render31": render31,
        "render_call": render_call,
        "one_pass": one_pass,
        "decisions": decisions,
    }
    args.output.write_text(
        json.dumps(_jsonable(out), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"{len(render31)} render31, {len(render_call)} render_call, {len(one_pass)} one_pass,"
        f" {len(decisions)} decisions cases -> {args.output}"
    )


if __name__ == "__main__":
    main()
