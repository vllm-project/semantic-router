"""Open-label definition prompts, cache and package writer selection."""

from __future__ import annotations

import json
from pathlib import Path

from vllm_srun.families.decision3 import prompt
from vllm_srun.families.vela3_d3.labeldef import (
    BATCH,
    CTX_MAX,
    LabelDefiner,
    first_line,
    prompt_for,
    user_turn,
)


class Recording:
    """An engine that decodes each prompt back and answers with a fixed sentence."""

    lm_head = object()

    def __init__(self) -> None:
        self.calls: list[list[list[int]]] = []

    def generate(self, rows, *, max_new_tokens, eos_token_id, pad_token_id):
        self.calls.append([list(row) for row in rows])
        assert max_new_tokens == 40
        return [self._ids("A short code.\nignored") for _ in rows]

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [ord(char) for char in text]


class Letters:
    """A stand-in tokenizer: one id per character, so a prompt round-trips."""

    def encode(self, text: str, add_special_tokens: bool = True):
        assert add_special_tokens is False
        return type("Encoding", (), {"ids": [ord(char) for char in text]})()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)

    def token_to_id(self, token: str) -> int:
        assert token == "<|im_end|>"
        return 7


def test_the_prompt_adds_the_label_set_only_up_to_sixty() -> None:
    assert "The full label set is: drug, dose." in prompt_for("drug", ["drug", "dose"])
    names = [f"label {index}" for index in range(CTX_MAX + 1)]
    assert "The full label set is" not in prompt_for(names[0], names)
    assert 'entity type "label 0"' in prompt_for(names[0], names)


def test_the_user_turn_is_thinking_off_with_no_system_turn() -> None:
    text = user_turn("Define it.")
    assert text.startswith("<|im_start|>user\nDefine it.<|im_end|>\n")
    assert "<|im_start|>system" not in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_definitions_keep_the_first_line_and_the_set_cache() -> None:
    engine = Recording()
    writer = LabelDefiner(engine, Letters(), pad_id=0, cache=None)  # type: ignore[arg-type]
    assert writer.define(["dose", "drug"], {"drug"}) == {"drug": "A short code."}
    encoded = "".join(chr(i) for i in engine.calls[0][0])
    assert "The full label set is: dose, drug." in encoded
    assert writer.define(["dose", "drug"], {"dose", "drug"}) == {
        "dose": "A short code.",
        "drug": "A short code.",
    }
    # The set was cached after the first label, so the second call generates only the one still missing.
    assert len(engine.calls) == 2
    assert len(engine.calls[1]) == 1


def test_a_recorded_set_is_not_generated(tmp_path: Path) -> None:
    cache = tmp_path / "label_definitions.json"
    cache.write_text(
        json.dumps(
            {
                "sets": [
                    {
                        "labels": ["dose", "drug"],
                        "definitions": {"drug": "from the file", "dose": "also"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    engine = Recording()
    writer = LabelDefiner(engine, Letters(), pad_id=0, cache=cache)  # type: ignore[arg-type]
    assert writer.define(["drug", "dose"], {"drug"}) == {"drug": "from the file"}
    assert engine.calls == []


def test_more_than_one_batch_is_split() -> None:
    engine = Recording()
    writer = LabelDefiner(engine, Letters(), pad_id=0, cache=None, batch=2)  # type: ignore[arg-type]
    names = ["a", "b", "c"]
    assert set(writer.define(names, set(names))) == set(names)
    assert [len(call) for call in engine.calls] == [2, 1]
    assert BATCH == 64


def test_a_large_set_is_cached_per_label_without_the_context_line() -> None:
    engine = Recording()
    writer = LabelDefiner(engine, Letters(), pad_id=0, cache=None)  # type: ignore[arg-type]
    names = [f"n{index}" for index in range(CTX_MAX + 1)]
    assert writer.define(names, {names[0]})[names[0]] == "A short code."
    encoded = "".join(chr(i) for i in engine.calls[0][0])
    assert "The full label set is" not in encoded
    assert writer.define(names, {names[0]})[names[0]] == "A short code."
    assert len(engine.calls) == 1


def test_generated_definitions_are_a_bounded_lru_and_the_file_stays(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "label_definitions.json"
    cache.write_text(
        json.dumps(
            {"sets": [{"labels": ["dose"], "definitions": {"dose": "from the file"}}]}
        ),
        encoding="utf-8",
    )
    engine = Recording()
    writer = LabelDefiner(engine, Letters(), pad_id=0, cache=cache, max_cached=2)  # type: ignore[arg-type]
    for name in ("a", "b"):
        writer.define([name], {name})
    writer.define(["a"], {"a"})  # a is now the most recently used
    writer.define(["c"], {"c"})  # evicts b
    calls = len(engine.calls)
    writer.define(["a"], {"a"})
    assert len(engine.calls) == calls
    writer.define(["b"], {"b"})
    assert len(engine.calls) == calls + 1
    # The package's file is never evicted.
    assert writer.define(["dose"], {"dose"}) == {"dose": "from the file"}
    assert len(engine.calls) == calls + 1


def test_first_line_strips() -> None:
    assert first_line("  Alpha.\nBeta  ") == "Alpha."


def test_the_real_tokenizer_round_trips_the_user_turn(tmp_path: Path) -> None:
    from vllm_srun.testing.decision3 import write_tokenizer

    write_tokenizer(tmp_path)
    tokenizer, pad = prompt.load_tokenizer(tmp_path)
    text = user_turn(prompt_for("drug", ["drug", "dose"]))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert pad != tokenizer.token_to_id("<|im_end|>")
    assert "<|im_start|>user" in tokenizer.decode(ids, skip_special_tokens=False)
