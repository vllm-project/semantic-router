"""Open-label definitions, as the released ``vela3_labeldef.py`` writes them.

A user turn only, thinking off, with the label set as context when it holds at
most 60 labels. The base generates greedily, every adapter off, up to 40 new
tokens; the first line is the definition. Definitions are cached per sorted
label set (per label above 60), and a package's ``label_definitions.json``
fills that cache. ``names`` and ``labeldef_writer`` do not come here: the
former stays name-only, and no released package selects the latter.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ...plugins.base import EngineModel, LMHeadSpec

CTX_MAX = 60
MAX_NEW = 40
BATCH = 64
# Generated cache entries kept (label sets, or single labels above CTX_MAX); the
# least recently used goes first. Request labels are open, so the cache is bounded.
MAX_CACHED = 4096
IM_END = "<|im_end|>"
PROMPT = (
    "You are writing annotation guidelines for named entity recognition.{ctx}\n"
    'Define the entity type "{label}" in one short sentence: what kind of text span it covers. '
    "Reply with the definition only."
)
# The chat template's user turn and thinking-off generation prompt, with no system turn.
USER_TURN = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt_for(label: str, names: list[str]) -> str:
    """The package's ``prompt``: the label-set line only for a set of at most ``CTX_MAX`` labels."""
    context = (
        f"\nThe full label set is: {', '.join(names)}." if len(names) <= CTX_MAX else ""
    )
    return PROMPT.format(ctx=context, label=label)


def user_turn(text: str) -> str:
    """``apply_chat_template`` of one user message with the generation prompt and thinking off."""
    return USER_TURN.format(prompt=text)


def cache_key(names: list[str]) -> tuple[str, ...] | None:
    """The package's ``_key``: sorted names, or None when the set is too large to share a prompt."""
    if len(names) > CTX_MAX:
        return None
    return tuple(sorted(names))


def first_line(text: str) -> str:
    """The generated text stripped, then its first line, stripped again."""
    return text.strip().split("\n", 1)[0].strip()


def load_cache(path: Path) -> dict[Any, dict[str, str]]:
    """Definitions already recorded in ``label_definitions.json`` (``sets``), keyed as the package keys them."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found: dict[Any, dict[str, str]] = {}
    for entry in data.get("sets") or []:
        if not isinstance(entry, dict):
            continue
        labels, definitions = entry.get("labels"), entry.get("definitions")
        if not isinstance(labels, list) or not isinstance(definitions, dict):
            continue
        found[cache_key([str(name) for name in labels])] = {
            str(name): str(text) for name, text in definitions.items()
        }
    return found


def lm_head_spec(
    root: Path, config: dict[str, Any], text: dict[str, Any]
) -> LMHeadSpec | None:
    """The package's ``definitions.writer`` as an ``LMHeadSpec``, or None for ``names`` and ``labeldef_writer``.

    ``base`` needs a tied embedding. ``grafted_head`` reads ``definitions.lm_head_file`` from ``root``.
    """
    definitions = config.get("definitions") or {}
    if not isinstance(definitions, dict):
        return None
    mode = definitions.get("writer") or "names"
    if mode == "base":
        if text.get("tie_word_embeddings", True) is False:
            return None
        return LMHeadSpec(tied=True)
    if mode == "grafted_head":
        file = definitions.get("lm_head_file")
        if not isinstance(file, str) or not file:
            return None
        return LMHeadSpec(tied=False, weight_file=root / file, tensor="lm_head.weight")
    return None


class LabelDefiner:
    """The package's ``LabelDefiner.define`` over an engine that can ``generate``."""

    def __init__(
        self,
        engine: EngineModel,
        tokenizer: Any,
        pad_id: int,
        cache: Path | None,
        batch: int = BATCH,
        max_cached: int = MAX_CACHED,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.batch = batch
        self.eos = tokenizer.token_to_id(IM_END)
        # The package's label_definitions.json stays; generated entries are an LRU.
        self.shipped: dict[Any, dict[str, str]] = (
            load_cache(cache) if cache is not None else {}
        )
        self.generated: OrderedDict[Any, dict[str, str] | str] = OrderedDict()
        self.max_cached = max_cached
        self._lock = threading.Lock()

    def define(self, names: list[str], only: set[str]) -> dict[str, str]:
        """``{label: definition}`` for the labels in ``only`` that this writer can define."""
        with self._lock:
            return self._define(list(names), only)

    def _cached(self, key: Any) -> Any:
        if key in self.generated:
            self.generated.move_to_end(key)
            return self.generated[key]
        return self.shipped.get(key)

    def _remember(self, key: Any, value: Any) -> None:
        self.generated[key] = value
        self.generated.move_to_end(key)
        while len(self.generated) > self.max_cached:
            self.generated.popitem(last=False)

    def _define(self, names: list[str], only: set[str]) -> dict[str, str]:
        want = [name for name in names if name in only]
        key = cache_key(names)
        if key is not None:
            got = dict(self._cached(key) or {})
        else:
            got = {
                name: text
                for name in want
                if (text := self._cached((None, name))) is not None
            }
        missing = [name for name in want if not got.get(name)]
        if missing:
            written = self._generate([prompt_for(name, names) for name in missing])
            fresh = {}
            for name, text in zip(missing, written, strict=True):
                if not text:
                    continue
                got[name] = text
                fresh[name] = text
                if key is None:
                    self._remember((None, name), text)
            if key is not None and fresh:
                self._remember(key, got)
        return {name: got[name] for name in want if got.get(name)}

    def _generate(self, prompts: list[str]) -> list[str]:
        lines: list[str] = []
        for start in range(0, len(prompts), self.batch):
            chunk = prompts[start : start + self.batch]
            rows = [
                self.tokenizer.encode(user_turn(text), add_special_tokens=False).ids
                for text in chunk
            ]
            generated = self.engine.generate(
                rows,
                max_new_tokens=MAX_NEW,
                eos_token_id=self.eos,
                pad_token_id=self.pad_id,
            )
            lines += [
                first_line(self.tokenizer.decode(ids, skip_special_tokens=True))
                for ids in generated
            ]
        return lines


def writer_of(
    mode: str,
    engine: EngineModel,
    tokenizer: Any,
    pad_id: int,
    cache: Path | None,
) -> LabelDefiner | None:
    """The definition writer for ``mode``; None for ``names`` and anything without an LM head."""
    if mode not in ("base", "grafted_head") or engine.lm_head is None:
        return None
    return LabelDefiner(engine, tokenizer, pad_id, cache)
