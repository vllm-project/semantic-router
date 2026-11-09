"""Which gated adapter answers each decision question (the released ``Vela3MiniModel.decide``).

- Router question IDs (``router_question_ids``) run the ``decision`` adapter;
  with ``router_gate`` ``id_and_wording`` only when the instructions are one of
  the trained wordings of ``router_questions.json`` (whitespace collapsed).
- ``decision_task_adapters`` moves some router IDs to their own adapter
  (``domain``, ``feedback``, ...); a key ending in ``:`` is a prefix that takes
  every question ID starting with it that is not a router ID (``hazard:``).
- With ``decision_opt_in``, a question that sets ``"decision_adapter": true``
  runs the ``decision_optin`` adapter, or the ``decision`` adapter when the
  package has none; router IDs keep theirs. The key is the package's, never
  passed to the base.
- Every other question runs with every adapter off: the base, bit for bit.

The package runs one pass per adapter in this order: ``decision``, the task
adapters by name, the opt-in adapter; a later pass answers its rows, so a row
that two rules claim takes the later pass's adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

DECISION = "decision"
OPT_IN = "decision_optin"
OPT_IN_KEY = "decision_adapter"


def _wording(text: Any) -> str:
    return " ".join(str(text).split())


@dataclass(frozen=True)
class Gates:
    router_ids: frozenset[str]
    wordings: Mapping[str, frozenset[str]]
    by_wording: bool
    tasks: Mapping[str, str]
    prefixes: tuple[tuple[str, str], ...]
    opt_in: str | None

    @classmethod
    def from_config(
        cls,
        vela3: Mapping[str, Any],
        trained_questions: Mapping[str, Any],
        adapters: Iterable[str],
    ) -> Gates:
        """The gates a package's ``vela3_config.json`` and trained router questions declare."""
        adapters = frozenset(adapters)
        wordings = {}
        for question_id, entry in trained_questions.items():
            variants = [entry] if "question" in entry else list(entry.values())
            wordings[question_id] = frozenset(
                _wording(variant["question"]["instructions"]) for variant in variants
            )
        declared = dict(vela3.get("decision_task_adapters") or {})
        for adapter in declared.values():
            if adapter not in adapters:
                raise ValueError(f"decision_task_adapters names no adapter {adapter!r}")
        opt_in = None
        if vela3.get("decision_opt_in"):
            opt_in = OPT_IN if OPT_IN in adapters else DECISION
        return cls(
            router_ids=frozenset(vela3.get("router_question_ids") or ()),
            wordings=wordings,
            by_wording=vela3.get("router_gate", "id") == "id_and_wording",
            tasks={k: v for k, v in declared.items() if not k.endswith(":")},
            prefixes=tuple((k, v) for k, v in declared.items() if k.endswith(":")),
            opt_in=opt_in,
        )

    def _router(self, question_id: str, question: Any) -> bool:
        if question_id not in self.router_ids:
            return False
        if not self.by_wording:
            return True
        instructions = (
            question.get("instructions") if isinstance(question, dict) else None
        )
        return instructions is not None and _wording(instructions) in self.wordings.get(
            question_id, frozenset()
        )

    def assign(self, questions: Mapping[str, Any]) -> dict[str, str]:
        """The adapter of every gated question, by question ID; absent: every adapter off."""
        router = {k for k, q in questions.items() if self._router(k, q)}
        task = {k: self.tasks[k] for k in router if k in self.tasks}
        for prefix, adapter in self.prefixes:
            task.update(
                {
                    k: adapter
                    for k in questions
                    if k.startswith(prefix) and k not in router
                }
            )
        router -= set(task)
        asked = set()
        if self.opt_in is not None:
            asked = {
                k
                for k, q in questions.items()
                if isinstance(q, dict) and q.get(OPT_IN_KEY) is True and k not in router
            }
        if self.opt_in == DECISION:
            router |= asked
            asked = set()
        assigned = dict.fromkeys(router, DECISION)
        assigned.update(task)
        assigned.update(dict.fromkeys(asked, OPT_IN))
        return assigned

    @staticmethod
    def strip(questions: Mapping[str, Any]) -> dict[str, Any]:
        """The questions without the package's own ``decision_adapter`` key."""
        return {
            k: (
                {a: b for a, b in q.items() if a != OPT_IN_KEY}
                if isinstance(q, dict)
                else q
            )
            for k, q in questions.items()
        }

    @staticmethod
    def pass_order(adapters: Iterable[str | None]) -> list[str]:
        """The passes a batch runs: ``decision``, the task adapters by name, then the opt-in adapter."""
        present = {name for name in adapters if name is not None}
        tasks = sorted(present - {DECISION, OPT_IN})
        return (
            ([DECISION] if DECISION in present else [])
            + tasks
            + ([OPT_IN] if OPT_IN in present else [])
        )
