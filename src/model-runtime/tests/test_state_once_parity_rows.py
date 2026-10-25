"""tools/state_once_parity.py's grouped per-row path (``--row-tokens``): groups never mix states, stay within the
padded-token budget unless one row alone is longer, and the answers come back in the items' order.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "state_once_parity.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("state_once_parity", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(eq=False)
class State:
    name: str


@dataclass
class Item:
    question_id: str
    ids: list[int]
    state: State


class Rows:
    """Answers each row with its own question id and records every call's rows."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run_rows(self, items):
        self.calls.append([item.question_id for item in items])
        return [item.question_id for item in items]


def items() -> list[Item]:
    first, second = State("a"), State("b")
    lengths = [("a", 100), ("a", 120), ("a", 90), ("a", 400), ("b", 50), ("b", 60)]
    return [
        Item(f"q{i}", [0] * n, first if s == "a" else second)
        for i, (s, n) in enumerate(lengths)
    ]


def test_groups_respect_states_and_budget():
    tool = load_tool()
    groups = tool.row_groups(items(), 300)
    assert [[item.question_id for item in group] for group in groups] == [
        ["q0", "q1"],  # q2 would make 3 x 120 = 360 > 300
        ["q2"],
        ["q3"],  # 400 alone is over the budget: its own group
        ["q4", "q5"],  # another state starts a group
    ]
    for group in groups:
        assert len({id(item.state) for item in group}) == 1
        assert (
            len(group) == 1 or len(group) * max(len(item.ids) for item in group) <= 300
        )


def test_no_budget_is_one_call_per_request():
    tool = load_tool()
    model = Rows()
    assert tool.run_rows(model, items(), None) == [f"q{i}" for i in range(6)]
    assert model.calls == [[f"q{i}" for i in range(6)]]


def test_grouped_answers_keep_the_items_order():
    tool = load_tool()
    model = Rows()
    assert tool.run_rows(model, items(), 300) == [f"q{i}" for i in range(6)]
    assert model.calls == [["q0", "q1"], ["q2"], ["q3"], ["q4", "q5"]]
