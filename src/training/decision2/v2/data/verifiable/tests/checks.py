"""Row-level assertions shared by the verifiable generator tests."""

from __future__ import annotations

import collections
import re
import unittest
from collections.abc import Iterable
from typing import Any

from v2.data.verifiable import core

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_PUNCT = re.compile(r"[,;?!()]|(?<!\d)[.:](?!\d)")


def texts(row: dict[str, Any]) -> list[str]:
    return [
        row["state"],
        row["instructions"],
        *(o["description"] for o in row["options"]),
    ]


def assert_language(case: unittest.TestCase, row: dict[str, Any]) -> None:
    if row["language"] == "zh":
        case.assertRegex(row["state"], CJK, row["id"])
        case.assertRegex(row["instructions"], CJK, row["id"])
        case.assertIn("。", row["state"], row["id"])
        for text in texts(row):
            case.assertIsNone(
                ASCII_PUNCT.search(text),
                f"{row['id']}: ASCII punctuation in {text[:80]!r}",
            )
    else:
        for text in texts(row):
            case.assertIsNone(
                CJK.search(text), f"{row['id']}: CJK text in an English row"
            )


def assert_no_answer_in_question(case: unittest.TestCase, row: dict[str, Any]) -> None:
    question = row["instructions"]
    if row["task_type"] == "noul":
        if row["language"] == "en":
            case.assertIsNone(
                re.search(r"\b(yes|no)\b", question, re.IGNORECASE), row["id"]
            )
        else:
            case.assertNotRegex(question, "[是否]", row["id"])
        return
    for option in row["options"]:
        case.assertNotIn(
            option["description"].casefold(), question.casefold(), row["id"]
        )


def assert_presence_balanced(case: unittest.TestCase, row: dict[str, Any]) -> None:
    if row["task_type"] != "choice" or row["source"].endswith(("a4h", "a4r")):
        return
    injected = row["audit_metadata"].get("injected_option")
    case.assertTrue(
        core.presence_ok(row["state"], row["options"], [injected] if injected else []),
        row["id"],
    )


def assert_oracles(case: unittest.TestCase, row: dict[str, Any]) -> None:
    meta = row["audit_metadata"]
    case.assertEqual(meta["oracle_label"], row["label"], row["id"])
    case.assertEqual(meta["reparse_label"], row["label"], row["id"])
    case.assertEqual(meta["generator"], core.GENERATOR)


def assert_counterfactual_groups(
    case: unittest.TestCase, rows: Iterable[dict[str, Any]]
) -> int:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for group_id, members in groups.items():
        first = members[0]
        for row in members:
            case.assertEqual(row["instructions"], first["instructions"], group_id)
            case.assertEqual(row["options"], first["options"], group_id)
        case.assertEqual(
            sorted(r["label"] for r in members),
            list(range(len(first["options"]))),
            group_id,
        )
        case.assertEqual(len({r["state"] for r in members}), len(members), group_id)
    return len(groups)


def assert_row(case: unittest.TestCase, row: dict[str, Any]) -> None:
    assert_oracles(case, row)
    assert_language(case, row)
    assert_no_answer_in_question(case, row)
    assert_presence_balanced(case, row)
