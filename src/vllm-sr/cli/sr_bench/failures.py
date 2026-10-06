"""Controlled public summaries; raw provider or subprocess text stays in evidence."""

import re


def failure_reason(message):
    text = str(message or "")
    if re.fullmatch(r"(?:Target|Preview) HTTP [1-5][0-9]{2}", text):
        return text
    lower = text.lower()
    for fragments, summary in (
        (("deadline", "wall-time"), "Request or case deadline exceeded"),
        (("readtimeout", "read timeout", "idle"), "Target idle/read timeout"),
        (("connectionerror", "connecttimeout"), "Target connection failed"),
        (("cancel",), "Request cancelled after the run stopped"),
        (("repeated output",), "Repeated output guard triggered"),
        (
            ("identity", "acknowledgement"),
            "Frozen runtime identity verification failed",
        ),
        (
            ("price", "pricing", "usage"),
            "Required usage or pricing evidence is incomplete",
        ),
        (("budget", "reservation"), "Frozen request or cost budget exhausted"),
        (("truncated",), "Auxiliary model output was truncated"),
        (("cap", "limit"), "Frozen output or call limit exceeded"),
        (("harness",), "Benchmark harness failed; inspect retained artifacts"),
        (("judge", "verdict", "grader"), "Benchmark grading failed"),
        (("incomplete final",), "Target final response was incomplete"),
    ):
        if any(fragment in lower for fragment in fragments):
            return summary
    return "Evaluation execution failed; inspect retained case evidence"


def first_saved_failure(results):
    failures = [
        row
        for row in results
        if row["status"] not in {"running", "completed"} and row.get("error")
    ]
    if not failures:
        return None
    row = min(failures, key=lambda item: item.get("finished_at", ""))
    return {
        "case_id": row["case_id"],
        "target_id": row["target_id"],
        "reason": failure_reason(row["error"]),
        "inferred_from_saved_results": True,
    }


def failure_summary(failure):
    case = re.sub(r"[^A-Za-z0-9._/-]", "_", failure["case_id"])[:128]
    target = re.sub(r"[^A-Za-z0-9._/-]", "_", failure["target_id"])[:128]
    return f"Case {case}, target {target}: {failure['reason']}"
