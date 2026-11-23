"""Answer bank: per-case, per-model saved answers with correctness grades.

Built once by querying each candidate model directly (bypassing the router),
then joined into attribution records by (case_id, routed model). The join is
what turns replay-tier `no_evidence` rows into CORRECT / E2 / E3 outcomes
without generating any new answers.

Grading: the per-case grader is dispatched by `case["grader"]` (I). The
default is `contains` (case-insensitive substring against
`expected_contains`) - cheap, deterministic, auditable. `mcq` grades
multiple-choice answers by comparing the extracted option letter against
`case["expected"]`; verified (I): src/training/model_eval/mcq_judge.py is
importable as a plain function module (`extract_letter`, stdlib-only,
`__main__`-guarded), so grading calls it in-process and never shells out;
when the module is not on sys.path the error says so instead of failing
the grade silently.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import ContractError

_MCQ_LETTER: Any = None


@dataclass(frozen=True)
class BankEntry:
    case_id: str
    model: str
    answer: str
    answer_ok: bool | None  # None when the case declares no grading rule

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "answer": self.answer,
            "answer_ok": self.answer_ok,
        }


def _load_mcq_letter() -> Any:
    """Lazy-import mcq_judge.extract_letter (I, verified importable).

    Result is cached; tests may monkeypatch the module-level `_MCQ_LETTER`
    to grade without the training package on sys.path.
    """
    global _MCQ_LETTER
    if _MCQ_LETTER is None:
        try:
            from training.model_eval.mcq_judge import extract_letter  # type: ignore
        except ImportError as exc:
            raise ContractError(
                "the mcq grader needs src/ on sys.path to import "
                "training.model_eval.mcq_judge.extract_letter (verified importable); "
                f"import failed: {exc}"
            ) from exc
        _MCQ_LETTER = extract_letter
    return _MCQ_LETTER


def grade(answer: str, case: dict[str, Any]) -> bool | None:
    """Grade one answer against its case's declared grader (I).

    - grader "contains" (default when the case carries `expected_contains`):
      case-insensitive substring match; `expected_contains: null` means the
      case declares no grading rule -> None.
    - grader "mcq": `case["expected"]` is the correct option letter; the
      model answer's letter is extracted with mcq_judge.extract_letter. An
      unextractable letter is a wrong answer, never unevaluated - the
      sr-bench mcq_judge convention (no_answer counts as wrong).
    - any other grader name is a ContractError (never silently ungraded).
    """
    grader = case.get("grader")
    if grader is None:
        grader = "contains" if case.get("expected_contains") is not None else "none"
    if grader == "none":
        return None
    if grader == "contains":
        expected = case.get("expected_contains")
        if expected is None:
            return None
        return str(expected).strip().lower() in answer.strip().lower()
    if grader == "mcq":
        expected = case.get("expected")
        if expected is None:
            return None
        letter = _load_mcq_letter()(answer)
        return letter == str(expected).strip().upper()
    raise ContractError(f"case {case.get('case_id', '?')!r}: unknown grader {grader!r}")


def _chat_once(endpoint: str, model: str, prompt: str, timeout: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
    ) as exc:
        raise ContractError(
            f"answer bank query failed for model {model!r}: {exc}"
        ) from exc
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ContractError(
            f"answer bank query returned an unexpected payload for model {model!r}: {exc}"
        ) from exc


def build_bank(
    cases: list[dict[str, Any]],
    models: list[str],
    endpoint: str,
    *,
    timeout: float = 180.0,
) -> list[BankEntry]:
    """Query every case against every model directly (no router) and grade."""
    entries: list[BankEntry] = []
    for case in cases:
        for model in models:
            answer = _chat_once(endpoint, model, case["prompt"], timeout)
            entries.append(
                BankEntry(
                    case_id=case["case_id"],
                    model=model,
                    answer=answer,
                    answer_ok=grade(answer, case),
                )
            )
    return entries


def write_bank(entries: list[BankEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(
                json.dumps(entry.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            )
    return path


def load_bank(path: str | Path) -> dict[tuple[str, str], BankEntry]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"answer bank file not found: {path}")
    bank: dict[tuple[str, str], BankEntry] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"answer bank line {lineno} is not valid JSON: {exc}"
                ) from exc
            for name in ("case_id", "model", "answer", "answer_ok"):
                if name not in raw:
                    raise ContractError(
                        f"answer bank line {lineno} is missing required field: {name}"
                    )
            key = (raw["case_id"], raw["model"])
            if key in bank:
                raise ContractError(
                    f"answer bank line {lineno}: duplicate entry for {key}"
                )
            bank[key] = BankEntry(
                case_id=raw["case_id"],
                model=raw["model"],
                answer=str(raw["answer"]),
                answer_ok=(
                    raw["answer_ok"]
                    if raw["answer_ok"] is None
                    else bool(raw["answer_ok"])
                ),
            )
    return bank


def apply_bank(
    rows: list[dict[str, Any]], bank: dict[tuple[str, str], BankEntry]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join bank correctness into attribution rows by (case_id, routed model)."""
    stats = {"joined": 0, "no_bank_entry": 0}
    out: list[dict[str, Any]] = []
    for row in rows:
        entry = bank.get((row.get("case_id"), row.get("routed")))
        new_row = dict(row)
        if entry is None:
            new_row["answer_ok"] = None
            stats["no_bank_entry"] += 1
        else:
            new_row["answer_ok"] = entry.answer_ok
            stats["joined"] += 1
        out.append(new_row)
    return out, stats
