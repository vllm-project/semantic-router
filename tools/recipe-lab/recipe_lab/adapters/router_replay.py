"""Adapter for the router's Router Replay observability API.

The router records one RoutingRecord per request for every decision that
mounts the `router_replay` plugin, into a store (memory/redis/postgres/...)
and exposes them through the management API:

    GET /api/v1/observability/replays?showDetails=true

The list endpoint is paginated: `limit` (default 20, max 100) and `offset`
query parameters, and the response body carries `data`, `has_more` and
`next_offset` (present only while has_more is true). A single unpaginated
GET therefore returns at most the first page and silently truncates, so
fetch_replays follows next_offset until the list is drained.

Verified against the router source (pkg/decision engine + classification
tests): `signal_values` / `signal_confidences` keys are `<type>:<name>`,
e.g. `decision:difficulty` or `projection:balance_reasoning`. For choice-
type decision signals the value is stored per choice under
`decision:<name>:<choice>` (e.g. `decision:domain:code` -> probability of
that choice), while score-type signals keep the numeric score under
`decision:<name>` (e.g. `decision:difficulty` -> 0..4). `decision_tier`
carries the routing tier as an integer when present.

This adapter pulls those records and joins them against a hand-labeled case
list (prompt -> truth label) to produce attribution input records:

    {"case_id": ..., "truth": ..., "matched_decision": <record.decision>,
     "routed": <record.selected_model>, "family": ..., "answer_ok": null}

The row field is matched_decision (F1): with string-only truth, a mismatched
decision is decision_mismatch, never E1_signal. Rows carry the signal
evidence (signal_values / signal_confidences / decision_tier) verbatim so
later diagnostics do not need another fetch.

`records_to_rows` is the unlabeled path: no labels, no join, one row per
record (including records without a prompt) for distribution and coverage
diagnostics (Sub-issue 7's diagnose consumes these).

answer_ok stays null here: correctness is joined later from saved answers
(Sub-issue 2's answer bank). The evidence-first rule handles the null.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contract import ContractError

REPLAYS_PATH = "/api/v1/observability/replays"


class ReplayFetchError(RuntimeError):
    """Raised when the management API cannot be reached or returns bad JSON."""


# The management API caps response bodies (HTTP 413); pagination floors here.
_MIN_PAGE_SIZE = 4


def _get_json(url: str, timeout: float, api_key_env: str | None) -> Any:
    """GET one JSON document from the management API (shared error mapping)."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    if api_key_env and os.environ.get(api_key_env):
        request.add_header("Authorization", f"Bearer {os.environ[api_key_env]}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ReplayFetchError(
            f"management API returned HTTP {exc.code}: {url}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReplayFetchError(f"cannot reach management API at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayFetchError(f"management API returned non-JSON body: {url}") from exc


def fetch_replays(
    management_url: str,
    *,
    show_details: bool = True,
    timeout: float = 30.0,
    api_key_env: str | None = None,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """Fetch all replay records from the router management API.

    `management_url` is the router management base URL, e.g.
    http://127.0.0.1:8080. The endpoint pages its responses (default limit
    20, max 100), so this follows `next_offset` while `has_more` is true;
    a single unpaginated GET would silently truncate to the first page.
    Returns the list of RoutingRecord dicts.
    """
    base = management_url.rstrip("/") + REPLAYS_PATH
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"{base}?limit={page_size}&offset={offset}"
        if show_details:
            url += "&showDetails=true"
        try:
            payload = _get_json(url, timeout, api_key_env)
        except ReplayFetchError as exc:
            # The management API caps response bodies (HTTP 413): a page of
            # detailed records can exceed the cap late in the list. Retry the
            # same offset with a halved page until it fits; a floor-sized page
            # that still overflows is a real failure.
            if "HTTP 413" in str(exc) and page_size > _MIN_PAGE_SIZE:
                page_size = max(_MIN_PAGE_SIZE, page_size // 2)
                continue
            raise
        page = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(page, list):
            raise ReplayFetchError(
                "unexpected replay list payload: expected {'data': [...]}"
            )
        records.extend(record for record in page if isinstance(record, dict))
        if not isinstance(payload, dict) or not payload.get("has_more"):
            break
        next_offset = payload.get("next_offset")
        if next_offset is None or int(next_offset) <= offset:
            break  # guard against a non-advancing cursor
        offset = int(next_offset)
    return records


@dataclass(frozen=True)
class CaseJoin:
    """Hand-labeled case list: one row per prompt sent through the router."""

    case_id: str
    prompt: str
    truth: str


def load_cases(path: str | Path) -> list[CaseJoin]:
    """Load the labeled case list (JSONL: case_id / prompt / truth)."""
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"cases file not found: {path}")
    cases: list[CaseJoin] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"cases line {lineno} is not valid JSON: {exc}"
                ) from exc
            for name in ("case_id", "prompt", "truth"):
                if not isinstance(raw.get(name), str) or not raw[name].strip():
                    raise ContractError(
                        f"cases line {lineno}: {name} must be a non-empty string"
                    )
            cases.append(
                CaseJoin(
                    case_id=raw["case_id"], prompt=raw["prompt"], truth=raw["truth"]
                )
            )
    return cases


def _record_field(record: dict, *names: str) -> Any:
    """Return the first present field (tolerates camelCase/snake_case wires)."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def record_to_case(
    record: dict[str, Any],
    cases_by_prompt: dict[str, CaseJoin],
    *,
    family: str = "d3",
) -> tuple[dict[str, Any] | None, str]:
    """Convert one RoutingRecord into an attribution input row.

    Returns (row, join_state) where join_state is "matched" or the skip
    reason ("no_prompt", "unlabeled"). Only matched records become rows.
    """
    prompt = _record_field(record, "prompt", "Prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "no_prompt"
    case = cases_by_prompt.get(prompt)
    if case is None:
        return None, "unlabeled"
    decision = _record_field(record, "decision", "Decision") or ""
    selected = _record_field(record, "selected_model", "SelectedModel") or ""
    request_id = _record_field(record, "request_id", "RequestID") or ""
    row = {
        "case_id": case.case_id,
        "truth": case.truth,
        "matched_decision": decision,
        "routed": selected,
        "family": family,
        "answer_ok": None,  # correctness is joined later from the answer bank
        "request_id": str(request_id),
        "signal_values": _record_field(record, "signal_values", "SignalValues") or {},
        "signal_confidences": _record_field(
            record, "signal_confidences", "SignalConfidences"
        )
        or {},
        "decision_tier": _record_field(record, "decision_tier", "DecisionTier"),
    }
    return row, "matched"


def records_to_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten replay records into diagnostic rows (no labels, no join).

    One row per record, including records without a prompt: coverage and
    distribution diagnostics want every request, not only labeled ones.
    Never raises on skip reasons; callers that need the labeled join should
    use records_to_cases instead.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = _record_field(record, "prompt", "Prompt")
        rows.append(
            {
                "request_id": str(
                    _record_field(record, "request_id", "RequestID") or ""
                ),
                "matched_decision": _record_field(record, "decision", "Decision") or "",
                "routed": _record_field(record, "selected_model", "SelectedModel")
                or "",
                "prompt": prompt if isinstance(prompt, str) else "",
                "signal_values": _record_field(record, "signal_values", "SignalValues")
                or {},
                "signal_confidences": _record_field(
                    record, "signal_confidences", "SignalConfidences"
                )
                or {},
                "decision_tier": _record_field(record, "decision_tier", "DecisionTier"),
            }
        )
    return rows


def records_to_cases(
    records: list[dict[str, Any]],
    cases: list[CaseJoin],
    *,
    family: str = "d3",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join replay records against the labeled case list.

    Returns (rows, join_stats). Duplicate prompts keep their first record;
    later duplicates count as skipped with reason "duplicate_prompt".
    """
    by_prompt: dict[str, CaseJoin] = {}
    for case in cases:
        by_prompt.setdefault(case.prompt, case)
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "matched": 0,
        "no_prompt": 0,
        "unlabeled": 0,
        "duplicate_prompt": 0,
    }
    seen_prompts: set[str] = set()
    for record in records:
        row, state = record_to_case(record, by_prompt, family=family)
        if state == "matched":
            prompt = _record_field(record, "prompt", "Prompt")
            if prompt in seen_prompts:
                stats["duplicate_prompt"] += 1
                continue
            seen_prompts.add(prompt)
            rows.append(row)
        stats[state] = stats.get(state, 0) + 1
    return rows, stats


def write_cases(cases: list[CaseJoin], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "prompt": case.prompt,
                        "truth": case.truth,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
