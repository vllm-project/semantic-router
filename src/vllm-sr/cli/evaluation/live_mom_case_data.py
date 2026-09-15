"""Load the versioned live Mixture-of-Models case catalog resource."""

from __future__ import annotations

from importlib.resources import files
from typing import cast

from cli.evaluation.canonical import strict_json_loads

_LiveMoMCaseRow = tuple[str, str, str, str, str]

_RESOURCE_PATH = ("resources", "live_mom_cases.v1.json")
_SCHEMA_VERSION = "live-mom-case-catalog.v1"
_COLUMNS = ("id", "prompt", "expected_answer", "domain", "difficulty")


def _load_live_mom_case_rows() -> tuple[_LiveMoMCaseRow, ...]:
    payload = strict_json_loads(
        files("cli.evaluation").joinpath(*_RESOURCE_PATH).read_bytes()
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "columns",
        "cases",
    }:
        raise RuntimeError("live MoM case catalog must use the versioned object schema")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise RuntimeError("live MoM case catalog schema version is unsupported")
    columns = payload["columns"]
    if not isinstance(columns, list) or tuple(columns) != _COLUMNS:
        raise RuntimeError("live MoM case catalog columns do not match the v1 contract")

    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("live MoM case catalog must contain cases")
    rows: list[_LiveMoMCaseRow] = []
    for index, raw_row in enumerate(cases):
        if (
            not isinstance(raw_row, list)
            or len(raw_row) != len(_COLUMNS)
            or not all(
                isinstance(value, str) and bool(value) and value.strip() == value
                for value in raw_row
            )
        ):
            raise RuntimeError(f"live MoM case row {index} is malformed")
        rows.append(cast(_LiveMoMCaseRow, tuple(raw_row)))

    case_ids = [row[0] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("live MoM case catalog contains duplicate case IDs")
    return tuple(rows)


LIVE_MOM_CASE_ROWS = _load_live_mom_case_rows()
