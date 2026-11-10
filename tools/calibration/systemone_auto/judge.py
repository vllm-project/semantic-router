"""Independent, prior-answer-conditioned judge collection and scoring."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

from .artifacts import canonical, digest, file_digest, read_json, read_jsonl, write_json
from .collection import CHAT_MODEL, request_once, validate_targets
from .dataset import validate_dataset
from .metrics import evaluate_response, summarize

MAX_JUDGE_OUTPUT_TOKENS = 128
PROMPT_ROLES = ["system", "user"]


def validate_inputs(rows: list[dict], manifest: dict) -> None:
    if manifest.get("schema_version") != "systemone-judge-inputs/v1":
        raise ValueError("unknown judge input manifest")
    if manifest.get("request_builder") != "systemone.JudgeRequest":
        raise ValueError("judge inputs must use the serving request builder")
    identifiers = set()
    for row in rows:
        if not row.get("record_id") or row["record_id"] in identifiers:
            raise ValueError("duplicate or missing judge source ID")
        identifiers.add(row["record_id"])
        payload = row["request"]
        if digest(payload) != row["judge_request_sha256"]:
            raise ValueError("judge request digest mismatch")
        if (
            payload.get("temperature") != 0
            or payload.get("max_tokens") != MAX_JUDGE_OUTPUT_TOKENS
        ):
            raise ValueError("judge generation settings differ from the frozen pilot")
        if payload.get("model") != manifest["served_model_id"]:
            raise ValueError("judge request targets a different model")
        messages = payload.get("messages", [])
        if [m.get("role") for m in messages] != PROMPT_ROLES:
            raise ValueError("judge prompt must retain the original two messages")
        if digest(messages[0]["content"]) != manifest["system_prompt_sha256"]:
            raise ValueError("judge system prompt changed")
        data = json.loads(messages[1]["content"])
        if set(data) != {"request", "candidates"}:
            raise ValueError("judge data must contain only request and candidates")
        if digest(data["request"]) != row["original_request_sha256"]:
            raise ValueError("judge original request changed")
        if set(data["candidates"]) != set(manifest["candidate_stages"]):
            raise ValueError("judge candidate pool changed")
        if {k: digest(v) for k, v in data["candidates"].items()} != row[
            "native_candidate_sha256"
        ]:
            raise ValueError("judge must see the actual frozen native answers")
        expected = {
            "type": "json_schema",
            "json_schema": {
                "name": "systemone_judge",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["selected"],
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": ["abstain", *manifest["candidate_stages"]],
                        }
                    },
                },
            },
        }
        if payload.get("response_format") != expected:
            raise ValueError("judge response schema changed")
    if len(rows) != manifest["requests"]:
        raise ValueError("judge input count mismatch")


def selection(raw: dict, stages: list[str]) -> tuple[str | None, str]:
    try:
        choices = raw["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            return None, "incomplete"
        value = json.loads(choices[0]["message"]["content"])
        if not isinstance(value, dict) or set(value) != {"selected"}:
            return None, "invalid_selection"
        chosen = value["selected"]
        if chosen == "abstain":
            return None, "abstained"
        if not isinstance(chosen, str) or chosen not in stages:
            return None, "invalid_selection"
        return chosen, "selected"
    except (KeyError, IndexError, TypeError, ValueError):
        return None, "invalid_selection"


def collect_judge(
    request_path: Path,
    manifest_path: Path,
    target_path: Path,
    deployment_receipt: Path,
    output: Path,
    *,
    limit: int | None = None,
) -> dict:
    rows, inputs = read_jsonl(request_path), read_json(manifest_path)
    validate_inputs(rows, inputs)
    if file_digest(request_path) != inputs["requests_sha256"]:
        raise ValueError("judge input file digest mismatch")
    target = read_json(target_path)
    validate_targets([target])
    if target["protocol"] != "chat" or target["model_id"] != CHAT_MODEL:
        raise ValueError("the judge pilot permits only Qwen Flash Next")
    if any(target[k] != inputs[k] for k in ("model_id", "revision")):
        raise ValueError("judge model identity changed")
    deployment = read_json(deployment_receipt)
    if (
        deployment.get("model_id") != target["model_id"]
        or deployment.get("revision") != target["revision"]
        or deployment.get("default_chat_template_kwargs") != {"enable_thinking": False}
    ):
        raise ValueError("judge deployment must bind the pinned nonthinking default")
    if limit is not None and limit <= 0:
        raise ValueError("judge smoke limit must be positive")
    identity = {
        "schema_version": "systemone-judge-collection/v1",
        "input_manifest_sha256": file_digest(manifest_path),
        "dataset_sha256": inputs["dataset_sha256"],
        "deployment_receipt_sha256": file_digest(deployment_receipt),
        "collector_source_sha256": digest(
            {
                name: file_digest(Path(__file__).with_name(name))
                for name in ("judge.py", "collection.py", "artifacts.py")
            }
        ),
        "model_id": target["model_id"],
        "revision": target["revision"],
        "timeout_seconds": 120,
        "retries": 0,
        "concurrency": 1,
        "max_model_calls": 1,
        "max_output_tokens": MAX_JUDGE_OUTPUT_TOKENS,
    }
    collection_id = digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path, trace_path = output / "collection.json", output / "observations.jsonl"
    if (
        receipt_path.exists()
        and read_json(receipt_path)["collection_identity"] != collection_id
    ):
        raise ValueError("judge collection identity changed; use a separate directory")
    existing = read_jsonl(trace_path) if trace_path.exists() else []
    by_id = {row["record_id"]: row for row in rows}
    done = set()
    for observation in existing:
        record_id = observation["record_id"]
        if (
            record_id in done
            or record_id not in by_id
            or observation["collection_identity"] != collection_id
            or observation["judge_request_sha256"]
            != by_id[record_id]["judge_request_sha256"]
        ):
            raise ValueError("judge resume observation identity mismatch")
        done.add(record_id)

    def save_receipt() -> dict:
        receipt = {
            **identity,
            "collection_identity": collection_id,
            "observation_count": len(done),
            "expected_count": len(rows),
            "complete": len(done) == len(rows),
        }
        write_json(receipt_path, receipt)
        return receipt

    save_receipt()
    for row in rows if limit is None else rows[:limit]:
        if row["record_id"] in done:
            continue
        key = os.environ.get(target.get("api_key_env", ""))
        status, raw, elapsed = request_once(
            target["endpoint"], key, row["request"], 120
        )
        chosen, outcome = selection(raw, inputs["candidate_stages"])
        if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            chosen, outcome = None, "request_failed"
        observation = {
            "collection_identity": collection_id,
            "record_id": row["record_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "judge_request_sha256": row["judge_request_sha256"],
            "http_status": status,
            "raw_response": raw,
            "client_elapsed_ms": elapsed,
            "selected": chosen,
            "outcome": outcome,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        with trace_path.open("a") as stream:
            stream.write(canonical(observation) + "\n")
        done.add(row["record_id"])
        save_receipt()
    return save_receipt()


def score_judge(dataset_path: Path, request_path: Path, collection: Path) -> dict:
    """Component quality; abstention/failure is unresolved, never a fabricated answer."""
    dataset = read_json(dataset_path)
    validate_dataset(dataset)
    rows = {row["record_id"]: row for row in read_jsonl(request_path)}
    observations = read_jsonl(collection / "observations.jsonl")
    receipt = read_json(collection / "collection.json")
    if receipt["dataset_sha256"] != digest(dataset):
        raise ValueError("judge scoring dataset differs from the frozen inputs")
    if not receipt["complete"] or len(observations) != len(rows):
        raise ValueError(
            "judge scoring requires the complete paired component collection"
        )
    observed = {o["record_id"]: o for o in observations}
    if set(observed) != set(rows):
        raise ValueError("judge scoring has duplicate or missing sources")
    groups = {name: [] for name in ("train", "calibration", "held_out", "diagnostic")}
    outcomes = {}
    for source in dataset["records"]:
        row, observation = rows[source["id"]], observed[source["id"]]
        if (
            observation["collection_identity"] != receipt["collection_identity"]
            or observation["judge_request_sha256"] != row["judge_request_sha256"]
        ):
            raise ValueError("judge scoring observation identity mismatch")
        data = json.loads(row["request"]["messages"][1]["content"])
        if digest(source["request"]) != digest(data["request"]):
            raise ValueError("judge scoring dataset request mismatch")
        chosen, outcome = selection(
            observation["raw_response"], list(data["candidates"])
        )
        if (
            not HTTPStatus.OK
            <= observation["http_status"]
            < HTTPStatus.MULTIPLE_CHOICES
        ):
            chosen, outcome = None, "request_failed"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        result = evaluate_response(
            source, data["candidates"].get(chosen, {}), native=True
        )
        split = source["split"] if source["cohort"] == "public" else "diagnostic"
        groups[split].append(result)
    return {
        "schema_version": "systemone-judge-component-results/v1",
        "collection_identity": receipt["collection_identity"],
        "trace_sha256": file_digest(collection / "observations.jsonl"),
        "outcomes": outcomes,
        "splits": {name: summarize(results) for name, results in groups.items()},
        "interpretation": "Actual prior-answer-conditioned selector component; no three-stage runtime latency or certified quality claim. Every failed/abstained selection remains an unresolved bundle.",
    }
