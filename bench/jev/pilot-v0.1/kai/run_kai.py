#!/usr/bin/env python3
"""Run the Kai arm of shared pilot v0.1 against a local Decision runtime.

Sequential, one attempt per request, no retries. One warmup request with a
text outside the pilot set is sent first and written to its own file. After a
call or contract failure the run continues and the failure is recorded.
Standard library only.
"""

import argparse
import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
MODEL_REVISION = "9d6872cde6950c2c2b5786d182ec9a06ca1bdd66"
RUNTIME_REVISION = "58cd660b51ba19aff01ad6f89e66f2d07f13908b"
QUESTION_KEY = "intent"
TIMEOUT_S = 60
PROBABILITY_SUM_TOLERANCE = 0.001
HTTP_OK = 200
WARMUP_STATE = (
    "Describe how a bicycle gear system changes the effort needed to pedal uphill."
)
TIMING_BOUNDARY = (
    "client perf_counter from just before the HTTP request is opened (body "
    "already serialized) until the response body is read and JSON-parsed; "
    "excludes contract validation and file writes"
)


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate JSON keys: {keys}")
    return dict(pairs)


def check_contract(parsed, labels):
    """Return (errors, answer). No renormalization, values kept as returned."""

    if not isinstance(parsed, dict):
        return [f"response is {type(parsed).__name__}, not an object"], None
    errors = []
    if parsed.get("model") != MODEL:
        errors.append(f"model echoed as {parsed.get('model')!r}")
    answers = parsed.get("answers")
    if not isinstance(answers, dict) or set(answers) != {QUESTION_KEY}:
        return [*errors, f"answers keys are not exactly [{QUESTION_KEY!r}]"], None
    answer = answers[QUESTION_KEY]
    if not isinstance(answer, dict):
        return [*errors, f"answer is {type(answer).__name__}, not an object"], None
    if answer.get("type") != "choice":
        errors.append(f"answer type is {answer.get('type')!r}")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        return [*errors, "probabilities missing or not an object"], answer
    if set(probabilities) != set(labels) or len(probabilities) != len(labels):
        missing = sorted(set(labels) - set(probabilities))
        extra = sorted(set(probabilities) - set(labels))
        errors.append(f"label set mismatch: missing={missing} extra={extra}")
    for label, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{label}: not a number")
        elif not 0.0 <= value <= 1.0:
            errors.append(f"{label}: {value!r} not finite in [0,1]")
    numbers = [
        v
        for v in probabilities.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0
    ]
    total = math.fsum(numbers)
    if not abs(total - 1.0) <= PROBABILITY_SUM_TOLERANCE:
        errors.append(f"sum {total!r} differs from 1 by more than 0.001")
    if answer.get("choice") not in labels:
        errors.append(f"choice {answer.get('choice')!r} is not a candidate label")
    return errors, answer


def call(base_url, body_bytes):
    """One attempt. Returns status, raw body, parsed JSON, error, elapsed ms."""

    request = urllib.request.Request(
        f"{base_url}/v1/systemone",
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, raw, parsed, error = None, None, None, None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            status = response.status
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw, object_pairs_hook=reject_duplicates)
    except urllib.error.HTTPError as http_error:
        status = http_error.code
        raw = http_error.read().decode("utf-8", errors="replace")
        error = f"HTTP {status}"
    except TimeoutError:
        error = f"timeout after {TIMEOUT_S} s"
    except urllib.error.URLError as url_error:
        error = f"URL error: {url_error.reason}"
    except http.client.IncompleteRead as read_error:
        raw = read_error.partial.decode("utf-8", errors="replace")
        error = f"response body incomplete: {read_error!r}"
    except (http.client.HTTPException, OSError) as connection_error:
        error = f"connection failed: {connection_error!r}"
    except ValueError as parse_error:
        error = f"response JSON invalid: {parse_error}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return status, raw, parsed, error, elapsed_ms


def run_one(base_url, state, question, labels):
    body = {"model": MODEL, "state": state, "questions": {QUESTION_KEY: question}}
    body_text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    timestamp = utc_now()
    status, raw, parsed, error, elapsed_ms = call(base_url, body_text.encode("utf-8"))
    record = {
        "timestamp": timestamp,
        "request_body": body_text,
        "http_status": status,
        "raw_response": raw,
        "attempts": 1,
        "timeout_ms": TIMEOUT_S * 1000,
        "elapsed_ms": elapsed_ms,
        "timing_boundary": TIMING_BOUNDARY,
    }
    errors, answer = (["no parsed response"], None)
    if parsed is not None and status == HTTP_OK:
        errors, answer = check_contract(parsed, labels)
    record.update(
        {
            "native_choice": answer.get("choice") if answer else None,
            "confidence": answer.get("confidence") if answer else None,
            "probabilities": answer.get("probabilities") if answer else None,
            "contract_valid": error is None and not errors,
            "contract_errors": errors,
            "error": error,
        }
    )
    probabilities = record["probabilities"]
    if record["contract_valid"]:
        top = max(probabilities.values())
        record["top1_from_probabilities"] = sorted(
            label for label, value in probabilities.items() if value == top
        )
        record["top1_probability"] = top
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18431")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads((args.pilot_dir / "protocol.json").read_text("utf-8"))
    inputs_path = args.pilot_dir / "inputs.jsonl"
    question_path = args.pilot_dir / "question.json"
    dataset_sha = sha256(inputs_path)
    question_sha = sha256(question_path)
    pins = protocol["files"]
    if (
        dataset_sha != pins["inputs"]["sha256"]
        or question_sha != pins["question_and_descriptions"]["sha256"]
    ):
        raise SystemExit(
            f"hash mismatch: inputs {dataset_sha}, question {question_sha}"
        )
    kai = protocol["models"]["kai"]
    if (
        kai["model"] != MODEL
        or kai["revision"] != MODEL_REVISION
        or kai["runtime_revision"] != RUNTIME_REVISION
        or protocol["question_key"] != QUESTION_KEY
    ):
        raise SystemExit("protocol pins differ from this script")

    question = json.loads(question_path.read_text("utf-8"))
    labels = protocol["label_order"]
    if list(question["criteria"]) != labels:
        raise SystemExit("question criteria differ from protocol label order")
    cases = [json.loads(line) for line in inputs_path.read_text("utf-8").splitlines()]
    scored = set(protocol["scored_cases"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run start {utc_now()}", flush=True)
    warmup = {"kind": "warmup", "state": WARMUP_STATE}
    warmup.update(run_one(args.base_url, WARMUP_STATE, question, labels))
    (args.out_dir / "kai-warmup.jsonl").write_text(
        json.dumps(warmup, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"warmup {warmup['native_choice']} {warmup['elapsed_ms']} ms", flush=True)

    with (args.out_dir / "kai-results.jsonl").open("w", encoding="utf-8") as out:
        for case in cases:
            record = {
                "schema": "kai-pilot-record.v1",
                "id": case["id"],
                "group": case["group"],
                "expected": case["expected"],
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "runtime_revision": RUNTIME_REVISION,
                "dataset_sha256": dataset_sha,
                "question_sha256": question_sha,
            }
            record.update(run_one(args.base_url, case["state"], question, labels))
            if case["id"] in scored:
                record["correct"] = (
                    record["native_choice"] == case["expected"]
                    if record["contract_valid"]
                    else None
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{case['id']} {record['http_status']} {record['native_choice']} "
                f"top={record.get('top1_probability')} valid={record['contract_valid']} "
                f"{record['elapsed_ms']} ms",
                flush=True,
            )
    print(f"run finish {utc_now()}", flush=True)


if __name__ == "__main__":
    main()
