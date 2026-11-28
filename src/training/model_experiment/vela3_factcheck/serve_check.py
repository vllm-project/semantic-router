"""Replay the same requests against released, candidate and rollback workers."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from ...model_eval.provenance.metrics import latency_percentiles
from .data import NEGATIVE, POSITIVE
from .train import INSTRUCTIONS

REQUESTS = (
    "Calculate the total of four boxes of 17 red pencils and two boxes of 12 blue pencils.",
    "Which teams reached the semifinals of the 2022 FIFA World Cup? Cite the result.",
    "Our public timetable lists an 18:30 departure. At what time does this service leave?",
    "Rewrite the receipt I have not uploaded as a two-column table.",
)


def requests():
    for prompt in REQUESTS:
        yield {
            "state": {
                "request": prompt,
                "answer": "Contact Alex at alex@example.com for the details.",
            },
            "questions": {
                "factcheck": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "overflow": "truncate",
                    "criteria": {NEGATIVE: None, POSITIVE: None},
                },
                "domain": {
                    "type": "choice",
                    "instructions": "Which subject area is this request about?",
                },
                "feedback": {
                    "type": "choice",
                    "instructions": "What feedback does this user turn give about the previous answer?",
                },
                "pii": {"preset": "pii", "over": "answer"},
                "halu": {"preset": "halu"},
            },
        }


def request_json(endpoint, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = Request(endpoint.rstrip("/") + path, data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def shadow(primary, candidate):
    """Mirror fixed requests to a second worker and keep the released answers."""
    path = "/v1/decisions"
    rows = []
    with ThreadPoolExecutor(max_workers=1) as worker:
        for body in requests():
            unmirrored = request_json(primary, path, body)
            comparison = worker.submit(request_json, candidate, path, body)
            served = request_json(primary, path, body)
            if served["answers"] != unmirrored["answers"]:
                raise ValueError("the released responses changed during the shadow run")
            scores = [
                result["answers"]["factcheck"]["probabilities"][POSITIVE]
                for result in (served, comparison.result())
            ]
            rows.append({"primary": scores[0], "shadow": scores[1]})
    return rows


def observe(endpoint, rounds):
    inputs = list(requests())
    (model,) = request_json(endpoint, "/v1/models")["data"]
    identity = {
        key: model[key]
        for key in ("model_sha256", "engine", "device", "dtype", "engine_args")
    }
    for body in inputs:
        request_json(endpoint, "/v1/decisions", body)
    answers, timings = [], []
    for i in range(rounds):
        for body in inputs:
            started = time.perf_counter()
            result = request_json(endpoint, "/v1/decisions", body)
            timings.append((time.perf_counter() - started) * 1000)
            if i == 0:
                for name in body["questions"]:
                    answer = result["answers"][name]
                    if "error" in answer:
                        raise ValueError(f"{name} returned {answer['error']}")
                answers.append(
                    {
                        "answers": result["answers"],
                        "spans": result.get("spans", {}),
                        "thresholds": result.get("thresholds", {}),
                    }
                )
    return {
        "model": identity,
        "outputs": answers,
        "latency": latency_percentiles(timings),
    }


def compare(base, candidate, rollback):
    if base["model"] != rollback["model"]:
        raise ValueError("rollback worker has a different model identity or settings")
    base_digest = base["model"]["model_sha256"]
    candidate_digest = candidate["model"]["model_sha256"]
    if base_digest == candidate_digest or any(
        candidate["model"].get(key) != value
        for key, value in base["model"].items()
        if key != "model_sha256"
    ):
        raise ValueError(
            "expected distinct model identities with matching runtime settings"
        )
    expected = set(next(requests())["questions"])
    if any(len(run["outputs"]) != len(REQUESTS) for run in (base, candidate, rollback)):
        raise ValueError("incomplete HTTP replay")
    rows = []
    for b, c, r in zip(
        base["outputs"], candidate["outputs"], rollback["outputs"], strict=True
    ):
        if any(set(item["answers"]) != expected for item in (b, c, r)):
            raise ValueError("HTTP replay is missing expected answers")
        unrelated = expected - {"factcheck"}
        if b != r:
            raise ValueError("rollback did not restore the released response")
        if any(b["answers"][key] != c["answers"][key] for key in unrelated) or any(
            b[key] != c[key] for key in ("spans", "thresholds")
        ):
            raise ValueError("candidate changed an unrelated signal")
        rows.append(
            {
                side: entry["answers"]["factcheck"]["probabilities"][POSITIVE]
                for side, entry in (("released", b), ("candidate", c), ("rollback", r))
            }
        )
    if all(row["released"] == row["candidate"] for row in rows):
        raise ValueError("the candidate did not change any FactCheck response")
    return {
        "factcheck": rows,
        "released_model_sha256": base_digest,
        "candidate_model_sha256": candidate_digest,
        "released_latency": base["latency"],
        "candidate_latency": candidate["latency"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("endpoint")
    capture.add_argument("--rounds", type=int, default=20)
    mirror = commands.add_parser("shadow")
    mirror.add_argument("primary")
    mirror.add_argument("candidate")
    check = commands.add_parser("compare")
    for name in ("released", "candidate", "rollback"):
        check.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        result = observe(args.endpoint, args.rounds)
    elif args.command == "shadow":
        result = shadow(args.primary, args.candidate)
    else:
        result = compare(
            *(
                json.loads(path.read_text())
                for path in (args.released, args.candidate, args.rollback)
            )
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
