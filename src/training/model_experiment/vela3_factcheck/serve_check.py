"""Replay the same requests against released, candidate and rollback workers."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .data import POSITIVE
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
                "factcheck": {"type": "choice", "instructions": INSTRUCTIONS},
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


def observe(endpoint, rounds):
    from ...model_eval.provenance.metrics import latency_percentiles

    inputs = list(requests())
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
    return {"outputs": answers, "latency": latency_percentiles(timings)}


def compare(base, candidate, rollback):
    rows = []
    for b, c, r in zip(
        base["outputs"], candidate["outputs"], rollback["outputs"], strict=True
    ):
        unrelated = {key for key in b["answers"] if key != "factcheck"}
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
        "released_latency": base["latency"],
        "candidate_latency": candidate["latency"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("endpoint")
    capture.add_argument("--rounds", type=int, default=20)
    check = commands.add_parser("compare")
    for name in ("released", "candidate", "rollback"):
        check.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        result = observe(args.endpoint, args.rounds)
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
