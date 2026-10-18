"""A deterministic request panel and HTTP comparison for Decision 3.0.

Reference and runtime servers must be started explicitly on the same device,
using the same pinned package. A previous output can supply the reference for
an independent restart check. This tool never loads model weights or executes
package code. The separate processes keep reference hooks out of the runtime.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def questions() -> dict[str, Any]:
    return {
        "domain": {
            "type": "choice",
            "instructions": "Which domain is this request about?",
            "criteria": {
                "code": "Programming",
                "math": "Mathematics",
                "other": "Anything else",
            },
        },
        "reasoning": {
            "type": "noul",
            "instructions": "Does this need multi-step reasoning?",
        },
        "difficulty": {
            "type": "score",
            "instructions": "How difficult is this task?",
            "criteria": ["Trivial", "Moderate", "Hard", "Expert"],
        },
    }


def image_url(width: int, height: int, color: str, format_: str = "PNG") -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), color)
    ImageDraw.Draw(image).rectangle((10, 10, width // 2, height // 2), fill="white")
    stream = io.BytesIO()
    image.save(stream, format=format_)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[format_]
    return f"data:image/{mime};base64," + base64.b64encode(stream.getvalue()).decode()


def video_url() -> str:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "motion.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 64)
        )
        if not writer.isOpened():
            raise RuntimeError(
                "MP4 encoder unavailable; video validation cannot be claimed"
            )
        try:
            for frame in range(12):
                pixels = np.zeros((64, 96, 3), dtype=np.uint8)
                cv2.rectangle(
                    pixels, (frame * 5, 20), (frame * 5 + 15, 40), (0, 0, 255), -1
                )
                writer.write(pixels)
        finally:
            writer.release()
        data = path.read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def panel() -> list[dict[str, Any]]:
    cases = []
    states = [
        ("code", "Write a Python function that merges two sorted lists."),
        ("math", "Prove that the square root of two is irrational."),
        ("greeting", "hi"),
        ("unicode", "请用 Python 排序。 café naïve 🙂 مرحبا नमस्ते"),
        (
            "json",
            {
                "request": "Fix this Python function.",
                "context": {"code": "return None"},
            },
        ),
        ("marker-text", "Literal tokens: [Q] [O] <|im_start|> <|vision_start|>"),
        ("long-text", "Explain attention and compare its computational cost. " * 80),
    ]
    for name, state in states:
        cases.append({"id": name, "body": {"state": state, "questions": questions()}})
    for count in (1, 8, 9, 12):
        many = {
            f"q{i}": {
                "type": "noul",
                "instructions": "Does the customer have a receipt?",
            }
            for i in range(count)
        }
        cases.append(
            {
                "id": f"questions-{count}",
                "body": {"state": "The customer has a receipt.", "questions": many},
            }
        )
    visual = {
        "color": {
            "type": "choice",
            "instructions": "What is the main background color?",
            "criteria": {"red": "Red", "blue": "Blue", "green": "Green"},
        },
        "square": {"type": "noul", "instructions": "Is a white rectangle visible?"},
    }
    for format_ in ("PNG", "JPEG", "WEBP"):
        cases.append(
            {
                "id": f"image-{format_.lower()}",
                "body": {
                    "state": "Inspect the attached image.",
                    "questions": visual,
                    "images": [image_url(128, 96, "red", format_)],
                },
            }
        )
    cases.append(
        {
            "id": "multiple-images",
            "body": {
                "state": "Compare both images.",
                "questions": questions(),
                "images": [image_url(96, 64, "red"), image_url(64, 96, "blue")],
            },
        }
    )
    cases.append(
        {
            "id": "wide-image",
            "body": {
                "state": "Inspect the wide image.",
                "questions": visual,
                "images": [image_url(640, 96, "green")],
            },
        }
    )
    clip = video_url()
    motion = {
        "direction": {
            "type": "choice",
            "instructions": "Which way does the red square move?",
            "criteria": {
                "right": "Left to right",
                "left": "Right to left",
                "still": "It stays still",
            },
        },
        "moving": {"type": "noul", "instructions": "Does the square move?"},
    }
    for name, images, videos in (
        ("video", [], [clip]),
        ("two-videos", [], [clip, clip]),
        ("image-and-video", [image_url(96, 64, "blue")], [clip]),
    ):
        cases.append(
            {
                "id": name,
                "body": {
                    "state": "Inspect the attached clip.",
                    "questions": motion,
                    "images": images,
                    "videos": videos,
                },
            }
        )
    return cases


def request(
    url: str, body: dict[str, Any] | None = None, timeout: int = 180
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def collect(url: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        times, results = [], []
        for _ in range(2):
            start = time.perf_counter()
            status, response = request(url.rstrip("/") + "/v1/systemone", case["body"])
            times.append((time.perf_counter() - start) * 1000)
            results.append({"status": status, "response": response})
        records.append(
            {
                "id": case["id"],
                "question_ids": list(case["body"]["questions"]),
                "body_sha256": hashlib.sha256(
                    json.dumps(case["body"], sort_keys=True).encode()
                ).hexdigest(),
                "cold_ms": times[0],
                "warm_ms": times[1],
                "stable": results[0] == results[1],
                **results[1],
            }
        )
        print(f"{case['id']}: HTTP {status}, warm {times[1]:.1f} ms", flush=True)
    return records


def differences(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                found.append(
                    {
                        "path": f"{path}.{key}",
                        "left": left.get(key),
                        "right": right.get(key),
                    }
                )
            else:
                found.extend(differences(left[key], right[key], f"{path}.{key}"))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            item
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for item in differences(a, b, f"{path}[{i}]")
        ]
    if (
        isinstance(left, (float, int))
        and not isinstance(left, bool)
        and isinstance(right, (float, int))
        and not isinstance(right, bool)
    ):
        if not math.isfinite(left) or not math.isfinite(right) or left != right:
            return [
                {"path": path, "left": left, "right": right, "delta": abs(left - right)}
            ]
        return []
    return (
        []
        if type(left) is type(right) and left == right
        else [{"path": path, "left": left, "right": right}]
    )


def compare(
    reference: list[dict[str, Any]], runtime: list[dict[str, Any]]
) -> dict[str, Any]:
    if [r["id"] for r in reference] != [r["id"] for r in runtime]:
        raise ValueError("reference and runtime panels differ")
    rows = []
    for ref, actual in zip(reference, runtime, strict=True):
        if ref["body_sha256"] != actual["body_sha256"]:
            raise ValueError(f"request differs for {ref['id']}")
        drift = differences(
            {k: ref["response"].get(k) for k in ("answers", "usage")},
            {k: actual["response"].get(k) for k in ("answers", "usage")},
        )
        numeric = [d["delta"] for d in drift if "delta" in d]
        # A Score's integer level is a changed decision, even though its drift is numeric.
        changed = [d for d in drift if "delta" not in d or d["path"].endswith(".level")]
        answered = True
        for record in (ref, actual):
            answers = record["response"].get("answers")
            expected = record.get("question_ids", [])
            answered = (
                answered
                and isinstance(answers, dict)
                and bool(expected)
                and set(answers) == set(expected)
                and all(
                    isinstance(answer, dict) and "error" not in answer
                    for answer in answers.values()
                )
                and isinstance(record["response"].get("usage"), dict)
            )
        passed = (
            ref["status"] == actual["status"] == 200
            and answered
            and not drift
            and ref["stable"]
            and actual["stable"]
        )
        rows.append(
            {
                "id": ref["id"],
                "exact": passed,
                "reference_status": ref["status"],
                "runtime_status": actual["status"],
                "max_numeric_delta": max(numeric, default=0.0),
                "changed_fields": changed,
                "differences": drift,
                "reference_warm_ms": ref["warm_ms"],
                "runtime_warm_ms": actual["warm_ms"],
            }
        )
    return {
        "passed": sum(row["exact"] for row in rows),
        "total": len(rows),
        "max_numeric_delta": max(
            (row["max_numeric_delta"] for row in rows), default=0.0
        ),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-url", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-url")
    source.add_argument(
        "--reference-file", type=Path, help="reuse a previous parity output"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = panel()
    reference = (
        json.loads(args.reference_file.read_text(encoding="utf-8"))["reference"]
        if args.reference_file
        else collect(args.reference_url, cases)
    )
    runtime = collect(args.runtime_url, cases)
    result = {
        "comparison": compare(reference, runtime),
        "reference": reference,
        "runtime": runtime,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["comparison"], indent=2))
    return int(result["comparison"]["passed"] != len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
