"""Smoke tests of a Decision 2.5 package through its public entry points (needs torch + transformers).

    python -m d25.vega.release.smoke --model <package dir or Hub id> [--revision <sha>] [--device cuda:0] \
        [--card README.md] [--replace-repo vllm-sr/Decision-2.5-Vega-27B=vllm-sr/d25-vega-staging] \
        [--server] [--plain] --out smoke.json

- ``--card``: runs the README's Python Quickstart block verbatim in a fresh interpreter (``AutoModel`` from
  the Hub with ``trust_remote_code=True``), optionally pointing it at another repository.
- always: ``AutoModel.from_pretrained(model, trust_remote_code=True).system_one`` equals the runtime's own
  ``Decision25.system_one`` on the Quickstart request and the API probes; answer shapes are checked; the
  ``decision`` pipeline gives the same response; an over-limit question is answered ``max_length_exceeded``.
- ``--server``: starts ``decision25_server.py`` and checks ``/health``, ``POST /v1/systemone`` (same answers)
  and the HTTP 422 refusals (over-limit input, malformed question).
- ``--plain``: without ``trust_remote_code`` the repository loads as a stock ``Qwen3_5Model``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from d25.vega.release.examples import (
    PROBES,
    QUICKSTART,
    QUICKSTART_IMAGE,
    QUICKSTART_VIDEO,
)


def check_answer(question: dict, answer: dict) -> list[str]:
    problems = []
    kind = question.get("type")
    if "error" in answer:
        return [f"error answer {answer}"]
    if answer.get("type") != kind:
        return [f"type {answer.get('type')} for a {kind} question"]
    if kind == "noul":
        if not 0 <= answer["noul"] <= 1:
            problems.append("noul outside [0, 1]")
        return problems
    probabilities = answer["probabilities"]
    keys = (
        list(question["criteria"])
        if kind == "choice"
        else [str(i) for i in range(len(question["criteria"]))]
    )
    if list(probabilities) != keys:
        problems.append(f"probability keys {list(probabilities)} != {keys}")
    if abs(sum(probabilities.values()) - 1) > 1e-6 or any(
        not math.isfinite(v) or v < 0 for v in probabilities.values()
    ):
        problems.append("probabilities are not a distribution")
    if not 0 <= answer["confidence"] <= 1:
        problems.append("confidence outside [0, 1]")
    if kind == "choice" and answer["choice"] != max(keys, key=probabilities.get):
        problems.append("choice is not the argmax")
    if kind == "score":
        expected = sum(i * probabilities[k] for i, k in enumerate(keys))
        if abs(answer["score"] - expected) > 1e-9 or list(answer["legend"]) != keys:
            problems.append("score or legend inconsistent")
    return problems


def same(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def quickstart(card: Path, replace: list[str]) -> dict:
    text = card.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    if len(blocks) != 1:
        return {"ok": False, "problem": f"{len(blocks)} python blocks in the card"}
    code = blocks[0]
    for item in replace:
        old, new = item.split("=", 1)
        code = code.replace(old, new)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=3600
    )
    result = {
        "ok": proc.returncode == 0,
        "seconds": round(time.time() - started, 1),
        "stderr_tail": proc.stderr[-2000:],
    }
    if proc.returncode == 0:
        printed = json_objects(proc.stdout)
        answers = printed[0] if printed else {}
        problems = [
            p
            for key, q in QUICKSTART["questions"].items()
            for p in check_answer(q, answers.get(key, {}))
        ]
        if len(printed) > 1:
            # The image example of an image-capable card: the second printed object.
            image_answers = printed[1]
            problems += [
                p
                for key, q in QUICKSTART_IMAGE["questions"].items()
                for p in check_answer(q, image_answers.get(key, {}))
            ]
            result["image_answers"] = image_answers
        if len(printed) > 2:
            # The video example of a video-capable card: the third printed object.
            video_answers = printed[2]
            problems += [
                p
                for key, q in QUICKSTART_VIDEO["questions"].items()
                for p in check_answer(q, video_answers.get(key, {}))
            ]
            result["video_answers"] = video_answers
        result.update(answers=answers, problems=problems, ok=not problems)
    return result


def json_objects(text: str) -> list[dict]:
    """Every JSON object printed on stdout, in order (the Quickstart prints one per request)."""
    decoder, found, at = json.JSONDecoder(), [], text.find("{")
    while at != -1:
        value, end = decoder.raw_decode(text, at)
        found.append(value)
        at = text.find("{", end)
    return found


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_server(package_dir: Path, device: str, port: int, log):
    return subprocess.Popen(
        [
            sys.executable,
            str(
                next(
                    p
                    for p in (
                        package_dir / "d3_server.py",
                        package_dir / "decision25_server.py",
                    )
                    if p.is_file()
                )
            ),
            "--model",
            str(package_dir),
            "--device",
            device,
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _local_json(base: str, path: str, body=None):
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def close(a, b, tolerance: float = 1e-4) -> bool:
    """Same structure and labels, numbers within ``tolerance`` (another process may autotune other kernels)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tolerance) for k in a)
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
    ):
        return abs(a - b) <= tolerance
    return a == b


def server_checks(
    package_dir: Path,
    device: str,
    expected: dict,
    over_limit: dict,
    uncapped: bool = False,
) -> dict:
    import tempfile
    import urllib.error

    port = free_port()
    log_path = Path(tempfile.gettempdir()) / f"decision25-server-{port}.log"
    log = open(log_path, "w")
    proc = _spawn_server(package_dir, device, port, log)
    base = f"http://127.0.0.1:{port}"
    result: dict = {"ok": False}
    try:
        deadline = time.time() + 1800
        while time.time() < deadline:
            try:
                status, body = _local_json(base, "/health")
                if body.get("status") == "ready":
                    result["health"] = body
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with code {proc.returncode}")
            time.sleep(2)
        status, body = _local_json(
            base, "/v1/systemone", {"model": "default", **QUICKSTART}
        )
        result["systemone_status"] = status
        result["systemone_same"] = status == 200 and close(
            body["answers"], expected["answers"]
        )
        status, body = _local_json(
            base, "/v1/systemone", {"model": "default", **over_limit}
        )
        result["over_limit"] = {
            "status": status,
            "uncapped": uncapped,
            "ok": (
                status == 200
                if uncapped
                else "maximum context length" in json.dumps(body)
            ),
        }
        status, body = _local_json(
            base,
            "/v1/systemone",
            {
                "model": "default",
                "state": "x",
                "questions": {"q": {"type": "choice", "criteria": {}}},
            },
        )
        result["invalid_status"] = status
        result["ok"] = (
            result["systemone_same"]
            and result["over_limit"]["ok"]
            and (uncapped or result["over_limit"]["status"] == 422)
            and result["invalid_status"] == 422
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the report records why the server check failed
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if not result["ok"]:
        result["log_tail"] = log_path.read_text(errors="replace")[-3000:]
    return result


def over_limit_request(max_length: int) -> dict:
    """One question over the input limit and one that fits."""
    return {
        "state": " ".join(["filler"] * (max_length + 64)),
        "questions": {
            "q": {"type": "noul", "instructions": "Is it long?"},
            "ok": {"type": "noul", "instructions": "Short?"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--card", type=Path)
    ap.add_argument("--replace-repo", action="append", default=[])
    ap.add_argument("--server", action="store_true")
    ap.add_argument(
        "--server-only",
        action="store_true",
        help="only the server checks, in a process holding no model (one 27B copy fits a 96 GB GPU)",
    )
    ap.add_argument(
        "--expected",
        type=Path,
        help="with --server-only: an earlier smoke report (its Quickstart answers)",
    )
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    report: dict = {"model": args.model, "revision": args.revision}
    if args.server_only:
        package_dir = Path(args.model)
        expected = json.loads(args.expected.read_text())["checks"]["quickstart"][
            "response"
        ]
        limit = json.loads((package_dir / "decision_config.json").read_text()).get(
            "max_length"
        )
        report["server"] = server_checks(
            package_dir,
            args.device,
            expected,
            over_limit_request(limit or 8192),
            limit is None,
        )
        report["ok"] = report["server"]["ok"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "server": {
                        k: v for k, v in report["server"].items() if k != "log_tail"
                    },
                }
            )
        )
        return 0 if report["ok"] else 1
    if args.card:
        report["quickstart"] = quickstart(args.card, args.replace_repo)

    import torch
    from transformers import AutoModel, pipeline

    hub = {"revision": args.revision} if args.revision else {}
    started = time.time()
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, device=args.device, **hub
    )
    report["automodel_class"] = type(model).__name__
    report["load_seconds"] = round(time.time() - started, 1)
    runtime = model.runtime
    report["provenance"] = runtime.provenance()
    report["runtime"] = runtime.runtime_info()
    checks = {}
    for name, request in (("quickstart", QUICKSTART), ("probes", PROBES)):
        via_model = model.system_one(**request)
        direct = runtime.system_one(**request)
        problems = [
            p
            for key, q in request["questions"].items()
            if key != "broken"
            for p in check_answer(q, via_model["answers"][key])
        ]
        if (
            name == "probes"
            and via_model["answers"]["broken"].get("error") != "invalid_question"
        ):
            problems.append("a one-level score question must be invalid_question")
        checks[name] = {
            "response": via_model,
            "same_as_runtime": same(via_model, direct),
            "problems": problems,
        }
    over = over_limit_request(runtime.max_length or 8192)
    response = model.system_one(state="short", questions={"q": over["questions"]["q"]})
    overflow = model.system_one(**over)
    uncapped = runtime.max_length is None
    checks["over_limit"] = {
        "uncapped": uncapped,
        "answer": overflow["answers"]["q"].get("error") or "answered",
        "ok": "noul" in response["answers"]["q"]
        and (
            "noul" in overflow["answers"]["q"]
            if uncapped
            else overflow["answers"]["q"].get("error") == "max_length_exceeded"
        ),
    }
    package_dir = Path(runtime.root)
    report["checks"] = checks
    # One 27B copy at a time fits a 96 GB GPU: free each model before the next entry point loads its own.
    del model, runtime
    gc.collect()
    torch.cuda.empty_cache()
    decide = pipeline(
        "decision",
        model=args.model,
        trust_remote_code=True,
        device_map=args.device,
        **hub,
    )
    checks["pipeline_same"] = same(decide(QUICKSTART), checks["quickstart"]["response"])
    del decide
    gc.collect()
    torch.cuda.empty_cache()
    if args.server:
        report["server"] = server_checks(
            package_dir, args.device, checks["quickstart"]["response"], over, uncapped
        )
    if args.plain:
        plain = AutoModel.from_pretrained(args.model, device_map=args.device, **hub)
        report["plain_class"] = type(plain).__name__
        del plain
        gc.collect()
        torch.cuda.empty_cache()
    ok = (
        all(
            c["same_as_runtime"] and not c["problems"]
            for c in (checks["quickstart"], checks["probes"])
        )
        and checks["over_limit"]["ok"]
        and checks["pipeline_same"]
        and report.get("quickstart", {"ok": True})["ok"]
        and report.get("server", {"ok": True})["ok"]
        and report.get("plain_class", "Qwen3_5Model") == "Qwen3_5Model"
    )
    report["ok"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(
        json.dumps(
            {
                "ok": ok,
                "automodel_class": report["automodel_class"],
                "quickstart": report.get("quickstart", {}).get("ok"),
                "server": report.get("server", {}).get("ok"),
                "plain_class": report.get("plain_class"),
                "same": {
                    k: checks[k]["same_as_runtime"] for k in ("quickstart", "probes")
                },
                "problems": {
                    k: checks[k]["problems"] for k in ("quickstart", "probes")
                },
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
