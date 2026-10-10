"""Smoke test of the image path: AutoModel, pipeline, kit engine and the HTTP server.

- ``AutoModel.from_pretrained(package, trust_remote_code=True).system_one(..., images=[example])``: the card
  Quickstart with the rendered example receipt; answers recorded. Requests with 4, 5 and 6 images (no image
  count cap) answer every question. The ``decision`` pipeline and the kit engine
  (``decision25_engine.Decision25Engine``) must give the same probabilities for 1, 4 and 6 images; a literal
  image placeholder in the text and an over-limit image request are refused per question.
- ``decision25_server.py`` in its own process, started after the in-process checks' child process has exited
  (one 27B copy at a time, so a 96 GB GPU fits): ``/health`` and ``/v1/models`` advertise images (no image
  count); text requests answer the same with ``images`` absent, null or empty; requests with 1, 4, 5 and 6
  data-URL images answer; GIF data, a PNG over 8,000,000 bytes, an image over 16,000,000 pixels, invalid
  base64, truncated PNG data, an http URL and a non-list get HTTP 422.

    python -m d25.omni.runtime.smoke --package PKG --suite SUITE --example example-receipt.png --out smoke.json
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from d25.omni.runtime.common import data_url, read_jsonl, sha_key, write_json

QUICKSTART_STATE = (
    "The customer says the blender arrived cracked and attached the receipt."
)
QUICKSTART_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "on_receipt": {
        "type": "noul",
        "instructions": "Does the receipt list the blender?",
    },
    "payment": {
        "type": "choice",
        "instructions": "How was the order paid?",
        "criteria": {"card": None, "cash": None, "gift card": None},
    },
}


def probabilities(response: dict) -> dict[str, list[float]]:
    out = {}
    for key, answer in response["answers"].items():
        if "probabilities" in answer:
            out[key] = list(answer["probabilities"].values())
        elif "noul" in answer:
            out[key] = [1 - answer["noul"], answer["noul"]]
    return out


def max_dp(a: dict, b: dict) -> float:
    pa, pb = probabilities(a), probabilities(b)
    if set(pa) != set(pb):
        return float("inf")
    return max(max(abs(x - y) for x, y in zip(pa[k], pb[k])) for k in pa)


def png_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def post(base: str, body: dict, timeout: float = 600) -> tuple[int, dict | str]:
    request = urllib.request.Request(
        base + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        return exc.code, text[:400]


def get(base: str, path: str) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:400]
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return 0, str(exc)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def release() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--example", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--phase", choices=("all", "in-process", "server"), default="all")
    ap.add_argument(
        "--in-process",
        type=Path,
        help="report of the in-process phase (for --phase server)",
    )
    args = ap.parse_args()
    if args.phase == "all":
        # The server must start in a process that holds no model copy: a 96 GB GPU holds one 27B copy at a
        # time, and the pipeline step does not give its copy back to the caching allocator.
        partial = args.out.with_name(args.out.stem + ".in-process.json")
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "d25.omni.runtime.smoke",
                "--phase",
                "in-process",
                "--package",
                str(args.package),
                "--suite",
                str(args.suite),
                "--example",
                str(args.example),
                "--device",
                args.device,
                "--out",
                str(partial),
            ]
        )
        if not partial.exists():
            raise SystemExit(
                f"in-process checks wrote no report (exit {child.returncode})"
            )
        args.phase, args.in_process = "server", partial

    from PIL import Image

    rows = sorted(
        read_jsonl(args.suite / "rows.jsonl.gz"), key=lambda r: sha_key(r["id"])
    )
    suite_images = [args.suite / r["images"][0] for r in rows if r.get("images")][:5]
    # The receipt first, then distinct suite images; five copies of the receipt for the old over-cap case.
    multi = {n: [args.example] + suite_images[: n - 1] for n in (4, 6)}
    multi[5] = [args.example] * 5
    if args.phase == "server":
        report: dict = json.loads(args.in_process.read_text())
        quick = report["quickstart"]["response"]
        text_only = report["quickstart_without_image"]
        many = {int(n): r for n, r in report["multi_image"].items()}
    else:
        report = {"package": str(args.package), "checks": {}}
    checks = report["checks"]

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(args.out.with_suffix(".server.log"), "w")
    server = None
    try:
        if args.phase == "in-process":
            # In-process paths, one model copy at a time.
            sys.path.insert(0, str(args.package))
            from transformers import AutoModel, pipeline

            model = AutoModel.from_pretrained(str(args.package), trust_remote_code=True)
            report["automodel_class"] = type(model).__name__
            started = time.time()
            quick = model.system_one(
                state=QUICKSTART_STATE,
                questions=QUICKSTART_QUESTIONS,
                images=[str(args.example)],
            )
            report["quickstart"] = {
                "seconds": round(time.time() - started, 2),
                "response": quick,
            }
            report["quickstart_answers_as_expected"] = (
                quick["answers"]["route"].get("choice") == "returns"
                and quick["answers"]["on_receipt"].get("noul", 0) > 0.5
                and quick["answers"]["payment"].get("choice") == "card"
            )
            checks["quickstart_answered"] = all(
                "error" not in a for a in quick["answers"].values()
            )
            text_only = model.system_one(
                state=QUICKSTART_STATE, questions=QUICKSTART_QUESTIONS
            )
            report["quickstart_without_image"] = text_only
            checks["image_changes_tokens"] = (
                quick["usage"]["input_tokens"] > text_only["usage"]["input_tokens"]
            )
            conflict = model.system_one(
                state="See <|image_pad|> here.",
                questions={"q": {"type": "noul"}},
                images=[str(args.example)],
            )
            checks["placeholder_refused"] = (
                conflict["answers"]["q"].get("error") == "invalid_question"
            )
            runtime = model.runtime
            saved, runtime.max_length = runtime.max_length, 500
            over = runtime.system_one(
                state="x", questions={"q": {"type": "noul"}}, images=[str(args.example)]
            )
            runtime.max_length = saved
            checks["over_limit_refused"] = over["answers"]["q"].get(
                "error"
            ) == "max_length_exceeded" and "maximum context length" in over["answers"][
                "q"
            ].get(
                "message", ""
            )
            many = {}
            for n, images in sorted(multi.items()):
                try:
                    many[n] = model.system_one(
                        state=QUICKSTART_STATE,
                        questions=QUICKSTART_QUESTIONS,
                        images=[str(p) for p in images],
                    )
                    checks[f"automodel_{n}_images_answered"] = all(
                        "error" not in a for a in many[n]["answers"].values()
                    )
                except ValueError as exc:
                    report[f"automodel_{n}_images_error"] = str(exc)[:300]
                    checks[f"automodel_{n}_images_answered"] = False
            report["multi_image"] = many
            report["multi_image_input_tokens"] = {
                n: r["usage"]["input_tokens"] for n, r in many.items()
            }
            checks["more_images_more_tokens"] = len(many) == 3 and (
                quick["usage"]["input_tokens"]
                < many[4]["usage"]["input_tokens"]
                < many[6]["usage"]["input_tokens"]
            )
            kinds = {
                "pil": [Image.open(args.example)],
                "data_url": [data_url(args.example)],
                "path_object": [args.example],
            }
            checks["input_kinds_identical"] = all(
                max_dp(
                    quick,
                    runtime.system_one(
                        state=QUICKSTART_STATE, questions=QUICKSTART_QUESTIONS, images=v
                    ),
                )
                == 0
                for v in kinds.values()
            )
            del model, runtime
            release()

            decide = pipeline(
                "decision", model=str(args.package), trust_remote_code=True
            )
            piped = decide(
                state=QUICKSTART_STATE,
                questions=QUICKSTART_QUESTIONS,
                images=[str(args.example)],
            )
            piped_dict = decide(
                {
                    "state": QUICKSTART_STATE,
                    "questions": QUICKSTART_QUESTIONS,
                    "images": [str(args.example)],
                }
            )
            checks["pipeline_equals_automodel"] = (
                max_dp(quick, piped) == 0 and max_dp(quick, piped_dict) == 0
            )
            for n in (4, 6):
                piped_n = decide(
                    state=QUICKSTART_STATE,
                    questions=QUICKSTART_QUESTIONS,
                    images=[str(p) for p in multi[n]],
                )
                checks[f"pipeline_{n}_images_equal_automodel"] = (
                    n in many and max_dp(many[n], piped_n) == 0
                )
            del decide
            release()

            if (args.package / "d3_engine.py").is_file():
                from d3_engine import D3Engine as Decision25Engine
            else:
                from decision25_engine import Decision25Engine
            from decision_index.engines import Unsupported

            engine = Decision25Engine(model=str(args.package), device=args.device)
            response, _ = engine(
                QUICKSTART_STATE, QUICKSTART_QUESTIONS, images=[str(args.example)]
            )
            checks["kit_engine_equals_automodel"] = max_dp(quick, response) == 0
            for n in (4, 5, 6):
                try:
                    response_n, _ = engine(
                        QUICKSTART_STATE,
                        QUICKSTART_QUESTIONS,
                        images=[str(p) for p in multi[n]],
                    )
                    checks[f"kit_engine_{n}_images_equal_automodel"] = (
                        n in many and max_dp(many[n], response_n) == 0
                    )
                except Unsupported as exc:
                    report[f"kit_engine_{n}_images_error"] = str(exc)[:300]
                    checks[f"kit_engine_{n}_images_equal_automodel"] = False
            text_engine, _ = engine(QUICKSTART_STATE, QUICKSTART_QUESTIONS)
            checks["kit_engine_text_equals_automodel"] = (
                max_dp(text_only, text_engine) == 0
            )
            del engine
            release()

        else:
            # Server in its own process; the in-process child has exited (one 27B copy fits a 96 GB GPU).
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(
                        next(
                            p
                            for p in (
                                args.package / "d3_server.py",
                                args.package / "decision25_server.py",
                            )
                            if p.is_file()
                        )
                    ),
                    "--model",
                    str(args.package),
                    "--device",
                    args.device,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(args.package),
            )
            deadline = time.time() + 1800
            while time.time() < deadline:
                status, health = get(base, "/health")
                if (
                    status == 200
                    and isinstance(health, dict)
                    and health.get("status") == "ready"
                ):
                    break
                if server.poll() is not None:
                    raise RuntimeError(f"server exited with {server.returncode}")
                time.sleep(5)
            report["health"] = health
            status, models = get(base, "/v1/models")
            report["models"] = models
            checks["models_advertise_images"] = (
                status == 200
                and models["models"][0].get("modalities") == ["text", "image"]
                and "max_images" not in models["models"][0]
            )
            text_body = {
                "model": "x",
                "state": QUICKSTART_STATE,
                "questions": QUICKSTART_QUESTIONS,
            }
            s1, r1 = post(base, text_body)
            s2, r2 = post(base, {**text_body, "images": []})
            s3, r3 = post(base, {**text_body, "images": None})
            checks["server_text_same_with_images_absent_null_empty"] = (
                s1 == s2 == s3 == 200
                and r1["answers"] == r2["answers"] == r3["answers"]
            )
            report["server_text_max_dp_vs_in_process"] = (
                max_dp(text_only, r1) if s1 == 200 else None
            )
            s, r = post(base, {**text_body, "images": [data_url(args.example)]})
            checks["server_one_image"] = s == 200
            report["server_quickstart"] = r
            report["server_quickstart_max_dp_vs_in_process"] = (
                max_dp(quick, r) if s == 200 else None
            )
            report["server_multi_image_max_dp_vs_in_process"] = {}
            for n in (4, 5, 6):
                s, r = post(
                    base, {**text_body, "images": [data_url(p) for p in multi[n]]}
                )
                checks[f"server_{n}_images"] = s == 200 and all(
                    "error" not in a for a in r["answers"].values()
                )
                report["server_multi_image_max_dp_vs_in_process"][n] = (
                    max_dp(many[n], r) if s == 200 and n in many else None
                )

            noise = Image.frombytes("RGB", (2000, 1500), os.urandom(2000 * 1500 * 3))
            gif = io.BytesIO()
            Image.new("RGB", (64, 64), (1, 2, 3)).save(gif, format="GIF")
            gif_bytes = base64.b64encode(gif.getvalue()).decode()
            whole = io.BytesIO()
            Image.frombytes("RGB", (256, 256), os.urandom(256 * 256 * 3)).save(
                whole, format="PNG"
            )
            truncated = (
                "data:image/png;base64,"
                + base64.b64encode(
                    whole.getvalue()[: len(whole.getvalue()) // 2]
                ).decode()
            )
            refusals = {
                "gif": ["data:image/gif;base64," + gif_bytes],
                "gif_labelled_png": ["data:image/png;base64," + gif_bytes],
                "over_8mb": [png_url(noise)],
                "over_16mp": [png_url(Image.new("RGB", (4100, 4000), (200, 200, 200)))],
                "bad_base64": ["data:image/png;base64,@@@notbase64@@@"],
                "truncated_png": [truncated],
                "http_url": ["http://127.0.0.1:1/x.png"],
                "not_a_list": data_url(args.example),
            }
            report["refusals"] = {}
            for name, images in refusals.items():
                s, r = post(base, {**text_body, "images": images})
                report["refusals"][name] = {
                    "status": s,
                    "message": r if isinstance(r, str) else "200 OK",
                }
                checks[f"server_refuses_{name}"] = s == 422
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        log.close()
    report["pass"] = all(checks.values())
    write_json(args.out, report)
    print(json.dumps({"checks": checks, "pass": report["pass"]}, indent=1))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
