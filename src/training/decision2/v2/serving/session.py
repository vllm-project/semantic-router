"""Serving-track sessions, run inside the scored ROCm image on one GPU (node side).

    python3 -m v2.serving.session plugin  --out RUN --plugin-src DIR --package PKG --dtype bfloat16
                                          [--panel NAME:PROMPTS:PREDICTIONS:COUNT]... [--bench PROMPTS:N]
    python3 -m v2.serving.session runtime --out RUN --package PKG --bench PROMPTS:N [--panel ...]...
    python3 -m v2.serving.session vela    --out RUN --plugin-src DIR --model SNAPSHOT --texts JSON...
    python3 -m v2.serving.session coloc   --out RUN --plugin-src DIR --package PKG --model SNAPSHOT
                                          --bench PROMPTS:N --texts JSON...

``plugin`` installs vllm-sr-plugins from its mirrored source as a wheel into
RUN/site, serves the package with ``vllm serve`` on 127.0.0.1 inside the
container, answers every panel prompt through ``/v1/system_one`` (one item per
request, as the release parity does) and compares with the stored scored
predictions, then measures latency and throughput. ``runtime`` measures the
shipped package runtime in-process, then again with the backbone's Linear
weights kept BF16-resident. Every result is a JSON file in RUN; per-prompt
answers stay in RUN on the node.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .measure import PanelTally, latency_summary, payload_sha256

PLUGINS = "vllm_sr_decision2,vllm_sr_system_one"
DECISION_PORT = 8100
VELA_PORT = 8200


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in (
        "vllm",
        "torch",
        "transformers",
        "tokenizers",
        "triton",
        "safetensors",
    ):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------- plugin install


def install_plugin(source: Path, out: Path) -> dict[str, Any]:
    """Build a wheel from a copy of the (read-only) mirrored source and install it into out/site."""
    site, wheels = out / "site", out / "wheel"
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        copy = Path(scratch) / "src"
        shutil.copytree(source, copy)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "-q",
                "-w",
                str(wheels),
                str(copy),
            ],
            check=True,
        )
    (wheel,) = wheels.glob("vllm_sr_plugins-*.whl")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "-q",
            "--target",
            str(site),
            str(wheel),
        ],
        check=True,
    )
    installed = {
        str(path.relative_to(site)): sha_file(path)
        for path in sorted((site / "vllm_sr_plugins").rglob("*.py"))
    }
    mirrored = {
        str(path.relative_to(source)): sha_file(path)
        for path in sorted((source / "vllm_sr_plugins").rglob("*.py"))
    }
    if installed != mirrored:
        raise RuntimeError("installed plugin files differ from the mirrored source")
    return {
        "wheel": wheel.name,
        "wheel_sha256": sha_file(wheel),
        "files": len(installed),
    }


# ---------------------------------------------------------------- servers


class Server:
    """One ``vllm serve`` subprocess on 127.0.0.1, logging to out/<name>.log."""

    def __init__(
        self, name: str, args: list[str], port: int, out: Path, env: dict[str, str]
    ):
        self.name, self.port = name, port
        self.log_path = out / f"{name}.log"
        self.command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            *args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        self._log = self.log_path.open("w")
        self.started = time.time()
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.ready_seconds: float | None = None

    def wait_ready(self, timeout: float = 1800) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                status, _ = request(self.port, "GET", "/health", None, timeout=5)
                if status == 200:
                    self.ready_seconds = time.time() - self.started
                    return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.name} not ready after {timeout} s")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._log.close()

    def log_facts(self) -> dict[str, Any]:
        keep = (
            "loaded",
            "Model loading took",
            "Loading weights took",
            "KV cache",
            "GPU KV cache",
            "Maximum concurrency",
            "graph",
            "compile",
            "Available KV cache memory",
            "System One ready",
            "Decision 2.0 scoring model",
            "attention backend",
            "Using ",
            "tested against",
            "WARNING",
            "Error",
            "Traceback",
        )
        lines = self.log_path.read_text(errors="replace").splitlines()
        return {
            "command": self.command[3:],
            "ready_seconds": self.ready_seconds,
            "log_lines": [
                line[-400:] for line in lines if any(k in line for k in keep)
            ][:120],
        }


def request(
    port: int,
    method: str,
    path: str,
    body: Any,
    timeout: float = 600,
    connection: http.client.HTTPConnection | None = None,
) -> tuple[int, Any]:
    own = connection is None
    connection = connection or http.client.HTTPConnection(
        "127.0.0.1", port, timeout=timeout
    )
    try:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        connection.request(
            method, path, body=data, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        raw = response.read()
        return response.status, (json.loads(raw) if raw else None)
    finally:
        if own:
            connection.close()


def serving_env(site: Path, out: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(site),
            "VLLM_PLUGINS": PLUGINS,
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_CACHE_ROOT": str(out / "vllm-cache"),
            "XDG_CACHE_HOME": str(out / "xdg-cache"),
            "TRITON_CACHE_DIR": str(out / "triton-cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def decision_server_args(
    package: Path, dtype: str, batched_tokens: int, memory: float, extra: list[str]
) -> list[str]:
    return [
        str(package),
        "--runner",
        "pooling",
        "--hf-config-path",
        str(package / "backbone"),
        "--tokenizer",
        str(package),
        "--hf-overrides",
        json.dumps({"architectures": ["Decision2Qwen3_5ForScoring"]}),
        "--dtype",
        dtype,
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        str(batched_tokens),
        "--gpu-memory-utilization",
        str(memory),
        "--served-model-name",
        package.name,
        *extra,
    ]


# ---------------------------------------------------------------- clients


class Pool:
    """Keep-alive HTTP clients, one per worker thread."""

    def __init__(self, port: int, workers: int):
        self.port = port
        self.executor = concurrent.futures.ThreadPoolExecutor(workers)
        self.local = threading.local()

    def post(self, path: str, body: Any) -> tuple[float, int, Any]:
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = self.local.connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=600
            )
        started = time.perf_counter()
        try:
            status, value = request(
                self.port, "POST", path, body, connection=connection
            )
        except (OSError, http.client.HTTPException):
            self.local.connection = None
            raise
        return time.perf_counter() - started, status, value

    def map(self, path: str, bodies: list[Any]) -> list[tuple[float, int, Any]]:
        return list(self.executor.map(lambda body: self.post(path, body), bodies))

    def close(self) -> None:
        self.executor.shutdown()


def system_one_body(prompt: dict[str, Any]) -> dict[str, Any]:
    return {"state": prompt["state"], "questions": prompt["questions"]}


def run_panels(
    port: int, panels: list[str], out: Path, concurrency: int
) -> dict[str, Any]:
    results = {}
    pool = Pool(port, concurrency)
    try:
        for spec in panels:
            name, prompts_path, predictions_path, count = spec.split(":")
            prompts = read_jsonl(Path(prompts_path))[: int(count)]
            stored = {row["id"]: row for row in read_jsonl(Path(predictions_path))}
            started = time.perf_counter()
            outcomes = pool.map("/v1/system_one", [system_one_body(p) for p in prompts])
            seconds = time.perf_counter() - started
            tally = PanelTally(name)
            with (out / f"{name}.predictions.jsonl").open("w") as sink:
                for prompt, (latency, status, value) in zip(prompts, outcomes):
                    response = value if status == 200 else None
                    tally.add(prompt, response, stored.get(prompt["id"]))
                    sink.write(
                        json.dumps(
                            {
                                "id": prompt["id"],
                                "status": status,
                                "answers": (response or {}).get("answers"),
                                "usage": (response or {}).get("usage"),
                                "latency_ms": latency * 1000,
                                "source_input_sha256": payload_sha256(prompt),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            results[name] = {
                **tally.summary(),
                "seconds": seconds,
                "concurrency": concurrency,
                "prompts_sha256": sha_file(Path(prompts_path)),
                "predictions_sha256": sha_file(Path(predictions_path)),
            }
            print(
                json.dumps(
                    {
                        "panel": name,
                        **{
                            k: results[name][k]
                            for k in (
                                "prompts",
                                "slots",
                                "category_changes",
                                "missing",
                                "max_abs_drift",
                                "errors",
                            )
                        },
                    }
                ),
                flush=True,
            )
    finally:
        pool.close()
    return results


def bench_http(
    port: int,
    path: str,
    bodies: list[Any],
    levels: list[int],
    warmup: int,
    count_items: Any,
) -> dict[str, Any]:
    """Single-request latency (sequential), then throughput at each concurrency level."""
    pool = Pool(port, max(levels))
    try:
        pool.map(path, bodies[:warmup])
        sequential = [pool.post(path, body) for body in bodies]
        if any(status != 200 for _, status, _ in sequential):
            raise RuntimeError("benchmark request failed")
        result: dict[str, Any] = {
            "single": latency_summary([s for s, _, _ in sequential]),
            "throughput": {},
        }
        for level in levels:
            level_pool = Pool(port, level)
            try:
                level_pool.map(path, bodies[: min(warmup, len(bodies))])
                started = time.perf_counter()
                outcomes = level_pool.map(path, bodies)
                wall = time.perf_counter() - started
            finally:
                level_pool.close()
            items, questions, tokens = count_items([value for _, _, value in outcomes])
            result["throughput"][str(level)] = {
                "wall_s": wall,
                "items_per_s": items / wall,
                "questions_per_s": questions / wall,
                "input_tokens_per_s": tokens / wall,
                "latency": latency_summary([s for s, _, _ in outcomes]),
                "failed": sum(status != 200 for _, status, _ in outcomes),
            }
        return result
    finally:
        pool.close()


def count_system_one(values: list[Any]) -> tuple[int, int, int]:
    items = questions = tokens = 0
    for value in values:
        if isinstance(value, dict) and "answers" in value:
            items += 1
            questions += len(value["answers"])
            tokens += value["usage"]["input_tokens"]
    return items, questions, tokens


def parse_bench(spec: str) -> tuple[list[dict[str, Any]], str]:
    path, count = spec.rsplit(":", 1)
    return read_jsonl(Path(path))[: int(count)], sha_file(Path(path))


# ---------------------------------------------------------------- sessions


def session_plugin(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    install = install_plugin(args.plugin_src, out)
    server = Server(
        "decision",
        decision_server_args(
            args.package,
            args.dtype,
            args.max_num_batched_tokens,
            args.gpu_memory_utilization,
            args.server_arg,
        ),
        DECISION_PORT,
        out,
        serving_env(out / "site", out),
    )
    result: dict[str, Any] = {
        "session": "plugin",
        "install": install,
        "dtype": args.dtype,
    }
    try:
        server.wait_ready()
        status, health = request(
            DECISION_PORT,
            "POST",
            "/v1/system_one",
            {
                "state": "warmup",
                "questions": {
                    "q": {"type": "noul", "instructions": "Is this a warmup?"}
                },
            },
        )
        if status != 200:
            raise RuntimeError(f"warmup failed: {status} {health}")
        if args.panel:
            result["parity"] = run_panels(
                DECISION_PORT, args.panel, out, args.concurrency
            )
        if args.bench:
            prompts, digest = parse_bench(args.bench)
            result["bench"] = {
                "prompts_sha256": digest,
                "items": len(prompts),
                **bench_http(
                    DECISION_PORT,
                    "/v1/system_one",
                    [system_one_body(p) for p in prompts],
                    args.levels,
                    args.warmup,
                    count_system_one,
                ),
            }
    finally:
        server.stop()
        result["server"] = server.log_facts()
    return result


def session_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """The shipped package runtime, FP32 master weights under BF16 autocast, then BF16-resident Linear."""
    import torch

    from v2.release import examples

    sys.path.insert(0, "/opt/decision-fla")
    sys.path.insert(0, str(args.package))
    from decision2 import Decision2

    started = time.perf_counter()
    model = Decision2.from_pretrained(args.package, device="cuda:0")
    load_seconds = time.perf_counter() - started
    kernels = examples.kernel_runtime(True)
    prompts, digest = parse_bench(args.bench)

    def run(items: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
        latencies, answers = [], []
        for prompt in items:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            response = model.system_one(
                state=prompt["state"], questions=prompt["questions"]
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - begin)
            answers.append(response["answers"])
        return latencies, answers

    result: dict[str, Any] = {
        "session": "runtime",
        "load_seconds": load_seconds,
        "kernels": kernels,
        "bench_prompts_sha256": digest,
        "items": len(prompts),
        "variants": {},
    }
    for variant in ("fp32-master", "bf16-resident"):
        if variant == "bf16-resident":
            converted = 0
            for module in model.backend.model.backbone.modules():
                if isinstance(module, torch.nn.Linear):
                    exact = module.weight.data.to(torch.bfloat16)
                    if not torch.equal(exact.float(), module.weight.data):
                        raise RuntimeError("a backbone Linear weight is not BF16-exact")
                    module.weight.data = exact
                    converted += 1
            torch.cuda.empty_cache()
            result["variants"][variant] = {"converted_linear": converted}
        else:
            result["variants"][variant] = {}
        run(prompts[: args.warmup])
        torch.cuda.reset_peak_memory_stats()
        latencies, answers = run(prompts)
        entry = result["variants"][variant]
        entry.update(
            {
                "single": latency_summary(latencies),
                "items_per_s": len(prompts) / sum(latencies),
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "parameter_bytes_gib": sum(
                    p.numel() * p.element_size()
                    for p in model.backend.model.parameters()
                )
                / 2**30,
            }
        )
        with (args.out / f"runtime-{variant}.answers.jsonl").open("w") as sink:
            for prompt, answer in zip(prompts, answers):
                sink.write(json.dumps({"id": prompt["id"], "answers": answer}) + "\n")
        entry["answers"] = answers
        if args.panel and variant == "bf16-resident":
            entry["parity"] = {}
            for spec in args.panel:
                name, prompts_path, predictions_path, count = spec.split(":")
                stored = {row["id"]: row for row in read_jsonl(Path(predictions_path))}
                tally = PanelTally(name)
                for prompt in read_jsonl(Path(prompts_path))[: int(count)]:
                    response = model.system_one(
                        state=prompt["state"], questions=prompt["questions"]
                    )
                    tally.add(prompt, response, stored.get(prompt["id"]))
                entry["parity"][name] = tally.summary()
    from v2.release.examples import compare_answers

    base, resident = (
        result["variants"][v].pop("answers") for v in ("fp32-master", "bf16-resident")
    )
    diff = {"slots": 0, "category_changes": 0, "missing": 0, "max_abs_drift": 0.0}
    identical = 0
    for left, right in zip(base, resident):
        one = compare_answers(left, right)
        identical += left == right
        for key in ("slots", "category_changes", "missing"):
            diff[key] += one[key]
        diff["max_abs_drift"] = max(diff["max_abs_drift"], one["max_abs_drift"])
    result["bf16_resident_vs_fp32_master"] = {**diff, "bit_identical_items": identical}
    return result


def load_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            texts.append(row.get("question") or row.get("text"))
    return [t for t in texts if isinstance(t, str) and t]


def long_texts(texts: list[str], tokenizer: Any, targets: list[int]) -> list[str]:
    """Deterministic long inputs: the sample texts concatenated up to each token target."""
    out = []
    for target in targets:
        parts, count, i = [], 0, 0
        while count < target - 64:
            parts.append(texts[i % len(texts)])
            count += (
                len(
                    tokenizer(texts[i % len(texts)], add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                + 1
            )
            i += 1
        out.append("\n".join(parts))
    return out


def vela_reference(
    model_path: Path, texts: list[str], out: Path
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.float32, local_files_only=True
        )
        .to("cuda:0")
        .eval()
    )
    rows = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, truncation=False)["input_ids"]
            try:
                logits = (
                    model(input_ids=torch.tensor([ids], device="cuda:0"))
                    .logits[0]
                    .float()
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            rows.append(
                {
                    "ids": ids,
                    "probs": torch.softmax(logits, -1).tolist(),
                    "attn": model.config._attn_implementation,
                }
            )
    del model
    torch.cuda.empty_cache()
    with (out / "vela-reference.jsonl").open("w") as sink:
        for row in rows:
            sink.write(
                json.dumps({"n_tokens": len(row["ids"]), "probs": row["probs"]}) + "\n"
            )
    return rows


def compare_vela(
    reference: list[dict[str, Any]], served: list[list[float]]
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for ref, probs in zip(reference, served):
        n = len(ref["ids"])
        bucket = "<=512" if n <= 512 else "<=8192" if n <= 8192 else ">8192"
        entry = buckets.setdefault(
            bucket, {"n": 0, "argmax_changes": 0, "max_prob_drift": 0.0}
        )
        entry["n"] += 1
        a, b = ref["probs"], probs
        entry["argmax_changes"] += a.index(max(a)) != b.index(max(b))
        entry["max_prob_drift"] = max(
            entry["max_prob_drift"], max(abs(x - y) for x, y in zip(a, b))
        )
    return buckets


def classify(port: int, name: str, token_ids: list[list[int]]) -> list[list[float]]:
    pool = Pool(port, 8)
    try:
        outcomes = pool.map(
            "/classify", [{"model": name, "input": ids} for ids in token_ids]
        )
    finally:
        pool.close()
    probs = []
    for _, status, value in outcomes:
        if status != 200:
            raise RuntimeError(f"classify failed: {status} {value}")
        probs.append(value["data"][0]["probs"])
    return probs


def vela_server(
    args: argparse.Namespace, dtype: str, out: Path, memory: float
) -> Server:
    return Server(
        f"vela-{dtype}",
        [
            str(args.model),
            "--runner",
            "pooling",
            "--dtype",
            dtype,
            "--max-model-len",
            "32768",
            "--max-num-batched-tokens",
            "32768",
            "--gpu-memory-utilization",
            str(memory),
            "--served-model-name",
            "vela",
        ],
        VELA_PORT,
        out,
        serving_env(out / "site", out),
    )


def session_vela(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    out = args.out
    result: dict[str, Any] = {
        "session": "vela",
        "install": install_plugin(args.plugin_src, out),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    short = load_texts(args.texts)
    texts = short + long_texts(short, tokenizer, [2048, 8192, 16384, 32000])
    reference = vela_reference(args.model, texts, out)
    result["texts"] = {
        "short": len(short),
        "long_tokens": [len(r["ids"]) for r in reference[len(short) :]],
        "long_skipped_oom": len(texts) - len(reference),
        "attn_implementation": reference[0]["attn"],
    }
    for dtype in args.dtypes:
        server = vela_server(args, dtype, out, 0.2)
        try:
            server.wait_ready()
            token_probs = classify(VELA_PORT, "vela", [r["ids"] for r in reference])
            matches = 0
            for text, ref in zip(short, reference):
                status, tokenized = request(
                    VELA_PORT, "POST", "/tokenize", {"model": "vela", "prompt": text}
                )
                matches += status == 200 and tokenized.get("tokens") == ref["ids"]
            text_probs = classify(VELA_PORT, "vela", short)
            result[dtype] = {
                "token_ids": compare_vela(reference, token_probs),
                "text_input_short": compare_vela(reference[: len(short)], text_probs),
                "tokenize_matches": f"{matches}/{len(short)}",
            }
            bodies = [
                {"model": "vela", "input": r["ids"]} for r in reference[: len(short)]
            ]
            result[dtype]["bench"] = bench_http(
                VELA_PORT,
                "/classify",
                bodies,
                args.levels,
                args.warmup,
                lambda values: (len(values), len(values), 0),
            )
        finally:
            server.stop()
            result.setdefault("servers", {})[dtype] = server.log_facts()
    return result


def session_coloc(args: argparse.Namespace) -> dict[str, Any]:
    """A Decision engine and a Vela engine sharing one GPU: latency alone, then under joint load."""
    out = args.out
    result: dict[str, Any] = {
        "session": "coloc",
        "install": install_plugin(args.plugin_src, out),
    }
    env = serving_env(out / "site", out)
    decision = Server(
        "decision",
        decision_server_args(
            args.package,
            args.dtype,
            16384,
            args.gpu_memory_utilization,
            args.server_arg,
        ),
        DECISION_PORT,
        out,
        env,
    )
    vela = None
    try:
        decision.wait_ready()
        vela = vela_server(args, "bfloat16", out, 0.1)
        vela.wait_ready()
        prompts, _ = parse_bench(args.bench)
        decision_bodies = [system_one_body(p) for p in prompts]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        vela_bodies = [
            {"model": "vela", "input": tokenizer(t)["input_ids"]}
            for t in load_texts(args.texts)
        ]

        def drive(
            port: int, path: str, bodies: list[Any], workers: int
        ) -> dict[str, Any]:
            pool = Pool(port, workers)
            try:
                pool.map(path, bodies[: args.warmup])
                started = time.perf_counter()
                outcomes = pool.map(path, bodies)
                wall = time.perf_counter() - started
            finally:
                pool.close()
            return {
                "wall_s": wall,
                "requests_per_s": len(bodies) / wall,
                "latency": latency_summary([s for s, _, _ in outcomes]),
                "failed": sum(status != 200 for _, status, _ in outcomes),
            }

        for workers in args.levels:
            alone_d = drive(DECISION_PORT, "/v1/system_one", decision_bodies, workers)
            alone_v = drive(VELA_PORT, "/classify", vela_bodies, workers)
            with concurrent.futures.ThreadPoolExecutor(2) as both:
                joint_d = both.submit(
                    drive, DECISION_PORT, "/v1/system_one", decision_bodies, workers
                )
                joint_v = both.submit(
                    drive, VELA_PORT, "/classify", vela_bodies * 4, workers
                )
                joint = {"decision": joint_d.result(), "vela": joint_v.result()}
            result[f"workers={workers}"] = {
                "decision_alone": alone_d,
                "vela_alone": alone_v,
                "joint": joint,
            }
    finally:
        for server in (vela, decision):
            if server is not None:
                server.stop()
        result["servers"] = {
            s.name: s.log_facts() for s in (decision, vela) if s is not None
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="session", required=True)
    for name in ("plugin", "runtime", "vela", "coloc"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--warmup", type=int, default=16)
        p.add_argument(
            "--levels",
            type=lambda s: [int(x) for x in s.split(",")],
            default=[1, 8, 32, 128],
        )
        if name != "runtime":
            p.add_argument("--plugin-src", type=Path, required=True)
        if name in ("plugin", "runtime", "coloc"):
            p.add_argument("--package", type=Path, required=True)
            p.add_argument("--bench")
            p.add_argument("--panel", action="append", default=[])
        if name in ("plugin", "coloc"):
            p.add_argument("--dtype", default="bfloat16")
            p.add_argument("--gpu-memory-utilization", type=float, default=0.2)
            p.add_argument("--server-arg", action="append", default=[])
        if name == "plugin":
            p.add_argument("--max-num-batched-tokens", type=int, default=16384)
            p.add_argument("--concurrency", type=int, default=1)
        if name in ("vela", "coloc"):
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--texts", type=Path, action="append", required=True)
        if name == "vela":
            p.add_argument(
                "--dtypes", type=lambda s: s.split(","), default=["float32", "bfloat16"]
            )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    handler = {
        "plugin": session_plugin,
        "runtime": session_runtime,
        "vela": session_vela,
        "coloc": session_coloc,
    }[args.session]
    result = handler(args)
    result.update(
        {
            "versions": versions(),
            "wall_seconds": time.time() - started,
            "argv": sys.argv[1:],
            "mirror": os.environ.get("DEV2_MIRROR"),
        }
    )
    write_json(args.out / f"{args.session}.result.json", result)
    print(
        json.dumps(
            {
                "session": args.session,
                "wall_seconds": result["wall_seconds"],
                "result": str(args.out / f"{args.session}.result.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
