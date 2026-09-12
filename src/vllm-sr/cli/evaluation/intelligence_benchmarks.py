"""Versioned execution plans for the Intelligence 1.0 core benchmarks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from cli.evaluation.contract_validation import validate_secret_env

INTELLIGENCE_BENCHMARK_SCHEMA_VERSION = "vllm-sr.intelligence-benchmarks.v1"
INTELLIGENCE_RUN_SCHEMA_VERSION = "vllm-sr.intelligence-run.v1"
TERMINAL_BENCH_ATTEMPTS = 5

_RUNNER_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HF_HOME",
    "XDG_CACHE_HOME",
    "UV_CACHE_DIR",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)

BenchmarkRunner = Literal["aiperf", "inspect-hle", "inspect-scicode", "harbor"]
DataVerification = Literal["head-attested", "runner-pinned", "content-digest"]


@dataclass(frozen=True)
class SourcePin:
    cache_key: str
    url: str
    revision: str


@dataclass(frozen=True)
class DataPin:
    source: str
    revision: str
    verification: DataVerification


@dataclass(frozen=True)
class IntelligenceBenchmark:
    id: str
    name: str
    capability: str
    profile: str
    metric: str
    runner: BenchmarkRunner
    runner_benchmark: str
    source: SourcePin
    data: tuple[DataPin, ...]
    sample_count: int | None
    required_environment: tuple[str, ...] = ()
    access_requirements: tuple[str, ...] = ()


AIPERF = SourcePin(
    cache_key="aiperf",
    url="https://github.com/ai-dynamo/aiperf.git",
    revision="56945bb52114af818f3e681e21363aa68bc9b381",
)
INSPECT_EVALS = SourcePin(
    cache_key="inspect-evals",
    url="https://github.com/UKGovernmentBEIS/inspect_evals.git",
    revision="a3837b57f40805efb4064734fbad7f93f5291c38",
)
HARBOR = SourcePin(
    cache_key="harbor",
    url="https://github.com/laude-institute/harbor.git",
    revision="eeab9f0843e6af3fea2488b308b0098d8474ca98",
)

_BENCHMARKS = (
    IntelligenceBenchmark(
        id="tiger-ai-lab/mmlu-pro@1.0.0",
        name="MMLU-Pro",
        capability="general",
        profile="independent-standard",
        metric="accuracy",
        runner="aiperf",
        runner_benchmark="mmlu_pro",
        source=AIPERF,
        data=(
            DataPin(
                source="hf://datasets/TIGER-Lab/MMLU-Pro",
                revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
                verification="head-attested",
            ),
        ),
        sample_count=12032,
    ),
    IntelligenceBenchmark(
        id="idavidrein/gpqa-diamond@1.0.0",
        name="GPQA Diamond",
        capability="reasoning",
        profile="independent-standard",
        metric="accuracy",
        runner="aiperf",
        runner_benchmark="gpqa_diamond",
        source=AIPERF,
        data=(
            DataPin(
                source="hf://datasets/Idavidrein/gpqa",
                revision="633f5ee89ab8ad4522a9f850766b73f62147ffdd",
                verification="head-attested",
            ),
        ),
        sample_count=198,
        required_environment=("HF_TOKEN",),
        access_requirements=("Accept the GPQA dataset terms before execution.",),
    ),
    IntelligenceBenchmark(
        id="cais/humanitys-last-exam@1.0.0",
        name="Humanity's Last Exam 1.0 (text-only)",
        capability="reasoning",
        profile="independent-text-only",
        metric="accuracy",
        runner="inspect-hle",
        runner_benchmark="inspect_evals/hle",
        source=INSPECT_EVALS,
        data=(
            DataPin(
                source="hf://datasets/cais/hle:text-only-2158",
                revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
                verification="runner-pinned",
            ),
        ),
        sample_count=2158,
        required_environment=("HF_TOKEN", "OPENROUTER_API_KEY"),
        access_requirements=(
            "Accept the cais/hle dataset terms before execution.",
            "Set OPENROUTER_API_KEY for the two judges pinned by the runner config.",
        ),
    ),
    IntelligenceBenchmark(
        id="livecodebench/livecodebench@6.0.0",
        name="LiveCodeBench v6",
        capability="coding",
        profile="independent-code-generation",
        metric="pass_at_1",
        runner="aiperf",
        runner_benchmark="lcb_codegeneration",
        source=AIPERF,
        data=(
            DataPin(
                source="hf://datasets/livecodebench/code_generation_lite:v6",
                revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                verification="head-attested",
            ),
        ),
        sample_count=1055,
    ),
    IntelligenceBenchmark(
        id="scicode-bench/scicode@1.0.0",
        name="SciCode",
        capability="coding",
        profile="independent-test-288",
        metric="score",
        runner="inspect-scicode",
        runner_benchmark="inspect_evals/scicode",
        source=INSPECT_EVALS,
        data=(
            DataPin(
                source="https://github.com/scicode-bench/SciCode",
                revision=(
                    "69a8cfc829fe8788a426ce8b5de6292366dce7ef;"
                    "sha256:38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d"
                ),
                verification="runner-pinned",
            ),
            DataPin(
                source="gdrive://17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW/test_data.h5",
                revision=(
                    "sha256:48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
                ),
                verification="runner-pinned",
            ),
        ),
        sample_count=65,
        access_requirements=(
            "Provide a working Docker runtime for isolated code scoring.",
        ),
    ),
    IntelligenceBenchmark(
        id="harbor/terminal-bench@2.1.0",
        name="Terminal-Bench 2.1",
        capability="agentic",
        profile="independent-agent",
        metric="resolved",
        runner="harbor",
        runner_benchmark="terminal-bench/terminal-bench-2-1",
        source=HARBOR,
        data=(
            DataPin(
                source="harbor://terminal-bench/terminal-bench-2-1",
                revision=(
                    "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
                ),
                verification="content-digest",
            ),
        ),
        sample_count=None,
        access_requirements=(
            "Provide a Harbor-supported sandbox and a working container runtime.",
        ),
    ),
)

_BENCHMARK_BY_ID = {benchmark.id: benchmark for benchmark in _BENCHMARKS}


@dataclass(frozen=True)
class IntelligenceRunOptions:
    model: str
    base_url: str
    source_root: Path
    output_root: Path
    tokenizer: str = "builtin"
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 8
    reasoning_effort: str | None = None
    sample_limit: int | None = None
    terminal_attempts: int = TERMINAL_BENCH_ATTEMPTS


def intelligence_benchmark_catalog() -> dict[str, Any]:
    return {
        "schema_version": INTELLIGENCE_BENCHMARK_SCHEMA_VERSION,
        "index": "vllm-sr/intelligence@1.0.0",
        "missing_policy": "require_all",
        "benchmarks": [asdict(benchmark) for benchmark in _BENCHMARKS],
    }


def select_intelligence_benchmarks(
    ids: tuple[str, ...],
) -> tuple[IntelligenceBenchmark, ...]:
    if not ids or ids == ("all",):
        return _BENCHMARKS
    if "all" in ids:
        raise ValueError("all cannot be combined with an explicit benchmark ID")
    unknown = sorted(set(ids) - set(_BENCHMARK_BY_ID))
    if unknown:
        raise ValueError(f"unknown Intelligence 1.0 benchmark(s): {', '.join(unknown)}")
    return tuple(_BENCHMARK_BY_ID[benchmark_id] for benchmark_id in ids)


def build_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError("model must contain at least one filename-safe character")
    return slug[:120]


def normalize_openai_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("benchmark base URL must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("benchmark credentials must use --api-key-env")
    if parsed.query or parsed.fragment:
        raise ValueError("benchmark base URL must not contain query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/v1", "", ""))


def build_intelligence_run_plan(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
) -> dict[str, Any]:
    if not options.model.strip():
        raise ValueError("model must be non-empty")
    if options.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if options.sample_limit is not None and options.sample_limit < 1:
        raise ValueError("sample-limit must be at least 1")
    if options.terminal_attempts < 1:
        raise ValueError("terminal-attempts must be at least 1")
    validate_secret_env(options.api_key_env)

    source_path = (
        options.source_root.expanduser().resolve() / benchmark.source.cache_key
    )
    output_path = options.output_root.expanduser().resolve() / _benchmark_output_name(
        benchmark
    )
    openai_base = normalize_openai_base_url(options.base_url)
    command, cwd, required_env = _runner_command(
        benchmark,
        options,
        source_path,
        output_path,
        openai_base,
    )
    comparable = options.sample_limit is None
    if benchmark.runner == "harbor":
        comparable = comparable and options.terminal_attempts == TERMINAL_BENCH_ATTEMPTS
    return {
        "schema_version": INTELLIGENCE_RUN_SCHEMA_VERSION,
        "benchmark": asdict(benchmark),
        "subject": {
            "model": options.model,
            "base_url": openai_base,
            "reasoning_effort": options.reasoning_effort or "default",
        },
        "execution": {
            "cwd": str(cwd),
            "command": command,
            "required_environment": sorted(required_env),
            "output": str(output_path),
            "sample_limit": options.sample_limit,
            "terminal_attempts": (
                options.terminal_attempts if benchmark.runner == "harbor" else None
            ),
        },
        "index_eligible_protocol": comparable,
    }


def run_intelligence_benchmarks(
    benchmarks: tuple[IntelligenceBenchmark, ...],
    options: IntelligenceRunOptions,
) -> dict[str, Any]:
    output_root = options.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"output path already exists: {output_root}")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required to execute Intelligence 1.0 benchmarks")

    plans = [
        build_intelligence_run_plan(benchmark, options) for benchmark in benchmarks
    ]
    for plan in plans:
        _verify_source_checkout(
            plan["benchmark"]["source"], Path(plan["execution"]["cwd"])
        )
        _verify_data_pins(plan["benchmark"]["data"])
        missing = [
            name
            for name in plan["execution"]["required_environment"]
            if not os.getenv(name, "").strip()
        ]
        if missing:
            raise ValueError(
                f"{plan['benchmark']['id']} requires environment variable(s): "
                + ", ".join(missing)
            )

    output_root.mkdir(parents=True, mode=0o700)
    receipts: list[dict[str, Any]] = []
    for plan in plans:
        output_path = Path(plan["execution"]["output"])
        output_path.mkdir(mode=0o700)
        started_at = _now()
        command = list(plan["execution"]["command"])
        command[0] = uv
        environment = _runner_environment(
            options.api_key_env,
            plan["execution"]["required_environment"],
        )
        stdout_path = output_path / "runner.stdout.log"
        stderr_path = output_path / "runner.stderr.log"
        with _private_log(stdout_path) as stdout, _private_log(stderr_path) as stderr:
            result = subprocess.run(
                command,
                cwd=plan["execution"]["cwd"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        receipt = {
            **plan,
            "started_at": started_at,
            "finished_at": _now(),
            "exit_code": result.returncode,
            "completed": result.returncode == 0,
            "logs": {
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
        }
        _write_private_json(output_path / "receipt.json", receipt)
        receipts.append(receipt)
        if result.returncode != 0:
            break

    suite = {
        "schema_version": INTELLIGENCE_RUN_SCHEMA_VERSION,
        "index": "vllm-sr/intelligence@1.0.0",
        "model": options.model,
        "completed": len(receipts) == len(plans)
        and all(receipt["completed"] for receipt in receipts),
        "runs": receipts,
    }
    _write_private_json(output_root / "suite-receipt.json", suite)
    return suite


def _runner_command(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
    source_path: Path,
    output_path: Path,
    openai_base: str,
) -> tuple[list[str], Path, set[str]]:
    if benchmark.runner == "aiperf":
        return _aiperf_command(
            benchmark, options, source_path, output_path, openai_base
        )
    if benchmark.runner == "inspect-hle":
        return _hle_command(benchmark, options, source_path, output_path, openai_base)
    if benchmark.runner == "inspect-scicode":
        return _scicode_command(
            benchmark, options, source_path, output_path, openai_base
        )
    return _harbor_command(benchmark, options, source_path, output_path, openai_base)


def _aiperf_command(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
    source_path: Path,
    output_path: Path,
    openai_base: str,
) -> tuple[list[str], Path, set[str]]:
    count = options.sample_limit or benchmark.sample_count
    if count is None:
        raise ValueError(f"{benchmark.id} does not declare a request count")
    extra_inputs: dict[str, Any] = {"temperature": 0}
    if options.reasoning_effort:
        extra_inputs["reasoning_effort"] = options.reasoning_effort
    command = [
        "uv",
        "run",
        "--project",
        str(source_path),
        "aiperf",
        "profile",
        "--model",
        options.model,
        "--url",
        openai_base,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        options.tokenizer,
        "--accuracy-benchmark",
        benchmark.runner_benchmark,
        "--num-requests",
        str(count),
        "--concurrency",
        str(options.concurrency),
        "--extra-inputs",
        json.dumps(extra_inputs, separators=(",", ":")),
        "--output-artifact-dir",
        str(output_path),
        "--profile-export-level",
        "records",
    ]
    required_env = set(benchmark.required_environment)
    if os.getenv(options.api_key_env, "").strip():
        required_env.add(options.api_key_env)
    return command, source_path, required_env


def _hle_command(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
    source_path: Path,
    output_path: Path,
    openai_base: str,
) -> tuple[list[str], Path, set[str]]:
    command = [
        "uv",
        "run",
        "--project",
        str(source_path),
        "inspect",
        "eval",
        "--run-config",
        str(source_path / "src/inspect_evals/hle/run_configs/default.yaml"),
        "--model",
        _inspect_model(options.model),
        "--model-base-url",
        openai_base,
        "--temperature",
        "0",
        "--max-connections",
        str(options.concurrency),
        "--log-dir",
        str(output_path / "inspect-logs"),
        "-T",
        "include_multi_modal=false",
    ]
    if options.sample_limit is not None:
        command.extend(("--limit", str(options.sample_limit)))
    if options.reasoning_effort:
        command.extend(("--reasoning-effort", options.reasoning_effort))
    return command, source_path, set(benchmark.required_environment)


def _scicode_command(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
    source_path: Path,
    output_path: Path,
    openai_base: str,
) -> tuple[list[str], Path, set[str]]:
    command = [
        "uv",
        "run",
        "--project",
        str(source_path),
        "inspect",
        "eval",
        benchmark.runner_benchmark,
        "--model",
        _inspect_model(options.model),
        "--model-base-url",
        openai_base,
        "--temperature",
        "0",
        "--max-connections",
        str(options.concurrency),
        "--log-dir",
        str(output_path / "inspect-logs"),
        "-T",
        "provide_scientific_background=false",
        "-T",
        "include_dev_set=false",
    ]
    if options.sample_limit is not None:
        command.extend(("--limit", str(options.sample_limit)))
    if options.reasoning_effort:
        command.extend(("--reasoning-effort", options.reasoning_effort))
    return command, source_path, set(benchmark.required_environment)


def _harbor_command(
    benchmark: IntelligenceBenchmark,
    options: IntelligenceRunOptions,
    source_path: Path,
    output_path: Path,
    openai_base: str,
) -> tuple[list[str], Path, set[str]]:
    dataset = benchmark.data[0]
    command = [
        "uv",
        "run",
        "--project",
        str(source_path),
        "harbor",
        "run",
        "--dataset",
        f"{dataset.source.removeprefix('harbor://')}@{dataset.revision}",
        "--agent",
        "terminus-2",
        "--model",
        _inspect_model(options.model),
        "--agent-kwarg",
        f"api_base={openai_base}",
        "--agent-kwarg",
        "temperature=1",
        "--n-attempts",
        str(options.terminal_attempts),
        "--n-concurrent",
        str(options.concurrency),
        "--jobs-dir",
        str(output_path / "harbor-jobs"),
        "--quiet",
    ]
    if options.reasoning_effort:
        command.extend(
            ("--agent-kwarg", f"reasoning_effort={options.reasoning_effort}")
        )
    if options.sample_limit is not None:
        command.extend(("--n-tasks", str(options.sample_limit)))
    return command, source_path, set(benchmark.required_environment)


def _inspect_model(model: str) -> str:
    return model if model.startswith("openai/") else f"openai/{model}"


def _benchmark_output_name(benchmark: IntelligenceBenchmark) -> str:
    return benchmark.id.replace("/", "--").replace("@", "--")


def _verify_source_checkout(source: dict[str, str], path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"benchmark source checkout is missing or invalid: {path}")
    try:
        head = subprocess.run(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"could not verify benchmark source checkout: {path}") from exc
    if head != source["revision"] or status:
        raise ValueError(
            f"benchmark source must be clean and pinned to {source['revision']}: {path}"
        )


def _verify_data_pins(data_pins: list[dict[str, str]]) -> None:
    for pin in data_pins:
        if pin["verification"] != "head-attested":
            continue
        source = pin["source"]
        prefix = "hf://datasets/"
        if not source.startswith(prefix):
            raise ValueError(f"unsupported head-attested dataset source: {source}")
        repository = source[len(prefix) :].split(":", maxsplit=1)[0]
        actual = _hugging_face_dataset_head(repository)
        if actual != pin["revision"]:
            raise ValueError(
                f"dataset {repository} moved from frozen revision "
                f"{pin['revision']} to {actual}; update the benchmark protocol "
                "deliberately before running it"
            )


def _hugging_face_dataset_head(repository: str) -> str:
    request = Request(
        f"https://huggingface.co/api/datasets/{repository}/revision/main",
        headers={"Accept": "application/json", **_hugging_face_auth_header()},
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise ValueError(
            f"could not attest frozen Hugging Face dataset revision for {repository}"
        ) from exc
    revision = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Hugging Face returned no dataset revision for {repository}")
    return revision


def _hugging_face_auth_header() -> dict[str, str]:
    token = os.getenv("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _runner_environment(
    api_key_env: str,
    required_environment: list[str],
) -> dict[str, str]:
    environment = {
        name: value
        for name in _RUNNER_ENVIRONMENT_ALLOWLIST
        if (value := os.getenv(name, ""))
    }
    for name in required_environment:
        if name == api_key_env:
            continue
        if value := os.getenv(name, ""):
            environment[name] = value
    environment["AIPERF_ACCURACY_LCB_RELEASE_TAG"] = "v6"
    if token := os.getenv(api_key_env, ""):
        environment["OPENAI_API_KEY"] = token
    return environment


def _private_log(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _write_private_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
