#!/usr/bin/env python3
"""Run VSR EvalScope recipes on a remote AMD router host.

The eval path exposes one public model API, ``vllm-sr/auto``. Each benchmark
owns a tuned auto recipe. This helper syncs recipes, switches the router to the
benchmark-specific recipe, runs EvalScope for selected model keys, collects the
joined table, and optionally pulls artifacts back to the local workspace.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO_ROOT / "bench/router_flow"
REAL_EVAL_ROOT = BENCH_ROOT / "real_eval"
CONFIG_ROOT = BENCH_ROOT / "configs"
CLI_ROOT = REPO_ROOT / "src/vllm-sr"
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.config_generator import generate_envoy_config_from_user_config  # noqa: E402
from cli.parser import parse_user_config  # noqa: E402

DEFAULT_REMOTE_EVAL_ROOT = "/root/router-flow-eval"
DEFAULT_REMOTE_CONFIG_DIR = "/root/vllm-sr-flow-eval/bench/router_flow/configs"
DEFAULT_REMOTE_ROUTER_CONFIG = "amd_auto_omni.yaml"
DEFAULT_REMOTE_SOURCE_CONFIG_DIR = "sources"
DEFAULT_REMOTE_OPENROUTER_API_KEY_FILE = (
    f"{DEFAULT_REMOTE_EVAL_ROOT}/.openrouter_api_key"
)
DEFAULT_EVALSCOPE_PYTHON = "/root/evalscope-venv/bin/python"
DEFAULT_EVALSCOPE_BIN = "/root/evalscope-venv/bin/evalscope"
DEFAULT_REMOTE_ENVOY_CONTAINER = "vllm-sr-envoy-container"
DEFAULT_ENVOY_EXTPROC_ADDRESS = "vllm-sr-router-container"
DEFAULT_ENVOY_ROUTER_API_ADDRESS = "vllm-sr-router-container"
DEFAULT_OUTPUT_ROOT = "results/evalscope-smoke-v2"
DEFAULT_REPORT_DIR = "results/evalscope-report-v2"
RETRYABLE_RETURN_CODES = {12, 255}

DEFAULT_SUITE = "evalscope_suite.yaml"
GLM52_SUITE = "evalscope_suite_livecode_glm52.yaml"
KIMI_K27_CODE_SUITE = "evalscope_suite_livecode_kimi_k27_code.yaml"
GLM52_HLE_SWE_SUITE = "evalscope_suite_hle_swe_glm52.yaml"
HLE_HYBRID_SUITE = "evalscope_suite_hle_hybrid.yaml"
RECIPE_SETS = {
    "closed": {
        "suite": DEFAULT_SUITE,
        "default_models": ["auto"],
        "configs": {
            "gpqa_d": "amd_auto_gpqa_omni.yaml",
            "live_code_bench": "amd_auto_livecode_omni.yaml",
            "scicode": "amd_auto_scicode_omni.yaml",
        },
    },
    "glm52": {
        "suite": GLM52_SUITE,
        "default_models": ["auto", "glm52_native"],
        "configs": {
            "live_code_bench": "amd_auto_livecode_glm52_omni.yaml",
        },
    },
    "kimi_k27_code": {
        "suite": KIMI_K27_CODE_SUITE,
        "default_models": ["auto", "kimi_k27_code_native"],
        "configs": {
            "live_code_bench": "amd_auto_livecode_kimi_k27_code_omni.yaml",
        },
    },
    "glm52_hle_swe": {
        "suite": GLM52_HLE_SWE_SUITE,
        "default_models": ["auto"],
        "configs": {
            "hle_text": "amd_auto_hle_glm52.yaml",
            "swe_bench_verified_mini_agentic": "amd_auto_swe_glm52.yaml",
            "swe_bench_pro": "amd_auto_swe_glm52.yaml",
        },
    },
    "hle_hybrid": {
        "suite": HLE_HYBRID_SUITE,
        "default_models": ["auto"],
        "configs": {
            "hle_text": "amd_auto_hle_hybrid.yaml",
        },
    },
}
BENCHMARK_CONFIGS = RECIPE_SETS["closed"]["configs"]
DIRECT_OPENROUTER_MODEL_KEYS = {"glm52_native", "kimi_k27_code_native"}
ROUTER_MODEL_KEYS = {"auto"}
MODEL_API_URLS = {
    "auto": "http://127.0.0.1:8899/v1",
    "glm52_native": "https://openrouter.ai/api/v1",
    "kimi_k27_code_native": "https://openrouter.ai/api/v1",
}

OMNI_MODEL_KEY = "auto"
OMNI_MODEL_NAME = "vllm-sr/auto"
SSH_OPTIONS = [
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("VLLM_SR_AMD_HOST", ""),
        help="SSH target for the AMD eval host. Defaults to VLLM_SR_AMD_HOST.",
    )
    parser.add_argument(
        "--recipe-set",
        choices=sorted(RECIPE_SETS),
        default="closed",
        help="Recipe/config set to run. Use hle_hybrid for competitive HLE hybrid runs.",
    )
    parser.add_argument(
        "--suite",
        default="",
        help="EvalScope suite filename under real_eval/. Defaults to the recipe set suite.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key from the selected suite. Defaults to the recipe set models.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(
            {
                benchmark
                for recipe in RECIPE_SETS.values()
                for benchmark in recipe["configs"]
            }
        ),
        default=[],
        help="Benchmark id from the selected suite. Repeatable. Defaults to all configured recipes for the recipe set.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--remote-eval-root", default=DEFAULT_REMOTE_EVAL_ROOT)
    parser.add_argument("--remote-config-dir", default=DEFAULT_REMOTE_CONFIG_DIR)
    parser.add_argument("--remote-router-config", default=DEFAULT_REMOTE_ROUTER_CONFIG)
    parser.add_argument(
        "--remote-source-config-dir", default=DEFAULT_REMOTE_SOURCE_CONFIG_DIR
    )
    parser.add_argument(
        "--remote-openrouter-api-key-file",
        default=DEFAULT_REMOTE_OPENROUTER_API_KEY_FILE,
        help="Remote file containing OPENROUTER_API_KEY, used when no existing router container can be inspected.",
    )
    parser.add_argument(
        "--remote-envoy-container",
        default=DEFAULT_REMOTE_ENVOY_CONTAINER,
        help="Remote Envoy container name used to discover and restart /etc/envoy/envoy.yaml.",
    )
    parser.add_argument(
        "--remote-envoy-config",
        default="",
        help="Remote Envoy config path. Defaults to discovering the /etc/envoy/envoy.yaml bind mount.",
    )
    parser.add_argument(
        "--envoy-extproc-address",
        default=DEFAULT_ENVOY_EXTPROC_ADDRESS,
        help="Host written into generated Envoy config for the router extproc cluster.",
    )
    parser.add_argument(
        "--envoy-router-api-address",
        default=DEFAULT_ENVOY_ROUTER_API_ADDRESS,
        help="Host written into generated Envoy config for the router API cluster.",
    )
    parser.add_argument("--evalscope-python", default=DEFAULT_EVALSCOPE_PYTHON)
    parser.add_argument("--evalscope-bin", default=DEFAULT_EVALSCOPE_BIN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--router-image",
        default="",
        help="Optional router image to recreate vllm-sr-router-container with after switching configs.",
    )
    parser.add_argument("--restart-wait-seconds", type=int, default=8)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-envoy-sync", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument(
        "--local-results-dir",
        type=Path,
        default=BENCH_ROOT / "results",
        help="Destination root for pulled report and EvalScope artifacts.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-delay-seconds", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(
    cmd: list[str], *, dry_run: bool, retries: int = 1, retry_delay_seconds: int = 0
) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in RETRYABLE_RETURN_CODES or attempt >= attempts:
                raise
            time.sleep(max(0, retry_delay_seconds))


def run_with_input(
    cmd: list[str],
    stdin: str,
    *,
    dry_run: bool,
    retries: int = 1,
    retry_delay_seconds: int = 0,
) -> None:
    print(shlex.join(cmd))
    if dry_run:
        print(stdin.rstrip())
        return
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(cmd, input=stdin, text=True, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in RETRYABLE_RETURN_CODES or attempt >= attempts:
                raise
            time.sleep(max(0, retry_delay_seconds))


def run_capture_with_input(
    cmd: list[str],
    stdin: str,
    *,
    dry_run: bool,
    retries: int = 1,
    retry_delay_seconds: int = 0,
) -> str:
    print(shlex.join(cmd))
    if dry_run:
        print(stdin.rstrip())
        return ""
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                cmd,
                input=stdin,
                text=True,
                check=True,
                capture_output=True,
            )
            return completed.stdout.strip()
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in RETRYABLE_RETURN_CODES or attempt >= attempts:
                raise
            time.sleep(max(0, retry_delay_seconds))
    return ""


def ssh(args: argparse.Namespace, host: str, script: str) -> None:
    run_with_input(
        ["ssh", *SSH_OPTIONS, host, "bash", "-se"],
        script + "\n",
        dry_run=args.dry_run,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )


def ssh_capture(args: argparse.Namespace, host: str, script: str) -> str:
    return run_capture_with_input(
        ["ssh", *SSH_OPTIONS, host, "bash", "-se"],
        script + "\n",
        dry_run=args.dry_run,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )


def rsync_to(args: argparse.Namespace, src: str | Path, host: str, dest: str) -> None:
    run(
        ["rsync", "-az", str(src), f"{host}:{dest}"],
        dry_run=args.dry_run,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )


def rsync_from(args: argparse.Namespace, host: str, src: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    run(
        ["rsync", "-az", f"{host}:{src}", str(dest)],
        dry_run=args.dry_run,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )


def require_host(host: str) -> str:
    if host.strip():
        return host.strip()
    raise SystemExit("AMD eval host is required: pass --host or set VLLM_SR_AMD_HOST")


def recipe_set(args: argparse.Namespace) -> dict[str, object]:
    name = getattr(args, "recipe_set", "closed")
    return RECIPE_SETS[str(name)]


def benchmark_configs(args: argparse.Namespace) -> dict[str, str]:
    return recipe_set(args)["configs"]  # type: ignore[return-value]


def suite_filename(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "suite", "") or "").strip()
    if requested:
        return requested
    return str(recipe_set(args)["suite"])


def selected_models(args: argparse.Namespace) -> list[str]:
    requested = list(getattr(args, "model", []) or [])
    if requested:
        return requested
    return list(recipe_set(args)["default_models"])  # type: ignore[arg-type]


def selected_benchmarks(
    requested: list[str], recipe_set_name: str = "closed"
) -> list[str]:
    configs = RECIPE_SETS[recipe_set_name]["configs"]
    selected = requested or list(configs)
    unknown = sorted(set(selected) - set(configs))
    if unknown:
        raise SystemExit(
            f"benchmark(s) not configured for recipe set {recipe_set_name}: "
            + ", ".join(unknown)
        )
    return selected


def dir_contents(path: Path) -> str:
    return str(path) + "/"


def sync_inputs(args: argparse.Namespace, host: str) -> None:
    if args.skip_sync:
        return
    remote_eval = quote(args.remote_eval_root)
    source_dir = remote_source_config_dir(args)
    ssh(
        args,
        host,
        (
            f"mkdir -p {remote_eval}/real_eval {quote(source_dir)}\n"
            f"find {quote(source_dir)} -maxdepth 1 -type f -name '*.yaml' -delete"
        ),
    )
    rsync_to(
        args, dir_contents(REAL_EVAL_ROOT), host, f"{args.remote_eval_root}/real_eval/"
    )
    rsync_to(
        args,
        BENCH_ROOT / "public_reference_scores.json",
        host,
        f"{args.remote_eval_root}/real_eval/public_reference_scores.json",
    )
    rsync_to(args, dir_contents(CONFIG_ROOT), host, f"{source_dir}/")


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def remote_source_config_dir(args: argparse.Namespace) -> str:
    source_dir = args.remote_source_config_dir.strip("/")
    return f"{args.remote_config_dir}/{source_dir}"


def remote_openrouter_key_script(args: argparse.Namespace) -> str:
    openrouter_key_file = quote(args.remote_openrouter_api_key_file)
    return "\n".join(
        [
            "set -euo pipefail",
            'openrouter_api_key=""',
            "if docker inspect vllm-sr-router-container >/dev/null 2>&1; then",
            "  openrouter_api_key=$(docker inspect vllm-sr-router-container --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^OPENROUTER_API_KEY=//p' | head -n1)",
            "fi",
            f'if [ -z "$openrouter_api_key" ] && [ -r {openrouter_key_file} ]; then openrouter_api_key="$(tr -d \'\\r\\n\' < {openrouter_key_file})"; fi',
            'if [ -z "$openrouter_api_key" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then openrouter_api_key="$OPENROUTER_API_KEY"; fi',
            'if [ -z "$openrouter_api_key" ]; then echo "OPENROUTER_API_KEY is unavailable; set the existing router env, remote key file, or remote shell env" >&2; exit 2; fi',
            'printf "%s" "$openrouter_api_key"',
        ]
    )


def fetch_openrouter_api_key(args: argparse.Namespace, host: str) -> str:
    if args.dry_run:
        return "dry-run-openrouter-key"
    return ssh_capture(args, host, remote_openrouter_key_script(args))


def remote_envoy_config_script(args: argparse.Namespace) -> str:
    container = quote(args.remote_envoy_container)
    return "\n".join(
        [
            "set -euo pipefail",
            f"docker inspect {container} >/dev/null",
            f"docker inspect {container} --format '{{{{range .Mounts}}}}{{{{if eq .Destination \"/etc/envoy/envoy.yaml\"}}}}{{{{.Source}}}}{{{{end}}}}{{{{end}}}}'",
        ]
    )


def resolve_remote_envoy_config(args: argparse.Namespace, host: str) -> str:
    if args.remote_envoy_config:
        return args.remote_envoy_config
    if args.dry_run:
        return "/remote/discovered/envoy.yaml"
    remote_envoy_config = ssh_capture(args, host, remote_envoy_config_script(args))
    if not remote_envoy_config:
        raise SystemExit(
            "Unable to discover remote Envoy config mount; pass --remote-envoy-config"
        )
    return remote_envoy_config


def render_envoy_config_for_benchmark(
    args: argparse.Namespace, benchmark: str, api_key: str, output_file: Path
) -> None:
    config_path = CONFIG_ROOT / benchmark_configs(args)[benchmark]
    env_overrides = {
        "OPENROUTER_API_KEY": api_key,
        "ENVOY_EXTPROC_ADDRESS": args.envoy_extproc_address,
        "ENVOY_ROUTER_API_ADDRESS": args.envoy_router_api_address,
    }
    previous = {key: os.environ.get(key) for key in env_overrides}
    os.environ.update(env_overrides)
    try:
        generate_envoy_config_from_user_config(
            parse_user_config(str(config_path)), str(output_file)
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sync_envoy_config(args: argparse.Namespace, host: str, benchmark: str) -> None:
    if args.skip_envoy_sync:
        return
    api_key = fetch_openrouter_api_key(args, host)
    remote_envoy_config = resolve_remote_envoy_config(args, host)
    ssh(args, host, f"mkdir -p {quote(Path(remote_envoy_config).parent)}")
    with tempfile.TemporaryDirectory(prefix="vllm-sr-omni-envoy-") as temp_dir:
        local_envoy_config = Path(temp_dir) / f"{benchmark}_envoy.yaml"
        render_envoy_config_for_benchmark(args, benchmark, api_key, local_envoy_config)
        rsync_to(args, local_envoy_config, host, remote_envoy_config)


def envoy_restart_script(args: argparse.Namespace) -> list[str]:
    container = quote(args.remote_envoy_container)
    return [
        f"docker inspect {container} >/dev/null",
        f"docker restart {container} >/dev/null",
    ]


def router_restart_script(args: argparse.Namespace, target_config: str) -> list[str]:
    target = quote(target_config)
    openrouter_key_file = quote(args.remote_openrouter_api_key_file)
    image_assignment = (
        f"router_image={quote(args.router_image)}"
        if args.router_image
        else "router_image=\"$(docker inspect vllm-sr-router-container --format '{{.Config.Image}}')\""
    )
    script = [
        'current_config_mount=""',
        'openrouter_api_key=""',
        'models_mount="vllm-sr-router-models"',
        "if docker inspect vllm-sr-router-container >/dev/null 2>&1; then",
        "  current_config_mount=$(docker inspect vllm-sr-router-container --format '{{range .Mounts}}{{if eq .Destination \"/app/config.yaml\"}}{{.Source}}{{end}}{{end}}')",
        "  openrouter_api_key=$(docker inspect vllm-sr-router-container --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^OPENROUTER_API_KEY=//p' | head -n1)",
        "  inspected_mount=$(docker inspect vllm-sr-router-container --format '{{range .Mounts}}{{if eq .Destination \"/app/models\"}}{{if .Name}}{{.Name}}{{else}}{{.Source}}{{end}}{{end}}{{end}}')",
        '  if [ -n "$inspected_mount" ]; then models_mount="$inspected_mount"; fi',
        "fi",
        image_assignment,
        f'if [ -z "{args.router_image}" ] && [ -n "$current_config_mount" ] && [ "$current_config_mount" = {target} ]; then',
        "  docker restart vllm-sr-router-container >/dev/null",
        "else",
        '  if [ -z "$router_image" ]; then echo "router image is unavailable; pass --router-image or keep an inspectable existing router container" >&2; exit 2; fi',
        f'  if [ -z "$openrouter_api_key" ] && [ -r {openrouter_key_file} ]; then openrouter_api_key="$(tr -d \'\\r\\n\' < {openrouter_key_file})"; fi',
        '  if [ -z "$openrouter_api_key" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then openrouter_api_key="$OPENROUTER_API_KEY"; fi',
        '  if [ -z "$openrouter_api_key" ]; then echo "OPENROUTER_API_KEY is unavailable; set the existing router env, remote key file, or remote shell env" >&2; exit 2; fi',
        '  if [ -z "$models_mount" ]; then models_mount=vllm-sr-router-models; fi',
        "  docker rm -f vllm-sr-router-container >/dev/null 2>&1 || true",
        (
            "  docker run -d --name vllm-sr-router-container "
            "--network vllm-sr-network "
            "--restart unless-stopped "
            '-e OPENROUTER_API_KEY="$openrouter_api_key" '
            f"-v {target}:/app/config.yaml:ro "
            '-v "$models_mount":/app/models '
            '"$router_image" /app/config.yaml >/dev/null'
        ),
        "fi",
    ]
    return script


def router_readiness_script(args: argparse.Namespace, model_name: str) -> list[str]:
    attempts = max(1, int(args.restart_wait_seconds))
    quoted_model = quote(model_name)
    return [
        "ready=0",
        f"for _ in $(seq 1 {attempts}); do",
        "  models_json=$(curl -fsS http://127.0.0.1:8899/v1/models 2>/dev/null || true)",
        f'  if [ -n "$models_json" ] && printf \'%s\\n\' "$models_json" | grep -F {quoted_model} >/dev/null; then',
        "    ready=1",
        "    printf '%s\\n' \"$models_json\"",
        "    break",
        "  fi",
        "  sleep 1",
        "done",
        'if [ "$ready" != "1" ]; then',
        "  curl -fsS http://127.0.0.1:8899/v1/models",
        f"  curl -fsS http://127.0.0.1:8899/v1/models | grep -F {quoted_model} >/dev/null",
        "fi",
    ]


def switch_benchmark_recipe(
    args: argparse.Namespace, host: str, benchmark: str
) -> None:
    source = f"{remote_source_config_dir(args)}/{benchmark_configs(args)[benchmark]}"
    target = f"{args.remote_config_dir}/{args.remote_router_config}"
    copy_script = "\n".join(
        [
            "set -euo pipefail",
            f"if [ {quote(source)} != {quote(target)} ]; then cp {quote(source)} {quote(target)}; fi",
        ]
    )
    ssh(args, host, copy_script)
    sync_envoy_config(args, host, benchmark)
    restart_script = "\n".join(
        [
            "set -euo pipefail",
            *router_restart_script(args, target),
            *envoy_restart_script(args),
            *router_readiness_script(args, OMNI_MODEL_NAME),
        ]
    )
    ssh(args, host, restart_script)


def run_evalscope(
    args: argparse.Namespace, host: str, benchmark: str, model_key: str
) -> None:
    api_url = MODEL_API_URLS.get(model_key, MODEL_API_URLS["auto"])
    api_key = "EMPTY"
    cmd = [
        args.evalscope_python,
        "real_eval/run_evalscope_suite.py",
        "--suite",
        f"real_eval/{suite_filename(args)}",
        "--output-root",
        args.output_root,
        "--evalscope-bin",
        args.evalscope_bin,
        "--api-url",
        api_url,
        "--api-key",
        api_key,
        "--model",
        model_key,
        "--benchmark",
        benchmark,
        "--limit-mode",
        args.limit_mode,
    ]
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    command = shlex.join(cmd)
    if model_key in DIRECT_OPENROUTER_MODEL_KEYS:
        command = command.replace("--api-key EMPTY", '--api-key "$openrouter_api_key"')
        script = "\n".join(
            [
                "set -euo pipefail",
                'openrouter_api_key="$(',
                remote_openrouter_key_script(args),
                ')"',
                f"cd {quote(args.remote_eval_root)}",
                command,
            ]
        )
    else:
        script = "\n".join(
            [
                "set -euo pipefail",
                'export OPENROUTER_API_KEY="$(',
                remote_openrouter_key_script(args),
                ')"',
                'export OPENAI_API_KEY="$OPENROUTER_API_KEY"',
                f"cd {quote(args.remote_eval_root)}",
                command,
            ]
        )
    ssh(args, host, script)


def collect(
    args: argparse.Namespace, host: str, benchmarks: list[str], models: list[str]
) -> None:
    cmd = [
        args.evalscope_python,
        "real_eval/collect_evalscope_results.py",
        "--suite",
        f"real_eval/{suite_filename(args)}",
        "--output-root",
        args.output_root,
        "--reference-json",
        "real_eval/public_reference_scores.json",
        "--output-dir",
        args.report_dir,
    ]
    for benchmark in benchmarks:
        cmd.extend(["--benchmark", benchmark])
    for model_key in models:
        cmd.extend(["--model", model_key])
    if args.require_complete:
        cmd.append("--require-complete")
    script = (
        "set -euo pipefail\n" + f"cd {quote(args.remote_eval_root)}\n" + shlex.join(cmd)
    )
    ssh(args, host, script)


def pull_outputs(args: argparse.Namespace, host: str) -> None:
    if not args.pull:
        return
    rsync_from(
        args,
        host,
        f"{args.remote_eval_root}/{args.report_dir}/",
        args.local_results_dir / Path(args.report_dir).name,
    )
    rsync_from(
        args,
        host,
        f"{args.remote_eval_root}/{args.output_root}/evalscope_manifest.json",
        args.local_results_dir / Path(args.output_root).name,
    )
    for benchmark in selected_benchmarks(args.benchmark, args.recipe_set):
        rsync_from(
            args,
            host,
            f"{args.remote_eval_root}/{args.output_root}/{benchmark}/",
            args.local_results_dir / Path(args.output_root).name / benchmark,
        )


def main() -> int:
    args = parse_args()
    host = require_host(args.host)
    benchmarks = selected_benchmarks(args.benchmark, args.recipe_set)
    models = selected_models(args)
    sync_inputs(args, host)
    for benchmark in benchmarks:
        if any(model in ROUTER_MODEL_KEYS for model in models):
            switch_benchmark_recipe(args, host, benchmark)
        for model_key in models:
            run_evalscope(args, host, benchmark, model_key)
    collect(args, host, benchmarks, models)
    pull_outputs(args, host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
