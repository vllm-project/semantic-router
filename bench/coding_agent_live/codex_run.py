"""Run Codex CLI on Harbor tasks through the Router.

Builds one ``harbor run`` job: Harbor's Codex agent at a pinned version, pointed
at the Router's Responses ingress, with ``web_search`` disabled and
``multi_agent`` off. The model Codex would otherwise keep, including Harbor's
stripped ``--model`` suffix, is the Router entrypoint. ``codex_report.py`` reads
the job afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_VERSION = "0.163.0"
DEFAULT_ATTEMPTS = 1
DEFAULT_CONCURRENCY = 4
AGENT_IMPORT_PATH = "coding_agent_live.codex_agent:RouterCodex"
BENCH_DIR = Path(__file__).resolve().parent.parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", required=True, help="Harbor task or dataset path (-p)"
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to include; repeatable (default: every task)",
    )
    parser.add_argument(
        "--router-url",
        required=True,
        help="Router inference URL as seen from the task containers, "
        "for example http://172.17.0.1:8899; /v1 is appended when missing",
    )
    parser.add_argument(
        "--model", required=True, help="Router entrypoint for every Codex model setting"
    )
    parser.add_argument(
        "--codex-version",
        default=DEFAULT_VERSION,
        help=f"Codex CLI version to install (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--api-key",
        default="router",
        help="Key Codex sends; the Router checks it only when auth is on",
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--job-name", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Codex config and Harbor command without running Harbor",
    )
    return parser.parse_args(argv)


def responses_base_url(router_url: str) -> str:
    """Responses base URL. Codex appends ``/responses`` to this path."""
    base = router_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def toml_basic_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_config(model: str) -> str:
    """Codex ``config.toml`` body: one entrypoint, search off, multi-agent off."""
    return (
        f"model = {toml_basic_string(model)}\n"
        'web_search = "disabled"\n'
        "\n"
        "[features]\n"
        "multi_agent = false\n"
    )


def _is_model_setting(key: str, value: object) -> bool:
    return isinstance(value, str) and (key == "model" or key.endswith("_model"))


def _rewrite_models(config: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Copy config, pointing every model setting at the Router entrypoint.

    ``model_providers`` names providers and base URLs, not model ids, so it is
    left unchanged. Codex's root and a child can both carry a model id; the
    lane does not leave either on Codex's own default.
    """
    rewritten: dict[str, Any] = {}
    for key, value in config.items():
        if _is_model_setting(key, value):
            rewritten[key] = model
        elif isinstance(value, dict) and key != "model_providers":
            rewritten[key] = _rewrite_models(value, model)
        else:
            rewritten[key] = value
    return rewritten


def merge_router_settings(config: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Protocol-guide settings on top of Harbor's effective Codex config."""
    merged = _rewrite_models(config, model)
    merged["model"] = model
    merged["web_search"] = "disabled"
    features = merged.get("features")
    feature_table = dict(features) if isinstance(features, dict) else {}
    feature_table["multi_agent"] = False
    merged["features"] = feature_table
    return merged


def apply_router_model(command: str, model_name: str) -> str:
    """Put the full entrypoint back into Harbor's ``codex exec --model``.

    Harbor keeps only the segment after the last slash, so ``vllm-sr/parent``
    would be sent as ``parent`` and miss the Router entrypoint.
    """
    if "/" not in model_name:
        return command
    stripped = model_name.rsplit("/", 1)[-1]
    needle = f"--model {stripped} "
    if needle not in command:
        return command
    return command.replace(needle, f"--model {shlex.quote(model_name)} ", 1)


def codex_config_path(jobs_dir: Path, job_name: str) -> Path:
    return (jobs_dir / f"{job_name}.codex.toml").resolve()


def lane_metadata_path(jobs_dir: Path, job_name: str) -> Path:
    return (jobs_dir / f"{job_name}.codex-lane.json").resolve()


def harbor_command(args: argparse.Namespace) -> list[str]:
    command = [
        "harbor",
        "run",
        "-p",
        str(args.dataset),
        "-a",
        AGENT_IMPORT_PATH,
        "-m",
        args.model,
        "--ak",
        f"version={args.codex_version}",
        "--ak",
        "web_search=disabled",
        "--ak",
        f"config={codex_config_path(args.jobs_dir, args.job_name)}",
        "--ae",
        f"OPENAI_BASE_URL={responses_base_url(args.router_url)}",
        "--ae",
        f"OPENAI_API_KEY={args.api_key}",
        "-k",
        str(args.attempts),
        "-n",
        str(args.concurrency),
        "-o",
        str(args.jobs_dir),
        "--job-name",
        args.job_name,
    ]
    for task in args.task:
        command += ["-i", task]
    return command


def write_lane_files(args: argparse.Namespace) -> None:
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    codex_config_path(args.jobs_dir, args.job_name).write_text(
        render_config(args.model)
    )
    lane_metadata_path(args.jobs_dir, args.job_name).write_text(
        json.dumps({"codex_version": args.codex_version}) + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = harbor_command(args)
    if args.dry_run:
        print(render_config(args.model), end="")
        print(shlex.join(command))
        return 0
    write_lane_files(args)
    env = os.environ.copy()
    path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(BENCH_DIR) + (os.pathsep + path if path else "")
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
