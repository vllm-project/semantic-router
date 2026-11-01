"""Run Claude Code on Harbor tasks through the Router.

Builds one ``harbor run`` job: Harbor's Claude Code agent at a pinned version,
pointed at the Router's Anthropic Messages ingress, with each model alias and
the subagent default mapped to a Router entrypoint. Run it from the Python
environment that has Harbor installed, for example the one ``vllm-sr bench``
sets up for Terminal-Bench. ``report.py`` reads the job afterwards.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_VERSION: str = "2.1.304"
AGENT_IMPORT_PATH: str = "coding_agent_live.agent:RouterClaudeCode"
BENCH_DIR: Path = Path(__file__).resolve().parent.parent
ALIAS_VARIABLES: dict[str, str] = {
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "subagent": "CLAUDE_CODE_SUBAGENT_MODEL",
}


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
        "for example http://172.17.0.1:8899",
    )
    parser.add_argument(
        "--model", required=True, help="Router model name for the main agent"
    )
    for alias, variable in ALIAS_VARIABLES.items():
        parser.add_argument(
            f"--{alias}-model",
            help=f"Router model name for {variable} "
            "(default: --model; an empty value leaves it to Claude Code)",
        )
    parser.add_argument(
        "--claude-code-version",
        default=DEFAULT_VERSION,
        help=f"Claude Code version to install (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--api-key",
        default="router",
        help="Key Claude Code sends; the Router checks it only when auth is on",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Harbor Claude Code option, for example reasoning_effort=xhigh or "
        "append_system_prompt=...; repeatable",
    )
    parser.add_argument(
        "--agent-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for Claude Code in the task container, for "
        "example CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1; repeatable",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--job-name", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Harbor command and alias mapping without running it",
    )
    return parser.parse_args(argv)


def lane_env(args: argparse.Namespace) -> dict[str, str]:
    """LANE_<variable> entries that the agent applies inside the task container."""
    env = {}
    for alias, variable in ALIAS_VARIABLES.items():
        value = getattr(args, f"{alias}_model")
        env["LANE_" + variable] = args.model if value is None else value
    return env


def apply_lane_env(
    env: Mapping[str, str], environ: Mapping[str, str]
) -> dict[str, str]:
    """Container env with each alias taken from its LANE_ entry, removed when empty."""
    result = dict(env)
    for variable in ALIAS_VARIABLES.values():
        value = environ.get("LANE_" + variable)
        if value is None:
            continue
        if value:
            result[variable] = value
        else:
            result.pop(variable, None)
    return result


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
        f"version={args.claude_code_version}",
        *[part for kwarg in args.agent_kwarg for part in ("--ak", kwarg)],
        "--ae",
        f"ANTHROPIC_BASE_URL={args.router_url}",
        "--ae",
        f"ANTHROPIC_API_KEY={args.api_key}",
        *[part for env in args.agent_env for part in ("--ae", env)],
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = harbor_command(args)
    env = lane_env(args)
    if args.dry_run:
        print(" ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()))
        print(shlex.join(command))
        return 0
    path = os.environ.get("PYTHONPATH")
    env["PYTHONPATH"] = str(BENCH_DIR) + (os.pathsep + path if path else "")
    return subprocess.run(command, env={**os.environ, **env}, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
