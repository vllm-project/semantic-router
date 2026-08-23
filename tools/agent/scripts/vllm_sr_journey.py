#!/usr/bin/env python3
"""Contributor journey orchestration for vLLM-SR deployment, validation, and review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = REPO_ROOT / ".agent-harness" / "vllm-sr-journey"
REQUIRED_RECIPE_FILES = (
    "config.yaml",
    "metadata.yaml",
    "recipe.dsl",
    "probes.yaml",
    "README.md",
)
PLACEHOLDER_ENDPOINT = "host.docker.internal:8000"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _load_repo_manifest() -> dict[str, Any]:
    manifest_path = REPO_ROOT / "tools" / "agent" / "repo-manifest.yaml"
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8"))


def _detect_gpu_vendor() -> str | None:
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return "nvidia"
    if Path("/dev/kfd").exists() or shutil.which("rocm-smi"):
        return "amd"
    return None


def detect_environment() -> dict[str, Any]:
    manifest = _load_repo_manifest()
    platform_hint = os.environ.get("VLLM_SR_PLATFORM", "").strip().lower()
    gpu_vendor = _detect_gpu_vendor()
    default_env = manifest.get("default_env", "cpu")
    supported = manifest.get("supported_envs", {})

    selected = default_env
    reasons: list[str] = []

    if platform_hint in {"cpu", "amd", "nvidia"}:
        selected = platform_hint
        reasons.append(f"VLLM_SR_PLATFORM={platform_hint}")
    elif gpu_vendor == "amd":
        selected = "amd"
        reasons.append("detected AMD/ROCm tooling")
    elif gpu_vendor == "nvidia":
        selected = "nvidia"
        reasons.append("detected NVIDIA GPU tooling")
    else:
        reasons.append("default cpu-local path")

    env_key = {
        "cpu": "cpu-local",
        "amd": "amd-local",
        "nvidia": "nvidia-local",
        "ci-k8s": "ci-k8s",
    }.get(selected, "cpu-local")

    env_config = supported.get(env_key, {})
    aliases = env_config.get("aliases", [])
    return {
        "detected_at": _utc_now(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "gpu_vendor": gpu_vendor,
        "selected_env": env_key,
        "selected_alias": selected,
        "reasons": reasons,
        "build_target": env_config.get("build_target"),
        "serve_command": env_config.get("serve_command"),
        "smoke_config": env_config.get("smoke_config"),
        "deployment_reference": env_config.get("deployment_reference"),
        "example_config": env_config.get("example_config"),
        "supported_envs": sorted(supported.keys()),
        "aliases": aliases,
    }


def _validate_config(config_path: Path) -> dict[str, Any]:
    vllm_sr = shutil.which("vllm-sr")
    if vllm_sr is None:
        validate_script = (
            REPO_ROOT / "src" / "vllm-sr" / "cli" / "commands" / "validate.py"
        )
        parser_script = REPO_ROOT / "src" / "vllm-sr" / "cli" / "parser.py"
        if not parser_script.exists():
            return {
                "passed": False,
                "error": "vllm-sr CLI is not installed and fallback parser is unavailable",
            }
        result = _run_command(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, 'src/vllm-sr'); "
                    "from cli.parser import parse_user_config; "
                    "from cli.validator import validate_user_config; "
                    f"cfg = parse_user_config('{config_path}', log_summary=False); "
                    "errors = validate_user_config(cfg, log_summary=False); "
                    "print('ok' if not errors else '\\n'.join(errors)); "
                    "sys.exit(0 if not errors else 1)"
                ),
            ]
        )
    else:
        result = _run_command(["vllm-sr", "validate", "--config", str(config_path)])
    return {
        "kind": "vllm-sr validate",
        "config": str(config_path),
        **result,
    }


def _validate_recipe_directory(recipe_dir: Path) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_RECIPE_FILES if not (recipe_dir / name).is_file()
    ]
    extra = [
        entry.name
        for entry in recipe_dir.iterdir()
        if entry.is_file() and entry.name not in REQUIRED_RECIPE_FILES
    ]
    structure_ok = not missing and not extra
    conformance_errors: list[str] = []
    if structure_ok:
        sys.path.insert(0, str(REPO_ROOT / "tools" / "agent" / "scripts"))
        import recipe_conformance

        try:
            recipe_conformance.validate_recipe_directory(recipe_dir)
            recipe_conformance.validate_recipe_model_card(recipe_dir)
            recipe_conformance.build_recipe_inventory(recipe_dir)
        except (TypeError, ValueError) as error:
            conformance_errors.append(str(error))
    return {
        "kind": "recipe directory",
        "recipe_dir": str(recipe_dir),
        "missing_files": missing,
        "extra_files": extra,
        "structure_ok": structure_ok,
        "conformance_errors": conformance_errors,
        "passed": structure_ok and not conformance_errors,
    }


def validate_artifacts(config_path: Path, recipe_dir: Path | None) -> dict[str, Any]:
    config_result = _validate_config(config_path)
    recipe_result = None
    if recipe_dir is not None:
        recipe_result = _validate_recipe_directory(recipe_dir)
    passed = config_result.get("passed", False) and (
        recipe_result is None or recipe_result.get("passed", False)
    )
    return {
        "validated_at": _utc_now(),
        "config": config_result,
        "recipe": recipe_result,
        "passed": passed,
    }


def evaluate_artifacts(
    config_path: Path,
    *,
    router_url: str | None,
    probes_path: Path | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"evaluated_at": _utc_now(), "live": None, "static": None}

    static = _run_command(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "agent" / "scripts" / "recipe_conformance.py"),
            "static",
        ]
    )
    results["static"] = static

    if router_url and probes_path is not None:
        live = _run_command(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "tools"
                    / "agent"
                    / "scripts"
                    / "router_calibration_loop.py"
                ),
                "eval",
                "--router-url",
                router_url,
                "--probes",
                str(probes_path),
            ]
        )
        results["live"] = live

    passed = static.get("passed", False)
    if results["live"] is not None:
        passed = passed and results["live"].get("passed", False)
    results["passed"] = passed
    results["config"] = str(config_path)
    return results


def _routing_summary(config_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src" / "vllm-sr"))
    from cli.config_contract import iter_routing_profiles
    from cli.parser import parse_user_config

    user_config = parse_user_config(str(config_path), log_summary=False)
    profiles = list(iter_routing_profiles(user_config))
    decisions = [
        {
            "name": decision.name,
            "priority": decision.priority,
            "algorithm": getattr(decision.algorithm, "type", None),
            "models": [ref.model for ref in decision.model_refs],
            "plugins": [plugin.type.value for plugin in decision.plugins or []],
        }
        for _, profile in profiles
        for decision in profile.decisions
    ]
    return {
        "listeners": len(user_config.listeners),
        "providers": [model.name for model in user_config.providers.models],
        "default_model": user_config.providers.default_model,
        "entrypoints": len(user_config.entrypoints),
        "recipes": len(user_config.recipes),
        "decisions": decisions,
    }


def build_review_bundle(
    config_path: Path,
    *,
    recipe_dir: Path | None,
    output_dir: Path,
    router_url: str | None,
    probes_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_artifacts(config_path, recipe_dir)
    evaluation = evaluate_artifacts(
        config_path,
        router_url=router_url,
        probes_path=probes_path,
    )
    artifacts = [{"path": str(config_path), "sha256": _sha256_file(config_path)}]
    if recipe_dir is not None:
        for filename in REQUIRED_RECIPE_FILES:
            file_path = recipe_dir / filename
            if file_path.is_file():
                artifacts.append(
                    {"path": str(file_path), "sha256": _sha256_file(file_path)}
                )

    try:
        routing_summary = _routing_summary(config_path)
    except (
        Exception
    ) as error:  # noqa: BLE001 - review bundle should capture parse failures
        routing_summary = {"error": str(error)}

    bundle = {
        "generated_at": _utc_now(),
        "activated": False,
        "config": str(config_path),
        "recipe_dir": str(recipe_dir) if recipe_dir else None,
        "artifacts": artifacts,
        "validation": validation,
        "evaluation": evaluation,
        "routing_summary": routing_summary,
        "rollback": {
            "git": "Revert the commit or discard unstaged changes for generated artifacts.",
            "live_apiserver": "Use GET /config/router/versions before deploy when activating on a live router.",
            "helm": "Restore the previous configOverride values snapshot.",
        },
        "activation_options": [
            "make agent-serve-local ENV=cpu|amd",
            "vllm-sr serve --config <path>",
            "router_calibration_loop.py deploy --router-url <url> --yaml <path>",
        ],
    }

    _write_json(output_dir / "review.json", bundle)
    markdown = [
        "# vLLM-SR Journey Review Bundle",
        "",
        f"- Generated: {bundle['generated_at']}",
        f"- Activated: `{bundle['activated']}`",
        f"- Config: `{config_path}`",
        "",
        "## Validation",
        f"- Passed: `{validation['passed']}`",
        "",
        "## Evaluation",
        f"- Passed: `{evaluation['passed']}`",
        "",
        "## Routing Summary",
        "```json",
        json.dumps(routing_summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Rollback",
        "- Git: revert or discard local artifact changes.",
        "- Live apiserver: capture `/config/router/versions` before deploy.",
        "- Helm: restore the previous `configOverride` snapshot.",
        "",
        "## Activation",
        "Activation is explicit. Do not treat this bundle as production approval.",
    ]
    (output_dir / "review.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return bundle


def cmd_detect_env(_args: argparse.Namespace) -> int:
    payload = detect_environment()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    payload = validate_artifacts(
        Path(args.config), Path(args.recipe_dir) if args.recipe_dir else None
    )
    if args.output:
        _write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    payload = evaluate_artifacts(
        Path(args.config),
        router_url=args.router_url,
        probes_path=Path(args.probes) if args.probes else None,
    )
    if args.output:
        _write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def cmd_review(args: argparse.Namespace) -> int:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_REPORT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    payload = build_review_bundle(
        Path(args.config),
        recipe_dir=Path(args.recipe_dir) if args.recipe_dir else None,
        output_dir=output_dir,
        router_url=args.router_url,
        probes_path=Path(args.probes) if args.probes else None,
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "activated": payload["activated"]}, indent=2
        )
    )
    return 0 if payload["validation"]["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "detect-env", help="Detect the supported local deployment path"
    )

    validate = subparsers.add_parser(
        "validate", help="Validate config and optional recipe directory"
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--recipe-dir", type=Path, default=None)
    validate.add_argument("--output", type=Path, default=None)

    evaluate = subparsers.add_parser(
        "evaluate", help="Run static and optional live evaluation gates"
    )
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--router-url", default=None)
    evaluate.add_argument("--probes", type=Path, default=None)
    evaluate.add_argument("--output", type=Path, default=None)

    review = subparsers.add_parser(
        "review", help="Write a review bundle with provenance and rollback hints"
    )
    review.add_argument("--config", required=True, type=Path)
    review.add_argument("--recipe-dir", type=Path, default=None)
    review.add_argument("--router-url", default=None)
    review.add_argument("--probes", type=Path, default=None)
    review.add_argument("--output-dir", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect-env":
        return cmd_detect_env(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "review":
        return cmd_review(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
