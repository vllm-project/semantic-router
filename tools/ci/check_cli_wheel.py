#!/usr/bin/env python3
"""Exercise agent bootstrap capabilities from an installed wheel, without Docker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from collections.abc import Mapping
from pathlib import Path


def check_wheel(wheel: Path, decision_images: Mapping[str, str] | None = None) -> None:
    wheel = wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="vllm-sr-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        child_env: dict[str, str] | None = None
        launcher_version: str | None = None
        if sys.platform not in {"darwin", "linux"}:
            venv.EnvBuilder(with_pip=True).create(environment)
            subprocess.run(
                args=[
                    str(python),
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    str(wheel),
                ],
                cwd=root,
                check=True,
            )
        else:
            child_env = {
                "PATH": os.defpath,
                "HOME": str(root),
                "TMPDIR": str(root),
                "SHELL": "/bin/bash",
                "PIP_CONFIG_FILE": os.devnull,
            }
            shell_config = root / ".bashrc"
            existing_shell_config = "export EXISTING_SETTING=preserved\n"
            shell_config.write_text(data=existing_shell_config, encoding="utf-8")
            subprocess.run(
                args=[
                    "bash",
                    str(Path(__file__).resolve().parents[2] / "install.sh"),
                    "--mode",
                    "cli",
                    "--runtime",
                    "skip",
                    "--no-launch",
                    "--python",
                    sys.executable,
                    "--pip-spec",
                    str(wheel),
                    "--install-root",
                    str(root),
                    "--bin-dir",
                    str(root / "bin"),
                ],
                cwd=root,
                env=child_env,
                check=True,
                timeout=300,
            )
            completion = shell_config.read_text(encoding="utf-8")
            if not completion.startswith(existing_shell_config) or (
                "vllm-sr completion show bash" not in completion
            ):
                raise RuntimeError(
                    "installer did not preserve and extend shell configuration"
                )
            if (root / "runtime.env").exists():
                raise RuntimeError(
                    "CLI-only installation configured a container runtime"
                )
            launcher_version = subprocess.check_output(
                args=[str(root / "bin" / "vllm-sr"), "--version"],
                cwd=root,
                env=child_env,
                text=True,
                timeout=60,
            ).strip()
            print(
                f"Installer launcher: {launcher_version}; completion stayed in isolated HOME."
            )

        def cli(*arguments: str) -> str:
            # Neither the working directory nor PYTHONPATH can supply CLI code
            # or missing resources from the checkout under isolated Python.
            result = subprocess.run(
                args=[str(python), "-I", "-m", "cli.main", *arguments],
                cwd=root,
                env=child_env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError(
                    f"installed CLI failed: {' '.join(arguments)}\n"
                    f"{result.stdout}{result.stderr}"
                )
            return result.stdout

        version = cli("--version").strip()
        if launcher_version is not None and launcher_version != version:
            raise RuntimeError("installer launcher does not match the installed wheel")
        print(version)
        schema = json.loads(cli("config", "schema"))
        if not schema.get("sections"):
            raise RuntimeError("installed CLI has no progressive schema index")
        json.loads(cli("config", "schema", "--section", "providers.models"))
        serve_help = cli("serve", "--help")
        if "--replace-active-config" not in serve_help:
            raise RuntimeError("installed CLI cannot explicitly replace runtime config")
        for command in ("get", "plan", "apply", "validate"):
            cli("config", command, "--help")
        cli("route", "preview", "--help")
        probe_help = cli("route", "probe", "--help")
        for option in (
            "--expect-selected-model",
            "--expect-response-model",
            "--max-completion-tokens",
        ):
            if option not in probe_help:
                raise RuntimeError(f"installed route probe is missing {option}")
        config_path = root / "config.yaml"
        cli("config", "init", "--output", str(config_path))
        cli("config", "validate", "--config", str(config_path))
        catalog = json.loads(cli("recipe", "builtin", "list"))
        mom = next(item for item in catalog["bundles"] if item["name"] == "mom-v1")
        balance = next(item for item in mom["recipes"] if item["name"] == "balance")
        if not balance["decisions"]:
            raise RuntimeError("installed built-in balance Recipe has no decisions")
        builtin_help = cli("recipe", "builtin", "init", "--help")
        for option in ("--bindings", "--model-name", "--exclude-decision"):
            if option not in builtin_help:
                raise RuntimeError(
                    f"installed built-in Recipe init is missing {option}"
                )
        bundle = root / "mom-v1"
        exported = json.loads(
            cli("recipe", "builtin", "export", "mom-v1", "--output-dir", str(bundle))
        )
        if exported["sha256"] != mom["sha256"] or set(mom["files"]) != {
            path.name for path in bundle.iterdir() if path.is_file()
        }:
            raise RuntimeError(
                "installed wheel did not export its complete verified built-in bundle"
            )
        print(
            "Installed wheel supports schema, config lifecycle, route evidence, and built-in Recipe export."
        )
        if decision_images is not None:
            cli("decision", "serve", "--help")
            result = subprocess.run(
                args=[
                    str(python),
                    "-I",
                    "-c",
                    "import json; "
                    "from cli.decision_runtime.image_lock import load_decision_image_lock; "
                    "print(json.dumps(dict(load_decision_image_lock().images), sort_keys=True))",
                ],
                cwd=root,
                env=child_env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError("installed Decision image lock is unreadable")
            try:
                installed_images = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError) as error:
                raise RuntimeError(
                    "installed Decision image lock is unreadable"
                ) from error
            if installed_images != dict(decision_images):
                raise RuntimeError(
                    "installed Decision images do not match the published release"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--decision-images-json")
    args = parser.parse_args()
    images = (
        json.loads(args.decision_images_json) if args.decision_images_json else None
    )
    check_wheel(args.wheel, decision_images=images)


if __name__ == "__main__":
    main()
