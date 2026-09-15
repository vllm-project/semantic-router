#!/usr/bin/env python3
"""Exercise agent bootstrap capabilities from an installed wheel, without Docker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def check_wheel(wheel: Path) -> None:
    wheel = wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="vllm-sr-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
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

        def cli(*arguments: str) -> str:
            # Neither the working directory nor PYTHONPATH can supply CLI code
            # or missing resources from the checkout under isolated Python.
            result = subprocess.run(
                [str(python), "-I", "-m", "cli.main", *arguments],
                cwd=root,
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

        print(cli("--version").strip())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check_wheel(parser.parse_args().wheel)


if __name__ == "__main__":
    main()
