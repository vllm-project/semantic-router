#!/usr/bin/env python3
"""Build and qualify the final CLI and runtime distributions before they can be published."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib
from check_cli_wheel import check_wheel
from check_runtime_wheel import check_runtime_wheel
from prepare_dev_package import prepare_version
from runtime_evidence import host_platform

ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
# Both distributions share one version; vllm-sr[runtime] pins vllm-srun to it.
DISTRIBUTIONS = ("vllm_sr", "vllm_srun")
CHECKS = [
    "catalog.compiler",
    "catalog.completeness",
    "catalog.immutable",
    "package.version",
    "package.staging",
    "package.build",
    "package.metadata",
    "package.resources",
    "package.installed",
    "package.runtime",
]
BUILD_ONLY_CHECKS = [
    "package.version",
    "package.staging",
    "package.build",
    "package.metadata",
    "package.resources",
]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def verify_resources(wheel: Path, sdist: Path) -> None:
    """Every authored snapshot and public template must survive both formats."""
    package_root = ROOT / "src/vllm-sr"
    paths = sorted((package_root / "cli/model_assets").rglob("*"))
    paths.extend(
        (package_root / "cli/templates" / name)
        for name in (
            "config.template.yaml",
            "envoy.template.yaml",
            "tools_db.json",
            "grafana.serve.ini",
            "prometheus.serve.yaml",
        )
    )
    with zipfile.ZipFile(wheel) as whl, tarfile.open(sdist) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        for path in paths:
            if not path.is_file() or path.suffix == ".pyc":
                continue
            name = path.relative_to(package_root).as_posix()
            member = archive.extractfile(prefix + "/" + name)
            if (
                member is None
                or member.read() != path.read_bytes()
                or whl.read(name) != path.read_bytes()
            ):
                raise ValueError(
                    f"Packaged resource differs from the qualified source: {name}"
                )


def verify_runtime_resources(wheel: Path, sdist: Path) -> None:
    """Every runtime module and data file (contract, goldens) survives both formats."""
    package_root = ROOT / "src/model-runtime"
    paths = sorted(
        path
        for path in (package_root / "vllm_srun").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    with zipfile.ZipFile(wheel) as whl, tarfile.open(sdist) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        for path in paths:
            name = path.relative_to(package_root).as_posix()
            member = archive.extractfile(prefix + "/" + name)
            if (
                member is None
                or member.read() != path.read_bytes()
                or whl.read(name) != path.read_bytes()
            ):
                raise ValueError(
                    f"Packaged runtime file differs from the qualified source: {name}"
                )


def distribution_files(dist: Path) -> dict[str, tuple[Path, Path]]:
    """The one wheel and source distribution of each package, by distribution."""
    found = {}
    for name in DISTRIBUTIONS:
        wheels = list(dist.glob(f"{name}-*.whl"))
        sources = list(dist.glob(f"{name}-*.tar.gz"))
        if len(wheels) != 1 or len(sources) != 1:
            raise ValueError(
                f"Expected exactly one {name} wheel and one source distribution"
            )
        found[name] = (wheels[0], sources[0])
    return found


def verify_distribution(directory: Path, mode: str, tag: str) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (manifest["source_sha"], manifest["mode"], manifest["tag"]) != (
        source,
        mode,
        tag,
    ):
        raise ValueError(
            "Package candidate differs from the qualified source or publication context"
        )
    files = manifest["files"]
    identities = sorted(
        (name.split("-")[0], name.split("-")[1].removesuffix(suffix), suffix)
        for name in files
        for suffix in DISTRIBUTION_SUFFIXES
        if name.endswith(suffix)
    )
    if len(identities) != len(files) or identities != sorted(
        (distribution, manifest["version"], suffix)
        for distribution in DISTRIBUTIONS
        for suffix in DISTRIBUTION_SUFFIXES
    ):
        raise ValueError("Incomplete final package inventory")
    if {path.name for path in directory.iterdir()} != {*files, "manifest.json"}:
        raise ValueError("Unexpected files in the final distribution")
    for name, digest in files.items():
        if (
            Path(name).name != name
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Final distribution content differs from qualification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["pr", "main", "nightly", "release"], default="pr"
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build release distributions without rerunning qualification tests.",
    )
    parser.add_argument("--verify-dist", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_dist:
        verify_distribution(output, args.mode, args.tag)
        return
    if args.build_only and args.mode != "release":
        parser.error("--build-only requires --mode release")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "runtime": "none",
        "device": "none",
        "platform": host_platform(),
        "expected_checks": BUILD_ONLY_CHECKS if args.build_only else CHECKS,
        "checks": [],
        "artifacts": [],
    }
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "mode": args.mode,
        "tag": args.tag,
    }
    if not args.build_only:
        base_sha = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                args.base_ref + "^{commit}",
            ],
            text=True,
        ).strip()

    def check(name, action):
        item = {"id": name, "status": "failed"}
        report["checks"].append(item)
        try:
            action()
            item["status"] = "passed"
        finally:
            (output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

    python = sys.executable
    if not args.build_only:
        check(
            "catalog.compiler",
            lambda: run(
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tools/catalog/tests",
                "-p",
                "test_*.py",
            ),
        )
        check(
            "catalog.completeness",
            lambda: run(
                python,
                "tools/catalog/audit_model_catalog.py",
                "--require-min-evaluations-per-model",
                "5",
            ),
        )
        check(
            "catalog.immutable",
            lambda: run(
                python,
                "tools/release/snapshot_model_catalog.py",
                "--check-published",
                "--base-ref",
                base_sha,
            ),
        )

    def version():
        project = ROOT / "src/vllm-sr"
        runtime = ROOT / "src/model-runtime"
        if args.mode in {"main", "nightly"}:
            epoch = int(
                subprocess.check_output(
                    ["git", "show", "-s", "--format=%ct", "HEAD"], text=True
                )
            )
            prepare_version(project, epoch, runtime=runtime)
        cli = tomllib.loads((project / "pyproject.toml").read_text())["project"]
        value = cli["version"]
        runtime_version = tomllib.loads((runtime / "pyproject.toml").read_text())[
            "project"
        ]["version"]
        if runtime_version != value or cli["optional-dependencies"]["runtime"] != [
            f"vllm-srun[multimodal]=={value}"
        ]:
            raise ValueError(
                "vllm-srun and vllm-sr[runtime] must carry vllm-sr's version"
            )
        if args.mode == "release":
            if args.tag != "v" + value:
                raise ValueError("Release tag does not match the final package version")
            run(
                python,
                "tools/release/snapshot_model_catalog.py",
                "--check",
                "--version",
                value,
            )
        manifest["version"] = value

    check("package.version", version)
    check(
        "package.staging",
        lambda: run(python, "tools/release/stage_model_catalog_package.py"),
    )
    run(python, "tools/release/stage_model_catalog_package.py", "--check")
    dist = output / "dist"

    def build():
        for project in ("src/vllm-sr", "src/model-runtime"):
            run(python, "-m", "build", project, "--outdir", str(dist))

    check("package.build", build)
    built = distribution_files(dist)
    wheels, sources = [built["vllm_sr"][0]], [built["vllm_sr"][1]]
    runtime_wheel, runtime_sdist = built["vllm_srun"]
    files = [path for pair in built.values() for path in pair]
    check(
        "package.metadata",
        lambda: run(python, "-m", "twine", "check", *map(str, files)),
    )

    def resources():
        verify_resources(wheels[0], sources[0])
        verify_runtime_resources(runtime_wheel, runtime_sdist)

    check("package.resources", resources)
    if not args.build_only:
        check("package.installed", lambda: check_wheel(wheels[0]))
        check("package.runtime", lambda: check_runtime_wheel(runtime_wheel, wheels[0]))
    manifest["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
