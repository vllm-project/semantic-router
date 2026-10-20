"""Build native RISC-V Router images with a public or existing SDK."""

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "docker.io/library/debian@sha256:c6d58429cec88e03b2fe5c3d285bf90517be50b20bc1151b1b5a436d05865a9d"


def validate_builder(description):
    if not re.search(r"(?m)^Driver:\s+docker\s*$", description):
        raise ValueError("local build-image handoffs require the buildx docker driver")


def build_commands(args):
    common = [
        "docker",
        "buildx",
        "build",
        "--progress=plain",
        "--platform",
        "linux/riscv64",
        "--load",
    ]
    if args.no_cache:
        common.append("--no-cache")
    # Older Engine-integrated BuildKit versions leave globally redeclared
    # automatic arguments empty during dynamic FROM stage resolution.
    for variable in (
        "TARGETARCH=riscv64",
        "TARGETOS=linux",
        "BUILDPLATFORM=linux/riscv64",
    ):
        common += ["--build-arg", variable]
    # Docker's predefined proxy arguments are optional invocation environment,
    # never embedded in a Dockerfile or a published image configuration.
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        common += ["--build-arg", variable]
    if args.network:
        common += ["--network", args.network]
    sdk = args.sdk_image or "semantic-router-native-sdk:" + args.tag
    commands = (
        []
        if args.sdk_image
        else [
            [
                *common,
                "--build-arg",
                "BASE_IMAGE=" + args.base_image,
                "--build-arg",
                "BUILD_JOBS=" + str(args.jobs or 2),
                "-f",
                "tools/docker/riscv64/Dockerfile.toolchain",
                "-t",
                sdk,
                ".",
            ]
        ]
    )
    if args.component == "sdk":
        return commands
    build_args = {
        "GO_IMAGE": sdk,
        "IMAGE_REGISTRY": "",
        "RUST_IMAGE": sdk,
    }
    if args.go_proxy:
        build_args["GOPROXY"] = args.go_proxy
    command = [
        *common,
        "-f",
        "tools/docker/riscv64/Dockerfile",
        "--target",
        "router-binary",
    ]
    for key, value in build_args.items():
        command += ["--build-arg", key + "=" + value]
    binary_image = "semantic-router-native-router:" + args.tag
    command += ["-t", binary_image, "."]
    runtime = [
        *common,
        "-f",
        "tools/docker/Dockerfile.attached",
        "--target",
        "vllm-sr-attached",
        "--build-arg",
        "ROUTER_BUILD=" + binary_image,
        "--build-arg",
        "IMAGE_REGISTRY=",
        "--build-arg",
        "ATTACHED_BASE_IMAGE=" + args.base_image,
        "--build-arg",
        "ASSETS_IMAGE=" + args.base_image,
        "-t",
        "semantic-router-router:" + args.tag,
        ".",
    ]
    return [*commands, command, runtime]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("sdk", "router"),
        default="router",
    )
    parser.add_argument("--base-image", default=BASE)
    parser.add_argument("--tag", default="public")
    parser.add_argument("--go-proxy")
    parser.add_argument("--network", choices=("default", "host"))
    parser.add_argument(
        "--no-cache", action="store_true", help="Rebuild all Docker layers"
    )
    parser.add_argument(
        "--sdk-image", help="Reuse an existing SDK instead of building it"
    )
    parser.add_argument(
        "--jobs", type=int, help="Go/Cargo concurrency in a new SDK (default: 2)"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plan", action="store_true", help="Print commands without running Docker"
    )
    args = parser.parse_args()
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.sdk_image:
        if args.component != "router" or args.jobs is not None:
            parser.error(
                "--sdk-image requires --component router and cannot set --jobs"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]*", args.sdk_image):
            parser.error("invalid SDK image reference")
    if not re.fullmatch(r"[a-zA-Z0-9._:/-]+@sha256:[a-f0-9]{64}", args.base_image):
        parser.error("--base-image must be pinned by digest")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", args.tag):
        parser.error("invalid image tag")
    commands = build_commands(args)
    if args.plan:
        print(json.dumps({"commands": commands}, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() != "riscv64":
        parser.error("execute on native riscv64 Linux; --plan works on other hosts")
    validate_builder(
        subprocess.check_output(["docker", "buildx", "inspect"], text=True)
    )
    output = args.output or Path(tempfile.mkdtemp(prefix="vsr-public-build-"))
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"machine": platform.machine(), "commands": commands, "steps": []}
    result_file = output / "build.json"
    if result_file.exists():
        parser.error(
            "output already contains a build receipt; choose a fresh directory"
        )
    print(f"Build records: {output}", flush=True)
    try:
        for index, command in enumerate(commands):
            log = output / f"build-{index}.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            receipt["steps"].append({"log": log.name, "exit_code": result.returncode})
            result_file.write_text(json.dumps(receipt, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"build failed; see {log}")
        image = commands[-1][commands[-1].index("-t") + 1]
        inspected = subprocess.check_output(
            ["docker", "image", "inspect", image], text=True
        )
        config = json.loads(inspected)[0]
        if config["Architecture"] != "riscv64" or config["Os"] != "linux":
            raise RuntimeError("built image does not match linux/riscv64")
        receipt["image"] = {
            "name": image,
            "id": config["Id"],
            "architecture": config["Architecture"],
        }
    finally:
        result_file.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
