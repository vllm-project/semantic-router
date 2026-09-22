#!/usr/bin/env python3
"""Run compatible Kubernetes profiles with shared images and isolated state."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import threading
import uuid
from pathlib import Path

from ci_results import actual_platform, artifact_records, make_receipt
from deployment_test_results import kubernetes
from execution_batches import validate_execution_batch
from image_artifacts import IMAGE_ENV

ROOT = Path(__file__).resolve().parents[2]
MINUTE = 60
PROFILE_TIMEOUT_MINUTES = 90
STATE_VARIABLES = {
    "KUBECONFIG": "kubeconfig",
    "TMPDIR": "tmp",
    "HELM_CACHE_HOME": "helm/cache",
    "HELM_CONFIG_HOME": "helm/config",
    "HELM_DATA_HOME": "helm/data",
    "E2E_KIND_STORAGE_DIR": "storage",
    "E2E_KIND_MODELS_DIR": "models",
}


def stop_process_group(process: subprocess.Popen) -> None:
    """Also stop descendant port forwards or log followers after the parent exits."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_command(command: list[str], env: dict, log: Path, timeout: int) -> None:
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout)
        finally:
            stop_process_group(process)
            reader.join(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
        if code:
            raise subprocess.CalledProcessError(code, command)


def profile_environment(record: dict, raw: Path, state: Path, env: dict) -> dict:
    isolated = {key: str(state / path) for key, path in STATE_VARIABLES.items()}
    for key, value in isolated.items():
        if key != "KUBECONFIG":
            Path(value).mkdir(parents=True)
    return {
        **env,
        **isolated,
        "PREBUILT_RUNTIME_IMAGES": "1",
        "E2E_REPORT_DIR": str(raw),
        "E2E_BASELINE_SUITE": record.get("baseline_suite", "standard"),
        "E2E_USE_WORKSPACE_MODELS": "false",
    }


def profile_command(record: dict, cluster: str) -> list[str]:
    return [
        str(ROOT / "bin/e2e"),
        "-profile=" + record["profile"],
        "-cluster=" + cluster,
        "-baseline-suite=" + record.get("baseline_suite", "standard"),
        "-keep-cluster=false",
        "-use-existing-cluster=false",
        "-use-workspace-models=false",
        "-verbose=true",
    ]


def cleanup_cluster(cluster: str, env: dict, raw: Path) -> None:
    # Framework teardown normally removes it. This also handles setup failures,
    # crashes and timeouts before the framework registered its cleanup callback.
    with (raw / "cleanup.log").open("a") as stream:
        try:
            subprocess.run(
                [
                    "kind",
                    "export",
                    "logs",
                    str(raw / "cluster-logs"),
                    "--name",
                    cluster,
                ],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=2 * MINUTE,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            stream.write(f"Cluster log export unavailable: {error}\n")
    run_command(
        ["kind", "delete", "cluster", "--name", cluster],
        env,
        raw / "cleanup.log",
        2 * MINUTE,
    )
    remaining = subprocess.check_output(
        ["kind", "get", "clusters"],
        cwd=ROOT,
        env=env,
        text=True,
        timeout=MINUTE,
    ).splitlines()
    if cluster in remaining:
        raise ValueError(f"profile cluster remains after teardown: {cluster}")


def remove_state(state: Path) -> None:
    try:
        shutil.rmtree(state)
    except PermissionError:
        # Kind's root-owned PVC contents otherwise exhaust a shared CI worker.
        if os.environ.get("GITHUB_ACTIONS") != "true":
            raise
        subprocess.run(
            ["sudo", "-n", "rm", "-rf", "--", str(state)],
            timeout=2 * MINUTE,
            check=True,
        )


def profile_evidence(record: dict, cluster: str, raw: Path) -> dict:
    report = json.loads((raw / "test-report.json").read_text())
    if report.get("cluster_name") != cluster:
        raise ValueError("Kubernetes report belongs to a different cluster")
    evidence = kubernetes(report, record["profile"])
    evidence.update(
        runtime=record["runtime"],
        device=record["device"],
        platform=actual_platform(),
        profile=record["profile"],
        cluster_name=cluster,
    )
    return evidence


def run_profile(record: dict, raw: Path, state: Path, env: dict) -> tuple[bool, bool]:
    cluster = "ci-" + record["profile"][:35] + "-" + uuid.uuid4().hex[:8]
    isolated = profile_environment(record, raw, state, env)
    (raw / "execution.json").write_text(
        json.dumps(
            {
                "cluster": cluster,
                "state": str(state),
                "environment": {key: isolated[key] for key in STATE_VARIABLES},
            },
            indent=2,
        )
        + "\n"
    )
    passed, cleaned = False, False
    try:
        run_command(
            profile_command(record, cluster),
            isolated,
            raw / "commands.log",
            record.get("timeout_minutes", PROFILE_TIMEOUT_MINUTES) * MINUTE,
        )
        passed = True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        (raw / "failure.txt").write_text(str(error) + "\n")
        print(f"::error::{record['id']}: {error}", flush=True)
    finally:
        try:
            cleanup_cluster(cluster, isolated, raw)
            remove_state(state)
            cleaned = True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with (raw / "failure.txt").open("a") as stream:
                stream.write("Teardown failed: " + str(error) + "\n")
            print(f"::error::{record['id']} teardown: {error}", flush=True)
    return passed, cleaned


def run_batch(batch: dict, output: Path) -> bool:
    validate_execution_batch(batch, "e2e")
    output.mkdir(parents=True, exist_ok=False)
    (output / "results").mkdir()
    (output / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    env = dict(os.environ)
    state_root = Path(env.get("E2E_BATCH_STATE_ROOT", str(output / "state"))).resolve()
    state_root.mkdir(parents=True, exist_ok=False)
    artifacts = artifact_records({}, environ=env)
    failures = []
    safe_to_continue = True
    for record in batch["verifications"]:
        identity = record["id"]
        raw, state = output / "raw" / identity, state_root / identity
        raw.mkdir(parents=True)
        (raw / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"::group::{record['display_name']}", flush=True)
        try:
            if not safe_to_continue:
                raise ValueError(
                    "previous profile teardown failed; worker isolation unavailable"
                )
            if source != record["source_sha"]:
                raise ValueError("executed source SHA differs from planned source")
            for image in record["images"]:
                if not env.get(IMAGE_ENV[image][0]):
                    raise ValueError(f"required prebuilt fixture missing: {image}")
                if "image:" + image not in {row["id"] for row in artifacts}:
                    raise ValueError(f"required image receipt missing: {image}")
            passed, safe_to_continue = run_profile(record, raw, state, env)
            if not passed or not safe_to_continue:
                raise ValueError("profile execution or teardown failed")
            execution = json.loads((raw / "execution.json").read_text())
            evidence = profile_evidence(record, execution["cluster"], raw)
            evidence["artifacts"] = [
                row
                for row in artifacts
                if row["id"] in {"image:" + image for image in record["images"]}
            ]
            (raw / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            receipt = make_receipt(
                record,
                evidence,
                source_sha=source,
                execution_platform=actual_platform(),
                environ={},
            )
            (output / "results" / f"{identity}.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as error:
            failures.append(identity)
            with (raw / "failure.txt").open("a") as stream:
                stream.write(str(error) + "\n")
            print(f"::error::{identity}: {error}", flush=True)
        finally:
            print("::endgroup::", flush=True)
    summary = {
        "selected": [row["id"] for row in batch["verifications"]],
        "failed": failures,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return not failures


def cleanup_batch(output: Path) -> None:
    failures = []
    for path in sorted((output / "raw").glob("*/execution.json")):
        execution = json.loads(path.read_text())
        state = Path(execution["state"])
        if not state.exists():
            continue
        env = {**os.environ, **execution["environment"]}
        try:
            cleanup_cluster(execution["cluster"], env, path.parent)
            remove_state(state)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append(f"{execution['cluster']}: {error}")
    if failures:
        raise ValueError("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cleanup:
        cleanup_batch(args.output.resolve())
        return 0
    if not args.batch:
        parser.error("--batch is required for execution")
    return 0 if run_batch(json.loads(args.batch), args.output.resolve()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
