"""Isolated adapter entrypoint; all inference goes through the journaled bridge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

import yaml

from .contracts import digest


def call(messages, role="subject", extra_body=None):
    payload = json.dumps(
        {"messages": messages, "role": role, "extra_body": extra_body or {}}
    ).encode()
    request = urllib.request.Request(
        os.environ["SR_BENCH_BRIDGE"],
        data=payload,
        headers={
            "Authorization": "Bearer " + os.environ["SR_BENCH_BRIDGE_TOKEN"],
            "Content-Type": "application/json",
        },
    )
    # The bridge replies only after inference finishes. Its parent owns the
    # frozen call/case deadlines and cancellation; an independent socket timeout
    # would interrupt healthy long generations without backend progress here.
    # No HTTP retry: the parent journals the single dispatch and its outcome.
    with urllib.request.urlopen(request, timeout=None) as response:
        return json.load(response)


def _code(text):
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return "\n\n".join(matches) if matches else ""


def _tau_role(model):
    return {"sr-bench-subject": "subject", "sr-bench-simulator": "simulator"}.get(
        model, "judge"
    )


def _verify_tau_task(request):
    metadata = request["case"]["metadata"]
    domain = metadata["domain"]
    if domain not in {"airline", "retail", "telecom"}:
        raise ValueError("Unsupported frozen tau3 domain")
    path = (
        Path(request["source_root"])
        / "data"
        / "tau2"
        / "domains"
        / domain
        / "tasks.json"
    )
    rows = json.loads(path.read_text())
    matches = [row for row in rows if str(row.get("id")) == str(metadata["task_id"])]
    if len(matches) != 1 or digest(matches[0]) != metadata.get("source_task_sha256"):
        raise ValueError("tau3 raw task differs from the frozen dataset")
    # Pin the upstream loader to exactly the verified source, not ambient data.
    os.environ["TAU2_DATA_DIR"] = str(Path(request["source_root"]) / "data")


def _case_timeout(request):
    return min(
        request["limits"]["case_timeout_s"], request["limits"]["max_run_seconds"]
    )


def _persist_resource(directory, kind, value):
    path = Path(directory) / "owned-resources.json"
    receipt = (
        json.loads(path.read_text())
        if path.exists()
        else {"version": 1, "containers": [], "compose_projects": []}
    )
    if value not in receipt[kind]:
        receipt[kind].append(value)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(receipt, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cleanup_owned_resources(directory):
    """Recover only resources recorded before launch, including after SIGKILL."""
    path = Path(directory) / "owned-resources.json"
    if not path.exists():
        return {"complete": True, "errors": []}
    receipt = json.loads(path.read_text())
    if receipt.get("version") != 1:
        raise ValueError("Invalid sandbox ownership receipt")
    errors = []

    def remove(kind, filter_value):
        try:
            command = ["docker", kind, "ls", "-q", "--filter", filter_value]
            if kind == "container":
                command.insert(4, "-a")
            selected = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=True
            ).stdout.split()
            # IDs are returned, never arbitrary commands or unrestricted names.
            if kind != "volume" and any(
                not re.fullmatch(r"[a-f0-9]{12,64}", item) for item in selected
            ):
                raise ValueError("Unexpected Docker resource identity")
            if kind == "volume" and any(
                not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", item)
                for item in selected
            ):
                raise ValueError("Unexpected Docker volume identity")
            if selected:
                subprocess.run(
                    [
                        "docker",
                        kind,
                        "rm",
                        *(["-f"] if kind == "container" else []),
                        *selected,
                    ],
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append({"kind": kind, "error": type(exc).__name__})

    for name in receipt.get("containers", []):
        if not re.fullmatch(r"sr-bench-[a-f0-9]{32}", name):
            raise ValueError("Invalid owned sandbox identity")
        remove("container", "name=^/" + name + "$")
    for project in receipt.get("compose_projects", []):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,200}", project):
            raise ValueError("Invalid owned Compose project identity")
        for kind in ("container", "network", "volume"):
            remove(kind, "label=com.docker.compose.project=" + project)
    result = {"complete": not errors, "errors": errors}
    (Path(directory) / "cleanup-result.json").write_text(json.dumps(result))
    return result


def _tau3(request):
    _verify_tau_task(request)
    from litellm import (  # noqa: PLC0415
        ModelResponse,
    )
    from tau2.data_model.simulation import (  # noqa: PLC0415
        TextRunConfig,
    )
    from tau2.run import (  # noqa: PLC0415 - isolated optional harness environment
        get_tasks,
        run_single_task,
    )
    from tau2.utils import (  # noqa: PLC0415
        llm_utils,
    )

    def completion(model, messages, tools=None, tool_choice=None, **_kwargs):
        role = _tau_role(model)
        response = call(messages, role, {"tools": tools, "tool_choice": tool_choice})
        message = {"role": "assistant", "content": response["final"] or None}
        if response.get("tool_calls"):
            message["tool_calls"] = response["tool_calls"]
        return ModelResponse(
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response["finish_reason"],
                }
            ],
            usage=response.get("raw_usage") or {},
        )

    # The upstream protocol/tools/grader stay intact. Its provider seam is
    # replaced so every call uses the common ledger and never retries.
    llm_utils.completion = completion
    llm_utils.get_response_cost = lambda _response: None
    metadata = request["case"]["metadata"]
    config = TextRunConfig(
        domain=metadata["domain"],
        llm_agent="sr-bench-subject",
        llm_user="sr-bench-simulator",
        llm_args_agent={"num_retries": 0},
        llm_args_user={"num_retries": 0},
        max_steps=request["config"].get("max_steps", 100),
        max_errors=3,
        num_trials=1,
        max_concurrency=1,
        seed=request["seed"],
        timeout=_case_timeout(request),
    )
    tasks = get_tasks(metadata["domain"], task_ids=[metadata["task_id"]])
    if len(tasks) != 1:
        raise ValueError("Frozen task ID did not resolve uniquely")
    simulation = run_single_task(
        config,
        tasks[0],
        seed=request["seed"],
        save_dir=Path(request["artifact_dir"]) / "tau3",
        auto_review=False,
    )
    (Path(request["artifact_dir"]) / "trajectory.json").write_text(
        simulation.model_dump_json(indent=2)
    )
    if simulation.reward_info is None:
        raise ValueError("tau3 returned no evaluation reward")
    reward = float(simulation.reward_info.reward)
    return {
        "answer": None,
        "correct": reward == 1,
        "score": float(reward == 1),
        "details": {
            "reward": reward,
            "termination_reason": str(simulation.termination_reason),
            "release": "1.0.1",
            "trials": 1,
            "domain": metadata["domain"],
        },
    }


def _sandbox(request, payload, name):
    owner_directory = Path(request["artifact_dir"])
    # Generated code cannot alter the host-only ownership/call receipts.
    directory = owner_directory / ("sandbox-" + name)
    directory.mkdir(mode=0o700)
    input_path = directory / (name + "-input.json")
    output_path = directory / (name + "-result.json")
    input_path.write_text(json.dumps(payload))
    image = request["config"]["sandbox_image"]
    cli_root = Path(__file__).resolve().parents[2]
    root = Path(request["source_root"])
    container = "sr-bench-" + uuid.uuid4().hex
    _persist_resource(owner_directory, "containers", container)
    args = [
        "docker",
        "run",
        "--name",
        container,
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        "--memory=4g",
        "--cpus=2",
        "--tmpfs=/tmp:rw,nosuid,size=512m",
        "-e",
        "PYTHONPATH=/harness:/upstream:/upstream/src",
        "-v",
        f"{cli_root}:/harness:ro",
        "-v",
        f"{root}:/upstream:ro",
        "-v",
        f"{directory}:/artifacts:rw",
    ]
    if request["case"]["benchmark"] == "scicode":
        data_path = Path(os.environ["SR_BENCH_SCICODE_TEST_DATA"])
        if (
            hashlib.sha256(data_path.read_bytes()).hexdigest()
            != "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
        ):
            raise ValueError("SciCode test_data.h5 digest mismatch")
        args += ["-v", f"{data_path}:/artifacts/test_data.h5:ro"]
    args += [
        image,
        "python",
        "-m",
        "cli.sr_bench.sandbox_grade",
        "/artifacts/" + input_path.name,
        "/artifacts/" + output_path.name,
    ]
    try:
        subprocess.run(
            args,
            check=True,
            timeout=min(
                request["config"].get("grade_timeout_s", 300), _case_timeout(request)
            ),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Sandbox grading deadline exceeded") from exc
    finally:
        if not cleanup_owned_resources(owner_directory)["complete"]:
            raise ValueError(
                "Owned sandbox cleanup failed; inspect cleanup-result.json"
            )
    return json.loads(output_path.read_text())


def _lcb(request):
    response = call(request["case"]["messages"])
    code = _code(response["final"])
    if not code:
        return {
            "answer": response["final"],
            "correct": False,
            "score": 0.0,
            "details": {"reason": "no_python_code_block"},
        }
    graded = _sandbox(
        request,
        {
            "benchmark": "livecodebench",
            "record": request["case"]["metadata"]["source_record"],
            "code": code,
            "timeout": request["config"].get("test_timeout_s", 6),
        },
        "lcb",
    )
    return {
        "answer": response["final"],
        "correct": graded["correct"],
        "score": float(graded["correct"]),
        "details": graded,
    }


def _scicode(request):
    from inspect_evals.scicode.prompt_templates import (  # noqa: PLC0415 - isolated optional harness environment
        INITIAL_PROMPT,
        SUBPROBLEM_PROMPT,
    )

    row = request["case"]["metadata"]["source_record"]
    messages = [
        {
            "role": "system",
            "content": INITIAL_PROMPT.format(
                required_dependencies=row["required_dependencies"]
            ),
        }
    ]
    generated = []
    for step in row["sub_steps"]:
        messages.append({"role": "user", "content": SUBPROBLEM_PROMPT.format(**step)})
        response = call(messages)
        messages.append({"role": "assistant", "content": response["final"]})
        generated.append(_code(response["final"]))
    graded = _sandbox(
        request,
        {
            "benchmark": "scicode",
            "record": row,
            "generated": generated,
            "timeout": request["config"].get("test_timeout_s", 60),
        },
        "scicode",
    )
    return {
        "answer": generated,
        "correct": graded["correct"],
        "score": float(graded["correct"]),
        "details": graded,
    }


async def _terminal_job(request):
    from harbor.job import Job  # noqa: PLC0415 - isolated optional harness environment
    from harbor.models.job.config import (  # noqa: PLC0415
        JobConfig,
    )

    task_path = Path(request["case"]["metadata"]["task_path"])
    tree = {
        str(f.relative_to(task_path)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(task_path.rglob("*"))
        if f.is_file() and not f.is_symlink()
    }
    if digest(tree) != request["case"]["metadata"]["tree_sha256"]:
        raise ValueError("Terminal task changed after dataset preparation")
    directory = Path(request["artifact_dir"]) / "harbor"
    config = JobConfig.model_validate(
        {
            "job_name": "task",
            "jobs_dir": str(directory),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "tasks": [{"path": str(task_path)}],
            "agents": [
                {
                    "import_path": "cli.sr_bench.terminal_agent:SRBenchTerminalAgent",
                    "model_name": "sr-bench-subject",
                    "override_timeout_sec": _case_timeout(request),
                    "kwargs": {
                        "max_steps": request["config"].get("max_steps", 100),
                        "max_log_bytes": request["limits"]["max_log_bytes"],
                    },
                }
            ],
            "environment": {"type": "docker", "delete": True, "override_gpus": 0},
            "quiet": True,
        }
    )
    from harbor.environments.docker.docker import (  # noqa: PLC0415 - isolated optional harness environment
        DockerEnvironment,
        _sanitize_docker_compose_project_name,
    )

    original_start = DockerEnvironment.start

    async def owned_start(environment, force_build):
        # All task and separate-verifier environments pass through this seam.
        project = _sanitize_docker_compose_project_name(environment.session_id)
        _persist_resource(request["artifact_dir"], "compose_projects", project)
        _freeze_terminal_image(environment, request)
        return await original_start(environment, force_build=False)

    DockerEnvironment.start = owned_start
    try:
        job = await Job.create(config)
        await asyncio.wait_for(job.run(), timeout=_case_timeout(request))
    finally:
        DockerEnvironment.start = original_start
        cleanup = await asyncio.to_thread(
            cleanup_owned_resources, request["artifact_dir"]
        )
        if not cleanup["complete"]:
            raise ValueError(
                "Owned Terminal environment cleanup failed; inspect cleanup-result.json"
            )
    results = []
    for path in directory.rglob("result.json"):
        result = json.loads(path.read_text())
        if "verifier_result" in result:
            results.append(result)
    if len(results) != 1 or results[0].get("exception_info"):
        raise ValueError("Terminal task did not produce one completed trial")
    rewards = (results[0].get("verifier_result") or {}).get("rewards")
    if not isinstance(rewards, dict) or "reward" not in rewards:
        raise ValueError("Terminal verifier did not return reward")
    correct = float(rewards["reward"]) == 1
    return {
        "answer": None,
        "correct": correct,
        "score": float(correct),
        "details": {"rewards": rewards, "agent": "sr-bench-terminal-v1", "trials": 1},
    }


def frozen_terminal_image(image, config):
    frozen = config.get("task_images", {}).get(image, image)
    if not isinstance(frozen, str) or not re.search(
        r"(?:@|^)sha256:[a-f0-9]{64}$", frozen
    ):
        raise ValueError("Terminal task requires its own frozen image digest")
    return frozen


def validate_terminal_compose(path):
    path = Path(path)
    if not path.exists():
        return
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not isinstance(
        document.get("services", {}), dict
    ):
        raise ValueError("Invalid Terminal Compose services")
    for service in document.get("services", {}).values():
        if service.get("build") or (
            service.get("image")
            and not re.search(r"@sha256:[a-f0-9]{64}$", service["image"])
        ):
            raise ValueError("Terminal Compose services require digest-pinned images")


def _freeze_terminal_image(environment, request):
    frozen = frozen_terminal_image(
        environment.task_env_config.docker_image, request["config"]
    )
    environment.task_env_config.docker_image = frozen
    environment._env_vars.prebuilt_image_name = frozen
    # The generic scientific grading image is never a Terminal environment.
    validate_terminal_compose(environment.environment_dir / "docker-compose.yaml")


def _terminate(_signum, _frame):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        raise SystemExit(
            "Harness terminated; owned resources will be cleaned"
        ) from None
    for task in asyncio.all_tasks(loop):
        task.cancel()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    signal.signal(signal.SIGTERM, _terminate)
    try:
        _main(request)
    finally:
        cleanup_owned_resources(request["artifact_dir"])


def _main(request):
    benchmark = request["case"]["benchmark"]
    if benchmark == "tau3":
        result = _tau3(request)
    elif benchmark == "livecodebench":
        result = _lcb(request)
    elif benchmark == "scicode":
        result = _scicode(request)
    elif benchmark == "terminal-bench-2.1":
        result = asyncio.run(_terminal_job(request))
    else:
        raise ValueError("Unsupported harness")
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
