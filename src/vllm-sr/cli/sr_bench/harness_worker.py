"""Isolated adapter entrypoint; all inference goes through the journaled bridge."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


def call(messages, role="subject", extra_body=None):
    payload = json.dumps({"messages": messages, "role": role, "extra_body": extra_body or {}}).encode()
    request = urllib.request.Request(os.environ["SR_BENCH_BRIDGE"], data=payload, headers={"Authorization": "Bearer " + os.environ["SR_BENCH_BRIDGE_TOKEN"], "Content-Type": "application/json"})
    # No HTTP retry. The parent enforces deadlines/cancellation and journals calls.
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def _code(text):
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return "\n\n".join(matches) if matches else ""


def _tau3(request):
    from litellm import ModelResponse
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import get_tasks, run_single_task
    from tau2.utils import llm_utils

    def completion(model, messages, tools=None, tool_choice=None, **_kwargs):
        role = "simulator" if model == "sr-bench-simulator" else "subject"
        response = call(messages, role, {"tools": tools, "tool_choice": tool_choice})
        message = {"role": "assistant", "content": response["final"] or None}
        if response.get("tool_calls"):
            message["tool_calls"] = response["tool_calls"]
        return ModelResponse(model=model, choices=[{"index": 0, "message": message, "finish_reason": response["finish_reason"]}], usage=response.get("raw_usage") or {})

    # The upstream protocol/tools/grader stay intact. Its provider seam is
    # replaced so every call uses the common ledger and never retries.
    llm_utils.completion = completion
    llm_utils.get_response_cost = lambda _response: None
    metadata = request["case"]["metadata"]
    config = TextRunConfig(domain=metadata["domain"], llm_agent="sr-bench-subject", llm_user="sr-bench-simulator", llm_args_agent={"num_retries": 0}, llm_args_user={"num_retries": 0}, max_steps=request["config"].get("max_steps", 100), max_errors=3, num_trials=1, max_concurrency=1, seed=request["seed"], timeout=request["limits"]["max_run_seconds"])
    tasks = get_tasks(metadata["domain"], task_ids=[metadata["task_id"]])
    if len(tasks) != 1:
        raise ValueError("Frozen task ID did not resolve uniquely")
    simulation = run_single_task(config, tasks[0], seed=request["seed"], save_dir=Path(request["artifact_dir"]) / "tau3", auto_review=False)
    (Path(request["artifact_dir"]) / "trajectory.json").write_text(simulation.model_dump_json(indent=2))
    if simulation.reward_info is None:
        raise ValueError("tau3 returned no evaluation reward")
    reward = float(simulation.reward_info.reward)
    return {"answer": None, "correct": reward == 1, "score": float(reward == 1), "details": {"reward": reward, "termination_reason": str(simulation.termination_reason), "release": "1.0.1", "trials": 1, "domain": metadata["domain"]}}


def _sandbox(request, payload, name):
    directory = Path(request["artifact_dir"])
    input_path = directory / (name + "-input.json")
    output_path = directory / (name + "-result.json")
    input_path.write_text(json.dumps(payload))
    image = request["config"]["sandbox_image"]
    cli_root = Path(__file__).resolve().parents[2]
    root = Path(request["source_root"])
    args = ["docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256", "--memory=4g", "--cpus=2", "--tmpfs=/tmp:rw,nosuid,size=512m", "-e", "PYTHONPATH=/harness:/upstream:/upstream/src", "-v", f"{cli_root}:/harness:ro", "-v", f"{root}:/upstream:ro", "-v", f"{directory}:/artifacts:rw"]
    if request["case"]["benchmark"] == "scicode":
        data_path = Path(os.environ["SR_BENCH_SCICODE_TEST_DATA"])
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890":
            raise ValueError("SciCode test_data.h5 digest mismatch")
        args += ["-v", f"{data_path}:/artifacts/test_data.h5:ro"]
    args += [image, "python", "-m", "cli.sr_bench.sandbox_grade", "/artifacts/" + input_path.name, "/artifacts/" + output_path.name]
    try:
        subprocess.run(args, check=True, timeout=request["config"].get("grade_timeout_s", 300))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Sandbox grading deadline exceeded") from exc
    return json.loads(output_path.read_text())


def _lcb(request):
    response = call(request["case"]["messages"])
    code = _code(response["final"])
    if not code:
        return {"answer": response["final"], "correct": False, "score": 0., "details": {"reason": "no_python_code_block"}}
    graded = _sandbox(request, {"benchmark": "livecodebench", "record": request["case"]["metadata"]["source_record"], "code": code, "timeout": request["config"].get("test_timeout_s", 6)}, "lcb")
    return {"answer": response["final"], "correct": graded["correct"], "score": float(graded["correct"]), "details": graded}


def _scicode(request):
    from inspect_evals.scicode.prompt_templates import INITIAL_PROMPT, SUBPROBLEM_PROMPT
    row = request["case"]["metadata"]["source_record"]
    messages = [{"role": "system", "content": INITIAL_PROMPT.format(required_dependencies=row["required_dependencies"])}]
    generated = []
    for step in row["sub_steps"]:
        messages.append({"role": "user", "content": SUBPROBLEM_PROMPT.format(**step)})
        response = call(messages)
        messages.append({"role": "assistant", "content": response["final"]})
        generated.append(_code(response["final"]))
    graded = _sandbox(request, {"benchmark": "scicode", "record": row, "generated": generated, "timeout": request["config"].get("test_timeout_s", 60)}, "scicode")
    return {"answer": generated, "correct": graded["correct"], "score": float(graded["correct"]), "details": graded}


async def _terminal_job(request):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    task_path = Path(request["case"]["metadata"]["task_path"])
    from .contracts import digest
    tree = {str(f.relative_to(task_path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(task_path.rglob("*")) if f.is_file() and not f.is_symlink()}
    if digest(tree) != request["case"]["metadata"]["tree_sha256"]:
        raise ValueError("Terminal task changed after dataset preparation")
    directory = Path(request["artifact_dir"]) / "harbor"
    config = JobConfig.model_validate({"job_name": "task", "jobs_dir": str(directory), "n_attempts": 1, "n_concurrent_trials": 1, "retry": {"max_retries": 0}, "tasks": [{"path": str(task_path)}], "agents": [{"import_path": "cli.sr_bench.terminal_agent:SRBenchTerminalAgent", "model_name": "sr-bench-subject", "override_timeout_sec": request["limits"]["max_run_seconds"], "kwargs": {"max_steps": request["config"].get("max_steps", 100)}}], "environment": {"type": "docker", "delete": True, "override_gpus": 0}, "quiet": True})
    job = await Job.create(config)
    await job.run()
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
    return {"answer": None, "correct": correct, "score": float(correct), "details": {"rewards": rewards, "agent": "sr-bench-terminal-v1", "trials": 1}}


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
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
