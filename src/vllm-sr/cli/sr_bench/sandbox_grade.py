"""Offline code graders. This module runs only inside a restricted container."""

import json
import os
import selectors
import signal
import time
import subprocess
import sys
from pathlib import Path


def _run_script(path, env, timeout, max_log_bytes=131072):
    """Drain noisy generated code with fixed memory and terminate its group."""
    process = subprocess.Popen(
        [sys.executable, str(path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    tail = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline, total, reason = time.monotonic() + timeout, 0, None
    try:
        while selector.get_map():
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            for key, _ in selector.select(
                min(0.1, max(0, deadline - time.monotonic()))
            ):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                tail.extend(chunk)
                del tail[:-65536]
                if total > max_log_bytes:
                    reason = "output_limit"
                    break
            if reason:
                break
        if reason:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=max(0.1, deadline - time.monotonic()) if not reason else 2)
        return process.returncode == 0 and reason is None, bytes(tail), reason
    except subprocess.TimeoutExpired:
        return False, bytes(tail), "timeout"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        selector.close()
        process.stdout.close()


def grade(payload):
    if payload["benchmark"] == "livecodebench":
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation.compute_code_generation_metrics import (
            evaluate_generations_by_problem,
        )

        record = payload["record"]
        fields = CodeGenerationProblem.__dataclass_fields__
        sample = CodeGenerationProblem(
            **{k: v for k, v in record.items() if k in fields}
        ).get_evaluation_sample()
        results, details = evaluate_generations_by_problem(
            [[payload["code"]], sample, False, payload["timeout"]]
        )
        checks = results[0]
        return {
            "correct": bool(checks) and all(x is True or x == 1 for x in checks),
            "tests": checks,
            "grader_details": details,
        }
    if payload["benchmark"] == "scicode":
        row = payload["record"]
        generated = payload["generated"]
        root = Path("/upstream/src/inspect_evals/scicode")
        results = {}
        os.chdir("/artifacts")
        for index, step in enumerate(row["sub_steps"]):
            if not generated[index]:
                results[step["step_number"]] = False
                continue
            code = [
                row["required_dependencies"],
                "from test_util import are_dicts_close, cmp_tuple_or_list",
                *generated[: index + 1],
                "from process_data import process_hdf5_to_tuple",
                f"targets=process_hdf5_to_tuple({step['step_number']!r}, {len(step['test_cases'])})",
            ]
            for test_index, test in enumerate(step["test_cases"]):
                code += [
                    f"target=targets[{test_index}]",
                    test.replace(
                        "from scicode.compare.cmp import cmp_tuple_or_list", ""
                    ).replace("from scicode.compare.cmp import are_dicts_close", ""),
                ]
            path = Path("/tmp") / f"step-{index}.py"
            path.write_text("\n".join(code))
            env = {**os.environ, "PYTHONPATH": str(root)}
            passed, log, reason = _run_script(path, env, payload["timeout"])
            results[step["step_number"]] = passed
            Path(f"step-{index}.log").write_bytes(log)
            if reason:
                Path(f"step-{index}-limit.json").write_text(
                    json.dumps({"reason": reason})
                )
        # Match the pinned upstream main-problem metric: every subproblem
        # must pass. The final subproblem alone is not sufficient.
        return {
            "correct": bool(results) and all(results.values()),
            "subproblems": results,
            "main_problem": row["problem_id"],
        }
    raise ValueError("Unsupported sandbox grader")


if __name__ == "__main__":
    result = grade(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
