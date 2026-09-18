"""Offline code graders. This module runs only inside a restricted container."""
import json
import os
import subprocess
import sys
from pathlib import Path


def grade(payload):
    if payload["benchmark"] == "livecodebench":
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
        record = payload["record"]
        fields = CodeGenerationProblem.__dataclass_fields__
        sample = CodeGenerationProblem(**{k: v for k, v in record.items() if k in fields}).get_evaluation_sample()
        results, details = evaluate_generations_by_problem([[payload["code"]], sample, False, payload["timeout"]])
        checks = results[0]
        return {"correct": bool(checks) and all(x is True or x == 1 for x in checks), "tests": checks, "grader_details": details}
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
            code = [row["required_dependencies"], "from test_util import are_dicts_close, cmp_tuple_or_list", *generated[:index + 1], "from process_data import process_hdf5_to_tuple", f"targets=process_hdf5_to_tuple({step['step_number']!r}, {len(step['test_cases'])})"]
            for test_index, test in enumerate(step["test_cases"]):
                code += [f"target=targets[{test_index}]", test.replace("from scicode.compare.cmp import cmp_tuple_or_list", "").replace("from scicode.compare.cmp import are_dicts_close", "")]
            path = Path("/tmp") / f"step-{index}.py"
            path.write_text("\n".join(code))
            env = {**os.environ, "PYTHONPATH": str(root)}
            try:
                completed = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, timeout=payload["timeout"])
                results[step["step_number"]] = completed.returncode == 0
                Path(f"step-{index}.log").write_bytes(completed.stdout[-65536:] + completed.stderr[-65536:])
            except subprocess.TimeoutExpired:
                results[step["step_number"]] = False
        # SciCode's final subproblem is the composed main-problem solution.
        main_key = row["sub_steps"][-1]["step_number"]
        return {"correct": results[main_key], "subproblems": results, "main_problem": main_key}
    raise ValueError("Unsupported sandbox grader")


if __name__ == "__main__":
    result = grade(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
