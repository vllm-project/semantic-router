#!/usr/bin/env python3
"""Run real EvalScope benchmarks against VSR model names.

This runner evaluates vLLM-SR model aliases, not public baseline models. Public
provider/orchestrator scores should be joined later from public_reference_scores.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by users without PyYAML.
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


DEFAULT_SUITE = Path(__file__).with_name("evalscope_suite.yaml")
DEFAULT_OUTPUT_ROOT = Path("bench/router_flow/results/evalscope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--evalscope-bin", default="evalscope")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key from suite.models. Repeatable. Defaults to every suite model.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark id from suite.benchmarks. Defaults to default_run core benchmarks.",
    )
    parser.add_argument(
        "--limit-mode",
        choices=("smoke", "formal"),
        default="smoke",
        help="Use smoke_limit or formal_limit from the suite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Override benchmark limits for all selected benchmarks.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Allow heavy benchmarks when no --benchmark filter is supplied.",
    )
    parser.add_argument(
        "--include-adapter-needed",
        action="store_true",
        help="Allow adapter_needed entries. These usually fail until a custom adapter exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip-sandbox-prepare",
        action="store_true",
        help="Skip suite-defined Docker image preparation for sandboxed benchmarks.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Pass the selected work-dir as EvalScope's prediction/review cache.",
    )
    parser.add_argument(
        "--rerun-review",
        action="store_true",
        help="When --use-cache is set, force EvalScope to rescore cached predictions.",
    )
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def load_suite(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def mode_specific_mapping(
    bench: dict[str, Any], base_key: str, limit_mode: str
) -> dict[str, Any]:
    base = bench.get(base_key) or {}
    if not isinstance(base, dict):
        raise ValueError(f"{bench.get('id')}.{base_key} must be a mapping")
    mode_key = f"{limit_mode}_{base_key}"
    override = bench.get(mode_key) or {}
    if not isinstance(override, dict):
        raise ValueError(f"{bench.get('id')}.{mode_key} must be a mapping")
    return merge_dicts(base, override)


def selected_models(suite: dict[str, Any], requested: list[str]) -> dict[str, str]:
    models = suite.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("suite.models must be a mapping")
    keys = requested or list(models)
    missing = [key for key in keys if key not in models]
    if missing:
        raise ValueError(f"unknown model key(s): {', '.join(missing)}")
    return {key: str(models[key]) for key in keys}


def selected_benchmarks(
    suite: dict[str, Any],
    requested: list[str],
    include_heavy: bool,
    include_adapter_needed: bool,
) -> list[dict[str, Any]]:
    benchmarks = suite.get("benchmarks") or []
    if not isinstance(benchmarks, list):
        raise ValueError("suite.benchmarks must be a list")

    if requested:
        by_id = {str(bench.get("id")): bench for bench in benchmarks}
        selected = [by_id[bench_id] for bench_id in requested if bench_id in by_id]
        missing = set(requested) - set(by_id)
        if missing:
            raise ValueError(f"unknown benchmark id(s): {', '.join(sorted(missing))}")
        return selected

    selected = []
    for bench in benchmarks:
        tier = str(bench.get("tier") or "core")
        if not bench.get("default_run"):
            continue
        if tier == "heavy" and not include_heavy:
            continue
        if tier == "adapter_needed" and not include_adapter_needed:
            continue
        selected.append(bench)
    return selected


def benchmark_limit(bench: dict[str, Any], limit_mode: str, override: int) -> int:
    if override > 0:
        return override
    key = "formal_limit" if limit_mode == "formal" else "smoke_limit"
    return int(bench.get(key) or 0)


def evalscope_package_root() -> Path:
    spec = importlib.util.find_spec("evalscope")
    if spec is None or spec.origin is None:
        raise ValueError("evalscope is required to resolve evalscope: sandbox paths")
    return Path(spec.origin).resolve().parent


def resolve_sandbox_context(context: str) -> Path:
    if context.startswith("evalscope:"):
        relative = context.removeprefix("evalscope:")
        return evalscope_package_root() / relative
    return Path(context)


def docker_image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def prepare_sandbox(bench: dict[str, Any], dry_run: bool) -> None:
    prepare = bench.get("sandbox_prepare")
    if not prepare:
        return
    if not isinstance(prepare, dict):
        raise ValueError(f"{bench.get('id')}.sandbox_prepare must be a mapping")
    image = str(prepare.get("image") or "").strip()
    context = str(prepare.get("context") or "").strip()
    if not image or not context:
        raise ValueError(
            f"{bench.get('id')}.sandbox_prepare requires image and context"
        )
    if not dry_run and docker_image_exists(image):
        print(f"# sandbox image already present: {image}")
        return
    try:
        context_path = resolve_sandbox_context(context)
    except ValueError:
        if dry_run:
            print(f"# sandbox prepare requires evalscope to resolve context: {context}")
            return
        raise
    if not context_path.exists():
        raise ValueError(
            f"{bench.get('id')}.sandbox_prepare context does not exist: {context_path}"
        )
    cmd = ["docker", "build", "-t", image, str(context_path)]
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


LIVE_CODE_BENCH_STDIN_PATCH = """import sys
from io import BytesIO, TextIOWrapper

# Redirect stdin. EvalScope's stock StringIO stdin does not expose .buffer,
# while many LiveCodeBench contest-style solutions use sys.stdin.buffer.read().
sys.stdin = TextIOWrapper(BytesIO(('''{test_input}''').encode()), encoding='utf-8')
"""

LIVE_CODE_BENCH_SANDBOX_HELPERS_PATCH = """

def _vllm_sr_lcb_token_close(actual_token, expected_token):
    if actual_token == expected_token:
        return True
    numeric_markers = '.eE'
    if not any(marker in actual_token + expected_token for marker in numeric_markers):
        return False
    try:
        import math

        return math.isclose(
            float(actual_token),
            float(expected_token),
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
    except Exception:
        return False


def _vllm_sr_lcb_output_close(actual_output, expected_output):
    actual_output = actual_output.strip()
    expected_output = expected_output.strip()
    if actual_output == expected_output:
        return True
    actual_lines = [line.strip() for line in actual_output.split('\\n')]
    expected_lines = [line.strip() for line in expected_output.split('\\n')]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        actual_tokens = actual_line.split()
        expected_tokens = expected_line.split()
        if len(actual_tokens) != len(expected_tokens):
            return False
        if not all(
            _vllm_sr_lcb_token_close(actual_token, expected_token)
            for actual_token, expected_token in zip(actual_tokens, expected_tokens)
        ):
            return False
    return True
"""

LIVE_CODE_BENCH_CALL_BASED_GENERATED_HELPERS_PATCH = """import ast
import math

def _vllm_sr_lcb_values_close(actual_value, expected_value):
    if actual_value == expected_value:
        return True
    if isinstance(actual_value, bool) or isinstance(expected_value, bool):
        return False
    if isinstance(actual_value, (list, tuple)) and isinstance(expected_value, (list, tuple)):
        if len(actual_value) != len(expected_value):
            return False
        return all(
            _vllm_sr_lcb_values_close(actual_item, expected_item)
            for actual_item, expected_item in zip(actual_value, expected_value)
        )
    if isinstance(actual_value, dict) and isinstance(expected_value, dict):
        if actual_value.keys() != expected_value.keys():
            return False
        return all(
            _vllm_sr_lcb_values_close(actual_value[key], expected_value[key])
            for key in actual_value
        )
    if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
        if isinstance(actual_value, int) and isinstance(expected_value, int):
            return False
        return math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
    return False
"""

LIVE_CODE_BENCH_REVIEW_TIMEOUT_EXTRA_PARAM_PATCH = """            'review_timeout': {
                'type': 'int',
                'description': 'Sandbox execution timeout in seconds for each LiveCodeBench test case.',
                'value': 6
            },
"""

LIVE_CODE_BENCH_REVIEW_TIMEOUT_INIT_PATCH = (
    "        self.review_timeout = int("
    "self.extra_params.get('review_timeout', self.review_timeout))\n"
)

TERMINAL_BENCH_TERMINUS2_KWARGS_PATCH = """            terminus2_kwargs = self.extra_params.get('terminus2_kwargs', {}) or {}
            if not isinstance(terminus2_kwargs, dict):
                raise ValueError('terminal_bench extra_params.terminus2_kwargs must be a dict')
            allowed_terminus2_kwargs = {
                'collect_rollout_details',
                'enable_summarize',
                'interleaved_thinking',
                'llm_call_kwargs',
                'llm_kwargs',
                'model_info',
                'proactive_summarization_threshold',
                'store_all_messages',
            }
            unsupported_terminus2_kwargs = sorted(set(terminus2_kwargs) - allowed_terminus2_kwargs)
            if unsupported_terminus2_kwargs:
                raise ValueError(
                    'unsupported terminal_bench extra_params.terminus2_kwargs: '
                    + ', '.join(unsupported_terminus2_kwargs)
                )
            agent_kwargs.update(terminus2_kwargs)
"""

TERMINAL_BENCH_TERMINUS2_EXTRA_PARAM_PATCH = """    'terminus2_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Harbor Terminus2 AgentConfig kwargs.',
        'value': {},
    },
"""


def apply_evalscope_patches(benchmarks: list[dict[str, Any]], dry_run: bool) -> None:
    benchmark_ids = {str(bench.get("id")) for bench in benchmarks}
    if "live_code_bench" in benchmark_ids:
        patch_live_code_bench_review_timeout(dry_run)
        patch_live_code_bench_stdin(dry_run)
        patch_live_code_bench_sandbox_scorer(dry_run)
        patch_live_code_bench_extractor(dry_run)
    if "terminal_bench_2_1" in benchmark_ids:
        patch_terminal_bench_terminus2_kwargs(dry_run)


def patch_live_code_bench_review_timeout(dry_run: bool) -> None:
    try:
        target = (
            evalscope_package_root()
            / "benchmarks/live_code_bench/live_code_bench_adapter.py"
        )
    except ValueError:
        if dry_run:
            print(
                "# EvalScope LiveCodeBench review timeout patch requires evalscope to be installed"
            )
            return
        raise
    if not target.exists():
        raise ValueError(f"LiveCodeBench EvalScope adapter not found: {target}")
    original = target.read_text()
    patched = original

    if LIVE_CODE_BENCH_REVIEW_TIMEOUT_EXTRA_PARAM_PATCH not in patched:
        old = """            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging and bypass certain safety checks.',
                'value': False
            }
"""
        new = old + "," + "\n" + LIVE_CODE_BENCH_REVIEW_TIMEOUT_EXTRA_PARAM_PATCH
        if old not in patched:
            raise ValueError(
                f"EvalScope LiveCodeBench review timeout metadata anchor not found: {target}"
            )
        patched = patched.replace(old, new)

    if LIVE_CODE_BENCH_REVIEW_TIMEOUT_INIT_PATCH not in patched:
        old = """        self.end_date = self.extra_params.get('end_date')

        self.save_metadata = False  # Don't save metadata, since they are large
"""
        new = (
            "        self.end_date = self.extra_params.get('end_date')\n"
            + LIVE_CODE_BENCH_REVIEW_TIMEOUT_INIT_PATCH
            + "\n        self.save_metadata = False  # Don't save metadata, since they are large\n"
        )
        if old not in patched:
            raise ValueError(
                f"EvalScope LiveCodeBench review timeout init anchor not found: {target}"
            )
        patched = patched.replace(old, new)

    if patched == original:
        print(
            f"# EvalScope LiveCodeBench review timeout patch already present: {target}"
        )
        return
    print(f"# patch EvalScope LiveCodeBench configurable review timeout: {target}")
    if not dry_run:
        target.write_text(patched)


def patch_live_code_bench_stdin(dry_run: bool) -> None:
    try:
        target = (
            evalscope_package_root()
            / "benchmarks/live_code_bench/sandbox_evaluate_utils.py"
        )
    except ValueError:
        if dry_run:
            print(
                "# EvalScope LiveCodeBench stdin patch requires evalscope to be installed"
            )
            return
        raise
    if not target.exists():
        raise ValueError(f"LiveCodeBench EvalScope adapter not found: {target}")
    original = target.read_text()
    if LIVE_CODE_BENCH_STDIN_PATCH in original:
        print(f"# EvalScope LiveCodeBench stdin patch already present: {target}")
        return
    old = """import sys
from io import StringIO

# Redirect stdin
sys.stdin = StringIO('''{test_input}''')
"""
    if old not in original:
        raise ValueError(
            f"EvalScope LiveCodeBench stdin patch anchor not found: {target}"
        )
    print(f"# patch EvalScope LiveCodeBench stdin buffer support: {target}")
    if not dry_run:
        target.write_text(original.replace(old, LIVE_CODE_BENCH_STDIN_PATCH))


def patch_live_code_bench_sandbox_scorer(dry_run: bool) -> None:  # noqa: C901, PLR0912
    try:
        target = (
            evalscope_package_root()
            / "benchmarks/live_code_bench/sandbox_evaluate_utils.py"
        )
    except ValueError:
        if dry_run:
            print(
                "# EvalScope LiveCodeBench sandbox scorer patch requires evalscope to be installed"
            )
            return
        raise
    if not target.exists():
        raise ValueError(f"LiveCodeBench EvalScope adapter not found: {target}")
    original = target.read_text()
    patched = original

    if LIVE_CODE_BENCH_SANDBOX_HELPERS_PATCH not in patched:
        old = "logger = get_logger()\n"
        if old not in patched:
            raise ValueError(
                f"EvalScope LiveCodeBench scorer helper patch anchor not found: {target}"
            )
        patched = patched.replace(old, old + LIVE_CODE_BENCH_SANDBOX_HELPERS_PATCH)

    if LIVE_CODE_BENCH_CALL_BASED_GENERATED_HELPERS_PATCH not in patched:
        old = """import ast

#Convert multi-type string to list with original data type
"""
        new = (
            LIVE_CODE_BENCH_CALL_BASED_GENERATED_HELPERS_PATCH
            + """
#Convert multi-type string to list with original data type
"""
        )
        if old not in patched:
            raise ValueError(
                f"EvalScope LiveCodeBench call-based helper patch anchor not found: {target}"
            )
        patched = patched.replace(old, new)

    old = """    if result == expected_output:
        print("TEST_PASSED")
    else:
        print(f"TEST_FAILED: expected {{expected_output}}, got {{result}}")
"""
    new = """    if _vllm_sr_lcb_values_close(result, expected_output):
        print("TEST_PASSED")
    else:
        print(f"TEST_FAILED: expected {{expected_output}}, got {{result}}")
"""
    if old in patched:
        patched = patched.replace(old, new)
    elif "_vllm_sr_lcb_values_close(result, expected_output)" not in patched:
        raise ValueError(
            f"EvalScope LiveCodeBench call-based comparison patch anchor not found: {target}"
        )

    old = """            if actual_output == expected_output:
                passed_count += 1
            else:
"""
    new = """            if _vllm_sr_lcb_output_close(actual_output, expected_output):
                passed_count += 1
            else:
"""
    if old in patched:
        patched = patched.replace(old, new)
    elif "_vllm_sr_lcb_output_close(actual_output, expected_output)" not in patched:
        raise ValueError(
            f"EvalScope LiveCodeBench stdio comparison patch anchor not found: {target}"
        )

    if patched == original:
        print(
            f"# EvalScope LiveCodeBench sandbox scorer patch already present: {target}"
        )
        return
    print(f"# patch EvalScope LiveCodeBench sandbox scorer alignment: {target}")
    if not dry_run:
        target.write_text(patched)


def patch_live_code_bench_extractor(dry_run: bool) -> None:
    try:
        target = (
            evalscope_package_root() / "benchmarks/live_code_bench/extract_utils.py"
        )
    except ValueError:
        if dry_run:
            print(
                "# EvalScope LiveCodeBench extractor patch requires evalscope to be installed"
            )
            return
        raise
    if not target.exists():
        raise ValueError(f"LiveCodeBench EvalScope extractor not found: {target}")
    original = target.read_text()
    patched = original
    old = """    if len(indexlines) < 2:
        return ''
    return '\\n'.join(outputlines[indexlines[-2] + 1:indexlines[-1]])
"""
    new = """    if len(indexlines) == 1:
        return '\\n'.join(outputlines[indexlines[0] + 1:]).strip()
    if len(indexlines) < 2:
        stripped_output = model_output.strip()
        code_markers = ('def ', 'class ', 'import ', 'input(', 'sys.stdin')
        if any(marker in stripped_output for marker in code_markers):
            return stripped_output
        return ''
    return '\\n'.join(outputlines[indexlines[-2] + 1:indexlines[-1]]).strip()
"""
    if old in patched:
        patched = patched.replace(old, new)
    elif "if len(indexlines) == 1:" not in patched:
        raise ValueError(
            f"EvalScope LiveCodeBench extractor patch anchor not found: {target}"
        )
    if patched == original:
        print(f"# EvalScope LiveCodeBench extractor patch already present: {target}")
        return
    print(f"# patch EvalScope LiveCodeBench code extraction fallback: {target}")
    if not dry_run:
        target.write_text(patched)


def patch_terminal_bench_terminus2_kwargs(dry_run: bool) -> None:
    try:
        target = (
            evalscope_package_root()
            / "benchmarks/terminal_bench/terminal_bench_adapter.py"
        )
    except ValueError:
        if dry_run:
            print(
                "# EvalScope TerminalBench Terminus2 kwargs patch requires evalscope to be installed"
            )
            return
        raise
    if not target.exists():
        raise ValueError(f"TerminalBench EvalScope adapter not found: {target}")
    original = target.read_text()
    patched = patch_terminal_bench_extra_param_metadata(original, target)
    if TERMINAL_BENCH_TERMINUS2_KWARGS_PATCH in patched:
        if patched != original:
            print(
                f"# patch EvalScope TerminalBench Terminus2 extra param metadata: {target}"
            )
            if not dry_run:
                target.write_text(patched)
            return
        print(
            f"# EvalScope TerminalBench Terminus2 kwargs patch already present: {target}"
        )
        return
    old = """            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })
        else:
"""
    new = (
        """            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })
"""
        + TERMINAL_BENCH_TERMINUS2_KWARGS_PATCH
        + """        else:
"""
    )
    if old in patched:
        patched = patched.replace(old, new)
    else:
        old = """            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })

        agent_config = AgentConfig(
"""
        new = (
            """            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })
"""
            + TERMINAL_BENCH_TERMINUS2_KWARGS_PATCH
            + """
        agent_config = AgentConfig(
"""
        )
        if old not in patched:
            raise ValueError(
                f"EvalScope TerminalBench Terminus2 kwargs patch anchor not found: {target}"
            )
        patched = patched.replace(old, new)
    if patched == original:
        raise ValueError(
            f"EvalScope TerminalBench Terminus2 kwargs patch anchor not found: {target}"
        )
    print(f"# patch EvalScope TerminalBench Terminus2 kwargs support: {target}")
    if not dry_run:
        target.write_text(patched)


def patch_terminal_bench_extra_param_metadata(original: str, target: Path) -> str:
    if "COMMON_EXTRA_PARAMS" not in original:
        return original
    metadata_block = original.split("def _validate_environment_requirements", 1)[0]
    if "'terminus2_kwargs'" in metadata_block:
        return original
    old = """    'environment_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Harbor EnvironmentConfig. '
        'Supported keys: override_cpus, override_memory_mb, override_storage_mb, override_gpus, '
        'force_build, delete, env, etc.',
        'value': {},
    },
}
"""
    if old not in original:
        raise ValueError(
            f"EvalScope TerminalBench extra param metadata patch anchor not found: {target}"
        )
    return original.replace(
        old,
        old.replace(
            "}\n",
            TERMINAL_BENCH_TERMINUS2_EXTRA_PARAM_PATCH + "}\n",
        ),
    )


def build_command(
    *,
    evalscope_bin: str,
    model_key: str,
    model_name: str,
    bench: dict[str, Any],
    defaults: dict[str, Any],
    api_url: str,
    api_key: str,
    output_root: Path,
    limit_mode: str,
    limit_override: int,
    use_cache: bool,
    rerun_review: bool,
) -> list[str]:
    dataset = str(bench["dataset"])
    generation_config = merge_dicts(
        defaults.get("generation_config") or {},
        mode_specific_mapping(bench, "generation_config", limit_mode),
    )
    dataset_args = mode_specific_mapping(bench, "dataset_args", limit_mode)
    judge_model_args = merge_dicts(
        defaults.get("judge_model_args") or {},
        mode_specific_mapping(bench, "judge_model_args", limit_mode),
    )
    judge_model_args = resolve_api_key_env(judge_model_args)
    judge_strategy = str(
        bench.get("judge_strategy") or defaults.get("judge_strategy") or ""
    )
    judge_worker_num = bench.get("judge_worker_num") or defaults.get("judge_worker_num")
    limit = benchmark_limit(bench, limit_mode, limit_override)
    work_dir = output_root / str(bench["id"]) / model_key

    cmd = [
        evalscope_bin,
        "eval",
        "--model",
        model_name,
        "--model-id",
        f"{model_key}_{bench['id']}",
        "--api-url",
        api_url,
        "--api-key",
        api_key,
        "--eval-type",
        str(defaults.get("eval_type") or "openai_api"),
        "--datasets",
        dataset,
        "--eval-batch-size",
        str(bench.get("eval_batch_size") or defaults.get("eval_batch_size") or 1),
        "--generation-config",
        stable_json(generation_config),
        "--work-dir",
        str(work_dir),
        "--no-timestamp",
    ]
    if use_cache:
        cmd.extend(["--use-cache", str(work_dir)])
        if rerun_review:
            cmd.append("--rerun-review")
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if dataset_args:
        cmd.extend(["--dataset-args", stable_json({dataset: dataset_args})])
    if judge_strategy:
        cmd.extend(["--judge-strategy", judge_strategy])
    if judge_model_args:
        cmd.extend(["--judge-model-args", stable_json(judge_model_args)])
    if judge_worker_num:
        cmd.extend(["--judge-worker-num", str(judge_worker_num)])
    sandbox = bench.get("sandbox")
    if sandbox:
        cmd.extend(["--sandbox", stable_json(sandbox)])
    return cmd


def resolve_api_key_env(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return value
    resolved = dict(value)
    resolved.pop("api_key_env", None)
    return resolved


def redacted_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for idx, value in enumerate(redacted):
        if (
            value == "--api-key"
            and idx + 1 < len(redacted)
            and redacted[idx + 1] != "EMPTY"
        ):
            redacted[idx + 1] = "<redacted>"
        if value == "--judge-model-args" and idx + 1 < len(redacted):
            redacted[idx + 1] = redact_json_api_key(redacted[idx + 1])
    return redacted


def redact_json_api_key(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict):
        api_key = payload.get("api_key")
        if api_key and api_key != "EMPTY":
            payload["api_key"] = "<redacted>"
        return stable_json(payload)
    return value


def write_manifest(output_root: Path, commands: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "evalscope_manifest.json"
    manifest.write_text(stable_json({"commands": commands}) + "\n")


def command_manifest_record(
    bench: dict[str, Any],
    model_key: str,
    model_name: str,
    cmd: list[str],
) -> dict[str, Any]:
    return {
        "benchmark": bench["id"],
        "public_name": bench.get("public_name"),
        "model_key": model_key,
        "model": model_name,
        "command": redacted_command(cmd),
    }


def run_evalscope_command(
    cmd: list[str],
    *,
    dry_run: bool,
    continue_on_error: bool,
) -> None:
    print(shlex.join(redacted_command(cmd)))
    if dry_run:
        return
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if continue_on_error:
            return
        raise


def main() -> int:
    args = parse_args()
    suite = load_suite(args.suite)
    defaults = suite.get("defaults") or {}
    api_url = args.api_url or str(defaults.get("api_url") or "")
    api_key = args.api_key or str(defaults.get("api_key") or "EMPTY")
    if not api_url:
        raise ValueError("api_url is required")

    models = selected_models(suite, args.model)
    benchmarks = selected_benchmarks(
        suite,
        args.benchmark,
        args.include_heavy,
        args.include_adapter_needed,
    )
    if not benchmarks:
        raise ValueError("no benchmarks selected")

    if not args.skip_sandbox_prepare:
        for bench in benchmarks:
            prepare_sandbox(bench, args.dry_run)
    apply_evalscope_patches(benchmarks, args.dry_run)

    planned: list[dict[str, Any]] = []
    for bench in benchmarks:
        for model_key, model_name in models.items():
            cmd = build_command(
                evalscope_bin=args.evalscope_bin,
                model_key=model_key,
                model_name=model_name,
                bench=bench,
                defaults=defaults,
                api_url=api_url,
                api_key=api_key,
                output_root=args.output_root,
                limit_mode=args.limit_mode,
                limit_override=args.limit,
                use_cache=args.use_cache,
                rerun_review=args.rerun_review,
            )
            planned.append(command_manifest_record(bench, model_key, model_name, cmd))
            run_evalscope_command(
                cmd,
                dry_run=args.dry_run,
                continue_on_error=args.continue_on_error,
            )
    write_manifest(args.output_root, planned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
