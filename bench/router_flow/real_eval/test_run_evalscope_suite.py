# ruff: noqa: PLR2004
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bench.router_flow.real_eval import run_evalscope_suite
from bench.router_flow.real_eval.run_evalscope_suite import (
    apply_evalscope_patches,
    build_command,
    load_suite,
    prepare_sandbox,
    redacted_command,
    resolve_sandbox_context,
    selected_benchmarks,
    selected_models,
)

TERMINUS2_MAX_TURNS = 200


def test_selected_benchmarks_preserves_requested_order() -> None:
    suite = {
        "benchmarks": [
            {"id": "gpqa_d", "default_run": True},
            {"id": "live_code_bench", "default_run": True},
            {"id": "openai_mrcr_32k", "default_run": True},
        ]
    }

    selected = selected_benchmarks(
        suite,
        ["openai_mrcr_32k", "gpqa_d"],
        include_heavy=False,
        include_adapter_needed=False,
    )

    assert [bench["id"] for bench in selected] == ["openai_mrcr_32k", "gpqa_d"]


def test_selected_models_defaults_to_suite_models() -> None:
    suite = {"models": {"auto": "vllm-sr/auto"}}

    assert selected_models(suite, []) == {"auto": "vllm-sr/auto"}


def test_livecode_smoke_uses_jan_apr_window_and_high_concurrency() -> None:
    suite = load_suite(Path(__file__).with_name("evalscope_suite.yaml"))
    bench = next(
        bench for bench in suite["benchmarks"] if bench["id"] == "live_code_bench"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=20,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "24"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 24
    assert generation_config["max_tokens"] == 32768
    dataset_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    extra_params = dataset_args["live_code_bench"]["extra_params"]
    assert extra_params["start_date"] == "2025-01-01"
    assert extra_params["end_date"] == "2025-04-30"
    assert extra_params["review_timeout"] == 30


def test_build_command_can_rescore_cached_predictions() -> None:
    bench = {
        "id": "live_code_bench",
        "dataset": "live_code_bench",
        "eval_batch_size": 24,
        "generation_config": {"batch_size": 24},
    }

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults={"eval_type": "openai_api"},
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="formal",
        limit_override=175,
        use_cache=True,
        rerun_review=True,
    )

    assert "--use-cache" in cmd
    assert cmd[cmd.index("--use-cache") + 1] == ("/tmp/evalscope/live_code_bench/auto")
    assert "--rerun-review" in cmd


def test_livecode_omni_config_routes_every_decision_to_loop_algorithm() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amd_auto_livecode_omni.yaml"
    config = yaml.safe_load(config_path.read_text())

    decisions = config["routing"]["decisions"]
    assert decisions
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}


def test_livecode_glm52_suite_uses_batch_24() -> None:
    suite = load_suite(Path(__file__).with_name("evalscope_suite_livecode_glm52.yaml"))
    bench = next(
        bench for bench in suite["benchmarks"] if bench["id"] == "live_code_bench"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="glm52_native",
        model_name="z-ai/glm-5.2",
        bench=bench,
        defaults=suite["defaults"],
        api_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        output_root=Path("/tmp/evalscope"),
        limit_mode="formal",
        limit_override=32,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "24"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 24
    assert generation_config["max_tokens"] == 32768
    assert generation_config["retries"] == 5
    assert generation_config["timeout"] == 1800


def test_livecode_kimi_k27_code_suite_uses_batch_48() -> None:
    suite = load_suite(
        Path(__file__).with_name("evalscope_suite_livecode_kimi_k27_code.yaml")
    )
    bench = next(
        bench for bench in suite["benchmarks"] if bench["id"] == "live_code_bench"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="kimi_k27_code_native",
        model_name="moonshotai/kimi-k2.7-code",
        bench=bench,
        defaults=suite["defaults"],
        api_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        output_root=Path("/tmp/evalscope"),
        limit_mode="formal",
        limit_override=32,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "48"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 48
    assert generation_config["max_tokens"] == 16384
    assert generation_config["retries"] == 5
    assert generation_config["timeout"] == 1800


def test_glm52_hle_suite_uses_official_sampling_and_batch_24(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    suite = load_suite(Path(__file__).with_name("evalscope_suite_hle_swe_glm52.yaml"))
    bench = next(bench for bench in suite["benchmarks"] if bench["id"] == "hle_text")

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=24,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "24"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 24
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_p"] == 0.95
    assert generation_config["max_tokens"] == 32768
    judge_model_args = json.loads(cmd[cmd.index("--judge-model-args") + 1])
    assert judge_model_args["model_id"] == "gpt55-judge"
    assert judge_model_args["api_url"] == "http://127.0.0.1:8899/v1"
    assert judge_model_args["api_key"] == "EMPTY"
    dataset_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    extra_params = dataset_args["hle"]["extra_params"]
    assert extra_params["include_multi_modal"] is False


def test_redacted_command_redacts_judge_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-unit")
    bench = {
        "id": "hle_text",
        "dataset": "hle",
        "eval_batch_size": 24,
        "generation_config": {"batch_size": 24},
        "judge_strategy": "auto",
        "judge_model_args": {
            "model_id": "openai/gpt-5.5",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    }

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults={"eval_type": "openai_api"},
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=1,
        use_cache=False,
        rerun_review=False,
    )

    raw_judge_args = json.loads(cmd[cmd.index("--judge-model-args") + 1])
    redacted = redacted_command(cmd)
    redacted_judge_args = json.loads(redacted[redacted.index("--judge-model-args") + 1])
    assert "api_key" not in raw_judge_args
    assert "api_key" not in redacted_judge_args


def test_glm52_swe_suite_uses_official_sampling_and_records_agentic_batch_limit() -> (
    None
):
    suite = load_suite(Path(__file__).with_name("evalscope_suite_hle_swe_glm52.yaml"))
    bench = next(
        bench
        for bench in suite["benchmarks"]
        if bench["id"] == "swe_bench_verified_mini_agentic"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=3,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "1"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 1
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_p"] == 1.0
    assert generation_config["max_tokens"] == 32768
    assert generation_config["stream"] is True
    assert "not safely batchable" in bench["notes"]
    assert "not the official SWE-Bench Pro row" in bench["notes"]


def test_glm52_swe_pro_suite_uses_real_pro_adapter_and_agentic_batch_limit() -> None:
    suite = load_suite(Path(__file__).with_name("evalscope_suite_hle_swe_glm52.yaml"))
    bench = next(
        bench for bench in suite["benchmarks"] if bench["id"] == "swe_bench_pro"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=1,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--datasets") + 1] == "swe_bench_pro"
    assert cmd[cmd.index("--eval-batch-size") + 1] == "1"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 1
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_p"] == 1.0
    assert generation_config["max_tokens"] == 32768
    assert generation_config["stream"] is True
    dataset_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    extra_params = dataset_args["swe_bench_pro"]["extra_params"]
    assert extra_params["action_protocol"] == "backticks"
    assert extra_params["eval_timeout"] == 3600
    assert extra_params["dockerhub_username"] == "jefzda"
    assert "Official SWE-Bench Pro row" in bench["notes"]


def test_hle_glm52_config_uses_only_glm52_loop_models() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amd_auto_hle_glm52.yaml"
    config = yaml.safe_load(config_path.read_text())

    provider_models = config["providers"]["models"]
    provider_model_by_name = {
        model["name"]: model["provider_model_id"] for model in provider_models
    }
    assert provider_model_by_name["gpt55-judge"] == "openai/gpt-5.5"

    decisions = config["routing"]["decisions"]
    assert decisions
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}
        assert {ref["model"] for ref in decision.get("modelRefs", [])} <= {
            "glm52-solver-a",
            "glm52-solver-b",
            "glm52-solver-c",
            "glm52-finalizer",
        }
        assert {
            provider_model_by_name[ref["model"]]
            for ref in decision.get("modelRefs", [])
        } == {"z-ai/glm-5.2"}
        assert all(
            ref["use_reasoning"] is True for ref in decision.get("modelRefs", [])
        )
        algorithm_config = algorithm[algorithm["type"]]
        assert algorithm_config["round_timeout_seconds"] >= 900


def test_hle_hybrid_suite_uses_competitive_label_and_batch_24(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    suite = load_suite(Path(__file__).with_name("evalscope_suite_hle_hybrid.yaml"))
    bench = next(bench for bench in suite["benchmarks"] if bench["id"] == "hle_text")

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=24,
        use_cache=False,
        rerun_review=False,
    )

    assert suite["model_labels"]["auto"] == "VSR Hybrid"
    assert cmd[cmd.index("--eval-batch-size") + 1] == "24"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 24
    assert generation_config["temperature"] == 1.0
    assert generation_config["top_p"] == 0.95
    judge_model_args = json.loads(cmd[cmd.index("--judge-model-args") + 1])
    assert judge_model_args["model_id"] == "gpt55-verifier"
    assert judge_model_args["api_url"] == "http://127.0.0.1:8899/v1"
    assert judge_model_args["api_key"] == "EMPTY"


def test_hle_hybrid_config_discloses_glm_and_closed_model_pool() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amd_auto_hle_hybrid.yaml"
    config = yaml.safe_load(config_path.read_text())

    provider_model_ids = {
        model["provider_model_id"] for model in config["providers"]["models"]
    }
    assert "z-ai/glm-5.2" in provider_model_ids
    assert {
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.8",
        "google/gemini-3.1-pro-preview",
    } <= provider_model_ids

    decisions = config["routing"]["decisions"]
    assert decisions
    closed_refs = set()
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}
        refs = {ref["model"] for ref in decision.get("modelRefs", [])}
        assert refs <= {
            "glm52-breadth-a",
            "glm52-breadth-b",
            "gpt55-verifier",
            "opus48-verifier",
            "gemini31-finalizer",
        }
        closed_refs |= refs & {
            "gpt55-verifier",
            "opus48-verifier",
            "gemini31-finalizer",
        }
        assert all(
            ref["use_reasoning"] is True for ref in decision.get("modelRefs", [])
        )
        algorithm_config = algorithm[algorithm["type"]]
        assert algorithm_config["round_timeout_seconds"] >= 900
    assert closed_refs


def test_swe_glm52_config_uses_only_glm52_loop_models() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amd_auto_swe_glm52.yaml"
    config = yaml.safe_load(config_path.read_text())

    provider_models = config["providers"]["models"]
    provider_model_by_name = {
        model["name"]: model["provider_model_id"] for model in provider_models
    }
    assert set(provider_model_by_name.values()) == {"z-ai/glm-5.2"}

    decisions = config["routing"]["decisions"]
    assert decisions
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}
        assert {ref["model"] for ref in decision.get("modelRefs", [])} <= {
            "glm52-planner",
            "glm52-patcher-a",
            "glm52-patcher-b",
            "glm52-verifier",
            "glm52-finalizer",
        }
        assert {
            provider_model_by_name[ref["model"]]
            for ref in decision.get("modelRefs", [])
        } == {"z-ai/glm-5.2"}
        assert all(
            ref["use_reasoning"] is True for ref in decision.get("modelRefs", [])
        )
        algorithm_config = algorithm[algorithm["type"]]
        assert algorithm_config["round_timeout_seconds"] >= 1200


def test_livecode_glm52_omni_config_uses_only_glm52_loop_models() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "amd_auto_livecode_glm52_omni.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    provider_models = config["providers"]["models"]
    provider_model_by_name = {
        model["name"]: model["provider_model_id"] for model in provider_models
    }
    assert set(provider_model_by_name.values()) == {"z-ai/glm-5.2"}

    decisions = config["routing"]["decisions"]
    assert decisions
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}
        assert {ref["model"] for ref in decision.get("modelRefs", [])} <= {
            "glm52-solver-a",
            "glm52-solver-b",
            "glm52-solver-c",
            "glm52-finalizer",
        }
        assert {
            provider_model_by_name[ref["model"]]
            for ref in decision.get("modelRefs", [])
        } == {"z-ai/glm-5.2"}
        algorithm_config = algorithm[algorithm["type"]]
        assert algorithm_config["round_timeout_seconds"] >= 900


def test_livecode_kimi_k27_code_omni_config_uses_only_kimi_loop_models() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "amd_auto_livecode_kimi_k27_code_omni.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    provider_models = config["providers"]["models"]
    assert {model["provider_model_id"] for model in provider_models} == {
        "moonshotai/kimi-k2.7-code"
    }

    decisions = config["routing"]["decisions"]
    assert decisions
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}
        assert {ref["model"] for ref in decision.get("modelRefs", [])} <= {
            "kimi-k27-code-solver-a",
            "kimi-k27-code-solver-b",
            "kimi-k27-code-solver-c",
            "kimi-k27-code-finalizer",
        }
        algorithm_config = algorithm[algorithm["type"]]
        assert algorithm_config["round_timeout_seconds"] >= 900


def test_scicode_suite_uses_background_and_high_concurrency() -> None:
    suite = load_suite(Path(__file__).with_name("evalscope_suite.yaml"))
    bench = next(bench for bench in suite["benchmarks"] if bench["id"] == "scicode")

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="auto",
        model_name="vllm-sr/auto",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="formal",
        limit_override=65,
        use_cache=False,
        rerun_review=False,
    )

    assert cmd[cmd.index("--eval-batch-size") + 1] == "24"
    generation_config = json.loads(cmd[cmd.index("--generation-config") + 1])
    assert generation_config["batch_size"] == 24
    assert generation_config["max_tokens"] == 32768
    assert generation_config["retries"] == 3
    dataset_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    extra_params = dataset_args["scicode"]["extra_params"]
    assert extra_params["provide_background"] is True


def test_scicode_omni_config_routes_every_decision_to_loop_algorithm() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "amd_auto_scicode_omni.yaml"
    config = yaml.safe_load(config_path.read_text())

    decisions = config["routing"]["decisions"]
    assert decisions
    assert decisions[-1]["priority"] == 1
    for decision in decisions:
        algorithm = decision.get("algorithm")
        assert algorithm, f"{decision['name']} must use a looper algorithm"
        assert algorithm["type"] in {"fusion", "remom", "workflows", "flow"}


def test_resolve_sandbox_context_supports_evalscope_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_evalscope_suite, "evalscope_package_root", lambda: Path("/opt/evalscope")
    )

    assert resolve_sandbox_context("evalscope:benchmarks/scicode/docker") == Path(
        "/opt/evalscope/benchmarks/scicode/docker"
    )


def test_prepare_sandbox_builds_missing_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    context = tmp_path / "sandbox"
    context.mkdir()
    monkeypatch.setattr(run_evalscope_suite, "docker_image_exists", lambda image: False)

    def fake_run(cmd: list[str], check: bool) -> None:
        commands.append(cmd)
        assert check is True

    monkeypatch.setattr(run_evalscope_suite.subprocess, "run", fake_run)

    prepare_sandbox(
        {
            "id": "scicode",
            "sandbox_prepare": {
                "image": "scicode-benchmark:latest",
                "context": str(context),
            },
        },
        dry_run=False,
    )

    assert commands == [
        ["docker", "build", "-t", "scicode-benchmark:latest", str(context)]
    ]


def test_prepare_sandbox_skips_existing_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_evalscope_suite, "docker_image_exists", lambda image: True)

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("docker build should not run")

    monkeypatch.setattr(run_evalscope_suite.subprocess, "run", fail_run)

    prepare_sandbox(
        {
            "id": "scicode",
            "sandbox_prepare": {
                "image": "scicode-benchmark:latest",
                "context": str(tmp_path),
            },
        },
        dry_run=False,
    )


def test_prepare_sandbox_dry_run_tolerates_missing_evalscope_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run_evalscope_suite,
        "resolve_sandbox_context",
        lambda context: (_ for _ in ()).throw(ValueError("missing evalscope")),
    )

    prepare_sandbox(
        {
            "id": "scicode",
            "sandbox_prepare": {
                "image": "scicode-benchmark:latest",
                "context": "evalscope:benchmarks/scicode/docker",
            },
        },
        dry_run=True,
    )

    assert "requires evalscope" in capsys.readouterr().out


def test_apply_evalscope_patches_adds_livecodebench_buffer_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "benchmarks/live_code_bench/sandbox_evaluate_utils.py"
    benchmark_adapter = (
        tmp_path / "benchmarks/live_code_bench/live_code_bench_adapter.py"
    )
    extractor = tmp_path / "benchmarks/live_code_bench/extract_utils.py"
    adapter.parent.mkdir(parents=True)
    benchmark_adapter.write_text(
        """
BENCHMARK = dict(
        extra_params={
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging and bypass certain safety checks.',
                'value': False
            }
        },
)

class LiveCodeBenchAdapter:
    def __init__(self):
        self.debug = self.extra_params.get('debug', False)
        self.start_date = self.extra_params.get('start_date')
        self.end_date = self.extra_params.get('end_date')

        self.save_metadata = False  # Don't save metadata, since they are large
"""
    )
    adapter.write_text(
        """
from evalscope.utils.logger import get_logger

logger = get_logger()

def _evaluate_stdio_in_sandbox():
            test_code = f\"\"\"
import sys
from io import StringIO

# Redirect stdin
sys.stdin = StringIO('''{test_input}''')

# User's code
{code}
\"\"\"

            if actual_output == expected_output:
                passed_count += 1
            else:
                pass

def _evaluate_call_based_in_sandbox():
            test_code = f\"\"\"
import json
import sys
from typing import TYPE_CHECKING, Dict, List, Tuple, Set, Sequence, Mapping
import ast

#Convert multi-type string to list with original data type
def parse_mixed_data(data_string):
    return []

try:
    if result == expected_output:
        print("TEST_PASSED")
    else:
        print(f"TEST_FAILED: expected {{expected_output}}, got {{result}}")
except Exception:
    pass
\"\"\"
"""
    )
    extractor.write_text(
        """
def extract_code_generation(model_output: str, model_type: str = 'chat'):
    outputlines = model_output.split('\\n')
    indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    if len(indexlines) < 2:
        return ''
    return '\\n'.join(outputlines[indexlines[-2] + 1:indexlines[-1]])
"""
    )
    monkeypatch.setattr(run_evalscope_suite, "evalscope_package_root", lambda: tmp_path)

    apply_evalscope_patches([{"id": "live_code_bench"}], dry_run=False)
    apply_evalscope_patches([{"id": "live_code_bench"}], dry_run=False)

    patched = adapter.read_text()
    assert "BytesIO" in patched
    assert "TextIOWrapper" in patched
    assert patched.count("sys.stdin = TextIOWrapper") == 1
    assert "_vllm_sr_lcb_output_close(actual_output, expected_output)" in patched
    assert "_vllm_sr_lcb_values_close(result, expected_output)" in patched
    assert patched.count("def _vllm_sr_lcb_token_close") == 1

    patched_extractor = extractor.read_text()
    assert "if len(indexlines) == 1:" in patched_extractor
    assert "stripped_output = model_output.strip()" in patched_extractor

    patched_benchmark_adapter = benchmark_adapter.read_text()
    assert "'review_timeout':" in patched_benchmark_adapter
    assert "self.review_timeout = int(" in patched_benchmark_adapter


def test_apply_evalscope_patches_adds_terminalbench_terminus2_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "benchmarks/terminal_bench/terminal_bench_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        """
def _on_inference(self):
        agent_kwargs = {'max_turns': self.max_turns}
        if self.agent_name == 'terminus-2':
            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })
        else:
            pass
"""
    )
    monkeypatch.setattr(run_evalscope_suite, "evalscope_package_root", lambda: tmp_path)

    apply_evalscope_patches([{"id": "terminal_bench_2_1"}], dry_run=False)
    apply_evalscope_patches([{"id": "terminal_bench_2_1"}], dry_run=False)

    patched = adapter.read_text()
    assert "terminus2_kwargs" in patched
    assert "unsupported terminal_bench extra_params.terminus2_kwargs" in patched
    assert patched.count("agent_kwargs.update(terminus2_kwargs)") == 1


def test_apply_evalscope_patches_supports_terminalbench_without_else(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "benchmarks/terminal_bench/terminal_bench_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        """
def _on_inference(self):
        agent_kwargs = {'max_turns': self.max_turns}
        if self.agent_name == 'terminus-2':
            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })

        agent_config = AgentConfig(
            kwargs=agent_kwargs,
        )
"""
    )
    monkeypatch.setattr(run_evalscope_suite, "evalscope_package_root", lambda: tmp_path)

    apply_evalscope_patches([{"id": "terminal_bench_2_1"}], dry_run=False)

    patched = adapter.read_text()
    assert "terminus2_kwargs" in patched
    assert patched.count("agent_kwargs.update(terminus2_kwargs)") == 1
    assert "agent_config = AgentConfig(" in patched


def test_apply_evalscope_patches_adds_terminalbench_extra_param_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "benchmarks/terminal_bench/terminal_bench_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        """
COMMON_EXTRA_PARAMS = {
    'environment_kwargs': {
        'type': 'dict',
        'description': 'Extra kwargs passed to Harbor EnvironmentConfig. '
        'Supported keys: override_cpus, override_memory_mb, override_storage_mb, override_gpus, '
        'force_build, delete, env, etc.',
        'value': {},
    },
}

def _on_inference(self):
        agent_kwargs = {'max_turns': self.max_turns}
        if self.agent_name == 'terminus-2':
            agent_kwargs.update({
                'parser_name': 'json',
                'enable_summarize': True,
                'proactive_summarization_threshold': 8000,
                'collect_rollout_details': False,
            })

        agent_config = AgentConfig(
            kwargs=agent_kwargs,
        )
"""
    )
    monkeypatch.setattr(run_evalscope_suite, "evalscope_package_root", lambda: tmp_path)

    apply_evalscope_patches([{"id": "terminal_bench_2_1"}], dry_run=False)

    patched = adapter.read_text()
    assert "'terminus2_kwargs': {" in patched
    assert "Extra kwargs passed to Harbor Terminus2 AgentConfig kwargs." in patched
    assert patched.count("agent_kwargs.update(terminus2_kwargs)") == 1


def test_terminalbench_suite_exposes_terminus2_kwargs() -> None:
    suite = load_suite(Path(__file__).with_name("evalscope_suite.yaml"))
    bench = next(
        bench for bench in suite["benchmarks"] if bench["id"] == "terminal_bench_2_1"
    )

    cmd = build_command(
        evalscope_bin="evalscope",
        model_key="flow",
        model_name="vllm-sr/flow",
        bench=bench,
        defaults=suite["defaults"],
        api_url="http://127.0.0.1:8899/v1",
        api_key="EMPTY",
        output_root=Path("/tmp/evalscope"),
        limit_mode="smoke",
        limit_override=1,
        use_cache=False,
        rerun_review=False,
    )

    dataset_args = json.loads(cmd[cmd.index("--dataset-args") + 1])
    extra_params = dataset_args["terminal_bench_v2_1"]["extra_params"]
    assert extra_params["agent_name"] == "terminus-2"
    assert extra_params["max_turns"] == TERMINUS2_MAX_TURNS
    assert extra_params["terminus2_kwargs"]["proactive_summarization_threshold"] == 0
