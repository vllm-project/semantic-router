# ruff: noqa: PLR2004
from types import SimpleNamespace

from bench.router_flow.real_eval import run_amd_eval_matrix as matrix


def test_benchmark_configs_include_scicode_omni_recipe():
    assert matrix.BENCHMARK_CONFIGS["scicode"] == "amd_auto_scicode_omni.yaml"
    assert "scicode" in matrix.selected_benchmarks([])


def test_glm52_recipe_set_defaults_to_livecode_and_two_models():
    args = SimpleNamespace(recipe_set="glm52", suite="", model=[])

    assert matrix.selected_benchmarks([], "glm52") == ["live_code_bench"]
    assert matrix.suite_filename(args) == "evalscope_suite_livecode_glm52.yaml"
    assert matrix.selected_models(args) == ["auto", "glm52_native"]


def test_kimi_k27_code_recipe_set_defaults_to_livecode_and_two_models():
    args = SimpleNamespace(recipe_set="kimi_k27_code", suite="", model=[])

    assert matrix.selected_benchmarks([], "kimi_k27_code") == ["live_code_bench"]
    assert matrix.suite_filename(args) == "evalscope_suite_livecode_kimi_k27_code.yaml"
    assert matrix.selected_models(args) == ["auto", "kimi_k27_code_native"]


def test_glm52_hle_swe_recipe_set_defaults_to_hle_swe_and_auto_only():
    args = SimpleNamespace(recipe_set="glm52_hle_swe", suite="", model=[])

    assert matrix.selected_benchmarks([], "glm52_hle_swe") == [
        "hle_text",
        "swe_bench_verified_mini_agentic",
        "swe_bench_pro",
    ]
    assert matrix.suite_filename(args) == "evalscope_suite_hle_swe_glm52.yaml"
    assert matrix.selected_models(args) == ["auto"]


def test_hle_hybrid_recipe_set_defaults_to_hle_and_auto_only():
    args = SimpleNamespace(recipe_set="hle_hybrid", suite="", model=[])

    assert matrix.selected_benchmarks([], "hle_hybrid") == ["hle_text"]
    assert matrix.suite_filename(args) == "evalscope_suite_hle_hybrid.yaml"
    assert matrix.selected_models(args) == ["auto"]


def test_glm52_recipe_set_rejects_unconfigured_benchmark():
    try:
        matrix.selected_benchmarks(["scicode"], "glm52")
    except SystemExit as exc:
        assert "not configured for recipe set glm52" in str(exc)
    else:
        raise AssertionError("expected SystemExit for unconfigured glm52 benchmark")


def test_glm52_hle_swe_recipe_set_rejects_livecode():
    try:
        matrix.selected_benchmarks(["live_code_bench"], "glm52_hle_swe")
    except SystemExit as exc:
        assert "not configured for recipe set glm52_hle_swe" in str(exc)
    else:
        raise AssertionError(
            "expected SystemExit for unconfigured glm52_hle_swe benchmark"
        )


def test_hle_hybrid_recipe_set_rejects_swe():
    try:
        matrix.selected_benchmarks(["swe_bench_verified_mini_agentic"], "hle_hybrid")
    except SystemExit as exc:
        assert "not configured for recipe set hle_hybrid" in str(exc)
    else:
        raise AssertionError(
            "expected SystemExit for unconfigured hle_hybrid benchmark"
        )


def test_ssh_streams_script_over_stdin(monkeypatch):
    calls = []

    def fake_run_with_input(cmd, stdin, **kwargs):
        calls.append((cmd, stdin, kwargs))

    monkeypatch.setattr(matrix, "run_with_input", fake_run_with_input)
    args = SimpleNamespace(dry_run=False, retries=2, retry_delay_seconds=3)

    matrix.ssh(args, "root@example", "set -e\necho ok")

    assert calls == [
        (
            [
                "ssh",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
                "root@example",
                "bash",
                "-se",
            ],
            "set -e\necho ok\n",
            {"dry_run": False, "retries": 2, "retry_delay_seconds": 3},
        )
    ]


def test_router_restart_script_falls_back_to_remote_openrouter_key_file():
    args = SimpleNamespace(
        router_image="ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:test",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
    )

    script = "\n".join(matrix.router_restart_script(args, "/tmp/config.yaml"))

    assert "if docker inspect vllm-sr-router-container" in script
    assert "tr -d '\\r\\n' < /tmp/openrouter_key" in script
    assert "OPENROUTER_API_KEY is unavailable" in script
    assert "docker rm -f vllm-sr-router-container >/dev/null 2>&1 || true" in script
    assert '-e OPENROUTER_API_KEY="$openrouter_api_key"' in script
    assert (
        "router_image=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:test" in script
    )
    assert '"$router_image" /app/config.yaml >/dev/null' in script


def test_router_restart_script_recreates_when_config_mount_differs():
    args = SimpleNamespace(
        router_image="",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
    )

    script = "\n".join(matrix.router_restart_script(args, "/tmp/omni.yaml"))

    assert "current_config_mount=$(docker inspect vllm-sr-router-container" in script
    assert 'router_image="$(docker inspect vllm-sr-router-container' in script
    assert '[ "$current_config_mount" = /tmp/omni.yaml ]' in script
    assert "docker restart vllm-sr-router-container >/dev/null" in script
    assert "-v /tmp/omni.yaml:/app/config.yaml:ro" in script
    assert "exit 0" not in script


def test_router_readiness_script_polls_for_omni_model():
    args = SimpleNamespace(restart_wait_seconds=7)

    script = "\n".join(matrix.router_readiness_script(args, "vllm-sr/auto"))

    assert "for _ in $(seq 1 7)" in script
    assert "curl -fsS http://127.0.0.1:8899/v1/models 2>/dev/null || true" in script
    assert "grep -F vllm-sr/auto >/dev/null" in script
    assert 'if [ "$ready" != "1" ]; then' in script


def test_remote_openrouter_key_script_uses_container_env_then_key_file():
    args = SimpleNamespace(remote_openrouter_api_key_file="/tmp/openrouter_key")

    script = matrix.remote_openrouter_key_script(args)

    assert "docker inspect vllm-sr-router-container" in script
    assert "sed -n 's/^OPENROUTER_API_KEY=//p'" in script
    assert "tr -d '\\r\\n' < /tmp/openrouter_key" in script
    assert 'printf "%s" "$openrouter_api_key"' in script


def test_remote_envoy_config_script_discovers_envoy_yaml_mount():
    args = SimpleNamespace(remote_envoy_container="vllm-sr-envoy-container")

    script = matrix.remote_envoy_config_script(args)

    assert "docker inspect vllm-sr-envoy-container >/dev/null" in script
    assert 'eq .Destination "/etc/envoy/envoy.yaml"' in script
    assert "{{.Source}}" in script


def test_sync_envoy_config_renders_provider_backend_refs(monkeypatch, tmp_path):
    calls = []
    args = SimpleNamespace(
        dry_run=False,
        retries=1,
        retry_delay_seconds=0,
        skip_envoy_sync=False,
        remote_envoy_config="/remote/envoy.yaml",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        remote_envoy_container="vllm-sr-envoy-container",
        envoy_extproc_address="vllm-sr-router-container",
        envoy_router_api_address="vllm-sr-router-container",
    )

    monkeypatch.setattr(
        matrix, "fetch_openrouter_api_key", lambda _args, _host: "sk-unit"
    )

    def fake_rsync_to(rsync_args, src, host, dest):
        rendered = src.read_text()
        calls.append((rsync_args, host, dest, rendered))

    monkeypatch.setattr(matrix, "rsync_to", fake_rsync_to)
    monkeypatch.setattr(
        matrix,
        "ssh",
        lambda _args, host, script: calls.append(("ssh", host, script)),
    )

    matrix.sync_envoy_config(args, "root@example", "live_code_bench")

    assert len(calls) == 2
    assert calls[0] == ("ssh", "root@example", "mkdir -p /remote")
    _, host, dest, rendered = calls[1]
    assert host == "root@example"
    assert dest == "/remote/envoy.yaml"
    assert "gemini_worker_cluster" in rendered
    assert "gpt_worker_cluster" in rendered
    assert "opus_worker_cluster" in rendered
    assert 'key: "Authorization"' in rendered
    assert "Bearer sk-unit" in rendered


def test_switch_benchmark_recipe_copies_recipe_syncs_envoy_and_restarts(monkeypatch):
    calls = []
    args = SimpleNamespace(
        remote_config_dir="/remote/configs",
        remote_router_config="amd_auto_omni.yaml",
        remote_source_config_dir="sources",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        router_image="",
        restart_wait_seconds=3,
        remote_envoy_container="vllm-sr-envoy-container",
    )

    def fake_ssh(_args, host, script):
        calls.append(("ssh", host, script))

    def fake_sync_envoy_config(_args, host, benchmark):
        calls.append(("envoy", host, benchmark))

    monkeypatch.setattr(matrix, "ssh", fake_ssh)
    monkeypatch.setattr(matrix, "sync_envoy_config", fake_sync_envoy_config)

    matrix.switch_benchmark_recipe(args, "root@example", "gpqa_d")

    assert calls[0][0] == "ssh"
    assert (
        "cp /remote/configs/sources/amd_auto_gpqa_omni.yaml /remote/configs/amd_auto_omni.yaml"
        in calls[0][2]
    )
    assert calls[1] == ("envoy", "root@example", "gpqa_d")
    assert calls[2][0] == "ssh"
    assert "docker restart vllm-sr-envoy-container >/dev/null" in calls[2][2]
    assert "grep -F vllm-sr/auto >/dev/null" in calls[2][2]


def test_glm52_switch_uses_glm_only_recipe(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="glm52",
        remote_config_dir="/remote/configs",
        remote_router_config="amd_auto_omni.yaml",
        remote_source_config_dir="sources",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        router_image="",
        restart_wait_seconds=3,
        remote_envoy_container="vllm-sr-envoy-container",
    )

    monkeypatch.setattr(matrix, "ssh", lambda _args, host, script: calls.append(script))
    monkeypatch.setattr(matrix, "sync_envoy_config", lambda *_args: None)

    matrix.switch_benchmark_recipe(args, "root@example", "live_code_bench")

    assert (
        "cp /remote/configs/sources/amd_auto_livecode_glm52_omni.yaml "
        "/remote/configs/amd_auto_omni.yaml"
    ) in calls[0]


def test_kimi_k27_code_switch_uses_kimi_only_recipe(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="kimi_k27_code",
        remote_config_dir="/remote/configs",
        remote_router_config="amd_auto_omni.yaml",
        remote_source_config_dir="sources",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        router_image="",
        restart_wait_seconds=3,
        remote_envoy_container="vllm-sr-envoy-container",
    )

    monkeypatch.setattr(matrix, "ssh", lambda _args, host, script: calls.append(script))
    monkeypatch.setattr(matrix, "sync_envoy_config", lambda *_args: None)

    matrix.switch_benchmark_recipe(args, "root@example", "live_code_bench")

    assert (
        "cp /remote/configs/sources/amd_auto_livecode_kimi_k27_code_omni.yaml "
        "/remote/configs/amd_auto_omni.yaml"
    ) in calls[0]


def test_glm52_hle_swe_switch_uses_glm_only_hle_recipe(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="glm52_hle_swe",
        remote_config_dir="/remote/configs",
        remote_router_config="amd_auto_omni.yaml",
        remote_source_config_dir="sources",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        router_image="",
        restart_wait_seconds=3,
        remote_envoy_container="vllm-sr-envoy-container",
    )

    monkeypatch.setattr(matrix, "ssh", lambda _args, host, script: calls.append(script))
    monkeypatch.setattr(matrix, "sync_envoy_config", lambda *_args: None)

    matrix.switch_benchmark_recipe(args, "root@example", "hle_text")

    assert (
        "cp /remote/configs/sources/amd_auto_hle_glm52.yaml "
        "/remote/configs/amd_auto_omni.yaml"
    ) in calls[0]


def test_hle_hybrid_switch_uses_hybrid_hle_recipe(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="hle_hybrid",
        remote_config_dir="/remote/configs",
        remote_router_config="amd_auto_omni.yaml",
        remote_source_config_dir="sources",
        remote_openrouter_api_key_file="/tmp/openrouter_key",
        router_image="",
        restart_wait_seconds=3,
        remote_envoy_container="vllm-sr-envoy-container",
    )

    monkeypatch.setattr(matrix, "ssh", lambda _args, host, script: calls.append(script))
    monkeypatch.setattr(matrix, "sync_envoy_config", lambda *_args: None)

    matrix.switch_benchmark_recipe(args, "root@example", "hle_text")

    assert (
        "cp /remote/configs/sources/amd_auto_hle_hybrid.yaml "
        "/remote/configs/amd_auto_omni.yaml"
    ) in calls[0]


def test_pull_outputs_fetches_report_manifest_and_selected_benchmark(
    monkeypatch, tmp_path
):
    calls = []
    args = SimpleNamespace(
        pull=True,
        recipe_set="glm52_hle_swe",
        benchmark=["hle_text"],
        remote_eval_root="/remote/eval",
        report_dir="results/hle-report",
        output_root="results/hle-output",
        local_results_dir=tmp_path,
        dry_run=False,
        retries=1,
        retry_delay_seconds=0,
    )

    monkeypatch.setattr(
        matrix,
        "rsync_from",
        lambda _args, host, src, dest: calls.append((host, src, dest)),
    )

    matrix.pull_outputs(args, "root@example")

    assert calls == [
        (
            "root@example",
            "/remote/eval/results/hle-report/",
            tmp_path / "hle-report",
        ),
        (
            "root@example",
            "/remote/eval/results/hle-output/evalscope_manifest.json",
            tmp_path / "hle-output",
        ),
        (
            "root@example",
            "/remote/eval/results/hle-output/hle_text/",
            tmp_path / "hle-output" / "hle_text",
        ),
    ]


def test_direct_glm52_evalscope_uses_remote_openrouter_key(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="glm52",
        suite="",
        evalscope_python="/venv/bin/python",
        evalscope_bin="/venv/bin/evalscope",
        output_root="results/glm52-livecode",
        limit_mode="formal",
        limit=175,
        continue_on_error=False,
        remote_eval_root="/remote/eval",
        remote_openrouter_api_key_file="/remote/key",
    )

    monkeypatch.setattr(
        matrix,
        "ssh",
        lambda _args, host, script: calls.append((host, script)),
    )

    matrix.run_evalscope(args, "root@example", "live_code_bench", "glm52_native")

    assert calls[0][0] == "root@example"
    script = calls[0][1]
    assert "real_eval/evalscope_suite_livecode_glm52.yaml" in script
    assert "--model glm52_native" in script
    assert "--api-url https://openrouter.ai/api/v1" in script
    assert '--api-key "$openrouter_api_key"' in script
    assert "tr -d '\\r\\n' < /remote/key" in script
    assert "sk-" not in script


def test_direct_kimi_k27_code_evalscope_uses_remote_openrouter_key(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="kimi_k27_code",
        suite="",
        evalscope_python="/venv/bin/python",
        evalscope_bin="/venv/bin/evalscope",
        output_root="results/kimi-livecode",
        limit_mode="formal",
        limit=175,
        continue_on_error=False,
        remote_eval_root="/remote/eval",
        remote_openrouter_api_key_file="/remote/key",
    )

    monkeypatch.setattr(
        matrix,
        "ssh",
        lambda _args, host, script: calls.append((host, script)),
    )

    matrix.run_evalscope(
        args, "root@example", "live_code_bench", "kimi_k27_code_native"
    )

    assert calls[0][0] == "root@example"
    script = calls[0][1]
    assert "real_eval/evalscope_suite_livecode_kimi_k27_code.yaml" in script
    assert "--model kimi_k27_code_native" in script
    assert "--api-url https://openrouter.ai/api/v1" in script
    assert '--api-key "$openrouter_api_key"' in script
    assert "tr -d '\\r\\n' < /remote/key" in script
    assert "sk-" not in script


def test_router_evalscope_exports_remote_openrouter_key_for_judges(monkeypatch):
    calls = []
    args = SimpleNamespace(
        recipe_set="glm52_hle_swe",
        suite="",
        evalscope_python="/venv/bin/python",
        evalscope_bin="/venv/bin/evalscope",
        output_root="results/hle",
        limit_mode="smoke",
        limit=24,
        continue_on_error=False,
        remote_eval_root="/remote/eval",
        remote_openrouter_api_key_file="/remote/key",
    )

    monkeypatch.setattr(
        matrix,
        "ssh",
        lambda _args, host, script: calls.append((host, script)),
    )

    matrix.run_evalscope(args, "root@example", "hle_text", "auto")

    assert calls[0][0] == "root@example"
    script = calls[0][1]
    assert "real_eval/evalscope_suite_hle_swe_glm52.yaml" in script
    assert "--model auto" in script
    assert "--api-url http://127.0.0.1:8899/v1" in script
    assert 'export OPENROUTER_API_KEY="$(' in script
    assert 'export OPENAI_API_KEY="$OPENROUTER_API_KEY"' in script
    assert "tr -d '\\r\\n' < /remote/key" in script
    assert "sk-" not in script
