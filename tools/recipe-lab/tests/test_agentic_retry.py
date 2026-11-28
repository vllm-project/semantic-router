import json

import pytest
from recipe_lab.agentic_retry import retry_reason
from recipe_lab.cli import main
from tests import test_agentic_driver as driver_fixtures

pytest_plugins = ["tests.test_agentic_driver"]


@pytest.mark.parametrize("model,expected", [("silent", 2), ("normal", 1)])
def test_retry_normal_cli_and_resume(
    tmp_path, mock_backend, monkeypatch, model, expected
):
    out = tmp_path / "run"
    inst = "local__example"
    sha = driver_fixtures._prepare_task_repo(tmp_path / "seed", inst)
    seed = tmp_path / "seed" / "tasks" / inst
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "instance_id": inst,
                "repo": "local/test",
                "base_commit": sha,
                "problem_statement": "Fix it.",
            }
        )
    )
    policy = tmp_path / "policy.json"
    monkeypatch.setenv("RETRY_SECRET", "test-retry-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unrelated-auth-token")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Authorization: test-only")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    policy.write_text(
        json.dumps(
            {
                "target": mock_backend,
                "model": "strong",
                "api_key_env": "RETRY_SECRET",
                "trigger": "client_error_or_empty_patch",
            }
        )
    )
    silent_client = """
assert not ({"RETRY_SECRET", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_USE_BEDROCK"} & os.environ.keys())
assert os.environ["ANTHROPIC_API_KEY"] == ("test-retry-key" if os.environ["ANTHROPIC_MODEL"] == "strong" else "test-primary-key")
if os.environ['ANTHROPIC_MODEL'] == 'silent':
    print(json.dumps({'total_cost_usd': 0.02, 'usage': {'output_tokens': 2}}))
    raise SystemExit(0)
"""
    monkeypatch.setattr(
        driver_fixtures,
        "FAKE_CLAUDE",
        driver_fixtures.FAKE_CLAUDE.replace(
            'tracked = Path("file.txt")', silent_client + '\ntracked = Path("file.txt")'
        ),
    )
    args = [
        "run-agentic",
        "--tasks",
        str(tasks),
        "--out",
        str(out),
        "--target",
        mock_backend,
        "--retry-policy",
        str(policy),
        "--model",
        model,
        "--api-key",
        "test-primary-key",
        "--clone-url-template",
        str(seed),
        "--claude-bin",
        driver_fixtures._fake_claude_launcher(tmp_path),
    ]
    assert main(args) == 0
    path = out / "results" / (inst + ".json")
    result = json.loads(path.read_text())
    assert result["retry"]["attempt_count"] == expected
    assert result["claude"]["total_cost_usd"] == pytest.approx(
        0.03 if expected == 2 else 0.01
    )
    assert result["claude"]["usage"]["output_tokens"] == (6 if expected == 2 else 4)
    assert "fixed" in result["model_patch"]
    journal = (out / "requests" / (inst + ".jsonl")).read_text().splitlines()
    assert len({json.loads(line)["request_id"] for line in journal}) == 2 * expected
    assert main(args) == 0
    assert json.loads(path.read_text()) == result
    if model == "normal":
        leg = out / "attempts" / "primary"
        path.write_text('{"partial":')
        (leg / "results" / (inst + ".json")).write_text('{"instance_id":')
        (leg / "tasks" / inst / "stale.txt").write_text("interrupted edit")
        assert main(args) == 0
        recovered = json.loads(path.read_text())
        assert recovered["exit_code"] == 0
        assert recovered["model_patch"] == result["model_patch"]
        assert not (leg / "tasks" / inst / "stale.txt").exists()
    policy.write_text(policy.read_text() + " ")
    assert main(args) == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--env", " anthropic_base_url =http://example.invalid"],
        ["--env", " ANTHROPIC_CUSTOM_HEADERS=Authorization: test-only"],
        ["--target", "http://example.invalid/path?key=test-secret"],
        ["--clone-url-template", "https://user:test-secret@example.invalid/{repo}.git"],
    ],
)
def test_retry_rejects_identity_overrides_before_archiving(tmp_path, extra):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"instance_id": "task", "repo": "local/test"}))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {"target": "http://localhost:1", "trigger": "client_error_or_empty_patch"}
        )
    )
    out = tmp_path / "out"
    assert (
        main(
            [
                "run-agentic",
                "--tasks",
                str(tasks),
                "--out",
                str(out),
                "--target",
                "http://localhost:2",
                "--retry-policy",
                str(policy),
                *extra,
            ]
        )
        == 2
    )
    assert not (out / "retry-manifest.json").exists()
    assert not (out / "attempts").exists()


@pytest.mark.parametrize("ids", [["../escape"], ["a", "a"], [r"..\escape"]])
def test_rejects_non_unique_or_escaping_task_ids(tmp_path, ids):
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps({"instance_id": inst}) + "\n" for inst in ids))
    out = tmp_path / "run"
    assert (
        main(
            [
                "run-agentic",
                "--tasks",
                str(path),
                "--out",
                str(out),
                "--target",
                "http://127.0.0.1",
            ]
        )
        == 2
    )
    assert not out.exists()


def test_retry_trigger_ignores_official_answer():
    assert (
        retry_reason(
            {"status": "done", "exit_code": 0, "model_patch": "patch", "correct": False}
        )
        is None
    )
    assert (
        retry_reason({"status": "done", "exit_code": 1, "model_patch": "patch"})
        == "client_error"
    )
