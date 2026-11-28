"""The cold-start orchestration script (C10) stays honest.

The script chains the generation flow (split -> observe-labels -> profile ->
bootstrap -> bank import -> run -> verify --final -> bundle) with inputs from
environment variables only. These checks cover the CLI surface, private
content and the router startup failure path.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from recipe_lab.bundle import scan_private_content
from recipe_lab.cli import build_parser

SCRIPT = Path(__file__).resolve().parents[1] / "examples/cold-start/cold-start.sh"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _parser_map() -> dict:
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("build_parser() registers no subcommands")


def test_script_syntax():
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(os.name != "posix", reason="uses a Bash executable as the router")
@pytest.mark.parametrize("occupied", [False, True])
def test_router_startup_failure_stops_before_verify(tmp_path, occupied):
    binary = tmp_path / "router"
    binary.write_text(
        "#!/usr/bin/env bash\n"
        """echo '{"msg":"router_process_failed","error":"test replay store unavailable"}'\n"""
        "exit 3\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    helpers = _script_text().split("free_port() {", 1)[1].split("# ---- phase A:", 1)[0]
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        if occupied:
            reserved.listen()
        else:
            reserved.close()
        (tmp_path / "config.yaml").write_text(
            f"listeners: [{{port: {port}}}]\n", encoding="utf-8"
        )
        proc = subprocess.run(
            [
                "bash",
                "-c",
                "free_port() {"
                + helpers
                + "\nstart_router config.yaml router.log ports",
            ],
            cwd=tmp_path,
            env={**os.environ, "PYBIN": sys.executable, "ROUTER_BIN": str(binary)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    assert proc.returncode != 0
    if occupied:
        assert "cannot start this router on listener port" in proc.stderr
        assert not (tmp_path / "router.log").exists()
    else:
        assert "test replay store unavailable" in proc.stderr


def test_script_invocations_exist():
    """Every recipe-lab call in the script names a real subcommand (S4 style).

    Quoted form catches both shell calls (`"$RL/recipe-lab" split`) and the
    prose examples inside the echo messages; the bare comment header
    ("recipe-lab cold start") is deliberately not matched.
    """
    parsers = _parser_map()
    unknown = []
    text = _script_text()
    for m in re.finditer(r"[\"']recipe-lab\s+([a-z][a-z0-9-]*)(\s+(\S+))?", text):
        name, second = m.group(1), m.group(3)
        if name not in parsers:
            unknown.append(f"recipe-lab {name}")
            continue
        sub_names = set()
        for action in parsers[name]._actions:
            if isinstance(action, argparse._SubParsersAction):
                sub_names = set(action.choices)
        if (
            second
            and re.fullmatch(r"[a-z][a-z0-9-]*", second)
            and sub_names
            and second not in sub_names
        ):
            unknown.append(f"recipe-lab {name} {second}")
    assert not unknown, "cold-start script uses nonexistent commands: " + ", ".join(
        unknown
    )


def test_script_has_no_private_content():
    """The script never embeds private paths (inputs come from env vars)."""
    text = _script_text()
    scan_private_content({"cold-start.sh": text}, [])
    # beyond the bundle's own patterns: no Windows drive-letter path at all
    scan_private_content({"cold-start.sh": text}, [r"[A-Za-z]:\\"])


def test_script_covers_y1_model_eval_chain():
    """Y1: the default model_eval driver observes the frozen split too (its
    verify rewrites the task categories from those C2 labels), pins the
    answer-plan fixture and the validate binary, ranks only bootstrap
    candidates and serves the winner from `ledger top` byte-for-byte."""
    text = _script_text()
    # Y1 gate: frozen_test is observed through the grouping source as well
    assert 'OBSERVE_SPLITS="calib dev frozen_test"' in text
    # fixture inputs and validator override of the model_eval driver
    for token in (
        "DRIVER",
        "ANSWER_PLAN",
        "CLASSIFICATION_CONFIG",
        "VALIDATE_BIN",
        "VERIFY_REPETITIONS",
    ):
        assert token in text
    # Both generators return candidate paths separately from their inputs.
    assert 'c["candidate"] for c in json.load(open(sys.argv[1]))["candidates"]' in text
    # The config that verify grades is the same file the router serves.
    for token in (
        "ledger top",
        "to-model-eval",
        'start_router "$BUNDLE_CONFIG"',
        "--candidate-management-endpoint",
        '--candidate-config "$BUNDLE_CONFIG"',
    ):
        assert token in text


def test_script_covers_template_mode():
    """The default MODE=template path (sub-issue-8 §2.2) walks
    observe-lanes -> profile --lanes -> bindings -> verify --lanes ->
    bundle --bindings, and the legacy path keeps its explicit
    --legacy-domain-template gate."""
    text = _script_text()
    for token in (
        'MODE="${MODE:-template}"',
        "observe-lanes",
        "--lanes",
        '"$RL/recipe-lab" bindings',
        "--exclude-decision",
        "BASELINE_CONFIG",
        "BUNDLE_BINDINGS",
        '--bindings "$BUNDLE_BINDINGS"',
    ):
        assert token in text, f"template path token missing: {token}"
    # the legacy path stays gated behind the explicit flag
    assert "bootstrap --legacy-domain-template" in text
    assert "MODE=legacy" in text
