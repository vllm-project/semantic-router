"""Exercise the management-port startup wrapper with a real child process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from cli import container_management_listener

CLI_ROOT = Path(__file__).resolve().parents[1]
STUB_ENTRYPOINT = """\
import json
import sys
from pathlib import Path

print(json.dumps({
    "args": sys.argv[1:],
    "port": Path(sys.argv[1]).read_text(encoding="utf-8"),
}))
"""


def _management_config(port):
    return {"global": {"services": {"management_api": {"port": port}}}}


def _run_startup(config_path, snapshot_path, *arguments):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(CLI_ROOT), env.get("PYTHONPATH")) if item
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            container_management_listener._ROUTER_MANAGEMENT_STARTUP_SCRIPT,
            str(config_path),
            str(snapshot_path),
            sys.executable,
            "-c",
            STUB_ENTRYPOINT,
            str(snapshot_path),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env=env,
    )


@pytest.mark.parametrize(
    ("document", "expected_port"),
    [({}, 8080), (_management_config(0), 8080), (_management_config(9095), 9095)],
)
def test_startup_records_the_port_before_launching_router(
    tmp_path, document, expected_port
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    snapshot_path = tmp_path / "management-port"

    result = _run_startup(config_path, snapshot_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["port"] == f"{expected_port}\n"
    assert snapshot_path.read_text(encoding="utf-8") == f"{expected_port}\n"


@pytest.mark.parametrize(
    "raw_config",
    [
        "token: [private-review-token\n",
        "- private-review-token\n",
        *[
            yaml.safe_dump(_management_config(port))
            for port in (True, "9095", -1, 65536)
        ],
    ],
)
def test_invalid_startup_clears_stale_port_and_hides_config_errors(
    tmp_path, raw_config
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(raw_config, encoding="utf-8")
    snapshot_path = tmp_path / "management-port"
    snapshot_path.write_text("8080\n", encoding="utf-8")

    result = _run_startup(config_path, snapshot_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "private-review-token" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not snapshot_path.exists()


def test_startup_refreshes_the_record_when_container_restarts(tmp_path):
    config_path = tmp_path / "config.yaml"
    snapshot_path = tmp_path / "management-port"
    config_path.write_text(yaml.safe_dump(_management_config(8081)), encoding="utf-8")

    first = _run_startup(config_path, snapshot_path)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["port"] == "8081\n"

    config_path.write_text(yaml.safe_dump(_management_config(9095)), encoding="utf-8")
    assert snapshot_path.read_text(encoding="utf-8") == "8081\n"

    restarted = _run_startup(config_path, snapshot_path)
    assert restarted.returncode == 0, restarted.stderr
    assert json.loads(restarted.stdout)["port"] == "9095\n"


def test_startup_preserves_literal_paths_and_entrypoint_arguments(tmp_path):
    config_path = tmp_path / "config 'quoted'; file.yaml"
    snapshot_path = tmp_path / "management 'quoted' port"
    marker = tmp_path / "unexpected-command"
    arguments = ("a b", "'quoted'", f"$(touch {marker})", f"; touch {marker}")
    config_path.write_text(yaml.safe_dump(_management_config(9095)), encoding="utf-8")

    result = _run_startup(config_path, snapshot_path, *arguments)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "args": [str(snapshot_path), *arguments],
        "port": "9095\n",
    }
    assert not marker.exists()


def test_startup_does_not_launch_router_when_snapshot_cannot_be_saved(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_management_config(9095)), encoding="utf-8")
    snapshot_path = tmp_path / "missing-directory" / "management-port"

    result = _run_startup(config_path, snapshot_path)

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
