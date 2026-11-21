from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import test_sr_bench_agent_session
from cli.sr_bench import agent_session
from cli.sr_bench.service import Server
from cli.sr_bench.store import Store
from test_sr_bench_agent_session import REPO_FIXTURE, _manifest

endpoint = test_sr_bench_agent_session.endpoint


def replay(store_root: Path, target: ThreadingHTTPServer) -> dict[str, Any]:
    document = _manifest(target)
    document["cases"] = [
        agent_session.build_case(
            agent_session.parse_session(json.loads(REPO_FIXTURE.read_text()))
        )
    ]
    manifest_path = store_root / "replay.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    service = Server(address=("127.0.0.1", 0), store=Store(store_root))
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(agent_session.__file__).resolve().parents[2]),
    }
    try:
        process = subprocess.run(
            args=[
                str(Path(sys.executable).with_name("vllm-sr")),
                "benchmark",
                "--store",
                str(store_root),
                "--url",
                f"http://127.0.0.1:{service.server_port}",
                "--no-autostart",
                "run",
                "--manifest",
                str(manifest_path),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        report = json.loads(process.stdout)
        calls = service.store.calls(run_id=report["run_id"], summary=True)
        result = service.store.results(report["run_id"])[0]
        return {
            "models": [call["selected_model"] for call in calls],
            "phases": [call["fixture_phase"] for call in calls],
            "complete": [call["output_complete"] for call in calls],
            "session_count": len({call["session_id"] for call in calls}),
            "correct": result["correct"],
        }
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


@pytest.mark.skipif(not REPO_FIXTURE.is_file(), reason="repository fixture not present")
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_cli_replay_preserves_recorded_turns(
    tmp_path: Path, endpoint: ThreadingHTTPServer, finish_reason: str
) -> None:
    endpoint.finish_reason = finish_reason
    evidence = replay(store_root=tmp_path, target=endpoint)
    assert evidence == {
        "models": ["small", "large", "small"],
        "phases": ["user_turn", "tool_loop", "follow_up"],
        "complete": [finish_reason == "stop"] * 3,
        "session_count": 1,
        "correct": finish_reason == "stop",
    }
