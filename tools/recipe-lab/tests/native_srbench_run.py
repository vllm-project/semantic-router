"""Opt-in installed CLI integration with the real sr-bench worker and local HTTP fixtures.

Put recipe-lab and the checkout-compatible vllm-sr on PATH, then run
python -m pytest tools/recipe-lab/tests/native_srbench_run.py.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CLI_ROOT = ROOT / "src/vllm-sr"
sys.path[:0] = [str(CLI_ROOT), str(CLI_ROOT / "tests")]
from cli.sr_bench.service import Server  # noqa: E402
from cli.sr_bench.store import Store  # noqa: E402
from test_sr_bench import Target, manifest, wait_run  # noqa: E402

pytest_plugins = ["test_sr_bench"]


def test_installed_candidate_against_a_saved_single(tmp_path, target, monkeypatch):
    source = ROOT / "config/config.yaml"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    active_hash = "a" * 64
    document = yaml.safe_load(source.read_bytes())

    def config_api(handler):
        if handler.path == "/api/v1/config/hash":
            data = {
                "source_config_hash": source_hash,
                "active_runtime_hash": active_hash,
                "generated_runtime_hash": active_hash,
                "activation_status": "active",
            }
        elif handler.path == "/api/v1/config":
            data = document
        else:
            handler.send_error(404)
            return
        body = json.dumps(data, default=str).encode()
        handler.send_response(200)
        handler.send_header("ETag", f'"{source_hash}"')
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(Target, "do_GET", config_api, raising=False)
    store = Store(tmp_path / "store")
    server = Server(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target.delay = 0.07
        base = manifest(target, cost_policy="capability_only")
        base["targets"][0]["request_params"] = {"reasoning_effort": "low"}
        target.ack = active_hash
        mom = {
            **base["targets"][0],
            "id": "candidate",
            "kind": "mom",
            "model": "vllm-sr/exhaustive-default",
            "preview_url": f"http://127.0.0.1:{target.server_port}/api/v1/routing/preview",
            "config_hash": active_hash,
            "capture_recipe": True,
            "max_inference_calls": 1,
        }
        base["targets"].append(
            {
                **mom,
                "id": "parent",
                "request_params": {"reasoning_effort": "high"},
            }
        )
        baseline_id = server.engine.start(base)["id"]
        assert wait_run(store, baseline_id)["status"] == "completed"
        target.delay = 0
        (store.root / "targets.json").write_text(json.dumps([mom]))
        recipe_lab = shutil.which("recipe-lab")
        assert recipe_lab, "install the Recipe Lab wheel and put its executable on PATH"
        r = subprocess.run(
            [
                recipe_lab,
                "score-candidate",
                "--workspace",
                str(tmp_path / "workspace"),
                "--candidate",
                str(source),
                "--proposer",
                "agent:integration",
                "--baseline-run",
                baseline_id,
                "--baseline-target",
                "single",
                "--candidate-target",
                "candidate",
                "--active-runtime-hash",
                active_hash,
                "--guard-case",
                "q1",
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "SR_BENCH_URL": f"http://127.0.0.1:{server.server_port}",
            },
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        receipt = json.loads(r.stdout)
        assert (
            receipt["verdict"] == "PASS" and receipt["metrics"]["common_successes"] == 1
        )
        assert receipt["candidate_sha256"] == source_hash
        assert receipt["candidate_run"] != baseline_id
        assert len(target.requests) == 3
        # sr-bench compare qualifies the compatible low-effort single. The
        # explicitly selected high-effort parent must never inherit that PASS.
        for reuse in (False, True):
            argv = list(r.args)
            argv[argv.index("--baseline-target") + 1] = "parent"
            if reuse:
                argv.extend(["--candidate-run", receipt["candidate_run"]])
            drift = subprocess.run(
                argv,
                cwd=tmp_path,
                env={
                    **os.environ,
                    "SR_BENCH_URL": f"http://127.0.0.1:{server.server_port}",
                },
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
            assert drift.returncode == 2, drift.stdout + drift.stderr
            observation = json.loads(drift.stdout)
            assert observation["verdict"] == "INCONCLUSIVE"
            assert observation["changed_request_fields"] == ["reasoning_effort"]
            assert len(target.requests) == 3 and len(store.list()) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
