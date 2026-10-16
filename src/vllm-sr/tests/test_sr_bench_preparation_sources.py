"""A source-backed dataset stays readable after its download worker exits."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from cli.sr_bench import preparation_runtime as runtime
from cli.sr_bench import service, setup
from cli.sr_bench.datasets import DatasetReader


@pytest.mark.parametrize("explicit_home", [False, True])
def test_tau3_worker_and_reader_share_task_sources(
    tmp_path, monkeypatch, explicit_home
):
    store = tmp_path / "store"
    store.mkdir()
    source_home = tmp_path / "source-home"
    monkeypatch.delenv("SR_BENCH_TAU3_ROOT", raising=False)
    if explicit_home:
        monkeypatch.setenv("SR_BENCH_HOME", str(source_home))
    else:
        monkeypatch.delenv("SR_BENCH_HOME", raising=False)
        # Isolate the embedded owner's default without changing the user's HOME.
        monkeypatch.setattr(setup, "home", lambda: source_home)
    original = subprocess.Popen
    environments = []
    code = """
import json, sys
from pathlib import Path
from cli.sr_bench.setup import harness_paths
from cli.sr_bench.source_records import normalize_records
from cli.sr_bench.sources import _write_dataset
store, state = Path(sys.argv[1]), Path(sys.argv[2])
source, _ = harness_paths('tau3')
path = source / 'data/tau2/domains/retail/tasks.json'
path.parent.mkdir(parents=True)
row = {'id': 'synthetic-1', 'user_scenario': {'instructions': 'Find my synthetic order.'}}
path.write_text(json.dumps([row]))
cases = normalize_records('tau3', [{**row, 'domain': 'retail'}], 20260918)
dataset = _write_dataset(store, cases, 'smoke', 20260918,
    {'tau3': {'revision': 'synthetic-v1', 'url': 'local-import'}}, True)
state.write_text(json.dumps({'phase': 'completed', 'dataset': dataset}))
"""

    def spawn(command, **kwargs):
        environments.append(kwargs["env"])
        return original([sys.executable, "-c", code, *command[-3:]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    dataset = runtime.execute(
        {"benchmark": "tau3", "profile": "smoke"},
        store,
        lambda phase: None,
        threading.Event(),
    )
    page = DatasetReader(store).page(dataset["id"])
    assert page["cases"][0]["input_status"] == "available"
    assert "Find my synthetic order." in page["cases"][0]["question"]
    assert environments[0]["SR_BENCH_HOME"] == str(source_home)
    assert os.environ.get("SR_BENCH_HOME") == (
        str(source_home) if explicit_home else None
    )


@pytest.mark.parametrize("explicit_home", [False, True])
def test_service_startup_chooses_shared_home_and_preserves_explicit_value(
    tmp_path, monkeypatch, explicit_home
):
    store = tmp_path / "store"
    configured = str(tmp_path / "operator-sources")
    if explicit_home:
        monkeypatch.setenv("SR_BENCH_HOME", configured)
    else:
        monkeypatch.delenv("SR_BENCH_HOME", raising=False)
    observed = []

    class Owner:
        def __init__(self, address, store, token, identity):
            observed.append(setup.home())
            store.db.close()
            self.preparations = self.engine = self
            self.threads = {}

        def serve_forever(self, **kwargs):
            pass

        def shutdown(self):
            pass

        def close(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(service, "Server", Owner)
    monkeypatch.setattr(
        service, "service_credentials", lambda: ("SR_BENCH_TOKEN", None)
    )
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)
    service.serve(store=store, port=0)
    expected = (
        Path(configured) if explicit_home else store / "preparation-runtime" / "sources"
    )
    assert observed == [expected]
    assert setup.home() == expected
    assert json.loads((store / "service.json").read_text())["version"] == "sr-bench-1.0"
