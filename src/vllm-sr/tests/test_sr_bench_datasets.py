"""Dataset browsing projects task inputs and never exposes grading material."""

import hashlib
import json
import threading
from pathlib import Path

import pytest
import requests
from cli.sr_bench.contracts import digest, plan
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.service import PREFIX, Server
from cli.sr_bench.sources import _write_dataset
from cli.sr_bench.store import Store


def _case(identity, benchmark="mmlu-pro", category="math"):
    return {
        "id": identity,
        "benchmark": benchmark,
        "messages": [
            {"role": "user", "content": f"Question {identity}\nA. First\nB. Second"}
        ],
        "answer": "hidden-answer-marker",
        "metadata": {"stratum": category, "reference": "hidden-reference-marker"},
    }


def _dataset(root, cases, profile="smoke", seed=7):
    sources = {
        case["benchmark"]: {
            "url": "https://huggingface.co/datasets/example/test",
            "revision": "fixture-revision",
            "revision_verification": "declared-revision-with-content-digest",
            "files": [{"name": "/private/hidden-source-path", "sha256": "a" * 64}],
            "raw": "hidden-source-marker",
        }
        for case in cases
    }
    return _write_dataset(root, cases, profile, seed, sources)


def test_detail_groups_and_filtered_pages_only_search_public_input(tmp_path):
    rows = [
        _case(f"case-{i}", category="math" if i % 2 else "history") for i in range(7)
    ]
    rows.append(_case("other", "gpqa-diamond", "physics"))
    rows[0]["messages"].extend(
        [
            {"role": "assistant", "content": "hidden-trajectory-marker"},
            {"role": "tool", "content": "hidden-credential-marker"},
        ]
    )
    rows[0]["messages"][0]["token"] = "hidden-message-secret"
    rows[0]["choices"] = ["first choice", "second choice"]
    manifest = _dataset(tmp_path, rows)
    reader = DatasetReader(tmp_path)
    detail = reader.detail(manifest["id"])
    assert detail["case_count"] == 8
    assert detail["benchmarks"][1]["count"] == 7
    assert detail["categories"] == [
        {"benchmark": "gpqa-diamond", "name": "physics", "count": 1},
        {"benchmark": "mmlu-pro", "name": "history", "count": 4},
        {"benchmark": "mmlu-pro", "name": "math", "count": 3},
    ]
    first = reader.page(
        manifest["id"], benchmark="mmlu-pro", category="history", limit="2"
    )
    assert [c["id"] for c in first["cases"]] == ["case-0", "case-2"]
    assert first["total"] == 4 and first["dataset_total"] == 8
    assert first["next_cursor"] == "2"
    second = reader.page(
        manifest["id"],
        benchmark="mmlu-pro",
        category="history",
        limit="2",
        cursor=first["next_cursor"],
    )
    assert [c["id"] for c in second["cases"]] == ["case-4", "case-6"]
    assert second["next_cursor"] is None
    assert reader.page(manifest["id"], q="FIRST CHOICE")["total"] == 1
    assert reader.page(manifest["id"], q="hidden-answer-marker")["total"] == 0
    assert "hidden-" not in json.dumps([detail, first, second])
    assert first["cases"][0]["messages"] == [
        {"role": "user", "content": rows[0]["messages"][0]["content"]}
    ]
    assert (
        json.loads(Path(manifest["path"]).read_text().split("\n")[0])["answer"]
        == "hidden-answer-marker"
    )


def test_code_and_agent_inputs_are_coherent_without_references(tmp_path, monkeypatch):
    terminal = tmp_path / "task"
    terminal.mkdir()
    (terminal / "instruction.md").write_text(
        "Trace the polygon path and save its length."
    )
    (terminal / "test.py").write_text("hidden-terminal-test")
    tree = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in terminal.iterdir()
    }
    tau_root = tmp_path / "tau"
    domain = tau_root / "data/tau2/domains/retail"
    domain.mkdir(parents=True)
    tau = {
        "id": "task-1",
        "user_scenario": {
            "persona": "A customer",
            "instructions": {
                "reason_for_call": "Return the unopened lamp",
                "known_info": "Order 123",
                "task_instructions": "Ask for a refund",
                "password": "hidden-password",
            },
        },
        "evaluation_criteria": {"actions": ["hidden-expected-action"]},
        "initial_state": "hidden-initial-state",
    }
    (domain / "tasks.json").write_text(json.dumps([tau]))
    monkeypatch.setenv("SR_BENCH_TAU3_ROOT", str(tau_root))
    sci = _case("sci", "scicode")
    sci["messages"] = []
    sci["metadata"]["source_record"] = {
        "required_dependencies": "numpy",
        "sub_steps": [
            {
                "step_description_prompt": "Integrate the smooth function.",
                "function_header": "def integrate(x):",
                "return_line": "return value",
                "solution": "hidden-code-solution",
                "test_cases": "hidden-scientific-tests",
            }
        ],
    }
    tb = _case("tb", "terminal-bench-2.1")
    tb["messages"] = []
    tb["metadata"].update(task_path=str(terminal), tree_sha256=digest(tree))
    agent = _case("agent", "tau3", "retail")
    agent["messages"] = []
    agent["metadata"].update(
        domain="retail", task_id="task-1", source_task_sha256=digest(tau)
    )
    manifest = _dataset(tmp_path / "store", [sci, tb, agent])
    reader = DatasetReader(tmp_path / "store")
    result = reader.page(manifest["id"])
    questions = [row["question"] for row in result["cases"]]
    assert "Integrate the smooth function." in questions[0]
    assert "def integrate(x):" in questions[0]
    assert questions[1] == "Trace the polygon path and save its length."
    assert "Return the unopened lamp" in questions[2]
    assert "Ask for a refund" in questions[2]
    assert "hidden-" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)
    (terminal / "test.py").write_text("changed hidden tests")
    tau["user_scenario"]["persona"] = "changed"
    (domain / "tasks.json").write_text(json.dumps([tau]))
    changed = reader.page(manifest["id"])
    assert [r["input_status"] for r in changed["cases"]] == [
        "available",
        "unavailable",
        "unavailable",
    ]
    assert changed["cases"][1]["question"] == ""


@pytest.mark.parametrize(
    "query",
    [
        {"cursor": "-1"},
        {"cursor": "9" * 1000},
        {"limit": "101"},
        {"limit": "0"},
        {"q": "x" * 201},
    ],
)
def test_page_rejects_unbounded_or_invalid_queries(tmp_path, query):
    manifest = _dataset(tmp_path, [_case("case")])
    with pytest.raises(ValueError):
        DatasetReader(tmp_path).page(manifest["id"], **query)


def test_registered_dataset_path_and_digest_are_rechecked_after_cache(tmp_path):
    manifest = _dataset(tmp_path / "store", [_case("case")])
    reader = DatasetReader(tmp_path / "store")
    reader.page(manifest["id"])
    path = Path(manifest["path"])
    original = path.read_bytes()
    path.write_bytes(original.replace(b"Question", b"Modified"))
    with pytest.raises(ValueError, match="digest"):
        reader.page(manifest["id"])
    path.write_bytes(original)
    escaped = tmp_path / "outside.jsonl"
    escaped.write_bytes(original)
    path.unlink()
    path.symlink_to(escaped)
    with pytest.raises(ValueError, match="regular file"):
        reader.page(manifest["id"])
    with pytest.raises(ValueError, match="identifier"):
        reader.page("../outside")
    with pytest.raises(KeyError):
        reader.page("f" * 64)


def test_copied_metadata_never_follows_an_external_data_path(tmp_path):
    manifest = _dataset(tmp_path / "store", [_case("case")])
    path = Path(manifest["path"]).parent / "manifest.json"
    manifest["path"] = "/private/do-not-read/secret.jsonl"
    manifest["sources"]["mmlu-pro"][
        "url"
    ] = "https://:hidden-credential@huggingface.co/datasets/example/test"
    path.write_text(json.dumps(manifest))
    reader = DatasetReader(tmp_path / "store")
    assert reader.page(manifest["id"])["total"] == 1
    assert "hidden-credential" not in json.dumps(reader.detail(manifest["id"]))
    composed = reader.compose([manifest["id"]], ["mmlu-pro"])
    assert composed["id"] == manifest["id"]
    assert Path(composed["path"]).parent == path.parent
    assert json.loads(path.read_text())["path"] == manifest["path"]


def test_dataset_directory_symlink_cannot_escape_store(tmp_path):
    root = tmp_path / "store"
    manifest = _dataset(root, [_case("case")])
    original = Path(manifest["path"]).parent
    outside = tmp_path / "outside"
    original.rename(outside)
    original.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside its registered store"):
        DatasetReader(root).detail(manifest["id"])


def test_non_scalar_provenance_and_category_cannot_expose_hidden_payloads(tmp_path):
    case = _case("case")
    case["metadata"]["stratum"] = {"answer": "hidden-stratum-answer"}
    manifest = _dataset(tmp_path, [case])
    manifest["selection"] = {"cases": [case], "credentials": "hidden-selection-secret"}
    manifest["seed"] = {"raw_source": "hidden-seed-secret"}
    Path(manifest["path"]).with_name("manifest.json").write_text(json.dumps(manifest))
    reader = DatasetReader(tmp_path)
    detail = reader.detail(manifest["id"])
    assert "selection" not in detail["provenance"]
    assert "seed" not in detail["provenance"]
    assert "hidden-" not in json.dumps(detail)
    assert reader.page(manifest["id"])["cases"][0]["category"] == "all"


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", {"answer": "hidden-answer"}),
        ("profile", {"credential": "hidden-secret"}),
        ("case_count", True),
        ("custom_subset", {}),
        ("sources", {"mmlu-pro": "hidden-source"}),
    ],
)
def test_detail_rejects_structured_header_payloads(tmp_path, field, value):
    manifest = _dataset(tmp_path, [_case("case")])
    manifest[field] = value
    Path(manifest["path"]).with_name("manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        DatasetReader(tmp_path).detail(manifest["id"])


def test_composition_rejects_cross_benchmark_duplicate_ids_without_writing(tmp_path):
    first = _dataset(tmp_path, [_case("same-id")])
    second = _dataset(tmp_path, [_case("same-id", "gpqa-diamond")])
    before = set((tmp_path / "datasets").iterdir())
    with pytest.raises(ValueError, match="duplicate case IDs"):
        DatasetReader(tmp_path).compose(
            [first["id"], second["id"]], ["mmlu-pro", "gpqa-diamond"]
        )
    assert set((tmp_path / "datasets").iterdir()) == before


def test_composition_rejects_case_level_split_mismatch(tmp_path):
    case = _case("case")
    case["metadata"]["split"] = "dev"
    manifest = _dataset(tmp_path, [case], profile="standard")
    with pytest.raises(ValueError, match="case split"):
        DatasetReader(tmp_path).compose([manifest["id"]], ["mmlu-pro"])


def test_reader_rejects_duplicate_ids_even_if_source_digest_matches(tmp_path):
    manifest = _dataset(tmp_path, [_case("same-id"), _case("same-id")])
    with pytest.raises(ValueError, match="duplicate case"):
        DatasetReader(tmp_path).page(manifest["id"])


@pytest.mark.parametrize(
    "field,value",
    [("id", "x" * 513), ("benchmark", "x" * 65), ("metadata", None), ("messages", {})],
)
def test_reader_rejects_malformed_case_headers_before_paging(tmp_path, field, value):
    case = _case("case")
    case[field] = value
    manifest = _dataset(tmp_path, [case])
    with pytest.raises(ValueError):
        DatasetReader(tmp_path).page(manifest["id"])


def test_standard_composition_requires_case_level_holdout_evidence(tmp_path):
    manifest = _dataset(tmp_path, [_case("case")], profile="standard")
    with pytest.raises(ValueError, match="case split"):
        DatasetReader(tmp_path).compose([manifest["id"]], ["mmlu-pro"])


def test_composition_preserves_exact_whole_benchmarks_and_is_idempotent(tmp_path):
    a = [_case("m1"), _case("m2", category="history")]
    b = [_case("g1", "gpqa-diamond", "physics")]
    first = _dataset(tmp_path, a)
    second = _dataset(tmp_path, b)
    bundle = _dataset(tmp_path, a + b)
    reader = DatasetReader(tmp_path)
    assert reader.compose([bundle["id"]], ["gpqa-diamond", "mmlu-pro"]) == bundle
    combined = reader.compose([first["id"], second["id"]], ["mmlu-pro", "gpqa-diamond"])
    assert combined == reader.compose(
        [bundle["id"], first["id"]], ["gpqa-diamond", "mmlu-pro"]
    )
    assert combined == reader.compose(
        [first["id"], second["id"]], ["mmlu-pro", "gpqa-diamond"]
    )
    subset = reader.compose([bundle["id"]], ["mmlu-pro"])
    assert subset["benchmarks"] == ["mmlu-pro"]
    assert subset["profile"] == "smoke" and subset["split"] == "dev"
    assert [
        json.loads(line)
        for line in Path(subset["path"]).read_text().split("\n")
        if line
    ] == a
    holdout = _dataset(tmp_path, b, profile="standard")
    conflict = _dataset(tmp_path, [_case("different")])
    for ids, benchmarks in [
        ([first["id"], holdout["id"]], ["mmlu-pro", "gpqa-diamond"]),
        ([first["id"], conflict["id"]], ["mmlu-pro"]),
        ([first["id"]], ["hle"]),
    ]:
        with pytest.raises(ValueError):
            reader.compose(ids, benchmarks)


def test_dataset_http_is_authenticated_read_only_and_compose_is_editor_only(
    tmp_path, monkeypatch
):
    store = Store(tmp_path / "store")
    manifest = _dataset(store.root, [_case("case")])
    service = Server(("127.0.0.1", 0), store, "dataset-token")
    monkeypatch.setattr(
        service.engine,
        "start",
        lambda *a, **k: pytest.fail("Dataset endpoints must never start a run"),
    )
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{service.server_port}{PREFIX}"
    headers = {
        "Authorization": "Bearer dataset-token",
        "X-SR-Bench-Actor-ID": "reader",
        "X-SR-Bench-Actor-Role": "viewer",
    }
    active, _ = store.create(
        plan(
            {
                "version": "sr-bench-1.0",
                "profile": "smoke",
                "seed": 7,
                "cost_policy": "capability_only",
                "cases": [{**_case("active"), "answer": "A"}],
                "targets": [
                    {
                        "id": "single",
                        "kind": "single",
                        "model": "fixture",
                        "base_url": "http://127.0.0.1:1/v1",
                    }
                ],
            }
        )
    )
    store.status(active["id"], "running")
    before = store.get(active["id"])
    events = store.events(active["id"])
    try:
        url = base + "/datasets/" + manifest["id"]
        assert requests.get(url, timeout=2).status_code == 403
        assert requests.get(url, headers=headers, timeout=2).json()["case_count"] == 1
        page = requests.get(url + "/cases", headers=headers, timeout=2)
        assert page.status_code == 200
        assert page.json()["cases"][0]["question"].startswith("Question case")
        assert "hidden-" not in page.text
        assert (
            requests.get(url + "/cases?q=a&q=b", headers=headers, timeout=2).status_code
            == 400
        )
        assert (
            requests.get(base + "/datasets", headers=headers, timeout=2).json()[
                "datasets"
            ][0]["id"]
            == manifest["id"]
        )
        body = {"dataset_ids": [manifest["id"]], "benchmarks": ["mmlu-pro"]}
        assert (
            requests.post(
                base + "/datasets/compose", headers=headers, json=body, timeout=2
            ).status_code
            == 403
        )
        headers["X-SR-Bench-Actor-Role"] = "editor"
        response = requests.post(
            base + "/datasets/compose", headers=headers, json=body, timeout=2
        )
        assert response.status_code == 200
        assert response.json()["dataset"]["case_count"] == 1
        assert store.get(active["id"]) == before
        assert store.events(active["id"]) == events
        assert store.calls(active["id"]) == []
    finally:
        service.shutdown()
        service.server_close()
