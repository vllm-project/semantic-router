"""Synthetic history selection across the CLI, immutable datasets and frozen plans."""

import copy
import csv
import json
import sqlite3
from pathlib import Path

import pytest
from cli.commands.benchmark import benchmark
from cli.sr_bench import history_exclusions, sources
from cli.sr_bench.contracts import (
    digest,
    load_document,
    plan,
    plan_digest,
    resolve_dataset,
)
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.history_exclusions import compile_snapshot
from cli.sr_bench.history_snapshot import load_snapshot, save_snapshot
from cli.sr_bench.report import make_report
from cli.sr_bench.store import Store, run_summary
from cli.sr_bench.task_identity import native_task_identity, source_identity, task_key
from click.testing import CliRunner


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setitem(sources.COUNTS, "mmlu-pro", (1, 2, 4))
    path = tmp_path / "native.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": i,
                    "question": f"Synthetic {i}",
                    "options": ["x", "y"],
                    "answer": "A",
                    "category": str(i % 2),
                }
                for i in range(12)
            ]
        )
    )
    return {
        "benchmark": "mmlu-pro",
        "store": tmp_path / "store",
        "source_path": path,
        "revision": "fixture-v1",
        "source_partition": "upstream-test",
    }


def cases(manifest):
    return load_document(manifest["path"])


def write_snapshot(native, tmp_path, *manifests, run_ids=()):
    snapshot = compile_snapshot(
        native["store"], dataset_ids=[m["id"] for m in manifests], run_ids=run_ids
    )
    path = tmp_path / "history.json"
    save_snapshot(snapshot, path)
    return snapshot, path


def test_exact_union_once_deterministic_and_immutable(native, tmp_path):
    quick = sources.prepare_dataset(profile="quick", **native)
    other = sources.prepare_dataset(profile="smoke", seed=31, **native)
    before = {m["path"]: Path(m["path"]).read_bytes() for m in [quick, other]}
    snapshot, path = write_snapshot(native, tmp_path, quick, other)
    duplicate = compile_snapshot(
        native["store"], dataset_ids=[other["id"], quick["id"], quick["id"]]
    )
    assert snapshot == duplicate
    result = sources.prepare_dataset(
        profile="standard", exclusion_snapshot=path, **native
    )
    entry = result["preparation"]["mmlu-pro"]
    raw, source = sources._local_source("mmlu-pro", native["source_path"], "fixture-v1")
    identity = source_identity(source, "upstream-test")
    ordered = sources._stratified_order(
        sources.normalize_records("mmlu-pro", raw, 20260918, task_source=identity),
        20260918,
    )
    excluded = set(snapshot["families"]["mmlu-pro"]["task_keys"]) | {
        task_key(c, identity) for c in ordered[:2]
    }
    expected = [c["id"] for c in ordered if task_key(c, identity) not in excluded][:4]
    assert [c["id"] for c in cases(result)] == expected
    assert entry["selected_count"] == 4
    assert entry["excluded_count"] == len(excluded)
    assert entry["evaluation_role"] == "holdout"
    assert result == sources.prepare_dataset(
        profile="standard", exclusion_snapshot=path, **native
    )
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    assert (
        DatasetReader(native["store"]).detail(result["id"])["provenance"]["preparation"]
        == result["preparation"]
    )


def test_failed_run_reserves_all_planned_cases_without_status_or_outcomes(
    native, tmp_path, monkeypatch
):
    prior = sources.prepare_dataset(profile="quick", **native)
    frozen = {"dataset": prior, "cases": cases(prior)}
    frozen["case_sha256"] = digest(frozen["cases"])
    frozen["plan_sha256"] = plan_digest(frozen)
    run_id = "run-" + "a" * 20
    db_path = Path(native["store"]) / "journal.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE runs(id TEXT,manifest TEXT,status TEXT)")
        db.execute(
            "INSERT INTO runs VALUES(?,?,?)", (run_id, json.dumps(frozen), "failed")
        )
    before = db_path.read_bytes()
    snapshot = compile_snapshot(native["store"], run_ids=[run_id])
    assert len(snapshot["families"]["mmlu-pro"]["task_keys"]) == 2
    assert snapshot["coverage"] == "named-memberships-only"
    assert db_path.read_bytes() == before
    assert set(snapshot["references"][0]) == {
        "kind",
        "id",
        "manifest_sha256",
        "case_sha256",
    }
    assert "Synthetic" not in json.dumps(snapshot)
    monkeypatch.setattr(history_exclusions, "MAX_SCAN_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds its read limit"):
        compile_snapshot(native["store"], run_ids=[run_id])
    assert db_path.read_bytes() == before


def test_exhaustion_publishes_nothing_and_retest_is_explicit(native, tmp_path):
    # Eight native tasks leave only two after the original Quick and old Standard.
    native["source_path"].write_text(
        json.dumps(json.loads(native["source_path"].read_text())[:8])
    )
    old = sources.prepare_dataset(profile="standard", **native)
    quick = sources.prepare_dataset(profile="quick", **native)
    _, path = write_snapshot(native, tmp_path, old, quick)
    manifests_before = sorted(Path(native["store"]).glob("datasets/*/manifest.json"))
    with pytest.raises(ValueError, match="requested=4, eligible="):
        sources.prepare_dataset(profile="standard", exclusion_snapshot=path, **native)
    assert (
        sorted(Path(native["store"]).glob("datasets/*/manifest.json"))
        == manifests_before
    )
    result = sources.prepare_dataset(
        profile="standard", evaluation_role="retest", **native
    )
    assert result["preparation"]["mmlu-pro"]["evaluation_role"] == "retest"
    assert result["preparation"]["mmlu-pro"]["coverage"] == "no-history-qualification"
    with pytest.raises(ValueError, match="requires an explicit history"):
        sources.prepare_dataset(profile="standard", evaluation_role="holdout", **native)


def test_source_partition_domain_and_format_independence(native, tmp_path):
    prior = sources.prepare_dataset(profile="quick", **native)
    _, path = write_snapshot(native, tmp_path, prior)
    with pytest.raises(ValueError, match="Cross-source"):
        sources.prepare_dataset(
            profile="standard",
            exclusion_snapshot=path,
            **{**native, "source_partition": "upstream-dev"},
        )
    with pytest.raises(ValueError, match="Cross-source"):
        sources.prepare_dataset(
            profile="standard",
            exclusion_snapshot=path,
            **{**native, "revision": "another-revision"},
        )
    source = prior["preparation"]["mmlu-pro"]["task_source"]
    a = native_task_identity("tau3", {"id": 1, "domain": "airline"}, source)
    b = native_task_identity("tau3", {"id": 1, "domain": "retail"}, source)
    assert a != b
    case = cases(prior)[0]
    renamed = copy.deepcopy(case)
    renamed["id"] = "display-alias"
    renamed["metadata"]["split"] = "holdout"
    renamed["messages"] = []
    assert task_key(case, source) == task_key(renamed, source)


def test_imports_and_legacy_history_do_not_acquire_identity(native, tmp_path):
    plain = {k: v for k, v in native.items() if k != "source_partition"}
    prior = sources.prepare_dataset(profile="quick", **plain)
    assert "preparation" not in prior
    with pytest.raises(ValueError, match="lacks source-bound"):
        compile_snapshot(native["store"], dataset_ids=[prior["id"]])
    imported = tmp_path / "import.json"
    imported.write_text(json.dumps(cases(prior)))
    with pytest.raises(ValueError, match="Normalized imports"):
        sources.prepare_dataset(profile="smoke", **{**native, "source_path": imported})
    # An explicit retest does not claim disjointness or require identity recovery.
    retest = sources.prepare_dataset(
        profile="smoke", evaluation_role="retest", **{**plain, "source_path": imported}
    )
    assert retest["preparation"]["mmlu-pro"]["task_source"] is None


@pytest.mark.parametrize("original_index", [0, "0", 407, "407"])
def test_simpleqa_native_original_index_precedes_generic_ids(native, original_index):
    prepared = sources.prepare_dataset(profile="quick", **native)
    source = prepared["preparation"]["mmlu-pro"]["task_source"]
    identity = native_task_identity(
        "simpleqa-verified", {"original_index": original_index, "id": "alias"}, source
    )
    assert identity["task_id"] == str(original_index)
    assert identity["domain"] == ""


@pytest.mark.parametrize("original_index", [True, False, None, "", " ", "\t", 0.0])
def test_simpleqa_invalid_original_index_never_falls_back(native, original_index):
    prepared = sources.prepare_dataset(profile="quick", **native)
    source = prepared["preparation"]["mmlu-pro"]["task_source"]
    with pytest.raises(ValueError):
        native_task_identity(
            "simpleqa-verified",
            {"original_index": original_index, "id": "alias"},
            source,
        )


def test_simpleqa_generic_native_id_retains_contract_without_index_fallback(native):
    prepared = sources.prepare_dataset(profile="quick", **native)
    source = prepared["preparation"]["mmlu-pro"]["task_source"]
    assert (
        native_task_identity("simpleqa-verified", {"id": 0}, source)["task_id"] == "0"
    )
    assert (
        native_task_identity(
            "mmlu-pro", {"original_index": 9, "question_id": 0}, source
        )["task_id"]
        == "0"
    )
    with pytest.raises(ValueError, match="native upstream task ID"):
        native_task_identity("simpleqa-verified", {"problem": "Synthetic"}, source)


def test_simpleqa_csv_native_ids_survive_cli_history_without_changing_case_ids(
    native, tmp_path, monkeypatch
):
    monkeypatch.setitem(sources.COUNTS, "simpleqa-verified", (1, 2, 4))
    path = tmp_path / "simpleqa.csv"
    native_ids = [501, 0, 31, 20, 72, 94, 23, 801]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "original_index",
                "problem",
                "answer",
                "topic",
                "answer_type",
                "multi_step",
                "requires_reasoning",
                "urls",
            ],
        )
        writer.writeheader()
        for identity in native_ids:
            writer.writerow(
                {
                    "original_index": identity,
                    "problem": f"Synthetic {identity}",
                    "answer": "fixture",
                    "topic": "fixture",
                    "answer_type": "str",
                    "multi_step": False,
                    "requires_reasoning": False,
                    "urls": "[]",
                }
            )
    options = {**native, "benchmark": "simpleqa-verified", "source_path": path}
    legacy = sources.prepare_dataset(
        profile="quick", **{k: v for k, v in options.items() if k != "source_partition"}
    )
    legacy_bytes = Path(legacy["path"]).read_bytes()
    runner = CliRunner()
    prefix = ["--store", str(native["store"]), "--no-autostart", "dataset"]
    prepare = [
        "prepare",
        "--local",
        "--benchmark",
        "simpleqa-verified",
        "--source-path",
        str(path),
        "--revision",
        "fixture-v1",
        "--source-partition",
        "upstream-test",
    ]
    quick = runner.invoke(benchmark, prefix + prepare + ["--profile", "quick"])
    assert quick.exit_code == 0, quick.output
    qualified = json.loads(quick.output)
    assert [c["id"] for c in cases(qualified)] == [c["id"] for c in cases(legacy)]
    snapshot_path = tmp_path / "simpleqa-history.json"
    snapshot = runner.invoke(
        benchmark,
        [
            *prefix,
            "exclusions",
            "--dataset",
            qualified["id"],
            "--output",
            str(snapshot_path),
        ],
    )
    assert snapshot.exit_code == 0, snapshot.output
    standard = runner.invoke(
        benchmark,
        prefix
        + prepare
        + [
            "--profile",
            "standard",
            "--exclusion-snapshot",
            str(snapshot_path),
            "--evaluation-role",
            "holdout",
        ],
    )
    assert standard.exit_code == 0, standard.output
    result = json.loads(standard.output)
    selected = cases(result)
    assert len(selected) == 4
    source = result["preparation"]["simpleqa-verified"]["task_source"]
    excluded = set(
        load_snapshot(snapshot_path)["families"]["simpleqa-verified"]["task_keys"]
    )
    assert not {task_key(c, source) for c in selected} & excluded
    expected = {
        f"simpleqa-verified/{position}": str(identity)
        for position, identity in enumerate(native_ids)
    }
    for case in cases(qualified) + selected:
        assert case["metadata"]["task_identity"]["task_id"] == expected[case["id"]]
    assert Path(legacy["path"]).read_bytes() == legacy_bytes
    assert "preparation" not in legacy


def test_snapshot_tamper_immutable_write_and_plan_override(native, tmp_path):
    quick = sources.prepare_dataset(profile="quick", **native)
    snapshot, path = write_snapshot(native, tmp_path, quick)
    assert save_snapshot(snapshot, path) == snapshot
    original = path.read_bytes()
    damaged = copy.deepcopy(snapshot)
    damaged["families"]["mmlu-pro"]["task_keys"].pop()
    with pytest.raises(ValueError, match="identity or policy"):
        save_snapshot(damaged, path)
    assert path.read_bytes() == original
    result = sources.prepare_dataset(
        profile="standard", exclusion_snapshot=path, **native
    )
    request = {"profile": "standard", "seed": result["seed"], "dataset": result}
    resolved = resolve_dataset(request)
    assert resolved["dataset"]["preparation"] == result["preparation"]
    override = copy.deepcopy(request)
    override["dataset"]["preparation"]["mmlu-pro"]["evaluation_role"] = "retest"
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_dataset(override)


def test_cli_freeze_prepare_round_trip(native, tmp_path, monkeypatch):
    def no_requests(*args, **kwargs):
        pytest.fail("History preparation must not request the worker or a model")

    monkeypatch.setattr("cli.sr_bench.client.Client.request", no_requests)
    quick = sources.prepare_dataset(profile="quick", **native)
    runner = CliRunner()
    path = tmp_path / "cli-history.json"
    frozen = runner.invoke(
        benchmark,
        [
            "--store",
            str(native["store"]),
            "--no-autostart",
            "dataset",
            "exclusions",
            "--dataset",
            quick["id"],
            "--output",
            str(path),
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    prepared = runner.invoke(
        benchmark,
        [
            "--store",
            str(native["store"]),
            "--no-autostart",
            "dataset",
            "prepare",
            "--local",
            "--benchmark",
            "mmlu-pro",
            "--profile",
            "standard",
            "--source-path",
            str(native["source_path"]),
            "--revision",
            "fixture-v1",
            "--source-partition",
            "upstream-test",
            "--exclusion-snapshot",
            str(path),
            "--evaluation-role",
            "holdout",
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    result = json.loads(prepared.output)
    assert result["case_count"] == 4
    assert (
        result["preparation"]["mmlu-pro"]["history_snapshot"]["id"]
        == load_snapshot(path)["id"]
    )


@pytest.mark.parametrize(
    "override", ["benchmarks", "integer_as_float", "unsupported_claim"]
)
def test_plan_rejects_all_prepared_metadata_overrides(native, override):
    prepared = sources.prepare_dataset(
        profile="standard", evaluation_role="retest", **native
    )
    request = {
        "profile": "standard",
        "seed": prepared["seed"],
        "dataset": copy.deepcopy(prepared),
    }
    if override == "benchmarks":
        request["dataset"]["benchmarks"] = ["gpqa-diamond"]
    elif override == "integer_as_float":
        request["dataset"]["preparation"]["mmlu-pro"]["selected_count"] = 4.0
    else:
        request["dataset"]["unseen_certified"] = True
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_dataset(request)


def test_stripped_preparation_fails_reader_and_plan(native):
    prepared = sources.prepare_dataset(
        profile="standard", evaluation_role="retest", **native
    )
    path = Path(prepared["path"]).with_name("manifest.json")
    tampered = copy.deepcopy(prepared)
    del tampered["preparation"]
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="provenance is missing"):
        DatasetReader(native["store"]).detail(prepared["id"])
    with pytest.raises(ValueError, match="provenance is missing"):
        resolve_dataset(
            {
                "profile": "standard",
                "dataset": {k: prepared[k] for k in ("path", "sha256")},
            }
        )


def test_membership_and_read_bounds_fail_before_snapshot_publication(
    native, tmp_path, monkeypatch
):
    prepared = sources.prepare_dataset(profile="quick", **native)
    monkeypatch.setattr(history_exclusions, "MAX_ROWS", 1)

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Oversized declared membership must fail before its case scan")

    monkeypatch.setattr(history_exclusions, "verified_lines", unexpected_scan)
    with pytest.raises(ValueError, match="membership count exceeds"):
        compile_snapshot(native["store"], dataset_ids=[prepared["id"]])
    missing = tmp_path / "missing-store"
    with pytest.raises(ValueError, match="existing read-only journal"):
        compile_snapshot(missing, run_ids=["run-" + "1" * 20])
    assert not missing.exists()


def test_declared_count_stops_a_scan_before_collecting_extra_rows(native, monkeypatch):
    prepared = sources.prepare_dataset(profile="quick", **native)
    membership = json.dumps(cases(prepared)[0]).encode()

    def overlong_scan(*args, **kwargs):
        yield membership
        yield membership
        yield membership
        pytest.fail("Reader should stop at the first excess membership")

    monkeypatch.setattr(history_exclusions, "verified_lines", overlong_scan)
    with pytest.raises(ValueError, match="membership count changed"):
        compile_snapshot(native["store"], dataset_ids=[prepared["id"]])


def test_mixed_roles_survive_combine_compose_plan_and_report(
    native, tmp_path, monkeypatch
):
    quick = sources.prepare_dataset(profile="quick", **native)
    _, path = write_snapshot(native, tmp_path, quick)
    holdout = sources.prepare_dataset(
        profile="standard", exclusion_snapshot=path, **native
    )
    monkeypatch.setitem(sources.COUNTS, "gpqa-diamond", (1, 2, 4))
    gpqa = tmp_path / "gpqa-native.json"
    gpqa.write_text(
        json.dumps(
            [
                {
                    "Question": f"Synthetic {i}\n" + "Long native question. " * 110,
                    "Correct Answer": "x",
                    "Incorrect Answer 1": "a",
                    "Incorrect Answer 2": "b",
                    "Incorrect Answer 3": "c",
                }
                for i in range(8)
            ]
        )
    )
    retest = sources.prepare_dataset(
        profile="standard",
        evaluation_role="retest",
        **{**native, "benchmark": "gpqa-diamond", "source_path": gpqa},
    )
    original = {m["path"]: Path(m["path"]).read_bytes() for m in [holdout, retest]}
    bundle = sources.combine_datasets([holdout, retest], native["store"])
    reader = DatasetReader(native["store"])
    composed = reader.compose(
        [bundle["id"], holdout["id"]], ["mmlu-pro", "gpqa-diamond"]
    )
    assert composed["preparation"] == bundle["preparation"]
    subset = reader.compose([bundle["id"]], ["gpqa-diamond"])
    assert set(subset["preparation"]) == {"gpqa-diamond"}
    assert subset["case_count"] == 4
    frozen = plan(
        {
            "version": "sr-bench-1.0",
            "profile": "standard",
            "dataset": composed,
            "cost_policy": "capability_only",
            "targets": [
                {
                    "id": "fixture",
                    "kind": "single",
                    "model": "fixture",
                    "base_url": "http://127.0.0.1:1/v1",
                }
            ],
        }
    )
    store = Store(tmp_path / "reports")
    run, _ = store.create(frozen)
    compact = run_summary(run)["manifest"]["dataset"]
    assert "preparation" not in compact
    assert compact["preparation_summary"]["gpqa-diamond"]["evaluation_role"] == "retest"
    report = make_report(store, run["id"])
    assert report["provenance"]["dataset"]["preparation"] == bundle["preparation"]
    assert any(
        "named frozen dataset/run memberships" in value
        for value in report["limitations"]
    )
    assert any(
        "Explicit retest families" in value and "gpqa-diamond" in value
        for value in report["limitations"]
    )
    retest_plan = plan(
        {
            "version": "sr-bench-1.0",
            "profile": "standard",
            "dataset": subset,
            "cost_policy": "capability_only",
            "targets": frozen["targets"],
        }
    )
    retest_run, _ = store.create(retest_plan)
    retest_report = make_report(store, retest_run["id"])
    assert not any(
        "History exclusions cover" in value for value in retest_report["limitations"]
    )
    assert any(
        "Explicit retest families" in value for value in retest_report["limitations"]
    )
    assert all(Path(p).read_bytes() == content for p, content in original.items())
