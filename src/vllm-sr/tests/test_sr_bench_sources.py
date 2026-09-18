"""Dataset contracts that keep iteration and holdout evidence comparable."""

import json
from collections import Counter
from pathlib import Path

import pytest

from cli.sr_bench.sources import _stratified_order, combine_datasets, prepare_dataset
from cli.sr_bench.setup import PACKAGES, TASK_SOURCES
from cli.sr_bench.contracts import load_document


def _source(tmp_path, count=198):
    source = tmp_path / "gpqa.json"
    source.write_text(
        json.dumps(
            [
                {
                    "benchmark": "gpqa-diamond",
                    "id": f"gpqa-diamond/{i}",
                    "messages": [{"role": "user", "content": f"Question {i}"}],
                    "answer": "A",
                    "metadata": {"stratum": "science"},
                }
                for i in range(count)
            ]
        )
    )
    return source


def test_fixed_dev_smoke_and_disjoint_holdout(tmp_path):
    source = _source(tmp_path)
    kwargs = {
        "benchmark": "gpqa-diamond",
        "store": tmp_path / "store",
        "source_path": source,
        "revision": "fixture-v1",
    }
    datasets = {
        profile: prepare_dataset(profile=profile, **kwargs)
        for profile in ["smoke", "quick", "standard"]
    }
    ids = {
        profile: {
            json.loads(line)["id"]
            for line in Path(manifest["path"]).read_text().splitlines()
        }
        for profile, manifest in datasets.items()
    }
    assert ids["smoke"] <= ids["quick"]
    assert ids["quick"].isdisjoint(ids["standard"])
    assert len(ids["quick"] | ids["standard"]) == 198
    assert prepare_dataset(profile="quick", **kwargs) == datasets["quick"]
    assert datasets["standard"]["split"] == "holdout"
    assert (
        combine_datasets([datasets["quick"]], tmp_path / "store") == datasets["quick"]
    )


def test_prepared_dataset_never_overwrites_corrupted_evidence(tmp_path):
    source = _source(tmp_path)
    kwargs = {
        "benchmark": "gpqa-diamond",
        "profile": "quick",
        "store": tmp_path / "store",
        "source_path": source,
        "revision": "fixture-v1",
    }
    manifest = prepare_dataset(**kwargs)
    path = Path(manifest["path"])
    path.write_text("corrupted")
    with pytest.raises(ValueError, match="immutable dataset content changed"):
        prepare_dataset(**kwargs)
    assert path.read_text() == "corrupted"


def test_jsonl_preserves_unicode_separators_inside_task_strings(tmp_path):
    source = tmp_path / "source.jsonl"
    content = "Question with embedded separators:\u2028paragraph\u2029next\u0085line"
    cases = [
        {
            "benchmark": "gpqa-diamond",
            "id": f"gpqa-diamond/{i}",
            "messages": [{"role": "user", "content": content}],
            "answer": "A",
        }
        for i in range(198)
    ]
    source.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n"
    )
    assert load_document(source) == cases
    prepared = prepare_dataset(
        benchmark="gpqa-diamond",
        profile="standard",
        store=tmp_path / "store",
        source_path=source,
        revision="fixture-v1",
    )
    combined = combine_datasets([prepared], tmp_path / "store")
    assert combined == prepared
    assert len(load_document(combined["path"])) == 158
    assert load_document(combined["path"])[0]["messages"][0]["content"] == content


def test_strata_coverage_and_balanced_agent_domains():
    cases = [
        {"id": f"{g}/{i}", "metadata": {"stratum": str(g)}}
        for g in range(14)
        for i in range(g + 1)
    ]
    assert (
        len({c["metadata"]["stratum"] for c in _stratified_order(cases, 7)[:14]}) == 14
    )
    domains = [
        {"id": f"{g}/{i}", "metadata": {"stratum": g}}
        for g, n in [("a", 50), ("b", 114), ("c", 114)]
        for i in range(n)
    ]
    assert Counter(
        c["metadata"]["stratum"]
        for c in _stratified_order(domains, 7, balanced=True)[:12]
    ) == {"a": 4, "b": 4, "c": 4}


def test_optional_harness_and_task_sources_are_pinned():
    for spec in [*PACKAGES.values(), *TASK_SOURCES.values()]:
        assert len(spec["revision"]) == 40
        assert all(c in "0123456789abcdef" for c in spec["revision"])
        assert spec["repo"].startswith("https://github.com/")
