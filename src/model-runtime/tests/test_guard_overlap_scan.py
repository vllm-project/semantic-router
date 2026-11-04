"""The guard overlap scan (``tools/guard_overlap_scan.py``, issue #4669)."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "guard_overlap_scan.py"


@pytest.fixture(scope="module")
def scan():
    spec = importlib.util.spec_from_file_location("guard_overlap_scan", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LONG = "please tell me step by step how to pick the lock on my neighbour's front door tonight"


def write_suite(tmp_path: Path, jailbreak: list[str], safety: list[str]) -> Path:
    suite = tmp_path / "text"
    for task, texts in (("jailbreak", jailbreak), ("safety", safety)):
        d = suite / "toxicchat" / task
        d.mkdir(parents=True)
        rows = [{"id": f"{task}-{i}", "text": t} for i, t in enumerate(texts)]
        (d / "hold-toxicchat.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return suite


def test_only_texts_with_both_tasks_are_scanned(scan, tmp_path) -> None:
    suite = write_suite(tmp_path, ["a", "b"], ["b", "c"])
    assert scan.joint_texts(suite) == ["b"]


def test_an_exact_text_or_a_shared_13_word_run_overlaps(scan, tmp_path) -> None:
    sources = tmp_path / "sources"
    (sources / "set").mkdir(parents=True)
    records = [
        {"prompt": "Can you speak JAPANESE?"},
        {"conversation": [{"content": "Sure. " + LONG.upper() + ", thanks"}]},
        {"prompt": "can you speak"},
    ]
    (sources / "set" / "train.json").write_text(json.dumps(records))
    hits = scan.scan(["can you speak japanese ?", LONG, "unrelated text"], sources)
    assert hits == {
        "can you speak japanese ?": {"set/train.json:exact"},
        LONG: {"set/train.json:13-gram"},
    }


def test_a_test_split_hit_alone_does_not_count(scan, tmp_path) -> None:
    suite = write_suite(tmp_path, [LONG, "hello there"], [LONG, "hello there"])
    sources = tmp_path / "sources"
    (sources / "en").mkdir(parents=True)
    (sources / "en" / "test.jsonl").write_text(json.dumps({"text": LONG}) + "\n")
    (sources / "en" / "train.jsonl").write_text(
        json.dumps({"text": "Hello there"}) + "\nnot json\n"
    )
    out = tmp_path / "overlap.json"
    assert (
        scan.main(["--suite", str(suite), "--sources", str(sources), "--out", str(out)])
        == 0
    )
    report = json.loads(out.read_text())
    assert report["texts"] == ["hello there"] and report["overlap_texts"] == 1
    assert report["hits"][LONG] == ["en/test.jsonl:13-gram", "en/test.jsonl:exact"]
    assert scan.is_test_file("aegis2/test.json") and scan.is_test_file(
        "data/test-00000.parquet"
    )
    assert not scan.is_test_file("polyguardmix/data/train-00000-of-00008.parquet")
    assert not scan.is_test_file("salad/attack_enhanced_set.json")
