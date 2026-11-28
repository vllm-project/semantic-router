"""C3 tests: model profiles from router-label-grouped MCQ results.

Stubs replace the real model_eval scripts (they are invoked, never
imported), so every assertion here runs without a model endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from recipe_lab.bundle import _BASE_PRIVATE_PATTERNS
from recipe_lab.cli import main
from recipe_lab.contract import API_VERSION, ContractError
from recipe_lab.model_eval_tasks import load_source_rows
from recipe_lab.profile import (
    PROFILE_SCHEMA,
    _portable_commands,
    _portable_path,
    model_from_results_dirname,
    results_dirname,
    rewrite_task_categories,
    run_profile,
)

STUB_MULTI = """\
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    mode = os.environ.get("PROFILE_STUB_MODE", "ok")
    with open(args.split, encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
    rows = []
    for task in tasks:
        letter = task["answer"]
        row = {
            "task_id": task["task_id"],
            "dataset": task.get("dataset"),
            "category": task.get("category"),
            "language": task.get("language"),
            "completion": "Answer: " + letter,
            "prompt_tokens": 21,
            "completion_tokens": 13,
            "finish_reason": "stop",
            "ttft_s": 0.1,
            "latency_s": 0.5,
            "error": None,
        }
        if mode == "truncate":
            row["completion"] = "partial answer that never finishes"
            row["completion_tokens"] = 9999
            row["finish_reason"] = "length"
        elif mode == "noanswer":
            row["completion"] = "no idea"
        rows.append(row)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\\n")

if __name__ == "__main__":
    main()
"""

STUB_JUDGE = """\
import argparse, json, re

def extract_letter(text):
    found = re.findall(r"\\b([A-J])\\b", text or "")
    return found[-1] if found else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--analysis", default=None)
    args = ap.parse_args()
    with open(args.tasks, encoding="utf-8") as fh:
        tasks = {}
        for line in fh:
            if line.strip():
                row = json.loads(line)
                tasks[row["task_id"]] = row
    with open(args.records, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    judged = []
    for rec in records:
        task = tasks.get(rec["task_id"])
        if task is None:
            continue
        letter = extract_letter(rec.get("completion") or "")
        judged.append({
            "task_id": rec["task_id"],
            "dataset": rec.get("dataset"),
            "category": rec.get("category"),
            "language": rec.get("language"),
            "answer_expected": task["answer"],
            "answer_extracted": letter,
            "is_correct": letter == task["answer"],
            "no_answer": letter is None,
            "gen_error": rec.get("error"),
            "finish_reason": rec.get("finish_reason"),
            "completion_tokens": rec.get("completion_tokens"),
            "latency_s": rec.get("latency_s"),
            "completion": rec.get("completion"),
        })
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in judged:
            fh.write(json.dumps(row) + "\\n")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump({"n_total": len(judged)}, fh, indent=2)
    if args.analysis:
        totals = {}
        correct = {}
        for row in judged:
            cat = row["category"]
            totals[cat] = totals.get(cat, 0) + 1
            if row["is_correct"]:
                correct[cat] = correct.get(cat, 0) + 1
        analysis = {
            "category_accuracy": {
                cat: correct.get(cat, 0) / totals[cat] for cat in sorted(totals)
            }
        }
        with open(args.analysis, "w", encoding="utf-8") as fh:
            json.dump(analysis, fh, indent=2)

if __name__ == "__main__":
    main()
"""

STUB_COST = """\
import argparse, json, os
from collections import defaultdict
from urllib.parse import quote

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--lambdas", default="")
    ap.add_argument("--acc-tolerance", type=float, default=0.0)
    ap.add_argument("--token-costs-dir", default="")
    ap.add_argument("--token-costs-only", action="store_true")
    args = ap.parse_args()
    if not args.token_costs_dir:
        return
    sums = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(lambda: defaultdict(int))
    for entry in args.model:
        name, path = entry.split(",", 1)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                cat = row.get("category") or "other"
                sums[name][cat] += row.get("completion_tokens") or 0
                counts[name][cat] += 1
    os.makedirs(args.token_costs_dir, exist_ok=True)
    for name in sums:
        avg = {cat: sums[name][cat] / counts[name][cat] for cat in sorted(sums[name])}
        target = os.path.join(args.token_costs_dir, quote(name, safe="") + ".json")
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(avg, fh, indent=2)

if __name__ == "__main__":
    main()
"""


def _write_stub_scripts(tmp_path: Path) -> str:
    stub_dir = tmp_path / "model_eval"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "multi_benchmark_eval.py").write_text(STUB_MULTI, encoding="utf-8")
    (stub_dir / "mcq_judge.py").write_text(STUB_JUDGE, encoding="utf-8")
    (stub_dir / "cost_aware_calib.py").write_text(STUB_COST, encoding="utf-8")
    return str(stub_dir)


def _write_profile_spec(
    tmp_path: Path,
    split_cases: dict[str, str],
    *,
    with_models: bool = True,
    with_generation: bool = True,
) -> str:
    spec: dict = {
        "api_version": API_VERSION,
        "objective": {
            "metric_key": "capability.score",
            "direction": "maximize",
            "min_improvement_pct": 2.0,
        },
        "workloads": [
            {
                "name": "mmlu-pro",
                "profile": "quick",
                "splits": {
                    name: {
                        "manifest": f"{name}.jsonl",
                        "content_hash": "sha256:"
                        + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    }
                    for name, payload in split_cases.items()
                },
            }
        ],
        "attribution": {"oracle_policy": {"math": "Qwen/Qwen3.5-4B"}},
        "budget": {"rounds": 1, "max_minutes_per_trial": 5},
    }
    if with_models:
        spec["models"] = [
            {
                "name": "Qwen/Qwen3.5-4B",
                "endpoint": "http://127.0.0.1:8001/v1",
            },
            {
                "name": "plainmodel",
                "endpoint": "http://127.0.0.1:8002/v1",
                "served_name": "plain-served",
            },
        ]
    if with_generation:
        spec["generation"] = {
            "temperature": 0.0,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    spec["cost_basis"] = {
        "source": "provider_pricing",
        "assumed_pricing": {
            "Qwen/Qwen3.5-4B": {
                "prompt_per_1m": 0.5,
                "completion_per_1m": 1.0,
            }
        },
    }
    for name, payload in split_cases.items():
        (tmp_path / f"{name}.jsonl").write_bytes(payload.encode("utf-8"))
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _write_labels(labels_dir: Path, labels_by_split: dict[str, dict[str, str]]) -> str:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for split, mapping in labels_by_split.items():
        payload = "".join(
            json.dumps({"case_id": cid, "router_label": label}) + "\n"
            for cid, label in mapping.items()
        )
        (labels_dir / f"labels-{split}.jsonl").write_text(payload, encoding="utf-8")
    return str(labels_dir)


CALIB_CASES = (
    '{"case_id":"m1","prompt":"p1","truth":"A","task_id":"m1","prompt_user":"p1",'
    '"answer":"A","dataset":"office","category":"math"}\n'
    '{"case_id":"l1","prompt":"p2","truth":"B","task_id":"l1","prompt_user":"p2",'
    '"answer":"B","dataset":"office","category":"law"}\n'
    '{"case_id":"r1","prompt":"p3","truth":"C","task_id":"r1","prompt_user":"p3",'
    '"answer":"C","dataset":"office","category":"rare"}\n'
)
DEV_CASES = (
    '{"case_id":"d1","prompt":"pd","truth":"A","task_id":"d1","prompt_user":"pd",'
    '"answer":"A","dataset":"office","category":"math"}\n'
)


def test_results_dirname_round_trip():
    assert results_dirname("Qwen/Qwen3.5-4B") == "Qwen_Qwen3.5-4B_direct"
    assert model_from_results_dirname("Qwen_Qwen3.5-4B_direct") == "Qwen/Qwen3.5-4B"
    assert results_dirname("plainmodel") == "plainmodel_direct"
    assert model_from_results_dirname("plainmodel_direct") == "plainmodel"
    with pytest.raises(ContractError, match="round-trip"):
        results_dirname("bad_alias")  # underscore without a slash
    with pytest.raises(ContractError, match="round-trip"):
        results_dirname("a/b/c")  # two slashes cannot map to one basename
    with pytest.raises(ContractError, match="round-trip"):
        results_dirname("my_cot/model")  # collides with the _cot stripping


def test_rewrite_task_categories_merges_rare_labels(tmp_path):
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        '{"task_id":"t1","case_id":"t1","category":"math","answer":"A"}\n'
        '{"task_id":"t2","case_id":"t2","category":"law","answer":"B"}\n'
        '{"task_id":"t3","case_id":"t3","category":"math","answer":"C"}\n'
        '{"task_id":"t4","case_id":"t4","answer":"A"}\n',
        encoding="utf-8",
    )
    rows = load_source_rows(source)
    labels = {
        "t1": "math",
        "t2": "law",
        "t3": "math",
        # t4 has no observed label
    }
    tasks, merge_record, uncovered = rewrite_task_categories(
        rows, labels, min_label_cases=2
    )
    by_id = {row["task_id"]: row["category"] for row in tasks}
    assert by_id == {
        "t1": "math",
        "t2": "other",  # law has 1 case < min_label_cases
        "t3": "math",
        "t4": "other",  # uncovered joins other too
    }
    assert merge_record == {
        "min_label_cases": 2,
        "merged": {"law": 1},
        "merged_into": "other",
    }
    assert uncovered == 1


def test_portable_path_recognizes_login_and_real_home(tmp_path, monkeypatch):
    login = tmp_path / "login" / "researcher"
    storage = tmp_path / "storage" / "researcher"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: login))
    monkeypatch.setattr(os.path, "realpath", lambda path: os.fspath(storage))
    suffix = Path("venv") / "bin" / "python"
    for home_path in (login, storage):
        assert _portable_path(str(home_path)) == "~"
        assert _portable_path(str(home_path / suffix)) == "~" + os.sep + str(suffix)
    sibling = storage.with_name("researcher-archive") / suffix
    assert _portable_path(str(sibling)) == str(sibling)
    assert _portable_path("--output") == "--output"
    judged = suffix.with_name("sample,1.jsonl")
    argv = ["script.py", "--model", f"provider/model,{storage / judged}"]
    recorded = _portable_commands({"calib": {"token_costs": argv}})["calib"][
        "token_costs"
    ]
    assert recorded == [
        "script.py",
        "--model",
        "provider/model,~" + os.sep + str(judged),
    ]


def test_run_profile_stub_end_to_end(tmp_path):
    spec_path = _write_profile_spec(tmp_path, {"calib": CALIB_CASES, "dev": DEV_CASES})
    labels_dir = _write_labels(
        tmp_path / "labels",
        {
            "calib": {"m1": "math", "l1": "law", "r1": "other"},
            "dev": {"d1": "math"},
        },
    )
    stub_dir = _write_stub_scripts(tmp_path)
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    cases_paths = {
        "calib": tmp_path / "calib.jsonl",
        "dev": tmp_path / "dev.jsonl",
    }
    cases_hashes = {
        "calib": spec.verify_workload(split="calib"),
        "dev": spec.verify_workload(split="dev"),
    }
    out_dir = tmp_path / "profile"
    profile_path = run_profile(
        spec=spec,
        cases_paths=cases_paths,
        cases_hashes=cases_hashes,
        labels_dir=labels_dir,
        splits=["calib", "dev"],
        out_dir=out_dir,
        model_eval_dir=stub_dir,
        python_bin=sys.executable,
    )
    assert profile_path == out_dir / "profile.json"

    # Directory layout: per split, tasks + per-model results + token costs.
    for split in ("calib", "dev"):
        assert (out_dir / split / "tasks.jsonl").is_file()
        for base in ("Qwen_Qwen3.5-4B_direct", "plainmodel_direct"):
            rdir = out_dir / split / "results" / base
            for name in (
                "records.jsonl",
                "judged.jsonl",
                "summary.json",
                "analysis.json",
            ):
                assert (rdir / name).is_file(), (split, base, name)
        assert (out_dir / split / "token_costs").is_dir()
    # model_eval category rewrite: every calib label has 1 case < the
    # default min_label_cases=20, so all merge into `other`.
    analysis = json.loads(
        (
            out_dir / "calib" / "results" / "Qwen_Qwen3.5-4B_direct" / "analysis.json"
        ).read_text(encoding="utf-8")
    )
    assert set(analysis["category_accuracy"]) == {"other"}

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["schema"] == PROFILE_SCHEMA
    assert profile["models"] == ["Qwen/Qwen3.5-4B", "plainmodel"]
    # The merge record names the dropped labels and their case counts.
    assert profile["label_merges"]["calib"]["merged"] == {"law": 1, "math": 1}
    assert profile["uncovered_cases"] == {"calib": 0, "dev": 0}
    stats = profile["model_split_stats"]["calib"]["Qwen/Qwen3.5-4B"]
    assert stats["total"]["n"] == 3
    assert stats["total"]["accuracy"] == pytest.approx(1.0)
    # The M3 gate rates live in the quality block, not the stats.
    quality = profile["quality"]["calib"]["Qwen/Qwen3.5-4B"]
    assert quality["truncation_rate"] == 0.0
    assert quality["no_answer_rate"] == 0.0
    # Cost from the contract's assumed_pricing (assumed, per question):
    # 21 prompt tokens + 13 completion tokens.
    expected_cost = 21 / 1e6 * 0.5 + 13 / 1e6 * 1.0
    assert stats["total"]["cost_per_question"] == pytest.approx(expected_cost)
    assert stats["total"]["cost_basis"] == "assumed_pricing"
    # The unpriced model has no cost - never a fake zero.
    plain = profile["model_split_stats"]["calib"]["plainmodel"]["total"]
    assert plain["cost_per_question"] is None
    assert "cost_basis" not in plain
    # Provenance: script hashes + full command lines. Recorded paths are
    # portable (a $HOME prefix renders as '~'), so the bundle's C9 private
    # content scan passes on real deployments.
    scripts = profile["inputs"]["scripts"]
    assert scripts["multi_benchmark_eval.py"]["sha256"].startswith("sha256:")
    assert scripts["multi_benchmark_eval.py"]["git_blob"]
    gen_cmd = profile["inputs"]["commands"]["calib"]["Qwen/Qwen3.5-4B/generate"]
    assert gen_cmd[0] == _portable_path(sys.executable)
    assert "--endpoint" in gen_cmd and "http://127.0.0.1:8001/v1" in gen_cmd
    assert "--model" in gen_cmd and "Qwen/Qwen3.5-4B" in gen_cmd

    # C9: the packaged profile artifact carries no absolute personal user
    # path (real deployments live under /home/<user> or C:\\Users\\<user>).
    profile_text = profile_path.read_text(encoding="utf-8")
    for pattern_name, pattern in _BASE_PRIVATE_PATTERNS:
        assert not pattern.search(profile_text), f"leaked {pattern_name}"

    # Byte stability: rerunning the same inputs into the same directory
    # rewrites profile.json with identical bytes.
    first = profile_path.read_bytes()
    run_profile(
        spec=spec,
        cases_paths=cases_paths,
        cases_hashes=cases_hashes,
        labels_dir=labels_dir,
        splits=["calib", "dev"],
        out_dir=out_dir,
        model_eval_dir=stub_dir,
        python_bin=sys.executable,
    )
    assert profile_path.read_bytes() == first

    # Token costs feed C4: one percent-encoded file per model.
    tc = out_dir / "calib" / "token_costs"
    assert (tc / "Qwen%2FQwen3.5-4B.json").is_file()
    assert (tc / "plainmodel.json").is_file()


def test_run_profile_refuses_missing_generation(tmp_path):
    spec_path = _write_profile_spec(
        tmp_path,
        {"calib": CALIB_CASES},
        with_generation=False,
    )
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="generation"):
        run_profile(
            spec=spec,
            cases_paths={"calib": tmp_path / "calib.jsonl"},
            cases_hashes={"calib": spec.verify_workload(split="calib")},
            labels_dir=_write_labels(tmp_path / "labels", {"calib": {"m1": "math"}}),
            splits=["calib"],
            out_dir=tmp_path / "profile",
            model_eval_dir=_write_stub_scripts(tmp_path),
            python_bin=sys.executable,
        )


def test_run_profile_truncation_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_STUB_MODE", "truncate")
    spec_path = _write_profile_spec(tmp_path, {"calib": CALIB_CASES})
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="truncation rate"):
        run_profile(
            spec=spec,
            cases_paths={"calib": tmp_path / "calib.jsonl"},
            cases_hashes={"calib": spec.verify_workload(split="calib")},
            labels_dir=_write_labels(tmp_path / "labels", {"calib": {"m1": "math"}}),
            splits=["calib"],
            out_dir=tmp_path / "profile",
            model_eval_dir=_write_stub_scripts(tmp_path),
            python_bin=sys.executable,
        )


def test_run_profile_no_answer_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILE_STUB_MODE", "noanswer")
    spec_path = _write_profile_spec(tmp_path, {"calib": CALIB_CASES})
    from recipe_lab.contract import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    with pytest.raises(ContractError, match="no-answer rate"):
        run_profile(
            spec=spec,
            cases_paths={"calib": tmp_path / "calib.jsonl"},
            cases_hashes={"calib": spec.verify_workload(split="calib")},
            labels_dir=_write_labels(tmp_path / "labels", {"calib": {"m1": "math"}}),
            splits=["calib"],
            out_dir=tmp_path / "profile",
            model_eval_dir=_write_stub_scripts(tmp_path),
            python_bin=sys.executable,
        )


def test_profile_cli_end_to_end(tmp_path, capsys):
    spec_path = _write_profile_spec(tmp_path, {"calib": CALIB_CASES, "dev": DEV_CASES})
    labels_dir = _write_labels(
        tmp_path / "labels",
        {
            "calib": {"m1": "math", "l1": "law", "r1": "other"},
            "dev": {"d1": "math"},
        },
    )
    stub_dir = _write_stub_scripts(tmp_path)
    exit_code = main(
        [
            "profile",
            "--spec",
            spec_path,
            "--labels",
            labels_dir,
            "--splits",
            "calib,dev",
            "--out",
            str(tmp_path / "profile"),
            "--model-eval-dir",
            stub_dir,
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["profile"]).is_file()
    assert payload["models"] == ["Qwen/Qwen3.5-4B", "plainmodel"]


def test_profile_cli_refuses_frozen_test(tmp_path):
    spec_path = _write_profile_spec(
        tmp_path, {"calib": CALIB_CASES, "dev": DEV_CASES, "frozen_test": DEV_CASES}
    )
    labels_dir = _write_labels(
        tmp_path / "labels",
        {"calib": {"m1": "math"}, "dev": {"d1": "math"}, "frozen_test": {"d1": "math"}},
    )
    stub_dir = _write_stub_scripts(tmp_path)
    exit_code = main(
        [
            "profile",
            "--spec",
            spec_path,
            "--labels",
            labels_dir,
            "--splits",
            "calib,frozen_test",
            "--out",
            str(tmp_path / "profile"),
            "--model-eval-dir",
            stub_dir,
        ]
    )
    assert exit_code == 2


def test_profile_cli_refuses_spec_without_models(tmp_path):
    spec_path = _write_profile_spec(tmp_path, {"calib": CALIB_CASES}, with_models=False)
    labels_dir = _write_labels(tmp_path / "labels", {"calib": {"m1": "math"}})
    stub_dir = _write_stub_scripts(tmp_path)
    exit_code = main(
        [
            "profile",
            "--spec",
            spec_path,
            "--labels",
            labels_dir,
            "--out",
            str(tmp_path / "profile"),
            "--model-eval-dir",
            stub_dir,
        ]
    )
    assert exit_code == 2
