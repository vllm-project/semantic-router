"""C1 tests: cases to-model-eval conversion + deterministic 40/30/30 split.

Round trips use one small fixture per source shape (sr-bench normalized
dataset rows and raw office-style MCQ rows). The conversion is validated
against the facts encoded in recipe_lab/model_eval_tasks.py's docstring
(model_eval reads task_id/prompt_user only; mcq_judge compares the answer
letter exactly, A-J uppercase).
"""

from __future__ import annotations

import hashlib
import json

from recipe_lab.cli import main
from recipe_lab.contract import dataset_content_hash
from recipe_lab.splitting import load_cases, parse_ratios, split_cases, write_splits

SRBENCH_SOURCE = (
    {
        "id": "mmlu-pro/1",
        "benchmark": "mmlu-pro",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?\n\nA. 3\nB. 4\n\n"
                "Return only the letter of the correct answer.",
            }
        ],
        "answer": "B",
        "metadata": {"stratum": "math"},
    },
    {
        "id": "mmlu-pro/2",
        "benchmark": "mmlu-pro",
        "messages": [{"role": "user", "content": "Which law binds contracts?"}],
        "answer": "a",
        "metadata": {"stratum": "law"},
    },
)

RAW_SOURCE = (
    {
        "id": "office-1",
        "source": "mmlu",
        "category": "astronomy",
        "question": "Which planet is largest?",
        "options": ["Mars", "Jupiter", "Venus"],
        "answer": "B",
        "answer_type": "mcq",
    },
    {
        "id": "office-2",
        "source": "cmmlu",
        "category": "chinese-history",
        "question": "Which dynasty came first?",
        "options": ["Tang", "Song"],
        "answer": "A",
        "answer_type": "mcq",
    },
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return path


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---- sr-bench normalized source ----


def test_srbench_source_round_trip(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", SRBENCH_SOURCE)
    out = tmp_path / "tasks.jsonl"
    assert (
        main(["cases", "to-model-eval", "--source", str(source), "--out", str(out)])
        == 0
    )
    tasks = _read_jsonl(out)
    assert len(tasks) == 2
    first = tasks[0]
    # model_eval vocabulary (a): prompt carried verbatim, category from stratum
    assert first["task_id"] == "mmlu-pro/1"
    assert first["prompt_user"].startswith("What is 2+2?")
    assert first["category"] == "math"
    assert first["dataset"] == "mmlu-pro"
    assert first["answer"] == "B"
    # cases-vocabulary mirrors keep split/hash working on the same file
    assert first["case_id"] == "mmlu-pro/1"
    assert first["prompt"] == first["prompt_user"]
    assert first["truth"] == "B"
    # load_cases accepts the output (the split command's own loader)
    cases = load_cases(out)
    assert [case["case_id"] for case in cases] == ["mmlu-pro/1", "mmlu-pro/2"]


def test_lowercase_answer_is_normalized(tmp_path):
    source = _write_jsonl(tmp_path / "source.jsonl", SRBENCH_SOURCE)
    out = tmp_path / "tasks.jsonl"
    assert (
        main(["cases", "to-model-eval", "--source", str(source), "--out", str(out)])
        == 0
    )
    tasks = _read_jsonl(out)
    # mcq_judge compares against the uppercase extracted letter exactly
    assert tasks[1]["answer"] == "A"


def test_srbench_source_is_deterministic(tmp_path, capsys):
    source = _write_jsonl(tmp_path / "source.jsonl", SRBENCH_SOURCE)

    def argv(out):
        return ["cases", "to-model-eval", "--source", str(source), "--out", str(out)]

    assert main(argv(tmp_path / "a.jsonl")) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["tasks_count"] == 2
    assert first["tasks_sha256"].startswith("sha256:")
    assert main(argv(tmp_path / "b.jsonl")) == 0
    capsys.readouterr()
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert (
        first["tasks_sha256"]
        == "sha256:" + hashlib.sha256((tmp_path / "a.jsonl").read_bytes()).hexdigest()
    )


# ---- raw MCQ (office) source ----


def test_raw_source_round_trip(tmp_path):
    source = _write_jsonl(tmp_path / "office.jsonl", RAW_SOURCE)
    out = tmp_path / "tasks.jsonl"
    assert (
        main(["cases", "to-model-eval", "--source", str(source), "--out", str(out)])
        == 0
    )
    tasks = _read_jsonl(out)
    first = tasks[0]
    # options are rendered into prompt_user (model_eval has no options field)
    assert first["prompt_user"] == (
        "Which planet is largest?\n\nA. Mars\nB. Jupiter\nC. Venus\n\n"
        "Return only the letter of the correct answer."
    )
    assert first["task_id"] == "office-1"
    assert first["dataset"] == "mmlu"
    assert first["category"] == "astronomy"
    assert first["answer"] == "B"
    assert first["truth"] == "B"
    assert tasks[1]["dataset"] == "cmmlu"
    # raw rows without a source field fall back to --dataset
    assert "language" not in first


def test_raw_source_dataset_fallback(tmp_path):
    rows = [dict(row) for row in RAW_SOURCE]
    for row in rows:
        row.pop("source")
    source = _write_jsonl(tmp_path / "office.jsonl", rows)
    out = tmp_path / "tasks.jsonl"
    assert (
        main(
            [
                "cases",
                "to-model-eval",
                "--source",
                str(source),
                "--out",
                str(out),
                "--dataset",
                "office759",
            ]
        )
        == 0
    )
    tasks = _read_jsonl(out)
    assert all(task["dataset"] == "office759" for task in tasks)


# ---- rejections (never mangle silently) ----


def test_unknown_source_shape_is_rejected(tmp_path):
    source = _write_jsonl(tmp_path / "bad.jsonl", [{"id": "x", "prompt_user": "q"}])
    assert (
        main(
            [
                "cases",
                "to-model-eval",
                "--source",
                str(source),
                "--out",
                str(tmp_path / "t.jsonl"),
            ]
        )
        == 2
    )


def test_non_letter_answer_is_rejected(tmp_path):
    rows = [dict(row) for row in SRBENCH_SOURCE]
    rows[0] = dict(rows[0], answer="42")
    source = _write_jsonl(tmp_path / "source.jsonl", rows)
    assert (
        main(
            [
                "cases",
                "to-model-eval",
                "--source",
                str(source),
                "--out",
                str(tmp_path / "t.jsonl"),
            ]
        )
        == 2
    )


def test_too_many_options_are_rejected(tmp_path):
    rows = [
        {
            "id": "wide",
            "question": "Pick one",
            "options": [f"opt{i}" for i in range(11)],
            "answer": "A",
        }
    ]
    source = _write_jsonl(tmp_path / "wide.jsonl", rows)
    assert (
        main(
            [
                "cases",
                "to-model-eval",
                "--source",
                str(source),
                "--out",
                str(tmp_path / "t.jsonl"),
            ]
        )
        == 2
    )


# ---- deterministic 40/30/30 split on the tasks output (seed 42) ----


def _make_tasks(n_tasks, tmp_path):
    rows = []
    for i in range(n_tasks):
        rows.append(
            {
                "id": f"office-{i}",
                "source": "mmlu",
                "category": f"cat-{i % 3}",
                "question": f"Question {i}?",
                "options": ["yes", "no"],
                "answer": "A" if i % 2 else "B",
                "answer_type": "mcq",
            }
        )
    source = _write_jsonl(tmp_path / "source.jsonl", rows)
    out = tmp_path / "tasks.jsonl"
    assert (
        main(["cases", "to-model-eval", "--source", str(source), "--out", str(out)])
        == 0
    )
    return out


def test_split_on_tasks_output_is_deterministic_and_hashed(tmp_path):
    tasks = _make_tasks(30, tmp_path)
    cases = load_cases(tasks)
    ratios = parse_ratios("0.4,0.3,0.3")
    first = split_cases(cases, ratios, "case_id", 42)
    second = split_cases(cases, ratios, "case_id", 42)
    # same cases + group key + seed => identical assignment
    assert first == second
    hashes_a = write_splits(first, tmp_path / "a")
    write_splits(second, tmp_path / "b")
    # byte-identical files, and every printed hash re-verifies (recipe-lab hash)
    for name in ("calib", "dev", "frozen_test"):
        assert (tmp_path / "a" / f"{name}.jsonl").read_bytes() == (
            tmp_path / "b" / f"{name}.jsonl"
        ).read_bytes()
        assert hashes_a[name]["content_hash"] == dataset_content_hash(
            tmp_path / "a" / f"{name}.jsonl"
        )
    # 40/30/30 at question granularity
    sizes = {
        name: len(_read_jsonl(tmp_path / "a" / f"{name}.jsonl"))
        for name in ("calib", "dev", "frozen_test")
    }
    assert sizes == {"calib": 12, "dev": 9, "frozen_test": 9}
    # every task lands in exactly one split
    assert sum(sizes.values()) == 30
