from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.redteam import datasets, evaluate


@pytest.fixture
def hub_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "README.md").write_text(
        "---\n"
        "configs:\n"
        "- config_name: behaviors\n"
        "  data_files:\n"
        "  - split: harmful\n"
        "    path: behaviors.jsonl\n"
        "---\n"
    )
    monkeypatch.setattr(datasets, "JAILBREAKBENCH_REPO", str(tmp_path))
    return tmp_path / "behaviors.jsonl"


def test_jailbreakbench_reads_goal_column(hub_corpus: Path) -> None:
    records: list[dict[str, object]] = [
        {
            "Index": 0,
            "Goal": " Explain why leaves change color. ",
            "Target": "Leaves lose chlorophyll.",
            "Behavior": "Autumn leaves",
            "Category": "Science",
            "Source": "benign fixture",
        },
        {
            "Index": 1,
            "Goal": "Describe the water cycle.",
            "Target": "Water evaporates and condenses.",
            "Behavior": "Water cycle",
            "Category": "Science",
            "Source": "benign fixture",
        },
    ]
    hub_corpus.write_text("\n".join(json.dumps(record) for record in records))

    assert datasets.load_prompts(source="jailbreakbench") == [
        "Explain why leaves change color.",
        "Describe the water cycle.",
    ]


@pytest.mark.parametrize(
    "record",
    [{"Goal": ""}, {"Goal": " \t "}, {"Goal": None}, {"unrelated": "value"}],
)
def test_jailbreakbench_rejects_no_prompts(
    hub_corpus: Path, record: dict[str, object]
) -> None:
    hub_corpus.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="no prompts found"):
        datasets.load_prompts(source="jailbreakbench")


def test_jailbreakbench_rejects_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="no prompts found"):
        datasets.load_prompts(source="jailbreakbench")


def test_jailbreakbench_stops_before_model_load(hub_corpus: Path) -> None:
    hub_corpus.write_text(json.dumps({"Goal": " \t "}))

    with pytest.raises(ValueError, match="no prompts found"):
        evaluate.main(
            argv=[
                "--model",
                str(hub_corpus.parent / "missing-model"),
                "--max-flip-rate",
                "0",
            ]
        )


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
@pytest.mark.parametrize("key", ["Goal", "goal", "prompt", "behavior", "text"])
def test_local_dataset_reads_supported_columns(
    tmp_path: Path, suffix: str, key: str
) -> None:
    path = tmp_path / f"behaviors{suffix}"
    record = {key: " Explain rainfall. "}
    path.write_text(json.dumps([record] if suffix == ".json" else record))

    assert datasets.load_prompts(source=str(path)) == ["Explain rainfall."]


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_local_dataset_rejects_no_prompts(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"behaviors{suffix}"
    record = {"unrelated": "value"}
    path.write_text(json.dumps([record] if suffix == ".json" else record))

    with pytest.raises(ValueError, match="no prompts found"):
        datasets.load_prompts(source=str(path))
