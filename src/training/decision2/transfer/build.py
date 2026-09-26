"""Build a gold-free typed-choice panel from the CSS replication sources.

Run on the experiment host with the two pinned GitHub snapshots. The builder
parses the authors' task maps as Python literals; it never imports their
API-calling script or needs an API key. The original test text stays remote.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPLICATION_REVISION = "311956c2d1096cabc0f09a1f248e7dc0e1c0c41e"
DATA_REVISION = "55183a64d7faaf6d5fc23eddf2bfc48ece4cacad"
PANEL_VERSION = "css-transfer/1"
PILOT_TASKS = ("semeval_stance", "implicit_hate", "discourse")
EVALUATION_TASKS = (
    "emotion",
    "ibc",
    "media_ideology",
    "indian_english_dialect",
    "raop",
    "talklife",
    "wiki_politeness",
    "tempowic",
    "tropes",
    "flute",
    "mrf",
    "conv_go_awry",
    "persuasion",
    "reddit_humor",
    "wiki_corpus",
)
ALL_TASKS = PILOT_TASKS + EVALUATION_TASKS


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_value(value: Any) -> str:
    return sha_bytes(json_text(value).encode("utf-8"))


def git_head(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_clean_tracked_files(path: Path) -> None:
    import subprocess

    changed = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if changed:
        raise ValueError(f"source snapshot has changed tracked files: {path}")


def source_maps(
    script_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    module = ast.parse(
        script_path.read_text(encoding="utf-8"), filename=str(script_path)
    )
    assignments = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    tropes = ast.literal_eval(assignments["TROPES_LABELS"])
    option_node = assignments["OPTION_TO_GOLD"]
    if not isinstance(option_node, ast.Dict):
        raise ValueError("authors' OPTION_TO_GOLD is no longer a dictionary literal")
    option_to_gold = {}
    for key_node, value_node in zip(option_node.keys, option_node.values):
        task = ast.literal_eval(key_node)
        option_to_gold[task] = (
            {label: label for label in tropes}
            if task == "tropes"
            else ast.literal_eval(value_node)
        )
    binary_criteria = ast.literal_eval(assignments["BINARY_CRITERIA"])
    if set(option_to_gold) | set(binary_criteria) != set(ALL_TASKS):
        raise ValueError(
            "authors' CSS task mapping differs from the frozen 18-task protocol"
        )
    if set(option_to_gold) & set(binary_criteria):
        raise ValueError("overlapping option and binary maps")
    return option_to_gold, binary_criteria


def parse_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    """Match the replication package's `pilot_jev.py::parse_prompt` contract."""
    instructions: list[str] = []
    options: dict[str, str] = {}
    current: str | None = None
    for line in prompt.strip().split("\n"):
        match = re.match(r"^([A-Z]+): (.*)$", line)
        if match:
            current = match.group(1)
            options[current] = match.group(2)
        elif line.startswith("Constraint:"):
            current = None
        elif current:
            options[current] += " " + line.strip()
        else:
            instructions.append(line)
    return " ".join(instructions).strip(), options


def criteria_for(
    task: str,
    prompt: str,
    option_to_gold: dict[str, dict[str, Any]],
    binary_criteria: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]]:
    instructions, options = parse_prompt(prompt)
    if not instructions:
        raise ValueError(f"{task}: empty instructions")
    if task in binary_criteria:
        criteria = binary_criteria[task]
    elif task == "tropes":
        criteria = {}
        for label in option_to_gold[task]:
            parts = [part.strip() for part in label.split(",")]
            if any(part not in options for part in parts):
                raise ValueError("tropes prompt misses a composite label component")
            criteria[label] = "; ".join(f"{part}: {options[part]}" for part in parts)
    else:
        mapping = option_to_gold[task]
        if not set(options) <= set(mapping):
            raise ValueError(f"{task}: unmapped prompt option")
        criteria = {}
        for letter, gold in mapping.items():
            description = options.get(letter)
            if (
                description is None
                and task == "indian_english_dialect"
                and gold == "None of the Above"
            ):
                description = (
                    "None of the listed Indian English dialect features apply."
                )
            if description is None:
                raise ValueError(f"{task}: missing option {letter}")
            criteria[str(gold)] = description
    if not 2 <= len(criteria) <= 255 or len(criteria) != len(set(criteria)):
        raise ValueError(f"{task}: invalid criterion count or duplicate label")
    return instructions, criteria


def source_data_path(data_root: Path, task: str) -> Path:
    file_name = "test-classification.json" if task in {"flute", "mrf"} else "test.json"
    return data_root / "css_data" / task / file_name


def normalized_context_sha256(context: str) -> str:
    return sha_bytes(" ".join(context.casefold().split()).encode("utf-8"))


def build_task(
    task: str,
    data_root: Path,
    option_to_gold: dict[str, dict[str, Any]],
    binary_criteria: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = source_data_path(data_root, task)
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"context", "labels", "prompts"}:
        raise ValueError(f"{task}: unexpected source JSON keys")
    if (
        data["context"].keys() != data["labels"].keys()
        or data["labels"].keys() != data["prompts"].keys()
    ):
        raise ValueError(f"{task}: source ID maps differ")
    pilot = task in PILOT_TASKS
    prompts, gold_rows = [], []
    for source_id, raw_gold in data["labels"].items():
        context, raw_prompt = data["context"][source_id], data["prompts"][source_id]
        if not isinstance(context, str) or not isinstance(raw_prompt, str):
            raise ValueError(
                f"{task}/{source_id}: source context or prompt is not text"
            )
        instructions, criteria = criteria_for(
            task, raw_prompt, option_to_gold, binary_criteria
        )
        gold = str(raw_gold)
        if gold not in criteria:
            raise ValueError(
                f"{task}/{source_id}: gold label absent from criteria: {gold}"
            )
        item_id = f"css/{task}/{source_id}"
        question = {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
        payload = {"state": context, "questions": {"label": question}}
        prompts.append({"id": item_id, **payload})
        gold_rows.append(
            {
                "id": item_id,
                "panel_version": PANEL_VERSION,
                "role": "pilot" if pilot else "evaluation",
                "task": task,
                "source_id": source_id,
                "gold": gold,
                "labels": list(criteria),
                "source_context_sha256": sha_bytes(context.encode("utf-8")),
                "normalized_context_sha256": normalized_context_sha256(context),
                "source_prompt_sha256": sha_bytes(raw_prompt.encode("utf-8")),
                "input_sha256": sha_value(payload),
            }
        )
    if not prompts:
        raise ValueError(f"{task}: empty test split")
    counts = Counter(row["gold"] for row in gold_rows)
    return (
        prompts,
        gold_rows,
        {
            "task": task,
            "role": "pilot" if pilot else "evaluation",
            "source_path": f"css_data/{task}/{path.name}",
            "source_sha256": sha_file(path),
            "n": len(prompts),
            "class_counts": dict(sorted(counts.items())),
            "criteria_labels": gold_rows[0]["labels"],
        },
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=".css-transfer.", dir=path.parent)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            for row in rows:
                target.write(json_text(row) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(
    replication_root: Path,
    data_root: Path,
    output_dir: Path,
    tasks: tuple[str, ...] = ALL_TASKS,
) -> dict[str, Any]:
    if git_head(replication_root) != REPLICATION_REVISION:
        raise ValueError("replication package is not at the pinned commit")
    if git_head(data_root) != DATA_REVISION:
        raise ValueError("SALT data repository is not at the pinned commit")
    require_clean_tracked_files(replication_root)
    require_clean_tracked_files(data_root)
    if not tasks or not set(tasks) <= set(ALL_TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a nonempty set drawn from the 18 paper tasks")
    script = replication_root / "pilot_jev.py"
    option_to_gold, binary_criteria = source_maps(script)
    by_role: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {"prompts": [], "gold": []} for role in ("pilot", "evaluation")
    }
    task_manifest = []
    for task in tasks:
        prompts, gold, info = build_task(
            task, data_root, option_to_gold, binary_criteria
        )
        role = info["role"]
        by_role[role]["prompts"].extend(prompts)
        by_role[role]["gold"].extend(gold)
        task_manifest.append(info)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is nonempty; choose a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    outputs = {}
    for role, records in by_role.items():
        if not records["prompts"]:
            continue
        for kind, rows in records.items():
            path = output_dir / f"css-{role}.{kind}.jsonl"
            write_jsonl(path, rows)
            outputs[f"{role}_{kind}"] = {
                "file": path.name,
                "sha256": sha_file(path),
                "n": len(rows),
            }
    manifest = {
        "panel_version": PANEL_VERSION,
        "protocol": "authors' zero-shot typed-choice task mapping; full released test splits",
        "paper": "https://arxiv.org/html/2609.24574v2",
        "replication_repository": "https://github.com/hazemibrahim97/decision-models-css",
        "replication_revision": REPLICATION_REVISION,
        "replication_script_sha256": sha_file(script),
        "replication_code_license": "MIT",
        "replication_license_sha256": sha_file(replication_root / "LICENSE"),
        "data_repository": "https://github.com/SALT-NLP/LLMs_for_CSS",
        "data_revision": DATA_REVISION,
        "data_license_status": "The SALT snapshot has no top-level LICENSE file; task datasets retain their original creators' rights. Do not redistribute the generated text without reviewing each source's terms.",
        "selected_tasks": list(tasks),
        "task_details": task_manifest,
        "outputs": outputs,
        "evaluation_policy": "Pilot tasks may be used for adapter checks or predeclared calibration. The 15 evaluation tasks must not be used for model or threshold selection.",
    }
    manifest_path = output_dir / "css-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional pilot/subset export; default is all 18 tasks",
    )
    args = parser.parse_args()
    tasks = tuple(args.tasks) if args.tasks else ALL_TASKS
    manifest = build(args.replication_root, args.data_root, args.output_dir, tasks)
    print(
        json_text(
            {
                "panel_version": PANEL_VERSION,
                "tasks": len(tasks),
                "outputs": manifest["outputs"],
            }
        )
    )


if __name__ == "__main__":
    main()
