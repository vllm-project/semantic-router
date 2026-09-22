"""Benchmark-specific source normalization; labels stay outside subject input."""

from __future__ import annotations

import json
import random

from .contracts import digest
from .task_identity import native_task_identity


def _case(benchmark, identity, prompt, answer=None, metadata=None):
    case = {
        "id": f"{benchmark}/{identity}",
        "benchmark": benchmark,
        "messages": [{"role": "user", "content": prompt}],
        "metadata": metadata or {},
    }
    if answer is not None:
        case["answer"] = answer
    return case


def _mcq(prompt, options):
    return (
        prompt
        + "\n\n"
        + "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
        + "\n\nReturn only the letter of the correct answer."
    )


def normalize_records(benchmark, rows, seed, *, task_source=None):
    result = []
    for index, row in enumerate(rows):
        # A normalized import is useful for offline preparation, but is never
        # accepted under another benchmark identity.
        if row.get("benchmark") == benchmark and "messages" in row:
            if task_source is not None:
                raise ValueError(
                    "Normalized imports have no verified native task identity"
                )
            result.append(row)
            continue
        identity = str(
            row.get("id", row.get("question_id", row.get("problem_id", index)))
        )
        if benchmark == "mmlu-pro":
            result.append(
                _case(
                    benchmark,
                    identity,
                    _mcq(row["question"], row["options"]),
                    row["answer"],
                    {"stratum": row["category"]},
                )
            )
        elif benchmark == "gpqa-diamond":
            identity = digest(row["Question"])[:24]
            choices = [(row["Correct Answer"], True)] + [
                (row[f"Incorrect Answer {i}"], False) for i in range(1, 4)
            ]
            random.Random(digest([seed, identity])).shuffle(choices)
            answer = chr(
                65 + next(i for i, (_, correct) in enumerate(choices) if correct)
            )
            result.append(
                _case(
                    benchmark,
                    identity,
                    _mcq(row["Question"], [c[0] for c in choices]),
                    answer,
                    {
                        "stratum": row.get("Subdomain", "all"),
                        "retest_notice": "Public labels may have been previously inspected; not an unseen-data claim.",
                    },
                )
            )
        elif benchmark == "hle":
            if row.get("image"):
                continue
            result.append(
                _case(
                    benchmark,
                    identity,
                    row["question"],
                    row["answer"],
                    {
                        "stratum": row.get("category", "all"),
                        "answer_type": row.get("answer_type"),
                    },
                )
            )
        elif benchmark == "simpleqa-verified":
            prompt = row.get("problem", row.get("question"))
            if not prompt:
                raise ValueError("SimpleQA source lacks question/problem")
            result.append(
                _case(
                    benchmark,
                    identity,
                    prompt,
                    row["answer"],
                    {"stratum": row.get("topic", "all")},
                )
            )
        elif benchmark == "arc-agi-2":
            # Multiple test inputs belong to one puzzle and one score.
            prompt = (
                "Infer the grid transformation from the examples. Return a JSON array containing one output grid for each test input, in order.\n"
                + json.dumps(
                    {
                        "train": row["train"],
                        "test": [{"input": t["input"]} for t in row["test"]],
                    },
                    separators=(",", ":"),
                )
            )
            result.append(
                _case(
                    benchmark,
                    identity,
                    prompt,
                    [t["output"] for t in row["test"]],
                    {"stratum": "public-evaluation", "output_format": "grids"},
                )
            )
        elif benchmark == "livecodebench":
            prompt = (
                row["question_content"]
                + "\n\nWrite a Python solution. Return the solution in one Python code block."
            )
            if row.get("starter_code"):
                prompt += "\n\nStarter code:\n" + row["starter_code"]
            result.append(
                _case(
                    benchmark,
                    identity,
                    prompt,
                    metadata={
                        "stratum": row.get("difficulty", "all"),
                        "source_record": row,
                    },
                )
            )
        elif benchmark == "scicode":
            result.append(
                _case(
                    benchmark,
                    identity,
                    "",
                    metadata={
                        "stratum": row.get("domain", "all"),
                        "source_record": row,
                    },
                )
            )
        elif benchmark == "tau3":
            result.append(
                _case(
                    benchmark,
                    row["domain"] + "/" + identity,
                    "",
                    metadata={
                        "stratum": row["domain"],
                        "domain": row["domain"],
                        "task_id": identity,
                        "source_task_sha256": digest(
                            {k: v for k, v in row.items() if k != "domain"}
                        ),
                    },
                )
            )
        elif benchmark == "terminal-bench-2.1":
            result.append(
                _case(
                    benchmark,
                    identity,
                    "",
                    metadata={
                        "stratum": "terminal",
                        "task_path": row["task_path"],
                        "tree_sha256": row["tree_sha256"],
                    },
                )
            )
        else:
            raise ValueError("Unsupported source benchmark")
        if task_source is not None:
            result[-1]["metadata"]["task_identity"] = native_task_identity(
                benchmark, row, task_source
            )
    return result
