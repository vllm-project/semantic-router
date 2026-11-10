"""Rows for policy_eval.py from the public Terminal-Bench 2.0 leaderboard trajectories.

Source: yoonholee/terminalbench-trajectories (Apache-2.0, scraped from the
tbench.ai leaderboard), pinned to a revision. Only the terminus-2 scaffold is
used, so every model runs under the same agent, with about five trials per
task and model.

A row is one trial of a cheap model on a task. Its attempts are that trial, the
mean of the cheap model's other trials on the task (`<cheap>#retry`, what a
fresh retry is expected to do), and each strong model's mean outcome and cost
over its own trials, since the strong model's runs do not depend on the cheap
trial. Trials without an agent step are skipped. Row ids are
`<task>::<cheap model>::<trial name>`, so scores computed elsewhere can be
joined to them.

Usage: python prepare_terminal_bench.py --out rows.jsonl [--parquet-dir DIR]
Without --parquet-dir the two parquet files are downloaded at the pinned
revision (needs huggingface_hub). Reading them needs pandas and pyarrow.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = "yoonholee/terminalbench-trajectories"
REVISION = "04e8940f5b6736a7ce8d22224fe2f2af74163ed2"
FILES = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
SCAFFOLD = "terminus-2"
CHEAP = ("deepseek-v3.2@deepseek", "gpt-5-mini@openai")
STRONG = (
    "claude-opus-4-6@anthropic",
    "gemini-3-pro-preview@gemini",
    "gpt-5.1@openai",
    "claude-opus-4-5-20251101@anthropic",
)
COLUMNS = ["task_name", "agent", "model", "reward", "cost_cents", "trial_name", "steps"]


def short(model):
    return model.split("@")[0]


def has_agent_step(steps):
    if steps in (None, "null"):
        return False
    return any(step.get("src") == "agent" for step in json.loads(steps))


def build_rows(records):
    """Rows, as policy_eval.py reads them, from parquet records with COLUMNS."""
    trials = defaultdict(list)
    for record in records:
        if record["agent"] == SCAFFOLD:
            trials[(record["model"], record["task_name"])].append(record)
    strong = {
        (m, task): (
            mean([float(t["reward"]) for t in ts]),
            mean([float(t["cost_cents"]) for t in ts]) / 100,
        )
        for (m, task), ts in trials.items()
        if m in STRONG
    }
    rows = []
    for model in CHEAP:
        for task in sorted(task for m, task in trials if m == model):
            if any((m, task) not in strong for m in STRONG):
                continue
            group = trials[(model, task)]
            for i, trial in enumerate(group):
                if not has_agent_step(trial["steps"]):
                    continue
                others = group[:i] + group[i + 1 :]
                attempts = {
                    short(model): {
                        "ok": float(bool(trial["reward"])),
                        "cost": trial["cost_cents"] / 100,
                    },
                    **{
                        short(m): {
                            "ok": strong[(m, task)][0],
                            "cost": strong[(m, task)][1],
                        }
                        for m in STRONG
                    },
                }
                if others:
                    attempts[short(model) + "#retry"] = {
                        "ok": mean([float(o["reward"]) for o in others]),
                        "cost": mean([float(o["cost_cents"]) for o in others]) / 100,
                    }
                rows.append(
                    {
                        "id": f"{task}::{short(model)}::{trial['trial_name'] or i}",
                        "task": task,
                        "attempts": attempts,
                    }
                )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--parquet-dir",
        help="directory holding the two parquet files already downloaded",
    )
    a = ap.parse_args(argv)
    import pandas as pd  # noqa: PLC0415

    if a.parquet_dir:
        paths = [Path(a.parquet_dir) / Path(f).name for f in FILES]
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        paths = [
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)
            for f in FILES
        ]
    rows = build_rows(
        pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in paths]).to_dict(
            "records"
        )
    )
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(
        f"{len(rows)} rows over {len({r['task'] for r in rows})} tasks written to {a.out}"
    )


if __name__ == "__main__":
    main()
