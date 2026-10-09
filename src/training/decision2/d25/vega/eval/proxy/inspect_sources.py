"""Print compact schemas of proxy source files (development aid for the builders)."""

from __future__ import annotations

import collections
import io
import json
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RAW = Path("/work021/artifacts/benchmark-suite")
api = HfApi()


def short(x, n=160):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s[:n].replace("\n", "\\n")


def show_records(name, recs):
    recs = list(recs)
    print(f"## {name}: n>={len(recs)}")
    for r in recs[:2]:
        if isinstance(r, dict):
            print("   ", {k: short(v) for k, v in r.items()})
        else:
            print("   ", short(r, 400))


def hf_files(repo, sub=""):
    try:
        files = [
            f
            for f in api.list_repo_files(repo, repo_type="dataset")
            if f.startswith(sub)
        ]
    except Exception as e:  # noqa: BLE001
        print(f"## {repo}: ERROR {e}")
        return []
    print(f"## {repo} files ({len(files)}): {files[:25]}")
    return files


def read_any(path):
    p = str(path)
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(p).to_pylist()
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if p.endswith(".json"):
        x = json.load(open(p, encoding="utf-8"))
        return (
            x
            if isinstance(x, list)
            else [x] if not isinstance(x, dict) or len(x) < 50 else list(x.items())
        )
    if p.endswith(".csv") or p.endswith(".tsv"):
        import csv

        return list(
            csv.DictReader(
                open(p, encoding="utf-8", newline=""),
                delimiter="\t" if p.endswith(".tsv") else ",",
            )
        )
    return [open(p, encoding="utf-8", errors="replace").read(400)]


def hf_peek(repo, patterns, limit=3):
    files = hf_files(repo)
    shown = 0
    for f in files:
        if any(f.endswith(p) or p in f for p in patterns) and shown < limit:
            try:
                path = hf_hub_download(repo, f, repo_type="dataset")
                show_records(f"{repo}/{f}", read_any(path))
            except Exception as e:  # noqa: BLE001
                print(f"## {repo}/{f}: ERROR {e}")
            shown += 1


def main():
    which = sys.argv[1:] or ["hf", "local"]
    if "local" in which:
        lv3 = RAW / "raw/repos/apibank/api-bank"
        print(
            "## api-bank lv3:",
            sorted(p.name for p in (lv3 / "lv3-samples").iterdir())[:10],
            len(list((lv3 / "lv3_apis").glob("*.py"))),
        )
        for p in sorted((lv3 / "lv3-samples").rglob("*"))[:2]:
            if p.is_file():
                print("   ", p, short(p.read_text()[:600], 600))
        print(
            "## api-bank data dir:",
            (
                [
                    str(p.relative_to(lv3))
                    for p in sorted((lv3 / "data").rglob("*"))[:20]
                ]
                if (lv3 / "data").exists()
                else None
            ),
        )
        import pyarrow.parquet as pq

        hab = RAW / "raw/repos/habermas_machine/hm_all_candidate_comparisons.parquet"
        t = pq.read_table(
            hab,
            columns=[
                "metadata.version",
                "question.split",
                "iteration_index",
                "metadata.status",
            ],
        ).to_pylist()
        print(
            "## habermas versions:",
            collections.Counter(
                (r["metadata.version"][:12], r["question.split"]) for r in t
            ).most_common(20),
        )
        for sub in ("bright/examples", "toolret_queries"):
            ps = sorted((RAW / "raw" / sub).rglob("*.parquet"))
            print(f"## {sub}: {len(ps)} files")
            show_records(str(ps[0]), pq.read_table(ps[0]).to_pylist()[:2])
        subs = json.load(
            open(
                "/data/d25/shared/decision-index-kit/decision_index/data/release-v2/retrieval-subsets.json"
            )
        )
        print(
            "## retrieval-subsets keys:",
            {k: (type(v).__name__, short(v, 300)) for k, v in subs.items()},
        )
        with zipfile.ZipFile(RAW / "raw/repos/nli4ct/training_data.zip") as z:
            names = [
                n
                for n in z.namelist()
                if n.endswith(".json") and "MACOSX" not in n and "CT json" not in n
            ]
            print("## nli4ct zip:", names)
            for n in names:
                d = json.loads(z.read(n))
                k = next(iter(d))
                print("   ", n, len(d), k, short(d[k], 300))
        es = pq.read_table(
            RAW
            / "raw/repos/esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
        )
        print("## esci columns", es.column_names, es.num_rows)
        cl = json.load(open(RAW / "raw/cladder/data/cladder-v1-questions.json"))
        print("## cladder-v1-questions", type(cl).__name__, len(cl), short(cl[0], 500))
        with zipfile.ZipFile(RAW / "raw/cladder/data/cladder-v1.zip") as z:
            print("## cladder zip", z.namelist())
        print(
            "## clinc keys",
            {
                k: len(v)
                for k, v in json.load(open(RAW / "raw/clinc150/data_full.json")).items()
            },
        )
        print(
            "## hover train",
            short(
                json.load(
                    open(
                        RAW
                        / "raw/repos/cand-hover/data/hover/hover_train_release_v1.1.json"
                    )
                )[0],
                400,
            ),
        )
        import csv

        print(
            "## isarcasm train cols",
            next(
                csv.reader(
                    open(
                        RAW / "raw/repos/isarcasm/train/train.En.csv",
                        encoding="utf-8-sig",
                    )
                )
            ),
        )
        print(
            "## vast dev cols",
            next(
                csv.reader(
                    open(RAW / "raw/vast/data/VAST/vast_dev.csv", encoding="utf-8")
                )
            ),
        )
        with zipfile.ZipFile(RAW / "raw/downloads/humicroedit-full.zip") as z:
            print(
                "## humicroedit dev",
                z.read("semeval-2020-task-7-dataset/subtask-2/dev.csv").decode()[:300],
            )
        print(
            "## bfcl possible_answer:",
            sorted(
                p.name
                for p in (
                    RAW
                    / "raw/repos/bfcl/berkeley-function-call-leaderboard/data/possible_answer"
                ).glob("*.json")
            ),
        )
        print(
            "## phish questions file exists:",
            (RAW / "raw/repos/jev-phishing-bench/run_jev.py").exists(),
        )
        print(
            "## bag size",
            (RAW / "raw/downloads/chessbench-test-action-value.bag").stat().st_size,
        )
    if "hf" in which:
        hf_peek(
            "SeacowX/OpenToM",
            [
                "opentom_data/location_fg_fo_new.json",
                "opentom_data/metadata.json",
                "opentom_long.json",
            ],
        )
        hf_peek(
            "hubert233/BigBenchExtraHard",
            [
                "disambiguation_qa",
                "boardgame_qa",
                "web_of_lies",
                "causal_understanding",
                "geometric_shapes",
                "hyperbaton",
                "movie_recommendation",
                "sportqa",
                "temporal_sequence",
                "spatial_reasoning",
                "zebra_puzzles",
                "shuffled_objects",
                "object_properties",
            ],
            limit=13,
        )
        hf_peek(
            "nvidia/When2Call",
            ["train_pref.jsonl", "test_llm_judge.jsonl", "train_sft.jsonl"],
        )
        hf_peek(
            "AreLit/PhishNChips",
            [
                "cross_domain_legitimate_v5.csv",
                "infrastructure_phishing_expanded.csv",
                "real_phishing_validation.csv",
                "core_emails.csv",
            ],
            limit=4,
        )
        hf_peek("pauri32/fiqa-2018", [".parquet", ".json", ".csv"])
        hf_peek("livecodebench/execution-v2", [".parquet"])
        hf_peek("m-a-p/SuperGPQA", [".jsonl"])
        hf_peek(
            "tasksource/bigbench",
            [
                "navigate/train",
                "logical_deduction/train",
                "date_understanding/validation",
            ],
        )
        hf_peek("aps/super_glue", ["multirc/validation"])
        for repo in [
            "ucirvine/reuters21578",
            "NLP-AUEB/eurlex",
            "coastalcph/multi_eurlex",
            "CohereLabs/include-base-44",
            "nguha/legalbench",
            "allenai/reward-bench",
            "lmsys/mt_bench_human_judgments",
            "DFKI-SLT/cross_re",
            "mmathys/openai-moderation-api-evaluation",
            "Paul/XSTest",
            "Paul/hatecheck",
            "pminervini/HaluEval",
            "Salesforce/summedits",
            "tdiggelm/climate_fever",
            "wenhu/tab_fact",
            "hkust-nlp/felm",
            "google/code_x_glue_cc_clone_detection_big_clone_bench",
            "SetFit/20_newsgroups",
            "tuetschek/atis",
            "akariasai/PopQA",
            "truthfulqa/truthful_qa",
        ]:
            hf_peek(repo, [".parquet", ".jsonl", ".json", ".csv", ".tsv"], limit=1)


if __name__ == "__main__":
    main()
