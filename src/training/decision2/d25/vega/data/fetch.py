"""Download pinned raw sources into DATA_ROOT/raw (run inside a pod; resumable)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from d25.vega.data.util import DATA_ROOT, write_json

TASKSOURCE = (
    "tasksource/tasksource-jev-typed-decisions",
    "5c3e4ebb6e7120d06d8e2c22ff3047c6fd2f00ae",
)
PROCEDURAL = (
    "tasksource/procedural-typed-decisions",
    "609513a3faddf729123266efe9861456fe748f95",
)
D20 = ("vllm-sr/decision-2.0-training-data", "4785575188dc03608408b93f9dadfb2e41bcf677")

D20_POOLS = {
    "A0s-strict": "m3/pk1/A0s-strict/train.jsonl",
    **{
        p: f"m2/arms/{p}/train.jsonl"
        for p in ("H1", "H3", "H5", "H6", "E11", "G2", "G6", "G4h")
    },
    **{
        f"V1:{p}": f"v2/arms/{p}/train.jsonl"
        for p in ("A1", "A2", "A3", "A4v2h", "A5", "A6g", "A6h")
    },
    **{
        p: f"v2/a7/arms/{p}/train.jsonl"
        for p in ("A7g", "A7h", "A7i", "A7k", "A7m", "A7o", "A7p", "A7q", "A7r", "A7s")
    },
    **{p: f"m3/arms/{p}/train.jsonl" for p in ("H7", "H8")},
}
D20_EXTRA = {
    "IB1": ("m6/ib1/ib1.train.jsonl", "m6/ib1/ib1.dev.jsonl"),
    "IB2": ("m6/ib2/ib2.train.jsonl", "m6/ib2/ib2.dev.jsonl"),
    "IB3": ("m6/ib3/ib3.train.jsonl", "m6/ib3/ib3.dev.jsonl"),
    "IB4": ("m6/ib4/p1/ib4.train.jsonl", "m6/ib4/p1/ib4.dev.jsonl"),
    "HS1": ("m4/hs1/train.jsonl", "m4/hs1/dev.jsonl"),
    "PN1": ("m4/pn1/arms/pn1.train.jsonl", "m4/pn1/dev/pn1.dev.jsonl"),
}


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - network retries
            if attempt == 5:
                raise
            print(f"retry {attempt + 1}: {type(error).__name__}: {error}", flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def fetch_core(raw: Path) -> dict:
    from huggingface_hub import snapshot_download

    report = {}
    repo, rev = TASKSOURCE
    path = retry(
        snapshot_download,
        repo,
        repo_type="dataset",
        revision=rev,
        local_dir=str(raw / "tasksource"),
        allow_patterns=[
            "filtered-full/*",
            "sources.yaml",
            "README.md",
            "AGENTS.md",
            "build-manifest.json",
        ],
        max_workers=8,
    )
    report["tasksource"] = {"repo": repo, "revision": rev, "path": path}
    repo, rev = PROCEDURAL
    path = retry(
        snapshot_download,
        repo,
        repo_type="dataset",
        revision=rev,
        local_dir=str(raw / "procedural"),
        allow_patterns=["all/*", "README.md"],
        max_workers=8,
    )
    report["procedural"] = {"repo": repo, "revision": rev, "path": path}
    repo, rev = D20
    files = list(D20_POOLS.values()) + [f for pair in D20_EXTRA.values() for f in pair]
    files += [
        "m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl",
        "m3/mixtures/xl-r2/README.md",
        "m3/mixtures/xl-r2/mx-xl-r2.manifest.json",
        "README.md",
        "v2/README.md",
        "v2/a7/README.md",
        "m6/ib1/README.md",
        "m6/ib2/README.md",
        "m6/ib3/README.md",
        "m6/ib4/p1/README.md",
        "m4/hs1/README.md",
        "m4/pn1/README.md",
    ]
    patterns = files + [
        "v2/arms/*/manifest*.json",
        "v2/a7/arms/*/manifest*.json",
        "m2/arms/*/train.manifest.json",
        "m3/arms/*/manifest*.json",
        "m6/*/manifest*.json",
        "m6/ib4/p1/manifest*.json",
        "m4/*/manifest*.json",
        "m6/*/registry.json",
        "m4/*/registry.json",
    ]
    path = retry(
        snapshot_download,
        repo,
        repo_type="dataset",
        revision=rev,
        local_dir=str(raw / "d20"),
        allow_patterns=patterns,
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )
    report["d20"] = {"repo": repo, "revision": rev, "path": path, "files": len(files)}
    return report


def fetch_gsm8k(raw: Path) -> dict:
    """GSM8K main train/test. Test problems feed the 0.3 suite: decontamination reference only, never trained."""
    from huggingface_hub import HfApi, snapshot_download

    repo = "openai/gsm8k"
    rev = HfApi().dataset_info(repo).sha
    path = retry(
        snapshot_download,
        repo,
        repo_type="dataset",
        revision=rev,
        local_dir=str(raw / "gsm8k"),
        allow_patterns=["main/*"],
    )
    import pyarrow.parquet as pq

    from d25.vega.data.util import write_jsonl

    counts = {}
    for split in ("train", "test"):
        files = sorted((raw / "gsm8k/main").glob(f"{split}-*.parquet"))
        rows = [r for f in files for r in pq.read_table(f).to_pylist()]
        counts[split] = write_jsonl(raw / f"gsm8k/{split}.jsonl", rows)
    test = [json.loads(line) for line in open(raw / "gsm8k/test.jsonl")]
    pseudo = (
        {
            "id": f"gsm8k-test:{i}",
            "benchmark": "GSM8K-test(0.3 source)",
            "state": r["question"],
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "What is the answer?",
                    "criteria": {r["answer"].split("####")[-1].strip(): None},
                }
            },
        }
        for i, r in enumerate(test)
    )
    write_jsonl(raw / "gsm8k/test-pseudo-suite.jsonl.gz", pseudo)
    return {"repo": repo, "revision": rev, "path": path, "rows": counts}


# M2 sources: (repo, revision or None = resolve main, allow_patterns, licence). Licences checked on the cards.
M2_HF = {
    "medmcqa": ("openlifescienceai/medmcqa", None, ["data/train-*"], "apache-2.0"),
    "aqua_rat": ("deepmind/aqua_rat", None, ["raw/train-*"], "apache-2.0"),
    "math_qa": ("allenai/math_qa", "refs/convert/parquet", ["*train*"], "apache-2.0"),
    "commonsense_qa": ("tau/commonsense_qa", None, ["data/train-*"], "mit"),
    "qasc": ("allenai/qasc", None, ["data/train-*"], "cc-by-4.0"),
    "ai2_arc": (
        "allenai/ai2_arc",
        None,
        ["ARC-Challenge/train-*", "ARC-Easy/train-*"],
        "cc-by-sa-4.0",
    ),
    "boolq": ("google/boolq", None, ["data/train-*"], "cc-by-sa-3.0"),
    "exams": (
        "mhardalov/exams",
        None,
        ["multilingual/train-*", "README.md"],
        "cc-by-sa-4.0",
    ),
    "winogrande": (
        "allenai/winogrande",
        None,
        ["winogrande_xl/train-*"],
        "apache-2.0 (allenai/winogrande GitHub LICENSE)",
    ),
    "hellaswag": ("Rowan/hellaswag", None, ["data/train-*"], "mit"),
    "clinc_oos": ("clinc/clinc_oos", None, ["plus/train-*", "README.md"], "cc-by-3.0"),
    "when2call": ("nvidia/When2Call", None, ["train/*", "README.md"], "cc-by-4.0"),
    "newyorker": (
        "jmhessel/newyorker_caption_contest",
        None,
        ["matching/train-*", "matching/validation-*", "README.md"],
        "cc-by-4.0",
    ),
    "glaive_fc": (
        "glaiveai/glaive-function-calling-v2",
        None,
        ["*.json"],
        "apache-2.0",
    ),
}
M2_URLS = {
    "banking77_train": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "cc-by-4.0",
    ),
    "banking77_categories": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json",
        "cc-by-4.0",
    ),
    "hover_train": (
        "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json",
        "cc-by-sa-4.0",
    ),
    "isarcasm_train_en": (
        "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
        "see repository",
    ),
}


def fetch_m2(raw: Path) -> dict:
    import hashlib
    import urllib.request

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    report: dict = {"hf": {}, "urls": {}}
    for name, (repo, rev, patterns, licence) in M2_HF.items():
        revision = rev or api.dataset_info(repo).sha
        path = retry(
            snapshot_download,
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(raw / "m2" / name),
            allow_patterns=patterns,
            max_workers=8,
        )
        report["hf"][name] = {
            "repo": repo,
            "revision": revision,
            "licence": licence,
            "path": path,
        }
        print(name, revision, flush=True)
    for name, (url, licence) in M2_URLS.items():
        dest = raw / "m2" / "urls" / Path(url).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            retry(urllib.request.urlretrieve, url, str(dest))
        report["urls"][name] = {
            "url": url,
            "licence": licence,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "path": str(dest),
        }
        print(name, report["urls"][name]["sha256"][:12], flush=True)
    return report


PROXY = (
    "vllm-sr/d25-vega-proxy",
    "pv1/protected-items.jsonl.gz",
    "51cc55feac7d77eaae6a2cf7b94acd01131f2d272f289b29cd891a063df5c8c0",
)


def fetch_proxy(raw: Path) -> dict:
    """ws-proxy's frozen proxy items (holdouts v2 protected_items): decontamination reference only."""
    import gzip
    import hashlib

    from huggingface_hub import HfApi, hf_hub_download

    repo, name, want = PROXY
    revision = HfApi(token=os.environ["HF_TOKEN"]).dataset_info(repo).sha
    path = retry(
        hf_hub_download,
        repo,
        name,
        repo_type="dataset",
        revision=revision,
        local_dir=str(raw / "proxy"),
        token=os.environ["HF_TOKEN"],
    )
    digest = hashlib.sha256(gzip.open(path, "rb").read()).hexdigest()
    if digest != want:
        raise SystemExit(f"protected items sha256 {digest} != holdouts.json {want}")
    return {
        "repo": repo,
        "revision": revision,
        "path": path,
        "uncompressed_sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", default="core")
    args = parser.parse_args()
    raw = DATA_ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    report = {
        "core": fetch_core,
        "gsm8k": fetch_gsm8k,
        "m2": fetch_m2,
        "proxy": fetch_proxy,
    }[args.what](raw)
    write_json(raw / f"fetch-{args.what}.json", report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
