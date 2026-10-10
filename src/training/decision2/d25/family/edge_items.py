"""Runner queue items for d3-edge (one GPU lane) and a results report.

    python -m d25.family.edge_items write --out DIR
    python -m d25.family.edge_items report [--out RESULTS/edge-summary.json]

Items, in queue order (paths are node paths; every item resumes after a restart):

- ``80-edge-init``: fetch Qwen3-0.6B-Base and Qwen3.5-0.8B at the pinned revisions, build the composite
  (``d25.family.edge build``) into ``/data/d25/omni/inits/family/d3-edge-init``.
- ``81-edge-pilot``: 30 E1 updates on M2-v5 shard 0 (gold), to measure speed and memory on the GPU.
- ``82-edge-e1``: ``edge bench`` (one update's forward + backward per batch layout), then E1 with the
  fastest layout into ``ckpt/edge-e1f``: full fine-tune of language model + readout on M2T-d3 (1 epoch,
  lr 1e-5, warmup 0.15, cosine to 0.1x, 256 rows per update, causal, max length 8192); waits for M2T-d3.
- ``83-edge-evals-e1``: evaluations of E1's middle and final checkpoints.
- ``84-edge-zs``: text proxies (pv1) and the vision suite + 7 proxies of the untrained composite (the
  baseline; runs while E1 waits for its data or after E1's evaluations).
- ``85-edge-e0``: E0, merger alignment (encoder, language model and readout frozen) on the mm-v1a image
  rows with d3 soft labels (gold until ``62-d3lab-mm`` has written them), from E1's final checkpoint;
  waits for mm-v1a READY. (E1 runs first because the image corpus is not ready; the merger is aligned
  to the language model it will be used with.)
- ``86-edge-e2``: E2, mm-v1a image rows + M2T-d3 replay at 0.5 (shards 0-2, ~146k rows, of which ~47k are
  used), merger + language model + readout, one pass over the image rows (~370 updates) at 0.3x lr, from
  E0's final checkpoint.
- ``87-edge-evals-e2``: evaluations of E0's and E2's final checkpoints.

``report`` collects every evaluated checkpoint's text O_proxy/S_proxy and vision public/proxy scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILY = "/data/d25/omni/family"
EDGE = f"{FAMILY}/edge"
MODELS = "/data/d25/omni/models/family"
TEXT_DIR = f"{MODELS}/qwen3-0.6b-base"
VISION_DIR = f"{MODELS}/qwen35-0.8b"
TEXT_BASE = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
VISION_BASE = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
INIT = "/data/d25/omni/inits/family/d3-edge-init"
CKPT = f"{FAMILY}/ckpt"
RESULTS = f"{FAMILY}/results"
M2T = f"{FAMILY}/data/v1/M2T-d3"
M2V5 = f"{FAMILY}/data/v1/M2-v5"
MMV1A = "/data/d25/omni/data/mm-v1a"
LABELS = f"{FAMILY}/teacher/d3-mmv1a/probs.jsonl"
MIXED = f"{EDGE}/data/mmv1a-d3"
PV1 = f"{FAMILY}/proxy/pv1"
VISION_SUITE = "/data/d25/omni/suite/vision-0.3.1b"
VISION_PROXIES = (
    "blink",
    "moderation",
    "charxiv",
    "cvbench",
    "infovqa",
    "kie",
    "mind2web",
)
E1_LR = 1e-5
COMMON = (
    "--effective-batch-size 256 --token-budget 49152 --max-rows-per-microbatch 64 "
    "--lr-floor 0.1 --resume-every 200 --code-tag $(basename $D25_CODE)"
)


def last_ckpt(run: str) -> str:
    return f"$(ls -d {CKPT}/{run}/checkpoints/step-[0-9]* | grep -v -E 'partial|tmp' | sort | tail -1)"


def train_files(directory: str) -> str:
    return f"$(ls {directory}/train-*.jsonl.gz | sort)"


def fetch(directory: str, repo: str, revision: str) -> dict:
    return {
        "run": (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "from huggingface_hub import snapshot_download\n"
            f"out = Path('{directory}')\n"
            "if not (out / 'FETCHED').exists():\n"
            f"    snapshot_download(repo_id='{repo}', revision='{revision}', local_dir=str(out), max_workers=8)\n"
            f"    (out / 'FETCHED').write_text('{repo}@{revision}\\n')\n"
            "PY"
        ),
        "creates": f"{directory}/FETCHED",
        "timeout_h": 1,
    }


def evals_step(checkpoints: list[str], results: str, marker: str) -> dict:
    suites = " ".join(
        [f"--suite public={VISION_SUITE}"]
        + [f"--suite {p}-proxy=/data/d25/omni/proxy/{p}-proxy" for p in VISION_PROXIES]
    )
    loop = (
        f"test -d {PV1} || {{ echo 'missing {PV1} (copy of Vega pv1)'; exit 3; }}; "
        f"for C in {' '.join(checkpoints)}; do S=$(basename $C); "
        f"test -e {results}/text/$S/result.json || python -m d25.vega.eval.ckpt_eval --ckpt $C "
        f"--out {results}/text/$S --gpus $D25_GPUS --what proxy --proxy {PV1} || exit 1; "
        f"test -e {results}/vision/$S/summary.json || python -m d25.omni.eval.eval_ckpt --ckpt $C "
        f"--out {results}/vision/$S --gpus $D25_GPUS {suites} || exit 1; done; "
        f"python -m d25.family.edge_items report && date -u > {marker}"
    )
    return {"run": loop, "creates": marker, "timeout_h": 8}


def train_step(run: str, args: str, timeout_h: float, extra: str = "") -> dict:
    """``extra`` comes after the common flags, so its ``--token-budget`` wins."""
    out = f"{CKPT}/{run}"
    return {
        "run": (
            f"python -m d25.family.edge_train {args} {COMMON} {extra} --out {out} "
            f"&& date -u > {out}/TRAIN_DONE"
        ),
        "creates": f"{out}/TRAIN_DONE",
        "timeout_h": timeout_h,
    }


def items() -> list[dict]:
    e1, e0, e2 = "edge-e1f", "edge-e0", "edge-e2"
    m2t_dev = f"{M2T}/dev.jsonl.gz"
    bench = f"{EDGE}/bench-e1.json"
    return [
        {
            "name": "80-edge-init",
            "max_attempts": 2,
            "steps": [
                fetch(TEXT_DIR, *TEXT_BASE),
                fetch(VISION_DIR, *VISION_BASE),
                {
                    "run": (
                        f"rm -rf {INIT}.partial && python -m d25.family.edge build --text {TEXT_DIR} "
                        f"--vision {VISION_DIR} --out {INIT}.partial && mv -T {INIT}.partial {INIT}"
                    ),
                    "creates": f"{INIT}/decision_config.json",
                    "timeout_h": 0.5,
                },
            ],
        },
        {
            "name": "81-edge-pilot",
            "max_attempts": 2,
            "wait_for": [
                f"{INIT}/decision_config.json",
                f"{M2V5}/train-00000-of-00027.jsonl.gz",
            ],
            "defer": "skip",
            "steps": [
                train_step(
                    "edge-pilot",
                    f"--init {INIT} --arm E1-text --rows {M2V5}/train-00000-of-00027.jsonl.gz "
                    f"--replay-ratio 0 --dev-rows {M2V5}/dev.jsonl.gz --lr {E1_LR:g} --warmup-ratio 0.15 "
                    "--max-length 8192 --save-every 1000 --eval-every 30 --stop-after 30",
                    1.5,
                )
            ],
        },
        {
            "name": "82-edge-e1",
            "max_attempts": 3,
            "wait_for": [f"{INIT}/decision_config.json", f"{M2T}/manifest.json"],
            "defer": "skip",
            "steps": [
                {
                    "run": (
                        f"python -m d25.family.edge bench --ckpt {INIT} --rows {M2T}/train-00000-of-00027.jsonl.gz "
                        f"--report {bench} || true"
                    ),
                    "creates": bench,
                    "timeout_h": 0.5,
                },
                train_step(
                    e1,
                    f"--init {INIT} --arm E1-text --rows {train_files(M2T)} --replay-ratio 0 "
                    f"--dev-rows {m2t_dev} --lr {E1_LR:g} --warmup-ratio 0.15 --max-length 8192 "
                    "--save-every 1000 --eval-every 500",
                    20,
                    f"$(python -m d25.family.edge bench-choice --report {bench})",
                ),
            ],
        },
        {
            "name": "83-edge-evals-e1",
            "max_attempts": 2,
            "wait_for": [f"{CKPT}/{e1}/TRAIN_DONE"],
            "defer": "skip",
            "steps": [
                evals_step(
                    [f"{CKPT}/{e1}/checkpoints/step-03000", last_ckpt(e1)],
                    f"{RESULTS}/{e1}",
                    f"{RESULTS}/{e1}/EVALS_DONE",
                )
            ],
        },
        {
            "name": "84-edge-zs",
            "max_attempts": 2,
            "wait_for": [f"{INIT}/decision_config.json"],
            "defer": "skip",
            "steps": [
                evals_step(
                    [INIT], f"{RESULTS}/edge-init", f"{RESULTS}/edge-init/EVALS_DONE"
                )
            ],
        },
        {
            "name": "85-edge-e0",
            "max_attempts": 2,
            "wait_for": [f"{MMV1A}/READY", f"{CKPT}/{e1}/TRAIN_DONE"],
            "defer": "skip",
            "steps": [
                {
                    "run": (
                        f"python -m d25.family.edge_data mix --rows $(ls {MMV1A}/*.jsonl.gz | grep -v -E 'dev|holdout' | sort) "
                        f"--out {MIXED} --probs {LABELS}"
                    ),
                    "creates": f"{MIXED}/manifest.json",
                    "timeout_h": 1,
                },
                train_step(
                    e0,
                    f"--init {last_ckpt(e1)} --arm E0-merger --rows $(ls {MIXED}/*.jsonl.gz | sort) "
                    "--replay-ratio 0 --lr 1e-4 --warmup-ratio 0.05 --epochs 2 --max-length 16384 "
                    "--save-every 1000 --eval-every 1000",
                    6,
                ),
            ],
        },
        {
            "name": "86-edge-e2",
            "max_attempts": 2,
            "wait_for": [f"{CKPT}/{e0}/TRAIN_DONE"],
            "defer": "skip",
            "steps": [
                train_step(
                    e2,
                    f"--init {last_ckpt(e0)} --arm E2-joint --rows $(ls {MIXED}/*.jsonl.gz | sort) "
                    f"--replay-rows {' '.join(f'{M2T}/train-{i:05d}-of-00027.jsonl.gz' for i in range(3))} "
                    f"--replay-ratio 0.5 --dev-rows {m2t_dev} "
                    f"--lr {0.3 * E1_LR:g} --merger-lr-scale 3 --warmup-ratio 0.05 --epochs 1 "
                    "--max-length 16384 --save-every 200 --eval-every 100",
                    8,
                )
            ],
        },
        {
            "name": "87-edge-evals-e2",
            "max_attempts": 2,
            "wait_for": [f"{CKPT}/{e2}/TRAIN_DONE"],
            "defer": "skip",
            "steps": [
                evals_step(
                    [last_ckpt(e0), last_ckpt(e2)],
                    f"{RESULTS}/{e2}",
                    f"{RESULTS}/{e2}/EVALS_DONE",
                )
            ],
        },
    ]


def report(out: Path) -> dict:
    rows = {}
    for result in sorted(Path(RESULTS).glob("edge-*/text/*/result.json")):
        proxy = json.loads(result.read_text()).get("proxy") or {}
        key = f"{result.parents[2].name}/{result.parent.name}"
        rows.setdefault(key, {}).update(
            {"O_proxy": proxy.get("O_proxy"), "S_proxy": proxy.get("S_proxy")}
        )
    for summary in sorted(Path(RESULTS).glob("edge-*/vision/*/summary.json")):
        suites = json.loads(summary.read_text()).get("suites") or {}
        key = f"{summary.parents[2].name}/{summary.parent.name}"
        rows.setdefault(key, {}).update(
            {
                name: (
                    (v.get("scores") or {}).get("public") if isinstance(v, dict) else v
                )
                for name, v in suites.items()
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out", default=f"{RESULTS}/edge-summary.json")
    args = parser.parse_args()
    if args.cmd == "report":
        print(json.dumps(report(Path(args.out)), indent=1, sort_keys=True))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in items():
        path = out / f"{item['name']}.json"
        path.write_text(json.dumps(item, indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
