"""Runner queue items for the family (``d25.omni.runner`` format), written as JSON files.

    python -m d25.family.items stage-t --size mini --arm t1 --out DIR [--lr 5e-6] [--data DIR]
    python -m d25.family.items evals --size mini --arm t1 --out DIR [--full]

``stage-t``: fetch the base (once per node), then d3's recipe through ``d25.family.train`` (M2T-d3, one epoch,
noncausal full attention, exports at half and full horizon). ``evals``: for every export of the arm, Vega's
text proxies and the Omni vision suite + proxies; ``--full`` adds the complete public text kit run on the
final export. Paths are node paths under ``/data/d25/omni/family``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from d25.family.sizes import SIZES, Size

FAMILY = "/data/d25/omni/family"
MODELS = "/data/d25/omni/models/family"
DATA = f"{FAMILY}/data/v1/M2T-d3"
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


def model_dir(size: Size) -> str:
    """``Qwen/Qwen3.5-9B`` -> ``<MODELS>/qwen35-9b`` (the names the zero-shot baselines used)."""
    return f"{MODELS}/{size.base.split('/')[-1].lower().replace('qwen3.5', 'qwen35')}"


def fetch_step(size: Size) -> dict:
    target = model_dir(size)
    return {
        "run": (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "from huggingface_hub import snapshot_download\n"
            f"out = Path('{target}')\n"
            "if not (out / 'FETCHED').exists():\n"
            f"    snapshot_download(repo_id='{size.base}', revision='{size.revision}', local_dir=str(out), max_workers=8)\n"
            f"    (out / 'FETCHED').write_text('{size.base}@{size.revision}\\n')\n"
            "PY"
        ),
        "creates": f"{target}/FETCHED",
        "timeout_h": 1,
    }


def run_name(size: Size, arm: str) -> str:
    return f"{size.short}-{arm}"


def stage_t(size: Size, arm: str, lr: float | None, data: str) -> dict:
    run = run_name(size, arm)
    out = f"{FAMILY}/ckpt/{run}"
    lr = lr or size.lr
    train = (
        f"python -m d25.vega.train.launch --nproc $D25_GPUS --log-dir {out}/launch -- d25.family.train "
        f"--base-model {size.base} --base-revision {size.revision} "
        f"--run {run} --train {data} --dev {data}/dev.jsonl.gz --output {out} "
        f"--cache-dir {FAMILY}/train-cache --init {model_dir(size)} --init-kind base "
        f"--attention-mode noncausal_full_attention --lr {lr:g} --readout-lr {lr:g} "
        f"--warmup-ratio 0.15 --rows-per-update 256 --max-length 8192 "
        f"--token-budget {size.token_budget} --max-rows-per-micro 256 --save-every 500 --keep-dcp 1 --final-dcp off "
        f"--export-fractions 0.5,1.0 && date -u > {out}/TRAIN_DONE"
    )
    return {
        "name": f"70-train-{run}",
        "max_attempts": 3,
        "wait_for": [f"{data}/manifest.json"],
        "defer": "skip",
        "steps": [
            fetch_step(size),
            {"run": train, "creates": f"{out}/TRAIN_DONE", "timeout_h": 30},
        ],
    }


def evals(size: Size, arm: str, full: bool, vision_suite: str) -> dict:
    run = run_name(size, arm)
    ckpt = f"{FAMILY}/ckpt/{run}"
    res = f"{FAMILY}/results/{run}"
    suites = " ".join(
        [f"--suite public={vision_suite}"]
        + [f"--suite {p}-proxy=/data/d25/omni/proxy/{p}-proxy" for p in VISION_PROXIES]
    )
    loop = (
        f"for C in $(ls -d {ckpt}/step-[0-9]* | grep -v -E 'partial|tmp' | sort); do S=$(basename $C); "
        f"test -e {res}/text/$S/result.json || python -m d25.vega.eval.ckpt_eval --ckpt $C "
        f"--out {res}/text/$S --gpus $D25_GPUS --what proxy --proxy {PV1} || exit 1; "
        f"test -e {res}/vision/$S/summary.json || python -m d25.omni.eval.eval_ckpt --ckpt $C "
        f"--out {res}/vision/$S --gpus $D25_GPUS {suites} || exit 1; done; date -u > {res}/EVALS_DONE"
    )
    steps = [{"run": loop, "creates": f"{res}/EVALS_DONE", "timeout_h": 6}]
    if full:
        steps.append(
            {
                "run": (
                    f"C=$(ls -d {ckpt}/step-[0-9]* | grep -v -E 'partial|tmp' | sort | tail -1); "
                    f"python -m d25.vega.eval.ckpt_eval --ckpt $C --out {res}/full/$(basename $C) "
                    f"--gpus $D25_GPUS --what full --proxy {PV1} && ln -sfn {res}/full/$(basename $C) {res}/full/final"
                ),
                "creates": f"{res}/full/final/result.json",
                "timeout_h": 8,
            }
        )
    return {
        "name": f"75-evals-{run}",
        "max_attempts": 2,
        "wait_for": [f"{ckpt}/TRAIN_DONE"],
        "defer": "skip",
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("stage-t")
    t.add_argument("--lr", type=float)
    t.add_argument("--data", default=DATA)
    e = sub.add_parser("evals")
    e.add_argument("--full", action="store_true")
    e.add_argument("--vision-suite", default=VISION_SUITE)
    for p in (t, e):
        p.add_argument("--size", required=True, choices=sorted(SIZES))
        p.add_argument("--arm", required=True)
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    size = SIZES[args.size]
    item = (
        stage_t(size, args.arm, args.lr, args.data)
        if args.cmd == "stage-t"
        else evals(size, args.arm, args.full, args.vision_suite)
    )
    path = Path(args.out) / f"{item['name']}.json"
    path.write_text(json.dumps(item, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
