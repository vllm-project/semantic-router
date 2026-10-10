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
VISION_PROXIES = {
    "blink": "blink-proxy-1500",
    "moderation": "moderation-proxy",
    "charxiv": "charxiv-proxy",
    "cvbench": "cvbench-proxy",
    "infovqa": "infovqa-proxy",
    "kie": "kie-proxy",
    "mind2web": "mind2web-proxy",
}


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
        + [
            f"--suite {p}-proxy=/data/d25/omni/proxy/{d}"
            for p, d in VISION_PROXIES.items()
        ]
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


PPLX_PUBLIC = "/data/d25/omni/results/cx/pplx-v1.1/public-11/results.jsonl"
PPLX_PROXY = {
    "blink": "/data/d25/omni/imports/n03/results/enl/pplx-v1.1/blink-proxy/results.jsonl",
    "moderation": "/data/d25/omni/imports/n03/results/zs/pplx-v1.1/moderation-proxy/results.jsonl",
}
PROXY_BENCH = {
    "blink": "BLINK",
    "moderation": "Moderation (Hateful Memes)",
    "cvbench": "CV-Bench",
    "charxiv": "CharXiv",
    "infovqa": "InfographicVQA",
    "kie": "KIE (CORD+FUNSD)",
    "mind2web": "Mind2Web",
}
TEXT_REF = {
    "flash": "lux2",
    "mini": "nox2",
    "nano": "sol2",
    "lite": "eos2",
    "edge": "kai2",
}


def gates(size: Size, arm: str) -> dict:
    """Paired estimates of the arm's final export: vision vs pplx (node 02 references), text vs a same-size anchor."""
    run = run_name(size, arm)
    res = f"{FAMILY}/results/{run}"
    last = f"S=$(ls -d {FAMILY}/ckpt/{run}/step-[0-9]* | grep -v -E 'partial|tmp' | sort | tail -1 | xargs basename)"
    proxies = " ".join(
        f'--proxy "{bench}=/data/d25/omni/proxy/{VISION_PROXIES[p]}/rows.jsonl.gz,'
        f"{res}/vision/$S/{p}-proxy/results.jsonl,"
        + PPLX_PROXY.get(
            p,
            f"/data/d25/omni/imports/n03/results/zs5/pplx-v1.1/{p}-proxy/results.jsonl",
        )
        + '"'
        for p, bench in PROXY_BENCH.items()
    )
    vision = (
        f"{last}; python -m d25.omni.proxy.paired --board /data/d25/omni/suite/vision-live-20261009T2353Z.json "
        f"--reference-name 'Perplexity Decider v1.1 (27B)' --rows {VISION_SUITE}/rows.jsonl.gz "
        f"--ours {res}/vision/$S/public/results.jsonl --ref {PPLX_PUBLIC} {proxies} --out {res}/vgate.json"
    )
    ref = TEXT_REF[size.short]
    o_ref = (
        ""
        if ref in ("lux2", "nox2")
        else f" --ref-o-proxy $(python -c \"import json; print(json.load(open('{FAMILY}/anchors/{ref}/scores.json'))['O_proxy'])\")"
    )
    text = f"python -m d25.family.gate --result {res}/full/final/result.json --ref {ref}{o_ref} --out {res}/tgate.json"
    return {
        "name": f"77-gates-{run}",
        "max_attempts": 2,
        "wait_for": [f"{res}/full/final/result.json"],
        "defer": "skip",
        "steps": [
            {"run": vision, "creates": f"{res}/vgate.json", "timeout_h": 0.5},
            {"run": text, "creates": f"{res}/tgate.json", "timeout_h": 0.5},
        ],
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
    g = sub.add_parser("gates")
    for p in (t, e, g):
        p.add_argument("--size", required=True, choices=sorted(SIZES))
        p.add_argument("--arm", required=True)
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    size = SIZES[args.size]
    if args.cmd == "stage-t":
        item = stage_t(size, args.arm, args.lr, args.data)
    elif args.cmd == "evals":
        item = evals(size, args.arm, args.full, args.vision_suite)
    else:
        item = gates(size, args.arm)
    path = Path(args.out) / f"{item['name']}.json"
    path.write_text(json.dumps(item, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
