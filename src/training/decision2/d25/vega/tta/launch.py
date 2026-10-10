"""Submit ``tta_job.sh`` Hugging Face Jobs (RTX PRO 6000, one model per job) for the permutation-average arm.

    python -m d25.vega.tta.launch --kit <kit copy> --run tta-lat --mode lat
    python -m d25.vega.tta.launch --kit <kit copy> --run tta-proxy --mode proxy --shards 12
    python -m d25.vega.tta.launch --kit <kit copy> --run tta-vision --mode vision --vision-dir vision/cvbench

The code snapshot (this ``d25`` tree plus the kit at its pinned commit, ``release.hf_job.snapshot``) goes to
``code/<sha256>.tar.gz`` in the PRIVATE work dataset; each job uploads its results to ``runs/<run>[-<i>]/``
there. ``HF_TOKEN`` from the environment (or the local login) is passed only as a job secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from d25.vega.release.hf_job import IMAGE, LATENCY_SHA, PARITY_SHA, snapshot

WORK = "vllm-sr/d25-vega-tta-work"
MODEL = "vllm-sr/d3"
REVISION = "216ba44f0494c922e16cde55f956710163d8158f"  # v3.0.2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True, choices=("lat", "par", "proxy", "vision"))
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument(
        "--only", default="", help="comma-separated shard indexes to (re)submit"
    )
    ap.add_argument("--vision-dir", default="")
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--flavor", default="rtx-pro-6000")
    ap.add_argument("--timeout", default="3h")
    ap.add_argument("--namespace", default="vllm-sr")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    data = snapshot(args.kit)
    digest = hashlib.sha256(data).hexdigest()
    code = f"code/{digest}.tar.gz"
    url = f"https://huggingface.co/datasets/{args.work}/resolve/main/{code}"
    fetch = f'curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o /tmp/code.tar.gz {url}'
    command = (
        "set -e; command -v curl > /dev/null || (apt-get update -qq && apt-get install -y -qq curl > /dev/null); "
        "pip install -q -U 'huggingface_hub>=1.0'; "
        f"for i in 1 2 3 4 5 6; do {fetch} && break; echo retry download; sleep 30; done; "
        f"echo '{digest}  /tmp/code.tar.gz' | sha256sum -c -; "
        "mkdir -p /tmp/src; tar xzf /tmp/code.tar.gz -C /tmp/src; bash /tmp/src/d25/vega/tta/tta_job.sh"
    )
    api = HfApi(token=token)
    if not args.dry_run:
        api.create_repo(args.work, repo_type="dataset", private=True, exist_ok=True)
        if api.repo_info(args.work, repo_type="dataset").private is not True:
            raise SystemExit(
                f"{args.work} is not private; refusing to stage suite-derived data there"
            )
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo=code,
            repo_id=args.work,
            repo_type="dataset",
            commit_message=f"code snapshot for {args.run}",
        )
    only = {int(i) for i in args.only.split(",") if i}
    submitted = []
    for index in range(args.shards):
        if only and index not in only:
            continue
        run = args.run if args.shards == 1 else f"{args.run}-{index:02d}"
        env = {
            "WORK": args.work,
            "RUN": run,
            "MODEL": args.model,
            "REVISION": args.revision,
            "MODE": args.mode,
            "SHARD": f"{index}/{args.shards}",
            "VISION_DIR": args.vision_dir,
            "LATENCY_SHA": LATENCY_SHA,
            "PARITY_SHA": PARITY_SHA,
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if args.dry_run:
            submitted.append({"run": run, "env": env})
            continue
        job = api.run_job(
            image=IMAGE,
            command=["bash", "-c", command],
            env=env,
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,
            timeout=args.timeout,
            namespace=args.namespace,
            labels={"d25-campaign": "vega", "d25-ws": "tta", "run": run},
        )
        submitted.append({"run": run, "job_id": getattr(job, "id", None)})
    print(
        json.dumps({"code": code, "code_bytes": len(data), "jobs": submitted}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
