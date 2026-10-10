"""Submit ``cuda_job.sh`` as one Hugging Face Job on an NVIDIA RTX PRO 6000 (stdlib + huggingface_hub).

    python -m d25.vega.release.hf_job --model standin --run pplx-standin-1 --kit <decision-index kit checkout>
    python -m d25.vega.release.hf_job --model vllm-sr/d25-vega-staging --revision <sha> --run cand-1 \
        --kit <kit> --reference runs/<mi325x run>/kit-760/results.jsonl.gz

The code snapshot (this ``d25`` tree plus the kit at its pinned commit) goes to ``code/<sha256>.tar.gz`` in the
PRIVATE work dataset; the job installs it, runs the steps of ``cuda_job.sh`` and uploads its results to
``runs/<run>/`` there (also when a step fails). ``HF_TOKEN`` from the environment is passed as a job secret.
``--dry-run`` prints the job without creating anything.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

D25 = Path(__file__).resolve().parents[2]
WORK = "vllm-sr/d25-vega-release-work"
FLAVOR = "rtx-pro-6000"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
KIT_COMMIT = "9eb2dbe2a358004c8782c66e40a83ac07b953fec"
# Run ids of the latency-v1-style samples of suite 0.3 (sample.py, seeds 20260926 and 20261010).
LATENCY_SHA = "4c7a49eb7a2369fbd61b530afcc63f5c45bd6eee13f7543dda2fd5cf46a4ca1e"
PARITY_SHA = "8f64cdd42ee6b690c359d0194e8d9ee991ea2b929072658cf6aa8a4d1132a890"


def snapshot(kit: Path, omni: Path | None = None) -> bytes:
    """d25/ (no caches) plus the kit at KIT_COMMIT under decision-index-kit/.

    ``kit`` is a git checkout (``git archive`` of KIT_COMMIT) or an exported copy whose ``COMMIT`` file names
    that commit (the nodes' ``/data/d25/shared/decision-index-kit``).
    """
    skip = lambda ti: (
        None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti
    )  # noqa: E731
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(D25, arcname="d25", filter=skip)
        if omni:
            tar.add(omni, arcname="d25/omni", filter=skip)
        if (kit / ".git").exists():
            archive = subprocess.run(
                [
                    "git",
                    "-C",
                    str(kit),
                    "archive",
                    "--format=tar",
                    "--prefix=decision-index-kit/",
                    KIT_COMMIT,
                ],
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as kit_tar:
                for member in kit_tar.getmembers():
                    tar.addfile(
                        member, kit_tar.extractfile(member) if member.isfile() else None
                    )
        else:
            commit = (
                (kit / "COMMIT").read_text().strip()
                if (kit / "COMMIT").exists()
                else ""
            )
            if not commit or not KIT_COMMIT.startswith(commit):
                raise SystemExit(
                    f"{kit} is not a git checkout and its COMMIT file does not name {KIT_COMMIT[:7]}"
                )
            tar.add(kit, arcname="decision-index-kit", filter=skip)
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model", required=True, help="Hub repo id of a package, or standin"
    )
    ap.add_argument("--revision", default="main")
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--kit",
        required=True,
        type=Path,
        help="decision-index git checkout holding KIT_COMMIT",
    )
    ap.add_argument("--work", default=WORK)
    ap.add_argument(
        "--reference",
        default="",
        help="path in the work dataset of an MI325X kit-760 results file",
    )
    ap.add_argument("--variants", default="batch16 batch64 noconv nofla")
    ap.add_argument("--variant-rows", type=int, default=150)
    ap.add_argument("--steps", default="smoke latency parity variants")
    ap.add_argument(
        "--image-checks",
        default="smoke parity latency",
        help="checks of the images step",
    )
    ap.add_argument(
        "--probe-ids", default="", help="run ids for latency_probe.py (space separated)"
    )
    ap.add_argument("--probe-runtime", default="package", choices=("package", "tree"))
    ap.add_argument(
        "--torch", default="", help='pip spec installed first, e.g. "torch==2.14.0"'
    )
    ap.add_argument(
        "--torch-index",
        default="",
        help="pip index for --torch, e.g. https://download.pytorch.org/whl/cu130",
    )
    ap.add_argument("--flavor", default=FLAVOR)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--timeout", default="3h")
    ap.add_argument("--namespace", default="vllm-sr")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--omni",
        type=Path,
        help="a newer d25/omni tree to ship over this one (image runtime checks)",
    )
    ap.add_argument(
        "--overlay",
        default="",
        help="dir in the work dataset with package/ and SHA256SUMS: a new code revision of --revision's weights",
    )
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    data = snapshot(args.kit, args.omni)
    digest = hashlib.sha256(data).hexdigest()
    code = f"code/{digest}.tar.gz"
    env = {
        "WORK": args.work,
        "RUN": args.run,
        "MODEL": args.model,
        "REVISION": args.revision,
        "REFERENCE": args.reference,
        "LATENCY_SHA": LATENCY_SHA,
        "PARITY_SHA": PARITY_SHA,
        "VARIANTS": args.variants,
        "VARIANT_ROWS": str(args.variant_rows),
        "TORCH_SPEC": args.torch,
        "STEPS": args.steps,
        "IMAGE_CHECKS": args.image_checks,
        "OVERLAY": args.overlay,
        "PROBE_IDS": args.probe_ids,
        "PROBE_RUNTIME": args.probe_runtime,
        "TORCH_INDEX": args.torch_index,
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONUNBUFFERED": "1",
    }
    # curl + SHA-256 check: hf download of a just-uploaded file failed in rtx-pro-6000 jobs (no X-Repo-Commit).
    url = f"https://huggingface.co/datasets/{args.work}/resolve/main/{code}"
    fetch = f'curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o /tmp/code.tar.gz {url}'
    command = (
        "set -e; command -v curl > /dev/null || (apt-get update -qq && apt-get install -y -qq curl > /dev/null); "
        "pip install -q -U 'huggingface_hub>=1.0'; "
        f"for i in 1 2 3 4 5 6; do {fetch} && break; echo retry download; sleep 30; done; "
        f"echo '{digest}  /tmp/code.tar.gz' | sha256sum -c -; "
        "mkdir -p /tmp/src; tar xzf /tmp/code.tar.gz -C /tmp/src; bash /tmp/src/d25/vega/release/cuda_job.sh"
    )
    plan = {
        "flavor": args.flavor,
        "image": args.image,
        "timeout": args.timeout,
        "namespace": args.namespace,
        "code": code,
        "code_bytes": len(data),
        "env": env,
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return 0
    api = HfApi(token=token)
    api.create_repo(args.work, repo_type="dataset", private=True, exist_ok=True)
    if api.repo_info(args.work, repo_type="dataset").private is not True:
        raise SystemExit(
            f"{args.work} is not private; refusing to upload suite-derived results there"
        )
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=code,
        repo_id=args.work,
        repo_type="dataset",
        commit_message=f"code snapshot for {args.run}",
    )
    job = api.run_job(
        image=args.image,
        command=["bash", "-c", command],
        env=env,
        secrets={"HF_TOKEN": token},
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        labels={"d25-ws": "release", "run": args.run},
    )
    plan.update(job_id=getattr(job, "id", None), url=getattr(job, "url", None))
    print(
        json.dumps({k: plan[k] for k in ("job_id", "url", "flavor", "code")}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
