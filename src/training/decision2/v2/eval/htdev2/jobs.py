"""HT-DEV v2 collection jobs: one script per validation model, on its formal runtime.

    python3 -m v2.eval.htdev2.jobs --spec <spec.json> --src <mirror dir name> --ops <ops dir>

A model with an HT-DEV v1 job (`/data/dev2/runs/eval/htdev/ops/jobs/<key>.sh`, already checked
against its formal run) gets that job with the v2 mirror, run directory, autotune-cache copy,
lease and panels. Any other model gets a job rebuilt from its formal `COLLECT.json`: image,
adapter spec, model path and revision, extras, the mounts they need and a copy of its frozen
Triton autotune cache. Models without a stored node-A pilot readout also collect
`css-pilot` in the same job. Stdlib only (runs on the node host).
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

V1_JOBS = Path("/data/dev2/runs/eval/htdev/ops/jobs")
ROOT = Path("/data/dev2/runs/eval/htdev2")
DEFAULT_IMAGE_ID = (
    "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
)
LEASE = "eval-htdev2"


def panels_for(entry: dict) -> str:
    return "ht-dev2" if entry.get("pilot_predictions") else "ht-dev2,css-pilot"


def header(key: str) -> list[str]:
    """`<script> GPU [smoke]`: smoke runs 20 items per panel into separate directories."""
    return [
        f"KEY={key}; RUN_DIR={ROOT}/collect/$KEY; TRITON={ROOT}/triton/$KEY; SMOKE=()",
        'if [ "${2:-}" = smoke ]; then RUN_DIR='
        + f"{ROOT}/smoke/$KEY; TRITON={ROOT}/triton-smoke/$KEY; SMOKE=(--max-items 20); fi",
    ]


def from_v1(entry: dict, src: str) -> str:
    text = (V1_JOBS / f"{entry['key']}.sh").read_text(encoding="utf-8")
    key = entry["key"]
    lines = []
    for line in text.splitlines():
        if line.startswith("HTDEV_SRC="):
            line = f"HTDEV_SRC={src}"
        line = line.replace(f"/data/dev2/runs/eval/htdev/collect/{key}", "$RUN_DIR")
        line = line.replace(f"/data/dev2/runs/eval/htdev/triton/{key}", "$TRITON")
        line = line.replace("--shared-lease eval-htdev ", f"--shared-lease {LEASE} ")
        line = line.replace("HT-DEV v1 dev collection", "HT-DEV v2 dev collection")
        line = line.replace(
            "--panels ht-dev,css-pilot,typed-dev",
            f'--panels {panels_for(entry)} "${{SMOKE[@]}}"',
        )
        lines.append(line)
        if line == "GPU=$1":
            lines += header(key)
    out = "\n".join(lines) + "\n"
    if (
        f"--panels {panels_for(entry)}" not in out
        or f"HTDEV_SRC={src}" not in out
        or "/runs/eval/htdev/" in out
    ):
        raise ValueError(f"{key}: v1 job did not convert")
    return out


def path_mounts(value: str) -> list[str]:
    path = Path(value)
    if not path.is_absolute() or not path.exists():
        return []
    if str(path).startswith("/data/dev2/hf-cache/"):
        return ["/data/dev2/hf-cache"]
    return [str(path if path.is_dir() else path.parent)]


def from_formal(entry: dict, src: str, ops: Path) -> str:
    run = Path(entry["formal_run"])
    collect = json.loads((run / "COLLECT.json").read_text(encoding="utf-8"))
    adapter = collect["adapter"]
    spec_path = ops / "adapters" / f"{entry['key']}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(adapter, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    model = collect["model_path"]
    mounts: list[str] = []
    for value in collect.get("extra", {}).values():
        mounts += path_mounts(str(value))
    python = adapter.get("python") or ""
    if python.startswith("/data/dev2/tools/envs/"):
        mounts.append("/".join(python.split("/")[:6]))
    mounts += [str(m) for m in adapter.get("extra_mounts") or []]
    mounts = [
        m for m in dict.fromkeys(mounts) if not model.startswith(m + "/") and m != model
    ]
    env = collect.get("runtime_env") or {}
    triton = "$TRITON"
    q = shlex.quote
    lines = [
        "#!/usr/bin/env bash",
        "# usage: <script> GPU [smoke]",
        "set -euo pipefail",
        "GPU=$1",
        *header(entry["key"]),
        f"HTDEV_SRC={src}",
        "S=/data/dev2/src/$HTDEV_SRC/src/training/decision2",
    ]
    args = [
        "--gpu $GPU",
        "--track eval",
        f"--shared-lease {LEASE}",
        "--src $HTDEV_SRC",
        "--run-dir $RUN_DIR",
        f"--model-dir {q(model)}",
    ]
    if collect.get("image_id") and collect["image_id"] != DEFAULT_IMAGE_ID:
        args.append(f"--image {q(collect['image_id'])}")
    args += [f"--mount {q(m)}" for m in [*mounts, str(spec_path.parent)]]
    if env.get("HIP_FORCE_DEV_KERNARG"):
        args.append(f"--env HIP_FORCE_DEV_KERNARG={q(env['HIP_FORCE_DEV_KERNARG'])}")
    cache = env.get("TRITON_CACHE_DIR")
    if cache:
        lines.append(f"mkdir -p {triton}")
        if Path(cache).is_dir():
            lines.append(f"cp -a {q(cache)}/. {triton}/")
        args += [
            "--env TRITON_CACHE_AUTOTUNING=1",
            f"--env TRITON_CACHE_DIR={triton}",
            f"--mount-rw {triton}",
        ]
    args += [
        f'--purpose "HT-DEV v2 dev collection {entry["key"]}"',
        "--expected-end \"$(date -u -d '+30 minutes' +%FT%TZ)\"",
        "--",
        f"--adapter-spec {spec_path}",
        f"--model-path {q(model)}",
        f"--revision {q(collect['model_revision'])}",
    ]
    args += [
        f"--extra {q(f'{k}={v}')}" for k, v in sorted(collect.get("extra", {}).items())
    ]
    args.append(f'--panels {panels_for(entry)} "${{SMOKE[@]}}"')
    lines.append("bash $S/v2/eval/run_same_panel.sh \\\n  " + " \\\n  ".join(args))
    return "\n".join(lines) + "\n"


def cache_state(entry: dict, v1: bool) -> str:
    if v1:
        return "as the v1 job"
    collect = json.loads(
        (Path(entry["formal_run"]) / "COLLECT.json").read_text(encoding="utf-8")
    )
    cache = (collect.get("runtime_env") or {}).get("TRITON_CACHE_DIR")
    if not cache:
        return "none"
    if Path(cache).is_dir():
        return "copy of the formal cache"
    return "fresh (formal cache not on node A)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--ops", type=Path, default=ROOT / "ops")
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    (args.ops / "jobs").mkdir(parents=True, exist_ok=True)
    summary = []
    for entry in spec["models"]:
        v1 = entry.get("v1_collect") and (V1_JOBS / f"{entry['key']}.sh").is_file()
        text = (
            from_v1(entry, args.src) if v1 else from_formal(entry, args.src, args.ops)
        )
        path = args.ops / "jobs" / f"{entry['key']}.sh"
        path.write_text(text, encoding="utf-8")
        summary.append(
            {
                "key": entry["key"],
                "tier": entry["tier"],
                "from": "v1-job" if v1 else "formal-COLLECT",
                "panels": panels_for(entry),
                "autotune_cache": cache_state(entry, v1),
            }
        )
    (args.ops / "jobs.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "jobs": len(summary),
                "from_v1": sum(s["from"] == "v1-job" for s in summary),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
