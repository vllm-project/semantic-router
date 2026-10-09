"""Gate one of our checkpoints on the proxies: run it over the frozen proxy, score, estimate Full.

    python -m d25.vega.eval.proxy.run_proxy --ckpt /data/d25/vega/ckpt/<run>/<step> --gpus 7 \
        --out /data/d25/vega/evals/<run>/<step>/proxy [--public 61.9]

Runs ``d25.vega.eval.proxy.run`` with the shared code-readout engine (``ckpt:<dir>``, the checkpoint's
own prompt/attention/limits from ``decision_config.json``), one worker per GPU, resumable; then prints
S_proxy, S_proxy_clean, O_proxy, the O families and every per-benchmark / per-task skill, and, given
the local public index, the gate estimate (``gate.py``). The proxy files are read from ``--proxy``
(default: the frozen pv1 on the node; fetch with ``--fetch`` from the private dataset if absent).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_PROXY = "/data/d25/vega/proxy/final/pv1"
HF_REPO, HF_REVISION = (
    "vllm-sr/d25-vega-proxy",
    "f5a1c97d1217237035c569e4922911d64e9ad509",
)


def fetch(proxy: Path):
    from huggingface_hub import snapshot_download

    tmp = snapshot_download(
        HF_REPO, repo_type="dataset", revision=HF_REVISION, allow_patterns=["pv1/*"]
    )
    proxy.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-rL", str(Path(tmp) / "pv1"), str(proxy)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="code-readout checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=7)
    ap.add_argument("--proxy", default=DEFAULT_PROXY)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument(
        "--public",
        type=float,
        help="local public index (kit 0.3) for the gate estimate",
    )
    ap.add_argument(
        "--option",
        action="append",
        default=[],
        help="CodeReadoutModel options, e.g. readout_dtype=bfloat16",
    )
    a = ap.parse_args(argv)
    proxy, out = Path(a.proxy), Path(a.out)
    if not (proxy / "manifest.json").exists():
        if not a.fetch:
            sys.exit(f"{proxy} missing: pass --fetch to download it from {HF_REPO}")
        fetch(proxy)
    cmd = [
        sys.executable,
        "-m",
        "d25.vega.eval.proxy.run",
        "--anchor",
        f"ckpt:{a.ckpt}",
        "--rows",
        str(proxy),
        "--out",
        str(out),
        "--spawn",
        str(a.gpus),
    ]
    cmd += [x for kv in a.option for x in ("--option", kv)]
    code = subprocess.call(cmd)
    if code:
        sys.exit(
            f"runner failed ({code}); see {out}/worker*.log (resumable: rerun the same command)"
        )
    from d25.vega.eval.proxy.score import load_results, score_all

    sc = score_all(proxy, load_results([out]))
    (out / "scores.json").write_text(json.dumps(sc, indent=1))
    print(
        json.dumps(
            {
                k: sc[k]
                for k in (
                    "S_proxy",
                    "S_proxy_clean",
                    "O_proxy",
                    "O_families",
                    "coverage",
                )
            },
            indent=1,
        )
    )
    print(
        "per-benchmark S skills:",
        json.dumps({v["name"]: round(100 * v["skill"], 1) for v in sc["s"].values()}),
    )
    print(
        "per-task O skills:",
        json.dumps({t: round(100 * v["skill"], 1) for t, v in sc["o"].items()}),
    )
    if a.public is not None:
        from d25.vega.eval.proxy.gate import HERE, gate

        print(
            json.dumps(
                gate(
                    a.public,
                    sc["S_proxy"],
                    sc["O_proxy"],
                    HERE / "calibration_pv1.json",
                    sc.get("S_proxy_clean"),
                ),
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
