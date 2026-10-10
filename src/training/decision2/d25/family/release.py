"""Card inputs and Hugging Face steps of a d3-family release (``vllm-sr/d3-<size>``); run in a pod with HF_TOKEN set.

    python -m d25.family.release collection [--add vllm-sr/d3-flash ...] --out collection.json
    python -m d25.family.release measured --size flash --scores <kit scores.json> --result <ckpt_eval result.json> \
        --vgate vgate.json [--cuda-run <work dataset run>] --out measured.json
    python -m d25.family.release hub-new --repo vllm-sr/d3-flash --package <pkg> --out new.json
    python -m d25.family.release hub-final --repo vllm-sr/d3-flash --package <final pkg> --parent <sha> --out final.json
    python -m d25.family.release publish --repo vllm-sr/d3-flash --revision <sha> --out publish.json
    python -m d25.family.release dataset --stage <staged run> --model vllm-sr/d3-flash --revision <sha> --out ds.json

Public repositories get ONE clean commit: ``hub-new`` creates the model repository fresh and private,
``hub-final`` commits the final package on top and squashes the history to a single commit named after the
model, and ``publish`` tags that commit ``v3.0.0`` and makes the repository public (main and the tag must
resolve to it anonymously). ``dataset`` creates ``vllm-sr/d3-<size>-decision-index`` fresh with the staged
public run in one commit. ``measured`` writes the card inputs (``d25-measured/1``): the complete public kit run,
the paired text estimate recomputed with that run's public skills, the paired vision estimate with the
per-benchmark public scores, and the CUDA latency of the HF Job (placeholders with ``--provisional``).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

FAMILY_ROOT = Path("/data/d25/omni/family")
WORK = "vllm-sr/d25-vega-release-work"
KIT_COMMIT = "9eb2dbe"
TAG = "v3.0.0"
COLLECTION_TITLE = "Decision 3.0"
COLLECTION_DESCRIPTION = (
    "Open multimodal foundation decision models: text and images in, "
    "a probability for every answer out."
)
ORDER = (
    "vllm-sr/d3",
    "vllm-sr/d3-flash",
    "vllm-sr/d3-mini",
    "vllm-sr/d3-nano",
    "vllm-sr/d3-lite",
    "vllm-sr/d3-edge",
)
# Same-size Decision 2.0 reference of each paired text estimate and its board engine id.
REFERENCE = {
    "flash": ("lux2", "decision-2.0-lux-9b"),
    "mini": ("nox2", "decision-2.0-nox-4b"),
    "nano": ("sol2", "decision-2.0-sol-2b"),
    "lite": ("eos2", "decision-2.0-eos-0.8b"),
    "edge": ("kai2", "decision-2.0-kai-0.6b"),
}
VISION_BENCHMARKS = (
    "CV-Bench",
    "BLINK",
    "RealWorldQA",
    "CharXiv",
    "InfographicVQA",
    "Mind2Web",
    "Winoground",
    "KIE (CORD+FUNSD)",
    "Moderation (Hateful Memes)",
    "R-Bench-M",
    "MMMU-Pro vision",
)
APPROXIMATE = ("CharXiv", "InfographicVQA", "Mind2Web", "KIE (CORD+FUNSD)")
MAX_PIXELS = 1_638_400
# Files of a public results dataset must not carry internal paths, names or hosts.
TRACE = re.compile(
    r"/data/|decision25|(?<![0-9a-fA-F])d25(?![0-9a-fA-F])|(?<![\d.])2\.5(?![\d])|\bvega\b|\bomni\b|staging|"
    r"teacher|pplx|M2T|\b(?:flash|mini|nano|lite|edge)-t\d\b|step-00|[0-9]{1,3}(?:\.[0-9]{1,3}){3}|ml-ai-|mi325x8",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data, indent=1))


def files(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and ".cache" not in p.parts
    }


# ---------------------------------------------------------------- collection


def collection(add: list[str]) -> dict[str, Any]:
    """The vllm-sr "Decision 3.0" collection (created once), with the listed models in size order."""
    from huggingface_hub import HfApi

    api = HfApi()
    found = [
        c
        for c in api.list_collections(owner="vllm-sr", limit=100)
        if c.title == COLLECTION_TITLE
    ]
    if found:
        slug = found[0].slug
    else:
        slug = api.create_collection(
            COLLECTION_TITLE,
            namespace="vllm-sr",
            description=COLLECTION_DESCRIPTION,
            private=False,
        ).slug
    have = {i.item_id for i in api.get_collection(slug).items}
    for repo in sorted(set(add) - have, key=ORDER.index):
        api.add_collection_item(slug, item_id=repo, item_type="model", exists_ok=True)
    items = api.get_collection(slug).items
    wanted = sorted(
        items, key=lambda i: ORDER.index(i.item_id) if i.item_id in ORDER else 99
    )
    for position, item in enumerate(wanted):
        if item.position != position:
            api.update_collection_item(slug, item.item_object_id, position=position)
    final = api.get_collection(slug)
    return {
        "slug": slug,
        "url": f"https://huggingface.co/collections/{slug}",
        "title": final.title,
        "description": final.description,
        "private": final.private,
        "items": [i.item_id for i in sorted(final.items, key=lambda i: i.position)],
    }


# ---------------------------------------------------------------- card inputs


def public_skills(scores: dict) -> dict[str, Any]:
    return {
        "index": scores["decision_index"],
        "per_benchmark": {
            k: round(100 * v["index_skill"], 2)
            for k, v in scores["benchmarks"].items()
            if v.get("in_index")
        },
    }


def text_estimate(size: str, scores: dict, result: dict) -> dict[str, Any]:
    """The family gate's paired estimate with the complete run's public skills (O_proxy from the gate run)."""
    from d25.family import gate as family_gate
    from d25.vega.eval.proxy import gate as G

    name, _ = REFERENCE[size]
    cal = json.loads(Path(G.DEFAULT).read_text())
    o_ref = None
    anchor = FAMILY_ROOT / "anchors" / name / "scores.json"
    if not any(a["name"] == name for a in cal["anchor_points"]):
        o_ref = json.loads(anchor.read_text())["O_proxy"]
    public = public_skills(scores)
    ours = {
        "public": public["index"],
        "per_benchmark": public["per_benchmark"],
        "O_proxy": result["proxy"]["O_proxy"],
    }
    return family_gate.paired(ours, family_gate.reference(name, cal, o_ref), cal)


def measured(
    size: str,
    scores: dict,
    result: dict,
    vgate: dict,
    board: dict,
    latency: dict | None,
    latency_images: dict | None,
    sources: dict[str, str],
    provisional: bool = False,
) -> dict[str, Any]:
    """``provisional``: inputs of the private first upload only (any earlier public run, placeholder latency)."""
    complete = (
        scores["complete"] is True
        and scores["counts"] == {"ok": scores["completed"]}
        and scores["completed"] == 140_178
    )
    if not (complete or provisional) or (latency is None and not provisional):
        raise SystemExit(
            f"needs a complete public run and the CUDA latency: {scores['counts']}"
        )
    gate = text_estimate(size, scores, result)
    name, engine = REFERENCE[size]
    ref = next(m["v03"] for m in board["models"] if m["engine"] == engine)
    s_hat, o_hat = gate["S_hat"], gate["O_hat"]
    if provisional:
        latency = {"median_ms": 0.0, "mean_ms": 0.0, "p80_ms": 0.0, "timed_rows": 0}
        latency_images = {
            "scenarios": {
                k: {"median_ms": 0.0, "mean_ms": 0.0, "p80_ms": 0.0, "n": 0}
                for k in ("one_image", "four_images")
            }
        }
    scen = latency_images["scenarios"]
    return {
        "schema": "d25-measured/1",
        "provisional": provisional,
        "text": {
            "edition": "0.3",
            "board_edition": "0.3.1",
            "kit_commit": KIT_COMMIT,
            "public": round(scores["decision_index"], 2),
            "raw": round(scores["raw_index"], 2),
            "complete": True,
            "requests": 140_178 if provisional else scores["completed"],
            "ok": 140_178 if provisional else scores["counts"]["ok"],
            "unsupported": 0,
            "areas": {a["id"]: round(100 * a["skill"], 2) for a in scores["areas"]},
            "source": sources["scores"],
        },
        "vision": {
            "edition": "0.3.1",
            "rows": 12_210,
            "benchmarks": [
                {
                    "name": b,
                    "ours": round(vgate["benchmarks"][b]["public_ours"], 2),
                    "approximate": b in APPROXIMATE,
                }
                for b in VISION_BENCHMARKS
            ],
            "index": {"ours": round(vgate["estimate_public"], 2)},
            "source": sources["vgate"],
        },
        "latency": {
            "gpu": "NVIDIA RTX PRO 6000",
            "text": {
                "median_ms": latency["median_ms"],
                "mean_ms": latency["mean_ms"],
                "p80_ms": latency["p80_ms"],
                "timed": latency["timed_rows"],
            },
            **{
                k: {
                    "median_ms": scen[k]["median_ms"],
                    "mean_ms": scen[k]["mean_ms"],
                    "p80_ms": scen[k]["p80_ms"],
                    "timed": scen[k]["n"],
                }
                for k in ("one_image", "four_images")
            },
            "source": sources.get("latency", "provisional placeholder"),
        },
        "context": {"source": "not on the card"},
        "images": {"max_pixels": MAX_PIXELS},
        "internal": {
            "label": f"d3-{size}: internal evaluation",
            "text": {
                "full": round(gate["estimate"], 2),
                "same_skill": round(ref["same_skills"] + s_hat[0] - s_hat[1], 2),
                "new_domain": round(ref["new_domains"] + o_hat[0] - o_hat[1], 2),
                "paired_se": gate["paired_se"],
                "reference": name,
                "source": "d25.family.gate paired estimate, public skills of the complete run",
            },
            "vision": {
                "full": round(vgate["estimate_full"], 2),
                "public": round(vgate["estimate_public"], 2),
                "private": round(vgate["estimate_private"], 2),
                "se": round(vgate["se"], 2),
                "source": sources["vgate"],
            },
        },
    }


def work_json(run: str, name: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WORK, f"runs/{run}/{name}", repo_type="dataset")
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- model repository


def same_remote(entry, path: Path) -> bool:
    if getattr(entry, "lfs", None):
        return entry.lfs.sha256 == sha256(path)
    return entry.blob_id == git_blob(path)


def remote_tree(api, repo: str, revision: str) -> dict:
    return {
        e.path: e
        for e in api.list_repo_tree(repo, revision=revision, recursive=True)
        if hasattr(e, "blob_id")
    }


def hub_new(repo: str, package: Path, message: str) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.model_info(repo)
        # An interrupted first upload leaves the fresh private repo with only the Hub's initial commit.
        commits = api.list_repo_commits(repo)
        empty = set(api.list_repo_files(repo)) <= {".gitattributes"}
        if not (info.id == repo and info.private and len(commits) == 1 and empty):
            raise SystemExit(f"{repo} exists already (resolves to {info.id})")
    except RepositoryNotFoundError:
        api.create_repo(repo, private=True, exist_ok=False)
    info = api.model_info(repo)
    if info.id != repo or info.private is not True:
        raise SystemExit(
            f"{repo}: created repository resolves to {info.id}, private {info.private}"
        )
    api.upload_folder(repo_id=repo, folder_path=str(package), commit_message=message)
    info = api.model_info(repo)
    tree = remote_tree(api, repo, info.sha)
    local = files(package)
    bad = sorted(
        n for n, p in local.items() if n not in tree or not same_remote(tree[n], p)
    )
    return {
        "repo": repo,
        "revision": info.sha,
        "private": info.private,
        "files": len(local),
        "mismatched": bad,
        "ok": not bad and set(tree) == set(local) | {".gitattributes"},
    }


def hub_final(repo: str, package: Path, parent: str, message: str) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    info = api.model_info(repo)
    commits = api.list_repo_commits(repo)
    local = files(package)
    changed: list[str] = []
    stale: list[str] = []
    squashed = (
        len(commits) == 1
        and commits[0].title == message
        and commits[0].commit_id != parent
    )
    if not squashed:
        if info.sha != parent or info.private is not True:
            raise SystemExit(
                f"{repo}: main is {info.sha} (private {info.private}), expected private {parent}"
            )
        tree = remote_tree(api, repo, parent)
        changed = [
            n for n, p in local.items() if n not in tree or not same_remote(tree[n], p)
        ]
        stale = sorted(set(tree) - set(local) - {".gitattributes"})
        if changed or stale:
            api.create_commit(
                repo,
                operations=[CommitOperationAdd(n, str(local[n])) for n in changed]
                + [CommitOperationDelete(n) for n in stale],
                commit_message=message,
                parent_commit=parent,
            )
        api.super_squash_history(repo_id=repo, commit_message=message)
    # Right after a squash the Hub can report the old main for a while; wait until main is the single commit.
    for _ in range(30):
        info = api.model_info(repo)
        commits = api.list_repo_commits(repo)
        if len(commits) == 1 and commits[0].commit_id == info.sha:
            break
        time.sleep(10)
    tree = remote_tree(api, repo, info.sha)
    bad = sorted(
        n for n, p in local.items() if n not in tree or not same_remote(tree[n], p)
    )
    refs = api.list_repo_refs(repo)
    return {
        "repo": repo,
        "revision": info.sha,
        "changed": changed,
        "deleted": stale,
        "commits": [(c.commit_id, c.title) for c in commits],
        "branches": [b.name for b in refs.branches],
        "tags": [t.name for t in refs.tags],
        "files": len(local),
        "mismatched": bad,
        "ok": len(commits) == 1
        and commits[0].commit_id == info.sha
        and commits[0].title == message
        and not bad
        and set(tree) == set(local) | {".gitattributes"}
        and not refs.tags,
    }


def readback(repo: str, revision: str, package: Path) -> dict[str, Any]:
    """Fresh download of ``revision`` into a temporary directory; every file re-hashed against the package."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(repo, revision=revision)
    commits = api.list_repo_commits(repo)
    local = files(package)
    with tempfile.TemporaryDirectory(dir=str(package.parent)) as tmp:
        snapshot_download(repo, revision=revision, local_dir=tmp, max_workers=8)
        got = files(Path(tmp))
        got.pop(".gitattributes", None)
        bad = sorted(
            n for n in local if n not in got or sha256(got[n]) != sha256(local[n])
        )
        extra = sorted(set(got) - set(local))
    return {
        "repo": repo,
        "revision": info.sha,
        "private": info.private,
        "commits": len(commits),
        "files": len(local),
        "mismatched": bad,
        "extra": extra,
        "ok": info.sha == revision and len(commits) == 1 and not bad and not extra,
    }


def publish(repo: str, revision: str, tag: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo)
    commits = api.list_repo_commits(repo)
    if info.sha != revision or len(commits) != 1:
        raise SystemExit(
            f"{repo}: main {info.sha}, {len(commits)} commits; expected one commit {revision}"
        )
    if tag not in [t.name for t in api.list_repo_refs(repo).tags]:
        api.create_tag(
            repo, tag=tag, revision=revision, tag_message=f"{repo.split('/')[1]} {tag}"
        )
    api.update_repo_settings(repo_id=repo, private=False)
    anon = HfApi(token=False)
    for attempt in range(10):
        try:
            main, tagged = anon.model_info(repo), anon.model_info(repo, revision=tag)
            break
        except Exception:  # noqa: BLE001 - visibility can take a moment to propagate
            if attempt == 9:
                raise
            time.sleep(10)
    refs = api.list_repo_refs(repo)
    card = main.card_data.to_dict() if main.card_data else {}
    return {
        "repo": repo,
        "private": main.private,
        "main": main.sha,
        "tag": {tag: tagged.sha},
        "branches": [b.name for b in refs.branches],
        "commits": len(commits),
        "card": {k: card.get(k) for k in ("license", "base_model", "pipeline_tag")},
        "ok": main.private is False
        and main.sha == tagged.sha == revision
        and [b.name for b in refs.branches] == ["main"],
    }


# ---------------------------------------------------------------- results dataset


def dataset(stage: Path, model: str, revision: str, tag: str) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    name = model.split("/")[1]
    repo = f"vllm-sr/{name}-decision-index"
    title = f"{name}: Decision Index 0.3 public run"
    sums = {
        line.split()[1]: line.split()[0]
        for line in (stage / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    }
    names = sorted(p.name for p in stage.iterdir() if p.is_file())
    if set(sums) != set(names) - {"SHA256SUMS"} or any(
        sha256(stage / n) != h for n, h in sums.items()
    ):
        raise SystemExit("the staged run differs from its SHA256SUMS")
    scores = json.loads((stage / "scores.json").read_text())
    if (
        scores["complete"],
        scores["engine"],
        scores["completed"],
        scores["counts"],
    ) != (
        True,
        name,
        140_178,
        {"ok": 140_178},
    ):
        raise SystemExit(f"not a complete run of {name}: {scores['counts']}")
    if revision not in (stage / "RUN_NOTES.md").read_text():
        raise SystemExit("RUN_NOTES.md does not name the release revision")
    for n in names:
        opener = gzip.open if n.endswith(".gz") else open
        with opener(stage / n, "rt") as lines:
            hit = next((line for line in lines if TRACE.search(line)), None)
        if hit:
            raise SystemExit(f"trace in {n}: {TRACE.search(hit).group(0)!r}")
    readme = f"""# {name}: Decision Index 0.3 public run

Results of [{model}](https://huggingface.co/{model}) at tag `{tag}` = revision `{revision}` on the complete
Decision Index 0.3 public suite, run with the unmodified kit
[apolinario/decision-index](https://github.com/apolinario/decision-index) `{KIT_COMMIT}` (edition 0.3).

- Public index: **{scores['decision_index']}** (raw {scores['raw_index']}); {scores['completed']:,} scoreable requests,
  all answered, 0 unsupported; `scores.json` says complete: true.
- `runs/{name}/`: `scores.json`, `index.json`, `benchmark-summary.json`, `results.jsonl.gz` (the answers to every
  request), `environment.json`, `status.json`, `RUN_NOTES.md` (engine, settings, hardware) and `SHA256SUMS`.
- Re-scoring `results.jsonl.gz` with `decision_index score --edition 0.3` reproduces `scores.json`, `index.json`
  and `benchmark-summary.json` (apart from `generated_utc`).
"""
    if TRACE.search(readme):
        raise SystemExit("trace in the dataset README")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=False)
    if api.dataset_info(repo).id != repo:
        raise SystemExit(f"{repo} resolves to another repository")
    ops = [CommitOperationAdd("README.md", readme.encode())]
    ops += [CommitOperationAdd(f"runs/{name}/{n}", str(stage / n)) for n in names]
    api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=title)
    api.super_squash_history(repo_id=repo, repo_type="dataset", commit_message=title)
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=False)
    info = api.dataset_info(repo)
    commits = api.list_repo_commits(repo, repo_type="dataset")
    anon = HfApi(token=False)
    ainfo = anon.dataset_info(repo)
    remote = set(anon.list_repo_files(repo, repo_type="dataset", revision=info.sha))
    checked = {}
    with tempfile.TemporaryDirectory() as cache:
        for n in names:
            p = hf_hub_download(
                repo,
                f"runs/{name}/{n}",
                repo_type="dataset",
                revision=info.sha,
                token=False,
                cache_dir=cache,
            )
            checked[n] = sha256(Path(p)) == (sums.get(n) or sha256(stage / n))
        p = hf_hub_download(
            repo,
            "README.md",
            repo_type="dataset",
            revision=info.sha,
            token=False,
            cache_dir=cache,
        )
        checked["README.md"] = Path(p).read_bytes() == readme.encode()
    return {
        "dataset": repo,
        "commit": info.sha,
        "private": ainfo.private,
        "commits": [c.title for c in commits],
        "public_index": scores["decision_index"],
        "results": f"https://huggingface.co/datasets/{repo}/blob/{info.sha}/runs/{name}/scores.json",
        "anonymous_sha256_ok": checked,
        "ok": not ainfo.private
        and len(commits) == 1
        and remote
        == {".gitattributes", "README.md", *(f"runs/{name}/{n}" for n in names)}
        and all(checked.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collection")
    c.add_argument("--add", action="append", default=[])
    m = sub.add_parser("measured")
    m.add_argument("--size", required=True, choices=sorted(REFERENCE))
    m.add_argument("--scores", required=True, type=Path)
    m.add_argument("--result", required=True, type=Path)
    m.add_argument("--vgate", required=True, type=Path)
    m.add_argument("--board", required=True, type=Path)
    m.add_argument("--cuda-run", help="HF Job run name in the work dataset (latency)")
    m.add_argument("--provisional", action="store_true")
    n = sub.add_parser("hub-new")
    f = sub.add_parser("hub-final")
    f.add_argument("--parent", required=True)
    for p in (n, f):
        p.add_argument("--repo", required=True)
        p.add_argument("--package", required=True, type=Path)
    r = sub.add_parser("readback")
    r.add_argument("--repo", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--package", required=True, type=Path)
    p = sub.add_parser("publish")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--stage", required=True, type=Path)
    d.add_argument("--model", required=True)
    d.add_argument("--revision", required=True)
    for s in (c, m, n, f, r, p, d):
        s.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    read = lambda path: json.loads(Path(path).read_text())  # noqa: E731
    if a.cmd == "collection":
        report = collection(a.add)
    elif a.cmd == "measured":
        latency = images = None
        sources = {"scores": str(a.scores), "vgate": str(a.vgate)}
        if a.cuda_run:
            latency = work_json(a.cuda_run, "latency.json")
            images = work_json(a.cuda_run, "latency-images.json")
            sources["latency"] = f"HF Job run {a.cuda_run} (RTX PRO 6000)"
        report = measured(
            a.size,
            read(a.scores),
            read(a.result),
            read(a.vgate),
            read(a.board),
            latency,
            images,
            sources,
            a.provisional,
        )
    elif a.cmd == "hub-new":
        report = hub_new(a.repo, a.package, a.repo.split("/")[1])
    elif a.cmd == "hub-final":
        report = hub_final(a.repo, a.package, a.parent, a.repo.split("/")[1])
    elif a.cmd == "readback":
        report = readback(a.repo, a.revision, a.package)
    elif a.cmd == "publish":
        report = publish(a.repo, a.revision, TAG)
    else:
        report = dataset(a.stage, a.model, a.revision, TAG)
    write(a.out, report)
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
