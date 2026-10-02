"""Independent Hub state check of Decision-2.0-Lux-9B@259a4550 (read-only; prints no secrets)."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session

REPO = "llm-semantic-router/Decision-2.0-Lux-9B"
OLD = "llm-semantic-router/DEV2.0-9B"
NEW = "259a45502bfca2f585a59ef99079533106e0b136"
SUPERSEDED = "586af77916ee508320421bda6c22f7f0305a7279"
BUDGET = "5de3f9ed6e6f52308099a79a070ab7f311c66fc2"
FROZEN = Path(
    "/data/dev2/runs/release/dev2-9b-ka13ib-prerelease-20261001T191732Z/package/Decision-2.0-Lux-9B/MODEL_MANIFEST.json"
)
COLL = "llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00"
PURGED = {
    "1c964494281f878cc7a917c0a590af7c25d42c219fe6fe8174f0ef7afb010bd2",
    "264ebb5ff22724d7a38ecfdf8241b121fcd2a59cd00014ad67cd4c63ed36e9ec",
    "3275040901c5ae3e06bd335e885ebb93d105adae4b189fbcb351cdf727dc27c3",
    "36d5dc229e27493b1c78dd8f07cfe073a42980bc8fed6fd1fd717d7af41d6118",
    "5a867565c00fc94c1f64fd7126ee2623fc3ac7ea8c3519a0722534eb87aa8d0d",
    "5cb73b1e4fcd13cf63e9402407866817620146e925d00e1249f6bca6cde9d29d",
    "60a71a7aeae01cdba8b5b63ea2314781372eb7b4f9fd81bb5d792cdbe017b3f2",
    "7f9b096901ec874f044397a450e2d0249b1303b8d63dd8d8322aeca3b3a5e87a",
    "a3fb769f6a9785ec978fa91802ad01f926d552de6c75cdfd07bf863fbfe27335",
    "a58975a1c0000a2b96a2e72d302e864b295e31a313f64fff6cefa4aa843bc918",
}

api = HfApi()
session = get_session()


def get(url, follow):
    headers = {**build_hf_headers(), "Range": "bytes=0-1023"}
    try:
        r = session.get(url, headers=headers, follow_redirects=follow)
    except TypeError:
        r = session.get(url, headers=headers, allow_redirects=follow)
    loc = r.headers.get("location") or ""
    if "://" in loc:
        loc = "/" + loc.split("://", 1)[1].split("/", 1)[1]
    return r.status_code, loc.split("?", 1)[0]


def tree(rev):
    out = {}
    for e in api.list_repo_tree(REPO, revision=rev, recursive=True, expand=True):
        if hasattr(e, "blob_id"):
            out[e.path] = {
                "blob": e.blob_id,
                "lfs": e.lfs.sha256 if getattr(e, "lfs", None) else None,
                "size": e.size,
            }
    return out


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


res = {"huggingface_hub": huggingface_hub.__version__}
info = api.model_info(
    REPO, expand=["sha", "private", "lastModified", "usedStorage", "gated", "disabled"]
)
res["main"] = {
    "id": info.id,
    "sha": info.sha,
    "private": info.private,
    "gated": getattr(info, "gated", None),
    "disabled": getattr(info, "disabled", None),
    "last_modified": str(info.last_modified),
    "used_storage_bytes": getattr(info, "used_storage", None),
    "sha_is_259a4550": info.sha == NEW,
}
old = api.model_info(OLD, expand=["sha", "private"])
res["old_id"] = {
    "requested": OLD,
    "resolved_id": old.id,
    "sha": old.sha,
    "private": old.private,
}
res["old_id"]["api_no_follow"] = get(f"https://huggingface.co/api/models/{OLD}", False)
res["old_id"]["resolve_config_no_follow"] = get(
    hf_hub_url(OLD, "config.json", revision="main"), False
)
res["old_id"]["resolve_config_follow"] = get(
    hf_hub_url(OLD, "config.json", revision="main"), True
)[0]

refs = api.list_repo_refs(REPO)
commits = api.list_repo_commits(REPO)
res["refs"] = {
    "branches": {b.name: b.target_commit for b in refs.branches},
    "tags": {t.name: t.target_commit for t in refs.tags},
}
res["commits"] = {
    "count": len(commits),
    "latest": [
        {"id": c.commit_id[:12], "title": c.title, "created_at": str(c.created_at)}
        for c in commits[:3]
    ],
}

new_tree = tree(NEW)
res["tree_new"] = {
    "files": len(new_tree),
    "lfs_files": sum(1 for v in new_tree.values() if v["lfs"]),
}

with tempfile.TemporaryDirectory(prefix="d2-hubcheck-", dir="/data/dev2/tmp") as tmp:
    mpath = hf_hub_download(REPO, "MODEL_MANIFEST.json", revision=NEW, cache_dir=tmp)
    mbytes = Path(mpath).read_bytes()
    manifest = json.loads(mbytes)
    small = {}
    for path, meta in new_tree.items():
        if meta["lfs"] is None and path != "MODEL_MANIFEST.json":
            small[path] = Path(
                hf_hub_download(REPO, path, revision=NEW, cache_dir=tmp)
            ).read_bytes()
    qwen_new = small.get("decision2/qwen.py", b"")
    cfg = json.loads(small.get("config.json", b"{}"))
    old_qwen = {}
    for rev in (SUPERSEDED, BUDGET):
        old_qwen[rev[:8]] = hashlib.sha256(
            Path(
                hf_hub_download(REPO, "decision2/qwen.py", revision=rev, cache_dir=tmp)
            ).read_bytes()
        ).hexdigest()

res["manifest"] = {
    "sha256": hashlib.sha256(mbytes).hexdigest(),
    "is_01d642a1": hashlib.sha256(mbytes).hexdigest().startswith("01d642a1"),
    "files": len(manifest["files_sha256"]),
    "identity": manifest["identity"]["model_sha256"],
    "runtime_source": manifest["runtime"]["runtime_source"]["commit"],
    "vendor_source": manifest["runtime"]["vendor_source"]["commit"],
    "remote_code": manifest.get("remote_code"),
}
files = manifest["files_sha256"]
mism = []
for path, want in files.items():
    meta = new_tree.get(path)
    if meta is None:
        mism.append([path, "missing on Hub"])
    elif meta["lfs"] is not None:
        if meta["lfs"] != want:
            mism.append([path, "lfs sha256 differs"])
    elif (
        hashlib.sha256(small[path]).hexdigest() != want
        or git_blob(small[path]) != meta["blob"]
    ):
        mism.append([path, "content differs"])
extra = sorted(set(new_tree) - set(files) - {"MODEL_MANIFEST.json", ".gitattributes"})
res["hub_vs_released_manifest"] = {"mismatch": mism, "extra_on_hub": extra}

frozen_bytes = FROZEN.read_bytes()
frozen = json.loads(frozen_bytes)
ff = frozen["files_sha256"]
weights = sorted(p for p in ff if p.endswith(".safetensors"))
res["hub_vs_frozen_c1_package"] = {
    "frozen_manifest_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    "frozen_identity": frozen["identity"]["model_sha256"],
    "weight_files": len(weights),
    "weights_equal_on_hub": all(
        new_tree.get(p, {}).get("lfs") == ff[p] for p in weights
    ),
    "weight_bytes_on_hub": sum(new_tree[p]["size"] for p in weights if p in new_tree),
    "files_differing_from_frozen": sorted(
        p for p in set(ff) | set(files) if ff.get(p) != files.get(p)
    ),
    "identity_equal": frozen["identity"]["model_sha256"]
    == manifest["identity"]["model_sha256"],
}
fp = manifest["identity"]["fingerprint_files"]
canon = json.dumps(
    fp, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
)
res["identity_check"] = {
    "recomputed_from_fingerprint_files": hashlib.sha256(canon.encode()).hexdigest(),
    "fingerprint_files_equal_manifest": all(files.get(p) == h for p, h in fp.items()),
}

sup_tree = tree(SUPERSEDED)
runtime_paths = sorted(
    p
    for p in new_tree
    if p.startswith("decision2/")
    and not p.startswith("decision2/_vendor/")
    or p
    in (
        "configuration_decision2.py",
        "modeling_decision2.py",
        "pipeline_decision2.py",
        "config.json",
    )
)
res["runtime"] = {
    "auto_map": cfg.get("auto_map"),
    "custom_pipelines": sorted((cfg.get("custom_pipelines") or {}).keys()),
    "remote_code_files": [p for p in runtime_paths if p.endswith("_decision2.py")],
    "qwen_py_sha256": hashlib.sha256(qwen_new).hexdigest(),
    "qwen_py_has_forward_token_budget": b"def forward_token_budget" in qwen_new,
    "qwen_py_has_micro_batches": b"def micro_batches" in qwen_new,
    "qwen_py_equal_to": {
        k: v == hashlib.sha256(qwen_new).hexdigest() for k, v in old_qwen.items()
    },
    "runtime_and_remote_code_blobs_vs_586af779": {
        p: (
            "same"
            if sup_tree.get(p, {}).get("blob") == new_tree[p]["blob"]
            else "changed"
        )
        for p in runtime_paths
    },
}

lfs = list(api.list_lfs_files(REPO))
main_lfs = {v["lfs"] for v in new_tree.values() if v["lfs"]}
res["lfs"] = {
    "objects": len(lfs),
    "bytes": sum(f.size for f in lfs),
    "referenced_by_main": sum(1 for f in lfs if f.file_oid in main_lfs),
    "unreferenced": sorted(
        [
            [getattr(f, "filename", None), f.size, f.file_oid[:12]]
            for f in lfs
            if f.file_oid not in main_lfs
        ],
        key=lambda r: str(r[0]),
    ),
    "unreferenced_safetensors": sum(
        1
        for f in lfs
        if f.file_oid not in main_lfs
        and str(getattr(f, "filename", "")).endswith(".safetensors")
    ),
    "purged_targets_still_present": sorted(
        f.file_oid[:12] for f in lfs if f.file_oid in PURGED
    ),
}
res["superseded_weight_paths_served"] = {
    p: get(hf_hub_url(REPO, p, revision=SUPERSEDED), True)[0]
    for p, v in sorted(sup_tree.items())
    if p.endswith(".safetensors")
}
res["main_weight_paths_served"] = {
    p: get(hf_hub_url(REPO, p, revision=NEW), True)[0] for p in weights
}

coll = api.get_collection(COLL)
res["collection"] = {
    "title": coll.title,
    "private": coll.private,
    "items": [i.item_id for i in coll.items],
    "contains_repo": REPO in [i.item_id for i in coll.items],
}
json.dump(res, sys.stdout, indent=1, default=str)
print()
