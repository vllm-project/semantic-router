"""Immutable identities of the frozen 9B feature sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

SOURCES: dict[str, dict[str, Any]] = {
    "qwen35-9b-posttrained": {
        "repo": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "license": "apache-2.0",
        "loader": "qwen3_5_conditional_generation",
        "text_parameters": 7_936_684_544,
        "files": {
            "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
            "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5",
            "model.safetensors-00001-of-00004.safetensors": "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
            "model.safetensors-00002-of-00004.safetensors": "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
            "model.safetensors-00003-of-00004.safetensors": "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
            "model.safetensors-00004-of-00004.safetensors": "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
        },
    },
    "qwen35-9b-base": {
        "repo": "Qwen/Qwen3.5-9B-Base",
        "revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
        "license": "apache-2.0",
        "loader": "qwen3_5_conditional_generation",
        "text_parameters": 7_936_684_544,
        "files": {
            "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
            "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
            "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
            "model.safetensors.index.json": "026b9d9fe03f19fd065f2a2f56a332c67640878106c0ca6be2f60c655ed5a8c1",
            "model.safetensors-00001-of-00004.safetensors": "862bf7bba8a50145d19d0ae463931fae515284024736592a73a336bc4dfa54ee",
            "model.safetensors-00002-of-00004.safetensors": "bace8e115e11ca93c22f0352a60d2fb0c76ac6d7d1c2993c143b7ad2b6c8868c",
            "model.safetensors-00003-of-00004.safetensors": "63a021ac0011cbfc66166e77103327a8b45dee95832e36551f6b4c3337448959",
            "model.safetensors-00004-of-00004.safetensors": "1a643bbed669266917b5058b5d3f660c03233599249ff7d8fd083decfe662ae0",
        },
    },
    "lux1-9b": {
        "repo": "llm-semantic-router/Decision-1.0-Lux-9B",
        "revision": "bd45a30aee8c84032791c245c70f86dee5389cc8",
        "license": "apache-2.0",
        "loader": "qwen3_5_text_backbone_dir",
        "text_parameters": 7_936_684_544,
        "files": {
            "bundle-manifest.json": "985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd",
            "backbone/config.json": "4a87e4e7e11a11284a210066648b6d7616fda2872caeadec015aecb26dd9cffe",
            "backbone/model.safetensors.index.json": "f943816d8882f0acb572029805817240c2310d0d7e5b76fe1ececff63eab0686",
            "backbone/model-00001-of-00004.safetensors": "38b8c6b3cacebb8e5e24557fb29854efd0db217224ed0c5fb7482513de7436ca",
            "backbone/model-00002-of-00004.safetensors": "2b5686373f785b0043289bcd0d374bfdccd8b513754b66401ef54bd4c8ad2403",
            "backbone/model-00003-of-00004.safetensors": "ff152c4b46e7f2df8e257c7dc2818d569fd4ab0dc659494b26d65004f0f83d50",
            "backbone/model-00004-of-00004.safetensors": "99d7d3bf53cf1a3c5c355bdf7a139c7daa3ab47ce703980eaa8a27106590a8a6",
            "decision_head.safetensors": "f810788b27ee5fbeede232041d2b61022033d2b717d1d90070d2d903fca6137f",
            "decision_config.json": "656b1717b8cfebad6253e9d1321023323651ad3c80327562b7d1542ec74c23d1",
            "temperature.json": "ca7dfe0c3f28ab5d66804688ad779e2bc1f98f2e57a97669033a9d3bc4bf2e63",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
            "model-card-example.json": "12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18",
        },
    },
}

DATA: dict[str, str] = {
    "train": "fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "dev": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "css_pilot": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
}

GOLD: dict[str, str] = {
    "dev": "c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc",
    "css_pilot": "9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391",
}

SCORERS: dict[str, str] = {
    "benchmark/score.py": "d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc",
    "transfer/score.py": "cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca",
}

TRAINER_IMAGE = (
    "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
)
LAYERS = (16, 24, 32)


def file_sha256(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_source(name: str, root: str | Path) -> dict[str, str]:
    """Fail closed unless every pinned source file has its recorded digest."""
    if name not in SOURCES:
        raise ValueError(f"Unknown feature source {name}")
    root = Path(root)
    observed = {}
    for relative, expected in SOURCES[name]["files"].items():
        digest = file_sha256(root / relative)
        if digest != expected:
            raise ValueError(f"{name}: {relative} digest {digest} != pinned {expected}")
        observed[relative] = digest
    return observed


def verify_data(name: str, path: str | Path, table: dict[str, str] = DATA) -> str:
    digest = file_sha256(path)
    if table.get(name) != digest:
        raise ValueError(f"{name}: {path} digest {digest} is not the pinned partition")
    return digest
