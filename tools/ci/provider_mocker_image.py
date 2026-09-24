"""Content-addressed publication and reuse of the independent provider fixture."""

from __future__ import annotations

import hashlib
import json
import re
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "provider-mocker"
CONTEXT = "tools/test/services/provider-mocker"
REPOSITORY = "vllm-project/semantic-router"
REGISTRY_PATH = f"{REPOSITORY}/{IMAGE}"
REGISTRY = f"ghcr.io/{REGISTRY_PATH}"
INPUT_LABEL = "io.vllm.semantic-router.fixture.inputs"
CHANNEL_LABEL = "io.vllm.semantic-router.fixture.channel"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
SHA = re.compile(r"[0-9a-f]{40}")


class PublicationUnavailableError(ValueError):
    """The requested input version has not been published yet."""


def is_build_input(path: str) -> bool:
    relative = path.removeprefix(CONTEXT + "/")
    if relative == path:
        return False
    return relative in {".dockerignore", "Dockerfile", "requirements.txt"} or (
        relative.startswith("provider_mocker/")
        and "__pycache__" not in Path(relative).parts
        and not relative.endswith((".pyc", ".pyo"))
    )


def input_fingerprint(root: Path = ROOT) -> str:
    context = root / CONTEXT
    files = [
        context / name for name in (".dockerignore", "Dockerfile", "requirements.txt")
    ]
    runtime_files = [
        path
        for path in (context / "provider_mocker").rglob("*")
        if path.is_file() and is_build_input(path.relative_to(root).as_posix())
    ]
    if not runtime_files:
        raise ValueError("provider-mocker runtime package is missing")
    files.extend(runtime_files)
    digest = hashlib.sha256(b"provider-mocker-build-inputs-v1\0")
    for path in sorted(files):
        if path.is_symlink():
            raise ValueError("provider-mocker build inputs cannot be symlinks")
        name = path.relative_to(context).as_posix().encode()
        content = path.read_bytes()
        executable = b"1" if path.stat().st_mode & 0o111 else b"0"
        for value in (name, executable, content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def input_tag(fingerprint: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid provider-mocker input fingerprint")
    return "inputs-" + fingerprint


def acquisition(paths: list[str], root: Path = ROOT) -> dict:
    fingerprint = input_fingerprint(root)
    return {
        "id": IMAGE,
        "source": "candidate" if any(map(is_build_input, paths)) else "published",
        "inputs_sha256": fingerprint,
        "lookup_ref": f"{REGISTRY}:{input_tag(fingerprint)}",
    }


def labels(mode: str, source_sha: str) -> dict[str, str]:
    return {
        "org.opencontainers.image.source": f"https://github.com/{REPOSITORY}",
        "org.opencontainers.image.revision": source_sha,
        INPUT_LABEL: input_fingerprint(),
        CHANNEL_LABEL: mode,
    }


class RegistryClient:
    """Read public GHCR manifests without giving PR code registry write access."""

    def __init__(self) -> None:
        query = urlencode(
            {"service": "ghcr.io", "scope": f"repository:{REGISTRY_PATH}:pull"}
        )
        self.token = json.loads(self._fetch(f"https://ghcr.io/token?{query}"))["token"]

    def _fetch(self, url: str, *, authenticated: bool = False) -> bytes:
        headers = {
            "Accept": ", ".join(
                (
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                )
            )
        }
        if authenticated:
            headers["Authorization"] = "Bearer " + self.token
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return response.read()

    def document(self, kind: str, reference: str) -> tuple[dict, str]:
        raw = self._fetch(
            f"https://ghcr.io/v2/{REGISTRY_PATH}/{kind}/{reference}", authenticated=True
        )
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if reference.startswith("sha256:") and digest != reference:
            raise ValueError("registry document digest differs from requested content")
        return json.loads(raw), digest


def resolve_published(record: dict, client: RegistryClient | None = None) -> dict:
    fingerprint = record["inputs_sha256"]
    tag = input_tag(fingerprint)
    client = client or RegistryClient()
    try:
        manifest, digest = client.document("manifests", tag)
    except HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            raise PublicationUnavailableError(
                f"Qualified provider-mocker {tag} has not been published"
            ) from error
        raise
    images = []
    revisions = set()
    for descriptor in manifest.get("manifests", [{"digest": digest}]):
        native, native_digest = client.document("manifests", descriptor["digest"])
        config, config_digest = client.document("blobs", native["config"]["digest"])
        image_labels = config.get("config", {}).get("Labels", {})
        revision = image_labels.get("org.opencontainers.image.revision", "")
        if (
            image_labels.get(INPUT_LABEL) != fingerprint
            or image_labels.get(CHANNEL_LABEL) != "main"
            or image_labels.get("org.opencontainers.image.source")
            != f"https://github.com/{REPOSITORY}"
            or not SHA.fullmatch(revision)
        ):
            raise ValueError(
                "provider-mocker publication identity differs from required inputs"
            )
        revisions.add(revision)
        images.append(
            {
                "platform": f"{config['os']}/{config['architecture']}",
                "manifest": native_digest,
                "config": config_digest,
            }
        )
    platforms = [image["platform"] for image in images]
    if (
        "linux/amd64" not in platforms
        or len(set(platforms)) != len(platforms)
        or len(revisions) != 1
    ):
        raise ValueError(
            "provider-mocker publication has invalid platforms or source identity"
        )
    return {
        **record,
        "ref": f"{REGISTRY}@{digest}",
        "registry_digest": digest,
        "image_source_sha": revisions.pop(),
        "images": sorted(images, key=lambda image: image["platform"]),
    }


def validate_acquisition(record: dict, expected: dict) -> None:
    for key in (
        "inputs_sha256",
        "registry_digest",
        "image_source_sha",
        "ref",
        "images",
    ):
        if record.get(key) != expected.get(key):
            raise ValueError(f"published provider-mocker {key} differs from plan")
    digest = record.get("registry_digest", "")
    if not DIGEST.fullmatch(digest) or record.get("ref") != f"{REGISTRY}@{digest}":
        raise ValueError("provider-mocker must use its qualified registry digest")
    if not SHA.fullmatch(record.get("image_source_sha", "")):
        raise ValueError("provider-mocker original source revision is unavailable")


def published_from_plan(plan: dict) -> dict | None:
    record = plan.get("image_sources", {}).get(IMAGE)
    return record if record and record["source"] == "published" else None
