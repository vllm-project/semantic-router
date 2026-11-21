"""A malformed safetensors header is a package error, not a retryable load failure."""

from __future__ import annotations

import hashlib
import json
import struct

from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.decision3 import package as pkg
from vllm_srun.plugins.base import PackageRef
from vllm_srun.runtime import Runtime
from vllm_srun.testing.decision3 import write_fixture


def test_malformed_safetensors_header_fails_without_retrying(tmp_path, monkeypatch) -> None:
    root = write_fixture(tmp_path / "package", "text", 3)
    name = "readout.safetensors"
    payload = b'{"__metadata__": '
    (root / name).write_bytes(struct.pack("<Q", len(payload)) + payload)

    # The package still verifies: pin the rewritten file and the identity its
    # hash contributes, so the failure comes from parsing the header itself.
    manifest_path = root / pkg.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest["files_sha256"][name] = digest
    if isinstance(manifest.get("files_bytes"), dict):
        manifest["files_bytes"][name] = (root / name).stat().st_size
    decision = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    identity = {
        file: manifest["files_sha256"][file]
        for file in (
            *[w for w in pkg.weight_files(root) if w.endswith(".safetensors")],
            *[n for n in pkg.IDENTITY_FILES if (root / n).is_file()],
        )
    }
    manifest["identity"]["model_sha256"] = pkg.model_identity(identity, decision)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolves = 0

    def resolve_fixture(model, **_options):
        nonlocal resolves
        resolves += 1
        assert model == "malformed"
        return PackageRef(root)

    monkeypatch.setattr("vllm_srun.runtime.resolve", resolve_fixture)
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model="malformed", device="cpu"),),
            load_attempts=3,
            load_retry_seconds=0.001,
        )
    )
    runtime.start(background=True)
    try:
        assert not runtime.wait(timeout=30)
        health = runtime.served[0].health
        assert health.state == "failed"
        assert health.reason is not None and health.reason.startswith("PackageError:"), health.reason
        assert name in health.reason
        assert resolves == 1, f"a permanent header failure was retried {resolves} times"
    finally:
        runtime.stop()
