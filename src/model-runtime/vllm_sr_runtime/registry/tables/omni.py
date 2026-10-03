"""Vela 1.0 Omni prepared bundles (Phase 3): Nano and Mini.

Each entry pins the source revision a bundle must be prepared from. The Hub
repositories hold native source, so the runtime serves the bundle that
``tools/models/vela_omni`` exports from that revision (named after the
repository, for example ``vela-1.0-omni-nano``); the bundle's own inventory
and parity receipt are verified, and its identity is the digest of that
inventory, so ``model_sha256`` is not pinned. Golden references per device
class live in ``registry/golden_answers_omni.json``.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded

LICENCE = "apache-2.0"

MODELS: tuple[BuiltinModel, ...] = (
    BuiltinModel(
        repo_id=f"{ORG}/Vela-1.0-Omni-Nano",
        revision="2ff2d66385dbdd661a560ec3e8bcb45a0527d92e",
        family="multimodal_embedding",
        model_sha256="",
        manifest_sha256="",
        loaded_parameters=0,
        backbone="vela_omni",
        min_device_memory_gib=2,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Vela-1.0-Omni-Mini",
        revision="801bae3ad28df6891408f0e0441c676b30e132e3",
        family="multimodal_embedding",
        model_sha256="",
        manifest_sha256="",
        loaded_parameters=0,
        backbone="vela_omni",
        min_device_memory_gib=8,
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_omni.json", "answers", "golden_answers")
_BY_REPO = {model.repo_id.lower(): model for model in MODELS}


def lookup(repo_id: str) -> BuiltinModel | None:
    """The pin of an Omni repository (case-insensitive), or None."""
    return _BY_REPO.get(repo_id.strip().lower())


def bundle_name(model: BuiltinModel) -> str:
    """The prepared bundle's directory name: ``vela-1.0-omni-nano`` for ``vllm-sr/Vela-1.0-Omni-Nano``."""
    return model.repo_id.split("/", 1)[1].lower()


def variant(model: BuiltinModel) -> str:
    return model.repo_id.rsplit("-", 1)[1].lower()
