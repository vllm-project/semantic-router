"""Vela 1.0 Omni (Phase 3): Nano and Mini, served from the published repositories.

Each entry pins the revision, the SHA-256 of every file the family reads
(``families/multimodal_embedding/package.FILES``: the weights, the towers'
configs, the tokenizer and the preprocessor configs; never the repository's
Python source), the identity those digests give and the parameter count the
repository declares. Golden references per device class live in
``registry/golden_answers_omni.json``.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded

LICENCE = "apache-2.0"

MODELS: tuple[BuiltinModel, ...] = (
    BuiltinModel(
        repo_id=f"{ORG}/Vela-1.0-Omni-Nano",
        revision="2ff2d66385dbdd661a560ec3e8bcb45a0527d92e",
        family="multimodal_embedding",
        model_sha256="cd445a707e6a7ade41e7ab2cf2f55768a148512b95b52ec09a9ef700c04a5e08",
        manifest_sha256="",
        loaded_parameters=163_771_288,
        backbone="vela_omni",
        min_device_memory_gib=2,
        files={
            "components/audio/config.json": "372c430053183035fb9d2d7079482f10053b6ff7d7367525ada9fdeead8adeac",
            "components/audio/preprocessor_config.json": "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
            "components/audio_clap/config.json": "9efb9557bc804f2ca6e394486af2e45dfed0b18554909735a99c6220b84e4288",
            "components/audio_clap/preprocessor_config.json": "9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753",
            "components/image/config.json": "c2b09e8c0a60f3405d3e53897358a8a373bc4286234f30fcc9324d065fc13728",
            "components/image/preprocessor_config.json": "5a0e5062d04603dc0e5eff01440f6bebfc66d396f9ca16e0128b31c5730cafc5",
            "components/text/config.json": "419ef27bf56a60adc670c50610f8099c64b23e4a1912f4ee7e321139518569b2",
            "components/text/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            "config.json": "ebd558ac77fa99b721b2f2c5d07e6d19c6aa1cc516c12df2900077ce937e78a3",
            "model.safetensors": "d5aa7f00e217bd4fd5314140a3603b9a873636320c8d09732a14f79f4cb0667e",
        },
        engines={"cpu": "native"},
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Vela-1.0-Omni-Mini",
        revision="801bae3ad28df6891408f0e0441c676b30e132e3",
        family="multimodal_embedding",
        model_sha256="213aed89f2058c223e7b650eb5762fc96890c992909d0a552cbc4bd4920905e6",
        manifest_sha256="",
        loaded_parameters=1_361_475_288,
        backbone="vela_omni",
        min_device_memory_gib=8,
        files={
            "components/audio/config.json": "f7d84d1a366f82beb8aef034bf014022d72e58f6f44fb25e5d92d80229353c09",
            "components/audio/preprocessor_config.json": "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
            "components/audio_clap/config.json": "9efb9557bc804f2ca6e394486af2e45dfed0b18554909735a99c6220b84e4288",
            "components/audio_clap/preprocessor_config.json": "9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753",
            "components/image/config.json": "73477b47ae9a395f008f993ebd8fb5c16ce0b5e8419ffdf2f5b2183b260f9cff",
            "components/image/preprocessor_config.json": "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90",
            "components/text/config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
            "components/text/tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
            "config.json": "914dfe932755d72a3392e25dcd93169775a43d8bd341d99add79695dba9ac509",
            "model.safetensors": "e8d8a5672405f7013138a411bf57bd402c59bafa9583fa91d1e15f70e0bf1117",
        },
        engines={"cpu": "native"},
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_omni.json", "answers", "golden_answers")
_BY_REPO = {model.repo_id.lower(): model for model in MODELS}


def lookup(repo_id: str) -> BuiltinModel | None:
    """The pin of an Omni repository (case-insensitive), or None."""
    return _BY_REPO.get(repo_id.strip().lower())


def variant(model: BuiltinModel) -> str:
    return model.repo_id.rsplit("-", 1)[1].lower()
